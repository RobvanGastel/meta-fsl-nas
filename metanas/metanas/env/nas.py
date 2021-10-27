import math
import copy
import time
import igraph

import numpy as np

import gym
from gym import spaces

import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision import transforms
import torchvision.models as models

from metanas.meta_predictor.meta_predictor import MetaPredictor
import metanas.utils.genotypes as gt
from metanas.utils import utils


"""Wrapper for the RL agent to interact with the meta-model in the outer-loop
utilizing the OpenAI gym interface
"""


class NasEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, meta_model,
                 test_phase=False, cell_type="normal",
                 reward_estimation=False,
                 max_ep_len=100, test_env=None):
        super().__init__()
        self.config = config
        self.test_env = test_env
        self.cell_type = cell_type
        self.primitives = config.primitives
        self.n_ops = len(config.primitives)
        self.reward_estimation = reward_estimation

        self.test_phase = test_phase
        self.meta_model = meta_model

        # Task reward estimator
        if self.reward_estimation:
            self.meta_predictor = MetaPredictor(config)

            # remove last fully-connected layer
            model = models.resnet18(pretrained=True).eval()
            self.feature_extractor = torch.nn.Sequential(
                *list(model.children())[:-1])

        # The task is set in the meta-loop
        self.current_task = None
        self.max_ep_len = max_ep_len  # max_steps
        self.reward_range = (-0.1, 4)

        # Pick reward range,
        # self.reward_range = (-1, 1)

        # Initialize the step counter
        self.step_count = 0
        self.terminate_episode = False

        # Set baseline accuracy to scale the reward
        self.baseline_acc = 0

        # Initialize State / Observation space
        # Intermediate + input nodes
        self.n_nodes = self.config.nodes + 2

        # Adjacency matrix
        self.A = np.ones((self.n_nodes, self.n_nodes)) - np.eye(self.n_nodes)

        # Remove the 2 input nodes from A
        self.A[0, 1] = 0
        self.A[1, 0] = 0

        self.initialize_observation_space()

        # Initialize action space
        # |A| + 2*|O| + 1, the +1 for the termination
        action_size = len(self.A) + 2*len(self. primitives) + 1
        self.action_space = spaces.Discrete(action_size)

        # Store best alphas/or model obtained yet,
        self.max_alphas = []
        self.max_acc = 0.0

        # weights optimizer
        self.w_optim = torch.optim.Adam(
            self.meta_model.weights(),
            lr=self.config.w_lr,
            betas=(0.0, 0.999),
            weight_decay=self.config.w_weight_decay,
        )

    def reset(self):
        """Reset the environment state"""
        # Add clause for testing the environment in which the task
        # is not defined.
        assert not (self.current_task is None and self.test_env is False), \
            "A task needs to be set before evaluation"

        # Initialize the step counters
        self.step_count = 0
        self.terminate_episode = False

        # Set alphas and weights of the model
        self.meta_model.load_state_dict(self.meta_state)
        self.update_states()

        # Set starting edge for agent
        self.set_start_state()

        # Set baseline accuracy to scale the reward
        _, self.baseline_acc = self.compute_reward()

        return self.current_state

    def set_task(self, task, meta_state):
        """The meta-loop passes the task for the environment to solve"""
        print("Set new task for environment")
        self.current_task = task
        self.meta_state = copy.deepcopy(meta_state)

        self.reset()

        # Reset best alphas and accuracy for current trial
        self.max_acc = 0.0

        self.max_alphas = []
        if self.cell_type == "normal":
            for _, row in enumerate(self.meta_model.alpha_normal):
                self.max_alphas.append(row.to(self.config.device))
        elif self.cell_type == "reduce":
            for _, row in enumerate(self.meta_model.alpha_reduce):
                self.max_alphas.append(row.to(self.config.device))

        else:
            raise RuntimeError(f"Cell type {self.cell_type} is not supported.")

    def initialize_observation_space(self):
        # Generate the internal states of the graph
        self.update_states()

        # Set starting edge for agent
        self.set_start_state()

        self.observation_space = spaces.Box(
            0, 1,
            shape=self.current_state.shape,
            dtype=np.int32)

    def update_states(self):
        """Set all the state variables for the environment on
        reset and updates.

        Raises:
            RuntimeError: On passing invalid cell types
        """
        s_idx = 0
        self.states = []
        self.edge_to_index = {}
        self.edge_to_alpha = {}

        # Define (normalized) alphas
        if self.cell_type == "normal":
            # Idea of letting RL observe the normalized alphas,
            # and mutate the actual alpha values
            self.normalized_alphas = [
                F.softmax(alpha, dim=-1).detach().cpu()
                for alpha in self.meta_model.alpha_normal]

            self.alphas = [
                alpha.detach().cpu()
                for alpha in self.meta_model.alpha_normal]

        elif self.cell_type == "reduce":
            self.normalized_alphas = [
                F.softmax(alpha, dim=-1).detach().cpu()
                for alpha in self.meta_model.alpha_reduce]

            self.alphas = [
                alpha.detach().cpu()
                for alpha in self.meta_model.alpha_reduce]

        else:
            raise RuntimeError(f"Cell type {self.cell_type} is not supported.")

        for i, edges in enumerate(self.normalized_alphas):
            # edges: Tensor(n_edges, n_ops)
            edge_max, _ = torch.topk(edges[:, :], 1)
            # selecting the top-k input nodes, k=2
            _, topk_edge_indices = torch.topk(edge_max.view(-1), k=2)

            for j, edge in enumerate(edges):
                self.edge_to_index[(j, i+2)] = s_idx
                self.edge_to_index[(i+2, j)] = s_idx+1

                self.edge_to_alpha[(j, i+2)] = (i, j)
                self.edge_to_alpha[(i+2, j)] = (i, j)

                # For undirected edge we add the edge twice
                self.states.append(
                    np.concatenate((
                        [j],
                        [i+2],
                        [int(j in topk_edge_indices)],
                        self.A[i+2],
                        edge.detach().numpy())))

                self.states.append(
                    np.concatenate((
                        [i+2],
                        [j],
                        [int(j in topk_edge_indices)],
                        self.A[j],
                        edge.detach().numpy())))
                s_idx += 2

        self.states = np.array(self.states)

    def set_start_state(self):
        # TODO: Add probability to the starting edge?
        self.current_state_index = 0
        self.current_state = self.states[
            self.current_state_index]

    # Methods to increase alphas
    def _inverse_softmax(self, x, C):
        return torch.log(x) + C

    def increase_op(self, row_idx, edge_idx, op_idx, prob=0.3):
        C = math.log(10.)

        # Set short-hands
        curr_op = self.normalized_alphas[row_idx][edge_idx][op_idx]
        curr_edge = self.normalized_alphas[row_idx][edge_idx]

        # Allow for increasing to 0.99
        if curr_op + prob > 1.0:
            surplus = curr_op + prob - 0.99
            prob -= surplus

        if curr_op + prob < 1.0:
            # Increase chosen op
            with torch.no_grad():
                curr_op += prob

            # Prevent 0.00 normalized alpha values, resulting in
            # -inf
            with torch.no_grad():
                curr_edge += 0.01

            # Set the meta-model, update the env state in
            # self.update_states()
            if self.cell_type == "normal":
                with torch.no_grad():
                    self.meta_model.alpha_normal[
                        row_idx][edge_idx] = self._inverse_softmax(
                        curr_edge, C)
            elif self.cell_type == "reduce":
                with torch.no_grad():
                    self.meta_model.alpha_reduce[
                        row_idx][edge_idx] = self._inverse_softmax(
                        curr_edge, C)

            # True if state is mutated
            return True
        # False if no update occured
        return False

    def decrease_op(self, row_idx, edge_idx, op_idx, prob=0.3):
        C = math.log(10.)

        # Set short-hands
        curr_op = self.normalized_alphas[row_idx][edge_idx][op_idx]
        curr_edge = self.normalized_alphas[row_idx][edge_idx]

        # Allow for increasing to 0.99
        if curr_op - prob < 0.0:
            surplus = prob - curr_op + 0.01
            prob -= surplus

        if curr_op - prob > 0.0:
            # Increase chosen op
            with torch.no_grad():
                curr_op -= prob

            # Prevent 0.00 normalized alpha values, resulting in
            # -inf
            with torch.no_grad():
                curr_edge += 0.01

            if self.cell_type == "normal":
                with torch.no_grad():
                    self.meta_model.alpha_normal[
                        row_idx][edge_idx] = self._inverse_softmax(
                        curr_edge, C)
            elif self.cell_type == "reduce":
                with torch.no_grad():
                    self.meta_model.alpha_reduce[
                        row_idx][edge_idx] = self._inverse_softmax(
                        curr_edge, C)

            # True if state is mutated
            return True
        # False if no update occured
        return False

    def update_meta_model(self, increase, row_idx, edge_idx, op_idx):
        """Adjust alpha value of the meta-model for a given element
        and value

        Raises:
            RuntimeError: On passing invalid cell types
        """

        if self.cell_type == "normal":
            if increase:
                return self.increase_op(row_idx, edge_idx, op_idx)
            return self.decrease_op(row_idx, edge_idx, op_idx)

        elif self.cell_type == "reduce":
            if increase:
                return self.increase_op(row_idx, edge_idx, op_idx)
            return self.decrease_op(row_idx, edge_idx, op_idx)

        else:
            raise RuntimeError(f"Cell type {self.cell_type} is not supported.")

    def render(self, mode='human'):
        """Render the environment, according to the specified mode."""
        for row in self.states:
            print(row)

    def get_max_alphas(self):
        return self.max_alphas

    def step(self, action):
        start = time.time()

        # cur_node = int(self.current_state[0])
        # next_node = int(self.current_state[1])
        # row_idx, edge_idx = self.edge_to_alpha[(cur_node, next_node)]
        # norm_a1 = F.softmax(
        #     self.meta_model.alpha_normal[row_idx][edge_idx], dim=-1).detach().cpu()

        # Mutates the meta_model and the local state
        action_info, reward, acc = self._perform_action(action)

        if acc is not None and acc > 0.0:
            self.baseline_acc = acc

            if self.max_acc < acc:
                self.max_acc = acc

                if self.cell_type == "normal":
                    for _, row in enumerate(self.meta_model.alpha_normal):
                        self.max_alphas.append(row.to(self.config.device))
                elif self.cell_type == "reduce":
                    for _, row in enumerate(self.meta_model.alpha_reduce):
                        self.max_alphas.append(row.to(self.config.device))
                else:
                    raise RuntimeError(
                        f"Cell type {self.cell_type} is not supported.")

        # The final step time
        end = time.time()
        running_time = int(end - start)

        self.step_count += 1

        # Conditions to terminate the episode
        done = self.step_count == self.max_ep_len or \
            self.terminate_episode

        info_dict = {
            "step_count": self.step_count,
            "action_id": action,
            "action": action_info,
            "acc": acc,
            "running_time": running_time,
        }

        # norm_a2 = F.softmax(
        #     self.meta_model.alpha_normal[row_idx][edge_idx], dim=-1).detach().cpu()

        # if acc is not None:
        #     acc = round(acc, 2)
        # print(
        #     f"\nstep: {self.step_count}, action: {action}, {action_info}, rew: {reward:.2f}, acc: {acc}")
        # print(['%.2f' % elem for elem in list(norm_a1)])
        # print(['%.2f' % elem for elem in list(norm_a2)])

        return self.current_state, reward, done, info_dict

    def close(self):
        return NotImplemented

    def _perform_action(self, action):
        """Perform the action on both the meta-model and local state"""

        action_info = ""
        reward = 0.0
        acc = None

        # denotes the current edge it is on
        cur_node = int(self.current_state[0])
        next_node = int(self.current_state[1])

        # Adjacancy matrix A, navigating to the next node
        if action in np.arange(len(self.A)):

            # Determine if agent is allowed to traverse
            # the edge
            if self.A[next_node][action] > 0:
                # Legal action
                cur_node = next_node
                next_node = action

                s_idx = self.edge_to_index[(cur_node, next_node)]
                self.current_state = self.states[s_idx]

                action_info = f"Legal move from {cur_node} to {action}"

            elif self.A[next_node][action] < 1:
                # Illegal next_node is not connected the action node
                # return reward -1, and stay in the same edge
                reward = -0.1

                action_info = f"Illegal move from {cur_node} to {action}"

        # Increasing the alpha for the given operation
        if action in np.arange(len(self.A),
                               len(self.A)+len(self.primitives)):
            # Adjust action indices to fit the operations
            action = action - len(self.A)

            # Find the current edge to mutate
            row_idx, edge_idx = self.edge_to_alpha[(cur_node, next_node)]
            s_idx = self.edge_to_index[(cur_node, next_node)]

            # True = increase
            update = self.update_meta_model(True,
                                            row_idx,
                                            edge_idx,
                                            action)

            if update:
                # Update the local state after increasing the alphas
                self.update_states()

            # Set current state again!
            self.current_state = self.states[s_idx]

            # Compute reward after updating
            reward, acc = self.compute_reward()

            action_info = f"Increase alpha ({row_idx}, {edge_idx}, {action})"

        # Decreasing the alpha for the given operation
        if action in np.arange(len(self.A)+len(self.primitives),
                               len(self.A)+2*len(self.primitives)):
            # Adjust action indices to fit the operations
            action = action - len(self.A) - len(self.primitives)

            # Find the current edge to mutate
            row_idx, edge_idx = self.edge_to_alpha[(cur_node, next_node)]
            s_idx = self.edge_to_index[(cur_node, next_node)]

            # False = decrease
            update = self.update_meta_model(False,
                                            row_idx,
                                            edge_idx,
                                            action)

            if update:
                # Update the local state after increasing the alphas
                self.update_states()

            # Set current state again!
            self.current_state = self.states[s_idx]

            # Compute reward after updating
            reward, acc = self.compute_reward()

            action_info = f"Decrease alpha ({row_idx}, {edge_idx}, {action})"

        # Terminate the episode
        if action in np.arange(len(self.A)+2*len(self.primitives),
                               len(self.A)+2*len(self.primitives)+1,
                               ):
            self.terminate_episode = True
            action_info = f"Terminate the episode at step {self.step_count}"

        return action_info, reward, acc

    def compute_reward(self):
        # Calculation/Estimations of the reward
        # For testing env
        if self.test_env is not None:
            return np.random.uniform(low=-1, high=1, size=(1,))[0]

        if self.reward_estimation:
            acc = self._meta_predictor_estimation(self.current_task)
        else:
            acc = self._darts_estimation(self.current_task)

        # Scale reward to (-1, 1) range
        reward = self.scale_reward(acc)
        return reward, acc

    def scale_reward(self, accuracy):
        """
        Map the accuracy of the network to [-1, 1] for
        the environment.

        Mapping the accuracy in [s1, s2] to [b1, b2]

        for s in [s1, s2] to obtain the reward we compute
        reward = b1 + ((s-a1)*(b2-b1)) / (a2-a1)
        """
        # Map accuracies greater than the baseline to
        # [0, 1]
        reward = 0

        # print(self.baseline_acc, accuracy)

        # Else, the reward is 0
        if self.baseline_acc == accuracy:
            return 0.0

        if self.baseline_acc <= accuracy:
            a1, a2 = self.baseline_acc, 1.0
            b1, b2 = 0.0, 4.0

            reward = b1 + ((accuracy-a1)*(b2-b1)) / (a2-a1)
        # Map accuracies smaller than the baseline to
        # [-1, 0]
        elif self.baseline_acc >= accuracy:
            a1, a2 = 0.0, self.baseline_acc
            b1, b2 = -0.1, 0.0

            reward = b1 + ((accuracy-a1)*(b2-b1)) / (a2-a1)

        return reward

    def _darts_estimation(self, task):
        # First train the weights with few steps on current batch
        self.meta_model.train()

        for _, (train_X, train_y) in enumerate(task.train_loader):
            train_X, train_y = train_X.to(
                self.config.device), train_y.to(self.config.device)

            self.w_optim.zero_grad()
            logits = self.meta_model(train_X)

            loss = self.meta_model.criterion(logits, train_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.meta_model.weights(),
                                     self.config.w_grad_clip)
            self.w_optim.step()

        for batch_idx, batch in enumerate(task.test_loader):
            x_test, y_test = batch
            x_test = x_test.to(self.config.device, non_blocking=True)
            y_test = y_test.to(self.config.device, non_blocking=True)

            logits = self.meta_model(
                x_test, sparsify_input_alphas=True
            )

            prec1, _ = utils.accuracy(logits, y_test, topk=(1, 5))

        reward = prec1.item()
        return reward

    def _meta_predictor_estimation(self, task):

        graph = self._meta_predictor_graph_preprocess()
        dataset = self._meta_predictor_dataset_preprocess(task)

        # Evaluate on the MetaD2A predictor
        y_pred = self.meta_predictor.evaluate_architecture(
            dataset, graph
        )

        y_pred = y_pred.item()
        print(y_pred)

        # x_min, x_max = -1, 1
        # acc = (y_pred - x_min)/(x_max - x_min)

        return y_pred

    def _meta_predictor_graph_preprocess(self):
        # TODO: Use genotype function from meta_model possibly
        geno = parse(self.normalized_alphas, k=2,
                     primitives=gt.PRIMITIVES_NAS_BENCH_201)

        # Convert genotype to graph
        edges = []

        # All networks have transformed edge => node,
        # node => edge the adjacency matrix is,
        connections = [[1],
                       [1, 0],
                       [0, 1, 0],
                       [1, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 1, 1, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1]]

        start_node = [0]
        edges.append(start_node)
        index = 0

        for node in geno:
            for op, _ in node:
                # plus two, to not confuse the
                # start node and end node
                op = [self.primitives.index(op) + 2]
                op.extend(connections[index])
                edges.append(op)
                index += 1

        stop_node = [1]
        stop_node.extend(connections[-1])
        edges.append(stop_node)

        graph, _ = decode_metad2a_to_igraph(edges)
        return graph

    def _meta_predictor_dataset_preprocess(self, task):

        # Get num_samples, n_train * k
        # Testing dataset does not have enough samples
        train_y, _ = next(iter(task.train_loader))
        assert train_y.shape[0] == self.config.num_samples, \
            "Number of samples should equal training of meta_predictor" \
            f"{train_y.shape[0]}, {self.config.num_samples}"

        # Shape the image as (3, 32, 32)
        dataset = F.interpolate(train_y, size=(32, 32))

        # Make sure there are 3 channels
        if self.config.dataset == "omniglot" or \
                self.config.dataset == "triplemnist":
            dataset = dataset.repeat(1, 3, 1, 1)

        # Normalize the features
        # TODO: Configurable
        if self.config.dataset == "omniglot":
            mean = torch.tensor([0.9221])
            std = torch.tensor([0.1257])

        elif self.config.dataset == "triplemnist":
            mean = torch.tensor([0.1307])
            std = torch.tensor([0.3081])

        elif self.config.dataset == "miniimagenet":
            mean = torch.tensor([0.1307])
            std = torch.tensor([0.3081])
        else:
            raise RuntimeError(
                f"Dataset {self.config.dataset} is not supported.")
        dataset = dataset.sub_(mean).div_(std)

        # Extracts features by ResNet18
        return self.feature_extractor(dataset)


def decode_metad2a_to_igraph(row):
    if isinstance(row, str):
        row = eval(row)
    n = len(row)

    g = igraph.Graph(directed=True)
    g.add_vertices(n)

    for i, node in enumerate(row):
        g.vs[i]['type'] = node[0]

        if i < (n - 2) and i > 0:
            g.add_edge(i, i + 1)  # always connect from last node
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                g.add_edge(j, i)
    return g, n


def parse(alpha, k, primitives=gt.PRIMITIVES_NAS_BENCH_201):
    gene = []
    for edges in alpha:
        edge_max, primitive_indices = torch.topk(
            edges[:, :], 1
        )

        topk_edge_values, topk_edge_indices = torch.topk(
            edge_max.view(-1), k)

        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = primitives[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)
    return gene
