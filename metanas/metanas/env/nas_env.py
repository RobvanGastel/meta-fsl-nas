import math
import copy
import time
import igraph

from collections import OrderedDict, namedtuple
import numpy as np

import gym
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from metanas.task_optimizer.darts import CellArchitect
from metanas.meta_predictor.meta_predictor import MetaPredictor
import metanas.utils.genotypes as gt
from metanas.utils import utils


"""Wrapper for the RL agent to interact with the meta-model in the outer-loop
utilizing the OpenAI gym interface
"""


class NasEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, meta_model, test_phase=False,
                 cell_type="normal", reward_estimation=False,
                 max_ep_len=200, disable_pairwise_alphas=False,
                 test_env=None):
        super().__init__()
        self.config = config
        self.test_env = test_env
        self.cell_type = cell_type
        self.primitives = config.primitives
        self.n_ops = len(config.primitives)

        self.disable_pairwise_alphas = disable_pairwise_alphas

        self.test_phase = test_phase
        self.meta_model = meta_model

        # Task
        self.current_task = None
        self.states = []
        self.discrete_alphas = []

        # Store reward previous estimation
        self.baseline_acc = 0.0

        # Store best alphas/or model obtained yet,
        self.max_alphas = []
        self.max_acc = 0.0

        # Task acuracy estimator
        self.reward_estimation = reward_estimation
        self.max_task_train_steps = config.darts_estimation_steps
        if self.reward_estimation:
            self.meta_predictor = MetaPredictor(config)

            # remove last fully-connected layer
            model = models.resnet18(pretrained=True).eval().to(config.device)
            self.feature_extractor = torch.nn.Sequential(
                *list(model.children())[:-1])
        else:
            # DARTS estimation of the network is used
            self.task_train_steps = 0

            if cell_type == "normal":
                self.w_optim = torch.optim.Adam(
                    self.meta_model.reduce_weights(),
                    lr=self.config.w_lr,
                    betas=(0.0, 0.999),
                    weight_decay=self.config.w_weight_decay,
                )

                self.a_optim = torch.optim.Adam(
                    self.meta_model.reduce_alphas(),
                    self.config.alpha_lr,
                    betas=(0.0, 0.999),
                    weight_decay=self.config.alpha_weight_decay,
                )
            else:
                self.w_optim = torch.optim.Adam(
                    self.meta_model.reduce_weights(),
                    lr=self.config.w_lr,
                    betas=(0.0, 0.999),
                    weight_decay=self.config.w_weight_decay,
                )

                self.a_optim = torch.optim.Adam(
                    self.meta_model.reduce_alphas(),
                    self.config.alpha_lr,
                    betas=(0.0, 0.999),
                    weight_decay=self.config.alpha_weight_decay,
                )

            self.architect = CellArchitect(
                self.meta_model,
                self.config.w_momentum,
                self.config.w_weight_decay,
                self.config.use_first_order_darts,
                cell_type
            )

        # Episode step counter
        self.step_count = 0

        # DARTS Cell graph
        # Intermediate + input nodes
        self.n_nodes = self.config.nodes + 2

        # Adjacency matrix
        self.A = np.ones((self.n_nodes, self.n_nodes)) - np.eye(self.n_nodes)

        # Remove the 2 input nodes from A
        self.A[0, 1] = 0
        self.A[1, 0] = 0

        # A's upper triangle
        self.A_up = np.triu(self.A)

        # Initialize action space
        # |A| + 2*|O| + 1, +1 if termination action
        action_size = len(self.A) + 2*len(self. primitives)
        self.action_space = spaces.Discrete(action_size)

        # Tracking statistics
        self.init_tracking_vars()

        # Environment/Gym.Env variables
        self.do_update = False
        self.max_ep_len = max_ep_len  # max_steps per episode

        # Reward range
        self.encourage_exploration = config.encourage_exploration
        self.encourage_increase = 2.0
        self.encourage_decrease = 0.0

        self.min_rew, self.max_rew = config.min_rew, config.max_rew

        self.reward_range = (self.min_rew, self.max_rew)
        if self.encourage_exploration:
            self.reward_range = (self.min_rew, self.max_rew * 2.0)

        # Initialize State / Observation space
        self.initialize_observation_space()

        # Settings specific for unit testing
        if test_env:
            self.meta_state = copy.deepcopy(meta_model.state_dict())

    def init_tracking_vars(self):
        """Track environment stastics for logging
        """
        self.encourage_edges = {(i, j): 0
                                for i in range(self.A_up.shape[0])
                                for j in range(self.A_up.shape[1])
                                if self.A_up[i, j] == 1}

        self.unique_edges = {(i, j): 0
                             for i in range(self.A_up.shape[0])
                             for j in range(self.A_up.shape[1])
                             if self.A_up[i, j] == 1}

        self.path_graph = []

        # Moving average of the number of alpha adjust before traversing
        self.n_a_adj = 0
        self.avg_a_adj = 0
        self.a_adj_trav = 0

        self.alpha_adjustments = 0
        self.acc_estimations = 0
        self.edge_traversals = 0
        self.illegal_edge_traversals = 0

    def reset(self):
        """Reset the environment state"""
        # Add clause for testing the environment in which the task
        # is not defined.
        assert not (self.current_task is None and self.test_env is False), \
            "A task needs to be set before evaluation"

        # Initialize the step counters
        self.step_count = 0

        # Reset alphas and weights of the model
        self.meta_model.load_state_dict(self.meta_state)
        self.update_states()

        if not self.reward_estimation:
            self._init_darts_training()

        # Reset tracking statistics
        self.init_tracking_vars()

        # Set starting edge for agent
        self.set_start_state()

        # Set baseline accuracy to scale the reward
        _, self.baseline_acc = self.compute_reward()

        # Invalid action mask
        mask = self.invalid_mask[self.current_state_index]
        return self.current_state, mask

    def set_task(self, task, meta_state, test_phase=False):
        """The meta-loop passes the task for the environment to solve"""

        self.current_task = task
        self.meta_state = copy.deepcopy(meta_state)

        self.reset()

        # Test phase with adjusted DARTS training
        self.test_phase = test_phase

        # Reset best alphas and accuracy for current trial
        self.max_acc = 0.0

        self.max_meta_model = copy.deepcopy(meta_state)

        self.max_alphas = nn.ParameterList()
        if self.cell_type == "normal":
            for _, row in enumerate(self.meta_model.alpha_normal):
                self.max_alphas.append(
                    nn.Parameter(row.to(self.config.device)))
        elif self.cell_type == "reduce":
            for _, row in enumerate(self.meta_model.alpha_reduce):
                self.max_alphas.append(
                    nn.Parameter(row.to(self.config.device)))
        else:
            raise RuntimeError(f"Cell type {self.cell_type} is not supported.")

    def initialize_observation_space(self):
        # Generate the internal states of the graph
        self.update_states()

        # Set starting edge for agent
        self.set_start_state()

        self.observation_space = spaces.Box(
            0, self.n_nodes,
            shape=self.current_state.shape,
            dtype=np.int32)

    def update_states(self):
        """Set all the state variables for the environment on
        reset and updates.

        Raises:
            RuntimeError: On passing invalid cell types
        """
        s_idx = 0

        prev_alphas = copy.deepcopy(self.discrete_alphas)
        prev_states = copy.deepcopy(self.states)

        self.discrete_alphas = []
        self.states = []
        self.invalid_mask = []
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
            edge_max, edge_idx = torch.topk(edges[:, :], 1)

            # selecting the top-k input nodes, k=2
            _, topk_edge_indices = torch.topk(edge_max.view(-1), k=2)

            # one-hot edges: Tensor(n_edges, n_ops)
            edge_one_hot = torch.zeros_like(edges[:, :])
            for hot_e, op in zip(edge_one_hot, edge_idx):
                hot_e[op.item()] = 1

            for j, edge in enumerate(edge_one_hot):
                self.edge_to_index[(j, i+2)] = s_idx
                self.edge_to_index[(i+2, j)] = s_idx+1

                self.edge_to_alpha[(j, i+2)] = (i, j)
                self.edge_to_alpha[(i+2, j)] = (i, j)

                # Store to check if edge has changed
                self.discrete_alphas.append(edge.detach().numpy())
                self.discrete_alphas.append(edge.detach().numpy())

                # For undirected edge we add the edge twice
                self.states.append(
                    np.concatenate((
                        [j],
                        [i+2],
                        [int(j in topk_edge_indices)],
                        self.A[i+2],
                        edge.detach().numpy())))

                self.invalid_mask.append(
                    np.hstack((self.A[i+2], np.ones((self.n_ops*2)))))

                self.states.append(
                    np.concatenate((
                        [i+2],
                        [j],
                        [int(j in topk_edge_indices)],
                        self.A[j],
                        edge.detach().numpy())))

                self.invalid_mask.append(
                    np.hstack((self.A[j], np.ones((self.n_ops*2)))))

                s_idx += 2

        self.states = np.array(self.states)
        self.invalid_mask = np.array(self.invalid_mask)
        self.discrete_alphas = np.array(self.discrete_alphas)

        return {
            'prev_states': prev_states,
            'prev_alphas': prev_alphas
        }

    def set_start_state(self):
        if self.config.use_env_random_start:
            # Random starting point
            idx = np.random.choice(range(len(self.encourage_edges)))
            cur_node, next_node = list(self.encourage_edges.keys())[idx]

            s_idx = self.edge_to_index[(cur_node, next_node)]
            self.current_state_index = s_idx
            self.current_state = self.states[s_idx]
        else:
            # Fixed starting point
            self.current_state_index = 0
            self.current_state = self.states[
                self.current_state_index]

    def _inverse_softmax(self, x, C):
        """Reverse calculation of the normalized alpha
        """
        return torch.log(x) + C

    def increase_op(self, row_idx, edge_idx, op_idx, prob=0.6):
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

            # Prevent 0.00 normalized alpha values resulting in -inf
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

    def decrease_op(self, row_idx, edge_idx, op_idx, prob=0.6):
        C = math.log(10.)

        # Set short-hands
        curr_op = self.normalized_alphas[row_idx][edge_idx][op_idx]
        curr_edge = self.normalized_alphas[row_idx][edge_idx]

        # Allow for increasing to 0.99
        if curr_op - prob < 0.0:
            surplus = prob - curr_op + 0.01
            prob -= surplus

        if curr_op - prob > 0.0:
            # Decrease chosen op
            with torch.no_grad():
                curr_op -= prob

            # Prevent 0.00 normalized alpha values resulting in -inf
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

    def get_max_meta_model(self):
        return self.max_meta_model

    def step(self, action):
        start = time.time()

        # Mutates the meta_model and the local state
        action_info, reward, acc = self._perform_action(action)

        if acc is not None and acc > 0.0:
            self.baseline_acc = acc

            if self.max_acc < acc:
                self.max_acc = acc

                # Max alphas and weights
                self.max_meta_model = copy.deepcopy(
                    self.meta_model.state_dict())

                # Max alphas
                self.max_alphas = nn.ParameterList()
                if self.cell_type == "normal":
                    for _, row in enumerate(self.meta_model.alpha_normal):
                        self.max_alphas.append(
                            nn.Parameter(row.to(self.config.device)))
                elif self.cell_type == "reduce":
                    for _, row in enumerate(self.meta_model.alpha_reduce):
                        self.max_alphas.append(
                            nn.Parameter(row.to(self.config.device)))
                else:
                    raise RuntimeError(
                        f"Cell type {self.cell_type} is not supported.")

        # Conditions to terminate the episode
        done = self.step_count == self.max_ep_len-1 or \
            self.acc_estimations == self.max_task_train_steps-1

        # Invalid action mask
        mask = self.invalid_mask[self.current_state_index]
        self.step_count += 1

        info_dict = {
            "steps": self.step_count,
            "mask": mask,
            "action_id": action,
            "action": action_info,
            "acc": acc,
            "running_time": time.time() - start,
            "illegal_edge_traversals": self.illegal_edge_traversals,
        }

        # Final episode statistics
        if done:
            print(self.step_count, self.acc_estimations)
            info_dict['path_graph'] = self.path_graph
            info_dict["acc_estimations"] = self.acc_estimations
            info_dict['alpha_adjustments'] = self.alpha_adjustments
            info_dict['edge_traversals'] = self.edge_traversals
            info_dict['alpha_adj_before_trav'] = self.avg_a_adj
            info_dict['unique_edges'] = number_of_unique_visits(
                self.unique_edges)

        return self.current_state, reward, done, info_dict

    def close(self):
        pass

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

            # Determine if agent is allowed to traverse the edge
            if self.A[next_node][action] > 0:
                # Legal action
                cur_node = next_node
                next_node = action

                s_idx = self.edge_to_index[(cur_node, next_node)]
                self.current_state_index = s_idx
                self.current_state = self.states[s_idx]

                action_info = f"Legal move from {cur_node} to {action}"

                # Increase unique edge tracking
                increase_edge(self.unique_edges,
                              cur_node, next_node)

                # Compute reward after updating
                if self.do_update:
                    self.acc_estimations += 1
                    reward, acc = self.compute_reward()
                    self.do_update = False

                    if check_if_visited(self.encourage_edges,
                                        cur_node, next_node):

                        # Increase the edge visists, (a, b) = (b,a)
                        increase_edge(self.encourage_edges,
                                      cur_node, next_node)

                        if self.encourage_exploration:
                            if reward > 0.0:
                                reward = reward * self.encourage_increase
                    else:
                        # Increase the edge visists, (a, b) = (b,a)
                        increase_edge(self.encourage_edges,
                                      cur_node, next_node)

                        if self.encourage_exploration:
                            if reward > 0.0:
                                # Decrease reward
                                reward = reward * \
                                    (self.encourage_decrease)

                # States might change due to DARTS reward estimation
                self.update_states()

                # Action statistics
                self.edge_traversals += 1

                # Update running average
                self.n_a_adj += 1
                avg = self.avg_a_adj
                n = self.n_a_adj

                self.avg_a_adj = ((n-1) * avg + self.a_adj_trav)/n

                self.a_adj_trav = 0
                self.path_graph.append((cur_node, next_node))

            elif self.A[next_node][action] < 1:

                # Action statistics
                self.illegal_edge_traversals += 1
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
                prev_states = self.update_states()

                # Only "Calculate reward/do_update" for reward if
                # in top-k
                self.do_update = edge_become_topk(
                    prev_states, self.states, self.discrete_alphas, s_idx)

            # Set current state again!
            self.current_state = self.states[s_idx]

            # Action statistics
            self.alpha_adjustments += 1
            self.a_adj_trav += 1

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
                prev_states = self.update_states()

                # Only "Calculate reward/do_update" for reward if
                # in top-k or if the topk edge changed.
                self.do_update = edge_become_topk(
                    prev_states, self.states, self.discrete_alphas, s_idx)

            # Set current state again!
            self.current_state = self.states[s_idx]

            # Action statistics
            self.alpha_adjustments += 1
            self.a_adj_trav += 1

            action_info = f"Decrease alpha ({row_idx}, {edge_idx}, {action})"

        return action_info, reward, acc

    def compute_reward(self):
        """Calculation or estimations of the reward"""
        # Dummy acc and reward for testing purposes
        if self.test_env is not None:
            acc = np.random.uniform(low=0, high=1, size=(1,))[0]
            reward = self.scale_reward(acc)
            return reward, acc

        start = time.time()
        if self.reward_estimation:
            acc = self._meta_predictor_estimation(self.current_task)
        else:
            if self.config.update_weights_and_alphas and \
                    not self.config.use_tse_darts:
                acc = self._darts_weight_alpha_estimation(self.current_task)
            elif self.config.use_tse_darts:
                acc = self._tse_darts_weight_alpha_estimation(
                    self.current_task)
            else:
                acc = self._darts_weight_estimation(self.current_task)

        # Scale reward to (min_rew, max_rew) range, [-min, max]
        reward = self.scale_reward(acc)

        return reward, acc

    def scale_reward(self, accuracy):
        """
        Map the accuracy of the network to [min_rew, max_rew]
        for the environment.

        Mapping the accuracy in [s1, s2] to [b1, b2]

        for s in [s1, s2] to obtain the reward we compute
        reward = b1 + ((s-a1)*(b2-b1)) / (a2-a1)
        """
        # Map accuracies greater than the baseline to
        # [0, 1]
        reward = 0.0

        # Else, the reward is 0
        if self.baseline_acc == accuracy:
            return 0.0

        if self.baseline_acc <= accuracy:
            a1, a2 = self.baseline_acc, 1.0
            b1, b2 = 0.0, self.max_rew

            reward = b1 + ((accuracy-a1)*(b2-b1)) / (a2-a1)

        # Map accuracies smaller than the baseline to
        # [-1, 0]
        elif self.baseline_acc >= accuracy:
            a1, a2 = 0.0, self.baseline_acc
            b1, b2 = self.min_rew, 0.0

            reward = b1 + ((accuracy-a1)*(b2-b1)) / (a2-a1)
            # reward = 0.1
        return reward

    def _init_darts_training(self):
        if self.test_phase:
            self.train_steps = self.config.test_task_train_steps
            self.arch_adap_steps = int(
                self.train_steps * self.config.test_adapt_steps)
        else:
            self.train_steps = self.config.darts_estimation_steps
            self.arch_adap_steps = self.train_steps
        self.task_train_steps = 0

        if self.config.w_task_anneal:
            # reset lr to base lr
            for group in self.w_optim.param_groups:
                group["lr"] = self.config.w_lr

            self.w_task_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.w_optim, self.train_steps, eta_min=0.0
            )
        else:
            self.w_task_lr_scheduler = None

        if self.config.a_task_anneal:
            for group in self.a_optim.param_groups:
                group["lr"] = self.config.alpha_lr

            self.a_task_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.a_optim, self.arch_adap_steps, eta_min=0.0
            )
        else:
            self.a_task_lr_scheduler = None

        self.model_has_normalizer = hasattr(self.meta_model, "normalizer")
        if self.model_has_normalizer:
            self.meta_model.normalizer["params"]["curr_step"] = 0.0
            self.meta_model.normalizer["params"]["max_steps"] = float(
                self.arch_adap_steps)

        if self.config.drop_path_prob > 0.0:
            # do drop path if not test phase (=in train phase) or if also use
            # in test phase
            if not self.test_phase or self.config.use_drop_path_in_meta_testing:
                self.meta_model.drop_path_prob(self.config.drop_path_prob)

        self.dropout_stage = self.config.dropout_op
        self.scale_factor = self.config.dropout_scale_factor

    def _darts_weight_alpha_estimation(self, task):
        self.meta_model.train()

        # Set operation level dropout
        if self.config.dropout_skip_connections and not \
                self.test_phase:

            # Exponential decay in dropout rate
            dropout_rate = self.dropout_stage * \
                np.exp(-self.task_train_steps * self.scale_factor)
            self.meta_model.drop_out_skip_connections(dropout_rate)

        # Take w step scheduler step
        if self.w_task_lr_scheduler is not None:
            self.w_task_lr_scheduler.step()

        if self.a_task_lr_scheduler is not None:
            self.a_task_lr_scheduler.step()

        lr = self.config.w_lr

        for step, ((train_X, train_y), (val_X, val_y)) in enumerate(
            zip(task.train_loader, task.valid_loader)
        ):
            train_X, train_y = train_X.to(
                self.config.device), train_y.to(self.config.device)
            val_X, val_y = val_X.to(
                self.config.device), val_y.to(self.config.device)

            # phase 2. architect step (alpha)
            self.a_optim.zero_grad()

            self.architect.backward(
                train_X, train_y, val_X, val_y, lr, self.w_optim)
            self.a_optim.step()

            # phase 1. child network step (w)
            self.w_optim.zero_grad()
            logits = self.meta_model(
                train_X, disable_pairwise_alphas=self.disable_pairwise_alphas)

            loss = self.meta_model.criterion(logits, train_y)
            loss.backward()

            if self.cell_type == "normal":
                nn.utils.clip_grad_norm_(
                    self.meta_model.normal_weights(), self.config.w_grad_clip)
            else:
                nn.utils.clip_grad_norm_(
                    self.meta_model.reduce_weights(), self.config.w_grad_clip)

            self.w_optim.step()

            # Obtain accuracy with gradient step
            logits = self.meta_model(
                train_X, sparsify_input_alphas=True,
                disable_pairwise_alphas=self.disable_pairwise_alphas)
            prec1, _ = utils.accuracy(logits, train_y, topk=(1, 5))

        if (
            self.model_has_normalizer
            and self.task_train_steps < (self.arch_adap_steps - 1)
        ):
            self.meta_model.normalizer["params"]["curr_step"] += 1

        # Step increment
        self.task_train_steps += 1

        acc = prec1.item()
        return acc

    def _tse_darts_weight_alpha_estimation(self, task):
        self.meta_model.train()

        # Set operation level dropout
        if self.config.dropout_skip_connections and not \
                self.test_phase:

            # Exponential decay in dropout rate
            dropout_rate = self.dropout_stage * \
                np.exp(-self.task_train_steps * self.scale_factor)
            self.meta_model.drop_out_skip_connections(dropout_rate)

        # Take w step scheduler step
        if self.w_task_lr_scheduler is not None:
            self.w_task_lr_scheduler.step()

        if self.a_task_lr_scheduler is not None:
            self.a_task_lr_scheduler.step()

        lr = self.config.w_lr

        self.meta_model.zero_grad()
        model_init = copy.deepcopy(self.meta_model.state_dict())

        for step, (train_X, train_y) in enumerate(task.train_loader):
            train_X, train_y = train_X.to(
                self.config.device), train_y.to(self.config.device)

            # Step 1 of Algorithm 3 - collect TSE gradient
            base_loss = self.meta_model.loss(train_X, train_y)
            base_loss.backward()

            self.w_optim.step()  # Train the weights during unrolling as normal,
            # but the architecture gradients are not zeroed during the unrolling
            self.w_optim.zero_grad()

        # Step 2 of Algorithm 3 - update the architecture encoding using
        # accumulated gradients
        self.a_optim.step()
        self.a_optim.zero_grad()  # Reset to get ready for new unrolling
        self.w_optim.zero_grad()

        # Temporary backup for new architecture encoding
        new_arch_params = copy.deepcopy(self.meta_model.alpha_normal)

        # Old weights are loaded, which also reverts the architecture encoding
        self.meta_model.load_state_dict(model_init)
        for p1, p2 in zip(self.meta_model.alpha_normal, new_arch_params):
            p1.data = p2.data

        # Step 3 of Algorithm 3 - training weights after updating the
        # architecture encoding
        for step, (train_X, train_y) in enumerate(task.train_loader):
            train_X, train_y = train_X.to(
                self.config.device), train_y.to(self.config.device)
            base_loss = self.meta_model.loss(train_X, train_y)
            base_loss.backward()
            self.w_optim.step()

            self.w_optim.zero_grad()
            self.a_optim.zero_grad()

            # Obtain accuracy with gradient step
            logits = self.meta_model(
                train_X, sparsify_input_alphas=True,
                disable_pairwise_alphas=self.disable_pairwise_alphas)
            prec1, _ = utils.accuracy(logits, train_y, topk=(1, 5))

        if (
            self.model_has_normalizer
            and self.task_train_steps < (self.arch_adap_steps - 1)
        ):
            self.meta_model.normalizer["params"]["curr_step"] += 1

        # Step increment
        self.task_train_steps += 1

        acc = prec1.item()
        return acc

    def _darts_weight_estimation(self, task):
        """Train network with one step gradient descent on the training set
        and calculate the accuracy of the test set.

        Args:
            task (Task): few-shot learning

        Returns:
            [Double]: Network Accuracy
        """

        self.meta_model.train()

        # Set operation level dropout
        if self.config.dropout_skip_connections and not \
                self.test_phase:

            # Exponential decay in dropout rate
            dropout_rate = self.dropout_stage * \
                np.exp(-self.task_train_steps * self.scale_factor)
            self.meta_model.drop_out_skip_connections(dropout_rate)

        # Take w step scheduler step
        if self.w_task_lr_scheduler is not None:
            self.w_task_lr_scheduler.step()

        # Train the weights
        for _, (train_X, train_y) in enumerate(task.train_loader):

            train_X, train_y = train_X.to(
                self.config.device), train_y.to(self.config.device)

            self.w_optim.zero_grad()
            logits = self.meta_model(train_X,
                                     disable_pairwise_alphas=self.disable_pairwise_alphas)

            loss = self.meta_model.criterion(logits, train_y)
            loss.backward()

            if self.cell_type == "normal":
                nn.utils.clip_grad_norm_(
                    self.meta_model.normal_weights(), self.config.w_grad_clip)
            else:
                nn.utils.clip_grad_norm_(
                    self.meta_model.reduce_weights(), self.config.w_grad_clip)
            # nn.utils.clip_grad_norm_(self.meta_model.weights(),
            #                          self.config.w_grad_clip)
            self.w_optim.step()

            # Obtain accuracy with gradient step
            logits = self.meta_model(train_X, sparsify_input_alphas=True,
                                     disable_pairwise_alphas=self.disable_pairwise_alphas)
            prec1, _ = utils.accuracy(logits, train_y, topk=(1, 5))

        if (
            self.model_has_normalizer
            and self.task_train_steps < (self.arch_adap_steps - 1)
        ):
            self.meta_model.normalizer["params"]["curr_step"] += 1

        # Step increment
        self.task_train_steps += 1

        acc = prec1.item()
        return acc

    def darts_evaluate_test_set(self):
        """Final evaluation over the test set
        """
        # Set max_meta_model weights
        self.meta_model.load_state_dict(self.max_meta_model)

        # for test data evaluation, turn off drop path
        if self.config.drop_path_prob > 0.0:
            self.meta_model.drop_path_prob(0.0)

        # Also, remove skip-connection dropouts during evaluation,
        # evaluation is on the train-test set.
        self.meta_model.drop_out_skip_connections(0.0)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.current_task.test_loader):
                x_test, y_test = batch
                x_test = x_test.to(self.config.device, non_blocking=True)
                y_test = y_test.to(self.config.device, non_blocking=True)

                logits = self.meta_model(
                    x_test, sparsify_input_alphas=True,
                    disable_pairwise_alphas=self.disable_pairwise_alphas)

                loss = self.meta_model.criterion(logits, y_test)
                y_test_pred = logits.softmax(dim=1)

                prec1, _ = utils.accuracy(logits, y_test, topk=(1, 5))

        acc = prec1.item()

        # Task info
        w_task = OrderedDict(
            {
                layer_name: copy.deepcopy(layer_weight)
                for layer_name, layer_weight in self.meta_model.named_weights()
                if layer_weight.grad is not None
            }
        )

        a_task = OrderedDict(
            {
                layer_name: copy.deepcopy(layer_alpha)
                for layer_name, layer_alpha in self.meta_model.named_alphas()
                if layer_alpha.grad is not None
            }
        )
        genotype = self.meta_model.genotype()

        task_info = namedtuple(
            "task_info",
            [
                "genotype",
                "top1",
                "w_task",
                "a_task",
                "loss",
                "y_test_pred",
                "sparse_num_params",
            ],
        )
        task_info.w_task = w_task
        task_info.a_task = a_task
        task_info.loss = loss
        y_test_pred = y_test_pred
        task_info.y_test_pred = y_test_pred
        task_info.genotype = genotype
        task_info.top1 = acc

        task_info.sparse_num_params = self.meta_model.get_sparse_num_params(
            self.meta_model.alpha_prune_threshold
        )
        return task_info

    def _meta_predictor_estimation(self, task):
        # Encode graph and dataset
        graph = self._meta_predictor_graph_preprocess()
        dataset = self._meta_predictor_dataset_preprocess(task)

        # Evaluate on the MetaD2A predictor
        y_pred = self.meta_predictor.evaluate_architecture(
            dataset, graph
        )

        y_pred = y_pred.item()
        return y_pred

    def _meta_predictor_graph_preprocess(self):
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
        train_X, _ = next(iter(task.train_loader))

        # Shape the image as (3, 32, 32)
        train_X = train_X.to(self.config.device)
        dataset = F.interpolate(train_X, size=(32, 32))
        dataset = fill_up_dataset(dataset, self.config.num_samples)

        assert dataset.shape[0] == self.config.num_samples, \
            "Number of samples should equal training of meta_predictor " \
            f"{dataset.shape[0]}, {self.config.num_samples}"

        # Make sure there are 3 channels
        if self.config.dataset == "omniglot" or \
                self.config.dataset == "triplemnist":
            dataset = dataset.repeat(1, 3, 1, 1)

        # Normalize the features
        if self.config.dataset == "omniglot":
            mean = torch.tensor([0.9221]).to(self.config.device)
            std = torch.tensor([0.1257]).to(self.config.device)

        elif self.config.dataset == "triplemnist":
            mean = torch.tensor([0.0439]).to(self.config.device)
            std = torch.tensor([0.1879]).to(self.config.device)

        elif self.config.dataset == "miniimagenet":
            mean = torch.tensor([0.4416]).to(self.config.device)
            std = torch.tensor([0.2328]).to(self.config.device)

        elif self.config.dataset == "omniprint" and self.config.print_split == "meta1":
            mean = torch.tensor([0.8693]).to(self.config.device)
            std = torch.tensor([0.2104]).to(self.config.device)

        elif self.config.dataset == "omniprint" and self.config.print_split == "meta2":
            mean = torch.tensor([0.8481]).to(self.config.device)
            std = torch.tensor([0.2308]).to(self.config.device)

        elif self.config.dataset == "omniprint" and self.config.print_split == "meta3":
            mean = torch.tensor([0.8670]).to(self.config.device)
            std = torch.tensor([0.2171]).to(self.config.device)

        elif self.config.dataset == "omniprint" and self.config.print_split == "meta4":
            mean = torch.tensor([0.6226]).to(self.config.device)
            std = torch.tensor([0.0818]).to(self.config.device)

        elif self.config.dataset == "omniprint" and self.config.print_split == "meta5":
            mean = torch.tensor([0.5685]).to(self.config.device)
            std = torch.tensor([0.1241]).to(self.config.device)

        else:
            raise RuntimeError(
                f"Dataset {self.config.dataset} is not supported.")
        dataset = dataset.sub_(mean).div_(std)

        # Extracts features by ResNet18
        return self.feature_extractor(dataset)

# MetaD2A helper functions


def fill_up_dataset(dataset, required_size):
    ds_size = dataset.shape[0]
    ds = copy.deepcopy(dataset)

    # In case only downsizing is required
    if ds_size >= required_size:
        return dataset[:required_size]

    while ds.size(0) < required_size:
        perm = torch.randperm(len(ds))[:1].item()
        ds = torch.cat((ds, ds[perm].unsqueeze(dim=0)))
    return ds


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

# Graph traversal helper functions


def edge_become_topk(prev_dict, states, alphas, s_idx):
    prev_topk = prev_dict['prev_states'][:, 2]
    prev_alphas = prev_dict['prev_alphas']
    topk = states[:, 2]

    if topk[s_idx] > 0.0:
        # If true, the edge became topk, calculate reward
        if (prev_topk[s_idx] < topk[s_idx]):
            return True

        return (prev_alphas[s_idx] < alphas[s_idx]).any()

    return False


def increase_edge(edges, cur_node, next_node):
    if (cur_node, next_node) in edges:
        edges[(cur_node, next_node)] += 1
    else:
        edges[(next_node, cur_node)] += 1


def get_edge_vists(edges, cur_node, next_node):
    if (cur_node, next_node) in edges:
        return edges[(cur_node, next_node)]
    else:
        return edges[(next_node, cur_node)]


def check_if_visited(edges, cur_node, next_node):
    if (cur_node, next_node) in edges:
        return edges[(cur_node, next_node)] > 0
    else:
        return edges[(next_node, cur_node)] > 0


def number_of_unique_visits(edges):
    unique_visits = 0
    for i in edges.values():
        if i > 0:
            unique_visits += 1
    return unique_visits
