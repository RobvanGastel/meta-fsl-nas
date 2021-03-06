import math
import copy
import time
import igraph

import numpy as np

import gym
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from metanas.utils import utils

import metanas.utils.genotypes as gt
from metanas.task_optimizer.darts import Architect


"""Wrapper for the RL agent to interact with the meta-model in the outer-loop
utilizing the OpenAI gym interface
"""


class NasEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, meta_model, test_phase=False,
                 cell_type="normal",
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
        self.max_acc = 0.0

        # Task acuracy estimator
        self.max_task_train_steps = config.darts_estimation_steps

        # DARTS estimation of the network
        self.task_train_steps = 0

        self.w_optim = torch.optim.Adam(
            self.meta_model.weights(),
            lr=self.config.w_lr,
            betas=(0.0, 0.999),
            weight_decay=self.config.w_weight_decay,
        )

        self.a_optim = torch.optim.Adam(
            self.meta_model.alphas(),
            self.config.alpha_lr,
            betas=(0.0, 0.999),
            weight_decay=self.config.alpha_weight_decay,
        )

        self.architect = Architect(
            self.meta_model,
            self.config.w_momentum,
            self.config.w_weight_decay,
            self.config.use_first_order_darts)

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
        # |A| + 2*|O| + 1, 2*|O| in the case of increase and decrease actions
        if self.config.env_increase_actions:
            action_size = len(self.A) + len(self.primitives)
        else:
            action_size = len(self.A) + 2*len(self.primitives)
        
        self.action_size = action_size
        self.action_space = spaces.Discrete(action_size)

        self.alpha_mask = np.zeros((action_size))

        self.alpha_prob = config.env_alpha_probability

        # Tracking statistics
        self.init_tracking_vars()

        # Environment/Gym.Env variables
        self.do_update = False
        self.max_ep_len = max_ep_len  # max_steps per episode

        # Reward range
        self.min_rew, self.max_rew = config.min_rew, config.max_rew

        self.encourage_exploration = config.encourage_exploration
        self.encourage_increase = config.encourage_increase
        self.encourage_decrease = config.encourage_decrease

        self.reward_range = (self.min_rew, self.max_rew)
        if self.encourage_exploration:
            self.reward_range = (
                self.min_rew, self.max_rew * self.encourage_increase)

        # Initialize State / Observation space
        self.initialize_observation_space()

        # Settings specific for unit testing
        if test_env:
            self.meta_state = copy.deepcopy(meta_model.state_dict())

    def init_tracking_vars(self):
        """Reset the statistics to track the trial
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
        """Reset the environment state
        """
        # Add clause for testing the environment in which the task
        # is not defined.
        assert not (self.current_task is None and self.test_env is False), \
            "A task needs to be set before evaluation"

        # Initialize the step counters
        self.step_count = 0
        self.alpha_mask = np.zeros((self.action_size))

        # Reset alphas and weights of the model
        self.meta_model.load_state_dict(copy.deepcopy(self.meta_state))

        self.update_states()

        self._init_darts_training()

        # Reset tracking statistics
        self.init_tracking_vars()

        # Set starting edge for agent
        self.set_start_state()

        # Reset best alphas and accuracy for current trial
        self.max_acc = 0.0

        # Set baseline accuracy to scale the reward
        _, self.baseline_acc = self.compute_reward()
        # self.baseline_acc = 0

        # Invalid action mask
        mask = self.invalid_mask[self.current_state_index]
        return self.current_state, mask

    def set_task(self, task, meta_state, test_phase=False):
        """The meta-loop passes the task for the environment to solve
        """

        self.current_task = task
        self.meta_state = copy.deepcopy(meta_state)

        self.reset()

        # Test phase with adjusted DARTS training
        self.test_phase = test_phase

    def initialize_observation_space(self):
        """Initialize the observation space of the environment
        """

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

        # Set (normalized) alphas
        if self.cell_type == "normal":
            # Normalize with normalizer dict
            self.normalized_alphas = self.meta_model.normalized_normal_alphas()

            self.alphas = [
                alpha.detach().cpu() for alpha in self.meta_model.alpha_normal]

        elif self.cell_type == "reduce":
            self.normalized_alphas = self.meta_model.normalized_reduce_alphas()

            self.alphas = [
                alpha.detach().cpu() for alpha in self.meta_model.alpha_reduce]

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
                self.discrete_alphas.append(edge.detach().numpy())
                self.discrete_alphas.append(edge.detach().numpy())

            for j, edge in enumerate(edges[:, :]):
                # for j, edge in enumerate(edge_one_hot):
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

                if self.config.env_increase_actions:
                    self.invalid_mask.append(
                        np.hstack((self.A[i+2], np.ones((self.n_ops)))))
                else:
                    self.invalid_mask.append(
                        np.hstack((self.A[i+2], np.ones((2*self.n_ops)))))

                self.states.append(
                    np.concatenate((
                        [i+2],
                        [j],
                        [int(j in topk_edge_indices)],
                        self.A[j],
                        edge.detach().numpy())))

                if self.config.env_increase_actions:
                    self.invalid_mask.append(
                        np.hstack((self.A[j], np.ones((self.n_ops)))))
                else:
                    self.invalid_mask.append(
                        np.hstack((self.A[j], np.ones((2*self.n_ops)))))

                s_idx += 2

        self.states = np.array(self.states)
        self.invalid_mask = np.array(self.invalid_mask)
        self.discrete_alphas = np.array(self.discrete_alphas)

        return {
            'prev_states': prev_states,
            'prev_alphas': prev_alphas
        }

    def set_start_state(self):
        """Set starting edge of the episode
        """

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
        return (torch.log(x) + C).cuda()

    def increase_op(self, row_idx, edge_idx, op_idx, prob=0.6):
        """Increase alpha value for the given incoming connection of a node.

        Args:
            row_idx (int): the index of the node
            edge_idx (int): the index of the node with the incoming connection
            op_idx (int): the index of the operation
            prob (float, optional): increase of the alpha. Defaults to 0.6.

        Returns:
            bool: whether the current state is mutated
        """
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
        """Decrease alpha value for the given incoming connection of a node.

        Args:
            row_idx (int): the index of the node
            edge_idx (int): the index of the node with the incoming connection
            op_idx (int): the index of the operation
            prob (float, optional): increase of the alpha. Defaults to 0.6.

        Returns:
            bool: whether the current state is mutated
        """
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
        """Adjust alpha value for a given edge and operation.
        """

        if self.cell_type == "normal":
            if increase:
                return self.increase_op(row_idx, edge_idx, op_idx,
                                        prob=self.alpha_prob)
            return self.decrease_op(row_idx, edge_idx, op_idx,
                                    prob=self.alpha_prob)

        elif self.cell_type == "reduce":
            if increase:
                return self.increase_op(row_idx, edge_idx, op_idx,
                                        prob=self.alpha_prob)
            return self.decrease_op(row_idx, edge_idx, op_idx,
                                    prob=self.alpha_prob)

        else:
            raise RuntimeError(f"Cell type {self.cell_type} is not supported.")

    def render(self, mode='human'):
        """Render the environment, according to the specified mode.
        """
        for row in self.states:
            print(row)

    def step(self, action):
        """Perform the given action on the environment

        Args:
            action (int): action within the range of the action space

        Returns:
            numpy.array: The next observation
            int: The step reward
            bool: Whether the current episode is finished
            dict: Episodic information
        """
        start = time.time()

        # Mutates the meta_model and the local state
        action_info, reward, acc = self._perform_action(action)

        if acc is not None and acc > 0.0:
            self.baseline_acc = acc

        # Conditions to terminate the episode
        done = self.step_count == self.max_ep_len-1 or \
            self.acc_estimations == self.max_task_train_steps-1

        # Invalid action mask
        mask = self.invalid_mask[self.current_state_index]

        # Alpha action masking
        if self.config.env_alpha_action_masking:
            mask += self.alpha_mask
        self.step_count += 1

        info_dict = {
            "steps": self.step_count,
            "mask": mask,
            "action_id": action,
            "action": action_info,
            "acc": acc,
            "max_acc": self.max_acc,
            "running_time": time.time() - start,
            "illegal_edge_traversals": self.illegal_edge_traversals,
        }

        # Final episode statistics
        if done:
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
                self.alpha_mask = np.zeros((self.action_size))

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
                    self.do_update = False

                    if not check_if_visited(self.encourage_edges,
                                            cur_node, next_node):

                        self.acc_estimations += 1
                        reward, acc = self.compute_reward()

                        # Increase the edge visists, (a, b) = (b,a)
                        increase_edge(self.encourage_edges,
                                      cur_node, next_node)

                        if self.encourage_exploration:

                            # Increase first reward
                            if reward > 0.0:
                                reward = reward * self.encourage_increase
                    else:
                        # Increase the edge visists, (a, b) = (b,a)
                        increase_edge(self.encourage_edges,
                                      cur_node, next_node)

                        if self.encourage_exploration:
                            # Decrease later rewards
                            if reward > 0.0:
                                reward = reward * self.encourage_decrease

                # States might change due to DARTS reward estimation
                self.update_states()

                # Action statistics
                self.edge_traversals += 1

                self.n_a_adj += 1
                avg = self.avg_a_adj
                n = self.n_a_adj

                # Running average update
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
        
            self.alpha_mask[action] = 1

            # Adjust action indices to fit the operations
            action = action - len(self.A)

            # Find the current edge to mutate
            row_idx, edge_idx = self.edge_to_alpha[(cur_node, next_node)]
            s_idx = self.edge_to_index[(cur_node, next_node)]

            # True parameter indicates increase op
            update = self.update_meta_model(True,
                                            row_idx,
                                            edge_idx,
                                            action)

            if update:
                # Update the local state after increasing the alphas
                prev_states = self.update_states()

                if self.config.env_topk_update:
                    # Only "Calculate reward/do_update" for reward if
                    # in top-k
                    if self.do_update is False:
                        self.do_update = edge_become_topk(
                            prev_states, self.states, self.discrete_alphas, s_idx)
                
                else:
                    self.do_update = update


            # Set current state again!
            self.current_state = self.states[s_idx]

            # Action statistics
            self.alpha_adjustments += 1
            self.a_adj_trav += 1

            action_info = f"Increase alpha ({row_idx}, {edge_idx}, {action})"

        # Decreasing the alpha for the given operation
        if action in np.arange(len(self.A)+len(self.primitives),
                               len(self.A)+2*len(self.primitives)):
            
            self.alpha_mask[action] = 1

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

                if self.config.env_topk_update:
                    # Only "Calculate reward/do_update" for reward if
                    # in top-k
                    if self.do_update is False:
                        self.do_update = edge_become_topk(
                            prev_states, self.states, self.discrete_alphas, s_idx)
                else:
                    self.do_update = update

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

        if self.config.update_weights_and_alphas:
            acc = self._darts_weight_alpha_estimation(self.current_task)
        else:
            acc = self._darts_weight_estimation(self.current_task)

        # Scale reward to (min_rew, max_rew) range, [-min, max]
        reward = self.scale_postive(acc)
        reward += self.acc_estimations * 0.2

        if self.max_acc < acc:
            self.max_acc = acc

        return reward, acc

    def scale_postive(self, accuracy):
        reward = 0.0

        # Map accuracies smaller than the baseline to
        # [-1, 0]
        if self.baseline_acc == accuracy or self.baseline_acc >= accuracy:
            return 0.0

        # Map accuracies greater than the baseline to
        # [0.5, 1]
        if self.baseline_acc <= accuracy:
            a1, a2 = self.baseline_acc, 1.0
            b1, b2 = 0.2, self.max_rew

            reward = b1 + ((accuracy-a1)*(b2-b1)) / (a2-a1)

        return reward

    def _init_darts_training(self):
        self.train_steps = self.config.darts_estimation_steps
        self.task_train_steps = 0

        if self.config.w_task_anneal:
            # reset lr to base lr
            for group in self.w_optim.param_groups:
                group["lr"] = self.config.w_lr

            self.w_task_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.w_optim, self.train_steps, eta_min=0.0)
        else:
            self.w_task_lr_scheduler = None

        if self.config.a_task_anneal:
            for group in self.a_optim.param_groups:
                group["lr"] = self.config.alpha_lr

            self.a_task_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.a_optim, self.train_steps, eta_min=0.0)
        else:
            self.a_task_lr_scheduler = None

        self.model_has_normalizer = hasattr(self.meta_model, "normalizer")
        if self.model_has_normalizer:
            self.meta_model.normalizer["params"]["curr_step"] = 0.0
            self.meta_model.normalizer["params"]["max_steps"] = float(
                self.train_steps)

        self.dropout_stage = self.config.dropout_op
        self.scale_factor = self.config.dropout_scale_factor

        if self.config.drop_path_prob > 0.0:
            # do drop path if not test phase (=in train phase) or if also use
            # in test phase
            if not self.test_phase or self.config.use_drop_path_in_meta_testing:
                self.meta_model.drop_path_prob(self.config.drop_path_prob)

    def _darts_weight_alpha_estimation(self, task):
        self.meta_model.train()

        # Exponential decay in dropout rate
        if self.config.dropout_skip_connections and not \
                self.test_phase:
            dropout_rate = self.dropout_stage * \
                np.exp(-self.task_train_steps * self.scale_factor)
            self.meta_model.drop_out_skip_connections(dropout_rate)

        # Take w step scheduler step
        if self.w_task_lr_scheduler is not None:
            self.w_task_lr_scheduler.step()

        if self.a_task_lr_scheduler is not None:
            self.a_task_lr_scheduler.step()

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
                train_X, train_y, val_X, val_y, self.config.w_lr, self.w_optim)
            self.a_optim.step()

            # phase 1. child network step (w)
            self.w_optim.zero_grad()
            logits = self.meta_model(
                train_X, disable_pairwise_alphas=self.disable_pairwise_alphas)

            loss = self.meta_model.criterion(logits, train_y)
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.meta_model.weights(), self.config.w_grad_clip)
            self.w_optim.step()

        self.task_train_steps += 1

        # Obtain accuracy with gradient step
        accs = []
        for step, (val_X, val_y) in enumerate(task.valid_loader):
            val_X, val_y = val_X.to(
                self.config.device), val_y.to(self.config.device)
            # TODO
            logits = self.meta_model(
                val_X,  # sparsify_input_alphas=True,
                disable_pairwise_alphas=self.disable_pairwise_alphas)
            prec1, _ = utils.accuracy(logits, val_y, topk=(1, 5))
            accs.append(prec1.item())

        return np.mean(accs)

    def _darts_weight_estimation(self, task):
        """Train network with one step gradient descent on the training set
        and calculate the accuracy of the test set.

        Args:
            task (Task): few-shot learning

        Returns:
            [Double]: Network Accuracy
        """

        self.meta_model.train()

        # Exponential decay in dropout rate
        if self.config.dropout_skip_connections and not \
                self.test_phase:
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
            logits = self.meta_model(
                train_X,
                disable_pairwise_alphas=self.disable_pairwise_alphas)

            loss = self.meta_model.criterion(logits, train_y)
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.meta_model.weights(), self.config.w_grad_clip)
            self.w_optim.step()

        self.task_train_steps += 1

        # Obtain accuracy with gradient step
        accs = []
        for step, (val_X, val_y) in enumerate(task.valid_loader):
            val_X, val_y = val_X.to(
                self.config.device), val_y.to(self.config.device)
            # TODO
            logits = self.meta_model(
                val_X,  # sparsify_input_alphas=True,
                disable_pairwise_alphas=self.disable_pairwise_alphas)
            prec1, _ = utils.accuracy(logits, val_y, topk=(1, 5))
            accs.append(prec1.item())
        return np.mean(accs)


def edge_become_topk(prev_dict, states, alphas, s_idx):
    prev_topk = prev_dict['prev_states'][:, 2]
    prev_alphas = prev_dict['prev_alphas']
    topk = states[:, 2]

    if topk[s_idx] > 0.0:
        # If true, the edge became topk, calculate reward
        if (prev_topk[s_idx] < topk[s_idx]):
            return True

        # TODO: For discrete
        # return (prev_alphas[s_idx] < alphas[s_idx]).any()

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
