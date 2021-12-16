import shelve
import numpy as np

import time
import torch
import torch.nn as nn
from torch.optim import Adam

from metanas.meta_optimizer.agents.agent import RL_agent
from metanas.meta_optimizer.agents.core import (combined_shape)
from metanas.meta_optimizer.agents.PPO.core import (ActorCritic,
                                                    aggregate_dicts,
                                                    discount_cumsum,
                                                    aggregate_info_dicts)


class RolloutBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation
    (GAE-Lambda) for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, hidden_size, device,
                 use_exploration_sampling=False, gamma=0.99, lam=0.95):

        # Pick sampling episodes or time-steps
        self.exploration_batch = np.array([])
        self.exploitation_batch = np.array([])

        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size = size

        self.obs_buf = np.zeros(
            combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(
            combined_shape(size, act_dim), dtype=np.float32)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        # RL^2 variables
        self.prev_act_buf = np.zeros(
            combined_shape(size, act_dim), dtype=np.float32)
        self.prev_rew_buf = np.zeros(size, dtype=np.float32)
        self.hxs_buf = np.zeros((size, hidden_size), dtype=np.float32)

        self.use_exploration_sampling = use_exploration_sampling
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, logp, val, prev_act, prev_rew, hidden):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # buffer has to have room so you can store
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp

        self.hxs_buf[self.ptr] = hidden
        self.prev_act_buf[self.ptr] = prev_act
        self.prev_rew_buf[self.ptr] = prev_rew
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # EXPLOITATION calculation

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas,
                                                   self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value
        # function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.adv_buf[path_slice] = (self.adv_buf[path_slice] - self.adv_buf[
            path_slice].mean()) / (self.adv_buf[path_slice].std() + 1e-8)

        data = dict(obs=self.obs_buf[path_slice],
                    act=self.act_buf[path_slice],
                    adv=self.adv_buf[path_slice],
                    ret=self.ret_buf[path_slice],
                    logp=self.logp_buf[path_slice],
                    val=self.val_buf[path_slice],
                    prev_act=self.prev_act_buf[path_slice],
                    prev_rew=self.prev_rew_buf[path_slice],
                    hidden=self.hxs_buf[self.path_start_idx])

        data = {k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                for k, v in data.items()}
        self.exploitation_batch = np.append(self.exploitation_batch, data)

        # EXPLORATION calculation
        # set the return of the episode to 0

        if self.use_exploration_sampling:
            rews_zero = np.zeros_like(rews)

            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews_zero[:-1] + self.gamma * vals[1:] - vals[:-1]
            adv_buf = discount_cumsum(deltas, self.gamma * self.lam)

            # the next line computes rewards-to-go, to be targets for the value
            # function
            ret_buf = discount_cumsum(rews_zero, self.gamma)[:-1]

            # adv buff normalization
            adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

            data = dict(obs=self.obs_buf[path_slice],
                        act=self.act_buf[path_slice],
                        adv=np.array(adv_buf, copy=True),
                        ret=np.array(ret_buf, copy=True),
                        logp=self.logp_buf[path_slice],
                        val=self.val_buf[path_slice],
                        prev_act=self.prev_act_buf[path_slice],
                        prev_rew=self.prev_rew_buf[path_slice],
                        hidden=self.hxs_buf[self.path_start_idx])

            data = {k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                    for k, v in data.items()}
            self.exploration_batch = np.append(self.exploration_batch, data)

        self.path_start_idx = self.ptr

    def get(self, p_exploration=0.3):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.

        Args:
            p_exploration (float, optional): Percentange of exploration samples
            Defaults to 0.3, 30 percent of the batch for exploration. Only used
            in crase of exploration sampling.

        Returns:
            list: batch of episodes
        """
        if self.use_exploration_sampling:
            k = len(self.exploitation_batch)
            p = int(k*p_exploration)

            # p explore-rollouts
            explore_idx = np.random.choice(
                np.arange(len(self.exploration_batch)), p, replace=False)

            b_mask = np.ones_like(self.exploration_batch, dtype=bool)
            # Exclude p explore rollouts
            b_mask[explore_idx] = False

            # k-p exploit-rollouts
            explore = self.exploration_batch[explore_idx]
            exploit = self.exploitation_batch[b_mask]

            batch = [*explore, *exploit]
            np.random.shuffle(batch)
            return batch

        np.random.shuffle(self.exploitation_batch)
        return self.exploitation_batch

    def reset(self):
        self.obs_buf = np.zeros_like(self.obs_buf)
        self.act_buf = np.zeros_like(self.act_buf)
        self.adv_buf = np.zeros_like(self.adv_buf)
        self.rew_buf = np.zeros_like(self.rew_buf)
        self.ret_buf = np.zeros_like(self.ret_buf)
        self.val_buf = np.zeros_like(self.val_buf)
        self.logp_buf = np.zeros_like(self.logp_buf)
        self.prev_act_buf = np.zeros_like(self.prev_act_buf)
        self.prev_rew_buf = np.zeros_like(self.prev_rew_buf)

        self.exploration_batch = np.array([])
        self.exploitation_batch = np.array([])
        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.size


class PPO(RL_agent):
    def __init__(
            self, config, env, logger_kwargs=dict(), seed=42, save_freq=1,
            gamma=0.99, lr=3e-4, clip_ratio=0.2, ppo_iter=4,
            lam=0.97, target_kl=0.01, value_coef=0.25, entropy_coef=0.01,
            epochs=100, steps_per_epoch=4000, hidden_size=256,
            count_trajectories=True, number_of_trajectories=100,
            exploration_sampling=False):
        super().__init__(config, env, logger_kwargs,
                         seed, gamma, lr, save_freq)

        # Currently too init low for my purpose?
        self.lmbda = lam
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_iters = ppo_iter
        self.target_kl = target_kl
        self.hidden_size = hidden_size

        # Meta-learning parameters
        self.count_trajectories = count_trajectories
        if count_trajectories:
            self.number_of_trajectories = number_of_trajectories
            self.total_traj = 0
            self.current_test_epoch = 0
            self.current_epoch = 0
            self.steps_per_epoch = self.number_of_trajectories * self.max_ep_len
        else:
            self.steps_per_epoch = steps_per_epoch

        # NAS environment
        self.graph_walks = []

        # epochs = 1, if every task gets a single trial
        self.epochs = epochs

        # Meta-testing environment
        self.test_env = None

        self.global_steps = 0
        self.global_test_steps = 0

        self.ac = ActorCritic(env, hidden_size, self.device).to(self.device)
        self.optimizer = Adam(self.ac.parameters(), lr=lr, eps=1e-5)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        self.storage = RolloutBuffer(
            obs_dim, act_dim, self.steps_per_epoch,
            hidden_size, self.device,
            use_exploration_sampling=exploration_sampling,
            gamma=gamma, lam=lam)

    def get_action(self, obs, prev_act, prev_rew, hid):

        # obs shape: [1, obs_dim]
        obs = torch.as_tensor(
            obs, dtype=torch.float32
        ).to(self.device).unsqueeze(0)

        # Don't unsqueeze for one-hot encoding
        # act shape: [1]
        prev_act = torch.as_tensor(
            [prev_act.item()], dtype=torch.float32
        ).to(self.device)

        # rew shape: [1, 1]
        prev_rew = torch.as_tensor(
            [prev_rew], dtype=torch.float32
        ).to(self.device).unsqueeze(0)

        return self.ac.step(obs, prev_act, prev_rew, hid)

    def compute_loss(self, batch):
        obs, act, adv = batch['obs'], batch['act'], batch['adv']
        ret, logp_old = batch['ret'], batch['logp']

        # RL^2 variables
        prev_act, prev_rew = batch['prev_act'], batch['prev_rew'].view(-1, 1)
        h = batch['hidden'].view(1, -1, self.hidden_size)

        pi, _, logp = self.ac.pi(
            obs, prev_act, prev_rew, h, act, training=True)

        # Policy loss
        ratio = torch.exp(logp - logp_old)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = (torch.min(surr1, surr2)).mean()

        # Value loss
        loss_v = (
            (self.ac.v(obs, prev_act, prev_rew, h, training=True) - ret
             )**2).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        loss = -(loss_pi - self.value_coef * loss_v + self.entropy_coef * ent)

        # Setting variables for logging
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac, loss_pi=loss_pi,
                       loss_v=loss_v, loss=loss)

        return loss, pi_info

    def update(self):
        batch = self.storage.get()

        # Average results over entire batch
        pi_info_old = []
        for episode in batch:
            _, info_old = self.compute_loss(episode)
            pi_info_old.append(info_old)
        pi_info_old = aggregate_info_dicts(pi_info_old)

        pi_info_lst = []
        for _ in range(self.ppo_iters):
            batch = self.storage.get()
            for episode in batch:
                loss, loss_info = self.compute_loss(episode)
                pi_info_lst.append(loss_info)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                self.optimizer.step()

        pi_info = aggregate_dicts(pi_info_lst)

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info['ent'], pi_info['cf']
        self.logger.store(
            LossPi=pi_info['loss_pi'],
            LossV=pi_info['loss_v'], Loss=pi_info['loss'],
            KL=kl, Entropy=ent, ClipFrac=cf,
            DeltaLoss=(pi_info['loss'] - pi_info_old['loss']),
            DeltaLossPi=(pi_info['loss_pi'] - pi_info_old['loss_pi']),
            DeltaLossV=(pi_info['loss_v'] - pi_info_old['loss_v']))

    def train_agent(self, env):
        self.graph_walks = []

        if self.count_trajectories:
            # Use k trajectories per trial
            self.train_ep_agent(env)
        else:
            # Use k steps per trial
            self.train_step_agent(env)

    def train_ep_agent(self, env):
        assert env is not None, "Pass a task for the current trial"

        # Prepare for interaction with environment
        self.env = env
        start_time = time.time()

        # RL^2 variables
        a2 = np.array([[0]])
        r2 = 0

        for epoch in range(self.epochs):
            # Inbetween trials reset the hidden weights
            h_in = torch.zeros([1, 1, self.hidden_size]).to(self.device)
            h_out = torch.zeros([1, 1, self.hidden_size]).to(self.device)

            # To sample k trajectories
            for tr in range(self.number_of_trajectories):
                d, ep_ret, ep_len = False, 0, 0
                o = self.env.reset()

                while not(d or (ep_len == self.max_ep_len)):
                    # Keeping track of current hidden states
                    h_in = h_out

                    a, v, logp_a, h_out = self.get_action(o, a2, r2, h_in)
                    next_o, r, d, info_dict = self.env.step(a[0])
                    ep_ret += r
                    ep_len += 1

                    # DARTS environment information logging
                    if 'acc' in info_dict:
                        self._log_nas_info_dict(info_dict)

                    # save and log
                    self.storage.store(o, a, r, logp_a, v, a2,
                                       r2, h_in.cpu().numpy())
                    self.logger.store(VVals=v)

                    # Set the previous reward and action
                    r2 = r
                    a2 = a

                    # Update obs (critical!)
                    o = next_o

                    timeout = ep_len == self.max_ep_len

                    # Keep track of total environment interactions
                    self.global_steps += 1

                # End of trajectory handling
                if timeout:
                    _, v, _, _ = self.get_action(o, a2, r2, h_out)
                else:
                    v = 0
                self.storage.finish_path(v)

                self.logger.store(EpRet=ep_ret, EpLen=ep_len)

                if tr >= 10 and \
                        tr % 10 == 0:
                    self.update()
                    self.storage.reset()

                # Perform PPO update!
                # self.update()
                # self.storage.reset()

            self.current_epoch += 1

            self._log_trial(self.current_epoch, start_time)

        # Write graph walks
        self._log_graph_walks()

    def train_step_agent(self, env):
        assert env is not None, "Pass a task for the current trial"

        # Prepare for interaction with environment
        self.env = env
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # RL^2 variables
        h_in = torch.zeros([1, 1, self.hidden_size]).to(self.device)
        h_out = torch.zeros([1, 1, self.hidden_size]).to(self.device)

        a2 = np.array([[0]])
        r2 = 0

        for epoch in range(self.epochs):
            for t in range(self.steps_per_epoch):
                # Keeping track of current hidden states
                h_in = h_out

                a, v, logp_a, h_out = self.get_action(o, a2, r2, h_in)

                next_o, r, d, info_dict = self.env.step(a)
                ep_ret += r
                ep_len += 1

                # DARTS information
                if 'acc' in info_dict:
                    self._log_nas_info_dict(info_dict)

                # save and log
                self.storage.store(o, a, r, logp_a, v, a2,
                                   r2, h_in.cpu().numpy())
                self.logger.store(VVals=v)

                # Set the previous reward and action
                r2 = r
                a2 = a

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.steps_per_epoch-1

                # TODO: Introduce variable for update step
                if t >= 1000 and \
                        self.global_steps % 1000 == 0:
                    self.update()
                    self.storage.reset()

                # End of trajectory handling
                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print(
                            'Warning: trajectory cut off by epoch %d steps.' %
                            ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap
                    # value target
                    if timeout or epoch_ended:
                        _, v, _, _ = self.get_action(o, a2, r2, h_out)
                    else:
                        v = 0

                    self.storage.finish_path(v)

                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0

                # Increase global steps for the next trial
                self.global_steps += 1

            # TODO: Perform updates every n steps?
            # Perform PPO update!
            # self.update()
            # self.storage.reset()

            self._log_trial(epoch, start_time)

        # Write graph walks
        self._log_graph_walks()

    def test_agent(self, test_env, num_test_episodes=10):
        self.test_env = test_env

        h = torch.zeros([1, 1, self.hidden_size]).to(self.device)
        start_time = time.time()
        a2, r2 = np.array([[0]]), 0

        for _ in range(num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0

            while not(d or (ep_len == self.max_ep_len)):
                a, _, _, h = self.get_action(o, a2, r2, h)

                o2, r, d, info_dict = self.test_env.step(a)

                o = o2
                r2 = r
                a2 = a

                ep_ret += r
                ep_len += 1

                # DARTS information
                if 'acc' in info_dict:
                    self._log_nas_info_dict(info_dict)

                self.global_test_steps += 1

            # self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len,
            #                   TestEpMaxAcc=ep_max_acc)
        # self._log_test_trial(self.global_test_steps, start_time)
    def _log_graph_walks(self):
        d = shelve.open(self.config.graph_walk_path)
        d[str(self.config.graph_walk_index)] = self.graph_walks
        d.close()
        self.config.graph_walk_index += 1

    def _log_nas_info_dict(self, info_dict):
        """Log NAS environment information

        Args:
            info_dict (dict): action dict
        """
        # Accuracy information
        acc = info_dict['acc']
        if acc is not None:
            self.logger.store(
                Acc=info_dict['acc']
            )
        if 'test_acc' in info_dict:
            self.logger.store(
                TestAcc=info_dict['test_acc']
            )

        # Log graph walk information
        self.logger.store(
            NumAlphaAdj=info_dict[
                'alpha_adjustments'])
        self.logger.store(
            NumEstimations=info_dict[
                'acc_estimations'])
        self.logger.store(
            NumEdgeTrav=info_dict[
                'edge_traversals'])
        self.logger.store(
            NumIllegalEdgeTrav=info_dict[
                'illegal_edge_traversals'])
        self.logger.store(
            NumAlphaAdjBeforeTrav=info_dict[
                'alpha_adj_before_trav'])

        # End of episode logging
        if 'unique_edges' in info_dict:
            self.logger.store(
                UniqueEdges=info_dict['unique_edges']
            )

        # Graph walk logging
        if 'path_graph' in info_dict:
            self.graph_walks.append(info_dict['path_graph'])

    def _log_trial(self, epoch, start_time):
        # Log to tensorboard
        log_board = {
            'Performance': [
                'EpRet', 'EpLen', 'VVals', 'Entropy',
                'KL', 'ClipFrac', 'Time'
            ],
            'Environment': [
                'NumAlphaAdj', 'NumEstimations', 'Acc',
                'TestAcc',
                'NumEdgeTrav', 'NumIllegalEdgeTrav',
                'NumAlphaAdjBeforeTrav', 'UniqueEdges'
            ],
            'Loss': ['LossPi', 'LossV', 'Loss',
                     'DeltaLossPi', 'DeltaLossV',
                     'DeltaLoss'
                     ]}

        for key, value in log_board.items():
            for val in value:
                if val is not "Time":
                    mean, std = self.logger.get_stats(val)
                if key == 'Performance' or key == "Environment":
                    if val == 'Time':
                        self.summary_writer.add_scalar(
                            key+'/Time', time.time()-start_time,
                            self.global_steps)
                    else:
                        self.summary_writer.add_scalar(
                            key+'/Average'+val, mean, self.global_steps)
                        self.summary_writer.add_scalar(
                            key+'/Std'+val, std, self.global_steps)
                else:
                    self.summary_writer.add_scalar(
                        key+'/'+val, mean, self.global_steps)

        # Log to console with SpinningUp logger
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        # Ignore this metric for non-NAS environments
        self.logger.log_tabular(
            'Acc', average_only=True, with_min_and_max=True)
        self.logger.log_tabular(
            'TestAcc', average_only=True, with_min_and_max=True)
        # self.logger.log_tabular('EpMaxAcc', with_min_and_max=True)
        self.logger.log_tabular('VVals', with_min_and_max=True)
        self.logger.log_tabular('TotalEnvInteracts',
                                self.global_steps)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('Loss', average_only=True)
        self.logger.log_tabular('DeltaLossPi', average_only=True)
        self.logger.log_tabular('DeltaLossV', average_only=True)
        self.logger.log_tabular('DeltaLoss', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('KL', average_only=True)

        self.logger.log_tabular('NumAlphaAdj', average_only=True)
        self.logger.log_tabular('NumEstimations', average_only=True)
        self.logger.log_tabular('NumEdgeTrav', average_only=True)
        self.logger.log_tabular(
            'NumIllegalEdgeTrav', average_only=True)
        self.logger.log_tabular(
            'NumAlphaAdjBeforeTrav', average_only=True)
        self.logger.log_tabular(
            'UniqueEdges', average_only=True)

        self.logger.log_tabular('ClipFrac', average_only=True)
        self.logger.log_tabular('Time', time.time()-start_time)
        self.logger.dump_tabular()

    # def _log_test_trial(self, t, start_time):
    #     pass
        # trial = (t+1) // self.steps_per_epoch

        # # Save model
        # if (trial % self.save_freq == 0) or (trial == self.epochs):
        #     self.logger.save_state({'env': self.env}, None)

        # # Log info about the current trial
        # log_board = {
        #     'Performance': ['TestEpRet', 'TestEpLen', 'TestEpMaxAcc']}

        # # Update tensorboard
        # for key, value in log_board.items():
        #     for val in value:
        #         mean, std = self.logger.get_stats(val)

        #         if key == 'Performance':
        #             self.summary_writer.add_scalar(
        #                 key+'/Average'+val, mean, t)
        #             self.summary_writer.add_scalar(
        #                 key+'/Std'+val, std, t)
        #         else:
        #             self.summary_writer.add_scalar(
        #                 key+'/'+val, mean, t)

        # self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        # self.logger.log_tabular('TestEpLen', average_only=True)
        # # Ignore this metric for non-NAS environments
        # self.logger.log_tabular('TestEpMaxAcc', with_min_and_max=True)
        # self.logger.log_tabular('Time', time.time()-start_time)
        # self.logger.dump_tabular()
