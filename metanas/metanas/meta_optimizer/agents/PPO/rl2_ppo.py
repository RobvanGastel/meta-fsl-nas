import numpy as np

import time
import torch
import torch.nn as nn
from torch.optim import Adam

from metanas.meta_optimizer.agents.agent import RL_agent
from metanas.meta_optimizer.agents.core import combined_shape
from metanas.meta_optimizer.agents.PPO.core import (ActorCritic, aggregate_dicts,
                                                    discount_cumsum,
                                                    aggregate_info_dicts)


class RolloutBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation
    (GAE-Lambda) for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, hidden_size, device,
                 gamma=0.99, lam=0.95):

        # Pick sampling episodes or time-steps
        self.episode_batch = []

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
        self.episode_batch.append(data)

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # buffer has to be full before you can get
        assert self.ptr == self.max_size

        np.random.shuffle(self.episode_batch)
        return self.episode_batch

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

        self.episode_batch = []
        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.size


class PPO(RL_agent):
    def __init__(
            self, config, env, logger_kwargs=dict(), seed=42, save_freq=1,
            gamma=0.99, lr=3e-4, clip_ratio=0.2, ppo_iter=4,
            lam=0.97, target_kl=0.01, value_coef=0.25, entropy_coef=0.01,
            epochs=100, steps_per_epoch=4000, hidden_size=256):
        super().__init__(config, env, logger_kwargs,
                         seed, gamma, lr, save_freq)

        self.ppo_iters = ppo_iter
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

        # Currently too init low for my purpose?
        self.target_kl = target_kl
        self.lmbda = lam
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.hidden_size = hidden_size

        self.global_steps = 0

        self.ac = ActorCritic(env, hidden_size, self.device).to(self.device)
        self.optimizer = Adam(self.ac.parameters(), lr=lr, eps=1e-5)

        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        self.storage = RolloutBuffer(obs_dim, act_dim,
                                     self.steps_per_epoch,
                                     hidden_size,
                                     self.device,
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

        pi_info = []
        for _ in range(self.ppo_iters):
            batch = self.storage.get()
            for episode in batch:
                loss, loss_info = self.compute_loss(episode)
                pi_info.append(loss_info)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                self.optimizer.step()

            # TODO: Currently, prematurely stops the learning
            # if pi_info['kl'] > 1.5 * self.target_kl:
            #     self.logger.log(
            #         'Early stopping at step %d due to reaching max kl.' % i)
            #     break

        pi_info = aggregate_dicts(pi_info)

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(
            LossPi=pi_info['loss_pi'],
            LossV=pi_info['loss_v'], Loss=pi_info['loss'],
            KL=kl, Entropy=ent, ClipFrac=cf,
            DeltaLoss=(pi_info['loss'] - pi_info_old['loss']),
            DeltaLossPi=(pi_info['loss_pi'] - pi_info_old['loss_pi']),
            DeltaLossV=(pi_info['loss_v'] - pi_info_old['loss_v']))

    def train_agent(self):
        # Prepare for interaction with environment
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # RL^2 variables
        h_in = torch.zeros([1, 1, self.hidden_size]).to(self.device)
        h_out = torch.zeros([1, 1, self.hidden_size]).to(self.device)

        a2 = np.array([[self.env.action_space.sample()]])
        r2 = 0

        for epoch in range(self.epochs):
            for t in range(self.steps_per_epoch):
                # Keeping track of current hidden states
                h_in = h_out

                a, v, logp_a, _ = self.get_action(o, a2, r2, h_in)

                next_o, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1

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

            # Perform PPO update!
            self.update()
            self.storage.reset()

            self.log_episode(epoch, start_time)

    def log_episode(self, epoch, start_time):
        # Log to tensorboard
        log_board = {
            'Performance': [
                'EpRet', 'EpLen', 'VVals', 'Entropy',
                'KL', 'ClipFrac'
            ],
            'Loss': ['LossPi', 'LossV', 'Loss',
                     'DeltaLossPi', 'DeltaLossV',
                     'DeltaLoss']}

        for key, value in log_board.items():
            for val in value:
                mean, std = self.logger.get_stats(val)
                if key == 'Performance':
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
        self.logger.log_tabular('VVals', with_min_and_max=True)
        self.logger.log_tabular('TotalEnvInteracts',
                                (epoch+1)*self.steps_per_epoch)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('Loss', average_only=True)
        self.logger.log_tabular('DeltaLossPi', average_only=True)
        self.logger.log_tabular('DeltaLossV', average_only=True)
        self.logger.log_tabular('DeltaLoss', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('KL', average_only=True)
        self.logger.log_tabular('ClipFrac', average_only=True)
        self.logger.log_tabular('Time', time.time()-start_time)
        self.logger.dump_tabular()
