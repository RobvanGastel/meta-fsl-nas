import torch
from torch.optim import Adam

import numpy as np


from metanas.meta_optimizer.agents.agent import RL_agent
from metanas.meta_optimizer.agents.core import combined_shape
from metanas.meta_optimizer.agents.PPO.core import (MLPActorCritic,
                                                    discount_cumsum)


class RolloutBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation
    (GAE-Lambda) for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(
            size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, logp, val):
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
        self.adv_buf[path_slice] = discount_cumsum(
            deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value
        # function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # buffer has to be full before you can get
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick

        adv_mean = np.sum(self.adv_buf) / len(self.adv_buf)
        adv_std = np.sqrt(
            np.sum((self.adv_buf - adv_mean)**2) / len(self.adv_buf))
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class PPO(RL_agent):
    def __init__(
            self, config, env, logger_kwargs=dict(), seed=42, save_freq=1,
            gamma=0.99, pi_lr=3e-4, vf_lr=1e-3, clip_ratio=0.2,
            train_pi_iters=80, train_v_iters=80, lam=0.97, target_kl=0.01,
            epochs=100, steps_per_epoch=2000, hidden_size=256):
        super().__init__(config, env, logger_kwargs,
                         seed, gamma, pi_lr, save_freq)

        self.lmbda = lam
        self.epochs = epochs
        self.clip_ratio = clip_ratio
        self.hidden_size = hidden_size

        self.steps_per_epoch = steps_per_epoch
        self.target_kl = target_kl
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters

        self.ac = MLPActorCritic(env, hidden_size)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        self.storage = RolloutBuffer(obs_dim, act_dim, self.steps_per_epoch,
                                     gamma=gamma, lam=lam)

    def compute_loss_pi(self, batch):
        obs, act, adv, logp_old = batch['obs'], batch['act'], batch['adv'], batch['logp']

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio,
                               1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, batch):
        obs, ret = batch['obs'], batch['ret']
        return ((self.ac.v(obs) - ret)**2).mean()

    def update(self):
        batch = self.storage.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(batch)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(batch).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(batch)
            if pi_info['kl'] > 1.5 * self.target_kl:
                print(
                    'Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            self.pi_optimizer.step()

        # Value function learning
        for _ in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(batch)
            loss_v.backward()
            self.vf_optimizer.step()

    def train_agent(self):
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        for episode in range(self.epochs):
            for t in range(self.steps_per_epoch):
                a, v, logp_a = self.ac.step(
                    torch.as_tensor(o, dtype=torch.float32))

                next_o, r, d, _ = self.env.step(a)

                ep_ret += r
                ep_len += 1

                self.storage.store(o, a, r, logp_a, v)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' %
                              ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(
                            torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0

                    self.storage.finish_path(v)

                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Perform PPO update!
            self.update()

            self.logger.log_tabular('Epoch', episode)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.dump_tabular()
