import copy
import time
import itertools
import numpy as np

import torch
from torch.optim import Adam


from metanas.meta_optimizer.agents.core import (count_vars,
                                                combined_shape)
from metanas.meta_optimizer.agents.SAC.core import MLPActorCritic
from metanas.metanas.meta_optimizer.agents.agent import RL_agent


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, device, size):
        self.obs_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                for k, v in batch.items()}


class SAC(RL_agent):
    def __init__(self, config, env, test_env, ac_kwargs=dict(),
                 steps_per_epoch=4000, epochs=100, replay_size=int(1e6),
                 gamma=0.99, polyak=0.995, lr=3e-3, alpha=0.2, batch_size=100,
                 start_steps=10000, update_after=1000, update_every=50,
                 num_test_episodes=10, logger_kwargs=dict(), save_freq=1,
                 seed=42):
        super().__init__(config, env, logger_kwargs,
                         seed, gamma, lr, save_freq)

        self.test_env = test_env
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_test_episodes = num_test_episodes
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = steps_per_epoch * epochs

        self.polyak = polyak
        self.update_every = update_every
        self.start_steps = start_steps
        self.update_after = update_after

        # The online and target networks
        self.ac = MLPActorCritic(env.observation_space,
                                 env.action_space, **ac_kwargs).to(self.device)
        self.ac_targ = copy.deepcopy(self.ac)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

        # Freeze target networks with respect to optimizers
        # (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(),
                                        self.ac.q2.parameters())

        # Set replay buffer
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape[0], self.env.action_space.n,
            self.device, replay_size)

        # Optimize alpha value for set to 0.2
        self.entropy_target = 0.98 * (-np.log(1 / self.env.action_space.n))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)

        self.update_counter = 0

        # Count variables
        var_counts = tuple(count_vars(module)
                           for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log(
            '\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    def compute_critic_loss(self, batch):
        o, o2 = batch['obs'],  batch['obs2']
        a, r, d = batch['act'], batch['rew'], batch['done']

        r = r.unsqueeze(1)
        d = d.unsqueeze(1)

        q1 = self.ac.q1(o)
        q2 = self.ac.q2(o)

        with torch.no_grad():
            # Target actions come from *current* policy
            _, a2, logp_a2 = self.ac.pi.sample(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2)
            q2_pi_targ = self.ac_targ.q2(o2)

            # Next Q-value
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            next_q = (a2 * (q_pi_targ - self.alpha * logp_a2)
                      ).sum(-1).unsqueeze(-1)

            backup = r + self.gamma * (1 - d) * next_q

        # MSE loss against Bellman backup
        loss_q1 = ((q1.gather(1, a.long()) - backup).pow(2)).mean()
        loss_q2 = ((q2.gather(1, a.long()) - backup).pow(2)).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    def compute_policy_loss(self, batch):
        o = batch['obs']

        # (Log of) probabilities to calculate expectations of Q and entropies.
        # action probability, log action probabilities
        _, pi, logp_pi = self.ac.pi.sample(o)

        with torch.no_grad():
            q1_pi = self.ac.q1(o)
            q2_pi = self.ac.q2(o)
            q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        entropy = -torch.sum(pi * logp_pi, dim=1, keepdim=True)

        # Expectations of Q
        q = torch.sum(q_pi * pi, dim=1, keepdim=True)

        # Entropy-regularized policy loss
        loss_pi = (- q - self.alpha * entropy).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy(),
                       entropy=entropy.cpu().detach().numpy())

        return loss_pi, logp_pi, pi_info

    def update(self):
        batch = self.replay_buffer.sample_batch(self.batch_size)

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_critic_loss(batch)
        loss_q.backward()
        self.q_optimizer.step()

        # Recording Q-values
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, logp_pi, pi_info = self.compute_policy_loss(batch)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Recording policy values
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Entropy values
        alpha_loss = -(self.log_alpha * (logp_pi.detach() +
                                         self.entropy_target)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        # Recording alpha and alpha loss
        self.logger.store(Alpha=alpha_loss.cpu().detach().numpy(),
                          AlphaLoss=self.alpha.cpu().detach().numpy())

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(),
                                 self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update
                # target params, as opposed to "mul" and "add", which would
                # make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, greedy=True):
        # Greedy action selection by the policy
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32)) if greedy \
            else self.ac.explore(torch.as_tensor(o, dtype=torch.float32))

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(
                    self.get_action(o, greedy=True))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train_agent(self):
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        for t in range(self.total_steps):

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if t > self.start_steps:
                a = self.ac.explore(o)
            else:
                a = self.env.action_space.sample()

            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    self.update()

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the
                # agent.
                self.test_agent()

                # Log info about epoch

                log_perf_board = ['EpRet', 'EpLen', 'TestEpRet',
                                  'TestEpLen', 'Q2Vals',
                                  'Q1Vals', 'LogPi']
                log_loss_board = ['LossPi', 'LossQ']
                log_board = {'Performance': log_perf_board,
                             'Loss': log_loss_board}

                # Update tensorboard
                for key, value in log_board.items():
                    for val in value:
                        mean, std = self.logger.get_stats(val)

                        if key == 'Performance':
                            self.summary_writer.add_scalar(
                                key+'/Average'+val, mean, t)
                            self.summary_writer.add_scalar(
                                key+'/Std'+val, std, t)
                        else:
                            self.summary_writer.add_scalar(
                                key+'/'+val, mean, t)

                epoch = (t+1) // self.steps_per_epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                self.logger.log_tabular('LogPi', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)

                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()
