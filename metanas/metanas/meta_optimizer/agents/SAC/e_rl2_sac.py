import copy
import time
import random
import collections
import itertools
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from metanas.meta_optimizer.agents.agent import RL_agent
from metanas.meta_optimizer.agents.SAC.rnn_core import ActorCritic, count_vars


class EpisodicReplayBuffer:
    def __init__(self, replay_size=int(1e6), max_ep_len=500,
                 batch_size=1, time_step=8, device=None,
                 random_update=False):

        if random_update is False and batch_size > 1:
            raise AssertionError(
                "Cant apply sequential updates with different sequence sizes in a batch")

        self.random_update = random_update
        self.max_ep_len = max_ep_len
        self.batch_size = batch_size
        self.time_step = time_step
        self.device = device

        self.memory = collections.deque(maxlen=replay_size)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_eps = []

        if self.random_update:
            sampled_episodes = random.sample(self.memory, self.batch_size)

            min_step = self.max_ep_len
            # get minimum time step possible
            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))

            for episode in sampled_episodes:
                if min_step > self.time_step:
                    idx = np.random.randint(0, len(episode)-self.time_step+1)
                    sample = episode.sample(
                        time_step=self.time_step,
                        idx=idx)
                else:
                    idx = np.random.randint(0, len(episode)-min_step+1)
                    sample = episode.sample(
                        time_step=min_step,
                        idx=idx)
                sampled_eps.append(sample)
        else:
            idx = np.random.randint(0, len(self.memory))
            sampled_eps.append(self.memory[idx].sample())

        return self._sample_to_tensor(sampled_eps, len(sampled_eps[0]['obs']))

    def _sample_to_tensor(self, sample, seq_len):
        obs = [sample[i]["obs"] for i in range(self.batch_size)]
        act = [sample[i]["acts"] for i in range(self.batch_size)]
        rew = [sample[i]["rews"] for i in range(self.batch_size)]
        next_obs = [sample[i]["next_obs"] for i in range(self.batch_size)]
        prev_act = [sample[i]["prev_act"] for i in range(self.batch_size)]
        prev_rew = [sample[i]["prev_rew"] for i in range(self.batch_size)]
        done = [sample[i]["done"] for i in range(self.batch_size)]

        obs = torch.FloatTensor(obs).reshape(
            self.batch_size, seq_len, -1).to(self.device)
        act = torch.LongTensor(act).reshape(
            self.batch_size, seq_len).to(self.device)
        rew = torch.FloatTensor(rew).reshape(
            self.batch_size, seq_len, -1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).reshape(
            self.batch_size, seq_len, -1).to(self.device)
        prev_act = torch.FloatTensor(prev_act).reshape(
            self.batch_size, seq_len).to(self.device)
        prev_rew = torch.FloatTensor(prev_rew).reshape(
            self.batch_size, seq_len, -1).to(self.device)
        done = torch.FloatTensor(done).reshape(
            self.batch_size, seq_len, -1).to(self.device)

        return (obs, act, rew, next_obs, prev_act, prev_rew, done), seq_len


class EpisodeMemory:
    """Tracks the transitions within an episode
    """

    def __init__(self, random_update):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.prev_act = []
        self.prev_rew = []
        self.done = []

        self.random_update = random_update

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.prev_act.append(transition[4])
        self.prev_rew.append(transition[5])
        self.done.append(transition[6])

    def sample(self, time_step=None, idx=None):
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        prev_act = np.array(self.prev_act)
        prev_rew = np.array(self.prev_rew)
        done = np.array(self.done)

        if self.random_update is True:
            obs = obs[idx:idx+time_step]
            action = action[idx:idx+time_step]
            reward = reward[idx:idx+time_step]
            next_obs = next_obs[idx:idx+time_step]
            prev_act = prev_act[idx:idx+time_step]
            prev_rew = prev_rew[idx:idx+time_step]
            done = done[idx:idx+time_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    prev_rew=prev_rew,
                    prev_act=prev_act,
                    done=done)

    def __len__(self):
        return len(self.obs)


class SAC(RL_agent):
    def __init__(self, config, env, ac_kwargs=dict(), gamma=0.99,
                 polyak=0.995, lr=1e-3, hidden_size=256, logger_kwargs=dict(),
                 epochs=100, steps_per_epoch=4000, start_steps=10000,
                 update_after=1000, update_every=20, batch_size=32,
                 replay_size=int(1e6), seed=42, save_freq=1,
                 num_test_episodes=10
                 ):
        super().__init__(config, env, epochs, steps_per_epoch,
                         num_test_episodes, logger_kwargs,
                         seed, gamma, lr, batch_size,
                         update_every, save_freq, hidden_size)

        self.max_ep_len = self.env.max_ep_len
        self.update_multiplier = 20
        self.random_update = True
        self.update_counter = 0

        self.polyak = polyak
        self.start_steps = start_steps
        self.update_after = update_after

        # The online and target networks
        self.ac = ActorCritic(env.observation_space,
                              env.action_space, **ac_kwargs).to(self.device)
        self.ac_targ = copy.deepcopy(self.ac)

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
        self.episode_buffer = EpisodicReplayBuffer(
            random_update=True,
            replay_size=replay_size,
            max_ep_len=self.max_ep_len,
            batch_size=batch_size,
            device=self.device,
            time_step=10
        )

        # Optimize entropy exploration-exploitation parameter
        # self.entropy_target = 0.95 * (-np.log(1 / self.env.action_space.n))
        # self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = 0.2  # self.log_alpha.exp()
        # self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)

        self.pi_params = itertools.chain(self.ac.pi.parameters(),
                                         self.ac.memory.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.pi_params, lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)

        # Logging for meta-training
        self.start_time = None

        # Count variables
        var_counts = tuple(count_vars(module)
                           for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log(
            '\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    def init_hidden_states(self, batch_size):
        return torch.zeros([1, batch_size, self.hidden_size]).to(self.device)

    def compute_critic_loss(self, batch):
        obs, act, rew, next_obs, prev_act, prev_rew, done = batch

        # Init hiddens
        h = self.init_hidden_states(batch_size=self.batch_size)
        h_q = self.init_hidden_states(batch_size=self.batch_size)

        memory_emb, _ = self.ac.memory(obs, prev_act, prev_rew, h_q)
        # print(memory_emb.shape)
        q1 = self.ac.q1(memory_emb)
        q2 = self.ac.q2(memory_emb)

        with torch.no_grad():
            # Target actions come from *current* policy
            memory_emb_pi, _ = self.ac.memory(next_obs, act, rew, h)
            _, a2, logp_a2 = self.ac.pi.sample(memory_emb_pi)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(memory_emb)
            q2_pi_targ = self.ac_targ.q2(memory_emb)

            # Next Q-value
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            # print(a2.shape, q_pi_targ.shape, logp_a2.shape)
            # To map R^|A| -> R
            # .detach()
            next_q = (a2 * (q_pi_targ - self.alpha * logp_a2)
                      ).sum(dim=-1).unsqueeze(-1)

            backup = (rew + self.gamma * (1 - done) * next_q)

        # MSE loss against Bellman backup
        loss_q1 = (
            (q1.gather(-1, act.unsqueeze(-1).long()) - backup).pow(2)).mean()
        loss_q2 = (
            (q2.gather(-1, act.unsqueeze(-1).long()) - backup).pow(2)).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().mean(1).numpy(),
                      Q2Vals=q2.cpu().detach().mean(1).numpy())

        return loss_q, q_info

    def compute_policy_loss(self, batch):
        obs, act, rew, _, prev_act, prev_rew, _ = batch

        h = self.init_hidden_states(batch_size=self.batch_size)

        # (Log of) probabilities to calculate expectations of Q and entropies.
        # action probability, log action probabilities
        memory_emb, _ = self.ac.memory(obs, prev_act, prev_rew, h)
        _, pi, logp_pi = self.ac.pi.sample(memory_emb)

        with torch.no_grad():
            q1_pi = self.ac.q1(memory_emb)
            q2_pi = self.ac.q2(memory_emb)
            q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (pi * (self.alpha * logp_pi - q_pi)).sum(-1).mean()

        # Entropy
        entropy = -torch.sum(pi * logp_pi, dim=1)

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().mean(1).numpy(),
                       entropy=entropy.cpu().detach().mean(1).numpy())

        return loss_pi, logp_pi, pi_info

    def update(self):
        batch, seq_len = self.episode_buffer.sample()

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_critic_loss(batch)
        loss_q.backward()

        nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
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

        nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Recording policy values
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # # Entropy values
        # alpha_loss = -(self.log_alpha * (logp_pi.detach() +
        #                                  self.entropy_target)).mean()

        # self.alpha_optimizer.zero_grad()
        # alpha_loss.backward()
        # self.alpha_optimizer.step()

        # self.alpha = self.log_alpha.exp()

        # # Recording alpha and alpha loss
        # self.logger.store(LossAlpha=alpha_loss.cpu().detach().numpy(),
        #                   Alpha=self.alpha.cpu().detach().numpy())

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(),
                                 self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update
                # target params, as opposed to "mul" and "add", which would
                # make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, prev_a, prev_r, hid, greedy=True):
        o = torch.Tensor(o).float().to(
            self.device).unsqueeze(0).unsqueeze(0)
        prev_a = torch.Tensor([prev_a]).float().to(
            self.device).unsqueeze(0)
        prev_r = torch.Tensor([prev_r]).float().to(
            self.device).unsqueeze(0).unsqueeze(-1)

        # Greedy action selection by the policy
        return self.ac.act(o, prev_a, prev_r, hid) if greedy \
            else self.ac.explore(o, prev_a, prev_r, hid)

    def test_agent(self):
        h = self.init_hidden_states(batch_size=1)
        a2 = self.test_env.action_space.sample()
        r2 = 0

        for _ in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0

            while not(d or (ep_len == self.max_ep_len)):
                a, h = self.get_action(
                    o, a2, r2, h, greedy=True)

                o2, r, d, _ = self.test_env.step(a)

                o = o2
                r2 = r
                a2 = a

                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train_agent(self, env=None):
        # Set start time for the entire process
        self.start_time = time.time()

        if env is not None:
            self.env = env

        episode_record = EpisodeMemory(self.random_update)
        h = self.init_hidden_states(batch_size=1)
        o, ep_ret, ep_len, ep_max_acc = self.env.reset(), 0, 0, 0

        # Previous action and reward
        a2 = self.env.action_space.sample()
        r2 = 0

        # A single trial
        for t in range(self.global_steps,
                       self.global_steps+self.total_steps):

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if t > self.start_steps:
                a, h = self.get_action(o, a2, r2, h, greedy=False)
            else:
                a = self.env.action_space.sample()

            o2, r, d, info_dict = self.env.step(a)

            if 'acc' in info_dict:
                acc = info_dict['acc']
                if acc is not None and acc > ep_max_acc:
                    ep_max_acc = acc

            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            episode_record.put([o, a, r/100.0, o2, a2, r2/100.0, d])

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2
            # Set previous action and reward
            r2 = r
            a2 = a

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.episode_buffer.put(episode_record)
                episode_record = EpisodeMemory(self.random_update)

                self.logger.store(EpRet=ep_ret, EpLen=ep_len,
                                  MaxAcc=ep_max_acc)
                o, ep_ret, ep_len, ep_max_acc = self.env.reset(), 0, 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_multiplier):
                    self.update()

            # End of trial handling
            if (t+1) % self.steps_per_epoch == 0:
                trial = (t+1) // self.steps_per_epoch

                # Save model
                if (trial % self.save_freq == 0) or (trial == self.epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Log info about the current trial
                log_perf_board = ['EpRet', 'EpLen', 'MaxAcc', 'Q2Vals',
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

                self.logger.log_tabular('Trial', trial)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                # Ignore this metric for non-NAS environments
                self.logger.log_tabular('MaxAcc', with_min_and_max=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                self.logger.log_tabular('LogPi', with_min_and_max=True)
                # self.logger.log_tabular('Alpha', average_only=True)
                # self.logger.log_tabular('LossAlpha', average_only=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)

                self.logger.log_tabular('Time', time.time()-self.start_time)
                self.logger.dump_tabular()

        # Increase global steps for the next trial
        self.global_steps += self.total_steps
