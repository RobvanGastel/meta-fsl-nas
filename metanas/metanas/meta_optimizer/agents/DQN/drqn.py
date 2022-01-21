
import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import collections
import time
import random

from metanas.meta_optimizer.agents.agent import RL_agent
from metanas.meta_optimizer.agents.core import count_vars
from metanas.meta_optimizer.agents.DQN.core import LSTMQNetwork


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
        done = [sample[i]["done"] for i in range(self.batch_size)]

        obs = torch.FloatTensor(obs).reshape(
            self.batch_size, seq_len, -1).to(self.device)
        act = torch.LongTensor(act).reshape(
            self.batch_size, seq_len, -1).to(self.device)
        rew = torch.FloatTensor(rew).reshape(
            self.batch_size, seq_len, -1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).reshape(
            self.batch_size, seq_len, -1).to(self.device)
        done = torch.FloatTensor(done).reshape(
            self.batch_size, seq_len, -1).to(self.device)

        return obs, act, rew, next_obs, done, seq_len


class EpisodeMemory:
    """Tracks the transitions within an episode
    """

    def __init__(self, random_update):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

        self.random_update = random_update

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, time_step=None, idx=None):
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if self.random_update is True:
            obs = obs[idx:idx+time_step]
            action = action[idx:idx+time_step]
            reward = reward[idx:idx+time_step]
            next_obs = next_obs[idx:idx+time_step]
            done = done[idx:idx+time_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self):
        return len(self.obs)


class DRQN(RL_agent):
    def __init__(self, config, env, test_env, steps_per_epoch=5000,
                 epochs=100, gamma=0.99, lr=3e-4, batch_size=32,
                 num_test_episodes=10, logger_kwargs=dict(), seed=42,
                 save_freq=1, qnet_kwargs=dict(),
                 replay_size=int(1e6), update_after=4000,
                 update_every=1, update_target=2, polyak=0.995, time_step=20,
                 random_update=True,
                 epsilon=0.1, final_epsilon=0.001, epsilon_decay=0.995):
        super().__init__(config, env, logger_kwargs,
                         seed, gamma, lr, save_freq)

        self.test_env = test_env
        self.num_test_episodes = num_test_episodes

        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = self.epochs * self.steps_per_epoch

        self.update_every = update_every
        self.update_after = update_after
        self.update_target = update_target

        self.polyak = polyak
        self.random_update = random_update
        self.hidden_size = qnet_kwargs["hidden_size"]

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        self.online_network = LSTMQNetwork(
            obs_dim, act_dim, **qnet_kwargs).to(self.device)
        self.target_network = LSTMQNetwork(
            obs_dim, act_dim, **qnet_kwargs).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.episode_buffer = EpisodicReplayBuffer(
            random_update=self.random_update,
            replay_size=replay_size,
            max_ep_len=self.max_ep_len,
            batch_size=batch_size,
            device=self.device,
            time_step=time_step)

        self.optimizer = optim.Adam(
            params=self.online_network.parameters(), lr=lr)

        # Decaying eps-greedy
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        # Track when to update
        self.update_counter = 0

        # Count variables
        var_counts = tuple(count_vars(module)
                           for module in [self.online_network,
                                          self.target_network])
        self.logger.log(
            '\nNumber of parameters: \t q1: %d, \t q2: %d\n' % var_counts)

    def get_action(self, o, hidden_state, cell_state):
        """Selects action from the learned Q-function
        """
        o = torch.Tensor(o).float().to(self.device).unsqueeze(0).unsqueeze(0)

        if torch.rand(1)[0] > self.epsilon:
            with torch.no_grad():
                act, h, c = self.online_network(
                    o, hidden_state, cell_state)
                act = torch.argmax(act).item()
        else:
            with torch.no_grad():
                _, h, c = self.online_network(
                    o, hidden_state, cell_state)
            act = self.env.action_space.sample()
        return act, h, c

    def init_hidden_states(self, batch_size):
        h = torch.zeros([1, batch_size, self.hidden_size]).to(self.device)
        c = torch.zeros([1, batch_size, self.hidden_size]).to(self.device)
        return h, c

    def update(self):
        obs, act, rew, next_obs, done, seq_len = self.episode_buffer.sample()

        h_targ, c_targ = self.init_hidden_states(batch_size=self.batch_size)
        h, c = self.init_hidden_states(batch_size=self.batch_size)

        # Update method might, rl2_dqn
        q_values, _, _ = self.online_network(obs, h, c)
        q_value = q_values.gather(2, act)

        with torch.no_grad():
            q_target, _, _ = self.target_network(next_obs, h_targ, c_targ)
            next_q_value = q_target.max(2)[0].view(
                self.batch_size, seq_len, -1).detach()

        # Bellman backup equation
        expected_q_value = rew + self.gamma * next_q_value * (1 - done)
        loss = F.smooth_l1_loss(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        for param in self.online_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        if self.update_counter % self.update_target == 0:
            # Applying naive update
            # self.target_network.load_state_dict(
            #     self.online_network.state_dict())

            # Applying soft-update of the target network
            for target_param, p in zip(self.target_network.parameters(),
                                       self.online_network.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * p.data)

        self.update_counter += 1

        # Useful info for logging
        q_info = dict(QVals=q_values.cpu().mean().detach().numpy())
        self.logger.store(LossQ=loss.item(), **q_info)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            h, c = self.init_hidden_states(batch_size=1)
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                a, h, c = self.get_action(o, h, c)
                o, r, d, _ = self.test_env.step(a)
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train_agent(self, env):
        self.env = env
        episode_record = EpisodeMemory(self.random_update)
        h, c = self.init_hidden_states(batch_size=1)

        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        for t in range(self.total_steps):

            a, h, c = self.get_action(o, h, c)
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            episode_record.put([o, a, r, o2, d])

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.episode_buffer.put(episode_record)
                episode_record = EpisodeMemory(self.random_update)

                self.logger.store(EpRet=ep_ret, EpLen=ep_len, Eps=self.epsilon)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

                h, c = self.init_hidden_states(batch_size=1)

                # Update epsilon and Linear annealing
                self.epsilon = max(self.final_epsilon,
                                   self.epsilon * self.epsilon_decay)

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
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
                                  'TestEpLen', 'QVals']
                log_loss_board = ['LossQ']
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
                self.logger.log_tabular('Epsilon', self.epsilon)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('QVals', with_min_and_max=True)
                self.logger.log_tabular('LossQ', average_only=True)

                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()
