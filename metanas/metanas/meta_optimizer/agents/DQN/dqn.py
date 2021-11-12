import torch
import torch.optim as optim
import torch.nn.functional as F

import time
import random

from metanas.meta_optimizer.agents.agent import RL_agent
from metanas.meta_optimizer.agents.core import count_vars
from metanas.meta_optimizer.agents.DQN.core import MLPQNetwork


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, obs, action, new_obs, reward, done):
        transition = (obs, action, new_obs, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))


class DQN(RL_agent):
    def __init__(self, config, env, test_env, steps_per_epoch=4000,
                 epochs=100, gamma=0.99, polyak=0.995, lr=5e-3,
                 num_test_episodes=10, logger_kwargs=dict(), seed=42,
                 save_freq=1, qnet_kwargs=dict(),
                 replay_size=int(1e6), batch_size=64,
                 update_after=2000, update_every=10, update_target=4,
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
        self.hidden_size = qnet_kwargs["hidden_size"]

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        self.online_network = MLPQNetwork(
            obs_dim, act_dim, **qnet_kwargs).to(self.device)
        self.target_network = MLPQNetwork(
            obs_dim, act_dim, **qnet_kwargs).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(replay_size)

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

    def get_action(self, obs):
        """Selects action from the learned Q-function
        """
        if torch.rand(1)[0] > self.epsilon:
            with torch.no_grad():
                obs = torch.Tensor(obs).to(self.device)
                act = self.online_network(obs)
                act = torch.argmax(act).item()
        else:
            act = self.env.action_space.sample()
        return act

    def update(self, batch_size=32):
        """Perform the update rule on the function approximator,
           the Neural Network
        """
        obs, action, new_obs, reward, done = self.replay_buffer.sample(
            batch_size)

        obs = torch.Tensor(obs).to(self.device)
        new_obs = torch.Tensor(new_obs).to(self.device)
        reward = torch.Tensor([reward]).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        done = torch.Tensor(done).to(self.device)

        q_values = self.online_network(obs)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_network(new_obs).detach()
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

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
        q_info = dict(QVals=q_values.cpu().detach().numpy())
        self.logger.store(LossQ=loss.item(), **q_info)

    def test_agent(self):
        for _ in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                a = self.get_action(o)
                o, r, d, _ = self.test_env.step(a)
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train_agent(self, env):
        self.env = env
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        for t in range(self.total_steps):
            a = self.get_action(o)
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            self.replay_buffer.push(o, a, o2, r, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len, Eps=self.epsilon)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

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
