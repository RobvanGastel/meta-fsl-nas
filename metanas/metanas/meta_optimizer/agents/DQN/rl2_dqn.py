
import torch
import torch.optim as optim
import torch.nn.functional as F

import time

from metanas.meta_optimizer.agents.agent import RL_agent
from metanas.meta_optimizer.agents.core import count_vars
from metanas.meta_optimizer.agents.buffer import EpisodicBuffer
from metanas.meta_optimizer.agents.DQN.core import RL2QNetwork


class DQN(RL_agent):
    def __init__(self, config, env, logger_kwargs=dict(), seed=42, save_freq=1,
                 gamma=0.99, lr=5e-4, qnet_kwargs=dict(), polyak=0.995,
                 steps_per_epoch=4000, epochs=1, batch_size=8,
                 replay_size=int(1e6), time_step=50, use_time_steps=False,
                 update_every=4, update_target=1000, update_after=3500,
                 epsilon=0.3, final_epsilon=0.05, epsilon_decay=0.9995):
        super().__init__(config, env, logger_kwargs,
                         seed, gamma, lr, save_freq)

        # Meta-learning parameters
        # Number of steps per trial
        self.steps_per_epoch = steps_per_epoch
        # if epochs = 1, every task runs a single trial
        self.total_steps = steps_per_epoch * epochs
        # Track the steps of all trials combined
        self.global_steps = 0

        # Updating the network parameters
        self.polyak = polyak
        self.update_counter = 0
        self.update_every = update_every
        self.update_after = update_after
        self.update_target = update_target

        self.batch_size = batch_size
        self.use_time_steps = use_time_steps
        self.hidden_size = qnet_kwargs["hidden_size"]

        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        # replay buffer to stay close to the idea of updating
        # on the whole trajectories for meta-learning purposes
        self.buffer = EpisodicBuffer(
            obs_dim, act_dim, replay_size, self.hidden_size, self.device)

        # Set epsilon greedy decaying parameters
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        # The online and target networks
        self.online_network = RL2QNetwork(
            obs_dim, act_dim, self.device, **qnet_kwargs).to(self.device)
        self.target_network = RL2QNetwork(
            obs_dim, act_dim, self.device, **qnet_kwargs).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.optimizer = optim.RMSprop(
            params=self.online_network.parameters(), lr=lr)

        # Count variables
        var_counts = tuple(count_vars(module)
                           for module in [self.online_network,
                                          self.target_network])
        self.logger.log(
            '\nNumber of parameters: \t q1: %d, \t q2: %d\n' % var_counts)

    def get_action(self, obs, prev_act, prev_rew, hid_in):
        # obs shape: [1, obs_dim]
        obs = torch.as_tensor(obs,
                              dtype=torch.float32
                              ).to(self.device).unsqueeze(0)

        # Don't unsqueeze for one-hot encoding
        # act shape: [1]
        prev_act = torch.as_tensor(
            [prev_act], dtype=torch.float32).to(self.device)

        # rew shape: [1, 1]
        prev_rew = torch.as_tensor([prev_rew],
                                   dtype=torch.float32
                                   ).to(self.device).unsqueeze(0)

        with torch.no_grad():
            act, hid_out = self.online_network(
                obs, prev_act, prev_rew, hid_in)

        if torch.rand(1)[0] > self.epsilon:
            act = torch.argmax(act).item()
        else:
            act = self.env.action_space.sample()
        return act, hid_out

    def compute_loss(self, batch):
        obs, next_obs, act = batch['obs'], batch['obs2'], batch['act']
        done, rew = batch['done'], batch['rew']

        # RL^2 variables
        prev_act, prev_rew = batch['prev_act'], batch['prev_rew'].view(-1, 1)
        h = batch['hidden'].view(1, -1, self.hidden_size)

        # Q_value for next_obs
        next_q_values, _ = self.online_network(
            next_obs, act, rew.view(-1, 1), h, training=True)
        next_q_values = next_q_values.squeeze(0)

        # argmax(Q_value(next_obs))
        next_action = torch.argmax(next_q_values, dim=-1)

        # Omit the gradients on the target network
        with torch.no_grad():
            q_target, _ = self.target_network(
                next_obs, act, rew.view(-1, 1), h, training=True)
            q_target = q_target.squeeze(0)

        max_q_target = q_target.gather(1,
                                       next_action.unsqueeze(1).long()
                                       ).squeeze(1)

        # Update with expected bellman equation
        expected_q_value = rew + self.gamma * max_q_target * (1 - done)

        # The predicted Q-value
        q_values, _ = self.online_network(
            obs, prev_act, prev_rew, h, training=True)
        q_values = q_values.squeeze(0)

        predicted_q_value = q_values.gather(-1,
                                            act.unsqueeze(1).long()
                                            ).squeeze(1)

        loss = F.smooth_l1_loss(predicted_q_value,
                                expected_q_value.detach())
        q_info = dict(QVals=q_values.cpu().mean().detach().numpy())

        return loss, q_info

    def update(self):
        batch = self.buffer.get(self.batch_size)

        for episode in batch:
            loss, q_info = self.compute_loss(episode)

            self.optimizer.zero_grad()
            loss.backward()

            for param in self.online_network.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()

        if self.update_counter % self.update_target == 0:
            # Applying naive update
            self.target_network.load_state_dict(
                self.online_network.state_dict())

        self.update_counter += 1

        # Useful info for logging
        self.logger.store(LossQ=loss.item(), **q_info)

    # def test_agent(self):
    #     for j in range(self.num_test_episodes):
    #         h = torch.zeros([1, 1, self.hidden_size]).to(self.device)

    #         o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0

    #         a2 = self.test_env.action_space.sample()
    #         r2 = 0

    #         while not(d or (ep_len == self.max_ep_len)):
    #             a, h = self.get_action(o, a2, r2, h)
    #             o, r, d, _ = self.test_env.step(a)
    #             ep_ret += r
    #             ep_len += 1
    #             r2 = r
    #             a2 = a
    #         self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train_agent(self, env):

        self.env = env
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # RL^2 variables
        h_in = torch.zeros([1, 1, self.hidden_size]).to(self.device)
        h_out = torch.zeros([1, 1, self.hidden_size]).to(self.device)

        start_time = time.time()
        a2 = 0
        r2 = 0

        for t in range(self.global_steps,
                       self.global_steps+self.total_steps):
            h_in = h_out

            a, h_out = self.get_action(o, a2, r2, h_in)
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            self.buffer.store(o, o2, a, r, d, a2, r2, h_in.cpu().numpy())

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2
            # Set previous action and reward
            r2 = r
            a2 = a

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.buffer.finish_path()

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

                # Test the performance
                # self.test_agent()

                # Log info about epoch
                log_perf_board = ['EpRet', 'EpLen', 'QVals']
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
                # self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                # self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('Epsilon', self.epsilon)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('QVals', with_min_and_max=True)
                self.logger.log_tabular('LossQ', average_only=True)

                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()

        # Increase global steps for the next trial
        self.global_steps += self.total_steps
