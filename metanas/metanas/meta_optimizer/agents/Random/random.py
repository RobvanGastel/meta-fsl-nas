import time

from metanas.meta_optimizer.agents.agent import RL_agent


class RandomAgent(RL_agent):
    def __init__(self, config, env, logger_kwargs=dict(), seed=42, save_freq=1,
                 gamma=0.99, lr=1e-3, epochs=1, steps_per_epoch=4000,
                 number_of_trajectories=10, count_trajectories=True):
        super().__init__(config, env, logger_kwargs,
                         seed, gamma, lr, save_freq)

        # Meta-learning parameters
        # Either give a fixed number of trajectories or steps per trial
        self.count_trajectories = count_trajectories
        if count_trajectories:
            self.number_of_trajectories = number_of_trajectories
            self.current_test_epoch = 0
            self.current_epoch = 0
        else:
            self.steps_per_epoch = steps_per_epoch
        # epochs = 1, if every task gets a single trial
        self.epochs = epochs

        # Meta-testing environment
        self.test_env = None

        self.global_steps = 0
        self.global_test_steps = 0
        self.start_time = None
        self.total_steps = epochs * steps_per_epoch

    def train_agent(self, env):
        """Performs one meta-training trial in RL^2 fashion

        Args:
            env (gym.Env): The given task for the trial
        """
        assert env is not None, "Pass a task for the current trial"

        self.env = env
        self.start_time = time.time()

        if self.count_trajectories:
            for epoch in range(self.epochs):
                # Inbetween trials reset the hidden weights

                # To sample k trajectories
                for _ in range(self.number_of_trajectories):
                    d, ep_ret, ep_len, ep_max_acc = False, 0, 0, 0
                    self.env.reset()

                    while not(d or (ep_len == self.max_ep_len)):
                        a = self.env.action_space.sample()
                        _, r, _, info = self.env.step(a)

                        # DARTS information
                        if 'acc' in info:
                            acc = info['acc']
                            if acc is not None and acc > ep_max_acc:
                                ep_max_acc = acc

                        ep_ret += r
                        ep_len += 1

                        # Keep track of total environment interactions
                        self.global_steps += 1

                    self.logger.store(EpRet=ep_ret, EpLen=ep_len,
                                      MaxAcc=ep_max_acc)
                self.current_epoch += 1

                self._log_trial(self.global_steps, self.current_epoch)

        else:
            # Based on number of steps per steps_per_epoch
            d, ep_ret, ep_len, ep_max_acc = False, 0, 0, 0
            self.env.reset()

            for t in range(self.global_steps,
                           self.global_steps+self.total_steps):

                a = self.env.action_space.sample()
                _, r, d, info = self.env.step(a)

                # DARTS information
                if 'acc' in info:
                    acc = info['acc']
                    if acc is not None and acc > ep_max_acc:
                        ep_max_acc = acc

                ep_ret += r
                ep_len += 1

                if d or (ep_len == self.max_ep_len):
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len,
                                      MaxAcc=ep_max_acc)

                    d, ep_ret, ep_len, ep_max_acc = False, 0, 0, 0
                    self.env.reset()

                # End of epoch handling
                if (t+1) % self.steps_per_epoch == 0:
                    epoch = (t+1) // self.steps_per_epoch
                    self._log_trial(t, epoch)

            self.global_steps += self.total_steps

    def test_agent(self, env):
        """Performs one meta-testing trial in RL^2 fashion

        Args:
            env (gym.Env): The given task for the trial
        """
        self.test_env = env
        self.start_time = time.time()

        if self.count_trajectories:
            for epoch in range(self.epochs):
                # Inbetween trials reset the hidden weights

                # To sample k trajectories
                for _ in range(self.number_of_trajectories):
                    d, ep_ret, ep_len, ep_max_acc = False, 0, 0, 0
                    self.test_env.reset()

                    while not(d or (ep_len == self.max_ep_len)):
                        a = self.test_env.action_space.sample()
                        _, r, _, info = self.test_env.step(a)

                        # DARTS information
                        if 'acc' in info:
                            acc = info['acc']
                            if acc is not None and acc > ep_max_acc:
                                ep_max_acc = acc

                        ep_ret += r
                        ep_len += 1

                        # Keep track of total environment interactions
                        self.global_test_steps += 1

                    self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len,
                                      TestMaxAcc=ep_max_acc)
                self.current_test_epoch += 1

                self._log_test_trial(self.global_test_steps,
                                     self.current_test_epoch)

        else:
            # Based on number of steps per steps_per_epoch
            d, ep_ret, ep_len, ep_max_acc = False, 0, 0, 0
            self.test_env.reset()

            for t in range(self.global_test_steps,
                           self.global_test_steps+self.total_steps):

                a = self.test_env.action_space.sample()
                _, r, d, info = self.test_env.step(a)

                # DARTS information
                if 'acc' in info:
                    acc = info['acc']
                    if acc is not None and acc > ep_max_acc:
                        ep_max_acc = acc

                ep_ret += r
                ep_len += 1

                if d or (ep_len == self.max_ep_len):
                    self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len,
                                      TestMaxAcc=ep_max_acc)

                    d, ep_ret, ep_len, ep_max_acc = False, 0, 0, 0
                    self.test_env.reset()

                # End of epoch handling
                if (t+1) % self.steps_per_epoch == 0:
                    epoch = (t+1) // self.steps_per_epoch
                    self._log_test_trial(t, epoch)

            self.global_test_steps += self.total_steps

    def _log_trial(self, step, trial):
        log_perf_board = ['EpRet', 'EpLen', 'MaxAcc']

        for val in log_perf_board:
            mean, std = self.logger.get_stats(val)
            self.summary_writer.add_scalar(
                'Performance/Average'+val, mean, step)
            self.summary_writer.add_scalar(
                'Performance/Std'+val, std, step)

        self.logger.log_tabular('Trial', trial)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('MaxAcc', with_min_and_max=True)
        self.logger.log_tabular('TotalEnvInteracts', step)
        self.logger.log_tabular('Time', time.time()-self.start_time)
        self.logger.dump_tabular()

    def _log_test_trial(self, step, trial):
        log_perf_board = ['TestEpRet', 'TestEpLen', 'TestMaxAcc']

        for val in log_perf_board:
            mean, std = self.logger.get_stats(val)
            self.summary_writer.add_scalar(
                'Performance/Average'+val, mean, step)
            self.summary_writer.add_scalar(
                'Performance/Std'+val, std, step)

        self.logger.log_tabular('Trial', trial)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('TestMaxAcc', with_min_and_max=True)
        self.logger.log_tabular('TotalEnvInteracts', step)
        self.logger.log_tabular('Time', time.time()-self.start_time)
        self.logger.dump_tabular()
