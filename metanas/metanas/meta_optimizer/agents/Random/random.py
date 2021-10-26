import time

from metanas.meta_optimizer.agents.agent import base_agent

# TODO: Refactor random search follow
# Random search and reproducibility for neural architecture search
# https://arxiv.org/abs/1902.07638


class RandomAgent(base_agent):
    def __init__(self, config, env, epochs, steps_per_epoch,
                 num_test_episodes, logger_kwargs=dict()):
        super().__init__(config, env, epochs, steps_per_epoch,
                         num_test_episodes, logger_kwargs)

        self.meta_epoch = 0
        self.start_time = None

    def train_agent(self, env):
        if self.start_time is not None:
            self.start_time = time.time()

        if env is not None:
            self.env = env

        _, d, ep_ret, ep_len = self.env.reset(), False, 0, 0

        # Epochs correspond to number of episodes in trial
        for t in range(self.total_steps):
            a = self.env.action_space.sample()
            _, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            if d or (ep_len == self.env.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                _, ep_ret, ep_len = self.env.reset(), 0, 0

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch
                self.meta_epoch += 1
                self._log_episode(t, epoch)

    def _log_episode(self, step, episode):
        log_perf_board = ['EpRet', 'EpLen']

        for val in log_perf_board:
            mean, std = self.logger.get_stats(val)
            self.summary_writer.add_scalar(
                'Performance/Average'+val, mean, step)
            self.summary_writer.add_scalar(
                'Performance/Std'+val, std, step)

        self.logger.log_tabular('Epoch', self.meta_epoch+episode)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        # self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        # self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', step)
        self.logger.log_tabular('Time', time.time()-self.start_time)
        self.logger.dump_tabular()
