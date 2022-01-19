import time

from metanas.meta_optimizer.agents.agent import RL_agent


class RandomAgent(RL_agent):
    def __init__(self, config, env, logger_kwargs=dict(), seed=42,
                 gamma=0.99, lr=1e-3, epochs=1, steps_per_epoch=4000,
                 number_of_trajectories=10, count_trajectories=True):
        super().__init__(config, env, logger_kwargs,
                         seed, gamma, lr)

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
        start_time = time.time()

        if self.count_trajectories:
            for epoch in range(self.epochs):
                # Inbetween trials reset the hidden weights

                # To sample k trajectories
                for _ in range(self.number_of_trajectories):
                    d, ep_ret, ep_len, ep_max_acc = False, 0, 0, 0
                    self.env.reset()

                    while not(d or (ep_len == self.max_ep_len)):
                        a = self.env.action_space.sample()
                        _, r, _, info_dict = self.env.step(a)

                        # DARTS environment logging
                        self._log_nas_info_dict(info_dict)

                        ep_ret += r
                        ep_len += 1

                        # Keep track of total environment interactions
                        self.global_steps += 1

                    self.logger.store(EpRet=ep_ret, EpLen=ep_len,
                                      MaxAcc=ep_max_acc)
                self.current_epoch += 1

            # Calculate final test reward, at the end of the episode
            task_info = self.env.darts_evaluate_test_set()
            self.logger.store(TestAcc=task_info.top1)

            # Log the trial experiment
            self._log_trial(self.current_epoch, start_time)

            # Return the task-info for the MAML loop
            return task_info
        else:
            raise RuntimeError(
                "step-wise training random agent not supported.")

    def test_agent(self, test_env, num_test_episodes=10):
        """Performs one meta-testing trial in RL^2 fashion

        Args:
            env (gym.Env): The given task for the trial
        """
        self.start_time = time.time()

        for _ in range(num_test_episodes):
            d, ep_ret, ep_len, ep_max_acc = False, 0, 0, 0
            test_env.reset()

            while not(d or (ep_len == self.max_ep_len)):
                a = self.test_env.action_space.sample()
                _, r, _, info_dict = test_env.step(a)

                # DARTS information logging
                self._log_test_nas_info(info_dict)

                ep_ret += r
                ep_len += 1

                # Keep track of total environment interactions
                self.global_test_steps += 1

            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len,
                              TestMaxAcc=ep_max_acc)

        # Calculate final test reward, at the end of the episode
        task_info = test_env.darts_evaluate_test_set()
        self.logger.store(MetaTestTestAcc=task_info.top1)

        self._log_test_trial(self.current_epoch)

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

        # Log graph walk information
        self.logger.store(
            NumIllegalEdgeTrav=info_dict[
                'illegal_edge_traversals'])

        # End of episode logging
        if 'unique_edges' in info_dict:
            self.logger.store(
                UniqueEdges=info_dict['unique_edges']
            )

        if 'alpha_adjustments' in info_dict:
            self.logger.store(
                NumAlphaAdj=info_dict[
                    'alpha_adjustments'])

        if 'edge_traversals' in info_dict:
            self.logger.store(
                NumEdgeTrav=info_dict[
                    'edge_traversals'])

        if 'alpha_adj_before_trav' in info_dict:
            self.logger.store(
                NumAlphaAdjBeforeTrav=info_dict[
                    'alpha_adj_before_trav'])

        if 'acc_estimations' in info_dict:
            self.logger.store(
                NumEstimations=info_dict[
                    'acc_estimations'])

        if 'test_acc' in info_dict:
            self.logger.store(
                TestAcc=info_dict['test_acc']
            )

    def _log_test_nas_info(self, info_dict):
        # Accuracy information
        acc = info_dict['acc']
        if acc is not None:
            self.logger.store(
                MetaTestAcc=info_dict['acc']
            )

        if 'test_acc' in info_dict:
            self.logger.store(
                MetaTestTestAcc=info_dict['test_acc']
            )

    def _log_trial(self, epoch, start_time):
        try:
            # Log to tensorboard
            log_board = {
                'Performance': [
                    'EpRet', 'EpLen', 'Time'
                ],
                'Environment': [
                    'NumAlphaAdj', 'NumEstimations', 'Acc',
                    'TestAcc', 'NumEdgeTrav', 'NumIllegalEdgeTrav',
                    'NumAlphaAdjBeforeTrav', 'UniqueEdges'
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

            self.logger.log_tabular('TotalEnvInteracts',
                                    self.global_steps)

            self.logger.log_tabular('NumAlphaAdj', average_only=True)
            self.logger.log_tabular('NumEstimations', average_only=True)
            self.logger.log_tabular('NumEdgeTrav', average_only=True)
            self.logger.log_tabular(
                'NumIllegalEdgeTrav', average_only=True)
            self.logger.log_tabular(
                'NumAlphaAdjBeforeTrav', average_only=True)
            self.logger.log_tabular(
                'UniqueEdges', average_only=True)

            self.logger.log_tabular('Time', time.time()-start_time)
            self.logger.dump_tabular()
        except:
            print(f"Unable to log meta-training epoch {epoch}")

    def _log_test_trial(self, epoch):
        try:
            # Log info about the current trial
            log_board = {
                'Performance': ['MetaTestEpRet', 'MetaTestEpLen'],
                'Environment': ['MetaTestAcc', 'MetaTestTestAcc']}

            for key, value in log_board.items():
                for val in value:
                    mean, std = self.logger.get_stats(val)
                    if key == 'Performance' or key == "Environment":
                        self.summary_writer.add_scalar(
                            key+'/Average'+val, mean,
                            self.global_test_steps)
                        self.summary_writer.add_scalar(
                            key+'/Std'+val, std, self.global_test_steps)
        except:
            print(f"Unable to log meta-testing epoch {epoch}")
