import time
import numpy as np

from metanas.meta_optimizer.agents.agent import RL_agent
from metanas.meta_optimizer.agents.utils.mp_utils import Worker


class RandomPolicy:
    def __init__(self, action_space, n_workers):
        self._action_space = action_space
        self.n_workers = n_workers

    def act(self):
        return np.random.randint(
            self._action_space.n, size=self.n_workers)


class RandomAgent(RL_agent):
    def __init__(self, config, envs, logger_kwargs=dict(), seed=42,
                 steps_per_worker=2500, is_nas_env=False):
        super().__init__(config, envs[0], logger_kwargs,
                         seed, 0, 0)

        self.is_nas_env = is_nas_env

        # Initialize environment workers
        self.n_workers = len(envs)
        self.workers = [Worker(env) for env in envs]

        # Steps variables
        self.total_steps = 0
        self.total_test_steps = 0
        self.steps_per_worker = steps_per_worker

        self.total_trials = 0

        self.policy = RandomPolicy(envs[0].action_space, self.n_workers)

    def set_task(self, envs):
        assert len(envs) == self.n_workers, \
            "The number of environments should equal to the number of workers"

        self.workers = [Worker(env) for env in envs]

    def run_test_trial(self):

        # Track statistics
        start_time = time.time()
        ep_stats = {'ep_len': np.zeros(2), 'ep_rew': np.zeros(2)}

        # Reset environments
        for worker in self.workers:
            worker.child.send(("reset", None))

        for worker in self.workers:
            worker.child.recv()

        for _ in range(self.steps_per_worker):
            actions = self.policy.act()

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w]))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                _, rew, done, info = worker.child.recv()

                # Track episode statistics
                ep_stats['ep_len'][w] += 1
                ep_stats['ep_rew'][w] += rew

                # DARTS environment logging
                if self.is_nas_env:
                    acc = info['acc']
                    if acc is not None:
                        self.logger.store(MetaTestAcc=info['acc'])

                # Check if done or timeout
                if done or ep_stats['ep_len'][w] == self.max_ep_len:

                    # Store the information of the completed episode
                    # (e.g. total reward, episode length)
                    self.logger.store(MetaTestEpRet=ep_stats['ep_rew'][w],
                                      MetaTestEpLen=ep_stats['ep_len'][w])

                    worker.child.send(("reset", None))
                    worker.child.recv()

                    # Reset statistics
                    ep_stats['ep_len'][w] = 0
                    ep_stats['ep_rew'][w] = 0

            self.total_test_steps += self.n_workers

        # Close the workers at the end of the trial
        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        if self.is_nas_env:
            # Calculate final test reward, at the end of the episode
            task_info = self.env.darts_evaluate_test_set()
            self.logger.store(MetaTestTestAcc=task_info.top1)

        self.log_test_trial()

    def run_trial(self):

        # Track statistics
        start_time = time.time()
        ep_stats = {'ep_len': np.zeros(2), 'ep_rew': np.zeros(2)}

        # Reset environments
        for worker in self.workers:
            worker.child.send(("reset", None))

        # Set observations
        for w, worker in enumerate(self.workers):
            worker.child.recv()

        for t in range(self.steps_per_worker):
            actions = self.policy.act()

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w]))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs, rew, done, info = worker.child.recv()

                # Track episode statistics
                ep_stats['ep_len'][w] += 1
                ep_stats['ep_rew'][w] += rew

                if self.is_nas_env:
                    # DARTS environment logging
                    self._log_nas_info_dict(info)

                # Check if done or timeout
                if done or ep_stats['ep_len'][w] == self.max_ep_len:

                    # Store the information of the completed episode
                    # (e.g. total reward, episode length)
                    self.logger.store(EpRet=ep_stats['ep_rew'][w],
                                      EpLen=ep_stats['ep_len'][w])

                    worker.child.send(("reset", None))
                    worker.child.recv()

                    # Reset statistics
                    ep_stats['ep_len'][w] = 0
                    ep_stats['ep_rew'][w] = 0

            self.total_steps += self.n_workers

        # Close the workers at the end of the trial
        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        if self.is_nas_env:
            # Calculate final test reward, at the end of the episode
            task_info = self.env.darts_evaluate_test_set()
            self.logger.store(TestAcc=task_info.top1)

        self.total_trials += 1
        self.log_trial(start_time, self.total_trials)

        if self.is_nas_env:
            return task_info

    def log_trial(self, start_time, trial):
        log_board = {'Performance': ['EpRet', 'EpLen', 'Time']}

        if self.is_nas_env:
            log_board['Environment'] = [
                'NumAlphaAdj', 'NumEstimations', 'Acc', 'TestAcc',
                'NumEdgeTrav', 'NumIllegalEdgeTrav', 'NumAlphaAdjBeforeTrav',
                'UniqueEdges'
            ]

        for key, value in log_board.items():
            for val in value:
                if val is not "Time":
                    mean, std = self.logger.get_stats(val)
                if key == 'Performance' or key == "Environment":
                    if val == 'Time':
                        self.summary_writer.add_scalar(
                            key+'/Time', time.time()-start_time,
                            self.total_steps)
                    else:
                        self.summary_writer.add_scalar(
                            key+'/Average'+val, mean, self.total_steps)
                        self.summary_writer.add_scalar(
                            key+'/Std'+val, std, self.total_steps)
                else:
                    self.summary_writer.add_scalar(
                        key+'/'+val, mean, self.total_steps)

        # Log to console with SpinningUp logger
        self.logger.log_tabular('Epoch', trial)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)

        # Ignore this metric for non-NAS environments
        if self.is_nas_env:
            self.logger.log_tabular(
                'Acc', average_only=True, with_min_and_max=True)
            self.logger.log_tabular(
                'TestAcc', average_only=True, with_min_and_max=True)

            self.logger.log_tabular('NumAlphaAdj', average_only=True)
            self.logger.log_tabular('NumEstimations', average_only=True)
            self.logger.log_tabular('NumEdgeTrav', average_only=True)
            self.logger.log_tabular('NumIllegalEdgeTrav', average_only=True)
            self.logger.log_tabular('NumAlphaAdjBeforeTrav', average_only=True)
            self.logger.log_tabular('UniqueEdges', average_only=True)

        self.logger.log_tabular('Time', time.time()-start_time)
        self.logger.dump_tabular()

    def log_test_trial(self):
        # Log info about the current trial
        log_board = {
            'Performance': ['MetaTestEpRet', 'MetaTestEpLen']}

        # Ignore this metric for non-NAS environments
        if self.is_nas_env:
            log_board['Environment'] = ['MetaTestAcc', 'MetaTestTestAcc']

        for key, value in log_board.items():
            for val in value:
                mean, std = self.logger.get_stats(val)
                if key == 'Performance' or key == "Environment":
                    self.summary_writer.add_scalar(
                        key+'/Average'+val, mean, self.total_test_steps)
                    self.summary_writer.add_scalar(
                        key+'/Std'+val, std, self.total_test_steps)

    def _log_nas_info_dict(self, info_dict):
        """Log NAS environment information

        Args:
            info_dict (dict): action dict
        """
        # Accuracy information
        acc = info_dict['acc']
        if acc is not None:
            self.logger.store(Acc=info_dict['acc'])

            # TODO: If terminate on 100% accuracy.
            # if acc > 0.99:
            #     terminate = True

        self.logger.store(
            NumIllegalEdgeTrav=info_dict['illegal_edge_traversals'])

        # End of episode logging
        if 'unique_edges' in info_dict:
            self.logger.store(UniqueEdges=info_dict['unique_edges'])

        if 'alpha_adjustments' in info_dict:
            self.logger.store(NumAlphaAdj=info_dict['alpha_adjustments'])

        if 'edge_traversals' in info_dict:
            self.logger.store(NumEdgeTrav=info_dict['edge_traversals'])

        if 'alpha_adj_before_trav' in info_dict:
            self.logger.store(
                NumAlphaAdjBeforeTrav=info_dict['alpha_adj_before_trav'])

        if 'acc_estimations' in info_dict:
            self.logger.store(NumEstimations=info_dict['acc_estimations'])
