import time
import numpy as np
import copy

from metanas.meta_optimizer.agents.agent import RL_agent
from metanas.meta_optimizer.agents.utils.mp_utils import Worker


class RandomPolicy:
    def __init__(self, action_space, n_workers):
        self._action_space = action_space
        self.n_workers = n_workers

    def act(self, mask=None):
        if mask is None:
            return np.random.randint(
                self._action_space.n, size=self.n_workers)

        # Masked random actions
        rand = np.random.random_sample(
            (self.n_workers, self._action_space.n)) * mask
        return np.argmax(rand, axis=1)


class RandomAgent(RL_agent):
    def __init__(self, config, meta_model, envs, logger_kwargs=dict(), seed=42,
                 steps_per_worker=2500, epochs=3,
                 is_nas_env=False, use_mask=False):
        super().__init__(config, envs[0], logger_kwargs,
                         seed, 0, 0)

        self.is_nas_env = is_nas_env
        self.use_mask = use_mask

        # Initialize environment workers
        self.n_workers = len(envs)
        self.workers = [Worker(env) for env in envs]

        # Steps variables
        self.total_episodes = 0
        self.total_test_episodes = 0
        # Steps per epochs, * number epochs
        self.steps_per_worker = steps_per_worker

        self.epochs = epochs
        self.total_epochs = 0
        self.total_trials = 0

        self.max_acc = 0.0
        self.meta_model = meta_model
        self.max_meta_model = copy.deepcopy(meta_model)

        act_dim = envs[0].action_space.n
        if use_mask:
            self.masks = np.ones((self.n_workers, act_dim), dtype=np.float32)

        self.policy = RandomPolicy(envs[0].action_space, self.n_workers)

    def set_task(self, envs):
        assert len(envs) == self.n_workers, \
            "The number of environments should equal to the number of workers"

        self.max_acc = 0.0
        self.workers = [Worker(env) for env in envs]

    def run_test_trial(self):

        # Track statistics
        ep_stats = {'ep_len': np.zeros(self.n_workers),
                    'ep_rew': np.zeros(self.n_workers)}

        # Compute final statistics
        stats = {'MetaTestEpLen': {i: [] for i in range(self.n_workers)},
                 'MetaTestEpRet': {i: [] for i in range(self.n_workers)},
                 'MetaTestAcc': {i: [] for i in range(self.n_workers)},
                 'MetaTestMaxAcc': {i: [] for i in range(self.n_workers)}}

        self.number_episodes = self.config.agent_test_episodes
        self.current_episodes = 0

        # Reset environments
        for worker in self.workers:
            worker.child.send(("reset", None))

        for w, worker in enumerate(self.workers):
            if self.use_mask:
                _, self.masks[w] = worker.child.recv()
            else:
                worker.child.recv()

        for _ in range(self.steps_per_worker):
            if self.use_mask:
                actions = self.policy.act(self.masks)
            else:
                actions = self.policy.act()

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w]))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                _, rew, done, info = worker.child.recv()

                if self.use_mask:
                    mask = info['mask']

                # Track episode statistics
                ep_stats['ep_len'][w] += 1
                ep_stats['ep_rew'][w] += rew

                # DARTS environment logging
                if self.is_nas_env:
                    acc = info['acc']
                    if acc is not None:
                        stats['MetaTestAcc'][w].append(acc)

                        if acc > self.max_acc:
                            self.max_acc = acc
                            stats['MetaTestMaxAcc'][w].append(acc)

                            self.max_meta_model = copy.deepcopy(
                                self.meta_model.state_dict())

                if done:
                    # Calclate the information of the completed episode
                    # (e.g. total reward, episode length)
                    stats['MetaTestEpLen'][w].append(ep_stats['ep_len'][w])
                    stats['MetaTestEpRet'][w].append(ep_stats['ep_rew'][w])

                    worker.child.send(("reset", None))
                    if self.use_mask:
                        _, mask = worker.child.recv()
                    else:
                        _ = worker.child.recv()

                    # Reset statistics
                    ep_stats['ep_len'][w] = 0
                    ep_stats['ep_rew'][w] = 0

                    self.current_episodes += 1
                    self.total_test_episodes += 1

                if self.use_mask:
                    self.masks[w] = mask

            if self.current_episodes == self.number_episodes:
                break

        # Close the workers at the end of the trial
        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        # Aggregate test statistics
        trial_stats = {}
        for key in stats.keys():

            curr_stat = []
            for w in range(self.n_workers):
                curr_stat += stats[key][w]
            trial_stats[key] = np.mean(curr_stat)

        # Log test statistics
        self.log_test_trial(trial_stats)

        return self.max_meta_model

    def run_trial(self):
        # Track statistics
        ep_stats = {'ep_len': np.zeros(self.n_workers),
                    'ep_rew': np.zeros(self.n_workers)}

        self.number_episodes = self.config.agent_train_episodes * self.epochs
        self.current_episodes = 0

        # Reset environments
        for worker in self.workers:
            worker.child.send(("reset", None))

        for w, worker in enumerate(self.workers):
            if self.use_mask:
                _, self.masks[w] = worker.child.recv()
            else:
                worker.child.recv()

        for t in range(
                self.steps_per_worker * self.epochs):
            if self.use_mask:
                actions = self.policy.act(self.masks)
            else:
                actions = self.policy.act()

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w]))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs, rew, done, info = worker.child.recv()

                if self.use_mask:
                    mask = info['mask']

                # Track episode statistics
                ep_stats['ep_len'][w] += 1
                ep_stats['ep_rew'][w] += rew

                if self.is_nas_env:
                    # DARTS environment logging
                    self._log_nas_info_dict(info)

                if done:
                    # Store the information of the completed episode
                    # (e.g. total reward, episode length)
                    self.logger.store(EpRet=ep_stats['ep_rew'][w],
                                      EpLen=ep_stats['ep_len'][w])

                    worker.child.send(("reset", None))
                    if self.use_mask:
                        _, mask = worker.child.recv()
                    else:
                        worker.child.recv()

                    # Reset statistics
                    ep_stats['ep_len'][w] = 0
                    ep_stats['ep_rew'][w] = 0

                    self.current_episodes += 1
                    self.total_episodes += 1

                if self.use_mask:
                    self.masks[w] = mask

            if self.current_episodes == self.number_episodes:
                break

        # Close the workers at the end of the trial
        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        self.total_epochs += self.epochs

        return self.max_meta_model

    def log_trial(self, start_time, trial):
        log_board = {'Performance': ['EpRet', 'EpLen', 'Time']}

        if self.is_nas_env:
            log_board['Environment'] = [
                'NumAlphaAdj', 'NumEstimations', 'Acc', 'MaxTrialAcc',
                'TestFinetuneAcc', 'TestFinetuneLoss', 'TestFinetuneParam',
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
                            self.total_episodes)
                    else:
                        self.summary_writer.add_scalar(
                            key+'/Average'+val, mean, self.total_episodes)
                        self.summary_writer.add_scalar(
                            key+'/Std'+val, std, self.total_episodes)
                else:
                    self.summary_writer.add_scalar(
                        key+'/'+val, mean, self.total_episodes)

        # Log to console with SpinningUp logger
        self.logger.log_tabular('Epoch', trial//2)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)

        # Ignore this metric for non-NAS environments
        if self.is_nas_env:
            self.logger.log_tabular(
                'Acc', average_only=True, with_min_and_max=True)
            self.logger.log_tabular(
                'MaxTrialAcc', average_only=True, with_min_and_max=True)

            self.logger.log_tabular(
                'TestFinetuneAcc', average_only=True)
            self.logger.log_tabular(
                'TestFinetuneLoss', average_only=True)
            self.logger.log_tabular(
                'TestFinetuneParam', average_only=True)

            self.logger.log_tabular('NumAlphaAdj', average_only=True)
            self.logger.log_tabular('NumEstimations', average_only=True)
            self.logger.log_tabular('NumEdgeTrav', average_only=True)
            self.logger.log_tabular('NumIllegalEdgeTrav', average_only=True)
            self.logger.log_tabular('NumAlphaAdjBeforeTrav', average_only=True)
            self.logger.log_tabular('UniqueEdges', average_only=True)

        self.logger.log_tabular('Time', time.time()-start_time)
        self.logger.dump_tabular()

    def log_test_trial(self, stats):
        """Log meta-test trial to tensorboard
        """

        log_board = {
            'Testing': ['MetaTestEpRet', 'MetaTestEpLen',
                        'MetaTestAcc', 'MetaTestMaxAcc']}

        for key in log_board['Testing']:
            self.summary_writer.add_scalar(
                'Testing/Average'+key, stats[key], self.total_test_episodes)

    def log_test_test_accuracy(self, task_info):

        self.summary_writer.add_scalar(
            'Testing/MetaTestTestFinetuneAcc',
            task_info.top1, self.total_test_episodes)
        self.summary_writer.add_scalar(
            'Testing/MetaTestTestFinetuneParam',
            int(task_info.sparse_num_params//1000), self.total_test_episodes)
        self.summary_writer.add_scalar(
            'Testing/MetaTestTestFinetuneLoss',
            task_info.loss, self.total_test_episodes)

    def _log_nas_info_dict(self, info_dict):
        """Log NAS environment information

        Args:
            info_dict (dict): action dict
        """
        # Accuracy information
        acc = info_dict['acc']
        if acc is not None:
            self.logger.store(Acc=info_dict['acc'])

            if acc > self.max_acc:
                self.max_acc = acc
                self.max_meta_model = copy.deepcopy(
                    self.meta_model.state_dict())

                self.logger.store(MaxTrialAcc=acc)

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
