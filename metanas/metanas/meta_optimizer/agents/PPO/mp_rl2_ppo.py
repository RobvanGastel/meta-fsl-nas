import time
import pickle

import copy
import torch
from torch.optim import Adam
import numpy as np

from metanas.meta_optimizer.agents.agent import RL_agent
from metanas.meta_optimizer.agents.utils.mp_utils import Worker, Buffer
from metanas.meta_optimizer.agents.PPO.mp_core import ActorCritic


def masked_mean(tensor, mask):
    return (tensor.T * mask).sum() / torch.clamp(
        (torch.ones_like(tensor.T) * mask).float().sum(), min=1.0)


class PPO(RL_agent):
    def __init__(
            self, config, meta_model, envs, logger_kwargs=dict(), seed=42,
            gamma=0.99, lr=3e-4, clip_ratio=0.2, lam=0.97, n_mini_batch=4,
            target_kl=0.05, value_coef=0.25, entropy_coef=0.01, epochs=100,
            hidden_size=256, steps_per_worker=800, sequence_length=8,
            exploration_sampling=False, use_mask=False, model_path=None,
            is_nas_env=False):
        super().__init__(config, envs[0], logger_kwargs,
                         seed, gamma, lr)

        # PPO variables
        self.lmbda = lam
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.hidden_size = hidden_size

        self.is_nas_env = is_nas_env
        self.use_mask = use_mask
        self.exploration_sampling = exploration_sampling

        # Steps variables
        self.total_steps = 0
        self.total_test_steps = 0

        self.epochs = epochs
        self.total_epochs = 0

        self.max_acc = 0.0
        self.meta_model = meta_model
        self.max_meta_model = copy.deepcopy(meta_model)

        # Buffer variables
        self.n_workers = len(envs)
        self.n_mini_batch = n_mini_batch
        self.sequence_length = sequence_length
        self.steps_per_worker = steps_per_worker

        act_dim = envs[0].action_space.n
        obs_dim = envs[0].observation_space.shape

        self.obs = np.zeros((self.n_workers,) + obs_dim, dtype=np.float32)
        if use_mask:
            self.masks = np.ones((self.n_workers, act_dim), dtype=np.float32)

        # Initialize environment workers
        self.workers = [Worker(env) for env in envs]

        self.buffer = Buffer(self.n_workers, steps_per_worker, n_mini_batch,
                             obs_dim, act_dim, hidden_size, sequence_length,
                             use_mask, self.device,
                             exploration_sampling=exploration_sampling)

        # Define the model and optimizer
        self.ac = ActorCritic(
            envs[0], hidden_size, self.device, self.sequence_length,
            use_mask=self.use_mask).to(self.device)

        self.optimizer = Adam(self.ac.parameters(), lr=lr, eps=1e-5)
        # self.optimizer = AdamW(self.ac.parameters(), lr=lr)

        # Load existing model
        if model_path['model'] and model_path['vars']:
            meta_state = torch.load(model_path['model'])
            self.ac.load_state_dict(meta_state['ac'].state_dict())
            self.optimizer.load_state_dict(meta_state['opt'].state_dict())

            vars_dict = pickle.load(open(model_path['vars'], 'r'))
            self.total_steps = vars_dict['steps']
            self.total_test_steps = vars_dict['test_steps']
            self.total_epochs = vars_dict['epoch']

            # Set up model saving
        self.logger.setup_pytorch_saver({'ac': self.ac, 'opt': self.optimizer})

    def get_action(self, obs, prev_act, prev_rew, hid, mask=None):
        # obs shape: [n_workers, obs_dim]
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)

        # Don't unsqueeze for one-hot encoding
        # act shape: [n_workers]
        prev_act = torch.as_tensor(
            prev_act, dtype=torch.float32).to(self.device)

        # rew shape: [n_workers, 1]
        prev_rew = torch.as_tensor(
            prev_rew, dtype=torch.float32).to(self.device).unsqueeze(1)

        # Mask the illegal transitions, given by the environment
        if self.use_mask:
            mask = torch.tensor(mask, dtype=torch.bool).to(self.device)

        return self.ac.step(obs, prev_act, prev_rew, hid, mask)

    def set_task(self, envs):
        assert len(envs) == self.n_workers, \
            "The number of environments should equal to the number of workers"

        self.acc_estimated = False
        self.max_acc = 0.0

        self.workers = [Worker(env) for env in envs]

    def run_test_trial(self):
        """Run single meta-reinforcement learning testing trial for a given
        number of worker steps.
        """

        # Track statistics
        ep_stats = {'ep_len': np.zeros(self.n_workers),
                    'ep_rew': np.zeros(self.n_workers)}
        # 'eps': np.zeros(self.n_workers)

        # Compute final statistics
        stats = {'MetaTestEpLen': {i: [] for i in range(self.n_workers)},
                 'MetaTestEpRet': {i: [] for i in range(self.n_workers)},
                 'MetaTestAcc': {i: [] for i in range(self.n_workers)},
                 'MetaTestMaxAcc': {i: [] for i in range(self.n_workers)}}

        # RL2 variables
        prev_act = np.zeros((self.n_workers,))
        prev_rew = np.zeros((self.n_workers,))

        # Set hidden states, resets each trial
        self.hidden_states = torch.zeros(
            [1, self.n_workers, self.hidden_size]).to(self.device)

        # Reset environments
        for worker in self.workers:
            worker.child.send(("reset", None))

        for w, worker in enumerate(self.workers):
            if self.use_mask:
                self.obs[w], self.masks[w] = worker.child.recv()
            else:
                self.obs[w] = worker.child.recv()

        # Start sampling the training data
        for t in range(self.steps_per_worker):
            actions, _, _, self.hidden_states = self.get_action(
                torch.tensor(self.obs), prev_act, prev_rew,
                self.hidden_states, self.masks)

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w].cpu().item()))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs, rew, done, info = worker.child.recv()

                if self.use_mask:
                    mask = info['mask']

                # Store temporary previous reward and action
                prev_rew[w] = rew
                prev_act[w] = actions[w]

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
                        obs, mask = worker.child.recv()
                    else:
                        obs = worker.child.recv()

                    # Reset statistics
                    ep_stats['ep_len'][w] = 0
                    ep_stats['ep_rew'][w] = 0

                # Store latest observations
                self.obs[w] = obs
                if self.use_mask:
                    self.masks[w] = mask

            self.total_test_steps += self.n_workers

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
                curr_stat += stats[key]
            trial_stats[key] = np.mean(curr_stat)

        # Log test statistics
        self.log_test_trial(trial_stats)

        return self.max_meta_model

    def run_trial(self):
        """Run single meta-reinforcement learning trial for a given number
        of worker steps.

        Returns:
            dict: The final information dictionary of the trial
        """

        for _ in range(self.epochs):

            # Sample environment steps
            self.sample_training_data()

            # Prepare the sampled data inside the buffer (splits data
            # into sequences)
            self.buffer.prepare_batch_dict()

            # Update the policy with sampled sequences
            self.update_policy()

            self.total_epochs += 1

        # Close the workers at the end of the trial
        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        if self.is_nas_env:
            if self.acc_estimated is False:
                self.logger.store(Acc=0.0)

        return self.max_meta_model

    def update_policy(self):
        for i in range(self.n_mini_batch):
            # Retrieve the to be trained mini batches via a generator
            mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            for mini_batch in mini_batch_generator:
                self.optimizer.zero_grad()
                loss, pi_info = self._train_mini_batch(mini_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ac.parameters(), max_norm=0.5)
                self.optimizer.step()

                self.logger.store(
                    Loss=loss, KL=pi_info['kl'], Entropy=pi_info['ent'],
                    ClipFrac=pi_info['cf'])

    def _train_mini_batch(self, samples):
        obs, adv = samples["obs"], samples["adv"]
        logp_old = samples["log_probs"]
        mask = samples['masks']

        # RL2 variables
        prev_act = samples['prev_actions']
        prev_rew = samples['prev_rewards'].view(-1, 1)

        # Hidden states
        hidden_states = samples["hxs"].unsqueeze(0)

        # Policy loss
        policy, _, _ = self.ac.pi(
            obs, prev_act, prev_rew, hidden_states, mask, training=True)

        normalized_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logp = policy.log_prob(samples["actions"])
        ratio = torch.exp(logp - logp_old)
        surr1 = ratio * normalized_adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio,
                            1.0 + self.clip_ratio) * normalized_adv

        loss_pi = torch.min(surr1, surr2)
        loss_pi = masked_mean(loss_pi, samples["loss_mask"])

        # Value loss
        value = self.ac.v(
            obs, prev_act, prev_rew, hidden_states, training=True)

        sampled_return = samples["values"] + samples["adv"]
        clipped_value = samples["values"] + (
            value - samples["values"]
        ).clamp(min=-self.clip_ratio, max=self.clip_ratio)

        loss_v = torch.max((value - sampled_return) ** 2,
                           (clipped_value - sampled_return) ** 2)
        loss_v = masked_mean(loss_v, samples["loss_mask"])

        # Entropy Bonus
        entropy = masked_mean(policy.entropy(), samples["loss_mask"])

        # Complete loss
        loss = -(
            loss_pi - self.value_coef * loss_v + self.entropy_coef * entropy
        )

        # Useful extra info
        # Approximate KL divergence
        approx_kl = (logp_old - logp).mean().item()

        # Clipped fraction
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        return loss, dict(kl=approx_kl, ent=entropy.item(), cf=clipfrac)

    def sample_training_data(self):
        # Track statistics
        ep_stats = {'ep_len': np.zeros(self.n_workers),
                    'ep_rew': np.zeros(self.n_workers)}

        # RL2 variables
        prev_act = np.zeros((self.n_workers,))
        prev_rew = np.zeros((self.n_workers,))

        # Set hidden states, resets each trial
        self.hidden_states = torch.zeros(
            [1, self.n_workers, self.hidden_size]).to(self.device)

        # Reset environments
        for worker in self.workers:
            worker.child.send(("reset", None))

        for w, worker in enumerate(self.workers):
            if self.use_mask:
                self.obs[w], self.masks[w] = worker.child.recv()
            else:
                self.obs[w] = worker.child.recv()

        # Start sampling the training data
        for t in range(self.steps_per_worker):

            with torch.no_grad():
                # Store variables in buffer
                self.buffer.obs[:, t] = torch.tensor(self.obs)
                self.buffer.masks[:, t] = torch.tensor(self.masks)
                self.buffer.hxs[:, t] = self.hidden_states.squeeze(0)

                actions, v, logp_a, self.hidden_states = self.get_action(
                    self.obs, prev_act, prev_rew,
                    self.hidden_states, self.masks)

                self.buffer.values[:, t] = v
                self.buffer.actions[:, t] = actions
                self.buffer.log_probs[:, t] = logp_a

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                # worker.child.send(
                #     ("step", self.buffer.actions[w, t].cpu().numpy()))
                worker.child.send(("step", actions[w].cpu().item()))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs, rew, done, info = worker.child.recv()

                self.buffer.rewards[w, t] = rew
                self.buffer.dones[w, t] = done

                if self.use_mask:
                    mask = info['mask']

                # Store temporary previous reward and action
                prev_rew[w] = rew
                prev_act[w] = actions[w]

                # Track episode statistics
                ep_stats['ep_len'][w] += 1
                ep_stats['ep_rew'][w] += rew

                # DARTS environment logging
                if self.is_nas_env:
                    self._log_nas_info_dict(info)

                if done:
                    # Store the information of the completed episode
                    # (e.g. total reward, episode length)
                    self.logger.store(EpRet=ep_stats['ep_rew'][w],
                                      EpLen=ep_stats['ep_len'][w])

                    worker.child.send(("reset", None))
                    if self.use_mask:
                        obs, mask = worker.child.recv()
                    else:
                        obs = worker.child.recv()

                    # Reset statistics
                    ep_stats['ep_len'][w] = 0
                    ep_stats['ep_rew'][w] = 0

                # Store latest observations and mask
                self.obs[w] = obs
                if self.use_mask:
                    self.masks[w] = mask

            # Additional RL2 storage
            self.buffer.prev_rewards[:, t] = torch.tensor(prev_rew)
            self.buffer.prev_actions[:, t] = torch.tensor(prev_act)

            self.total_test_steps += self.n_workers

        # Calculate advantages
        _, last_value, _, _ = self.get_action(
            torch.tensor(self.obs), prev_act, prev_rew,
            self.hidden_states, self.masks)
        self.buffer.calc_advantages(last_value, self.gamma, self.lmbda)

    def log_trial(self, start_time, trial):
        """Log meta-training trial to tensorboard

        Args:
            start_time (time.Time): start time of trial
            trial (int): Number of the trial
        """

        log_board = {
            'Performance': ['EpRet', 'EpLen', 'Entropy', 'KL', 'ClipFrac',
                            'Time'],
            'Loss': ['Loss']}

        if self.is_nas_env:
            log_board['Environment'] = [
                'NumAlphaAdj', 'NumEstimations', 'Acc', 'MaxTrialAcc',
                'TestFinetuneAcc', 'TestFinetuneLoss',
                'NumEdgeTrav', 'NumIllegalEdgeTrav', 'NumAlphaAdjBeforeTrav',
                'UniqueEdges']

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

        self.logger.log_tabular('Loss', average_only=True)
        self.logger.log_tabular('ClipFrac', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('KL', average_only=True)

        # Ignore this metric for non-NAS environments
        if self.is_nas_env:
            self.logger.log_tabular(
                'Acc', average_only=True, with_min_and_max=True)
            self.logger.log_tabular('MaxTrialAcc', average_only=True)

            self.logger.log_tabular(
                'TestFinetuneAcc', average_only=True, with_min_and_max=True)
            self.logger.log_tabular(
                'TestFinetuneLoss', average_only=True, with_min_and_max=True)

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
                'Testing/Average'+key, stats[key], self.total_test_steps)

    def log_test_test_accuracy(self, task_info):
        self.summary_writer.add_scalar(
            'Testing/MetaTestTestFinetuneAcc',
            task_info.top1, self.total_test_steps)
        self.summary_writer.add_scalar(
            'Testing/MetaTestTestFinetuneLoss',
            task_info.loss, self.total_test_steps)

    def _log_nas_info_dict(self, info_dict):
        """Log info dict for NAS environment information

        Args:
            info_dict (dict): environment information dictionary
        """

        # Accuracy information
        acc = info_dict['acc']
        if acc is not None:
            self.acc_estimated = True
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
