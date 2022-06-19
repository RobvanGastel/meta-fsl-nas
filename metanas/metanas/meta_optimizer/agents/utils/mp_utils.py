"""Copyright (c) 2021 Marco Pleines

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import torch.multiprocessing as mp

import torch
import numpy as np


def worker_process(remote, env):
    # Communication interface between the processes
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "close":
                remote.send(env.close())
                remote.close()
                break
            else:
                raise NotImplementedError
        except:
            break


class Worker:
    def __init__(self, env):
        self.env = env

        self.child, parent = mp.Pipe()
        self.process = mp.Process(target=worker_process, args=(parent, env))
        self.process.start()


class Buffer:
    """The buffer stores and prepares the training data. It supports
        recurrent policies. """

    def __init__(self, n_workers, steps_per_worker, n_mini_batch,
                 obs_dim, act_dim, hidden_size, sequence_length,
                 use_mask, device, exploration_sampling=False):

        # Setup members
        self.device = device
        self.n_workers = n_workers
        self.worker_steps = steps_per_worker

        # Settings for agent
        self.use_mask = use_mask
        self.exploration_sampling = exploration_sampling

        self.n_mini_batches = n_mini_batch
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batches

        self.sequence_length = sequence_length
        self.true_sequence_length = 0

        # Initialize the buffer's data storage
        self.rewards = np.zeros(
            (self.n_workers, self.worker_steps), dtype=np.float32)
        self.actions = torch.zeros(
            (self.n_workers, self.worker_steps), dtype=torch.long)

        # Additional RL2 variables
        self.prev_rewards = torch.zeros(
            (self.n_workers, self.worker_steps), dtype=torch.float32)
        self.prev_actions = torch.zeros(
            (self.n_workers, self.worker_steps), dtype=torch.long)

        if self.use_mask:
            self.masks = torch.zeros(
                (self.n_workers, self.worker_steps, act_dim),
                dtype=torch.float32)

        self.dones = np.zeros(
            (self.n_workers, self.worker_steps), dtype=np.bool)
        self.obs = torch.zeros(
            (self.n_workers, self.worker_steps) + obs_dim)

        self.hxs = torch.zeros(
            (self.n_workers, self.worker_steps, hidden_size))
        self.log_probs = torch.zeros((self.n_workers, self.worker_steps))
        self.values = torch.zeros((self.n_workers, self.worker_steps))
        self.advantages = torch.zeros((self.n_workers, self.worker_steps))

    def prepare_batch_dict(self):
        """Flattens the training samples and stores them inside a dictionary.
        Due to using a recurrent policy, the data is split into episodes
        or sequences beforehand.
        """

        samples = {
            "actions": self.actions,
            "values": self.values,
            "prev_rewards": self.prev_rewards,
            "prev_actions": self.prev_actions,
            "log_probs": self.log_probs,
            "adv": self.advantages,
            "obs": self.obs,
            "hxs": self.hxs,
            # The loss mask is used for masking the padding while computing
            # the loss function. This is only of significance while using
            # recurrence.
            "loss_mask": torch.ones((self.n_workers, self.worker_steps),
                                    dtype=torch.float32)
        }

        if self.use_mask:
            samples['masks'] = self.masks.bool()

        # Split data into sequences and apply zero-padding. Retrieve the
        # indices of dones as these are the last step of a whole episode
        episode_done_indices = []
        for w in range(self.n_workers):
            episode_done_indices.append(list(self.dones[w].nonzero()[0]))
            # Append the index of the last element of a trajectory as well,
            # as it "artifically" marks the end of an episode
            if len(episode_done_indices[w]) == 0 or \
                    episode_done_indices[w][-1] != self.worker_steps - 1:
                episode_done_indices[w].append(self.worker_steps - 1)

        # Split obs, values, advantages, recurrent cell states, actions and
        # log_probs into episodes and then into sequences
        max_sequence_length = 1
        for key, value in samples.items():
            sequences = []
            for w in range(self.n_workers):
                start_index = 0
                for done_index in episode_done_indices[w]:
                    # Split trajectory into episodes
                    episode = value[w, start_index:done_index + 1]
                    start_index = done_index + 1
                    # Split episodes into sequences
                    if self.sequence_length > 0:
                        for start in range(0, len(episode),
                                           self.sequence_length):
                            end = start + self.sequence_length
                            sequences.append(episode[start:end])
                        max_sequence_length = self.sequence_length
                    else:
                        # If the sequence length is not set to a proper value,
                        # sequences will be based on whole episodes
                        sequences.append(episode)
                        max_sequence_length = len(episode) if len(
                            episode
                        ) > max_sequence_length else max_sequence_length

            # Apply zero-padding to ensure that each sequence has the same
            # length. Therefore we can train batches of sequences in parallel
            # instead of one sequence at a time
            for i, sequence in enumerate(sequences):
                sequences[i] = self.pad_sequence(sequence, max_sequence_length)

            # Stack sequences (target shape: (Sequence, Step, Data ...) and
            # apply data to the samples dictionary
            samples[key] = torch.stack(sequences, axis=0)

            if key == "hxs":
                # Select only the very first recurrent cell state of a sequence
                # and add it to the samples.
                samples[key] = samples[key][:, 0]

        # If the sequence length is based on entire episodes, it will be as
        # long as the longest episode. Hence, this information has to be stored
        # for the mini batch generation.
        self.true_sequence_length = max_sequence_length

        # Flatten all samples and convert them to a tensor
        self.samples_flat = {}
        for key, value in samples.items():
            if not key == "hxs":
                value = value.reshape(
                    value.shape[0] * value.shape[1], *value.shape[2:])
            self.samples_flat[key] = value

    def pad_sequence(self, sequence, target_length):
        """Pads a sequence to the target length using zeros.

        Args:
            sequence {np.ndarray} -- The to be padded array (i.e. sequence)
            target_length {int} -- The desired length of the sequence

        Returns:
            {torch.tensor} -- Returns the padded sequence
        """
        # Determine the number of zeros that have to be added to the sequence
        delta_length = target_length - len(sequence)
        # If the sequence is already as long as the target length, don't pad
        if delta_length <= 0:
            return sequence
        # Construct array of zeros
        if len(sequence.shape) > 1:
            # Case: pad multi-dimensional array (e.g. visual observation)
            padding = torch.zeros(
                ((delta_length,) + sequence.shape[1:]), dtype=sequence.dtype)
        else:
            padding = torch.zeros((delta_length), dtype=sequence.dtype)
        # Concatenate the zeros to the sequence
        return torch.cat((sequence, padding), axis=0)

    def recurrent_mini_batch_generator(self, exploration_p=0.3):
        """A recurrent generator that returns a dictionary providing training
        data arranged in mini batches. This generator shuffles the data by
        sequences.

        Yields:
            {dict} -- Mini batch data for training
        """
        # Determine the number of sequences per mini batch
        num_sequences = len(
            self.samples_flat["values"]) // self.true_sequence_length
        num_sequences_per_batch = num_sequences // self.n_mini_batches
        # Arrange a list that determines the sequence count for each
        # mini batch
        num_sequences_per_batch = [
            num_sequences_per_batch] * self.n_mini_batches
        remainder = num_sequences % self.n_mini_batches
        for i in range(remainder):
            # Add the remainder if the sequence count and the number of
            # mini batches do not share a common divider
            num_sequences_per_batch[i] += 1

        # Prepare indices, but only shuffle the sequence indices and not
        # the entire batch.
        indices = torch.arange(
            0, num_sequences * self.true_sequence_length).reshape(
            num_sequences, self.true_sequence_length)
        sequence_indices = torch.randperm(num_sequences)
        # At this point it is assumed that all of the available training
        # data (values, observations, actions, ...) is padded.

        # Compose mini batches
        start = 0
        for n_sequences in num_sequences_per_batch:
            end = start + n_sequences

            mini_batch_indices = indices[
                sequence_indices[start:end]].reshape(-1)

            # Exploration sampling
            if self.exploration_sampling:
                mini_batch_seqs = indices[sequence_indices[start:end]]

                n = len(mini_batch_seqs)
                # p explore-rollouts
                p = int(n * exploration_p)
                explore_indices = np.random.choice(
                    len(mini_batch_seqs), p, replace=False)

            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key != "hxs":
                    if self.exploration_sampling and key == "values":

                        # k rollouts sequences
                        rollouts = value[mini_batch_seqs].to(self.device)

                        # Leaving with k-p exploit-rollouts
                        rollouts[explore_indices] = torch.zeros(
                            self.sequence_length).to(self.device)

                        mini_batch[key] = rollouts.reshape(-1)
                    else:
                        mini_batch[key] = value[
                            mini_batch_indices].to(self.device)
                else:
                    # Collect only the recurrent cell states that are at
                    # the beginning of a sequence
                    mini_batch[key] = value[sequence_indices[start:end]].to(
                        self.device)
            start = end

            yield mini_batch

    def calc_advantages(self, last_value: torch.tensor,
                        gamma: float, lamda: float) -> None:
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {torch.tensor} -- Value of the last agent's state
            gamma {float} -- Discount factor
            lamda {float} -- GAE regularization parameter
        """
        with torch.no_grad():
            last_advantage = 0
            # mask values on terminal states
            mask = torch.tensor(self.dones).logical_not()
            rewards = torch.tensor(self.rewards)
            for t in reversed(range(self.worker_steps)):
                last_value = last_value * mask[:, t]
                last_advantage = last_advantage * mask[:, t]
                delta = rewards[:, t] + gamma * last_value - self.values[:, t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.advantages[:, t] = last_advantage
                last_value = self.values[:, t]
