from abc import ABC

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from metanas.meta_optimizer.agents.utils.logx import EpochLogger


class base_agent(ABC):
    """Same interface for all agents for MetaNAS
    """

    def __init__(self, config, env, epochs, steps_per_epoch,
                 num_test_episodes=10, logger_kwargs=dict()):
        self.config = config

        self.env = env
        self.test_env = env  # not used

        # Number of episodes in a trial, thus,
        # epochs = 1
        self.epochs = epochs
        self.num_test_episodes = num_test_episodes

        self.global_steps = 0

        # Number of steps in a single trial
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = steps_per_epoch * epochs

        # SpinngingUp logging & Tensorboard
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.summary_writer = SummaryWriter(
            log_dir=logger_kwargs['output_dir'],
            flush_secs=1)

    def train_agent(self):
        """Agent mutates the DARTS alphas based on the given
        environment and task for a single trial.

        Returns:
            dict: Task and environment info
        """
        return


class RL_agent(base_agent):
    def __init__(self, config, env, epochs, steps_per_epoch,
                 num_test_episodes, logger_kwargs,
                 seed, gamma, lr, batch_size,
                 update_every, save_freq, hidden_size):
        super().__init__(config, env, epochs, steps_per_epoch,
                         num_test_episodes, logger_kwargs)

        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Model parameters
        self.lr = lr
        self.gamma = gamma
        self.hidden_size = hidden_size

        self.save_freq = save_freq
        self.batch_size = batch_size
        self.update_every = update_every

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def train_agent(self):
        """Agent mutates the DARTS alphas based on the given
        environment and task for a single trial.

        Returns:
            dict: Task and environment info
        """
        return
