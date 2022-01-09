from abc import ABC

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from metanas.meta_optimizer.agents.utils.logx import EpochLogger


class RL_agent(ABC):
    def __init__(self, config, env, logger_kwargs,
                 seed, gamma, lr):

        # SpinngingUp logging & Tensorboard
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.config = config

        self.env = env
        self.max_ep_len = self.env.max_ep_len

        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Model parameters
        self.lr = lr
        self.gamma = gamma

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.summary_writer = SummaryWriter(
            log_dir=logger_kwargs['output_dir'],
            flush_secs=1)

    def train_agent(self):
        """Run an iteration of learning on the agent.
        """
        return NotImplementedError
