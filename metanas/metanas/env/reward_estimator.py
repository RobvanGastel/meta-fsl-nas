import torch
import torchvision.models as models

from metanas.task_optimizer.darts import CellArchitect
from metanas.meta_predictor.meta_predictor import MetaPredictor

"""Aranges the interactions with the underlying neural network"""

"""TODO: Simple parameter to adjust reduce or normal"""


class RewardEstimator:
    def __init__(self, config, meta_model, cell_type, reward_estimation):
        self.config = config
        self.cell_type = cell_type
        self.meta_model = meta_model

        # Task acuracy estimator
        self.reward_estimation = reward_estimation
        self.max_task_train_steps = config.darts_estimation_steps

        if self.reward_estimation:
            self.meta_predictor = MetaPredictor(config)

            # remove last fully-connected layer
            model = models.resnet18(pretrained=True).eval().to(config.device)
            self.feature_extractor = torch.nn.Sequential(
                *list(model.children())[:-1])
        else:
            # DARTS estimation of the network is used
            self.task_train_steps = 0

            if cell_type == "normal":
                self.w_optim = torch.optim.Adam(
                    self.meta_model.reduce_weights(),
                    lr=self.config.w_lr,
                    betas=(0.0, 0.999),
                    weight_decay=self.config.w_weight_decay,
                )

                self.a_optim = torch.optim.Adam(
                    self.meta_model.reduce_alphas(),
                    self.config.alpha_lr,
                    betas=(0.0, 0.999),
                    weight_decay=self.config.alpha_weight_decay,
                )
            else:
                self.w_optim = torch.optim.Adam(
                    self.meta_model.reduce_weights(),
                    lr=self.config.w_lr,
                    betas=(0.0, 0.999),
                    weight_decay=self.config.w_weight_decay,
                )

                self.a_optim = torch.optim.Adam(
                    self.meta_model.reduce_alphas(),
                    self.config.alpha_lr,
                    betas=(0.0, 0.999),
                    weight_decay=self.config.alpha_weight_decay,
                )

            self.architect = CellArchitect(
                self.meta_model,
                self.config.w_momentum,
                self.config.w_weight_decay,
                self.config.use_first_order_darts,
                cell_type
            )

    def update_meta_model(self, increase, row_idx, edge_idx, op_idx):
        pass

    def decrease_op(self, row_idx, edge_idx, op_idx, prob=0.6):
        pass

    def increase_op(self, row_idx, edge_idx, op_idx, prob=0.6):
        pass

    def get_max_meta_model(self):
        pass
