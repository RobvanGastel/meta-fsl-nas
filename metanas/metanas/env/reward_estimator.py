import torch
import torchvision.models as models

from metanas.task_optimizer.darts import CellArchitect
from metanas.meta_predictor.meta_predictor import MetaPredictor

"""MetaD2A reward estimation"""

"""TODO: Simple parameter to adjust reduce or normal"""


class RewardEstimator:
    def __init__(self, config, meta_model, cell_type, reward_estimation):
        self.config = config
        self.cell_type = cell_type
        self.meta_model = meta_model

        # Task acuracy estimator
        self.reward_estimation = reward_estimation
        self.max_task_train_steps = config.darts_estimation_steps

        self.meta_predictor = MetaPredictor(config)

        # remove last fully-connected layer
        model = models.resnet18(pretrained=True).eval().to(config.device)
        self.feature_extractor = torch.nn.Sequential(
            *list(model.children())[:-1])

    def update_meta_model(self, increase, row_idx, edge_idx, op_idx):
        pass

    def decrease_op(self, row_idx, edge_idx, op_idx, prob=0.6):
        pass

    def increase_op(self, row_idx, edge_idx, op_idx, prob=0.6):
        pass

    def get_max_meta_model(self):
        pass
