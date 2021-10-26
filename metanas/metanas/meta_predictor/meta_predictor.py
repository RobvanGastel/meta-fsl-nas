###############################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from
# Datasets, ICLR 2021
###############################################################################

import os

import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
from torch import optim

from torch.utils.tensorboard import SummaryWriter
from metanas.meta_predictor.predictor.model import PredictorModel
from metanas.meta_predictor.utils import (load_graph_config,
                                          load_pretrained_model,
                                          save_model)
from metanas.meta_predictor.loader import get_meta_train_loader


class MetaPredictor:
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.num_samples = config.num_samples

        self.logger = config.logger
        self.meta_test = config.use_rew_estimation
        self.max_corr_dict = {'corr': 0, 'epoch': 0, 'loss': -1}

        # Test when used as discrete estimate on the RL environment
        # TODO: Stoch or discrete sampling
        if self.meta_test:
            self.model_path = config.rew_model_path
            self.data_path = config.rew_data_path

            # NAS_bench 201 graph configuration
            graph_config = load_graph_config(
                config.graph_data_name, config.nvt, self.data_path)

            # Load predictor model
            self.model = PredictorModel(config, graph_config).to(self.device)

            # If model path is given, load pretrained model
            if self.model_path is not None:
                load_pretrained_model(self.model_path, self.model)

        else:
            # NAS_bench 201 graph configuration
            graph_config = load_graph_config(
                config.graph_data_name, config.nvt, config.data_path)

            # Load predictor model
            self.model = PredictorModel(config, graph_config).to(self.device)

            # TODO: Fix model path set
            # If model path is given, load pretrained model
            if config.model_path is not None:
                load_pretrained_model(self.model_path, self.model)

            self.epochs = config.epochs
            self.batch_size = config.batch_size
            self.save_epoch = config.save_epoch

            self.model_path = config.model_path
            self.data_path = config.data_path
            self.save_path = config.save_path

            self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min',
                factor=0.1,
                patience=10,
                verbose=True)

            self.mtrloader = get_meta_train_loader(
                self.batch_size,
                self.data_path,
                self.num_samples,
                self.device,
                is_pred=True)

            # During meta-training log to tensorboard
            log_dir = os.path.join(self.save_path, "log")
            self.summary_writer = SummaryWriter(
                log_dir=log_dir, flush_secs=1)

            # Used to normalize the accuracy
            # self.acc_mean = self.mtrloader.dataset.mean
            # self.acc_std = self.mtrloader.dataset.std

    def forward(self, dataset, arch):
        D_mu = self.model.set_encode(dataset.to(self.device))
        G_mu = self.model.graph_encode(arch)

        return self.model.predict(D_mu, G_mu)

    def meta_train(self):
        for epoch in range(1, self.epochs+1):
            loss, corr = self.meta_train_epoch()
            self.scheduler.step(loss)

            # self.config.logger print pred log train
            self.logger.info(
                f"Train epoch:{epoch:3d} \n"
                f"meta-training loss: {loss:.6f} "
                f"meta-training correlation: {corr:.4f}"
            )

            # Logging to tensorboard
            self.summary_writer.add_scalar(
                'meta-training/loss', loss, epoch)
            self.summary_writer.add_scalar(
                'meta-training/corr', corr, epoch)

            valoss, vacorr = self.meta_validation()
            if self.max_corr_dict['corr'] < vacorr:
                self.max_corr_dict['corr'] = vacorr
                self.max_corr_dict['epoch'] = epoch
                self.max_corr_dict['loss'] = valoss
                save_model(self.save_path, self.model, epoch, max_corr=True)

            # self.config.logger print pred log validation
            max_loss = self.max_corr_dict['loss']
            max_corr = self.max_corr_dict['corr']
            self.logger.info(
                f"Validation epoch:{epoch:3d} \n"
                f"validation loss: {valoss:.6f} ({max_loss:.6f}) "
                f"max validation correlation: {vacorr:.4f} ({max_corr:.4f})"
            )

            # Logging to tensorboard
            self.summary_writer.add_scalar(
                'meta-validation/loss', valoss, epoch)
            self.summary_writer.add_scalar(
                'meta-validation/corr', vacorr, epoch)

            if epoch % self.save_epoch == 0:
                save_model(self.save_path, self.model, epoch)

    def meta_train_epoch(self):
        self.model.to(self.device)
        self.model.train()
        self.mtrloader.dataset.set_mode('train')
        N = len(self.mtrloader.dataset)

        trloss = 0
        y_all, y_pred_all = [], []

        for x, g, acc in tqdm(self.mtrloader):

            self.optimizer.zero_grad()
            y_pred = self.forward(x, g)
            y = acc.to(self.device)
            loss = self.model.mseloss(y_pred, y.unsqueeze(-1))
            loss.backward()
            self.optimizer.step()

            y = y.tolist()
            y_pred = y_pred.squeeze().tolist()
            y_all += y
            y_pred_all += y_pred
            trloss += float(loss)

        return trloss/N, pearsonr(np.array(y_all),
                                  np.array(y_pred_all))[0]

    def meta_validation(self):
        self.model.eval()
        self.mtrloader.dataset.set_mode('valid')

        valoss = 0
        N = len(self.mtrloader.dataset)
        y_all, y_pred_all = [], []

        with torch.no_grad():
            for x, g, acc in tqdm(self.mtrloader):
                y_pred = self.forward(x, g)
                y = acc.to(self.device)
                loss = self.model.mseloss(y_pred, y.unsqueeze(-1))

                y = y.tolist()
                y_pred = y_pred.squeeze().tolist()
                y_all += y
                y_pred_all += y_pred
                valoss += float(loss)

        return valoss/N, pearsonr(np.array(y_all),
                                  np.array(y_pred_all))[0]

    def evaluate_architecture(self, dataset, architecture):
        """Meta-training evaluation for the RL environment
        """
        dataset = [dataset.to(self.device)]
        architecture = [architecture]

        self.model.eval()

        with torch.no_grad():
            D = self.model.set_encode(dataset).unsqueeze(0)
            G = self.model.graph_encode(architecture)
            y_pred = self.model.predict(D, G)

        return y_pred
