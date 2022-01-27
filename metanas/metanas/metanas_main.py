import argparse
import copy
import os
import time
import numpy as np
import pickle
from collections import OrderedDict, namedtuple

import torch.multiprocessing as mp
import torch
import torch.nn as nn

from metanas.meta_optimizer.reptile import NAS_Reptile
from metanas.models.search_cnn import SearchCNNController
from metanas.models.augment_cnn import AugmentCNN
from metanas.models.maml_model import MamlModel
from metanas.task_optimizer.darts import Darts
from metanas.utils import genotypes as gt
from metanas.utils import utils

from metanas.utils.cosine_power_annealing import cosine_power_annealing
from metanas.meta_optimizer.agents.Random.mp_random import RandomAgent
from metanas.meta_optimizer.agents.PPO.mp_rl2_ppo import PPO
from metanas.env.nas_env import NasEnv


""" Script for metanas & baseline trainings
Copyright (c) 2021 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

"""

"""
Based on https://github.com/khanrc/pt.darts
which is licensed under MIT License,
cf. 3rd-party-licenses.txt in root directory.
"""


def meta_architecture_search(
    config, task_optimizer_cls=Darts, meta_optimizer_cls=NAS_Reptile
):
    config.logger.info("Start meta architecture search")

    # Find mistakes in gradient computation
    torch.autograd.set_detect_anomaly(True)

    mp.set_start_method('spawn')

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the
    # best algorithm to use for your hardware. Benchmark mode is good whenever
    # your input sizes for your network do not vary
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.benchmark = True

    # hyperparameter settings
    if config.use_hp_setting:
        config = utils.set_hyperparameter(config)
    else:
        print("Not using hp_setting.")

    # Primitives
    if config.primitives_type == "fewshot":
        config.primitives = gt.PRIMITIVES_FEWSHOT
    elif config.primitives_type == "nasbench201":
        config.primitives = gt.PRIMITIVES_NAS_BENCH_201
    else:
        raise RuntimeError(
            f"This {config.primitives_type} set is not supported.")

    # task distribution
    if config.use_torchmeta_loader:
        from metanas.tasks.torchmeta_loader import (
            OmniglotFewShot,
            MiniImageNetFewShot as miniImageNetFewShot,
            TripleMNISTFewShot,
            OmniPrintFewShot,
            OmniPrintDomainAdaptationFewShot
        )
    else:
        raise RuntimeError("Other data loaders deprecated.")

    if config.dataset == "omniglot":
        task_distribution_class = OmniglotFewShot
    elif config.dataset == "miniimagenet":
        task_distribution_class = miniImageNetFewShot
    elif config.dataset == "triplemnist":
        task_distribution_class = TripleMNISTFewShot
    elif config.dataset == "omniprint":
        task_distribution_class = OmniPrintFewShot
    else:
        raise RuntimeError(f"Dataset {config.dataset} is not supported.")

    if config.use_domain_adaptation and config.dataset == "omniprint":
        if config.source_domain is None and config.target_domain is None:
            raise RuntimeError("Source and/or target domain not defined")

        task_distribution = OmniPrintDomainAdaptationFewShot(
            config, source_domain=config.source_domain,
            target_domain=config.target_domain,
            download=True)
    elif config.use_domain_adaptation:
        raise RuntimeError(
            f"Dataset {config.dataset} is not supported.")
    else:
        task_distribution = task_distribution_class(
            config, download=True)

    # meta model
    normalizer = _init_alpha_normalizer(
        config.normalizer,
        config.task_train_steps,
        config.normalizer_t_max,
        config.normalizer_t_min,
        config.normalizer_temp_anneal_mode,
    )
    meta_model = _build_model(config, task_distribution, normalizer)

    # Disabled share memory for multiprocessing
    # meta_model.share_memory()

    # task & meta optimizer
    config, meta_optimizer = _init_meta_optimizer(
        config, meta_optimizer_cls, meta_model
    )
    config, task_optimizer = _init_task_optimizer(
        config, task_optimizer_cls, meta_model
    )

    # Meta-RL agent
    config = utils.set_rl_hyperparameters(config)
    agent = _init_meta_rl_agent(config, meta_model)

    # load pretrained model
    if config.model_path is not None:
        load_pretrained_model(
            config.model_path, meta_model, task_optimizer, meta_optimizer
        )

    config.logger.info(
        f"alpha initial = {[alpha for alpha in meta_model.alphas()]}")

    utils.print_config_params(config, config.logger.info)

    # meta training
    ###########################################################################

    train_info = dict()  # this is added to the experiment.pickle

    if not config.eval:
        config, meta_model, train_info = train(
            config,
            meta_model,
            task_distribution,
            task_optimizer,
            meta_optimizer,
            agent,
            train_info,
        )

    # meta testing
    ###########################################################################
    config.logger.info(
        f"train steps for evaluation:{ config.test_task_train_steps}")

    # run the evaluation
    config, alpha_logger, sparse_params = evaluate(
        config, meta_model, task_distribution, task_optimizer, agent
    )

    # save results
    experiment = {
        "meta_genotype": meta_model.genotype(),
        "alphas": [alpha for alpha in meta_model.alphas()],
        "final_eval_test_accu": config.top1_logger_test.avg,
        "final_eval_test_loss": config.losses_logger_test.avg,
        "alpha_logger": alpha_logger,
        "sparse_params_logger": sparse_params,
    }
    experiment.update(train_info)

    prefix = config.eval_prefix
    pickle_to_file(experiment, os.path.join(
        config.path, prefix + "experiment.pickle"))
    pickle_to_file(config, os.path.join(config.path, prefix + "config.pickle"))
    config.logger.info("Finished meta architecture search")


def _init_meta_rl_agent(config, meta_model):
    # Dummy environment to set shapes and sizes
    env_normal = NasEnv(
        config, meta_model, test_phase=False, cell_type="normal",
        reward_estimation=config.use_metad2a_estimation,
        max_ep_len=config.env_max_ep_len,
        disable_pairwise_alphas=config.env_disable_pairwise_alphas)

    # If one of the model path is undefined raise error
    if bool(config.agent_model_vars) ^ bool(config.agent_model):
        raise RuntimeError("One of the agent model paths is undefined.")

    if config.agent == "random":
        agent = RandomAgent(config, meta_model,
                            [env_normal],
                            seed=config.seed,
                            steps_per_worker=config.agent_steps_per_trial,
                            logger_kwargs=config.logger_kwargs,
                            is_nas_env=True)
    elif config.agent == "ppo":
        agent = PPO(config, meta_model,
                    [env_normal],
                    logger_kwargs=config.logger_kwargs,
                    seed=config.seed,
                    gamma=config.gamma,
                    lam=config.agent_lambda,
                    lr=config.agent_lr,
                    n_mini_batch=config.agent_n_mini_batch,
                    steps_per_worker=config.agent_steps_per_epoch,
                    epochs=config.agent_epochs_per_trial,
                    hidden_size=config.agent_hidden_size,
                    sequence_length=config.agent_seq_len,
                    exploration_sampling=config.exploration_sampling,
                    use_mask=config.use_agent_mask, is_nas_env=True,
                    model_path={'model': config.agent_model,
                                'vars': config.agent_model_vars})

    else:
        raise ValueError(f"The given agent {config.agent} is not supported.")
    return agent


def evaluate_test_set(config, task, meta_model):
    """Final evaluation over the test set
    """
    # Set max_meta_model weights
    # meta_model.load_state_dict(meta_state)

    # for test data evaluation, turn off drop path
    if config.drop_path_prob > 0.0:
        meta_model.drop_path_prob(0.0)

    # Also, remove skip-connection dropouts during evaluation,
    # evaluation is on the train-test set.
    meta_model.drop_out_skip_connections(0.0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(task.test_loader):
            x_test, y_test = batch
            x_test = x_test.to(config.device, non_blocking=True)
            y_test = y_test.to(config.device, non_blocking=True)

            logits = meta_model(
                x_test, sparsify_input_alphas=True,
                disable_pairwise_alphas=config.env_disable_pairwise_alphas)

            loss = meta_model.criterion(logits, y_test)
            y_test_pred = logits.softmax(dim=1)

            prec1, _ = utils.accuracy(logits, y_test, topk=(1, 5))

    acc = prec1.item()

    # Task info
    w_task = OrderedDict(
        {
            layer_name: copy.deepcopy(layer_weight)
            for layer_name, layer_weight in meta_model.named_weights()
            if layer_weight.grad is not None
        }
    )

    a_task = OrderedDict(
        {
            layer_name: copy.deepcopy(layer_alpha)
            for layer_name, layer_alpha in meta_model.named_alphas()
            if layer_alpha.grad is not None
        }
    )
    genotype = meta_model.genotype()

    task_info = namedtuple(
        "task_info",
        [
            "genotype",
            "top1",
            "w_task",
            "a_task",
            "loss",
            "y_test_pred",
            "sparse_num_params",
        ],
    )
    task_info.w_task = w_task
    task_info.a_task = a_task
    task_info.loss = loss
    task_info.y_test_pred = y_test_pred
    task_info.genotype = genotype
    task_info.top1 = acc

    task_info.sparse_num_params = meta_model.get_sparse_num_params(
        meta_model.alpha_prune_threshold
    )
    return task_info


def meta_rl_optimization(
        config, task, env_normal, env_reduce, agent,
        meta_state, meta_model, meta_epoch, test_phase=False):

    # # Set few-shot task
    env_normal.set_task(task, meta_state, test_phase)

    start = time.time()

    agent.set_task([env_normal])
    start_time, max_meta_state = agent.run_trial()
    env_reduce.set_task(task, max_meta_state, test_phase)

    agent.set_task([env_reduce])
    start_time, max_meta_state = agent.run_trial()

    config.logger.info(
        f"Meta epoch {meta_epoch}, time: {(time.time() - start)/60}")

    if (meta_epoch % config.print_freq == 0) or \
            (meta_epoch == config.meta_epochs) and not test_phase:
        agent_vars = {"steps": agent.total_steps,
                      "test_steps": agent.total_test_steps,
                      "epoch": agent.total_epochs}
        agent.logger.save_state(agent_vars, meta_epoch)

    # Update the meta_model for task-learner or meta update
    # if meta_epoch <= config.warm_up_epochs:
    #     meta_model.load_state_dict(meta_state)
    # else:
    if not config.use_meta_model:
        task_info = evaluate_test_set(
            config, task, meta_model)
    else:
        meta_model.load_state_dict(max_meta_state)
        task_info = evaluate_test_set(
            config, task, meta_model)

    agent.logger.store(TestAcc=task_info.top1)
    # The number of trials = total epochs / epochs per trial
    agent.log_trial(start_time, agent.total_epochs//agent.epochs)

    normalizer = meta_model.normalizer
    config.logger.info("####### ALPHA #######")
    config.logger.info("# Alpha - normal")
    for alpha in meta_model.alpha_normal:
        config.logger.info(meta_model.apply_normalizer(alpha))

    config.logger.info("\n# Alpha - reduce")
    for alpha in meta_model.alpha_reduce:
        config.logger.info(meta_model.apply_normalizer(alpha))
    config.logger.info("#####################")

    # Set task_info to None for metaD2A
    if config.use_metad2a_estimation:
        task_info = None

    return task_info, meta_model


def _init_alpha_normalizer(name, task_train_steps, t_max, t_min,
                           temp_anneal_mode):
    normalizer = dict()
    normalizer["name"] = name
    normalizer["params"] = dict()
    # current step for scheduling normalizer
    normalizer["params"]["curr_step"] = 0.0
    normalizer["params"]["max_steps"] = float(
        task_train_steps
    )  # for scheduling normalizer
    normalizer["params"]["t_max"] = t_max
    normalizer["params"]["t_min"] = t_min
    # temperature annealing
    normalizer["params"]["temp_anneal_mode"] = temp_anneal_mode
    return normalizer


def _build_model(config, task_distribution, normalizer):

    if config.meta_model == "searchcnn":
        meta_model = SearchCNNController(
            task_distribution.n_input_channels,
            config.init_channels,
            task_distribution.n_classes,
            config.layers,
            n_nodes=config.nodes,
            reduction_layers=config.reduction_layers,
            device_ids=config.gpus,
            normalizer=normalizer,
            PRIMITIVES=config.primitives,
            feature_scale_rate=1,
            primitive_space=config.primitives_type,
            weight_regularization=config.darts_regularization,
            dropout_skip_connections=True if config.dropout_skip_connections else False,
            use_hierarchical_alphas=config.use_hierarchical_alphas,
            use_pairwise_input_alphas=config.use_pairwise_input_alphas,
            alpha_prune_threshold=config.alpha_prune_threshold,
        )

    elif config.meta_model == "maml":
        if config.dataset == "omniglot":
            conv_channels = config.init_channels
            final_layer_size = conv_channels * 1 ** 2
        elif config.dataset == "miniimagenet":
            conv_channels = config.init_channels
            final_layer_size = conv_channels * 5 ** 2
        else:
            raise RuntimeError(f"Unknown argument dataset {config.dataset}")

        meta_model = MamlModel(
            task_distribution.n_input_channels,
            conv_channels,
            task_distribution.n_classes,
            final_layer_size,
        )

    elif config.meta_model == "auto_meta":
        # augmented/predefined cell model
        meta_model = AugmentCNN(
            input_size=task_distribution.input_size,
            C_in=task_distribution.n_input_channels,
            C=config.init_channels,
            n_classes=config.k,
            n_layers=config.layers,
            auxiliary=False,
            genotype=gt.genotype_auto_meta,
            stem_multiplier=3,
            feature_scale_rate=1,
            PRIMITIVES=gt.PRIMITIVES_FEWSHOT,
        )

    elif config.meta_model == "metanas_v1":
        # augmented/predefined cell model
        meta_model = AugmentCNN(
            input_size=task_distribution.input_size,
            C_in=task_distribution.n_input_channels,
            C=config.init_channels,
            n_classes=config.k,
            n_layers=config.layers,
            auxiliary=False,
            genotype=gt.genotype_metanas_v1,
            stem_multiplier=3,
            feature_scale_rate=1,
            PRIMITIVES=gt.PRIMITIVES_FEWSHOT,
        )
    elif config.meta_model == "metanas_in_v2":
        # augmented/predefined cell model
        meta_model = AugmentCNN(
            input_size=task_distribution.input_size,
            C_in=task_distribution.n_input_channels,
            C=config.init_channels,
            n_classes=config.k,
            n_layers=config.layers,
            auxiliary=False,
            genotype=gt.genotype_metanas_in_v2,
            stem_multiplier=3,
            feature_scale_rate=1,
            PRIMITIVES=gt.PRIMITIVES_FEWSHOT,
        )
    elif config.meta_model == "metanas_og_v2":
        # augmented/predefined cell model
        meta_model = AugmentCNN(
            input_size=task_distribution.input_size,
            C_in=task_distribution.n_input_channels,
            C=config.init_channels,
            n_classes=config.k,
            n_layers=config.layers,
            auxiliary=False,
            genotype=gt.genotype_metanas_og_v2,
            stem_multiplier=3,
            feature_scale_rate=1,
            PRIMITIVES=gt.PRIMITIVES_FEWSHOT,
        )
    else:
        raise RuntimeError(f"Unknown meta_model {config.meta_model}")
    return meta_model.to(config.device)


def load_pretrained_model(
    model_path, meta_model, task_optimizer=None, meta_optimizer=None
):
    model_path, name = os.path.split(model_path)
    assert name, "Specify the full path for argument 'model_path'."
    print(f"Loading pretrained model from {model_path}")
    utils.load_state(
        meta_model,
        meta_optimizer,
        task_optimizer,
        model_path,
        filename=name,
    )


def _init_meta_optimizer(config, meta_optimizer_class, meta_model):
    if meta_optimizer_class == NAS_Reptile:
        # reptile uses SGD as meta optim
        config.w_meta_optim = torch.optim.SGD(
            meta_model.weights(), lr=config.w_meta_lr)

        if meta_model.alphas() is not None:
            config.a_meta_optim = torch.optim.SGD(
                meta_model.alphas(), lr=config.a_meta_lr
            )
        else:
            config.a_meta_optim = None
    else:
        raise RuntimeError(
            f"Meta-Optimizer {meta_optimizer_class} is not yet supported."
        )
    meta_optimizer = meta_optimizer_class(meta_model, config)
    return config, meta_optimizer


def _init_task_optimizer(config, task_optimizer_class, meta_model):
    return config, task_optimizer_class(meta_model, config)


def _get_meta_lr_scheduler(config, meta_optimizer):
    if config.w_meta_anneal:
        w_meta_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            meta_optimizer.w_meta_optim, config.meta_epochs, eta_min=0.0
        )

        if w_meta_lr_scheduler.last_epoch == -1:
            w_meta_lr_scheduler.step()
    else:
        w_meta_lr_scheduler = None

    if config.a_meta_anneal and config.a_meta_optim is not None:
        a_meta_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            meta_optimizer.a_meta_optim,
            (config.meta_epochs - config.warm_up_epochs),
            eta_min=0.0,
        )

        if a_meta_lr_scheduler.last_epoch == -1:
            a_meta_lr_scheduler.step()
    else:
        a_meta_lr_scheduler = None

    return w_meta_lr_scheduler, a_meta_lr_scheduler


def _prune_alphas(meta_model, meta_model_prune_threshold):
    """Remove operations with alphas of that are below threshold from meta
    model (indirectly)

    Currently this only has an effect for the :class:`SearchCNNController`
    meta_model

    Args:
        meta_model: The meta_model
        meta_model_prune_threshold: threshold for pruning
    """
    if isinstance(meta_model, SearchCNNController):
        meta_model.prune_alphas(prune_threshold=meta_model_prune_threshold)


def train(
    config,
    meta_model,
    task_distribution,
    task_optimizer,
    meta_optimizer,
    agent,
    train_info=None
):
    """Meta-training loop

    Args:
        config: Training configuration parameters
        meta_model: The meta_model
        task_distribution: Task distribution object
        task_optimizer: A pytorch optimizer for task training
        meta_optimizer: A pytorch optimizer for meta training
        normalizer: To be able to reinit the task optimizer for staging
        agent: A Meta-RL agent for meta training
        train_info: Dictionary that is added to the experiment.pickle
            file in addition to training internal data.

    Returns:
        A tuple containing the updated config, meta_model and updated
        train_info.
    """
    if train_info is None:
        train_info = dict()
    else:
        assert isinstance(train_info, dict)

    # add training performance to train_info
    train_test_loss = list()
    train_test_accu = list()
    test_test_loss = list()
    test_test_accu = list()
    train_info["train_test_loss"] = train_test_loss
    train_info["train_test_accu"] = train_test_accu
    train_info["test_test_loss"] = test_test_loss
    train_info["test_test_accu"] = test_test_accu

    # time averages for logging (are reset during evaluation)
    io_time = utils.AverageMeter()
    sample_time = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    total_time = utils.AverageMeter()

    # performance logger
    config.top1_logger = utils.AverageMeter()
    config.top1_logger_test = utils.AverageMeter()
    config.losses_logger = utils.AverageMeter()
    config.losses_logger_test = utils.AverageMeter()

    # meta lr annealing
    if config.use_cosine_power_annealing:
        w_meta_lr_power_schedule = cosine_power_annealing(
            epochs=config.meta_epochs, max_lr=config.w_meta_lr,
            min_lr=1e-8, exponent_order=2,  # min_lr adjustable
            max_epoch=config.meta_epochs,
            warmup_epochs=config.warm_up_epochs
        )
        a_meta_lr_power_schedule = cosine_power_annealing(
            epochs=config.meta_epochs, max_lr=config.a_meta_lr,
            min_lr=1e-8, exponent_order=2,
            max_epoch=config.meta_epochs,
            warmup_epochs=config.warm_up_epochs
        )
    else:
        w_meta_lr_scheduler, a_meta_lr_scheduler = _get_meta_lr_scheduler(
            config, meta_optimizer
        )

    # Environment to learn reduction and normal cell
    env_normal = NasEnv(
        config, meta_model, test_phase=False, cell_type="normal",
        reward_estimation=config.use_metad2a_estimation,
        max_ep_len=config.env_max_ep_len,
        disable_pairwise_alphas=config.env_disable_pairwise_alphas)
    env_reduce = NasEnv(
        config, meta_model, test_phase=False, cell_type="reduce",
        reward_estimation=config.use_metad2a_estimation,
        max_ep_len=config.env_max_ep_len,
        disable_pairwise_alphas=config.env_disable_pairwise_alphas)

    for meta_epoch in range(config.start_epoch, config.meta_epochs + 1):

        if config.use_cosine_power_annealing:
            if (meta_epoch >= config.warm_up_epochs):
                for param_group in config.w_meta_optim.param_groups:
                    param_group['lr'] = w_meta_lr_power_schedule[meta_epoch-1]

            for param_group in config.a_meta_optim.param_groups:
                param_group['lr'] = a_meta_lr_power_schedule[meta_epoch-1]

        time_es = time.time()
        meta_train_batch = task_distribution.sample_meta_train()
        time_samp = time.time()

        sample_time.update(time_samp - time_es)

        # Each task starts with the current meta state
        meta_state = copy.deepcopy(meta_model.state_dict())
        global_progress = f"[Meta-Epoch {meta_epoch:2d}/{config.meta_epochs}]"
        task_infos = []

        time_bs = time.time()
        for task in meta_train_batch:

            # Meta-RL optimization
            task_info, meta_model = meta_rl_optimization(
                config, task, env_normal, env_reduce, agent,
                meta_state, meta_model, meta_epoch, test_phase=False)

            # if task_info is not None:
            #     # Use information of the training during the meta-RL loop.
            #     task_infos += [task_info]

            #     config.top1_logger.update(task_info.top1, 1)
            #     config.losses_logger.update(task_info.loss, 1)
            # else:
            # Train task-learner with max alphas from the meta-RL loop,
            # on metaD2A reward estimation.

            task_info = task_optimizer.step(
                task, epoch=meta_epoch,
                global_progress=global_progress
            )

            config.logger.info(
                f"Training accuracy: {task_info.top1}, loss: {task_info.loss}")

            task_infos += [task_info]

            meta_model.load_state_dict(meta_state)

        time_be = time.time()
        batch_time.update(time_be - time_bs)

        train_test_loss.append(config.losses_logger.avg)
        train_test_accu.append(config.top1_logger.avg)

        # do a meta update
        meta_optimizer.step(task_infos)

        # update meta LR
        if not config.use_cosine_power_annealing:
            if (a_meta_lr_scheduler is not None) and \
                    (meta_epoch >= config.warm_up_epochs):
                a_meta_lr_scheduler.step()

            if w_meta_lr_scheduler is not None:
                w_meta_lr_scheduler.step()

        time_ee = time.time()
        total_time.update(time_ee - time_es)

        if meta_epoch % config.print_freq == 0:
            config.logger.info(
                f"Train: [{meta_epoch:2d}/{config.meta_epochs}] "
                f"Time (sample, batch, sp_io, total): {sample_time.avg:.2f},"
                f"{batch_time.avg:.2f}, "
                f"{io_time.avg:.2f}, {total_time.avg:.2f} "
                f"Train-TestLoss {config.losses_logger.avg:.3f} "
                f"Train-TestPrec@(1,) ({config.top1_logger.avg:.1%}, {1.00:.1%})"
            )

        # meta testing every config.eval_freq epochs
        if meta_epoch % config.eval_freq == 0:
            meta_test_batch = task_distribution.sample_meta_test()

            # Each task starts with the current meta state
            meta_state = copy.deepcopy(meta_model.state_dict())

            # copy also the optimizer states
            meta_optims_state = [
                copy.deepcopy(meta_optimizer.w_meta_optim.state_dict()),
                copy.deepcopy(meta_optimizer.a_meta_optim.state_dict()),
                copy.deepcopy(task_optimizer.w_optim.state_dict()),
                copy.deepcopy(task_optimizer.a_optim.state_dict()),
            ]

            global_progress = f"[Meta-Epoch {meta_epoch:2d}/{config.meta_epochs}]"

            task_infos = []
            for task in meta_test_batch:

                # Meta-RL optimization
                task_info, meta_model = meta_rl_optimization(
                    config, task, env_normal, env_reduce, agent,
                    meta_state, meta_model, meta_epoch, test_phase=True)

                # Train task-learner with max alphas from the meta-RL loop,
                # on metaD2A reward estimation.
                if task_info is not None:
                    task_infos += [task_info]

                    config.top1_logger_test.update(task_info.top1, 1)
                    config.losses_logger_test.update(task_info.loss, 1)

                else:
                    task_infos += [
                        task_optimizer.step(
                            task,
                            epoch=meta_epoch,
                            global_progress=global_progress,
                            test_phase=True,
                            num_of_skip_connections=config.limit_skip_connections,
                        )
                    ]
                meta_model.load_state_dict(meta_state)

            config.logger.info(
                f"Train: [{meta_epoch:2d}/{config.meta_epochs}] "
                f"Test-TestLoss {config.losses_logger_test.avg:.3f} "
                "Test-TestPrec@(1,) "
                f"({config.top1_logger_test.avg:.1%}, {1.00:.1%})"
            )

            test_test_loss.append(config.losses_logger_test.avg)
            test_test_accu.append(config.top1_logger_test.avg)

            # print cells
            config.logger.info(f"genotype = {task_infos[0].genotype}")
            config.logger.info(
                f"alpha vals = {[a for a in meta_model.alphas()]}")

            # reset the states so that meta training doesnt see
            # meta-testing
            meta_optimizer.w_meta_optim.load_state_dict(
                meta_optims_state[0])
            meta_optimizer.a_meta_optim.load_state_dict(
                meta_optims_state[1])
            task_optimizer.w_optim.load_state_dict(meta_optims_state[2])
            task_optimizer.a_optim.load_state_dict(meta_optims_state[3])

            print(meta_model.genotype())
            # save checkpoint
            experiment = {
                "genotype": [task_info.genotype for task_info in task_infos],
                "meta_genotype": meta_model.genotype(),
                "alphas": [alpha for alpha in meta_model.alphas()],
            }
            experiment.update(train_info)
            pickle_to_file(experiment, os.path.join(
                config.path, "experiment.pickle"))

            utils.save_state(
                meta_model,
                meta_optimizer,
                task_optimizer,
                config.path,
                meta_epoch,
                job_id=config.job_id,
            )

            # reset time averages during testing
            sample_time.reset()
            batch_time.reset()
            total_time.reset()
            io_time.reset()

            # prune alpha values in meta model every config.eval_freq
            # epochs
            _prune_alphas(
                meta_model,
                meta_model_prune_threshold=config.meta_model_prune_threshold
            )

    # end of meta train
    utils.save_state(
        meta_model, meta_optimizer, task_optimizer, config.path,
        job_id=config.job_id
    )

    print(meta_model.genotype())
    experiment = {
        "meta_genotype": meta_model.genotype(),
        "alphas": [alpha for alpha in meta_model.alphas()],
    }
    experiment.update(train_info)
    pickle_to_file(experiment, os.path.join(config.path, "experiment.pickle"))
    pickle_to_file(config, os.path.join(config.path, "config.pickle"))

    return config, meta_model, train_info


def pickle_to_file(var, file_path):
    """Save a single variable to a file using pickle"""
    with open(file_path, "wb") as handle:
        pickle.dump(var, handle)


def evaluate(config, meta_model, task_distribution, task_optimizer, agent):
    """Meta-testing

    Returns:
        A tuple consisting of (config, alpha_logger). The config
        contains the fields `top1_logger_test` with the average
        top1 accuracy and `losses_logger_test` with the average
        loss during meta test test. The alpha logger contains
        lists of architecture alpha parameters.
    """
    # Each task starts with the current meta state, make a backup
    meta_state = copy.deepcopy(meta_model.state_dict())

    # copy also the task optimizer states
    meta_optims_state = [
        copy.deepcopy(task_optimizer.w_optim.state_dict()),
        copy.deepcopy(task_optimizer.a_optim.state_dict()),
    ]

    top1_test = utils.AverageMeter()
    losses_test = utils.AverageMeter()
    config.top1_logger_test = top1_test
    config.losses_logger_test = losses_test
    paramas_logger = list()

    if config.meta_model == "searchcnn":
        alpha_logger = OrderedDict()
        alpha_logger["normal_relaxed"] = list()
        alpha_logger["reduced_relaxed"] = list()
        alpha_logger["genotype"] = list()
        alpha_logger["all_alphas"] = list()
        alpha_logger["normal_hierarchical"] = list()
        alpha_logger["reduced_hierarchical"] = list()
        alpha_logger["normal_pairwise"] = list()
        alpha_logger["reduced_pairwise"] = list()
    else:
        alpha_logger = None

    # Environment to learn reduction and normal cell
    env_normal = NasEnv(
        config, meta_model, test_phase=True, cell_type="normal",
        reward_estimation=config.use_metad2a_estimation,
        max_ep_len=config.env_max_ep_len,
        disable_pairwise_alphas=config.env_disable_pairwise_alphas)

    env_reduce = NasEnv(
        config, meta_model, test_phase=True, cell_type="reduce",
        reward_estimation=config.use_metad2a_estimation,
        max_ep_len=config.env_max_ep_len,
        disable_pairwise_alphas=config.env_disable_pairwise_alphas)

    for eval_epoch in range(config.eval_epochs):

        meta_test_batch = task_distribution.sample_meta_test()
        global_progress = f"[Eval-Epoch {eval_epoch:2d}/{config.eval_epochs}]"
        task_infos = []

        for task in meta_test_batch:

            # Meta-RL optimization
            task_info, meta_model = meta_rl_optimization(
                config, task, env_normal, env_reduce, agent,
                meta_state, config.meta_epochs, eval_epoch, test_phase=False)

            if task_info is not None:
                # Use information of the training during the meta-RL loop.
                task_infos += [task_info]

                config.top1_logger_test.update(task_info.top1, 1)
                config.losses_logger_test.update(task_info.loss, 1)
            else:
                # Train task-learner with max alphas from the meta-RL loop,
                # on metaD2A reward estimation.
                task_infos += [
                    task_optimizer.step(
                        task,
                        epoch=config.meta_epochs,
                        global_progress=global_progress,
                        test_phase=True,
                        alpha_logger=alpha_logger,
                        sparsify_input_alphas=config.sparsify_input_alphas,
                        num_of_skip_connections=config.limit_skip_connections
                    )
                ]

            # load meta state
            meta_model.load_state_dict(meta_state)

            task_optimizer.w_optim.load_state_dict(meta_optims_state[0])
            task_optimizer.a_optim.load_state_dict(meta_optims_state[1])

            if isinstance(meta_model, SearchCNNController):
                paramas_logger.append(task_infos[-1].sparse_num_params)
            else:
                paramas_logger.append(utils.count_params(meta_model))

        prefix = f" (prefix: {config.eval_prefix})" if config.eval_prefix else ""

        config.logger.info(
            f"Test data evaluation{prefix}:: "
            f"[{eval_epoch:2d}/{config.eval_epochs}] "
            f"Test-TestLoss {config.losses_logger_test.avg:.3f} "
            f"Test-TestPrec@(1,) ({config.top1_logger_test.avg:.1%}, "
            f"{1.00:.1%})"
            f" \n Sparse_num_params (mean, min, max): {np.mean(paramas_logger)}, "
            f"{np.min(paramas_logger)}, {np.max(paramas_logger)}"
        )

    return config, alpha_logger, paramas_logger


def _str_or_none(x):
    """Convert multiple possible input strings to None"""
    return None if (x is None or not x or x.capitalize() == "None") else x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Search Config", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Execution
    parser.add_argument("--name", required=True)
    parser.add_argument("--job_id", default=None, type=_str_or_none)
    parser.add_argument("--path", default="/home/elt4hi/")
    parser.add_argument("--data_path")
    parser.add_argument("--seed", type=int, default=2, help="random seed")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation only")
    parser.add_argument(
        "--eval_prefix",
        type=str,
        default="",
        help="Prefix added to all output files during evaluation",
    )

    # only for hp search
    parser.add_argument(
        "--hp_setting", type=str, default="in",
        help="use predefined HP configuration"
    )
    parser.add_argument("--use_hp_setting", type=int, default=0)

    ################################

    # Dataset parameters
    parser.add_argument("--workers", type=int, default=4, help="# of workers")
    parser.add_argument("--print_freq", type=int,
                        default=50, help="print frequency")
    parser.add_argument(
        "--use_torchmeta_loader",
        action="store_true",
        help="Use torchmeta for data loading.",
    )
    parser.add_argument(
        "--dataset", default="omniglot",
        help="omniglot / miniimagenet / triplemnist / omniprint")

    parser.add_argument(
        "--use_domain_adaptation",
        action="store_true",
        help="Perform domain adaptation experiments"
    )

    parser.add_argument(
        "--print_split",
        default="meta1",
        help="For the specific omniprint split"
    )

    parser.add_argument(
        "--source_domain",
        default=None, help="meta1 / meta2 / meta3 / meta4 / meta5"
    )

    parser.add_argument(
        "--target_domain",
        default=None, help="meta1 / meta2 / meta3 / meta4 / meta5"
    )

    parser.add_argument(
        "--use_vinyals_split",
        action="store_true",
        help="Only relevant for Omniglot: Use the vinyals split. Requires the "
        "torchmeta data loading.",
    )

    parser.add_argument(
        "--gpus",
        default="0",
        help="gpu device ids separated by comma. " "`all` indicates use all"
        " gpus.",
    )

    # Meta Learning
    parser.add_argument(
        "--meta_model", type=str, default="searchcnn", help="meta model to use"
    )
    parser.add_argument("--model_path", default=None,
                        help="load model from path")

    parser.add_argument(
        "--meta_epochs", type=int, default=10, help="Number meta train epochs"
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=1,
        help="Start training at a specific epoch (for resuming training from"
        " a checkpoint)",
    )
    parser.add_argument(
        "--meta_batch_size", type=int, default=5,
        help="Number of tasks in a meta batch"
    )
    parser.add_argument(
        "--test_meta_batch_size",
        type=int,
        default=25,
        help="Number of tasks in a test meta batch",
    )
    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=100,
        help="Number of epochs for final evaluation of test data",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1000,
        help="how often to run meta-testing for intermediate evaluation "
        "(in epochs)",
    )

    parser.add_argument(
        "--task_train_steps",
        type=int,
        default=5,
        help="Number of training steps per task",
    )
    parser.add_argument(
        "--test_task_train_steps",
        type=int,
        default=50,
        help="Number of training steps per task",
    )

    parser.add_argument(
        "--warm_up_epochs",
        type=int,
        default=1e6,
        help="warm up epochs before architecture search is enabled",
    )

    parser.add_argument(
        "--test_adapt_steps",
        type=float,
        default=1.0,
        help="for how many test-train steps should architectue be adapted "
        "(relative to test_train_steps)?",
    )

    parser.add_argument(
        "--w_meta_optim", default=None, help="Meta optimizer of weights"
    )
    parser.add_argument(
        "--w_meta_lr", type=float, default=0.001, help="meta lr for weights"
    )
    parser.add_argument(
        "--w_meta_anneal", type=int, default=1,
        help="Anneal Meta weights optimizer LR"
    )
    parser.add_argument(
        "--w_task_anneal", type=int, default=0,
        help="Anneal task weights optimizer LR"
    )
    parser.add_argument("--a_meta_optim", default=None,
                        help="Meta optimizer of alphas")
    parser.add_argument(
        "--a_meta_lr", type=float, default=0.001, help="meta lr for alphas"
    )
    parser.add_argument(
        "--a_meta_anneal",
        type=int,
        default=1,
        help="Anneal Meta architecture optimizer LR",
    )
    parser.add_argument(
        "--a_task_anneal",
        type=int,
        default=0,
        help="Anneal task architecture optimizer LR",
    )
    parser.add_argument(
        "--normalizer",
        type=str,
        default="softmax",
        help="Alpha normalizer",
        choices=["softmax", "relusoftmax", "gumbel_softmax"],
    )
    parser.add_argument(
        "--normalizer_temp_anneal_mode",
        type=str,
        default=None,
        help="Temperature anneal mode (if applicable to normalizer)",
    )
    parser.add_argument(
        "--normalizer_t_max", type=float, default=5.0,
        help="Initial temperature"
    )
    parser.add_argument(
        "--normalizer_t_min",
        type=float,
        default=0.1,
        help="Final temperature after task_train_steps",
    )

    # Few Shot Learning
    parser.add_argument(
        "--n",
        default=1,
        type=int,
        help="Training examples per class / support set (for meta testing).",
    )

    parser.add_argument(
        "--n_train",
        default=15,
        type=int,
        help="Training examples per class for meta training.",
    )

    parser.add_argument("--k", default=5, type=int, help="Number of classes.")
    parser.add_argument(
        "--q", default=1, type=int, help="Test examples per class / query set"
    )

    # Weights
    parser.add_argument("--batch_size", type=int,
                        default=64, help="batch size")
    parser.add_argument("--w_lr", type=float,
                        default=0.025, help="lr for weights")
    parser.add_argument(
        "--w_momentum", type=float, default=0.0, help="momentum for weights"
    )
    parser.add_argument(
        "--w_weight_decay", type=float, default=0.0,
        help="weight decay for weights"
    )
    parser.add_argument(
        "--w_grad_clip", type=float, default=10e5,
        help="gradient clipping for weights"
    )

    parser.add_argument(
        "--drop_path_prob", type=float, default=0.0,
        help="drop path probability"
    )
    parser.add_argument(
        "--use_drop_path_in_meta_testing",
        action="store_true",
        help="Whether to use drop path also during meta testing.",
    )

    # P-DARTS, SharpDARTS, TSE-DARTS configurations
    # Enabling both approaches, specificly for ablation study
    parser.add_argument(
        "--dropout_skip_connections",
        action="store_true",
        help="Use dropouts on skip-connections",
    )

    parser.add_argument(
        "--use_limit_skip_connections",
        action="store_true",
        help="Change skip-connections to M in final gene"
    )

    # Discovered cells are allowed to keep M = 2, skip connections.
    parser.add_argument("--limit_skip_connections", type=int, default=2)

    # Regularize the weights based on maximum weight of the alphas
    parser.add_argument("--darts_regularization", default="scalar",
                        help="Either scalar or max_w")

    parser.add_argument("--use_cosine_power_annealing", action="store_true")

    parser.add_argument("--use_tse_darts", action="store_true",
                        help="Training Speed Estimation (TSE)")

    # Architectures
    parser.add_argument("--primitives_type", default="fewshot",
                        help="Either fewshot, nasbench201")

    parser.add_argument("--init_channels", type=int, default=16)

    parser.add_argument("--layers", type=int, default=5,
                        help="# of layers (cells)")

    parser.add_argument("--nodes", type=int, default=3,
                        help="# of nodes per cell")

    parser.add_argument(
        "--use_hierarchical_alphas",
        action="store_true",
        help="Whether to use hierarhical alphas in search_cnn model.",
    )
    parser.add_argument(
        "--use_pairwise_input_alphas",
        action="store_true",
        help="Whether to use alphas on pairwise inputs in search_cnn model.",
    )
    parser.add_argument(
        "--reduction_layers",
        nargs="+",
        default=[],
        type=int,
        help="Where to use reduction cell",
    )
    parser.add_argument("--alpha_lr", type=float,
                        default=3e-4, help="lr for alpha")
    parser.add_argument(
        "--alpha_prune_threshold",
        type=float,
        default=0.0,
        help="During forward pass, alphas below the threshold probability are "
        "pruned (meaning the respective operations are not executed anymore).",
    )
    parser.add_argument(
        "--meta_model_prune_threshold",
        type=float,
        default=0.0,
        help="During meta training, prune alphas from meta model "
        "below this threshold to not train them any longer.",
    )

    parser.add_argument(
        "--alpha_weight_decay", type=float, default=0.001,
        help="weight decay for alpha"
    )
    parser.add_argument(
        "--anneal_softmax_temperature",
        action="store_true",
        help="anneal temperature of softmax",
    )
    parser.add_argument(
        "--do_unrolled_architecture_steps",
        action="store_true",
        help="do one step in w before computing grad of alpha",
    )

    parser.add_argument(
        "--use_first_order_darts",
        action="store_true",
        help="Whether to use first order DARTS.",
    )

    parser.add_argument(
        "--sparsify_input_alphas",
        type=float,
        default=None,
        help="sparsify_input_alphas input for the search_cnn forward pass "
        "during final evaluation.",
    )  # deprecated

    # Meta-RL agent settings
    parser.add_argument("--agent", default="random",
                        help="random / ppo")

    parser.add_argument("--agent_model", default=None,
                        type=str, help="Path to pretrained model")

    parser.add_argument("--agent_model_vars", default=None,
                        type=str)

    parser.add_argument("--agent_exploration", action="store_true")

    parser.add_argument("--agent_hidden_size",
                        type=int, default=None)

    parser.add_argument("--agent_use_mask", action="store_true")

    # Environment settings
    parser.add_argument("--darts_estimation_steps",
                        type=int, default=7)

    parser.add_argument("--tse_steps", type=int, default=2)

    parser.add_argument("--use_env_random_start",
                        action="store_true")

    parser.add_argument("--env_encourage_exploration",
                        action="store_true")

    parser.add_argument("--env_min_rew", type=float, default=None)
    parser.add_argument("--env_max_rew", type=float, default=None)

    parser.add_argument("--use_meta_model",
                        action="store_true")

    parser.add_argument("--env_update_weights_and_alphas",
                        action="store_true")

    parser.add_argument("--env_disable_pairwise_alphas",
                        action="store_true")

    # MetaD2A reward estimation settings
    parser.add_argument(
        "--rew_model_path", default='/home/rob/Git/meta-fsl-nas/')
    parser.add_argument(
        "--rew_data_path",
        default='/home/rob/Git/meta-fsl-nas/')

    parser.add_argument(
        "--use_metad2a_estimation",
        action="store_true",
        help="Use meta_predictor for the env reward estimation")

    args = parser.parse_args()

    # Fixed MetaD2A variables
    # num_samples in few-shot setting, n*k
    args.num_samples = 20
    # the graph data used, nas-bench-201
    args.graph_data_name = 'nasbench201'
    args.nvt = 7
    args.hs = 512
    args.nz = 56

    args.path = os.path.join(
        args.path, ""
    )  # add file separator at end if it does not exist
    args.plot_path = os.path.join(args.path, "plots")

    # Setup data and hardware config
    args.gpus = utils.parse_gpus(args.gpus)
    args.device = torch.device("cuda")

    # Logging
    logger = utils.get_logger(os.path.join(args.path, f"{args.name}.log"))
    args.logger = logger

    meta_architecture_search(args)
