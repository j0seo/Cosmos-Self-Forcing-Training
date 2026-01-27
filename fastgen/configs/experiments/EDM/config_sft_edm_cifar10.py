# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import CIFAR10_Loader_Config
from fastgen.configs.net import EDM_CIFAR10_Config, CKPT_ROOT_DIR
from fastgen.utils import LazyCall as L
from fastgen.configs.callbacks import EMA_POWER_CALLBACKS
from fastgen.datasets.augment import AugmentPipe

"""Configs for SFT (Supervised Fine-Tuning) on EDM and CIFAR-10 dataset."""


def create_config():
    config = config_sft_default.create_config()

    # model
    # EDM works on raw pixel space: [C, H, W] = [3, 32, 32]
    config.model.input_shape = [3, 32, 32]

    # EDM uses sigma-based noise schedule (log-normal for training)
    config.model.sample_t_cfg.time_dist_type = "lognormal"
    config.model.sample_t_cfg.train_p_mean = -1.2
    config.model.sample_t_cfg.train_p_std = 1.2

    # Pretrained model path
    # EDM-CIFAR10 checkpoint from Karras et al. (converted to .pth state dict)
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cifar10/edm-cifar10-32x32-cond-vp.pth"

    # Network config
    config.model.net = EDM_CIFAR10_Config
    config.model.net.dropout = 0.13

    # Optimizer config (lower LR for fine-tuning; 2e-4 is for training from scratch)
    config.model.net_optimizer.optim_type = "adam"
    config.model.net_optimizer.lr = 2e-5  # 10x lower for fine-tuning
    config.model.net_optimizer.betas = (0.9, 0.999)
    config.model.net_optimizer.weight_decay = 0.0

    # EMA config
    config.model.use_ema = ["ema_1", "ema_5", "ema_10"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_POWER_CALLBACKS)

    # Data augmentation (following EDM paper)
    # The augment_dim=9 comes from: scale(1) + rotate_frac(2) + brightness(1) + contrast(1) + lumaflip(1) + hue(2) + saturation(1)
    config.trainer.augment_pipe = L(AugmentPipe)(
        p=0.12,  # Overall augmentation probability
        # Geometric transforms (3 dims total)
        scale=1,
        scale_std=0.2,
        rotate_frac=1,
        rotate_frac_max=1,
        # Color transforms (6 dims total)
        brightness=1,
        brightness_std=0.2,
        contrast=1,
        contrast_std=0.5,
        lumaflip=1,
        hue=1,
        hue_max=1,
        saturation=1,
        saturation_std=1,
    )
    config.model.net.augment_dim = 9

    # Dataloader
    config.dataloader_train = CIFAR10_Loader_Config
    config.dataloader_train.xflip = True

    # Sampling config for visualization
    config.model.student_sample_steps = 18  # EDM typically uses 18-50 steps

    # Trainer
    config.trainer.batch_size_global = 512
    config.trainer.max_iter = 100000
    config.trainer.save_ckpt_iter = 10000

    config.log_config.group = "edm_cifar10_sft"

    return config
