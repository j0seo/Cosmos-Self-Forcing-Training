# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.discriminator import Discriminator_Flux_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import FluxConfig

"""Configs for DMD2 distillation on Flux model."""


def create_config():
    config = config_dmd2_default.create_config()

    # Optimizer settings
    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    # Flux latent shape: [C, H, W] = [16, H//8, W//8]
    # For 512x512 images: [16, 64, 64]
    config.model.input_shape = [16, 64, 64]

    # Discriminator config - ImageDiT with simple conv2d architecture
    config.model.discriminator = Discriminator_Flux_Config

    # GAN loss weight
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"

    # Network config
    config.model.net = FluxConfig
    # Flux uses embedded guidance via config.model.net.guidance_scale instead
    config.model.net.guidance_scale = 3.5

    # Precision
    config.model.precision = "bfloat16"

    # Student sampling steps
    config.model.student_sample_steps = 4

    # Time sampling config
    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Dataloader
    config.dataloader_train = ImageLoaderConfig
    config.dataloader_train.batch_size = 2
    config.dataloader_train.input_res = 512

    # Training iterations
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    # Logging
    config.log_config.group = "flux_dmd2"

    return config
