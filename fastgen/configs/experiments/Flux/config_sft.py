# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import FluxConfig

"""Configs for SFT (Supervised Fine-Tuning) on Flux."""


def create_config():
    config = config_sft_default.create_config()

    # Model precision
    config.model.precision = "bfloat16"

    # Flux latent shape: [C, H, W] = [16, H//8, W//8]
    # For 512x512 images: [16, 64, 64]
    # For 1024x1024 images: [16, 128, 128]
    config.model.input_shape = [16, 64, 64]  # 512x512 images

    # Network config
    config.model.net = FluxConfig

    # Optimizer config
    config.model.net_optimizer.lr = 1e-5

    # Timestep sampling config
    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Sampling config for visualization
    config.model.guidance_scale = 3.5
    config.model.student_sample_steps = 50

    # Dataloader (raw images - Flux VAE encodes to 16ch latents)
    config.dataloader_train = ImageLoaderConfig
    config.dataloader_train.batch_size = 4

    # Trainer
    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 500
    config.trainer.batch_size_global = 128

    config.log_config.group = "flux_sft"

    return config
