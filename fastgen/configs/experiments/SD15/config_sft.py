# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import SD15Config

"""Configs for SFT (Supervised Fine-Tuning) on Stable Diffusion v1.5."""


def create_config():
    config = config_sft_default.create_config()

    # Model precision
    config.model.precision_amp = "bfloat16"
    config.model.precision_amp_infer = "bfloat16"
    config.model.precision_amp_enc = "bfloat16"

    # SD1.5 latent shape: [C, H, W] = [4, 64, 64] for 512x512 images
    config.model.input_shape = [4, 64, 64]

    # Network config
    config.model.net = SD15Config

    # Optimizer config
    config.model.net_optimizer.lr = 1e-5

    # Timestep sampling config
    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Sampling config for visualization
    config.model.guidance_scale = 7.5
    config.model.student_sample_steps = 50

    # Dataloader
    config.dataloader_train = ImageLoaderConfig

    # Trainer
    config.trainer.batch_size_global = 256
    config.trainer.max_iter = 100000
    config.trainer.logging_iter = 500

    config.log_config.group = "sd15_sft"

    return config
