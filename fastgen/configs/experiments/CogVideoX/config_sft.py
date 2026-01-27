# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import VideoLatentLoaderConfig
from fastgen.configs.net import CogVideoXConfig

""" Configs for the SFT model on CogVideoX model. """


def create_config():
    config = config_sft_default.create_config()

    config.trainer.logging_iter = 500

    config.model.net_optimizer.lr = 5e-5

    config.model.guidance_scale = 6.0
    config.model.student_sample_steps = 50

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # CogVideoX latent shape: [C, T, H, W] = [16, 13, 60, 90]
    # Corresponds to 49 frames at 480x720 resolution after VAE encoding
    config.model.input_shape = [16, 13, 60, 90]
    config.model.net = CogVideoXConfig
    config.model.enable_preprocessors = False  # Using precomputed latents

    config.dataloader_train = VideoLatentLoaderConfig
    config.dataloader_train.batch_size = 2

    config.log_config.group = "CogVideoX_sft"
    return config
