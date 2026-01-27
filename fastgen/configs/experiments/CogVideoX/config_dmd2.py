# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from fastgen.configs.discriminator import Discriminator_CogVideoX2B_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import VideoLatentLoaderConfig
from fastgen.configs.net import CogVideoXConfig

""" Configs for the DMD2 model on CogVideoX model. """


def create_config():
    config = config_dmd2_default.create_config()

    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    config.model.input_shape = [16, 13, 60, 90]
    config.model.discriminator = Discriminator_CogVideoX2B_Config
    config.model.discriminator.feature_indices = [15, 22, 29]
    config.model.gan_loss_weight_gen = 0.03
    config.model.net = CogVideoXConfig
    config.model.guidance_scale = 6.0
    config.model.enable_preprocessors = False

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"
    config.model.student_sample_type = "ode"

    # setting for 4-step training
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    config.dataloader_train = VideoLatentLoaderConfig
    config.dataloader_train.batch_size = 2

    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "CogVideoX_dmd2"
    return config
