# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.discriminator import Discriminator_SD15_Res512_Config
import fastgen.configs.methods.config_f_distill as config_f_distill_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import SD15Config

""" Configs for the f-distill model on Stable Diffusion v1.5. """


def create_config():
    config = config_f_distill_default.create_config()

    config.model.precision_amp = "bfloat16"

    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5
    config.model.fake_score_pred_type = "x0"

    config.model.input_shape = [4, 64, 64]
    config.model.discriminator = Discriminator_SD15_Res512_Config
    config.model.gan_loss_weight_gen = 1e-3
    config.model.guidance_scale = 1.75
    config.model.student_update_freq = 10
    config.model.net = SD15Config

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.shift = 1
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.dataloader_train = ImageLoaderConfig

    config.trainer.batch_size_global = 2048
    config.trainer.max_iter = 100000
    config.trainer.save_ckpt_iter = 2000
    config.log_config.group = "sd15_fdistill"
    return config
