# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from fastgen.configs.discriminator import Discriminator_SDXL_Res1024_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import ImageLatentLoaderConfig
from fastgen.configs.net import SDXLConfig

""" Configs for the DMD2 model on SDXL. """


def create_config():
    config = config_dmd2_default.create_config()

    config.model.net_optimizer.lr = 5e-5
    config.model.discriminator_optimizer.lr = 5e-5
    config.model.fake_score_optimizer.lr = 5e-5

    config.model.input_shape = [4, 128, 128]
    config.model.discriminator = Discriminator_SDXL_Res1024_Config
    config.model.gan_loss_weight_gen = 5e-3
    config.model.guidance_scale = 8.0
    config.model.net = SDXLConfig
    config.model.enable_preprocessors = False

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.dataloader_train = ImageLatentLoaderConfig
    config.dataloader_train.batch_size = 6

    config.trainer.max_iter = 25000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 2000

    config.log_config.group = "sdxl_dmd2"
    return config
