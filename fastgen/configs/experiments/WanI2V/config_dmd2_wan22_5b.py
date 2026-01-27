# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.discriminator import Discriminator_Wan22_5B_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan22_I2V_5B_Config

""" Configs for the DMD2 model on Wan-2.2-5B model. """


def create_config():
    config = config_dmd2_default.create_config()
    config.model.fsdp_meta_init = True

    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    config.model.precision = "bfloat16"
    # VAE compress ratio: (1+T/4) * H / 16 * W / 16
    config.model.input_shape = [48, 21, 44, 80]
    config.model.discriminator = Discriminator_Wan22_5B_Config
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [15, 22, 29]
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"
    config.model.student_sample_type = "ode"
    config.model.guidance_scale = 5.0
    config.model.net = Wan22_I2V_5B_Config

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # setting for 2-step training
    config.model.student_sample_steps = 2
    config.model.sample_t_cfg.t_list = [0.999, 0.833, 0.0]

    config.dataloader_train = VideoLoaderConfig
    config.dataloader_train.batch_size = 2

    # 480p (832x480) resolution
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 16, config.model.input_shape[-2] * 16)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.log_config.group = "wan22_5b_i2v_dmd2"
    return config
