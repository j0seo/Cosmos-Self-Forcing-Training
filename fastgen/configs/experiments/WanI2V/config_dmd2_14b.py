# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.discriminator import Discriminator_Wan_14B_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan21_I2V_14B_480P_Config

""" Configs for the DMD2 model on Wan-14B model. """


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
    # VAE compress ratio: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]
    config.model.discriminator = Discriminator_Wan_14B_Config
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [21, 30, 39]
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"
    config.model.guidance_scale = 5.0
    config.model.net = Wan21_I2V_14B_480P_Config

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.student_sample_type = "ode"

    # setting for 2-step training
    config.model.student_sample_steps = 2
    config.model.sample_t_cfg.t_list = [0.999, 0.833, 0.0]

    config.dataloader_train = VideoLoaderConfig

    # Wan2.2 5B uses 16*16 spatial downsampling factor
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.dataloader_train.batch_size = 1

    config.log_config.group = "wan21_14b_i2v_dmd2"
    return config
