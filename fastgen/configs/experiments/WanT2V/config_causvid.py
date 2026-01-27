# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.discriminator import Discriminator_Wan_1_3B_Config
import fastgen.configs.methods.config_causvid as config_causvid_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan_1_3B_Config, CausalWan_1_3B_Config, CKPT_ROOT_DIR

""" Configs for the CausVid model on Wan-1.3B model. """


def create_config():
    config = config_causvid_default.create_config()
    config.model.net_optimizer.lr = 5e-5
    config.model.discriminator_optimizer.lr = 5e-5
    config.model.fake_score_optimizer.lr = 5e-5

    config.model.precision = "bfloat16"
    # VAE compress ratio: (1+T/4) * H / 8 * W / 8
    config.model.input_shape = [16, 21, 60, 104]
    config.model.fake_score_pred_type = "x0"
    config.model.guidance_scale = 5.0
    config.model.net = CausalWan_1_3B_Config
    config.model.net.total_num_frames = config.model.input_shape[1]
    config.model.teacher = Wan_1_3B_Config

    # GAN settings
    config.model.gan_loss_weight_gen = 0.001
    config.model.discriminator = Discriminator_Wan_1_3B_Config
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [15, 22, 29]
    config.model.gan_use_same_t_noise = True

    # Pretrained Self-Forcing checkpoint for causal WAN 1.3B student (see networks/README.md for download)
    config.model.pretrained_student_net_path = f"{CKPT_ROOT_DIR}/Self-Forcing/checkpoints/ode_init.pt"
    config.model.net.use_wan_official_sinusoidal = True

    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]
    config.model.use_ema = True

    config.dataloader_train = VideoLoaderConfig
    config.dataloader_train.batch_size = 2

    # 480p (832x480) resolution
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.log_config.group = "wan_causvid"
    return config
