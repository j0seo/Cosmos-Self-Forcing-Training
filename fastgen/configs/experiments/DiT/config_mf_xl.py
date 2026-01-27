# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.net import DiT_IN256_XL_Config
import fastgen.configs.experiments.DiT.config_mf_b as config_mf_default


""" Configs for the MeanFlow model, on DiT-XL and ImageNet-256 dataset. """


def create_config():
    config = config_mf_default.create_config()

    config.model.guidance_t_end = 0.75
    config.model.guidance_scale = 0.2
    config.model.guidance_mixture_ratio = 0.92

    config.model.net = DiT_IN256_XL_Config
    # remove the additional 1000 factor in JVP
    config.model.net.scale_t = False
    config.model.net.r_timestep = True
    config.model.net.time_cond_type = "diff"

    config.dataloader_train.batch_size = 8

    return config
