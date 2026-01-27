# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import fastgen.configs.methods.config_kd as config_kd_default
from fastgen.configs.data import PairLoaderConfig
from fastgen.configs.net import CogVideoXConfig

""" Configs for the KD model on CogVideoX model. """


def create_config():
    config = config_kd_default.create_config()

    config.trainer.max_iter = 6000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500

    config.model.net_optimizer.lr = 1e-4

    config.model.input_shape = [16, 13, 60, 90]
    config.model.net = CogVideoXConfig
    config.model.enable_preprocessors = False
    config.model.precision = "bfloat16"

    config.dataloader_train = PairLoaderConfig
    config.dataloader_train.batch_size = 2

    config.log_config.group = "CogVideoX_kd"
    return config
