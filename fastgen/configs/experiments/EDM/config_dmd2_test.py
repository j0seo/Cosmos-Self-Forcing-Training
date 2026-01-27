# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.methods.config_dmd2 import create_config as dmd2_create_config
from fastgen.configs.net import CKPT_ROOT_DIR


def create_config():
    config = dmd2_create_config()

    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cifar10/edm-cifar10-32x32-cond-vp.pth"

    config.trainer.max_iter = 100000
    config.trainer.logging_iter = 50
    config.trainer.save_ckpt_iter = 1000
    config.dataloader_train.batch_size = 64

    return config
