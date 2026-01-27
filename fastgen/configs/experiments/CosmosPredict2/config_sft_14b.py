# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configs for SFT on Cosmos-Predict2.5-14B model."""

import fastgen.configs.experiments.CosmosPredict2.config_sft as config_sft_base
from fastgen.configs.net import CosmosPredict2_14B_Config, CKPT_ROOT_DIR


def create_config():
    config = config_sft_base.create_config()

    # Network for 14B
    config.model.net = CosmosPredict2_14B_Config
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cosmos_predict2/Cosmos-Predict2.5-14B/base/post-trained/e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt"

    config.log_config.group = "cosmos_predict2_14b_sft"

    return config
