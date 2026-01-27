# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import pytest
from omegaconf import DictConfig
from fastgen.configs.methods.config_dmd2 import create_config
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.trainer import Trainer
from fastgen.utils import instantiate
from fastgen.utils.io_utils import set_env_vars
from fastgen.configs.callbacks import GradClip_CALLBACK


@pytest.mark.skip
def test_trainer():
    set_env_vars(credentials_path="./credentials/s3.json")
    config = create_config()
    config.log_config.name = "test"  # so the wandb.init doesn't hang
    config.trainer.callbacks = DictConfig({**GradClip_CALLBACK})

    model_config = config.model
    model_config.gan_loss_weight_gen = 0
    opts_network = ["-", "img_resolution=2", "channel_mult=[1]", "channel_mult_noise=1"]
    model_config.teacher = override_config_with_opts(model_config.net, opts_network)
    model_config.net = override_config_with_opts(model_config.net, opts_network)
    model_config.fake_score = override_config_with_opts(model_config.net, opts_network)

    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    model_config.discriminator = override_config_with_opts(model_config.discriminator, opts_discriminator)

    model_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config.pretrained_model_path = ""  # disable ckpt loading
    model_config.input_shape = [3, 2, 2]

    data_loader_config = config.dataloader_train
    config.dataloader_train = override_config_with_opts(data_loader_config, ["-", "batch_size=1"])
    config.dataloader_val = override_config_with_opts(data_loader_config, ["-", "batch_size=1"])

    config.trainer = override_config_with_opts(config.trainer, ["-", "max_iter=2"])

    # initiate the model
    config.model_class.config = config.model
    model = instantiate(config.model_class)
    config.model_class.config = None

    # initiate the dataloaders
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val = instantiate(config.dataloader_val)

    # initiate the trainer
    fastgen_trainer = Trainer(config)
    fastgen_trainer.run(model, dataloader_train, dataloader_val)
