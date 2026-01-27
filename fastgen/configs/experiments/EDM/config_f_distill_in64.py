# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig
from fastgen.configs.discriminator import Discriminator_EDM_ImageNet64_Config
import fastgen.configs.methods.config_f_distill as config_f_distill_default
from fastgen.configs.data import ImageNet64_Loader_Config
from fastgen.configs.net import EDM_ImageNet64_Config, CKPT_ROOT_DIR
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS

""" Configs for the f-distill model, on EDM and ImageNet-64 dataset. """


def create_config():
    config = config_f_distill_default.create_config()

    config.model.net_optimizer.lr = 2e-6
    config.model.discriminator_optimizer.lr = 2e-6
    config.model.fake_score_optimizer.lr = 2e-6

    config.model.input_shape = [3, 64, 64]
    config.model.discriminator = Discriminator_EDM_ImageNet64_Config
    config.model.gan_loss_weight_gen = 3e-3
    config.model.f_distill.ratio_ema_rate = 0.5

    # ema
    config.model.use_ema = ["ema_9999", "ema_99995", "ema_9996"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)

    # path to the pretrained diffusion model ckpt
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-64/edm-imagenet-64x64-cond-adm.pth"
    config.model.net = EDM_ImageNet64_Config

    config.dataloader_train = ImageNet64_Loader_Config

    config.trainer.batch_size_global = 512
    config.trainer.max_iter = 600000
    config.log_config.group = "edm_imagenet64_fdistill"
    return config
