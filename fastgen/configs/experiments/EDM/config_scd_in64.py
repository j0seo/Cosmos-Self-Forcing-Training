# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

import fastgen.configs.methods.config_scm as config_scm_default
from fastgen.configs.data import ImageNet64_Loader_Config
from fastgen.configs.net import EDM_ImageNet64_Config, CKPT_ROOT_DIR
from fastgen.utils import LazyCall as L
from fastgen.utils.lr_scheduler import LambdaInverseSquareRootScheduler
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS

""" Configs for the sCM model, on EDM and ImageNet-64 dataset. """


def create_config():
    config = config_scm_default.create_config()

    config.trainer.callbacks.grad_clip.grad_norm = 10

    # precision
    config.model.precision_amp = "bfloat16"
    config.model.precision_amp_jvp = "float32"

    config.model.input_shape = [3, 64, 64]

    config.model.net_optimizer.optim_type = "adam"
    config.model.net_optimizer.lr = 7e-5
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.eps = 1e-11
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler = L(LambdaInverseSquareRootScheduler)(
        warm_up_steps=0,
        decay_steps=35000,
    )

    # path to the pretrained diffusion model ckpt
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-64/edm-imagenet-64x64-cond-adm.pth"
    config.model.net = EDM_ImageNet64_Config
    config.model.net.dropout = 0.0
    config.model.loss_config.use_cd = True

    # ema
    config.model.use_ema = ["ema_9999", "ema_99995", "ema_9996"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)

    config.model.sample_t_cfg.train_p_mean = -1.0
    config.model.sample_t_cfg.train_p_std = 1.6

    # It can help to increase the max. time beyond 80
    # config.model.net.max_t = 80.0
    # config.model.sample_t_cfg.max_t = config.model.net.max_t

    # Recommended setting for 2-step:
    # config.model.sample_t_cfg.t_list = [config.model.net.max_t, 1.1, 0.0]
    # config.model.student_sample_steps = 2

    # During inference, sigma_shift can improve 2-step results
    # config.model.net.sigma_shift = 0.003

    config.dataloader_train = ImageNet64_Loader_Config

    config.trainer.batch_size_global = 1024
    config.trainer.max_iter = 600000
    config.log_config.group = "adm_imagenet64_scm"
    return config
