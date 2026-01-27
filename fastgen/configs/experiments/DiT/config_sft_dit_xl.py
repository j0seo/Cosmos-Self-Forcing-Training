# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageNet256_Loader_Config
from fastgen.configs.net import DiT_IN256_XL_Config, CKPT_ROOT_DIR
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS

"""Configs for SFT (Supervised Fine-Tuning) on DiT-XL and ImageNet-256 dataset."""


def create_config():
    config = config_sft_default.create_config()

    # model
    # DiT latent shape: [C, H, W] = [4, 32, 32] for 256x256 images (256/8 = 32)
    config.model.input_shape = [4, 32, 32]
    config.model.precision_amp = "bfloat16"
    config.model.cond_dropout_prob = 0.1  # 10% dropout for CFG training

    # Timestep sampling config
    config.model.sample_t_cfg.time_dist_type = "logitnormal"
    config.model.sample_t_cfg.train_p_mean = -0.4
    config.model.sample_t_cfg.train_p_std = 1.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Pretrained model path
    # DiT-XL/2 ImageNet-256 checkpoint from https://github.com/facebookresearch/DiT
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/imagenet-256/DiT-XL-2-256x256.pt"

    # Network config
    config.model.net = DiT_IN256_XL_Config
    config.model.net.learn_sigma = True  # Facebook DiT checkpoint was trained with learn_sigma=True
    # Facebook DiT was trained with DDPM (epsilon prediction), not flow matching
    config.model.net.net_pred_type = "eps"
    config.model.net.schedule_type = "sd"  # Uses same linear beta schedule as DiT (0.0001 to 0.02)

    # Optimizer config (lower LR for fine-tuning; 1e-4 is for training from scratch)
    config.model.net_optimizer.optim_type = "adamw"
    config.model.net_optimizer.lr = 1e-5  # 10x lower for fine-tuning
    config.model.net_optimizer.betas = (0.9, 0.95)
    config.model.net_optimizer.weight_decay = 0.0

    # EMA config
    config.model.use_ema = ["ema_9999", "ema_99995"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)

    # Dataloader
    config.dataloader_train = ImageNet256_Loader_Config

    # Sampling config for visualization
    config.model.student_sample_steps = 50  # DiT typically uses 50-250 steps

    # Trainer
    config.trainer.batch_size_global = 256
    config.trainer.max_iter = 400000
    config.trainer.save_ckpt_iter = 10000
    config.trainer.logging_iter = 1000

    config.log_config.group = "dit_xl_imagenet256_sft"

    return config
