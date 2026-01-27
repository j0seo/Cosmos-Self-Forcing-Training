# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import gc
import pytest
import torch

from fastgen.configs.methods.config_self_forcing import ModelConfig
from fastgen.configs.net import CausalWan_1_3B_Config, Wan_1_3B_Config
from fastgen.methods import SelfForcingModel
from fastgen.utils.test_utils import check_grad_zero
from fastgen.utils.test_utils import RunIf
from fastgen.utils.io_utils import set_env_vars


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    set_env_vars()

    config = ModelConfig()
    config.net = CausalWan_1_3B_Config
    config.teacher = Wan_1_3B_Config  # teacher is still bidirectional network

    # Set device and precision
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.precision = "float32" if config.device == torch.device("cpu") else "bfloat16"
    config.pretrained_model_path = ""  # disable ckpt loading
    config.student_update_freq = 2
    config.student_sample_steps = 4  # multistep student distillation

    # Set the correct timestep range for Wan models
    config.sample_t_cfg.max_t = 0.999
    config.sample_t_cfg.min_t = 0.001
    config.sample_t_cfg.time_dist_type = "uniform"

    # Use dimensions appropriate for Wan models
    config.input_shape = [16, 21, 30, 52]  # [C, T, H, W] - smaller for testing
    config.enable_preprocessors = True  # Enable text encoder for proper condition generation
    config.gan_loss_weight_gen = 0.0  # Set GAN loss weight

    # Self-forcing specific settings
    config.enable_gradient_in_rollout = True
    config.start_gradient_frame = 0
    config.same_step_across_blocks = True
    config.last_step_only = False

    model = SelfForcingModel(config)
    # only use a single block for testing
    model.net.transformer.blocks = model.net.transformer.blocks[:1]
    model.teacher.transformer.blocks = model.teacher.transformer.blocks[:1]

    # Initialize the text encoder to generate proper condition tensors
    model.net.init_preprocessors()
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    channels, n_frames, height, width = config.input_shape

    # Create mock video data appropriate for Wan models
    data = {
        "real": torch.randn(batch_size, channels, n_frames, height, width, device=model.device, dtype=model.precision),
        "condition": torch.randn(batch_size, 512, 4096, device=model.device, dtype=model.precision),
        "neg_condition": torch.zeros(batch_size, 512, 4096, device=model.device, dtype=model.precision),
    }

    return model, data


@RunIf(min_gpus=1)
def test_sf_denoising_step_sampling(get_model_data):
    model, data = get_model_data

    # Test the denoising step sampling specific to self-forcing
    num_frames = data["real"].shape[2]
    chunk_size = getattr(model.net, "chunk_size", 3)

    # Calculate number of blocks
    num_blocks = (num_frames + chunk_size - 1) // chunk_size

    # Sample denoising end steps
    end_indices = model._sample_denoising_end_steps(num_blocks)

    # Check that we get the right number of indices
    assert len(end_indices) == num_blocks

    # Check that all indices are within valid range [0, sample_steps)
    sample_steps = model.config.student_sample_steps
    for idx in end_indices:
        assert 0 <= idx < sample_steps


@RunIf(min_gpus=1)
def test_sf_rollout_with_gradient(get_model_data):
    model, data = get_model_data

    # Test the self-forcing specific rollout_with_gradient method
    noise = torch.randn_like(data["real"])
    condition = data["condition"]

    # Test rollout with gradient enabled
    gen_frames = model.rollout_with_gradient(
        noise=noise, condition=condition, enable_gradient=True, start_gradient_frame=0
    )

    # Check output shape matches input
    assert gen_frames.shape == noise.shape

    # Test rollout with gradient disabled
    with torch.no_grad():
        gen_frames_no_grad = model.rollout_with_gradient(
            noise=noise, condition=condition, enable_gradient=False, start_gradient_frame=0
        )

    assert gen_frames_no_grad.shape == noise.shape


@RunIf(min_gpus=1)
def test_sf_single_train_step_student_update(get_model_data):
    model, data = get_model_data
    # Run the training step for student update
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "total_loss" in loss_map
    assert "vsd_loss" in loss_map
    assert "gan_loss_gen" in loss_map
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)
    assert isinstance(outputs["input_rand"], torch.Tensor)

    gen_tensor = outputs["gen_rand"]()
    assert gen_tensor.shape == data["real"].shape
    assert outputs["input_rand"].shape == data["real"].shape


@RunIf(min_gpus=1)
def test_sf_single_train_step_fake_score_update(get_model_data):
    model, data = get_model_data

    # Run the training step for fake score update
    loss_map, outputs = model.single_train_step(data, 1)

    # Assertions
    assert "fake_score_loss" in loss_map
    assert "gan_loss_disc" in loss_map
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)
    assert isinstance(outputs["input_rand"], torch.Tensor)

    # Check that losses are finite
    assert torch.isfinite(loss_map["total_loss"])
    assert torch.isfinite(loss_map["fake_score_loss"])


@RunIf(min_gpus=1)
def test_sf_optimizers(get_model_data):
    model, data = get_model_data

    gc.collect()
    torch.cuda.empty_cache()

    # Test for both student and fake_score optimizer
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)

    # Test for both student and fake_score optimizer zero grad, after at least one backward pass
    model.optimizers_zero_grad(2)
    check_grad_zero(model.net)
    model.optimizers_zero_grad(3)
    check_grad_zero(model.fake_score)
