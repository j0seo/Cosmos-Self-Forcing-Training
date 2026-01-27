# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import copy
import gc
import pytest
import torch

from fastgen.configs.methods.config_dmd2 import ModelConfig
from fastgen.configs.net import CausalWan_1_3B_Config, Wan_1_3B_Config
from fastgen.configs.discriminator import Discriminator_Wan_1_3B_Config
from fastgen.methods import CausVidModel
from fastgen.utils.test_utils import check_grad_zero
from fastgen.utils.test_utils import RunIf
from fastgen.utils.io_utils import set_env_vars


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    set_env_vars()

    config = ModelConfig()
    config.net = copy.deepcopy(CausalWan_1_3B_Config)
    config.net.chunk_size = 2  # Override chunk_size to match 2-frame test data
    config.teacher = Wan_1_3B_Config  # teacher is still bidirectional network

    # Set discriminator config with lightweight overrides
    config.discriminator = Discriminator_Wan_1_3B_Config
    config.discriminator.num_blocks = 1

    # Set device and precision
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.precision = "float32" if config.device == torch.device("cpu") else "bfloat16"
    config.pretrained_model_path = ""  # disable ckpt loading
    config.student_update_freq = 2
    config.student_sample_steps = 4  # multistep student distillation

    # Set the correct timestep range
    config.sample_t_cfg.max_t = 0.999
    config.sample_t_cfg.min_t = 0.001

    # Use small dimensions for testing (Wan uses 16 channels)
    config.input_shape = [16, 2, 4, 4]  # [C, T, H, W]
    config.enable_preprocessors = True  # Enable text encoder for proper condition generation
    config.gan_loss_weight_gen = 0.001  # Set GAN loss weight

    model = CausVidModel(config)
    # only use a single block for testing
    model.net.transformer.blocks = model.net.transformer.blocks[:1]
    model.teacher.transformer.blocks = model.teacher.transformer.blocks[:1]

    # Initialize the text encoder to generate proper condition tensors
    model.net.init_preprocessors()
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    channels, n_frames, height, width = config.input_shape

    # Create much smaller mock video data (Wan uses UMT5: seq_len=512, hidden_dim=4096)
    data = {
        "real": torch.randn(batch_size, channels, n_frames, height, width, device=model.device, dtype=model.precision),
        "condition": torch.randn(batch_size, 512, 4096, device=model.device, dtype=model.precision),
        "neg_condition": torch.zeros(batch_size, 512, 4096, device=model.device, dtype=model.precision),
    }

    return model, data


@RunIf(min_gpus=1)
def test_causal_timestep_sampling(get_model_data):
    model, data = get_model_data

    # Test that the causal model uses heterogeneous timesteps
    batch_size = data["real"].shape[0]
    num_frames = data["real"].shape[2]

    # Use model's chunk size (overridden to 2 in fixture to match 2-frame test data)
    chunk_size = getattr(model.net, "chunk_size", 2)

    # Access noise scheduler correctly through the model's noise_scheduler attribute
    t_inhom, _ = model.teacher.noise_scheduler.sample_t_inhom(
        n=batch_size, seq_len=num_frames, chunk_size=chunk_size, sample_steps=4
    )

    # Check that timesteps have the right shape
    assert t_inhom.shape == (batch_size, num_frames)

    # Check that different chunks can have different timesteps (causal property)
    if num_frames > chunk_size:
        # First chunk timesteps
        first_chunk_t = t_inhom[:, :chunk_size]
        # Second chunk timesteps
        second_chunk_t = (
            t_inhom[:, chunk_size : 2 * chunk_size] if num_frames >= 2 * chunk_size else t_inhom[:, chunk_size:]
        )

        # Timesteps within a chunk should be the same
        assert torch.allclose(first_chunk_t[:, 0:1], first_chunk_t, atol=1e-6)
        if second_chunk_t.shape[1] > 1:
            assert torch.allclose(second_chunk_t[:, 0:1], second_chunk_t, atol=1e-6)
    else:
        # With only 2 frames and chunk_size=2, we should have homogeneous timesteps
        assert torch.allclose(t_inhom[:, 0:1], t_inhom, atol=1e-6)


@RunIf(min_gpus=1)
def test_single_train_step_student_update(get_model_data):
    model, data = get_model_data
    # Run the training step
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
def test_single_train_step_fake_score_update(get_model_data):
    model, data = get_model_data

    # Run the training step
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
def test_optimizers(get_model_data):
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
    check_grad_zero(model.discriminator)
    check_grad_zero(model.fake_score)
