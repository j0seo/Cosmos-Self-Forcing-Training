# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import torch
import pytest
from typing import Callable
from fastgen.methods import LADDModel
from fastgen.configs.methods.config_ladd import ModelConfig
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.utils.test_utils import check_grad_zero


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    instance = ModelConfig()
    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_discriminator)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""  # disable ckpt loading
    instance.student_update_freq = 2
    instance.input_shape = [3, 8, 8]

    model = LADDModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    neg_condition = torch.zeros(batch_size, 10)

    # Create mock data
    data = {
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_condition.to(model.device, model.precision),
    }
    return model, data


def test_single_train_step_student_update(get_model_data):
    model, data = get_model_data
    # Run the training step
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "total_loss" in loss_map
    assert "gan_loss_gen" in loss_map
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["gen_rand"], torch.Tensor)
    assert isinstance(outputs["input_rand"], torch.Tensor)


def test_single_train_step_disc_update(get_model_data):
    model, data = get_model_data
    # Run the training step
    loss_map, outputs = model.single_train_step(data, 1)

    # Assertions
    assert "gan_loss_disc" in loss_map
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["gen_rand"], torch.Tensor)
    assert isinstance(outputs["input_rand"], torch.Tensor)


def test_optimizers(get_model_data):
    model, data = get_model_data
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


@pytest.fixture
def get_multistep_model_data():
    """Fixture for multi-step distillation testing with student_sample_steps=2."""
    instance = ModelConfig()
    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_discriminator)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if instance.device == torch.device("cpu"):
        instance.precision = "float32"
    instance.pretrained_model_path = ""  # disable ckpt loading
    instance.student_update_freq = 2
    instance.input_shape = [3, 8, 8]

    # Enable multi-step distillation
    instance.student_sample_steps = 2

    model = LADDModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    neg_condition = torch.zeros(batch_size, 10)

    # Create mock data
    data = {
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_condition.to(model.device, model.precision),
    }
    return model, data


def test_multistep_student_update(get_multistep_model_data):
    """Test multi-step student distillation with student_sample_steps=2."""
    model, data = get_multistep_model_data

    # Verify multi-step configuration
    assert model.config.student_sample_steps == 2, "Test should use multi-step configuration"

    # Run the training step (student update)
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions for loss_map
    assert "total_loss" in loss_map
    assert "gan_loss_gen" in loss_map

    # Assertions for outputs - should be Callable in multi-step mode
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable), "Multi-step mode should return Callable"
    assert isinstance(outputs["input_rand"], torch.Tensor)

    # Execute the callable and verify shape
    gen_tensor = outputs["gen_rand"]()
    assert isinstance(gen_tensor, torch.Tensor)
    assert gen_tensor.shape == data["real"].shape
    assert outputs["input_rand"].shape == data["real"].shape


def test_multistep_discriminator_update(get_multistep_model_data):
    """Test multi-step discriminator update with student_sample_steps=2."""
    model, data = get_multistep_model_data

    # Verify multi-step configuration
    assert model.config.student_sample_steps == 2, "Test should use multi-step configuration"

    # Run the training step (discriminator update)
    loss_map, outputs = model.single_train_step(data, 1)

    # Assertions for loss_map
    assert "gan_loss_disc" in loss_map

    # Assertions for outputs - should be Callable in multi-step mode
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable), "Multi-step mode should return Callable"
    assert isinstance(outputs["input_rand"], torch.Tensor)

    # Verify callable can be executed (don't need to check shape for discriminator test)
    gen_tensor = outputs["gen_rand"]()
    assert isinstance(gen_tensor, torch.Tensor)
