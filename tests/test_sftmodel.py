# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import torch
import pytest
from fastgen.methods import SFTModel
from fastgen.configs.methods.config_sft import ModelConfig
from fastgen.configs.config_utils import override_config_with_opts


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    instance = ModelConfig()
    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1", "r_timestep=False"]
    instance.net = override_config_with_opts(instance.net, opts)

    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""  # disable ckpt loading
    instance.input_shape = [3, 8, 8]

    # SFT-specific configurations
    instance.cond_dropout_prob = 0.1
    instance.cond_keys_no_dropout = []
    instance.guidance_scale = None

    model = SFTModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    # Use one-hot encoded labels as EDM network expects them
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
    neg_condition = torch.zeros(batch_size, 10)

    # Create mock data
    data = {
        "real": torch.randn(batch_size, 3, 8, 8).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_condition.to(model.device, model.precision),
    }

    return model, data


def test_single_train_step(get_model_data):
    """Test the single training step of SFT model."""
    model, data = get_model_data

    # Run the training step
    loss_map, outputs = model.single_train_step(data, 0)

    # Test loss_map structure
    assert isinstance(loss_map, dict)
    assert "total_loss" in loss_map
    assert "dsm_loss" in loss_map
    assert isinstance(loss_map["total_loss"], torch.Tensor)
    assert isinstance(loss_map["dsm_loss"], torch.Tensor)
    assert loss_map["total_loss"].detach().item() >= 0.0
    assert loss_map["dsm_loss"].detach().item() >= 0.0
    # For SFT, total_loss should equal dsm_loss
    assert torch.allclose(loss_map["total_loss"], loss_map["dsm_loss"])

    # Test outputs structure
    assert isinstance(outputs, dict)
    assert "gen_rand" in outputs
    assert "input_rand" in outputs
    assert isinstance(outputs["input_rand"], torch.Tensor)
    assert outputs["input_rand"].shape == data["real"].shape
    # gen_rand should be a callable (partial function)
    assert callable(outputs["gen_rand"])


def test_mix_condition_tensor(get_model_data):
    """Test the _mix_condition method with tensor inputs."""
    model, data = get_model_data

    condition = data["condition"]
    neg_condition = data["neg_condition"]

    # Test with cond_dropout_prob = None (no dropout)
    model.config.cond_dropout_prob = None
    mixed_condition = model._mix_condition(condition, neg_condition)
    assert torch.allclose(mixed_condition, condition)

    # Test with cond_dropout_prob = 1.0 (full dropout)
    model.config.cond_dropout_prob = 1.0
    mixed_condition = model._mix_condition(condition, neg_condition)
    assert torch.allclose(mixed_condition, neg_condition)

    # Test with intermediate cond_dropout_prob
    model.config.cond_dropout_prob = 0.5
    mixed_condition = model._mix_condition(condition, neg_condition)
    assert mixed_condition.shape == condition.shape


def test_mix_condition_dict(get_model_data):
    """Test the _mix_condition method with dictionary inputs."""
    model, data = get_model_data

    # Create dictionary conditions
    dict_condition = {"text_embeds": data["condition"], "other_info": torch.ones_like(data["condition"])}
    dict_neg_condition = {"text_embeds": data["neg_condition"], "other_info": torch.zeros_like(data["neg_condition"])}

    # Test with cond_dropout_prob = None (no dropout)
    model.config.cond_dropout_prob = None
    mixed_condition = model._mix_condition(dict_condition, dict_neg_condition)
    assert torch.allclose(mixed_condition["text_embeds"], dict_condition["text_embeds"])
    assert torch.allclose(mixed_condition["other_info"], dict_condition["other_info"])

    # Test with cond_dropout_prob = 1.0 (full dropout)
    model.config.cond_dropout_prob = 1.0
    mixed_condition = model._mix_condition(dict_condition, dict_neg_condition)
    assert torch.allclose(mixed_condition["text_embeds"], dict_neg_condition["text_embeds"])
    assert torch.allclose(mixed_condition["other_info"], dict_neg_condition["other_info"])


def test_mix_condition_with_no_dropout_keys(get_model_data):
    """Test the _mix_condition method with keys that should not be dropped."""
    model, data = get_model_data

    # Set cond_keys_no_dropout
    model.config.cond_keys_no_dropout = {"other_info"}

    # Create dictionary conditions
    dict_condition = {"text_embeds": data["condition"], "other_info": torch.ones_like(data["condition"])}
    dict_neg_condition = {"text_embeds": data["neg_condition"], "other_info": torch.zeros_like(data["neg_condition"])}

    # Test with cond_dropout_prob = 1.0 (full dropout)
    model.config.cond_dropout_prob = 1.0
    mixed_condition = model._mix_condition(dict_condition, dict_neg_condition)
    # text_embeds should be dropped (replaced with neg_condition)
    assert torch.allclose(mixed_condition["text_embeds"], dict_neg_condition["text_embeds"])
    # other_info should NOT be dropped (kept as original condition)
    assert torch.allclose(mixed_condition["other_info"], dict_condition["other_info"])


def test_generator_fn(get_model_data):
    """Test the static generator_fn method."""
    model, data = get_model_data

    # Mock a network with sample method
    class MockNet(torch.nn.Module):
        def sample(self, noise, condition=None, neg_condition=None, guidance_scale=None, **kwargs):
            return torch.randn_like(noise)

    mock_net = MockNet()
    noise = torch.randn_like(data["real"])

    # Test generator function
    result = SFTModel.generator_fn(
        net=mock_net,
        noise=noise,
        condition=data["condition"],
        neg_condition=data["neg_condition"],
        guidance_scale=None,
    )

    assert isinstance(result, torch.Tensor)
    assert result.shape == noise.shape


def test_optimizers(get_model_data):
    """Test optimizer functionality."""
    model, data = get_model_data

    # Test that optimizer operations run without errors
    model.optimizers_zero_grad(0)
    loss_map, _ = model.single_train_step(data, 0)

    # Test that loss requires gradients and can be backpropagated
    assert loss_map["total_loss"].requires_grad
    model.grad_scaler.scale(loss_map["total_loss"]).backward()

    # Test that gradients exist after backward pass
    has_gradients = False
    for param in model.net.parameters():
        if param.requires_grad and param.grad is not None:
            has_gradients = True
            break
    assert has_gradients, "No gradients found after backward pass"

    # Test that optimizer step runs without errors
    model.optimizers_schedulers_step(0)


def test_loss_computation(get_model_data):
    """Test that loss computation produces reasonable values."""
    model, data = get_model_data

    loss_map, outputs = model.single_train_step(data, 0)

    # Test loss properties
    total_loss = loss_map["total_loss"]
    dsm_loss = loss_map["dsm_loss"]

    assert total_loss.requires_grad
    assert dsm_loss.requires_grad
    assert not torch.isnan(total_loss)
    assert not torch.isnan(dsm_loss)
    assert not torch.isinf(total_loss)
    assert not torch.isinf(dsm_loss)
