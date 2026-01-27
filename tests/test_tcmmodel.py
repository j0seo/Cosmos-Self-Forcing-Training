# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Callable
import pytest
import gc

import torch

from fastgen.configs.config_utils import override_config_with_opts
from fastgen.methods import TCMModel
from fastgen.methods.consistency_model.TCM import TCMPrecond
from fastgen.utils.test_utils import check_grad_zero
from fastgen.utils import instantiate


class MockNetwork(torch.nn.Module):
    """Mock network for testing TCMPrecond"""

    def __init__(self, output_shape, device=torch.device("cpu")):
        super().__init__()
        self.output_shape = output_shape
        self.device = device

        # Mock noise scheduler
        self.noise_scheduler = type("MockScheduler", (), {"min_t": 0.0, "max_t": 10.0})()

        # Mock prediction type
        self.net_pred_type = "x0"

        # Simple linear layer to make it a real network
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x_t, t, condition=None, fwd_pred_type="x0", **kwargs):
        # Return tensor with same shape as input but with modified values
        # to distinguish between teacher and student outputs
        if self.training:
            # simulate randomness, e.g., from dropout layers
            scale = torch.randn_like(x_t)
            print(f"scale: {scale[0,0,0]}")
            x_t = x_t * scale

        if hasattr(self, "_is_teacher"):
            return x_t * 0.5  # Teacher returns scaled input
        else:
            return x_t * 2.0  # Student returns different scaled input


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    config_module = importlib.import_module("fastgen.configs.methods.config_tcm")
    config = config_module.create_config()
    assert config.model.transition_t == 1.0

    instance = config.model
    opts = [
        "-",
        "img_resolution=2",
        "channel_mult=[1]",
        "channel_mult_noise=1",
        "r_timestep=False",
    ]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""  # disable ckpt loading
    instance.input_shape = [3, 2, 2]

    model = TCMModel(instance)
    model.on_train_begin()
    model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    neg_labels = torch.zeros(batch_size, 10)  # negative condition (unconditional)

    # Create mock data
    data = {
        "real": torch.randn(batch_size, 3, 2, 2).to(model.device, model.precision),
        "condition": labels.to(model.device, model.precision),
        "neg_condition": neg_labels.to(model.device, model.precision),
    }
    return model, data


@pytest.fixture
def get_tcm_precond_setup():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    """Setup for TCMPrecond tests"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create mock networks
    net_t = MockNetwork([3, 4, 4], device)
    net_t._is_teacher = True  # Mark as teacher for different behavior
    net_s = MockNetwork([3, 4, 4], device)

    # Create TCMPrecond
    transition_t = 2.0
    tcm_precond = TCMPrecond(net_t, net_s, transition_t)
    tcm_precond.to(device)

    # Test data
    batch_size = 4
    x_t = torch.randn(batch_size, 3, 4, 4, device=device)
    t = torch.tensor([1.0, 1.5, 2.5, 3.0], device=device)  # Mix of first/second stage
    condition = torch.randn(batch_size, 10, device=device)

    return tcm_precond, x_t, t, condition


def test_single_train_step_update(get_model_data):
    model, data = get_model_data
    # Run the training step
    assert model.config.transition_t == 1.0
    assert model.config.sample_t_cfg.train_p_mean == 0.0
    assert model.config.sample_t_cfg.train_p_std == 0.2
    assert model.net_tcm.net_s.training
    assert model.net_tcm.net_t.training

    # check intialization
    net_s_params = list(model.net_tcm.net_s.parameters())
    net_t_params = list(model.net_tcm.net_t.parameters())
    assert len([p for p in net_t_params if p.requires_grad]) == 0
    assert torch.allclose(net_s_params[0], net_t_params[0])
    assert torch.allclose(net_s_params[1], net_t_params[1])
    assert torch.allclose(net_s_params[-1], net_t_params[-1])

    # check single train step output
    loss_map, outputs = model.single_train_step(data, 0)

    # Assertions
    assert "total_loss" in loss_map
    assert "cm_loss" in loss_map
    assert "loss_boundary" in loss_map
    assert "unweighted_cm_loss" in loss_map
    assert "gen_rand" in outputs
    assert isinstance(outputs["gen_rand"], Callable)


def test_ct_ratio(get_model_data):
    model, _ = get_model_data

    # instantiate ct_schedule callback
    config_module = importlib.import_module("fastgen.configs.methods.config_tcm")
    config = config_module.create_config()
    ct_schedule_callback = instantiate(config.trainer.callbacks.ct_schedule)
    ct_schedule_callback.config = config

    # check ct ratio (starts at ratio_limit for 2nd stage of TCM)
    ct_schedule_callback.on_train_begin(model, iteration=100_000)
    assert model.ratio == config.trainer.callbacks.ct_schedule.ratio_limit


def test_optimizers(get_model_data):
    model, data = get_model_data
    # Test for net optimizer
    for iteration in range(2):
        model.optimizers_zero_grad(iteration)
        if iteration > 1:
            check_grad_zero(model.net)
        loss_map, _ = model.single_train_step(data, iteration)
        model.grad_scaler.scale(loss_map["total_loss"]).backward()
        model.optimizers_schedulers_step(iteration)


def test_tcm_precond_forward_mixed_stages(get_tcm_precond_setup):
    """Test forward pass with mixed first/second stage samples"""
    tcm_precond, x_t, t, condition = get_tcm_precond_setup

    # first: test in eval mode
    tcm_precond.eval()

    # t = [1.0, 1.5, 2.5, 3.0] with transition_t = 2.0
    # First two samples in first stage, last two in second stage

    with torch.no_grad():
        output = tcm_precond(x_t, t, condition=condition)

    # First two samples should use teacher (x_t * 0.5)
    # Last two samples should use student (x_t * 2.0)
    expected = x_t.clone()
    expected[:2] *= 0.5  # Teacher output for first stage
    expected[2:] *= 2.0  # Student output for second stage

    assert torch.allclose(output, expected, atol=1e-6)

    # second: test in training mode
    tcm_precond.train()

    # get random scale (should be shared across teacher and student) and reset RNG state
    with torch.random.fork_rng(devices=[x_t.device] if x_t.device.type == "cuda" else []):
        scale = torch.randn_like(x_t)

    with torch.no_grad():
        output = tcm_precond(x_t, t, condition=condition)

    # First two samples should use teacher (x_t * scale * 0.5)
    # Last two samples should use student (x_t * scale * 2.0)
    expected = x_t.clone()
    expected[:2] *= scale[:2] * 0.5  # Teacher output for first stage
    expected[2:] *= scale[2:] * 2.0  # Student output for second stage

    assert torch.allclose(output, expected, atol=1e-6)

    # simulate CM loss (expect same output due to shared randomness)
    device = x_t.device
    t_stage1 = torch.tensor([1.0, 1.5, 0.5, 0.25], device=device)  # First stage only
    t_stage2 = torch.tensor([2.5, 3.0, 4.0, 5.0], device=device)  # Second stage only
    for t_test in [t, t_stage1, t_stage2]:
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            pred_grad = tcm_precond(x_t, t_test, condition=condition)
        with torch.no_grad():
            pred_no_grad = tcm_precond(x_t, t_test, condition=condition)

        assert torch.allclose(pred_grad, pred_no_grad, atol=1e-6)
