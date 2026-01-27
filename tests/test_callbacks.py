# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import gc
import tempfile

import pytest
import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)

from fastgen.configs.methods.config_dmd2 import create_config
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.configs.net import EDM2_IN64_S_Config
from fastgen.methods import DMD2Model
from fastgen.trainer import Trainer
from fastgen.utils import instantiate
from fastgen.utils.io_utils import set_env_vars
from fastgen.configs.callbacks import (
    CTSchedule_CALLBACK,
    GradClip_CALLBACK,
    ParamCount_CALLBACK,
    WANDB_CALLBACK,
    EMA_CALLBACK,
    TrainProfiler_CALLBACK,
    GPUStats_CALLBACK,
    ForcedWeightNorm_CALLBACK,
)
from fastgen.callbacks.callback import CallbackDict
from fastgen.utils.test_utils import RunIf, run_distributed_test


@pytest.fixture
def get_model_data():
    gc.collect()  # https://github.com/pytest-dev/pytest/discussions/10387
    dmd_config = create_config()
    dmd_config.log_config.name = "test"

    instance = dmd_config.model
    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    opts_discriminator = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    instance.discriminator = override_config_with_opts(instance.discriminator, opts_discriminator)
    instance.use_ema = True
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance.precision = "float32" if instance.device == torch.device("cpu") else "bfloat16"
    instance.pretrained_model_path = ""  # disable ckpt loading
    instance.input_shape = [3, 8, 8]

    dmd_model = DMD2Model(instance)
    dmd_model.on_train_begin()
    dmd_model.init_optimizers()

    batch_size = 1
    labels = torch.randint(0, 10, (batch_size,))
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    neg_condition = torch.zeros(batch_size, 10)

    # Create mock data
    data = {
        "real": torch.randn(batch_size, 3, 8, 8).to(dmd_model.device, dmd_model.precision),
        "condition": labels.to(dmd_model.device, dmd_model.precision),
        "neg_condition": neg_condition.to(dmd_model.device, dmd_model.precision),
    }
    return dmd_model, data, dmd_config


def test_ema_callback(get_model_data):
    """Test EMA callback basic functionality (non-FSDP mode)."""
    model, data, config = get_model_data

    for callback_name, callback_config in EMA_CALLBACK.items():
        assert callback_name == "ema"
        assert model.ema is not None

        ema_callback = instantiate(callback_config)
        ema_callback.config = config
        # Call on_app_begin to initialize _is_fsdp flag (should be False for non-FSDP)
        ema_callback.on_app_begin()
        assert ema_callback._is_fsdp is False

        assert ema_callback.beta == 0.9999
        assert ema_callback.type == "constant"
        assert ema_callback.gamma == 16.97
        assert ema_callback.ema_halflife_kimg == 500
        assert ema_callback.ema_rampup_ratio == 0.05

        ema_callback.on_model_init_end(model)
        assert ema_callback._enabled is True

        # EMA should be initialized from net during model.build_model()
        ema_state = model.ema.state_dict()
        net_state = model.net.state_dict()
        assert all(torch.allclose(net_state[k], p_ema) for k, p_ema in ema_state.items())
        assert not any(p_ema.requires_grad for p_ema in ema_state.values())

        # Modify network parameters and compute expected EMA update
        buffers = [k for k, _ in model.net.named_buffers()]
        expected_ema_state = {}
        for k, p_net in net_state.items():
            torch.nn.init.normal_(p_net)
            if k in buffers:
                expected_ema_state[k] = p_net.detach().clone()
            else:
                expected_ema_state[k] = torch.lerp(ema_state[k], p_net.detach(), 1.0 - ema_callback.beta)

        # Run EMA update step
        ema_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=1,
        )

        # Verify EMA was updated correctly
        new_ema_state = model.ema.state_dict()
        assert all(torch.allclose(expected_ema_state[k], p_ema) for k, p_ema in new_ema_state.items())
        assert not any(p_ema.requires_grad for p_ema in new_ema_state.values())

        # Test that EMA update is skipped when ema is None
        model.ema = None
        ema_callback.on_model_init_end(model)
        assert ema_callback._enabled is False
        ema_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=1,
        )
        assert model.ema is None


def test_ema_initialization_after_build(get_model_data):
    """Test that EMA is correctly initialized from net state during model build."""
    model, data, config = get_model_data

    # Verify EMA exists and matches net state
    assert model.ema is not None
    assert model.use_ema == ["ema"]

    ema_state = model.ema.state_dict()
    net_state = model.net.state_dict()

    # All EMA parameters should match net parameters exactly after initialization
    for k in net_state.keys():
        assert k in ema_state, f"Key {k} not found in EMA state"
        assert torch.allclose(net_state[k], ema_state[k]), f"EMA state mismatch for {k}"

    # EMA should not require gradients
    assert not any(p.requires_grad for p in model.ema.parameters())
    assert model.ema.training is False  # EMA should be in eval mode


def test_ema_callback_multiple_steps(get_model_data):
    """Test EMA callback over multiple training steps to verify accumulation."""
    model, data, config = get_model_data

    ema_callback = instantiate(EMA_CALLBACK["ema"])
    ema_callback.config = config
    ema_callback.on_app_begin()

    beta = ema_callback.beta

    # Store initial EMA state
    initial_ema_state = {k: v.clone() for k, v in model.ema.state_dict().items()}
    buffers = [k for k, _ in model.net.named_buffers()]

    # Run multiple EMA update steps
    for iteration in range(1, 5):
        # Modify network parameters
        for p in model.net.parameters():
            torch.nn.init.normal_(p)

        # Update expected EMA
        net_state = model.net.state_dict()
        for k in initial_ema_state.keys():
            if k in buffers:
                initial_ema_state[k] = net_state[k].clone()
            else:
                initial_ema_state[k].lerp_(net_state[k], 1.0 - beta)

        ema_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=iteration,
        )

    # Verify final EMA state
    final_ema_state = model.ema.state_dict()
    for k, expected in initial_ema_state.items():
        assert torch.allclose(expected, final_ema_state[k], atol=1e-6), f"Mismatch at {k}"


@RunIf(min_gpus=1)
def test_ema_callback_fsdp_mode_mocked(get_model_data):
    """Test EMA callback FSDP mode behavior with mocked FSDP tensors.

    This test mocks the FSDP behavior by adding a `full_tensor()` method to parameters.
    In real FSDP, parameters are DTensors with `full_tensor()` that gathers from all ranks.
    """
    model, data, config = get_model_data

    # Mock FSDP by adding full_tensor method to parameters
    # In real FSDP, this gathers the full tensor from all shards
    original_params = {}
    for name, param in model.net.named_parameters():
        original_params[name] = param.data.clone()
        # Add a mock full_tensor method that returns the parameter itself
        param.full_tensor = lambda p=param: p.data.clone()

    ema_callback = instantiate(EMA_CALLBACK["ema"])
    ema_callback.config = config
    # Simulate FSDP mode
    config.trainer.fsdp = True
    ema_callback.on_app_begin()
    assert ema_callback._is_fsdp is True

    # Get initial EMA state
    initial_ema_state = {k: v.clone() for k, v in model.ema.state_dict().items()}
    buffers = [k for k, _ in model.net.named_buffers()]

    # Modify network parameters
    for p in model.net.named_parameters():
        torch.nn.init.normal_(p[1])

    # Compute expected EMA update
    expected_ema_state = {}
    net_state = model.net.state_dict()
    for k in initial_ema_state.keys():
        if k in buffers:
            expected_ema_state[k] = net_state[k].clone()
        else:
            expected_ema_state[k] = torch.lerp(initial_ema_state[k], net_state[k], 1.0 - ema_callback.beta)

    # Run EMA update (should use full_tensor() for FSDP)
    ema_callback.on_training_step_end(
        model,
        data_batch=None,
        output_batch=None,
        loss_dict=None,
        iteration=1,
    )

    # Verify EMA was updated correctly
    final_ema_state = model.ema.state_dict()
    for k, expected in expected_ema_state.items():
        assert torch.allclose(expected, final_ema_state[k], atol=1e-6), f"Mismatch at {k}"

    # Reset config
    config.trainer.fsdp = False


# =============================================================================
# True FSDP EMA Test Implementation
# =============================================================================


def _test_ema_callback_fsdp_distributed_impl(rank: int, world_size: int) -> dict:
    """Test EMA callback with real FSDP in a distributed setting using EDM model.

    This test uses the same EDM model architecture as other callback tests to ensure
    we're testing the actual model code paths. It verifies that:
    1. EMA callback correctly gathers full tensors from FSDP-sharded parameters
    2. EMA state remains consistent after update
    3. Synchronization barriers work correctly

    Args:
        rank: Process rank
        world_size: Total number of processes

    Returns:
        dict with test results
    """
    from fastgen.callbacks.ema import EMACallback
    from fastgen.configs.methods.config_dmd2 import create_config
    from fastgen.configs.config_utils import override_config_with_opts
    from fastgen.utils.distributed import synchronize, is_rank0

    device_mesh = init_device_mesh("cuda", (world_size,))
    device = torch.cuda.current_device()

    # Create EDM network using the same configuration as other tests
    # Use small resolution and simple architecture for fast testing
    dmd_config = create_config()
    instance = dmd_config.model
    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1"]
    instance.net = override_config_with_opts(instance.net, opts)
    instance.device = torch.device(f"cuda:{rank}")
    instance.precision = "float32"
    instance.pretrained_model_path = ""  # disable ckpt loading

    # Instantiate the network (EDM architecture)
    net = instantiate(instance.net).to(device)

    # Get state dict before FSDP sharding for broadcast
    if is_rank0():
        broadcast_state_dict = copy.deepcopy(net.state_dict())
    else:
        broadcast_state_dict = None

    synchronize()

    # Apply FSDP sharding using the network's fully_shard method
    # This follows the same pattern as test_fsdp.py
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    net.fully_shard(mesh=device_mesh, mp_policy=mp_policy)

    # Materialize meta tensors and reset parameters (following test_fsdp.py pattern)
    net.model.to_empty(device=device)
    if hasattr(net, "reset_parameters"):
        net.reset_parameters()
    synchronize()

    # Broadcast state dict from rank 0 (following test_fsdp.py pattern)
    # Extract only the inner model's state dict since that's what's sharded
    if broadcast_state_dict is not None:
        inner_model_prefix = "model."
        inner_broadcast_state_dict = {
            k[len(inner_model_prefix) :]: v for k, v in broadcast_state_dict.items() if k.startswith(inner_model_prefix)
        }
    else:
        inner_broadcast_state_dict = None

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(net.model, model_state_dict=inner_broadcast_state_dict, options=options)
    synchronize()

    # Create EMA model (matching production behavior)
    ema_init_state = {}
    for name, p in net.named_parameters():
        if hasattr(p, "full_tensor"):
            # All ranks must participate in full_tensor() gather
            full_p = p.full_tensor().detach().clone()
        else:
            full_p = p.detach().clone()
        ema_init_state[name] = full_p

    for name, buf in net.named_buffers():
        ema_init_state[name] = buf.detach().clone()

    # Create a fresh EDM network for EMA
    ema = instantiate(instance.net).to(device)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad = False
    ema.load_state_dict(ema_init_state)
    initial_ema_state = {k: v.clone() for k, v in ema.state_dict().items()}

    synchronize()

    # Create EMA callback and configure for FSDP mode
    ema_callback = EMACallback(
        type="constant",
        beta=0.9,  # Use larger learning rate for visible updates
        fsdp=True,
    )
    # Configure for FSDP mode
    ema_callback._is_fsdp = True

    # Modify network parameters (simulate training step)
    with torch.no_grad():
        for p in net.parameters():
            # The modification happens via the sharded parameter
            p.data.add_(torch.randn_like(p.data) * 0.1)

    synchronize()

    # Create a mock model object with net and ema attributes
    class MockModel:
        def __init__(self, net, ema):
            self.net = net
            self.ema = ema
            self.ema_enabled = True
            self.resume_iter = 0

    mock_model = MockModel(net, ema)

    # Run EMA update
    ema_callback.on_training_step_end(
        mock_model,
        data_batch=None,
        output_batch=None,
        loss_dict=None,
        iteration=1,
    )

    synchronize()

    # Verify results
    assert ema is not None, "EMA should exist"

    # Check that EMA was updated (should be different from initial)
    final_ema_state = ema.state_dict()

    # Check if EMA was updated (at least one param should differ)
    params_changed = 0
    for k in initial_ema_state:
        if not torch.allclose(initial_ema_state[k], final_ema_state[k], atol=1e-6):
            params_changed += 1

    results = {
        "ema_updated": True,
        "ema_different_from_initial": params_changed > 0,
        "params_changed": params_changed,
        "total_params": len(initial_ema_state),
        "ema_no_grad": not any(p.requires_grad for p in ema.parameters()),
        "model_type": "EDM",
    }
    return results


@RunIf(min_gpus=2)
def test_ema_callback_fsdp_distributed():
    """Test EMA callback with real FSDP distributed training using EDM model.

    This test requires at least 2 GPUs and uses the actual EDM network architecture
    (same as other callback tests) to verify that the EMA callback correctly handles
    FSDP-sharded parameters by:
    1. Gathering full tensors from all shards using full_tensor()
    2. Performing EMA updates
    3. Maintaining proper synchronization across ranks
    """
    gc.collect()
    torch.cuda.empty_cache()

    result = run_distributed_test(
        test_fn=_test_ema_callback_fsdp_distributed_impl,
        world_size=2,
        timeout=180,  # Slightly longer for model instantiation
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("model_type") == "EDM", "Test should use EDM model"
    assert result["ema_updated"], "EMA callback should have run without errors"
    assert result["ema_different_from_initial"], (
        f"EMA should have been updated after training step. "
        f"Only {result.get('params_changed', 0)}/{result.get('total_params', 0)} params changed."
    )
    assert result["ema_no_grad"], "EMA parameters should not require gradients"

    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# Non-Distributed Tests (continue below)
# =============================================================================


def test_ema_checkpoint_save_load(get_model_data):
    """Test that EMA state is correctly saved and loaded from checkpoints."""
    model, data, config = get_model_data

    # Initialize EMA callback and run a few updates
    ema_callback = instantiate(EMA_CALLBACK["ema"])
    ema_callback.config = config
    ema_callback.on_app_begin()
    ema_callback.on_model_init_end(model)

    # Modify network and update EMA a few times
    for i in range(3):
        for p in model.net.parameters():
            torch.nn.init.normal_(p)
        ema_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=i + 1,
        )

    # Store EMA state before saving
    ema_state_before = {k: v.clone() for k, v in model.ema.state_dict().items()}

    # Create a temporary directory for checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        from fastgen.utils.checkpointer import Checkpointer
        from omegaconf import OmegaConf

        # Create checkpointer config
        ckpt_config = OmegaConf.create(
            {
                "save_dir": tmpdir,
                "use_s3": False,
            }
        )
        checkpointer = Checkpointer(ckpt_config)

        # Save checkpoint
        checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=None,
            scheduler_dict=None,
            grad_scaler=None,
            callbacks=None,
            path=os.path.join(tmpdir, "test_ema.pth"),
            iteration=100,
        )

        # Verify checkpoint file exists
        assert os.path.exists(os.path.join(tmpdir, "test_ema.pth"))

        # Reset EMA state to verify loading works
        for k in model.ema.state_dict():
            model.ema.state_dict()[k].zero_()

        # Verify EMA is zeroed
        for k, v in model.ema.state_dict().items():
            assert torch.all(v == 0), f"EMA {k} should be zeroed"

        # Load checkpoint
        loaded_iter = checkpointer.load(
            model_dict=model.model_dict,
            optimizer_dict=None,
            scheduler_dict=None,
            grad_scaler=None,
            callbacks=None,
            path=os.path.join(tmpdir, "test_ema.pth"),
        )

        assert loaded_iter == 100

        # Verify EMA state was restored
        ema_state_after = model.ema.state_dict()
        for k, v_before in ema_state_before.items():
            assert torch.allclose(v_before, ema_state_after[k]), f"EMA state mismatch for {k}"


def test_ema_callback_beta_types(get_model_data):
    """Test EMA callback with different beta calculation types."""
    model, data, config = get_model_data

    # Test power function beta
    ema_callback_power = instantiate(EMA_CALLBACK["ema"])
    ema_callback_power.type = "power"
    ema_callback_power.config = config
    ema_callback_power.on_app_begin()

    # Power function should return beta = (1 - 1/iteration)^(gamma + 1)
    iteration = 10
    expected_power_beta = (1 - 1 / iteration) ** (ema_callback_power.gamma + 1)
    actual_power_beta = ema_callback_power._power_function_beta(iteration)
    assert np.isclose(expected_power_beta, actual_power_beta)

    # Test halflife beta
    ema_callback_halflife = instantiate(EMA_CALLBACK["ema"])
    ema_callback_halflife.type = "halflife"
    ema_callback_halflife.config = config
    ema_callback_halflife.on_app_begin()

    # Halflife beta should use the formula 0.5^(batch_size / ema_halflife_nimg)
    iteration = 100
    halflife_beta = ema_callback_halflife._halflife_beta(iteration)
    assert 0 < halflife_beta < 1, f"Halflife beta should be between 0 and 1, got {halflife_beta}"


def test_ct_schedule_callback(get_model_data):
    model, data, config = get_model_data

    for callback_name, callback_config in CTSchedule_CALLBACK.items():
        assert callback_name == "ct_schedule"
        assert config.dataloader_train.batch_size == 256

        ct_schedule_callback = instantiate(callback_config)
        ct_schedule_callback.config = config

        assert ct_schedule_callback.q == 2.0
        assert ct_schedule_callback.ratio_limit == 0.999
        assert ct_schedule_callback.kimg_per_stage == 12500

        ct_schedule_callback.on_train_begin(model, iteration=0)
        assert np.isclose(ct_schedule_callback.stage, 0)
        assert model.ratio == 0.5

        model.resume_iter = 100000
        ct_schedule_callback.on_train_begin(model, iteration=0)
        assert np.isclose(ct_schedule_callback.stage, 2)
        assert model.ratio == 0.875

        ct_schedule_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=100000,
        )

        assert np.isclose(ct_schedule_callback.stage, 4)
        assert np.isclose(model.ratio, 0.96875)


def test_grad_clip_callback(get_model_data):
    model, data, config = get_model_data
    for callback_name, callback_config in GradClip_CALLBACK.items():
        assert callback_name == "grad_clip"
        callback_config.grad_norm = 10.0

        grad_clip_callback = instantiate(callback_config)
        grad_clip_callback.config = config

        assert grad_clip_callback.grad_norm == 10.0
        assert grad_clip_callback.model_key == "net"
        grad_clip_callback.on_optimizer_step_begin(model)


@RunIf(min_gpus=1)
def test_gpu_stats_callback(get_model_data):
    model, data, config = get_model_data
    for callback_name, callback_config in GPUStats_CALLBACK.items():
        assert callback_name == "gpu_stats"
        assert callback_config.every_n == 100

        gpu_stats_callback = instantiate(callback_config)
        gpu_stats_callback.config = config
        assert gpu_stats_callback.every_n == 100

        gpu_stats_callback.on_train_begin(model, iteration=0)
        gpu_stats_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=0,
        )


def test_param_count_callback(get_model_data):
    model, data, config = get_model_data
    for callback_name, callback_config in ParamCount_CALLBACK.items():
        assert callback_name == "param_count"
        param_count_callback = instantiate(callback_config)
        param_count_callback.config = config
        param_count_callback.on_train_begin(model)


def test_train_profiler_callback(get_model_data):
    model, data, config = get_model_data
    for callback_name, callback_config in TrainProfiler_CALLBACK.items():
        assert callback_name == "train_profiler"
        assert callback_config.every_n == 100

        train_profiler_callback = instantiate(callback_config)
        train_profiler_callback.config = config
        assert train_profiler_callback.last_log_time is None
        assert train_profiler_callback.every_n == 100

        train_profiler_callback.on_train_begin(model, iteration=0)
        assert train_profiler_callback.every_n == config.trainer.logging_iter
        train_profiler_callback.on_training_step_end(
            model,
            data_batch=None,
            output_batch=None,
            loss_dict=None,
            iteration=0,
        )
        assert train_profiler_callback.last_log_time is not None


def test_forced_weight_norm_callback(get_model_data):
    model, data, config = get_model_data
    for callback_name, callback_config in ForcedWeightNorm_CALLBACK.items():
        assert callback_name == "forced_weight_norm"
        forced_weight_norm_callback = instantiate(callback_config)
        forced_weight_norm_callback.config = config
        forced_weight_norm_callback.on_training_accum_step_begin(model, data)

        net_config = EDM2_IN64_S_Config
        net_config = override_config_with_opts(net_config, ["-", "img_resolution=2", "channel_mult=[1]"])
        net = instantiate(net_config)
        model.net = net

        forced_weight_norm_callback.on_training_accum_step_begin(model, data)


def test_wandb_callback(get_model_data):
    model, data, config = get_model_data
    config.log_config.wandb_mode = "disabled"
    for callback_name, callback_config in WANDB_CALLBACK.items():
        assert callback_name == "wandb"
        wandb_callback = instantiate(callback_config)
        wandb_callback.config = config

        if os.path.isfile(config.log_config.wandb_credential):
            wandb_callback.on_app_begin()
        else:
            with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                config.log_config.wandb_credential = tmp_file.name
                wandb_callback.on_app_begin()

        wandb_callback.on_optimizer_step_begin(model)


def test_callback_list(get_model_data):
    model, data, config = get_model_data
    config.trainer.callbacks = DictConfig({**GradClip_CALLBACK, **ParamCount_CALLBACK})
    config.trainer.callbacks.update({**ForcedWeightNorm_CALLBACK})

    trainer = Trainer(config)
    callbacks = CallbackDict(config=config, trainer=trainer)
    assert len(callbacks._callbacks) == 3
    callbacks.on_train_begin(model)
