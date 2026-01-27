# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for checkpointing functionality.

These tests verify:
1. Standard checkpointing (Checkpointer class)
   - Model state dict save/load
   - Optimizer state save/load
   - Scheduler state save/load
   - Iteration number save/load
2. EMA state checkpointing
   - EMA weights are correctly saved and loaded
   - EMA states are independent from main network states
3. Teacher state exclusion
   - Teacher networks are NOT included in checkpoint state dicts
   - Models with teacher (e.g., DMD2) exclude teacher from saves
4. FSDP checkpointing (FSDPCheckpointer class)
   - Sharded checkpoint save/load for SFTModel
   - Sharded checkpoint save/load for multi-network models (DMD2Model)

Tests use small EDM networks for fast execution and temporary directories
for checkpoint storage.

Usage:
    pytest tests/test_checkpointing.py -v

    # Run specific tests:
    pytest tests/test_checkpointing.py::test_standard_checkpoint_save_load -v
    pytest tests/test_checkpointing.py::test_ema_checkpoint -v
    pytest tests/test_checkpointing.py::test_fsdp_checkpoint_sft -v
"""

import copy
import gc
import tempfile
import os

import torch
import torch.distributed as dist

from fastgen.methods import SFTModel, DMD2Model
from fastgen.configs.methods.config_sft import ModelConfig as SFTModelConfig
from fastgen.configs.methods.config_dmd2 import ModelConfig as DMD2ModelConfig
from fastgen.configs.net import EDM_CIFAR10_Config
from fastgen.configs.config_utils import override_config_with_opts
from fastgen.configs.config import BaseCheckpointerConfig
from fastgen.utils.checkpointer import Checkpointer, FSDPCheckpointer
from fastgen.utils.test_utils import RunIf, run_distributed_test
from fastgen.utils.io_utils import set_env_vars
from fastgen.utils.basic_utils import clear_gpu_memory


# =============================================================================
# Helper Functions
# =============================================================================


def _get_small_edm_config():
    """Get a small EDM config for fast testing."""
    config = copy.deepcopy(EDM_CIFAR10_Config)
    opts = ["-", "img_resolution=8", "channel_mult=[1]", "channel_mult_noise=1", "r_timestep=False"]
    return override_config_with_opts(config, opts)


def _get_small_discriminator_config():
    """Get a small discriminator config for DMD2 testing."""
    from fastgen.configs.discriminator import Discriminator_EDM_CIFAR10_Config

    opts = ["-", "feature_indices=[0]", "all_res=[8]", "in_channels=128"]
    return override_config_with_opts(copy.deepcopy(Discriminator_EDM_CIFAR10_Config), opts)


def _create_sft_model(use_ema: bool = False, device: str = "cpu") -> SFTModel:
    """Create a small SFT model for testing."""
    config = SFTModelConfig()
    config.net = _get_small_edm_config()
    config.input_shape = [3, 8, 8]
    config.precision = "float32"
    config.device = device
    config.pretrained_model_path = ""
    config.use_ema = use_ema

    model = SFTModel(config)
    model.on_train_begin()
    model.init_optimizers()
    return model


def _create_dmd2_model(use_ema: bool = False, device: str = "cpu", with_gan: bool = False) -> DMD2Model:
    """Create a small DMD2 model for testing."""
    config = DMD2ModelConfig()
    config.net = _get_small_edm_config()
    config.input_shape = [3, 8, 8]
    config.precision = "float32"
    config.device = device
    config.pretrained_model_path = ""
    config.use_ema = use_ema
    config.student_update_freq = 2

    # Disable GAN by default to simplify testing
    if with_gan:
        config.gan_loss_weight_gen = 0.001
        config.discriminator = _get_small_discriminator_config()
    else:
        config.gan_loss_weight_gen = 0.0

    # Don't add teacher to FSDP dict for standard checkpointing tests
    config.add_teacher_to_fsdp_dict = False

    model = DMD2Model(config)
    model.on_train_begin()
    model.init_optimizers()
    return model


def _modify_model_weights(model, scale: float = 2.0):
    """Modify model weights by scaling them."""
    with torch.no_grad():
        for param in model.net.parameters():
            param.mul_(scale)


def _compare_state_dicts(state_dict1, state_dict2, tolerance: float = 1e-6) -> bool:
    """Compare two state dicts and return True if they match."""
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]

        if tensor1.shape != tensor2.shape:
            return False

        if not torch.allclose(tensor1.float(), tensor2.float(), atol=tolerance):
            return False

    return True


def _get_checkpointer_config(save_dir: str) -> BaseCheckpointerConfig:
    """Create a checkpointer config with the given save directory."""
    config = BaseCheckpointerConfig()
    config.save_dir = save_dir
    config.use_s3 = False
    return config


# =============================================================================
# Standard Checkpointer Tests
# =============================================================================


def test_standard_checkpoint_save_load():
    """Test basic checkpoint save and load with Checkpointer class."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model and checkpointer
        model = _create_sft_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Save original state
        original_state = copy.deepcopy(model.net.state_dict())
        iteration = 100

        # Save checkpoint
        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            callbacks=None,
            iteration=iteration,
        )

        # Verify checkpoint file exists
        assert os.path.exists(path), f"Checkpoint file not found at {path}"

        # Modify weights
        _modify_model_weights(model, scale=2.0)
        modified_state = model.net.state_dict()
        assert not _compare_state_dicts(original_state, modified_state), "States should differ after modification"

        # Load checkpoint
        loaded_iteration = checkpointer.load(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            callbacks=None,
            path=path,
        )

        # Verify loaded state matches original
        loaded_state = model.net.state_dict()
        assert loaded_iteration == iteration, f"Iteration mismatch: {loaded_iteration} != {iteration}"
        assert _compare_state_dicts(original_state, loaded_state), "Loaded state doesn't match original"


def test_checkpoint_multiple_networks():
    """Test checkpoint save/load with multiple networks (DMD2Model)."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create DMD2 model (has net and fake_score)
        model = _create_dmd2_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Save original states
        original_net_state = copy.deepcopy(model.net.state_dict())
        original_fake_score_state = copy.deepcopy(model.fake_score.state_dict())
        iteration = 200

        # Save checkpoint
        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            iteration=iteration,
        )

        # Modify both networks
        _modify_model_weights(model, scale=2.0)
        with torch.no_grad():
            for param in model.fake_score.parameters():
                param.mul_(3.0)

        # Load checkpoint
        loaded_iteration = checkpointer.load(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            path=path,
        )

        # Verify both networks are restored
        assert loaded_iteration == iteration
        assert _compare_state_dicts(original_net_state, model.net.state_dict()), "Net state doesn't match"
        assert _compare_state_dicts(
            original_fake_score_state, model.fake_score.state_dict()
        ), "Fake score state doesn't match"


def test_optimizer_state_save_load():
    """Test that optimizer states are correctly saved and loaded."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_sft_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Run a training step to populate optimizer state
        batch_size = 2
        data = {
            "real": torch.randn(batch_size, 3, 8, 8),
            "condition": torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10).float(),
            "neg_condition": torch.zeros(batch_size, 10),
        }

        model.optimizers_zero_grad(0)
        loss_map, _ = model.single_train_step(data, 0)
        loss_map["total_loss"].backward()
        model.optimizers_schedulers_step(0)

        # Save optimizer state
        original_optimizer_state = copy.deepcopy(model.net_optimizer.state_dict())
        iteration = 50

        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            iteration=iteration,
        )

        # Create a fresh model and load the checkpoint
        fresh_model = _create_sft_model(use_ema=False, device="cpu")

        checkpointer.load(
            model_dict=fresh_model.model_dict,
            optimizer_dict=fresh_model.optimizer_dict,
            scheduler_dict=fresh_model.scheduler_dict,
            grad_scaler=fresh_model.grad_scaler,
            path=path,
        )

        # Verify optimizer state
        loaded_optimizer_state = fresh_model.net_optimizer.state_dict()
        assert (
            original_optimizer_state["param_groups"] == loaded_optimizer_state["param_groups"]
        ), "Optimizer param_groups mismatch"


# =============================================================================
# EMA Checkpointing Tests
# =============================================================================


def test_ema_checkpoint_save_load():
    """Test that EMA states are correctly saved and loaded."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model with EMA enabled
        model = _create_sft_model(use_ema=True, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Verify EMA exists
        assert hasattr(model, "ema"), "Model should have EMA network"
        assert model.ema is not None, "EMA network should not be None"

        # Save original EMA state
        original_ema_state = copy.deepcopy(model.ema.state_dict())
        iteration = 100

        # Save checkpoint
        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            iteration=iteration,
        )

        # Modify EMA weights
        with torch.no_grad():
            for param in model.ema.parameters():
                param.mul_(5.0)

        modified_ema_state = model.ema.state_dict()
        assert not _compare_state_dicts(original_ema_state, modified_ema_state), "EMA states should differ"

        # Load checkpoint
        checkpointer.load(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            path=path,
        )

        # Verify EMA state is restored
        loaded_ema_state = model.ema.state_dict()
        assert _compare_state_dicts(original_ema_state, loaded_ema_state), "EMA state should be restored"


def test_ema_independent_from_net():
    """Test that EMA and net states are independent in checkpoints."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_sft_model(use_ema=True, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Make EMA different from net
        with torch.no_grad():
            for ema_param, net_param in zip(model.ema.parameters(), model.net.parameters()):
                ema_param.copy_(net_param * 0.5)  # EMA is half of net

        original_net_state = copy.deepcopy(model.net.state_dict())
        original_ema_state = copy.deepcopy(model.ema.state_dict())

        # Verify they're different
        assert not _compare_state_dicts(original_net_state, original_ema_state), "Net and EMA should be different"

        # Save checkpoint
        path = checkpointer.save(
            model_dict=model.model_dict,
            iteration=100,
        )

        # Modify both
        _modify_model_weights(model, scale=3.0)
        with torch.no_grad():
            for param in model.ema.parameters():
                param.mul_(7.0)

        # Load checkpoint
        checkpointer.load(
            model_dict=model.model_dict,
            path=path,
        )

        # Verify both are restored to their original (different) states
        assert _compare_state_dicts(original_net_state, model.net.state_dict()), "Net state mismatch"
        assert _compare_state_dicts(original_ema_state, model.ema.state_dict()), "EMA state mismatch"
        assert not _compare_state_dicts(
            model.net.state_dict(), model.ema.state_dict()
        ), "Net and EMA should still be different"


# =============================================================================
# Teacher Exclusion Tests
# =============================================================================


def test_teacher_excluded_from_checkpoint():
    """Test that teacher network is NOT included in the saved checkpoint."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create DMD2 model which has a teacher network
        model = _create_dmd2_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Verify teacher exists
        assert hasattr(model, "teacher"), "DMD2 model should have teacher"
        assert model.teacher is not None, "Teacher should not be None"

        # Verify teacher is NOT in model_dict (which is what gets checkpointed)
        model_dict = model.model_dict
        assert "teacher" not in model_dict, "Teacher should NOT be in model_dict for checkpointing"

        # Verify other networks ARE in model_dict
        assert "net" in model_dict, "Net should be in model_dict"
        assert "fake_score" in model_dict, "Fake score should be in model_dict"

        # Save checkpoint
        iteration = 100
        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            iteration=iteration,
        )

        # Load the checkpoint file and verify teacher is not in it
        checkpoint = torch.load(path, weights_only=False)
        assert "teacher" not in checkpoint["model"], "Teacher should NOT be saved in checkpoint"
        assert "net" in checkpoint["model"], "Net should be in checkpoint"
        assert "fake_score" in checkpoint["model"], "Fake score should be in checkpoint"


def test_teacher_state_unchanged_after_load():
    """Test that loading a checkpoint doesn't affect teacher state."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_dmd2_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Save original teacher state
        original_teacher_state = copy.deepcopy(model.teacher.state_dict())

        # Save checkpoint (teacher not included)
        path = checkpointer.save(
            model_dict=model.model_dict,
            iteration=100,
        )

        # Modify teacher weights manually
        with torch.no_grad():
            for param in model.teacher.parameters():
                param.mul_(10.0)

        modified_teacher_state = copy.deepcopy(model.teacher.state_dict())
        assert not _compare_state_dicts(original_teacher_state, modified_teacher_state), "Teacher should be modified"

        # Load checkpoint - teacher should remain modified (not overwritten)
        checkpointer.load(
            model_dict=model.model_dict,
            path=path,
        )

        # Verify teacher is still in modified state (not overwritten by load)
        current_teacher_state = model.teacher.state_dict()
        assert _compare_state_dicts(
            modified_teacher_state, current_teacher_state
        ), "Teacher should remain modified after checkpoint load"


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_checkpoint_auto_resume():
    """Test automatic resume finding latest checkpoint."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_sft_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        # Save multiple checkpoints
        checkpointer.save(model_dict=model.model_dict, iteration=100)
        checkpointer.save(model_dict=model.model_dict, iteration=200)
        checkpointer.save(model_dict=model.model_dict, iteration=300)

        # Create fresh model
        fresh_model = _create_sft_model(use_ema=False, device="cpu")

        # Load without specifying path (should find latest)
        loaded_iteration = checkpointer.load(
            model_dict=fresh_model.model_dict,
            path=None,  # Provide the latest path directly
        )

        assert loaded_iteration == 300, f"Should load latest checkpoint, got iteration {loaded_iteration}"


def test_checkpoint_missing_keys_handling():
    """Test that loading handles missing keys gracefully."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model and save checkpoint
        model = _create_sft_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        path = checkpointer.save(
            model_dict=model.model_dict,
            iteration=100,
        )

        # Create a different model type (DMD2) and try to load
        # This simulates loading with missing keys
        different_model = _create_dmd2_model(use_ema=False, device="cpu")

        # Load should not raise an error, just warn about missing keys
        loaded_iteration = checkpointer.load(
            model_dict=different_model.model_dict,
            path=path,
        )

        assert loaded_iteration == 100, "Should still return correct iteration"


# =============================================================================
# FSDP Checkpointing Tests (require multiple GPUs)
# =============================================================================


def _test_fsdp_checkpoint_sft_impl(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
) -> dict:
    """Test FSDP checkpointing with SFTModel (single network)."""
    from fastgen.utils.distributed.fsdp import model_to_fsdp

    # Create model config
    config = SFTModelConfig()
    config.net = _get_small_edm_config()
    config.input_shape = [3, 8, 8]
    config.precision = "float32"
    config.device = "cuda"
    config.pretrained_model_path = ""
    config.use_ema = False
    config.fsdp_meta_init = True

    # Build model
    model = SFTModel(config)

    dist.barrier()

    # Apply FSDP
    model = model_to_fsdp(
        model,
        min_num_params=1_000_000,
        apply_cpu_offload=False,
        sync_module_states=True,
    )

    model.init_optimizers()

    dist.barrier()

    # Create FSDP checkpointer
    checkpointer_config = _get_checkpointer_config(checkpoint_dir)
    checkpointer = FSDPCheckpointer(checkpointer_config)

    # Get original net state for comparison (gather to rank 0)
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    original_state = get_model_state_dict(model.net, options=options)

    dist.barrier()

    # Save checkpoint
    iteration = 150
    path = checkpointer.save(
        model_dict=model.model_dict,
        optimizer_dict=model.optimizer_dict,
        scheduler_dict=model.scheduler_dict,
        grad_scaler=model.grad_scaler,
        iteration=iteration,
    )

    dist.barrier()

    # Modify weights
    with torch.no_grad():
        for param in model.net.parameters():
            if hasattr(param, "_local_tensor"):
                param._local_tensor.mul_(2.0)
            else:
                param.mul_(2.0)

    dist.barrier()

    # Load checkpoint
    loaded_iteration = checkpointer.load(
        model_dict=model.model_dict,
        optimizer_dict=model.optimizer_dict,
        scheduler_dict=model.scheduler_dict,
        grad_scaler=model.grad_scaler,
        path=path,
    )

    dist.barrier()

    # Verify state is restored
    loaded_state = get_model_state_dict(model.net, options=options)

    result = {
        "iteration_match": loaded_iteration == iteration,
        "loaded_iteration": loaded_iteration,
        "expected_iteration": iteration,
    }

    if rank == 0:
        # Compare states on rank 0
        result["state_match"] = _compare_state_dicts(original_state, loaded_state)
        result["all_passed"] = result["iteration_match"] and result["state_match"]
    else:
        result["state_match"] = True
        result["all_passed"] = True

    del model
    torch.cuda.empty_cache()

    return result


@RunIf(min_gpus=2)
def test_fsdp_checkpoint_sft():
    """Test FSDP checkpointing with SFTModel."""
    clear_gpu_memory()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_distributed_test(
            test_fn=_test_fsdp_checkpoint_sft_impl,
            world_size=2,
            timeout=300,
            setup_fn=set_env_vars,
            checkpoint_dir=tmpdir,
        )

        assert result is not None, "Test did not return a result"
        assert result.get("iteration_match", False), (
            f"Iteration mismatch: got {result.get('loaded_iteration')}, " f"expected {result.get('expected_iteration')}"
        )
        assert result.get("state_match", False), "Model state not correctly restored"
        assert result.get("all_passed", False), "FSDP checkpoint test failed"

    clear_gpu_memory()


def _test_fsdp_checkpoint_dmd2_impl(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
) -> dict:
    """Test FSDP checkpointing with DMD2Model (multiple networks)."""
    from fastgen.utils.distributed.fsdp import model_to_fsdp

    # Create DMD2 model config
    config = DMD2ModelConfig()
    config.net = _get_small_edm_config()
    config.input_shape = [3, 8, 8]
    config.precision = "float32"
    config.device = "cuda"
    config.pretrained_model_path = ""
    config.use_ema = False
    config.fsdp_meta_init = True
    config.gan_loss_weight_gen = 0.0  # Disable GAN for simpler testing
    config.add_teacher_to_fsdp_dict = True  # Include teacher in FSDP

    # Build model
    model = DMD2Model(config)

    dist.barrier()

    # Apply FSDP
    model = model_to_fsdp(
        model,
        min_num_params=1_000_000,
        apply_cpu_offload=False,
        sync_module_states=True,
    )

    model.init_optimizers()

    dist.barrier()

    # Create FSDP checkpointer
    checkpointer_config = _get_checkpointer_config(checkpoint_dir)
    checkpointer = FSDPCheckpointer(checkpointer_config)

    # Get original states for comparison
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    original_net_state = get_model_state_dict(model.net, options=options)
    original_fake_score_state = get_model_state_dict(model.fake_score, options=options)

    dist.barrier()

    # Save checkpoint
    iteration = 250
    path = checkpointer.save(
        model_dict=model.model_dict,
        optimizer_dict=model.optimizer_dict,
        scheduler_dict=model.scheduler_dict,
        grad_scaler=model.grad_scaler,
        iteration=iteration,
    )

    dist.barrier()

    # Modify weights for both networks
    with torch.no_grad():
        for param in model.net.parameters():
            if hasattr(param, "_local_tensor"):
                param._local_tensor.mul_(2.0)
            else:
                param.mul_(2.0)
        for param in model.fake_score.parameters():
            if hasattr(param, "_local_tensor"):
                param._local_tensor.mul_(3.0)
            else:
                param.mul_(3.0)

    dist.barrier()

    # Load checkpoint
    loaded_iteration = checkpointer.load(
        model_dict=model.model_dict,
        optimizer_dict=model.optimizer_dict,
        scheduler_dict=model.scheduler_dict,
        grad_scaler=model.grad_scaler,
        path=path,
    )

    dist.barrier()

    # Verify states are restored
    loaded_net_state = get_model_state_dict(model.net, options=options)
    loaded_fake_score_state = get_model_state_dict(model.fake_score, options=options)

    result = {
        "iteration_match": loaded_iteration == iteration,
        "loaded_iteration": loaded_iteration,
        "expected_iteration": iteration,
    }

    if rank == 0:
        result["net_state_match"] = _compare_state_dicts(original_net_state, loaded_net_state)
        result["fake_score_state_match"] = _compare_state_dicts(original_fake_score_state, loaded_fake_score_state)
        result["all_passed"] = (
            result["iteration_match"] and result["net_state_match"] and result["fake_score_state_match"]
        )
    else:
        result["net_state_match"] = True
        result["fake_score_state_match"] = True
        result["all_passed"] = True

    del model
    torch.cuda.empty_cache()

    return result


@RunIf(min_gpus=2)
def test_fsdp_checkpoint_dmd2():
    """Test FSDP checkpointing with DMD2Model (multiple networks)."""
    clear_gpu_memory()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_distributed_test(
            test_fn=_test_fsdp_checkpoint_dmd2_impl,
            world_size=2,
            timeout=300,
            setup_fn=set_env_vars,
            checkpoint_dir=tmpdir,
        )

        assert result is not None, "Test did not return a result"
        assert result.get("iteration_match", False), (
            f"Iteration mismatch: got {result.get('loaded_iteration')}, " f"expected {result.get('expected_iteration')}"
        )
        assert result.get("net_state_match", False), "Net state not correctly restored"
        assert result.get("fake_score_state_match", False), "Fake score state not correctly restored"
        assert result.get("all_passed", False), "FSDP checkpoint DMD2 test failed"

    clear_gpu_memory()


def _test_fsdp_ema_checkpoint_impl(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
) -> dict:
    """Test FSDP checkpointing with EMA enabled (only rank 0 has EMA)."""
    from fastgen.utils.distributed.fsdp import model_to_fsdp

    # Create model config with EMA
    config = SFTModelConfig()
    config.net = _get_small_edm_config()
    config.input_shape = [3, 8, 8]
    config.precision = "float32"
    config.device = "cuda"
    config.pretrained_model_path = ""
    config.use_ema = True
    config.fsdp_meta_init = True

    # Build model
    model = SFTModel(config)

    dist.barrier()

    # Apply FSDP
    model = model_to_fsdp(
        model,
        min_num_params=1_000_000,
        apply_cpu_offload=False,
        sync_module_states=True,
    )

    model.init_optimizers()

    dist.barrier()

    result = {"rank": rank}

    # Verify EMA exists
    assert hasattr(model, "ema") and model.ema is not None, "Model should have EMA"
    original_ema_state = copy.deepcopy(model.ema.state_dict())

    # Create FSDP checkpointer
    checkpointer_config = _get_checkpointer_config(checkpoint_dir)
    checkpointer = FSDPCheckpointer(checkpointer_config)

    dist.barrier()

    # Save checkpoint
    iteration = 175
    path = checkpointer.save(
        model_dict=model.model_dict,
        optimizer_dict=model.optimizer_dict,
        scheduler_dict=model.scheduler_dict,
        iteration=iteration,
    )

    dist.barrier()

    # Modify EMA weights
    with torch.no_grad():
        for param in model.ema.parameters():
            param.mul_(5.0)

    dist.barrier()

    # Load checkpoint
    loaded_iteration = checkpointer.load(
        model_dict=model.model_dict,
        optimizer_dict=model.optimizer_dict,
        scheduler_dict=model.scheduler_dict,
        path=path,
    )

    dist.barrier()

    result["iteration_match"] = loaded_iteration == iteration

    # Verify EMA state is restored
    loaded_ema_state = model.ema.state_dict()
    result["ema_state_match"] = _compare_state_dicts(original_ema_state, loaded_ema_state)
    result["all_passed"] = result["iteration_match"] and result["ema_state_match"]

    del model
    torch.cuda.empty_cache()

    return result


@RunIf(min_gpus=2)
def test_fsdp_ema_checkpoint():
    """Test FSDP checkpointing with EMA."""
    clear_gpu_memory()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_distributed_test(
            test_fn=_test_fsdp_ema_checkpoint_impl,
            world_size=2,
            timeout=300,
            setup_fn=set_env_vars,
            checkpoint_dir=tmpdir,
        )

        assert result is not None, "Test did not return a result"
        assert result.get("iteration_match", False), "Iteration mismatch in FSDP EMA checkpoint test"
        assert result.get("ema_state_match", False), "EMA state not correctly restored"
        assert result.get("all_passed", False), "FSDP EMA checkpoint test failed"

    clear_gpu_memory()


# =============================================================================
# Callback State Checkpointing Tests
# =============================================================================


def test_checkpoint_without_callbacks():
    """Test that checkpointing works correctly when callbacks=None."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_sft_model(use_ema=False, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        iteration = 100

        # Save without callbacks
        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            callbacks=None,
            iteration=iteration,
        )

        # Load without callbacks
        loaded_iteration = checkpointer.load(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            callbacks=None,
            path=path,
        )

        assert loaded_iteration == iteration, "Iteration mismatch"


def test_checkpoint_file_structure():
    """Test that checkpoint file has expected structure."""
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _create_sft_model(use_ema=True, device="cpu")
        checkpointer_config = _get_checkpointer_config(tmpdir)
        checkpointer = Checkpointer(checkpointer_config)

        iteration = 150

        path = checkpointer.save(
            model_dict=model.model_dict,
            optimizer_dict=model.optimizer_dict,
            scheduler_dict=model.scheduler_dict,
            grad_scaler=model.grad_scaler,
            callbacks=None,
            iteration=iteration,
        )

        # Load and verify structure
        checkpoint = torch.load(path, weights_only=False)

        # Verify top-level keys
        assert "model" in checkpoint, "Checkpoint should have 'model' key"
        assert "optimizer" in checkpoint, "Checkpoint should have 'optimizer' key"
        assert "scheduler" in checkpoint, "Checkpoint should have 'scheduler' key"
        assert "grad_scaler" in checkpoint, "Checkpoint should have 'grad_scaler' key"
        assert "iteration" in checkpoint, "Checkpoint should have 'iteration' key"

        # Verify model contains expected networks
        assert "net" in checkpoint["model"], "Checkpoint should contain 'net'"
        assert "ema" in checkpoint["model"], "Checkpoint should contain 'ema' (EMA is enabled)"

        # Verify iteration
        assert checkpoint["iteration"] == iteration, "Iteration mismatch"
