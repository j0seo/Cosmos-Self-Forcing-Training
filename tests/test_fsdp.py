# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for FSDP2 weight synchronization and forward pass correctness.

These tests verify:
1. FSDP weight synchronization from rank 0 to all ranks
2. Buffer synchronization across ranks (e.g., RoPE freqs_cos/freqs_sin)
3. Forward pass consistency between FSDP and reference models
4. State dict gathering and comparison
5. Meta device initialization for memory-efficient loading (fsdp_meta_init)
6. End-to-end model_to_fsdp code path with real model instantiation
7. Standard FSDP path without meta init (all ranks load independently)
8. Multi-network models (DMD2 with net, teacher, fake_score)

Tests require at least 2 GPUs and are run using torch.multiprocessing.spawn
to simulate distributed training.

Usage:
    pytest tests/test_fsdp.py -v

    # Or run with specific tests:
    pytest tests/test_fsdp.py::test_fsdp_weight_sync -v
    pytest tests/test_fsdp.py::test_fsdp_meta_device_init -v
    pytest tests/test_fsdp.py::test_model_to_fsdp_code_path -v
    pytest tests/test_fsdp.py::test_model_to_fsdp_no_meta_init -v
    pytest tests/test_fsdp.py::test_dmd2_model_to_fsdp -v
"""

import contextlib
import copy
from typing import Dict

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from fastgen.utils import instantiate
from fastgen.utils.test_utils import RunIf, run_distributed_test
from fastgen.utils.basic_utils import clear_gpu_memory
from fastgen.utils.io_utils import set_env_vars


# =============================================================================
# Network Loading and FSDP Utilities (using internal library code)
# =============================================================================


def _get_meta_init_context(fsdp_meta_init: bool):
    """Get context manager for FSDP meta initialization.

    When fsdp_meta_init is enabled, non-rank-0 processes use meta device
    for memory-efficient loading. Rank 0 loads weights normally, then
    FSDP syncs weights to other ranks via sync_module_states.

    Args:
        fsdp_meta_init: Whether to use meta initialization.

    Returns:
        Context manager (meta device for non-rank-0, nullcontext otherwise)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    use_meta = fsdp_meta_init and rank != 0
    if use_meta:
        return torch.device("meta")
    return contextlib.nullcontext()


def create_edm_network(fsdp_meta_init: bool = False, apply_checkpointing: bool = False):
    """Create an EDM CIFAR10 network using the internal library config.

    This is a small, fast network suitable for testing FSDP functionality
    without the overhead of loading large models like Wan.

    Args:
        fsdp_meta_init: If True, non-rank-0 processes use meta device
        apply_checkpointing: If True, enable gradient checkpointing (unused for EDM)

    Returns:
        Instantiated EDM network
    """
    from fastgen.configs.net import EDM_CIFAR10_Config

    # Clone config
    config = copy.deepcopy(EDM_CIFAR10_Config)
    # EDM doesn't have disable_grad_ckpt, but we keep the parameter for API compatibility

    # Use meta device context for non-rank-0 processes when fsdp_meta_init is True
    with _get_meta_init_context(fsdp_meta_init):
        network = instantiate(config)

    return network


def apply_fsdp_to_network(network, device_mesh):
    """Apply FSDP sharding to a FastGenNetwork.

    Uses the network's built-in fully_shard method.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )

    # Use the network's fully_shard method
    network.fully_shard(mesh=device_mesh, mp_policy=mp_policy)


def gather_fsdp_state_dict(model) -> Dict[str, torch.Tensor]:
    """Gather full state dict from FSDP model."""
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )
    return get_model_state_dict(model, options=options)


def generate_dummy_inputs(
    batch_size: int = 2,
    img_resolution: int = 32,
    img_channels: int = 3,
    label_dim: int = 10,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
):
    """Generate dummy input data for EDM CIFAR10 model forward pass."""
    if device is None:
        device = torch.cuda.current_device()

    generator = torch.Generator(device="cpu").manual_seed(42)

    # EDM expects image-shaped input
    x = torch.randn(batch_size, img_channels, img_resolution, img_resolution, generator=generator, dtype=dtype).to(
        device
    )

    # Timestep (sigma value for EDM)
    timestep = torch.tensor([0.5] * batch_size, dtype=dtype, device=device)

    # One-hot labels for class conditioning
    labels = torch.zeros(batch_size, label_dim, dtype=dtype, device=device)
    labels[:, 0] = 1.0  # Set first class as the label

    return {
        "x_t": x,
        "t": timestep,
        "condition": labels,
    }


def load_reference_network():
    """Load a fresh reference network on CPU for comparison.

    Note: This loads a fresh network with NEW random weights for any
    randomly-initialized layers. For tests that need to compare against
    the exact weights that were synced via FSDP, use the original
    broadcast_state_dict instead.
    """
    from fastgen.configs.net import EDM_CIFAR10_Config

    config = copy.deepcopy(EDM_CIFAR10_Config)

    # Load on CPU to avoid GPU memory issues
    network = instantiate(config)

    return network


# =============================================================================
# Test Functions (run inside distributed workers)
# =============================================================================


def _test_fsdp_weight_sync_impl(
    rank: int,
    world_size: int,
    apply_checkpointing: bool = False,
) -> Dict:
    """Test that FSDP weights are correctly synced from rank 0 to all ranks.

    This test:
    1. Rank 0 loads full model, other ranks use meta device
    2. Apply FSDP sharding with state dict broadcast
    3. Gather state dict back and compare to the ORIGINAL state dict from rank 0

    This verifies that ALL weights (including randomly-initialized ones like
    logvar_linear) are correctly synced via FSDP2's set_model_state_dict.
    """
    device_mesh = init_device_mesh("cuda", (world_size,))

    # Load network (rank 0 loads weights, others use meta device via fsdp_meta_init)
    network = create_edm_network(fsdp_meta_init=True, apply_checkpointing=apply_checkpointing)

    # Extract state dict from rank 0 BEFORE FSDP sharding
    # We need TWO copies:
    # - broadcast_state_dict: passed to set_model_state_dict (gets modified in place to DTensor)
    # - reference_state_dict: kept intact for comparison after gathering
    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
        # Create a separate reference copy with tensors moved to CPU to avoid DTensor issues
        reference_state_dict = {k: v.cpu().clone() for k, v in broadcast_state_dict.items()}
    else:
        broadcast_state_dict = None
        reference_state_dict = None

    dist.barrier()

    # Apply FSDP sharding using the network's method
    apply_fsdp_to_network(network, device_mesh)

    # Materialize meta tensors and reset parameters
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    # Broadcast state dict from rank 0
    # NOTE: This modifies broadcast_state_dict in place (tensors become DTensors)
    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()

    # Gather state dict back from FSDP model
    gathered_state_dict = gather_fsdp_state_dict(network)
    dist.barrier()

    # Only rank 0 performs comparison
    result = {"weights_match": True, "mismatches": [], "num_keys_compared": 0}
    if rank == 0:
        # Compare gathered state dict against the REFERENCE state dict
        # (the unmodified copy we kept before set_model_state_dict)
        reference_keys = set(reference_state_dict.keys())
        gathered_keys = set(gathered_state_dict.keys())
        common_keys = reference_keys & gathered_keys

        mismatches = []
        for key in common_keys:
            # reference_state_dict tensors are already on CPU
            reference_tensor = reference_state_dict[key].float()
            gathered_tensor = gathered_state_dict[key].cpu().float()

            if reference_tensor.shape != gathered_tensor.shape:
                mismatches.append((key, "shape_mismatch", str(reference_tensor.shape), str(gathered_tensor.shape)))
                continue

            max_diff = (reference_tensor - gathered_tensor).abs().max().item()
            if max_diff > 1e-5:
                mismatches.append((key, "value_mismatch", max_diff))

        # Also check for missing/extra keys
        missing_keys = reference_keys - gathered_keys
        extra_keys = gathered_keys - reference_keys
        for key in missing_keys:
            mismatches.append((key, "missing_in_gathered"))
        for key in extra_keys:
            mismatches.append((key, "extra_in_gathered"))

        result["weights_match"] = len(mismatches) == 0
        result["mismatches"] = mismatches
        result["num_keys_compared"] = len(common_keys)
        result["missing_keys"] = len(missing_keys)
        result["extra_keys"] = len(extra_keys)

        torch.cuda.empty_cache()

    return result


def _test_fsdp_tensor_sharding_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test that FSDP actually shards tensors across ranks.

    This test verifies that:
    1. After FSDP sharding, local tensor shapes differ from full tensor shapes
    2. The total number of local elements is less than full elements
    3. When gathered, the full shapes are restored

    This is important to verify that FSDP is actually sharding the model
    and not just wrapping it without sharding.
    """
    from torch.distributed.tensor import DTensor

    device_mesh = init_device_mesh("cuda", (world_size,))

    # Load network
    network = create_edm_network(fsdp_meta_init=True, apply_checkpointing=False)

    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
        # Store original shapes for comparison
        original_shapes = {k: v.shape for k, v in broadcast_state_dict.items()}
    else:
        broadcast_state_dict = None
        original_shapes = None

    dist.barrier()

    # Apply FSDP sharding
    apply_fsdp_to_network(network, device_mesh)
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()

    # Check that parameters are DTensors and have different local vs full shapes
    dtensor_params = []
    local_vs_full_diffs = []
    total_local_numel = 0
    total_full_numel = 0

    for name, param in network.named_parameters():
        if isinstance(param, DTensor):
            dtensor_params.append(name)
            local_shape = tuple(param._local_tensor.shape)
            full_shape = tuple(param.shape)
            local_numel = param._local_tensor.numel()
            full_numel = param.numel()

            total_local_numel += local_numel
            total_full_numel += full_numel

            # Record if shapes differ (they should for sharded params)
            if local_shape != full_shape:
                local_vs_full_diffs.append(
                    {
                        "name": name,
                        "local_shape": local_shape,
                        "full_shape": full_shape,
                        "local_numel": local_numel,
                        "full_numel": full_numel,
                    }
                )

    # Gather state dict and verify full shapes are restored
    gathered_state_dict = gather_fsdp_state_dict(network)
    dist.barrier()

    result = {
        "rank": rank,
        "world_size": world_size,
        "num_dtensor_params": len(dtensor_params),
        "num_params_with_shape_diff": len(local_vs_full_diffs),
        "total_local_numel": total_local_numel,
        "total_full_numel": total_full_numel,
        "local_vs_full_diffs": local_vs_full_diffs[:5],  # First 5 for debugging
    }

    # Verify that at least some parameters have different local vs full shapes
    # This confirms FSDP is actually sharding
    has_sharded_params = len(local_vs_full_diffs) > 0
    result["has_sharded_params"] = has_sharded_params

    # Verify local numel is less than full numel (sharding reduces per-rank memory)
    # For FSDP with world_size ranks, local should be approximately 1/world_size of full
    if total_full_numel > 0:
        sharding_ratio = total_local_numel / total_full_numel
        expected_ratio = 1.0 / world_size
        # Allow some tolerance since not all params may be sharded equally
        ratio_is_reasonable = sharding_ratio < 0.8  # Should be significantly less than 1.0
        result["sharding_ratio"] = sharding_ratio
        result["expected_ratio"] = expected_ratio
        result["ratio_is_reasonable"] = ratio_is_reasonable
    else:
        result["ratio_is_reasonable"] = False
        result["sharding_ratio"] = None

    # Verify gathered shapes match original shapes (on rank 0)
    if rank == 0 and original_shapes is not None:
        shapes_match = True
        shape_mismatches = []
        for key in gathered_state_dict:
            if key in original_shapes:
                gathered_shape = tuple(gathered_state_dict[key].shape)
                orig_shape = tuple(original_shapes[key])
                if gathered_shape != orig_shape:
                    shapes_match = False
                    shape_mismatches.append((key, orig_shape, gathered_shape))
        result["gathered_shapes_match"] = shapes_match
        result["shape_mismatches"] = shape_mismatches[:5]
    else:
        result["gathered_shapes_match"] = True
        result["shape_mismatches"] = []

    result["all_passed"] = (
        has_sharded_params and result.get("ratio_is_reasonable", False) and result.get("gathered_shapes_match", False)
    )

    return result


def _test_fsdp_buffer_sync_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test that FSDP buffers are correctly synced across ranks.

    This is important for non-persistent buffers like RoPE freqs_cos/freqs_sin
    which are NOT included in state_dict and need the reset_parameters() call.
    """
    device_mesh = init_device_mesh("cuda", (world_size,))

    # Load network
    network = create_edm_network(fsdp_meta_init=True, apply_checkpointing=False)

    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
    else:
        broadcast_state_dict = None

    dist.barrier()

    # Apply FSDP
    apply_fsdp_to_network(network, device_mesh)

    # Materialize and reset (reset_parameters reinitializes RoPE buffers)
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()

    # Check buffer synchronization across ranks
    buffer_max_diff = 0.0
    buffer_issues = []

    for name, buffer in network.named_buffers():
        # Clone rank 0's buffer and broadcast
        rank0_buffer = buffer.clone()
        dist.broadcast(rank0_buffer, src=0)

        # Compare local buffer to rank 0's buffer
        local_vs_rank0_diff = (buffer - rank0_buffer).abs().max().item()

        if local_vs_rank0_diff > buffer_max_diff:
            buffer_max_diff = local_vs_rank0_diff
        if local_vs_rank0_diff > 1e-6:
            buffer_issues.append((name, local_vs_rank0_diff))

    # Reduce buffer_max_diff across all ranks
    buffer_max_diff_tensor = torch.tensor([buffer_max_diff], device=torch.cuda.current_device())
    dist.all_reduce(buffer_max_diff_tensor, op=dist.ReduceOp.MAX)
    global_buffer_max_diff = buffer_max_diff_tensor.item()

    result = {
        "buffers_synced": global_buffer_max_diff < 1e-6,
        "max_diff": global_buffer_max_diff,
        "issues": buffer_issues if rank == 0 else [],
    }

    return result


def _test_fsdp_forward_pass_impl(
    rank: int,
    world_size: int,
    apply_checkpointing: bool = False,
) -> Dict:
    """Test that FSDP forward pass produces consistent results.

    Tests:
    1. FSDP forward is deterministic (same input -> same output)
    2. All ranks produce the same output
    3. FSDP output matches reference model output
    """
    device_mesh = init_device_mesh("cuda", (world_size,))

    # Load and setup FSDP model
    network = create_edm_network(fsdp_meta_init=True, apply_checkpointing=apply_checkpointing)

    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
    else:
        broadcast_state_dict = None

    dist.barrier()

    apply_fsdp_to_network(network, device_mesh)
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()

    # Generate inputs (same on all ranks due to fixed seed)
    inputs = generate_dummy_inputs(
        batch_size=2,
        img_resolution=32,
        img_channels=3,
        label_dim=10,
        device=torch.cuda.current_device(),
        dtype=torch.float32,
    )

    network.eval()

    # Test 1: FSDP forward is deterministic
    with torch.no_grad():
        output1 = network(
            x_t=inputs["x_t"],
            t=inputs["t"],
            condition=inputs["condition"],
            return_features_early=False,
            feature_indices=set(),
        )

        output2 = network(
            x_t=inputs["x_t"],
            t=inputs["t"],
            condition=inputs["condition"],
            return_features_early=False,
            feature_indices=set(),
        )

    consistency_diff = (output1.cpu() - output2.cpu()).abs().max().item()

    # Test 2: All ranks produce same output
    if hasattr(output1, "full_tensor"):
        output_full = output1.full_tensor()
    else:
        output_full = output1

    rank0_output = output_full.clone()
    dist.broadcast(rank0_output, src=0)
    rank_consistency_diff = (output_full - rank0_output).abs().max().item()

    fsdp_output_cpu = output_full.cpu().float()

    dist.barrier()

    # Test 3: Compare to reference (only on rank 0)
    result = {
        "deterministic": consistency_diff < 1e-6,
        "consistency_diff": consistency_diff,
        "ranks_consistent": rank_consistency_diff < 1e-6,
        "rank_consistency_diff": rank_consistency_diff,
    }

    if rank == 0:
        # Offload FSDP model
        network.to("cpu")
        torch.cuda.empty_cache()

        # Load reference model
        ref_network = load_reference_network()
        ref_network = ref_network.to(device=torch.cuda.current_device(), dtype=torch.float32)
        ref_network.eval()

        ref_inputs = {k: v.to(torch.cuda.current_device()) for k, v in inputs.items()}

        with torch.no_grad():
            ref_output = ref_network(
                x_t=ref_inputs["x_t"],
                t=ref_inputs["t"],
                condition=ref_inputs["condition"],
                return_features_early=False,
                feature_indices=set(),
            )

        ref_output_cpu = ref_output.cpu().float()
        del ref_network
        torch.cuda.empty_cache()

        # Compare outputs
        max_diff = (fsdp_output_cpu - ref_output_cpu).abs().max().item()
        mean_diff = (fsdp_output_cpu - ref_output_cpu).abs().mean().item()
        ref_norm = ref_output_cpu.abs().mean().item()
        relative_error = mean_diff / (ref_norm + 1e-8)

        result["max_diff"] = max_diff
        result["mean_diff"] = mean_diff
        result["relative_error"] = relative_error
        result["forward_matches"] = relative_error < 1e-3 and max_diff < 1e-2

    return result


def _test_fsdp_meta_device_init_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test that fsdp_meta_init correctly uses meta device for non-rank-0 processes.

    This test verifies:
    1. Rank 0 loads real tensors on CPU/CUDA
    2. Non-rank-0 processes have tensors on meta device
    3. After FSDP sync, all ranks have real tensors
    """
    # Load network with fsdp_meta_init=True
    network = create_edm_network(fsdp_meta_init=True, apply_checkpointing=False)

    # Check tensor devices BEFORE FSDP sharding
    meta_tensors_before = []
    real_tensors_before = []

    for name, param in network.named_parameters():
        if param.device.type == "meta":
            meta_tensors_before.append(name)
        else:
            real_tensors_before.append(name)

    # Verify meta device usage based on rank
    if rank == 0:
        # Rank 0 should have real tensors (not meta)
        has_correct_device_before = len(meta_tensors_before) == 0
    else:
        # Non-rank-0 should have meta tensors
        has_correct_device_before = len(meta_tensors_before) > 0 and len(real_tensors_before) == 0

    result = {
        "rank": rank,
        "meta_tensors_before_fsdp": len(meta_tensors_before),
        "real_tensors_before_fsdp": len(real_tensors_before),
        "has_correct_device_before": has_correct_device_before,
    }

    # Now test that after FSDP sharding and sync, all tensors are materialized
    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
    else:
        broadcast_state_dict = None

    dist.barrier()

    device_mesh = init_device_mesh("cuda", (world_size,))
    apply_fsdp_to_network(network, device_mesh)

    # Materialize meta tensors and reset parameters
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    # Broadcast state dict from rank 0
    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()

    # Check tensor devices AFTER FSDP sync
    meta_tensors_after = []
    real_tensors_after = []

    for name, param in network.named_parameters():
        # After FSDP, parameters may be DTensor or regular tensor on CUDA
        # Check the underlying device
        if hasattr(param, "_local_tensor"):
            # DTensor case
            device = param._local_tensor.device
        else:
            device = param.device

        if device.type == "meta":
            meta_tensors_after.append(name)
        else:
            real_tensors_after.append(name)

    # After FSDP sync, no tensors should be on meta device
    has_correct_device_after = len(meta_tensors_after) == 0

    result["meta_tensors_after_fsdp"] = len(meta_tensors_after)
    result["real_tensors_after_fsdp"] = len(real_tensors_after)
    result["has_correct_device_after"] = has_correct_device_after
    result["all_passed"] = has_correct_device_before and has_correct_device_after

    return result


def _test_fsdp_gathered_weights_forward_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test that a model loaded with gathered FSDP weights produces correct output.

    This isolates whether issues come from FSDP forward computation vs weight gathering.
    """
    device_mesh = init_device_mesh("cuda", (world_size,))

    # Setup FSDP model
    network = create_edm_network(fsdp_meta_init=True, apply_checkpointing=False)

    if rank == 0:
        broadcast_state_dict = copy.deepcopy(network.state_dict())
    else:
        broadcast_state_dict = None

    dist.barrier()

    apply_fsdp_to_network(network, device_mesh)
    network.to_empty(device=torch.cuda.current_device())
    network.reset_parameters()
    dist.barrier()

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    set_model_state_dict(network, model_state_dict=broadcast_state_dict, options=options)
    dist.barrier()

    # Gather state dict
    gathered_state_dict = gather_fsdp_state_dict(network)
    dist.barrier()

    result = {}

    if rank == 0:
        # Generate inputs
        inputs = generate_dummy_inputs(
            batch_size=2,
            img_resolution=32,
            img_channels=3,
            label_dim=10,
            device=torch.cuda.current_device(),
            dtype=torch.float32,
        )

        # Load fresh model with gathered weights
        gathered_network = load_reference_network()
        gathered_network = gathered_network.to(device=torch.cuda.current_device(), dtype=torch.float32)

        missing, unexpected = gathered_network.load_state_dict(gathered_state_dict, strict=False)

        gathered_network.eval()
        with torch.no_grad():
            gathered_output = gathered_network(
                x_t=inputs["x_t"],
                t=inputs["t"],
                condition=inputs["condition"],
                return_features_early=False,
                feature_indices=set(),
            )

        gathered_output_cpu = gathered_output.cpu().float()
        del gathered_network
        torch.cuda.empty_cache()

        # Load reference model
        ref_network = load_reference_network()
        ref_network = ref_network.to(device=torch.cuda.current_device(), dtype=torch.float32)

        ref_network.eval()
        with torch.no_grad():
            ref_output = ref_network(
                x_t=inputs["x_t"],
                t=inputs["t"],
                condition=inputs["condition"],
                return_features_early=False,
                feature_indices=set(),
            )

        ref_output_cpu = ref_output.cpu().float()
        del ref_network
        torch.cuda.empty_cache()

        # Compare
        max_diff = (gathered_output_cpu - ref_output_cpu).abs().max().item()
        mean_diff = (gathered_output_cpu - ref_output_cpu).abs().mean().item()

        result = {
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "matches": max_diff < 1e-4,
        }

    return result


# =============================================================================
# Pytest Test Functions
# =============================================================================


@RunIf(min_gpus=2)
def test_fsdp_weight_sync():
    """Test that FSDP correctly synchronizes weights from rank 0 to all ranks."""
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_weight_sync_impl,
        world_size=2,
        setup_fn=set_env_vars,
        apply_checkpointing=False,
    )

    assert result is not None, "Test did not return a result"
    assert result[
        "weights_match"
    ], f"Weight sync failed with {len(result['mismatches'])} mismatches: {result['mismatches'][:5]}"

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_weight_sync_with_checkpointing():
    """Test FSDP weight sync with activation checkpointing enabled."""
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_weight_sync_impl,
        world_size=2,
        setup_fn=set_env_vars,
        apply_checkpointing=True,
    )

    assert result is not None, "Test did not return a result"
    assert result["weights_match"], f"Weight sync with checkpointing failed: {result['mismatches'][:5]}"

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_tensor_sharding():
    """Test that FSDP actually shards tensors across ranks.

    This verifies that:
    - Local tensor shapes differ from full tensor shapes for sharded parameters
    - The sharding ratio is reasonable (local < full)
    - Gathered shapes match original shapes
    """
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_tensor_sharding_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("has_sharded_params", False), (
        f"No parameters were sharded. num_dtensor_params={result.get('num_dtensor_params', 'N/A')}, "
        f"num_params_with_shape_diff={result.get('num_params_with_shape_diff', 'N/A')}"
    )
    assert result.get("ratio_is_reasonable", False), (
        f"Sharding ratio is not reasonable: {result.get('sharding_ratio', 'N/A')} "
        f"(expected ~{result.get('expected_ratio', 'N/A')})"
    )
    assert result.get(
        "gathered_shapes_match", False
    ), f"Gathered shapes don't match original shapes: {result.get('shape_mismatches', 'N/A')}"
    assert result.get("all_passed", False), (
        f"Tensor sharding test failed. Details: " f"local_vs_full_diffs={result.get('local_vs_full_diffs', 'N/A')}"
    )

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_buffer_sync():
    """Test that FSDP correctly synchronizes buffers (e.g., RoPE) across ranks."""
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_buffer_sync_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result["buffers_synced"], f"Buffer sync failed with max_diff={result['max_diff']}: {result['issues']}"

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_forward_deterministic():
    """Test that FSDP forward pass is deterministic."""
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_forward_pass_impl,
        world_size=2,
        setup_fn=set_env_vars,
        apply_checkpointing=False,
    )

    assert result is not None, "Test did not return a result"
    assert result["deterministic"], f"FSDP forward not deterministic: diff={result['consistency_diff']}"

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_forward_rank_consistency():
    """Test that all FSDP ranks produce the same output."""
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_forward_pass_impl,
        world_size=2,
        setup_fn=set_env_vars,
        apply_checkpointing=False,
    )

    assert result is not None, "Test did not return a result"
    assert result["ranks_consistent"], f"Ranks inconsistent: diff={result['rank_consistency_diff']}"

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_forward_matches_reference():
    """Test that FSDP forward pass matches reference model output."""
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_forward_pass_impl,
        world_size=2,
        setup_fn=set_env_vars,
        apply_checkpointing=False,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("forward_matches", False), (
        f"FSDP forward doesn't match reference: "
        f"max_diff={result.get('max_diff', 'N/A')}, "
        f"relative_error={result.get('relative_error', 'N/A')}"
    )

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_gathered_weights_forward():
    """Test that model loaded with gathered FSDP weights produces correct output."""
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_gathered_weights_forward_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("matches", False), (
        f"Gathered weights model doesn't match reference: "
        f"max_diff={result.get('max_diff', 'N/A')}, "
        f"missing_keys={result.get('missing_keys', 'N/A')}"
    )

    clear_gpu_memory()


@RunIf(min_gpus=2)
def test_fsdp_meta_device_init():
    """Test that fsdp_meta_init correctly uses meta device for non-rank-0 processes.

    This verifies:
    - Rank 0 loads real tensors (not meta)
    - Non-rank-0 processes use meta device for memory-efficient loading
    - After FSDP sync, all tensors are properly materialized
    """
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_meta_device_init_impl,
        world_size=2,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("has_correct_device_before", False), (
        f"Meta device not used correctly before FSDP: "
        f"rank={result.get('rank', 'N/A')}, "
        f"meta_tensors={result.get('meta_tensors_before_fsdp', 'N/A')}, "
        f"real_tensors={result.get('real_tensors_before_fsdp', 'N/A')}"
    )
    assert result.get("has_correct_device_after", False), (
        f"Tensors not properly materialized after FSDP: " f"meta_tensors={result.get('meta_tensors_after_fsdp', 'N/A')}"
    )

    clear_gpu_memory()


# =============================================================================
# End-to-End Model FSDP Test (testing model_to_fsdp code path)
# =============================================================================


def _test_model_to_fsdp_impl(
    rank: int,
    world_size: int,
    sharding_group_size: int = None,
) -> Dict:
    """Test the complete model_to_fsdp code path with a real model.

    This test verifies:
    1. Model instantiation with fsdp_meta_init=True
    2. meta device usage for non-rank-0 processes
    3. model_to_fsdp correctly wraps and syncs the model
    4. After FSDP, all tensors are properly materialized
    5. Forward pass produces consistent results across ranks

    Args:
        sharding_group_size: If set, creates a 2D mesh with (replicate, shard) dimensions.
    """
    import copy
    from fastgen.configs.methods.config_sft import ModelConfig
    from fastgen.configs.net import EDM_CIFAR10_Config
    from fastgen.methods import SFTModel
    from fastgen.utils.distributed.fsdp import model_to_fsdp

    # Create a minimal model config with EDM (small, fast network)
    config = ModelConfig()
    config.net = copy.deepcopy(EDM_CIFAR10_Config)
    config.input_shape = [3, 32, 32]
    config.precision = "float32"
    config.device = "cuda"
    config.fsdp_meta_init = True
    config.pretrained_model_path = ""  # No pretrained weights needed
    config.use_ema = False  # Disable EMA for simpler testing

    result = {
        "rank": rank,
        "world_size": world_size,
        "sharding_group_size": sharding_group_size,
    }

    # Step 1: Build model with fsdp_meta_init=True
    # The model's build_model() uses _get_meta_init_context() which wraps
    # network instantiation in meta device context for non-rank-0 processes
    model = SFTModel(config)

    # Step 2: Verify meta device usage BEFORE model_to_fsdp
    meta_tensors_before = []
    real_tensors_before = []

    for name, param in model.net.named_parameters():
        if param.device.type == "meta":
            meta_tensors_before.append(name)
        else:
            real_tensors_before.append(name)

    if rank == 0:
        # Rank 0 should have real tensors (not meta)
        has_correct_device_before = len(meta_tensors_before) == 0
    else:
        # Non-rank-0 should have meta tensors
        has_correct_device_before = len(meta_tensors_before) > 0 and len(real_tensors_before) == 0

    result["meta_tensors_before_fsdp"] = len(meta_tensors_before)
    result["real_tensors_before_fsdp"] = len(real_tensors_before)
    result["has_correct_device_before"] = has_correct_device_before

    dist.barrier()

    # Step 3: Apply model_to_fsdp (the actual production code path)
    # This tests the full FSDP wrapping including state dict broadcast
    model = model_to_fsdp(
        model,
        min_num_params=1_000_000,  # Lower threshold for the smaller EDM network
        apply_cpu_offload=False,
        sync_module_states=True,  # This is the key: broadcast from rank 0
        sharding_group_size=sharding_group_size,
    )

    dist.barrier()

    # Step 4: Verify tensors are materialized AFTER model_to_fsdp
    meta_tensors_after = []
    real_tensors_after = []

    for name, param in model.net.named_parameters():
        # After FSDP, parameters may be DTensor or regular tensor on CUDA
        if hasattr(param, "_local_tensor"):
            device = param._local_tensor.device
        else:
            device = param.device

        if device.type == "meta":
            meta_tensors_after.append(name)
        else:
            real_tensors_after.append(name)

    has_correct_device_after = len(meta_tensors_after) == 0
    result["meta_tensors_after_fsdp"] = len(meta_tensors_after)
    result["real_tensors_after_fsdp"] = len(real_tensors_after)
    result["has_correct_device_after"] = has_correct_device_after

    # Step 5: Test forward pass consistency across ranks
    model.net.eval()

    # Generate same random input on all ranks
    generator = torch.Generator(device="cpu").manual_seed(42)
    dummy_input = torch.randn(2, 3, 32, 32, generator=generator, dtype=torch.float32).cuda()
    dummy_t = torch.tensor([0.5, 0.5], dtype=torch.float32, device="cuda")
    dummy_labels = torch.zeros(2, 10, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        output = model.net(dummy_input, dummy_t, condition=dummy_labels)

    # Compare output across ranks
    if hasattr(output, "full_tensor"):
        output_full = output.full_tensor()
    else:
        output_full = output

    rank0_output = output_full.clone()
    dist.broadcast(rank0_output, src=0)
    rank_consistency_diff = (output_full - rank0_output).abs().max().item()

    result["forward_rank_consistent"] = rank_consistency_diff < 1e-5
    result["rank_consistency_diff"] = rank_consistency_diff
    result["all_passed"] = has_correct_device_before and has_correct_device_after and result["forward_rank_consistent"]

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return result


@RunIf(min_gpus=2)
def test_model_to_fsdp_code_path():
    """Test the complete model_to_fsdp code path with a real model.

    This is an end-to-end test that verifies:
    - Model builds correctly with fsdp_meta_init=True
    - Non-rank-0 processes use meta device
    - model_to_fsdp correctly wraps and syncs weights
    - Forward pass is consistent across ranks
    """
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_model_to_fsdp_impl,
        world_size=2,
        timeout=300,  # Allow more time for model instantiation
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("has_correct_device_before", False), (
        f"Meta device not used correctly before FSDP: "
        f"rank={result.get('rank', 'N/A')}, "
        f"meta_tensors={result.get('meta_tensors_before_fsdp', 'N/A')}, "
        f"real_tensors={result.get('real_tensors_before_fsdp', 'N/A')}"
    )
    assert result.get("has_correct_device_after", False), (
        f"Tensors not properly materialized after model_to_fsdp: "
        f"meta_tensors={result.get('meta_tensors_after_fsdp', 'N/A')}"
    )
    assert result.get("forward_rank_consistent", False), (
        f"Forward pass inconsistent across ranks: " f"diff={result.get('rank_consistency_diff', 'N/A')}"
    )
    assert result.get("all_passed", False), "model_to_fsdp test failed"

    clear_gpu_memory()


@RunIf(min_gpus=4)
def test_model_to_fsdp_sharding_group():
    """Test model_to_fsdp with a sharding group size (2D mesh).

    This test verifies FSDP with a 2D device mesh where:
    - sharding_group_size=2 means weights are sharded within groups of 2 GPUs
    - The mesh has (replicate, shard) dimensions

    Requires at least 4 GPUs and world_size must be divisible by 2.
    """
    import pytest

    clear_gpu_memory()

    # Check if GPU count is divisible by sharding_group_size
    num_gpus = torch.cuda.device_count()
    sharding_group_size = 2
    if num_gpus % sharding_group_size != 0:
        pytest.skip(f"Number of GPUs ({num_gpus}) not divisible by sharding_group_size ({sharding_group_size})")

    result = run_distributed_test(
        test_fn=_test_model_to_fsdp_impl,
        world_size=4,
        timeout=300,
        setup_fn=set_env_vars,
        sharding_group_size=sharding_group_size,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("has_correct_device_before", False), (
        f"Meta device not used correctly before FSDP: "
        f"rank={result.get('rank', 'N/A')}, "
        f"meta_tensors={result.get('meta_tensors_before_fsdp', 'N/A')}, "
        f"real_tensors={result.get('real_tensors_before_fsdp', 'N/A')}"
    )
    assert result.get("has_correct_device_after", False), (
        f"Tensors not properly materialized after model_to_fsdp: "
        f"meta_tensors={result.get('meta_tensors_after_fsdp', 'N/A')}"
    )
    assert result.get("forward_rank_consistent", False), (
        f"Forward pass inconsistent across ranks: " f"diff={result.get('rank_consistency_diff', 'N/A')}"
    )
    assert result.get("all_passed", False), "model_to_fsdp sharding group test failed"

    clear_gpu_memory()


def _test_model_to_fsdp_no_meta_init_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test model_to_fsdp code path WITHOUT meta init.

    This test verifies that the standard (non-meta-init) FSDP path works:
    1. All ranks load real tensors (no meta device)
    2. model_to_fsdp correctly wraps the model
    3. Forward pass produces consistent results across ranks
    """
    import copy
    from fastgen.configs.methods.config_sft import ModelConfig
    from fastgen.configs.net import EDM_CIFAR10_Config
    from fastgen.methods import SFTModel
    from fastgen.utils.distributed.fsdp import model_to_fsdp

    # Create a minimal model config with EDM (small, fast network)
    # fsdp_meta_init=False means all ranks load their own weights
    config = ModelConfig()
    config.net = copy.deepcopy(EDM_CIFAR10_Config)
    config.input_shape = [3, 32, 32]
    config.precision = "float32"
    config.device = "cuda"
    config.fsdp_meta_init = False  # All ranks load weights independently
    config.pretrained_model_path = ""
    config.use_ema = False

    result = {
        "rank": rank,
        "world_size": world_size,
    }

    # Step 1: Build model with fsdp_meta_init=False
    model = SFTModel(config)

    # Step 2: Verify all ranks have real tensors (no meta device)
    meta_tensors_before = []
    real_tensors_before = []

    for name, param in model.net.named_parameters():
        if param.device.type == "meta":
            meta_tensors_before.append(name)
        else:
            real_tensors_before.append(name)

    # All ranks should have real tensors (not meta)
    has_correct_device_before = len(meta_tensors_before) == 0
    result["meta_tensors_before_fsdp"] = len(meta_tensors_before)
    result["real_tensors_before_fsdp"] = len(real_tensors_before)
    result["has_correct_device_before"] = has_correct_device_before

    dist.barrier()

    # Step 3: Apply model_to_fsdp without sync_module_states
    # (since all ranks already have their own weights)
    model = model_to_fsdp(
        model,
        min_num_params=1_000_000,
        apply_cpu_offload=False,
        sync_module_states=False,  # No need to sync - all ranks have weights
    )

    dist.barrier()

    # Step 4: Verify tensors are on CUDA after FSDP
    meta_tensors_after = []
    real_tensors_after = []

    for name, param in model.net.named_parameters():
        if hasattr(param, "_local_tensor"):
            device = param._local_tensor.device
        else:
            device = param.device

        if device.type == "meta":
            meta_tensors_after.append(name)
        else:
            real_tensors_after.append(name)

    has_correct_device_after = len(meta_tensors_after) == 0
    result["meta_tensors_after_fsdp"] = len(meta_tensors_after)
    result["real_tensors_after_fsdp"] = len(real_tensors_after)
    result["has_correct_device_after"] = has_correct_device_after

    # Step 5: Test forward pass consistency across ranks
    model.net.eval()

    generator = torch.Generator(device="cpu").manual_seed(42)
    dummy_input = torch.randn(2, 3, 32, 32, generator=generator, dtype=torch.float32).cuda()
    dummy_t = torch.tensor([0.5, 0.5], dtype=torch.float32, device="cuda")
    dummy_labels = torch.zeros(2, 10, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        output = model.net(dummy_input, dummy_t, condition=dummy_labels)

    if hasattr(output, "full_tensor"):
        output_full = output.full_tensor()
    else:
        output_full = output

    rank0_output = output_full.clone()
    dist.broadcast(rank0_output, src=0)
    rank_consistency_diff = (output_full - rank0_output).abs().max().item()

    result["forward_rank_consistent"] = rank_consistency_diff < 1e-5
    result["rank_consistency_diff"] = rank_consistency_diff
    result["all_passed"] = has_correct_device_before and has_correct_device_after and result["forward_rank_consistent"]

    del model
    torch.cuda.empty_cache()

    return result


@RunIf(min_gpus=2)
def test_model_to_fsdp_no_meta_init():
    """Test model_to_fsdp code path WITHOUT meta init.

    This verifies the standard FSDP path where all ranks load weights independently.
    """
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_model_to_fsdp_no_meta_init_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("has_correct_device_before", False), (
        f"All ranks should have real tensors before FSDP: "
        f"meta_tensors={result.get('meta_tensors_before_fsdp', 'N/A')}"
    )
    assert result.get("has_correct_device_after", False), (
        f"Tensors not on CUDA after model_to_fsdp: " f"meta_tensors={result.get('meta_tensors_after_fsdp', 'N/A')}"
    )
    assert result.get("forward_rank_consistent", False), (
        f"Forward pass inconsistent across ranks: " f"diff={result.get('rank_consistency_diff', 'N/A')}"
    )

    clear_gpu_memory()


def _test_dmd2_model_to_fsdp_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Test model_to_fsdp with DMD2Model which has multiple networks.

    DMD2 has:
    - net: student network
    - teacher: frozen teacher network
    - fake_score: trainable fake score network

    This tests that FSDP correctly handles multiple networks in fsdp_dict.
    """
    import copy
    from fastgen.configs.methods.config_dmd2 import ModelConfig
    from fastgen.configs.net import EDM_CIFAR10_Config
    from fastgen.methods import DMD2Model
    from fastgen.utils.distributed.fsdp import model_to_fsdp

    # Create DMD2 config with EDM network
    config = ModelConfig()
    config.net = copy.deepcopy(EDM_CIFAR10_Config)
    config.input_shape = [3, 32, 32]
    config.precision = "float32"
    config.device = "cuda"
    config.fsdp_meta_init = True
    config.pretrained_model_path = ""
    config.use_ema = False
    config.add_teacher_to_fsdp_dict = True
    # Disable GAN loss to avoid discriminator complexity
    config.gan_loss_weight_gen = 0.0

    result = {
        "rank": rank,
        "world_size": world_size,
    }

    # Step 1: Build DMD2 model with fsdp_meta_init=True
    model = DMD2Model(config)

    # Step 2: Verify meta device usage for all networks
    networks_to_check = ["net", "teacher", "fake_score"]
    meta_status = {}

    for net_name in networks_to_check:
        if hasattr(model, net_name):
            network = getattr(model, net_name)
            meta_count = sum(1 for p in network.parameters() if p.device.type == "meta")
            real_count = sum(1 for p in network.parameters() if p.device.type != "meta")
            meta_status[net_name] = {
                "meta": meta_count,
                "real": real_count,
            }

    # Verify correct meta device usage based on rank
    if rank == 0:
        # Rank 0: all networks should have real tensors
        has_correct_device_before = all(status["meta"] == 0 for status in meta_status.values())
    else:
        # Non-rank-0: all networks should have meta tensors
        has_correct_device_before = all(status["real"] == 0 for status in meta_status.values())

    result["meta_status_before"] = meta_status
    result["has_correct_device_before"] = has_correct_device_before

    dist.barrier()

    # Step 3: Apply model_to_fsdp (handles multiple networks via fsdp_dict)
    model = model_to_fsdp(
        model,
        min_num_params=1_000_000,
        apply_cpu_offload=False,
        sync_module_states=True,
    )

    dist.barrier()

    # Step 4: Verify all networks are materialized after FSDP
    meta_status_after = {}
    for net_name in networks_to_check:
        if hasattr(model, net_name):
            network = getattr(model, net_name)
            meta_count = 0
            real_count = 0
            for p in network.parameters():
                if hasattr(p, "_local_tensor"):
                    device = p._local_tensor.device
                else:
                    device = p.device
                if device.type == "meta":
                    meta_count += 1
                else:
                    real_count += 1
            meta_status_after[net_name] = {
                "meta": meta_count,
                "real": real_count,
            }

    has_correct_device_after = all(status["meta"] == 0 for status in meta_status_after.values())
    result["meta_status_after"] = meta_status_after
    result["has_correct_device_after"] = has_correct_device_after

    # Step 5: Test forward pass for net and fake_score
    model.net.eval()
    model.fake_score.eval()
    model.teacher.eval()

    generator = torch.Generator(device="cpu").manual_seed(42)
    dummy_input = torch.randn(2, 3, 32, 32, generator=generator, dtype=torch.float32).cuda()
    dummy_t = torch.tensor([0.5, 0.5], dtype=torch.float32, device="cuda")
    dummy_labels = torch.zeros(2, 10, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        net_output = model.net(dummy_input, dummy_t, condition=dummy_labels)
        fake_score_output = model.fake_score(dummy_input, dummy_t, condition=dummy_labels)
        teacher_output = model.teacher(dummy_input, dummy_t, condition=dummy_labels)

    # Check consistency across ranks for all networks
    forward_results = {}
    for name, output in [("net", net_output), ("fake_score", fake_score_output), ("teacher", teacher_output)]:
        if hasattr(output, "full_tensor"):
            output_full = output.full_tensor()
        else:
            output_full = output

        rank0_output = output_full.clone()
        dist.broadcast(rank0_output, src=0)
        diff = (output_full - rank0_output).abs().max().item()
        forward_results[name] = {
            "consistent": diff < 1e-5,
            "diff": diff,
        }

    result["forward_results"] = forward_results
    all_forward_consistent = all(r["consistent"] for r in forward_results.values())
    result["all_forward_consistent"] = all_forward_consistent

    result["all_passed"] = has_correct_device_before and has_correct_device_after and all_forward_consistent

    del model
    torch.cuda.empty_cache()

    return result


@RunIf(min_gpus=2)
def test_dmd2_model_to_fsdp():
    """Test model_to_fsdp with DMD2Model which has multiple networks.

    This verifies that FSDP correctly handles models with multiple networks
    (net, teacher, fake_score) in their fsdp_dict.
    """
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_dmd2_model_to_fsdp_impl,
        world_size=2,
        timeout=300,
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"
    assert result.get("has_correct_device_before", False), (
        f"Meta device not used correctly before FSDP: " f"meta_status={result.get('meta_status_before', 'N/A')}"
    )
    assert result.get("has_correct_device_after", False), (
        f"Tensors not properly materialized after model_to_fsdp: "
        f"meta_status={result.get('meta_status_after', 'N/A')}"
    )
    assert result.get("all_forward_consistent", False), (
        f"Forward pass inconsistent across ranks: " f"forward_results={result.get('forward_results', 'N/A')}"
    )
    assert result.get("all_passed", False), "DMD2 model_to_fsdp test failed"

    clear_gpu_memory()


# =============================================================================
# Integration Test (runs full verification similar to original script)
# =============================================================================


def _test_fsdp_full_verification_impl(
    rank: int,
    world_size: int,
) -> Dict:
    """Full FSDP verification test combining all checks."""
    results = {}

    # Weight sync test
    weight_result = _test_fsdp_weight_sync_impl(
        rank=rank,
        world_size=world_size,
        apply_checkpointing=False,
    )
    results["weight_sync"] = weight_result.get("weights_match", False)

    dist.barrier()
    torch.cuda.empty_cache()

    # Buffer sync test
    buffer_result = _test_fsdp_buffer_sync_impl(
        rank=rank,
        world_size=world_size,
    )
    results["buffer_sync"] = buffer_result.get("buffers_synced", False)

    dist.barrier()
    torch.cuda.empty_cache()

    # Forward pass test
    forward_result = _test_fsdp_forward_pass_impl(
        rank=rank,
        world_size=world_size,
        apply_checkpointing=False,
    )
    results["forward_deterministic"] = forward_result.get("deterministic", False)
    results["forward_rank_consistent"] = forward_result.get("ranks_consistent", False)
    results["forward_matches_reference"] = forward_result.get("forward_matches", False)

    results["all_passed"] = all(results.values())

    return results


@RunIf(min_gpus=2)
def test_fsdp_full_verification():
    """Full FSDP verification test - combines all individual tests."""
    clear_gpu_memory()

    result = run_distributed_test(
        test_fn=_test_fsdp_full_verification_impl,
        world_size=2,
        timeout=600,  # Longer timeout for full test
        setup_fn=set_env_vars,
    )

    assert result is not None, "Test did not return a result"

    # Check each component
    assert result.get("weight_sync", False), "Weight synchronization failed"
    assert result.get("buffer_sync", False), "Buffer synchronization failed"
    assert result.get("forward_deterministic", False), "Forward pass not deterministic"
    assert result.get("forward_rank_consistent", False), "Forward outputs differ across ranks"
    assert result.get("forward_matches_reference", False), "Forward output doesn't match reference"

    assert result["all_passed"], "Not all verification checks passed"

    clear_gpu_memory()
