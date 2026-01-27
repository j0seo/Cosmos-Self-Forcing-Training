# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for fastgen dataloaders.

This module tests:
1. Dummy dataset creation with correct specifications
2. Loader initialization and batch validation (keys, types, shapes)
3. Edge cases with missing files in WebDataset shards
4. Deterministic loading with resuming capabilities
"""

import io
import json
import os
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import shutil
import importlib

import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import TypeAdapter, ValidationError, ConfigDict

from fastgen.utils import instantiate


# =============================================================================
# Dummy Dataset Generation Utilities
# =============================================================================


class DummyImageDatasetBuilder:
    """Builder for creating dummy CIFAR10/ImageNet-style ZIP datasets.

    Creates ZIP files containing images and optional labels (dataset.json).
    Supports both pixel images (.png) and latent representations (.npy).

    Example:
        >>> builder = DummyImageDatasetBuilder("/tmp/test", "cifar10")
        >>> builder.add_image("00000.png", np.random.rand(32, 32, 3), label=0)
        >>> builder.add_image("00001.png", np.random.rand(32, 32, 3), label=1)
        >>> zip_path = builder.build()
    """

    def __init__(self, tmp_dir: str, dataset_name: str = "dummy_dataset"):
        """Initialize the builder.

        Args:
            tmp_dir: Directory to store the zip file
            dataset_name: Name for the zip file (without extension)
        """
        self.tmp_dir = tmp_dir
        self.dataset_name = dataset_name
        self.zip_path = os.path.join(tmp_dir, f"{dataset_name}.zip")
        self.images: List[Tuple[str, np.ndarray]] = []
        self.labels: Dict[str, int] = {}

    def add_image(self, filename: str, image: np.ndarray, label: int | None = None) -> "DummyImageDatasetBuilder":
        """Add an image to the dataset.

        Args:
            filename: Image filename (e.g., "00000.png" or "00000.npy")
            image: Image array. For .png: (H, W, C) uint8, for .npy: (C, H, W) float32
            label: Optional class label (integer)

        Returns:
            self for method chaining
        """
        self.images.append((filename, image))
        if label is not None:
            self.labels[filename] = label
        return self

    def build(self) -> str:
        os.makedirs(self.tmp_dir, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, "w") as zf:
            for filename, image in self.images:
                if filename.endswith(".npy"):
                    buf = io.BytesIO()
                    np.save(buf, image)
                    zf.writestr(filename, buf.getvalue())
                else:
                    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                        image = image.transpose(1, 2, 0)
                    if image.shape[-1] == 1:
                        image = image.squeeze(-1)
                    img = Image.fromarray(image.astype(np.uint8))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    zf.writestr(filename, buf.getvalue())

            if self.labels:
                labels_data = {"labels": [[fname, label] for fname, label in self.labels.items()]}
                zf.writestr("dataset.json", json.dumps(labels_data))
        return self.zip_path

    @classmethod
    def create_cifar10_style(
        cls, tmp_dir: str, num_samples: int = 100, resolution: int = 32, num_classes: int = 10
    ) -> str:
        """Create a CIFAR10-style dataset with random images.

        Args:
            tmp_dir: Directory for the dataset
            num_samples: Number of images to generate
            resolution: Image resolution (square)
            num_classes: Number of classes for labels

        Returns:
            Path to the created ZIP file
        """
        builder = cls(tmp_dir, f"cifar10-{resolution}x{resolution}")
        for i in range(num_samples):
            image = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
            builder.add_image(f"{i:05d}.png", image, i % num_classes)
        return builder.build()

    @classmethod
    def create_imagenet_style(
        cls,
        tmp_dir: str,
        num_samples: int = 100,
        resolution: int = 64,
        num_classes: int = 1000,
        latent: bool = False,
        latent_channels: int = 8,
    ) -> str:
        """Create an ImageNet-style dataset with random images or latents.

        Args:
            tmp_dir: Directory for the dataset
            num_samples: Number of images to generate
            resolution: Image resolution (square)
            num_classes: Number of classes for labels
            latent: If True, create latent representations instead of images
            latent_channels: Number of channels for latent representations

        Returns:
            Path to the created ZIP file
        """
        suffix = "_sd" if latent else ""
        builder = cls(tmp_dir, f"imagenet_{resolution}{suffix}")
        for i in range(num_samples):
            label = num_classes - 1 if i == num_samples - 1 else (i * num_classes) // num_samples
            if latent:
                latent_res = resolution // 8
                image = np.random.randn(latent_channels, latent_res, latent_res).astype(np.float32)
                builder.add_image(f"{i:05d}.npy", image, label)
            else:
                image = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
                builder.add_image(f"{i:05d}.png", image, label)
        return builder.build()


class DummyWebDatasetBuilder:
    """Builder for creating dummy WebDataset tar files for testing.

    Creates tar files containing samples with specified file types and shapes.
    Supports: .npy, .npz, .pth, .jpg, .png, .txt, .json, .mp4, .caption

    Example:
        >>> builder = DummyWebDatasetBuilder("/tmp/test", "00000.tar")
        >>> builder.add_sample("sample_0", {"latents.npy": np.random.randn(4, 64, 64), "txt": "caption"})
        >>> builder.add_sample("sample_1", {"latents.npy": np.random.randn(4, 64, 64), "txt": "caption"})
        >>> tar_path = builder.build()
    """

    NPY_EXTENSIONS = frozenset(
        [
            "npy",
            "latents.npy",
            "sample.npy",
            "noise.npy",
            "text_embedding.npy",
            "pooled_text_embedding.npy",
            "neg_text_embedding.npy",
            "sample_path.npy",
            "text_emb.npy",
            "path.npy",
        ]
    )
    PTH_EXTENSIONS = frozenset(["pth", "latent.pth", "txt_emb.pth", "depth_latent.pth", "noise.pth", "path.pth"])

    def __init__(self, tmp_dir: str, shard_name: str = "00000.tar"):
        self.tmp_dir = tmp_dir
        self.shard_name = shard_name
        self.shard_path = os.path.join(tmp_dir, shard_name)
        self.samples: List[Dict[str, bytes]] = []

    def add_sample(self, key: str, files: Dict[str, Any]) -> "DummyWebDatasetBuilder":
        sample_files = {f"{key}.{ext}": self._encode_data(ext, data) for ext, data in files.items()}
        self.samples.append(sample_files)
        return self

    def _encode_data(self, ext: str, data: Any) -> bytes:
        if ext == "npy" or ext.endswith(".npy"):
            buf = io.BytesIO()
            np.save(buf, data)
            return buf.getvalue()
        elif ext == "npz":
            buf = io.BytesIO()
            np.savez(buf, **data)
            return buf.getvalue()
        elif ext == "pth" or ext.endswith(".pth"):
            buf = io.BytesIO()
            torch.save(data, buf)
            return buf.getvalue()
        elif ext == "json":
            return json.dumps(data).encode("utf-8")
        elif ext in ("txt", "caption"):
            return data.encode("utf-8") if isinstance(data, str) else data
        elif ext in ("jpg", "jpeg"):
            buf = io.BytesIO()
            img = data if isinstance(data, Image.Image) else Image.fromarray(data.astype(np.uint8))
            img.save(buf, format="JPEG")
            return buf.getvalue()
        elif ext == "png":
            buf = io.BytesIO()
            img = data if isinstance(data, Image.Image) else Image.fromarray(data.astype(np.uint8))
            img.save(buf, format="PNG")
            return buf.getvalue()
        elif ext == "mp4":
            return data if isinstance(data, bytes) else self._create_dummy_mp4(data)
        else:
            return data if isinstance(data, bytes) else str(data).encode("utf-8")

    def _create_dummy_mp4(self, frames: np.ndarray) -> bytes:
        import av

        buf = io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("h264", rate=24)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for frame_data in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame_data, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return buf.getvalue()

    def build(self) -> str:
        os.makedirs(os.path.dirname(self.shard_path) or ".", exist_ok=True)
        with tarfile.open(self.shard_path, "w") as tar:
            for sample_files in self.samples:
                for filename, data in sample_files.items():
                    info = tarfile.TarInfo(name=filename)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        return self.shard_path

    @classmethod
    def create_dataset_dir(
        cls, tmp_dir: str, num_shards: int, samples_per_shard: int, file_specs: Dict[str, Any]
    ) -> str:
        """Create a complete WebDataset directory with multiple shards.

        Args:
            tmp_dir: Base directory for the dataset
            num_shards: Number of tar shards to create
            samples_per_shard: Number of samples per shard
            file_specs: Dict mapping extension -> shape/value specification
                        For arrays: tuple of shape (e.g., (4, 64, 64))
                        For json: dict template or callable(shard_idx, sample_idx)
                        For txt/caption: str template or callable(shard_idx, sample_idx)
                        For mp4: (T, H, W) tuple for frame dimensions

        Returns:
            Path to the dataset directory

        Example:
            >>> dataset_dir = DummyWebDatasetBuilder.create_dataset_dir(
            ...     "/tmp", num_shards=2, samples_per_shard=4,
            ...     file_specs={"latents.npy": (4, 64, 64), "txt": lambda s, i: f"Sample {s}_{i}"}
            ... )
        """
        dataset_dir = os.path.join(tmp_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        for shard_idx in range(num_shards):
            builder = cls(dataset_dir, f"{shard_idx:05d}.tar")
            for sample_idx in range(samples_per_shard):
                key = f"{shard_idx:05d}_{sample_idx:05d}"
                files = {}
                for ext, spec in file_specs.items():
                    if ext in cls.NPY_EXTENSIONS:
                        files[ext] = np.random.randn(*spec).astype(np.float32)
                    elif ext in cls.PTH_EXTENSIONS:
                        files[ext] = torch.randn(*spec)
                    elif ext == "npz":
                        files[ext] = {k: np.random.randn(*v).astype(np.float32) for k, v in spec.items()}
                    elif ext == "json":
                        files[ext] = (
                            spec(shard_idx, sample_idx)
                            if callable(spec)
                            else (spec.copy() if isinstance(spec, dict) else spec)
                        )
                    elif ext in ("txt", "caption"):
                        files[ext] = spec(shard_idx, sample_idx) if callable(spec) else spec
                    elif ext in ("jpg", "jpeg", "png"):
                        h, w = spec[:2]
                        files[ext] = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                    elif ext == "mp4":
                        t, h, w = spec
                        files[ext] = np.random.randint(0, 255, (t, h, w, 3), dtype=np.uint8)
                    else:
                        files[ext] = spec
                builder.add_sample(key, files)
            builder.build()
        return dataset_dir


def create_shard_count_file(dataset_dir: str, samples_per_shard: int) -> str:
    """Create a shard count JSON file for deterministic loading."""
    shard_count = {f: samples_per_shard for f in os.listdir(dataset_dir) if f.endswith(".tar")}
    count_file = os.path.join(dataset_dir, "shard_count.json")
    with open(count_file, "w", encoding="utf-8") as f:
        json.dump(shard_count, f)
    return count_file


def create_ignore_index_file(dataset_dir: str, ignore_spec: Dict[str, List[str]]) -> str:
    """Create an ignore index JSON file."""
    ignore_file = os.path.join(dataset_dir, "ignore_index.json")
    with open(ignore_file, "w", encoding="utf-8") as f:
        json.dump(ignore_spec, f)
    return ignore_file


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tmp_dataset_dir():
    """Create a temporary directory for test datasets."""
    tmp_dir = tempfile.mkdtemp(prefix="fastgen_test_")
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Assertion Helpers
# =============================================================================


def assert_batch_keys(batch: Dict, expected_keys: List[str], allow_extra: bool = False):
    """Assert that batch contains expected keys."""
    for key in expected_keys:
        assert key in batch, f"Missing expected key: {key}"
    if not allow_extra:
        for key in batch:
            if not key.startswith("__"):
                assert key in expected_keys, f"Unexpected key in batch: {key}"


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str = "tensor"):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"


def assert_batch_type(batch: Dict, key: str, expected_type: Type):
    """Assert that batch[key] has expected type using pydantic TypeAdapter.

    This function validates types including nested types like List[str] and Dict[str, Any].
    For torch.Tensor, it falls back to isinstance check since pydantic doesn't handle it natively.

    Args:
        batch: The batch dictionary
        key: The key to check
        expected_type: The expected type (can be a generic type like List[str])
    """
    value = batch[key]

    try:
        adapter = TypeAdapter(expected_type, config=ConfigDict(arbitrary_types_allowed=True))
        adapter.validate_python(value)
    except ValidationError as e:
        raise AssertionError(f"batch['{key}'] type validation failed for {expected_type}: {e}") from e


# =============================================================================
# Test Specification Classes
# =============================================================================


@dataclass
class LoaderTestSpec:
    """Specification for a loader test.

    The expected keys are derived from expected_types.keys(), so you don't need
    to specify them separately. Multiple configs with the same specs can be
    tested together by providing a list of config_paths.

    The test will
    run twice: once with dummy data (always), once with real data (skipped by default).
    """

    name: str
    config_paths: List[str]  # e.g., ["fastgen.configs.data.VideoLatentLoaderConfig"]
    file_specs: Dict[str, Any]
    expected_types: Dict[str, type]  # key -> expected type (keys define expected batch keys)
    expected_shapes: Dict[str, Tuple] = field(default_factory=dict)  # key -> expected shape
    num_shards: int = 1
    samples_per_shard: int = 4
    batch_size: int = 2
    extra_config: Dict[str, Any] = field(default_factory=dict)
    extra_files: Optional[Callable[[str], Dict[str, str]]] = None  # Creates extra files, returns config updates
    # Real data integration testing (optional - skipped by default)
    credentials_path: str = "./credentials/s3.json"  # Path to credentials for S3


# =============================================================================
# Generic Test Runners
# =============================================================================


def run_loader_test(tmp_dir: str, spec: LoaderTestSpec, use_real_data: bool = False):
    """Run a generic loader test from specification.

    Tests all configs in spec.config_paths with either dummy or real data.

    Args:
        tmp_dir: Temporary directory for dummy dataset
        spec: Test specification
        use_real_data: If True, use real_datatags from spec instead of dummy data
    """
    if use_real_data:
        # Real data integration test
        from fastgen.utils.io_utils import set_env_vars

        set_env_vars(credentials_path=spec.credentials_path)
        extra_config_updates = {}
        datatags = None
    else:
        # Dummy data test
        dataset_dir = DummyWebDatasetBuilder.create_dataset_dir(
            tmp_dir, spec.num_shards, spec.samples_per_shard, spec.file_specs
        )
        datatags = [f"WDS:{dataset_dir}"]
        extra_config_updates = spec.extra_files(dataset_dir) if spec.extra_files else {}
        extra_config_updates.update(spec.extra_config)

    # Test each config path
    for config_path in spec.config_paths:
        # Import config dynamically
        module_path, config_name = config_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        config_class = getattr(module, config_name)

        # Configure loader
        config = config_class.copy()
        if datatags is not None:
            config.datatags = datatags
        config.batch_size = spec.batch_size
        config.num_workers = 0
        for k, v in extra_config_updates.items():
            setattr(config, k, v)

        # Get batch
        loader = instantiate(config)
        batch = next(iter(loader))

        # Assertions - expected keys derived from expected_types
        assert_batch_keys(batch, list(spec.expected_types.keys())), f"Failed for {config_name}"
        for key, expected_type in spec.expected_types.items():
            assert_batch_type(batch, key, expected_type)
        for key, expected_shape in spec.expected_shapes.items():
            assert_tensor_shape(batch[key], expected_shape, key)


# =============================================================================
# ImageLoader Tests
# =============================================================================


@dataclass
class ImageLoaderTestSpec:
    """Specification for an ImageLoader test.

    Similar to LoaderTestSpec but for class-conditional image datasets (CIFAR10, ImageNet).
    The expected keys are derived from expected_types.keys().
    """

    name: str
    config_path: str  # e.g., "fastgen.configs.data.CIFAR10_Loader_Config"
    expected_types: Dict[str, type]  # key -> expected type
    expected_shapes: Dict[str, Tuple]  # key -> expected shape (with batch_size)
    num_samples: int = 64
    batch_size: int = 8
    latent_channels: Optional[int] = None  # If not None, used for latent-based dummy data generation
    resolution: int = 64  # Used for dummy data generation
    num_classes: int = 1000  # Used for dummy data generation
    credentials_path: str = "./credentials/s3.json"


# Test specifications for ImageLoaders
IMAGE_LOADER_SPECS = [
    ImageLoaderTestSpec(
        name="cifar10",
        config_path="fastgen.configs.data.CIFAR10_Loader_Config",
        expected_types={
            "real": torch.Tensor,
            "condition": torch.Tensor,
            "neg_condition": torch.Tensor,
            "idx": torch.Tensor,
        },
        expected_shapes={"real": (8, 3, 32, 32), "condition": (8, 10), "neg_condition": (8, 10), "idx": (8,)},
        resolution=32,
        num_classes=10,
    ),
    ImageLoaderTestSpec(
        name="imagenet64",
        config_path="fastgen.configs.data.ImageNet64_Loader_Config",
        expected_types={
            "real": torch.Tensor,
            "condition": torch.Tensor,
            "neg_condition": torch.Tensor,
            "idx": torch.Tensor,
        },
        expected_shapes={"real": (8, 3, 64, 64), "condition": (8, 1000), "neg_condition": (8, 1000), "idx": (8,)},
        resolution=64,
        num_classes=1000,
    ),
    ImageLoaderTestSpec(
        name="imagenet64_edmv2",
        config_path="fastgen.configs.data.ImageNet64_EDMV2_Loader_Config",
        expected_types={
            "real": torch.Tensor,
            "condition": torch.Tensor,
            "neg_condition": torch.Tensor,
            "idx": torch.Tensor,
        },
        expected_shapes={"real": (8, 3, 64, 64), "condition": (8, 1000), "neg_condition": (8, 1000), "idx": (8,)},
        resolution=64,
        num_classes=1000,
    ),
    ImageLoaderTestSpec(
        name="imagenet256_latent",
        config_path="fastgen.configs.data.ImageNet256_Loader_Config",
        expected_types={
            "real": torch.Tensor,
            "condition": torch.Tensor,
            "neg_condition": torch.Tensor,
            "idx": torch.Tensor,
        },
        expected_shapes={"real": (8, 8, 32, 32), "condition": (8, 1000), "neg_condition": (8, 1000), "idx": (8,)},
        resolution=256,
        num_classes=1000,
        latent_channels=8,
    ),
]


def run_image_loader_test(tmp_dir: str, spec: ImageLoaderTestSpec, use_real_data: bool = False):
    """Run an ImageLoader test from specification.

    Args:
        tmp_dir: Temporary directory for dummy dataset
        spec: Test specification
        use_real_data: If True, use real data paths from config instead of dummy data
    """
    # Import config dynamically
    module_path, config_name = spec.config_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    config_class = getattr(module, config_name)

    config = config_class.copy()
    config.batch_size = spec.batch_size
    config.sampler_start_idx = 0

    if use_real_data:
        # Real data integration test - use paths from config
        from fastgen.utils.io_utils import set_env_vars

        set_env_vars(credentials_path=spec.credentials_path)
    else:
        # Dummy data test - create synthetic dataset
        if spec.latent_channels is not None:
            zip_path = DummyImageDatasetBuilder.create_imagenet_style(
                tmp_dir,
                spec.num_samples,
                spec.resolution,
                spec.num_classes,
                latent=True,
                latent_channels=spec.latent_channels,
            )
        elif spec.num_classes == 10:  # CIFAR10-style
            zip_path = DummyImageDatasetBuilder.create_cifar10_style(
                tmp_dir, spec.num_samples, spec.resolution, spec.num_classes
            )
        else:  # ImageNet-style
            zip_path = DummyImageDatasetBuilder.create_imagenet_style(
                tmp_dir, spec.num_samples, spec.resolution, spec.num_classes
            )
        config.dataset_path = zip_path
        config.s3_path = None

    # Get batch
    loader = instantiate(config)
    batch = next(iter(loader))

    # Assert keys (derived from expected_types)
    expected_keys = list(spec.expected_types.keys())
    assert_batch_keys(batch, expected_keys)

    # Assert types
    for key, expected_type in spec.expected_types.items():
        assert_batch_type(batch, key, expected_type)

    # Shape checks (only for dummy data - real data shapes depend on config batch_size)
    if not use_real_data:
        for key, expected_shape in spec.expected_shapes.items():
            assert_tensor_shape(batch[key], expected_shape, key)
        if spec.latent_channels is None:
            assert batch["real"].min() >= -1.0 and batch["real"].max() <= 1.0
    else:
        # For real data, just check that shapes are consistent
        assert batch["condition"].shape[1] == spec.num_classes, f"Expected {spec.num_classes} classes"


class TestImageLoader:
    """Tests for ImageLoader with CIFAR10 and ImageNet-style datasets."""

    @pytest.mark.parametrize("spec", IMAGE_LOADER_SPECS, ids=lambda s: s.name)
    def test_image_loader(self, tmp_dataset_dir, spec):
        """Test image loaders with dummy datasets."""
        run_image_loader_test(tmp_dataset_dir, spec, use_real_data=False)

    @pytest.mark.parametrize("spec", IMAGE_LOADER_SPECS, ids=lambda s: s.name)
    @pytest.mark.integration
    def test_image_loader_real_data(self, tmp_dataset_dir, spec):
        """Test image loaders with real data (integration test)."""
        run_image_loader_test(tmp_dataset_dir, spec, use_real_data=True)

    def test_sampler_start_idx_resume(self, tmp_dataset_dir):
        """Test that sampler_start_idx correctly resumes from a given position."""
        from fastgen.configs.data import CIFAR10_Loader_Config

        zip_path = DummyImageDatasetBuilder.create_cifar10_style(tmp_dataset_dir, 32, 32, 10)

        def create_loader(start_idx):
            config = CIFAR10_Loader_Config.copy()
            config.dataset_path = zip_path
            config.s3_path = None
            config.batch_size = 4
            config.sampler_start_idx = start_idx
            config.shuffle = True
            return instantiate(config)

        # Get first 3 batches from start
        loader_start = iter(create_loader(0))
        batches = [next(loader_start)["idx"].tolist() for _ in range(3)]

        # Resume from index 8 should match 3rd batch
        loader_resume = iter(create_loader(8))
        resumed_batch = next(loader_resume)["idx"].tolist()

        assert resumed_batch == batches[2], f"Resume mismatch: expected {batches[2]}, got {resumed_batch}"

    def test_samples_unique_within_epoch(self, tmp_dataset_dir):
        """Test that all samples within one epoch are unique."""
        from fastgen.configs.data import CIFAR10_Loader_Config

        num_samples, batch_size = 32, 4
        zip_path = DummyImageDatasetBuilder.create_cifar10_style(tmp_dataset_dir, num_samples, 32, 10)

        config = CIFAR10_Loader_Config.copy()
        config.dataset_path = zip_path
        config.s3_path = None
        config.batch_size = batch_size
        config.sampler_start_idx = 0
        config.shuffle = True

        loader = instantiate(config)
        all_indices = []
        for i, batch in enumerate(loader):
            if i >= num_samples // batch_size:
                break
            all_indices.extend(batch["idx"].tolist())

        assert len(all_indices) == num_samples
        assert len(set(all_indices)) == num_samples, "Found duplicate samples in epoch"

    def test_samples_unique_across_resumed_training(self, tmp_dataset_dir):
        """Test that resumed training doesn't repeat samples."""
        from fastgen.configs.data import CIFAR10_Loader_Config

        num_samples, batch_size = 32, 4
        zip_path = DummyImageDatasetBuilder.create_cifar10_style(tmp_dataset_dir, num_samples, 32, 10)

        def create_loader(start_idx):
            config = CIFAR10_Loader_Config.copy()
            config.dataset_path = zip_path
            config.s3_path = None
            config.batch_size = batch_size
            config.sampler_start_idx = start_idx
            config.shuffle = False
            return instantiate(config)

        # First half
        loader1 = create_loader(0)
        first_half = []
        for i, batch in enumerate(loader1):
            if i >= num_samples // (2 * batch_size):
                break
            first_half.extend(batch["idx"].tolist())

        # Second half (resumed)
        loader2 = create_loader(len(first_half))
        second_half = []
        for i, batch in enumerate(loader2):
            if i >= num_samples // (2 * batch_size):
                break
            second_half.extend(batch["idx"].tolist())

        assert set(first_half).isdisjoint(set(second_half)), "Found overlapping samples"


# =============================================================================
# WDSLoader Tests
# =============================================================================


def _create_neg_prompt_file(dataset_dir: str, shape: Tuple[int, ...]) -> Dict[str, str]:
    """Create a neg_prompt_emb.npy file for testing files_map loading.

    Args:
        dataset_dir: Directory to create the file in
        shape: Shape of the tensor to create

    Returns:
        Dict with files_map config update pointing to the created file
    """
    neg_prompt_path = os.path.join(dataset_dir, "neg_prompt_emb.npy")
    np.save(neg_prompt_path, np.random.randn(*shape).astype(np.float32))
    return {"files_map": {"neg_condition": neg_prompt_path}}


# Test specifications for WDS loaders (generic template configs)
WDS_LOADER_SPECS = [
    # ImageWDSLoader - generic image loader with jpg + txt
    LoaderTestSpec(
        name="image_wds",
        config_paths=["fastgen.configs.data.ImageLoaderConfig"],
        file_specs={
            "jpg": (512, 512),
            "txt": lambda s, i: f"A sample image {s}_{i}",
        },
        expected_types={
            "real": torch.Tensor,
            "condition": List[str],
            "neg_condition": List[str],
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={"real": (2, 3, 512, 512)},
        extra_config={"input_res": 512},
    ),
    # VideoWDSLoader - generic video loader with mp4 + txt
    LoaderTestSpec(
        name="video_wds",
        config_paths=["fastgen.configs.data.VideoLoaderConfig"],
        file_specs={
            "mp4": (90, 480, 832),
            "txt": lambda s, i: f"A sample video {s}_{i}",
        },
        expected_types={
            "real": torch.Tensor,
            "condition": List[str],
            "neg_condition": List[str],
            "cropping_params": Dict[str, Any],
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={"real": (1, 3, 81, 480, 832)},  # (B, C, T, H, W)
        batch_size=1,
        samples_per_shard=2,
        extra_config={"sequence_length": 81, "img_size": (832, 480)},
    ),
    # WDSLoader - generic image latent loader with latent.pth + txt_emb.pth + neg_condition from files_map
    LoaderTestSpec(
        name="image_latent_wds",
        config_paths=["fastgen.configs.data.ImageLatentLoaderConfig"],
        file_specs={
            "latent.pth": (4, 128, 128),
            "txt_emb.pth": (77, 2048),
        },
        expected_types={
            "real": torch.Tensor,
            "condition": torch.Tensor,
            "neg_condition": torch.Tensor,
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={
            "real": (2, 4, 128, 128),
            "condition": (2, 77, 2048),
            "neg_condition": (2, 77, 2048),
        },
        extra_files=lambda d: _create_neg_prompt_file(d, (77, 2048)),
        batch_size=2,
    ),
    # WDSLoader - generic video latent loader with latent.pth + txt_emb.pth + neg_condition from files_map
    LoaderTestSpec(
        name="video_latent_wds",
        config_paths=["fastgen.configs.data.VideoLatentLoaderConfig"],
        file_specs={
            "latent.pth": (16, 21, 60, 104),
            "txt_emb.pth": (512, 4096),
        },
        expected_types={
            "real": torch.Tensor,
            "condition": torch.Tensor,
            "neg_condition": torch.Tensor,
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={
            "real": (2, 16, 21, 60, 104),
            "condition": (2, 512, 4096),
            "neg_condition": (2, 512, 4096),
        },
        extra_files=lambda d: _create_neg_prompt_file(d, (512, 4096)),
    ),
    # PairLoaderConfig - for single-step KD with (real, noise, condition)
    # Data requirements from KD.py: {"real": clean, "noise": noise, "condition": cond}
    LoaderTestSpec(
        name="pair_wds",
        config_paths=["fastgen.configs.data.PairLoaderConfig"],
        file_specs={
            "latent.pth": (16, 21, 60, 104),
            "noise.pth": (16, 21, 60, 104),
            "txt_emb.pth": (512, 4096),
        },
        expected_types={
            "real": torch.Tensor,
            "noise": torch.Tensor,
            "condition": torch.Tensor,
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={
            "real": (2, 16, 21, 60, 104),
            "noise": (2, 16, 21, 60, 104),
            "condition": (2, 512, 4096),
        },
    ),
    # PathLoaderConfig - for multi-step KD with (real, path, condition)
    # Data requirements from KD.py: {"real": clean, "path": [B, steps, C, ...], "condition": cond}
    # path shape: [B, num_inf_steps=4, C, ...]
    LoaderTestSpec(
        name="path_wds",
        config_paths=["fastgen.configs.data.PathLoaderConfig"],
        file_specs={
            "latent.pth": (16, 21, 60, 104),
            "path.pth": (4, 16, 21, 60, 104),  # [num_inf_steps=4, C, T, H, W]
            "txt_emb.pth": (512, 4096),
        },
        expected_types={
            "real": torch.Tensor,
            "path": torch.Tensor,
            "condition": torch.Tensor,
            "fname": List[str],
            "shard": List[str],
        },
        expected_shapes={
            "real": (2, 16, 21, 60, 104),
            "path": (2, 4, 16, 21, 60, 104),  # [B, num_inf_steps=4, C, T, H, W]
            "condition": (2, 512, 4096),
        },
    ),
]


class TestWDSLoader:
    """Tests for WDSLoader with generic template configs.

    Each test runs with dummy data by default. A second test variant runs with real data
    (skipped by default, enable with pytest --run-integration).
    """

    @pytest.mark.parametrize("spec", WDS_LOADER_SPECS, ids=lambda s: s.name)
    def test_wds_loader(self, tmp_dataset_dir, spec):
        """Test WDS loaders with dummy datasets."""
        run_loader_test(tmp_dataset_dir, spec, use_real_data=False)

    @pytest.mark.parametrize("spec", WDS_LOADER_SPECS, ids=lambda s: s.name)
    @pytest.mark.integration
    def test_wds_loader_real_data(self, tmp_dataset_dir, spec):
        """Test WDS loaders with real data (integration test)."""
        run_loader_test(tmp_dataset_dir, spec, use_real_data=True)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_optional_files(self, tmp_dataset_dir):
        """Test handling of missing optional files."""
        from fastgen.datasets.wds_dataloaders import WDSLoader

        dataset_dir = DummyWebDatasetBuilder.create_dataset_dir(
            tmp_dataset_dir, 1, 4, {"latents.npy": (16, 64, 64), "text_embedding.npy": (512, 4096)}
        )

        loader = WDSLoader(
            datatags=[f"WDS:{dataset_dir}"],
            batch_size=2,
            key_map={"real": "latents.npy", "condition": "text_embedding.npy"},
            presets_map={"neg_condition": "empty_string"},
            num_workers=0,
        )

        batch = next(iter(loader))
        assert_batch_keys(batch, ["real", "condition", "neg_condition", "fname", "shard"])
        assert_batch_type(batch, "neg_condition", List[str])
        assert all(c == "" for c in batch["neg_condition"])

    def test_partial_files_filtering(self, tmp_dataset_dir):
        """Test that samples with missing required files are filtered out."""
        from fastgen.datasets.wds_dataloaders import WDSLoader

        partial_dir = os.path.join(tmp_dataset_dir, "partial")
        os.makedirs(partial_dir, exist_ok=True)
        builder = DummyWebDatasetBuilder(partial_dir, "00000.tar")

        # Add complete and incomplete samples (indices 0,2 have text, 1,3 don't)
        for i, has_text in enumerate([True, False, True, False]):
            key = f"00000_{i:05d}"
            sample = {f"{key}.latents.npy": builder._encode_data("npy", np.random.randn(16, 64, 64).astype(np.float32))}
            if has_text:
                sample[f"{key}.text_embedding.npy"] = builder._encode_data(
                    "npy", np.random.randn(512, 4096).astype(np.float32)
                )
            builder.samples.append(sample)
        builder.build()

        loader = WDSLoader(
            datatags=[f"WDS:{partial_dir}"],
            batch_size=2,
            key_map={"real": "latents.npy", "condition": "text_embedding.npy"},
            num_workers=0,
            train=False,
        )

        batch = next(iter(loader))
        # check that only the two complete samples are loaded based on fname
        assert batch["fname"] == ["00000_00000", "00000_00002"]

    def test_ignore_index_filtering(self, tmp_dataset_dir):
        """Test ignore index filtering."""
        from fastgen.datasets.wds_dataloaders import WDSLoader

        dataset_dir = DummyWebDatasetBuilder.create_dataset_dir(tmp_dataset_dir, 1, 4, {"latents.npy": (16, 64, 64)})

        ignore_file = create_ignore_index_file(dataset_dir, {"00000.tar": ["00000_00000", "00000_00001"]})

        loader = WDSLoader(
            datatags=[f"WDS:{dataset_dir}"],
            batch_size=2,
            key_map={"real": "latents.npy"},
            ignore_index_paths=[ignore_file],
            num_workers=0,
        )

        batch = next(iter(loader))
        assert_batch_type(batch, "fname", List[str])
        assert "00000_00000" not in batch["fname"]
        assert "00000_00001" not in batch["fname"]


# =============================================================================
# Deterministic WDSLoader Tests
# =============================================================================


class TestDeterministicWDSLoader:
    """Tests for deterministic WDSLoader with resuming capabilities."""

    @pytest.fixture
    def deterministic_dataset_dir(self, tmp_dataset_dir):
        """Create a deterministic test dataset with multiple shards."""
        dataset_dir = os.path.join(tmp_dataset_dir, "deterministic")
        os.makedirs(dataset_dir, exist_ok=True)

        for shard_idx in range(3):
            builder = DummyWebDatasetBuilder(dataset_dir, f"{shard_idx:05d}.tar")
            for sample_idx in range(8):
                # Deterministic data for verification
                latent_data = np.full((16, 64, 64), float(shard_idx * 8 + sample_idx), dtype=np.float32)
                builder.samples.append(
                    {
                        f"{shard_idx:05d}_{sample_idx:05d}.latents.npy": builder._encode_data("npy", latent_data),
                        f"{shard_idx:05d}_{sample_idx:05d}.text_embedding.npy": builder._encode_data(
                            "npy", np.random.randn(512, 4096).astype(np.float32)
                        ),
                    }
                )
            builder.build()

        shard_count_file = create_shard_count_file(dataset_dir, 8)
        return dataset_dir, shard_count_file

    def _create_det_loader(self, dataset_dir, shard_count_file, start_idx=0, **kwargs):
        """Helper to create a deterministic WDSLoader."""
        from fastgen.datasets.wds_dataloaders import WDSLoader

        defaults = {
            "datatags": [f"WDS:{dataset_dir}"],
            "batch_size": 4,
            "key_map": {"real": "latents.npy", "condition": "text_embedding.npy"},
            "num_workers": 1,
            "deterministic": True,
            "sampler_start_idx": start_idx,
            "shard_count_file": shard_count_file,
        }
        return WDSLoader(**{**defaults, **kwargs})

    def _collect_fnames(self, loader, max_samples=None, max_batches=None):
        """Helper to collect fnames from loader."""
        fnames = []
        for i, batch in enumerate(loader):
            fnames.extend(batch["fname"])
            if max_samples and len(fnames) >= max_samples:
                break
            if max_batches and i >= max_batches - 1:
                break
        return fnames[:max_samples] if max_samples else fnames

    def test_deterministic_same_order(self, deterministic_dataset_dir):
        """Test that deterministic loader produces same order every run."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        fnames1 = self._collect_fnames(self._create_det_loader(dataset_dir, shard_count_file), max_batches=6)
        fnames2 = self._collect_fnames(self._create_det_loader(dataset_dir, shard_count_file), max_batches=6)

        assert fnames1 == fnames2, "Deterministic loader should produce same order"

    def test_deterministic_resume_from_index(self, deterministic_dataset_dir):
        """Test resuming from a specific sampler_start_idx."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        full_fnames = self._collect_fnames(self._create_det_loader(dataset_dir, shard_count_file), max_batches=6)
        resumed_fnames = self._collect_fnames(
            self._create_det_loader(dataset_dir, shard_count_file, start_idx=12), max_batches=3
        )

        assert resumed_fnames == full_fnames[12 : 12 + len(resumed_fnames)]

    def test_deterministic_no_overlap_on_resume(self, deterministic_dataset_dir):
        """Test that resumed samples don't overlap with pre-resume samples."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        start_fnames = self._collect_fnames(self._create_det_loader(dataset_dir, shard_count_file), max_batches=2)
        resumed_fnames = self._collect_fnames(
            self._create_det_loader(dataset_dir, shard_count_file, start_idx=8), max_batches=2
        )

        assert set(start_fnames).isdisjoint(set(resumed_fnames))

    def test_deterministic_unique_samples_per_epoch(self, deterministic_dataset_dir):
        """Test that samples are unique within one epoch."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        all_fnames = self._collect_fnames(self._create_det_loader(dataset_dir, shard_count_file), max_samples=24)

        assert len(all_fnames) == 24
        assert len(set(all_fnames)) == 24, "Found duplicate samples"

    def test_deterministic_data_integrity(self, deterministic_dataset_dir):
        """Test that data values are correctly loaded."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        loader = self._create_det_loader(dataset_dir, shard_count_file, batch_size=1)

        for i, batch in enumerate(loader):
            fname = batch["fname"][0]
            shard_idx, sample_idx = int(fname.split("_")[0]), int(fname.split("_")[1])
            expected_value = float(shard_idx * 8 + sample_idx)
            actual_value = batch["real"][0, 0, 0, 0].item()

            assert abs(actual_value - expected_value) < 1e-5, f"Data mismatch for {fname}"
            assert_batch_type(batch, "real", torch.Tensor)

            if i >= 5:
                break

    def test_deterministic_with_ignore_index(self, deterministic_dataset_dir):
        """Test deterministic loader with ignore index filtering."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        ignore_file = create_ignore_index_file(
            dataset_dir,
            {
                "00000.tar": ["00000_00000", "00000_00001"],
                "00001.tar": ["00001_00000"],
            },
        )

        loader = self._create_det_loader(dataset_dir, shard_count_file, ignore_index_paths=[ignore_file], partial=True)
        all_fnames = self._collect_fnames(loader, max_samples=21)

        assert "00000_00000" not in all_fnames
        assert "00000_00001" not in all_fnames
        assert "00001_00000" not in all_fnames
        assert len(all_fnames) == 21

    def test_deterministic_resume_with_ignore_index(self, deterministic_dataset_dir):
        """Test resuming deterministic loader with ignore index."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        ignore_file = create_ignore_index_file(dataset_dir, {"00000.tar": ["00000_00000", "00000_00001"]})

        full_fnames = self._collect_fnames(
            self._create_det_loader(dataset_dir, shard_count_file, ignore_index_paths=[ignore_file], partial=True),
            max_samples=22,
        )
        resumed_fnames = self._collect_fnames(
            self._create_det_loader(
                dataset_dir, shard_count_file, start_idx=8, ignore_index_paths=[ignore_file], partial=True
            ),
            max_samples=14,
        )

        assert resumed_fnames == full_fnames[8:]

    def test_deterministic_batch_types(self, deterministic_dataset_dir):
        """Test batch types are correct in deterministic mode."""
        dataset_dir, shard_count_file = deterministic_dataset_dir

        batch = next(iter(self._create_det_loader(dataset_dir, shard_count_file)))

        assert_batch_type(batch, "real", torch.Tensor)
        assert_batch_type(batch, "condition", torch.Tensor)
        assert_batch_type(batch, "fname", List[str])
        assert_batch_type(batch, "shard", List[str])
        assert_tensor_shape(batch["real"], (4, 16, 64, 64))
        assert_tensor_shape(batch["condition"], (4, 512, 4096))
