# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Download and prepare CIFAR-10 and ImageNet models and data for FastGen.

This script:
1. Downloads/converts datasets to EDM/EDM2 format using the respective repo's dataset_tool.py:
   - CIFAR-10: Downloads raw data and converts to cifar10-32x32.zip
   - ImageNet-64 (EDM): Converts from Kaggle download to imagenet-64x64.zip
   - ImageNet-64 (EDM2): Converts from Kaggle download to imagenet-64x64-edmv2.zip
   - ImageNet-256 (EDM2): Converts from Kaggle download to imagenet_256_sd.zip (VAE-encoded latents)

2. Downloads pretrained models and converts them from .pkl to .pth format:
   - EDM models: CIFAR-10 and ImageNet-64 models
   - EDM2 models: ImageNet-64 models (S/M/L/XL variants)

Output locations (defaults: $DATA_ROOT_DIR and $CKPT_ROOT_DIR):
- Data:
  - $DATA_ROOT_DIR/cifar10/cifar10-32x32.zip
  - $DATA_ROOT_DIR/imagenet-64/imagenet-64x64.zip (EDM format)
  - $DATA_ROOT_DIR/imagenet-64/imagenet-64x64-edmv2.zip (EDM2 format)
  - $DATA_ROOT_DIR/imagenet-256/imagenet_256_sd.zip (EDM2 VAE-encoded)
- Models:
  - $CKPT_ROOT_DIR/cifar10/edm-cifar10-32x32-{uncond,cond}-vp.pth
  - $CKPT_ROOT_DIR/imagenet-64/edm-imagenet-64x64-cond-adm.pth
  - $CKPT_ROOT_DIR/imagenet-64/edm2-img64-{s,m,l,xl}-fid.pth

The EDM/EDM2 repos are cloned temporarily to:
- Use dataset_tool.py for proper dataset conversion
- Unpickle the .pkl model files (which require the repos' custom modules)

Usage:
    # Download CIFAR-10 (default):
    python scripts/download_data.py

    # Download ImageNet-64 (requires Kaggle ImageNet download):
    python scripts/download_data.py --dataset imagenet-64 --imagenet-source /path/to/imagenet

    # Download ImageNet-256 with VAE encoding (for latent diffusion):
    python scripts/download_data.py --dataset imagenet-256 --imagenet-source /path/to/imagenet

    # Download only data or only models:
    python scripts/download_data.py --only-data
    python scripts/download_data.py --only-models

    # Specify custom directories:
    python scripts/download_data.py --output-dir /path/to/data --ckpt-dir /path/to/checkpoints
"""

import argparse
import hashlib
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

import fastgen.utils.logging_utils as logger
from fastgen.utils.logging_utils import set_log_level
from fastgen.configs.data import DATA_ROOT_DIR
from fastgen.configs.net import CKPT_ROOT_DIR


# URLs for downloads
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"

EDM_BASE_URL = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
EDM2_BASE_URL = "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions"

# EDM models (CIFAR-10 and ImageNet-64)
EDM_CIFAR10_MODELS = {
    "edm-cifar10-32x32-uncond-vp": {
        "url": f"{EDM_BASE_URL}/edm-cifar10-32x32-uncond-vp.pkl",
        "output": "edm-cifar10-32x32-uncond-vp.pth",
    },
    "edm-cifar10-32x32-cond-vp": {
        "url": f"{EDM_BASE_URL}/edm-cifar10-32x32-cond-vp.pkl",
        "output": "edm-cifar10-32x32-cond-vp.pth",
    },
}

EDM_IMAGENET64_MODELS = {
    "edm-imagenet-64x64-cond-adm": {
        "url": f"{EDM_BASE_URL}/edm-imagenet-64x64-cond-adm.pkl",
        "output": "edm-imagenet-64x64-cond-adm.pth",
    },
}

# EDM2 models (ImageNet-64, posthoc-reconstruction versions for best FID)
EDM2_IMAGENET64_MODELS = {
    "edm2-img64-s-fid": {
        "url": f"{EDM2_BASE_URL}/edm2-img64-s-1073741-0.075.pkl",
        "output": "edm2-img64-s-fid.pth",
    },
    "edm2-img64-m-fid": {
        "url": f"{EDM2_BASE_URL}/edm2-img64-m-2147483-0.060.pkl",
        "output": "edm2-img64-m-fid.pth",
    },
    "edm2-img64-l-fid": {
        "url": f"{EDM2_BASE_URL}/edm2-img64-l-1073741-0.040.pkl",
        "output": "edm2-img64-l-fid.pth",
    },
    "edm2-img64-xl-fid": {
        "url": f"{EDM2_BASE_URL}/edm2-img64-xl-0671088-0.040.pkl",
        "output": "edm2-img64-xl-fid.pth",
    },
}

EDM_REPO_URL = "https://github.com/NVlabs/edm.git"
EDM2_REPO_URL = "https://github.com/NVlabs/edm2.git"

# ImageNet data paths (relative to the Kaggle download root)
IMAGENET_TRAIN_SUBPATH = "ILSVRC/Data/CLS-LOC/train"


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url: str, output_path: Path, description: str = "Downloading", expected_md5: Optional[str] = None):
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists and has correct hash
    if output_path.exists() and expected_md5:
        if compute_md5(output_path) == expected_md5:
            logger.info(f"File already exists and verified: {output_path}")
            return
        else:
            logger.warning(f"File exists but hash mismatch, re-downloading: {output_path}")

    headers = {"User-Agent": "FastGen/1.0"}
    from urllib.request import urlopen, Request

    req = Request(url, headers=headers)

    with urlopen(req) as response:
        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=description) as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Verify hash if provided
    if expected_md5:
        actual_md5 = compute_md5(output_path)
        if actual_md5 != expected_md5:
            raise ValueError(f"MD5 mismatch: expected {expected_md5}, got {actual_md5}")
        logger.debug(f"MD5 verified: {expected_md5}")


def clone_repo(repo_url: str, target_dir: Path, name: str) -> Path:
    """Clone a git repository."""
    repo_dir = target_dir / name

    if repo_dir.exists():
        logger.debug(f"{name} repo already exists at {repo_dir}")
        return repo_dir

    logger.info(f"Cloning {name} repo to {repo_dir}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
        check=True,
        capture_output=True,
    )

    return repo_dir


def run_dataset_tool(
    repo_dir: Path,
    source: Path,
    dest: Path,
    resolution: Optional[str] = None,
    transform: Optional[str] = None,
    use_subcommand: bool = False,
    subcommand: str = "convert",
) -> None:
    """Run dataset_tool.py with common error handling."""
    dataset_tool = repo_dir / "dataset_tool.py"
    if not dataset_tool.exists():
        raise FileNotFoundError(f"dataset_tool.py not found at {dataset_tool}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(dataset_tool.absolute())]
    if use_subcommand:
        cmd.append(subcommand)
    cmd.extend([f"--source={source.absolute()}", f"--dest={dest.absolute()}"])
    if resolution:
        cmd.append(f"--resolution={resolution}")
    if transform:
        cmd.append(f"--transform={transform}")

    logger.debug(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
        raise RuntimeError(f"dataset_tool.py failed with return code {result.returncode}")

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            logger.debug(line)

    if not dest.exists():
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
        raise RuntimeError(f"dataset_tool.py did not create output file at {dest}")


def convert_pickle_to_pth(pkl_path: Path, pth_path: Path, repo_dir: Path):
    """
    Convert an EDM/EDM2 pickle file to a PyTorch state dict.

    The pickle files contain the full network object with custom modules,
    so we need the repo in the path to unpickle them. The 'ema' key contains
    the network with EMA weights (preferred for inference).
    """
    import torch

    # Add repo to path for unpickling
    repo_path = str(repo_dir)
    original_path = sys.path.copy()
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        logger.info(f"Loading {pkl_path.name}...")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Extract network from pickle
        if isinstance(data, dict):
            if "ema" in data:
                logger.debug("Found 'ema' key in pickle, using EMA weights")
                network = data["ema"]
            elif "model" in data:
                logger.debug("Found 'model' key in pickle")
                network = data["model"]
            else:
                first_key = next(iter(data.keys())) if data else None
                if first_key and isinstance(data[first_key], torch.Tensor):
                    logger.debug("Pickle appears to be a state dict")
                    state_dict = data
                    network = None
                else:
                    raise ValueError(f"Unknown pickle format. Keys: {list(data.keys())}")
        else:
            network = data

        if network is not None:
            logger.debug(f"Network type: {type(network).__name__}")
            if hasattr(network, "state_dict"):
                state_dict = network.state_dict()
            else:
                raise ValueError(f"Cannot extract state dict from {type(network)}")

        logger.debug(f"State dict has {len(state_dict)} keys")
        logger.debug(f"First 5 keys: {list(state_dict.keys())[:5]}")

        logger.info(f"Saving to {pth_path.name}...")
        pth_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, pth_path)

        # Verify
        loaded = torch.load(pth_path, weights_only=True)
        logger.debug(f"Verified: saved file has {len(loaded)} keys")

    finally:
        sys.path = original_path


def prepare_models(
    models: Dict[str, dict],
    ckpt_dir: Path,
    output_subdir: str,
    repo_dir: Path,
    tmpdir: Path,
    description: str,
    force: bool = False,
):
    """Generic function to download and convert pretrained models."""
    output_dir = ckpt_dir / output_subdir

    # Check if all models already exist
    all_exist = all((output_dir / model["output"]).exists() for model in models.values())

    if all_exist and not force:
        logger.info(f"All {description} models already exist:")
        for model in models.values():
            logger.info(f"  {output_dir / model['output']}")
        logger.info("Use --force to re-download and convert")
        return

    logger.info(f"Preparing {description} pretrained models")

    for i, (name, model) in enumerate(models.items(), start=1):
        output_path = output_dir / model["output"]

        if output_path.exists() and not force:
            logger.info(f"{i}. {name} already exists at {output_path}")
            continue

        logger.info(f"{i}. Processing {name}")

        pkl_path = tmpdir / f"{name}.pkl"
        logger.info(f"Downloading from {model['url']}")
        download_file(model["url"], pkl_path, f"Downloading {name}")

        convert_pickle_to_pth(pkl_path, output_path, repo_dir)

        logger.success(f"Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    logger.success(f"{description} models prepared successfully!")


def prepare_cifar10_data(output_dir: Path, edm_dir: Path, force: bool = False):
    """Download and prepare CIFAR-10 data in EDM format."""
    output_path = output_dir / "cifar10" / "cifar10-32x32.zip"

    if output_path.exists() and not force:
        logger.info(f"CIFAR-10 data already exists at {output_path}")
        logger.info("Use --force to re-download and recreate")
        return

    logger.info("Preparing CIFAR-10 data")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tar_path = tmpdir / "cifar-10-python.tar.gz"
        logger.info(f"Downloading CIFAR-10 from {CIFAR10_URL}")
        download_file(CIFAR10_URL, tar_path, "Downloading CIFAR-10", CIFAR10_MD5)

        logger.info("Converting to EDM format using dataset_tool.py...")
        run_dataset_tool(edm_dir, tar_path, output_path)

    logger.success(f"CIFAR-10 data prepared: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def validate_imagenet_source(imagenet_source: Path) -> Path:
    """Validate and return the ImageNet training data path."""
    train_path = imagenet_source / IMAGENET_TRAIN_SUBPATH
    if not train_path.exists():
        raise FileNotFoundError(
            f"ImageNet training data not found at {train_path}. "
            f"Please provide the path to your Kaggle ImageNet download via --imagenet-source. "
            f"Expected structure: <imagenet-source>/{IMAGENET_TRAIN_SUBPATH}"
        )
    return train_path


def prepare_imagenet_data(
    output_dir: Path,
    repo_dir: Path,
    imagenet_source: Path,
    output_subdir: str,
    output_filename: str,
    resolution: str,
    transform: str,
    description: str,
    use_edm2_format: bool = False,
    vae_encode: bool = False,
    force: bool = False,
):
    """Generic function to prepare ImageNet data in EDM/EDM2 format.

    Args:
        vae_encode: If True, first convert to RGB then encode through VAE (for latent diffusion).
    """
    output_path = output_dir / output_subdir / output_filename

    if output_path.exists() and not force:
        logger.info(f"{description} data already exists at {output_path}")
        logger.info("Use --force to recreate")
        return

    train_path = validate_imagenet_source(imagenet_source)
    logger.info(f"Preparing {description} data")
    logger.info(f"Source: {train_path}")
    logger.info("This may take a while...")

    if vae_encode:
        # Two-step process: convert to RGB, then encode through VAE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            rgb_path = Path(tmpdir) / "rgb_dataset.zip"

            logger.info("Step 1/2: Converting to RGB format...")
            run_dataset_tool(
                repo_dir,
                train_path,
                rgb_path,
                resolution=resolution,
                transform=transform,
                use_subcommand=True,
                subcommand="convert",
            )
            logger.info(f"RGB dataset created: {rgb_path} ({rgb_path.stat().st_size / 1024 / 1024:.1f} MB)")

            logger.info("Step 2/2: Encoding through VAE (this may take a long time)...")
            run_dataset_tool(
                repo_dir,
                rgb_path,
                output_path,
                use_subcommand=True,
                subcommand="encode",
            )
    else:
        run_dataset_tool(
            repo_dir,
            train_path,
            output_path,
            resolution=resolution,
            transform=transform,
            use_subcommand=use_edm2_format,
            subcommand="convert",
        )

    logger.success(f"{description} data prepared: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare CIFAR-10 and ImageNet models and data for FastGen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download CIFAR-10 (default):
    python scripts/download_data.py

    # Download ImageNet-64 (requires Kaggle ImageNet download):
    python scripts/download_data.py --dataset imagenet-64 --imagenet-source /path/to/imagenet

    # Download ImageNet-256 with VAE encoding:
    python scripts/download_data.py --dataset imagenet-256 --imagenet-source /path/to/imagenet

    # Download all datasets:
    python scripts/download_data.py --dataset all --imagenet-source /path/to/imagenet

    # Specify custom directories:
    python scripts/download_data.py --output-dir ./data --ckpt-dir ./checkpoints

    # Download only data or models:
    python scripts/download_data.py --only-data
    python scripts/download_data.py --only-models

    # Force re-download:
    python scripts/download_data.py --force
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["cifar10", "imagenet-64", "imagenet-256", "all"],
        help="Dataset to prepare (default: cifar10). 'imagenet-64' prepares both EDM and EDM2 formats.",
    )
    parser.add_argument(
        "--imagenet-source",
        type=Path,
        default=None,
        help="Path to Kaggle ImageNet download (directory containing ILSVRC/). Required for ImageNet datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory for datasets (default: $DATA_ROOT_DIR)",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Root directory for model checkpoints (default: $CKPT_ROOT_DIR)",
    )
    parser.add_argument(
        "--only-data",
        action="store_true",
        help="Only download and prepare data, skip models",
    )
    parser.add_argument(
        "--only-models",
        action="store_true",
        help="Only download and convert models, skip data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()
    set_log_level(args.log_level)

    # Validate ImageNet source requirement
    if args.dataset in ["imagenet-64", "imagenet-256", "all"] and args.imagenet_source is None:
        if not args.only_models:
            parser.error(f"--imagenet-source is required for dataset '{args.dataset}' (unless --only-models is set)")

    # Determine output directories
    output_dir = args.output_dir or Path(DATA_ROOT_DIR)
    ckpt_dir = args.ckpt_dir or Path(CKPT_ROOT_DIR)

    logger.info(f"FastGen Dataset Setup: {args.dataset}")
    logger.info(f"Data directory:       {output_dir.absolute()}")
    logger.info(f"Checkpoint directory: {ckpt_dir.absolute()}")
    if args.imagenet_source:
        logger.info(f"ImageNet source:      {args.imagenet_source.absolute()}")

    # Determine which repos we need
    need_edm = args.dataset in ["cifar10", "imagenet-64", "all"]
    need_edm2 = args.dataset in ["imagenet-64", "imagenet-256", "all"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        edm_dir = clone_repo(EDM_REPO_URL, tmpdir, "edm") if need_edm else None
        edm2_dir = clone_repo(EDM2_REPO_URL, tmpdir, "edm2") if need_edm2 else None

        # ============ CIFAR-10 ============
        if args.dataset in ["cifar10", "all"]:
            logger.info("=" * 50)
            logger.info("Processing CIFAR-10")
            logger.info("=" * 50)

            if not args.only_models:
                prepare_cifar10_data(output_dir, edm_dir, force=args.force)

            if not args.only_data:
                prepare_models(
                    EDM_CIFAR10_MODELS, ckpt_dir, "cifar10", edm_dir, tmpdir, "EDM CIFAR-10", force=args.force
                )

        # ============ ImageNet-64 ============
        if args.dataset in ["imagenet-64", "all"]:
            logger.info("=" * 50)
            logger.info("Processing ImageNet-64")
            logger.info("=" * 50)

            if not args.only_models:
                # EDM format
                prepare_imagenet_data(
                    output_dir,
                    edm_dir,
                    args.imagenet_source,
                    output_subdir="imagenet-64",
                    output_filename="imagenet-64x64.zip",
                    resolution="64x64",
                    transform="center-crop",
                    description="ImageNet-64 (EDM)",
                    use_edm2_format=False,
                    force=args.force,
                )
                # EDM2 format
                prepare_imagenet_data(
                    output_dir,
                    edm2_dir,
                    args.imagenet_source,
                    output_subdir="imagenet-64",
                    output_filename="imagenet-64x64-edmv2.zip",
                    resolution="64x64",
                    transform="center-crop-dhariwal",
                    description="ImageNet-64 (EDM2)",
                    use_edm2_format=True,
                    force=args.force,
                )

            if not args.only_data:
                prepare_models(
                    EDM_IMAGENET64_MODELS, ckpt_dir, "imagenet-64", edm_dir, tmpdir, "EDM ImageNet-64", force=args.force
                )
                prepare_models(
                    EDM2_IMAGENET64_MODELS,
                    ckpt_dir,
                    "imagenet-64",
                    edm2_dir,
                    tmpdir,
                    "EDM2 ImageNet-64",
                    force=args.force,
                )

        # ============ ImageNet-256 ============
        if args.dataset in ["imagenet-256", "all"]:
            logger.info("=" * 50)
            logger.info("Processing ImageNet-256")
            logger.info("=" * 50)

            if not args.only_models:
                prepare_imagenet_data(
                    output_dir,
                    edm2_dir,
                    args.imagenet_source,
                    output_subdir="imagenet-256",
                    output_filename="imagenet_256_sd.zip",
                    resolution="256x256",
                    transform="center-crop-dhariwal",
                    description="ImageNet-256 (EDM2 latent)",
                    use_edm2_format=True,
                    vae_encode=True,
                    force=args.force,
                )

    logger.success("Setup complete!")

    # Print summary
    logger.info("")
    logger.info("Output locations:")

    if args.dataset in ["cifar10", "all"]:
        if not args.only_models:
            logger.info(f"  CIFAR-10 data: {output_dir / 'cifar10' / 'cifar10-32x32.zip'}")
        if not args.only_data:
            logger.info("  CIFAR-10 models:")
            for model in EDM_CIFAR10_MODELS.values():
                logger.info(f"    {ckpt_dir / 'cifar10' / model['output']}")

    if args.dataset in ["imagenet-64", "all"]:
        if not args.only_models:
            logger.info(f"  ImageNet-64 data (EDM): {output_dir / 'imagenet-64' / 'imagenet-64x64.zip'}")
            logger.info(f"  ImageNet-64 data (EDM2): {output_dir / 'imagenet-64' / 'imagenet-64x64-edmv2.zip'}")
        if not args.only_data:
            logger.info("  ImageNet-64 models (EDM):")
            for model in EDM_IMAGENET64_MODELS.values():
                logger.info(f"    {ckpt_dir / 'imagenet-64' / model['output']}")
            logger.info("  ImageNet-64 models (EDM2):")
            for model in EDM2_IMAGENET64_MODELS.values():
                logger.info(f"    {ckpt_dir / 'imagenet-64' / model['output']}")

    if args.dataset in ["imagenet-256", "all"]:
        if not args.only_models:
            logger.info(f"  ImageNet-256 data (latent): {output_dir / 'imagenet-256' / 'imagenet_256_sd.zip'}")

    logger.info("")
    logger.info("Example training commands:")
    if args.dataset in ["cifar10", "all"]:
        logger.info("  # CIFAR-10:")
        logger.info(f"  DATA_ROOT_DIR={output_dir} CKPT_ROOT_DIR={ckpt_dir} python train.py \\")
        logger.info("    --config=fastgen/configs/experiments/EDM/config_dmd2_cifar10.py")
    if args.dataset in ["imagenet-64", "all"]:
        logger.info("  # ImageNet-64:")
        logger.info(f"  DATA_ROOT_DIR={output_dir} CKPT_ROOT_DIR={ckpt_dir} python train.py \\")
        logger.info("    --config=fastgen/configs/experiments/EDM/config_dmd2_in64.py")


if __name__ == "__main__":
    main()
