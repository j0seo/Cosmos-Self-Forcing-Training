# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import glob
import pickle
import re
import json
import click
import tqdm
import numpy as np
import scipy.linalg

import torch

from fastgen.networks.inception import InceptionV3
from fastgen.datasets.class_cond_dataset import ImageFolderDataset
from fastgen.utils.distributed import get_rank, is_rank0, synchronize, world_size
import fastgen.utils.logging_utils as logger
from fastgen.utils.io_utils import open_url
from fastgen.configs.data import DATA_ROOT_DIR


def calculate_inception_stats(
    detector_net,
    feature_dim,
    image_path,
    num_expected=None,
    seed=0,
    max_batch_size=64,
    num_workers=3,
    prefetch_factor=2,
    device=torch.device("cuda"),
):
    # Rank 0 goes first.
    if not is_rank0():
        synchronize()

    # List images.
    logger.info(f'Loading images from "{image_path}"...')
    dataset_obj = ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f"Found {len(dataset_obj)} images, but expected at least {num_expected}")
    if len(dataset_obj) < 2:
        raise click.ClickException(f"Found {len(dataset_obj)} images, but need at least 2 to compute statistics")

    # Other ranks follow.
    if is_rank0():
        synchronize()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * world_size()) + 1) * world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[get_rank() :: world_size()]
    data_loader = torch.utils.data.DataLoader(
        dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor
    )

    # Accumulate statistics.
    logger.info(f"Calculating statistics for {len(dataset_obj)} images...")
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for data in tqdm.tqdm(data_loader, unit="batch", disable=(get_rank() != 0)):
        synchronize()
        images = data["real"]

        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        with torch.no_grad():
            features = detector_net(images.to(device))
            features = features.to(torch.float64)

        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate grand totals.
    if world_size() > 1:
        torch.distributed.all_reduce(mu)
        torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


def calc(
    samples_dir, num_expected, seed, min_ckpt, max_ckpt, batch, dataset, regenerate=False, device=torch.device("cuda")
):
    """Calculate FID for a given set of images."""

    ref = None
    if dataset == "cifar10":
        ref_path = f"{DATA_ROOT_DIR}/fid-refs/cifar10-32x32.npz"
    elif dataset == "imagenet64":
        ref_path = f"{DATA_ROOT_DIR}/fid-refs/imagenet-64x64.npz"
    elif dataset == "imagenet64-edmv2":
        ref_path = f"{DATA_ROOT_DIR}/fid-refs/imagenet-64x64-edmv2.npz"
    elif dataset == "imagenet256":
        ref_path = f"{DATA_ROOT_DIR}/fid-refs/imagenet_256.pkl"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    logger.info(f'Loading dataset reference statistics from "{ref_path}"...')
    if is_rank0():
        if ref_path.endswith(".npz"):
            with open_url(ref_path) as f:
                ref = dict(np.load(f))
        else:
            assert ref_path.endswith(".pkl"), f"Unknown file type: {ref_path}"
            with open_url(ref_path) as f:
                ref = pickle.load(f)["fid"]

    stats = glob.glob(f"{samples_dir}/iter_[0-9]*")
    stats.sort(key=lambda x: int(re.search(r"iter_(\d+)", x).group(1)))

    ckpt_num_list = []
    fid_list = []
    if os.path.exists(f"{samples_dir}/fid.json"):
        with open(f"{samples_dir}/fid.json", "r") as f:
            metric_scores = json.load(f)
        logger.info(f"metric_scores in the existing file: {metric_scores}")
        ckpt_num_list = metric_scores["ckpt_num"]
        fid_list = metric_scores["fid"]

    # Load Inception-v3 model.
    logger.info("Loading Inception-v3 model...")
    feature_dim = 2048
    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[feature_dim]
    # detector_net = InceptionV3([block_idx], resize_input=False, normalize_input=False).to(device)
    detector_net = InceptionV3().to(device)
    detector_net.eval()

    for path in stats:
        ckpt_num = int(re.search(r"iter_(\d+)", path).group(1))

        if ckpt_num in ckpt_num_list and not regenerate:
            logger.info(f"ckpt {ckpt_num} already has metrics. Skip.")
            continue

        if ckpt_num < min_ckpt or ckpt_num > max_ckpt:
            continue

        mu, sigma = calculate_inception_stats(
            detector_net, feature_dim, image_path=path, num_expected=num_expected, seed=seed, max_batch_size=batch
        )

        logger.info(f"Calculating FID for {path}... ")
        if is_rank0():
            fid = calculate_fid_from_inception_stats(mu, sigma, ref["mu"], ref["sigma"])
            logger.info(f"path: {path}")
            logger.info(f"FID: {fid}")
            logger.info("=" * 20)
            fid_list.append(fid)
            ckpt_num_list.append(ckpt_num)

        synchronize()

    # dump the FID scores to a json file
    if is_rank0():
        metric_scores = {}
        # read metrics again in case another process altered file
        if os.path.exists(f"{samples_dir}/fid.json"):
            with open(f"{samples_dir}/fid.json", "r") as f:
                metric_scores = json.load(f)
            metric_scores = {ckpt: fid for ckpt, fid in zip(metric_scores["ckpt_num"], metric_scores["fid"])}

        # merge metrics
        for ckpt, fid in zip(ckpt_num_list, fid_list):
            metric_scores[ckpt] = fid
        metric_scores = sorted(metric_scores.items(), key=lambda x: x[0])
        metric_scores = {"ckpt_num": [ckpt for ckpt, _ in metric_scores], "fid": [fid for _, fid in metric_scores]}
        with open(f"{samples_dir}/fid.json", "w") as f:
            json.dump(metric_scores, f)
