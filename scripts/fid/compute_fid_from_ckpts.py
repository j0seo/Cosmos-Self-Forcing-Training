# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import argparse
import os
import re
import tqdm
import time
import numpy as np
import PIL.Image
import glob
import json
import pandas as pd
from pathlib import Path
import shutil

import torch
from torchvision.utils import make_grid, save_image

import fastgen.utils.logging_utils as logger
from fastgen.configs.config import BaseConfig
from fastgen.utils import instantiate
from fastgen.utils.checkpointer import Checkpointer, FSDPCheckpointer
from fastgen.utils import basic_utils
from fastgen.utils.distributed import world_size, get_rank, synchronize, clean_up
from fastgen.configs.data import DATA_ROOT_DIR
from fastgen.utils.scripts import setup, parse_args

from fastgen.eval.fid.fid import calc

DATASETS = {
    "cifar10-32x32.zip": "cifar10",
    "imagenet-64x64.zip": "imagenet64",
    "imagenet-64x64-edmv2.zip": "imagenet64-edmv2",
    "imagenet_256_sd.zip": "imagenet256",
}


"""Generating samples, then calling FID score for evaluation.

Examples: 
    PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT='FASTGEN_OUTPUT' torchrun --nproc_per_node=8 --standalone eval/compute_fid_from_ckpts.py --config fastgen/configs/experiments/EDM/config_dmd2_cifar10.py - log_config.name=dmd2_cifar10 trainer.ddp=True
"""


def remove_iter_dirs(root_dir: str, ckpt_num_visited: list) -> None:
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"✗ Folder not found: {root.resolve()}")

    removed, errors = 0, 0
    visited_set = set(str(x) for x in ckpt_num_visited)
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("iter_"):
            suffix = path.name[5:]  # remove "iter_" prefix
            if suffix in visited_set:
                try:
                    shutil.rmtree(path)
                    removed += 1
                except Exception as exc:
                    errors += 1
                    raise RuntimeError(f"  – Could not delete {path}: {exc}")

    logger.info(f"✓ Removed {removed} directorie(s) (errors: {errors}) from {root.resolve()}")


def main(config: BaseConfig):
    # fix seeds
    basic_utils.set_random_seed(config.trainer.seed, by_rank=True)

    # Initialize the model.
    config.model_class.config = config.model
    model = instantiate(config.model_class)
    config.model_class.config = None

    # Initialize the checkpointer and samples directory.
    if config.trainer.fsdp:
        checkpointer = FSDPCheckpointer(config.trainer.checkpointer)
    else:
        checkpointer = Checkpointer(config.trainer.checkpointer)
    samples_dir = os.path.join(config.log_config.save_path, config.eval.samples_dir)

    try:
        dataset = DATASETS[config.dataloader_train.dataset_path.split("/")[-1]]
    except (KeyError, AttributeError):
        dataset = DATASETS[config.dataloader_train.datatags[0].split("/")[-1]]

    if dataset == "coco2014":
        path = Path(f"{DATA_ROOT_DIR}/prompts/coco2014_eval_prompt_30k.csv")
        df = pd.read_csv(path)
        all_text = list(df["caption"])
        config.eval.num_samples = min(len(all_text), config.eval.num_samples)

    # Initialize the batches.
    num_batches = (
        (config.eval.num_samples - 1) // (config.dataloader_train.batch_size * world_size()) + 1
    ) * world_size()
    logger.info(f"world size: {world_size()}, num batches: {num_batches}")
    all_batches = torch.as_tensor(np.arange(config.eval.num_samples)).tensor_split(num_batches)
    rank_batches = all_batches[get_rank() :: world_size()]

    # Get the list of checkpoints.
    stats = glob.glob(f"{config.trainer.checkpointer.save_dir}/*.pth")
    filter_stats = [path for path in stats if re.search(r"(\d+).pth", path) is not None]
    filter_stats.sort(key=lambda x: int(re.search(r"(\d+).pth", x).group(1)))

    # Load previously saved fid file to skip redundant ckpt evaluations
    fid_runs_file = f"{samples_dir}/fid.json"
    runs_visited = []
    if os.path.isfile(fid_runs_file):
        with open(fid_runs_file, "r", encoding="utf-8") as f:
            fid_runs = json.load(f)
        assert "ckpt_num" in fid_runs.keys() and "fid" in fid_runs.keys()
        assert len(fid_runs["ckpt_num"]) == len(fid_runs["fid"])
        runs_visited = fid_runs["ckpt_num"]

    logger.info(f"Evaluating student sample steps: {model.config.student_sample_steps}")
    # sweep over all checkpoints
    for ckpt_path in filter_stats:
        # Load network.
        synchronize()
        ckpt_num = int(re.search(r"(\d+).pth", ckpt_path).group(1))

        if ckpt_num < config.eval.min_ckpt or ckpt_num > config.eval.max_ckpt:
            continue

        # check if we evaluated fid already
        if ckpt_num in runs_visited:
            logger.info(f"Skipping checkpoint {ckpt_path} since we already evaluated its FID in fid.json")
            continue

        outdir = os.path.join(samples_dir, f"iter_{ckpt_num}")
        if (
            os.path.exists(outdir)
            and len(glob.glob(os.path.join(outdir, "**"), recursive=True)) >= config.eval.num_samples
            and not config.eval.save_images
        ):
            logger.info(f"Skipping checkpoint {ckpt_path} since there are already {config.eval.num_samples} samples")
            continue

        logger.info(f'Loading model from "{ckpt_path}"...')
        checkpointer.load(
            model.model_dict,
            path=ckpt_path,
        )
        model.on_train_begin()
        inference_net = getattr(model, model.use_ema[0]) if model.use_ema else model.net
        inference_net.eval()
        ctx = dict(device=model.device, dtype=model.precision)

        if hasattr(model.net, "init_preprocessors") and config.model.enable_preprocessors:
            inference_net.init_preprocessors()
            inference_net.vae.to(**ctx)

        # Loop over batches.
        conditional = dataset in ["coco2014", "imagenet256"] or inference_net.label_dim > 0
        logger.info(
            f"{'Conditional' if conditional else 'Unconditional'} sampling of {config.eval.num_samples} "
            f"images to {outdir}..."
        )

        for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(get_rank() != 0)):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            if dataset == "coco2014":
                # text-to-image generation on coco2014 prompt set
                sublist = [all_text[i] for i in batch_seeds]
                with basic_utils.inference_mode(
                    inference_net.text_encoder,
                    precision_amp=model.precision_amp_enc,
                    device_type=model.device.type,
                ):
                    condition = basic_utils.to(inference_net.text_encoder.encode(sublist), **ctx)
            elif dataset == "imagenet256":
                condition = torch.randint(1000, size=[batch_size], device=model.device)
            elif conditional:
                condition = torch.eye(inference_net.label_dim, **ctx)[
                    torch.randint(inference_net.label_dim, size=[batch_size], device=model.device)
                ]
            else:
                condition = None

            # Pick noise and labels.
            noise = torch.randn([batch_size, *model.input_shape], **ctx)

            images = model.generator_fn(
                inference_net,
                noise,
                condition=condition,
                student_sample_steps=model.config.student_sample_steps,
                student_sample_type=model.config.student_sample_type,
                t_list=model.config.sample_t_cfg.t_list,
                precision_amp=model.precision_amp_infer,
            )

            if hasattr(model.net, "init_preprocessors") and config.model.enable_preprocessors:
                with basic_utils.inference_mode(
                    inference_net.vae, precision_amp=model.precision_amp_enc, device_type=model.device.type
                ):
                    images = inference_net.vae.decode(images)

            if config.eval.save_images:
                visdir = os.path.join(outdir, "vis")
                os.makedirs(visdir, exist_ok=True)
                # save a small batch of images
                images_ = (images + 1) / 2.0
                image_grid = make_grid(images_, nrow=int(np.sqrt(len(images))), padding=0)
                save_image(image_grid, os.path.join(visdir, f"{ckpt_num}.png"))
                logger.info(f"Saved {len(images)} images to {visdir}/{ckpt_num}.png")
                break

            # Save images.
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, f"{seed - seed % 1000:06d}")
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f"{seed:06d}.png")
                PIL.Image.fromarray(image_np, "RGB").save(image_path)

    if config.eval.save_images:
        exit(0)

    synchronize()
    time.sleep(10)

    calc(
        samples_dir,
        config.eval.num_samples,
        config.trainer.seed,
        config.eval.min_ckpt,
        config.eval.max_ckpt,
        config.dataloader_train.batch_size,
        dataset,
        device=model.device,
    )

    if get_rank() == 0:
        # remove generated samples to free the occupied space
        runs_visited_new = []
        if os.path.isfile(fid_runs_file):
            with open(fid_runs_file, "r", encoding="utf-8") as f:
                fid_runs = json.load(f)
            assert "ckpt_num" in fid_runs.keys() and "fid" in fid_runs.keys()
            assert len(fid_runs["ckpt_num"]) == len(fid_runs["fid"])
            runs_visited_new = [
                ckpt_num
                for ckpt_num in fid_runs["ckpt_num"]
                if ckpt_num >= config.eval.min_ckpt
                and ckpt_num <= config.eval.max_ckpt
                and ckpt_num not in runs_visited
            ]
        if runs_visited_new:
            remove_iter_dirs(samples_dir, runs_visited_new)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FID evaluation")
    args = parse_args(parser)
    config = setup(args, evaluation=True)
    synchronize()
    main(config)

    clean_up()

# ----------------------------------------------------------------------------
