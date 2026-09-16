"""Shared deterministic training and inference helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(requested="auto"):
    requested = str(requested).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device


def make_loader(dataset, *, batch_size, shuffle, seed, num_workers=0):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=int(num_workers) > 0,
    )


def train_one_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()
        count = int(images.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
    return total_loss / total_count


@torch.inference_mode()
def collect_predictions(model, loader, device, statistics, loss_function=None):
    model.eval()
    sample_ids = []
    targets = []
    predictions = []
    total_loss = 0.0
    total_count = 0
    target_mean = float(statistics["target_mean"])
    target_std = float(statistics["target_std"])
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        normalized_targets = batch["target"].to(device, non_blocking=True)
        normalized_predictions = model(images)
        if loss_function is not None:
            loss = loss_function(normalized_predictions, normalized_targets)
            total_loss += float(loss.detach().cpu()) * int(images.shape[0])
        raw_predictions = normalized_predictions.detach().cpu().numpy() * target_std + target_mean
        sample_ids.extend(str(value) for value in batch["sample_id"])
        targets.extend(batch["target_raw"].detach().cpu().numpy().astype(float).tolist())
        predictions.extend(raw_predictions.astype(float).tolist())
        total_count += int(images.shape[0])
    return {
        "sample_ids": sample_ids,
        "targets": np.asarray(targets, dtype=np.float64),
        "predictions": np.asarray(predictions, dtype=np.float64),
        "normalized_loss": None if loss_function is None else total_loss / total_count,
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_output_directory(path, overwrite=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Pass --overwrite to replace files.")
    path.mkdir(parents=True, exist_ok=True)
    return path
