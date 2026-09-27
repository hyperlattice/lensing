"""PyTorch data adapter with train-only normalization and no truth leakage."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import LensDataset


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TorchLensDataset(Dataset):
    """Expose only an image, normalized scalar target, and opaque sample ID.

    The full manifest row remains inside ``LensDataset`` and is never returned
    by this adapter. ``sample_id`` is provided solely for joining predictions
    to evaluation diagnostics and must not be passed to the model.
    """

    def __init__(
        self,
        release,
        *,
        split,
        target_field="theta_E_arcsec",
        statistics=None,
        max_samples=None,
        sample_ids=None,
        expected_shape=(256, 256),
    ):
        self.release = Path(release)
        self.split = str(split)
        self.target_field = str(target_field)
        self.dataset = LensDataset(
            self.release,
            root=self.release,
            split=self.split,
            domain="simulated",
            eligible_for="ml",
            target_fields={self.target_field},
            expected_shape=expected_shape,
        )
        if sample_ids is not None and max_samples is not None:
            raise ValueError("Use explicit sample_ids or max_samples, not both")
        count = len(self.dataset)
        if max_samples is not None:
            count = min(count, int(max_samples))
        if sample_ids is not None:
            requested_ids = [str(value) for value in sample_ids]
            if not requested_ids:
                raise ValueError(f"Split {self.split!r} contains no selected samples")
            if len(requested_ids) != len(set(requested_ids)):
                raise ValueError("Explicit sample_ids contains duplicates")
            available = {
                str(sample_id): index
                for index, sample_id in enumerate(self.dataset.metadata["sample_id"].astype(str))
            }
            unknown = sorted(set(requested_ids).difference(available))
            if unknown:
                raise ValueError(f"Explicit sample_ids are absent from split {self.split!r}: {unknown[:5]}")
            self.indices = [available[sample_id] for sample_id in requested_ids]
        elif count > 0:
            self.indices = list(range(count))
        else:
            raise ValueError(f"Split {self.split!r} contains no usable samples")
        self.statistics = statistics
        if statistics is not None:
            required = {"image_mean", "image_std", "target_mean", "target_std"}
            missing = required.difference(statistics)
            if missing:
                raise ValueError(f"Normalization statistics are missing: {sorted(missing)}")
            if float(statistics["image_std"]) <= 0 or float(statistics["target_std"]) <= 0:
                raise ValueError("Normalization standard deviations must be positive")

    def __len__(self):
        return len(self.indices)

    def raw_item(self, index):
        image, row = self.dataset[self.indices[index]]
        available = bool(row.get("theta_E_target_available", False))
        target = row.get(self.target_field)
        if not available or target is None or not np.isfinite(float(target)):
            raise ValueError(f"{row['sample_id']}: unavailable target {self.target_field}")
        return np.asarray(image, dtype=np.float32), float(target), str(row["sample_id"])

    def __getitem__(self, index):
        image, target, sample_id = self.raw_item(index)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        target_value = target
        if self.statistics is not None:
            image_tensor = (image_tensor - float(self.statistics["image_mean"])) / float(
                self.statistics["image_std"]
            )
            target_value = (target - float(self.statistics["target_mean"])) / float(
                self.statistics["target_std"]
            )
        return {
            "image": image_tensor,
            "target": torch.tensor(target_value, dtype=torch.float32),
            "target_raw": torch.tensor(target, dtype=torch.float32),
            "sample_id": sample_id,
        }


def compute_training_statistics(dataset):
    """Compute global image and scalar-target statistics from one split."""

    if dataset.split != "train":
        raise ValueError("Normalization statistics must come from the train split")
    image_sum = 0.0
    image_square_sum = 0.0
    pixel_count = 0
    targets = []
    for index in range(len(dataset)):
        image, target, _ = dataset.raw_item(index)
        values = image.astype(np.float64, copy=False)
        image_sum += float(values.sum(dtype=np.float64))
        image_square_sum += float(np.square(values).sum(dtype=np.float64))
        pixel_count += int(values.size)
        targets.append(target)
    image_mean = image_sum / pixel_count
    image_variance = max(image_square_sum / pixel_count - image_mean**2, 0.0)
    image_std = float(np.sqrt(image_variance))
    target_values = np.asarray(targets, dtype=np.float64)
    target_std = float(target_values.std(ddof=0))
    if image_std <= 0 or target_std <= 0:
        raise ValueError("Training data have zero image or target variance")
    manifest_path = dataset.release / "manifest.csv"
    return {
        "source_split": dataset.split,
        "sample_count": len(dataset),
        "pixel_count": pixel_count,
        "image_mean": float(image_mean),
        "image_std": image_std,
        "target_field": dataset.target_field,
        "target_mean": float(target_values.mean()),
        "target_std": target_std,
        "release_manifest": manifest_path.as_posix(),
        "release_manifest_sha256": file_sha256(manifest_path),
        "source_split_manifest": dataset.dataset.manifest_path.as_posix(),
        "source_split_manifest_sha256": file_sha256(dataset.dataset.manifest_path),
    }
