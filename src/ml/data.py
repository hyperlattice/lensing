"""Native-shape grouped CNN forwards and source-only validation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch


def image_features_and_prediction(
    model: torch.nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose the existing image-only CNN pooled features without changing weights.

    This is the declared initial alignment site; it is not claimed to reproduce
    an author's architecture. The caller controls train/eval mode and dropout.
    """
    if model.metadata_features != 0 or images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("Adaptation expects an image-only single-band CNN")
    features = model.pool(model.features(images)).flatten(start_dim=1)
    return features, model.regressor(features).squeeze(-1)


def grouped_forward(
    model: torch.nn.Module, images: list[np.ndarray], statistics: dict[str, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain native geometries, one full optimizer batch via grouped forwards."""
    groups = defaultdict(list)
    for image_index, image in enumerate(images):
        groups[image.shape].append(image_index)
    features, predictions = [None] * len(images), [None] * len(images)
    for indices in groups.values():
        arrays = np.stack([images[image_index] for image_index in indices])
        arrays = (arrays - statistics["image_mean"]) / statistics["image_std"]
        image_tensor = torch.from_numpy(arrays[:, None].astype(np.float32))
        batch_features, batch_predictions = image_features_and_prediction(model, image_tensor)
        for image_index, output_index in enumerate(indices):
            features[output_index], predictions[output_index] = (
                batch_features[image_index],
                batch_predictions[image_index],
            )
    return torch.stack(features), torch.stack(predictions)


@torch.inference_mode()
def source_validation(
    model: torch.nn.Module, bank: Any, statistics: dict[str, float], *, batch_size: int
) -> dict[str, Any]:
    model.eval()
    parent_ids, predicted = list(bank.by_id), []
    for start in range(0, len(parent_ids), batch_size):
        _, normalized_predictions = grouped_forward(
            model,
            [bank.source_image(p) for p in parent_ids[start : start + batch_size]],
            statistics,
        )
        predicted.extend(
            (
                normalized_predictions.numpy() * statistics["target_std"]
                + statistics["target_mean"]
            ).tolist()
        )
    true_radii = np.asarray([bank.targets[p] for p in parent_ids])
    return {
        "mae_arcsec": float(np.mean(abs(np.asarray(predicted) - true_radii))),
        "parent_ids": parent_ids,
        "prediction_arcsec": predicted,
    }
