"""Mask-aware normalization transforms and their recorded inverse."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationResult:
    values: np.ndarray
    method: str
    offset: float
    scale: float
    clip_low: float
    clip_high: float
    status: str
    valid_pixels: int
    invertible: bool

    def inverse(self, values=None):
        """Invert values on valid pixels; clipped transforms are not exact."""
        if not self.invertible:
            raise ValueError(f"Normalization method is not invertible: {self.method}")
        values = self.values if values is None else np.asarray(values)
        return np.asarray(values, dtype=float) * self.scale + self.offset

    def to_dict(self):
        return {
            "method": self.method,
            "offset": self.offset,
            "scale": self.scale,
            "clip_low": self.clip_low,
            "clip_high": self.clip_high,
            "status": self.status,
            "valid_pixels": self.valid_pixels,
            "invertible": self.invertible,
        }


def _robust_scale(values):
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        return 1.4826 * mad
    return float(np.std(values))


def normalize_image(
    image,
    valid_mask=None,
    method="masked_percentile",
    lower=1.0,
    upper=99.0,
    invalid_fill=0.0,
):
    """Normalize a two-dimensional image without using invalid pixels."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("Normalization requires a two-dimensional image")
    if valid_mask is None:
        valid_mask = np.ones(image.shape, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != image.shape:
        raise ValueError("Normalization mask must match image shape")
    valid_mask &= np.isfinite(image)
    valid_pixels = int(valid_mask.sum())
    output = np.full(image.shape, invalid_fill, dtype=np.float32)
    if valid_pixels == 0:
        return NormalizationResult(
            output, method, float("nan"), float("nan"), float("nan"), float("nan"),
            "normalization_not_constrained", 0, method in {"none", "affine"},
        )

    valid_values = image[valid_mask]
    if method == "none":
        offset, scale = 0.0, 1.0
        output[valid_mask] = valid_values.astype(np.float32)
        return NormalizationResult(output, method, offset, scale, float(np.min(valid_values)), float(np.max(valid_values)), "pass", valid_pixels, True)

    if method == "affine":
        offset = float(np.median(valid_values))
        scale = _robust_scale(valid_values)
        status = "pass"
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
            status = "constant_valid_region"
        output[valid_mask] = ((valid_values - offset) / scale).astype(np.float32)
        return NormalizationResult(output, method, offset, scale, float(np.min(valid_values)), float(np.max(valid_values)), status, valid_pixels, True)

    if method != "masked_percentile":
        raise ValueError(f"Unknown normalization method: {method}")
    clip_low, clip_high = (float(value) for value in np.percentile(valid_values, [lower, upper]))
    scale = clip_high - clip_low
    if not np.isfinite(scale) or scale <= 0:
        output[valid_mask] = 0.0
        return NormalizationResult(output, method, clip_low, 1.0, clip_low, clip_high, "constant_valid_region", valid_pixels, False)
    output[valid_mask] = (np.clip(valid_values, clip_low, clip_high) - clip_low) / scale
    return NormalizationResult(output, method, clip_low, scale, clip_low, clip_high, "pass", valid_pixels, False)
