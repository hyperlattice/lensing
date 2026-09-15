"""Variance products with explicit provenance and unit assumptions."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VarianceResult:
    values: np.ndarray
    method: str
    status: str
    unit: str
    valid_pixels: int
    correlation_assumption: str

    def to_dict(self):
        return {
            "method": self.method,
            "status": self.status,
            "unit": self.unit,
            "valid_pixels": self.valid_pixels,
            "correlation_assumption": self.correlation_assumption,
        }


def _validate_mask(image, valid_mask):
    image = np.asarray(image)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if image.ndim != 2 or valid_mask.shape != image.shape:
        raise ValueError("Variance image and valid mask must be matching two-dimensional arrays")
    return image, valid_mask & np.isfinite(image)


def variance_from_error(error, valid_mask=None, unit="unknown", correlation_assumption="unknown"):
    """Convert a supplied per-pixel 1-sigma error array to variance."""
    error = np.asarray(error, dtype=float)
    if error.ndim != 2:
        raise ValueError("Error image must be two-dimensional")
    if valid_mask is None:
        valid_mask = np.ones(error.shape, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != error.shape:
        raise ValueError("Error image and valid mask must have matching shapes")
    valid = valid_mask & np.isfinite(error) & (error > 0)
    values = np.zeros(error.shape, dtype=np.float32)
    values[valid] = (error[valid] ** 2).astype(np.float32)
    status = "pass" if valid.any() else "variance_not_constrained"
    return VarianceResult(values, "supplied_error_squared", status, f"({unit})^2", int(valid.sum()), correlation_assumption)


def estimate_scalar_variance(
    image,
    valid_mask,
    background_scale,
    image_unit="unknown",
    noise_floor=1e-6,
    correlation_assumption="independent_pixels_approximation",
):
    """Create a positive scalar variance for an explicitly toy/no-unprofiled model."""
    image, valid_mask = _validate_mask(image, valid_mask)
    valid_pixels = int(valid_mask.sum())
    try:
        scale = float(background_scale)
    except (TypeError, ValueError):
        scale = float("nan")
    try:
        floor = float(noise_floor)
    except (TypeError, ValueError):
        raise ValueError("noise_floor must be numeric")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("noise_floor must be finite and positive")
    if valid_pixels == 0:
        return VarianceResult(
            np.zeros(image.shape, dtype=np.float32),
            "toy_scalar_background",
            "variance_not_constrained",
            f"({image_unit})^2",
            0,
            correlation_assumption,
        )
    variance_value = max(scale * scale if np.isfinite(scale) else 0.0, floor * floor)
    values = np.zeros(image.shape, dtype=np.float32)
    values[valid_mask] = np.float32(variance_value)
    return VarianceResult(
        values,
        "toy_scalar_background",
        "toy_unprofiled",
        f"({image_unit})^2",
        valid_pixels,
        correlation_assumption,
    )
