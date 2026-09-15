"""Robust scalar background estimation with explicit exclusion semantics."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BackgroundEstimate:
    value: float
    scale: float
    status: str
    method: str
    valid_pixels: int
    source_pixels: int
    exclusion_mask: np.ndarray

    def to_dict(self):
        return {
            "value": self.value,
            "scale": self.scale,
            "status": self.status,
            "method": self.method,
            "valid_pixels": self.valid_pixels,
            "source_pixels": self.source_pixels,
        }


def _robust_scale(values):
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        return 1.4826 * mad
    standard_deviation = float(np.std(values))
    return standard_deviation


def estimate_scalar_background(
    image,
    exclusion_mask=None,
    sigma_clip=4.0,
    max_iterations=5,
    min_valid_pixels=64,
):
    """Estimate a scalar background and return the pixels excluded as sources.

    ``True`` in ``exclusion_mask`` means that a pixel must not influence the
    estimate. Bright or faint outliers rejected by the robust clip are added
    to the returned exclusion mask as a conservative source/background mask.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("Background estimation requires a two-dimensional image")
    if exclusion_mask is None:
        exclusion_mask = np.zeros(image.shape, dtype=bool)
    exclusion_mask = np.asarray(exclusion_mask, dtype=bool)
    if exclusion_mask.shape != image.shape:
        raise ValueError("Background exclusion mask must match image shape")

    finite = np.isfinite(image)
    base_exclusion = exclusion_mask | ~finite
    candidate = finite & ~base_exclusion
    valid_pixels = int(candidate.sum())
    if valid_pixels < min_valid_pixels:
        return BackgroundEstimate(
            value=float("nan"),
            scale=float("nan"),
            status="background_not_constrained",
            method="robust_scalar_sigma_clip",
            valid_pixels=valid_pixels,
            source_pixels=0,
            exclusion_mask=base_exclusion,
        )

    keep = candidate.copy()
    for _ in range(max_iterations):
        values = image[keep]
        center = float(np.median(values))
        scale = _robust_scale(values)
        if not np.isfinite(scale) or scale == 0:
            break
        updated = candidate & (np.abs(image - center) <= sigma_clip * scale)
        if np.array_equal(updated, keep):
            break
        if int(updated.sum()) < min_valid_pixels:
            break
        keep = updated

    values = image[keep]
    center = float(np.median(values))
    scale = _robust_scale(values)
    source_mask = candidate & ~keep
    return BackgroundEstimate(
        value=center,
        scale=float(scale),
        status="pass",
        method="robust_scalar_sigma_clip",
        valid_pixels=int(keep.sum()),
        source_pixels=int(source_mask.sum()),
        exclusion_mask=base_exclusion | source_mask,
    )
