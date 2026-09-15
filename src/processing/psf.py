"""PSF kernel records, controls, validation, and small-image convolution."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PSFKernel:
    psf_id: str
    values: np.ndarray
    pixel_scale: float
    source: str
    unit: str = "normalized"

    @property
    def center_yx(self):
        return tuple((np.asarray(self.values.shape) - 1) / 2.0)

    def to_dict(self, values_path=None):
        record = {
            "psf_id": self.psf_id,
            "shape": list(self.values.shape),
            "pixel_scale_arcsec_per_pixel": self.pixel_scale,
            "center_yx_pixels": list(self.center_yx),
            "source": self.source,
            "unit": self.unit,
            "sum": float(np.sum(self.values)),
        }
        if values_path is not None:
            record["values_path"] = Path(values_path).as_posix()
        return record


def validate_psf(kernel, tolerance=1e-6):
    """Return validation errors for a PSF kernel record."""
    errors = []
    values = np.asarray(kernel.values)
    if values.ndim != 2:
        errors.append("PSF kernel must be two-dimensional")
    if values.shape[0] % 2 == 0 or values.shape[1] % 2 == 0:
        errors.append("PSF kernel dimensions must be odd")
    if not np.isfinite(values).all():
        errors.append("PSF kernel contains non-finite values")
    if (values < 0).any():
        errors.append("PSF kernel contains negative values")
    if not np.isfinite(kernel.pixel_scale) or kernel.pixel_scale <= 0:
        errors.append("PSF pixel scale must be finite and positive")
    if not np.isclose(np.sum(values), 1.0, atol=tolerance):
        errors.append("PSF kernel must have unit sum")
    return errors


def gaussian_psf(psf_id="gaussian-control", sigma_pixels=1.0, size=9, pixel_scale=0.05):
    """Create a centered, unit-sum Gaussian control kernel."""
    if not isinstance(size, int) or size <= 0 or size % 2 == 0:
        raise ValueError("Gaussian PSF size must be a positive odd integer")
    if not np.isfinite(sigma_pixels) or sigma_pixels <= 0:
        raise ValueError("Gaussian PSF sigma must be finite and positive")
    if not np.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError("Gaussian PSF pixel scale must be finite and positive")
    center = (size - 1) / 2.0
    y, x = np.indices((size, size), dtype=float)
    values = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2.0 * sigma_pixels**2))
    values /= np.sum(values)
    kernel = PSFKernel(psf_id, values.astype(np.float32), float(pixel_scale), "analytic_gaussian_control")
    errors = validate_psf(kernel)
    if errors:
        raise ValueError("Invalid Gaussian PSF: " + "; ".join(errors))
    return kernel


def save_psf(kernel, output_dir):
    """Save one shared kernel and its metadata record."""
    errors = validate_psf(kernel)
    if errors:
        raise ValueError("Invalid PSF: " + "; ".join(errors))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values_path = output_dir / f"{kernel.psf_id}.npy"
    metadata_path = output_dir / f"{kernel.psf_id}.json"
    np.save(values_path, kernel.values)
    metadata_path.write_text(json.dumps(kernel.to_dict(values_path), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return values_path, metadata_path


def convolve_same(image, kernel):
    """Convolve a 2D image with a centered kernel using zero boundaries."""
    image = np.asarray(image, dtype=float)
    values = np.asarray(kernel.values if isinstance(kernel, PSFKernel) else kernel, dtype=float)
    if image.ndim != 2 or values.ndim != 2:
        raise ValueError("Image and PSF must be two-dimensional")
    if values.shape[0] % 2 == 0 or values.shape[1] % 2 == 0:
        raise ValueError("PSF dimensions must be odd")
    pad_y, pad_x = values.shape[0] // 2, values.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, values.shape)
    return np.einsum("ij,xyij->xy", values, windows)
