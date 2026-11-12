"""Native observation conditions, PSFs and deterministic detector sampling."""

from __future__ import annotations

from typing import Any

import numpy as np

CELLS = (
    "pixel-scale-004",
    "pixel-scale-006",
    "exposure-half",
    "read-noise-double",
    "gaussian-fwhm-075",
    "gaussian-fwhm-125",
    "fresh-noise",
)


def condition_for(
    cell: str, *, shape: tuple[int, int] = (256, 256), pixel_scale_arcsec: float = 0.05
) -> dict[str, Any]:
    if cell not in CELLS or len(shape) != 2 or any(int(n) != n or n <= 0 for n in shape):
        raise ValueError("Invalid transformation or canonical shape")
    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
        raise ValueError("Canonical scale must be positive and finite")
    scale = {"pixel-scale-004": 0.04, "pixel-scale-006": 0.06}.get(cell, pixel_scale_arcsec)
    native = [max(1, int(np.floor(n * pixel_scale_arcsec / scale + 0.5))) for n in shape]
    return {
        "name": cell,
        "shape": native,
        "pixel_scale_arcsec": float(scale),
        "psf_family": "gaussian",
        "psf_fwhm_multiplier": {"gaussian-fwhm-075": 0.75, "gaussian-fwhm-125": 1.25}.get(
            cell, 1.0
        ),
        "exposure_multiplier": 0.5 if cell == "exposure-half" else 1.0,
        "read_noise_multiplier": 2.0 if cell == "read-noise-double" else 1.0,
    }


CONDITIONS = (
    "canonical",
    "pixel-scale-004",
    "pixel-scale-006",
    "exposure-half",
    "read-noise-double",
    "moffat-2.5",
    "moffat-3.5",
    "moffat-4.5",
    "moffat-3.5-exposure-half",
)
RENDERERS = ("lenstronomy", "autolens")
PRIMARY_CONDITIONS = ("moffat-2.5", "moffat-3.5", "moffat-4.5")


def evaluation_condition(name: str) -> dict[str, Any]:
    if name not in CONDITIONS:
        raise ValueError("Unknown frozen evaluation condition")
    if name == "canonical":
        return {**condition_for("fresh-noise"), "name": name}
    if name.startswith("moffat-"):
        return {
            **condition_for("fresh-noise"),
            "name": name,
            "psf_family": "moffat",
            "moffat_beta": float(name.split("-")[1]),
            "exposure_multiplier": 0.5 if name.endswith("exposure-half") else 1.0,
        }
    return condition_for(name)


def sample_detector(
    expected: np.ndarray, exposure: float, read: float, background: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a rate image and its analytic Poisson-plus-read-noise variance."""
    expected = np.asarray(expected, dtype=float)
    if (
        not np.isfinite([exposure, read, background]).all()
        or exposure <= 0
        or min(read, background) < 0
        or not np.isfinite(expected).all()
        or np.any(expected < 0)
    ):
        raise ValueError("Invalid detector observation")
    mean_counts = (expected + background) * exposure
    rng = np.random.default_rng(int(seed))
    raw = (rng.poisson(mean_counts) + rng.normal(0, read, expected.shape)) / exposure
    return raw, (mean_counts + read**2) / exposure**2


def psf_kernel(
    fwhm_arcsec: float, pixel_scale: float, family: str = "gaussian", beta: float = 3.5
) -> np.ndarray:
    """Return a unit-sum finite kernel; Moffat is an explicit new PSF family."""
    if not np.isfinite([fwhm_arcsec, pixel_scale]).all() or min(fwhm_arcsec, pixel_scale) <= 0:
        raise ValueError("PSF width and pixel scale must be positive")
    fwhm_pixels = float(fwhm_arcsec) / float(pixel_scale)
    if not np.isfinite(fwhm_pixels) or fwhm_pixels <= 0:
        raise ValueError("PSF width and pixel scale must be positive")
    size = max(
        5, int(np.ceil((10 / np.sqrt(8 * np.log(2)) if family == "gaussian" else 5) * fwhm_pixels))
    )
    size += 1 - size % 2
    yy, xx = np.indices((size, size), dtype=float) - (size - 1) / 2
    rr = xx**2 + yy**2
    if family == "gaussian":
        sigma = fwhm_pixels / np.sqrt(8 * np.log(2))
        kernel = np.exp(-rr / (2 * sigma**2))
    elif family == "moffat":
        if not np.isfinite(beta) or beta <= 1:
            raise ValueError("Moffat beta must exceed one")
        alpha = fwhm_pixels / (2 * np.sqrt(2 ** (1 / beta) - 1))
        kernel = (1 + rr / alpha**2) ** (-beta)
    else:
        raise ValueError("Unsupported PSF family")
    return kernel / kernel.sum()
