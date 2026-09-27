"""Observation rendering with an independently implemented SIS/SIE mapping."""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from typing import Any, Mapping

import numpy as np

from src.processing.background import estimate_scalar_background
from src.processing.geometry import (
    background_exclusion_mask,
    center_crop_or_pad_with_transform,
    coverage_mask,
    invalid_pixel_mask,
)
from src.processing.normalization import normalize_image
from src.processing.psf import convolve_same
from src.simulation.independent_sie import (
    independent_sie_deflection,
    independent_sie_magnification,
    solve_independent_sie_images,
)
from src.simulation.models import build_light_model
from src.simulation.observation import (
    _deposit_bilinear,
    observation_psf,
    realize_detector_observation,
)


def render_independent_expected_scene(
    parameters: Mapping[str, Any], *, light_model=None
) -> dict[str, Any]:
    """Ray-shoot a Sérsic source with the independent SIE equation.

    Lenstronomy evaluates the shared source and foreground-light profiles. Its
    lens equation, image solver, PSF, and image renderer are not used here.
    """
    shape = tuple(int(value) for value in parameters.get("shape", (256, 256)))
    scale = float(parameters.get("pixel_scale_arcsec", 0.05))
    supersampling = int(parameters.get("supersampling_factor", 2))
    if len(shape) != 2 or min(shape) <= 0 or scale <= 0 or supersampling <= 0:
        raise ValueError("Invalid image shape, pixel scale, or supersampling factor")
    if any(abs(float(parameters.get(key, 0.0))) > 0.0 for key in ("gamma1", "gamma2")):
        raise ValueError("Week 7 independent renderer is frozen for the no-shear SIE proposal")
    if light_model is None:
        light_model = build_light_model(["SERSIC_ELLIPSE"])

    height, width = shape
    high_height, high_width = height * supersampling, width * supersampling
    x_arcsec = (
        (np.arange(high_width, dtype=np.float64) + 0.5) / supersampling
        - 0.5
        - (width - 1) / 2.0
    ) * scale
    y_arcsec = (
        (np.arange(high_height, dtype=np.float64) + 0.5) / supersampling
        - 0.5
        - (height - 1) / 2.0
    ) * scale
    theta_x, theta_y = np.meshgrid(x_arcsec, y_arcsec)
    alpha_x, alpha_y = independent_sie_deflection(
        theta_x,
        theta_y,
        theta_E_arcsec=float(parameters["theta_E_arcsec"]),
        axis_ratio=float(parameters["q"]),
        phi_rad=float(parameters["phi_rad"]),
        center_x_arcsec=float(parameters.get("lens_center_x_arcsec", 0.0)),
        center_y_arcsec=float(parameters.get("lens_center_y_arcsec", 0.0)),
    )
    beta_x = theta_x - alpha_x
    beta_y = theta_y - alpha_y
    source_high = np.asarray(
        light_model.surface_brightness(beta_x, beta_y, [parameters["source_parameters"]]),
        dtype=np.float64,
    )
    lens_light_high = np.asarray(
        light_model.surface_brightness(theta_x, theta_y, [parameters["lens_light_parameters"]]),
        dtype=np.float64,
    )
    if source_high.shape != (high_height, high_width) or lens_light_high.shape != source_high.shape:
        raise ValueError("Shared Lenstronomy light model returned an unexpected image shape")
    # Lenstronomy's ImageModel returns flux integrated over each native pixel.
    # The independent path evaluates surface brightness at subpixel centers,
    # so convert arcsec^-2 brightness to native-pixel flux before averaging.
    clean_high = (source_high + lens_light_high) * (scale**2)
    clean_scene = clean_high.reshape(height, supersampling, width, supersampling).mean(axis=(1, 3))

    images = solve_independent_sie_images(
        source_x_arcsec=float(parameters["source_x_arcsec"]),
        source_y_arcsec=float(parameters["source_y_arcsec"]),
        theta_E_arcsec=float(parameters["theta_E_arcsec"]),
        axis_ratio=float(parameters["q"]),
        phi_rad=float(parameters["phi_rad"]),
        center_x_arcsec=float(parameters.get("lens_center_x_arcsec", 0.0)),
        center_y_arcsec=float(parameters.get("lens_center_y_arcsec", 0.0)),
    )
    if len(images) not in {2, 4}:
        raise ValueError(f"Independent SIS/SIE root solver found {len(images)} images; expected 2 or 4")
    positions_x = np.round(images[:, 0], 8)
    positions_y = np.round(images[:, 1], 8)
    point_amplitudes = []
    point_scene = np.zeros(shape, dtype=np.float64)
    for image_x, image_y in zip(positions_x, positions_y):
        magnification = independent_sie_magnification(
            float(image_x),
            float(image_y),
            theta_E_arcsec=float(parameters["theta_E_arcsec"]),
            axis_ratio=float(parameters["q"]),
            phi_rad=float(parameters["phi_rad"]),
            center_x_arcsec=float(parameters.get("lens_center_x_arcsec", 0.0)),
            center_y_arcsec=float(parameters.get("lens_center_y_arcsec", 0.0)),
        )
        amplitude = float(np.round(abs(magnification) * float(parameters["point_source_amp"]), 3))
        point_amplitudes.append(amplitude)
        _deposit_bilinear(
            point_scene,
            x_pixel=(width - 1) / 2.0 + float(image_x) / scale,
            y_pixel=(height - 1) / 2.0 + float(image_y) / scale,
            amplitude=amplitude,
        )

    psf_kernel = observation_psf(
        psf_fwhm_arcsec=float(parameters["psf_fwhm_arcsec"]),
        pixel_scale_arcsec=scale,
    )
    expected_scene = np.asarray(convolve_same(clean_scene + point_scene, psf_kernel), dtype=np.float32)
    clean_scene = np.asarray(clean_scene, dtype=np.float32)
    if not np.isfinite(clean_scene).all() or not np.isfinite(expected_scene).all() or np.any(expected_scene < -1e-8):
        raise ValueError("Independent renderer produced invalid scene pixels")
    return {
        "clean_scene": clean_scene,
        "expected_scene": expected_scene,
        "psf": psf_kernel,
        "point_positions_x_arcsec": positions_x.astype(float).tolist(),
        "point_positions_y_arcsec": positions_y.astype(float).tolist(),
        "point_amplitudes": point_amplitudes,
        "root_count": int(len(images)),
    }


def observe_independent_scene(
    expected_scene: np.ndarray,
    parameters: Mapping[str, Any],
    *,
    exposure_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Draw detector noise and apply the project's frozen preprocessing."""
    multiplier = float(exposure_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("exposure_multiplier must be positive and finite")
    exposure = float(parameters["exposure_time"]) * multiplier
    detector = realize_detector_observation(
        expected_scene,
        background_rate=float(parameters["background_rate"]),
        exposure_time=exposure,
        read_noise_electrons=float(parameters["read_noise_electrons"]),
        noise_seed=int(parameters["noise_seed"]),
    )
    raw = np.asarray(detector.observed_rate, dtype=np.float32)
    cleaned = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    target_shape = tuple(int(value) for value in parameters.get("shape", (256, 256)))
    standardized, transform = center_crop_or_pad_with_transform(cleaned, target_shape, pad_value=0.0)
    invalid = transform.apply(invalid_pixel_mask(raw), pad_value=True)
    coverage = coverage_mask(raw.shape, transform)
    background_mask = background_exclusion_mask(invalid, coverage)
    background = estimate_scalar_background(
        standardized,
        exclusion_mask=background_mask,
        sigma_clip=4.0,
        max_iterations=5,
        min_valid_pixels=64,
    )
    if background.status == "pass" and np.isfinite(background.value):
        science = standardized - background.value
    else:
        science = standardized.copy()
    valid_observation = ~invalid & ~coverage
    processed = normalize_image(
        science,
        valid_mask=valid_observation,
        method="none",
        lower=1.0,
        upper=99.0,
        invalid_fill=0.0,
    )
    return {
        "raw_observed_rate": raw,
        "observed_science": detector.science_rate,
        "variance": detector.variance_rate,
        "science": np.asarray(science, dtype=np.float32),
        "processed": processed.values,
        "background_value": float(background.value),
        "background_scale": float(background.scale),
        "background_status": str(background.status),
        "exposure_time": exposure,
        "noise_seed": int(parameters["noise_seed"]),
    }
