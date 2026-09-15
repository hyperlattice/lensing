"""Versioned parameter proposals and deterministic draws for simulation pilots."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def _finite_range(config: Mapping[str, Any], key: str, *, lower: float | None = None, upper: float | None = None) -> tuple[float, float]:
    values = config[key]
    if len(values) != 2:
        raise ValueError(f"{key} must contain exactly two values")
    first, second = (float(value) for value in values)
    if not np.isfinite([first, second]).all() or first >= second:
        raise ValueError(f"{key} must be a finite increasing range")
    if lower is not None and first < lower:
        raise ValueError(f"{key} lower bound must be at least {lower}")
    if upper is not None and second > upper:
        raise ValueError(f"{key} upper bound must be at most {upper}")
    return first, second


def validate_sie_parameters(parameters: Mapping[str, Any]) -> list[str]:
    """Return validation errors for one SIE plus extended-source proposal."""

    errors: list[str] = []
    positive = ("theta_E_arcsec", "source_re_arcsec", "source_n", "source_amp")
    for key in positive:
        try:
            if not np.isfinite(float(parameters[key])) or float(parameters[key]) <= 0:
                errors.append(f"{key} must be positive and finite")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{key} must be numeric")
    for key in ("q", "source_q"):
        try:
            if not 0 < float(parameters[key]) <= 1:
                errors.append(f"{key} must be in (0, 1]")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{key} must be numeric")
    for key in (
        "phi_rad",
        "source_phi_rad",
        "lens_center_x_arcsec",
        "lens_center_y_arcsec",
        "source_x_arcsec",
        "source_y_arcsec",
    ):
        try:
            if not np.isfinite(float(parameters[key])):
                errors.append(f"{key} must be finite")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{key} must be numeric")
    if "gamma1" in parameters or "gamma2" in parameters:
        for key in ("gamma1", "gamma2"):
            try:
                if not np.isfinite(float(parameters[key])):
                    errors.append(f"{key} must be finite")
            except (KeyError, TypeError, ValueError):
                errors.append(f"{key} must be numeric")
    return errors


def draw_sie_parameters(seed: int, index: int, config: Mapping[str, Any]) -> dict[str, Any]:
    """Draw one reproducible SIE/extended-source parameter set."""

    proposal = config["pilot"]["sie_extended"]
    parameter_seed, noise_seed = (
        int(value)
        for value in np.random.SeedSequence([int(seed), 3003, int(index)]).generate_state(2)
    )
    rng = np.random.default_rng(parameter_seed)
    theta_min, theta_max = _finite_range(proposal, "theta_E_arcsec", lower=0)
    q_min, q_max = _finite_range(proposal, "q", lower=0, upper=1)
    center_min, center_max = _finite_range(proposal, "lens_center_offset_arcsec")
    source_min, source_max = _finite_range(proposal, "source_position_arcsec")
    source_q_min, source_q_max = _finite_range(proposal, "source_q", lower=0, upper=1)
    source_n_min, source_n_max = _finite_range(proposal, "source_n", lower=0)
    source_re_min, source_re_max = _finite_range(proposal, "source_re_arcsec", lower=0)
    return {
        "parameter_seed": parameter_seed,
        "noise_seed": noise_seed,
        "mass_model": "SIE",
        "theta_E_arcsec": float(rng.uniform(theta_min, theta_max)),
        "q": float(rng.uniform(q_min, q_max)),
        "phi_rad": float(rng.uniform(0.0, math.pi)),
        "lens_center_x_arcsec": float(rng.uniform(center_min, center_max)),
        "lens_center_y_arcsec": float(rng.uniform(center_min, center_max)),
        "source_x_arcsec": float(rng.uniform(source_min, source_max)),
        "source_y_arcsec": float(rng.uniform(source_min, source_max)),
        "source_q": float(rng.uniform(source_q_min, source_q_max)),
        "source_phi_rad": float(rng.uniform(0.0, math.pi)),
        "source_n": float(rng.uniform(source_n_min, source_n_max)),
        "source_re_arcsec": float(rng.uniform(source_re_min, source_re_max)),
        "source_amp": float(rng.uniform(*_finite_range(proposal, "source_amp", lower=0))),
        "shape": list(config["grid"]["shape"]),
        "pixel_scale_arcsec": float(config["grid"]["pixel_scale_arcsec"]),
        "render_mode": "lenstronomy_extended_source_noiseless_psf_none",
    }


def draw_observation_parameters(seed: int, index: int, config: Mapping[str, Any]) -> dict[str, Any]:
    """Draw SIE/extended-source parameters plus P60 observation components."""

    parameters = draw_sie_parameters(seed, index, config)
    proposal = config["pilot"]["observation"]
    lens_light_q_min, lens_light_q_max = _finite_range(proposal, "lens_light_q", lower=0, upper=1)
    lens_light_n_min, lens_light_n_max = _finite_range(proposal, "lens_light_n", lower=0)
    lens_light_re_min, lens_light_re_max = _finite_range(proposal, "lens_light_re_arcsec", lower=0)
    lens_light_amp_min, lens_light_amp_max = _finite_range(proposal, "lens_light_amp", lower=0)
    point_amp_min, point_amp_max = _finite_range(proposal, "point_source_amp", lower=0)
    rng = np.random.default_rng(parameters["parameter_seed"] + 17)
    effects = config["pilot"]["observation_effects"]
    psf_min, psf_max = _finite_range(effects, "psf_fwhm_arcsec", lower=0)
    background_min, background_max = _finite_range(effects, "background_rate", lower=0)
    read_min, read_max = _finite_range(effects, "read_noise_electrons", lower=0)
    exposure_min, exposure_max = _finite_range(effects, "exposure_time", lower=0)
    parameters.update(
        {
            "lens_light_q": float(rng.uniform(lens_light_q_min, lens_light_q_max)),
            "lens_light_phi_rad": float(rng.uniform(0.0, math.pi)),
            "lens_light_n": float(rng.uniform(lens_light_n_min, lens_light_n_max)),
            "lens_light_re_arcsec": float(rng.uniform(lens_light_re_min, lens_light_re_max)),
            "lens_light_amp": float(rng.uniform(lens_light_amp_min, lens_light_amp_max)),
            "point_source_amp": float(rng.uniform(point_amp_min, point_amp_max)),
            "psf_fwhm_arcsec": float(rng.uniform(psf_min, psf_max)),
            "background_rate": float(rng.uniform(background_min, background_max)),
            "read_noise_electrons": float(rng.uniform(read_min, read_max)),
            "exposure_time": float(rng.uniform(exposure_min, exposure_max)),
            "render_mode": "lenstronomy_sie_extended_point_source_psf",
        }
    )
    return parameters


def draw_stress_parameters(seed: int, index: int, config: Mapping[str, Any]) -> dict[str, Any]:
    """Draw P100 SIE-plus-shear parameters with broader observation stress ranges."""

    parameters = draw_observation_parameters(seed, index, config)
    proposal = config["pilot"]["stress"]
    rng = np.random.default_rng(parameters["parameter_seed"] + 7001)
    shear_min, shear_max = _finite_range(proposal, "shear_amplitude", lower=0)
    quad_source_min, quad_source_max = _finite_range(proposal, "quad_source_position_arcsec")
    quad_q_min, quad_q_max = _finite_range(proposal, "quad_q", lower=0, upper=1)
    quad_shear_min, quad_shear_max = _finite_range(proposal, "quad_shear_amplitude", lower=0)
    lens_q_min, lens_q_max = _finite_range(proposal, "lens_light_q", lower=0, upper=1)
    lens_n_min, lens_n_max = _finite_range(proposal, "lens_light_n", lower=0)
    lens_re_min, lens_re_max = _finite_range(proposal, "lens_light_re_arcsec", lower=0)
    lens_amp_min, lens_amp_max = _finite_range(proposal, "lens_light_amp", lower=0)
    point_amp_min, point_amp_max = _finite_range(proposal, "point_source_amp", lower=0)
    source_amp_min, source_amp_max = _finite_range(proposal, "source_amp", lower=0)
    psf_min, psf_max = _finite_range(proposal, "psf_fwhm_arcsec", lower=0)
    background_min, background_max = _finite_range(proposal, "background_rate", lower=0)
    read_min, read_max = _finite_range(proposal, "read_noise_electrons", lower=0)
    exposure_min, exposure_max = _finite_range(proposal, "exposure_time", lower=0)
    supersampling_min, supersampling_max = _finite_range(proposal, "supersampling_factor", lower=1)
    quad_candidate = int(index) % 4 in (0, 1)
    if quad_candidate:
        shear_amplitude = float(rng.uniform(quad_shear_min, quad_shear_max))
    else:
        shear_amplitude = float(rng.uniform(shear_min, shear_max))
    shear_phi = float(rng.uniform(0.0, math.pi))
    parameters.update(
        {
            "mass_model": "SIE+SHEAR",
            "gamma_amplitude": shear_amplitude,
            "gamma_phi_rad": shear_phi,
            "gamma1": shear_amplitude * math.cos(2.0 * shear_phi),
            "gamma2": shear_amplitude * math.sin(2.0 * shear_phi),
            "stress_case": "quad_candidate" if quad_candidate else "double_candidate",
            "lens_light_q": float(rng.uniform(lens_q_min, lens_q_max)),
            "lens_light_n": float(rng.uniform(lens_n_min, lens_n_max)),
            "lens_light_re_arcsec": float(rng.uniform(lens_re_min, lens_re_max)),
            "lens_light_amp": float(rng.uniform(lens_amp_min, lens_amp_max)),
            "point_source_amp": float(rng.uniform(point_amp_min, point_amp_max)),
            "source_amp": float(rng.uniform(source_amp_min, source_amp_max)),
            "psf_fwhm_arcsec": float(rng.uniform(psf_min, psf_max)),
            "background_rate": float(rng.uniform(background_min, background_max)),
            "read_noise_electrons": float(rng.uniform(read_min, read_max)),
            "exposure_time": float(rng.uniform(exposure_min, exposure_max)),
            "supersampling_factor": int(rng.integers(int(supersampling_min), int(supersampling_max))),
            "render_mode": "lenstronomy_sie_shear_extended_point_source_psf",
            "stress_profile": "sie_shear_observation_v1",
        }
    )
    if quad_candidate:
        parameters["source_x_arcsec"] = float(rng.uniform(quad_source_min, quad_source_max))
        parameters["source_y_arcsec"] = float(rng.uniform(quad_source_min, quad_source_max))
        parameters["q"] = float(rng.uniform(quad_q_min, quad_q_max))
    return parameters
