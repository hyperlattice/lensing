"""Versioned parameter proposals and deterministic draws for simulation pilots."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.ml.artifacts import sha256 as digest


def _finite_range(
    config: Mapping[str, Any], key: str, *, lower: float | None = None, upper: float | None = None
) -> tuple[float, float]:
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


SOURCE = Path("data/simulations/sim-eng1000-v1")


def identity_ledger():
    columns = {
        "parent_system_id",
        "parameter_seed",
        "noise_seed",
        "source_template_id",
        "background_template_id",
    }
    ledger, hashes = [], {}
    for path in (
        SOURCE / "splits/test.csv",
        Path("data/simulations/reliability-blind-holdout-v1/manifest.csv"),
    ):
        # Never parse radius/physical labels or read any final array/sidecar.
        frame = pd.read_csv(path, usecols=lambda name: name in columns)
        hashes[path.as_posix()] = digest(path)
        for row in frame.to_dict("records"):
            assets = [
                str(row[k])
                for k in ("source_template_id", "background_template_id")
                if k in row and pd.notna(row[k])
            ]
            ledger.append(
                {
                    "parent_system_id": str(row["parent_system_id"]),
                    "parameter_seed": int(row["parameter_seed"]),
                    "noise_seed": int(row["noise_seed"]),
                    "ancestry_ids": assets,
                    "role": "final",
                }
            )
    refs = Path("data/simulations/renderer-qualification-v1-complete/references.json")
    numerical = json.loads(refs.read_text())
    hashes[refs.as_posix()] = digest(refs)
    done = set()
    for row in numerical["cases"]:
        if row["index"] in done:
            continue
        done.add(row["index"])
        p = row["parameters"]
        ledger.append(
            {
                "parent_system_id": f"numerical-v1-{row['index']}",
                "role": "numerical_development",
                "parameter_seed": p["parameter_seed"],
                "noise_seed": p["noise_seed"],
                "ancestry_ids": [],
            }
        )
    proposal = yaml.safe_load(Path("configs/simulation.yaml").read_text())
    for index in set(range(106)) - done:
        p = draw_observation_parameters(720101, index, proposal)
        ledger.append(
            {
                "parent_system_id": f"numerical-v1-{index}",
                "role": "numerical_development",
                "parameter_seed": p["parameter_seed"],
                "noise_seed": p["noise_seed"],
                "ancestry_ids": [],
            }
        )
    return ledger, hashes


def physical_fingerprint(p):
    keys = (
        "theta_E_arcsec",
        "q",
        "phi_rad",
        "lens_center_x_arcsec",
        "lens_center_y_arcsec",
        "source_x_arcsec",
        "source_y_arcsec",
        "source_q",
        "source_phi_rad",
        "source_n",
        "source_re_arcsec",
        "source_amp",
        "lens_light_q",
        "lens_light_phi_rad",
        "lens_light_n",
        "lens_light_re_arcsec",
        "lens_light_amp",
        "point_source_amp",
    )
    return hashlib.sha256(
        json.dumps({k: p[k] for k in keys}, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def source_proposals(role: str) -> tuple[list[tuple], dict[str, str]]:
    """Load fixed source identities and parameters without a historical T-arm release.

    C/G/R use canonical detector seeds or explicit fresh draws. Historical
    condition fields remain canonical compatibility metadata in new banks.
    """
    if role not in {"source_train", "source_development"}:
        raise ValueError("Expected a source training or development role")
    split = "train" if role == "source_train" else "validation"
    allowed = {
        "sample_id",
        "parent_system_id",
        "parameters_path",
        "source_template_id",
        "background_template_id",
    }
    path = SOURCE / "splits" / f"{split}.csv"
    frame = pd.read_csv(path, usecols=lambda name: name in allowed)
    records = []
    for row in frame.to_dict("records"):
        parameters = json.loads((SOURCE / row["parameters_path"]).read_text(encoding="utf-8"))
        canonical = {
            "pixel_scale_arcsec": 0.05,
            "exposure_multiplier": 1.0,
            "read_noise_multiplier": 1.0,
        }
        records.append((row, parameters, canonical, parameters["noise_seed"]))
    return records, {path.as_posix(): digest(path)}
