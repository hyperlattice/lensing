"""Generate the deterministic P10 analytic SIS control release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from src.layout import DEFAULT_PREVIEW_ROOT, preview_directory
from src.processing.preprocess import preprocess_image
from src.processing.preview_dataset import create_previews
from src.simulation.models import backend_status
from src.simulation.observation import observation_psf, realize_detector_observation, render_observation_scene
from src.simulation.priors import (
    draw_observation_parameters,
    draw_sie_parameters,
    draw_stress_parameters,
    validate_sie_parameters,
)
from src.simulation.render import render_from_parameters, render_sie_extended_source


GENERATOR_VERSION = "0.1.0"
DEFAULT_CONFIG = Path("configs/simulation/pilot.yaml")


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _create_simulation_previews(raw_root: Path, simulation_id: str, preview_root) -> tuple[Any, Path]:
    """Write visual QC artifacts outside the self-contained simulation release."""
    output = preview_directory("simulated", simulation_id, root=preview_root)
    index = create_previews(
        root=raw_root,
        output=output,
        manifest_path=None,
        dpi=100,
        contact_sheet=True,
    )
    return index, output


def _seed_pair(seed: int, index: int) -> tuple[int, int]:
    state = np.random.SeedSequence([int(seed), int(index)]).generate_state(2)
    return int(state[0]), int(state[1])


def _draw_parameters(seed: int, index: int, config: dict[str, Any]) -> dict[str, Any]:
    parameter_seed, noise_seed = _seed_pair(seed, index)
    rng = np.random.default_rng(parameter_seed)
    control = config["pilot"]["sis_control"]
    theta_min, theta_max = control["theta_E_arcsec"]
    beta_min, beta_max = control["source_radius_arcsec"]
    theta_E = float(rng.uniform(theta_min, theta_max))
    beta_radius = float(rng.uniform(beta_min, min(beta_max, 0.45 * theta_E)))
    beta_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    return {
        "parameter_seed": parameter_seed,
        "noise_seed": noise_seed,
        "mass_model": "SIS",
        "theta_E_arcsec": theta_E,
        "source_radius_arcsec": beta_radius,
        "source_x_arcsec": float(beta_radius * np.cos(beta_angle)),
        "source_y_arcsec": float(beta_radius * np.sin(beta_angle)),
        "source_angle_rad": beta_angle,
        "source_flux": float(control.get("source_flux", 1.0)),
        "shape": list(config["grid"]["shape"]),
        "pixel_scale_arcsec": float(config["grid"]["pixel_scale_arcsec"]),
        "render_mode": "analytic_point_source_bilinear_noiseless",
    }


def _write_raw_fits(
    path: Path,
    image: np.ndarray,
    parameters: dict[str, Any],
    sample_id: str,
    parent_id: str,
    *,
    simulation_id: str = "sim-p10-v1",
    model: str = "SIS",
    simulation_type: str = "ANALYTIC_SIS_CONTROL",
    image_unit: str = "RELATIVE_FLUX",
    overwrite: bool = False,
) -> None:
    header = fits.Header()
    header["SIMID"] = simulation_id
    header["PARENTID"] = parent_id
    header["SAMPLEID"] = sample_id
    header["MODEL"] = model
    header["THETAE"] = (float(parameters["theta_E_arcsec"]), "Einstein radius [arcsec]")
    if "source_x_arcsec" in parameters:
        header["BETAX"] = (float(parameters["source_x_arcsec"]), "Source x [arcsec]")
        header["BETAY"] = (float(parameters["source_y_arcsec"]), "Source y [arcsec]")
    if "q" in parameters:
        header["AXISRAT"] = (float(parameters["q"]), "Lens minor/major axis ratio")
        header["POSANG"] = (float(parameters["phi_rad"]), "Lens angle [radian]")
    header["SIMTYPE"] = simulation_type
    header["PIXSCALE"] = (float(parameters["pixel_scale_arcsec"]), "Arcsec per pixel")
    header["BUNIT"] = image_unit
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.asarray(image, dtype=np.float32), header=header).writeto(path, overwrite=overwrite)


def _manifest_row(
    *,
    sample_id: str,
    parent_id: str,
    simulation_id: str,
    parameters: dict[str, Any],
    render: Any,
    output_root: Path,
    paths: dict[str, Path],
    preprocess: dict[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    variance = preprocess["variance"]
    background = preprocess["background"]
    normalization = preprocess["normalization"]
    return {
        "sample_id": sample_id,
        "lens_id": parent_id,
        "domain": "simulated",
        "schema_version": "2.0",
        "dataset_version": simulation_id,
        "simulation_id": simulation_id,
        "parent_system_id": parent_id,
        "split_group_id": parent_id,
        "source_template_id": f"analytic-point:{parent_id}",
        "background_template_id": f"analytic-zero:{parent_id}",
        "raw_path": _relative(paths["raw"], output_root),
        "science_path": _relative(paths["science"], output_root),
        "processed_path": _relative(paths["normalized"], output_root),
        "mask_path": _relative(paths["invalid_mask"], output_root),
        "coverage_mask_path": _relative(paths["coverage_mask"], output_root),
        "background_mask_path": _relative(paths["background_mask"], output_root),
        "variance_path": _relative(paths["variance"], output_root),
        "geometry_transform_path": _relative(paths["geometry_transform"], output_root),
        "render_geometry_path": _relative(paths["render_geometry"], output_root),
        "background_path": _relative(paths["background"], output_root),
        "normalization_path": _relative(paths["normalization"], output_root),
        "noise_model_path": _relative(paths["noise_model"], output_root),
        "image_positions_path": _relative(paths["diagnostics"], output_root),
        "parameters_path": _relative(paths["parameters"], output_root),
        "image_unit": "relative_flux",
        "calibration_status": "pass",
        "original_shape": "x".join(str(value) for value in render.image.shape),
        "processed_shape": "x".join(str(value) for value in preprocess["normalized"].shape),
        "pixel_scale": parameters["pixel_scale_arcsec"],
        "pixel_scale_x": parameters["pixel_scale_arcsec"],
        "pixel_scale_y": parameters["pixel_scale_arcsec"],
        "processing_config_id": f"simulation-preprocessing-{config_hash[:12]}",
        "processing_version": "2.0.0",
        "background_status": background.status,
        "background_method": background.method,
        "normalization_method": normalization.method,
        "variance_method": "noiseless_control_floor",
        "variance_status": variance.status,
        "variance_unit": variance.unit,
        "noise_correlation_assumption": "not_applicable_noiseless_control",
        "mask_semantics": "true means excluded",
        "masked_fraction": float(preprocess["background_mask"].mean()),
        "source_type": "point",
        "mass_target_available": False,
        "theta_E_target_available": True,
        "structural_status": "pass",
        "observation_status": "pass",
        "science_status": "pass",
        "eligible_for": "simulation;classical_fit;ml",
        "exclusion_reason": "",
        "mass_model": "SIS",
        "theta_E_arcsec": parameters["theta_E_arcsec"],
        "q": 1.0,
        "phi_rad": 0.0,
        "source_x_arcsec": parameters["source_x_arcsec"],
        "source_y_arcsec": parameters["source_y_arcsec"],
        "source_flux": parameters["source_flux"],
        "image_count_physical": len(render.solutions),
        "image_count_detectable": len(render.solutions),
        "lens_equation_max_residual_arcsec": max(solution.residual_norm_arcsec for solution in render.solutions),
        "simulator_name": "lensing-analytic-sis-control",
        "simulator_version": GENERATOR_VERSION,
        "parameter_seed": parameters["parameter_seed"],
        "noise_seed": parameters["noise_seed"],
        "config_hash": config_hash,
    }


def generate_sis_controls(
    output_dir: str | Path,
    *,
    count: int = 10,
    seed: int = 1001,
    simulation_id: str = "sim-p10-v1",
    config_path: str | Path = DEFAULT_CONFIG,
    overwrite: bool = False,
    create_preview: bool = True,
    preview_root: str | Path = DEFAULT_PREVIEW_ROOT,
) -> dict[str, Any]:
    """Generate P10 into a self-contained artifact directory."""

    output_root = Path(output_dir)
    config_path = Path(config_path)
    if count <= 0:
        raise ValueError("count must be positive")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty and overwrite is disabled: {output_root}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = _config_hash(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    directories = {
        name: output_root / name
        for name in (
            "raw",
            "science",
            "normalized",
            "masks",
            "variance",
            "geometry",
            "background",
            "transforms",
            "noise",
            "parameters",
            "diagnostics",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for index in range(count):
        sample_id = f"{simulation_id}-{index + 1:04d}"
        parent_id = f"{simulation_id}-parent-{index + 1:04d}"
        parameters = _draw_parameters(seed, index, config)
        paths = {
            "raw": directories["raw"] / f"{sample_id}.fits",
            "science": directories["science"] / f"{sample_id}.npy",
            "normalized": directories["normalized"] / f"{sample_id}.npy",
            "invalid_mask": directories["masks"] / f"{sample_id}_invalid.npy",
            "coverage_mask": directories["masks"] / f"{sample_id}_coverage.npy",
            "background_mask": directories["masks"] / f"{sample_id}_background.npy",
            "variance": directories["variance"] / f"{sample_id}.npy",
            "geometry_transform": directories["geometry"] / f"{sample_id}.json",
            "render_geometry": directories["geometry"] / f"{sample_id}_render.json",
            "background": directories["background"] / f"{sample_id}.json",
            "normalization": directories["transforms"] / f"{sample_id}.json",
            "noise_model": directories["noise"] / f"{sample_id}.json",
            "parameters": directories["parameters"] / f"{sample_id}.json",
            "diagnostics": directories["diagnostics"] / f"{sample_id}.json",
        }
        try:
            render = render_from_parameters(parameters)
            _write_raw_fits(paths["raw"], render.image, parameters, sample_id, parent_id, overwrite=overwrite)
            processed = preprocess_image(
                paths["raw"],
                target_shape=tuple(config["grid"]["shape"]),
                normalization_method=config["preprocessing"]["normalization_method"],
                lower=config["preprocessing"]["lower_percentile"],
                upper=config["preprocessing"]["upper_percentile"],
                background_sigma_clip=config["preprocessing"]["background_sigma_clip"],
                background_max_iterations=config["preprocessing"]["background_max_iterations"],
                background_min_valid_pixels=config["preprocessing"]["background_min_valid_pixels"],
                image_unit="relative_flux",
                noise_floor=config["preprocessing"]["noise_floor"],
                noise_correlation_assumption="not_applicable_noiseless_control",
            )
            np.save(paths["science"], processed["science"])
            np.save(paths["normalized"], processed["normalized"])
            np.save(paths["invalid_mask"], processed["invalid_mask"])
            np.save(paths["coverage_mask"], processed["coverage_mask"])
            np.save(paths["background_mask"], processed["background_mask"])
            np.save(paths["variance"], processed["variance"].values)
            _write_json(paths["geometry_transform"], processed["geometry_transform"].to_dict())
            _write_json(paths["render_geometry"], render.geometry())
            _write_json(paths["background"], processed["background"].to_dict())
            _write_json(paths["normalization"], processed["normalization"].to_dict())
            _write_json(paths["noise_model"], processed["variance"].to_dict())
            diagnostics = {
                "sample_id": sample_id,
                "parent_system_id": parent_id,
                "source_position_arcsec": [parameters["source_x_arcsec"], parameters["source_y_arcsec"]],
                "solutions": [solution.to_dict() for solution in render.solutions],
                "max_residual_arcsec": max(solution.residual_norm_arcsec for solution in render.solutions),
                "rendered_flux": float(render.image.sum()),
                "expected_lensed_flux": float(
                    parameters["source_flux"]
                    * sum(solution.magnification_abs for solution in render.solutions)
                ),
            }
            _write_json(paths["diagnostics"], diagnostics)
            _write_json(
                paths["parameters"],
                {
                    **parameters,
                    "sample_id": sample_id,
                    "parent_system_id": parent_id,
                    "simulation_id": simulation_id,
                    "simulator_name": "lensing-analytic-sis-control",
                    "simulator_version": GENERATOR_VERSION,
                    "config_path": config_path.as_posix(),
                    "config_hash": config_hash,
                    "backend_reference": backend_status().to_dict(),
                    "solutions": [solution.to_dict() for solution in render.solutions],
                },
            )
            rows.append(
                _manifest_row(
                    sample_id=sample_id,
                    parent_id=parent_id,
                    simulation_id=simulation_id,
                    parameters=parameters,
                    render=render,
                    output_root=output_root,
                    paths=paths,
                    preprocess=processed,
                    config_hash=config_hash,
                )
            )
        except Exception as exc:
            rejections.append(
                {
                    "sample_id": sample_id,
                    "parent_system_id": parent_id,
                    "stage": "render_or_write",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    manifest_path = output_root / "manifest.csv"
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    rejection_path = output_root / "rejection_log.csv"
    with rejection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "parent_system_id", "stage", "error_type", "message"],
        )
        writer.writeheader()
        writer.writerows(rejections)

    preview_index = None
    preview_output = None
    if create_preview and rows:
        preview_index, preview_output = _create_simulation_previews(
            directories["raw"], simulation_id, preview_root
        )

    run_manifest = {
        "run_id": simulation_id,
        "generator_version": GENERATOR_VERSION,
        "simulation_id": simulation_id,
        "config_path": config_path.as_posix(),
        "config_hash": config_hash,
        "seed": int(seed),
        "count_requested": int(count),
        "count_attempted": int(count),
        "count_accepted": len(rows),
        "count_rejected": len(rejections),
        "output_root": output_root.as_posix(),
        "manifest_path": _relative(manifest_path, output_root),
        "rejection_log_path": _relative(rejection_path, output_root),
        "preview_index_path": None if preview_output is None else (preview_output / "preview_index.csv").as_posix(),
        "backend_reference": backend_status().to_dict(),
        "noise_policy": "noiseless image; positive numerical variance floor only",
        "rejections": rejections,
    }
    run_manifest_path = output_root / "run_manifest.json"
    _write_json(run_manifest_path, run_manifest)
    return run_manifest


def _sie_manifest_row(
    *,
    sample_id: str,
    parent_id: str,
    simulation_id: str,
    parameters: dict[str, Any],
    render: Any,
    output_root: Path,
    paths: dict[str, Path],
    preprocess: dict[str, Any],
    config_hash: str,
    simulator_version: str,
) -> dict[str, Any]:
    variance = preprocess["variance"]
    background = preprocess["background"]
    normalization = preprocess["normalization"]
    return {
        "sample_id": sample_id,
        "lens_id": parent_id,
        "domain": "simulated",
        "schema_version": "2.0",
        "dataset_version": simulation_id,
        "simulation_id": simulation_id,
        "parent_system_id": parent_id,
        "split_group_id": parent_id,
        "source_template_id": f"analytic-sersic:{parent_id}",
        "background_template_id": f"analytic-zero:{parent_id}",
        "raw_path": _relative(paths["raw"], output_root),
        "science_path": _relative(paths["science"], output_root),
        "processed_path": _relative(paths["normalized"], output_root),
        "mask_path": _relative(paths["invalid_mask"], output_root),
        "coverage_mask_path": _relative(paths["coverage_mask"], output_root),
        "background_mask_path": _relative(paths["background_mask"], output_root),
        "variance_path": _relative(paths["variance"], output_root),
        "geometry_transform_path": _relative(paths["geometry_transform"], output_root),
        "render_geometry_path": _relative(paths["render_geometry"], output_root),
        "background_path": _relative(paths["background"], output_root),
        "normalization_path": _relative(paths["normalization"], output_root),
        "noise_model_path": _relative(paths["noise_model"], output_root),
        "image_positions_path": _relative(paths["diagnostics"], output_root),
        "parameters_path": _relative(paths["parameters"], output_root),
        "image_unit": "relative_flux",
        "calibration_status": "pass",
        "original_shape": "x".join(str(value) for value in render.image.shape),
        "processed_shape": "x".join(str(value) for value in preprocess["normalized"].shape),
        "pixel_scale": parameters["pixel_scale_arcsec"],
        "pixel_scale_x": parameters["pixel_scale_arcsec"],
        "pixel_scale_y": parameters["pixel_scale_arcsec"],
        "processing_config_id": f"simulation-preprocessing-{config_hash[:12]}",
        "processing_version": "2.0.0",
        "background_status": background.status,
        "background_method": background.method,
        "normalization_method": normalization.method,
        "variance_method": "noiseless_control_floor",
        "variance_status": variance.status,
        "variance_unit": variance.unit,
        "noise_correlation_assumption": "not_applicable_noiseless_control",
        "mask_semantics": "true means excluded",
        "masked_fraction": float(preprocess["background_mask"].mean()),
        "source_type": "sersic",
        "mass_target_available": False,
        "theta_E_target_available": True,
        "structural_status": "pass",
        "observation_status": "pass",
        "science_status": "pass",
        "eligible_for": "simulation;classical_fit;ml",
        "exclusion_reason": "",
        "mass_model": "SIE",
        "theta_E_arcsec": parameters["theta_E_arcsec"],
        "q": parameters["q"],
        "phi_rad": parameters["phi_rad"],
        "e1": render.lens_parameters["e1"],
        "e2": render.lens_parameters["e2"],
        "lens_center_x_arcsec": parameters["lens_center_x_arcsec"],
        "lens_center_y_arcsec": parameters["lens_center_y_arcsec"],
        "source_x_arcsec": parameters["source_x_arcsec"],
        "source_y_arcsec": parameters["source_y_arcsec"],
        "source_q": parameters["source_q"],
        "source_phi_rad": parameters["source_phi_rad"],
        "source_re_arcsec": parameters["source_re_arcsec"],
        "source_n": parameters["source_n"],
        "source_amp": parameters["source_amp"],
        "supersampling_factor": render.supersampling_factor,
        "simulator_name": "lenstronomy",
        "simulator_version": simulator_version,
        "parameter_seed": parameters["parameter_seed"],
        "noise_seed": parameters["noise_seed"],
        "config_hash": config_hash,
    }


def generate_sie_extended(
    output_dir: str | Path,
    *,
    count: int = 20,
    seed: int = 3001,
    simulation_id: str = "sim-p30-v1",
    config_path: str | Path = DEFAULT_CONFIG,
    overwrite: bool = False,
    create_preview: bool = True,
    preview_root: str | Path = DEFAULT_PREVIEW_ROOT,
) -> dict[str, Any]:
    """Generate the twenty additional P30 SIE/extended-source systems."""

    # Keep the local Python 3.13 import deterministic; callers can explicitly set
    # NUMBA_DISABLE_JIT=0 when profiling a larger generation run.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    output_root = Path(output_dir)
    config_path = Path(config_path)
    if count <= 0:
        raise ValueError("count must be positive")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty and overwrite is disabled: {output_root}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = _config_hash(config_path)
    simulator_version = backend_status().version or "unavailable"
    output_root.mkdir(parents=True, exist_ok=True)
    directories = {
        name: output_root / name
        for name in (
            "raw",
            "science",
            "normalized",
            "masks",
            "variance",
            "geometry",
            "background",
            "transforms",
            "noise",
            "parameters",
            "diagnostics",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for index in range(count):
        sample_id = f"{simulation_id}-{index + 1:04d}"
        parent_id = f"{simulation_id}-parent-{index + 1:04d}"
        parameters = draw_sie_parameters(seed, index, config)
        paths = {
            "raw": directories["raw"] / f"{sample_id}.fits",
            "science": directories["science"] / f"{sample_id}.npy",
            "normalized": directories["normalized"] / f"{sample_id}.npy",
            "invalid_mask": directories["masks"] / f"{sample_id}_invalid.npy",
            "coverage_mask": directories["masks"] / f"{sample_id}_coverage.npy",
            "background_mask": directories["masks"] / f"{sample_id}_background.npy",
            "variance": directories["variance"] / f"{sample_id}.npy",
            "geometry_transform": directories["geometry"] / f"{sample_id}.json",
            "render_geometry": directories["geometry"] / f"{sample_id}_render.json",
            "background": directories["background"] / f"{sample_id}.json",
            "normalization": directories["transforms"] / f"{sample_id}.json",
            "noise_model": directories["noise"] / f"{sample_id}.json",
            "parameters": directories["parameters"] / f"{sample_id}.json",
            "diagnostics": directories["diagnostics"] / f"{sample_id}.json",
        }
        try:
            parameter_errors = validate_sie_parameters(parameters)
            if parameter_errors:
                raise ValueError("; ".join(parameter_errors))
            render = render_sie_extended_source(
                parameters,
                supersampling_factor=int(config["pilot"]["sie_extended"]["supersampling_factor"]),
            )
            _write_raw_fits(
                paths["raw"],
                render.image,
                parameters,
                sample_id,
                parent_id,
                simulation_id=simulation_id,
                model="SIE",
                simulation_type="LENSTRONOMY_EXTENDED_NOISELESS",
                overwrite=overwrite,
            )
            processed = preprocess_image(
                paths["raw"],
                target_shape=tuple(config["grid"]["shape"]),
                normalization_method=config["preprocessing"]["normalization_method"],
                lower=config["preprocessing"]["lower_percentile"],
                upper=config["preprocessing"]["upper_percentile"],
                background_sigma_clip=config["preprocessing"]["background_sigma_clip"],
                background_max_iterations=config["preprocessing"]["background_max_iterations"],
                background_min_valid_pixels=config["preprocessing"]["background_min_valid_pixels"],
                image_unit="relative_flux",
                noise_floor=config["preprocessing"]["noise_floor"],
                noise_correlation_assumption="not_applicable_noiseless_control",
            )
            np.save(paths["science"], processed["science"])
            np.save(paths["normalized"], processed["normalized"])
            np.save(paths["invalid_mask"], processed["invalid_mask"])
            np.save(paths["coverage_mask"], processed["coverage_mask"])
            np.save(paths["background_mask"], processed["background_mask"])
            np.save(paths["variance"], processed["variance"].values)
            _write_json(paths["geometry_transform"], processed["geometry_transform"].to_dict())
            _write_json(paths["render_geometry"], render.geometry())
            _write_json(paths["background"], processed["background"].to_dict())
            _write_json(paths["normalization"], processed["normalization"].to_dict())
            _write_json(paths["noise_model"], processed["variance"].to_dict())
            diagnostics = {
                "sample_id": sample_id,
                "parent_system_id": parent_id,
                "supersampling_factor": render.supersampling_factor,
                "lens_parameters": render.lens_parameters,
                "source_parameters": render.source_parameters,
                "rendered_flux": float(render.image.sum()),
                "rendered_max": float(render.image.max()),
            }
            _write_json(paths["diagnostics"], diagnostics)
            _write_json(
                paths["parameters"],
                {
                    **parameters,
                    "sample_id": sample_id,
                    "parent_system_id": parent_id,
                    "simulation_id": simulation_id,
                    "simulator_name": "lenstronomy",
                    "simulator_version": simulator_version,
                    "config_path": config_path.as_posix(),
                    "config_hash": config_hash,
                    "backend_reference": backend_status().to_dict(),
                },
            )
            rows.append(
                _sie_manifest_row(
                    sample_id=sample_id,
                    parent_id=parent_id,
                    simulation_id=simulation_id,
                    parameters=parameters,
                    render=render,
                    output_root=output_root,
                    paths=paths,
                    preprocess=processed,
                    config_hash=config_hash,
                    simulator_version=simulator_version,
                )
            )
        except Exception as exc:
            rejections.append(
                {
                    "sample_id": sample_id,
                    "parent_system_id": parent_id,
                    "stage": "render_or_write",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    manifest_path = output_root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    rejection_path = output_root / "rejection_log.csv"
    with rejection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "parent_system_id", "stage", "error_type", "message"],
        )
        writer.writeheader()
        writer.writerows(rejections)
    preview_index = None
    preview_output = None
    if create_preview and rows:
        preview_index, preview_output = _create_simulation_previews(
            directories["raw"], simulation_id, preview_root
        )
    run_manifest = {
        "run_id": simulation_id,
        "generator_version": GENERATOR_VERSION,
        "simulation_id": simulation_id,
        "config_path": config_path.as_posix(),
        "config_hash": config_hash,
        "seed": int(seed),
        "count_requested": int(count),
        "count_attempted": int(count),
        "count_accepted": len(rows),
        "count_rejected": len(rejections),
        "output_root": output_root.as_posix(),
        "manifest_path": "manifest.csv",
        "rejection_log_path": "rejection_log.csv",
        "preview_index_path": None if preview_output is None else (preview_output / "preview_index.csv").as_posix(),
        "backend": backend_status().to_dict(),
        "noise_policy": "noiseless image; positive numerical variance floor only",
        "supersampling_factor": int(config["pilot"]["sie_extended"]["supersampling_factor"]),
        "rejections": rejections,
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    return run_manifest


def generate_stress_pilot(
    output_dir: str | Path,
    *,
    count: int = 40,
    seed: int = 9001,
    simulation_id: str = "sim-p100-v1",
    config_path: str | Path = DEFAULT_CONFIG,
    overwrite: bool = False,
    create_preview: bool = True,
    preview_root: str | Path = DEFAULT_PREVIEW_ROOT,
) -> dict[str, Any]:
    """Generate the forty-system P100 SIE-plus-shear stress increment."""

    return generate_observation_pilot(
        output_dir,
        count=count,
        seed=seed,
        simulation_id=simulation_id,
        config_path=config_path,
        overwrite=overwrite,
        create_preview=create_preview,
        preview_root=preview_root,
        parameter_drawer=draw_stress_parameters,
        stage_name="stress_increment",
    )


def _observation_manifest_row(
    *,
    sample_id: str,
    parent_id: str,
    simulation_id: str,
    parameters: dict[str, Any],
    scene: Any,
    detector: Any,
    preprocess: dict[str, Any],
    output_root: Path,
    paths: dict[str, Path],
    config_hash: str,
    detectable_threshold: float,
    simulator_version: str,
) -> dict[str, Any]:
    background = preprocess["background"]
    variance = preprocess["variance"]
    normalization = preprocess["normalization"]
    physical_count = len(scene.point_positions_x_arcsec)
    detectable_count = sum(value >= detectable_threshold for value in scene.point_amplitudes)
    return {
        "sample_id": sample_id,
        "lens_id": parent_id,
        "domain": "simulated",
        "schema_version": "2.0",
        "dataset_version": simulation_id,
        "simulation_id": simulation_id,
        "parent_system_id": parent_id,
        "split_group_id": parent_id,
        "source_template_id": f"analytic-sersic-point:{parent_id}",
        "background_template_id": f"analytic-scalar:{parent_id}",
        "raw_path": _relative(paths["raw"], output_root),
        "science_path": _relative(paths["science"], output_root),
        "processed_path": _relative(paths["normalized"], output_root),
        "mask_path": _relative(paths["invalid_mask"], output_root),
        "coverage_mask_path": _relative(paths["coverage_mask"], output_root),
        "background_mask_path": _relative(paths["background_mask"], output_root),
        "variance_path": _relative(paths["variance"], output_root),
        "expected_variance_path": _relative(paths["expected_variance"], output_root),
        "clean_scene_path": _relative(paths["clean_scene"], output_root),
        "expected_scene_path": _relative(paths["expected_scene"], output_root),
        "observed_science_path": _relative(paths["observed_science"], output_root),
        "psf_path": _relative(paths["psf_values"], output_root),
        "psf_metadata_path": _relative(paths["psf_metadata"], output_root),
        "geometry_transform_path": _relative(paths["geometry_transform"], output_root),
        "render_geometry_path": _relative(paths["render_geometry"], output_root),
        "background_path": _relative(paths["background"], output_root),
        "normalization_path": _relative(paths["normalization"], output_root),
        "noise_model_path": _relative(paths["noise_model"], output_root),
        "image_positions_path": _relative(paths["diagnostics"], output_root),
        "parameters_path": _relative(paths["parameters"], output_root),
        "image_unit": "electrons_per_second",
        "calibration_status": "pass",
        "original_shape": "x".join(str(value) for value in scene.expected_scene.shape),
        "processed_shape": "x".join(str(value) for value in preprocess["normalized"].shape),
        "pixel_scale": parameters["pixel_scale_arcsec"],
        "pixel_scale_x": parameters["pixel_scale_arcsec"],
        "pixel_scale_y": parameters["pixel_scale_arcsec"],
        "processing_config_id": f"simulation-preprocessing-{config_hash[:12]}",
        "processing_version": "2.0.0",
        "background_status": background.status,
        "background_method": background.method,
        "normalization_method": normalization.method,
        "variance_method": "preprocessing_toy_scalar_background",
        "variance_status": variance.status,
        "variance_unit": variance.unit,
        "expected_variance_method": "poisson_plus_read_noise_rate_variance",
        "noise_correlation_assumption": "independent_poisson_plus_independent_read_noise",
        "mask_semantics": "true means excluded",
        "masked_fraction": float(preprocess["background_mask"].mean()),
        "source_type": parameters.get("source_type", "sersic_plus_point_source"),
        "mass_target_available": False,
        "theta_E_target_available": True,
        "structural_status": "pass",
        "observation_status": "pass",
        "science_status": "pass",
        "eligible_for": "simulation;classical_fit;ml",
        "exclusion_reason": "",
        "mass_model": parameters.get("mass_model", "SIE"),
        "theta_E_arcsec": parameters["theta_E_arcsec"],
        "q": parameters["q"],
        "phi_rad": parameters["phi_rad"],
        "e1": scene.lens_parameters["e1"],
        "e2": scene.lens_parameters["e2"],
        "gamma1": parameters.get("gamma1"),
        "gamma2": parameters.get("gamma2"),
        "gamma_amplitude": parameters.get("gamma_amplitude"),
        "gamma_phi_rad": parameters.get("gamma_phi_rad"),
        "lens_center_x_arcsec": parameters["lens_center_x_arcsec"],
        "lens_center_y_arcsec": parameters["lens_center_y_arcsec"],
        "source_x_arcsec": parameters["source_x_arcsec"],
        "source_y_arcsec": parameters["source_y_arcsec"],
        "source_q": parameters["source_q"],
        "source_phi_rad": parameters["source_phi_rad"],
        "source_re_arcsec": parameters["source_re_arcsec"],
        "source_n": parameters["source_n"],
        "source_amp": parameters["source_amp"],
        "point_source_amp": parameters["point_source_amp"],
        "stress_case": parameters.get("stress_case"),
        "image_count_physical": physical_count,
        "image_count_detectable": detectable_count,
        "detectable_amplitude_threshold": detectable_threshold,
        "psf_fwhm_arcsec": parameters["psf_fwhm_arcsec"],
        "psf_method": "analytic_gaussian_control_zero_boundary",
        "psf_id": "observation-gaussian-control",
        "exposure_time": parameters["exposure_time"],
        "background_rate": parameters["background_rate"],
        "read_noise_electrons": parameters["read_noise_electrons"],
        "supersampling_factor": scene.supersampling_factor,
        "simulator_name": "lenstronomy",
        "simulator_version": simulator_version,
        "parameter_seed": parameters["parameter_seed"],
        "noise_seed": parameters["noise_seed"],
        "config_hash": config_hash,
    }


def generate_observation_pilot(
    output_dir: str | Path,
    *,
    count: int = 30,
    seed: int = 6001,
    simulation_id: str = "sim-p60-v1",
    config_path: str | Path = DEFAULT_CONFIG,
    overwrite: bool = False,
    create_preview: bool = True,
    preview_root: str | Path = DEFAULT_PREVIEW_ROOT,
    parameter_drawer: Callable[[int, int, dict[str, Any]], dict[str, Any]] = draw_observation_parameters,
    stage_name: str = "observation",
) -> dict[str, Any]:
    """Generate P60 systems with point sources, PSF, and detector noise."""

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    output_root = Path(output_dir)
    config_path = Path(config_path)
    if count <= 0:
        raise ValueError("count must be positive")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty and overwrite is disabled: {output_root}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = _config_hash(config_path)
    simulator_version = backend_status().version or "unavailable"
    effects = config["pilot"]["observation_effects"]
    detectable_threshold = float(effects["detectable_amplitude_threshold"])
    output_root.mkdir(parents=True, exist_ok=True)
    directory_names = (
        "raw",
        "science",
        "normalized",
        "masks",
        "variance",
        "expected_variance",
        "scenes",
        "geometry",
        "background",
        "transforms",
        "noise",
        "psf",
        "parameters",
        "diagnostics",
    )
    directories = {name: output_root / name for name in directory_names}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for index in range(count):
        sample_id = f"{simulation_id}-{index + 1:04d}"
        parent_id = f"{simulation_id}-parent-{index + 1:04d}"
        parameters = parameter_drawer(seed, index, config)
        parameters["supersampling_factor"] = int(
            parameters.get("supersampling_factor", config["pilot"]["sie_extended"]["supersampling_factor"])
        )
        paths = {
            "raw": directories["raw"] / f"{sample_id}.fits",
            "science": directories["science"] / f"{sample_id}.npy",
            "normalized": directories["normalized"] / f"{sample_id}.npy",
            "invalid_mask": directories["masks"] / f"{sample_id}_invalid.npy",
            "coverage_mask": directories["masks"] / f"{sample_id}_coverage.npy",
            "background_mask": directories["masks"] / f"{sample_id}_background.npy",
            "variance": directories["variance"] / f"{sample_id}.npy",
            "expected_variance": directories["expected_variance"] / f"{sample_id}.npy",
            "clean_scene": directories["scenes"] / f"{sample_id}_clean.npy",
            "expected_scene": directories["scenes"] / f"{sample_id}_expected.npy",
            "observed_science": directories["scenes"] / f"{sample_id}_observed_science.npy",
            "geometry_transform": directories["geometry"] / f"{sample_id}.json",
            "render_geometry": directories["geometry"] / f"{sample_id}_render.json",
            "background": directories["background"] / f"{sample_id}.json",
            "normalization": directories["transforms"] / f"{sample_id}.json",
            "noise_model": directories["noise"] / f"{sample_id}.json",
            "psf": directories["psf"] / f"{sample_id}.json",
            "psf_metadata": directories["psf"] / f"{sample_id}_metadata.json",
            "psf_values": directories["psf"] / f"{sample_id}.npy",
            "parameters": directories["parameters"] / f"{sample_id}.json",
            "diagnostics": directories["diagnostics"] / f"{sample_id}.json",
        }
        try:
            parameter_errors = validate_sie_parameters(parameters)
            if parameter_errors:
                raise ValueError("; ".join(parameter_errors))
            solved_scene = render_observation_scene(
                parameters,
                psf_fwhm_arcsec=float(parameters["psf_fwhm_arcsec"]),
                supersampling_factor=int(parameters["supersampling_factor"]),
            )
            # Freeze the backend's solved point roots at the artifact boundary.
            # Re-rendering from LENSED_POSITION avoids tiny source-solver changes
            # altering PSF-painted pixels or downstream Poisson draws.
            point_position_decimals = 8
            parameters["point_positions_x_arcsec"] = [
                float(np.round(value, point_position_decimals))
                for value in solved_scene.point_positions_x_arcsec
            ]
            parameters["point_positions_y_arcsec"] = [
                float(np.round(value, point_position_decimals))
                for value in solved_scene.point_positions_y_arcsec
            ]
            parameters["point_amplitudes"] = list(solved_scene.point_amplitudes)
            parameters["point_position_rounding_decimals"] = point_position_decimals
            scene = render_observation_scene(
                parameters,
                psf_fwhm_arcsec=float(parameters["psf_fwhm_arcsec"]),
                supersampling_factor=int(parameters["supersampling_factor"]),
            )
            parameters["point_amplitudes"] = list(scene.point_amplitudes)
            parameters["point_amplitude_rounding_decimals"] = 3
            detector = realize_detector_observation(
                scene.expected_scene,
                background_rate=parameters["background_rate"],
                exposure_time=parameters["exposure_time"],
                read_noise_electrons=parameters["read_noise_electrons"],
                noise_seed=parameters["noise_seed"],
            )
            _write_raw_fits(
                paths["raw"],
                detector.observed_rate,
                parameters,
                sample_id,
                parent_id,
                simulation_id=simulation_id,
                model="SIE",
                simulation_type="LENSTRONOMY_OBSERVED_NOISE",
                image_unit="ELECTRONS/S",
                overwrite=overwrite,
            )
            processed = preprocess_image(
                paths["raw"],
                target_shape=tuple(config["grid"]["shape"]),
                normalization_method=config["preprocessing"]["normalization_method"],
                lower=config["preprocessing"]["lower_percentile"],
                upper=config["preprocessing"]["upper_percentile"],
                background_sigma_clip=config["preprocessing"]["background_sigma_clip"],
                background_max_iterations=config["preprocessing"]["background_max_iterations"],
                background_min_valid_pixels=config["preprocessing"]["background_min_valid_pixels"],
                image_unit="electrons_per_second",
                noise_floor=config["preprocessing"]["noise_floor"],
                noise_correlation_assumption="independent_poisson_plus_independent_read_noise",
            )
            np.save(paths["science"], processed["science"])
            np.save(paths["normalized"], processed["normalized"])
            np.save(paths["invalid_mask"], processed["invalid_mask"])
            np.save(paths["coverage_mask"], processed["coverage_mask"])
            np.save(paths["background_mask"], processed["background_mask"])
            np.save(paths["variance"], processed["variance"].values)
            np.save(paths["expected_variance"], detector.variance_rate)
            np.save(paths["clean_scene"], scene.clean_scene)
            np.save(paths["expected_scene"], scene.expected_scene)
            np.save(paths["observed_science"], detector.science_rate)
            _write_json(paths["geometry_transform"], processed["geometry_transform"].to_dict())
            _write_json(paths["render_geometry"], {
                "coordinate_convention": "x_y_arcsec_about_lens_center",
                "image_index_order": "image[y, x]",
                "pixel_center_convention": "zero_based_pixel_centers",
                "shape": list(scene.expected_scene.shape),
                "pixel_scale_arcsec": scene.pixel_scale_arcsec,
                "supersampling_factor": scene.supersampling_factor,
                "lens_model": parameters.get("mass_model", "SIE"),
                "shear_parameters": scene.shear_parameters,
            })
            _write_json(paths["background"], processed["background"].to_dict())
            _write_json(paths["normalization"], processed["normalization"].to_dict())
            _write_json(paths["noise_model"], {
                **detector.to_dict(),
                "expected_variance_path": _relative(paths["expected_variance"], output_root),
            })
            psf_kernel = observation_psf(
                psf_fwhm_arcsec=scene.psf_fwhm_arcsec,
                pixel_scale_arcsec=scene.pixel_scale_arcsec,
            )
            np.save(paths["psf_values"], psf_kernel.values)
            psf_record = {
                "psf_type": "GAUSSIAN",
                "fwhm_arcsec": scene.psf_fwhm_arcsec,
                "pixel_scale_arcsec": scene.pixel_scale_arcsec,
                "sigma_pixels": scene.psf_fwhm_arcsec / (
                    2.0 * np.sqrt(2.0 * np.log(2.0)) * scene.pixel_scale_arcsec
                ),
                "normalization": "unit_sum",
                "kernel_size_policy": "odd ceil(10 sigma), minimum 5",
                "source": "analytic_gaussian_control",
                "convolution": "deterministic_zero_boundary",
                "values_path": _relative(paths["psf_values"], output_root),
                "psf_id": psf_kernel.psf_id,
                "shape": list(psf_kernel.values.shape),
                "sum": float(np.sum(psf_kernel.values)),
            }
            _write_json(paths["psf"], psf_record)
            _write_json(paths["psf_metadata"], psf_kernel.to_dict(_relative(paths["psf_values"], output_root)))
            _write_json(paths["diagnostics"], {
                "sample_id": sample_id,
                "parent_system_id": parent_id,
                "supersampling_factor": scene.supersampling_factor,
                "point_positions_x_arcsec": list(scene.point_positions_x_arcsec),
                "point_positions_y_arcsec": list(scene.point_positions_y_arcsec),
                "point_amplitudes": list(scene.point_amplitudes),
                "image_count_physical": len(scene.point_positions_x_arcsec),
                "image_count_detectable": sum(value >= detectable_threshold for value in scene.point_amplitudes),
                "detectable_amplitude_threshold": detectable_threshold,
                "clean_scene_flux": float(scene.clean_scene.sum()),
                "expected_scene_flux": float(scene.expected_scene.sum()),
                "observed_science_mean": float(detector.science_rate.mean()),
                "expected_variance_mean": float(detector.variance_rate.mean()),
            })
            _write_json(paths["parameters"], {
                **parameters,
                "sample_id": sample_id,
                "parent_system_id": parent_id,
                "simulation_id": simulation_id,
                "simulator_name": "lenstronomy",
                "simulator_version": simulator_version,
                "config_path": config_path.as_posix(),
                "config_hash": config_hash,
                "backend_reference": backend_status().to_dict(),
                "lens_parameters": scene.lens_parameters,
                "shear_parameters": scene.shear_parameters,
                "source_parameters": scene.source_parameters,
                "lens_light_parameters": scene.lens_light_parameters,
                "point_source_parameters": scene.point_source_parameters,
                "point_positions_x_arcsec": list(scene.point_positions_x_arcsec),
                "point_positions_y_arcsec": list(scene.point_positions_y_arcsec),
                "point_amplitudes": list(scene.point_amplitudes),
            })
            rows.append(
                _observation_manifest_row(
                    sample_id=sample_id,
                    parent_id=parent_id,
                    simulation_id=simulation_id,
                    parameters=parameters,
                    scene=scene,
                    detector=detector,
                    preprocess=processed,
                    output_root=output_root,
                    paths=paths,
                    config_hash=config_hash,
                    detectable_threshold=detectable_threshold,
                    simulator_version=simulator_version,
                )
            )
        except Exception as exc:
            rejections.append({
                "sample_id": sample_id,
                "parent_system_id": parent_id,
                "stage": "render_or_write",
                "error_type": type(exc).__name__,
                "message": str(exc),
            })

    manifest_path = output_root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    rejection_path = output_root / "rejection_log.csv"
    with rejection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "parent_system_id", "stage", "error_type", "message"])
        writer.writeheader()
        writer.writerows(rejections)
    preview_index = None
    preview_output = None
    if create_preview and rows:
        preview_index, preview_output = _create_simulation_previews(
            directories["raw"], simulation_id, preview_root
        )
    run_manifest = {
        "run_id": simulation_id,
        "generator_version": GENERATOR_VERSION,
        "simulation_id": simulation_id,
        "simulation_stage": stage_name,
        "config_path": config_path.as_posix(),
        "config_hash": config_hash,
        "seed": int(seed),
        "count_requested": int(count),
        "count_attempted": int(count),
        "count_accepted": len(rows),
        "count_rejected": len(rejections),
        "output_root": output_root.as_posix(),
        "manifest_path": "manifest.csv",
        "rejection_log_path": "rejection_log.csv",
        "preview_index_path": None if preview_output is None else (preview_output / "preview_index.csv").as_posix(),
        "backend": backend_status().to_dict(),
        "noise_policy": "Poisson in electrons plus Gaussian read noise before background subtraction",
        "point_source_rendering": "bilinear_pixel_deposition_then_analytic_gaussian_control_convolution",
        "point_position_rounding_decimals": 8,
        "point_amplitude_rounding_decimals": 3,
        "rejections": rejections,
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    return run_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/simulations/sim-p10-v1")
    parser.add_argument("--stage", choices=("sis", "sie", "observation", "stress"), default="sis")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--simulation-id", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--preview-root", default=str(DEFAULT_PREVIEW_ROOT))
    args = parser.parse_args()
    generators = {
        "sis": generate_sis_controls,
        "sie": generate_sie_extended,
        "observation": generate_observation_pilot,
        "stress": generate_stress_pilot,
    }
    defaults = {
        "sis": (10, "sim-p10-v1"),
        "sie": (20, "sim-p30-v1"),
        "observation": (30, "sim-p60-v1"),
        "stress": (40, "sim-p100-v1"),
    }
    generator = generators[args.stage]
    default_count, default_id = defaults[args.stage]
    simulation_id = args.simulation_id or default_id
    summary = generator(
        args.output,
        count=args.count or default_count,
        seed=args.seed,
        simulation_id=simulation_id,
        config_path=args.config,
        overwrite=args.overwrite,
        create_preview=not args.no_preview,
        preview_root=args.preview_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["count_rejected"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
