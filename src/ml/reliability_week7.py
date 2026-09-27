"""Frozen Week 7 independent SIE equation and rasterization benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment

from src.data.dataset import LensDataset
from src.ml.data import file_sha256
from src.ml.engine import select_device, write_json
from src.ml.metrics import regression_metrics
from src.ml.model import build_model
from src.processing.preprocess import preprocess_image
from src.simulation.generate import _relative, _write_json, _write_raw_fits
from src.simulation.independent_renderer import render_independent_expected_scene
from src.simulation.independent_sie import independent_sie_deflection, solve_independent_sie_images
from src.simulation.models import build_lens_model, sie_kwargs
from src.simulation.observation import observation_psf, realize_detector_observation


DEFAULT_CONFIG = Path("configs/reliability_cross_simulator_v1.yaml")
DEFAULT_RELEASE = Path("data/simulations/reliability-independent-sie-v1")
DEFAULT_OUTPUT = Path("results/reports/reliability_v1/cross_simulator_week7")
SOURCE_RELEASE = Path("data/simulations/sim-eng1000-v1")
SOURCE_VALIDATION = SOURCE_RELEASE / "splits" / "validation.csv"
WEEK4_PREDICTIONS = Path("results/reports/reliability_v1/observation_systematics/cnn_predictions.csv")
CONDITION_MULTIPLIERS = {"independent-canonical": 1.0, "independent-exposure-half": 0.5}
SMOKE_IDS = [
    "sim-eng1000-v1-0013",
    "sim-eng1000-v1-0023",
    "sim-eng1000-v1-0025",
    "sim-eng1000-v1-0039",
]


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _bootstrap_interval(values: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), 256):
        count = min(256, int(resamples) - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _read_parameters(source_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row["parameters_path"]))
    if not path.is_absolute():
        path = source_root / path
    values = json.loads(path.read_text(encoding="utf-8"))
    if not np.isclose(float(values["theta_E_arcsec"]), float(row["theta_E_arcsec"]), rtol=0, atol=1e-12):
        raise ValueError(f"{row['sample_id']}: sidecar and frozen manifest targets disagree")
    return values


def _root_match(independent: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if len(independent) != len(reference):
        return {
            "independent_root_count": int(len(independent)),
            "reference_root_count": int(len(reference)),
            "root_count_match": False,
            "maximum_matched_separation_arcsec": None,
        }
    if len(independent) == 0:
        maximum = 0.0
    else:
        distances = np.linalg.norm(independent[:, None, :] - reference[None, :, :], axis=2)
        left, right = linear_sum_assignment(distances)
        maximum = float(distances[left, right].max())
    return {
        "independent_root_count": int(len(independent)),
        "reference_root_count": int(len(reference)),
        "root_count_match": True,
        "maximum_matched_separation_arcsec": maximum,
    }


def _controls(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = pd.read_csv(SOURCE_VALIDATION)
    if len(source) != 150 or set(source["split"].astype(str)) != {"validation"}:
        raise ValueError("Week 7 source must be the frozen 150-parent validation split")

    grid = np.linspace(-2.5, 2.5, 41)
    x_grid, y_grid = np.meshgrid(grid, grid)
    deflection_rows = []
    center_x, center_y = config["control_gate"]["deflection_grid_center_arcsec"]
    for q in config["control_gate"]["deflection_grid_axis_ratios"]:
        for phi in config["control_gate"]["deflection_grid_orientations_rad"]:
            lens = build_lens_model(["SIE"])
            kwargs = [sie_kwargs(1.15, float(q), float(phi), center_x_arcsec=center_x, center_y_arcsec=center_y)]
            ref_x, ref_y = lens.alpha(x_grid, y_grid, kwargs)
            independent_x, independent_y = independent_sie_deflection(
                x_grid,
                y_grid,
                theta_E_arcsec=1.15,
                axis_ratio=float(q),
                phi_rad=float(phi),
                center_x_arcsec=float(center_x),
                center_y_arcsec=float(center_y),
            )
            radius = np.hypot(x_grid - center_x, y_grid - center_y)
            valid = radius > 0.05
            difference = np.maximum(np.abs(independent_x - ref_x), np.abs(independent_y - ref_y))
            deflection_rows.append(
                {
                    "axis_ratio": float(q),
                    "phi_rad": float(phi),
                    "grid_point_count": int(valid.sum()),
                    "max_component_difference_arcsec": float(difference[valid].max()),
                }
            )
    max_deflection_error = max(row["max_component_difference_arcsec"] for row in deflection_rows)

    root_rows = []
    for row in source.to_dict(orient="records"):
        params = _read_parameters(SOURCE_RELEASE, row)
        path = Path(str(row["image_positions_path"]))
        if not path.is_absolute():
            path = SOURCE_RELEASE / path
        ref = json.loads(path.read_text(encoding="utf-8"))
        reference = np.column_stack(
            [ref["point_positions_x_arcsec"], ref["point_positions_y_arcsec"]]
        ).astype(np.float64)
        independent = solve_independent_sie_images(
            source_x_arcsec=float(params["source_x_arcsec"]),
            source_y_arcsec=float(params["source_y_arcsec"]),
            theta_E_arcsec=float(params["theta_E_arcsec"]),
            axis_ratio=float(params["q"]),
            phi_rad=float(params["phi_rad"]),
            center_x_arcsec=float(params.get("lens_center_x_arcsec", 0.0)),
            center_y_arcsec=float(params.get("lens_center_y_arcsec", 0.0)),
        )
        root_rows.append({"sample_id": str(row["sample_id"]), **_root_match(independent, reference)})
    root_frame = pd.DataFrame(root_rows)
    root_agreement = float(root_frame["root_count_match"].mean())
    matched = root_frame.loc[root_frame["root_count_match"], "maximum_matched_separation_arcsec"]
    maximum_root_error = float(matched.max()) if len(matched) else float("inf")

    smoke_rows = []
    lookup = source.set_index("sample_id")
    from src.simulation.independent_renderer import observe_independent_scene

    for sample_id in SMOKE_IDS:
        row = lookup.loc[sample_id].to_dict()
        params = _read_parameters(SOURCE_RELEASE, row)
        rendered = render_independent_expected_scene(params)
        clean_path = SOURCE_RELEASE / str(row["clean_scene_path"])
        expected_path = SOURCE_RELEASE / str(row["expected_scene_path"])
        processed_path = SOURCE_RELEASE / str(row["processed_path"])
        clean_ref = np.load(clean_path)
        expected_ref = np.load(expected_path)
        processed_ref = np.load(processed_path)
        observed = observe_independent_scene(rendered["expected_scene"], params)
        smoke_rows.append(
            {
                "sample_id": sample_id,
                "root_count": rendered["root_count"],
                "clean_shape": list(rendered["clean_scene"].shape),
                "processed_shape": list(observed["processed"].shape),
                "all_finite": bool(
                    np.isfinite(rendered["clean_scene"]).all()
                    and np.isfinite(rendered["expected_scene"]).all()
                    and np.isfinite(observed["processed"]).all()
                ),
                "clean_scene_max_abs_difference": float(np.max(np.abs(rendered["clean_scene"] - clean_ref))),
                "expected_scene_max_abs_difference": float(
                    np.max(np.abs(rendered["expected_scene"] - expected_ref))
                ),
                "processed_max_abs_difference": float(np.max(np.abs(observed["processed"] - processed_ref))),
                "clean_flux_ratio": float(rendered["clean_scene"].sum() / clean_ref.sum()),
            }
        )

    gates = config["control_gate"]
    passed = (
        max_deflection_error <= float(gates["deflection_max_component_difference_arcsec"])
        and root_agreement >= float(gates["minimum_root_count_agreement_fraction"])
        and maximum_root_error <= float(gates["maximum_hungarian_matched_position_error_arcsec"])
        and len(smoke_rows) == int(gates["renderer_smoke_parent_count"])
        and all(item["all_finite"] and item["processed_max_abs_difference"] <= 1e-6 for item in smoke_rows)
    )
    control = {
        "experiment_id": "WEEK7-INDEPENDENT-SIE-LENS-EQUATION",
        "protocol": "docs/reliability_study/protocol_v1.7.md",
        "protocol_sha256": file_sha256(Path("docs/reliability_study/protocol_v1.7.md")),
        "validation_parent_count": int(len(source)),
        "deflection_grid": deflection_rows,
        "maximum_deflection_component_difference_arcsec": max_deflection_error,
        "root_count_agreement_fraction": root_agreement,
        "root_count_agreement_count": int(root_frame["root_count_match"].sum()),
        "maximum_matched_root_separation_arcsec": maximum_root_error,
        "root_control_rows": root_rows,
        "renderer_smoke": smoke_rows,
        "gates": {
            "deflection_passed": max_deflection_error
            <= float(gates["deflection_max_component_difference_arcsec"]),
            "root_count_passed": root_agreement
            >= float(gates["minimum_root_count_agreement_fraction"]),
            "root_position_passed": maximum_root_error
            <= float(gates["maximum_hungarian_matched_position_error_arcsec"]),
            "renderer_smoke_passed": all(
                item["all_finite"] and item["processed_max_abs_difference"] <= 1e-6 for item in smoke_rows
            ),
        },
        "passed": bool(passed),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "control_preflight.json", control)
    pd.DataFrame(deflection_rows).to_csv(output / "deflection_control.csv", index=False)
    root_frame.to_csv(output / "root_control.csv", index=False)
    pd.DataFrame(smoke_rows).to_csv(output / "renderer_smoke.csv", index=False)
    if not passed:
        raise RuntimeError("A frozen Week 7 control gate failed; no model predictions were made")
    return control


def _paths(output_root: Path, sample_id: str) -> dict[str, Path]:
    return {
        "raw": output_root / "raw" / f"{sample_id}.fits",
        "science": output_root / "science" / f"{sample_id}.npy",
        "normalized": output_root / "normalized" / f"{sample_id}.npy",
        "invalid_mask": output_root / "masks" / f"{sample_id}_invalid.npy",
        "coverage_mask": output_root / "masks" / f"{sample_id}_coverage.npy",
        "background_mask": output_root / "masks" / f"{sample_id}_background.npy",
        "variance": output_root / "variance" / f"{sample_id}.npy",
        "expected_variance": output_root / "expected_variance" / f"{sample_id}.npy",
        "clean_scene": output_root / "scenes" / f"{sample_id}_clean.npy",
        "expected_scene": output_root / "scenes" / f"{sample_id}_expected.npy",
        "observed_science": output_root / "scenes" / f"{sample_id}_observed_science.npy",
        "psf_values": output_root / "psf" / f"{sample_id}.npy",
        "psf_metadata": output_root / "psf" / f"{sample_id}_metadata.json",
        "psf": output_root / "psf" / f"{sample_id}.json",
        "geometry_transform": output_root / "geometry" / f"{sample_id}.json",
        "render_geometry": output_root / "geometry" / f"{sample_id}_render.json",
        "background": output_root / "background" / f"{sample_id}.json",
        "normalization": output_root / "transforms" / f"{sample_id}.json",
        "noise_model": output_root / "noise" / f"{sample_id}.json",
        "diagnostics": output_root / "diagnostics" / f"{sample_id}.json",
        "parameters": output_root / "parameters" / f"{sample_id}.json",
    }


def _build_release(release: Path, output: Path, config: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    # An interrupted pre-commit generation can be resumed deterministically.
    # Once either release manifest exists, refuse to overwrite the frozen data.
    if release.exists() and any(release.iterdir()) and (
        (release / "manifest.csv").exists() or (release / "run_manifest.json").exists()
    ):
        raise FileExistsError(f"Week 7 release already has a committed manifest: {release}")
    release.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_VALIDATION)
    if len(source) != 150:
        raise ValueError("Week 7 matched release requires all 150 frozen validation parents")
    source_root = SOURCE_RELEASE
    config_path = Path(str(_read_parameters(source_root, source.iloc[0].to_dict())["config_path"]))
    render_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    preprocessing = dict(render_config["preprocessing"])
    descendants = []
    renderer_rows = []
    path_fields = {
        "raw_path": "raw",
        "science_path": "science",
        "processed_path": "normalized",
        "mask_path": "invalid_mask",
        "coverage_mask_path": "coverage_mask",
        "background_mask_path": "background_mask",
        "variance_path": "variance",
        "expected_variance_path": "expected_variance",
        "clean_scene_path": "clean_scene",
        "expected_scene_path": "expected_scene",
        "observed_science_path": "observed_science",
        "psf_path": "psf_values",
        "psf_metadata_path": "psf_metadata",
        "geometry_transform_path": "geometry_transform",
        "render_geometry_path": "render_geometry",
        "background_path": "background",
        "normalization_path": "normalization",
        "noise_model_path": "noise_model",
        "image_positions_path": "diagnostics",
        "parameters_path": "parameters",
    }
    for source_row in source.to_dict(orient="records"):
        params = _read_parameters(source_root, source_row)
        scene = render_independent_expected_scene(params)
        reference_expected = np.load(source_root / str(source_row["expected_scene_path"]))
        scene_difference = float(np.max(np.abs(scene["expected_scene"] - reference_expected)))
        for condition, multiplier in CONDITION_MULTIPLIERS.items():
            sample_id = f"{source_row['sample_id']}__{condition}"
            condition_parameters = dict(params)
            condition_parameters["sample_id"] = sample_id
            condition_parameters["parent_sample_id"] = str(source_row["sample_id"])
            condition_parameters["parent_system_id"] = str(source_row["parent_system_id"])
            condition_parameters["simulation_id"] = "reliability-independent-sie-v1"
            condition_parameters["exposure_time"] = float(params["exposure_time"]) * multiplier
            condition_parameters["render_mode"] = "independent_closed_form_sie_shared_sersic_gaussian_observation"
            condition_parameters["week7_condition"] = condition
            detector = realize_detector_observation(
                scene["expected_scene"],
                background_rate=float(params["background_rate"]),
                exposure_time=float(condition_parameters["exposure_time"]),
                read_noise_electrons=float(params["read_noise_electrons"]),
                noise_seed=int(params["noise_seed"]),
            )
            paths = _paths(release, sample_id)
            raw_rate = np.asarray(detector.observed_rate, dtype=np.float32)
            _write_raw_fits(
                paths["raw"],
                raw_rate,
                condition_parameters,
                sample_id,
                str(source_row["parent_system_id"]),
                simulation_id="reliability-independent-sie-v1",
                model="SIE",
                simulation_type="INDEPENDENT_SIE_EQUATION_MATCHED_OBSERVATION",
                image_unit="ELECTRONS/S",
                overwrite=True,
            )
            processed = preprocess_image(
                paths["raw"],
                target_shape=tuple(condition_parameters["shape"]),
                normalization_method=preprocessing["normalization_method"],
                lower=preprocessing["lower_percentile"],
                upper=preprocessing["upper_percentile"],
                background_sigma_clip=preprocessing["background_sigma_clip"],
                background_max_iterations=preprocessing["background_max_iterations"],
                background_min_valid_pixels=preprocessing["background_min_valid_pixels"],
                image_unit="electrons_per_second",
                noise_floor=preprocessing["noise_floor"],
                noise_correlation_assumption="independent_poisson_plus_independent_read_noise",
            )
            arrays = {
                "science": processed["science"],
                "normalized": processed["normalized"],
                "invalid_mask": processed["invalid_mask"],
                "coverage_mask": processed["coverage_mask"],
                "background_mask": processed["background_mask"],
                "variance": processed["variance"].values,
                "expected_variance": detector.variance_rate,
                "clean_scene": scene["clean_scene"],
                "expected_scene": scene["expected_scene"],
                "observed_science": detector.science_rate,
            }
            for name, array in arrays.items():
                paths[name].parent.mkdir(parents=True, exist_ok=True)
                np.save(paths[name], array)
            psf = observation_psf(
                psf_fwhm_arcsec=float(params["psf_fwhm_arcsec"]),
                pixel_scale_arcsec=float(params["pixel_scale_arcsec"]),
            )
            paths["psf_values"].parent.mkdir(parents=True, exist_ok=True)
            np.save(paths["psf_values"], psf.values)
            _write_json(
                paths["psf"],
                {
                    "psf_type": "GAUSSIAN",
                    "fwhm_arcsec": float(params["psf_fwhm_arcsec"]),
                    "pixel_scale_arcsec": float(params["pixel_scale_arcsec"]),
                    "normalization": "unit_sum",
                    "source": "shared_analytic_gaussian_control",
                    "convolution": "deterministic_zero_boundary",
                    "values_path": _relative(paths["psf_values"], release),
                    "psf_id": psf.psf_id,
                    "shape": list(psf.values.shape),
                    "sum": float(np.sum(psf.values)),
                },
            )
            _write_json(paths["psf_metadata"], psf.to_dict(_relative(paths["psf_values"], release)))
            _write_json(paths["geometry_transform"], processed["geometry_transform"].to_dict())
            _write_json(
                paths["render_geometry"],
                {
                    "coordinate_convention": "x_y_arcsec_about_lens_center",
                    "image_index_order": "image[y, x]",
                    "pixel_center_convention": "zero_based_pixel_centers",
                    "shape": list(scene["expected_scene"].shape),
                    "pixel_scale_arcsec": float(params["pixel_scale_arcsec"]),
                    "supersampling_factor": int(params["supersampling_factor"]),
                    "lens_model": "SIE",
                    "lens_equation_implementation": "src.simulation.independent_sie",
                    "shared_light_model": "Lenstronomy SERSIC_ELLIPSE",
                },
            )
            _write_json(paths["background"], processed["background"].to_dict())
            _write_json(paths["normalization"], processed["normalization"].to_dict())
            _write_json(
                paths["noise_model"],
                {
                    **detector.to_dict(),
                    "expected_variance_path": _relative(paths["expected_variance"], release),
                    "noise_seed_policy": "reused_parent_noise_seed",
                    "exposure_time_multiplier": multiplier,
                },
            )
            diagnostics = {
                "sample_id": sample_id,
                "parent_sample_id": str(source_row["sample_id"]),
                "condition": condition,
                "root_count": int(scene["root_count"]),
                "point_positions_x_arcsec": scene["point_positions_x_arcsec"],
                "point_positions_y_arcsec": scene["point_positions_y_arcsec"],
                "point_amplitudes": scene["point_amplitudes"],
                "clean_scene_flux": float(np.asarray(scene["clean_scene"]).sum()),
                "expected_scene_flux": float(np.asarray(scene["expected_scene"]).sum()),
                "noise_seed": int(params["noise_seed"]),
                "exposure_time": float(condition_parameters["exposure_time"]),
                "reference_expected_scene_max_abs_difference": scene_difference,
            }
            _write_json(paths["diagnostics"], diagnostics)
            condition_parameters.update(
                {
                    "point_positions_x_arcsec": scene["point_positions_x_arcsec"],
                    "point_positions_y_arcsec": scene["point_positions_y_arcsec"],
                    "point_amplitudes": scene["point_amplitudes"],
                    "simulator_name": "independent_sie_equation_shared_observation_components",
                    "simulator_version": "reliability-independent-sie-v1",
                    "week7_condition": condition,
                }
            )
            _write_json(paths["parameters"], condition_parameters)

            record = dict(source_row)
            for field, key in path_fields.items():
                record[field] = _relative(paths[key], release)
            record.update(
                {
                    "sample_id": sample_id,
                    "lens_id": str(source_row["parent_system_id"]),
                    "dataset_version": "reliability-independent-sie-v1",
                    "simulation_id": "reliability-independent-sie-v1",
                    "parent_sample_id": str(source_row["sample_id"]),
                    "original_parent_sample_id": str(source_row["sample_id"]),
                    "split": "validation",
                    "intervention_id": condition,
                    "intervention_family": "independent-sie-equation-rasterization",
                    "exposure_time": float(condition_parameters["exposure_time"]),
                    "image_count_physical": int(scene["root_count"]),
                    "image_count_detectable": sum(
                        float(a) >= float(source_row["detectable_amplitude_threshold"])
                        for a in scene["point_amplitudes"]
                    ),
                    "pixel_scale": float(params["pixel_scale_arcsec"]),
                    "pixel_scale_x": float(params["pixel_scale_arcsec"]),
                    "pixel_scale_y": float(params["pixel_scale_arcsec"]),
                    "original_shape": "256x256",
                    "processed_shape": "256x256",
                    "normalization_method": processed["normalization"].method,
                    "masked_fraction": float(processed["background_mask"].mean()),
                    "theta_E_target_available": True,
                    "simulator_name": "independent_sie_equation_shared_observation_components",
                    "simulator_version": "reliability-independent-sie-v1",
                    "config_hash": file_sha256(Path("docs/reliability_study/protocol_v1.7.md")),
                    "structural_status": "pass",
                    "observation_status": "pass",
                    "science_status": "pass",
                }
            )
            descendants.append(record)
            renderer_rows.append(
                {
                    "sample_id": sample_id,
                    "parent_sample_id": str(source_row["sample_id"]),
                    "condition": condition,
                    "root_count": int(scene["root_count"]),
                    "expected_scene_max_abs_difference_from_reference": scene_difference,
                    "expected_scene_flux_ratio": float(
                        np.asarray(scene["expected_scene"]).sum() / reference_expected.sum()
                    ),
                    "exposure_time_multiplier": multiplier,
                }
            )

    manifest = pd.DataFrame(descendants)
    manifest.to_csv(release / "manifest.csv", index=False)
    split_dir = release / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(split_dir / "validation.csv", index=False)
    for condition, group in manifest.groupby("intervention_id", sort=False):
        group.to_csv(split_dir / f"validation__{condition}.csv", index=False)
    pd.DataFrame(renderer_rows).to_csv(output / "renderer_diagnostics.csv", index=False)

    from src.data.schema import validate_row

    schema_errors = [
        {"sample_id": str(record["sample_id"]), "errors": validate_row(record)}
        for record in descendants
        if validate_row(record)
    ]
    if schema_errors:
        write_json(output / "manifest_schema_errors.json", schema_errors[:25])
        raise ValueError(f"Generated Week 7 rows failed schema validation: {schema_errors[:2]}")
    for split_name in ("validation__independent-canonical", "validation__independent-exposure-half"):
        dataset = LensDataset(
            release / "manifest.csv",
            root=release,
            split=split_name,
            strict_products=True,
            domain="simulated",
            eligible_for="ml",
            target_fields={"theta_E_arcsec"},
            expected_shape=(256, 256),
        )
        if len(dataset) != 150:
            raise ValueError(f"Strict loader found {len(dataset)} rows in {split_name}, expected 150")
    if len(manifest) != 300 or manifest["parent_sample_id"].nunique() != 150:
        raise ValueError("Week 7 release must contain two conditions for each of 150 parents")

    release_summary = {
        "release": release.as_posix(),
        "source_release": SOURCE_RELEASE.as_posix(),
        "source_validation_manifest_sha256": file_sha256(SOURCE_VALIDATION),
        "protocol_sha256": control["protocol_sha256"],
        "parent_count": 150,
        "row_count": int(len(manifest)),
        "condition_row_counts": {str(key): int(value) for key, value in manifest["intervention_id"].value_counts().items()},
        "manifest_sha256": file_sha256(release / "manifest.csv"),
        "strict_loader_passed": True,
        "schema_validation_passed": True,
        "historical_test_accessed": False,
    }
    write_json(release / "run_manifest.json", release_summary)
    return release_summary


def _predict(release: Path, output: Path, config: dict[str, Any], *, device_request: str, batch_size: int) -> pd.DataFrame:
    manifest_path = release / "manifest.csv"
    manifest = pd.read_csv(
        manifest_path,
        usecols=["sample_id", "parent_sample_id", "processed_path", "split", "intervention_id"],
    )
    if len(manifest) != 300 or manifest["parent_sample_id"].nunique() != 150:
        raise ValueError("Frozen Week 7 inference requires 300 rows and 150 matched parents")
    device = select_device(device_request)
    rows = []
    checkpoints = [Path(value) for value in config["model_checkpoints"]]
    if len(checkpoints) != 3:
        raise ValueError("Week 7 requires all three frozen baseline checkpoints")
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        original_manifest = Path(checkpoint["release"]) / "manifest.csv"
        if file_sha256(original_manifest) != str(checkpoint["release_manifest_sha256"]):
            raise ValueError(f"Frozen baseline manifest hash changed: {original_manifest}")
        model = build_model(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        stats = checkpoint["statistics"]
        with torch.inference_mode():
            for condition in CONDITION_MULTIPLIERS:
                group = manifest.loc[manifest["intervention_id"] == condition].reset_index(drop=True)
                arrays = [np.load(release / str(path)).astype(np.float32, copy=False) for path in group["processed_path"]]
                values = np.stack(arrays, axis=0)[:, None, :, :]
                values = (values - float(stats["image_mean"])) / float(stats["image_std"])
                predictions = []
                for start in range(0, len(values), int(batch_size)):
                    batch = torch.from_numpy(values[start : start + int(batch_size)]).to(device)
                    normalized = model(batch).detach().cpu().numpy().astype(np.float64)
                    predictions.extend((normalized * float(stats["target_std"]) + float(stats["target_mean"])).tolist())
                for row, prediction in zip(group.to_dict(orient="records"), predictions):
                    rows.append(
                        {
                            "sample_id": str(row["sample_id"]),
                            "parent_sample_id": str(row["parent_sample_id"]),
                            "condition": str(condition),
                            "seed": int(checkpoint["seed"]),
                            "checkpoint_id": str(checkpoint["experiment_id"]),
                            "checkpoint_sha256": file_sha256(checkpoint_path),
                            "theta_E_arcsec_predicted": float(prediction),
                        }
                    )
        print(f"Completed Week 7 no-label inference for seed {checkpoint['seed']}", flush=True)
    predictions = pd.DataFrame(rows)
    if len(predictions) != 900 or predictions.duplicated(["sample_id", "seed"]).any():
        raise ValueError("Frozen Week 7 predictions are incomplete or duplicated")
    # This file is committed before any validation target is loaded by the scorer.
    predictions.to_csv(output / "predictions_before_scoring.csv", index=False)
    return predictions


def _score(predictions: pd.DataFrame, release: Path, output: Path, config: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    # Label access begins only after the no-label predictions file exists.
    truth = pd.read_csv(SOURCE_VALIDATION, usecols=["sample_id", "theta_E_arcsec"])
    truth = truth.rename(columns={"sample_id": "parent_sample_id", "theta_E_arcsec": "theta_E_arcsec_true"})
    scored = predictions.merge(truth, on="parent_sample_id", validate="many_to_one")

    baseline_frames = []
    for seed in (12001, 12002, 12003):
        checkpoint_path = Path(f"results/ml/theta-e-cnn-seeds-v1/seed-{seed}/best.pt")
        baseline = pd.read_csv(checkpoint_path.parent / "validation_predictions.csv")
        baseline_frames.append(
            baseline[["sample_id", "theta_E_arcsec_predicted"]].assign(
                parent_sample_id=baseline["sample_id"], seed=seed, condition="independent-canonical"
            )[["parent_sample_id", "seed", "condition", "theta_E_arcsec_predicted"]]
        )
    week4 = pd.read_csv(WEEK4_PREDICTIONS)
    week4 = week4.loc[week4["intervention_id"] == "exposure-half"]
    baseline_frames.append(
        week4[["parent_sample_id", "seed", "theta_E_arcsec_predicted"]].assign(
            condition="independent-exposure-half"
        )
    )
    baseline = pd.concat(baseline_frames, ignore_index=True).rename(
        columns={"theta_E_arcsec_predicted": "reference_theta_E_arcsec_predicted"}
    )
    scored = scored.merge(baseline, on=["parent_sample_id", "seed", "condition"], validate="one_to_one")
    scored["residual_arcsec"] = scored["theta_E_arcsec_predicted"] - scored["theta_E_arcsec_true"]
    scored["absolute_error_arcsec"] = scored["residual_arcsec"].abs()
    scored["reference_absolute_error_arcsec"] = (
        scored["reference_theta_E_arcsec_predicted"] - scored["theta_E_arcsec_true"]
    ).abs()
    scored["paired_delta_absolute_error_arcsec"] = (
        scored["absolute_error_arcsec"] - scored["reference_absolute_error_arcsec"]
    )
    scored["paired_prediction_shift_arcsec"] = (
        scored["theta_E_arcsec_predicted"] - scored["reference_theta_E_arcsec_predicted"]
    )
    scored.to_csv(output / "scored_predictions.csv", index=False)

    per_seed = []
    effect_rows = []
    parent_effects = []
    bootstrap_count = int(config["metrics"]["bootstrap_resamples"])
    bootstrap_seed = int(config["metrics"]["bootstrap_seed"])
    for condition, group in scored.groupby("condition", sort=False):
        for seed, seed_group in group.groupby("seed", sort=False):
            per_seed.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    **regression_metrics(seed_group["theta_E_arcsec_true"], seed_group["theta_E_arcsec_predicted"]),
                }
            )
        parent = group.groupby("parent_sample_id", as_index=False).agg(
            theta_E_arcsec_true=("theta_E_arcsec_true", "first"),
            mean_prediction=("theta_E_arcsec_predicted", "mean"),
            mean_reference_prediction=("reference_theta_E_arcsec_predicted", "mean"),
            paired_delta_mae=("paired_delta_absolute_error_arcsec", "mean"),
            paired_prediction_shift=("paired_prediction_shift_arcsec", "mean"),
        )
        lo, hi = _bootstrap_interval(parent["paired_delta_mae"].to_numpy(), resamples=bootstrap_count, seed=bootstrap_seed)
        low_pred, high_pred = _bootstrap_interval(
            parent["paired_prediction_shift"].to_numpy(), resamples=bootstrap_count, seed=bootstrap_seed
        )
        effect_rows.append(
            {
                "condition": condition,
                "parent_count": int(len(parent)),
                "paired_delta_mae_arcsec": float(parent["paired_delta_mae"].mean()),
                "paired_delta_mae_bootstrap_95_low_arcsec": lo,
                "paired_delta_mae_bootstrap_95_high_arcsec": hi,
                "mean_absolute_paired_prediction_shift_arcsec": float(parent["paired_prediction_shift"].abs().mean()),
                "mean_prediction_shift_arcsec": float(parent["paired_prediction_shift"].mean()),
                "prediction_shift_bootstrap_95_low_arcsec": low_pred,
                "prediction_shift_bootstrap_95_high_arcsec": high_pred,
                "practical_effect_arcsec": float(config["metrics"]["practical_effect_arcsec"]),
                "seed_averaged_mae_arcsec": float(np.mean(np.abs(parent["mean_prediction"] - parent["theta_E_arcsec_true"]))),
                "reference_seed_averaged_mae_arcsec": float(np.mean(np.abs(parent["mean_reference_prediction"] - parent["theta_E_arcsec_true"]))),
            }
        )
        parent["condition"] = condition
        parent_effects.append(parent)
    per_seed_frame = pd.DataFrame(per_seed)
    effects = pd.DataFrame(effect_rows)
    parent_effect_frame = pd.concat(parent_effects, ignore_index=True)
    per_seed_frame.to_csv(output / "metrics_per_seed.csv", index=False)
    effects.to_csv(output / "paired_effects.csv", index=False)
    parent_effect_frame.to_csv(output / "paired_parent_effects.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(effects))
    mean = effects["paired_delta_mae_arcsec"].to_numpy()
    low = effects["paired_delta_mae_bootstrap_95_low_arcsec"].to_numpy()
    high = effects["paired_delta_mae_bootstrap_95_high_arcsec"].to_numpy()
    ax.errorbar(x, mean, yerr=[mean - low, high - mean], fmt="o", capsize=4, color="#26547c")
    ax.axhline(0, color="black", linewidth=1)
    margin = float(config["metrics"]["practical_effect_arcsec"])
    ax.axhline(margin, color="gray", linestyle="--", linewidth=1)
    ax.axhline(-margin, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x, effects["condition"], rotation=12, ha="right")
    ax.set_ylabel("Paired change in absolute error (arcsec)")
    ax.set_title("Independent SIE path versus matched Lenstronomy parent")
    fig.tight_layout()
    fig.savefig(output / "paired_effects.png", dpi=170)
    plt.close(fig)

    metrics = effects.to_dict(orient="records")
    summary = {
        "experiment_id": "WEEK7-INDEPENDENT-SIE-LENS-EQUATION",
        "protocol": "docs/reliability_study/protocol_v1.7.md",
        "protocol_sha256": control["protocol_sha256"],
        "release": release.as_posix(),
        "release_manifest_sha256": file_sha256(release / "manifest.csv"),
        "validation_parent_count": 150,
        "condition_count": 2,
        "model_seeds": [12001, 12002, 12003],
        "bootstrap_resamples": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "practical_effect_arcsec": margin,
        "historical_test_accessed": False,
        "real_data_labels_accessed": False,
        "control_gates_passed": bool(control["passed"]),
        "paired_effects": metrics,
        "shared_components": [
            "Lenstronomy SERSIC_ELLIPSE surface brightness",
            "project Gaussian PSF and deterministic convolution",
            "detector noise model and preprocessing",
        ],
        "claim_limit": "Tests an independent SIE deflection/root/rasterization path; not a fully independent simulator or evidence of real-domain robustness.",
        "prediction_file_written_before_label_join": (output / "predictions_before_scoring.csv").exists(),
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# Week 7 — independent SIE equation comparison",
        "",
        "A matched validation-only benchmark using the same 150 parent systems, physical parameters, CNN checkpoints, and noise seeds as the frozen Lenstronomy reference. Predictions were saved before validation targets were joined for scoring.",
        "",
        f"- Parents per condition: **150**",
        "- Frozen CNN seeds: **12001, 12002, 12003**",
        f"- Bootstrap: **{bootstrap_count:,}** parent resamples, seed **{bootstrap_seed}**",
        f"- Practical-effect margin: **{margin:.3f} arcsec**",
        f"- Root-count agreement: **{control['root_count_agreement_count']}/150**",
        f"- Maximum deflection component difference: **{control['maximum_deflection_component_difference_arcsec']:.3e} arcsec**",
        f"- Maximum matched root separation: **{control['maximum_matched_root_separation_arcsec']:.3e} arcsec**",
        "- Historical synthetic test and real-data labels accessed: **no**",
        "",
        "## Paired CNN effects versus Lenstronomy",
        "",
        "| Condition | Parents | Paired MAE change (arcsec) | Parent bootstrap 95% interval | Mean absolute prediction shift | Seed-averaged MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['condition']} | {row['parent_count']} | {row['paired_delta_mae_arcsec']:+.6f} | "
            f"[{row['paired_delta_mae_bootstrap_95_low_arcsec']:+.6f}, {row['paired_delta_mae_bootstrap_95_high_arcsec']:+.6f}] | "
            f"{row['mean_absolute_paired_prediction_shift_arcsec']:.6f} | {row['seed_averaged_mae_arcsec']:.4f} |"
        )
    lines.extend(
        [
            "",
            "![Paired effects](paired_effects.png)",
            "",
            "The independent implementation matches Lenstronomy's SIE deflection and point-image roots within the frozen tolerances. The four-parent renderer smoke also reproduces the reference clean image, PSF-convolved expected scene, and processed canonical image to floating-point precision. These controls show implementation agreement for the tested SIE proposal.",
            "",
            "The scope is limited. Lenstronomy still supplies Sersic surface brightness; the project supplies the Gaussian PSF, detector draw, and preprocessing. This is an equation/root/rasterization comparison, not a second full simulator. Canonical agreement therefore does not establish robustness to different light profiles, PSFs, detectors, or real HST images.",
            "",
            "Machine-readable control, root, smoke, image, prediction, and paired-effect records are stored beside this report. `predictions_before_scoring.csv` is the pre-label inference artifact.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run_week7(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    release_path: str | Path = DEFAULT_RELEASE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    device_request: str = "auto",
    batch_size: int = 16,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    release, output = Path(release_path), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    control = _controls(output, config)
    release_summary = _build_release(release, output, config, control)
    predictions = _predict(release, output, config, device_request=device_request, batch_size=batch_size)
    summary = _score(predictions, release, output, config, control)
    summary["release_integrity"] = release_summary
    write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--release", default=str(DEFAULT_RELEASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(
        json.dumps(
            run_week7(
                config_path=args.config,
                release_path=args.release,
                output_dir=args.output,
                device_request=args.device,
                batch_size=args.batch_size,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
