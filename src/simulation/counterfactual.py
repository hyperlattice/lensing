"""Render ancestry-aware, matched counterfactual strong-lens observations."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from src.data.schema import validate_row
from src.data.dataset import LensDataset
from src.processing.preprocess import preprocess_image
from src.processing.preview_dataset import create_previews
from src.simulation.generate import _relative, _write_json, _write_raw_fits
from src.simulation.observation import observation_psf, realize_detector_observation, render_observation_scene


DEFAULT_PARENT_MANIFEST = Path("data/simulations/sim-eng1000-v1/manifest.csv")
DEFAULT_CONFIG = Path("configs/reliability_counterfactual_v1.yaml")
DEFAULT_OUTPUT = Path("data/simulations/reliability-counterfactual-smoke-v1")
BENCHMARK_VERSION = "reliability-counterfactual-v1"

TRUTH_AND_ID_FIELDS = {
    "theta_E_arcsec",
    "mass_model",
    "parameter_seed",
    "parent_system_id",
    "split_group_id",
    "sample_id",
    "simulation_id",
    "config_hash",
}
ALLOWED_PARAMETER_OVERRIDES = {
    "pixel_scale_arcsec",
    "psf_fwhm_arcsec",
    "background_rate",
    "exposure_time",
    "read_noise_electrons",
    "supersampling_factor",
    "shape",
    "source_x_arcsec",
    "source_y_arcsec",
    "source_re_arcsec",
    "source_n",
    "source_q",
    "source_phi_rad",
    "source_amp",
    "point_source_amp",
    "lens_light_amp",
    "lens_light_re_arcsec",
    "lens_light_n",
    "lens_light_q",
    "lens_light_phi_rad",
    "gamma1",
    "gamma2",
}
ALLOWED_PREPROCESS_OVERRIDES = {
    "normalization_method",
    "lower_percentile",
    "upper_percentile",
    "background_sigma_clip",
    "background_max_iterations",
    "background_min_valid_pixels",
    "noise_floor",
}
RECOMPUTE_POINT_IMAGES_FIELDS = {"source_x_arcsec", "source_y_arcsec", "gamma1", "gamma2"}
ANCESTRY_FIELDS = (
    "parent_system_id",
    "split_group_id",
    "source_template_id",
    "background_template_id",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    if not value:
        raise ValueError("IDs must contain at least one alphanumeric character")
    return value


def _read_split_assignments(split_manifests: Mapping[str, str | Path]) -> dict[str, str]:
    if "test" in split_manifests:
        raise ValueError("The opened historical test split is prohibited for counterfactual development")
    if set(split_manifests).difference({"train", "validation"}):
        raise ValueError("Only train and validation parent manifests may be used")
    if not split_manifests:
        raise ValueError("At least one development split manifest is required")

    assignments: dict[str, str] = {}
    for split_name, path in split_manifests.items():
        frame = pd.read_csv(path)
        if "sample_id" not in frame:
            raise ValueError(f"{split_name} manifest is missing sample_id")
        sample_ids = frame["sample_id"].astype(str)
        if sample_ids.duplicated().any():
            raise ValueError(f"{split_name} manifest contains duplicate sample IDs")
        for sample_id in sample_ids:
            if sample_id in assignments:
                raise ValueError(f"Parent sample {sample_id} is assigned to multiple splits")
            assignments[sample_id] = split_name
    return assignments


def _resolve_override(current: Any, override: Any, *, field: str) -> Any:
    if isinstance(override, dict):
        if len(override) != 1 or next(iter(override)) not in {"value", "add", "multiply"}:
            raise ValueError(f"{field} override must be a scalar or one of value/add/multiply")
        operation, operand = next(iter(override.items()))
        if operation == "value":
            return operand
        if not isinstance(current, (int, float)) or not isinstance(operand, (int, float)):
            raise ValueError(f"{operation} override for {field} requires numeric values")
        return current + operand if operation == "add" else current * operand
    return override


def apply_intervention(
    canonical_parameters: Mapping[str, Any],
    intervention: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one declared nuisance change without changing the target or ancestry."""

    parameters = copy.deepcopy(dict(canonical_parameters))
    overrides = intervention.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("intervention overrides must be a mapping")
    forbidden = TRUTH_AND_ID_FIELDS.intersection(overrides)
    if forbidden:
        raise ValueError(f"Intervention may not change protected truth/identity fields: {sorted(forbidden)}")
    unknown = set(overrides).difference(ALLOWED_PARAMETER_OVERRIDES)
    if unknown:
        raise ValueError(f"Unsupported intervention fields: {sorted(unknown)}")

    applied: dict[str, Any] = {}
    for field, value in overrides.items():
        if field not in parameters and field not in {"gamma1", "gamma2"}:
            raise ValueError(f"Canonical state has no field {field!r}")
        resolved = _resolve_override(parameters.get(field), value, field=field)
        if field == "shape":
            if not isinstance(resolved, (list, tuple)) or len(resolved) != 2:
                raise ValueError("shape override must contain two dimensions")
            resolved = [int(item) for item in resolved]
            if min(resolved) <= 0:
                raise ValueError("shape dimensions must be positive")
        elif isinstance(resolved, (int, float)) and not np.isfinite(resolved):
            raise ValueError(f"{field} override must be finite")
        parameters[field] = resolved
        applied[field] = resolved

    if bool({"gamma1", "gamma2"}.intersection(overrides)):
        parameters.setdefault("gamma1", 0.0)
        parameters.setdefault("gamma2", 0.0)
        if not {"gamma1", "gamma2"}.issubset(parameters):
            raise ValueError("External shear requires both gamma1 and gamma2")
        amplitude = float(np.hypot(parameters["gamma1"], parameters["gamma2"]))
        parameters["gamma_amplitude"] = amplitude
        parameters["gamma_phi_rad"] = (
            float(0.5 * np.arctan2(parameters["gamma2"], parameters["gamma1"]))
            if amplitude > 0
            else 0.0
        )
        parameters["mass_model"] = "SIE+SHEAR" if amplitude > 0 else "SIE"

    if RECOMPUTE_POINT_IMAGES_FIELDS.intersection(overrides):
        for field in (
            "point_positions_x_arcsec",
            "point_positions_y_arcsec",
            "point_amplitudes",
            "point_position_rounding_decimals",
            "point_amplitude_rounding_decimals",
        ):
            parameters.pop(field, None)

    if float(parameters["theta_E_arcsec"]) != float(canonical_parameters["theta_E_arcsec"]):
        raise AssertionError("Intervention changed the primary target")
    return parameters, applied


def validate_counterfactual_ancestry(
    parent_rows: pd.DataFrame,
    counterfactual_rows: pd.DataFrame,
    split_by_parent_sample: Mapping[str, str],
) -> dict[str, Any]:
    """Validate target invariance and split isolation for all generated descendants."""

    if counterfactual_rows.empty:
        raise ValueError("Counterfactual manifest is empty")
    if counterfactual_rows["sample_id"].duplicated().any():
        raise ValueError("Counterfactual manifest contains duplicate sample IDs")
    parent_by_sample = parent_rows.set_index(parent_rows["sample_id"].astype(str), drop=False)
    errors: list[str] = []
    combined: list[dict[str, Any]] = []
    for row in counterfactual_rows.to_dict(orient="records"):
        parent_sample_id = str(row["parent_sample_id"])
        if parent_sample_id not in parent_by_sample.index:
            errors.append(f"{row['sample_id']}: unknown parent sample {parent_sample_id}")
            continue
        parent = parent_by_sample.loc[parent_sample_id]
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]
        if float(row["theta_E_arcsec"]) != float(parent["theta_E_arcsec"]):
            errors.append(f"{row['sample_id']}: theta_E_arcsec changed")
        expected_split = split_by_parent_sample.get(parent_sample_id)
        if expected_split not in {"train", "validation"}:
            errors.append(f"{row['sample_id']}: parent is not in an allowed development split")
        elif row.get("split") != expected_split:
            errors.append(f"{row['sample_id']}: descendant split differs from parent")
        combined.append({**parent.to_dict(), "split": expected_split})
        combined.append(row)

    crossings: dict[str, list[str]] = {}
    if combined:
        joined = pd.DataFrame(combined)
        for field in ANCESTRY_FIELDS:
            if field not in joined:
                continue
            usable = joined.loc[joined[field].notna() & (joined[field].astype(str) != ""), [field, "split"]]
            crossing = usable.groupby(field)["split"].nunique()
            offenders = crossing[crossing > 1].index.astype(str).tolist()
            if offenders:
                crossings[field] = offenders
                errors.append(f"{field} crosses splits: {offenders[:5]}")

    report = {
        "parent_count": int(counterfactual_rows["parent_sample_id"].nunique()),
        "descendant_count": int(len(counterfactual_rows)),
        "theta_E_invariant": not any("theta_E_arcsec changed" in item for item in errors),
        "split_isolation_passed": not crossings,
        "ancestry_crossings": crossings,
        "errors": errors,
        "passed": not errors,
    }
    if errors:
        raise ValueError("Counterfactual ancestry validation failed: " + "; ".join(errors))
    return report


def _load_render_config(parameters: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    config_path = Path(str(parameters["config_path"]))
    if not config_path.exists():
        raise FileNotFoundError(f"Parent simulation config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return config, config_path


def _render_parent(
    parameters: dict[str, Any],
    *,
    sample_id: str,
    parent_id: str,
    intervention: Mapping[str, Any],
    output_root: Path,
    identity_source_row: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, _ = _load_render_config(parameters)
    preprocessing = dict(config["preprocessing"])
    preprocess_overrides = intervention.get("preprocessing", {})
    if not isinstance(preprocess_overrides, Mapping):
        raise ValueError("preprocessing overrides must be a mapping")
    unknown_preprocessing = set(preprocess_overrides).difference(ALLOWED_PREPROCESS_OVERRIDES)
    if unknown_preprocessing:
        raise ValueError(f"Unsupported preprocessing fields: {sorted(unknown_preprocessing)}")
    preprocessing.update(preprocess_overrides)

    noise_seed_offset = int(intervention.get("noise_seed_offset", 0))
    canonical_noise_seed = int(parameters["noise_seed"])
    if noise_seed_offset:
        parameters["noise_seed"] = int(
            np.random.SeedSequence([canonical_noise_seed, noise_seed_offset]).generate_state(1)[0]
        )

    scene = render_observation_scene(
        parameters,
        shape=tuple(parameters.get("shape", config["grid"]["shape"])),
        pixel_scale_arcsec=float(parameters["pixel_scale_arcsec"]),
        psf_fwhm_arcsec=float(parameters["psf_fwhm_arcsec"]),
        supersampling_factor=int(parameters["supersampling_factor"]),
    )
    # Save solved image positions at the same precision used by the original
    # engineering release, then render once more from the frozen roots.
    if not parameters.get("point_positions_x_arcsec"):
        digits = int(parameters.get("point_position_rounding_decimals", 8))
        parameters["point_positions_x_arcsec"] = [float(np.round(value, digits)) for value in scene.point_positions_x_arcsec]
        parameters["point_positions_y_arcsec"] = [float(np.round(value, digits)) for value in scene.point_positions_y_arcsec]
        parameters["point_amplitudes"] = [float(np.round(value, 3)) for value in scene.point_amplitudes]
        parameters["point_position_rounding_decimals"] = digits
        parameters["point_amplitude_rounding_decimals"] = 3
        scene = render_observation_scene(
            parameters,
            shape=tuple(parameters.get("shape", config["grid"]["shape"])),
            pixel_scale_arcsec=float(parameters["pixel_scale_arcsec"]),
            psf_fwhm_arcsec=float(parameters["psf_fwhm_arcsec"]),
            supersampling_factor=int(parameters["supersampling_factor"]),
        )

    detector = realize_detector_observation(
        scene.expected_scene,
        background_rate=float(parameters["background_rate"]),
        exposure_time=float(parameters["exposure_time"]),
        read_noise_electrons=float(parameters["read_noise_electrons"]),
        noise_seed=int(parameters["noise_seed"]),
    )
    paths = {
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
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    _write_raw_fits(
        paths["raw"],
        detector.observed_rate,
        parameters,
        sample_id,
        parent_id,
        simulation_id=BENCHMARK_VERSION,
        model=str(parameters.get("mass_model", "SIE")),
        simulation_type="MATCHED_COUNTERFACTUAL_OBSERVATION",
        image_unit="ELECTRONS/S",
    )
    processed = preprocess_image(
        paths["raw"],
        target_shape=tuple(parameters.get("shape", config["grid"]["shape"])),
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
        "clean_scene": scene.clean_scene,
        "expected_scene": scene.expected_scene,
        "observed_science": detector.science_rate,
    }
    for key, array in arrays.items():
        np.save(paths[key], array)

    psf = observation_psf(
        psf_fwhm_arcsec=scene.psf_fwhm_arcsec,
        pixel_scale_arcsec=scene.pixel_scale_arcsec,
    )
    np.save(paths["psf_values"], psf.values)
    _write_json(paths["psf"], {
        "psf_type": "GAUSSIAN",
        "fwhm_arcsec": scene.psf_fwhm_arcsec,
        "pixel_scale_arcsec": scene.pixel_scale_arcsec,
        "sigma_pixels": scene.psf_fwhm_arcsec / (
            2.0 * np.sqrt(2.0 * np.log(2.0)) * scene.pixel_scale_arcsec
        ),
        "normalization": "unit_sum",
        "source": "analytic_gaussian_control",
        "convolution": "deterministic_zero_boundary",
        "values_path": _relative(paths["psf_values"], output_root),
        "psf_id": psf.psf_id,
        "shape": list(psf.values.shape),
        "sum": float(np.sum(psf.values)),
    })
    _write_json(paths["psf_metadata"], psf.to_dict(_relative(paths["psf_values"], output_root)))
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
        "noise_seed_offset": noise_seed_offset,
    })

    diagnostics = {
        "sample_id": sample_id,
        "parent_system_id": parent_id,
        "supersampling_factor": scene.supersampling_factor,
        "point_positions_x_arcsec": list(scene.point_positions_x_arcsec),
        "point_positions_y_arcsec": list(scene.point_positions_y_arcsec),
        "point_amplitudes": list(scene.point_amplitudes),
        "image_count_physical": len(scene.point_positions_x_arcsec),
        "image_count_detectable": sum(
            value >= float(parameters.get("detectable_amplitude_threshold", 0.05))
            for value in scene.point_amplitudes
        ),
        "detectable_amplitude_threshold": float(parameters.get("detectable_amplitude_threshold", 0.05)),
        "clean_scene_flux": float(scene.clean_scene.sum()),
        "expected_scene_flux": float(scene.expected_scene.sum()),
        "observed_science_mean": float(detector.science_rate.mean()),
        "expected_variance_mean": float(detector.variance_rate.mean()),
    }
    parameters.update({
        "sample_id": sample_id,
        "parent_system_id": parent_id,
        "simulation_id": BENCHMARK_VERSION,
        "lens_parameters": scene.lens_parameters,
        "shear_parameters": scene.shear_parameters,
        "source_parameters": scene.source_parameters,
        "lens_light_parameters": scene.lens_light_parameters,
        "point_source_parameters": scene.point_source_parameters,
        "point_positions_x_arcsec": list(scene.point_positions_x_arcsec),
        "point_positions_y_arcsec": list(scene.point_positions_y_arcsec),
        "point_amplitudes": list(scene.point_amplitudes),
        "intervention_id": str(intervention["intervention_id"]),
        "intervention_family": str(intervention["family"]),
        "intervention_overrides": dict(intervention.get("overrides", {})),
        "preprocessing_overrides": dict(preprocess_overrides),
        "parent_sample_id": str(intervention.get("parent_sample_id", "")),
        "parent_state_path": str(intervention.get("canonical_state_path", "")),
    })
    _write_json(paths["diagnostics"], diagnostics)
    _write_json(paths["parameters"], parameters)

    identity = {"checked": False, "passed": True, "max_abs_difference": {}}
    if identity_source_row is not None:
        identity["checked"] = True
        comparisons = {
            "expected_scene": (
                np.load(Path(str(identity_source_row["expected_scene_path"]))),
                scene.expected_scene,
            ),
            "observed_rate": (
                np.asarray(fits.getdata(Path(str(identity_source_row["raw_path"]))), dtype=np.float32),
                detector.observed_rate,
            ),
            "normalized": (
                np.load(Path(str(identity_source_row["processed_path"]))),
                processed["normalized"],
            ),
        }
        identity["passed"] = True
        for name, (expected, actual) in comparisons.items():
            difference = float(np.max(np.abs(np.asarray(expected, dtype=np.float64) - np.asarray(actual, dtype=np.float64))))
            identity["max_abs_difference"][name] = difference
            identity["passed"] = identity["passed"] and bool(np.allclose(expected, actual, rtol=0.0, atol=1e-7))
        if not identity["passed"]:
            raise ValueError(f"Canonical rerender did not reproduce parent products: {identity}")
    return paths, {"scene": scene, "detector": detector, "processed": processed, "identity": identity, "diagnostics": diagnostics}


def create_counterfactual_benchmark(
    parent_manifest_path: str | Path,
    split_manifests: Mapping[str, str | Path],
    output_dir: str | Path,
    interventions: list[Mapping[str, Any]],
    *,
    max_parents_per_split: int | Mapping[str, int] | None = None,
    selection_seed: int = 6031,
    create_preview: bool = True,
    require_canonical: bool = True,
) -> dict[str, Any]:
    """Create canonical and intervened rerenders for train/validation parents only."""

    parent_manifest_path = Path(parent_manifest_path)
    source_root = parent_manifest_path.parent
    output_root = Path(output_dir)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    if isinstance(max_parents_per_split, Mapping):
        invalid_limits = {
            split: value
            for split, value in max_parents_per_split.items()
            if split not in {"train", "validation"}
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        }
        if invalid_limits:
            raise ValueError(f"Invalid per-split parent limits: {invalid_limits}")
    elif max_parents_per_split is not None and (
        isinstance(max_parents_per_split, bool)
        or not isinstance(max_parents_per_split, int)
        or max_parents_per_split <= 0
    ):
        raise ValueError("max_parents_per_split must be positive, a nonnegative split mapping, or None")
    if not interventions:
        raise ValueError("At least one intervention is required")
    intervention_ids = [_safe_id(str(item.get("intervention_id", ""))) for item in interventions]
    if len(intervention_ids) != len(set(intervention_ids)):
        raise ValueError("intervention_id values must be unique")
    canonical_specs = [
        item for item in interventions
        if item.get("intervention_id") == "canonical"
        and not item.get("overrides")
        and not item.get("preprocessing")
        and int(item.get("noise_seed_offset", 0)) == 0
    ]
    if require_canonical and len(canonical_specs) != 1:
        raise ValueError("Exactly one unchanged canonical intervention is required")
    if not require_canonical and canonical_specs:
        raise ValueError("Canonical rerender is omitted only when require_canonical is false")

    parent_frame = pd.read_csv(parent_manifest_path)
    if "sample_id" not in parent_frame or "parameters_path" not in parent_frame:
        raise ValueError("Parent manifest must include sample_id and parameters_path")
    if parent_frame["sample_id"].duplicated().any():
        raise ValueError("Parent manifest contains duplicate sample IDs")
    split_by_sample = _read_split_assignments(split_manifests)
    unknown_split_samples = set(split_by_sample).difference(parent_frame["sample_id"].astype(str))
    if unknown_split_samples:
        raise ValueError(f"Split manifest references unknown parents: {sorted(unknown_split_samples)[:5]}")

    selected_ids: list[str] = []
    rng = np.random.default_rng(int(selection_seed))
    for split_name in ("train", "validation"):
        candidates = [sample_id for sample_id, split in split_by_sample.items() if split == split_name]
        if not candidates:
            continue
        limit = (
            max_parents_per_split.get(split_name, len(candidates))
            if isinstance(max_parents_per_split, Mapping)
            else max_parents_per_split
        )
        if limit == 0:
            continue
        if limit is not None and len(candidates) > limit:
            choices = np.sort(rng.choice(len(candidates), size=limit, replace=False))
            candidates = [candidates[int(index)] for index in choices]
        selected_ids.extend(candidates)
    if not selected_ids:
        raise ValueError("No train/validation parents were selected")

    output_root.mkdir(parents=True, exist_ok=True)
    selected = parent_frame[parent_frame["sample_id"].astype(str).isin(selected_ids)].copy()
    parent_lookup = {str(row["sample_id"]): row for row in selected.to_dict(orient="records")}
    canonical_state_paths: dict[str, str] = {}
    descendants: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    identity_checks: list[dict[str, Any]] = []

    for parent_sample_id in selected_ids:
        source_row = parent_lookup[parent_sample_id]
        parameter_source = Path(str(source_row["parameters_path"]))
        if not parameter_source.is_absolute():
            parameter_source = source_root / parameter_source
        canonical_parameters = json.loads(parameter_source.read_text(encoding="utf-8"))
        if not np.isclose(
            float(canonical_parameters["theta_E_arcsec"]),
            float(source_row["theta_E_arcsec"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{parent_sample_id}: manifest and parameter-sidecar target disagree")

        state_path = output_root / "canonical_states" / f"{_safe_id(parent_sample_id)}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(state_path, {
            "schema_version": 1,
            "parent_sample_id": parent_sample_id,
            "parent_system_id": str(source_row["parent_system_id"]),
            "split": split_by_sample[parent_sample_id],
            "theta_E_arcsec": float(source_row["theta_E_arcsec"]),
            "source_manifest_sha256": _sha256(parent_manifest_path),
            "source_parameter_sha256": _sha256(parameter_source),
            "parameters": canonical_parameters,
        })
        state_relative_path = _relative(state_path, output_root)
        canonical_state_paths[parent_sample_id] = state_relative_path

        for spec, intervention_id in zip(interventions, intervention_ids):
            family = _safe_id(str(spec.get("family", "")))
            sample_id = f"{_safe_id(parent_sample_id)}__{intervention_id}"
            try:
                parameters, _ = apply_intervention(canonical_parameters, spec)
                parameters["sample_id"] = sample_id
                parameters["parent_system_id"] = str(source_row["parent_system_id"])
                render_spec = dict(spec)
                render_spec["parent_sample_id"] = parent_sample_id
                render_spec["canonical_state_path"] = state_relative_path
                identity_parent = None
                if intervention_id == "canonical":
                    identity_parent = {
                        **source_row,
                        "raw_path": str(source_root / str(source_row["raw_path"])),
                        "expected_scene_path": str(source_root / str(source_row["expected_scene_path"])),
                        "processed_path": str(source_root / str(source_row["processed_path"])),
                    }
                paths, products = _render_parent(
                    parameters,
                    sample_id=sample_id,
                    parent_id=str(source_row["parent_system_id"]),
                    intervention=render_spec,
                    output_root=output_root,
                    identity_source_row=identity_parent,
                )
                if products["identity"]["checked"]:
                    identity_checks.append({"sample_id": sample_id, **products["identity"]})

                updated = dict(source_row)
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
                for field, key in path_fields.items():
                    updated[field] = _relative(paths[key], output_root)
                updated.update({
                    "sample_id": sample_id,
                    "lens_id": str(source_row["parent_system_id"]),
                    "dataset_version": BENCHMARK_VERSION,
                    "simulation_id": BENCHMARK_VERSION,
                    "parent_sample_id": parent_sample_id,
                    "canonical_state_path": state_relative_path,
                    "split": split_by_sample[parent_sample_id],
                    "intervention_id": intervention_id,
                    "intervention_family": family,
                    "original_parent_sample_id": parent_sample_id,
                    "theta_E_arcsec": float(source_row["theta_E_arcsec"]),
                    "pixel_scale": float(parameters["pixel_scale_arcsec"]),
                    "pixel_scale_x": float(parameters["pixel_scale_arcsec"]),
                    "pixel_scale_y": float(parameters["pixel_scale_arcsec"]),
                    "psf_fwhm_arcsec": float(parameters["psf_fwhm_arcsec"]),
                    "psf_id": "observation-gaussian-control",
                    "exposure_time": float(parameters["exposure_time"]),
                    "background_rate": float(parameters["background_rate"]),
                    "read_noise_electrons": float(parameters["read_noise_electrons"]),
                    "noise_seed": int(parameters["noise_seed"]),
                    "source_x_arcsec": float(parameters["source_x_arcsec"]),
                    "source_y_arcsec": float(parameters["source_y_arcsec"]),
                    "source_re_arcsec": float(parameters["source_re_arcsec"]),
                    "source_n": float(parameters["source_n"]),
                    "source_q": float(parameters["source_q"]),
                    "source_phi_rad": float(parameters["source_phi_rad"]),
                    "source_amp": float(parameters["source_amp"]),
                    "point_source_amp": float(parameters["point_source_amp"]),
                    "lens_light_amp": float(parameters["lens_light_amp"]),
                    "lens_light_re_arcsec": float(parameters["lens_light_re_arcsec"]),
                    "lens_light_n": float(parameters["lens_light_n"]),
                    "lens_light_q": float(parameters["lens_light_q"]),
                    "lens_light_phi_rad": float(parameters["lens_light_phi_rad"]),
                    "supersampling_factor": int(parameters["supersampling_factor"]),
                    "original_shape": "x".join(str(value) for value in products["scene"].expected_scene.shape),
                    "processed_shape": "x".join(str(value) for value in products["processed"]["normalized"].shape),
                    "normalization_method": products["processed"]["normalization"].method,
                    "masked_fraction": float(products["processed"]["background_mask"].mean()),
                    "image_count_physical": int(products["diagnostics"]["image_count_physical"]),
                    "image_count_detectable": int(products["diagnostics"]["image_count_detectable"]),
                    "theta_E_target_available": True,
                    "e1": products["scene"].lens_parameters["e1"],
                    "e2": products["scene"].lens_parameters["e2"],
                    "gamma1": parameters.get("gamma1"),
                    "gamma2": parameters.get("gamma2"),
                    "gamma_amplitude": parameters.get("gamma_amplitude"),
                    "gamma_phi_rad": parameters.get("gamma_phi_rad"),
                    "mass_model": parameters.get("mass_model", "SIE"),
                })
                errors = validate_row(updated)
                if errors:
                    raise ValueError(f"Generated manifest row failed schema validation: {errors}")
                descendants.append(updated)
            except Exception as error:
                rejections.append({
                    "sample_id": sample_id,
                    "parent_sample_id": parent_sample_id,
                    "parent_system_id": str(source_row["parent_system_id"]),
                    "split": split_by_sample[parent_sample_id],
                    "intervention_id": intervention_id,
                    "family": family,
                    "error_type": type(error).__name__,
                    "message": str(error),
                })

    benchmark_manifest = pd.DataFrame(descendants)
    if benchmark_manifest.empty:
        raise RuntimeError(f"No counterfactuals rendered; see rejections: {rejections[:3]}")
    parent_subset = selected.copy()
    parent_subset["split"] = parent_subset["sample_id"].astype(str).map(split_by_sample)
    ancestry_report = validate_counterfactual_ancestry(parent_subset, benchmark_manifest, split_by_sample)

    manifest_path = output_root / "manifest.csv"
    benchmark_manifest.to_csv(manifest_path, index=False)
    split_root = output_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation"):
        split_frame = benchmark_manifest.loc[benchmark_manifest["split"] == split_name]
        split_frame.to_csv(split_root / f"{split_name}.csv", index=False)
        for intervention_id, intervention_rows in split_frame.groupby("intervention_id"):
            intervention_rows.to_csv(
                split_root / f"{split_name}__{_safe_id(str(intervention_id))}.csv",
                index=False,
            )
    rejection_path = output_root / "rejection_log.csv"
    pd.DataFrame(
        rejections,
        columns=["sample_id", "parent_sample_id", "parent_system_id", "split", "intervention_id", "family", "error_type", "message"],
    ).to_csv(rejection_path, index=False)

    preview_outputs: dict[str, str] = {}
    if create_preview:
        for family, family_rows in benchmark_manifest.groupby("intervention_family"):
            preview_root = output_root / "previews" / _safe_id(str(family))
            input_manifest = output_root / "preview_inputs" / f"{_safe_id(str(family))}.csv"
            input_manifest.parent.mkdir(parents=True, exist_ok=True)
            preview_frame = family_rows[["sample_id", "raw_path"]].copy()
            preview_frame["raw_path"] = preview_frame["raw_path"].map(lambda path: (output_root / path).resolve().as_posix())
            preview_frame.to_csv(input_manifest, index=False)
            create_previews(
                root=output_root / "raw",
                output=preview_root,
                manifest_path=input_manifest,
                dpi=120,
                contact_sheet=True,
            )
            preview_outputs[str(family)] = preview_root.as_posix()

    rejection_frame = pd.DataFrame(rejections)
    loader_checks: dict[str, Any] = {}
    for split_name in ("train", "validation"):
        dataset = LensDataset(
            manifest_path,
            root=output_root,
            split=split_name,
            strict_products=True,
            domain="simulated",
            eligible_for="ml",
            target_fields={"theta_E_arcsec"},
            expected_shape=None,
        )
        loader_checks[split_name] = {"sample_count": len(dataset), "passed": True}

    summary = {
        "benchmark_version": BENCHMARK_VERSION,
        "parent_manifest_path": parent_manifest_path.as_posix(),
        "parent_manifest_sha256": _sha256(parent_manifest_path),
        "selection_seed": int(selection_seed),
        "selected_parent_count": int(len(selected)),
        "descendant_count": int(len(benchmark_manifest)),
        "rejection_count": int(len(rejections)),
        "intervention_ids": intervention_ids,
        "split_counts": benchmark_manifest.groupby("split").size().astype(int).to_dict(),
        "identity_checks": identity_checks,
        "identity_check_required": bool(require_canonical),
        "identity_checks_passed": (
            all(item["passed"] for item in identity_checks)
            if require_canonical
            else True
        ),
        "ancestry_validation": ancestry_report,
        "strict_loader_validation": loader_checks,
        "preview_outputs": preview_outputs,
        "manifest_path": manifest_path.as_posix(),
        "rejection_log_path": rejection_path.as_posix(),
        "passed": (
            not rejections
            and ancestry_report["passed"]
            and (
                all(item["passed"] for item in identity_checks)
                if require_canonical
                else True
            )
            and all(item["passed"] for item in loader_checks.values())
        ),
    }
    _write_json(output_root / "run_manifest.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_PARENT_MANIFEST))
    parser.add_argument("--splits-dir", default="data/simulations/sim-eng1000-v1/splits")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-parents-per-split", type=int, default=None)
    parser.add_argument("--selection-seed", type=int, default=6031)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split_dir = Path(args.splits_dir)
    split_manifests = config.get(
        "split_manifests",
        {"train": split_dir / "train.csv", "validation": split_dir / "validation.csv"},
    )
    if "max_parents_per_split" in config:
        parent_limit = config["max_parents_per_split"]
    elif args.max_parents_per_split is not None:
        parent_limit = args.max_parents_per_split
    else:
        parent_limit = 2
    summary = create_counterfactual_benchmark(
        config.get("parent_manifest", args.manifest),
        split_manifests,
        args.output,
        config["interventions"],
        max_parents_per_split=parent_limit,
        selection_seed=int(config.get("selection_seed", args.selection_seed)),
        create_preview=not args.no_preview,
        require_canonical=bool(config.get("require_canonical", True)),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
