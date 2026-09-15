"""Validate simulation manifests, products, and analytic control diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

from src.data.schema import validate_row
from src.simulation.observation import realize_detector_observation, render_observation_scene
from src.simulation.render import render_from_parameters, render_sie_extended_source


def validate_run(manifest_path: str | Path, *, residual_tolerance_arcsec: float = 1e-10) -> dict[str, Any]:
    # Avoid a costly first Numba import on the local Python 3.13 environment;
    # validation is not a performance benchmark.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = pd.read_csv(manifest_path)
    errors: list[dict[str, Any]] = []
    if manifest["sample_id"].duplicated().any():
        errors.append({"scope": "manifest", "error": "duplicate sample_id"})
    if manifest["parent_system_id"].duplicated().any():
        errors.append({"scope": "manifest", "error": "P10 parents must be independent"})

    for row in manifest.to_dict(orient="records"):
        sample_id = str(row.get("sample_id"))
        schema_errors = validate_row(row)
        for error in schema_errors:
            errors.append({"sample_id": sample_id, "scope": "schema", "error": error})
        try:
            raw_path = root / str(row["raw_path"])
            science_path = root / str(row["science_path"])
            normalized_path = root / str(row["processed_path"])
            parameters_path = root / str(row["parameters_path"])
            diagnostics_path = root / str(row["image_positions_path"])
            if not raw_path.exists():
                raise FileNotFoundError(raw_path)
            raw = np.asarray(fits.getdata(raw_path))
            science = np.load(science_path)
            normalized = np.load(normalized_path)
            if raw.shape != (256, 256) or science.shape != raw.shape or normalized.shape != raw.shape:
                errors.append({"sample_id": sample_id, "scope": "products", "error": "shape mismatch"})
            if not np.isfinite(raw).all() or not np.isfinite(science).all() or not np.isfinite(normalized).all():
                errors.append({"sample_id": sample_id, "scope": "products", "error": "non-finite array"})
            parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            observation_mode = "point_source_psf" in str(parameters.get("render_mode", ""))
            if observation_mode:
                rerender_scene = render_observation_scene(
                    parameters,
                    supersampling_factor=int(diagnostics.get("supersampling_factor", parameters.get("supersampling_factor", 1))),
                    psf_fwhm_arcsec=float(parameters["psf_fwhm_arcsec"]),
                )
                rerender_detector = realize_detector_observation(
                    rerender_scene.expected_scene,
                    background_rate=float(parameters["background_rate"]),
                    exposure_time=float(parameters["exposure_time"]),
                    read_noise_electrons=float(parameters["read_noise_electrons"]),
                    noise_seed=int(parameters["noise_seed"]),
                )
                rerender_image = rerender_detector.observed_rate
                expected_scene = np.load(root / str(row["expected_scene_path"]))
                expected_variance = np.load(root / str(row["expected_variance_path"]))
                observed_science = np.load(root / str(row["observed_science_path"]))
                if not np.array_equal(expected_scene.astype(np.float32), rerender_scene.expected_scene.astype(np.float32)):
                    errors.append({"sample_id": sample_id, "scope": "reproducibility", "error": "expected scene differs on rerender"})
                if not np.array_equal(expected_variance.astype(np.float32), rerender_detector.variance_rate.astype(np.float32)):
                    errors.append({"sample_id": sample_id, "scope": "reproducibility", "error": "expected variance differs on rerender"})
                if not np.array_equal(observed_science.astype(np.float32), rerender_detector.science_rate.astype(np.float32)):
                    errors.append({"sample_id": sample_id, "scope": "reproducibility", "error": "observed science differs on rerender"})
                expected_flux = float(diagnostics["expected_scene_flux"])
            elif parameters.get("mass_model") == "SIE":
                rerender = render_sie_extended_source(
                    parameters,
                    supersampling_factor=int(diagnostics["supersampling_factor"]),
                )
                rerender_image = rerender.image
            else:
                rerender = render_from_parameters(parameters)
                rerender_image = rerender.image
            if not np.array_equal(raw.astype(np.float32), rerender_image.astype(np.float32)) and not observation_mode:
                errors.append({"sample_id": sample_id, "scope": "reproducibility", "error": "raw array differs on rerender"})
            if observation_mode:
                if not np.array_equal(raw.astype(np.float32), rerender_image.astype(np.float32)):
                    errors.append({"sample_id": sample_id, "scope": "reproducibility", "error": "observed detector image differs on rerender"})
                if len(diagnostics["point_positions_x_arcsec"]) != int(row["image_count_physical"]):
                    errors.append({"sample_id": sample_id, "scope": "multiplicity", "error": "physical point-image count mismatch"})
            elif "solutions" in diagnostics:
                residuals = [float(solution["residual_norm_arcsec"]) for solution in diagnostics["solutions"]]
                if len(residuals) != 2 or max(residuals) > residual_tolerance_arcsec:
                    errors.append({"sample_id": sample_id, "scope": "lens_equation", "error": f"residuals {residuals}"})
                expected_flux = float(diagnostics["expected_lensed_flux"])
            elif not observation_mode:
                expected_flux = float(diagnostics["rendered_flux"])
            if not np.isclose(float(raw.sum()), expected_flux, rtol=0, atol=5e-6):
                if not observation_mode:
                    errors.append({"sample_id": sample_id, "scope": "flux", "error": "finite-grid flux mismatch"})
        except Exception as exc:
            errors.append({"sample_id": sample_id, "scope": "products", "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "manifest_path": manifest_path.as_posix(),
        "count": int(len(manifest)),
        "parent_count": int(manifest["parent_system_id"].nunique()),
        "error_count": len(errors),
        "passed": not errors,
        "residual_tolerance_arcsec": residual_tolerance_arcsec,
        "errors": errors,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output", default=None)
    parser.add_argument("--residual-tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    summary = validate_run(args.manifest, residual_tolerance_arcsec=args.residual_tolerance)
    output = Path(args.output) if args.output else Path(args.manifest).with_name("validation.json")
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
