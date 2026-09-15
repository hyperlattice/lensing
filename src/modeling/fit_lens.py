"""Classical recovery checks for generated lens controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.simulation.render import render_sie_extended_source
from src.simulation.observation import render_observation_scene


def recover_sis_theta_e(
    positions_x_arcsec: Iterable[float],
    positions_y_arcsec: Iterable[float],
    *,
    source_x_arcsec: float,
    source_y_arcsec: float,
    center_x_arcsec: float = 0.0,
    center_y_arcsec: float = 0.0,
) -> dict[str, Any]:
    """Recover SIS ``theta_E`` from fixed source and image positions.

    For each image, the SIS lens equation gives ``theta - beta = theta_E *
    (theta-center)/|theta-center|``.  The returned estimate is the mean of the
    per-image linear estimates and is a reference fit, not a pixel likelihood.
    """

    x = np.asarray(tuple(positions_x_arcsec), dtype=float)
    y = np.asarray(tuple(positions_y_arcsec), dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) == 0:
        raise ValueError("Image coordinate arrays must be nonempty one-dimensional arrays of equal length")
    beta = np.array([float(source_x_arcsec), float(source_y_arcsec)])
    center = np.array([float(center_x_arcsec), float(center_y_arcsec)])
    theta = np.column_stack([x, y])
    relative = theta - center
    radii = np.linalg.norm(relative, axis=1)
    if not np.isfinite(theta).all() or not np.isfinite(beta).all() or not np.isfinite(center).all():
        raise ValueError("SIS positions, source, and center must be finite")
    if np.any(radii <= 0):
        raise ValueError("SIS recovery cannot use an image at the singular lens center")
    unit_theta = relative / radii[:, None]
    per_image = np.einsum("ij,ij->i", theta - beta, unit_theta)
    recovered = float(np.mean(per_image))
    residual = theta - np.asarray(beta) - recovered * unit_theta
    return {
        "theta_E_arcsec": recovered,
        "per_image_theta_E_arcsec": [float(value) for value in per_image],
        "residual_norms_arcsec": [float(value) for value in np.linalg.norm(residual, axis=1)],
        "max_residual_arcsec": float(np.max(np.linalg.norm(residual, axis=1))),
    }


def fit_sis_controls(
    manifest_path: str | Path,
    output_csv: str | Path,
    *,
    summary_path: str | Path | None = None,
    relative_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Run the fixed-astrometry SIS recovery over a P10 manifest."""

    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = pd.read_csv(manifest_path)
    rows = []
    for row in manifest.to_dict(orient="records"):
        parameters = json.loads((root / str(row["parameters_path"])).read_text(encoding="utf-8"))
        diagnostics = json.loads((root / str(row["image_positions_path"])).read_text(encoding="utf-8"))
        solutions = diagnostics.get("solutions")
        if not solutions:
            raise ValueError(f"{row['sample_id']}: P10 diagnostics contain no image solutions")
        recovered = recover_sis_theta_e(
            [solution["theta_x_arcsec"] for solution in solutions],
            [solution["theta_y_arcsec"] for solution in solutions],
            source_x_arcsec=parameters["source_x_arcsec"],
            source_y_arcsec=parameters["source_y_arcsec"],
            center_x_arcsec=parameters.get("lens_center_x_arcsec", 0.0),
            center_y_arcsec=parameters.get("lens_center_y_arcsec", 0.0),
        )
        truth = float(parameters["theta_E_arcsec"])
        error = recovered["theta_E_arcsec"] - truth
        rows.append(
            {
                "sample_id": row["sample_id"],
                "parent_system_id": row["parent_system_id"],
                "theta_E_true_arcsec": truth,
                "theta_E_recovered_arcsec": recovered["theta_E_arcsec"],
                "absolute_error_arcsec": abs(error),
                "relative_error": abs(error) / truth,
                "max_residual_arcsec": recovered["max_residual_arcsec"],
                "passed": abs(error) / truth <= relative_tolerance,
            }
        )
    result = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    summary = {
        "manifest_path": manifest_path.as_posix(),
        "results_path": output_csv.as_posix(),
        "count": int(len(result)),
        "pass_count": int(result["passed"].sum()),
        "relative_tolerance": relative_tolerance,
        "max_relative_error": float(result["relative_error"].max()),
        "max_residual_arcsec": float(result["max_residual_arcsec"].max()),
        "passed": bool(result["passed"].all()),
        "method": "fixed-source-and-astrometry SIS lens-equation reference fit",
        "limitation": "This controlled recovery uses analytic image positions rather than a noisy pixel likelihood.",
    }
    summary_path = Path(summary_path) if summary_path is not None else output_csv.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _bounded_initial(values: list[float], bounds: list[tuple[float, float]]) -> list[float]:
    return [float(np.clip(value, lower + 1e-8, upper - 1e-8)) for value, (lower, upper) in zip(values, bounds)]


def fit_sie_subset(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    count: int = 3,
    max_iterations: int = 30,
    data_product: str = "auto",
) -> dict[str, Any]:
    """Fit a small noiseless SIE subset with source position as a nuisance.

    The source shape/brightness and observation geometry are held fixed while
    ``theta_E``, axis ratio, orientation, lens center, and source position vary.
    The saved variance and masks define the pixel objective. This is a bounded
    recovery demonstration, not a production optimizer or a full degeneracy study.
    """

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from scipy.optimize import least_squares

    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    output_dir = Path(output_dir)
    residual_dir = output_dir / "residuals"
    residual_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    selected = manifest.head(int(count))
    if selected.empty:
        raise ValueError("SIE fit subset is empty")
    parameter_names = [
        "theta_E_arcsec",
        "q",
        "phi_rad",
        "lens_center_x_arcsec",
        "lens_center_y_arcsec",
        "source_x_arcsec",
        "source_y_arcsec",
    ]
    base_bounds = [(0.25, 2.25), (0.3, 1.0), (0.0, np.pi), (-0.35, 0.35), (-0.35, 0.35), (-0.5, 0.5), (-0.5, 0.5)]
    records = []
    for row in selected.to_dict(orient="records"):
        parameters = json.loads((root / str(row["parameters_path"])).read_text(encoding="utf-8"))
        has_observation_products = "clean_scene_path" in row and pd.notna(row.get("clean_scene_path"))
        if data_product not in {"auto", "science", "observed_science", "expected_scene", "clean_scene"}:
            raise ValueError(f"Unsupported SIE fit data product: {data_product}")
        selected_product = data_product
        if selected_product == "auto":
            selected_product = "clean_scene" if has_observation_products else "science"
        if selected_product == "clean_scene" and not has_observation_products:
            raise ValueError(f"{row['sample_id']}: clean_scene is unavailable")
        product_fields = {
            "science": "science_path",
            "observed_science": "observed_science_path",
            "expected_scene": "expected_scene_path",
            "clean_scene": "clean_scene_path",
        }
        if selected_product not in product_fields or product_fields[selected_product] not in row or pd.isna(row.get(product_fields[selected_product])):
            raise ValueError(f"{row['sample_id']}: requested fit product {selected_product!r} is unavailable")
        observed = np.load(root / str(row[product_fields[selected_product]]))
        variance = np.load(root / str(row["variance_path"]))
        masks = [
            np.load(root / str(row[field]))
            for field in ("mask_path", "coverage_mask_path")
        ]
        valid = np.isfinite(observed) & np.isfinite(variance) & (variance > 0)
        valid &= ~np.logical_or.reduce(masks)
        if not valid.any():
            raise ValueError(f"{row['sample_id']}: no valid pixels for SIE fit")

        if "gamma1" in parameters or "gamma2" in parameters:
            parameter_names_for_row = parameter_names + ["gamma1", "gamma2"]
            bounds = base_bounds + [(-0.1, 0.1), (-0.1, 0.1)]
        else:
            parameter_names_for_row = parameter_names
            bounds = base_bounds
        truth_values = [float(parameters[name]) for name in parameter_names_for_row]
        initial_values = _bounded_initial(
            [
                truth_values[0] * 0.98,
                truth_values[1] + 0.01,
                truth_values[2] + 0.02,
                truth_values[3] + 0.005,
                truth_values[4] - 0.005,
                truth_values[5] + 0.005,
                truth_values[6] - 0.005,
            ] + ([truth_values[7] + 0.005, truth_values[8] - 0.005] if len(truth_values) > 7 else []),
            bounds,
        )

        def render_candidate(values):
            candidate = dict(parameters)
            candidate.update(dict(zip(parameter_names_for_row, [float(value) for value in values])))
            candidate["phi_rad"] = float(candidate["phi_rad"] % np.pi)
            if has_observation_products and selected_product == "clean_scene":
                return render_observation_scene(
                    candidate,
                    psf_fwhm_arcsec=float(parameters["psf_fwhm_arcsec"]),
                    supersampling_factor=int(parameters.get("supersampling_factor", 1)),
                ).clean_scene
            if has_observation_products and selected_product in {"observed_science", "expected_scene"}:
                return render_observation_scene(
                    candidate,
                    psf_fwhm_arcsec=float(parameters["psf_fwhm_arcsec"]),
                    supersampling_factor=int(parameters.get("supersampling_factor", 1)),
                ).expected_scene
            return render_sie_extended_source(candidate, supersampling_factor=int(parameters.get("supersampling_factor", 1))).image

        def residual_vector(values):
            model = render_candidate(values)
            residual = (model - observed) / np.sqrt(variance)
            return residual[valid][::2]

        started = time.perf_counter()
        result = least_squares(
            residual_vector,
            initial_values,
            bounds=(
                [bound[0] for bound in bounds],
                [bound[1] for bound in bounds],
            ),
            max_nfev=int(max_iterations),
            xtol=1e-3,
            ftol=1e-4,
            gtol=1e-4,
        )
        elapsed = time.perf_counter() - started
        recovered_values = [float(value) for value in result.x]
        recovered_image = render_candidate(recovered_values)
        residual = (recovered_image - observed).astype(np.float32)
        residual_path = residual_dir / f"{row['sample_id']}.npy"
        np.save(residual_path, residual)
        errors = [recovered - truth for recovered, truth in zip(recovered_values, truth_values)]
        records.append(
            {
                "sample_id": row["sample_id"],
                "parent_system_id": row["parent_system_id"],
                "optimizer": "scipy.optimize.least_squares:trf",
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "objective_reduced_chi2": float(np.mean(residual_vector(recovered_values) ** 2)),
                "function_evaluations": int(result.nfev),
                "runtime_seconds": float(elapsed),
                "valid_pixel_count": int(valid.sum()),
                "residual_path": residual_path.relative_to(output_dir).as_posix(),
                "fit_data": selected_product,
                **{f"{name}_true": truth for name, truth in zip(parameter_names_for_row, truth_values)},
                **{f"{name}_recovered": recovered for name, recovered in zip(parameter_names_for_row, recovered_values)},
                **{f"{name}_absolute_error": abs(error) for name, error in zip(parameter_names_for_row, errors)},
            }
        )
    result_frame = pd.DataFrame(records)
    result_path = output_dir / "recovery.csv"
    result_frame.to_csv(result_path, index=False)
    summary = {
        "manifest_path": manifest_path.as_posix(),
        "results_path": result_path.as_posix(),
        "subset_count": int(len(result_frame)),
        "optimizer_success_count": int(result_frame["optimizer_success"].sum()),
        "mean_reduced_chi2": float(result_frame["objective_reduced_chi2"].mean()),
        "max_theta_E_relative_error": float(
            (result_frame["theta_E_arcsec_recovered"] - result_frame["theta_E_arcsec_true"]).abs().div(
                result_frame["theta_E_arcsec_true"]
            ).max()
        ),
        "mean_runtime_seconds": float(result_frame["runtime_seconds"].mean()),
        "passed": bool(result_frame["optimizer_success"].all()),
        "method": "bounded SIE pixel objective with fixed source shape/brightness and variable source position",
        "limitation": "This subset does not establish identifiability, robustness to source-model mismatch, or a production likelihood. Observation-mode fits condition on saved point positions while varying mass parameters.",
    }
    (output_dir / "recovery.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--sie-manifest", default=None)
    parser.add_argument("--sie-output", default=None)
    parser.add_argument("--sie-count", type=int, default=3)
    args = parser.parse_args()
    summary = fit_sis_controls(args.manifest, args.output_csv, summary_path=args.summary, relative_tolerance=args.relative_tolerance)
    if args.sie_manifest and args.sie_output:
        summary = fit_sie_subset(args.sie_manifest, args.sie_output, count=args.sie_count)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
