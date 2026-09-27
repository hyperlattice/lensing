"""Truth-free point-image geometry baseline for the H1 pixel-scale study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

from src.simulation.models import build_lens_model, sie_kwargs


DEFAULT_CONFIG = Path("configs/classical_pixel_scale_baseline_v1.yaml")
DEFAULT_OUTPUT = Path("results/reports/reliability_v1/pixel_scale/classical_geometric")
ALLOWED_VALIDATION_COLUMNS = [
    "sample_id",
    "parent_sample_id",
    "split",
    "intervention_id",
    "pixel_scale",
    "psf_fwhm_arcsec",
    "observed_science_path",
    "variance_path",
    "mask_path",
    "coverage_mask_path",
    "psf_path",
]
SCALE_FILES = {
    "canonical": "canonical",
    "low": "pixel-scale-004",
    "high": "pixel-scale-006",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _estimate_center(image, valid, pixel_scale, *, sigma_arcsec, search_radius_arcsec):
    sigma_pixels = max(float(sigma_arcsec) / float(pixel_scale), 0.5)
    weights = valid.astype(np.float64)
    smoothed_values = gaussian_filter(np.where(valid, image, 0.0), sigma=sigma_pixels)
    smoothed_weights = gaussian_filter(weights, sigma=sigma_pixels)
    smooth = np.divide(
        smoothed_values,
        smoothed_weights,
        out=np.full_like(smoothed_values, -np.inf),
        where=smoothed_weights > 0.5,
    )
    height, width = image.shape
    yy, xx = np.indices(image.shape, dtype=np.float64)
    center_x_pixel = (width - 1) / 2.0
    center_y_pixel = (height - 1) / 2.0
    dx = (xx - center_x_pixel) * pixel_scale
    dy = (yy - center_y_pixel) * pixel_scale
    search = valid & (dx * dx + dy * dy <= float(search_radius_arcsec) ** 2)
    if not search.any():
        return 0.0, 0.0
    smooth[~search] = -np.inf
    y_index, x_index = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    return (
        float((x_index - center_x_pixel) * pixel_scale),
        float((y_index - center_y_pixel) * pixel_scale),
    )


def detect_point_image_peaks(
    image,
    variance,
    invalid_mask,
    coverage_mask,
    psf,
    *,
    pixel_scale_arcsec,
    threshold_snr=5.0,
    center_smoothing_sigma_arcsec=0.15,
    lens_center_search_radius_arcsec=0.35,
    central_exclusion_radius_arcsec=0.25,
    outer_detection_radius_arcsec=2.25,
    minimum_peak_separation_arcsec=0.30,
    maximum_peaks=4,
):
    """Detect PSF-like image peaks using only observed products and geometry."""

    image = np.asarray(image, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    coverage_mask = np.asarray(coverage_mask, dtype=bool)
    if image.ndim != 2 or variance.shape != image.shape:
        raise ValueError("Image and variance must be matching two-dimensional arrays")
    if invalid_mask.shape != image.shape or coverage_mask.shape != image.shape:
        raise ValueError("Masks must have the same shape as the observation")
    if psf.ndim != 2 or not np.isfinite(psf).all() or float(psf.sum()) <= 0:
        raise ValueError("PSF must be a finite, positive two-dimensional kernel")
    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
        raise ValueError("pixel_scale_arcsec must be positive and finite")
    valid = (
        np.isfinite(image)
        & np.isfinite(variance)
        & (variance > 0)
        & ~invalid_mask
        & ~coverage_mask
    )
    if not valid.any():
        return {"center_x_arcsec": 0.0, "center_y_arcsec": 0.0, "peaks": [], "snr_map": None}

    scale = float(pixel_scale_arcsec)
    center_x, center_y = _estimate_center(
        image,
        valid,
        scale,
        sigma_arcsec=center_smoothing_sigma_arcsec,
        search_radius_arcsec=lens_center_search_radius_arcsec,
    )
    psf = psf / psf.sum()
    precision = np.zeros_like(variance)
    precision[valid] = 1.0 / variance[valid]
    matched_signal = fftconvolve(np.where(valid, image, 0.0) * precision, psf[::-1, ::-1], mode="same")
    matched_variance = fftconvolve(precision, np.square(psf[::-1, ::-1]), mode="same")
    snr_map = np.divide(
        matched_signal,
        np.sqrt(np.maximum(matched_variance, 0.0)),
        out=np.full_like(matched_signal, -np.inf),
        where=matched_variance > 0,
    )

    height, width = image.shape
    yy, xx = np.indices(image.shape, dtype=np.float64)
    field_x = (xx - (width - 1) / 2.0) * scale
    field_y = (yy - (height - 1) / 2.0) * scale
    dx = field_x - center_x
    dy = field_y - center_y
    radius = np.hypot(dx, dy)
    eligible = (
        valid
        & (radius >= float(central_exclusion_radius_arcsec))
        & (radius <= float(outer_detection_radius_arcsec))
        & (snr_map >= float(threshold_snr))
    )
    max_filter_width = max(3, int(round(float(minimum_peak_separation_arcsec) / scale)))
    if max_filter_width % 2 == 0:
        max_filter_width += 1
    local_maximum = snr_map == maximum_filter(snr_map, size=max_filter_width, mode="constant", cval=-np.inf)
    candidate_y, candidate_x = np.nonzero(eligible & local_maximum)
    candidates = sorted(
        (
            {
                "x_arcsec": float(field_x[y, x]),
                "y_arcsec": float(field_y[y, x]),
                "snr": float(snr_map[y, x]),
            }
            for y, x in zip(candidate_y, candidate_x)
        ),
        key=lambda item: item["snr"],
        reverse=True,
    )
    selected = []
    min_separation = float(minimum_peak_separation_arcsec)
    for candidate in candidates:
        if all(
            np.hypot(candidate["x_arcsec"] - item["x_arcsec"], candidate["y_arcsec"] - item["y_arcsec"])
            >= min_separation
            for item in selected
        ):
            selected.append(candidate)
        if len(selected) >= int(maximum_peaks):
            break
    return {
        "center_x_arcsec": center_x,
        "center_y_arcsec": center_y,
        "peaks": selected,
        "snr_map": snr_map,
    }


def fit_point_image_geometry(
    peaks,
    *,
    center_x_arcsec,
    center_y_arcsec,
    initial_theta_e_arcsec,
    max_function_evaluations=50,
):
    """Fit an SIS for doubles and an SIE for three or four detected images."""

    if len(peaks) < 2:
        return {"success": False, "model": None, "theta_E_arcsec": None, "message": "fewer than two peaks"}
    x = np.asarray([item["x_arcsec"] for item in peaks], dtype=np.float64)
    y = np.asarray([item["y_arcsec"] for item in peaks], dtype=np.float64)
    cx = float(center_x_arcsec)
    cy = float(center_y_arcsec)
    if len(peaks) == 2:
        relative_x = x - cx
        relative_y = y - cy
        radius = np.hypot(relative_x, relative_y)
        if np.any(radius <= 0):
            return {"success": False, "model": "SIS", "theta_E_arcsec": None, "message": "peak at center"}
        ux, uy = relative_x / radius, relative_y / radius
        matrix = np.zeros((2 * len(peaks), 3), dtype=np.float64)
        target = np.zeros(2 * len(peaks), dtype=np.float64)
        matrix[0::2, 0] = 1.0
        matrix[1::2, 1] = 1.0
        matrix[0::2, 2] = ux
        matrix[1::2, 2] = uy
        target[0::2] = x
        target[1::2] = y
        beta_x, beta_y, theta_e = np.linalg.lstsq(matrix, target, rcond=None)[0]
        success = bool(0.25 <= theta_e <= 2.25 and np.isfinite([beta_x, beta_y, theta_e]).all())
        return {
            "success": success,
            "model": "SIS",
            "theta_E_arcsec": float(theta_e) if success else None,
            "beta_x_arcsec": float(beta_x),
            "beta_y_arcsec": float(beta_y),
            "residual_rms_arcsec": float(np.sqrt(np.mean(np.square(matrix @ [beta_x, beta_y, theta_e] - target)))),
            "function_evaluations": 1,
            "initialization_used": False,
            "jacobian_condition_number": None,
            "boundary_hit": bool(success and not (0.25 < theta_e < 2.25)),
            "message": "closed-form bounded SIS image-position fit" if success else "SIS radius outside bounds",
        }

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    lens_model = build_lens_model(["SIE"])

    def residual(values):
        theta_e, axis_ratio, phi, beta_x, beta_y = [float(value) for value in values]
        kwargs = sie_kwargs(
            theta_e,
            axis_ratio,
            phi,
            center_x_arcsec=cx,
            center_y_arcsec=cy,
            backend="lenstronomy",
        )
        alpha_x, alpha_y = lens_model.alpha(x, y, [kwargs])
        return np.concatenate((x - alpha_x - beta_x, y - alpha_y - beta_y))

    result = least_squares(
        residual,
        x0=[float(np.clip(initial_theta_e_arcsec, 0.250001, 2.249999)), 0.8, 0.0, 0.0, 0.0],
        bounds=([0.25, 0.3, 0.0, -0.5, -0.5], [2.25, 1.0, np.pi, 0.5, 0.5]),
        max_nfev=int(max_function_evaluations),
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    values = [float(value) for value in result.x]
    success = bool(result.success and np.isfinite(values).all())
    boundary_hit = bool(
        np.any(np.isclose(result.x, [0.25, 0.3, 0.0, -0.5, -0.5], atol=1e-6))
        or np.any(np.isclose(result.x, [2.25, 1.0, np.pi, 0.5, 0.5], atol=1e-6))
    )
    try:
        jacobian_condition_number = float(np.linalg.cond(result.jac))
    except np.linalg.LinAlgError:
        jacobian_condition_number = float("inf")
    residual_values = residual(result.x)
    return {
        "success": success,
        "model": "SIE",
        "theta_E_arcsec": values[0] if success else None,
        "axis_ratio": values[1],
        "orientation_rad": values[2],
        "beta_x_arcsec": values[3],
        "beta_y_arcsec": values[4],
        "residual_rms_arcsec": float(np.sqrt(np.mean(np.square(residual_values)))),
        "objective_sum_squares_arcsec2": float(np.square(residual_values).sum()),
        "function_evaluations": int(result.nfev),
        "initialization_used": True,
        "jacobian_condition_number": jacobian_condition_number,
        "boundary_hit": boundary_hit,
        "message": str(result.message),
    }


def _load_validation_observations(release, split_file):
    path = Path(release) / "splits" / f"validation__{split_file}.csv"
    frame = pd.read_csv(path, usecols=ALLOWED_VALIDATION_COLUMNS)
    if len(frame) != 150 or frame["parent_sample_id"].nunique() != 150:
        raise ValueError(f"Expected 150 unique validation parents in {path}")
    return path, frame


def _bootstrap_interval(values, *, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), 256):
        count = min(256, int(resamples) - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[indices].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def evaluate_classical_pixel_scale(config_path=DEFAULT_CONFIG, output_dir=DEFAULT_OUTPUT):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = Path(config["training_release"]) / "splits" / "train.csv"
    # The only target-derived fitting input is the explicitly allowed train-set fallback.
    fallback_theta_e = float(
        pd.read_csv(train_manifest, usecols=["theta_E_arcsec"])["theta_E_arcsec"].mean()
    )
    if not np.isfinite(fallback_theta_e) or fallback_theta_e <= 0:
        raise ValueError("Training-mean fallback must be positive and finite")

    release = Path(config["validation_release"])
    all_records = []
    source_hashes = {"training_split_manifest": _sha256(train_manifest)}
    for stratum, intervention_id in config["validation_interventions"].items():
        manifest_path, observations = _load_validation_observations(release, intervention_id)
        source_hashes[f"validation_{stratum}_manifest"] = _sha256(manifest_path)
        for index, row in enumerate(observations.to_dict(orient="records"), start=1):
            root = release
            image = np.load(root / row["observed_science_path"])
            variance = np.load(root / row["variance_path"])
            invalid_mask = np.load(root / row["mask_path"])
            coverage_mask = np.load(root / row["coverage_mask_path"])
            psf = np.load(root / row["psf_path"])
            detection = detect_point_image_peaks(
                image,
                variance,
                invalid_mask,
                coverage_mask,
                psf,
                pixel_scale_arcsec=float(row["pixel_scale"]),
                **config["peak_detection"],
            )
            fit = fit_point_image_geometry(
                detection["peaks"],
                center_x_arcsec=detection["center_x_arcsec"],
                center_y_arcsec=detection["center_y_arcsec"],
                initial_theta_e_arcsec=fallback_theta_e,
                max_function_evaluations=int(
                    config["lens_equation"]["max_function_evaluations"]
                ),
            )
            used_fallback = not fit["success"]
            estimate = fallback_theta_e if used_fallback else float(fit["theta_E_arcsec"])
            all_records.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "parent_sample_id": str(row["parent_sample_id"]),
                    "split": str(row["split"]),
                    "intervention_id": str(row["intervention_id"]),
                    "stratum": stratum,
                    "pixel_scale_arcsec": float(row["pixel_scale"]),
                    "estimated_lens_center_x_arcsec": detection["center_x_arcsec"],
                    "estimated_lens_center_y_arcsec": detection["center_y_arcsec"],
                    "detected_image_count": len(detection["peaks"]),
                    "detected_peak_snr": json.dumps([peak["snr"] for peak in detection["peaks"]]),
                    "geometry_model": fit["model"],
                    "geometry_fit_success": bool(fit["success"]),
                    "used_training_mean_fallback": bool(used_fallback),
                    "fit_message": str(fit["message"]),
                    "fit_function_evaluations": int(fit.get("function_evaluations", 0)),
                    "fit_residual_rms_arcsec": fit.get("residual_rms_arcsec"),
                    "theta_E_arcsec_predicted": estimate,
                }
            )
        print(f"Completed classical detections: {stratum} ({index} parents)", flush=True)

    predictions = pd.DataFrame(all_records)
    predictions_path = output_dir / "predictions_before_scoring.csv"
    predictions.to_csv(predictions_path, index=False)

    # Only after fitting and writing predictions are validation labels loaded for scoring.
    truth_columns = ["sample_id", "parent_sample_id", "intervention_id", "theta_E_arcsec"]
    truth_frames = []
    for intervention_id in config["validation_interventions"].values():
        truth_path = release / "splits" / f"validation__{intervention_id}.csv"
        truth_frames.append(pd.read_csv(truth_path, usecols=truth_columns))
    truth = pd.concat(truth_frames, ignore_index=True).drop_duplicates("sample_id")
    truth = truth.rename(columns={"theta_E_arcsec": "theta_E_arcsec_true"})
    predictions = predictions.merge(
        truth,
        on=["sample_id", "parent_sample_id", "intervention_id"],
        validate="one_to_one",
    )
    predictions["residual_arcsec"] = (
        predictions["theta_E_arcsec_predicted"] - predictions["theta_E_arcsec_true"]
    )
    predictions["absolute_error_arcsec"] = predictions["residual_arcsec"].abs()
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    baseline_path = Path(config["baseline_cnn_predictions"])
    baseline = pd.read_csv(baseline_path)
    baseline = baseline.rename(
        columns={
            "theta_E_arcsec_true": "cnn_theta_E_arcsec_true",
            "absolute_error_arcsec": "cnn_absolute_error_arcsec",
        }
    )
    cnn_parent = baseline.groupby(
        ["parent_sample_id", "intervention_id"], as_index=False
    ).agg(
        cnn_mae_component=("cnn_absolute_error_arcsec", "mean"),
        cnn_seed_count=("checkpoint_id", "nunique"),
    )
    paired = predictions.merge(
        cnn_parent,
        on=["parent_sample_id", "intervention_id"],
        validate="one_to_one",
    )
    paired["classical_minus_cnn_abs_error_arcsec"] = (
        paired["absolute_error_arcsec"] - paired["cnn_mae_component"]
    )
    paired.to_csv(output_dir / "paired_predictions.csv", index=False)

    metrics_rows = []
    for stratum, group in paired.groupby("stratum", sort=True):
        all_error = group["absolute_error_arcsec"].to_numpy(dtype=float)
        detected = group.loc[~group["used_training_mean_fallback"]]
        baseline_cnn = group["cnn_mae_component"].to_numpy(dtype=float)
        effect_by_parent = group.groupby("parent_sample_id")[
            "classical_minus_cnn_abs_error_arcsec"
        ].mean()
        effect_interval = _bootstrap_interval(
            effect_by_parent.to_numpy(),
            resamples=int(config["bootstrap"]["resamples"]),
            seed=int(config["bootstrap"]["seed"]),
        )
        metrics_rows.append(
            {
                "stratum": stratum,
                "intervention_id": str(group["intervention_id"].iloc[0]),
                "parent_count": int(group["parent_sample_id"].nunique()),
                "classical_mae_arcsec_all_parents": float(all_error.mean()),
                "classical_rmse_arcsec_all_parents": float(
                    np.sqrt(np.mean(np.square(group["residual_arcsec"].to_numpy(dtype=float))))
                ),
                "classical_bias_arcsec_all_parents": float(group["residual_arcsec"].mean()),
                "classical_fit_only_mae_arcsec": (
                    float(detected["absolute_error_arcsec"].mean()) if len(detected) else None
                ),
                "detected_and_fit_count": int(len(detected)),
                "fallback_count": int(group["used_training_mean_fallback"].sum()),
                "mean_detected_image_count": float(group["detected_image_count"].mean()),
                "cnn_mae_arcsec_three_seed_parent_average": float(baseline_cnn.mean()),
                "classical_minus_cnn_delta_mae_arcsec": float(effect_by_parent.mean()),
                "classical_minus_cnn_bootstrap_95_low_arcsec": effect_interval[0],
                "classical_minus_cnn_bootstrap_95_high_arcsec": effect_interval[1],
                "cnn_seed_count": int(group["cnn_seed_count"].min()),
            }
        )
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(output_dir / "scale_metrics.csv", index=False)
    classical_pivot = paired.pivot_table(
        index="parent_sample_id",
        columns="stratum",
        values="absolute_error_arcsec",
        aggfunc="first",
    )
    effects = []
    canonical = classical_pivot["canonical"]
    for stratum in ("low", "high"):
        parent_delta = classical_pivot[stratum] - canonical
        interval = _bootstrap_interval(
            parent_delta.to_numpy(),
            resamples=int(config["bootstrap"]["resamples"]),
            seed=int(config["bootstrap"]["seed"]),
        )
        effects.append(
            {
                "comparison": "classical_shifted_minus_canonical_mae",
                "stratum": stratum,
                "intervention_id": str(metrics_frame.loc[metrics_frame["stratum"] == stratum, "intervention_id"].iloc[0]),
                "parent_count": int(len(parent_delta)),
                "mean_delta_mae_arcsec": float(parent_delta.mean()),
                "bootstrap_95_low_arcsec": interval[0],
                "bootstrap_95_high_arcsec": interval[1],
            }
        )
    pd.DataFrame(effects).to_csv(output_dir / "classical_scale_effects.csv", index=False)

    summary = {
        "experiment_id": config["experiment_id"],
        "method": "PSF-matched point-image detection followed by analytic SIS/SIE image-position fitting",
        "validation_parent_count": 150,
        "validation_scales": list(config["validation_interventions"]),
        "allowed_validation_manifest_columns": ALLOWED_VALIDATION_COLUMNS,
        "validation_truth_loaded_only_after_prediction_write": True,
        "parameters_sidecars_or_position_diagnostics_loaded": False,
        "historical_test_split_accessed": False,
        "fallback_theta_E_arcsec_training_mean": fallback_theta_e,
        "source_hashes": source_hashes,
        "prediction_sha256": _sha256(output_dir / "predictions.csv"),
        "cnn_reference_sha256": _sha256(baseline_path),
        "scale_metrics_path": (output_dir / "scale_metrics.csv").as_posix(),
        "paired_predictions_path": (output_dir / "paired_predictions.csv").as_posix(),
        "classical_scale_effects_path": (output_dir / "classical_scale_effects.csv").as_posix(),
        "limitations": [
            "This is an image-derived point-image geometry baseline, not a full pixel-likelihood SIE fit.",
            "The mass center is estimated from central light and the point-image astrometry is approximated with SIS for doubles and SIE for three/four detections.",
            "The fallback is the training-set mean; report fallback frequency and do not interpret fallback rows as successful classical fits.",
            "The released variance map and Gaussian control PSF are synthetic products; real-domain performance is not established.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Week 3 — classical point-image geometry baseline",
        "",
        "Validation-only, image-derived comparator. The fitting code reads no parameter sidecars or saved analytic image-position diagnostics; labels are loaded only after predictions have been written.",
        "",
        "This is a limited point-image SIS/SIE geometry control, not a pixel-likelihood fit or posterior analysis.",
        "",
        "| Scale | Parents | Classical MAE, fallback included | Fit-only MAE | Fit success | CNN MAE | Classical − CNN MAE (95% parent interval) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['stratum']} | {row['parent_count']} | {row['classical_mae_arcsec_all_parents']:.4f} | "
            f"{row['classical_fit_only_mae_arcsec'] if row['classical_fit_only_mae_arcsec'] is not None else float('nan'):.4f} | "
            f"{row['detected_and_fit_count']}/150 | {row['cnn_mae_arcsec_three_seed_parent_average']:.4f} | "
            f"{row['classical_minus_cnn_delta_mae_arcsec']:+.4f} "
            f"[{row['classical_minus_cnn_bootstrap_95_low_arcsec']:+.4f}, {row['classical_minus_cnn_bootstrap_95_high_arcsec']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Paired classical scale effects",
            "",
            "| Shift | Mean MAE change versus canonical | Parent-bootstrap 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for row in effects:
        lines.append(
            f"| {row['stratum']} | {row['mean_delta_mae_arcsec']:+.4f} arcsec | "
            f"[{row['bootstrap_95_low_arcsec']:+.4f}, {row['bootstrap_95_high_arcsec']:+.4f}] |"
        )
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary": summary, "scale_metrics": metrics_rows, "classical_scale_effects": effects}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = evaluate_classical_pixel_scale(args.config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
