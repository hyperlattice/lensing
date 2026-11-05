"""Week 4 truth-free point-image geometry comparison on matched conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.ml.data import file_sha256
from src.modeling.pixel_scale_geometric_baseline import (
    ALLOWED_VALIDATION_COLUMNS,
    detect_point_image_peaks,
    fit_point_image_geometry,
)


DEFAULT_CONFIG = Path("configs/reliability/classical_observation_systematics.yaml")
DEFAULT_OUTPUT = Path("results/reports/reliability_v1/observation_systematics/classical")


def _bootstrap(values, *, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), 256):
        count = min(256, int(resamples) - start)
        index = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[index].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate_classical_observation_systematics(
    *, config_path=DEFAULT_CONFIG, output_dir=DEFAULT_OUTPUT
):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol_path = str(config.get("protocol", "archive/docs/reliability_study/protocol_v1.4.md"))
    release = Path(config["validation_release"])
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    run_manifest = json.loads((release / "run_manifest.json").read_text(encoding="utf-8"))
    if not run_manifest.get("passed") or not run_manifest.get("identity_checks_passed"):
        raise ValueError("Counterfactual release did not pass canonical identity and ancestry gates")
    train_manifest = Path(config["training_release"]) / "splits" / "train.csv"
    fallback = float(pd.read_csv(train_manifest, usecols=["theta_E_arcsec"])["theta_E_arcsec"].mean())
    if not np.isfinite(fallback) or fallback <= 0:
        raise ValueError("Training-split mean fallback is not finite and positive")

    predictions = []
    manifest_hashes = {}
    for stratum, intervention_id in config["validation_interventions"].items():
        split_path = release / "splits" / f"validation__{intervention_id}.csv"
        frame = pd.read_csv(split_path, usecols=ALLOWED_VALIDATION_COLUMNS)
        if len(frame) != 150 or frame["parent_sample_id"].nunique() != 150:
            raise ValueError(f"Expected 150 independent parents in {split_path}")
        if set(frame["split"].astype(str)) != {"validation"}:
            raise ValueError(f"Non-validation records found in {split_path}")
        manifest_hashes[str(intervention_id)] = file_sha256(split_path)
        for row in frame.to_dict(orient="records"):
            image = np.load(release / row["observed_science_path"])
            variance = np.load(release / row["variance_path"])
            invalid = np.load(release / row["mask_path"])
            coverage = np.load(release / row["coverage_mask_path"])
            psf = np.load(release / row["psf_path"])
            detection = detect_point_image_peaks(
                image,
                variance,
                invalid,
                coverage,
                psf,
                pixel_scale_arcsec=float(row["pixel_scale"]),
                **config["peak_detection"],
            )
            fit = fit_point_image_geometry(
                detection["peaks"],
                center_x_arcsec=detection["center_x_arcsec"],
                center_y_arcsec=detection["center_y_arcsec"],
                initial_theta_e_arcsec=fallback,
                max_function_evaluations=int(config["lens_equation"]["max_function_evaluations"]),
            )
            used_fallback = not fit["success"]
            predictions.append({
                "sample_id": str(row["sample_id"]),
                "parent_sample_id": str(row["parent_sample_id"]),
                "split": str(row["split"]),
                "intervention_id": str(row["intervention_id"]),
                "stratum": stratum,
                "detected_image_count": int(len(detection["peaks"])),
                "detected_peak_snr": json.dumps([item["snr"] for item in detection["peaks"]]),
                "geometry_model": fit["model"],
                "geometry_fit_success": bool(fit["success"]),
                "used_training_mean_fallback": bool(used_fallback),
                "fit_message": str(fit["message"]),
                "fit_function_evaluations": int(fit.get("function_evaluations", 0)),
                "fit_residual_rms_arcsec": fit.get("residual_rms_arcsec"),
                "theta_E_arcsec_predicted": fallback if used_fallback else float(fit["theta_E_arcsec"]),
            })
        print(f"Completed classical geometry observations: {intervention_id}", flush=True)

    before = pd.DataFrame(predictions)
    before.to_csv(output / "classical_predictions_before_scoring.csv", index=False)

    # Validation labels are joined only after all observation-only estimates have been saved.
    truth = []
    for intervention_id in config["validation_interventions"].values():
        truth_path = release / "splits" / f"validation__{intervention_id}.csv"
        truth.append(pd.read_csv(truth_path, usecols=["sample_id", "parent_sample_id", "intervention_id", "theta_E_arcsec"]))
    target = pd.concat(truth, ignore_index=True).rename(columns={"theta_E_arcsec": "theta_E_arcsec_true"})
    scored = before.merge(target, on=["sample_id", "parent_sample_id", "intervention_id"], validate="one_to_one")
    scored["residual_arcsec"] = scored["theta_E_arcsec_predicted"] - scored["theta_E_arcsec_true"]
    scored["absolute_error_arcsec"] = scored["residual_arcsec"].abs()
    scored.to_csv(output / "classical_predictions.csv", index=False)

    cnn = pd.read_csv(config["cnn_predictions"])
    cnn_parent = cnn.groupby(["intervention_id", "parent_sample_id"], as_index=False).agg(
        cnn_absolute_error_arcsec=("absolute_error_arcsec", "mean"),
        cnn_prediction_arcsec=("theta_E_arcsec_predicted", "mean"),
        cnn_seed_count=("seed", "nunique"),
    )
    paired = scored.merge(cnn_parent, on=["intervention_id", "parent_sample_id"], validate="one_to_one")
    paired["classical_minus_cnn_abs_error_arcsec"] = paired["absolute_error_arcsec"] - paired["cnn_absolute_error_arcsec"]
    paired["classical_minus_cnn_prediction_arcsec"] = paired["theta_E_arcsec_predicted"] - paired["cnn_prediction_arcsec"]
    paired.to_csv(output / "classical_cnn_paired_predictions.csv", index=False)

    metric_rows = []
    for intervention_id, group in paired.groupby("intervention_id", sort=False):
        err = group["absolute_error_arcsec"].to_numpy(dtype=float)
        effect = group.groupby("parent_sample_id")["classical_minus_cnn_abs_error_arcsec"].mean()
        ci = _bootstrap(effect.to_numpy(), resamples=config["bootstrap"]["resamples"], seed=config["bootstrap"]["seed"])
        metric_rows.append({
            "intervention_id": intervention_id,
            "parent_count": int(group["parent_sample_id"].nunique()),
            "classical_mae_arcsec": float(err.mean()),
            "classical_rmse_arcsec": float(np.sqrt(np.mean(np.square(group["residual_arcsec"].to_numpy(dtype=float))))),
            "classical_bias_arcsec": float(group["residual_arcsec"].mean()),
            "fit_count": int((~group["used_training_mean_fallback"]).sum()),
            "fallback_count": int(group["used_training_mean_fallback"].sum()),
            "fallback_fraction": float(group["used_training_mean_fallback"].mean()),
            "mean_detected_image_count": float(group["detected_image_count"].mean()),
            "cnn_mae_arcsec_seed_average": float(group["cnn_absolute_error_arcsec"].mean()),
            "classical_minus_cnn_delta_mae_arcsec": float(effect.mean()),
            "classical_minus_cnn_bootstrap_95_low_arcsec": ci[0],
            "classical_minus_cnn_bootstrap_95_high_arcsec": ci[1],
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "classical_metrics.csv", index=False)

    pivot = paired.pivot(index="parent_sample_id", columns="intervention_id", values="absolute_error_arcsec")
    canonical = pivot["canonical"]
    effects = []
    for intervention_id in config["validation_interventions"].values():
        if intervention_id == "canonical":
            continue
        delta = pivot[intervention_id] - canonical
        ci = _bootstrap(delta.to_numpy(), resamples=config["bootstrap"]["resamples"], seed=config["bootstrap"]["seed"])
        effects.append({
            "intervention_id": intervention_id,
            "parent_count": int(len(delta)),
            "classical_paired_delta_mae_arcsec": float(delta.mean()),
            "bootstrap_95_low_arcsec": ci[0],
            "bootstrap_95_high_arcsec": ci[1],
        })
    pd.DataFrame(effects).to_csv(output / "classical_paired_effects.csv", index=False)

    summary = {
        "experiment_id": config["experiment_id"],
        "protocol": protocol_path,
        "protocol_sha256": file_sha256(protocol_path),
        "release_manifest_sha256": file_sha256(release / "manifest.csv"),
        "training_split_manifest_sha256": file_sha256(train_manifest),
        "validation_split_manifest_sha256": manifest_hashes,
        "cnn_prediction_sha256": file_sha256(config["cnn_predictions"]),
        "training_mean_fallback_theta_E_arcsec": fallback,
        "fitted_conditions": list(config["validation_interventions"].values()),
        "point_image_approximation": True,
        "full_pixel_likelihood": False,
        "paired_classical_effects": effects,
        "condition_metrics": metric_rows,
        "historical_test_split_accessed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    week_label = "Week 5" if str(config["experiment_id"]).startswith("WEEK5") else "Week 4"
    lines = [
        f"# {week_label} — Classical point-image comparison",
        "",
        "The frozen v1.3 SIS/SIE point-image geometry control was applied to the same validation parents. Fitting used observation products only; validation labels were joined after `classical_predictions_before_scoring.csv` was written.",
        "",
        f"- Independent parents per condition: **150**",
        f"- Training-mean fallback: **{fallback:.4f} arcsec**",
        "- Historical test split accessed: **no**",
        "- Estimator scope: point-image geometry with fallback; not full pixel-likelihood inference.",
        "",
        "## Condition results",
        "",
        "| Condition | Classical MAE | Fitted | Fallbacks | CNN MAE | Classical minus CNN MAE [95% parent interval] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['intervention_id']} | {row['classical_mae_arcsec']:.4f} | {row['fit_count']}/150 | {row['fallback_count']} | "
            f"{row['cnn_mae_arcsec_seed_average']:.4f} | {row['classical_minus_cnn_delta_mae_arcsec']:+.4f} "
            f"[{row['classical_minus_cnn_bootstrap_95_low_arcsec']:+.4f}, {row['classical_minus_cnn_bootstrap_95_high_arcsec']:+.4f}] |"
        )
    lines.extend(["", "Full predictions, paired effects, and fit diagnostics are retained in this directory.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    summary = evaluate_classical_observation_systematics(config_path=args.config, output_dir=args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
