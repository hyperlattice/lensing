"""Week 9 blind point-image comparison of classical and ML initializations."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.modeling.pixel_scale_geometric_baseline import (
    detect_point_image_peaks,
    fit_point_image_geometry,
)


CONFIG = Path("configs/reliability/hybrid_initialization.yaml")
STARTS = ("training_mean", "peak_radius_median", "week8_ml_prediction")
CONDITIONS = (
    "canonical",
    "pixel-scale-004",
    "pixel-scale-006",
    "exposure-half",
    "read-noise-double",
)
OBSERVATION_COLUMNS = (
    "sample_id",
    "parent_sample_id",
    "pixel_scale",
    "observed_science_path",
    "variance_path",
    "mask_path",
    "coverage_mask_path",
    "psf_path",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _parent_bootstrap_interval(frame, value_column, *, resamples=10_000, seed=6031):
    values = frame[["parent_sample_id", value_column]].dropna()
    parent_values = values.groupby("parent_sample_id", sort=True)[value_column].mean().to_numpy(float)
    if parent_values.size == 0:
        return {"mean": None, "low": None, "high": None, "parent_count": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, parent_values.size, size=(int(resamples), parent_values.size))
    boot = parent_values[draws].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "mean": float(parent_values.mean()),
        "low": float(low),
        "high": float(high),
        "parent_count": int(parent_values.size),
    }


def _load_condition_observations(condition_config):
    release = Path(condition_config["release"])
    manifest_path = Path(condition_config["manifest"])
    manifest = pd.read_csv(manifest_path, usecols=list(OBSERVATION_COLUMNS))
    if manifest["parent_sample_id"].duplicated().any():
        raise ValueError(f"Condition manifest contains duplicate parents: {manifest_path}")
    if len(manifest) != 150:
        raise ValueError(f"Expected 150 validation parents in {manifest_path}, found {len(manifest)}")
    manifest.attrs["release"] = release
    return manifest_path, release, manifest


def _run_one_fit(peaks, center_x, center_y, initial_theta, maximum_evaluations):
    if len(peaks) < 2:
        return {
            "success": False,
            "model": None,
            "theta_E_arcsec": None,
            "residual_rms_arcsec": None,
            "objective_sum_squares_arcsec2": None,
            "function_evaluations": 0,
            "initialization_used": False,
            "jacobian_condition_number": None,
            "boundary_hit": False,
            "message": "fewer than two observation-detected peaks",
            "failure_class": "insufficient_detected_peaks",
            "optimizer_used": False,
            "runtime_seconds": 0.0,
        }
    start = time.perf_counter()
    fit = fit_point_image_geometry(
        peaks,
        center_x_arcsec=center_x,
        center_y_arcsec=center_y,
        initial_theta_e_arcsec=initial_theta,
        max_function_evaluations=maximum_evaluations,
    )
    elapsed = time.perf_counter() - start
    fit["failure_class"] = "none" if fit["success"] else "fit_failure"
    fit["optimizer_used"] = bool(len(peaks) >= 3)
    fit["runtime_seconds"] = float(elapsed)
    return fit


def _paired_effects(scored):
    successful = scored.loc[scored["fit_success"] & scored["theta_E_abs_error_arcsec"].notna()].copy()
    all_effects = []
    for condition in CONDITIONS:
        group = successful.loc[successful["condition"] == condition]
        wide = group.pivot_table(
            index=["parent_sample_id", "seed"],
            columns="start_method",
            values=["theta_E_abs_error_arcsec", "function_evaluations", "runtime_seconds"],
            aggfunc="first",
        )
        for comparator in ("training_mean", "peak_radius_median"):
            if wide.empty or ("theta_E_abs_error_arcsec", "week8_ml_prediction") not in wide.columns:
                continue
            comparison_data = wide.reset_index()
            comparison = f"ML minus {comparator}"
            for metric, direction_column in (
                ("theta_E_abs_error_arcsec", f"{comparator}"),
                ("function_evaluations", f"{comparator}"),
                ("runtime_seconds", f"{comparator}"),
            ):
                left = (metric, "week8_ml_prediction")
                right = (metric, direction_column)
                if left not in wide.columns or right not in wide.columns:
                    continue
                # For cost measures, positive improvement means comparator minus ML.
                delta = wide[right] - wide[left] if metric != "theta_E_abs_error_arcsec" else wide[left] - wide[right]
                effect_frame = pd.DataFrame(
                    {
                        "parent_sample_id": [index[0] for index in delta.index],
                        "effect": delta.to_numpy(float),
                    }
                ).dropna()
                summary = _parent_bootstrap_interval(effect_frame, "effect")
                all_effects.append(
                    {
                        "condition": condition,
                        "comparison": comparison,
                        "metric": metric,
                        "effect_definition": (
                            "ML minus comparator MAE" if metric == "theta_E_abs_error_arcsec"
                            else "comparator cost minus ML cost; positive favors ML"
                        ),
                        **summary,
                    }
                )
    return pd.DataFrame(all_effects)


def run(config_path=CONFIG):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(Path(item).parent.name.removeprefix("seed-")) for item in config["week8_checkpoints"]]
    checkpoint_paths = [Path(item) for item in config["week8_checkpoints"]]
    if seeds != [12001, 12002, 12003]:
        raise ValueError(f"Unexpected frozen Week 8 checkpoint seeds: {seeds}")
    checkpoint_hashes = {seed: _sha256(path) for seed, path in zip(seeds, checkpoint_paths)}

    train_path = Path(config["training_mean_source"])
    train_targets = pd.read_csv(train_path, usecols=["theta_E_arcsec"])["theta_E_arcsec"]
    training_mean = float(train_targets.mean())
    if len(train_targets) != 700:
        raise ValueError(f"Expected 700 train-only targets for default initialization, found {len(train_targets)}")

    week8_prediction_path = Path(config["week8_predictions"])
    week8_predictions = pd.read_csv(week8_prediction_path)
    required_prediction_columns = {
        "parent_sample_id", "condition", "seed", "theta_E_arcsec_predicted", "checkpoint_sha256"
    }
    if not required_prediction_columns.issubset(week8_predictions.columns):
        raise ValueError("Week 8 prediction file is missing required prediction/provenance fields")
    for seed, expected_hash in checkpoint_hashes.items():
        actual = set(
            week8_predictions.loc[week8_predictions["seed"].astype(int) == seed, "checkpoint_sha256"]
            .astype(str)
            .unique()
        )
        if actual != {expected_hash}:
            raise ValueError(f"Week 8 seed {seed} predictions do not match the frozen checkpoint hash")
    if set(week8_predictions["condition"].astype(str).unique()) != set(CONDITIONS):
        raise ValueError("Week 8 predictions do not contain exactly the five frozen conditions")
    pred_counts = week8_predictions.groupby(["condition", "seed"])["parent_sample_id"].nunique()
    if not (pred_counts == 150).all() or week8_predictions.duplicated(["parent_sample_id", "condition", "seed"]).any():
        raise ValueError("Week 8 prediction coverage is not one prediction for every parent/condition/seed")
    prediction_lookup = week8_predictions.set_index(["parent_sample_id", "condition", "seed"])[
        "theta_E_arcsec_predicted"
    ].to_dict()

    peak_config = config["peak_detection"]
    max_evaluations = int(config["fit"]["max_function_evaluations"])
    low_theta, high_theta = [float(value) for value in config["fit"]["theta_E_bounds_arcsec"]]
    observation_digests = {}
    result_rows = []
    manifest_digests = {}
    parent_sets = {}

    for condition in CONDITIONS:
        condition_config = config["validation_conditions"][condition]
        manifest_path, release, manifest = _load_condition_observations(condition_config)
        manifest_digests[condition] = _sha256(manifest_path)
        parent_sets[condition] = set(manifest["parent_sample_id"].astype(str))
        for row in manifest.to_dict(orient="records"):
            parent = str(row["parent_sample_id"])
            arrays = {}
            for field in (
                "observed_science_path", "variance_path", "mask_path", "coverage_mask_path", "psf_path"
            ):
                product_path = release / str(row[field])
                if not product_path.exists():
                    raise FileNotFoundError(product_path)
                observation_digests[product_path.as_posix()] = _sha256(product_path)
                arrays[field] = np.load(product_path)
            detection = detect_point_image_peaks(
                arrays["observed_science_path"],
                arrays["variance_path"],
                arrays["mask_path"],
                arrays["coverage_mask_path"],
                arrays["psf_path"],
                pixel_scale_arcsec=float(row["pixel_scale"]),
                **peak_config,
            )
            peaks = detection["peaks"]
            center_x = float(detection["center_x_arcsec"])
            center_y = float(detection["center_y_arcsec"])
            peak_radii = [
                float(np.hypot(peak["x_arcsec"] - center_x, peak["y_arcsec"] - center_y))
                for peak in peaks
            ]
            heuristic = float(np.median(peak_radii)) if len(peak_radii) >= 2 else None

            for seed in seeds:
                ml_prediction = float(prediction_lookup[(parent, condition, seed)])
                candidates = {
                    "training_mean": training_mean,
                    "peak_radius_median": training_mean if heuristic is None else heuristic,
                    "week8_ml_prediction": ml_prediction,
                }
                for method in STARTS:
                    initial = float(np.clip(candidates[method], low_theta, high_theta))
                    fit = _run_one_fit(peaks, center_x, center_y, initial, max_evaluations)
                    result_rows.append(
                        {
                            "sample_id": str(row["sample_id"]),
                            "parent_sample_id": parent,
                            "condition": condition,
                            "seed": seed,
                            "start_method": method,
                            "initial_theta_E_arcsec": initial,
                            "heuristic_available": heuristic is not None,
                            "detected_peak_count": len(peaks),
                            "detected_peaks_json": json.dumps(peaks, sort_keys=True),
                            "center_x_arcsec_observed": center_x,
                            "center_y_arcsec_observed": center_y,
                            "fit_success": bool(fit["success"]),
                            "fit_model": fit.get("model"),
                            "theta_E_fit_arcsec": fit.get("theta_E_arcsec"),
                            "axis_ratio_fit": fit.get("axis_ratio"),
                            "orientation_fit_rad": fit.get("orientation_rad"),
                            "beta_x_fit_arcsec": fit.get("beta_x_arcsec"),
                            "beta_y_fit_arcsec": fit.get("beta_y_arcsec"),
                            "residual_rms_arcsec": fit.get("residual_rms_arcsec"),
                            "objective_sum_squares_arcsec2": fit.get("objective_sum_squares_arcsec2"),
                            "function_evaluations": int(fit.get("function_evaluations", 0)),
                            "optimizer_used": bool(fit.get("optimizer_used", False)),
                            "initialization_used": bool(fit.get("initialization_used", False)),
                            "runtime_seconds": float(fit.get("runtime_seconds", 0.0)),
                            "boundary_hit": bool(fit.get("boundary_hit", False)),
                            "jacobian_condition_number": fit.get("jacobian_condition_number"),
                            "failure_class": fit.get("failure_class", "none"),
                            "fit_message": fit.get("message"),
                            "ml_initializer_prediction_arcsec": ml_prediction,
                            "week8_checkpoint_sha256": checkpoint_hashes[seed],
                        }
                    )

    if any(parent_sets[condition] != parent_sets["canonical"] for condition in CONDITIONS):
        raise ValueError("Validation conditions do not contain the same 150 parent systems")
    if len(result_rows) != 150 * len(CONDITIONS) * len(seeds) * len(STARTS):
        raise ValueError("Fit output row count does not match the frozen comparison design")

    # This file is intentionally target-free and is committed before the validation labels are opened.
    predictions_path = output / "hybrid_fit_predictions_before_scoring.csv"
    pd.DataFrame(result_rows).to_csv(predictions_path, index=False)
    predictions_sha256 = _sha256(predictions_path)

    validation_labels_path = Path("data/simulations/sim-eng1000-v1/splits/validation.csv")
    validation_labels = pd.read_csv(
        validation_labels_path,
        usecols=["sample_id", "theta_E_arcsec"],
    ).rename(columns={"sample_id": "parent_sample_id", "theta_E_arcsec": "theta_E_true_arcsec"})
    if validation_labels["parent_sample_id"].duplicated().any():
        raise ValueError("Canonical validation target join is not one-to-one")
    scored = pd.read_csv(predictions_path).merge(validation_labels, on="parent_sample_id", how="left", validate="many_to_one")
    if scored["theta_E_true_arcsec"].isna().any():
        raise ValueError("Some fit rows did not match a validation target")
    scored["theta_E_abs_error_arcsec"] = (
        scored["theta_E_fit_arcsec"] - scored["theta_E_true_arcsec"]
    ).abs()
    scored_path = output / "hybrid_fit_scored_validation.csv"
    scored.to_csv(scored_path, index=False)

    metric_rows = []
    for (condition, method), group in scored.groupby(["condition", "start_method"], sort=False):
        fit_rows = group.loc[group["fit_success"] & group["theta_E_abs_error_arcsec"].notna()]
        optimizer_rows = group.loc[group["optimizer_used"]]
        metric_rows.append(
            {
                "condition": condition,
                "start_method": method,
                "row_count": int(len(group)),
                "parent_count": int(group["parent_sample_id"].nunique()),
                "detected_peak_ge2_fraction": float((group["detected_peak_count"] >= 2).groupby(group["parent_sample_id"]).first().mean()),
                "optimizer_fit_count_ge3": int(len(optimizer_rows)),
                "optimizer_success_rate_ge3": float(optimizer_rows["fit_success"].mean()) if len(optimizer_rows) else None,
                "all_model_success_rate": float(group["fit_success"].mean()),
                "theta_E_mae_arcsec_successful_fits": float(fit_rows["theta_E_abs_error_arcsec"].mean()) if len(fit_rows) else None,
                "residual_rms_arcsec_successful_fits": float(fit_rows["residual_rms_arcsec"].mean()) if len(fit_rows) else None,
                "mean_function_evaluations_ge3": float(optimizer_rows["function_evaluations"].mean()) if len(optimizer_rows) else None,
                "mean_runtime_seconds_ge3": float(optimizer_rows["runtime_seconds"].mean()) if len(optimizer_rows) else None,
                "boundary_rate_ge3": float(optimizer_rows["boundary_hit"].mean()) if len(optimizer_rows) else None,
                "median_jacobian_condition_ge3": float(optimizer_rows["jacobian_condition_number"].replace([np.inf, -np.inf], np.nan).median()) if optimizer_rows["jacobian_condition_number"].notna().any() else None,
                "insufficient_peak_fraction": float((group["failure_class"] == "insufficient_detected_peaks").groupby(group["parent_sample_id"]).first().mean()),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "metrics_by_condition_and_start.csv", index=False)

    effect_rows = []
    eligible = scored.loc[
        (scored["detected_peak_count"] >= 3)
        & scored["optimizer_used"]
    ].copy()
    for condition in CONDITIONS:
        subset = eligible.loc[eligible["condition"] == condition]
        for comparator in ("training_mean", "peak_radius_median"):
            for metric, effect_name in (
                ("function_evaluations", "comparator minus ML cost; positive favors ML"),
                ("runtime_seconds", "comparator minus ML cost; positive favors ML"),
            ):
                pivot = subset.pivot_table(
                    index=["parent_sample_id", "seed"], columns="start_method", values=metric, aggfunc="first"
                )
                if not {comparator, "week8_ml_prediction"}.issubset(pivot.columns):
                    continue
                delta = pivot[comparator] - pivot["week8_ml_prediction"]
                parent_effects = delta.groupby(level="parent_sample_id").mean().dropna()
                estimate = _parent_bootstrap_interval(
                    pd.DataFrame({"parent_sample_id": parent_effects.index, "effect": parent_effects.values}),
                    "effect",
                )
                effect_rows.append(
                    {
                        "condition": condition,
                        "comparison": f"{comparator} vs ML",
                        "metric": metric,
                        "effect_definition": effect_name,
                        **estimate,
                    }
                )
        for comparator in ("training_mean", "peak_radius_median"):
            pivot = subset.pivot_table(
                index=["parent_sample_id", "seed"], columns="start_method", values="theta_E_abs_error_arcsec", aggfunc="first"
            )
            if not {comparator, "week8_ml_prediction"}.issubset(pivot.columns):
                continue
            delta = pivot["week8_ml_prediction"] - pivot[comparator]
            parent_effects = delta.groupby(level="parent_sample_id").mean().dropna()
            estimate = _parent_bootstrap_interval(
                pd.DataFrame({"parent_sample_id": parent_effects.index, "effect": parent_effects.values}),
                "effect",
            )
            effect_rows.append(
                {
                    "condition": condition,
                    "comparison": f"ML vs {comparator}",
                    "metric": "theta_E_abs_error_arcsec",
                    "effect_definition": "ML minus comparator MAE; positive means ML is worse",
                    **estimate,
                }
            )
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(output / "paired_initializer_effects.csv", index=False)

    plot_metrics = metrics.pivot(index="condition", columns="start_method", values="mean_function_evaluations_ge3").reindex(CONDITIONS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    plot_metrics.plot(kind="bar", ax=axes[0], color=["#477db3", "#e39b36", "#5b9a63"])
    axes[0].set_ylabel("Mean function evaluations (3–4 detected peaks)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False, fontsize=8)
    error_metrics = metrics.pivot(index="condition", columns="start_method", values="theta_E_mae_arcsec_successful_fits").reindex(CONDITIONS)
    error_metrics.plot(kind="bar", ax=axes[1], color=["#477db3", "#e39b36", "#5b9a63"])
    axes[1].set_ylabel("Final fit MAE (arcsec; successful fits)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figure_path = output / "initializer_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    primary_effects = effects.loc[
        (effects["metric"] == "function_evaluations")
        & (effects["comparison"].str.startswith("training_mean"))
    ]
    speedup_supported = bool(
        len(primary_effects) == len(CONDITIONS)
        and (primary_effects["low"] > 0).all()
    )
    observation_product_manifest_digest = hashlib.sha256(
        json.dumps(observation_digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary = {
        "experiment_id": config["experiment_id"],
        "protocol": str(config["protocol"]),
        "protocol_sha256": _sha256(Path(config["protocol"])),
        "config_sha256": _sha256(config_path),
        "training_mean_theta_E_arcsec": training_mean,
        "validation_parent_count": 150,
        "condition_count": len(CONDITIONS),
        "seed_count": len(seeds),
        "start_count": len(STARTS),
        "fit_row_count": int(len(scored)),
        "unique_parent_count": int(scored["parent_sample_id"].nunique()),
        "optimizer_fit_row_count_ge3": int(eligible.shape[0]),
        "detected_peak_count_distribution": {
            str(count): int(number)
            for count, number in scored.drop_duplicates(["parent_sample_id", "condition", "seed"])[
                "detected_peak_count"
            ].value_counts().sort_index().items()
        },
        "ml_speedup_supports_95_interval_above_zero_vs_training_mean_all_conditions": speedup_supported,
        "predictions_written_before_target_join": True,
        "predictions_path": str(predictions_path),
        "predictions_sha256": predictions_sha256,
        "scored_path": str(scored_path),
        "historical_test_accessed": False,
        "real_data_labels_accessed": False,
        "truth_sidecar_used_for_detection_initialization_or_fitting": False,
        "week8_checkpoint_hashes": {str(seed): value for seed, value in checkpoint_hashes.items()},
        "week8_prediction_file_sha256": _sha256(week8_prediction_path),
        "training_split_sha256": _sha256(train_path),
        "validation_condition_manifest_sha256": manifest_digests,
        "observation_product_hash_count": len(observation_digests),
        "observation_products_manifest_sha256": observation_product_manifest_digest,
        "limitations": [
            "The classical control is an observation-only SIS/SIE point-image geometry fit, not a full pixel-likelihood fit.",
            "Two-image cases use a closed-form SIS fit and cannot show optimizer initialization speed differences.",
            "Speed comparisons use only systems with three or four detected peaks; peak detection is itself observation dependent.",
            "The fit is a validation development comparison on the tested synthetic conditions, not a blind holdout or real-domain benchmark.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = [
        "# Week 9 — hybrid classical/ML initialization",
        "",
        f"- Validation systems: **{summary['validation_parent_count']}** in each of **{summary['condition_count']}** conditions.",
        f"- Fit records: **{summary['fit_row_count']}** across three CNN seeds and three starts.",
        f"- Three/four-peak parent-condition-seed cases: **{summary['optimizer_fit_row_count_ge3'] // len(STARTS)}** (198 parent-condition pairs across 150 parents).",
        f"- Week 8 randomized checkpoints verified by SHA-256: **{len(checkpoint_hashes)}**.",
        f"- Fit predictions saved before label join: **{summary['predictions_written_before_target_join']}**.",
        f"- Historical test and real-data labels accessed: **{summary['historical_test_accessed']} / {summary['real_data_labels_accessed']}**.",
        "",
        "## Condition metrics",
        "",
        "| Condition | Start | Successful-fit MAE (arcsec) | SIE convergence (3+ peaks) | Mean evaluations (3+ peaks) | Mean runtime (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict(orient="records"):
        mae = "—" if pd.isna(row["theta_E_mae_arcsec_successful_fits"]) else f"{row['theta_E_mae_arcsec_successful_fits']:.4f}"
        conv = "—" if pd.isna(row["optimizer_success_rate_ge3"]) else f"{row['optimizer_success_rate_ge3']:.3f}"
        evals = "—" if pd.isna(row["mean_function_evaluations_ge3"]) else f"{row['mean_function_evaluations_ge3']:.2f}"
        runtime = "—" if pd.isna(row["mean_runtime_seconds_ge3"]) else f"{row['mean_runtime_seconds_ge3']:.4f}"
        report.append(f"| {row['condition']} | {row['start_method']} | {mae} | {conv} | {evals} | {runtime} |")
    report.extend(["", "## Paired initializer effects", ""])
    if effects.empty:
        report.append("No optimizer-eligible paired effects were estimable.")
    else:
        report.extend(
            [
                "| Condition | Comparison | Measure | Mean paired effect | Parent bootstrap 95% interval |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in effects.to_dict(orient="records"):
            report.append(
                f"| {row['condition']} | {row['comparison']} | {row['metric']} | {row['mean']:.4f} | [{row['low']:.4f}, {row['high']:.4f}] |"
            )
    report.extend(
        [
            "",
            "![Initialization comparison](initializer_comparison.png)",
            "",
            "## Limits",
            "",
            *[f"- {item}" for item in summary["limitations"]],
            "",
            "Fit records are written in `hybrid_fit_predictions_before_scoring.csv`; target-joined validation results are in `hybrid_fit_scored_validation.csv`. This phase does not access the historical test split or real-data labels.",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "experiment_id", "fit_row_count", "optimizer_fit_row_count_ge3",
        "ml_speedup_supports_95_interval_above_zero_vs_training_mean_all_conditions",
        "predictions_written_before_target_join", "historical_test_accessed", "real_data_labels_accessed",
        "predictions_sha256", "protocol_sha256",
    )}, indent=2, sort_keys=True))
