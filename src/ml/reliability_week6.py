"""Reproducible Week 6 baselines, learning curve, and shortcut interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.ml.data import TorchLensDataset, file_sha256
from src.ml.engine import collect_predictions, make_loader, select_device, write_json
from src.ml.metrics import regression_metrics, target_bin_metrics
from src.ml.model import build_model
from src.ml.provenance import config_sha256
from src.ml.train import train_experiment


ROOT = Path("results/reports/reliability_v1/baselines_week6")
TRAINING_RELEASE = Path("data/simulations/sim-eng1000-v1")
H1_RELEASE = Path("data/simulations/reliability-pixel-scale-v1-final")
W4_RELEASE = Path("data/simulations/reliability-observation-systematics-v1")
W5_RELEASE = Path("data/simulations/reliability-source-physics-v1")
BASELINE_CONFIG = Path("configs/ml_baseline_v1.yaml")
CHECKPOINTS = [
    Path(f"results/ml/theta-e-cnn-seeds-v1/seed-{seed}/best.pt")
    for seed in (12001, 12002, 12003)
]
SELECTION_SEED = 6031
BOOTSTRAP_RESAMPLES = 10000
PRACTICAL_EFFECT = 0.01
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
TARGET_BINS = (0.5, 1.0, 1.5, 2.0)

CONDITIONS = [
    {"condition": "canonical", "release": TRAINING_RELEASE, "split_id": "canonical"},
    {"condition": "pixel_scale_004", "release": H1_RELEASE, "split_id": "pixel-scale-004"},
    {"condition": "pixel_scale_006", "release": H1_RELEASE, "split_id": "pixel-scale-006"},
    {"condition": "exposure_half", "release": W4_RELEASE, "split_id": "exposure-half"},
    {"condition": "read_noise_double", "release": W4_RELEASE, "split_id": "read-noise-double"},
    {"condition": "source_x_plus", "release": W5_RELEASE, "split_id": "source-x-plus"},
    {"condition": "shear_gamma2_005", "release": W5_RELEASE, "split_id": "shear-gamma2-005"},
]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bootstrap_mean(values, *, seed=SELECTION_SEED, resamples=BOOTSTRAP_RESAMPLES):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite nonempty vector")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), 256):
        count = min(256, int(resamples) - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        samples[start : start + count] = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _load_split_rows(release: Path, split_name: str) -> pd.DataFrame:
    split_file = Path(release) / "splits" / f"{split_name}.csv"
    rows = pd.read_csv(split_file)
    parent_column = "parent_sample_id" if "parent_sample_id" in rows.columns else None
    selected = ["sample_id", "processed_path", "pixel_scale"]
    if parent_column is not None:
        selected.append(parent_column)
    rows = rows[selected].copy()
    if parent_column is None:
        rows["parent_sample_id"] = rows["sample_id"].astype(str).str.split("__", n=1).str[0]
    elif parent_column != "parent_sample_id":
        rows = rows.rename(columns={parent_column: "parent_sample_id"})
    if rows["sample_id"].duplicated().any() or rows["parent_sample_id"].duplicated().any():
        raise ValueError(f"{split_file} must contain one row per independent parent")
    return rows


def _image_path(release: Path, relative_path: str) -> Path:
    path = Path(str(relative_path))
    return path if path.is_absolute() else Path(release) / path


def _load_image(release: Path, relative_path: str) -> np.ndarray:
    image = np.asarray(np.load(_image_path(release, relative_path)), dtype=np.float64)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"Invalid processed image: {_image_path(release, relative_path)}")
    return image


def fixed_image_features(image: np.ndarray, pixel_scale_arcsec: float) -> np.ndarray:
    """Return the frozen radial/angular feature vector using image and scale only."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError("Feature extraction requires a finite two-dimensional image")
    scale = float(pixel_scale_arcsec)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("pixel_scale_arcsec must be finite and positive")
    height, width = image.shape
    yy, xx = np.indices(image.shape, dtype=np.float64)
    x_arcsec = (xx - (width - 1) / 2.0) * scale
    y_arcsec = (yy - (height - 1) / 2.0) * scale
    radius = np.hypot(x_arcsec, y_arcsec)
    positive = np.maximum(image, 0.0)
    total = float(positive.sum(dtype=np.float64))
    epsilon = np.finfo(np.float64).eps
    denominator = max(total, epsilon)

    radial_edges = np.linspace(0.0, 3.0, 33)
    radial, _ = np.histogram(radius.ravel(), bins=radial_edges, weights=positive.ravel())
    radial_fractions = radial / denominator

    angle = np.mod(np.arctan2(y_arcsec, x_arcsec), 2.0 * np.pi)
    annulus = (radius >= 0.7) & (radius <= 3.0)
    sectors, _ = np.histogram(
        angle[annulus],
        bins=np.linspace(0.0, 2.0 * np.pi, 9),
        weights=positive[annulus],
    )
    sector_fractions = sectors / denominator

    center_flux = float(positive[radius < 0.7].sum(dtype=np.float64))
    outer_flux = float(positive[(radius >= 0.7) & (radius <= 3.0)].sum(dtype=np.float64))
    centroid_radius = float(
        np.sum(positive * radius, dtype=np.float64) / denominator
    )
    median = float(np.median(image))
    robust_rms = 1.4826 * float(np.median(np.abs(image - median)))
    tail_features = np.asarray(
        [
            np.log1p(total),
            robust_rms,
            float(positive.max(initial=0.0)),
            center_flux / max(outer_flux, epsilon),
            centroid_radius,
        ],
        dtype=np.float64,
    )
    features = np.concatenate((radial_fractions, sector_fractions, tail_features))
    if features.shape != (45,) or not np.isfinite(features).all():
        raise ValueError("Fixed feature vector is invalid")
    return features


def _feature_matrix(release: Path, rows: pd.DataFrame) -> np.ndarray:
    values = [
        fixed_image_features(_load_image(release, row.processed_path), row.pixel_scale)
        for row in rows.itertuples(index=False)
    ]
    return np.vstack(values)


def _fit_ridge(x_train, y_train, x_eval, alpha):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0, ddof=0)
    std = np.where(std > 1e-12, std, 1.0)
    train = (x_train - mean) / std
    evaluation = (x_eval - mean) / std
    design = np.column_stack((np.ones(len(train)), train))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    predicted = np.column_stack((np.ones(len(evaluation)), evaluation)) @ coefficient
    return predicted, {"feature_mean": mean.tolist(), "feature_std": std.tolist(), "coefficients": coefficient.tolist()}


def _select_ridge_alpha(x_train, y_train):
    permutation = np.random.default_rng(SELECTION_SEED).permutation(len(y_train))
    folds = [fold for fold in np.array_split(permutation, 5)]
    records = []
    for alpha in ALPHAS:
        fold_errors = []
        for fold_index, validation_indices in enumerate(folds):
            training_indices = np.concatenate(
                [fold for index, fold in enumerate(folds) if index != fold_index]
            )
            predictions, _ = _fit_ridge(
                x_train[training_indices],
                y_train[training_indices],
                x_train[validation_indices],
                alpha,
            )
            mae = float(np.mean(np.abs(predictions - y_train[validation_indices])))
            fold_errors.append(mae)
            records.append(
                {"alpha": float(alpha), "fold": fold_index + 1, "validation_parent_count": len(validation_indices), "mae_arcsec": mae}
            )
    mean_scores = {
        alpha: float(np.mean([row["mae_arcsec"] for row in records if row["alpha"] == alpha]))
        for alpha in ALPHAS
    }
    selected = min(ALPHAS, key=lambda alpha: (mean_scores[alpha], alpha))
    summary = [{"alpha": float(alpha), "mean_fold_mae_arcsec": mean_scores[alpha]} for alpha in ALPHAS]
    return float(selected), records, summary


def run_ridge_baseline(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _load_split_rows(TRAINING_RELEASE, "train")
    train_labels = pd.read_csv(TRAINING_RELEASE / "splits" / "train.csv", usecols=["sample_id", "theta_E_arcsec"])
    train_rows = train_rows.merge(train_labels, on="sample_id", validate="one_to_one")
    if len(train_rows) != 700:
        raise ValueError(f"Expected 700 independent training parents; got {len(train_rows)}")
    x_train = _feature_matrix(TRAINING_RELEASE, train_rows)
    y_train = train_rows["theta_E_arcsec"].to_numpy(dtype=np.float64)
    alpha, fold_records, alpha_summary = _select_ridge_alpha(x_train, y_train)
    pd.DataFrame(fold_records).to_csv(output_dir / "ridge_alpha_folds.csv", index=False)
    pd.DataFrame(alpha_summary).to_csv(output_dir / "ridge_alpha_summary.csv", index=False)

    before_scoring = []
    for condition in CONDITIONS:
        release = Path(condition["release"])
        split_id = str(condition["split_id"])
        split_name = "validation" if release == TRAINING_RELEASE else f"validation__{split_id}"
        rows = _load_split_rows(release, split_name)
        if len(rows) != 150:
            raise ValueError(f"{condition['condition']} must contain exactly 150 validation parents")
        x_eval = _feature_matrix(release, rows)
        ridge_prediction, _ = _fit_ridge(x_train, y_train, x_eval, alpha)
        for row, predicted in zip(rows.itertuples(index=False), ridge_prediction):
            before_scoring.append(
                {
                    "condition": condition["condition"],
                    "sample_id": str(row.sample_id),
                    "parent_sample_id": str(row.parent_sample_id),
                    "theta_E_arcsec_predicted": float(predicted),
                }
            )
    predictions_before_scoring = pd.DataFrame(before_scoring)
    predictions_before_scoring.to_csv(output_dir / "ridge_predictions_before_scoring.csv", index=False)

    scored = []
    for condition in CONDITIONS:
        release = Path(condition["release"])
        split_id = str(condition["split_id"])
        split_name = "validation" if release == TRAINING_RELEASE else f"validation__{split_id}"
        labels = pd.read_csv(
            release / "splits" / f"{split_name}.csv",
            usecols=["sample_id", "theta_E_arcsec"],
        ).rename(columns={"theta_E_arcsec": "theta_E_arcsec_true"})
        labels = _load_split_rows(release, split_name)[["sample_id", "parent_sample_id"]].merge(
            labels, on="sample_id", validate="one_to_one"
        )
        candidate = predictions_before_scoring.loc[
            predictions_before_scoring["condition"] == condition["condition"]
        ].merge(labels, on=["sample_id", "parent_sample_id"], validate="one_to_one")
        if len(candidate) != 150:
            raise ValueError(f"Scoring join failed for {condition['condition']}")
        candidate["residual_arcsec"] = candidate["theta_E_arcsec_predicted"] - candidate["theta_E_arcsec_true"]
        candidate["absolute_error_arcsec"] = candidate["residual_arcsec"].abs()
        scored.append(candidate)
    predictions = pd.concat(scored, ignore_index=True)
    predictions.to_csv(output_dir / "ridge_predictions.csv", index=False)
    fit_prediction, fit_record = _fit_ridge(x_train, y_train, x_train, alpha)
    (output_dir / "ridge_final_fit.json").write_text(
        json.dumps(
            {
                "feature_names": [f"radial_fraction_{index:02d}" for index in range(32)]
                + [f"angular_fraction_{index:02d}" for index in range(8)]
                + ["log_positive_flux", "robust_image_rms", "max_positive_pixel", "central_to_outer_flux_ratio", "positive_centroid_radius_arcsec"],
                "alpha_selected_by_train_only_cv": alpha,
                "training_parent_count": len(train_rows),
                "training_sample_ids_sha256": _sha256_text("\n".join(train_rows["sample_id"].astype(str))),
                "fit": fit_record,
                "training_fit_metrics": regression_metrics(y_train, fit_prediction),
                "feature_definition": "positive-flux radial fractions; eight angular positive-flux fractions for 0.7-3 arcsec; log1p total positive flux; 1.4826*MAD; maximum positive pixel; central/outer positive-flux ratio; positive-flux centroid radius",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "experiment_id": "WEEK6-FIXED-FEATURE-RIDGE",
        "selected_alpha": alpha,
        "training_parent_count": len(train_rows),
        "validation_parent_count": 150,
        "feature_count": int(x_train.shape[1]),
        "train_only_cv_seed": SELECTION_SEED,
        "fold_count": 5,
        "conditions": {
            name: regression_metrics(group["theta_E_arcsec_true"], group["theta_E_arcsec_predicted"])
            for name, group in predictions.groupby("condition", sort=False)
        },
        "historical_test_accessed": False,
    }
    write_json(output_dir / "ridge_summary.json", summary)
    return summary


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_learning_curve(output_dir: Path, *, device_request="auto") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _load_split_rows(TRAINING_RELEASE, "train")
    if len(train_rows) != 700 or train_rows["parent_sample_id"].nunique() != 700:
        raise ValueError("Learning curve requires the 700 independent train parents")
    permutation = np.random.default_rng(SELECTION_SEED).permutation(len(train_rows))
    selected = train_rows.iloc[permutation].reset_index(drop=True)
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    selected["included_n100"] = selected["selection_rank"] <= 100
    selected["included_n300"] = selected["selection_rank"] <= 300
    selected["included_n700"] = selected["selection_rank"] <= 700
    selection_path = output_dir / "nested_selection_manifest.csv"
    if selection_path.exists():
        prior = pd.read_csv(selection_path)
        if prior["sample_id"].astype(str).tolist() != selected["sample_id"].astype(str).tolist():
            raise ValueError("Existing nested learning-curve selection does not match seed 6031")
    else:
        selected.to_csv(selection_path, index=False)

    base_config = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    all_ids = selected["sample_id"].astype(str).tolist()
    summaries = []
    prediction_frames = []
    for sample_count in (100, 300, 700):
        selected_ids = all_ids[:sample_count]
        for seed in (12001, 12002, 12003):
            run_dir = output_dir / f"n{sample_count}" / f"seed-{seed}"
            summary_path = run_dir / "training_summary.json"
            config = dict(base_config)
            config["experiment_id"] = f"week6-learning-curve-n{sample_count}-seed-{seed}"
            config["seed"] = seed
            config["epochs"] = 20
            config["patience"] = 6
            config["release"] = TRAINING_RELEASE.as_posix()
            config["output"] = run_dir.as_posix()
            if summary_path.exists():
                summary = _load_json(summary_path)
                expected_hash = _sha256_text("\n".join(selected_ids))
                recorded = summary.get("run_controls", {}).get("train_sample_ids_sha256")
                if recorded != expected_hash or int(summary.get("train_samples", -1)) != sample_count:
                    raise ValueError(f"Existing learning-curve run differs from its frozen selection: {run_dir}")
                print(f"Reusing completed learning-curve run n={sample_count}, seed={seed}", flush=True)
            else:
                summary = train_experiment(
                    config,
                    output_override=run_dir,
                    device_request=device_request,
                    train_sample_ids=selected_ids,
                    sample_selection_id=f"nested-n{sample_count}-seed-{SELECTION_SEED}",
                )
                print(
                    f"Completed learning-curve run n={sample_count}, seed={seed}, "
                    f"validation MAE={summary['validation_metrics']['mae_arcsec']:.5f}",
                    flush=True,
                )
            summaries.append(summary)
            validation = pd.read_csv(run_dir / "validation_predictions.csv")
            validation["sample_count"] = sample_count
            validation["seed"] = seed
            validation["parent_sample_id"] = validation["sample_id"].astype(str).str.split("__", n=1).str[0]
            prediction_frames.append(validation)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions.groupby(["sample_count", "seed"]).size().ne(150).any():
        raise ValueError("Learning-curve validation runs do not each contain all 150 fixed parents")
    predictions.to_csv(output_dir / "learning_curve_predictions.csv", index=False)
    run_rows = []
    for summary in summaries:
        run_rows.append(
            {
                "sample_count": int(summary["train_samples"]),
                "seed": int(summary["seed"]),
                "best_epoch": int(summary["best_epoch"]),
                "epochs_completed": int(summary["epochs_completed"]),
                "runtime_seconds": float(summary["runtime_seconds"]),
                "train_mae_arcsec": float(summary["training_metrics"]["mae_arcsec"]),
                "validation_mae_arcsec": float(summary["validation_metrics"]["mae_arcsec"]),
                "validation_rmse_arcsec": float(summary["validation_metrics"]["rmse_arcsec"]),
                "trainable_parameters": int(summary["trainable_parameters"]),
                "checkpoint_sha256": str(summary["checkpoint_sha256"]),
                "sample_ids_sha256": str(summary["run_controls"]["train_sample_ids_sha256"]),
            }
        )
    run_metrics = pd.DataFrame(run_rows).sort_values(["sample_count", "seed"])
    run_metrics.to_csv(output_dir / "learning_curve_run_metrics.csv", index=False)

    per_parent = predictions.groupby(["sample_count", "parent_sample_id", "theta_E_arcsec_true"], as_index=False).agg(
        mean_seed_absolute_error_arcsec=("absolute_error_arcsec", "mean"),
        mean_seed_prediction_arcsec=("theta_E_arcsec_predicted", "mean"),
        seed_count=("seed", "nunique"),
    )
    if per_parent["seed_count"].ne(3).any():
        raise ValueError("Each learning-curve parent must have predictions from all three seeds")
    per_parent.to_csv(output_dir / "learning_curve_parent_metrics.csv", index=False)
    curve_rows = []
    for sample_count, group in per_parent.groupby("sample_count", sort=True):
        low, high = _bootstrap_mean(group["mean_seed_absolute_error_arcsec"].to_numpy())
        seed_group = run_metrics.loc[run_metrics["sample_count"] == sample_count]
        curve_rows.append(
            {
                "sample_count": int(sample_count),
                "parent_count": int(group["parent_sample_id"].nunique()),
                "seed_averaged_mae_arcsec": float(group["mean_seed_absolute_error_arcsec"].mean()),
                "parent_bootstrap_95_low_arcsec": low,
                "parent_bootstrap_95_high_arcsec": high,
                "seed_mean_mae_arcsec": float(seed_group["validation_mae_arcsec"].mean()),
                "seed_sd_mae_arcsec": float(seed_group["validation_mae_arcsec"].std(ddof=1)),
                "train_seed_mean_mae_arcsec": float(seed_group["train_mae_arcsec"].mean()),
                "train_seed_sd_mae_arcsec": float(seed_group["train_mae_arcsec"].std(ddof=1)),
            }
        )
    curve = pd.DataFrame(curve_rows).sort_values("sample_count")
    curve.to_csv(output_dir / "learning_curve_summary.csv", index=False)

    paired_rows = []
    reference = per_parent.loc[per_parent["sample_count"] == 700, ["parent_sample_id", "mean_seed_absolute_error_arcsec"]].rename(
        columns={"mean_seed_absolute_error_arcsec": "n700_mean_seed_absolute_error_arcsec"}
    )
    for sample_count in (100, 300):
        candidate = per_parent.loc[per_parent["sample_count"] == sample_count].merge(
            reference, on="parent_sample_id", validate="one_to_one"
        )
        delta = candidate["mean_seed_absolute_error_arcsec"] - candidate["n700_mean_seed_absolute_error_arcsec"]
        low, high = _bootstrap_mean(delta.to_numpy())
        paired_rows.append(
            {
                "sample_count": sample_count,
                "reference_sample_count": 700,
                "parent_count": int(len(candidate)),
                "paired_delta_mae_arcsec": float(delta.mean()),
                "parent_bootstrap_95_low_arcsec": low,
                "parent_bootstrap_95_high_arcsec": high,
            }
        )
    pd.DataFrame(paired_rows).to_csv(output_dir / "learning_curve_paired_to_n700.csv", index=False)

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for row in run_metrics.itertuples(index=False):
        axis.scatter(row.sample_count, row.validation_mae_arcsec, color="0.65", s=22, zorder=2)
    axis.errorbar(
        curve["sample_count"],
        curve["seed_averaged_mae_arcsec"],
        yerr=np.vstack(
            [
                curve["seed_averaged_mae_arcsec"] - curve["parent_bootstrap_95_low_arcsec"],
                curve["parent_bootstrap_95_high_arcsec"] - curve["seed_averaged_mae_arcsec"],
            ]
        ),
        marker="o",
        capsize=4,
        color="#174a7e",
        label="seed-averaged MAE; parent bootstrap 95% interval",
    )
    axis.set_xscale("log")
    axis.minorticks_off()
    axis.set_xticks([100, 300, 700], labels=["100", "300", "700"])
    axis.set_xlabel("Independent training parents")
    axis.set_ylabel("Validation MAE (arcsec)")
    axis.set_title("Week 6 nested learning curve")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(output_dir / "learning_curve.png", dpi=180)
    plt.close(figure)
    summary = {
        "selection_seed": SELECTION_SEED,
        "nested_sizes": [100, 300, 700],
        "seed_ids": [12001, 12002, 12003],
        "fixed_validation_parent_count": 150,
        "selection_manifest_sha256": file_sha256(selection_path),
        "runs": run_rows,
        "curve": curve.to_dict(orient="records"),
        "paired_to_700": paired_rows,
        "historical_test_accessed": False,
    }
    write_json(output_dir / "learning_curve_summary.json", summary)
    return summary


def _transform_image(image: np.ndarray, *, intervention: str, pixel_scale: float) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if intervention == "canonical":
        return image.copy()
    if intervention == "rotate_ccw_90":
        return np.rot90(image, k=1).copy()
    if intervention == "brightness_half":
        return image * np.float32(0.5)
    if intervention == "brightness_double":
        return image * np.float32(2.0)
    height, width = image.shape
    yy, xx = np.indices(image.shape, dtype=np.float64)
    radius = np.hypot((xx - (width - 1) / 2.0) * pixel_scale, (yy - (height - 1) / 2.0) * pixel_scale)
    if intervention == "central_disk_zero":
        result = image.copy()
        result[radius < 0.50] = 0.0
        return result
    if intervention == "arc_annulus_zero":
        result = image.copy()
        result[(radius >= 0.70) & (radius <= 2.20)] = 0.0
        return result
    raise ValueError(f"Unknown image intervention: {intervention}")


def run_shortcut_audit(output_dir: Path, *, device_request="auto") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_split_rows(TRAINING_RELEASE, "validation")
    if len(rows) != 150:
        raise ValueError("Shortcut audit requires the fixed 150 canonical validation parents")
    interventions = (
        "canonical",
        "rotate_ccw_90",
        "brightness_half",
        "brightness_double",
        "central_disk_zero",
        "arc_annulus_zero",
    )
    device = select_device(device_request)
    records = []
    checkpoint_hashes = {}
    for checkpoint_path in CHECKPOINTS:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        source_manifest = Path(checkpoint["release"]) / "manifest.csv"
        if file_sha256(source_manifest) != checkpoint["release_manifest_sha256"]:
            raise ValueError(f"Frozen training release changed for {checkpoint_path}")
        model = build_model(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        seed = int(checkpoint["seed"])
        checkpoint_sha = file_sha256(checkpoint_path)
        checkpoint_hashes[str(seed)] = checkpoint_sha
        image_mean = float(checkpoint["statistics"]["image_mean"])
        image_std = float(checkpoint["statistics"]["image_std"])
        target_mean = float(checkpoint["statistics"]["target_mean"])
        target_std = float(checkpoint["statistics"]["target_std"])
        for intervention in interventions:
            transformed = [
                _transform_image(
                    _load_image(TRAINING_RELEASE, row.processed_path),
                    intervention=intervention,
                    pixel_scale=float(row.pixel_scale),
                )
                for row in rows.itertuples(index=False)
            ]
            prediction_values = []
            with torch.inference_mode():
                for start in range(0, len(transformed), 16):
                    batch = np.stack(transformed[start : start + 16]).astype(np.float32)
                    batch = (batch - image_mean) / image_std
                    tensor = torch.from_numpy(batch).unsqueeze(1).to(device)
                    normalized = model(tensor).detach().cpu().numpy().astype(np.float64)
                    prediction_values.extend((normalized * target_std + target_mean).tolist())
            for row, prediction in zip(rows.itertuples(index=False), prediction_values):
                records.append(
                    {
                        "sample_id": str(row.sample_id),
                        "parent_sample_id": str(row.parent_sample_id),
                        "intervention_id": intervention,
                        "seed": seed,
                        "checkpoint_sha256": checkpoint_sha,
                        "theta_E_arcsec_predicted": float(prediction),
                    }
                )
            print(f"Completed shortcut predictions: {intervention}, seed {seed}", flush=True)

    before_scoring = pd.DataFrame(records)
    if len(before_scoring) != 150 * len(interventions) * 3:
        raise ValueError("Shortcut predictions are incomplete")
    before_scoring.to_csv(output_dir / "shortcut_predictions_before_scoring.csv", index=False)
    baseline_frames = []
    for checkpoint_path in CHECKPOINTS:
        seed = int(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["seed"])
        baseline = pd.read_csv(checkpoint_path.parent / "validation_predictions.csv")
        baseline["seed"] = seed
        baseline_frames.append(baseline[["sample_id", "theta_E_arcsec_predicted", "seed"]])
    frozen_baseline = pd.concat(baseline_frames, ignore_index=True).rename(
        columns={"theta_E_arcsec_predicted": "frozen_theta_E_arcsec_predicted"}
    )
    canonical_check = before_scoring.loc[before_scoring["intervention_id"] == "canonical"].merge(
        frozen_baseline, on=["sample_id", "seed"], validate="one_to_one"
    )
    max_difference = float(
        np.max(np.abs(canonical_check["theta_E_arcsec_predicted"] - canonical_check["frozen_theta_E_arcsec_predicted"]))
    )
    if max_difference > 1e-6:
        raise ValueError(f"Canonical intervention does not reproduce the frozen CNN predictions ({max_difference})")

    labels = pd.read_csv(
        TRAINING_RELEASE / "splits" / "validation.csv",
        usecols=["sample_id", "theta_E_arcsec"],
    ).rename(columns={"theta_E_arcsec": "theta_E_arcsec_true"})
    predictions = before_scoring.merge(labels, on="sample_id", validate="many_to_one")
    predictions["residual_arcsec"] = predictions["theta_E_arcsec_predicted"] - predictions["theta_E_arcsec_true"]
    predictions["absolute_error_arcsec"] = predictions["residual_arcsec"].abs()
    predictions.to_csv(output_dir / "shortcut_predictions.csv", index=False)

    per_seed_metrics = []
    for (intervention, seed), group in predictions.groupby(["intervention_id", "seed"], sort=False):
        per_seed_metrics.append(
            {"intervention_id": intervention, "seed": int(seed), **regression_metrics(group["theta_E_arcsec_true"], group["theta_E_arcsec_predicted"])}
        )
    pd.DataFrame(per_seed_metrics).to_csv(output_dir / "shortcut_metrics_per_seed.csv", index=False)
    summary_metrics = []
    for intervention, group in predictions.groupby("intervention_id", sort=False):
        parent = group.groupby(["parent_sample_id", "theta_E_arcsec_true"], as_index=False).agg(
            prediction=("theta_E_arcsec_predicted", "mean")
        )
        summary_metrics.append(
            {"intervention_id": intervention, "seed_count": 3, **regression_metrics(parent["theta_E_arcsec_true"], parent["prediction"])}
        )
    pd.DataFrame(summary_metrics).to_csv(output_dir / "shortcut_summary_metrics.csv", index=False)

    canonical = predictions.loc[predictions["intervention_id"] == "canonical", [
        "parent_sample_id", "seed", "theta_E_arcsec_true", "theta_E_arcsec_predicted", "absolute_error_arcsec"
    ]].rename(
        columns={
            "theta_E_arcsec_predicted": "canonical_prediction_arcsec",
            "absolute_error_arcsec": "canonical_absolute_error_arcsec",
        }
    )
    paired = predictions.loc[predictions["intervention_id"] != "canonical"].merge(
        canonical,
        on=["parent_sample_id", "seed", "theta_E_arcsec_true"],
        validate="many_to_one",
    )
    paired["paired_delta_absolute_error_arcsec"] = paired["absolute_error_arcsec"] - paired["canonical_absolute_error_arcsec"]
    paired["paired_prediction_shift_arcsec"] = paired["theta_E_arcsec_predicted"] - paired["canonical_prediction_arcsec"]
    paired.to_csv(output_dir / "shortcut_paired_predictions.csv", index=False)
    effect_rows = []
    for intervention, group in paired.groupby("intervention_id", sort=False):
        parent = group.groupby("parent_sample_id", as_index=False).agg(
            delta_mae=("paired_delta_absolute_error_arcsec", "mean"),
            prediction_shift=("paired_prediction_shift_arcsec", "mean"),
        )
        low, high = _bootstrap_mean(parent["delta_mae"].to_numpy())
        mean_delta = float(parent["delta_mae"].mean())
        effect_rows.append(
            {
                "intervention_id": intervention,
                "parent_count": len(parent),
                "paired_delta_mae_arcsec": mean_delta,
                "parent_bootstrap_95_low_arcsec": low,
                "parent_bootstrap_95_high_arcsec": high,
                "mean_absolute_paired_prediction_shift_arcsec": float(parent["prediction_shift"].abs().mean()),
                "practically_important_increase_supported": bool(mean_delta >= PRACTICAL_EFFECT and low > 0.0),
                "effect_practically_small": bool(low >= -PRACTICAL_EFFECT and high <= PRACTICAL_EFFECT),
            }
        )
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(output_dir / "shortcut_paired_effects.csv", index=False)
    bin_rows = []
    for intervention, group in predictions.groupby("intervention_id", sort=False):
        parent = group.groupby(["parent_sample_id", "theta_E_arcsec_true"], as_index=False).agg(
            prediction=("theta_E_arcsec_predicted", "mean")
        )
        for row in target_bin_metrics(parent["theta_E_arcsec_true"], parent["prediction"], TARGET_BINS):
            bin_rows.append({"intervention_id": intervention, **row})
    pd.DataFrame(bin_rows).to_csv(output_dir / "shortcut_target_bins.csv", index=False)

    summary = {
        "interventions": list(interventions),
        "parent_count": 150,
        "seed_ids": [12001, 12002, 12003],
        "checkpoint_sha256_by_seed": checkpoint_hashes,
        "canonical_prediction_max_abs_difference_arcsec": max_difference,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": SELECTION_SEED,
        "practical_effect_arcsec": PRACTICAL_EFFECT,
        "paired_effects": effect_rows,
        "historical_test_accessed": False,
    }
    write_json(output_dir / "shortcut_summary.json", summary)
    return summary


def _read_cnn_condition(condition: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    name = condition["condition"]
    if name in {"canonical", "pixel_scale_004", "pixel_scale_006"}:
        path = Path("results/reports/reliability_v1/pixel_scale/frozen_baseline/predictions.csv")
        intervention = {
            "canonical": "canonical",
            "pixel_scale_004": "pixel-scale-004",
            "pixel_scale_006": "pixel-scale-006",
        }[name]
    elif name in {"exposure_half", "read_noise_double"}:
        path = Path("results/reports/reliability_v1/observation_systematics/cnn_predictions.csv")
        intervention = {
            "exposure_half": "exposure-half",
            "read_noise_double": "read-noise-double",
        }[name]
    else:
        path = Path("results/reports/reliability_v1/source_physics/cnn_predictions.csv")
        intervention = {
            "source_x_plus": "source-x-plus",
            "shear_gamma2_005": "shear-gamma2-005",
        }[name]
    all_rows = pd.read_csv(path)
    rows = all_rows.loc[all_rows["intervention_id"].astype(str) == intervention].copy()
    if rows.empty:
        raise ValueError(f"No frozen CNN predictions for {name} in {path}")
    if "seed" not in rows:
        parsed = rows["checkpoint_id"].astype(str).str.extract(r"seed-(\d+)", expand=False)
        rows["seed"] = pd.to_numeric(parsed, errors="raise").astype(int)
    per_seed = rows.groupby("seed").size()
    if len(per_seed) != 3 or not per_seed.eq(150).all():
        raise ValueError(f"Frozen CNN outputs for {name} are not three complete 150-parent seeds")
    metrics = []
    for seed, group in rows.groupby("seed", sort=True):
        metrics.append({"condition": name, "seed": int(seed), **regression_metrics(group["theta_E_arcsec_true"], group["theta_E_arcsec_predicted"])})
    aggregate = rows.groupby("parent_sample_id", as_index=False).agg(
        theta_E_arcsec_true=("theta_E_arcsec_true", "mean"),
        theta_E_arcsec_predicted=("theta_E_arcsec_predicted", "mean"),
        seed_count=("seed", "nunique"),
    )
    if aggregate["seed_count"].ne(3).any():
        raise ValueError(f"Frozen CNN predictions for {name} are missing seeds")
    aggregate["condition"] = name
    return aggregate, pd.DataFrame(metrics)


def _read_classical_condition(condition: dict) -> pd.DataFrame:
    name = condition["condition"]
    if name in {"canonical", "pixel_scale_004", "pixel_scale_006"}:
        path = Path("results/reports/reliability_v1/pixel_scale/classical_geometric/predictions.csv")
        intervention = {
            "canonical": "canonical",
            "pixel_scale_004": "pixel-scale-004",
            "pixel_scale_006": "pixel-scale-006",
        }[name]
    elif name in {"exposure_half", "read_noise_double"}:
        path = Path("results/reports/reliability_v1/observation_systematics/classical/classical_predictions.csv")
        intervention = {
            "exposure_half": "exposure-half",
            "read_noise_double": "read-noise-double",
        }[name]
    else:
        path = Path("results/reports/reliability_v1/source_physics/classical/classical_predictions.csv")
        intervention = {
            "source_x_plus": "source-x-plus",
            "shear_gamma2_005": "shear-gamma2-005",
        }[name]
    rows = pd.read_csv(path)
    rows = rows.loc[rows["intervention_id"].astype(str) == intervention].copy()
    if len(rows) != 150 or rows["parent_sample_id"].nunique() != 150:
        raise ValueError(f"Classical point-image predictions for {name} are incomplete")
    rows = rows.rename(columns={"theta_E_arcsec_predicted": "prediction"})
    rows["condition"] = name
    rows["used_fallback"] = rows["used_training_mean_fallback"].astype(bool)
    rows = rows.rename(columns={"theta_E_arcsec_true": "classical_theta_E_arcsec_true"})
    return rows[["condition", "parent_sample_id", "prediction", "used_fallback", "classical_theta_E_arcsec_true"]]


def run_baseline_comparison(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ridge = pd.read_csv(ROOT / "ridge" / "ridge_predictions.csv")
    training_labels = pd.read_csv(TRAINING_RELEASE / "splits" / "train.csv", usecols=["theta_E_arcsec"])
    train_mean = float(training_labels["theta_E_arcsec"].mean())
    all_predictions = []
    seed_metrics = []
    reference_by_condition = {}
    cnn_frames = {}
    for condition in CONDITIONS:
        name = str(condition["condition"])
        truth = ridge.loc[ridge["condition"] == name, ["parent_sample_id", "theta_E_arcsec_true"]].copy()
        if len(truth) != 150:
            raise ValueError(f"Ridge scoring output lacks the complete truth set for {name}")
        reference_by_condition[name] = truth
        for method in ("training_mean", "fixed_feature_ridge"):
            candidate = ridge.loc[ridge["condition"] == name, ["parent_sample_id", "theta_E_arcsec_true", "theta_E_arcsec_predicted"]].copy()
            candidate["condition"] = name
            candidate["method"] = method
            candidate["prediction"] = train_mean if method == "training_mean" else candidate["theta_E_arcsec_predicted"]
            candidate["used_fallback"] = False
            all_predictions.append(candidate[["condition", "method", "parent_sample_id", "theta_E_arcsec_true", "prediction", "used_fallback"]])

        cnn, cnn_seed_metrics = _read_cnn_condition(condition)
        cnn_frames[name] = cnn
        seed_metrics.extend(cnn_seed_metrics.to_dict(orient="records"))
        cnn_scored = truth.merge(
            cnn[["parent_sample_id", "theta_E_arcsec_true", "theta_E_arcsec_predicted"]],
            on="parent_sample_id",
            suffixes=("_expected", "_cnn"),
            validate="one_to_one",
        )
        if np.max(np.abs(cnn_scored["theta_E_arcsec_true_expected"] - cnn_scored["theta_E_arcsec_true_cnn"])) > 1e-5:
            raise ValueError(f"CNN and benchmark labels do not match for {name}")
        cnn_scored["condition"] = name
        cnn_scored["method"] = "frozen_cnn_seed_mean"
        cnn_scored["prediction"] = cnn_scored["theta_E_arcsec_predicted"]
        cnn_scored["theta_E_arcsec_true"] = cnn_scored["theta_E_arcsec_true_expected"]
        cnn_scored["used_fallback"] = False
        all_predictions.append(cnn_scored[["condition", "method", "parent_sample_id", "theta_E_arcsec_true", "prediction", "used_fallback"]])

        classical = _read_classical_condition(condition)
        classical_scored = truth.merge(classical, on="parent_sample_id", validate="one_to_one")
        if np.max(np.abs(classical_scored["theta_E_arcsec_true"] - classical_scored["classical_theta_E_arcsec_true"])) > 1e-5:
            raise ValueError(f"Classical and benchmark labels do not match for {name}")
        classical_scored["method"] = "point_image_classical"
        classical_scored["prediction"] = classical_scored["prediction"]
        all_predictions.append(classical_scored[["condition", "method", "parent_sample_id", "theta_E_arcsec_true", "prediction", "used_fallback"]])

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions["residual_arcsec"] = predictions["prediction"] - predictions["theta_E_arcsec_true"]
    predictions["absolute_error_arcsec"] = predictions["residual_arcsec"].abs()
    predictions.to_csv(output_dir / "baseline_comparison_predictions.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(output_dir / "cnn_seed_metrics.csv", index=False)

    metric_rows = []
    radius_rows = []
    for (condition, method), group in predictions.groupby(["condition", "method"], sort=False):
        row = {"condition": condition, "method": method, **regression_metrics(group["theta_E_arcsec_true"], group["prediction"])}
        row["classical_fit_count"] = int((~group["used_fallback"].astype(bool)).sum()) if method == "point_image_classical" else None
        row["classical_fallback_count"] = int(group["used_fallback"].astype(bool).sum()) if method == "point_image_classical" else None
        row["classical_fallback_fraction"] = float(group["used_fallback"].astype(bool).mean()) if method == "point_image_classical" else None
        metric_rows.append(row)
        for bin_row in target_bin_metrics(group["theta_E_arcsec_true"], group["prediction"], TARGET_BINS):
            radius_rows.append({"condition": condition, "method": method, **bin_row})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
    pd.DataFrame(radius_rows).to_csv(output_dir / "model_comparison_radius_bins.csv", index=False)

    paired_rows = []
    cnn_errors = predictions.loc[predictions["method"] == "frozen_cnn_seed_mean", [
        "condition", "parent_sample_id", "absolute_error_arcsec"
    ]].rename(columns={"absolute_error_arcsec": "cnn_absolute_error_arcsec"})
    for method in ("training_mean", "fixed_feature_ridge", "point_image_classical"):
        candidate = predictions.loc[predictions["method"] == method, [
            "condition", "parent_sample_id", "absolute_error_arcsec"
        ]].rename(columns={"absolute_error_arcsec": "candidate_absolute_error_arcsec"})
        paired = candidate.merge(cnn_errors, on=["condition", "parent_sample_id"], validate="one_to_one")
        paired["candidate_minus_cnn_absolute_error_arcsec"] = paired["candidate_absolute_error_arcsec"] - paired["cnn_absolute_error_arcsec"]
        for condition, group in paired.groupby("condition", sort=False):
            low, high = _bootstrap_mean(group["candidate_minus_cnn_absolute_error_arcsec"].to_numpy())
            paired_rows.append(
                {
                    "condition": condition,
                    "method": method,
                    "parent_count": len(group),
                    "candidate_minus_cnn_delta_mae_arcsec": float(group["candidate_minus_cnn_absolute_error_arcsec"].mean()),
                    "parent_bootstrap_95_low_arcsec": low,
                    "parent_bootstrap_95_high_arcsec": high,
                }
            )
    pd.DataFrame(paired_rows).to_csv(output_dir / "method_paired_effects_vs_cnn.csv", index=False)
    summary = {
        "experiment_id": "WEEK6-BASELINES-COMPARISON",
        "condition_count": len(CONDITIONS),
        "parent_count_per_condition": 150,
        "training_mean_arcsec": train_mean,
        "ridge_selected_alpha": _load_json(ROOT / "ridge" / "ridge_summary.json")["selected_alpha"],
        "metrics": metric_rows,
        "classical_method_scope": "truth-free observation-only SIS/SIE point-image geometry with training-mean fallback; not full pixel-likelihood SIE",
        "historical_test_accessed": False,
    }
    write_json(output_dir / "model_comparison_summary.json", summary)
    return summary


def _write_shortcut_report(path: Path, shortcut_summary: dict) -> None:
    effect_by_id = {row["intervention_id"]: row for row in shortcut_summary["paired_effects"]}
    metric_by_id = {
        row["intervention_id"]: row
        for row in pd.read_csv(path.parent / "shortcuts" / "shortcut_summary_metrics.csv").to_dict(orient="records")
    }
    lines = [
        "# Week 6 shortcut audit",
        "",
        "This report describes paired image interventions on the frozen three-seed CNN and the same 150 canonical validation parents. Each intervention was applied to processed images before the checkpoint's train-only global normalization. Targets and parent identities were held fixed.",
        "",
        "The masks use only fixed image-centered angular radii (central disk <0.50 arcsec; annulus 0.70–2.20 arcsec). They are not truth-derived lens or source masks. Brightness and rotation tests are image interventions, not re-rendered physical observations. Effects are descriptive and not corrected for the number of interventions.",
        "",
        f"Canonical predictions reproduced the saved frozen-checkpoint outputs with maximum absolute difference {shortcut_summary['canonical_prediction_max_abs_difference_arcsec']:.3g} arcsec.",
        "",
        "| Intervention | Seed-averaged MAE | Paired ΔMAE [parent-bootstrap 95% interval] | Mean absolute prediction shift | Interpretation under 0.01 arcsec margin |",
        "|---|---:|---:|---:|---|",
    ]
    labels = {
        "rotate_ccw_90": "Rotate 90° counter-clockwise",
        "brightness_half": "Brightness ×0.5",
        "brightness_double": "Brightness ×2",
        "central_disk_zero": "Zero central disk <0.50 arcsec",
        "arc_annulus_zero": "Zero annulus 0.70–2.20 arcsec",
    }
    for intervention, row in effect_by_id.items():
        metrics = metric_by_id[intervention]
        if row["practically_important_increase_supported"]:
            interpretation = "practically important increase supported"
        elif row["effect_practically_small"]:
            interpretation = "practically small for this intervention"
        else:
            interpretation = "inconclusive under the frozen rule"
        lines.append(
            f"| {labels[intervention]} | {metrics['mae_arcsec']:.4f} | {row['paired_delta_mae_arcsec']:+.4f} "
            f"[{row['parent_bootstrap_95_low_arcsec']:+.4f}, {row['parent_bootstrap_95_high_arcsec']:+.4f}] | "
            f"{row['mean_absolute_paired_prediction_shift_arcsec']:.4f} | {interpretation} |"
        )
    lines.extend(
        [
            "",
            "## What the interventions establish",
            "",
            "The paired effects quantify how these fixed image transformations change predictions and validation errors. A large effect shows dependence on information changed by the intervention, but it does not identify a unique causal feature or show that the model uses an astrophysically invalid shortcut. In particular, masking removes both nuisance and potentially useful lensing information; rotation changes the orientation relative to the image axes; and brightness scaling changes the image amplitude while retaining the original normalization statistics.",
            "",
            "Source and PSF changes were assessed through the matched rerender studies in Weeks 3–5. Attribution maps were not used as causal evidence. Historical test data were not accessed.",
            "",
            "Machine-readable predictions, paired effects, target-bin summaries, and checkpoint hashes are stored beside this report.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_week6(*, output_dir=ROOT, device_request="auto") -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for checkpoint in CHECKPOINTS:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Frozen Week 6 CNN checkpoint is missing: {checkpoint}")
    protocol_path = Path("docs/reliability_study/protocol_v1.6.md")
    if not protocol_path.exists():
        raise FileNotFoundError("Week 6 protocol must be frozen before model predictions")

    learning = run_learning_curve(output / "learning_curve", device_request=device_request)
    ridge = run_ridge_baseline(output / "ridge")
    shortcuts = run_shortcut_audit(output / "shortcuts", device_request=device_request)
    comparison = run_baseline_comparison(output / "comparison")
    _write_shortcut_report(output / "shortcut_audit.md", shortcuts)
    summary = {
        "week": 6,
        "protocol": protocol_path.as_posix(),
        "protocol_sha256": file_sha256(protocol_path),
        "baseline_config_sha256": file_sha256(BASELINE_CONFIG),
        "source_code_sha256": file_sha256(Path(__file__)),
        "frozen_cnn_checkpoint_sha256_by_seed": {
            str(seed): file_sha256(path) for seed, path in zip((12001, 12002, 12003), CHECKPOINTS)
        },
        "learning_curve": learning,
        "ridge": ridge,
        "shortcuts": shortcuts,
        "comparison": comparison,
        "historical_test_accessed": False,
        "real_data_truth_accessed": False,
    }
    write_json(output / "week6_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = run_week6(output_dir=args.output, device_request=args.device)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
