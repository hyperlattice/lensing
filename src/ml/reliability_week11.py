"""Generate and evaluate the frozen Week 11 independent synthetic holdout once."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.ml.model import build_model
from src.simulation.counterfactual import create_counterfactual_benchmark
from src.simulation.generate import generate_observation_pilot


CONFIG = Path("configs/reliability/blind_holdout.yaml")
FREEZE_MANIFEST = Path("docs/reliability_study/week11_freeze_manifest.json")
DEVELOPMENT_SPLITS = (
    Path("data/simulations/sim-eng1000-v1/splits/train.csv"),
    Path("data/simulations/sim-eng1000-v1/splits/validation.csv"),
)
HOLDOUT_CONDITIONS = (
    "canonical", "pixel-scale-004", "pixel-scale-006", "exposure-half", "read-noise-double"
)
IMAGE_PRODUCT_COLUMNS = (
    "sample_id",
    "parent_sample_id",
    "processed_path",
    "intervention_id",
    "condition",
    "pixel_scale",
    "split",
    "original_shape",
    "processed_shape",
)


def _write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_freeze(config_path=CONFIG):
    if not FREEZE_MANIFEST.exists():
        raise FileNotFoundError(f"Week 11 freeze manifest is missing: {FREEZE_MANIFEST}")
    frozen = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if Path(config_path) != Path(frozen["config_path"]):
        raise ValueError("Week 11 config differs from the frozen config path")
    return frozen


def _known_development_ids_and_seeds():
    ids = set()
    parameter_seeds = set()
    noise_seeds = set()
    for path in DEVELOPMENT_SPLITS:
        frame = pd.read_csv(path, usecols=["sample_id", "parent_system_id", "parameter_seed", "noise_seed"])
        ids.update(frame["sample_id"].astype(str))
        ids.update(frame["parent_system_id"].astype(str))
        parameter_seeds.update(frame["parameter_seed"].dropna().astype(int).tolist())
        noise_seeds.update(frame["noise_seed"].dropna().astype(int).tolist())
    return ids, parameter_seeds, noise_seeds


def _intervention_ids(config):
    return [str(item["intervention_id"]) for item in config["interventions"]]


def generate_holdout(config_path=CONFIG):
    frozen = _verify_freeze(config_path)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    release = Path(config["release"])
    generation_record_path = release / "holdout_generation_manifest.json"
    if generation_record_path.exists():
        existing = _verify_release_inputs(config)
        raise FileExistsError("A Week 11 holdout release already exists with different frozen inputs")
    if release.exists() and any(release.iterdir()):
        raise FileExistsError(f"Week 11 release path is nonempty without a completion manifest: {release}")
    release.mkdir(parents=True, exist_ok=True)

    generation = config["generation"]["generator"]
    if tuple(config["generation"]["conditions"]) != HOLDOUT_CONDITIONS:
        raise ValueError("The configured holdout conditions differ from the frozen Week 11 condition order")
    if tuple(_intervention_ids(config)) != HOLDOUT_CONDITIONS:
        raise ValueError("The configured interventions differ from the frozen Week 11 condition order")
    parent_root = release / "parent_roots"
    parent_summary_path = parent_root / "run_manifest.json"
    if parent_summary_path.exists():
        parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    else:
        parent_summary = generate_observation_pilot(
            parent_root,
            count=int(generation["count"]),
            seed=int(generation["seed"]),
            simulation_id=str(generation["simulation_id"]),
            config_path=Path(generation["config"]),
            create_preview=False,
            stage_name="blind_holdout_fresh_parent_generation",
        )
    if int(parent_summary["count_accepted"]) != 150 or int(parent_summary["count_rejected"]) != 0:
        raise ValueError("The frozen fresh-parent generator did not yield exactly 150 accepted systems")

    parent_manifest_path = parent_root / "manifest.csv"
    parent_manifest = pd.read_csv(parent_manifest_path)
    if len(parent_manifest) != 150 or parent_manifest["sample_id"].duplicated().any():
        raise ValueError("Fresh parent manifest is incomplete or has duplicate IDs")
    prior_ids, prior_parameter_seeds, prior_noise_seeds = _known_development_ids_and_seeds()
    generated_ids = set(parent_manifest["sample_id"].astype(str)) | set(parent_manifest["parent_system_id"].astype(str))
    if generated_ids & prior_ids:
        raise ValueError("Fresh holdout root IDs overlap a train/validation development release")
    generated_parameter_seeds = set(parent_manifest["parameter_seed"].astype(int))
    generated_noise_seeds = set(parent_manifest["noise_seed"].astype(int))
    if len(generated_parameter_seeds) != 150 or len(generated_noise_seeds) != 150:
        raise ValueError("Fresh holdout parameter/noise seeds are not unique")
    if generated_parameter_seeds & prior_parameter_seeds or generated_noise_seeds & prior_noise_seeds:
        raise ValueError("Fresh holdout seed values overlap development train/validation seeds")

    build_root = release / "generator_selection"
    build_root.mkdir(parents=True, exist_ok=True)
    selector_path = build_root / "counterfactual_parent_selector.csv"
    pd.DataFrame(
        {"sample_id": parent_manifest["sample_id"].astype(str), "split": ["validation"] * len(parent_manifest)}
    ).to_csv(selector_path, index=False)
    observations_root = release / "observation_products"
    counterfactual = create_counterfactual_benchmark(
        parent_manifest_path,
        {"validation": selector_path},
        observations_root,
        config["interventions"],
        max_parents_per_split={"train": 0, "validation": 150},
        selection_seed=int(config["generation"]["counterfactual_selection_seed"]),
        create_preview=False,
        require_canonical=True,
    )
    if not counterfactual["passed"] or counterfactual["rejection_count"] != 0:
        raise ValueError("Fresh holdout counterfactual release did not pass its ancestry/product gates")

    generated_manifest_path = observations_root / "manifest.csv"
    holdout = pd.read_csv(generated_manifest_path)
    expected_ids = set(_intervention_ids(config))
    if len(holdout) != 750 or set(holdout["intervention_id"].astype(str).unique()) != expected_ids:
        raise ValueError("Counterfactual output does not contain the frozen 750-row, five-condition design")
    if set(holdout["sample_id"].astype(str)) & prior_ids:
        raise ValueError("Final holdout observation IDs overlap a train/validation development release")
    if holdout.groupby("intervention_id")["parent_sample_id"].nunique().to_dict() != {condition: 150 for condition in expected_ids}:
        raise ValueError("A holdout condition does not contain all 150 independent parents")
    if holdout.groupby("parent_sample_id")["theta_E_arcsec"].nunique().max() != 1:
        raise ValueError("Counterfactual generation changed theta_E within a matched parent")

    path_columns = [column for column in holdout.columns if column.endswith("_path")]
    for column in path_columns:
        holdout[column] = holdout[column].map(
            lambda value: (
                f"observation_products/{value}"
                if pd.notna(value) and not Path(str(value)).is_absolute()
                else value
            )
        )
    holdout["split"] = "holdout"
    holdout["holdout_role"] = "blind_synthetic_once"
    holdout["condition"] = holdout["intervention_id"].astype(str)

    manifest_path = release / "manifest.csv"
    holdout.to_csv(manifest_path, index=False)
    split_root = release / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(split_root / "holdout.csv", index=False)

    product_path_columns = [column for column in holdout.columns if column.endswith("_path")]
    for row in holdout.to_dict(orient="records"):
        for kind in product_path_columns:
            value = row.get(kind)
            if value is None or pd.isna(value):
                continue
            product_path = release / str(value)
            if not product_path.is_file():
                raise FileNotFoundError(f"Generated holdout product is missing: {product_path}")
    parent_path_columns = [column for column in parent_manifest.columns if column.endswith("_path")]
    for row in parent_manifest.to_dict(orient="records"):
        for kind in parent_path_columns:
            value = row.get(kind)
            if value is None or pd.isna(value):
                continue
            source_path = Path(str(value))
            if not source_path.is_absolute():
                source_path = parent_root / source_path
            if not source_path.is_file():
                raise FileNotFoundError(f"Fresh parent input product is missing: {source_path}")
            relative = source_path.relative_to(release).as_posix()
    metadata_paths = [
        parent_manifest_path,
        parent_summary_path,
        parent_root / "rejection_log.csv",
        selector_path,
        observations_root / "run_manifest.json",
        generated_manifest_path,
        observations_root / "rejection_log.csv",
        manifest_path,
        split_root / "holdout.csv",
    ]
    for path in metadata_paths:
        relative = path.relative_to(release).as_posix()

    split_files = sorted(path.name for path in split_root.iterdir() if path.is_file())
    if not split_files or any(not name.startswith("holdout") for name in split_files):
        raise ValueError(f"Final holdout root has an unexpected split file: {split_files}")
    summary = {
        "experiment_id": config["experiment_id"],
        "release_id": "reliability-blind-holdout-v1",
        "fresh_parent_seed": int(generation["seed"]),
        "fresh_parent_simulation_id": str(generation["simulation_id"]),
        "counterfactual_selection_seed": int(config["generation"]["counterfactual_selection_seed"]),
        "release_manifest": str(manifest_path),
        "generator_selection_artifact": str(selector_path),
        "final_split_role": "holdout",
        "root_split_files": split_files,
        "parent_count": 150,
        "observation_count": 750,
        "condition_count": 5,
        "conditions": {str(key): int(value) for key, value in holdout.groupby("condition").size().items()},
        "parameter_seed_unique": len(generated_parameter_seeds) == 150,
        "noise_seed_unique": len(generated_noise_seeds) == 150,
        "parameter_seeds_disjoint_from_development": not bool(generated_parameter_seeds & prior_parameter_seeds),
        "noise_seeds_disjoint_from_development": not bool(generated_noise_seeds & prior_noise_seeds),
        "root_ids_disjoint_from_development": not bool(generated_ids & prior_ids),
        "observation_ids_disjoint_from_development": not bool(set(holdout["sample_id"].astype(str)) & prior_ids),
        "parent_ancestry_passed": bool(counterfactual["ancestry_validation"]["passed"]),
        "canonical_identity_checks_passed": bool(counterfactual["identity_checks_passed"]),
        "strict_product_loaders_passed": bool(counterfactual["passed"]),
        "historical_test_accessed": False,
        "real_data_labels_accessed": False,
    }
    _write_json(generation_record_path, summary)
    return summary


def _read_holdout_observations(config):
    release = Path(config["release"])
    manifest_path = release / "splits" / "holdout.csv"
    manifest = pd.read_csv(manifest_path, usecols=list(IMAGE_PRODUCT_COLUMNS))
    if len(manifest) != 750 or manifest["sample_id"].duplicated().any():
        raise ValueError("Final holdout manifest has an incorrect row count or duplicated sample IDs")
    if set(manifest["intervention_id"].astype(str).unique()) != set(HOLDOUT_CONDITIONS):
        raise ValueError("Final holdout manifest condition set changed")
    if manifest.groupby("condition")["parent_sample_id"].nunique().to_dict() != {condition: 150 for condition in HOLDOUT_CONDITIONS}:
        raise ValueError("Final holdout manifest has incomplete parent coverage")
    if not (manifest["split"].astype(str) == "holdout").all():
        raise ValueError("Final holdout split marker changed")
    if manifest.duplicated(["parent_sample_id", "condition"]).any():
        raise ValueError("Final holdout has duplicate parent-condition observations")
    return manifest


def _verify_release_inputs(config):
    release = Path(config["release"])
    summary_path = release / "holdout_generation_manifest.json"
    if not summary_path.exists():
        raise FileNotFoundError("The Week 11 blind holdout release has not been generated")
    generation = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = release / "manifest.csv"
    _read_holdout_observations(config)
    return generation


def _load_models(config, device):
    models = {}
    for model_group, entries in config["models"].items():
        models[model_group] = {}
        for entry in entries:
            checkpoint_path = Path(entry["checkpoint"])
            normalization_path = Path(entry["normalization"])
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            normalization_file = json.loads(normalization_path.read_text(encoding="utf-8"))
            stats = checkpoint["statistics"]
            for key in ("image_mean", "image_std", "target_mean", "target_std"):
                if not np.isclose(float(stats[key]), float(normalization_file[key]), rtol=0.0, atol=1e-12):
                    raise ValueError(f"Checkpoint and normalization artifact disagree for {checkpoint_path} ({key})")
            model = build_model(checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            seed = int(checkpoint["seed"])
            models[model_group][seed] = (model, stats)
    for group in models:
        if sorted(models[group]) != [12001, 12002, 12003]:
            raise ValueError(f"Frozen model group {group} does not contain the three declared seeds")
    return models


def preflight(config_path=CONFIG):
    frozen = _verify_freeze(config_path)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    release_summary = _verify_release_inputs(config)
    device = torch.device("cpu")
    torch.set_num_threads(6)
    models = _load_models(config, device)
    observations = _read_holdout_observations(config)
    shape_counts = {}
    native_shapes = {
        float(scale): tuple(int(value) for value in shape)
        for scale, shape in config["generation"]["native_shape_by_pixel_scale"].items()
    }
    expected_scales = {
        "canonical": 0.05,
        "pixel-scale-004": 0.04,
        "pixel-scale-006": 0.06,
        "exposure-half": 0.05,
        "read-noise-double": 0.05,
    }
    for condition, rows in observations.groupby("condition", sort=False):
        shapes = set()
        for row in rows.to_dict(orient="records"):
            path = str(row["processed_path"])
            array = np.load(Path(config["release"]) / path, mmap_mode="r")
            if array.ndim != 2:
                raise ValueError(f"Holdout input must be a two-dimensional image: {path}")
            shapes.add(tuple(int(value) for value in array.shape))
            expected_shape = native_shapes[expected_scales[condition]]
            expected_original = "x".join(str(value) for value in expected_shape)
            if str(row["original_shape"]) != expected_original or str(row["processed_shape"]) != expected_original:
                raise ValueError(f"{condition}: manifest dimensions do not match the frozen native shape")
            if not np.isclose(float(row["pixel_scale"]), expected_scales[condition], rtol=0.0, atol=1e-12):
                raise ValueError(f"{condition}: pixel scale differs from the frozen condition")
        if len(shapes) != 1:
            raise ValueError(f"Condition {condition} contains mixed input shapes: {sorted(shapes)}")
        shape_counts[condition] = list(next(iter(shapes)))
    result = {
        "preflight_passed": True,
        "parent_count": 150,
        "observation_count": int(len(observations)),
        "condition_shape": shape_counts,
        "final_split_role": "holdout",
        "historical_test_accessed": False,
        "real_data_labels_accessed": False,
    }
    _write_json(Path(config["output"]) / "holdout_preflight.json", result)
    return result


def _predict_once(config, observations, models, device):
    release = Path(config["release"])
    batch_size = int(config["evaluation"]["batch_size"])
    per_checkpoint = []
    for condition in HOLDOUT_CONDITIONS:
        condition_rows = observations.loc[observations["condition"] == condition].reset_index(drop=True)
        arrays = [np.load(release / str(path)).astype(np.float32, copy=False) for path in condition_rows["processed_path"]]
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1:
            raise ValueError(f"{condition}: inputs do not share the frozen native shape")
        values = np.stack(arrays, axis=0)[:, None, :, :]
        for model_group in ("domain_randomized", "canonical_baseline"):
            for seed, (model, stats) in models[model_group].items():
                normalized = (values - float(stats["image_mean"])) / float(stats["image_std"])
                predictions = []
                with torch.inference_mode():
                    for start in range(0, len(normalized), batch_size):
                        batch = torch.from_numpy(normalized[start:start + batch_size]).to(device)
                        outputs = model(batch).detach().cpu().numpy().astype(np.float64)
                        predictions.extend((outputs * float(stats["target_std"]) + float(stats["target_mean"])).tolist())
                if not np.isfinite(predictions).all():
                    raise ValueError(f"{condition}/{model_group}/{seed}: inference produced non-finite predictions")
                for row, prediction in zip(condition_rows.to_dict(orient="records"), predictions):
                    per_checkpoint.append(
                        {
                            "sample_id": str(row["sample_id"]),
                            "parent_sample_id": str(row["parent_sample_id"]),
                            "condition": condition,
                            "model_group": model_group,
                            "seed": seed,
                            "theta_E_arcsec_predicted": float(prediction),
                        }
                    )
            print(f"completed frozen inference: condition={condition} model={model_group}", flush=True)
    per_checkpoint_frame = pd.DataFrame(per_checkpoint)
    expected = 750 * 2 * 3
    if len(per_checkpoint_frame) != expected or per_checkpoint_frame.duplicated(
        ["sample_id", "condition", "model_group", "seed"]
    ).any():
        raise ValueError("Per-checkpoint holdout predictions are incomplete or duplicated")
    per_checkpoint_path = Path(config["output"]) / "holdout_predictions_per_checkpoint_before_scoring.csv"
    per_checkpoint_frame.to_csv(per_checkpoint_path, index=False)

    ensemble_frames = []
    for model_group, group in per_checkpoint_frame.groupby("model_group", sort=False):
        pivot = group.pivot(
            index=["sample_id", "parent_sample_id", "condition"],
            columns="seed",
            values="theta_E_arcsec_predicted",
        ).reset_index()
        pivot = pivot.rename(columns={seed: f"theta_E_seed_{seed}_arcsec" for seed in (12001, 12002, 12003)})
        seed_columns = [f"theta_E_seed_{seed}_arcsec" for seed in (12001, 12002, 12003)]
        pivot["model_group"] = model_group
        pivot["theta_E_arcsec_predicted"] = pivot[seed_columns].mean(axis=1)
        pivot["seed_prediction_sd_arcsec"] = pivot[seed_columns].std(axis=1, ddof=1)
        ensemble_frames.append(pivot)
    ensemble_frame = pd.concat(ensemble_frames, ignore_index=True)
    ensemble_frame = ensemble_frame.sort_values(["condition", "parent_sample_id", "model_group"]).reset_index(drop=True)
    ensemble_path = Path(config["output"]) / config["evaluation"]["prediction_file"]
    ensemble_frame.to_csv(ensemble_path, index=False)
    return per_checkpoint_frame, ensemble_frame, ensemble_path, per_checkpoint_path


def _regression_summary(true, predicted):
    true = np.asarray(true, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - true
    absolute = np.abs(error)
    return {
        "count": int(len(true)),
        "mae_arcsec": float(absolute.mean()),
        "rmse_arcsec": float(np.sqrt(np.mean(np.square(error)))),
        "bias_arcsec": float(error.mean()),
        "median_absolute_error_arcsec": float(np.median(absolute)),
        "p90_absolute_error_arcsec": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error_arcsec": float(np.quantile(absolute, 0.95)),
        "fraction_within_005_arcsec": float(np.mean(absolute <= 0.05)),
        "fraction_within_010_arcsec": float(np.mean(absolute <= 0.10)),
        "count_error_gt_005_arcsec": int(np.sum(absolute > 0.05)),
        "count_error_gt_010_arcsec": int(np.sum(absolute > 0.10)),
    }


def _bootstrap_delta(values, *, resamples, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(values), size=(int(resamples), len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _score_once(config, per_checkpoint, ensemble_frame, prediction_paths, frozen):
    output = Path(config["output"])
    once_marker = output / config["evaluation"]["once_marker"]
    if once_marker.exists():
        raise FileExistsError("Week 11 holdout has already been scored; exactly-once guard refused a second run")
    scored_path = output / config["evaluation"]["scored_file"]
    if scored_path.exists():
        raise FileExistsError("A scored Week 11 output already exists without the completion marker")

    predictions_path, per_checkpoint_path = prediction_paths

    # This is the first point where holdout target columns are read.
    holdout_manifest_path = Path(config["release"]) / "manifest.csv"
    truth_rows = pd.read_csv(
        holdout_manifest_path,
        usecols=["parent_sample_id", "theta_E_arcsec"],
    )
    truth_counts = truth_rows.groupby("parent_sample_id")["theta_E_arcsec"].nunique()
    if len(truth_counts) != 150 or truth_counts.max() != 1:
        raise ValueError("Holdout targets are not one invariant value per parent system")
    truth = truth_rows.drop_duplicates("parent_sample_id").rename(columns={"theta_E_arcsec": "theta_E_arcsec_true"})

    scored_ensemble = ensemble_frame.merge(truth, on="parent_sample_id", validate="many_to_one")
    if scored_ensemble["theta_E_arcsec_true"].isna().any():
        raise ValueError("A holdout prediction did not join to exactly one target")
    scored_ensemble["absolute_error_arcsec"] = (
        scored_ensemble["theta_E_arcsec_predicted"] - scored_ensemble["theta_E_arcsec_true"]
    ).abs()
    scored_ensemble["prediction_shift_arcsec"] = np.nan
    for condition in HOLDOUT_CONDITIONS:
        mask = scored_ensemble["condition"] == condition
        random_values = scored_ensemble.loc[mask & (scored_ensemble["model_group"] == "domain_randomized")].set_index("parent_sample_id")["theta_E_arcsec_predicted"]
        base_values = scored_ensemble.loc[mask & (scored_ensemble["model_group"] == "canonical_baseline")].set_index("parent_sample_id")["theta_E_arcsec_predicted"]
        shifts = random_values - base_values
        scored_ensemble.loc[mask & (scored_ensemble["model_group"] == "domain_randomized"), "prediction_shift_arcsec"] = scored_ensemble.loc[
            mask & (scored_ensemble["model_group"] == "domain_randomized"), "parent_sample_id"
        ].map(shifts)
    scored_ensemble.to_csv(scored_path, index=False)

    per_scored = per_checkpoint.merge(truth, on="parent_sample_id", validate="many_to_one")
    per_scored["absolute_error_arcsec"] = (
        per_scored["theta_E_arcsec_predicted"] - per_scored["theta_E_arcsec_true"]
    ).abs()
    per_checkpoint_scored_path = output / "holdout_scored_per_checkpoint.csv"
    per_scored.to_csv(per_checkpoint_scored_path, index=False)

    metric_rows = []
    for (model_group, condition), group in scored_ensemble.groupby(["model_group", "condition"], sort=False):
        metrics = _regression_summary(group["theta_E_arcsec_true"], group["theta_E_arcsec_predicted"])
        metric_rows.append({"model_group": model_group, "predictor": "three_seed_ensemble", "condition": condition, **metrics})
    for (model_group, seed, condition), group in per_scored.groupby(["model_group", "seed", "condition"], sort=False):
        metrics = _regression_summary(group["theta_E_arcsec_true"], group["theta_E_arcsec_predicted"])
        metric_rows.append({"model_group": model_group, "predictor": str(seed), "condition": condition, **metrics})
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / "metrics_by_condition_seed_and_ensemble.csv", index=False)

    bins = [0.5, 1.0, 1.5, 2.0]
    bin_rows = []
    scored_ensemble["theta_E_bin"] = pd.cut(
        scored_ensemble["theta_E_arcsec_true"], bins=bins, right=False, include_lowest=True
    )
    for (model_group, condition, radius_bin), group in scored_ensemble.groupby(
        ["model_group", "condition", "theta_E_bin"], observed=False, sort=False
    ):
        if group.empty:
            continue
        bin_rows.append(
            {
                "model_group": model_group,
                "predictor": "three_seed_ensemble",
                "condition": condition,
                "theta_E_bin": str(radius_bin),
                **_regression_summary(group["theta_E_arcsec_true"], group["theta_E_arcsec_predicted"]),
            }
        )
    pd.DataFrame(bin_rows).to_csv(output / "metrics_by_theta_E_bin.csv", index=False)

    training_targets = pd.read_csv(
        "data/simulations/reliability-domain-randomized-v1/splits/train.csv",
        usecols=["theta_E_arcsec"],
    )["theta_E_arcsec"].to_numpy(float)
    target_min, target_max = float(training_targets.min()), float(training_targets.max())
    scored_ensemble["prediction_outside_training_target_range"] = (
        (scored_ensemble["theta_E_arcsec_predicted"] < target_min)
        | (scored_ensemble["theta_E_arcsec_predicted"] > target_max)
    )
    target_range_rows = []
    for (model_group, condition), group in scored_ensemble.groupby(["model_group", "condition"], sort=False):
        target_range_rows.append(
            {
                "model_group": model_group,
                "condition": condition,
                "training_theta_E_min_arcsec": target_min,
                "training_theta_E_max_arcsec": target_max,
                "outside_training_range_count": int(group["prediction_outside_training_target_range"].sum()),
                "outside_training_range_fraction": float(group["prediction_outside_training_target_range"].mean()),
            }
        )
    pd.DataFrame(target_range_rows).to_csv(output / "target_range_coverage.csv", index=False)

    parent_effects = []
    bootstrap_count = int(config["evaluation"]["bootstrap_resamples"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    ensemble_only = scored_ensemble.loc[scored_ensemble["model_group"].isin(["domain_randomized", "canonical_baseline"])]
    for condition in HOLDOUT_CONDITIONS:
        subset = ensemble_only.loc[ensemble_only["condition"] == condition]
        wide_error = subset.pivot(index="parent_sample_id", columns="model_group", values="absolute_error_arcsec")
        wide_pred = subset.pivot(index="parent_sample_id", columns="model_group", values="theta_E_arcsec_predicted")
        for parent in wide_error.index:
            parent_effects.append(
                {
                    "parent_sample_id": parent,
                    "condition": condition,
                    "domain_randomized_abs_error_arcsec": float(wide_error.loc[parent, "domain_randomized"]),
                    "canonical_baseline_abs_error_arcsec": float(wide_error.loc[parent, "canonical_baseline"]),
                    "delta_abs_error_randomized_minus_baseline_arcsec": float(
                        wide_error.loc[parent, "domain_randomized"] - wide_error.loc[parent, "canonical_baseline"]
                    ),
                    "prediction_shift_randomized_minus_baseline_arcsec": float(
                        wide_pred.loc[parent, "domain_randomized"] - wide_pred.loc[parent, "canonical_baseline"]
                    ),
                }
            )
    parent_effects_frame = pd.DataFrame(parent_effects)
    stress = parent_effects_frame.loc[parent_effects_frame["condition"].isin(HOLDOUT_CONDITIONS[1:])]
    stress_composite = stress.groupby("parent_sample_id", as_index=False).agg(
        delta_abs_error_randomized_minus_baseline_arcsec=("delta_abs_error_randomized_minus_baseline_arcsec", "mean"),
        prediction_shift_randomized_minus_baseline_arcsec=("prediction_shift_randomized_minus_baseline_arcsec", "mean"),
    )
    composite_path = output / "stress_composite_parent_effects.csv"
    stress_composite.to_csv(composite_path, index=False)
    parent_effects_frame.to_csv(output / "paired_parent_effects.csv", index=False)

    paired_rows = []
    for condition, group in list(parent_effects_frame.groupby("condition", sort=False)) + [("four_stress_composite", stress_composite)]:
        for metric in ("delta_abs_error_randomized_minus_baseline_arcsec", "prediction_shift_randomized_minus_baseline_arcsec"):
            values = group[metric].to_numpy(float)
            mean, low, high = _bootstrap_delta(values, resamples=bootstrap_count, seed=bootstrap_seed)
            paired_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "effect_definition": "domain_randomized ensemble minus canonical baseline ensemble",
                    "parent_count": int(len(values)),
                    "mean": mean,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "bootstrap_resamples": bootstrap_count,
                    "bootstrap_seed": bootstrap_seed,
                }
            )
    effects_frame = pd.DataFrame(paired_rows)
    effects_frame.to_csv(output / "paired_effects.csv", index=False)

    final_model = scored_ensemble.loc[scored_ensemble["model_group"] == "domain_randomized"].copy()
    failure_catalog = final_model.sort_values("absolute_error_arcsec", ascending=False).head(30)
    failure_catalog[
        ["parent_sample_id", "sample_id", "condition", "theta_E_arcsec_true", "theta_E_arcsec_predicted", "absolute_error_arcsec", "seed_prediction_sd_arcsec"]
    ].to_csv(output / "largest_final_model_errors.csv", index=False)

    plot_effects = effects_frame.loc[
        effects_frame["metric"] == "delta_abs_error_randomized_minus_baseline_arcsec"
    ].copy()
    labels = plot_effects["condition"].tolist()
    x = np.arange(len(labels))
    mean = plot_effects["mean"].to_numpy(float)
    low = plot_effects["bootstrap_95_low"].to_numpy(float)
    high = plot_effects["bootstrap_95_high"].to_numpy(float)
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.vlines(x, low, high, color="#365f91", linewidth=1.5)
    axis.scatter(x, mean, color="#365f91", zorder=3)
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.axhline(-0.01, color="#777777", linewidth=0.9, linestyle=":")
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylabel("Paired MAE change (randomized − baseline, arcsec)")
    axis.set_title("Independent synthetic holdout: 95% parent bootstrap intervals")
    fig.tight_layout()
    effect_figure = output / "holdout_paired_effects.png"
    fig.savefig(effect_figure, dpi=180)
    plt.close(fig)

    mae_pivot = metrics_frame.loc[metrics_frame["predictor"] == "three_seed_ensemble"].pivot(
        index="condition", columns="model_group", values="mae_arcsec"
    ).reindex(HOLDOUT_CONDITIONS)
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    mae_pivot.plot(kind="bar", ax=axis, color=["#7896b5", "#e08b4d"])
    axis.set_ylabel("Ensemble MAE (arcsec)")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(frameon=False, title="Frozen model")
    fig.tight_layout()
    metrics_figure = output / "holdout_ensemble_metrics.png"
    fig.savefig(metrics_figure, dpi=180)
    plt.close(fig)

    metric_lookup = metrics_frame.loc[metrics_frame["predictor"] == "three_seed_ensemble"].set_index(["model_group", "condition"])
    stress_effect = effects_frame.loc[
        (effects_frame["condition"] == "four_stress_composite")
        & (effects_frame["metric"] == "delta_abs_error_randomized_minus_baseline_arcsec")
    ].iloc[0]
    summary = {
        "experiment_id": config["experiment_id"],
        "protocol": str(config["protocol"]),
        "release_id": "reliability-blind-holdout-v1",
        "parent_count": 150,
        "observation_count": 750,
        "conditions": list(HOLDOUT_CONDITIONS),
        "evaluated_checkpoint_count": 6,
        "ensemble_prediction_count": int(len(ensemble_frame)),
        "per_checkpoint_prediction_count": int(len(per_checkpoint)),
        "predictions_written_before_target_join": True,
        "target_joined_once": True,
        "target_source": "new holdout manifest only, after predictions were saved",
        "historical_test_accessed": False,
        "real_data_labels_accessed": False,
        "post_holdout_tuning_or_checkpoint_selection": False,
        "training_target_range_arcsec": [target_min, target_max],
        "equal_weight_four_stress_paired_mae_delta_arcsec": float(stress_effect["mean"]),
        "equal_weight_four_stress_paired_mae_delta_95_ci_arcsec": [
            float(stress_effect["bootstrap_95_low"]), float(stress_effect["bootstrap_95_high"])
        ],
        "condition_ensemble_mae_arcsec": {
            model: {condition: float(metric_lookup.loc[(model, condition), "mae_arcsec"]) for condition in HOLDOUT_CONDITIONS}
            for model in ("domain_randomized", "canonical_baseline")
        },
        "limitations": [
            "Evidence is limited to an independent holdout from the same synthetic parent prior and renderer.",
            "Stress conditions are the four preselected Week 8 nuisance mismatches; no alternate full renderer or real images are represented.",
            "The result cannot establish simulation-to-CASTLES transfer or calibrated HST detector behavior.",
        ],
    }
    _write_json(output / "summary.json", summary)
    report = [
        "# Week 11 — independent blind synthetic holdout",
        "",
        "This is the single target-joined evaluation of 150 new parent systems and 750 observations. The three-seed Week 8 randomized ensemble is the frozen final model; the original canonical-trained ensemble is a fixed comparator.",
        "",
        "## Ensemble metrics",
        "",
        "| Condition | Randomized MAE | Baseline MAE | Randomized RMSE | Baseline RMSE | Randomized bias | Baseline bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in HOLDOUT_CONDITIONS:
        random_metrics = metric_lookup.loc[("domain_randomized", condition)]
        baseline_metrics = metric_lookup.loc[("canonical_baseline", condition)]
        report.append(
            f"| {condition} | {random_metrics['mae_arcsec']:.4f} | {baseline_metrics['mae_arcsec']:.4f} | "
            f"{random_metrics['rmse_arcsec']:.4f} | {baseline_metrics['rmse_arcsec']:.4f} | "
            f"{random_metrics['bias_arcsec']:.4f} | {baseline_metrics['bias_arcsec']:.4f} |"
        )
    report.extend(
        [
            "",
            f"Equal-weight four-stress parent-paired MAE change (randomized minus baseline): **{stress_effect['mean']:+.4f} arcsec** (95% parent-bootstrap interval [{stress_effect['bootstrap_95_low']:+.4f}, {stress_effect['bootstrap_95_high']:+.4f}]).",
            "",
            "![Paired effects](holdout_paired_effects.png)",
            "",
            "![Ensemble condition metrics](holdout_ensemble_metrics.png)",
            "",
            "",
            "## Integrity and scope",
            "",
            f"- Independent parents / observations: **{summary['parent_count']} / {summary['observation_count']}**.",
            f"- Checkpoints evaluated: **{summary['evaluated_checkpoint_count']}** across two frozen model groups.",
            f"- Prediction files were target-free and saved before target join: **{summary['predictions_written_before_target_join']}**.",
            f"- Historical test accessed / real labels accessed / post-holdout tuning: **{summary['historical_test_accessed']} / {summary['real_data_labels_accessed']} / {summary['post_holdout_tuning_or_checkpoint_selection']}**.",
            "- This is independent synthetic evidence for the tested prior, simulator, preprocessing, and nuisance strata. It does not establish real-domain performance.",
            "- Error counts use the fixed descriptive thresholds |error| > 0.05 arcsec and |error| > 0.10 arcsec.",
        ]
    )
    report.extend(
        [
            "",
            "## Error counts by condition",
            "",
            "| Condition | Model | Errors > 0.05 arcsec | Errors > 0.10 arcsec |",
            "|---|---|---:|---:|",
        ]
    )
    for condition in HOLDOUT_CONDITIONS:
        for model_group in ("domain_randomized", "canonical_baseline"):
            row = metric_lookup.loc[(model_group, condition)]
            report.append(
                f"| {condition} | {model_group} | {int(row['count_error_gt_005_arcsec'])} | {int(row['count_error_gt_010_arcsec'])} |"
            )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    marker = {
        "experiment_id": config["experiment_id"],
        "scored_once": True,
        "target_join_count": 1,
        "target_join_source": str(holdout_manifest_path),
        "target_joined_at_unix": time.time(),
        "historical_test_accessed": False,
        "post_holdout_tuning": False,
    }
    _write_json(once_marker, marker)
    return summary


def evaluate_once(config_path=CONFIG):
    frozen = _verify_freeze(config_path)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    exact_once_paths = (
        output / config["evaluation"]["inference_started_marker"],
        output / config["evaluation"]["once_marker"],
        output / config["evaluation"]["prediction_file"],
        output / "holdout_predictions_per_checkpoint_before_scoring.csv",
        output / config["evaluation"]["scored_file"],
    )
    if any(path.exists() for path in exact_once_paths):
        raise FileExistsError("Week 11 holdout inference/scoring has already started; no second run is allowed")
    preflight_path = output / "holdout_preflight.json"
    if not preflight_path.exists():
        raise FileNotFoundError("Run the frozen holdout preflight before the exactly-once evaluation")
    preflight_summary = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight_summary.get("preflight_passed"):
        raise ValueError("Week 11 holdout preflight did not pass")
    release_generation = _verify_release_inputs(config)
    observations = _read_holdout_observations(config)
    device = torch.device("cpu")
    torch.set_num_threads(int(config["evaluation"]["torch_num_threads"]))
    models = _load_models(config, device)
    inference_started_path = output / config["evaluation"]["inference_started_marker"]
    _write_json(
        inference_started_path,
        {
            "experiment_id": config["experiment_id"],
            "inference_started": True,
            "started_at_unix": time.time(),
            "historical_test_accessed": False,
            "real_data_labels_accessed": False,
        },
    )
    started = time.perf_counter()
    per_checkpoint, ensembles, ensemble_path, per_checkpoint_path = _predict_once(
        config, observations, models, device
    )
    elapsed = time.perf_counter() - started
    result = _score_once(config, per_checkpoint, ensembles, (ensemble_path, per_checkpoint_path), frozen)
    result["inference_runtime_seconds"] = float(elapsed)
    result["evaluation_environment"] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "torch_num_threads": int(config["evaluation"]["torch_num_threads"]),
    }
    _write_json(Path(config["output"]) / "summary.json", result)
    marker_path = Path(config["output"]) / config["evaluation"]["once_marker"]
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    _write_json(marker_path, marker)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("generate", "preflight", "evaluate"))
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args()
    if args.phase == "generate":
        result = generate_holdout(args.config)
    elif args.phase == "preflight":
        result = preflight(args.config)
    else:
        result = evaluate_once(args.config)
    compact_keys = (
        "experiment_id", "release_id", "parent_count", "observation_count", "condition_count",
        "preflight_passed", "evaluated_checkpoint_count", "equal_weight_four_stress_paired_mae_delta_arcsec",
        "equal_weight_four_stress_paired_mae_delta_95_ci_arcsec", "historical_test_accessed",
        "real_data_labels_accessed", "post_holdout_tuning_or_checkpoint_selection",
    )
    print(json.dumps({key: result[key] for key in compact_keys if key in result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
