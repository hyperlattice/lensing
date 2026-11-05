"""Build, fit, and evaluate the frozen Week 8 nuisance-randomization proposal."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import LensDataset
from src.data.schema import validate_row
from src.ml.data import TorchLensDataset, compute_training_statistics, file_sha256
from src.ml.engine import collect_predictions, make_loader, seed_everything, select_device, train_one_epoch, write_json
from src.ml.losses import build_loss
from src.ml.metrics import improvement_fraction, mean_baseline_metrics, regression_metrics, target_bin_metrics
from src.ml.model import build_model
from src.ml.plots import plot_predictions, plot_training_history
from src.ml.provenance import config_sha256, environment_record, source_provenance
from src.simulation.counterfactual import _render_parent, apply_intervention
from src.ml.scale_aware_train import ShapeBucketBatchSampler


DEFAULT_CONFIG = Path("configs/reliability/mitigation.yaml")
CANONICAL_RELEASE = Path("data/simulations/sim-eng1000-v1")
RANDOMIZED_RELEASE = Path("data/simulations/reliability-domain-randomized-v1")
OUTPUT = Path("results/reports/reliability_v1/mitigation")
BASELINE_MODEL_ROOT = Path("results/ml/theta-e-cnn-seeds-v1")
PIXEL_RELEASE = Path("data/simulations/reliability-pixel-scale-v1-final")
OBSERVATION_RELEASE = Path("data/simulations/reliability-observation-systematics-v1")
PIXEL_PREDICTIONS = Path("results/reports/reliability_v1/pixel_scale/frozen_baseline/predictions.csv")
OBSERVATION_PREDICTIONS = Path("results/reports/reliability_v1/observation_systematics/cnn_predictions.csv")
CONDITIONS = ["canonical", "pixel-scale-004", "pixel-scale-006", "exposure-half", "read-noise-double"]
STRESS_CONDITIONS = CONDITIONS[1:]
CANONICAL_COPY_FIELDS = {
    "raw_path",
    "science_path",
    "processed_path",
    "mask_path",
    "coverage_mask_path",
    "background_mask_path",
    "variance_path",
    "expected_variance_path",
    "clean_scene_path",
    "expected_scene_path",
    "observed_science_path",
    "psf_path",
    "psf_metadata_path",
    "geometry_transform_path",
    "render_geometry_path",
    "background_path",
    "normalization_path",
    "noise_model_path",
    "image_positions_path",
    "parameters_path",
}
RENDER_PATH_FIELDS = {
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


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _source_split_rows(split: str) -> pd.DataFrame:
    return pd.read_csv(CANONICAL_RELEASE / "splits" / f"{split}.csv")


def _copy_canonical_row(row: dict[str, Any], split: str, release: Path) -> dict[str, Any]:
    output = dict(row)
    for field in CANONICAL_COPY_FIELDS:
        value = row.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            output[field] = None
            continue
        source_path = CANONICAL_RELEASE / str(value)
        if not source_path.is_file():
            output[field] = None
            continue
        relative = Path("canonical") / field.removesuffix("_path") / source_path.name
        target_path = release / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        output[field] = relative.as_posix()
    for field in CANONICAL_COPY_FIELDS.difference(RENDER_PATH_FIELDS):
        if field not in output:
            output[field] = None
    output.update(
        {
            "dataset_version": "reliability-domain-randomized-v1",
            "simulation_id": "reliability-domain-randomized-v1",
            "parent_sample_id": str(row["sample_id"]),
            "original_parent_sample_id": str(row["sample_id"]),
            "split": split,
            "intervention_id": "canonical",
            "intervention_family": "canonical-parent-view",
        }
    )
    if not output.get("raw_path") or not output.get("science_path") or not output.get("processed_path") or not output.get("parameters_path"):
        raise ValueError(f"{row['sample_id']}: canonical training view is missing a required product")
    return output


def _draw_randomization(train_rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    proposal = config["randomization"]
    rng = np.random.Generator(np.random.PCG64(int(proposal["seed"])))
    count = len(train_rows)
    choices = pd.DataFrame(
        {
            "parent_sample_id": train_rows["sample_id"].astype(str).tolist(),
            "pixel_scale_arcsec": rng.choice(proposal["pixel_scale_arcsec"], size=count, replace=True),
            "exposure_time_multiplier": rng.choice(proposal["exposure_time_multiplier"], size=count, replace=True),
            "read_noise_multiplier": rng.choice(proposal["read_noise_multiplier"], size=count, replace=True),
        }
    )
    shape_map = {
        round(float(scale), 2): tuple(int(value) for value in shape)
        for scale, shape in proposal["native_shape_by_pixel_scale"].items()
    }
    choices["native_shape"] = choices["pixel_scale_arcsec"].map(
        lambda value: "x".join(str(part) for part in shape_map[round(float(value), 2)])
    )
    return choices


def _render_randomized_row(parent: dict[str, Any], draw: dict[str, Any], release: Path) -> dict[str, Any]:
    parameters_path = CANONICAL_RELEASE / str(parent["parameters_path"])
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    sample_id = f"{parent['sample_id']}__domain-randomized-v1"
    spec = {
        "intervention_id": "domain-randomized-v1",
        "family": "ranked-observation-nuisance-randomization",
        "overrides": {
            "pixel_scale_arcsec": float(draw["pixel_scale_arcsec"]),
            "shape": [int(value) for value in str(draw["native_shape"]).split("x")],
            "exposure_time": {"multiply": float(draw["exposure_time_multiplier"])},
            "read_noise_electrons": {"multiply": float(draw["read_noise_multiplier"])},
        },
        "parent_sample_id": str(parent["sample_id"]),
        "canonical_state_path": "",
    }
    parameters, applied = apply_intervention(parameters, spec)
    parameters["sample_id"] = sample_id
    parameters["parent_system_id"] = str(parent["parent_system_id"])
    parameters["simulation_id"] = "reliability-domain-randomized-v1"
    parameters["parent_sample_id"] = str(parent["sample_id"])
    paths, products = _render_parent(
        parameters,
        sample_id=sample_id,
        parent_id=str(parent["parent_system_id"]),
        intervention=spec,
        output_root=release,
        identity_source_row=None,
    )
    output = dict(parent)
    for field, key in RENDER_PATH_FIELDS.items():
        output[field] = _rel(paths[key], release)
    output.update(
        {
            "sample_id": sample_id,
            "lens_id": str(parent["parent_system_id"]),
            "dataset_version": "reliability-domain-randomized-v1",
            "simulation_id": "reliability-domain-randomized-v1",
            "parent_sample_id": str(parent["sample_id"]),
            "original_parent_sample_id": str(parent["sample_id"]),
            "split": "train",
            "intervention_id": "domain-randomized-v1",
            "intervention_family": "ranked-observation-nuisance-randomization",
            "pixel_scale": float(applied["pixel_scale_arcsec"]),
            "pixel_scale_x": float(applied["pixel_scale_arcsec"]),
            "pixel_scale_y": float(applied["pixel_scale_arcsec"]),
            "exposure_time": float(parameters["exposure_time"]),
            "read_noise_electrons": float(parameters["read_noise_electrons"]),
            "original_shape": "x".join(str(value) for value in products["scene"].expected_scene.shape),
            "processed_shape": "x".join(str(value) for value in products["processed"]["normalized"].shape),
            "normalization_method": products["processed"]["normalization"].method,
            "masked_fraction": float(products["processed"]["background_mask"].mean()),
            "image_count_physical": int(products["diagnostics"]["image_count_physical"]),
            "image_count_detectable": int(products["diagnostics"]["image_count_detectable"]),
            "theta_E_target_available": True,
            "simulator_name": "lenstronomy",
            "simulator_version": str(parent["simulator_version"]),
        }
    )
    errors = validate_row(output)
    if errors:
        raise ValueError(f"{sample_id}: randomized manifest row failed schema validation: {errors}")
    return output


def create_randomized_release(config: dict[str, Any], release: Path) -> dict[str, Any]:
    if release.exists() and (release / "run_manifest.json").exists():
        summary = json.loads((release / "run_manifest.json").read_text(encoding="utf-8"))
        if summary.get("manifest_sha256") != file_sha256(release / "manifest.csv"):
            raise ValueError("Committed Week 8 release manifest hash no longer matches its run manifest")
        return summary
    if release.exists() and any(release.iterdir()) and not (release / "manifest.csv").exists():
        raise FileExistsError(f"Week 8 release has unrecognized partial contents: {release}")
    release.mkdir(parents=True, exist_ok=True)
    if (release / "manifest.csv").exists():
        return _validate_randomized_release(config, release)
    train = _source_split_rows("train")
    validation = _source_split_rows("validation")
    if len(train) != 700 or len(validation) != 150:
        raise ValueError("Week 8 requires the frozen 700/150 training/validation parents")
    if set(train["parent_system_id"].astype(str)) & set(validation["parent_system_id"].astype(str)):
        raise ValueError("Source parent ancestry crosses the frozen train/validation boundary")

    draws = _draw_randomization(train, config)
    assignment_path = release / "randomization_assignment.csv"
    draws.to_csv(assignment_path, index=False)
    draw_lookup = draws.set_index("parent_sample_id").to_dict(orient="index")
    train_canonical, randomized_rows = [], []
    val_canonical = []
    for row in train.to_dict(orient="records"):
        train_canonical.append(_copy_canonical_row(row, "train", release))
        randomized_rows.append(_render_randomized_row(row, draw_lookup[str(row["sample_id"])], release))
        if len(randomized_rows) % 50 == 0:
            print(f"Rendered Week 8 randomized views: {len(randomized_rows)}/700", flush=True)
    for row in validation.to_dict(orient="records"):
        val_canonical.append(_copy_canonical_row(row, "validation", release))

    train_frame = pd.DataFrame(train_canonical + randomized_rows)
    validation_frame = pd.DataFrame(val_canonical)
    manifest = pd.concat([train_frame, validation_frame], ignore_index=True)
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Week 8 release contains duplicate sample IDs")
    manifest.to_csv(release / "manifest.csv", index=False)
    splits = release / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(splits / "train.csv", index=False)
    validation_frame.to_csv(splits / "validation.csv", index=False)
    train_frame.loc[train_frame["intervention_id"] == "canonical"].to_csv(splits / "train__canonical.csv", index=False)
    train_frame.loc[train_frame["intervention_id"] == "domain-randomized-v1"].to_csv(splits / "train__randomized.csv", index=False)
    validation_frame.to_csv(splits / "validation__canonical.csv", index=False)
    pd.DataFrame(columns=manifest.columns).to_csv(splits / "test.csv", index=False)

    errors = []
    for row in manifest.to_dict(orient="records"):
        row_errors = validate_row(row)
        if row_errors:
            errors.append({"sample_id": row["sample_id"], "errors": row_errors})
    if errors:
        _json(release / "schema_errors.json", errors[:50])
        raise ValueError(f"Week 8 release has invalid schema rows: {errors[:2]}")

    return _validate_randomized_release(config, release)


def _validate_randomized_release(config: dict[str, Any], release: Path) -> dict[str, Any]:
    train_frame = pd.read_csv(release / "splits" / "train.csv")
    validation_frame = pd.read_csv(release / "splits" / "validation.csv")
    manifest = pd.read_csv(release / "manifest.csv")
    errors = [
        {"sample_id": row["sample_id"], "errors": row_errors}
        for row in manifest.to_dict(orient="records")
        if (row_errors := validate_row(row))
    ]
    if errors:
        raise ValueError(f"Week 8 release has invalid schema rows: {errors[:2]}")
    for split, expected in (("train", 1400), ("validation", 150)):
        dataset = LensDataset(
            release / "manifest.csv",
            root=release,
            split=split,
            strict_products=True,
            domain="simulated",
            eligible_for="ml",
            target_fields={"theta_E_arcsec"},
            expected_shape=None,
        )
        if len(dataset) != expected:
            raise ValueError(f"Strict loader found {len(dataset)} rows in {split}; expected {expected}")

    split_ancestry = train_frame.groupby("parent_system_id")["split"].nunique()
    if len(split_ancestry) != 700 or (split_ancestry > 1).any():
        raise ValueError("Week 8 training descendants do not preserve one split per parent")
    if train_frame["parent_system_id"].nunique() != 700 or validation_frame["parent_system_id"].nunique() != 150:
        raise ValueError("Week 8 train/validation parent counts changed")
    assignment_path = release / "randomization_assignment.csv"
    draws = pd.read_csv(assignment_path)
    summary = {
        "release": release.as_posix(),
        "source_train_split_sha256": file_sha256(CANONICAL_RELEASE / "splits" / "train.csv"),
        "source_validation_split_sha256": file_sha256(CANONICAL_RELEASE / "splits" / "validation.csv"),
        "randomization_assignment_sha256": file_sha256(assignment_path),
        "manifest_sha256": file_sha256(release / "manifest.csv"),
        "train_row_count": int(len(train_frame)),
        "train_unique_parent_count": int(train_frame["parent_system_id"].nunique()),
        "validation_row_count": int(len(validation_frame)),
        "pixel_scale_counts": {str(k): int(v) for k, v in draws["pixel_scale_arcsec"].value_counts().sort_index().items()},
        "exposure_multiplier_counts": {str(k): int(v) for k, v in draws["exposure_time_multiplier"].value_counts().sort_index().items()},
        "read_noise_multiplier_counts": {str(k): int(v) for k, v in draws["read_noise_multiplier"].value_counts().sort_index().items()},
        "strict_loader_passed": True,
        "schema_validation_passed": True,
        "ancestry_passed": True,
        "historical_test_accessed": False,
    }
    _json(release / "run_manifest.json", summary)
    return summary


def train_randomized_models(config: dict[str, Any], release_summary: dict[str, Any], output: Path, *, device: str) -> list[dict[str, Any]]:
    baseline_config = yaml.safe_load(Path(config["training"]["baseline_config"]).read_text(encoding="utf-8"))
    summaries = []
    for seed in config["training"]["seeds"]:
        train_config = copy.deepcopy(baseline_config)
        train_config["experiment_id"] = f"reliability-domain-randomized-v1-seed-{int(seed)}"
        train_config["parent_baseline_id"] = "theta-e-cnn-seeds-v1"
        train_config["release"] = RANDOMIZED_RELEASE.as_posix()
        train_config["seed"] = int(seed)
        model_output = Path("results/ml/reliability-domain-randomized-v1") / f"seed-{int(seed)}"
        train_config["output"] = model_output.as_posix()
        result = _train_randomized_seed(train_config, model_output, device_request=device)
        summaries.append(result)
    return summaries


class _ShapePreservingDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], statistics: dict[str, Any]):
        self.records = records
        self.statistics = statistics

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        image = torch.from_numpy(np.ascontiguousarray(row["image"], dtype=np.float32)).unsqueeze(0)
        image = (image - float(self.statistics["image_mean"])) / float(self.statistics["image_std"])
        target = (float(row["theta_E_arcsec"]) - float(self.statistics["target_mean"])) / float(self.statistics["target_std"])
        return {
            "image": image,
            "target": torch.tensor(target, dtype=torch.float32),
            "target_raw": torch.tensor(float(row["theta_E_arcsec"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
        }


def _raw_records(dataset: TorchLensDataset) -> list[dict[str, Any]]:
    metadata = dataset.dataset.metadata.iloc[dataset.indices].reset_index(drop=True)
    records = []
    for index in range(len(dataset)):
        image, target, sample_id = dataset.raw_item(index)
        row = metadata.iloc[index]
        records.append(
            {
                "image": np.ascontiguousarray(image, dtype=np.float32),
                "image_shape": tuple(int(value) for value in image.shape),
                "theta_E_arcsec": float(target),
                "sample_id": str(sample_id),
                "parent_sample_id": str(row.get("parent_sample_id", sample_id)),
            }
        )
    return records


def _train_randomized_seed(config: dict[str, Any], output_dir: Path, *, device_request: str) -> dict[str, Any]:
    import time

    if output_dir.exists() and (output_dir / "training_summary.json").exists():
        return json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    resumed = bool(
        output_dir.exists()
        and (output_dir / "best.pt").exists()
        and (output_dir / "history.csv").exists()
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not resumed:
        raise FileExistsError(f"Week 8 model output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seed_everything(int(config["seed"]))
    device = select_device(device_request)
    release = Path(config["release"])
    raw_train = TorchLensDataset(release, split="train", target_field="theta_E_arcsec", expected_shape=None)
    raw_validation = TorchLensDataset(release, split="validation", target_field="theta_E_arcsec", expected_shape=None)
    statistics = compute_training_statistics(raw_train)
    train_records = _raw_records(raw_train)
    validation_records = _raw_records(raw_validation)
    train_data = _ShapePreservingDataset(train_records, statistics)
    validation_data = _ShapePreservingDataset(validation_records, statistics)
    batch_sampler = ShapeBucketBatchSampler(
        train_records,
        batch_size=int(config["batch_size"]),
        seed=int(config["seed"]),
    )
    train_loader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers=0)
    train_eval_sampler = ShapeBucketBatchSampler(train_records, batch_size=int(config["batch_size"]), seed=0)
    train_eval_sampler.set_epoch(0)
    train_eval_loader = DataLoader(train_data, batch_sampler=train_eval_sampler, num_workers=0)
    validation_loader = make_loader(validation_data, batch_size=config["batch_size"], shuffle=False, seed=0, num_workers=0)
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_function = build_loss({"kind": "huber", "delta": float(config.get("huber_delta", 1.0))})
    split_hashes = {
        split: file_sha256(release / "splits" / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    write_json(output_dir / "resolved_config.json", config)
    history = []
    best_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = output_dir / "best.pt"
    if resumed:
        history = pd.read_csv(output_dir / "history.csv").to_dict(orient="records")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        best_epoch = int(checkpoint["best_epoch"])
        best_mae = float(checkpoint["validation_metrics"]["mae_arcsec"])
    else:
        for epoch in range(1, int(config["epochs"]) + 1):
            batch_sampler.set_epoch(epoch)
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device)
            validation = collect_predictions(model, validation_loader, device, statistics, loss_function=loss_function)
            metrics = regression_metrics(validation["targets"], validation["predictions"])
            row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": validation["normalized_loss"],
                "validation_mae_arcsec": metrics["mae_arcsec"],
                "validation_rmse_arcsec": metrics["rmse_arcsec"],
                "validation_r2": metrics["r2"],
            }
            history.append(row)
            print(json.dumps({"seed": config["seed"], **row}, sort_keys=True), flush=True)
            if metrics["mae_arcsec"] < best_mae:
                best_mae = float(metrics["mae_arcsec"])
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "format_version": 1,
                        "experiment_id": config["experiment_id"],
                        "parent_baseline_id": config.get("parent_baseline_id"),
                        "seed": int(config["seed"]),
                        "config_sha256": config_sha256(config),
                        "model_state": model.state_dict(),
                        "model_config": model.model_config,
                        "statistics": statistics,
                        "target_field": "theta_E_arcsec",
                        "release": release.as_posix(),
                        "release_manifest_sha256": statistics["release_manifest_sha256"],
                        "split_manifest_sha256": split_hashes,
                        "best_epoch": best_epoch,
                        "validation_metrics": metrics,
                        "torch_version": str(torch.__version__),
                        "training_parent_count": 700,
                        "training_record_count": len(train_records),
                        "shape_bucketed_batches": True,
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
            if stale_epochs >= int(config["patience"]):
                break

        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    plot_training_history(history, output_dir / "training_curves.png")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    best_training = collect_predictions(model, train_eval_loader, device, statistics)
    best_validation = collect_predictions(model, validation_loader, device, statistics)
    train_metrics = regression_metrics(best_training["targets"], best_training["predictions"])
    val_metrics = regression_metrics(best_validation["targets"], best_validation["predictions"])
    train_predictions = pd.DataFrame(
        {
            "sample_id": best_training["sample_ids"],
            "theta_E_arcsec_true": best_training["targets"],
            "theta_E_arcsec_predicted": best_training["predictions"],
        }
    )
    train_predictions.to_csv(output_dir / "training_predictions.csv", index=False)
    val_predictions = pd.DataFrame(
        {
            "sample_id": best_validation["sample_ids"],
            "theta_E_arcsec_true": best_validation["targets"],
            "theta_E_arcsec_predicted": best_validation["predictions"],
        }
    )
    val_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    plot_predictions(
        best_validation["targets"],
        best_validation["predictions"],
        output_dir / "validation_predictions.png",
        title="Week 8 canonical validation",
    )
    baseline = mean_baseline_metrics(float(statistics["target_mean"]), best_validation["targets"])
    summary = {
        "experiment_id": config["experiment_id"],
        "parent_baseline_id": config.get("parent_baseline_id"),
        "seed": int(config["seed"]),
        "status": "trained_validation_selected",
        "config_sha256": config_sha256(config),
        "environment": environment_record(device),
        "provenance": source_provenance(),
        "release_id": release.name,
        "release_manifest_sha256": file_sha256(release / "manifest.csv"),
        "train_samples": len(train_data),
        "training_parent_count": 700,
        "training_metrics": train_metrics,
        "validation_samples": len(validation_data),
        "validation_metrics": val_metrics,
        "validation_target_bins": target_bin_metrics(
            best_validation["targets"], best_validation["predictions"], config.get("target_bin_edges_arcsec", [0.5, 1.0, 1.5, 2.0])
        ),
        "mean_baseline_metrics": baseline,
        "validation_mae_improvement_over_mean": improvement_fraction(val_metrics["mae_arcsec"], baseline["mae_arcsec"]),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "trainable_parameters": model.trainable_parameter_count,
        "statistics": statistics,
        "split_manifest_sha256": split_hashes,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "test_samples_accessed": 0,
        "runtime_seconds": None if resumed else time.perf_counter() - started,
        "resumed_from_best_checkpoint": resumed,
    }
    write_json(output_dir / "normalization.json", statistics)
    write_json(output_dir / "training_summary.json", summary)
    return summary


def _evaluation_manifests(release: Path) -> dict[str, tuple[Path, Path, str]]:
    return {
        "canonical": (release, release / "splits" / "validation.csv", ""),
        "pixel-scale-004": (PIXEL_RELEASE, PIXEL_RELEASE / "splits" / "validation__pixel-scale-004.csv", "pixel-scale-004"),
        "pixel-scale-006": (PIXEL_RELEASE, PIXEL_RELEASE / "splits" / "validation__pixel-scale-006.csv", "pixel-scale-006"),
        "exposure-half": (OBSERVATION_RELEASE, OBSERVATION_RELEASE / "splits" / "validation__exposure-half.csv", "exposure-half"),
        "read-noise-double": (OBSERVATION_RELEASE, OBSERVATION_RELEASE / "splits" / "validation__read-noise-double.csv", "read-noise-double"),
    }


def _predict_randomized(config: dict[str, Any], release: Path, output: Path, *, device_request: str, batch_size: int) -> pd.DataFrame:
    device = select_device(device_request)
    rows = []
    for condition, (image_release, manifest_path, intervention) in _evaluation_manifests(release).items():
        manifest = pd.read_csv(
            manifest_path,
            usecols=[column for column in ("sample_id", "parent_sample_id", "processed_path") if column in pd.read_csv(manifest_path, nrows=0).columns],
        )
        if "parent_sample_id" not in manifest:
            manifest["parent_sample_id"] = manifest["sample_id"].astype(str)
        if len(manifest) != 150:
            raise ValueError(f"{condition}: expected exactly 150 validation observations")
        image_arrays = [np.load(image_release / str(path)).astype(np.float32, copy=False) for path in manifest["processed_path"]]
        values = np.stack(image_arrays, axis=0)[:, None, :, :]
        for seed in config["training"]["seeds"]:
            checkpoint_path = Path("results/ml/reliability-domain-randomized-v1") / f"seed-{int(seed)}" / "best.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model = build_model(checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            stats = checkpoint["statistics"]
            normalized_images = (values - float(stats["image_mean"])) / float(stats["image_std"])
            predicted = []
            with torch.inference_mode():
                for start in range(0, len(normalized_images), int(batch_size)):
                    batch = torch.from_numpy(normalized_images[start : start + int(batch_size)]).to(device)
                    normalized = model(batch).detach().cpu().numpy().astype(np.float64)
                    predicted.extend((normalized * float(stats["target_std"]) + float(stats["target_mean"])).tolist())
            for row, prediction in zip(manifest.to_dict(orient="records"), predicted):
                rows.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "parent_sample_id": str(row["parent_sample_id"]),
                        "condition": condition,
                        "seed": int(seed),
                        "theta_E_arcsec_predicted": float(prediction),
                        "checkpoint_sha256": file_sha256(checkpoint_path),
                    }
                )
    predictions = pd.DataFrame(rows)
    if len(predictions) != 2250 or predictions.duplicated(["condition", "parent_sample_id", "seed"]).any():
        raise ValueError("Week 8 predictions are incomplete or duplicated")
    predictions.to_csv(output / "mitigation_predictions_before_scoring.csv", index=False)
    return predictions


def _baseline_predictions() -> pd.DataFrame:
    frames = []
    for seed in (12001, 12002, 12003):
        source = pd.read_csv(BASELINE_MODEL_ROOT / f"seed-{seed}" / "validation_predictions.csv")
        frames.append(
            source[["sample_id", "theta_E_arcsec_predicted"]].assign(
                parent_sample_id=source["sample_id"].astype(str),
                condition="canonical",
                seed=seed,
            )[["parent_sample_id", "condition", "seed", "theta_E_arcsec_predicted"]]
        )
    pixels = pd.read_csv(PIXEL_PREDICTIONS)
    for condition in ("pixel-scale-004", "pixel-scale-006"):
        group = pixels.loc[pixels["intervention_id"] == condition]
        seed_values = group["checkpoint_id"].astype(str).str.extract(r"seed-(\d+)$", expand=False).astype(int)
        frames.append(
            group[["parent_sample_id", "theta_E_arcsec_predicted"]].assign(
                condition=condition,
                seed=seed_values.to_numpy(),
            )[["parent_sample_id", "condition", "seed", "theta_E_arcsec_predicted"]]
        )
    observations = pd.read_csv(OBSERVATION_PREDICTIONS)
    for condition in ("exposure-half", "read-noise-double"):
        group = observations.loc[observations["intervention_id"] == condition]
        frames.append(
            group[["parent_sample_id", "seed", "theta_E_arcsec_predicted"]].assign(condition=condition)
        )
    result = pd.concat(frames, ignore_index=True).rename(
        columns={"theta_E_arcsec_predicted": "baseline_theta_E_arcsec_predicted"}
    )
    return result


def _bootstrap(values: np.ndarray, count: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(count, dtype=np.float64)
    for start in range(0, count, 256):
        n = min(256, count - start)
        indices = rng.integers(0, len(values), size=(n, len(values)))
        means[start : start + n] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def score_mitigation(config: dict[str, Any], predictions: pd.DataFrame, output: Path) -> dict[str, Any]:
    # Targets are joined only after mitigation_predictions_before_scoring.csv exists.
    truth = pd.read_csv(CANONICAL_RELEASE / "splits" / "validation.csv", usecols=["sample_id", "theta_E_arcsec"])
    truth = truth.rename(columns={"sample_id": "parent_sample_id", "theta_E_arcsec": "theta_E_arcsec_true"})
    baseline = _baseline_predictions()
    scored = predictions.merge(truth, on="parent_sample_id", validate="many_to_one")
    scored = scored.merge(baseline, on=["parent_sample_id", "condition", "seed"], validate="one_to_one")
    scored["randomized_absolute_error_arcsec"] = (scored["theta_E_arcsec_predicted"] - scored["theta_E_arcsec_true"]).abs()
    scored["baseline_absolute_error_arcsec"] = (
        scored["baseline_theta_E_arcsec_predicted"] - scored["theta_E_arcsec_true"]
    ).abs()
    scored["delta_absolute_error_arcsec"] = (
        scored["randomized_absolute_error_arcsec"] - scored["baseline_absolute_error_arcsec"]
    )
    scored["prediction_shift_arcsec"] = (
        scored["theta_E_arcsec_predicted"] - scored["baseline_theta_E_arcsec_predicted"]
    )
    scored.to_csv(output / "mitigation_scored_predictions.csv", index=False)

    per_seed = []
    condition_effects = []
    condition_summary = []
    parent_condition = scored.groupby(["condition", "parent_sample_id"], as_index=False).agg(
        theta_E_arcsec_true=("theta_E_arcsec_true", "first"),
        delta_error=("delta_absolute_error_arcsec", "mean"),
        prediction_shift=("prediction_shift_arcsec", "mean"),
        randomized_prediction=("theta_E_arcsec_predicted", "mean"),
        baseline_prediction=("baseline_theta_E_arcsec_predicted", "mean"),
    )
    count = int(config["evaluation"]["bootstrap_resamples"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    for condition, group in scored.groupby("condition", sort=False):
        for model_name, prediction_field in (
            ("canonical-trained", "baseline_theta_E_arcsec_predicted"),
            ("domain-randomized", "theta_E_arcsec_predicted"),
        ):
            for model_seed, seed_group in group.groupby("seed", sort=False):
                per_seed.append(
                    {
                        "condition": condition,
                        "method": model_name,
                        "seed": int(model_seed),
                        **regression_metrics(seed_group["theta_E_arcsec_true"], seed_group[prediction_field]),
                    }
                )
        parent = parent_condition.loc[parent_condition["condition"] == condition]
        lo, hi = _bootstrap(parent["delta_error"].to_numpy(), count, seed)
        condition_effects.append(
            {
                "condition": condition,
                "parent_count": int(len(parent)),
                "paired_delta_mae_arcsec": float(parent["delta_error"].mean()),
                "bootstrap_95_low_arcsec": lo,
                "bootstrap_95_high_arcsec": hi,
                "mean_absolute_prediction_shift_arcsec": float(parent["prediction_shift"].abs().mean()),
                "mean_prediction_shift_arcsec": float(parent["prediction_shift"].mean()),
            }
        )
        for method, pred_field in (
            ("canonical-trained", "baseline_prediction"),
            ("domain-randomized", "randomized_prediction"),
        ):
            condition_summary.append(
                {
                    "condition": condition,
                    "method": method,
                    "parent_count": int(len(parent)),
                    **regression_metrics(parent["theta_E_arcsec_true"], parent[pred_field]),
                    "p90_absolute_error_arcsec": float(
                        np.quantile(np.abs(parent[pred_field] - parent["theta_E_arcsec_true"]), 0.90)
                    ),
                    "p95_absolute_error_arcsec": float(
                        np.quantile(np.abs(parent[pred_field] - parent["theta_E_arcsec_true"]), 0.95)
                    ),
                }
            )
    per_seed_frame = pd.DataFrame(per_seed)
    effect_frame = pd.DataFrame(condition_effects)
    summary_frame = pd.DataFrame(condition_summary)
    per_seed_frame.to_csv(output / "metrics_per_seed.csv", index=False)
    effect_frame.to_csv(output / "paired_effects.csv", index=False)
    summary_frame.to_csv(output / "condition_metrics.csv", index=False)
    parent_condition.to_csv(output / "paired_parent_effects.csv", index=False)

    stress = parent_condition.loc[parent_condition["condition"].isin(STRESS_CONDITIONS)]
    composite = stress.groupby("parent_sample_id", as_index=False).agg(
        mean_stress_delta_mae=("delta_error", "mean"),
        mean_stress_prediction_shift=("prediction_shift", "mean"),
    )
    composite_low, composite_high = _bootstrap(composite["mean_stress_delta_mae"].to_numpy(), count, seed)
    canonical = effect_frame.loc[effect_frame["condition"] == "canonical"].iloc[0]
    practical = float(config["evaluation"]["practical_effect_arcsec"])
    stress_pass = composite["mean_stress_delta_mae"].mean() <= -practical and composite_high < 0.0
    canonical_noninferior = canonical["bootstrap_95_high_arcsec"] <= practical
    decision = {
        "stress_parent_count": int(len(composite)),
        "stress_condition_count": len(STRESS_CONDITIONS),
        "mean_stress_delta_mae_arcsec": float(composite["mean_stress_delta_mae"].mean()),
        "stress_bootstrap_95_low_arcsec": composite_low,
        "stress_bootstrap_95_high_arcsec": composite_high,
        "stress_rule_passed": bool(stress_pass),
        "canonical_delta_mae_arcsec": float(canonical["paired_delta_mae_arcsec"]),
        "canonical_bootstrap_95_high_arcsec": float(canonical["bootstrap_95_high_arcsec"]),
        "canonical_noninferiority_passed": bool(canonical_noninferior),
        "overall_mitigation_passed": bool(stress_pass and canonical_noninferior),
        "conditional_scale_metadata_extension_required": bool(not (stress_pass and canonical_noninferior)),
        "practical_effect_arcsec": practical,
    }
    write_json(output / "mitigation_decision.json", decision)
    composite.to_csv(output / "composite_stress_parent_effects.csv", index=False)

    plot_data = effect_frame.loc[effect_frame["condition"].isin(STRESS_CONDITIONS)].copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(plot_data))
    mean = plot_data["paired_delta_mae_arcsec"].to_numpy()
    lower = plot_data["bootstrap_95_low_arcsec"].to_numpy()
    upper = plot_data["bootstrap_95_high_arcsec"].to_numpy()
    ax.errorbar(x, mean, yerr=[mean - lower, upper - mean], fmt="o", capsize=4, color="#26547c")
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(-practical, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x, plot_data["condition"], rotation=18, ha="right")
    ax.set_ylabel("Domain-randomized minus baseline MAE (arcsec)")
    ax.set_title("Week 8 paired robustness effects")
    fig.tight_layout()
    fig.savefig(output / "paired_effects.png", dpi=170)
    plt.close(fig)

    summary = {
        "experiment_id": "WEEK8-TARGETED-RANDOMIZATION",
        "protocol": "archive/docs/reliability_study/protocol_v1.8.md",
        "protocol_sha256": file_sha256(Path("archive/docs/reliability_study/protocol_v1.8.md")),
        "training_release_manifest_sha256": file_sha256(RANDOMIZED_RELEASE / "manifest.csv"),
        "training_parent_count": 700,
        "training_row_count": 1400,
        "validation_parent_count": 150,
        "seeds": [12001, 12002, 12003],
        "bootstrap_resamples": count,
        "bootstrap_seed": seed,
        "historical_test_accessed": False,
        "real_data_labels_accessed": False,
        "decision": decision,
        "paired_effects": condition_effects,
        "prediction_file_written_before_target_join": (output / "mitigation_predictions_before_scoring.csv").exists(),
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# Week 8 — targeted nuisance randomization",
        "",
        "A matched validation comparison of the frozen CNN ensemble and a three-seed CNN retrained with one deterministic nuisance-randomized view for each original training parent. The training release contains 1,400 rows from 700 independent parents.",
        "",
        f"- Composite stress MAE change: **{decision['mean_stress_delta_mae_arcsec']:+.4f} arcsec** ({decision['stress_bootstrap_95_low_arcsec']:+.4f}, {decision['stress_bootstrap_95_high_arcsec']:+.4f})",
        f"- Canonical MAE change: **{decision['canonical_delta_mae_arcsec']:+.4f} arcsec** (upper 95% bound {decision['canonical_bootstrap_95_high_arcsec']:+.4f})",
        f"- Stress rule passed: **{'yes' if stress_pass else 'no'}**",
        f"- Canonical non-inferiority passed: **{'yes' if canonical_noninferior else 'no'}**",
        "- Historical synthetic test and real-data labels accessed: **no**",
        "",
        "## Paired condition effects",
        "",
        "| Condition | Paired MAE change (arcsec) | Parent bootstrap 95% interval | Mean absolute prediction shift |",
        "|---|---:|---:|---:|",
    ]
    for row in condition_effects:
        lines.append(
            f"| {row['condition']} | {row['paired_delta_mae_arcsec']:+.4f} | "
            f"[{row['bootstrap_95_low_arcsec']:+.4f}, {row['bootstrap_95_high_arcsec']:+.4f}] | "
            f"{row['mean_absolute_prediction_shift_arcsec']:.4f} |"
        )
    lines.extend(
        [
            "",
            "![Paired stress effects](paired_effects.png)",
            "",
            "The randomized view draws pixel scale from 0.04/0.05/0.06 arcsec per pixel, exposure multiplier from 0.5/1.0, and read-noise multiplier from 1/2 using seed 6031. The original canonical training view remains in the training set. The randomized sample is an augmentation of the same parent, not a new independent lens system.",
            "",
            "The study only supports claims about these synthetic nuisance ranges and the frozen CNN. A pass does not establish simulation-to-real transfer. If the overall rule failed, the protocol permits one scale-metadata diagnostic; no other architecture search is allowed.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run_week8(config_path: str | Path = DEFAULT_CONFIG, *, device: str = "auto", batch_size: int = 16) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    release = Path(config["randomized_release"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    release_summary = create_randomized_release(config, release)
    training_summaries = train_randomized_models(config, release_summary, output, device=device)
    _json(output / "training_run_summaries.json", training_summaries)
    predictions = _predict_randomized(config, release, output, device_request=device, batch_size=batch_size)
    summary = score_mitigation(config, predictions, output)
    summary["release_integrity"] = release_summary
    write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run_week8(args.config, device=args.device, batch_size=args.batch_size), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
