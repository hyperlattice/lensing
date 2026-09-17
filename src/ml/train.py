"""Train the first image-only Einstein-radius CNN baseline."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn

from src.ml.data import TorchLensDataset, compute_training_statistics, file_sha256
from src.ml.engine import (
    collect_predictions,
    make_loader,
    require_output_directory,
    seed_everything,
    select_device,
    train_one_epoch,
    write_json,
)
from src.ml.metrics import (
    improvement_fraction, mean_baseline_metrics, regression_metrics, target_bin_metrics, validate_bin_edges,
)
from src.ml.model import EinsteinRadiusCNN
from src.ml.plots import plot_predictions, plot_training_history
from src.ml.provenance import config_sha256, environment_record, source_provenance


DEFAULT_CONFIG = Path("configs/ml_baseline_v1.yaml")


def load_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment_id",
        "release",
        "output",
        "target_field",
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "patience",
        "model",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Experiment config is missing: {sorted(missing)}")
    return config


def train_experiment(
    config,
    *,
    output_override=None,
    device_request="auto",
    overwrite=False,
    max_train_samples=None,
    max_validation_samples=None,
):
    started = time.perf_counter()
    config = copy.deepcopy(config)
    config["output"] = Path(output_override or config["output"]).as_posix()
    for field in ("epochs", "batch_size", "patience"):
        if isinstance(config[field], bool) or not isinstance(config[field], int) or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if config["target_field"] != "theta_E_arcsec":
        raise ValueError("This baseline supports only theta_E_arcsec")
    bin_edges = validate_bin_edges(config.get("target_bin_edges_arcsec", [0.5, 1.0, 1.5, 2.0]))
    config["target_bin_edges_arcsec"] = bin_edges
    controls = {"max_train_samples": max_train_samples, "max_validation_samples": max_validation_samples}
    for field, value in controls.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{field} must be a positive integer")
    provenance = source_provenance()
    seed_everything(config["seed"])
    device = select_device(device_request)
    output_dir = require_output_directory(output_override or config["output"], overwrite=overwrite)
    write_json(output_dir / "resolved_config.json", config)
    release = Path(config["release"])
    raw_training = TorchLensDataset(
        release,
        split="train",
        target_field=config["target_field"],
        max_samples=max_train_samples,
    )
    statistics = compute_training_statistics(raw_training)
    training = TorchLensDataset(
        release,
        split="train",
        target_field=config["target_field"],
        statistics=statistics,
        max_samples=max_train_samples,
    )
    validation = TorchLensDataset(
        release,
        split="validation",
        target_field=config["target_field"],
        statistics=statistics,
        max_samples=max_validation_samples,
    )
    training_loader = make_loader(
        training,
        batch_size=config["batch_size"],
        shuffle=True,
        seed=config["seed"],
        num_workers=config.get("num_workers", 0),
    )
    validation_loader = make_loader(
        validation,
        batch_size=config["batch_size"],
        shuffle=False,
        seed=config["seed"],
        num_workers=config.get("num_workers", 0),
    )
    model = EinsteinRadiusCNN(**config["model"]).to(device)
    split_manifest_sha256 = {
        split: file_sha256(release / "splits" / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_function = nn.HuberLoss(delta=float(config.get("huber_delta", 1.0)))
    history = []
    best_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = time.perf_counter()
        training_loss = train_one_epoch(
            model, training_loader, optimizer, loss_function, device
        )
        validation_result = collect_predictions(
            model, validation_loader, device, statistics, loss_function=loss_function
        )
        validation_metrics = regression_metrics(
            validation_result["targets"], validation_result["predictions"]
        )
        row = {
            "epoch": epoch,
            "train_loss": training_loss,
            "validation_loss": validation_result["normalized_loss"],
            "validation_mae_arcsec": validation_metrics["mae_arcsec"],
            "validation_rmse_arcsec": validation_metrics["rmse_arcsec"],
            "validation_r2": validation_metrics["r2"],
            "runtime_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if validation_metrics["mae_arcsec"] < best_mae:
            best_mae = validation_metrics["mae_arcsec"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "format_version": 1,
                    "experiment_id": config["experiment_id"],
                    "seed": config["seed"],
                    "config_sha256": config_sha256(config),
                    "model_state": model.state_dict(),
                    "model_config": model.model_config,
                    "statistics": statistics,
                    "target_field": config["target_field"],
                    "release": release.as_posix(),
                    "release_manifest_sha256": statistics["release_manifest_sha256"],
                    "split_manifest_sha256": split_manifest_sha256,
                    "best_epoch": best_epoch,
                    "validation_metrics": validation_metrics,
                    "torch_version": str(torch.__version__),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(config["patience"]):
            break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    plot_training_history(history, output_dir / "training_curves.png")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    best_validation = collect_predictions(model, validation_loader, device, statistics)
    best_metrics = regression_metrics(
        best_validation["targets"], best_validation["predictions"]
    )
    baseline = mean_baseline_metrics(statistics["target_mean"], best_validation["targets"])
    predictions = pd.DataFrame(
        {
            "sample_id": best_validation["sample_ids"],
            "theta_E_arcsec_true": best_validation["targets"],
            "theta_E_arcsec_predicted": best_validation["predictions"],
        }
    )
    predictions["residual_arcsec"] = (
        predictions["theta_E_arcsec_predicted"] - predictions["theta_E_arcsec_true"]
    )
    predictions["absolute_error_arcsec"] = predictions["residual_arcsec"].abs()
    predictions["relative_error"] = predictions["absolute_error_arcsec"] / predictions["theta_E_arcsec_true"].abs()
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    plot_predictions(
        best_validation["targets"],
        best_validation["predictions"],
        output_dir / "validation_predictions.png",
        title="Validation split: best checkpoint",
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "parent_baseline_id": config.get("parent_baseline_id"),
        "seed": config["seed"],
        "config_sha256": config_sha256(config),
        "run_controls": controls,
        "environment": environment_record(device),
        "provenance": provenance,
        "release_id": release.name,
        "release_manifest_sha256": statistics["release_manifest_sha256"],
        "status": "trained_validation_only",
        "device": str(device),
        "torch_version": str(torch.__version__),
        "train_samples": len(training),
        "validation_samples": len(validation),
        "test_samples_accessed": 0,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "trainable_parameters": model.trainable_parameter_count,
        "statistics": statistics,
        "split_manifest_sha256": split_manifest_sha256,
        "validation_metrics": best_metrics,
        "validation_target_bins": target_bin_metrics(
            best_validation["targets"], best_validation["predictions"], bin_edges
        ),
        "mean_baseline_metrics": baseline,
        "validation_mae_improvement_over_mean": improvement_fraction(
            best_metrics["mae_arcsec"], baseline["mae_arcsec"]
        ),
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "validation_predictions_sha256": file_sha256(output_dir / "validation_predictions.csv"),
        "runtime_seconds": time.perf_counter() - started,
        "scientific_scope": "in-domain synthetic validation only; test and real data not accessed",
    }
    write_json(output_dir / "normalization.json", statistics)
    write_json(output_dir / "training_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-validation-samples", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    summary = train_experiment(
        config,
        output_override=args.output,
        device_request=args.device,
        overwrite=args.overwrite,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
