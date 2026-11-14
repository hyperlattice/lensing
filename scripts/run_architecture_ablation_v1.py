"""Run the exploratory compact-versus-reference CNN ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import ObservationBank
from src.ml.data import grouped_forward
from src.ml.model import build_model
from src.ml.train import train_arm
from src.simulation.observation import CELLS

DEFAULT_CONFIG = {
    "schema_version": 1,
    "experiment_id": "architecture-ablation-5seed-v1",
    "output": "results/ml/architecture-ablation-5seed-v1",
    "source_train": "data/simulations/robustness-study-source-train-v4/bank.json",
    "source_development": "data/simulations/robustness-study-source-development-v4/bank.json",
    "holdout": "data/simulations/reliability-blind-holdout-v1/splits/holdout.csv",
    "reference_checkpoint_root": "results/ml/equal-budget-final-v1/fits/G",
    "arm": "G",
    "seeds": [730001, 730002, 730003, 730004, 730005],
    "training": {
        "optimizer_updates": 880,
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "huber_delta": 1.0,
        "source_exposures_per_fit": 14080,
    },
    "architectures": {
        "compact": {"channels": [4, 8, 16, 32], "pooled_size": 4, "hidden_features": 64, "dropout": 0.1},
        "reference": {
            "channels": [8, 16, 32, 64],
            "pooled_size": 4,
            "hidden_features": 128,
            "dropout": 0.1,
        },
    },
}
STRESS = ("pixel-scale-004", "pixel-scale-006", "exposure-half", "read-noise-double")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the small, explicit ablation configuration."""
    config = (
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if path is not None
        else DEFAULT_CONFIG.copy()
    )
    required = {
        "schema_version",
        "experiment_id",
        "output",
        "source_train",
        "source_development",
        "holdout",
        "reference_checkpoint_root",
        "arm",
        "seeds",
        "training",
        "architectures",
    }
    if not isinstance(config, dict) or set(config) != required or config["schema_version"] != 1:
        raise ValueError("Unexpected architecture-ablation configuration")
    if config["arm"] != "G" or not config["seeds"]:
        raise ValueError("This ablation is defined for the uniform-variation G arm")
    if set(config["architectures"]) != {"compact", "reference"}:
        raise ValueError("Expected compact and reference architectures")
    if any(int(seed) != seed for seed in config["seeds"]):
        raise ValueError("Seeds must be integers")
    controls = config["training"]
    if controls["optimizer_updates"] < 1 or controls["batch_size"] < 1:
        raise ValueError("Invalid training budget")
    if controls["source_exposures_per_fit"] != controls["optimizer_updates"] * controls["batch_size"]:
        raise ValueError("Exposure count does not match update budget")
    return config


def checkpoint_path(config: dict[str, Any], architecture: str, seed: int) -> Path:
    """Return the compact fit or the frozen primary-study reference fit."""
    if architecture == "compact":
        return Path(config["output"]) / "fits" / architecture / f"seed-{seed}" / "final.pt"
    return Path(config["reference_checkpoint_root"]) / f"seed-{seed}" / "final.pt"


def fit(config: dict[str, Any]) -> None:
    """Fit compact models without opening holdout images or labels."""
    source = ObservationBank(config["source_train"], expected_role="source_train")
    development = ObservationBank(
        config["source_development"], expected_role="source_development"
    )
    controls = config["training"]
    probabilities = np.full(len(CELLS), 1.0 / len(CELLS))
    for seed in config["seeds"]:
        output = Path(config["output"]) / "fits" / "compact" / f"seed-{seed}"
        if (output / "complete.json").exists():
            continue
        if output.exists():
            raise FileExistsError(f"Incomplete fit directory exists: {output}")
        settings = {
            "fit_purpose": "development_only",
            "optimizer_updates": controls["optimizer_updates"],
            "validation_interval_updates": controls["optimizer_updates"] // 2,
            "batch_size": controls["batch_size"],
            "torch_threads": 6,
            "learning_rate": controls["learning_rate"],
            "weight_decay": controls["weight_decay"],
            "huber_delta": controls["huber_delta"],
            "mmd_weight": 0.0,
            "model": config["architectures"]["compact"],
        }
        train_arm(
            settings,
            seed=seed,
            arm="G",
            source=source,
            development=development,
            target=None,
            output=output,
            probabilities=probabilities,
        )


def predict(config: dict[str, Any]) -> None:
    """Write predictions before reading holdout target labels."""
    output = Path(config["output"])
    prediction_path = output / "predictions_before_scoring.csv"
    if prediction_path.exists():
        return
    holdout_path = Path(config["holdout"])
    rows = pd.read_csv(
        holdout_path,
        usecols=["sample_id", "parent_sample_id", "condition", "processed_path"],
    )
    if len(rows) != 750 or rows["sample_id"].duplicated().any():
        raise ValueError("Unexpected holdout identity")
    root = holdout_path.parent.parent
    images = [np.load(root / value, allow_pickle=False) for value in rows["processed_path"].astype(str)]
    records = []
    for architecture in ("compact", "reference"):
        for seed in config["seeds"]:
            checkpoint = torch.load(
                checkpoint_path(config, architecture, seed), map_location="cpu", weights_only=True
            )
            model = build_model(checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            statistics = checkpoint["statistics"]
            predictions = []
            for start in range(0, len(images), config["training"]["batch_size"]):
                _, output_tensor = grouped_forward(
                    model, images[start : start + config["training"]["batch_size"]], statistics
                )
                predictions.extend(
                    (
                        output_tensor.detach().numpy() * statistics["target_std"]
                        + statistics["target_mean"]
                    ).tolist()
                )
            records.extend(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "sample_id": row.sample_id,
                    "parent_sample_id": row.parent_sample_id,
                    "condition": row.condition,
                    "prediction_arcsec": float(prediction),
                }
                for row, prediction in zip(rows.itertuples(index=False), predictions)
            )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(prediction_path, index=False)
    (output / "predictions_complete.json").write_text(
        json.dumps({"rows": len(records), "target_labels_accessed": False}, indent=2) + "\n",
        encoding="utf-8",
    )


def score(config: dict[str, Any]) -> None:
    """Join the sealed predictions to holdout labels and write the report."""
    output = Path(config["output"])
    predictions = pd.read_csv(output / "predictions_before_scoring.csv")
    holdout = pd.read_csv(
        config["holdout"],
        usecols=["sample_id", "parent_sample_id", "condition", "theta_E_arcsec"],
    )
    joined = predictions.merge(
        holdout,
        on=["sample_id", "parent_sample_id", "condition"],
        validate="many_to_one",
    )
    joined["absolute_error_arcsec"] = abs(joined["prediction_arcsec"] - joined["theta_E_arcsec"])
    joined.to_csv(output / "predictions.csv", index=False)
    metrics = []
    for architecture in ("compact", "reference"):
        subset = joined[joined["architecture"] == architecture]
        for condition, group in subset.groupby("condition", sort=False):
            residual = group["prediction_arcsec"] - group["theta_E_arcsec"]
            metrics.append(
                {
                    "architecture": architecture,
                    "condition": condition,
                    "mae_arcsec": float(group["absolute_error_arcsec"].mean()),
                    "rmse_arcsec": float(np.sqrt(np.mean(np.square(residual)))),
                    "bias_arcsec": float(residual.mean()),
                }
            )
    parents = sorted(joined["parent_sample_id"].unique())
    arrays = {}
    for architecture in ("compact", "reference"):
        values = np.empty((len(config["seeds"]), len(parents), len(STRESS)))
        for seed_index, seed in enumerate(config["seeds"]):
            for parent_index, parent in enumerate(parents):
                subset = joined[
                    (joined["architecture"] == architecture)
                    & (joined["seed"] == seed)
                    & (joined["parent_sample_id"] == parent)
                ]
                values[seed_index, parent_index] = [
                    subset[subset["condition"] == condition]["absolute_error_arcsec"].iloc[0]
                    for condition in STRESS
                ]
        arrays[architecture] = values
        metrics.append(
            {
                "architecture": architecture,
                "condition": "four-stress",
                "mae_arcsec": float(values.mean()),
                "rmse_arcsec": float(np.sqrt(np.mean(np.square(values)))),
                "bias_arcsec": None,
            }
        )
    rng = np.random.default_rng(8731)
    difference = arrays["compact"] - arrays["reference"]
    indices = rng.integers(len(parents), size=(10000, len(parents)))
    bootstrap = difference[:, indices].mean(axis=(0, 2, 3))
    contrast = {
        "compact_minus_reference_four_stress_arcsec": float(difference.mean()),
        "bootstrap_95_low_arcsec": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_95_high_arcsec": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_seed": 8731,
        "bootstrap_resamples": 10000,
    }
    architecture_settings = {}
    for architecture in ("compact", "reference"):
        model = build_model(config["architectures"][architecture])
        architecture_settings[architecture] = {
            **config["architectures"][architecture],
            "trainable_parameters": model.trainable_parameter_count,
        }
    per_seed = {
        architecture: {
            str(seed): float(arrays[architecture][index].mean())
            for index, seed in enumerate(config["seeds"])
        }
        for architecture in ("compact", "reference")
    }
    report = {
        "schema": "architecture_ablation_5seed_v1",
        "arm": "G_uniform_variation",
        "architectures": architecture_settings,
        "parent_count": len(parents),
        "holdout_observation_count": int(joined["sample_id"].nunique()),
        "seed_count": len(config["seeds"]),
        "seeds": config["seeds"],
        "four_stress_mae_by_seed_arcsec": per_seed,
        "metrics": metrics,
        "contrast_four_stress": contrast,
        "limitations": [
            "Exploratory five-seed capacity check; the primary C/G/R result uses ten seeds.",
            "The comparison uses the same synthetic holdout and does not validate real data.",
        ],
    }
    report_root = Path("results/reports/architecture_ablation_v1")
    report_root.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output / "metrics.csv", index=False)
    metrics_frame.to_csv(report_root / "metrics.csv", index=False)
    joined.to_csv(report_root / "predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (report_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    canonical = {
        architecture: next(
            row["mae_arcsec"]
            for row in metrics
            if row["architecture"] == architecture and row["condition"] == "canonical"
        )
        for architecture in ("compact", "reference")
    }
    stress = {
        architecture: float(arrays[architecture].mean())
        for architecture in ("compact", "reference")
    }
    markdown = [
        "# Architecture ablation (five paired seeds)",
        "",
        f"Uniform-variation arm; {len(config['seeds'])} paired seeds; {len(parents)} holdout parents ({joined['sample_id'].nunique()} condition images).",
        "",
        "| Architecture | Parameters | Canonical MAE | Four-stress MAE |",
        "|---|---:|---:|---:|",
        f"| Compact | {architecture_settings['compact']['trainable_parameters']:,} | {canonical['compact']:.6f} | {stress['compact']:.6f} |",
        f"| Reference | {architecture_settings['reference']['trainable_parameters']:,} | {canonical['reference']:.6f} | {stress['reference']:.6f} |",
        "",
        f"Compact minus reference four-stress difference: {contrast['compact_minus_reference_four_stress_arcsec']:.6f} arcsec (parent-bootstrap 95% interval [{contrast['bootstrap_95_low_arcsec']:.6f}, {contrast['bootstrap_95_high_arcsec']:.6f}]).",
        "",
        "This is an exploratory capacity comparison. The primary C/G/R result continues to use ten seeds and a fixed architecture.",
        "",
        "## Four-stress MAE by seed",
        "",
        "| Seed | Compact | Reference |",
        "|---:|---:|---:|",
    ]
    markdown.extend(
        f"| {seed} | {per_seed['compact'][str(seed)]:.6f} | {per_seed['reference'][str(seed)]:.6f} |"
        for seed in config["seeds"]
    )
    markdown.append("")
    (report_root / "report.md").write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "predict", "score", "all"))
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in {"fit", "all"}:
        fit(config)
    if args.stage in {"predict", "all"}:
        predict(config)
    if args.stage in {"score", "all"}:
        score(config)


if __name__ == "__main__":
    main()
