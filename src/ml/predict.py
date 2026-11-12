"""Target-free primary and paired-renderer checkpoint predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.dataset import PairedEvaluationImages
from src.ml.artifacts import sha256, verify_resolved_config
from src.ml.data import grouped_forward
from src.ml.model import build_model
from src.simulation.observation import CONDITIONS as PAIRED_CONDITIONS
from src.simulation.observation import RENDERERS


def _prediction_for_checkpoint(
    checkpoint_path: Path, rows: pd.DataFrame, config: dict[str, Any]
) -> np.ndarray:
    holdout_path = Path(config["holdout"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    stats = checkpoint["statistics"]
    images = []
    # Intentionally access only the image path, sample identity and condition.
    for value in rows["processed_path"].astype(str):
        array = np.load(holdout_path.parent.parent / value, allow_pickle=False)
        if array.ndim != 2 or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Invalid target-free holdout image: {value}")
        images.append(array)
    predictions = []
    batch_size = config["training"]["batch_size"]
    for start in range(0, len(images), batch_size):
        _, output = grouped_forward(model, images[start : start + batch_size], stats)
        predictions.extend(
            (output.detach().numpy() * stats["target_std"] + stats["target_mean"]).tolist()
        )
    return np.asarray(predictions, dtype=float)


def predict(config: dict[str, Any]) -> None:
    """Write primary predictions without reading the target column."""
    output_root, holdout_path = Path(config["output"]), Path(config["holdout"])
    conditions = config["conditions"]
    verify_resolved_config(config)
    fit_manifest = json.loads((output_root / "fit_complete.json").read_text(encoding="utf-8"))
    if fit_manifest["target_labels_accessed"] or fit_manifest["final_images_accessed"]:
        raise ValueError("Fit manifest records forbidden test access")
    prediction_path = output_root / "predictions_before_scoring.csv"
    if prediction_path.exists():
        print(json.dumps({"stage": "predictions_already_sealed", "path": str(prediction_path)}))
        return
    # The target column is explicitly excluded from this read.
    rows = pd.read_csv(
        holdout_path, usecols=["sample_id", "parent_sample_id", "condition", "processed_path"]
    )
    if len(rows) != 750 or rows["sample_id"].duplicated().any():
        raise ValueError("Unexpected holdout identity")
    if set(rows["condition"]) != set(conditions):
        raise ValueError("Unexpected holdout conditions")
    outputs = []
    for record in fit_manifest["fits"]:
        frame = rows.copy()
        frame["arm"] = record["arm"]
        frame["seed"] = record["seed"]
        frame["theta_E_arcsec_predicted"] = _prediction_for_checkpoint(
            Path(record["checkpoint"]), rows, config
        )
        outputs.append(
            frame[
                [
                    "sample_id",
                    "parent_sample_id",
                    "condition",
                    "arm",
                    "seed",
                    "theta_E_arcsec_predicted",
                ]
            ]
        )
    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(prediction_path, index=False)
    seal = {
        "schema": "equal_budget_predictions_v1",
        "prediction_path": str(prediction_path),
        "rows": len(result),
        "target_labels_accessed": False,
        "sealed_before_target_join": True,
    }
    with (output_root / "predictions_complete.json").open("x", encoding="utf-8") as stream:
        json.dump(seal, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {"stage": "predictions_sealed", "rows": len(result), "target_labels_accessed": False}
        )
    )


PAIRED_ROOT = Path("results/ml/cross-simulator-rapid-v2")
METHOD = PAIRED_ROOT / "method_freeze.json"
PAIRED_RELEASE = Path("data/simulations/cross-simulator-rapid-v2/manifest.json")
PAIRED_PREDICTIONS = PAIRED_ROOT / "predictions_before_scoring.csv"
PAIRED_ARMS = ("C", "G", "R")
PAIRED_SEEDS = tuple(730001 + i for i in range(10))


def predict_paired() -> None:
    """Predict both renderer views before opening paired truth targets."""
    if PAIRED_PREDICTIONS.exists():
        print(json.dumps({"stage": "predictions_already_sealed", "path": str(PAIRED_PREDICTIONS)}))
        return
    images = PairedEvaluationImages(
        PAIRED_RELEASE, role="power_pilot", freeze_sha256=sha256(METHOD)
    )
    images.verify_all_products()
    records = []
    for arm in PAIRED_ARMS:
        for seed in PAIRED_SEEDS:
            checkpoint = torch.load(_checkpoint(arm, seed), map_location="cpu", weights_only=True)
            model = build_model(checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            stats = checkpoint["statistics"]
            for renderer in RENDERERS:
                for condition in PAIRED_CONDITIONS:
                    values = []
                    for start in range(0, len(images.parent_ids), 16):
                        ids = images.parent_ids[start : start + 16]
                        arrays = [images.image(parent, renderer, condition) for parent in ids]
                        _, output = grouped_forward(model, arrays, stats)
                        values.extend(
                            (
                                output.detach().numpy() * stats["target_std"] + stats["target_mean"]
                            ).tolist()
                        )
                    records.extend(
                        {
                            "parent_system_id": parent,
                            "renderer": renderer,
                            "condition": condition,
                            "arm": arm,
                            "seed": seed,
                            "theta_E_arcsec_predicted": float(value),
                        }
                        for parent, value in zip(images.parent_ids, values)
                    )
    frame = pd.DataFrame.from_records(records)
    expected = (
        len(images.parent_ids)
        * len(RENDERERS)
        * len(PAIRED_CONDITIONS)
        * len(PAIRED_ARMS)
        * len(PAIRED_SEEDS)
    )
    if len(frame) != expected or not np.isfinite(frame["theta_E_arcsec_predicted"]).all():
        raise ValueError("Prediction dimensions or finite values differ")
    frame.to_csv(PAIRED_PREDICTIONS, index=False)
    seal = {
        "schema": "cross_simulator_rapid_predictions_v1",
        "prediction_path": str(PAIRED_PREDICTIONS),
        "prediction_sha256": sha256(PAIRED_PREDICTIONS),
        "release_manifest_sha256": sha256(PAIRED_RELEASE),
        "rows": len(frame),
        "truth_opened": False,
    }
    (PAIRED_ROOT / "predictions_complete.json").write_text(
        json.dumps(seal, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": "predictions_sealed", "rows": len(frame), "truth_opened": False}))


def _checkpoint(arm, seed):
    return Path(f"results/ml/equal-budget-final-v1/fits/{arm}/seed-{seed}/final.pt")
