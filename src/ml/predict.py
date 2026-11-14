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


@torch.inference_mode()
def predict_images(
    checkpoint_path: Path, images: list[np.ndarray], *, batch_size: int
) -> np.ndarray:
    """Predict physical radii from native images using checkpoint-only statistics.

    The caller supplies images, never references or fitted HST normalization.
    Image ordering is preserved across the native-shape groups.
    """
    if type(batch_size) is not int or batch_size < 1 or not images:
        raise ValueError("Inference needs images and a positive integer batch size")
    for array in images:
        if array.ndim != 2 or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("Inference images must be finite two-dimensional float32 arrays")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    stats = checkpoint["statistics"]
    if (
        not np.isfinite(list(stats.values())).all()
        or min(stats["image_std"], stats["target_std"]) <= 0
    ):
        raise ValueError("Checkpoint normalization must be finite with positive scales")
    predictions = []
    for start in range(0, len(images), batch_size):
        _, output = grouped_forward(model, images[start : start + batch_size], stats)
        predictions.extend((output.numpy() * stats["target_std"] + stats["target_mean"]).tolist())
    result = np.asarray(predictions, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Checkpoint produced non-finite radius predictions")
    return result


def _prediction_for_checkpoint(
    checkpoint_path: Path, rows: pd.DataFrame, config: dict[str, Any]
) -> np.ndarray:
    holdout_path = Path(config["holdout"])
    images = []
    # Intentionally access only the image path, sample identity and condition.
    for value in rows["processed_path"].astype(str):
        array = np.load(holdout_path.parent.parent / value, allow_pickle=False)
        if array.ndim != 2 or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Invalid target-free holdout image: {value}")
        images.append(array)
    return predict_images(checkpoint_path, images, batch_size=config["training"]["batch_size"])


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


def predict_hst(config: dict[str, Any], cohort: str) -> None:
    """Seal complete paired HST predictions without opening classical references."""
    from src.data.hst import load_prepared, verify_confirmation
    from src.ml.artifacts import (
        create_once,
        environment_record,
        load_config,
        seed_everything,
        verified_hst_assets,
    )

    output = Path(config["output"])
    destination = output / cohort
    prediction_path = destination / "predictions_before_scoring.csv"
    if prediction_path.exists() or (destination / "predictions_complete.json").exists():
        raise FileExistsError("HST predictions already exist; analyze saved predictions")
    if cohort == "confirmation":
        verify_confirmation(config, images=True)
    assets_path = output / "model_assets.json"
    assets = verified_hst_assets(assets_path)
    rows, images = load_prepared(config, cohort)
    study = load_config(config["study_config"])
    torch.set_num_threads(study["training"]["torch_threads"])
    seed_everything(0)
    records = []
    for checkpoint in assets["checkpoints"]:
        predictions = predict_images(
            Path(checkpoint["path"]), images, batch_size=config["batch_size"]
        )
        for row, value in zip(rows, predictions, strict=True):
            records.append(
                {
                    "system_id": row["system_id"],
                    "observation_id": row["observation_id"],
                    "survey": row["survey"],
                    "band": row["band"],
                    "arm": checkpoint["arm"],
                    "seed": checkpoint["seed"],
                    "theta_E_arcsec_predicted": float(value),
                    "nonpositive_prediction": bool(value <= 0),
                }
            )
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(prediction_path, index=False, mode="x")
    create_once(
        destination / "predictions_complete.json",
        {
            "schema": "hst_predictions_v1",
            "cohort": cohort,
            "config": config,
            "arms": assets["arms"],
            "seeds": assets["seeds"],
            "rows": len(records),
            "systems": [row["system_id"] for row in rows],
            "observation_ids": [row["observation_id"] for row in rows],
            "prediction_sha256": sha256(prediction_path),
            "assets_sha256": sha256(assets_path),
            "prepared_manifest_sha256": sha256(
                Path(config["prepared_root"]) / cohort / "manifest.json"
            ),
            "references_opened": False,
            "environment": environment_record(torch.device("cpu")),
            "sealed_before_reference_join": True,
        },
    )
