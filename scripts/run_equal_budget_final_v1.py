"""Equal-budget C/G/R final fit, target-free prediction, and one-way scoring.

This rapid confirmatory package uses the already frozen 150-parent blind
holdout. The fit and prediction actions never read its target column. Scoring
opens the target only after predictions have been sealed. C is canonical
training, G is uniform observing-condition variation, and R is the frozen
failure-weighted source-only policy.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.review_robustness_development_v1 import checked_fit
from src.data.benchmark_contract_v1 import sha256
from src.data.observation_bank_v1 import ObservationBank
from src.ml.model import build_model
from src.ml.robustness_training_v1 import grouped_forward


DEFAULT_CONFIG = Path("configs/ml/equal_budget.yaml")


def load_config(path=DEFAULT_CONFIG):
    """Read settings only; never inspect data banks, checkpoints or holdout."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "output", "source_train", "source_development",
                "policy", "holdout", "seeds", "arms", "conditions", "bootstrap", "training"}
    if not isinstance(config, dict) or set(config) != required or config["schema_version"] != 1:
        raise ValueError("Unexpected equal-budget configuration schema")
    if config["arms"] != ["C", "G", "R"] or config["conditions"] != [
            "canonical", "pixel-scale-004", "pixel-scale-006", "exposure-half", "read-noise-double"]:
        raise ValueError("This runner requires the fixed C/G/R observation matrix")
    seeds = config["seeds"]
    if not seeds or any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Expected unique integer seeds")
    controls = config["bootstrap"]
    if set(controls) != {"resamples", "stress_seed", "canonical_seed"} or any(
            type(value) is not int or value < 1 for value in controls.values()):
        raise ValueError("Expected positive integer bootstrap controls")
    return config


def verify_resolved_config(config):
    """Require downstream stages to use the configuration sealed at fit time."""
    path = Path(config["output"]) / "resolved_config.json"
    resolved = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    if resolved != {"config": config, "config_sha256": digest}:
        raise ValueError("Equal-budget configuration changed; use a new output identity")


def fit(config):
    ROOT = Path(config["output"])
    SOURCE, DEVELOPMENT = Path(config["source_train"]), Path(config["source_development"])
    POLICY, HOLDOUT = Path(config["policy"]), Path(config["holdout"])
    SEEDS, ARMS = config["seeds"], config["arms"]
    if (ROOT / "fit_complete.json").exists():
        verify_resolved_config(config)
        print(json.dumps(json.loads((ROOT / "fit_complete.json").read_text()), indent=2))
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    source = ObservationBank(SOURCE, expected_role="source_train")
    development = ObservationBank(DEVELOPMENT, expected_role="source_development")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    settings = config["training"]
    resolved = {"config": config, "config_sha256": hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()}
    resolved_path = ROOT / "resolved_config.json"
    if resolved_path.exists():
        if json.loads(resolved_path.read_text(encoding="utf-8")) != resolved:
            raise ValueError("Equal-budget configuration changed; use a new output identity")
    else:
        with resolved_path.open("x", encoding="utf-8") as stream:
            json.dump(resolved, stream, indent=2)
            stream.write("\n")
    records = []
    for seed in SEEDS:
        for arm in ARMS:
            folder = ROOT / "fits" / arm / f"seed-{seed}"
            report = checked_fit(folder, settings, seed=seed, arm=arm,
                                 source=source, development=development,
                                 target=None, probabilities=policy["probabilities"])
            records.append({"arm": arm, "seed": seed, "folder": str(folder),
                            "checkpoint": str(folder / "final.pt"),
                            "checkpoint_sha256": report["checkpoint_sha256"],
                            "target_labels_accessed": False,
                            "final_images_accessed": False})
    manifest = {
        "schema": "equal_budget_final_v1",
        "stage": "fit_complete_target_free",
        "arms": list(ARMS), "seeds": list(SEEDS),
        "optimizer_updates": settings["optimizer_updates"],
        "batch_size": settings["batch_size"],
        "source_bank": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
        "development_bank": {"path": str(DEVELOPMENT), "sha256": sha256(DEVELOPMENT)},
        "policy": {"path": str(POLICY), "sha256": sha256(POLICY)},
        "settings": settings,
        "holdout_manifest_not_read": str(HOLDOUT),
        "fits": records,
        "target_labels_accessed": False,
        "final_images_accessed": False,
    }
    with (ROOT / "fit_complete.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"stage": "fit_complete", "fits": len(records),
                      "target_labels_accessed": False}))


def _prediction_for_checkpoint(checkpoint_path: Path, rows: pd.DataFrame, config):
    HOLDOUT = Path(config["holdout"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    stats = checkpoint["statistics"]
    images = []
    # Intentionally access only the image path, sample identity and condition.
    for value in rows["processed_path"].astype(str):
        array = np.load(HOLDOUT.parent.parent / value, allow_pickle=False)
        if array.ndim != 2 or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Invalid target-free holdout image: {value}")
        images.append(array)
    predictions = []
    batch_size = config["training"]["batch_size"]
    for start in range(0, len(images), batch_size):
        _, output = grouped_forward(model, images[start:start + batch_size], stats)
        predictions.extend((output.detach().numpy() * stats["target_std"] + stats["target_mean"]).tolist())
    return np.asarray(predictions, dtype=float)


def predict(config):
    ROOT, HOLDOUT = Path(config["output"]), Path(config["holdout"])
    CONDITIONS = config["conditions"]
    verify_resolved_config(config)
    fit_manifest = json.loads((ROOT / "fit_complete.json").read_text(encoding="utf-8"))
    if fit_manifest["target_labels_accessed"] or fit_manifest["final_images_accessed"]:
        raise ValueError("Fit manifest records forbidden test access")
    prediction_path = ROOT / "predictions_before_scoring.csv"
    if prediction_path.exists():
        print(json.dumps({"stage": "predictions_already_sealed", "path": str(prediction_path)}))
        return
    # The target column is explicitly excluded from this read.
    rows = pd.read_csv(HOLDOUT, usecols=["sample_id", "parent_sample_id", "condition", "processed_path"])
    if len(rows) != 750 or rows["sample_id"].duplicated().any():
        raise ValueError("Unexpected holdout identity")
    if set(rows["condition"]) != set(CONDITIONS):
        raise ValueError("Unexpected holdout conditions")
    outputs = []
    by_arm = {record["arm"]: [] for record in fit_manifest["fits"]}
    for record in fit_manifest["fits"]:
        frame = rows.copy()
        frame["arm"] = record["arm"]
        frame["seed"] = record["seed"]
        frame["theta_E_arcsec_predicted"] = _prediction_for_checkpoint(Path(record["checkpoint"]), rows, config)
        outputs.append(frame[["sample_id", "parent_sample_id", "condition", "arm", "seed",
                              "theta_E_arcsec_predicted"]])
    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(prediction_path, index=False)
    seal = {
        "schema": "equal_budget_predictions_v1",
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "holdout_manifest_sha256": sha256(HOLDOUT),
        "rows": len(result), "target_labels_accessed": False,
        "sealed_before_target_join": True,
    }
    with (ROOT / "predictions_complete.json").open("x", encoding="utf-8") as stream:
        json.dump(seal, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"stage": "predictions_sealed", "rows": len(result),
                      "target_labels_accessed": False}))


def _bootstrap_difference(values_a, values_b, seed=8731, draws=10000):
    rng = np.random.default_rng(seed)
    n = values_a.shape[1]
    indices = rng.integers(0, n, size=(draws, n))
    diffs = (values_a[:, indices].mean(axis=(0, 2)) -
             values_b[:, indices].mean(axis=(0, 2)))
    return {"difference_arcsec": float((values_a - values_b).mean()),
            "bootstrap_95_low_arcsec": float(np.quantile(diffs, 0.025)),
            "bootstrap_95_high_arcsec": float(np.quantile(diffs, 0.975)),
            "bootstrap_seed": seed, "bootstrap_resamples": draws}


def score(config):
    ROOT, HOLDOUT = Path(config["output"]), Path(config["holdout"])
    SEEDS, ARMS, CONDITIONS = config["seeds"], config["arms"], config["conditions"]
    STRESS = CONDITIONS[1:]
    verify_resolved_config(config)
    seal = json.loads((ROOT / "predictions_complete.json").read_text(encoding="utf-8"))
    if sha256(seal["prediction_path"]) != seal["prediction_sha256"]:
        raise ValueError("Prediction seal changed")
    predictions = pd.read_csv(seal["prediction_path"])
    # This is the one-way target join. No score is written before this point.
    truth = pd.read_csv(HOLDOUT, usecols=["sample_id", "parent_sample_id", "condition", "theta_E_arcsec"])
    joined = predictions.merge(truth, on=["sample_id", "parent_sample_id", "condition"], validate="many_to_one")
    if len(joined) != len(predictions) or not np.isfinite(joined["theta_E_arcsec"]).all():
        raise ValueError("Prediction/truth join is incomplete")
    joined["absolute_error_arcsec"] = abs(joined["theta_E_arcsec_predicted"] - joined["theta_E_arcsec"])
    metrics = {}
    for arm in ARMS:
        arm_frame = joined[joined["arm"] == arm]
        metrics[arm] = {}
        for condition in CONDITIONS:
            subset = arm_frame[arm_frame["condition"] == condition]
            metrics[arm][condition] = {
                "mae_arcsec": float(subset["absolute_error_arcsec"].mean()),
                "rmse_arcsec": float(np.sqrt(np.mean(np.square(subset["theta_E_arcsec_predicted"] - subset["theta_E_arcsec"])))),
                "bias_arcsec": float((subset["theta_E_arcsec_predicted"] - subset["theta_E_arcsec"]).mean()),
                "count_per_seed": int(len(subset) // len(SEEDS)),
            }
    parent_order = sorted(truth["parent_sample_id"].unique())
    parent_index = {value: i for i, value in enumerate(parent_order)}
    four_stress = {}
    canonical = {}
    per_seed_stress = {}
    for arm in ARMS:
        arr = np.empty((len(SEEDS), len(parent_order), len(STRESS)), dtype=float)
        can = np.empty((len(SEEDS), len(parent_order)), dtype=float)
        for si, seed in enumerate(SEEDS):
            frame = joined[(joined["arm"] == arm) & (joined["seed"] == seed)]
            for pi, parent in enumerate(parent_order):
                row = frame[frame["parent_sample_id"] == parent]
                can[si, pi] = row[row["condition"] == "canonical"]["absolute_error_arcsec"].iloc[0]
                arr[si, pi] = [row[row["condition"] == c]["absolute_error_arcsec"].iloc[0] for c in STRESS]
        four_stress[arm] = arr
        canonical[arm] = can
        per_seed_stress[arm] = arr.mean(axis=(1, 2)).tolist()
    contrasts = {}
    for candidate, comparator in (("G", "C"), ("R", "G"), ("R", "C")):
        contrasts[f"{candidate}_vs_{comparator}_four_stress"] = _bootstrap_difference(
            four_stress[candidate], four_stress[comparator],
            seed=config["bootstrap"]["stress_seed"], draws=config["bootstrap"]["resamples"])
        contrasts[f"{candidate}_vs_{comparator}_canonical"] = _bootstrap_difference(
            canonical[candidate][:, :, None], canonical[comparator][:, :, None],
            seed=config["bootstrap"]["canonical_seed"], draws=config["bootstrap"]["resamples"])
    report = {
        "schema": "equal_budget_final_report_v1",
        "stage": "scored_after_target_join",
        "holdout_manifest_sha256": sha256(HOLDOUT),
        "parent_count": len(parent_order), "seed_count": len(SEEDS),
        "seeds": list(SEEDS), "conditions": list(CONDITIONS),
        "arms": list(ARMS), "optimizer_updates": config["training"]["optimizer_updates"],
        "metrics_by_arm_condition": metrics,
        "per_seed_four_stress_mae_arcsec": per_seed_stress,
        "contrasts": contrasts,
        "limitations": [
            "The frozen final release is the existing synthetic blind holdout, not real data.",
            "The holdout was previously scored by older models; these equal-budget checkpoints were fit without accessing it.",
            "The crossed bootstrap resamples parents with seed runs held fixed.",
            "This compares source-only arms on a common benchmark and does not reproduce a published paper model.",
        ],
    }
    with (ROOT / "final_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    lines = ["# Equal-budget final result", "",
             f"Frozen synthetic holdout: {len(parent_order)} parents; {len(SEEDS)} paired seeds; {config['training']['optimizer_updates']} updates.", "",
             "| Arm | Canonical MAE | Four-stress MAE |", "|---|---:|---:|"]
    for arm in ARMS:
        lines.append(f"| {arm} | {metrics[arm]['canonical']['mae_arcsec']:.6f} | "
                     f"{np.mean([metrics[arm][c]['mae_arcsec'] for c in STRESS]):.6f} |")
    lines.extend(["", "| Contrast | Difference (candidate - comparator) | Bootstrap 95% interval |",
                  "|---|---:|---:|"])
    for name, value in contrasts.items():
        if "four_stress" in name:
            lines.append(f"| {name} | {value['difference_arcsec']:.6f} | "
                         f"[{value['bootstrap_95_low_arcsec']:.6f}, {value['bootstrap_95_high_arcsec']:.6f}] |")
    lines.extend(["", "The result is a source-only, equal-budget synthetic robustness comparison.",
                  "It is not a claim of superiority over published models or real-data accuracy.", ""])
    (ROOT / "result_overview.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"stage": "scored", "report": str(ROOT / "final_report.json"),
                      "four_stress": {name: value for name, value in contrasts.items() if "four_stress" in name}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fit", "predict", "score"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    {"fit": fit, "predict": predict, "score": score}[args.action](load_config(args.config))
