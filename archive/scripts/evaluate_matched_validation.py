"""Evaluate final fixed-update models on existing source validation stresses.

No frozen final holdout or alternate-simulator data is read. This is
descriptive development evidence and must never be relabeled a final test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.ml.data import file_sha256
from src.ml.engine import write_json
from src.ml.model import build_model


def run(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    if config["status"] != "source_validation_development_only":
        raise ValueError("This evaluator is only for source development")
    root, output = Path(config["training_root"]), Path(config["output"])
    training_report = json.loads((root / "report.json").read_text())
    if training_report["status"] != config["status"]:
        raise ValueError("Mismatched training scope")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output already exists")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "started.json", {"config": config, "config_sha256": file_sha256(config_path)})
    torch.set_num_threads(config["torch_threads"])
    truth = pd.read_csv(config["canonical_truth"], usecols=["sample_id", "theta_E_arcsec"])
    truth = truth.rename(columns={"sample_id": "parent_sample_id"})
    parents = set(truth.parent_sample_id)
    controls, run_summaries = {}, {}
    for seed in config["seeds"]:
        for arm in ("canonical", "randomized"):
            summary = json.loads((root / f"seed-{seed}" / arm / "summary.json").read_text())
            run_summaries[seed, arm] = summary
            controls.setdefault(seed, summary["run_controls"])
            if summary["run_controls"] != controls[seed]:
                raise ValueError("Unmatched training controls")
    prediction_rows, hashes = [], {}
    for condition, entry in config["conditions"].items():
        manifest = pd.read_csv(entry["manifest"], usecols=lambda c: c in {
            "sample_id", "parent_sample_id", "processed_path"})
        if "parent_sample_id" not in manifest:
            manifest["parent_sample_id"] = manifest["sample_id"]
        if len(manifest) != len(parents) or set(manifest.parent_sample_id) != parents:
            raise ValueError("Validation condition does not match the canonical parent cohort")
        hashes[entry["manifest"]] = file_sha256(entry["manifest"])
        arrays = []
        for value in manifest.processed_path:
            path = Path(entry["release"]) / str(value)
            arrays.append(np.load(path).astype(np.float32))
            hashes[str(path)] = file_sha256(path)
        values = np.stack(arrays)[:, None]
        for seed in config["seeds"]:
            for arm in ("canonical", "randomized"):
                checkpoint_path = root / f"seed-{seed}" / arm / "final.pt"
                if file_sha256(checkpoint_path) != run_summaries[seed, arm]["checkpoint_sha256"]:
                    raise ValueError("Checkpoint hash changed")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if (checkpoint["optimizer_updates"] != controls[seed]["optimizer_updates"]
                    or checkpoint["arm"] != arm or checkpoint["seed"] != seed):
                    raise ValueError("Checkpoint update count changed")
                model = build_model(checkpoint["model_config"])
                model.load_state_dict(checkpoint["model_state"])
                model.eval()
                stats = checkpoint["statistics"]
                normalized = (values - stats["image_mean"]) / stats["image_std"]
                predictions = []
                with torch.inference_mode():
                    for start in range(0, len(values), config["batch_size"]):
                        out = model(torch.from_numpy(normalized[start:start + config["batch_size"]]))
                        predictions.extend((out.numpy() * stats["target_std"] + stats["target_mean"]).tolist())
                for parent, prediction in zip(manifest.parent_sample_id, predictions):
                    prediction_rows.append({"condition": condition, "parent_sample_id": parent,
                                            "seed": seed, "arm": arm,
                                            "theta_E_arcsec_predicted": prediction})
                hashes[str(checkpoint_path)] = file_sha256(checkpoint_path)
        print(f"Completed source validation condition: {condition}", flush=True)
    predictions = pd.DataFrame(prediction_rows)
    if predictions.duplicated(["condition", "parent_sample_id", "seed", "arm"]).any():
        raise ValueError("Duplicate predictions")
    prediction_path = output / "predictions_before_scoring.csv"
    predictions.to_csv(prediction_path, index=False)
    write_json(output / "prediction_identity.json", {"predictions_sha256": file_sha256(prediction_path),
                                                     "input_hashes": hashes})
    ensemble = predictions.groupby(["condition", "parent_sample_id", "arm"], as_index=False).theta_E_arcsec_predicted.mean()
    scored = ensemble.merge(truth, on="parent_sample_id", validate="many_to_one")
    scored["absolute_error_arcsec"] = abs(scored.theta_E_arcsec_predicted - scored.theta_E_arcsec)
    scored["signed_error_arcsec"] = scored.theta_E_arcsec_predicted - scored.theta_E_arcsec
    scored.to_csv(output / "ensemble_scored.csv", index=False)
    seed_scored = predictions.merge(truth, on="parent_sample_id", validate="many_to_one")
    seed_scored["absolute_error_arcsec"] = abs(seed_scored.theta_E_arcsec_predicted - seed_scored.theta_E_arcsec)
    seed_metrics = seed_scored.groupby(["seed", "condition", "arm"]).absolute_error_arcsec.mean().reset_index()
    seed_metrics.to_csv(output / "per_seed_condition_metrics.csv", index=False)
    seed_composite = seed_metrics.loc[seed_metrics.condition != "canonical"].groupby(
        ["seed", "arm"]).absolute_error_arcsec.mean().unstack("arm")
    seed_composite["delta_randomized_minus_canonical_arcsec"] = seed_composite.randomized - seed_composite.canonical
    seed_composite.to_csv(output / "per_seed_stress_composite.csv")
    metrics = scored.groupby(["condition", "arm"]).agg(
        mae_arcsec=("absolute_error_arcsec", "mean"), bias_arcsec=("signed_error_arcsec", "mean"),
        p95_arcsec=("absolute_error_arcsec", lambda s: s.quantile(.95))).reset_index()
    metrics.to_csv(output / "ensemble_metrics.csv", index=False)
    wide = scored.pivot(index=["condition", "parent_sample_id"], columns="arm", values="absolute_error_arcsec")
    wide["delta"] = wide.randomized - wide.canonical
    stress = wide.loc[wide.index.get_level_values("condition") != "canonical"].groupby("parent_sample_id").delta.mean().to_numpy()
    rng = np.random.default_rng(config["bootstrap_seed"])
    means = stress[rng.integers(0, len(stress), size=(config["bootstrap_resamples"], len(stress)))].mean(axis=1)
    interval = np.quantile(means, [.025, .975]).tolist()
    result = {"status": config["status"], "independent_validation_parents": len(parents),
              "conditions": len(config["conditions"]), "checkpoint_predictions": len(predictions),
              "stress_paired_delta_mae_arcsec": float(stress.mean()),
              "descriptive_parent_bootstrap_95_interval": interval,
              "test_images_accessed": 0, "target_simulator_images_accessed": 0,
              "interpretation": "Source validation only; not a new independent final result"}
    write_json(output / "summary.json", result)
    lines = ["# Matched-budget source-validation stress comparison", "",
             result["interpretation"], "", "| Condition | Arm | MAE (arcsec) | Bias | P95 |",
             "|---|---|---:|---:|---:|"]
    for row in metrics.to_dict("records"):
        lines.append(f"| {row['condition']} | {row['arm']} | {row['mae_arcsec']:.6f} | {row['bias_arcsec']:+.6f} | {row['p95_arcsec']:.6f} |")
    lines += ["", f"Equal-weight four-stress paired MAE change (randomized minus canonical): {stress.mean():+.6f} arcsec.",
              f"Descriptive parent-bootstrap 95% interval: [{interval[0]:+.6f}, {interval[1]:+.6f}].", "",
              "Both arms used final 880-update checkpoints, shared normalization and matched parent exposures. These development parents have been used previously; the interval is conditional on the fixed ensemble and this validation cohort. It does not include training-seed uncertainty or establish final cross-simulator accuracy.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml/matched_budget_validation_v1.yaml")
    args = parser.parse_args()
    run(args.config)
