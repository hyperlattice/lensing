"""Summarize saved source-validation runs without loading or rerunning models."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd


def summarize(root, output):
    root, output = Path(root), Path(output)
    report = json.loads((root / "report.json").read_text())
    rows, predictions = [], {"canonical": [], "randomized": []}
    for pair in sorted(root.glob("seed-*")):
        summaries = [json.loads((pair / arm / "summary.json").read_text())
                     for arm in predictions]
        if summaries[0]["run_controls"] != summaries[1]["run_controls"]:
            raise ValueError("Training controls differ between arms")
        for arm, summary in zip(predictions, summaries):
            folder = pair / arm
            for filename, key in [("final.pt", "checkpoint_sha256"),
                                  ("final_validation_predictions.csv", "predictions_sha256")]:
                if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != summary[key]:
                    raise ValueError(f"Artifact changed: {folder / filename}")
            frame = pd.read_csv(folder / "final_validation_predictions.csv").sort_values("sample_id")
            predictions[arm].append(frame)
            rows.append({"seed": summary["seed"], "arm": arm,
                         "updates": summary["run_controls"]["optimizer_updates"],
                         "image_exposures": summary["run_controls"]["image_exposures"],
                         "pixels_processed": summary["pixels_processed"],
                         "validation_mae_arcsec": summary["primary_validation_metrics"]["mae_arcsec"],
                         "validation_p95_arcsec": summary["primary_validation_metrics"]["p95_absolute_error_arcsec"],
                         "runtime_seconds": summary["runtime_seconds"]})
    frames = [frame for arm in predictions.values() for frame in arm]
    if len(frames) != len(report["runs"]):
        raise ValueError("Missing run predictions")
    reference = frames[0][["sample_id", "theta_E_arcsec_true"]].reset_index(drop=True)
    if any(not f[["sample_id", "theta_E_arcsec_true"]].reset_index(drop=True).equals(reference) for f in frames):
        raise ValueError("Validation targets or identities differ")
    ensembles = {arm: np.mean([frame.theta_E_arcsec_predicted.to_numpy() for frame in fs], axis=0)
                 for arm, fs in predictions.items()}
    truth = reference.theta_E_arcsec_true.to_numpy()
    errors = {arm: np.abs(values - truth) for arm, values in ensembles.items()}
    # Descriptive validation comparison only. Selection/development reuse means
    # this is not an unbiased final generalization estimate.
    paired = reference.copy()
    for arm in predictions:
        paired[f"{arm}_predicted_arcsec"] = ensembles[arm]
        paired[f"{arm}_absolute_error_arcsec"] = errors[arm]
    paired["delta_error_randomized_minus_canonical_arcsec"] = errors["randomized"] - errors["canonical"]
    summary = {"status": "source_validation_development_only", "seed_count": len(predictions["canonical"]),
               "validation_parent_count": len(reference), "matched_controls_verified": True,
               "ensemble_mae_arcsec": {arm: float(e.mean()) for arm, e in errors.items()},
               "ensemble_paired_delta_mae_arcsec": float((errors["randomized"] - errors["canonical"]).mean()),
               "test_images_accessed": 0, "target_simulator_images_accessed": 0,
               "limitations": report["limitations"]}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Summary output already exists")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "per_seed.csv", index=False)
    paired.to_csv(output / "paired_source_validation.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Matched-budget source validation", "",
             "Development evidence only: this is canonical source validation, not final stress or cross-simulator accuracy.", "",
             "Both arms used 880 updates, 16 examples per update, identical parent exposure schedules and shared canonical-training normalization.", "",
             "| Seed | Arm | Updates | Validation MAE (arcsec) | P95 (arcsec) | Runtime (s) |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['arm']} | {row['updates']} | {row['validation_mae_arcsec']:.6f} | {row['validation_p95_arcsec']:.6f} | {row['runtime_seconds']:.1f} |")
    lines += ["", f"Three-seed ensemble canonical MAE: {errors['canonical'].mean():.6f} arcsec.",
              f"Three-seed ensemble randomized MAE: {errors['randomized'].mean():.6f} arcsec.",
              f"Paired change (randomized minus canonical): {summary['ensemble_paired_delta_mae_arcsec']:+.6f} arcsec.", "",
              "Training controls and saved prediction/checkpoint hashes were verified. Native pixel counts and runtime differ; see per_seed.csv.", "",
              "These results do not replace the original 27.4% stress-holdout result and cannot show superiority to a published method. A fresh frozen paired A/B-renderer evaluation is still required.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/ml/theta-e-matched-budget-v1")
    parser.add_argument("--output", default="results/reports/matched-budget-v1")
    args = parser.parse_args()
    summarize(args.root, args.output)
