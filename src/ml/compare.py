"""Compare validation suites using matched seeds and verified predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.data import file_sha256
from src.ml.engine import require_output_directory, write_json
from src.ml.metrics import regression_metrics, target_bin_metrics


def load_predictions(root):
    root = Path(root)
    report = json.loads((root / "aggregate_validation.json").read_text())
    if report["status"] != "completed_validation_benchmark" or report["test_samples_accessed"] != 0:
        raise ValueError("Comparison requires full validation-only suites")
    frames = {}
    for run in report["runs"]:
        path = root / f"seed-{run['seed']}" / "validation_predictions.csv"
        if file_sha256(path) != run["validation_predictions_sha256"]:
            raise ValueError("Prediction hash mismatch")
        frame = pd.read_csv(path).sort_values("sample_id").reset_index(drop=True)
        if frame.sample_id.duplicated().any():
            raise ValueError("Duplicate prediction IDs")
        frames[run["seed"]] = frame
    return report, frames


def compare_suites(reference, candidate, output):
    ref, ref_frames = load_predictions(reference)
    cand, cand_frames = load_predictions(candidate)
    for field in ("release_manifest_sha256", "split_manifest_sha256", "seeds", "train_samples_per_seed", "validation_samples_per_seed"):
        if ref[field] != cand[field]:
            raise ValueError(f"Comparison identity mismatch: {field}")
    rows = []
    for seed, frame in ref_frames.items():
        other = cand_frames[seed]
        if not frame[["sample_id", "theta_E_arcsec_true"]].equals(other[["sample_id", "theta_E_arcsec_true"]]):
            raise ValueError("Prediction samples or targets differ")
        pair = []
        for predictions in (frame, other):
            truth = predictions.theta_E_arcsec_true.to_numpy()
            pred = predictions.theta_E_arcsec_predicted.to_numpy()
            metrics = regression_metrics(truth, pred)
            metrics["out_of_prior_fraction"] = float(np.mean((pred < 0.5) | (pred > 2.0)))
            bins = target_bin_metrics(truth, pred, [0.5, 1.0, 1.5, 2.0])
            metrics["worst_radius_bin_absolute_bias_arcsec"] = max(abs(b["metrics"]["bias_arcsec"]) for b in bins if b["metrics"])
            pair.append(metrics)
        for metric in pair[0]:
            if metric != "count":
                a, b = pair[0][metric], pair[1][metric]
                rows.append(dict(seed=seed, metric=metric, reference=a, candidate=b, delta=None if a is None or b is None else b-a))
    table = pd.DataFrame(rows)
    averages = table.groupby("metric")[["reference", "candidate", "delta"]].mean()
    # Predeclared conservative guards: no increase in mean tails or worst-bin bias.
    passed = (averages.loc["mae_arcsec", "delta"] <= -0.01
              and averages.loc["p95_absolute_error_arcsec", "delta"] <= 0
              and averages.loc["worst_radius_bin_absolute_bias_arcsec", "delta"] <= 0
              and (table.loc[table.metric == "mae_arcsec", "delta"] < 0).all())
    output = require_output_directory(output)
    table.to_csv(output / "paired_metrics.csv", index=False)
    summary = dict(reference=ref["experiment_id"], candidate=cand["experiment_id"], passes_gate=bool(passed),
                   test_samples_accessed=0, metrics=averages.to_dict(orient="index"),
                   gate="MAE improves >=0.01 arcsec; P95 and mean worst-bin absolute bias do not increase; all seeds improve MAE.")
    write_json(output / "comparison.json", summary)
    lines = [f"# {cand['experiment_id']} versus {ref['experiment_id']}", "", summary["gate"], "",
             f"Passes conservative development gate: {bool(passed)}. Validation only; this is not a significance test.", "",
             "Calibration fits predicted radius = slope * true radius + intercept. Prior bounds are fixed at [0.5, 2.0] arcsec.", "",
             "| Metric | Reference mean | Candidate mean | Paired mean delta |", "|---|---:|---:|---:|"]
    for name, row in averages.iterrows():
        lines.append(f"| {name} | {row.reference:.6f} | {row.candidate:.6f} | {row.delta:+.6f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare_suites(args.reference, args.candidate, args.output)


if __name__ == "__main__":
    main()
