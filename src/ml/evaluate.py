"""Evaluate a trained Einstein-radius checkpoint on one frozen split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.ml.data import TorchLensDataset
from src.ml.engine import collect_predictions, make_loader, require_output_directory, select_device, write_json
from src.ml.metrics import improvement_fraction, mean_baseline_metrics, regression_metrics
from src.ml.model import build_model
from src.ml.plots import plot_predictions


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subgroup_metrics(frame, column):
    results = {}
    for value, group in frame.groupby(column, dropna=False, observed=True):
        if len(group) < 2:
            continue
        results[str(value)] = regression_metrics(
            group["theta_E_arcsec_true"].to_numpy(),
            group["theta_E_arcsec_predicted"].to_numpy(),
        )
    return results


def _interpretation(metrics, baseline, validation_metrics=None):
    improvement = improvement_fraction(metrics["mae_arcsec"], baseline["mae_arcsec"])
    if improvement is None:
        baseline_statement = "The mean baseline has zero error, so improvement is undefined."
    elif improvement > 0:
        baseline_statement = f"The CNN lowers MAE by {improvement:.1%} relative to the train-mean baseline."
    else:
        baseline_statement = f"The CNN is {-improvement:.1%} worse than the train-mean MAE baseline."
    validation_statement = ""
    if validation_metrics:
        validation_mae = float(validation_metrics["mae_arcsec"])
        gap = metrics["mae_arcsec"] - validation_mae
        validation_statement = (
            f" Test MAE differs from the checkpoint validation MAE by {gap:+.4f} arcsec."
        )
    return (
        baseline_statement
        + validation_statement
        + " These numbers measure only the frozen synthetic distribution and do not establish real-HST accuracy."
    )


def evaluate_checkpoint(
    checkpoint_path,
    *,
    split="test",
    output_dir=None,
    device_request="auto",
    batch_size=16,
    num_workers=0,
    overwrite=False,
    max_samples=None,
):
    checkpoint_path = Path(checkpoint_path)
    device = select_device(device_request)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    release = Path(checkpoint["release"])
    manifest_path = release / "manifest.csv"
    expected_hash = str(checkpoint["release_manifest_sha256"])
    actual_hash = _sha256(manifest_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Release manifest hash changed: expected {expected_hash}, found {actual_hash}"
        )
    split_manifest_path = release / "splits" / f"{split}.csv"
    split_manifest_hash = _sha256(split_manifest_path)
    expected_split_hash = checkpoint.get("split_manifest_sha256", {}).get(split)
    if expected_split_hash is not None and split_manifest_hash != expected_split_hash:
        raise ValueError(
            f"{split} split manifest hash changed: expected {expected_split_hash}, "
            f"found {split_manifest_hash}"
        )
    output_dir = require_output_directory(
        output_dir or checkpoint_path.parent / f"evaluation-{split}", overwrite=overwrite
    )
    dataset = TorchLensDataset(
        release,
        split=split,
        target_field=checkpoint["target_field"],
        statistics=checkpoint["statistics"],
        max_samples=max_samples,
    )
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
    )
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    result = collect_predictions(model, loader, device, checkpoint["statistics"])
    metrics = regression_metrics(result["targets"], result["predictions"])
    baseline = mean_baseline_metrics(
        checkpoint["statistics"]["target_mean"], result["targets"]
    )
    frame = pd.DataFrame(
        {
            "sample_id": result["sample_ids"],
            "theta_E_arcsec_true": result["targets"],
            "theta_E_arcsec_predicted": result["predictions"],
        }
    )
    frame["residual_arcsec"] = frame["theta_E_arcsec_predicted"] - frame["theta_E_arcsec_true"]
    frame["absolute_error_arcsec"] = frame["residual_arcsec"].abs()
    frame["relative_error"] = frame["absolute_error_arcsec"] / frame["theta_E_arcsec_true"].abs()
    metadata = dataset.dataset.metadata.iloc[dataset.indices].copy()
    diagnostic_columns = [
        column
        for column in (
            "sample_id",
            "image_count_physical",
            "image_count_detectable",
            "psf_fwhm_arcsec",
        )
        if column in metadata
    ]
    if diagnostic_columns:
        frame = frame.merge(metadata[diagnostic_columns], on="sample_id", how="left", validate="one_to_one")
    if "psf_fwhm_arcsec" in frame:
        frame["psf_width_quartile"] = pd.qcut(
            frame["psf_fwhm_arcsec"], q=4, labels=("Q1", "Q2", "Q3", "Q4"), duplicates="drop"
        )
    frame.to_csv(output_dir / "predictions.csv", index=False)
    plot_predictions(
        result["targets"],
        result["predictions"],
        output_dir / "predictions.png",
        title=f"{split.capitalize()} split: {checkpoint['experiment_id']}",
    )
    subgroup_metrics = {}
    for column in ("image_count_physical", "image_count_detectable", "psf_width_quartile"):
        if column in frame:
            subgroup_metrics[column] = _subgroup_metrics(frame, column)
    worst_examples = (
        frame.nlargest(min(5, len(frame)), "absolute_error_arcsec")[
            [
                column
                for column in (
                    "sample_id",
                    "theta_E_arcsec_true",
                    "theta_E_arcsec_predicted",
                    "residual_arcsec",
                    "relative_error",
                    "image_count_physical",
                    "image_count_detectable",
                    "psf_fwhm_arcsec",
                )
                if column in frame
            ]
        ]
        .to_dict(orient="records")
    )
    interpretation = _interpretation(metrics, baseline, checkpoint.get("validation_metrics"))
    report = {
        "experiment_id": checkpoint["experiment_id"],
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "release_manifest_sha256": actual_hash,
        "split_manifest_sha256": split_manifest_hash,
        "split": split,
        "sample_count": len(dataset),
        "device": str(device),
        "metrics": metrics,
        "train_mean_baseline_metrics": baseline,
        "mae_improvement_over_mean": improvement_fraction(
            metrics["mae_arcsec"], baseline["mae_arcsec"]
        ),
        "checkpoint_validation_metrics": checkpoint.get("validation_metrics"),
        "subgroup_metrics": subgroup_metrics,
        "largest_absolute_error_examples": worst_examples,
        "interpretation": interpretation,
        "scientific_scope": "in-domain synthetic evaluation only; no real-data claim",
    }
    write_json(output_dir / "metrics.json", report)
    lines = [
        f"# {checkpoint['experiment_id']} — {split} evaluation",
        "",
        interpretation,
        "",
        "## Overall metrics",
        "",
        f"- Samples: **{metrics['count']}**",
        f"- MAE: **{metrics['mae_arcsec']:.4f} arcsec**",
        f"- RMSE: **{metrics['rmse_arcsec']:.4f} arcsec**",
        f"- Median relative error: **{metrics['median_relative_error']:.2%}**",
        f"- Bias: **{metrics['bias_arcsec']:+.4f} arcsec**",
        f"- R²: **{metrics['r2']:.4f}**" if metrics["r2"] is not None else "- R²: undefined",
        "",
        "## How to inspect",
        "",
        "- `predictions.png`: closeness to the diagonal, residual trends, and skew/outliers.",
        "- `predictions.csv`: every truth, prediction, residual, relative error, and diagnostic group.",
        "- `metrics.json`: exact overall, baseline, and subgroup values.",
        "- The largest-error rows in `metrics.json` help identify failure modes; they must not be used to retune this locked test run.",
        "",
        "This report does not measure transfer to real HST observations.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = evaluate_checkpoint(
        args.checkpoint,
        split=args.split,
        output_dir=args.output,
        device_request=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
