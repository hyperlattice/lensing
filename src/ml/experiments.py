"""Run a registered seed benchmark using training and validation only."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.ml.data import file_sha256
from src.ml.engine import require_output_directory, select_device, write_json
from src.ml.metrics import validate_bin_edges
from src.ml.plots import plot_seed_comparison
from src.ml.provenance import config_sha256, environment_record, source_provenance
from src.ml.train import load_config, train_experiment


DEFAULT_REGISTRY = Path("configs/ml/registry_v1.yaml")
DEFAULT_EXPERIMENT = "theta-e-cnn-seeds-v1"


def load_registered_experiment(experiment_id, registry_path=DEFAULT_REGISTRY):
    """Registry and config paths follow the repository-root CLI convention."""
    registry = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise ValueError("Experiment registry must use schema_version 1")
    entries = registry.get("experiments", {})
    if experiment_id not in entries:
        raise ValueError(f"Unknown experiment {experiment_id!r}; registered: {sorted(entries)}")
    suite = yaml.safe_load(Path(entries[experiment_id]).read_text(encoding="utf-8"))
    validate_suite(suite)
    if suite["experiment_id"] != experiment_id:
        raise ValueError("Registry experiment ID does not match the suite config")
    return suite


def validate_suite(suite):
    required = {"schema_version", "experiment_id", "parent_baseline_id", "base_config", "output", "seeds", "target_bin_edges_arcsec"}
    if not isinstance(suite, dict) or required.difference(suite):
        raise ValueError(f"Suite config requires: {sorted(required)}")
    if suite["schema_version"] != 1:
        raise ValueError("Suite must use schema_version 1")
    if set(suite).difference(required):
        raise ValueError(f"Unknown suite fields: {sorted(set(suite).difference(required))}")
    if not isinstance(suite["experiment_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", suite["experiment_id"]):
        raise ValueError("experiment_id must contain only letters, numbers, underscores, and hyphens")
    seeds = suite["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 2:
        raise ValueError("A seed benchmark requires at least two seeds")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("seeds must be integers in [0, 2**32)")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    validate_bin_edges(suite["target_bin_edges_arcsec"])


def _aggregate(values):
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return {"mean": None, "sample_std": None, "min": None, "max": None, "defined_seeds": 0}
    if not np.isfinite(defined).all():
        raise ValueError("Cannot aggregate nonfinite metrics")
    return {
        "mean": float(np.mean(defined)),
        "sample_std": float(np.std(defined, ddof=1)) if len(defined) > 1 else None,
        "min": min(defined),
        "max": max(defined),
        "defined_seeds": len(defined),
    }


def aggregate_runs(summaries):
    """Equal-weight summary of seed-level metrics, not an ensemble score."""
    if len(summaries) < 2 or len({run["seed"] for run in summaries}) != len(summaries):
        raise ValueError("Aggregation requires at least two distinct seeds")
    reference = summaries[0]
    for run in summaries:
        if run["test_samples_accessed"] != 0 or run["status"] != "trained_validation_only":
            raise ValueError("Seed benchmarks must be validation-only")
        for field in ("release_manifest_sha256", "split_manifest_sha256", "train_samples", "validation_samples", "run_controls"):
            if run[field] != reference[field]:
                raise ValueError(f"Incompatible seed runs: {field} differs")
        if run["validation_metrics"].keys() != reference["validation_metrics"].keys():
            raise ValueError("Incompatible seed metrics")
        bins = [(row["lower_arcsec"], row["upper_arcsec"], row["count"]) for row in run["validation_target_bins"]]
        reference_bins = [(row["lower_arcsec"], row["upper_arcsec"], row["count"]) for row in reference["validation_target_bins"]]
        if bins != reference_bins:
            raise ValueError("Incompatible target bins")
    metric_names = [name for name in reference["validation_metrics"] if name != "count"]
    metrics = {name: _aggregate([run["validation_metrics"][name] for run in summaries]) for name in metric_names}
    bins = []
    for index, row in enumerate(reference["validation_target_bins"]):
        bins.append({
            "lower_arcsec": row["lower_arcsec"],
            "upper_arcsec": row["upper_arcsec"],
            "count_per_seed": row["count"],
            "metrics": {
                name: _aggregate([run["validation_target_bins"][index]["metrics"][name] for run in summaries])
                for name in metric_names
            } if row["count"] else None,
        })
    return {
        "seed_count": len(summaries),
        "seeds": [run["seed"] for run in summaries],
        "validation_samples_per_seed": reference["validation_samples"],
        "train_samples_per_seed": reference["train_samples"],
        "test_samples_accessed": 0,
        "validation_metrics": metrics,
        "validation_target_bins": bins,
        "runtime_seconds": _aggregate([run["runtime_seconds"] for run in summaries]),
        "best_epoch": _aggregate([run["best_epoch"] for run in summaries]),
        "total_training_runtime_seconds": sum(run["runtime_seconds"] for run in summaries),
        "aggregation": "Arithmetic mean and sample standard deviation (ddof=1) across seeds; not a confidence interval or ensemble evaluation.",
    }


def _load_verified_run(run_dir, config, plan):
    summary_path = run_dir / "training_summary.json"
    if not summary_path.exists():
        raise ValueError(f"Incomplete run at {run_dir}; preserve it and choose a new --output, or move this unfinished seed directory before --resume")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    if resolved != config or summary["config_sha256"] != config_sha256(config):
        raise ValueError(f"Configuration changed for {run_dir}")
    if summary["seed"] != config["seed"] or summary["experiment_id"] != config["experiment_id"]:
        raise ValueError(f"Run identity changed for {run_dir}")
    for field in ("release_manifest_sha256", "split_manifest_sha256", "run_controls", "environment"):
        if summary[field] != plan[field]:
            raise ValueError(f"{field} changed for {run_dir}")
    if summary["provenance"]["source_sha256"] != plan["source_sha256"]:
        raise ValueError(f"Source changed for {run_dir}")
    for filename, field in (("best.pt", "checkpoint_sha256"), ("validation_predictions.csv", "validation_predictions_sha256")):
        if file_sha256(run_dir / filename) != summary[field]:
            raise ValueError(f"Artifact hash changed: {run_dir / filename}")
    return summary


def run_seed_suite(suite, *, output_override=None, device_request="auto", resume=False, smoke=False):
    validate_suite(suite)
    suite = copy.deepcopy(suite)
    base = load_config(suite["base_config"])
    if base["experiment_id"] != suite["parent_baseline_id"]:
        raise ValueError("parent_baseline_id must match the base config experiment_id")
    output = Path(output_override or (suite["output"] + "-smoke" if smoke else suite["output"]))
    suite["output"] = output.as_posix()
    controls = {"max_train_samples": 32 if smoke else None, "max_validation_samples": 16 if smoke else None}
    if smoke:
        base["epochs"] = 2
    base["target_bin_edges_arcsec"] = validate_bin_edges(suite["target_bin_edges_arcsec"])
    release = Path(base["release"])
    plan = {
        "suite": suite,
        "base_config": base,
        "device_request": device_request,
        "environment": environment_record(select_device(device_request)),
        "smoke": smoke,
        "run_controls": controls,
        "release_manifest_sha256": file_sha256(release / "manifest.csv"),
        "split_manifest_sha256": {split: file_sha256(release / "splits" / f"{split}.csv") for split in ("train", "validation", "test")},
        "source_sha256": source_provenance()["source_sha256"],
    }
    plan_path = output / "resolved_suite.json"
    if resume:
        if not plan_path.exists() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Cannot resume: missing or changed suite, release, source, device request, or run controls")
    else:
        require_output_directory(output)
        write_json(plan_path, plan)

    summaries = []
    reference_targets = None
    for seed in suite["seeds"]:
        run_dir = output / f"seed-{seed}"
        config = copy.deepcopy(base)
        config.update(experiment_id=f"{suite['experiment_id']}-seed-{seed}", parent_baseline_id=suite["parent_baseline_id"], seed=seed, output=run_dir.as_posix())
        if not run_dir.exists() or not any(run_dir.iterdir()):
            print(f"Training seed {seed} ({'smoke' if smoke else 'full validation benchmark'})", flush=True)
            train_experiment(config, device_request=device_request, **controls)
        elif not resume:
            raise FileExistsError(f"Run already exists: {run_dir}")
        summary = _load_verified_run(run_dir, config, plan)
        frame = pd.read_csv(run_dir / "validation_predictions.csv").sort_values("sample_id").reset_index(drop=True)
        targets = frame[["sample_id", "theta_E_arcsec_true"]]
        if targets["sample_id"].duplicated().any() or len(targets) != summary["validation_samples"]:
            raise ValueError("Validation prediction IDs/counts are inconsistent")
        if reference_targets is not None and not targets.equals(reference_targets):
            raise ValueError("Seed runs do not share identical validation samples and targets")
        reference_targets = targets
        summaries.append(summary)

    report = aggregate_runs(summaries)
    report.update(
        experiment_id=suite["experiment_id"], parent_baseline_id=suite["parent_baseline_id"],
        status="smoke_only" if smoke else "completed_validation_benchmark",
        suite_sha256=config_sha256(plan), release_manifest_sha256=plan["release_manifest_sha256"],
        split_manifest_sha256=plan["split_manifest_sha256"],
        runs=summaries,
        scientific_scope="Synthetic validation only; test manifests hashed for identity, zero test images loaded. Smoke runs are wiring checks only." if smoke else "In-domain synthetic validation; test manifests hashed for identity, zero test images loaded.",
    )
    rows = [{"seed": run["seed"], "best_epoch": run["best_epoch"], "runtime_seconds": run["runtime_seconds"], **run["validation_metrics"]} for run in summaries]
    pd.DataFrame(rows).to_csv(output / "per_seed_metrics.csv", index=False)
    bin_rows = []
    for run in summaries:
        for row in run["validation_target_bins"]:
            bin_rows.append({"seed": run["seed"], "lower_arcsec": row["lower_arcsec"], "upper_arcsec": row["upper_arcsec"], "count": row["count"], **(row["metrics"] or {})})
    pd.DataFrame(bin_rows).to_csv(output / "validation_target_bins.csv", index=False)
    plot_seed_comparison(summaries, output / "seed_comparison.png")
    _write_report(output / "report.md", report)
    write_json(output / "aggregate_validation.json", report)
    return report


def _write_report(path, report):
    lines = [
        f"# {report['experiment_id']}", "",
        "SMOKE CHECK ONLY: these metrics are not a development benchmark." if report["status"] == "smoke_only" else "Baseline stability across fixed random seeds, using validation only.", "",
        f"Each seed uses {report['train_samples_per_seed']} training and {report['validation_samples_per_seed']} validation samples. Test images accessed: 0.", "",
        "| Seed | Best epoch | MAE (arcsec) | RMSE (arcsec) | Median relative error | Bias (arcsec) | R2 | P90 (arcsec) | P95 (arcsec) | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        m = run["validation_metrics"]
        r2 = "undefined" if m["r2"] is None else f"{m['r2']:.4f}"
        lines.append(f"| {run['seed']} | {run['best_epoch']} | {m['mae_arcsec']:.4f} | {m['rmse_arcsec']:.4f} | {m['median_relative_error']:.2%} | {m['bias_arcsec']:+.4f} | {r2} | {m['p90_absolute_error_arcsec']:.4f} | {m['p95_absolute_error_arcsec']:.4f} | {run['runtime_seconds']:.1f} |")
    lines.extend(["", "## Across-seed metrics", "", "| Metric | Mean | Sample standard deviation |", "|---|---:|---:|"])
    for name, values in report["validation_metrics"].items():
        mean = "undefined" if values["mean"] is None else f"{values['mean']:.6f}"
        std = "undefined" if values["sample_std"] is None else f"{values['sample_std']:.6f}"
        lines.append(f"| {name} | {mean} | {std} |")
    lines.extend(["", report["aggregation"], "", "## Radius-dependent errors", "", "Bins are [lower, upper), except the last bounded bin includes its upper edge. Outside-range groups and empty bins remain in the CSV/JSON.", "", "| Radius bin (arcsec) | Samples per seed | Mean MAE | Mean bias | Mean P95 |", "|---|---:|---:|---:|---:|"])
    for row in report["validation_target_bins"]:
        if row["metrics"] is not None:
            m = row["metrics"]
            lines.append(f"| {row['lower_arcsec']} to {row['upper_arcsec']} | {row['count_per_seed']} | {m['mae_arcsec']['mean']:.4f} | {m['bias_arcsec']['mean']:+.4f} | {m['p95_absolute_error_arcsec']['mean']:.4f} |")
    lines.extend(["", "## Interpretation", "", "MAE is the average angular miss. P90/P95 are error thresholds containing 90%/95% of validation cases. Bias is prediction minus truth. Seed variation measures sensitivity to initialization, dropout, and training order on the same fixed data split.", "", "Compare future candidates using the same seeds and report paired changes. Improvements smaller than the baseline seed variation are inconclusive; three seeds do not establish statistical significance. Inspect P95 and radius-bin bias before accepting a lower mean MAE. These are individual-model metrics, not predictions from an ensemble.", "", "The checkpoint for each seed is selected by validation MAE. Do not pick a lucky seed as the headline result or use existing test metrics to choose models. A final challenger requires the independent holdout described in the planning notebook.", "", "Open `seed_comparison.png` for errors and radius-bin bias, `per_seed_metrics.csv` for per-run results, and `aggregate_validation.json` for hashes, configurations, environments, and exact metrics.", "", report["scientific_scope"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed seeds with identical config, data, and source")
    parser.add_argument("--smoke", action="store_true", help="Two epochs on at most 32 train/16 validation examples per seed; not a benchmark")
    args = parser.parse_args()
    suite = load_registered_experiment(args.experiment, args.registry)
    report = run_seed_suite(suite, output_override=args.output, device_request=args.device, resume=args.resume, smoke=args.smoke)
    print(json.dumps({"experiment_id": report["experiment_id"], "status": report["status"], "validation_mae": report["validation_metrics"]["mae_arcsec"], "test_samples_accessed": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
