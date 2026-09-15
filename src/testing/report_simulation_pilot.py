"""Build a compact evidence report for a cumulative simulation pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.simulation.render import render_sie_extended_source
from src.testing.validate_simulations import validate_run


def _numeric_range(frame: pd.DataFrame, field: str) -> dict[str, float] | None:
    if field not in frame:
        return None
    values = pd.to_numeric(frame[field], errors="coerce").dropna()
    if values.empty:
        return None
    return {"min": float(values.min()), "max": float(values.max())}


def _noise_summary(root: Path, manifest: pd.DataFrame) -> dict[str, Any]:
    required = {"observed_science_path", "expected_scene_path", "expected_variance_path"}
    if not required.issubset(manifest.columns):
        return {"sample_count": 0, "status": "not_available"}
    residual_means = []
    residual_stds = []
    predicted_stds = []
    for row in manifest.to_dict(orient="records"):
        values = [row.get(field) for field in required]
        if any(value is None or (isinstance(value, float) and np.isnan(value)) for value in values):
            continue
        observed = np.load(root / str(row["observed_science_path"]))
        expected = np.load(root / str(row["expected_scene_path"]))
        variance = np.load(root / str(row["expected_variance_path"]))
        residual = np.asarray(observed, dtype=float) - np.asarray(expected, dtype=float)
        residual_means.append(float(np.mean(residual)))
        residual_stds.append(float(np.std(residual)))
        predicted_stds.append(float(np.mean(np.sqrt(np.maximum(variance, 0.0)))))
    if not residual_means:
        return {"sample_count": 0, "status": "not_available"}
    return {
        "sample_count": len(residual_means),
        "status": "measured_engineering_moment",
        "mean_residual_mean": float(np.mean(residual_means)),
        "mean_residual_std": float(np.mean(residual_stds)),
        "mean_predicted_pixel_std": float(np.mean(predicted_stds)),
        "noise_policy": "Poisson in electrons plus Gaussian read noise before background subtraction",
    }


def _convergence_summary(root: Path, manifest: pd.DataFrame) -> dict[str, Any]:
    if "release_stage" not in manifest or not (manifest["release_stage"] == "sim-p30-v1").any():
        return {"status": "not_available"}
    row = manifest.loc[manifest["release_stage"] == "sim-p30-v1"].iloc[0]
    parameters_path = root / str(row["parameters_path"])
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    factor_one = render_sie_extended_source(parameters, supersampling_factor=1).image
    factor_two = render_sie_extended_source(parameters, supersampling_factor=2).image
    relative_change = np.linalg.norm(factor_one - factor_two) / np.linalg.norm(factor_two)
    return {
        "status": "measured_single_control",
        "sample_id": str(row["sample_id"]),
        "factor_one": 1,
        "factor_two": 2,
        "relative_l2_change": float(relative_change),
    }


def build_pilot_report(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    stage_roots: Iterable[str | Path] = (),
    validation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write JSON and Markdown evidence records for one cumulative manifest."""

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = pd.read_csv(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = Path(validation_path) if validation_path is not None else root / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    else:
        validation = validate_run(manifest_path)
        validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stage_records = []
    roots = [Path(value) for value in stage_roots]
    if not roots and (root / "run_manifest.json").exists():
        roots = [Path(value) for value in json.loads((root / "run_manifest.json").read_text(encoding="utf-8")).get("stage_roots", [])]
    for stage_root in roots:
        stage_manifest = stage_root / "manifest.csv"
        stage_run = stage_root / "run_manifest.json"
        record = {"stage_root": stage_root.as_posix()}
        if stage_manifest.exists():
            stage_frame = pd.read_csv(stage_manifest)
            record.update({
                "sample_count": int(len(stage_frame)),
                "parent_count": int(stage_frame["parent_system_id"].nunique()),
            })
        if stage_run.exists():
            run = json.loads(stage_run.read_text(encoding="utf-8"))
            record.update({
                "count_requested": run.get("count_requested"),
                "count_accepted": run.get("count_accepted"),
                "count_rejected": run.get("count_rejected", run.get("count_rejected_in_stages", 0)),
            })
        stage_records.append(record)

    disk_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    report = {
        "manifest_path": manifest_path.as_posix(),
        "release_id": str(manifest["dataset_version"].iloc[0]),
        "sample_count": int(len(manifest)),
        "parent_count": int(manifest["parent_system_id"].nunique()),
        "stage_counts": manifest.groupby("release_stage").size().to_dict() if "release_stage" in manifest else {},
        "multiplicity_counts": manifest["image_count_physical"].value_counts().sort_index().to_dict()
        if "image_count_physical" in manifest else {},
        "detectable_multiplicity_counts": manifest["image_count_detectable"].value_counts().sort_index().to_dict()
        if "image_count_detectable" in manifest else {},
        "parameter_ranges": {
            field: _numeric_range(manifest, field)
            for field in ("theta_E_arcsec", "q", "gamma1", "gamma2", "psf_fwhm_arcsec", "supersampling_factor")
        },
        "validation": validation,
        "noise": _noise_summary(root, manifest),
        "supersampling_convergence": _convergence_summary(root, manifest),
        "stage_records": stage_records,
        "rejection_count": int(sum(record.get("count_rejected", 0) or 0 for record in stage_records)),
        "disk_bytes": int(disk_bytes),
        "limitations": [
            "The Gaussian PSF is an analytic control profile, not an empirical CASTLES/HST kernel.",
            "Noise uses a toy independent-pixel model with sampled exposure/background/read-noise parameters.",
            "The pilot is an angular engineering dataset; no calibrated physical mass likelihood is claimed.",
            "The cumulative release contains stage increments with different model complexity and should retain release_stage in evaluation.",
        ],
    }
    json_path = output_dir / "validation_report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        f"# Simulation pilot report: {report['release_id']}",
        "",
        f"- Samples: **{report['sample_count']}**",
        f"- Independent parents: **{report['parent_count']}**",
        f"- Artifact size at report time: **{report['disk_bytes'] / (1024**2):.2f} MiB**",
        f"- Rejected stage draws: **{report['rejection_count']}**",
        "",
        "## Validation",
        "",
        f"- Validator passed: **{report['validation'].get('passed', False)}**",
        f"- Validator errors: **{report['validation'].get('error_count', 'unknown')}**",
        f"- Stage counts: `{report['stage_counts']}`",
        f"- Physical multiplicities: `{report['multiplicity_counts']}`",
        f"- Detectable multiplicities: `{report['detectable_multiplicity_counts']}`",
        "",
        "## Numerical and noise checks",
        "",
        f"- Supersampling control: `{report['supersampling_convergence']}`",
        f"- Noise moments: `{report['noise']}`",
        "",
        "## Limitations",
        "",
    ]
    markdown.extend(f"- {limitation}" for limitation in report["limitations"])
    (output_dir / "validation_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage-root", action="append", default=[])
    parser.add_argument("--validation", default=None)
    args = parser.parse_args()
    report = build_pilot_report(
        args.manifest,
        args.output,
        stage_roots=args.stage_root,
        validation_path=args.validation,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["validation"].get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
