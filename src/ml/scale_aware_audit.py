"""Compare H1 mitigation predictions with the frozen baseline by parent."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CANDIDATE_ROOT = Path("results/ml/reliability_pixel_scale_v1")
DEFAULT_BASELINE_PREDICTIONS = Path(
    "results/reports/reliability_v1/pixel_scale/frozen_baseline/predictions.csv"
)
DEFAULT_OUTPUT = Path("results/reports/reliability_v1/pixel_scale/mitigations")
STRATUM_INTERVENTION = {
    "canonical": "canonical",
    "low": "pixel-scale-004",
    "high": "pixel-scale-006",
}


def _bootstrap_mean(values, *, resamples=10000, seed=6031):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Bootstrap input must be a nonempty vector")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(resamples), dtype=np.float64)
    for offset in range(0, int(resamples), 256):
        count = min(256, int(resamples) - offset)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[offset : offset + count] = values[indices].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _candidate_predictions(candidate_root, family):
    frames = []
    for seed in (12001, 12002, 12003):
        path = candidate_root / family / f"seed-{seed}" / "validation_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate predictions: {path}")
        frame = pd.read_csv(path)
        if set(frame["seed"].astype(int)) != {seed}:
            raise ValueError(f"Unexpected seed labels in {path}")
        frame["seed"] = seed
        frame["intervention_id"] = frame["stratum"].map(STRATUM_INTERVENTION)
        if frame["intervention_id"].isna().any():
            raise ValueError(f"Unknown validation stratum in {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def audit_scale_aware_mitigations(
    *,
    candidate_root=DEFAULT_CANDIDATE_ROOT,
    baseline_predictions=DEFAULT_BASELINE_PREDICTIONS,
    output_dir=DEFAULT_OUTPUT,
    bootstrap_resamples=10000,
    bootstrap_seed=6031,
    practical_effect_arcsec=0.01,
):
    candidate_root = Path(candidate_root)
    baseline_predictions = Path(baseline_predictions)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(baseline_predictions)
    if "input_strategy" in baseline:
        baseline = baseline.loc[baseline["input_strategy"] == "native"].copy()
    baseline["seed"] = baseline["checkpoint_id"].str.extract(r"seed-(\d+)").astype(int)
    baseline = baseline.rename(
        columns={
            "theta_E_arcsec_predicted": "baseline_prediction_arcsec",
            "absolute_error_arcsec": "baseline_absolute_error_arcsec",
        }
    )
    baseline_columns = [
        "seed",
        "parent_sample_id",
        "intervention_id",
        "theta_E_arcsec_true",
        "baseline_prediction_arcsec",
        "baseline_absolute_error_arcsec",
    ]
    baseline = baseline[baseline_columns]

    all_effects = []
    seed_metrics = []
    paired_frames = []
    summaries = {}
    for family in ("pixel_target", "metadata"):
        candidate = _candidate_predictions(candidate_root, family)
        candidate = candidate.rename(
            columns={
                "theta_E_arcsec_predicted": "candidate_prediction_arcsec",
                "absolute_error_arcsec": "candidate_absolute_error_arcsec",
            }
        )
        paired = candidate.merge(
            baseline,
            on=["seed", "parent_sample_id", "intervention_id", "theta_E_arcsec_true"],
            validate="one_to_one",
        )
        if len(paired) != 1350 or paired["parent_sample_id"].nunique() != 150:
            raise ValueError(f"{family} predictions are incomplete or misaligned with baseline")
        paired["candidate_minus_baseline_abs_error_arcsec"] = (
            paired["candidate_absolute_error_arcsec"]
            - paired["baseline_absolute_error_arcsec"]
        )
        paired["family"] = family
        paired_frames.append(paired)

        family_effects = []
        for intervention_id, group in paired.groupby("intervention_id", sort=True):
            parent = group.groupby("parent_sample_id", as_index=False).agg(
                candidate_mae_component=("candidate_absolute_error_arcsec", "mean"),
                baseline_mae_component=("baseline_absolute_error_arcsec", "mean"),
                effect=("candidate_minus_baseline_abs_error_arcsec", "mean"),
            )
            interval = _bootstrap_mean(
                parent["effect"].to_numpy(),
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
            row = {
                "family": family,
                "comparison": "candidate_minus_frozen_baseline",
                "intervention_id": intervention_id,
                "parent_count": len(parent),
                "baseline_mae_arcsec": float(parent["baseline_mae_component"].mean()),
                "candidate_mae_arcsec": float(parent["candidate_mae_component"].mean()),
                "paired_delta_mae_arcsec": float(parent["effect"].mean()),
                "bootstrap_95_low_arcsec": interval[0],
                "bootstrap_95_high_arcsec": interval[1],
            }
            family_effects.append(row)
            all_effects.append(row)

        own_scale_effects = []
        pivot = paired.pivot_table(
            index=["parent_sample_id", "seed"],
            columns="intervention_id",
            values="candidate_absolute_error_arcsec",
            aggfunc="first",
        )
        for intervention_id in ("pixel-scale-004", "pixel-scale-006"):
            parent_effects = (
                pivot[intervention_id].sub(pivot["canonical"])
                .groupby("parent_sample_id")
                .mean()
            )
            interval = _bootstrap_mean(
                parent_effects.to_numpy(),
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
            own_scale_effects.append(
                {
                    "family": family,
                    "comparison": "shifted_scale_minus_candidate_canonical",
                    "intervention_id": intervention_id,
                    "parent_count": len(parent_effects),
                    "paired_delta_mae_arcsec": float(parent_effects.mean()),
                    "bootstrap_95_low_arcsec": interval[0],
                    "bootstrap_95_high_arcsec": interval[1],
                }
            )
            all_effects.append(own_scale_effects[-1])

        shifted_lowers_error = all(
            row["paired_delta_mae_arcsec"] < 0.0
            for row in family_effects
            if row["intervention_id"] in {"pixel-scale-004", "pixel-scale-006"}
        )
        canonical = next(
            row for row in family_effects if row["intervention_id"] == "canonical"
        )
        canonical_noninferior = (
            canonical["paired_delta_mae_arcsec"] <= float(practical_effect_arcsec)
        )
        summaries[family] = {
            "baseline_comparisons": family_effects,
            "candidate_scale_penalties": own_scale_effects,
            "lowers_error_at_both_shifted_scales": shifted_lowers_error,
            "canonical_degradation_at_most_threshold": canonical_noninferior,
            "utility_rule_passed": shifted_lowers_error and canonical_noninferior,
        }

        for (seed, stratum), group in candidate.groupby(["seed", "stratum"]):
            seed_metrics.append(
                {
                    "family": family,
                    "seed": int(seed),
                    "stratum": stratum,
                    "mae_arcsec": float(group["candidate_absolute_error_arcsec"].mean()),
                    "bias_arcsec": float(group["residual_arcsec"].mean()),
                    "p95_absolute_error_arcsec": float(
                        group["candidate_absolute_error_arcsec"].quantile(0.95)
                    ),
                }
            )
    effects_frame = pd.DataFrame(all_effects)
    effects_frame.to_csv(output_dir / "paired_effects.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(output_dir / "per_seed_metrics.csv", index=False)
    paired_frame = pd.concat(paired_frames, ignore_index=True)
    paired_frame.to_csv(output_dir / "paired_predictions.csv", index=False)
    result = {
        "experiment_id": "H1-PIXEL-SCALE-MITIGATION",
        "validation_parent_count": 150,
        "seeds": [12001, 12002, 12003],
        "historical_test_split_accessed": False,
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "practical_effect_arcsec": float(practical_effect_arcsec),
        "mitigations": summaries,
        "effects_path": (output_dir / "paired_effects.csv").as_posix(),
        "per_seed_metrics_path": (output_dir / "per_seed_metrics.csv").as_posix(),
        "paired_predictions_path": (output_dir / "paired_predictions.csv").as_posix(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Pixel-scale mitigation audit",
        "",
        "Validation-only paired comparison with the frozen baseline. Checkpoint epochs were selected using the predeclared equal-weight mean across the three validation scale strata.",
        "",
        "The utility rule requires lower shifted-scale MAE at both 0.04 and 0.06 arcsec/pixel and no more than 0.01 arcsec canonical MAE degradation. Parent-bootstrap intervals average the three seed effects within each parent before resampling.",
        "",
        "The models were selected on these validation parents, so these figures are development evidence. The historical test split was not accessed; conclusions remain within the current synthetic simulator.",
        "",
    ]
    for family, summary in summaries.items():
        lines.extend(
            [
                f"## {family}",
                "",
                f"- Lower MAE at both shifted scales: **{summary['lowers_error_at_both_shifted_scales']}**",
                f"- Canonical degradation within 0.01 arcsec: **{summary['canonical_degradation_at_most_threshold']}**",
                f"- Utility rule passed: **{summary['utility_rule_passed']}**",
                "",
                "| Comparison | Paired MAE change (arcsec) | Parent-bootstrap 95% interval |",
                "|---|---:|---:|",
            ]
        )
        for row in summary["baseline_comparisons"]:
            lines.append(
                f"| Candidate minus baseline, {row['intervention_id']} | {row['paired_delta_mae_arcsec']:+.4f} | "
                f"[{row['bootstrap_95_low_arcsec']:+.4f}, {row['bootstrap_95_high_arcsec']:+.4f}] |"
            )
        for row in summary["candidate_scale_penalties"]:
            lines.append(
                f"| Candidate {row['intervention_id']} minus candidate canonical | {row['paired_delta_mae_arcsec']:+.4f} | "
                f"[{row['bootstrap_95_low_arcsec']:+.4f}, {row['bootstrap_95_high_arcsec']:+.4f}] |"
            )
        lines.append("")
    lines.extend(
        [
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", default=str(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument("--baseline-predictions", default=str(DEFAULT_BASELINE_PREDICTIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = audit_scale_aware_mitigations(
        candidate_root=args.candidate_root,
        baseline_predictions=args.baseline_predictions,
        output_dir=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
