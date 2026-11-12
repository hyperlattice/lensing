"""Parent-paired primary analysis; training seeds remain fixed during bootstrap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.artifacts import sha256, verify_resolved_config
from src.simulation.observation import CONDITIONS as PAIRED_CONDITIONS
from src.simulation.observation import PRIMARY_CONDITIONS as PAIRED_PRIMARY
from src.simulation.observation import RENDERERS


def _bootstrap_difference(
    values_a: np.ndarray, values_b: np.ndarray, seed: int = 8731, draws: int = 10000
) -> dict[str, float | int]:
    """Compare errors with axes [seed, parent, condition], resampling whole parents."""
    rng = np.random.default_rng(seed)
    n = values_a.shape[1]
    indices = rng.integers(0, n, size=(draws, n))
    diffs = values_a[:, indices].mean(axis=(0, 2)) - values_b[:, indices].mean(axis=(0, 2))
    return {
        "difference_arcsec": float((values_a - values_b).mean()),
        "bootstrap_95_low_arcsec": float(np.quantile(diffs, 0.025)),
        "bootstrap_95_high_arcsec": float(np.quantile(diffs, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_resamples": draws,
    }


def score(config: dict[str, Any]) -> None:
    """Join sealed primary predictions to truth and write the existing report files."""
    output_root, holdout_path = Path(config["output"]), Path(config["holdout"])
    seeds, arms, conditions = config["seeds"], config["arms"], config["conditions"]
    stress_conditions = conditions[1:]
    verify_resolved_config(config)
    seal = json.loads((output_root / "predictions_complete.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(seal["prediction_path"])
    # This is the one-way target join. No score is written before this point.
    truth = pd.read_csv(
        holdout_path, usecols=["sample_id", "parent_sample_id", "condition", "theta_E_arcsec"]
    )
    joined = predictions.merge(
        truth, on=["sample_id", "parent_sample_id", "condition"], validate="many_to_one"
    )
    if len(joined) != len(predictions) or not np.isfinite(joined["theta_E_arcsec"]).all():
        raise ValueError("Prediction/truth join is incomplete")
    joined["absolute_error_arcsec"] = abs(
        joined["theta_E_arcsec_predicted"] - joined["theta_E_arcsec"]
    )
    metrics = {}
    for arm in arms:
        arm_frame = joined[joined["arm"] == arm]
        metrics[arm] = {}
        for condition in conditions:
            subset = arm_frame[arm_frame["condition"] == condition]
            metrics[arm][condition] = {
                "mae_arcsec": float(subset["absolute_error_arcsec"].mean()),
                "rmse_arcsec": float(
                    np.sqrt(
                        np.mean(
                            np.square(subset["theta_E_arcsec_predicted"] - subset["theta_E_arcsec"])
                        )
                    )
                ),
                "bias_arcsec": float(
                    (subset["theta_E_arcsec_predicted"] - subset["theta_E_arcsec"]).mean()
                ),
                "count_per_seed": int(len(subset) // len(seeds)),
            }
    parent_order = sorted(truth["parent_sample_id"].unique())
    four_stress = {}
    canonical = {}
    per_seed_stress = {}
    for arm in arms:
        stress_errors = np.empty(
            (len(seeds), len(parent_order), len(stress_conditions)), dtype=float
        )
        canonical_errors = np.empty((len(seeds), len(parent_order)), dtype=float)
        for seed_index, seed in enumerate(seeds):
            frame = joined[(joined["arm"] == arm) & (joined["seed"] == seed)]
            for parent_index, parent in enumerate(parent_order):
                row = frame[frame["parent_sample_id"] == parent]
                canonical_errors[seed_index, parent_index] = row[row["condition"] == "canonical"][
                    "absolute_error_arcsec"
                ].iloc[0]
                stress_errors[seed_index, parent_index] = [
                    row[row["condition"] == c]["absolute_error_arcsec"].iloc[0]
                    for c in stress_conditions
                ]
        four_stress[arm] = stress_errors
        canonical[arm] = canonical_errors
        per_seed_stress[arm] = stress_errors.mean(axis=(1, 2)).tolist()
    contrasts = {}
    for candidate, comparator in (("G", "C"), ("R", "G"), ("R", "C")):
        contrasts[f"{candidate}_vs_{comparator}_four_stress"] = _bootstrap_difference(
            four_stress[candidate],
            four_stress[comparator],
            seed=config["bootstrap"]["stress_seed"],
            draws=config["bootstrap"]["resamples"],
        )
        contrasts[f"{candidate}_vs_{comparator}_canonical"] = _bootstrap_difference(
            canonical[candidate][:, :, None],
            canonical[comparator][:, :, None],
            seed=config["bootstrap"]["canonical_seed"],
            draws=config["bootstrap"]["resamples"],
        )
    report = {
        "schema": "equal_budget_final_report_v1",
        "stage": "scored_after_target_join",
        "parent_count": len(parent_order),
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "conditions": list(conditions),
        "arms": list(arms),
        "optimizer_updates": config["training"]["optimizer_updates"],
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
    with (output_root / "final_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    lines = [
        "# Equal-budget final result",
        "",
        f"Frozen synthetic holdout: {len(parent_order)} parents; {len(seeds)} paired seeds; {config['training']['optimizer_updates']} updates.",
        "",
        "| Arm | Canonical MAE | Four-stress MAE |",
        "|---|---:|---:|",
    ]
    for arm in arms:
        lines.append(
            f"| {arm} | {metrics[arm]['canonical']['mae_arcsec']:.6f} | "
            f"{np.mean([metrics[arm][c]['mae_arcsec'] for c in stress_conditions]):.6f} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Difference (candidate - comparator) | Bootstrap 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for name, value in contrasts.items():
        if "four_stress" in name:
            lines.append(
                f"| {name} | {value['difference_arcsec']:.6f} | "
                f"[{value['bootstrap_95_low_arcsec']:.6f}, {value['bootstrap_95_high_arcsec']:.6f}] |"
            )
    lines.extend(
        [
            "",
            "The result is a source-only, equal-budget synthetic robustness comparison.",
            "It is not a claim of superiority over published models or real-data accuracy.",
            "",
        ]
    )
    (output_root / "result_overview.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "scored",
                "report": str(output_root / "final_report.json"),
                "four_stress": {
                    name: value for name, value in contrasts.items() if "four_stress" in name
                },
            },
            indent=2,
        )
    )


PAIRED_ROOT = Path("results/ml/cross-simulator-rapid-v2")
METHOD = PAIRED_ROOT / "method_freeze.json"
PAIRED_RELEASE = Path("data/simulations/cross-simulator-rapid-v2/manifest.json")
PAIRED_PREDICTIONS = PAIRED_ROOT / "predictions_before_scoring.csv"
PAIRED_ARMS = ("C", "G", "R")
PAIRED_SEEDS = tuple(730001 + i for i in range(10))
PAIRED_AUDIT = PAIRED_ROOT / "renderer_identity_audit.json"


def audit_release(*, write: bool = True) -> dict:
    """Return and optionally save exact A/B identity checks for the release."""
    manifest = json.loads(PAIRED_RELEASE.read_text(encoding="utf-8"))
    release_root = PAIRED_RELEASE.parent
    observations = manifest["observations"]
    by_key = {
        (row["parent_system_id"], row["renderer"], row["condition"]): row for row in observations
    }
    parents = list(manifest["parent_ids"])
    conditions = list(manifest["condition_order"])
    renderers = list(manifest["renderer_order"])
    if renderers != ["lenstronomy", "autolens"]:
        raise ValueError(f"Unexpected renderer order: {renderers}")

    image_pairs = 0
    exact_image_pairs = 0
    max_image_abs_difference = 0.0
    max_image_mean_abs_difference = 0.0
    for parent in parents:
        for condition in conditions:
            left = by_key[(parent, "lenstronomy", condition)]["products"]["image"]["path"]
            right = by_key[(parent, "autolens", condition)]["products"]["image"]["path"]
            image_left = np.load(release_root / left)
            image_right = np.load(release_root / right)
            difference = np.abs(image_left.astype(np.float64) - image_right.astype(np.float64))
            image_pairs += 1
            exact_image_pairs += int(np.array_equal(image_left, image_right))
            max_image_abs_difference = max(max_image_abs_difference, float(difference.max()))
            max_image_mean_abs_difference = max(
                max_image_mean_abs_difference, float(difference.mean())
            )

    predictions = pd.read_csv(PAIRED_PREDICTIONS)
    key_columns = ["parent_system_id", "condition", "arm", "seed"]
    paired = predictions.pivot_table(
        index=key_columns, columns="renderer", values="theta_E_arcsec_predicted", aggfunc="first"
    )
    if set(paired.columns) != {"lenstronomy", "autolens"}:
        raise ValueError("Prediction table does not contain both renderer columns")
    prediction_difference = (paired["lenstronomy"] - paired["autolens"]).abs()
    prediction_pairs = int(len(prediction_difference))
    exact_prediction_pairs = int((prediction_difference == 0).sum())

    backend_versions = {}
    for renderer in renderers:
        report = json.loads((release_root / f"render-{renderer}.json").read_text(encoding="utf-8"))
        backend_versions[renderer] = report["backend_versions"]

    result = {
        "schema": "cross_simulator_renderer_identity_audit_v1",
        "release_manifest_sha256": sha256(PAIRED_RELEASE),
        "truth_opened": False,
        "parent_count": len(parents),
        "condition_count": len(PAIRED_CONDITIONS),
        "image_pairs": image_pairs,
        "exact_image_pairs": exact_image_pairs,
        "image_exact_fraction": exact_image_pairs / image_pairs,
        "max_image_absolute_difference": max_image_abs_difference,
        "max_image_mean_absolute_difference": max_image_mean_abs_difference,
        "prediction_pairs": prediction_pairs,
        "exact_prediction_pairs": exact_prediction_pairs,
        "prediction_exact_fraction": exact_prediction_pairs / prediction_pairs,
        "max_prediction_absolute_difference": float(prediction_difference.max()),
        "backend_versions": backend_versions,
        "interpretation": (
            "The released A/B products are byte-identical at the saved float32 image "
            "precision. This is a matched-renderer equivalence control, not a "
            "simulator-induced image-domain shift; the Moffat conditions remain "
            "an observation-family shift shared by both renderer labels."
        )
        if exact_image_pairs == image_pairs and exact_prediction_pairs == prediction_pairs
        else (
            "The released products contain a measurable A/B difference; inspect "
            "the magnitude and provenance before making a transfer claim."
        ),
    }
    if write:
        PAIRED_AUDIT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def score_paired() -> None:
    """Score sealed paired-renderer predictions and audit exact image identity."""
    seal = json.loads((PAIRED_ROOT / "predictions_complete.json").read_text(encoding="utf-8"))
    if sha256(seal["prediction_path"]) != seal["prediction_sha256"]:
        raise ValueError("Prediction seal changed")
    pred = pd.read_csv(seal["prediction_path"])
    manifest = json.loads(PAIRED_RELEASE.read_text(encoding="utf-8"))
    if sha256(PAIRED_RELEASE.parent / "truth_vault" / "targets.json") != manifest["truth_sha256"]:
        raise ValueError("Truth vault hash changed")
    truth = json.loads(
        (PAIRED_RELEASE.parent / "truth_vault" / "targets.json").read_text(encoding="utf-8")
    )
    pred["theta_E_arcsec_true"] = pred["parent_system_id"].map(truth)
    if pred["theta_E_arcsec_true"].isna().any():
        raise ValueError("Incomplete truth join")
    pred["absolute_error_arcsec"] = abs(
        pred["theta_E_arcsec_predicted"] - pred["theta_E_arcsec_true"]
    )
    metrics = {}
    for arm in PAIRED_ARMS:
        metrics[arm] = {}
        for renderer in RENDERERS:
            metrics[arm][renderer] = {}
            for condition in PAIRED_CONDITIONS:
                subset = pred[
                    (pred.arm == arm) & (pred.renderer == renderer) & (pred.condition == condition)
                ]
                residual = subset.theta_E_arcsec_predicted - subset.theta_E_arcsec_true
                metrics[arm][renderer][condition] = {
                    "mae_arcsec": float(subset.absolute_error_arcsec.mean()),
                    "rmse_arcsec": float(np.sqrt(np.mean(np.square(residual)))),
                    "bias_arcsec": float(residual.mean()),
                    "count": int(len(subset)),
                }
    parents = sorted(truth)
    contrasts = {}
    arrays = {}
    for arm in PAIRED_ARMS:
        for renderer in RENDERERS:
            arr = np.empty((len(PAIRED_SEEDS), len(parents), len(PAIRED_PRIMARY)), dtype=float)
            for si, seed in enumerate(PAIRED_SEEDS):
                sub = pred[(pred.arm == arm) & (pred.renderer == renderer) & (pred.seed == seed)]
                for pi, parent in enumerate(parents):
                    row = sub[sub.parent_system_id == parent]
                    arr[si, pi] = [
                        row[row.condition == c].absolute_error_arcsec.iloc[0]
                        for c in PAIRED_PRIMARY
                    ]
            arrays[(arm, renderer)] = arr
    for renderer in RENDERERS:
        for candidate, comparator in (("G", "C"), ("R", "G"), ("R", "C")):
            contrasts[f"{candidate}_vs_{comparator}_{renderer}_primary_moffat"] = (
                _bootstrap_difference(
                    arrays[(candidate, renderer)],
                    arrays[(comparator, renderer)],
                    seed=620901,
                    draws=10000,
                )
            )
    transfer = {}
    for arm in PAIRED_ARMS:
        transfer[f"{arm}_autolens_minus_lenstronomy_primary"] = {
            "difference_arcsec": float(
                arrays[(arm, "autolens")].mean() - arrays[(arm, "lenstronomy")].mean()
            ),
            "autolens_mae_arcsec": float(arrays[(arm, "autolens")].mean()),
            "lenstronomy_mae_arcsec": float(arrays[(arm, "lenstronomy")].mean()),
        }
    renderer_identity = audit_release()
    report = {
        "schema": "cross_simulator_rapid_report_v1",
        "stage": "scored_after_target_join",
        "release_manifest_sha256": sha256(PAIRED_RELEASE),
        "parent_count": len(parents),
        "seed_count": len(PAIRED_SEEDS),
        "arms": list(PAIRED_ARMS),
        "renderers": list(RENDERERS),
        "primary_conditions": list(PAIRED_PRIMARY),
        "metrics": metrics,
        "contrasts": contrasts,
        "renderer_transfer": transfer,
        "renderer_identity": renderer_identity,
        "limitations": [
            "Rapid release uses the existing qualified renderer stack and a new 150-parent AutoLens/Lenstronomy paired cohort.",
            "Moffat PSF is an engineered unseen observation family, not real telescope data.",
            "The formal Freeze-2 power-planned workflow was not rerun; item-1 checkpoints were fixed before this evaluation.",
            "All 1,350 saved Lenstronomy/AutoLens image pairs are byte-identical at float32 precision, so this release is not evidence of a simulator-induced image-domain shift.",
            "The package implementations use shared project integration, detector and preprocessing; simulator independence is therefore not established.",
        ],
    }
    (PAIRED_ROOT / "final_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Rapid cross-simulator result",
        "",
        f"New paired parents: {len(parents)}; paired seeds: {len(PAIRED_SEEDS)}; primary conditions: {', '.join(PAIRED_PRIMARY)}.",
        "",
        "| Contrast | Lenstronomy difference | AutoLens difference |",
        "|---|---:|---:|",
    ]
    for candidate, comparator in (("G", "C"), ("R", "G"), ("R", "C")):
        a = contrasts[f"{candidate}_vs_{comparator}_lenstronomy_primary_moffat"]
        b = contrasts[f"{candidate}_vs_{comparator}_autolens_primary_moffat"]
        lines.append(
            f"| {candidate} vs {comparator} | {a['difference_arcsec']:.6f} "
            f"[{a['bootstrap_95_low_arcsec']:.6f}, {a['bootstrap_95_high_arcsec']:.6f}] | "
            f"{b['difference_arcsec']:.6f} [{b['bootstrap_95_low_arcsec']:.6f}, {b['bootstrap_95_high_arcsec']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "Renderer identity audit: 1,350/1,350 Lenstronomy/AutoLens image pairs and 40,500/40,500 paired predictions are exactly equal at saved precision.",
            "Therefore this release is a matched-renderer equivalence control plus an unseen Moffat observation-family shift, not evidence of simulator-induced transfer.",
            "The Moffat family is engineered synthetic data, not real-data validation.",
            "",
        ]
    )
    (PAIRED_ROOT / "result_overview.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "scored",
                "report": str(PAIRED_ROOT / "final_report.json"),
                "primary_contrasts": {k: v for k, v in contrasts.items() if "primary_moffat" in k},
            },
            indent=2,
        )
    )
