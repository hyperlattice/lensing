"""Score the saved ten-seed C/G/R fits and report seed-level consistency.

Predictions are written before the holdout target column is opened. This is a
reanalysis of existing fits on the frozen synthetic holdout, not a new test
cohort or a new training run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.ml.artifacts import load_config, sha256
from src.ml.data import grouped_forward
from src.ml.model import build_model

STUDY_CONFIG = load_config(profile="primary")
ROOT = Path(STUDY_CONFIG["output"])
HOLDOUT = Path(STUDY_CONFIG["holdout"])
OUTPUT = Path("results/reports/cgr_seed_validation_v1")
ARMS = tuple(STUDY_CONFIG["arms"])
SEEDS = tuple(STUDY_CONFIG["seeds"])
CONDITIONS = tuple(STUDY_CONFIG["conditions"])
STRESS = CONDITIONS[1:]
UPDATES = STUDY_CONFIG["training"]["optimizer_updates"]
BATCH_SIZE = STUDY_CONFIG["training"]["batch_size"]
BOOTSTRAP_DRAWS = STUDY_CONFIG["bootstrap"]["resamples"]
BOOTSTRAP_SEED = STUDY_CONFIG["bootstrap"]["stress_seed"]


def predict_all() -> tuple[pd.DataFrame, dict[str, str]]:
    """Generate target-free predictions from all 30 existing checkpoints."""
    target_free = pd.read_csv(
        HOLDOUT,
        usecols=["sample_id", "parent_sample_id", "condition", "processed_path"],
    )
    if (
        len(target_free) != 750
        or target_free["sample_id"].duplicated().any()
        or set(target_free["condition"]) != set(CONDITIONS)
        or target_free["parent_sample_id"].nunique() != 150
    ):
        raise ValueError("Unexpected frozen holdout cohort")

    image_root = HOLDOUT.parent.parent
    images = []
    for relative in target_free["processed_path"].astype(str):
        image = np.load(image_root / relative, allow_pickle=False)
        if image.ndim != 2 or image.dtype != np.float32 or not np.isfinite(image).all():
            raise ValueError(f"Invalid holdout image: {relative}")
        images.append(image)

    torch.set_num_threads(4)
    predictions = []
    checkpoint_hashes = {}
    shared_model_config = None
    for arm in ARMS:
        for seed in SEEDS:
            path = ROOT / "fits" / arm / f"seed-{seed}" / "final.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoint_hashes[f"{arm}/{seed}"] = sha256(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("arm") != arm
                or checkpoint.get("seed") != seed
                or checkpoint.get("optimizer_updates") != UPDATES
            ):
                raise ValueError(f"Checkpoint metadata mismatch: {path}")
            model_config = checkpoint["model_config"]
            if shared_model_config is None:
                shared_model_config = model_config
            elif model_config != shared_model_config:
                raise ValueError("C/G/R checkpoints do not share one architecture")

            model = build_model(model_config)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            statistics = checkpoint["statistics"]
            values = []
            with torch.inference_mode():
                for start in range(0, len(images), BATCH_SIZE):
                    _, output = grouped_forward(
                        model, images[start : start + BATCH_SIZE], statistics
                    )
                    values.extend(
                        (
                            output.detach().numpy() * statistics["target_std"]
                            + statistics["target_mean"]
                        ).tolist()
                    )
            if len(values) != len(target_free) or not np.isfinite(values).all():
                raise ValueError(f"Invalid prediction count or value: {path}")
            frame = target_free[["sample_id", "parent_sample_id", "condition"]].copy()
            frame["arm"] = arm
            frame["seed"] = seed
            frame["theta_E_arcsec_predicted"] = values
            predictions.append(frame)

    return pd.concat(predictions, ignore_index=True), checkpoint_hashes


def assemble_metrics(joined: pd.DataFrame) -> dict:
    """Compute per-seed MAEs and parent-bootstrap paired contrasts."""
    parent_ids = sorted(joined["parent_sample_id"].unique())
    errors = {}
    per_seed = {}
    for arm in ARMS:
        error_cube = np.empty((len(SEEDS), len(parent_ids), len(CONDITIONS)), dtype=float)
        for seed_index, seed in enumerate(SEEDS):
            subset = joined[(joined["arm"] == arm) & (joined["seed"] == seed)]
            matrix = subset.pivot(
                index="parent_sample_id", columns="condition", values="absolute_error_arcsec"
            ).reindex(index=parent_ids, columns=CONDITIONS)
            if matrix.isna().any().any():
                raise ValueError(f"Missing holdout scores for {arm}, seed {seed}")
            error_cube[seed_index] = matrix.to_numpy()
        errors[arm] = error_cube
        per_seed[arm] = {
            str(seed): {
                "canonical_mae_arcsec": float(error_cube[i, :, 0].mean()),
                "four_stress_mae_arcsec": float(error_cube[i, :, 1:].mean()),
            }
            for i, seed in enumerate(SEEDS)
        }

    summaries = {}
    for arm in ARMS:
        canonical = errors[arm][:, :, 0].mean(axis=1)
        stress = errors[arm][:, :, 1:].mean(axis=(1, 2))
        summaries[arm] = {
            "canonical_mae_mean_arcsec": float(canonical.mean()),
            "canonical_mae_sd_across_seeds_arcsec": float(canonical.std(ddof=1)),
            "four_stress_mae_mean_arcsec": float(stress.mean()),
            "four_stress_mae_sd_across_seeds_arcsec": float(stress.std(ddof=1)),
        }

    contrasts = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(parent_ids), size=(BOOTSTRAP_DRAWS, len(parent_ids)))
    for contrast_index, (candidate, comparator) in enumerate((("G", "C"), ("R", "G"), ("R", "C"))):
        difference = errors[candidate][:, :, 1:] - errors[comparator][:, :, 1:]
        by_seed = difference.mean(axis=(1, 2))
        bootstrap = difference[:, indices, :].mean(axis=(0, 2, 3))
        seed_rng = np.random.default_rng(BOOTSTRAP_SEED + 1 + contrast_index)
        seed_indices = seed_rng.integers(0, len(SEEDS), size=(BOOTSTRAP_DRAWS, len(SEEDS)))
        seed_bootstrap = by_seed[seed_indices].mean(axis=1)
        contrasts[f"{candidate}_minus_{comparator}"] = {
            "mean_difference_arcsec": float(difference.mean()),
            "parent_bootstrap_95_low_arcsec": float(np.quantile(bootstrap, 0.025)),
            "parent_bootstrap_95_high_arcsec": float(np.quantile(bootstrap, 0.975)),
            "seed_bootstrap_95_low_arcsec": float(np.quantile(seed_bootstrap, 0.025)),
            "seed_bootstrap_95_high_arcsec": float(np.quantile(seed_bootstrap, 0.975)),
            "seeds_candidate_lower_mae": int(np.sum(by_seed < 0)),
            "seeds_candidate_higher_mae": int(np.sum(by_seed > 0)),
            "seed_differences_arcsec": {
                str(seed): float(value) for seed, value in zip(SEEDS, by_seed)
            },
        }

    return {
        "parent_count": len(parent_ids),
        "observation_count": len(joined) // (len(ARMS) * len(SEEDS)),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "seed_count_per_arm": len(SEEDS),
        "optimizer_updates_per_fit": UPDATES,
        "metrics_by_arm": summaries,
        "metrics_by_seed": per_seed,
        "four_stress_paired_contrasts": contrasts,
        "uncertainty_note": (
            "Parent-bootstrap intervals hold the ten trained seeds fixed; "
            "seed-to-seed spread is reported separately."
        ),
        "limitations": [
            "This scores the existing 30 fits; no new model fits were trained.",
            "All seed comparisons reuse the same frozen synthetic holdout, so they are not an independent test cohort.",
            "The holdout was previously scored by older project models.",
            "The results do not validate real telescope data or a different simulator.",
        ],
    }


def write_markdown(report: dict, prediction_sha256: str, checkpoint_hashes: dict[str, str]) -> None:
    lines = [
        "# C/G/R seed validation",
        "",
        "Reanalysis of the existing ten paired C/G/R training seeds on the frozen synthetic holdout. No additional training fits were made.",
        "",
        f"Holdout: {report['parent_count']} parents and {report['observation_count']} observations per arm/seed; {report['seed_count_per_arm']} seeds per arm.",
        "",
        "## Mean performance across seeds",
        "",
        "| Arm | Canonical MAE (mean +/- seed SD) | Four-stress MAE (mean +/- seed SD) |",
        "|---|---:|---:|",
    ]
    for arm in ARMS:
        item = report["metrics_by_arm"][arm]
        lines.append(
            f"| {arm} | {item['canonical_mae_mean_arcsec']:.6f} +/- {item['canonical_mae_sd_across_seeds_arcsec']:.6f} | "
            f"{item['four_stress_mae_mean_arcsec']:.6f} +/- {item['four_stress_mae_sd_across_seeds_arcsec']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Four-stress MAE by seed",
            "",
            "Positive differences mean the candidate arm had higher error than its comparator.",
            "",
            "| Seed | C | G | R | G-C | R-G | R-C |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    contrast_map = report["four_stress_paired_contrasts"]
    for seed in SEEDS:
        values = {
            arm: report["metrics_by_seed"][arm][str(seed)]["four_stress_mae_arcsec"] for arm in ARMS
        }
        diffs = {
            "G-C": contrast_map["G_minus_C"]["seed_differences_arcsec"][str(seed)],
            "R-G": contrast_map["R_minus_G"]["seed_differences_arcsec"][str(seed)],
            "R-C": contrast_map["R_minus_C"]["seed_differences_arcsec"][str(seed)],
        }
        lines.append(
            f"| {seed} | {values['C']:.6f} | {values['G']:.6f} | {values['R']:.6f} | "
            f"{diffs['G-C']:+.6f} | {diffs['R-G']:+.6f} | {diffs['R-C']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired seed-averaged contrasts",
            "",
            "Parent intervals resample lens systems while holding all ten seeds fixed. Seed intervals resample the ten paired training-seed effects while holding the 150 systems fixed.",
            "",
            "| Contrast | Mean difference (arcsec) | Parent-bootstrap 95% interval | Seed-bootstrap 95% interval | Seeds with lower candidate MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, value in contrast_map.items():
        lines.append(
            f"| {name.replace('_minus_', '-')} | {value['mean_difference_arcsec']:+.6f} | "
            f"[{value['parent_bootstrap_95_low_arcsec']:+.6f}, {value['parent_bootstrap_95_high_arcsec']:+.6f}] | "
            f"[{value['seed_bootstrap_95_low_arcsec']:+.6f}, {value['seed_bootstrap_95_high_arcsec']:+.6f}] | "
            f"{value['seeds_candidate_lower_mae']}/{len(SEEDS)} |"
        )
    lines.extend(
        [
            "",
            "## Scope and provenance",
            "",
            "Predictions were written before joining the target column. The prediction file is sealed by SHA-256; the JSON report records all 30 checkpoint hashes.",
            "",
            "This is a seed-consistency analysis on the same frozen synthetic holdout, not an independent test cohort. The separate parent and seed bootstrap intervals measure uncertainty across those two sampling axes; neither reuses a newly generated test population.",
            "",
            f"Prediction SHA-256: `{prediction_sha256}`",
            f"Checkpoint hashes recorded: {len(checkpoint_hashes)}",
            "",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    prediction_path = OUTPUT / "predictions_before_scoring.csv"
    seal_path = OUTPUT / "predictions_complete.json"
    report_path = OUTPUT / "report.json"
    if OUTPUT.exists():
        if not prediction_path.is_file() or not seal_path.is_file() or not report_path.is_file():
            raise FileExistsError(f"Incomplete existing validation output: {OUTPUT}")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if sha256(prediction_path) != seal["prediction_sha256"]:
            raise ValueError("Sealed predictions changed")
        if sha256(HOLDOUT) != seal["holdout_manifest_sha256"]:
            raise ValueError("Holdout manifest changed")
        predictions = pd.read_csv(prediction_path)
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        checkpoint_hashes = previous["checkpoint_sha256"]
        for arm in ARMS:
            for seed in SEEDS:
                key = f"{arm}/{seed}"
                checkpoint = ROOT / "fits" / arm / f"seed-{seed}" / "final.pt"
                if sha256(checkpoint) != checkpoint_hashes[key]:
                    raise ValueError(f"Checkpoint changed since prediction: {key}")
        prediction_sha256 = seal["prediction_sha256"]
    else:
        OUTPUT.mkdir(parents=True)
        prediction_frame, checkpoint_hashes = predict_all()
        prediction_frame.to_csv(prediction_path, index=False)
        prediction_sha256 = sha256(prediction_path)
        seal = {
            "schema": "cgr_seed_validation_predictions_v1",
            "prediction_path": str(prediction_path),
            "prediction_sha256": prediction_sha256,
            "holdout_manifest_sha256": sha256(HOLDOUT),
            "rows": len(prediction_frame),
            "truth_opened_before_prediction_seal": False,
        }
        seal_path.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
        predictions = prediction_frame

    truth = pd.read_csv(
        HOLDOUT,
        usecols=["sample_id", "parent_sample_id", "condition", "theta_E_arcsec"],
    )
    joined = predictions.merge(
        truth,
        on=["sample_id", "parent_sample_id", "condition"],
        validate="many_to_one",
    )
    if len(joined) != len(predictions) or not np.isfinite(joined["theta_E_arcsec"]).all():
        raise ValueError("Target join is incomplete")
    joined["absolute_error_arcsec"] = abs(
        joined["theta_E_arcsec_predicted"] - joined["theta_E_arcsec"]
    )
    report = assemble_metrics(joined)
    report.update(
        {
            "prediction_sha256": prediction_sha256,
            "holdout_manifest_sha256": seal["holdout_manifest_sha256"],
            "checkpoint_sha256": checkpoint_hashes,
            "model_config": torch.load(
                ROOT / "fits" / "C" / f"seed-{SEEDS[0]}" / "final.pt",
                map_location="cpu",
                weights_only=True,
            )["model_config"],
        }
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, prediction_sha256, checkpoint_hashes)
    print(
        json.dumps(
            {
                "stage": "seed_validation_scored",
                "report": str(OUTPUT / "report.md"),
                "parent_count": report["parent_count"],
                "seeds": report["seed_count_per_arm"],
                "contrasts": {
                    name: value["mean_difference_arcsec"]
                    for name, value in report["four_stress_paired_contrasts"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
