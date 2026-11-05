"""Week 5 matched source-profile, source-position, and shear evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.ml.data import file_sha256
from src.ml.engine import select_device, write_json
from src.ml.metrics import regression_metrics
from src.ml.observation_systematics_audit import _bootstrap_interval, _predict_checkpoint
from src.modeling.pixel_scale_geometric_baseline import (
    ALLOWED_VALIDATION_COLUMNS,
    detect_point_image_peaks,
)


DEFAULT_CONFIG = Path("configs/reliability/source_physics.yaml")
DEFAULT_RELEASE = Path("data/simulations/reliability-source-physics-v1")
DEFAULT_OUTPUT = Path("results/reports/reliability_v1/source_physics")
BASELINE_VALIDATION = Path("data/simulations/reliability-pixel-scale-v1-final/splits/validation__canonical.csv")
W5_PEAK_DETECTION = {
    "threshold_snr": 5.0,
    "center_smoothing_sigma_arcsec": 0.15,
    "lens_center_search_radius_arcsec": 0.35,
    "central_exclusion_radius_arcsec": 0.25,
    "outer_detection_radius_arcsec": 2.25,
    "minimum_peak_separation_arcsec": 0.30,
    "maximum_peaks": 4,
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metric_or_none(targets, predictions):
    if len(targets) == 0:
        return None
    return regression_metrics(targets, predictions)


def _failure_map(w5_effects: pd.DataFrame) -> pd.DataFrame:
    """Join the completed Week 3–5 paired effects into one descriptive ranking."""
    entries = []
    h1 = pd.read_csv("results/reports/reliability_v1/pixel_scale/frozen_baseline/paired_effects.csv")
    if "input_strategy" in h1:
        h1 = h1.loc[h1["input_strategy"] == "native"]
    for row in h1.to_dict(orient="records"):
        entries.append({
            "phase": "week3_pixel_scale",
            "family": "pixel_scale",
            "intervention_id": row["intervention_id"],
            "parent_count": int(row["parent_count"]),
            "paired_delta_mae_arcsec": float(row["paired_mean_delta_mae_arcsec"]),
            "bootstrap_95_low_arcsec": float(row["bootstrap_95_low_arcsec"]),
            "bootstrap_95_high_arcsec": float(row["bootstrap_95_high_arcsec"]),
            "mean_absolute_prediction_shift_arcsec": float(row["mean_absolute_prediction_shift_arcsec"]),
            "practically_important_increase_supported": bool(row["practically_important_increase_supported"]),
            "effect_practically_small": bool(row["effect_practically_small"]),
        })

    h4 = pd.read_csv("results/reports/reliability_v1/observation_systematics/cnn_paired_effects.csv")
    for row in h4.to_dict(orient="records"):
        entries.append({
            "phase": "week4_observation_systematics",
            "family": str(row["intervention_id"]).split("-")[0],
            "intervention_id": row["intervention_id"],
            "parent_count": int(row["parent_count"]),
            "paired_delta_mae_arcsec": float(row["paired_delta_mae_arcsec"]),
            "bootstrap_95_low_arcsec": float(row["bootstrap_95_low_arcsec"]),
            "bootstrap_95_high_arcsec": float(row["bootstrap_95_high_arcsec"]),
            "mean_absolute_prediction_shift_arcsec": float(row["mean_absolute_paired_prediction_shift_arcsec"]),
            "practically_important_increase_supported": bool(row["practically_important_increase_supported"]),
            "effect_practically_small": bool(row["effect_practically_small"]),
        })
    for row in w5_effects.to_dict(orient="records"):
        entries.append({
            "phase": "week5_source_physics",
            "family": str(row["intervention_id"]).split("-")[0],
            "intervention_id": row["intervention_id"],
            "parent_count": int(row["parent_count"]),
            "paired_delta_mae_arcsec": float(row["paired_delta_mae_arcsec"]),
            "bootstrap_95_low_arcsec": float(row["bootstrap_95_low_arcsec"]),
            "bootstrap_95_high_arcsec": float(row["bootstrap_95_high_arcsec"]),
            "mean_absolute_prediction_shift_arcsec": float(row["mean_absolute_paired_prediction_shift_arcsec"]),
            "practically_important_increase_supported": bool(row["practically_important_increase_supported"]),
            "effect_practically_small": bool(row["effect_practically_small"]),
        })
    result = pd.DataFrame(entries)
    result["positive_penalty_arcsec"] = result["paired_delta_mae_arcsec"].clip(lower=0.0)
    result = result.sort_values(
        ["practically_important_increase_supported", "positive_penalty_arcsec"],
        ascending=[False, False],
    ).reset_index(drop=True)
    result.insert(0, "positive_penalty_rank", np.arange(1, len(result) + 1))
    return result


def rebuild_multiplicity_products(release_path, output_dir):
    """Join saved observation-only detections by IDs, then score multiplicity.

    Float target values are checked for agreement after the join; they are not
    used as identity keys because prediction targets were stored as float32.
    """
    release = Path(release_path)
    output = Path(output_dir)
    detection = pd.read_csv(output / "observed_multiplicity_before_scoring.csv")
    validation = pd.read_csv(release / "manifest.csv")
    validation = validation.loc[validation["split"] == "validation"]
    diagnostic_columns = [
        "sample_id", "parent_sample_id", "intervention_id", "theta_E_arcsec",
        "image_count_physical", "image_count_detectable", "source_n", "source_re_arcsec",
        "source_x_arcsec", "source_y_arcsec", "gamma1", "gamma2", "gamma_amplitude", "mass_model",
    ]
    diagnostic_columns = [name for name in diagnostic_columns if name in validation.columns]
    diagnostics = validation[diagnostic_columns].rename(columns={"theta_E_arcsec": "theta_E_arcsec_true"})
    multiplicity = detection.merge(
        diagnostics,
        on=["sample_id", "parent_sample_id", "intervention_id"],
        validate="one_to_one",
    )
    multiplicity.to_csv(output / "physical_detectable_observed_multiplicity.csv", index=False)

    predictions = pd.read_csv(output / "cnn_predictions.csv")
    cnn_parent = predictions.groupby(["intervention_id", "parent_sample_id"], as_index=False).agg(
        cnn_theta_E_arcsec_true=("theta_E_arcsec_true", "mean"),
        theta_E_arcsec_predicted=("theta_E_arcsec_predicted", "mean"),
        seed_mean_absolute_error_arcsec=("absolute_error_arcsec", "mean"),
    )
    scored = multiplicity.merge(
        cnn_parent,
        on=["intervention_id", "parent_sample_id"],
        validate="one_to_one",
    )
    if not np.allclose(
        scored["theta_E_arcsec_true"].to_numpy(dtype=float),
        scored["cnn_theta_E_arcsec_true"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("Prediction and manifest target values disagree after parent/condition join")
    scored["absolute_error_arcsec"] = scored["seed_mean_absolute_error_arcsec"]
    scored["physical_detectable_agree"] = (
        scored["image_count_physical"] == scored["image_count_detectable"]
    )
    scored["physical_observed_peak_agree"] = (
        scored["image_count_physical"] == scored["observed_peak_count"]
    )
    scored.to_csv(output / "multiplicity_scored.csv", index=False)

    distribution_rows = []
    for intervention_id, group in scored.groupby("intervention_id", sort=False):
        for kind, field in [
            ("physical_roots", "image_count_physical"),
            ("renderer_amplitude_threshold", "image_count_detectable"),
            ("observed_peak_detector_snr_ge_5", "observed_peak_count"),
        ]:
            by_count = group.groupby(field).agg(
                parent_count=("parent_sample_id", "nunique"),
                mae_arcsec=("absolute_error_arcsec", "mean"),
            )
            for count, row in by_count.iterrows():
                distribution_rows.append({
                    "intervention_id": intervention_id,
                    "multiplicity_definition": kind,
                    "image_count": int(count),
                    "parent_count": int(row["parent_count"]),
                    "fraction": float(row["parent_count"] / len(group)),
                    "cnn_mae_arcsec_seed_average": float(row["mae_arcsec"]),
                })
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(output / "multiplicity_summary.csv", index=False)
    agreement = scored.groupby("intervention_id", as_index=False).agg(
        parent_count=("parent_sample_id", "nunique"),
        amplitude_threshold_agreement_fraction=("physical_detectable_agree", "mean"),
        observed_peak_agreement_fraction=("physical_observed_peak_agree", "mean"),
        mean_physical_roots=("image_count_physical", "mean"),
        mean_amplitude_detectable=("image_count_detectable", "mean"),
        mean_observed_peaks=("observed_peak_count", "mean"),
    )
    agreement.to_csv(output / "multiplicity_agreement.csv", index=False)

    counts = scored.groupby("intervention_id", sort=False).agg(
        physical=("image_count_physical", "mean"),
        threshold=("image_count_detectable", "mean"),
        detected=("observed_peak_count", "mean"),
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    loc = np.arange(len(counts))
    width = 0.26
    ax.bar(loc - width, counts["physical"], width, label="physical roots")
    ax.bar(loc, counts["threshold"], width, label="renderer amplitude threshold")
    ax.bar(loc + width, counts["detected"], width, label="observed peaks, S/N ≥ 5")
    ax.set_xticks(loc, counts.index, rotation=35, ha="right")
    ax.set_ylabel("Mean image count per parent")
    ax.set_title("Physical, thresholded, and observation-detected multiplicity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "multiplicity_comparison.png", dpi=170)
    plt.close(fig)
    return scored, agreement


def evaluate_source_physics(
    *,
    config_path=DEFAULT_CONFIG,
    release_path=DEFAULT_RELEASE,
    output_dir=DEFAULT_OUTPUT,
    device_request="auto",
    batch_size=16,
):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    release = Path(release_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_manifest_path = release / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if not run_manifest.get("passed") or not run_manifest.get("identity_checks_passed"):
        raise ValueError("Week 5 rerender release failed identity/ancestry/loader gates")

    validation_ids = set(
        pd.read_csv(release / "splits" / "validation__canonical.csv")["parent_sample_id"].astype(str)
    )
    expected_ids = set(pd.read_csv(BASELINE_VALIDATION)["parent_sample_id"].astype(str))
    if len(validation_ids) != 150 or validation_ids != expected_ids:
        raise ValueError("Week 5 parent set differs from the frozen H1 validation parent set")
    interventions = [str(item["intervention_id"]) for item in config["interventions"]]
    if not interventions or interventions[0] != "canonical" or len(interventions) != len(set(interventions)):
        raise ValueError("Interventions must be unique and begin with the canonical reference")
    for intervention_id in interventions:
        rows = pd.read_csv(release / "splits" / f"validation__{intervention_id}.csv")
        if len(rows) != 150 or set(rows["parent_sample_id"].astype(str)) != expected_ids:
            raise ValueError(f"Intervention {intervention_id} lacks complete paired validation parents")

    checkpoints = [Path(item) for item in config["model_checkpoints"]]
    device = select_device(device_request)
    cnn_records = []
    seed_metrics = []
    for checkpoint in checkpoints:
        frame, metrics = _predict_checkpoint(
            checkpoint,
            release=release,
            intervention_ids=interventions,
            device=device,
            batch_size=batch_size,
        )
        cnn_records.append(frame)
        seed_metrics.extend(metrics)
    predictions = pd.concat(cnn_records, ignore_index=True)
    predictions.to_csv(output / "cnn_predictions.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(output / "cnn_metrics_per_seed.csv", index=False)

    canonical = predictions.loc[predictions["intervention_id"] == "canonical", [
        "checkpoint_id", "parent_sample_id", "theta_E_arcsec_true", "theta_E_arcsec_predicted", "absolute_error_arcsec"
    ]].rename(columns={
        "theta_E_arcsec_predicted": "canonical_prediction_arcsec",
        "absolute_error_arcsec": "canonical_absolute_error_arcsec",
    })
    shifted = predictions.loc[predictions["intervention_id"] != "canonical"].merge(
        canonical,
        on=["checkpoint_id", "parent_sample_id", "theta_E_arcsec_true"],
        validate="many_to_one",
    )
    shifted["paired_delta_absolute_error_arcsec"] = shifted["absolute_error_arcsec"] - shifted["canonical_absolute_error_arcsec"]
    shifted["paired_prediction_shift_arcsec"] = shifted["theta_E_arcsec_predicted"] - shifted["canonical_prediction_arcsec"]
    shifted.to_csv(output / "cnn_paired_predictions.csv", index=False)

    effect_rows = []
    parent_delta_frames = []
    for intervention_id, group in shifted.groupby("intervention_id", sort=False):
        parent = group.groupby("parent_sample_id", as_index=False).agg(
            delta_mae=("paired_delta_absolute_error_arcsec", "mean"),
            prediction_shift=("paired_prediction_shift_arcsec", "mean"),
        )
        low, high = _bootstrap_interval(
            parent["delta_mae"].to_numpy(),
            resamples=int(config["metrics"]["bootstrap_resamples"]),
            seed=int(config["metrics"]["bootstrap_seed"]),
        )
        mean_delta = float(parent["delta_mae"].mean())
        threshold = float(config["metrics"]["practical_effect_arcsec"])
        effect_rows.append({
            "intervention_id": intervention_id,
            "parent_count": int(len(parent)),
            "paired_delta_mae_arcsec": mean_delta,
            "bootstrap_95_low_arcsec": low,
            "bootstrap_95_high_arcsec": high,
            "mean_absolute_paired_prediction_shift_arcsec": float(parent["prediction_shift"].abs().mean()),
            "practically_important_increase_supported": bool(mean_delta >= threshold and low > 0.0),
            "effect_practically_small": bool(low >= -threshold and high <= threshold),
        })
        targets = group.groupby("parent_sample_id", as_index=False)["theta_E_arcsec_true"].first()
        parent = parent.merge(targets, on="parent_sample_id", validate="one_to_one")
        parent["intervention_id"] = intervention_id
        parent_delta_frames.append(parent)
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(output / "cnn_paired_effects.csv", index=False)
    parent_delta = pd.concat(parent_delta_frames, ignore_index=True)

    radius_rows = []
    for intervention_id, group in parent_delta.groupby("intervention_id", sort=False):
        for name, lower, upper, upper_inclusive in [
            ("0.5_to_1.0", 0.5, 1.0, False),
            ("1.0_to_1.5", 1.0, 1.5, False),
            ("1.5_to_2.0", 1.5, 2.0, True),
        ]:
            use = (group["theta_E_arcsec_true"] >= lower) & (
                group["theta_E_arcsec_true"] <= upper if upper_inclusive else group["theta_E_arcsec_true"] < upper
            )
            values = group.loc[use, "delta_mae"].to_numpy(dtype=float)
            if len(values):
                low, high = _bootstrap_interval(
                    values,
                    resamples=int(config["metrics"]["bootstrap_resamples"]),
                    seed=int(config["metrics"]["bootstrap_seed"]),
                )
                mean = float(values.mean())
            else:
                low = high = mean = None
            radius_rows.append({
                "intervention_id": intervention_id,
                "theta_E_bin": name,
                "parent_count": int(use.sum()),
                "paired_delta_mae_arcsec": mean,
                "bootstrap_95_low_arcsec": low,
                "bootstrap_95_high_arcsec": high,
            })
    pd.DataFrame(radius_rows).to_csv(output / "radius_bin_effects.csv", index=False)

    summaries = []
    for intervention_id, group in predictions.groupby("intervention_id", sort=False):
        parent = group.groupby(["parent_sample_id", "theta_E_arcsec_true"], as_index=False).agg(
            theta_E_arcsec_predicted=("theta_E_arcsec_predicted", "mean")
        )
        summaries.append({
            "intervention_id": intervention_id,
            "seed_count": int(group["checkpoint_id"].nunique()),
            **regression_metrics(parent["theta_E_arcsec_true"], parent["theta_E_arcsec_predicted"]),
        })
    pd.DataFrame(summaries).to_csv(output / "cnn_summary_metrics.csv", index=False)

    # Peak detection is run using only released observation products. Physical and
    # amplitude-threshold truth descriptors are loaded only after this table is saved.
    detection_records = []
    release_manifest_path = release / "manifest.csv"
    release_manifest = pd.read_csv(release_manifest_path)
    validation_manifest = release_manifest.loc[release_manifest["split"] == "validation"]
    usecols = [name for name in ALLOWED_VALIDATION_COLUMNS if name in validation_manifest.columns]
    observation_rows = validation_manifest[usecols]
    for row in observation_rows.to_dict(orient="records"):
        image = np.load(release / row["observed_science_path"])
        variance = np.load(release / row["variance_path"])
        invalid = np.load(release / row["mask_path"])
        coverage = np.load(release / row["coverage_mask_path"])
        psf = np.load(release / row["psf_path"])
        found = detect_point_image_peaks(
            image, variance, invalid, coverage, psf,
            pixel_scale_arcsec=float(row["pixel_scale"]), **W5_PEAK_DETECTION
        )
        detection_records.append({
            "sample_id": str(row["sample_id"]),
            "parent_sample_id": str(row["parent_sample_id"]),
            "split": str(row["split"]),
            "intervention_id": str(row["intervention_id"]),
            "observed_peak_count": int(len(found["peaks"])),
            "detected_peak_snr": json.dumps([float(peak["snr"]) for peak in found["peaks"]]),
        })
    detection = pd.DataFrame(detection_records)
    detection.to_csv(output / "observed_multiplicity_before_scoring.csv", index=False)

    multiplicity_metrics, agreement = rebuild_multiplicity_products(release, output)

    # Descriptive ranking of positive paired error effects across Weeks 3–5.
    ranking = _failure_map(effects)
    ranking.to_csv(output / "failure_mechanism_ranking.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(effects))
    means = effects["paired_delta_mae_arcsec"].to_numpy()
    low = effects["bootstrap_95_low_arcsec"].to_numpy()
    high = effects["bootstrap_95_high_arcsec"].to_numpy()
    ax.errorbar(x, means, yerr=[means - low, high - means], fmt="o", capsize=3, color="#26547c")
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(float(config["metrics"]["practical_effect_arcsec"]), color="gray", linestyle="--", linewidth=1)
    ax.axhline(-float(config["metrics"]["practical_effect_arcsec"]), color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x, effects["intervention_id"], rotation=35, ha="right")
    ax.set_ylabel("Paired change in absolute error (arcsec)")
    ax.set_title("Source/physics shifts versus matched canonical parents")
    fig.tight_layout()
    fig.savefig(output / "paired_effects.png", dpi=170)
    plt.close(fig)

    summary = {
        "experiment_id": "WEEK5-SOURCE-PHYSICS",
        "protocol": "archive/docs/reliability_study/protocol_v1.5.md",
        "protocol_sha256": file_sha256("archive/docs/reliability_study/protocol_v1.5.md"),
        "config_sha256": file_sha256(config_path),
        "release_manifest_sha256": file_sha256(release / "manifest.csv"),
        "parent_manifest_sha256": run_manifest["parent_manifest_sha256"],
        "source_parent_count": int(len(validation_ids)),
        "condition_count_including_canonical": len(interventions),
        "cnn_checkpoint_hashes": {path.as_posix(): file_sha256(path) for path in checkpoints},
        "historical_test_split_accessed": False,
        "source_family_alternative_available": False,
        "mass_family_alternative_available": False,
        "physical_multiplicity_definition": "number of point-source image roots returned by the renderer",
        "detectable_multiplicity_definitions": [
            "renderer point-amplitude threshold 0.05",
            "observed PSF-matched peak count at S/N >= 5",
        ],
        "bootstrap_resamples": int(config["metrics"]["bootstrap_resamples"]),
        "bootstrap_seed": int(config["metrics"]["bootstrap_seed"]),
        "practical_effect_arcsec": float(config["metrics"]["practical_effect_arcsec"]),
        "paired_effects": effect_rows,
        "multiplicity_agreement": agreement.to_dict(orient="records"),
        "failure_mechanism_map_path": (output / "failure_mechanism_ranking.csv").as_posix(),
        "outputs": [
            "cnn_predictions.csv", "cnn_metrics_per_seed.csv", "cnn_paired_predictions.csv",
            "cnn_paired_effects.csv", "radius_bin_effects.csv", "cnn_summary_metrics.csv",
            "observed_multiplicity_before_scoring.csv", "physical_detectable_observed_multiplicity.csv",
            "multiplicity_scored.csv", "multiplicity_summary.csv", "multiplicity_agreement.csv",
            "failure_mechanism_ranking.csv", "paired_effects.png", "multiplicity_comparison.png",
        ],
    }
    write_json(output / "summary.json", summary)
    report = [
        "# Week 5 — Source and physics mismatch audit",
        "",
        "Validation-only matched evaluation on the same 150 parents as Weeks 3–4. Three frozen CNN seeds, paired parent bootstrap, and the v1.3 observation-only peak detector were used. The historical test split was not accessed.",
        "",
        "## Paired CNN effects",
        "",
        "| Intervention | Parents | Paired MAE change (arcsec) | 95% parent interval | Supported >=0.01 | Practically small |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    for row in effect_rows:
        report.append(
            f"| {row['intervention_id']} | {row['parent_count']} | {row['paired_delta_mae_arcsec']:+.4f} | "
            f"[{row['bootstrap_95_low_arcsec']:+.4f}, {row['bootstrap_95_high_arcsec']:+.4f}] | "
            f"{'yes' if row['practically_important_increase_supported'] else 'no'} | {'yes' if row['effect_practically_small'] else 'no'} |"
        )
    report.extend([
        "",
        "![Paired effects](paired_effects.png)",
        "",
        "## Multiplicity interpretation",
        "",
        "Physical multiplicity is the renderer's point-source image-root count. The renderer's 0.05-amplitude threshold is a simulator diagnostic. The observation-detected count is a separate PSF-matched peak count at S/N ≥ 5 using observed products only. These quantities are preserved parent-by-parent and are not interchangeable.",
        "",
        "![Multiplicity counts](multiplicity_comparison.png)",
        "",
        "## Failure-mechanism ranking",
        "",
        "`failure_mechanism_ranking.csv` joins Week 3 pixel-scale, Week 4 observation-systematics, and Week 5 source/physics paired effects. It sorts supported positive penalties first, then remaining positive mean effects. This across-condition ranking is descriptive; its intervals are unadjusted for the number of conditions.",
        "",
        "## Limits",
        "",
        "The validated simulator uses one elliptical Sersic source family and SIE, optionally with external shear. No alternative source or mass family was available for a validated comparison. The source-index/radius controls and fixed 0.05 shear are synthetic stress conditions, not a calibrated population prior. The classical control remains point-image geometry with fallback.",
        "",
    ])
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--release", default=str(DEFAULT_RELEASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rebuild-multiplicity", action="store_true")
    args = parser.parse_args()
    if args.rebuild_multiplicity:
        scored, agreement = rebuild_multiplicity_products(args.release, args.output)
        summary_path = Path(args.output) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["multiplicity_agreement"] = agreement.to_dict(orient="records")
        summary["multiplicity_scoring_join"] = "parent_sample_id + intervention_id, followed by target consistency check at 1e-5 arcsec"
        summary["multiplicity_scored_parent_condition_rows"] = int(len(scored))
        summary["reproducibility_hashes"] = {
            "config": file_sha256(args.config),
            "audit_code": file_sha256(__file__),
        }
        write_json(summary_path, summary)
        print(json.dumps({"scored_rows": len(scored), "condition_count": int(len(agreement)), "agreement": summary["multiplicity_agreement"]}, indent=2))
        return 0
    summary = evaluate_source_physics(
        config_path=args.config,
        release_path=args.release,
        output_dir=args.output,
        device_request=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
