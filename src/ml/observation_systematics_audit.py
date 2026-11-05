"""Validation-only matched audit of PSF, detector, lens-light, and input scaling."""

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

from src.ml.data import TorchLensDataset, file_sha256
from src.ml.engine import collect_predictions, make_loader, select_device, write_json
from src.ml.metrics import regression_metrics
from src.ml.model import build_model


DEFAULT_CONFIG = Path("configs/reliability/observation_systematics.yaml")
DEFAULT_RELEASE = Path("data/simulations/reliability-observation-systematics-v1")
DEFAULT_OUTPUT = Path("results/reports/reliability_v1/observation_systematics")
BASELINE_VALIDATION = Path("data/simulations/reliability-pixel-scale-v1-final/splits/validation__canonical.csv")


def _bootstrap_interval(values: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap input must be a finite, nonempty vector")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), 256):
        count = min(256, int(resamples) - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _predict_checkpoint(
    checkpoint_path: Path,
    *,
    release: Path,
    intervention_ids: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, list[dict]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    original_manifest = Path(checkpoint["release"]) / "manifest.csv"
    if file_sha256(original_manifest) != str(checkpoint["release_manifest_sha256"]):
        raise ValueError(f"Frozen source release hash changed: {original_manifest}")

    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    records: list[dict] = []
    metrics_rows: list[dict] = []
    for intervention_id in intervention_ids:
        dataset = TorchLensDataset(
            release,
            split=f"validation__{intervention_id}",
            target_field="theta_E_arcsec",
            statistics=checkpoint["statistics"],
            expected_shape=(256, 256),
        )
        loader = make_loader(dataset, batch_size=batch_size, shuffle=False, seed=0, num_workers=0)
        result = collect_predictions(model, loader, device, checkpoint["statistics"])
        metadata = dataset.dataset.metadata.iloc[dataset.indices].copy()
        keep = [
            "sample_id",
            "parent_sample_id",
            "split",
            "intervention_id",
            "theta_E_arcsec",
            "pixel_scale",
            "psf_fwhm_arcsec",
            "exposure_time",
            "background_rate",
            "read_noise_electrons",
            "image_count_physical",
            "image_count_detectable",
            "normalization_method",
        ]
        metadata = metadata[keep]
        frame = pd.DataFrame(
            {
                "sample_id": result["sample_ids"],
                "theta_E_arcsec_true": result["targets"],
                "theta_E_arcsec_predicted": result["predictions"],
            }
        ).merge(metadata, on="sample_id", validate="one_to_one")
        frame["checkpoint_id"] = str(checkpoint["experiment_id"])
        frame["checkpoint_sha256"] = file_sha256(checkpoint_path)
        frame["seed"] = int(checkpoint["seed"])
        frame["residual_arcsec"] = frame["theta_E_arcsec_predicted"] - frame["theta_E_arcsec_true"]
        frame["absolute_error_arcsec"] = frame["residual_arcsec"].abs()
        frame["relative_error"] = frame["absolute_error_arcsec"] / frame["theta_E_arcsec_true"].abs()
        records.extend(frame.to_dict(orient="records"))
        metrics_rows.append(
            {
                "checkpoint_id": str(checkpoint["experiment_id"]),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "seed": int(checkpoint["seed"]),
                "intervention_id": intervention_id,
                **regression_metrics(result["targets"], result["predictions"]),
            }
        )
        print(f"Completed CNN validation predictions: {intervention_id}, seed {checkpoint['seed']}", flush=True)
    return pd.DataFrame(records), metrics_rows


def _observation_descriptors(release: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """Calculate post-prediction, simulator-defined quality descriptors."""
    result = []
    for row in manifest.to_dict(orient="records"):
        signal = np.load(release / row["expected_scene_path"]).astype(np.float64)
        variance = np.load(release / row["expected_variance_path"]).astype(np.float64)
        invalid = np.load(release / row["mask_path"]).astype(bool)
        coverage = np.load(release / row["coverage_mask_path"]).astype(bool)
        valid = ~invalid & ~coverage & np.isfinite(signal) & np.isfinite(variance) & (variance > 0)
        denominator = float(np.sqrt(variance[valid].sum())) if valid.any() else float("nan")
        snr = float(signal[valid].sum() / denominator) if denominator > 0 else float("nan")
        result.append(
            {
                "sample_id": str(row["sample_id"]),
                "parent_sample_id": str(row["parent_sample_id"]),
                "intervention_id": str(row["intervention_id"]),
                "integrated_scene_snr_proxy": snr,
                "image_count_physical": int(row["image_count_physical"]),
                "image_count_detectable": int(row["image_count_detectable"]),
            }
        )
    return pd.DataFrame(result)


def evaluate_observation_systematics(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    release_path: str | Path = DEFAULT_RELEASE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    device_request: str = "auto",
    batch_size: int = 16,
) -> dict:
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
        raise ValueError("Week 4 counterfactual release did not pass its generation and identity gates")

    checkpoints = [Path(value) for value in config["model_checkpoints"]]
    if len(checkpoints) != 3:
        raise ValueError("Week 4 requires all three frozen baseline seeds")
    checkpoint_payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in checkpoints]
    source_hashes = {Path(item["release"]) / "manifest.csv" for item in checkpoint_payloads}
    if len(source_hashes) != 1:
        raise ValueError("Frozen CNN checkpoints do not share one source release")
    for item in checkpoint_payloads:
        if file_sha256(Path(item["release"]) / "manifest.csv") != item["release_manifest_sha256"]:
            raise ValueError("A frozen CNN checkpoint's source manifest hash changed")
        if run_manifest["parent_manifest_sha256"] != item["release_manifest_sha256"]:
            raise ValueError("Week 4 release does not descend from the frozen CNN's source data")

    canonical_ids = set(
        pd.read_csv(release / "splits" / "validation__canonical.csv")["parent_sample_id"].astype(str)
    )
    expected_ids = set(pd.read_csv(BASELINE_VALIDATION)["parent_sample_id"].astype(str))
    if len(canonical_ids) != 150 or canonical_ids != expected_ids:
        raise ValueError("Week 4 validation parents differ from the frozen H1 validation set")
    intervention_ids = [str(item["intervention_id"]) for item in config["interventions"]]
    if intervention_ids[0] != "canonical" or len(intervention_ids) != len(set(intervention_ids)):
        raise ValueError("Intervention list must be unique and begin with canonical")
    for intervention_id in intervention_ids:
        rows = pd.read_csv(release / "splits" / f"validation__{intervention_id}.csv")
        if len(rows) != 150 or set(rows["parent_sample_id"].astype(str)) != expected_ids:
            raise ValueError(f"Validation intervention {intervention_id} is not a complete paired 150-parent set")
        if set(rows["split"].astype(str)) != {"validation"}:
            raise ValueError(f"Non-validation descendants found in {intervention_id}")

    device = select_device(device_request)
    prediction_frames = []
    seed_rows = []
    for checkpoint in checkpoints:
        frame, metrics = _predict_checkpoint(
            checkpoint,
            release=release,
            intervention_ids=intervention_ids,
            device=device,
            batch_size=batch_size,
        )
        prediction_frames.append(frame)
        seed_rows.extend(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output / "cnn_predictions.csv", index=False)
    per_seed = pd.DataFrame(seed_rows)
    per_seed.to_csv(output / "cnn_metrics_per_seed.csv", index=False)

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
    shifted["paired_delta_absolute_error_arcsec"] = (
        shifted["absolute_error_arcsec"] - shifted["canonical_absolute_error_arcsec"]
    )
    shifted["paired_prediction_shift_arcsec"] = (
        shifted["theta_E_arcsec_predicted"] - shifted["canonical_prediction_arcsec"]
    )
    shifted.to_csv(output / "cnn_paired_predictions.csv", index=False)

    metrics_rows = []
    for (intervention_id, seed), group in predictions.groupby(["intervention_id", "seed"], sort=False):
        metrics_rows.append({
            "intervention_id": intervention_id,
            "seed": int(seed),
            **regression_metrics(group["theta_E_arcsec_true"], group["theta_E_arcsec_predicted"]),
        })
    pd.DataFrame(metrics_rows).to_csv(output / "cnn_metrics_per_seed.csv", index=False)

    effect_rows = []
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
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(output / "cnn_paired_effects.csv", index=False)

    summaries = []
    for intervention_id, group in predictions.groupby("intervention_id", sort=False):
        combined = group.groupby(["parent_sample_id", "theta_E_arcsec_true"], as_index=False).agg(
            theta_E_arcsec_predicted=("theta_E_arcsec_predicted", "mean")
        )
        summaries.append({
            "intervention_id": intervention_id,
            "seed_count": int(group["seed"].nunique()),
            **regression_metrics(combined["theta_E_arcsec_true"], combined["theta_E_arcsec_predicted"]),
        })
    pd.DataFrame(summaries).to_csv(output / "cnn_summary_metrics.csv", index=False)

    bin_rows = []
    for intervention_id, group in predictions.groupby("intervention_id", sort=False):
        parent_mean = group.groupby(["parent_sample_id", "theta_E_arcsec_true"], as_index=False).agg(
            theta_E_arcsec_predicted=("theta_E_arcsec_predicted", "mean")
        )
        for lower, upper, include_upper in [(0.5, 1.0, False), (1.0, 1.5, False), (1.5, 2.0, True)]:
            mask = (parent_mean["theta_E_arcsec_true"] >= lower) & (
                parent_mean["theta_E_arcsec_true"] <= upper if include_upper else parent_mean["theta_E_arcsec_true"] < upper
            )
            subset = parent_mean.loc[mask]
            bin_rows.append({
                "intervention_id": intervention_id,
                "theta_E_bin_lower_arcsec": lower,
                "theta_E_bin_upper_arcsec": upper,
                "upper_inclusive": include_upper,
                "parent_count": int(len(subset)),
                "metrics": json.dumps(regression_metrics(subset["theta_E_arcsec_true"], subset["theta_E_arcsec_predicted"]) if len(subset) else None),
            })
    pd.DataFrame(bin_rows).to_csv(output / "cnn_radius_bins.csv", index=False)

    source_manifest = pd.read_csv(release / "manifest.csv")
    validation_manifest = source_manifest.loc[source_manifest["split"] == "validation"]
    descriptors = _observation_descriptors(release, validation_manifest)
    descriptors.to_csv(output / "simulator_observation_descriptors.csv", index=False)
    subgroup_rows = []
    pred_parent = predictions.groupby(["intervention_id", "parent_sample_id", "theta_E_arcsec_true"], as_index=False).agg(
        theta_E_arcsec_predicted=("theta_E_arcsec_predicted", "mean"),
        absolute_error_arcsec=("absolute_error_arcsec", "mean"),
    )
    pred_parent = pred_parent.merge(descriptors, on=["intervention_id", "parent_sample_id"], validate="one_to_one")
    for intervention_id, group in pred_parent.groupby("intervention_id", sort=False):
        for label, mask in [
            ("physical_multiplicity_2", group["image_count_physical"] == 2),
            ("physical_multiplicity_3plus", group["image_count_physical"] >= 3),
            ("detectable_multiplicity_2", group["image_count_detectable"] == 2),
            ("detectable_multiplicity_3plus", group["image_count_detectable"] >= 3),
        ]:
            subset = group.loc[mask]
            subgroup_rows.append({
                "intervention_id": intervention_id,
                "stratum": label,
                "parent_count": int(len(subset)),
                **(regression_metrics(subset["theta_E_arcsec_true"], subset["theta_E_arcsec_predicted"]) if len(subset) else {"mae_arcsec": None, "bias_arcsec": None}),
            })
    pd.DataFrame(subgroup_rows).to_csv(output / "cnn_multiplicity_subgroups.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(effects))
    mean = effects["paired_delta_mae_arcsec"].to_numpy()
    lo = effects["bootstrap_95_low_arcsec"].to_numpy()
    hi = effects["bootstrap_95_high_arcsec"].to_numpy()
    ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="o", capsize=3, color="#26547c")
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(float(config["metrics"]["practical_effect_arcsec"]), color="gray", linestyle="--", linewidth=1)
    ax.axhline(-float(config["metrics"]["practical_effect_arcsec"]), color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x, effects["intervention_id"], rotation=35, ha="right")
    ax.set_ylabel("Paired change in absolute error (arcsec)")
    ax.set_title("Observation-systematics effects relative to matched canonical parents")
    fig.tight_layout()
    fig.savefig(output / "paired_effects.png", dpi=170)
    plt.close(fig)

    summary = {
        "experiment_id": "WEEK4-OBSERVATION-SYSTEMATICS",
        "protocol": "archive/docs/reliability_study/protocol_v1.4.md",
        "protocol_sha256": file_sha256(Path("archive/docs/reliability_study/protocol_v1.4.md")),
        "release": release.as_posix(),
        "release_manifest_sha256": file_sha256(release / "manifest.csv"),
        "parent_manifest_sha256": run_manifest["parent_manifest_sha256"],
        "source_parent_count": int(len(canonical_ids)),
        "condition_count_including_canonical": len(intervention_ids),
        "cnn_checkpoint_hashes": {path.as_posix(): file_sha256(path) for path in checkpoints},
        "historical_test_split_accessed": False,
        "alternate_psf_family_tested": False,
        "alternate_psf_family_reason": "The validated observation renderer exposes analytic Gaussian PSFs only.",
        "bootstrap_resamples": int(config["metrics"]["bootstrap_resamples"]),
        "bootstrap_seed": int(config["metrics"]["bootstrap_seed"]),
        "practical_effect_arcsec": float(config["metrics"]["practical_effect_arcsec"]),
        "paired_effects": effect_rows,
        "output_files": [
            "cnn_predictions.csv", "cnn_paired_predictions.csv", "cnn_paired_effects.csv",
            "cnn_metrics_per_seed.csv", "cnn_summary_metrics.csv", "cnn_radius_bins.csv",
            "cnn_multiplicity_subgroups.csv", "simulator_observation_descriptors.csv", "paired_effects.png",
        ],
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# Week 4 — PSF and observation-systematics audit",
        "",
        "Validation-only matched evaluation of the three frozen CNN seeds. All conditions share the same 150 H1 validation parents; repeated noise realizations are clustered by parent.",
        "",
        f"- Independent validation parents: **{len(canonical_ids)}**",
        f"- Frozen CNN seeds: **{len(checkpoints)}**",
        f"- Conditions including canonical: **{len(intervention_ids)}**",
        f"- Practical-effect threshold: **{summary['practical_effect_arcsec']:.3f} arcsec**",
        "- Historical synthetic test split accessed: **no**",
        "- Alternate PSF family: **not available in the validated renderer; width-only comparison**",
        "- The integrated-scene SNR is a simulator-defined diagnostic, not calibrated astronomical SNR.",
        "",
        "## Paired CNN effects",
        "",
        "| Condition | Parents | Paired MAE change (arcsec) | Parent-bootstrap 95% interval | Practical increase supported | Practically small |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    for row in effect_rows:
        lines.append(
            f"| {row['intervention_id']} | {row['parent_count']} | {row['paired_delta_mae_arcsec']:+.4f} | "
            f"[{row['bootstrap_95_low_arcsec']:+.4f}, {row['bootstrap_95_high_arcsec']:+.4f}] | "
            f"{'yes' if row['practically_important_increase_supported'] else 'no'} | {'yes' if row['effect_practically_small'] else 'no'} |"
        )
    lines.extend([
        "",
        "![Paired intervention effects](paired_effects.png)",
        "",
        "The parent bootstrap averages seed-specific paired effects within parent first. `cnn_metrics_per_seed.csv` keeps optimization-seed variation separate. Full per-parent predictions, radius-bin metrics, multiplicity counts, and the simulator SNR proxy are preserved alongside this report.",
        "",
        "PSF-family differences were not tested because the validated renderer supplies only a Gaussian control. Detector noise and variance are simulator abstractions and do not establish calibrated instrument-domain performance. Source-position/detectability interventions remain ordered for Week 5.",
        "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--release", default=str(DEFAULT_RELEASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    summary = evaluate_observation_systematics(
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
