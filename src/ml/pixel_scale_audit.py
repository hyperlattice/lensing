"""Evaluate frozen CNN checkpoints on matched pixel-scale counterfactuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import map_coordinates

from src.ml.data import TorchLensDataset, file_sha256
from src.ml.engine import collect_predictions, make_loader, select_device, write_json
from src.ml.metrics import regression_metrics
from src.ml.model import build_model


def resample_image_to_angular_scale(
    image: np.ndarray,
    *,
    input_pixel_scale_arcsec: float,
    output_pixel_scale_arcsec: float = 0.05,
    output_shape: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Bilinearly sample an image on a centered angular grid."""

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("image must be a finite two-dimensional array")
    input_scale = float(input_pixel_scale_arcsec)
    output_scale = float(output_pixel_scale_arcsec)
    if not np.isfinite(input_scale) or input_scale <= 0 or not np.isfinite(output_scale) or output_scale <= 0:
        raise ValueError("pixel scales must be finite and positive")
    if len(output_shape) != 2 or min(output_shape) <= 0:
        raise ValueError("output_shape must contain two positive dimensions")

    output_y = (np.arange(output_shape[0], dtype=np.float64) - (output_shape[0] - 1) / 2.0) * output_scale
    output_x = (np.arange(output_shape[1], dtype=np.float64) - (output_shape[1] - 1) / 2.0) * output_scale
    input_y = output_y / input_scale + (image.shape[0] - 1) / 2.0
    input_x = output_x / input_scale + (image.shape[1] - 1) / 2.0
    grid_y, grid_x = np.meshgrid(input_y, input_x, indexing="ij")
    sampled = map_coordinates(
        image,
        [grid_y, grid_x],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.asarray(sampled, dtype=np.float32)


class _AngularResampledDataset(torch.utils.data.Dataset):
    """Apply angular resampling before the baseline's train-only normalization."""

    def __init__(self, dataset: TorchLensDataset, statistics: dict):
        self.dataset = dataset
        self.statistics = statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target, sample_id = self.dataset.raw_item(index)
        row = self.dataset.dataset.metadata.iloc[self.dataset.indices[index]]
        image = resample_image_to_angular_scale(
            image,
            input_pixel_scale_arcsec=float(row["pixel_scale"]),
            output_pixel_scale_arcsec=0.05,
            output_shape=(256, 256),
        )
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        image_tensor = (image_tensor - float(self.statistics["image_mean"])) / float(
            self.statistics["image_std"]
        )
        target_normalized = (target - float(self.statistics["target_mean"])) / float(
            self.statistics["target_std"]
        )
        return {
            "image": image_tensor,
            "target": torch.tensor(target_normalized, dtype=torch.float32),
            "target_raw": torch.tensor(target, dtype=torch.float32),
            "sample_id": sample_id,
        }


def _bootstrap_mean(values: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Bootstrap values must be a nonempty vector")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(resamples), len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _predict_one_checkpoint(
    checkpoint_path: Path,
    *,
    release: Path,
    intervention_ids: list[str],
    device: torch.device,
    batch_size: int,
    strategies: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    source_manifest = Path(checkpoint["release"]) / "manifest.csv"
    if file_sha256(source_manifest) != str(checkpoint["release_manifest_sha256"]):
        raise ValueError(f"Baseline manifest hash changed for checkpoint {checkpoint_path}")
    run_manifest = json.loads((release / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest["parent_manifest_sha256"] != checkpoint["release_manifest_sha256"]:
        raise ValueError("Counterfactual benchmark does not descend from the checkpoint's release")

    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    predictions: list[pd.DataFrame] = []
    metrics: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        for intervention_id in intervention_ids:
            split_name = f"validation__{intervention_id}"
            base_dataset = TorchLensDataset(
                release,
                split=split_name,
                target_field="theta_E_arcsec",
                statistics=checkpoint["statistics"],
                expected_shape=None,
            )
            dataset = (
                _AngularResampledDataset(base_dataset, checkpoint["statistics"])
                if strategy == "angular_resampled"
                else base_dataset
            )
            if strategy not in {"native", "angular_resampled"}:
                raise ValueError(f"Unsupported pixel-scale input strategy: {strategy}")
            loader = make_loader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                seed=0,
                num_workers=0,
            )
            result = collect_predictions(model, loader, device, checkpoint["statistics"])
            metric_key = f"{strategy}::{intervention_id}"
            metrics[metric_key] = regression_metrics(result["targets"], result["predictions"])
            metadata = base_dataset.dataset.metadata.iloc[base_dataset.indices][
                [
                    "sample_id",
                    "parent_sample_id",
                    "split",
                    "pixel_scale",
                    "original_shape",
                    "image_count_physical",
                    "image_count_detectable",
                ]
            ]
            frame = pd.DataFrame(
                {
                    "sample_id": result["sample_ids"],
                    "theta_E_arcsec_true": result["targets"],
                    "theta_E_arcsec_predicted": result["predictions"],
                }
            ).merge(metadata, on="sample_id", validate="one_to_one")
            frame["checkpoint_id"] = checkpoint["experiment_id"]
            frame["checkpoint_sha256"] = file_sha256(checkpoint_path)
            frame["intervention_id"] = intervention_id
            frame["input_strategy"] = strategy
            frame["residual_arcsec"] = frame["theta_E_arcsec_predicted"] - frame["theta_E_arcsec_true"]
            frame["absolute_error_arcsec"] = frame["residual_arcsec"].abs()
            frame["relative_error"] = frame["absolute_error_arcsec"] / frame["theta_E_arcsec_true"].abs()
            predictions.append(frame)
    return pd.concat(predictions, ignore_index=True), metrics


def evaluate_pixel_scale_sweep(
    checkpoint_paths: list[str | Path],
    *,
    release: str | Path,
    output_dir: str | Path,
    intervention_ids: list[str],
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 6031,
    practical_effect_arcsec: float = 0.01,
    batch_size: int = 16,
    device_request: str = "auto",
    strategies: list[str] | None = None,
) -> dict:
    """Score fixed checkpoints and paired pixel-scale effects on validation parents."""

    release = Path(release)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if "canonical" not in intervention_ids:
        raise ValueError("Intervention list must include canonical")
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required")
    strategies = ["native"] if strategies is None else list(strategies)
    run_manifest = json.loads((release / "run_manifest.json").read_text(encoding="utf-8"))
    if not run_manifest.get("passed"):
        raise ValueError("Counterfactual benchmark did not pass its generation gates")
    device = select_device(device_request)

    prediction_frames = []
    per_seed_rows = []
    checkpoint_hashes = {}
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        prediction_frame, metrics = _predict_one_checkpoint(
            checkpoint_path,
            release=release,
            intervention_ids=intervention_ids,
            device=device,
            batch_size=batch_size,
            strategies=strategies,
        )
        checkpoint_hashes[checkpoint_path.as_posix()] = file_sha256(checkpoint_path)
        prediction_frames.append(prediction_frame)
        for metric_id, values in metrics.items():
            strategy, _, intervention_id = metric_id.partition("::")
            per_seed_rows.append(
                {
                    "checkpoint_id": str(prediction_frame["checkpoint_id"].iloc[0]),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "seed": int(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["seed"]),
                    "input_strategy": strategy,
                    "intervention_id": intervention_id,
                    **values,
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)

    canonical = predictions.loc[predictions["intervention_id"] == "canonical"].rename(
        columns={
            "theta_E_arcsec_predicted": "theta_E_arcsec_predicted_canonical",
            "absolute_error_arcsec": "absolute_error_arcsec_canonical",
        }
    )[[
        "checkpoint_id",
        "input_strategy",
        "parent_sample_id",
        "theta_E_arcsec_true",
        "theta_E_arcsec_predicted_canonical",
        "absolute_error_arcsec_canonical",
    ]]
    shifted = predictions.loc[predictions["intervention_id"] != "canonical"].merge(
        canonical,
        on=["checkpoint_id", "input_strategy", "parent_sample_id", "theta_E_arcsec_true"],
        validate="many_to_one",
    )
    shifted["paired_delta_absolute_error_arcsec"] = (
        shifted["absolute_error_arcsec"] - shifted["absolute_error_arcsec_canonical"]
    )
    shifted["paired_prediction_shift_arcsec"] = (
        shifted["theta_E_arcsec_predicted"] - shifted["theta_E_arcsec_predicted_canonical"]
    )
    shifted.to_csv(output_dir / "paired_predictions.csv", index=False)

    effect_rows = []
    for (strategy, intervention_id), group in shifted.groupby(["input_strategy", "intervention_id"], sort=True):
        # Average model-seed deltas within each independent parent before
        # resampling parents; seeds are not treated as independent lenses.
        parent_effects = group.groupby("parent_sample_id")["paired_delta_absolute_error_arcsec"].mean()
        low, high = _bootstrap_mean(
            parent_effects.to_numpy(),
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        mean_delta = float(parent_effects.mean())
        practical_increase = mean_delta >= practical_effect_arcsec and low > 0.0
        practically_small = low >= -practical_effect_arcsec and high <= practical_effect_arcsec
        effect_rows.append(
            {
                "intervention_id": intervention_id,
                "input_strategy": strategy,
                "parent_count": int(len(parent_effects)),
                "paired_mean_delta_mae_arcsec": mean_delta,
                "bootstrap_95_low_arcsec": low,
                "bootstrap_95_high_arcsec": high,
                "mean_absolute_prediction_shift_arcsec": float(
                    group.groupby("parent_sample_id")["paired_prediction_shift_arcsec"].mean().abs().mean()
                ),
                "practically_important_increase_supported": bool(practical_increase),
                "effect_practically_small": bool(practically_small),
            }
        )
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(output_dir / "paired_effects.csv", index=False)

    scale_means = (
        predictions.groupby(["input_strategy", "intervention_id", "pixel_scale"], as_index=False)
        .agg(
            mean_mae_arcsec=("absolute_error_arcsec", "mean"),
            mean_bias_arcsec=("residual_arcsec", "mean"),
            mean_p95_arcsec=("absolute_error_arcsec", lambda values: values.quantile(0.95)),
            parent_count=("parent_sample_id", "nunique"),
            seed_count=("checkpoint_id", "nunique"),
        )
    )
    scale_means.to_csv(output_dir / "scale_summary.csv", index=False)

    plot_path = output_dir / "paired_scale_effects.png"
    figure, axis = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(effects))
    means = effects["paired_mean_delta_mae_arcsec"].to_numpy()
    lows = effects["bootstrap_95_low_arcsec"].to_numpy()
    highs = effects["bootstrap_95_high_arcsec"].to_numpy()
    axis.errorbar(x, means, yerr=[means - lows, highs - means], fmt="o", capsize=4)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.axhline(practical_effect_arcsec, color="gray", linestyle="--", linewidth=1)
    axis.set_xticks(
        x,
        effects["input_strategy"] + ": " + effects["intervention_id"],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Paired increase in absolute error (arcsec)")
    axis.set_title("Pixel-scale shift versus canonical sampling")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    summary = {
        "experiment_id": "H1-PIXEL-SCALE",
        "split": "validation",
        "parent_count": int(predictions["parent_sample_id"].nunique()),
        "checkpoint_count": int(predictions["checkpoint_id"].nunique()),
        "intervention_ids": intervention_ids,
        "input_strategies": strategies,
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "practical_effect_arcsec": float(practical_effect_arcsec),
        "checkpoint_sha256": checkpoint_hashes,
        "parent_manifest_sha256": run_manifest["parent_manifest_sha256"],
        "per_seed_metrics_path": (output_dir / "per_seed_metrics.csv").as_posix(),
        "paired_effects_path": (output_dir / "paired_effects.csv").as_posix(),
        "scale_summary_path": (output_dir / "scale_summary.csv").as_posix(),
        "predictions_path": (output_dir / "predictions.csv").as_posix(),
        "plot_path": plot_path.as_posix(),
        "effects": effect_rows,
    }
    write_json(output_dir / "summary.json", summary)
    report_lines = [
        "# Pixel-scale audit — frozen CNN checkpoints",
        "",
        "Validation-only matched counterfactual evaluation. No checkpoint was retrained or selected here.",
        "",
        f"- Independent validation parents: **{summary['parent_count']}**",
        f"- Frozen checkpoints: **{summary['checkpoint_count']}**",
        f"- Practical-effect threshold: **{practical_effect_arcsec:.3f} arcsec**",
        "- Historical synthetic test split accessed: **no**",
        "",
        "## Paired effects",
        "",
        "| Input strategy | Shift | Parents | Mean paired MAE change (arcsec) | Parent-bootstrap 95% interval | Supported >= 0.01 arcsec increase |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for row in effect_rows:
        report_lines.append(
            f"| {row['input_strategy']} | {row['intervention_id']} | {row['parent_count']} | {row['paired_mean_delta_mae_arcsec']:+.4f} | "
            f"[{row['bootstrap_95_low_arcsec']:+.4f}, {row['bootstrap_95_high_arcsec']:+.4f}] | "
            f"{'yes' if row['practically_important_increase_supported'] else 'no'} |"
        )
    report_lines.extend(
        [
            "",
            "The parent-bootstrap interval resamples independent lens parents after averaging paired effects across the three training seeds. Seed variation is listed separately in `per_seed_metrics.csv` and is not a confidence interval.",
            "",
            "The tested scenes hold the current simulator's detector settings fixed while changing angular sampling. This is a controlled sampling stress test, not a calibrated transfer simulation for a different instrument.",
            "",
            "See `scale_summary.csv`, `predictions.csv`, `paired_predictions.csv`, `paired_effects.csv`, and `paired_scale_effects.png` for complete measurements.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reliability_pixel_scale_v1.yaml")
    parser.add_argument("--release", default="data/simulations/reliability-pixel-scale-v1-final")
    parser.add_argument("--output", default="results/reports/reliability_v1/pixel_scale/baseline")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--strategy", action="append", choices=("native", "angular_resampled"))
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary = evaluate_pixel_scale_sweep(
        config["model_checkpoints"],
        release=args.release,
        output_dir=args.output,
        intervention_ids=[item["intervention_id"] for item in config["interventions"]],
        bootstrap_resamples=int(config["metrics"]["bootstrap_resamples"]),
        bootstrap_seed=int(config["metrics"]["bootstrap_seed"]),
        practical_effect_arcsec=float(config["metrics"]["practical_effect_arcsec"]),
        batch_size=args.batch_size,
        device_request=args.device,
        strategies=args.strategy or ["native", "angular_resampled"],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
