"""Train the two predeclared H1 pixel-scale mitigation models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

from src.ml.data import TorchLensDataset, file_sha256
from src.ml.engine import seed_everything, select_device, write_json
from src.ml.losses import build_loss
from src.ml.metrics import regression_metrics
from src.ml.model import build_model


DEFAULT_CONFIG = Path("configs/reliability_pixel_scale_training_v1.yaml")
DEFAULT_PIXEL_SCALE_RELEASE = Path("data/simulations/reliability-pixel-scale-train-v1")
DEFAULT_VALIDATION_RELEASE = Path("data/simulations/reliability-pixel-scale-v1-final")
DEFAULT_OUTPUT = Path("results/ml/reliability_pixel_scale_v1")
VALIDATION_STRATA = {
    "canonical": "canonical",
    "low": "pixel-scale-004",
    "high": "pixel-scale-006",
}


def _read_records(release: Path, split: str) -> list[dict]:
    dataset = TorchLensDataset(
        release,
        split=split,
        target_field="theta_E_arcsec",
        expected_shape=None,
    )
    metadata = dataset.dataset.metadata.iloc[dataset.indices].reset_index(drop=True)
    records = []
    for index in range(len(dataset)):
        image, theta_arcsec, sample_id = dataset.raw_item(index)
        row = metadata.iloc[index]
        parent_id = row.get("parent_sample_id")
        if parent_id is None or pd.isna(parent_id) or not str(parent_id):
            parent_id = sample_id
        pixel_scale = float(row["pixel_scale"])
        records.append(
            {
                "image": np.ascontiguousarray(image, dtype=np.float32),
                "theta_arcsec": float(theta_arcsec),
                "sample_id": str(sample_id),
                "parent_sample_id": str(parent_id),
                "pixel_scale_arcsec": pixel_scale,
                "image_shape": tuple(int(value) for value in image.shape),
                "intervention_id": str(row.get("intervention_id", split)),
            }
        )
    return records


class ScaleRecordsDataset(Dataset):
    def __init__(self, records, *, family, statistics):
        self.records = records
        self.family = family
        self.statistics = statistics

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        theta_arcsec = float(record["theta_arcsec"])
        pixel_scale = float(record["pixel_scale_arcsec"])
        if self.family == "pixel_target":
            target_raw = theta_arcsec / pixel_scale
        elif self.family == "metadata":
            target_raw = theta_arcsec
        else:
            raise ValueError(f"Unknown mitigation family: {self.family}")
        image = torch.from_numpy(record["image"]).unsqueeze(0)
        image = (image - float(self.statistics["image_mean"])) / float(
            self.statistics["image_std"]
        )
        target = (target_raw - float(self.statistics["target_mean"])) / float(
            self.statistics["target_std"]
        )
        item = {
            "image": image,
            "target": torch.tensor(target, dtype=torch.float32),
            "target_model_raw": torch.tensor(target_raw, dtype=torch.float32),
            "theta_E_arcsec": torch.tensor(theta_arcsec, dtype=torch.float32),
            "pixel_scale_arcsec": torch.tensor(pixel_scale, dtype=torch.float32),
            "sample_id": record["sample_id"],
            "parent_sample_id": record["parent_sample_id"],
            "intervention_id": record["intervention_id"],
        }
        if self.family == "metadata":
            item["metadata"] = torch.tensor([pixel_scale / 0.05], dtype=torch.float32)
        return item


class ShapeBucketBatchSampler(Sampler):
    """Shuffle within dimensions and then shuffle batches without mixing shapes."""

    def __init__(self, records, *, batch_size, seed):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.groups = {}
        for index, record in enumerate(records):
            self.groups.setdefault(tuple(record["image_shape"]), []).append(index)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for indices in self.groups.values():
            indices = np.asarray(indices, dtype=np.int64)
            rng.shuffle(indices)
            batches.extend(
                indices[start : start + self.batch_size].tolist()
                for start in range(0, len(indices), self.batch_size)
            )
        rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())


def _target_statistics(records, family, image_statistics):
    if family == "pixel_target":
        targets = np.asarray(
            [item["theta_arcsec"] / item["pixel_scale_arcsec"] for item in records],
            dtype=np.float64,
        )
        target_field = "theta_E_pixels"
    elif family == "metadata":
        targets = np.asarray([item["theta_arcsec"] for item in records], dtype=np.float64)
        target_field = "theta_E_arcsec"
    else:
        raise ValueError(f"Unknown mitigation family: {family}")
    standard_deviation = float(targets.std(ddof=0))
    if len(targets) == 0 or not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Scale-diverse training targets have no usable variation")
    return {
        "source_split": "train",
        "sample_count": len(targets),
        "target_field": target_field,
        "target_mean": float(targets.mean()),
        "target_std": standard_deviation,
        "image_mean": float(image_statistics["image_mean"]),
        "image_std": float(image_statistics["image_std"]),
        "image_statistics_source": "baseline train split; reused for both H1 mitigation families",
        "image_statistics_release_manifest_sha256": image_statistics[
            "release_manifest_sha256"
        ],
    }


def _run_validation(model, datasets, statistics, family, device, batch_size):
    model.eval()
    predictions = []
    metrics = {}
    with torch.inference_mode():
        for stratum, dataset in datasets.items():
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            stratum_rows = []
            for batch in loader:
                images = batch["image"].to(device)
                model_input = batch.get("metadata")
                if model_input is not None:
                    model_input = model_input.to(device)
                normalized = model(images, model_input)
                prediction_model_units = (
                    normalized.detach().cpu().numpy() * float(statistics["target_std"])
                    + float(statistics["target_mean"])
                )
                scales = batch["pixel_scale_arcsec"].numpy().astype(float)
                if family == "pixel_target":
                    prediction_arcsec = prediction_model_units * scales
                else:
                    prediction_arcsec = prediction_model_units
                truth = batch["theta_E_arcsec"].numpy().astype(float)
                for index, sample_id in enumerate(batch["sample_id"]):
                    row = {
                        "sample_id": str(sample_id),
                        "parent_sample_id": str(batch["parent_sample_id"][index]),
                        "intervention_id": str(batch["intervention_id"][index]),
                        "stratum": stratum,
                        "pixel_scale_arcsec": float(scales[index]),
                        "theta_E_arcsec_true": float(truth[index]),
                        "theta_E_arcsec_predicted": float(prediction_arcsec[index]),
                    }
                    row["residual_arcsec"] = (
                        row["theta_E_arcsec_predicted"] - row["theta_E_arcsec_true"]
                    )
                    row["absolute_error_arcsec"] = abs(row["residual_arcsec"])
                    stratum_rows.append(row)
            frame = pd.DataFrame(stratum_rows)
            if frame.empty or frame["parent_sample_id"].nunique() != 150:
                raise ValueError(f"Validation stratum {stratum} must contain 150 parents")
            metrics[stratum] = regression_metrics(
                frame["theta_E_arcsec_true"].to_numpy(),
                frame["theta_E_arcsec_predicted"].to_numpy(),
            )
            predictions.extend(stratum_rows)
    metrics["equal_weight_mean_mae_arcsec"] = float(
        np.mean([metrics[key]["mae_arcsec"] for key in VALIDATION_STRATA])
    )
    return metrics, pd.DataFrame(predictions)


def _train_seed(
    *,
    config,
    family,
    seed,
    train_records,
    validation_datasets,
    image_statistics,
    output_root,
    device,
):
    output_dir = output_root / family / f"seed-{seed}"
    if output_dir.exists() and (output_dir / "training_summary.json").exists():
        return json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Partial output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    statistics = _target_statistics(train_records, family, image_statistics)
    training = ScaleRecordsDataset(train_records, family=family, statistics=statistics)
    batch_sampler = ShapeBucketBatchSampler(
        train_records, batch_size=int(config["batch_size"]), seed=seed
    )
    training_loader = DataLoader(training, batch_sampler=batch_sampler, num_workers=0)
    candidate_validation = {
        name: ScaleRecordsDataset(records, family=family, statistics=statistics)
        for name, records in validation_datasets.items()
    }

    model_config = dict(config["model"])
    model_config["metadata_features"] = 1 if family == "metadata" else 0
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_function = build_loss({"kind": "huber", "delta": float(config["huber_delta"])})
    history = []
    best_selection_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, int(config["epochs"]) + 1):
        batch_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        count = 0
        for batch in training_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            metadata = batch.get("metadata")
            if metadata is not None:
                metadata = metadata.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images, metadata)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(images)
            count += len(images)
        validation_metrics, _ = _run_validation(
            model,
            candidate_validation,
            statistics,
            family,
            device,
            int(config["batch_size"]),
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / count,
            "validation_equal_weight_mean_mae_arcsec": validation_metrics[
                "equal_weight_mean_mae_arcsec"
            ],
            "validation_canonical_mae_arcsec": validation_metrics["canonical"]["mae_arcsec"],
            "validation_004_mae_arcsec": validation_metrics["low"]["mae_arcsec"],
            "validation_006_mae_arcsec": validation_metrics["high"]["mae_arcsec"],
        }
        history.append(row)
        print(json.dumps({"family": family, "seed": seed, **row}, sort_keys=True), flush=True)
        if row["validation_equal_weight_mean_mae_arcsec"] < best_selection_score:
            best_selection_score = row["validation_equal_weight_mean_mae_arcsec"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "format_version": 1,
                    "experiment_id": f"H1-PIXEL-SCALE-{family.upper()}",
                    "family": family,
                    "seed": int(seed),
                    "selection_metric": "equal_weight_mean_validation_mae_across_0.04_0.05_0.06_arcsec_per_pixel",
                    "selection_score_arcsec": best_selection_score,
                    "model_state": model.state_dict(),
                    "model_config": model.model_config,
                    "statistics": statistics,
                    "best_epoch": best_epoch,
                    "training_record_count": len(train_records),
                    "training_parent_count": len(
                        {item["parent_sample_id"] for item in train_records}
                    ),
                    "validation_parent_count_per_stratum": 150,
                    "test_samples_accessed": 0,
                    "training_release_manifest_sha256": config[
                        "training_release_manifest_sha256"
                    ],
                    "validation_release_manifest_sha256": config[
                        "validation_release_manifest_sha256"
                    ],
                    "baseline_checkpoint_sha256": config["baseline_checkpoint_sha256"],
                    "validation_metrics": validation_metrics,
                    "torch_version": str(torch.__version__),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= int(config["patience"]):
            break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    selected_metrics, selected_predictions = _run_validation(
        model,
        candidate_validation,
        statistics,
        family,
        device,
        int(config["batch_size"]),
    )
    selected_predictions["family"] = family
    selected_predictions["seed"] = int(seed)
    selected_predictions["checkpoint_sha256"] = file_sha256(checkpoint_path)
    selected_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    summary = {
        "experiment_id": f"H1-PIXEL-SCALE-{family.upper()}",
        "family": family,
        "seed": int(seed),
        "status": "trained_validation_selected",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_selection_score_arcsec": best_selection_score,
        "train_record_count": len(train_records),
        "train_parent_count": len({item["parent_sample_id"] for item in train_records}),
        "validation_parents_per_scale": 150,
        "test_samples_accessed": 0,
        "trainable_parameters": model.trainable_parameter_count,
        "statistics": statistics,
        "validation_metrics": selected_metrics,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "validation_predictions_path": (output_dir / "validation_predictions.csv").as_posix(),
        "validation_predictions_sha256": file_sha256(output_dir / "validation_predictions.csv"),
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


def train_scale_aware_models(
    *,
    config_path=DEFAULT_CONFIG,
    pixel_scale_release=DEFAULT_PIXEL_SCALE_RELEASE,
    validation_release=DEFAULT_VALIDATION_RELEASE,
    output_root=DEFAULT_OUTPUT,
    family="both",
    device_request="auto",
):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    pixel_scale_release = Path(pixel_scale_release)
    validation_release = Path(validation_release)
    output_root = Path(output_root)
    run = json.loads((pixel_scale_release / "run_manifest.json").read_text(encoding="utf-8"))
    if not run.get("passed"):
        raise ValueError("Scale-diverse training counterfactual release failed its generation gates")
    validation_run = json.loads((validation_release / "run_manifest.json").read_text(encoding="utf-8"))
    if not validation_run.get("passed"):
        raise ValueError("Validation pixel-scale release failed its generation gates")

    selected_families = ["pixel_target", "metadata"] if family == "both" else [family]
    checkpoint_paths = [Path(item) for item in config["baseline_checkpoints"]]
    baseline_checkpoints = [
        torch.load(path, map_location="cpu", weights_only=True) for path in checkpoint_paths
    ]
    stats = [item["statistics"] for item in baseline_checkpoints]
    image_statistics = stats[0]
    if any(
        float(item["image_mean"]) != float(image_statistics["image_mean"])
        or float(item["image_std"]) != float(image_statistics["image_std"])
        for item in stats[1:]
    ):
        raise ValueError("Baseline seeds do not share the frozen train-only image statistics")
    if any(int(item["seed"]) != expected for item, expected in zip(baseline_checkpoints, (12001, 12002, 12003))):
        raise ValueError("Baseline checkpoint seeds do not match the frozen seed set")
    if run["parent_manifest_sha256"] != baseline_checkpoints[0]["release_manifest_sha256"]:
        raise ValueError("Training counterfactual release does not descend from the baseline release")
    if validation_run["parent_manifest_sha256"] != baseline_checkpoints[0]["release_manifest_sha256"]:
        raise ValueError("Validation release does not descend from the baseline release")

    print("Loading scale-diverse train records...", flush=True)
    canonical_train = _read_records(Path(baseline_checkpoints[0]["release"]), "train")
    shifted_train = _read_records(pixel_scale_release, "train")
    train_records = canonical_train + shifted_train
    train_parents = {item["parent_sample_id"] for item in train_records}
    if len(canonical_train) != 700 or len(shifted_train) != 700 or len(train_parents) != 700:
        raise ValueError("Expected 700 matched train parents with two scale records each")
    if {item["parent_sample_id"] for item in canonical_train} != {
        item["parent_sample_id"] for item in shifted_train
    }:
        raise ValueError("Canonical and shifted training records do not have identical parents")

    print("Loading validation scale strata...", flush=True)
    validation_records = {
        name: _read_records(validation_release, f"validation__{intervention_id}")
        for name, intervention_id in VALIDATION_STRATA.items()
    }
    expected_parents = None
    for name, records in validation_records.items():
        parent_ids = {item["parent_sample_id"] for item in records}
        if len(records) != 150 or len(parent_ids) != 150:
            raise ValueError(f"Validation stratum {name} must contain 150 unique parents")
        if expected_parents is None:
            expected_parents = parent_ids
        elif parent_ids != expected_parents:
            raise ValueError("Validation scale strata do not contain identical parents")
    if expected_parents & train_parents:
        raise ValueError("Train and validation parent identities overlap")

    config.update(
        {
            "training_release_manifest_sha256": file_sha256(pixel_scale_release / "manifest.csv"),
            "validation_release_manifest_sha256": file_sha256(validation_release / "manifest.csv"),
            "baseline_checkpoint_sha256": {
                path.as_posix(): file_sha256(path) for path in checkpoint_paths
            },
        }
    )
    device = select_device(device_request)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for candidate_family in selected_families:
        for seed in (12001, 12002, 12003):
            summaries.append(
                _train_seed(
                    config=config,
                    family=candidate_family,
                    seed=seed,
                    train_records=train_records,
                    validation_datasets=validation_records,
                    image_statistics=image_statistics,
                    output_root=output_root,
                    device=device,
                )
            )
    result = {
        "experiment_id": config["experiment_id"],
        "families": selected_families,
        "device": str(device),
        "training_parents": len(train_parents),
        "training_records": len(train_records),
        "validation_parents_per_scale": 150,
        "validation_strata": list(VALIDATION_STRATA),
        "test_samples_accessed": 0,
        "summaries": summaries,
    }
    write_json(output_root / "run_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pixel-scale-release", default=str(DEFAULT_PIXEL_SCALE_RELEASE))
    parser.add_argument("--validation-release", default=str(DEFAULT_VALIDATION_RELEASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--family", choices=("both", "pixel_target", "metadata"), default="both")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = train_scale_aware_models(
        config_path=args.config,
        pixel_scale_release=args.pixel_scale_release,
        validation_release=args.validation_release,
        output_root=args.output,
        family=args.family,
        device_request=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
