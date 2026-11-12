"""Paired source-only CNN training with exactly matched optimizer budgets.

The randomized arm chooses shape-homogeneous batches. The canonical arm uses
the identical sequence of parent systems, including repetitions, at every
update. Only the view supplied for a parent changes. Primary checkpoints use
the final fixed update, with validation selection recorded as secondary.
Frozen trainers and holdouts are never written or evaluated here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src.ml.data import TorchLensDataset, compute_training_statistics
from src.ml.engine import seed_everything, select_device, write_json
from src.ml.losses import build_loss
from src.ml.metrics import regression_metrics
from src.ml.model import build_model
from src.ml.provenance import environment_record, source_provenance


def paired_schedule(shapes, parents, *, updates, batch_size, seed):
    """Sample views uniformly in expectation, retaining a paired parent list.

    Choosing a bucket proportional to its size and then uniformly within it
    makes each view equally likely. Equal views per parent ensures uniform
    marginal parent probabilities. Pairing makes realized parent exposure
    counts identical between arms, without mixing native image geometries.
    """
    if updates < 1 or batch_size < 1 or len(shapes) != len(parents) or not shapes:
        raise ValueError("Invalid schedule dimensions")
    counts = Counter(parents)
    if len(set(counts.values())) != 1:
        raise ValueError("Every training parent must have the same number of views")
    groups = {}
    for index, shape in enumerate(shapes):
        groups.setdefault(tuple(shape), []).append(index)
    buckets = list(groups.values())
    weights = np.asarray([len(bucket) for bucket in buckets], dtype=float)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    return [rng.choice(buckets[int(rng.choice(len(buckets), p=weights))],
                       size=batch_size, replace=True).tolist() for _ in range(updates)]


def _cache(dataset):
    return [dataset.raw_item(index) for index in range(len(dataset))]


def _normalized_batch(records, indices, stats, device):
    images = np.stack([records[index][0] for index in indices])
    images = (images - stats["image_mean"]) / stats["image_std"]
    targets = np.asarray([records[index][1] for index in indices], dtype=np.float32)
    targets = (targets - stats["target_mean"]) / stats["target_std"]
    return (torch.from_numpy(images[:, None].astype(np.float32)).to(device),
            torch.from_numpy(targets).to(device))


@torch.inference_mode()
def predict(model, records, stats, device, batch_size):
    model.eval()
    values = []
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        images, _ = _normalized_batch(records, indices, stats, device)
        values.extend((model(images).cpu().numpy() * stats["target_std"] + stats["target_mean"]).tolist())
    return pd.DataFrame({"sample_id": [r[2] for r in records],
                         "theta_E_arcsec_true": [r[1] for r in records],
                         "theta_E_arcsec_predicted": values})


def _metrics(frame):
    return regression_metrics(frame.theta_E_arcsec_true.to_numpy(),
                              frame.theta_E_arcsec_predicted.to_numpy())


def _inputs(config):
    """Read only source train and validation products; identify shared parents."""
    canonical = TorchLensDataset(config["canonical_release"], split="train", expected_shape=None)
    randomized = TorchLensDataset(config["randomized_release"], split="train", expected_shape=None)
    validation = TorchLensDataset(config["canonical_release"], split="validation", expected_shape=None)
    stats = compute_training_statistics(canonical)
    standard, varied, val = map(_cache, (canonical, randomized, validation))
    by_id = {record[2]: index for index, record in enumerate(standard)}
    metadata = randomized.dataset.metadata.iloc[randomized.indices]
    parents = metadata["original_parent_sample_id"].astype(str).tolist()
    if set(parents) != set(by_id):
        raise ValueError("Training arms do not have identical parent systems")
    canonical_indices = [by_id[parent] for parent in parents]
    for row, index in zip(varied, canonical_indices):
        if not np.isclose(row[1], standard[index][1], rtol=0, atol=1e-10):
            raise ValueError("Paired views have different targets")
    train_groups = set(canonical.dataset.metadata.iloc[canonical.indices]["parent_system_id"].astype(str))
    validation_groups = set(validation.dataset.metadata.iloc[validation.indices]["parent_system_id"].astype(str))
    if train_groups & validation_groups:
        raise ValueError("Training and validation parent overlap")
    identity = {"training_parent_count": len(by_id), "validation_parent_count": len(val),
                "test_images_accessed": 0, "target_simulator_images_accessed": 0}
    return standard, varied, val, stats, parents, canonical_indices, identity


def train_pair(config, seed, output, inputs, device, source):
    standard, varied, validation, stats, parents, canonical_indices, identity = inputs
    pair_output = output / f"seed-{seed}"
    pair_output.mkdir()
    schedule = paired_schedule([r[0].shape for r in varied], parents,
                               updates=config["optimizer_updates"], batch_size=config["batch_size"],
                               seed=seed + 100000)
    schedule_record = {"randomized_sample_ids": [[varied[i][2] for i in batch] for batch in schedule],
                       "parent_sample_ids": [[parents[i] for i in batch] for batch in schedule]}
    write_json(pair_output / "paired_schedule.json", schedule_record)
    exposure_counts = Counter(parents[i] for batch in schedule for i in batch)
    write_json(pair_output / "parent_exposures.json", dict(exposure_counts))
    runs = []
    for arm, records in (("canonical", standard), ("randomized", varied)):
        folder = pair_output / arm
        folder.mkdir()
        seed_everything(seed)
        model = build_model(config["model"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"])
        loss_function = build_loss({"kind": "huber", "delta": config["huber_delta"]})
        history, best, best_update, pixels = [], float("inf"), 0, 0
        started = time.perf_counter()
        for update, batch in enumerate(schedule, 1):
            indices = batch if arm == "randomized" else [canonical_indices[i] for i in batch]
            images, targets = _normalized_batch(records, indices, stats, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            pixels += int(images.numel())
            if update % config["validation_interval_updates"] and update != len(schedule):
                continue
            frame = predict(model, validation, stats, device, config["batch_size"])
            metrics = _metrics(frame)
            row = {"update": update, "train_loss_last_batch": float(loss.detach().cpu()),
                   "validation_mae_arcsec": metrics["mae_arcsec"]}
            history.append(row)
            checkpoint = {"format_version": 2, "experiment_id": config["experiment_id"],
                          "model_state": model.state_dict(), "model_config": model.model_config,
                          "statistics": stats, "seed": seed, "arm": arm,
                          "optimizer_updates": update, "target_field": "theta_E_arcsec"}
            if metrics["mae_arcsec"] < best:
                best, best_update = metrics["mae_arcsec"], update
                torch.save(checkpoint, folder / "best_validation.pt")
            if update == len(schedule):
                torch.save(checkpoint, folder / "final.pt")
                frame.to_csv(folder / "final_validation_predictions.csv", index=False)
            print(json.dumps({"seed": seed, "arm": arm, **row}), flush=True)
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        summary = {"seed": seed, "arm": arm, "status": "source_validation_development_only",
                   "primary_checkpoint": "final.pt", "primary_validation_metrics": metrics,
                   "best_validation_update_secondary": best_update,
                   "run_controls": {"optimizer_updates": len(schedule), "batch_size": config["batch_size"],
                                    "image_exposures": len(schedule) * config["batch_size"]},
                   "pixels_processed": pixels, "runtime_seconds": time.perf_counter() - started,
                   "source": source, "test_images_accessed": 0, "target_simulator_images_accessed": 0}
        write_json(folder / "summary.json", summary)
        runs.append(summary)
    if runs[0]["run_controls"] != runs[1]["run_controls"]:
        raise ValueError("Paired training controls differ")
    write_json(pair_output / "pair_complete.json", {"runs": runs, "paired_controls_identical": True})
    return runs


def run(config_path, *, device_request="cpu"):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for field in ("optimizer_updates", "batch_size", "validation_interval_updates", "torch_threads"):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if (config["schema_version"] != 1 or config["status"] != "source_validation_development_only"
        or config["normalization"] != "canonical_training_only_shared"
        or config["primary_checkpoint"] != "final_fixed_update"
        or config["parent_sampling"] != "paired_shape_bucket_schedule_with_replacement"):
        raise ValueError("Unsupported scientific controls")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    torch.set_num_threads(config["torch_threads"])
    device = select_device(device_request)
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("No implicit resume or overwrite; choose a new versioned output")
    output.mkdir(parents=True, exist_ok=True)
    source = source_provenance()
    write_json(output / "started.json", {"config": config, "source": source,
                                         "environment": environment_record(device)})
    inputs = _inputs(config)
    write_json(output / "input_identity.json", inputs[-1])
    write_json(output / "normalization.json", inputs[3])
    runs = []
    for seed in config["seeds"]:
        runs.extend(train_pair(config, int(seed), output, inputs, device, source))
    report = {"experiment_id": config["experiment_id"], "status": config["status"],
              "primary_checkpoint": "final_fixed_update", "runs": runs,
              "independent_training_parents": inputs[-1]["training_parent_count"],
              "independent_validation_parents": inputs[-1]["validation_parent_count"],
              "test_images_accessed": 0, "target_simulator_images_accessed": 0,
              "limitations": ["Canonical source validation only; not an independent final result",
                              "Equal updates and parent exposures, unequal native pixel counts",
                              "No claim of cross-simulator transfer or superiority to prior papers"]}
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml/matched_budget.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.config, device_request=args.device)


if __name__ == "__main__":
    main()
