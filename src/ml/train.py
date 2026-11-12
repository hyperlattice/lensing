"""Paired fixed-budget C/G/R training; final images and labels are inaccessible."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.dataset import ObservationBank
from src.data.schema import validate_lineage
from src.ml.artifacts import (
    environment_record,
    seed_everything,
    source_provenance,
    verify_resolved_config,
    write_json,
)
from src.ml.data import grouped_forward, source_validation
from src.ml.model import build_model
from src.ml.policy import CELLS, paired_policy_schedule

ARMS = {"C", "G", "R"}


def train_arm(
    config: dict[str, Any],
    *,
    seed: int,
    arm: str,
    source: ObservationBank,
    development: ObservationBank,
    target: ObservationBank | None,
    output: str | Path,
    probabilities: list[float] | np.ndarray | None,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Run a fixed-update development fit; no final scoring or implicit resume."""
    if arm not in ARMS:
        raise ValueError("Unknown comparison arm")
    purpose = config.get("fit_purpose", "development_only")
    if purpose not in {"development_only", "power_pilot_fit", "confirmatory_fit"}:
        raise ValueError("Unknown source-data fit purpose")
    if source.role != "source_train" or development.role != "source_development":
        raise ValueError("Source training and development roles must be distinct")
    for key in ("optimizer_updates", "batch_size", "validation_interval_updates", "torch_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive-integer control: {key}")
    if config["mmd_weight"] != 0:
        raise ValueError("The maintained C/G/R trainer requires zero MMD weight")
    if not allow_fixture and any(
        b.document["stage"] != "qualified_development" for b in (source, development)
    ):
        raise ValueError("Unqualified scientific input bank")
    if target is not None:
        raise ValueError("Source-only arms cannot receive a target bank")
    source_ids = list(source.by_id)
    source_lineage = [*source.parents, *development.parents]
    validate_lineage(source_lineage)
    fingerprints = {}
    for parent in source_lineage:
        fingerprint = parent["latent_id"]
        if fingerprint in fingerprints and fingerprints[fingerprint] != parent["role"]:
            raise ValueError("Physical latent fingerprint overlaps across roles")
        fingerprints[fingerprint] = parent["role"]
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new fit identity")
    output.mkdir(parents=True)
    statistics = source.source_statistics()
    torch.set_num_threads(config["torch_threads"])
    seed_everything(seed)
    model = build_model(config["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    policy = arm
    probabilities = probabilities if policy == "R" else np.ones(len(CELLS)) / len(CELLS)
    schedule = paired_policy_schedule(
        source_ids,
        probabilities,
        updates=config["optimizer_updates"],
        batch_size=config["batch_size"],
        seed=seed + 100000,
    )
    # Target and historical view streams never advance the source-parent stream.
    write_json(
        output / "started.json",
        {
            "seed": seed,
            "arm": arm,
            "config": config,
            "statistics": statistics,
            "target_information_access": None,
            "source": source_provenance(),
            "environment": environment_record(torch.device("cpu")),
        },
    )
    parent_schedule = [row["parents"] for row in schedule]
    write_json(
        output / "sampling_controls.json",
        {
            "parent_schedule": parent_schedule,
            "source_parent_order": source_ids,
            "cell_probabilities": np.asarray(probabilities).tolist(),
            "cell_order": list(CELLS),
            "schedule_seed": seed + 100000,
            "target_seed_sequence": [seed, 50001],
            "historical_seed_sequence": [seed, 50002],
            "parent_exposures": dict(
                Counter(parent_id for row in schedule for parent_id in row["parents"])
            ),
            "scheduled_cell_counts": dict(Counter(row["cell"] for row in schedule)),
            "reconstruction": "paired_policy_schedule with recorded NumPy version; source-only C/N/T ignore scheduled cell",
        },
    )
    history, processed_pixels, started = [], 0, time.perf_counter()
    actual_views = Counter()
    loss_fn = torch.nn.HuberLoss(delta=config["huber_delta"])
    for update, row in enumerate(schedule, 1):
        images = []
        for parent, draw_seed in zip(row["parents"], row["noise_seeds"]):
            if policy == "C":
                image = source.source_image(parent)
                actual_views["canonical_fixed_noise"] += 1
            else:
                image = source.source_image(parent, cell=row["cell"], noise_seed=draw_seed)
                actual_views[row["cell"]] += 1
            images.append(image)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, prediction = grouped_forward(model, images, statistics)
        truth = torch.tensor(
            [
                (source.targets[parent_id] - statistics["target_mean"]) / statistics["target_std"]
                for parent_id in row["parents"]
            ],
            dtype=torch.float32,
        )
        supervised = loss_fn(prediction, truth)
        discrepancy = torch.zeros(())
        loss = supervised + config["mmd_weight"] * discrepancy
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss; retain failed fit")
        loss.backward()
        optimizer.step()
        processed_pixels += sum(image.size for image in images)
        if update % config["validation_interval_updates"] == 0 or update == len(schedule):
            validation = source_validation(
                model, development, statistics, batch_size=config["batch_size"]
            )
            record = {
                "update": update,
                "validation_mae_arcsec": validation["mae_arcsec"],
                "supervised_loss": float(supervised.detach()),
                "mmd_loss": float(discrepancy.detach()),
            }
            history.append(record)
            print(json.dumps({"arm": arm, "seed": seed, **record}), flush=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model.model_config,
        "statistics": statistics,
        "arm": arm,
        "seed": seed,
        "optimizer_updates": len(schedule),
        "target_information_access": None,
    }
    torch.save(checkpoint, output / "final.pt")
    write_json(output / "source_validation_predictions.json", validation)
    write_json(output / "history.json", history)
    report = {
        "stage": purpose,
        "arm": arm,
        "seed": seed,
        "validation_mae_arcsec": validation["mae_arcsec"],
        "optimizer_updates": len(schedule),
        "batch_size": config["batch_size"],
        "source_exposures": len(schedule) * config["batch_size"],
        "target_exposures": 0,
        "pixels_processed": processed_pixels,
        "runtime_seconds": time.perf_counter() - started,
        "final_images_accessed": False,
        "target_information_access": None,
        "target_regression_labels_accessed": False,
        "realized_source_view_counts": dict(actual_views),
    }
    write_json(output / "complete.json", report)
    return report


def checked_fit(folder: str | Path, settings: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Reuse a completed fit only when its configuration, arm and seed match."""
    folder = Path(folder)
    if folder.exists():
        report = json.loads((folder / "complete.json").read_text())
        started = json.loads((folder / "started.json").read_text())
        if (
            started["config"] != settings
            or started["seed"] != kwargs["seed"]
            or started["arm"] != kwargs["arm"]
        ):
            raise ValueError("Existing completed fit identity changed")
        return report
    return train_arm(settings, output=folder, **kwargs)


def fit(config: dict[str, Any]) -> None:
    """Fit the configured paired arms without opening the holdout manifest."""
    output_root = Path(config["output"])
    source_path, development_path = Path(config["source_train"]), Path(config["source_development"])
    policy_path, holdout_path = Path(config["policy"]), Path(config["holdout"])
    seeds, arms = config["seeds"], config["arms"]
    if (output_root / "fit_complete.json").exists():
        verify_resolved_config(config)
        print(json.dumps(json.loads((output_root / "fit_complete.json").read_text()), indent=2))
        return
    output_root.mkdir(parents=True, exist_ok=True)
    source = ObservationBank(source_path, expected_role="source_train")
    development = ObservationBank(development_path, expected_role="source_development")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    settings = config["training"]
    resolved = {"config": config}
    resolved_path = output_root / "resolved_config.json"
    if resolved_path.exists():
        if json.loads(resolved_path.read_text(encoding="utf-8")).get("config") != config:
            raise ValueError("Equal-budget configuration changed; use a new output identity")
    else:
        with resolved_path.open("x", encoding="utf-8") as stream:
            json.dump(resolved, stream, indent=2)
            stream.write("\n")
    records = []
    for seed in seeds:
        for arm in arms:
            folder = output_root / "fits" / arm / f"seed-{seed}"
            checked_fit(
                folder,
                settings,
                seed=seed,
                arm=arm,
                source=source,
                development=development,
                target=None,
                probabilities=policy["probabilities"],
            )
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "folder": str(folder),
                    "checkpoint": str(folder / "final.pt"),
                    "target_labels_accessed": False,
                    "final_images_accessed": False,
                }
            )
    manifest = {
        "schema": "equal_budget_final_v1",
        "stage": "fit_complete_target_free",
        "arms": list(arms),
        "seeds": list(seeds),
        "optimizer_updates": settings["optimizer_updates"],
        "batch_size": settings["batch_size"],
        "source_bank": {"path": str(source_path)},
        "development_bank": {"path": str(development_path)},
        "policy": {"path": str(policy_path)},
        "settings": settings,
        "holdout_manifest_not_read": str(holdout_path),
        "fits": records,
        "target_labels_accessed": False,
        "final_images_accessed": False,
    }
    with (output_root / "fit_complete.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({"stage": "fit_complete", "fits": len(records), "target_labels_accessed": False})
    )
