"""Integrated development fits for C/N/G/T/R and their matched MMD ports.

Only qualified source train/development and unlabeled adaptation banks are
accepted. This module cannot load a final evaluation bank. Every optimizer
update sees the identical prescribed source-parent batch across policies.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.ml.adaptive_mmd_v1 import adaptive_gaussian_mmd
from src.ml.augmentation_policy_v1 import CELLS, paired_policy_schedule
from src.ml.domain_adaptation_v1 import image_features_and_prediction
from src.ml.engine import seed_everything, write_json
from src.ml.model import build_model
from src.ml.provenance import source_provenance, environment_record

ARMS = {"C", "N", "G", "T", "R", "C+MMD", "G+MMD", "R+MMD"}


def grouped_forward(model, images, statistics):
    """Retain native geometries, one full optimizer batch via grouped forwards."""
    groups = defaultdict(list)
    for i, a in enumerate(images):
        groups[a.shape].append(i)
    features, predictions = [None]*len(images), [None]*len(images)
    for indices in groups.values():
        arrays = np.stack([images[i] for i in indices])
        arrays = (arrays-statistics["image_mean"])/statistics["image_std"]
        x = torch.from_numpy(arrays[:, None].astype(np.float32))
        z, y = image_features_and_prediction(model, x)
        for i, row in enumerate(indices):
            features[row], predictions[row] = z[i], y[i]
    return torch.stack(features), torch.stack(predictions)


@torch.inference_mode()
def source_validation(model, bank, statistics, *, batch_size):
    model.eval()
    ids, predicted = list(bank.by_id), []
    for start in range(0, len(ids), batch_size):
        _, y = grouped_forward(model, [bank.source_image(p) for p in ids[start:start+batch_size]], statistics)
        predicted.extend((y.numpy()*statistics["target_std"]+statistics["target_mean"]).tolist())
    true = np.asarray([bank.targets[p] for p in ids])
    return {"mae_arcsec": float(np.mean(abs(np.asarray(predicted)-true))),
            "parent_ids": ids, "prediction_arcsec": predicted}


def train_arm(config, *, seed, arm, source, development, target, output, probabilities, allow_fixture=False):
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
    if not np.isfinite(config["mmd_weight"]) or config["mmd_weight"] < 0:
        raise ValueError("MMD weight must be nonnegative and finite")
    if not allow_fixture and any(b.document["stage"] != "qualified_development" for b in (source, development)):
        raise ValueError("Unqualified scientific input bank")
    adaptation = arm.endswith("+MMD")
    if adaptation and (target is None or target.role != "target_adaptation"):
        raise ValueError("MMD requires an unlabeled adaptation bank")
    if adaptation and not allow_fixture and target.document["stage"] != "qualified_development":
        raise ValueError("Unqualified adaptation bank")
    if not adaptation and target is not None:
        raise ValueError("Source-only arms cannot receive a target bank")
    source_ids = list(source.by_id)
    source_lineage = [*source.parents, *development.parents]
    if adaptation:
        source_lineage.extend(target.parents)
    from src.data.benchmark_contract_v1 import validate_lineage
    validate_lineage(source_lineage)
    fingerprints={}
    for parent in source_lineage:
        fingerprint=parent["latent_id"]
        if fingerprint in fingerprints and fingerprints[fingerprint]!=parent["role"]:
            raise ValueError("Physical latent fingerprint overlaps across roles")
        fingerprints[fingerprint]=parent["role"]
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new fit identity")
    output.mkdir(parents=True)
    stats = source.source_statistics()
    torch.set_num_threads(config["torch_threads"])
    seed_everything(seed)
    model = build_model(config["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    policy = arm.split("+")[0]
    probabilities = probabilities if policy == "R" else np.ones(len(CELLS))/len(CELLS)
    schedule = paired_policy_schedule(source_ids, probabilities, updates=config["optimizer_updates"],
                                     batch_size=config["batch_size"], seed=seed+100000)
    # Target and historical view streams never advance the source-parent stream.
    target_rng = np.random.default_rng(np.random.SeedSequence([seed, 50001]))
    historical_rng = np.random.default_rng(np.random.SeedSequence([seed, 50002]))
    write_json(output/"started.json", {"seed": seed, "arm": arm, "config": config,
        "statistics": stats, "target_information_access":None if target is None else target.document["condition_access"],
        "source": source_provenance(), "environment": environment_record(torch.device("cpu"))})
    parent_schedule = [row["parents"] for row in schedule]
    write_json(output/"sampling_controls.json",{
        "parent_schedule":parent_schedule,
        "source_parent_order":source_ids,"cell_probabilities":np.asarray(probabilities).tolist(),
        "cell_order":list(CELLS),"schedule_seed":seed+100000,
        "target_seed_sequence":[seed,50001],"historical_seed_sequence":[seed,50002],
        "parent_exposures":dict(Counter(p for row in schedule for p in row["parents"])),
        "scheduled_cell_counts":dict(Counter(row["cell"] for row in schedule)),
        "reconstruction":"paired_policy_schedule with recorded NumPy version; source-only C/N/T ignore scheduled cell"})
    history, pixels, started = [], 0, time.perf_counter()
    actual_views=Counter()
    loss_fn = torch.nn.HuberLoss(delta=config["huber_delta"])
    target_ids = list(target.by_id) if adaptation else []
    for update, row in enumerate(schedule, 1):
        images = []
        for parent, draw_seed in zip(row["parents"], row["noise_seeds"]):
            if policy == "C":
                a = source.source_image(parent)
                actual_views["canonical_fixed_noise"]+=1
            elif policy == "N":
                a = source.source_image(parent, noise_seed=draw_seed)
                actual_views["canonical_fresh_noise"]+=1
            elif policy == "T":
                historical=bool(historical_rng.integers(2))
                a = source.source_image(parent, historical=historical)
                key="historical_"+json.dumps(source.by_id[parent]["historical_condition"],sort_keys=True) if historical else "canonical_fixed_noise"
                actual_views[key]+=1
            else:
                a = source.source_image(parent, cell=row["cell"], noise_seed=draw_seed)
                actual_views[row["cell"]]+=1
            images.append(a)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        zs, prediction = grouped_forward(model, images, stats)
        truth = torch.tensor([(source.targets[p]-stats["target_mean"])/stats["target_std"] for p in row["parents"]], dtype=torch.float32)
        supervised = loss_fn(prediction, truth)
        discrepancy = torch.zeros(())
        if adaptation:
            chosen = target_rng.integers(len(target_ids), size=config["batch_size"])
            views = target_rng.integers(2**31, size=config["batch_size"])
            target_images = [target.target_image(target_ids[i], int(v)) for i,v in zip(chosen, views)]
            zt, _ = grouped_forward(model, target_images, stats)
            discrepancy = adaptive_gaussian_mmd(zs, zt)
            pixels += sum(a.size for a in target_images)
        loss = supervised + config["mmd_weight"]*discrepancy
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss; retain failed fit")
        loss.backward()
        optimizer.step()
        pixels += sum(a.size for a in images)
        if update % config["validation_interval_updates"] == 0 or update == len(schedule):
            val = source_validation(model, development, stats, batch_size=config["batch_size"])
            record = {"update": update, "validation_mae_arcsec": val["mae_arcsec"],
                      "supervised_loss": float(supervised.detach()), "mmd_loss": float(discrepancy.detach())}
            history.append(record)
            print(json.dumps({"arm": arm, "seed": seed, **record}), flush=True)
    checkpoint = {"model_state": model.state_dict(), "model_config": model.model_config,
                  "statistics": stats, "arm": arm, "seed": seed, "optimizer_updates": len(schedule),
                  "target_information_access":None if target is None else target.document["condition_access"]}
    torch.save(checkpoint, output/"final.pt")
    write_json(output/"source_validation_predictions.json", val)
    write_json(output/"history.json", history)
    report = {"stage": purpose, "arm": arm, "seed": seed,
              "validation_mae_arcsec": val["mae_arcsec"], "optimizer_updates": len(schedule),
              "batch_size": config["batch_size"], "source_exposures": len(schedule)*config["batch_size"],
              "target_exposures": len(schedule)*config["batch_size"] if adaptation else 0,
              "pixels_processed": pixels, "runtime_seconds": time.perf_counter()-started,
              "final_images_accessed": False,
              "target_information_access":None if target is None else target.document["condition_access"],
              "target_regression_labels_accessed": False,"realized_source_view_counts":dict(actual_views)}
    write_json(output/"complete.json", report)
    return report
