"""Source-only condition allocation and paired parent schedules."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.ml.model import build_model
from src.simulation.observation import CELLS


def failure_weights(
    canonical_errors: np.ndarray,
    transformed_errors: np.ndarray,
    *,
    role: str,
    epsilon: float = 0.25,
) -> dict[str, Any]:
    """Errors have [seed,parent] and [seed,parent,cell] paired axes, in arcsec.

    Caller must use the same identities in both arrays and record their IDs.
    Negative paired degradation is clipped only after averaging across parents
    and pilot seeds. This does not accept target-domain or final-role scores.
    """
    if role != "source_development":
        raise ValueError("Failure weighting requires source-development errors")
    canonical_error_array, transformed_error_array = (
        np.asarray(canonical_errors, dtype=float),
        np.asarray(transformed_errors, dtype=float),
    )
    if (
        canonical_error_array.ndim != 2
        or min(canonical_error_array.shape) < 1
        or transformed_error_array.shape != (*canonical_error_array.shape, len(CELLS))
        or not np.isfinite(canonical_error_array).all()
        or not np.isfinite(transformed_error_array).all()
        or np.any(canonical_error_array < 0)
        or np.any(transformed_error_array < 0)
    ):
        raise ValueError("Expected paired nonnegative finite seed/parent/cell errors")
    if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("Exploration fraction must be in (0,1]")
    degradation = (transformed_error_array - canonical_error_array[..., None]).mean(axis=(0, 1))
    positive_degradation = np.maximum(degradation, 0)
    if positive_degradation.sum() == 0:
        probabilities = np.full(len(CELLS), 1 / len(CELLS))
    else:
        probabilities = (
            epsilon / len(CELLS) + (1 - epsilon) * positive_degradation / positive_degradation.sum()
        )
    return {
        "cells": list(CELLS),
        "degradation_arcsec": degradation.tolist(),
        "probabilities": probabilities.tolist(),
        "epsilon": float(epsilon),
        "role": role,
    }


def paired_policy_schedule(
    parents: list[str],
    probabilities: list[float] | np.ndarray,
    *,
    updates: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Full batches share one native geometry; parent draws do not depend on policy.

    Independent RNG streams for parents, cells and detector seeds mean changing
    weights cannot change parent exposure. A cell per batch avoids resizing or
    padding native geometries. Draws are reproducible with replacement.
    """
    if (
        not parents
        or len(set(parents)) != len(parents)
        or any(
            not isinstance(cell_probabilities, str) or not cell_probabilities
            for cell_probabilities in parents
        )
        or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (updates, batch_size)
        )
    ):
        raise ValueError("Expected unique parent IDs and positive integer schedule sizes")
    cell_probabilities = np.asarray(probabilities, dtype=float)
    if (
        cell_probabilities.shape != (len(CELLS),)
        or not np.isfinite(cell_probabilities).all()
        or np.any(cell_probabilities < 0)
        or not np.isclose(cell_probabilities.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Invalid cell probabilities")
    streams = np.random.SeedSequence(seed).spawn(3)
    parent_rng, cell_rng, noise_rng = [np.random.default_rng(stream) for stream in streams]
    records = []
    for _ in range(updates):
        indices = parent_rng.integers(len(parents), size=batch_size)
        records.append(
            {
                "parents": [parents[parent_index] for parent_index in indices],
                "cell": CELLS[int(cell_rng.choice(len(CELLS), p=cell_probabilities))],
                "noise_seeds": [
                    int(noise_seed)
                    for noise_seed in noise_rng.integers(0, 2**63, size=batch_size, dtype=np.int64)
                ],
            }
        )
    return deepcopy(records)


@torch.inference_mode()
def source_error_matrix(
    checkpoint_path: str | Path, bank: Any, *, noise_replicates: int = 2, batch_size: int = 16
) -> tuple[np.ndarray, dict[str, Any]]:
    """Measure source-development absolute errors with explicit parent/condition axes."""
    if bank.role != "source_development":
        raise ValueError("Policy/HPO scores require source development, not target or final")
    if type(noise_replicates) is not int or noise_replicates < 1:
        raise ValueError("Expected positive noise replicate count")
    from src.ml.data import grouped_forward

    started = time.perf_counter()
    torch.set_num_threads(4)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    statistics = checkpoint["statistics"]
    parent_ids = list(bank.by_id)
    conditions = ["canonical", *CELLS]
    errors = np.zeros((len(parent_ids), len(conditions)))
    for condition_index, condition in enumerate(conditions):
        for replicate in range(noise_replicates):
            for start in range(0, len(parent_ids), batch_size):
                chosen = parent_ids[start : start + batch_size]
                images = []
                for parent in chosen:
                    seed = int(
                        np.random.SeedSequence(
                            [
                                bank.by_id[parent]["parameter_seed"],
                                720610,
                                condition_index,
                                replicate,
                            ]
                        ).generate_state(1)[0]
                    )
                    images.append(
                        bank.source_image(
                            parent,
                            cell="fresh-noise" if condition == "canonical" else condition,
                            noise_seed=seed,
                        )
                    )
                _, predicted_radii = grouped_forward(model, images, statistics)
                predicted_radii = (
                    predicted_radii.numpy() * statistics["target_std"] + statistics["target_mean"]
                )
                true_radii = np.asarray([bank.targets[parent_id] for parent_id in chosen])
                errors[start : start + len(chosen), condition_index] += (
                    abs(predicted_radii - true_radii) / noise_replicates
                )
    return errors, {
        "role": "source_development",
        "parent_ids": parent_ids,
        "conditions": conditions,
        "noise_replicates": noise_replicates,
        "noise_seed_namespace": 720610,
        "selection_criterion": "mean_absolute_error_canonical_plus_seven_source_cells",
        "canonical_mae_arcsec": float(errors[:, 0].mean()),
        "shared_source_score_arcsec": float(errors.mean()),
        "runtime_seconds": time.perf_counter() - started,
        "inference_torch_threads": 4,
    }


def score_allocation(checkpoints: list[Path], bank: Any, *, output: str | Path) -> dict[str, Any]:
    """Estimate and save R probabilities from at least three canonical reference fits."""
    started = time.perf_counter()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Policy scores already exist; use a new version")
    output.mkdir(parents=True)
    matrices, records = [], []
    for path in checkpoints:
        errors, record = source_error_matrix(path, bank)
        matrices.append(errors)
        records.append(record)
    if len(matrices) < 3:
        raise ValueError("Allocation needs at least three independent reference fits")
    if any(
        record["parent_ids"] != records[0]["parent_ids"]
        or record["conditions"] != records[0]["conditions"]
        for record in records
    ):
        raise ValueError("Reference axes differ")
    cube = np.stack(matrices)
    policy = failure_weights(cube[..., 0], cube[..., 1:], role="source_development", epsilon=0.25)
    np.save(output / "absolute_errors.npy", cube)
    policy.update(
        reference_scores=records,
        stage="source_only_allocation_not_method_freeze",
        target_images_used=False,
        final_images_used=False,
        policy_scoring_runtime_seconds=time.perf_counter() - started,
        reference_fit_cost_accounting="Separate three C reference complete.json runtimes plus scoring cost; not included in per-candidate equal source exposures",
    )
    (output / "policy.json").write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n")
    return policy


def develop(stage: str, config_path: str | Path) -> None:
    """Estimate allocation or compare source-only hyperparameters on development parents.

    The grid and iteration order match the retained source-only part of the
    earlier development study. Review is descriptive and does not freeze a
    method or inspect final data.
    """
    from src.data.dataset import ObservationBank
    from src.ml.artifacts import load_development_config
    from src.ml.train import checked_fit

    config = load_development_config(config_path)
    if config["stage"] != "development_only_no_final_access":
        raise ValueError("Development requires a source-only configuration")
    root = Path(config["output"])
    if stage == "review":
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "tuning").glob("*/trial-*/seed-*/source_selection.json"))
        ]
        if not records:
            raise ValueError("No completed source-only tuning records")
        groups = {}
        for record in records:
            key = (record["arm"], record["trial"])
            groups.setdefault(key, []).append(record)
        selected = {}
        for arm in ("C", "G", "R"):
            trials = []
            for (trial_arm, trial), runs in groups.items():
                if trial_arm != arm:
                    continue
                if sorted(run["seed"] for run in runs) != sorted(config["tuning_seeds"]):
                    raise ValueError("Tuning trial has incomplete seed support")
                score = float(
                    np.mean([run["source_score"]["shared_source_score_arcsec"] for run in runs])
                )
                trials.append(
                    {"trial": trial, "score_arcsec": score, "settings": runs[0]["settings"]}
                )
            if not trials:
                raise ValueError(f"No completed tuning trials for {arm}")
            selected[arm] = min(trials, key=lambda trial: (trial["score_arcsec"], trial["trial"]))
        from src.ml.artifacts import write_json

        write_json(
            root / "source_tuning_review.json",
            {
                "stage": "source_only_review_not_freeze",
                "independently_selected": selected,
                "target_prediction_errors_used": False,
                "final_images_used": False,
                "requires": "Review search boundaries and convergence before choosing final settings",
            },
        )
        return

    source = ObservationBank(config["source_train"], expected_role="source_train")
    development = ObservationBank(config["source_development"], expected_role="source_development")
    if stage == "reference":
        for seed in config["reference_seeds"]:
            checked_fit(
                root / "reference" / f"seed-{seed}",
                config,
                seed=seed,
                arm="C",
                source=source,
                development=development,
                target=None,
                probabilities=None,
            )
    elif stage == "weights":
        checkpoints = [
            root / "reference" / f"seed-{seed}" / "final.pt" for seed in config["reference_seeds"]
        ]
        score_allocation(checkpoints, development, output=root / "allocation")
    elif stage == "tune":
        policy = json.loads((root / "allocation" / "policy.json").read_text(encoding="utf-8"))
        if policy["target_images_used"] or policy["final_images_used"]:
            raise ValueError("Allocation must use source development only")
        records = []
        for arm in ("C", "G", "R"):
            trial = 0
            for learning_rate in (0.0003, 0.0006, 0.001, 0.003):
                for weight_decay in (0.0, 0.0001, 0.001):
                    settings = {
                        **config,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                    }
                    for seed in config["tuning_seeds"]:
                        folder = root / "tuning" / arm / f"trial-{trial:02d}" / f"seed-{seed}"
                        result = checked_fit(
                            folder,
                            settings,
                            seed=seed,
                            arm=arm,
                            source=source,
                            development=development,
                            target=None,
                            probabilities=policy["probabilities"],
                        )
                        _, score = source_error_matrix(folder / "final.pt", development)
                        record = {
                            "arm": arm,
                            "information_access": arm,
                            "trial": trial,
                            "seed": seed,
                            "settings": settings,
                            "source_score": score,
                            "run": result,
                        }
                        from src.ml.artifacts import write_json

                        write_json(folder / "source_selection.json", record)
                        records.append(record)
                    trial += 1
        from src.ml.artifacts import create_once

        create_once(
            root / "tuning_initial_complete.json",
            {
                "records": records,
                "initial_configs_per_arm": 12,
                "seeds_per_trial": len(config["tuning_seeds"]),
                "status": "development_only_requires_boundary_and_convergence_review",
            },
        )
    else:
        raise ValueError(f"Unknown development stage: {stage}")
