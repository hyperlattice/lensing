"""Resume the frozen source-only tuning grid at completed fit boundaries.

Completed fits are verified and reused. An interrupted fit directory is
preserved, and its deterministic retry is written to a new sibling identity.
No target performance or final-evaluation data are read by this runner.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.run_robustness_development_v1 import trial_grid
from src.data.benchmark_contract_v1 import sha256
from src.data.observation_bank_v1 import ObservationBank
from src.ml.robustness_training_v1 import ARMS, train_arm
from src.ml.source_policy_scoring_v1 import source_error_matrix


def atomic_json(path: Path, document):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def pick_fit_path(root: Path, base: Path, settings, seed, arm):
    if not base.exists():
        return base
    complete = base / "complete.json"
    if complete.exists():
        started = json.loads((base / "started.json").read_text(encoding="utf-8"))
        report = json.loads(complete.read_text(encoding="utf-8"))
        if (started["config"] != settings or started["seed"] != seed or started["arm"] != arm or
                sha256(base / "final.pt") != report["checkpoint_sha256"] or
                sha256(base / "started.json") != report["started_sha256"] or
                sha256(base / "sampling_controls.json") != report["sampling_controls_sha256"]):
            raise ValueError(f"Completed fit provenance mismatch: {base}")
        return base
    attempt = 1
    while True:
        retry = base.with_name(base.name + f"-retry-{attempt:02d}")
        if not retry.exists():
            return retry
        if (retry / "complete.json").exists():
            started = json.loads((retry / "started.json").read_text(encoding="utf-8"))
            report = json.loads((retry / "complete.json").read_text(encoding="utf-8"))
            if (started["config"] != settings or started["seed"] != seed or started["arm"] != arm or
                    sha256(retry / "final.pt") != report["checkpoint_sha256"]):
                raise ValueError(f"Retry fit provenance mismatch: {retry}")
            return retry
        attempt += 1


def run(config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("stage") != "development_only_no_final_access":
        raise ValueError("Tuning resume is restricted to source-only development")
    root = Path(config["output"])
    final_summary = root / "tuning_initial_complete.json"
    if final_summary.exists():
        raise FileExistsError("Initial tuning summary already exists; do not rerun it")
    source = ObservationBank(config["source_train"], expected_role="source_train")
    dev = ObservationBank(config["source_development"], expected_role="source_development")
    target = ObservationBank(config["target_gaussian"], expected_role="target_adaptation")
    representative = ObservationBank(config["target_representative"], expected_role="target_adaptation")
    policy_path = root / "allocation" / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("target_images_used") is not False or policy.get("final_images_used") is not False:
        raise ValueError("Allocation policy is not source-only")
    experiments = [(arm, None if "+MMD" not in arm else target, arm) for arm in sorted(ARMS)]
    experiments += [("G+MMD", representative, "G+MMD-representative"),
                    ("R+MMD", representative, "R+MMD-representative")]
    records = []
    for arm, adaptation, label in experiments:
        for trial, settings in enumerate(trial_grid(arm, config)):
            for seed in config["tuning_seeds"]:
                standard = root / "tuning" / label / f"trial-{trial:02d}" / f"seed-{seed}"
                folder = pick_fit_path(root, standard, settings, seed, arm)
                relative = folder.relative_to(root).as_posix()
                selection_path = folder / "source_selection.json"
                if selection_path.exists():
                    record = json.loads(selection_path.read_text(encoding="utf-8"))
                    if (record["settings"] != settings or record["seed"] != seed or record["arm"] != arm or
                            record["allocation_sha256"] != sha256(policy_path) or
                            record["run"]["checkpoint_sha256"] != sha256(folder / "final.pt")):
                        raise ValueError(f"Existing source-selection record mismatch: {folder}")
                    record["output_relative_path"] = relative
                else:
                    if not (folder / "complete.json").exists():
                        report = train_arm(settings, seed=seed, arm=arm, source=source, development=dev,
                            target=adaptation, output=folder, probabilities=policy["probabilities"])
                    else:
                        report = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
                    _, score = source_error_matrix(folder / "final.pt", dev)
                    record = {"arm": arm, "information_access": label, "trial": trial, "seed": seed,
                        "settings": settings, "source_score": score, "run": report,
                        "allocation_sha256": sha256(policy_path), "output_relative_path": relative}
                    atomic_json(selection_path, record)
                records.append(record)
                atomic_json(root / "tuning_resume_checkpoint.json", {"records": records,
                    "completed_fit_count": len(records), "planned_fit_count": 240,
                    "final_images_used": False})
                print(json.dumps({"completed": len(records), "planned": 240, "arm": label,
                    "trial": trial, "seed": seed,
                    "source_score_arcsec": record["source_score"]["shared_source_score_arcsec"]}), flush=True)
    with final_summary.open("x", encoding="utf-8") as stream:
        json.dump({"records": records, "initial_configs_per_arm": 12, "seeds_per_trial": 2,
            "status": "development_only_requires_boundary_and_convergence_review",
            "resume_checkpoint_sha256": sha256(root / "tuning_resume_checkpoint.json")},
            stream, indent=2)
        stream.write("\n")
    print("Source-only initial tuning complete; inspect boundaries before any method freeze.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml/robustness_development_v1.yaml")
    args = parser.parse_args()
    run(args.config)
