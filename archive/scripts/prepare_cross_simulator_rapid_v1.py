"""Freeze the rapid cross-simulator generation identity before rendering."""
from __future__ import annotations

import json
from pathlib import Path

from src.ml.study_freeze_v1 import freeze_methods


ROOT = Path("results/ml/cross-simulator-rapid-v2")
METHOD = ROOT / "method_freeze.json"
DESIGN = ROOT / "generation_design.json"
SIM_CONFIG = Path("configs/simulation_engineering_v1.yaml")
POLICY = Path("results/ml/robustness-development-v4/allocation/policy.json")
SOURCE = Path("data/simulations/robustness-study-source-train-v4/bank.json")
DEVELOPMENT = Path("data/simulations/robustness-study-source-development-v4/bank.json")
TARGET = Path("data/simulations/robustness-study-target-adaptation-v4/bank.json")


def main():
    if ROOT.exists() and any(ROOT.iterdir()):
        raise FileExistsError("Rapid cross-simulator identity already exists")
    ROOT.mkdir(parents=True, exist_ok=True)
    design = {
        "role": "power_pilot",
        "release_directory": "data/simulations/cross-simulator-rapid-v2",
        "parent_count": 150,
        "generation_seed": 62031,
        "method_freeze": str(METHOD),
        "evaluation_freeze": None,
        "exclusion_banks": [str(SOURCE), str(DEVELOPMENT), str(TARGET)],
        "simulation_config": str(SIM_CONFIG),
        "noise_namespace": 6203101,
    }
    DESIGN.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    method_design = {
        "training_roles": ["source_train", "source_development", "target_adaptation"],
        "arms": ["C", "G", "R"],
        "target_information_access": {"C": "none", "G": "none", "R": "none"},
        "primary_endpoint": {"renderer": "autolens", "conditions": ["moffat-2.5", "moffat-3.5", "moffat-4.5"]},
        "secondary_contrasts": ["G-C", "R-G", "R-C", "renderer-A-vs-B"],
        "alpha_allocation": {"primary_one_sided": 0.025, "secondary": "descriptive"},
        "margins": {"meaningful_arcsec": 0.005, "canonical_allowance_arcsec": 0.005},
        "source_selection_rule": "item-1-frozen-checkpoints-no-retuning",
        "training_budget": {"optimizer_updates": 880, "batch_size": 16, "source_exposures": 14080},
        "noise_and_seed_policy": {"fit_seed_ids": list(range(730001, 730011)), "prediction_seed_policy": "fixed-checkpoints"},
        "numerical_qualification": "renderer-qualification-v5-plus-cross-simulator-controls-v2",
        "failure_and_recovery_policy": "preserve-all-proposals-and-failures-use-new-identity-for-corrections",
        "power_planning_rule": "descriptive-rapid-extension; no outcome-driven enlargement",
    }
    artifacts = [
        DESIGN, SIM_CONFIG, POLICY, SOURCE, DEVELOPMENT, TARGET,
        Path("scripts/generate_paired_evaluation_v1.py"),
        Path("scripts/run_equal_budget_final_v1.py"),
        Path("src/simulation/low_memory_renderer_v2.py"),
        Path("src/simulation/autolens_backend_v2.py"),
        Path("src/simulation/priors.py"),
        Path("results/reports/renderer-qualification-v5/qualification.json"),
        Path("results/reports/cross-simulator-controls-v2/controls.json"),
    ]
    freeze_methods(METHOD, study_id="cross-simulator-rapid-v2", design=method_design, artifacts=artifacts)
    print(json.dumps({"method_freeze": str(METHOD), "generation_design": str(DESIGN),
                      "parent_count": 150, "renderers": ["lenstronomy", "autolens"]}, indent=2))


if __name__ == "__main__":
    main()
