"""Run configuration, reproducibility records and write-once study records."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch and request deterministic Torch operations."""
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def write_json(path: str | Path, value: Any) -> None:
    """Write a sorted, indented run record after creating its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_provenance() -> dict[str, Any]:
    """Record checkout context without hashing or locking the source tree.

    A dirty checkout is explicitly marked; its revision alone cannot recreate
    uncommitted edits.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    return {"git_revision": revision, "working_tree_dirty": dirty}


def environment_record(device: Any) -> dict[str, Any]:
    """Record runtime and library versions for the selected Torch device."""
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_version": torch.version.cuda,
        "packages": {
            name: version(name) for name in ("torch", "numpy", "pandas", "matplotlib", "PyYAML")
        },
    }


def create_once(path: str | Path, value: Any) -> str:
    """Create a study record exclusively; refuse to replace existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return str(path)


def timestamp() -> str:
    """Return a timezone-aware UTC timestamp for a new run record."""
    return datetime.now(timezone.utc).isoformat()


def freeze_methods(
    path: str | Path, *, study_id: str, design: dict[str, Any], artifacts: list[str | Path]
) -> str:
    """Bind the declared secondary-study method and input bytes before generation."""
    required = {
        "training_roles",
        "arms",
        "target_information_access",
        "primary_endpoint",
        "secondary_contrasts",
        "alpha_allocation",
        "margins",
        "source_selection_rule",
        "training_budget",
        "noise_and_seed_policy",
        "numerical_qualification",
        "failure_and_recovery_policy",
        "power_planning_rule",
    }
    if set(design) != required or set(design["training_roles"]) != {
        "source_train",
        "source_development",
        "target_adaptation",
    }:
        raise ValueError("Incomplete method design or disallowed training roles")
    if set(design["arms"]) != set(design["target_information_access"]):
        raise ValueError("Every method needs an explicit target-information policy")
    if len(design["arms"]) != len(set(design["arms"])) or not study_id:
        raise ValueError("Invalid study or duplicate method identity")
    return create_once(
        path,
        {
            "schema": "robustness_method_freeze_v1",
            "study_id": study_id,
            "created_utc": timestamp(),
            "design": design,
            "artifacts": [str(Path(p)) for p in artifacts],
            "artifact_sha256": {str(Path(p)): sha256(p) for p in artifacts},
            "status": "methods_locked_before_power_pilot",
        },
    )


def verify_method_freeze(path: str | Path) -> dict[str, Any]:
    """Reject a changed input before using a frozen secondary-study method."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record["schema"] != "robustness_method_freeze_v1":
        raise ValueError("Not a method freeze")
    for artifact, expected in record["artifact_sha256"].items():
        if sha256(artifact) != expected:
            raise ValueError(f"Frozen method artifact changed: {artifact}")
    return record


def sha256(path: str | Path) -> str:
    """Return the SHA256 identity of a saved input or product."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


DEFAULT_CONFIG = Path("configs/study.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read settings only; never inspect data banks, checkpoints or holdout."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "output",
        "source_train",
        "source_development",
        "policy",
        "holdout",
        "seeds",
        "arms",
        "conditions",
        "bootstrap",
        "training",
    }
    if not isinstance(config, dict) or set(config) != required or config["schema_version"] != 1:
        raise ValueError("Unexpected equal-budget configuration schema")
    if config["arms"] != ["C", "G", "R"] or config["conditions"] != [
        "canonical",
        "pixel-scale-004",
        "pixel-scale-006",
        "exposure-half",
        "read-noise-double",
    ]:
        raise ValueError("This runner requires the fixed C/G/R observation matrix")
    seeds = config["seeds"]
    if not seeds or any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Expected unique integer seeds")
    controls = config["bootstrap"]
    if set(controls) != {"resamples", "stress_seed", "canonical_seed"} or any(
        type(value) is not int or value < 1 for value in controls.values()
    ):
        raise ValueError("Expected positive integer bootstrap controls")
    return config


def verify_resolved_config(config: dict[str, Any]) -> None:
    """Require downstream stages to use the configuration sealed at fit time."""
    path = Path(config["output"]) / "resolved_config.json"
    resolved = json.loads(path.read_text(encoding="utf-8"))
    if resolved.get("config", resolved) != config:
        raise ValueError("Equal-budget configuration changed; use a new output identity")
