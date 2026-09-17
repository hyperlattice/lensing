"""Small, local provenance records for reproducible ML experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch

from src.ml.data import file_sha256


def config_sha256(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def source_provenance():
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    # Content hashes capture uncommitted source edits as well as the Git revision.
    hashes = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted((root / "src").rglob("*.py"))
    }
    return {"git_revision": revision, "source_sha256": hashes}


def environment_record(device):
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_version": torch.version.cuda,
        "packages": {name: version(name) for name in ("torch", "numpy", "pandas", "matplotlib", "PyYAML")},
    }
