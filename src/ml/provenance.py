"""Small, local provenance records for reproducible ML experiments."""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch




def source_provenance():
    """Record checkout context without hashing or locking the source tree.

    A dirty checkout is explicitly marked; its revision alone cannot recreate
    uncommitted edits.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root,
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    return {"git_revision": revision, "working_tree_dirty": dirty}


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
