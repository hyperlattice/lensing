"""Strict native-grid benchmark overlay, separate from frozen CSV releases.

An image-only view never opens a target sidecar. Roles and lineage are checked
over the entire supplied ledger before any role is selected. Numerical
qualification requires a separately reviewed report. This module does not
authorize final evaluation.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

ROLES = frozenset(
    {
        "source_train",
        "source_development",
        "source_calibration",
        "target_adaptation",
        "numerical_development",
        "power_pilot",
        "final",
    }
)


def contained_path(root: str | Path, relative: str) -> Path:
    """Reject absolute paths, traversal and symlinks escaping the release."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Artifact paths must be relative POSIX paths")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(p in {"..", "."} for p in relative.split("/")):
        raise ValueError("Artifact path escapes release")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact path escapes release")
    return path


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_lineage(
    rows: list[dict[str, Any]], excluded: list[dict[str, Any]] | tuple = ()
) -> None:
    """Reject reused ancestry across roles, including renamed parent systems.

    Excluded identities contain role, parent_system_id, parameter_seed,
    noise_seed and ancestry_ids only; old pixels/targets need not be opened.
    Ancestry IDs identify reusable assets, not an analytic profile family.
    """
    assigned, parent_identity = {}, {}
    for row in [*excluded, *rows]:
        role, parent = row["role"], row["parent_system_id"]
        if role not in ROLES or not isinstance(parent, str) or not parent:
            raise ValueError("Invalid lineage role or parent ID")
        if not all(_integer(row[k]) for k in ("parameter_seed", "noise_seed")):
            raise ValueError("Parameter/noise seeds must be nonnegative integers")
        assets = row["ancestry_ids"]
        if not isinstance(assets, list) or any(not isinstance(a, str) or not a for a in assets):
            raise ValueError("ancestry_ids must list nonempty asset identities")
        identity = (row["parameter_seed"], tuple(sorted(set(assets))))
        tokens = [
            ("parent", parent),
            ("parameter_seed", row["parameter_seed"]),
            ("noise_seed", row["noise_seed"]),
            *[("asset", a) for a in assets],
        ]
        for token in tokens:
            if token in assigned and assigned[token] != role:
                raise ValueError(f"Cross-role lineage overlap: {token}")
            assigned[token] = role
        if parent in parent_identity and parent_identity[parent] != identity:
            raise ValueError("Parent identity changed between observations")
        parent_identity[parent] = identity
