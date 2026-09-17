"""Small, explicit helpers for repository-owned artifact locations.

The data releases remain self-contained because their manifests reference
products relative to each release. Review images are intentionally external:
they are derived visual artifacts, grouped by the domain they depict.
"""

from __future__ import annotations

import re
from pathlib import Path


DEFAULT_PREVIEW_ROOT = Path("results/previews")
PREVIEW_DOMAINS = frozenset({"actual", "simulated"})
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def preview_directory(domain: str, release_id: str, *, root=DEFAULT_PREVIEW_ROOT) -> Path:
    """Return a safe domain-separated directory for generated preview images."""
    if domain not in PREVIEW_DOMAINS:
        raise ValueError(f"Preview domain must be one of {sorted(PREVIEW_DOMAINS)}")
    release_id = str(release_id)
    if not _SAFE_IDENTIFIER.fullmatch(release_id):
        raise ValueError("release_id must contain only letters, numbers, dots, underscores, or hyphens")
    return Path(root) / domain / release_id
