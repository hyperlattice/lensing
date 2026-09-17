from pathlib import Path

import pytest

from src.layout import DEFAULT_PREVIEW_ROOT, preview_directory


def test_preview_directory_separates_actual_and_simulated_releases():
    assert preview_directory("actual", "real-castles-v2") == (
        DEFAULT_PREVIEW_ROOT / "actual" / "real-castles-v2"
    )
    assert preview_directory("simulated", "sim-p60-v1", root=Path("scratch/previews")) == (
        Path("scratch/previews/simulated/sim-p60-v1")
    )


@pytest.mark.parametrize("domain, release_id", [
    ("real", "real-castles-v2"),
    ("actual", "../escape"),
    ("simulated", "nested/release"),
    ("simulated", ""),
])
def test_preview_directory_rejects_ambiguous_locations(domain, release_id):
    with pytest.raises(ValueError):
        preview_directory(domain, release_id)
