"""Exports contain only canonical bytes and never replace existing files."""

import zipfile

import pytest

from scripts.submission import ALLOWED, BUNDLE, export


def test_export_matches_canonical_bundle_and_is_deterministic(tmp_path):
    first = export(tmp_path / "first.zip")
    second = export(tmp_path / "second.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert set(archive.namelist()) == ALLOWED
        for name in ALLOWED:
            assert archive.read(name) == (BUNDLE / name).read_bytes()


def test_export_refuses_existing_file_and_source_directory(tmp_path):
    path = tmp_path / "existing.zip"
    path.write_bytes(b"keep this")
    with pytest.raises(FileExistsError):
        export(path)
    assert path.read_bytes() == b"keep this"
    with pytest.raises(ValueError, match="outside"):
        export(BUNDLE / "unexpected.zip")
