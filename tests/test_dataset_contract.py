import numpy as np
import pandas as pd

from src.processing.preprocess import center_crop_or_pad
from src.processing.create_splits import create_splits


def test_center_crop_or_pad_produces_canonical_shape_and_dtype():
    result = center_crop_or_pad(np.ones((128, 128), dtype=np.float64))
    assert result.shape == (256, 256)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_split_assignments_do_not_leak_lenses(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "lens_id": ["lens1", "lens1", "lens2", "lens3"],
        "processed_path": ["a", "b", "c", "d"],
    }).to_csv(manifest_path, index=False)
    output_dir = tmp_path / "splits"
    result = create_splits(manifest_path, output_dir=output_dir, seed=1)
    assert result.groupby("lens_id")["split"].nunique().max() == 1
    assert (output_dir / "train.csv").exists()
    assert (output_dir / "assignments.csv").exists()
    assert not (tmp_path / "data" / "splits" / "train.csv").exists()


def test_split_assignments_can_be_preserved(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "sample_id": ["a", "b", "c"],
        "lens_id": ["lens1", "lens2", "lens3"],
        "processed_path": ["a", "b", "c"],
    }).to_csv(manifest_path, index=False)
    existing_path = tmp_path / "existing.csv"
    pd.DataFrame({
        "lens_id": ["lens1", "lens2"],
        "split": ["test", "train"],
    }).to_csv(existing_path, index=False)

    result = create_splits(
        manifest_path,
        output_dir=tmp_path / "splits",
        seed=1,
        existing_assignments_path=existing_path,
    )
    assignments = result.set_index("lens_id")["split"].to_dict()
    assert assignments["lens1"] == "test"
    assert assignments["lens2"] == "train"
