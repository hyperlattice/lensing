import numpy as np
import pandas as pd

from src.processing.preprocess import center_crop_or_pad


def test_center_crop_or_pad_produces_canonical_shape_and_dtype():
    result = center_crop_or_pad(np.ones((128, 128), dtype=np.float64))
    assert result.shape == (256, 256)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_split_assignments_do_not_leak_lenses(tmp_path):
    from src.processing.create_splits import create_splits

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "lens_id": ["lens1", "lens1", "lens2", "lens3"],
        "processed_path": ["a", "b", "c", "d"],
    }).to_csv(manifest_path, index=False)
    result = create_splits(manifest_path, seed=1)
    assert result.groupby("lens_id")["split"].nunique().max() == 1