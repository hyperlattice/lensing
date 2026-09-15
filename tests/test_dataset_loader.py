import json

import numpy as np
import pandas as pd

from src.data.dataset import LensDataset
from src.processing.psf import gaussian_psf


def test_structured_loader_validates_products_and_separates_truth(tmp_path):
    raw = tmp_path / "raw.fits"
    raw.write_bytes(b"fixture")
    image = np.ones((256, 256), dtype=np.float32)
    science_path = tmp_path / "science.npy"
    processed_path = tmp_path / "processed.npy"
    np.save(science_path, image)
    np.save(processed_path, image)
    mask = np.zeros((256, 256), dtype=bool)
    mask[:2, :] = True
    mask_path = tmp_path / "mask.npy"
    coverage_path = tmp_path / "coverage.npy"
    background_path = tmp_path / "background.npy"
    np.save(mask_path, mask)
    np.save(coverage_path, mask)
    np.save(background_path, mask)
    variance_path = tmp_path / "variance.npy"
    np.save(variance_path, np.where(mask, 0.0, 0.25).astype(np.float32))
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps({"target_shape": [256, 256]}), encoding="utf-8")
    psf_path = tmp_path / "psf.npy"
    np.save(psf_path, gaussian_psf(size=5).values)

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "lens_id": ["lens-1"],
            "raw_path": [raw.as_posix()],
            "science_path": [science_path.as_posix()],
            "processed_path": [processed_path.as_posix()],
            "mask_path": [mask_path.as_posix()],
            "coverage_mask_path": [coverage_path.as_posix()],
            "background_mask_path": [background_path.as_posix()],
            "variance_path": [variance_path.as_posix()],
            "geometry_transform_path": [geometry_path.as_posix()],
            "psf_path": [psf_path.as_posix()],
            "pixel_scale": [0.05],
            "theta_E_arcsec": [1.0],
            "theta_E_target_available": [True],
            "mass_target_available": [False],
            "q": [0.8],
            "parent_system_id": ["parent-1"],
        }
    ).to_csv(manifest, index=False)

    dataset = LensDataset(manifest, strict_products=True)
    assert dataset.validate_all(require_products=True) == []
    sample = dataset.get_sample(0)

    assert sample["inputs"]["products"]["image"].shape == (256, 256)
    assert sample["inputs"]["products"]["variance"].dtype == np.float32
    assert sample["inputs"]["products"]["psf"].shape == (5, 5)
    assert sample["targets"]["theta_E_arcsec"] == 1.0
    assert sample["targets"]["availability"]["theta_E_target_available"] is True
    assert "theta_E_arcsec" not in sample["inputs"]["metadata"]
    assert "q" not in sample["inputs"]["metadata"]
    assert "parent_system_id" not in sample["inputs"]["metadata"]
