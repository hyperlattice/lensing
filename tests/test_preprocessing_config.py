from pathlib import Path

import pandas as pd
import pytest
import numpy as np
from astropy.io import fits

from src.processing.preprocess import main
from src.processing.preprocessing_config import config_hash, load_config, validate_config


def test_preprocessing_config_loads_and_hash_is_stable():
    config = load_config("configs/preprocessing/default.yaml")
    assert validate_config(config) == []
    assert config_hash(config) == config_hash(config)


def test_invalid_preprocessing_config_is_rejected_before_processing(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "schema_version: '1.0'\npreprocessing_version: ''\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid preprocessing configuration"):
        load_config(config_path)


def test_preprocessing_dry_run_plans_outputs_without_writing_arrays(tmp_path):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "sample_id": ["sample-a", "sample-b"],
            "raw_path": ["data/raw/a.fits", "data/raw/b.fits"],
        }
    ).to_csv(metadata, index=False)
    output_dir = tmp_path / "products"
    manifest = tmp_path / "processed_metadata.csv"
    run_manifest = tmp_path / "run.json"

    summary = main(
        metadata_path=metadata,
        output_dir=output_dir,
        manifest_path=manifest,
        config_path="configs/preprocessing/default.yaml",
        dry_run=True,
        run_manifest_path=run_manifest,
    )

    assert summary["dry_run"] is True
    assert summary["planned_count"] == 2
    assert not output_dir.exists()
    assert not manifest.exists()
    assert run_manifest.exists()
    assert all(Path(item["normalized_path"]).name.endswith(".npy") for item in summary["planned_outputs"])


def test_preprocessing_writes_shared_geometry_and_mask_products(tmp_path):
    raw = tmp_path / "sample.fits"
    fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32)).writeto(raw)
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"sample_id": ["sample-a"], "raw_path": [raw.as_posix()]}).to_csv(metadata, index=False)
    products = tmp_path / "products"

    summary = main(
        metadata_path=metadata,
        output_dir=products,
        manifest_path=tmp_path / "manifest.csv",
        config_path="configs/preprocessing/default.yaml",
    )

    assert summary["accepted_count"] == 1
    assert (products / "normalized" / "sample-a.npy").exists()
    assert (products / "science" / "sample-a.npy").exists()
    assert (products / "masks" / "sample-a_invalid.npy").exists()
    assert (products / "masks" / "sample-a_coverage.npy").exists()
    assert (products / "geometry" / "sample-a.json").exists()
    assert (products / "background" / "sample-a.json").exists()
    assert (products / "transforms" / "sample-a.json").exists()
    assert (products / "variance" / "sample-a.npy").exists()
    assert (products / "noise" / "sample-a.json").exists()
    coverage = np.load(products / "masks" / "sample-a_coverage.npy")
    assert coverage[0, 0]
    assert not coverage[127, 127]
