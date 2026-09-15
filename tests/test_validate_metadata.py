import numpy as np
import pandas as pd

from src.processing.validate_metadata import validate_metadata


def _write_valid_fixture(tmp_path):
    raw = tmp_path / "raw.fits"
    raw.write_bytes(b"fixture")
    processed = tmp_path / "image.npy"
    np.save(processed, np.zeros((256, 256), dtype=np.float32))
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "lens_id": ["lens-1"],
            "raw_path": [raw.as_posix()],
            "processed_path": [processed.as_posix()],
            "pixel_scale": [0.05],
            "lens_redshift": [0.5],
            "source_redshift": [1.5],
            "instrument": ["ACS"],
            "image_unit": ["unknown"],
        }
    ).to_csv(metadata, index=False)
    return metadata


def test_metadata_report_distinguishes_reviewable_unknown_units(tmp_path):
    metadata = _write_valid_fixture(tmp_path)
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "lens_id": ["lens-1"],
            "redshift_status": ["complete"],
            "model_suitability": ["review"],
        }
    ).to_csv(catalog, index=False)

    report, summary = validate_metadata(metadata, catalog, tmp_path / "report")

    assert report.loc[0, "status"] == "review"
    assert "unknown_unit" in report.loc[0, "issue_classes"]
    assert "preprocessing" in report.loc[0, "eligible_for"]
    assert summary["missingness_counts"]["detector"] == 1
    assert (tmp_path / "report" / "metadata_validation_summary.json").exists()


def test_metadata_report_marks_duplicate_catalog_keys_as_fatal(tmp_path):
    metadata = _write_valid_fixture(tmp_path)
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "lens_id": ["lens-1", "lens-1"],
            "redshift_status": ["complete", "complete"],
            "model_suitability": ["review", "review"],
        }
    ).to_csv(catalog, index=False)

    report, summary = validate_metadata(metadata, catalog, tmp_path / "report")

    assert "duplicate_catalog_key" in report.loc[0, "issue_classes"]
    assert report.loc[0, "fatal_classes"] == "duplicate_catalog_key"
    assert summary["fatal_rows"] == 1


def test_metadata_stage_without_processed_path_is_reported(tmp_path):
    metadata = tmp_path / "metadata.csv"
    raw = tmp_path / "raw.fits"
    raw.write_bytes(b"fixture")
    pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "lens_id": ["lens-1"],
            "raw_path": [raw.as_posix()],
            "pixel_scale": [0.05],
        }
    ).to_csv(metadata, index=False)

    report, summary = validate_metadata(metadata, None, tmp_path / "report")

    assert "missing_processed_path" in report.loc[0, "issue_classes"]
    assert summary["missingness_counts"]["image_unit"] == 1


def test_empty_metadata_report_is_still_machine_readable(tmp_path):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(columns=["sample_id", "raw_path", "processed_path", "pixel_scale"]).to_csv(metadata, index=False)

    report, summary = validate_metadata(metadata, None, tmp_path / "report")

    assert list(report.columns) == [
        "sample_id", "lens_id", "instrument", "status",
        "issue_classes", "issues", "fatal_classes", "eligible_for",
    ]
    assert summary["rows"] == 0
