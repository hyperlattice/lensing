import pandas as pd
import pytest

from src.processing.enrich_metadata import enrich_metadata


def test_enrichment_supports_canonical_lens_id_without_legacy_lens(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    catalog_path = tmp_path / "catalog.csv"
    output_path = tmp_path / "enriched.csv"
    pd.DataFrame({"sample_id": ["sample-1"], "lens_id": ["L1"]}).to_csv(metadata_path, index=False)
    pd.DataFrame({"lens_id": ["L1"], "lens_redshift": [0.5], "source_redshift": [1.5]}).to_csv(
        catalog_path, index=False
    )

    result = enrich_metadata(metadata_path, catalog_path, output_path)

    assert result.loc[0, "lens_id"] == "L1"
    assert bool(result.loc[0, "catalog_match"])
    assert output_path.exists()


def test_enrichment_rejects_duplicate_catalog_lens_ids(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame({"sample_id": ["sample-1"], "lens": ["L1"]}).to_csv(metadata_path, index=False)
    pd.DataFrame({"lens": ["L1", "L1"], "lens_redshift": [0.5, 0.6]}).to_csv(catalog_path, index=False)

    with pytest.raises(ValueError, match="duplicate lens IDs"):
        enrich_metadata(metadata_path, catalog_path, tmp_path / "enriched.csv")
