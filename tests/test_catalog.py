import pandas as pd
import pytest

from src.processing.build_catalog import build_catalog


def test_catalog_is_built_from_curated_source_without_averaging_ambiguous_redshifts(tmp_path):
    metadata = tmp_path / "metadata.csv"
    output = tmp_path / "generated_catalog.csv"
    pd.DataFrame({"lens_id": ["B0631", "B0850", "new-lens"]}).to_csv(metadata, index=False)

    result = build_catalog(metadata, "data/labels/source/lens_catalog_source.csv", output)

    assert output.exists()
    assert len(result) == 3
    b0631 = result.loc[result["lens_id"] == "B0631"].iloc[0]
    b0850 = result.loc[result["lens_id"] == "B0850"].iloc[0]
    assert pd.isna(b0631["lens_redshift"])
    assert b0631["lens_redshift_alternatives"] == "0.09;0.62"
    assert pd.isna(b0850["source_redshift"])
    assert b0850["source_redshift_alternatives"] == "1.14;3.93"
    assert result.loc[result["lens_id"] == "new-lens", "source_match"].item() is False
    assert result.loc[result["source_match"], "model_suitability"].notna().all()
    assert result.loc[result["lens_id"] == "SDSS1004"].empty


def test_catalog_retains_conservative_suitability_classifications(tmp_path):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"lens_id": ["B1359", "SDSS1004"]}).to_csv(metadata, index=False)

    result = build_catalog(metadata, "data/labels/source/lens_catalog_source.csv", tmp_path / "catalog.csv")

    assert set(result["model_suitability"]) == {"not_single_sie"}
    assert set(result["lens_environment"]) == {"multiple_deflector", "cluster_scale"}


def test_system_classifications_and_candidate_list_are_traceable():
    source = pd.read_csv("data/labels/source/lens_catalog_source.csv")
    candidates = pd.read_csv("data/labels/source/pilot_comparison_candidates.csv")

    assert len(source) == 69
    assert source["classification_evidence"].notna().all()
    assert source["model_suitability"].notna().all()
    assert set(candidates["lens_id"]).issubset(set(source["lens_id"]))
    assert set(candidates["candidate_status"]) == {"review"}


def test_catalog_rejects_duplicate_curated_source_keys(tmp_path):
    metadata = tmp_path / "metadata.csv"
    source = tmp_path / "source.csv"
    pd.DataFrame({"lens_id": ["B0218"]}).to_csv(metadata, index=False)
    source_frame = pd.read_csv("data/labels/source/lens_catalog_source.csv")
    pd.concat([source_frame, source_frame.iloc[[0]]], ignore_index=True).to_csv(source, index=False)

    with pytest.raises(ValueError, match="duplicate lens IDs"):
        build_catalog(metadata, source, tmp_path / "output.csv")
