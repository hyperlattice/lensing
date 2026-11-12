
import numpy as np
import pytest

from src.data.external_reference import readiness_blockers, catalog_rows




def test_missing_individual_error_blocks_scoring_even_after_other_reviews():
    sample = dict.fromkeys(["center_reviewed","calibration_reviewed","real_image_provenance_reviewed",
                           "reference_quality_reviewed","target_mapping_reviewed"],True)
    sample["reference_sigma_arcsec"] = None
    assert readiness_blockers(sample,[],"electrons/s") == ["individual_reference_uncertainty_required"]
    sample["reference_sigma_arcsec"] = 0.02
    assert readiness_blockers(sample,[],"electrons/s") == []


def test_catalog_radius_is_not_stellar_radius_or_uncertainty(tmp_path):
    path=tmp_path/"catalog.cat"
    path.write_text("# name Reff_arcsec theta_Ein lMstar_err\nLENS 2.42 1.52 0.12\n")
    row=catalog_rows(path).loc["LENS"]
    assert row["theta_Ein"] == 1.52
    assert "reference_sigma_arcsec" not in row


def test_real_intake_preserves_supplied_noise_and_does_not_create_simulator_truth(tmp_path):
    import yaml
    from astropy.io import fits
    from src.data.external_reference import prepare

    raw = tmp_path/"raw"
    folder = raw/"pilot"
    folder.mkdir(parents=True)
    catalog = raw/"catalog.cat"
    catalog.write_text("# name theta_Ein\nLENS 1.52\n")
    assets = {}
    for kind, array in {"science":np.zeros((9,9)),"noise":np.full((9,9),2.0),
                        "psf":np.ones((3,3))/9}.items():
        path=folder/f"{kind}.fits"
        fits.writeto(path,array)
        assets[kind]={"filename":path.name}
    config={"experiment_id":"test","raw_root":str(raw),"output":str(tmp_path/"out"),
            "report":str(tmp_path/"report.json"),
            "reference_catalog":{"filename":"catalog.cat","citation":"test"},
            "samples":[{"sample_id":"pilot","lens_id":"LENS","assets":assets,"science_hdu":0,
                        "noise_semantics":"standard_deviation","pixel_scale_arcsec":0.05,
                        "reference_definition":"unmapped","image_unit":"unknown"}]}
    path=tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    report=prepare(path)
    assert report["accuracy_ready_count"] == 0
    assert "theta_E_arcsec" not in report["samples"][0]
    assert np.all(np.load(tmp_path/"out/products/pilot/variance.npy") == 4)
