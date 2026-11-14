"""Protect native geometry, source identity, and development-only HST intake."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.hst_pilot import (
    acquire_assets,
    git_blob_sha1,
    inspect_pair,
    load_manifest,
    target_convention_control,
)

h5py = pytest.importorskip("h5py", reason="HST intake requires the hst extra")


@pytest.fixture
def native_pair(tmp_path):
    image_path = tmp_path / "image.h5"
    psf_path = tmp_path / "psf.h5"
    image = np.arange(48, dtype=np.float32).reshape(6, 8) - 5
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    with h5py.File(image_path, "w") as handle:
        handle["image_data"] = image
        handle["exposure_time"] = np.full(image.shape, 100.0)
        handle["background_rms"] = 0.01
        handle["transform_pix2angle"] = rotation @ np.diag([0.05, 0.08])
    with h5py.File(psf_path, "w") as handle:
        handle["kernel_point_source"] = np.ones((3, 3)) * 2
    return {"image": image_path, "psf": psf_path}, image


def test_native_values_and_rotated_angular_geometry_are_preserved(native_pair):
    paths, image = native_pair
    report, arrays = inspect_pair(paths, {"shape": [256, 256], "pixel_scale_arcsec": 0.05})
    np.testing.assert_array_equal(arrays["image_data"], image)
    assert report["shape_yx"] == [6, 8]
    np.testing.assert_allclose(report["field_edge_lengths_xy_arcsec"], [0.4, 0.48])
    assert report["pixel_scale_equal_area_arcsec"] == pytest.approx(np.sqrt(0.05 * 0.08))
    assert report["pixel_anisotropy_ratio"] == pytest.approx(1.6)
    assert report["psf_sum_original"] == 18
    assert arrays["psf_normalized"].sum() == pytest.approx(1)
    assert not report["canonical_edge_support"]
    assert not report["model_input_ready"]
    assert not report["unit_calibration_verified"]
    assert "native_cutout_edges_smaller_than_canonical_field" in report["blockers"]
    assert "reference" not in arrays


@pytest.mark.parametrize("fault", ["nan_image", "singular_transform", "bad_exposure", "zero_psf"])
def test_invalid_products_fail_explicitly(native_pair, fault):
    paths, _ = native_pair
    if fault == "zero_psf":
        with h5py.File(paths["psf"], "r+") as handle:
            handle["kernel_point_source"][:] = 0
    else:
        with h5py.File(paths["image"], "r+") as handle:
            if fault == "nan_image":
                handle["image_data"][0, 0] = np.nan
            elif fault == "singular_transform":
                handle["transform_pix2angle"][:] = 0
            else:
                handle["exposure_time"][0, 0] = 0
    with pytest.raises(ValueError):
        inspect_pair(paths, {"shape": [256, 256], "pixel_scale_arcsec": 0.05})


def test_unreviewed_mask_is_preserved_without_assigning_semantics(native_pair, tmp_path):
    paths, image = native_pair
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[0, 0] = 1
    paths["mask"] = tmp_path / "mask.h5"
    with h5py.File(paths["mask"], "w") as handle:
        handle["mask"] = mask
    report, arrays = inspect_pair(paths, {"shape": [256, 256], "pixel_scale_arcsec": 0.05})
    np.testing.assert_array_equal(arrays["upstream_mask_uninterpreted"], mask)
    assert report["mask"]["nonzero_pixels"] == 1
    assert "upstream_mask_semantics_not_reviewed" in report["blockers"]


def test_manifest_rejects_duplicate_systems_and_confirmation_role(tmp_path):
    document = load_manifest(Path("data/manifests/hst_compatibility_pilot_v1.json"))
    path = tmp_path / "manifest.json"
    changed = copy.deepcopy(document)
    changed["systems"].append(changed["systems"][0])
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(path)
    document["role"] = "final"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="development"):
        load_manifest(path)


def test_offline_cache_tampering_is_detected(tmp_path):
    document = load_manifest(Path("data/manifests/hst_compatibility_pilot_v1.json"))
    first = document["systems"][0]
    asset = first["assets"]["image"]
    folder = tmp_path / first["system_id"]
    folder.mkdir()
    (folder / Path(asset["repository_path"]).name).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Git verification"):
        acquire_assets(document, tmp_path, offline=True)


def test_git_blob_hash_includes_header():
    assert git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_sie_epl_control_does_not_certify_observed_reference_equivalence():
    pytest.importorskip("lenstronomy")
    result = target_convention_control()
    assert result["passed"]
    assert result["evaluated_parameter_pairs"] == 9
    assert not result["observed_reference_equivalence_established"]
