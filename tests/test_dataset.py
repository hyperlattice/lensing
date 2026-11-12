"""Native product and image-only contract tests without a truth sidecar."""

import json

import numpy as np
import pytest
from conftest import paired_release as fixture

from src.data.dataset import PairedEvaluationImages


def test_image_only_loader_never_needs_a_truth_file(tmp_path):
    path, _ = fixture(tmp_path)
    images = PairedEvaluationImages(path, role="final")
    images.verify_all_products()
    assert images.image("p0", "autolens", "pixel-scale-004").shape == (320, 320)
    assert not (tmp_path / "truth_vault").exists()


def test_saved_evaluation_images_do_not_require_renderer_source(tmp_path):
    path, manifest = fixture(tmp_path)
    qualification = tmp_path / "qualification.json"
    report = json.loads(qualification.read_text())
    qualification.write_text(json.dumps(report))
    path.write_text(json.dumps(manifest))
    images = PairedEvaluationImages(path, role="final")
    np.testing.assert_array_equal(
        images.image("p0", "autolens", "canonical"), np.full((256, 256), 0.1, np.float32)
    )


def test_truth_injection_and_missing_view_are_rejected(tmp_path):
    path, manifest = fixture(tmp_path)
    manifest["observations"][0]["theta_E_arcsec"] = 1.0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="truth"):
        PairedEvaluationImages(path, role="final")
    del manifest["observations"][0]["theta_E_arcsec"]
    manifest["observations"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Missing paired"):
        PairedEvaluationImages(path, role="final")


def test_native_image_reads_current_array(tmp_path):
    path, _ = fixture(tmp_path)
    images = PairedEvaluationImages(path, role="final")
    np.save(tmp_path / "image-256.npy", np.zeros((256, 256), np.float32))
    np.testing.assert_array_equal(
        images.image("p0", "autolens", "canonical"), np.zeros((256, 256), np.float32)
    )
