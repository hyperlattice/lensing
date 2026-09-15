import json

import numpy as np
import pytest

from src.processing.psf import convolve_same, gaussian_psf, save_psf, validate_psf


def test_gaussian_control_has_explicit_center_sampling_and_unit_sum(tmp_path):
    kernel = gaussian_psf("control", sigma_pixels=1.2, size=7, pixel_scale=0.05)

    assert validate_psf(kernel) == []
    assert kernel.values.shape == (7, 7)
    assert kernel.center_yx == (3.0, 3.0)
    assert np.isclose(kernel.values.sum(), 1.0)
    values_path, metadata_path = save_psf(kernel, tmp_path)
    metadata = json.loads(metadata_path.read_text())
    assert values_path.exists()
    assert metadata["psf_id"] == "control"
    assert metadata["center_yx_pixels"] == [3.0, 3.0]


def test_point_source_convolution_preserves_centroid_and_near_unit_flux():
    kernel = gaussian_psf(size=9, sigma_pixels=1.0)
    image = np.zeros((41, 41), dtype=float)
    image[20, 20] = 1.0

    convolved = convolve_same(image, kernel)
    y, x = np.indices(convolved.shape)
    assert np.isclose((convolved * x).sum() / convolved.sum(), 20.0)
    assert np.isclose((convolved * y).sum() / convolved.sum(), 20.0)
    assert np.isclose(convolved.sum(), 1.0)


def test_psf_validation_rejects_non_unit_kernel():
    kernel = gaussian_psf()
    invalid = type(kernel)(kernel.psf_id, kernel.values * 2, kernel.pixel_scale, kernel.source)
    assert any("unit sum" in error for error in validate_psf(invalid))


def test_gaussian_psf_requires_odd_size():
    with pytest.raises(ValueError, match="odd"):
        gaussian_psf(size=8)
