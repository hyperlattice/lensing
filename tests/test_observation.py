"""Detector units and backend conventions are independent of model outcomes."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.simulation.backends import deflections_xy, mapped_light_parameters, mass_profile
from src.simulation.independent import independent_sie_deflection
from src.simulation.observation import psf_kernel, sample_detector


def test_detector_units_background_and_same_seed_are_explicit():
    expected = np.full((128, 128), 0.3)
    first, variance = sample_detector(expected, 100.0, 2.0, 0.03, 19)
    second, _ = sample_detector(expected, 100.0, 2.0, 0.03, 19)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(variance, ((0.3 + 0.03) * 100 + 2**2) / 100**2)
    assert abs(first.mean() - 0.33) < 5 * np.sqrt(variance[0, 0] / first.size)
    with pytest.raises(ValueError, match="Invalid detector"):
        sample_detector(expected, 0.0, 2.0, 0.03, 19)


def test_bad_light_parameters_fail_before_optional_backend_import():
    for intensity, radius, index in ((-1, 0.2, 1), (1, 0, 1), (1, 0.2, np.nan)):
        with pytest.raises(ValueError, match="Light parameters"):
            mapped_light_parameters(intensity, radius, index)


@pytest.mark.parametrize("family", ["gaussian", "moffat"])
def test_psf_is_finite_nonnegative_and_normalized(family):
    kernel = psf_kernel(0.1, 0.05, family)
    assert kernel.ndim == 2
    assert np.isfinite(kernel).all()
    assert (kernel >= 0).all()
    np.testing.assert_allclose(kernel.sum(), 1.0)


@pytest.mark.skipif(
    importlib.util.find_spec("autolens") is None,
    reason="optional AutoLens physical backend is unavailable",
)
def test_autolens_deflections_match_independent_sie_control():
    record = json.loads((Path(__file__).parent / "fixtures/render_reference.json").read_text())
    parameters = record["parameters"]
    coordinates = np.array([[0.7, 0.4], [-1.2, 0.2], [0.2, -0.9]])
    actual = deflections_xy(mass_profile(parameters), coordinates)
    horizontal, vertical = independent_sie_deflection(
        coordinates[:, 0],
        coordinates[:, 1],
        theta_E_arcsec=parameters["theta_E_arcsec"],
        axis_ratio=parameters["q"],
        phi_rad=parameters["phi_rad"],
        center_x_arcsec=parameters["lens_center_x_arcsec"],
        center_y_arcsec=parameters["lens_center_y_arcsec"],
    )
    np.testing.assert_allclose(actual, np.column_stack((horizontal, vertical)), atol=1e-7)
