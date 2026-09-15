import numpy as np

from src.processing.background import estimate_scalar_background


def test_background_estimator_recovers_scalar_and_rejects_source_outlier():
    image = np.full((32, 32), 5.0, dtype=float)
    image[15, 15] = 100.0
    exclusion = np.zeros_like(image, dtype=bool)
    exclusion[:4, :] = True

    estimate = estimate_scalar_background(image, exclusion, min_valid_pixels=64)

    assert estimate.status == "pass"
    assert np.isclose(estimate.value, 5.0)
    assert estimate.source_pixels >= 1
    assert estimate.exclusion_mask[15, 15]
    assert estimate.exclusion_mask[0, 0]


def test_background_estimator_reports_unconstrained_region():
    image = np.ones((8, 8), dtype=float)
    estimate = estimate_scalar_background(image, min_valid_pixels=65)

    assert estimate.status == "background_not_constrained"
    assert np.isnan(estimate.value)
    assert estimate.valid_pixels == 64


def test_background_estimator_rejects_non_two_dimensional_inputs():
    try:
        estimate_scalar_background(np.ones((2, 2, 1)))
    except ValueError as error:
        assert "two-dimensional" in str(error)
    else:
        raise AssertionError("Expected a two-dimensionality validation error")
