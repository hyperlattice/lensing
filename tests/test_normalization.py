import numpy as np
import pytest

from src.processing.normalization import normalize_image


def test_masked_percentile_excludes_invalid_extreme_values():
    image = np.array([[1.0, 2.0], [3.0, 1000.0]])
    valid = np.array([[True, True], [True, False]])

    result = normalize_image(image, valid, method="masked_percentile", lower=0, upper=100)

    assert result.status == "pass"
    assert result.valid_pixels == 3
    assert result.values[1, 1] == 0.0
    assert np.isclose(result.values[0, 0], 0.0)
    assert np.isclose(result.values[1, 0], 1.0)
    assert result.invertible is False


def test_affine_normalization_round_trips_valid_values():
    image = np.array([[2.0, 4.0], [6.0, 8.0]])
    result = normalize_image(image, method="affine")

    restored = result.inverse(result.values)
    assert np.allclose(restored, image)
    assert result.invertible is True


def test_percentile_normalization_rejects_exact_inverse_after_clipping():
    result = normalize_image(np.array([[0.0, 1.0, 2.0, 100.0]]), lower=25, upper=75)

    with pytest.raises(ValueError, match="not invertible"):
        result.inverse()


def test_empty_valid_region_is_documented_as_unusable():
    result = normalize_image(np.ones((2, 2)), np.zeros((2, 2), dtype=bool))

    assert result.status == "normalization_not_constrained"
    assert result.valid_pixels == 0
