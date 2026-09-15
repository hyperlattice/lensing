import numpy as np
import pytest

from src.processing.noise import estimate_scalar_variance, variance_from_error


def test_supplied_error_is_squared_into_positive_variance():
    error = np.array([[2.0, 3.0], [4.0, -1.0]])
    valid = np.array([[True, True], [True, True]])

    result = variance_from_error(error, valid, unit="electrons")

    assert result.method == "supplied_error_squared"
    assert result.status == "pass"
    assert result.values.tolist() == [[4.0, 9.0], [16.0, 0.0]]
    assert result.unit == "(electrons)^2"


def test_toy_variance_is_positive_on_valid_pixels_and_zero_on_exclusions():
    image = np.ones((2, 2), dtype=float)
    valid = np.array([[True, False], [True, True]])

    result = estimate_scalar_variance(image, valid, background_scale=0.0, noise_floor=0.1)

    assert result.status == "toy_unprofiled"
    assert result.values[0, 0] > 0
    assert result.values[0, 1] == 0
    assert result.valid_pixels == 3


def test_variance_rejects_invalid_noise_floor():
    with pytest.raises(ValueError, match="noise_floor"):
        estimate_scalar_variance(np.ones((2, 2)), np.ones((2, 2), dtype=bool), 1.0, noise_floor=0)
