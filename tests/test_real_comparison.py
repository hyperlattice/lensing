import numpy as np
import pytest

from src.modeling.real_comparison import measure_peak_radius


def test_measure_peak_radius_recovers_four_point_ring():
    science = np.zeros((65, 65), dtype=float)
    for x, y in ((32, 20), (44, 32), (32, 44), (20, 32)):
        science[y, x] = 10.0
    result = measure_peak_radius(
        science,
        image_count=4,
        pixel_scale_arcsec=0.05,
        gaussian_sigma_pixels=1.0,
        minimum_peak_separation_pixels=7,
    )
    assert result["center_x_pixel"] == pytest.approx(32.0)
    assert result["center_y_pixel"] == pytest.approx(32.0)
    assert result["radius_arcsec"] == pytest.approx(0.6)


def test_measure_peak_radius_rejects_non_image():
    with pytest.raises(ValueError, match="two-dimensional"):
        measure_peak_radius(np.zeros(4), image_count=2, pixel_scale_arcsec=0.05)
