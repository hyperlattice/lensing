import numpy as np
from astropy.io.fits import Header

from src.processing.build_metadata import calculate_pixel_scale


def test_pixel_scale_uses_both_wcs_axes():
    header = Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CD1_1"] = -0.00001
    header["CD1_2"] = 0.000002
    header["CD2_1"] = 0.000003
    header["CD2_2"] = 0.000012

    expected = np.mean([
        np.hypot(-0.00001, 0.000003),
        np.hypot(0.000002, 0.000012),
    ]) * 3600
    assert np.isclose(calculate_pixel_scale(header), expected)


def test_pixel_scale_falls_back_to_cdelt_in_degrees():
    header = Header()
    header["CDELT1"] = -0.000011
    header["CDELT2"] = 0.000011
    assert np.isclose(calculate_pixel_scale(header), 0.000011 * 3600)