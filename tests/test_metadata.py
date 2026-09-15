import numpy as np
from astropy.io import fits
from astropy.io.fits import Header

from src.processing.build_metadata import build_metadata, calculate_pixel_scale, pixel_scale_axes


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


def test_pixel_scale_does_not_infer_scale_from_empty_wcs():
    header = Header()
    assert calculate_pixel_scale(header) is None


def test_pixel_scale_accepts_explicit_arcseconds_header_value():
    header = Header()
    header["PIXSCALE"] = 0.05
    assert np.isclose(calculate_pixel_scale(header), 0.05)


def test_pixel_scale_axes_preserves_anisotropy_and_source():
    header = Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CD1_1"] = -0.00001
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.00002

    x_scale, y_scale, source = pixel_scale_axes(header)
    assert np.isclose(x_scale, 0.036)
    assert np.isclose(y_scale, 0.072)
    assert source == "celestial_cd_matrix"


def test_metadata_selects_named_science_hdu_and_instrument_filter(tmp_path):
    root = tmp_path / "raw"
    image_dir = root / "QTEST" / "FIT" / "AWFCI"
    image_dir.mkdir(parents=True)
    fits_path = image_dir / "sample.fits"

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "ACS"
    primary.header["TARGNAME"] = "QTEST"
    primary.header["PROPOSID"] = "1234"
    science = fits.ImageHDU(data=np.ones((4, 4), dtype=np.float32), name="SCI")
    science.header["FILTNAM1"] = "F814W"
    science.header["EXPTIME"] = 12.5
    science.header["CTYPE1"] = "RA---TAN"
    science.header["CTYPE2"] = "DEC--TAN"
    science.header["CD1_1"] = -0.00001
    science.header["CD1_2"] = 0.0
    science.header["CD2_1"] = 0.0
    science.header["CD2_2"] = 0.00001
    fits.HDUList([primary, science]).writeto(fits_path)

    result = build_metadata(root, tmp_path / "metadata.csv")
    row = result.iloc[0]
    assert row["science_hdu"] == 1
    assert row["filter_name"] == "F814W"
    assert row["target_id"] == "QTEST"
    assert row["pixel_scale_source"] == "celestial_cd_matrix"
