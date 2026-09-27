import numpy as np
import pytest

from src.modeling.pixel_scale_geometric_baseline import (
    detect_point_image_peaks,
    fit_point_image_geometry,
)


def test_sis_point_geometry_recovers_known_two_image_radius():
    beta = np.array([0.2, 0.1])
    direction = beta / np.linalg.norm(beta)
    theta_e = 1.1
    peaks = [
        {
            "x_arcsec": float(beta[0] + theta_e * direction[0]),
            "y_arcsec": float(beta[1] + theta_e * direction[1]),
            "snr": 10.0,
        },
        {
            "x_arcsec": float(beta[0] - theta_e * direction[0]),
            "y_arcsec": float(beta[1] - theta_e * direction[1]),
            "snr": 8.0,
        },
    ]

    result = fit_point_image_geometry(
        peaks,
        center_x_arcsec=0.0,
        center_y_arcsec=0.0,
        initial_theta_e_arcsec=1.25,
    )

    assert result["success"]
    assert result["model"] == "SIS"
    assert result["theta_E_arcsec"] == pytest.approx(theta_e, abs=1e-10)
    assert result["beta_x_arcsec"] == pytest.approx(beta[0], abs=1e-10)
    assert result["beta_y_arcsec"] == pytest.approx(beta[1], abs=1e-10)


def test_point_geometry_reports_insufficient_detections():
    result = fit_point_image_geometry(
        [{"x_arcsec": 1.0, "y_arcsec": 0.0, "snr": 8.0}],
        center_x_arcsec=0.0,
        center_y_arcsec=0.0,
        initial_theta_e_arcsec=1.25,
    )

    assert not result["success"]
    assert result["theta_E_arcsec"] is None


def test_matched_filter_detects_two_synthetic_point_images_without_truth():
    shape = (64, 64)
    pixel_scale = 0.05
    yy, xx = np.indices(shape, dtype=np.float64)
    center = (np.asarray(shape) - 1.0) / 2.0
    psf_size = 9
    psf_y, psf_x = np.indices((psf_size, psf_size), dtype=np.float64)
    psf_center = (psf_size - 1.0) / 2.0
    psf = np.exp(-((psf_x - psf_center) ** 2 + (psf_y - psf_center) ** 2) / 2.0)
    psf /= psf.sum()
    image = 20.0 * np.exp(-((xx - center[1]) ** 2 + (yy - center[0]) ** 2) / (2.0 * 4.0**2))
    for x_arcsec in (-1.1, 1.1):
        x_pixel = int(round(center[1] + x_arcsec / pixel_scale))
        y_pixel = int(round(center[0]))
        y0, x0 = y_pixel - psf_center, x_pixel - psf_center
        image[
            int(y0) : int(y0 + psf_size), int(x0) : int(x0 + psf_size)
        ] += 5.0 * psf

    detection = detect_point_image_peaks(
        image,
        np.full(shape, 0.01),
        np.zeros(shape, dtype=bool),
        np.zeros(shape, dtype=bool),
        psf,
        pixel_scale_arcsec=pixel_scale,
        threshold_snr=5.0,
        center_smoothing_sigma_arcsec=0.15,
        lens_center_search_radius_arcsec=0.35,
        central_exclusion_radius_arcsec=0.25,
        outer_detection_radius_arcsec=2.25,
        minimum_peak_separation_arcsec=0.30,
        maximum_peaks=4,
    )

    assert detection["center_x_arcsec"] == pytest.approx(0.0, abs=0.1)
    assert detection["center_y_arcsec"] == pytest.approx(0.0, abs=0.1)
    assert len(detection["peaks"]) == 2
    assert all(item["snr"] >= 5.0 for item in detection["peaks"])
