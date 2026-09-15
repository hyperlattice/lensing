import numpy as np

from src.modeling.fit_lens import recover_sis_theta_e


def test_recover_sis_theta_e_from_two_exact_images():
    result = recover_sis_theta_e(
        [1.2, -0.8],
        [0.0, 0.0],
        source_x_arcsec=0.2,
        source_y_arcsec=0.0,
    )

    assert np.isclose(result["theta_E_arcsec"], 1.0)
    assert result["max_residual_arcsec"] < 1e-12
