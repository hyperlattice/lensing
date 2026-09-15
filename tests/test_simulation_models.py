import importlib.util

import numpy as np
import pytest

from src.simulation.models import (
    backend_convention_check,
    backend_status,
    e1_e2_to_q_phi,
    ellipticity_from_q_phi,
    sie_kwargs,
    shear_kwargs,
    sis_deflection,
)


def test_q_phi_conversion_round_trips_and_matches_expected_formula():
    e1, e2 = ellipticity_from_q_phi(0.7, 0.3)
    assert np.isclose(e1, ((1 - 0.7) / (1 + 0.7)) * np.cos(0.6))
    assert np.isclose(e2, ((1 - 0.7) / (1 + 0.7)) * np.sin(0.6))
    phi, q = e1_e2_to_q_phi(e1, e2)
    assert np.isclose(phi, 0.3)
    assert np.isclose(q, 0.7)


def test_sis_control_has_constant_deflection_norm_away_from_center():
    x, y = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
    alpha_x, alpha_y = sis_deflection(x, y, 0.8)
    radius = np.hypot(x, y)
    assert np.isfinite(alpha_x).all()
    assert np.isfinite(alpha_y).all()
    assert np.allclose(np.hypot(alpha_x, alpha_y)[radius > 0], 0.8)
    assert alpha_x[5, 5] == 0.0
    assert alpha_y[5, 5] == 0.0


def test_sie_kwargs_are_backend_shaped():
    kwargs = sie_kwargs(0.8, 0.6, 0.25)
    assert set(kwargs) == {"theta_E", "e1", "e2", "center_x", "center_y"}
    assert kwargs["theta_E"] == 0.8


def test_shear_kwargs_are_backend_shaped():
    assert shear_kwargs(-0.03, 0.04) == {"gamma1": -0.03, "gamma2": 0.04}


@pytest.mark.skipif(
    importlib.util.find_spec("lenstronomy") is None,
    reason="optional lenstronomy backend is not installed",
)
def test_backend_conversion_agrees_with_project_convention(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    result = backend_convention_check()
    assert result["passed"] is True
    assert result["max_absolute_difference"] <= result["atol"]


def test_backend_status_is_serializable():
    status = backend_status().to_dict()
    assert status["name"] == "lenstronomy"
    assert isinstance(status["python"], str)
