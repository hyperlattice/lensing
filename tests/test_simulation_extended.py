import importlib.util

import numpy as np
import pytest
import yaml

from src.simulation.priors import draw_sie_parameters, draw_stress_parameters, validate_sie_parameters
from src.simulation.render import render_sie_extended_source, render_sis_extended_source


def _config():
    with open("configs/simulation_pilot_v1.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_sie_prior_draw_is_reproducible_and_inside_configured_ranges():
    config = _config()
    first = draw_sie_parameters(3001, 2, config)
    second = draw_sie_parameters(3001, 2, config)
    assert first == second
    assert validate_sie_parameters(first) == []
    proposal = config["pilot"]["sie_extended"]
    assert proposal["q"][0] <= first["q"] <= proposal["q"][1]
    assert proposal["source_n"][0] <= first["source_n"] <= proposal["source_n"][1]


def test_stress_prior_is_reproducible_and_has_external_shear():
    first = draw_stress_parameters(9001, 0, _config())
    second = draw_stress_parameters(9001, 0, _config())
    assert first == second
    assert first["mass_model"] == "SIE+SHEAR"
    assert np.isclose(
        first["gamma_amplitude"],
        np.hypot(first["gamma1"], first["gamma2"]),
    )
    assert first["stress_case"] == "quad_candidate"
    assert validate_sie_parameters(first) == []


@pytest.mark.skipif(
    importlib.util.find_spec("lenstronomy") is None,
    reason="optional lenstronomy backend is not installed",
)
def test_circular_sie_matches_sis_and_supersampling_is_measurable(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    parameters = draw_sie_parameters(3001, 0, _config())
    parameters.update({"q": 1.0, "lens_center_x_arcsec": 0.0, "lens_center_y_arcsec": 0.0})
    sie = render_sie_extended_source(parameters, shape=(64, 64), supersampling_factor=1)
    sis = render_sis_extended_source(parameters, shape=(64, 64), supersampling_factor=1)
    assert np.isfinite(sie.image).all()
    assert np.allclose(sie.image, sis.image, rtol=1e-5, atol=1e-7)

    high_resolution = render_sie_extended_source(parameters, shape=(64, 64), supersampling_factor=2)
    relative_change = np.linalg.norm(sie.image - high_resolution.image) / np.linalg.norm(high_resolution.image)
    assert np.isfinite(relative_change)
    assert high_resolution.supersampling_factor == 2
