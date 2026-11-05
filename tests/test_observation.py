import importlib.util

import numpy as np
import pytest
import yaml

from src.simulation.observation import render_observation_scene, realize_detector_observation
from src.simulation.priors import draw_observation_parameters


def test_detector_noise_is_sampled_in_count_space_and_reproducible():
    expected = np.ones((2, 2), dtype=float)
    first = realize_detector_observation(
        expected,
        background_rate=0.2,
        exposure_time=10.0,
        read_noise_electrons=3.0,
        noise_seed=42,
    )
    second = realize_detector_observation(
        expected,
        background_rate=0.2,
        exposure_time=10.0,
        read_noise_electrons=3.0,
        noise_seed=42,
    )
    assert np.array_equal(first.observed_electrons, second.observed_electrons)
    assert np.allclose(first.expected_electrons, 12.0)
    assert np.allclose(first.variance_rate, (12.0 + 9.0) / 100.0)


@pytest.mark.skipif(
    importlib.util.find_spec("lenstronomy") is None,
    reason="optional lenstronomy backend is not installed",
)
def test_observation_scene_contains_point_sources_psf_and_foreground_light(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    with open("configs/simulation/pilot.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    parameters = draw_observation_parameters(6001, 0, config)
    scene = render_observation_scene(parameters, shape=(64, 64), supersampling_factor=1)

    assert scene.expected_scene.shape == (64, 64)
    assert np.isfinite(scene.expected_scene).all()
    assert np.isfinite(scene.clean_scene).all()
    assert np.all(scene.expected_scene >= 0)
    assert len(scene.point_positions_x_arcsec) >= 2
    assert len(scene.point_positions_x_arcsec) == len(scene.point_amplitudes)
    assert not np.array_equal(scene.clean_scene, scene.expected_scene)


@pytest.mark.skipif(
    importlib.util.find_spec("lenstronomy") is None,
    reason="optional lenstronomy backend is not installed",
)
def test_zero_external_shear_matches_the_no_shear_observation_scene(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    with open("configs/simulation/pilot.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    parameters = draw_observation_parameters(6001, 0, config)
    no_shear = render_observation_scene(parameters, shape=(64, 64), supersampling_factor=1)
    zero_shear_parameters = {**parameters, "gamma1": 0.0, "gamma2": 0.0}
    zero_shear = render_observation_scene(
        zero_shear_parameters,
        shape=(64, 64),
        supersampling_factor=1,
    )

    assert zero_shear.shear_parameters == {"gamma1": 0.0, "gamma2": 0.0}
    assert np.allclose(zero_shear.clean_scene, no_shear.clean_scene, rtol=1e-6, atol=1e-7)
    assert np.allclose(zero_shear.expected_scene, no_shear.expected_scene, rtol=1e-6, atol=1e-7)
