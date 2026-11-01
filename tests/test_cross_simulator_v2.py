"""Release gating and physical detector controls for the versioned adapter."""
import json

import numpy as np
import pytest
import yaml

from src.simulation.autolens_backend_v2 import mapped_light_parameters, sample_detector
from src.simulation.cross_simulator_v2 import run


@pytest.mark.parametrize("passed,recorded_hash", [(False, "unused"), (True, "stale")])
def test_development_release_rejects_failed_or_stale_controls(tmp_path, passed, recorded_hash):
    validation = tmp_path / "controls.json"
    validation.write_text(json.dumps({"passed": passed, "adapter_sha256": recorded_hash}))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"validation_report": str(validation)}))
    with pytest.raises(ValueError, match="Missing, failed or stale"):
        run(config, generate=True)


def test_detector_units_background_and_same_seed_are_explicit():
    expected = np.full((128, 128), .3)
    a, variance = sample_detector(expected, 100., 2., .03, 19)
    b, _ = sample_detector(expected, 100., 2., .03, 19)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(variance, ((.3 + .03) * 100 + 2**2) / 100**2)
    assert abs(a.mean() - .33) < 5 * np.sqrt(variance[0, 0] / a.size)
    with pytest.raises(ValueError, match="Invalid detector"):
        sample_detector(expected, 0., 2., .03, 19)


def test_bad_light_parameters_rejected_before_optional_backend_import():
    for intensity, radius, index in ((-1, .2, 1), (1, 0, 1), (1, .2, np.nan)):
        with pytest.raises(ValueError, match="Light parameters"):
            mapped_light_parameters(intensity, radius, index)
