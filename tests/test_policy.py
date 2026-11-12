"""Policy formulas and independent RNG streams retained from the original study."""

import numpy as np
import pytest

from src.ml.policy import CELLS, failure_weights, paired_policy_schedule
from src.simulation.observation import condition_for


def test_failure_weights_pairing_clipping_exploration_and_role():
    canonical = np.full((3, 9), 0.1)
    transformed = np.repeat(canonical[..., None], len(CELLS), axis=-1)
    transformed[..., 0] += 0.2
    transformed[..., 1] -= 0.05
    r = failure_weights(canonical, transformed, role="source_development")
    p = np.asarray(r["probabilities"])
    assert p[0] == pytest.approx(0.75 + 0.25 / 7)
    np.testing.assert_allclose(p[1:], 0.25 / 7)
    assert sum(p) == pytest.approx(1)
    uniform = failure_weights(canonical, np.zeros_like(transformed), role="source_development")
    np.testing.assert_allclose(uniform["probabilities"], np.ones(7) / 7)
    with pytest.raises(ValueError, match="source-development"):
        failure_weights(canonical, transformed, role="final")


def test_policy_changes_do_not_change_parent_exposure_or_noise_stream():
    uniform = np.ones(7) / 7
    weighted = np.array([0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    kwargs = dict(updates=60, batch_size=16, seed=12071)
    a = paired_policy_schedule(["a", "b", "c"], uniform, **kwargs)
    b = paired_policy_schedule(["a", "b", "c"], weighted, **kwargs)
    assert a == paired_policy_schedule(["a", "b", "c"], uniform, **kwargs)
    assert [x["parents"] for x in a] == [x["parents"] for x in b]
    assert [x["noise_seeds"] for x in a] == [x["noise_seeds"] for x in b]
    assert [x["cell"] for x in a] != [x["cell"] for x in b]
    assert all(len(x["parents"]) == 16 for x in a)
    seeds = [s for x in a for s in x["noise_seeds"]]
    assert len(set(seeds)) == len(seeds)
    for cell in CELLS:
        c = condition_for(cell)
        assert c["psf_family"] == "gaussian"
        assert abs(c["shape"][0] * c["pixel_scale_arcsec"] - 12.8) <= c["pixel_scale_arcsec"] / 2
