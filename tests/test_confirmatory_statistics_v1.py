"""Statistical checks distinguish seed uncertainty from parent-only intervals."""
import numpy as np


def test_relative_improvement_uses_jointly_resampled_comparator():
    from src.ml.confirmatory_statistics_v1 import crossed_paired_difference
    baseline=np.full((3,8,2),.1)
    candidate=baseline*.8
    result=crossed_paired_difference(candidate,baseline,resamples=100,alpha=.025)
    bounds=result["relative_improvement_bounds_percent"]
    np.testing.assert_allclose([bounds["lower_one_sided_percent"],bounds["upper_one_sided_percent"]],[20.,20.])
    zero=crossed_paired_difference(np.zeros_like(baseline),np.zeros_like(baseline),resamples=100)
    assert zero["relative_improvement_bounds_percent"] is None
    assert zero["undefined_relative_bootstrap_resamples"]==100
import pytest
import torch

from src.ml.adaptive_mmd_v1 import adaptive_gaussian_mmd
from src.ml.confirmatory_statistics_v1 import crossed_paired_difference, paired_variance_components


def test_constant_paired_effect_and_condition_averaging():
    b = np.full((10, 80, 3), .1)
    a = b-.01
    r = crossed_paired_difference(a, b, resamples=1000)
    assert r["difference_arcsec"] == pytest.approx(-.01)
    assert r["upper_one_sided_arcsec"] == pytest.approx(-.01)
    assert r["relative_improvement_percent"] == pytest.approx(10.)
    components = paired_variance_components(a, b)
    assert components["seed_variance"] == pytest.approx(0., abs=1e-30)
    assert components["parent_variance"] == pytest.approx(0., abs=1e-30)


def test_more_parents_cannot_erase_seed_only_variation():
    effect = np.linspace(-.02, .02, 10)
    b = np.full((10, 20, 3), .1)
    a = b+effect[:, None, None]
    short = crossed_paired_difference(a, b, resamples=2000)
    long = crossed_paired_difference(np.repeat(a, 20, axis=1), np.repeat(b, 20, axis=1), resamples=2000)
    assert short["upper_one_sided_arcsec"] > .005
    assert long["upper_one_sided_arcsec"] > .005
    c = paired_variance_components(a, b)
    assert c["seed_variance"] == pytest.approx(np.var(effect, ddof=1))
    assert c["parent_variance"] == pytest.approx(0., abs=1e-30)


def test_paired_sign_reversal_reproducibility_and_invalid_inputs():
    rng = np.random.default_rng(1201)
    a, b = rng.uniform(.03, .2, (10, 40, 3)), rng.uniform(.03, .2, (10, 40, 3))
    r = crossed_paired_difference(a, b, resamples=500)
    assert r == crossed_paired_difference(a, b, resamples=500)
    reverse = crossed_paired_difference(b, a, resamples=500)
    assert reverse["upper_one_sided_arcsec"] == pytest.approx(-r["lower_one_sided_arcsec"])
    with pytest.raises(ValueError, match="nonnegative"):
        crossed_paired_difference(-a, b)
    with pytest.raises(ValueError, match="multiple"):
        crossed_paired_difference(a[:1], b[:1])


def test_adaptive_mmd_collapse_guard_symmetry_and_active_gradients():
    torch.manual_seed(1313)
    a = torch.randn(8, 4, requires_grad=True)
    b = torch.randn(8, 4)+.5
    r = adaptive_gaussian_mmd(a, b)
    torch.testing.assert_close(r, adaptive_gaussian_mmd(b, a))
    r.backward()
    assert torch.isfinite(a.grad).all() and a.grad.norm() > 0
    assert adaptive_gaussian_mmd(torch.zeros(8, 4), torch.zeros(8, 4)) == 0
