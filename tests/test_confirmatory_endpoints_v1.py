import numpy as np
import pytest

from src.ml.confirmatory_endpoints_v1 import (weighted_resample_quantile,
    crossed_tail_difference,seed_t_sensitivity)


def test_weighted_tail_matches_explicit_whole_parent_resampling():
    rng=np.random.default_rng(12)
    errors=rng.uniform(size=(3,8,3))
    counts=np.array([[0,3,1,0,2,0,1,1],[8,0,0,0,0,0,0,0],[1]*8])
    actual=weighted_resample_quantile(errors,counts)
    expected=np.array([np.quantile(errors[:,np.repeat(np.arange(8),row)].reshape(3,-1),.95,
        axis=1,method="linear") for row in counts])
    np.testing.assert_allclose(actual,expected,atol=1e-14)


def test_tail_preserves_constant_paired_difference():
    a=np.random.default_rng(3).uniform(.1,.3,(4,15,3))
    report=crossed_tail_difference(a,a+.02,resamples=100)
    assert report["difference_arcsec"]==pytest.approx(-.02)
    assert report["upper_one_sided_arcsec"]==pytest.approx(-.02)


def test_seed_sensitivity_retains_training_uncertainty():
    a=np.broadcast_to(np.array([.02,.12,.22])[:,None,None],(3,40,2))
    b=np.full_like(a,.13)
    report=seed_t_sensitivity(a,b)
    assert report["difference_arcsec"]<0
    assert report["upper_one_sided_arcsec"]>0
