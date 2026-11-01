import pytest

from src.ml.power_planning_v1 import normal_power,simulate_crossed_power,wilson_interval


def test_gaussian_upper_components_include_finite_pilot_uncertainty():
    from src.ml.power_planning_v1 import gaussian_upper_components
    c={"pilot_seeds":5,"pilot_parents":120,"seed_variance":.0001,"raw_seed_variance":.0001,
       "parent_variance":.0002,"raw_parent_variance":.0002,"interaction_variance":.0003}
    upper=gaussian_upper_components(c)
    assert upper["seed_variance"]>c["seed_variance"]
    assert upper["parent_variance"]>c["parent_variance"]
    assert upper["interaction_variance"]>c["interaction_variance"]


def test_more_parents_cannot_remove_seed_variance_floor():
    c={"seed_variance":.001,"parent_variance":0.,"interaction_variance":0.}
    assert normal_power(c,seeds=10,parents=600)==normal_power(c,seeds=10,parents=2000)
    assert normal_power(c,seeds=20,parents=600)>normal_power(c,seeds=10,parents=600)


def test_planning_uses_stipulated_effect_not_pilot_mean():
    c={"seed_variance":0.,"parent_variance":0.,"interaction_variance":0.,"pilot_mean_ignored":999.}
    good=simulate_crossed_power(c,seeds=3,parents=4,trials=20,resamples=100,true_delta=-.01)
    bad=simulate_crossed_power(c,seeds=3,parents=4,trials=20,resamples=100,true_delta=0.)
    assert good["estimated_power"]==1.
    assert bad["estimated_power"]==0.
    assert not good["observed_pilot_mean_used"]
    assert wilson_interval(20,20)[0]<1.
    with pytest.raises(ValueError):
        normal_power({**c,"seed_variance":-1},seeds=10,parents=600)
