"""Variance-only Gaussian crossed-effects power planning for a fixed endpoint.

No observed pilot treatment mean enters these calculations. Planning assumes
additive Gaussian seed, parent and interaction effects and remains sensitive
to variance estimation from few pilot seeds. Simulated decisions use the same
crossed-percentile statistic as confirmation, at a stipulated treatment mean.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm,chi2

from src.ml.confirmatory_statistics_v1 import crossed_paired_difference


def gaussian_upper_components(components,*,confidence=.9):
    """Conservative component bounds under a balanced Gaussian crossed model.

    Main-effect bounds retain residual noise in row/column mean variances.
    Bonferroni allocates the error probability across three chi-square bounds.
    These are model-dependent planning diagnostics, not distribution-free CIs.
    """
    s,n=components["pilot_seeds"],components["pilot_parents"]
    if min(s,n)<2 or not 0<confidence<1:
        raise ValueError("Invalid pilot dimensions or confidence")
    residual=components["interaction_variance"]
    seed_mean=max(0.,components["raw_seed_variance"]+residual/n)
    parent_mean=max(0.,components["raw_parent_variance"]+residual/s)
    alpha=(1-confidence)/3
    return {"seed_variance":float((s-1)*seed_mean/chi2.ppf(alpha,s-1)),
        "parent_variance":float((n-1)*parent_mean/chi2.ppf(alpha,n-1)),
        "interaction_variance":float((s-1)*(n-1)*residual/chi2.ppf(alpha,(s-1)*(n-1))),
        "joint_confidence_under_gaussian_model":confidence,
        "note":"Main-effect bounds include residual noise; Bonferroni upper bounds under balanced Gaussian crossed effects only"}


def normal_power(components,*,seeds,parents,alpha=.025,true_delta=-.010,margin=-.005,inflation=1.):
    if seeds<2 or parents<2 or not 0<alpha<.5 or inflation<1:
        raise ValueError("Invalid power-design controls")
    values=np.asarray([components[k] for k in ("seed_variance","parent_variance","interaction_variance")])
    if not np.isfinite(values).all() or np.any(values<0):
        raise ValueError("Invalid nonnegative variance components")
    variance=inflation*(values[0]/seeds+values[1]/parents+values[2]/(seeds*parents))
    if variance==0:
        return float(true_delta<margin)
    return float(norm.cdf((margin-true_delta)/np.sqrt(variance)-norm.ppf(1-alpha)))


def wilson_interval(successes,total,*,confidence=.95):
    if type(total) is not int or total<1 or not 0<=successes<=total:
        raise ValueError("Invalid Monte Carlo successes")
    z=norm.ppf((1+confidence)/2)
    p=successes/total
    denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [float(center-half),float(center+half)]


def simulate_crossed_power(components,*,seeds,parents,trials=200,resamples=10000,alpha=.025,
        true_delta=-.010,margin=-.005,inflation=1.,random_seed=720831):
    normal=normal_power(components,seeds=seeds,parents=parents,alpha=alpha,
        true_delta=true_delta,margin=margin,inflation=inflation)
    if type(trials) is not int or trials<20:
        raise ValueError("At least twenty planning simulations required")
    sd=np.sqrt(inflation*np.asarray([components[k] for k in
        ("seed_variance","parent_variance","interaction_variance")]))
    rng=np.random.default_rng(random_seed)
    successes=0
    for trial in range(trials):
        d=(true_delta+rng.normal(0,sd[0],(seeds,1))+rng.normal(0,sd[1],(1,parents))+
            rng.normal(0,sd[2],(seeds,parents)))
        # Positive/negative parts provide valid nonnegative error inputs while
        # preserving the simulated paired difference exactly, without clipping.
        a=np.maximum(d,0)[...,None]
        b=np.maximum(-d,0)[...,None]
        result=crossed_paired_difference(a,b,resamples=resamples,alpha=alpha,
            seed=int(rng.integers(2**31)),relative_uncertainty=False)
        successes+=result["upper_one_sided_arcsec"]<margin
    return {"seeds":seeds,"parents":parents,"trials":trials,"bootstrap_resamples":resamples,
        "alpha_one_sided":alpha,"assumed_true_delta_arcsec":true_delta,"margin_arcsec":margin,
        "variance_inflation":inflation,"planning_seed":random_seed,"successes":int(successes),
        "estimated_power":successes/trials,"monte_carlo_wilson_95_interval":wilson_interval(int(successes),trials),
        "normal_approximation_power":normal,"observed_pilot_mean_used":False,
        "model":"additive independent Gaussian seed/parent/interaction paired differences"}
