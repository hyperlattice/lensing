"""Paired parent/seed inference for a predeclared scalar-regression benchmark.

Inputs are absolute errors in arcsec with [seed,parent,condition] axes. This
module does not load checkpoints, images or target sidecars and never changes
sample size in response to final outcomes. Percentile bootstrap bounds are
approximate; few-seed inference needs sensitivity analysis, not parent-only CIs.
"""
from __future__ import annotations

import numpy as np


def validate_errors(candidate, comparator):
    a, b = np.asarray(candidate, dtype=float), np.asarray(comparator, dtype=float)
    if (a.ndim != 3 or a.shape != b.shape or min(a.shape) < 1 or
        not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0)):
        raise ValueError("Expected paired finite nonnegative [seed,parent,condition] errors")
    return a, b


def crossed_paired_difference(candidate, comparator, *, resamples=10000, alpha=.05, seed=20261027, relative_uncertainty=True):
    """One-sided percentile bound with independent paired parent/seed resampling.

    All conditions of a parent and all methods share the same resampled IDs.
    The statistic averages single-model errors rather than ensemble predictions.
    Caller must first validate identical ordered seed/parent/condition identities.
    """
    a, b = validate_errors(candidate, comparator)
    if type(resamples) is not int or resamples < 100 or not 0 < alpha < .5:
        raise ValueError("Invalid bootstrap count or alpha")
    seeds, parents, _ = a.shape
    if min(seeds, parents) < 2:
        raise ValueError("Crossed inference requires multiple independent seeds and parents")
    d = (a-b).mean(axis=2)
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    baseline_boot = np.empty(resamples) if relative_uncertainty else None
    for start in range(0, resamples, 256):
        size = min(256, resamples-start)
        sw = rng.multinomial(seeds, np.ones(seeds)/seeds, size=size)
        pw = rng.multinomial(parents, np.ones(parents)/parents, size=size)
        boot[start:start+size] = np.einsum("bi,ij,bj->b", sw, d, pw, optimize=True)/(seeds*parents)
        if relative_uncertainty:
            baseline_boot[start:start+size] = np.einsum("bi,ij,bj->b",sw,b.mean(2),pw,optimize=True)/(seeds*parents)
    ratio_bounds=None
    undefined=0
    if relative_uncertainty:
        undefined=int(np.count_nonzero(baseline_boot<=0))
        if not undefined:
            ratio=-100*boot/baseline_boot
            ratio_bounds={"lower_one_sided_percent":float(np.quantile(ratio,alpha,method="linear")),
                "upper_one_sided_percent":float(np.quantile(ratio,1-alpha,method="linear")),
                "alpha_one_sided":alpha,"denominator":"comparator MAE resampled jointly with the difference"}
    return {"candidate_mae_arcsec": float(a.mean()), "comparator_mae_arcsec": float(b.mean()),
            "difference_arcsec": float(d.mean()),
            "upper_one_sided_arcsec": float(np.quantile(boot, 1-alpha, method="linear")),
            "lower_one_sided_arcsec": float(np.quantile(boot, alpha, method="linear")),
            "alpha_one_sided": float(alpha), "resamples": resamples, "bootstrap_seed": seed,
            "seed_count": seeds, "parent_count": parents, "condition_count": a.shape[2],
            "inference": "crossed_paired_percentile_bootstrap",
            "relative_improvement_bounds_percent":ratio_bounds,
            "undefined_relative_bootstrap_resamples":undefined if relative_uncertainty else None,
            "relative_improvement_percent": None if b.mean() == 0 else float(100*(b.mean()-a.mean())/b.mean())}


def paired_variance_components(candidate, comparator):
    """Balanced crossed random-effects method-of-moments planning approximation.

    Seed and parent variance components are estimated from the paired-error
    matrix after averaging conditions. Negative estimates are clipped to zero
    for planning and raw values retained. This is not itself a power analysis.
    """
    a, b = validate_errors(candidate, comparator)
    s, n, _ = a.shape
    if min(s, n) < 2:
        raise ValueError("Variance planning needs at least two seeds and parents")
    d = (a-b).mean(2)
    mu = d.mean()
    residual = d-d.mean(0, keepdims=True)-d.mean(1, keepdims=True)+mu
    mse = float((residual**2).sum()/((s-1)*(n-1)))
    ms_seed = float(n*np.var(d.mean(1), ddof=1))
    ms_parent = float(s*np.var(d.mean(0), ddof=1))
    raw_seed, raw_parent = (ms_seed-mse)/n, (ms_parent-mse)/s
    return {"seed_variance": max(raw_seed, 0.), "parent_variance": max(raw_parent, 0.),
            "interaction_variance": mse, "raw_seed_variance": raw_seed,
            "raw_parent_variance": raw_parent, "pilot_seeds": s, "pilot_parents": n}
