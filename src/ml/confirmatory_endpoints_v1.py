"""Paired tail guardrails and conditional seed sensitivity in arcseconds.

These supplement the crossed MAE bootstrap. They never select methods or
sample sizes from final outcomes. Percentiles use NumPy's linear convention.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import t

from src.ml.confirmatory_statistics_v1 import validate_errors


def weighted_resample_quantile(errors, parent_counts, quantile=.95):
    """Exact linear quantile after whole-parent bootstrap replication.

    Sorting once per seed avoids repeatedly sorting expanded observations.
    parent_counts has [bootstrap,parent] shape; each condition of a parent
    receives the same integer replication count.
    """
    errors=np.asarray(errors,dtype=float)
    counts=np.asarray(parent_counts)
    if (errors.ndim!=3 or counts.ndim!=2 or counts.shape[1]!=errors.shape[1] or
        not np.isfinite(errors).all() or np.any(errors<0) or
        not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0) or
        not np.all(counts.sum(1)==errors.shape[1]) or not 0<=quantile<=1):
        raise ValueError("Invalid parent-resampling quantile inputs")
    seeds,parents,conditions=errors.shape
    rank=quantile*(parents*conditions-1)
    lower,upper=int(np.floor(rank)),int(np.ceil(rank))
    out=np.empty((len(counts),seeds))
    for seed in range(seeds):
        values=errors[seed].ravel()
        order=np.argsort(values,kind="stable")
        sorted_values=values[order]
        cumulative=np.cumsum(counts[:,order//conditions],axis=1)
        lo=np.argmax(cumulative>lower,axis=1)
        hi=np.argmax(cumulative>upper,axis=1)
        out[:,seed]=sorted_values[lo]+(rank-lower)*(sorted_values[hi]-sorted_values[lo])
    return out


def crossed_tail_difference(candidate,comparator,*,resamples=10000,alpha=.025,seed=720901):
    a,b=validate_errors(candidate,comparator)
    if type(resamples) is not int or resamples<100 or not 0<alpha<.5 or min(a.shape[:2])<2:
        raise ValueError("Invalid crossed tail controls")
    seeds,parents,_=a.shape
    rng=np.random.default_rng(seed)
    boot=np.empty(resamples)
    for start in range(0,resamples,32):
        size=min(32,resamples-start)
        sw=rng.multinomial(seeds,np.ones(seeds)/seeds,size=size)
        pw=rng.multinomial(parents,np.ones(parents)/parents,size=size)
        difference=weighted_resample_quantile(a,pw)-weighted_resample_quantile(b,pw)
        boot[start:start+size]=(sw*difference).sum(1)/seeds
    qa=np.quantile(a.reshape(seeds,-1),.95,axis=1,method="linear")
    qb=np.quantile(b.reshape(seeds,-1),.95,axis=1,method="linear")
    return {"candidate_mean_seed_p95_arcsec":float(qa.mean()),
        "comparator_mean_seed_p95_arcsec":float(qb.mean()),"difference_arcsec":float((qa-qb).mean()),
        "upper_one_sided_arcsec":float(np.quantile(boot,1-alpha,method="linear")),
        "alpha_one_sided":alpha,"resamples":resamples,"bootstrap_seed":seed,
        "quantile":.95,"quantile_method":"linear","statistic":"mean_seed_parent_condition_p95"}


def seed_t_sensitivity(candidate,comparator,*,alpha=.025):
    a,b=validate_errors(candidate,comparator)
    d=(a-b).mean((1,2))
    if len(d)<2 or not 0<alpha<.5:
        raise ValueError("Sensitivity requires multiple seeds and valid alpha")
    standard_error=d.std(ddof=1)/np.sqrt(len(d))
    return {"per_seed_differences_arcsec":d.tolist(),"difference_arcsec":float(d.mean()),
        "upper_one_sided_arcsec":float(d.mean()+t.ppf(1-alpha,len(d)-1)*standard_error),
        "alpha_one_sided":alpha,"degrees_of_freedom":len(d)-1,
        "scope":"seed-only Student t sensitivity conditional on sampled parents; not crossed primary inference"}
