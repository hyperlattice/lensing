# Equal-budget final comparison (item 1)

The maintained runner accepts `--config configs/ml/equal_budget.yaml` and is
invoked from repository root as `python -m scripts.run_equal_budget_final_v1`.
The YAML extracts the recorded settings without changing the comparison.
New fits save `resolved_config.json`, required by prediction/scoring; historical
completed runs remain associated with their original runner. See the
[experiment index](experiments.md) for input availability and the
[cleanup record](maintenance/cleanup_v1.md) for provenance boundaries.

This rapid study addresses the first missing priority: a fair, independent
source-only comparison under a common training budget. It fits three arms:

- **C:** canonical-condition training only;
- **G:** uniform sampling over the seven frozen observation variations;
- **R:** the frozen failure-weighted observation policy learned from source
  development errors.

All arms use the same CNN, AdamW settings, 880 optimizer updates, batch size
16, paired initialization seeds, and the same source-parent draw stream. G and
R differ only in the observation-cell probabilities; C uses the canonical view
for the same parent draws. The source training and development banks are the
qualified v4 banks. The holdout is the existing 150-parent blind release with
canonical, pixel-scale, exposure, and read-noise conditions.

`scripts/run_equal_budget_final_v1.py fit` never opens the holdout. The
`predict` action reads only image paths and seals predictions. The `score`
action performs the target join afterward and reports per-condition MAE, RMSE,
bias, four-stress composite MAE, and parent-bootstrap intervals for G-C, R-G,
and R-C. The R-G contrast is the primary comparison for whether targeted
variation adds value beyond uniform variation.

This is a final result for the fixed synthetic benchmark, not a new simulator or
real-telescope test. The holdout was previously scored by older models, so the
result must be described as an independent equal-budget refit on a frozen
holdout rather than a never-before-opened dataset. It cannot establish
superiority over Agarwal, Swierc, or another published method.

## Completed result (2026-11-01)

The 30 fits completed (10 seeds per arm). Every seed shared the same initial
weights and parent schedule across C/G/R; every arm used 880 updates and 14,080
source exposures. The source-train and holdout parent IDs have zero overlap.
Predictions were sealed before the target join in
`results/ml/equal-budget-final-v1/predictions_before_scoring.csv`.

| Arm | Canonical MAE (arcsec) | Four-stress MAE (arcsec) |
|---|---:|---:|
| C | 0.091536 | 0.141251 |
| G | 0.087591 | 0.111507 |
| R | 0.092366 | 0.107385 |

The four-stress difference is G minus C = **-0.029744 arcsec**, with a
parent-bootstrap 95% interval of **[-0.045725, -0.012141]**. R minus G is
**-0.004122 arcsec**, with interval **[-0.026208, 0.009341]**. Thus the final
equal-budget result supports the value of varied-condition training over
canonical-only training, but it does not show a reliable additional gain from
the failure-weighted policy over uniform variation. R is better than C on the
four-stress composite by **-0.033867 arcsec** (interval
[-0.068335, -0.005619]), while its canonical MAE is slightly worse than G.

This is the result to use for item 1. It sharpens the claim to: **source-only
condition variation improves robustness on unseen synthetic lens systems under
an equal CNN/update/exposure budget**. It does not establish cross-simulator,
real-data, or published-method superiority. The complete machine-readable
record is `results/ml/equal-budget-final-v1/final_report.json`.
