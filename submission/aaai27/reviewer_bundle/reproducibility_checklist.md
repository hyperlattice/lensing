# Reproducibility checklist (reviewer-safe supplement)

## Study identity

- The primary question is whether source-only observation-condition variation
  improves a CNN Einstein-radius regressor under a matched training budget.
- Arms are canonical (C), uniform variation (G), and failure-weighted variation
  (R).
- All arms use the same architecture, initialization, optimizer, batch size,
  update count, and source-image exposure.
- Ten paired training seeds are used, with 880 optimizer updates per fit and
  batch size 16.
- The final holdout contains 150 parent systems and is parent-disjoint from
  source fitting.

## Evaluation safeguards

- Model predictions are sealed before target labels are joined.
- The primary endpoint is parent-averaged four-stress MAE in arcseconds.
- Bootstrap resampling is performed over independent parent systems, keeping
  observation-condition views grouped within each parent.
- The R-versus-G contrast is reported as inconclusive rather than selected out.
- No real-image score, independent-simulator transfer claim, or published-model
  leaderboard comparison is included.

## Main numbers

| Arm | Four-stress MAE (arcsec) |
|---|---:|
| C | 0.141251 |
| G | 0.111507 |
| R | 0.107385 |

The primary G-minus-C contrast is -0.029744 arcsec with a parent-bootstrap 95%
interval of [-0.045725, -0.012141]. The R-minus-G contrast is -0.004122 arcsec
with interval [-0.026208, 0.009341].

## Interpretation boundary

The result is bounded to the declared synthetic generator, target definition,
preprocessing and stress matrix. The alternate-renderer run is an image and
prediction identity audit, not evidence of simulator transfer. The real-data
benchmark remains an unscored, separately gated follow-on.
