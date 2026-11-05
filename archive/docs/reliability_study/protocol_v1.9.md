# Reliability study protocol v1.9 — Week 9 hybrid initialization

**Status:** frozen before Week 9 image-derived peak extraction, optimizer fitting, and validation target scoring  
**Date:** 2026-09-26  
**Base protocol:** [`protocol_v1.md`](protocol_v1.md)  
**Prior amendments:** [`protocol_v1.1.md`](protocol_v1.1.md) through [`protocol_v1.8.md`](protocol_v1.8.md)

## Question and phase boundary

Test whether the successful Week 8 domain-randomized CNN is useful as an initializer for the existing blind classical point-image geometry estimator. This is a validation-only development comparison, not the final blind holdout. `fit_sie_subset` is prohibited because its initialization reads truth-sidecar parameters.

## Fixed sample and conditions

- Use all 150 Week 3 validation parents in each of the five Week 8 conditions: canonical, pixel scale 0.04, pixel scale 0.06, half exposure, and doubled read noise.
- Use condition manifests from the frozen pixel-scale and observation-systematics releases. Match every condition by `parent_sample_id`.
- Use the Week 8 randomized checkpoints for seeds 12001, 12002, and 12003. No retraining, checkpoint changes, or optimizer tuning is allowed.
- Save the complete observation-only peak detections and fit outputs before joining any validation `theta_E` target. The target join is permitted only for final error summaries after the predictions file exists.

## Frozen starts and shared fit

Use the same peak detector and bounded SIS/SIE point-image estimator as the established observation-only comparator, with its v1.3 thresholds and bounds. Every fit uses the same observed peaks, image-derived center, lens equation, parameter bounds, tolerances, and maximum of 50 function evaluations.

For each parent-condition-checkpoint seed, compare:

1. **Default:** the arithmetic mean `theta_E` in the original 700-parent training split, calculated from training labels only.
2. **Heuristic:** the median radius of the detected peaks about the observation-derived center, clipped to the frozen `[0.25, 2.25]` arcsec bounds. If fewer than two peaks are detected, record the heuristic as unavailable and use the training mean only for the recorded fallback path.
3. **ML:** the Week 8 randomized CNN prediction for the same parent, condition, and seed, clipped to `[0.25, 2.25]` arcsec.

For two detected peaks, the established estimator solves a closed-form SIS geometry and does not use an initialization. Record all three rows as the same analytic fit and mark them `initialization_used=false`; exclude these pairs from optimizer-speed comparisons. Fewer than two detected peaks are recorded as no-fit fallbacks and excluded from optimizer-speed comparisons. Three or four detected peaks use the same bounded SIE least-squares optimizer for all three starts.

## Measures and analysis

Record detected peak count, fit success, fallback class, final `theta_E`, absolute error after target join, final residual RMS, function evaluations, runtime, boundary proximity, and local Jacobian conditioning when returned by the optimizer. Report canonical and each stress condition separately, all three CNN seeds separately and averaged, plus paired parent-level differences between ML and each comparator.

The comparison is descriptive and validation-based. The primary practical diagnostic is whether ML reduces mean function evaluations versus both default and heuristic starts on the subset with at least three detected peaks, without lowering convergence or worsening final `theta_E` MAE by more than 0.01 arcsec. Report paired parent-bootstrap 95% intervals with 10,000 resamples and seed 6031. Do not select or tune the Week 11 method from Week 9 fits beyond the stated validation-only model choice.

## Integrity and interpretation limits

- No truth-sidecar parameters, historical test data, or CASTLES labels may enter detection, initialization, fitting, or checkpoint selection.
- The image-derived point peaks and lens center are measurements from observations, not physical truth.
- This control is a point-image lens-equation fit, not a full pixel-likelihood fit. Its residual and local Jacobian do not represent a pixel likelihood or posterior uncertainty.
- Closed-form two-peak cases cannot demonstrate initialization speedup. Any efficiency conclusion applies only to the fitted three/four-peak subset and tested synthetic conditions.
