# Equal-budget observation-condition randomization improves synthetic Einstein-radius regression robustness

## Abstract

Observation-condition mismatch can degrade learned strong-lens parameter
estimates, but extra augmented images and extra optimizer updates can confound
the apparent benefit of randomization. We evaluate a one-target CNN regressor
for the angular Einstein radius, \(\theta_E\), under a matched training
budget. Canonical training (C), uniform observation-condition variation (G),
and a source-development failure-weighted policy (R) use the same backbone,
initialization, source-parent schedule, 880 optimizer updates, batch size and
source-image exposure. On a frozen parent-disjoint synthetic holdout, the
four-stress mean absolute error (MAE) was 0.141251 arcsec for C and 0.111507
arcsec for G. The paired change G−C was −0.029744 arcsec (10,000-resample
parent-bootstrap 95% interval [−0.045725, −0.012141]). R obtained 0.107385
arcsec, but R−G was −0.004122 arcsec with an interval spanning zero
([−0.026208, 0.009341]). Thus condition variation improved robustness within
the tested synthetic generator, while the failure-weighted allocation did not
demonstrate an additional advantage over uniform sampling. A separate
AutoLens/Lenstronomy audit produced byte-identical saved image pairs, so we do
not claim simulator transfer, real-observation accuracy, or superiority over
published methods.

## 1. Introduction

Convolutional inference of strong-lens parameters is established, as are
sensitivity to noise and image formation and the use of domain adaptation for
simulated lensing domains. The unresolved question addressed here is narrower:
when the optimizer budget and source-image exposure are held fixed, does
source-only variation of observation conditions improve a scalar Einstein-radius
regressor on unseen lens systems?

The study is designed as a controlled reliability experiment rather than a new
CNN architecture. It separates the effect of exposing training to nuisance
conditions from the effect of simply providing more updates or more images.
It also records a negative result for a failure-weighted allocation policy and
audits a proposed alternate renderer instead of assuming that a package change
creates a domain shift.

Our claims are intentionally bounded to the declared synthetic generator,
parent prior, preprocessing and observation conditions. We do not compare our
MAE numerically with published \(R^2\), signed residual, posterior coverage or
real-data results because those studies use different targets, generators,
metrics and information-access settings.

## 2. Related work and comparison boundary

Hezaveh et al. established CNN inference of lens parameters and used varied
observational effects; Pearson et al. studied sensitivity to lens light,
signal-to-noise and mass/light diversity. Swierc et al. and Agarwal et al.
provide close domain-adaptation precedents for Einstein-radius inference, but
use unlabeled target images during adaptation and report objectives and metrics
that are not identical to ours. Gawade et al. provide simulation-realism and
HSC-oriented evidence beyond this study. Cross-code comparison is also
established in classical lens modeling (e.g. Galan et al.). A full comparison
matrix, including target access, metrics and permitted claims, is in
[`step3_claim_comparison_matrix_2026-11-01.md`](../../../archive/docs/step3_claim_comparison_matrix_2026-11-01.md).

Accordingly, the contribution claimed here is a matched-budget synthetic
robustness measurement and reproducible benchmark record, not a new universal
augmentation algorithm or a published-method leaderboard result.

## 3. Methods

### 3.1 Synthetic task

Each image is generated from an SIE lens with extended Sérsic source and lens
light, an unresolved lensed point source, a Gaussian PSF and a simplified
detector/noise model. The target is the simulator's angular Einstein radius in
arcseconds. Data roles are separated by parent system. Source training uses 697
qualified parents; the frozen holdout contains 150 parents and was not used for
fit selection. The CNN receives image pixels and predicts one scalar.

The primary holdout stress cells are native pixel scale 0.04 and 0.06
arcsec/pixel, half exposure, and doubled read noise. The canonical condition is
reported as a guardrail, not folded into the stress composite.

### 3.2 Equal-budget arms

All arms use the same CNN, optimizer, Huber loss, normalization policy,
initialization weights, batch size 16, and 880 optimizer updates per seed. Ten
paired seeds (730001–730010) are fit for each arm.

- **C:** canonical observation only.
- **G:** uniform sampling over the frozen observation-condition cells.
- **R:** source-development failure-weighted sampling over the same cells.

The arms share the same realized source-parent schedule within each seed. The
holdout is accessed only after predictions are sealed. Training and holdout
parent IDs have zero overlap.

### 3.3 Metrics and uncertainty

The primary endpoint is the equal-weight mean absolute error averaged across
the four stress conditions within each parent. Paired contrasts are reported
in arcseconds. Confidence intervals are percentile intervals from 10,000
resamples of the 150 independent parents; condition views remain grouped within
each parent. Variation across the ten training seeds is reported separately in
the machine-readable result.

### 3.4 Renderer audit and unseen observation family

As a secondary engineering evaluation, fixed checkpoints were applied to 150
new paired parents rendered through the project’s Lenstronomy and AutoLens
paths under nine conditions, including three Moffat-PSF cells absent from source
training. Predictions were sealed before labels were joined. The audit found
1,350/1,350 paired image products and 40,500/40,500 paired predictions exactly
equal at saved precision. The Moffat cells therefore measure an unseen
observation-family shift shared by both renderer labels; they do not measure a
simulator-induced image-domain shift.

## 4. Results

### 4.1 Equal-budget primary result

| Arm | Canonical MAE (arcsec) | Four-stress MAE (arcsec) |
|---|---:|---:|
| C | 0.091536 | 0.141251 |
| G | 0.087591 | 0.111507 |
| R | 0.092366 | 0.107385 |

![Equal-budget MAE by condition](../../../results/ml/equal-budget-final-v1/figures/equal_budget_mae_by_condition.png)

The primary G−C contrast was −0.029744 arcsec, with parent-bootstrap 95%
interval [−0.045725, −0.012141]. This supports improved robustness from
source-only condition variation under the matched budget for the tested
synthetic stresses. The R−G contrast was −0.004122 arcsec, with interval
[−0.026208, 0.009341], so the data do not establish a reliable benefit from
failure weighting beyond uniform variation.

![Parent-bootstrap paired contrasts](../../../results/ml/equal-budget-final-v1/figures/equal_budget_paired_contrasts.png)

The older 27.4% improvement came from a historical canonical-versus-randomized
comparison with unequal update and image-exposure budgets. It is retained as
context but is not used as the causal estimate in this paper.

### 4.2 Renderer and Moffat audit

The rapid paired release yielded the following primary Moffat contrasts for
both renderer labels because the saved A/B images were identical:

| Contrast | MAE change (arcsec) | Parent-bootstrap 95% interval |
|---|---:|---:|
| G−C | −0.001000 | [−0.009085, 0.006570] |
| R−G | +0.009602 | [+0.005284, +0.013910] |
| R−C | +0.008602 | [−0.000662, 0.017561] |

These results are reported as an unseen PSF-family robustness check and
renderer-equivalence control, not as cross-simulator transfer.

## 5. Limitations

The benchmark uses one synthetic parent prior, one scalar target and a
project-specific detector, PSF and preprocessing stack. The Moffat family is
engineered synthetic data, not a calibrated telescope PSF. The AutoLens and
Lenstronomy release paths share enough image-forming code and matched
conventions to produce identical saved products; an independent simulator
domain shift remains future work. The equal-budget holdout is frozen and
parent-disjoint from training, but it was previously scored by older models, so
the present result is an independent equal-budget refit on a frozen holdout,
not a never-before-opened dataset. The ten-seed interval conditions on the
observed training seeds and does not quantify all source-data or policy-
construction uncertainty.

No real-image \(\theta_E\) labels were used. CASTLES and other telescope data
are not accuracy benchmarks here. No exact reproduction of the Swierc or
Agarwal methods was run, and no numerical superiority claim over those papers
is made.

## 6. Conclusion

Under matched CNN, optimization and source-exposure budgets, uniform
observation-condition variation reduced four-stress Einstein-radius MAE on
parent-disjoint synthetic systems. A failure-weighted allocation did not show
a reliable additional gain over uniform variation. The result supports a
controlled synthetic robustness conclusion and a reproducible benchmark
boundary; it does not establish simulator-independent inference, real-data
accuracy or superiority over published methods.

## 7. Reproducibility artifacts

- Equal-budget protocol and result: [`equal_budget_final_v1.md`](../../equal_budget_final_v1.md).
- Equal-budget machine-readable report:
  `results/ml/equal-budget-final-v1/final_report.json`.
- Renderer audit: [`cross_simulator_rapid_v2_audit.md`](../../../archive/docs/cross_simulator_rapid_v2_audit.md).
- Renderer identity record:
  `results/ml/cross-simulator-rapid-v2/renderer_identity_audit.json`.
- Prior-work and claim matrix:
  [`step3_claim_comparison_matrix_2026-11-01.md`](../../../archive/docs/step3_claim_comparison_matrix_2026-11-01.md).
- Submission reproducibility checklist:
  [`reproducibility_checklist_submission_v1.md`](../../../archive/docs/reproducibility_checklist_submission_v1.md).
