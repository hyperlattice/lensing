# Reliability study protocol v1

**Status:** frozen for development before any new reliability experiment  
**Date:** 2026-09-25  
**Reference:** `gravitational_lensing_field_analysis_and_contribution_plan (1).md`, especially Sections 23, 27, 29, and 30

## Research question

When does image-based inference of a galaxy-scale lens Einstein radius remain reliable under controlled changes to pixel scale, PSF, noise, source structure, and lens-model assumptions, and which targeted mitigations help without using unavailable simulator truth?

The primary estimand is the simulator-defined `theta_E_arcsec` under the declared SIE convention. This study does not estimate total mass, an arbitrary mass map, dark-matter properties, or calibrated real-observation accuracy.

## Starting system

- Reference release: `sim-eng1000-v1`, 1,000 independent parents; fixed ancestry-safe split of 700 train, 150 validation, and 150 historical test parents.
- Reference model: image-only four-block CNN, one normalized 256 x 256 image input, 248,409 trainable parameters, Huber loss, three training seeds `12001`, `12002`, `12003`.
- Canonical angular pixel scale: 0.05 arcsec/pixel.
- Canonical validation reference: three-seed MAE 0.09210 arcsec (sample SD 0.00872); mean P95 absolute error 0.23782 arcsec.
- The previously evaluated 150-parent test split is historical, opened evidence. It is prohibited for all new model, preprocessing, threshold, or mitigation selection. The saved test report may be cited only as historical context. Final evaluation requires a new independent holdout after freeze.
- The P100 suite remains a separate stress set and is not training data.

## Scientific and data rules

1. Preserve the scalar target `theta_E_arcsec` until this study is complete.
2. Never feed seeds, exact nuisance parameters, ancestry identifiers, truth sidecars, or derived target encodings to a model. Metadata may be used only when explicitly declared as deployment-available and tested as such.
3. Use matched counterfactual descendants to isolate each nuisance. Hold the physical lens truth fixed and change one major factor at a time before selected interactions.
4. Every descendant inherits its parent split. No related render, rotation, noise realization, source variant, or PSF variant may cross train/validation/holdout boundaries.
5. Use train and validation for development. Keep the existing test split unopened for further selection and reserve a fresh holdout for the final evaluation.
6. Keep real CASTLES observations separate from synthetic truth. They may support QC and domain-gap analysis; the present products do not establish calibrated real-domain accuracy.
7. Compare the existing CNN, classical SIE fit, and simple baselines on scientifically matched cases where each method applies. Do not launch broad architecture searches.
8. Preserve rejected samples, failures, optimizer outcomes, predictions, configuration/data/source hashes, environment, sample counts, and runtime.

## Frozen hypotheses and decision rules

All paired metrics are computed on the same independent validation parents. The primary effect threshold is 0.01 arcsec, selected before new experiments because it is larger than the recorded three-seed validation MAE sample SD (0.00872 arcsec) and matches the scale used by the existing conservative development gate. It is a practical-effect threshold, not a claim of statistical significance. Report parent-bootstrap 95% percentile intervals using 10,000 resamples (seed 6031); report training-seed variation separately and never label it a confidence interval.

| ID | Hypothesis and intervention | Primary estimand / metric | Support rule |
|---|---|---|---|
| H1 | Pixel-scale dependence: rerender the same parents at angular scales different from the 0.05 arcsec/pixel training scale, initially evaluating the frozen CNN without retraining. Keep lens/source truth, angular PSF, and declared observation conditions fixed. | Paired change in MAE, shifted scale minus canonical scale; also report prediction shift and error versus scale ratio. | A practically important scale penalty is supported when the paired MAE increase is at least 0.01 arcsec and its 95% parent-bootstrap interval excludes zero on the positive side. A mitigation is considered useful only if it lowers shifted-scale error and does not worsen canonical validation MAE by more than 0.01 arcsec. |
| H2 | PSF sensitivity: rerender matched parents with changed PSF width, and an alternate PSF family if the renderer supports a validated comparison. Compare CNN and classical fitting. | Paired change in MAE versus the matched-PSF render, by predeclared mismatch level. | Sensitivity is supported by an increase of at least 0.01 arcsec with interval excluding zero. Compare methods using the paired difference in their mismatch penalties; do not claim one is universally superior from a single condition. |
| H3 | Source-model sensitivity: vary source morphology/profile and source size at fixed lens mass truth; use an alternate source family only after its renderer and labels are validated. | Paired change in MAE and paired prediction shift across source variants. | A practically important source effect requires an MAE increase of at least 0.01 arcsec with interval excluding zero. If the interval is wholly within +/-0.01 arcsec, report evidence that the tested effect is practically small; otherwise call the result inconclusive. |
| H4 | Observation-quality sensitivity: vary exposure/noise and source position to alter image detectability; repeat noise realizations for identical parents. Separate physical multiplicity from detectable multiplicity. | Paired change in MAE from canonical to each lower-quality condition, plus radius-bin and multiplicity effects. | A practically important quality penalty requires an MAE increase of at least 0.01 arcsec with interval excluding zero. Report doubles/quads and detectability strata with counts; do not pool repeat noise images as independent parents. |
| H5 | Targeted domain randomization: train on broader ranges only for nuisance families ranked as high-impact by H1-H4 and later audits. Use the same CNN, data split, and three seeds for the first comparison. | Paired difference in stress MAE (randomized minus original training) across the predeclared stress matrix; canonical MAE is the non-inferiority check. | Support requires stress MAE improvement of at least 0.01 arcsec with the interval wholly below zero, while the upper interval for canonical MAE degradation is no greater than +0.01 arcsec. If no nuisance is shown to matter, do not run an arbitrary mitigation. |
| H6 | Hybrid use: compare default, documented heuristic, and CNN-predicted initializations for the same bounded classical SIE optimizer on canonical and selected mismatch cases. | Relative change in objective evaluations; wall time is secondary. Final `theta_E` error and convergence are quality constraints. | A useful initialization reduces objective evaluations by at least 20% with an interval supporting a reduction, while the upper interval for final MAE degradation is no greater than +0.01 arcsec and convergence success is not lower by more than 5 percentage points. |

If a support rule is not met, retain the measured effect and classify it as practically small only when its full interval lies inside the predeclared +/-0.01 arcsec region; otherwise report it as inconclusive. Report all subgroup and tail results even when the aggregate rule is not met.

## Metrics and analysis

Primary metric: MAE in arcseconds. For every primary comparison also report RMSE, median absolute error, median and mean relative error, bias, P90/P95 absolute error, prediction-versus-truth slope/intercept, fraction outside the training target range, and paired effect. Report errors and bias in fixed radius bins `[0.5, 1.0)`, `[1.0, 1.5)`, and `[1.5, 2.0]` arcsec, plus multiplicity/SNR/source/PSF subgroups when defined. Include parent count, descendant count, seed count, runtime, and parameter count where applicable.

Bootstrap over independent parent systems, not descendant images. Keep training-seed variation separate from parent-level uncertainty. Use paired comparisons for counterfactuals and describe effect sizes before significance language. Exploratory multi-factor results must be labeled exploratory.

## Ordered execution timeline

This is the source plan's relative 12-week order. Weeks are milestones rather than calendar deadlines; scientific gates take precedence. Do not start a later phase while its prerequisites are unresolved.

1. **Week 1 — Freeze research design:** audit artifacts, quarantine the opened test split, record tests and validations, establish the research branch and study documents, freeze H1-H6 and metrics, create the registry and decision log. **Milestone:** reproducible starting point and frozen protocol.
2. **Week 2 — Counterfactual infrastructure:** canonical parent state, rerender mechanism, ancestry tracking, deterministic and label-invariance checks, manifests/previews, smoke benchmark. **Milestone:** controlled rerenders preserve physical truth and split isolation.
3. **Week 3 — Pixel scale:** evaluate frozen CNN and classical method; test the declared normalization/metadata options. **Milestone:** measured scale sensitivity and a validation-only decision.
4. **Week 4 — Observation systematics:** PSF, noise/SNR, lens light, and normalization audits. **Milestone:** paired effect tables and plots.
5. **Week 5 — Source/physics mismatch:** source morphology/position, multiplicity, shear, and a limited mass-family shift if validated. **Milestone:** ranked failure mechanisms.
6. **Week 6 — Baselines and shortcuts:** simple baselines, learning curve, interventions, and classical/CNN comparison. **Milestone:** evidence on data, representation, and distribution limits.
7. **Week 7 — Cross-simulator:** validate conventions, build a matched benchmark, and test transfer. **Milestone:** bounded estimate of simulator dependence.
8. **Week 8 — Mitigation:** randomize only measured high-impact nuisances; compare with the fixed baseline. **Milestone:** measured improvement and remaining limits.
9. **Week 9 — Hybrid classical/ML:** compare initializations on canonical and selected shifted conditions. **Milestone:** determine the useful role of ML in physical fitting.
10. **Week 10 — Real-domain gap:** compare supported CASTLES and simulated observables and document unsupported regimes. **Milestone:** bounded domain-coverage statement.
11. **Week 11 — Fresh blind holdout:** freeze method/protocol, create independent parents, validate hashes and ancestry, evaluate once. **Milestone:** final unbiased synthetic evidence.
12. **Week 12 — Analysis/manuscript:** paired analysis, figures, failure atlas, model/research card, claim-evidence map, limitations, manuscript and reproducibility package. **Milestone:** auditable paper package.

The source plan prioritizes protocol/test quarantine, counterfactual infrastructure, pixel scale, PSF, source morphology, classical comparison, learning curve, targeted mitigation, fresh holdout, CASTLES gap, then manuscript. Cross-simulator work, uncertainty modeling, extra architectures, and hybrid optimization must not displace this core order.

## Changes after freeze

Any change to hypotheses, thresholds, primary metrics, split policy, allowed inputs, or holdout policy must be recorded as a new protocol version before looking at the affected experiment's predictions. Record the reason and affected registry rows in `decision_log.md`.
