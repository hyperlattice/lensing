# Week 3 — Pixel-scale study status and next steps

**Status:** complete with a truth-free point-image geometry control; full pixel-likelihood fitting remains outside this comparison.  
**Date:** 2026-09-26  
**Reference:** `gravitational_lensing_field_analysis_and_contribution_plan (1).md`  
**Protocols:** [`protocol_v1.md`](protocol_v1.md), [`protocol_v1.1.md`](protocol_v1.1.md), [`protocol_v1.2.md`](protocol_v1.2.md), [`protocol_v1.3.md`](protocol_v1.3.md)

## Work completed

- Generated matched validation renders for 150 independent parents at 0.04, 0.05, and 0.06 arcsec/pixel. Shapes are 320×320, 256×256, and 214×214, preserving the center-to-center angular field span within the frozen tolerance.
- Evaluated all three frozen CNN seeds on native images and after angular resampling to 0.05 arcsec/pixel.
- Trained the predeclared pixel-space-target and metadata-conditioned alternatives on 700 training parents at 0.05 plus matched 0.04 arcsec/pixel descendants. Each model used seeds 12001–12003 and selected epochs by the frozen equal-weight mean validation MAE across the three scale strata.
- Compared both alternatives with the frozen baseline using paired parent-level bootstrap intervals. The 150 validation parents selected epochs and remain development evidence.
- Evaluated a predeclared, image-derived point-image geometry comparator on the same 150 validation parents at all three scales. It reads observed science, variance, masks, and the released PSF; it does not read parameter sidecars or saved image-position diagnostics. Validation labels were loaded only after predictions were written.
- The counterfactual train release passed target invariance, ancestry/split isolation, zero-rejection, and strict-loader checks. The two validation descendants in that release were only for loader integrity; training and model selection used the frozen H1 validation release.
- The full repository suite passed: **106 tests**. The historical test split was not accessed by these experiments.

## Results

| Model/input | 0.05 canonical MAE | 0.04 MAE | 0.06 MAE |
|---|---:|---:|---:|
| Frozen baseline, native | 0.0921 | 0.1628 | 0.1410 |
| Frozen baseline, angularly resampled | 0.0921 | 0.0991 | 0.0878 |
| Pixel-space target | 0.0780 | 0.0893 | 0.1957 |
| Pixel-scale metadata | 0.0789 | 0.0919 | 0.1619 |

For the frozen CNN, the paired MAE increase versus canonical was **+0.0707 arcsec** at 0.04 (95% parent-bootstrap interval **[+0.0519, +0.0900]**) and **+0.0489 arcsec** at 0.06 (**[+0.0326, +0.0647]**). Both exceed the 0.01 arcsec practical threshold, with intervals above zero.

Angular resampling reduced the raw-scale penalties. Its paired effects were **+0.0070** at 0.04 (**[−0.0042, +0.0186]**) and **−0.0043** at 0.06 (**[−0.0143, +0.0053]**). Neither interval is wholly inside the predeclared ±0.01 arcsec practical-small region, so this mitigation is **inconclusive**, not evidence that the scale effect is practically zero.

Compared with the frozen baseline, the pixel-space-target model changed MAE by **−0.0141** at canonical (**[−0.0224, −0.0061]**), **−0.0734** at 0.04 (**[−0.0913, −0.0566]**), and **+0.0547** at 0.06 (**[+0.0331, +0.0764]**). The metadata model changed MAE by **−0.0132** at canonical (**[−0.0213, −0.0052]**), **−0.0709** at 0.04 (**[−0.0881, −0.0541]**), and **+0.0209** at 0.06 (**[+0.0094, +0.0322]**). Each interval resamples independent parents after averaging the paired effects across the three seeds.

Neither retrained mitigation passes the frozen H1 utility rule: each lowers error at 0.04 and preserves or improves canonical MAE, but each raises error at 0.06 versus the frozen baseline. The observed 0.06 penalty is also present relative to each candidate's own canonical result. These validation-selected results do not establish performance on an independent holdout, a different instrument, or real HST data.

### Classical point-image geometry control

| Scale | Detected and fit | Fallbacks | All-parent classical MAE | Fit-only MAE | CNN MAE (three-seed parent average) | Classical minus CNN MAE (95% parent interval) |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 canonical | 106/150 | 44 | 0.2007 | 0.0698 | 0.0921 | +0.1086 [+0.0724, +0.1469] |
| 0.04 | 100/150 | 50 | 0.2168 | 0.0688 | 0.1628 | +0.0540 [+0.0173, +0.0905] |
| 0.06 | 105/150 | 45 | 0.1944 | 0.0587 | 0.1410 | +0.0534 [+0.0216, +0.0862] |

All-parent MAE includes the predeclared training-mean fallback, so it is the operational comparison for this fixed detector-and-fit pipeline. The lower fit-only MAE is conditional on successful point-image detection and fitting; it is subject to selection and is not directly comparable as an all-parent score. About one-third of parents fell back at each scale. The classical paired scale effects were **+0.0161 arcsec** at 0.04 (**[−0.0083, +0.0414]**) and **−0.0063 arcsec** at 0.06 (**[−0.0348, +0.0219]**); both are inconclusive under the practical-small rule.

This comparator is an analytic point-image SIS/SIE geometry control, not a full pixel-likelihood SIE fit. It assumes an image-estimated center and uses a point-source lens equation; its fallback rate and model assumptions limit the conclusion. A broader classical pixel-likelihood comparison remains for the Week 6 baseline work.

Detailed outputs are in `results/reports/reliability_v1/pixel_scale/`: `frozen_baseline/`, `angular_resampling/`, `mitigations/`, and `classical_geometric/`. Candidate checkpoints, histories, and per-seed validation predictions are in `results/ml/reliability_pixel_scale_v1/`.

## Week 3 decision

The Week 3 pixel-scale milestone is closed with an explicitly limited, truth-free classical control. The frozen CNN has a supported practically important scale penalty at both shifts. Angular resampling remains inconclusive; pixel-target and metadata conditioning fail the mitigation utility rule because of the 0.06 result. The point-image geometry pipeline has a substantial fallback rate and is weaker than the CNN when scored on all parents. Do not generalize these synthetic validation results to real instruments or treat the point-image control as a full pixel-likelihood fit. The historical test remains quarantined.

## Ordered next work

1. **Week 4 — Observation systematics:** audit PSF width/family, noise/SNR with repeated noise realizations, lens light, and normalization using matched parents and paired parent-level analysis.
2. **Week 5 — Source/physics mismatch:** source morphology and position, multiplicity, shear, then a limited mass-family shift only if its renderer and labels are validated.
3. **Week 6 — Baselines and shortcuts:** simple baselines, learning curve, interventions, and broader classical/CNN comparisons, including a validated pixel-likelihood classical fit if available.
4. **Week 7 — Cross-simulator:** validate conventions, create a matched benchmark, and measure bounded transfer.
5. **Week 8 — Targeted mitigation:** randomize only nuisance factors established as high impact by prior gates.
6. **Week 9 — Hybrid classical/ML:** compare initializations on canonical and selected shifted conditions.
7. **Week 10 — Real-domain gap:** compare only supported CASTLES and simulated observables; document unsupported regimes.
8. **Week 11 — Fresh blind holdout:** freeze the method first, then create and evaluate independent parents once.
9. **Week 12 — Analysis/manuscript:** prepare paired analysis, figures, failure atlas, claim-evidence map, limitations, and reproducibility package.

Week 4 is next. The later weeks remain in the reference's exact order; they are not completed by this status report.
