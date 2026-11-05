# Reliability study protocol v1.6 — Week 6 baselines and shortcut checks

**Status:** frozen before Week 6 model predictions  
**Date:** 2026-09-26  
**Base protocol:** [`protocol_v1.md`](protocol_v1.md)  
**Prior completed amendments:** [`protocol_v1.1.md`](protocol_v1.1.md) through [`protocol_v1.5.md`](protocol_v1.5.md)

This amendment predeclares simple baselines, an independent-parent learning curve, selected shortcut interventions, and their comparisons. It uses only train/validation development parents; the historical test split remains prohibited.

## Validation population and comparison systems

- Use the original 700-parent train / 150-parent validation split in `sim-eng1000-v1` for the canonical baseline.
- Evaluate matched robustness conditions on the same 150 validation parents: pixel scale 0.04 and 0.06 arcsec/pixel (Week 3); half exposure and double read noise (Week 4); source x+0.15 arcsec and shear `gamma2=0.05` (Week 5).
- Report the frozen three-seed CNN, training-mean predictor, fixed-feature ridge predictor, and v1.3 observation-only SIS/SIE point-image geometry baseline. For existing classical outputs, retain their reported fallback-inclusive operational metric.
- Do not use `fit_sie_subset` as a blind fit: inspection shows it reads parameter sidecars and initializes mass/source parameters from truth. The available truth-free classical estimator is the v1.3 point-image geometry method; label it as such and retain the full pixel-likelihood limitation.

## Fixed-feature ridge baseline

Build image-only features from the processed science image and released pixel scale, with no labels, parameter sidecars, diagnostics, seeds, or ancestry IDs as inputs:

- 32 equal-width radial positive-flux fractions from 0 to 3.0 arcsec around the geometric image center;
- 8 positive-flux sector fractions over radii 0.7–3.0 arcsec;
- log positive total flux, robust image RMS, maximum positive pixel, central-to-outer flux ratio, and positive-flux centroid radius.

Standardize features from the 700-parent training split only. Select Ridge `alpha` from `[0.01,0.1,1,10,100]` using five deterministic folds of independent training parents (fold seed 6031), scored by fold MAE. Refit on all 700 train parents. Do not tune on validation. A training-mean baseline is computed from those same 700 labels.

## Learning curve

Train the frozen baseline CNN architecture and optimizer on nested independent-parent subsets of **100, 300, and 700** train parents. Create one deterministic permutation with seed **6031**; the first 100 are contained in the first 300, which are contained in all 700. Train each sample size with the established seeds `12001`, `12002`, and `12003`, the original 20-epoch maximum and patience 6. Keep the validation set fixed at all 150 parents. Epoch selection may use validation MAE as in the original training protocol. Report parent-bootstrap intervals separately from seed variation.

## Frozen CNN image interventions

Apply each transform directly to the canonical validation science image before the checkpoint's frozen train-only global normalization. Keep the target and parent fixed and use the same 150 paired systems:

1. `rotate_ccw_90`: rotate the image 90 degrees counter-clockwise;
2. `brightness_half`: multiply image values by 0.5;
3. `brightness_double`: multiply image values by 2.0;
4. `central_disk_zero`: zero pixels within 0.50 arcsec of the geometric image center;
5. `arc_annulus_zero`: zero pixels at radii 0.70–2.20 arcsec from the geometric center.

These are controlled information-ablation/representation tests, not physically rendered observations. The central disk and annulus are fixed in image coordinates and are not truth-derived lens/source masks. Source and PSF swaps are already covered by Weeks 3–5. Attribution maps are not required and will not be interpreted causally.

## Analysis and outputs

- Report baseline MAE, RMSE, median absolute error, bias, P90/P95, fixed radius bins, training size, seed, parameter count, and runtime where applicable.
- For each image intervention, report paired MAE change, parent-bootstrap 95% interval (10,000 resamples, seed 6031), and absolute prediction shift; average seed-specific effects within parent before resampling.
- Compare the learned and non-neural systems on canonical and all seven preselected validation conditions where each applies. Report all classical fallbacks/failures.
- Save the nested parent-selection manifest, fold/alpha records, run configs, predictions, per-seed metrics, paired effects, learning-curve and shortcut plots, and a `shortcut_audit.md` under `results/reports/reliability_v1/baselines_week6/`.
- Use the 0.01 arcsec practical-effect rule from v1.0. The shortcut matrix and subgroup comparisons are descriptive and not multiple-comparison adjusted.

No architecture challenger is included; previous phases identify data-distribution sensitivities, while the current fixed model suffices to answer this phase's baseline and sample-count questions.
