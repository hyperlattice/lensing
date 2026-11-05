# Reliability study protocol v1.8 — Week 8 targeted nuisance randomization

**Status:** frozen before Week 8 augmented training data generation and model fitting  
**Date:** 2026-09-26  
**Base protocol:** [`protocol_v1.md`](protocol_v1.md)  
**Prior completed amendments:** [`protocol_v1.1.md`](protocol_v1.1.md) through [`protocol_v1.7.md`](protocol_v1.7.md)

## Evidence motivating the intervention

Use the frozen Week 3–7 findings. Pixel scales 0.04 and 0.06 arcsec/pixel increased MAE by +0.0707 and +0.0489 arcsec; half exposure increased MAE by +0.0463; doubled read noise increased MAE by +0.0350. The Week 7 equation comparison showed no material implementation shift and does not motivate a training change.

## Single domain-randomized training proposal

- Keep all 700 original training parents and their canonical images.
- Add exactly one randomized observation descendant per training parent. Use PCG64 with seed **6031** and independent uniform draws from:
  - pixel scale and native shape: **0.04 arcsec/pixel at 320×320**, **0.05 at 256×256**, or **0.06 at 214×214**. This keeps the center-to-center angular field of view matched to Week 3; the existing preprocessing center-crops or pads to 256×256.
  - exposure multiplier: **0.5 or 1.0**;
  - read-noise multiplier: **1.0 or 2.0**.
- Keep the original parent’s mass truth, source, lens light, PSF FWHM, background rate, and noise seed. The randomized view's native shape is paired with its scale as specified above. The same parent may appear in canonical and randomized form only in the training split. Do not treat descendants as independent parent systems.
- The resulting training manifest has 1,400 rows from 700 independent parent systems. Training-only normalization is recomputed over those rows.
- Preserve the native raster shapes and bucket training batches by shape; do not resize the randomized views to the canonical pixel grid before the CNN.
- Keep all 150 validation parents canonical for early stopping. The established CNN architecture, Huber loss, optimizer, epoch limit, patience, batch size, and seeds **12001, 12002, 12003** remain fixed.

## Frozen evaluation

After fitting, evaluate each randomized-training checkpoint once on the fixed 150-parent validation conditions: canonical, pixel scale 0.04, pixel scale 0.06, half exposure, and doubled read noise. Pair each checkpoint seed with its canonical-trained baseline seed. Do not use stress-condition results to select epochs, seeds, or other configurations.

Save all predictions before joining validation labels. For each condition report seed-specific metrics and parent-level paired error change, averaging the seed-specific effect within parent before 10,000 bootstrap resamples (seed **6031**). Report mean prediction shift, RMSE, bias, P90/P95 absolute error, and theta_E bins. The primary mitigation effect is the within-parent mean error change averaged equally across the four stress conditions.

## Decision rules

- The targeted mitigation passes if its average stress-condition MAE change is at most **−0.01 arcsec** and the parent-bootstrap 95% upper bound is below zero.
- Canonical performance is non-inferior if the parent-bootstrap 95% upper bound on canonical MAE degradation is at most **+0.01 arcsec**.
- Report each stress condition separately even if the composite rule passes. No claim extends beyond these tested conditions.
- If the randomized model fails the primary rule, run one predeclared diagnostic extension only: add pixel-scale as an explicit scalar model input while keeping the CNN trunk and all other settings fixed. Use the same parent set and seeds. Do not conduct broader architecture search.

## Integrity and claim limits

The historical test split and real-data labels remain prohibited. Preserve parent ancestry and train/validation separation. Report the number of unique training parents separately from the 1,400 training rows. Week 8 can show whether randomization over the tested pixel-scale, exposure, and read-noise ranges improves synthetic robustness; it cannot show that domain randomization solves simulation-to-real shift.
