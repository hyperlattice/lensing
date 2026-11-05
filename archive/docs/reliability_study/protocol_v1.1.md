# Reliability study protocol v1.1 — pixel-scale audit amendment

**Status:** frozen before Week 3 model predictions  
**Date:** 2026-09-25  
**Base protocol:** [`protocol_v1.md`](protocol_v1.md)

This version preserves every v1 rule and threshold. It fixes the previously open H1 scale grid and field-of-view treatment before evaluation. It applies only to the Week 3 pixel-scale audit; the other hypotheses and timeline remain as recorded in v1.

## H1 intervention

- Compare the canonical 0.05 arcsec/pixel image against 0.04 and 0.06 arcsec/pixel rerenders of the same validation parents.
- Keep the center-to-center angular field of view within 0.03 arcsec of the canonical 12.75 arcsec span. Render 320 x 320 pixels at 0.04 arcsec/pixel and 214 x 214 pixels at 0.06 arcsec/pixel; the spans are 12.76 and 12.78 arcsec, respectively. The canonical image is 256 x 256 pixels.
- Hold lens truth, source parameters, angular PSF FWHM, exposure, background, read noise, and parent noise seed fixed. The angular PSF is therefore sampled at different pixel widths as a consequence of changing pixel scale.
- Use all 150 independent validation parents, with their canonical images and both matched scale descendants. Do not use any historical test parent or test result for decisions. No retraining is used for the initial frozen-model measurement.
- Score the existing three checkpoints (seeds 12001-12003) separately. Evaluate one scale stratum at a time so each batch contains a common image shape; the CNN's adaptive pooling accepts each of the declared dimensions.
- Treat this as a controlled sampling stress test under the current simulator. Holding the existing detector-noise parameters fixed does not make the alternate pixel scales calibrated models of a different instrument.

## Predeclared pixel-scale mitigation comparisons

If the frozen-model audit shows a scale penalty, compare these strategies on the same validation parents:

1. **Angular resampling:** resample each shifted-scale normalized image to the canonical 0.05 arcsec/pixel grid, then apply the unchanged checkpoints. Preserve the fixed angular field of view and report interpolation as part of this intervention.
2. **Pixel-space target:** train the same CNN to predict `theta_E_pixels = theta_E_arcsec / pixel_scale_arcsec` on scale-diverse training parents, then convert its output back to arcseconds using the declared pixel scale. Compute target normalization statistics on training parents only.
3. **Metadata-conditioned model:** train the same CNN with the single deployment-available scalar `pixel_scale_arcsec / 0.05` as an auxiliary input and predict `theta_E_arcsec`. No exact truth or nuisance sidecar fields are supplied.

For strategies 2 and 3, use the same scale-diverse training parents, fixed train/validation parent identities, architecture width, optimizer/loss schedule, and seeds 12001-12003. Keep all descendants of a parent in that parent's split. Report 0.05 arcsec/pixel canonical MAE and the paired MAE at each shifted scale; accept a strategy only under the v1 H1 rule. If the frozen model has no practically important scale penalty, report that result first and do not claim that a mitigation is needed.

## Reproducibility

Run parameters are in `configs/reliability_pixel_scale_v1.yaml`. The counterfactual manifest must retain the original parent IDs and split assignments. Variable-size arrays are loaded by scale-specific split manifests, never batched across different shapes. Record exact scale, shape, angular field span, checkpoint hashes, parent IDs, preprocessing, seed, and paired predictions.
