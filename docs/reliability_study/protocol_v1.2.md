# Reliability study protocol v1.2 — scale-aware mitigation amendment

**Status:** frozen before mitigation model training  
**Date:** 2026-09-25  
**Base protocol:** [`protocol_v1.md`](protocol_v1.md)  
**Pixel-scale design:** [`protocol_v1.1.md`](protocol_v1.1.md)

This amendment fixes training and selection details for the two retrained H1 mitigations. It does not change the v1 hypotheses, effect thresholds, or the 12-week order.

## Shared training data and selection

- Use the same 700 training parent identities: one canonical 0.05 arcsec/pixel image and one matched 0.04 arcsec/pixel descendant per parent. The 0.06 arcsec/pixel condition is validation-only extrapolation.
- Evaluate each epoch on the fixed 150 validation parents at 0.04, 0.05, and 0.06 arcsec/pixel. Select each seed's checkpoint by the lowest equal-weight mean of the three stratum MAEs in arcseconds. Each stratum has 150 independent parents.
- Keep the v1 CNN feature widths and regression head, AdamW optimizer (learning rate 0.001, weight decay 0.0001), Huber loss (delta 1.0), batch size 16, 20-epoch maximum, patience 6, and seeds 12001-12003. Batches are grouped by image shape.
- Reuse the baseline's train-only image normalization statistics. Compute each candidate's target mean and standard deviation from the 1,400 training records only. The pixel-target model trains on `theta_E_arcsec / pixel_scale_arcsec` and converts predictions back by multiplying by the record's pixel scale. The metadata model predicts `theta_E_arcsec` and receives only `pixel_scale_arcsec / 0.05` as a scalar input.
- Do not access the historical test split. The validation results select checkpoints and therefore remain development evidence, not an independent final estimate.

## Utility rule and reporting

Compare each selected mitigation to the frozen baseline on the same parent and scale. A mitigation is useful only if it lowers MAE at both shifted scales and its canonical MAE degradation is no more than 0.01 arcsec. Report per-seed metrics and paired parent-bootstrap 95% intervals (10,000 resamples, seed 6031); average seed effects within each parent before resampling. Include whether the utility rule passes, along with all three scale strata and checkpoint hashes. The existing H1 support and practically-small rules are unchanged.

The candidate training release and predictions remain synthetic. Any success supports only this simulator's controlled pixel-sampling intervention; it does not establish different-instrument or real-HST accuracy.
