# Week 7 — Cross-simulator study: next-phase brief

**Status:** ready to start after Week 6 close (D-017)  
**Source:** Phase 8 / Week 7 in the field-analysis plan  
**Week 6 result:** [`week6_baselines_shortcuts_status.md`](week6_baselines_shortcuts_status.md)

## Goal

Estimate dependence on the current simulation implementation independently of real-data complications. The existing development release identifies Lenstronomy 1.14.2 as its simulator. An alternate implementation has not yet been selected; renderer choice and convention checks are the first Week 7 tasks.

## Ordered work

1. Inventory the local environment and available packages for a feasible alternate lens renderer or an independently implemented lens-equation path. Record version, source, and installation constraints before selecting it.
2. Write a convention map for coordinates and pixel centers, angular units, SIE ellipticity/axis ratio, shear, source light, PSF convolution, flux normalization, and noise. Resolve or explicitly bound each mismatch.
3. Validate SIS and SIE controls between implementations before comparing CNN predictions. Compare image positions, Einstein-radius convention, and rendered image structure under controlled cases. Do not score the frozen CNN if the physical controls fail.
4. Freeze the matched parent set, sample size, nuisance-matched conditions, and any nuisance-mismatched condition in a new protocol amendment before generating evaluation predictions.
5. Build the alternate-renderer release with ancestry and provenance. Use new render seeds and do not mix training and validation parents.
6. Evaluate the frozen Week 1 CNN checkpoints on the alternate-renderer images without retraining or selecting a model from these results.
7. Compare within-simulator validation and alternate-renderer outputs. Record image-statistic differences and distinguish convention errors from observable rendering differences where the controls allow it.

## Gates and reporting

- Keep the historical test split quarantined. This phase uses development validation evidence only.
- Do not tune the CNN or its preprocessing against the alternate-renderer results.
- Preserve failed controls and renderer limitations in the report.
- Do not claim simulator-independent inference unless the matched cross-simulator evidence supports that claim.
- Close the phase with `week7_cross_simulator_status.md`, machine-readable predictions and control comparisons, a decision-log entry, and updated registry status before starting Week 8 mitigation.

The alternate renderer and sample size remain unfrozen until the local compatibility audit is complete. A new protocol version must record them before Week 7 evaluation predictions.
