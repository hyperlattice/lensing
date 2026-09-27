# Reliability study protocol v1.11 — Week 11 blind synthetic holdout

**Status:** frozen before independent holdout system generation  
**Date:** 2026-09-26  
**Base protocol:** [`protocol_v1.md`](protocol_v1.md)  
**Prior amendments:** [`protocol_v1.1.md`](protocol_v1.1.md) through [`protocol_v1.10.md`](protocol_v1.10.md)

## Frozen model choice

Use the Week 8 randomized CNN checkpoints from seeds 12001, 12002, and 12003 as the final model. The prediction for a parent-condition is the arithmetic mean of the three fixed checkpoint predictions, with each checkpoint's own frozen training-only normalization. Use the original canonical-trained checkpoints from the same three seeds as a fixed development comparator; they do not select the final model and cannot be retuned. Before holdout generation, record SHA-256 hashes for all six checkpoints, normalization files, this protocol, config, model/dataset loader code, and the holdout evaluation script.

The Week 8 randomized ensemble is selected because its preregistered validation mitigation rule passed. Week 9 found no consistent point-fit initializer speedup, and Week 10 found material real-domain measurement and simulator gaps. Neither phase changes the frozen synthetic model. No model, checkpoint, preprocessing, or hyperparameter selection is allowed after holdout scoring.

## Independent holdout release

- Generate **150 new parent systems** from the same `configs/simulation_engineering_v1.yaml` and `draw_observation_parameters` prior used for the original study, using PCG64 seed **11031**. The generator derives each fresh parameter/noise seed from `SeedSequence([11031, parent_index])`.
- Assign new root IDs under `reliability-blind-holdout-v1`; check IDs and parameter/noise seeds against prior development train/validation and Week 7/8 artifacts. The root-ID namespace and fresh generation seed are new. Do not inspect the historical test manifest or products; it remains quarantined.
- Generate one canonical observation and four matched counterfactual observations per parent: pixel scale 0.04 arcsec/pixel, pixel scale 0.06 arcsec/pixel, half exposure, and double read noise. Pair pixel scales with 320×320 and 214×214 native shapes to retain the established center-to-center FOV. Keep nuisance interventions isolated, parent physical state fixed, and condition/noise derivation deterministic from the fresh parent seed.
- The release is materialized as one `holdout` split. Any `validation` split label used internally by the counterfactual renderer is a temporary generator selector only; the final release manifest and the sole root split file use `holdout`.
- The release contains 750 observations but only 150 independent parent systems. The holdout contains no training or validation split and is not merged into any prior release.
- Validate schemas, image products, condition counts, parent ancestry, parameter-seed uniqueness, and all input/output hashes before model inference. Do not open theta_E truth values during prediction generation.

## Exactly-once evaluation

After release and protocol hashes are validated, run both frozen ensembles once on the 750 observations. Save per-checkpoint and ensemble predictions, with sample ID, parent ID, condition, model, checkpoint hashes, prediction, and preprocessing hash, to `holdout_predictions_before_scoring.csv`; hash the file. The script must refuse a second scored run when its completion marker exists. Only after that target-free file is closed and hashed may the evaluator join holdout `theta_E` labels once and write scored results.

Write an inference-start marker immediately before the first model forward pass and refuse any further inference if that marker or any prediction output exists, including after an interrupted run. Write the scored completion marker after the single label join and scoring pass.

Do not tune from holdout results, rerun with changed subsets, or inspect a heldout failure and then choose a new configuration. If the result is unfavorable or a release/prediction check fails, record it as observed; any corrected evaluation must be explicitly classified as a new experiment and cannot replace this holdout result.

## Frozen analysis

Report ensemble and seed-specific MAE, RMSE, bias, median/P90/P95 absolute error, target-range coverage, and parent-level paired error changes for all five conditions. Report condition-specific and equal-weight four-stress composite differences between randomized and canonical-trained ensembles. Use 10,000 parent bootstrap resamples with seed 6031. Include target-radius bins, counts above the descriptive absolute-error thresholds 0.05 and 0.10 arcsec, sample/parent/seed counts, and prediction-shift summaries. Non-finite predictions are a release failure and must be reported without target joining. No adaptive thresholds or post-hoc conditions are allowed.

The output directory is `results/reports/reliability_v1/blind_holdout_week11/`. Retain release manifests, exact-once marker, hashes, target-free predictions, joined score table, per-condition metrics, paired effects, plots, summary JSON, and limitations report.

## Claim boundary

This is the final independent **synthetic** holdout for the tested simulator and fixed nuisance conditions. It cannot establish alternate-renderer agreement beyond Week 7's limited equation comparison, simulation-to-CASTLES transfer, calibrated HST noise, or real-observation theta_E accuracy. The historical synthetic test split remains quarantined.
