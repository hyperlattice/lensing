# Data and output contracts

Source banks retain the `observation_bank_v1` JSON schema and their existing
release-relative products. Source images are unnormalized float32 rates in
electrons/second. Expected-rate arrays and source labels are simulator/training
inputs, never predictor features. Parent identities, latent fingerprints,
parameter/noise seeds, roles and exclusion ledgers enforce split independence.
Only `source_train` supplies normalization statistics.

The primary holdout retains its CSV columns and release-relative image paths.
Prediction reads only sample ID, parent ID, condition and image path. It writes
`predictions_before_scoring.csv` and `predictions_complete.json` before analysis
joins `theta_E_arcsec` truth. The fixed primary cohort has 750 views of 150 parents.
The output remains `results/ml/equal-budget-final-v1/`, including `fits/<arm>/seed-<seed>/`,
`final_report.json`, `result_overview.md` and existing figure filenames.

Paired releases retain native image, variance, PSF and mask products, parent,
renderer and condition axes, and an isolated truth vault. Prediction does not
open the truth vault. Secondary outputs remain under
`results/ml/cross-simulator-rapid-v2/`.

Historical result schemas retain zero-valued MMD/access fields for compatibility;
MMD training is no longer supported by the maintained source-only trainer.
Existing outputs are never retroactively given new provenance or qualification.
