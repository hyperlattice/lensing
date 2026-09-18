# theta-e-huber-delta-050-v1

Baseline stability across fixed random seeds, using validation only.

Each seed uses 700 training and 150 validation samples. Test images accessed: 0.

| Seed | Best epoch | MAE (arcsec) | RMSE (arcsec) | Median relative error | Bias (arcsec) | R2 | P90 (arcsec) | P95 (arcsec) | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12001 | 19 | 0.0871 | 0.1161 | 4.68% | -0.0008 | 0.9287 | 0.1964 | 0.2357 | 238.1 |
| 12002 | 16 | 0.0852 | 0.1139 | 5.32% | +0.0013 | 0.9314 | 0.1853 | 0.2250 | 263.2 |
| 12003 | 12 | 0.0968 | 0.1305 | 6.46% | +0.0082 | 0.9100 | 0.1992 | 0.2404 | 213.6 |

## Across-seed metrics

| Metric | Mean | Sample standard deviation |
|---|---:|---:|
| bias_arcsec | 0.002888 | 0.004732 |
| fraction_within_005_arcsec | 0.386667 | 0.063596 |
| fraction_within_010_arcsec | 0.662222 | 0.003849 |
| mae_arcsec | 0.089680 | 0.006200 |
| mean_relative_error | 0.086933 | 0.006280 |
| median_absolute_error_arcsec | 0.070754 | 0.006426 |
| median_relative_error | 0.054847 | 0.008988 |
| p90_absolute_error_arcsec | 0.193642 | 0.007344 |
| p95_absolute_error_arcsec | 0.233704 | 0.007868 |
| prediction_vs_truth_intercept_arcsec | 0.129162 | 0.029137 |
| prediction_vs_truth_slope | 0.898276 | 0.020727 |
| r2 | 0.923352 | 0.011657 |
| rmse_arcsec | 0.120177 | 0.008992 |

Arithmetic mean and sample standard deviation (ddof=1) across seeds; not a confidence interval or ensemble evaluation.

## Radius-dependent errors

Bins are [lower, upper), except the last bounded bin includes its upper edge. Outside-range groups and empty bins remain in the CSV/JSON.

| Radius bin (arcsec) | Samples per seed | Mean MAE | Mean bias | Mean P95 |
|---|---:|---:|---:|---:|
| 0.5 to 1.0 | 51 | 0.0963 | +0.0529 | 0.2500 |
| 1.0 to 1.5 | 47 | 0.0951 | +0.0005 | 0.2277 |
| 1.5 to 2.0 | 52 | 0.0784 | -0.0440 | 0.1993 |

## Interpretation

MAE is the average angular miss. P90/P95 are error thresholds containing 90%/95% of validation cases. Bias is prediction minus truth. Seed variation measures sensitivity to initialization, dropout, and training order on the same fixed data split.

Compare future candidates using the same seeds and report paired changes. Improvements smaller than the baseline seed variation are inconclusive; three seeds do not establish statistical significance. Inspect P95 and radius-bin bias before accepting a lower mean MAE. These are individual-model metrics, not predictions from an ensemble.

The checkpoint for each seed is selected by validation MAE. Do not pick a lucky seed as the headline result or use existing test metrics to choose models. A final challenger requires the independent holdout described in the planning notebook.

Open `seed_comparison.png` for errors and radius-bin bias, `per_seed_metrics.csv` for per-run results, and `aggregate_validation.json` for hashes, configurations, environments, and exact metrics.

In-domain synthetic validation; test manifests hashed for identity, zero test images loaded.
