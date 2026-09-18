# theta-e-huber-delta-025-v1

Baseline stability across fixed random seeds, using validation only.

Each seed uses 700 training and 150 validation samples. Test images accessed: 0.

| Seed | Best epoch | MAE (arcsec) | RMSE (arcsec) | Median relative error | Bias (arcsec) | R2 | P90 (arcsec) | P95 (arcsec) | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12001 | 18 | 0.0856 | 0.1186 | 4.80% | +0.0170 | 0.9256 | 0.1711 | 0.2245 | 282.6 |
| 12002 | 15 | 0.0811 | 0.1092 | 5.32% | -0.0135 | 0.9370 | 0.1768 | 0.2134 | 252.1 |
| 12003 | 19 | 0.0893 | 0.1143 | 5.89% | +0.0113 | 0.9309 | 0.1878 | 0.2187 | 167.5 |

## Across-seed metrics

| Metric | Mean | Sample standard deviation |
|---|---:|---:|
| bias_arcsec | 0.004908 | 0.016221 |
| fraction_within_005_arcsec | 0.388889 | 0.066778 |
| fraction_within_010_arcsec | 0.688889 | 0.044389 |
| mae_arcsec | 0.085348 | 0.004093 |
| mean_relative_error | 0.083279 | 0.004734 |
| median_absolute_error_arcsec | 0.062421 | 0.002836 |
| median_relative_error | 0.053369 | 0.005446 |
| p90_absolute_error_arcsec | 0.178585 | 0.008481 |
| p95_absolute_error_arcsec | 0.218873 | 0.005573 |
| prediction_vs_truth_intercept_arcsec | 0.118301 | 0.036517 |
| prediction_vs_truth_slope | 0.908653 | 0.016357 |
| r2 | 0.931176 | 0.005699 |
| rmse_arcsec | 0.114025 | 0.004735 |

Arithmetic mean and sample standard deviation (ddof=1) across seeds; not a confidence interval or ensemble evaluation.

## Radius-dependent errors

Bins are [lower, upper), except the last bounded bin includes its upper edge. Outside-range groups and empty bins remain in the CSV/JSON.

| Radius bin (arcsec) | Samples per seed | Mean MAE | Mean bias | Mean P95 |
|---|---:|---:|---:|---:|
| 0.5 to 1.0 | 51 | 0.0959 | +0.0492 | 0.2349 |
| 1.0 to 1.5 | 47 | 0.0870 | +0.0039 | 0.2193 |
| 1.5 to 2.0 | 52 | 0.0734 | -0.0376 | 0.1832 |

## Interpretation

MAE is the average angular miss. P90/P95 are error thresholds containing 90%/95% of validation cases. Bias is prediction minus truth. Seed variation measures sensitivity to initialization, dropout, and training order on the same fixed data split.

Compare future candidates using the same seeds and report paired changes. Improvements smaller than the baseline seed variation are inconclusive; three seeds do not establish statistical significance. Inspect P95 and radius-bin bias before accepting a lower mean MAE. These are individual-model metrics, not predictions from an ensemble.

The checkpoint for each seed is selected by validation MAE. Do not pick a lucky seed as the headline result or use existing test metrics to choose models. A final challenger requires the independent holdout described in the planning notebook.

Open `seed_comparison.png` for errors and radius-bin bias, `per_seed_metrics.csv` for per-run results, and `aggregate_validation.json` for hashes, configurations, environments, and exact metrics.

In-domain synthetic validation; test manifests hashed for identity, zero test images loaded.
