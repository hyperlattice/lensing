# theta-e-huber-delta-025-v1

SMOKE CHECK ONLY: these metrics are not a development benchmark.

Each seed uses 32 training and 16 validation samples. Test images accessed: 0.

| Seed | Best epoch | MAE (arcsec) | RMSE (arcsec) | Median relative error | Bias (arcsec) | R2 | P90 (arcsec) | P95 (arcsec) | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12001 | 1 | 0.3885 | 0.4639 | 27.95% | -0.1340 | -0.0561 | 0.6671 | 0.7079 | 3.3 |
| 12002 | 2 | 0.3910 | 0.4548 | 24.75% | -0.0497 | -0.0153 | 0.7270 | 0.7413 | 1.3 |
| 12003 | 2 | 0.3862 | 0.4538 | 26.31% | -0.1124 | -0.0108 | 0.6569 | 0.6910 | 1.4 |

## Across-seed metrics

| Metric | Mean | Sample standard deviation |
|---|---:|---:|
| bias_arcsec | -0.098710 | 0.043821 |
| fraction_within_005_arcsec | 0.083333 | 0.036084 |
| fraction_within_010_arcsec | 0.145833 | 0.036084 |
| mae_arcsec | 0.388564 | 0.002406 |
| mean_relative_error | 0.363890 | 0.017503 |
| median_absolute_error_arcsec | 0.392009 | 0.042750 |
| median_relative_error | 0.263363 | 0.016043 |
| p90_absolute_error_arcsec | 0.683671 | 0.037857 |
| p95_absolute_error_arcsec | 0.713396 | 0.025605 |
| prediction_vs_truth_intercept_arcsec | 1.175804 | 0.059170 |
| prediction_vs_truth_slope | 0.014837 | 0.013604 |
| r2 | -0.027387 | 0.024942 |
| rmse_arcsec | 0.457504 | 0.005535 |

Arithmetic mean and sample standard deviation (ddof=1) across seeds; not a confidence interval or ensemble evaluation.

## Radius-dependent errors

Bins are [lower, upper), except the last bounded bin includes its upper edge. Outside-range groups and empty bins remain in the CSV/JSON.

| Radius bin (arcsec) | Samples per seed | Mean MAE | Mean bias | Mean P95 |
|---|---:|---:|---:|---:|
| 0.5 to 1.0 | 3 | 0.5455 | +0.5455 | 0.6741 |
| 1.0 to 1.5 | 6 | 0.1263 | +0.1011 | 0.2033 |
| 1.5 to 2.0 | 7 | 0.5461 | -0.5461 | 0.7349 |

## Interpretation

MAE is the average angular miss. P90/P95 are error thresholds containing 90%/95% of validation cases. Bias is prediction minus truth. Seed variation measures sensitivity to initialization, dropout, and training order on the same fixed data split.

Compare future candidates using the same seeds and report paired changes. Improvements smaller than the baseline seed variation are inconclusive; three seeds do not establish statistical significance. Inspect P95 and radius-bin bias before accepting a lower mean MAE. These are individual-model metrics, not predictions from an ensemble.

The checkpoint for each seed is selected by validation MAE. Do not pick a lucky seed as the headline result or use existing test metrics to choose models. A final challenger requires the independent holdout described in the planning notebook.

Open `seed_comparison.png` for errors and radius-bin bias, `per_seed_metrics.csv` for per-run results, and `aggregate_validation.json` for hashes, configurations, environments, and exact metrics.

Synthetic validation only; test manifests hashed for identity, zero test images loaded. Smoke runs are wiring checks only.
