# theta-e-cnn-seeds-v1

Baseline stability across fixed random seeds, using validation only.

Each seed uses 700 training and 150 validation samples. Test images accessed: 0.

| Seed | Best epoch | MAE (arcsec) | RMSE (arcsec) | Median relative error | Bias (arcsec) | R2 | P90 (arcsec) | P95 (arcsec) | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12001 | 19 | 0.0884 | 0.1163 | 5.71% | -0.0031 | 0.9285 | 0.1968 | 0.2375 | 131.4 |
| 12002 | 15 | 0.0859 | 0.1124 | 5.92% | +0.0085 | 0.9332 | 0.1806 | 0.2286 | 146.0 |
| 12003 | 12 | 0.1021 | 0.1360 | 6.57% | -0.0109 | 0.9022 | 0.2132 | 0.2473 | 143.5 |

## Across-seed metrics

| Metric | Mean | Sample standard deviation |
|---|---:|---:|
| bias_arcsec | -0.001829 | 0.009753 |
| mae_arcsec | 0.092097 | 0.008719 |
| mean_relative_error | 0.087810 | 0.007838 |
| median_absolute_error_arcsec | 0.073340 | 0.005393 |
| median_relative_error | 0.060638 | 0.004478 |
| p90_absolute_error_arcsec | 0.196896 | 0.016296 |
| p95_absolute_error_arcsec | 0.237815 | 0.009401 |
| r2 | 0.921273 | 0.016719 |
| rmse_arcsec | 0.121584 | 0.012663 |

Arithmetic mean and sample standard deviation (ddof=1) across seeds; not a confidence interval or ensemble evaluation.

## Radius-dependent errors

Bins are [lower, upper), except the last bounded bin includes its upper edge. Outside-range groups and empty bins remain in the CSV/JSON.

| Radius bin (arcsec) | Samples per seed | Mean MAE | Mean bias | Mean P95 |
|---|---:|---:|---:|---:|
| 0.5 to 1.0 | 51 | 0.0932 | +0.0466 | 0.2287 |
| 1.0 to 1.5 | 47 | 0.1011 | -0.0003 | 0.2382 |
| 1.5 to 2.0 | 52 | 0.0828 | -0.0507 | 0.2139 |

## Interpretation

MAE is the average angular miss. P90/P95 are error thresholds containing 90%/95% of validation cases. Bias is prediction minus truth. Seed variation measures sensitivity to initialization, dropout, and training order on the same fixed data split.

Compare future candidates using the same seeds and report paired changes. Improvements smaller than the baseline seed variation are inconclusive; three seeds do not establish statistical significance. Inspect P95 and radius-bin bias before accepting a lower mean MAE. These are individual-model metrics, not predictions from an ensemble.

The checkpoint for each seed is selected by validation MAE. Do not pick a lucky seed as the headline result or use existing test metrics to choose models. A final challenger requires the independent holdout described in the planning notebook.

Open `seed_comparison.png` for errors and radius-bin bias, `per_seed_metrics.csv` for per-run results, and `aggregate_validation.json` for hashes, configurations, environments, and exact metrics.

In-domain synthetic validation; test manifests hashed for identity, zero test images loaded.
