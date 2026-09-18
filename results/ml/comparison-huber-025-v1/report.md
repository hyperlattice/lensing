# theta-e-huber-delta-025-v1 versus theta-e-cnn-seeds-v1

MAE improves >=0.01 arcsec; P95 and mean worst-bin absolute bias do not increase; all seeds improve MAE.

Passes conservative development gate: False. Validation only; this is not a significance test.

Calibration fits predicted radius = slope * true radius + intercept. Prior bounds are fixed at [0.5, 2.0] arcsec.

| Metric | Reference mean | Candidate mean | Paired mean delta |
|---|---:|---:|---:|
| bias_arcsec | -0.001829 | 0.004908 | +0.006737 |
| fraction_within_005_arcsec | 0.375556 | 0.388889 | +0.013333 |
| fraction_within_010_arcsec | 0.635556 | 0.688889 | +0.053333 |
| mae_arcsec | 0.092097 | 0.085348 | -0.006749 |
| mean_relative_error | 0.087810 | 0.083279 | -0.004531 |
| median_absolute_error_arcsec | 0.073340 | 0.062421 | -0.010919 |
| median_relative_error | 0.060638 | 0.053369 | -0.007269 |
| out_of_prior_fraction | 0.006667 | 0.013333 | +0.006667 |
| p90_absolute_error_arcsec | 0.196896 | 0.178585 | -0.018311 |
| p95_absolute_error_arcsec | 0.237815 | 0.218873 | -0.018942 |
| prediction_vs_truth_intercept_arcsec | 0.127216 | 0.118301 | -0.008915 |
| prediction_vs_truth_slope | 0.896044 | 0.908653 | +0.012609 |
| r2 | 0.921273 | 0.931176 | +0.009903 |
| rmse_arcsec | 0.121584 | 0.114025 | -0.007559 |
| worst_radius_bin_absolute_bias_arcsec | 0.057316 | 0.057050 | -0.000266 |
