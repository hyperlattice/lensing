# theta-e-huber-delta-050-v1 versus theta-e-cnn-seeds-v1

MAE improves >=0.01 arcsec; P95 and mean worst-bin absolute bias do not increase; all seeds improve MAE.

Passes conservative development gate: False. Validation only; this is not a significance test.

Calibration fits predicted radius = slope * true radius + intercept. Prior bounds are fixed at [0.5, 2.0] arcsec.

| Metric | Reference mean | Candidate mean | Paired mean delta |
|---|---:|---:|---:|
| bias_arcsec | -0.001829 | 0.002888 | +0.004717 |
| fraction_within_005_arcsec | 0.375556 | 0.386667 | +0.011111 |
| fraction_within_010_arcsec | 0.635556 | 0.662222 | +0.026667 |
| mae_arcsec | 0.092097 | 0.089680 | -0.002417 |
| mean_relative_error | 0.087810 | 0.086933 | -0.000877 |
| median_absolute_error_arcsec | 0.073340 | 0.070754 | -0.002586 |
| median_relative_error | 0.060638 | 0.054847 | -0.005791 |
| out_of_prior_fraction | 0.006667 | 0.004444 | -0.002222 |
| p90_absolute_error_arcsec | 0.196896 | 0.193642 | -0.003254 |
| p95_absolute_error_arcsec | 0.237815 | 0.233704 | -0.004111 |
| prediction_vs_truth_intercept_arcsec | 0.127216 | 0.129162 | +0.001946 |
| prediction_vs_truth_slope | 0.896044 | 0.898276 | +0.002232 |
| r2 | 0.921273 | 0.923352 | +0.002079 |
| rmse_arcsec | 0.121584 | 0.120177 | -0.001407 |
| worst_radius_bin_absolute_bias_arcsec | 0.057316 | 0.052888 | -0.004428 |
