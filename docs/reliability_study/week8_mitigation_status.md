# Week 8 status — targeted nuisance randomization

Week 8 completed under the frozen v1.8 protocol. The intervention retained all 700 canonical training parents and added one seed-6031 randomized observation for each parent. The release contains 1,400 training rows from 700 independent systems, with matched angular field of view across the 0.04/0.05/0.06 arcsec per pixel shape buckets. Schema validation, ancestry checks, strict loading, and train/validation separation passed.

The three randomized CNN seeds were evaluated once on the fixed 150 validation parents in the canonical condition and four preregistered stress conditions. Predictions were written before validation target joins. The historical test split and real-data labels were not accessed.

| Condition | Paired MAE change (arcsec) | Parent bootstrap 95% interval | Mean absolute prediction shift (arcsec) |
|---|---:|---:|---:|
| Canonical | -0.0109 | [-0.0189, -0.0031] | 0.0464 |
| Pixel scale 0.04 | -0.0398 | [-0.0533, -0.0263] | 0.0875 |
| Pixel scale 0.06 | -0.0427 | [-0.0523, -0.0332] | 0.0683 |
| Half exposure | -0.0195 | [-0.0297, -0.0099] | 0.0591 |
| Double read noise | -0.0140 | [-0.0247, -0.0037] | 0.0610 |

The equal-weight mean stress change was -0.0290 arcsec (95% interval [-0.0351, -0.0231]). The stress improvement rule passed, as did canonical non-inferiority: its 95% upper bound was -0.0031 arcsec. The predeclared pixel-scale metadata diagnostic was therefore not run.

This supports mitigation only for the tested synthetic nuisance ranges and the frozen CNN family. It does not establish transfer to real images. The primary report, machine-readable summary, paired predictions, and figure are in [`results/reports/reliability_v1/mitigation/`](../../results/reports/reliability_v1/mitigation/).

## Next phase

Proceed to Week 9 using the successful Week 8 randomized checkpoints as the ML initializers. Freeze the matched validation conditions, observation-only peak extraction, default/heuristic/ML starting values, and the shared SIE point-image fitting procedure before producing Week 9 fit outputs. Keep the classical-comparator limitation explicit: its current blind optimizer fits detected point-image geometry, not the full pixel likelihood.
