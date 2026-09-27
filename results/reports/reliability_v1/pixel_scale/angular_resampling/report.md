# Pixel-scale audit — frozen CNN checkpoints

Validation-only matched counterfactual evaluation. No checkpoint was retrained or selected here.

- Independent validation parents: **150**
- Frozen checkpoints: **3**
- Practical-effect threshold: **0.010 arcsec**
- Historical synthetic test split accessed: **no**

## Paired effects

| Input strategy | Shift | Parents | Mean paired MAE change (arcsec) | Parent-bootstrap 95% interval | Supported >= 0.01 arcsec increase |
|---|---|---:|---:|---:|:---:|
| angular_resampled | pixel-scale-004 | 150 | +0.0070 | [-0.0042, +0.0186] | no |
| angular_resampled | pixel-scale-006 | 150 | -0.0043 | [-0.0143, +0.0053] | no |
| native | pixel-scale-004 | 150 | +0.0707 | [+0.0519, +0.0900] | yes |
| native | pixel-scale-006 | 150 | +0.0489 | [+0.0326, +0.0647] | yes |

The parent-bootstrap interval resamples independent lens parents after averaging paired effects across the three training seeds. Seed variation is listed separately in `per_seed_metrics.csv` and is not a confidence interval.

The tested scenes hold the current simulator's detector settings fixed while changing angular sampling. This is a controlled sampling stress test, not a calibrated transfer simulation for a different instrument.

See `scale_summary.csv`, `predictions.csv`, `paired_predictions.csv`, `paired_effects.csv`, and `paired_scale_effects.png` for complete measurements.
