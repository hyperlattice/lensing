# Week 9 status — hybrid classical/ML initialization

Week 9 completed on the fixed 150 validation parents in the five v1.9 conditions, using the three SHA-256 verified Week 8 randomized checkpoints. The comparison produced 6,750 fit records: three starts for each parent, condition, and checkpoint seed. The full target-free fit file was written before validation `theta_E` values were joined. No truth-sidecar initialization, historical test data, or real-data labels were used.

| Condition | Train-mean MAE | Peak-radius MAE | ML-start MAE | ML minus train-mean evaluations (95% parent interval) |
|---|---:|---:|---:|---:|
| Canonical | 0.0698 | 0.0700 | 0.0701 | +0.17 [-0.51, +0.85] |
| Pixel scale 0.04 | 0.0688 | 0.0681 | 0.0685 | +0.23 [-0.63, +1.21] |
| Pixel scale 0.06 | 0.0587 | 0.0593 | 0.0588 | +0.55 [-0.20, +1.44] |
| Half exposure | 0.0776 | 0.0779 | 0.0749 | +0.86 [-0.73, +3.45] |
| Double read noise | 0.0713 | 0.0734 | 0.0718 | +0.28 [-0.34, +0.95] |

MAE is among successful final fits. Only 198 of the 750 parent-condition pairs produced three or four detected peaks, the cases where the estimator runs the bounded SIE optimizer; this corresponds to 594 parent-condition-seed cases. The remaining observations either produced a two-image closed-form SIS fit or lacked enough detected peaks. SIE convergence on the optimizer-eligible subset was 98–100% by condition.

The Week 8 CNN initialization preserved final fit quality within the 0.01 arcsec practical margin but showed no consistent speed advantage over the train-mean start. Its paired function-evaluation intervals included zero in every condition. The median peak-radius heuristic was similarly accurate and generally required fewer evaluations, though its differences were also condition dependent. The evidence does not support a general ML-initializer speedup.

This is a limited observation-only point-image SIS/SIE estimator, not a full pixel-likelihood fit. Two-image systems are solved in closed form, so those cases cannot test optimizer initialization. Peak detection itself limits coverage. See [`results/reports/reliability_v1/hybrid_week9/`](../../results/reports/reliability_v1/hybrid_week9/) for target-free fits, scored results, paired bootstrap effects, metrics, and the figure.

## Next phase

Proceed to Week 10 using the 137 processed CASTLES observation products and the final Week 8 randomized training distribution. Compare only measurable metadata and image-derived proxies; treat the 69 lens systems as clusters. Do not use the catalog's Einstein-radius or redshift fields as model truth, and do not report real-image parameter accuracy. The deliverable is a coverage map and a specification for a future blinded real benchmark.
