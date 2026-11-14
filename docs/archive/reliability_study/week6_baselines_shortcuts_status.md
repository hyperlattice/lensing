# Week 6 — Baselines, learning curve, and shortcut checks

**Status:** complete with a documented classical-method limitation  
**Protocol:** [`protocol_v1.6.md`](protocol_v1.6.md)  
**Decision:** D-017 in [`decision_log.md`](decision_log.md)

## Scope and integrity

Week 6 used the original 700 independent training parents and the fixed 150-parent validation set. The learning curve used nested deterministic subsets of 100, 300, and 700 parents (selection seed 6031), with the established CNN architecture, optimizer, and seeds 12001, 12002, and 12003. The validation population stayed fixed. Parent bootstrap intervals and seed variation are reported separately.

A fixed 45-feature radial/angular Ridge model was standardized using training parents only. Its regularization strength was selected by five-fold cross-validation over training parents only. The CNN and classical estimators were compared on canonical validation data and six matched Week 3–5 stress conditions. The classical control is the v1.3 observation-only SIS/SIE point-image geometry estimator with its training-mean fallback. The repository's pixel optimizer initializes from truth-sidecar parameters, so a blind pixel-likelihood SIE fit was not available and is not claimed here.

Five image interventions were applied to the same canonical validation images before each frozen checkpoint's training normalization: 90-degree rotation, brightness ×0.5 and ×2, a central disk mask below 0.50 arcsec, and an annulus mask from 0.70 to 2.20 arcsec. Masks use fixed image-centered radii, not truth-derived lens or source regions. These are image ablations, not physically rendered observations.

The historical test examples were not loaded or scored. No real-data labels were used. Canonical intervention predictions reproduced the saved checkpoints to a maximum absolute difference of **1.4×10⁻⁷ arcsec**.

## Independent-parent learning curve

| Train parents | Validation MAE, mean across seeds (arcsec) | Parent-bootstrap 95% interval | Seed SD of MAE | Train MAE, mean across seeds |
|---:|---:|---:|---:|---:|
| 100 | 0.2039 | [0.1844, 0.2241] | 0.0537 | 0.0504 |
| 300 | 0.1163 | [0.1043, 0.1287] | 0.0111 | 0.0514 |
| 700 | 0.0946 | [0.0831, 0.1068] | 0.0061 | 0.0348 |

Relative to 700 parents, the paired MAE penalty was **+0.1093 arcsec** for 100 parents (95% parent interval [+0.0911, +0.1277]) and **+0.0217 arcsec** for 300 parents ([+0.0135, +0.0299]). The fixed model continued to improve through the largest tested training set. This supports prioritizing additional independent-parent volume and diversity before a major architecture change. The 100-parent condition also had much larger seed spread, so the curve is not a precise extrapolation beyond 700 parents.

![Week 6 learning curve](../../results/reports/reliability_v1/baselines_week6/learning_curve/learning_curve.png)

## Baseline comparison

Metrics below use the three-seed CNN prediction average for a condition. The training-mean prediction is constant across conditions because the matched validation labels are unchanged.

| Condition | CNN mean prediction MAE | Ridge MAE | Train-mean MAE | Point-image classical MAE | Classical fallbacks |
|---|---:|---:|---:|---:|---:|
| Canonical | 0.0874 | 0.2464 | 0.3803 | 0.2007 | 44/150 |
| Pixel scale 0.04 | 0.1592 | 0.2715 | 0.3803 | 0.2168 | 50/150 |
| Pixel scale 0.06 | 0.1397 | 0.2618 | 0.3803 | 0.1944 | 45/150 |
| Exposure ×0.5 | 0.1323 | 0.2759 | 0.3803 | 0.2354 | 65/150 |
| Read noise ×2 | 0.1193 | 0.2702 | 0.3803 | 0.2167 | 59/150 |
| Source x +0.15 arcsec | 0.1133 | 0.2642 | 0.3803 | 0.2419 | 64/150 |
| Shear γ₂ = 0.05 | 0.0946 | 0.2436 | 0.3803 | 0.2090 | 48/150 |

Five-fold train-only selection chose Ridge α=0.01 (mean fold MAE 0.2364 arcsec). Ridge improved on the train mean, but did not match the CNN or the point-image control on canonical validation. The classical comparison remains a limited point-image estimator; fallbacks are included in its operational MAE, and fit-only cases are not substituted for all-parent performance.

## Shortcut intervention results

The annulus mask caused a large, practically important increase in error: **+0.2632 arcsec** (95% parent interval [+0.2246, +0.3022]); its mean absolute prediction shift was 0.3127 arcsec. Brightness scaling was practically small under the frozen rule. Rotation also had a practically small paired MAE change (−0.0007, interval [−0.0074, +0.0060]), although the mean absolute prediction shift was 0.0405 arcsec. The central mask's +0.0074 MAE change (interval [+0.0036, +0.0119]) is inconclusive against the 0.01-arcsec rule.

The annulus result shows that the CNN depends strongly on image information in the fixed arc-like radial region. It does not establish an invalid shortcut: masking also removes useful lensed-source signal. Rotation and brightness are artificial image changes, and no causal claim is made from these tests. The complete table and interpretation are in [`shortcut_audit.md`](../../results/reports/reliability_v1/baselines_week6/shortcut_audit.md).

## Artifacts

- Protocol and decision: [`protocol_v1.6.md`](protocol_v1.6.md), [`decision_log.md`](decision_log.md)
- Registry: [`experiment_registry.csv`](experiment_registry.csv)
- Learning-curve selections, run configs, checkpoints, predictions, metrics, and plot: [`learning_curve/`](../../results/reports/reliability_v1/baselines_week6/learning_curve/)
- Ridge folds, coefficients, predictions, and condition metrics: [`ridge/`](../../results/reports/reliability_v1/baselines_week6/ridge/)
- Shortcut predictions, paired effects, target bins, and audit: [`shortcuts/`](../../results/reports/reliability_v1/baselines_week6/shortcuts/), [`shortcut_audit.md`](../../results/reports/reliability_v1/baselines_week6/shortcut_audit.md)
- Combined model comparison and paired method differences: [`comparison/`](../../results/reports/reliability_v1/baselines_week6/comparison/)
- Reproducible runner: [`reliability_week6.py`](../../src/ml/reliability_week6.py)

## Next phase

Proceed to Week 7 in [`week7_cross_simulator_next.md`](week7_cross_simulator_next.md): select and validate an alternate renderer, document conventions, create a small matched benchmark, and evaluate the frozen CNN without retraining. Do not claim simulator-independent inference unless that comparison supports it.
