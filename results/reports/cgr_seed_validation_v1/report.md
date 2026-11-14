# C/G/R seed validation

Reanalysis of the existing ten paired C/G/R training seeds on the frozen synthetic holdout. No additional training fits were made.

Holdout: 150 parents and 750 observations per arm/seed; 10 seeds per arm.

## Mean performance across seeds

| Arm | Canonical MAE (mean +/- seed SD) | Four-stress MAE (mean +/- seed SD) |
|---|---:|---:|
| C | 0.091536 +/- 0.005195 | 0.141251 +/- 0.005824 |
| G | 0.087591 +/- 0.011760 | 0.111507 +/- 0.008870 |
| R | 0.092366 +/- 0.010102 | 0.107385 +/- 0.005559 |

## Four-stress MAE by seed

Positive differences mean the candidate arm had higher error than its comparator.

| Seed | C | G | R | G-C | R-G | R-C |
|---:|---:|---:|---:|---:|---:|---:|
| 730001 | 0.146847 | 0.109919 | 0.119048 | -0.036928 | +0.009129 | -0.027799 |
| 730002 | 0.142559 | 0.112743 | 0.114147 | -0.029816 | +0.001404 | -0.028412 |
| 730003 | 0.142862 | 0.103415 | 0.108333 | -0.039446 | +0.004918 | -0.034529 |
| 730004 | 0.141142 | 0.106952 | 0.107015 | -0.034190 | +0.000063 | -0.034127 |
| 730005 | 0.147444 | 0.105610 | 0.103113 | -0.041834 | -0.002497 | -0.044331 |
| 730006 | 0.144114 | 0.129326 | 0.108779 | -0.014788 | -0.020547 | -0.035335 |
| 730007 | 0.126648 | 0.101452 | 0.100876 | -0.025196 | -0.000577 | -0.025773 |
| 730008 | 0.141846 | 0.114954 | 0.105282 | -0.026892 | -0.009672 | -0.036564 |
| 730009 | 0.140858 | 0.123193 | 0.103839 | -0.017665 | -0.019355 | -0.037019 |
| 730010 | 0.138195 | 0.107504 | 0.103416 | -0.030690 | -0.004089 | -0.034779 |

## Paired seed-averaged contrasts

Parent intervals resample lens systems while holding all ten seeds fixed. Seed intervals resample the ten paired training-seed effects while holding the 150 systems fixed.

| Contrast | Mean difference (arcsec) | Parent-bootstrap 95% interval | Seed-bootstrap 95% interval | Seeds with lower candidate MAE |
|---|---:|---:|---:|---:|
| G-C | -0.029744 | [-0.036982, -0.022676] | [-0.034720, -0.024479] | 10/10 |
| R-G | -0.004122 | [-0.008228, -0.000087] | [-0.010125, +0.001348] | 6/10 |
| R-C | -0.033867 | [-0.042824, -0.025072] | [-0.037131, -0.030719] | 10/10 |

## Scope and provenance

Predictions were written before joining the target column. The prediction file is sealed by SHA-256; the JSON report records all 30 checkpoint hashes.

This is a seed-consistency analysis on the same frozen synthetic holdout, not an independent test cohort. The separate parent and seed bootstrap intervals measure uncertainty across those two sampling axes; neither reuses a newly generated test population.

Prediction SHA-256: `60aa014f8b465b3df96ea9f9e813359ee45d9fd21aa2a85a5f59f7c012c19fd8`
Checkpoint hashes recorded: 30
