# Architecture ablation (five paired seeds)

Uniform-variation arm; 5 paired seeds; 150 holdout parents (750 condition images).

| Architecture | Parameters | Canonical MAE | Four-stress MAE |
|---|---:|---:|---:|
| Compact | 62,397 | 0.092197 | 0.117802 |
| Reference | 248,409 | 0.082345 | 0.107728 |

Compact minus reference four-stress difference: 0.010075 arcsec (parent-bootstrap 95% interval [0.007529, 0.012557]).

This is an exploratory capacity comparison. The primary C/G/R result continues to use ten seeds and a fixed architecture.

## Four-stress MAE by seed

| Seed | Compact | Reference |
|---:|---:|---:|
| 730001 | 0.116951 | 0.109919 |
| 730002 | 0.123526 | 0.112743 |
| 730003 | 0.116266 | 0.103415 |
| 730004 | 0.119868 | 0.106952 |
| 730005 | 0.112401 | 0.105610 |
