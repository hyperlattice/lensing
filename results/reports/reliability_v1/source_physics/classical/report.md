# Week 4 â€” Classical point-image comparison

The frozen v1.3 SIS/SIE point-image geometry control was applied to the same validation parents. Fitting used observation products only; validation labels were joined after `classical_predictions_before_scoring.csv` was written.

- Independent parents per condition: **150**
- Training-mean fallback: **1.2483 arcsec**
- Historical test split accessed: **no**
- Estimator scope: point-image geometry with fallback; not full pixel-likelihood inference.

## Condition results

| Condition | Classical MAE | Fitted | Fallbacks | CNN MAE | Classical minus CNN MAE [95% parent interval] |
|---|---:|---:|---:|---:|---:|
| canonical | 0.2007 | 106/150 | 44 | 0.0921 | +0.1086 [+0.0724, +0.1469] |
| source-n-050 | 0.2123 | 100/150 | 50 | 0.1018 | +0.1105 [+0.0750, +0.1470] |
| source-n-500 | 0.2008 | 109/150 | 41 | 0.0818 | +0.1190 [+0.0840, +0.1562] |
| source-re-half | 0.2118 | 100/150 | 50 | 0.1050 | +0.1068 [+0.0699, +0.1450] |
| source-re-double | 0.1851 | 116/150 | 34 | 0.0765 | +0.1086 [+0.0732, +0.1452] |
| source-x-minus | 0.2200 | 102/150 | 48 | 0.1106 | +0.1094 [+0.0723, +0.1470] |
| source-x-plus | 0.2419 | 86/150 | 64 | 0.1169 | +0.1251 [+0.0881, +0.1633] |
| source-y-minus | 0.2105 | 100/150 | 50 | 0.1076 | +0.1029 [+0.0664, +0.1395] |
| source-y-plus | 0.2358 | 95/150 | 55 | 0.1029 | +0.1329 [+0.0961, +0.1710] |
| shear-gamma1-005 | 0.2105 | 109/150 | 41 | 0.0984 | +0.1122 [+0.0761, +0.1494] |
| shear-gamma2-005 | 0.2090 | 102/150 | 48 | 0.0996 | +0.1094 [+0.0730, +0.1467] |

Full predictions, paired effects, and fit diagnostics are retained in this directory.
