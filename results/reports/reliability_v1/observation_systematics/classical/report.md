# Week 4 — Classical point-image comparison

The frozen v1.3 SIS/SIE point-image geometry control was applied to the same validation parents. Fitting used observation products only; validation labels were joined after `classical_predictions_before_scoring.csv` was written.

- Independent parents per condition: **150**
- Training-mean fallback: **1.2483 arcsec**
- Historical test split accessed: **no**
- Estimator scope: point-image geometry with fallback; not full pixel-likelihood inference.

## Condition results

| Condition | Classical MAE | Fitted | Fallbacks | CNN MAE | Classical minus CNN MAE [95% parent interval] |
|---|---:|---:|---:|---:|---:|
| canonical | 0.2007 | 106/150 | 44 | 0.0921 | +0.1086 [+0.0724, +0.1469] |
| psf-width-075 | 0.1963 | 112/150 | 38 | 0.0930 | +0.1033 [+0.0677, +0.1394] |
| psf-width-125 | 0.1925 | 107/150 | 43 | 0.0932 | +0.0993 [+0.0659, +0.1350] |
| noise-repeat-1 | 0.2074 | 107/150 | 43 | 0.0896 | +0.1178 [+0.0821, +0.1550] |
| noise-repeat-2 | 0.2033 | 103/150 | 47 | 0.0934 | +0.1099 [+0.0736, +0.1473] |
| exposure-half | 0.2354 | 85/150 | 65 | 0.1384 | +0.0970 [+0.0594, +0.1361] |
| read-noise-double | 0.2167 | 91/150 | 59 | 0.1271 | +0.0896 [+0.0541, +0.1275] |
| lens-light-zero | 0.1796 | 114/150 | 36 | 0.0979 | +0.0817 [+0.0474, +0.1167] |
| lens-light-double | 0.1936 | 106/150 | 44 | 0.0868 | +0.1068 [+0.0720, +0.1433] |

Full predictions, paired effects, and fit diagnostics are retained in this directory.
