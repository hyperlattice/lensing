# theta-e-cnn-baseline-v1 — validation evaluation

The CNN lowers MAE by 76.8% relative to the train-mean baseline. Test MAE differs from the checkpoint validation MAE by +0.0000 arcsec. These numbers measure only the frozen synthetic distribution and do not establish real-HST accuracy.

## Overall metrics

- Samples: **150**
- MAE: **0.0884 arcsec**
- RMSE: **0.1163 arcsec**
- Median relative error: **5.71%**
- Bias: **-0.0031 arcsec**
- R²: **0.9285**

## How to inspect

- `predictions.png`: closeness to the diagonal, residual trends, and skew/outliers.
- `predictions.csv`: every truth, prediction, residual, relative error, and diagnostic group.
- `metrics.json`: exact overall, baseline, and subgroup values.
- The largest-error rows in `metrics.json` help identify failure modes; they must not be used to retune this locked test run.

This report does not measure transfer to real HST observations.
