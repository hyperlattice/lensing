# theta-e-cnn-baseline-v1 — test evaluation

The CNN lowers MAE by 77.2% relative to the train-mean baseline. Test MAE differs from the checkpoint validation MAE by -0.0007 arcsec. These numbers measure only the frozen synthetic distribution and do not establish real-HST accuracy.

## Overall metrics

- Samples: **150**
- MAE: **0.0876 arcsec**
- RMSE: **0.1292 arcsec**
- Median relative error: **4.93%**
- Bias: **-0.0090 arcsec**
- R²: **0.9127**

## How to inspect

- `predictions.png`: closeness to the diagonal, residual trends, and skew/outliers.
- `predictions.csv`: every truth, prediction, residual, relative error, and diagnostic group.
- `metrics.json`: exact overall, baseline, and subgroup values.
- The largest-error rows in `metrics.json` help identify failure modes; they must not be used to retune this locked test run.

This report does not measure transfer to real HST observations.
