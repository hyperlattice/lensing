"""Dependency-light scalar-regression metrics and subgroup summaries."""

from __future__ import annotations

import math

import numpy as np


def regression_metrics(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.ndim != 1 or len(targets) == 0:
        raise ValueError("targets and predictions must be nonempty one-dimensional arrays")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("metrics require finite targets and predictions")
    residuals = predictions - targets
    absolute = np.abs(residuals)
    relative = absolute / np.maximum(np.abs(targets), np.finfo(np.float64).eps)
    denominator = float(np.square(targets - targets.mean()).sum())
    r2 = None if denominator == 0 else 1.0 - float(np.square(residuals).sum()) / denominator
    return {
        "count": int(len(targets)),
        "mae_arcsec": float(absolute.mean()),
        "rmse_arcsec": float(np.sqrt(np.square(residuals).mean())),
        "median_absolute_error_arcsec": float(np.median(absolute)),
        "mean_relative_error": float(relative.mean()),
        "median_relative_error": float(np.median(relative)),
        "bias_arcsec": float(residuals.mean()),
        "r2": None if r2 is None or not math.isfinite(r2) else float(r2),
    }


def mean_baseline_metrics(train_target_mean, targets):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.full(targets.shape, float(train_target_mean), dtype=np.float64)
    return regression_metrics(targets, predictions)


def improvement_fraction(model_value, baseline_value):
    baseline_value = float(baseline_value)
    if baseline_value <= 0:
        return None
    return float(1.0 - float(model_value) / baseline_value)
