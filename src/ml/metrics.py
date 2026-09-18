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
    slope = None if denominator == 0 else float(np.sum((targets - targets.mean()) * (predictions - predictions.mean())) / denominator)
    r2 = None if denominator == 0 else 1.0 - float(np.square(residuals).sum()) / denominator
    return {
        "count": int(len(targets)),
        "fraction_within_005_arcsec": float(np.mean(absolute <= 0.05)),
        "fraction_within_010_arcsec": float(np.mean(absolute <= 0.10)),
        "prediction_vs_truth_slope": slope,
        "prediction_vs_truth_intercept_arcsec": None if slope is None else float(predictions.mean() - slope * targets.mean()),
        "mae_arcsec": float(absolute.mean()),
        "rmse_arcsec": float(np.sqrt(np.square(residuals).mean())),
        "median_absolute_error_arcsec": float(np.median(absolute)),
        "p90_absolute_error_arcsec": float(np.percentile(absolute, 90)),
        "p95_absolute_error_arcsec": float(np.percentile(absolute, 95)),
        "mean_relative_error": float(relative.mean()),
        "median_relative_error": float(np.median(relative)),
        "bias_arcsec": float(residuals.mean()),
        "r2": None if r2 is None or not math.isfinite(r2) else float(r2),
    }


def target_bin_metrics(targets, predictions, edges):
    """Report fixed radius bins, including empty bins and out-of-range samples.

    Interior bins are left-closed/right-open; the final bin includes its upper
    edge. Edges must be chosen in the experiment config, never from test data.
    """
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    regression_metrics(targets, predictions)  # Validate before slicing.
    edges = validate_bin_edges(edges)
    groups = []
    masks = [(None, edges[0], targets < edges[0])]
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        upper_mask = targets <= upper if index == len(edges) - 2 else targets < upper
        masks.append((lower, upper, (targets >= lower) & upper_mask))
    masks.append((edges[-1], None, targets > edges[-1]))
    for lower, upper, mask in masks:
        groups.append({
            "lower_arcsec": lower,
            "upper_arcsec": upper,
            "count": int(mask.sum()),
            "metrics": regression_metrics(targets[mask], predictions[mask]) if mask.any() else None,
        })
    return groups


def validate_bin_edges(edges):
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
        raise ValueError("target bin edges must be finite, strictly increasing, and contain at least two values")
    return edges.tolist()


def mean_baseline_metrics(train_target_mean, targets):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.full(targets.shape, float(train_target_mean), dtype=np.float64)
    return regression_metrics(targets, predictions)


def improvement_fraction(model_value, baseline_value):
    baseline_value = float(baseline_value)
    if baseline_value <= 0:
        return None
    return float(1.0 - float(model_value) / baseline_value)
