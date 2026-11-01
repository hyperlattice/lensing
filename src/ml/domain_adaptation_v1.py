"""Literature-inspired MMD primitive, not an exact published reproduction.

Gaussian kernels use exp(-squared_distance/(2*sigma^2)), averaged over explicit
fixed bandwidths in feature units. No target labels enter this interface.
"""
import math

import torch


def gaussian_mmd(source, target, *, bandwidths, estimator="biased"):
    """Differentiable squared MMD; unbiased estimates may legitimately be negative.

    The biased V-statistic is nonnegative up to floating point roundoff and
    includes kernel diagonals. The unbiased U-statistic excludes within-domain
    diagonals and requires at least two samples per domain. Cross terms use
    every source-target pair; batch sizes may differ.
    """
    if (source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1] or
        source.shape[1] < 1 or min(source.shape[0], target.shape[0]) < 1 or
        not source.is_floating_point() or not target.is_floating_point() or
        source.dtype != target.dtype or source.device != target.device or
        not torch.isfinite(source).all() or not torch.isfinite(target).all()):
        raise ValueError("Expected finite, compatible two-dimensional feature batches")
    sigmas = tuple(float(s) for s in bandwidths)
    if not sigmas or any(not math.isfinite(s) or s <= 0 for s in sigmas):
        raise ValueError("MMD bandwidths must be explicit, finite and positive")
    if estimator not in {"biased", "unbiased"}:
        raise ValueError("Unknown MMD estimator")
    n, m = source.shape[0], target.shape[0]
    if estimator == "unbiased" and min(n, m) < 2:
        raise ValueError("Unbiased MMD requires two examples per domain")

    def kernel(a, b):
        distances = (a[:, None, :] - b[None, :, :]).square().sum(dim=-1)
        return torch.stack([torch.exp(-distances / (2*s*s)) for s in sigmas]).mean(0)

    xx, yy, xy = kernel(source, source), kernel(target, target), kernel(source, target)
    if estimator == "biased":
        return xx.mean() + yy.mean() - 2*xy.mean()
    return ((xx.sum() - xx.diagonal().sum()) / (n*(n-1)) +
            (yy.sum() - yy.diagonal().sum()) / (m*(m-1)) - 2*xy.mean())


def image_features_and_prediction(model, images):
    """Expose the existing image-only CNN pooled features without changing weights.

    This is the declared initial alignment site; it is not claimed to reproduce
    an author's architecture. The caller controls train/eval mode and dropout.
    """
    if model.metadata_features != 0 or images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("Adaptation expects an image-only single-band CNN")
    features = model.pool(model.features(images)).flatten(start_dim=1)
    return features, model.regressor(features).squeeze(-1)
