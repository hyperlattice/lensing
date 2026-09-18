"""Scalar regression objectives; all operate on standardized targets."""
import math

import torch
from torch import nn


class LogCoshLoss(nn.Module):
    def forward(self, predictions, targets):
        error = (predictions - targets).abs()
        return (error + torch.nn.functional.softplus(-2 * error) - math.log(2)).mean()


def build_loss(config):
    settings = dict(config)
    kind = settings.pop("kind", "huber")
    if kind == "huber":
        delta = settings.pop("delta", 1.0)
        if isinstance(delta, bool) or not math.isfinite(float(delta)) or float(delta) <= 0:
            raise ValueError("Huber delta must be finite and positive")
        loss = nn.HuberLoss(delta=float(delta))
    elif kind in {"mae", "mse", "log_cosh"}:
        loss = {"mae": nn.L1Loss, "mse": nn.MSELoss, "log_cosh": LogCoshLoss}[kind]()
    else:
        raise ValueError(f"Unknown loss: {kind}")
    if settings:
        raise ValueError(f"Unknown loss settings: {sorted(settings)}")
    return loss
