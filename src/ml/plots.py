"""Noninteractive diagnostic plots for the baseline experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(xlabel="epoch", ylabel="Huber loss (normalized target)", title="Optimization")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_mae_arcsec"] for row in history], color="tab:orange")
    axes[1].set(xlabel="epoch", ylabel="MAE (arcsec)", title="Validation accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_predictions(targets, predictions, output_path, *, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    residuals = predictions - targets
    low = float(min(targets.min(), predictions.min()))
    high = float(max(targets.max(), predictions.max()))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(targets, predictions, s=16, alpha=0.65)
    axes[0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[0].set(xlabel="true theta_E (arcsec)", ylabel="predicted theta_E (arcsec)", title="Prediction")
    axes[1].scatter(targets, residuals, s=16, alpha=0.65, color="tab:orange")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="true theta_E (arcsec)", ylabel="prediction - truth (arcsec)", title="Residual by target")
    axes[2].hist(residuals, bins=min(20, max(5, len(residuals) // 5)), color="tab:green", alpha=0.8)
    axes[2].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[2].set(xlabel="residual (arcsec)", ylabel="count", title="Residual distribution")
    figure.suptitle(title)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_seed_comparison(summaries, output_path):
    """Show seed spread and radius-dependent bias on the same validation set."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    positions = np.arange(len(summaries))
    for offset, (metric, label) in enumerate((
        ("mae_arcsec", "MAE"), ("rmse_arcsec", "RMSE"), ("p95_absolute_error_arcsec", "P95"),
    )):
        axes[0].bar(positions + (offset - 1) * 0.24, [run["validation_metrics"][metric] for run in summaries], width=0.24, label=label)
    axes[0].set_xticks(positions, [str(run["seed"]) for run in summaries])
    axes[0].set(xlabel="training seed", ylabel="absolute error (arcsec)", title="Validation errors across seeds")
    for run in summaries:
        bins = [row for row in run["validation_target_bins"] if row["metrics"] is not None and row["lower_arcsec"] is not None and row["upper_arcsec"] is not None]
        centers = [(row["lower_arcsec"] + row["upper_arcsec"]) / 2 for row in bins]
        axes[1].plot(centers, [row["metrics"]["bias_arcsec"] for row in bins], marker="o", label=str(run["seed"]))
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="true radius bin center (arcsec)", ylabel="prediction - truth (arcsec)", title="Radius-dependent validation bias")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
