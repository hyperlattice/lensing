"""Create publication figures from the sealed equal-budget report only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/ml/equal-budget-final-v1")
REPORT = ROOT / "final_report.json"
FIGURES = ROOT / "figures"
CONDITIONS = (
    "canonical",
    "pixel-scale-004",
    "pixel-scale-006",
    "exposure-half",
    "read-noise-double",
)
LABELS = ("Canonical", "Pixel 0.04", "Pixel 0.06", "Half exposure", "2x read noise")
ARMS = ("C", "G", "R")
COLORS = {"C": "#4C78A8", "G": "#F58518", "R": "#54A24B"}


def _save(fig, name, figures):
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_mae(report: dict[str, Any], figures: Path = FIGURES) -> None:
    values = np.array(
        [
            [
                report["metrics_by_arm_condition"][arm][condition]["mae_arcsec"]
                for condition in CONDITIONS
            ]
            for arm in ARMS
        ]
    )
    x = np.arange(len(CONDITIONS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for index, arm in enumerate(ARMS):
        ax.bar(
            x + (index - 1) * width,
            values[index],
            width,
            label={"C": "C: canonical", "G": "G: uniform variation", "R": "R: failure-weighted"}[
                arm
            ],
            color=COLORS[arm],
        )
    ax.set_xticks(x, LABELS, rotation=18, ha="right")
    ax.set_ylabel("MAE (arcsec)")
    ax.set_title("Equal-budget performance by observation condition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "equal_budget_mae_by_condition", figures)


def plot_contrasts(report: dict[str, Any], figures: Path = FIGURES) -> None:
    names = ("G_vs_C_four_stress", "R_vs_G_four_stress", "R_vs_C_four_stress")
    labels = ("G − C", "R − G", "R − C")
    values = np.array([report["contrasts"][name]["difference_arcsec"] for name in names])
    lows = np.array([report["contrasts"][name]["bootstrap_95_low_arcsec"] for name in names])
    highs = np.array([report["contrasts"][name]["bootstrap_95_high_arcsec"] for name in names])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.errorbar(
        values,
        y,
        xerr=[values - lows, highs - values],
        fmt="o",
        color="#222222",
        ecolor="#222222",
        capsize=4,
        markersize=6,
        linewidth=1.5,
    )
    ax.axvline(0.0, color="#888888", linewidth=1)
    ax.axvline(
        -0.005, color="#B22222", linewidth=1, linestyle="--", label="−0.005 arcsec reference margin"
    )
    ax.set_yticks(y, labels)
    ax.set_xlabel("MAE change (candidate − comparator, arcsec)")
    ax.set_title("Parent-bootstrap contrasts (95% intervals)")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "equal_budget_paired_contrasts", figures)


def plot(config: dict, *, report_output: Path | None = None) -> None:
    """Generate the existing figure filenames beside the configured report."""
    root = Path(config["output"]) if report_output is None else report_output
    report = json.loads((root / "final_report.json").read_text(encoding="utf-8"))
    plot_mae(report, root / "figures")
    plot_contrasts(report, root / "figures")


def plot_hst(config: dict[str, Any], cohort: str) -> None:
    """Plot saved per-lens reference agreement and paired changes without inference."""
    import pandas as pd

    root = Path(config["output"]) / cohort
    report = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(root / "reference_disagreements.csv")
    changes = pd.read_csv(root / "per_lens_changes.csv").sort_values("system_id")
    systems = changes["system_id"].tolist()
    references = report["references"]
    y = np.arange(len(systems))
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(systems) * 0.22)), sharey=True)
    for index, system in enumerate(systems):
        reference = references[system]
        axes[0].errorbar(
            reference["median_arcsec"],
            index,
            xerr=[[reference["error_lower"]], [reference["error_upper"]]],
            fmt="s",
            color="black",
            capsize=2,
            label="Classical reference" if index == 0 else None,
        )
    for arm_index, arm in enumerate(ARMS):
        subset = frame[frame["arm"] == arm]
        positions = subset["system_id"].map(dict(zip(systems, y, strict=True)))
        axes[0].scatter(
            subset["theta_E_arcsec_predicted"],
            positions + (arm_index - 1) * 0.16,
            s=9,
            alpha=0.55,
            color=COLORS[arm],
            label=arm,
        )
    axes[0].set_yticks(y, systems)
    axes[0].set_xlabel("Radius (arcsec); each colored point is one training seed")
    axes[0].set_title("Predictions and model-dependent references", pad=32)
    axes[0].legend(frameon=False, ncols=4, loc="lower left", bbox_to_anchor=(0, 1.01))
    for name, offset, color, label in (
        ("G_minus_C_arcsec", -0.1, COLORS["G"], "G - C"),
        ("R_minus_G_arcsec", 0.1, COLORS["R"], "R - G"),
    ):
        axes[1].scatter(changes[name], y + offset, s=16, color=color, label=label)
    axes[1].axvline(0, color="gray", linewidth=1)
    axes[1].set_xlabel("Change in mean absolute reference disagreement (arcsec)")
    axes[1].set_title(f"{cohort.capitalize()}: {len(systems)} independent lenses", pad=32)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save(fig, "hst_reference_agreement", root / "figures")
    lines = [
        "# HST reference agreement",
        "",
        f"Cohort: {cohort}; {len(systems)} independent lenses.",
        "",
        "| Arm | Mean absolute disagreement (arcsec) | Signed bias (arcsec) |",
        "|---|---:|---:|",
    ]
    for arm, metrics in report["metrics"].items():
        lines.append(
            f"| {arm} | {metrics['mean_absolute_disagreement_arcsec']:.6f} | {metrics['signed_bias_arcsec']:.6f} |"
        )
    lines.extend(
        ["", "| Contrast | Mean change (arcsec) | 95% paired lens interval |", "|---|---:|---|"]
    )
    for name, contrast in report["contrasts"].items():
        interval = "Development: descriptive"
        if contrast["bootstrap_95_low_arcsec"] is not None:
            interval = (
                f"[{contrast['bootstrap_95_low_arcsec']:.6f}, "
                f"{contrast['bootstrap_95_high_arcsec']:.6f}]"
            )
        lines.append(f"| {name} | {contrast['difference_arcsec']:.6f} | {interval} |")
    lines.extend(
        ["", "Intervals condition on the evaluated training seeds. Development is descriptive.", ""]
    )
    (root / "result_overview.md").write_text("\n".join(lines), encoding="utf-8")


def preview_hst_preparation(config: dict[str, Any], cohort: str) -> None:
    """Show the prepared physical image and normalized native PSF for visual review."""
    root = Path(config["prepared_root"]) / cohort
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["observations"]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 3 * len(rows)), squeeze=False)
    for index, row in enumerate(rows):
        image = np.load(root / row["image"]["path"], allow_pickle=False)
        psf = np.load(root / row["psf"]["path"], allow_pickle=False)
        low, high = np.percentile(image, [1, 99])
        display = axes[index, 0].imshow(image, origin="lower", cmap="gray", vmin=low, vmax=high)
        fig.colorbar(display, ax=axes[index, 0], label="electrons/s per pixel")
        axes[index, 0].set_title(f"{row['system_id']} {row['band']}: prepared native field")
        center = row["preparation"]["center_xy_prepared"]
        axes[index, 0].plot(*center, marker="+", color="#F58518", markersize=10)
        axes[index, 1].imshow(psf, origin="lower", cmap="gray")
        axes[index, 1].set_title("Normalized native PSF")
        for axis in axes[index]:
            axis.set_xlabel("Native pixel x")
            axis.set_ylabel("Native pixel y")
    fig.tight_layout()
    fig.savefig(root / "preview.png", dpi=100)
    plt.close(fig)
