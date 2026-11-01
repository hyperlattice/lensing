"""Create publication figures from the sealed equal-budget report only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/ml/equal-budget-final-v1")
REPORT = ROOT / "final_report.json"
FIGURES = ROOT / "figures"
CONDITIONS = ("canonical", "pixel-scale-004", "pixel-scale-006", "exposure-half", "read-noise-double")
LABELS = ("Canonical", "Pixel 0.04", "Pixel 0.06", "Half exposure", "2x read noise")
ARMS = ("C", "G", "R")
COLORS = {"C": "#4C78A8", "G": "#F58518", "R": "#54A24B"}


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_mae(report):
    values = np.array([[report["metrics_by_arm_condition"][arm][condition]["mae_arcsec"]
                        for condition in CONDITIONS] for arm in ARMS])
    x = np.arange(len(CONDITIONS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for index, arm in enumerate(ARMS):
        ax.bar(x + (index - 1) * width, values[index], width,
               label={"C": "C: canonical", "G": "G: uniform variation",
                      "R": "R: failure-weighted"}[arm], color=COLORS[arm])
    ax.set_xticks(x, LABELS, rotation=18, ha="right")
    ax.set_ylabel("MAE (arcsec)")
    ax.set_title("Equal-budget performance by observation condition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "equal_budget_mae_by_condition")


def plot_contrasts(report):
    names = ("G_vs_C_four_stress", "R_vs_G_four_stress", "R_vs_C_four_stress")
    labels = ("G − C", "R − G", "R − C")
    values = np.array([report["contrasts"][name]["difference_arcsec"] for name in names])
    lows = np.array([report["contrasts"][name]["bootstrap_95_low_arcsec"] for name in names])
    highs = np.array([report["contrasts"][name]["bootstrap_95_high_arcsec"] for name in names])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.errorbar(values, y, xerr=[values - lows, highs - values], fmt="o",
                color="#222222", ecolor="#222222", capsize=4, markersize=6,
                linewidth=1.5)
    ax.axvline(0.0, color="#888888", linewidth=1)
    ax.axvline(-0.005, color="#B22222", linewidth=1, linestyle="--",
               label="−0.005 arcsec reference margin")
    ax.set_yticks(y, labels)
    ax.set_xlabel("MAE change (candidate − comparator, arcsec)")
    ax.set_title("Parent-bootstrap contrasts (95% intervals)")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "equal_budget_paired_contrasts")


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("schema") != "equal_budget_final_report_v1":
        raise ValueError("Unexpected equal-budget report schema")
    plot_mae(report)
    plot_contrasts(report)
    print(json.dumps({"figures": [
        str(FIGURES / "equal_budget_mae_by_condition.png"),
        str(FIGURES / "equal_budget_paired_contrasts.png"),
    ]}, indent=2))


if __name__ == "__main__":
    main()
