"""Plot saved development metrics; no inference or label joins are performed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path("results/reports/matched-budget-validation-v1")
    metrics = pd.read_csv(root / "ensemble_metrics.csv")
    summary = json.loads((root / "summary.json").read_text())
    conditions = ["canonical", "pixel-scale-004", "pixel-scale-006", "exposure-half", "read-noise-double"]
    labels = ["Standard", "Pixel scale\n0.04", "Pixel scale\n0.06", "Half\nexposure", "Double\nread noise"]
    table = metrics.pivot(index="condition", columns="arm", values="mae_arcsec").loc[conditions]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [3, 1.25]})
    x = np.arange(len(conditions))
    axes[0].bar(x - .18, table.canonical, .36, label="Standard-condition training", color="#34699a")
    axes[0].bar(x + .18, table.randomized, .36, label="Varied-condition training", color="#e09a36")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean absolute error (arcsec)")
    axes[0].set_ylim(0, .19)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)
    mean = summary["stress_paired_delta_mae_arcsec"]
    low, high = summary["descriptive_parent_bootstrap_95_interval"]
    axes[1].errorbar([mean], [0], xerr=[[mean - low], [high - mean]], fmt="o", color="#34699a", capsize=5)
    axes[1].axvline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_xlim(-.03, .005)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Paired MAE change (arcsec)")
    axes[1].set_title("Four-stress average\nRandomized minus standard", fontsize=10)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Matched training budget: source-validation performance", fontsize=13)
    fig.text(.5, .015, "Development only: 150 parents, three-seed ensembles, 880 updates per model. The 95% interval resamples parents; it excludes seed uncertainty.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94))
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"matched_validation.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
