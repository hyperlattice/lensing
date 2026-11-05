"""Create a reviewer-safe PDF showing synthetic lens examples and results.

The PDF contains no raw metadata or local paths. It visualizes normalized
synthetic image products, aggregate equal-budget metrics, and the paper's claim
boundary. It is an explanatory artifact, not a new evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "simulations" / "reliability-blind-holdout-v1"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
REPORT_PATH = ROOT / "results" / "ml" / "equal-budget-final-v1" / "final_report.json"
BUNDLE = ROOT / "submission" / "aaai27" / "reviewer_bundle"
FIGURE_DIR = BUNDLE / "figures"
OUTPUT = FIGURE_DIR / "simulated_lens_examples_and_results.pdf"

CONDITIONS = [
    "canonical",
    "pixel-scale-004",
    "pixel-scale-006",
    "exposure-half",
    "read-noise-double",
]


def load_image(row: pd.Series) -> np.ndarray:
    image = np.load(DATA_ROOT / str(row["processed_path"]))
    image = np.asarray(image, dtype=float).squeeze()
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D image, got {image.shape}")
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def condition_label(condition: str) -> str:
    return {
        "canonical": "canonical\n0.05 arcsec/pixel",
        "pixel-scale-004": "pixel scale\n0.04 arcsec/pixel",
        "pixel-scale-006": "pixel scale\n0.06 arcsec/pixel",
        "exposure-half": "half exposure",
        "read-noise-double": "double read noise",
    }[condition]


def display_limits(images: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([image.ravel() for image in images])
    low, high = np.nanpercentile(values, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
    return float(low), float(high)


def add_image_page(
    pdf: PdfPages,
    images: list[np.ndarray],
    titles: list[str],
    suptitle: str,
    subtitle: str,
) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(11, 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    low, high = display_limits(images)
    image_artist = None
    for axis, image, title in zip(axes, images, titles):
        image_artist = axis.imshow(
            image,
            cmap="magma",
            vmin=low,
            vmax=high,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title, fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle(suptitle, fontsize=13, y=1.04)
    fig.text(0.5, 1.005, subtitle, ha="center", va="bottom", fontsize=8)
    if image_artist is not None:
        fig.colorbar(image_artist, ax=axes.tolist(), shrink=0.72, pad=0.02, label="normalized intensity")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_results_page(pdf: PdfPages, report: dict) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    grid = fig.add_gridspec(3, 1, height_ratios=[0.55, 1.0, 1.25], hspace=0.5)
    fig.suptitle("Equal-budget synthetic robustness results", fontsize=15, y=0.98)
    fig.text(
        0.5,
        0.945,
        "Frozen parent-disjoint holdout; 150 parents; 10 paired seeds; 880 updates per fit",
        ha="center",
        fontsize=9,
    )

    table_axis = fig.add_subplot(grid[0])
    table_axis.axis("off")
    rows = []
    for arm, label in [("C", "canonical"), ("G", "uniform variation"), ("R", "difficult-condition sampling")]:
        metrics = report["metrics_by_arm_condition"][arm]
        stress = np.mean(
            [
                metrics[condition]["mae_arcsec"]
                for condition in ("pixel-scale-004", "pixel-scale-006", "exposure-half", "read-noise-double")
            ]
        )
        rows.append([label, f"{metrics['canonical']['mae_arcsec']:.6f}", f"{stress:.6f}"])
    table = table_axis.table(
        cellText=rows,
        colLabels=["Arm", "Canonical MAE (arcsec)", "Four-stress MAE (arcsec)"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.28, 0.32, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    figure_paths = [
        FIGURE_DIR / "equal_budget_mae_by_condition.png",
        FIGURE_DIR / "equal_budget_paired_contrasts.png",
    ]
    for axis, path in zip([fig.add_subplot(grid[1]), fig.add_subplot(grid[2])], figure_paths):
        axis.imshow(plt.imread(path))
        axis.axis("off")
    fig.text(
        0.08,
        0.025,
        "Primary G-minus-C contrast: -0.029744 arcsec; parent-bootstrap 95% interval "
        "[-0.045725, -0.012141]. R-minus-G is inconclusive. These are synthetic-only results.",
        fontsize=8.5,
        wrap=True,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_method_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("What the study shows and does not show", fontsize=15, y=0.96)
    text = (
        "Synthetic scene: SIE lens + extended Sersic source and lens light + unresolved lensed point source, "
        "Gaussian PSF, and detector/noise model.\n\n"
        "Intervention: render the same latent parent under canonical, pixel-scale, exposure, and read-noise "
        "conditions. Train canonical (C), uniform variation (G), and a source-development policy that samples "
        "difficult conditions more often (R) "
        "arms with identical CNN architecture, initialization, optimizer, update count, and source exposure.\n\n"
        "Leakage control: 697 qualified source parents and a frozen 150-parent holdout; predictions are sealed "
        "before target labels are joined; bootstrap resampling is over parent systems, not rendered views.\n\n"
        "Interpretation: uniform variation improves robustness on the declared synthetic stress matrix. The "
        "difficult-condition policy does not show a reliable extra gain. The AutoLens/Lenstronomy run compares "
        "two image-generation implementations; it is not an independent-simulator test. No real-data accuracy "
        "claim is made."
    )
    fig.text(0.09, 0.86, text, va="top", ha="left", fontsize=11, linespacing=1.55, wrap=True)
    fig.text(
        0.09,
        0.12,
        "Study identifiers: reliability-blind-holdout-v1 / equal-budget-final-v1\n"
        "This PDF is a visual explanatory supplement, not a new scoring run.",
        fontsize=9,
        color="#444444",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    canonical = manifest[manifest["condition"].eq("canonical")].drop_duplicates("parent_system_id")
    if len(canonical) < 5:
        raise RuntimeError("need at least five canonical parent systems")
    reference_parent = str(canonical.iloc[0]["parent_system_id"])
    condition_rows = manifest[
        manifest["parent_system_id"].eq(reference_parent)
        & manifest["condition"].isin(CONDITIONS)
    ].set_index("condition")
    if set(CONDITIONS) != set(condition_rows.index):
        raise RuntimeError("reference parent does not have the complete condition panel")

    with PdfPages(
        OUTPUT,
        metadata={
            "Title": "Synthetic lens examples and equal-budget robustness results",
            "Author": "Anonymous Submission",
            "Subject": "Reviewer-safe synthetic strong-lensing methodology and results",
        },
    ) as pdf:
        same_parent_images = [load_image(condition_rows.loc[condition]) for condition in CONDITIONS]
        add_image_page(
            pdf,
            same_parent_images,
            [condition_label(condition) for condition in CONDITIONS],
            "One simulated lens under five observation conditions",
            "The latent lens/source parameters are held fixed; only the observation condition changes.",
        )

        population_rows = canonical.iloc[:5]
        population_images = [load_image(row) for _, row in population_rows.iterrows()]
        add_image_page(
            pdf,
            population_images,
            [f"Parent {chr(65 + index)}\ncanonical" for index in range(len(population_images))],
            "Representative simulated-lens population",
            "Five independent canonical parents from the synthetic benchmark; displayed with a shared intensity scale.",
        )

        add_results_page(pdf, report)
        add_method_page(pdf)

    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
