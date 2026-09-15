"""Exploratory image-astrometry comparison for preselected real lenses.

This module deliberately does not call its result a full lens fit.  It measures
the radius of the brightest lensed-image configuration and compares that simple
monopole diagnostic with compatible published radii.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter, maximum_filter


def _circle_radius(points):
    points = np.asarray(points, dtype=float)
    if points.shape == (2, 2):
        center = points.mean(axis=0)
        radii = np.linalg.norm(points - center, axis=1)
        return center, radii
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError("At least two two-dimensional points are required")
    design = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))))
    target = np.square(points).sum(axis=1)
    solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError("Peak positions do not define a circle")
    center = solution[:2]
    radii = np.linalg.norm(points - center, axis=1)
    return center, radii


def measure_peak_radius(
    science,
    *,
    image_count,
    pixel_scale_arcsec,
    invalid_mask=None,
    gaussian_sigma_pixels=1.25,
    minimum_peak_separation_pixels=9,
):
    """Measure a two-image half-separation or multi-image best-fit circle."""

    science = np.asarray(science, dtype=float)
    if science.ndim != 2:
        raise ValueError("science must be a two-dimensional image")
    image_count = int(image_count)
    if image_count < 2:
        raise ValueError("image_count must be at least two")
    valid = np.isfinite(science)
    if invalid_mask is not None:
        valid &= ~np.asarray(invalid_mask, dtype=bool)
    if valid.sum() < image_count:
        raise ValueError("Not enough valid pixels to locate peaks")
    fill = float(np.nanmedian(science[valid]))
    prepared = np.where(valid, science, fill)
    smoothed = gaussian_filter(prepared, float(gaussian_sigma_pixels))
    size = max(3, int(minimum_peak_separation_pixels))
    if size % 2 == 0:
        size += 1
    local_maximum = smoothed == maximum_filter(smoothed, size=size, mode="nearest")
    candidates_y, candidates_x = np.where(local_maximum & valid)
    values = smoothed[candidates_y, candidates_x]
    if len(values) < image_count:
        raise ValueError(f"Found only {len(values)} local maxima for {image_count} images")
    selected = np.argsort(values)[::-1][:image_count]
    points = np.column_stack((candidates_x[selected], candidates_y[selected])).astype(float)
    peak_values = values[selected]
    center, radii_pixels = _circle_radius(points)
    radius_arcsec = float(radii_pixels.mean() * float(pixel_scale_arcsec))
    radial_scatter_arcsec = float(
        np.sqrt(np.mean(np.square(radii_pixels - radii_pixels.mean()))) * float(pixel_scale_arcsec)
    )
    diagnostic_scale_arcsec = max(float(pixel_scale_arcsec) / 2.0, radial_scatter_arcsec)
    return {
        "center_x_pixel": float(center[0]),
        "center_y_pixel": float(center[1]),
        "radius_arcsec": radius_arcsec,
        "radial_scatter_arcsec": radial_scatter_arcsec,
        "diagnostic_scale_arcsec": diagnostic_scale_arcsec,
        "peak_x_pixels": points[:, 0].tolist(),
        "peak_y_pixels": points[:, 1].tolist(),
        "peak_values": peak_values.astype(float).tolist(),
        "smoothed": smoothed,
    }


def _plot_diagnostic(science, measurement, output_path, lens_id):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.5, 5.5))
    finite = np.asarray(science)[np.isfinite(science)]
    low, high = np.percentile(finite, [1.0, 99.7])
    axis.imshow(science, origin="lower", cmap="magma", vmin=low, vmax=high)
    axis.scatter(
        measurement["peak_x_pixels"],
        measurement["peak_y_pixels"],
        s=80,
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.5,
        label="selected lensed-image peaks",
    )
    radius_pixels = measurement["radius_arcsec"] / measurement["pixel_scale_arcsec"]
    circle = plt.Circle(
        (measurement["center_x_pixel"], measurement["center_y_pixel"]),
        radius_pixels,
        fill=False,
        color="white",
        linestyle="--",
        linewidth=1.2,
        label="monopole radius",
    )
    axis.add_patch(circle)
    axis.set(title=f"{lens_id}: exploratory peak astrometry", xlabel="x (pixel)", ylabel="y (pixel)")
    axis.legend(loc="upper right", fontsize=7)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_real_comparison(config_path, output_dir):
    """Run the frozen three-system comparison and write auditable evidence."""

    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(config["manifest"])
    rows = []
    details = []
    for system in config["systems"]:
        matches = manifest[manifest["sample_id"].astype(str) == str(system["sample_id"])]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one manifest row for {system['sample_id']}")
        source = matches.iloc[0]
        science = np.load(source["science_path"])
        invalid = np.load(source["mask_path"])
        measurement = measure_peak_radius(
            science,
            image_count=system["expected_image_count"],
            pixel_scale_arcsec=source["pixel_scale"],
            invalid_mask=invalid,
            gaussian_sigma_pixels=config["gaussian_sigma_pixels"],
            minimum_peak_separation_pixels=config["minimum_peak_separation_pixels"],
        )
        measurement["pixel_scale_arcsec"] = float(source["pixel_scale"])
        reference = float(system["reference_radius_arcsec"])
        relative_difference = abs(measurement["radius_arcsec"] - reference) / reference
        agreement = relative_difference <= float(config["agreement_threshold_fraction"])
        row = {
            "comparison_id": config["comparison_id"],
            "sample_id": system["sample_id"],
            "lens_id": system["lens_id"],
            "expected_image_count": int(system["expected_image_count"]),
            "detected_peak_count": len(measurement["peak_x_pixels"]),
            "astrometric_radius_arcsec": measurement["radius_arcsec"],
            "diagnostic_scale_arcsec": measurement["diagnostic_scale_arcsec"],
            "reference_radius_arcsec": reference,
            "reference_uncertainty_arcsec": system.get("reference_uncertainty_arcsec"),
            "reference_kind": system["reference_kind"],
            "reference_url": system["reference_url"],
            "relative_difference": relative_difference,
            "within_configured_15_percent": bool(agreement),
        }
        rows.append(row)
        serializable_measurement = {key: value for key, value in measurement.items() if key != "smoothed"}
        details.append({**row, "measurement": serializable_measurement})
        _plot_diagnostic(
            science,
            measurement,
            output_dir / "diagnostics" / f"{system['lens_id']}.png",
            system["lens_id"],
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "comparison.csv", index=False)
    report = {
        "comparison_id": config["comparison_id"],
        "config_path": config_path.as_posix(),
        "method": config["method"],
        "system_count": int(len(frame)),
        "agreement_count": int(frame["within_configured_15_percent"].sum()),
        "maximum_relative_difference": float(frame["relative_difference"].max()),
        "passed": bool(frame["within_configured_15_percent"].all()),
        "claim_boundary": (
            "Exploratory peak astrometry reproduces the scale of preselected real lenses. "
            "This is not a calibrated pixel-level mass fit, posterior, or unbiased accuracy estimate."
        ),
        "limitations": [
            "The three systems were preselected and are not a population-level validation sample.",
            "Brightest-peak selection assumes the lensed images outrank lens-galaxy and field peaks.",
            "The diagnostic scale combines radial scatter with a half-pixel floor; it is not a confidence interval.",
            "HE0435 uses CASTLES Size/2 as a screening proxy rather than a like-for-like SIE critical radius.",
            "Real products have a toy variance model and no empirical PSF, so likelihood-based claims are deferred.",
        ],
        "systems": details,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Real-lens astrometric comparison",
        "",
        report["claim_boundary"],
        "",
        "| Lens | Astrometric radius | Reference | Difference | Within 15% |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['lens_id']} | {row['astrometric_radius_arcsec']:.3f}\" | "
            f"{row['reference_radius_arcsec']:.3f}\" | {row['relative_difference']:.1%} | "
            f"{'yes' if row['within_configured_15_percent'] else 'no'} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in report["limitations"])
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
