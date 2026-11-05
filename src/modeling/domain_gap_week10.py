"""Week 10 descriptive comparison of synthetic training images and CASTLES products."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


CONFIG = Path("configs/reliability/domain_gap.yaml")
REAL_METADATA_COLUMNS = [
    "sample_id", "lens_id", "raw_path", "telescope", "instrument", "detector", "filter",
    "pixel_scale", "original_height", "original_width", "exposure", "bunit", "calstage",
    "biascorr", "darkcorr", "flatcorr", "crcorr", "photcorr",
]
REAL_PRODUCT_COLUMNS = [
    "sample_id", "processed_path", "original_shape", "processed_shape",
    "clip_low_percentile", "clip_high_percentile",
]
SIMULATION_COLUMNS = [
    "sample_id", "parent_sample_id", "processed_path", "original_shape", "processed_shape",
    "pixel_scale", "psf_fwhm_arcsec", "psf_method", "mass_model", "source_type", "simulator_name",
    "simulator_version", "intervention_id", "normalization_method", "split",
]
ANNULI = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4.0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape(value, fallback):
    if pd.isna(value):
        return tuple(fallback)
    pieces = str(value).lower().replace("x", " ").split()
    if len(pieces) != 2:
        return tuple(fallback)
    return int(pieces[0]), int(pieces[1])


def _centered_coverage(shape, original_shape):
    height, width = shape
    source_height, source_width = original_shape
    valid_height = min(height, source_height)
    valid_width = min(width, source_width)
    y_start = max((height - valid_height) // 2, 0)
    x_start = max((width - valid_width) // 2, 0)
    mask = np.zeros((height, width), dtype=bool)
    mask[y_start:y_start + valid_height, x_start:x_start + valid_width] = True
    cropped_fraction = 0.0
    if source_height * source_width > valid_height * valid_width:
        cropped_fraction = 1.0 - (valid_height * valid_width) / (source_height * source_width)
    padding_fraction = 1.0 - (valid_height * valid_width) / (height * width)
    return mask, float(padding_fraction), float(cropped_fraction)


def _image_features(image, pixel_scale, valid):
    image = np.asarray(image, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(image)
    values = image[valid]
    if values.size < 16:
        raise ValueError("fewer than 16 valid pixels")
    q01, q50, q99 = np.quantile(values, [0.01, 0.50, 0.99])
    abs_scale = float(np.quantile(np.abs(values), 0.99))
    if not np.isfinite(abs_scale) or abs_scale <= 0:
        abs_scale = 1.0
    height, width = image.shape
    yy, xx = np.indices(image.shape, dtype=np.float64)
    radius = np.hypot(xx - (width - 1) / 2.0, yy - (height - 1) / 2.0) * float(pixel_scale)
    scaled_abs = np.abs(image) / abs_scale
    radial = {}
    radial_counts = {}
    for lower, upper in ANNULI:
        annulus = valid & (radius >= lower) & (radius < upper)
        key = f"radial_abs_median_{lower:.1f}_{upper:.1f}_arcsec"
        radial[key] = float(np.median(scaled_abs[annulus])) if annulus.any() else np.nan
        radial_counts[f"count_{lower:.1f}_{upper:.1f}_arcsec"] = int(annulus.sum())
    inner = valid & (radius < 0.5)
    aperture = valid & (radius < 2.5)
    outer = valid & (radius >= 2.5)
    inner_sum = float(np.abs(image[inner]).sum())
    aperture_sum = float(np.abs(image[aperture]).sum())
    outer_values = image[outer] / abs_scale
    outer_median = float(np.median(outer_values)) if outer_values.size else np.nan
    outer_mad = float(np.median(np.abs(outer_values - np.median(outer_values)))) if outer_values.size else np.nan
    abs_values = np.abs(values)
    return {
        "valid_pixel_fraction": float(valid.sum() / image.size),
        "finite_pixel_fraction": float(np.isfinite(image).sum() / image.size),
        "intensity_q01": float(q01),
        "intensity_q50": float(q50),
        "intensity_q99": float(q99),
        "intensity_robust_range_q99_q01": float(q99 - q01),
        "intensity_abs_q99_scale": abs_scale,
        "intensity_abs_mad_over_q99": float(np.median(np.abs(abs_values - np.median(abs_values))) / abs_scale),
        "central_concentration_abs_r05_over_r25": float(inner_sum / aperture_sum) if aperture_sum > 0 else np.nan,
        "outer_region_relative_median_brightness": outer_median,
        "outer_region_relative_mad": outer_mad,
        **radial,
        **radial_counts,
    }


def _summaries(observations, fields):
    records = []
    for domain, group in observations.groupby("domain", sort=False):
        for field in fields:
            if field not in group:
                continue
            numeric = pd.to_numeric(group[field], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if numeric.empty:
                continue
            records.append(
                {
                    "domain": domain,
                    "feature": field,
                    "count": int(numeric.size),
                    "unique_parent_or_lens_count": int(
                        group.loc[numeric.index, "cluster_id"].astype(str).nunique()
                    ),
                    "min": float(numeric.min()),
                    "p05": float(numeric.quantile(0.05)),
                    "p25": float(numeric.quantile(0.25)),
                    "median": float(numeric.median()),
                    "p75": float(numeric.quantile(0.75)),
                    "p95": float(numeric.quantile(0.95)),
                    "max": float(numeric.max()),
                    "mean": float(numeric.mean()),
                }
            )
    return pd.DataFrame(records)


def _metadata_quality(real_metadata, real_products, simulation):
    rows = []
    for domain, frame, fields in (
        ("castles_real", real_metadata, REAL_METADATA_COLUMNS),
        ("synthetic_week8_train", simulation, SIMULATION_COLUMNS),
    ):
        for field in fields:
            if field not in frame:
                continue
            values = frame[field]
            rows.append(
                {
                    "domain": domain,
                    "field": field,
                    "row_count": int(len(frame)),
                    "available_count": int(values.notna().sum()),
                    "missing_count": int(values.isna().sum()),
                    "missing_fraction": float(values.isna().mean()),
                    "distinct_nonmissing_count": int(values.dropna().astype(str).nunique()),
                }
            )
    for field in REAL_PRODUCT_COLUMNS:
        if field in real_products:
            values = real_products[field]
            rows.append(
                {
                    "domain": "castles_real_processed_products",
                    "field": field,
                    "row_count": int(len(real_products)),
                    "available_count": int(values.notna().sum()),
                    "missing_count": int(values.isna().sum()),
                    "missing_fraction": float(values.isna().mean()),
                    "distinct_nonmissing_count": int(values.dropna().astype(str).nunique()),
                }
            )
    return pd.DataFrame(rows)


def _coverage_map(observations, config):
    criteria = config["coverage_criteria"]
    summaries = _summaries(observations, ["pixel_scale_arcsec", "processed_field_of_view_x_arcsec", "processed_field_of_view_y_arcsec", "intensity_robust_range_q99_q01", "central_concentration_abs_r05_over_r25"])
    definitions = [
        ("pixel_scale_arcsec", True, "Same WCS-derived angular scale in both domains; synthetic pixel scale was randomized over three levels."),
        ("processed_field_of_view_x_arcsec", True, "Processed-image center-to-center width times pixel scale; this represents the angular field in the retained model input."),
        ("processed_field_of_view_y_arcsec", True, "Processed-image center-to-center height times pixel scale; this represents the angular field in the retained model input."),
        ("intensity_robust_range_q99_q01", False, "Image products use different calibration and normalization semantics; range is descriptive only."),
        ("central_concentration_abs_r05_over_r25", False, "Scale-free image proxy about the fixed cutout center; not lens-centered or a physical morphology label."),
    ]
    rows = []
    for feature, semantics_match, limitation in definitions:
        by_domain = summaries.loc[summaries["feature"] == feature].set_index("domain")
        if not {"castles_real", "synthetic_week8_train"}.issubset(by_domain.index):
            fraction = np.nan
            low = high = np.nan
            synth_min = synth_max = np.nan
        else:
            real_values = pd.to_numeric(
                observations.loc[observations["domain"] == "castles_real", feature], errors="coerce"
            ).dropna().to_numpy(float)
            synth_values = pd.to_numeric(
                observations.loc[observations["domain"] == "synthetic_week8_train", feature], errors="coerce"
            ).dropna().to_numpy(float)
            low, high = np.quantile(synth_values, [0.01, 0.99])
            synth_min, synth_max = np.min(synth_values), np.max(synth_values)
            fraction = float(np.mean((real_values >= low) & (real_values <= high))) if real_values.size else np.nan
        if not np.isfinite(fraction):
            rating = "unsupported"
        elif not semantics_match:
            rating = "weakly represented"
        elif fraction >= float(criteria["well_represented_real_fraction_within_synthetic_p01_p99"]):
            rating = "well represented"
        elif fraction >= float(criteria["partial_overlap_minimum_real_fraction"]):
            rating = "weakly represented"
        else:
            rating = "unsupported"
        rows.append(
            {
                "feature": feature,
                "synthetic_p01": low,
                "synthetic_p99": high,
                "synthetic_min": synth_min,
                "synthetic_max": synth_max,
                "real_fraction_within_synthetic_p01_p99": fraction,
                "measurement_semantics_match": semantics_match,
                "coverage_rating": rating,
                "limitation": limitation,
            }
        )
    rows.extend(
        [
            {"feature": "instrument_filter", "coverage_rating": "unsupported", "measurement_semantics_match": False, "limitation": "CASTLES instrument/filter categories are measured, but the simulator has no instrument- or filter-specific rendering."},
            {"feature": "psf_family_and_calibrated_width", "coverage_rating": "unsupported", "measurement_semantics_match": False, "limitation": "Synthetic PSFs are analytic Gaussian controls; eligible real PSF products/proxies are absent."},
            {"feature": "invalid_and_coverage_masks", "coverage_rating": "unsupported", "measurement_semantics_match": False, "limitation": "Real normalized products have no retained invalid-pixel or coverage masks."},
            {"feature": "calibrated_detector_background_and_noise", "coverage_rating": "unsupported", "measurement_semantics_match": False, "limitation": "Synthetic variance/read-noise settings are toy controls and cannot calibrate HST detector noise."},
            {"feature": "reviewed_image_multiplicity", "coverage_rating": "unsupported", "measurement_semantics_match": False, "limitation": "No reviewed multiplicity field is available in the eligible real metadata release."},
            {"feature": "source_and_lens_morphology_family", "coverage_rating": "weakly represented", "measurement_semantics_match": False, "limitation": "The simulator covers analytic Sersic plus point-source light and SIE mass with optional shear; realistic diversity is not established by radial proxies."},
        ]
    )
    return pd.DataFrame(rows)


def run(config_path=CONFIG):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)

    real_metadata_path = Path(config["real_metadata"])
    real_products_path = Path(config["real_processed_metadata"])
    simulation_split_path = Path(config["simulation_train_split"])
    real_metadata = pd.read_csv(real_metadata_path, usecols=REAL_METADATA_COLUMNS)
    real_products = pd.read_csv(real_products_path, usecols=REAL_PRODUCT_COLUMNS)
    if real_metadata["sample_id"].duplicated().any() or real_products["sample_id"].duplicated().any():
        raise ValueError("CASTLES metadata/product keys must be unique per image")
    real = real_metadata.merge(real_products, on="sample_id", how="inner", validate="one_to_one")
    if len(real) != 137:
        raise ValueError(f"Expected 137 eligible CASTLES products, found {len(real)}")
    simulation = pd.read_csv(simulation_split_path, usecols=SIMULATION_COLUMNS)
    if len(simulation) != 1400 or simulation["parent_sample_id"].nunique() != 700:
        raise ValueError("Week 8 training distribution must contain 1,400 views from 700 parents")
    if set(real["sample_id"].astype(str)) & set(simulation["sample_id"].astype(str)):
        raise ValueError("Real and synthetic image IDs unexpectedly overlap")

    records = []
    product_hash_entries = []
    exclusions = []
    real_root = Path(config["real_products_root"])
    sim_root = Path(config["simulation_release"])
    for domain, frame, root, cluster_field in (
        ("castles_real", real, Path("."), "lens_id"),
        ("synthetic_week8_train", simulation, sim_root, "parent_sample_id"),
    ):
        for row in frame.to_dict(orient="records"):
            sample_id = str(row["sample_id"])
            product_path = Path(str(row["processed_path"]))
            if domain == "castles_real" and not product_path.is_absolute():
                # processed_metadata paths are repository-relative.
                product_path = Path(".") / product_path
            elif domain == "synthetic_week8_train":
                product_path = root / product_path
            try:
                if not product_path.exists():
                    raise FileNotFoundError(str(product_path))
                image = np.load(product_path)
                if image.ndim != 2:
                    raise ValueError(f"not a two-dimensional image: {image.shape}")
                finite = np.isfinite(image)
                pixel_scale = float(row["pixel_scale"])
                if not np.isfinite(pixel_scale) or pixel_scale <= 0:
                    raise ValueError("invalid or missing pixel scale")
                original_shape = _shape(row.get("original_shape"), image.shape)
                coverage, padding_fraction, cropping_fraction = _centered_coverage(image.shape, original_shape)
                valid = finite & coverage
                feature_values = _image_features(image, pixel_scale, valid)
                native_h, native_w = original_shape
                record = {
                    "domain": domain,
                    "sample_id": sample_id,
                    "cluster_id": str(row[cluster_field]),
                    "native_height": int(native_h),
                    "native_width": int(native_w),
                    "processed_height": int(image.shape[0]),
                    "processed_width": int(image.shape[1]),
                    "pixel_scale_arcsec": pixel_scale,
                    "field_of_view_x_arcsec": float(max(native_w - 1, 0) * pixel_scale),
                    "field_of_view_y_arcsec": float(max(native_h - 1, 0) * pixel_scale),
                    "processed_field_of_view_x_arcsec": float(max(image.shape[1] - 1, 0) * pixel_scale),
                    "processed_field_of_view_y_arcsec": float(max(image.shape[0] - 1, 0) * pixel_scale),
                    "padding_fraction_from_shape": padding_fraction,
                    "cropping_fraction_from_shape": cropping_fraction,
                    "product_normalization_method": (
                        f"masked_percentile_{row.get('clip_low_percentile')}_{row.get('clip_high_percentile')}"
                        if domain == "castles_real"
                        else str(row.get("normalization_method", "unknown"))
                    ),
                    **feature_values,
                }
                if domain == "castles_real":
                    for key in ("telescope", "instrument", "detector", "filter", "exposure", "bunit", "calstage", "biascorr", "darkcorr", "flatcorr", "crcorr", "photcorr"):
                        record[key] = row.get(key)
                else:
                    for key in ("psf_fwhm_arcsec", "psf_method", "mass_model", "source_type", "simulator_name", "simulator_version", "intervention_id"):
                        record[key] = row.get(key)
                records.append(record)
                product_hash_entries.append((product_path.as_posix(), _sha256(product_path)))
            except Exception as exc:
                exclusions.append({"domain": domain, "sample_id": sample_id, "reason": str(exc)})

    observations = pd.DataFrame(records)
    if observations.loc[observations["domain"] == "castles_real", "sample_id"].nunique() != 137:
        raise ValueError("At least one required CASTLES image was excluded; inspect exclusions.csv")
    if observations.loc[observations["domain"] == "synthetic_week8_train", "sample_id"].nunique() != 1400:
        raise ValueError("At least one frozen synthetic training image was excluded; inspect exclusions.csv")

    observations.to_csv(output / "per_observation_features.csv", index=False)
    pd.DataFrame(exclusions, columns=["domain", "sample_id", "reason"]).to_csv(output / "exclusions.csv", index=False)
    feature_fields = [
        "pixel_scale_arcsec", "field_of_view_x_arcsec", "field_of_view_y_arcsec",
        "processed_field_of_view_x_arcsec", "processed_field_of_view_y_arcsec", "native_height", "native_width",
        "processed_height", "processed_width", "padding_fraction_from_shape", "cropping_fraction_from_shape",
        "valid_pixel_fraction", "finite_pixel_fraction", "intensity_q01", "intensity_q50", "intensity_q99",
        "intensity_robust_range_q99_q01", "intensity_abs_mad_over_q99", "central_concentration_abs_r05_over_r25",
        "outer_region_relative_median_brightness", "outer_region_relative_mad",
        *[f"radial_abs_median_{lower:.1f}_{upper:.1f}_arcsec" for lower, upper in ANNULI],
        *[f"count_{lower:.1f}_{upper:.1f}_arcsec" for lower, upper in ANNULI],
    ]
    distributions = _summaries(observations, feature_fields)
    distributions.to_csv(output / "feature_distributions.csv", index=False)
    quality = _metadata_quality(real_metadata, real_products, simulation)
    quality.to_csv(output / "metadata_quality.csv", index=False)

    category_rows = []
    for domain, frame, fields in (
        ("castles_real", real_metadata.assign(normalization_method="masked_percentile_1_99"), ("telescope", "instrument", "detector", "filter", "bunit", "calstage", "biascorr", "darkcorr", "flatcorr", "crcorr", "photcorr", "normalization_method")),
        ("synthetic_week8_train", simulation, ("simulator_name", "simulator_version", "mass_model", "source_type", "psf_method", "intervention_id", "normalization_method")),
    ):
        for field in fields:
            counts = frame[field].fillna("<missing>").astype(str).value_counts(dropna=False)
            category_rows.extend({"domain": domain, "field": field, "category": category, "count": int(count)} for category, count in counts.items())
    categories = pd.DataFrame(category_rows)
    categories.to_csv(output / "metadata_category_counts.csv", index=False)

    coverage = _coverage_map(observations, config)
    coverage.to_csv(output / "coverage_map.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for domain, group in observations.groupby("domain", sort=False):
        axes[0].hist(group["pixel_scale_arcsec"], bins=18, alpha=0.55, density=True, label=domain)
        axes[1].hist(group["processed_field_of_view_x_arcsec"], bins=18, alpha=0.55, density=True, label=domain)
    axes[0].set_xlabel("Pixel scale (arcsec/pixel)")
    axes[1].set_xlabel("Processed-image field of view (arcsec)")
    for axis in axes:
        axis.set_ylabel("Density (descriptive)")
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "pixel_scale_field_of_view.png", dpi=180)
    plt.close(fig)

    radial_fields = [f"radial_abs_median_{lower:.1f}_{upper:.1f}_arcsec" for lower, upper in ANNULI]
    radial_labels = [f"{lower:g}–{upper:g}" for lower, upper in ANNULI]
    x_positions = np.arange(len(radial_labels))
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for domain, group in observations.groupby("domain", sort=False):
        medians = group[radial_fields].median(axis=0).to_numpy(float)
        q25 = group[radial_fields].quantile(0.25, axis=0).to_numpy(float)
        q75 = group[radial_fields].quantile(0.75, axis=0).to_numpy(float)
        axis.plot(x_positions, medians, marker="o", label=domain)
        axis.fill_between(x_positions, q25, q75, alpha=0.15)
    axis.set_xticks(x_positions, radial_labels)
    axis.set(xlabel="Radius from fixed cutout center (arcsec)", ylabel="Median |intensity| / per-image |I| P99")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "radial_profile_proxies.png", dpi=180)
    plt.close(fig)

    metadata_missingness = quality.loc[quality["domain"] == "castles_real"].set_index("field")["missing_fraction"].to_dict()
    product_manifest_hash = hashlib.sha256(
        json.dumps(sorted(product_hash_entries), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    sim_run_path = Path(config["simulation_run_manifest"])
    summary = {
        "experiment_id": config["experiment_id"],
        "protocol": config["protocol"],
        "protocol_sha256": _sha256(Path(config["protocol"])),
        "config_sha256": _sha256(config_path),
        "analysis_source_sha256": _sha256(Path(__file__)),
        "real_metadata_sha256": _sha256(real_metadata_path),
        "real_processed_metadata_sha256": _sha256(real_products_path),
        "simulation_train_split_sha256": _sha256(simulation_split_path),
        "simulation_run_manifest_sha256": _sha256(sim_run_path),
        "real_observation_count": int(observations.loc[observations["domain"] == "castles_real", "sample_id"].nunique()),
        "real_lens_system_count": int(observations.loc[observations["domain"] == "castles_real", "cluster_id"].nunique()),
        "synthetic_training_view_count": int(observations.loc[observations["domain"] == "synthetic_week8_train", "sample_id"].nunique()),
        "synthetic_independent_parent_count": int(observations.loc[observations["domain"] == "synthetic_week8_train", "cluster_id"].nunique()),
        "excluded_row_count": int(len(exclusions)),
        "processed_image_product_count": len(product_hash_entries),
        "processed_image_products_manifest_sha256": product_manifest_hash,
        "real_metadata_missing_fraction": metadata_missingness,
        "real_parameter_labels_read": False,
        "real_model_predictions_scored": False,
        "historical_test_accessed": False,
        "toy_variance_interpreted_as_detector_calibration": False,
        "three_system_astrometric_context": "results/reports/real-comparison-v1/comparison.md (exploratory only)",
        "coverage_map_path": str(output / "coverage_map.csv"),
        "limitations": [
            "CASTLES intensity products and synthetic images have different calibration/normalization semantics.",
            "The 137 CASTLES rows are repeated images from 69 lens systems; distribution summaries are descriptive and lens clustered.",
            "No retained CASTLES invalid/coverage masks, eligible PSF products, or reviewed multiplicity labels were available.",
            "Outer-region brightness is a relative image proxy, not calibrated detector background or noise.",
            "Radial proxies use the fixed cutout center, not a fitted or truth lens center.",
            "No real-data model predictions or Einstein-radius/redshift labels were used.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cat = categories.loc[categories["domain"] == "castles_real"]
    def counts_for(field):
        values = cat.loc[cat["field"] == field]
        return ", ".join(f"{row['category']}: {row['count']}" for row in values.to_dict(orient="records"))
    report = [
        "# Week 10 — CASTLES/synthetic domain gap",
        "",
        f"Compared **{summary['synthetic_training_view_count']}** Week 8 training views from **{summary['synthetic_independent_parent_count']}** independent parents with **{summary['real_observation_count']}** CASTLES images from **{summary['real_lens_system_count']}** lens systems.",
        "The analysis uses metadata and image proxies only. It does not score Einstein radii or model predictions on real images.",
        "",
        "## Metadata coverage",
        "",
        f"- Telescope: {counts_for('telescope')}",
        f"- Instrument: {counts_for('instrument')}",
        f"- Filter: {counts_for('filter')}",
        f"- Detector: {counts_for('detector')}",
        f"- Calibration stage: {counts_for('calstage')}",
        "- Missing categories remain explicit in `metadata_category_counts.csv`; repeated images are grouped by lens in the row-level data.",
        "",
        "## Frozen coverage map",
        "",
        "| Feature | Rating | Real within synthetic P01–P99 | Limitation |",
        "|---|---|---:|---|",
    ]
    for row in coverage.to_dict(orient="records"):
        fraction = row.get("real_fraction_within_synthetic_p01_p99")
        formatted = "unavailable" if fraction is None or pd.isna(fraction) else f"{fraction:.3f}"
        report.append(f"| {row['feature']} | {row['coverage_rating']} | {formatted} | {row['limitation']} |")
    report.extend(
        [
            "",
            "![Pixel scale and field of view](pixel_scale_field_of_view.png)",
            "",
            "![Radial image proxies](radial_profile_proxies.png)",
            "",
            "Intensity ranges are shown only as within-product descriptive values: CASTLES products are percentile normalized, while synthetic products follow the simulation processing path. The outer-region proxy is not detector-background calibration. Neither the synthetic variance model nor read-noise parameter is treated as calibrated HST noise.",
            "",
            "The existing [three-system astrometric comparison](../real-comparison-v1/comparison.md) is exploratory context only, not a general accuracy benchmark.",
            "",
            "## Requirements for a future blinded real benchmark",
            "",
            "1. Define eligibility from reviewed image quality and publish a lens-system-level split before model scoring.",
            "2. Obtain independent, uncertainty-qualified Einstein-radius references and document model/decomposition provenance; do not treat a single heterogeneous catalog value as exact truth.",
            "3. Preserve raw calibrated science, variance, bad-pixel/coverage masks, PSF products, filter/instrument metadata, exposure, and full crop/normalization provenance.",
            "4. Calibrate simulator source/lens morphology, mass-profile, PSF, detector/background/noise, pixel scale, and selection distributions against those products.",
            "5. Freeze checkpoints, preprocessing, eligible lens IDs, metrics, and scorer before predictions. Save and hash predictions before a separate custodian joins labels once.",
            "6. Report system-level paired uncertainty and label/reference uncertainty; keep a separate blind test set unavailable during all model development.",
            "",
            "## Limitations",
            "",
            *[f"- {value}" for value in summary["limitations"]],
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "experiment_id", "real_observation_count", "real_lens_system_count",
        "synthetic_training_view_count", "synthetic_independent_parent_count", "excluded_row_count",
        "real_parameter_labels_read", "historical_test_accessed", "protocol_sha256",
    )}, indent=2, sort_keys=True))
