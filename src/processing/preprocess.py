import argparse
import json
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd

from src.processing.geometry import (
    background_exclusion_mask,
    center_crop_or_pad_with_transform,
    coverage_mask,
    invalid_pixel_mask,
)
from src.processing.background import estimate_scalar_background
from src.processing.normalization import normalize_image
from src.processing.noise import estimate_scalar_variance
from src.processing.preprocessing_config import config_hash, load_config


METADATA_PATH = Path("data/labels/metadata.csv")
OUTPUT_DIR = Path("data/processed/normalized")
MANIFEST_PATH = Path("data/labels/processed_metadata.csv")
ENRICHED_METADATA_PATH = Path("data/labels/metadata_enriched.csv")

TARGET_SHAPE = (256, 256)

LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0
CODE_VERSION = "2.0.0"


def load_fits_image(path):
    with fits.open(path) as hdul:
        science_hdu = next(
            (
                hdu
                for hdu in hdul
                if str(hdu.name).upper() in {"SCI", "SCIENCE"}
                and hdu.data is not None
                and np.asarray(hdu.data).ndim == 2
            ),
            hdul[0],
        )
        if science_hdu.data is None or np.asarray(science_hdu.data).ndim != 2:
            raise ValueError(f"No science image data found in {path}")
        return np.asarray(science_hdu.data, dtype=np.float32)


def remove_bad_pixels(image):
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def percentile_normalize(image, lower=LOWER_PERCENTILE, upper=UPPER_PERCENTILE):
    low, high = np.percentile(image, [lower, upper])

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(image, dtype=np.float32), low, high

    clipped = np.clip(image, low, high)
    normalized = (clipped - low) / (high - low)
    return normalized.astype(np.float32), low, high


def center_crop_or_pad(image, target_shape=TARGET_SHAPE):
    """Center-crop or zero-pad an image to the canonical shape."""
    output, _ = center_crop_or_pad_with_transform(
        np.asarray(image, dtype=np.float32), target_shape, pad_value=0.0
    )
    return output


def preprocess_image(
    path,
    target_shape=TARGET_SHAPE,
    normalization_method="masked_percentile",
    lower=LOWER_PERCENTILE,
    upper=UPPER_PERCENTILE,
    background_sigma_clip=4.0,
    background_max_iterations=5,
    background_min_valid_pixels=64,
    image_unit="unknown",
    noise_floor=1e-6,
    noise_correlation_assumption="independent_pixels_approximation",
):
    raw = load_fits_image(path)
    cleaned = remove_bad_pixels(raw)
    standardized, transform = center_crop_or_pad_with_transform(cleaned, target_shape, pad_value=0.0)
    invalid = transform.apply(invalid_pixel_mask(raw), pad_value=True)
    coverage = coverage_mask(raw.shape, transform)
    background_mask = background_exclusion_mask(invalid, coverage)
    background = estimate_scalar_background(
        standardized,
        exclusion_mask=background_mask,
        sigma_clip=background_sigma_clip,
        max_iterations=background_max_iterations,
        min_valid_pixels=background_min_valid_pixels,
    )
    background_mask = background.exclusion_mask
    if background.status == "pass" and np.isfinite(background.value):
        science = standardized - background.value
    else:
        science = standardized.copy()
    # Source/outlier pixels are excluded from estimating the background, but they
    # remain valid science pixels. Only non-finite input and missing padded
    # coverage are invalid for normalization and variance products.
    valid_observation = ~invalid & ~coverage
    normalization = normalize_image(
        science,
        valid_mask=valid_observation,
        method=normalization_method,
        lower=lower,
        upper=upper,
        invalid_fill=0.0,
    )
    variance = estimate_scalar_variance(
        science,
        valid_mask=valid_observation,
        background_scale=background.scale,
        image_unit=image_unit,
        noise_floor=noise_floor,
        correlation_assumption=noise_correlation_assumption,
    )

    return {
        "raw": raw,
        "cleaned": cleaned,
        "science": science.astype(np.float32),
        "normalized": normalization.values.astype(np.float32),
        "invalid_mask": invalid,
        "coverage_mask": coverage,
        "background_mask": background_mask,
        "background": background,
        "geometry_transform": transform,
        "clip_low": normalization.clip_low,
        "clip_high": normalization.clip_high,
        "normalization": normalization,
        "variance": variance,
    }


def _row_value(row, primary, fallback=None, default=np.nan):
    """Read a row field without evaluating an absent fallback column eagerly."""
    if isinstance(row, dict):
        if primary in row:
            return row[primary]
        if fallback is not None and fallback in row:
            return row[fallback]
        return default
    if hasattr(row, primary):
        return getattr(row, primary)
    if fallback is not None and hasattr(row, fallback):
        return getattr(row, fallback)
    return default


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _set_default(row, key, value):
    if _is_missing(row.get(key)):
        row[key] = value


def _write_run_manifest(path, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(
    metadata_path=None,
    output_dir=OUTPUT_DIR,
    manifest_path=MANIFEST_PATH,
    config_path=None,
    dry_run=False,
    run_manifest_path=None,
    overwrite=None,
):
    config_path = Path(config_path) if config_path is not None else Path("configs/preprocessing_v2.yaml")
    config = load_config(config_path)
    configuration_hash = config_hash(config)
    metadata_path = Path(metadata_path) if metadata_path is not None else (
        ENRICHED_METADATA_PATH if ENRICHED_METADATA_PATH.exists() else METADATA_PATH
    )
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    metadata = pd.read_csv(metadata_path)

    required_columns = {"sample_id", "raw_path"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Missing required metadata columns: {sorted(missing_columns)}")

    product_root = output_dir.parent if output_dir.name == "normalized" else output_dir
    normalized_output_dir = output_dir if output_dir.name == "normalized" else product_root / config["outputs"]["normalized_subdir"]
    science_output_dir = product_root / config["outputs"]["science_subdir"]
    mask_output_dir = product_root / config["outputs"]["mask_subdir"]
    variance_output_dir = product_root / config["outputs"]["variance_subdir"]
    geometry_output_dir = product_root / config["outputs"]["geometry_subdir"]
    background_output_dir = product_root / config["outputs"]["background_subdir"]
    transform_output_dir = product_root / config["outputs"]["transform_subdir"]
    noise_output_dir = product_root / config["outputs"]["noise_subdir"]
    overwrite = config["outputs"]["overwrite"] if overwrite is None else overwrite
    planned = [
        {
            "sample_id": row["sample_id"],
            "input_path": Path(row["raw_path"]).as_posix(),
            "normalized_path": (normalized_output_dir / f"{row['sample_id']}.npy").as_posix(),
            "science_path": (science_output_dir / f"{row['sample_id']}.npy").as_posix(),
            "invalid_mask_path": (mask_output_dir / f"{row['sample_id']}_invalid.npy").as_posix(),
            "coverage_mask_path": (mask_output_dir / f"{row['sample_id']}_coverage.npy").as_posix(),
            "background_mask_path": (mask_output_dir / f"{row['sample_id']}_background.npy").as_posix(),
            "variance_path": (variance_output_dir / f"{row['sample_id']}.npy").as_posix(),
            "noise_model_path": (noise_output_dir / f"{row['sample_id']}.json").as_posix(),
            "geometry_transform_path": (geometry_output_dir / f"{row['sample_id']}.json").as_posix(),
            "background_path": (background_output_dir / f"{row['sample_id']}.json").as_posix(),
            "normalization_path": (transform_output_dir / f"{row['sample_id']}.json").as_posix(),
        }
        for row in metadata.to_dict(orient="records")
    ]
    if dry_run:
        summary = {
            "run_id": f"preprocess-{configuration_hash[:12]}",
            "config_path": config_path.as_posix(),
            "config_hash": configuration_hash,
            "code_version": CODE_VERSION,
            "metadata_path": metadata_path.as_posix(),
            "input_root": metadata_path.parent.as_posix(),
            "output_root": output_dir.as_posix(),
            "manifest_path": manifest_path.as_posix(),
            "dry_run": True,
            "planned_count": len(planned),
            "accepted_count": 0,
            "failed_count": 0,
            "planned_outputs": planned,
            "failures": [],
        }
        if run_manifest_path is not None:
            _write_run_manifest(run_manifest_path, summary)
        return summary

    for directory in (normalized_output_dir, science_output_dir, mask_output_dir, variance_output_dir, geometry_output_dir, background_output_dir, transform_output_dir, noise_output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []

    for input_row in metadata.to_dict(orient="records"):
        row = dict(input_row)
        sample_id = row["sample_id"]
        raw_path = Path(row["raw_path"])
        output_path = normalized_output_dir / f"{sample_id}.npy"
        science_path = science_output_dir / f"{sample_id}.npy"
        invalid_mask_path = mask_output_dir / f"{sample_id}_invalid.npy"
        coverage_mask_path = mask_output_dir / f"{sample_id}_coverage.npy"
        background_mask_path = mask_output_dir / f"{sample_id}_background.npy"
        geometry_path = geometry_output_dir / f"{sample_id}.json"
        background_path = background_output_dir / f"{sample_id}.json"
        normalization_path = transform_output_dir / f"{sample_id}.json"
        variance_path = variance_output_dir / f"{sample_id}.npy"
        noise_model_path = noise_output_dir / f"{sample_id}.json"

        try:
            product_paths = (output_path, science_path, invalid_mask_path, coverage_mask_path, background_mask_path, geometry_path, background_path, normalization_path, variance_path, noise_model_path)
            existing = next((path for path in product_paths if path.exists()), None)
            if existing is not None and not overwrite:
                raise FileExistsError(f"Output exists and overwrite is disabled: {existing}")
            image_unit = row.get("image_unit", row.get("bunit", "unknown"))
            if _is_missing(image_unit):
                image_unit = "unknown"
            processed = preprocess_image(
                raw_path,
                target_shape=tuple(config["geometry"]["target_shape"]),
                normalization_method=config["normalization"]["method"],
                lower=config["normalization"].get("lower_percentile", LOWER_PERCENTILE),
                upper=config["normalization"].get("upper_percentile", UPPER_PERCENTILE),
                background_sigma_clip=config["background"].get("sigma_clip", 4.0),
                background_max_iterations=config["background"].get("max_iterations", 5),
                background_min_valid_pixels=config["background"].get("min_valid_pixels", 64),
                image_unit=image_unit,
                noise_floor=config["variance"].get("noise_floor", 1e-6),
                noise_correlation_assumption=config["variance"].get("correlation_assumption", "unknown"),
            )
            raw = processed["raw"]
            cleaned = processed["cleaned"]
            normalized = processed["normalized"]

            np.save(output_path, normalized)
            np.save(science_path, processed["science"])
            np.save(invalid_mask_path, processed["invalid_mask"])
            np.save(coverage_mask_path, processed["coverage_mask"])
            np.save(background_mask_path, processed["background_mask"])
            geometry_path.write_text(json.dumps(processed["geometry_transform"].to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            background_path.write_text(json.dumps(processed["background"].to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            normalization_path.write_text(json.dumps(processed["normalization"].to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            np.save(variance_path, processed["variance"].values)
            noise_model_path.write_text(json.dumps(processed["variance"].to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error_type": type(exc).__name__, "message": str(exc)})
            continue

        row["lens_id"] = _row_value(row, "lens_id", "lens")
        row["raw_path"] = raw_path.as_posix()
        row["science_path"] = science_path.as_posix()
        row["processed_path"] = output_path.as_posix()
        row["mask_path"] = invalid_mask_path.as_posix()
        row["coverage_mask_path"] = coverage_mask_path.as_posix()
        row["background_mask_path"] = background_mask_path.as_posix()
        row["geometry_transform_path"] = geometry_path.as_posix()
        row["background_path"] = background_path.as_posix()
        row["normalization_path"] = normalization_path.as_posix()
        row["variance_path"] = variance_path.as_posix()
        row["noise_model_path"] = noise_model_path.as_posix()
        row["mask_semantics"] = "true means excluded"
        row["masked_fraction"] = float(processed["background_mask"].mean())
        row["background_value"] = processed["background"].value
        row["background_scale"] = processed["background"].scale
        row["background_valid_pixels"] = processed["background"].valid_pixels
        row["background_source_pixels"] = processed["background"].source_pixels
        row["background_method"] = processed["background"].method
        row["background_status"] = processed["background"].status
        row["normalization_method"] = processed["normalization"].method
        row["normalization_status"] = processed["normalization"].status
        row["normalization_offset"] = processed["normalization"].offset
        row["normalization_scale"] = processed["normalization"].scale
        row["normalization_invertible"] = processed["normalization"].invertible
        row["variance_method"] = processed["variance"].method
        row["variance_status"] = processed["variance"].status
        row["variance_unit"] = processed["variance"].unit
        row["noise_correlation_assumption"] = processed["variance"].correlation_assumption
        row["original_shape"] = "x".join(str(value) for value in raw.shape)
        row["processed_shape"] = "x".join(str(value) for value in normalized.shape)
        row.update(
            {
                "raw_min": float(np.nanmin(raw)),
                "raw_max": float(np.nanmax(raw)),
                "raw_mean": float(np.nanmean(raw)),
                "cleaned_min": float(cleaned.min()),
                "cleaned_max": float(cleaned.max()),
                "cleaned_mean": float(cleaned.mean()),
                "science_min": float(np.nanmin(processed["science"])),
                "science_max": float(np.nanmax(processed["science"])),
                "science_mean": float(np.nanmean(processed["science"])),
                "normalized_min": float(normalized.min()),
                "normalized_max": float(normalized.max()),
                "normalized_mean": float(normalized.mean()),
                "clip_low_percentile": config["normalization"].get("lower_percentile", LOWER_PERCENTILE),
                "clip_high_percentile": config["normalization"].get("upper_percentile", UPPER_PERCENTILE),
                "clip_low_value": float(processed["clip_low"]),
                "clip_high_value": float(processed["clip_high"]),
            }
        )
        _set_default(row, "domain", "real")
        _set_default(row, "schema_version", "2.0")
        _set_default(row, "dataset_version", "real-dev-v2")
        image_unit = row.get("bunit")
        if _is_missing(image_unit):
            image_unit = "unknown"
        _set_default(row, "image_unit", image_unit)
        _set_default(row, "calibration_status", "unknown")
        _set_default(row, "processing_config_id", f"preprocessing-v2-{configuration_hash[:12]}")
        _set_default(row, "processing_version", config["preprocessing_version"])
        _set_default(row, "background_status", "unknown")
        _set_default(row, "source_type", "unknown")
        _set_default(row, "mass_target_available", False)
        _set_default(row, "theta_E_target_available", False)
        _set_default(row, "structural_status", "pass")
        _set_default(row, "observation_status", "review")
        _set_default(row, "science_status", "review")
        _set_default(row, "eligible_for", "preprocessing")
        _set_default(row, "exclusion_reason", None)
        rows.append(row)

    manifest = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(metadata.columns) + ["processed_path"])

    if rows and manifest["sample_id"].duplicated().any():
        duplicates = manifest.loc[manifest["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate processed sample IDs found: {duplicates[:5]}")

    manifest.to_csv(manifest_path, index=False)
    summary = {
        "run_id": f"preprocess-{configuration_hash[:12]}",
        "config_path": config_path.as_posix(),
        "config_hash": configuration_hash,
        "code_version": CODE_VERSION,
        "metadata_path": metadata_path.as_posix(),
        "input_root": metadata_path.parent.as_posix(),
        "output_root": output_dir.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "dry_run": False,
        "overwrite": bool(overwrite),
        "planned_count": len(planned),
        "accepted_count": len(manifest),
        "failed_count": len(failures),
        "failures": failures,
    }
    run_manifest_path = Path(run_manifest_path) if run_manifest_path is not None else manifest_path.with_suffix(".run.json")
    _write_run_manifest(run_manifest_path, summary)

    print(f"Processed {len(manifest)} images")
    print(f"Saved arrays to {normalized_output_dir}")
    print(f"Saved manifest to {manifest_path}")
    print(f"Saved run manifest to {run_manifest_path}")
    return summary


def _cli():
    parser = argparse.ArgumentParser(description="Run the configured preprocessing pipeline.")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--config", default="configs/preprocessing_v2.yaml")
    parser.add_argument("--run-manifest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", default=None)
    args = parser.parse_args()
    summary = main(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        config_path=args.config,
        dry_run=args.dry_run,
        run_manifest_path=args.run_manifest,
        overwrite=args.overwrite,
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
