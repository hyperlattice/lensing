from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd


METADATA_PATH = Path("data/labels/metadata.csv")
OUTPUT_DIR = Path("data/processed/normalized")
MANIFEST_PATH = Path("data/labels/processed_metadata.csv")

LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0


def load_fits_image(path):
    image = fits.getdata(path)
    return np.asarray(image, dtype=np.float32)


def remove_bad_pixels(image):
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def percentile_normalize(image, lower=LOWER_PERCENTILE, upper=UPPER_PERCENTILE):
    low, high = np.percentile(image, [lower, upper])

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(image, dtype=np.float32), low, high

    clipped = np.clip(image, low, high)
    normalized = (clipped - low) / (high - low)
    return normalized.astype(np.float32), low, high


def preprocess_image(path):
    raw = load_fits_image(path)
    cleaned = remove_bad_pixels(raw)
    normalized, clip_low, clip_high = percentile_normalize(cleaned)

    return {
        "raw": raw,
        "cleaned": cleaned,
        "normalized": normalized,
        "clip_low": clip_low,
        "clip_high": clip_high,
    }


def main():
    metadata = pd.read_csv(METADATA_PATH)

    required_columns = {"sample_id", "raw_path"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Missing required metadata columns: {sorted(missing_columns)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for row in metadata.itertuples(index=False):
        raw_path = Path(row.raw_path)
        output_path = OUTPUT_DIR / f"{row.sample_id}.npy"

        processed = preprocess_image(raw_path)
        raw = processed["raw"]
        cleaned = processed["cleaned"]
        normalized = processed["normalized"]

        np.save(output_path, normalized)

        rows.append({
            "sample_id": row.sample_id,
            "raw_path": raw_path.as_posix(),
            "processed_path": output_path.as_posix(),
            "original_shape": "x".join(str(value) for value in raw.shape),
            "processed_shape": "x".join(str(value) for value in normalized.shape),
            "raw_min": float(np.nanmin(raw)),
            "raw_max": float(np.nanmax(raw)),
            "raw_mean": float(np.nanmean(raw)),
            "cleaned_min": float(cleaned.min()),
            "cleaned_max": float(cleaned.max()),
            "cleaned_mean": float(cleaned.mean()),
            "normalized_min": float(normalized.min()),
            "normalized_max": float(normalized.max()),
            "normalized_mean": float(normalized.mean()),
            "clip_low_percentile": LOWER_PERCENTILE,
            "clip_high_percentile": UPPER_PERCENTILE,
            "clip_low_value": float(processed["clip_low"]),
            "clip_high_value": float(processed["clip_high"]),
        })

    manifest = pd.DataFrame(rows)

    if manifest["sample_id"].duplicated().any():
        duplicates = manifest.loc[manifest["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate processed sample IDs found: {duplicates[:5]}")

    manifest.to_csv(MANIFEST_PATH, index=False)

    print(f"Processed {len(manifest)} images")
    print(f"Saved arrays to {OUTPUT_DIR}")
    print(f"Saved manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()