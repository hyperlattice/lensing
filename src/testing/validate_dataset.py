from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST_PATH = Path("data/labels/processed_metadata.csv")
REQUIRED_COLUMNS = {
    "sample_id", "lens_id", "raw_path", "processed_path", "pixel_scale",
}


def validate_manifest(manifest_path=MANIFEST_PATH):
    manifest = pd.read_csv(manifest_path)
    errors = []
    missing_columns = REQUIRED_COLUMNS.difference(manifest.columns)
    if missing_columns:
        errors.append(f"missing columns: {sorted(missing_columns)}")
        return errors

    if manifest["sample_id"].duplicated().any():
        errors.append("duplicate sample_id values")

    for row in manifest.itertuples(index=False):
        raw_path = Path(row.raw_path)
        processed_path = Path(row.processed_path)
        if not raw_path.exists():
            errors.append(f"{row.sample_id}: missing raw file {raw_path}")
        if not processed_path.exists():
            errors.append(f"{row.sample_id}: missing processed file {processed_path}")
            continue
        array = np.load(processed_path)
        if array.shape != (256, 256):
            errors.append(f"{row.sample_id}: shape is {array.shape}, expected (256, 256)")
        if array.dtype != np.float32:
            errors.append(f"{row.sample_id}: dtype is {array.dtype}, expected float32")
        if not np.isfinite(array).all():
            errors.append(f"{row.sample_id}: contains NaN or infinity")

        lens_redshift = getattr(row, "lens_redshift", np.nan)
        source_redshift = getattr(row, "source_redshift", np.nan)
        if pd.notna(lens_redshift) and pd.notna(source_redshift):
            if not 0 < lens_redshift < source_redshift:
                errors.append(f"{row.sample_id}: invalid redshift ordering")
    return errors


if __name__ == "__main__":
    validation_errors = validate_manifest()
    if validation_errors:
        print("Dataset validation failed:")
        print("\n".join(f"- {error}" for error in validation_errors))
        raise SystemExit(1)
    print(f"Dataset validation passed: {MANIFEST_PATH}")