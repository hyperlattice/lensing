"""Load and validate the versioned preprocessing configuration."""

from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/preprocessing/default.yaml")
REQUIRED_SECTIONS = {"science", "geometry", "background", "mask", "variance", "normalization", "outputs", "run"}
ALLOWED_NORMALIZATIONS = {"masked_percentile", "affine", "none"}


def validate_config(config):
    """Return configuration errors without making filesystem changes."""
    errors = []
    missing_sections = REQUIRED_SECTIONS.difference(config)
    if missing_sections:
        errors.append(f"missing sections: {sorted(missing_sections)}")
    if config.get("schema_version") != "2.0":
        errors.append("schema_version must be 2.0")
    if not str(config.get("preprocessing_version", "")).strip():
        errors.append("preprocessing_version is required")

    geometry = config.get("geometry", {})
    target_shape = geometry.get("target_shape")
    if not isinstance(target_shape, (list, tuple)) or len(target_shape) != 2:
        errors.append("geometry.target_shape must contain height and width")
    elif any(not isinstance(value, int) or value <= 0 for value in target_shape):
        errors.append("geometry.target_shape values must be positive integers")

    normalization = config.get("normalization", {})
    method = normalization.get("method")
    if method not in ALLOWED_NORMALIZATIONS:
        errors.append(f"normalization.method must be one of {sorted(ALLOWED_NORMALIZATIONS)}")
    if method == "masked_percentile":
        lower = normalization.get("lower_percentile")
        upper = normalization.get("upper_percentile")
        if not (isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and 0 <= lower < upper <= 100):
            errors.append("normalization percentiles must satisfy 0 <= lower < upper <= 100")

    outputs = config.get("outputs", {})
    for key in ("normalized_subdir", "science_subdir", "mask_subdir", "variance_subdir", "geometry_subdir", "background_subdir", "transform_subdir", "noise_subdir"):
        value = outputs.get(key)
        if not isinstance(value, str) or not value.strip() or Path(value).is_absolute() or ".." in Path(value).parts:
            errors.append(f"outputs.{key} must be a safe relative directory")
    return errors


def load_config(path=CONFIG_PATH):
    """Read and validate a YAML configuration before processing begins."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    errors = validate_config(config)
    if errors:
        raise ValueError("Invalid preprocessing configuration: " + "; ".join(errors))
    return config
