"""Machine-readable contract for version 2 dataset manifests.

The CSV manifest is intentionally kept flat for portability.  Large parameter
dictionaries and matrices may live in sidecars, but the fields below are the
stable interface shared by real observations, simulations, validation, and
future model loaders.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping, Sequence


SCHEMA_VERSION = "2.0"
IMAGE_SHAPE = (256, 256)
IMAGE_DTYPE = "float32"
VALID_DOMAINS = frozenset({"real", "simulated"})
VALID_STATUSES = frozenset({"pass", "review", "fail", "unknown"})
VALID_ELIGIBILITY = frozenset({"preprocessing", "simulation", "classical_fit", "ml", "real_comparison"})


@dataclass(frozen=True)
class FieldSpec:
    """Description of one manifest field."""

    dtype: str
    unit: str
    required: str
    null_behavior: str
    description: str


FIELD_SPECS = {
    "sample_id": FieldSpec("string", "-", "all rows", "forbidden", "Unique observation identifier."),
    "lens_id": FieldSpec("string", "-", "all rows", "forbidden", "Physical lens-system identifier."),
    "domain": FieldSpec("enum", "-", "all rows", "forbidden", "real or simulated."),
    "schema_version": FieldSpec("string", "-", "all rows", "forbidden", "Manifest contract version."),
    "dataset_version": FieldSpec("string", "-", "all rows", "forbidden", "Frozen dataset/release identifier."),
    "simulation_id": FieldSpec("string", "-", "simulated rows", "null for real", "Simulation run identifier."),
    "parent_system_id": FieldSpec("string", "-", "simulated rows", "null for real", "Independent synthetic parent group."),
    "parent_real_lens_id": FieldSpec("string", "-", "optional", "explicitly unknown", "Real lens used as a simulation reference, if any."),
    "source_template_id": FieldSpec("string", "-", "optional", "explicitly unknown", "Source stamp or analytic template family."),
    "background_template_id": FieldSpec("string", "-", "optional", "explicitly unknown", "Background/sky template family."),
    "split_group_id": FieldSpec("string", "-", "simulated rows", "null for real", "Group that must stay in one split."),
    "raw_path": FieldSpec("path", "repository-relative POSIX", "all rows", "forbidden", "Original or generated FITS artifact."),
    "science_path": FieldSpec("path", "repository-relative POSIX", "all rows", "forbidden", "Selected science image product."),
    "processed_path": FieldSpec("path", "repository-relative POSIX", "all rows", "forbidden", "Standardized model-input array."),
    "mask_path": FieldSpec("path", "repository-relative POSIX", "optional", "null when unavailable", "Invalid-pixel or coverage mask."),
    "coverage_mask_path": FieldSpec("path", "repository-relative POSIX", "optional", "null when unavailable", "True marks padded/no-coverage pixels."),
    "background_mask_path": FieldSpec("path", "repository-relative POSIX", "optional", "null when unavailable", "True marks pixels excluded from background estimation."),
    "mask_semantics": FieldSpec("string", "-", "optional", "null when unavailable", "Explicit boolean mask meaning."),
    "variance_path": FieldSpec("path", "repository-relative POSIX", "optional", "null when unavailable", "Per-pixel variance or uncertainty."),
    "psf_path": FieldSpec("path", "repository-relative POSIX", "optional", "null when unavailable", "PSF kernel used or associated with the image."),
    "psf_id": FieldSpec("string", "-", "optional", "null when unavailable", "Shared PSF registry identifier."),
    "science_hdu": FieldSpec("integer", "-", "all rows", "forbidden", "Selected science HDU index."),
    "science_hdu_name": FieldSpec("string", "-", "all rows", "forbidden", "Selected science HDU name."),
    "instrument": FieldSpec("string", "-", "optional", "explicitly unknown", "Telescope/instrument."),
    "detector": FieldSpec("string", "-", "optional", "explicitly unknown", "Detector, camera, or chip."),
    "filter_name": FieldSpec("string", "-", "optional", "explicitly unknown", "Verified observation filter."),
    "exposure_s": FieldSpec("float", "s", "optional", "null when unavailable", "Exposure time."),
    "image_unit": FieldSpec("string", "declared unit", "all rows", "use unknown when unverified", "Pixel-value unit after science-HDU selection."),
    "calibration_status": FieldSpec("enum", "-", "all rows", "use unknown when unverified", "Calibration state and confidence."),
    "calibration_source": FieldSpec("string", "-", "optional", "null when unavailable", "Documentation or header provenance."),
    "original_shape": FieldSpec("string", "px", "all rows", "forbidden", "Stored as HEIGHTxWIDTH before standardization."),
    "processed_shape": FieldSpec("string", "px", "all rows", "forbidden", "Stored as HEIGHTxWIDTH after standardization."),
    "pixel_scale": FieldSpec("float", "arcsec/px", "all rows", "forbidden", "Scalar angular scale used by the model grid."),
    "pixel_scale_x": FieldSpec("float", "arcsec/px", "optional", "null when unavailable", "Per-axis scale along x."),
    "pixel_scale_y": FieldSpec("float", "arcsec/px", "optional", "null when unavailable", "Per-axis scale along y."),
    "pixel_scale_source": FieldSpec("enum", "-", "all rows", "use unknown when unavailable", "WCS or explicit fallback rule used for scale."),
    "wcs_status": FieldSpec("enum", "-", "all rows", "use unknown when unavailable", "Whether the celestial WCS/Jacobian is valid."),
    "wcs_matrix_deg_per_pixel": FieldSpec("JSON sidecar/string", "deg/px", "optional", "null when unavailable", "Full local pixel-to-angle matrix."),
    "center_x_px": FieldSpec("float", "px", "optional", "null when unavailable", "Fractional lens center in image x."),
    "center_y_px": FieldSpec("float", "px", "optional", "null when unavailable", "Fractional lens center in image y."),
    "center_ra_deg": FieldSpec("float", "deg", "optional", "null when unavailable", "Angular lens-center right ascension."),
    "center_dec_deg": FieldSpec("float", "deg", "optional", "null when unavailable", "Angular lens-center declination."),
    "geometry_transform_path": FieldSpec("path", "repository-relative POSIX", "optional", "null when unavailable", "Sidecar containing WCS/Jacobian and crop/pad transform."),
    "processing_config_id": FieldSpec("string", "-", "all rows", "forbidden", "Preprocessing configuration identifier."),
    "processing_version": FieldSpec("string", "-", "all rows", "forbidden", "Preprocessing implementation version."),
    "background_status": FieldSpec("enum", "-", "all rows", "use unknown when unverified", "Background handling status."),
    "normalization_path": FieldSpec("path", "repository-relative POSIX", "optional", "null when unavailable", "Saved normalization transform."),
    "masked_fraction": FieldSpec("float", "fraction", "optional", "null when unavailable", "Fraction of pixels excluded from science/model use."),
    "background_path": FieldSpec("path", "repository-relative POSIX", "optional", "null when unavailable", "Saved scalar-background estimate and diagnostics."),
    "background_value": FieldSpec("float", "science-unit", "optional", "null when unconstrained", "Robust scalar background estimate."),
    "background_scale": FieldSpec("float", "science-unit", "optional", "null when unconstrained", "Robust background scale estimate."),
    "background_valid_pixels": FieldSpec("integer", "count", "optional", "zero when unconstrained", "Pixels used by the background estimator."),
    "background_source_pixels": FieldSpec("integer", "count", "optional", "zero when unconstrained", "Pixels rejected as source/outlier candidates."),
    "background_method": FieldSpec("string", "-", "optional", "null when unavailable", "Background estimator implementation."),
    "normalization_method": FieldSpec("string", "-", "optional", "null when unavailable", "Normalization transform implementation."),
    "normalization_status": FieldSpec("enum", "-", "optional", "unknown when unavailable", "Whether normalization statistics were constrained."),
    "normalization_offset": FieldSpec("float", "science-unit", "optional", "null when unconstrained", "Saved affine offset."),
    "normalization_scale": FieldSpec("float", "science-unit", "optional", "null when unconstrained", "Saved affine scale."),
    "normalization_invertible": FieldSpec("boolean", "-", "optional", "false for clipped transforms", "Whether the transform has an exact inverse on valid pixels."),
    "noise_model_path": FieldSpec("path", "repository-relative POSIX", "optional", "null when unavailable", "Saved variance/noise-model provenance."),
    "variance_method": FieldSpec("string", "-", "optional", "null when unavailable", "Variance estimator or supplied-error conversion."),
    "variance_status": FieldSpec("enum", "-", "optional", "unknown when unavailable", "Whether variance is calibrated, toy, or unconstrained."),
    "variance_unit": FieldSpec("string", "science-unit squared", "optional", "null when unavailable", "Units of the variance array."),
    "noise_correlation_assumption": FieldSpec("string", "-", "optional", "unknown when unavailable", "Pixel-correlation assumption for likelihood use."),
    "mass_model": FieldSpec("string", "named convention", "simulated rows", "null for real", "Mass profile name, such as SIS or SIE."),
    "theta_E_arcsec": FieldSpec("float", "arcsec", "simulated rows", "null for real unless a model truth is defined", "Simulator Einstein-radius parameter under the named convention."),
    "einstein_radius_estimate_arcsec": FieldSpec("float", "arcsec", "optional", "null when unavailable", "Observed/literature radius estimate, never simulator truth by implication."),
    "einstein_radius_method": FieldSpec("string", "-", "optional", "null when unavailable", "Measurement, model, Size/2 proxy, or other derivation."),
    "q": FieldSpec("float", "-", "optional", "null when unavailable", "Projected axis ratio."),
    "phi_rad": FieldSpec("float", "rad", "optional", "null when unavailable", "Mass-profile orientation, counter-clockwise in model coordinates."),
    "e1": FieldSpec("float", "-", "optional", "null when unavailable", "Model ellipticity component under the declared adapter convention."),
    "e2": FieldSpec("float", "-", "optional", "null when unavailable", "Model ellipticity component under the declared adapter convention."),
    "gamma1": FieldSpec("float", "-", "optional", "null when unavailable", "External-shear component 1."),
    "gamma2": FieldSpec("float", "-", "optional", "null when unavailable", "External-shear component 2."),
    "source_type": FieldSpec("string", "-", "all rows", "use unknown when unverified", "Point, Sersic, host+quasar, or other source class."),
    "lens_environment": FieldSpec("enum", "-", "optional", "unknown with reason", "Galaxy-scale, cluster-scale, multiple-deflector, or uncertain."),
    "num_images": FieldSpec("integer", "count", "optional", "unknown when unverified", "Number of resolved lensed images."),
    "lens_confidence": FieldSpec("enum", "-", "optional", "unknown with reason", "Confidence of the curated system classification."),
    "model_suitability": FieldSpec("enum", "-", "optional", "unknown with reason", "Proposed suitability for a named mass model."),
    "ring_extended_flag": FieldSpec("boolean", "-", "optional", "unknown when unreviewed", "Whether extended ring/arc structure is present."),
    "cropped_truncated_flag": FieldSpec("boolean", "-", "optional", "unknown when unreviewed", "Whether the image footprint may truncate relevant structure."),
    "classification_evidence": FieldSpec("string", "-", "optional", "unknown with reason", "Evidence supporting the suitability classification."),
    "source_x_arcsec": FieldSpec("float", "arcsec", "optional", "null when unavailable", "Source-plane x position."),
    "source_y_arcsec": FieldSpec("float", "arcsec", "optional", "null when unavailable", "Source-plane y position."),
    "source_re_arcsec": FieldSpec("float", "arcsec", "optional", "null when unavailable", "Sersic effective radius."),
    "source_n": FieldSpec("float", "-", "optional", "null when unavailable", "Sersic index."),
    "lens_redshift": FieldSpec("float", "z", "optional", "null when unavailable or ambiguous", "Foreground-lens redshift."),
    "source_redshift": FieldSpec("float", "z", "optional", "null when unavailable or ambiguous", "Background-source redshift."),
    "redshift_provenance": FieldSpec("string", "-", "optional", "null when unavailable", "Source and interpretation of redshifts."),
    "cosmology_id": FieldSpec("string", "-", "conditional for physical mass", "null for angular-only rows", "Named cosmological parameter set."),
    "mass_enclosed_msun": FieldSpec("float", "Msun", "optional", "null when unavailable", "Derived projected mass inside the stated aperture."),
    "mass_aperture_arcsec": FieldSpec("float", "arcsec", "conditional with mass", "null when mass unavailable", "Aperture for projected enclosed mass."),
    "mass_target_available": FieldSpec("boolean", "-", "all rows", "false, never zero-fill", "Whether the mass target is scientifically usable."),
    "theta_E_target_available": FieldSpec("boolean", "-", "all rows", "false, never zero-fill", "Whether an Einstein-radius target is available."),
    "simulator_name": FieldSpec("string", "-", "simulated rows", "null for real", "Simulation backend."),
    "simulator_version": FieldSpec("string", "-", "simulated rows", "null for real", "Backend version."),
    "parameter_seed": FieldSpec("integer", "-", "simulated rows", "null for real", "Seed for physical parameter draws."),
    "noise_seed": FieldSpec("integer", "-", "simulated rows", "null for real", "Seed for observation/noise draws."),
    "parameters_path": FieldSpec("path", "repository-relative POSIX", "simulated rows", "null for real", "Exact generating-parameter sidecar."),
    "structural_status": FieldSpec("enum", "-", "all rows", "use unknown when unreviewed", "File and shape integrity status."),
    "observation_status": FieldSpec("enum", "-", "all rows", "use unknown when unreviewed", "Instrument/calibration/geometry status."),
    "science_status": FieldSpec("enum", "-", "all rows", "use unknown when unreviewed", "Scientific suitability status."),
    "eligible_for": FieldSpec("string", "semicolon-separated enum values", "all rows", "empty only when excluded", "Declared uses such as preprocessing, classical_fit, or ml."),
    "exclusion_reason": FieldSpec("string", "-", "all rows", "null for eligible rows", "Why a row is not eligible for a stated use."),
}

REQUIRED_FIELDS = frozenset(
    {
        "sample_id",
        "lens_id",
        "domain",
        "schema_version",
        "dataset_version",
        "raw_path",
        "science_path",
        "processed_path",
        "image_unit",
        "calibration_status",
        "original_shape",
        "processed_shape",
        "pixel_scale",
        "processing_config_id",
        "processing_version",
        "background_status",
        "source_type",
        "mass_target_available",
        "theta_E_target_available",
        "structural_status",
        "observation_status",
        "science_status",
        "eligible_for",
        "exclusion_reason",
    }
)

CONDITIONAL_FIELDS = {
    "simulated": frozenset(
        {
            "simulation_id",
            "parent_system_id",
            "split_group_id",
            "mass_model",
            "theta_E_arcsec",
            "simulator_name",
            "simulator_version",
            "parameter_seed",
            "noise_seed",
            "parameters_path",
        }
    ),
    "real": frozenset(),
}


def required_fields_for(domain: str | None = None) -> frozenset[str]:
    """Return required fields for a manifest or a specific domain."""

    if domain is None:
        return REQUIRED_FIELDS
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Unknown domain: {domain}")
    return REQUIRED_FIELDS | CONDITIONAL_FIELDS[domain]


def validate_columns(columns: Sequence[str], domain: str | None = None) -> list[str]:
    """Return schema-column errors without inspecting row values."""

    available = set(columns)
    missing = sorted(required_fields_for(domain).difference(available))
    if missing:
        return [f"missing required fields: {missing}"]
    return []


def _is_null(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def validate_row(row: Mapping[str, object]) -> list[str]:
    """Validate domain-independent scalar conventions for one row."""

    errors = validate_columns(row.keys())
    if errors:
        return errors

    domain = row.get("domain")
    if domain not in VALID_DOMAINS:
        errors.append(f"domain must be one of {sorted(VALID_DOMAINS)}")
    if str(row.get("schema_version")) != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    for field in ("sample_id", "lens_id", "raw_path", "science_path", "processed_path"):
        if _is_null(row.get(field)) or str(row[field]).strip() == "":
            errors.append(f"{field} is required")

    for field in ("raw_path", "science_path", "processed_path", "mask_path", "variance_path", "psf_path"):
        value = row.get(field)
        if not _is_null(value):
            path = str(value)
            if path.startswith("/") or "\\" in path or PurePosixPath(path).is_absolute():
                errors.append(f"{field} must be a repository-relative POSIX path")

    try:
        if float(row.get("pixel_scale")) <= 0:
            errors.append("pixel_scale must be positive")
    except (TypeError, ValueError):
        errors.append("pixel_scale must be numeric")

    lens_z = row.get("lens_redshift")
    source_z = row.get("source_redshift")
    if not _is_null(lens_z) and not _is_null(source_z):
        try:
            if not 0 < float(lens_z) < float(source_z):
                errors.append("redshifts must satisfy 0 < lens_redshift < source_redshift")
        except (TypeError, ValueError):
            errors.append("redshifts must be numeric")

    for field in ("structural_status", "observation_status", "science_status", "calibration_status", "background_status"):
        value = row.get(field)
        if not _is_null(value) and value not in VALID_STATUSES:
            errors.append(f"{field} must be one of {sorted(VALID_STATUSES)}")

    eligibility = row.get("eligible_for")
    if not _is_null(eligibility) and str(eligibility).strip():
        unknown = sorted(set(str(eligibility).split(";")) - VALID_ELIGIBILITY)
        if unknown:
            errors.append(f"eligible_for contains unknown values: {unknown}")

    if domain == "simulated":
        for field in CONDITIONAL_FIELDS["simulated"]:
            if _is_null(row.get(field)):
                errors.append(f"{field} is required for simulated rows")
        if not _is_null(row.get("theta_E_arcsec")):
            try:
                if float(row["theta_E_arcsec"]) <= 0:
                    errors.append("theta_E_arcsec must be positive")
            except (TypeError, ValueError):
                errors.append("theta_E_arcsec must be numeric")
    return errors
