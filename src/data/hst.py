"""Reviewed HST preparation, physical-lens identities, and image-only releases.

Paths in observational manifests are repository/cwd relative, unlike the existing
synthetic bank contract. No HST normalization or reference fitting occurs here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares

from src.data.schema import contained_path
from src.ml.artifacts import create_once, sha256, source_provenance, write_json

REVIEW_FIELDS = (
    "centering",
    "photometry",
    "psf",
    "coverage_masks",
    "reference_convention",
    "morphology",
)
COHORTS = ("development", "confirmation")
HST_METHOD_FILES = (
    "src/data/hst.py",
    "src/data/dataset.py",
    "src/data/schema.py",
    "src/simulation/hst_bridge.py",
    "src/ml/predict.py",
    "src/ml/analysis.py",
    "src/ml/artifacts.py",
    "src/ml/data.py",
    "src/ml/model.py",
    "src/simulation/render.py",
    "src/simulation/observation.py",
    "src/simulation/backends.py",
    "scripts/hst_study.py",
)


def implementation_identity() -> dict[str, str]:
    """Identify the small set of files defining HST preparation and evaluation."""
    return {path: sha256(path) for path in HST_METHOD_FILES}


def cohort_identity(rows: list[dict[str, Any]], cohort: str) -> str:
    """Identify one cohort's candidate decisions without coupling the other cohort."""
    selected = [row for row in rows if row["cohort"] == cohort]
    content = json.dumps(selected, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def load_hst_config(path: Path) -> dict[str, Any]:
    """Load a flat HST configuration without opening scientific inputs."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "study_config",
        "observations",
        "references",
        "reserved_manifest",
        "pilot_audit",
        "prepared_root",
        "output",
        "batch_size",
        "bootstrap",
        "bridge",
        "eligibility_rules",
        "primary_band_rule",
        "reference_strategy",
    }
    if not isinstance(config, dict) or set(config) != required or config["schema_version"] != 1:
        raise ValueError("Unexpected HST study configuration")
    if type(config["batch_size"]) is not int or config["batch_size"] < 1:
        raise ValueError("HST batch size must be a positive integer")
    controls = config["bootstrap"]
    if set(controls) != {"resamples", "seed"} or any(
        type(value) is not int or value < 1 for value in controls.values()
    ):
        raise ValueError("Expected positive integer HST bootstrap controls")
    return config


def identity_tokens(row: dict[str, Any]) -> set[str]:
    """Compare reviewed canonical names and aliases independent of punctuation."""
    names = [row["system_id"], *row.get("aliases", [])]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Lens identities and aliases must be nonempty strings")
    tokens = {re.sub(r"[^A-Z0-9]", "", name.upper()) for name in names}
    if "" in tokens:
        raise ValueError("Lens identities must contain letters or digits")
    return tokens


def read_observations(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate identities across ALL candidates before selecting a cohort."""
    document = json.loads(Path(config["observations"]).read_text(encoding="utf-8"))
    if document["schema"] != "hst_observations_v1" or not document["observations"]:
        raise ValueError("Expected a nonempty HST candidate ledger")
    reserved = json.loads(Path(config["reserved_manifest"]).read_text(encoding="utf-8"))
    reserved_tokens = set().union(*(identity_tokens(row) for row in reserved["systems"]))
    seen_observations: set[str] = set()
    identities: dict[str, str] = {}
    cohorts: dict[str, str] = {}
    selected: set[str] = set()
    for row in document["observations"]:
        observation = row["observation_id"]
        if not re.fullmatch(r"[A-Za-z0-9_+-]+", observation) or observation in seen_observations:
            raise ValueError("Invalid or duplicate HST observation ID")
        seen_observations.add(observation)
        if row["cohort"] not in COHORTS or row["eligibility"] not in {
            "pending",
            "include",
            "exclude",
        }:
            raise ValueError(f"Invalid cohort or eligibility: {observation}")
        tokens = identity_tokens(row)
        if row["cohort"] == "confirmation" and tokens & reserved_tokens:
            raise ValueError(f"Reserved development lens in confirmation: {observation}")
        for token in tokens:
            if token in identities and identities[token] != row["system_id"]:
                raise ValueError(f"Lens aliases map to different canonical systems: {token}")
            identities[token] = row["system_id"]
            if token in cohorts and cohorts[token] != row["cohort"]:
                raise ValueError(f"Physical lens crosses HST cohorts: {token}")
            cohorts[token] = row["cohort"]
        if row["eligibility"] == "exclude" and not row.get("exclusion_reason"):
            raise ValueError(f"Excluded observation needs a reason: {observation}")
        if row["eligibility"] == "include":
            if tokens & selected:
                raise ValueError("Select one primary observation per physical lens")
            selected.update(tokens)
    return document["observations"]


def initialize_observations(config: dict[str, Any]) -> None:
    """Seed an UNQUALIFIED review ledger and separate references from the pilot."""
    audit = json.loads(Path(config["pilot_audit"]).read_text(encoding="utf-8"))
    rows, references = [], []
    for system in audit["systems"]:
        source = system["native_products"]
        rows.append(
            {
                "system_id": system["system_id"],
                "aliases": [],
                "observation_id": f"{system['system_id']}_{system['band']}",
                "survey": system["survey"],
                "band": system["band"],
                "cohort": "development",
                "eligibility": "pending",
                "exclusion_reason": "",
                "science": {**source, "key": "image_data"},
                "psf": {**source, "key": "psf_normalized"},
                "valid_mask": None,
                "pixel_transform_arcsec": None,
                "psf_pixel_transform_arcsec": None,
                "input_unit": None,
                "unit_multiplier": None,
                "background_raw": None,
                "center": {"method": "reviewed_external", "xy": None, "evidence": ""},
                "crop_shape_yx": None,
                "center_tolerance_arcsec": None,
                "psf_center_tolerance_pixels": None,
                "review": dict.fromkeys(REVIEW_FIELDS, ""),
            }
        )
        references.append(
            {
                "system_id": system["system_id"],
                **system["reference"],
                "mass_model": "EPL + external shear",
                "interval_semantics": "median minus percentile16; percentile84 minus median",
                "source": system["reference"]["table_source"],
                "alternatives": [],
            }
        )
    if Path(config["observations"]).exists() or Path(config["references"]).exists():
        raise FileExistsError("Initialization cannot replace observation or reference ledgers")
    create_once(config["observations"], {"schema": "hst_observations_v1", "observations": rows})
    create_once(config["references"], {"schema": "hst_references_v1", "references": references})


def read_array(descriptor: dict[str, Any]) -> tuple[np.ndarray, fits.Header | None]:
    """Read one pinned array, preserving FITS WCS where supplied."""
    path = Path(descriptor["path"])
    if sha256(path) != descriptor["sha256"]:
        raise ValueError(f"HST input bytes changed: {path}")
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False), None
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return archive[descriptor["key"]].copy(), None
    if path.suffix.lower() in {".fits", ".fit"}:
        with fits.open(path, memmap=False) as hdus:
            hdu = hdus[descriptor.get("hdu", "SCI")]
            return np.asarray(hdu.data).copy(), hdu.header.copy()
    raise ValueError(f"Unsupported HST array format: {path}")


def fit_light_center(image: np.ndarray, valid: np.ndarray, roi_yx: list[int]) -> dict[str, Any]:
    """Fit a local axis-aligned Gaussian plus constant foreground-light profile.

    This is a local light-center estimator, not a lens-mass fit. The reviewed ROI
    must isolate foreground light. Arcs, saturation and profile asymmetry can bias
    it; a successful optimizer alone does not qualify the observational center.
    """
    y0, y1, x0, x1 = roi_yx
    if not (0 <= y0 < y1 <= image.shape[0] and 0 <= x0 < x1 <= image.shape[1]):
        raise ValueError("Center-fitting ROI is outside the native image")
    patch = image[y0:y1, x0:x1]
    supported = valid[y0:y1, x0:x1]
    if min(patch.shape) < 5 or supported.sum() < 16:
        raise ValueError("Center fitting requires at least 16 valid pixels in a 5x5 ROI")
    yy, xx = np.indices(patch.shape, dtype=float)
    background = float(np.median(patch[supported]))
    weights = np.maximum(patch - background, 0.0) * supported
    if weights.sum() <= 0:
        raise ValueError("Center ROI has no positive light above its background")
    center_x = float((weights * xx).sum() / weights.sum())
    center_y = float((weights * yy).sum() / weights.sum())
    amplitude = float(patch[supported].max() - background)
    height, width = patch.shape

    def residual(parameters: np.ndarray) -> np.ndarray:
        amplitude, center_x, center_y, sigma_x, sigma_y, background = parameters
        exponent = ((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2
        profile = amplitude * np.exp(-0.5 * exponent) + background
        return (profile - patch)[supported]

    fitted = least_squares(
        residual,
        [amplitude, center_x, center_y, width / 6, height / 6, background],
        bounds=(
            [0, 0, 0, 0.3, 0.3, -np.inf],
            [np.inf, width - 1, height - 1, width, height, np.inf],
        ),
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f"Foreground light-center fit failed: {fitted.message}")
    return {
        "xy": [float(fitted.x[1] + x0), float(fitted.x[2] + y0)],
        "method": "gaussian_profile",
        "rms_raw": float(np.sqrt(np.mean(fitted.fun**2))),
    }


@dataclass(frozen=True)
class PreparedImage:
    """A physical-rate image and its explicit native-grid preparation record."""

    image: np.ndarray
    psf: np.ndarray
    valid: np.ndarray
    metadata: dict[str, Any]
    header: fits.Header | None


def prepare_image(
    science: np.ndarray,
    psf: np.ndarray,
    valid: np.ndarray,
    row: dict[str, Any],
    header: fits.Header | None = None,
) -> PreparedImage:
    """Convert reviewed units and crop native pixels; never pad or interpolate."""
    if science.ndim != 2 or valid.shape != science.shape or not np.isfinite(science).all():
        raise ValueError("Science/coverage must be finite matching two-dimensional arrays")
    if valid.dtype != np.bool_:
        raise ValueError("Coverage mask must explicitly use boolean true=valid semantics")
    if psf.ndim != 2 or not np.isfinite(psf).all() or np.any(psf < 0) or psf.sum() <= 0:
        raise ValueError("Native PSF must be finite, nonnegative, and have positive flux")
    transform = np.asarray(row["pixel_transform_arcsec"], dtype=float)
    psf_transform = np.asarray(row["psf_pixel_transform_arcsec"], dtype=float)
    if (
        transform.shape != (2, 2)
        or not np.isfinite(transform).all()
        or abs(np.linalg.det(transform)) == 0
    ):
        raise ValueError("Native pixel-to-angle transform must be finite and nonsingular")
    if psf_transform.shape != (2, 2) or not np.allclose(
        psf_transform, transform, rtol=1e-8, atol=1e-10
    ):
        raise ValueError("PSF must already match the native science pixel grid")
    normalized_psf = psf.astype(float) / psf.sum()
    psf_y, psf_x = np.indices(psf.shape, dtype=float)
    psf_center = np.array([(normalized_psf * psf_x).sum(), (normalized_psf * psf_y).sum()])
    psf_offset = psf_center - (np.asarray(psf.shape)[::-1] - 1) / 2
    psf_tolerance = row["psf_center_tolerance_pixels"]
    if psf_tolerance is None or not np.isfinite(psf_tolerance) or psf_tolerance < 0:
        raise ValueError("An explicit nonnegative native PSF-center tolerance is required")
    if np.linalg.norm(psf_offset) > psf_tolerance:
        raise ValueError("Native PSF centroid exceeds its reviewed centering tolerance")
    if header is not None:
        wcs = WCS(header)
        if not wcs.has_celestial or not np.allclose(
            wcs.celestial.pixel_scale_matrix * 3600, transform, rtol=1e-6, atol=1e-8
        ):
            raise ValueError("Declared geometry does not match native celestial FITS WCS")
    multiplier, background = row["unit_multiplier"], row["background_raw"]
    if not row["input_unit"] or multiplier is None or background is None:
        raise ValueError("Reviewed input units, conversion, and background offset are required")
    if not np.isfinite([multiplier, background]).all() or multiplier <= 0:
        raise ValueError("Unit multiplier must be positive and background must be finite")
    center = row["center"]
    if not center.get("evidence"):
        raise ValueError("Center needs reviewed light-profile evidence")
    if center["method"] == "gaussian_profile":
        center_record = fit_light_center(science, valid, center["roi_yx"])
    elif center["method"] == "reviewed_external":
        center_record = {"method": center["method"], "xy": center["xy"]}
    else:
        raise ValueError("Unsupported center method; no brightest-pixel fallback")
    center_xy = np.asarray(center_record["xy"], dtype=float)
    if center_xy.shape != (2,) or not np.isfinite(center_xy).all():
        raise ValueError("Reviewed native center must contain two finite coordinates")
    shape = row["crop_shape_yx"] or list(science.shape)
    if len(shape) != 2 or any(type(value) is not int or value < 2 for value in shape):
        raise ValueError("Crop dimensions must be integers of at least two pixels")
    height, width = shape
    origin_xy = np.floor(center_xy - (np.array([width, height]) - 1) / 2 + 0.5).astype(int)
    if row["crop_shape_yx"] is None:
        origin_xy = np.array([0, 0])
    x0, y0 = origin_xy
    if min(x0, y0) < 0 or x0 + width > science.shape[1] or y0 + height > science.shape[0]:
        raise ValueError("Requested centered field exceeds native coverage; padding is disallowed")
    slices = (slice(y0, y0 + height), slice(x0, x0 + width))
    if not valid[slices].all():
        raise ValueError(
            "Selected CNN field contains unsupported/masked pixels; no implicit filling"
        )
    output_center = center_xy - origin_xy
    offset_arcsec = transform @ (output_center - (np.array([width, height]) - 1) / 2)
    tolerance = row["center_tolerance_arcsec"]
    if tolerance is None or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("An explicit nonnegative center tolerance is required")
    if np.linalg.norm(offset_arcsec) > tolerance:
        raise ValueError("Native crop misses the declared center tolerance")
    image = ((science[slices].astype(float) - background) * multiplier).astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Physical-unit conversion overflowed float32")
    cropped_header = None
    if header is not None:
        cropped_header = WCS(header).slice(slices).to_header()
        cropped_header["BUNIT"] = "electron/s"
    metadata = {
        "source_shape_yx": list(science.shape),
        "shape_yx": list(image.shape),
        "origin_xy_pixels": origin_xy.tolist(),
        "center": center_record,
        "center_xy_prepared": output_center.tolist(),
        "center_offset_arcsec": offset_arcsec.tolist(),
        "pixel_transform_arcsec": transform.tolist(),
        "unit": "electrons_per_second",
        "input_unit": row["input_unit"],
        "unit_multiplier": multiplier,
        "background_raw_subtracted": background,
        "psf_sum_original": float(psf.sum()),
        "psf_centroid_offset_xy_pixels": psf_offset.tolist(),
        "operations": [
            "subtract_declared_background",
            "multiply_declared_unit_factor",
            "native_crop",
        ],
        "interpolation": False,
        "normalization_fitted": False,
    }
    return PreparedImage(image, normalized_psf, valid[slices], metadata, cropped_header)


def prepare_release(config: dict[str, Any], cohort: str) -> dict[str, Any]:
    """Prepare reviewed candidates; retain pending/excluded cases in the ledger."""
    if (Path(config["output"]) / "confirmation_freeze.json").exists():
        raise ValueError("Preparation is frozen; choose a new study identity for changes")
    if (Path(config["output"]) / cohort / "predictions_complete.json").exists():
        raise ValueError("Predictions already exist; use a new preparation/output identity")
    candidates = read_observations(config)
    root = Path(config["prepared_root"]) / cohort
    root.mkdir(parents=True, exist_ok=True)
    prepared, ledger = [], []
    for row in candidates:
        if row["cohort"] != cohort:
            continue
        review = row.get("review", {})
        blockers = [
            name
            for name in REVIEW_FIELDS
            if not isinstance(review.get(name), str) or not review[name].strip()
        ]
        ready = row["eligibility"] == "include" and not blockers
        ledger.append(
            {
                "observation_id": row["observation_id"],
                "system_id": row["system_id"],
                "eligibility": row["eligibility"],
                "prepared": ready,
                "blockers": blockers,
                "exclusion_reason": row.get("exclusion_reason", ""),
            }
        )
        if not ready:
            continue
        science, header = read_array(row["science"])
        psf, _ = read_array(row["psf"])
        valid = np.ones(science.shape, dtype=bool)
        if row["valid_mask"] is not None:
            valid, _ = read_array(row["valid_mask"])
        product = prepare_image(science, psf, valid, row, header)
        prefix = row["observation_id"]
        image_path = root / f"{prefix}.npy"
        psf_path = root / f"{prefix}_psf.npy"
        np.save(image_path, product.image, allow_pickle=False)
        np.save(psf_path, product.psf, allow_pickle=False)
        if product.header is not None:
            fits.PrimaryHDU(product.image, header=product.header).writeto(
                root / f"{prefix}.fits", overwrite=True, checksum=True
            )
        prepared.append(
            {
                "system_id": row["system_id"],
                "aliases": row.get("aliases", []),
                "observation_id": prefix,
                "cohort": cohort,
                "survey": row["survey"],
                "band": row["band"],
                "image": {"path": image_path.name, "sha256": sha256(image_path)},
                "psf": {"path": psf_path.name, "sha256": sha256(psf_path)},
                "preparation": product.metadata,
                "review": row["review"],
                "inputs": {key: row[key] for key in ("science", "psf", "valid_mask")},
            }
        )
    document = {
        "schema": "hst_prepared_v1",
        "cohort": cohort,
        "observations": prepared,
        "ledger": ledger,
        "observations_sha256": sha256(config["observations"]),
        "cohort_sha256": cohort_identity(candidates, cohort),
        "reserved_sha256": sha256(config["reserved_manifest"]),
        "implementation": implementation_identity(),
        "provenance": source_provenance(),
    }
    write_json(root / "manifest.json", document)
    return document


def load_prepared(
    config: dict[str, Any], cohort: str, *, load_images: bool = True
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Load qualified images only; never open classical references."""
    candidates = read_observations(config)
    root = Path(config["prepared_root"]) / cohort
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        document["schema"] != "hst_prepared_v1"
        or document["cohort"] != cohort
        or document["cohort_sha256"] != cohort_identity(candidates, cohort)
        or document["reserved_sha256"] != sha256(config["reserved_manifest"])
        or document["implementation"] != implementation_identity()
    ):
        raise ValueError("HST preparation identity changed")
    expected = {
        row["observation_id"]
        for row in candidates
        if row["cohort"] == cohort and row["eligibility"] == "include"
    }
    rows = document["observations"]
    if (
        not rows
        or {row["observation_id"] for row in rows} != expected
        or len(rows) != len(expected)
    ):
        raise ValueError("All included primary observations must be qualified exactly once")
    images = []
    if load_images:
        for row in rows:
            for kind in ("image", "psf"):
                path = contained_path(root, row[kind]["path"])
                if sha256(path) != row[kind]["sha256"]:
                    raise ValueError(f"Prepared HST {kind} changed: {path}")
            image = np.load(contained_path(root, row["image"]["path"]), allow_pickle=False)
            if image.ndim != 2 or image.dtype != np.float32 or not np.isfinite(image).all():
                raise ValueError("Prepared HST image must be finite two-dimensional float32")
            images.append(image)
    return rows, images


def read_references(config: dict[str, Any], system_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Validate explicitly model-dependent references for selected physical lenses."""
    document = json.loads(Path(config["references"]).read_text(encoding="utf-8"))
    if document["schema"] != "hst_references_v1":
        raise ValueError("Unexpected HST reference schema")
    selected = {}
    for row in document["references"]:
        identifier = row["system_id"]
        if identifier not in system_ids:
            continue
        if identifier in selected:
            raise ValueError(f"Duplicate HST radius reference: {identifier}")
        numbers = [row[key] for key in ("median_arcsec", "error_lower", "error_upper")]
        if not np.isfinite(numbers).all() or min(numbers) <= 0:
            raise ValueError(f"Invalid HST radius reference: {identifier}")
        for key in ("definition", "mass_model", "interval_semantics", "source"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Reference needs {key}: {identifier}")
        selected[identifier] = row
    if set(selected) != set(system_ids):
        raise ValueError("Incomplete classical reference coverage")
    return selected


def freeze_confirmation(config: dict[str, Any]) -> dict[str, Any]:
    """Bind cohort, methods, references and verified inputs before final inference."""
    from src.ml.artifacts import verified_hst_assets

    output = Path(config["output"])
    if (output / "confirmation" / "predictions_before_scoring.csv").exists():
        raise ValueError("Cannot freeze after confirmation predictions exist")
    for name in ("eligibility_rules", "primary_band_rule", "reference_strategy"):
        value = config[name]
        if not value or not isinstance(value, str) or not value.strip():
            raise ValueError(f"Confirmation requires an explicit {name}")
    candidates = read_observations(config)
    reserved = json.loads(Path(config["reserved_manifest"]).read_text(encoding="utf-8"))
    completed_development = set().union(
        *(
            identity_tokens(row)
            for row in candidates
            if row["cohort"] == "development" and row["eligibility"] != "pending"
        )
    )
    if any(not identity_tokens(row) & completed_development for row in reserved["systems"]):
        raise ValueError("Resolve every reserved development lens before confirmation")
    if any(
        row["cohort"] == "confirmation" and row["eligibility"] == "pending" for row in candidates
    ):
        raise ValueError("Resolve every confirmation candidate before freezing")
    rows, _ = load_prepared(config, "confirmation")
    from src.simulation.render import native_square_transform

    for row in rows:
        matrix = np.asarray(row["preparation"]["pixel_transform_arcsec"])
        native_square_transform(
            {
                "pixel_transform_arcsec": matrix,
                "pixel_scale_arcsec": float(np.sqrt(abs(np.linalg.det(matrix)))),
            }
        )
    if len(rows) < 2:
        raise ValueError(
            "Confirmation needs at least two independent lenses; this is not a power check"
        )
    read_references(config, [row["system_id"] for row in rows])
    assets_path = output / "model_assets.json"
    verified_hst_assets(assets_path)
    validation_path = output / "preparation_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    development_manifest = Path(config["prepared_root"]) / "development" / "manifest.json"
    from src.ml.artifacts import load_config

    study = load_config(config["study_config"])
    if (
        validation["passed"] is not True
        or validation["smoke"] is not False
        or validation["assets_sha256"] != sha256(assets_path)
        or validation["prepared_development_sha256"] != sha256(development_manifest)
        or validation["settings"] != config["bridge"]
        or validation["implementation"] != implementation_identity()
        or validation["source_bank_sha256"] != sha256(study["source_development"])
        or validation["parent_count"] != validation["available_parent_count"]
    ):
        raise ValueError("A complete, current development preparation validation is required")
    files = [
        Path(config[key])
        for key in ("observations", "references", "reserved_manifest", "study_config")
    ]
    files.extend(
        [
            assets_path,
            validation_path,
            development_manifest,
            Path(config["prepared_root"]) / "confirmation" / "manifest.json",
        ]
    )
    files.extend(Path(path) for path in HST_METHOD_FILES)
    files.append(Path(study["source_development"]))
    image_files = {}
    for cohort in COHORTS:
        cohort_rows, _ = load_prepared(config, cohort)
        for row in cohort_rows:
            for kind in ("image", "psf"):
                path = contained_path(Path(config["prepared_root"]) / cohort, row[kind]["path"])
                image_files[str(path)] = row[kind]["sha256"]
    record = {
        "schema": "hst_confirmation_freeze_v1",
        "config": config,
        "system_ids": [row["system_id"] for row in rows],
        "primary_contrast": "G_minus_C",
        "secondary_contrast": "R_minus_G_descriptive",
        "estimand": "mean_lens(mean_seed(abs(prediction-reference)))",
        "files": {str(path): sha256(path) for path in files},
        "images": image_files,
        "cohort_ledger": candidates,
    }
    create_once(output / "confirmation_freeze.json", record)
    return record


def verify_confirmation(config: dict[str, Any], *, images: bool) -> dict[str, Any]:
    """Verify the frozen protocol; saved-prediction analysis need not read images."""
    path = Path(config["output"]) / "confirmation_freeze.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["schema"] != "hst_confirmation_freeze_v1" or record["config"] != config:
        raise ValueError("Confirmation configuration changed")
    artifacts = dict(record["files"])
    if images:
        artifacts.update(record["images"])
    for filename, expected in artifacts.items():
        if sha256(filename) != expected:
            raise ValueError(f"Frozen HST input changed: {filename}")
    return record
