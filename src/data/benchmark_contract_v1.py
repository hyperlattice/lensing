"""Strict native-grid benchmark overlay, separate from frozen CSV releases.

An image-only view never opens a target sidecar. Roles and lineage are checked
over the entire supplied ledger before any role is selected. Hashes establish
artifact identity, not scientific qualification; that requires a separately
reviewed numerical report. This module does not authorize final evaluation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

ROLES = frozenset({"source_train", "source_development", "source_calibration",
                   "target_adaptation", "numerical_development", "power_pilot", "final"})
PRODUCTS = frozenset({"image", "variance", "invalid_mask", "coverage_mask", "psf"})
ROW_FIELDS = frozenset({"sample_id", "parent_system_id", "role", "renderer", "condition",
                        "parameter_seed", "noise_seed", "ancestry_ids", "shape",
                        "pixel_scale_arcsec", "psf_family", "products"})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observation_digest(rows):
    """Bind a numerical qualification report to exact observation definitions."""
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def contained_path(root, relative):
    """Reject absolute paths, traversal and symlinks escaping the release."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Artifact paths must be relative POSIX paths")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(p in {"..", "."} for p in relative.split("/")):
        raise ValueError("Artifact path escapes release")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact path escapes release")
    return path


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_lineage(rows, excluded=()):
    """Reject reused ancestry across roles, including renamed parent systems.

    Excluded identities contain role, parent_system_id, parameter_seed,
    noise_seed and ancestry_ids only; old pixels/targets need not be opened.
    Ancestry IDs identify reusable assets, not an analytic profile family.
    """
    assigned, parent_identity = {}, {}
    for row in [*excluded, *rows]:
        role, parent = row["role"], row["parent_system_id"]
        if role not in ROLES or not isinstance(parent, str) or not parent:
            raise ValueError("Invalid lineage role or parent ID")
        if not all(_integer(row[k]) for k in ("parameter_seed", "noise_seed")):
            raise ValueError("Parameter/noise seeds must be nonnegative integers")
        assets = row["ancestry_ids"]
        if not isinstance(assets, list) or any(not isinstance(a, str) or not a for a in assets):
            raise ValueError("ancestry_ids must list nonempty asset identities")
        identity = (row["parameter_seed"], tuple(sorted(set(assets))))
        tokens = [("parent", parent), ("parameter_seed", row["parameter_seed"]),
                  ("noise_seed", row["noise_seed"]), *[("asset", a) for a in assets]]
        for token in tokens:
            if token in assigned and assigned[token] != role:
                raise ValueError(f"Cross-role lineage overlap: {token}")
            assigned[token] = role
        if parent in parent_identity and parent_identity[parent] != identity:
            raise ValueError("Parent identity changed between observations")
        parent_identity[parent] = identity


class BenchmarkImages:
    """Validated image-only projection of an explicit benchmark role.

    The manifest is a new overlay, not a replacement for schema-v2 CSVs.
    Float32 native-grid images are in electrons/s without normalization;
    variance is in (electrons/s)^2, and True masks mark excluded pixels.
    Variance and PSF are validated products, never implicit CNN features.
    Numerical-development releases require an explicit engineering opt-in.
    """

    def __init__(self, manifest_path, *, role, excluded=(), allow_engineering=False):
        self.root = Path(manifest_path).resolve().parent
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        expected = {"schema", "release_id", "stage", "image_unit", "variance_unit",
                    "mask_semantics", "qualification", "observations"}
        if set(manifest) != expected or manifest["schema"] != "cross_simulator_benchmark_v1":
            raise ValueError("Unknown or incomplete benchmark schema")
        if not isinstance(manifest["release_id"], str) or not manifest["release_id"]:
            raise ValueError("Missing release identity")
        if role not in ROLES:
            raise ValueError("Unknown requested role")
        if (manifest["image_unit"] != "electrons_per_second" or
            manifest["variance_unit"] != "electrons_per_second_squared" or
            manifest["mask_semantics"] != "true_is_excluded"):
            raise ValueError("Undeclared image/variance units or mask semantics")
        rows = manifest["observations"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("Empty benchmark release")
        ids = set()
        for row in rows:
            if set(row) != ROW_FIELDS:
                raise ValueError("Unexpected row fields; truth belongs in a separate sidecar")
            for key in ("sample_id", "renderer", "condition", "psf_family"):
                if not isinstance(row[key], str) or not row[key]:
                    raise ValueError(f"Missing observation field: {key}")
            if row["sample_id"] in ids:
                raise ValueError("Duplicate observation identity")
            ids.add(row["sample_id"])
            shape = row["shape"]
            if (not isinstance(shape, list) or len(shape) != 2 or
                any(not _integer(n) or n == 0 for n in shape)):
                raise ValueError("Invalid native shape")
            if (not isinstance(row["pixel_scale_arcsec"], (int, float)) or
                isinstance(row["pixel_scale_arcsec"], bool) or
                not np.isfinite(row["pixel_scale_arcsec"]) or row["pixel_scale_arcsec"] <= 0):
                raise ValueError("Invalid native pixel scale")
            if row["role"] in {"source_train", "source_development", "source_calibration", "target_adaptation"} and row["psf_family"] != "gaussian":
                raise ValueError("Held-out PSF family cannot enter optimization or selection")
            if row["role"].startswith("source_") and row["renderer"] != "lenstronomy":
                raise ValueError("Source roles require renderer A (lenstronomy)")
            if row["role"] == "target_adaptation" and row["renderer"] != "autolens":
                raise ValueError("Adaptation role requires renderer B (autolens)")
            if set(row["products"]) != PRODUCTS:
                raise ValueError("Missing required image/variance/mask/PSF products")
        validate_lineage(rows, excluded)
        if manifest["stage"] == "engineering":
            if manifest["qualification"] is not None or not allow_engineering or any(r["role"] != "numerical_development" for r in rows):
                raise ValueError("Engineering release is not eligible for model training/evaluation")
        elif manifest["stage"] == "qualified":
            q = manifest["qualification"]
            if not isinstance(q, dict) or set(q) != {"path", "sha256"}:
                raise ValueError("Missing qualification report")
            report = json.loads(self._artifact(q).read_text(encoding="utf-8"))
            if (report.get("passed") is not True or report.get("release_id") != manifest["release_id"] or
                report.get("observation_contract_sha256") != observation_digest(rows)):
                raise ValueError("Failed or mismatched qualification report")
        else:
            raise ValueError("Unknown release stage")
        self._rows = deepcopy([r for r in rows if r["role"] == role])
        if not self._rows:
            raise ValueError("Requested role has no observations")
        for row in rows:
            self._products(row)

    def _artifact(self, record):
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("Artifact requires path and SHA256")
        path = contained_path(self.root, record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Missing or changed artifact: {record['path']}")
        return path

    def _products(self, row):
        arrays = {key: np.load(self._artifact(value), allow_pickle=False)
                  for key, value in row["products"].items()}
        shape = tuple(row["shape"])
        for key in ("image", "variance"):
            a = arrays[key]
            if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError(f"Invalid {key} geometry/dtype/finite values")
        for key in ("invalid_mask", "coverage_mask"):
            if arrays[key].shape != shape or arrays[key].dtype != bool:
                raise ValueError("Masks must be bool on the native image grid")
        valid = ~(arrays["invalid_mask"] | arrays["coverage_mask"])
        if not valid.any() or np.any(arrays["variance"][valid] <= 0):
            raise ValueError("Variance must be positive on valid pixels")
        psf = arrays["psf"]
        if (psf.ndim != 2 or any(n % 2 != 1 for n in psf.shape) or
            not np.isfinite(psf).all() or np.any(psf < 0) or
            not np.isclose(psf.sum(), 1, rtol=0, atol=1e-6)):
            raise ValueError("PSF must be finite, nonnegative, odd and unit sum")
        return arrays

    @property
    def identities(self):
        return tuple(row["sample_id"] for row in self._rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        # Recheck bytes at access time, preventing post-validation substitution.
        return self._products(self._rows[index])["image"]


def load_source_targets(path, images, *, role):
    """Explicit label join for permitted source roles only; never used by UDA."""
    if role not in {"source_train", "source_development", "source_calibration"}:
        raise ValueError("Target labels are unavailable through the training interface")
    if any(row["role"] != role for row in images._rows):
        raise ValueError("Target role differs from image role")
    targets = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(targets) != set(images.identities):
        raise ValueError("Target identities must match the image role exactly")
    values = np.asarray([targets[k] for k in images.identities], dtype=float)
    if values.shape != (len(images),) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Einstein-radius targets must be finite positive scalars in arcsec")
    return values
