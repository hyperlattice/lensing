"""Acquire and audit a fixed HST development sample without CNN scoring.

Run from the repository root with ``python -m scripts.hst_pilot --help``.
Native HST grids and science values are preserved. Products are diagnostic
inputs, not qualified model inputs or a final observational benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.data.schema import contained_path

DEFAULT_MANIFEST = Path("data/manifests/hst_compatibility_pilot_v1.json")
DEFAULT_DATA = Path("data/raw/external/hst_compatibility_pilot")
DEFAULT_OUTPUT = Path("results/reports/hst_compatibility_pilot")
MAX_ASSET_BYTES = 5_000_000


def file_sha256(path: Path) -> str:
    """Identify exact source or artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha1(content: bytes) -> str:
    """Compute Git's blob identity, including its type and byte-count header."""
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Validate all identities and assets before any downloads occur."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["schema"] != "hst_compatibility_pilot_v1":
        raise ValueError("Unexpected HST pilot schema")
    if document["role"] != "compatibility_development":
        raise ValueError("HST pilot must be reserved for compatibility development")
    if document.get("exclude_from_future_confirmation") is not True:
        raise ValueError("Compatibility systems must be excluded from future confirmation")
    if not re.fullmatch(r"[0-9a-f]{40}", document["source_commit"]):
        raise ValueError("Dinos source must be pinned to a full commit")
    if not document["systems"]:
        raise ValueError("Pilot must contain systems")
    identifiers = []
    for system in document["systems"]:
        identifier = system["system_id"]
        if not re.fullmatch(r"[A-Za-z0-9+_-]+", identifier):
            raise ValueError("Invalid system identity")
        identifiers.append(identifier)
        reference = system["reference"]
        values = [reference[key] for key in ("median_arcsec", "error_lower", "error_upper")]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError(f"Invalid reference interval for {identifier}")
        if reference["definition"] != "lenstronomy_EPL_theta_E_arcsec":
            raise ValueError("Reference convention must remain explicit")
        if set(system["assets"]) not in ({"image", "psf"}, {"image", "psf", "mask"}):
            raise ValueError("Each system requires an image and PSF, with optional mask")
        for kind, asset in system["assets"].items():
            expected = (
                f"2_dolphin_modelling/data/{identifier}/{kind}_{identifier}_{system['band']}.h5"
            )
            if asset["repository_path"] != expected:
                raise ValueError("Asset path does not match system, kind, and band")
            if not re.fullmatch(r"[0-9a-f]{40}", asset["git_blob_sha1"]):
                raise ValueError("Missing asset Git identity")
            if not 0 < asset["bytes"] <= MAX_ASSET_BYTES:
                raise ValueError("Asset exceeds the bounded pilot download size")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Pilot contains duplicate independent systems")
    grid = document["comparison_grid"]
    scale = grid["pixel_scale_arcsec"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid comparison scale")
    if len(grid["shape"]) != 2 or any(value <= 0 for value in grid["shape"]):
        raise ValueError("Invalid comparison image shape")
    return document


def acquire_assets(
    document: dict[str, Any], data_root: Path, *, offline: bool
) -> dict[str, dict[str, Path]]:
    """Fetch pinned small files; verify cached and downloaded bytes identically."""
    acquired = {}
    for system in document["systems"]:
        identifier = system["system_id"]
        acquired[identifier] = {}
        for kind, asset in system["assets"].items():
            filename = Path(asset["repository_path"]).name
            path = contained_path(data_root, f"{identifier}/{filename}")
            if path.exists():
                content = path.read_bytes()
            else:
                if offline:
                    raise FileNotFoundError(f"Offline pilot asset is missing: {path}")
                url = (
                    "https://raw.githubusercontent.com/Project-Dinos/dinos-i/"
                    f"{document['source_commit']}/{asset['repository_path']}"
                )
                request = urllib.request.Request(url, headers={"User-Agent": "hst-pilot-v1"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    content = response.read(MAX_ASSET_BYTES + 1)
            if len(content) != asset["bytes"] or git_blob_sha1(content) != asset["git_blob_sha1"]:
                raise ValueError(f"Source bytes failed pinned Git verification: {path}")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            acquired[identifier][kind] = path
    return acquired


def inspect_pair(
    paths: dict[str, Path], comparison_grid: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate HDF5 science/geometry and preserve arrays without resampling.

    The pixel-to-angle matrix maps pixel (x,y) into angular coordinates.
    Column norms give the physical pixel-edge lengths, and sqrt(abs(det))
    gives the square pixel scale with equal area, even on rotated grids.
    """
    import h5py

    arrays = {}
    with h5py.File(paths["image"], "r") as source:
        for key in ("image_data", "exposure_time", "transform_pix2angle", "background_rms"):
            arrays[key] = np.asarray(source[key][()])
        unit = source["image_data"].attrs.get("BUNIT", source.attrs.get("BUNIT"))
    with h5py.File(paths["psf"], "r") as source:
        psf = np.asarray(source["kernel_point_source"][()], dtype=np.float64)
    for key, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite HST {key}")
    image = arrays["image_data"]
    exposure = arrays["exposure_time"]
    transform = arrays["transform_pix2angle"]
    rms = arrays["background_rms"]
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError("Science image must be a nonempty two-dimensional array")
    if exposure.shape not in ((), image.shape) or np.any(exposure <= 0):
        raise ValueError("Exposure must be positive and scalar or match the native image")
    if rms.shape != () or rms < 0:
        raise ValueError("Background RMS must be a nonnegative scalar")
    if transform.shape != (2, 2) or abs(np.linalg.det(transform)) <= np.finfo(float).eps:
        raise ValueError("Pixel-to-angle transform must be a nonsingular 2 by 2 matrix")
    if psf.ndim != 2 or not np.isfinite(psf).all() or np.any(psf < 0):
        raise ValueError("PSF must be finite, nonnegative, and two-dimensional")
    psf_sum = float(psf.sum())
    if psf_sum <= 0:
        raise ValueError("PSF must have positive total flux")
    arrays["psf_normalized"] = psf / psf_sum
    mask_summary = {"present": "mask" in paths, "semantics": "not_reviewed"}
    if "mask" in paths:
        with h5py.File(paths["mask"], "r") as source:
            mask = np.asarray(source["mask"][()])
        if mask.shape != image.shape or not np.isfinite(mask).all():
            raise ValueError("Supplied mask must be finite and match the image")
        arrays["upstream_mask_uninterpreted"] = mask
        mask_summary["nonzero_pixels"] = int(np.count_nonzero(mask))
    pixel_edges = np.linalg.norm(transform, axis=0)
    field_edges = pixel_edges * np.asarray([image.shape[1], image.shape[0]])
    singular_values = np.linalg.svd(transform, compute_uv=False)
    reference_field = np.asarray(comparison_grid["shape"])[::-1]
    reference_field = reference_field * comparison_grid["pixel_scale_arcsec"]
    # Edge support is descriptive; it does not establish centering/sky coverage.
    edge_support = bool(np.all(field_edges >= reference_field))
    blockers = [
        "flux_unit_and_training_normalization_unresolved",
        "centering_and_usable_coverage_not_reviewed",
        "EPL_reference_versus_SIE_target_requires_scientific_decision",
        "galaxy_galaxy_sources_differ_from_simulated_point_sources",
        "full_noise_covariance_not_available_in_hdf5",
    ]
    if not edge_support:
        blockers.append("native_cutout_edges_smaller_than_canonical_field")
    if "mask" in paths:
        blockers.append("upstream_mask_semantics_not_reviewed")
    report = {
        "array_validation_passed": True,
        "model_input_ready": False,
        "shape_yx": list(image.shape),
        "image_dtype": str(image.dtype),
        "raw_intensity_percentiles_0_1_50_99_100": np.percentile(
            image, [0, 1, 50, 99, 100]
        ).tolist(),
        "unit_attribute": None if unit is None else str(unit),
        "unit_calibration_verified": False,
        "pixel_scale_equal_area_arcsec": float(np.sqrt(abs(np.linalg.det(transform)))),
        "pixel_edge_lengths_xy_arcsec": pixel_edges.tolist(),
        "pixel_anisotropy_ratio": float(singular_values.max() / singular_values.min()),
        "field_edge_lengths_xy_arcsec": field_edges.tolist(),
        "canonical_field_edge_lengths_xy_arcsec": reference_field.tolist(),
        "canonical_edge_support": edge_support,
        "background_rms_raw_units": float(rms),
        "exposure_range": [float(exposure.min()), float(exposure.max())],
        "psf_shape_yx": list(psf.shape),
        "psf_sum_original": psf_sum,
        "psf_sum_normalized": float(arrays["psf_normalized"].sum()),
        "mask": mask_summary,
        "blockers": blockers,
        "science_processing": "native_values_preserved_no_resampling_or_normalization",
    }
    return report, arrays


def target_convention_control() -> dict[str, Any]:
    """Compare lenstronomy SIE and EPL deflections at slope two.

    This checks a parameter-convention limit. It does not validate SIE as a
    model for the observed power-law lenses or translate their uncertainties.
    """
    from lenstronomy.LensModel.Profiles.epl import EPL
    from lenstronomy.LensModel.Profiles.sie import SIE

    positions = np.array([[0.2, 0.5], [0.7, -0.9], [-1.3, 0.4], [1.7, 1.1]])
    maximum_difference = 0.0
    for radius in (0.5, 1.2, 2.0):
        for axis_ratio in (0.5, 0.7, 0.95):
            amplitude = (1 - axis_ratio) / (1 + axis_ratio)
            parameters = {
                "theta_E": radius,
                "e1": amplitude * np.cos(0.6),
                "e2": amplitude * np.sin(0.6),
            }
            sie = np.asarray(SIE().derivatives(*positions.T, **parameters))
            epl = np.asarray(EPL().derivatives(*positions.T, gamma=2.0, **parameters))
            difference = float(np.max(np.abs(sie - epl)))
            maximum_difference = max(maximum_difference, difference)
    tolerance = 1e-6
    return {
        "control": "SIE_versus_EPL_gamma_two_deflections",
        "evaluated_parameter_pairs": 9,
        "positions_per_pair": len(positions),
        "maximum_component_difference_arcsec": maximum_difference,
        "absolute_tolerance_arcsec": tolerance,
        "passed": maximum_difference < tolerance,
        "observed_reference_equivalence_established": False,
    }


def write_preview(
    rows: list[dict[str, Any]], arrays: list[dict[str, np.ndarray]], path: Path
) -> None:
    """Render native science and normalized PSFs; stretches are display-only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(rows), 2, figsize=(8, 3 * len(rows)), squeeze=False)
    for index, (row, products) in enumerate(zip(rows, arrays, strict=True)):
        image = products["image_data"]
        stretch_scale = max(float(np.percentile(np.abs(image), 75)), np.finfo(float).eps)
        display = np.arcsinh(image / stretch_scale)
        axes[index, 0].imshow(display, origin="lower", cmap="gray")
        axes[index, 0].set_title(f"{row['system_id']} {row['band']} — native pixels")
        axes[index, 1].imshow(products["psf_normalized"], origin="lower", cmap="magma")
        axes[index, 1].set_title("PSF / original sum")
        for axis in axes[index]:
            axis.set_xlabel("Pixel x")
            axis.set_ylabel("Pixel y")
    figure.suptitle("HST compatibility development only — science display uses asinh")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def run_pilot(
    manifest_path: Path, data_root: Path, output: Path, *, offline: bool
) -> dict[str, Any]:
    """Acquire, audit, and write per-system evidence without invoking a CNN."""
    document = load_manifest(manifest_path)
    acquired = acquire_assets(document, data_root, offline=offline)
    rows, native_products = [], []
    products_directory = output / "products"
    products_directory.mkdir(parents=True, exist_ok=True)
    for system in document["systems"]:
        identifier = system["system_id"]
        audit, arrays = inspect_pair(acquired[identifier], document["comparison_grid"])
        product_path = products_directory / f"{identifier}_{system['band']}.npz"
        np.savez_compressed(product_path, **arrays)
        rows.append(
            {
                "system_id": identifier,
                "survey": system["survey"],
                "band": system["band"],
                "role": document["role"],
                "reference": system["reference"],
                "sources": {
                    kind: {"path": str(path), "sha256": file_sha256(path)}
                    for kind, path in acquired[identifier].items()
                },
                "native_products": {
                    "path": str(product_path),
                    "sha256": file_sha256(product_path),
                },
                **audit,
            }
        )
        native_products.append(arrays)
    control = target_convention_control()
    report = {
        "schema": "hst_compatibility_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": document["role"],
        "reserved_system_ids": [row["system_id"] for row in rows],
        "manifest_sha256": file_sha256(manifest_path),
        "implementation_sha256": file_sha256(Path(__file__)),
        "source_commit": document["source_commit"],
        "environment": {
            "python": platform.python_version(),
            "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
            **{
                package: importlib.metadata.version(package)
                for package in ("numpy", "h5py", "lenstronomy", "matplotlib")
            },
        },
        "system_count": len(rows),
        "array_validation_passed": all(row["array_validation_passed"] for row in rows),
        "model_input_ready": False,
        "predictions_generated": False,
        "reference_convention_control": control,
        "systems": rows,
    }
    write_preview(rows, native_products, output / "preview.png")
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = [
        "# HST compatibility pilot",
        "",
        f"{len(rows)} independent development systems; no predictions or final evaluation.",
        "Native science arrays preserved. Every system remains unqualified for scoring.",
        "",
        "| System | Band | Shape | Pixel scale (arcsec) | Field edges (arcsec) | PSF sum |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        field = row["field_edge_lengths_xy_arcsec"]
        table.append(
            f"| {row['system_id']} | {row['band']} | {row['shape_yx']} | "
            f"{row['pixel_scale_equal_area_arcsec']:.5f} | "
            f"{field[0]:.2f}, {field[1]:.2f} | {row['psf_sum_original']:.6f} |"
        )
    table.extend(
        [
            "",
            "See audit.json for exact references, hashes, environment, and per-system blockers.",
            "Do not reuse these reserved systems as untouched confirmation data.",
        ]
    )
    (output / "audit.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Run the bounded pilot, optionally using verified cached files only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true")
    arguments = parser.parse_args()
    report = run_pilot(
        arguments.manifest, arguments.data_root, arguments.output, offline=arguments.offline
    )
    print(
        f"Audited {report['system_count']} HST systems; arrays validated; "
        "model-input readiness is blocked. No CNN predictions generated."
    )
    if not report["reference_convention_control"]["passed"]:
        raise SystemExit("SIE/EPL convention control failed; inspect audit.json")


if __name__ == "__main__":
    main()
