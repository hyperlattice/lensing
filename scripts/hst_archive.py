"""Inspect a wider HST archive cutout using bounded, cached HTTP range reads.

This follows the fixed pilot's single archive probe. It preserves the native
SCI/WHT pixels and WCS, and does not run a CNN or reuse the Dinos cutout's PSF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.nddata.utils import overlap_slices
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from scripts.hst_pilot import DEFAULT_DATA, DEFAULT_MANIFEST, file_sha256, load_manifest

HEADER_PROBE_BYTES = 1_048_576
MAX_RANGE_BYTES = 8_000_000
FITS_BLOCK_BYTES = 2880
OUTPUT = Path("results/reports/hst_compatibility_pilot/archive")


def read_range(
    url: str, start: int, length: int, total: int, cache: Path, *, offline: bool
) -> tuple[bytes, dict[str, Any]]:
    """Verify exact HTTP range identity, or exact cached bytes during replay."""
    if start < 0 or length <= 0 or length > MAX_RANGE_BYTES or start + length > total:
        raise ValueError("Archive range is outside the declared file or download limit")
    identity = hashlib.sha256(url.encode()).hexdigest()[:16]
    path = cache / f"{identity}_{start}_{length}.bin"
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        content = path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["url"] != url or metadata["start"] != start or metadata["length"] != length:
            raise ValueError("Archive cache identity mismatch")
        if metadata["total_file_bytes"] != total:
            raise ValueError("Archive cache file-size identity mismatch")
        if len(content) != length or hashlib.sha256(content).hexdigest() != metadata["sha256"]:
            raise ValueError("Archive cache byte verification failed")
        return content, metadata
    if offline:
        raise FileNotFoundError(f"Missing verified archive range: {path}")
    end = start + length - 1
    # Explicit range in the URL prevents intermediaries reusing a different range.
    request_url = url + "&hst_pilot_range=" + f"{start}-{end}"
    request = urllib.request.Request(
        request_url,
        headers={"User-Agent": "hst-compatibility-pilot", "Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        expected_range = f"bytes {start}-{end}/{total}"
        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
            raise ValueError("Server did not return the exact requested FITS range")
        content = response.read(length + 1)
        metadata = {
            "url": url,
            "start": start,
            "length": length,
            "total_file_bytes": total,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    if len(content) != length:
        raise ValueError("Archive range was truncated")
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return content, metadata


def parse_header(content: bytes) -> tuple[fits.Header, int]:
    """Read complete FITS header blocks, stopping at an aligned END card."""
    for offset in range(0, len(content) - 79, 80):
        if content[offset : offset + 8] == b"END     ":
            size = ((offset + 80 + FITS_BLOCK_BYTES - 1) // FITS_BLOCK_BYTES) * FITS_BLOCK_BYTES
            if size > len(content):
                raise ValueError("FITS header padding is truncated")
            return fits.Header.fromstring(content[:size].decode("ascii")), size
    raise ValueError("FITS header exceeds bounded header probe or lacks END")


def data_byte_count(header: fits.Header) -> int:
    """Return FITS payload size excluding block padding for an ordinary HDU."""
    axis_count = int(header["NAXIS"])
    elements = 0
    if axis_count:
        elements = int(
            np.prod([int(header[f"NAXIS{index}"]) for index in range(1, axis_count + 1)])
        )
    return (
        (elements + int(header.get("PCOUNT", 0)))
        * int(header.get("GCOUNT", 1))
        * abs(int(header["BITPIX"]))
        // 8
    )


def extract_rows(content: bytes, header: fits.Header, x_slice: slice, row_count: int) -> np.ndarray:
    """Decode complete native rows, then slice columns; no interpolation occurs."""
    types = {8: "u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}
    dtype = np.dtype(types[int(header["BITPIX"])])
    columns = int(header["NAXIS1"])
    if len(content) != row_count * columns * dtype.itemsize:
        raise ValueError("FITS row payload size mismatch")
    rows = np.frombuffer(content, dtype=dtype).reshape(row_count, columns)
    cutout = rows[:, x_slice].copy()
    scale, zero = float(header.get("BSCALE", 1)), float(header.get("BZERO", 0))
    if scale != 1 or zero != 0:
        cutout = cutout.astype(float) * scale + zero
    return cutout


def unit_evidence(primary: fits.Header, science: fits.Header) -> dict[str, Any]:
    """Interpret WFPC2 rate metadata without applying calibration.

    WFPC2 native counts are DN. Drizzle output cards establish rate units;
    BUNIT may retain a pre-drizzle label. Separately processed Dinos values
    are not certified by this archive product's metadata.
    """
    drizzle_units = sorted(
        {str(primary[key]).strip() for key in primary if re.fullmatch(r"D\d{3}OUUN", key)}
    )
    gain = primary.get("ATODGAIN")
    supported = (
        str(primary.get("INSTRUME", "")).strip() == "WFPC2"
        and str(science.get("BUNIT", "")).strip().lower() in {"counts", "dn"}
        and drizzle_units == ["cps"]
        and gain is not None
        and np.isfinite(gain)
        and gain > 0
    )
    return {
        "header_supported_unit": "DN_per_second" if supported else "unresolved",
        "candidate_electron_rate_multiplier": float(gain) if supported else None,
        "interpretation_uses_drizzle_metadata": supported,
        "conversion_applied": False,
        "Dinos_HDF5_units_established": False,
        "documentation": "https://hst-docs.stsci.edu/hstdhb/4-hst-data-analysis/4-6-analyzing-hst-images",
    }


def write_preview(science: np.ndarray, valid: np.ndarray, path: Path) -> None:
    """Display native values and positive-weight support without altering either."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = max(float(np.nanpercentile(np.abs(science), 75)), np.finfo(float).eps)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(np.arcsinh(science / scale), origin="lower", cmap="gray")
    axes[0].set_title("SDSSJ0037-0942 — native archive pixels\n12.84 arcsec; asinh display only")
    axes[1].imshow(valid, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Finite science and positive WHT\nDoes not certify absence of artifacts")
    for axis in axes:
        axis.set_xlabel("Native pixel x")
        axis.set_ylabel("Native pixel y")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def select_probe(manifest: Path, product_id: str | None) -> tuple[dict[str, Any], bool]:
    """Select an explicit archive product; acquisition never qualifies a model input."""
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if document.get("schema") != "hst_archive_products_v1":
        document = load_manifest(manifest)
        probe = document["archive_probe"]
        if product_id is not None:
            raise ValueError("--product-id requires an hst_archive_products_v1 manifest")
        if probe["system_id"] not in {row["system_id"] for row in document["systems"]}:
            raise ValueError("Archive probe must use a reserved pilot system")
        if probe["role"] != "compatibility_development":
            raise ValueError("Archive probe cannot use a final system")
        return probe, False
    products = document["products"]
    identifiers = [row["observation_id"] for row in products]
    if not products or len(set(identifiers)) != len(identifiers):
        raise ValueError("Archive product ledger must have unique observation IDs")
    if any(not re.fullmatch(r"[A-Za-z0-9_+-]+", identifier) for identifier in identifiers):
        raise ValueError("Invalid archive observation ID")
    if product_id is None:
        if len(products) != 1:
            raise ValueError("Select --product-id when multiple archive products are listed")
        product_id = identifiers[0]
    selected = [row for row in products if row["observation_id"] == product_id]
    if len(selected) != 1:
        raise ValueError("Requested archive product is not in the explicit ledger")
    probe = selected[0]
    if probe["role"] not in {"compatibility_development", "confirmation_candidate"}:
        raise ValueError("Archive acquisition needs an explicit development/candidate role")
    if not re.fullmatch(r"[A-Za-z0-9_+-]+", probe["system_id"]):
        raise ValueError("Invalid physical lens ID")
    return probe, True


def run_probe(
    manifest: Path, cache: Path, output: Path, *, offline: bool, product_id: str | None = None
) -> dict[str, Any]:
    """Retrieve native wider SCI/WHT coverage for one explicitly selected product."""
    probe, product_ledger = select_probe(manifest, product_id)
    if product_ledger:
        output = output / probe["observation_id"]
    uri = probe["product_uri"]
    if not re.fullmatch(r"mast:HST/product/[A-Za-z0-9_]+\.fits", uri):
        raise ValueError("Archive probe must identify a public HST FITS product")
    url = "https://mast.stsci.edu/api/v0.1/Download/file?" + urllib.parse.urlencode({"uri": uri})
    total = int(probe["product_bytes"])
    records, headers, offsets = [], {}, {}
    offset = 0
    primary = None
    while offset < total:
        content, record = read_range(
            url, offset, min(HEADER_PROBE_BYTES, total - offset), total, cache, offline=offline
        )
        records.append(record)
        header, header_bytes = parse_header(content)
        name = str(header.get("EXTNAME", "PRIMARY")).strip()
        if primary is None:
            primary = header
        headers[name] = header
        offsets[name] = offset + header_bytes
        if {"SCI", "WHT"}.issubset(headers):
            break
        size = data_byte_count(header)
        offset += (
            header_bytes + ((size + FITS_BLOCK_BYTES - 1) // FITS_BLOCK_BYTES) * FITS_BLOCK_BYTES
        )
    if not {"SCI", "WHT"}.issubset(headers) or primary is None:
        raise ValueError("Archive product lacks SCI/WHT support")
    science_header = headers["SCI"]
    wcs = WCS(science_header).celestial
    pixel_scales = proj_plane_pixel_scales(wcs) * 3600
    minimum_field = float(probe["minimum_field_arcsec"])
    if not np.isfinite(minimum_field) or minimum_field <= 0:
        raise ValueError("Archive field size must be finite and positive")
    dimensions_xy = np.ceil(minimum_field / pixel_scales).astype(int)
    center_xy = wcs.world_to_pixel_values(probe["ra_deg"], probe["dec_deg"])
    full_shape = (int(science_header["NAXIS2"]), int(science_header["NAXIS1"]))
    slices, _ = overlap_slices(
        full_shape, tuple(dimensions_xy[::-1]), center_xy[::-1], mode="strict"
    )
    y_slice, x_slice = slices
    row_count = y_slice.stop - y_slice.start
    products = {}
    for name in ("SCI", "WHT"):
        header = headers[name]
        shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
        if int(header["NAXIS"]) != 2 or shape != full_shape:
            raise ValueError("SCI and WHT must share a two-dimensional native grid")
        corners = np.array([[0, 0], [shape[1] - 1, shape[0] - 1]])
        if not np.allclose(
            WCS(header).celestial.all_pix2world(corners, 0),
            wcs.all_pix2world(corners, 0),
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError("SCI and WHT coordinate transforms differ")
        item_size = abs(int(header["BITPIX"])) // 8
        row_bytes = shape[1] * item_size
        content, record = read_range(
            url,
            offsets[name] + y_slice.start * row_bytes,
            row_count * row_bytes,
            total,
            cache,
            offline=offline,
        )
        records.append(record)
        products[name] = extract_rows(content, header, x_slice, row_count)
    etags = {row["etag"] for row in records if row["etag"] is not None}
    if len(etags) > 1:
        raise ValueError("Archive source ETag changed between range reads")
    valid = np.isfinite(products["SCI"]) & np.isfinite(products["WHT"]) & (products["WHT"] > 0)
    if not np.any(valid):
        raise ValueError("Requested archive cutout has no finite positive-weight support")
    supported_science = np.where(valid, products["SCI"], np.nan)
    brightest_yx = np.asarray(np.unravel_index(np.nanargmax(supported_science), valid.shape))
    catalog_xy = np.asarray(center_xy) - np.asarray([x_slice.start, y_slice.start])
    peak_offset_xy = (brightest_yx[::-1] - catalog_xy) * pixel_scales
    output.mkdir(parents=True, exist_ok=True)
    primary_copy = primary.copy()
    primary_copy["NEXTEND"] = 3
    hdus = [fits.PrimaryHDU(header=primary_copy)]
    cutout_wcs = wcs.slice(slices)
    for name in ("SCI", "WHT"):
        header = headers[name].copy()
        header.update(cutout_wcs.to_header())
        # Existing CD terms must not override the updated PC/CDELT representation.
        for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
            header.remove(key, ignore_missing=True)
        hdus.append(fits.ImageHDU(products[name], header=header, name=name))
    hdus.append(
        fits.ImageHDU(valid.astype(np.uint8), header=cutout_wcs.to_header(), name="COVERAGE")
    )
    path = output / f"{probe['system_id']}_native_wide.fits"
    fits.HDUList(hdus).writeto(path, overwrite=True, checksum=True)
    drizzle_units = sorted(
        {str(primary[key]).strip() for key in primary if re.fullmatch(r"D\d{3}OUUN", key)}
    )
    report = {
        "schema": "hst_archive_compatibility_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "system_id": probe["system_id"],
        "role": probe["role"],
        "manifest_sha256": file_sha256(manifest),
        "implementation_sha256": file_sha256(Path(__file__)),
        "archive_probe": probe,
        "archive_shape_yx": list(full_shape),
        "cutout_shape_yx": list(products["SCI"].shape),
        "pixel_scales_xy_arcsec": pixel_scales.tolist(),
        "cutout_field_edges_xy_arcsec": (dimensions_xy * pixel_scales).tolist(),
        "archive_slices_yx": [[part.start, part.stop] for part in slices],
        "covered_fraction": float(valid.mean()),
        "centering_diagnostic": {
            "catalog_position_xy_pixels": catalog_xy.tolist(),
            "brightest_pixel_yx": brightest_yx.tolist(),
            "brightest_to_catalog_offset_arcsec_approx": float(np.linalg.norm(peak_offset_xy)),
            "brightest_pixel_is_not_a_verified_lens_center": True,
            "recentering_applied": False,
        },
        "SCI_BUNIT": science_header.get("BUNIT"),
        "drizzle_output_units": drizzle_units,
        "ATODGAIN_electrons_per_DN": primary.get("ATODGAIN"),
        "EXPTIME_seconds": primary.get("EXPTIME"),
        "PHOTFLAM": science_header.get("PHOTFLAM"),
        "WHTTYPE": primary.get("WHTTYPE"),
        "unit_evidence": unit_evidence(primary, science_header),
        "unit_conversion_applied": False,
        "PSF_transferred_from_Dinos": False,
        "model_input_ready": False,
        "predictions_generated": False,
        "source_ranges": records,
        "retrieved_bytes_including_header_probes": sum(row["length"] for row in records),
        "native_cutout": {"path": str(path), "sha256": file_sha256(path)},
        "blockers": [
            "interpret_legacy_BUNIT_with_drizzle_metadata_and_instrument_documentation",
            "new_archive_processing_requires_matching_PSF_and_noise_review",
            "background_and_training_photometry_not_matched",
            "observational_reference_and_source_mismatch_remain",
        ],
    }
    write_preview(products["SCI"], valid, output / "preview.png")
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Run one wider-image archive probe, with verified offline replay available."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_DATA / "archive_ranges")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--product-id", help="Observation ID in an hst_archive_products_v1 manifest"
    )
    arguments = parser.parse_args()
    report = run_probe(
        arguments.manifest,
        arguments.cache,
        arguments.output,
        offline=arguments.offline,
        product_id=arguments.product_id,
    )
    print(
        f"Wider native cutout: {report['cutout_shape_yx']}; "
        f"coverage {report['covered_fraction']:.3f}; no CNN scoring."
    )


if __name__ == "__main__":
    main()
