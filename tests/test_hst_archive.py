"""Check partial FITS decoding and exact-range cache behavior."""

import hashlib
import json

import numpy as np
import pytest
from astropy.io import fits

from scripts.hst_archive import (
    data_byte_count,
    extract_rows,
    parse_header,
    read_range,
    select_probe,
    unit_evidence,
)


def test_archive_product_ledger_requires_an_explicit_unique_selection(tmp_path):
    path = tmp_path / "products.json"
    rows = [
        {"system_id": "LENS", "observation_id": name, "role": "confirmation_candidate"}
        for name in ("visit1_F606W", "visit2_F814W")
    ]
    path.write_text(json.dumps({"schema": "hst_archive_products_v1", "products": rows}))
    with pytest.raises(ValueError, match="Select --product-id"):
        select_probe(path, None)
    probe, separate = select_probe(path, "visit2_F814W")
    assert separate and probe == rows[1]
    rows[1]["observation_id"] = rows[0]["observation_id"]
    path.write_text(json.dumps({"schema": "hst_archive_products_v1", "products": rows}))
    with pytest.raises(ValueError, match="unique"):
        select_probe(path, "visit1_F606W")


def test_header_parser_stops_before_binary_data():
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = 8
    header["NAXIS2"] = 6
    content = header.tostring().encode("ascii")
    parsed, size = parse_header(content + b"\xff" * 100)
    assert size == len(content) == 2880
    assert data_byte_count(parsed) == 8 * 6 * 4
    with pytest.raises(ValueError, match="truncated"):
        parse_header(content[:1000])


def test_big_endian_rows_are_cropped_without_resampling():
    rows = np.arange(24, dtype=">f4").reshape(3, 8)
    header = fits.Header({"BITPIX": -32, "NAXIS1": 8})
    crop = extract_rows(rows.tobytes(), header, slice(2, 6), 3)
    np.testing.assert_array_equal(crop, rows[:, 2:6])
    header["BSCALE"] = 2
    header["BZERO"] = 3
    scaled = extract_rows(rows.tobytes(), header, slice(2, 6), 3)
    np.testing.assert_array_equal(scaled, rows[:, 2:6] * 2 + 3)


def test_verified_offline_range_and_corruption_detection(tmp_path):
    url = "https://example.org/file?uri=public"
    identity = hashlib.sha256(url.encode()).hexdigest()[:16]
    path = tmp_path / f"{identity}_10_4.bin"
    path.write_bytes(b"abcd")
    metadata = {
        "url": url,
        "start": 10,
        "length": 4,
        "sha256": hashlib.sha256(b"abcd").hexdigest(),
        "total_file_bytes": 100,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    content, _ = read_range(url, 10, 4, 100, tmp_path, offline=True)
    assert content == b"abcd"
    path.write_bytes(b"abce")
    with pytest.raises(ValueError, match="verification failed"):
        read_range(url, 10, 4, 100, tmp_path, offline=True)


def test_range_bounds_and_missing_cache_fail_before_network(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        read_range("https://example.org/?x=1", 99, 2, 100, tmp_path, offline=True)
    with pytest.raises(FileNotFoundError, match="Missing"):
        read_range("https://example.org/?x=1", 0, 2, 100, tmp_path, offline=True)


def test_legacy_count_label_requires_drizzle_and_gain_evidence():
    primary = fits.Header({"INSTRUME": "WFPC2", "ATODGAIN": 7, "D001OUUN": "cps"})
    science = fits.Header({"BUNIT": "counts"})
    result = unit_evidence(primary, science)
    assert result["header_supported_unit"] == "DN_per_second"
    assert result["candidate_electron_rate_multiplier"] == 7
    assert not result["conversion_applied"]
    assert not result["Dinos_HDF5_units_established"]
    primary["D002OUUN"] = "counts"
    assert unit_evidence(primary, science)["header_supported_unit"] == "unresolved"
    del primary["D002OUUN"]
    primary["ATODGAIN"] = -1
    assert unit_evidence(primary, science)["header_supported_unit"] == "unresolved"
