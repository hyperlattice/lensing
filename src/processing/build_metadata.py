import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


ROOT = Path("data/raw/castles")
OUTPUT = Path("data/labels/derived/metadata.csv")


def sample_id_from_path(path, root=ROOT):
    """Build a stable ID from every path component below the raw-data root."""
    path = Path(path)
    relative = path.relative_to(Path(root))
    return "_".join(relative.with_suffix("").parts)


def normalize_date(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if pd.isna(value):
        return None

    parsed = pd.to_datetime(value, errors="coerce", dayfirst="/" in str(value))
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _clean_header_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.upper() in {"INDEF", "N/A", "NONE", "NULL"}:
            return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _first_header_value(header, keys):
    for key in keys:
        value = _clean_header_value(header.get(key))
        if value is not None:
            return value
    return None


def _as_float(value):
    value = _clean_header_value(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _valid_celestial_header(header):
    ctype1 = str(header.get("CTYPE1", "")).upper()
    ctype2 = str(header.get("CTYPE2", "")).upper()
    return ("RA" in ctype1 and "DEC" in ctype2) or ("DEC" in ctype1 and "RA" in ctype2)


def _matrix_scales(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    matrix = matrix[:2, :2]
    if not np.isfinite(matrix).all():
        return None

    axis_scales_deg = np.linalg.norm(matrix, axis=0)
    if not np.all(axis_scales_deg > 0):
        return None
    return tuple(float(value * 3600.0) for value in axis_scales_deg)


def pixel_scale_axes(header):
    """Return (x-scale, y-scale, source) in arcsec/pixel."""
    if _valid_celestial_header(header):
        cd_keys = ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]
        cd_values = [header.get(key) for key in cd_keys]
        if all(value is not None for value in cd_values):
            scales = _matrix_scales(np.asarray(cd_values, dtype=float).reshape(2, 2))
            if scales is not None:
                return scales[0], scales[1], "celestial_cd_matrix"

        try:
            scales = _matrix_scales(WCS(header, fix=False).pixel_scale_matrix)
            if scales is not None:
                return scales[0], scales[1], "celestial_wcs"
        except Exception:
            pass

    cdelt_values = [header.get("CDELT1"), header.get("CDELT2")]
    if all(value is not None for value in cdelt_values):
        try:
            values = np.abs(np.asarray(cdelt_values, dtype=float))
        except (TypeError, ValueError):
            values = np.asarray([], dtype=float)
        if values.shape == (2,) and np.isfinite(values).all() and np.all(values > 0):
            return float(values[0] * 3600.0), float(values[1] * 3600.0), "explicit_cdelt"

    for key in ["D001SCAL", "PIXSCALE"]:
        value = _as_float(header.get(key))
        if value is not None and value > 0:
            return value, value, f"explicit_{key.lower()}"

    return None, None, "unavailable"


def calculate_pixel_scale(header):
    """Return the mean angular scale in arcsec/pixel, or None."""
    x_scale, y_scale, _ = pixel_scale_axes(header)
    if x_scale is None or y_scale is None:
        return None
    return float(np.mean([x_scale, y_scale]))


def _select_science_hdu(hdul):
    for index, hdu in enumerate(hdul):
        if str(hdu.name).upper() in {"SCI", "SCIENCE"} and hdu.data is not None:
            if np.asarray(hdu.data).ndim == 2:
                return index, hdu
    if hdul[0].data is not None and np.asarray(hdul[0].data).ndim == 2:
        return 0, hdul[0]
    raise ValueError("No two-dimensional science image found")


def _instrument_filter(header, instrument):
    instrument = str(instrument or "").upper()
    if instrument in {"ACS", "WFPC2"}:
        keys = ["FILTNAM1", "FILTNAM2", "FILTER", "PHOTMODE"]
    else:
        keys = ["FILTER", "FILTNAM1", "FILTNAM2", "PHOTMODE"]

    value = _first_header_value(header, keys)
    if value is None or str(value).strip() in {"0", "0.0"}:
        return None
    return value


def _instrument_detector(header, instrument):
    instrument = str(instrument or "").upper()
    if instrument == "NICMOS":
        return _first_header_value(header, ["CAMERA", "DETECTOR"])
    return _first_header_value(header, ["DETECTOR", "CAMERA"])


def _merged_header(primary, science):
    merged = primary.copy()
    if science is not primary:
        for card in science.cards:
            if card.keyword:
                merged[card.keyword] = card.value
    return merged


def _metadata_row(fits_file, root):
    with fits.open(fits_file) as hdul:
        science_index, science_hdu = _select_science_hdu(hdul)
        header = _merged_header(hdul[0].header, science_hdu.header)
        image_shape = np.asarray(science_hdu.data).shape

    relative = Path(fits_file).relative_to(Path(root))
    if len(relative.parts) < 3:
        raise ValueError(f"Expected lens/FIT/instrument path structure: {fits_file}")
    lens_name = relative.parts[0]
    instrument_dir = relative.parts[-2]
    instrument = _first_header_value(header, ["INSTRUME"])
    x_scale, y_scale, scale_source = pixel_scale_axes(header)
    selected_filter = _instrument_filter(header, instrument)
    detector = _instrument_detector(header, instrument)
    target_id = _first_header_value(header, ["TARGNAME", "OBJECT"])
    observation_id = _first_header_value(header, ["OBS_ID", "OBSID", "ROOTNAME", "FILENAME"])

    matrix = None
    matrix_keys = ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]
    if all(header.get(key) is not None for key in matrix_keys):
        matrix = [
            [_as_float(header.get("CD1_1")), _as_float(header.get("CD1_2"))],
            [_as_float(header.get("CD2_1")), _as_float(header.get("CD2_2"))],
        ]
        if any(value is None for row in matrix for value in row):
            matrix = None

    return {
        "sample_id": sample_id_from_path(fits_file, root),
        "lens_id": lens_name,
        "raw_path": Path(fits_file).as_posix(),
        "lens": lens_name,
        "image": Path(fits_file).name,
        "instrument_dir": instrument_dir,
        "science_hdu": science_index,
        "science_hdu_name": str(science_hdu.name),
        "telescope": _first_header_value(header, ["TELESCOP"]),
        "instrument": instrument,
        "detector": detector,
        "camera": _first_header_value(header, ["CAMERA"]),
        "filter": selected_filter,
        "filter_name": selected_filter,
        "exposure": _as_float(header.get("EXPTIME")),
        "exposure_s": _as_float(header.get("EXPTIME")),
        "bunit": _first_header_value(header, ["BUNIT"]),
        "image_unit": _first_header_value(header, ["BUNIT"]) or "unknown",
        "pixel_scale": None if x_scale is None else float(np.mean([x_scale, y_scale])),
        "pixel_scale_x": x_scale,
        "pixel_scale_y": y_scale,
        "pixel_scale_source": scale_source,
        "wcs_status": "valid_celestial" if scale_source.startswith("celestial_") else scale_source,
        "wcs_matrix_deg_per_pixel": None if matrix is None else json.dumps(matrix, separators=(",", ":")),
        "original_height": image_shape[-2],
        "original_width": image_shape[-1],
        "crval1": _as_float(header.get("CRVAL1")),
        "crval2": _as_float(header.get("CRVAL2")),
        "crpix1": _as_float(header.get("CRPIX1")),
        "crpix2": _as_float(header.get("CRPIX2")),
        "ctype1": _first_header_value(header, ["CTYPE1"]),
        "ctype2": _first_header_value(header, ["CTYPE2"]),
        "cd1_1": _as_float(header.get("CD1_1")),
        "cd1_2": _as_float(header.get("CD1_2")),
        "cd2_1": _as_float(header.get("CD2_1")),
        "cd2_2": _as_float(header.get("CD2_2")),
        "orientation": _as_float(header.get("ORIENTAT")),
        "photflam": _as_float(header.get("PHOTFLAM")),
        "photfnu": _as_float(header.get("PHOTFNU")),
        "photzpt": _as_float(header.get("PHOTZPT")),
        "photplam": _as_float(header.get("PHOTPLAM")),
        "photbw": _as_float(header.get("PHOTBW")),
        "photmode": _first_header_value(header, ["PHOTMODE"]),
        "target_id": target_id,
        "observation_id": observation_id,
        "proposal_id": _first_header_value(header, ["PROPOSID"]),
        "gain_e_per_adu": _as_float(_first_header_value(header, ["GAIN", "CCDGAIN"])),
        "read_noise_e": _as_float(_first_header_value(header, ["READNOIS", "READNSE", "READNOISE"])),
        "observation_date": normalize_date(header.get("DATE-OBS")),
        "calstage": _first_header_value(header, ["CALSTAGE"]),
        "biascorr": _first_header_value(header, ["BIASCORR"]),
        "darkcorr": _first_header_value(header, ["DARKCORR"]),
        "flatcorr": _first_header_value(header, ["FLATCORR"]),
        "crcorr": _first_header_value(header, ["CRCORR"]),
        "photcorr": _first_header_value(header, ["PHOTCORR"]),
        "calibration_source": _first_header_value(header, ["PHOTTAB", "CAL_VER"]),
        "metadata_source": "FITS primary/science header",
        "notes": "",
    }


def build_metadata(root=ROOT, output=OUTPUT):
    """Scan raw FITS files and write the image-level metadata manifest."""
    root = Path(root)
    output = Path(output)
    if not root.exists():
        raise FileNotFoundError(f"Raw FITS directory does not exist: {root}")

    fits_files = sorted(root.rglob("*.fits"))
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found below: {root}")

    rows = [_metadata_row(path, root) for path in fits_files]
    dataframe = pd.DataFrame(rows).sort_values(
        ["lens", "instrument_dir", "image"]
    ).reset_index(drop=True)

    if dataframe["sample_id"].duplicated().any():
        duplicates = dataframe.loc[dataframe["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample IDs found: {duplicates[:5]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)
    return dataframe


def main():
    dataframe = build_metadata()
    print(dataframe.head())
    print(f"\\nSaved {len(dataframe)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
