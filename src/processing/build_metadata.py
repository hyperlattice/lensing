from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path("data/raw/castles")
OUTPUT = Path("data/labels/metadata.csv")

def sample_id_from_path(path):
    relative = path.relative_to(ROOT)
    return "_".join(relative.with_suffix("").parts)

def normalize_date(value):
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst="/" in str(value))
    if pd.isna(parsed):
        return value
    return parsed.date().isoformat()

def calculate_pixel_scale(header):
    """
    Calculates the pixel scale in arcseconds per pixel.
    Tries WCS CD matrix first, then falls back to known header keys.
    """
    cd_keys = ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]
    cd_values = [header.get(key) for key in cd_keys]
    if all(value is not None for value in cd_values):
        matrix = np.asarray(cd_values, dtype=float).reshape(2, 2)
        if np.isfinite(matrix).all():
            axis_scales_deg = np.linalg.norm(matrix, axis=0)
            if np.all(axis_scales_deg > 0):
                return float(axis_scales_deg.mean() * 3600.0)

    try:
        wcs = WCS(header, fix=False)
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=float)
        if matrix.shape[0] >= 2 and matrix.shape[1] >= 2 and np.isfinite(matrix).all():
            axis_scales_deg = np.linalg.norm(matrix[:2, :2], axis=0)
            if np.all(axis_scales_deg > 0):
                return float(axis_scales_deg.mean() * 3600.0)
    except Exception:
        pass

    cdelt_values = [header.get("CDELT1"), header.get("CDELT2")]
    if all(value is not None for value in cdelt_values):
        values = np.abs(np.asarray(cdelt_values, dtype=float))
        if np.isfinite(values).all() and np.all(values > 0):
            return float(values.mean() * 3600.0)

    for key in ["D001SCAL", "PIXSCALE"]:
        value = header.get(key)
        if value is not None and np.isfinite(float(value)) and float(value) > 0:
            return float(value)
    return None

rows = []

for fits_file in ROOT.rglob("*.fits"):
    with fits.open(fits_file) as hdul:
        header = hdul[0].header
        image_shape = hdul[0].data.shape
    lens_name = fits_file.parts[-4]
    relative_path = fits_file.relative_to(Path("."))

    # Robust scale calculation
    px_scale = calculate_pixel_scale(header)

    rows.append({
        "sample_id": sample_id_from_path(fits_file),
        "lens_id": lens_name,
        "raw_path": relative_path.as_posix(),
        "lens": lens_name,
        "image": fits_file.name,
        "instrument_dir": fits_file.parts[-2],
        "telescope": header.get("TELESCOP"),
        "instrument": header.get("INSTRUME"),
        "detector": header.get("DETECTOR"),
        "filter": header.get("FILTER"),
        "exposure": header.get("EXPTIME"),
        "bunit": header.get("BUNIT"),
        "pixel_scale": px_scale,
        "original_height": image_shape[-2],
        "original_width": image_shape[-1],
        "crval1": header.get("CRVAL1"),
        "crval2": header.get("CRVAL2"),
        "crpix1": header.get("CRPIX1"),
        "crpix2": header.get("CRPIX2"),
        "ctype1": header.get("CTYPE1"),
        "ctype2": header.get("CTYPE2"),
        "cd1_1": header.get("CD1_1"),
        "cd1_2": header.get("CD1_2"),
        "cd2_1": header.get("CD2_1"),
        "cd2_2": header.get("CD2_2"),
        "orientation": header.get("ORIENTAT"),
        "photflam": header.get("PHOTFLAM"),
        "photfnu": header.get("PHOTFNU"),
        "photzpt": header.get("PHOTZPT"),
        "photmode": header.get("PHOTMODE"),
        "observation_date": normalize_date(header.get("DATE-OBS")),
        "calstage": header.get("CALSTAGE"),
        "biascorr": header.get("BIASCORR"),
        "darkcorr": header.get("DARKCORR"),
        "flatcorr": header.get("FLATCORR"),
        "crcorr": header.get("CRCORR"),
        "photcorr": header.get("PHOTCORR"),
        "notes": ""
    })

df = pd.DataFrame(rows).sort_values(["lens", "instrument_dir", "image"]).reset_index(drop=True)

if df["sample_id"].duplicated().any():
    duplicates = df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()
    raise ValueError(f"Duplicate sample IDs found: {duplicates[:5]}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print(df.head())
print(f"\nSaved {len(df)} rows to {OUTPUT}")
