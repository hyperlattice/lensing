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
    try:
        wcs = WCS(header)
        # pixel_scale_matrix returns scale in degrees/pixel
        # We take the norm of the first column (or average of diag) as the primary scale
        scale_deg = np.sqrt(wcs.pixel_scale_matrix[0,0]**2 + wcs.pixel_scale_matrix[1,0]**2)
        return scale_deg * 3600.0  # Convert deg to arcsec
    except Exception:
        # Fallback to common CASTLES/FITS keys
        for key in ["D001SCAL", "CDELT1", "PIXSCALE"]:
            val = header.get(key)
            if val:
                # Most of these are already in arcsec or deg.
                # This is a simplification; in a real pipeline we'd check units.
                return float(val) if float(val) > 0.001 else float(val) * 3600.0
    return None

rows = []

for fits_file in ROOT.rglob("*.fits"):
    header = fits.getheader(fits_file)
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
        "original_height": fits.getdata(fits_file).shape[-2],
        "original_width": fits.getdata(fits_file).shape[-1],
        "crval1": header.get("CRVAL1"),
        "crval2": header.get("CRVAL2"),
        "cd1_1": header.get("CD1_1"),
        "cd1_2": header.get("CD1_2"),
        "cd2_1": header.get("CD2_1"),
        "cd2_2": header.get("CD2_2"),
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
