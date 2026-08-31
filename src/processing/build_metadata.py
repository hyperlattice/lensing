from astropy.io import fits
import pandas as pd
from pathlib import Path

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


rows = []

for fits_file in ROOT.rglob("*.fits"):
    header = fits.getheader(fits_file)

    lens_name = fits_file.parts[-4]
    relative_path = fits_file.relative_to(Path("."))

    rows.append({
        "sample_id": sample_id_from_path(fits_file),
        "raw_path": relative_path.as_posix(),
        "lens": lens_name,
        "image": fits_file.name,
        "instrument_dir": fits_file.parts[-2],
        "telescope": header.get("TELESCOP"),
        "instrument": header.get("INSTRUME"),
        "detector": header.get("DETECTOR"),
        "filter": header.get("FILTER"),
        "exposure": header.get("EXPTIME"),
        "pixel_scale": header.get("D001SCAL"),
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
