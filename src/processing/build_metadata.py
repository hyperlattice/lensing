from astropy.io import fits
import pandas as pd
from pathlib import Path

ROOT = Path("data/raw/castles")

rows = []

for fits_file in ROOT.rglob("*.fits"):
    header = fits.getheader(fits_file)

    lens_name = fits_file.parts[-4]

    rows.append({
        "image": fits_file.name,
        "lens": lens_name,
        "telescope": header.get("TELESCOP"),
        "instrument": header.get("INSTRUME"),
        "detector": header.get("DETECTOR"),
        "filter": header.get("FILTER"),
        "exposure": header.get("EXPTIME"),
        "pixel_scale": header.get("D001SCAL"),
        "observation_date": header.get("DATE-OBS"),
        "notes": ""
    })

    df = pd.DataFrame(rows)

    output = Path("data/labels/metadata.csv")
    df.to_csv(output, index=False)

    print(df.head())
    print(f"\nSaved {len(df)} rows to {output}")