from pathlib import Path
import pandas as pd

METADATA_PATH = Path("data/labels/metadata.csv")
CATALOG_OUTPUT = Path("data/labels/lens_catalog.csv")

# Scientific data extracted from the official CASTLES catalog
# Format: { "lens_id": (z_lens, z_source, einstein_radius_arcsec) }
# Note: Einstein radius derived as Size / 2 per catalog documentation.
KNOWN_LENSES = {
    "APM08279": (None, 3.87, 0.38 / 2),
    "B0218": (0.68, 0.96, 0.34 / 2),
    "B0445": (0.557, None, 1.35 / 2),
    "B0631": (0.35, None, 1.16 / 2), # Averaged 0.09/0.62
    "B0712": (0.41, 1.34, 1.46 / 2),
    "B0739": (None, None, 0.53 / 2),
    "B0850": (0.59, 2.53, 0.68 / 2), # Averaged 1.14/3.93
    "B1030": (0.60, 1.54, 1.65 / 2),
    "B1127": (None, None, 0.74 / 2),
    "B1152": (0.439, 1.019, 1.59 / 2),
    "B1359": (None, 3.235, 1.71 / 2),
    "B1422": (0.34, 3.62, 1.68 / 2),
    "B1555": (None, None, 0.42 / 2),
    "B1600": (0.41, 1.59, 1.40 / 2),
    "B1608": (0.63, 1.39, 2.27 / 2),
    "B1933": (0.76, 2.63, 1.00 / 2),
    "B1938": (0.881, 2.059, 1.0 / 2),
    "B2045": (0.87, 1.28, 2.74 / 2),
    "B2108": (None, None, 4.57 / 2),
    "B2114": (0.32, 0.59, 1.31 / 2),
    "B2319": (0.62, None, 1.36 / 2),
    "CFRS03": (0.938, 2.941, 2.1 / 2),
    "CTQ414": (0.317, 1.29, 1.22 / 2),
    "FSC10214": (0.75, 2.29, 1.59 / 2),
    "H1413": (None, 2.55, 1.35 / 2),
    "HE0047": (0.41, 1.66, 1.44 / 2),
    "HE0230": (0.52, 2.162, 2.05 / 2),
    "HE0435": (0.46, 1.689, 2.42 / 2),
    "HE0512": (0.93, 1.57, 0.65 / 2),
    "HE1104": (0.73, 2.32, 3.19 / 2),
    "HE2149": (0.50, 2.03, 1.70 / 2),
    "HS0810": (None, 1.50, 0.96 / 2),
    "HS0818": (0.39, 3.115, 2.83 / 2),
    "HST01247": (None, None, 2.20 / 2),
    "HST12531": (0.69, None, 1.23 / 2),
    "HST14113": (0.46, 2.81, 1.80 / 2),
    "HST14164": (None, None, 2.24 / 2),
    "J1004": (0.95, 2.65, 1.54 / 2),
    "LBQS1333": (None, 1.57, 1.63 / 2),
    "MG0414": (0.96, 2.64, 2.40 / 2),
    "MG1131": (0.844, None, 2.1 / 2),
    "MG1549": (0.11, 1.17, 1.7 / 2),
    "MG2016": (1.01, 3.27, 3.52 / 2),
    "PG1115": (0.31, 1.72, 2.32 / 2),
    "PKS1830": (0.89, 2.51, 0.99 / 2),
    "PMN0134": (0.77, 2.216, 0.73 / 2),
    "PMN1632": (None, 3.424, 1.47 / 2),
    "PMN1838": (None, 2.78, 0.99 / 2),
    "PMN2004": (None, None, 1.18 / 2),
    "PSS2322": (None, 4.12, 1.49 / 2),
    "Q0047": (0.48, 3.60, 2.7 / 2),
    "Q0957": (0.36, 1.41, 6.26 / 2),
    "Q1009": (0.87, 2.74, 1.54 / 2),
    "Q1355": (0.48, 1.37, 1.23 / 2),
    "Q2237": (0.04, 1.69, 1.78 / 2),
    "RXJ0911": (0.77, 2.80, 2.47 / 2),
    "RXJ0921": (0.31, 1.65, 6.97 / 2),
    "RXJ1131": (0.295, 0.658, 3.80 / 2),
    "SBS0909": (0.83, 1.38, 1.17 / 2),
    "SBS1520": (0.72, 1.86, 1.59 / 2),
    "SDSS0246": (None, 1.68, 1.19 / 2),
    "SDSS0903": (0.388, 3.605, 2.99 / 2),
    "SDSS0924": (0.39, 1.524, 1.75 / 2),
    "SDSS1004": (0.68, 1.734, 15.99 / 2),
    "SDSS1138": (0.45, 2.44, 1.34 / 2),
    "SDSS1155": (0.176, 2.89, 1.95 / 2),
    "SDSS1226": (0.52, 1.12, 1.26 / 2),
    "WFI2026": (None, 2.23, 1.34 / 2),
    "WFI2033": (0.66, 1.66, 2.33 / 2),
}

def build_catalog():
    if not METADATA_PATH.exists():
        print(f"Error: {METADATA_PATH} not found. Run build_metadata.py first.")
        return

    df_meta = pd.read_csv(METADATA_PATH)
    unique_lenses = df_meta["lens"].unique()

    catalog_rows = []
    for lens in unique_lenses:
        if lens in KNOWN_LENSES:
            z_l, z_s, theta_e = KNOWN_LENSES[lens]
        else:
            z_l, z_s, theta_e = None, None, None

        catalog_rows.append({
            "lens": lens,
            "lens_redshift": z_l,
            "source_redshift": z_s,
            "einstein_radius_arcsec": theta_e
        })

    df_catalog = pd.DataFrame(catalog_rows)
    CATALOG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_catalog.to_csv(CATALOG_OUTPUT, index=False)

    print(f"Saved {len(df_catalog)} lens entries to {CATALOG_OUTPUT}")
    print("\nPreview of populated lenses:")
    print(df_catalog.dropna().head())

if __name__ == "__main__":
    build_catalog()
