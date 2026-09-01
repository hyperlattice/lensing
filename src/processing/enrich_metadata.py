from pathlib import Path
import pandas as pd

METADATA_PATH = Path("data/labels/metadata.csv")
CATALOG_PATH = Path("data/labels/lens_catalog.csv")
ENRICHED_OUTPUT = Path("data/labels/metadata_enriched.csv")

def enrich_metadata():
    if not METADATA_PATH.exists() or not CATALOG_PATH.exists():
        print("Error: Required input files (metadata.csv or lens_catalog.csv) missing.")
        return

    print("Loading metadata and catalog...")
    df_meta = pd.read_csv(METADATA_PATH)
    df_catalog = pd.read_csv(CATALOG_PATH)

    # Merge image metadata with scientific labels on the 'lens' column
    # We use a left join to keep every image, even if its lens isn't in the catalog
    df_enriched = pd.merge(df_meta, df_catalog, on="lens", how="left")

    ENRICHED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(ENRICHED_OUTPUT, index=False)

    print(f"Successfully merged {len(df_meta)} images with {len(df_catalog)} lens entries.")
    print(f"Saved enriched metadata to {ENRICHED_OUTPUT}")

    # Basic validation: check how many samples have complete physics labels
    complete_samples = df_enriched.dropna(subset=["lens_redshift", "source_redshift"]).shape[0]
    print(f"Samples with complete physics labels: {complete_samples} / {len(df_enriched)}")

if __name__ == "__main__":
    enrich_metadata()
