from pathlib import Path

import pandas as pd


METADATA_PATH = Path("data/labels/metadata.csv")
CATALOG_PATH = Path("data/labels/lens_catalog.csv")
ENRICHED_OUTPUT = Path("data/labels/metadata_enriched.csv")


def _canonical_lens_id(dataframe, label):
    if "lens_id" in dataframe.columns:
        result = dataframe["lens_id"].copy()
    elif "lens" in dataframe.columns:
        result = dataframe["lens"].copy()
    else:
        raise ValueError(f"{label} must contain either lens_id or lens")
    if result.isna().any():
        raise ValueError(f"{label} contains missing lens identifiers")
    return result.astype(str)


def enrich_metadata(
    metadata_path=METADATA_PATH,
    catalog_path=CATALOG_PATH,
    output=ENRICHED_OUTPUT,
):
    metadata_path = Path(metadata_path)
    catalog_path = Path(catalog_path)
    output = Path(output)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file does not exist: {catalog_path}")

    df_meta = pd.read_csv(metadata_path)
    df_catalog = pd.read_csv(catalog_path)
    df_meta["lens_id"] = _canonical_lens_id(df_meta, "Metadata")
    df_catalog["lens_id"] = _canonical_lens_id(df_catalog, "Catalog")

    if df_catalog["lens_id"].duplicated().any():
        duplicates = df_catalog.loc[df_catalog["lens_id"].duplicated(), "lens_id"].tolist()
        raise ValueError(f"Catalog has duplicate lens IDs: {duplicates[:5]}")

    catalog_columns = [column for column in df_catalog.columns if column != "lens"]
    df_enriched = pd.merge(
        df_meta,
        df_catalog[catalog_columns],
        on="lens_id",
        how="left",
        validate="many_to_one",
        indicator="catalog_join",
    )
    if len(df_enriched) != len(df_meta):
        raise ValueError("Metadata enrichment changed the number of image rows")

    df_enriched["catalog_match"] = df_enriched["catalog_join"].eq("both")
    df_enriched = df_enriched.drop(columns="catalog_join")
    df_enriched["metadata_status"] = "complete"
    missing_labels = df_enriched[["lens_redshift", "source_redshift"]].isna().any(axis=1)
    df_enriched.loc[missing_labels, "metadata_status"] = "missing_redshift"
    df_enriched.loc[~df_enriched["catalog_match"], "metadata_status"] = "missing_catalog_match"

    output.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(output, index=False)
    return df_enriched


def main():
    enriched = enrich_metadata()
    complete_samples = enriched.dropna(subset=["lens_redshift", "source_redshift"]).shape[0]
    print(f"Successfully merged {len(enriched)} image rows.")
    print(f"Saved enriched metadata to {ENRICHED_OUTPUT}")
    print(f"Samples with complete redshift labels: {complete_samples} / {len(enriched)}")


if __name__ == "__main__":
    main()
