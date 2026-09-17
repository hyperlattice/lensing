from pathlib import Path

import pandas as pd


METADATA_PATH = Path("data/labels/derived/metadata.csv")
SOURCE_PATH = Path("data/labels/source/lens_catalog_source.csv")
CATALOG_OUTPUT = Path("data/labels/derived/lens_catalog.csv")
SOURCE_COLUMNS = {
    "lens_id",
    "lens_redshift",
    "source_redshift",
    "lens_redshift_alternatives",
    "source_redshift_alternatives",
    "castles_size_arcsec",
    "einstein_radius_estimate_arcsec",
    "einstein_radius_method",
    "redshift_status",
    "source_url",
    "source_reference",
    "accessed_date",
    "selection_rationale",
}


def _canonical_lens_id(frame):
    if "lens_id" in frame.columns:
        return frame["lens_id"]
    if "lens" in frame.columns:
        return frame["lens"]
    raise ValueError("Input must contain lens_id or lens")


def _validate_source(source):
    missing = SOURCE_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError(f"Catalog source is missing columns: {sorted(missing)}")
    if source["lens_id"].isna().any() or (source["lens_id"].astype(str).str.strip() == "").any():
        raise ValueError("Catalog source contains empty lens IDs")
    if source["lens_id"].duplicated().any():
        duplicates = sorted(source.loc[source["lens_id"].duplicated(), "lens_id"].unique())
        raise ValueError(f"Catalog source contains duplicate lens IDs: {duplicates}")
    if (source["einstein_radius_estimate_arcsec"].dropna() <= 0).any():
        raise ValueError("Catalog source contains non-positive radius estimates")

    both_redshifts = source["lens_redshift"].notna() & source["source_redshift"].notna()
    invalid = both_redshifts & ~(
        (source["lens_redshift"] > 0) & (source["lens_redshift"] < source["source_redshift"])
    )
    if invalid.any():
        ids = sorted(source.loc[invalid, "lens_id"].tolist())
        raise ValueError(f"Catalog source contains invalid redshift ordering: {ids}")


def build_catalog(
    metadata_path=METADATA_PATH,
    source_path=SOURCE_PATH,
    output=CATALOG_OUTPUT,
):
    """Build a generated catalog from the immutable curated source table."""

    metadata_path = Path(metadata_path)
    source_path = Path(source_path)
    output = Path(output)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata manifest not found: {metadata_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Curated catalog source not found: {source_path}")

    metadata = pd.read_csv(metadata_path)
    metadata_lens_ids = pd.DataFrame({"lens_id": _canonical_lens_id(metadata)})
    if metadata_lens_ids["lens_id"].isna().any():
        raise ValueError("Metadata contains empty lens IDs")
    metadata_lens_ids = metadata_lens_ids.drop_duplicates().reset_index(drop=True)

    source = pd.read_csv(source_path)
    _validate_source(source)

    catalog = metadata_lens_ids.merge(
        source,
        on="lens_id",
        how="left",
        validate="one_to_one",
        indicator="source_match",
    )
    catalog["source_match"] = catalog["source_match"].eq("both")
    if len(catalog) != len(metadata_lens_ids):
        raise ValueError("Catalog build changed the number of metadata lens systems")

    output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output, index=False)
    return catalog


def main():
    catalog = build_catalog()
    print(f"Saved {len(catalog)} lens entries to {CATALOG_OUTPUT}")
    print(catalog["source_match"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
