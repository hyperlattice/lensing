"""Validate image metadata, catalog joins, and experiment eligibility."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METADATA_PATH = Path("data/labels/derived/metadata.csv")
CATALOG_PATH = Path("data/labels/derived/lens_catalog.csv")
REPORT_DIR = Path("results/reports/metadata")
REQUIRED_COLUMNS = {"sample_id", "raw_path", "pixel_scale"}
FATAL_DEFAULT = {
    "duplicate_sample_id",
    "missing_raw_path",
    "missing_processed_path",
    "invalid_scale",
    "invalid_redshift_order",
    "duplicate_catalog_key",
}
MISSINGNESS_FIELDS = (
    "telescope",
    "instrument",
    "detector",
    "filter_name",
    "filter",
    "image_unit",
    "bunit",
    "pixel_scale",
    "wcs_status",
)


def _canonical_lens_id(frame):
    if "lens_id" in frame.columns:
        return frame["lens_id"]
    if "lens" in frame.columns:
        return frame["lens"]
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"unknown", "none", "nan"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _path_value(value, root):
    if _is_missing(value):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else Path(root) / path


def _append_issue(issues, issue_class, message):
    issues.append((issue_class, message))


def _catalog_index(catalog):
    if catalog is None:
        return {}, set()
    catalog = catalog.copy()
    catalog["lens_id"] = _canonical_lens_id(catalog)
    duplicate_ids = set(catalog.loc[catalog["lens_id"].duplicated(keep=False), "lens_id"].dropna())
    if duplicate_ids:
        return {}, duplicate_ids
    return catalog.set_index("lens_id", drop=False).to_dict(orient="index"), set()


def _check_array(path):
    try:
        array = np.load(path)
    except Exception as exc:
        return f"cannot_load_processed:{exc}"
    if array.shape != (256, 256):
        return f"processed_shape:{array.shape}"
    if array.dtype != np.float32:
        return f"processed_dtype:{array.dtype}"
    if not np.isfinite(array).all():
        return "processed_nonfinite"
    return None


def validate_metadata(
    metadata_path=METADATA_PATH,
    catalog_path=CATALOG_PATH,
    output_dir=REPORT_DIR,
    root=Path("."),
    fatal_classes=None,
):
    """Validate metadata and write row-level and summary reports.

    The function returns ``(row_report, summary)``. Only issue classes listed
    in ``fatal_classes`` affect the command exit status; reviewable metadata
    gaps remain visible in the report without blocking an angular pilot.
    """
    metadata_path = Path(metadata_path)
    catalog_path = Path(catalog_path) if catalog_path is not None else None
    output_dir = Path(output_dir)
    root = Path(root)
    fatal_classes = set(FATAL_DEFAULT if fatal_classes is None else fatal_classes)

    metadata = pd.read_csv(metadata_path)
    catalog = pd.read_csv(catalog_path) if catalog_path is not None and catalog_path.exists() else None
    missing_columns = REQUIRED_COLUMNS.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {sorted(missing_columns)}")

    catalog_rows, duplicate_catalog_ids = _catalog_index(catalog)
    metadata_lens_ids = _canonical_lens_id(metadata)
    duplicate_samples = set(metadata.loc[metadata["sample_id"].duplicated(keep=False), "sample_id"])
    missingness = {
        field: int(field not in metadata.columns or metadata[field].map(_is_missing).sum())
        for field in MISSINGNESS_FIELDS
    }
    report_rows = []

    for row in metadata.to_dict(orient="records"):
        sample_id = row.get("sample_id")
        lens_id = row.get("lens_id", row.get("lens"))
        issues = []
        if sample_id in duplicate_samples:
            _append_issue(issues, "duplicate_sample_id", "sample_id appears more than once")
        if _is_missing(lens_id):
            _append_issue(issues, "missing_lens_id", "lens_id/lens is missing")

        raw_path = _path_value(row.get("raw_path"), root)
        processed_path = _path_value(row.get("processed_path"), root)
        if raw_path is None or not raw_path.exists():
            _append_issue(issues, "missing_raw_path", f"raw file is unavailable: {row.get('raw_path')}")
        if processed_path is None or not processed_path.exists():
            _append_issue(issues, "missing_processed_path", f"processed file is unavailable: {row.get('processed_path')}")
        elif (array_issue := _check_array(processed_path)) is not None:
            issue_class, _, detail = array_issue.partition(":")
            _append_issue(issues, issue_class, detail or array_issue)

        try:
            pixel_scale = float(row.get("pixel_scale"))
            if not np.isfinite(pixel_scale) or pixel_scale <= 0:
                _append_issue(issues, "invalid_scale", "pixel_scale must be finite and positive")
        except (TypeError, ValueError):
            _append_issue(issues, "invalid_scale", "pixel_scale is not numeric")

        lens_z = row.get("lens_redshift")
        source_z = row.get("source_redshift")
        if not _is_missing(lens_z) and not _is_missing(source_z):
            try:
                if not 0 < float(lens_z) < float(source_z):
                    _append_issue(issues, "invalid_redshift_order", "expected 0 < lens_redshift < source_redshift")
            except (TypeError, ValueError):
                _append_issue(issues, "invalid_redshift_order", "redshifts must be numeric")

        catalog_row = catalog_rows.get(lens_id)
        if catalog is not None and lens_id not in catalog_rows:
            if lens_id in duplicate_catalog_ids:
                _append_issue(issues, "duplicate_catalog_key", "catalog contains duplicate lens IDs")
            else:
                _append_issue(issues, "missing_catalog_match", "lens ID has no curated catalog row")
        if catalog_row is not None:
            if str(catalog_row.get("redshift_status", "")).lower() == "ambiguous":
                _append_issue(issues, "ambiguous_redshift", "catalog preserves alternative redshifts")
            if str(catalog_row.get("model_suitability", "")).lower() in {"not_single_sie", "exclude"}:
                _append_issue(issues, "model_unsuitable", "excluded from the single-SIE experiment")

        unit = row.get("image_unit", row.get("bunit"))
        if _is_missing(unit) or str(unit).strip().upper() in {"0", "UNKNOWN"}:
            _append_issue(issues, "unknown_unit", "image unit is unavailable or unverified")

        classes = sorted({issue_class for issue_class, _ in issues})
        fatal = sorted(set(classes).intersection(fatal_classes))
        structural_ok = not set(classes).intersection(
            {"duplicate_sample_id", "missing_raw_path", "missing_processed_path", "processed_shape", "processed_dtype", "processed_nonfinite", "cannot_load_processed", "invalid_scale"}
        )
        eligible = ["preprocessing"] if structural_ok else []
        if catalog_row is not None and not set(classes).intersection({"ambiguous_redshift", "model_unsuitable", "missing_catalog_match"}):
            eligible.append("real_comparison")
        report_rows.append(
            {
                "sample_id": sample_id,
                "lens_id": lens_id,
                "instrument": row.get("instrument"),
                "status": "fail" if fatal else "review" if classes else "pass",
                "issue_classes": ";".join(classes),
                "issues": ";".join(f"{issue_class}: {message}" for issue_class, message in issues),
                "fatal_classes": ";".join(fatal),
                "eligible_for": ";".join(eligible),
            }
        )

    report_columns = [
        "sample_id",
        "lens_id",
        "instrument",
        "status",
        "issue_classes",
        "issues",
        "fatal_classes",
        "eligible_for",
    ]
    row_report = pd.DataFrame(report_rows, columns=report_columns)
    summary = {
        "metadata_path": metadata_path.as_posix(),
        "catalog_path": None if catalog_path is None else catalog_path.as_posix(),
        "rows": int(len(metadata)),
        "lens_systems": int(metadata_lens_ids.nunique(dropna=True)),
        "status_counts": row_report["status"].value_counts(dropna=False).to_dict(),
        "issue_counts": {
            issue_class: int(row_report["issue_classes"].str.split(";").explode().eq(issue_class).sum())
            for issue_class in sorted(set(row_report["issue_classes"].str.split(";").explode()) - {""})
        },
        "eligibility_counts": {
            eligibility: int(row_report["eligible_for"].str.split(";").explode().eq(eligibility).sum())
            for eligibility in sorted(set(row_report["eligible_for"].str.split(";").explode()) - {""})
        },
        "missingness_counts": missingness,
        "fatal_classes": sorted(fatal_classes),
        "fatal_rows": int(row_report["fatal_classes"].ne("").sum()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    row_report.to_csv(output_dir / "metadata_validation.csv", index=False)
    (output_dir / "metadata_validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row_report, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=str(METADATA_PATH))
    parser.add_argument("--catalog", default=str(CATALOG_PATH))
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    parser.add_argument("--root", default=".")
    parser.add_argument("--fatal", action="append", dest="fatal_classes")
    args = parser.parse_args()
    _, summary = validate_metadata(
        args.metadata,
        args.catalog,
        args.output_dir,
        args.root,
        args.fatal_classes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["fatal_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
