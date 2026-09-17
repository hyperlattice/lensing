from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits

from src.layout import preview_directory


ROOT = Path("data/raw/castles")
OUTPUT = preview_directory("actual", "castles-development-v1")
MANIFEST_PATH = Path("data/labels/derived/metadata.csv")


def _input_rows(root, manifest_path=None):
    if manifest_path is not None:
        manifest = pd.read_csv(manifest_path)
        required = {"sample_id", "raw_path"}
        missing = required.difference(manifest.columns)
        if missing:
            raise ValueError(f"Preview manifest is missing columns: {sorted(missing)}")
        return [(str(row.sample_id), Path(row.raw_path)) for row in manifest.itertuples(index=False)]

    root = Path(root)
    rows = []
    for fits_file in sorted(root.rglob("*.fits")):
        sample_id = fits_file.relative_to(root).with_suffix("").as_posix().replace("/", "_")
        rows.append((sample_id, fits_file))
    return rows


def _create_contact_sheet(rows, output, dpi=150, columns=10):
    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        return None
    sheet_rows = (len(successful) + columns - 1) // columns
    figure, axes = plt.subplots(
        sheet_rows,
        columns,
        figsize=(columns * 2.0, sheet_rows * 2.0),
        squeeze=False,
    )
    axes = axes.ravel()
    for axis, record in zip(axes, successful):
        try:
            image = fits.getdata(record["raw_path"])
            axis.imshow(image, origin="lower", cmap="gray")
            axis.set_title(record["sample_id"], fontsize=6)
        except Exception:
            axis.set_title(f"{record['sample_id']} (error)", fontsize=6)
        axis.set_axis_off()
    for axis in axes[len(successful):]:
        axis.set_axis_off()
    contact_sheet_path = output / "contact_sheet.png"
    figure.tight_layout(pad=0.2)
    figure.savefig(contact_sheet_path, dpi=dpi)
    plt.close(figure)
    return contact_sheet_path


def create_previews(root=ROOT, output=OUTPUT, manifest_path=MANIFEST_PATH, dpi=200, contact_sheet=True):
    """Create one preview per observation, an index, and an optional contact sheet."""
    output = Path(output)
    manifest_path = Path(manifest_path) if manifest_path is not None else None
    rows = _input_rows(root, manifest_path if manifest_path and manifest_path.exists() else None)
    sample_ids = [sample_id for sample_id, _ in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Preview inputs contain duplicate sample IDs")

    output.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for sample_id, fits_file in rows:
        preview_path = output / f"{sample_id}.png"
        record = {
            "sample_id": sample_id,
            "raw_path": Path(fits_file).as_posix(),
            "preview_path": preview_path.as_posix(),
            "status": "ok",
            "error": "",
        }
        try:
            image = fits.getdata(fits_file)
            figure, axis = plt.subplots(figsize=(5, 5))
            image_artist = axis.imshow(image, origin="lower", cmap="gray")
            figure.colorbar(image_artist, ax=axis, label="Pixel Value")
            axis.set_title(f"{sample_id} - {Path(fits_file).name}")
            figure.tight_layout()
            figure.savefig(preview_path, dpi=dpi)
            plt.close(figure)
        except Exception as error:
            record["status"] = "error"
            record["error"] = str(error)
        index_rows.append(record)

    contact_sheet_path = _create_contact_sheet(index_rows, output, dpi=max(100, dpi // 2)) if contact_sheet else None
    for record in index_rows:
        record["contact_sheet_path"] = "" if contact_sheet_path is None else contact_sheet_path.as_posix()
    index = pd.DataFrame(index_rows)
    index.to_csv(output / "preview_index.csv", index=False)
    return index


def main():
    index = create_previews()
    errors = index.loc[index["status"] != "ok"]
    print(f"Created {len(index) - len(errors)} previews; errors: {len(errors)}")
    if not errors.empty:
        print(errors[["sample_id", "error"]].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
