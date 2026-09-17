"""Assemble validated simulation increments into one cumulative release."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.layout import DEFAULT_PREVIEW_ROOT, preview_directory
from src.processing.preview_dataset import create_previews


SKIP_FILES = {"manifest.csv", "rejection_log.csv", "run_manifest.json", "validation.json"}
SKIP_DIRECTORIES = {"previews"}


def _copy_stage_artifacts(stage_root: Path, output_root: Path) -> int:
    copied = 0
    for source in stage_root.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(stage_root)
        if relative.name in SKIP_FILES or relative.parts[0] in SKIP_DIRECTORIES:
            continue
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"Conflicting artifact path during assembly: {relative}")
            continue
        shutil.copy2(source, target)
        copied += 1
    return copied


def assemble_pilot_release(
    stage_roots: Iterable[str | Path],
    output_dir: str | Path,
    *,
    release_id: str = "sim-p100-cumulative-v1",
    overwrite: bool = False,
    create_preview: bool = True,
    preview_root: str | Path = DEFAULT_PREVIEW_ROOT,
) -> dict[str, Any]:
    """Copy stage artifacts and combine their manifests without renaming parents."""

    roots = [Path(root) for root in stage_roots]
    if not roots:
        raise ValueError("At least one stage root is required")
    if any(not root.is_dir() for root in roots):
        missing = [str(root) for root in roots if not root.is_dir()]
        raise FileNotFoundError(f"Missing stage roots: {missing}")
    output_root = Path(output_dir)
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty and overwrite is disabled: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    frames = []
    rejection_records = []
    copied_artifacts = 0
    for stage_root in roots:
        manifest_path = stage_root / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Stage manifest is missing: {manifest_path}")
        frame = pd.read_csv(manifest_path)
        if frame["sample_id"].duplicated().any() or frame["parent_system_id"].duplicated().any():
            raise ValueError(f"Stage contains duplicate samples or parents: {stage_root}")
        frame["dataset_version"] = release_id
        frame["release_stage"] = stage_root.name
        frames.append(frame)
        rejection_path = stage_root / "rejection_log.csv"
        if rejection_path.exists():
            rejection_frame = pd.read_csv(rejection_path)
            for record in rejection_frame.to_dict(orient="records"):
                record["stage_root"] = stage_root.as_posix()
                rejection_records.append(record)
        copied_artifacts += _copy_stage_artifacts(stage_root, output_root)

    manifest = pd.concat(frames, ignore_index=True, sort=False)
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Cumulative release contains duplicate sample IDs")
    if manifest["parent_system_id"].duplicated().any():
        raise ValueError("Cumulative release contains duplicate parent IDs")
    manifest_path = output_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    rejection_path = output_root / "rejection_log.csv"
    rejection_fields = ["sample_id", "parent_system_id", "stage", "error_type", "message", "stage_root"]
    with rejection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rejection_fields)
        writer.writeheader()
        writer.writerows(rejection_records)

    preview_index_path = None
    if create_preview:
        preview_output = preview_directory("simulated", release_id, root=preview_root)
        preview_index = create_previews(
            root=output_root / "raw",
            output=preview_output,
            manifest_path=None,
            dpi=100,
            contact_sheet=True,
        )
        preview_index_path = (preview_output / "preview_index.csv").as_posix()
        preview_errors = int((preview_index["status"] != "ok").sum())
    else:
        preview_errors = 0

    run_manifest = {
        "run_id": release_id,
        "release_id": release_id,
        "release_type": "cumulative_simulation_pilot",
        "stage_roots": [root.as_posix() for root in roots],
        "count_accepted": int(len(manifest)),
        "parent_count": int(manifest["parent_system_id"].nunique()),
        "count_rejected_in_stages": len(rejection_records),
        "copied_artifact_count": copied_artifacts,
        "preview_index_path": preview_index_path,
        "preview_error_count": preview_errors,
        "manifest_path": "manifest.csv",
        "rejection_log_path": "rejection_log.csv",
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--release-id", default="sim-p100-cumulative-v1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--preview-root", default=str(DEFAULT_PREVIEW_ROOT))
    args = parser.parse_args()
    summary = assemble_pilot_release(
        args.stage_root,
        args.output,
        release_id=args.release_id,
        overwrite=args.overwrite,
        create_preview=not args.no_preview,
        preview_root=args.preview_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["count_rejected_in_stages"] == 0 and summary["preview_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
