"""Reproducible command-line entry point for real and simulated releases."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from src.data.dataset import LensDataset
from src.layout import DEFAULT_PREVIEW_ROOT, preview_directory
from src.modeling.fit_lens import fit_sie_recovery_grid
from src.modeling.real_comparison import run_real_comparison
from src.processing.build_catalog import build_catalog
from src.processing.build_metadata import build_metadata
from src.processing.create_splits import create_ancestry_splits, create_splits
from src.processing.enrich_metadata import enrich_metadata
from src.processing.preprocess import main as preprocess
from src.processing.preview_dataset import create_previews
from src.processing.validate_metadata import validate_metadata
from src.simulation.assemble import assemble_pilot_release
from src.simulation.generate import (
    generate_observation_pilot,
    generate_sie_extended,
    generate_sis_controls,
    generate_stress_pilot,
)
from src.testing.validate_simulations import validate_run


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")




def _require_fresh_directory(path, overwrite=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Release directory is not empty: {path}. Use a new release ID or explicit --overwrite."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_real_release(
    release_dir,
    *,
    release_id="real-castles-v2",
    raw_root="data/raw/castles",
    catalog_source="data/labels/source/lens_catalog_source.csv",
    preprocessing_config="configs/preprocessing/default.yaml",
    seed=4201,
    overwrite=False,
    make_previews=True,
    preview_root=DEFAULT_PREVIEW_ROOT,
):
    """Build and validate a self-contained versioned real-data release."""

    started = time.perf_counter()
    release_dir = _require_fresh_directory(release_dir, overwrite=overwrite)
    metadata_path = release_dir / "metadata.csv"
    catalog_path = release_dir / "lens_catalog.csv"
    enriched_path = release_dir / "metadata_enriched.csv"
    manifest_path = release_dir / "manifest.csv"
    products = release_dir / "products"

    metadata = build_metadata(raw_root, metadata_path)
    catalog = build_catalog(metadata_path, catalog_source, catalog_path)
    enriched = enrich_metadata(metadata_path, catalog_path, enriched_path)
    preprocessing = preprocess(
        metadata_path=enriched_path,
        output_dir=products / "normalized",
        manifest_path=manifest_path,
        config_path=preprocessing_config,
        run_manifest_path=release_dir / "preprocessing_run.json",
        overwrite=overwrite,
    )
    processed = pd.read_csv(manifest_path)
    processed["dataset_version"] = release_id
    processed.to_csv(manifest_path, index=False)
    split_manifest = create_splits(
        manifest_path,
        output_dir=release_dir / "splits",
        seed=int(seed),
    )
    row_report, metadata_summary = validate_metadata(
        manifest_path,
        catalog_path,
        release_dir / "validation",
        root=Path("."),
    )
    dataset = LensDataset(manifest_path, root=Path("."))
    strict_errors = dataset.validate_all(require_products=True)
    preview_errors = 0
    preview_index_path = None
    if make_previews:
        preview_output = preview_directory("actual", release_id, root=preview_root)
        preview_index = create_previews(
            root=raw_root,
            output=preview_output,
            manifest_path=metadata_path,
            dpi=120,
            contact_sheet=True,
        )
        preview_errors = int((preview_index["status"] != "ok").sum())
        preview_index_path = (preview_output / "preview_index.csv").as_posix()

    release = {
        "release_id": release_id,
        "release_type": "real_castles_preprocessing_v2",
        "manifest_path": manifest_path.as_posix(),
        "observation_count": int(len(processed)),
        "lens_count": int(processed["lens_id"].nunique()),
        "metadata_rows": int(len(metadata)),
        "catalog_rows": int(len(catalog)),
        "enriched_rows": int(len(enriched)),
        "preprocessing": preprocessing,
        "split_counts": split_manifest.groupby("split").size().astype(int).to_dict(),
        "metadata_validation": metadata_summary,
        "strict_product_error_count": len(strict_errors),
        "strict_product_errors": strict_errors,
        "preview_index_path": preview_index_path,
        "preview_error_count": preview_errors,
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "The local CASTLES products do not provide empirical PSF, ERR, or DQ extensions.",
            "The variance fallback is toy_unprofiled and is not a calibrated detector likelihood.",
            "WFPC2 units and all real-system model suitability require purpose-specific review.",
        ],
    }
    release["passed"] = bool(
        preprocessing["failed_count"] == 0
        and metadata_summary["fatal_rows"] == 0
        and not strict_errors
        and preview_errors == 0
    )
    _write_json(release_dir / "release.json", release)
    if not release["passed"]:
        raise RuntimeError(f"Real release failed validation; see {release_dir / 'release.json'}")
    return release


def validate_release(manifest_path, *, root=None, output=None):
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    domain = str(frame["domain"].dropna().iloc[0]) if "domain" in frame and not frame.empty else "unknown"
    loader = LensDataset(manifest_path, root=root, strict_products=True)
    report = {
        "manifest_path": manifest_path.as_posix(),
        "domain": domain,
        "sample_count": len(loader),
        "strict_loader_passed": True,
    }
    if domain == "simulated":
        report["simulation_validation"] = validate_run(manifest_path)
        report["passed"] = bool(report["simulation_validation"]["passed"])
    else:
        report["passed"] = True
    if output is not None:
        _write_json(output, report)
    return report


def make_splits(manifest, output, *, seed=4201, ancestry=False):
    if ancestry:
        _, report = create_ancestry_splits(manifest, output, seed=seed)
        return report
    result = create_splits(manifest, output_dir=output, seed=seed)
    return {
        "manifest_path": Path(manifest).as_posix(),
        "sample_count": int(len(result)),
        "split_counts": result.groupby("split").size().astype(int).to_dict(),
    }


def build_engineering_release(
    output_dir,
    *,
    audit_dir="data/manifests/sim-eng1000-v1",
    release_id="sim-eng1000-v1",
    config_path="configs/simulation/engineering.yaml",
    count=1000,
    seed=11001,
    split_seed=4201,
    overwrite=False,
):
    """Generate, validate, and split an engineering release."""

    started = time.perf_counter()
    output_dir = Path(output_dir)
    audit_dir = Path(audit_dir)
    if audit_dir.exists() and any(audit_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Audit directory is not empty: {audit_dir}. Use a new release ID or explicit --overwrite."
        )
    run = generate_observation_pilot(
        output_dir,
        count=int(count),
        seed=int(seed),
        simulation_id=release_id,
        config_path=config_path,
        overwrite=overwrite,
        create_preview=False,
        stage_name="engineering_release",
    )
    manifest_path = output_dir / "manifest.csv"
    split_frame, split_report = create_ancestry_splits(
        manifest_path,
        output_dir / "splits",
        seed=int(split_seed),
    )
    validation = validate_run(manifest_path)
    loader = LensDataset(
        manifest_path,
        root=output_dir,
        strict_products=True,
        domain="simulated",
        eligible_for="ml",
        target_fields={"theta_E_arcsec"},
    )
    manifest = pd.read_csv(manifest_path)
    seed_uniqueness = {
        "parameter_seed_unique": bool(manifest["parameter_seed"].is_unique),
        "noise_seed_unique": bool(manifest["noise_seed"].is_unique),
    }
    gates = {
        "requested_count_accepted": run["count_accepted"] == int(count),
        "no_rejections": run["count_rejected"] == 0,
        "unique_sample_ids": bool(manifest["sample_id"].is_unique),
        "unique_parent_ids": bool(manifest["parent_system_id"].is_unique),
        **seed_uniqueness,
        "simulation_validation": bool(validation["passed"]),
        "ancestry_split_validation": not split_report["leakage_detected"],
        "strict_loader_validation": len(loader) == int(count),
    }
    release_report = {
        "release_id": release_id,
        "release_type": "simulation_engineering",
        "config_path": Path(config_path).as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "requested_parent_count": int(count),
        "accepted_parent_count": int(run["count_accepted"]),
        "observation_count": int(len(manifest)),
        "split_counts": {
            name: int((split_frame["split"] == name).sum())
            for name in ("train", "validation", "test")
        },
        "primary_target": "theta_E_arcsec",
        "input_product": "processed_path",
        "gates": gates,
        "passed": bool(all(gates.values())),
        "validation": validation,
        "split_report": split_report,
        "limitations": [
            "The release uses analytic sources and Gaussian PSFs, not empirical HST morphologies or PSFs.",
            "Noise is independent Poisson plus read noise and omits correlated detector/resampling effects.",
            "Uniform engineering priors are not a calibrated lens-population selection function.",
            "P100 is retained separately as a stress suite and is not part of these splits.",
        ],
    }
    _write_json(output_dir / "engineering_release.json", release_report)

    audit_dir.mkdir(parents=True, exist_ok=True)
    compact_files = {
        manifest_path: audit_dir / "manifest.csv",
        output_dir / "run_manifest.json": audit_dir / "run_manifest.json",
        output_dir / "rejection_log.csv": audit_dir / "rejection_log.csv",
        output_dir / "engineering_release.json": audit_dir / "engineering_release.json",
    }
    for split_name in ("train", "validation", "test", "assignments"):
        compact_files[output_dir / "splits" / f"{split_name}.csv"] = (
            audit_dir / "splits" / f"{split_name}.csv"
        )
    compact_files[output_dir / "splits" / "split_report.json"] = (
        audit_dir / "splits" / "split_report.json"
    )
    for source, destination in compact_files.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if not release_report["passed"]:
        raise RuntimeError(f"Engineering release failed validation; see {output_dir / 'engineering_release.json'}")
    return release_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    real = commands.add_parser("prepare-real", help="build a versioned CASTLES release")
    real.add_argument("--release-id", default="real-castles-v2")
    real.add_argument("--output", default="data/releases/real-castles-v2")
    real.add_argument("--raw-root", default="data/raw/castles")
    real.add_argument("--catalog-source", default="data/labels/source/lens_catalog_source.csv")
    real.add_argument("--config", default="configs/preprocessing/default.yaml")
    real.add_argument("--seed", type=int, default=4201)
    real.add_argument("--overwrite", action="store_true")
    real.add_argument("--no-preview", action="store_true")
    real.add_argument("--preview-root", default=str(DEFAULT_PREVIEW_ROOT))

    simulate = commands.add_parser("simulate", help="generate one simulation stage")
    simulate.add_argument("--stage", choices=("sis", "sie", "observation", "stress"), required=True)
    simulate.add_argument("--output", required=True)
    simulate.add_argument("--simulation-id", required=True)
    simulate.add_argument("--count", type=int, required=True)
    simulate.add_argument("--seed", type=int, required=True)
    simulate.add_argument("--config", default="configs/simulation/pilot.yaml")
    simulate.add_argument("--overwrite", action="store_true")
    simulate.add_argument("--no-preview", action="store_true")
    simulate.add_argument("--preview-root", default=str(DEFAULT_PREVIEW_ROOT))

    assemble = commands.add_parser("assemble", help="assemble simulation stages")
    assemble.add_argument("--stage-root", action="append", required=True)
    assemble.add_argument("--output", required=True)
    assemble.add_argument("--release-id", required=True)
    assemble.add_argument("--overwrite", action="store_true")
    assemble.add_argument("--no-preview", action="store_true")
    assemble.add_argument("--preview-root", default=str(DEFAULT_PREVIEW_ROOT))

    validate = commands.add_parser("validate", help="validate a complete release")
    validate.add_argument("manifest")
    validate.add_argument("--root", default=None)
    validate.add_argument("--output", default=None)

    split = commands.add_parser("make-splits", help="write leakage-safe split manifests")
    split.add_argument("manifest")
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=4201)
    split.add_argument("--ancestry", action="store_true")

    fit_grid = commands.add_parser("fit-grid", help="run the pre-ML SIE recovery grid")
    fit_grid.add_argument("--p30-manifest", default="data/simulations/sim-p30-v1/manifest.csv")
    fit_grid.add_argument("--p60-manifest", default="data/simulations/sim-p60-v1/manifest.csv")
    fit_grid.add_argument("--p100-manifest", default="data/simulations/sim-p100-v1/manifest.csv")
    fit_grid.add_argument("--output", default="results/reports/fit-sie-grid-v1")
    fit_grid.add_argument("--count-per-condition", type=int, default=2)
    fit_grid.add_argument("--max-iterations", type=int, default=40)

    real_comparison = commands.add_parser(
        "real-comparison", help="run the frozen exploratory real-lens comparison"
    )
    real_comparison.add_argument("--config", default="configs/observations/comparison.yaml")
    real_comparison.add_argument("--output", default="results/reports/real-comparison-v1")

    engineering = commands.add_parser(
        "engineering-release", help="generate and freeze the pre-ML engineering release"
    )
    engineering.add_argument("--output", default="data/simulations/sim-eng1000-v1")
    engineering.add_argument("--audit-dir", default="data/manifests/sim-eng1000-v1")
    engineering.add_argument("--release-id", default="sim-eng1000-v1")
    engineering.add_argument("--config", default="configs/simulation/engineering.yaml")
    engineering.add_argument("--count", type=int, default=1000)
    engineering.add_argument("--seed", type=int, default=11001)
    engineering.add_argument("--split-seed", type=int, default=4201)
    engineering.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare-real":
        result = prepare_real_release(
            args.output,
            release_id=args.release_id,
            raw_root=args.raw_root,
            catalog_source=args.catalog_source,
            preprocessing_config=args.config,
            seed=args.seed,
            overwrite=args.overwrite,
            make_previews=not args.no_preview,
            preview_root=args.preview_root,
        )
    elif args.command == "simulate":
        generators = {
            "sis": generate_sis_controls,
            "sie": generate_sie_extended,
            "observation": generate_observation_pilot,
            "stress": generate_stress_pilot,
        }
        result = generators[args.stage](
            args.output,
            count=args.count,
            seed=args.seed,
            simulation_id=args.simulation_id,
            config_path=args.config,
            overwrite=args.overwrite,
            create_preview=not args.no_preview,
            preview_root=args.preview_root,
        )
    elif args.command == "assemble":
        result = assemble_pilot_release(
            args.stage_root,
            args.output,
            release_id=args.release_id,
            overwrite=args.overwrite,
            create_preview=not args.no_preview,
            preview_root=args.preview_root,
        )
    elif args.command == "validate":
        result = validate_release(args.manifest, root=args.root, output=args.output)
    elif args.command == "make-splits":
        result = make_splits(args.manifest, args.output, seed=args.seed, ancestry=args.ancestry)
    elif args.command == "fit-grid":
        result = fit_sie_recovery_grid(
            args.p30_manifest,
            args.p60_manifest,
            args.p100_manifest,
            args.output,
            count_per_condition=args.count_per_condition,
            max_iterations=args.max_iterations,
        )
    elif args.command == "real-comparison":
        result = run_real_comparison(args.config, args.output)
    else:
        result = build_engineering_release(
            args.output,
            audit_dir=args.audit_dir,
            release_id=args.release_id,
            config_path=args.config,
            count=args.count,
            seed=args.seed,
            split_seed=args.split_seed,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
