"""Materialize an image-only engineering overlay of existing development data.

This command never upgrades data eligibility, opens target values, runs a CNN,
or modifies the source release. Masks are explicitly empty because these are
finite native-grid toy-detector images without cropping/padding. The image is
the unnormalized raw rate, with background retained, so analytic detector
variance has matching units and geometry. This differs from the historical
background-subtracted processed image and must not be substituted into its
reported experiments.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np
from astropy.io import fits

from src.data.benchmark_contract_v1 import BenchmarkImages, contained_path

SOURCE = Path("data/simulations/cross-simulator-development-v2")
OUTPUT = Path("data/simulations/benchmark-contract-development-v1")
REPORT = Path("results/reports/benchmark-foundations-v1/engineering_release.json")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(source=SOURCE, output=OUTPUT, report_path=REPORT):
    source, output, report_path = map(Path, (source, output, report_path))
    if output.exists() or report_path.exists():
        raise FileExistsError("Use a new release/report identity; existing evidence is immutable")
    completion = json.loads((source / "generation_complete.json").read_text())
    if completion["stage"] != "development" or completion["cnn_inference_run"] is not False:
        raise ValueError("This importer accepts only the existing engineering development release")
    # Read only allowed fields; never parse the theta_E column into a target.
    with (source / "development_manifest.csv").open(newline="", encoding="utf-8") as stream:
        records = [{k: row[k] for k in ("sample_id", "parent_system_id", "condition", "raw_path",
                    "parameter_seed", "noise_seed", "pixel_scale", "shape", "simulator_name",
                    "accuracy_evaluation_eligible")} for row in csv.DictReader(stream)]
    if not records or any(r["accuracy_evaluation_eligible"].lower() != "false" for r in records):
        raise ValueError("Unexpected eligibility in source engineering release")
    output.mkdir(parents=True)
    write_json(output / "generation_started.json", {"source": source.as_posix()})
    rows = []
    for r in records:
        folder = output / r["sample_id"]
        if folder.parent.resolve() != output.resolve():
            raise ValueError("Invalid sample identity")
        folder.mkdir()
        source_folder = Path(r["raw_path"]).parent
        paths = {"image": Path(r["raw_path"]), "variance": source_folder / "expected_variance.npy",
                 "psf": source_folder / "psf.npy"}
        for product, rel in paths.items():
            original = contained_path(source, rel.as_posix())
            if product == "image":
                with fits.open(original) as hdul:
                    if hdul[0].header.get("BUNIT") != "ELECTRONS/S":
                        raise ValueError("Unverified raw image units")
                    np.save(folder / "image.npy", np.asarray(hdul[0].data, dtype=np.float32))
            else:
                shutil.copyfile(original, folder / f"{product}.npy")
        shape = [int(n) for n in r["shape"].split("x")]
        for mask in ("invalid_mask", "coverage_mask"):
            np.save(folder / f"{mask}.npy", np.zeros(shape, dtype=bool))
        products = {k: {"path": (folder / f"{k}.npy").relative_to(output).as_posix()}
                    for k in ("image", "variance", "psf", "invalid_mask", "coverage_mask")}
        rows.append({"sample_id": r["sample_id"], "parent_system_id": r["parent_system_id"],
                     "role": "numerical_development", "renderer": r["simulator_name"],
                     "condition": r["condition"], "parameter_seed": int(r["parameter_seed"]),
                     "noise_seed": int(r["noise_seed"]), "ancestry_ids": [], "shape": shape,
                     "pixel_scale_arcsec": float(r["pixel_scale"]), "products": products,
                     "psf_family": "moffat" if r["condition"] == "moffat-psf" else "gaussian"})
    manifest = {"schema": "cross_simulator_benchmark_v1", "release_id": output.name,
                "stage": "engineering", "image_unit": "electrons_per_second",
                "variance_unit": "electrons_per_second_squared", "mask_semantics": "true_is_excluded",
                "qualification": None, "observations": rows}
    manifest_path = output / "benchmark.json"
    write_json(manifest_path, manifest)
    images = BenchmarkImages(manifest_path, role="numerical_development", allow_engineering=True)
    shapes = sorted({tuple(images[i].shape) for i in range(len(images))})
    try:
        BenchmarkImages(manifest_path, role="source_train", allow_engineering=True)
    except ValueError:
        training_blocked = True
    else:
        raise AssertionError("Engineering data became accessible as source training")
    report = {"release_id": output.name, "stage": "engineering", "observations": len(images),
              "parents": len({r["parent_system_id"] for r in rows}), "native_shapes": shapes,
              "contract_validation_passed": True, "training_role_blocked": training_blocked,
              "accuracy_qualified": False, "cnn_inference_run": False,
              "final_holdout_accessed": False, "target_sidecars_opened": 0,
              "image_semantics": "raw rate; background retained; no normalization/crop/pad",
              "variance_semantics": "analytic toy-detector variance, not a CNN input",
              "noise_coupling": "legacy shared parent noise seed across conditions; not fresh augmentation",
              "source": source.as_posix(), "bytes_materialized": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
              "next_gate": "Broader numerical qualification including central floor and native-grid integration"}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    write_json(output / "generation_complete.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
