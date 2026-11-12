"""Download, inspect and stage explicitly referenced real images without scoring.

Catalog radii remain observational estimates. They are never written as exact
simulator targets. Intake can succeed while scientific scoring remains blocked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

DEFAULT_CONFIG = Path("configs/observations/reference_intake.yaml")


def read_config(path=DEFAULT_CONFIG):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))




def _filename(value):
    if Path(value).name != value or value in {"", ".", ".."}:
        raise ValueError("Asset filename must be a plain filename")
    return value


def fetch(config_path=DEFAULT_CONFIG):
    config = read_config(config_path)
    root = Path(config["raw_root"])
    root.mkdir(parents=True, exist_ok=True)
    entries = [(root, config["reference_catalog"])]
    for sample in config["samples"]:
        sample_root = root / _filename(sample["sample_id"])
        sample_root.mkdir(exist_ok=True)
        entries.extend((sample_root, spec) for spec in sample["assets"].values())
    records = []
    for folder, spec in entries:
        path = folder / _filename(spec["filename"])
        if path.exists():
            data = path.read_bytes()
        else:
            if not spec["url"].startswith("https://"):
                raise ValueError("External asset URL must use HTTPS")
            with urllib.request.urlopen(spec["url"], timeout=60) as response:
                data = response.read(20_000_001)
            if len(data) > 20_000_000:
                raise ValueError("Pilot asset exceeds the 20 MB intake limit")
        if not path.exists():
            path.write_bytes(data)
        records.append({"path": path.as_posix(), "url": spec["url"], "bytes": len(data)})
    (root / "download_manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return {"available_assets": len(records), "raw_root": root.as_posix()}


def catalog_rows(path):
    """Read the actual catalog column names; do not infer errors from other columns."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# name "):
        raise ValueError("Unexpected reference catalog header")
    names = lines[0].lstrip("# ").split()
    frame = pd.read_csv(path, sep=r"\s+", comment="#", names=names)
    if frame["name"].duplicated().any():
        raise ValueError("Duplicate lens identities in reference catalog")
    return frame.set_index("name")


def readiness_blockers(sample: dict, structural_errors: list[str], image_unit: str):
    blockers = list(structural_errors)
    for key in ("center_reviewed", "calibration_reviewed", "real_image_provenance_reviewed",
                "reference_quality_reviewed", "target_mapping_reviewed"):
        if sample.get(key) is not True:
            blockers.append(key + "_required")
    sigma = sample.get("reference_sigma_arcsec")
    if sigma is None or not np.isfinite(float(sigma)) or float(sigma) <= 0:
        blockers.append("individual_reference_uncertainty_required")
    if not image_unit or image_unit.lower() == "unknown":
        blockers.append("verified_science_units_required")
    return blockers


def prepare(config_path=DEFAULT_CONFIG):
    """Inspect native geometry and preserve supplied noise, without arbitrary scaling."""
    config = read_config(config_path)
    raw_root, output = Path(config["raw_root"]), Path(config["output"])
    catalog_spec = config["reference_catalog"]
    catalog_path = raw_root / _filename(catalog_spec["filename"])
    catalog = catalog_rows(catalog_path)
    output.mkdir(parents=True, exist_ok=True)
    rows, reports = [], []
    for sample in config["samples"]:
        sample_id = _filename(sample["sample_id"])
        arrays, headers = {}, {}
        for kind, spec in sample["assets"].items():
            path = raw_root / sample_id / _filename(spec["filename"])
            with fits.open(path) as hdus:
                hdu = hdus[int(sample["science_hdu"]) if kind == "science" else 0]
                arrays[kind] = np.array(hdu.data, dtype=np.float32, copy=True)
                headers[kind] = {key: str(hdu.header[key]) for key in
                                ("BUNIT", "TELESCOP", "INSTRUME", "FILTER", "EXPTIME", "OBJECT") if key in hdu.header}
        science, noise, psf = (arrays[key] for key in ("science", "noise", "psf"))
        errors = []
        if science.ndim != 2 or noise.shape != science.shape:
            errors.append("science_noise_geometry_invalid")
        if any(not np.isfinite(array).all() for array in arrays.values()):
            errors.append("nonfinite_image_products")
        if np.any(noise <= 0):
            errors.append("nonpositive_noise_map")
        if psf.ndim != 2 or not np.isfinite(psf.sum()) or psf.sum() <= 0 or np.any(psf < 0):
            errors.append("invalid_psf")
        if sample["noise_semantics"] != "standard_deviation":
            raise ValueError("Pilot supports explicitly declared standard-deviation noise maps only")
        scale = float(sample["pixel_scale_arcsec"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid angular pixel scale")
        reference = catalog.loc[sample["lens_id"]]
        radius = float(reference["theta_Ein"])
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Invalid catalog radius")
        image_unit = headers["science"].get("BUNIT", sample["image_unit"])
        blockers = readiness_blockers(sample, errors, image_unit)
        folder = output / "products" / sample_id
        if not errors:
            folder.mkdir(parents=True, exist_ok=True)
            for name, array in {"science": science, "variance": noise**2,
                                "psf": psf / psf.sum(), "invalid_mask": ~np.isfinite(science)}.items():
                np.save(folder / f"{name}.npy", array)
        rows.append({"sample_id": sample_id, "lens_id": sample["lens_id"], "domain": "real",
                     "pixel_scale": scale, "original_shape": "x".join(map(str, science.shape)),
                     "image_unit": image_unit, "reference_radius_arcsec": radius,
                     "reference_sigma_arcsec": sample.get("reference_sigma_arcsec"),
                     "reference_definition": sample["reference_definition"],
                     "reference_citation": catalog_spec["citation"],
                     "accuracy_ready": not blockers, "status": "blocked" if blockers else "ready"})
        reports.append({**rows[-1], "structural_errors": errors, "blockers": blockers,
                        "comparison_references": sample.get("comparison_references", []),
                        "headers": headers, "psf_input_sum": float(psf.sum()) if np.isfinite(psf.sum()) else None,
                        "angular_pixel_extent_arcsec": [float(n * scale) for n in science.shape]})
    pd.DataFrame(rows).to_csv(output / "reference_intake.csv", index=False)
    report = {"experiment_id": config["experiment_id"], "sample_count": len(rows),
              "accuracy_ready_count": sum(row["accuracy_ready"] for row in rows),
              "cnn_inference_run": False, "samples": reports}
    path = Path(config["report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "prepare"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    result = fetch(args.config) if args.action == "fetch" else prepare(args.config)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
