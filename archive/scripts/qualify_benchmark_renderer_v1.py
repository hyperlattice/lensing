"""Export renderer-A numerical references, then validate B in its environment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ["NUMBA_DISABLE_JIT"] = "1"
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve
import yaml

from src.simulation.benchmark_renderer_v1 import SceneEngine
from src.simulation import autolens_backend_v2 as kernels
from src.simulation.priors import draw_observation_parameters

CONFIG = Path("configs/renderer_qualification_v1.yaml")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def geometry(g):
    return {**g, "psf_family": "gaussian", "exposure_multiplier": 1., "read_noise_multiplier": 1.}


def relative_l1(a, b):
    return float(np.abs(a-b).sum()/max(np.abs(b).sum(), 1e-100))


def export(config_path=CONFIG):
    config = yaml.safe_load(Path(config_path).read_text())
    assert importlib.metadata.version("lenstronomy") == config["required_lenstronomy_version"]
    output = Path(config["reference_output"])
    if output.exists():
        raise FileExistsError("Use a new numerical release ID")
    output.mkdir(parents=True)
    write(output/"started.json", {"config_sha256": digest(config_path)})
    proposal = yaml.safe_load(Path(config["source_config"]).read_text())
    records, failures, lights = [], [], []
    started = time.perf_counter()
    # Includes the floor and exact center as well as off-axis radial samples.
    center = np.array([-.13, .07])
    radii = np.r_[0., 1e-8, 1e-5, 1e-4, np.geomspace(3e-4, 10., 96)]
    angles = np.arange(8)*np.pi/4
    xy = center+np.stack((np.outer(radii, np.cos(angles)).ravel(),
                          np.outer(radii, np.sin(angles)).ravel()), 1)
    np.save(output/"profile_xy.npy", xy)
    base = draw_observation_parameters(config["seed"], 1000, proposal)
    for n in (.5, 1., 2., 4.):
        for q in (1., .7, .4):
            for phi in (0., .3, 1.1):
                p = {**base, "source_amp": 1.3, "source_re_arcsec": .2, "source_n": n,
                     "source_q": q, "source_phi_rad": phi,
                     "source_x_arcsec": center[0], "source_y_arcsec": center[1]}
                name = f"light-{len(lights):03d}"
                values = SceneEngine(p, "lenstronomy").light("source", xy)
                np.save(output/f"{name}.npy", values)
                lights.append({"name": name, "parameters": p})
    count = config["random_parent_count"]
    for index in range(count+6):
        p = draw_observation_parameters(config["seed"], index, proposal)
        kind = "random_prior"
        if index == count:
            kind = "nearly_aligned"
            p.update(source_x_arcsec=p["lens_center_x_arcsec"]+1e-3,
                     source_y_arcsec=p["lens_center_y_arcsec"]+2e-3)
        elif index == count+1:
            kind = "small_source_offset"
            p.update(source_x_arcsec=p["lens_center_x_arcsec"]+.01,
                     source_y_arcsec=p["lens_center_y_arcsec"]+.02)
        elif index == count+2:
            kind = "pixel_center"
            p.update(lens_center_x_arcsec=.025, lens_center_y_arcsec=.025)
        elif index == count+3:
            kind = "subpixel_boundary"
            p.update(lens_center_x_arcsec=0., lens_center_y_arcsec=0.)
        elif index == count+4:
            kind = "compact_high_index"
            p.update(source_re_arcsec=.05, source_n=4.)
        elif index == count+5:
            kind = "elongated_low_q"
            p.update(q=.5, source_q=.4)
        engine = SceneEngine(p, "lenstronomy")
        for g in config["geometries"]:
            name = f"parent-{index:03d}-{g['name']}"
            try:
                cond = geometry(g)
                clean = {f: engine.clean(cond, factor=f) for f in config["integration_factors"]}
                reference = engine.expected(cond, factor=config["reference_factor"], clean=clean[config["reference_factor"]])
                kernel = reference["psf"]
                convergence = {"clean": relative_l1(clean[4], clean[8]),
                    "expected": float(np.abs(fftconvolve(clean[4]-clean[8], kernel, mode="same")).sum()/max(reference["expected"].sum(), 1e-100))}
                np.savez(output/f"{name}.npz", **reference)
                records.append({"name": name, "index": index, "kind": kind, "parameters": p,
                    "condition": cond, "factor_4_to_8_l1": convergence,
                    "convergence_passed": bool(max(convergence.values()) <= config["convergence_normalized_l1_tolerance"])})
            except Exception as exc:
                failures.append({"name": name, "kind": kind, "error": type(exc).__name__, "message": str(exc)})
        print(json.dumps({"parent": index+1, "total": count+6, "accepted": len(records),
                          "failures": len(failures), "seconds": time.perf_counter()-started}), flush=True)
    paths = [Path(__file__), Path("src/simulation/benchmark_renderer_v1.py"),
             Path("src/simulation/autolens_backend_v2.py"), Path("src/simulation/priors.py"),
             Path(config_path), Path(config["source_config"])]
    metadata = {"stage": "numerical_development", "requested_parents": count+6,
                "requested_observations": (count+6)*len(config["geometries"]),
                "cases": records, "failures": failures, "lights": lights,
                "source_sha256": {p.as_posix(): digest(p) for p in paths},
                "array_sha256": {p.name: digest(p) for p in output.iterdir() if p.suffix in {".npy", ".npz"}},
                "runtime_seconds": time.perf_counter()-started,
                "environment": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "lenstronomy")},
                "final_holdout_accessed": False, "cnn_inference_run": False}
    write(output/"references.json", metadata)
    print("Reference export complete", flush=True)


def validate(config_path=CONFIG):
    config = yaml.safe_load(Path(config_path).read_text())
    assert importlib.metadata.version("autolens") == config["required_autolens_version"]
    root, output = Path(config["reference_output"]), Path(config["report_output"])
    if output.exists():
        raise FileExistsError("Use a new numerical report ID")
    metadata = json.loads((root/"references.json").read_text())
    for name, expected in metadata["array_sha256"].items():
        if digest(root/name) != expected:
            raise ValueError("Reference array changed")
    for path, expected in metadata["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Numerical source/config changed since export")
    output.mkdir(parents=True)
    write(output/"started.json", {"references_sha256": digest(root/"references.json")})
    started = time.perf_counter()
    profiles, scenes, failures = [], [], list(metadata["failures"])
    xy = np.load(root/"profile_xy.npy")
    for row in metadata["lights"]:
        values = SceneEngine(row["parameters"], "autolens").light("source", xy)
        expected = np.load(root/f"{row['name']}.npy")
        error = float(np.max(abs(values-expected)/np.maximum(abs(expected), 1e-100)))
        p = row["parameters"]
        bn = max(1.9992*p["source_n"]-.3271, 1e-5)
        center = p["source_amp"]*np.exp(-bn*((1e-4/p["source_re_arcsec"])**(1/p["source_n"])-1))
        center_error = abs(float(values[0])/center-1)
        profiles.append({"name": row["name"], "max_relative_error": error,
            "analytic_center_relative_error": center_error,
            "passed": bool(max(error, center_error) <= config["profile_relative_tolerance"])})
    engine, current = None, None
    for row in metadata["cases"]:
        try:
            if row["index"] != current:
                engine, current = SceneEngine(row["parameters"], "autolens"), row["index"]
            rendered = engine.expected(row["condition"], factor=config["reference_factor"])
            with np.load(root/f"{row['name']}.npz") as reference:
                if len(rendered["positions_xy"]) != len(reference["positions_xy"]):
                    raise ValueError("Point multiplicity mismatch")
                distances = np.linalg.norm(rendered["positions_xy"][:, None]-reference["positions_xy"][None], axis=2)
                i, j = linear_sum_assignment(distances)
                position = float(distances[i, j].max())
                amplitude = float(np.max(abs(rendered["point_amplitudes"][i]/reference["point_amplitudes"][j]-1)))
                errors = {k: relative_l1(rendered[k], reference[k]) for k in ("clean", "expected")}
            scenes.append({"name": row["name"], "kind": row["kind"], "normalized_l1": errors,
                "point_position_error_arcsec": position, "point_amplitude_relative_error": amplitude,
                "factor_4_to_8_l1": row["factor_4_to_8_l1"],
                "cross_engine_passed": bool(max(errors.values()) <= config["cross_engine_normalized_l1_tolerance"] and
                    position <= config["point_position_tolerance_arcsec"] and amplitude <= config["point_amplitude_relative_tolerance"]),
                "convergence_passed": row["convergence_passed"]})
        except Exception as exc:
            failures.append({"name": row["name"], "error": type(exc).__name__, "message": str(exc)})
        if len(scenes) % 9 == 0:
            print(json.dumps({"validated": len(scenes), "failures": len(failures), "seconds": time.perf_counter()-started}), flush=True)
    # PSF and detector controls avoid any CNN or accuracy scoring.
    psfs = []
    for scale in (.04, .05, .06):
        for family, beta in (("gaussian", 3.5), ("moffat", 2.5), ("moffat", 3.5), ("moffat", 4.5)):
            kernel = kernels.psf_kernel(.15, scale, family, beta)
            psfs.append({"scale": scale, "family": family, "beta": beta,
                         "passed": bool(np.isfinite(kernel).all() and np.all(kernel>=0) and abs(kernel.sum()-1)<1e-12)})
    raw, variance = kernels.sample_detector(np.full((1024, 1024), .3), 100., 2., .03, 720102)
    mean_error = abs(raw.mean()-.33)
    variance_error = abs(raw.var()/variance[0, 0]-1)
    detector = {"mean_error": float(mean_error), "variance_relative_error": float(variance_error),
                "passed": bool(mean_error < 6*np.sqrt(variance[0, 0]/raw.size) and variance_error < .02)}
    passed = not failures and all(r["passed"] for r in profiles+psfs) and detector["passed"] and all(
        r["cross_engine_passed"] and r["convergence_passed"] for r in scenes)
    result = {"experiment_id": config["experiment_id"], "passed": bool(passed),
              "requested_parents": metadata["requested_parents"], "requested_observations": metadata["requested_observations"],
              "profile_checks": profiles, "scene_checks": scenes, "psf_checks": psfs,
              "detector_check": detector, "failures": failures,
              "reference_sha256": digest(root/"references.json"), "source_sha256": metadata["source_sha256"],
              "backend_environment": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "autolens", "autogalaxy")},
              "validation_runtime_seconds": time.perf_counter()-started,
              "final_holdout_accessed": False, "cnn_inference_run": False,
              "scope": "Finite development sample, shared integration/PSF/detector; factor 8 remains a candidate reference"}
    write(output/"report.json", result)
    print(json.dumps({"passed": bool(passed), "scenes": len(scenes), "failures": len(failures)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "validate"])
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args()
    {"export": export, "validate": validate}[args.action](Path(args.config))
