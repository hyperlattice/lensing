"""Independent phases of adaptive numerical qualification; no ML evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment, root as solve_root
import yaml

from scripts.qualify_benchmark_renderer_v1 import digest, geometry, relative_l1, write
from src.simulation.benchmark_renderer_v1 import SceneEngine
from src.simulation.adaptive_integration_v1 import adaptive_clean
from src.simulation.priors import draw_observation_parameters

V1 = Path("data/simulations/renderer-qualification-v1-complete")
ROOT = Path("data/simulations/renderer-qualification-v2")
REPORT = Path("results/reports/renderer-qualification-v2")
SOURCES = ["src/simulation/benchmark_renderer_v1.py", "src/simulation/adaptive_integration_v1.py",
           "src/simulation/autolens_backend_v2.py", "src/simulation/priors.py",
           "scripts/qualify_adaptive_renderer_v1.py", "docs/renderer_qualification_protocol_v2.md"]


def metadata():
    return json.loads((V1/"references.json").read_text())


def source_hashes():
    return {p: digest(p) for p in SOURCES}


def fresh_folder(path):
    if path.exists():
        raise FileExistsError(f"Use a new evidence identity: {path}")
    path.mkdir(parents=True)


def export():
    fresh_folder(ROOT)
    m = metadata()
    inputs = source_hashes()
    write(ROOT/"started.json", {"source_sha256": inputs, "v1_sha256": digest(V1/"references.json")})
    records, failures = [], []
    current, engine = None, None
    started = time.perf_counter()
    for row in m["cases"]:
        try:
            if row["index"] != current:
                engine, current = SceneEngine(row["parameters"], "lenstronomy"), row["index"]
            clean, diagnostics = adaptive_clean(engine, row["condition"])
            rendered = engine.expected(row["condition"], factor=4, clean=clean)
            np.savez(ROOT/f"{row['name']}.npz", **rendered)
            records.append({**row, "adaptive_diagnostics": diagnostics,
                            "array_sha256": digest(ROOT/f"{row['name']}.npz")})
        except Exception as exc:
            failures.append({"name": row["name"], "error": type(exc).__name__, "message": str(exc)})
        if (len(records)+len(failures)) % 9 == 0:
            print(json.dumps({"adaptive_A": len(records), "failures": len(failures), "seconds": time.perf_counter()-started}), flush=True)
    write(ROOT/"references.json", {"cases": records, "failures": failures,
          "original_proposal_failures": m["failures"], "source_sha256": inputs,
          "runtime_seconds": time.perf_counter()-started, "final_holdout_accessed": False})


def checked_references():
    m = json.loads((ROOT/"references.json").read_text())
    if m["source_sha256"] != source_hashes():
        raise ValueError("Adaptive numerical sources changed")
    for r in m["cases"]:
        if digest(ROOT/f"{r['name']}.npz") != r["array_sha256"]:
            raise ValueError("Adaptive reference changed")
    return m


def refine():
    m = checked_references()
    output = REPORT/"fine_reference"
    fresh_folder(output)
    names = {r["name"] for r in m["cases"] if not r["convergence_passed"] or r["kind"] != "random_prior"}
    rng = np.random.default_rng(720301)
    names.update(rng.choice([r["name"] for r in m["cases"]], size=12, replace=False))
    write(output/"started.json", {"selected_cases": sorted(names), "source_sha256": source_hashes()})
    rows, failures, current, engine = [], [], None, None
    started = time.perf_counter()
    for row in m["cases"]:
        if row["name"] not in names:
            continue
        try:
            if row["index"] != current:
                engine, current = SceneEngine(row["parameters"], "lenstronomy"), row["index"]
            prior, refinements = None, {}
            finest = None
            for factor in (16, 32, 64):
                rendered = engine.expected(row["condition"], factor=factor)
                if prior is not None:
                    changes = {k: relative_l1(prior[k], rendered[k]) for k in ("clean", "expected")}
                    refinements[f"{factor//2}_to_{factor}"] = changes
                    if max(changes.values()) <= .005:
                        finest = rendered
                        break
                prior = rendered
                finest = rendered
            with np.load(ROOT/f"{row['name']}.npz") as adaptive:
                differences = {k: relative_l1(adaptive[k], finest[k]) for k in ("clean", "expected")}
            reference_pass = max(list(refinements.values())[-1].values()) <= .005
            record = {"name": row["name"], "uniform_finest_factor": factor, "uniform_refinement_l1": refinements,
                      "adaptive_vs_uniform_l1": differences,
                      "passed": bool(reference_pass and max(differences.values()) <= .005)}
            rows.append(record)
            write(output/f"{row['name']}.json", record)
        except Exception as exc:
            failures.append({"name": row["name"], "error": type(exc).__name__, "message": str(exc)})
        print(json.dumps({"fine_cases": len(rows), "selected": len(names), "failures": len(failures),
                          "seconds": time.perf_counter()-started}), flush=True)
    write(output/"report.json", {"cases": rows, "failures": failures, "selected_cases": sorted(names),
         "passed": not failures and len(rows)==len(names) and all(r["passed"] for r in rows),
         "runtime_seconds": time.perf_counter()-started, "source_sha256": source_hashes()})


def validate():
    m = checked_references()
    output = REPORT/"cross_engine"
    fresh_folder(output)
    write(output/"started.json", {"source_sha256": source_hashes()})
    rows, failures, current, engine = [], [], None, None
    started = time.perf_counter()
    for row in m["cases"]:
        try:
            if row["index"] != current:
                engine, current = SceneEngine(row["parameters"], "autolens"), row["index"]
            clean, diagnostics = adaptive_clean(engine, row["condition"])
            rendered = engine.expected(row["condition"], factor=4, clean=clean)
            with np.load(ROOT/f"{row['name']}.npz") as reference:
                distance = np.linalg.norm(rendered["positions_xy"][:, None]-reference["positions_xy"][None], axis=2)
                if distance.shape[0] != distance.shape[1]:
                    raise ValueError("Point multiplicity mismatch")
                i,j = linear_sum_assignment(distance)
                position = float(distance[i,j].max())
                amplitude = float(np.max(abs(rendered["point_amplitudes"][i]/reference["point_amplitudes"][j]-1)))
                differences = {k: relative_l1(rendered[k], reference[k]) for k in ("clean", "expected")}
            record = {"name": row["name"], "cross_engine_l1": differences,
                      "position_error_arcsec": position, "amplitude_relative_error": amplitude,
                      "adaptive_diagnostics": diagnostics,
                      "passed": bool(max(differences.values())<=1e-5 and position<=1e-5 and amplitude<=1e-3)}
            rows.append(record)
            write(output/f"{row['name']}.json", record)
        except Exception as exc:
            failures.append({"name": row["name"], "error": type(exc).__name__, "message": str(exc)})
        if (len(rows)+len(failures)) % 9 == 0:
            print(json.dumps({"adaptive_B": len(rows), "failures": len(failures), "seconds": time.perf_counter()-started}), flush=True)
    # Independently classify the single-image proposal, rather than hiding it.
    proposal = yaml.safe_load(Path("configs/simulation_engineering_v1.yaml").read_text())
    p = draw_observation_parameters(720101, 72, proposal)
    from src.simulation.independent_sie import solve_independent_sie_images
    independent = solve_independent_sie_images(source_x_arcsec=p["source_x_arcsec"], source_y_arcsec=p["source_y_arcsec"],
        theta_E_arcsec=p["theta_E_arcsec"], axis_ratio=p["q"], phi_rad=p["phi_rad"],
        center_x_arcsec=p["lens_center_x_arcsec"], center_y_arcsec=p["lens_center_y_arcsec"], angular_starts=96)
    external_engine = SceneEngine(p, "autolens")
    center = np.array([p["lens_center_x_arcsec"], p["lens_center_y_arcsec"]])
    source = np.array([p["source_x_arcsec"], p["source_y_arcsec"]])
    def residual(x):
        return x-external_engine.alpha(x[None])[0]-source
    solutions = []
    for radius in (.1,.35,.65,.9,1.1,1.4,1.8,2.5,4.):
        for angle in np.linspace(0, 2*np.pi, 96, endpoint=False):
            solved = solve_root(residual, center+radius*p["theta_E_arcsec"]*np.array([np.cos(angle),np.sin(angle)]))
            x = solved.x
            if np.isfinite(x).all() and np.linalg.norm(residual(x))<1e-7 and np.linalg.norm(x-center)<4*p["theta_E_arcsec"]:
                if not any(np.linalg.norm(x-other)<1e-5 for other in solutions):
                    solutions.append(x)
    classification = {"proposal_index": 72, "A_count": 1, "independent_count": len(independent),
                      "external_broad_scan_count": len(solutions), "broad_scan_starts": 864,
                      "agreed_single_image": len(independent)==1 and len(solutions)==1}
    write(output/"report.json", {"cases": rows, "failures": failures, "single_image_classification": classification,
         "passed": not failures and len(rows)==len(m["cases"]) and all(r["passed"] for r in rows) and classification["agreed_single_image"],
         "runtime_seconds": time.perf_counter()-started, "source_sha256": source_hashes()})


def assemble():
    m = checked_references()
    fine = json.loads((REPORT/"fine_reference/report.json").read_text())
    cross = json.loads((REPORT/"cross_engine/report.json").read_text())
    old = json.loads(Path("results/reports/renderer-qualification-v1/report.json").read_text())
    primitives = all(r["passed"] for r in old["profile_checks"]+old["psf_checks"]) and old["detector_check"]["passed"]
    if fine["source_sha256"] != source_hashes() or cross["source_sha256"] != source_hashes():
        raise ValueError("Stale adaptive numerical reports")
    path = REPORT/"qualification.json"
    if path.exists():
        raise FileExistsError("Qualification result already exists")
    result = {"passed": bool(fine["passed"] and cross["passed"] and primitives and not m["failures"]),
        "rendering_factor": "adaptive_midpoint_1e-4_max512_v1", "source_sha256": source_hashes(),
        "proposal_parents": 106, "multiple_image_parents": len({r['index'] for r in m['cases']}),
        "declared_single_image_rejections": 1, "unconditional_single_image_fraction": 1/106,
        "validated_views": len(cross["cases"]), "fine_reference_views": len(fine["cases"]),
        "profile_psf_detector_primitives_passed": bool(primitives),
        "fine_report_sha256": digest(REPORT/"fine_reference/report.json"),
        "cross_engine_report_sha256": digest(REPORT/"cross_engine/report.json"),
        "v1_report_sha256": digest("results/reports/renderer-qualification-v1/report.json"),
        "final_holdout_accessed": False, "cnn_inference_run": False,
        "scope": "Finite numerical development grid, conditional on declared 2/4-image population; shared integration, PSF and detector"}
    write(path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "refine", "validate", "assemble"])
    args = parser.parse_args()
    globals()[args.action]()
