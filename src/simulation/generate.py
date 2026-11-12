"""Deterministic process-parallel rendering of the declared development banks.

Physical settings and bank schema match generate_robustness_banks_v1. Worker
completion order cannot change the cohort: results are consumed in proposal
order. All computed proposals, including rejected source proposals, are
audited; numerical failures anywhere block bank completion.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from src.data.dataset import ObservationBank
from src.data.schema import validate_lineage
from src.data.validation import product_records, save_array, write
from src.ml.artifacts import sha256 as digest
from src.simulation.independent import solve_independent_sie_images
from src.simulation.observation import condition_for
from src.simulation.render import SceneEngine, adaptive_clean
from src.simulation.scene import (
    SOURCE,
    identity_ledger,
    physical_fingerprint,
    source_proposals,
)

QUALIFICATION = Path("results/reports/renderer-qualification-v5/qualification.json")


def independent_positions(p: dict[str, Any]) -> np.ndarray:
    """Solve independent SIE positions to distinguish physical single-image cases."""
    return solve_independent_sie_images(
        source_x_arcsec=p["source_x_arcsec"],
        source_y_arcsec=p["source_y_arcsec"],
        theta_E_arcsec=p["theta_E_arcsec"],
        axis_ratio=p["q"],
        phi_rad=p["phi_rad"],
        center_x_arcsec=p["lens_center_x_arcsec"],
        center_y_arcsec=p["lens_center_y_arcsec"],
        angular_starts=96,
    )


def write_native_masks(output: str | Path) -> None:
    """Write all-valid masks for the three retained native geometries."""
    for cell in ("fresh-noise", "pixel-scale-004", "pixel-scale-006"):
        shape = tuple(condition_for(cell)["shape"])
        path = Path(output) / f"masks/{shape[0]}x{shape[1]}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.zeros(shape, dtype=bool))


def render_one(task: tuple) -> dict[str, Any]:
    """Render one source parent and retain diagnostics if numerical work fails."""
    role, index, row, p, historical, historical_seed, output = task
    output = Path(output)
    parent = row["parent_system_id"]
    identity = {
        "parent_system_id": parent,
        "role": role,
        "parameter_seed": int(p["parameter_seed"]),
        "noise_seed": int(p["noise_seed"]),
        "ancestry_ids": [],
        "latent_sha256": physical_fingerprint(p),
    }
    diagnostics = []
    observations = []
    try:
        if any(abs(float(p.get(k, 0.0))) > 0 for k in ("gamma1", "gamma2")):
            raise ValueError("This bank is restricted to the declared no-shear SIE prior")
        engine = SceneEngine(p, "lenstronomy")
        positions, _ = engine.points()
        expected = {}
        for cell in ("fresh-noise", "pixel-scale-004", "pixel-scale-006"):
            condition = condition_for(cell)
            clean, diag = adaptive_clean(engine, condition)
            diagnostics.append({"parent": parent, "geometry": cell, **diag})
            rendered = engine.expected(condition, factor=4, clean=clean)
            key = "base" if cell == "fresh-noise" else cell
            expected[key] = save_array(output, f"products/{parent}-{key}.npy", rendered["expected"])
            psf = save_array(output, f"products/{parent}-{key}-psf.npy", rendered["psf"])
            observations.append(
                {
                    "parent": parent,
                    "condition": condition,
                    "expected_rate": expected[key],
                    "psf": psf,
                    "variance_method": "reconstruct from expected rate and explicit detector parameters for each training draw",
                    "mask_semantics": "all_native_pixels_valid; no crop/pad; background retained",
                }
            )
            if cell == "fresh-noise":
                for width in ("gaussian-fwhm-075", "gaussian-fwhm-125"):
                    shifted = engine.expected(condition_for(width), factor=4, clean=clean)
                    expected[width] = save_array(
                        output, f"products/{parent}-{width}.npy", shifted["expected"]
                    )
                    psf = save_array(output, f"products/{parent}-{width}-psf.npy", shifted["psf"])
                    observations.append(
                        {
                            "parent": parent,
                            "condition": condition_for(width),
                            "expected_rate": expected[width],
                            "psf": psf,
                            "variance_method": "reconstruct from expected rate and explicit detector parameters for each training draw",
                            "mask_semantics": "all_native_pixels_valid; no crop/pad; background retained",
                        }
                    )
        parameter_path = SOURCE / row["parameters_path"]
        identity.update(
            expected=expected,
            detector={
                "exposure_s": p["exposure_time"],
                "read_noise_electrons": p["read_noise_electrons"],
                "background_rate": p["background_rate"],
            },
            historical_condition=historical,
            historical_noise_seed=historical_seed,
            source_parameter_sha256=digest(parameter_path),
            source_parameter_path=parameter_path.as_posix(),
        )
        for observation in observations:
            h, w = observation["condition"]["shape"]
            relative = f"masks/{h}x{w}.npy"
            mask = {"path": relative, "sha256": digest(output / relative)}
            observation.update(invalid_mask=mask, coverage_mask=mask)
        return {
            "proposal": index,
            "identity": identity,
            "label": p["theta_E_arcsec"],
            "diagnostics": diagnostics,
            "observations": observations,
            "failure": None,
        }
    except Exception as exc:
        single_count = "found 1 roots" in str(exc) or "root multiplicity 1" in str(exc)
        approved_single = single_count and len(independent_positions(p)) == 1
        return {
            "proposal": index,
            "identity": identity,
            "label": None,
            "diagnostics": diagnostics,
            "observations": observations,
            "failure": {
                "proposal": index,
                "parent": parent,
                "error": type(exc).__name__,
                "message": str(exc),
                "approved_physical_single_image_rejection": bool(approved_single),
            },
        }


def run(role: str, workers: int) -> None:
    """Generate a bounded-worker source bank using the unchanged v4 destinations."""
    if role not in {"source_train", "source_development"}:
        raise ValueError("Only source training/development banks are maintained")
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError("Use 1-2 bounded rendering workers; run roles sequentially")
    q = json.loads(QUALIFICATION.read_text())
    if q["passed"] is not True:
        raise ValueError("Numerical qualification failed")
    for path, expected in q["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Qualified rendering source changed")
    name = {
        "source_train": "source-train",
        "source_development": "source-development",
    }[role]
    output = Path(f"data/simulations/robustness-study-{name}-v4")
    report_path = Path(f"results/reports/robustness-study-banks-v4/{name}.json")
    if report_path.exists() or (output / "complete.json").exists():
        raise FileExistsError("This bank role is already complete")
    ledger, input_hashes = identity_ledger()
    proposals, more = source_proposals(role)
    input_hashes.update(more)
    count = len(proposals)
    for row, p, _, _ in proposals:
        validate_lineage(
            [
                {
                    "parent_system_id": row["parent_system_id"],
                    "role": role,
                    "parameter_seed": int(p["parameter_seed"]),
                    "noise_seed": int(p["noise_seed"]),
                    "ancestry_ids": [],
                }
            ],
            ledger,
        )
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "started.json").exists():
        write_native_masks(output)
    elif not all((output / f"masks/{h}x{h}.npy").exists() for h in (256, 320, 213)):
        raise ValueError("Existing v4 checkpoint has incomplete native masks")
    started = time.perf_counter()
    started_record = {
        "role": role,
        "workers": workers,
        "requested_parents": count,
        "computed_proposal_pool": len(proposals),
        "qualification_sha256": digest(QUALIFICATION),
        "source_identity_sha256": input_hashes,
        "generator_sha256": {
            __file__: digest(__file__),
            "src/simulation/scene.py": digest("src/simulation/scene.py"),
        },
    }
    if (output / "started.json").exists():
        if json.loads((output / "started.json").read_text()) != started_record:
            raise ValueError("Existing v4 output belongs to different frozen inputs")
    else:
        write(output / "started.json", started_record)
    tasks = [(role, index, *record, output.as_posix()) for index, record in enumerate(proposals)]
    checkpoint = output / "checkpoint.json"
    completed = {}
    if checkpoint.exists():
        for result in json.loads(checkpoint.read_text()):
            refs = product_records(result)
            valid = True
            for ref in refs:
                path = output / ref["path"]
                if not path.exists():
                    valid = False
                    break
                if digest(path) != ref["sha256"]:
                    raise ValueError(f"Checkpoint product hash mismatch: {ref['path']}")
            if valid:
                completed[int(result["proposal"])] = result
    results = list(completed.values())
    pending = [task for task in tasks if int(task[1]) not in completed]
    since_checkpoint = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(render_one, pending):
            completed[int(result["proposal"])] = result
            results = [completed[i] for i in sorted(completed)]
            since_checkpoint += 1
            if since_checkpoint >= 10:
                tmp = checkpoint.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
                tmp.replace(checkpoint)
                since_checkpoint = 0
            if len(completed) % 10 == 0 or not pending:
                print(
                    json.dumps(
                        {
                            "role": role,
                            "computed": len(results),
                            "pool": len(tasks),
                            "failures": sum(r["failure"] is not None for r in results),
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    results = [completed[i] for i in range(len(tasks))]
    tmp = checkpoint.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(checkpoint)
    failures = [r["failure"] for r in results if r["failure"] is not None]
    accepted = [r for r in results if r["failure"] is None]
    # Preserve all proposal diagnostics even if a numerical failure blocks the
    # index. Source pools are filtered only by the declared 2/4-image rule;
    # they are not replenished with new parents or selected by CNN outcomes.
    write(output / "worker_results.json", results)
    requested = count
    count = len(accepted)
    selected = accepted[:count]
    if (
        not count
        or len(selected) != count
        or any(not f["approved_physical_single_image_rejection"] for f in failures)
    ):
        write(
            output / "failed.json",
            {"requested": count, "accepted": len(accepted), "failures": failures},
        )
        raise ValueError("Bank has unresolved numerical/cohort failures")
    for path, expected in q["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Qualified rendering source changed during generation")
    parents = [r["identity"] for r in selected]
    labels = None
    write(
        output / "source_targets.json",
        {r["identity"]["parent_system_id"]: r["label"] for r in selected},
    )
    labels = {"path": "source_targets.json", "sha256": digest(output / "source_targets.json")}
    document = {
        "schema": "observation_bank_v1",
        "stage": "qualified_development",
        "role": role,
        "rendering_factor": q["rendering_factor"],
        "qualification": {"path": QUALIFICATION.as_posix(), "sha256": digest(QUALIFICATION)},
        "exclusion_ledger": ledger,
        "targets": labels,
        "parents": parents,
        "image_semantics": "unnormalized raw rate with background retained",
        "condition_access": "gaussian_only",
    }
    write(output / "observation_provenance.json", [o for r in results for o in r["observations"]])
    document.update(
        image_unit="electrons_per_second",
        target_definition="lenstronomy_SIE_circularized_parameter_arcsec",
        observation_provenance={
            "path": "observation_provenance.json",
            "sha256": digest(output / "observation_provenance.json"),
        },
    )
    write(output / "bank.json", document)
    bank = ObservationBank(output / "bank.json", expected_role=role)
    selected_ids = set(bank.by_id)
    write(
        output / "proposal_audit.json",
        [
            {
                "proposal": r["proposal"],
                **{
                    k: r["identity"][k]
                    for k in ("parent_system_id", "parameter_seed", "noise_seed", "latent_sha256")
                },
                "selected": r["identity"]["parent_system_id"] in selected_ids,
                "failure": r["failure"],
            }
            for r in results
        ],
    )
    write(output / "integration_diagnostics.json", [d for r in results for d in r["diagnostics"]])
    report = {
        "role": role,
        "parents": len(bank.parents),
        "requested_parents": requested,
        "computed_proposals": len(results),
        "accepted_proposals": len(accepted),
        "unused_accepted": len(accepted) - count,
        "failures": failures,
        "manifest_sha256": digest(output / "bank.json"),
        "qualification_sha256": digest(QUALIFICATION),
        "runtime_seconds": time.perf_counter() - started,
        "workers": workers,
        "bank_validated": True,
        "final_arrays_accessed": False,
        "final_identity_columns_only": True,
        "target_labels_serialized": False,
        "source_labels_serialized": True,
        "bytes_materialized": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write(report_path, report)
    write(output / "complete.json", report)
    print(json.dumps(report, indent=2))
