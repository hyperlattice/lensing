"""Product writing and numerical qualification identity checks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.ml.artifacts import sha256 as digest


def write(path: str | Path, value: Any) -> None:
    """Write an explicit report to an already prepared output directory."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_array(output: Path, relative: str, values: np.ndarray) -> dict[str, str]:
    """Save a float32 product and return its relative path and byte identity."""
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.float32))
    return {"path": relative, "sha256": digest(path)}


def qualify_rendering(renderers: list[str]) -> None:
    """Requalify the consolidated implementation against all 315 saved controls.

    This is an explicit, potentially expensive numerical run. It never opens
    model checkpoints or evaluation labels and never replaces a prior report.
    Both backends must pass before new paired comparison data can be generated.
    """
    from scipy.optimize import linear_sum_assignment

    from src.simulation.render import SceneEngine, adaptive_clean

    if not renderers or len(set(renderers)) != len(renderers):
        raise ValueError("Expected distinct rendering backends")
    if not set(renderers).issubset({"lenstronomy", "autolens"}):
        raise ValueError("Unknown physical backend")
    output = Path("results/reports/renderer-qualification-v5")
    references = Path("data/simulations/renderer-qualification-v3")
    metadata_path = references / "references.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(metadata["cases"]) != 315:
        raise ValueError("Qualification requires the complete 315-case reference set")
    if output.exists():
        raise FileExistsError("Qualification output already exists; preserve historical evidence")
    source_paths = [
        "src/simulation/render.py",
        "src/simulation/backends.py",
        "src/simulation/observation.py",
        "src/data/validation.py",
    ]
    source_hashes = {path: digest(path) for path in source_paths}
    reference_hashes = {str(metadata_path): digest(metadata_path)}
    output.mkdir(parents=True)
    write(output / "started.json", {"source_sha256": source_hashes, "renderers": renderers})
    started = time.perf_counter()
    records = []
    failures = []
    for renderer in renderers:
        for case in metadata["cases"]:
            reference_path = references / f"{case['name']}.npz"
            reference_hashes[str(reference_path)] = digest(reference_path)
            try:
                engine = SceneEngine(case["parameters"], renderer)
                clean, diagnostics = adaptive_clean(engine, case["condition"])
                actual = engine.expected(case["condition"], factor=4, clean=clean)
                with np.load(reference_path, allow_pickle=False) as expected:
                    positions = actual["positions_xy"]
                    reference_positions = expected["positions_xy"]
                    if positions.shape != reference_positions.shape:
                        raise ValueError("Point-image multiplicity differs")
                    distances = np.linalg.norm(
                        positions[:, None] - reference_positions[None], axis=2
                    )
                    left, right = linear_sum_assignment(distances)
                    position_error = float(distances[left, right].max())
                    amplitude_error = float(
                        np.max(
                            np.abs(
                                actual["point_amplitudes"][left]
                                / expected["point_amplitudes"][right]
                                - 1
                            )
                        )
                    )
                    image_error = float(
                        np.abs(actual["expected"] - expected["expected"]).sum()
                        / max(np.abs(expected["expected"]).sum(), 1e-100)
                    )
                records.append(
                    {
                        "name": case["name"],
                        "renderer": renderer,
                        "position_error_arcsec": position_error,
                        "amplitude_relative_error": amplitude_error,
                        "expected_normalized_l1": image_error,
                        "integration": diagnostics,
                        "passed": position_error <= 1e-5
                        and amplitude_error <= 1e-3
                        and image_error <= 1e-5,
                    }
                )
            except Exception as error:
                # Preserve the whole qualification cohort and fail the gate.
                failures.append(
                    {
                        "name": case["name"],
                        "renderer": renderer,
                        "error": type(error).__name__,
                        "message": str(error),
                    }
                )
    passed = not failures and all(record["passed"] for record in records)
    if any(digest(path) != expected for path, expected in source_hashes.items()):
        raise ValueError("Rendering source changed during qualification")
    report = {
        "passed": passed,
        "cases": records,
        "failures": failures,
        "qualified_backends": renderers,
        "source_sha256": source_hashes,
        "reference_sha256": reference_hashes,
        "rendering_factor": "adaptive_midpoint_1e-4_max512_precise_sparse_roots_v2",
        "runtime_seconds": time.perf_counter() - started,
    }
    write(output / "qualification.json", report)
    if not passed:
        raise ValueError("Rendering qualification failed; inspect the saved report")


def product_records(value: object) -> list[dict]:
    """Collect serialized path/hash pairs from a nested worker checkpoint."""
    records = []
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            records.append(value)
        for child in value.values():
            records.extend(product_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(product_records(child))
    return records
