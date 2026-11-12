"""Qualified paired pilot/final observations with truth isolated from inputs.

Prepare is run only after the appropriate freeze. A and B rendering phases
run sequentially in their respective Python environments with <=2 workers.
Every proposal is audited; numerical failures block assembly. Final manifests
contain image/variance/PSF/mask records and identities, never physical labels.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from src.data.schema import validate_lineage
from src.data.validation import save_array, write
from src.ml.artifacts import sha256
from src.simulation.generate import independent_positions
from src.simulation.observation import CONDITIONS, RENDERERS, evaluation_condition
from src.simulation.render import SceneEngine, adaptive_clean
from src.simulation.scene import (
    draw_observation_parameters,
    identity_ledger,
    physical_fingerprint,
)

QUALIFICATION = Path("results/reports/renderer-qualification-v5/qualification.json")


def verify_qualification() -> dict[str, Any]:
    """Require both physical backends to match the current numerical source identity."""
    q = json.loads(QUALIFICATION.read_text())
    if q["passed"] is not True:
        raise ValueError("Numerical qualification must pass")
    if set(q.get("qualified_backends", [])) != set(RENDERERS):
        raise ValueError("New paired releases require qualification of both physical backends")
    for path, expected in q["source_sha256"].items():
        if sha256(path) != expected:
            raise ValueError("Qualified rendering source changed")
    return q


def prepare(design_path: str | Path) -> None:
    """Freeze paired proposals and exclusions without using model predictions."""
    from src.ml.artifacts import (
        create_once,
        verify_method_freeze,
    )

    design = json.loads(Path(design_path).read_text())
    required = {
        "role",
        "release_directory",
        "parent_count",
        "generation_seed",
        "method_freeze",
        "evaluation_freeze",
        "exclusion_banks",
        "simulation_config",
        "noise_namespace",
    }
    if set(design) != required or design["role"] != "power_pilot":
        raise ValueError("Incomplete paired release design")
    method = verify_method_freeze(design["method_freeze"])
    if design["evaluation_freeze"] is not None:
        raise ValueError("The retained paired comparison does not accept a final evaluation freeze")
    if type(design["parent_count"]) is not int or design["parent_count"] < 2:
        raise ValueError("Invalid parent count")
    q = verify_qualification()
    # The fully specified generation design itself is an input to Freeze 1.
    if str(Path(design_path)) not in method["artifact_sha256"]:
        raise ValueError("Paired generation design must be bound before method freeze")
    output = Path(design["release_directory"])
    if output.exists():
        raise FileExistsError("Use a new release identity")
    excluded, hashes = identity_ledger()
    fingerprints = set()
    roles = set()
    for path in design["exclusion_banks"]:
        bank = json.loads(Path(path).read_text())
        hashes[str(Path(path))] = sha256(path)
        rows = bank.get("parents", bank.get("lineage", []))
        if not rows:
            raise ValueError("Exclusion bank contains no lineage")
        excluded.extend(rows)
        fingerprints.update(row["latent_sha256"] for row in rows)
        roles.update(row["role"] for row in rows)
        audit_path = Path(path).parent / "proposal_audit.json"
        if bank.get("schema") == "observation_bank_v1" and audit_path.exists():
            audit = json.loads(audit_path.read_text())
            # Include every unused/rejected adaptation proposal, not just the
            # selected bank. Its role is inherited from the validated bank.
            hashes[str(audit_path)] = sha256(audit_path)
            for row in audit:
                identity = {
                    k: row[k]
                    for k in ("parent_system_id", "parameter_seed", "noise_seed", "latent_sha256")
                }
                identity.update(role=bank["role"], ancestry_ids=[])
                excluded.append(identity)
                fingerprints.add(identity["latent_sha256"])
    if not {"source_train", "source_development", "target_adaptation"}.issubset(roles):
        raise ValueError("Missing scientific development-role exclusions")
    if design["role"] == "final" and "power_pilot" not in roles:
        raise ValueError("Final generation must exclude pilot proposals")
    config = yaml.safe_load(Path(design["simulation_config"]).read_text())
    hashes[design["simulation_config"]] = sha256(design["simulation_config"])
    pool = int(np.ceil(design["parent_count"] * 1.1)) + 20
    proposals = []
    for index in range(pool):
        p = draw_observation_parameters(design["generation_seed"], index, config)
        row = {
            "parent_system_id": f"{design['role']}-{design['generation_seed']}-{index:05d}",
            "role": design["role"],
            "parameter_seed": int(p["parameter_seed"]),
            "noise_seed": int(p["noise_seed"]),
            "ancestry_ids": [],
            "latent_sha256": physical_fingerprint(p),
        }
        if row["latent_sha256"] in fingerprints:
            raise ValueError("New proposal duplicates an earlier physical parent")
        fingerprints.add(row["latent_sha256"])
        proposals.append({"index": index, "lineage": row, "parameters": p})
    validate_lineage([row["lineage"] for row in proposals], excluded)
    output.mkdir(parents=True)
    (output / "truth_vault").mkdir()
    create_once(output / "truth_vault/proposals.json", proposals)
    create_once(
        output / "exclusion_lineage.json", {"lineage": [row["lineage"] for row in proposals]}
    )
    (output / "masks").mkdir()
    for shape in ((256, 256), (320, 320), (213, 213)):
        np.save(output / f"masks/{shape[0]}x{shape[1]}.npy", np.zeros(shape, dtype=bool))
    create_once(
        output / "generation_started.json",
        {
            "design": design,
            "design_sha256": sha256(design_path),
            "method_freeze_sha256": sha256(design["method_freeze"]),
            "evaluation_freeze_sha256": None,
            "qualification_sha256": sha256(QUALIFICATION),
            "rendering_factor": q["rendering_factor"],
            "generator_sha256": {str(Path(__file__)): sha256(__file__)},
            "exclusion_input_sha256": hashes,
            "proposals_sha256": sha256(output / "truth_vault/proposals.json"),
            "requested_parents": design["parent_count"],
            "proposal_pool": pool,
            "truth_isolated": True,
            "condition_order": list(CONDITIONS),
            "renderer_order": list(RENDERERS),
        },
    )


def render_one(task: tuple) -> dict[str, Any]:
    """Render all declared conditions for one parent in one physical backend."""
    output, renderer, record, namespace = task
    output = Path(output)
    p = record["parameters"]
    parent = record["lineage"]["parent_system_id"]
    observations = []
    diagnostics = []
    try:
        engine = SceneEngine(p, renderer)
        other = independent_positions(p)
        try:
            positions, _ = engine.points()
        except ValueError:
            # Independently identified single images must be agreed by the
            # selected engine, not inferred from a generic solver exception.
            raise
        if len(positions) != len(other):
            raise ValueError("Independent point multiplicity disagreement")
        i, j = linear_sum_assignment(np.linalg.norm(positions[:, None] - other[None], axis=2))
        if np.max(np.linalg.norm(positions[i] - other[j], axis=1)) > 1e-5:
            raise ValueError("Independent root position disagreement")
        cleans = {}
        for index, name in enumerate(CONDITIONS):
            condition = evaluation_condition(name)
            geometry = tuple(condition["shape"])
            if geometry not in cleans:
                clean, diagnostic = adaptive_clean(engine, condition)
                cleans[geometry] = clean
                diagnostics.append({"geometry": list(geometry), **diagnostic})
            expected = engine.expected(condition, factor=4, clean=cleans[geometry])
            noise = int(
                np.random.SeedSequence([p["noise_seed"], namespace, index]).generate_state(1)[0]
            )
            observed = engine.observe(condition, factor=4, noise_seed=noise, expected=expected)
            prefix = f"products/{renderer}/{parent}-{name}"
            image = save_array(output, prefix + "-image.npy", observed["image"])
            variance = save_array(output, prefix + "-variance.npy", observed["variance"])
            psf = save_array(output, prefix + "-psf.npy", expected["psf"])
            maskpath = f"masks/{geometry[0]}x{geometry[1]}.npy"
            mask = {"path": maskpath, "sha256": sha256(output / maskpath)}
            observations.append(
                {
                    **record["lineage"],
                    "renderer": renderer,
                    "condition": name,
                    "shape": list(geometry),
                    "pixel_scale_arcsec": condition["pixel_scale_arcsec"],
                    "psf_family": condition["psf_family"],
                    "condition_parameters": condition,
                    "detector": {
                        "exposure_s": p["exposure_time"] * condition["exposure_multiplier"],
                        "read_noise_electrons": p["read_noise_electrons"]
                        * condition["read_noise_multiplier"],
                        "background_rate": p["background_rate"],
                        "noise_seed": noise,
                    },
                    "products": {
                        "image": image,
                        "variance": variance,
                        "psf": psf,
                        "invalid_mask": mask,
                        "coverage_mask": mask,
                    },
                }
            )
        return {
            "proposal": record["index"],
            "parent_system_id": parent,
            "observations": observations,
            "diagnostics": diagnostics,
            "failure": None,
        }
    except Exception as exc:
        message = str(exc)
        # Both current engines report explicit one-root counts in these forms.
        single_message = "found 1 roots" in message or "multiplicity 1" in message
        approved = single_message and len(independent_positions(p)) == 1
        return {
            "proposal": record["index"],
            "parent_system_id": parent,
            "observations": observations,
            "diagnostics": diagnostics,
            "failure": {
                "type": type(exc).__name__,
                "message": message,
                "approved_physical_single_image_rejection": bool(approved),
            },
        }


def phase(output: str | Path, renderer: str, workers: int) -> None:
    """Render one backend in proposal order with explicit restart protection."""
    output = Path(output)
    if not 1 <= workers <= 2:
        raise ValueError("Use one or two workers; phases must be sequential")
    verify_qualification()
    started = json.loads((output / "generation_started.json").read_text())
    for path, expected in started["generator_sha256"].items():
        if sha256(path) != expected:
            raise ValueError("Generator changed after prepare")
    if sha256(output / "truth_vault/proposals.json") != started["proposals_sha256"]:
        raise ValueError("Proposal vault changed")
    records = json.loads((output / "truth_vault/proposals.json").read_text())
    report = output / f"render-{renderer}.json"
    marker = output / f"render-{renderer}-started.json"
    if marker.exists() or report.exists():
        raise FileExistsError("Rendering phase already started; explicit recovery required")
    write(marker, {"workers": workers, "renderer": renderer, "generator_sha256": sha256(__file__)})
    results = []
    clock = time.perf_counter()
    tasks = [(str(output), renderer, r, started["design"]["noise_namespace"]) for r in records]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(render_one, tasks):
            results.append(result)
            if len(results) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "role": started["design"]["role"],
                            "renderer": renderer,
                            "computed": len(results),
                            "pool": len(records),
                            "failures": sum(r["failure"] is not None for r in results),
                            "seconds": time.perf_counter() - clock,
                        }
                    ),
                    flush=True,
                )
    write(
        report,
        {
            "results": results,
            "runtime_seconds": time.perf_counter() - clock,
            "renderer": renderer,
            "generator_sha256": sha256(__file__),
            "backend_versions": {
                package: version(package)
                for package in ("numpy", "scipy", "autolens", "autogalaxy")
                if renderer == "autolens"
            }
            if renderer == "autolens"
            else {package: version(package) for package in ("numpy", "scipy", "lenstronomy")},
        },
    )
    if any(
        r["failure"] and not r["failure"]["approved_physical_single_image_rejection"]
        for r in results
    ):
        raise ValueError("Numerical failures block paired release")


def assemble(output: str | Path) -> None:
    """Join agreed A/B parents and isolate truth from image-only products."""
    output = Path(output)
    from src.ml.artifacts import create_once

    q = verify_qualification()
    started = json.loads((output / "generation_started.json").read_text())
    for path, expected in started["generator_sha256"].items():
        if sha256(path) != expected:
            raise ValueError("Generator changed after prepare")
    proposals = json.loads((output / "truth_vault/proposals.json").read_text())
    if sha256(output / "truth_vault/proposals.json") != started["proposals_sha256"]:
        raise ValueError("Proposal vault changed after prepare")
    reports = {r: json.loads((output / f"render-{r}.json").read_text()) for r in RENDERERS}
    for renderer, report in reports.items():
        indices = [row["proposal"] for row in report["results"]]
        if (
            report["renderer"] != renderer
            or report["generator_sha256"] != sha256(__file__)
            or sorted(indices) != list(range(len(proposals)))
        ):
            raise ValueError("Renderer report identity or proposal completeness differs")
    by_engine = {
        r: {row["proposal"]: row for row in report["results"]} for r, report in reports.items()
    }
    accepted = []
    audit = []
    for proposal in proposals:
        index = proposal["index"]
        pair = [by_engine[r][index] for r in RENDERERS]
        failures = [row["failure"] for row in pair]
        if any(f is not None for f in failures):
            if not all(f and f["approved_physical_single_image_rejection"] for f in failures):
                raise ValueError("Unresolved numerical failure or A/B rejection disagreement")
        else:
            accepted.append(index)
        audit.append(
            {
                "proposal": index,
                **proposal["lineage"],
                "renderer_failures": dict(zip(RENDERERS, failures)),
            }
        )
    count = started["requested_parents"]
    if len(accepted) < count:
        raise ValueError("Insufficient agreed renderable parents in the declared pool")
    selected = accepted[:count]
    selected_set = set(selected)
    parent_ids = [proposals[i]["lineage"]["parent_system_id"] for i in selected]
    truth = {
        proposals[i]["lineage"]["parent_system_id"]: proposals[i]["parameters"]["theta_E_arcsec"]
        for i in selected
    }
    create_once(output / "truth_vault/targets.json", truth)
    observations = [
        row for r in RENDERERS for i in selected for row in by_engine[r][i]["observations"]
    ]
    audit = [{**row, "selected": row["proposal"] in selected_set} for row in audit]
    create_once(output / "proposal_audit.json", audit)
    create_once(
        output / "manifest.json",
        {
            "schema": "paired_evaluation_release_v1",
            "role": started["design"]["role"],
            "method_freeze_sha256": started["method_freeze_sha256"],
            "evaluation_freeze_sha256": started["evaluation_freeze_sha256"],
            "qualification": {"path": QUALIFICATION.as_posix(), "sha256": sha256(QUALIFICATION)},
            "rendering_factor": q["rendering_factor"],
            "parent_ids": parent_ids,
            "condition_order": list(CONDITIONS),
            "renderer_order": list(RENDERERS),
            "observations": observations,
            "truth_sha256": sha256(output / "truth_vault/targets.json"),
            "image_unit": "electrons_per_second",
            "variance_unit": "electrons_per_second_squared",
            "mask_semantics": "true_is_excluded",
            "proposal_count": len(proposals),
            "accepted_proposals": len(accepted),
            "physical_single_image_rejections": len(proposals) - len(accepted),
            "numerical_failures": 0,
            "noise_coupling": "same per-parent/condition seed in both engines; backend versions recorded separately",
            "generator_sha256": started["generator_sha256"],
            "proposal_audit_sha256": sha256(output / "proposal_audit.json"),
        },
    )
