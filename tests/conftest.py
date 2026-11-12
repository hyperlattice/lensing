"""Small explicit observation-bank fixtures; never production data."""

import json

import numpy as np

from src.simulation.observation import CONDITIONS, RENDERERS, evaluation_condition


def fixture_bank(root, role, seed):
    root.mkdir()
    records, targets = [], {}
    for i in range(2):
        parent = f"{role}-parent-{i}"
        products = {}
        keys = [
            "base",
            "pixel-scale-004",
            "pixel-scale-006",
            "gaussian-fwhm-075",
            "gaussian-fwhm-125",
        ]
        for j, k in enumerate(keys):
            shape = (
                (40, 40)
                if k == "pixel-scale-004"
                else (27, 27)
                if k == "pixel-scale-006"
                else (32, 32)
            )
            a = np.full(shape, 0.3 + i * 0.2, dtype=np.float32)
            p = root / f"{i}-{k}.npy"
            np.save(p, a)
            products[k] = {"path": p.name}
        p = {
            "parent_system_id": parent,
            "role": role,
            "parameter_seed": seed + i,
            "noise_seed": seed + 100 + i,
            "ancestry_ids": [],
            "latent_id": f"{role}-{i}",
        }
        if role == "target_adaptation":
            p["images"] = [products["base"]]
        else:
            p.update(
                expected=products,
                detector={
                    "exposure_s": 100.0,
                    "read_noise_electrons": 2.0,
                    "background_rate": 0.03,
                },
                historical_condition={
                    "pixel_scale_arcsec": 0.04,
                    "read_noise_multiplier": 2.0,
                    "exposure_multiplier": 1.0,
                },
            )
            targets[parent] = 0.8 + i * 0.4
        records.append(p)
    label = root / "targets.json"
    label.write_text(json.dumps(targets))
    m = {
        "schema": "observation_bank_v1",
        "stage": "unit_fixture",
        "role": role,
        "parents": records,
        "exclusion_ledger": [],
        "condition_access": "gaussian_only",
        "targets": None if role == "target_adaptation" else {"path": label.name},
    }
    path = root / "bank.json"
    path.write_text(json.dumps(m))
    return path


def paired_release(root):
    def save(name, a):
        path = root / name
        np.save(path, a)
        return {"path": name}

    products = {}
    psf = save("psf.npy", np.ones((1, 1), np.float32))
    for shape in ((256, 256), (320, 320), (213, 213)):
        products[shape] = {
            "image": save(f"image-{shape[0]}.npy", np.full(shape, 0.1, np.float32)),
            "variance": save(f"variance-{shape[0]}.npy", np.full(shape, 0.01, np.float32)),
            "psf": psf,
            "invalid_mask": save(f"mask-{shape[0]}.npy", np.zeros(shape, bool)),
            "coverage_mask": save(f"coverage-{shape[0]}.npy", np.zeros(shape, bool)),
        }
    qualification = root / "qualification.json"
    qualification.write_text(json.dumps({"passed": True, "rendering_factor": "unit-only"}))
    rows = []
    for parent in range(2):
        for renderer in RENDERERS:
            for name in CONDITIONS:
                c = evaluation_condition(name)
                rows.append(
                    {
                        "parent_system_id": f"p{parent}",
                        "role": "final",
                        "parameter_seed": parent + 10,
                        "noise_seed": parent + 20,
                        "ancestry_ids": [],
                        "latent_id": f"unit-{parent}",
                        "renderer": renderer,
                        "condition": name,
                        "shape": c["shape"],
                        "pixel_scale_arcsec": c["pixel_scale_arcsec"],
                        "psf_family": c["psf_family"],
                        "condition_parameters": c,
                        "detector": {
                            "exposure_s": 100.0,
                            "read_noise_electrons": 2.0,
                            "background_rate": 0.03,
                            "noise_seed": 100 + parent,
                        },
                        "products": products[tuple(c["shape"])],
                    }
                )
    manifest = {
        "schema": "paired_evaluation_release_v1",
        "role": "final",
        "qualification": {"path": str(qualification)},
        "rendering_factor": "unit-only",
        "parent_ids": ["p0", "p1"],
        "condition_order": list(CONDITIONS),
        "renderer_order": list(RENDERERS),
        "observations": rows,
        "image_unit": "electrons_per_second",
        "variance_unit": "electrons_per_second_squared",
        "mask_semantics": "true_is_excluded",
        "proposal_count": 2,
        "accepted_proposals": 2,
        "physical_single_image_rejections": 0,
        "numerical_failures": 0,
        "noise_coupling": "fixture",
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest
