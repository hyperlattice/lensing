"""Source-only, fixed-budget augmentation allocation; no renderer dependencies.

These policies specify observation conditions, not pixel resampling. Rendering
and fresh detector draws must be done by a separately qualified data pipeline.
The failure-weighted policy is a candidate, not an established novel method.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

CELLS = ("pixel-scale-004", "pixel-scale-006", "exposure-half", "read-noise-double",
         "gaussian-fwhm-075", "gaussian-fwhm-125", "fresh-noise")


def condition_for(cell, *, shape=(256, 256), pixel_scale_arcsec=.05):
    """Preserve angular field to nearest native pixel; change one factor only."""
    if cell not in CELLS or len(shape) != 2 or any(int(n) != n or n <= 0 for n in shape):
        raise ValueError("Invalid transformation or canonical shape")
    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
        raise ValueError("Canonical scale must be positive and finite")
    scale = {"pixel-scale-004": .04, "pixel-scale-006": .06}.get(cell, pixel_scale_arcsec)
    native = [max(1, int(np.floor(n * pixel_scale_arcsec / scale + .5))) for n in shape]
    return {"name": cell, "shape": native, "pixel_scale_arcsec": float(scale),
            "psf_family": "gaussian", "psf_fwhm_multiplier":
            {"gaussian-fwhm-075": .75, "gaussian-fwhm-125": 1.25}.get(cell, 1.),
            "exposure_multiplier": .5 if cell == "exposure-half" else 1.,
            "read_noise_multiplier": 2. if cell == "read-noise-double" else 1.}


def failure_weights(canonical_errors, transformed_errors, *, role, epsilon=.25):
    """Errors have [seed,parent] and [seed,parent,cell] paired axes, in arcsec.

    Caller must use the same identities in both arrays and save their hashes.
    Negative paired degradation is clipped only after averaging across parents
    and pilot seeds. This does not accept target-domain or final-role scores.
    """
    if role != "source_development":
        raise ValueError("Failure weighting requires source-development errors")
    a, b = np.asarray(canonical_errors, dtype=float), np.asarray(transformed_errors, dtype=float)
    if (a.ndim != 2 or min(a.shape) < 1 or b.shape != (*a.shape, len(CELLS)) or
        not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0)):
        raise ValueError("Expected paired nonnegative finite seed/parent/cell errors")
    if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("Exploration fraction must be in (0,1]")
    degradation = (b - a[..., None]).mean(axis=(0, 1))
    w = np.maximum(degradation, 0)
    p = np.full(len(CELLS), 1 / len(CELLS)) if w.sum() == 0 else epsilon / len(CELLS) + (1-epsilon)*w/w.sum()
    return {"cells": list(CELLS), "degradation_arcsec": degradation.tolist(),
            "probabilities": p.tolist(), "epsilon": float(epsilon), "role": role}


def paired_policy_schedule(parents, probabilities, *, updates, batch_size, seed):
    """Full batches share one native geometry; parent draws do not depend on policy.

    Independent RNG streams for parents, cells and detector seeds mean changing
    weights cannot change parent exposure. A cell per batch avoids resizing or
    padding native geometries. Draws are reproducible with replacement.
    """
    if (not parents or len(set(parents)) != len(parents) or any(not isinstance(p, str) or not p for p in parents) or
        any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (updates, batch_size))):
        raise ValueError("Expected unique parent IDs and positive integer schedule sizes")
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(CELLS),) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Invalid cell probabilities")
    streams = np.random.SeedSequence(seed).spawn(3)
    parent_rng, cell_rng, noise_rng = [np.random.default_rng(s) for s in streams]
    records = []
    for _ in range(updates):
        indices = parent_rng.integers(len(parents), size=batch_size)
        records.append({"parents": [parents[i] for i in indices],
                        "cell": CELLS[int(cell_rng.choice(len(CELLS), p=p))],
                        "noise_seeds": [int(n) for n in noise_rng.integers(0, 2**63, size=batch_size, dtype=np.int64)]})
    return deepcopy(records)
