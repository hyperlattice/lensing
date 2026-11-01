"""Adaptive native-pixel midpoint quadrature for sharply peaked lensed light.

No CNN outputs, scalar labels or physical-parameter thresholds select pixels.
Refinement is driven by changes in integrated brightness. Error estimates are
empirical successive-grid differences, not rigorous bounds; independent fine
uniform-grid controls are required before scientific qualification.
"""
from __future__ import annotations

import numpy as np


def pixel_fluxes(engine, condition, pixel_indices, *, factor, chunk_points=262144):
    h, w = condition["shape"]
    scale = float(condition["pixel_scale_arcsec"])
    indices = np.asarray(pixel_indices, dtype=int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= h*w) or type(factor) is not int or factor < 1:
        raise ValueError("Invalid refinement pixels/factor")
    yy, xx = np.divmod(indices, w)
    offset = (np.arange(factor)+.5)/factor-.5
    ox, oy = np.meshgrid(offset, offset)
    offsets = np.column_stack((ox.ravel(), oy.ravel()))
    output = np.empty(len(indices), dtype=float)
    chunk_pixels = max(1, chunk_points//(factor*factor))
    for start in range(0, len(indices), chunk_pixels):
        stop = min(start+chunk_pixels, len(indices))
        centers = np.column_stack((xx[start:stop]-(w-1)/2, yy[start:stop]-(h-1)/2))
        xy = ((centers[:, None]+offsets[None])*scale).reshape(-1, 2)
        beta = xy-engine.alpha(xy)
        brightness = engine.light("source", beta)+engine.light("lens_light", xy)
        output[start:stop] = brightness.reshape(stop-start, -1).mean(1)*scale**2
    return output


def adaptive_clean(engine, condition, *, relative_tolerance=1e-4, maximum_factor=512, progress=None):
    """Refine highest estimated-error pixels until the global L1 estimate is small.

    Start with factors 2 and 4 everywhere. At each step refine pixels comprising
    90% of current estimated error, retaining conservative stale estimates on
    untouched pixels. Zero-change pixels are not repeatedly refined. Preserve
    unresolved error at the maximum factor and fail explicitly if it dominates.
    """
    if (not np.isfinite(relative_tolerance) or relative_tolerance <= 0 or
        type(maximum_factor) is not int or maximum_factor < 8 or maximum_factor & (maximum_factor-1)):
        raise ValueError("Invalid adaptive quadrature settings")
    low = engine.clean(condition, factor=2)
    high = engine.clean(condition, factor=4)
    values = high.ravel().copy()
    error = abs(high-low).ravel()
    factors = np.full(values.shape, 4, dtype=np.int16)
    rounds = 0
    while error.sum() > relative_tolerance*max(values.sum(), 1e-100):
        if error[factors == maximum_factor].sum() > relative_tolerance*max(values.sum(), 1e-100):
            break
        eligible = np.flatnonzero((factors < maximum_factor) & (error > 0))
        if not len(eligible):
            break
        ranked = eligible[np.argsort(-error[eligible])]
        cumulative = np.cumsum(error[ranked])
        cutoff = min(len(ranked), int(np.searchsorted(cumulative, .9*error.sum()))+1)
        selected = ranked[:cutoff]
        starting_factors = factors[selected].copy()
        for factor in np.unique(starting_factors):
            pixels = selected[starting_factors == factor]
            refined = pixel_fluxes(engine, condition, pixels, factor=int(factor)*2)
            error[pixels] = abs(refined-values[pixels])
            values[pixels] = refined
            factors[pixels] *= 2
        rounds += 1
        if progress is not None:
            progress({"round": rounds, "relative_error_estimate": float(error.sum()/values.sum()),
                      "selected_pixels": len(selected), "highest_factor": int(factors.max())})
        if rounds > 64:
            raise RuntimeError("Adaptive quadrature failed to settle")
    relative = float(error.sum()/max(values.sum(), 1e-100))
    diagnostics = {"relative_successive_grid_l1_estimate": relative, "relative_tolerance": relative_tolerance,
                   "maximum_factor": maximum_factor, "rounds": rounds,
                   "pixels_by_factor": {str(f): int(np.sum(factors == f)) for f in np.unique(factors)},
                   "passed_internal_estimate": bool(relative <= relative_tolerance)}
    if not diagnostics["passed_internal_estimate"]:
        raise ValueError(f"Unresolved adaptive integration: {diagnostics}")
    return values.reshape(high.shape), diagnostics
