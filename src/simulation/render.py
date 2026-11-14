"""Versioned A/B rendering for the new robustness benchmark.

Both engines use project-owned quadrature, point deposition, convolution and
detector sampling. Mass, ray tracing, source profiles and point solvers come
from the selected engine. The alternate branch never imports lenstronomy.
Coordinates are x,y arcsec; light radii are circularized; values integrate to
electrons/s per native pixel. This is not native AutoLens SimulatorImaging.
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
from scipy.optimize import root
from scipy.signal import fftconvolve

from src.simulation import backends as external
from src.simulation.observation import psf_kernel, sample_detector


def native_square_transform(condition: dict[str, Any]) -> np.ndarray | None:
    """Validate an optional rotated/reflected square-pixel angular transform.

    Default conditions retain their original arithmetic. Sheared/anisotropic
    pixels need a separately validated PSF integration and are rejected here.
    """
    if "pixel_transform_arcsec" not in condition:
        return None
    matrix = np.asarray(condition["pixel_transform_arcsec"], dtype=float)
    scale = float(condition["pixel_scale_arcsec"])
    if (
        matrix.shape != (2, 2)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix.T @ matrix, np.eye(2) * scale**2, rtol=1e-8, atol=1e-12)
    ):
        raise ValueError("Renderer requires orthogonal square native pixels")
    return matrix


class SceneEngine:
    """Cache physical profiles and point roots once per parent, not per view."""

    def __init__(self, parameters: dict[str, Any], renderer: str) -> None:
        self.parameters = dict(parameters)
        self.renderer = renderer
        parameters = self.parameters
        if renderer == "autolens":
            self.mass = external.mass_profile(parameters)
            self.source = external._light(
                parameters, "source", (parameters["source_x_arcsec"], parameters["source_y_arcsec"])
            )
            self.foreground = external._light(
                parameters,
                "lens_light",
                (parameters["lens_center_x_arcsec"], parameters["lens_center_y_arcsec"]),
            )
        elif renderer == "lenstronomy":
            from lenstronomy.LensModel.lens_model import LensModel
            from lenstronomy.LensModel.Solver.lens_equation_solver import (
                LensEquationSolver,
            )
            from lenstronomy.LightModel.light_model import LightModel

            self.mass = LensModel(["SIE"])
            self.source = self.foreground = LightModel(["SERSIC_ELLIPSE"])

            def ell(q, phi):
                e = (1 - q) / (1 + q)
                return e * np.cos(2 * phi), e * np.sin(2 * phi)

            e1, e2 = ell(parameters["q"], parameters["phi_rad"])
            self.mass_kwargs = [
                {
                    "theta_E": parameters["theta_E_arcsec"],
                    "e1": e1,
                    "e2": e2,
                    "center_x": parameters["lens_center_x_arcsec"],
                    "center_y": parameters["lens_center_y_arcsec"],
                }
            ]
            self.light_kwargs = {}
            for prefix, center in (
                ("source", (parameters["source_x_arcsec"], parameters["source_y_arcsec"])),
                (
                    "lens_light",
                    (parameters["lens_center_x_arcsec"], parameters["lens_center_y_arcsec"]),
                ),
            ):
                e1, e2 = ell(parameters[f"{prefix}_q"], parameters[f"{prefix}_phi_rad"])
                self.light_kwargs[prefix] = [
                    {
                        "amp": parameters[f"{prefix}_amp"],
                        "R_sersic": parameters[f"{prefix}_re_arcsec"],
                        "n_sersic": parameters[f"{prefix}_n"],
                        "e1": e1,
                        "e2": e2,
                        "center_x": center[0],
                        "center_y": center[1],
                    }
                ]
            self.solver = LensEquationSolver(self.mass)
        else:
            raise ValueError("Unknown physical renderer")
        self._points = None

    def alpha(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if self.renderer == "autolens":
            return external.deflections_xy(self.mass, xy)
        return np.column_stack(self.mass.alpha(xy[:, 0], xy[:, 1], self.mass_kwargs))

    def light(self, prefix: str, xy: np.ndarray) -> np.ndarray:
        """Use each engine's geometry/radial evaluator, with identical radius floor."""
        if self.renderer == "lenstronomy":
            return np.asarray(
                self.source.surface_brightness(xy[:, 0], xy[:, 1], self.light_kwargs[prefix])
            )
        profile = self.source if prefix == "source" else self.foreground
        grid = external.backend().Grid2DIrregular(values=xy[:, ::-1].copy())
        radii = np.asarray(profile._eccentric_radii_grid_from_cartesian(grid=grid))
        # Pin this private geometry API through the numerical qualification and
        # exact dependency version. Brightness remains evaluated by AutoLens.
        radii = np.maximum(radii, 1e-4)
        values = np.asarray(
            profile.image_2d_via_radii_from(
                grid_radii=external.backend().ArrayIrregular(values=radii)
            )
        )
        return np.where(radii / self.parameters[f"{prefix}_re_arcsec"] <= 1000.0, values, 0.0)

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        if self.renderer != "lenstronomy":
            if self._points is None:
                positions, _ = external.point_images(self.mass, self.parameters, radius_limit=20.0)
                self._points = positions, fourth_order_amplitudes(self, positions)
            return self._points
        if self._points is None:
            parameters = self.parameters
            source = np.array([parameters["source_x_arcsec"], parameters["source_y_arcsec"]])
            center = np.array(
                [parameters["lens_center_x_arcsec"], parameters["lens_center_y_arcsec"]]
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                x, y = self.solver.image_position_from_source(
                    *source, self.mass_kwargs, solver="analytical", arrival_time_sort=False
                )
            starts = list(np.column_stack((x, y)))
            angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            for radius in (0.005, 0.02, 0.1, 0.35, 0.65, 0.9, 1.1, 1.4, 1.8):
                starts.extend(
                    center
                    + radius
                    * parameters["theta_E_arcsec"]
                    * np.column_stack((np.cos(angles), np.sin(angles)))
                )
            positions = []

            def residual(x):
                return x - self.alpha(x[None])[0] - source

            for candidate in starts:
                solved = root(residual, candidate, options={"xtol": 1e-11, "maxfev": 100})
                position = solved.x
                radius = np.linalg.norm(position - center)
                if (
                    not np.isfinite(position).all()
                    or radius < 1e-7
                    or radius > 4 * parameters["theta_E_arcsec"]
                    or np.linalg.norm(residual(position)) > 1e-8
                ):
                    continue
                if not any(np.linalg.norm(position - other) < 1e-6 for other in positions):
                    positions.append(position)
            if len(positions) not in (2, 4):
                raise ValueError(f"Low-memory SIE root multiplicity {len(positions)}")
            positions = np.asarray(positions)
            self._points = positions, fourth_order_amplitudes(self, positions)
        return self._points

    def clean(self, condition: dict[str, Any], *, factor: int, tile_rows: int = 16) -> np.ndarray:
        """Midpoint quadrature, tiled to bound memory at fine integration factors."""
        h, w = condition["shape"]
        tile_rows = max(1, min(tile_rows, 262144 // (w * factor * factor)))
        scale = float(condition["pixel_scale_arcsec"])
        transform = native_square_transform(condition)
        if (
            type(factor) is not int
            or factor < 1
            or min(h, w, tile_rows) < 1
            or not np.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("Invalid integration geometry")
        x = ((np.arange(w * factor) + 0.5) / factor - w / 2) * scale
        clean = np.empty((h, w), dtype=float)
        for start in range(0, h, tile_rows):
            stop = min(h, start + tile_rows)
            y = ((np.arange(start * factor, stop * factor) + 0.5) / factor - h / 2) * scale
            xx, yy = np.meshgrid(x, y)
            xy = np.column_stack((xx.ravel(), yy.ravel()))
            if transform is not None:
                xy = (xy / scale) @ transform.T
            source_xy = xy - self.alpha(xy)
            flux = self.light("source", source_xy) + self.light("lens_light", xy)
            clean[start:stop] = (
                flux.reshape(stop - start, factor, w, factor).mean(axis=(1, 3)) * scale**2
            )
        if not np.isfinite(clean).all() or np.any(clean < 0):
            raise ValueError("Invalid integrated light")
        return clean

    def expected(
        self, condition: dict[str, Any], *, factor: int, clean: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        parameters = self.parameters
        clean = self.clean(condition, factor=factor) if clean is None else clean
        shape = tuple(condition["shape"])
        scale = condition["pixel_scale_arcsec"]
        point_scene = np.zeros(shape, dtype=float)
        positions, amplitudes = self.points()
        transform = native_square_transform(condition)
        pixel_positions = positions / scale
        if transform is not None:
            pixel_positions = np.linalg.solve(transform, positions.T).T
        for (x, y), amplitude in zip(pixel_positions, amplitudes):
            px, py = x + (shape[1] - 1) / 2, y + (shape[0] - 1) / 2
            ix, iy = int(np.floor(px)), int(np.floor(py))
            if min(ix, iy) < 0 or ix + 1 >= shape[1] or iy + 1 >= shape[0]:
                raise ValueError("Point image outside the common native field")
            dx, dy = px - ix, py - iy
            point_scene[iy : iy + 2, ix : ix + 2] += amplitude * np.outer(
                [1 - dy, dy], [1 - dx, dx]
            )
        kernel = psf_kernel(
            parameters["psf_fwhm_arcsec"] * condition.get("psf_fwhm_multiplier", 1.0),
            scale,
            condition["psf_family"],
            condition.get("moffat_beta", 3.5),
        )
        expected = np.maximum(fftconvolve(clean + point_scene, kernel, mode="same"), 0)
        return {
            "clean": clean,
            "expected": expected,
            "psf": kernel,
            "positions_xy": positions,
            "point_amplitudes": amplitudes,
        }

    def observe(
        self,
        condition: dict[str, Any],
        *,
        factor: int,
        noise_seed: int,
        expected: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Fresh independent detector draw; caller supplies its isolated seed."""
        rendered = self.expected(condition, factor=factor) if expected is None else expected
        parameters = self.parameters
        raw, variance = sample_detector(
            rendered["expected"],
            parameters["exposure_time"] * condition["exposure_multiplier"],
            parameters["read_noise_electrons"] * condition["read_noise_multiplier"],
            parameters["background_rate"],
            noise_seed,
        )
        return {
            **rendered,
            "image": raw.astype(np.float32),
            "variance": variance.astype(np.float32),
        }


def fourth_order_amplitudes(
    engine: SceneEngine, positions: np.ndarray, *, step: float = 0.0001
) -> np.ndarray:
    """Estimate absolute point magnifications with the fourth-order deflection Jacobian."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Magnification step must be finite and positive")
    amplitudes = []
    for position in positions:
        derivatives = []
        for axis in range(2):
            offset = np.eye(2)[axis] * step
            grid = np.array(
                [position - 2 * offset, position - offset, position + offset, position + 2 * offset]
            )
            alpha = engine.alpha(grid)
            derivatives.append((alpha[0] - 8 * alpha[1] + 8 * alpha[2] - alpha[3]) / (12 * step))
        determinant = np.linalg.det(np.eye(2) - np.column_stack(derivatives))
        amplitudes.append(abs(1 / determinant) * engine.parameters["point_source_amp"])
    amplitudes = np.asarray(amplitudes)
    if not np.isfinite(amplitudes).all() or np.any(amplitudes <= 0):
        raise ValueError("Invalid fourth-order magnification")
    return amplitudes


def pixel_fluxes(
    engine: SceneEngine,
    condition: dict[str, Any],
    pixel_indices: np.ndarray,
    *,
    factor: int,
    chunk_points: int = 262144,
) -> np.ndarray:
    """Integrate selected native pixels using midpoint subpixels and bounded chunks."""
    h, w = condition["shape"]
    scale = float(condition["pixel_scale_arcsec"])
    transform = native_square_transform(condition)
    indices = np.asarray(pixel_indices, dtype=int)
    if (
        indices.ndim != 1
        or np.any(indices < 0)
        or np.any(indices >= h * w)
        or type(factor) is not int
        or factor < 1
    ):
        raise ValueError("Invalid refinement pixels/factor")
    yy, xx = np.divmod(indices, w)
    offset = (np.arange(factor) + 0.5) / factor - 0.5
    ox, oy = np.meshgrid(offset, offset)
    offsets = np.column_stack((ox.ravel(), oy.ravel()))
    output = np.empty(len(indices), dtype=float)
    chunk_pixels = max(1, chunk_points // (factor * factor))
    for start in range(0, len(indices), chunk_pixels):
        stop = min(start + chunk_pixels, len(indices))
        centers = np.column_stack((xx[start:stop] - (w - 1) / 2, yy[start:stop] - (h - 1) / 2))
        xy = ((centers[:, None] + offsets[None]) * scale).reshape(-1, 2)
        if transform is not None:
            xy = (xy / scale) @ transform.T
        beta = xy - engine.alpha(xy)
        brightness = engine.light("source", beta) + engine.light("lens_light", xy)
        output[start:stop] = brightness.reshape(stop - start, -1).mean(1) * scale**2
    return output


def adaptive_clean(
    engine: SceneEngine,
    condition: dict[str, Any],
    *,
    relative_tolerance: float = 0.0001,
    maximum_factor: int = 512,
    progress: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine highest estimated-error pixels until the global L1 estimate is small.

    Start with factors 2 and 4 everywhere. At each step refine pixels comprising
    90% of current estimated error, retaining conservative stale estimates on
    untouched pixels. Zero-change pixels are not repeatedly refined. Preserve
    unresolved error at the maximum factor and fail explicitly if it dominates.
    """
    if (
        not np.isfinite(relative_tolerance)
        or relative_tolerance <= 0
        or type(maximum_factor) is not int
        or maximum_factor < 8
        or maximum_factor & (maximum_factor - 1)
    ):
        raise ValueError("Invalid adaptive quadrature settings")
    low = engine.clean(condition, factor=2)
    high = engine.clean(condition, factor=4)
    values = high.ravel().copy()
    error = abs(high - low).ravel()
    factors = np.full(values.shape, 4, dtype=np.int16)
    rounds = 0
    while error.sum() > relative_tolerance * max(values.sum(), 1e-100):
        if error[factors == maximum_factor].sum() > relative_tolerance * max(values.sum(), 1e-100):
            break
        eligible = np.flatnonzero((factors < maximum_factor) & (error > 0))
        if not len(eligible):
            break
        ranked = eligible[np.argsort(-error[eligible])]
        cumulative = np.cumsum(error[ranked])
        cutoff = min(len(ranked), int(np.searchsorted(cumulative, 0.9 * error.sum())) + 1)
        selected = ranked[:cutoff]
        starting_factors = factors[selected].copy()
        for factor in np.unique(starting_factors):
            pixels = selected[starting_factors == factor]
            refined = pixel_fluxes(engine, condition, pixels, factor=int(factor) * 2)
            error[pixels] = abs(refined - values[pixels])
            values[pixels] = refined
            factors[pixels] *= 2
        rounds += 1
        if progress is not None:
            progress(
                {
                    "round": rounds,
                    "relative_error_estimate": float(error.sum() / values.sum()),
                    "selected_pixels": len(selected),
                    "highest_factor": int(factors.max()),
                }
            )
        if rounds > 64:
            raise RuntimeError("Adaptive quadrature failed to settle")
    relative = float(error.sum() / max(values.sum(), 1e-100))
    diagnostics = {
        "relative_successive_grid_l1_estimate": relative,
        "relative_tolerance": relative_tolerance,
        "maximum_factor": maximum_factor,
        "rounds": rounds,
        "pixels_by_factor": {str(f): int(np.sum(factors == f)) for f in np.unique(factors)},
        "passed_internal_estimate": bool(relative <= relative_tolerance),
    }
    if not diagnostics["passed_internal_estimate"]:
        raise ValueError(f"Unresolved adaptive integration: {diagnostics}")
    return values.reshape(high.shape), diagnostics
