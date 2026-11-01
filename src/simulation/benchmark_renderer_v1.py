"""Versioned A/B rendering for the new robustness benchmark.

Both engines use project-owned quadrature, point deposition, convolution and
detector sampling. Mass, ray tracing, source profiles and point solvers come
from the selected engine. The alternate branch never imports lenstronomy.
Coordinates are x,y arcsec; light radii are circularized; values integrate to
electrons/s per native pixel. This is not native AutoLens SimulatorImaging.
"""
from __future__ import annotations

import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
from scipy.signal import fftconvolve

from src.simulation import autolens_backend_v2 as external


class SceneEngine:
    """Cache physical profiles and point roots once per parent, not per view."""

    def __init__(self, parameters, renderer):
        self.p = dict(parameters)
        self.renderer = renderer
        p = self.p
        if renderer == "autolens":
            self.mass = external.mass_profile(p)
            self.source = external._light(p, "source", (p["source_x_arcsec"], p["source_y_arcsec"]))
            self.foreground = external._light(p, "lens_light", (p["lens_center_x_arcsec"], p["lens_center_y_arcsec"]))
        elif renderer == "lenstronomy":
            from lenstronomy.LensModel.lens_model import LensModel
            from lenstronomy.LightModel.light_model import LightModel
            from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
            self.mass = LensModel(["SIE"])
            self.source = self.foreground = LightModel(["SERSIC_ELLIPSE"])
            def ell(q, phi):
                e = (1-q)/(1+q)
                return e*np.cos(2*phi), e*np.sin(2*phi)
            e1, e2 = ell(p["q"], p["phi_rad"])
            self.mass_kwargs = [{"theta_E": p["theta_E_arcsec"], "e1": e1, "e2": e2,
                                 "center_x": p["lens_center_x_arcsec"], "center_y": p["lens_center_y_arcsec"]}]
            self.light_kwargs = {}
            for prefix, center in (("source", (p["source_x_arcsec"], p["source_y_arcsec"])),
                                   ("lens_light", (p["lens_center_x_arcsec"], p["lens_center_y_arcsec"]))):
                e1, e2 = ell(p[f"{prefix}_q"], p[f"{prefix}_phi_rad"])
                self.light_kwargs[prefix] = [{"amp": p[f"{prefix}_amp"], "R_sersic": p[f"{prefix}_re_arcsec"],
                    "n_sersic": p[f"{prefix}_n"], "e1": e1, "e2": e2, "center_x": center[0], "center_y": center[1]}]
            self.solver = LensEquationSolver(self.mass)
        else:
            raise ValueError("Unknown physical renderer")
        self._points = None

    def alpha(self, xy):
        xy = np.asarray(xy, dtype=float)
        if self.renderer == "autolens":
            return external.deflections_xy(self.mass, xy)
        return np.column_stack(self.mass.alpha(xy[:, 0], xy[:, 1], self.mass_kwargs))

    def light(self, prefix, xy):
        """Use each engine's geometry/radial evaluator, with identical radius floor."""
        if self.renderer == "lenstronomy":
            return np.asarray(self.source.surface_brightness(xy[:, 0], xy[:, 1], self.light_kwargs[prefix]))
        profile = self.source if prefix == "source" else self.foreground
        grid = external.backend().Grid2DIrregular(values=xy[:, ::-1].copy())
        radii = np.asarray(profile._eccentric_radii_grid_from_cartesian(grid=grid))
        # Pin this private geometry API through the numerical qualification and
        # exact dependency version. Brightness remains evaluated by AutoLens.
        radii = np.maximum(radii, 1e-4)
        values = np.asarray(profile.image_2d_via_radii_from(
            grid_radii=external.backend().ArrayIrregular(values=radii)))
        return np.where(radii/self.p[f"{prefix}_re_arcsec"] <= 1000., values, 0.)

    def points(self):
        if self._points is None:
            p = self.p
            if self.renderer == "autolens":
                positions, amplitudes = external.point_images(self.mass, p, radius_limit=20.)
            else:
                x, y = self.solver.image_position_from_source(p["source_x_arcsec"], p["source_y_arcsec"],
                    self.mass_kwargs, min_distance=.005, search_window=12.8, precision_limit=1e-10,
                    num_iter_max=200)
                positions = np.column_stack((x, y))
                if len(x) not in (2, 4):
                    raise ValueError(f"Lenstronomy root multiplicity {len(x)}")
                amplitudes = abs(self.mass.magnification(x, y, self.mass_kwargs))*p["point_source_amp"]
            if not np.isfinite(positions).all() or not np.isfinite(amplitudes).all() or np.any(amplitudes <= 0):
                raise ValueError("Invalid point positions or magnifications")
            self._points = (positions, amplitudes)
        return self._points

    def clean(self, condition, *, factor, tile_rows=16):
        """Midpoint quadrature, tiled to bound memory at fine integration factors."""
        h, w = condition["shape"]
        scale = float(condition["pixel_scale_arcsec"])
        if type(factor) is not int or factor < 1 or min(h, w, tile_rows) < 1 or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid integration geometry")
        x = ((np.arange(w*factor)+.5)/factor - w/2)*scale
        clean = np.empty((h, w), dtype=float)
        for start in range(0, h, tile_rows):
            stop = min(h, start+tile_rows)
            y = ((np.arange(start*factor, stop*factor)+.5)/factor - h/2)*scale
            xx, yy = np.meshgrid(x, y)
            xy = np.column_stack((xx.ravel(), yy.ravel()))
            source_xy = xy-self.alpha(xy)
            flux = self.light("source", source_xy) + self.light("lens_light", xy)
            clean[start:stop] = flux.reshape(stop-start, factor, w, factor).mean(axis=(1, 3))*scale**2
        if not np.isfinite(clean).all() or np.any(clean < 0):
            raise ValueError("Invalid integrated light")
        return clean

    def expected(self, condition, *, factor, clean=None):
        p = self.p
        clean = self.clean(condition, factor=factor) if clean is None else clean
        shape = tuple(condition["shape"])
        scale = condition["pixel_scale_arcsec"]
        point_scene = np.zeros(shape, dtype=float)
        positions, amplitudes = self.points()
        for (x, y), amplitude in zip(positions, amplitudes):
            px, py = x/scale+(shape[1]-1)/2, y/scale+(shape[0]-1)/2
            ix, iy = int(np.floor(px)), int(np.floor(py))
            if min(ix, iy) < 0 or ix+1 >= shape[1] or iy+1 >= shape[0]:
                raise ValueError("Point image outside the common native field")
            dx, dy = px-ix, py-iy
            point_scene[iy:iy+2, ix:ix+2] += amplitude*np.outer([1-dy, dy], [1-dx, dx])
        kernel = external.psf_kernel(p["psf_fwhm_arcsec"]*condition.get("psf_fwhm_multiplier", 1.),
                                    scale, condition["psf_family"], condition.get("moffat_beta", 3.5))
        expected = np.maximum(fftconvolve(clean+point_scene, kernel, mode="same"), 0)
        return {"clean": clean, "expected": expected, "psf": kernel,
                "positions_xy": positions, "point_amplitudes": amplitudes}

    def observe(self, condition, *, factor, noise_seed, expected=None):
        """Fresh independent detector draw; caller supplies its isolated seed."""
        rendered = self.expected(condition, factor=factor) if expected is None else expected
        p = self.p
        raw, variance = external.sample_detector(rendered["expected"],
            p["exposure_time"]*condition["exposure_multiplier"],
            p["read_noise_electrons"]*condition["read_noise_multiplier"], p["background_rate"], noise_seed)
        return {**rendered, "image": raw.astype(np.float32), "variance": variance.astype(np.float32)}
