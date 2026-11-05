"""Optional PyAutoLens scene adapter with explicit project-coordinate boundaries.

This development adapter uses PyAutoLens mass and light profiles; it does not
call the frozen Lenstronomy renderer or reuse its source evaluator. Pixel
integration, root search, PSF convolution and detector sampling are explicit
here, so this is a PyAutoLens-based project pipeline, not SimulatorImaging.
"""
from __future__ import annotations

import importlib.metadata
import math

import numpy as np
from scipy.optimize import root
from scipy.signal import fftconvolve


def backend():
    try:
        import autolens as al
    except ImportError as exc:
        raise ImportError("Install requirements.txt in a Python 3.12 environment") from exc
    return al


def backend_version():
    return importlib.metadata.version("autolens")


def _ellipticity(q, phi):
    amplitude = (1.0 - float(q)) / (1.0 + float(q))
    # PyAutoLens uses (e_y, e_x), corresponding to project (e2, e1).
    return (amplitude * math.sin(2 * phi), amplitude * math.cos(2 * phi))


def mass_profile(parameters):
    al = backend()
    q = float(parameters["q"])
    if not 0 < q <= 1:
        raise ValueError("Mass axis ratio must be in (0,1]")
    # PyAutoGalaxy's PowerLawIntermediate documents the cross-code relation:
    # theta_Lenstronomy = 2*sqrt(q)/(1+q) * theta_PyAutoLens for slope=2.
    # Isothermal uses the latter convention, so translate before construction.
    backend_radius = float(parameters["theta_E_arcsec"]) * (1+q)/(2*np.sqrt(q))
    return al.mp.Isothermal(
        centre=(parameters.get("lens_center_y_arcsec", 0.0), parameters.get("lens_center_x_arcsec", 0.0)),
        ell_comps=_ellipticity(parameters["q"], parameters["phi_rad"]),
        einstein_radius=backend_radius,
    )


def deflections_xy(profile, xy):
    """Convert project (x,y) coordinates to backend (y,x), then convert back."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Coordinates must be finite (N,2) x,y pairs")
    grid = backend().Grid2DIrregular(values=xy[:, ::-1].copy())
    return np.asarray(profile.deflections_yx_2d_from(grid=grid))[:, ::-1]


def _light(parameters, prefix, center):
    return backend().lp.Sersic(
        centre=(center[1], center[0]),
        ell_comps=_ellipticity(parameters[f"{prefix}_q"], parameters[f"{prefix}_phi_rad"]),
        intensity=float(parameters[f"{prefix}_amp"]),
        effective_radius=float(parameters[f"{prefix}_re_arcsec"]),
        sersic_index=float(parameters[f"{prefix}_n"]),
    )


def psf_kernel(fwhm_arcsec, pixel_scale, family="gaussian", beta=3.5):
    """Return a unit-sum finite kernel; Moffat is an explicit new PSF family."""
    if not np.isfinite([fwhm_arcsec, pixel_scale]).all() or min(fwhm_arcsec, pixel_scale) <= 0:
        raise ValueError("PSF width and pixel scale must be positive")
    fwhm_pixels = float(fwhm_arcsec) / float(pixel_scale)
    if not np.isfinite(fwhm_pixels) or fwhm_pixels <= 0:
        raise ValueError("PSF width and pixel scale must be positive")
    size = max(5, int(np.ceil(5 * fwhm_pixels)))
    size += 1 - size % 2
    yy, xx = np.indices((size, size), dtype=float) - (size - 1) / 2
    rr = xx**2 + yy**2
    if family == "gaussian":
        sigma = fwhm_pixels / np.sqrt(8 * np.log(2))
        kernel = np.exp(-rr / (2 * sigma**2))
    elif family == "moffat":
        if not np.isfinite(beta) or beta <= 1:
            raise ValueError("Moffat beta must exceed one")
        alpha = fwhm_pixels / (2 * np.sqrt(2**(1 / beta) - 1))
        kernel = (1 + rr / alpha**2)**(-beta)
    else:
        raise ValueError("Unsupported PSF family")
    return kernel / kernel.sum()


def point_images(profile, parameters, radius_limit):
    """Solve roots with backend deflections, without copied Lenstronomy roots."""
    source = np.array([parameters["source_x_arcsec"], parameters["source_y_arcsec"]])
    center = np.array([parameters.get("lens_center_x_arcsec", 0), parameters.get("lens_center_y_arcsec", 0)])
    def residual(position):
        return position - deflections_xy(profile, np.asarray(position)[None, :])[0] - source
    solutions = []
    for radius in (0.5, 1.0, 1.5):
        for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            start = center + radius * parameters["theta_E_arcsec"] * np.array([np.cos(angle), np.sin(angle)])
            solved = root(residual, start)
            position = solved.x
            if not np.isfinite(position).all() or np.linalg.norm(residual(position)) > 1e-7 or np.linalg.norm(position-center) > radius_limit:
                continue
            if not any(np.linalg.norm(position-other) < 1e-5 for other in solutions):
                solutions.append(position)
    if len(solutions) not in (2, 4):
        raise ValueError(f"Backend root search found {len(solutions)} roots; expected 2 or 4")
    positions = np.asarray(sorted(solutions, key=lambda p: math.atan2(p[1]-center[1], p[0]-center[0])))
    amplitudes = []
    for position in positions:
        h = 1e-5
        jacobian = np.column_stack([(residual(position + np.eye(2)[axis] * h) - residual(position - np.eye(2)[axis] * h)) / (2*h) for axis in range(2)])
        amplitudes.append(abs(1 / np.linalg.det(jacobian)) * parameters["point_source_amp"])
    return positions, np.asarray(amplitudes)


def render_scene(parameters, condition, *, include_point_sources=True):
    """Render integrated pixel flux with smooth source and foreground light.

    Initial light-radius/amplitude mappings remain under development and must
    pass cross-backend image controls before this becomes an accuracy benchmark.
    """
    p = parameters
    shape = tuple(int(n) for n in condition["shape"])
    scale = float(condition["pixel_scale_arcsec"])
    ss = int(p.get("supersampling_factor", 2))
    if len(shape) != 2 or min(shape) <= 0 or scale <= 0 or ss < 1:
        raise ValueError("Invalid native image geometry")
    profile = mass_profile(p)
    y = ((np.arange(shape[0]*ss)+0.5)/ss-shape[0]/2)*scale
    x = ((np.arange(shape[1]*ss)+0.5)/ss-shape[1]/2)*scale
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    beta_xy = xy - deflections_xy(profile, xy)
    al = backend()
    source = _light(p, "source", (p["source_x_arcsec"],p["source_y_arcsec"]))
    foreground = _light(p, "lens_light", (p["lens_center_x_arcsec"],p["lens_center_y_arcsec"]))
    light = np.asarray(source.image_2d_from(grid=al.Grid2DIrregular(values=beta_xy[:, ::-1].copy())))
    light = light + np.asarray(foreground.image_2d_from(grid=al.Grid2DIrregular(values=xy[:, ::-1].copy())))
    clean = light.reshape(shape[0],ss,shape[1],ss).mean(axis=(1,3)) * scale**2
    positions, amplitudes = np.empty((0,2)), np.empty(0)
    point_scene = np.zeros(shape)
    if include_point_sources:
        positions, amplitudes = point_images(profile,p,radius_limit=max(shape)*scale)
        for position, amplitude in zip(positions,amplitudes):
            px, py = position / scale + np.array([(shape[1]-1)/2,(shape[0]-1)/2])
            ix, iy = int(np.floor(px)), int(np.floor(py))
            if ix < 0 or iy < 0 or ix+1 >= shape[1] or iy+1 >= shape[0]:
                raise ValueError("Point image falls outside the declared field")
            dx, dy = px-ix, py-iy
            point_scene[iy:iy+2,ix:ix+2] += amplitude*np.outer([1-dy,dy],[1-dx,dx])
    kernel = psf_kernel(p["psf_fwhm_arcsec"],scale,condition["psf_family"],condition.get("moffat_beta",3.5))
    expected = np.maximum(fftconvolve(clean+point_scene,kernel,mode="same"),0)
    exposure = float(p["exposure_time"])*float(condition["exposure_multiplier"])
    read = float(p["read_noise_electrons"])*float(condition["read_noise_multiplier"])
    if not np.isfinite(exposure) or exposure <= 0 or read < 0:
        raise ValueError("Invalid detector observation")
    mean_counts = (expected+float(p["background_rate"]))*exposure
    rng = np.random.default_rng(int(p["noise_seed"]))
    raw = (rng.poisson(mean_counts)+rng.normal(0,read,shape))/exposure
    variance = (mean_counts+read**2)/exposure**2
    return {"raw":raw.astype(np.float32), "expected":expected.astype(np.float32),
            "clean":clean.astype(np.float32), "expected_variance":variance.astype(np.float32),
            "psf":kernel.astype(np.float32), "positions_xy":positions, "point_amplitudes":amplitudes}
