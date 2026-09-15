"""Observation effects for the P60 simulation stage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.processing.psf import convolve_same, gaussian_psf

from src.simulation.models import (
    build_image_model,
    build_light_model,
    build_lens_model,
    build_point_source_model,
    build_psf,
    ellipticity_from_q_phi,
    sie_kwargs,
    shear_kwargs,
)


@dataclass(frozen=True)
class ObservationScene:
    """PSF-convolved scene and point-source diagnostics before detector noise."""

    clean_scene: np.ndarray
    expected_scene: np.ndarray
    lens_parameters: dict[str, float]
    shear_parameters: dict[str, float] | None
    source_parameters: dict[str, float]
    lens_light_parameters: dict[str, float]
    point_source_parameters: dict[str, float]
    point_positions_x_arcsec: tuple[float, ...]
    point_positions_y_arcsec: tuple[float, ...]
    point_amplitudes: tuple[float, ...]
    pixel_scale_arcsec: float
    supersampling_factor: int
    psf_fwhm_arcsec: float


@dataclass(frozen=True)
class DetectorObservation:
    """Expected and sampled detector products in rate units."""

    expected_electrons: np.ndarray
    observed_electrons: np.ndarray
    observed_rate: np.ndarray
    science_rate: np.ndarray
    variance_rate: np.ndarray
    background_rate: float
    exposure_time: float
    read_noise_electrons: float
    noise_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "background_rate": self.background_rate,
            "exposure_time": self.exposure_time,
            "read_noise_electrons": self.read_noise_electrons,
            "noise_seed": self.noise_seed,
            "variance_formula": "(expected_electrons + read_noise_electrons**2) / exposure_time**2",
            "image_unit": "electrons_per_second_after_background_subtraction",
            "poisson_draw_space": "nonnegative expected electrons before subtraction",
        }


def _light_kwargs(parameters: Mapping[str, Any], prefix: str, *, center_x: float, center_y: float) -> dict[str, float]:
    q = float(parameters[f"{prefix}_q"])
    phi = float(parameters[f"{prefix}_phi_rad"])
    e1, e2 = ellipticity_from_q_phi(q, phi)
    return {
        "amp": float(parameters[f"{prefix}_amp"]),
        "R_sersic": float(parameters[f"{prefix}_re_arcsec"]),
        "n_sersic": float(parameters[f"{prefix}_n"]),
        "e1": e1,
        "e2": e2,
        "center_x": float(center_x),
        "center_y": float(center_y),
    }


def _deposit_bilinear(
    image: np.ndarray,
    *,
    x_pixel: float,
    y_pixel: float,
    amplitude: float,
) -> None:
    """Deposit one sub-pixel point source deterministically into an image."""

    if not np.isfinite(x_pixel) or not np.isfinite(y_pixel) or not np.isfinite(amplitude):
        raise ValueError("Point-source coordinates and amplitude must be finite")
    height, width = image.shape
    if x_pixel < 0.0 or x_pixel > width - 1 or y_pixel < 0.0 or y_pixel > height - 1:
        raise ValueError("Point-source coordinate falls outside the rendered image")
    x0 = int(np.floor(x_pixel))
    y0 = int(np.floor(y_pixel))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    dx = x_pixel - x0
    dy = y_pixel - y0
    image[y0, x0] += amplitude * (1.0 - dx) * (1.0 - dy)
    image[y0, x1] += amplitude * dx * (1.0 - dy)
    image[y1, x0] += amplitude * (1.0 - dx) * dy
    image[y1, x1] += amplitude * dx * dy


def observation_psf(*, psf_fwhm_arcsec: float, pixel_scale_arcsec: float):
    """Return the repository Gaussian control PSF used for released images."""

    sigma_pixels = psf_fwhm_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0)) * pixel_scale_arcsec)
    size = max(5, int(np.ceil(10.0 * sigma_pixels)))
    if size % 2 == 0:
        size += 1
    return gaussian_psf(
        psf_id="observation-gaussian-control",
        sigma_pixels=sigma_pixels,
        size=size,
        pixel_scale=pixel_scale_arcsec,
    )


def render_observation_scene(
    parameters: Mapping[str, Any],
    *,
    shape: tuple[int, int] | None = None,
    pixel_scale_arcsec: float | None = None,
    psf_fwhm_arcsec: float = 0.1,
    supersampling_factor: int = 2,
) -> ObservationScene:
    """Render clean and PSF-convolved SIE/extended/point-source components."""

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    shape = tuple(parameters.get("shape", (256, 256)) if shape is None else shape)
    pixel_scale = float(
        parameters.get("pixel_scale_arcsec", 0.05)
        if pixel_scale_arcsec is None
        else pixel_scale_arcsec
    )
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape must contain two positive dimensions")
    if int(supersampling_factor) != supersampling_factor or supersampling_factor <= 0:
        raise ValueError("supersampling_factor must be a positive integer")
    if not np.isfinite(psf_fwhm_arcsec) or psf_fwhm_arcsec <= 0:
        raise ValueError("psf_fwhm_arcsec must be positive and finite")

    has_shear = "gamma1" in parameters or "gamma2" in parameters
    if has_shear and not {"gamma1", "gamma2"}.issubset(parameters):
        raise ValueError("External shear requires both gamma1 and gamma2")
    lens_model_list = ["SIE", "SHEAR"] if has_shear else ["SIE"]
    lens = build_lens_model(lens_model_list)
    source = build_light_model(["SERSIC_ELLIPSE"])
    lens_light = build_light_model(["SERSIC_ELLIPSE"])
    saved_positions_x = parameters.get("point_positions_x_arcsec")
    saved_positions_y = parameters.get("point_positions_y_arcsec")
    saved_amplitudes = parameters.get("point_amplitudes")
    use_saved_positions = (
        saved_positions_x is not None
        and saved_positions_y is not None
        and saved_amplitudes is not None
    )
    if use_saved_positions:
        point_source = None
    else:
        point_source = build_point_source_model(
            ["SOURCE_POSITION"],
            lens_model=lens,
            fixed_magnification_list=[True],
            additional_images_list=[True],
        )
    lens_parameters = sie_kwargs(
        parameters["theta_E_arcsec"],
        parameters["q"],
        parameters["phi_rad"],
        center_x_arcsec=parameters.get("lens_center_x_arcsec", 0.0),
        center_y_arcsec=parameters.get("lens_center_y_arcsec", 0.0),
        backend="lenstronomy",
    )
    shear_parameters = shear_kwargs(parameters["gamma1"], parameters["gamma2"]) if has_shear else None
    source_parameters = _light_kwargs(
        parameters,
        "source",
        center_x=parameters["source_x_arcsec"],
        center_y=parameters["source_y_arcsec"],
    )
    lens_light_parameters = _light_kwargs(
        parameters,
        "lens_light",
        center_x=parameters.get("lens_center_x_arcsec", 0.0),
        center_y=parameters.get("lens_center_y_arcsec", 0.0),
    )
    if use_saved_positions:
        point_source_parameters = {
            "ra_image": [float(value) for value in saved_positions_x],
            "dec_image": [float(value) for value in saved_positions_y],
            "point_amp": [float(value) for value in saved_amplitudes],
        }
    else:
        point_source_parameters = {
            "ra_source": float(parameters["source_x_arcsec"]),
            "dec_source": float(parameters["source_y_arcsec"]),
            "source_amp": float(parameters["point_source_amp"]),
        }

    from lenstronomy.Data.imaging_data import ImageData

    data = ImageData(
        image_data=np.zeros(shape, dtype=float),
        background_rms=1.0,
        ra_at_xy_0=-((shape[1] - 1) / 2.0) * pixel_scale,
        dec_at_xy_0=-((shape[0] - 1) / 2.0) * pixel_scale,
        transform_pix2angle=np.array([[pixel_scale, 0.0], [0.0, pixel_scale]]),
    )
    numerics = {"supersampling_factor": int(supersampling_factor)}
    no_psf = build_psf(psf_type="NONE")
    clean_renderer = build_image_model(
        data,
        no_psf,
        lens_model_class=lens,
        source_model_class=source,
        lens_light_model_class=lens_light,
        kwargs_numerics=numerics,
    )
    kwargs_lens = [lens_parameters] + ([] if shear_parameters is None else [shear_parameters])
    kwargs_source = [source_parameters]
    kwargs_lens_light = [lens_light_parameters]
    if use_saved_positions:
        positions_x = tuple(float(np.round(value, 8)) for value in saved_positions_x)
        positions_y = tuple(float(np.round(value, 8)) for value in saved_positions_y)
        if "point_source_amp" in parameters:
            magnifications = np.asarray(
                lens.magnification(np.asarray(positions_x), np.asarray(positions_y), kwargs_lens),
                dtype=float,
            ).ravel()
            amplitudes = tuple(
                float(np.round(abs(value) * float(parameters["point_source_amp"]), 3))
                for value in magnifications
            )
        else:
            amplitudes = tuple(float(np.round(value, 3)) for value in saved_amplitudes)
        point_source_parameters["ra_image"] = list(positions_x)
        point_source_parameters["dec_image"] = list(positions_y)
        point_source_parameters["point_amp"] = list(amplitudes)
    else:
        kwargs_ps = [point_source_parameters]
        positions_x, positions_y, amplitudes = point_source.point_source_list(kwargs_ps, kwargs_lens)
        positions_x = tuple(float(value) for value in np.asarray(positions_x, dtype=float).ravel())
        positions_y = tuple(float(value) for value in np.asarray(positions_y, dtype=float).ravel())
        amplitudes = tuple(float(value) for value in np.asarray(amplitudes, dtype=float).ravel())
    if not positions_x or len(positions_x) != len(positions_y) or len(positions_x) != len(amplitudes):
        raise RuntimeError("lenstronomy did not produce a valid point-source list")
    clean_scene = np.asarray(
        clean_renderer.image(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            # A PSF-free point source is not a detector image; the clean
            # diagnostic is the extended source + foreground light scene.
            kwargs_ps=[],
        ),
        dtype=float,
    )
    # The lenstronomy point-source painter is useful for backend diagnostics,
    # but is not stable enough across processes for released data products.
    # Deposit solved image positions on the pixel grid ourselves, then apply
    # the repository Gaussian control PSF with a deterministic zero boundary.
    point_scene = np.zeros(shape, dtype=float)
    image_center_x = (shape[1] - 1) / 2.0
    image_center_y = (shape[0] - 1) / 2.0
    for position_x, position_y, amplitude in zip(positions_x, positions_y, amplitudes):
        _deposit_bilinear(
            point_scene,
            x_pixel=image_center_x + position_x / pixel_scale,
            y_pixel=image_center_y + position_y / pixel_scale,
            amplitude=amplitude,
        )
    psf_kernel = observation_psf(
        psf_fwhm_arcsec=float(psf_fwhm_arcsec),
        pixel_scale_arcsec=pixel_scale,
    )
    expected_scene = np.asarray(convolve_same(clean_scene + point_scene, psf_kernel), dtype=np.float32)
    if expected_scene.shape != shape or not np.isfinite(expected_scene).all() or not np.isfinite(clean_scene).all():
        raise RuntimeError("lenstronomy produced invalid observation scene products")
    if np.any(expected_scene < -1e-8):
        raise RuntimeError("Expected scene contains negative flux before detector noise")
    return ObservationScene(
        clean_scene=clean_scene,
        expected_scene=expected_scene,
        lens_parameters=lens_parameters,
        shear_parameters=shear_parameters,
        source_parameters=source_parameters,
        lens_light_parameters=lens_light_parameters,
        point_source_parameters=point_source_parameters,
        point_positions_x_arcsec=positions_x,
        point_positions_y_arcsec=positions_y,
        point_amplitudes=amplitudes,
        pixel_scale_arcsec=pixel_scale,
        supersampling_factor=int(supersampling_factor),
        psf_fwhm_arcsec=float(psf_fwhm_arcsec),
    )


def realize_detector_observation(
    expected_scene: np.ndarray,
    *,
    background_rate: float,
    exposure_time: float,
    read_noise_electrons: float,
    noise_seed: int,
) -> DetectorObservation:
    """Draw Poisson counts and read noise before subtracting the background."""

    scene = np.asarray(expected_scene, dtype=float)
    if scene.ndim != 2 or not np.isfinite(scene).all():
        raise ValueError("expected_scene must be a finite two-dimensional array")
    if np.any(scene < -1e-8):
        raise ValueError("expected_scene must be nonnegative before Poisson sampling")
    background = float(background_rate)
    exposure = float(exposure_time)
    read_noise = float(read_noise_electrons)
    if not np.isfinite(background) or background < 0:
        raise ValueError("background_rate must be finite and nonnegative")
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("exposure_time must be positive and finite")
    if not np.isfinite(read_noise) or read_noise < 0:
        raise ValueError("read_noise_electrons must be finite and nonnegative")
    expected_rate = np.maximum(scene, 0.0)
    expected_electrons = exposure * (expected_rate + background)
    rng = np.random.default_rng(int(noise_seed))
    poisson_electrons = rng.poisson(expected_electrons).astype(float)
    read_electrons = rng.normal(0.0, read_noise, size=scene.shape)
    observed_electrons = poisson_electrons + read_electrons
    observed_rate = observed_electrons / exposure
    science_rate = observed_rate - background
    variance_rate = (expected_electrons + read_noise**2) / exposure**2
    return DetectorObservation(
        expected_electrons=expected_electrons.astype(np.float32),
        observed_electrons=observed_electrons.astype(np.float32),
        observed_rate=observed_rate.astype(np.float32),
        science_rate=science_rate.astype(np.float32),
        variance_rate=variance_rate.astype(np.float32),
        background_rate=background,
        exposure_time=exposure,
        read_noise_electrons=read_noise,
        noise_seed=int(noise_seed),
    )
