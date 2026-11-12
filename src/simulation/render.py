"""Deterministic analytic renderers used to validate the simulation pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.simulation.models import (
    ellipticity_from_q_phi,
    sie_kwargs,
    shear_kwargs,
    sis_kwargs,
    sis_deflection,
)


@dataclass(frozen=True)
class SISImageSolution:
    """One analytic point-source image of an SIS lens."""

    theta_x_arcsec: float
    theta_y_arcsec: float
    magnification_signed: float
    magnification_abs: float
    parity: int
    residual_x_arcsec: float
    residual_y_arcsec: float
    residual_norm_arcsec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta_x_arcsec": self.theta_x_arcsec,
            "theta_y_arcsec": self.theta_y_arcsec,
            "magnification_signed": self.magnification_signed,
            "magnification_abs": self.magnification_abs,
            "parity": self.parity,
            "residual_x_arcsec": self.residual_x_arcsec,
            "residual_y_arcsec": self.residual_y_arcsec,
            "residual_norm_arcsec": self.residual_norm_arcsec,
        }


@dataclass(frozen=True)
class SISRender:
    """Analytic image, source-plane position, and coordinate metadata."""

    image: np.ndarray
    solutions: tuple[SISImageSolution, ...]
    pixel_scale_arcsec: float
    center_pixel_yx: tuple[float, float]
    source_x_arcsec: float
    source_y_arcsec: float
    theta_E_arcsec: float

    def geometry(self) -> dict[str, Any]:
        scale = self.pixel_scale_arcsec
        return {
            "coordinate_convention": "x_y_arcsec_about_lens_center",
            "image_index_order": "image[y, x]",
            "pixel_center_convention": "zero_based_pixel_centers",
            "pixel_scale_arcsec": scale,
            "center_pixel_yx": list(self.center_pixel_yx),
            "center_angle_xy_arcsec": [0.0, 0.0],
            "pixel_to_angle_matrix_arcsec": [[scale, 0.0], [0.0, scale]],
        }


@dataclass(frozen=True)
class SIEExtendedRender:
    """A lenstronomy-rendered extended-source SIE image and its conventions."""

    image: np.ndarray
    lens_parameters: dict[str, float]
    shear_parameters: dict[str, float] | None
    source_parameters: dict[str, float]
    pixel_scale_arcsec: float
    supersampling_factor: int

    def geometry(self) -> dict[str, Any]:
        height, width = self.image.shape
        scale = self.pixel_scale_arcsec
        return {
            "coordinate_convention": "x_y_arcsec_about_lens_center",
            "image_index_order": "image[y, x]",
            "pixel_center_convention": "zero_based_pixel_centers",
            "shape": [int(height), int(width)],
            "pixel_scale_arcsec": scale,
            "center_pixel_yx": [(height - 1) / 2.0, (width - 1) / 2.0],
            "center_angle_xy_arcsec": [0.0, 0.0],
            "pixel_to_angle_matrix_arcsec": [[scale, 0.0], [0.0, scale]],
            "supersampling_factor": self.supersampling_factor,
        }


def sis_image_positions(
    theta_E_arcsec: float,
    source_x_arcsec: float,
    source_y_arcsec: float,
) -> tuple[SISImageSolution, ...]:
    """Solve the two SIS point-source image positions analytically.

    The source must be offset from the lens center and inside the Einstein radius;
    that is the two-image control regime used by P10.
    """

    theta_E = float(theta_E_arcsec)
    beta_x = float(source_x_arcsec)
    beta_y = float(source_y_arcsec)
    beta = math.hypot(beta_x, beta_y)
    if not np.isfinite([theta_E, beta_x, beta_y]).all() or theta_E <= 0:
        raise ValueError("SIS radius and source coordinates must be finite; radius must be positive")
    if beta <= 0:
        raise ValueError("A centered point source produces an Einstein ring, not two point roots")
    if beta >= theta_E:
        raise ValueError("The SIS control requires source radius smaller than theta_E")

    unit_x, unit_y = beta_x / beta, beta_y / beta
    positions = ((theta_E + beta), (beta - theta_E))
    magnifications = (1.0 + theta_E / beta, 1.0 - theta_E / beta)
    parities = (1, -1)
    solutions = []
    for radial, magnification, parity in zip(positions, magnifications, parities):
        theta_x = radial * unit_x
        theta_y = radial * unit_y
        alpha_x, alpha_y = sis_deflection(theta_x, theta_y, theta_E)
        residual_x = float(theta_x - float(alpha_x) - beta_x)
        residual_y = float(theta_y - float(alpha_y) - beta_y)
        solutions.append(
            SISImageSolution(
                theta_x_arcsec=float(theta_x),
                theta_y_arcsec=float(theta_y),
                magnification_signed=float(magnification),
                magnification_abs=float(abs(magnification)),
                parity=parity,
                residual_x_arcsec=residual_x,
                residual_y_arcsec=residual_y,
                residual_norm_arcsec=float(math.hypot(residual_x, residual_y)),
            )
        )
    return tuple(solutions)


def _deposit_bilinear(image: np.ndarray, x_pixel: float, y_pixel: float, value: float) -> None:
    """Deposit a point at fractional pixel coordinates while conserving in-bounds flux."""

    height, width = image.shape
    x0, y0 = math.floor(x_pixel), math.floor(y_pixel)
    dx, dy = x_pixel - x0, y_pixel - y0
    for offset_y, weight_y in ((0, 1.0 - dy), (1, dy)):
        for offset_x, weight_x in ((0, 1.0 - dx), (1, dx)):
            x_index, y_index = x0 + offset_x, y0 + offset_y
            if 0 <= x_index < width and 0 <= y_index < height:
                image[y_index, x_index] += np.float32(value * weight_x * weight_y)


def render_sis_point_source(
    *,
    theta_E_arcsec: float,
    source_x_arcsec: float,
    source_y_arcsec: float,
    source_flux: float = 1.0,
    shape: tuple[int, int] = (256, 256),
    pixel_scale_arcsec: float = 0.05,
) -> SISRender:
    """Render a noiseless, PSF-free SIS point source onto a pixel grid."""

    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("shape must contain positive dimensions")
    pixel_scale = float(pixel_scale_arcsec)
    flux = float(source_flux)
    if not np.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError("pixel_scale_arcsec must be positive and finite")
    if not np.isfinite(flux) or flux <= 0:
        raise ValueError("source_flux must be positive and finite")

    solutions = sis_image_positions(theta_E_arcsec, source_x_arcsec, source_y_arcsec)
    image = np.zeros((height, width), dtype=np.float32)
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    for solution in solutions:
        x_pixel = center_x + solution.theta_x_arcsec / pixel_scale
        y_pixel = center_y + solution.theta_y_arcsec / pixel_scale
        if not (-1 <= x_pixel <= width and -1 <= y_pixel <= height):
            raise ValueError("SIS image position falls outside the render grid")
        _deposit_bilinear(
            image,
            x_pixel,
            y_pixel,
            flux * solution.magnification_abs,
        )
    return SISRender(
        image=image,
        solutions=solutions,
        pixel_scale_arcsec=pixel_scale,
        center_pixel_yx=(center_y, center_x),
        source_x_arcsec=float(source_x_arcsec),
        source_y_arcsec=float(source_y_arcsec),
        theta_E_arcsec=float(theta_E_arcsec),
    )


def render_from_parameters(parameters: Mapping[str, Any]) -> SISRender:
    """Render a control from a saved parameter dictionary."""

    return render_sis_point_source(
        theta_E_arcsec=parameters["theta_E_arcsec"],
        source_x_arcsec=parameters["source_x_arcsec"],
        source_y_arcsec=parameters["source_y_arcsec"],
        source_flux=parameters.get("source_flux", 1.0),
        shape=tuple(parameters.get("shape", (256, 256))),
        pixel_scale_arcsec=parameters.get("pixel_scale_arcsec", 0.05),
    )


def _render_extended_source(
    parameters: Mapping[str, Any],
    *,
    lens_profile: str,
    shape: tuple[int, int] | None = None,
    pixel_scale_arcsec: float | None = None,
    supersampling_factor: int = 1,
) -> SIEExtendedRender:
    """Render an extended Sérsic source through the lenstronomy adapter.

    This function is intentionally free of file I/O. The generator owns FITS,
    manifest, and truth artifacts, while this function remains useful for unit
    tests and numerical convergence checks.
    """

    from lenstronomy.Data.psf import PSF
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel

    shape = tuple(parameters.get("shape", (256, 256)) if shape is None else shape)
    pixel_scale = float(
        parameters.get("pixel_scale_arcsec", 0.05)
        if pixel_scale_arcsec is None
        else pixel_scale_arcsec
    )
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape must contain two positive dimensions")
    if not np.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError("pixel_scale_arcsec must be positive and finite")
    if int(supersampling_factor) != supersampling_factor or supersampling_factor <= 0:
        raise ValueError("supersampling_factor must be a positive integer")

    theta_E = float(parameters["theta_E_arcsec"])
    lens_q = float(parameters.get("q", 1.0))
    lens_phi = float(parameters.get("phi_rad", 0.0))
    lens_center_x = float(parameters.get("lens_center_x_arcsec", 0.0))
    lens_center_y = float(parameters.get("lens_center_y_arcsec", 0.0))
    source_q = float(parameters["source_q"])
    source_phi = float(parameters["source_phi_rad"])
    source_e1, source_e2 = ellipticity_from_q_phi(source_q, source_phi)
    has_shear = "gamma1" in parameters or "gamma2" in parameters
    if has_shear and not {"gamma1", "gamma2"}.issubset(parameters):
        raise ValueError("External shear requires both gamma1 and gamma2")
    lens_model_list = [lens_profile, "SHEAR"] if has_shear else [lens_profile]
    lens = LensModel(lens_model_list=lens_model_list)
    source = LightModel(light_model_list=["SERSIC_ELLIPSE"])
    psf = PSF(psf_type="NONE")

    from lenstronomy.Data.imaging_data import ImageData

    data = ImageData(
        image_data=np.zeros(shape, dtype=float),
        background_rms=1.0,
        ra_at_xy_0=-((shape[1] - 1) / 2.0) * pixel_scale,
        dec_at_xy_0=-((shape[0] - 1) / 2.0) * pixel_scale,
        transform_pix2angle=np.array([[pixel_scale, 0.0], [0.0, pixel_scale]]),
    )
    renderer = ImageModel(
        data,
        psf,
        lens_model_class=lens,
        source_model_class=source,
        kwargs_numerics={"supersampling_factor": int(supersampling_factor)},
    )
    if lens_profile == "SIS":
        lens_parameters = sis_kwargs(
            theta_E,
            center_x_arcsec=lens_center_x,
            center_y_arcsec=lens_center_y,
        )
    elif lens_profile == "SIE":
        lens_parameters = sie_kwargs(
            theta_E,
            lens_q,
            lens_phi,
            center_x_arcsec=lens_center_x,
            center_y_arcsec=lens_center_y,
            backend="lenstronomy",
        )
    else:
        raise ValueError("lens_profile must be 'SIS' or 'SIE'")
    shear_parameters = shear_kwargs(parameters["gamma1"], parameters["gamma2"]) if has_shear else None
    source_parameters = {
        "amp": float(parameters["source_amp"]),
        "R_sersic": float(parameters["source_re_arcsec"]),
        "n_sersic": float(parameters["source_n"]),
        "e1": float(source_e1),
        "e2": float(source_e2),
        "center_x": float(parameters["source_x_arcsec"]),
        "center_y": float(parameters["source_y_arcsec"]),
    }
    kwargs_lens = [lens_parameters] + ([] if shear_parameters is None else [shear_parameters])
    image = renderer.image(
        kwargs_lens=kwargs_lens,
        kwargs_source=[source_parameters],
    )
    image = np.asarray(image, dtype=np.float32)
    if image.shape != shape or not np.isfinite(image).all():
        raise FloatingPointError("lenstronomy produced an invalid extended-source image")
    return SIEExtendedRender(
        image=image,
        lens_parameters=lens_parameters,
        shear_parameters=shear_parameters,
        source_parameters=source_parameters,
        pixel_scale_arcsec=pixel_scale,
        supersampling_factor=int(supersampling_factor),
    )


def render_sie_extended_source(
    parameters: Mapping[str, Any],
    *,
    shape: tuple[int, int] | None = None,
    pixel_scale_arcsec: float | None = None,
    supersampling_factor: int = 1,
) -> SIEExtendedRender:
    """Render an extended Sérsic source through a lenstronomy SIE lens."""

    return _render_extended_source(
        parameters,
        lens_profile="SIE",
        shape=shape,
        pixel_scale_arcsec=pixel_scale_arcsec,
        supersampling_factor=supersampling_factor,
    )


def render_sis_extended_source(
    parameters: Mapping[str, Any],
    *,
    shape: tuple[int, int] | None = None,
    pixel_scale_arcsec: float | None = None,
    supersampling_factor: int = 1,
) -> SIEExtendedRender:
    """Render the same extended source through an SIS lens for a circular-limit check."""

    return _render_extended_source(
        parameters,
        lens_profile="SIS",
        shape=shape,
        pixel_scale_arcsec=pixel_scale_arcsec,
        supersampling_factor=supersampling_factor,
    )
