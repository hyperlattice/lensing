"""AutoLens physical profiles and coordinate conversions; imported lazily."""

from __future__ import annotations

import importlib.metadata
import math
from typing import Any

import numpy as np
from scipy.optimize import root


def backend():
    try:
        import autolens as al
    except ImportError as exc:
        raise ImportError('Install this project with the "autolens" extra on Python 3.12') from exc
    return al


def backend_version() -> str:
    return importlib.metadata.version("autolens")


def _ellipticity(q, phi):
    amplitude = (1.0 - float(q)) / (1.0 + float(q))
    # PyAutoLens uses (e_y, e_x), corresponding to project (e2, e1).
    return (amplitude * math.sin(2 * phi), amplitude * math.cos(2 * phi))


def mass_profile(parameters: dict[str, float]) -> Any:
    al = backend()
    q = float(parameters["q"])
    if not 0 < q <= 1:
        raise ValueError("Mass axis ratio must be in (0,1]")
    # PyAutoGalaxy's PowerLawIntermediate documents the cross-code relation:
    # theta_Lenstronomy = 2*sqrt(q)/(1+q) * theta_PyAutoLens for slope=2.
    # Isothermal uses the latter convention, so translate before construction.
    backend_radius = float(parameters["theta_E_arcsec"]) * (1 + q) / (2 * np.sqrt(q))
    return al.mp.Isothermal(
        centre=(
            parameters.get("lens_center_y_arcsec", 0.0),
            parameters.get("lens_center_x_arcsec", 0.0),
        ),
        ell_comps=_ellipticity(parameters["q"], parameters["phi_rad"]),
        einstein_radius=backend_radius,
    )


def deflections_xy(profile: Any, xy: np.ndarray) -> np.ndarray:
    """Convert project (x,y) coordinates to backend (y,x), then convert back."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Coordinates must be finite (N,2) x,y pairs")
    grid = backend().Grid2DIrregular(values=xy[:, ::-1].copy())
    return np.asarray(profile.deflections_yx_2d_from(grid=grid))[:, ::-1]


def mapped_light_parameters(intensity: float, radius: float, n: float) -> dict[str, float]:
    """Preserve the original circularized radial profile across b_n conventions.

    Both parameters change because the implementations approximate b_n
    differently. The original 1e-4 arcsec central floor is not applied here;
    controls explicitly assess its remaining image-level effect.
    """
    if not np.isfinite([intensity, radius, n]).all() or min(radius, n) <= 0 or intensity < 0:
        raise ValueError(
            "Light parameters must be finite with positive radius/index and nonnegative intensity"
        )
    probe = backend().lp.Sersic(sersic_index=float(n))
    b_a = float(probe.sersic_constant)
    b_l = max(1.9992 * n - 0.3271, 1e-5)
    return {
        "intensity": float(intensity * np.exp(b_l - b_a)),
        "effective_radius": float(radius * (b_a / b_l) ** n),
        "sersic_index": float(n),
        "b_original": b_l,
        "b_backend": b_a,
    }


def _light(parameters, prefix, center):
    mapped = mapped_light_parameters(
        float(parameters[f"{prefix}_amp"]),
        float(parameters[f"{prefix}_re_arcsec"]),
        float(parameters[f"{prefix}_n"]),
    )
    return backend().lp.Sersic(
        centre=(center[1], center[0]),
        ell_comps=_ellipticity(parameters[f"{prefix}_q"], parameters[f"{prefix}_phi_rad"]),
        **{key: mapped[key] for key in ("intensity", "effective_radius", "sersic_index")},
    )


def point_images(
    profile: Any, parameters: dict[str, float], radius_limit: float
) -> tuple[np.ndarray, np.ndarray]:
    """Solve roots with backend deflections, without copied Lenstronomy roots."""
    source = np.array([parameters["source_x_arcsec"], parameters["source_y_arcsec"]])
    center = np.array(
        [parameters.get("lens_center_x_arcsec", 0), parameters.get("lens_center_y_arcsec", 0)]
    )

    def residual(position):
        return position - deflections_xy(profile, np.asarray(position)[None, :])[0] - source

    solutions = []
    for radius in (0.5, 1.0, 1.5):
        for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            start = center + radius * parameters["theta_E_arcsec"] * np.array(
                [np.cos(angle), np.sin(angle)]
            )
            solved = root(residual, start)
            position = solved.x
            if (
                not np.isfinite(position).all()
                or np.linalg.norm(residual(position)) > 1e-7
                or np.linalg.norm(position - center) > radius_limit
            ):
                continue
            if not any(np.linalg.norm(position - other) < 1e-5 for other in solutions):
                solutions.append(position)
    if len(solutions) not in (2, 4):
        raise ValueError(f"Backend root search found {len(solutions)} roots; expected 2 or 4")
    positions = np.asarray(
        sorted(
            solutions,
            key=lambda parameters: math.atan2(parameters[1] - center[1], parameters[0] - center[0]),
        )
    )
    amplitudes = []
    for position in positions:
        h = 1e-5
        jacobian = np.column_stack(
            [
                (
                    residual(position + np.eye(2)[axis] * h)
                    - residual(position - np.eye(2)[axis] * h)
                )
                / (2 * h)
                for axis in range(2)
            ]
        )
        amplitudes.append(abs(1 / np.linalg.det(jacobian)) * parameters["point_source_amp"])
    return positions, np.asarray(amplitudes)
