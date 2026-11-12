"""Independent SIS/SIE deflection equations for cross-implementation controls."""

from __future__ import annotations

import numpy as np


def independent_sie_deflection(
    theta_x_arcsec,
    theta_y_arcsec,
    *,
    theta_E_arcsec: float,
    axis_ratio: float,
    phi_rad: float,
    center_x_arcsec: float = 0.0,
    center_y_arcsec: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the singular isothermal ellipsoid deflection in sky coordinates.

    ``axis_ratio`` is minor/major and ``theta_E_arcsec`` is circularized, matching
    the project parameter convention. The calculation uses the closed-form
    isothermal ellipsoid equation in its principal-axis frame, then rotates the
    deflection back to the image frame. The circular limit is handled explicitly.
    """
    theta_e = float(theta_E_arcsec)
    q = float(axis_ratio)
    phi = float(phi_rad)
    center_x = float(center_x_arcsec)
    center_y = float(center_y_arcsec)
    scalars = np.asarray([theta_e, q, phi, center_x, center_y], dtype=np.float64)
    if not np.isfinite(scalars).all() or theta_e <= 0.0 or not 0.0 < q <= 1.0:
        raise ValueError("SIE parameters must be finite, with theta_E > 0 and 0 < q <= 1")
    x, y = np.broadcast_arrays(
        np.asarray(theta_x_arcsec, dtype=np.float64),
        np.asarray(theta_y_arcsec, dtype=np.float64),
    )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Image-plane coordinates must be finite")

    dx = x - center_x
    dy = y - center_y
    cosine = np.cos(phi)
    sine = np.sin(phi)
    major = cosine * dx + sine * dy
    minor = -sine * dx + cosine * dy
    radius = np.hypot(major, minor)
    if 1.0 - q * q <= 1e-12:
        safe_radius = np.where(radius == 0.0, 1.0, radius)
        alpha_major = theta_e * major / safe_radius
        alpha_minor = theta_e * minor / safe_radius
    else:
        eccentricity = float(np.sqrt(1.0 - q * q))
        elliptical_radius = np.sqrt(q * q * major * major + minor * minor)
        safe_radius = np.where(elliptical_radius == 0.0, 1.0, elliptical_radius)
        angle_x = eccentricity * major / safe_radius
        angle_y = eccentricity * minor / safe_radius
        alpha_scale = theta_e * np.sqrt(q) / eccentricity
        alpha_major = alpha_scale * np.arctan(angle_x)
        # The mathematical argument is strictly inside (-1, 1); clipping only
        # protects the singular-center floating-point limit.
        angle_y = np.clip(angle_y, -1.0 + 1e-14, 1.0 - 1e-14)
        alpha_minor = alpha_scale * np.arctanh(angle_y)
    alpha_major = np.where(radius == 0.0, 0.0, alpha_major)
    alpha_minor = np.where(radius == 0.0, 0.0, alpha_minor)

    alpha_x = cosine * alpha_major - sine * alpha_minor
    alpha_y = sine * alpha_major + cosine * alpha_minor
    return alpha_x.astype(np.float64, copy=False), alpha_y.astype(np.float64, copy=False)


def solve_independent_sie_images(
    *,
    source_x_arcsec: float,
    source_y_arcsec: float,
    theta_E_arcsec: float,
    axis_ratio: float,
    phi_rad: float,
    center_x_arcsec: float = 0.0,
    center_y_arcsec: float = 0.0,
    angular_starts: int = 24,
) -> np.ndarray:
    """Find distinct point-source images from the independent lens equation."""
    from scipy.optimize import root

    values = np.asarray(
        [
            source_x_arcsec,
            source_y_arcsec,
            theta_E_arcsec,
            axis_ratio,
            phi_rad,
            center_x_arcsec,
            center_y_arcsec,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or theta_E_arcsec <= 0 or not 0 < axis_ratio <= 1:
        raise ValueError("Point-source and SIE parameters must be finite and physical")
    if int(angular_starts) != angular_starts or angular_starts < 8:
        raise ValueError("angular_starts must be an integer of at least 8")

    def lens_residual(position):
        alpha_x, alpha_y = independent_sie_deflection(
            position[0],
            position[1],
            theta_E_arcsec=theta_E_arcsec,
            axis_ratio=axis_ratio,
            phi_rad=phi_rad,
            center_x_arcsec=center_x_arcsec,
            center_y_arcsec=center_y_arcsec,
        )
        return np.asarray(
            [position[0] - alpha_x - source_x_arcsec, position[1] - alpha_y - source_y_arcsec],
            dtype=np.float64,
        )

    center = np.asarray([center_x_arcsec, center_y_arcsec], dtype=np.float64)
    radii = np.asarray([0.35, 0.65, 0.90, 1.10, 1.40, 1.80]) * float(theta_E_arcsec)
    angles = np.linspace(0.0, 2.0 * np.pi, int(angular_starts), endpoint=False)
    images: list[np.ndarray] = []
    for radius in radii:
        for angle in angles:
            initial = center + radius * np.asarray([np.cos(angle), np.sin(angle)])
            solution = root(lens_residual, initial, method="hybr", options={"xtol": 1e-10})
            position = np.asarray(solution.x, dtype=np.float64)
            if not np.isfinite(position).all() or np.linalg.norm(lens_residual(position)) > 1e-7:
                continue
            if np.linalg.norm(position - center) > 4.0 * float(theta_E_arcsec):
                continue
            if not any(np.linalg.norm(position - previous) < 1e-4 for previous in images):
                images.append(position)
    images.sort(
        key=lambda point: float(np.arctan2(point[1] - center_y_arcsec, point[0] - center_x_arcsec))
    )
    return np.asarray(images, dtype=np.float64).reshape((-1, 2))


def independent_sie_magnification(
    image_x_arcsec: float,
    image_y_arcsec: float,
    *,
    theta_E_arcsec: float,
    axis_ratio: float,
    phi_rad: float,
    center_x_arcsec: float = 0.0,
    center_y_arcsec: float = 0.0,
    finite_difference_arcsec: float = 1e-5,
) -> float:
    """Estimate signed magnification from a centered finite-difference Jacobian."""
    step = float(finite_difference_arcsec)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("finite_difference_arcsec must be positive and finite")

    def alpha(x, y):
        return np.asarray(
            independent_sie_deflection(
                x,
                y,
                theta_E_arcsec=theta_E_arcsec,
                axis_ratio=axis_ratio,
                phi_rad=phi_rad,
                center_x_arcsec=center_x_arcsec,
                center_y_arcsec=center_y_arcsec,
            ),
            dtype=np.float64,
        )

    derivative_x = (
        alpha(image_x_arcsec + step, image_y_arcsec) - alpha(image_x_arcsec - step, image_y_arcsec)
    ) / (2.0 * step)
    derivative_y = (
        alpha(image_x_arcsec, image_y_arcsec + step) - alpha(image_x_arcsec, image_y_arcsec - step)
    ) / (2.0 * step)
    jacobian = np.eye(2, dtype=np.float64) - np.column_stack((derivative_x, derivative_y))
    determinant = float(np.linalg.det(jacobian))
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        return float(np.copysign(1e12, determinant if determinant else 1.0))
    return 1.0 / determinant
