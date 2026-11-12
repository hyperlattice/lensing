"""Lens-model conventions and analytic controls.

The rest of the project uses explicit conventions: angles are radians, axis ratios
are minor/major, coordinates are ``(x, y)`` in arcseconds, and image arrays are
indexed as ``[y, x]``.  This module is the only place where those conventions are
translated into lenstronomy keyword arguments.

The import of lenstronomy is deliberately lazy.  Metadata, preprocessing, and the
analytic SIS controls remain usable when the optional simulation environment has not
been installed.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class BackendStatus:
    """Serializable status record for the simulation environment."""

    name: str
    available: bool
    version: str | None
    python: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def backend_status() -> BackendStatus:
    """Return install status without importing lenstronomy's heavy modules."""

    available = importlib.util.find_spec("lenstronomy") is not None
    version = None
    detail = None
    if available:
        try:
            version = importlib.metadata.version("lenstronomy")
        except importlib.metadata.PackageNotFoundError:
            available = False
            detail = "lenstronomy was discoverable but has no package metadata"
    else:
        detail = "Install requirements.txt to enable backend rendering"
    return BackendStatus(
        name="lenstronomy",
        available=available,
        version=version,
        python=sys.version.split()[0],
        detail=detail,
    )


def _require_scalar(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return number


def ellipticity_from_q_phi(q: Any, phi_rad: Any) -> tuple[float, float]:
    """Convert axis ratio and orientation to lenstronomy's ``(e1, e2)``."""

    q_value = _require_scalar(q, "q")
    phi_value = _require_scalar(phi_rad, "phi_rad")
    if not 0 < q_value <= 1:
        raise ValueError("q must be in the interval (0, 1]")
    amplitude = (1.0 - q_value) / (1.0 + q_value)
    return (
        float(amplitude * math.cos(2.0 * phi_value)),
        float(amplitude * math.sin(2.0 * phi_value)),
    )


def q_phi_to_e1_e2(
    q: Any, phi_rad: Any, *, backend: str = "analytic"
) -> tuple[float, float]:
    """Convert project parameters using either the analytic or backend helper."""

    analytic = ellipticity_from_q_phi(q, phi_rad)
    if backend == "analytic":
        return analytic
    if backend != "lenstronomy":
        raise ValueError("backend must be 'analytic' or 'lenstronomy'")
    from lenstronomy.Util import param_util
    e1, e2 = param_util.phi_q2_ellipticity(float(phi_rad), float(q))
    return float(e1), float(e2)


def e1_e2_to_q_phi(
    e1: Any, e2: Any, *, backend: str = "analytic"
) -> tuple[float, float]:
    """Convert lenstronomy ellipticity components back to ``(q, phi_rad)``."""

    e1_value = _require_scalar(e1, "e1")
    e2_value = _require_scalar(e2, "e2")
    amplitude = math.hypot(e1_value, e2_value)
    if amplitude >= 1:
        raise ValueError("sqrt(e1**2 + e2**2) must be smaller than 1")
    if backend == "analytic":
        phi = math.atan2(e2_value, e1_value) / 2.0
        q = (1.0 - amplitude) / (1.0 + amplitude)
        return float(phi), float(q)
    if backend != "lenstronomy":
        raise ValueError("backend must be 'analytic' or 'lenstronomy'")
    from lenstronomy.Util import param_util
    phi, q = param_util.ellipticity2phi_q(e1_value, e2_value)
    return float(phi), float(q)


def backend_convention_check(
    samples: Sequence[tuple[float, float]] = ((1.0, 0.0), (0.7, 0.3), (0.4, -0.8)),
    *, atol: float = 1e-12,
) -> dict[str, Any]:
    """Compare project conversion with the installed backend helper."""

    expected = [ellipticity_from_q_phi(q, phi) for q, phi in samples]
    actual = [q_phi_to_e1_e2(q, phi, backend="lenstronomy") for q, phi in samples]
    differences = [
        max(abs(expected_value - actual_value) for expected_value, actual_value in zip(e, a))
        for e, a in zip(expected, actual)
    ]
    return {
        "samples": len(samples),
        "max_absolute_difference": max(differences, default=0.0),
        "passed": all(difference <= atol for difference in differences),
        "atol": atol,
    }


def sis_kwargs(
    theta_E_arcsec: Any,
    *,
    center_x_arcsec: Any = 0.0,
    center_y_arcsec: Any = 0.0,
) -> dict[str, float]:
    """Return canonical lenstronomy kwargs for an SIS mass profile."""

    return {
        "theta_E": _require_scalar(theta_E_arcsec, "theta_E_arcsec", positive=True),
        "center_x": _require_scalar(center_x_arcsec, "center_x_arcsec"),
        "center_y": _require_scalar(center_y_arcsec, "center_y_arcsec"),
    }


def sie_kwargs(
    theta_E_arcsec: Any,
    q: Any,
    phi_rad: Any,
    *,
    center_x_arcsec: Any = 0.0,
    center_y_arcsec: Any = 0.0,
    backend: str = "analytic",
) -> dict[str, float]:
    """Return canonical lenstronomy kwargs for an SIE mass profile."""

    e1, e2 = q_phi_to_e1_e2(q, phi_rad, backend=backend)
    return {
        "theta_E": _require_scalar(theta_E_arcsec, "theta_E_arcsec", positive=True),
        "e1": e1,
        "e2": e2,
        "center_x": _require_scalar(center_x_arcsec, "center_x_arcsec"),
        "center_y": _require_scalar(center_y_arcsec, "center_y_arcsec"),
    }


def shear_kwargs(gamma1: Any, gamma2: Any) -> dict[str, float]:
    """Return canonical lenstronomy kwargs for an external shear component."""

    return {
        "gamma1": _require_scalar(gamma1, "gamma1"),
        "gamma2": _require_scalar(gamma2, "gamma2"),
    }


def sis_deflection(
    theta_x_arcsec: Any,
    theta_y_arcsec: Any,
    theta_E_arcsec: Any,
    *,
    center_x_arcsec: Any = 0.0,
    center_y_arcsec: Any = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the analytic SIS deflection in arcseconds.

    The deflection is set to zero at the singular center so control-grid generation
    remains finite and deterministic.
    """

    theta_E = _require_scalar(theta_E_arcsec, "theta_E_arcsec", positive=True)
    center_x = _require_scalar(center_x_arcsec, "center_x_arcsec")
    center_y = _require_scalar(center_y_arcsec, "center_y_arcsec")
    x = np.asarray(theta_x_arcsec, dtype=float)
    y = np.asarray(theta_y_arcsec, dtype=float)
    x_shift = x - center_x
    y_shift = y - center_y
    radius = np.hypot(x_shift, y_shift)
    safe_radius = np.where(radius == 0, 1.0, radius)
    alpha_x = np.where(radius == 0, 0.0, theta_E * x_shift / safe_radius)
    alpha_y = np.where(radius == 0, 0.0, theta_E * y_shift / safe_radius)
    return alpha_x.astype(float), alpha_y.astype(float)
