"""Project-level wrappers around the optional lenstronomy model API.

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
from typing import Any, Mapping, Sequence

import numpy as np


class SimulationBackendUnavailable(ImportError):
    """Raised when a backend-dependent simulation operation cannot run."""


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
        detail = "Install requirements-simulation.txt to enable backend rendering"
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


def _normalise_model_list(models: str | Sequence[str], name: str) -> list[str]:
    if isinstance(models, str):
        models = [models]
    values = [str(model) for model in models]
    if not values:
        raise ValueError(f"{name} must contain at least one model")
    return values


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
    _, param_util = _load_lenstronomy()
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
    _, param_util = _load_lenstronomy()
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


def _load_lenstronomy() -> tuple[dict[str, Any], Any]:
    """Import the backend lazily and provide a consistent project error."""

    status = backend_status()
    if not status.available:
        raise SimulationBackendUnavailable(
            "lenstronomy is unavailable. Install requirements-simulation.txt "
            "or use the analytic controls."
        )
    try:
        from lenstronomy.Data.psf import PSF
        from lenstronomy.ImSim.image_model import ImageModel
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.LightModel.light_model import LightModel
        from lenstronomy.PointSource.point_source import PointSource
        from lenstronomy.Util import param_util
        from lenstronomy.Workflow.fitting_sequence import FittingSequence
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise SimulationBackendUnavailable(
            f"lenstronomy {status.version} could not be imported: {exc}"
        ) from exc
    return (
        {
            "LensModel": LensModel,
            "LightModel": LightModel,
            "PointSource": PointSource,
            "PSF": PSF,
            "ImageModel": ImageModel,
            "FittingSequence": FittingSequence,
        },
        param_util,
    )


def build_lens_model(
    lens_model_list: str | Sequence[str] = ("SIE",),
    *,
    z_lens: float | None = None,
    z_source: float | None = None,
    **kwargs: Any,
) -> Any:
    """Construct a lenstronomy ``LensModel`` through the project boundary."""

    classes, _ = _load_lenstronomy()
    parameters: dict[str, Any] = {
        "lens_model_list": _normalise_model_list(lens_model_list, "lens_model_list"),
        **kwargs,
    }
    if z_lens is not None:
        parameters["z_lens"] = _require_scalar(z_lens, "z_lens")
    if z_source is not None:
        parameters["z_source"] = _require_scalar(z_source, "z_source")
    return classes["LensModel"](**parameters)


def build_light_model(
    light_model_list: str | Sequence[str] = ("SERSIC_ELLIPSE",),
    **kwargs: Any,
) -> Any:
    """Construct a lenstronomy ``LightModel``."""

    classes, _ = _load_lenstronomy()
    parameters = {"light_model_list": _normalise_model_list(light_model_list, "light_model_list"), **kwargs}
    return classes["LightModel"](**parameters)


def build_point_source_model(
    point_source_type_list: str | Sequence[str] = ("UNLENSED",),
    *,
    lens_model: Any = None,
    **kwargs: Any,
) -> Any:
    """Construct a lenstronomy ``PointSource``."""

    classes, _ = _load_lenstronomy()
    parameters = {
        "point_source_type_list": _normalise_model_list(
            point_source_type_list, "point_source_type_list"
        ),
        "lens_model": lens_model,
        **kwargs,
    }
    return classes["PointSource"](**parameters)


def build_psf(**kwargs: Any) -> Any:
    """Construct a lenstronomy ``PSF``."""

    classes, _ = _load_lenstronomy()
    return classes["PSF"](**kwargs)


def build_image_model(
    data_class: Any,
    psf_class: Any,
    *,
    lens_model_class: Any = None,
    source_model_class: Any = None,
    lens_light_model_class: Any = None,
    point_source_class: Any = None,
    **kwargs: Any,
) -> Any:
    """Construct lenstronomy's image renderer from already-built components."""

    classes, _ = _load_lenstronomy()
    parameters = {
        "data_class": data_class,
        "psf_class": psf_class,
        "lens_model_class": lens_model_class,
        "source_model_class": source_model_class,
        "lens_light_model_class": lens_light_model_class,
        "point_source_class": point_source_class,
        **kwargs,
    }
    return classes["ImageModel"](**parameters)


def build_fitting_sequence(
    kwargs_data_joint: Mapping[str, Any],
    kwargs_model: Mapping[str, Any],
    kwargs_constraints: Mapping[str, Any],
    kwargs_likelihood: Mapping[str, Any],
    kwargs_params: Mapping[str, Any],
    **kwargs: Any,
) -> Any:
    """Construct lenstronomy's fitting workflow without starting a sampler."""

    classes, _ = _load_lenstronomy()
    return classes["FittingSequence"](
        dict(kwargs_data_joint),
        dict(kwargs_model),
        dict(kwargs_constraints),
        dict(kwargs_likelihood),
        dict(kwargs_params),
        **kwargs,
    )
