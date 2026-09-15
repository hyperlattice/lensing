"""Small executable smoke test for the analytic and lenstronomy simulation paths."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.simulation.models import (
    SimulationBackendUnavailable,
    backend_convention_check,
    backend_status,
    build_fitting_sequence,
    build_image_model,
    build_lens_model,
    build_light_model,
    build_point_source_model,
    build_psf,
    sie_kwargs,
    sis_deflection,
)


def _smoke_fitting_sequence(sie: dict[str, float]) -> None:
    """Construct, but do not run, a minimal fitting workflow."""

    build_fitting_sequence(
        kwargs_data_joint={"multi_band_list": []},
        kwargs_model={"lens_model_list": ["SIE"]},
        kwargs_constraints={},
        kwargs_likelihood={},
        kwargs_params={
            "lens_model": (
                [sie],
                [{"theta_E": 0.05, "e1": 0.02, "e2": 0.02, "center_x": 0.02, "center_y": 0.02}],
                [{}],
                [{"theta_E": 0.01, "e1": -0.8, "e2": -0.8, "center_x": -1.0, "center_y": -1.0}],
                [{"theta_E": 3.0, "e1": 0.8, "e2": 0.8, "center_x": 1.0, "center_y": 1.0}],
            )
        },
        verbose=False,
    )


def run_smoke(*, require_backend: bool = False) -> dict[str, Any]:
    """Run deterministic controls and optional backend construction/rendering."""

    # Numba JIT is not needed for this tiny check and can make first import very
    # slow on some Python 3.13 installations.  Real generation may enable it.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 9), np.linspace(-1.0, 1.0, 9))
    alpha_x, alpha_y = sis_deflection(x, y, 0.7)
    radial = np.hypot(alpha_x, alpha_y)
    nonzero = np.hypot(x, y) > 0
    analytic = {
        "sis_deflection_finite": bool(np.isfinite(alpha_x).all() and np.isfinite(alpha_y).all()),
        "sis_deflection_norm": float(np.median(radial[nonzero])),
        "sie_kwargs": sie_kwargs(0.7, 0.65, 0.3),
    }
    if not analytic["sis_deflection_finite"] or not np.isclose(
        analytic["sis_deflection_norm"], 0.7
    ):
        raise RuntimeError("Analytic SIS smoke control failed")

    status = backend_status()
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "backend": status.to_dict(),
        "analytic": analytic,
        "backend_smoke": {"available": status.available, "passed": False},
    }
    if not status.available:
        if require_backend:
            raise SimulationBackendUnavailable(status.detail or "lenstronomy unavailable")
        report["backend_smoke"]["detail"] = status.detail
        return report

    try:
        convention = backend_convention_check()
        lens = build_lens_model(["SIE"])
        light = build_light_model(["SERSIC_ELLIPSE"])
        point_source = build_point_source_model(["UNLENSED"])
        psf = build_psf(psf_type="GAUSSIAN", fwhm=0.1, pixel_size=0.05)

        from lenstronomy.Data.imaging_data import ImageData

        data = ImageData(
            image_data=np.zeros((32, 32), dtype=float),
            background_rms=0.01,
            transform_pix2angle=np.array([[0.05, 0.0], [0.0, 0.05]]),
        )
        renderer = build_image_model(
            data,
            psf,
            lens_model_class=lens,
            source_model_class=light,
            point_source_class=point_source,
            kwargs_numerics={"supersampling_factor": 1},
        )
        sie = sie_kwargs(0.7, 0.65, 0.3)
        e1_source, e2_source = 0.0, 0.0
        image = renderer.image(
            kwargs_lens=[sie],
            kwargs_source=[
                {
                    "amp": 1.0,
                    "R_sersic": 0.2,
                    "n_sersic": 1.0,
                    "e1": e1_source,
                    "e2": e2_source,
                    "center_x": 0.1,
                    "center_y": 0.1,
                }
            ],
            kwargs_ps=[{"ra_image": [0.2], "dec_image": [0.2], "point_amp": [1.0]}],
        )
        _smoke_fitting_sequence(sie)
        if image.shape != (32, 32) or not np.isfinite(image).all():
            raise RuntimeError("Rendered image failed shape or finite-value check")
        report["backend_smoke"] = {
            "available": True,
            "passed": bool(convention["passed"]),
            "convention": convention,
            "components": ["SIE", "SERSIC_ELLIPSE", "UNLENSED", "GAUSSIAN_PSF", "ImageModel", "FittingSequence"],
            "image_shape": list(image.shape),
            "image_sum": float(image.sum()),
        }
        if not convention["passed"]:
            raise RuntimeError("Project/backend ellipticity conventions disagree")
    except SimulationBackendUnavailable:
        raise
    except Exception as exc:
        raise RuntimeError(f"lenstronomy smoke test failed: {exc}") from exc
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-backend",
        action="store_true",
        help="fail if lenstronomy is not installed or cannot render",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    report = run_smoke(require_backend=args.require_backend)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
