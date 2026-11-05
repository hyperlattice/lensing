import importlib.util

import numpy as np
import pytest

from src.simulation.autolens_backend import psf_kernel


def test_new_psf_family_has_more_wing_flux_at_fixed_fwhm():
    gaussian=psf_kernel(0.15,0.05,"gaussian")
    moffat=psf_kernel(0.15,0.05,"moffat",3.5)
    yy,xx=np.indices(gaussian.shape)-np.array(gaussian.shape)[:,None,None]//2
    wings=xx**2+yy**2 > 16
    assert gaussian.sum() == pytest.approx(1)
    assert moffat.sum() == pytest.approx(1)
    assert moffat[wings].sum() > gaussian[wings].sum()


@pytest.mark.skipif(importlib.util.find_spec("autolens") is None,reason="optional external backend")
def test_external_backend_matches_sis_and_elliptical_deflections():
    import yaml
    from pathlib import Path
    from src.simulation.cross_simulator import deflection_controls
    config=yaml.safe_load(Path("configs/simulation/cross_simulator_legacy.yaml").read_text())
    report=deflection_controls(config)
    assert report["passed"], report
    assert len(report["checks"]) == 12
