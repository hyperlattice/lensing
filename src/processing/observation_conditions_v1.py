"""Torch-free physical observation cells for isolated simulation environments.

This is the data-pipeline counterpart of the hash-preserved ML policy v1.
Parity tests ensure their cell definitions remain identical.
"""
import numpy as np

CELLS=("pixel-scale-004","pixel-scale-006","exposure-half","read-noise-double",
       "gaussian-fwhm-075","gaussian-fwhm-125","fresh-noise")


def condition_for(cell, *, shape=(256,256),pixel_scale_arcsec=.05):
    if cell not in CELLS or len(shape)!=2 or any(int(n)!=n or n<=0 for n in shape):
        raise ValueError("Invalid transformation or canonical shape")
    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec<=0:
        raise ValueError("Canonical scale must be positive and finite")
    scale={"pixel-scale-004":.04,"pixel-scale-006":.06}.get(cell,pixel_scale_arcsec)
    native=[max(1,int(np.floor(n*pixel_scale_arcsec/scale+.5))) for n in shape]
    return {"name":cell,"shape":native,"pixel_scale_arcsec":float(scale),"psf_family":"gaussian",
        "psf_fwhm_multiplier":{"gaussian-fwhm-075":.75,"gaussian-fwhm-125":1.25}.get(cell,1.),
        "exposure_multiplier":.5 if cell=="exposure-half" else 1.,
        "read_noise_multiplier":2. if cell=="read-noise-double" else 1.}
