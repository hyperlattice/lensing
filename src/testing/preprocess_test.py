"""Interactive comparison of a FITS image before and after finite-value cleanup."""

import argparse

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


def inspect_image(path, *, show=True):
    image = np.asarray(fits.getdata(path))
    cleaned = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    summary = {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "minimum": float(np.nanmin(image)),
        "maximum": float(np.nanmax(image)),
        "mean": float(np.nanmean(image)),
        "median": float(np.nanmedian(image)),
        "nan_count": int(np.isnan(image).sum()),
        "infinite_count": int(np.isinf(image).sum()),
    }
    if show:
        figure, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, origin="lower", cmap="gray")
        axes[0].set_title("Original")
        axes[1].imshow(cleaned, origin="lower", cmap="gray")
        axes[1].set_title("Cleaned")
        plt.show()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="data/raw/castles/Q2237/FIT/NIC2H/q256.fits")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    for key, value in inspect_image(args.path, show=not args.no_show).items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
