"""Interactive FITS header and image inspector.

This module is side-effect free on import. Run it directly to print a header,
optionally save that text, and display the image.
"""

import argparse
from pathlib import Path

from astropy.io import fits
import matplotlib.pyplot as plt


SECTIONS = {
    "SIMPLE": "Basic FITS Information",
    "ROOTNAME": "Data Description",
    "TARGNAME": "Target Information",
    "PROPOSID": "Proposal Information",
    "ORIENTAT": "Exposure Information",
    "CAMERA": "Instrument Configuration",
    "BACKEST1": "Background Keywords",
    "PHOTMODE": "Photometry Keywords",
    "MASKFILE": "CALNICA Calibration Files",
    "MASKPDGR": "Calibration File Pedigree",
}


def format_header(path):
    path = Path(path)
    lines = [f"Filename: {path}", ""]
    with fits.open(path) as hdul:
        lines.append(f"HDU count: {len(hdul)}")
        for index, hdu in enumerate(hdul):
            lines.append(f"HDU {index}: name={hdu.name!r}, shape={getattr(hdu.data, 'shape', None)}")
        lines.append("")
        for card in hdul[0].header.cards:
            if card.keyword in SECTIONS:
                lines.extend(["=" * 60, SECTIONS[card.keyword], "=" * 60])
            if card.keyword in {"COMMENT", "HISTORY"}:
                lines.append(f"{card.keyword:<8} {card.value}")
            elif card.comment:
                lines.append(f"{card.keyword:<8}= {card.value!r:<30} / {card.comment}")
            else:
                lines.append(f"{card.keyword:<8}= {card.value!r}")
        lines.append("\nEnd of Header")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="data/raw/castles/Q2237/FIT/NIC2H/q256.fits")
    parser.add_argument("--output", default=None, help="optional text output path")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    rendered = format_header(args.path)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if not args.no_show:
        image = fits.getdata(args.path)
        figure, axis = plt.subplots(figsize=(8, 8))
        artist = axis.imshow(image, origin="lower", cmap="plasma")
        figure.colorbar(artist, ax=axis, label="Brightness")
        axis.set_title(args.path)
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
