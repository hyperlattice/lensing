from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt

ROOT = Path("data/raw/castles")

OUTPUT = Path("results/preview")
OUTPUT.mkdir(parents=True, exist_ok=True)

for fits_file in ROOT.rglob("*.fits"):

    try:
        image = fits.getdata(fits_file)

        lens_name = fits_file.parts[-4]

        output_name = f"{lens_name}_{fits_file.stem}.png"

        plt.figure(figsize=(5, 5))
        plt.imshow(image, origin="lower", cmap="gray")
        plt.colorbar(label="Pixel Value")
        plt.title(f"{lens_name} - {fits_file.name}")
        plt.tight_layout()

        # Save as PNG
        plt.savefig(OUTPUT / output_name, dpi=200)
        plt.close()

        print(f"Saved {output_name}")

    except Exception as e:
        print(f"Error reading {fits_file}: {e}")

print("\nFinished creating previews.")