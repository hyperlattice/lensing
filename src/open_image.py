from pathlib import Path

from astropy.io import fits
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
FILE_PATH = ROOT_DIR / "Castles" / "Q2237" / "FIT" / "NIC1H" / "q256.fits"

image = fits.getdata(FILE_PATH)

print("Image shape:", image.shape)
print("Data type:", image.dtype)

plt.figure(figsize=(8, 8))
plt.imshow(image, origin="lower", cmap="gray")
plt.colorbar(label="Brightness")
plt.title(str(FILE_PATH.relative_to(ROOT_DIR)))
plt.show()