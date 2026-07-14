from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt

image_path = Path("data/raw/castles/Q2237/FIT/NIC2H/q256.fits")

with fits.open(image_path) as hdul:
    image = hdul[0].data

print(image.shape)
print(image.dtype)

plt.figure(figsize=(6,6))
plt.imshow(image, origin="lower", cmap="gray")
plt.colorbar()

plt.show()