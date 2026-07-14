from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

image_path = Path("data/raw/castles/Q2237/FIT/NIC2H/q256.fits")

with fits.open(image_path) as hdul:
    image = hdul[0].data

print(image.shape)
print(image.dtype)

print("Minimum:", image.min())
print("Maximum:", image.max())
print("Mean:", image.mean())
print("Median:", image.mean())

print("NaNs:", np.isnan(image).sum())
print("Infinite:", np.isinf(image).sum())

clean_image = np.nan_to_num(
    image,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

print("NaNs after cleaning:", np.isnan(clean_image).sum())
print("Infinite after cleaning:", np.isinf(clean_image).sum())

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(image, origin="lower", cmap="gray")
ax[0].set_title("Original")

ax[1].imshow(clean_image, origin="lower", cmap="gray")
ax[1].set_title("Cleaned")

plt.show()