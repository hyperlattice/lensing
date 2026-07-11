from astropy.io import fits
import matplotlib.pyplot as plt

file = "Castles/Q2237/FIT/NIC2H/q256.fits"
image = fits.getdata(file)

print("Image shape:", image.shape)
print("Data type:", image.dtype)

plt.figure(figsize=(8, 8))
plt.imshow(image, origin="lower", cmap="plasma")
plt.colorbar(label="Brightness")
plt.title(file)
plt.show()