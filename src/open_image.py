from astropy.io import fits
import matplotlib.pyplot as plt

file = "Castles/Q2237/FIT/NIC2H/q256.fits"

hdul = fits.open(file)
header = hdul[0].header

# Print HDU summary
print(f"Filename: {file}\n")
hdul.info()

# Section titles inserted before these keywords
sections = {
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

print("\n")

for card in header.cards:

    # Print section heading
    if card.keyword in sections:
        print("=" * 60)
        print(sections[card.keyword])
        print("=" * 60)

    # Preserve COMMENT and HISTORY formatting
    if card.keyword == "COMMENT":
        print(f"COMMENT  {card.value}")
        continue

    if card.keyword == "HISTORY":
        print(f"HISTORY  {card.value}")
        continue

    keyword = card.keyword
    value = card.value
    comment = card.comment

    # Print exactly like a FITS header
    if comment:
        print(f"{keyword:<8}= {repr(value):<30} / {comment}")
    else:
        print(f"{keyword:<8}= {repr(value)}")

print("\nEnd of Header")

image = fits.getdata(file)

print("Image shape:", image.shape)
print("Data type:", image.dtype)

plt.figure(figsize=(8, 8))
plt.imshow(image, origin="lower", cmap="plasma")
plt.colorbar(label="Brightness")
plt.title(file)
plt.show()