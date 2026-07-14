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

# Output text file
output_file = "q256_header.txt"

with open(output_file, "w", encoding="utf-8") as f:

    # Helper function to print to console AND save to file
    def out(text=""):
        print(text)
        f.write(text + "\n")

    # Print filename
    out(f"Filename: {file}\n")

    # Capture hdul.info() output
    hdul.info(output=f)

    # Also show it in the terminal
    hdul.info()

    out("\n")

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

    for card in header.cards:

        # Print section heading
        if card.keyword in sections:
            out("=" * 60)
            out(sections[card.keyword])
            out("=" * 60)

        # Preserve COMMENT and HISTORY formatting
        if card.keyword == "COMMENT":
            out(f"COMMENT  {card.value}")
            continue

        if card.keyword == "HISTORY":
            out(f"HISTORY  {card.value}")
            continue

        keyword = card.keyword
        value = card.value
        comment = card.comment

        # Print exactly like a FITS header
        if comment:
            out(f"{keyword:<8}= {repr(value):<30} / {comment}")
        else:
            out(f"{keyword:<8}= {repr(value)}")

    out("\nEnd of Header")

    image = fits.getdata(file)

    out(f"Image shape: {image.shape}")
    out(f"Data type: {image.dtype}")

print(f"\nHeader saved to: {output_file}")

image = fits.getdata(file)

print("Image shape:", image.shape)
print("Data type:", image.dtype)

plt.figure(figsize=(8, 8))
plt.imshow(image, origin="lower", cmap="plasma")
plt.colorbar(label="Brightness")
plt.title(file)
plt.show()