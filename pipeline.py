# %%
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path


# %%
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "Astro" / "Fits_Data" / "mosaic.fits"

with fits.open(data_dir) as hdul:
    #hdul.info()

    data = hdul[0].data.copy()
    header = hdul[0].header.copy()

print(data)
print(data.shape)
print(header)


# %%
plt.imshow(data, cmap='gray')
plt.show()


# %%
plt.figure(figsize=(8, 5))
plt.grid(alpha=0.3)
plt.hist(
    data.ravel(),
    bins=150,
    range=(3300, 3600),
    color="tab:blue",
)
plt.xlabel("Pixel value (ADU)")
plt.ylabel("Number of pixels")
plt.title("Background peak")
plt.show()
# %%
