# %%
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.stats import sigma_clipped_stats
from scipy import ndimage


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
image = data.astype(float)

background, median, noise = sigma_clipped_stats(
    image, sigma=3, maxiters=5
)

threshold = background + 5 * noise

print(f"Background: {background:.2f} ADU")
print(f"Background noise: {noise:.2f} ADU")
print(f"Detection threshold: {threshold:.2f} ADU")

mask = image > threshold

labels, _ = ndimage.label(
    mask, structure=np.ones((3, 3), dtype=int)
)

sizes = np.bincount(labels.ravel())
keep = sizes >= 5
keep[0] = False

source_mask = keep[labels]
print("Number of candidate sources:", np.count_nonzero(keep))

plt.figure(figsize=(6, 8))
plt.imshow(
    image, origin="lower", cmap="gray",
    vmin=background - noise,
    vmax=background + 5 * noise
)
plt.contour(source_mask, levels=[0.5], colors="red", linewidths=0.5)
plt.title("Candidate source regions")
plt.savefig(base_dir / "candidate_sources.png", dpi=300)
plt.show()
# %%
