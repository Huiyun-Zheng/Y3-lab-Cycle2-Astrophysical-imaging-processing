# %%
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Circle
from scipy import ndimage as ndi

# %%
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "Astro" / "Fits_Data" / "mosaic.fits"

with fits.open(data_dir) as hdul:

    header = hdul[0].header.copy()
    data = hdul[0].data.copy()

print(data)
print(data.shape)  # 4611 rows, 2570 columns

image = np.asarray(data, dtype=float)
ny, nx = image.shape
# %%
# Detection parameters
detec_sigma = 5  # Require the candidate peak to exceed the background by five standard deviations
aperture_radius = 8  # Radius in pixels, used for preliminary aperture photometry
mask_radius = 12  # In pixels, set the exclusion radius used to avoid detecting the same source multiple times
min_pixels = 5  # Require at least five bright pixels in the local 5 * 5 neighbourhood to consider a candidate peak as a source

# Unusable regions
bad_mask = ~np.isfinite(image)

edge = 30
bad_mask[:edge, :] = True
bad_mask[-edge:, :] = True
bad_mask[:, :edge] = True
bad_mask[:, -edge:] = True

yy, xx = np.ogrid[:ny, :nx]  # Create row and column coordinate arrays for broadcasting

# Intial estimates for the foreground star and the vertical bleeding
bad_mask |= (xx - 1430) ** 2 + (yy - 3200) ** 2 <= 320 ** 2  # Exclude the upper central foreground
bad_mask[:, 1420:1450] = True  # Exclude the central vertical bleeding strip
bad_mask[0:450, 1100:1650] = True  # Exclude the artefacts (i.e the horizontal bleeding in the lower part)
# %%
# Background and noise estimation
values = image[~bad_mask]

if values.size == 0:
    raise ValueError("No usable pixels remain.")

# Repeat the background and noise estimation and outlier rejection five times
for _ in range(5):
    background = np.median(values)
    # Estimate background noise using MAD, scaled by 1.4826 to approximate the standard deviation for Gaussian noise.
    noise = 1.4826 * np.median(np.abs(values - background))

    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Estimated noise is not finite or non-positive.")

    # Three-sigma clipping to remove outliers
    condition = np.abs(values - background) < 3 * noise
    values = values[condition]

background = np.median(values)  # Final background from the clipped sample
noise = np.std(values)  # Take the std of the clipped sample as an approximate noise level.

if not np.isfinite(noise) or noise <= 0:
    raise ValueError("Final estimated background noise is not finite or non-positive.")

threshold = background + detec_sigma * noise

print(f"Estimated background: {background: .2f} ADU")
print(f"Estimated noise: {noise: .2f} ADU")
print(f"Detection threshold: {threshold: .2f} ADU")
# %%
# Sort candidate pixels by brightness
candidate_mask = (image > threshold) & ~bad_mask
candidate_indicies = np.flatnonzero(candidate_mask.ravel())
candidate_values = image.ravel()[candidate_indicies]
brightness_order = np.argsort(candidate_values)[::-1]  #
candidate_indices = candidate_indicies[brightness_order]

# Inirialise a mask to identify pixels that have already been processed
processed_mask = np.zeros(image.shape, dtype=bool)

sources = []
# %%
# Main detection loop (brighteset pixel method)
for index in candidate_indices:
    y, x = np.unravel_index(index, image.shape)  # Convert the flattened index back to 2D coordinates

    if bad_mask[y, x] or processed_mask[y, x]:
        continue  # Skip if the pixel is in a bad region or has already been processed

    # Extract a 5 * 5 neighbourhood around the candidate pixel, ensuring stay within the image bounds
    neighbourhood = image[max(0, y-2) : min(ny, y+3), max(0, x-2): min(nx, x+3)]
    # Count the number of pixels in the neighbourhood that exceed the detection threshold
    n_bright = np.count_nonzero(neighbourhood > threshold)

    # Reject peaks with insufficient neighbouring bright pixels
    if n_bright < min_pixels:
        processed_mask[y, x] = True  # Mark this pixel as processed
        continue

    r = aperture_radius

    # Check whether the complete aperture around the candidate pixel is within the image bounds
    if (y - r < 0) or (y + r >= ny) or (x - r < 0) or (x + r >= nx):
        processed_mask[y,x] = True
        continue

    # Define a square enclosing the circular aperture
    ys = slice(y - r, y + r + 1)
    xs = slice(x - r, x + r + 1)

    local_y, local_x = np.mgrid[-r : r + 1, -r : r + 1]  # Create coordinates relative to the candidate peak
    aperture = local_x ** 2 + local_y ** 2 <= r ** 2  # Slect pixels lying within the circular aperture

    # Check whether any aperture pixel belongs to an unusavble region
    if np.any(bad_mask[ys, xs][aperture]):
        processed_mask[y, x] = True
        continue

    # Flag contact with any previously processed pixels, not necessarily another physical source
    overlap = np.any(processed_mask[ys, xs][aperture])

    # Calculate the flux of the candidate source (background subtracted)
    flux = np.sum(image[ys, xs][aperture] - background)

    source = {}
    source["x"] = x
    source["y"] = y
    source["peak"] = float(image[y, x])  # Record the original peak count, including background
    source["aperture_flux"] = float(flux)
    source["overlap"]  = bool(overlap)
    sources.append(source)

    r_1 = mask_radius  # Switch to the larger exclusion radius to avoid repeated detections
    y0, y1 = max(0, y - r_1), min(ny, y + r_1 +1)
    x0, x1 = max(0, x - r_1), min(nx, x + r_1 +1)

    local_y_1, local_x_1 = np.mgrid[y0 : y1, x0 : x1]
    region = (local_x_1 - x) ** 2 + (local_y_1 - y) ** 2 <= r_1 ** 2

    # Add the exclusion circle to the existing mask while preserving previously masked pixels.
    processed_mask[y0 : y1, x0 : x1] |= region

print(f"Detected {len(sources)} sources.")
# %%
# Visualisation of the detected sources
fig, ax = plt.subplots(figsize=(6, 8))

ax.imshow(
    image,
    origin="lower",
    cmap="gray",
    vmin=background-noise,
    vmax=background+8*noise
)

overlay = np.ma.masked_where(~bad_mask, bad_mask)  # Hide usable pixels in the overlay so only excluded regions are shown.

ax.imshow(
    overlay,
    origin="lower",
    cmap="bone",
    vmin=0,
    vmax=1,
    alpha=0.35
)

for source in sources:
    colour = "red" if source["overlap"] else "blue"

    circle = Circle(
        (source["x"], source["y"]),
        aperture_radius,
        fill=False,
        edgecolor=colour,
        linewidth=0.6
    )

    ax.add_patch(circle)

ax.set_xlabel("x / pixel")
ax.set_ylabel("y / pixel")
ax.set_title(f"{len(sources)} candidate sources")
plt.tight_layout()
plt.savefig("output/preliminary_sources.png", dpi=300)
plt.show()
# %%
print(noise)
# %%
