import astropy as ap
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from matplotlib import image
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, AsinhNorm
from get import get_fits_data
from background import plot_histogram, fit_gaussian
from astropy.visualization import (AsinhStretch, 
                                   LogStretch, 
                                   ImageNormalize, 
                                   ZScaleInterval,)
from photutils.segmentation import deblend_sources, SegmentationImage
from photutils.aperture import CircularAperture
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from bleeding import (spot_linear_sources,
                    bleeding_band_mask, 
                    mask_whole_bleeding_region,
                    find_halo_radius,
                    mask_saturated)
from stars import find_stars

saturation = 36000

def detect_sources(
    data, bins=300, hist_range=(3300, 3600),
    threshold=3, min_area=5
):
    """
    Assuming a uniform background. Labels have the same shape as input data.
    """

    valid = np.isfinite(data)
    mean, stddev = fit_gaussian(data[valid], bins=bins, range=hist_range)
    if not np.isfinite(mean) or not np.isfinite(stddev) or stddev <= 0:
        raise ValueError("Invalid background fit")

    # Find pixels that are above the threshold
    mask = valid & (data > mean + threshold * stddev)

    # Label connected components in the mask
    connectivity = ndi.generate_binary_structure(2, 2)  # 8-connectivity
    print("Mask shape:", mask.shape)
    print('data shape:', data.shape)
    print('connectivity shape:', connectivity.shape)
    labels, _ = ndi.label(mask, structure=connectivity)

    # Delete regions with fewer pixels than min_area
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False                 # 0 is background, do not keep
    mask = keep[labels]

    # Relabel the mask after removing small regions
    labels, _ = ndi.label(mask, structure=connectivity)

    return labels, mean, stddev


def deblend(data, labels):
    """
    Deblends overlapping sources in the image.
    """
    segment_map = SegmentationImage(labels)
    segment_map_deblended = deblend_sources(
        data, segment_map, n_pixels=10, n_levels=32, contrast=0.001
    )
    labels_deblended = segment_map_deblended.data

    print("before deblending:", segment_map.nlabels)
    print("after deblending:", segment_map_deblended.nlabels)

    return labels_deblended

def crop_array(array, x_range, y_range):
    """
    x_range: (int, int)
    y_range: (int, int)
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    return array[y_min:y_max, x_min:x_max]

def plot_detected_sources(data, labels, contour='no_deblending', filename=None):
    """
    Plots the original data with detected sources overlaid.

    Parameters:
    data (numpy.ndarray): The original image data.
    labels (numpy.ndarray): The labeled mask of detected sources.
    """

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(data)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='gray', origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar(label='Pixel Value')
    plt.title('Detected Sources Overlay')

    # Overlay detected sources
    if contour == 'no_deblending':
        plt.contour(
            labels > 0,
            levels=[0.5],
            colors='red',
            linewidths=0.5
        )

    if contour == 'contour':
        for label_id in np.unique(labels):
            if label_id == 0:
                continue

            plt.contour(
                labels == label_id,
                levels=[0.5],
                colors='red',
                linewidths=0.5
            )

    elif contour == 'scatter':
        edges = np.zeros(labels.shape, dtype=bool)

        different = labels[:, 1:] != labels[:, :-1]
        edges[:, 1:] |= different
        edges[:, :-1] |= different

        different = labels[1:, :] != labels[:-1, :]
        edges[1:, :] |= different
        edges[:-1, :] |= different

        edges &= labels > 0
        del different
        y, x = np.nonzero(edges)
        plt.scatter(x, y, color="red", s=0.5, linewidths=0)

    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_single_source(labels, label_center):
    """
    Plots a single source from the labeled mask.
    label_center: (int, int) - The (y, x) coordinates of the source center in the labels array.
    """
    target = small_labels[label_center[0], label_center[1]]

    plt.xlabel('X Pixel (0 offset)')
    plt.ylabel('Y Pixel (0 offset)')
    plt.title(f'Source with label {target}')
    plt.imshow(
        small_labels == target,
        cmap="gray", origin="lower",
        interpolation="nearest"
    )
    plt.show()


project_dir = Path(__file__).resolve().parent.parent.parent
fits_path = project_dir / "Astro" / "Fits_Data" / "mosaic.fits"
header, data = get_fits_data(fits_path)

# Crop once before background fitting, detection, deblending, and plotting.
crop_x_range = (70, 2450)
crop_y_range = (0, 4550)
data = crop_array(data, crop_x_range, crop_y_range)
x_offset, y_offset = crop_x_range[0], crop_y_range[0]

print(type(data), data.shape, data.dtype)
labels, mean, stddev = detect_sources(data, threshold=3, min_area=5)
plot_detected_sources(data, labels)

# Original-image coordinates
center_original = (775, 3320)
lower_original = 3200
upper_original = 3430

# Cropped-image coordinates
center = (
    center_original[0] - x_offset,
    center_original[1] - y_offset
)

bleed_mask = bleeding_band_mask(
    data, center, 40, mean, stddev,
    half_width=15,
    upper_bound=upper_original - y_offset,
    lower_bound=lower_original - y_offset
)

mask_radius = find_halo_radius(
    data, center, 40,
    mean, stddev,
    bleed_mask=bleed_mask,
    max_radius=150
)

print("Halo radius:", mask_radius)

plot_detected_sources(data, bleed_mask)

# Mask stars
star_centers = [
    (775, 3320),
    (973, 2773),
    (907, 2280),
    (2137, 3755),
    (2090, 1423),
    # (2467, 3415),  # Outside the current crop
    (2132, 2306),
    (1425, 3208)
]

star_centers_cropped = [
    (x - x_offset, y - y_offset) for x, y in star_centers
]

star_centers_cropped = [
    (x - x_offset, y - y_offset) for x, y in star_centers
]

star_masks = np.zeros(data.shape, dtype=bool)

for center in star_centers_cropped:
    mask = find_stars(data, labels, center)
    star_masks |= mask > 0

plot_detected_sources(data, star_masks, filename='outputs/detected_stars_overlay.svg')

# deblend the sources
deblended_labels = deblend(data, labels)
#plot_detected_sources(data, deblended_labels, contour='scatter')

# Mask saturated regions
saturated_mask = mask_saturated(data, labels, saturation=saturation)
plot_detected_sources(data, saturated_mask, filename='outputs/saturated_regions_overlay.svg')

angles, rhos = spot_linear_sources(
    data, deblended_labels,
    # Convert these original-image coordinates to the cropped-image frame.
    x_range=(700 - x_offset, 900 - x_offset),
    y_range=(3200 - y_offset, 3450 - y_offset),
    all_possibilities=False
)

lines = spot_linear_sources(
    data, deblended_labels,
    x_range=(700 - x_offset, 900 - x_offset),
    y_range=(3200 - y_offset, 3450 - y_offset),
    threshold=10,
    line_length=150,
    line_gap=3
)

small_x_range = (1000 - x_offset, 1500 - x_offset)
small_y_range = (1000 - y_offset, 1500 - y_offset)
small_data = crop_array(data, small_x_range, small_y_range)
small_labels = crop_array(deblended_labels, small_x_range, small_y_range)
plot_detected_sources(small_data, small_labels)

# Local background section
x0, y0 = map(int, center)
outer_radius = 300

section = data[
    y0 - outer_radius:y0 + outer_radius + 1,
    x0 - outer_radius:x0 + outer_radius + 1
]

yy, xx = np.ogrid[
    -outer_radius:outer_radius + 1,
    -outer_radius:outer_radius + 1
]
distance = np.hypot(xx, yy)

# Background annulus
sky_ring = (distance >= 200) & (distance < outer_radius)

sky_ring &= ~bleed_mask[
    y0 - outer_radius:y0 + outer_radius + 1,
    x0 - outer_radius:x0 + outer_radius + 1
]

_, local_bg, local_std = sigma_clipped_stats(
    section[sky_ring], sigma=3, maxiters=5
)

mask_radius = find_halo_radius(
    data, center, 40,
    local_bg, local_std,
    bleed_mask=bleed_mask,
    max_radius=150
)

print("Halo radius:", mask_radius)

whole_mask = mask_whole_bleeding_region(
    data, center, 40, 70, mean, stddev,
    half_width=15,
    upper_bound=upper_original - y_offset,
    lower_bound=lower_original - y_offset)

plot_detected_sources(data, whole_mask)