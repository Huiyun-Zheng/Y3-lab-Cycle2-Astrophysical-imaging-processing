from pathlib import Path
import numpy as np
from get import get_fits_data
from bleeding import (mask_whole_bleeding_region, mask_saturated)
from stars import find_stars
from aperture_conversion import aperture_center_shape, plot_source_aperture, plot_aperture_scales, optimize_scale
from source_detection import detect_sources, deblend, crop_array, plot_detected_sources
from photometry import fixed_aperture_photometry, AB_calibration

saturation = 36000

# Read the file
project_dir = Path(__file__).resolve().parent.parent.parent
fits_path = project_dir / "Astro" / "Fits_Data" / "mosaic.fits"
header, data = get_fits_data(fits_path)

# Crop once before background fitting, detection, deblending, and plotting.
crop_x_range = (70, 2450)
crop_y_range = (0, 4550)
data = crop_array(data, crop_x_range, crop_y_range)
x_offset, y_offset = crop_x_range[0], crop_y_range[0]

# Find sources by brightest pixel method
print(type(data), data.shape, data.dtype)
labels, mean, stddev = detect_sources(data, threshold=3, min_area=5)
plot_detected_sources(data, labels)

# Example image coordinates
center_original = (775, 3320)
lower_original = 3200
upper_original = 3430

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

# Crop the picture to avoid edge disturbances
small_x_range = (1000 - x_offset, 1500 - x_offset)
small_y_range = (1000 - y_offset, 1500 - y_offset)
small_data = crop_array(data, small_x_range, small_y_range)
small_labels = crop_array(deblended_labels, small_x_range, small_y_range)
plot_detected_sources(small_data, small_labels)

# Mask all the bleeding sources
whole_mask = mask_whole_bleeding_region(
    data, center, 40, 70, mean, stddev,
    half_width=15,
    upper_bound=upper_original - y_offset,
    lower_bound=lower_original - y_offset)

#plot_detected_sources(data, whole_mask)


# Assign elliptic aperture to the detected sources
centers, ellipse = aperture_center_shape(data, labels, mean, stddev)
plot_source_aperture(data, labels, mean, stddev, target_label=89, scale=2.5)

# Optimize aperture size
scales, fluxes = optimize_scale(
    data, labels, mean, stddev, target_label=89
)

# Visualize to compare different scales
#plot_aperture_scales(data, labels, mean, stddev, target_label=4, scales=(1.5, 2, 3, 4), figsize=(10, 10))


results = fixed_aperture_photometry(
    data, deblended_labels, mean, stddev,
    scale=3.0,
    mask=star_masks | saturated_mask | whole_mask
)

good_results = results[results["flag"] == 0]
#print(good_results)

mags, mag_errs = AB_calibration(results, header)