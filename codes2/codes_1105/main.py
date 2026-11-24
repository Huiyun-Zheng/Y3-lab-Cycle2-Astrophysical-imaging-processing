from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from get import get_fits_data
from bleeding import (mask_whole_bleeding_region, mask_saturated)
from stars import find_stars
from aperture_conversion import aperture_center_shape, plot_source_aperture, plot_aperture_scales, optimize_scale
from source_detection import detect_sources, deblend, crop_array, plot_detected_sources
from photometry import fixed_aperture_photometry, AB_calibration, population_above_magnitude, plot_aperture_catalogue, adjustable_aperture
from astropy.io import fits
from psf_star_detection import measure_psf_widths, select_psf_candidates, plot_psf_candidates

from selection_diagnostics import selection_diagnostics

saturation = 36000
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Read the file
project_dir = Path(__file__).resolve().parent.parent.parent
fits_path = project_dir / "Astro" / "Fits_Data" / "mosaic.fits"
header, data = get_fits_data(fits_path)

# Crop once before background fitting, detection, deblending, and plotting.
crop_x_range = (100, 2400)
crop_y_range = (500, 4500)	
data = crop_array(data, crop_x_range, crop_y_range)
x_offset, y_offset = crop_x_range[0], crop_y_range[0]

# Find sources by brightest pixel method
print(type(data), data.shape, data.dtype)
detection_threshold = 1.5
min_area = 12
labels, mean, stddev = detect_sources(
    data, threshold=detection_threshold, min_area=min_area
)
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

star_masks = np.zeros(data.shape, dtype=bool)

for center in star_centers_cropped:
    mask = find_stars(data, labels, center)
    star_masks |= mask > 0

plot_detected_sources(data, star_masks, filename=output_dir / 'detected_stars_overlay.svg')

# deblend the sources
deblended_labels = deblend(data, labels)
#plot_detected_sources(data, deblended_labels, contour='scatter')

# Mask saturated regions
saturated_mask = mask_saturated(data, labels, saturation=saturation)
plot_detected_sources(data, saturated_mask, filename=output_dir / 'saturated_regions_overlay.svg')

# Crop the picture to avoid edge disturbances
small_x_range = (1000 - x_offset, 1500 - x_offset)
small_y_range = (1000 - y_offset, 1500 - y_offset)
small_data = crop_array(data, small_x_range, small_y_range)
small_labels = crop_array(deblended_labels, small_x_range, small_y_range)
plot_detected_sources(small_data, small_labels)

# Mask all the bleeding sources
bleeding_center = (
    center_original[0] - x_offset,
    center_original[1] - y_offset
)
whole_mask = mask_whole_bleeding_region(
    data, bleeding_center, 40, 70, mean, stddev,
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


results = adjustable_aperture(
    data, deblended_labels, mean, stddev,
    scales=np.arange(1.5, 5.1, 0.5),
    mask=star_masks | saturated_mask | whole_mask
)

# Check photometry selection before PSF classification
selection_summary, selection_counts, selection_catalogue = selection_diagnostics(
    data, results, mean, header,
    output_dir=output_dir / "selection_diagnostic"
)

# Classify this run's sources
psf_catalogue = measure_psf_widths(
    data, deblended_labels, results, header,
    mask=star_masks | saturated_mask | whole_mask
)
psf_catalogue = select_psf_candidates(psf_catalogue)
psf_catalogue.meta['crop_x'] = list(crop_x_range)
psf_catalogue.meta['crop_y'] = list(crop_y_range)
psf_catalogue.meta['detection_threshold'] = float(detection_threshold)
psf_catalogue.meta['detection_min_area'] = int(min_area)
psf_catalogue.meta['mask_note'] = 'main.py manual star, saturation and bleeding masks applied'
psf_catalogue['x_original'] = psf_catalogue['x_center'] + x_offset
psf_catalogue['y_original'] = psf_catalogue['y_center'] + y_offset

star_ids = psf_catalogue['label'][
    psf_catalogue['source_class'] == 'star_candidate'
]
psf_star_mask = np.isin(deblended_labels, star_ids)
is_star = np.isin(results['label'], star_ids)
results_for_counts = results[~is_star]
print(f'PSF star candidates excluded: {np.count_nonzero(is_star)}')

# Save matching catalogues and candidate mask
psf_dir = output_dir / 'psf_stars'
psf_dir.mkdir(parents=True, exist_ok=True)
psf_catalogue.write(psf_dir / 'psf_catalogue.ecsv', overwrite=True)
psf_catalogue[psf_catalogue['source_class'] == 'star_candidate'].write(
    psf_dir / 'star_candidates.csv', overwrite=True
)
mask_header = fits.Header()
mask_header['XOFFSET'] = x_offset
mask_header['YOFFSET'] = y_offset
mask_header['COMMENT'] = 'Provisional star-candidate segments in cropped coordinates'
fits.writeto(
    psf_dir / 'star_candidate_mask.fits', psf_star_mask.astype('uint8'),
    header=mask_header, overwrite=True
)
plot_psf_candidates(
    data, psf_catalogue, psf_dir,
    x_offset=x_offset, y_offset=y_offset, show=False
)
results_for_counts.write(output_dir / 'photometry_for_counts.ecsv', overwrite=True)

# Plot the catalogue used for counting
plot_aperture_catalogue(
    data, results_for_counts,
    filename=output_dir / 'detected_apertures.png'
)

flux = np.asarray(results_for_counts['flux'])
valid_flux = flux[np.isfinite(flux) & (flux > 0)]
if valid_flux.size > 0:
    print(valid_flux.min(), valid_flux.max())
else:
    raise ValueError('No positive valid flux measurements remain')

mags, mag_errs = AB_calibration(results_for_counts, header)

mag_thre, populations = population_above_magnitude(mags)
populations = np.array(populations)
positive = populations > 0
fitting_mask = (mag_thre <= 25) & (populations > 0)

coeffs, cov = np.polyfit(
    mag_thre[fitting_mask],
    np.log10(populations[fitting_mask]),
    deg=1, cov=True
)

fit = np.poly1d(coeffs)
plt.plot(mag_thre[fitting_mask], fit(mag_thre[fitting_mask]), c='red')
plt.scatter(mag_thre[positive], np.log10(populations[positive]), c='blue')
plt.title('N - m relationship')
plt.xlabel('Magnitude Thresholds')
plt.xlim((10, 30))
plt.ylabel('log10 N(m < threshold)')
plt.show()

print(coeffs, np.sqrt(np.diag(cov)))
