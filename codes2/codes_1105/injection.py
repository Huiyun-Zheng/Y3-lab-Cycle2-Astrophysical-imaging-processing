import numpy as np
from source_detection import crop_array
from get import get_fits_data
from pathlib import Path
import matplotlib.pyplot as plt
from source_detection import detect_sources, deblend
from photometry import fixed_aperture_photometry, adjustable_aperture, plot_aperture_catalogue
from run_copy_photometry import run_pipeline, recovery_rows
from photometry import plot_aperture_catalogue, AB_calibration

def elliptical_gaussian(
        shape, center, flux, sigma_major, sigma_minor, theta=0
        ):
    """Return a Gaussian source in ADU; theta is in radians from +x."""

    height, width = shape
    x_center, y_center = center

    yy, xx = np.ogrid[:height, :width]
    dx = xx - x_center
    dy = yy - y_center

    # Coordinates along the ellipse axes
    x_rot = dx * np.cos(theta) + dy * np.sin(theta)
    y_rot = -dx * np.sin(theta) + dy * np.cos(theta)

    # Continuous total flux, sampled at pixel centres
    amplitude = flux / (2 * np.pi * sigma_major * sigma_minor)
    source = amplitude * np.exp(
        -0.5 * ((x_rot / sigma_major)**2 + (y_rot / sigma_minor)**2)
    )

    return source

def generate_background(type='uniform', image=None):

    if type=='uniform':
        background = np.full((300, 200), fillvalue=0)

    if type=='image':
        background = crop_array(image, (677, 940), (4000, 4156))

    return background

from source_detection_copy import run_detection

# Create the simulated image
project_dir = Path(__file__).resolve().parent.parent.parent
fits_path = project_dir / "Astro" / "Fits_Data" / "mosaic.fits"
header, data = get_fits_data(fits_path)
background = generate_background(type='image', image=data)
flux = 2e6
sim_source_1 = elliptical_gaussian(background.shape, (100, 80), flux, 50, 20, np.pi / 6)
sim_image_1 = background + sim_source_1

# Detect and measure apertures
results, badmask, processed = run_pipeline(sim_image_1, project_dir / "Astro" / "codes2" / "codes_1105"/'outputs', header=header)

# Plot near the injected source
fig, ax = plot_aperture_catalogue(
    sim_image_1, results, show=False
)

# Calculate magnitude
mags, zp_errors = AB_calibration(
    results, header, show=False
)

# At injection
ZP = header['MAGZPT']
ZP_error = header['MAGZRR']
input_mag = ZP - 2.5 * np.log10(flux * header['GAIN'] / header['EXPTIME'])

print(input_mag)
print(min(mags))