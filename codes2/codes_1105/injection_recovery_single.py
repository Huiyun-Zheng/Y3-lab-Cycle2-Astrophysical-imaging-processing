"""create a uniform background or crop a single frame from the image"""

import numpy as np
from source_detection import crop_array
from get import get_fits_data
from pathlib import Path
import matplotlib.pyplot as plt
from photometry import plot_aperture_catalogue
from run_copy_photometry import run_pipeline
from photometry import plot_aperture_catalogue, AB_calibration
from injection import elliptical_gaussian
from background import fit_gaussian

# Import data
project_dir = Path(__file__).resolve().parent.parent.parent
fits_path = project_dir / "Astro" / "Fits_Data" / "mosaic.fits"
header, data = get_fits_data(fits_path)

def generate_single_background(data, type='sim'):

    if type == 'sim':
        mean, stddev = fit_gaussian(data)
        rng = np.random.default_rng(42)

        background = rng.normal(loc=mean, scale=stddev, size=(200, 200))

    elif type=='image':
        background = crop_array(data, (677, 940), (4000, 4156))

    else:
        raise Exception("Background type must be 'sim' or 'image'.")

    return background

# Perform single injection
# Gaussian background
background_sim = generate_single_background(data, type='sim')

# Real image cutout
background_real = generate_single_background(data, type='image')

# Adjust the paramters here
flux = 2e6
center = (100, 100)
semi_major = 10
semi_minor = 6
theta = np.pi/8

source = elliptical_gaussian(background_sim.shape, center, flux, semi_major, semi_minor, theta)
simulated_single_image = source + background_sim

# Detect and measure aperture
results, badmask, processed = run_pipeline(
    simulated_single_image, project_dir / "Astro" / "codes2" / "codes_1105"/'outputs', header=header
    )
fig, ax = plot_aperture_catalogue(simulated_single_image, results, show=True)

# Calculate magnitudes
mags, zp_errors = AB_calibration(results, header, show=False)
input_mag = header['MAGZPT'] - 2.5 * np.log10(flux * header['GAIN'] / header['EXPTIME'])
input_mag_errors = header['MAGZRR']
print(input_mag, mags)