import numpy as np
from source_detection import crop_array
from get import get_fits_data
from pathlib import Path
import matplotlib.pyplot as plt
from source_detection import detect_sources, deblend
from photometry import fixed_aperture_photometry, plot_aperture_catalogue

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


# Create the simulated image
project_dir = Path(__file__).resolve().parent.parent.parent
fits_path = project_dir / "Astro" / "Fits_Data" / "mosaic.fits"
header, data = get_fits_data(fits_path)
background = generate_background(type='image', image=data)
sim_source_1 = elliptical_gaussian(background.shape, (100, 80), 2e5, 30, 10, np.pi / 6)
sim_image_1 = background + sim_source_1

# Detect and deblend
labels, mean, stddev = detect_sources(
    sim_image_1, threshold=3, min_area=5
)
deblended_labels = deblend(sim_image_1, labels)

# Measure apertures
results = fixed_aperture_photometry(
    sim_image_1, deblended_labels, mean, stddev, scale=3
)

# Plot near the injected source
fig, ax = plot_aperture_catalogue(
    sim_image_1, results,
    x_range=(0, 250),
    y_range=(0, 200),
    figsize=(10, 8),
    show=False
)

ax.plot(100, 50, '+', color='lime', markersize=12)
plt.show()
plt.close(fig)