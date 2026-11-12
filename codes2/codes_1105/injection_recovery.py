# Used to perform the injection-recovery simulation to examine the completeness of the detection method
# Single random injection version

import numpy as np
from astropy.table import Table
from source_detection import crop_array
from get import get_fits_data
from pathlib import Path
import matplotlib.pyplot as plt
from source_detection import detect_sources, deblend, crop_array
from photometry import fixed_aperture_photometry, plot_aperture_catalogue
from injection import elliptical_gaussian
from match_recovery import match_recovery
from run_copy_photometry import run_pipeline, recovery_rows
from photometry import plot_aperture_catalogue, AB_calibration
from source_detection_copy import run_detection

seed = 42
rng = np.random.default_rng(seed)

# Number of simulated source
sim_source_count = 100

# Read the file
project_dir = Path(__file__).resolve().parent.parent.parent
fits_path = project_dir / "Astro" / "Fits_Data" / "mosaic.fits"
header, data = get_fits_data(fits_path)

def make_grid(shape, spacing, margin):
    """Create grids on the image to select different regions for tests."""
    height, width = shape

    x = np.arange(margin, width - margin, spacing)
    y = np.arange(margin, height - margin, spacing)
    xx, yy = np.meshgrid(x, y)

    return np.column_stack((xx.ravel(), yy.ravel()))

# Run the pipeline for completeness tests
catalogue, excluded, processed = run_detection(data, project_dir / "Astro" / "codes2" / "codes_1105"/'outputs')

spacing = 200
margin = 200
positions = make_grid(data.shape, spacing=spacing, margin=margin)
injected_rows = []
detected_rows = []

for injected_id, (grid_x, grid_y) in enumerate(positions, start=1):

    # Crop a background region
    x_min = int(grid_x - spacing // 2)
    x_max = int(grid_x + spacing // 2)
    y_min = int(grid_y - spacing // 2)
    y_max = int(grid_y + spacing // 2)

    background = crop_array(data, (x_min, x_max), (y_min, y_max))


    # Inject sources at this position
    # Random source parameters
    flux = rng.uniform(1e3, 2e4, sim_source_count)
    sigma_major = rng.uniform(2, 5, sim_source_count)
    sigma_minor = sigma_major * rng.uniform(0.5, 1, sim_source_count)
    theta = rng.uniform(0, np.pi, sim_source_count)

    # Random position
    height, width = background.shape
    margin = 5 * sigma_major

    x_center = rng.uniform(margin, width - margin, sim_source_count)
    y_center = rng.uniform(margin, height - margin, sim_source_count)
    center = (x_center, y_center)

    # Generate and inject
    sources = elliptical_gaussian(
    background.shape, center, flux,
    sigma_major, sigma_minor, theta
    )

    sim_image = np.full(sim_source_count, background.astype(float)) + sources

    # Record the injected source
    injected_rows.append((
        injected_id,
        center[0], center[1],
        x_min + center[0], y_min + center[1],
        flux, sources.sum(),
        sigma_major, sigma_minor, theta
    ))

    # Run the pipeline on the simulated image and calculate completeness
    detected_centers, output_fluxes = run_pipeline(sim_image)

    for detection_id, (position, measured_flux) in enumerate(
            zip(detected_centers, output_fluxes), start=1):

        detected_x, detected_y = position

        detected_rows.append((
            injected_id,
            detection_id,
            detected_x, detected_y,
            x_min + detected_x, y_min + detected_y,
            measured_flux
        ))


injected = Table(
    rows=injected_rows,
    names=(
        'injected_id',
        'x_local', 'y_local',
        'x_in', 'y_in',
        'flux_in', 'flux_added',
        'sigma_major', 'sigma_minor', 'theta'
    )
)

detected = Table(
    rows=detected_rows,
    names=(
        'trial_id', 'detection_id',
        'x_local', 'y_local',
        'x_out', 'y_out',
        'flux_out'
    )
)