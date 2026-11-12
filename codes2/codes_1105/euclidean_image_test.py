"""
Monte Carlo Simulation of Euclidean Space Assuming Uniform Distribution
Can be used to test the detection model
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from euclidean_mc import simulate_euclidean
from get import get_fits_data
from background import fit_gaussian
from source_detection import detect_sources, deblend, filter_for_detection
from photometry import adjustable_aperture, plot_aperture_catalogue


def make_mc_image(
        magnitudes, header, background_mean, background_std,
        shape=(2048, 2048), sigma_major=2, sigma_minor=1.5, seed=43
        ):
    """Render fixed-width Gaussian sources on Gaussian sky noise."""

    rng = np.random.default_rng(seed)
    height, width = shape
    image = rng.normal(background_mean, background_std, size=shape)
    n_sources = len(magnitudes)

    # Invert the laboratory magnitude calibration
    gain = header['GAIN']
    fluxes = header['EXPTIME'] / gain * 10**(0.4 * (header['MAGZPT'] - magnitudes))

    # Keep source stamps inside the image
    radius = int(np.ceil(6 * sigma_major))
    x_centers = rng.uniform(radius + 1, width - radius - 1, n_sources)
    y_centers = rng.uniform(radius + 1, height - radius - 1, n_sources)
    angles = rng.uniform(0, np.pi, n_sources)

    for x_center, y_center, flux, theta in zip(x_centers, y_centers, fluxes, angles):
        x_min = int(np.floor(x_center)) - radius
        x_max = int(np.floor(x_center)) + radius + 1
        y_min = int(np.floor(y_center)) - radius
        y_max = int(np.floor(y_center)) + radius + 1
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        dx, dy = xx - x_center, yy - y_center
        x_rot = dx * np.cos(theta) + dy * np.sin(theta)
        y_rot = -dx * np.sin(theta) + dy * np.cos(theta)

        amplitude = flux / (2 * np.pi * sigma_major * sigma_minor)
        source = amplitude * np.exp(
            -0.5 * ((x_rot / sigma_major)**2 + (y_rot / sigma_minor)**2)
        )

        # Source shot noise in electrons; sky noise is already present
        image[y_min:y_max, x_min:x_max] += rng.poisson(source * gain) / gain

    truth = Table({
        'id': np.arange(1, n_sources + 1),
        'x_center': x_centers, 'y_center': y_centers,
        'magnitude': magnitudes, 'flux_adu': fluxes, 'theta': angles
    })
    truth.meta['image_seed'] = seed
    truth.meta['sigma_major'] = sigma_major
    truth.meta['sigma_minor'] = sigma_minor
    return image, truth


def main(
        fits_path=None, output_dir=None, show=True,
        smooth_sigma=1.5, threshold=3.5, min_area=10
        ):
    code_dir = Path(__file__).resolve().parent
    if fits_path is None:
        fits_path = code_dir.parent.parent / 'Astro' / 'Fits_Data' / 'mosaic.fits'
    if output_dir is None:
        run_name = 'euclidean_unfiltered' if smooth_sigma is None else f'euclidean_smoothed_{smooth_sigma:g}px'
        output_dir = code_dir / 'outputs' / run_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_sources = 3000
    absolute_magnitude = -20
    max_distance = 1e10  # pc
    seed = 42
    shape = (2048, 2048)

    # Measure the reference sky distribution
    header, reference = get_fits_data(fits_path)
    reference = reference[:4550, 70:2450]
    bg_mean, bg_std = fit_gaussian(
        reference[np.isfinite(reference)], bins=300, range=(3300, 3600)
    )
    del reference

    # Euclidean catalogue, then a pixel image
    distances, magnitudes = simulate_euclidean(
        n_sources, absolute_magnitude, max_distance, seed
    )
    image, truth = make_mc_image(
        magnitudes, header, bg_mean, bg_std, shape=shape, seed=seed + 1
    )
    truth['distance_pc'] = distances
    truth.meta['catalogue_seed'] = seed

    # The same detection and photometry as main.py
    labels, mean, stddev = detect_sources(
        image, threshold=threshold, min_area=min_area, smooth_sigma=smooth_sigma
    )
    detection_image = image
    if smooth_sigma is not None:
        detection_image = filter_for_detection(image, mean, smooth_sigma)
    deblended_labels = deblend(detection_image, labels)
    results = adjustable_aperture(
        image, deblended_labels, mean, stddev,
        scales=np.arange(1.5, 5.1, 0.5)
    )
    valid = (results['flag'] == 0) & np.isfinite(results['flux']) & (results['flux'] > 0)
    count_rate = results['flux'][valid] * header['GAIN'] / header['EXPTIME']
    measured_magnitudes = header['MAGZPT'] - 2.5 * np.log10(count_rate)

    fig, ax = plot_aperture_catalogue(
        image, results, figsize=(10, 10),
        filename=output_dir / 'euclidean_image_apertures.png', show=False
    )
    if show:
        plt.show()
    plt.close(fig)

    # Compare catalogue counts; this is not a source-matched completeness curve
    m_limit = absolute_magnitude + 5 * np.log10(max_distance / 10)
    thresholds = np.arange(m_limit - 9, m_limit + 0.01, 0.25)
    input_counts = np.array([np.count_nonzero(magnitudes < m) for m in thresholds])
    measured_counts = np.array([np.count_nonzero(measured_magnitudes < m) for m in thresholds])
    expected_log_counts = np.log10(n_sources) + 0.6 * (thresholds - m_limit)

    fig, ax = plt.subplots(figsize=(8, 6))
    for counts, label, color in (
            (input_counts, 'Input Monte Carlo catalogue', 'black'),
            (measured_counts, 'Detected + accepted photometry', 'tab:blue')):
        positive = counts > 0
        ax.plot(thresholds[positive], np.log10(counts[positive]), 'o-',
                markersize=4, color=color, label=label)
    ax.plot(thresholds, expected_log_counts, '--', color='gray', label='Euclidean slope = 0.6')
    ax.set_xlabel('Magnitude threshold, m')
    ax.set_ylabel('log10 N(<m)')
    ax.set_title(f'Image test: {n_sources} sources, smoothing sigma = {smooth_sigma}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'euclidean_image_counts.png', dpi=180)
    if show:
        plt.show()
    plt.close(fig)

    # Save the image without copying the real sky WCS
    simulated_header = fits.Header()
    for key in ('EXPTIME', 'GAIN', 'MAGZPT', 'MAGZRR', 'FILTER'):
        simulated_header[key] = header[key]
    simulated_header['BUNIT'] = 'ADU'
    simulated_header['SIMULATE'] = True
    simulated_header['SIMSEED'] = seed
    simulated_header['IMGSEED'] = seed + 1
    simulated_header['SMTHSIG'] = 0.0 if smooth_sigma is None else float(smooth_sigma)
    simulated_header['DETSIG'] = float(threshold)
    simulated_header['DETAREA'] = int(min_area)
    simulated_header['BGMEAN'] = bg_mean
    simulated_header['BGSIGMA'] = bg_std
    fits.writeto(output_dir / 'euclidean_image.fits', image.astype(np.float32),
                 header=simulated_header, overwrite=True)
    truth.write(output_dir / 'euclidean_image_truth.ecsv', overwrite=True)
    results.write(output_dir / 'euclidean_image_results.ecsv', overwrite=True)

    print(f'Input: {n_sources}; after deblend: {len(results)}; accepted magnitudes: {np.count_nonzero(valid)}')
    print(f'Background: input {bg_mean:.3f}, fitted {mean:.3f} ADU')
    print(f'Noise: input {bg_std:.3f}, fitted {stddev:.3f} ADU')
    print(f'Output directory: {output_dir}')
    return image, truth, results


if __name__ == '__main__':
    main()