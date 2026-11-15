"""Compare both existing pipelines on the same seeded Gaussian-source image."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
import numpy as np
from astropy.io import fits
from astropy.table import Table

from euclidean_mc import simulate_euclidean_abs
from injection_recovery_many import elliptical_gaussian
from match_many_recovery import match_many_recovery
from photometry import fixed_aperture_photometry
from run_copy_photometry import run_pipeline
from source_detection import detect_sources, deblend
from source_detection_copy import Settings


def compare_recovery(truth, results):
    """Keep positional recovery separate from usable photometry."""
    
    detected = Table()
    detected['frame_id'] = np.ones(len(results), dtype=int)
    detected['detection_id'] = np.arange(1, len(results) + 1)
    detected['x_local'] = results['x_center']
    detected['y_local'] = results['y_center']
    detected['flux_out'] = results['flux']
    finite = np.isfinite(detected['x_local']) & np.isfinite(detected['y_local'])
    detected = detected[finite]
    original = Table(names=('frame_id', 'x_local', 'y_local'),
                     dtype=('i8', 'f8', 'f8'))
    matches = match_many_recovery(truth, detected, original, match_radius=3)
    unique = (matches['n_candidates'] == 1) & (matches['n_claimants'] == 1)
    selected = np.zeros(len(matches), dtype=bool)
    for i in np.flatnonzero(unique):
        row = results[int(matches['detection_id'][i]) - 1]
        selected[i] = (row['flag'] == 0 and np.isfinite(row['flux'])
                       and row['flux'] > 0)
    matches['m_in'] = truth['m_in']
    matches['position_recovered'] = unique
    matches['usable_recovered'] = selected
    matches['crowded_input'] = matches['n_injected_near'] > 0
    matches.meta['recovery_note'] = (
        'Position recovery requires a mutually unique candidate within 3 pixels. '
        'Usable recovery also requires valid pipeline photometry. Neither guarantees '
        'an uncontaminated flux in a crowded field; inspect crowded_input and status.'
    )
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--fits', type=Path,
                        default=here.parents[1] / 'Astro' / 'Fits_Data' / 'mosaic.fits')
    parser.add_argument('--output', type=Path, default=here / 'outputs' / 'euclidean_comparison')
    parser.add_argument('--n-sources', type=int, default=1000)
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pipeline', choices=('original', 'copy', 'both'), default='both',
                        help='Which pipeline to run (default: both)')
    parser.add_argument('--placement', choices=('grid', 'random'), default='grid',
                        help='Grid cells or independent random positions (default: grid)')
    parser.add_argument('--spacing', type=float, default=50,
                        help='Grid spacing in pixels; used only with --placement grid')
    args = parser.parse_args()
    if args.n_sources < 1 or args.size < 128:
        raise ValueError('Use n_sources >= 1 and size >= 128')
    if not np.isfinite(args.spacing) or args.spacing <= 0:
        raise ValueError('spacing must be positive and finite')
    output = args.output
    if args.pipeline != 'both' and output == here / 'outputs' / 'euclidean_comparison':
        output = output.with_name(f'euclidean_comparison_{args.pipeline}')
    output.mkdir(parents=True, exist_ok=True)

    # Read calibration only; this image has no existing real sources.
    original_header = fits.getheader(args.fits)
    header = fits.Header()
    for key in ('GAIN', 'EXPTIME', 'MAGZPT', 'MAGZRR', 'SATURATE'):
        header[key] = original_header[key]
    header['BUNIT'] = 'ADU'
    gain, exposure, zp = header['GAIN'], header['EXPTIME'], header['MAGZPT']

    # One truth catalogue for both methods.
    distances, magnitudes, sigma_major, sigma_minor = simulate_euclidean_abs(
        n_sources=args.n_sources, seed=args.seed,
    )
    fluxes = exposure / gain * 10**((zp - magnitudes) / 2.5)
    rng = np.random.default_rng(args.seed + 1)

    # Grid limits close pairs; random positions allow overlap.
    margin = 40
    if args.placement == 'grid':
        grid = np.arange(margin, args.size - margin, args.spacing)
        n_cells = len(grid)**2
        if args.n_sources > n_cells:
            raise ValueError(
                f'Only {n_cells} grid positions fit; increase --size, '
                'reduce --spacing or --n-sources, or use --placement random'
            )
        # Draw cell indices without building a full meshgrid.
        cells = rng.choice(n_cells, args.n_sources, replace=False)
        positions = np.column_stack((grid[cells % len(grid)], grid[cells // len(grid)])).astype(float)
        jitter = min(5.0, 0.1 * args.spacing)
        positions += rng.uniform(-jitter, jitter, positions.shape)
    else:
        # No minimum separation between centres.
        positions = rng.uniform(margin, args.size - margin, (args.n_sources, 2))
    angles = rng.uniform(0, np.pi, args.n_sources)

    # Gaussian sky noise; source photon noise is not added in this first test.
    background_mean, background_std = 3419.0, 12.0
    sim_image = rng.normal(background_mean, background_std, (args.size, args.size))
    added_fluxes = []
    for i, (x, y) in enumerate(positions):
        radius = int(np.ceil(5 * sigma_major[i]))
        x0, x1 = int(np.floor(x)) - radius, int(np.floor(x)) + radius + 2
        y0, y1 = int(np.floor(y)) - radius, int(np.floor(y)) + radius + 2
        source = elliptical_gaussian(
            (y1 - y0, x1 - x0), (x - x0, y - y0), fluxes[i],
            sigma_major[i], sigma_minor[i], angles[i],
        )
        sim_image[y0:y1, x0:x1] += source
        added_fluxes.append(source.sum())

    truth = Table({
        'frame_id': np.ones(args.n_sources, dtype=int),
        'injected_id': np.arange(1, args.n_sources + 1),
        'x_local': positions[:, 0], 'y_local': positions[:, 1],
        'distance_pc': distances, 'm_in': magnitudes, 'flux_in': fluxes,
        'flux_added': added_fluxes, 'sigma_major': sigma_major,
        'sigma_minor': sigma_minor, 'theta': angles,
    })
    truth.write(output / 'injected.ecsv', overwrite=True)
    fits.writeto(output / 'simulated_image.fits', sim_image.astype('f4'), header, overwrite=True)

    all_results = {}
    pipeline_settings = {}
    if args.pipeline in ('original', 'both'):
        # Original detection, deblending and fixed-scale photometry.
        print('Running source_detection...', flush=True)
        with patch('matplotlib.pyplot.show'):
            labels, mean, stddev = detect_sources(sim_image, threshold=3, min_area=5)
        plt.close('all')
        if labels.max() > 0:
            labels = deblend(sim_image, labels)
        all_results['source_detection'] = fixed_aperture_photometry(
            sim_image, labels, mean, stddev, scale=3,
        )
        pipeline_settings['source_detection'] = {
            'threshold': 3, 'min_area': 5, 'photometry_scale': 3,
            'deblend': 'current source_detection.deblend wrapper',
        }

    if args.pipeline in ('copy', 'both'):
        # Copy uses its own background, masks and photometry.
        print('Running source_detection_copy...', flush=True)
        settings = Settings()
        copy_results, _, _ = run_pipeline(sim_image, output / 'copy', settings, header=header)
        all_results['source_detection_copy'] = copy_results
        pipeline_settings['source_detection_copy'] = asdict(settings)
    recoveries = {}
    for name, results in all_results.items():
        valid = ((results['flag'] == 0) & np.isfinite(results['flux'])
                 & (results['flux'] > 0))
        results['m_out'] = np.full(len(results), np.nan)
        results['m_out'][valid] = zp - 2.5 * np.log10(results['flux'][valid] * gain / exposure)
        results.write(output / f'{name}.ecsv', overwrite=True)
        recoveries[name] = compare_recovery(truth, results)
        recoveries[name].write(output / f'{name}_recovery.ecsv', overwrite=True)
        print(f'{name}: {len(results)} candidates, {valid.sum()} usable, '
              f'{recoveries[name]["usable_recovered"].sum()}/{args.n_sources} usable recoveries')

    # The same zoom and display limits for both methods.
    n_panels = len(all_results)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 6),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, (name, results) in zip(axes, all_results.items()):
        ax.imshow(sim_image, origin='lower', cmap='gray', vmin=3407, vmax=3515)
        for usable, color in ((True, 'deepskyblue'), (False, 'orange')):
            rows = results[np.isfinite(results['m_out']) == usable]
            patches = [Ellipse((row['x_center'], row['y_center']), 2 * row['a'],
                               2 * row['b'], angle=np.degrees(row['theta']))
                       for row in rows if np.isfinite(row['a'])]
            ax.add_collection(PatchCollection(patches, facecolors='none',
                                             edgecolors=color, linewidths=0.8))
        ax.set_xlim(0, min(600, args.size) - 1)
        ax.set_ylim(0, min(600, args.size) - 1)
        ax.set_title(name)
        ax.set_xlabel('X pixel')
    axes[0].set_ylabel('Y pixel')
    fig.suptitle('Same simulated image: blue = usable; orange = excluded')
    fig.tight_layout()
    fig.savefig(output / 'aperture_comparison.png', dpi=180)
    plt.close(fig)

    thresholds = np.arange(18, 25.01, 0.25)
    bins = np.arange(18, 25.01, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    input_counts = np.array([(magnitudes < m).sum() for m in thresholds])
    positive = input_counts > 0
    axes[0].plot(thresholds[positive], np.log10(input_counts[positive]),
                 'k-', label='Input catalogue')
    axes[0].plot(thresholds, np.log10(args.n_sources) + 0.6 * (thresholds - 25),
                 'k:', label='Ideal expectation: slope 0.6')
    colors = {'source_detection': 'tab:blue', 'source_detection_copy': 'tab:orange'}
    for name, results in all_results.items():
        color = colors[name]
        counts = np.array([(results['m_out'] < m).sum() for m in thresholds])
        positive = counts > 0
        axes[0].plot(thresholds[positive], np.log10(counts[positive]),
                     color=color, label=name)
        recovery = recoveries[name]
        denominator, _ = np.histogram(magnitudes, bins=bins)
        for column, style in (('position_recovered', '--'), ('usable_recovered', '-')):
            numerator, _ = np.histogram(recovery['m_in'][recovery[column]], bins=bins)
            populated = denominator > 0
            axes[1].plot(((bins[:-1] + bins[1:]) / 2)[populated],
                         numerator[populated] / denominator[populated], style,
                         color=color, label=f'{name}: {column}')
    axes[0].set(xlabel='Magnitude threshold', ylabel='log10 N(<m)', title='Measured magnitude counts')
    axes[1].set(xlabel='Input magnitude', ylabel='Recovered fraction', ylim=(-0.03, 1.03),
                title='Unique matches within 3 pixels')
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / 'counts_and_recovery.png', dpi=180)
    fig.savefig(output / 'counts_and_recovery.pdf')
    plt.close(fig)
    summary = {
        'seed': args.seed, 'placement_noise_seed': args.seed + 1,
        'n_sources': args.n_sources, 'shape': list(sim_image.shape),
        'background_mean': background_mean, 'background_std': background_std,
        'source_photon_noise': False, 'placement': args.placement,
        'spacing_pixels': args.spacing if args.placement == 'grid' else None,
        'grid_jitter_pixels': min(5.0, 0.1 * args.spacing) if args.placement == 'grid' else None,
        'edge_margin_pixels': margin,
        'positions': 'No real sources; independent random positions may overlap' if args.placement == 'random'
                     else 'No real sources; randomly chosen grid cells with jitter',
        'size_model': 'sigma_major uniform(2,5) pixels, independent of distance and magnitude',
        'pipeline': args.pipeline, 'pipeline_settings': pipeline_settings,
        'matching_radius_pixels': 3,
        'note': 'Ideal-noise test, not real-image completeness or a calibrated purity estimate.',
    }
    (output / 'simulation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Figures and catalogues: {output.resolve()}')


if __name__ == '__main__':
    main()