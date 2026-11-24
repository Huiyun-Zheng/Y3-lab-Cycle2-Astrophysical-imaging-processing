"""Inject several Gaussian sources into each cutout, then match all sources.

Includes a default smoke experiment: 5 simultaneous sources in each of 3 cutouts.
Raw mode injects Gaussian signals into real cutouts without extra source noise.
Sim mode processes supplied simulated images, including their existing noise.
Both modes share matching, quality flags and output tables.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from injection import elliptical_gaussian
from match_many_recovery import match_recovery
from run_copy_photometry import Settings, recovery_rows, run_pipeline


def make_grid(shape, spacing, margin):
    """Return cutout centres in full-image (x, y) coordinates."""

    height, width = shape
    x = np.arange(margin, width - margin, spacing)
    y = np.arange(margin, height - margin, spacing)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack((xx.ravel(), yy.ravel()))


def as_frame_rows(results, frame_id, x_min, y_min):
    """Use the existing adapter, with frame IDs for the many-source matcher."""

    rows = recovery_rows(
        results, frame_id, x_offset=x_min, y_offset=y_min,
    )
    rows.rename_column('trial_id', 'frame_id')
    return rows


def plot_flux_ratio(matches, filename=None, show=True):
    """Plot output/input flux by status; retain finite flagged measurements.

    Ratios are computed here because the stored flux_ratio column is only
    populated for matched rows. Shared detections may appear more than once.
    """

    required = ('status', 'flux_in', 'flux_out')
    missing = [name for name in required if name not in matches.colnames]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    flux_in = np.asarray(np.ma.filled(matches['flux_in'], np.nan), dtype=float)
    flux_out = np.asarray(np.ma.filled(matches['flux_out'], np.nan), dtype=float)
    valid = np.isfinite(flux_in) & (flux_in > 0) & np.isfinite(flux_out)
    ratio = np.full(len(matches), np.nan)
    np.divide(flux_out, flux_in, out=ratio, where=valid)
    valid &= np.isfinite(ratio)

    fig, ax = plt.subplots(figsize=(8, 5))
    for status, color in (
            ('matched', 'tab:green'),
            ('near_injected', 'tab:blue'),
            ('shared_detection', 'tab:red')
            ):
        selected = np.asarray(matches['status'] == status)
        plotted = selected & valid
        ax.scatter(
            flux_in[plotted], ratio[plotted],
            color=color, s=30, alpha=0.75,
            label=f'{status}: {plotted.sum()}/{selected.sum()}'
        )

    ax.axhline(1, color='black', linestyle='--', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('Input flux (ADU)')
    ax.set_ylabel('Output flux / input flux')
    ax.set_title('Flux recovery by matching status')
    ax.legend(title='Plotted / total')
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if filename is not None:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=180, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def run_experiment(
        data, header, output_dir, *, sources_per_frame=5, max_frames=3,
        spacing=200, grid_margin=200, seed=42, match_radius=3.0,
        settings=None, input_path=None, bg_type='raw'):
    """
    Return injected, detected, original_detected, matches; save all four.

    Raw mode accepts a 2D image; each cutout is detected before and after
    injection. Sim mode accepts (images, truths, results); existing results
    are ignored and each supplied image is detected by the final pipeline.
    Both modes share matching, quality columns and saved output tables.
    Raw placements keep 5*sigma_major inside the baseline valid detection
    area and separate centres by >=3*(sigma_major_i+sigma_major_j).
    Existing real-source centres are NOT removed from the sampled area;
    the matcher records near_original cases separately.

    Arguments:
    data: np.ndarray or (images, truths, results)
    header: fits.Header
    output_dir: Path
    bg_type: str ('raw' for original crop and 'sim' for simulated background)
    """

    if bg_type == 'raw' and np.ndim(data) != 2:
        raise ValueError('data must be a two-dimensional image')
    for name, value in [('sources_per_frame', sources_per_frame),
                        ('max_frames', max_frames), ('spacing', spacing),
                        ('grid_margin', grid_margin)]:
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if grid_margin < spacing / 2:
        raise ValueError('grid_margin must be at least half the cutout spacing')
    if not np.isfinite(match_radius) or match_radius <= 0:
        raise ValueError('match_radius must be positive and finite')

    if bg_type not in ('raw', 'sim'):
        raise ValueError("bg_type must be 'raw' or 'sim'")
    settings = Settings() if settings is None else settings
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if bg_type == 'raw':
        positions = make_grid(data.shape, spacing, grid_margin)[:max_frames]
        if len(positions) == 0:
            raise ValueError('No cutout centres fit; reduce grid_margin or use a larger image')

        injected_rows = []
        detected_frames = []
        original_frames = []
        half = spacing // 2

        # create frame crop from the image
        for frame_id, (grid_x, grid_y) in enumerate(positions, start=1):
            x_min, y_min = int(grid_x - half), int(grid_y - half)
            background = np.asarray(
                data[y_min:y_min + spacing, x_min:x_min + spacing], dtype=float,
            )
            frame_dir = output_dir / f'frame_{frame_id:04d}'

            # Baseline must use this same cutout and settings, with its header.
            original, _, _ = run_pipeline(background, frame_dir / 'before', settings, header=header)
            original_frames.append(as_frame_rows(original, frame_id, x_min, y_min))
            with fits.open(frame_dir / 'before' / 'detection_products.fits') as hdul:
                detmask = hdul['DETMASK'].data.astype(bool)
            clearance = ndi.distance_transform_edt(~detmask)

            # This remains ONE two-dimensional image, with all sources added.
            sim_image = background.copy()
            placed = []
            height, width = background.shape
            for source_number in range(1, sources_per_frame + 1):
                flux = float(rng.uniform(1e3, 2e4))
                sigma_major = float(rng.uniform(2, 5))
                sigma_minor = sigma_major * float(rng.uniform(0.5, 1))
                theta = float(rng.uniform(0, np.pi))
                source_margin = 5 * sigma_major
                if 2 * source_margin >= min(height, width):
                    raise ValueError('The cutout is too small for the sampled source')

                for _ in range(2000):
                    x = float(rng.uniform(source_margin, width - source_margin))
                    y = float(rng.uniform(source_margin, height - source_margin))
                    # The extra pixel covers rounding from a subpixel centre.
                    if clearance[int(round(y)), int(round(x))] < source_margin + 1:
                        continue
                    if any(np.hypot(x - px, y - py) < 3 * (sigma_major + psigma)
                        for px, py, psigma in placed):
                        continue
                    break
                else:
                    raise RuntimeError(
                        f'Frame {frame_id}: could not place source {source_number} '
                        f'of {sources_per_frame}. Reduce the number or inspect DETMASK.'
                    )

                source = elliptical_gaussian(
                    background.shape, (x, y), flux, sigma_major, sigma_minor, theta,
                )
                sim_image += source
                placed.append((x, y, sigma_major))
                injected_rows.append((
                    frame_id, source_number, x, y, x_min + x, y_min + y,
                    flux, float(source.sum()), sigma_major, sigma_minor, theta,
                ))

            # Detect the combined image once, after every injection is added.
            after, _, _ = run_pipeline(
                sim_image, frame_dir / 'after', settings, header=header,
            )
            detected_frames.append(as_frame_rows(after, frame_id, x_min, y_min))

        injected = Table(
            rows=injected_rows,
            names=('frame_id', 'injected_id', 'x_local', 'y_local', 'x_in', 'y_in',
                'flux_in', 'flux_added', 'sigma_major', 'sigma_minor', 'theta'),
            dtype=('i8', 'i8') + ('f8',) * 9,
        )
        injected['theta'].unit = 'rad'
        injected.meta['model'] = 'Observed Gaussian sampled at pixel centres'
        injected.meta['source_noise'] = 'Deterministic signal; no added source Poisson noise'
        injected.meta['placement'] = (
            '5*sigma_major within baseline DETMASK clearance; '
            'injected pairs separated by >=3*(sigma_major_i+sigma_major_j)'
        )
        detected = vstack(detected_frames)
        original_detected = vstack(original_frames)

    else:
        images, truths, _ = data

        if len(images) == 0 or len(images) != len(truths):
            raise ValueError("images and truths must have the same nonzero length")

        injected_rows = []
        detected_frames = []

        # Process each image
        for frame_id, (image, truth) in enumerate(
                zip(images, truths), start=1
                ):
            if np.ndim(image) != 2 or len(truth) == 0:
                raise ValueError('Each simulated image must be 2D with a nonempty truth table')
            frame_dir = output_dir / f'frame_{frame_id:04d}'

            # Detect once per image
            after, _, _ = run_pipeline(
                image, frame_dir / 'after', settings, header=header
            )
            detected_frames.append(
                as_frame_rows(after, frame_id, 0, 0)
            )

            # Record each injected source
            for row in truth:
                x = float(row['x_center'])
                y = float(row['y_center'])

                injected_rows.append((
                    frame_id, int(row['id']),
                    x, y, x, y,
                    float(row['flux_adu']), np.nan,
                    float(row['sigma_major']),
                    float(row['sigma_minor']),
                    float(row['theta'])
                ))

        # Build tables after all images are processed
        injected = Table(
            rows=injected_rows,
            names=(
                'frame_id', 'injected_id',
                'x_local', 'y_local', 'x_in', 'y_in',
                'flux_in', 'flux_added',
                'sigma_major', 'sigma_minor', 'theta'
            ),
            dtype=('i8', 'i8') + ('f8',) * 9
        )

        injected['theta'].unit = 'rad'
        injected.meta['model'] = 'Observed Gaussian sampled at pixel centres'
        injected.meta['placement'] = (
            'Random centres inside a shared edge margin; '
            'injected sources may overlap'
        )
        injected.meta['source_noise'] = 'Poisson source noise added in electrons'

        detected = vstack(detected_frames)

        # No original astronomical sources in this simulated background
        original_detected = Table(
            names=('frame_id', 'x_local', 'y_local'),
            dtype=('i8', 'f8', 'f8')
        )

    # Match after both tables are ready
    matches = match_recovery(
        injected, detected, original_detected,
        match_radius=match_radius
    )

    # Keep detector quality alongside matching; do not drop flagged rows.
    by_id = {(int(row['frame_id']), int(row['detection_id'])): row
            for row in detected}
    matches['usable_out'] = np.zeros(len(matches), dtype=bool)
    matches['detection_flags'] = np.full(len(matches), '', dtype='U180')
    for i, row in enumerate(matches):
        candidate = by_id.get((int(row['frame_id']), int(row['detection_id'])))
        if candidate is not None:
            matches['usable_out'][i] = bool(candidate['usable'])
            matches['detection_flags'][i] = candidate['flags']
    matches['unique_position_match'] = (
        (matches['n_candidates'] == 1) & (matches['n_claimants'] == 1)
    )
    matches['clean_usable_recovery'] = (
        (matches['status'] == 'matched') & matches['usable_out']
    )
    matches.meta['position_note'] = (
        'unique_position_match is positional only; it can coincide with an '
        'original source. status includes original proximity and flux validity.'
    )

    for name, table in [('injected', injected), ('detected', detected),
                        ('original_detected', original_detected), ('matches', matches)]:
        table.write(output_dir / f'{name}.ecsv', overwrite=True)
    states, counts = np.unique(matches['status'], return_counts=True)
    n_clean = int(np.count_nonzero(matches['status'] == 'matched'))
    summary = {
        'input': None if input_path is None else str(Path(input_path).resolve()),
        'seed': seed, 'settings': asdict(settings), 'bg_type': bg_type,
        'spacing': spacing if bg_type == 'raw' else None,
        'grid_margin': grid_margin if bg_type == 'raw' else None,
        'frames': len(detected_frames),
        'sources_per_frame': sources_per_frame if bg_type == 'raw' else None,
        'injected': len(injected),
        'match_radius': match_radius,
        'status_counts': {str(s): int(n) for s, n in zip(states, counts)},
        'conservative_flux_match_fraction': n_clean / len(injected) if len(injected) else None,
        'clean_usable_recoveries': int(matches['clean_usable_recovery'].sum()),
        'scope': ('Cutout pipeline with sparse, mask-avoiding Gaussian injections'
                  if bg_type == 'raw' else 'Final pipeline on supplied simulated images; overlap allowed'),
        'note': 'matched includes proximity and positive finite flux requirements; '
                'this fraction is not pure detection completeness.',
    }
    (output_dir / 'experiment_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{len(injected)} injected in {len(detected_frames)} frames; '
        f'{n_clean} conservative flux matches.')
    print(summary['status_counts'])


    # Save the diagnostic for both background modes.
    fig, _ = plot_flux_ratio(matches, output_dir / 'flux_ratio.png', show=False)
    import matplotlib.pyplot as plt
    plt.close(fig)

    return injected, detected, original_detected, matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources-per-frame', type=int, default=5)
    parser.add_argument('--max-frames', type=int, default=3)
    parser.add_argument('--spacing', type=int, default=200)
    parser.add_argument('--grid-margin', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--match-radius', type=float, default=3.0)
    parser.add_argument('--background', '--backrgound', dest='background',
                        choices=('raw', 'sim'), default='sim')
    parser.add_argument('--n-sources', type=int, default=100,
                        help='Number of simulated sources')
    parser.add_argument('--size', type=int, default=200,
                        help='Side length of simulated images in pixels')
    args = parser.parse_args()
    with fits.open(args.fits, memmap=False) as hdul:
        header = hdul[0].header.copy()
        image = hdul[0].data.astype(np.float32)

    if args.background == 'sim':
        from unittest.mock import patch
        import matplotlib.pyplot as plt
        from background import fit_gaussian
        from euclidean_mc import simulate_euclidean
        from euclidean_image_test import make_mc_source_and_image

        if args.n_sources < 1 or args.size < 64:
            parser.error('--n-sources must be positive and --size must be at least 64')
        reference = image[:4550, 70:2450]
        # The histogram fit should not open a window during this batch run.
        with patch('matplotlib.pyplot.show'):
            bg_mean, bg_std = fit_gaussian(
                reference[np.isfinite(reference)], bins=300, range=(3300, 3600)
            )
        plt.close('all')
        distances, magnitudes = simulate_euclidean(n_sources=args.n_sources, seed=args.seed)
        images, truths = make_mc_source_and_image(
            header, bg_mean, bg_std, shape=(args.size, args.size),
            seed=args.seed + 1, magnitudes=magnitudes,
        )
        truths[0]['distance_pc'] = distances
        truths[0].meta['catalogue_seed'] = args.seed
        # Sim mode runs detection itself, so no previous results are needed.
        data = (images, truths, [])
    else:
        data = image

    run_experiment(
        data, header, args.output, sources_per_frame=args.sources_per_frame,
        max_frames=args.max_frames, spacing=args.spacing, grid_margin=args.grid_margin,
        seed=args.seed, match_radius=args.match_radius, input_path=args.fits,
        bg_type=args.background,
    )


if __name__ == '__main__':
    main()