"""Inject several Gaussian sources into each cutout, then match all sources.

Default smoke experiment: 5 simultaneous sources in each of 3 cutouts.
This measures the cutout pipeline under the recorded placement rules.
The model is an observed Gaussian with no additional source Poisson noise.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from scipy import ndimage as ndi

from match_many_recovery import match_many_recovery
from run_copy_photometry import recovery_rows, run_pipeline
from source_detection_copy import Settings


def elliptical_gaussian(shape, center, flux, sigma_major, sigma_minor, theta):
    """One observed Gaussian, sampled at pixel centres; theta is radians.

    Parameters describe one source and must be scalars. The generator is
    kept here so importing injection.py cannot execute its unfinished demo.
    """
    y, x = np.ogrid[:shape[0], :shape[1]]
    dx, dy = x - center[0], y - center[1]
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    amplitude = flux / (2 * np.pi * sigma_major * sigma_minor)
    return amplitude * np.exp(-0.5 * ((u / sigma_major)**2 + (v / sigma_minor)**2))


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


def run_experiment(
        data, header, output_dir, *, sources_per_frame=5, max_frames=3,
        spacing=200, grid_margin=200, seed=42, match_radius=3.0,
        settings=None, input_path=None):
    """Return injected, detected, original_detected, matches; save all four.

    A frame is one cutout containing several sources. Each source gets one
    truth row. Each frame is detected once before and once after injection.
    Placements keep 5*sigma_major inside the baseline valid detection area
    and separate injected centres by >=3*(sigma_major_i+sigma_major_j).
    Existing real-source centres are NOT removed from the sampled area;
    the matcher records near_original cases separately.
    """
    if np.ndim(data) != 2:
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

    settings = Settings() if settings is None else settings
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = make_grid(data.shape, spacing, grid_margin)[:max_frames]
    if len(positions) == 0:
        raise ValueError('No cutout centres fit; reduce grid_margin or use a larger image')

    injected_rows = []
    detected_frames = []
    original_frames = []
    half = spacing // 2

    for frame_id, (grid_x, grid_y) in enumerate(positions, start=1):
        x_min, y_min = int(grid_x - half), int(grid_y - half)
        background = np.asarray(
            data[y_min:y_min + spacing, x_min:x_min + spacing], dtype=float,
        )
        frame_dir = output_dir / f'frame_{frame_id:04d}'

        # Baseline must use this same cutout and settings, with its header.
        original, _, _ = run_pipeline(
            background, frame_dir / 'before', settings, header=header,
        )
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
    matches = match_many_recovery(
        injected, detected, original_detected, match_radius=match_radius,
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
        'seed': seed, 'settings': asdict(settings), 'spacing': spacing,
        'grid_margin': grid_margin, 'frames': len(positions),
        'sources_per_frame': sources_per_frame, 'injected': len(injected),
        'match_radius': match_radius,
        'status_counts': {str(s): int(n) for s, n in zip(states, counts)},
        'conservative_flux_match_fraction': n_clean / len(injected),
        'clean_usable_recoveries': int(matches['clean_usable_recovery'].sum()),
        'scope': 'Cutout pipeline with sparse, mask-avoiding Gaussian injections',
        'note': 'matched includes proximity and positive finite flux requirements; '
                'this fraction is not pure detection completeness.',
    }
    (output_dir / 'experiment_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{len(injected)} injected in {len(positions)} frames; '
          f'{n_clean} conservative flux matches.')
    print(summary['status_counts'])
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
    args = parser.parse_args()
    with fits.open(args.fits) as hdul:
        header = hdul[0].header.copy()
        image = hdul[0].data.astype(np.float32)
    run_experiment(
        image, header, args.output, sources_per_frame=args.sources_per_frame,
        max_frames=args.max_frames, spacing=args.spacing, grid_margin=args.grid_margin,
        seed=args.seed, match_radius=args.match_radius, input_path=args.fits,
    )


if __name__ == '__main__':
    main()
