"""Inject multiple Sersic + Moffat sources per frame and match recoveries."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaincinv
from astropy.modeling.models import Sersic2D
from astropy.table import Table, vstack
from astropy.convolution import convolve_fft

from get import get_fits_data
from background import fit_gaussian
from run_copy_photometry import run_pipeline, plot_aperture_catalogue
from injection_recovery_many import as_frame_rows
from match_many_recovery import match_recovery
from photometry import AB_calibration

project_dir = Path(__file__).resolve().parent.parent.parent


def galaxy(
        amplitude, r_eff, shape, center, ellipticity, theta,
        image_shape, moffat_alpha, moffat_beta, half_size=50
        ):
    """
    Create simulated galaxy profile by convolving an intrinsic Sersic profile with the Moffat psf.
    
    Args:
    ampltidue: float, intensity at effective radius
    r_eff: float, size of galaxy
    shape: str, choose from "gaussian", "exponential", "deVau"
    center: 2d tuple
    ellipticity: float, 1- b/a, in range [0, 1]
    theta: float, rotation angle
    image_shape: 2d tuple, same as backrgound shape

    Returns:
    observed_source: observed galaxy profile
    flux_in: input flux (for magnitude calculation)
    """

    if shape == 'gaussian':
        n = 0.5
    elif shape == 'exponential':
        n = 1
    elif shape == 'deVau':
        n = 4
    else:
        raise ValueError(
            "shape must be 'gaussian', 'exponential', or 'deVau'"
        )

    if r_eff <= 0 or not 0 <= ellipticity < 1:
        raise ValueError("Require r_eff > 0 and 0 <= ellipticity < 1")

    if moffat_alpha <= 0 or moffat_beta <= 1:
        raise ValueError("Require moffat_alpha > 0 and moffat_beta > 1")

    if not isinstance(half_size, (int, np.integer)) or half_size < 0:
        raise ValueError("half_size must be a nonnegative integer")

    height, width = image_shape
    x_center, y_center = center

    # Intrinsic galaxy model
    intrinsic_model = Sersic2D(
        amplitude=amplitude,
        r_eff=r_eff,
        n=n,
        x_0=x_center,
        y_0=y_center,
        ellip=ellipticity,
        theta=theta
    )

    # Include light that can scatter into the output image
    yy_image, xx_image = np.mgrid[
        -half_size:height + half_size,
        -half_size:width + half_size
    ]
    intrinsic_source = intrinsic_model(xx_image, yy_image)

    # PSF coordinates
    yy, xx = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
    r_squared = xx**2 + yy**2
    psf = (
        (moffat_beta - 1) / (np.pi * moffat_alpha**2)
        * (1 + r_squared / moffat_alpha**2)**(-moffat_beta)
    )

    # Convolve
    observed_source = convolve_fft(
        intrinsic_source,
        psf,
        normalize_kernel=True,
        boundary="fill",
        fill_value=0
    )

    # Analytical total flux
    b_n = gammaincinv(2 * n, 0.5)
    axis_ratio = 1 - ellipticity

    flux_in = (
        2 * np.pi * axis_ratio
        * amplitude * r_eff**2
        * n * np.exp(b_n)
        * gamma(2 * n) / b_n**(2 * n)
    )

    # Return the requested image region
    return observed_source[
        half_size:half_size + height,
        half_size:half_size + width
    ], float(flux_in)


def sample_centres(rng, n_sources, image_shape, edge_margin, min_separation):
    """Sample random positions; overlap is allowed when min_separation=0."""

    height, width = image_shape

    if not (0 <= edge_margin < (min(height, width) - 1) / 2):
        raise ValueError("edge_margin leaves no room for source centres")
    
    positions = []
    for _ in range(max(1000, 1000 * n_sources)):
        x = rng.uniform(edge_margin, width - 1 - edge_margin)
        y = rng.uniform(edge_margin, height - 1 - edge_margin)
        if positions:
            previous = np.asarray(positions)
            if np.any(np.hypot(previous[:, 0] - x, previous[:, 1] - y)
                      < min_separation):
                continue
        positions.append((x, y))
        if len(positions) == n_sources:
            return np.asarray(positions)
    raise ValueError("Cannot place all sources; reduce min_separation or n_sources")


def source_radius(source, center, fraction=0.9):
    """Circular radius containing a fraction of the rendered positive signal."""

    yy, xx = np.indices(source.shape)
    radius = np.hypot(xx - center[0], yy - center[1]).ravel()
    signal = np.maximum(source, 0).ravel()
    order = np.argsort(radius)
    cumulative = np.cumsum(signal[order])
    index = np.searchsorted(cumulative, fraction * cumulative[-1])
    return max(float(radius[order[index]]), 0.5)



def plot_recovery_ratios(matches, filename, show=False):
    """Plot finite measurements, including crowded and flagged cases."""

    flux_in = np.asarray(matches['flux_in'], dtype=float)
    flux_out = np.asarray(np.ma.filled(matches['flux_out'], np.nan), dtype=float)
    finite = np.isfinite(flux_in) & (flux_in > 0) & np.isfinite(flux_out)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'matched': 'tab:green', 'near_injected': 'tab:blue',
              'shared_detection': 'tab:red', 'near_original': 'tab:orange',
              'ambiguous': 'tab:purple', 'invalid_flux': 'tab:gray'}
    for status, color in colors.items():
        selected = np.asarray(matches['status'] == status)
        usable = selected & finite
        if np.any(selected):
            ax.scatter(flux_in[usable], flux_out[usable] / flux_in[usable],
                       color=color, s=30, alpha=0.7,
                       label=f'{status}: {usable.sum()}/{selected.sum()}')
    positive = flux_in[np.isfinite(flux_in) & (flux_in > 0)]
    if len(positive):
        ax.set_xlim(float(positive.min()) * 0.8, float(positive.max()) * 1.2)
    else:
        ax.set_xlim(1, 10)
    ax.set_xscale('log')
    ax.axhline(1, color='black', linestyle='--', linewidth=1)
    if not np.any(finite):
        ax.text(0.5, 0.5, 'No finite output fluxes to plot',
                ha='center', transform=ax.transAxes)
    ax.set_xlabel('Input flux (ADU)')
    ax.set_ylabel('Measured output flux / input flux')
    ax.set_title('Flux recovery by matching status')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title='Plotted / total')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=180, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def summarize_groups(matches):
    """Count positional recoveries separately from usable photometry."""

    rows = []
    for group in np.unique(matches['group']):
        selected = matches[matches['group'] == group]
        n_injected = len(selected)
        n_recovered = int(selected['detection_recovered'].sum())
        n_usable = int(selected['photometry_recovered'].sum())
        n_clean = int(selected['clean_usable_recovery'].sum())
        rows.append((group, n_injected, n_recovered, n_usable, n_clean,
                     n_recovered / n_injected, n_usable / n_injected,
                     n_clean / n_injected))
    summary = Table(
        rows=rows,
        names=('group', 'n_injected', 'n_recovered', 'n_usable', 'n_clean',
               'detection_fraction', 'usable_fraction', 'clean_fraction'),
        dtype=('U10', 'i8', 'i8', 'i8', 'i8', 'f8', 'f8', 'f8')
    )
    summary.meta['recovered'] = (
        'Mutually unique positional counterpart, with no nearby baseline detection; '
        'near_injected is included, shared and ambiguous detections are excluded.'
    )
    summary.meta['usable'] = 'Recovered and detector quality passed, with positive finite flux'
    summary.meta['clean'] = 'Usable and status=matched; excludes nearby injected sources'
    summary.meta['note'] = 'No input/output flux-agreement cut or chance-match correction'
    return summary


def run_experiment(
        fits_path, psf_result_path, output_dir, *,
        n_sources=5, n_frames=3, amplitude=200, r_eff=5,
        profile='exponential', ellipticity=0.3, theta=np.pi / 6,
        amplitude_range=None, r_eff_range=None, mag_range=None,
        random_angles=False, background_type='sim', seed=42, size=200,
        x_range=(677, 940), y_range=(4000, 4156),
        edge_margin=25, min_separation=0, half_size=50,
        match_radius=3, source_poisson=False, settings=None, show=False,
        n_bright=None, n_faint=None,
        bright_mag_range=None, faint_mag_range=None,
        bright_r_eff_range=None, faint_r_eff_range=None
        ):
    """Inject one population or simultaneous bright/faint groups in each frame.

    Set both group counts and the magnitude range for each nonempty group.
    Their sum replaces n_sources; group sizes default to r_eff/r_eff_range.
    Return injected, detected, original_detected, matches; save all four.

    Args:
    x_range, y_range: for cropping the image for real backrgound
    amplitude_range, r_eff_range: for random number generator, if None, using aplitude and r_eff
    """

    two_groups = n_bright is not None or n_faint is not None
    if two_groups:
        for name, value in [('n_bright', n_bright), ('n_faint', n_faint)]:
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer; set both counts")
        n_sources = n_bright + n_faint
        for name, count, limits in [('bright_mag_range', n_bright, bright_mag_range),
                                    ('faint_mag_range', n_faint, faint_mag_range)]:
            if count > 0 and limits is None:
                raise ValueError(f"Set {name} for this group")
        if mag_range is not None or amplitude_range is not None:
            raise ValueError("Use group magnitude ranges instead of mag_range/amplitude_range")
    elif any(value is not None for value in (
            bright_mag_range, faint_mag_range, bright_r_eff_range, faint_r_eff_range)):
        raise ValueError("Set n_bright and n_faint to use group parameters")
    for name, value in [('n_sources', n_sources), ('n_frames', n_frames)]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("amplitude must be positive and finite")
    if not np.isfinite(r_eff) or r_eff <= 0:
        raise ValueError("r_eff must be positive and finite")
    if not np.isfinite(ellipticity) or not 0 <= ellipticity < 1:
        raise ValueError("ellipticity must be in [0, 1)")
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    if not np.isfinite(match_radius) or match_radius <= 0:
        raise ValueError("match_radius must be positive and finite")
    if not isinstance(size, (int, np.integer)) or size < 32:
        raise ValueError("size must be an integer >= 32")
    if background_type not in ('sim', 'image'):
        raise ValueError("background_type must be 'sim' or 'image'")
    if not np.isfinite(edge_margin) or edge_margin < 0:
        raise ValueError("edge_margin must be nonnegative and finite")
    if not np.isfinite(min_separation) or min_separation < 0:
        raise ValueError("min_separation must be nonnegative and finite")
    if amplitude_range is not None and mag_range is not None:
        raise ValueError("Choose amplitude_range or mag_range, not both")
    for name, limits in [('amplitude_range', amplitude_range),
                         ('r_eff_range', r_eff_range), ('mag_range', mag_range),
                         ('bright_mag_range', bright_mag_range),
                         ('faint_mag_range', faint_mag_range),
                         ('bright_r_eff_range', bright_r_eff_range),
                         ('faint_r_eff_range', faint_r_eff_range)]:
        if limits is not None:
            if len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[0] > limits[1]:
                raise ValueError(f"{name} requires finite lower <= upper")
            if 'mag_range' not in name and limits[0] <= 0:
                raise ValueError(f"{name} must be positive")
    if profile not in ('gaussian', 'exponential', 'deVau'):
        raise ValueError("Unknown Sersic profile")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    background_rng = np.random.default_rng(seed + 1)
    poisson_rng = np.random.default_rng(seed + 2)
    header, data = get_fits_data(file_path=Path(fits_path))
    psf_results = Table.read(psf_result_path)
    alpha = float(psf_results['alpha'][0])
    beta = float(psf_results['beta'][0])
    n = {'gaussian': 0.5, 'exponential': 1., 'deVau': 4.}[profile]
    b_n = gammaincinv(2 * n, 0.5)

    # Reuse the same untouched real cutout for independent injections
    baseline = None
    if background_type == 'sim':
        mean, stddev = fit_gaussian(
            data, show=show,
            filename=output_dir / 'background_histogram.png'
        )
        image_shape = (size, size)
        x_offset = y_offset = 0
    else:
        x_min, x_max = x_range
        y_min, y_max = y_range
        if not (0 <= x_min < x_max <= data.shape[1]
                and 0 <= y_min < y_max <= data.shape[0]):
            raise ValueError("The background crop must lie inside the FITS image")
        real_background = data[y_min:y_max, x_min:x_max].astype(float).copy()
        image_shape = real_background.shape
        x_offset, y_offset = x_min, y_min

    # Validate placement before running any detection
    if edge_margin >= (min(image_shape) - 1) / 2:
        raise ValueError("edge_margin leaves no room for source centres")
    if background_type == 'image':
        baseline, _, _ = run_pipeline(
            real_background, output_dir / 'before', settings, header=header
        )

    injected_rows = []
    detected_frames = []
    original_frames = []

    for frame_id in range(1, n_frames + 1):
        frame_dir = output_dir / f'frame_{frame_id:04d}'
        frame_dir.mkdir(parents=True, exist_ok=True)
        if background_type == 'sim':
            background = background_rng.normal(mean, stddev, size=image_shape)
        else:
            background = real_background.copy()
            original_frames.append(as_frame_rows(
                baseline, frame_id, x_offset, y_offset
            ))

        # Place all sources before adding any source noise
        positions = sample_centres(
            rng, n_sources, image_shape, edge_margin, min_separation
        )
        amplitudes = (np.full(n_sources, amplitude) if amplitude_range is None
                      else rng.uniform(*amplitude_range, n_sources))
        sizes = (np.full(n_sources, r_eff) if r_eff_range is None
                 else rng.uniform(*r_eff_range, n_sources))
        angles = (rng.uniform(0, np.pi, n_sources) if random_angles
                  else np.full(n_sources, theta))
        input_mags = (None if mag_range is None
                      else rng.uniform(*mag_range, n_sources))
        groups = np.full(n_sources, 'single', dtype='U10')
        if two_groups:
            # Both groups share one image and one detection run
            groups[:n_bright] = 'bright'
            groups[n_bright:] = 'faint'
            input_mags = np.empty(n_sources)
            for group, limits, size_limits in [
                    ('bright', bright_mag_range, bright_r_eff_range),
                    ('faint', faint_mag_range, faint_r_eff_range)]:
                selected = groups == group
                count = int(selected.sum())
                if count:
                    input_mags[selected] = rng.uniform(*limits, count)
                    if size_limits is not None:
                        sizes[selected] = rng.uniform(*size_limits, count)
        sim_image = background.copy()
        injected_signal = np.zeros(image_shape, dtype=float)

        for source_id, ((x, y), source_amplitude, source_size, angle) in enumerate(
                zip(positions, amplitudes, sizes, angles), start=1):
            # Set total brightness independently of galaxy size when requested
            if input_mags is not None:
                target_flux = (10**(0.4 * (header['MAGZPT'] - input_mags[source_id - 1]))
                               * header['EXPTIME'] / header['GAIN'])
                unit_flux = (2 * np.pi * (1 - ellipticity) * source_size**2
                             * n * np.exp(b_n) * gamma(2 * n) / b_n**(2 * n))
                source_amplitude = target_flux / unit_flux
            source, input_flux = galaxy(
                source_amplitude, source_size, profile, (x, y), ellipticity,
                angle, image_shape, alpha, beta, half_size
            )
            flux_on_image = float(source.sum())
            added_source = source
            if source_poisson:
                added_source = poisson_rng.poisson(
                    np.maximum(source, 0) * header['GAIN']
                ) / header['GAIN']
            sim_image += added_source
            injected_signal += added_source
            input_mag = header['MAGZPT'] - 2.5 * np.log10(
                input_flux * header['GAIN'] / header['EXPTIME']
            )
            injected_rows.append({
                'frame_id': frame_id, 'injected_id': source_id,
                'group': str(groups[source_id - 1]),
                'x_local': x, 'y_local': y,
                'x_in': x + x_offset, 'y_in': y + y_offset,
                'profile': profile, 'sersic_n': n,
                'amplitude': float(source_amplitude), 'r_eff': float(source_size),
                'ellipticity': ellipticity, 'theta': float(angle),
                'flux_in': input_flux, 'flux_on_image': flux_on_image,
                'flux_added': float(added_source.sum()),
                'image_model_ratio': flux_on_image / input_flux,
                'input_magnitude': float(input_mag),
                'crowding_radius': source_radius(source, (x, y))
            })

        # Detect once after all sources have been injected
        results, _, _ = run_pipeline(
            sim_image, frame_dir / 'after', settings, header=header
        )
        valid = ((results['flag'] == 0) & np.isfinite(results['flux'])
                 & (results['flux'] > 0))
        results['magnitude'] = np.full(len(results), np.nan)
        if np.any(valid):
            mags, _ = AB_calibration(results, header, show=False)
            results['magnitude'][valid] = np.asarray(mags, dtype=float)
        results.write(frame_dir / 'photometry.ecsv', overwrite=True)
        frame_rows = as_frame_rows(results, frame_id, x_offset, y_offset)
        frame_rows['magnitude'] = results['magnitude']
        detected_frames.append(frame_rows)
        fig, ax = plot_aperture_catalogue(
            sim_image, results,
            filename=frame_dir / 'apertures.png', show=False
        )
        for group, color in [('single', 'lime'), ('bright', 'magenta'), ('faint', 'lime')]:
            selected = groups == group
            if np.any(selected):
                ax.scatter(positions[selected, 0], positions[selected, 1], marker='+',
                           color=color, s=50, label=f'Injected {group}: {selected.sum()}')
        ax.legend()
        fig.savefig(frame_dir / 'apertures.png', dpi=180, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        np.save(frame_dir / 'background.npy', background)
        np.save(frame_dir / 'injected_signal.npy', injected_signal)
        np.save(frame_dir / 'sim_image.npy', sim_image)
        print(f'Frame {frame_id}/{n_frames}: {n_sources} injected, '
              f'{len(results)} detections')

    injected = Table(rows=injected_rows)
    injected['theta'].unit = 'rad'
    injected.meta['model'] = 'Intrinsic Sersic convolved with normalized Moffat PSF'
    injected.meta['seed'] = int(seed)
    injected.meta['background_seed'] = int(seed + 1)
    injected.meta['source_noise_seed'] = int(seed + 2)
    injected.meta['source_poisson'] = source_poisson
    group_settings = ({
        'bright': {'n_per_frame': int(n_bright), 'mag_range': bright_mag_range,
                   'r_eff_range': bright_r_eff_range},
        'faint': {'n_per_frame': int(n_faint), 'mag_range': faint_mag_range,
                  'r_eff_range': faint_r_eff_range}
    } if two_groups else {'single': {'n_per_frame': int(n_sources), 'mag_range': mag_range}})
    injected.meta['groups'] = group_settings
    injected.meta['group_size_default'] = {'r_eff': r_eff, 'r_eff_range': r_eff_range}
    injected.meta['psf_alpha'] = alpha
    injected.meta['psf_beta'] = beta
    injected.meta['placement'] = 'Uniform random centres; min_separation is optional'
    injected.meta['sampling_note'] = (
        'Pixel-centre sampling retained; especially n=4 requires sampling checks. '
        'flux_in is analytical total; flux_on_image is the rendered finite-frame sum.'
    )
    injected['crowding_radius'].description = (
        'Circular radius containing 90% of rendered positive source flux within the frame; '
        'a crowding heuristic, not a Gaussian sigma or a deblending test'
    )
    detected = vstack(detected_frames)
    if original_frames:
        original_detected = vstack(original_frames)
    else:
        original_detected = Table(
            names=('frame_id', 'x_local', 'y_local'),
            dtype=('i8', 'f8', 'f8')
        )

    matches = match_recovery(
        injected, detected, original_detected, match_radius=match_radius
    )
    by_id = {(int(row['frame_id']), int(row['detection_id'])): row
             for row in detected}
    matches['usable_out'] = np.zeros(len(matches), dtype=bool)
    matches['detection_flags'] = np.full(len(matches), '', dtype='U180')
    matches['output_magnitude'] = np.full(len(matches), np.nan)
    for i, row in enumerate(matches):
        candidate = by_id.get((int(row['frame_id']), int(row['detection_id'])))
        if candidate is not None:
            matches['usable_out'][i] = bool(candidate['usable'])
            matches['detection_flags'][i] = str(candidate['flags'])
            matches['output_magnitude'][i] = candidate['magnitude']
    for name in ('group', 'profile', 'sersic_n', 'amplitude', 'r_eff', 'ellipticity',
                 'theta', 'input_magnitude', 'image_model_ratio', 'crowding_radius'):
        matches[name] = injected[name].copy()
    matches['unique_position_match'] = (
        (matches['n_candidates'] == 1) & (matches['n_claimants'] == 1)
    )
    matches['clean_usable_recovery'] = (
        (matches['status'] == 'matched') & matches['usable_out']
    )
    matches['detection_recovered'] = (
        matches['unique_position_match'] & (matches['n_original_near'] == 0)
    )
    matches['photometry_recovered'] = (
        matches['detection_recovered'] & matches['usable_out']
        & np.isfinite(matches['flux_out']) & (matches['flux_out'] > 0)
    )
    matches.meta['selection_note'] = (
        'detection_recovered includes near_injected but excludes nearby baseline sources, '
        'shared and ambiguous candidates. photometry_recovered also requires usable '
        'positive finite flux. clean_usable_recovery further excludes nearby injections.'
    )
    group_summary = summarize_groups(matches)
    group_summary.write(output_dir / 'group_summary.ecsv', overwrite=True)

    for name, table in [('injected', injected), ('detected', detected),
                        ('original_detected', original_detected), ('matches', matches)]:
        table.write(output_dir / f'{name}.ecsv', overwrite=True)
    states, counts = np.unique(matches['status'], return_counts=True)
    summary = {
        'frames': n_frames, 'sources_per_frame': n_sources,
        'injected': len(injected), 'seed': int(seed),
        'groups': group_settings,
        'group_size_default': {'r_eff': r_eff, 'r_eff_range': r_eff_range},
        'group_summary': [
            {name: (str(row[name]) if name == 'group'
                    else int(row[name]) if name.startswith('n_') else float(row[name]))
             for name in group_summary.colnames} for row in group_summary
        ],
        'background_type': background_type, 'profile': profile,
        'min_separation_pixels': min_separation,
        'edge_margin_pixels': edge_margin,
        'status_counts': {str(s): int(c) for s, c in zip(states, counts)},
        'unique_position_matches': int(matches['unique_position_match'].sum()),
        'detection_recoveries': int(matches['detection_recovered'].sum()),
        'usable_photometry_recoveries': int(matches['photometry_recovered'].sum()),
        'clean_usable_recoveries': int(matches['clean_usable_recovery'].sum()),
        'sampling_check_outside_2_percent': int(np.count_nonzero(
            np.abs(np.asarray(injected['image_model_ratio']) - 1) > 0.02)),
        'note': 'Centre matching with conservative crowding flags. '
                'This summary is not a calibrated completeness correction.',
        'real_background_note': 'The same untouched real cutout is reused for '
                                'each independent frame; its baseline is detected once.'
    }
    (output_dir / 'experiment_summary.json').write_text(json.dumps(summary, indent=2))
    plot_recovery_ratios(matches, output_dir / 'flux_ratio.png', show=show)
    print(f'{len(injected)} injected across {n_frames} frames')
    print(summary['status_counts'])
    print(group_summary)
    print('Outputs:', output_dir.resolve())
    return injected, detected, original_detected, matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits', type=Path,
                        default=project_dir / 'Astro' / 'Fits_Data' / 'mosaic.fits')
    parser.add_argument('--psf', type=Path,
                        default=Path(__file__).resolve().parent / 'outputs'
                        / 'psf_fit' / 'moffat_parameters.ecsv')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'outputs' / 'completeness_multiple')
    parser.add_argument('--n-sources', '--sources-per-frame', dest='n_sources',
                        type=int, default=None, help='Single group sources per frame (default: 5)')
    parser.add_argument('--n-bright', type=int, help='Bright sources per frame')
    parser.add_argument('--n-faint', type=int, help='Faint sources per frame')
    parser.add_argument('--bright-mag-range', type=float, nargs=2,
                        help='Bright group magnitude limits, e.g. 18 21')
    parser.add_argument('--faint-mag-range', type=float, nargs=2,
                        help='Faint group magnitude limits, e.g. 23 25')
    parser.add_argument('--bright-r-eff-range', type=float, nargs=2,
                        help='Bright group effective radii in pixels')
    parser.add_argument('--faint-r-eff-range', type=float, nargs=2,
                        help='Faint group effective radii in pixels')
    parser.add_argument('--n-frames', '--frames', '--max-frames', dest='n_frames',
                        type=int, default=3)
    parser.add_argument('--background', choices=('sim', 'image'), default='sim')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=200,
                        help='Side length for a simulated background')
    parser.add_argument('--x-range', type=int, nargs=2, default=(677, 940))
    parser.add_argument('--y-range', type=int, nargs=2, default=(4000, 4156))
    parser.add_argument('--profile', choices=('gaussian', 'exponential', 'deVau'),
                        default='exponential')
    parser.add_argument('--amplitude', type=float, default=200)
    brightness = parser.add_mutually_exclusive_group()
    brightness.add_argument('--amplitude-range', type=float, nargs=2)
    brightness.add_argument('--mag-range', type=float, nargs=2,
                            help='Uniform apparent magnitudes: bright_limit faint_limit')
    parser.add_argument('--r-eff', type=float, default=5)
    parser.add_argument('--r-eff-range', type=float, nargs=2)
    parser.add_argument('--ellipticity', type=float, default=0.3)
    parser.add_argument('--theta-deg', type=float, default=30)
    parser.add_argument('--random-angles', action='store_true')
    parser.add_argument('--edge-margin', type=float, default=25)
    parser.add_argument('--min-separation', type=float, default=0,
                        help='Optional minimum separation between injected centres')
    parser.add_argument('--half-size', type=int, default=50)
    parser.add_argument('--match-radius', type=float, default=3)
    parser.add_argument('--source-poisson', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.n_sources is not None and (args.n_bright is not None or args.n_faint is not None):
        parser.error('Choose --n-sources or --n-bright/--n-faint, not both')
    run_experiment(
        args.fits, args.psf, args.output,
        n_sources=5 if args.n_sources is None else args.n_sources, n_frames=args.n_frames,
        n_bright=args.n_bright, n_faint=args.n_faint,
        bright_mag_range=args.bright_mag_range, faint_mag_range=args.faint_mag_range,
        bright_r_eff_range=args.bright_r_eff_range,
        faint_r_eff_range=args.faint_r_eff_range,
        amplitude=args.amplitude, r_eff=args.r_eff, profile=args.profile,
        ellipticity=args.ellipticity, theta=np.deg2rad(args.theta_deg),
        amplitude_range=args.amplitude_range, mag_range=args.mag_range,
        r_eff_range=args.r_eff_range, random_angles=args.random_angles,
        background_type=args.background, seed=args.seed, size=args.size,
        x_range=args.x_range, y_range=args.y_range,
        edge_margin=args.edge_margin, min_separation=args.min_separation,
        half_size=args.half_size, match_radius=args.match_radius,
        source_poisson=args.source_poisson, show=args.show
    )


if __name__ == '__main__':
    main()