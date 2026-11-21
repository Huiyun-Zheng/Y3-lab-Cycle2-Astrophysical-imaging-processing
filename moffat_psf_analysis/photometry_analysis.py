"""Local Moffat classification and uncertainty-aware aperture number counts.

Run this file. All outputs are replaced in this folder's results directory.
Detection inputs, original source fluxes and all other folders are read-only.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scipy import ndimage as ndi
from scipy.stats import chi2
from astropy.stats import sigma_clipped_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from moffat_method import MoffatSettings, measure_profiles, classify_sources, moffat_pixels
from photometry_helpers import calibrate_magnitudes, cumulative_counts


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
FITS_PATH = PROJECT_DIR / 'Astro/Fits_Data/mosaic.fits'
DETECTION_DIR = PROJECT_DIR / 'neighbour_light_model/results'
OUTPUT_DIR = CODE_DIR / 'results'

# Less restrictive morphology settings, selected without using number-count slope.
ANALYSIS_SETTINGS = replace(
    MoffatSettings(),
    classification_mag_limit=23.0,
    reference_mag_limit=21.5,
    minimum_snr=20.0,
    reference_snr=60.0,
    isolation_radius=12.0,
    minimum_references=3,
    maximum_reference_distance=1500.0,
    max_relative_width_error=0.10,
    max_local_width_scatter=0.08,
    width_equivalence=0.20,
    confidence_multiplier=1.0,
    max_ellipticity_difference=0.20,
)
MAG_STEP = 0.5
MC_DRAWS = 2000
RANDOM_SEED = 20261118
BLANK_APERTURES = 48
MIN_BLANK_APERTURES = 24



def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def aperture_offsets(row):
    """Reproduce the detector's binary ellipse at its original subpixel centre."""
    x, y = float(row['x']), float(row['y'])
    a, b = float(row['aperture_a_pix']), float(row['aperture_b_pix'])
    half = int(np.ceil(max(a, b))) + 1
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    dx = xx - (x - round(x))
    dy = yy - (y - round(y))
    angle = np.radians(float(row['aperture_theta_deg']))
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    inside = (u / a)**2 + (v / b)**2 <= 1
    return xx[inside], yy[inside]


def estimate_blank_scale(residual, rms, forbidden, row, dx, dy, rng):
    """Scatter of non-overlapping, translated copies of the target aperture.

    Normalising each sum by its own RMS quadrature accounts for local depth.
    A three-pixel source-mask dilation limits detected-source wings. Undetected
    sources and residual wings are not guaranteed absent; use clipped scatter.
    """
    height, width = residual.shape
    a = max(float(row['aperture_a_pix']), float(row['aperture_b_pix']))
    half = int(np.ceil(a)) + 2
    samples, accepted_centres = [], []
    # Wider search is a fallback for crowded regions/large apertures, not a
    # replacement with smaller apertures. The actual radius is recorded.
    used_radius = 500.0
    for search_radius in (500.0, 1000.0):
        used_radius = search_radius
        for attempt in range(4):
            theta = rng.uniform(0, 2 * np.pi, 256)
            distance = np.sqrt(rng.uniform((2 * a + 20)**2, search_radius**2, 256))
            xs = np.rint(float(row['x']) + distance * np.cos(theta)).astype(int)
            ys = np.rint(float(row['y']) + distance * np.sin(theta)).astype(int)
            good = (xs >= half) & (xs < width - half) & (ys >= half) & (ys < height - half)
            xs, ys = xs[good], ys[good]
            if not len(xs):
                continue
            px, py = xs[:, None] + dx, ys[:, None] + dy
            clean = ~np.any(forbidden[py, px], axis=1)
            for x, y in zip(xs[clean], ys[clean]):
                if accepted_centres:
                    centres = np.asarray(accepted_centres)
                    if np.any(np.hypot(centres[:, 0] - x, centres[:, 1] - y) < 2 * a + 1):
                        continue
                aperture_rms = np.sqrt(np.sum(rms[y + dy, x + dx].astype(float)**2))
                samples.append(float(np.sum(residual[y + dy, x + dx], dtype=float)) / aperture_rms)
                accepted_centres.append((x, y))
                if len(samples) >= BLANK_APERTURES:
                    break
            if len(samples) >= BLANK_APERTURES:
                break
        if len(samples) >= MIN_BLANK_APERTURES:
            break
    if len(samples) < MIN_BLANK_APERTURES:
        # Explicitly incomplete estimate; retain the object, mark the fallback.
        return 1.0, len(samples), np.nan, used_radius, 'rms_only_fallback'
    _, median, scatter = sigma_clipped_stats(np.asarray(samples), sigma=3, maxiters=5)
    return max(1.0, float(scatter)), len(samples), float(median), used_radius, 'local_blank'


def add_photometric_uncertainties(table, residual, rms, labels, mask, header):
    """Background aperture scatter + source shot noise, with shared ZP separate."""
    result = table.copy()
    gain = float(header['GAIN'])
    zp_error = float(header.get('MAGZRR', 0.0))
    if not np.isfinite(zp_error) or zp_error < 0:
        raise ValueError('MAGZRR must be finite and nonnegative.')
    forbidden = (mask | ~np.isfinite(residual) | ~np.isfinite(rms) | (rms <= 0)
                 | ndi.binary_dilation(labels > 0, iterations=3))
    columns = ['flux_error_adu', 'background_flux_error_adu', 'source_flux_error_adu',
               'background_correlation_factor', 'blank_median_normalised', 'blank_search_radius',
               'magnitude_error_random', 'magnitude_error_total',
               'magnitude_error_lower', 'magnitude_error_upper', 'photometric_snr']
    for name in columns:
        result[name] = np.full(len(result), np.nan)
    result['blank_aperture_count'] = np.zeros(len(result), int)
    result['uncertainty_status'] = np.full(len(result), 'not_in_sample', dtype='U32')
    result['magnitude_error_zeropoint'] = np.full(len(result), zp_error)
    rng = np.random.default_rng(RANDOM_SEED)
    chosen = np.flatnonzero(result['count_sample'])
    for number, i in enumerate(chosen):
        row = result[i]
        dx, dy = aperture_offsets(row)
        x, y = int(round(float(row['x']))), int(round(float(row['y'])))
        px, py = x + dx, y + dy
        if (len(dx) == 0 or np.any(px < 0) or np.any(px >= residual.shape[1])
                or np.any(py < 0) or np.any(py >= residual.shape[0])):
            raise ValueError(f'Invalid accepted aperture: ID {row["id"]}.')
        variance = np.sum(rms[py, px].astype(float)**2)
        if not np.isfinite(variance) or variance <= 0 or np.any(mask[py, px]):
            raise ValueError(f'Invalid accepted-aperture noise/mask: ID {row["id"]}.')
        scale, count, median, search_radius, status = estimate_blank_scale(
            residual, rms, forbidden, row, dx, dy, rng
        )
        flux = float(row['aperture_flux_adu'])
        background_variance = variance * scale**2
        source_variance = flux / gain
        flux_error = np.sqrt(background_variance + source_variance)
        fractional_error = flux_error / flux
        result['flux_error_adu'][i] = flux_error
        result['background_flux_error_adu'][i] = np.sqrt(background_variance)
        result['source_flux_error_adu'][i] = np.sqrt(source_variance)
        result['background_correlation_factor'][i] = scale
        result['blank_aperture_count'][i] = count
        result['blank_median_normalised'][i] = median
        result['blank_search_radius'][i] = search_radius
        result['uncertainty_status'][i] = status
        result['photometric_snr'][i] = 1 / fractional_error
        random_mag_error = 2.5 / np.log(10) * fractional_error
        result['magnitude_error_random'][i] = random_mag_error
        result['magnitude_error_total'][i] = np.hypot(random_mag_error, zp_error)
        result['magnitude_error_lower'][i] = np.hypot(2.5 * np.log10(1 + fractional_error), zp_error)
        # A flux interval crossing zero has no finite faint magnitude endpoint.
        if fractional_error < 1:
            result['magnitude_error_upper'][i] = np.hypot(-2.5 * np.log10(1 - fractional_error), zp_error)
        if (number + 1) % 1000 == 0:
            print(f'Aperture uncertainties: {number + 1}/{len(chosen)}', flush=True)
    result.meta['error_note'] = ('Translated blank-aperture background scatter plus F_ADU/GAIN shot variance; '
                                 'MAGZRR is common to all sources. No flux correction applied.')
    summary = {
        'zeropoint_sigma_mag': zp_error,
        'effective_gain_assumed': gain,
        'blank_status': dict(Counter(str(v) for v in result['uncertainty_status'][chosen])),
        'median_background_factor': float(np.median(result['background_correlation_factor'][chosen])),
        'median_magnitude_random_error': float(np.median(result['magnitude_error_random'][chosen])),
        'median_magnitude_total_error': float(np.median(result['magnitude_error_total'][chosen])),
        'nonpositive_lower_flux_endpoints': int(np.count_nonzero(~np.isfinite(result['magnitude_error_upper'][chosen]))),
        'seed': RANDOM_SEED,
    }
    return result, summary


def counts_from_magnitudes(magnitudes, limits):
    """Strict N(<limit); nonpositive-flux draws have magnitude +infinity."""
    return np.searchsorted(np.sort(magnitudes), limits, side='left')


def count_error_model(flux, flux_error, header, limits, observed_counts, draws=MC_DRAWS):
    """Conditional flux-error propagation plus Poisson sampling covariance.

    No Poisson resampling occurs in the photometric simulation, so adding the
    Poisson covariance below does not add counting noise twice. Class labels,
    detected membership and area are fixed. This is not a completeness model.
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)
    simulated = np.zeros((draws, len(limits)), dtype=float)
    zero_point = float(header['MAGZPT'])
    conversion = float(header['GAIN']) / float(header['EXPTIME'])
    zp_error = float(header.get('MAGZRR', 0.0))
    for trial in range(draws):
        draw_flux = flux + rng.normal(0, flux_error)
        magnitude = np.full(len(flux), np.inf)
        positive = draw_flux > 0
        # Exactly one ZP offset per realisation, shared by the full catalogue.
        magnitude[positive] = zero_point + rng.normal(0, zp_error) - 2.5 * np.log10(draw_flux[positive] * conversion)
        simulated[trial] = counts_from_magnitudes(magnitude, limits)
    photometric_covariance = np.cov(simulated, rowvar=False, ddof=1)
    poisson_covariance = np.minimum.outer(observed_counts, observed_counts).astype(float)
    return simulated, poisson_covariance, photometric_covariance


def correlated_line_fit(x, y, covariance):
    """GLS with a pseudoinverse for duplicated/strongly correlated count points."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    keep = eigenvalues > max(np.max(eigenvalues) * 1e-8, 1e-14)
    if keep.sum() < 3:
        raise ValueError('Too few independent count constraints for a line fit.')
    inverse = (eigenvectors[:, keep] / eigenvalues[keep]) @ eigenvectors[:, keep].T
    origin = float(np.mean(x))
    design = np.column_stack([x - origin, np.ones(len(x))])
    normal = design.T @ inverse @ design
    coefficient_covariance = np.linalg.inv(normal)
    parameters = coefficient_covariance @ design.T @ inverse @ y
    residual = y - design @ parameters
    statistic = float(residual @ inverse @ residual)
    dof = int(keep.sum() - 2)
    return {
        'slope': float(parameters[0]), 'intercept': float(parameters[1] - origin * parameters[0]),
        'slope_error_formal': float(np.sqrt(coefficient_covariance[0, 0])),
        'chi2': statistic, 'dof': dof, 'retained_covariance_rank': int(keep.sum()),
        'reduced_chi2': statistic / dof,
    }


def descriptive_line_fit(x, y, covariance):
    """Original OLS trend, with correlated data errors propagated to its slope.

    This defines the same descriptive statistic as the old pipeline; it does
    not require the curve to be exactly linear. Its error excludes model bias.
    """
    origin = float(np.mean(x))
    design = np.column_stack([x - origin, np.ones(len(x))])
    projection = np.linalg.pinv(design)
    parameters = projection @ y
    parameter_covariance = projection @ covariance @ projection.T
    return {
        'slope': float(parameters[0]),
        'intercept': float(parameters[1] - origin * parameters[0]),
        'slope_error_propagated': float(np.sqrt(parameter_covariance[0, 0])),
    }


def plot_number_counts(table, area, output, fit_range, header):
    """Correlated count errors; retain the previous OLS trend for comparison."""
    selected = table['count_sample'] & ~table['is_star']
    limits, counts = cumulative_counts(table['magnitude'][selected], MAG_STEP)
    positive = counts > 0
    limits, counts = limits[positive], counts[positive]
    flux = np.asarray(table['aperture_flux_adu'][selected], float)
    error = np.asarray(table['flux_error_adu'][selected], float)
    if not np.all(np.isfinite(error)):
        raise ValueError('All count-sample sources need a finite flux error.')
    simulated, poisson, photometric = count_error_model(flux, error, header, limits, counts)
    conversion = np.log(10) * counts
    covariance_log = (poisson + photometric) / np.outer(conversion, conversion)
    y = np.log10(counts / area)
    fit_mask = (limits >= fit_range[0]) & (limits <= fit_range[1])
    if fit_range[0] >= fit_range[1] or fit_mask.sum() < 3:
        raise ValueError('Fit range must contain at least three points.')
    fit_covariance = covariance_log[np.ix_(fit_mask, fit_mask)]
    gls_fit = correlated_line_fit(limits[fit_mask], y[fit_mask], fit_covariance)
    fit = descriptive_line_fit(limits[fit_mask], y[fit_mask], fit_covariance)

    # Exact central Poisson endpoints, with first-order photometric log variance
    # added in quadrature. These combined bars are approximate, not exact Garwood.
    alpha = 1 - .682689492
    lower = .5 * chi2.ppf(alpha / 2, 2 * counts)
    upper = .5 * chi2.ppf(1 - alpha / 2, 2 * (counts + 1))
    photometric_log_error = np.sqrt(np.diag(photometric)) / conversion
    down = np.hypot(np.log10(counts / lower), photometric_log_error)
    up = np.hypot(np.log10(upper / counts), photometric_log_error)
    results = Table({
        'limiting_magnitude': limits, 'count': counts, 'log10_density': y,
        'poisson_lower_count': lower, 'poisson_upper_count': upper,
        'photometric_count_sigma': np.sqrt(np.diag(photometric)),
        'photometric_count_mean_shift': np.mean(simulated, axis=0) - counts,
        'log_error_lower': down, 'log_error_upper': up,
    })
    results.write(output / 'number_counts.ecsv', overwrite=True)
    np.savez_compressed(output / 'counts_covariance.npz', magnitude_limits=limits,
                        poisson_count_covariance=poisson, photometric_count_covariance=photometric,
                        total_log_covariance=covariance_log)
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.errorbar(limits, y, yerr=[down, up], fmt='o-', color='royalblue', markersize=4,
                capsize=2, label='Counts: Poisson + photometric errors')
    ax.axvspan(*fit_range, color='grey', alpha=.10, label=f'Fit range: {fit_range[0]:g}–{fit_range[1]:g}')
    xx = np.linspace(limits[fit_mask].min(), limits[fit_mask].max(), 200)
    line = np.poly1d([fit['slope'], fit['intercept']])
    ax.plot(xx, line(xx), color='red', linewidth=2,
            label=f"Descriptive slope = {fit['slope']:.3f} ± {fit['slope_error_propagated']:.3f}")
    anchor = np.mean(xx)
    ax.plot(limits, line(anchor) + .6 * (limits - anchor), '--', color='grey', label='Reference slope = 0.6')
    ax.set(xlabel='Limiting aperture magnitude, r',
           ylabel=r'$\log_{10}[(N(<m)/A)/(1\,\mathrm{deg}^{-2})]$', title='Observed cumulative counts')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(output / 'cumulative_number_counts.png', dpi=220)
    plt.close(fig)
    return {
        'descriptive_ols_fit': fit, 'gls_diagnostic': gls_fit,
        'mc_draws': MC_DRAWS, 'seed': RANDOM_SEED + 1,
        'fit_range_fixed': list(fit_range),
        'error_model': 'Poisson min(N_i,N_j) covariance + conditional flux MC covariance with shared MAGZRR offset.',
        'plot_errors': 'Garwood log errors plus first-order photometric log variance in quadrature; approximate combined intervals.',
        'slope_error_note': 'OLS slope is the previous descriptive statistic; its error uses A C A.T with correlated C. It excludes classification/model biases. GLS is saved separately as a straight-line adequacy diagnostic.',
        'excluded_uncertainties': ['classification', 'completeness', 'cosmic variance', 'effective-gain uncertainty', 'area uncertainty'],
    }


def plot_width_magnitude(table, output):
    valid = table['count_sample'] & np.isfinite(table['psf_fwhm'])
    stars = np.asarray(table['is_star'], bool)
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    for selection, colour, label in [
        (valid & ~stars, 'royalblue', 'Remaining sources'),
        (valid & stars, 'orange', 'Likely stars'),
    ]:
        finite_error = (selection & np.isfinite(table['magnitude_error_lower'])
                        & np.isfinite(table['magnitude_error_upper']))
        ax.errorbar(
            table['magnitude'][finite_error], table['psf_fwhm'][finite_error],
            xerr=[table['magnitude_error_lower'][finite_error], table['magnitude_error_upper'][finite_error]],
            fmt='none', ecolor=colour, elinewidth=.55, alpha=.25, zorder=1,
        )
        ax.scatter(table['magnitude'][selection], table['psf_fwhm'][selection],
                   s=14, alpha=.7, color=colour, edgecolors='none', zorder=2,
                   label=f'{label}: {selection.sum():,} plotted')
    references = table['psf_reference']
    if np.any(references):
        ax.axhline(np.median(table['psf_fwhm'][references]), color='orange',
                   linestyle='--', label='Median reference width')
    ax.set(xlabel='Calibrated aperture magnitude, r', ylabel='Moffat FWHM (pixel)',
           title='Moffat star selection')
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    fig.savefig(output / 'moffat_width_vs_magnitude.png', dpi=220)
    plt.close(fig)


def plot_diagnostics(image, table, profiles, output):
    diagnostic_dir = output / 'reference_diagnostics'
    diagnostic_dir.mkdir(exist_ok=True)
    for filename, selection in [('references', table['psf_reference']), ('star_candidates', table['is_star'])]:
        # Always rewrite the panel, including the no-candidate case.
        rows = np.flatnonzero(selection)
        rows = rows[np.argsort(table['magnitude'][rows])][:24]
        fig, axes = plt.subplots(max(1, int(np.ceil(len(rows) / 6))), 6,
                                 figsize=(15, 3.2 * max(1, int(np.ceil(len(rows) / 6)))),
                                 squeeze=False, layout='constrained')
        for ax in axes.ravel():
            ax.axis('off')
        for ax, i in zip(axes.ravel(), rows):
            row = table[i]
            x, y = int(round(row['x'])), int(round(row['y']))
            cutout = image[max(0, y - 15):y + 16, max(0, x - 15):x + 16]
            lo, hi = np.nanpercentile(cutout, [10, 99])
            ax.imshow(cutout, origin='lower', cmap='gray', vmin=lo, vmax=hi)
            ax.set_title(f"ID {row['id']} | m={row['magnitude']:.2f}", fontsize=9)
        fig.suptitle(f'{filename.replace("_", " ").title()} ({selection.sum()})')
        fig.savefig(diagnostic_dir / f'{filename}.png', dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 8), layout='constrained')
    ax.scatter(table['x'], table['y'], s=1, c='0.85')
    refs = table['psf_reference']
    if np.any(refs):
        points = ax.scatter(table['x'][refs], table['y'][refs], c=table['psf_fwhm'][refs], s=30)
        fig.colorbar(points, ax=ax, label='Reference FWHM (pixel)')
    ax.set(xlabel='x (pixel)', ylabel='y (pixel)', title='PSF reference coverage', aspect='equal')
    fig.savefig(diagnostic_dir / 'reference_map.png', dpi=180)
    plt.close(fig)

    shown = profiles[:6]
    fig, axes = plt.subplots(max(1, len(shown)), 3, figsize=(10, 2.7 * max(1, len(shown))), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for panels, (identifier, stamp, parameters) in zip(axes, shown):
        model = moffat_pixels(stamp['x'], stamp['y'], parameters)
        values = [(stamp['values'] - stamp['sky']) / stamp['noise'],
                  (model - stamp['sky']) / stamp['noise'], (stamp['values'] - model) / stamp['error']]
        half = int(max(abs(stamp['x']).max(), abs(stamp['y']).max()))
        for ax, value, title in zip(panels, values, ['Data', 'Local Moffat fit', 'Residual / error']):
            grid = np.full((2 * half + 1, 2 * half + 1), np.nan)
            grid[stamp['y'].astype(int) + half, stamp['x'].astype(int) + half] = value
            ax.imshow(grid, origin='lower', cmap='gray', vmin=-3,
                      vmax=3 if title.startswith('Residual') else max(10, np.percentile(values[0], 99)))
            ax.set_title(f'ID {identifier}: {title}', fontsize=9)
    fig.tight_layout()
    fig.savefig(diagnostic_dir / 'star_profile_checks.png', dpi=150)
    plt.close(fig)


def run_analysis(settings=None, fit_range=(19.0, 24.0), reuse_measurements=False, reference_ids=None):
    settings = settings or ANALYSIS_SETTINGS
    sources_path = DETECTION_DIR / 'sources.ecsv'
    products_path = DETECTION_DIR / 'detection_products.fits'
    input_hashes = {str(p): file_hash(p) for p in [sources_path, products_path, FITS_PATH]}
    fingerprint = {'inputs': input_hashes, 'settings': asdict(settings),
                   'measurement_code': file_hash(CODE_DIR / 'moffat_method.py'),
                   'calibration_code': file_hash(CODE_DIR / 'photometry_helpers.py')}
    original = Table.read(sources_path)
    with fits.open(FITS_PATH) as hdul:
        image = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()
    with fits.open(products_path) as hdul:
        labels = hdul['SOURCES'].data.copy()
        mask = hdul['DETMASK'].data.astype(bool)
        background = hdul['BACKGROUND'].data.copy()
        rms = hdul['RMS'].data.copy()
        if not all(a.shape == image.shape for a in [labels, mask, background, rms]):
            raise ValueError('Image and detection-product shapes differ.')
        mask |= ~np.isfinite(image) | ~np.isfinite(background) | ~np.isfinite(rms) | (rms <= 0)
        header['SATLEVEL'] = float(hdul[0].header['SATLEVEL'])
    area = np.count_nonzero(~mask) * abs(float(header['PIXSCAL1']) * float(header['PIXSCAL2'])) / 3600**2
    if not np.isfinite(area) or area <= 0:
        raise ValueError('Invalid unmasked sky area.')
    calibrated = calibrate_magnitudes(original, header)
    OUTPUT_DIR.mkdir(exist_ok=True)
    cache = OUTPUT_DIR / 'measured_sources.ecsv'
    cache_info = OUTPUT_DIR / 'measurement_manifest.json'
    if reuse_measurements:
        if not cache_info.exists() or json.loads(cache_info.read_text()) != fingerprint:
            raise ValueError('Measurements do not match inputs, settings or code; run without --reuse-measurements.')
        measured = Table.read(cache)
        if not np.array_equal(measured['id'], original['id']):
            raise ValueError('Cached catalogue rows do not match.')
    else:
        measured = measure_profiles(image, labels, mask, calibrated, header, settings)
        measured.write(cache, overwrite=True)
        cache_info.write_text(json.dumps(fingerprint, indent=2))
    print('Comparing sources with local Moffat references...', flush=True)
    classified, diagnostics = classify_sources(image, labels, mask, measured, header, settings, reference_ids)
    classified.meta['classification_note'] = 'Only precise local PSF matches removed; unknowns retained. Candidates, not confirmed stars.'
    classified.meta['settings'] = json.dumps(asdict(settings))
    classified.meta['input_sources_sha256'] = input_hashes[str(sources_path)]
    classified.meta['width_note'] = 'Geometric-mean FWHM of pre-pixel-integration elliptical Moffat; local_scale is not intrinsic galaxy size.'
    for name in original.colnames:
        a, b = np.asarray(original[name]), np.asarray(classified[name])
        if a.dtype.kind in 'fc':
            assert np.allclose(a, b, equal_nan=True), name
        else:
            assert np.array_equal(a, b), name
    assert not np.any(classified['is_star'] & ~classified['count_sample'])
    assert not np.any(classified['is_star'] & classified['blended'])
    for i in np.flatnonzero(classified['is_star']):
        assert str(classified['id'][i]) not in classified['reference_ids'][i].split(',')
    classified, uncertainty_summary = add_photometric_uncertainties(
        classified, image - background, rms, labels, mask, header
    )
    classified.write(OUTPUT_DIR / 'classified_sources.ecsv', overwrite=True)
    classified[classified['is_star']].write(OUTPUT_DIR / 'star_candidates.ecsv', overwrite=True)
    classified[classified['psf_reference']].write(OUTPUT_DIR / 'reference_stars.ecsv', overwrite=True)
    remaining = classified['count_sample'] & ~classified['is_star']
    classified[remaining].write(OUTPUT_DIR / 'remaining_sources.ecsv', overwrite=True)
    plot_width_magnitude(classified, OUTPUT_DIR)
    plot_diagnostics(image, classified, diagnostics, OUTPUT_DIR)
    counts_summary = plot_number_counts(classified, area, OUTPUT_DIR, fit_range, header)
    refs = classified['psf_reference']
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'Pixel-integrated elliptical Moffat; local profile and precision-bounded width equivalence',
        'settings': asdict(settings), 'input_hashes': input_hashes,
        'checked_reference_ids': reference_ids,
        'photometric_sources': int(classified['count_sample'].sum()),
        'star_candidates_removed': int(classified['is_star'].sum()),
        'remaining_sources': int(remaining.sum()), 'reference_sources': int(refs.sum()),
        'reference_fwhm_median_pix': float(np.median(classified['psf_fwhm'][refs])) if np.any(refs) else None,
        'classes': dict(Counter(str(v) for v in classified['source_class'][classified['count_sample']])),
        'fit_status': dict(Counter(str(v) for v in classified['fit_status'][classified['count_sample']])),
        'classification_reasons': dict(Counter(str(v) for v in classified['classification_reason'][classified['count_sample']])),
        'area_deg2': float(area), 'fit_range': list(fit_range),
        'photometric_uncertainty': uncertainty_summary, 'counts_analysis': counts_summary,
        'original_catalogue_columns_preserved': True,
        'limitations': [
            'Automatic references are provisional; inspect the reference panels. No external truth labels used.',
            'Compact galaxies and unresolved nonstellar sources can mimic stars.',
            'All blended sources and close neighbours are conservatively retained; residual stars remain.',
            'No reliable local references or low precision means unknown, not confirmed galaxy.',
            'Width-equivalence cuts and residual gates are heuristic, not calibrated star probabilities.',
            'Formal errors omit mosaic covariance; header gain as effective gain and electron/s zero point are inherited assumptions.',
            'Elliptical Moffat is an empirical observed-profile approximation, not a PSF-convolved galaxy model.',
            'Counts errors include Poisson, conditional flux perturbations and a shared zero point; no classification, completeness or cosmic-variance model.',
            'Blank apertures account for local background scatter, not a common unknown background bias.',
            'GLS is a descriptive straight-line fit with approximate correlated errors, not a cosmological inference.',
        ],
    }
    (OUTPUT_DIR / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Outputs: {OUTPUT_DIR}', flush=True)
    return classified, summary, OUTPUT_DIR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-range', type=float, nargs=2, default=(19.0, 24.0))
    parser.add_argument('--measurement-mag-limit', type=float, default=ANALYSIS_SETTINGS.measurement_mag_limit)
    parser.add_argument('--classification-mag-limit', type=float, default=ANALYSIS_SETTINGS.classification_mag_limit)
    parser.add_argument('--reference-ids', type=int, nargs='+', help='Optional visually checked source IDs, not segment IDs.')
    parser.add_argument('--reuse-measurements', action='store_true')
    args = parser.parse_args()
    settings = replace(ANALYSIS_SETTINGS, measurement_mag_limit=args.measurement_mag_limit,
                       classification_mag_limit=args.classification_mag_limit)
    run_analysis(settings, tuple(args.fit_range), args.reuse_measurements, args.reference_ids)
