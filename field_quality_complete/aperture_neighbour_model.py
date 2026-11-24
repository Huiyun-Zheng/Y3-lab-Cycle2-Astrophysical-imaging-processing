"""Reassess neighbour-affected sources using jointly fitted light profiles.

Size-constrained variant used by run_aperture_review.py.
Fit observed FWHM directly, with bounds tied to each detected source size.
Read existing strict detections; preserve input detections and apertures.
These are empirical image-profile fits, not a calibrated PSF/galaxy decomposition.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import shutil

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import least_squares
from scipy import ndimage as ndi
from scipy.special import logsumexp, expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import detection_snapshot as detector


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
PARAMETERS_PER_SOURCE = 7


@dataclass
class Settings:
    # New definitions: mean pixel contamination and integrated contamination.
    # Both use model light, including wings outside segmentation labels.
    max_overlap_area_fraction: float = 0.10
    max_overlap_flux_fraction: float = 0.10
    # Fit beta freely from two starting points to check solution stability.
    profile_betas: tuple = (2.5, 4.0)
    max_group_sources: int = 24
    max_fit_pixels: int = 30000
    max_nfev: int = 200
    max_reduced_residual: float = 5.0
    max_coherent_residual: float = 5.0
    max_score_disagreement: float = 0.03
    minimum_model_snr: float = 10.0
    # Apply this measured aperture S/N cut to old and recovered sources alike.
    minimum_aperture_snr: float = 5.0
    # Allow a small correction to a centroid measured from a split segment.
    max_centre_shift: float = 1.5
    # A high-S/N profile can have significant but small fractional residuals.
    # These are provisional quality tolerances, not confidence intervals.
    max_fractional_residual: float = 0.08
    max_bound_neighbour_fraction: float = 0.01
    max_sky_flux_fraction: float = 0.03
    # Fit FWHM directly so a tiny neighbour cannot grow an arbitrary broad core.
    # The bound follows its measured size, with room for truncated faint sources.
    max_fwhm_moment_factor: float = 3.0
    minimum_fwhm_upper_bound: float = 6.0

    def __post_init__(self):
        for value in (self.max_overlap_area_fraction, self.max_overlap_flux_fraction):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Contamination limits must lie in [0, 1].')
        for name, value in asdict(self).items():
            if name.startswith('max_overlap') or name == 'profile_betas':
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive.')


def aperture_grid(row, shape):
    x, y = float(row['x']), float(row['y'])
    a, b = float(row['aperture_a_pix']), float(row['aperture_b_pix'])
    theta = np.radians(row['aperture_theta_deg'])
    c, s = np.cos(theta), np.sin(theta)
    ex, ey = np.hypot(a * c, b * s), np.hypot(a * s, b * c)
    x0, x1 = max(0, int(np.floor(x - ex))), min(shape[1], int(np.ceil(x + ex)) + 1)
    y0, y1 = max(0, int(np.floor(y - ey))), min(shape[0], int(np.ceil(y + ey)) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    u, v = (xx - x) * c + (yy - y) * s, -(xx - x) * s + (yy - y) * c
    mask = (u / a) ** 2 + (v / b) ** 2 <= 1
    edge = x - ex < 0 or y - ey < 0 or x + ex >= shape[1] or y + ey >= shape[0]
    return (slice(y0, y1), slice(x0, x1)), xx, yy, mask, edge


def coordinates(rows, xx, yy):
    theta = np.radians(np.asarray(rows['aperture_theta_deg'], float))
    dx = np.asarray(xx).ravel()[:, None] - np.asarray(rows['x'])[None, :]
    dy = np.asarray(yy).ravel()[:, None] - np.asarray(rows['y'])[None, :]
    return dx * np.cos(theta) + dy * np.sin(theta), -dx * np.sin(theta) + dy * np.cos(theta)


def log_components(parameters, u, v, beta):
    # Parameters: log amplitude, log FWHM axes, offsets, angle, fitted beta.
    # The beta argument is the starting value, retained for the calling code.
    p = np.asarray(parameters[:-1]).reshape(-1, PARAMETERS_PER_SOURCE)
    du, dv = u - p[:, 3], v - p[:, 4]
    c, s = np.cos(p[:, 5]), np.sin(p[:, 5])
    uu, vv = du * c + dv * s, -du * s + dv * c
    width_conversion = 4 * np.expm1(np.log(2) / p[:, 6])
    q = width_conversion * (uu ** 2 * np.exp(-2 * p[:, 1])
                            + vv ** 2 * np.exp(-2 * p[:, 2]))
    return p[:, 0] - p[:, 6] * np.log1p(q)


def fit_profiles(rows, xx, yy, data, error, beta, settings):
    """Simultaneously fit positive Moffat profiles plus one residual sky level.

    Allow small centre shifts within the model; catalogue apertures stay fixed.
    Fit amplitudes, widths, angles and beta; keep the catalogue apertures fixed.
    """
    u, v = coordinates(rows, xx, yy)
    # Moment-to-Moffat conversion is only an initial guess, not a fixed width.
    aa = np.clip(np.asarray(rows['major_sigma_pix']) * np.sqrt(2 * (beta - 2)), 1, 25)
    bb = np.clip(np.asarray(rows['minor_sigma_pix']) * np.sqrt(2 * (beta - 2)), 1, 25)
    aa *= 2 * np.sqrt(np.expm1(np.log(2) / beta))
    bb *= 2 * np.sqrt(np.expm1(np.log(2) / beta))
    amplitudes = np.maximum(np.asarray(rows['peak_adu']) - np.median(rows['local_background']), 1)
    p0 = np.column_stack([np.log(amplitudes), np.log(aa), np.log(bb),
                          np.zeros(len(rows)), np.zeros(len(rows)),
                          np.zeros(len(rows)), np.full(len(rows), beta)]).ravel()
    sky_limit = 3 * float(np.median(error))
    p0 = np.r_[p0, 0.0]
    shift = settings.max_centre_shift
    # Angle is periodic: an artificial angle boundary can trap a valid ellipse.
    lower = np.r_[np.tile([np.log(.01), np.log(.6), np.log(.6), -shift, -shift, -np.inf, 1.2], len(rows)), -sky_limit]
    upper_sources = np.tile([np.log(1e6), 0., 0., shift, shift, np.inf, 12.0], (len(rows), 1))
    upper_sources[:, 1] = np.log(np.maximum(settings.minimum_fwhm_upper_bound,
                                           settings.max_fwhm_moment_factor * np.asarray(rows['major_sigma_pix'])))
    upper_sources[:, 2] = np.log(np.maximum(settings.minimum_fwhm_upper_bound,
                                           settings.max_fwhm_moment_factor * np.asarray(rows['minor_sigma_pix'])))
    upper = np.r_[upper_sources.ravel(), sky_limit]
    p0 = np.minimum(np.maximum(p0, lower + 1e-6), upper - 1e-6)

    def residual(p):
        return (np.exp(log_components(p, u, v, beta)).sum(axis=1) + p[-1] - data) / error

    def jacobian(p):
        pars = p[:-1].reshape(-1, PARAMETERS_PER_SOURCE)
        du, dv = u - pars[:, 3], v - pars[:, 4]
        c, s = np.cos(pars[:, 5]), np.sin(pars[:, 5])
        uu, vv = du * c + dv * s, -du * s + dv * c
        fitted_beta = pars[:, 6]
        width_conversion = 4 * np.expm1(np.log(2) / fitted_beta)
        inv_a2 = width_conversion * np.exp(-2 * pars[:, 1])
        inv_b2 = width_conversion * np.exp(-2 * pars[:, 2])
        qa, qb = uu ** 2 * inv_a2, vv ** 2 * inv_b2
        q = 1 + qa + qb
        model = np.exp(pars[:, 0] - fitted_beta * np.log(q))
        jac = np.empty((len(data), len(p)))
        jac[:, 0:-1:7] = model
        jac[:, 1:-1:7] = 2 * fitted_beta * model * qa / q
        jac[:, 2:-1:7] = 2 * fitted_beta * model * qb / q
        jac[:, 3:-1:7] = 2 * fitted_beta * model * (uu * c * inv_a2 - vv * s * inv_b2) / q
        jac[:, 4:-1:7] = 2 * fitted_beta * model * (uu * s * inv_a2 + vv * c * inv_b2) / q
        jac[:, 5:-1:7] = -2 * fitted_beta * model * uu * vv * (inv_a2 - inv_b2) / q
        # Beta also changes alpha at fixed FWHM: include this chain-rule term.
        beta_term = (np.log(2) / fitted_beta
                     * np.exp(np.log(2) / fitted_beta)
                     / np.expm1(np.log(2) / fitted_beta))
        jac[:, 6:-1:7] = model * (-np.log(q) + (qa + qb) / q * beta_term)
        jac[:, -1] = 1
        return jac / error[:, None]

    fit = least_squares(residual, p0, jac=jacobian, bounds=(lower, upper),
                        max_nfev=settings.max_nfev, loss='linear', x_scale='jac',
                        ftol=1e-6, xtol=1e-6, gtol=1e-6)
    return fit


def contamination_scores(log_models, target):
    """Continuous fractional area and flux, evaluated stably in log space."""
    others = np.delete(log_models, target, axis=1)
    if others.shape[1] == 0:
        return 0.0, 0.0
    target_light = log_models[:, target]
    neighbour_light = logsumexp(others, axis=1)
    # Mean fractional contamination per aperture pixel: effective area fraction.
    area_fraction = float(np.mean(expit(neighbour_light - target_light)))
    # Fraction of summed model light contributed by all neighbours.
    flux_fraction = float(expit(logsumexp(neighbour_light) - logsumexp(target_light)))
    return area_fraction, flux_fraction


def select_sources(table, settings):
    """Recover qualified neighbours, then apply the same S/N cut to everyone."""
    recover = (np.asarray(table['reassessment_candidate']) & np.asarray(table['model_valid'])
               & (table['model_area_fraction'] < settings.max_overlap_area_fraction)
               & (table['model_flux_fraction'] < settings.max_overlap_flux_fraction))
    signal_ok = np.isfinite(table['aperture_snr']) & (table['aperture_snr'] >= settings.minimum_aperture_snr)
    return (np.asarray(table['strict_usable']) | recover) & signal_ok


def score_catalogue(table, residual, noise, bad, saturation, settings, gain, only_parents=None):
    result = table.copy()
    result['strict_usable'] = np.asarray(table['usable'], bool)
    result['original_flags'] = np.asarray(table['flags'], str)
    result['original_aperture_flux_adu'] = np.asarray(table['aperture_flux_adu'], float)
    for name in ['model_area_fraction', 'model_flux_fraction', 'model_residual_score',
                 'model_score_disagreement', 'model_aperture_snr', 'model_coherent_residual',
                 'model_aperture_flux_adu', 'model_fractional_residual',
                 'model_bound_neighbour_fraction', 'model_sky_flux_fraction',
                 'aperture_snr', 'aperture_flux_error_adu']:
        result[name] = np.full(len(table), np.nan)
    result['model_valid'] = np.zeros(len(table), bool)
    result['model_status'] = np.full(len(table), 'not_reassessed', dtype='U32')
    result['local_background'] = np.asarray(table['local_background'], float)
    # Include source shot noise. GAIN must convert this image's ADU to electrons.
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('GAIN must be positive and match the image units.')
    total_noise = np.sqrt(noise ** 2 + np.maximum(residual, 0) / gain)
    hard_failure = np.zeros(len(table), bool)
    for i, row in enumerate(table):
        bounds, _, _, aperture, edge = aperture_grid(row, residual.shape)
        values = residual[bounds][aperture]
        errors = total_noise[bounds][aperture]
        hard_failure[i] = (edge or row['near_artefact'] or row['near_boundary']
                           or row['aperture_capped'] or row['elongation'] > 6
                           or np.any(bad[bounds][aperture])
                           or not np.all(np.isfinite(values))
                           or not np.all(np.isfinite(errors)) or np.any(errors <= 0))
        if len(values) and not hard_failure[i]:
            flux_error = float(np.sqrt(np.sum(errors ** 2)))
            result['aperture_flux_error_adu'][i] = flux_error
            result['aperture_snr'][i] = float(values.sum() / flux_error)
    # Contaminated but unblended sources also deserve a neighbour assessment.
    reassess = (~result['strict_usable'] & ~hard_failure
                & (table['blended'] | table['aperture_contaminated']))
    result['reassessment_candidate'] = reassess
    result['model_status'][hard_failure] = 'hard_quality_failure'
    xy = np.column_stack([table['x'], table['y']])
    parents = np.unique(table['parent_id'][reassess])
    if only_parents is not None:
        parents = np.intersect1d(parents, only_parents)
    diagnostics = []

    for count, parent in enumerate(parents):
        targets = np.flatnonzero((table['parent_id'] == parent) & reassess)
        group = table[targets]
        pad = np.maximum(np.asarray(group['aperture_a_pix']) + 5,
                         3 * np.asarray(group['major_sigma_pix']) + 8)
        x0 = max(0, int(np.floor(np.min(group['x'] - pad))))
        x1 = min(residual.shape[1], int(np.ceil(np.max(group['x'] + pad))) + 1)
        y0 = max(0, int(np.floor(np.min(group['y'] - pad))))
        y1 = min(residual.shape[0], int(np.ceil(np.max(group['y'] + pad))) + 1)
        # Include all detected neighbours with substantial potential wings nearby,
        # not only members of the target's original parent region.
        support = np.maximum(12, 4 * np.asarray(table['major_sigma_pix']))
        included = np.flatnonzero((xy[:, 0] + support >= x0) & (xy[:, 0] - support < x1)
                                  & (xy[:, 1] + support >= y0) & (xy[:, 1] - support < y1))
        if len(included) > settings.max_group_sources or (x1 - x0) * (y1 - y0) > settings.max_fit_pixels:
            result['model_status'][targets] = 'group_too_large'
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        data, error = residual[y0:y1, x0:x1], total_noise[y0:y1, x0:x1]
        clean = (~bad[y0:y1, x0:x1] & np.isfinite(data) & np.isfinite(error) & (error > 0))
        if clean.sum() < max(100, 10 * (PARAMETERS_PER_SOURCE * len(included) + 1)):
            result['model_status'][targets] = 'insufficient_pixels'
            continue
        rows = result[included]
        fits_for_group = []
        try:
            for beta in settings.profile_betas:
                fit = fit_profiles(rows, xx[clean], yy[clean], data[clean], error[clean], beta, settings)
                fits_for_group.append((beta, fit))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            result['model_status'][targets] = 'fit_failed'
            continue
        success = all(f.success for _, f in fits_for_group)
        if not success:
            result['model_status'][targets] = 'fit_not_converged'
            continue

        for i in targets:
            row = table[i]
            bounds, ax, ay, aperture, edge = aperture_grid(row, residual.shape)
            hard = (edge or row['near_artefact'] or row['near_boundary'] or row['aperture_capped']
                    or row['elongation'] > 6 or np.any(bad[bounds][aperture])
                    or not np.all(np.isfinite(residual[bounds][aperture])))
            if hard:
                result['model_status'][i] = 'hard_quality_failure'
                continue
            # Test all included neighbour peaks in the actual aperture.
            target_index = int(np.flatnonzero(included == i)[0])
            # A bound on this source's amplitude or size still makes it unreliable.
            start = PARAMETERS_PER_SOURCE * target_index
            if any(np.any(f.active_mask[start:start + 3])
                   for _, f in fits_for_group):
                result['model_status'][i] = 'target_profile_at_bound'
                continue
            other = rows[np.arange(len(rows)) != target_index]
            px = np.asarray(other['peak_x'], int) - bounds[1].start
            py = np.asarray(other['peak_y'], int) - bounds[0].start
            inside = (px >= 0) & (px < aperture.shape[1]) & (py >= 0) & (py < aperture.shape[0])
            if np.any(aperture[py[inside], px[inside]]):
                result['model_status'][i] = 'neighbour_peak_inside'
                continue
            u, v = coordinates(rows, ax[aperture], ay[aperture])
            scores, residual_scores, snrs, coherent = [], [], [], []
            fractional_residuals, bound_fractions, sky_fractions = [], [], []
            for beta, fit in fits_for_group:
                components = log_components(fit.x, u, v, beta)
                scores.append(contamination_scores(components, target_index))
                full_u, full_v = coordinates(rows, ax, ay)
                full_model = np.exp(log_components(fit.x, full_u, full_v, beta)).sum(axis=1).reshape(ax.shape) + fit.x[-1]
                whitened = (residual[bounds] - full_model) / total_noise[bounds]
                # Detect spatially coherent mismatch hidden by an average residual.
                kernel = np.exp(-np.arange(-4, 5, dtype=float) ** 2 / 2)
                kernel /= kernel.sum()
                smooth_rms = float(np.sum(kernel ** 2))
                smoothed = ndi.gaussian_filter(whitened, 1.0, mode='nearest') / smooth_rms
                coherent.append(float(np.percentile(np.abs(smoothed[aperture]), 95)))
                model = np.exp(components).sum(axis=1) + fit.x[-1]
                # This is a local residual score, not a formal reduced chi-square.
                # Mosaic correlations and profile mismatch are not calibrated.
                residual_scores.append(float(np.mean(((model - residual[bounds][aperture])
                                                      / total_noise[bounds][aperture]) ** 2)))
                snrs.append(float(np.exp(components[:, target_index]).sum()
                                  / np.sqrt(np.sum(total_noise[bounds][aperture] ** 2))))
                measured_flux = float(residual[bounds][aperture].sum())
                denominator = max(measured_flux, np.finfo(float).tiny)
                # Do not demand a perfect Moffat shape from a bright galaxy.
                # Still reject a mismatch large relative to its measured flux.
                fractional_residuals.append(float(np.abs(model - residual[bounds][aperture]).sum()
                                                 / denominator))
                # Angle endpoints describe equivalent axes, so ignore that bound.
                active = fit.active_mask[:-1].reshape(-1, PARAMETERS_PER_SOURCE)
                bound_sources = np.any(active[:, [0, 1, 2, 3, 4, 6]] != 0, axis=1)
                bound_sources[target_index] = False
                bound_fractions.append(float(np.exp(components[:, bound_sources]).sum() / denominator))
                # The extra fitted sky is not subtracted from the saved aperture.
                # Its contribution must therefore be small before accepting it.
                sky_fractions.append(float(abs(fit.x[-1]) * aperture.sum() / denominator))
            scores = np.asarray(scores)
            upper_scores = np.max(scores, axis=0)
            disagreement = float(np.max(np.ptp(scores, axis=0)))
            result['model_area_fraction'][i], result['model_flux_fraction'][i] = upper_scores
            result['model_residual_score'][i] = max(residual_scores)
            result['model_score_disagreement'][i] = disagreement
            result['model_aperture_snr'][i] = min(snrs)
            result['model_coherent_residual'][i] = max(coherent)
            result['model_fractional_residual'][i] = max(fractional_residuals)
            result['model_bound_neighbour_fraction'][i] = max(bound_fractions)
            result['model_sky_flux_fraction'][i] = max(sky_fractions)
            small_mismatch = max(fractional_residuals) <= settings.max_fractional_residual
            if not np.all(np.isfinite(scores)) or np.any(upper_scores < 0):
                result['model_status'][i] = 'unresolved_model_score'
            elif max(bound_fractions) > settings.max_bound_neighbour_fraction:
                result['model_status'][i] = 'unstable_neighbour'
            elif max(sky_fractions) > settings.max_sky_flux_fraction:
                result['model_status'][i] = 'local_sky_sensitive'
            elif max(residual_scores) > settings.max_reduced_residual and not small_mismatch:
                result['model_status'][i] = 'poor_profile_fit'
            elif max(coherent) > settings.max_coherent_residual and not small_mismatch:
                result['model_status'][i] = 'structured_residual'
            elif disagreement > settings.max_score_disagreement:
                result['model_status'][i] = 'profile_sensitive'
            elif min(snrs) < settings.minimum_model_snr:
                result['model_status'][i] = 'low_model_snr'
            else:
                result['model_status'][i] = 'ok'
                result['model_valid'][i] = True
                # Compute the measured sum, including neighbour light. We do not
                # silently substitute or subtract model flux from photometry.
                result['model_aperture_flux_adu'][i] = float(residual[bounds][aperture].sum())
            if result['model_valid'][i] and len(diagnostics) < 10:
                beta, fit = fits_for_group[0]
                all_u, all_v = coordinates(rows, xx, yy)
                model_cutout = np.exp(log_components(fit.x, all_u, all_v, beta)).sum(axis=1).reshape(xx.shape) + fit.x[-1]
                diagnostics.append((int(row['id']), (x0, x1, y0, y1), data.copy(), model_cutout, error.copy()))
        if count % 50 == 0:
            print(f'Groups checked: {count + 1}/{len(parents)}', flush=True)
    return result, diagnostics


def save_diagnostics(diagnostics, output):
    for identifier, bounds, data, model, error in diagnostics[:10]:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        for ax, values, title in zip(axes, [data / error, model / error, (data - model) / error],
                                    ['Data / error', 'Model / error', 'Residual / error']):
            ax.imshow(values, origin='lower', cmap='gray', vmin=-3, vmax=12 if title != 'Residual / error' else 3)
            ax.set_title(title)
        fig.suptitle(f'Source {identifier}')
        fig.tight_layout()
        fig.savefig(output / f'fit_{identifier}.png', dpi=140)
        plt.close(fig)


def save_bright_review(residual, noise, catalogue, output, recovered_only=False):
    """Show bright sources at a useful scale for a visual quality check."""
    from matplotlib.patches import Ellipse

    indices = np.arange(len(catalogue))
    if recovered_only:
        indices = indices[catalogue['usable'] & ~catalogue['strict_usable']]
    brightest = indices[np.argsort(np.asarray(catalogue['segment_flux_adu'][indices]))[-12:][::-1]]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, i in zip(axes.flat, brightest):
        row = catalogue[i]
        radius = int(max(24, row['aperture_a_pix'] + 10))
        x, y = int(round(row['x'])), int(round(row['y']))
        x0, x1 = max(0, x - radius), min(residual.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(residual.shape[0], y + radius + 1)
        cutout = residual[y0:y1, x0:x1]
        scale = max(float(np.median(noise[y0:y1, x0:x1])), 1.0)
        display = np.arcsinh(cutout / scale)
        ax.imshow(display, origin='lower', cmap='gray', vmin=-1,
                  vmax=max(2, float(np.percentile(display, 99.5))),
                  extent=(x0 - .5, x1 - .5, y0 - .5, y1 - .5))
        colour = 'royalblue' if row['usable'] else 'red'
        ax.add_patch(Ellipse((row['x'], row['y']), 2 * row['aperture_a_pix'],
                             2 * row['aperture_b_pix'], angle=row['aperture_theta_deg'],
                             fill=False, edgecolor=colour, linewidth=1.2))
        status = 'usable' if row['usable'] else str(row['model_status'])
        if status == 'ok':
            status = 'contamination limit'
        ax.set_title(f"ID {row['id']}: {status}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    filename = 'recovered_bright_sources.png' if recovered_only else 'bright_source_review.png'
    fig.savefig(output / filename, dpi=160)
    plt.close(fig)


def run(settings, scored_catalogue=None, output=None):
    output = BASE / 'results' if output is None else Path(output)
    products_dir = ROOT / 'output/source_detection_results'
    original = Table.read(products_dir / 'sources.ecsv')
    image, header = fits.getdata(ROOT / 'Astro/Fits_Data/mosaic.fits', header=True)
    image = image.astype(np.float32)
    with fits.open(products_dir / 'detection_products.fits') as hdul:
        background = hdul['BACKGROUND'].data.copy()
        noise = hdul['RMS'].data.copy()
        bad = hdul['BADMASK'].data.astype(bool)
        artefact = hdul['ARTEFACT'].data.astype(bool)
        boundary = hdul['BOUNDARY'].data.astype(bool)
        saturated = hdul['SATURATED'].data.astype(bool)
        saturation = float(hdul[0].header['SATLEVEL'])
    residual = image - background
    diagnostics = []
    if scored_catalogue is None:
        original['local_background'] = background[np.rint(original['y']).astype(int), np.rint(original['x']).astype(int)]
        table, diagnostics = score_catalogue(original, residual, noise, bad, saturation, settings,
                                             gain=float(header['GAIN']))
    else:
        table = Table.read(scored_catalogue)
        if table.meta.get('model_version') != 3:
            raise ValueError('This catalogue needs to be rescored with the current model.')
        assert np.array_equal(table['segment_id'], original['segment_id'])
        assert np.allclose(table['x'], original['x']) and np.allclose(table['y'], original['y'])
        # Model quality settings are fixed when scores are computed; only limits change.
        current_hash = hashlib.sha256((products_dir / 'sources.ecsv').read_bytes()).hexdigest()
        if current_hash != table.meta['input_catalogue_sha256']:
            raise ValueError('Input catalogue changed; recompute model scores.')
        saved = json.loads(table.meta['model_settings'])
        for key, value in asdict(settings).items():
            if not key.startswith('max_overlap') and saved[key] != (list(value) if isinstance(value, tuple) else value):
                raise ValueError(f'Rescore requires unchanged fitting setting: {key}')
    table['usable'] = select_sources(table, settings)
    recovered = table['usable'] & ~table['strict_usable']
    # Preserve old flags in a separate column. Successful re-evaluation can remove
    # only the old neighbour-label contamination veto, not any hard-quality flag.
    flags = np.asarray(table['original_flags'], dtype='U200').copy()
    for i in np.flatnonzero(recovered):
        parts = [p for p in flags[i].split('|') if p not in ('none', 'aperture_contaminated')]
        flags[i] = '|'.join(parts + ['model_recovered'])
    table['flags'] = flags
    faint = np.isfinite(table['aperture_snr']) & (table['aperture_snr'] < settings.minimum_aperture_snr)
    for i in np.flatnonzero(faint):
        parts = [p for p in str(table['flags'][i]).split('|') if p != 'none']
        table['flags'][i] = '|'.join(parts + ['low_aperture_snr'])
    table['aperture_flux_adu'] = np.asarray(original['aperture_flux_adu'], float).copy()
    table['aperture_flux_adu'][recovered] = table['model_aperture_flux_adu'][recovered]
    table['aperture_contaminated'] = np.asarray(original['aperture_contaminated'], bool).copy()
    table['aperture_contaminated'][recovered] = False
    table.meta['input_catalogue_sha256'] = hashlib.sha256((products_dir / 'sources.ecsv').read_bytes()).hexdigest()
    table.meta['model_settings'] = json.dumps(asdict(settings))
    table.meta['model_version'] = 3
    table.meta['snr_note'] = 'Aperture sum / sqrt(sum(background RMS^2 + positive signal / GAIN)); no correlated-noise correction.'
    table.meta['score_note'] = 'Free elliptical Moffat proxy; two initialisations check solution stability, not model-family uncertainty or a confidence bound.'
    table.meta['area_note'] = 'Mean N/(S+N) over aperture pixels; not binary label-overlap area.'
    table.meta['flux_note'] = 'Raw signed aperture sum; neighbour model not subtracted. Recovered sources need validation.'
    # Read and validate inputs before replacing any previous output. This also
    # allows --scored-catalogue to point to results/sources.ecsv itself.
    output.mkdir(parents=True, exist_ok=True)

    # Only catalogue quality decisions change. Preserve all original image
    # products (including SOURCES, DETMASK and SATLEVEL) for downstream photometry.
    shutil.copyfile(
        products_dir / 'detection_products.fits',
        output / 'detection_products.fits',
    )

    # These plots are optional; remove stale ones if this run cannot regenerate
    # them. Only remove recognised plot names produced by this script.
    for path in output.glob('fit_*.png'):
        if path.stem.removeprefix('fit_').isdigit():
            path.unlink()
    (output / 'bleeding_detail.png').unlink(missing_ok=True)

    save_diagnostics(diagnostics, output)
    table.write(output / 'sources.ecsv', overwrite=True)
    table[recovered].write(output / 'recovered_sources.ecsv', overwrite=True)
    values = [0, 1e-6, 1e-5, .0001, .001, .003, .005, .01, .02, .03, .05, .1, .2]
    sweep = []
    for limit in values:
        trial = Settings(**{**asdict(settings), 'max_overlap_area_fraction': limit, 'max_overlap_flux_fraction': limit})
        sweep.append((limit, int(select_sources(table, trial).sum())))
    Table(rows=sweep, names=['both_limits', 'usable']).write(
        output / 'threshold_sweep.csv', format='ascii.csv', overwrite=True
    )
    baseline = table['strict_usable'] & np.isfinite(table['aperture_snr']) & (table['aperture_snr'] >= settings.minimum_aperture_snr)
    assert sweep[0][1] == int(baseline.sum())
    assert np.all(np.diff([n for _, n in sweep]) >= 0)
    assert np.array_equal(table['blended'], original['blended'])
    assert np.all(table['usable'][baseline])
    assert not np.any(recovered & ~table['reassessment_candidate'])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([v for v, n in sweep], [n for v, n in sweep], 'o-')
    ax.set_xscale('symlog', linthresh=1e-5)
    ax.set_xlabel('Both model-contamination limits')
    ax.set_ylabel('Usable sources')
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'threshold_sweep.png', dpi=160); plt.close(fig)
    detector.save_diagnostics(image, background, noise, artefact, boundary, saturated, table, output)
    save_bright_review(residual, noise, table, output)
    save_bright_review(residual, noise, table, output, recovered_only=True)
    summary = {'settings': asdict(settings), 'strict_usable': int(table['strict_usable'].sum()),
               'baseline_after_snr_cut': int(baseline.sum()),
               'low_aperture_snr': int(faint.sum()),
               'usable': int(table['usable'].sum()), 'recovered': int(recovered.sum()),
               'fit_status': dict(Counter(str(s) for s in table['model_status'])),
               'threshold_sweep': sweep,
               'source_catalogue_sha256': hashlib.sha256((products_dir / 'sources.ecsv').read_bytes()).hexdigest()}
    (output / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(output)
    return output


if __name__ == '__main__':
    raise SystemExit('Run run_aperture_review.py to generate the aperture-review results.')
