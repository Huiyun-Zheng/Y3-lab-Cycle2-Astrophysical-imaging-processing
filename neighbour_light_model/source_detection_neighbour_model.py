"""Reassess blended sources using jointly fitted observed-light profiles.

Read existing strict detections; preserve input detections and apertures.
Each run replaces this experiment's outputs in the fixed results directory.
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


@dataclass
class Settings:
    # New definitions: mean pixel contamination and integrated contamination.
    # Both use model light, including wings outside segmentation labels.
    max_overlap_area_fraction: float = 0.10
    max_overlap_flux_fraction: float = 0.10
    profile_betas: tuple = (2.5, 4.0)
    max_group_sources: int = 12
    max_fit_pixels: int = 16000
    max_nfev: int = 100
    max_reduced_residual: float = 5.0
    max_coherent_residual: float = 5.0
    max_score_disagreement: float = 0.03
    minimum_model_snr: float = 10.0

    def __post_init__(self):
        for value in (self.max_overlap_area_fraction, self.max_overlap_flux_fraction):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Contamination limits must lie in [0, 1].')


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
    return (dx * np.cos(theta) + dy * np.sin(theta)) ** 2, (-dx * np.sin(theta) + dy * np.cos(theta)) ** 2


def log_components(parameters, u2, v2, beta):
    p = np.asarray(parameters[:-1]).reshape(-1, 3)
    q = u2 * np.exp(-2 * p[:, 1]) + v2 * np.exp(-2 * p[:, 2])
    return p[:, 0] - beta * np.log1p(q)


def fit_profiles(rows, xx, yy, data, error, beta, settings):
    """Simultaneously fit positive Moffat profiles plus one residual sky level.

    Centroids and orientations remain at catalogue values. Amplitudes and
    both widths are fitted. The analytic Jacobian avoids repeated image fits.
    """
    u2, v2 = coordinates(rows, xx, yy)
    # Moment-to-Moffat conversion is only an initial guess, not a fixed width.
    aa = np.clip(np.asarray(rows['major_sigma_pix']) * np.sqrt(2 * (beta - 2)), 1, 25)
    bb = np.clip(np.asarray(rows['minor_sigma_pix']) * np.sqrt(2 * (beta - 2)), 1, 25)
    amplitudes = np.maximum(np.asarray(rows['peak_adu']) - np.median(rows['local_background']), 1)
    p0 = np.column_stack([np.log(amplitudes), np.log(aa), np.log(bb)]).ravel()
    sky_limit = 3 * float(np.median(error))
    p0 = np.r_[p0, 0.0]
    lower = np.r_[np.tile([np.log(.01), np.log(.6), np.log(.6)], len(rows)), -sky_limit]
    upper = np.r_[np.tile([np.log(1e6), np.log(45), np.log(45)], len(rows)), sky_limit]

    def residual(p):
        return (np.exp(log_components(p, u2, v2, beta)).sum(axis=1) + p[-1] - data) / error

    def jacobian(p):
        pars = p[:-1].reshape(-1, 3)
        qa, qb = u2 * np.exp(-2 * pars[:, 1]), v2 * np.exp(-2 * pars[:, 2])
        q = 1 + qa + qb
        model = np.exp(pars[:, 0] - beta * np.log(q))
        jac = np.empty((len(data), len(p)))
        jac[:, 0:-1:3] = model
        jac[:, 1:-1:3] = 2 * beta * model * qa / q
        jac[:, 2:-1:3] = 2 * beta * model * qb / q
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
        return np.nan, np.nan
    target_light = log_models[:, target]
    neighbour_light = logsumexp(others, axis=1)
    # Mean fractional contamination per aperture pixel: effective area fraction.
    area_fraction = float(np.mean(expit(neighbour_light - target_light)))
    # Fraction of summed model light contributed by all neighbours.
    flux_fraction = float(expit(logsumexp(neighbour_light) - logsumexp(target_light)))
    return area_fraction, flux_fraction


def select_sources(table, settings):
    """Keep the strict baseline, and recover only model-qualified blended rows."""
    recover = (np.asarray(table['blended']) & np.asarray(table['model_valid'])
               & (table['model_area_fraction'] < settings.max_overlap_area_fraction)
               & (table['model_flux_fraction'] < settings.max_overlap_flux_fraction))
    return np.asarray(table['strict_usable']) | recover


def score_catalogue(table, residual, noise, bad, saturation, settings, only_parents=None):
    result = table.copy()
    result['strict_usable'] = np.asarray(table['usable'], bool)
    result['original_flags'] = np.asarray(table['flags'], str)
    result['original_aperture_flux_adu'] = np.asarray(table['aperture_flux_adu'], float)
    for name in ['model_area_fraction', 'model_flux_fraction', 'model_residual_score',
                 'model_score_disagreement', 'model_aperture_snr', 'model_coherent_residual',
                 'model_aperture_flux_adu']:
        result[name] = np.full(len(table), np.nan)
    result['model_valid'] = np.zeros(len(table), bool)
    result['model_status'] = np.full(len(table), 'not_reassessed', dtype='U32')
    result['local_background'] = np.asarray(table['local_background'], float)
    xy = np.column_stack([table['x'], table['y']])
    parents = np.unique(table['parent_id'][table['blended']])
    if only_parents is not None:
        parents = np.intersect1d(parents, only_parents)
    diagnostics = []

    for count, parent in enumerate(parents):
        targets = np.flatnonzero((table['parent_id'] == parent) & table['blended'])
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
        data, error = residual[y0:y1, x0:x1], noise[y0:y1, x0:x1]
        clean = (~bad[y0:y1, x0:x1] & np.isfinite(data) & np.isfinite(error) & (error > 0))
        if clean.sum() < max(100, 10 * (3 * len(included) + 1)):
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
        success = all(f.success and not np.any(f.active_mask) for _, f in fits_for_group)
        if not success:
            result['model_status'][targets] = 'fit_failed_or_at_bound'
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
            other = rows[np.arange(len(rows)) != target_index]
            px = np.asarray(other['peak_x'], int) - bounds[1].start
            py = np.asarray(other['peak_y'], int) - bounds[0].start
            inside = (px >= 0) & (px < aperture.shape[1]) & (py >= 0) & (py < aperture.shape[0])
            if np.any(aperture[py[inside], px[inside]]):
                result['model_status'][i] = 'neighbour_peak_inside'
                continue
            u2, v2 = coordinates(rows, ax[aperture], ay[aperture])
            scores, residual_scores, snrs, coherent = [], [], [], []
            for beta, fit in fits_for_group:
                components = log_components(fit.x, u2, v2, beta)
                scores.append(contamination_scores(components, target_index))
                full_u2, full_v2 = coordinates(rows, ax, ay)
                full_model = np.exp(log_components(fit.x, full_u2, full_v2, beta)).sum(axis=1).reshape(ax.shape) + fit.x[-1]
                whitened = (residual[bounds] - full_model) / noise[bounds]
                # Detect spatially coherent mismatch hidden by an average residual.
                kernel = np.exp(-np.arange(-4, 5, dtype=float) ** 2 / 2)
                kernel /= kernel.sum()
                smooth_rms = float(np.sum(kernel ** 2))
                smoothed = ndi.gaussian_filter(whitened, 1.0, mode='nearest') / smooth_rms
                coherent.append(float(np.percentile(np.abs(smoothed[aperture]), 95)))
                model = np.exp(components).sum(axis=1) + fit.x[-1]
                # This is a background-RMS residual diagnostic, not formal chi^2:
                # source Poisson noise and mosaic correlations are not calibrated.
                residual_scores.append(float(np.mean(((model - residual[bounds][aperture])
                                                      / noise[bounds][aperture]) ** 2)))
                snrs.append(float(np.exp(components[:, target_index]).sum()
                                  / np.sqrt(np.sum(noise[bounds][aperture] ** 2))))
            scores = np.asarray(scores)
            upper_scores = np.max(scores, axis=0)
            disagreement = float(np.max(np.ptp(scores, axis=0)))
            result['model_area_fraction'][i], result['model_flux_fraction'][i] = upper_scores
            result['model_residual_score'][i] = max(residual_scores)
            result['model_score_disagreement'][i] = disagreement
            result['model_aperture_snr'][i] = min(snrs)
            result['model_coherent_residual'][i] = max(coherent)
            if not np.all(np.isfinite(scores)) or np.any(upper_scores <= 0):
                result['model_status'][i] = 'unresolved_model_score'
            elif max(residual_scores) > settings.max_reduced_residual:
                result['model_status'][i] = 'poor_profile_fit'
            elif max(coherent) > settings.max_coherent_residual:
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
            if int(row['id']) in (215, 364) or (result['model_valid'][i] and len(diagnostics) < 6):
                beta, fit = fits_for_group[0]
                all_u2, all_v2 = coordinates(rows, xx, yy)
                model_cutout = np.exp(log_components(fit.x, all_u2, all_v2, beta)).sum(axis=1).reshape(xx.shape) + fit.x[-1]
                diagnostics.append((int(row['id']), (x0, x1, y0, y1), data.copy(), model_cutout, error.copy()))
        if count % 50 == 0:
            print(f'Groups checked: {count + 1}/{len(parents)}', flush=True)
    return result, diagnostics


def save_diagnostics(diagnostics, output):
    for identifier, bounds, data, model, error in diagnostics[:10]:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        for ax, values, title in zip(axes, [data / error, model / error, (data - model) / error],
                                    ['Data / RMS', 'Joint profile model / RMS', 'Residual / RMS']):
            ax.imshow(values, origin='lower', cmap='gray', vmin=-3, vmax=12 if title != 'Residual / RMS' else 3)
            ax.set_title(title)
        fig.suptitle(f'Source {identifier}')
        fig.tight_layout()
        fig.savefig(output / f'fit_{identifier}.png', dpi=140)
        plt.close(fig)


def run(settings, scored_catalogue=None):
    output = BASE / 'results'
    products_dir = ROOT / 'output/source_detection_results'
    original = Table.read(products_dir / 'sources.ecsv')
    image = fits.getdata(ROOT / 'Astro/Fits_Data/mosaic.fits').astype(np.float32)
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
        table, diagnostics = score_catalogue(original, residual, noise, bad, saturation, settings)
    else:
        table = Table.read(scored_catalogue)
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
    table['aperture_flux_adu'] = np.asarray(original['aperture_flux_adu'], float).copy()
    table['aperture_flux_adu'][recovered] = table['model_aperture_flux_adu'][recovered]
    table['aperture_contaminated'] = np.asarray(original['aperture_contaminated'], bool).copy()
    table['aperture_contaminated'][recovered] = False
    table.meta['input_catalogue_sha256'] = hashlib.sha256((products_dir / 'sources.ecsv').read_bytes()).hexdigest()
    table.meta['model_settings'] = json.dumps(asdict(settings))
    table.meta['score_note'] = 'Observed Moffat profile proxy; not calibrated true contamination or a confidence bound.'
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
    assert sweep[0][1] == int(table['strict_usable'].sum())
    assert np.all(np.diff([n for _, n in sweep]) >= 0)
    assert np.array_equal(table['blended'], original['blended'])
    assert np.all(table['usable'][table['strict_usable']])
    assert not np.any(recovered & ~table['blended'])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([v for v, n in sweep], [n for v, n in sweep], 'o-')
    ax.set_xscale('symlog', linthresh=1e-5)
    ax.set_xlabel('Both model-contamination limits')
    ax.set_ylabel('Usable sources')
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'threshold_sweep.png', dpi=160); plt.close(fig)
    detector.save_diagnostics(image, background, noise, artefact, boundary, saturated, table, output)
    summary = {'settings': asdict(settings), 'strict_usable': int(table['strict_usable'].sum()),
               'usable': int(table['usable'].sum()), 'recovered': int(recovered.sum()),
               'fit_status': dict(Counter(str(s) for s in table['model_status'])),
               'threshold_sweep': sweep,
               'source_catalogue_sha256': hashlib.sha256((products_dir / 'sources.ecsv').read_bytes()).hexdigest()}
    (output / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--area-limit', type=float, default=Settings.max_overlap_area_fraction)
    parser.add_argument('--flux-limit', type=float, default=Settings.max_overlap_flux_fraction)
    parser.add_argument('--scored-catalogue', type=Path)
    args = parser.parse_args()
    run(Settings(max_overlap_area_fraction=args.area_limit, max_overlap_flux_fraction=args.flux_limit), args.scored_catalogue)
