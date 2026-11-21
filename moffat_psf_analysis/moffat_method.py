"""Conservative star candidates from pixel-integrated local Moffat profiles.

No detections or aperture fluxes are changed. Uncertain sources remain unknown.
Automatic reference stars and numerical cuts require observational validation.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from astropy.stats import sigma_clipped_stats


@dataclass(frozen=True)
class MoffatSettings:
    # Fit source profiles only for magnitudes up to this value.
    measurement_mag_limit: float = 23.0
    # Try to classify sources only up to this magnitude.
    classification_mag_limit: float = 22.0
    # Use only brighter sources as possible reference stars
    reference_mag_limit: float = 21.0

    # Fit pixels within this distance from the source centre
    fit_radius: int = 12

    # Start the background ring at this radius.
    sky_inner: int = 22
    # End the background ring at this radius.
    sky_outer: int = 30

    # Require at least this fraction of the fitting pixels to be usable.
    minimum_fit_fraction: float = 0.90

    # Require other detected sources to be at least this far away.
    isolation_radius: float = 18.0

    # Require this minimum signal-to-noise ratio for the fitted peak.
    minimum_snr: float = 50.0
    # Require a higher signal-to-noise ratio for reference stars
    reference_snr: float = 100.0

    # Limit the width error divided by the measured width.
    max_relative_width_error: float = 0.05
    # Reject fits with a larger reduced chi-square
    max_reduced_chi2: float = 5.0
    # Reject large combine residuals within radial rings.
    max_annular_residual: float = 4.0

    # Require at least this many nearby reference sources.
    minimum_references: int = 5
    # Use no more than this many nearby reference sources.
    nearest_references: int = 12
    # Do not use reference sources farther away than this distance
    maximum_reference_distance: float = 1000.0

    # Limit the spread of local reference widths relative to their median.
    max_local_width_scatter: float = 0.06
    # Set the allowed width difference, including the error margin
    width_equivalence: float = 0.15
    # Multiply the combined error by this value when checking agreement.
    confidence_multiplier: float = 2.0
    # Limit the difference in elliptical shape, including its error margin
    max_ellipticity_difference: float = 0.12

    # Limit the number of function evaluations used by the optimiser
    max_nfev: int = 180

    def __post_init__(self):
        if not 0 < self.fit_radius < self.sky_inner < self.sky_outer:
            raise ValueError('Require fit_radius < sky_inner < sky_outer.')
        if not 3 <= self.minimum_references <= self.nearest_references:
            raise ValueError('Need at least three local references.')
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        if self.classification_mag_limit > self.measurement_mag_limit:
            raise ValueError('Classification limit cannot exceed measurement limit.')
        if self.reference_mag_limit > self.classification_mag_limit:
            raise ValueError('References must be within the classification limit.')


# Three-point Gaussian-Legendre quadrature in each pixel: nine evaluations per pixel.
# Generate 1D quadrature nodes and weights
_nodes, _weights = np.polynomial.legendre.leggauss(3)

# Generate nine coordinate offsets within a pixel
offsets = []
for x in _nodes:
    for y in _nodes:
        offsets.append((x/ 2, y / 2))
_offsets = np.array (offsets)

# Generate nine corresponding 2D quadrature weights
_pixel_weights = np.outer(_weights, _weights).ravel() / 4


# Predict the brightness of each for a given Moffat model
def moffat_pixels(x, y, parameters):
    """Estimate pixel brightness using nine points inside each unit pixel."""

    # x0 & y0 are the fitted coordinates of the source centre, in pixels.
    # All sample points (even in different pixels) use this same sample centre when calculating their offsets.

    # e1 & e2 together control how stretched the profile is and its direction.
    # e1 controls stretching along the x/y axes.
    # e2 controls stretching along the diagonal directions (i.e. Δx=Δy).
    # When both are zero, the profile is circular.
    log_amp, x0, y0, log_alpha, beta, e1, e2, sky = parameters

    # Calculate each sample point's horizontal/vertical distance from the source centre.
    dx = np.asarray(x).ravel()[:, None] + _offsets[:, 0] - x0
    dy = np.asarray(y).ravel()[:, None] + _offsets[:, 1] - y0

    # Combine the offsets using different weights to describe an elliptical profile.
    # This is a shape-adjusted coordinate, not the oridinary squared distance.
    # The cross term allows the ellipse to tilt relative to the x & y axes.
    radius2 = ((1 - e1) * dx**2 + (1 + e1) * dy**2 - 2 * e2 * dx * dy)
    profile = (1 + radius2 * np.exp(-2 * log_alpha)) ** (-beta)

    return sky + np.exp(log_amp) * (profile @ _pixel_weights)


def moffat_fwhm(parameters):
    """Geometric mean of major/minor intrinsic model FWHM, in pixels."""

    alpha = np.exp(parameters[3])
    beta = parameters[4]
    e1 = parameters[5]
    e2 = parameters[6]

    # Calculate the squared distortion.
    e_squared = e1**2 + e2**2
    # Calculate the FWHM of the corresponding circular profile
    circular_fwhm = 2 * alpha * np.sqrt(2**(1 / beta) - 1)

    # Return the geometric mean of the major-axis and minor-axis FWHM
    return circular_fwhm / (1 - e_squared)**0.25


def robust_scatter(values):
    centre = np.median(values)
    return 1.4826 * np.median(np.abs(values - centre))


def fit_covariance(fit, n_pixels):
    """Formal covariance inflated, never reduced, by residual variance.

    This does not calibrate mosaic correlations or PSF model inadequacy.
    """
    chi2 = np.sum(fit.fun**2) / max(n_pixels - len(fit.x), 1)
    covariance = np.linalg.inv(fit.jac.T @ fit.jac) * max(1.0, chi2)
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0):
        raise ValueError('Invalid covariance.')
    return covariance, chi2


def width_error(parameters, covariance):
    gradient = np.zeros(len(parameters))
    for j in (3, 4, 5, 6):
        delta = np.zeros(len(parameters))
        delta[j] = 1e-5
        gradient[j] = (moffat_fwhm(parameters + delta) - moffat_fwhm(parameters - delta)) / 2e-5
    return np.sqrt(max(float(gradient @ covariance @ gradient), 0))


def make_stamp(image, labels, mask, row, gain, saturation, settings):
    """Use an isolated central fit and a source-masked sky annulus."""
    xc, yc = float(row['x']), float(row['y'])
    half = settings.sky_outer
    x0, y0 = int(round(xc)), int(round(yc))
    if x0 < half or y0 < half or x0 + half >= image.shape[1] or y0 + half >= image.shape[0]:
        return None, 'edge'

    bounds = np.s_[y0 - half:y0 + half + 1, x0 - half:x0 + half + 1]
    data = image[bounds]
    seg = labels[bounds]
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    centre = (xc - x0, yc - y0)
    radius = np.hypot(xx - centre[0], yy - centre[1])
    good = ~mask[bounds] & np.isfinite(data)
    neighbours = (seg > 0) & (seg != row['segment_id'])
    core = radius <= 3
    aperture = radius <= settings.fit_radius

    if np.any((~good | neighbours) & core):
        return None, 'masked_or_crowded_core'
    if np.any(data[aperture] >= saturation):
        return None, 'saturated'

    sky_pixels = good & (seg == 0) & (radius >= settings.sky_inner) & (radius <= half)
    if sky_pixels.sum() < 100:
        return None, 'insufficient_sky'
    _, sky, noise = sigma_clipped_stats(data[sky_pixels], sigma=3)
    selected = good & ~neighbours & aperture
    if selected.sum() < settings.minimum_fit_fraction * aperture.sum() or not np.isfinite(noise) or noise <= 0:
        return None, 'insufficient_pixels'

    values = data[selected].astype(float)
    error = np.sqrt(noise**2 + np.maximum(values - sky, 0) / gain)
    return {
        'x': xx[selected], 'y': yy[selected], 'values': values, 'error': error,
        'radius': radius[selected], 'sky': float(sky), 'noise': float(noise),
        'centre': centre,
    }, 'ok'


def fit_free_profile(stamp, settings):
    xc, yc = stamp['centre']
    sky, noise = stamp['sky'], stamp['noise']
    peak = max(np.max(stamp['values']) - sky, 1.0)
    initial = [np.log(peak), xc, yc, np.log(4), 2.5, 0, 0, sky]
    lower = [-10, xc - 1.5, yc - 1.5, np.log(.7), 1.2, -.45, -.45, sky - 3 * noise]
    upper = [25, xc + 1.5, yc + 1.5, np.log(20), 10, .45, .45, sky + 3 * noise]
    fit = least_squares(
        lambda p: (moffat_pixels(stamp['x'], stamp['y'], p) - stamp['values']) / stamp['error'],
        initial, bounds=(lower, upper), x_scale='jac', max_nfev=settings.max_nfev,
    )
    if not fit.success or np.any(fit.active_mask):
        raise ValueError('fit_limit')
    covariance, chi2 = fit_covariance(fit, len(stamp['values']))
    return fit, covariance, chi2


def measure_profiles(image, labels, mask, catalogue, header, settings):
    """Fit usable sources, preserving all rows and all detection measurements."""
    table = catalogue.copy()
    n = len(table)
    names = ['psf_fwhm', 'fwhm_error', 'fit_snr', 'reduced_chi2', 'moffat_beta',
             'moffat_log_alpha', 'moffat_e1', 'moffat_e2', 'e1_error', 'e2_error',
             'nearest_source_distance']
    for name in names:
        table[name] = np.full(n, np.nan)
    table['fit_status'] = np.full(n, 'not_fitted', dtype='U40')
    positions = np.column_stack([table['x'], table['y']])
    finite_positions = np.all(np.isfinite(positions), axis=1)
    if finite_positions.sum() >= 2:
        tree = cKDTree(positions[finite_positions])
        table['nearest_source_distance'][finite_positions] = tree.query(positions[finite_positions], k=2)[0][:, 1]
    gain = float(header['GAIN'])
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('GAIN must be finite and positive.')
    selected = table['count_sample'] & (table['magnitude'] <= settings.measurement_mag_limit)
    for number, i in enumerate(np.flatnonzero(selected)):
        if not finite_positions[i]:
            table['fit_status'][i] = 'invalid_position'
            continue
        stamp, status = make_stamp(image, labels, mask, table[i], gain, header['SATLEVEL'], settings)
        table['fit_status'][i] = status
        if stamp is None:
            continue
        try:
            fit, covariance, chi2 = fit_free_profile(stamp, settings)
            table['psf_fwhm'][i] = moffat_fwhm(fit.x)
            table['fwhm_error'][i] = width_error(fit.x, covariance)
            table['fit_snr'][i] = 1 / np.sqrt(covariance[0, 0])
            table['reduced_chi2'][i] = chi2
            for name, j in [('moffat_log_alpha', 3), ('moffat_beta', 4), ('moffat_e1', 5), ('moffat_e2', 6)]:
                table[name][i] = fit.x[j]
            table['e1_error'][i] = np.sqrt(covariance[5, 5])
            table['e2_error'][i] = np.sqrt(covariance[6, 6])
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            table['fit_status'][i] = 'fit_failed_or_limit'
        if (number + 1) % 100 == 0:
            print(f'Moffat measurements: {number + 1}/{selected.sum()}', flush=True)
    return table


def select_references(table, settings, reference_ids=None):
    """Provisional compact width locus, or explicitly checked source IDs."""
    width = np.asarray(table['psf_fwhm'])
    precise = (np.asarray(table['fwhm_error']) / width < settings.max_relative_width_error)
    e = np.hypot(table['moffat_e1'], table['moffat_e2'])
    pool = (table['count_sample'] & (table['fit_status'] == 'ok')
            & (table['magnitude'] <= settings.reference_mag_limit)
            & (table['fit_snr'] >= settings.reference_snr)
            & (table['reduced_chi2'] <= settings.max_reduced_chi2)
            & precise & (width >= 2) & (width <= 9) & (e < .25)
            & (table['nearest_source_distance'] >= settings.isolation_radius)
            & ~np.asarray(table['blended'], bool))
    if reference_ids is not None:
        return np.asarray(pool & np.isin(table['id'], reference_ids), bool)
    if np.count_nonzero(pool) < settings.minimum_references + 1:
        return np.zeros(len(table), bool)

    # Sliding window avoids hard histogram-edge dependence. No source IDs or
    # desired star counts enter this selection. Inspect these automatic seeds.
    widths = np.sort(width[pool])
    populations = np.array([np.count_nonzero(np.abs(widths - w) <= .3) for w in widths])
    mode = widths[np.argmax(populations)]
    references = np.asarray(pool & (np.abs(width - mode) <= .4), bool)
    if references.sum() >= settings.minimum_references + 1:
        centre = np.median(width[references])
        spread = robust_scatter(width[references])
        references &= np.abs(width - centre) <= max(.15, 3 * spread)
    return references


def local_reference(table, index, references, settings):
    """Leave the target out; never extrapolate beyond the distance limit."""
    indices = np.flatnonzero(references)
    indices = indices[indices != index]
    distance = np.hypot(table['x'][indices] - table['x'][index], table['y'][indices] - table['y'][index])
    order = np.argsort(distance)
    indices = indices[order][distance[order] <= settings.maximum_reference_distance][:settings.nearest_references]
    if len(indices) < settings.minimum_references:
        return None

    values = np.asarray(table['psf_fwhm'][indices], float)
    width = np.median(values)
    # Scatter is retained as predictive uncertainty, not divided by sqrt(N).
    scatter = np.hypot(robust_scatter(values), np.median(table['fwhm_error'][indices]))
    if scatter / width > settings.max_local_width_scatter:
        return None
    shape = np.array([np.median(table[key][indices]) for key in
                      ['moffat_log_alpha', 'moffat_beta', 'moffat_e1', 'moffat_e2']])
    template = np.r_[0.0, 0.0, 0.0, shape, 0.0]
    # Use a well-defined median width even when alpha and beta are correlated.
    template[3] += np.log(width / moffat_fwhm(template))
    return indices, template, width, scatter


def equivalence_test(ratio, error, reference_fractional_scatter, settings):
    """Uncertainty consumes the fixed tolerance; it never widens acceptance."""
    allowance = settings.confidence_multiplier * np.hypot(error, reference_fractional_scatter)
    return abs(ratio - 1) + allowance < settings.width_equivalence


def fit_local_profile(stamp, template, settings):
    """Fit amplitude, centre, scale and sky, holding local PSF shape fixed.

    Scale is an observed-profile width ratio, NOT intrinsic galaxy radius.
    """
    xc, yc = stamp['centre']
    sky, noise = stamp['sky'], stamp['noise']
    initial = [np.log(max(np.max(stamp['values']) - sky, 1)), xc, yc, 0, sky]
    lower = [-10, xc - 1.5, yc - 1.5, np.log(.5), sky - 3 * noise]
    upper = [25, xc + 1.5, yc + 1.5, np.log(2.5), sky + 3 * noise]

    def parameters(p):
        result = template.copy()
        result[:3] = p[:3]
        result[3] += p[3]
        result[-1] = p[-1]
        return result

    fit = least_squares(
        lambda p: (moffat_pixels(stamp['x'], stamp['y'], parameters(p)) - stamp['values']) / stamp['error'],
        initial, bounds=(lower, upper), x_scale='jac', max_nfev=settings.max_nfev,
    )
    if not fit.success or np.any(fit.active_mask):
        raise ValueError('local_fit_limit')
    covariance, chi2 = fit_covariance(fit, len(stamp['values']))
    ratio = np.exp(fit.x[3])
    ratio_error = ratio * np.sqrt(covariance[3, 3])
    # Coherent radial residuals test the core and wings separately. This is a
    # diagnostic under independent-pixel errors, not a calibrated probability.
    annular = []
    for lo, hi in zip([0, 2, 4, 6, 9], [2, 4, 6, 9, settings.fit_radius]):
        selected = (stamp['radius'] >= lo) & (stamp['radius'] < hi)
        if selected.sum() >= 6:
            annular.append(abs(np.sum(fit.fun[selected])) / np.sqrt(selected.sum()))
    return ratio, ratio_error, chi2, max(annular, default=np.inf), parameters(fit.x)


def classify_sources(image, labels, mask, table, header, settings, reference_ids=None):
    result = table.copy()
    references = select_references(result, settings, reference_ids)
    result['psf_reference'] = references
    result['source_class'] = np.full(len(result), 'unknown', dtype='U24')
    result['classification_reason'] = np.full(len(result), 'not_eligible', dtype='U40')
    for name in ['local_psf_fwhm', 'local_psf_scatter', 'local_scale', 'local_scale_error',
                 'local_reduced_chi2', 'local_annular_residual', 'reference_max_distance']:
        result[name] = np.full(len(result), np.nan)
    result['reference_count'] = np.zeros(len(result), int)
    result['reference_ids'] = np.full(len(result), '', dtype='U240')
    diagnostics = []

    eligible = (result['count_sample'] & (result['fit_status'] == 'ok')
                & (result['magnitude'] <= settings.classification_mag_limit))
    for i in np.flatnonzero(eligible):
        row = result[i]
        if row['blended'] or row['nearest_source_distance'] < settings.isolation_radius:
            result['classification_reason'][i] = 'crowded_keep'
            continue
        if (row['fit_snr'] < settings.minimum_snr
                or row['fwhm_error'] / row['psf_fwhm'] > settings.max_relative_width_error):
            result['classification_reason'][i] = 'imprecise_keep'
            continue
        if row['reduced_chi2'] > settings.max_reduced_chi2:
            result['classification_reason'][i] = 'poor_free_profile'
            continue
        local = local_reference(result, i, references, settings)
        if local is None:
            result['classification_reason'][i] = 'no_reliable_local_psf'
            continue
        indices, template, width, scatter = local
        result['local_psf_fwhm'][i] = width
        result['local_psf_scatter'][i] = scatter
        result['reference_count'][i] = len(indices)
        result['reference_ids'][i] = ','.join(str(v) for v in result['id'][indices])
        result['reference_max_distance'][i] = np.max(np.hypot(
            result['x'][indices] - row['x'], result['y'][indices] - row['y']))
        stamp, _ = make_stamp(image, labels, mask, row, header['GAIN'], header['SATLEVEL'], settings)
        if stamp is None:
            result['classification_reason'][i] = 'invalid_stamp'
            continue
        try:
            ratio, error, chi2, annular, profile = fit_local_profile(stamp, template, settings)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            result['classification_reason'][i] = 'local_fit_failed'
            continue
        for name, value in [('local_scale', ratio), ('local_scale_error', error),
                            ('local_reduced_chi2', chi2), ('local_annular_residual', annular)]:
            result[name][i] = value
        if chi2 > settings.max_reduced_chi2 or annular > settings.max_annular_residual:
            result['classification_reason'][i] = 'local_profile_mismatch'
            continue
        shape_difference = np.hypot(row['moffat_e1'] - template[5], row['moffat_e2'] - template[6])
        shape_error = settings.confidence_multiplier * np.hypot(row['e1_error'], row['e2_error'])
        free_agrees = equivalence_test(row['psf_fwhm'] / width, row['fwhm_error'] / width, scatter / width, settings)
        local_agrees = equivalence_test(ratio, error, scatter / width, settings)
        if free_agrees and local_agrees and shape_difference + shape_error < settings.max_ellipticity_difference:
            result['source_class'][i] = 'star_candidate'
            result['classification_reason'][i] = 'precise_local_psf_match'
            if len(diagnostics) < 18:
                diagnostics.append((int(row['id']), stamp, profile))
        elif ratio - settings.confidence_multiplier * np.hypot(error, scatter / width) > 1 + settings.width_equivalence:
            result['source_class'][i] = 'extended_candidate'
            result['classification_reason'][i] = 'resolved_profile'
        else:
            result['classification_reason'][i] = 'inconclusive_keep'
    result['is_star'] = np.asarray(result['source_class']) == 'star_candidate'
    return result, diagnostics
