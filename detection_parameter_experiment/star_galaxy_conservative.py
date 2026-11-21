"""Local empirical PSF versus PSF-convolved extension classification.

Only well-measured, demonstrably unresolved sources are removed. This is not
SExtractor SPREAD_MODEL or a full oversampled ePSF reconstruction. PSF sampling,
reference contamination and correlated mosaic noise limit formal significance.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift, fourier_gaussian
from scipy.optimize import minimize


@dataclass
class MorphologySettings:
    """Reference-selection, PSF-fitting and unresolved-source thresholds."""

    inner_radius: float = 4.0
    middle_radius: float = 8.0
    outer_radius: float = 12.0
    measurement_sky_snr: float = 10.0
    star_sky_snr: float = 30.0
    star_detection_snr: float = 30.0
    reference_detection_snr: float = 80.0
    reference_sky_snr: float = 50.0
    maximum_elongation: float = 1.20
    saturation_fraction: float = 0.65
    local_neighbours: int = 25
    minimum_local_references: int = 8
    maximum_reference_distance: float = 1200.0
    stamp_half_size: int = 15
    fit_radius: float = 10.0
    minimum_fit_snr: float = 30.0
    maximum_reduced_chi2: float = 2.0
    maximum_extension_improvement: float = 4.0
    extension_delta_chi2: float = 9.0
    maximum_intrinsic_sigma: float = 0.8
    centroid_shift_limit: float = 1.0
    psf_fractional_floor: float = 0.01
    extension_grid: tuple = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def measure_morphology(catalogue, residual, noise, excluded, segments, settings):
    """Measure signed aperture fluxes, r50, and two curve-of-growth ratios.

    Use only fully unmasked apertures containing no other labelled source.
    The RMS-based S/N is background-only, not a total photon-noise S/N.
    Radius uses the first half-flux crossing; it is an aperture r50, not total r50.
    """

    result = catalogue.copy()
    measurements = np.full((len(result), 4), np.nan)
    status = np.full(len(result), "not_photometrically_usable", dtype="U32")
    height, width = residual.shape

    for index, source in enumerate(result):
        if not source["usable"]:
            continue

        x = float(source["x"])
        y = float(source["y"])
        radius = settings.outer_radius
        x0 = int(np.floor(x - radius))
        x1 = int(np.ceil(x + radius)) + 1
        y0 = int(np.floor(y - radius))
        y1 = int(np.ceil(y + radius)) + 1

        status[index] = "edge_or_neighbour_or_mask"
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            continue

        cutout = np.s_[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.hypot(xx - x, yy - y)
        aperture = distance <= radius
        labels = segments[cutout]
        neighbour = (labels != 0) & (labels != source["segment_id"])

        if np.any((excluded[cutout] | neighbour)[aperture]):
            continue

        signal = residual[cutout][aperture].astype(float)
        rms = noise[cutout][aperture].astype(float)
        radii = distance[aperture]

        status[index] = "invalid_flux_or_noise"
        if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(rms)):
            continue
        if np.any(rms <= 0):
            continue

        flux = signal.sum()
        sky_error = np.sqrt(np.sum(rms ** 2))
        if flux <= 0 or sky_error <= 0:
            continue

        sky_snr = flux / sky_error
        measurements[index, 3] = sky_snr
        status[index] = "low_measurement_snr"
        if sky_snr < settings.measurement_sky_snr:
            continue

        order = np.argsort(radii)
        growth = np.cumsum(signal[order])
        crossing = np.flatnonzero(growth >= 0.5 * flux)[0]
        r50 = radii[order][crossing]
        concentration_4 = signal[radii <= settings.inner_radius].sum() / flux
        concentration_8 = signal[radii <= settings.middle_radius].sum() / flux

        measurements[index, :3] = r50, concentration_4, concentration_8
        status[index] = "measured"

    result["half_light_radius_pix"] = measurements[:, 0]
    result["concentration_inner"] = measurements[:, 1]
    result["concentration_middle"] = measurements[:, 2]
    result["morphology_sky_snr"] = measurements[:, 3]
    result["morphology_status"] = status

    return result


def robust_centre_scale(features):
    """Median and MAD scatter; floors prevent zero-width cuts from pixel sampling."""

    centre = np.median(features, axis=0)
    scatter = 1.4826 * np.median(np.abs(features - centre), axis=0)
    floors = np.array([0.08, 0.015, 0.01])

    return centre, np.maximum(scatter, floors)


def select_reference_sources(catalogue, residual, noise, excluded, segments, saturation,
                             settings=None):
    """Find bright isolated compact seeds; final labels are decided by PSF fits."""

    settings = MorphologySettings() if settings is None else settings
    if not (0 < settings.inner_radius < settings.middle_radius < settings.outer_radius):
        raise ValueError("Aperture radii must be positive and increasing.")
    if not np.isfinite(saturation) or saturation <= 0:
        raise ValueError("Saturation must be positive and finite.")
    if not all(a.shape == residual.shape for a in (noise, excluded, segments)):
        raise ValueError("Morphology images must have identical shapes.")

    result = measure_morphology(
        catalogue, residual, noise, excluded, segments, settings
    )
    features = np.column_stack([
        result["half_light_radius_pix"],
        result["concentration_inner"],
        result["concentration_middle"],
    ])
    size, inner, middle = features.T
    sky_snr = np.asarray(result["morphology_sky_snr"])
    peak_snr = np.asarray(result["peak_detection_snr"])
    elongation = np.asarray(result["elongation"])

    reliable = (
        np.asarray(result["usable"], dtype=bool)
        & np.all(np.isfinite(features), axis=1)
        & (inner > 0)
        & (inner < middle)
        & (middle < 1)
        & (sky_snr >= settings.star_sky_snr)
        & (peak_snr >= settings.star_detection_snr)
        & (elongation <= settings.maximum_elongation)
        & (np.asarray(result["peak_adu"]) < settings.saturation_fraction * saturation)
    )
    pool = (
        reliable
        & (peak_snr >= settings.reference_detection_snr)
        & (sky_snr >= settings.reference_sky_snr)
        & (size >= 1.0)
        & (size <= 0.6 * settings.outer_radius)
    )
    if np.count_nonzero(pool) < 15:
        raise ValueError("Too few bright isolated sources to establish a stellar locus.")

    # Find the dominant compact size group, then require a consistent profile.
    edges = np.arange(1.0, 0.6 * settings.outer_radius + 0.25, 0.25)
    histogram, edges = np.histogram(size[pool], bins=edges)
    peak = int(np.argmax(histogram))
    modal_radius = 0.5 * (edges[peak] + edges[peak + 1])
    reference_mask = pool & (np.abs(size - modal_radius) <= 0.5)

    for _ in range(3):
        if np.count_nonzero(reference_mask) < 12:
            raise ValueError("No stable stellar reference; inspect bright source cutouts.")

        centre, scatter = robust_centre_scale(features[reference_mask])
        consistent = np.all(np.abs(features - centre) <= 3 * scatter, axis=1)
        reference_mask &= consistent

    if np.count_nonzero(reference_mask) < 12:
        raise ValueError("Too few references after profile clipping.")

    return result, reference_mask


def extract_stamp(source, residual, noise, excluded, segments, settings):
    """Keep science data on its original pixel grid; only PSF templates shift."""

    half = settings.stamp_half_size
    x, y = float(source["x"]), float(source["y"])
    ix, iy = int(np.rint(x)), int(np.rint(y))
    height, width = residual.shape
    if ix - half < 0 or iy - half < 0 or ix + half >= width or iy + half >= height:
        return None

    cutout = np.s_[iy - half:iy + half + 1, ix - half:ix + half + 1]
    data = residual[cutout].astype(float)
    rms = noise[cutout].astype(float)
    labels = segments[cutout]
    bad = (
        excluded[cutout] | ~np.isfinite(data) | ~np.isfinite(rms) | (rms <= 0)
        | ((labels != 0) & (labels != source["segment_id"]))
    )
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    radius = np.hypot(xx - (x - ix), yy - (y - iy))
    fit_mask = radius <= settings.fit_radius
    if np.any(bad[fit_mask]):
        return None

    return data, rms, bad, radius, fit_mask, (y - iy, x - ix)


def build_reference_stamps(result, reference_mask, residual, noise, excluded,
                           segments, settings):
    """Align and normalise isolated reference cutouts, keeping signed wings."""

    indices, stamps = [], []
    for index in np.flatnonzero(reference_mask):
        stamp = extract_stamp(result[index], residual, noise, excluded, segments, settings)
        if stamp is None:
            continue

        data, _, bad, radius, _, offset = stamp
        # Require clean data out to 14 pixels to protect the template wings.
        if np.any(bad[radius <= settings.outer_radius + 2]):
            continue

        annulus = (radius >= settings.outer_radius) & ~bad
        if np.count_nonzero(annulus) < 60:
            continue

        local_sky = np.median(data[annulus])
        clean = np.where(bad, 0.0, data - local_sky)
        aligned = shift(clean, (-offset[0], -offset[1]), order=3, mode="constant")
        half = settings.stamp_half_size
        yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
        aperture = np.hypot(xx, yy) <= settings.outer_radius
        flux = aligned[aperture].sum()
        if flux <= 0:
            continue

        indices.append(index)
        stamps.append(aligned / flux)

    if len(indices) < 12:
        raise ValueError("Fewer than 12 clean PSF stamps; inspect reference selection.")

    return np.asarray(indices), np.asarray(stamps)


def local_psf(position, source_index, positions, reference_indices, stamps, settings):
    """Median of nearest reference stamps; never train on the target itself."""

    distances = np.linalg.norm(positions[reference_indices] - position, axis=1)
    order = np.argsort(distances)
    selected = order[
        (distances[order] <= settings.maximum_reference_distance)
        & (reference_indices[order] != source_index)
    ][:settings.local_neighbours]
    if len(selected) < settings.minimum_local_references:
        return None

    stack = stamps[selected]
    model = np.median(stack, axis=0)
    # Use full local scatter, not scatter / sqrt(N): it also represents PSF variation.
    scatter = 1.4826 * np.median(np.abs(stack - model), axis=0)
    scatter = np.maximum(scatter, settings.psf_fractional_floor * np.abs(model))
    return model, scatter, len(selected)


def fit_template(data, variance, mask, template, offset, settings):
    """Profile over centroid, flux and a constant residual sky for one template."""

    values = data[mask]
    weight = 1.0 / variance[mask]
    sum_weight = weight.sum()

    def solve(delta, return_values=False):
        shifted = shift(
            template, (offset[0] + delta[0], offset[1] + delta[1]),
            order=3, mode="constant",
        )
        profile = shifted[mask]
        mean_profile = np.sum(weight * profile) / sum_weight
        mean_data = np.sum(weight * values) / sum_weight
        centred = profile - mean_profile
        denominator = np.sum(weight * centred ** 2)
        if denominator <= 0:
            return np.inf if not return_values else None

        flux = max(0.0, np.sum(weight * centred * (values - mean_data)) / denominator)
        sky = mean_data - flux * mean_profile
        chi2 = np.sum(weight * (values - flux * profile - sky) ** 2)
        if return_values:
            return chi2, flux, sky, np.sqrt(1.0 / denominator), shifted
        return chi2

    bound = settings.centroid_shift_limit
    optimum = minimize(
        solve, np.zeros(2), method="L-BFGS-B", bounds=[(-bound, bound)] * 2,
        options={"maxiter": 40, "ftol": 1e-8},
    )
    fitted = solve(optimum.x, return_values=True)
    if fitted is None:
        return None
    good = optimum.success and np.max(np.abs(optimum.x)) < 0.95 * bound
    return (*fitted, good)


def compare_models(stamp, psf, psf_scatter, settings, effective_gain=None):
    """Compare PSF to Gaussian-broadened PSF with identical frozen pixel weights.

    Broadening sigma is intrinsic Gaussian sigma, in pixels, not observed FWHM.
    A coarse profile grid supplies a conservative upper bracket at delta chi2=9.
    This is an approximate diagnostic, not a calibrated confidence probability.
    """

    data, rms, _, _, mask, offset = stamp
    initial = fit_template(data, rms ** 2, mask, psf, offset, settings)
    if initial is None or not initial[-1] or initial[1] <= 0:
        return None

    flux = initial[1]
    scatter = shift(psf_scatter, offset, order=1, mode="nearest")
    variance = rms ** 2 + (flux * scatter) ** 2
    if effective_gain is not None:
        variance += np.maximum(flux * initial[4], 0.0) / effective_gain

    # Frozen weights make chi2 values comparable across the nested models.
    grid = np.asarray(settings.extension_grid)
    fft_psf = np.fft.fftn(psf)
    fits = []
    for sigma in grid:
        template = psf if sigma == 0 else np.fft.ifftn(
            fourier_gaussian(fft_psf, sigma)
        ).real
        fits.append(fit_template(data, variance, mask, template, offset, settings))

    if any(f is None or not f[-1] for f in fits):
        return None

    chi2 = np.array([f[0] for f in fits])
    best = int(np.argmin(chi2))
    compatible = np.flatnonzero(chi2 <= chi2[best] + settings.extension_delta_chi2)
    last = compatible[-1]
    # The next grid point is conservative: do not understate width resolution.
    upper_sigma = float(grid[last + 1]) if last < len(grid) - 1 else np.inf
    point = fits[0]
    reduced = point[0] / (np.count_nonzero(mask) - 4)
    snr = point[1] / point[3]
    improvement = max(0.0, point[0] - chi2[best])

    return {
        "psf_reduced_chi2": float(reduced),
        "psf_fit_snr": float(snr),
        "extension_delta_chi2": float(improvement),
        "intrinsic_sigma_best_pix": float(grid[best]),
        "intrinsic_sigma_upper_pix": upper_sigma,
        "psf_flux_adu": float(point[1]),
        "psf_sky_offset_adu": float(point[2]),
    }


def classify_sources(catalogue, residual, noise, excluded, segments, saturation,
                     settings=None, effective_gain=None):
    """Remove only resolved-data-supported PSF matches; keep all uncertain sources.

    effective_gain is optional electrons per image ADU. For a mosaic its true
    value need not equal detector GAIN. If omitted, no source Poisson term is
    invented; the analysis caller explicitly records any supplied assumption.
    """

    settings = MorphologySettings() if settings is None else settings
    if effective_gain is not None and (not np.isfinite(effective_gain) or effective_gain <= 0):
        raise ValueError("Effective gain must be positive or None.")

    result, seed_mask = select_reference_sources(
        catalogue, residual, noise, excluded, segments, saturation, settings
    )
    reference_indices, stamps = build_reference_stamps(
        result, seed_mask, residual, noise, excluded, segments, settings
    )
    positions = np.column_stack([result["x"], result["y"]])
    diagnostics = [
        "psf_reduced_chi2", "psf_fit_snr", "extension_delta_chi2",
        "intrinsic_sigma_best_pix", "intrinsic_sigma_upper_pix",
        "psf_flux_adu", "psf_sky_offset_adu",
    ]
    for name in diagnostics:
        result[name] = np.full(len(result), np.nan)

    star = np.zeros(len(result), dtype=bool)
    local_counts = np.zeros(len(result), dtype=int)
    reason = np.full(len(result), "uncertain_or_unmeasured", dtype="U40")
    eligible = (
        np.asarray(result["usable"], dtype=bool)
        & (np.asarray(result["morphology_status"]) == "measured")
        & (np.asarray(result["morphology_sky_snr"]) >= settings.star_sky_snr)
        & (np.asarray(result["peak_detection_snr"]) >= settings.star_detection_snr)
        & (np.asarray(result["peak_adu"]) < settings.saturation_fraction * saturation)
    )
    candidates = np.flatnonzero(eligible)
    print(f"PSF library: {len(reference_indices)} references; fitting {len(candidates)} sources.", flush=True)

    for counter, index in enumerate(candidates):
        if counter % 100 == 0:
            print(f"PSF comparison: {counter}/{len(candidates)}", flush=True)

        local = local_psf(
            positions[index], index, positions, reference_indices, stamps, settings
        )
        reason[index] = "insufficient_local_psf"
        if local is None:
            continue
        model, scatter, local_counts[index] = local

        stamp = extract_stamp(result[index], residual, noise, excluded, segments, settings)
        reason[index] = "invalid_fit_stamp"
        if stamp is None:
            continue
        fitted = compare_models(stamp, model, scatter, settings, effective_gain)
        reason[index] = "fit_failed_or_centroid_at_bound"
        if fitted is None:
            continue

        for name, value in fitted.items():
            result[name][index] = value

        if fitted["psf_fit_snr"] < settings.minimum_fit_snr:
            reason[index] = "insufficient_resolution_snr"
        elif fitted["psf_reduced_chi2"] > settings.maximum_reduced_chi2:
            reason[index] = "poor_psf_fit"
        elif fitted["extension_delta_chi2"] > settings.maximum_extension_improvement:
            reason[index] = "extended_model_preferred"
        elif fitted["intrinsic_sigma_upper_pix"] > settings.maximum_intrinsic_sigma:
            reason[index] = "extension_not_ruled_out"
        elif result["elongation"][index] > settings.maximum_elongation:
            reason[index] = "elongated_or_blended"
        else:
            star[index] = True
            reason[index] = "well_constrained_psf_match"

    result["is_star"] = star
    result["source_class"] = np.where(star, "likely_star", "remaining")
    result["stellar_reference"] = np.isin(np.arange(len(result)), reference_indices)
    result["local_reference_count"] = local_counts
    result["classification_reason"] = reason
    reference = {
        "half_light_radius_pix": float(np.median(result["half_light_radius_pix"][reference_indices])),
        "reference_sources": len(reference_indices),
        "method": "local empirical PSF versus Gaussian-convolved PSF",
        "effective_gain_assumed": effective_gain,
    }
    return result, reference
