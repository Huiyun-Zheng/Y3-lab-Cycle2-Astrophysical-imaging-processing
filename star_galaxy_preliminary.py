"""Identify likely stars before calibration; treat every other source as a galaxy."""

from dataclasses import dataclass
import numpy as np


@dataclass
class MorphologySettings:
    """Pixel radii and deliberately conservative classification thresholds."""

    inner_radius: float = 4.0
    outer_radius: float = 12.0
    minimum_detection_snr: float = 15.0
    reference_detection_snr: float = 50.0
    minimum_aperture_snr: float = 10.0
    maximum_star_elongation: float = 1.3
    star_size_tolerance: float = 0.20


def measure_morphology(catalogue, residual, noise, excluded, segments, settings):
    """Measure half-light radius and F(4 pix) / F(12 pix) on clean cutouts.

    Half-light radii refer only to the light inside the outer aperture, not
    the total source light. Keep signed background-subtracted pixels: clipping
    negative pixels would artificially add sky light to faint objects.
    """

    result = catalogue.copy()
    half_light_radius = np.full(len(result), np.nan)
    concentration = np.full(len(result), np.nan)
    sky_snr = np.full(len(result), np.nan)
    height, width = residual.shape

    for index, source in enumerate(result):
        if not source["usable"]:
            continue

        x, y = float(source["x"]), float(source["y"])
        radius = settings.outer_radius
        x0, x1 = int(np.floor(x - radius)), int(np.ceil(x + radius)) + 1
        y0, y1 = int(np.floor(y - radius)), int(np.ceil(y + radius)) + 1

        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            continue

        cutout = np.s_[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.hypot(xx - x, yy - y)
        aperture = distance <= radius
        labels = segments[cutout]
        neighbours = (labels != 0) & (labels != source["segment_id"])

        if np.any((excluded[cutout] | neighbours)[aperture]):
            continue

        signal = residual[cutout][aperture].astype(float)
        local_noise = noise[cutout][aperture].astype(float)

        if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(local_noise)):
            continue

        total_flux = signal.sum()
        sky_error = np.sqrt(np.sum(local_noise ** 2))

        if total_flux <= 0 or sky_error <= 0:
            continue

        sky_snr[index] = total_flux / sky_error

        if sky_snr[index] < settings.minimum_aperture_snr:
            continue

        radii = distance[aperture]
        order = np.argsort(radii)
        sorted_radii = radii[order]
        growth_curve = np.cumsum(signal[order])

        # First crossing works even when signed sky pixels make the curve dip.
        crossing = np.flatnonzero(growth_curve >= 0.5 * total_flux)[0]
        half_light_radius[index] = sorted_radii[crossing]
        concentration[index] = signal[radii <= settings.inner_radius].sum() / total_flux

    result["half_light_radius_pix"] = half_light_radius
    result["concentration"] = concentration
    result["morphology_sky_snr"] = sky_snr

    return result


def classify_sources(catalogue, residual, noise, excluded, segments, saturation, settings=None):
    """Return a new table and an empirical reference for compact, round sources.

    Bright, round, unsaturated sources define the most populated size group.
    Its median size/concentration is a provisional stellar reference, not a
    confirmed PSF. Only sources matching the stellar criteria are removed.
    Every other source, including those without reliable morphology, is labelled
    galaxy by assumption. Photometry quality is checked separately downstream.
    """

    settings = MorphologySettings() if settings is None else settings
    result = measure_morphology(catalogue, residual, noise, excluded, segments, settings)
    size = np.asarray(result["half_light_radius_pix"])
    concentration = np.asarray(result["concentration"])
    detection_snr = np.asarray(result["peak_detection_snr"])
    elongation = np.asarray(result["elongation"])

    reliable = (
        np.isfinite(size)
        & (concentration > 0)
        & (concentration < 1)
        & (detection_snr >= settings.minimum_detection_snr)
        & (np.asarray(result["peak_adu"]) < 0.7 * saturation)
    )
    reference_pool = (
        reliable
        & (detection_snr >= settings.reference_detection_snr)
        & (elongation <= 1.2)
        & (size >= 1.0)
        & (size <= 0.65 * settings.outer_radius)
    )

    if np.count_nonzero(reference_pool) < 10:
        raise ValueError("Too few clean, bright, round sources for a stellar reference.")

    # A modal size group avoids letting a few extended galaxies shift the mean.
    bins = np.arange(1.0, 0.65 * settings.outer_radius + 0.25, 0.25)
    frequencies, edges = np.histogram(size[reference_pool], bins=bins)
    peak_bin = int(np.argmax(frequencies))
    modal_size = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])
    reference_stars = reference_pool & (np.abs(size - modal_size) <= 0.5)

    if np.count_nonzero(reference_stars) < 5:
        raise ValueError("No stable compact-source size group was found.")

    stellar_size = float(np.median(size[reference_stars]))
    stellar_concentration = float(np.median(concentration[reference_stars]))
    size_ratio = size / stellar_size
    concentration_ratio = concentration / stellar_concentration

    star = (
        reliable
        & (np.abs(size_ratio - 1) <= settings.star_size_tolerance)
        & (np.abs(concentration_ratio - 1) <= 0.20)
        & (elongation <= settings.maximum_star_elongation)
    )
    # No separate galaxy threshold: retain everything not identified as a star.
    labels = np.full(len(result), "galaxy", dtype="U6")
    labels[star] = "star"
    result["is_star"] = star
    result["source_class"] = labels
    result["stellar_reference"] = reference_stars

    reference = {
        "half_light_radius_pix": stellar_size,
        "concentration": stellar_concentration,
        "reference_sources": int(np.count_nonzero(reference_stars)),
    }

    return result, reference
