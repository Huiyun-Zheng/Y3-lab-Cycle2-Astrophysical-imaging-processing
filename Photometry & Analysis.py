"""Photometry, morphology plot and sumulative counts"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.table import Table
from scipy.stats import chi2

from star_galaxy_preliminary import MorphologySettings, classify_sources


BASE_DIR = Path(__file__).resolve().parent
FITS_PATH = BASE_DIR / "Astro" / "Fits_Data" / "mosaic.fits"
DETECTION_DIR = BASE_DIR/ "neighbour_light_model" / "results"
# DETECTION_DIR = BASE_DIR / "output" / "source_detection_results"
OUTPUT_DIR = BASE_DIR / "output" / "Photometry Analysis"

MAG_STEP = 0.5


def calibrate_magnitudes(catalogue, header):
    """Calculate source magnitudes using an explicitly chosen zero point."""
    result = catalogue.copy()

    gain = header["GAIN"]
    zero_point = float(header["MAGZPT"])
    exposure_time = header["EXPTIME"]

    flux_adu = np.asarray(result["aperture_flux_adu"], dtype=float)
    usable = np.asarray(result["usable"]).astype(str) == "True"
    accepted = usable & np.isfinite(flux_adu) & (flux_adu > 0)

    count_rate_e = (flux_adu[accepted] * gain) / exposure_time

    magnitude = np.full(len(result), np.nan)
    magnitude[accepted] = zero_point - 2.5 * np.log10(count_rate_e)

    result["magnitude"] = magnitude
    result["count_sample"] = accepted

    return result


def plot_morphology(catalogue, reference, settings, output):
    """Plot half-light radii against aperture magnitude."""

    selected = np.asarray(catalogue["count_sample"],dtype=bool)
    radius = np.asarray(catalogue["half_light_radius_pix"], dtype=float)
    magnitude = np.asarray(catalogue["magnitude"], dtype=float)

    if len(magnitude) != len(catalogue):
        raise ValueError("Magnitudes must match catalogue rows.")

    is_star = np.asarray(catalogue["is_star"].astype(str)) == "True"

    valid = selected & np.isfinite(magnitude) & np.isfinite(radius) & (radius >= 0)

    stars = valid & is_star
    galaxies = valid & ~is_star

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        magnitude[stars],
        radius[stars],
        color="orange",
        label=f"Likely stars: {np.count_nonzero(stars)} plotted."
    )

    ax.scatter(
        magnitude[galaxies],
        radius[galaxies],
        color="blue",
        label=f"Likely galaxies: {np.count_nonzero(galaxies)} plotted."
    )

    ax.axhline(
        reference["half_light_radius_pix"],
        color="orange",
        linestyle="--",
        label="Dashed Line: Emprical stellar size",
    )

    ax.legend()
    ax.grid(alpha=0.5)
    ax.set_xlabel("Calibrated magnitude, r(AB)")
    ax.set_ylabel(f"Half-light radius within {settings.outer_radius} pixels (pixel)")
    ax.set_title("Distinguish stars and galaxies")

    fig.savefig(output, dpi=300)
    plt.close(fig)


def cumulative_counts(magnitudes, step):
    "Count sources strictly brighter than uniformly spaced magnitude limits"
    magnitudes = np.asarray(magnitudes, dtype=float)
    magnitudes = np.sort(magnitudes[np.isfinite(magnitudes)])

    first = int(np.floor(magnitudes[0] / step))
    last = int(np.floor(magnitudes[-1] / step)) + 1
    limits = np.arange(first, last + 1) * step

    # No rounding of individual source magnitudes.
    counts = []
    for limit in limits:
        limit_mask = magnitudes < limit
        count = np.count_nonzero(limit_mask)
        counts.append(count)

    counts = np.asarray(counts)

    return limits, counts


def poisson_interval(counts):
    """Central 68.27% Garwood intervals for Poisson counts."""
    counts = np.asarray(counts, dtype=int)
    alpha = 1.0 - 0.682689492

    lower = np.zeros(counts.shape, dtype=float)
    positive = counts > 0

    lower[positive] = 0.5 * chi2.ppf(alpha / 2, 2 * counts[positive])
    upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (counts + 1))

    return lower, upper


def plot_number_counts(catalogue, area_deg2, output, fit_range):
    accepted = np.asarray(catalogue["count_sample"], dtype=bool)
    is_star = np.asarray(catalogue["is_star"], dtype=bool)
    galaxies = accepted & ~is_star

    magnitudes = np.asarray(catalogue["magnitude"][galaxies], dtype=float)
    limits, counts = cumulative_counts(magnitudes, MAG_STEP)

    # A zero count cannot be displayed on a logarithmic ordinate.
    positive = counts > 0
    x = limits[positive]
    n = counts[positive]

    lower, upper = poisson_interval(n)
    y = np.log10(n / area_deg2)

    # Calculate y-error (i.e. error in number counts)
    yerr_lower = y - np.log10(lower / area_deg2)
    yerr_upper = np.log10(upper / area_deg2) - y

    fit_min = float(fit_range[0])
    fit_max = float(fit_range[1])

    fit_mask = (x >= fit_min) & (x <= fit_max)
    if np.count_nonzero(fit_mask) < 3:
        raise ValueError("The fitting range must contain at least three plotted points.")

    fit_n, cov_n = np.polyfit(x[fit_mask], y[fit_mask], 1, cov='unscaled')
    pn = np.poly1d(fit_n)
    slope, intercept = fit_n

    x_fit = np.linspace(x[fit_mask].min(), x[fit_mask].max(), 300)
    y_fit = pn(x_fit)

    anchor_x = 0.5 * (x_fit[0] + x_fit[-1])
    anchor_y = pn(anchor_x)

    x_reference = np.linspace(x.min(), x.max(), 300)
    y_reference = anchor_y + 0.6 * (x_reference - anchor_x)

    fig,ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        x,
        y,
        yerr=[yerr_lower, yerr_upper],
        fmt='o-',
        color="royalblue",
        capsize=1.5,
    )

    ax.axvspan(
        fit_min,
        fit_max,
        color="grey",
        alpha=0.10,
        label=f"Fit range: {fit_min:g}--{fit_max:g}",
    )

    ax.plot(
        x_fit,
        y_fit,
        color="red",
        linewidth=3,
        label=f"Linear fit: slope = {slope:.3f}",
    )

    ax.plot(
        x_reference,
        y_reference,
        color="grey",
        linestyle="--",
        label="Reference slope = 0.6 (arbitrary normalisation)",
    )

    ax.set_xlabel("Magnitude m")
    ax.set_ylabel(r"$\log_{10}[(N(<m)/A)/(\mathrm{deg}^{-2})]$" )
    ax.set_title("Observed cumulative counts")
    ax.legend()
    ax.grid(alpha=0.5)

    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)

    return slope, intercept


def run_analysis(fit_range):
    """Read detection results, calibrate magnitudes, and generate two plots."""

    catalogue = Table.read(DETECTION_DIR / "sources.ecsv")
    settings = MorphologySettings()

    with fits.open(FITS_PATH) as hdul:
        header = hdul[0].header.copy()
        image = hdul[0].data.astype(np.float32)

    with fits.open(DETECTION_DIR / "detection_products.fits") as products:
        background = products["BACKGROUND"].data
        noise = products["RMS"].data
        excluded = products["DETMASK"].data.astype(bool)
        segments = products["SOURCES"].data
        saturation = products[0].header["SATLEVEL"]

        excluded |= ~np.isfinite(image) | ~np.isfinite(background) | ~np.isfinite(noise) | (noise <= 0)

        classified, reference = classify_sources(
            catalogue,
            image - background,
            noise,
            excluded,
            segments,
            saturation,
            settings,
        )

        pixel_area_deg2 = abs(float(header["PIXSCAL1"]) * float(header["PIXSCAL2"])) / 3600**2
        area_deg2 = np.count_nonzero(~excluded) * pixel_area_deg2

    calibrated = calibrate_magnitudes(classified, header)
    accepted = np.asarray(calibrated["count_sample"], dtype=bool)
    is_star = np.asarray(calibrated["is_star"], dtype=bool)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    morphology_path = OUTPUT_DIR / "morphology_vs_magnitude.png"
    counts_path = OUTPUT_DIR / "cumulative_number_counts.png"

    slope, intercept = plot_number_counts(calibrated, area_deg2, counts_path, fit_range=fit_range)
    plot_morphology(calibrated, reference, settings, morphology_path)

    print(f"GAIN used: {float(header['GAIN']):g} e⁻/ADU")
    print(f"Geometric unmasked area: {area_deg2:.6f} deg²")
    print(f"Likely stars: {np.count_nonzero(accepted & is_star):,}")
    print(f"Remaining sources:{np.count_nonzero(accepted & ~is_star):,}")

    return calibrated, reference, area_deg2, slope, intercept


if __name__ == "__main__":
    run_analysis(fit_range=(19.0, 24.0))
