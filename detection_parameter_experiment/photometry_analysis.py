"""Run local empirical PSF classification and save morphology/count figures.

Run with the astro-image Python environment. Existing detection files are read
only. Each execution creates a new results directory and never overwrites a run.
The electron/s magnitude convention is inherited from the user's current script;
it must be checked against the mosaic zero-point definition independently.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.stats import chi2

from star_galaxy_conservative import MorphologySettings, classify_sources


BASE_DIR = Path(__file__).resolve().parent.parent
FITS_PATH = BASE_DIR / "Astro" / "Fits_Data" / "mosaic.fits"
DETECTION_DIR = Path(__file__).resolve().parent / "variants" / "baseline" / "detections"
OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
MAG_STEP = 0.5


def calibrate_magnitudes(catalogue, header):
    """Keep the existing electron/s convention so classification is comparable."""

    result = catalogue.copy()
    gain = float(header["GAIN"])
    exposure_time = float(header["EXPTIME"])
    zero_point = float(header["MAGZPT"])
    if not np.isfinite([gain, exposure_time, zero_point]).all():
        raise ValueError("Calibration header values must be finite.")
    if gain <= 0 or exposure_time <= 0:
        raise ValueError("GAIN and EXPTIME must be positive.")

    flux = np.asarray(result["aperture_flux_adu"], dtype=float)
    accepted = np.asarray(result["usable"], dtype=bool) & np.isfinite(flux) & (flux > 0)
    magnitude = np.full(len(result), np.nan)
    rate = flux[accepted] * gain / exposure_time
    magnitude[accepted] = zero_point - 2.5 * np.log10(rate)

    result["magnitude"] = magnitude
    result["count_sample"] = accepted
    result.meta["magnitude_convention"] = "MAGZPT - 2.5 log10(flux_ADU * GAIN / EXPTIME)"
    result.meta["classification_note"] = "Remaining includes uncertain sources; not confirmed galaxies."

    return result


def plot_morphology(catalogue, reference, settings, output):
    """Plot measured sources only; unmeasured remaining sources still enter counts."""

    selected = np.asarray(catalogue["count_sample"], dtype=bool)
    magnitude = np.asarray(catalogue["magnitude"])
    radius = np.asarray(catalogue["half_light_radius_pix"])
    star = np.asarray(catalogue["is_star"], dtype=bool)
    valid = selected & np.isfinite(magnitude) & np.isfinite(radius)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for mask, colour, label in [
        (valid & ~star, "royalblue", "Remaining sources"),
        (valid & star, "orange", "Likely stars"),
    ]:
        ax.scatter(
            magnitude[mask], radius[mask], s=16, alpha=0.65,
            color=colour, edgecolors="none",
            label=f"{label}: {np.count_nonzero(mask):,} plotted",
        )

    ax.axhline(
        reference["half_light_radius_pix"], color="orange", linestyle="--",
        label="Median stellar reference",
    )
    ax.set(
        xlabel="Calibrated aperture magnitude, r",
        ylabel=f"Half-light radius within {settings.outer_radius:g} pixels (pixel)",
        title="Distinguish stars and galaxies",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def cumulative_counts(magnitudes, step):
    """Count magnitudes strictly below each threshold, without rounding sources."""

    magnitudes = np.asarray(magnitudes, dtype=float)
    magnitudes = np.sort(magnitudes[np.isfinite(magnitudes)])
    if magnitudes.size == 0 or not np.isfinite(step) or step <= 0:
        raise ValueError("Need finite magnitudes and a positive threshold spacing.")

    first = int(np.floor(magnitudes[0] / step))
    last = int(np.floor(magnitudes[-1] / step)) + 1
    limits = np.arange(first, last + 1) * step
    counts = np.searchsorted(magnitudes, limits, side="left")

    return limits, counts


def poisson_interval(counts):
    """68.27% Garwood intervals; excludes clustering and measurement uncertainty."""

    counts = np.asarray(counts, dtype=int)
    alpha = 1.0 - 0.682689492
    lower = np.zeros(counts.shape, dtype=float)
    positive = counts > 0
    lower[positive] = 0.5 * chi2.ppf(alpha / 2, 2 * counts[positive])
    upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (counts + 1))

    return lower, upper


def plot_number_counts(catalogue, area_deg2, output, fit_range):
    """Descriptive fit only: cumulative points are correlated, not independent."""

    selected = np.asarray(catalogue["count_sample"], dtype=bool)
    star = np.asarray(catalogue["is_star"], dtype=bool)
    x, n = cumulative_counts(catalogue["magnitude"][selected & ~star], MAG_STEP)
    positive = n > 0
    x, n = x[positive], n[positive]
    lower, upper = poisson_interval(n)
    y = np.log10(n / area_deg2)
    yerr = np.vstack([np.log10(n / lower), np.log10(upper / n)])

    fit_min, fit_max = fit_range
    if not np.isfinite(fit_range).all() or fit_min >= fit_max:
        raise ValueError("Fit limits must be finite and increasing.")
    fit_mask = (x >= fit_min) & (x <= fit_max)
    if np.count_nonzero(fit_mask) < 3:
        raise ValueError("Choose a fit range containing at least three count points.")

    coefficients = np.polyfit(x[fit_mask], y[fit_mask], 1)
    fitted_line = np.poly1d(coefficients)
    x_fit = np.linspace(x[fit_mask].min(), x[fit_mask].max(), 200)
    anchor = 0.5 * (x_fit[0] + x_fit[-1])
    y_reference = fitted_line(anchor) + 0.6 * (x - anchor)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.errorbar(
        x, y, yerr=yerr, fmt="o-", color="royalblue", markersize=4,
        capsize=2, label="Observed counts (Poisson intervals)",
    )
    ax.axvspan(
        fit_min, fit_max, color="grey", alpha=0.10,
        label=f"Fit range: {fit_min:g}–{fit_max:g}",
    )
    ax.plot(
        x_fit, fitted_line(x_fit), color="red", linewidth=2,
        label=f"Linear fit: slope = {coefficients[0]:.3f}",
    )
    ax.plot(x, y_reference, "--", color="grey", label="Reference slope = 0.6")
    ax.set(
        xlabel="Limiting aperture magnitude, r",
        ylabel=r"$\log_{10}[(N(<m)/A)/(1\,\mathrm{deg}^{-2})]$",
        title="Observed cumulative counts",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(output, dpi=220)
    plt.close(fig)

    return [float(value) for value in coefficients]


def run_analysis(fit_range=(19.0, 24.0), effective_gain=None):
    """Load existing detections and save all new products inside a fresh folder."""

    catalogue = Table.read(DETECTION_DIR / "sources.ecsv")
    settings = MorphologySettings()
    with fits.open(FITS_PATH, memmap=False) as hdul:
        header = hdul[0].header.copy()
        image = hdul[0].data.astype(np.float32)

    with fits.open(DETECTION_DIR / "detection_products.fits", memmap=False) as products:
        background = products["BACKGROUND"].data
        noise = products["RMS"].data
        excluded = products["DETMASK"].data.astype(bool)
        segments = products["SOURCES"].data
        saturation = float(products[0].header["SATLEVEL"])

        if not all(a.shape == image.shape for a in (background, noise, excluded, segments)):
            raise ValueError("Detection products and input image dimensions differ.")

        excluded |= (
            ~np.isfinite(image) | ~np.isfinite(background)
            | ~np.isfinite(noise) | (noise <= 0)
        )
        classified, reference = classify_sources(
            catalogue, image - background, noise, excluded, segments, saturation, settings,
            effective_gain=effective_gain,
        )
        pixel_area = abs(float(header["PIXSCAL1"]) * float(header["PIXSCAL2"])) / 3600 ** 2
        area_deg2 = float(np.count_nonzero(~excluded) * pixel_area)

    if not np.isfinite(area_deg2) or area_deg2 <= 0:
        raise ValueError("Unmasked geometric area must be positive.")

    calibrated = calibrate_magnitudes(classified, header)
    selected = np.asarray(calibrated["count_sample"], dtype=bool)
    stars = np.asarray(calibrated["is_star"], dtype=bool)
    if not np.any(selected & ~stars):
        raise ValueError("No remaining sources for number counts.")

    OUTPUT_ROOT.mkdir(exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="psf_run_", dir=OUTPUT_ROOT))
    fit = plot_number_counts(
        calibrated, area_deg2, output_dir / "cumulative_number_counts.png", fit_range
    )
    plot_morphology(
        calibrated, reference, settings, output_dir / "morphology_vs_magnitude.png"
    )
    calibrated.write(output_dir / "classified_sources.ecsv", overwrite=False)

    reason_values, reason_counts = np.unique(
        calibrated["classification_reason"][selected], return_counts=True
    )
    reference_mask = np.asarray(calibrated["stellar_reference"], dtype=bool)
    reference_chi2 = np.asarray(calibrated["psf_reduced_chi2"][reference_mask])
    reference_chi2 = reference_chi2[np.isfinite(reference_chi2)]

    summary = {
        "created": datetime.now().isoformat(),
        "settings": asdict(settings),
        "reference": reference,
        "classification_reasons": dict(zip(reason_values.tolist(), reason_counts.tolist())),
        "reference_psf_reduced_chi2_median": (
            float(np.median(reference_chi2)) if reference_chi2.size else None
        ),
        "area_deg2": area_deg2,
        "photometric_sources": int(selected.sum()),
        "likely_stars": int(np.count_nonzero(selected & stars)),
        "remaining_sources": int(np.count_nonzero(selected & ~stars)),
        "remaining_without_radius": int(np.count_nonzero(
            selected & ~stars & ~np.isfinite(calibrated["half_light_radius_pix"])
        )),
        "fit_range": list(fit_range),
        "descriptive_slope_intercept": fit,
        "magnitude_convention": calibrated.meta["magnitude_convention"],
        "limitations": [
            "Local empirical PSF fitting: approximate diagnostics, not calibrated probabilities.",
            "Compact galaxies may be removed; uncertain and faint stars may remain.",
            "Reference locus is inferred, not independently verified as stars.",
            "PSF fit includes empirical PSF scatter; pixel covariance is not modelled.",
            "Source Poisson term only included when an effective gain is supplied.",
            "Electron/s zero-point convention inherited, not independently verified.",
            "No completeness, total-flux or selection-area correction.",
            "Cumulative fit is descriptive; counting intervals exclude clustering.",
        ],
    }
    with (output_dir / "run_summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Outputs: {output_dir}")
    return calibrated, reference, area_deg2, fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-range", type=float, nargs=2, default=(19.0, 24.0))
    parser.add_argument("--effective-gain", type=float, default=None,
                        help="Optional mosaic effective e-/ADU; not inferred from detector header.")
    parser.add_argument("--detections", type=Path, default=DETECTION_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    DETECTION_DIR = args.detections
    OUTPUT_ROOT = args.output_root
    run_analysis(tuple(args.fit_range), args.effective_gain)
