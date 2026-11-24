"""Simple photometry errors and Monte Carlo counts using the existing circular Moffat method.

Only results/simple_analysis is written. Existing scripts and other results are unchanged.
Magnitude errors are stored, not plotted. No count covariance matrices or blank apertures.
Additional plots report the simple np.polyfit slope error and differential counts.
"""

import sys
sys.dont_write_bytecode = True

import argparse
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from circular_moffat import measure_psf_widths, select_psf_candidates


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
FITS_PATH = PROJECT_DIR / "Astro" / "Fits_Data" / "mosaic.fits"
DETECTION_DIR = PROJECT_DIR / "source_detection_neighbour_refined" / "results"
OUTPUT_DIR = CODE_DIR / "results" / "simple_analysis"

MEASUREMENT_MAG_LIMIT = 22.0
CLASSIFICATION_MAG_LIMIT = 21.0
MAG_STEP = 0.42
MC_DRAWS = 2000
RANDOM_SEED = 20261118


def calibrate_magnitudes(catalogue, header):
    """Calculate source magnitudes using an explicitly chosen zero point."""
    result = catalogue.copy()

    gain = float(header["GAIN"])
    exposure_time = float(header["EXPTIME"])
    zero_point = float(header["MAGZPT"])

    flux = np.asarray(result["aperture_flux_adu"], dtype=float)
    usable = np.asarray(result["usable"]).astype(str) == "True"
    accepted = usable & np.isfinite(flux) & (flux > 0)
    magnitude = np.full(len(result), np.nan)
    rate = flux[accepted] * gain / exposure_time
    magnitude[accepted] = zero_point - 2.5 * np.log10(rate)

    result["magnitude"] = magnitude
    result["count_sample"] = accepted
    result.meta["magnitude_convention"] = "MAGZPT - 2.5 log10(flux_ADU * GAIN / EXPTIME)"
    result.meta["classification_note"] = (
        "Remaining includes uncertain sources; not confirmed galaxies."
    )

    return result


def cumulative_counts(magnitudes, step):
    """Count sources strichtly brighter than uniformly spaced magnitude limits."""

    magnitudes = np.asarray(magnitudes, dtype=float)
    magnitudes = np.sort(magnitudes[np.isfinite(magnitudes)])

    first = int(np.floor(magnitudes[0] / step))
    last = int(np.floor(magnitudes[-1] / step)) + 1
    limits = np.arange(first, last + 1) * step

    counts = []
    for limit in limits:
        limit_mask = magnitudes < limit
        count = np.count_nonzero(limit_mask)
        counts.append(count)
    counts = np.asarray(counts)

    return limits, counts


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aperture_offsets(row):
    """Reproduce the detector's binary ellipse at its original subpixel centre."""
    x, y = float(row["x"]), float(row["y"])
    a, b = float(row["aperture_a_pix"]), float(row["aperture_b_pix"])

    # Determine the half-width of a square grid that is large enough to contain the entire ellipse at any rotation angle.
    # Add one extra pixel as a safety margin.
    half = int(np.ceil(max(a, b))) + 1
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]

    # Shift the x & y coordinates so that they are measured relative to the true subpixel centre rather than the rounded centre.
    dx = xx - (x - round(x))
    dy = yy - (y - round(y))
    angle = np.radians(float(row["aperture_theta_deg"]))
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)

    inside = (u / a) ** 2 + (v / b) ** 2 <= 1

    return xx[inside], yy[inside]


def add_photometric_uncertainties(table, rms, mask, header):
    """Calculate simple flux and magnitude errors."""
    # mask here identifies invalid or excluded pixels.
    result = table.copy()
    gain = float(header["GAIN"])
    zp_error = float(header["MAGZRR"])
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(zp_error) or zp_error < 0:
        raise ValueError("GAIN must be positive and MAGZRR must be nonnegative and finite.")

    columns = [
        "flux_error_adu",
        "background_flux_error_adu",
        "source_flux_error_adu",
        "magnitude_error_random",
        "magnitude_error_total",
        "photometric_snr",
    ]
    for name in columns:
        result[name] = np.full(len(result), np.nan)
    result["magnitude_error_zeropoint"] = np.full(len(result), zp_error)

    chosen = np.flatnonzero(result["count_sample"])
    for number, i in enumerate(chosen):
        row = result[i]
        dx, dy = aperture_offsets(row)
        x, y = int(round(float(row["x"]))), int(round(float(row["y"])))
        px, py = x + dx, y + dy

        # Check that the aperture contains pixels and lies completely within the image boundaries.
        if (
            len(dx) == 0
            or np.any(px < 0)
            or np.any(px >= rms.shape[1])
            or np.any(py < 0)
            or np.any(py >= rms.shape[0])
        ):
            raise ValueError(f'Invalid accepted aperture: ID {row["id"]}.')

        pixel_rms = rms[py, px].astype(float)
        if np.any(~np.isfinite(pixel_rms) | (pixel_rms <= 0)) or np.any(mask[py, px]):
            raise ValueError(f'Invalid accepted-aperture noise/mask: ID {row["id"]}.')

        # Calculate the background variance, assuming independent pixel noise.
        background_variance = np.sum(pixel_rms**2)
        flux = float(row["aperture_flux_adu"])
        # Estimate the shot-noise variance from the measured flux and detector gain, assuming Poisson photon statistics.
        source_variance = flux / gain
        # Combine background and source variances to calculate the total random flux uncertainty.
        flux_error = np.sqrt(background_variance + source_variance)
        fractional_error = flux_error / flux

        result["flux_error_adu"][i] = flux_error
        result["background_flux_error_adu"][i] = np.sqrt(background_variance)
        result["source_flux_error_adu"][i] = np.sqrt(source_variance)
        result["photometric_snr"][i] = 1 / fractional_error

        # Use first-order error propagation to convert flux error into magnitude error.
        # This approximation is less accurate for sources with low signal-to-noise.
        random_mag_error = 2.5 / np.log(10) * fractional_error
        result["magnitude_error_random"][i] = random_mag_error
        result["magnitude_error_total"][i] = np.hypot(random_mag_error, zp_error)

        if (number + 1) % 1000 == 0:
            print(f"Aperture uncertainties: {number + 1}/{len(chosen)}", flush=True)

    result.meta["error_note"] = (
        "Independent-pixel RMS variance + F_ADU/GAIN shot variance; no blank-aperture correction. "
        "First-order magnitude errors include shared MAGZRR and are stored, not plotted."
    )
    return result


def counts_from_magnitudes(magnitudes, limits):
    """Count the number of galaxies brighter than each magnitude limit."""
    counts = []

    for limit in limits:
        count = np.count_nonzero(magnitudes < limit)
        counts.append(count)

    return np.array(counts)


def count_error_model(flux, flux_error, header, limits, draws=MC_DRAWS):
    """Estimate count errors with flux perturbations and Poisson count draws."""
    if draws < 2:
        raise ValueError("At least two Monte Carlo draws are required.")
    if np.any(~np.isfinite(flux_error)) or np.any(flux_error < 0):
        raise ValueError("Flux errors must be finite and nonnegative.")
    if np.any(np.diff(limits) <= 0):
        raise ValueError("Magnitude limits must be strictly increasing.")

    rng = np.random.default_rng(RANDOM_SEED + 1)
    simulated = np.zeros((draws, len(limits)), dtype=float)  # An array to store cumulative counts from each simulation.
    zero_point = float(header["MAGZPT"])
    conversion = float(header["GAIN"]) / float(header["EXPTIME"])
    zp_error = float(header["MAGZRR"])

    for trial in range(draws):
        # Randomly vary each source's flux using its measured uncertainty.
        draw_flux = flux + rng.normal(0, flux_error)

        # Initially exclude nonpositive fluxes by assigning infinite magnitude.
        magnitude = np.full(len(flux), np.inf)
        positive = draw_flux > 0

        # Convert positive simulated fluxes into magnitudes.
        # Apply one shared zero-point offset to all sources in this draw.
        magnitude[positive] = (
            zero_point + rng.normal(0, zp_error)
            - 2.5 * np.log10(draw_flux[positive] * conversion)
        )
        draw_counts = counts_from_magnitudes(magnitude, limits)

        # Find the counts in non-overlapping magnitude intervals.
        # The first interval contains every source brighter than the first limit.
        interval_counts = np.diff(np.r_[0, draw_counts])

        # Add Poisson counting fluctuations once, then rebuild cumulative counts.
        # Drawing intervals instead of cumulative points keeps every curve increasing.
        simulated[trial] = np.cumsum(rng.poisson(interval_counts))

    # Use the spread across trials as the error at each magnitude limit.
    # No extra sqrt(N) term is needed: Poisson fluctuations are already included.
    return np.std(simulated, axis=0, ddof=1), np.mean(simulated, axis=0)


def descriptive_line_fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def plot_number_counts(table, area, output, fit_range, header):
    """Calculate and plot cumulative galaxy number counts with uncertainties."""

    selected = table["count_sample"] & ~table["is_star"]
    limits, counts = cumulative_counts(table["magnitude"][selected], MAG_STEP)
    positive = counts > 0
    limits, counts = limits[positive], counts[positive]
    flux = np.asarray(table["aperture_flux_adu"][selected], float)
    error = np.asarray(table["flux_error_adu"][selected], float)
    if not np.all(np.isfinite(error)):
        raise ValueError("All count-sample sources need a finite flux error.")
    count_error, mc_mean = count_error_model(flux, error, header, limits)
    y = np.log10(counts / area)

    # Propagate the Monte Carlo count standard deviation into log space.
    # These are approximate one-standard-deviation bars, not exact confidence intervals.
    log_error = count_error / (np.log(10) * counts)

    fit_mask = (limits >= fit_range[0]) & (limits <= fit_range[1])
    if fit_range[0] >= fit_range[1] or np.count_nonzero(fit_mask) < 3:
        raise ValueError("Fit range must contain at least three count points.")
    fit = descriptive_line_fit(limits[fit_mask], y[fit_mask])

    results = Table(
        {
            "limiting_magnitude": limits,
            "count": counts,
            "log10_density": y,
            "count_error_mc": count_error,
            "count_mean_mc": mc_mean,
            "log_error_mc": log_error,
        }
    )
    results.meta.update(fit)
    results.meta["area_deg2"] = float(area)
    results.meta["fit_range"] = list(fit_range)
    results.meta["mag_step"] = MAG_STEP
    results.meta["error_note"] = (
        "Monte Carlo Poisson + flux/zero-point errors; first-order log bars. No completeness, classification or cosmic-variance errors."
    )
    results.meta["mc_draws"] = MC_DRAWS
    results.meta["random_seed"] = RANDOM_SEED + 1
    results.write(output / "number_counts.ecsv", overwrite=True)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.errorbar(
        limits,
        y,
        yerr=log_error,
        fmt="o-",
        color="royalblue",
        markersize=4,
        capsize=2,
        label="Counts: Monte Carlo errors",
    )
    ax.axvspan(
        *fit_range, color="grey", alpha=0.10, label=f"Fit range: {fit_range[0]:g}–{fit_range[1]:g}"
    )
    xx = np.linspace(limits[fit_mask].min(), limits[fit_mask].max(), 200)
    line = np.poly1d([fit["slope"], fit["intercept"]])
    ax.plot(
        xx,
        line(xx),
        color="red",
        linewidth=3,
        label=f"Descriptive slope = {fit['slope']:.3f}",
    )
    anchor = np.mean(xx)
    ax.plot(
        limits,
        line(anchor) + 0.6 * (limits - anchor),
        "--",
        color="grey",
        label="Reference slope = 0.6",
    )
    ax.set(
        xlabel="Limiting aperture magnitude, r",
        ylabel=r"$\log_{10}[(N(<m))/(1\,\mathrm{deg}^{-2})]$",
        title="Observed cumulative counts",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output / "cumulative_number_counts.png", dpi=300)
    plt.close(fig)
    return fit


def plot_width_magnitude(table, output):
    valid = table["count_sample"] & np.isfinite(table["psf_fwhm"])
    stars = np.asarray(table["is_star"], bool)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for selection, colour, label in [
        (valid & ~stars, "royalblue", "Remaining sources"),
        (valid & stars, "orange", "Likely stars"),
    ]:
        # Magnitude errors are saved in the catalogue but are not plotted.
        ax.scatter(
            table["magnitude"][selection],
            table["psf_fwhm"][selection],
            s=14,
            alpha=0.7,
            color=colour,
            edgecolors="none",
            zorder=2,
            label=f"{label}: {selection.sum():,} plotted",
        )
    references = table["psf_reference"]
    if np.any(references):
        ax.axhline(
            np.median(table["psf_fwhm"][references]),
            color="orange",
            linestyle="--",
            label="Median reference width",
        )
    ax.set(
        xlabel="Calibrated aperture magnitude, r",
        ylabel="Moffat FWHM (pixel)",
        title="Circular Moffat star selection",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.savefig(output / "moffat_width_vs_magnitude.png", dpi=300)
    plt.close(fig)


def run_analysis(
    measurement_mag_limit=MEASUREMENT_MAG_LIMIT,
    classification_mag_limit=CLASSIFICATION_MAG_LIMIT,
    fit_range=(19.0, 23.0),
    reference_labels=None,
):
    """Keep every input row; remove only star candidates from the count sample."""

    if not np.all(np.isfinite([measurement_mag_limit, classification_mag_limit, *fit_range])):
        raise ValueError("Magnitude limits and fit range must be finite.")
    if classification_mag_limit > measurement_mag_limit or fit_range[0] >= fit_range[1]:
        raise ValueError(
            "Classification limit must not exceed measurement limit; fit range must increase."
        )

    sources_path = DETECTION_DIR / "sources.ecsv"
    products_path = DETECTION_DIR / "detection_products.fits"
    original = Table.read(sources_path)

    with fits.open(FITS_PATH) as hdul:
        image = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()

    with fits.open(products_path) as hdul:
        labels = hdul["SOURCES"].data.copy()
        mask = hdul["DETMASK"].data.astype(bool)
        background = hdul["BACKGROUND"].data.copy()
        rms = hdul["RMS"].data.copy()
        header["SATLEVEL"] = float(hdul[0].header["SATLEVEL"])

    if not all(a.shape == image.shape for a in (labels, mask, background, rms)):
        raise ValueError("Image and detection products have different shapes.")

    mask |= ~np.isfinite(image) | ~np.isfinite(background) | ~np.isfinite(rms) | (rms <= 0)
    area = (
        np.count_nonzero(~mask)
        * abs(float(header["PIXSCAL1"]) * float(header["PIXSCAL2"]))
        / 3600**2
    )
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Unmasked sky area must be finite and positive.")

    calibrated = calibrate_magnitudes(original, header)
    measured = measure_psf_widths(
        image, labels, calibrated, header, mask=mask, mag_limit=measurement_mag_limit
    )
    classified = select_psf_candidates(
        measured, reference_labels=reference_labels, mag_limit=classification_mag_limit
    )
    classified["is_star"] = np.asarray(classified["source_class"]) == "star_candidate"

    # Check that the analysis has not changed detection rows or aperture fluxes.
    for name in original.colnames:
        a = np.asarray(original[name])
        b = np.asarray(classified[name])
        same = np.allclose(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)
        if not same:
            raise ValueError(f"Original detection column changed: {name}")

    classified = add_photometric_uncertainties(classified, rms, mask, header)
    remaining = classified["count_sample"] & ~classified["is_star"]
    classified["remaining_sample"] = remaining

    classified.meta["input_sources_sha256"] = file_hash(sources_path)
    classified.meta["input_products_sha256"] = file_hash(products_path)
    classified.meta["measurement_mag_limit"] = float(measurement_mag_limit)
    classified.meta["area_deg2"] = float(area)
    classified.meta["sample_note"] = (
        "Unknown sources retained in counts; remaining sources are not confirmed galaxies."
    )
    classified.meta["width_note"] = (
        "Circular Moffat FWHM, before pixel integration; not a half-light radius."
    )
    classified.meta["calibration_assumption"] = (
        "MAGZPT refers to electrons/second; header GAIN used as effective gain."
    )
    classified.meta["mc_draws"] = MC_DRAWS
    classified.meta["random_seed"] = RANDOM_SEED

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_width_magnitude(classified, OUTPUT_DIR)
    fit = plot_number_counts(classified, area, OUTPUT_DIR, fit_range, header)
    classified.meta["descriptive_slope"] = fit["slope"]
    classified.meta["fit_range"] = list(fit_range)
    classified.write(OUTPUT_DIR / "classified_sources.ecsv", overwrite=True)

    print(f"Usable photometry: {int(classified['count_sample'].sum())}")
    print(f"Reference candidates: {int(classified['psf_reference'].sum())}")
    print(f"Likely stars removed: {int(classified['is_star'].sum())}")
    print(f"Remaining sources: {int(remaining.sum())}")
    print(f"Descriptive slope: {fit['slope']:.6f}")
    print(f"Area: {area:.6f} deg^2")
    print(f"Outputs: {OUTPUT_DIR}")
    return classified, fit


def line_fit_with_error(x, y):
    """Return the straight-line fit and the formal errors from np.polyfit."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 3 or np.ptp(x[good]) == 0:
        raise ValueError("At least three finite points with different x values are needed.")

    # Keep the original unweighted fit. The covariance describes the scatter
    # around this straight line; it does not include count correlations.
    coefficients, covariance = np.polyfit(x[good], y[good], 1, cov=True)
    return {
        "slope": float(coefficients[0]),
        "intercept": float(coefficients[1]),
        "slope_error": float(np.sqrt(covariance[0, 0])),
        "intercept_error": float(np.sqrt(covariance[1, 1])),
    }


def differential_count_errors(flux, flux_error, header, edges, draws=MC_DRAWS):
    """Simulate counts in separate magnitude bins, without accumulating them."""
    flux, flux_error = np.asarray(flux, float), np.asarray(flux_error, float)
    edges = np.asarray(edges, float)
    if draws < 2 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("Need at least two draws and increasing bin edges.")
    if flux.shape != flux_error.shape or np.any(~np.isfinite(flux)):
        raise ValueError("Flux arrays must have matching shapes and finite fluxes.")
    if np.any(~np.isfinite(flux_error)) or np.any(flux_error < 0):
        raise ValueError("Flux errors must be finite and nonnegative.")

    rng = np.random.default_rng(RANDOM_SEED + 2)
    simulated = np.zeros((draws, len(edges) - 1), dtype=float)
    zero_point = float(header["MAGZPT"])
    conversion = float(header["GAIN"]) / float(header["EXPTIME"])
    zp_error = float(header["MAGZRR"])

    for trial in range(draws):
        # Perturb each measured flux, just as in the cumulative-count method.
        draw_flux = flux + rng.normal(0, flux_error)
        magnitude = np.full(len(flux), np.inf)
        positive = draw_flux > 0
        magnitude[positive] = (
            zero_point + rng.normal(0, zp_error)
            - 2.5 * np.log10(draw_flux[positive] * conversion)
        )
        # Each bin contains lower_edge <= magnitude < upper_edge.
        # Differences of cumulative COUNTS give disjoint counts. We do not
        # subtract cumulative errors, which would be incorrect.
        bin_counts = np.diff(counts_from_magnitudes(magnitude, edges))
        simulated[trial] = rng.poisson(bin_counts)

    # Poisson fluctuations are already present; do not add sqrt(N) again.
    return np.std(simulated, axis=0, ddof=1), np.mean(simulated, axis=0)


def save_counts_plot_with_error(x, y, y_error, fit_range, output_path, differential=False):
    """Draw a counts plot with a simple fitted slope and its polyfit error."""
    x, y, y_error = np.asarray(x), np.asarray(y), np.asarray(y_error)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_error)
    fit_mask = good & (x >= fit_range[0]) & (x <= fit_range[1])
    if fit_range[0] >= fit_range[1]:
        raise ValueError("Fit range must increase.")
    # A sparse differential catalogue may have fewer than three nonempty bins.
    fit = line_fit_with_error(x[fit_mask], y[fit_mask]) if fit_mask.sum() >= 3 else None

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.errorbar(
        x[good], y[good], yerr=y_error[good], fmt="o-", color="royalblue",
        markersize=4, capsize=2, label="Counts: Monte Carlo errors",
    )
    ax.axvspan(
        *fit_range, color="grey", alpha=0.10,
        label=f"Fit range: {fit_range[0]:g}–{fit_range[1]:g}",
    )
    if fit is not None:
        xx = np.linspace(fit_range[0], fit_range[1], 200)
        line = np.poly1d([fit["slope"], fit["intercept"]])
        ax.plot(
            xx, line(xx), color="red", linewidth=3,
            label=f"Descriptive slope = {fit['slope']:.3f} ± {fit['slope_error']:.3f}",
        )
        anchor = np.mean(xx)
        ax.plot(
            x[good], line(anchor) + 0.6 * (x[good] - anchor), "--", color="grey",
            label="Reference slope = 0.6",
        )
    else:
        ax.text(0.03, 0.95, "Too few nonempty bins for a slope fit",
                transform=ax.transAxes, va="top")
    if differential:
        ax.set(
            xlabel="Aperture magnitude, r (bin centre)",
            ylabel=r"$\log_{10}[(\Delta N/(A\,\Delta m))/(1\,\mathrm{deg}^{-2}\,\mathrm{mag}^{-1})]$",
            title="Observed differential counts",
        )
    else:
        ax.set(
            xlabel="Limiting aperture magnitude, r",
            ylabel=r"$\log_{10}[(N(<m)/A)/(1\,\mathrm{deg}^{-2})]$",
            title="Observed cumulative counts",
        )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return fit


def add_slope_error_and_differential_counts(table, output, header):
    """Extend the saved analysis while leaving the original functions intact."""
    output = Path(output)
    cumulative = Table.read(output / "number_counts.ecsv")
    area = float(cumulative.meta["area_deg2"])
    fit_range = tuple(cumulative.meta["fit_range"])
    step = float(cumulative.meta["mag_step"])
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Sky area must be finite and positive.")
    selected = np.asarray(table["count_sample"], bool) & ~np.asarray(table["is_star"], bool)
    magnitudes = np.asarray(table["magnitude"][selected], float)
    if not len(magnitudes) or np.any(~np.isfinite(magnitudes)):
        raise ValueError("Need a nonempty count sample with finite magnitudes.")
    # Confirm that the existing counts and the supplied catalogue describe
    # the same sample before adding anything to its saved results.
    if not np.array_equal(
        counts_from_magnitudes(magnitudes, cumulative["limiting_magnitude"]),
        cumulative["count"],
    ):
        raise ValueError("Saved cumulative counts do not match this catalogue.")

    fit = save_counts_plot_with_error(
        cumulative["limiting_magnitude"], cumulative["log10_density"],
        cumulative["log_error_mc"], fit_range, output / "cumulative_number_counts.png",
    )
    fit_note = (
        "Unweighted np.polyfit(degree=1, cov=True); errors are square roots of "
        "its scaled covariance diagonal. Formal line-fit errors only; "
        "count correlations and systematic uncertainties are not included."
    )
    if fit is not None:
        cumulative.meta.update(fit)
    cumulative.meta["fit_error_note"] = fit_note
    cumulative.write(output / "number_counts.ecsv", overwrite=True)

    # Use the same 0.5-mag grid as the cumulative counts, including its first
    # zero-count edge. Plot each separate bin at the midpoint of its edges.
    edges, _ = cumulative_counts(magnitudes, step)
    counts = np.diff(counts_from_magnitudes(magnitudes, edges))
    widths = np.diff(edges)
    centres = (edges[:-1] + edges[1:]) / 2
    flux = np.asarray(table["aperture_flux_adu"][selected], float)
    error = np.asarray(table["flux_error_adu"][selected], float)
    count_error, mean_counts = differential_count_errors(flux, error, header, edges)

    # Divide by BOTH sky area and bin width to obtain counts per deg^2 per mag.
    density = counts / (area * widths)
    density_error = count_error / (area * widths)
    positive = counts > 0
    log_density = np.full(len(counts), np.nan)
    log_error = np.full(len(counts), np.nan)
    log_density[positive] = np.log10(density[positive])
    log_error[positive] = count_error[positive] / (np.log(10) * counts[positive])
    differential = Table({
        "magnitude_lower": edges[:-1], "magnitude_upper": edges[1:],
        "magnitude_centre": centres, "bin_width": widths, "count": counts,
        "density_deg2_mag": density, "density_error_mc": density_error,
        "log10_density": log_density, "count_error_mc": count_error,
        "count_mean_mc": mean_counts, "log_error_mc": log_error,
    })
    differential_fit = save_counts_plot_with_error(
        centres, log_density, log_error, fit_range,
        output / "differential_number_counts.png", differential=True,
    )
    if differential_fit is not None:
        differential.meta.update(differential_fit)
    differential.meta.update({
        "area_deg2": area, "mag_step": step, "fit_range": list(fit_range),
        "mc_draws": MC_DRAWS, "random_seed": RANDOM_SEED + 2,
        "fit_error_note": fit_note,
        "error_note": "Monte Carlo Poisson + flux/zero-point errors; first-order log bars. No completeness, classification or cosmic-variance errors.",
        "bin_note": "Lower edge included, upper edge excluded. Empty bins retained in table but omitted from logarithmic plot and fit. Fit uses bin centres.",
    })
    differential.write(output / "differential_number_counts.ecsv", overwrite=True)

    if fit is not None:
        table.meta["descriptive_slope_error"] = fit["slope_error"]
        table.meta["slope_error_note"] = fit_note
        table.write(output / "classified_sources.ecsv", overwrite=True)
        print(f"Cumulative slope: {fit['slope']:.6f} +/- {fit['slope_error']:.6f}")
    if differential_fit is not None:
        print(f"Differential slope: {differential_fit['slope']:.6f} +/- {differential_fit['slope_error']:.6f}")
    return fit, differential_fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-range", nargs=2, type=float, default=(19.0, 23.0))
    parser.add_argument("--measurement-mag-limit", type=float, default=MEASUREMENT_MAG_LIMIT)
    parser.add_argument("--classification-mag-limit", type=float, default=CLASSIFICATION_MAG_LIMIT)
    parser.add_argument(
        "--reference-labels",
        nargs="+",
        type=int,
        help="Optional checked segmentation IDs, not source row IDs.",
    )
    args = parser.parse_args()
    classified, fit = run_analysis(
        args.measurement_mag_limit,
        args.classification_mag_limit,
        tuple(args.fit_range),
        args.reference_labels,
    )
    add_slope_error_and_differential_counts(classified, OUTPUT_DIR, fits.getheader(FITS_PATH))
