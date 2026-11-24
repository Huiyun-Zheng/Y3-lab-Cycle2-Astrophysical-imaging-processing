"""Run circular Moffat classification and aperture photometry on neighbour-model results."""

import sys
sys.dont_write_bytecode = True

import argparse
from collections import Counter
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from scipy import ndimage as ndi
from scipy.stats import chi2

from circular_moffat import measure_psf_widths, select_psf_candidates


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
FITS_PATH = PROJECT_DIR / "Astro/Fits_Data/mosaic.fits"
# The complete runner builds these files locally before this final analysis.
DETECTION_DIR = CODE_DIR / "results" / "field_quality_review"
PRODUCTS_PATH = CODE_DIR / "results" / "aperture_review" / "detection_products.fits"
OUTPUT_DIR = CODE_DIR / "results" / "field_quality_review"

MEASUREMENT_MAG_LIMIT = 22.0
CLASSIFICATION_MAG_LIMIT = 22.0
REFERENCE_LABELS = [6720, 5677, 7009, 5741, 6315, 6998, 6658, 4392, 3534, 4958, 3324]
MAG_STEP = 0.5
MC_DRAWS = 2000
RANDOM_SEED = 20261118
BLANK_APERTURES = 48  # Desired number of blank apertures
MIN_BLANK_APERTURES = 24  # Minimum acceptable number of blank apertures


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


def estimate_blank_scale(residual, rms, forbidden, row, dx, dy, rng):
    """
    Estimate a local noise correction factor using blank apertures.

    Place non-overlapping copies of the target aperture at random
    positions around the target, avoiding image boundaries and
    forbidden pixels.

    Divide the residual flux of each blank aperture by its own RMS
    uncertainty to account for variations in local noise levels.
    Use the sigma-clipped standard deviation of these normalised
    fluxes to estimate the noise correction factor.
    """
    # N.B. dx/dy here are the integer x/y offsets of all pixels inside the target aperture,
    # measured relative to its rounded centre. (different form dx/dy in the function aperture_offsets).
    # rng is the random generator used to sample candidate apertures.

    height, width = residual.shape
    a = max(float(row["aperture_a_pix"]), float(row["aperture_b_pix"]))
    half = int(np.ceil(a)) + 2  # used for boundary chekcing

    # Store the normalised residual flux of each accepted aperture.
    # Also store its centre to avoid overlap with later apertures.
    samples, accepted_centres = [], []

    # Wider search is a fallback for crowded regions/large apertures, not a replacement with smaller apertures.
    # The actual radius is recorded.
    used_radius = 500.0
    for search_radius in (500.0, 1000.0):
        used_radius = search_radius

        for attempt in range(4):
            # Generate 256 random directions / distances around the target.
            theta = rng.uniform(0, 2 * np.pi, 256)
            distance = np.sqrt(rng.uniform((2 * a + 20) ** 2, search_radius**2, 256))

            # Convert the random distances and angles into absolute x/y coordinates in the image.
            # Round the coordinates to integers so that they can be used for imaging indexing.
            xs = np.rint(float(row["x"]) + distance * np.cos(theta)).astype(int)
            ys = np.rint(float(row["y"]) + distance * np.sin(theta)).astype(int)

            # Check whether each candidate centre is far enough from all four image boundaries.
            good = (xs >= half) & (xs < width - half) & (ys >= half) & (ys < height - half)
            xs, ys = xs[good], ys[good]  # candidate centres
            if len(xs) == 0:
                continue

            px, py = xs[:, None] + dx, ys[:, None] + dy
            clean = ~np.any(forbidden[py, px], axis=1)

            for x, y in zip(xs[clean], ys[clean]):
                if accepted_centres:
                    centres = np.asarray(accepted_centres)

                    # Calculate the distance from the current candidate to every previously accepted centre.
                    # Reject the candidate if any centre is closer than twice the semi-major axis plus one pixel.
                    # This conservatively prevents aperture overlap.
                    if np.any(np.hypot(centres[:, 0] - x, centres[:, 1] - y) < 2 * a + 1):
                        continue

                # Add the individual pixel uncertainties in quadrature to estimate the uncertainty of the total aperture flux.
                aperture_rms = np.sqrt(np.sum(rms[y + dy, x + dx].astype(float) ** 2))
                # Divide the total residual flux by the aperture RMS to account for differences in local noise levels.
                samples.append(float(np.sum(residual[y + dy, x + dx], dtype=float)) / aperture_rms)
                accepted_centres.append((x, y))

                if len(samples) >= BLANK_APERTURES:
                    break

            if len(samples) >= BLANK_APERTURES:
                break

        if len(samples) >= MIN_BLANK_APERTURES:
            break

    # If there are still too few blank apertures after searching both regions,
    # do not estimate an empirical noise correction.
    if len(samples) < MIN_BLANK_APERTURES:
        return 1.0, len(samples), np.nan, used_radius, "rms_only_fallback"

    _, median, scatter = sigma_clipped_stats(np.asarray(samples), sigma=3, maxiters=5)
    # Set a lower limit of 1 so that the empirical correction can increase the uncertainty but cannot reduce it.
    return max(1.0, float(scatter)), len(samples), float(median), used_radius, "local_blank"


def add_photometric_uncertainties(table, residual, rms, labels, mask, header):
    """
    Calculate flux and magnitude uncertainties for selected sources.

    Estimate background noise using nearby blank apertures and include
    source shot noise based on the detector gain.

    Convert flux uncertainties into magnitude uncertainties, including
    the shared photometric zero-point uncertainty.
    """
    # mask here identifying invalid or excluded pixels.

    result = table.copy()
    gain = float(header["GAIN"])
    zp_error = float(header["MAGZRR"])

    # Identify pixels that cannot be used for blank apertures.
    # Exclude masked pixels, invalid noise values, and detected sources
    # with their masks expanded to cover nearby wings.
    forbidden = (
        mask
        | ~np.isfinite(residual)
        | ~np.isfinite(rms)
        | (rms <= 0)
        | ndi.binary_dilation(labels > 0, iterations=3)
    )

    columns = [
        "flux_error_adu",
        "background_flux_error_adu",
        "source_flux_error_adu",
        "background_correlation_factor",
        "blank_median_normalised",
        "blank_search_radius",
        "magnitude_error_random",
        "magnitude_error_total",
        "magnitude_error_lower",
        "magnitude_error_upper",
        "photometric_snr",
    ]
    for name in columns:
        result[name] = np.full(len(result), np.nan)
    result["blank_aperture_count"] = np.zeros(len(result), int)
    result["uncertainty_status"] = np.full(len(result), "not_in_sample", dtype="U32")
    result["magnitude_error_zeropoint"] = np.full(len(result), zp_error)

    # Create a reproducible random generator for blank-aperture sampling.
    rng = np.random.default_rng(RANDOM_SEED)
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
            or np.any(px >= residual.shape[1])
            or np.any(py < 0)
            or np.any(py >= residual.shape[0])
        ):
            raise ValueError(f'Invalid accepted aperture: ID {row["id"]}.')

        # Calculate the background variance.
        variance = np.sum(rms[py, px].astype(float) ** 2)
        if not np.isfinite(variance) or variance <= 0 or np.any(mask[py, px]):
            raise ValueError(f'Invalid accepted-aperture noise/mask: ID {row["id"]}.')

        # Measure the scatter of nearby blank apertures to estimate the background noise correction factor.
        scale, count, median, search_radius, status = estimate_blank_scale(residual, rms, forbidden, row, dx, dy, rng)

        flux = float(row["aperture_flux_adu"])
        # Correct the background variance using the empirical blank-aperture noise scale.
        background_variance = variance * scale**2
        # Estimate the shot-noise variance from the measured flux and detector gain, assuming Poisson photon statistics.
        source_variance = flux / gain
        # Combine background and source variances to calculate the total random flux uncertainty.
        flux_error = np.sqrt(background_variance + source_variance)
        fractional_error = flux_error / flux

        result["flux_error_adu"][i] = flux_error
        result["background_flux_error_adu"][i] = np.sqrt(background_variance)
        result["source_flux_error_adu"][i] = np.sqrt(source_variance)
        result["background_correlation_factor"][i] = scale
        result["blank_aperture_count"][i] = count
        result["blank_median_normalised"][i] = median
        result["blank_search_radius"][i] = search_radius
        result["uncertainty_status"][i] = status
        result["photometric_snr"][i] = 1 / fractional_error
        random_mag_error = 2.5 / np.log(10) * fractional_error
        result["magnitude_error_random"][i] = random_mag_error
        result["magnitude_error_total"][i] = np.hypot(random_mag_error, zp_error)
        # Calculate the lower magnitude uncertainty from the brighter flux endpoint, including zero-point uncertainty.
        result["magnitude_error_lower"][i] = np.hypot(2.5 * np.log10(1 + fractional_error), zp_error)
        # Calculate the upper magnitude uncertainty from the fainter flux endpoint.
        # Only fo this when the lower flux endpoint is positive.
        # Otherwise, leave the upper magnitude error as NaN.
        if fractional_error < 1:
            result["magnitude_error_upper"][i] = np.hypot(-2.5 * np.log10(1 - fractional_error), zp_error)

        if (number + 1) % 1000 == 0:
            print(f"Aperture uncertainties: {number + 1}/{len(chosen)}", flush=True)
    result.meta["error_note"] = (
        "Translated blank-aperture background scatter plus F_ADU/GAIN shot variance; "
        "MAGZRR is common to all sources. No flux correction applied."
    )
    summary = {
        "zeropoint_sigma_mag": zp_error,
        "effective_gain_assumed": gain,
        "blank_status": dict(Counter(str(v) for v in result["uncertainty_status"][chosen])),
        "median_background_factor": float(
            np.median(result["background_correlation_factor"][chosen])
        ),
        "median_magnitude_random_error": float(np.median(result["magnitude_error_random"][chosen])),
        "median_magnitude_total_error": float(np.median(result["magnitude_error_total"][chosen])),
        "nonpositive_lower_flux_endpoints": int(
            np.count_nonzero(~np.isfinite(result["magnitude_error_upper"][chosen]))
        ),
        "seed": RANDOM_SEED,
    }
    return result, summary


def counts_from_magnitudes(magnitudes, limits):
    """Count the number of galaxies brighter than each magnitude limit."""
    counts = []

    for limit in limits:
        count = np.count_nonzero(magnitudes < limit)
        counts.append(count)

    return np.array(counts)


def count_error_model(flux, flux_error, header, limits, observed_counts, draws=MC_DRAWS):
    """
    Estimate the photometric and Poisson covariance of cumulative
    galaxy number counts.

    Use Monte Carlo simulations to propagate flux measurement errors
    and the shared zero-point uncertainty into cumulative counts.
    Calculate Poisson counting covariance separately.

    The detected sources and survey area remain fixed.
    """

    rng = np.random.default_rng(RANDOM_SEED + 1)
    simulated = np.zeros((draws, len(limits)), dtype=float)  # An array to store cumulative counts from each simulation.
    zero_point = float(header["MAGZPT"])
    conversion = float(header["GAIN"]) / float(header["EXPTIME"])
    zp_error = float(header["MAGZRR"])

    for trial in range(draws):
        # Randomly vary each galaxy's flux using its measured uncertainty.
        draw_flux = flux + rng.normal(0, flux_error)

        # Initially exclude nonpositive fluxes by assigning infinite magnitude.
        magnitude = np.full(len(flux), np.inf)
        positive = draw_flux > 0

        # Comvert positive simulated fluxes into magnitudes.
        # Apply one shared zero-point offset to all galaxies in this draw
        magnitude[positive] = (zero_point + rng.normal(0, zp_error) - 2.5 * np.log10(draw_flux[positive] * conversion))
        simulated[trial] = counts_from_magnitudes(magnitude, limits)

    # Calculate how flux measurement errors affect the cumulative galaxy counts.
    # The diagonal values give the count variances, while the off-diagonal
    # values show how count fluctuations at different magnitude limits are related.
    photometric_covariance = np.cov(simulated, rowvar=False, ddof=1)

    # Calculate the Poisson counting covariance from the observed galaxy counts.
    # The diagonal values give the Poisson variances (equal to the counts),
    # while the off-diagonal values account for galaxies shared between
    # different cumulative magnitude limits.
    poisson_covariance = np.minimum.outer(observed_counts, observed_counts).astype(float)

    return simulated, poisson_covariance, photometric_covariance


def descriptive_line_fit(x, y, covariance):
    """
    Fit a straight line using ordinary least squares (OLS) and
    propagate the data covariance into the slope uncertainty.

    The covariance matrix is used to estimate the uncertainty
    of the fitted slope, but not to determine the best-fit line.
    """

    origin = float(np.mean(x))

    # Build the design matrix for fitting a straight line.
    # The first column represents the slope term,
    # and the second column represents the intercept.
    design = np.column_stack([x - origin, np.ones(len(x))])
    # Prepare the matrix used to calculate the OLS fit parameters.
    projection = np.linalg.pinv(design)
    # Calculate the best-fit slope and centred intercept.
    parameters = projection @ y
    # Propagate the covariance of the y values into the fitted parameters,
    # accounting for correlations between data points.
    parameter_covariance = projection @ covariance @ projection.T

    return {
        "slope": float(parameters[0]),
        "intercept": float(parameters[1] - origin * parameters[0]),
        "slope_error_propagated": float(np.sqrt(parameter_covariance[0, 0])),
    }


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
    simulated, poisson, photometric = count_error_model(flux, error, header, limits, counts)
    # Convert the count covariance into log-count covariance.
    # This includes correlations between different magnitude limits.
    conversion = np.log(10) * counts
    covariance_log = (poisson + photometric) / np.outer(conversion, conversion)
    y = np.log10(counts / area)
    fit_mask = (limits >= fit_range[0]) & (limits <= fit_range[1])
    fit_covariance = covariance_log[np.ix_(fit_mask, fit_mask)]
    fit = descriptive_line_fit(limits[fit_mask], y[fit_mask], fit_covariance)

    # Calculate the lower and upper Poisson count limits for an approximately 68.27% confidence interval.
    alpha = 1 - 0.682689492
    lower = 0.5 * chi2.ppf(alpha / 2, 2 * counts)
    upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (counts + 1))
    photometric_log_error = np.sqrt(np.diag(photometric)) / conversion
    down = np.hypot(np.log10(counts / lower), photometric_log_error)
    up = np.hypot(np.log10(upper / counts), photometric_log_error)
    results = Table(
        {
            "limiting_magnitude": limits,
            "count": counts,
            "log10_density": y,
            "poisson_lower_count": lower,
            "poisson_upper_count": upper,
            "photometric_count_sigma": np.sqrt(np.diag(photometric)),
            "photometric_count_mean_shift": np.mean(simulated, axis=0) - counts,
            "log_error_lower": down,
            "log_error_upper": up,
        }
    )
    results.meta.update(fit)
    results.meta["area_deg2"] = float(area)
    results.meta["fit_range"] = list(fit_range)
    results.meta["mag_step"] = MAG_STEP
    results.meta["error_note"] = (
        "Approximate Poisson + flux/zero-point propagation; no completeness, classification or cosmic-variance errors."
    )
    results.write(output / "number_counts.ecsv", overwrite=True)

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.errorbar(
        limits,
        y,
        yerr=[down, up],
        fmt="o-",
        color="royalblue",
        markersize=4,
        capsize=2,
        label="Counts: Poisson + photometric errors",
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
        linewidth=2,
        label=f"Descriptive slope = {fit['slope']:.3f} ± {fit['slope_error_propagated']:.3f}",
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
        ylabel=r"$\log_{10}[(N(<m)/A)/(1\,\mathrm{deg}^{-2})]$",
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
        finite_error = (
            selection
            & np.isfinite(table["magnitude_error_lower"])
            & np.isfinite(table["magnitude_error_upper"])
        )
        ax.errorbar(
            table["magnitude"][finite_error],
            table["psf_fwhm"][finite_error],
            xerr=[
                table["magnitude_error_lower"][finite_error],
                table["magnitude_error_upper"][finite_error],
            ],
            fmt="none",
            ecolor=colour,
            elinewidth=0.55,
            alpha=0.25,
            zorder=1,
        )
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
    products_path = PRODUCTS_PATH
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
    if reference_labels is None:
        reference_labels = REFERENCE_LABELS
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

    classified, error_summary = add_photometric_uncertainties(
        classified, image - background, rms, labels, mask, header
    )
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
    print(f"Descriptive slope: {fit['slope']:.6f} +/- {fit['slope_error_propagated']:.6f}")
    print(f"Area: {area:.6f} deg^2")
    print(f"Outputs: {OUTPUT_DIR}")
    return classified, fit


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
    run_analysis(
        args.measurement_mag_limit,
        args.classification_mag_limit,
        tuple(args.fit_range),
        args.reference_labels,
    )
