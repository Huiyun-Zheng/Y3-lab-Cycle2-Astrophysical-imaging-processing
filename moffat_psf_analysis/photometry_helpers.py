"""Unchanged calibration/count plotting functions copied for an isolated run."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

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
