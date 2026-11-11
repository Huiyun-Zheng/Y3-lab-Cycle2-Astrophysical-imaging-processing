"""Remove likely stars, count all remaining photometrically usable sources.

Run in astro-image: python preliminary_number_counts.py
No existing detection code is imported or rerun. No catalogue is written.
Choose another --output path to run again without replacing any existing file.

Calibration follows B2 lab guide equations (3)-(5):
    m = MAGZPT - 2.5 * log10(background-subtracted aperture counts)
Do not divide these mosaic counts by EXPTIME under this lab convention.
The existing adaptive-aperture fluxes are reused, without total-flux correction.
All sources not identified as stars are treated as galaxies, even if their
morphology cannot be measured. Existing photometry-quality cuts still apply.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from star_galaxy_preliminary import MorphologySettings, classify_sources


def calibrate_magnitudes(catalogue, zero_point):
    """Calibrate only positive, finite aperture fluxes with no quality flags."""

    result = catalogue.copy()
    flux = np.asarray(result["aperture_flux_adu"], dtype=float)
    accepted = np.asarray(result["usable"], dtype=bool) & np.isfinite(flux) & (flux > 0)
    magnitude = np.full(len(result), np.nan)
    magnitude[accepted] = zero_point - 2.5 * np.log10(flux[accepted])
    result["magnitude"] = magnitude
    result["count_sample"] = accepted

    return result


def cumulative_counts(magnitudes, limits):
    """Count sources strictly brighter than each magnitude limit."""

    magnitudes = np.asarray(magnitudes)
    sorted_magnitudes = np.sort(magnitudes[np.isfinite(magnitudes)])

    return np.searchsorted(sorted_magnitudes, limits, side="left")


def save_final_figure(catalogue, reference, area, zero_point, zero_point_error, output):
    """Show morphology and observed cumulative counts in one final PNG."""

    sample = catalogue[np.asarray(catalogue["count_sample"])]

    if len(sample) == 0:
        raise ValueError("No positive, unflagged aperture fluxes remain.")

    magnitudes = np.asarray(sample["magnitude"])
    classes = np.asarray(sample["source_class"])
    limits = np.arange(np.floor(magnitudes.min()), np.ceil(magnitudes.max()) + 0.25, 0.25)
    colours = {"star": "#d48a00", "galaxy": "#187e9c"}
    names = {"star": "Stars removed", "galaxy": "Remaining sources = galaxies"}

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (shape_ax, count_ax) = plt.subplots(1, 2, figsize=(13, 5.6), width_ratios=[1, 1.2])

    for label in ["galaxy", "star"]:
        selected = classes == label
        shape_ax.scatter(
            magnitudes[selected], sample["half_light_radius_pix"][selected],
            s=12, alpha=0.55, color=colours[label], edgecolors="none",
            label=f"{names[label]} ({np.count_nonzero(selected):,})",
        )

    shape_ax.axhline(reference["half_light_radius_pix"], color=colours["star"], ls="--", lw=1)
    shape_ax.set(
        xlabel="Calibrated aperture magnitude, r (AB)",
        ylabel="Half-light radius within 12 pixels (pixel)",
        title="1  |  Identify and remove stars",
    )
    shape_ax.legend(loc="upper right", frameon=False, fontsize=9)
    missing_size = np.count_nonzero(~np.isfinite(sample["half_light_radius_pix"]))
    shape_ax.text(
        0.03, 0.03,
        f"Dashed line: empirical stellar size\n"
        f"{missing_size:,} lack a size: retained in counts, not plotted here",
        transform=shape_ax.transAxes, fontsize=8, color="#555555",
    )

    galaxy_counts = cumulative_counts(magnitudes[classes == "galaxy"], limits)
    positive = galaxy_counts > 0

    if np.any(positive):
        counts = galaxy_counts[positive]
        density = counts / area
        # Poisson-only approximation; cumulative points are correlated.
        lower = np.maximum(counts - np.sqrt(counts), 0.5) / area
        upper = (counts + np.sqrt(counts)) / area
        count_ax.errorbar(
            limits[positive], np.log10(density),
            yerr=[np.log10(density / lower), np.log10(upper / density)],
            fmt="o-", ms=3.5, lw=1.6, capsize=2,
            color=colours["galaxy"], label="All remaining sources (Poisson errors)",
        )

        # The 0.6 slope is a visual reference, not a fit or a completeness claim.
        anchor_index = np.flatnonzero(positive)[len(counts) // 2]
        reference_curve = (
            np.log10(galaxy_counts[anchor_index] / area)
            + 0.6 * (limits - limits[anchor_index])
        )
        count_ax.plot(limits, reference_curve, "--", color="#444444", lw=1.2, label="Slope 0.6 (arbitrary normalisation)")

    count_ax.set(
        xlabel="Limiting aperture magnitude, r (AB)",
        ylabel=r"$\log_{10}\,[N(<m)\,/\,\mathrm{deg}^{2}]$",
        title="2  |  Observed cumulative number counts",
        ylim=(np.log10(0.5 / area), np.log10(len(sample) / area) + 0.35),
    )
    count_ax.legend(loc="lower right", frameon=False, fontsize=8)
    count_ax.text(
        0.03, 0.97,
        f"Unmasked area: {area:.4f} deg²\nMAGZPT = {zero_point:.2f} ± {zero_point_error:.2f} mag",
        transform=count_ax.transAxes, va="top", fontsize=9,
    )

    for ax in (shape_ax, count_ax):
        ax.grid(alpha=0.18)

    fig.suptitle("Deep galaxy survey | preliminary result", fontsize=16, y=0.98)
    fig.text(
        0.5, 0.025,
        "Only identified stars are removed; all remaining sources with valid, unflagged photometry are counted as galaxies.\n"
        "No completeness or total-flux correction. Error bars exclude classification, flux and zero-point uncertainty.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))

    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects every existing file, including earlier plots.
    with output.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=180, facecolor="white")

    plt.close(fig)


def run_analysis(fits_path, detection_dir, output):
    """Keep classification before calibration; use the saved detection masks."""

    if output.exists():
        raise FileExistsError(f"Preserving existing file: {output}. Choose a new --output path.")

    catalogue = Table.read(detection_dir / "sources.ecsv")

    with fits.open(fits_path, memmap=False) as hdul:
        header = hdul[0].header.copy()
        image = hdul[0].data.astype(np.float32)

    with fits.open(detection_dir / "detection_products.fits", memmap=False) as products:
        background = products["BACKGROUND"].data
        noise = products["RMS"].data
        excluded = products["DETMASK"].data.astype(bool)
        segments = products["SOURCES"].data

        if not all(array.shape == image.shape for array in [background, noise, excluded, segments]):
            raise ValueError("Detection products do not match the input image dimensions.")

        excluded |= ~np.isfinite(image) | ~np.isfinite(background) | ~np.isfinite(noise)
        classified, reference = classify_sources(
            catalogue, image - background, noise, excluded, segments,
            float(products[0].header["SATLEVEL"]), MorphologySettings(),
        )

        # This is geometric usable area only. Quality cuts and spatially varying
        # detection depth introduce additional, uncorrected selection losses.
        pixel_area = abs(float(header["PIXSCAL1"]) * float(header["PIXSCAL2"])) / 3600 ** 2
        area = np.count_nonzero(~excluded) * pixel_area

    zero_point = float(header["MAGZPT"])
    zero_point_error = float(header["MAGZRR"])
    calibrated = calibrate_magnitudes(classified, zero_point)
    save_final_figure(calibrated, reference, area, zero_point, zero_point_error, output)
    print(f"Saved: {output.resolve()}")

    return calibrated, reference, area


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, default=base_dir / "Astro/Fits_Data/mosaic.fits")
    parser.add_argument("--detections", type=Path, default=base_dir / "output/source_detection_auto")
    parser.add_argument("--output", type=Path, default=base_dir / "output/preliminary_analysis/final_number_counts_stars_removed.png")
    args = parser.parse_args()
    run_analysis(args.fits, args.detections, args.output)
