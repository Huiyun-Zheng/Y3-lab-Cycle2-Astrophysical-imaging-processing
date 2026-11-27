"""
Full-field galaxy injections with the fixed detection/photometry pipeline.

Per-image detector products live in a temporary
directory and are overwritten, then deleted automatically.
"""

import sys
sys.dont_write_bytecode = True

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.special import gammaincinv

from detector_adapter import FixedPipeline, IMAGE_PATH, CATALOGUE_PATH, CODE_PATHS, ROOT


HERE = Path(__file__).resolve().parent
PSF_PATH = HERE / "moffat_parameters.ecsv"
OUTPUT = HERE / "results"

# Main experiment: two profiles, three intrinsic half-light radii, 0.5-mag bins.
PROFILES = {"exponential": 1.0, "de_vaucouleurs": 4.0}
RADII = (3.0, 4.0, 5.0)
MAG_EDGES = np.arange(16.0, 24.51, 0.5)
INJECTIONS_PER_CELL = 50
BATCH_SIZE = 40
SEED = 20261126
MATCH_RADIUS = 3.0
NEIGHBOUR_RADIUS = 15.0  # diagnostic only, not an extra rejection threshold.

AXIS_RATIO = 0.7  # the ratio of the minor axis to the major axis.
ANGLE = np.pi / 6
OVERSAMPLE = 5
PSF_RADIUS = 50
ENCLOSED_FRACTION = 0.99
# Subpixels offsets of the galaxy centre in detector-pixel units.
# Each pair gives the (x, y) offset from the pixel centre, in units of detector pixels.
PHASES = ((-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def magnitude_to_flux(magnitude, header):
    return 10 ** (0.4 * (header["MAGZPT"] - magnitude)) * header["EXPTIME"] / header["GAIN"]


def flux_to_magnitude(flux, header):
    return header["MAGZPT"] - 2.5 * np.log10(flux * header["GAIN"] / header["EXPTIME"])


def make_template(n, r_eff, phase, alpha, beta, oversample=OVERSAMPLE):
    """
    Generate a pixel-integrated artificial galaxy with a total flux of 1.

    Create an elliptical Sersic galaxy on a fine subpixel grid,
    blur it using a Moffat PSF, and combine the subpixels into
    normal detector pixels.

    The intrinsic galaxy is truncated at a chosen enclosed-light
    fraction to keep the template size finite.
    """

    # Calculate the Sersic constant so that the effective radius
    # enclose half of the total light of the untruncated galaxy.
    b_n = gammaincinv(2 * n, 0.5)
    # Find the radius that encloses the chosen fraction of galaxy light.
    # Light outside this radius will be removed.
    outer_radius = r_eff * (gammaincinv(2 * n, ENCLOSED_FRACTION) / b_n) ** n

    # Leave extra space around the galaxy for PSF broadening.
    half = int(np.ceil(outer_radius + PSF_RADIUS)) + 2
    # use an odd template size so that there is a central pixel.
    size = 2 * half + 1

    # Divide each detector pixel into smaller subpixels.
    grid = (np.arange(size * oversample) + 0.5) / oversample - half - 0.5
    dx = grid[None, :] - phase[0]
    dy = grid[:, None] - phase[1]
    u = dx * np.cos(ANGLE) + dy * np.sin(ANGLE)
    v = -dx * np.sin(ANGLE) + dy * np.cos(ANGLE)
    radius = np.hypot(u, v / AXIS_RATIO)


    # Calculate the relative surface brightness at each subpixel using the Sersic profile.
    # The overall amplitude is not needed because the profile will be normalised.
    intrinsic = np.exp(-b_n * (radius / r_eff) ** (1 / n))
    intrinsic[radius > outer_radius] = 0
    intrinsic /= intrinsic.sum()

    psf_grid = np.arange(-PSF_RADIUS * oversample, PSF_RADIUS * oversample + 1) / oversample  # fine coordinate grid for the PSF
    radius_squared = psf_grid[:, None] ** 2 + psf_grid[None, :] ** 2
    psf = (1 + radius_squared / alpha ** 2) ** (-beta)
    # Normalise the PSF so that convolution redistributes the light without changing its total flux.
    psf /= psf.sum()

    convolved = fftconvolve(intrinsic, psf, mode="same")
    # Group the subpixels according to their detector pixels.
    # The four axes represent: (pixel row, subpixel row, pixel column, subpixel column).
    # Sum over the two subpixel axes to obtain the total flux in each detector pixel.
    grouped = convolved.reshape(size, oversample, size, oversample)
    pixels = grouped.sum(axis=(1, 3))
    # Remove small negative errors caused by numerical errors.
    pixels = np.maximum(pixels, 0)
    pixels /= pixels.sum()

    return pixels


def template_library(alpha, beta):
    """Precompute and store all galaxy templates."""
    library = {}

    for profile, n in PROFILES.items():
        for r_eff in RADII:
            for phase_id, phase in enumerate(PHASES):
                library[profile, r_eff, phase_id] = make_template(n, r_eff, phase, alpha, beta)

    return library


def place_batch(tasks, rng, eligible_pixels, shape, library):
    """
    Choose injection positions and magnitudes for a batch of artificial galaxies.

    Allow artificial galaxies to overlap with real sources, but prevent artificial galaxies from overlapping with each other.
    """

    placed = []
    for task in tasks:
        profile, r_eff, low = task  # low is the lower edge of the magnitude bin
        # Randomly select one of the available subpixel offsets.
        phase_id = int(rng.integers(len(PHASES)))
        phase = PHASES[phase_id]

        template = library[profile, r_eff, phase_id]
        half = template.shape[0] // 2  # half-size of the square array containing the galaxy

        # Try up to 20000 random positions for this galaxy.
        for attempt in range(20000):

            # Randomly select an eligible pixel.
            # Convert its flattened index into row and column coordinates using the image width.
            py, px = divmod(int(rng.choice(eligible_pixels)), shape[1])
            x, y = px + phase[0], py + phase[1]

            # Accept the position only if it is sufficiently far from all previously placed injected galaxies.
            if all(np.hypot(x - p["x"], y - p["y"]) > np.sqrt(2) * (half + p["half"]) + 4 for p in placed):
                break
        else:
            raise ValueError("Cannot place this batch sparsely, reduce batch size.")

        placed.append({
            "profile": profile, "r_eff": r_eff, "mag_low": low,
            # Randomly choose a magnitude within the 0.5-magnitude bin
            "magnitude_in": float(rng.uniform(low, low + 0.5)),
            "phase_id": phase_id, "x": x, "y": y,
            "px": px, "py": py, "half": half,
        })

    return placed


def stamp_slices(source, shape):
    """Find matching regions in the original image and the galaxy template, accounting for image boundaries."""
    half, px, py = source["half"], source["px"], source["py"]

    x0, x1 = max(0, px - half), min(shape[1], px + half + 1)
    y0, y1 = max(0, py - half), min(shape[0], py + half + 1)

    image_slice = (slice(y0, y1), slice(x0, x1))
    # Convert the image coordinates into template coordinates.
    # Select only the part of the template that fits inside the original image.
    stamp_slice = (slice(y0 - py + half, y1 - py + half), slice(x0 - px + half, x1 - px + half))

    return image_slice, stamp_slice


def inject_batch(original, placed, library, header, rng):
    """
    Inject a batch of artificial galaxies into the original image.
    Including the photon shot noise of each injected galaxy.
    """

    image = original.copy()
    coverage = np.isfinite(original) & (original != 0)
    gain = float(header["GAIN"])

    for source in placed:
        template = library[source["profile"], source["r_eff"], source["phase_id"]]
        flux = float(magnitude_to_flux(source["magnitude_in"], header))
        image_slice, stamp_slice = stamp_slices(source, image.shape)
        # Distribute the total galaxy flux across individual pixels according to the normalised template.
        # Set the expected fluc to zero in pixels without valid covergae.
        expected = flux * template[stamp_slice] * coverage[image_slice]

        # Convert the expected flux in each pixel from ADU to electrons.
        # Draw electron counts from a Poisson distribution to simulate
        # the photon shot noise of the artificial galaxy.
        # Convert the simulated electron counts back to ADU.
        # No extra background noise is needed because the original
        # image already contains noise.
        added = rng.poisson(expected * gain) / gain
        # Add the simulated flux to the corresponding image pixels.
        image[image_slice] += added.astype(image.dtype)
        source["flux_in"] = flux
        source["flux_on_image_expected"] = float(expected.sum())
        source["flux_added_realized"] = float(added.sum())

    return image


def positional_matches(placed, catalogue, baseline_tree):
    """
    Match injected galaxies to detected sources using their positions.

    Require a unique one-to-one match and check for confusion with
    sources already detected in the original image.

    No flux agreement is required for positional matching.
    """

    positions = np.array([[p["x"], p["y"]] for p in placed])
    detected = np.column_stack([catalogue["x"], catalogue["y"]])
    # Build a spatial search tree from the detected source positions.
    # For each injected galaxy, find all detected sources within the matching radius.
    neighbours = cKDTree(detected).query_ball_point(positions, MATCH_RADIUS)
    # Count how many injected galaxies claim each detected source as a possible match.
    claims = np.zeros(len(catalogue), int)
    for candidates in neighbours:
        claims[candidates] += 1

    # Find the distance from each injected galaxy to the nearest source detected in the original image.
    baseline_distances = baseline_tree.query(positions)[0]
    matched_rows = []
    info = []

    # Examine the matching candidates and baseline distance of each injected galaxy.
    for candidates, distance in zip(neighbours, baseline_distances):
        # Accept a unique positional match only if:
        # 1. The injected galaxy has exactly one candidate.
        # 2. That candidate is not claimed by any other injection.
        # Otherwise, use -1 to indicate no unique match.
        row_index = candidates[0] if len(candidates) == 1 and claims[candidates[0]] == 1 else -1
        unique = row_index >= 0

        # For a uniquely matched source, find the distance from its detected position to the nearest baseline source.
        # Use NaN if there is no unique match.
        output_distance = baseline_tree.query(detected[row_index])[0] if unique else np.nan
        # Flag possible confusion with a pre-existing source if
        # either the injected position or the matched detected
        # position is within the matching radius of a baseline source.
        confused = bool(distance <= MATCH_RADIUS or (unique and output_distance <= MATCH_RADIUS))

        if not candidates:
            status = "no_match"
        elif not unique:
            status = "ambiguous_match"
        elif confused:
            status = "preexisting_match"
        else:
            status = "new_match"

        info.append({
            "unique_position_match": unique, "preexisting_confusion": confused,
            "baseline_distance_pix": float(distance),
            "near_neighbour_15pix": bool(distance < NEIGHBOUR_RADIUS),
            "match_status": status, "row_index": row_index,
            "detected": unique and not confused,
        })

        # Save the row index of every uniquely matched source,
        # including matches with pre-existing source confusion.
        if unique:
            matched_rows.append(row_index)

    return info, matched_rows


def expected_aperture_flux(source, row, library, pipeline, shape):
    """
    Calculate the expected flux of the injected galaxy inside the recovered photometric aperture, without noise.
    """
    template = library[source["profile"], source["r_eff"], source["phase_id"]]
    _, xx, yy, inside, _ = pipeline.review.aperture_grid(row, shape)
    sx = xx[inside] - source["px"] + source["half"]
    sy = yy[inside] - source["py"] + source["half"]
    good = (sx >= 0) & (sy >= 0) & (sx < template.shape[1]) & (sy < template.shape[0])

    # Sum the flux fractions of the template pixels inside the recovered aperture.
    # Multiply by the total injected flux to obtain the expected aperture flux in ADU.
    return float(source["flux_in"] * template[sy[good], sx[good]].sum())


def recovery_records(placed, info, catalogue, measured, library, pipeline, shape, batch_id):
    measured_by_id = {int(row["id"]): row for row in measured}
    rows = []

    for source, match in zip(placed, info):
        row = {
            "batch": batch_id, "profile": source["profile"], "r_eff_pix": source["r_eff"],
            "mag_low": source["mag_low"], "mag_high": source["mag_low"] + 0.5,
            "magnitude_in": source["magnitude_in"], "x": source["x"], "y": source["y"],
            "region": ("north" if source["y"] >= shape[0] / 2 else "south"),
            "flux_in_adu": source["flux_in"],
            "image_fraction": source["flux_on_image_expected"] / source["flux_in"],
            "flux_added_adu": source["flux_added_realized"],
            **{key: value for key, value in match.items() if key != "row_index"},
            "detection_id": -1, "usable_recovered": False, "counts_recovered": False,
            "flags": "no_unique_match", "source_class": "not_matched",
            "magnitude_out": np.nan, "magnitude_error": np.nan,
            "delta_mag_total": np.nan, "delta_mag_aperture": np.nan,
            "aperture_fraction": np.nan, "flux_pull": np.nan,
        }
        if match["unique_position_match"]:
            detected_row = catalogue[match["row_index"]]
            result = measured_by_id[int(detected_row["id"])]
            row["detection_id"] = int(result["id"])
            row["flags"] = str(result["flags"])
            row["source_class"] = str(result["source_class"])
            row["usable_recovered"] = bool(match["detected"] and result["count_sample"])
            row["counts_recovered"] = bool(match["detected"] and result["remaining_sample"])
            row["magnitude_out"] = float(result["magnitude"])
            row["magnitude_error"] = float(result["magnitude_error_total"])
            row["delta_mag_total"] = row["magnitude_out"] - source["magnitude_in"]

            expected = expected_aperture_flux(source, detected_row, library, pipeline, shape)
            row["aperture_fraction"] = expected / source["flux_in"]
            if result["count_sample"] and expected > 0:
                measured_flux = float(result["aperture_flux_adu"])
                row["delta_mag_aperture"] = -2.5 * np.log10(measured_flux / expected)
                row["flux_pull"] = (measured_flux - expected) / float(result["flux_error_adu"])
        rows.append(row)

    return rows


def wilson_interval(successes, total):
    """
    Calculate a 68% Wilson confidence interval for binomial proportion.

    Unlike the standard binomial error estimate, this gives meaningful bounds even when the recovery rate is 0 or 1.
    """

    probability = successes / total
    centre = (probability + 0.5 / total) / (1 + 1 / total)
    half = np.sqrt(probability * (1 - probability) / total + 0.25 / total ** 2) / (1 + 1 / total)

    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(records):
    rows = []
    for profile in PROFILES:
        for r_eff in RADII:
            for low in np.unique(records["mag_low"]):
                cell = records[(records["profile"] == profile) & (records["r_eff_pix"] == r_eff)
                               & (records["mag_low"] == low)]
                # Subsets are diagnostics; the all-field denominator includes
                # every injection, including pre-existing-source coincidences.
                for subset in ("all", "north", "south", "near_neighbour", "isolated"):
                    if subset in ("north", "south"):
                        group = cell[cell["region"] == subset]
                    elif subset == "near_neighbour":
                        group = cell[cell["near_neighbour_15pix"]]
                    elif subset == "isolated":
                        group = cell[~cell["near_neighbour_15pix"]]
                    else:
                        group = cell
                    total = len(group)
                    if total == 0:
                        continue

                    row = {"profile": profile, "r_eff_pix": r_eff, "mag_low": float(low),
                           "mag_high": float(low + 0.5), "mag_mid": float(low + 0.25),
                           "subset": subset, "n_injected": total,
                           "n_preexisting_confusion": int(group["preexisting_confusion"].sum())}
                    for column in ("unique_position_match", "detected", "usable_recovered", "counts_recovered"):
                        count = int(group[column].sum())
                        lower, upper = wilson_interval(count, total)
                        row[column + "_n"] = count
                        row[column + "_fraction"] = count / total
                        row[column + "_lower"] = lower
                        row[column + "_upper"] = upper

                    good = group["usable_recovered"]
                    row["n_bias_measurements"] = int(good.sum())
                    for column in ("delta_mag_total", "delta_mag_aperture", "aperture_fraction", "flux_pull"):
                        values = np.asarray(group[column][good], float)
                        values = values[np.isfinite(values)]
                        percentiles = np.percentile(values, [16, 50, 84]) if len(values) else [np.nan] * 3
                        for suffix, value in zip(("p16", "median", "p84"), percentiles):
                            row[column + "_" + suffix] = float(value)
                    rows.append(row)

    return Table(rows=rows)


def make_plots(records, summary, original, baseline_mask, output):
    all_field = summary[summary["subset"] == "all"]
    colours = ("#4077B6", "#8e3043", "#e99636")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True, layout="constrained")
    for i, profile in enumerate(PROFILES):
        for j, r_eff in enumerate(RADII):
            ax = axes[i, j]
            cell = all_field[(all_field["profile"] == profile) & (all_field["r_eff_pix"] == r_eff)]
            for column, label, colour in zip(
                ("detected", "usable_recovered", "counts_recovered"),
                ("New detection", "Usable photometry", "Retained after star selection"), colours,
            ):
                p = np.asarray(cell[column + "_fraction"])
                errors = np.array([p - cell[column + "_lower"], cell[column + "_upper"] - p])
                # Different line styles keep both stages visible when they agree.
                is_counts = column == "counts_recovered"
                ax.errorbar(
                    cell["mag_mid"], p, yerr=np.maximum(errors, 0),
                    linestyle="--" if is_counts else "-", marker="s" if is_counts else "o",
                    markerfacecolor="none" if is_counts else colour,
                    linewidth=1.2 if is_counts else 2, ms=4, capsize=2,
                    color=colour, label=label,
                )
            ax.set_title(f"{profile.replace('_', ' ')}; $r_e={r_eff:g}$ px")
            ax.set_ylim(-0.03, 1.03)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("Magnitude")
    fig.supylabel("Recovery Rate")
    fig.savefig(output / "completeness.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, layout="constrained")
    for i, profile in enumerate(PROFILES):
        for j, r_eff in enumerate(RADII):
            ax = axes[i, j]
            cell = all_field[(all_field["profile"] == profile) & (all_field["r_eff_pix"] == r_eff)]
            enough = np.asarray(cell["n_bias_measurements"]) >= 5
            for column, label, colour in [
                ("delta_mag_total", "Measured aperture minus injected total", "#1f22a4"),
                ("delta_mag_aperture", "Measured minus true aperture magnitude", "#da4242"),
            ]:
                median = np.where(enough, cell[column + "_median"], np.nan)
                lower = np.where(enough, cell[column + "_p16"], np.nan)
                upper = np.where(enough, cell[column + "_p84"], np.nan)
                ax.plot(cell["mag_mid"], median, "o-", ms=3, color=colour, label=label)
                ax.fill_between(cell["mag_mid"], lower, upper, color=colour, alpha=0.13)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{profile.replace('_', ' ')}; $r_e={r_eff:g}$ px")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Photometric bias: bins with at least five usable recoveries", fontsize=12)
    fig.supxlabel("Injected finite-template total magnitude")
    fig.supylabel("Magnitude difference; median and 16–84% spread of usable recoveries")
    fig.savefig(output / "photometry_bias.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 9), layout="constrained")
    valid_values = original[~baseline_mask]
    low, high = np.percentile(valid_values, [10, 99])
    ax.imshow(original, origin="lower", cmap="gray", vmin=low, vmax=high)
    ax.imshow(np.ma.masked_where(~baseline_mask, baseline_mask), origin="lower", cmap="autumn", alpha=0.35)
    for profile, colour in zip(PROFILES, ("cyan", "magenta")):
        selected = records["profile"] == profile
        ax.scatter(records["x"][selected], records["y"][selected], s=2, alpha=0.4, color=colour,
                   label=profile.replace("_", " "))
    ax.set(xlabel="x / pixel", ylabel="y / pixel", title="Injection centres across all separate batches")
    ax.legend(fontsize=8)
    fig.savefig(output / "injection_positions.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-bin", type=int, default=INJECTIONS_PER_CELL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.per_bin < 2 or args.batch_size < 1:
        raise ValueError("Need at least two injections per cell and a positive batch size.")

    started = time.monotonic()
    output = OUTPUT / "quick" if args.quick else OUTPUT
    edges = np.array([19.0, 19.5, 23.0, 23.5]) if args.quick else MAG_EDGES
    lows = (19.0, 23.0) if args.quick else edges[:-1]
    per_bin = 2 if args.quick else args.per_bin
    source_paths = [*CODE_PATHS, IMAGE_PATH, CATALOGUE_PATH, PSF_PATH]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    original, header = fits.getdata(IMAGE_PATH, header=True)
    original = original.astype(np.float32)
    psf = Table.read(PSF_PATH)
    alpha, beta = float(psf["alpha"][0]), float(psf["beta"][0])
    rng = np.random.default_rng(SEED)

    print("Rendering normalized, pixel-integrated templates...", flush=True)
    library = template_library(alpha, beta)
    tasks = [(profile, r_eff, float(low)) for profile in PROFILES for r_eff in RADII
             for low in lows for _ in range(per_bin)]
    rng.shuffle(tasks)
    records = []

    with tempfile.TemporaryDirectory(prefix="simple-injection-") as scratch:
        pipeline = FixedPipeline(scratch)
        print("Checking the unchanged full-field baseline...", flush=True)
        baseline, products = pipeline.detect(original, header)
        pipeline.check_baseline(baseline)
        baseline_mask = products["DETMASK"] | ~np.isfinite(original) | (original == 0)
        eligible = np.flatnonzero(~baseline_mask)
        baseline_tree = cKDTree(np.column_stack([baseline["x"], baseline["y"]]))
        area = len(eligible) * abs(header["PIXSCAL1"] * header["PIXSCAL2"]) / 3600 ** 2
        batches = int(np.ceil(len(tasks) / args.batch_size))

        for batch_id, start in enumerate(range(0, len(tasks), args.batch_size), 1):
            placed = place_batch(tasks[start:start + args.batch_size], rng, eligible, original.shape, library)
            injected = inject_batch(original, placed, library, header, rng)
            catalogue, products = pipeline.detect(injected, header)
            info, matched_rows = positional_matches(placed, catalogue, baseline_tree)
            measured = pipeline.photometry(injected, header, catalogue[matched_rows], products)
            records.extend(recovery_records(
                placed, info, catalogue, measured, library, pipeline, original.shape, batch_id,
            ))
            print(f"Batch {batch_id}/{batches}: {len(records)}/{len(tasks)} injections; "
                  f"elapsed {time.monotonic() - started:.0f} s", flush=True)

        settings = asdict(pipeline.settings)
        baseline_count = len(baseline)
        baseline_usable = int(baseline["usable"].sum())
        reference_count = len(pipeline.references)

    # Verify that using the original pipeline never changed its code or inputs.
    for path in source_paths:
        if sha256(path) != source_hashes[str(path.relative_to(ROOT))]:
            raise RuntimeError(f"Input changed during the experiment: {path}")

    records = Table(rows=records)
    records.add_column(np.arange(1, len(records) + 1), name="injection_id", index=0)
    summary = summarise(records)
    assert len(records) == len(tasks)
    assert np.all(~records["counts_recovered"] | records["usable_recovered"])
    assert np.all(~records["usable_recovered"] | records["detected"])
    assert np.all(~records["detected"] | records["unique_position_match"])
    assert np.all(summary[summary["subset"] == "all"]["n_injected"] == per_bin)

    output.mkdir(parents=True, exist_ok=True)
    records.meta["definitions"] = "See README.md. All injections remain in the denominator. No automatic completeness correction."
    records.write(output / "recovery.ecsv", overwrite=True)
    summary.write(output / "summary.csv", format="ascii.csv", overwrite=True)
    make_plots(records, summary, original, baseline_mask, output)
    manifest = {
        "quick_check": args.quick, "n_injected": len(records), "n_batches": batches,
        "per_cell": per_bin, "batch_size": args.batch_size, "seed": SEED,
        "profiles": PROFILES, "radii_pix": RADII, "mag_lows": list(map(float, lows)), "mag_step": 0.5,
        "axis_ratio": AXIS_RATIO, "angle_radians": ANGLE, "oversample": OVERSAMPLE,
        "intrinsic_enclosed_fraction_before_normalization": ENCLOSED_FRACTION,
        "psf_alpha": alpha, "psf_beta": beta, "psf_radius_pix": PSF_RADIUS,
        "subpixel_phases": PHASES, "source_poisson_noise": True,
        "match_radius_pix": MATCH_RADIUS, "diagnostic_neighbour_radius_pix": NEIGHBOUR_RADIUS,
        "eligible_area_deg2": float(area), "baseline_candidates": baseline_count,
        "baseline_usable": baseline_usable, "frozen_psf_references": reference_count,
        "detector_settings": settings, "input_sha256": source_hashes,
        "experiment_code_sha256": {name: sha256(HERE / name) for name in
                                   ("run_injection_recovery.py", "detector_adapter.py")},
        "validation": "Original baseline and star classes reproduced; inputs unchanged; nested selections and cell sizes checked.",
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "Finite normalized Sersic templates; fixed q=0.7 and angle=30 degrees.",
            "One saved Moffat PSF across the field; spatial PSF variation not simulated.",
            "Classification is conditional on the original frozen reference stars.",
            "Baseline coincidences cannot be attributed reliably and count as unrecovered.",
            "Wilson intervals describe binomial sampling, not all systematic uncertainties.",
            "Input total magnitude and recovered aperture magnitude are different quantities.",
            "Do not divide the observed counts by this table without modelling size/profile mixture and magnitude migration.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Finished. Six compact output files: {output}", flush=True)


if __name__ == "__main__":
    main()
