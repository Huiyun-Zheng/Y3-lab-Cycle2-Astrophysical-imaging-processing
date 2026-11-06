"""Automatic source detection for the laboratory mosaic.

Run with the astro-image interpreter:
    python source_detection_auto.py

No coordinates or interactive decisions are required. Results are written to
output/source_detection_auto. The existing source detection.py is not edited.

Pipeline: background -> saturation-connected artefacts -> refined background
-> connected sources -> peak separation -> brightest-first catalogue.

Masks follow measured image structure, with a small configurable safety margin.
They are heuristic, not a guarantee that every faint halo or artefact is removed.
Sources are NOT classified as stars or galaxies. Segment flux is not total flux.
Only NumPy, SciPy, Astropy and Matplotlib are required.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import heapq
import json

import numpy as np
from scipy import ndimage as ndi
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table


@dataclass
class Settings:
    """Adjust these parameters here, rather than changing the detection loop."""

    background_box: int = 128
    saturation_override: float | None = None
    artefact_sigma: float = 3.0
    halo_sigma: float = 1.5
    artefact_padding: int = 4
    boundary_padding: int = 4
    smooth_sigma: float = 1.0
    detection_sigma: float = 5.0
    growth_sigma: float = 2.0
    min_pixels: int = 8
    peak_separation: int = 8
    peak_prominence_sigma: float = 3.0
    peak_prominence_fraction: float = 0.05
    min_aperture_radius: float = 5.0
    max_aperture_radius: float = 35.0

    def __post_init__(self):
        """Reject settings that would silently change morphology semantics."""
        positive = [
            self.background_box, self.artefact_sigma, self.halo_sigma,
            self.artefact_padding, self.boundary_padding, self.smooth_sigma,
            self.detection_sigma, self.growth_sigma, self.min_pixels,
            self.peak_separation, self.peak_prominence_sigma,
            self.min_aperture_radius, self.max_aperture_radius,
        ]

        if any(value <= 0 for value in positive):
            raise ValueError("Scale, size and threshold settings must be positive.")

        if self.growth_sigma >= self.detection_sigma:
            raise ValueError("growth_sigma must be below detection_sigma.")

        if self.max_aperture_radius < self.min_aperture_radius:
            raise ValueError("Maximum aperture radius cannot be below the minimum.")

        if not 0 <= self.peak_prominence_fraction < 1:
            raise ValueError("Peak prominence fraction must be between zero and one.")


CONNECTED = np.ones((3, 3), dtype=bool)


def clipped_stats(values):
    """Return a robust background level and pixel-to-pixel scatter."""
    values = values[np.isfinite(values)]

    if values.size < 20:
        raise ValueError("Too few valid pixels to estimate the background.")

    _, median, scatter = sigma_clipped_stats(values, sigma=3, maxiters=8)

    if not np.isfinite(scatter) or scatter <= 0:
        raise ValueError("Background scatter is zero or invalid.")

    return float(median), float(scatter)


def find_missing_data(image):
    """Exclude NaN/Inf and constant padding connected to the image boundary."""
    invalid = ~np.isfinite(image)
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    border = border[np.isfinite(border)]

    if border.size == 0:
        return invalid

    values, counts = np.unique(border, return_counts=True)
    common = int(np.argmax(counts))

    # A repeated boundary value is evidence of padding, not of a sky pixel.
    # Do not exclude other pixels of the same value unless they touch padding.
    if counts[common] / border.size < 0.2:
        return invalid

    possible_padding = image == values[common]
    seeds = np.zeros(image.shape, dtype=bool)
    seeds[[0, -1], :] = possible_padding[[0, -1], :]
    seeds[:, [0, -1]] = possible_padding[:, [0, -1]]
    padding = ndi.binary_propagation(seeds, mask=possible_padding)

    return invalid | padding


def background_maps(image, excluded, box_size):
    """Measure clipped statistics in tiles, then interpolate between tiles.

    Entirely masked tiles use their nearest valid tile. The noise map describes
    background scatter, not the uncertainty of the fitted background itself.
    """
    height, width = image.shape
    rows = (height + box_size - 1) // box_size
    columns = (width + box_size - 1) // box_size
    levels = np.full((rows, columns), np.nan)
    scatters = np.full((rows, columns), np.nan)

    for row in range(rows):
        for column in range(columns):
            ys = slice(row * box_size, min((row + 1) * box_size, height))
            xs = slice(column * box_size, min((column + 1) * box_size, width))
            tile = image[ys, xs]
            usable = ~excluded[ys, xs] & np.isfinite(tile)

            if np.count_nonzero(usable) < max(40, 0.25 * tile.size):
                continue

            try:
                levels[row, column], scatters[row, column] = clipped_stats(tile[usable])
            except ValueError:
                continue

    missing = ~np.isfinite(levels)

    if np.all(missing):
        raise ValueError("No background tiles remain; inspect the mask/settings.")

    nearest = ndi.distance_transform_edt(missing, return_distances=False, return_indices=True)
    levels[missing] = levels[tuple(nearest[:, missing])]
    scatters[missing] = scatters[tuple(nearest[:, missing])]
    levels = ndi.median_filter(levels, size=3, mode="nearest")
    scatters = ndi.median_filter(scatters, size=3, mode="nearest")

    # Interpolate at actual tile centres, including the shorter final tiles.
    centres_y = (
        np.arange(rows) * box_size
        + np.minimum((np.arange(rows) + 1) * box_size, height) - 1
    ) / 2
    centres_x = (
        np.arange(columns) * box_size
        + np.minimum((np.arange(columns) + 1) * box_size, width) - 1
    ) / 2
    y_grid = np.interp(np.arange(height), centres_y, np.arange(rows))
    x_grid = np.interp(np.arange(width), centres_x, np.arange(columns))
    yy, xx = np.meshgrid(y_grid, x_grid, indexing="ij")
    background = ndi.map_coordinates(levels, [yy, xx], order=1, mode="nearest")
    noise = ndi.map_coordinates(scatters, [yy, xx], order=1, mode="nearest")

    return background.astype(np.float32), noise.astype(np.float32)


def identify_artefacts(image, missing, saturation, settings):
    """Grow saturation seeds through connected bright trails and stellar halos.

    A narrow trail need not itself reach the saturation threshold: it only needs
    to be connected to a seed through significant signal. This follows curved
    and diagonal structures too, without fixed rows, columns or rectangles.
    """
    sample = image[::3, ::3][~missing[::3, ::3]]
    level, scatter = clipped_stats(sample)
    safe_image = np.where(missing, level, image)
    saturated = (safe_image >= saturation) & ~missing

    # Global statistics deliberately keep extended stellar halos visible here;
    # fitting a local background first could absorb a halo into the background.
    soft_image = ndi.gaussian_filter(safe_image, sigma=1.2)
    bright_pixels = safe_image > level + settings.artefact_sigma * scatter
    halo_pixels = soft_image > level + settings.halo_sigma * scatter
    allowed = (bright_pixels | halo_pixels | saturated) & ~missing
    allowed = ndi.binary_closing(allowed, structure=CONNECTED) | saturated
    allowed &= ~missing

    artefact = ndi.binary_propagation(saturated, structure=CONNECTED, mask=allowed)
    artefact = ndi.binary_fill_holes(artefact) & ~missing
    artefact = ndi.binary_dilation(artefact, iterations=settings.artefact_padding)
    artefact &= ~missing

    return saturated, artefact, level, scatter


def select_significant_peaks(surface, region, positions, settings):
    """Keep peaks separated from stronger peaks by a sufficiently deep valley.

    The surface is in units of local smoothed-background scatter. A peak must
    rise above its connecting saddle by at least 3 sigma (by default) and 5%
    of ITS OWN height. It need not reach 5% of a much brighter neighbour.
    """
    positions = sorted(positions, key=lambda p: surface[p], reverse=True)
    accepted = []

    for position in positions:
        if not accepted:
            accepted.append(position)
            continue

        required_drop = max(
            settings.peak_prominence_sigma,
            settings.peak_prominence_fraction * surface[position],
        )
        high_ground = region & (surface > surface[position] - required_drop)
        seed = np.zeros(region.shape, dtype=bool)
        seed[position] = True
        connected = ndi.binary_propagation(seed, structure=CONNECTED, mask=high_ground)

        # A stronger peak reachable without descending far enough means that
        # this candidate is only a small bump on the same source's profile.
        if not any(connected[other] for other in accepted):
            accepted.append(position)

    return accepted


def grow_peak_regions(surface, region, positions):
    """Flood the inverted S/N surface from peaks, staying inside the footprint.

    heapq is Python's built-in priority queue. The lowest flood level is visited
    first; entry order breaks ties. Peak seeds stay assigned to their own source.
    There are no background seeds that could steal pixels from real sources.
    """
    children = np.zeros(region.shape, dtype=np.int32)
    queue = []
    height, width = region.shape
    arrival = 0

    for number, (y, x) in enumerate(positions, start=1):
        children[y, x] = number
        heapq.heappush(queue, (-float(surface[y, x]), arrival, int(y), int(x)))
        arrival += 1

    neighbours = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]

    while queue:
        level, _, y, x = heapq.heappop(queue)

        for dy, dx in neighbours:
            yy, xx = y + dy, x + dx

            if not (0 <= yy < height and 0 <= xx < width):
                continue

            if not region[yy, xx] or children[yy, xx] != 0:
                continue

            # A path cannot flood a pixel before crossing its highest barrier.
            next_level = max(level, -float(surface[yy, xx]))
            children[yy, xx] = children[y, x]
            heapq.heappush(queue, (next_level, arrival, yy, xx))
            arrival += 1

    return children


def separate_sources(smoothed, detection_snr, excluded, settings):
    """Find connected footprints, then divide them between significant peaks.

    This is a practical deblender, not a PSF fit: all split groups are flagged.
    Keep the smoothed argument for compatibility with existing analysis cells;
    both peak selection and region growth now use the same local S/N surface.
    """
    footprint = (detection_snr >= settings.growth_sigma) & ~excluded
    parents, _ = ndi.label(footprint, structure=CONNECTED)
    sizes = np.bincount(parents.ravel())

    # Masked bright pixels must not suppress nearby valid maxima.
    searchable = np.where(excluded, -np.inf, detection_snr)
    maxima = ndi.maximum_filter(searchable, size=settings.peak_separation + 1)
    peaks = (detection_snr == maxima) & (detection_snr >= settings.detection_sigma)
    peaks &= footprint
    segments = np.zeros(smoothed.shape, dtype=np.int32)
    metadata = {}
    next_id = 1

    for parent_id, bounds in enumerate(ndi.find_objects(parents), start=1):
        if bounds is None or sizes[parent_id] < settings.min_pixels:
            continue

        region = parents[bounds] == parent_id
        peak_groups, n_groups = ndi.label(peaks[bounds] & region, structure=CONNECTED)

        if n_groups == 0:
            continue

        # Collapse equal-valued peak plateaus to one seed each.
        positions = ndi.maximum_position(
            detection_snr[bounds], peak_groups, np.arange(1, n_groups + 1)
        )
        positions = select_significant_peaks(
            detection_snr[bounds], region, positions, settings
        )

        if len(positions) == 1:
            children = region.astype(np.int32)
        else:
            children = grow_peak_regions(detection_snr[bounds], region, positions)

        for number in range(1, len(positions) + 1):
            child = children == number

            if np.count_nonzero(child) < settings.min_pixels:
                continue

            local_labels = segments[bounds]
            local_labels[child] = next_id
            metadata[next_id] = (parent_id, len(positions) > 1)
            next_id += 1

    return segments, metadata


CATALOGUE_NAMES = [
    "id", "segment_id", "parent_id", "x", "y", "peak_x", "peak_y",
    "peak_adu", "peak_detection_snr", "n_pixels", "segment_flux_adu",
    "aperture_radius_pix", "aperture_flux_adu", "major_sigma_pix",
    "minor_sigma_pix", "elongation", "near_artefact", "near_boundary",
    "blended", "aperture_contaminated", "aperture_capped", "usable", "flags",
]
CATALOGUE_TYPES = [
    int, int, int, float, float, int, int, float, float, int, float,
    float, float, float, float, float, bool, bool, bool, bool, bool, bool, "U180",
]


def measure_sources(image, residual, detection_snr, segments, metadata, artefact, boundary, settings):
    """Record sources brightest first, without altering the original image.

    Segment flux misses sub-threshold wings. Adaptive circular aperture flux is
    also preliminary; contaminated apertures get NaN, never a partial sum.
    No Poisson errors are inferred from the unverified mosaic GAIN header.
    """
    ids = np.arange(1, segments.max() + 1)
    peak_values = ndi.maximum(image, labels=segments, index=ids)
    order = ids[np.argsort(peak_values)[::-1]]
    bounds_list = ndi.find_objects(segments)
    quality_margin = int(settings.min_aperture_radius) + 3
    near_artefact = ndi.binary_dilation(artefact, iterations=quality_margin)
    near_boundary = ndi.binary_dilation(boundary, iterations=quality_margin)
    aperture_bad = artefact | boundary
    processed_mask = np.zeros(image.shape, dtype=bool)
    rows = []
    height, width = image.shape

    for source_id in order:
        bounds = bounds_list[source_id - 1]
        region = segments[bounds] == source_id
        local_y, local_x = np.nonzero(region)
        yy = local_y + bounds[0].start
        xx = local_x + bounds[1].start
        signal = residual[bounds][region]
        weights = np.maximum(signal, 0)

        if weights.sum() <= 0:
            continue

        x = float(np.average(xx, weights=weights))
        y = float(np.average(yy, weights=weights))
        dx, dy = xx - x, yy - y
        covariance = np.array([
            [np.average(dx * dx, weights=weights), np.average(dx * dy, weights=weights)],
            [np.average(dx * dy, weights=weights), np.average(dy * dy, weights=weights)],
        ])
        minor, major = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0))
        elongation = major / max(minor, 0.25)
        brightest = int(np.argmax(image[bounds][region]))
        peak_x, peak_y = int(xx[brightest]), int(yy[brightest])

        # An adaptive circle is easier to inspect than a fixed radius for all sizes.
        requested_radius = max(settings.min_aperture_radius, 2.5 * major)
        radius = min(requested_radius, settings.max_aperture_radius)
        capped = requested_radius > settings.max_aperture_radius
        x0 = max(0, int(np.floor(x - radius)))
        x1 = min(width, int(np.ceil(x + radius)) + 1)
        y0 = max(0, int(np.floor(y - radius)))
        y1 = min(height, int(np.ceil(y + radius)) + 1)
        ay, ax = np.mgrid[y0:y1, x0:x1]
        aperture = (ax - x) ** 2 + (ay - y) ** 2 <= radius ** 2
        aperture_labels = segments[y0:y1, x0:x1]
        neighbours = (aperture_labels > 0) & (aperture_labels != source_id)
        cuts_edge = (
            x - radius < 0 or x + radius >= width
            or y - radius < 0 or y + radius >= height
        )
        contaminated = cuts_edge or bool(np.any(aperture & aperture_bad[y0:y1, x0:x1]))
        contaminated |= bool(np.any(aperture & neighbours))
        aperture_flux = (
            np.nan if contaminated
            else float(residual[y0:y1, x0:x1][aperture].sum())
        )

        touches_artefact = bool(np.any(near_artefact[bounds][region]))
        touches_boundary = bool(np.any(near_boundary[bounds][region]))
        parent_id, blended = metadata[source_id]
        flags = []

        for condition, name in [
            (touches_artefact, "near_artefact"), (touches_boundary, "near_boundary"),
            (blended, "blended"), (contaminated, "aperture_contaminated"),
            (capped, "aperture_capped"), (elongation > 6, "very_elongated"),
        ]:
            if condition:
                flags.append(name)

        rows.append((
            len(rows) + 1, source_id, parent_id, x, y, peak_x, peak_y,
            float(image[peak_y, peak_x]), float(detection_snr[bounds][region].max()),
            int(region.sum()), float(signal.sum()), radius, aperture_flux,
            major, minor, elongation, touches_artefact, touches_boundary,
            blended, contaminated, capped, not flags, "|".join(flags) or "none",
        ))
        processed_mask[bounds] |= region

    return Table(rows=rows, names=CATALOGUE_NAMES, dtype=CATALOGUE_TYPES), processed_mask


def save_diagnostics(image, background, noise, artefact, boundary, saturated, catalogue, output):
    """Save overview and automatically selected close-ups; never prompt or block."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr = (image - background) / noise
    masked = np.ma.masked_where(~artefact, np.ones(image.shape, dtype=np.uint8))

    def draw(ax, limits=None):
        ax.imshow(snr, origin="lower", cmap="gray", vmin=-1, vmax=12, interpolation="nearest")
        ax.imshow(masked, origin="lower", cmap="autumn", vmin=0, vmax=1, alpha=0.45, interpolation="nearest")
        ax.set_xlabel("x / pixel")
        ax.set_ylabel("y / pixel")

        if limits is not None:
            ax.set_xlim(limits[:2])
            ax.set_ylim(limits[2:])

    fig, axes = plt.subplots(1, 2, figsize=(13, 11))
    draw(axes[0])
    padding = np.ma.masked_where(~boundary, np.ones(image.shape, dtype=np.uint8))
    axes[0].imshow(padding, origin="lower", cmap="Blues", vmin=0, vmax=1, alpha=0.4)
    axes[0].set_title("Automatic mask: artefacts (gold), boundary/padding (blue)")
    draw(axes[1])

    for usable, colour, name in [(True, "cyan", "Usable candidates"), (False, "magenta", "Flagged candidates")]:
        subset = catalogue[catalogue["usable"] == usable]
        axes[1].scatter(subset["x"], subset["y"], s=9, facecolors="none", edgecolors=colour, linewidths=0.4, label=name)

    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title(f"{len(catalogue)} candidates; {np.count_nonzero(catalogue['usable'])} usable")
    fig.tight_layout()
    fig.savefig(output / "detection_overview.png", dpi=180)
    plt.close(fig)

    # Select the largest saturation-connected components for close-up inspection.
    groups, _ = ndi.label(artefact, structure=CONNECTED)
    areas = np.bincount(groups.ravel())
    selected = (np.argsort(areas[1:])[::-1] + 1)[:6]
    bounds = ndi.find_objects(groups)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, group_id in zip(axes.ravel(), selected):
        ys, xs = bounds[group_id - 1]
        limits = (
            max(0, xs.start - 30), min(image.shape[1], xs.stop + 30),
            max(0, ys.start - 30), min(image.shape[0], ys.stop + 30),
        )
        draw(ax, limits)
        ax.set_title(f"Artefact group {group_id}: {areas[group_id]} masked pixels")

    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output / "artefact_closeups.png", dpi=160)
    plt.close(fig)

    # A full-height trail is too tall to inspect in one small panel. Show its
    # lower and upper ends and broadest halo separately, selected from the mask.
    if len(selected):
        main_id = selected[0]
        ys, xs = bounds[main_id - 1]
        region = groups[ys, xs] == main_id
        widest_row = ys.start + int(np.argmax(np.count_nonzero(region, axis=1)))
        span = min(600, ys.stop - ys.start)
        centre_start = max(ys.start, widest_row - span // 2)
        views = [
            (ys.start, min(ys.stop, ys.start + span), "Lower end of dominant artefact"),
            (centre_start, min(image.shape[0], centre_start + span), "Broadest part / halo"),
            (max(ys.start, ys.stop - span), ys.stop, "Upper end of dominant artefact"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        for ax, (y0, y1, title) in zip(axes, views):
            draw(ax, (max(0, xs.start - 30), min(image.shape[1], xs.stop + 30), y0, y1))
            ax.set_title(title)

        fig.tight_layout()
        fig.savefig(output / "bleeding_detail.png", dpi=160)
        plt.close(fig)


def run_detection(fits_path, output_dir, settings=None):
    """Run the complete unattended pipeline and return its catalogue and masks."""
    settings = Settings() if settings is None else settings
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(fits_path) as hdul:
        header = hdul[0].header.copy()
        image = hdul[0].data.astype(np.float32)

    if image.ndim != 2:
        raise ValueError("The input must be a two-dimensional image.")

    saturation = settings.saturation_override

    if saturation is None:
        if "SATURATE" not in header:
            raise ValueError("Set saturation_override when the SATURATE header is absent.")
        saturation = float(header["SATURATE"])

    if not np.isfinite(saturation):
        raise ValueError("Saturation threshold must be finite.")

    missing = find_missing_data(image)
    print("1/5: Growing saturation-connected artefact masks...", flush=True)
    saturated, artefact, level, scatter = identify_artefacts(image, missing, saturation, settings)
    boundary = ndi.binary_dilation(missing, iterations=settings.boundary_padding)
    boundary[:settings.boundary_padding, :] = True
    boundary[-settings.boundary_padding:, :] = True
    boundary[:, :settings.boundary_padding] = True
    boundary[:, -settings.boundary_padding:] = True
    excluded = artefact | boundary
    safe_image = np.where(missing, level, image)

    print("2/5: Estimating local background and noise...", flush=True)
    background, noise = background_maps(safe_image, excluded, settings.background_box)
    residual = safe_image - background
    initial_sources = ndi.binary_dilation((residual > 3 * noise) & ~excluded, iterations=3)
    background, noise = background_maps(safe_image, excluded | initial_sources, settings.background_box)
    residual = safe_image - background

    # Smoothing helps faint, spatially extended signals. Estimate the scatter of
    # the smoothed image directly instead of assuming independent mosaic pixels.
    smoothed = ndi.gaussian_filter(np.where(excluded, 0, residual), settings.smooth_sigma)
    detection_excluded = ndi.binary_dilation(excluded, iterations=4)
    smooth_level, smooth_noise = background_maps(smoothed, detection_excluded | initial_sources, settings.background_box)
    detection_snr = (smoothed - smooth_level) / smooth_noise

    print("3/5: Detecting connected regions and separating peaks...", flush=True)
    segments, metadata = separate_sources(smoothed, detection_snr, detection_excluded, settings)

    # A negative-image control exposes noise peaks/background problems. It is
    # not a precise false-discovery rate, because artefacts need not be symmetric.
    negative_segments, _ = separate_sources(
        -smoothed, -detection_snr, detection_excluded, settings
    )
    negative_count = int(negative_segments.max())
    del negative_segments

    print("4/5: Measuring sources in descending peak brightness...", flush=True)
    catalogue, processed = measure_sources(
        image, residual, detection_snr, segments, metadata,
        artefact, boundary, settings,
    )
    catalogue.meta["description"] = "Unclassified source candidates; usable is a conservative photometry-quality flag."
    catalogue.meta["flux_note"] = "Background-subtracted ADU assumed; segment and aperture fluxes are not total fluxes."
    catalogue.write(output_dir / "sources.ecsv", overwrite=True)
    catalogue.write(output_dir / "sources.csv", format="ascii.csv", overwrite=True)
    catalogue[catalogue["usable"]].write(
        output_dir / "usable_sources.csv", format="ascii.csv", overwrite=True
    )

    # Keep an audit trail for excluded regions, including smaller bleeding stars.
    # One connected region may contain several stars or a star plus its trail.
    artefact_groups, _ = ndi.label(artefact, structure=CONNECTED)
    artefact_rows = []

    for group_id, bounds in enumerate(ndi.find_objects(artefact_groups), start=1):
        region = artefact_groups[bounds] == group_id
        ys, xs = bounds
        artefact_rows.append((
            group_id, xs.start, xs.stop, ys.start, ys.stop,
            int(region.sum()), int(np.count_nonzero(saturated[bounds] & region)),
        ))

    Table(
        rows=artefact_rows,
        names=["region_id", "x_start", "x_stop", "y_start", "y_stop", "masked_pixels", "saturated_pixels"],
        dtype=[int] * 7,
    ).write(output_dir / "artefact_regions.csv", format="ascii.csv", overwrite=True)
    del artefact_groups

    # Image products retain pixel alignment, but no unverified celestial WCS.
    products = [
        ("BADMASK", excluded.astype(np.uint8)), ("ARTEFACT", artefact.astype(np.uint8)),
        ("SATURATED", saturated.astype(np.uint8)), ("BOUNDARY", boundary.astype(np.uint8)),
        ("DETMASK", detection_excluded.astype(np.uint8)), ("SOURCES", segments),
        ("PROCESSED", processed.astype(np.uint8)), ("BACKGROUND", background), ("RMS", noise),
    ]
    primary = fits.PrimaryHDU()
    primary.header["ORIGFILE"] = Path(fits_path).name
    primary.header["SATLEVEL"] = saturation
    extensions = [fits.ImageHDU(values, name=name) for name, values in products]
    fits.HDUList([primary] + extensions).writeto(
        output_dir / "detection_products.fits", overwrite=True
    )

    summary = {
        "input": str(Path(fits_path).resolve()), "settings": asdict(settings),
        "saturation_used": saturation, "initial_background": level, "initial_noise": scatter,
        "candidates": len(catalogue), "usable_candidates": int(np.count_nonzero(catalogue["usable"])),
        "saturated_pixels": int(saturated.sum()), "artefact_pixels": int(artefact.sum()),
        "artefact_regions": len(artefact_rows),
        "excluded_fraction": float(excluded.mean()),
        "negative_image_candidates": negative_count,
        "limitations": [
            "SATURATE is inherited metadata: verify the threshold for this mosaic.",
            "Connected masks can absorb neighbours and miss disconnected or faint artefacts.",
            "An artefact without a saturation seed is not automatically identified by this method.",
            "Peaks and watershed regions are candidates, not a complete galaxy catalogue.",
            "Fluxes omit wings; GAIN, completeness and photometric uncertainty are not calibrated.",
        ],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("5/5: Saving diagnostic figures...", flush=True)
    save_diagnostics(image, background, noise, artefact, boundary, saturated, catalogue, output_dir)
    print(f"Finished: {summary['candidates']} candidates, {summary['usable_candidates']} usable.")
    print(f"Negative-image control: {negative_count} candidates (not a calibrated false-positive count).")
    print(f"Outputs: {output_dir.resolve()}")

    return catalogue, excluded, processed


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, default=base_dir / "Astro" / "Fits_Data" / "mosaic.fits")
    parser.add_argument("--output", type=Path, default=base_dir / "output" / "source_detection_auto")
    args = parser.parse_args()
    run_detection(args.fits, args.output)
