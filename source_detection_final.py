"""Run with the astro-image interpreter: python source_detection_auto.py

Results are written to output/source_detection_auto.

Pipeline: background -> saturation-connected artefacts -> refined background
-> connected sources -> peak separation -> brightest-first catalogue.

Sources are NOT classified as stars or galaxies.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import heapq  # Use Python's built-in minimum-priority queue for region growth.
import json

import numpy as np
from scipy import ndimage as ndi
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table


@dataclass
class Settings:  # Group all configurable detection parameters in one object.
    """Adjust these parameters here, rather than changing the detection loop."""

    # Estimate the background in image blocks up to 128 * 128 pixels.
    background_box: int = 128

    # Set the pixel value that counts as saturated.
    # None means use the SATURATE value stored in the FITS file.
    saturation_override: float | None = None

    # When growing an artefact region, include connected raw pixels that are
    # more than 3 times the global noise above the background.
    artefact_sigma: float = 3.0

    # Also include fainter halo pixels around the artefact
    # Check the smoothed image, using 1.5 times the raw image's global noise.
    halo_sigma: float = 1.5

    # Expand the excluded artefact area by 4 steps to leave a safety margin.
    artefact_padding: int = 4

    # Expand missing-data areas by 4 steps.
    # Also ignore the outermost 4 rows and columns of the image.
    boundary_padding: int = 4

    # Lightly blur the image before detection to reduce small noise fluctuations.
    # This is the Gaussian blur width in pixels; larger values mean more blur.
    smooth_sigma: float = 1.0

    # A peak must reach 5 times the local noise level to start a detection.
    # The noise is meadured after smoothing.
    detection_sigma: float = 5.0

    # Once a peak is found, grow its region into surrounding pixels
    # down to 2 times the local noise level measured after smoothing.
    growth_sigma: float = 2.0

    # Keep only regions with at least 8 pixels.
    # This also applies to smaller regions created by splitting a larger one.
    min_pixels: int = 8

    # Look for local peaks using a 9 * 9 pixel window.
    # N.B. This does not enforce an exact 8-pixel distance between peaks.
    peak_separation: int = 8

    # To count as a separate source, a secondary peak must rise at least
    # 3 signal-to-noise units above the dip connecting the peaks.
    peak_prominence_sigma: float = 3.0

    # That rise must also be at least 5% of the secondary peak's own height
    peak_prominence_fraction: float = 0.05

    # Use a circle with a radius of at least 5 pixels for initial brightness measurements.
    min_aperture_radius: float = 5.0

    # Limit that radius to 35 pixels.
    # Sources which need a larger circle will be flagged.
    max_aperture_radius: float = 35.0

    # Use an ellipse when the measured elongation reaches this value
    ellipse_threshold: float = 1.8

    def __post_init__(self):

        # Collect the scale and threshold parameters that must be strictly positive.
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

        # Require the fractional prominence threshold to lie in [0, 1).
        if not 0 <= self.peak_prominence_fraction < 1:
            raise ValueError("Peak prominence fraction must be between zero and one.")

# Count all 8 surrounding pixels as neighbours, including diagonal ones.
CONNECTED = np.ones((3, 3), dtype=bool)


def clipped_stats(values):
    """Estimate the background brightness and noise."""

    # values can either be the whole image or a region in the image.
    values = values[np.isfinite(values)]

    if values.size < 20:
        raise ValueError("Too few valid pixels to estimate the background.")

    # Repeatedly exclude values more than 3 standard deviations from the centre.
    # Stop when no more values are excluded, or after 8 rounds.
    # Use the mean as the background and the standard deviation as the noise estimate.
    _, median, scatter = sigma_clipped_stats(values, sigma=3, maxiters=8)

    if not np.isfinite(scatter) or scatter <= 0:
        raise ValueError("Background scatter is zero or invalid.")

    return float(median), float(scatter)


def find_missing_data(image):
    """Define a mask for invalid values and likely consatant edge padding."""

    invalid = ~np.isfinite(image)
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    border = border[np.isfinite(border)]  # Ignore non-finite boundary values when identifying a common padding value.

    if border.size == 0:
        return invalid

    # Find the distinct finite boundary values and their occurrence counts.
    values, counts = np.unique(border, return_counts=True)
    # Get the index of the most common edge value.
    common = int(np.argmax(counts))

    # Treat this valye as a possible fill value only if it accounts for at least 20% of the finite edge samples.
    if counts[common] / border.size < 0.2:
        return invalid

    # Find all pixels exactly equal to the possible fill value.
    possible_padding = image == values[common]
    # Create an empty mask for the starting points.
    seeds = np.zeros(image.shape, dtype=bool)
    # Start from matching pixels on the top and bottom edges.
    seeds[[0, -1], :] = possible_padding[[0, -1], :]
    # Also start from matching pixels on the left and right edges.
    seeds[:, [0, -1]] = possible_padding[:, [0, -1]]
    # Spread from the edges through pixels with the same fill value.
    # By default, this moves up, down, left, and right, not diagonally.
    # Matching pixels that cannot be reached from an edge are left unmarked.
    padding = ndi.binary_propagation(seeds, mask=possible_padding)

    return invalid | padding


def background_maps(image, excluded, box_size):
    """
    Estimate the background brightness and noise across the image.

    Measure small blocks first, then fill gaps and interpolate between them.
    Return two arrays with the same shape as the image.
    The noise describes pixel fluctuations, not uncertainty in the background estimate.
    """

    height, width = image.shape

    # Count the blocks needed, including any smaller blocks at the edges.
    rows = (height + box_size - 1) // box_size
    columns = (width + box_size - 1) // box_size

    # Store one background value and one noise value per block.
    levels = np.full((rows, columns), np.nan)
    scatters = np.full((rows, columns), np.nan)

    for row in range(rows):
        for column in range(columns):
            # Find this block's row and column ranges.
            # Stop at the image edges if the final block is smaller.
            ys = slice(row * box_size, min((row + 1) * box_size, height))
            xs = slice(column * box_size, min((column + 1) * box_size, width))
            tile = image[ys, xs]

            usable = ~excluded[ys, xs] & np.isfinite(tile)

            # Require at least 40 usable pixels and at least 25% of the block.
            # Otherwise, leave this block's results as NaN.
            if np.count_nonzero(usable) < max(40, 0.25 * tile.size):
                continue

            try:
                # Estimate this block's background and noise.
                levels[row, column], scatters[row, column] = clipped_stats(tile[usable])
            except ValueError:
                continue

    missing = ~np.isfinite(levels)

    if np.all(missing):
        raise ValueError("No background tiles remain; inspect the mask/settings.")

    # Use the background and noise values from the nearest valid block for eaching missing block.
    nearest = ndi.distance_transform_edt(missing, return_distances=False, return_indices=True)
    levels[missing] = levels[tuple(nearest[:, missing])]
    scatters[missing] = scatters[tuple(nearest[:, missing])]
    levels = ndi.median_filter(levels, size=3, mode="nearest")
    scatters = ndi.median_filter(scatters, size=3, mode="nearest")

    # Find each block's centre in image coordinates.
    # Account for smaller blocks at the bottom and right edges.
    centres_y = (np.arange(rows) * box_size + np.minimum((np.arange(rows) + 1) * box_size, height) - 1) / 2
    centres_x = (np.arange(columns) * box_size + np.minimum((np.arange(columns) + 1) * box_size, width) - 1) / 2

    # Convert pixel positions to positions on the block grid.
    y_grid = np.interp(np.arange(height), centres_y, np.arange(rows))
    x_grid = np.interp(np.arange(width), centres_x, np.arange(columns))

    yy, xx = np.meshgrid(y_grid, x_grid, indexing="ij")

    # Interpolate between block values to estimate the background and noise at every pixel.
    background = ndi.map_coordinates(levels, [yy, xx], order=1, mode="nearest")
    noise = ndi.map_coordinates(scatters, [yy, xx], order=1, mode="nearest")

    return background.astype(np.float32), noise.astype(np.float32)


def identify_artefacts(image, missing, saturation, settings):
    """
    Mark saturated pixels and connected bright trails or halos.

    Start from saturated pixels and grow through nearby bright regions.
    Fill enclosed holes, then add a safety margin around the result.
    """

    sample = image[::3, ::3][~missing[::3, ::3]]
    level, scatter = clipped_stats(sample)
    safe_image = np.where(missing, level, image)
    saturated = (safe_image >= saturation) & ~missing

    # Lightly blur the image to make faint connected halos easier to follow.
    soft_image = ndi.gaussian_filter(safe_image, sigma=1.2)
    bright_pixels = safe_image > level + settings.artefact_sigma * scatter
    halo_pixels = soft_image > level + settings.halo_sigma * scatter

    # Allow growth through bright pixels, halo pixels, or saturated pixels.
    # Never allow growth through missing data.
    allowed = (bright_pixels | halo_pixels | saturated) & ~missing
    allowed = ndi.binary_closing(allowed, structure=CONNECTED) | saturated
    allowed &= ~missing

    # Start at saturated pixels and spread only through allowed regions.
    # Connections can run horizontally, vertically, or diagonally.
    artefact = ndi.binary_propagation(saturated, structure=CONNECTED, mask=allowed)

    # Fill enclosed holes in the mask, but leave missing pixels out.
    artefact = ndi.binary_fill_holes(artefact) & ~missing

    # Expand the mask to leave a safety margin.
    artefact = ndi.binary_dilation(artefact, iterations=settings.artefact_padding)  # Add a safety margin using the default cross-shaped dilation footprint.
    artefact &= ~missing

    return saturated, artefact, level, scatter


def select_significant_peaks(surface, region, positions, settings):
    """
    Keep peaks separated from stronger peaks by a sufficiently deep valley.

    Peak heights use the local smoothed noise as their scale.
    By default, the dip must be at least 3 units deep and at least
    5% of the candidate peak's own height.

    Inputs:
        surface: A 2D array of detection signal-to-noise values.
        region: A Boolean array marking the region being checked.
            True means inside the region. Must match surface.shape.
        positions: Candidate peak coordinates as (row, column) pairs,
            using the coordinates of the supplied arrays.
        settings: A Settings object containing the minimum required
            peak-to-dip drop and the required fraction of peak height.

    Returns:
        accepted: A list of retained (row, column) peak coordinates,
            ordered from highest to lowest surface value.
            The highest candidate is always kept.
    """

    # Check the highest candidate peaks first.
    positions = sorted(positions, key=lambda p: surface[p], reverse=True)
    accepted = []

    for position in positions:
        if not accepted:
            accepted.append(position)
            continue

        # Work out how far the height must drop to separate this peak
        # from an already accepted peak.
        required_drop = max(settings.peak_prominence_sigma, settings.peak_prominence_fraction * surface[position])
        high_ground = region & (surface > surface[position] - required_drop)

        seed = np.zeros(region.shape, dtype=bool)
        seed[position] = True

        # Find all high-ground pixels reachable from this peak.
        # Allow horizontal, vertical, and diagonal connections.
        connected = ndi.binary_propagation(seed, structure=CONNECTED, mask=high_ground)

        # Keep this peak only if no accepted peak can be reached
        # without dropping to the cut-off height or below.
        if not any(connected[other] for other in accepted):
            accepted.append(position)

    return accepted


def grow_peak_regions(surface, region, positions):
    """
    Split a region into labelled areas, starting from the supplied peaks
    whose coordinates are stored in 'positions'.

    Grow only inside the region. Use the negative surface values to control
    the processiong order (like flooding outward from valleys).
    Once a pixels gets a label, it keeps that label.

    Inputs:
        surface: A 2D array of detection signal-to-noise values,
            used to control the growth order.
        region: A Boolean array marking where sources may grow.
            True means growth is allowed. Must match surface.shape.
        positions: Distinct peak coordinates as (row, column) pairs.
            Each peak must lie inside region and uses the coordinates
            of the supplied arrays. Each peak starts a separate source.

    Returns:
        source_labels: An integer array with the same shape as region.
            Labels 1, 2, 3, ... correspond to the order of positions.
            Zero marks pixels outside the region or not reachable from any peak.
            No new peaks are created during growth.
    """

    children= np.zeros(region.shape, dtype=np.int32)
    queue = []
    height, width = region.shape
    arrival = 0

    for number, (y, x) in enumerate(positions, start=1):

        # Give each peak its own label before any growth begins.
        children[y, x] = number

        # Queue each peak using its negative height.
        # Smaller queue values (i.e. brighter peaks) are processed first.
        heapq.heappush(queue, (-float(surface[y, x]), arrival, int(y), int(x)))

    # List all eight neighbours, excluding the central (0, 0) pixel (i.e. the current pixel itself).
    neighbours = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]

    while queue:  # Keep growing until no queued pixels remain.

        # Take the pixel with the lowest flood level.
        # Equal levels are processes in insertion order.
        level, _, y, x = heapq.heappop(queue)

        for dy, dx in neighbours:
            yy, xx = y + dy, x + dx

            # Skip neighbours outside the local array.
            if not (0 <= yy < height and 0 <= xx < width):
                continue

            # Skip pixels outside the region or already assigned to a source.
            if not region[yy, xx] or children[yy, xx] != 0:
                continue

            # Set the neighbour's queue priority.
            # After crossing a faint pixel, reaching a brighter pixel must not give this path a higher priority agian!
            next_level = max(level, -float(surface[yy, xx]))

            children[yy, xx] = children[y, x]
            # Queue the neighbour so it can continue growing the region.
            heapq.heappush(queue, (next_level, arrival, yy, xx))
            arrival += 1

    return children


def separate_sources(smoothed, detection_snr, excluded, settings):
    """
    Find source regions and split them around distinct peaks.

    Inputs:
        smoothed: A 2D smoothed image. Only its shape is used here.
        detection_snr: A 2D array giving the detection signal-to-noise ratio at each pixel.
            Used to find peaks and divide regions.
            Have the same shape as the input image.
        excluded: A Boolean array where True marks pixels to ignore.
            Have the same shape as the input image.
        settings: A Settings object containing the detection and growth thresholds, peak selection rules,
            and minimum source size.

    Returns:
        segments: An image-sized integer array containing source IDs.
            IDs start at 1. Zero means no retained source.
        metadata: A dictionary mapping each final source ID to (parent_region_id, multiple_peaks).
            multiple_peaks is True if the parent region had more than
            one retained peak before small source regions were removed.
    """

    footprint = (detection_snr >= settings.growth_sigma) & ~excluded
    parents, _ = ndi.label(footprint, structure=CONNECTED)
    sizes = np.bincount(parents.ravel())

    # Prevent excluded bright pixels from hiding nearby valid peaks.
    searchable = np.where(excluded, -np.inf, detection_snr)
    # Find the largest S/N value in the window around each pixel.
    maxima = ndi.maximum_filter(searchable, size=settings.peak_separation + 1)
    # A candidate peak must be a local maximum and reach the higher detection threshold.
    peaks = (detection_snr == maxima) & (detection_snr >= settings.detection_sigma)
    peaks &= footprint

    segments = np.zeros(smoothed.shape, dtype=np.int32)
    metadata = {}
    next_id = 1

    # Process each parent inside a small rectangle surrounding it.
    for parent_id, bounds in enumerate(ndi.find_objects(parents), start=1):

        # Skip missing regions and regions with too few pixels.
        if bounds is None or sizes[parent_id] < settings.min_pixels:
            continue

        region = parents[bounds] == parent_id  # Select the exact parent shape inside the rectangle.
        # Group touching peak pixels so a flat peak gets one starting point.
        peak_groups, n_groups = ndi.label(peaks[bounds] & region, structure=CONNECTED)

        # Reject regions with no peak above the detection threshold.
        if n_groups == 0:
            continue

        # Pick one peak position from each group.
        # N.B. Coordinates are relative to the current rectangle.
        positions = ndi.maximum_position(
            detection_snr[bounds], peak_groups, np.arange(1, n_groups + 1)
        )

        # Remove peaks that are not clearly separated from higher peaks.
        positions = select_significant_peaks(
            detection_snr[bounds], region, positions, settings
        )

        if len(positions) == 1:
            # With only one peak, assign the whole region to source 1.
            children = region.astype(np.int32)
        else:
            # Divide the region between the retained peaks.
            children = grow_peak_regions(detection_snr[bounds], region, positions)

        for number in range(1, len(positions) + 1):
            child = children == number

            # Discard sources with too few pixels.
            # Their pixels are not given to another source.
            if np.count_nonzero(child) < settings.min_pixels:
                continue

            # Write a unique source ID into the full-image result.
            local_labels = segments[bounds]
            local_labels[child] = next_id

            metadata[next_id] = (parent_id, len(positions) > 1)
            next_id += 1

    return segments, metadata


# Define column names for the final source table.
# Each row contains measurements and quality flags for one source.
CATALOGUE_NAMES = [
    "id",
    "segment_id",
    "parent_id",
    "x",
    "y",
    "peak_x",
    "peak_y",
    "peak_adu",
    "peak_detection_snr",
    "n_pixels",
    "segment_flux_adu",
    "aperture_shape",
    "aperture_a_pix",
    "aperture_b_pix",
    "aperture_theta_deg",
    "aperture_flux_adu",
    "major_sigma_pix",
    "minor_sigma_pix",
    "elongation",
    "near_artefact",
    "near_boundary",
    "blended",
    "aperture_contaminated",
    "aperture_capped",
    "usable",
    "flags",
]

# Data types in the same order as CATALOGUE_NAMES.
CATALOGUE_TYPES = [
    int,       # id
    int,       # segment_id
    int,       # parent_id
    float,     # x
    float,     # y
    int,       # peak_x
    int,       # peak_y
    float,     # peak_adu
    float,     # peak_detection_snr
    int,       # n_pixels
    float,     # segment_flux_adu
    "U7",      # aperture_shape
    float,     # aperture_a_pix
    float,     # aperture_b_pix
    float,     # aperture_theta_deg
    float,     # aperture_flux_adu
    float,     # major_sigma_pix
    float,     # minor_sigma_pix
    float,     # elongation
    bool,      # near_artefact
    bool,      # near_boundary
    bool,      # blended
    bool,      # aperture_contaminated
    bool,      # aperture_capped
    bool,      # usable
    "U180",    # flags
]


def measure_sources(image, residual, detection_snr, segments, metadata, artefact, boundary, settings):
    """
    Measure source positions, shapes, fluxes, and quality flags.

    Use circular apertures for nearly round sources and elliptical
    apertures for sources whose elongation reaches ellipse_threshold.
    Elliptical apertures follow the measured axis ratio and direction.

    The aperture size limits apply to the radius of a circle or the
    major semi-axis of an ellipse. The minor semi-axis may be smaller
    than min_aperture_radius.

    Fluxes are sums of background-subtracted pixel values, assuming ADU units.
    They are not calibrated physical fluxes and may miss light outside
    the segment or aperture.

    If an aperture is contaminated, its flux is recorded as NaN.
    No uncertainty is calculated for either flux measurement.

    Inputs:
        image: The original 2D image.
        residual: The unsmoothed image after background subtraction.
        detection_snr: The detection signal-to-noise map.
        segments: An integer source-label image. Zero means no source.
        metadata: A dictionary mapping each segment ID to
            (parent_region_id, multiple_peaks).
        artefact: A Boolean mask marking artefact pixels.
        boundary: A Boolean mask marking excluded boundary areas.
        settings: Settings containing the minimum and maximum
            aperture radii.

        All image arrays and masks must have the same shape.

    Returns:
        catalogue: An Astropy Table with one row per measured source.
            Includes positions, fluxes, shapes, and quality flags.
            Flagged sources remain in the table.
        processed_mask: A Boolean image marking the segment pixels
            of sources added to the table, including flagged sources.
            It does not mark the circular apertures.
    """

    ids = np.arange(1, segments.max() + 1)
    peak_values = ndi.maximum(image, labels=segments, index=ids)  # Measure the highest original pixel value in each source
    order = ids[np.argsort(peak_values)[::-1]]  # Process sources from highest to lowest original peak value.
    bounds_list = ndi.find_objects(segments)  # Find a small rectangle around each source.

    # Expand the masks temporarily to identify nearby sources.
    quality_margin = int(settings.min_aperture_radius) + 3
    near_artefact = ndi.binary_dilation(artefact, iterations=quality_margin)
    near_boundary = ndi.binary_dilation(boundary, iterations=quality_margin)

    # Use the original masks to check for bad pixels inside apertures.
    aperture_bad = artefact | boundary

    # Record which segment pixels belong to sources entered in the table.
    processed_mask = np.zeros(image.shape, dtype=bool)

    rows = []
    height, width = image.shape

    for source_id in order:

        # Select this source inside its surrounding rectangle
        bounds = bounds_list[source_id - 1]
        region = segments[bounds] == source_id

        # Get source-pixel coordinates and convert them to full-image coordinates.
        local_y, local_x = np.nonzero(region)
        yy = local_y + bounds[0].start
        xx = local_x + bounds[1].start

        signal = residual[bounds][region]  # Keep signed background-subtracted values for flux measurements.
        weights = np.maximum(signal, 0)  # Use only positive signal as the weight for position and shape.

        # Skip sources with no positive weight for a centre calculation.
        if weights.sum() <= 0:
            continue

        # Find the centre, giving brighter pixels more weight.
        x = float(np.average(xx, weights=weights))
        y = float(np.average(yy, weights=weights))

        dx, dy = xx - x, yy - y  # Measure each segment pixel's displacement from the centre.
        # Build a two-dimensional matrix describing how the weighted pixels spread around the centre
        covariance = np.array([
            [np.average(dx * dx, weights=weights), np.average(dx * dy, weights=weights)],
            [np.average(dx * dy, weights=weights), np.average(dy * dy, weights=weights)],
        ])

        # Measure the source widths and the directions of its axes.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        minor, major = np.sqrt(np.maximum(eigenvalues, 0))

        # Measure how stretched the source is.
        elongation = major / max(minor, 0.25)

        # Get the major-axis direction relative to the image x-axis.
        major_direction = eigenvectors[:, -1]
        theta = float(np.arctan2(major_direction[1], major_direction[0]))

        # Use an ellipse only for clearly elongated sources
        # with a non-zero minor-axis width.
        use_ellipse = (elongation >= settings.ellipse_threshold and minor > 0)

        # Find the brightest original pixel (i.e. in global coordinate) in this source.
        brightest = int(np.argmax(image[bounds][region]))
        peak_x, peak_y = int(xx[brightest]), int(yy[brightest])

        # Choose the aperture radius or major semi-axis from the source size.
        requested_a = max(settings.min_aperture_radius, 2.5 * major)
        a = min(requested_a, settings.max_aperture_radius)
        capped = requested_a > settings.max_aperture_radius

        if use_ellipse:
            b = a * minor / major
            aperture_shape = "ellipse"
        else:
            # Equal semi-axes give a circle.
            b = a
            theta = 0.0
            aperture_shape = "circle"

        # Find the aperture's horizontal and vertical reach.
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        extent_x = np.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2)
        extent_y = np.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2)

        # Get a rectangle around the aperture, limited to the image.
        x0 = max(0, int(np.floor(x - extent_x)))
        x1 = min(width, int(np.ceil(x + extent_x)) + 1)
        y0 = max(0, int(np.floor(y - extent_y)))
        y1 = min(height, int(np.ceil(y + extent_y)) + 1)

        # Get pixel positions relative to the aperture centre.
        ay, ax = np.mgrid[y0:y1, x0:x1]
        offset_x = ax - x
        offset_y = ay - y

        # Rotate into the aperture's major-axis and minor-axis directions.
        u = offset_x * cos_t + offset_y * sin_t
        v = -offset_x * sin_t + offset_y * cos_t

        # Select pixel centres inside the ellipse.
        # This also gives a circle when a equals b.
        aperture = (u / a) ** 2 + (v / b) ** 2 <= 1

        # Mark pixels assigned to other detected sources.
        aperture_labels = segments[y0:y1, x0:x1]
        neighbours = (aperture_labels > 0) & (aperture_labels != source_id)

        # Check whether the circle extends beyond the image limits.
        cuts_edge = (
            x - extent_x < 0
            or x + extent_x >= width
            or y - extent_y < 0
            or y + extent_y >= height
        )

        # Reject aperture fluxes if the circle crosses an image edge or includes pixels in the artefact or boundary masks.
        contaminated = cuts_edge or bool(np.any(aperture & aperture_bad[y0:y1, x0:x1]))
        # Also reject apertures containing another source's labelled pixels.
        contaminated |= bool(np.any(aperture & neighbours))

        # Sum the full aperture only when it passes these checks.
        # Keep negative residual values in the sum.
        aperture_flux = (
            np.nan if contaminated
            else float(residual[y0:y1, x0:x1][aperture].sum())
        )

        # Check whether any source pixel lies near an artefact or boundary.
        touches_artefact = bool(np.any(near_artefact[bounds][region]))
        touches_boundary = bool(np.any(near_boundary[bounds][region]))

        parent_id, blended = metadata[source_id]
        flags = []

        # Collect the names of all quality checks that were triggered.
        for condition, name in [
            (touches_artefact, "near_artefact"), (touches_boundary, "near_boundary"),
            (blended, "blended"), (contaminated, "aperture_contaminated"),
            (capped, "aperture_capped"), (elongation > 6, "very_elongated"),
        ]:
            if condition:
                flags.append(name)

        # Add one row in exactly the order of CATALOGUE_NAMES.
        # usable is True only when no quality flags were triggered.
        rows.append((
            len(rows) + 1,
            source_id,
            parent_id,
            x,
            y,
            peak_x,
            peak_y,
            float(image[peak_y, peak_x]),
            float(detection_snr[bounds][region].max()),
            int(region.sum()),
            float(signal.sum()),
            aperture_shape,
            float(a),
            float(b),
            float(np.degrees(theta)),
            aperture_flux,
            major,
            minor,
            elongation,
            touches_artefact,
            touches_boundary,
            blended,
            contaminated,
            capped,
            not flags,
            "|".join(flags) or "none",
        ))

        # Mark this source's segment pixels as recorded.
        # Include flagged sources, but do not mark the whole aperture.
        processed_mask[bounds] |= region

    return Table(rows=rows, names=CATALOGUE_NAMES, dtype=CATALOGUE_TYPES), processed_mask

def save_diagnostics(image, background, noise, artefact, boundary, saturated, catalogue, output):
    """Plotting"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Ellipse, Patch

    snr = (image - background) / noise
    masked = np.ma.masked_where(~artefact, np.ones(image.shape, dtype=np.uint8))

    def draw(ax, limits=None):
        """
        Show the S/N image in grey with artefacts highlighted in gold.
        Use limits to zoom into a selected part of the image.
        """
        ax.imshow(snr, origin="lower", cmap="gray", vmin=-1, vmax=12, interpolation="nearest")
        ax.imshow(masked, origin="lower", cmap="autumn", vmin=0, vmax=1, alpha=0.45, interpolation="nearest")
        ax.set_ylabel("y / pixel")

        if limits is not None:
            ax.set_xlim(limits[:2])
            ax.set_ylim(limits[2:])

    # Create a full-image overview with two panels.
    # Left: highlight artefacts in gold and excluded boundary areas in blue.
    # Right: show source apertures over the image and artefact overlay.
    fig, axes = plt.subplots(1, 2, figsize=(13, 11))
    draw(axes[0])
    padding = np.ma.masked_where(~boundary, np.ones(image.shape, dtype=np.uint8))
    axes[0].imshow(padding, origin="lower", cmap="Blues", vmin=0, vmax=1, alpha=0.4)
    axes[0].set_title("Automatic mask: artefacts (gold), boundary/padding (blue)")
    draw(axes[1])

    legend_handles = []

    for usable, colour, name in [(True, "blue", "Usable candidates"), (False, "red", "Flagged candidates")]:
        subset = catalogue[catalogue["usable"] == usable]

    # Draw the actual aperture used for each source.
    # Blue marks usable candidates; red marks candidates with quality flags.
        apertures = [
            Ellipse(
                (float(row["x"]), float(row["y"])),
                width=2 * float(row["aperture_a_pix"]),
                height=2 * float(row["aperture_b_pix"]),
                angle=float(row["aperture_theta_deg"]),
            )
            for row in subset
        ]
        axes[1].add_collection(PatchCollection(apertures, facecolors="none", edgecolors=colour, linewidths=0.4), autolim=False)
        legend_handles.append(Patch(facecolor="none", edgecolor=colour, label=name))

    axes[1].legend(handles=legend_handles, loc="upper right", fontsize=8)
    axes[1].set_title(f"Adaptive apertures: {len(catalogue)} candidates; {np.count_nonzero(catalogue['usable'])} usable")
    fig.tight_layout()
    fig.savefig(output / "detection_overview.png", dpi=300)
    plt.close(fig)

    # Select up to six largest connected artefact regions.
    # Create a 2-by-3 grid to inspect them individually.
    groups, _ = ndi.label(artefact, structure=CONNECTED)
    areas = np.bincount(groups.ravel())
    selected = (np.argsort(areas[1:])[::-1] + 1)[:6]
    bounds = ndi.find_objects(groups)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Show each selected artefact with a 30-pixel margin around it.
    # Label each panel with the group ID and masked pixel count.
    for ax, group_id in zip(axes.ravel(), selected):
        ys, xs = bounds[group_id - 1]
        limits = (
            max(0, xs.start - 30), min(image.shape[1], xs.stop + 30),
            max(0, ys.start - 30), min(image.shape[0], ys.stop + 30),
        )
        draw(ax, limits)
        ax.set_title(f"Artefact group {group_id}: {areas[group_id]} masked pixels")

    # Hide unused panels and save the artefact close-ups.
    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output / "artefact_closeups.png", dpi=300)
    plt.close(fig)

    # If an artefact exists, inspect the largest one in three detail views:
    # its lower end, its widest part, and its upper end.
    # Each view covers at most 600 pixels vertically.
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
        fig.savefig(output / "bleeding_detail.png", dpi=300)
        plt.close(fig)


def run_detection(fits_path, output_dir, settings=None):
    """Run source detection, measurements, and output generation."""

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
    # Exclude missing data and a margin around it.
    # Also exclude a strip along each outer image edge.
    boundary = ndi.binary_dilation(missing, iterations=settings.boundary_padding)
    boundary[:settings.boundary_padding, :] = True
    boundary[-settings.boundary_padding:, :] = True
    boundary[:, :settings.boundary_padding] = True
    boundary[:, -settings.boundary_padding:] = True
    excluded = artefact | boundary
    # Replace missing values temporarily so filtering can use finite numbers.
    safe_image = np.where(missing, level, image)

    print("2/5: Estimating local background and noise...", flush=True)
    # Make an initial background and noise estimate.
    background, noise = background_maps(safe_image, excluded, settings.background_box)
    residual = safe_image - background
    # Temporarily mask likely sources while refining the background.
    initial_sources = ndi.binary_dilation((residual > 3 * noise) & ~excluded, iterations=3)
    # Re-estimate the backgroud with likely sources left out.
    background, noise = background_maps(safe_image, excluded | initial_sources, settings.background_box)
    residual = safe_image - background

    # Smooth the background-subtracted image for detection.
    # before smoothing, temporarily replace excluded pixels with zero,
    # so their values do not spread into nearby pixels.
    smoothed = ndi.gaussian_filter(np.where(excluded, 0, residual), settings.smooth_sigma)
    # Add a detection-only margin around excluded areas.
    detection_excluded = ndi.binary_dilation(excluded, iterations=4)
    # Measure the local baseline and noise in the smoothed image.
    smooth_level, smooth_noise = background_maps(smoothed, detection_excluded | initial_sources, settings.background_box)
    # Express the smoothed signal in units of local smotthed noise.
    detection_snr = (smoothed - smooth_level) / smooth_noise

    print("3/5: Detecting connected regions and separating peaks...", flush=True)
    # Find source regions and divide them between distinct peaks.
    segments, metadata = separate_sources(smoothed, detection_snr, detection_excluded, settings)

    # Repeat detection with the signs reversed as a diagnostic check.
    negative_segments, _ = separate_sources(-smoothed, -detection_snr, detection_excluded, settings)
    negative_count = int(negative_segments.max())
    del negative_segments

    print("4/5: Measuring sources in descending peak brightness...", flush=True)
    # Measure sources using the original image and unsmoothed residual.
    catalogue, processed = measure_sources(image, residual, detection_snr, segments, metadata, artefact, boundary, settings)

    catalogue.meta["description"] = "Unclassified source candidates; usable is a conservative photometry-quality flag."
    catalogue.meta["flux_note"] = "Background-subtracted ADU assumed; segment and aperture fluxes are not total fluxes."
    catalogue.write(output_dir / "sources.ecsv", overwrite=True)
    catalogue.write(output_dir / "sources.csv", format="ascii.csv", overwrite=True)
    catalogue[catalogue["usable"]].write(output_dir / "usable_sources.csv", format="ascii.csv", overwrite=True)

    # Group connected artefact pixels for a record of excluded areas.
    # A group may contain more than one star or a star and its trail.
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
    parser.add_argument("--output", type=Path, default=base_dir / "output" / "source_detection_results")
    args = parser.parse_args()
    run_detection(args.fits, args.output)
