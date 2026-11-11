import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line


def spot_linear_sources(
        data, labels, x_range, y_range, 
        threshold=10, line_length=20, line_gap=3, 
        all_possibilities=True
        ):
    x_min, x_max = x_range
    y_min, y_max = y_range

    section = data[y_min:y_max, x_min:x_max]
    section_labels = labels[y_min:y_max, x_min:x_max]

    mask = section_labels > 0

    if all_possibilities:
        lines = probabilistic_hough_line(
            mask,
            threshold=threshold,    # minimum number of votes to consider a line
            line_length=line_length,   # minimum length of a line
            line_gap=line_gap    # maximum gap between segments to link them into a single line
        )

        vmin, vmax = ZScaleInterval().get_limits(section)
        fig, ax = plt.subplots()
        ax.imshow(section, cmap="gray", origin="lower",
                    vmin=vmin, vmax=vmax)
    
        for (x0, y0), (x1, y1) in lines:
            ax.plot([x0, x1], [y0, y1], color="red")
    
        ax.set_xlim(-0.5, section.shape[1] - 0.5)
        ax.set_ylim(-0.5, section.shape[0] - 0.5)
        plt.show()
        plt.close(fig)
    
        return lines

    else:
        tested_angles = np.deg2rad(np.arange(-90, 90))
        tested_angles = np.array([0, np.pi/2])
        h, theta, rho = hough_line(mask, theta=tested_angles)

        votes, peak_angles, peak_rhos = hough_line_peaks(
            h, theta, rho,
            min_distance=10,
            min_angle=15,
            num_peaks=2
        )

        vmin, vmax = ZScaleInterval().get_limits(section)
        fig, ax = plt.subplots()
        ax.imshow(section, cmap="gray", origin="lower",
                vmin=vmin, vmax=vmax)
        
        for angle, distance in zip(peak_angles, peak_rhos):
            # a point on the line in Cartesian coordinates
            x0 = distance * np.cos(angle)
            y0 = distance * np.sin(angle)

            # another point
            x1 = x0 - np.sin(angle)
            y1 = y0 + np.cos(angle)

            ax.axline((x0, y0), (x1, y1), color="red")

        ax.set_xlim(-0.5, section.shape[1] - 0.5)
        ax.set_ylim(-0.5, section.shape[0] - 0.5)
        plt.show()
        plt.close(fig)

        return peak_angles, peak_rhos

def find_halo_radius(
        data, guess_center, radius_low, mean, stddev,
        bleed_mask=None, max_radius=150,
        n_sigma=0.4, consecutive=6
        ):

    x_center, y_center = guess_center
    height, width = data.shape

    # Keep complete rings inside the image
    radius_limit = int(min(
        max_radius,
        x_center, width - 1 - x_center,
        y_center, height - 1 - y_center
    ))

    if radius_limit <= radius_low:
        raise ValueError("Not enough space to search outside radius_low")

    # Local section
    x_min = max(0, int(np.floor(x_center - radius_limit)))
    x_max = min(width, int(np.ceil(x_center + radius_limit)) + 1)
    y_min = max(0, int(np.floor(y_center - radius_limit)))
    y_max = min(height, int(np.ceil(y_center + radius_limit)) + 1)

    section = data[y_min:y_max, x_min:x_max]
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    distance = np.hypot(xx - x_center, yy - y_center)

    valid = np.isfinite(section)
    if bleed_mask is not None:
        valid &= ~bleed_mask[y_min:y_max, x_min:x_max]

    cutoff = mean + n_sigma * stddev
    count = 0

    # Search outward
    for r in range(int(np.ceil(radius_low)), radius_limit):
        ring = (distance >= r) & (distance < r + 1) & valid
        values = section[ring]

        if values.size == 0:
            count = 0
            continue

        if np.median(values) <= cutoff:
            count += 1
        else:
            count = 0

        if count == consecutive:
            return r - consecutive + 1

    return None

def mask_whole_bleeding_region(data, guess_center, band_radius, halo_radius, mean, stddev,
        half_width=3, keep_core=True, upper_bound='auto', lower_bound='auto'):

    # Circular mask
    yy, xx = np.ogrid[:data.shape[0], :data.shape[1]]

    core_mask = (
        (xx - guess_center[0])**2
        + (yy - guess_center[1])**2
    ) <= halo_radius**2

    # Bleeding band mask
    bleed_mask = bleeding_band_mask(
        data, guess_center, band_radius, mean, stddev,
        half_width=half_width,
        keep_core=keep_core,
        upper_bound=upper_bound,
        lower_bound=lower_bound
    )

    # Combined mask
    combined_mask = core_mask | bleed_mask

    return combined_mask

def bleeding_band_mask(
        data, guess_center, radius_low, mean, stddev,
        half_width=3, keep_core=True, upper_bound='auto', lower_bound='auto'
        ):
    """Assuming vertical bleeding bands only."""

    x_center = int(round(guess_center[0]))
    y_center = guess_center[1]
    y_high = int(np.ceil(y_center + radius_low))
    y_low = int(np.floor(y_center - radius_low))
    height, width = data.shape

    cutoff = mean + 0.4 * stddev
    L = 6

    y_max = height
    y_min = 0

        # Upper bound
    if upper_bound == 'auto':
        y_max = height
        count = 0

        for y in range(max(0, y_high), height):
            value = data[y, x_center]

            if np.isfinite(value) and value <= cutoff:
                count += 1
            else:
                count = 0

            if count == L:
                y_max = y - L + 1
                break
    else:
        y_max = int(np.clip(upper_bound, 0, height))

    # Lower bound
    if lower_bound == 'auto':
        y_min = 0
        count = 0

        for y in range(min(height - 1, y_low), -1, -1):
            value = data[y, x_center]

            if np.isfinite(value) and value <= cutoff:
                count += 1
            else:
                count = 0

            if count == L:
                y_min = y + L
                break
    else:
        y_min = int(np.clip(lower_bound, 0, height))

    if y_min >= y_max:
        raise ValueError("lower_bound must be below upper_bound")

    # Band mask
    half_width = int(np.ceil(half_width))
    x_min = max(0, x_center - half_width)
    x_max = min(width, x_center + half_width + 1)

    bleed_mask = np.zeros(data.shape, dtype=bool)
    bleed_mask[y_min:y_max, x_min:x_max] = True

    # Exclude the core
    if keep_core:
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        core_mask = (
            (xx - guess_center[0])**2
            + (yy - y_center)**2
        ) <= radius_low**2

        bleed_mask[y_min:y_max, x_min:x_max] &= ~core_mask

    return bleed_mask 

def mask_saturated(data, labels, saturation=36000):
    # Pixels above the good-data limit
    saturated_pixels = data > saturation

    # Labels containing these pixels
    bad_ids = np.unique(labels[saturated_pixels])
    bad_ids = bad_ids[bad_ids > 0]

    # Mask whole regions and any unlabeled saturated pixels
    saturation_mask = np.isin(labels, bad_ids) | saturated_pixels

    return saturation_mask