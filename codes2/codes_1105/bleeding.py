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

def mask_whole_bleeding_region(data, labels, mean, stddev, threshold=3):
    """
    Masks bleeding artifacts in the image.

    Parameters:
    data (numpy.ndarray): The original image data.
    labels (numpy.ndarray): The labeled mask of detected sources.
    mean (float): The mean of the background.
    stddev (float): The standard deviation of the background.
    threshold (float): The threshold for masking bleedings.

    Returns:
    masked_data (numpy.ndarray): The data with bleedings masked.
    """

    # Create a mask for bleedings
    bleeding_mask = (data > mean + threshold * stddev) & (labels > 0)

    # Mask the bleedings in the data
    masked_data = np.copy(data)
    masked_data[bleeding_mask] = mean  # Replace bleeding pixels with background mean

    return masked_data

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