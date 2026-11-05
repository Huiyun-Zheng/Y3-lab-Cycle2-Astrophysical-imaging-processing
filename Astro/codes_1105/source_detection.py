import astropy as ap
from astropy.io import fits
from matplotlib import image
import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, AsinhNorm
from get import get_fits_data
from background import plot_histogram, fit_gaussian
from astropy.visualization import (AsinhStretch, 
                                   LogStretch, 
                                   ImageNormalize, 
                                   ZScaleInterval,)
from photutils.segmentation import deblend_sources, SegmentationImage
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

def detect_sources(
    data, bins=50, hist_range=(3300, 3600),
    threshold=3, min_area=5
):
    """
    Assuming a uniform background. Labels have the same shape as input data.
    """

    valid = np.isfinite(data)
    mean, stddev = fit_gaussian(data[valid], bins=bins, range=hist_range)
    if not np.isfinite(mean) or not np.isfinite(stddev) or stddev <= 0:
        raise ValueError("Invalid background fit")

    # Find pixels that are above the threshold
    mask = valid & (data > mean + threshold * stddev)

    # Label connected components in the mask
    connectivity = ndi.generate_binary_structure(2, 2)  # 8-connectivity
    print("Mask shape:", mask.shape)
    print('data shape:', data.shape)
    print('connectivity shape:', connectivity.shape)
    labels, _ = ndi.label(mask, structure=connectivity)

    # Delete regions with fewer pixels than min_area
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False                 # 0 is background, do not keep
    mask = keep[labels]

    # Relabel the mask after removing small regions
    labels, _ = ndi.label(mask, structure=connectivity)

    return labels, mean, stddev


def deblend(data, labels):
    """
    Deblends overlapping sources in the image.
    """
    segment_map = SegmentationImage(labels)
    segment_map_deblended = deblend_sources(
        data, segment_map, n_pixels=10, n_levels=32, contrast=0.001
    )
    labels_deblended = segment_map_deblended.data

    print("before deblending:", segment_map.nlabels)
    print("after deblending:", segment_map_deblended.nlabels)

    return labels_deblended

def crop_array(array, x_range, y_range):
    """
    x_range: (int, int)
    y_range: (int, int)
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    return array[y_min:y_max, x_min:x_max]

def plot_detected_sources(data, labels, contour='no_deblending'):
    """
    Plots the original data with detected sources overlaid.

    Parameters:
    data (numpy.ndarray): The original image data.
    labels (numpy.ndarray): The labeled mask of detected sources.
    """

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(data)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='gray', origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar(label='Pixel Value')
    plt.title('Detected Sources Overlay')

    # Overlay detected sources
    if contour == 'no_deblending':
        plt.contour(
            labels > 0,
            levels=[0.5],
            colors='red',
            linewidths=0.5
        )

    if contour == 'contour':
        for label_id in np.unique(labels):
            if label_id == 0:
                continue

            plt.contour(
                labels == label_id,
                levels=[0.5],
                colors='red',
                linewidths=0.5
            )

    elif contour == 'scatter':
        edges = np.zeros(labels.shape, dtype=bool)

        different = labels[:, 1:] != labels[:, :-1]
        edges[:, 1:] |= different
        edges[:, :-1] |= different

        different = labels[1:, :] != labels[:-1, :]
        edges[1:, :] |= different
        edges[:-1, :] |= different

        edges &= labels > 0
        del different
        y, x = np.nonzero(edges)
        plt.scatter(x, y, color="red", s=0.5, linewidths=0)

    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()

def plot_single_source(labels, label_center):
    """
    Plots a single source from the labeled mask.
    label_center: (int, int) - The (y, x) coordinates of the source center in the labels array.
    """
    target = small_labels[label_center[0], label_center[1]]

    plt.xlabel('X Pixel (0 offset)')
    plt.ylabel('Y Pixel (0 offset)')
    plt.title(f'Source with label {target}')
    plt.imshow(
        small_labels == target,
        cmap="gray", origin="lower",
        interpolation="nearest"
    )
    plt.show()

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
    



header, data = get_fits_data('../Fits_Data/mosaic.fits')

# Crop once before background fitting, detection, deblending, and plotting.
crop_x_range = (70, 2450)
crop_y_range = (0, 4550)
data = crop_array(data, crop_x_range, crop_y_range)
x_offset, y_offset = crop_x_range[0], crop_y_range[0]

print(type(data), data.shape, data.dtype)
labels, mean, stddev = detect_sources(data, threshold=3, min_area=5)
plot_detected_sources(data, labels)
deblended_labels = deblend(data, labels)
#plot_detected_sources(data, deblended_labels, contour='scatter')

angles, rhos = spot_linear_sources(
    data, deblended_labels,
    # Convert these original-image coordinates to the cropped-image frame.
    x_range=(700 - x_offset, 900 - x_offset),
    y_range=(3200 - y_offset, 3450 - y_offset),
    all_possibilities=False
)

lines = spot_linear_sources(
    data, deblended_labels,
    x_range=(700 - x_offset, 900 - x_offset),
    y_range=(3200 - y_offset, 3450 - y_offset),
    threshold=10,
    line_length=150,
    line_gap=3
)

small_x_range = (1000 - x_offset, 1500 - x_offset)
small_y_range = (1000 - y_offset, 1500 - y_offset)
small_data = crop_array(data, small_x_range, small_y_range)
small_labels = crop_array(deblended_labels, small_x_range, small_y_range)
plot_detected_sources(small_data, small_labels)
