import astropy as ap
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from matplotlib import image
from pathlib import Path
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
from photutils.aperture import CircularAperture
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from bleeding import (spot_linear_sources,
                    bleeding_band_mask, 
                    mask_whole_bleeding_region,
                    find_halo_radius,
                    mask_saturated)
from stars import find_stars
from aperture_conversion import aperture_center_shape, plot_source_aperture, plot_aperture_scales, optimize_scale

saturation = 36000

def filter_for_detection(data, mean, smooth_sigma=1.5):
    """Smooth background-subtracted data; ignore nonfinite pixels."""

    if not np.isfinite(smooth_sigma) or smooth_sigma <= 0:
        raise ValueError("smooth_sigma must be positive")

    valid = np.isfinite(data)
    residual = np.where(valid, data - mean, 0.0)
    filtered = ndi.gaussian_filter(residual, sigma=smooth_sigma)
    coverage = ndi.gaussian_filter(valid.astype(float), sigma=smooth_sigma)
    np.divide(filtered, coverage, out=filtered, where=coverage > 0)
    filtered[~valid | (coverage == 0)] = np.nan
    return filtered


def detect_sources(
    data, bins=300, hist_range=(3300, 3600),
    threshold=3, min_area=5, smooth_sigma=None
):
    """
    Assume a uniform background; smooth_sigma=None disables smoothing.
    Returned mean and stddev describe the original image.
    """

    valid = np.isfinite(data)
    mean, stddev = fit_gaussian(data[valid], bins=bins, range=hist_range)
    if not np.isfinite(mean) or not np.isfinite(stddev) or stddev <= 0:
        raise ValueError("Invalid background fit")

    if smooth_sigma is None:
        mask = valid & (data > mean + threshold * stddev)
    else:
        filtered = filter_for_detection(data, mean, smooth_sigma)
        _, filtered_bg, filtered_std = sigma_clipped_stats(
            filtered[np.isfinite(filtered)], sigma=3, maxiters=5
        )
        if not np.isfinite(filtered_std) or filtered_std <= 0:
            raise ValueError("Invalid filtered background noise")
        mask = valid & (filtered > filtered_bg + threshold * filtered_std)
        print(f"Filtered background RMS: {filtered_std:.3f} ADU")

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
        data, segment_map, n_pixels=10, n_levels=32, contrast=0.05
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

def plot_detected_sources(data, labels, contour='no_deblending', filename=None):
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
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_single_source(labels, label_center):
    """
    Plots a single source from the labeled mask.
    label_center: (int, int) - The (y, x) coordinates of the source center in the labels array.
    """
    target = labels[label_center[0], label_center[1]]

    plt.xlabel('X Pixel (0 offset)')
    plt.ylabel('Y Pixel (0 offset)')
    plt.title(f'Source with label {target}')
    plt.imshow(
        labels == target,
        cmap="gray", origin="lower",
        interpolation="nearest"
    )
    plt.show()