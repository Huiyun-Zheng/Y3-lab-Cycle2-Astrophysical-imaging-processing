import astropy as ap
from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def plot_histogram(data, bins=50, range=(3300, 3600)):
    """
    Plots a histogram of the data.

    Parameters:
    data (numpy.ndarray): The data to be plotted.
    title (str): The title of the histogram.
    bins (int): The number of bins for the histogram.
    """

    plt.figure(figsize=(10, 6))
    plt.hist(data.ravel(), bins=bins, color='blue', alpha=0.7, range=range)
    plt.title('Histogram of Pixel Values indicating the Background')
    plt.xlabel('Pixel Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid()
    plt.show()
    

def fit_gaussian(data, bins=50, range=(3300, 3600)):
    """
    Fits a Gaussian to the histogram of the data.

    Parameters:
    data (numpy.ndarray): The data to be fitted.
    bins (int): The number of bins for the histogram.

    Returns:
    mean (float): The mean of the fitted Gaussian.
    stddev (float): The standard deviation of the fitted Gaussian.
    """

    # Create histogram
    hist, bin_edges = np.histogram(data.ravel(), bins=bins, range=range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if np.count_nonzero(hist) < 3:
        raise ValueError("Need at least three populated bins to fit the background")

    # Fit a Gaussian
    def gaussian(x, amp, mean, std):
        return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    # Initialize from the histogram being fitted, excluding bright-source tails.
    mean_initial = bin_centers[np.argmax(hist)]
    histogram_mean = np.average(bin_centers, weights=hist)
    std_initial = np.sqrt(
        np.average((bin_centers - histogram_mean) ** 2, weights=hist)
    )
    popt, _ = curve_fit(
        gaussian, bin_centers, hist,
        p0=[hist.max(), mean_initial, std_initial],
        bounds=(
            [0, bin_edges[0], np.finfo(float).eps],
            [np.inf, bin_edges[-1], np.inf]
        )
    )
    
    mean = popt[1]
    stddev = popt[2]

    # Plot the histogram and the fitted Gaussian
    plot_data = np.arange(range[0], range[1], 3)
    plt.figure(figsize=(10, 6))
    plt.hist(data.ravel(), bins=bins, color='blue', alpha=0.7, range=range, label='Data Histogram')
    plt.plot(plot_data, gaussian(plot_data, *popt), color='red', label='Fitted Gaussian')
    plt.title('Histogram with Fitted Gaussian')
    plt.xlabel('Pixel Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

    return mean, stddev
