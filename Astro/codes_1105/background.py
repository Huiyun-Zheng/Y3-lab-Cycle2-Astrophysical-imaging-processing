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

    # Fit a Gaussian
    def gaussian(x, amp, mean, std):
        return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), np.mean(data), np.std(data)])
    
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