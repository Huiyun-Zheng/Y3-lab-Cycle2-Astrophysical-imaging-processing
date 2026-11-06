import astropy as ap
from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from codes2.codes_1105.get import get_fits_data
from codes2.codes_1105.background import plot_histogram, fit_gaussian

def gaussian_background(file_path):
    """
    Reads a FITS file, applies scaling, plots the histogram of pixel values,
    and fits a Gaussian to the histogram to determine the background.

    Parameters:
    file_path (str): The path to the FITS file.

    Terminal:
    from main import gaussian_background
    project_dir = Path(__file__).resolve().parent.parent
    fits_path = project_dir / "Astro" / "Fits_Data" / "mosaic.fits"
    gaussian_background(str(fits_path))
    """

    # Get data from FITS file
    header, data = get_fits_data(file_path)

    # Plot histogram of pixel values
    plot_histogram(data)

    # Fit Gaussian to the histogram
    mean, stddev = fit_gaussian(data)

    print(f"Fitted Gaussian Mean: {mean}")
    print(f"Fitted Gaussian Standard Deviation: {stddev}")