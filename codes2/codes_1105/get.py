import astropy as ap
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def get_fits_data(file_path):
    """
    Reads a FITS file and returns the header and data.

    Parameters:
    file_path (str): The path to the FITS file.

    Returns:
    header (astropy.io.fits.Header): The header of the FITS file.
    data (numpy.ndarray): The data from the FITS file.
    """
    hdulist = fits.open(file_path)
    header = hdulist[0].header
    data = hdulist[0].data.astype(float)
    return header, data