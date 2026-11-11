import astropy as ap
from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def find_stars(data, labels, center):

    x, y = map(int, center)
    target_id = labels[y, x]

    if target_id == 0:
        raise ValueError("The selected pixel is background")
    
    artefact_mask = labels == target_id

    return artefact_mask