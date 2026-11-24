from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.cosmology import Planck18
from astropy.visualization import ZScaleInterval

project_dir = Path(__file__).resolve().parent.parent.parent
script_dir = Path(__file__).resolve().parent
catalogue_path = script_dir / "outputs" / "galaxy_info" / "galaxy_reference.ecsv"
completeness_path = project_dir / "Astro" / "red_shift" / "completeness_curves_1119.ecsv"

catalogue = Table.read(catalogue_path, format="ascii.ecsv")

print(catalogue.colnames)       # Column names
print(catalogue[:5])            # First five rows
print(catalogue["z_spec"][:5])  # First five spectroscopic redshifts

completeness_curve = Table.read(completeness_path, format="ascii.ecsv")

print(completeness_curve.colnames)       # Column names
print(completeness_curve[:5])            # First five rows

useful_r_eff = completeness_curve["r_eff_px"] == 3.0
mag_midpoint = (completeness_curve["mag_min"] + completeness_curve["mag_max"]) / 2

useful_fraction = completeness_curve["detection_fraction"][useful_r_eff]
useful_mag = mag_midpoint[useful_r_eff]

