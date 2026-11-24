from pathlib import Path

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

project_dir = Path(
    "/Users/wendy/Desktop/Imperial/Year 3/Lab/Cycle 2 Astronomical Data/"
    "Y3-lab-Cycle2-Astrophysical-imaging-processing"
)

image_path = project_dir / "Astro" / "Fits_Data" / "mosaic.fits"
catalogue_path = project_dir / "Astro" / "red_shift" / "red_shift_catalogue.fits"

catalogue = Table.read(catalogue_path, hdu=1)
wcs = WCS(fits.getheader(image_path)).celestial

x, y = wcs.world_to_pixel_values(catalogue["RAJ2000"], catalogue["DEJ2000"])

width, height = wcs.pixel_shape
inside = (x >= -0.5) & (x < width - 0.5) & (y >= -0.5) & (y < height - 0.5)

catalogue_in_image = catalogue[inside]
catalogue_in_image["x_pixel"] = x[inside]
catalogue_in_image["y_pixel"] = y[inside]
print(catalogue_in_image["IR", "zsp", "x_pixel", "y_pixel"])