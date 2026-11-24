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
fits_path = project_dir / "Astro" / "Fits_Data" / "mosaic.fits"

output_dir = Path(__file__).resolve().parent / "outputs" / "galaxy_info"
output_dir.mkdir(parents=True, exist_ok=True)

# Use the previously downloaded red-shift catalogue
use_saved_catalogue = True # file already exist in project directory, do not change the bool
catalogue_path = project_dir / "Astro" / "red_shift" / "catalogue_subset.ecsv"

# Observation information
header = fits.getheader(fits_path)
wcs = WCS(header).celestial

height = header["NAXIS2"]
width = header["NAXIS1"]

center = wcs.pixel_to_world((width - 1) / 2, (height - 1) / 2)

print("Target:", header["OBJECT"])
print("Observation start:", header["DATE-OBS"])
print("Telescope:", header["TELESCOP"])
print("Camera:", header["DETECTOR"])
print("Filter:", header["FILTER"])
print(f"Centre RA: {center.ra.deg:.6f} deg")
print(f"Centre Dec: {center.dec.deg:.6f} deg")

# Load (downloaded) or query (download from internet) the catalogue
if use_saved_catalogue:
    catalogue = Table.read(catalogue_path)

    catalogue.rename_column("ra", "RAJ2000") # right ascension, reference epoch J2000
    catalogue.rename_column("dec", "DEJ2000") # declination, reference epoch J2000
    catalogue.rename_column("z_spec", "zsp") # spectroscopic redshift

    print("Using the saved subset; the original query reached its row limit.")

else:
    from astroquery.vizier import Vizier

    # Cover the image corners
    corners = wcs.pixel_to_world(
        [-0.5, width - 0.5, width - 0.5, -0.5],
        [-0.5, -0.5, height - 0.5, height - 0.5]
    )
    radius = center.separation(corners).max() + 0.5 * u.arcmin

    query = Vizier(
        columns=[
            "IR", "Opt", "RAJ2000", "DEJ2000",
            "mst", "zsp", "q_zsp", "r_zsp"
        ],
        column_filters={"zsp": ">0"},
        row_limit=-1
    )
    query.TIMEOUT = 60

    tables = query.query_region(
        center,
        radius=radius,
        catalog="II/326/zcatrev"
    )

    if len(tables) == 0:
        raise RuntimeError("No spectroscopic catalogue table returned.")

    catalogue = tables[0]
    catalogue.write(output_dir / "swire_query.ecsv", overwrite=True)

# Keep positive spectroscopic redshifts
z = np.asarray(np.ma.filled(catalogue["zsp"], np.nan), dtype=float)
catalogue = catalogue[np.isfinite(z) & (z > 0)].copy()

# Convert sky coordinates to original pixels
x, y = wcs.world_to_pixel_values(catalogue["RAJ2000"], catalogue["DEJ2000"])

inside = (
    np.isfinite(x) & np.isfinite(y)
    & (x >= -0.5) & (x < width - 0.5)
    & (y >= -0.5) & (y < height - 0.5)
)

catalogue = catalogue[inside]

if len(catalogue) == 0:
    raise RuntimeError("No spectroscopic catalogue positions inside the image.")

# Build the reference table
results = Table()

results["IR"] = catalogue["IR"]
results["ra_deg"] = catalogue["RAJ2000"]
results["dec_deg"] = catalogue["DEJ2000"]
results["x_pixel"] = x[inside]
results["y_pixel"] = y[inside]
results["mst"] = catalogue["mst"]
results["z_spec"] = catalogue["zsp"]

# Calculate distances and times by red shift
z = np.asarray(results["z_spec"], dtype=float)

results["distance_Gpc"] = Planck18.luminosity_distance(z).to_value(u.Gpc)
results["lookback_Gyr"] = Planck18.lookback_time(z).to_value(u.Gyr)
results["universe_age_Gyr"] = Planck18.age(z).to_value(u.Gyr)

results.meta["catalogue"] = "II/326/zcatrev"
results.meta["saved_subset"] = use_saved_catalogue
results.meta["cosmology"] = str(Planck18)
results.meta["coordinates"] = "Original image, zero-based pixels"
results.meta["warning"] = (
    "Catalogue positions, not confirmed counterpart matches. "
    "No stellar ages measured."
)
results["distance_Gpc"].description = "Luminosity distance"

results.sort("z_spec") # rank from small to large

# Save the table
results.write(output_dir / "galaxy_reference.ecsv", overwrite=True)
results.write(output_dir / "galaxy_reference.csv", format="ascii.csv", overwrite=True)

print("Catalogue positions with spectroscopic redshifts:", len(results))
results[
    ["IR", "z_spec", "distance_Gpc", "lookback_Gyr", "universe_age_Gyr"]
].pprint(max_lines=15)

# Plot catalogue positions on the original fits image
with fits.open(fits_path, memmap=False) as hdul:
    data = hdul[0].data

    vmin, vmax = ZScaleInterval().get_limits(data)

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(
        data, cmap="gray", origin="lower",
        vmin=vmin, vmax=vmax
    )

    points = ax.scatter(
        results["x_pixel"],
        results["y_pixel"],
        c=results["z_spec"],
        cmap="plasma",
        marker="+",
        s=60
    )

    fig.colorbar(points, ax=ax, label="Spectroscopic redshift")

    ax.set_title(
        f"{len(results)} catalogue positions with spectroscopic redshifts"
    )
    ax.set_xlabel("X pixel (original image)")
    ax.set_ylabel("Y pixel (original image)")

    fig.tight_layout()
    fig.savefig(output_dir / "catalogue_positions.png", dpi=200)
    plt.show()
    plt.close(fig)

print("Outputs:", output_dir)