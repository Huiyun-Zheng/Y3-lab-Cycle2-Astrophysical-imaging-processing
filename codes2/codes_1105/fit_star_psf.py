import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Const2D
from astropy.modeling.fitting import TRFLSQFitter
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.psf import MoffatPRF
from get import get_fits_data
from source_detection import crop_array
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
fits_path = project_dir / "Astro" / "Fits_Data" / "mosaic.fits"
header, data = get_fits_data(file_path=fits_path)
output_dir = Path(__file__).resolve().parent / "outputs" / "psf_fit"
output_dir.mkdir(parents=True, exist_ok=True)

# Reference candidate
x_range = (875, 946)
y_range = (1570, 1641)

x_range = (2269, 2340)
y_range = (4125, 4196)

# Find the region where the star is on the image
star_1 = crop_array(data, x_range, y_range).astype(float)
yy, xx = np.indices(star_1.shape)
y_peak, x_peak = np.unravel_index(np.nanargmax(star_1), star_1.shape)
radius = np.hypot(xx - x_peak, yy - y_peak)

# Estimate the local background by 3-sigma clipping and remove background
sky_mask = (radius >= 25) & (radius <= 32) & np.isfinite(star_1)
if min(x_peak, y_peak, star_1.shape[1] - 1 - x_peak,
       star_1.shape[0] - 1 - y_peak) < 32:
    raise ValueError("Enlarge the crop to contain the background annulus")
_, background, stddev = sigma_clipped_stats(star_1[sky_mask], sigma=3, maxiters=5)
star_subtracted = star_1 - background

# Fit the central region only
fit_radius = 15
fit_mask = (radius <= fit_radius) & np.isfinite(star_1)
saturation = header["SATURATE"]
if np.any(star_1[fit_mask] >= saturation):
    raise ValueError("The selected source contains saturated pixels")

initial_flux = np.sum(star_subtracted[fit_mask])
model = MoffatPRF(
    flux=initial_flux,
    x_0=x_peak, y_0=y_peak,
    alpha=3, beta=2.5,
    n_nodes=5
)    # guess parameters
# Set bounds or fitting
model.alpha.fixed = False
model.beta.fixed = False
model.alpha.bounds = (0.3, 20)
model.beta.bounds = (1.05, 20)
model.x_0.bounds = (x_peak - 2, x_peak + 2)
model.y_0.bounds = (y_peak - 2, y_peak + 2)
model.flux.bounds = (0, None)

# Allow a small background adjustment
# For optimization, set background to be fitted as well
sky_model = Const2D(
    amplitude=background,
    bounds={"amplitude": (background - 5 * stddev,
                          background + 5 * stddev)}
)
# Total noise = background gaussian + source poisson, sum of variance
error = np.sqrt(stddev**2 + np.maximum(star_subtracted, 0) / header["GAIN"])
# Find the fit by minimizing residual of both the Moffat model and background
fitter = TRFLSQFitter()
fitted = fitter(
    model + sky_model,
    xx[fit_mask], yy[fit_mask], star_1[fit_mask],
    weights=1 / error[fit_mask],
    maxiter=500
)
if not fitter.fit_info.get("success", False):
    raise RuntimeError("Moffat fit failed: " + str(fitter.fit_info.get("message")))

# Print and plot the fitting result
fitted_star = fitted[0]
fitted_background = fitted[1].amplitude.value
model_image = fitted(xx, yy)
residual = star_1 - model_image
weighted_rms = np.sqrt(np.mean((residual[fit_mask] / error[fit_mask])**2))

print("Local centre (x, y):", fitted_star.x_0.value, fitted_star.y_0.value)
print("Original centre (x, y):",
      x_range[0] + fitted_star.x_0.value,
      y_range[0] + fitted_star.y_0.value)
print("Background:", fitted_background)
print("Moffat alpha:", fitted_star.alpha.value)
print("Moffat beta:", fitted_star.beta.value)
print("FWHM (pixels):", fitted_star.fwhm)
print("Weighted residual RMS:", weighted_rms)
print("Diagnostic fit only; inspect residuals before using this source as a PSF.")

# Save the fitted parameters
result = Table(
    rows=[(
        x_range[0] + fitted_star.x_0.value,
        y_range[0] + fitted_star.y_0.value,
        fitted_star.alpha.value, fitted_star.beta.value,
        float(fitted_star.fwhm), fitted_background, stddev, weighted_rms
    )],
    names=("x_original", "y_original", "alpha", "beta", "fwhm_pixels",
           "background", "sky_stddev", "weighted_residual_rms")
)
result.meta["model"] = "Circular Moffat integrated over pixels, plus constant background"
result.meta["fit_radius_pixels"] = fit_radius
result.meta["crop_x"] = list(x_range)
result.meta["crop_y"] = list(y_range)
result.meta["warning"] = "Fit convergence does not confirm stellar identity or PSF suitability"
result.write(output_dir / "moffat_parameters.ecsv", overwrite=True)

# Compare the source, model, and residual
fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
vmin, vmax = -3 * stddev, np.nanmax(star_subtracted)
for ax, image, title in zip(
    axes[:2],
    [star_subtracted, model_image - fitted_background],
    ["Background-subtracted source", "Circular Moffat fit"]
):
    ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)

residual_plot = axes[2].imshow(
    residual, origin="lower", cmap="RdBu_r",
    vmin=-5 * stddev, vmax=5 * stddev
)
axes[2].set_title("Residual (clipped colour scale)")
fig.colorbar(residual_plot, ax=axes[2], label="Data - model (ADU)")

for ax in axes:
    ax.add_patch(plt.Circle((x_peak, y_peak), fit_radius,
                           fill=False, color="cyan", linewidth=0.8))
    ax.set_xlabel("Local X pixel")
    ax.set_ylabel("Local Y pixel")

fig.savefig(output_dir / "moffat_fit.png", dpi=180)
plt.show()
plt.close(fig)