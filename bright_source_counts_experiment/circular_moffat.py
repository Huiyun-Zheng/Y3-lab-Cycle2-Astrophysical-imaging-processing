"""Circular Moffat fitting and local width classification"""

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.optimize import least_squares


# Three-point Gaussian-Legendre quadrature in each pixel: nine evaluations per pixel.
# Generate 1D quadrature nodes and weights
_nodes, _weights = np.polynomial.legendre.leggauss(3)

# Generate nine coordinate offsets within a pixel.
# Dividing by two maps [-1, 1] to the pixel interval [-0.5, 0.5].
offsets = []
for x in _nodes:
    for y in _nodes:
        offsets.append((x / 2, y / 2))
_offsets = np.array(offsets)

# Generate nine 2D weights in the same order as the coordinate offsets.
# Each direction contributes a factor of 1/2, giving 1/4 in two dimensions.
_pixel_weights = np.outer(_weights, _weights).ravel() / 4


def moffat_pixels(x, y, amplitude, x_center, y_center, alpha, beta, background):
    """Integrate a circular Moffat over each pixel using nine Gauss-Legendre points."""

    # Keep the input shape so that both pixel lists and coordinate grids work.
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    shape = x.shape

    # Each row contains the nine sample offsets from the same source centre.
    dx = x.ravel()[:, None] + _offsets[:, 0] - x_center
    dy = y.ravel()[:, None] + _offsets[:, 1] - y_center
    r2 = dx**2 + dy**2
    profile = (1 + r2 / alpha**2) ** (-beta)

    # Sum the nine weighted values for each pixel, then restore the input shape.
    # @ is the matrix mutiplication operator
    return background + amplitude * (profile @ _pixel_weights).reshape(shape)


def moffat_fwhm(alpha, beta):
    return 2 * alpha * np.sqrt(2 ** (1 / beta) - 1)


def measure_psf_widths(data, labels, results, header, mask=None, mag_limit=22):
    """
    Fit a circular Moffat profile to each selected object.

    Calculate the FWHM, its uncertainty, and other fitting
    quantities without modifying the original photometry
    catalogue.

    Inputs:
        data : 2D array
            Original astronomical image containing pixel intensities.
        labels : 2D array
            Segmentation map containing the object ID of each pixel.
            A value of zero means that the pixel is not assigned to any detected object.
        results : astropy.table.Table
            Existing photometry catalogue.
        header : FITS header
            Observational information, including gain, exposure time and magnitude zero point.
        mask : 2D boolean array, optional
            Mask of bad pixels. True means the pixel is excluded.
        mag_limit : float, optional
            Faintest magnitude allowed for fitting.

    Returns:
        catalogue : astropy.table.Table
            A copy of the original catalogue with additional Moffat fitting results.
    """

    if data.shape != labels.shape:
        raise ValueError("data and labels must have the same shape")
    if mask is not None and np.shape(mask) != data.shape:
        raise ValueError("mask and data must have the same shape")

    catalogue = results.copy()
    n = len(catalogue)

    for name in ("magnitude", "psf_fwhm", "fwhm_error", "fit_snr", "reduced_chi2", "moffat_beta",
                 "profile_residual_fraction", "core_flux_fraction", "middle_flux_fraction",
                 "moment_radius", "moment_ellipticity", "morphology_snr"):
        catalogue[name] = np.full(n, np.nan)
    catalogue["fit_status"] = np.full(n, "not_fitted", dtype="U24")

    gain = float(header["GAIN"])
    saturation = float(header.get("SATLEVEL", 36000))

    valid = (
        np.asarray(catalogue["count_sample"], dtype=bool)
        & np.isfinite(catalogue["aperture_flux_adu"])
        & (catalogue["aperture_flux_adu"] > 0)
    )
    catalogue["magnitude"][valid] = header["MAGZPT"] - 2.5 * np.log10(catalogue["aperture_flux_adu"][valid] * gain / header["EXPTIME"])

    height, width = data.shape

    selected_rows = np.flatnonzero(valid & (catalogue["magnitude"] <= mag_limit))
    # number: position in the selected object list
    # i: row index in the original catalogue.
    for number, i in enumerate(selected_rows):
        # Print the fitting progress every 100 objects.
        if number % 100 == 0:
            # flush=True displays the message immediately.
            print(f"Circular Moffat fits: {number}/{len(selected_rows)}", flush=True)

        row = catalogue[i]
        xc, yc = float(row["x"]), float(row["y"])  # These coordinates may contain decimal values.
        if not np.isfinite(xc + yc):
            continue

        # Round the centre coordinates to the nearest integer.
        # Integer coordinates are needed for array slicing.
        x0, y0 = int(round(xc)), int(round(yc))
        if x0 < 30 or y0 < 30 or x0 + 30 >= width or y0 + 30 >= height:
            catalogue["fit_status"][i] = "edge"
            continue

        # Extract a 61*61 pixel section from the image.
        section = data[y0 - 30 : y0 + 31, x0 - 30 : x0 + 31]
        section_labels = labels[y0 - 30 : y0 + 31, x0 - 30 : x0 + 31]
        yy, xx = np.mgrid[y0 - 30 : y0 + 31, x0 - 30 : x0 + 31]
        rr = np.hypot(xx - xc, yy - yc)  # radial distance of each pixel centre from the estimated object centre.

        good = np.isfinite(section)
        if mask is not None:
            good &= ~np.asarray(mask[y0 - 30 : y0 + 31, x0 - 30 : x0 + 31], dtype=bool)
        # Check whether any bad pixel lies within 3 pixels of the estimated object centre.
        if np.any(~good & (rr < 3)):
            # Reject the fit because the core is unreliable
            catalogue["fit_status"][i] = "masked_core"
            continue

        # Find pixels that belong to another detected object.
        neighbours = (section_labels > 0) & (section_labels != row["segment_id"])
        # Check whether another object enters the core region.
        if np.any(neighbours & (rr < 3)):
            # Reject heavily crowded objects
            catalogue["fit_status"][i] = "crowded"
            continue

        # Select background pixels in an annulus around the object, between radii of 22 and 30 pixels.
        sky_pixels = good & (rr >= 22) & (rr <= 30) & (section_labels == 0)
        # Require at least 50 available background pixels.
        if sky_pixels.sum() < 50:
            catalogue["fit_status"][i] = "no_local_sky"
            continue

        # sky: median background level
        # noise: standard deviation of the background pixels
        _, sky, noise = sigma_clipped_stats(section[sky_pixels], sigma=3)

        # Use good pixels within 9 pixels of the object centre for the profile fitting.
        fit_pixels = good & ~neighbours & (rr <= 9)
        # Require at least 100 fitting pixels and a finite. positive background noise estimate.
        if fit_pixels.sum() < 100 or not np.isfinite(noise) or noise <= 0:
            catalogue["fit_status"][i] = "insufficient_pixels"
            continue
        if np.any(section[rr <= 9] >= saturation):
            catalogue["fit_status"][i] = "saturated"
            continue

        x, y, values = xx[fit_pixels], yy[fit_pixels], section[fit_pixels]
        error = np.sqrt(noise**2 + np.maximum(values - sky, 0) / gain)

        # Compare inner and outer light using the same apertures for every source.
        # Do not infer concentration if neighbours or bad pixels remove much area.
        inner = rr <= 3
        outer = rr <= 9
        if (np.count_nonzero(fit_pixels & inner) >= .9 * inner.sum()
                and fit_pixels.sum() >= .9 * outer.sum()):
            inner_flux = float((section - sky)[fit_pixels & inner].sum())
            outer_flux = float((section - sky)[fit_pixels].sum())
            if outer_flux > 0:
                catalogue['core_flux_fraction'][i] = inner_flux / outer_flux
                catalogue['middle_flux_fraction'][i] = float((section - sky)[fit_pixels & (rr <= 5)].sum()) / outer_flux
                catalogue['morphology_snr'][i] = outer_flux / np.sqrt(np.sum(error ** 2))
                weights = np.maximum(values - sky, 0)
                centre_x = np.average(x, weights=weights)
                centre_y = np.average(y, weights=weights)
                dx, dy = x - centre_x, y - centre_y
                moments = np.array([[np.average(dx ** 2, weights=weights), np.average(dx * dy, weights=weights)],
                                    [np.average(dx * dy, weights=weights), np.average(dy ** 2, weights=weights)]])
                eigenvalues = np.linalg.eigvalsh(moments)
                catalogue['moment_radius'][i] = np.sqrt(np.sum(eigenvalues))
                catalogue['moment_ellipticity'][i] = 1 - np.sqrt(eigenvalues[0] / eigenvalues[1])

        peak = max(float(np.max(values) - sky), 1)
        initial = [peak, xc, yc, 4.0, 2.5, float(sky)]  # Initial parameter guesses
        lower = [0, xc - 2, yc - 2, 0.5, 1.1, sky - 5 * noise]  # Minimum allowed value of each parameter
        upper = [np.inf, xc + 2, yc + 2, 20, 20, sky + 5 * noise]  # Maximum allowed value of each parameter
        try:
            fit = least_squares(
                lambda p: (moffat_pixels(x, y, *p) - values) / error,
                initial,  # Initial parameter guesses
                bounds=(lower, upper),  # Allowed parameter ranges
                x_scale="jac",  # Adapt parameter scaling using the Jacobian
                max_nfev=100,  # Limit the number of function evaluations counted by the optimiser
            )
            # Reject the fit if the optimisation fails or if any parameter reaches an active bound.
            if not fit.success or np.any(fit.active_mask):
                catalogue["fit_status"][i] = "fit_limit"
                continue

            # Calculate parameter uncertainties.
            covariance = np.linalg.inv(fit.jac.T @ fit.jac)
            errors = np.sqrt(np.diag(covariance))
            if not np.all(np.isfinite(errors)) or errors[0] <= 0:
                continue

            # Calculate the FWHM.
            alpha, beta = fit.x[3:5]
            fwhm = moffat_fwhm(alpha, beta)
            # Calculate the partial derivatives of the FWHM with respect to alpha and beta.
            gradient = np.array(
                [
                    fwhm / alpha,
                    -alpha * np.log(2) * 2 ** (1 / beta) / (beta**2 * np.sqrt(2 ** (1 / beta) - 1)),
                ]
            )
            variance = gradient @ covariance[3:5, 3:5] @ gradient

            catalogue["psf_fwhm"][i] = fwhm
            catalogue["fwhm_error"][i] = np.sqrt(max(variance, 0))
            catalogue["moffat_beta"][i] = beta
            catalogue["fit_snr"][i] = fit.x[0] / errors[0]
            catalogue["reduced_chi2"][i] = np.sum(fit.fun**2) / (len(values) - 6)
            model_values = moffat_pixels(x, y, *fit.x)
            # Estimate model-shape mismatch as a fraction of the fitted signal.
            # Subtract the expected noise power, but never return negative power.
            excess = max(float(np.sum((values - model_values) ** 2 - error ** 2)), 0.0)
            signal_power = float(np.sum((model_values - fit.x[-1]) ** 2))
            if signal_power > 0:
                catalogue['profile_residual_fraction'][i] = np.sqrt(excess / signal_power)
            catalogue["fit_status"][i] = "ok"
        except (ValueError, np.linalg.LinAlgError):
            catalogue["fit_status"][i] = "fit_failed"

    catalogue["psf_fwhm"].unit = "pix"
    catalogue["fwhm_error"].unit = "pix"
    catalogue.meta["width_method"] = (
        "circular Moffat with 3x3 Gauss-Legendre pixel integration, radius 9 pixels"
    )
    return catalogue


def select_psf_candidates(catalogue, reference_labels=None, mag_limit=21, quality_mode='reference_profile'):
    """
    Classify objects by comparing their measured FWHM with the local PSF width.

    Inputs:
        catalogue : astropy.table.Table
            Photometry catalogue containing Moffat fitting results.
        reference_labels : array-like, optional
            Segment IDs of manually selected reference stars.
            If None, reference candidates are selected automatically from the peak of the FWHM histogram.
        mag_limit : float, optional
            Faintest magnitude allowed for classification.

    Returns:
        catalogue : astropy.table.Table
            A copy of the input catalogue with local PSF widths,
            width tolerances and candidate classifications.
    """

    catalogue = catalogue.copy()
    fwhm = np.asarray(catalogue["psf_fwhm"])
    error = np.asarray(catalogue["fwhm_error"])

    # Create a boolean array identifying objects that can be considered for classification.
    quality = (
        (catalogue["fit_status"] == "ok")
        & (catalogue["fit_snr"] >= 50)
        & (catalogue["magnitude"] <= mag_limit)
        & np.isfinite(fwhm)
        & np.isfinite(error)
    )

    if reference_labels is None:
        # Start with the objects that pass the quality checks.
        # Apply a stricter SNR requirement for reference objects.
        pool = quality & (catalogue["fit_snr"] >= 100)
        pool &= (catalogue["reduced_chi2"] < 10) & (fwhm >= 2) & (fwhm <= 9)
        if np.count_nonzero(pool) < 5:
            raise ValueError("Too few reference candidates; supply checked reference_labels")

        bins = np.arange(2, 9.26, 0.25)
        # Build a histogram of the FWHM values in the pool.
        histogram, edges = np.histogram(fwhm[pool], bins=bins)
        # Find the index of the bin containing the most objects.
        j = int(np.argmax(histogram))
        # Calculate the centre of the most populated bin.
        # Use this as an initial estimate of the stellar width.
        mode = (edges[j] + edges[j + 1]) / 2
        references = pool & (np.abs(fwhm - mode) <= 0.4)
    else:
        references = quality & np.isin(catalogue["segment_id"], reference_labels)
    if np.count_nonzero(references) < 5:
        raise ValueError("At least five usable reference candidates are required")

    catalogue["psf_reference"] = references
    catalogue["local_psf_fwhm"] = np.full(len(catalogue), np.nan)
    catalogue["width_tolerance"] = np.full(len(catalogue), np.nan)
    catalogue["source_class"] = np.full(len(catalogue), "unknown", dtype="U24")

    reference_rows = np.flatnonzero(references)

    # Set a fitting quality threshold.
    # Use a minimum threshold of 3.0.
    q_limit = max(3.0, 2 * np.percentile(catalogue["reduced_chi2"][references], 90))
    if quality_mode not in ('original', 'reference_profile', 'empirical_profile'):
        raise ValueError('Unknown PSF quality mode.')
    reference_residuals = np.asarray(catalogue['profile_residual_fraction'][references])
    reference_residuals = reference_residuals[np.isfinite(reference_residuals)]
    profile_limit = float(1.5 * np.percentile(reference_residuals, 90))
    catalogue.meta['profile_residual_limit'] = profile_limit
    catalogue.meta['psf_quality_mode'] = quality_mode
    if quality_mode == 'empirical_profile':
        # Match observed reference-star profiles even if an analytic Moffat fails.
        # Brightness is used only to require a reliable shape measurement.
        empirical = (catalogue['count_sample'] & (catalogue['magnitude'] <= mag_limit)
                     & (catalogue['morphology_snr'] >= 50))
        features = ['core_flux_fraction', 'middle_flux_fraction', 'moment_radius', 'moment_ellipticity']
        for i in np.flatnonzero(empirical):
            other = reference_rows[reference_rows != i]
            distance = np.hypot(catalogue['x'][other] - catalogue['x'][i],
                                catalogue['y'][other] - catalogue['y'][i])
            nearby = other[np.argsort(distance)[:8]]
            if empirical_profile_match(catalogue[i], catalogue[nearby]):
                catalogue['source_class'][i] = 'star_candidate'
            catalogue['local_psf_fwhm'][i] = np.median(fwhm[nearby])
            catalogue['width_tolerance'][i] = .15 * catalogue['local_psf_fwhm'][i]
        catalogue.meta['classification_mag_limit'] = float(mag_limit)
        catalogue.meta['reference_selection'] = 'supplied labels' if reference_labels is not None else 'automatic width peak'
        catalogue.meta['classification_warning'] = 'Empirical point-source candidates; compact galaxies remain ambiguous. Reference cross-validation is not an external truth catalogue.'
        return catalogue
    # Classify the suitable object
    for i in np.flatnonzero(quality):
        other = reference_rows[reference_rows != i]
        distance = np.hypot(
            catalogue["x"][other] - catalogue["x"][i], catalogue["y"][other] - catalogue["y"][i]
        )
        # Select up to eight nearest reference objects.
        nearby = other[np.argsort(distance)[:8]]
        # Extract the measured FWHM of these nearby references.
        local_widths = fwhm[nearby]

        psf_width = np.median(local_widths)
        # 1.4826 converts MAD into an estimate comparable to the standard deviation of a Gaussian distribution.
        scatter = 1.4826 * np.median(np.abs(local_widths - psf_width))

        # Calculate the allowed width difference.
        tolerance = max(0.15 * psf_width, 3 * np.hypot(scatter, error[i]))

        catalogue["local_psf_fwhm"][i] = psf_width
        catalogue["width_tolerance"][i] = tolerance

        # Check the Moffat fitting quality.
        profile_ok = (np.isfinite(catalogue['profile_residual_fraction'][i])
                      and catalogue['profile_residual_fraction'][i] <= profile_limit)
        if catalogue["reduced_chi2"][i] > q_limit and not (quality_mode == 'reference_profile' and profile_ok):
            continue

        if quality_mode == 'reference_profile':
            # A broad confidence interval must not turn an uncertain galaxy into a star.
            if error[i] > .10 * fwhm[i]:
                continue
            reference_core = np.asarray(catalogue['core_flux_fraction'][nearby])
            reference_core = reference_core[np.isfinite(reference_core)]
            if len(reference_core) < 3 or not np.isfinite(catalogue['core_flux_fraction'][i]):
                continue
            core_median = np.median(reference_core)
            core_scatter = 1.4826 * np.median(np.abs(reference_core - core_median))
            core_tolerance = max(.10 * core_median, 3 * core_scatter)
            if abs(catalogue['core_flux_fraction'][i] - core_median) > core_tolerance:
                continue

        # Compare the object width with the local PSF.
        difference = fwhm[i] - psf_width
        if abs(difference) <= tolerance:
            catalogue["source_class"][i] = "star_candidate"
        elif difference > tolerance:
            catalogue["source_class"][i] = "extended_candidate"

    catalogue.meta["classification_warning"] = (
        "Width-based candidates only. Automatic reference selection is provisional; "
        "inspect references and candidates. Compact galaxies can be star candidates. "
        "Thresholds are initial choices, not calibrated classification probabilities."
    )
    catalogue.meta["classification_mag_limit"] = float(mag_limit)
    if reference_labels is None:
        catalogue.meta["reference_selection"] = "automatic width peak"
    else:
        catalogue.meta["reference_selection"] = "supplied labels"

    return catalogue


def empirical_profile_match(row, references):
    """Require agreement in two flux ratios, radial size and ellipticity."""
    features = ['core_flux_fraction', 'middle_flux_fraction', 'moment_radius', 'moment_ellipticity']
    for name in features:
        if not np.isfinite(row[name]):
            return False
        values = np.asarray(references[name], float)
        values = values[np.isfinite(values)]
        if len(values) < 5:
            return False
        centre = np.median(values)
        scatter = 1.4826 * np.median(np.abs(values - centre))
        # Small floors avoid treating a small reference set as perfectly precise.
        floor = .05 if name == 'moment_ellipticity' else .05 * abs(centre)
        tolerance = max(3 * scatter, floor)
        if abs(row[name] - centre) > tolerance:
            return False
    return True
