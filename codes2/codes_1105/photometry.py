import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from photutils.aperture import EllipticalAperture
from aperture_conversion import aperture_center_shape

def fixed_aperture_photometry(data, labels, mean, stddev, scale=3.0, mask=None):
    """Measure fixed-scale ellipses; flagged fluxes are NaN."""

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != data.shape:
            raise ValueError("mask and data must have the same shape")

    centers, ellipse = aperture_center_shape(data, labels, mean, stddev)
    height, width = data.shape
    rows = []

    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        if label_id not in ellipse:
            rows.append((label_id, *([np.nan] * 6), 8))
            continue

        x_center, y_center = centers[label_id]
        sigma_major, sigma_minor, theta = ellipse[label_id]
        a = scale * sigma_major
        b = scale * sigma_minor
        aperture = EllipticalAperture((x_center, y_center), a, b, theta=theta)

        # Check the full aperture
        extent_x = np.hypot(a * np.cos(theta), b * np.sin(theta))
        extent_y = np.hypot(a * np.sin(theta), b * np.cos(theta))
        cuts_edge = (
            x_center - extent_x < -0.5 or x_center + extent_x > width - 0.5
            or y_center - extent_y < -0.5 or y_center + extent_y > height - 0.5
        )
        flag = 1 if cuts_edge else 0

        # Work only inside the aperture bounding box
        aperture_mask = aperture.to_mask(method="exact")
        image_slice, mask_slice = aperture_mask.get_overlap_slices(data.shape)
        weights = aperture_mask.data[mask_slice]    # counts to what fraction (0-1) the pixel is inside the aperture
        footprint = weights > 0
        values = data[image_slice]
        local_labels = labels[image_slice]

        neighbours = (local_labels > 0) & (local_labels != label_id)
        if np.any(footprint & neighbours):
            flag |= 2

        bad_pixels = ~np.isfinite(values)
        if mask is not None:    # exclude any source whose aperture includes masked pixels
            bad_pixels = bad_pixels | mask[image_slice]
        if np.any(footprint & bad_pixels):
            flag |= 4

        flux = np.nan
        if flag == 0:
            flux = np.sum((values[footprint] - mean) * weights[footprint])

        rows.append((label_id, x_center, y_center, a, b, theta, flux, flag))

    results = Table(
        rows=rows,
        names=("label", "x_center", "y_center", "a", "b", "theta", "flux", "flag"),
        dtype=("i8", "f8", "f8", "f8", "f8", "f8", "f8", "i4")
    )
    results.meta["scale"] = float(scale)
    results.meta["background"] = float(mean)
    results.meta["flags"] = "1=edge, 2=neighbour, 4=masked/nonfinite, 8=invalid ellipse"
    results["theta"].unit = "rad"
    results["flux"].description = "Background-subtracted sum in input image units"
    return results

def AB_calibration(result, header, bins=100):

    valid = (
        (result['flag'] == 0)
        & np.isfinite(result['flux'])
        & (result['flux'] > 0)
    )

    valid_data = result['flux'][valid]  # ADU
    counts = valid_data * header['GAIN'] # number, dimensionless
    exposure_time = header['EXPTIME']
    count_rate = counts / exposure_time  # number/s

    ZP = header['MAGZPT']
    ZP_error = header['MAGZRR']

    magnitudes = ZP - 2.5 * np.log10(count_rate)

    # Zero-point uncertainty only for now
    magnitudes_zp_error = np.full(magnitudes.shape, ZP_error)

    # plot the result
    plt.hist(magnitudes, bins=bins)
    plt.title(f'Magnitude distribution of {magnitudes.shape} valid sources')
    plt.ylabel('Number of sources')
    plt.xlabel('Magnitude')
    plt.grid()
    plt.show()

    return magnitudes, magnitudes_zp_error

def population_above_magnitude(magnitudes):
    mag_thresholds = np.linspace(0, max(magnitudes), 20)
    populations = []
    for m in mag_thresholds:
        pop = len(magnitudes[magnitudes < m])
        populations.append(pop)

    populations = np.asarray(populations)
    positive = populations > 0
    plt.plot(mag_thresholds[positive], np.log10(populations[positive]))
    plt.title('N - m relationship')
    plt.xlabel('Magnitude Thresholds')
    #plt.xlim((0, 20))
    plt.ylabel('log10 N(m < threshold)')
    plt.show()

    return mag_thresholds, populations