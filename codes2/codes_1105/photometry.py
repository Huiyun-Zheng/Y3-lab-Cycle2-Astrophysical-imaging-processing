import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from photutils.aperture import EllipticalAperture
from aperture_conversion import aperture_center_shape
from astropy.visualization import ZScaleInterval
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse, Patch

def photometry_from_detection(catalogue):
    """Reuse source_detection_copy measurements in the photometry table format.

    No flux is remeasured and no background is subtracted here. Keep every
    candidate, including NaN fluxes and rejected candidates, for recovery.
    ``label`` refers to the segmentation ID, not the brightness-ranked ID.
    ``flag=16`` represents the detector's quality rejection; see ``flags``
    for its reasons. Bits 1, 2, 4 and 8 retain their old photometry meanings.
    """
    aliases = {
        'label': 'segment_id',
        'x_center': 'x',
        'y_center': 'y',
        'a': 'aperture_a_pix',
        'b': 'aperture_b_pix',
        'flux': 'aperture_flux_adu',
    }
    required = set(aliases.values()) | {
        'aperture_theta_deg', 'usable', 'flags',
    }
    missing = sorted(required - set(catalogue.colnames))
    if missing:
        raise ValueError(f'Detection catalogue is missing columns: {missing}')

    results = catalogue.copy()
    for name, source in aliases.items():
        results[name] = catalogue[source].copy()
    results['theta'] = np.deg2rad(catalogue['aperture_theta_deg'])
    results['theta'].unit = 'rad'
    results['flag'] = np.where(catalogue['usable'], 0, 16).astype(np.int32)
    results['flux'].description = (
        'Reused background-subtracted aperture flux in input image units'
    )
    results.meta['method'] = 'Reused source_detection_copy aperture photometry'
    results.meta['flux_column'] = 'aperture_flux_adu'
    results.meta['background_subtracted'] = True
    results.meta['flags'] = (
        '16=detection quality rejection; see the original flags column. '
        'No old aperture flag bits are inferred.'
    )
    return results


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

def adjustable_aperture(data, labels, mean, stddev, scale=3.0, mask=None):
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



def adjustable_aperture(data, labels, mean, stddev, scales=None, mask=None):
    """Choose the valid scale with maximum background-subtracted flux."""

    if scales is None:
        scales = np.arange(1.5, 5.1, 0.5)
    scales = np.asarray(scales, dtype=float)
    if (scales.ndim != 1 or scales.size == 0
            or not np.all(np.isfinite(scales)) or np.any(scales <= 0)):
        raise ValueError("scales must be a nonempty sequence of positive values")
    scales = np.unique(scales)

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
            rows.append((label_id, *([np.nan] * 6), 8, np.nan, False))
            continue

        x_center, y_center = centers[label_id]
        sigma_major, sigma_minor, theta = ellipse[label_id]
        best_flux = -np.inf
        best_scale = np.nan
        best_a = scales[0] * sigma_major
        best_b = scales[0] * sigma_minor
        failed_flags = 0

        for scale in scales:
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

            # Pixel coverage weights
            aperture_mask = aperture.to_mask(method='exact')
            image_slice, mask_slice = aperture_mask.get_overlap_slices(data.shape)
            weights = aperture_mask.data[mask_slice]
            footprint = weights > 0
            values = data[image_slice]
            local_labels = labels[image_slice]

            neighbours = (local_labels > 0) & (local_labels != label_id)
            if np.any(footprint & neighbours):
                flag |= 2

            bad_pixels = ~np.isfinite(values)
            if mask is not None:
                bad_pixels = bad_pixels | mask[image_slice]
            if np.any(footprint & bad_pixels):
                flag |= 4

            if flag != 0:
                failed_flags |= flag
                continue

            # Keep the largest net flux
            flux = np.sum((values[footprint] - mean) * weights[footprint])
            if flux > best_flux:
                best_flux = flux
                best_scale = scale
                best_a, best_b = a, b

        if np.isnan(best_scale):
            best_flux = np.nan
            flag = failed_flags
        else:
            flag = 0

        rows.append((
            label_id, x_center, y_center, best_a, best_b, theta,
            best_flux, flag, best_scale, best_scale == scales[-1]
        ))

    results = Table(
        rows=rows,
        names=(
            'label', 'x_center', 'y_center', 'a', 'b', 'theta',
            'flux', 'flag', 'scale', 'at_max_scale'
        ),
        dtype=('i8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'i4', 'f8', 'bool')
    )
    results.meta['method'] = 'maximum valid background-subtracted flux'
    results.meta['scales'] = scales.tolist()
    results.meta['background'] = float(mean)
    results.meta['flags'] = '1=edge, 2=neighbour, 4=masked/nonfinite, 8=invalid ellipse'
    results['theta'].unit = 'rad'
    results['flux'].description = 'Maximum valid net flux in input image units'
    results['scale'].description = 'Chosen scale; NaN if every candidate failed'
    results['at_max_scale'].description = 'Chosen scale equals the search upper limit'
    return results


def plot_aperture_catalogue(
        data, results, x_range=None, y_range=None,
        figsize=(10, 16), filename=None, show=True
        ):
    """Plot measured ellipses on the image, coloured by catalogue selection."""

    height, width = data.shape
    x_min, x_max = (0, width) if x_range is None else x_range
    y_min, y_max = (0, height) if y_range is None else y_range
    x_min, x_max = max(0, int(x_min)), min(width, int(x_max))
    y_min, y_max = max(0, int(y_min)), min(height, int(y_max))
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Plot ranges must overlap the image")

    x = np.asarray(results['x_center'])
    y = np.asarray(results['y_center'])
    a = np.asarray(results['a'])
    b = np.asarray(results['b'])
    theta = np.asarray(results['theta'])
    flux = np.asarray(results['flux'])
    usable = (results['flag'] == 0) & np.isfinite(flux) & (flux > 0)

    # Keep ellipses intersecting this section
    extent_x = np.hypot(a * np.cos(theta), b * np.sin(theta))
    extent_y = np.hypot(a * np.sin(theta), b * np.cos(theta))
    visible = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
        & np.isfinite(theta) & (a > 0) & (b > 0)
        & (x + extent_x >= x_min - 0.5) & (x - extent_x <= x_max - 0.5)
        & (y + extent_y >= y_min - 0.5) & (y - extent_y <= y_max - 0.5)
    )

    section = data[y_min:y_max, x_min:x_max]
    vmin, vmax = ZScaleInterval().get_limits(section)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        section, cmap='gray', origin='lower', vmin=vmin, vmax=vmax,
        extent=(x_min - 0.5, x_max - 0.5, y_min - 0.5, y_max - 0.5),
        interpolation='nearest'
    )

    # Draw all outlines in two collections
    handles = []
    for selected, color, label in (
            (usable, '#00bfff', 'Used for magnitudes'),
            (~usable, '#ff7f0e', 'Excluded')):
        indices = np.flatnonzero(visible & selected)
        patches = [
            Ellipse((x[i], y[i]), 2 * a[i], 2 * b[i], angle=np.rad2deg(theta[i]))
            for i in indices
        ]
        ax.add_collection(PatchCollection(
            patches, facecolors='none', edgecolors=color, linewidths=0.65
        ))
        handles.append(Patch(
            facecolor='none', edgecolor=color, label=f'{label}: {len(indices)}'
        ))

    ax.set_xlim(x_min - 0.5, x_max - 0.5)
    ax.set_ylim(y_min - 0.5, y_max - 0.5)
    ax.set_xlabel('X Pixel (cropped image)')
    ax.set_ylabel('Y Pixel (cropped image)')
    ax.set_title(f'{np.count_nonzero(visible)} detected sources with fitted ellipses')
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.065))
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=220, bbox_inches='tight')
    if show:
        plt.show()
        plt.close(fig)

    return fig, ax


def AB_calibration(result, header, bins=100, *, show=True, filename=None):

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
    fig, ax = plt.subplots()
    ax.hist(magnitudes, bins=bins)
    ax.set_title(f'Magnitude distribution of {magnitudes.size} valid sources')
    ax.set_ylabel('Number of sources')
    ax.set_xlabel('Magnitude')
    ax.grid()
    if filename is not None:
        fig.savefig(filename, dpi=220, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    return magnitudes, magnitudes_zp_error

def population_above_magnitude(magnitudes):
    mag_thresholds = np.linspace(0, max(magnitudes), 50)
    populations = []
    for m in mag_thresholds:
        pop = len(magnitudes[magnitudes < m])
        populations.append(pop)

    populations = np.asarray(populations)
    positive = populations > 0
    plt.plot(mag_thresholds[positive], np.log10(populations[positive]))
    plt.title('N - m relationship')
    plt.xlabel('Magnitude Thresholds')
    plt.xlim((10, 30))
    plt.ylabel('log10 N(m < threshold)')
    plt.show()

    return mag_thresholds, populations
