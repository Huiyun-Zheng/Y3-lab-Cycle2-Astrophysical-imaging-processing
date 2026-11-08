import numpy as np
from scipy import ndimage as ndi
from photutils.aperture import EllipticalAperture
from photutils.aperture import aperture_photometry
import matplotlib.pyplot as plt


def aperture_center_shape(data, labels, mean, stddev, target_label=None):
    """Return centers and (major, minor, theta), keyed by label ID."""

    if data.ndim != 2 or labels.shape != data.shape:
        raise ValueError("data and labels must have the same 2D shape")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must contain integer IDs")
    if not np.isfinite(mean):
        raise ValueError("Invalid background mean")

    centers = {}
    ellipse = {}
    bounds_list = ndi.find_objects(labels)

    if target_label is None:
        regions = enumerate(bounds_list, start=1)
    else:
        if not isinstance(target_label, (int, np.integer)) or target_label <= 0:
            raise ValueError("target_label must be a positive integer")
        if target_label > len(bounds_list) or bounds_list[target_label - 1] is None:
            raise ValueError(f"Label {target_label} is not present")
        regions = [(target_label, bounds_list[target_label - 1])]

    for label_id, bounds in regions:
        if bounds is None:
            continue

        # Source pixels in a local section
        y, x = np.nonzero(labels[bounds] == label_id)
        values = data[bounds][y, x]
        x = x + bounds[1].start
        y = y + bounds[0].start
        weights = np.where(np.isfinite(values), np.maximum(values - mean, 0), 0)

        if weights.sum() <= 0:
            continue

        # Centroid
        x_center = np.average(x, weights=weights)
        y_center = np.average(y, weights=weights)
        dx = x - x_center
        dy = y - y_center

        # Second moments
        cxx = np.average(dx**2, weights=weights)
        cyy = np.average(dy**2, weights=weights)
        cxy = np.average(dx * dy, weights=weights)
        covariance = np.array([[cxx, cxy], [cxy, cyy]])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0:
            continue

        sigma_minor, sigma_major = np.sqrt(eigenvalues)
        theta = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
        centers[label_id] = (float(x_center), float(y_center))
        ellipse[label_id] = (float(sigma_major), float(sigma_minor), float(theta))

    return centers, ellipse


def plot_source_aperture(data, labels, mean, stddev, target_label, scale=3.0):
    apertures = plot_aperture_scales(
        data, labels, mean, stddev, target_label, scales=(scale,)
    )
    return apertures[0]


def plot_aperture_scales(
        data, labels, mean, stddev, target_label,
        scales=(1.5, 2, 3, 4), filename=None, figsize=(8, 8)
        ):
    """Compare aperture scales on the same local image."""

    scales = np.asarray(scales, dtype=float)
    if (scales.ndim != 1 or scales.size == 0
            or not np.all(np.isfinite(scales)) or np.any(scales <= 0)):
        raise ValueError("scales must be a nonempty sequence of positive values")

    centers, ellipse = aperture_center_shape(
        data, labels, mean, stddev, target_label=target_label
    )
    if target_label not in ellipse:
        raise ValueError(f"Label {target_label} has no valid ellipse")

    x_center, y_center = centers[target_label]
    sigma_major, sigma_minor, theta = ellipse[target_label]

    # Section containing the largest aperture
    padding = 1.3 * scales.max() * sigma_major
    x_min = max(0, int(np.floor(x_center - padding)))
    x_max = min(data.shape[1], int(np.ceil(x_center + padding)) + 1)
    y_min = max(0, int(np.floor(y_center - padding)))
    y_max = min(data.shape[0], int(np.ceil(y_center + padding)) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        data[y_min:y_max, x_min:x_max], cmap="gray", origin="lower",
        extent=(x_min - 0.5, x_max - 0.5, y_min - 0.5, y_max - 0.5),
        vmin=mean - stddev, vmax=mean + 10 * stddev
    )

    apertures = []
    for i, scale in enumerate(scales):
        aperture = EllipticalAperture(
            (x_center, y_center),
            a=scale * sigma_major,
            b=scale * sigma_minor,
            theta=theta
        )
        aperture.plot(ax=ax, color=f"C{i % 10}", label=f"scale = {scale:g}")
        apertures.append(aperture)

    ax.set_xlim(x_min - 0.5, x_max - 0.5)
    ax.set_ylim(y_min - 0.5, y_max - 0.5)
    ax.set_xlabel("X Pixel")
    ax.set_ylabel("Y Pixel")
    ax.set_title(f"Aperture scales: label {target_label}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=180)
    plt.show()
    plt.close(fig)

    return apertures


def optimize_scale(data, labels, mean, stddev, target_label, show=True):
    """Return a growth curve; contaminated apertures have NaN flux."""

    scales = np.arange(1.5, 5.1, 0.5)
    fluxes = []
    centers, ellipse = aperture_center_shape(
        data, labels, mean, stddev, target_label=target_label
    )
    if target_label not in ellipse:
        raise ValueError(f"Label {target_label} has no valid ellipse")

    x_center, y_center = centers[target_label]
    sigma_major, sigma_minor, theta = ellipse[target_label]

    # Section containing the largest aperture
    padding = scales.max() * sigma_major + 1
    x_min = max(0, int(np.floor(x_center - padding)))
    x_max = min(data.shape[1], int(np.ceil(x_center + padding)) + 1)
    y_min = max(0, int(np.floor(y_center - padding)))
    y_max = min(data.shape[0], int(np.ceil(y_center + padding)) + 1)
    residual = data[y_min:y_max, x_min:x_max] - mean
    local_labels = labels[y_min:y_max, x_min:x_max]
    neighbours = (local_labels > 0) & (local_labels != target_label)
    position = (x_center - x_min, y_center - y_min)

    for scale in scales:
        aperture = EllipticalAperture(
            position,
            a=scale * sigma_major,
            b=scale * sigma_minor,
            theta=theta
        )

        # Reject partial or contaminated apertures
        extent_x = np.hypot(aperture.a * np.cos(theta), aperture.b * np.sin(theta))
        extent_y = np.hypot(aperture.a * np.sin(theta), aperture.b * np.cos(theta))
        cuts_edge = (
            x_center - extent_x < -0.5 or x_center + extent_x > data.shape[1] - 0.5
            or y_center - extent_y < -0.5 or y_center + extent_y > data.shape[0] - 0.5
        )
        footprint = aperture.to_mask(method="exact").to_image(residual.shape) > 0
        contaminated = np.any(footprint & (neighbours | ~np.isfinite(residual)))
        if cuts_edge or contaminated:
            fluxes.append(np.nan)
            continue

        result = aperture_photometry(residual, aperture)
        fluxes.append(float(result["aperture_sum"][0]))

    fluxes = np.array(fluxes)
    if show:
        fig, ax = plt.subplots()
        ax.plot(scales, fluxes, "o-")
        ax.set_xlabel("Ellipse scale")
        ax.set_ylabel("Background-subtracted flux / ADU")
        ax.set_title(f"Growth curve: label {target_label}")
        plt.show()
        plt.close(fig)

    return scales, fluxes
