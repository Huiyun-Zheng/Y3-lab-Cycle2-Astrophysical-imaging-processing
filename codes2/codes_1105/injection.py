import numpy as np

def elliptical_gaussian(
        shape, center, flux, sigma_major, sigma_minor, theta=0
        ):
    """Return a Gaussian source in ADU; theta is in radians from +x."""

    height, width = shape
    x_center, y_center = center

    yy, xx = np.ogrid[:height, :width]
    dx = xx - x_center
    dy = yy - y_center

    # Coordinates along the ellipse axes
    x_rot = dx * np.cos(theta) + dy * np.sin(theta)
    y_rot = -dx * np.sin(theta) + dy * np.cos(theta)

    # Continuous total flux, sampled at pixel centres
    amplitude = flux / (2 * np.pi * sigma_major * sigma_minor)
    source = amplitude * np.exp(
        -0.5 * ((x_rot / sigma_major)**2 + (y_rot / sigma_minor)**2)
    )

    return source