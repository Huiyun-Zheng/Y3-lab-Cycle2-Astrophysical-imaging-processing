"""Check the FWHM parameterisation, its derivatives and synthetic fit recovery."""

from unittest.mock import patch

import numpy as np
from astropy.table import Table

import aperture_neighbour_model as model


def run():
    rows = Table({
        'x': [0.], 'y': [0.], 'aperture_theta_deg': [0.],
        'major_sigma_pix': [2.5], 'minor_sigma_pix': [2.],
        'peak_adu': [100.], 'local_background': [0.],
    })
    yy, xx = np.mgrid[-15:16, -15:16]
    u, v = model.coordinates(rows, xx.ravel(), yy.ravel())
    parameters = np.array([np.log(100), np.log(4), np.log(3), .2, -.3, .27, 3., 1.])
    data = np.exp(model.log_components(parameters, u, v, 3.)).sum(axis=1) + parameters[-1]
    real_solver = model.least_squares

    def checked_solver(residual, initial, **kwargs):
        # Check at a rotated, off-centre profile so cross terms are exercised.
        trial = initial.copy()
        trial[3:6] = [.2, -.3, .27]
        numeric = np.empty((len(data), len(trial)))
        for k in range(len(trial)):
            step = np.zeros(len(trial))
            step[k] = 1e-5
            numeric[:, k] = (residual(trial + step) - residual(trial - step)) / (2e-5)
        np.testing.assert_allclose(kwargs['jac'](trial), numeric, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(np.exp(kwargs['bounds'][1][1:3]), [7.5, 6.])
        return real_solver(residual, initial, **kwargs)

    with patch.object(model, 'least_squares', checked_solver):
        fit = model.fit_profiles(rows, xx.ravel(), yy.ravel(), data,
                                 np.ones(data.shape), 2.5, model.Settings())
    assert fit.success
    np.testing.assert_allclose(fit.fun, 0, atol=1e-5)
    np.testing.assert_allclose(np.exp(fit.x[1:3]), [4., 3.], rtol=1e-5)

    # At half the major-axis FWHM the profile must be half its peak value.
    parameters[3:6] = 0
    half_peak = np.exp(model.log_components(
        parameters, np.array([[2.]]), np.array([[0.]]), 3.,
    ))[0, 0]
    np.testing.assert_allclose(half_peak, 50.)
    print('FWHM, analytic derivatives, width bounds and synthetic recovery passed.')


if __name__ == '__main__':
    run()
