"""Numerical checks for aperture and cumulative-count error propagation."""

import unittest
import numpy as np
from scipy import ndimage as ndi
from astropy.table import Table
from photometry_analysis import (
    aperture_offsets, estimate_blank_scale, add_photometric_uncertainties,
    counts_from_magnitudes, count_error_model, correlated_line_fit, descriptive_line_fit,
    ANALYSIS_SETTINGS,
)


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.row = {'x': 400.2, 'y': 400.1, 'aperture_a_pix': 5.,
                    'aperture_b_pix': 4., 'aperture_theta_deg': 31.,
                    'aperture_flux_adu': 10000., 'count_sample': True, 'id': 1}
        self.header = {'GAIN': 2., 'EXPTIME': 100., 'MAGZPT': 25., 'MAGZRR': .02}

    def test_correlated_background_has_larger_aperture_error(self):
        rng = np.random.default_rng(18)
        white = rng.normal(0, 1, (800, 800))
        correlated = ndi.gaussian_filter(white, 1)
        correlated /= correlated.std()
        dx, dy = aperture_offsets(self.row)
        mask = np.zeros(white.shape, bool)
        rms = np.ones(white.shape)
        a = estimate_blank_scale(white, rms, mask, self.row, dx, dy, np.random.default_rng(13))
        b = estimate_blank_scale(correlated, rms, mask, self.row, dx, dy, np.random.default_rng(13))
        self.assertGreaterEqual(a[1], 24)
        self.assertGreater(b[0], 1.5 * a[0])

    def test_flux_error_and_shared_zero_point_are_separate(self):
        table = Table(rows=[self.row])
        noise = np.random.default_rng(9).normal(0, 3, (800, 800))
        result, summary = add_photometric_uncertainties(
            table, noise, np.full(noise.shape, 3.), np.zeros(noise.shape, int),
            np.zeros(noise.shape, bool), self.header,
        )
        self.assertEqual(result['aperture_flux_adu'][0], 10000)
        self.assertAlmostEqual(result['source_flux_error_adu'][0]**2, 5000)
        self.assertAlmostEqual(result['magnitude_error_total'][0]**2,
                               result['magnitude_error_random'][0]**2 + .02**2)
        self.assertGreater(result['magnitude_error_upper'][0], result['magnitude_error_lower'][0])

    def test_strict_counts_and_nonpositive_flux_draws(self):
        np.testing.assert_array_equal(counts_from_magnitudes([19., 20., np.inf], [19., 20., 21.]), [0, 1, 2])

    def test_zero_photometric_errors_recover_poisson_covariance(self):
        header = {**self.header, 'MAGZRR': 0.}
        flux = np.array([100., 500., 1000.])
        limits = np.array([21., 23., 25.])
        mags = header['MAGZPT'] - 2.5 * np.log10(flux * header['GAIN'] / header['EXPTIME'])
        counts = counts_from_magnitudes(mags, limits)
        simulated, poisson, photometric = count_error_model(flux, np.zeros(3), header, limits, counts, draws=100)
        np.testing.assert_array_equal(photometric, np.zeros((3, 3)))
        np.testing.assert_array_equal(poisson, np.minimum.outer(counts, counts))

    def test_gls_known_line_with_correlated_errors(self):
        x = np.arange(19., 24.5, .5)
        y = .47 * x - 6
        covariance = .002 * np.minimum.outer(np.arange(1, len(x) + 1), np.arange(1, len(x) + 1))
        result = correlated_line_fit(x, y, covariance)
        self.assertAlmostEqual(result['slope'], .47, places=10)
        self.assertGreater(result['slope_error_formal'], 0)

    def test_descriptive_slope_error_matches_correlated_simulation(self):
        x = np.arange(19., 24.5, .5)
        covariance = .002 * np.minimum.outer(np.arange(1, len(x) + 1), np.arange(1, len(x) + 1))
        result = descriptive_line_fit(x, .47 * x - 6, covariance)
        perturbations = np.random.default_rng(71).multivariate_normal(np.zeros(len(x)), covariance, size=10000)
        slopes = perturbations @ ((x - x.mean()) / np.sum((x - x.mean())**2))
        self.assertAlmostEqual(result['slope'], .47, places=10)
        self.assertLess(abs(slopes.std() / result['slope_error_propagated'] - 1), .03)

    def test_relaxed_settings_on_synthetic_stars_and_extended_sources(self):
        from test_moffat_method import MoffatTests
        experiment = MoffatTests()
        experiment.setUp()
        experiment.settings = ANALYSIS_SETTINGS
        experiment.test_full_synthetic_catalogue_keeps_extended_and_blended()


if __name__ == '__main__':
    unittest.main()
