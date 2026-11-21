"""Numerical and selection checks; no astronomical truth labels are assumed."""

import unittest
import numpy as np
from astropy.table import Table

from moffat_method import (
    MoffatSettings, moffat_pixels, moffat_fwhm, fit_free_profile,
    fit_local_profile, equivalence_test, local_reference, make_stamp,
    measure_profiles, classify_sources,
)


class MoffatTests(unittest.TestCase):
    def setUp(self):
        self.settings = MoffatSettings()
        self.template = np.array([np.log(2000), .13, -.21, np.log(4), 2.7, .04, -.03, 100])

    def stamp(self, scale=1, seed=19):
        yy, xx = np.mgrid[-12:13, -12:13]
        rr = np.hypot(xx - .13, yy + .21)
        selected = rr <= 12
        truth = self.template.copy()
        truth[3] += np.log(scale)
        values = moffat_pixels(xx[selected], yy[selected], truth)
        error = np.sqrt(4**2 + np.maximum(values - 100, 0) / 3.1)
        values += np.random.default_rng(seed).normal(0, error)
        return {'x': xx[selected], 'y': yy[selected], 'values': values,
                'error': error, 'radius': rr[selected], 'sky': 100.,
                'noise': 4., 'centre': (.13, -.21)}, truth

    def test_free_profile_recovers_width(self):
        stamp, truth = self.stamp()
        fit, covariance, chi2 = fit_free_profile(stamp, self.settings)
        self.assertLess(abs(moffat_fwhm(fit.x) / moffat_fwhm(truth) - 1), .03)
        self.assertTrue(.5 < chi2 < 1.5)
        self.assertTrue(np.all(np.diag(covariance) > 0))

    def test_point_matches_but_extended_does_not(self):
        for scale, expected in [(1., True), (1.4, False)]:
            for seed in range(5):
                stamp, truth = self.stamp(scale, seed)
                ratio, error, chi2, residual, parameters = fit_local_profile(stamp, self.template, self.settings)
                self.assertLess(abs(ratio - scale), .04)
                self.assertEqual(bool(equivalence_test(ratio, error, .01, self.settings)), expected)

    def test_uncertainty_never_makes_acceptance_easier(self):
        decisions = [equivalence_test(1.03, error, .01, self.settings)
                     for error in [.001, .01, .03, .1, .3]]
        self.assertTrue(decisions[0])
        self.assertFalse(decisions[-1])
        self.assertTrue(np.all(np.diff(np.asarray(decisions, int)) <= 0))

    def test_local_reference_is_leave_one_out_and_bounded(self):
        n = 7
        table = Table({'x': np.arange(n) * 20., 'y': np.zeros(n),
                       'psf_fwhm': np.full(n, moffat_fwhm(self.template)),
                       'fwhm_error': np.full(n, .01),
                       'moffat_log_alpha': np.full(n, self.template[3]),
                       'moffat_beta': np.full(n, self.template[4]),
                       'moffat_e1': np.full(n, self.template[5]),
                       'moffat_e2': np.full(n, self.template[6])})
        refs = np.ones(n, bool)
        local = local_reference(table, 0, refs, self.settings)
        self.assertNotIn(0, local[0])
        table['x'][0] = 10000
        self.assertIsNone(local_reference(table, 0, refs, self.settings))

    def test_bad_core_is_not_fit(self):
        image = np.full((81, 81), 100.)
        labels = np.zeros(image.shape, int)
        mask = np.zeros(image.shape, bool)
        mask[40, 40] = True
        stamp, status = make_stamp(image, labels, mask, {'x': 40., 'y': 40., 'segment_id': 1},
                                  3.1, 36000, self.settings)
        self.assertIsNone(stamp)
        self.assertEqual(status, 'masked_or_crowded_core')

    def test_full_synthetic_catalogue_keeps_extended_and_blended(self):
        rng = np.random.default_rng(12)
        image = 100 + rng.normal(0, 4, (400, 400))
        labels = np.zeros(image.shape, int)
        positions = [(x, y) for y in [70, 190, 310] for x in [70, 190, 310]]
        n = len(positions)
        for i, (x, y) in enumerate(positions):
            yy, xx = np.mgrid[y - 30:y + 31, x - 30:x + 31]
            p = self.template.copy()
            p[0] = np.log(5000)
            p[1:3] = x + .13, y - .21
            p[-1] = 0
            if i == n - 1:
                p[3] += np.log(1.4)
            signal = moffat_pixels(xx, yy, p).reshape(xx.shape)
            image[y - 30:y + 31, x - 30:x + 31] += signal + rng.normal(0, np.sqrt(signal / 3.1))
            cutout = labels[y - 30:y + 31, x - 30:x + 31]
            cutout[np.hypot(xx - p[1], yy - p[2]) <= 8] = i + 1
        table = Table({'id': np.arange(1, n + 1), 'segment_id': np.arange(1, n + 1),
                       'x': np.array(positions)[:, 0] + .13,
                       'y': np.array(positions)[:, 1] - .21,
                       'count_sample': np.ones(n, bool), 'magnitude': np.full(n, 19.),
                       'blended': np.arange(n) == n - 2})
        header = {'GAIN': 3.1, 'SATLEVEL': 36000}
        mask = np.zeros(image.shape, bool)
        measured = measure_profiles(image, labels, mask, table, header, self.settings)
        classified, _ = classify_sources(image, labels, mask, measured, header, self.settings)
        self.assertGreaterEqual(np.count_nonzero(classified['is_star'][:-2]), 5)
        self.assertFalse(classified['is_star'][-1])
        self.assertFalse(classified['is_star'][-2])
        self.assertEqual(classified['classification_reason'][-2], 'crowded_keep')


if __name__ == '__main__':
    unittest.main()
