"""Small numerical checks for the neighbour-light review."""

import unittest

import numpy as np
from astropy.table import Table

import source_detection_neighbour_model as model


class NeighbourModelTests(unittest.TestCase):
    def test_rotated_profiles_and_derivatives(self):
        rows = Table({
            'x': [0., 9.], 'y': [0., 6.],
            'aperture_theta_deg': [25., 0.],
            'major_sigma_pix': [2., 1.], 'minor_sigma_pix': [1.5, 1.],
            'peak_adu': [110., 15.], 'local_background': [10., 10.],
        })
        yy, xx = np.mgrid[-12:16, -12:19]
        u, v = model.coordinates(rows, xx, yy)
        truth = np.r_[
            np.log(100.), np.log(3.), np.log(2.), .4, -.3, .2, 3.1,
            np.log(5.), np.log(1.5), np.log(1.2), 0., 0., -.1, 2.1, .2,
        ]

        def prediction(parameters):
            components = model.log_components(parameters, u, v, 2.5)
            return np.exp(components).sum(axis=1) + parameters[-1]

        data = prediction(truth)
        error = np.full(len(data), 2.)
        fit = model.fit_profiles(rows, xx.ravel(), yy.ravel(), data, error,
                                 2.5, model.Settings())
        self.assertTrue(fit.success)
        np.testing.assert_allclose(prediction(fit.x), data, atol=1e-4)

        # Check every derivative against an independent finite difference.
        numerical = np.empty_like(fit.jac)
        for k in range(len(fit.x)):
            delta = np.zeros_like(fit.x)
            delta[k] = 1e-5
            numerical[:, k] = (prediction(fit.x + delta) - prediction(fit.x - delta)) / (2e-5 * error)
        np.testing.assert_allclose(fit.jac, numerical, atol=1e-6, rtol=1e-5)

    def test_thresholds_and_faint_source_cut(self):
        table = Table({
            'strict_usable': [True, True, False, False, False],
            'reassessment_candidate': [False, False, True, True, True],
            'model_valid': [False, False, True, True, False],
            'model_area_fraction': [np.nan, np.nan, .01, .05, .01],
            'model_flux_fraction': [np.nan, np.nan, .02, .08, .01],
            'aperture_snr': [20., 3., 30., 40., 50.],
        })
        counts = []
        for limit in [0., .01, .03, .1]:
            settings = model.Settings(max_overlap_area_fraction=limit,
                                      max_overlap_flux_fraction=limit)
            counts.append(int(model.select_sources(table, settings).sum()))
        self.assertEqual(counts, [1, 1, 2, 3])

    def test_aperture_noise_includes_source_shot_noise(self):
        table = Table({
            'x': [16.], 'y': [16.], 'aperture_a_pix': [2.],
            'aperture_b_pix': [2.], 'aperture_theta_deg': [0.],
            'usable': [True], 'flags': ['none'], 'aperture_flux_adu': [100.],
            'local_background': [0.], 'near_artefact': [False],
            'near_boundary': [False], 'aperture_capped': [False],
            'elongation': [1.], 'blended': [False],
            'aperture_contaminated': [False], 'parent_id': [1],
        })
        signal = np.zeros((33, 33))
        signal[16, 16] = 100.
        noise = np.full(signal.shape, 2.)
        result, _ = model.score_catalogue(table, signal, noise,
                                          np.zeros(signal.shape, bool), 36000.,
                                          model.Settings(), gain=2.)
        # This radius-two binary circle contains thirteen pixels.
        expected_error = np.sqrt(13 * 2. ** 2 + 100. / 2.)
        self.assertAlmostEqual(result['aperture_flux_error_adu'][0], expected_error)
        self.assertAlmostEqual(result['aperture_snr'][0], 100. / expected_error)

        # Proximity of the segment alone is advisory. A truly masked aperture
        # must still be rejected before any model recovery can be attempted.
        table['usable'] = [False]
        table['near_artefact'] = [True]
        clean = np.zeros(signal.shape, bool)
        reviewed, _ = model.score_catalogue(table, signal, noise, clean, 36000.,
            model.Settings(), gain=2., only_parents=[])
        self.assertTrue(reviewed['reassessment_candidate'][0])
        self.assertTrue(np.isfinite(reviewed['aperture_snr'][0]))
        clean[16,16] = True
        masked, _ = model.score_catalogue(table, signal, noise, clean, 36000.,
            model.Settings(), gain=2., only_parents=[])
        self.assertFalse(masked['reassessment_candidate'][0])
        self.assertEqual(masked['model_status'][0], 'hard_quality_failure')

    def test_observed_overlap_veto_applies_to_old_usable(self):
        yy, xx = np.mgrid[:80, :80]
        signal = (100*np.exp(-((xx-32)**2+(yy-40)**2)/8)
                  + 100*np.exp(-((xx-42)**2+(yy-40)**2)/8))
        labels = np.where(xx < 37, 1, 2)
        rows = Table(dict(id=[1,2], segment_id=[1,2], x=[32.,42.], y=[40.,40.],
                          peak_x=[32,42], peak_y=[40,40], aperture_a_pix=[7.,7.],
                          aperture_b_pix=[7.,7.], aperture_theta_deg=[0.,0.]))
        checked, pairs = model.image_quality_checks(rows, signal, np.ones_like(signal),
            labels, np.zeros_like(signal, bool), model.Settings())
        self.assertTrue(np.all(checked['shared_light_rejected']))
        self.assertTrue(np.any(pairs['rejected']))
        checked['strict_usable'] = [True, True]
        checked['reassessment_candidate'] = [False, False]
        checked['model_valid'] = [False, False]
        checked['model_area_fraction'] = [np.nan, np.nan]
        checked['model_flux_fraction'] = [np.nan, np.nan]
        checked['aperture_snr'] = [100., 100.]
        self.assertFalse(np.any(model.select_sources(checked, model.Settings())))

    def test_clean_single_and_internal_double(self):
        yy, xx = np.mgrid[:80, :80]
        single = 100*np.exp(-((xx-35)**2+(yy-40)**2)/8)
        labels = np.ones(single.shape, int)
        args = (np.ones_like(single), labels, 1, 35., 40., 18., 12., 0., model.Settings())
        self.assertFalse(model.has_internal_pair(single, *args))
        double = single + 70*np.exp(-((xx-43)**2+(yy-40)**2)/5.12)
        self.assertTrue(model.has_internal_pair(double, *args))
        weak = single + 2*np.exp(-((xx-43)**2+(yy-40)**2)/5.12)
        self.assertFalse(model.has_internal_pair(weak, *args))


if __name__ == '__main__':
    unittest.main()
