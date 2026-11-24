"""Synthetic checks independent of the survey's target count."""
import numpy as np
from astropy.table import Table
from source_detection_neighbour_model import (
    Settings, coordinates, log_components, fit_profiles,
    contamination_scores, select_sources,
)


def run():
    rows = Table({'x': [19., 29.], 'y': [24., 24.],
                  'aperture_theta_deg': [0., 0.],
                  'major_sigma_pix': [2.5, 2.], 'minor_sigma_pix': [2.5, 2.],
                  'peak_adu': [3500., 3450.], 'local_background': [3400., 3400.]})
    yy, xx = np.mgrid[:49, :49]
    u2, v2 = coordinates(rows, xx, yy)
    truth = np.r_[np.log([100., 2.5, 2.5]), 0., 0., 0., 2.5,
                  np.log([50., 2., 2.]), 0., 0., 0., 2.5, .25]
    beta = 2.5
    clean = np.exp(log_components(truth, u2, v2, beta)).sum(axis=1) + truth[-1]
    rng = np.random.default_rng(184)
    data = clean + rng.normal(0, .3, len(clean))
    fit = fit_profiles(rows, xx.ravel(), yy.ravel(), data, np.full(len(clean), .3), beta, Settings())
    assert fit.success
    # Circular profiles have no unique angle: compare the predicted image.
    fitted_image = np.exp(log_components(fit.x, u2, v2, beta)).sum(axis=1) + fit.x[-1]
    assert np.sqrt(np.mean((fitted_image - clean) ** 2)) < .1
    aperture = ((xx - 19) ** 2 + (yy - 24) ** 2 <= 25).ravel()
    fitted = contamination_scores(log_components(fit.x, u2[aperture], v2[aperture], beta), 0)
    known = contamination_scores(log_components(truth, u2[aperture], v2[aperture], beta), 0)
    assert np.allclose(fitted, known, rtol=.1)
    separated = rows.copy(); separated['x'][1] = 39.
    su, sv = coordinates(separated, xx.ravel()[aperture], yy.ravel()[aperture])
    distant = contamination_scores(log_components(truth, su, sv, beta), 0)
    assert np.all(np.array(distant) > 0)
    assert np.all(np.array(distant) < known)
    t = Table({'reassessment_candidate': [False, True, True, False],
               'aperture_snr': [20., 20., 20., 20.], 'strict_usable': [True, False, False, False],
               'model_valid': [False, True, False, True], 'model_area_fraction': [np.nan, .02, .001, .001],
               'model_flux_fraction': [np.nan, .01, .001, .001]})
    assert select_sources(t, Settings(max_overlap_area_fraction=0, max_overlap_flux_fraction=0)).tolist() == [True, False, False, False]
    assert select_sources(t, Settings(max_overlap_area_fraction=.03, max_overlap_flux_fraction=.03)).tolist() == [True, True, False, False]
    print('PASS: joint model recovery, contamination estimate, positive wings, distance response, strict baseline, failed-fit veto.')
    print('Known contamination:', known, 'fitted:', fitted, 'distant:', distant)


if __name__ == '__main__':
    run()
