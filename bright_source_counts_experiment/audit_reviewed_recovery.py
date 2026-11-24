"""Check that the seven-screenshot review has only the intended local effects."""

from pathlib import Path
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.table import Table

from reviewed_recovery import match_positions, recovery_decision, POSITIONS


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'results/aperture_review'
BEFORE = BASE / 'results/aperture_before_seven_source_review'


def check_decisions():
    row = dict(near_boundary=False, aperture_capped=False, elongation=1.,
               neighbour_core_rejected=False, neighbour_peak_inside=False,
               shared_aperture_rejected=True, shared_aperture_flux_fraction=.012)
    fit = dict(model_valid=True, model_status='ok', usable=True,
               model_flux_fraction=.004, model_score_disagreement=.001, aperture_snr=100.)
    assert recovery_decision(row, fit, True, False, False)[0]
    for clean, partner, wings in [(False, False, False), (True, True, False), (True, False, True)]:
        assert not recovery_decision(row, fit, clean, partner, wings)[0]
    for key, value in [('neighbour_core_rejected', True), ('neighbour_peak_inside', True),
                       ('shared_aperture_flux_fraction', .03)]:
        assert not recovery_decision(dict(row, **{key: value}), fit, True, False, False)[0]
    for key, value in [('model_valid', False), ('model_flux_fraction', .04),
                       ('model_score_disagreement', .02)]:
        assert not recovery_decision(row, dict(fit, **{key: value}), True, False, False)[0]


def run():
    check_decisions()
    old = Table.read(BEFORE / 'sources.ecsv')
    new = Table.read(OUTPUT / 'sources.ecsv')
    np.testing.assert_array_equal(old['id'], new['id'])
    recovered = np.isin(new['id'], [91, 154, 243])
    np.testing.assert_array_equal(new['local_review_recovered'], recovered)
    assert not old['usable'][recovered].any() and new['usable'][recovered].all()
    for name in old.colnames:
        if old[name].dtype.kind == 'f':
            np.testing.assert_allclose(old[name][~recovered], new[name][~recovered],
                                       rtol=0, atol=0, equal_nan=True, err_msg=name)
        else:
            np.testing.assert_array_equal(old[name][~recovered], new[name][~recovered], err_msg=name)
    for name in ['x', 'y', 'aperture_a_pix', 'aperture_b_pix', 'aperture_theta_deg', 'aperture_flux_adu']:
        np.testing.assert_allclose(old[name], new[name], rtol=0, atol=0, equal_nan=True, err_msg=name)
    assert not new['usable'][np.isin(new['id'], [4, 7, 102, 223])].any()
    assert new['usable'][np.isin(new['id'], [6, 14])].all()
    assert hashlib.sha256((OUTPUT / 'detection_products.fits').read_bytes()).hexdigest() == old.meta['products_sha256']
    indices = match_positions(new)
    displaced = new.copy()
    displaced['x'][indices[0]] += 5
    try:
        match_positions(displaced)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed source positions must stop local review.')

    oc = Table.read(BEFORE / 'classified_sources.ecsv')
    nc = Table.read(OUTPUT / 'classified_sources.ecsv')
    for name in ['count_sample', 'is_star', 'remaining_sample']:
        np.testing.assert_array_equal(oc[name][~recovered], nc[name][~recovered], err_msg=name)
    counts = Table.read(OUTPUT / 'number_counts.ecsv')
    mags = np.asarray(nc['magnitude'][nc['remaining_sample']])
    np.testing.assert_array_equal(counts['count'], [np.sum(mags < m) for m in counts['limiting_magnitude']])

    image = fits.getdata(BASE.parent / 'Astro/Fits_Data/mosaic.fits')
    with fits.open(OUTPUT / 'detection_products.fits') as f:
        display = (image - f['BACKGROUND'].data) / f['RMS'].data
    labels = ['1 / 4: unstable neighbour fit', '2: clean measurement aperture',
              '3: weak neighbour wings', '5: neighbour core inside aperture',
              '6: weak neighbour wings', '7: outer wings still unresolved']
    fig, axes = plt.subplots(6, 2, figsize=(10, 23))
    for k, (index, description) in enumerate(zip(indices, labels)):
        r = new[index]
        half = int(max(28, r['aperture_a_pix'] + 12))
        x0, x1 = int(r['x'])-half, int(r['x'])+half+1
        y0, y1 = int(r['y'])-half, int(r['y'])+half+1
        near = np.hypot(new['x']-r['x'], new['y']-r['y']) < 2*half
        for j, (table, stage) in enumerate([(old, 'Before'), (new, 'After')]):
            ax = axes[k, j]
            ax.imshow(display[y0:y1, x0:x1], origin='lower', cmap='gray', vmin=-1, vmax=12,
                      extent=(x0-.5, x1-.5, y0-.5, y1-.5))
            for s in table[near]:
                ax.add_patch(Ellipse((s['x'], s['y']), 2*s['aperture_a_pix'], 2*s['aperture_b_pix'],
                    angle=s['aperture_theta_deg'], fill=False,
                    edgecolor='blue' if s['usable'] else 'red', linewidth=1.3))
            ax.set(xlim=(x0, x1), ylim=(y0, y1), title=f"{stage} | ID {int(r['id'])} | {description}")
            ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'reviewed_sources_before_after.png', dpi=130)
    plt.close(fig)
    print('Recovered only IDs 91, 154, 243. All 7035 other detection rows unchanged.')
    print('Every aperture/flux and mask unchanged; position/quality guards and per-bin counts passed.')
    print(nc[recovered][['id', 'magnitude', 'is_star', 'remaining_sample']])


if __name__ == '__main__':
    run()
