"""Verify the local crowded-wing review and show its before/after image."""

from pathlib import Path
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.table import Table

from reviewed_blends import apply_reviewed_blends, FLAG
from run_aperture_review import reject_shared_apertures


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'results/aperture_review'
BEFORE = BASE / 'results/aperture_before_crowded_wing_review'


def run():
    old = Table.read(BEFORE / 'sources.ecsv')
    new = Table.read(OUTPUT / 'sources.ecsv')
    np.testing.assert_array_equal(old['id'], new['id'])
    target = np.asarray(new['id']) == 4
    assert target.sum() == 1
    # Every old field of all other detections must be exactly unchanged.
    for name in old.colnames:
        if old[name].dtype.kind == 'f':
            np.testing.assert_allclose(old[name][~target], new[name][~target],
                                       rtol=0, atol=0, equal_nan=True, err_msg=name)
        else:
            np.testing.assert_array_equal(old[name][~target], new[name][~target], err_msg=name)
    for name in ['aperture_a_pix', 'aperture_b_pix', 'aperture_flux_adu']:
        np.testing.assert_array_equal(old[name], new[name])
    assert old['usable'][target][0] and not new['usable'][target][0]
    assert new['reviewed_blend_rejected'].sum() == 1
    assert FLAG in str(new['flags'][target][0])
    # Repeated local review must not grow an aperture or duplicate a flag.
    repeated, _ = apply_reviewed_blends(new.copy())
    for name in ['usable', 'flags', 'aperture_a_pix']:
        np.testing.assert_array_equal(new[name], repeated[name])
    displaced = old.copy()
    displaced['x'][target] += 5
    try:
        apply_reviewed_blends(displaced)
    except ValueError:
        pass
    else:
        raise AssertionError('A moved source must not silently receive the local exclusion.')
    assert hashlib.sha256((OUTPUT / 'detection_products.fits').read_bytes()).hexdigest() == old.meta['products_sha256']

    image = fits.getdata(BASE.parent / 'Astro/Fits_Data/mosaic.fits')
    with fits.open(OUTPUT / 'detection_products.fits') as f:
        residual = image - f['BACKGROUND'].data
        rms = f['RMS'].data.copy()
        bad = f['BADMASK'].data.astype(bool)
        labels = f['SOURCES'].data.copy()
    row = old[target][0]
    nearby = np.hypot(old['x'] - row['x'], old['y'] - row['y']) < 50
    yy, xx = np.mgrid[3810:3885, 2238:2320]
    radius = np.hypot(xx - row['x'], yy - row['y'])
    signal = residual[yy, xx]
    lab = labels[yy, xx]
    measurements = []
    for a in [float(row['aperture_a_pix']), 12., 14., 16., 18., 20., 25.]:
        aperture = radius <= a
        other = aperture & (lab > 0) & (lab != row['segment_id'])
        measurements.append((a, float(signal[aperture].sum()), int(other.sum())))
    Table(rows=measurements, names=['trial_radius_pix', 'observed_flux_adu', 'other_label_pixels']).write(
        OUTPUT / 'reviewed_blend_growth.csv', format='ascii.csv', overwrite=True)
    # A diagnostic trial, not an adopted measurement or a claim of total flux.
    trial = old[nearby].copy()
    trial['aperture_a_pix'][trial['id'] == 4] = 16.
    trial['aperture_b_pix'][trial['id'] == 4] = 16.
    _, pairs = reject_shared_apertures(trial, residual, rms, bad)
    pair = pairs[(pairs['source_id_1'] == 4) & (pairs['source_id_2'] == 776)]
    assert len(pair) == 1 and pair['pair_rejected'][0]
    pairs.write(OUTPUT / 'reviewed_blend_trial_overlap.csv', format='ascii.csv', overwrite=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    x0, x1, y0, y1 = 2248, 2318, 3825, 3880
    display = residual / rms
    for ax, table, title in zip(axes, (old, new), ('Before: recovered core aperture', 'After: excluded unresolved wings')):
        ax.imshow(display[y0:y1, x0:x1], origin='lower', cmap='gray', vmin=-1, vmax=12,
                  extent=(x0-.5, x1-.5, y0-.5, y1-.5))
        for source in table[nearby]:
            ax.add_patch(Ellipse((source['x'], source['y']), 2*source['aperture_a_pix'],
                                 2*source['aperture_b_pix'], angle=source['aperture_theta_deg'],
                                 fill=False, edgecolor='blue' if source['usable'] else 'red', linewidth=1.5))
        ax.set(xlim=(x0, x1), ylim=(y0, y1), title=title, xlabel='x / pixel', ylabel='y / pixel')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'reviewed_blend_before_after.png', dpi=160)
    plt.close(fig)

    oc = Table.read(BEFORE / 'classified_sources.ecsv')
    nc = Table.read(OUTPUT / 'classified_sources.ecsv')
    for name in ['is_star', 'count_sample', 'remaining_sample']:
        np.testing.assert_array_equal(oc[name][~target], nc[name][~target], err_msg=name)
    assert oc['remaining_sample'][target][0] and not nc['remaining_sample'][target][0]
    counts = Table.read(OUTPUT / 'number_counts.ecsv')
    # Recompute every count from the old sample minus this one source.
    mags = np.asarray(oc['magnitude'][oc['remaining_sample'] & ~target])
    limit_col = 'magnitude_limit' if 'magnitude_limit' in counts.colnames else counts.colnames[0]
    expected = np.array([np.sum(mags < m) for m in counts[limit_col]])
    np.testing.assert_array_equal(expected, counts['count'])
    print('Only ID 4 excluded. All 7037 other detections unchanged; all aperture sizes/fluxes unchanged.')
    print('Mask hash, idempotence, position guard, classification and per-bin counts passed.')
    print(pair)


if __name__ == '__main__':
    run()
