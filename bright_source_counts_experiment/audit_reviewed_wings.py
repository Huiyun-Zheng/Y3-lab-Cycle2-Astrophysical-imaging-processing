"""Check that the local wing review changes only the two requested apertures."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.table import Table

from reviewed_wings import growth_radius


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'results' / 'aperture_review'
BEFORE = BASE / 'results' / 'aperture_before_local_wing_review'


def run():
    # A known winged profile grows; noise-free blank wings and a dirty
    # reference circle must not cause growth.
    yy, xx = np.mgrid[-31:32, -31:32]
    radius = np.hypot(xx, yy)
    angle = np.arctan2(yy, xx)
    noise = np.ones(radius.shape)
    clean = np.ones(radius.shape, bool)
    signal = 1000 * (1 + (radius / 3.4) ** 2) ** -2.5
    adopted, status, _ = growth_radius(signal, noise, radius, angle, clean, 6.)
    assert status == 'accepted' and 6 < adopted < 15
    blank_wings = signal.copy()
    blank_wings[radius > 6] = 0
    assert growth_radius(blank_wings, noise, radius, angle, clean, 6.)[0] is None
    clean[31, 40] = False
    assert growth_radius(signal, noise, radius, angle, clean, 6.)[0] is None

    old = Table.read(BEFORE / 'sources.ecsv')
    new = Table.read(OUTPUT / 'sources.ecsv')
    np.testing.assert_array_equal(old['id'], new['id'])
    target = np.isin(new['id'], [6, 14])
    assert target.sum() == 2
    for field in old.colnames:
        # Every existing column of every other source must be unchanged.
        if old[field].dtype.kind == 'f':
            np.testing.assert_allclose(new[field][~target], old[field][~target],
                                       rtol=0, atol=0, equal_nan=True, err_msg=field)
        else:
            np.testing.assert_array_equal(new[field][~target], old[field][~target], err_msg=field)
    np.testing.assert_array_equal(new['usable'], old['usable'])
    assert np.all(new['aperture_a_pix'][target] > old['aperture_a_pix'][target])
    assert np.all(new['aperture_flux_adu'][target] > old['aperture_flux_adu'][target])
    assert old.meta['products_sha256'] == new.meta['products_sha256']

    image = fits.getdata(BASE.parent / 'Astro/Fits_Data/mosaic.fits')
    with fits.open(OUTPUT / 'detection_products.fits') as products:
        display = (image - products['BACKGROUND'].data) / products['RMS'].data
    fig, axes = plt.subplots(1, 2, figsize=(8, 11))
    x0, x1, y0, y1 = 542, 629, 4197, 4344
    for ax, table, title in zip(axes, (old, new), ('Before', 'After')):
        ax.imshow(display[y0:y1, x0:x1], origin='lower', cmap='gray', vmin=-1, vmax=12,
                  extent=(x0 - .5, x1 - .5, y0 - .5, y1 - .5))
        nearby = ((table['x'] >= x0) & (table['x'] < x1)
                  & (table['y'] >= y0) & (table['y'] < y1))
        for row in table[nearby]:
            ax.add_patch(Ellipse(
                (row['x'], row['y']), 2 * row['aperture_a_pix'],
                2 * row['aperture_b_pix'], angle=row['aperture_theta_deg'],
                fill=False, edgecolor='blue' if row['usable'] else 'red', linewidth=1.5,
            ))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(title)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'reviewed_wings_before_after.png', dpi=150)
    plt.close(fig)

    old_class = Table.read(BEFORE / 'classified_sources.ecsv')
    new_class = Table.read(OUTPUT / 'classified_sources.ecsv')
    np.testing.assert_array_equal(old_class['is_star'], new_class['is_star'])
    np.testing.assert_array_equal(old_class['remaining_sample'], new_class['remaining_sample'])
    old_counts = Table.read(BEFORE / 'number_counts.ecsv')
    new_counts = Table.read(OUTPUT / 'number_counts.ecsv')
    np.testing.assert_array_equal(old_counts['count'], new_counts['count'])
    print('Only IDs 6 and 14 changed aperture/flux. All other detections, masks,')
    print('source selection, star classification and cumulative counts are unchanged.')
    print(Table.read(OUTPUT / 'reviewed_wing_apertures.csv', format='ascii.csv'))


if __name__ == '__main__':
    run()
