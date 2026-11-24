"""Compare the aperture review with its saved version before the quality changes."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'results' / 'aperture_review'
BEFORE = BASE / 'results' / 'aperture_before_quality_review'


def comparison_panel(axes, position, half, old, new, residual, manual):
    x, y = position
    x0, x1 = max(0, int(x) - half), min(residual.shape[1], int(x) + half + 1)
    y0, y1 = max(0, int(y) - half), min(residual.shape[0], int(y) + half + 1)
    cut = np.arcsinh(residual[y0:y1, x0:x1] / 15)
    for ax, table, title in zip(axes, (old, new), ('Before', 'After')):
        ax.imshow(cut, origin='lower', cmap='gray', vmin=-1,
                  vmax=max(2, np.percentile(cut, 99.5)),
                  extent=(x0 - .5, x1 - .5, y0 - .5, y1 - .5))
        nearby = (np.abs(table['x'] - x) < half) & (np.abs(table['y'] - y) < half)
        for row in table[nearby]:
            ax.add_patch(Ellipse(
                (row['x'], row['y']), 2 * row['aperture_a_pix'],
                2 * row['aperture_b_pix'], angle=row['aperture_theta_deg'],
                fill=False, color='blue' if row['usable'] else 'red', lw=1.1,
            ))
        if title == 'After' and manual[y0:y1, x0:x1].any():
            ax.contour(np.arange(x0, x1), np.arange(y0, y1),
                       manual[y0:y1, x0:x1], levels=[.5], colors=['gold'], linewidths=1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(f'{title}: ({x:.0f}, {y:.0f})', fontsize=10)
        ax.tick_params(labelsize=7)


def plot_shared_examples(new, residual, rms):
    """Show the two matched user examples using the overview's display stretch."""
    previous = BASE / 'results' / 'aperture_before_shared_overlap_review' / 'sources.ecsv'
    if not previous.exists():
        return
    old = Table.read(previous)
    np.testing.assert_array_equal(old['id'], new['id'])
    np.testing.assert_array_equal(old['x'], new['x'])
    np.testing.assert_array_equal(old['aperture_a_pix'], new['aperture_a_pix'])
    assert not np.any(new['usable'] & ~old['usable'])
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for pair, (x, y) in zip(axes, ((369, 3590), (415, 3711))):
        x0, x1, y0, y1 = x - 26, x + 27, y - 26, y + 27
        for ax, table, title in zip(pair, (old, new), ('Before', 'After')):
            ax.imshow(residual[y0:y1, x0:x1] / rms[y0:y1, x0:x1],
                      origin='lower', cmap='gray', vmin=-1, vmax=12,
                      extent=(x0 - .5, x1 - .5, y0 - .5, y1 - .5))
            selected = (np.abs(table['x'] - x) < 26) & (np.abs(table['y'] - y) < 26)
            for row in table[selected]:
                ax.add_patch(Ellipse(
                    (row['x'], row['y']), 2 * row['aperture_a_pix'],
                    2 * row['aperture_b_pix'], angle=row['aperture_theta_deg'],
                    fill=False, color='blue' if row['usable'] else 'red', lw=1.6,
                ))
                ax.text(row['x'], row['y'] + row['aperture_a_pix'] + 1,
                        str(row['id']), color='orange', fontsize=9)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_title(title)
            ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'blended_examples_before_after.png', dpi=150)
    plt.close(fig)
    print('Newly excluded by shared-light rule:', int(np.sum(old['usable'] & ~new['usable'])))


def run():
    old = Table.read(BEFORE / 'sources.ecsv')
    new = Table.read(OUTPUT / 'sources.ecsv')
    with fits.open(OUTPUT / 'detection_products.fits') as products:
        residual = fits.getdata(BASE.parent / 'Astro/Fits_Data/mosaic.fits') - products['BACKGROUND'].data
        manual = products['MANUALMASK'].data.astype(bool)
        rms = products['RMS'].data.copy()
    plot_shared_examples(new, residual, rms)
    tree = cKDTree(np.column_stack([new['x'], new['y']]))
    distance, indices = tree.query(np.column_stack([old['x'], old['y']]))
    matched = distance < 2
    audit = Table({
        'old_id': old['id'], 'new_id': new['id'][indices],
        'x': old['x'], 'y': old['y'], 'match_distance': distance,
        'matched': matched, 'old_usable': old['usable'],
        'new_usable': matched & new['usable'][indices],
        'old_a': old['aperture_a_pix'], 'new_a': new['aperture_a_pix'][indices],
        'new_flags': new['flags'][indices],
    })
    audit.write(OUTPUT / 'quality_changes.csv', format='ascii.csv', overwrite=True)

    # First show the exact manually identified defect, then representative
    # core overlaps and the largest reductions among previously usable sources.
    locations = [(1036., 432., 45)]
    for row in new[new['usable_before_core_check'] & new['neighbour_core_rejected']][:3]:
        locations.append((float(row['x']), float(row['y']), max(25, int(row['aperture_a_pix']) + 10)))
    shrink = np.asarray(old['aperture_a_pix'] - new['aperture_a_pix'][indices])
    shrink[~(matched & old['usable'] & new['usable'][indices])] = -np.inf
    for i in np.argsort(shrink)[::-1][:4]:
        locations.append((float(old['x'][i]), float(old['y'][i]), max(25, int(old['aperture_a_pix'][i]) + 8)))
    for page, start in enumerate(range(0, len(locations), 4), start=1):
        group = locations[start:start + 4]
        fig, axes = plt.subplots(len(group), 2, figsize=(10, 4 * len(group)), squeeze=False)
        for pair, (x, y, half) in zip(axes, group):
            comparison_panel(pair, (x, y), half, old, new, residual, manual)
        fig.tight_layout()
        fig.savefig(OUTPUT / f'quality_before_after_{page}.png', dpi=140)
        plt.close(fig)

    print('Candidates / usable before:', len(old), int(old['usable'].sum()))
    print('Candidates / usable after:', len(new), int(new['usable'].sum()))
    print('Matched usable sources with radius reduced by > 1 pixel:', int(np.sum(shrink > 1)))
    print('Additional core exclusions:', int(np.sum(new['usable_before_core_check'] & new['neighbour_core_rejected'])))
    for position in ((1048., 472.), (1028., 403.)):
        d, i = tree.query(position)
        print('Neighbour of manual defect:', position, 'distance', float(d), 'usable', bool(new['usable'][i]))


if __name__ == '__main__':
    run()
