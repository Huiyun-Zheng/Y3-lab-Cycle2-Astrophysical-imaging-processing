"""Check masks, measured fluxes, references, and protected original files."""

import sys
sys.dont_write_bytecode = True

from pathlib import Path
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from matplotlib.patches import Ellipse
from scipy.spatial import cKDTree

import detection_aperture_review as detector
import aperture_neighbour_model as neighbour
from test_aperture_neighbour_model import run as check_profile_model
from run_aperture_review import reject_neighbour_cores, reject_shared_apertures


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'results' / 'aperture_review'


def plot_aperture_changes(table, residual):
    """Show old and adopted apertures on exactly the same pixels."""
    selected = table[table['usable'] & (table['aperture_method'] == 'bright_kron')]
    selected = selected[np.argsort(selected['segment_flux_adu'])[::-1]][:12]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, row in zip(axes.flat, selected):
        x, y = float(row['x']), float(row['y'])
        a, b = float(row['aperture_a_pix']), float(row['aperture_b_pix'])
        old_a = float(row['original_aperture_a_pix'])
        old_b = old_a * b / a
        half = int(max(20, a + 6))
        x0, x1 = max(0, int(x) - half), min(residual.shape[1], int(x) + half + 1)
        y0, y1 = max(0, int(y) - half), min(residual.shape[0], int(y) + half + 1)
        cut = np.arcsinh(residual[y0:y1, x0:x1] / 15)
        ax.imshow(cut, origin='lower', cmap='gray', vmin=-1,
                  vmax=max(2, np.percentile(cut, 99.5)),
                  extent=(x0 - .5, x1 - .5, y0 - .5, y1 - .5))
        for aa, bb, colour, style in [(old_a, old_b, 'orange', '--'), (a, b, 'royalblue', '-')]:
            ax.add_patch(Ellipse((x, y), 2 * aa, 2 * bb,
                                 angle=row['aperture_theta_deg'], fill=False,
                                 edgecolor=colour, linestyle=style, linewidth=1.1))
        ax.set_title(f'ID {row["id"]}: {old_a:.1f} → {a:.1f} pixels', fontsize=9)
    fig.suptitle('Orange: original aperture; blue: adopted aperture', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'aperture_comparison.png', dpi=150)
    plt.close(fig)


def check_sharp_controls():
    """A PSF-sized source must survive; a one-pixel spike must be masked."""
    yy, xx = np.mgrid[:61, :61]
    noise = np.ones((61, 61))
    excluded = np.zeros((61, 61), bool)
    settings = detector.Settings()
    for amplitude in (20., 100., 1000.):
        # Both circular and elongated PSFs have a resolved minor axis.
        for sx, sy in ((1.6, 1.6), (1.6, 4.0)):
            signal = amplitude * np.exp(-0.5 * (((xx - 30.2) / sx) ** 2
                                                 + ((yy - 29.7) / sy) ** 2))
            mask, _ = detector.sharp_feature_mask(signal, noise, excluded, settings)
            assert not mask.any(), (amplitude, sx, sy)
    spike = np.zeros((61, 61))
    spike[30, 30] = 100.
    mask, _ = detector.sharp_feature_mask(spike, noise, excluded, settings)
    assert mask[30, 30]
    streak = np.zeros((61, 61))
    streak[25:36, 30] = 100.
    mask, _ = detector.sharp_feature_mask(streak, noise, excluded, settings)
    assert np.all(mask[25:36, 30])


def check_shared_aperture_controls():
    """Separate real common light from touching outlines and empty overlaps."""
    yy, xx = np.mgrid[:51, :51]
    noise = np.ones(xx.shape)
    bad = np.zeros(xx.shape, bool)

    def pair(separation):
        return Table({
            'id': [1, 2], 'x': [20., 20. + separation], 'y': [25., 25.],
            'aperture_a_pix': [5., 5.], 'aperture_b_pix': [5., 5.],
            'aperture_theta_deg': [0., 0.], 'usable': [True, True],
            'flags': ['none', 'none'],
        })

    close = pair(8.)
    light = sum(100 * np.exp(-((xx - x) ** 2 + (yy - 25) ** 2) / 8)
                for x in close['x'])
    selected, audit = reject_shared_apertures(close.copy(), light, noise, bad)
    assert np.all(selected['shared_aperture_rejected'])
    assert not np.any(selected['usable'])
    assert len(audit) == 1 and audit['pair_rejected'][0]

    # A prior failure cannot be undone by this quality check.
    failed = close.copy()
    failed['usable'][0] = False
    checked, _ = reject_shared_apertures(failed, np.zeros(xx.shape), noise, bad)
    np.testing.assert_array_equal(checked['usable'], [False, True])

    # Tangent binary apertures have at most one common pixel, not a bright bridge.
    touching, _ = reject_shared_apertures(pair(10.), light, noise, bad)
    assert np.all(touching['usable'])
    loose, _ = reject_shared_apertures(close.copy(), light, noise, bad, flux_limit=1.)
    assert np.all(loose['usable'])


def run():
    check_profile_model()
    check_sharp_controls()
    check_shared_aperture_controls()
    # A strict-source flag must not bypass the core-overlap check.
    control = Table({
        'x': [15., 15., 15.], 'y': [15., 15., 15.],
        'aperture_a_pix': [2., 7., 7.], 'aperture_b_pix': [2., 7., 7.],
        'aperture_theta_deg': [0., 0., 0.], 'segment_id': [1, 1, 1],
        'usable': [True, True, False], 'flags': ['none', 'none', 'other_failure'],
    })
    signal = np.zeros((31, 31))
    labels = np.zeros((31, 31), int)
    signal[14:17, 14:17] = 100
    labels[14:17, 14:17] = 1
    signal[14:17, 20:22] = 100
    labels[14:17, 20:22] = 2
    checked = reject_neighbour_cores(control, signal, np.ones(signal.shape), labels)
    np.testing.assert_array_equal(checked['usable'], [True, False, False])
    # A known Moffat profile should gain real wing flux, not just a larger circle.
    yy, xx = np.mgrid[:101, :101]
    radius = np.hypot(xx - 50, yy - 50)
    signal = 1000 * (1 + (radius / 3.4) ** 2) ** -2.5
    test_labels = np.zeros(signal.shape, int)
    test_labels[radius < 6] = 1
    new_radius, method = detector.bright_kron_aperture(
        signal, np.full(signal.shape, 5.), test_labels,
        np.zeros(signal.shape, bool), 1, 50., 50., 2.1, 2.1, 0., False,
        detector.Settings(),
    )
    assert method == 'bright_kron' and 5.25 < new_radius < 12
    assert signal[radius <= new_radius].sum() > signal[radius <= 5.25].sum()
    for line in (BASE / 'inputs/original_file_hashes.tsv').read_text().splitlines()[1:]:
        name, expected = line.split('\t')
        assert hashlib.sha256((BASE.parent / name).read_bytes()).hexdigest() == expected, name

    image, header = fits.getdata(BASE.parent / 'Astro/Fits_Data/mosaic.fits', header=True)
    with fits.open(OUTPUT / 'detection_products.fits') as products:
        bad = products['BADMASK'].data.astype(bool)
        mask = products['DETMASK'].data.astype(bool)
        sharp = products['SHARPMASK'].data.astype(bool)
        manual = products['MANUALMASK'].data.astype(bool)
        labels = products['SOURCES'].data.copy()
        residual = image - products['BACKGROUND'].data
    assert np.all(bad[:, 2450:]) and np.all(mask[:, 2446:])
    assert np.all(bad[sharp]) and np.all(mask[sharp])
    assert manual[432, 1036] and manual[426, 1028]
    assert np.all(bad[manual]) and np.all(mask[manual])
    assert not manual[472, 1048] and not manual[403, 1028]
    assert not np.any(labels[mask])

    table = Table.read(OUTPUT / 'classified_sources.ecsv')
    assert np.array_equal(table['count_sample'], table['usable'])
    selected = np.asarray(table['usable'], bool)
    assert not np.any(table['neighbour_core_rejected'][selected])
    shared_selected = selected & table['shared_aperture_rejected']
    if np.any(shared_selected):
        # A recorded local review may keep the brighter aperture when modelled
        # contamination is small and the overlapping neighbours remain excluded.
        assert np.all(table['local_review_recovered'][shared_selected])
        assert np.all(table['local_review_aperture_clean'][shared_selected])
        assert np.all(table['review_model_flux_fraction'][shared_selected] <= .03)
        assert np.all(table['review_model_score_disagreement'][shared_selected] <= .01)
        assert np.all(table['shared_aperture_flux_fraction'][shared_selected] <= .02)
        pairs = Table.read(OUTPUT / 'shared_aperture_pairs.csv', format='ascii.csv')
        selected_ids = set(map(int, table['id'][selected]))
        rejected_pairs = np.asarray(pairs['pair_rejected']).astype(str) == 'True'
        for pair in pairs[rejected_pairs]:
            assert not ({int(pair['source_id_1']), int(pair['source_id_2'])} <= selected_ids)
    # These source positions were matched to the two user-supplied cutouts.
    for x, y in ((364.91, 3597.07), (373.69, 3581.74),
                 (404.39, 3715.86), (412.85, 3704.47)):
        distance = np.hypot(table['x'] - x, table['y'] - y)
        i = int(np.argmin(distance))
        assert distance[i] < 1 and table['shared_aperture_rejected'][i]
        assert not table['usable'][i]
    assert not np.any(selected & (np.hypot(table['x'] - 1036, table['y'] - 432) < 10))
    assert np.all(table['aperture_snr'][selected] >= 5)
    assert np.all(np.isfinite(table['flux_error_adu'][selected]))
    for row in table[selected]:
        bounds, _, _, aperture, edge = neighbour.aperture_grid(row, image.shape)
        assert not edge and not np.any(bad[bounds][aperture])
        np.testing.assert_allclose(residual[bounds][aperture].sum(), row['aperture_flux_adu'], rtol=1e-6)

    area = np.count_nonzero(~mask) * abs(header['PIXSCAL1'] * header['PIXSCAL2']) / 3600 ** 2
    np.testing.assert_allclose(area, table.meta['area_deg2'])
    counts = Table.read(OUTPUT / 'number_counts.ecsv')
    assert counts.meta['fit_range'] == [19., 23.]
    assert np.all(np.diff(counts['count']) >= 0)
    assert counts['count'][-1] == table['remaining_sample'].sum()

    # Show the actual sharp-feature cutouts, with the mask outlined in orange.
    features = Table.read(OUTPUT / 'sharp_features.ecsv')
    columns = 5
    rows = max(1, int(np.ceil(len(features) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(13, rows * 2.5), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, row in zip(axes.flat, features):
        x, y = round(row['x']), round(row['y'])
        bounds = (slice(max(0, y - 14), y + 15), slice(max(0, x - 14), x + 15))
        cut = residual[bounds]
        ax.imshow(cut, origin='lower', cmap='gray', vmin=-15, vmax=max(50, cut.max() * .5))
        ax.contour(sharp[bounds], levels=[.5], colors=['orange'], linewidths=.8)
        ax.set_title(f'({x}, {y})', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'sharp_feature_review.png', dpi=150)
    plt.close(fig)

    # Compare the same old bright candidates inside the new footprint.
    # A position match is an audit, not an injected-source completeness test.
    old = Table.read(BASE / 'results/balanced_recovery/sources.ecsv')
    tree = cKDTree(np.column_stack([table['x'], table['y']]))
    distances, indices = tree.query(np.column_stack([old['x'], old['y']]))
    matched = distances < 3.0
    ox, oy = np.rint(old['x']).astype(int), np.rint(old['y']).astype(int)
    common = ~mask[oy, ox]
    now_usable = matched & np.asarray(table['usable'][indices], bool)
    audit = Table({
        'old_id': old['id'], 'x': old['x'], 'y': old['y'],
        'old_usable': old['usable'], 'in_new_footprint': common,
        'matched_within_3pix': matched, 'new_id': table['id'][indices],
        'new_usable': now_usable, 'match_distance': distances,
    })
    audit.write(OUTPUT / 'source_matches.csv', format='ascii.csv', overwrite=True)
    order = np.argsort(np.asarray(old['segment_flux_adu']))[::-1]
    order = order[common[order]]
    for number in (100, 200, 500):
        rows = order[:number]
        print(f'Same bright {number}: old {int(old["usable"][rows].sum())}, new {int(now_usable[rows].sum())}')

    recovered = order[(~np.asarray(old['usable'], bool) & now_usable)[order]][:12]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, old_index in zip(axes.flat, recovered):
        row = table[indices[old_index]]
        x, y = float(row['x']), float(row['y'])
        half = max(20, int(row['aperture_a_pix']) + 6)
        x0, x1 = max(0, int(x) - half), min(image.shape[1], int(x) + half + 1)
        y0, y1 = max(0, int(y) - half), min(image.shape[0], int(y) + half + 1)
        cut = residual[y0:y1, x0:x1]
        ax.imshow(cut, origin='lower', cmap='gray', vmin=-15,
                  vmax=max(80, np.percentile(cut, 99)), extent=(x0, x1, y0, y1))
        ax.add_patch(Ellipse((x, y), 2 * row['aperture_a_pix'], 2 * row['aperture_b_pix'],
                             angle=row['aperture_theta_deg'], fill=False, color='royalblue', lw=1))
        ax.set_title(f'Old ID {old["id"][old_index]}: excluded → usable', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'bright_source_comparison.png', dpi=150)
    plt.close(fig)
    plot_aperture_changes(table, residual)
    print('Mask propagation, aperture sums, counts, controls and original-file hashes passed.')


if __name__ == '__main__':
    run()
