"""Check the field review and make a visual audit without changing detections."""
import sys
sys.dont_write_bytecode = True
import hashlib
from pathlib import Path
import numpy as np
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from field_internal_peaks import scan_internal_peaks
from run_field_quality_review import BASE, INPUT, OUTPUT, IMAGE, PRODUCTS, significant_structure, contact_sheet
from aperture_neighbour_model import aperture_grid


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def synthetic_checks():
    # An elongated single source must not be rejected just for its shape.
    yy, xx = np.mgrid[:61, :61]
    row = Table(dict(id=[1], segment_id=[1], x=[30.], y=[30.],
                     aperture_a_pix=[18.], aperture_b_pix=[18.],
                     aperture_theta_deg=[0.], usable=[True]))
    noise = np.ones(xx.shape)
    labels = np.ones(xx.shape, dtype=int)
    for sx, sy in [(2., 2.), (2., 5.)]:
        signal = 100 * np.exp(-.5 * (((xx-30)/sx)**2 + ((yy-30)/sy)**2))
        assert len(scan_internal_peaks(row, signal, noise, labels)) == 0
    main = 100 * np.exp(-((xx-26)**2 + (yy-30)**2)/(2*1.5**2))
    companion = np.exp(-((xx-34)**2 + (yy-30)**2)/(2*1.5**2))
    strong = scan_internal_peaks(row, main + 70*companion, noise, labels)
    assert len(strong) == 1 and significant_structure(strong)[0]
    assert len(scan_internal_peaks(row, main + 3*companion, noise, labels)) == 0


def make_report(old, new, display):
    # A warning does not silently change usable or the counts sample.
    warning = new['usable'] & (new['field_internal_peak_candidate'] |
                               new['field_shape_review'] | new['field_wing_review'])
    # This screenshot target has a weaker secondary feature at one scale only.
    # Record the visual concern separately; it does not satisfy the uniform cut.
    warning |= new['usable'] & (new['id'] == 2099)
    queue = new[warning].copy()
    reasons = []
    for row in queue:
        tags = []
        if row['field_internal_peak_candidate']: tags.append('weaker_internal_peaks')
        if row['field_shape_review']: tags.append('narrow_or_elongated_shape')
        if row['field_wing_review']: tags.append('compact_core_outer_light')
        if row['id'] == 2099: tags.append('screenshot_single_scale_secondary')
        reasons.append('|'.join(tags))
    queue['review_reason'] = reasons
    queue.write(OUTPUT/'review_queue.ecsv', overwrite=True)
    contact_sheet(new, np.flatnonzero(new['usable'] & new['field_wing_review']),
                  display, 'compact_core_wing_review')

    # Put the screenshot objects on identical before/after pixels and scales.
    fig, axes = plt.subplots(3, 2, figsize=(10, 13))
    for k, ident in enumerate([410, 239, 379]):
        row = new[new['id'] == ident][0]
        half = 34 if ident == 379 else 28
        x0, x1 = int(row['x'])-half, int(row['x'])+half+1
        y0, y1 = int(row['y'])-half, int(row['y'])+half+1
        for col, table in enumerate([old, new]):
            ax = axes[k, col]
            ax.imshow(display[y0:y1, x0:x1], origin='lower', cmap='gray',
                      vmin=-1, vmax=12, extent=(x0-.5,x1-.5,y0-.5,y1-.5))
            near = np.hypot(table['x']-row['x'], table['y']-row['y']) < 2*half
            for s in table[near]:
                colour = 'royalblue' if s['usable'] else 'red'
                if col and s['id'] == 2099: colour = 'darkorange'
                ax.add_patch(Ellipse((s['x'],s['y']), 2*s['aperture_a_pix'],
                    2*s['aperture_b_pix'], angle=s['aperture_theta_deg'],
                    fill=False, edgecolor=colour, lw=1.5))
                if s['id'] in (ident, 2099):
                    ax.text(s['x'],s['y']-s['aperture_a_pix']-3,
                            str(s['id']), color=colour, fontsize=10)
            ax.set(xlim=(x0,x1), ylim=(y0,y1),
                   title=f"ID {ident} — {'before' if col == 0 else 'field review'}")
            ax.set_axis_off()
    fig.suptitle('Blue: usable | Red: excluded | Orange: retained, needs review\n'
                 'Same pixels, apertures and display scale in both columns')
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(OUTPUT/'screenshot_examples_before_after.png', dpi=140)
    plt.close(fig)

    removed = old['usable'] & ~new['usable']
    recovered = ~old['usable'] & new['usable']
    fig, ax = plt.subplots(figsize=(9, 14))
    ax.imshow(display, origin='lower', cmap='gray', vmin=-1, vmax=12)
    for selection, colour, label, marker in [
        (warning, 'orange', f'Retained, needs review ({sum(warning)})', 'o'),
        (removed, 'red', f'Newly excluded ({sum(removed)})', 'x'),
        (recovered, 'lime', f'Recovered bright sources ({sum(recovered)})', '+')]:
        style = dict(facecolors='none', edgecolors=colour) if marker=='o' else dict(color=colour)
        ax.scatter(new['x'][selection],new['y'][selection],s=45,
                   marker=marker, linewidths=1.1,label=label, **style)
    ax.legend(loc='upper left', fontsize=9)
    ax.set(title='Full-field quality audit — unchanged apertures', xlabel='x / pixel', ylabel='y / pixel')
    fig.tight_layout()
    fig.savefig(OUTPUT/'field_review_status_map.png', dpi=150)
    plt.close(fig)
    return int(sum(warning))


def run():
    synthetic_checks()
    old = Table.read(INPUT/'sources.ecsv')
    new = Table.read(OUTPUT/'sources.ecsv')
    snapshot = Table.read(OUTPUT/'baseline_sources.ecsv')
    assert sha(INPUT/'sources.ecsv') == new.meta['field_review_baseline_sha256']
    for col in old.colnames:
        a, b = np.asarray(old[col]), np.asarray(snapshot[col])
        assert np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b), col
    for col in ['id','segment_id','parent_id','x','y','aperture_a_pix','aperture_b_pix',
                'aperture_theta_deg','aperture_shape','blended','near_artefact','near_boundary']:
        assert np.array_equal(old[col],new[col]), col
    finite = np.isfinite(old['aperture_flux_adu'])
    assert np.array_equal(old['aperture_flux_adu'][finite], new['aperture_flux_adu'][finite])
    assert len(new) == 7038
    removed = old['usable'] & ~new['usable']
    recovered = ~old['usable'] & new['usable']
    assert set(new['id'][removed]) == {239,339,379,410,438,1018,1524,1615,1621}
    assert sum(recovered) == 16 and sum(new['usable']) == 5676
    assert np.all(new['field_internal_peak_rejected'][removed])
    assert not np.any(new['usable'] & new['field_internal_peak_rejected'])
    assert np.array_equal(recovered,new['field_review_recovered'])
    for ident in [3,20,4,737]:
        assert not new['usable'][new['id']==ident][0]
    for ident in [6,14,91,154,243,2099]:
        assert new['usable'][new['id']==ident][0]

    fits_before = sha(PRODUCTS)
    image, header = fits.getdata(IMAGE, header=True)
    with fits.open(PRODUCTS) as f:
        residual = image - f['BACKGROUND'].data
        rms = f['RMS'].data.copy()
        mask = f['DETMASK'].data.astype(bool) | f['BADMASK'].data.astype(bool)
    fit = Table.read(OUTPUT/'refitted_bright_scores.ecsv')
    for row in new[recovered]:
        scored = fit[fit['id']==row['id']][0]
        assert scored['model_valid'] and scored['usable']
        assert scored['model_flux_fraction'] <= .03
        assert scored['model_score_disagreement'] <= .01
        assert scored['model_area_fraction'] <= .15
        assert row['aperture_snr'] >= 50
        assert not (row['neighbour_core_rejected'] or row['neighbour_peak_inside']
                    or row['near_boundary'] or row['aperture_capped']
                    or row['field_internal_peak_candidate'] or row['field_wing_review'])
        bounds, _, _, ap, edge = aperture_grid(row, residual.shape)
        assert not edge and not np.any(mask[bounds][ap])
        assert np.isclose(row['aperture_flux_adu'], np.sum(residual[bounds][ap]), rtol=2e-6)
    pairs = Table.read(INPUT/'shared_aperture_pairs.csv', format='ascii.csv')
    accepted = set(new['id'][new['usable']])
    for pair in pairs[np.asarray(pairs['pair_rejected']).astype(str)=='True']:
        assert not (pair['source_id_1'] in accepted and pair['source_id_2'] in accepted)
    classified = Table.read(OUTPUT/'classified_sources.ecsv')
    assert sum(classified['remaining_sample']) == 5607
    assert set(classified['id'][classified['count_sample']]) == accepted
    protected = Table.read(BASE/'inputs/original_file_hashes.tsv', format='ascii.tab')
    for entry in protected:
        assert sha(BASE.parent/entry['path']) == entry['sha256'], entry['path']
    count = make_report(old, new, residual/rms)
    assert sha(PRODUCTS) == fits_before
    lines = ['PASS: synthetic single, elongated, strong-double and weak-secondary controls.',
             'PASS: baseline snapshot and protected original files unchanged.',
             'PASS: fixed candidate positions, segmentation IDs, apertures and original finite fluxes.',
             'PASS: all 16 recoveries satisfy mask, model, S/N and neighbour checks.',
             'PASS: 9 new multi-peak exclusions; no rejected shared pair has both members usable.',
             'PASS: photometry selects exactly the 5676 usable sources, 5607 remaining after star selection.',
             f'Retained review queue: {count}; these warnings do not remove objects from counts.',
             f'Products SHA256: {fits_before}']
    (OUTPUT/'VALIDATION.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    run()
