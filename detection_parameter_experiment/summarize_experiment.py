"""Compare every declared variant at fixed calibration, classification and fit range."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

from photometry_analysis import cumulative_counts, MAG_STEP


ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / 'comparison.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    plan = json.loads((ROOT / 'experiment_plan.json').read_text())
    sensitivity = []
    diagnostics = []
    baseline = None
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')

    for row in rows:
        table = Table.read(ROOT / row['analysis_directory'] / 'classified_sources.ecsv')
        selected = np.asarray(table['count_sample'], dtype=bool)
        remaining = selected & ~np.asarray(table['is_star'], dtype=bool)
        magnitude = np.asarray(table['magnitude'][remaining])
        x, n = cumulative_counts(magnitude, MAG_STEP)
        positive = n > 0
        x, n = x[positive], n[positive]
        y = np.log10(n / float(row['area_deg2']))
        ax.plot(x, y, 'o-', markersize=3, label=f"{row['variant']}: {float(row['slope_19_24']):.3f}")

        for lo, hi in plan['fit_sensitivity_ranges']:
            mask = (x >= lo) & (x <= hi)
            slope = np.polyfit(x[mask], y[mask], 1)[0] if mask.sum() >= 3 else np.nan
            sensitivity.append({'variant':row['variant'], 'fit_min':lo, 'fit_max':hi, 'slope':float(slope)})

        coordinates = np.column_stack([table['x'], table['y']])
        if row['variant'] == 'baseline':
            baseline = cKDTree(coordinates)
        distances, _ = baseline.query(coordinates)
        new = distances > 3.0
        diagnostic = {
            'variant': row['variant'],
            'negative_to_positive_candidate_ratio': int(row['negative_candidates']) / int(row['candidates']),
            'usable_fraction': int(row['photometric_sources']) / int(row['candidates']),
            'remaining_within_19_24': int(np.count_nonzero((magnitude >= 19) & (magnitude < 24))),
            'remaining_at_24': int(np.count_nonzero(magnitude < 24)),
            'new_positions_vs_baseline_3pix': int(np.count_nonzero(new)),
            'new_remaining_within_19_24': int(np.count_nonzero(
                new & remaining & (table['magnitude'] >= 19) & (table['magnitude'] < 24)
            )),
        }
        for flag in ['blended', 'aperture_contaminated', 'aperture_capped']:
            diagnostic[flag] = int(np.count_nonzero(table[flag]))
        diagnostic['psf_references'] = int(np.count_nonzero(table['stellar_reference']))
        diagnostic['insufficient_local_psf'] = int(np.count_nonzero(
            table['classification_reason'] == 'insufficient_local_psf'
        ))
        diagnostics.append(diagnostic)

    ax.set(xlabel='Limiting aperture magnitude, r', ylabel=r'$\log_{10}[(N(<m)/A)/(1\,\mathrm{deg}^{-2})]$', title='Detection-parameter comparison')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, title='Variant: slope over 19–24 mag')
    fig.savefig(ROOT / 'comparison.png', dpi=200)
    plt.close(fig)

    for name, data in [('fit_sensitivity.csv', sensitivity), ('quality_diagnostics.csv', diagnostics)]:
        with (ROOT / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    hashes = json.loads((ROOT / 'original_file_hashes.json').read_text())
    changed = [name for name, digest in hashes.items()
               if hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() != digest]
    assert not changed, changed
    (ROOT / 'verification.json').write_text(json.dumps({
        'original_files_checked':len(hashes), 'original_files_changed':changed,
        'all_declared_variants_completed':len(rows) == len(plan['variants']),
        'negative_ratio_note':'Raw negative detections / raw positive candidates; NOT a calibrated contamination fraction.',
        'matching_note':'3-pixel positional comparison is diagnostic; centroid shifts/deblending can count as new.',
        'slope_note':'Cumulative slopes are descriptive and correlated; no completeness calibration.'
    }, indent=2))
    print(json.dumps({'comparison':rows,'diagnostics':diagnostics,'sensitivity':sensitivity},indent=2))


if __name__ == '__main__':
    main()
