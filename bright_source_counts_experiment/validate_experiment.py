"""Check reproducibility, protected files and simple morphology controls."""

import sys
sys.dont_write_bytecode = True

from pathlib import Path
import hashlib

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter

from circular_moffat import empirical_profile_match


BASE = Path(__file__).resolve().parent


def profile_features(signal, xx, yy, xc, yc):
    radius = np.hypot(xx - xc, yy - yc)
    outer = radius <= 9
    total = signal[outer].sum()
    weights = np.maximum(signal[outer], 0)
    dx = xx[outer] - np.average(xx[outer], weights=weights)
    dy = yy[outer] - np.average(yy[outer], weights=weights)
    covariance = np.array([[np.average(dx * dx, weights=weights), np.average(dx * dy, weights=weights)],
                           [np.average(dx * dy, weights=weights), np.average(dy * dy, weights=weights)]])
    eigenvalues = np.linalg.eigvalsh(covariance)
    return dict(core_flux_fraction=signal[radius <= 3].sum() / total,
                middle_flux_fraction=signal[radius <= 5].sum() / total,
                moment_radius=np.sqrt(eigenvalues.sum()),
                moment_ellipticity=1 - np.sqrt(eigenvalues[0] / eigenvalues[1]))


def run():
    records = (BASE / 'inputs/original_file_hashes.tsv').read_text().splitlines()[1:]
    for record in records:
        name, expected = record.split('\t')
        assert hashlib.sha256((BASE.parent / name).read_bytes()).hexdigest() == expected, name

    original = Table.read(BASE / 'inputs/scored_sources.ecsv')
    previous_counts = Table.read(BASE / 'inputs/baseline_counts.ecsv')
    baseline_counts = Table.read(BASE / 'results/baseline/number_counts.ecsv')
    np.testing.assert_array_equal(previous_counts['count'], baseline_counts['count'])
    np.testing.assert_array_equal(previous_counts['limiting_magnitude'], baseline_counts['limiting_magnitude'])

    summary = Table.read(BASE / 'results/comparison.csv', format='ascii.csv')
    for name in summary['variant']:
        table = Table.read(BASE / 'results' / name / 'classified_sources.ecsv')
        for column in ['id', 'segment_id', 'x', 'y', 'aperture_a_pix', 'aperture_b_pix']:
            np.testing.assert_array_equal(table[column], original[column])
        selected = np.asarray(table['count_sample'], bool)
        for column in ['near_artefact', 'near_boundary', 'aperture_capped']:
            assert not np.any(table[column][selected]), (name, column)
        assert np.all(table['aperture_snr'][selected] >= 5)
        assert np.all(np.isfinite(table['flux_error_adu'][selected]))
        assert np.all(table['aperture_flux_adu'][selected] > 0)
        counts = Table.read(BASE / 'results' / name / 'number_counts.ecsv')
        assert counts.meta['fit_range'] == [19., 23.]
        assert np.all(np.diff(counts['count']) >= 0)
        assert counts['count'][-1] == table['remaining_sample'].sum()

    # Reference-star holdouts and deliberately broadened images are diagnostic
    # controls, not a measurement of real-galaxy purity or survey completeness.
    table = Table.read(BASE / 'results/empirical_psf/classified_sources.ecsv')
    refs = table[table['psf_reference']]
    image = fits.getdata(BASE.parent / 'Astro/Fits_Data/mosaic.fits').astype(float)
    controls = []
    for row in refs:
        others = refs[refs['segment_id'] != row['segment_id']]
        distance = np.hypot(others['x'] - row['x'], others['y'] - row['y'])
        local = others[np.argsort(distance)[:8]]
        holdout = empirical_profile_match(row, local)
        xc, yc = float(row['x']), float(row['y'])
        x, y = int(round(xc)), int(round(yc))
        yy, xx = np.mgrid[y - 30:y + 31, x - 30:x + 31]
        cutout = image[y - 30:y + 31, x - 30:x + 31]
        radius = np.hypot(xx - xc, yy - yc)
        _, sky, _ = sigma_clipped_stats(cutout[(radius >= 22) & (radius <= 30)], sigma=3)
        broadened = gaussian_filter(cutout - sky, 1.5)
        features = profile_features(broadened, xx, yy, xc, yc)
        false_star = empirical_profile_match(features, local)
        controls.append((int(row['segment_id']), holdout, false_star))
    control_table = Table(rows=controls, names=['reference_segment', 'holdout_recognised', 'broadened_classed_star'])
    control_table.write(BASE / 'results/morphology_controls.csv', format='ascii.csv', overwrite=True)
    print('Reference holdouts recognised:', sum(control_table['holdout_recognised']), '/', len(refs))
    print('Broadened profiles misclassified as stars:', sum(control_table['broadened_classed_star']), '/', len(refs))
    print('PASS: baseline reproduced, all variants use 19--23, original files unchanged, valid photometry and masks.')


if __name__ == '__main__':
    run()
