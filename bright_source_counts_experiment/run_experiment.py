"""Compare copied detection and photometry methods at the same 19--23 mag range.

Read the saved neighbour-model fits instead of repeating source detection.
Only this experiment directory is written. No variant is selected by its slope.
"""

import sys
sys.dont_write_bytecode = True

from dataclasses import replace, asdict
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

import source_detection_neighbour_model as detection
import photometry_analysis as photometry
from circular_moffat import measure_psf_widths, select_psf_candidates


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'results'
REFERENCES = [6720, 5677, 7009, 5741, 6315, 6998, 6658, 4392, 3534, 4958, 3324]
FIT_RANGE = (19., 23.)


def review_saved_scores(original, residual, settings):
    """Apply explicit quality limits to the previously measured diagnostics."""
    table = original.copy()
    candidates = np.asarray(table['reassessment_candidate'], bool)
    for i in np.flatnonzero(candidates):
        row = table[i]
        names = ['model_area_fraction', 'model_flux_fraction', 'model_residual_score',
                 'model_coherent_residual', 'model_fractional_residual',
                 'model_bound_neighbour_fraction', 'model_sky_flux_fraction',
                 'model_score_disagreement', 'model_aperture_snr']
        if not all(np.isfinite(row[name]) for name in names):
            continue
        small_mismatch = row['model_fractional_residual'] <= settings.max_fractional_residual
        if row['model_bound_neighbour_fraction'] > settings.max_bound_neighbour_fraction:
            status = 'unstable_neighbour'
        elif row['model_sky_flux_fraction'] > settings.max_sky_flux_fraction:
            status = 'local_sky_sensitive'
        elif row['model_residual_score'] > settings.max_reduced_residual and not small_mismatch:
            status = 'poor_profile_fit'
        elif row['model_coherent_residual'] > settings.max_coherent_residual and not small_mismatch:
            status = 'structured_residual'
        elif row['model_score_disagreement'] > settings.max_score_disagreement:
            status = 'profile_sensitive'
        elif row['model_aperture_snr'] < settings.minimum_model_snr:
            status = 'low_model_snr'
        else:
            status = 'ok'
        table['model_status'][i] = status
        table['model_valid'][i] = status == 'ok'

    table['usable'] = detection.select_sources(table, settings)
    recovered = table['usable'] & ~table['strict_usable']
    table['aperture_flux_adu'] = table['original_aperture_flux_adu'].copy()
    table['flags'] = np.asarray(table['original_flags'], dtype='U240')
    table['aperture_contaminated'] = np.array(['aperture_contaminated' in str(f)
                                               for f in table['original_flags']])
    table.meta['model_settings'] = json.dumps(asdict(settings))
    for i in np.flatnonzero(recovered):
        bounds, _, _, aperture, edge = detection.aperture_grid(table[i], residual.shape)
        assert not edge
        table['aperture_flux_adu'][i] = float(residual[bounds][aperture].sum())
        table['model_aperture_flux_adu'][i] = table['aperture_flux_adu'][i]
        table['aperture_contaminated'][i] = False
        flags = [f for f in str(table['flags'][i]).split('|')
                 if f not in ('none', 'aperture_contaminated')]
        table['flags'][i] = '|'.join(flags + ['model_recovered'])
    faint = np.isfinite(table['aperture_snr']) & (table['aperture_snr'] < settings.minimum_aperture_snr)
    for i in np.flatnonzero(faint):
        table['flags'][i] = str(table['flags'][i]).replace('none', '').strip('|') + '|low_aperture_snr'
    return table


def run():
    OUTPUT.mkdir(exist_ok=True)
    original = Table.read(BASE / 'inputs/scored_sources.ecsv')
    image, header = fits.getdata(BASE.parent / 'Astro/Fits_Data/mosaic.fits', header=True)
    image = image.astype(np.float32)
    with fits.open(BASE / 'inputs/detection_products.fits') as products:
        labels = products['SOURCES'].data.copy()
        mask = products['DETMASK'].data.astype(bool)
        background = products['BACKGROUND'].data.copy()
        rms = products['RMS'].data.copy()
        artefact = products['ARTEFACT'].data.astype(bool)
        boundary = products['BOUNDARY'].data.astype(bool)
        saturated = products['SATURATED'].data.astype(bool)
        header['SATLEVEL'] = float(products[0].header['SATLEVEL'])
    mask |= ~np.isfinite(image) | ~np.isfinite(background) | ~np.isfinite(rms) | (rms <= 0)
    residual = image - background
    area = np.count_nonzero(~mask) * abs(header['PIXSCAL1'] * header['PIXSCAL2']) / 3600 ** 2

    standard = detection.Settings()
    balanced = replace(standard, max_overlap_area_fraction=.15,
                       max_bound_neighbour_fraction=.02, max_sky_flux_fraction=.05)
    # A stricter flux-contamination control is shown as a sensitivity test.
    strict_flux = replace(balanced, max_overlap_flux_fraction=.05)
    variants = [
        ('baseline', original.copy(), 'original'),
        ('psf_review', original.copy(), 'reference_profile'),
        ('balanced_recovery', review_saved_scores(original, residual, balanced), 'reference_profile'),
        ('strict_flux_control', review_saved_scores(original, residual, strict_flux), 'reference_profile'),
        ('empirical_psf', original.copy(), 'empirical_profile'),
        ('balanced_empirical', review_saved_scores(original, residual, balanced), 'empirical_profile'),
    ]

    # Measure each potentially accepted source once, using the copied Moffat code.
    union = original.copy()
    union['usable'] = np.logical_or.reduce([t['usable'] for _, t, _ in variants])
    for _, table, _ in variants:
        use = np.asarray(table['usable'], bool)
        union['aperture_flux_adu'][use] = table['aperture_flux_adu'][use]
    measured = measure_psf_widths(image, labels, photometry.calibrate_magnitudes(union, header),
                                 header, mask=mask, mag_limit=22.)
    measured, _ = photometry.add_photometric_uncertainties(measured, residual, rms, labels, mask, header)
    measured.write(OUTPUT / 'measured_sources.ecsv', overwrite=True)

    comparison = []
    sensitivities = []
    top100 = np.argsort(original['segment_flux_adu'])[-100:]
    top200 = np.argsort(original['segment_flux_adu'])[-200:]
    all_counts = {}
    all_classified = {}
    for name, sources, mode in variants:
        output = OUTPUT / name
        output.mkdir(exist_ok=True)
        sources.write(output / 'sources.ecsv', overwrite=True)
        table = measured.copy()
        table.meta.update(sources.meta)
        for column in sources.colnames:
            table[column] = sources[column].copy()
        table['count_sample'] = np.asarray(sources['usable'], bool)
        table['magnitude'][~table['count_sample']] = np.nan
        table = select_psf_candidates(table, REFERENCES, mag_limit=22., quality_mode=mode)
        table['is_star'] = table['source_class'] == 'star_candidate'
        table['remaining_sample'] = table['count_sample'] & ~table['is_star']
        table.meta['experiment_variant'] = name
        table.meta['area_deg2'] = float(area)
        table.meta['fit_range'] = list(FIT_RANGE)
        table.meta['input_sources_sha256'] = hashlib.sha256((output / 'sources.ecsv').read_bytes()).hexdigest()
        table.meta['input_products_sha256'] = hashlib.sha256((BASE / 'inputs/detection_products.fits').read_bytes()).hexdigest()
        photometry.plot_width_magnitude(table, output)
        fit = photometry.plot_number_counts(table, area, output, FIT_RANGE, header)
        table.meta['descriptive_slope'] = fit['slope']
        table.write(output / 'classified_sources.ecsv', overwrite=True)
        counts = Table.read(output / 'number_counts.ecsv')
        all_counts[name] = counts
        all_classified[name] = table
        comparison.append((name, int(table['count_sample'].sum()), int(sources['usable'][top100].sum()),
                           int(sources['usable'][top200].sum()), int(table['is_star'].sum()),
                           int(table['remaining_sample'].sum()),
                           int(np.count_nonzero(table['remaining_sample'] & (table['magnitude'] < 19))),
                           fit['slope'], fit['slope_error_propagated']))
        for low, high in [(18., 22.), (19., 23.), (20., 23.), (20., 24.)]:
            inside = (counts['limiting_magnitude'] >= low) & (counts['limiting_magnitude'] <= high)
            slope = np.polyfit(counts['limiting_magnitude'][inside], counts['log10_density'][inside], 1)[0]
            sensitivities.append((name, low, high, float(slope)))
        print(name, comparison[-1], flush=True)

    summary = Table(rows=comparison, names=['variant', 'usable', 'brightest_100_usable',
                    'brightest_200_usable', 'stars_removed', 'remaining', 'remaining_brighter_19',
                    'slope_19_23', 'slope_error'])
    summary.write(OUTPUT / 'comparison.csv', format='ascii.csv', overwrite=True)
    Table(rows=sensitivities, names=['variant', 'fit_start', 'fit_end', 'descriptive_slope']).write(
        OUTPUT / 'fit_range_sensitivity.csv', format='ascii.csv', overwrite=True)

    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    for row in summary:
        c = all_counts[row['variant']]
        ax.plot(c['limiting_magnitude'], c['log10_density'], 'o-', markersize=3,
                label=f"{row['variant']}: slope {row['slope_19_23']:.3f}")
    ax.axvspan(*FIT_RANGE, color='grey', alpha=.08)
    x = np.linspace(18, 24, 100)
    anchor = np.interp(21., all_counts['baseline']['limiting_magnitude'], all_counts['baseline']['log10_density'])
    ax.plot(x, anchor + .6 * (x - 21), '--', color='grey', label='Reference slope 0.6')
    ax.set(xlabel='Limiting aperture magnitude, r', ylabel=r'$\log_{10}[(N(<m)/A)/(1\,\mathrm{deg}^{-2})]$',
           title='Detection and classification comparison', xlim=(16, 26))
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(OUTPUT / 'counts_comparison.png', dpi=200)
    plt.close(fig)

    # Display the more inclusive recovery trial without calling it ground truth.
    chosen = variants[2][1]
    detection.detector.save_diagnostics(image, background, rms, artefact, boundary, saturated,
                                        chosen, OUTPUT / 'balanced_recovery')
    detection.save_bright_review(residual, rms, chosen, OUTPUT / 'balanced_recovery', recovered_only=True)
    return all_classified, summary


if __name__ == '__main__':
    run()
