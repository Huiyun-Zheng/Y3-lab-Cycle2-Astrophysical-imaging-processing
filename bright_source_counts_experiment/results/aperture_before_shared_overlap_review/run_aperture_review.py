"""Rerun the copied pipeline with an edge mask and explicit quality cuts.

Only results/aperture_review is replaced. Original project folders are untouched.
The count slope is measured over the fixed 19--23 mag range, not optimised for 0.6.
"""

import sys
sys.dont_write_bytecode = True

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree
from scipy import ndimage as ndi

import detection_aperture_review as detector
import source_detection_neighbour_model as neighbour
import photometry_analysis as photometry
from run_experiment import review_saved_scores


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'results' / 'aperture_review'
IMAGE = BASE.parent / 'Astro' / 'Fits_Data' / 'mosaic.fits'

# Keep the previous balanced contamination limits. Allow modest galaxy-profile
# mismatch, but do not accept large neighbour-light contamination or bad pixels.
MODEL_SETTINGS = neighbour.Settings(
    max_overlap_area_fraction=0.15,
    max_overlap_flux_fraction=0.10,
    max_bound_neighbour_fraction=0.02,
    max_sky_flux_fraction=0.05,
    max_fractional_residual=0.15,
    minimum_aperture_snr=5.0,
)


def reject_neighbour_cores(table, residual, rms, labels):
    """Check every aperture, including strict sources and smaller fallbacks.

    An overlap between two drawn ellipses is not enough to reject a source.
    Instead, look for actual bright pixels owned by another detected source.
    This is an observed core-overlap diagnostic, not a deblended flux estimate.
    """
    indices = np.arange(int(labels.max()) + 1)
    peaks = ndi.maximum(residual, labels, indices)
    owned_peak = peaks[labels]
    core = (labels > 0) & (residual >= 0.30 * owned_peak) & (owned_peak >= 10 * rms)

    count = np.zeros(len(table), int)
    fraction = np.zeros(len(table), float)
    neighbour_label = np.zeros(len(table), int)
    for i, row in enumerate(table):
        bounds, _, _, aperture, _ = neighbour.aperture_grid(row, residual.shape)
        other = aperture & core[bounds] & (labels[bounds] != row['segment_id'])
        count[i] = int(other.sum())
        if count[i] == 0:
            continue
        positive = np.maximum(residual[bounds], 0)
        fraction[i] = positive[other].sum() / max(positive[aperture].sum(), 1e-30)
        neighbour_labels = labels[bounds][other]
        unique = np.unique(neighbour_labels)
        fluxes = [positive[other][neighbour_labels == label].sum() for label in unique]
        neighbour_label[i] = unique[np.argmax(fluxes)]

    rejected = (count >= 2) & (fraction >= 0.01)
    table['neighbour_core_pixels'] = count
    table['neighbour_core_flux_fraction'] = fraction
    table['dominant_neighbour_core_label'] = neighbour_label
    table['neighbour_core_rejected'] = rejected
    table['usable_before_core_check'] = np.asarray(table['usable'], bool).copy()
    table['usable'] = np.asarray(table['usable'], bool) & ~rejected
    table['flags'] = np.asarray(table['flags'], dtype='U300')
    for i in np.flatnonzero(rejected):
        parts = [part for part in str(table['flags'][i]).split('|') if part != 'none']
        table['flags'][i] = '|'.join(parts + ['neighbour_core_intrusion'])
    table.meta['core_check'] = 'Other-label pixels >= 30% of its peak; neighbour peak >= 10 RMS; reject >= 2 such pixels with >= 1% of aperture positive flux. Applies to every source.'
    print(f'Additional core-overlap exclusions: {int(np.sum(rejected & table["usable_before_core_check"]))}', flush=True)
    return table


def match_references(table):
    """Match the checked stars by position: segmentation IDs change on redetection."""
    old = Table.read(BASE / 'inputs' / 'scored_sources.ecsv')
    reference = old[np.isin(old['segment_id'], photometry.REFERENCE_LABELS)]
    tree = cKDTree(np.column_stack([table['x'], table['y']]))
    distance, indices = tree.query(np.column_stack([reference['x'], reference['y']]))
    keep = (distance < 2.0) & np.asarray(table['usable'][indices], bool)
    mapping = Table({
        'old_segment_id': reference['segment_id'],
        'new_segment_id': table['segment_id'][indices],
        'distance_pix': distance,
        'retained': keep,
    })
    mapping.write(OUTPUT / 'reference_matches.csv', format='ascii.csv', overwrite=True)
    labels = np.asarray(table['segment_id'][indices[keep]], int)
    if len(labels) < 5 or len(np.unique(labels)) != len(labels):
        raise ValueError('Too few unique, checked PSF references remain.')
    return labels.tolist()


def retain_clean_original_apertures(table, image, residual, rms, products):
    """Prefer larger apertures; fall back only after checking the smaller one.

    A larger circle must not turn a previously measurable neighbour into an
    automatic loss. Keep the limitation explicit rather than claiming total flux.
    """
    targets = (np.asarray(table['aperture_method']) == 'bright_kron') & ~table['usable']
    table['preferred_aperture_a_pix'] = np.asarray(table['aperture_a_pix'], float)
    table['preferred_aperture_b_pix'] = np.asarray(table['aperture_b_pix'], float)
    table['aperture_wings_limited'] = np.zeros(len(table), bool)
    table['aperture_method'] = np.asarray(table['aperture_method'], dtype='U40')
    if not np.any(targets):
        return table

    metadata = {int(r['segment_id']): (int(r['parent_id']), bool(r['blended'])) for r in table}
    original_settings = replace(detector.Settings(), kron_minimum_snr=1e30)
    smaller, _ = detector.measure_sources(
        image, residual, np.zeros(image.shape), products['SOURCES'].data,
        metadata, products['ARTEFACT'].data.astype(bool),
        products['BOUNDARY'].data.astype(bool), original_settings, noise=rms,
    )
    np.testing.assert_array_equal(smaller['segment_id'], table['segment_id'])
    smaller['peak_detection_snr'] = table['peak_detection_snr'].copy()
    smaller['usable'] = np.asarray(smaller['strict_usable'], bool)
    smaller['local_background'] = (image - residual)[
        np.rint(smaller['y']).astype(int), np.rint(smaller['x']).astype(int)
    ]
    # Only fit groups containing an enlarged aperture that failed its checks.
    parents = np.unique(table['parent_id'][targets])
    checked, _ = neighbour.score_catalogue(
        smaller, residual, rms, products['BADMASK'].data.astype(bool),
        float(products[0].header['SATLEVEL']), MODEL_SETTINGS,
        gain=float(fits.getheader(IMAGE)['GAIN']), only_parents=parents,
    )
    checked = review_saved_scores(checked, residual, MODEL_SETTINGS)
    recovered = targets & np.asarray(checked['usable'], bool)
    for name in checked.colnames:
        if name in table.colnames:
            table[name][recovered] = checked[name][recovered]
    table['aperture_method'][recovered] = 'original_clean_fallback'
    table['aperture_wings_limited'][recovered] = True
    table.meta['fallback_note'] = 'A larger aperture failed; the original aperture was rechecked and used. Not total flux.'
    print(f'Clean original-aperture fallbacks: {int(recovered.sum())}/{int(targets.sum())}', flush=True)
    return table


def run(analysis_only=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not analysis_only:
        detector.run_detection(IMAGE, OUTPUT, detector.Settings())
        detected = Table.read(OUTPUT / 'sources.ecsv')
        detected.write(OUTPUT / 'detected_sources.ecsv', overwrite=True)

        image, header = fits.getdata(IMAGE, header=True)
        with fits.open(OUTPUT / 'detection_products.fits') as products:
            background = products['BACKGROUND'].data.copy()
            rms = products['RMS'].data.copy()
            bad = products['BADMASK'].data.astype(bool)
            saturation = float(products[0].header['SATLEVEL'])
        residual = image - background

        # The joint model starts from the strict selection, exactly as before.
        detected['usable'] = np.asarray(detected['strict_usable'], bool)
        detected['local_background'] = background[
            np.rint(detected['y']).astype(int), np.rint(detected['x']).astype(int)
        ]
        scored, _ = neighbour.score_catalogue(
            detected, residual, rms, bad, saturation, MODEL_SETTINGS,
            gain=float(header['GAIN']),
        )
        table = review_saved_scores(scored, residual, MODEL_SETTINGS)
        with fits.open(OUTPUT / 'detection_products.fits') as products:
            table = retain_clean_original_apertures(table, image, residual, rms, products)
            table = reject_neighbour_cores(table, residual, rms, products['SOURCES'].data)
        table.meta['detection_settings'] = json.dumps(asdict(detector.Settings()))
        table.meta['model_settings'] = json.dumps(asdict(MODEL_SETTINGS))
        table.meta['mask_note'] = 'Right strip x >= 2450, sharp features, and manually identified cross at (1036, 432); all included in DETMASK.'
        table.meta['image_sha256'] = photometry.file_hash(IMAGE)
        table.meta['products_sha256'] = photometry.file_hash(OUTPUT / 'detection_products.fits')
        table.write(OUTPUT / 'sources.ecsv', overwrite=True)
        table.write(OUTPUT / 'sources.csv', format='ascii.csv', overwrite=True)
        table[table['usable']].write(OUTPUT / 'usable_sources.csv', format='ascii.csv', overwrite=True)

        with fits.open(OUTPUT / 'detection_products.fits') as products:
            detector.save_diagnostics(
                image, background, rms, products['ARTEFACT'].data.astype(bool),
                products['BOUNDARY'].data.astype(bool),
                products['SATURATED'].data.astype(bool), table, OUTPUT,
            )
        neighbour.save_bright_review(residual, rms, table, OUTPUT, recovered_only=True)

    table = Table.read(OUTPUT / 'sources.ecsv')
    references = match_references(table)
    photometry.DETECTION_DIR = OUTPUT
    photometry.PRODUCTS_PATH = OUTPUT / 'detection_products.fits'
    photometry.OUTPUT_DIR = OUTPUT
    classified, fit = photometry.run_analysis(
        measurement_mag_limit=22.0, classification_mag_limit=22.0,
        fit_range=(19.0, 23.0), reference_labels=references,
    )

    # Record the actual final totals, not the intermediate strict-detection totals.
    summary_path = OUTPUT / 'run_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['final_usable'] = int(table['usable'].sum())
    summary['final_stars'] = int(classified['is_star'].sum())
    summary['final_remaining'] = int(classified['remaining_sample'].sum())
    summary['area_deg2'] = float(classified.meta['area_deg2'])
    summary['fit_range'] = [19.0, 23.0]
    summary['slope'] = float(fit['slope'])
    summary['slope_error'] = float(fit['slope_error_propagated'])
    summary['neighbour_settings'] = asdict(MODEL_SETTINGS)
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    return classified


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-only', action='store_true',
                        help='Explicitly reuse this review\'s saved detection and masks.')
    run(parser.parse_args().analysis_only)
