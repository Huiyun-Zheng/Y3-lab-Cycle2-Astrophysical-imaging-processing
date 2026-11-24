"""Explicit local exclusions after visual review of crowded source wings.

This is a recorded manual review, not a new field-wide blending threshold.
Small-aperture contamination estimates do not validate missing outer light.
"""

import numpy as np
from astropy.table import Table
from scipy import ndimage as ndi


# Matched to the central source in the 2026-11-22 15:46:44 screenshot.
# Match by zero-based position because IDs can change on redetection.
REVIEWED_POSITIONS = ((2278.612, 3846.148),)
FLAG = 'reviewed_blend_unresolved_wings'

# A second reviewed object has two maxima inside ONE segmentation label.
# This is a provisional visual exclusion, not proof of two separate galaxies.
INTERNAL_PEAK_POSITION = (647.2980, 2129.2226)
INTERNAL_PEAK_FLAG = 'reviewed_internal_multiple_peaks'


def apply_reviewed_internal_peak(table, residual, rms, labels):
    """Check the selected unresolved structure without splitting the catalogue.

    The default sigma=1 smoothing merges its shoulder. We record local maxima
    at lighter smoothing, but do not interpret their pixel S/N as a calibrated
    probability of a second object. Original blended/segmentation flags remain.
    """
    x, y = INTERNAL_PEAK_POSITION
    matches = np.flatnonzero(np.hypot(table['x'] - x, table['y'] - y) < 1.)
    if len(matches) != 1:
        raise ValueError('The internally structured source changed; inspect it again.')
    i = int(matches[0])
    row = table[i]
    half = int(np.ceil(max(row['aperture_a_pix'], row['aperture_b_pix']))) + 5
    cx, cy = int(round(row['x'])), int(round(row['y']))
    yy, xx = np.mgrid[cy-half:cy+half+1, cx-half:cx+half+1]
    if xx.min() < 0 or yy.min() < 0 or xx.max() >= residual.shape[1] or yy.max() >= residual.shape[0]:
        raise ValueError('The internal-peak review cutout crosses an image edge.')
    data = residual[yy, xx].astype(float)
    noise = rms[yy, xx].astype(float)
    theta = np.radians(row['aperture_theta_deg'])
    dx, dy = xx-row['x'], yy-row['y']
    u, v = dx*np.cos(theta)+dy*np.sin(theta), -dx*np.sin(theta)+dy*np.cos(theta)
    aperture = (u/row['aperture_a_pix'])**2 + (v/row['aperture_b_pix'])**2 <= 1
    own = labels[yy, xx] == row['segment_id']
    measurements, counts, light_peaks = [], {}, []
    for sigma in (.5, .7, 1.):
        smooth = ndi.gaussian_filter(data, sigma)
        peaks = ((smooth == ndi.maximum_filter(smooth, size=3))
                 & (smooth > 5*noise) & aperture & own)
        positions = sorted(zip(*np.where(peaks)), key=lambda p: smooth[p], reverse=True)
        counts[sigma] = len(positions)
        if sigma == .5:
            light_peaks = positions
        for rank, p in enumerate(positions, 1):
            measurements.append((int(row['id']), sigma, rank, int(xx[p]), int(yy[p]),
                                 float(smooth[p]), float(noise[p]), int(labels[yy[p], xx[p]])))
    if counts[.5] < 2 or counts[.7] < 2 or counts[1.] != 1:
        raise ValueError('The reviewed internal-peak pattern changed; inspect it again.')
    separation = float(np.linalg.norm(np.subtract(light_peaks[0], light_peaks[1])))
    if separation < 3:
        raise ValueError('The reviewed maxima are no longer distinct at this sampling.')
    table['reviewed_internal_peak_rejected'] = np.zeros(len(table), bool)
    table['reviewed_internal_peak_rejected'][i] = True
    table['reviewed_blend_rejected'][i] = True
    table['usable'][i] = False
    table['flags'] = np.asarray(table['flags'], dtype='U500')
    parts = [p for p in str(table['flags'][i]).split('|') if p not in ('none', INTERNAL_PEAK_FLAG)]
    table['flags'][i] = '|'.join(parts+[INTERNAL_PEAK_FLAG])
    table.meta['internal_peak_review_note'] = (
        'Only position (647.2980, 2129.2226): two local maxima at smoothing '
        'sigma 0.5/0.7, one at sigma 1, in one source label. Provisional visual '
        'exclusion from counts; not an automatic deblend or a confirmed object pair.'
    )
    audit = Table(rows=measurements, names=['source_id', 'smooth_sigma_pix', 'rank',
        'x', 'y', 'smoothed_signal_adu', 'unsmoothed_rms_adu', 'segment_id'])
    print(f'Internal-peak review: exclude ID {row["id"]}; separation {separation:.2f} pixels.', flush=True)
    return table, audit


def apply_reviewed_blends(table):
    """Exclude the reviewed blend; preserve its measured core aperture.

    A radius of 14 pixels adds about 4% observed flux, while a trial radius
    of 16 pixels fails the existing shared-light check with the left neighbour.
    Neither is an independently established total-light aperture. Keep the
    original measured aperture for auditing, but exclude this source from counts
    until its wings can be measured with reliable deblending.
    """
    table['reviewed_blend_rejected'] = np.zeros(len(table), bool)
    table['flags'] = np.asarray(table['flags'], dtype='U400')
    decisions = []
    for x, y in REVIEWED_POSITIONS:
        distance = np.hypot(table['x'] - x, table['y'] - y)
        matches = np.flatnonzero(distance < 1.)
        if len(matches) != 1:
            raise ValueError('The reviewed blend no longer has a unique match; inspect it again.')
        i = int(matches[0])
        if not table['blended'][i]:
            raise ValueError('The reviewed blend segmentation changed; inspect it again.')
        table['reviewed_blend_rejected'][i] = True
        table['usable'][i] = False
        table['aperture_wings_limited'][i] = True
        parts = [p for p in str(table['flags'][i]).split('|') if p not in ('none', FLAG)]
        table['flags'][i] = '|'.join(parts + [FLAG])
        decisions.append((int(table['id'][i]), float(table['x'][i]), float(table['y'][i]),
                          FLAG, 'Core aperture retained for audit; excluded from counts.'))
    table.meta['reviewed_blends_note'] = (
        'Explicit position-selected visual exclusion: crowded outer wings of '
        'the source at (2278.612, 3846.148) are not validated by its small-core fit. '
        'No global thresholds or aperture sizes changed.'
    )
    audit = Table(rows=decisions, names=['source_id', 'x', 'y', 'reason', 'action'])
    return table, audit
