"""Explicit local exclusions after visual review of crowded source wings.

This is a recorded manual review, not a new field-wide blending threshold.
Small-aperture contamination estimates do not validate missing outer light.
"""

import numpy as np
from astropy.table import Table


# Matched to the central source in the 2026-11-22 15:46:44 screenshot.
# Match by zero-based position because IDs can change on redetection.
REVIEWED_POSITIONS = ((2278.612, 3846.148),)
FLAG = 'reviewed_blend_unresolved_wings'


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
