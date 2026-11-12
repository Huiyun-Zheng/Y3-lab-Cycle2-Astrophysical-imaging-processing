"""Conservative catalogue matching for one injected source per small image.

Uses the columns in injection_recovery.py. Positions are LOCAL x, y pixels.
Input IDs are trial IDs, not labels carried through the image.
Nearby original detections are flagged, not counted as clean recoveries.
"""

import numpy as np
from astropy.table import Table


def match_recovery(
        injected, detected, original_detected,
        match_radius=3.0, isolation_radius=None
        ):
    """Return one row per injection without changing any input table.

    Run the original pipeline for EVERY trial before injection. A trial with
    no original detections contributes zero rows to original_detected.
    Matching uses centres only: large halos and extended-source overlap
    need separate inspection.
    No flux-agreement cut is used, so flux errors can be studied afterwards.
    """

    # Check the input tables
    required = (
        (injected, 'injected', ('injected_id', 'x_local', 'y_local', 'flux_in')),
        (detected, 'detected', ('trial_id', 'detection_id', 'x_local', 'y_local', 'flux_out')),
        (original_detected, 'original_detected', ('trial_id', 'x_local', 'y_local'))
    )
    for table, name, columns in required:
        missing = [column for column in columns if column not in table.colnames]
        if missing:
            raise ValueError(f'{name} is missing columns: {missing}')
        if not (np.all(np.isfinite(table['x_local']))
                and np.all(np.isfinite(table['y_local']))):
            raise ValueError(f'{name} positions must be finite')
        for column in ('injected_id', 'trial_id', 'detection_id'):
            if column in table.colnames:
                ids = np.asarray(table[column], dtype=float)
                if not np.all(np.isfinite(ids) & (ids == np.floor(ids))):
                    raise ValueError(f'{name}.{column} must contain integer IDs')
    if len(np.unique(injected['injected_id'])) != len(injected):
        raise ValueError('This version requires one injected source per trial')
    if not np.isfinite(match_radius) or match_radius <= 0:
        raise ValueError('match_radius must be positive')
    if isolation_radius is not None:
        if not np.isfinite(isolation_radius) or isolation_radius < match_radius:
            raise ValueError('isolation_radius must be at least match_radius')
    elif 'sigma_major' not in injected.colnames:
        raise ValueError('Supply isolation_radius, or include sigma_major in injected')
    if not np.all(np.isfinite(injected['flux_in']) & (injected['flux_in'] > 0)):
        raise ValueError('Injected fluxes must be positive and finite')
    if isolation_radius is None:
        if not np.all(np.isfinite(injected['sigma_major']) & (injected['sigma_major'] > 0)):
            raise ValueError('sigma_major must be positive and finite')
    known_trials = np.asarray(injected['injected_id'])
    for table, name in ((detected, 'detected'), (original_detected, 'original_detected')):
        if np.any(~np.isin(table['trial_id'], known_trials)):
            raise ValueError(f'{name} contains trial IDs absent from injected')

    rows = []
    for source in injected:
        trial_id = int(source['injected_id'])
        x_in, y_in = float(source['x_local']), float(source['y_local'])
        flux_in = float(source['flux_in'])

        # 1. Look only inside this trial
        after = detected[detected['trial_id'] == trial_id]
        before = original_detected[original_detected['trial_id'] == trial_id]

        # 2. Find detections near the injected position
        distance = np.hypot(after['x_local']-x_in, after['y_local']-y_in)
        nearby = np.flatnonzero(distance <= match_radius)
        n_candidates = len(nearby)

        # 3. Check whether the original image was already crowded here
        if isolation_radius is None:
            check_radius = match_radius + 3*float(source['sigma_major'])
        else:
            check_radius = float(isolation_radius)
        original_distance = np.hypot(before['x_local']-x_in, before['y_local']-y_in)
        n_original = int(np.count_nonzero(original_distance <= check_radius))

        # 4. Keep ambiguous cases separate
        detection_id = -1
        x_out = y_out = separation = flux_out = flux_ratio = np.nan
        if n_candidates == 0:
            status = 'no_match'
        elif n_candidates > 1:
            status = 'ambiguous'
        else:
            j = nearby[0]
            candidate = after[j]
            detection_id = int(candidate['detection_id'])
            x_out, y_out = float(candidate['x_local']), float(candidate['y_local'])
            separation = float(distance[j])
            flux_out = float(candidate['flux_out'])
            if n_original > 0:
                status = 'near_original'
            elif not np.isfinite(flux_out) or flux_out <= 0:
                status = 'invalid_flux'
            else:
                status = 'matched'
                flux_ratio = flux_out / flux_in

        rows.append((
            trial_id, detection_id, status,
            x_in, y_in, x_out, y_out, separation,
            flux_in, flux_out, flux_ratio,
            n_candidates, n_original, check_radius
        ))

    matched = Table(
        rows=rows,
        names=(
            'injected_id', 'detection_id', 'status',
            'x_in_local', 'y_in_local', 'x_out_local', 'y_out_local', 'distance',
            'flux_in', 'flux_out', 'flux_ratio',
            'n_candidates', 'n_original_near', 'isolation_radius'
        ),
        dtype=('i8', 'i8', 'U20', 'f8', 'f8', 'f8', 'f8', 'f8',
               'f8', 'f8', 'f8', 'i8', 'i8', 'f8')
    )
    matched.meta['match_radius_pixels'] = float(match_radius)
    matched.meta['isolation_rule'] = (
        'match_radius + 3*sigma_major' if isolation_radius is None
        else f'fixed {float(isolation_radius)} pixels')
    matched.meta['status_note'] = (
        'matched=one nearby candidate and no known original centre nearby; '
        'no_match=no candidate in radius; ambiguous=multiple candidates; '
        'near_original=one candidate in a potentially blended region; '
        'invalid_flux=one isolated candidate with nonpositive/nonfinite flux. '
        'These are conservative catalogue-only decisions, not proof of source identity.')
    return matched


if __name__ == '__main__':
    # Artificial examples only; no FITS image is loaded
    injected = Table(
        rows=[(i, 50., 50., 1000., 3.) for i in range(1, 6)],
        names=('injected_id', 'x_local', 'y_local', 'flux_in', 'sigma_major')
    )
    detected = Table(
        rows=[(1, 7, 50.4, 50.2, 900.),
              (2, 8, 80., 80., 500.),
              (3, 9, 50.2, 50., 1000.), (3, 10, 51., 50., 700.),
              (4, 11, 50.3, 50.1, 5000.),
              (5, 12, 50.2, 50., np.nan)],
        names=('trial_id', 'detection_id', 'x_local', 'y_local', 'flux_out')
    )
    original_detected = Table(
        rows=[(4, 50.5, 50.1)],
        names=('trial_id', 'x_local', 'y_local')
    )
    matches = match_recovery(injected, detected, original_detected)
    print('DEMO DATA ONLY: this is not your real recovery result.\n')
    matches['injected_id', 'detection_id', 'status', 'distance', 'flux_ratio'].pprint(
        max_lines=-1, max_width=-1)
