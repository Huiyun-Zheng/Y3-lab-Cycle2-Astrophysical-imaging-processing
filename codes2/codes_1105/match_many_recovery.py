"""Conservative matching when several sources are injected in one image.

Each image has a frame_id. injected_id and detection_id are separate IDs.
Only mutually unique candidates can be marked matched; no nearest-source
assignment is forced when a detection could belong to several injections.
"""

import numpy as np
from astropy.table import Table


def match_many_recovery(
        injected, detected, original_detected,
        match_radius=3., isolation_radius=None
        ):
    """Match by local pixel positions, preserving one output row per injection.

    Run the original pipeline once for EVERY frame before injection. Empty
    frames contribute no rows to original_detected. These centre-only checks
    cannot identify every halo or extended blend; inspect uncertain cases.
    """

    # Check table columns and coordinate conventions
    required = (
        (injected, 'injected', ('frame_id', 'injected_id', 'x_local', 'y_local',
                                'flux_in', 'sigma_major')),
        (detected, 'detected', ('frame_id', 'detection_id', 'x_local', 'y_local', 'flux_out')),
        (original_detected, 'original_detected', ('frame_id', 'x_local', 'y_local'))
    )
    for table, name, columns in required:
        missing = [column for column in columns if column not in table.colnames]
        if missing:
            raise ValueError(f'{name} is missing columns: {missing}')
        for column in ('x_local', 'y_local'):
            if not np.all(np.isfinite(table[column])):
                raise ValueError(f'{name}.{column} must be finite')
        for column in ('frame_id', 'injected_id', 'detection_id'):
            if column in table.colnames:
                values = np.asarray(table[column], dtype=float)
                if not np.all(np.isfinite(values) & (values == np.floor(values))):
                    raise ValueError(f'{name}.{column} must contain integer IDs')
    for table, id_column in ((injected, 'injected_id'), (detected, 'detection_id')):
        keys = list(zip(table['frame_id'], table[id_column]))
        if len(set(keys)) != len(keys):
            raise ValueError(f'{id_column} must be unique within each frame')
    for column in ('flux_in', 'sigma_major'):
        if not np.all(np.isfinite(injected[column]) & (injected[column] > 0)):
            raise ValueError(f'{column} must be positive and finite')
    if not np.isfinite(match_radius) or match_radius <= 0:
        raise ValueError('match_radius must be positive')
    if isolation_radius is not None:
        if not np.isfinite(isolation_radius) or isolation_radius < match_radius:
            raise ValueError('isolation_radius must be at least match_radius')
    for table in (detected, original_detected):
        if np.any(~np.isin(table['frame_id'], injected['frame_id'])):
            raise ValueError('A detection frame is absent from injected')

    output_rows = [None] * len(injected)
    for frame_id in np.unique(injected['frame_id']):
        indices = np.flatnonzero(injected['frame_id'] == frame_id)
        sources = injected[indices]
        after = detected[detected['frame_id'] == frame_id]
        before = original_detected[original_detected['frame_id'] == frame_id]

        # 1. Find candidates for every injected source
        candidate_lists = []
        candidate_use = np.zeros(len(after), dtype=int)
        for source in sources:
            distance = np.hypot(after['x_local']-source['x_local'],
                                after['y_local']-source['y_local'])
            nearby = np.flatnonzero(distance <= match_radius)
            candidate_lists.append(nearby)
            candidate_use[nearby] += 1

        # 2. Decide whether each source has a unique counterpart
        for local_i, (source, nearby) in enumerate(zip(sources, candidate_lists)):
            x_in, y_in = float(source['x_local']), float(source['y_local'])
            flux_in = float(source['flux_in'])
            check_radius = (match_radius + 3*float(source['sigma_major'])
                            if isolation_radius is None else float(isolation_radius))
            old_distance = np.hypot(before['x_local']-x_in, before['y_local']-y_in)
            n_original = int(np.count_nonzero(old_distance <= check_radius))

            # 3. Flag artificial crowding between injected sources
            injection_distance = np.hypot(sources['x_local']-x_in, sources['y_local']-y_in)
            separation_limit = 3*(sources['sigma_major']+source['sigma_major'])
            other_injections = np.asarray(injection_distance < separation_limit)
            other_injections[local_i] = False
            n_injected = int(other_injections.sum())

            detection_id = -1
            x_out = y_out = distance_out = flux_out = flux_ratio = np.nan
            n_claimants = 0
            if len(nearby) == 0:
                status = 'no_match'
            elif len(nearby) > 1:
                status = 'ambiguous'
            else:
                j = nearby[0]
                row = after[j]
                detection_id = int(row['detection_id'])
                x_out, y_out = float(row['x_local']), float(row['y_local'])
                distance_out = float(np.hypot(x_out-x_in, y_out-y_in))
                flux_out = float(row['flux_out'])
                n_claimants = int(candidate_use[j])

                # 4. Never count the same detection twice as matched
                if n_claimants > 1:
                    status = 'shared_detection'
                elif n_original > 0:
                    status = 'near_original'
                elif n_injected > 0:
                    status = 'near_injected'
                elif not np.isfinite(flux_out) or flux_out <= 0:
                    status = 'invalid_flux'
                else:
                    status = 'matched'
                    flux_ratio = flux_out / flux_in

            output_rows[indices[local_i]] = (
                int(frame_id), int(source['injected_id']), detection_id, status,
                x_in, y_in, x_out, y_out, distance_out,
                flux_in, flux_out, flux_ratio,
                len(nearby), n_claimants, n_original, n_injected
            )

    matches = Table(
        rows=output_rows,
        names=(
            'frame_id', 'injected_id', 'detection_id', 'status',
            'x_in_local', 'y_in_local', 'x_out_local', 'y_out_local', 'distance',
            'flux_in', 'flux_out', 'flux_ratio',
            'n_candidates', 'n_claimants', 'n_original_near', 'n_injected_near'
        ),
        dtype=('i8', 'i8', 'i8', 'U20', 'f8', 'f8', 'f8', 'f8', 'f8',
               'f8', 'f8', 'f8', 'i8', 'i8', 'i8', 'i8')
    )
    matches.meta['match_radius_pixels'] = float(match_radius)
    matches.meta['matching_rule'] = 'Only mutually unique candidates can be matched'
    matches.meta['isolation_rule'] = (
        'match_radius + 3*sigma_major' if isolation_radius is None
        else f'fixed {float(isolation_radius)} pixels')
    matches.meta['injection_overlap_rule'] = 'Flag pairs closer than 3*(sigma_major_i+sigma_major_j)'
    matches.meta['warning'] = (
        'Conservative centre-only candidate matching, not a deblender or a global '
        'optimal assignment. Ambiguous, shared and crowded cases are not automatically '
        'clean recoveries. No input/output flux-agreement cut is applied.')
    return matches


if __name__ == '__main__':
    # Artificial example: two injections compete for detection 9
    injected = Table(
        rows=[(1, 1, 30., 30., 1000., 2.),
              (1, 2, 80., 80., 1000., 2.),
              (1, 3, 82., 80., 1000., 2.)],
        names=('frame_id', 'injected_id', 'x_local', 'y_local', 'flux_in', 'sigma_major')
    )
    detected = Table(
        rows=[(1, 7, 30.3, 30.1, 950.), (1, 9, 81., 80., 2000.)],
        names=('frame_id', 'detection_id', 'x_local', 'y_local', 'flux_out')
    )
    original_detected = Table(
        names=('frame_id', 'x_local', 'y_local'), dtype=('i8', 'f8', 'f8'))
    matches = match_many_recovery(injected, detected, original_detected)
    print('DEMO DATA ONLY: not your real recovery result.\n')
    matches['frame_id', 'injected_id', 'detection_id', 'status', 'n_claimants'].pprint(
        max_lines=-1, max_width=-1)