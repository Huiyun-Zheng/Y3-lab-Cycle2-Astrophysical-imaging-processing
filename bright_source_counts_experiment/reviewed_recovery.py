"""Position-selected remeasurement of the seven screenshot examples.

The first and fourth screenshots show the same source. This review affects
only these six positions, not global detection or aperture thresholds.
Shared observed light and modelled neighbour contamination are different:
the former can largely belong to the bright target itself.
"""

from dataclasses import replace
import numpy as np
from astropy.table import Table

import aperture_neighbour_model as neighbour
from run_experiment import review_saved_scores


# Zero-based positions matched to the 2026-11-22 15:53--15:56 screenshots.
# The last source still requires a reliable outer-wing aperture, as in ID 4.
POSITIONS = (
    (988.0145, 184.2612, '1,4', False),
    (1118.4881, 282.9907, '2', False),
    (1733.3550, 853.7287, '3', False),
    (1918.1644, 1898.3077, '5', False),
    (1482.6579, 793.0499, '6', False),
    (1597.8166, 464.4759, '7', True),
)
# Provisional quality tolerances, not calibrated confidence intervals.
# Three percent neighbour light corresponds to about 0.032 mag if uncorrected.
MAX_NEIGHBOUR_FLUX = .03
MAX_SHARED_FLUX = .02
MAX_START_DISAGREEMENT = .01
MINIMUM_APERTURE_SNR = 50.
RECOVERY_FLAG = 'local_aperture_review_recovered'


def match_positions(table):
    indices = []
    for x, y, _, _ in POSITIONS:
        matches = np.flatnonzero(np.hypot(table['x'] - x, table['y'] - y) < 1.)
        if len(matches) != 1:
            raise ValueError('A locally reviewed source changed position; inspect it again.')
        indices.append(int(matches[0]))
    return indices


def recovery_decision(row, fit_row, aperture_clean, partner_retained, unresolved_wings):
    """Keep core, mask, profile-stability and wing limitations explicit."""
    if unresolved_wings:
        return False, 'outer_wings_unresolved'
    if not aperture_clean:
        return False, 'aperture_not_clean'
    if row['near_boundary'] or row['aperture_capped'] or row['elongation'] > 6:
        return False, 'hard_quality_failure'
    if row['neighbour_core_rejected'] or row['neighbour_peak_inside']:
        return False, 'neighbour_core_intrusion'
    if not fit_row['model_valid']:
        return False, str(fit_row['model_status'])
    if not fit_row['usable']:
        return False, 'model_selection_failed'
    if not np.isfinite(fit_row['model_flux_fraction']) or fit_row['model_flux_fraction'] > MAX_NEIGHBOUR_FLUX:
        return False, 'neighbour_flux_too_large'
    if fit_row['model_score_disagreement'] > MAX_START_DISAGREEMENT:
        return False, 'profile_sensitive'
    if not np.isfinite(fit_row['aperture_snr']) or fit_row['aperture_snr'] < MINIMUM_APERTURE_SNR:
        return False, 'low_aperture_snr'
    if row['shared_aperture_rejected']:
        if row['shared_aperture_flux_fraction'] > MAX_SHARED_FLUX:
            return False, 'shared_light_too_large'
        if partner_retained:
            return False, 'shared_partner_retained'
    return True, 'aperture_and_model_passed'


def apply_reviewed_recovery(table, residual, rms, bad, detection_mask, saturation,
                            gain, settings, shared_pairs):
    """Refit selected groups, but copy no group-wide changes into the catalogue.

    The segmentation-wide near_artefact flag is retained as historical context.
    Recovery instead requires the complete measurement aperture to avoid even
    the padded detection mask, and a stable, small fitted background correction.
    Neither source flux nor neighbour model light is subtracted or replaced.
    """
    indices = match_positions(table)
    work = table.copy()
    work['usable'] = np.asarray(work['strict_usable'], bool).copy()
    work['flags'] = work['original_flags'].copy()
    clean = {}
    for i in indices:
        bounds, _, _, aperture, edge = neighbour.aperture_grid(table[i], residual.shape)
        clean[i] = bool(not edge and np.any(aperture)
                        and not np.any(detection_mask[bounds][aperture])
                        and not np.any(bad[bounds][aperture])
                        and np.all(np.isfinite(residual[bounds][aperture])))
        if clean[i]:
            # Temporary fit request only: do not change the original segmentation
            # flags. An isolated target also needs a fit before mask-proximity recovery.
            work['near_artefact'][i] = False
            work['usable'][i] = False
            work['blended'][i] = True
    fit_settings = replace(settings, max_nfev=600, profile_betas=(2., 3., 6.))
    fitted, _ = neighbour.score_catalogue(
        work, residual, rms, bad, saturation, fit_settings, gain,
        only_parents=np.unique(work['parent_id'][indices]),
    )
    fitted = review_saved_scores(fitted, residual, fit_settings)
    table['local_review_recovered'] = np.zeros(len(table), bool)
    table['local_review_aperture_clean'] = np.zeros(len(table), bool)
    table['local_review_status'] = np.full(len(table), 'not_reviewed', dtype='U48')
    metrics = ['model_flux_fraction', 'model_area_fraction', 'model_score_disagreement',
               'model_sky_flux_fraction', 'model_bound_neighbour_fraction']
    for name in metrics:
        table['review_' + name] = np.full(len(table), np.nan)
    table['flags'] = np.asarray(table['flags'], dtype='U500')
    decisions = []
    initial_usable = np.asarray(table['usable'], bool).copy()
    id_to_index = {int(r['id']): i for i, r in enumerate(table)}
    proposals = {}
    partner_indices = {}
    for i, (_, _, _, wings) in zip(indices, POSITIONS):
        ident = int(table['id'][i])
        pairs = shared_pairs[shared_pairs['pair_rejected'] & (
            (shared_pairs['source_id_1'] == ident) | (shared_pairs['source_id_2'] == ident))]
        partners = [int(p['source_id_2']) if int(p['source_id_1']) == ident
                    else int(p['source_id_1']) for p in pairs]
        partner_indices[i] = [id_to_index[p] for p in partners]
        proposals[i] = recovery_decision(table[i], fitted[i], clean[i],
                                         any(initial_usable[j] for j in partner_indices[i]), wings)
    # Do not let two proposed recoveries bypass each other's shared-light veto.
    proposed_indices = {i for i, (accept, _) in proposals.items() if accept}
    for i in proposed_indices:
        if any(j in proposed_indices for j in partner_indices[i]):
            proposals[i] = False, 'shared_partner_retained'
    for i, (_, _, screenshots, _) in zip(indices, POSITIONS):
        accepted, reason = proposals[i]
        table['local_review_status'][i] = reason
        table['local_review_aperture_clean'][i] = clean[i]
        for name in metrics:
            table['review_' + name][i] = fitted[name][i]
        if accepted:
            table['usable'][i] = True
            table['local_review_recovered'][i] = True
            # Existing apertures and signed flux sums remain exactly unchanged.
            table['aperture_snr'][i] = fitted['aperture_snr'][i]
            table['aperture_flux_error_adu'][i] = fitted['aperture_flux_error_adu'][i]
            parts = [p for p in str(table['flags'][i]).split('|') if p not in ('none', RECOVERY_FLAG)]
            table['flags'][i] = '|'.join(parts + [RECOVERY_FLAG])
        decisions.append((screenshots, int(table['id'][i]), float(table['x'][i]),
                          float(table['y'][i]), accepted, reason,
                          str(fitted['model_status'][i]),
                          *[float(fitted[name][i]) for name in metrics]))
    table.meta['local_recovery_note'] = (
        'Six screenshot-selected positions only. Padded mask/core veto retained; '
        'three-start neighbour fit must pass original model quality cuts plus '
        'flux <=3%, start disagreement <=1%, measured S/N >=50. Shared-light '
        'exception requires own shared flux <=2% and all rejected partners excluded. '
        'Original shared/near-artefact flags retained; no aperture changes or flux subtraction.'
    )
    audit = Table(rows=decisions, names=['screenshots', 'source_id', 'x', 'y',
        'recovered', 'decision', 'refit_status', *metrics])
    print(audit[['source_id', 'recovered', 'decision']], flush=True)
    return table, audit
