"""Local curve-of-growth review of two sources identified in the user's image.

This is an explicit, position-selected review, not a field-wide aperture rule.
The enclosed fraction refers to a finite clean reference circle, not total flux.
"""

import numpy as np
from astropy.table import Table


# Zero-based image positions, matched to the 2026-11-22 15:32:52 screenshot.
# Use positions rather than catalogue IDs, which can change on redetection.
REVIEWED_POSITIONS = ((577.069, 4315.028), (582.845, 4222.238))
REFERENCE_RADIUS_FACTOR = 2.5
ENCLOSED_FRACTION = 0.98


def growth_radius(signal, noise, radius, angle, clean, initial_radius):
    """Find a modest larger circle only when clean, extended wings are measured."""
    reference_radius = REFERENCE_RADIUS_FACTOR * initial_radius
    reference = radius <= reference_radius
    original = radius <= initial_radius
    if not np.all(clean[reference]):
        return None, 'reference_not_clean', None

    # Retain negative pixels so noise cannot create artificial wing flux.
    base_flux = float(signal[original].sum())
    reference_flux = float(signal[reference].sum())
    if base_flux <= 0 or reference_flux <= 0:
        return None, 'nonpositive_flux', None
    wings = reference & ~original
    wing_flux = float(signal[wings].sum())
    wing_snr = wing_flux / np.sqrt(np.sum(noise[wings] ** 2))
    if wing_flux < .03 * base_flux or wing_snr < 10:
        return None, 'insufficient_wings', None

    # Do not choose a size from a curve that is still rising rapidly at its edge.
    outer = reference & (radius > .8 * reference_radius)
    if abs(float(signal[outer].sum())) > .02 * reference_flux:
        return None, 'reference_not_flat_enough', None

    radii = np.r_[np.arange(initial_radius, reference_radius, .25), reference_radius]
    flux = np.array([signal[radius <= r].sum() for r in radii], dtype=float)
    reached = np.flatnonzero(flux >= ENCLOSED_FRACTION * reference_flux)
    if not len(reached) or radii[reached[0]] <= initial_radius:
        return None, 'no_growth_needed', None
    adopted = float(radii[reached[0]])

    # Require light around most of the circle, not a single bright neighbour.
    added = (radius > initial_radius) & (radius <= adopted)
    sector_index = np.floor((angle + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    significant_sectors = 0
    for k in range(8):
        sector = added & (sector_index == k)
        if sector.any() and signal[sector].sum() > 3 * np.sqrt(np.sum(noise[sector] ** 2)):
            significant_sectors += 1
    if significant_sectors < 6:
        return None, 'asymmetric_wings', None
    curve = Table({'radius_pix': radii, 'flux_adu': flux,
                   'fraction_of_reference': flux / reference_flux})
    return adopted, 'accepted', curve


def apply_reviewed_wings(table, residual, rms, labels, bad, gain):
    """Update only the two checked, isolated circular sources and their fluxes.

    The runner applies its universal core and shared-aperture checks afterwards.
    Other catalogue rows, detections, masks and neighbour fits stay unchanged.
    """
    for field in ('pre_wing_radius_pix', 'pre_wing_flux_adu', 'wing_reference_radius_pix'):
        if field not in table.colnames:
            table[field] = np.full(len(table), np.nan)
    table['aperture_method'] = np.asarray(table['aperture_method'], dtype='U40')
    curves, decisions = [], []
    for x_expected, y_expected in REVIEWED_POSITIONS:
        distance = np.hypot(table['x'] - x_expected, table['y'] - y_expected)
        i = int(np.argmin(distance))
        if distance[i] > 1:
            raise ValueError('A reviewed source was not matched; verify its image position.')
        row = table[i]
        original_radius = float(row['pre_wing_radius_pix'])
        if not np.isfinite(original_radius):
            original_radius = float(row['aperture_a_pix'])
        original_flux = float(row['pre_wing_flux_adu'])
        if not np.isfinite(original_flux):
            original_flux = float(row['aperture_flux_adu'])
        status, adopted, curve = 'not_isolated_clean_circle', None, None

        if (row['strict_usable'] and row['usable'] and not row['blended']
                and row['aperture_shape'] == 'circle' and row['aperture_snr'] >= 500):
            x, y = float(row['x']), float(row['y'])
            half = int(np.ceil(REFERENCE_RADIUS_FACTOR * original_radius)) + 1
            x0, x1 = int(x) - half, int(x) + half + 1
            y0, y1 = int(y) - half, int(y) + half + 1
            if x0 < 0 or y0 < 0 or x1 > residual.shape[1] or y1 > residual.shape[0]:
                status = 'edge'
            else:
                yy, xx = np.mgrid[y0:y1, x0:x1]
                radius = np.hypot(xx - x, yy - y)
                angle = np.arctan2(yy - y, xx - x)
                signal, noise = residual[y0:y1, x0:x1], rms[y0:y1, x0:x1]
                lab = labels[y0:y1, x0:x1]
                clean = ~bad[y0:y1, x0:x1] & ((lab == 0) | (lab == row['segment_id']))
                clean &= np.isfinite(signal) & np.isfinite(noise) & (noise > 0)
                adopted, status, curve = growth_radius(
                    signal, noise, radius, angle, clean, original_radius,
                )
                if adopted is not None:
                    aperture = radius <= adopted
                    flux = float(signal[aperture].sum())
                    error = float(np.sqrt(np.sum(
                        noise[aperture] ** 2 + np.maximum(signal[aperture], 0) / gain,
                    )))
                    table['pre_wing_radius_pix'][i] = original_radius
                    table['pre_wing_flux_adu'][i] = original_flux
                    table['wing_reference_radius_pix'][i] = REFERENCE_RADIUS_FACTOR * original_radius
                    for name in ('aperture_a_pix', 'aperture_b_pix',
                                 'preferred_aperture_a_pix', 'preferred_aperture_b_pix'):
                        table[name][i] = adopted
                    table['aperture_flux_adu'][i] = flux
                    table['original_aperture_flux_adu'][i] = flux
                    table['aperture_flux_error_adu'][i] = error
                    table['aperture_snr'][i] = flux / error
                    table['aperture_method'][i] = 'reviewed_growth_curve'
                    curve['source_id'] = np.full(len(curve), int(row['id']))
                    curves.append(curve)

        decisions.append((int(row['id']), float(row['x']), float(row['y']), status,
                          original_radius, float(table['aperture_a_pix'][i]),
                          original_flux, float(table['aperture_flux_adu'][i])))
    audit = Table(rows=decisions, names=(
        'source_id', 'x', 'y', 'status', 'old_radius_pix', 'new_radius_pix',
        'old_flux_adu', 'new_flux_adu',
    ))
    table.meta['reviewed_wings_note'] = (
        'Only two visually reviewed positions: 98% of signed flux inside a clean '
        '2.5-times-original-radius circle. Finite-aperture flux, not total flux.'
    )
    print(audit, flush=True)
    return table, audit, curves
