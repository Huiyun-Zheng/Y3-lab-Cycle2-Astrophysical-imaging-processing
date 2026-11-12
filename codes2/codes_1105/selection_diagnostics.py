from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from photutils.aperture import EllipticalAperture


def selection_diagnostics(
        data, results, mean, header, mag_limits=(18, 20, 22, 24),
        fit_range=(18, 24), output_dir=None, show=True
        ):
    """Check selection losses; rejected-source magnitudes are estimates only."""

    flags = np.asarray(results['flag'], dtype=int)
    flux = np.asarray(results['flux'], dtype=float)
    used = (flags == 0) & np.isfinite(flux) & (flux > 0)
    diagnostic_flux = flux.copy()

    # Rejected rows retain a fallback ellipse
    for i in np.flatnonzero(flags != 0):
        row = results[i]
        shape = np.array([row[name] for name in
                          ('x_center', 'y_center', 'a', 'b', 'theta')])
        if not np.all(np.isfinite(shape)) or np.any(shape[2:4] <= 0):
            continue

        x, y, a, b, theta = shape
        aperture = EllipticalAperture((x, y), a, b, theta=theta)
        aperture_mask = aperture.to_mask(method='exact')
        image_slice, mask_slice = aperture_mask.get_overlap_slices(data.shape)
        if image_slice is None:
            continue

        weights = aperture_mask.data[mask_slice]
        inside = weights > 0
        values = data[image_slice]
        if np.all(np.isfinite(values[inside])):
            diagnostic_flux[i] = np.sum((values[inside] - mean) * weights[inside])

    # Use the laboratory calibration convention
    measurable = np.isfinite(diagnostic_flux) & (diagnostic_flux > 0)
    magnitude = np.full(len(results), np.nan)
    magnitude[measurable] = header['MAGZPT'] - 2.5 * np.log10(
        diagnostic_flux[measurable] * header['GAIN'] / header['EXPTIME']
    )

    reasons = [
        ('All detected labels', len(results)),
        ('Used for magnitudes', np.count_nonzero(used)),
        ('Excluded total', np.count_nonzero(~used)),
        ('Edge flag', np.count_nonzero(flags & 1)),
        ('Neighbour flag', np.count_nonzero(flags & 2)),
        ('Mask/nonfinite flag', np.count_nonzero(flags & 4)),
        ('Invalid ellipse flag', np.count_nonzero(flags & 8)),
        ('Neighbour only', np.count_nonzero(flags == 2)),
        ('Unflagged nonpositive/nonfinite flux', np.count_nonzero((flags == 0) & ~used))
    ]
    summary = Table(rows=reasons, names=('selection', 'number'))

    rows = []
    for limit in mag_limits:
        brighter = measurable & (magnitude < limit)
        rows.append((
            limit, np.count_nonzero(brighter),
            np.count_nonzero(brighter & used),
            np.count_nonzero(brighter & ~used)
        ))
    counts = Table(rows=rows, names=(
        'mag_limit', 'all_estimated', 'used', 'excluded_estimated'
    ))

    print('\nSelection summary (flag reasons can overlap):')
    summary.pprint(max_lines=-1, max_width=-1)
    print('\nCumulative counts: N(m < limit)')
    counts.pprint(max_lines=-1, max_width=-1)
    print('Rejected-source estimates may include neighbours, saturation, or partial apertures.')
    print('These estimates are NOT a completeness correction.')

    thresholds = np.arange(fit_range[0], fit_range[1] + 0.001, 0.25)
    fig, ax = plt.subplots(figsize=(7, 5))
    for selected, label, style in (
            (measurable, 'All estimated (diagnostic only)', '--'),
            (used, 'Used for magnitudes', '-')):
        populations = np.array([np.count_nonzero(selected & (magnitude < m))
                                for m in thresholds])
        positive = populations > 0
        ax.plot(thresholds[positive], np.log10(populations[positive]),
                style, label=label)
        if np.count_nonzero(positive) >= 2:
            slope = np.polyfit(thresholds[positive],
                               np.log10(populations[positive]), 1)[0]
            print(f'{label}: slope = {slope:.4f} over {fit_range}')

    ax.set_xlabel('Magnitude threshold')
    ax.set_ylabel('log10 N(m < threshold)')
    ax.set_title('Selection diagnostic')
    ax.legend()
    fig.tight_layout()

    catalogue = results.copy()
    catalogue['diagnostic_flux'] = diagnostic_flux
    catalogue['diagnostic_mag'] = magnitude
    catalogue['used_for_counts'] = used
    catalogue.meta['diagnostic_warning'] = (
        'Rejected rows use stored fallback ellipses; their fluxes may be contaminated '
        'or truncated. Not reliable photometry or a completeness correction.'
    )
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.write(output_dir / 'selection_summary.csv', overwrite=True)
        counts.write(output_dir / 'selection_counts.csv', overwrite=True)
        catalogue.write(output_dir / 'selection_catalogue.ecsv', overwrite=True)
        fig.savefig(output_dir / 'selection_nm.png', dpi=200)

    if show:
        plt.show()
    plt.close(fig)
    return summary, counts, catalogue
