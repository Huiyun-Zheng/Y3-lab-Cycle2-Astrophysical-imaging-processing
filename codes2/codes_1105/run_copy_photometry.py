"""Connect source_detection_copy to existing photometry and recovery tables.

Example:
    python run_copy_photometry.py --fits /path/to/mosaic.fits --output outputs/copy

The existing AB_calibration formula is used unchanged. Its reported error
contains only the zero-point uncertainty, not the flux uncertainty.
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from source_detection_copy import run_detection
from photometry import (
    AB_calibration, photometry_from_detection, plot_aperture_catalogue,
)


def run_pipeline(data, output_dir, settings=None, *, header=None):
    """Return (photometry_table, badmask, processed); retain all detections."""
    catalogue, badmask, processed = run_detection(
        data, output_dir, settings, header=header,
    )
    results = photometry_from_detection(catalogue)
    results.write(Path(output_dir) / 'photometry.ecsv', overwrite=True)
    return results, badmask, processed


def recovery_rows(results, trial_id, *, x_offset=0.0, y_offset=0.0):
    """Build match_recovery's detected table for one trial, without filtering.

    x_local/y_local are coordinates in the pipeline's input array. Supply
    the cutout origin as offsets when global positions are also needed.
    An existing detection may have NaN flux or usable=False; retain it.
    """
    detected = Table()
    detected['trial_id'] = np.full(len(results), trial_id, dtype=np.int64)
    for name, source in {
        'detection_id': 'id',
        'segment_id': 'label',
        'x_local': 'x_center',
        'y_local': 'y_center',
        'flux_out': 'flux',
        'usable': 'usable',
        'flags': 'flags',
    }.items():
        detected[name] = results[source].copy()
    detected['x_out'] = detected['x_local'] + x_offset
    detected['y_out'] = detected['y_local'] + y_offset
    detected.meta['flux_note'] = 'flux_out is background-subtracted aperture flux'
    detected.meta['selection_note'] = (
        'All rows retained. Position recovery, valid flux, and usable selection '
        'are separate decisions. match_recovery currently requires positive '
        'finite flux for its matched status.'
    )
    return detected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trial-id', type=int)
    args = parser.parse_args()

    with fits.open(args.fits) as hdul:
        image = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()

    results, _, _ = run_pipeline(args.fits, args.output)

    fig, _ = plot_aperture_catalogue(
        image, results,
        filename=args.output / 'photometry_apertures.png', show=False,
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    # AB_calibration returns only selected rows. Restore alignment to all
    # detection rows before saving or associating magnitudes with positions.
    selected = (
        (results['flag'] == 0)
        & np.isfinite(results['flux'])
        & (results['flux'] > 0)
    )
    results['used_for_magnitudes'] = selected
    results['magnitude'] = np.full(len(results), np.nan)
    results['magnitude_zp_error'] = np.full(len(results), np.nan)
    if np.any(selected):
        magnitudes, zp_errors = AB_calibration(
            results, header, show=False,
            filename=args.output / 'magnitude_distribution.png',
        )
        results['magnitude'][selected] = magnitudes
        results['magnitude_zp_error'][selected] = zp_errors
        plt.close('all')
    results.meta['magnitude_note'] = (
        'Existing AB_calibration convention retained; '
        'magnitude_zp_error is zero-point uncertainty only.'
    )
    results.write(args.output / 'photometry.ecsv', overwrite=True)

    if args.trial_id is not None:
        detected = recovery_rows(results, args.trial_id)
        detected.write(args.output / 'recovery_detections.ecsv', overwrite=True)


if __name__ == '__main__':
    main()
