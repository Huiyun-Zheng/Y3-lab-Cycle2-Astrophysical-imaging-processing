"""Apply the new masks to the saved balanced catalogue without changing detection.

Keep the old fluxes, apertures, noise maps and neighbour decisions. This is a
controlled masking comparison, not a new fit to a different segmentation map.
"""

import sys
sys.dont_write_bytecode = True

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage as ndi

import photometry_analysis as photometry
import source_detection_neighbour_model as neighbour
import detection_masked_review as detector


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'results' / 'mask_only_control'


def affected_sources(table, labels, extra_mask):
    """Use the original aperture and near-mask checks on the new mask only."""
    near = ndi.binary_dilation(extra_mask, iterations=7)
    affected = ndi.maximum(near.astype(np.uint8), labels, table['segment_id']).astype(bool)
    for i, row in enumerate(table):
        bounds, _, _, aperture, _ = neighbour.aperture_grid(row, labels.shape)
        affected[i] |= np.any(extra_mask[bounds][aperture])
    return affected


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = Table.read(BASE / 'results/balanced_recovery/sources.ecsv')
    original_usable = np.asarray(table['usable'], bool).copy()
    with fits.open(BASE / 'inputs/detection_products.fits') as source:
        products = fits.HDUList([hdu.copy() for hdu in source])
    with fits.open(BASE / 'results/masked_review/detection_products.fits') as review:
        strip = review['STRIPMASK'].data.astype(bool)
        sharp = review['SHARPMASK'].data.astype(bool)
    labels = products['SOURCES'].data
    strip_affected = affected_sources(table, labels, strip)
    sharp_affected = affected_sources(table, labels, sharp)
    table['mask_strip_excluded'] = strip_affected
    table['mask_sharp_excluded'] = sharp_affected
    table['usable'] = original_usable & ~strip_affected & ~sharp_affected
    table['flags'] = np.asarray(table['flags'], dtype='U300')
    for i in np.flatnonzero(strip_affected | sharp_affected):
        table['flags'][i] = str(table['flags'][i]) + '|review_mask'
    table.meta['control_note'] = 'Only new masks applied; original apertures/fluxes/detection unchanged.'
    table.write(OUTPUT / 'sources.ecsv', overwrite=True)

    bad = products['BADMASK'].data.astype(bool) | strip | sharp
    products['BADMASK'].data = bad.astype(np.uint8)
    products['BOUNDARY'].data = (products['BOUNDARY'].data.astype(bool) | strip).astype(np.uint8)
    products['ARTEFACT'].data = (products['ARTEFACT'].data.astype(bool) | sharp).astype(np.uint8)
    products['DETMASK'].data = ndi.binary_dilation(bad, iterations=4).astype(np.uint8)
    products.append(fits.ImageHDU(strip.astype(np.uint8), name='STRIPMASK'))
    products.append(fits.ImageHDU(sharp.astype(np.uint8), name='SHARPMASK'))
    products.writeto(OUTPUT / 'detection_products.fits', overwrite=True)

    print(f'Before masking: {original_usable.sum()}', flush=True)
    print(f'Strip including aperture/margin: {(original_usable & strip_affected).sum()}', flush=True)
    print(f'Additional sharp mask: {(original_usable & ~strip_affected & sharp_affected).sum()}', flush=True)
    print(f'After masking: {table["usable"].sum()}', flush=True)
    photometry.DETECTION_DIR = OUTPUT
    photometry.PRODUCTS_PATH = OUTPUT / 'detection_products.fits'
    photometry.OUTPUT_DIR = OUTPUT
    classified, fit = photometry.run_analysis(fit_range=(19., 23.))
    Table(rows=[('balanced_before', int(original_usable.sum())),
                ('strip_excluded', int((original_usable & strip_affected).sum())),
                ('sharp_additional', int((original_usable & ~strip_affected & sharp_affected).sum())),
                ('mask_only_usable', int(table['usable'].sum()))],
          names=['step', 'sources']).write(OUTPUT / 'mask_audit.csv', format='ascii.csv', overwrite=True)
    image = fits.getdata(BASE.parent / 'Astro/Fits_Data/mosaic.fits')
    centre_unmasked = ~products['DETMASK'].data[
        np.rint(table['y']).astype(int), np.rint(table['x']).astype(int)
    ].astype(bool)
    detector.save_diagnostics(image, products['BACKGROUND'].data, products['RMS'].data,
                              products['ARTEFACT'].data.astype(bool), products['BOUNDARY'].data.astype(bool),
                              products['SATURATED'].data.astype(bool), table[centre_unmasked], OUTPUT)
    return classified, fit


if __name__ == '__main__':
    run()
