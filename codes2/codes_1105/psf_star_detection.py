"""Provisional star candidates from local Moffat PSF widths.

Run separately from main.py. Unknown sources are not classified as galaxies.
Automatic reference stars and thresholds require visual validation.
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from scipy.optimize import least_squares


def moffat_pixels(x, y, amplitude, x_center, y_center, alpha, beta, background):
    """Average a circular Moffat profile over nine subpixels."""

    profile = np.zeros_like(x, dtype=float)
    for dx in (-1/3, 0, 1/3):
        for dy in (-1/3, 0, 1/3):
            r2 = (x+dx-x_center)**2 + (y+dy-y_center)**2
            profile += (1+r2/alpha**2)**(-beta)
    return background + amplitude*profile/9


def moffat_fwhm(alpha, beta):
    return 2*alpha*np.sqrt(2**(1/beta)-1)


def measure_psf_widths(data, labels, results, header, mask=None, mag_limit=22):
    """Fit Moffat widths without changing the photometry catalogue."""

    if data.shape != labels.shape:
        raise ValueError('data and labels must have the same shape')
    if mask is not None and np.shape(mask) != data.shape:
        raise ValueError('mask and data must have the same shape')

    catalogue = results.copy()
    n = len(catalogue)
    for name in ('magnitude', 'psf_fwhm', 'fwhm_error', 'fit_snr', 'reduced_chi2', 'moffat_beta'):
        catalogue[name] = np.full(n, np.nan)
    catalogue['fit_status'] = np.full(n, 'not_fitted', dtype='U24')
    gain = float(header['GAIN'])
    saturation = float(header.get('SATURATE', 36000))
    if gain <= 0 or header['EXPTIME'] <= 0:
        raise ValueError('GAIN and EXPTIME must be positive')

    valid = ((catalogue['flag'] == 0) & np.isfinite(catalogue['flux'])
             & (catalogue['flux'] > 0))
    catalogue['magnitude'][valid] = header['MAGZPT'] - 2.5 * np.log10(
        catalogue['flux'][valid] * gain / header['EXPTIME'])
    height, width = data.shape

    for i in np.flatnonzero(valid & (catalogue['magnitude'] <= mag_limit)):
        row = catalogue[i]
        xc, yc = float(row['x_center']), float(row['y_center'])
        if not np.isfinite(xc + yc):
            continue
        x0, y0 = int(round(xc)), int(round(yc))
        if x0 < 30 or y0 < 30 or x0 + 30 >= width or y0 + 30 >= height:
            catalogue['fit_status'][i] = 'edge'
            continue

        section = data[y0-30:y0+31, x0-30:x0+31]
        section_labels = labels[y0-30:y0+31, x0-30:x0+31]
        yy, xx = np.mgrid[y0-30:y0+31, x0-30:x0+31]
        rr = np.hypot(xx-xc, yy-yc)
        good = np.isfinite(section)
        if mask is not None:
            good &= ~np.asarray(mask[y0-30:y0+31, x0-30:x0+31], dtype=bool)
        if np.any(~good & (rr < 3)):
            catalogue['fit_status'][i] = 'masked_core'
            continue
        neighbours = (section_labels > 0) & (section_labels != row['label'])
        if np.any(neighbours & (rr < 3)):
            catalogue['fit_status'][i] = 'crowded'
            continue
        sky_pixels = good & (rr >= 22) & (rr <= 30) & (section_labels == 0)
        if sky_pixels.sum() < 50:
            catalogue['fit_status'][i] = 'no_local_sky'
            continue
        _, sky, noise = sigma_clipped_stats(section[sky_pixels], sigma=3)
        fit_pixels = good & ~neighbours & (rr <= 9)
        if fit_pixels.sum() < 100 or not np.isfinite(noise) or noise <= 0:
            catalogue['fit_status'][i] = 'insufficient_pixels'
            continue
        if np.any(section[rr <= 9] >= saturation):
            catalogue['fit_status'][i] = 'saturated'
            continue

        x, y, values = xx[fit_pixels], yy[fit_pixels], section[fit_pixels]
        error = np.sqrt(noise**2 + np.maximum(values-sky, 0) / gain)
        peak = max(float(np.max(values)-sky), 1)
        initial = [peak, xc, yc, 4., 2.5, float(sky)]
        lower = [0, xc-2, yc-2, 0.5, 1.1, sky-5*noise]
        upper = [np.inf, xc+2, yc+2, 20, 20, sky+5*noise]
        try:
            fit = least_squares(
                lambda p: (moffat_pixels(x, y, *p)-values) / error,
                initial, bounds=(lower, upper), x_scale='jac', max_nfev=100)
            if not fit.success or np.any(fit.active_mask):
                catalogue['fit_status'][i] = 'fit_limit'
                continue
            covariance = np.linalg.inv(fit.jac.T @ fit.jac)
            errors = np.sqrt(np.diag(covariance))
            if not np.all(np.isfinite(errors)) or errors[0] <= 0:
                continue
            alpha, beta = fit.x[3:5]
            fwhm = moffat_fwhm(alpha, beta)
            gradient = np.array([fwhm/alpha,
                -alpha*np.log(2)*2**(1/beta)/(beta**2*np.sqrt(2**(1/beta)-1))])
            variance = gradient @ covariance[3:5, 3:5] @ gradient
            catalogue['psf_fwhm'][i] = fwhm
            catalogue['fwhm_error'][i] = np.sqrt(max(variance, 0))
            catalogue['moffat_beta'][i] = beta
            catalogue['fit_snr'][i] = fit.x[0] / errors[0]
            catalogue['reduced_chi2'][i] = np.sum(fit.fun**2) / (len(values)-6)
            catalogue['fit_status'][i] = 'ok'
        except (ValueError, np.linalg.LinAlgError):
            catalogue['fit_status'][i] = 'fit_failed'

    catalogue['psf_fwhm'].unit = 'pix'
    catalogue['fwhm_error'].unit = 'pix'
    catalogue.meta['width_method'] = 'circular Moffat with 3x3 subpixel integration, radius 9 pixels'
    return catalogue


def select_psf_candidates(catalogue, reference_labels=None, mag_limit=21):
    """Compare widths with provisional references; return candidate classes."""

    catalogue = catalogue.copy()
    fwhm = np.asarray(catalogue['psf_fwhm'])
    error = np.asarray(catalogue['fwhm_error'])
    quality = ((catalogue['fit_status'] == 'ok') & (catalogue['fit_snr'] >= 50)
               & (catalogue['magnitude'] <= mag_limit)
               & np.isfinite(fwhm) & np.isfinite(error))

    if reference_labels is None:
        # The width peak is a provisional stellar locus
        pool = quality & (catalogue['fit_snr'] >= 100)
        pool &= (catalogue['reduced_chi2'] < 10) & (fwhm >= 2) & (fwhm <= 9)
        if np.count_nonzero(pool) < 5:
            raise ValueError('Too few reference candidates; supply checked reference_labels')
        bins = np.arange(2, 9.26, 0.25)
        histogram, edges = np.histogram(fwhm[pool], bins=bins)
        j = int(np.argmax(histogram))
        mode = (edges[j] + edges[j+1]) / 2
        references = pool & (np.abs(fwhm-mode) <= 0.4)
    else:
        references = quality & np.isin(catalogue['label'], reference_labels)
    if np.count_nonzero(references) < 5:
        raise ValueError('At least five usable reference candidates are required')

    catalogue['psf_reference'] = references
    catalogue['local_psf_fwhm'] = np.full(len(catalogue), np.nan)
    catalogue['width_tolerance'] = np.full(len(catalogue), np.nan)
    catalogue['source_class'] = np.full(len(catalogue), 'unknown', dtype='U24')
    reference_rows = np.flatnonzero(references)
    q_limit = max(3., 2*np.percentile(catalogue['reduced_chi2'][references], 90))
    for i in np.flatnonzero(quality):
        # Nearby references allow a slowly varying PSF
        other = reference_rows[reference_rows != i]
        distance = np.hypot(catalogue['x_center'][other]-catalogue['x_center'][i],
                            catalogue['y_center'][other]-catalogue['y_center'][i])
        nearby = other[np.argsort(distance)[:8]]
        local_widths = fwhm[nearby]
        psf_width = np.median(local_widths)
        scatter = 1.4826*np.median(np.abs(local_widths-psf_width))
        tolerance = max(0.15*psf_width, 3*np.hypot(scatter, error[i]))
        catalogue['local_psf_fwhm'][i] = psf_width
        catalogue['width_tolerance'][i] = tolerance
        if catalogue['reduced_chi2'][i] > q_limit:
            continue
        difference = fwhm[i]-psf_width
        if abs(difference) <= tolerance:
            catalogue['source_class'][i] = 'star_candidate'
        elif difference > tolerance:
            catalogue['source_class'][i] = 'extended_candidate'

    catalogue.meta['classification_warning'] = (
        'Width-based candidates only. Automatic reference selection is provisional; '
        'inspect references and candidates. Compact galaxies can be star candidates. '
        'Thresholds are initial choices, not calibrated classification probabilities.')
    catalogue.meta['classification_mag_limit'] = float(mag_limit)
    catalogue.meta['reference_selection'] = (
        'automatic width peak' if reference_labels is None else 'supplied labels')
    return catalogue


def star_candidate_mask(labels, catalogue):
    """Mask candidate segments only; review the candidates before using it."""

    ids = catalogue['label'][catalogue['source_class'] == 'star_candidate']
    return np.isin(labels, ids)


def plot_psf_candidates(data, catalogue, output_dir, x_offset=0, y_offset=0, show=True):
    """Save the size plot and small reference/candidate cutouts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for kind, color in [('unknown', '0.7'), ('extended_candidate', 'tab:blue'),
                        ('star_candidate', 'tab:orange')]:
        selected = catalogue['source_class'] == kind
        ax.scatter(catalogue['magnitude'][selected], catalogue['psf_fwhm'][selected],
                   s=12, c=color, label=f'{kind}: {np.count_nonzero(selected)}')
    refs = catalogue['psf_reference']
    ax.scatter(catalogue['magnitude'][refs], catalogue['psf_fwhm'][refs],
               s=45, facecolors='none', edgecolors='black', label='Reference candidates')
    ax.set(xlabel='Magnitude', ylabel='Moffat FWHM (pixels)',
           title='Provisional PSF width classification', ylim=(0, 15))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir/'size_magnitude.png', dpi=180)
    if show:
        plt.show()
    plt.close(fig)

    for name, selected in [('references', refs),
                            ('star_candidates', catalogue['source_class'] == 'star_candidate')]:
        indices = np.flatnonzero(selected)
        indices = indices[np.argsort(catalogue['magnitude'][indices])][:24]
        if len(indices) == 0:
            continue
        fig, axes = plt.subplots(int(np.ceil(len(indices)/6)), 6,
                                  figsize=(15, 3.2*int(np.ceil(len(indices)/6))), squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for ax, i in zip(axes.ravel(), indices):
            row = catalogue[i]
            x, y = int(round(row['x_center'])), int(round(row['y_center']))
            cutout = data[max(0,y-14):y+15, max(0,x-14):x+15]
            low, high = np.percentile(cutout[np.isfinite(cutout)], [10, 99])
            ax.imshow(cutout, origin='lower', cmap='gray', vmin=low, vmax=high)
            ax.set_title(f"ID {row['label']} | m={row['magnitude']:.2f}\n"
                         f"({row['x_center']+x_offset:.1f}, {row['y_center']+y_offset:.1f})",
                         fontsize=8)
        fig.suptitle(f'{name}: inspect before masking; original pixel coordinates')
        fig.subplots_adjust(top=.88, hspace=.45, wspace=.08)
        fig.savefig(output_dir/f'{name}.png', dpi=160)
        if show:
            plt.show()
        plt.close(fig)


def main(catalogue_path=None, output_dir=None, show=True, fits_path=None,
         crop_x=(70, 2450), crop_y=(0, 4550), reference_labels=None):
    """Run independently; leave the main photometry selection unchanged."""

    code_dir = Path(__file__).resolve().parent
    if fits_path is None:
        fits_path = code_dir.parent.parent/'Astro/Fits_Data/mosaic.fits'
    with fits.open(fits_path, memmap=False) as hdul:
        header = hdul[0].header.copy()
        original = hdul[0].data
        if not (0 <= crop_x[0] < crop_x[1] <= original.shape[1]
                and 0 <= crop_y[0] < crop_y[1] <= original.shape[0]):
            raise ValueError('Crop must lie inside the original image')
        data = original[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]].astype(float)

    from source_detection import detect_sources, deblend
    from bleeding import mask_saturated
    from photometry import adjustable_aperture
    parent_labels, mean, stddev = detect_sources(data, threshold=3, min_area=4)
    labels = deblend(data, parent_labels)
    bad_mask = mask_saturated(data, parent_labels, saturation=header.get('SATURATE', 36000))
    if catalogue_path is None:
        results = adjustable_aperture(data, labels, mean, stddev, mask=bad_mask)
    else:
        results = Table.read(catalogue_path)
        if not np.array_equal(np.sort(results['label']), np.unique(labels[labels > 0])):
            raise ValueError('Catalogue labels do not match this detection/crop')
        if not np.isclose(results.meta.get('background', np.nan), mean, atol=1e-5, rtol=0):
            raise ValueError('Catalogue background does not match this detection/crop')

    catalogue = measure_psf_widths(data, labels, results, header, mask=bad_mask)
    catalogue = select_psf_candidates(catalogue, reference_labels=reference_labels)
    catalogue.meta['crop_x'] = list(crop_x)
    catalogue.meta['crop_y'] = list(crop_y)
    catalogue.meta['detection_threshold'] = 3.
    catalogue.meta['detection_min_area'] = 4
    catalogue.meta['mask_note'] = 'Saturated parent labels excluded; main.py manual masks not added here'
    output_dir = Path(output_dir) if output_dir is not None else code_dir/'outputs/psf_stars'
    output_dir.mkdir(parents=True, exist_ok=True)
    catalogue['x_original'] = catalogue['x_center'] + crop_x[0]
    catalogue['y_original'] = catalogue['y_center'] + crop_y[0]
    catalogue.write(output_dir/'psf_catalogue.ecsv', overwrite=True)
    candidates = catalogue[catalogue['source_class'] == 'star_candidate']
    candidates.write(output_dir/'star_candidates.csv', overwrite=True)
    candidate_mask = star_candidate_mask(labels, catalogue)
    mask_header = fits.Header()
    mask_header['XOFFSET'] = crop_x[0]
    mask_header['YOFFSET'] = crop_y[0]
    mask_header['COMMENT'] = 'Provisional star-candidate segments; not a validated exclusion mask'
    fits.writeto(output_dir/'star_candidate_mask.fits', candidate_mask.astype('uint8'),
                  header=mask_header, overwrite=True)
    plot_psf_candidates(data, catalogue, output_dir,
                        x_offset=crop_x[0], y_offset=crop_y[0], show=show)
    kinds, numbers = np.unique(catalogue['source_class'], return_counts=True)
    print(dict(zip(kinds.tolist(), numbers.tolist())))
    reference_width = np.median(catalogue['psf_fwhm'][catalogue['psf_reference']])
    print(f'Reference median FWHM: {reference_width:.2f} pixels')
    print('Provisional candidates only; the photometry selection has not changed.')
    return catalogue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect Moffat PSF width candidates')
    parser.add_argument('--catalogue', type=Path)
    parser.add_argument('--fits-path', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--crop-x', type=int, nargs=2, default=(70, 2450))
    parser.add_argument('--crop-y', type=int, nargs=2, default=(0, 4550))
    parser.add_argument('--reference-labels', type=int, nargs='+')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    main(args.catalogue, args.output_dir, show=not args.no_show, fits_path=args.fits_path,
         crop_x=args.crop_x, crop_y=args.crop_y, reference_labels=args.reference_labels)
