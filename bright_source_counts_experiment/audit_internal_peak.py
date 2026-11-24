"""Verify the one-source internal-structure exclusion and show the evidence."""
from pathlib import Path
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import ndimage as ndi
from astropy.table import Table
from astropy.io import fits
from reviewed_blends import apply_reviewed_internal_peak

BASE = Path(__file__).resolve().parent
OUT = BASE/'results/aperture_review'
BEFORE = BASE/'results/aperture_before_internal_peak_review'

def run():
    old, new = Table.read(BEFORE/'sources.ecsv'), Table.read(OUT/'sources.ecsv')
    np.testing.assert_array_equal(old['id'],new['id'])
    target = np.asarray(new['id']) == 737
    assert target.sum() == 1
    for col in old.colnames:
        if old[col].dtype.kind == 'f':
            np.testing.assert_allclose(old[col][~target],new[col][~target],rtol=0,atol=0,equal_nan=True,err_msg=col)
        else:
            np.testing.assert_array_equal(old[col][~target],new[col][~target],err_msg=col)
    for col in ['x','y','segment_id','blended','aperture_a_pix','aperture_b_pix','aperture_flux_adu']:
        np.testing.assert_array_equal(old[col],new[col],err_msg=col)
    assert old['usable'][target][0] and not new['usable'][target][0]
    assert new['reviewed_internal_peak_rejected'].sum() == 1
    assert hashlib.sha256((OUT/'detection_products.fits').read_bytes()).hexdigest() == old.meta['products_sha256']
    image=fits.getdata(BASE.parent/'Astro/Fits_Data/mosaic.fits')
    with fits.open(OUT/'detection_products.fits') as f:
        residual=image-f['BACKGROUND'].data; rms=f['RMS'].data.copy(); labels=f['SOURCES'].data.copy()
    again,_=apply_reviewed_internal_peak(new.copy(),residual,rms,labels)
    for col in ['flags','usable','reviewed_blend_rejected']:
        np.testing.assert_array_equal(again[col],new[col])
    shifted=new.copy();shifted['x'][target]+=5
    try:
        apply_reviewed_internal_peak(shifted,residual,rms,labels)
    except ValueError:
        pass
    else:
        raise AssertionError('Position guard did not catch changed target.')
    oc,nc=Table.read(BEFORE/'classified_sources.ecsv'),Table.read(OUT/'classified_sources.ecsv')
    for col in ['is_star','remaining_sample','count_sample']:
        np.testing.assert_array_equal(oc[col][~target],nc[col][~target],err_msg=col)
    counts=Table.read(OUT/'number_counts.ecsv')
    mags=oc['magnitude'][oc['remaining_sample'] & ~target]
    np.testing.assert_array_equal(counts['count'],[np.sum(mags<m) for m in counts['limiting_magnitude']])
    row=new[target][0]
    x0,x1,y0,y1=631,666,2113,2145
    cut=residual[y0:y1,x0:x1]/rms[y0:y1,x0:x1]
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for j,(ax,title) in enumerate(zip(axs,['Before: usable, one catalogue label','After: suspected internal blend','Default smoothing: sigma = 1 pixel'])):
        ax.imshow(cut if j<2 else ndi.gaussian_filter(residual[y0:y1,x0:x1],1)/rms[y0:y1,x0:x1],origin='lower',cmap='gray',vmin=-1,vmax=12,extent=(x0-.5,x1-.5,y0-.5,y1-.5))
        ax.add_patch(Ellipse((row['x'],row['y']),2*row['aperture_a_pix'],2*row['aperture_b_pix'],angle=row['aperture_theta_deg'],fill=False,color='blue' if j==0 else 'red',linewidth=1.4))
        if j==1:
            peaks=Table.read(OUT/'reviewed_internal_peaks.csv',format='ascii.csv')
            peaks=peaks[peaks['smooth_sigma_pix']==.5]
            ax.scatter(peaks['x'],peaks['y'],marker='+',c='cyan',s=70,label='Local maxima at sigma = 0.5')
            ax.legend(loc='lower left',fontsize=7)
        ax.set(title=title,xlabel='x / pixel',ylabel='y / pixel',xlim=(x0,x1),ylim=(y0,y1))
        ax.tick_params(labelsize=8)
    fig.tight_layout();fig.savefig(OUT/'reviewed_internal_peak_before_after.png',dpi=150);plt.close(fig)
    print('Only ID 737 excluded; all 7037 other detections unchanged.')
    print('Apertures, fluxes, segmentation/masks, prior recoveries and star labels unchanged.')
    print('Position guard, repeatability and all cumulative count bins passed.')

if __name__=='__main__':
    run()
