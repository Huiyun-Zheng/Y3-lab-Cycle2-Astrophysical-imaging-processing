"""Small injection/recovery experiment on actual image cutouts.

This is a conditional detection/aperture-selection diagnostic, NOT a measured
selection function for the final galaxy catalogue. Background and mask maps
are fixed; hand reviews, neighbour-model rescue and star classification are not
rerun. Never apply these recovery fractions as weights to the observed counts.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
import contextlib
import io
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import detection_aperture_review as detector
import aperture_neighbour_model as neighbour
import photometry_analysis as photo
from run_aperture_review import reject_neighbour_cores, reject_shared_apertures
from field_internal_peaks import scan_internal_peaks
from run_field_quality_review import significant_structure

OUTPUT=HERE/'results/injection'
PRODUCTS=BASE/'results/aperture_review/detection_products.fits'
IMAGE=BASE.parent/'Astro/Fits_Data/mosaic.fits'
SEED=20261123
HALF=64
N_POSITIONS=40
MAGNITUDES=(19.,20.,21.,22.,23.,24.)
PROFILES=('point','exponential_re2','exponential_re4')


def source_profile(profile, phase_x, phase_y, angle, fwhm=4.):
    """Pixel-integrated, positive, unit-flux profiles on a 3x sampled grid.

    The two galaxies are exponential disks with intrinsic semi-major half-light
    radii 2 and 4 pixels, axis ratio 0.7, convolved with a Moffat PSF. Point
    sources are a control and must not stand in for galaxy completeness.
    """
    size=2*HALF+1; oversample=3
    coords=(np.arange(size*oversample)+.5)/oversample-.5-HALF
    yy,xx=np.meshgrid(coords,coords,indexing='ij')
    beta=3.5;alpha=fwhm/(2*np.sqrt(2**(1/beta)-1))
    psf=(1+(xx**2+yy**2)/alpha**2)**(-beta);psf/=psf.sum()
    dx,dy=xx-phase_x,yy-phase_y
    if profile=='point':
        model=(1+(dx**2+dy**2)/alpha**2)**(-beta)
    else:
        re=2. if profile=='exponential_re2' else 4.
        u=dx*np.cos(angle)+dy*np.sin(angle)
        v=-dx*np.sin(angle)+dy*np.cos(angle)
        disk=np.exp(-1.67834699*np.sqrt(u*u+(v/.7)**2)/re)
        model=fftconvolve(disk,psf,mode='same')
    image=np.maximum(model,0).reshape(size,oversample,size,oversample).sum(axis=(1,3))
    return image/image.sum()


def wilson(success,total):
    # One-standard-deviation Wilson interval for a binomial proportion.
    p=success/total;centre=(p+.5/total)/(1+1/total)
    half=np.sqrt(p*(1-p)/total+.25/total**2)/(1+1/total)
    return p,centre-half,centre+half


def make_summary(trials):
    rows=[]
    for profile in PROFILES:
        for mag in MAGNITUDES:
            t=trials[(trials['profile']==profile)&(trials['input_total_mag']==mag)]
            row=[profile,mag,len(t)]
            for name in ['detected','strict_aperture_pass','error_rule_pass']:
                row.extend(wilson(int(sum(t[name])),len(t)))
            good=t['detected'] & np.isfinite(t['true_aperture_fraction'])
            row.append(float(np.median(t['true_aperture_fraction'][good])) if good.any() else np.nan)
            rows.append(row)
    names=['profile','input_total_mag','trials']
    for n in ['detected','strict_aperture','error_rule']:names.extend([n+'_fraction',n+'_lower',n+'_upper'])
    names.append('median_true_aperture_fraction')
    summary=Table(rows=rows,names=names)
    summary.write(OUTPUT/'recovery_summary.ecsv',overwrite=True)
    fig,axes=plt.subplots(1,3,figsize=(16,5),sharex=True,sharey=True)
    for ax,stage,title in zip(axes,['detected','strict_aperture','error_rule'],
            ['Distinct detection','Strict aperture + core/shared/peak checks',
             'Same checks + random mag error <=0.10']):
        for profile in PROFILES:
            t=summary[summary['profile']==profile];y=np.asarray(t[stage+'_fraction'])
            ax.errorbar(t['input_total_mag'],y,yerr=np.vstack([y-t[stage+'_lower'],t[stage+'_upper']-y]),
                        fmt='o-',capsize=3,label=profile)
        ax.set(title=title,xlabel='Injected total magnitude',ylim=(-.03,1.03));ax.grid(alpha=.2)
    axes[0].set_ylabel('Recovery fraction');axes[0].legend(fontsize=9)
    fig.suptitle('Conditional injection pilot: 40 placements per profile/magnitude; fixed masks/background\n'
                 'No neighbour-model rescue or star classification; not a final completeness correction',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(OUTPUT/'injection_recovery.png',dpi=150);plt.close(fig)
    print(summary,flush=True)
    return summary


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    image,header=fits.getdata(IMAGE,header=True)
    with fits.open(PRODUCTS) as f:
        bg=f['BACKGROUND'].data.copy();rms=f['RMS'].data.copy();labels=f['SOURCES'].data.copy()
        bad=f['BADMASK'].data.astype(bool);mask=f['DETMASK'].data.astype(bool)
        artefact=f['ARTEFACT'].data.astype(bool);boundary=f['BOUNDARY'].data.astype(bool)
    residual=image-bg;gain=float(header['GAIN']);settings=detector.Settings()
    # Recompute the smoothed noise once from the original data, then hold it
    # fixed for every injection so that changing background estimates cannot
    # masquerade as recovery. The original initial-source mask is approximated
    # from the saved final residual; this limitation is explicit in the report.
    smoothed=ndi.gaussian_filter(np.where(bad,0,residual),settings.smooth_sigma)
    source_mask=ndi.binary_dilation(residual>3*rms,iterations=3)
    smooth_level,smooth_noise=detector.background_maps(smoothed,mask|source_mask,settings.background_box)
    forbidden=mask|ndi.binary_dilation(labels>0,iterations=3)
    original=Table.read(BASE/'results/field_quality_review/sources.ecsv')
    peak_tree=cKDTree(np.column_stack([original['peak_x'],original['peak_y']]))
    positions=[]
    while len(positions)<N_POSITIONS:
        x=int(rng.integers(HALF+8,image.shape[1]-HALF-8))
        y=int(rng.integers(HALF+8,image.shape[0]-HALF-8))
        if mask[y,x]:continue
        if any(np.hypot(x-p[0],y-p[1])<2*HALF+4 for p in positions):continue
        positions.append((x,y,rng.uniform(-.45,.45),rng.uniform(-.45,.45),rng.uniform(0,np.pi)))
    Table(rows=positions,names=['x','y','phase_x','phase_y','angle']).write(OUTPUT/'injection_positions.ecsv',overwrite=True)
    rows=[]
    # Reuse locations at every magnitude/profile: this is a paired experiment.
    # Different positions have disjoint cutouts; each injection is independent.
    for pi,(x,y,px,py,angle) in enumerate(positions):
        cut=np.s_[y-HALF:y+HALF+1,x-HALF:x+HALF+1]
        before_peak_distance=float(peak_tree.query((x+px,y+py))[0])
        models={p:source_profile(p,px,py,angle) for p in PROFILES}
        for profile in PROFILES:
            for mag in MAGNITUDES:
                flux=10**((header['MAGZPT']-mag)/2.5)*header['EXPTIME']/gain
                truth=models[profile]*flux
                injected=rng.poisson(truth*gain)/gain
                data=residual[cut].astype(float)+injected
                # Apply the existing sharp-feature test, keeping old mask pixels.
                sharp,_=detector.sharp_feature_mask(data,rms[cut],bad[cut],settings)
                local_bad=bad[cut]|sharp
                local_mask=mask[cut]|ndi.binary_dilation(sharp,iterations=4)
                smooth=ndi.gaussian_filter(np.where(local_bad,0,data),settings.smooth_sigma)
                snr=(smooth-smooth_level[cut])/smooth_noise[cut]
                segments,metadata=detector.separate_sources(smooth,snr,local_mask,settings)
                table,_=detector.measure_sources(image[cut]+injected,data,snr,segments,metadata,
                           artefact[cut]|sharp,boundary[cut],settings,noise=rms[cut])
                detected=False;strict=False;errorpass=False;reason='not_detected'
                measured=np.nan;fraction=np.nan;magerror=np.nan;matched_id=-1
                if len(table):
                    distances=np.hypot(table['x']-(HALF+px),table['y']-(HALF+py))
                    i=int(np.argmin(distances));r=table[i]
                    distinct=(before_peak_distance>3 and distances[i]<=3
                        and np.hypot(r['peak_x']-(HALF+px),r['peak_y']-(HALF+py))<=3)
                    if not distinct and before_peak_distance<=3: reason='preexisting_peak_confusion'
                    if distinct:
                        detected=True;matched_id=int(r['id']);reason='strict_aperture_rejected'
                        bounds,_,_,ap,edge=neighbour.aperture_grid(r,data.shape)
                        measured=float(data[bounds][ap].sum())
                        fraction=float(truth[bounds][ap].sum()/flux)
                        rawnoise=float(np.sqrt(np.sum(rms[cut][bounds][ap]**2+np.maximum(data[bounds][ap],0)/gain)))
                        with contextlib.redirect_stdout(io.StringIO()):
                            table=reject_neighbour_cores(table,data,rms[cut],segments)
                            table,_=reject_shared_apertures(table,data,rms[cut],local_bad)
                        peaks=scan_internal_peaks(table[[i]],data,rms[cut],segments)
                        internal=bool(len(peaks) and significant_structure(peaks)[0])
                        strict=bool(table['usable'][i] and not internal and not edge
                            and not np.any(local_mask[bounds][ap]) and measured/rawnoise>=5)
                        if strict:
                            global_row=table[i:i+1].copy()
                            global_row['x']+=x-HALF;global_row['y']+=y-HALF
                            dx,dy=photo.aperture_offsets(global_row[0])
                            # Source placement is >2a+20 from sampled blank centres
                            # in the existing routine. Original real-source masks
                            # remain in force; synthetic source noise is added once.
                            scale,count,median,radius,status=photo.estimate_blank_scale(
                                residual,rms,forbidden,global_row[0],dx,dy,rng)
                            variance=np.sum(rms[cut][bounds][ap].astype(float)**2)*scale**2+max(measured,0)/gain
                            magerror=2.5/np.log(10)*np.sqrt(variance)/measured if measured>0 else np.inf
                            errorpass=bool(status=='local_blank' and magerror<=.10)
                            reason='error_rule_pass' if errorpass else 'photometric_error_or_blank_failure'
                rows.append((pi,profile,mag,x+px,y+py,before_peak_distance,detected,strict,errorpass,
                             measured,flux,fraction,magerror,matched_id,reason))
        print(f'Injection placements {pi+1}/{N_POSITIONS}; trials {len(rows)}',flush=True)
    trials=Table(rows=rows,names=['position_id','profile','input_total_mag','x','y','nearest_old_peak',
        'detected','strict_aperture_pass','error_rule_pass','measured_flux_adu','input_total_flux_adu',
        'true_aperture_fraction','magnitude_error_random','local_source_id','reason'])
    trials.meta['seed']=SEED
    trials.meta['scope']='Conditional fixed-background cutout experiment. No neighbour-model rescue, manual adjustments or PSF star classification. Do not use as catalogue completeness weights.'
    trials.write(OUTPUT/'injection_trials.ecsv',overwrite=True)
    make_summary(trials)
    text=['720 trials: 40 disjoint locations x 3 profiles x 6 total magnitudes.',
          'Locations are paired across profiles/magnitudes; 40 trials per individual point.',
          'Existing real noise and neighbours retained; source shot noise added using header gain.',
          'Point Moffat FWHM=4 pix beta=3.5; exponential half-light radii=2,4 pix; axis ratio=0.7.',
          'Masks, background and smoothed-noise maps fixed; no final star/galaxy classification.',
          'Cutouts 129x129 pix; injected centres >72 pix from image border, within unmasked area.',
          'Distinct match: no existing peak within3 pix, new centroid and peak each within3 pix.',
          'Strict selection includes native aperture rule, neighbour cores, shared light and internal peaks.',
          'Does not include neighbour-model rescue or the earlier manual aperture modifications.',
          'Input is TOTAL magnitude; observed counts use APERTURE magnitude. Their difference is recorded.',
          'These conditional fractions are not full survey completeness and MUST NOT correct observed counts.',
          'Uncertainties are 68% Wilson binomial intervals; morphology mix and field variance not included.']
    (OUTPUT/'METHOD.txt').write_text('\n'.join(text)+'\n')

if __name__=='__main__':run()
