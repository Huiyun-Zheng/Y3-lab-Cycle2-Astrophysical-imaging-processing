"""Audit the entire saved field with fixed detection, apertures and masks.

Write a separate review version, leaving results/aperture_review untouched.
Internal-peak cuts are provisional quality cuts, not a physical classifier.
Bright-source recovery requires actual clean apertures and stable neighbour fits.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from dataclasses import replace
from collections import Counter
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import aperture_neighbour_model as neighbour
import detection_aperture_review as detector
import photometry_analysis as photometry
from run_aperture_review import MODEL_SETTINGS
from run_experiment import review_saved_scores
from reviewed_recovery import recovery_decision
from field_internal_peaks import scan_internal_peaks

BASE=Path(__file__).resolve().parent
INPUT=BASE/'results/aperture_review'
OUTPUT=BASE/'results/field_quality_review'
IMAGE=BASE.parent/'Astro/Fits_Data/mosaic.fits'
PRODUCTS=INPUT/'detection_products.fits'


def significant_structure(a):
    # Peaks recur at sigma .5 and .7. Separate by at least 5 pixels (> median
    # stellar FWHM of 4 pixels), and require a resolved dip and appreciable core.
    # These chosen tolerances define a review sample, not a statistical detection.
    return ((a['separation']>=5)&(a['secondary_snr']>=7.5)&(a['peak_ratio']>=.15)
            &(a['dip_snr']>=1.5)&(a['secondary_core_fraction']>=.02))


def compact_core_wings(table, residual, rms, labels, mask):
    """Flag very bright compact cores with measurable light beyond the aperture.

    This targets the same high-contrast small-circle problem seen in the earlier
    local reviews. It is a warning, not a universal total-flux aperture definition.
    Existing accepted objects are retained for review; new recoveries must wait.
    """
    warning=np.zeros(len(table),bool)
    extra=np.full(len(table),np.nan)
    candidates=((table['peak_detection_snr']>1000)&(table['aperture_shape']=='circle')
                &(table['aperture_a_pix']<12))
    for i in np.flatnonzero(candidates):
        row=table[i]
        enlarged=table[[i]].copy()
        enlarged['aperture_a_pix']*=1.4;enlarged['aperture_b_pix']*=1.4
        bounds,xx,yy,ap,edge=neighbour.aperture_grid(enlarged[0],residual.shape)
        added=ap&(np.hypot(xx-row['x'],yy-row['y'])>row['aperture_a_pix'])
        clean=(added&~mask[bounds]&((labels[bounds]==0)|(labels[bounds]==row['segment_id'])))
        if edge or not added.any() or clean.sum()<.9*added.sum():
            warning[i]=True
            continue
        base_bounds,_,_,base_ap,_=neighbour.aperture_grid(row,residual.shape)
        flux=float(residual[base_bounds][base_ap].sum())
        wing=float(residual[bounds][clean].sum())
        extra[i]=wing/flux if flux>0 else np.nan
        warning[i]=(extra[i]>.03 and wing>5*np.sqrt(np.sum(rms[bounds][clean]**2)))
    return warning,extra


def contact_sheet(table, indices, display, name, peaks=None, page_size=20):
    indices=list(indices)
    for page,start in enumerate(range(0,len(indices),page_size),1):
        fig,axes=plt.subplots(4,5,figsize=(15,12))
        for ax in axes.flat:ax.set_axis_off()
        for ax,i in zip(axes.flat,indices[start:start+page_size]):
            r=table[i];half=int(max(15,min(45,r['aperture_a_pix']+7)))
            x0=max(0,int(r['x'])-half);x1=min(display.shape[1],int(r['x'])+half+1)
            y0=max(0,int(r['y'])-half);y1=min(display.shape[0],int(r['y'])+half+1)
            ax.imshow(display[y0:y1,x0:x1],origin='lower',cmap='gray',vmin=-1,vmax=12,
                      extent=(x0-.5,x1-.5,y0-.5,y1-.5))
            near=np.hypot(table['x']-r['x'],table['y']-r['y'])<2*half
            for s in table[near]:
                ax.add_patch(Ellipse((s['x'],s['y']),2*s['aperture_a_pix'],2*s['aperture_b_pix'],
                    angle=s['aperture_theta_deg'],fill=False,edgecolor='royalblue' if s['usable'] else 'red',lw=.8))
            if peaks is not None:
                found=peaks[peaks['id']==r['id']]
                if len(found):
                    q=found[0];ax.scatter([q['primary_x'],q['secondary_x']],[q['primary_y'],q['secondary_y']],marker='+',c='cyan',s=30)
            ax.set_xlim(x0,x1);ax.set_ylim(y0,y1)
            ax.set_title(f"ID {r['id']} | {'usable' if r['usable'] else 'excluded'}",fontsize=10)
        fig.suptitle(name.replace('_',' ')+f' — page {page}',fontsize=14)
        fig.tight_layout();fig.savefig(OUTPUT/f'{name}_{page:02d}.png',dpi=110);plt.close(fig)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    table=Table.read(INPUT/'sources.ecsv');old=table.copy()
    old.write(OUTPUT/'baseline_sources.ecsv',overwrite=True)
    image,header=fits.getdata(IMAGE,header=True)
    with fits.open(PRODUCTS) as f:
        background=f['BACKGROUND'].data.copy();rms=f['RMS'].data.copy();labels=f['SOURCES'].data.copy()
        bad=f['BADMASK'].data.astype(bool);mask=f['DETMASK'].data.astype(bool);saturation=float(f[0].header['SATLEVEL'])
    residual=image-background;gain=float(header['GAIN']);initial=np.asarray(table['usable'],bool).copy()
    print('Scanning every fixed aperture for internal peaks...',flush=True)
    peaks=scan_internal_peaks(table,residual,rms,labels)
    peaks['review_exclusion']=significant_structure(peaks)
    peaks.write(OUTPUT/'internal_peak_diagnostics.ecsv',overwrite=True)
    table['field_internal_peak_candidate']=np.isin(table['id'],peaks['id'])
    table['field_internal_peak_rejected']=np.isin(table['id'],peaks['id'][peaks['review_exclusion']])
    table['field_shape_review']=(table['minor_sigma_pix']<1.)|(table['elongation']>4.)
    table['field_wing_review'],table['field_wing_extra_fraction']=compact_core_wings(table,residual,rms,labels,mask)
    table['usable']=initial&~table['field_internal_peak_rejected']
    table['flags']=np.asarray(table['flags'],dtype='U700')
    for i in np.flatnonzero(table['field_internal_peak_rejected']):
        table['flags'][i]='|'.join([p for p in str(table['flags'][i]).split('|') if p!='none']+['field_internal_multipeak'])
    # Recompute flux/error for audit only. Preserve all adopted measurements.
    flux=np.zeros(len(table));snr=np.zeros(len(table));clean=np.zeros(len(table),bool)
    for i,row in enumerate(table):
        bounds,_,_,ap,edge=neighbour.aperture_grid(row,residual.shape)
        val=residual[bounds][ap];err=rms[bounds][ap]
        flux[i]=val.sum();noise=np.sqrt(np.sum(err.astype(float)**2+np.maximum(val,0)/gain))
        snr[i]=flux[i]/noise if noise>0 else np.nan
        clean[i]=(not edge and ap.any() and not np.any(mask[bounds][ap]) and not np.any(bad[bounds][ap])
                  and np.all(np.isfinite(val)) and np.all(np.isfinite(err)))
    mag=header['MAGZPT']-2.5*np.log10(np.maximum(flux,1)*gain/header['EXPTIME'])
    bright=(~initial)&(mag<22.)&(flux>0)
    manual=table['reviewed_blend_rejected']|table['reviewed_internal_peak_rejected']
    # Do not recover the already-reviewed inadequate small aperture (ID 7).
    manual |= (np.asarray(table['local_review_status'])=='outer_wings_unresolved')
    eligible=(bright&clean&(snr>=50)&~table['near_boundary']&~table['aperture_capped']
              &~manual&~table['field_internal_peak_candidate']&~table['field_shape_review']
              &~table['field_wing_review']
              &~table['neighbour_core_rejected']&~table['neighbour_peak_inside'])
    pairs=Table.read(INPUT/'shared_aperture_pairs.csv',format='ascii.csv')
    pairbad=np.asarray(pairs['pair_rejected']).astype(str)=='True'
    idmap={int(r['id']):i for i,r in enumerate(table)};partners={i:[] for i in range(len(table))}
    for p in pairs[pairbad]:
        i,j=idmap[int(p['source_id_1'])],idmap[int(p['source_id_2'])]
        partners[i].append(j);partners[j].append(i)
    for i in np.flatnonzero(eligible):
        if (table['shared_aperture_rejected'][i] and table['shared_aperture_flux_fraction'][i]>.02) or any(table['usable'][j] for j in partners[i]):
            eligible[i]=False
    print(f'Originally excluded, aperture r<22: {bright.sum()}; clean model-recheck candidates: {eligible.sum()}',flush=True)
    work=old.copy();work['usable']=work['strict_usable'].copy();work['flags']=work['original_flags'].copy()
    for i in np.flatnonzero(eligible):
        work['usable'][i]=False;work['near_artefact'][i]=False;work['blended'][i]=True
    settings=replace(MODEL_SETTINGS,max_nfev=600,profile_betas=(2.,3.,6.))
    fit,_=neighbour.score_catalogue(work,residual,rms,bad,saturation,settings,gain,
                                    only_parents=np.unique(work['parent_id'][eligible]))
    fit=review_saved_scores(fit,residual,settings)
    fit[eligible].write(OUTPUT/'refitted_bright_scores.ecsv',overwrite=True)
    table['field_review_recovered']=np.zeros(len(table),bool)
    table['field_review_aperture_clean']=clean
    table['field_review_magnitude']=mag
    table['field_review_status']=np.full(len(table),'unchanged',dtype='U48')
    table['field_review_status'][bright]='blocked_by_geometry_or_structure'
    metrics=['model_flux_fraction','model_area_fraction','model_score_disagreement','model_sky_flux_fraction','model_bound_neighbour_fraction']
    for metric in metrics:table['field_'+metric]=np.full(len(table),np.nan)
    decisions={}
    for i in np.flatnonzero(eligible):
        decisions[i]=recovery_decision(table[i],fit[i],clean[i],any(table['usable'][j] for j in partners[i]),False)
        for metric in metrics:table['field_'+metric][i]=fit[metric][i]
    proposals={i for i,(yes,_) in decisions.items() if yes}
    for i in proposals:
        if any(j in proposals for j in partners[i]):decisions[i]=False,'shared_partner_retained'
    for i,(yes,reason) in decisions.items():
        table['field_review_status'][i]=reason
        if yes:
            table['usable'][i]=True;table['field_review_recovered'][i]=True
            table['aperture_snr'][i]=fit['aperture_snr'][i];table['aperture_flux_error_adu'][i]=fit['aperture_flux_error_adu'][i]
            # Some previously rejected sources have no saved flux; retain a signed
            # pixel sum in exactly the same aperture, never substitute model light.
            if not np.isfinite(table['aperture_flux_adu'][i]):table['aperture_flux_adu'][i]=flux[i]
            table['flags'][i]='|'.join([p for p in str(table['flags'][i]).split('|') if p!='none']+['field_model_review_recovered'])
    table.meta['field_review_note']='Uniform fixed-aperture quality review. Multi-peak flags are provisional, not confirmed blends. Only clean high-S/N excluded sources with aperture r<22 considered for recovery. No detection/mask/aperture changes.'
    table.meta['field_review_baseline_sha256']=photometry.file_hash(INPUT/'sources.ecsv')
    table.meta['field_review_products_path']=str(PRODUCTS)
    table.write(OUTPUT/'sources.ecsv',overwrite=True);table.write(OUTPUT/'sources.csv',format='ascii.csv',overwrite=True)
    table[table['usable']].write(OUTPUT/'usable_sources.csv',format='ascii.csv',overwrite=True)
    table[bright].write(OUTPUT/'excluded_bright_review.ecsv',overwrite=True)
    changed=table['usable']!=initial
    table[changed].write(OUTPUT/'changed_sources.ecsv',overwrite=True)
    print(f'New exclusions {sum(initial&~table["usable"])}; recoveries {sum(~initial&table["usable"])}',flush=True)
    display=residual/rms
    contact_sheet(table,np.flatnonzero(initial&table['field_internal_peak_candidate']),display,'usable_internal_peak_review',peaks)
    contact_sheet(table,np.flatnonzero(table['field_review_recovered']),display,'recovered_bright_review')
    contact_sheet(table,np.flatnonzero(initial&table['field_shape_review']),display,'elongated_or_narrow_review')
    with fits.open(PRODUCTS) as f:
        detector.save_diagnostics(image,background,rms,f['ARTEFACT'].data.astype(bool),f['BOUNDARY'].data.astype(bool),f['SATURATED'].data.astype(bool),table,OUTPUT)
    # Reuse the same checked reference positions and fixed fitting interval.
    mapping=Table.read(INPUT/'reference_matches.csv',format='ascii.csv')
    keep=np.asarray(mapping['retained']).astype(str)=='True'
    reference_ids=np.asarray(mapping['new_segment_id'][keep],int)
    references=[int(i) for i in reference_ids if np.any((table['segment_id']==i)&table['usable'])]
    if len(references)!=len(reference_ids):raise ValueError('Field review changed PSF references; inspect before analysing counts.')
    photometry.DETECTION_DIR=OUTPUT;photometry.PRODUCTS_PATH=PRODUCTS;photometry.OUTPUT_DIR=OUTPUT
    classified,fit_result=photometry.run_analysis(measurement_mag_limit=22.,classification_mag_limit=22.,fit_range=(19.,23.),reference_labels=references)
    lines=[f'Baseline: {INPUT}',f'Products read unchanged from: {PRODUCTS}',f'Total candidates: {len(table)}',
           f'Internal-peak candidates among old usable: {sum(initial&table["field_internal_peak_candidate"])}',
           f'New internal-peak exclusions: {sum(initial&~table["usable"])}',f'Excluded bright sources evaluated: {bright.sum()}',
           f'Bright sources refitted: {eligible.sum()}',f'Bright sources recovered: {sum(~initial&table["usable"])}',
           f'Usable: {sum(initial)} -> {sum(table["usable"])}',f'Remaining sample: {sum(classified["remaining_sample"])}',
           f'Slope: {fit_result["slope"]} +/- {fit_result["slope_error_propagated"]}',
           'No old usable source was removed for faintness. Shape-only flags are not exclusions.']
    (OUTPUT/'FIELD_REVIEW_SUMMARY.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':run()
