"""Test error-based recovery over a fixed 19--23 mag interval.

Reuse the existing detection, neighbour fitting and photometry. Never adjust
the thresholds based on the resulting slope. Existing results are read only.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from dataclasses import replace
from collections import Counter
import numpy as np
from astropy.io import fits
from astropy.table import Table
import aperture_neighbour_model as neighbour
import detection_aperture_review as detector
import photometry_analysis as photo
import run_field_quality_review as field
from run_aperture_review import MODEL_SETTINGS
from run_experiment import review_saved_scores

INPUT = BASE/'results/field_quality_review'
OUTPUT = HERE/'results/recovery'
PRODUCTS = BASE/'results/aperture_review/detection_products.fits'
IMAGE = BASE.parent/'Astro/Fits_Data/mosaic.fits'
SETTINGS = replace(MODEL_SETTINGS, max_nfev=600, profile_betas=(2.,3.,6.))
MAX_MAG_ERROR = .10
ERROR_COLUMNS = ['flux_error_adu','background_flux_error_adu','source_flux_error_adu',
    'background_correlation_factor','blank_median_normalised','blank_search_radius',
    'magnitude_error_random','magnitude_error_total','magnitude_error_lower',
    'magnitude_error_upper','photometric_snr','blank_aperture_count',
    'uncertainty_status','magnitude_error_zeropoint']


def model_pass(row):
    """Retain the bright-review contamination tolerances, without S/N>=50."""
    names = ['model_flux_fraction','model_area_fraction','model_score_disagreement',
             'model_bound_neighbour_fraction','model_sky_flux_fraction','model_aperture_snr']
    if not row['model_valid'] or not row['usable']:
        return False
    if not all(np.isfinite(row[n]) for n in names):
        return False
    return (row['model_flux_fraction'] <= .03 and row['model_area_fraction'] < .15
            and row['model_score_disagreement'] <= .01
            and row['model_bound_neighbour_fraction'] <= .02
            and row['model_sky_flux_fraction'] <= .05)


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = Table.read(INPUT/'sources.ecsv')
    before = table.copy()
    baseline = Table.read(INPUT/'classified_sources.ecsv')
    initial = np.array(table['usable'], bool)
    image, header = fits.getdata(IMAGE, header=True)
    with fits.open(PRODUCTS) as f:
        bg = f['BACKGROUND'].data.copy(); rms = f['RMS'].data.copy()
        labels = f['SOURCES'].data.copy(); mask = f['DETMASK'].data.astype(bool)
        bad = f['BADMASK'].data.astype(bool); saturation = f[0].header['SATLEVEL']
    residual = image-bg
    mag = np.array(table['field_review_magnitude'])
    target = ~initial & (mag>=19) & (mag<23)
    manual = (table['reviewed_blend_rejected'] | table['reviewed_internal_peak_rejected']
              | (np.asarray(table['local_review_status'])=='outer_wings_unresolved'))
    eligible = np.array(target & table['field_review_aperture_clean']
        & ~table['near_boundary'] & ~table['aperture_capped'] & ~manual
        & ~table['field_internal_peak_candidate'] & ~table['field_shape_review']
        & ~table['field_wing_review'] & ~table['neighbour_core_rejected']
        & ~table['neighbour_peak_inside'], bool)
    pairs = Table.read(BASE/'results/aperture_review/shared_aperture_pairs.csv', format='ascii.csv')
    idmap = {int(r['id']):i for i,r in enumerate(table)}
    partners = {i:[] for i in range(len(table))}
    for p in pairs[np.asarray(pairs['pair_rejected']).astype(str)=='True']:
        i,j = idmap[int(p['source_id_1'])],idmap[int(p['source_id_2'])]
        partners[i].append(j); partners[j].append(i)
    for i in np.flatnonzero(eligible):
        if (table['shared_aperture_rejected'][i] and table['shared_aperture_flux_fraction'][i]>.02
            or any(initial[j] for j in partners[i])):
            eligible[i] = False
    preliminary = eligible.copy()
    print(f'Excluded in 19--23: {sum(target)}; geometry/shared-light pass: {sum(eligible)}', flush=True)

    # Measure errors before selecting objects, including correlated background
    # noise from the SAME existing blank-aperture routine. No RMS-only fallback.
    temporary = table.copy()
    temporary['count_sample'] = eligible
    for i in np.flatnonzero(eligible):
        bounds,_,_,ap,edge = neighbour.aperture_grid(table[i], residual.shape)
        assert not edge and not np.any((mask|bad)[bounds][ap])
        temporary['aperture_flux_adu'][i] = float(residual[bounds][ap].sum())
    errors,_ = photo.add_photometric_uncertainties(temporary, residual, rms, labels, mask|bad, header)
    errors[eligible].write(OUTPUT/'candidate_error_audit.ecsv', overwrite=True)
    eligible &= (np.asarray(errors['uncertainty_status'])=='local_blank')
    eligible &= np.isfinite(errors['magnitude_error_random']) & (errors['magnitude_error_random']<=MAX_MAG_ERROR)
    print(f'Random magnitude error <= {MAX_MAG_ERROR}: {sum(eligible)}', flush=True)

    work = before.copy(); work['usable'] = np.array(work['strict_usable'])
    work['flags'] = work['original_flags'].copy()
    for i in np.flatnonzero(eligible):
        work['usable'][i]=False; work['near_artefact'][i]=False; work['blended'][i]=True
    scored,_ = neighbour.score_catalogue(work,residual,rms,bad,saturation,SETTINGS,float(header['GAIN']),
                                         only_parents=np.unique(work['parent_id'][eligible]))
    scored = review_saved_scores(scored,residual,SETTINGS)
    scored[eligible].write(OUTPUT/'refitted_candidates.ecsv',overwrite=True)
    proposed = {i for i in np.flatnonzero(eligible) if model_pass(scored[i])}
    accepted = {i for i in proposed if not any(j in proposed for j in partners[i])}
    table['error_review_recovered'] = np.zeros(len(table),bool)
    table['error_review_status'] = np.full(len(table),'not_targeted',dtype='U48')
    table['error_review_status'][target] = 'geometry_or_structure_veto'
    table['error_review_status'][preliminary] = 'photometric_error_or_blank_failure'
    for i in np.flatnonzero(eligible):
        table['error_review_status'][i] = ('shared_partner_proposed' if i in proposed and i not in accepted
                                         else str(scored['model_status'][i]))
        if scored['model_valid'][i] and not model_pass(scored[i]):
            table['error_review_status'][i] = 'strict_contamination_or_stability_limit'
    table['flags'] = np.asarray(table['flags'],dtype='U900')
    for name in ['model_flux_fraction','model_area_fraction','model_score_disagreement',
                 'model_sky_flux_fraction','model_bound_neighbour_fraction']:
        table['error_review_'+name] = np.full(len(table),np.nan)
        table['error_review_'+name][eligible] = scored[name][eligible]
    for i in accepted:
        table['usable'][i]=True; table['error_review_recovered'][i]=True
        table['error_review_status'][i]='accepted'
        if not np.isfinite(table['aperture_flux_adu'][i]):
            table['aperture_flux_adu'][i]=temporary['aperture_flux_adu'][i]
        table['aperture_snr'][i]=scored['aperture_snr'][i]
        table['aperture_flux_error_adu'][i]=scored['aperture_flux_error_adu'][i]
        table['flags'][i] += '|error_based_model_recovery'
    table.meta['error_review_baseline_sha256'] = photo.file_hash(INPUT/'sources.ecsv')
    table.meta['error_review_rule'] = '19<=aperture r<23; random magnitude error<=0.10 using local blanks; model neighbour flux<=3%, shared<=2%, start disagreement<=1%; original geometry, core and mask vetoes retained.'
    table.write(OUTPUT/'sources.ecsv',overwrite=True)
    table.write(OUTPUT/'sources.csv',format='ascii.csv',overwrite=True)
    table[table['usable']].write(OUTPUT/'usable_sources.csv',format='ascii.csv',overwrite=True)
    table[target].write(OUTPUT/'recovery_decisions.ecsv',overwrite=True)
    print('Recovered IDs:',list(table['id'][table['error_review_recovered']]),flush=True)

    # Reuse measured errors for the unchanged sources; use the pre-selection
    # errors for new recoveries. This avoids changing random blank placements
    # merely because extra catalogue rows now enter the sample.
    cache = baseline.copy()
    for name in ERROR_COLUMNS:
        for i in accepted: cache[name][i]=errors[name][i]
    original_errors = photo.add_photometric_uncertainties
    def cached_errors(selected, *args):
        assert np.array_equal(selected['id'],cache['id'])
        for name in ERROR_COLUMNS: selected[name]=cache[name].copy()
        selected.meta['error_note']='Unchanged-source errors reused from field review; new-source errors measured with the same blank-aperture method before selection.'
        return selected, {}
    photo.add_photometric_uncertainties = cached_errors
    photo.DETECTION_DIR=OUTPUT;photo.PRODUCTS_PATH=PRODUCTS;photo.OUTPUT_DIR=OUTPUT
    references=list(np.asarray(baseline['segment_id'][baseline['psf_reference']],int))
    try:
        classified,fit = photo.run_analysis(measurement_mag_limit=22.,classification_mag_limit=22.,
                                            fit_range=(19.,23.),reference_labels=references)
    finally: photo.add_photometric_uncertainties=original_errors
    field.OUTPUT=OUTPUT
    field.contact_sheet(table,sorted(accepted),residual/rms,'recovered_sources')
    with fits.open(PRODUCTS) as f:
        detector.save_diagnostics(image,bg,rms,f['ARTEFACT'].data.astype(bool),
            f['BOUNDARY'].data.astype(bool),f['SATURATED'].data.astype(bool),table,OUTPUT)
    lines=[f'Baseline usable: {sum(initial)}',f'Targets 19--23: {sum(target)}',
           f'Geometry/shared pass: {sum(preliminary)}',f'Error-qualified refits: {sum(eligible)}',
           f'Recovered: {len(accepted)}',f'Usable: {sum(table["usable"])}',
           f'Remaining: {sum(classified["remaining_sample"])}',
           f'Slope 19--23: {fit["slope"]} +/- {fit["slope_error_propagated"]}',
           str(Counter(table['error_review_status'][target])),
           'No completeness correction applied. Existing usable sources and apertures unchanged.']
    (OUTPUT/'SUMMARY.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__': run()
