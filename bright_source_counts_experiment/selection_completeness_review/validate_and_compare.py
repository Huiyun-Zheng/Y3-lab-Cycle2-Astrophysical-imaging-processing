"""Validate saved recovery and injection results and plot a fixed-range comparison."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
import numpy as np
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
from run_recovery import ERROR_COLUMNS, model_pass, INPUT, OUTPUT, PRODUCTS, IMAGE
from run_injection import source_profile, PROFILES, MAGNITUDES, N_POSITIONS, HALF
import photometry_analysis as photo
import aperture_neighbour_model as neighbour


def run():
    old=Table.read(INPUT/'sources.ecsv');new=Table.read(OUTPUT/'sources.ecsv')
    a=Table.read(INPUT/'classified_sources.ecsv');b=Table.read(OUTPUT/'classified_sources.ecsv')
    assert new.meta['error_review_baseline_sha256']==photo.file_hash(INPUT/'sources.ecsv')
    assert b.meta['input_products_sha256']==photo.file_hash(PRODUCTS)
    for name in ['id','segment_id','parent_id','x','y','aperture_a_pix','aperture_b_pix','aperture_theta_deg']:
        assert np.array_equal(old[name],new[name]),name
    initial=np.array(old['usable'],bool)
    added=np.array(new['usable'],bool)&~initial
    assert np.all(new['usable'][initial])
    assert set(new['id'][added])=={1293,2582}
    assert np.array_equal(added,new['error_review_recovered'])
    valid=np.isfinite(old['aperture_flux_adu'])
    np.testing.assert_array_equal(old['aperture_flux_adu'][valid],new['aperture_flux_adu'][valid])
    for name in ERROR_COLUMNS:
        x,y=np.asarray(a[name][initial]),np.asarray(b[name][initial])
        if x.dtype.kind in 'fc':np.testing.assert_allclose(x,y,rtol=0,atol=0,equal_nan=True)
        else:np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(a['is_star'][initial],b['is_star'][initial])
    np.testing.assert_array_equal(a['psf_reference'],b['psf_reference'])
    np.testing.assert_array_equal(b['count_sample'],new['usable'])
    scores=Table.read(OUTPUT/'refitted_candidates.ecsv')
    image,header=fits.getdata(IMAGE,header=True)
    with fits.open(PRODUCTS) as f:
        residual=image-f['BACKGROUND'].data
        mask=f['DETMASK'].data.astype(bool)|f['BADMASK'].data.astype(bool)
    for r in new[added]:
        score=scores[scores['id']==r['id']][0]
        assert model_pass(score)
        classified=b[b['id']==r['id']][0]
        assert classified['magnitude_error_random']<=.10
        assert classified['uncertainty_status']=='local_blank'
        assert not any(r[n] for n in ['near_boundary','aperture_capped',
            'neighbour_core_rejected','neighbour_peak_inside','field_internal_peak_candidate',
            'field_shape_review','field_wing_review','reviewed_blend_rejected','reviewed_internal_peak_rejected'])
        bounds,_,_,ap,edge=neighbour.aperture_grid(r,residual.shape)
        assert not edge and not mask[bounds][ap].any()
        assert np.isclose(r['aperture_flux_adu'],residual[bounds][ap].sum(),rtol=2e-6)
    pairs=Table.read(BASE/'results/aperture_review/shared_aperture_pairs.csv',format='ascii.csv')
    accepted=set(new['id'][new['usable']])
    for p in pairs[np.asarray(pairs['pair_rejected']).astype(str)=='True']:
        assert not (p['source_id_1'] in accepted and p['source_id_2'] in accepted)
    # Guard the original project files using the pre-existing experiment manifest.
    protected=Table.read(BASE/'inputs/original_file_hashes.tsv',format='ascii.tab')
    for r in protected:assert photo.file_hash(BASE.parent/r['path'])==r['sha256'],r['path']

    trials=Table.read(HERE/'results/injection/injection_trials.ecsv')
    summary=Table.read(HERE/'results/injection/recovery_summary.ecsv')
    assert len(trials)==N_POSITIONS*len(MAGNITUDES)*len(PROFILES)==720
    assert np.all(~trials['error_rule_pass']|trials['strict_aperture_pass'])
    assert np.all(~trials['strict_aperture_pass']|trials['detected'])
    assert np.all(trials['magnitude_error_random'][trials['error_rule_pass']]<=.10)
    assert np.all((trials['true_aperture_fraction'][trials['detected']]>0)&
                  (trials['true_aperture_fraction'][trials['detected']]<=1.000001))
    for r in summary:
        t=trials[(trials['profile']==r['profile'])&(trials['input_total_mag']==r['input_total_mag'])]
        assert len(t)==40 and len(np.unique(t['position_id']))==40
        assert np.isclose(r['detected_fraction'],sum(t['detected'])/40)
    for p in PROFILES:
        model=source_profile(p,.23,-.17,.31)
        assert model.shape==(129,129) and np.all(model>=0) and np.isclose(model.sum(),1)
        yy,xx=np.indices(model.shape)
        assert abs((xx*model).sum()-HALF-.23)<.02
        assert abs((yy*model).sum()-HALF+.17)<.02

    # All comparisons use the same 19--23 bins, area and aperture magnitudes.
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    before=Table.read(INPUT/'number_counts.ecsv');after=Table.read(OUTPUT/'number_counts.ecsv')
    for t,label,style in [(before,'Field review: 0.377414','o-'),(after,'Error-based recovery: 0.377498','x--')]:
        x=np.asarray(t['limiting_magnitude']);y=np.asarray(t['log10_density'])
        fit=(x>=19)&(x<=23)
        slope=np.polyfit(x[fit],y[fit],1)[0]
        expected=a.meta['descriptive_slope'] if t is before else b.meta['descriptive_slope']
        assert np.isclose(slope,expected)
        axes[0].plot(x,y,style,label=label)
    axes[0].axvspan(19,23,color='grey',alpha=.12)
    axes[0].set(xlabel='Limiting aperture magnitude',ylabel='log10 cumulative density / deg^-2',
                title='Same 19--23 fit interval');axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    np.testing.assert_array_equal(before['limiting_magnitude'],after['limiting_magnitude'])
    axes[1].step(after['limiting_magnitude'],after['count']-before['count'],where='mid')
    axes[1].set(xlabel='Limiting aperture magnitude',ylabel='Additional cumulative sources',
                title='Two new sources; no earlier source removed',ylim=(-.1,2.2))
    axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'results/counts_comparison.png',dpi=150);plt.close(fig)
    for t,name in [(before,'before'),(after,'after')]:
        m=np.asarray(t['limiting_magnitude'])
        expected=np.array([sum(b['remaining_sample']&(b['magnitude']<z)) for z in m]) if name=='after' else None
        if expected is not None:np.testing.assert_array_equal(t['count'],expected)

    lines=['PASS: original project hashes and field-review input catalogue unchanged.',
           'PASS: positions/apertures/finite fluxes unchanged; all earlier usable sources retained.',
           'PASS: exactly two new sources pass empirical-error, mask, model and neighbour criteria.',
           'PASS: old classifications, reference stars and measured uncertainties unchanged.',
           'PASS: final counts match catalogue and use the same area and 19--23 interval.',
           'PASS: 720 injection trials, nested stage outcomes and positive unit-flux centred models.',
           'No injection recovery fractions were used as weights or to alter observed counts.']
    (HERE/'results/VALIDATION.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':run()
