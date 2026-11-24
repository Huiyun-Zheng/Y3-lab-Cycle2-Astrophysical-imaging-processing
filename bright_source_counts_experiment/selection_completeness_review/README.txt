Selection and injection pilot, 22 November 2026

This directory contains new scripts and results only. Existing detection,
field-review and original project folders are read-only inputs. The scripts
import the existing pipeline from the parent directory; keep that directory
available. No new JSON files or duplicate image FITS files are created.

Run with the astro-image Python environment, in this order:
  run_recovery.py
  run_injection.py
  validate_and_compare.py
Each script overwrites only its own results under this directory.

Recovery experiment
Input: ../results/field_quality_review/sources.ecsv
Products: ../results/aperture_review/detection_products.fits (unchanged)
Output: results/recovery

Retain every existing usable source and aperture. Reconsider excluded sources
with measured aperture magnitudes 19<=r<23. Keep the prior hard mask, core,
internal-peak and shape vetoes and the shared-light partner restrictions.
Replace the additional bright-source S/N>=50 criterion with random magnitude
error<=0.10, measured using the EXISTING local blank-aperture noise estimator
plus source shot noise. Require successful local blank estimation; no RMS-only
fallback. This is a condition on new recoveries, not a new cut on old sources.

286 of 430 geometry-qualified candidates pass the error criterion. All are
refitted with the same three-start Moffat neighbour model and 600 evaluations.
Neighbour flux<=3%, between-start score disagreement<=1%, model area<15%,
unreliable-neighbour contribution<=2%, local sky contribution<=5%; retain
the existing residual/model S/N conditions. Shared-light exceptions still
require own shared fraction<=2% and all rejected partners excluded.

Only two pass: IDs 1293 (r=22.4465) and 2582 (r=22.8785).
Of the other fits, 133 fail the unreliable-neighbour contribution test and 94
fail the local-sky contribution test. These are concerns about photometric
contamination/model degeneracy, not proof the sources do not exist.
Other failures and every measured score are saved in the audit tables.

Usable 5676->5678; remaining count sample 5607->5609; likely stars remain69.
Fixed 19--23 slope: 0.377414 -> 0.377498 +/-0.012456.
This is a negligible increase. The earlier original-aperture review had slope
0.390711; it is not replaced by this experiment. No thresholds were retuned
after inspecting the slope, and no completeness weights were applied.

Original-source fluxes, errors and classifications are unchanged. New-source
errors are computed BEFORE selection and reused in photometry, avoiding a
new random blank-aperture draw that could move a source across the criterion.
Classification still ends at22 mag; these two faint additions are unclassified
objects retained under the existing policy, not confirmed galaxies.

Injection experiment
Output: results/injection
720 trials =40 disjoint random image locations x3 profiles x6 total magnitudes.
Each source is injected individually into a129x129 real-image cutout. Real
background noise/neighbours remain; add source photon noise using header GAIN.
Models: Moffat point source (FWHM4 pixels, beta3.5); PSF-convolved exponential
disks with half-light semi-major axes2 and4 pixels, axis ratio0.7.
The same40 locations are used across magnitudes19,20,21,22,23,24 and profiles.

Use the existing peak detection, segmentation, aperture measurement, sharp
feature, core/shared-light and internal-peak routines. Record distinct-source
detection, conservative strict-aperture acceptance, and acceptance after the
additional0.10 mag random-error condition. The last stage illustrates how
restrictive this recovery condition is; it is NOT applied to all real sources.
Each plotted interval is a one-standard-deviation Wilson binomial interval.

At total magnitude23, distinct detection recovers35/40 point sources,
34/40 small disks and21/40 large disks. The strict aperture stage retains
31/40,29/40 and17/40 respectively. The error criterion further reduces these
to18/40,8/40 and0/40. Thus source size/surface brightness and quality selection
matter, even with the detection threshold held fixed.

Among detected disks at total magnitude23, the median fractions of injected
light inside the measured apertures are0.844 (small) and0.724 (large). This
illustrates missing outer light; it is not an aperture correction for real data.

Limitations
This is a CONDITIONAL PILOT, not a full final-catalogue completeness curve.
It holds background and masks fixed, approximates the initial source mask
when rebuilding the smoothed-noise map, and tests only40 interior positions.
It omits neighbour-model rescue, historical manual aperture changes and final
star/galaxy classification. True galaxy size/profile distributions are unknown.
Injected total magnitudes and observed aperture magnitudes are not equivalent.
The per-stage fractions cannot be divided into cumulative real counts. A valid
correction would need the full selection pipeline, a justified morphology mix,
and input-to-measured magnitude migration, with their uncertainties.

Useful outputs
results/counts_comparison.png: unchanged fit interval, observed before/after.
results/recovery/cumulative_number_counts.png: new observed counts with errors.
results/recovery/recovered_sources_01.png: the two added-source cutouts.
results/injection/injection_recovery.png: recovery versus magnitude and size.
results/injection/injection_trials.ecsv: every individual experiment.
results/VALIDATION.txt: completed checks.
