Bright-source recovery and number-count experiment

This directory contains copies of the existing neighbour-light model, detection
snapshot, circular Moffat classifier, and photometry analysis. The original
directories are read only. Input catalogues and detection products are snapshots
in inputs/. The large original mosaic is read from ../Astro/Fits_Data/mosaic.fits.

Run all six comparisons from the project directory:
    python -B bright_source_counts_experiment/run_experiment.py

Rerun the complete Moffat + photometry analysis for balanced_recovery:
    python -B bright_source_counts_experiment/photometry_analysis.py

Validate the results and confirm that the original files are unchanged:
    python -B bright_source_counts_experiment/validate_experiment.py

Use the existing astro-image Python environment. Each command overwrites only
its outputs within this experiment directory. No Git operations are performed.

Common settings
    Main fit interval: 19--23 mag. Magnitude grid: 0.5 mag.
    Measurement and classification limits: 22 mag.
    Same eleven supplied reference-star segment IDs as the current analysis.
    Same flux calibration, survey area, apertures, segmentation and image mask.
    Aperture S/N >= 5. Saturation/artefact, boundary and capped-aperture vetoes remain.
    Original blank-aperture + source-shot-noise photometric uncertainties retained.
    Counts use the copied Poisson + Monte Carlo photometric covariance calculation.

Variants
    baseline: reproduce the current accepted detections and circular Moffat rules.
    psf_review: same detections; add reference-calibrated relative-profile quality
        and an inner/outer-flux check. High chi-square alone need not reject a
        high-S/N star. Widths with >10% relative uncertainty are not classified.
    balanced_recovery: psf_review plus modest changes to existing saved-model gates:
        mean pixel contamination <= about 15% rather than 10% (strict < in code);
        summed neighbour flux remains <10%; bound-neighbour contribution <=2%
        rather than 1%; residual-sky contribution <=5% rather than 3%.
        The 8% fractional-profile-residual criterion is unchanged.
    strict_flux_control: same as balanced_recovery, but summed neighbour flux <5%.
    empirical_psf: original detections; match reference-star concentration at
        radii 3/9 and 5/9, radial second moment and ellipticity. S/N >=50 in the
        morphology aperture. This can classify sources whose analytic fit failed.
    balanced_empirical: balanced recovery with that empirical classifier.

No new source-finding pass is performed: the saved fit diagnostics are reused.
No source is accepted solely for being bright. No magnitude cut is chosen to
make the measured slope match 0.6. All six primary slopes use 19--23 mag.
The extra fit-range table is an explicitly labelled sensitivity check.

Outputs
    results/comparison.csv              Main fixed-range comparison
    results/counts_comparison.png       All six cumulative count curves
    results/fit_range_sensitivity.csv   Exploratory interval dependence
    results/measured_sources.ecsv       Common measurements before classification
    results/morphology_controls.csv     Reference and broadened-profile controls
    results/<variant>/sources.ecsv
    results/<variant>/classified_sources.ecsv
    results/<variant>/number_counts.ecsv
    results/<variant>/cumulative_number_counts.png
    results/<variant>/moffat_width_vs_magnitude.png
    results/balanced_recovery/detection_overview.png
    results/balanced_recovery/recovered_bright_sources.png

Observed outcome
    Baseline slope: 0.40129. psf_review: 0.42199.
    balanced_recovery: 0.41778. strict_flux_control: 0.41706.
    empirical_psf: 0.42565. balanced_empirical: 0.42115.
    Brightest 100 detections retained: 57 baseline, 61 balanced recovery.
    Brightest 200 detections retained: 108 baseline, 116 balanced recovery.
    The more inclusive recovery trial improves bright retention and is slightly
    closer to 0.6, but no tested fixed-range variant reaches 0.5.

Validation and limits
    The baseline cumulative counts reproduce the saved current counts exactly.
    All selected apertures have positive flux, finite uncertainty, and valid masks.
    Original files are checked against the recorded SHA-256 values.
    Empirical morphology recognises 11/11 reference holdouts; 0/11 copies broadened
    by a Gaussian with sigma=1.5 pixels are classified as stars. These limited
    controls do NOT establish real-galaxy purity, completeness or PSF accuracy
    across the whole field. Compact galaxies and unknown sources remain ambiguous.
    Neighbour diagnostics remain model proxies, not measured contamination truth.
    Slopes are descriptive; quoted errors exclude classification, completeness,
    model-family uncertainty and cosmic variance. No completeness correction or
    total-flux correction was inferred from the desired slope.

Background reading
    PSFEx discusses signal-dependent profile-model mismatch for bright sources:
    https://psfex.readthedocs.io/en/latest/Working.html
    SDSS morphology compares point-source and extended-source photometry:
    https://www.sdss4.org/dr14/algorithms/classify/
    This experiment does not implement PSFEx or SDSS's full classifier.
