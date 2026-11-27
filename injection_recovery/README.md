# Simple full-field injection recovery

This experiment measures how often the **fixed current pipeline** recovers
artificial galaxies added to the original science image. It also checks whether
their measured aperture magnitudes are biased. It does not tune the detector,
force a particular counts slope, or automatically correct the observed counts.

## Files to read, in order

1. `run_injection_recovery.py`: configuration, model rendering, injection,
   positional matching, summary statistics, plots, and the main loop.
2. `detector_adapter.py`: calls the existing detection and photometry code in
   an isolated temporary directory. No detection algorithm is copied here.
3. `moffat_parameters.ecsv`: the same saved single-star PSF parameters used in
   the previous completeness experiment. This is an input, not a newly fitted PSF.

All new code and outputs stay inside this folder, except for automatically
deleted operating-system temporary files. Existing source files and results
folders are read only. The local `.gitignore` excludes new generated results.

## Running

From the project root, using the existing astronomy environment:

```sh
MPLCONFIGDIR=/private/tmp/partner-mpl /opt/anaconda3/envs/jwst/bin/python -B injection_recovery_simple/run_injection_recovery.py
```

The default run uses 3,900 injections in 98 batches. On this machine a full-field
detection/review takes roughly nine seconds; the complete experiment takes
approximately 15–20 minutes. It reports batch progress in the terminal.
Rerunning overwrites only this experiment's six result files.

For an end-to-end check:

```sh
MPLCONFIGDIR=/private/tmp/partner-mpl /opt/anaconda3/envs/jwst/bin/python -B injection_recovery_simple/run_injection_recovery.py --quick
```

The quick check uses 24 injections in two magnitude intervals and writes to
`results/quick`, keeping the main experiment separate. It is a software check,
not a statistically useful completeness measurement.

`--per-bin 100` increases the injections per profile/size/magnitude cell.
`--batch-size 20` reduces artificial crowding further at the cost of more runs.
Changing batch size also changes the random-number sequence and hence the
positions and noise realizations. The seed and all settings are recorded.

## The experiment, step by step

### 1. Check the unmodified baseline

Run the actual full science frame through:

- `source_detection_neighbour_refined/source_detection_improved.py`
- `source_detection_neighbour_refined/source_detection_neighbour_refined.py`

All background estimation, smoothing, thresholds, deblending, apertures,
quality rules, the right strip, and the fixed defect mask are unchanged.
Working on the full frame avoids translating hard-coded mask coordinates or
changing the background grid by cropping. Only diagnostic plotting is disabled.

Positions, segment labels, usable flags, and fluxes must reproduce the saved
simple-photometry catalogue. The code also checks that the saved reference stars
reproduce the original classifications. If the baseline is stale, the program
stops instead of silently combining inconsistent versions.

### 2. Make artificial galaxies

The intrinsic intensity follows a Sersic profile:

```text
I(R) = I_e * exp[-b_n * ((R / r_e)^(1 / n) - 1)]
R^2 = u^2 + (v / q)^2
```

- Exponential: `n = 1`.
- de Vaucouleurs: `n = 4`, with a concentrated centre and extended wings.
- Nominal intrinsic half-light semi-major radius: `r_e = 3, 5, 8` pixels.
- Axis ratio: `q = 0.7`; position angle: 30 degrees.
- `b_n` is calculated so the infinite intrinsic model encloses half its light
  within the ellipse with semi-major radius `r_e`.

`r_e` is not the PSF FWHM, outer boundary, or detection aperture radius.
The source is truncated at its intrinsic 99%-light ellipse and normalized again.
Consequently the half-light radius of the finite normalized template is slightly
smaller than the nominal infinite-profile `r_e`. This small, explicit truncation
keeps the model practical and prevents unbounded wings from reaching other
injections. Its shape is not an exact infinite Sersic profile.

Each model is convolved with the saved circular Moffat PSF:

```text
P(r) is proportional to [1 + (r / alpha)^2]^(-beta)
alpha = 2.853263470415898 pixels
beta = 2.009238837988954
```

Convolution is performed on a grid with five samples per native pixel in each
direction, then summed into detector pixels. Four subpixel phases are used so
all artificial centres do not sit at exactly the same pixel phase. The finite
PSF has a half-width of 50 pixels. The final nonnegative template sums to one.
These 24 templates are generated once and reused; they are not saved to disk.

Compared with directly evaluating a sharply peaked profile at native pixel
centres, this makes the injected flux and the central sampling better controlled.
The calculation of `b_n` and the FFT convolution use standard SciPy functions;
there is no new profile-fitting algorithm in this experiment.

### 3. Set brightness and positions

For every profile/radius combination, use thirteen half-magnitude intervals
from 18 to 24.5. Draw 50 magnitudes uniformly within each interval:

```text
2 profiles * 3 sizes * 13 magnitude bins * 50 = 3900 galaxies
```

Convert input magnitude to total template flux using the same convention as the
current photometry:

```text
F_ADU = 10^[0.4 * (MAGZPT - m)] * EXPTIME / GAIN
```

Because the rendered template has unit sum, its expected total flux really is
`F_ADU` before any image-edge/coverage clipping. Input magnitude refers to this
finite-template total, rather than an analytical infinite profile with a
different discretized sum.

Shuffle the complete list of injections, then place at most 40 per image.
Centres sample the baseline unmasked field, including areas near real sources.
Only artificial sources avoid one another: their stamp circumscribed circles
must not overlap. Real-neighbour rejection is left to the existing detector.
This samples the field rather than only an unusually clean small cutout.

The mask defines where an injection **centre** may lie. A source's outer wings
or recovered aperture can still encounter a mask or boundary, and any resulting
rejection is retained in the statistics. Original nonfinite/zero coverage pixels
are never filled in by the injection. `image_fraction` records the expected
fraction remaining on valid image coverage.

### 4. Add source noise and rerun detection

Each batch starts from a fresh copy of the original image. At every pixel:

```text
expected source electrons = model_ADU * GAIN
added source ADU = Poisson(expected source electrons) / GAIN
new image = original image + added source ADU
```

The original image already contains background noise, so it is not simulated a
second time. Source Poisson noise assumes the header gain is the appropriate
effective gain for this image, consistently with the current photometry.
No extra sky background or PSF background constant is added.

Run the unchanged full-field detector and review again. Their intermediate
products overwrite a single temporary workspace; no per-batch frames, FITS
images, catalogues, or plots are kept in the results folder.

### 5. Match without knowing the answer from the flux

Search all detected candidates, including excluded ones, within three pixels of
each injected centre. Require exactly one candidate and one claimant. Ambiguous
matches are not assigned arbitrarily to the nearest object.

An existing baseline detection within three pixels of either the injection or
its matched output centre makes the attribution uncertain. These cases are
marked `preexisting_match` and counted as unrecovered, even if bright. This is a
conservative definition of a **new attributable detection**, not proof that no
injected light was detected. They remain in the denominator and their number
is reported separately.

This replaces the older, much wider `match_radius + r90` neighbour veto. A real
source 10 or 20 pixels away does not automatically invalidate a recovery.
Distance to the nearest baseline source is saved; the 15-pixel near-neighbour
split is only a diagnostic, never another veto.

No requirement such as `abs(output_mag - input_mag) < 0.5` is imposed: that would
hide precisely the biased photometry we want to measure.

### 6. Apply usable and counts selection separately

Four fractions are reported, always with the same all-injection denominator:

| Column | Numerator |
|---|---|
| `unique_position_match` | One-to-one positional matches, even if pre-existing or unusable |
| `detected` | Unique matches that are attributable to a new detection |
| `usable_recovered` | New matches passing the current usable flag and positive-flux photometry |
| `counts_recovered` | Usable recoveries that are not classified as star candidates |

Matched sources go through the existing simple magnitude calibration, aperture
uncertainty calculation, and circular-Moffat fitting functions. Fitting and
classification magnitude limits come from the saved analysis catalogue.

The 21 original PSF reference stars are held fixed, and the original classifier
is called with those explicit references. Artificial galaxies cannot become
reference stars. This measures the selection efficiency **conditional on the
existing reference calibration**; it does not refit all real reference stars
or reselect a new reference population on every injected frame. Unknown objects
remain in the counts sample, just as in the existing analysis.

### 7. Check photometric bias

For each uniquely matched usable object, use its actual recovered aperture to
sum the noiseless injected model. This gives its true injected aperture flux.

Two different comparisons are saved:

```text
delta_mag_total = measured aperture magnitude - injected total magnitude
delta_mag_aperture = -2.5 * log10(measured flux / true injected aperture flux)
```

The first includes aperture losses. A positive value means the measured aperture
is fainter than the injected total. The second removes the expected aperture
loss and can reveal background errors, neighbour light, and measurement bias.
It is not forced to agree with zero, and is conditioned on successful usable
recovery, so faint-end selection bias can remain.

Also record:

```text
aperture_fraction = true injected aperture flux / injected total flux
flux_pull = (measured flux - true injected aperture flux) / estimated flux error
```

The bias plot shows medians and the 16th–84th percentile **spread of individual
recoveries**, not confidence intervals on the median. A bin with very few usable
objects has an unreliable bias estimate. Bins with fewer than five usable
recoveries are omitted from the bias plot; their values and
`n_bias_measurements` remain in the summary table. They are not removed from
the recovery curves or their denominators.

### 8. Summarize the recovery uncertainty

For each cell, `p = recovered / injected`. The plot uses a binomial Wilson
interval with `z = 1` (approximately 68% coverage), which remains useful when
zero or all injections recover. It is not a standard deviation of the source
magnitudes and does not include PSF/model uncertainty or all effects of sharing
the same observed background image.

The summary includes the full field plus north/south and near-neighbour/isolated
subsets. These are overlapping diagnostic views, not independent populations
whose counts should be added. Subset sample sizes can be small.
Here `north` means the upper image half (`y >= image_height / 2`) and `south`
means the lower image half; these labels do not assert a celestial orientation.

## The six output files

| File | Contents |
|---|---|
| `results/recovery.ecsv` | One row per injection: truth, matches, flags, selection stages, output magnitude/errors, and photometric bias |
| `results/summary.csv` | Recovery fractions, interval bounds, sample sizes, and bias summaries per model/magnitude/subset |
| `results/manifest.json` | Configuration, input and code hashes, baseline checks, runtime, and limitations |
| `results/completeness.png` | Detection, usable, and counts recovery curves for the six models |
| `results/photometry_bias.png` | Total-magnitude and aperture-magnitude bias for the six models |
| `results/injection_positions.png` | All tested positions over the real image and its baseline mask |

ECSV preserves booleans, missing values, and column types. Fluxes are in ADU;
positions, distances, and radii are in native pixels. `detection_id` refers only
to that batch's catalogue and must not be interpreted as the original catalogue
ID. `injection_id` is the stable identifier in this output table.

## Implementation checks

The original full-field baseline and saved stellar classifications were
reproduced before injecting anything. A 24-injection end-to-end check also ran
successfully. Synthetic matching checks covered unique, missing, ambiguous,
shared, and pre-existing detections.

For the compact `r_e = 3` models, raising the rendering oversampling from 5 to 9
changed enclosed-flux fractions at radii 2, 4, 8, 16, and 32 pixels by less than
0.00008 in the checked subpixel phase. Templates are nonnegative and sum to one.
A repeated Poisson test verified the added source's ADU mean and variance,
including the gain conversion. Missing pixels and the original input array
were preserved. These numerical checks validate the implementation, not the
assumption that the model population represents every real galaxy.

## What this experiment can and cannot establish

This is a practical test of detection, quality cuts, fixed-reference star
selection, and aperture measurement for a controlled set of artificial galaxies.
It is substantially more informative than a single-region position-only test.

It is not a universal completeness correction. The real population has a mix
of sizes, axis ratios, angles, substructure, and profiles. The simulation uses
one saved PSF across the field, and does not inject a separate population of
stars to calibrate stellar contamination. Blended coincidences remain ambiguous.
The finite profiles and the assumed effective gain are further approximations.

In particular, **do not divide the observed aperture-magnitude counts by a
recovery curve indexed by injected total magnitude** without considering size
mixtures and migration between magnitude bins. Use these results first to
identify a defensible magnitude range and quantify model-dependent losses.
No detector threshold or counts slope is optimized using these results.
