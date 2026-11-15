# Euclidean Detection Comparison

`source_detection_copy` with the same input simulated sources.

## Run

From the `codes_1105` directory:

python3 compare_euclidean_detection.py

Optional arguments:

python3 compare_euclidean_detection.py --n-sources 1000 --size 2048 --seed 42

To specify the FITS file and output directory:

python3 compare_euclidean_detection.py --fits "/path/to/mosaic.fits" --output "outputs/euclidean_comparison"

- Original pipeline only
python3 compare_euclidean_detection.py --pipeline original

- Copy pipeline only
python3 compare_euclidean_detection.py --pipeline copy

- Both pipelines (default)
python3 compare_euclidean_detection.py --pipeline both

## Simulation

- Uniform spatial distribution in Euclidean volume.
- Fixed absolute magnitude of −20; apparent magnitudes depend on distance.
- Gaussian major-axis sigma between 2 and 5 pixels, with axis ratios between 0.5 and 1.
- Randomly selected grid positions with small random offsets.
- Uniform Gaussian sky noise; no additional source photon noise.
- Default seed: 42.

## Outputs

The default output directory is `outputs/euclidean_comparison/`.

- `injected.ecsv`: input source positions, magnitudes, fluxes and sizes.
- `simulated_image.fits`: the image supplied to both pipelines.
- `source_detection.ecsv` and `source_detection_copy.ecsv`: measured catalogues.
- `*_recovery.ecsv`: matching results using a 3-pixel radius.
- `aperture_comparison.png`: aperture overlays for both methods.
- `counts_and_recovery.png` and `.pdf`: cumulative magnitude counts and recovery fractions.
- `simulation.json`: simulation settings.

Positional recovery and usable photometry are recorded separately. This is an ideal-noise test, not a completeness calibration for the real mosaic.



# Injection–Recovery Experiment

Inject Gaussian sources and assess their recovery using the
`source_detection_copy` pipeline.

## Run

Run from the `codes_1105` directory.

### Real background

```bash
python3 injection_recovery_many.py \
    --fits "/path/to/mosaic.fits" \
    --output outputs/recovery_raw \
    --background raw
```

Sources are added to real image cutouts. Each cutout is detected before
and after injection. Placements avoid excluded regions and limit overlap
between injected sources; existing astronomical sources remain present.

### Simulated background

```bash
python3 injection_recovery_many.py \
    --fits "/path/to/mosaic.fits" \
    --output outputs/recovery_sim \
    --background sim \
    --n-sources 100 \
    --size 200
```

The FITS file supplies calibration information and reference background
statistics. The simulation uses Euclidean catalogue magnitudes,
random-width Gaussian sources, Gaussian sky noise and source Poisson
noise. Injected sources may overlap.

## Main parameters

| Argument | Default | Meaning |
|---|---:|---|
| `--background` | `sim` | `raw` or `sim` |
| `--seed` | 42 | Random seed |
| `--match-radius` | 3 | Matching radius in pixels |
| `--sources-per-frame` | 5 | Sources per real cutout |
| `--max-frames` | 3 | Maximum number of real cutouts |
| `--spacing` | 200 | Real-cutout side length and centre spacing |
| `--grid-margin` | 200 | Margin for real-cutout centres |
| `--n-sources` | 100 | Number of simulated sources |
| `--size` | 200 | Simulated image side length in pixels |

## Outputs

Both modes save:

- `injected.ecsv`: input source positions, fluxes and shapes.
- `detected.ecsv`: detections after injection, including flagged rows.
- `original_detected.ecsv`: detections before injection; empty in simulated mode.
- `matches.ecsv`: one matching result per injected source.
- `experiment_summary.json`: settings and recovery statistics.

Per-frame detector outputs are saved under `frame_XXXX/`.

## Recovery definitions

- `unique_position_match`: exactly one candidate within the matching
  radius, not shared with another injection.
- `usable_out`: the corresponding detection passes pipeline quality checks.
- `status == 'matched'`: additionally requires no nearby original or
  injected sources, and positive finite output flux.
- `clean_usable_recovery`: `matched` and `usable_out` are both true.

Other statuses include `no_match`, `ambiguous`, `shared_detection`,
`near_original`, `near_injected` and `invalid_flux`.

The clean recovery fraction is conservative: it excludes crowded cases
even when a positional counterpart exists. It should not be interpreted
directly as detection completeness.


# Flux Ratio Plot

Plot `flux_out / flux_in` against input flux, coloured by matching status.

## Usage

Experiments automatically save `flux_ratio.png` in their output directory.

To plot an existing result, run from `codes_1105`:

```python
from astropy.table import Table
from injection_recovery_many import plot_flux_ratio

matches = Table.read("outputs/recovery_sim_seed_43/matches.ecsv")

fig, ax = plot_flux_ratio(
    matches,
    filename="outputs/recovery_sim_seed_43/flux_ratio.png",
    show=False
)
```

## Interpretation

- **Green:** `matched`
- **Blue:** `near_injected`
- **Red:** `shared_detection`
- **Dashed line:** output flux equals input flux.
- The input-flux axis uses a logarithmic scale.
- Rows without finite flux measurements are omitted.

Legend counts show **plotted injection rows / total injection rows**,
not the number of distinct detections.

For shared detections, each injection contributes its own point:
the shared output flux divided by that injection’s input flux.
These points are not independent single-source measurements.

The plotting function does not modify the matching table.