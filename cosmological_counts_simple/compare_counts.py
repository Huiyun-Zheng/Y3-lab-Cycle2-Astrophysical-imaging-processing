"""Compare differential count slopes in the same bins as the observed catalogue."""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.integrate import trapezoid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)
# Blanton et al. (2001), r* band: M* - 5 log10(h) = -20.83.
M_STAR = -20.83 + 5 * np.log10(COSMOLOGY.h)
ALPHA = -1.20
DRAWS = 1000


def intrinsic_counts(magnitudes, k_coefficient):
    """Integrate the fixed luminosity function over comoving redshift shells."""
    z = np.geomspace(1e-5, 6, 2400)
    distance_modulus = COSMOLOGY.distmod(z).value
    shell_volume = COSMOLOGY.differential_comoving_volume(z).value
    k_correction = k_coefficient * np.log10(1 + z)

    density = []
    for magnitude in magnitudes:
        absolute_mag = magnitude - distance_modulus - k_correction
        luminosity_ratio = 10 ** (0.4 * (M_STAR - absolute_mag))
        # Constant factors cancel when we compare slopes.
        phi = luminosity_ratio ** (ALPHA + 1) * np.exp(-luminosity_ratio)
        density.append(trapezoid(phi * shell_volume, z))
    return np.array(density)


def slope(x, density):
    return np.polyfit(x, np.log10(density), 1)[0]


def run():
    observed = Table.read(ROOT / 'circular_moffat_analysis/results/simple_analysis/differential_number_counts.ecsv')
    injections = Table.read(ROOT / 'injection_recovery/results/recovery.ecsv')
    faint, bright = max(observed.meta['fit_range']), min(observed.meta['fit_range'])
    fit = observed[(observed['magnitude_centre'] >= bright) & (observed['magnitude_centre'] <= faint)]
    x = np.asarray(fit['magnitude_centre'])
    edges = np.r_[fit['magnitude_lower'], fit['magnitude_upper'][-1]]
    widths = np.diff(edges)
    measured = np.asarray(fit['density_deg2_mag'])
    measured_slope = slope(x, measured)
    np.testing.assert_allclose(measured_slope, observed.meta['slope'])

    # Re-use the same bootstrap samples for each model. Preserve every cell size.
    trials = injections.group_by(["profile", "r_eff_pix", "mag_low"])
    indices = trials.groups.indices
    rng = np.random.default_rng(20261127)
    samples = np.concatenate([rng.integers(a, b, size=(DRAWS, b - a))
                              for a, b in zip(indices[:-1], indices[1:])], axis=1)
    low, high = min(trials['mag_low']), max(trials['mag_high'])
    grid = np.linspace(low, high, round((high - low) * 100) + 1)
    # The all-field experiment draws equal numbers per profile, radius and bin.
    assert len(set(np.diff(indices))) == 1
    ok = np.asarray(trials['counts_recovered'], bool)
    m_out = np.asarray(trials['magnitude_out'])
    models = {'Euclidean': 10 ** (0.6 * (grid - 20)),
              'Flat LCDM, K=0': intrinsic_counts(grid, 0),
              'Flat LCDM, example K': intrinsic_counts(grid, 2.5)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, layout='constrained')
    rows = [('Observed', measured_slope, observed.meta['slope_error'])]
    for name, density in models.items():
        true_bins = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = np.linspace(a, b, 101)
            true_bins.append(trapezoid(np.interp(m, grid, density), m) / (b - a))
        weights = np.interp(trials['magnitude_in'], grid, density)
        recovered = np.histogram(m_out[ok], edges, weights=weights[ok])[0] / widths
        slopes = []
        for sample in samples:
            chosen = sample[ok[sample]]
            counts = np.histogram(m_out[chosen], edges, weights=weights[chosen])[0] / widths
            if np.all(counts > 0):
                slopes.append(slope(x, counts))
        for ax, values, stage, error in zip(axes, [true_bins, recovered], ['input', 'recovered'],
                                            [None, np.std(slopes, ddof=1)]):
            value = slope(x, values)
            rows.append((name + ' / ' + stage, value, error))
            # A free vertical offset compares shapes on the observed density scale.
            log_y = np.log10(values)
            offset = np.mean(np.log10(measured) - log_y)
            ax.plot(x, log_y + offset, label=f'{name} ({value:.3f})')
    for ax, title in zip(axes, ['Intrinsic model counts', 'After detection and aperture photometry']):
        ax.errorbar(x, np.log10(measured), yerr=fit['log_error_mc'], fmt='ko', label='Observed', ms=4)
        ax.set(title=title, xlabel='Magnitude (bin centre)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Log differential surface density (models shifted)')
    fig.savefig(HERE / 'counts_comparison.png', dpi=220)
    plt.close(fig)
    with (HERE / 'slopes.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model_and_stage', 'slope', 'statistical_error'])
        writer.writerows(rows)


if __name__ == '__main__':
    run()
