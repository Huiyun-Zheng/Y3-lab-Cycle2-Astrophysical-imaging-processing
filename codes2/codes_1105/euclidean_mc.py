"""
Monte Carlo Simulation of Euclidean Space Assuming Uniform Distribution
Can be used to examine the theoretical claim
"""


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def simulate_euclidean(
        n_sources=100000, absolute_magnitude=-20,
        max_distance=1e10, seed=42
        ):
    """Return distances in pc and apparent magnitudes."""

    rng = np.random.default_rng(seed)

    # Uniform in volume, not in distance
    u = 1 - rng.random(n_sources)
    distances = max_distance * u**(1 / 3)

    # Same absolute magnitude for every galaxy
    magnitudes = absolute_magnitude + 5 * np.log10(distances / 10)

    return distances, magnitudes


if __name__ == '__main__':
    n_sources = 100000
    absolute_magnitude = -20
    max_distance = 1e10  # pc; simulation boundary
    seed = 42

    distances, magnitudes = simulate_euclidean(
        n_sources, absolute_magnitude, max_distance, seed
    )

    # Cumulative counts
    m_limit = absolute_magnitude + 5 * np.log10(max_distance / 10)
    mag_thresholds = np.arange(m_limit - 9, m_limit + 0.01, 0.25)
    populations = []
    for m in mag_thresholds:
        populations.append(np.count_nonzero(magnitudes < m))
    populations = np.array(populations)

    # Fit where counts are reasonably large
    fitting_mask = (populations >= 100) & (mag_thresholds <= m_limit - 0.5)
    coeffs = np.polyfit(
        mag_thresholds[fitting_mask],
        np.log10(populations[fitting_mask]), deg=1
    )
    fit = np.poly1d(coeffs)
    slope = coeffs[0]
    print(f'Seed: {seed}')
    print(f'Fitted slope: {slope:.4f}; expected: 0.6')

    # Exact expectation for this ideal model
    expected_log_counts = np.log10(n_sources) + 0.6 * (mag_thresholds - m_limit)
    positive = populations > 0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        mag_thresholds[positive], np.log10(populations[positive]),
        s=22, label='Monte Carlo counts'
    )
    ax.plot(mag_thresholds, expected_log_counts, '--', color='black', label='Euclidean slope = 0.6')
    ax.plot(
        mag_thresholds[fitting_mask], fit(mag_thresholds[fitting_mask]),
        color='red', label=f'Fitted slope = {slope:.4f}'
    )
    ax.set_xlabel('Magnitude threshold, m')
    ax.set_ylabel('log10 N(<m)')
    ax.set_title(f'Uniform Euclidean simulation: {n_sources:,} galaxies, seed {seed}')
    ax.legend()
    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / 'euclidean_mc.png', dpi=180)
    plt.show()
    plt.close(fig)