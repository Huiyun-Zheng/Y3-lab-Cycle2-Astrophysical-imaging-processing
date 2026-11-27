# Cosmology and differential galaxy counts

Run `python compare_counts.py` with NumPy, SciPy, Astropy and Matplotlib. The script reads the final differential count table and the per-injection recovery table from the project. It writes one figure and one CSV beside the script. The saved comparison uses six 0.5 mag bins between 19 and 22.

The Euclidean model has differential counts proportional to 10^(0.6m). The second model uses flat ΛCDM with H0 = 70 km/s/Mpc and Ωm = 0.3. At each redshift compute absolute magnitude M = m − distance modulus − K, then integrate φ(M) × dV/(dz dΩ) over redshift. The Schechter luminosity function per magnitude is proportional to X^(α+1) exp(−X), where X = 10^[0.4(M*−M)]. Its shape and comoving density stay fixed with redshift. The local SDSS commissioning r* luminosity function provides α = −1.20 and M* − 5 log10(h) = −20.83, giving M* ≈ −21.60 for h = 0.7. The normalization cancels when fitting a slope. SDSS Petrosian magnitudes provide an approximate population model for the input total magnitudes.

The main comparison sets K = 0 to isolate luminosity-distance and volume effects. The third model illustrates sensitivity to band shifting with K = 2.5 log10(1+z), the same-filter AB approximation for a power-law spectrum fν proportional to ν^(−2). Real galaxies have different spectra; this example is not a measured correction for this field.

The input slope uses densities averaged over each bin. To predict measured counts, weight each injection by model density at its true magnitude. Keep final counts recoveries and histogram their output aperture magnitudes. This uses the six injected profile/size combinations in equal proportions at all magnitudes. The available input range is 18–24.5. Migration from outside that range and residual stars in the observed sample are not simulated.

Fit log10 differential density against bin centre with an unweighted straight line. In the figure each model has its own constant vertical shift, chosen to match the mean observed log density. The left panel gives the intrinsic predictions on the total-magnitude scale; the right panel compares predicted and observed aperture counts. The CSV errors for recovered models are sample standard deviations from 1,000 bootstrap draws within each original injection cell. They describe finite injection sampling conditional on the mosaic. The observed error comes from the existing counts Monte Carlo. Uncertainties in the luminosity function, K-correction, size mixture and cosmic variance are outside those errors.

The model parameters are fixed independently of the observed slope. Agreement tests whether a no-evolution example can account for the shape. These single-band counts cannot uniquely determine galaxy evolution or cosmological parameters.

Sources: [Blanton et al. (2001), SDSS luminosity function](https://arxiv.org/abs/astro-ph/0012085); [Yasuda et al. (2001), SDSS galaxy counts](https://arxiv.org/abs/astro-ph/0105545).
