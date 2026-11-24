"""Compare freshly computed results with the saved historical reference."""
from pathlib import Path
import hashlib
import numpy as np
from astropy.table import Table
from astropy.io import fits

BASE = Path(__file__).resolve().parent
REFERENCE = BASE.parent / 'bright_source_counts_experiment'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    lines = []
    failures = []
    # Metadata contains new paths and hashes; compare the scientific columns.
    for folder in ['aperture_review', 'field_quality_review']:
        for filename in ['sources.ecsv', 'classified_sources.ecsv', 'number_counts.ecsv']:
            old = Table.read(REFERENCE / 'results' / folder / filename)
            new = Table.read(BASE / 'results' / folder / filename)
            if set(old.colnames) != set(new.colnames) or len(old) != len(new):
                failures.append(f'{folder}/{filename}: table schema/length differs')
                continue
            if old.colnames != new.colnames:
                lines.append(f'{folder}/{filename}: column order differs; compare all columns by name')
            exact = 0
            for col in old.colnames:
                a, b = np.asarray(old[col]), np.asarray(new[col])
                numeric = a.dtype.kind in 'fc'
                equal = np.array_equal(a, b, equal_nan=True) if numeric else np.array_equal(a, b)
                if equal:
                    exact += 1
                elif numeric and np.allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=True):
                    lines.append(f'Numerical rounding only: {folder}/{filename}:{col}')
                else:
                    failures.append(f'{folder}/{filename}:{col} differs')
            lines.append(f'{folder}/{filename}: {len(new)} rows; {exact}/{len(old.colnames)} columns exactly equal')
            if filename == 'number_counts.ecsv':
                for key in ['slope', 'slope_error_propagated', 'area_deg2']:
                    if not np.isclose(old.meta[key], new.meta[key], rtol=1e-10, atol=1e-12):
                        failures.append(f'{folder}: {key} differs')
                lines.append(f"Slope: {new.meta['slope']:.12f} +/- {new.meta['slope_error_propagated']:.12f}")
    # Verify newly constructed image products, including masks and segmentation.
    with fits.open(REFERENCE / 'results/aperture_review/detection_products.fits') as old, fits.open(BASE / 'results/aperture_review/detection_products.fits') as new:
        if [h.name for h in old] != [h.name for h in new]:
            failures.append('FITS extension names differ')
        for a, b in zip(old, new):
            if a.data is not None and not np.array_equal(a.data, b.data, equal_nan=True):
                failures.append(f'FITS {a.name} differs')
        lines.append('Compared all regenerated FITS image arrays against the reference.')
    protected = Table.read(BASE / 'protected_files.tsv', format='ascii.tab')
    for row in protected:
        if sha(BASE.parent / row['path']) != row['sha256']:
            failures.append(f"Original file changed: {row['path']}")
    lines.append(f'Checked {len(protected)} protected original files.')
    lines += ['FAIL: ' + item for item in failures]
    lines.append('PASS: full detection-to-counts reproduction.' if not failures else 'FAIL: reproduction differs; inspect listed columns.')
    (BASE / 'results/REPRODUCTION_CHECK.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if failures:
        raise AssertionError('Reference comparison failed')


if __name__ == '__main__':
    main()
