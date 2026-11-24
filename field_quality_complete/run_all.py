"""Reproduce the reviewed field from the original image, in this folder only."""
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent


def main():
    # Each stage imports the local copies. No previous detection is used as input.
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MPLBACKEND='Agg', MPLCONFIGDIR=str(BASE / 'results' / 'mpl_cache'),
               PYTHONDONTWRITEBYTECODE='1')
    stages = ['run_aperture_review.py', 'run_field_quality_review.py',
              'validate_field_quality_review.py', 'compare_reference.py']
    for index, stage in enumerate(stages, 1):
        print(f'\nStage {index}/{len(stages)}: {stage}', flush=True)
        subprocess.run([sys.executable, '-u', '-B', str(BASE / stage)],
                       cwd=BASE, env=env, check=True)


if __name__ == '__main__':
    main()
