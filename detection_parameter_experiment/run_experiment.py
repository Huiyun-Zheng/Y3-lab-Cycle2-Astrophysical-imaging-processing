"""Run all predeclared detector variants without changing the detection algorithm.

All outputs stay in this experiment directory. Previously completed stages are
reused on restart; inspect logs before manually removing any failed stage.
"""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'experiment_plan.json').read_text())


def execute(command, log_path):
    """Stream progress to the terminal and retain a complete per-stage log."""

    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['MPLCONFIGDIR'] = '/tmp/astro_mpl_experiment'
    lines = []
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env,
    )
    for line in process.stdout:
        print(line, end='', flush=True)
        lines.append(line)
    code = process.wait()
    # OneDrive can time out on repeated flushes; write the completed log once.
    log_path.write_text(''.join(lines))
    if code != 0:
        raise RuntimeError(f'Stage failed with exit code {code}; inspect {log_path}')


def run():
    rows = []
    for name, changes in PLAN['variants'].items():
        start = time.monotonic()
        directory = ROOT / 'variants' / name
        detection_dir = directory / 'detections'
        print(f'\nVARIANT: {name}; changes: {changes}', flush=True)

        if not (detection_dir / 'run_summary.json').exists():
            if detection_dir.exists():
                raise RuntimeError(f'Incomplete detection output: {detection_dir}')
            execute(
                [sys.executable, str(directory / 'source_detection_final.py')],
                directory / 'detection.log',
            )

        analysis_root = directory / 'analysis'
        summaries = sorted(analysis_root.glob('psf_run_*/run_summary.json'))
        if not summaries:
            execute(
                [sys.executable, str(ROOT / 'photometry_analysis.py'),
                 '--detections', str(detection_dir), '--output-root', str(analysis_root),
                 '--fit-range', '19', '24'],
                directory / 'analysis.log',
            )
            summaries = sorted(analysis_root.glob('psf_run_*/run_summary.json'))

        detection = json.loads((detection_dir / 'run_summary.json').read_text())
        analysis = json.loads(summaries[-1].read_text())
        row = {
            'variant': name,
            'detection_sigma': detection['settings']['detection_sigma'],
            'growth_sigma': detection['settings']['growth_sigma'],
            'smooth_sigma': detection['settings']['smooth_sigma'],
            'candidates': detection['candidates'],
            'negative_candidates': detection['negative_image_candidates'],
            'photometric_sources': analysis['photometric_sources'],
            'likely_stars': analysis['likely_stars'],
            'remaining_sources': analysis['remaining_sources'],
            'area_deg2': analysis['area_deg2'],
            'slope_19_24': analysis['descriptive_slope_intercept'][0],
            'analysis_directory': str(summaries[-1].parent.relative_to(ROOT)),
            'elapsed_seconds': round(time.monotonic() - start, 1),
        }
        rows.append(row)
        with (ROOT / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print('COMPLETED:', json.dumps(row), flush=True)


if __name__ == '__main__':
    run()
