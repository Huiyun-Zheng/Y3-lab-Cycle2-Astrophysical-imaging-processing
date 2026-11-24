"""Run the repository detector on FITS files and in-memory injection images."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from astropy.io import fits
from astropy.table import Table

# Use the detector at the repository root regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from source_detection_final import Settings, run_detection as run_final_detection


def run_detection(data, output_dir, settings=None, *, header=None):
    """Return the final detector's catalogue and masks for a path or array."""
    output_dir = Path(output_dir).absolute()
    settings = Settings() if settings is None else settings

    # Isolate the final detector's Matplotlib backend from the caller.
    command = [sys.executable, str(Path(__file__).resolve())]
    arguments = [str(output_dir), json.dumps(asdict(settings))]
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("The input must be a two-dimensional image.")

        header = fits.Header() if header is None else header.copy()
        for keyword in ("BSCALE", "BZERO", "BLANK"):
            header.remove(keyword, ignore_missing=True)
        with TemporaryDirectory(prefix="astro-detection-") as scratch:
            image_path = Path(scratch) / "ndarray"
            fits.writeto(image_path, data.astype(np.float32), header)
            subprocess.run([*command, str(image_path), *arguments], check=True)

        # Keep array inputs distinct from persistent FITS paths in the summary.
        summary_path = output_dir / "run_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["input"] = None
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    else:
        image_path = Path(data).absolute()
        subprocess.run([*command, str(image_path), *arguments], check=True)

    catalogue = Table.read(output_dir / "sources.ecsv")
    with fits.open(output_dir / "detection_products.fits") as products:
        excluded = products["BADMASK"].data.astype(bool)
        processed = products["PROCESSED"].data.astype(bool)
    return catalogue, excluded, processed


if __name__ == "__main__":
    image_path, output_dir, settings_json = sys.argv[1:]
    run_final_detection(image_path, output_dir, Settings(**json.loads(settings_json)))
