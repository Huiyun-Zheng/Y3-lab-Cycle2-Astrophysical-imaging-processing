"""Call the existing detector without changing its files or settings."""

from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import io
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack


ROOT = Path(__file__).resolve().parent.parent
DETECTION = ROOT / "source_detection_neighbour_refined"
PHOTOMETRY = ROOT / "circular_moffat_analysis"
IMAGE_PATH = ROOT / "Astro"/ "Fits_Data" / "mosaic.fits"
CATALOGUE_PATH = PHOTOMETRY / "results" / "simple_analysis" / "classified_sources.ecsv"
CODE_PATHS = [
    DETECTION / "source_detection_improved.py",
    DETECTION / "source_detection_neighbour_refined.py",
    PHOTOMETRY / "circular_moffat.py",
    PHOTOMETRY / "photometry_analysis_simple.py",
]


def load_module(name, path):
    """Load an existing Python module from a given file path."""
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FixedPipeline:
    """
    Reuse the existing detection and photometry pipeline.

    Use a temporary workspace for each input image and keep the
    proginal PSF reference stars and magnitude limits fixed.
    """

    def __init__(self, workspace):
        workspace = Path(workspace)
        self.base = load_module("ir_base", CODE_PATHS[0])
        self.review = load_module("ir_review", CODE_PATHS[1])

        # The simple photometry module imports circular_moffat by name.
        self.moffat = load_module("circular_moffat", CODE_PATHS[2])
        self.photo = load_module("ir_photometry", CODE_PATHS[3])
        self.settings = self.base.Settings()

        self.image_path = workspace / "Astro" / "Fits_Data" / "mosaic.fits"
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        # Redirect the review module to the temporary workspace.
        self.review.ROOT = workspace
        self.review.INPUT = workspace / "before"
        self.review.OUTPUT = workspace / "after"

        # Disable diagnostic plotting to reduce runtime and usage.
        # (monkey patching)
        self.base.save_diagnostics = lambda *args, **kwargs: None
        self.review.save_plots = lambda *args, **kwargs: None

        self.original = Table.read(CATALOGUE_PATH)
        self.references = self.original[np.asarray(self.original["psf_reference"], bool)].copy()
        if len(self.references) < 5:
            raise ValueError("Need at least five saved PSF reference stars.")

        # Give the reference stars negative segments IDs so that
        # their IDs do not overlap with those of newly detected sources.
        self.reference_labels = -np.arange(1, len(self.references) + 1)
        self.references["segment_id"] = self.reference_labels

        self.measurement_limit = float(self.original.meta["measurement_mag_limit"])
        self.classification_limit = float(self.original.meta["classification_mag_limit"])

    def detect(self, image, header):
        """Run the existing detector on the full input image."""
        fits.PrimaryHDU(image, header=header).writeto(self.image_path, overwrite=True)

        with redirect_stdout(io.StringIO()):
            self.base.run_detection(self.image_path, self.review.INPUT, self.settings)
            catalogue = self.review.run()

        with fits.open(self.review.OUTPUT / "detection_products.fits") as hdul:
            products = {name: hdul[name].data.copy() for name in ("SOURCES", "DETMASK", "RMS")}
            products["DETMASK"] = products["DETMASK"].astype(bool)
            products["saturation"] = float(hdul[0].header["SATLEVEL"])

        return catalogue, products

    def check_baseline(self, catalogue):
        """Check that the pipeline reproduces the original results."""
        if len(catalogue) != len(self.original):
            raise ValueError("Baseline differs from the saved photometry catalogue; refresh it first.")

        # Compare the source positions, segment IDs, usability flags
        # and measured aperture fluxes with the original catalogue.
        for name in ("x", "y", "segment_id", "usable", "aperture_flux_adu"):
            np.testing.assert_allclose(
                catalogue[name],
                self.original[name],
                equal_nan=True,
                err_msg=f"Baseline mismatch: {name}",
            )

        #Rerun the original classifier using the saved reference stars
        # and the original classification magnitude limit.
        classified = self.moffat.select_psf_candidates(
            self.original,
            reference_labels=self.original["segment_id"][self.original["psf_reference"]],
            mag_limit=self.classification_limit,
        )
        # Check that every source receives the same class as before.
        np.testing.assert_array_equal(classified["source_class"], self.original["source_class"])

    def photometry(self, image, header, catalogue, products):
        """Measure and classify the injected sources using the existing photometry functions."""
        header = header.copy()
        header["SATLEVEL"] = products["saturation"]
        calibrated = self.photo.calibrate_magnitudes(catalogue, header)

        with redirect_stdout(io.StringIO()):
            measured = self.moffat.measure_psf_widths(
                image, products["SOURCES"],
                calibrated, header,
                mask=products["DETMASK"],
                mag_limit=self.measurement_limit,
            )
            measured = self.photo.add_photometric_uncertainties(measured, products["RMS"], products["DETMASK"], header,)

        # Keep the same columns in the reference table as in the newly measured catalogue.
        reference = self.references[measured.colnames].copy()
        measured.meta.clear()
        reference.meta.clear()
        # Append the original reference stars below the measured sources.
        combined = vstack([measured, reference], metadata_conflicts="silent")
        # Classify the sources using the original reference stars
        # and the original classification magnitude limit.
        # Keep only the newly measured sources in the output table.
        classified = self.moffat.select_psf_candidates(
            combined,
            reference_labels=self.reference_labels,
            mag_limit=self.classification_limit,
        )[:len(measured)]
        classified["is_star"] = classified["source_class"] == "star_candidate"
        classified["remaining_sample"] = classified["count_sample"] & ~classified["is_star"]

        # Return the final measured and classified catalogue.
        return classified
