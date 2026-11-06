"""Regression checks for detection, deblending and saturation-connected masks.

Run: python -m unittest -v test_source_detection_auto
Synthetic images test the algorithm; they do not establish survey completeness.
"""

import unittest

import numpy as np

from source_detection_auto import (
    Settings, find_missing_data, grow_peak_regions, identify_artefacts,
    select_significant_peaks, separate_sources,
)


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.yy, self.xx = np.mgrid[:100, :100]
        self.excluded = np.zeros((100, 100), dtype=bool)

    def gaussian(self, x, y, height, width):
        distance_squared = (self.xx - x) ** 2 + (self.yy - y) ** 2
        return height * np.exp(-distance_squared / (2 * width ** 2))

    def test_faint_companion_to_much_brighter_source(self):
        # A companion with only 1% of its neighbour's peak must still survive.
        surface = self.gaussian(35, 50, 10000, 5) + self.gaussian(61, 50, 100, 5)
        labels, _ = separate_sources(surface, surface, self.excluded, self.settings)
        self.assertGreater(labels[50, 35], 0)
        self.assertGreater(labels[50, 61], 0)
        self.assertNotEqual(labels[50, 35], labels[50, 61])

    def test_flood_retains_every_footprint_pixel_and_seed(self):
        surface = self.gaussian(40, 50, 250, 5) + self.gaussian(54, 42, 13, 3)
        region = surface >= self.settings.growth_sigma
        positions = [(50, 40), (42, 54)]
        labels = grow_peak_regions(surface, region, positions)
        np.testing.assert_array_equal(labels > 0, region)

        for number, position in enumerate(positions, start=1):
            self.assertEqual(labels[position], number)
            self.assertGreaterEqual(np.count_nonzero(labels == number), self.settings.min_pixels)

    def test_small_bump_on_profile_is_merged(self):
        surface = np.tile(np.array([10, 20, 18, 19, 10], dtype=float), (3, 1))
        positions = [(1, 1), (1, 3)]
        selected = select_significant_peaks(
            surface, np.ones_like(surface, dtype=bool), positions, self.settings
        )
        self.assertEqual(selected, [(1, 1)])

    def test_masked_bright_pixel_cannot_hide_valid_source(self):
        surface = self.gaussian(50, 50, 20, 1.5)
        surface[50, 53] = 10000
        self.excluded[50, 53] = True
        labels, _ = separate_sources(surface, surface, self.excluded, self.settings)
        self.assertGreater(labels[50, 50], 0)
        self.assertEqual(labels[50, 53], 0)

    def test_equal_height_plateau_has_one_seed(self):
        surface = np.zeros((100, 100))
        surface[45:55, 45:55] = 10
        labels, _ = separate_sources(surface, surface, self.excluded, self.settings)
        self.assertEqual(labels.max(), 1)

    def test_empty_image_and_single_pixel_spike_are_not_sources(self):
        surface = np.zeros((100, 100))

        for value in (0, 100):
            surface[50, 50] = value
            labels, metadata = separate_sources(surface, surface, self.excluded, self.settings)
            self.assertEqual(labels.max(), 0)
            self.assertEqual(metadata, {})

    def test_multiple_curved_bleeds_are_masked(self):
        rng = np.random.default_rng(16)
        image = rng.normal(100, 2, (100, 100))
        trail_y = np.arange(20, 80)
        trail_x = np.rint(70 + 5 * np.sin(trail_y / 10)).astype(int)
        image[trail_y, trail_x] = 180
        image[trail_y[0], trail_x[0]] = 400
        image[30:61, 25] = 180
        image[45, 25] = 400
        image[:3] = 100
        missing = find_missing_data(image)
        saturated, artefact, _, _ = identify_artefacts(image, missing, 300, self.settings)
        self.assertTrue(np.all(artefact[trail_y, trail_x]))
        self.assertTrue(np.all(artefact[30:61, 25]))
        self.assertTrue(np.all(artefact[saturated]))
        self.assertTrue(np.all(missing[:3]))
        self.assertFalse(artefact[50, 45])


if __name__ == "__main__":
    unittest.main()
