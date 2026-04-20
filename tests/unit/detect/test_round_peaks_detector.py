"""
Focused test suite for RoundPeaksDetector algorithm-specific logic.

Tests focus on peak detection parameters. Grid inference, helper methods, and
edge refinement tests shared with SinePeakDetector are in
test_grid_inference_mixin.py.
"""

import pytest
import numpy as np
from phenotypic.detect import RoundPeaksDetector

from ..resources.TestHelper import timeit


class TestRoundPeaksDetectorPeakDetection:
    """Test peak detection parameter effects on detection quality."""

    @timeit
    @pytest.mark.parametrize(
            "thresh_method", ["otsu", "mean", "local", "triangle", "isodata", "li"]
    )
    def test_different_thresholding_methods(self, thresh_method, plate_12hr_grid_image):
        """Test that different thresholding methods all work."""
        image = plate_12hr_grid_image.copy()
        detector = RoundPeaksDetector(thresh_method=thresh_method)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_minimum_threshold_on_gridimage(self, plate_12hr_grid_image):
        """Test minimum threshold on GridImage."""
        image = plate_12hr_grid_image.copy()
        detector = RoundPeaksDetector(thresh_method="minimum")
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0, 5.0])
    def test_different_smoothing_sigma(self, sigma, plate_12hr_grid_image):
        """Test detection with different smoothing sigma values."""
        image = plate_12hr_grid_image.copy()
        detector = RoundPeaksDetector(smoothing_sigma=sigma)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("footprint_width", [1, 3, 6, 9])
    def test_different_footprint_width(self, footprint_width, plate_12hr_grid_image):
        """Test detection with different shape widths."""
        image = plate_12hr_grid_image.copy()
        detector = RoundPeaksDetector(footprint_width=footprint_width)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("noise_radius", [1, 2, 4])
    def test_different_noise_radius(self, noise_radius, plate_12hr_grid_image):
        """Test detection with different noise removal radii."""
        image = plate_12hr_grid_image.copy()
        detector = RoundPeaksDetector(noise_radius=noise_radius)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_with_custom_peak_distance(self, plate_12hr_grid_image):
        """Test detection with custom minimum peak distance."""
        image = plate_12hr_grid_image.copy()
        detector = RoundPeaksDetector(min_peak_distance=20)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_with_custom_peak_prominence(self, plate_12hr_grid_image):
        """Test detection with custom peak prominence."""
        image = plate_12hr_grid_image.copy()
        detector = RoundPeaksDetector(peak_prominence=0.15)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0


class TestRoundPeaksDetectorWideGrid:
    """Test wide plate grid inference specific to RoundPeaksDetector."""

    @timeit
    def test_infer_grid_shape_wide_plate(self):
        """Dense wide masks should be treated like a standard plate layout."""
        detector = RoundPeaksDetector()
        binary_image = np.zeros((200, 300), dtype=bool)
        binary_image[::20, ::25] = True

        inferred_rows, inferred_cols = detector._infer_grid_shape(binary_image)

        assert inferred_rows == 16
        assert inferred_cols == 24


class TestRoundPeaksDetectorReproducibility:
    """Test output consistency and reproducibility."""

    @timeit
    def test_detection_reproducibility(self, plate_12hr_grid_image):
        """Test that detection is reproducible with same parameters."""
        image1 = plate_12hr_grid_image.copy()
        image2 = plate_12hr_grid_image.copy()

        detector = RoundPeaksDetector(
                thresh_method="otsu",
                subtract_background=True,
                remove_noise=True,
                footprint_width=3,
                smoothing_sigma=2.0,
                edge_refinement=True,
        )

        result1 = detector.apply(image1, inplace=False)
        result2 = detector.apply(image2, inplace=False)

        assert result1.num_objects == result2.num_objects
        assert np.array_equal(result1.objmap[:], result2.objmap[:])

    @timeit
    def test_objmap_has_sequential_labels(self, plate_12hr_grid_image):
        """Test that objmap has properly sequential labels after detection."""
        image = plate_12hr_grid_image.copy()
        detector = RoundPeaksDetector()
        result = detector.apply(image, inplace=False)

        unique_labels = np.unique(result.objmap[:])
        assert unique_labels[0] == 0 or unique_labels[0] == 1

        if result.num_objects > 0:
            max_label = unique_labels[-1]
            assert max_label <= result.num_objects + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
