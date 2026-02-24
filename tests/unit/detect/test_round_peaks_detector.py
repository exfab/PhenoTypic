"""
Focused test suite for RoundPeaksDetector algorithm-specific logic.

Tests focus on complex features: grid inference, peak detection parameters,
and edge refinement. Basic initialization and apply() are covered by smoke tests
in test_operation.py.
"""

import pytest
import numpy as np
import phenotypic
from phenotypic.detect import RoundPeaksDetector
from phenotypic.data import load_plate_12hr

from ..resources.TestHelper import timeit


class TestRoundPeaksDetectorGridInference:
    """Test grid inference algorithms unique to RoundPeaksDetector."""

    @timeit
    @pytest.mark.parametrize(
            "nrows,ncols",
            [
                (8, 12),  # 96-well plate
                (16, 24),  # 384-well plate
                (32, 48),  # 1536-well plate
                (4, 6),  # Small grid
            ],
    )
    def test_infer_grid_shape_with_synthetic_data(self, nrows, ncols):
        """Test grid shape inference with synthetic gridded data."""
        detector = RoundPeaksDetector()

        # Create synthetic binary image with grid pattern
        height, width = 200, 300
        binary_image = np.zeros((height, width), dtype=bool)

        row_spacing = height // nrows
        col_spacing = width // ncols

        for r in range(nrows):
            for c in range(ncols):
                # Add a small colony at each grid position
                r_center = r * row_spacing + row_spacing // 2
                c_center = c * col_spacing + col_spacing // 2
                binary_image[
                    r_center - 3: r_center + 3, c_center - 3: c_center + 3
                ] = True

        inferred_rows, inferred_cols = detector._infer_grid_shape(binary_image)

        # Inference should be close to actual grid size
        assert inferred_rows > 0
        assert inferred_cols > 0
        assert inferred_rows <= nrows * 2
        assert inferred_cols <= ncols * 2

    @timeit
    def test_infer_grid_shape_blank_defaults(self):
        """Blank mask should default to an 8x12 plate."""
        detector = RoundPeaksDetector()
        binary_image = np.zeros((120, 180), dtype=bool)

        inferred_rows, inferred_cols = detector._infer_grid_shape(binary_image)

        assert inferred_rows == 8
        assert inferred_cols == 12

    @timeit
    def test_infer_grid_shape_wide_plate(self):
        """Dense wide masks should be treated like a standard plate layout."""
        detector = RoundPeaksDetector()
        binary_image = np.zeros((200, 300), dtype=bool)
        binary_image[::20, ::25] = True  # seed objects across the plate

        inferred_rows, inferred_cols = detector._infer_grid_shape(binary_image)

        assert inferred_rows == 16
        assert inferred_cols == 24


class TestRoundPeaksDetectorPeakDetection:
    """Test peak detection parameter effects on detection quality."""

    @timeit
    @pytest.mark.parametrize(
            "thresh_method", ["otsu", "mean", "triangle", "minimum", "isodata", "li"]
    )
    def test_different_thresholding_methods(self, thresh_method):
        """Test that different thresholding methods all work."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = RoundPeaksDetector(thresh_method=thresh_method)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0, 5.0])
    def test_different_smoothing_sigma(self, sigma):
        """Test detection with different smoothing sigma values."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = RoundPeaksDetector(smoothing_sigma=sigma)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("footprint_width", [1, 3, 6, 9])
    def test_different_footprint_width(self, footprint_width):
        """Test detection with different shape widths."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = RoundPeaksDetector(footprint_width=footprint_width)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_with_custom_peak_distance(self):
        """Test detection with custom minimum peak distance."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = RoundPeaksDetector(min_peak_distance=20)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_with_custom_peak_prominence(self):
        """Test detection with custom peak prominence."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = RoundPeaksDetector(peak_prominence=0.15)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0


class TestRoundPeaksDetectorEdgeRefinement:
    """Test edge refinement logic."""

    @timeit
    def test_refine_edges_maintains_count(self):
        """Test that _refine_edges maintains the number of edges."""
        detector = RoundPeaksDetector()
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[10:20, :] = True

        initial_edges = np.array([0, 25, 50, 75, 100])
        refined_edges = detector._refine_edges(binary_image, initial_edges, axis=0)

        assert len(refined_edges) == len(initial_edges)
        assert refined_edges[0] == 0  # First edge should remain at border
        assert refined_edges[-1] == 100  # Last edge should remain at border
        assert np.all(np.diff(refined_edges) >= 0)

    @timeit
    def test_estimate_edges_returns_correct_number(self):
        """Test that _estimate_edges returns n_bins+1 edges."""
        detector = RoundPeaksDetector()
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[10:20, :] = True
        binary_image[30:40, :] = True
        binary_image[50:60, :] = True

        n_bins = 8
        edges = detector._estimate_edges(binary_image, axis=0, n_bins=n_bins)

        assert len(edges) == n_bins + 1
        assert edges[0] == 0  # Should start at 0
        assert edges[-1] == binary_image.shape[1]  # Should end at image width
        assert np.all(np.diff(edges) >= 0)  # Should be non-decreasing


class TestRoundPeaksDetectorHelperMethods:
    """Test helper methods for algorithm implementation."""

    @timeit
    def test_thresholding_creates_binary_mask(self):
        """Test that _thresholding creates a valid binary mask."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = RoundPeaksDetector()

        matrix = image.detect_mat[:]
        binary_mask = detector._thresholding(matrix)

        assert binary_mask.dtype == bool or binary_mask.dtype == np.bool_
        assert binary_mask.shape == matrix.shape
        assert np.all((binary_mask == 0) | (binary_mask == 1))

    @timeit
    def test_clean_and_sum_binary_axis0(self):
        """Test _clean_and_sum_binary for axis=0 (rows)."""
        detector = RoundPeaksDetector()
        # Create simple test pattern
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[20:30, 20:80] = True  # Horizontal stripe
        binary_image[50:60, 20:80] = True  # Another stripe

        sums = detector._clean_and_sum_binary(binary_image, axis=0)

        assert len(sums) == 100
        assert sums.sum() > 0

    @timeit
    def test_clean_and_sum_binary_axis1(self):
        """Test _clean_and_sum_binary for axis=1 (columns)."""
        detector = RoundPeaksDetector()
        # Create simple test pattern
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[20:80, 20:30] = True  # Vertical stripe
        binary_image[20:80, 50:60] = True  # Another stripe

        sums = detector._clean_and_sum_binary(binary_image, axis=1)

        assert len(sums) == 100
        assert sums.sum() > 0


class TestRoundPeaksDetectorReproducibility:
    """Test output consistency and reproducibility."""

    @timeit
    def test_detection_reproducibility(self):
        """Test that detection is reproducible with same parameters."""
        image1 = phenotypic.GridImage(load_plate_12hr())
        image2 = phenotypic.GridImage(load_plate_12hr())

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

        # Results should be identical
        assert result1.num_objects == result2.num_objects
        assert np.array_equal(result1.objmap[:], result2.objmap[:])

    @timeit
    def test_objmap_has_sequential_labels(self):
        """Test that objmap has properly sequential labels after detection."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = RoundPeaksDetector()
        result = detector.apply(image, inplace=False)

        unique_labels = np.unique(result.objmap[:])
        # Labels should start from 0 (background) or 1 (first object)
        assert unique_labels[0] == 0 or unique_labels[0] == 1

        if result.num_objects > 0:
            # Labels should be reasonably sequential after relabeling
            max_label = unique_labels[-1]
            assert max_label <= result.num_objects + 1


# Run all tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
