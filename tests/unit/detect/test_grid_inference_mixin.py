"""Shared tests for GridInferenceMixin behavior across detector implementations.

These tests verify mixin methods that are identical for RoundPeaksDetector
and SinePeakDetector, avoiding duplication.
"""

import pytest
import numpy as np
from phenotypic.detect import RoundPeaksDetector, SinePeakDetector

from ..resources.TestHelper import timeit

DETECTORS = [RoundPeaksDetector, SinePeakDetector]


class TestGridInferenceMixinShared:
    """Grid inference tests parametrized across detectors."""

    @timeit
    @pytest.mark.parametrize("DetectorClass", DETECTORS)
    @pytest.mark.parametrize(
        "nrows,ncols",
        [
            (8, 12),   # 96-well plate
            (16, 24),  # 384-well plate
            (32, 48),  # 1536-well plate
            (4, 6),    # Small grid
        ],
    )
    def test_infer_grid_shape_with_synthetic_data(self, DetectorClass, nrows, ncols):
        """Test grid shape inference with synthetic gridded data."""
        detector = DetectorClass()

        height, width = 200, 300
        binary_image = np.zeros((height, width), dtype=bool)

        row_spacing = height // nrows
        col_spacing = width // ncols

        for r in range(nrows):
            for c in range(ncols):
                r_center = r * row_spacing + row_spacing // 2
                c_center = c * col_spacing + col_spacing // 2
                binary_image[
                    r_center - 3:r_center + 3, c_center - 3:c_center + 3
                ] = True

        inferred_rows, inferred_cols = detector._infer_grid_shape(binary_image)

        assert inferred_rows > 0
        assert inferred_cols > 0
        assert inferred_rows <= nrows * 2
        assert inferred_cols <= ncols * 2

    @timeit
    @pytest.mark.parametrize("DetectorClass", DETECTORS)
    def test_infer_grid_shape_blank_defaults(self, DetectorClass):
        """Blank mask should default to an 8x12 plate."""
        detector = DetectorClass()
        binary_image = np.zeros((120, 180), dtype=bool)

        inferred_rows, inferred_cols = detector._infer_grid_shape(binary_image)

        assert inferred_rows == 8
        assert inferred_cols == 12


class TestHelperMethodsShared:
    """Helper method tests parametrized across detectors."""

    @timeit
    @pytest.mark.parametrize("DetectorClass", DETECTORS)
    @pytest.mark.parametrize("input_val,expected", [
        (1, 3), (2, 3), (3, 3), (4, 5), (5, 5), (6, 7), (10, 11),
    ])
    def test_round_odd(self, DetectorClass, input_val, expected):
        """Test _round_odd enforces odd integers with minimum 3."""
        assert DetectorClass._round_odd(input_val) == expected

    @timeit
    @pytest.mark.parametrize("DetectorClass", DETECTORS)
    def test_clean_and_sum_binary_axis0(self, DetectorClass):
        """Test _clean_and_sum_binary for axis=0 (rows)."""
        detector = DetectorClass()
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[20:30, 20:80] = True
        binary_image[50:60, 20:80] = True

        sums = detector._clean_and_sum_binary(binary_image, axis=0)

        assert len(sums) == 100
        assert sums.sum() > 0

    @timeit
    @pytest.mark.parametrize("DetectorClass", DETECTORS)
    def test_clean_and_sum_binary_axis1(self, DetectorClass):
        """Test _clean_and_sum_binary for axis=1 (columns)."""
        detector = DetectorClass()
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[20:80, 20:30] = True
        binary_image[20:80, 50:60] = True

        sums = detector._clean_and_sum_binary(binary_image, axis=1)

        assert len(sums) == 100
        assert sums.sum() > 0

    @timeit
    @pytest.mark.parametrize("DetectorClass", DETECTORS)
    def test_thresholding_creates_binary_mask(self, DetectorClass, plate_12hr_grid_image):
        """Test that _thresholding creates a valid binary mask."""
        image = plate_12hr_grid_image
        detector = DetectorClass()

        matrix = image.detect_mat[:]
        binary_mask = detector._thresholding(matrix)

        assert binary_mask.dtype == bool or binary_mask.dtype == np.bool_
        assert binary_mask.shape == matrix.shape
        assert np.all((binary_mask == 0) | (binary_mask == 1))

    @timeit
    @pytest.mark.parametrize("DetectorClass", DETECTORS)
    def test_thresholding_adaptive_kernel(self, DetectorClass, plate_12hr_grid_image):
        """Test _thresholding with adaptive kernel sizing via nrows/ncols."""
        image = plate_12hr_grid_image
        detector = DetectorClass()

        matrix = image.detect_mat[:]
        binary_mask = detector._thresholding(matrix, nrows=8, ncols=12)

        assert binary_mask.dtype == bool or binary_mask.dtype == np.bool_
        assert binary_mask.shape == matrix.shape
        assert np.any(binary_mask)


class TestEdgeRefinementShared:
    """Edge refinement tests parametrized across detectors."""

    @timeit
    @pytest.mark.parametrize("DetectorClass", DETECTORS)
    def test_refine_edges_maintains_count(self, DetectorClass):
        """Test that _refine_edges maintains the number of edges."""
        detector = DetectorClass()
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[10:20, :] = True

        initial_edges = np.array([0, 25, 50, 75, 100])
        refined_edges = detector._refine_edges(binary_image, initial_edges, axis=0)

        assert len(refined_edges) == len(initial_edges)
        assert refined_edges[0] == 0
        assert refined_edges[-1] == 100
        assert np.all(np.diff(refined_edges) >= 0)

    @timeit
    @pytest.mark.parametrize("DetectorClass", DETECTORS)
    def test_estimate_edges_returns_correct_number(self, DetectorClass):
        """Test that _estimate_edges returns n_bins+1 edges."""
        detector = DetectorClass()
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[10:20, :] = True
        binary_image[30:40, :] = True
        binary_image[50:60, :] = True

        n_bins = 8
        edges = detector._estimate_edges(binary_image, axis=0, n_bins=n_bins)

        assert len(edges) == n_bins + 1
        assert edges[0] == 0
        assert edges[-1] == binary_image.shape[1]
        assert np.all(np.diff(edges) >= 0)
