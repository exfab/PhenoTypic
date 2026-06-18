"""Shared tests for GridInferenceMixin behavior across detector implementations.

These tests verify mixin methods that are identical for RoundPeaksDetector
and SinePeakDetector, avoiding duplication.
"""

import pytest
import numpy as np
import scipy.ndimage as ndimage
from phenotypic.detect import RoundPeaksDetector, SinePeakDetector
from phenotypic.sdk_.mixin import GridInferenceMixin

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


def _make_disk(center_r: int, center_c: int, radius: int, shape: tuple[int, int]) -> np.ndarray:
    """Create a binary disk mask at the given center."""
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    return ((rr - center_r) ** 2 + (cc - center_c) ** 2) <= radius ** 2


class TestSplitMergedObjects:
    """Tests for _split_merged_objects EDT watershed splitting."""

    @timeit
    def test_no_split_single_cell_object(self):
        """Blob within one cell should remain unchanged."""
        labeled = np.zeros((100, 100), dtype=np.int32)
        labeled[20:30, 20:30] = 1
        row_edges = np.array([0, 50, 100])
        col_edges = np.array([0, 50, 100])

        result = GridInferenceMixin._split_merged_objects(labeled, row_edges, col_edges)

        # Object should still be a single contiguous label
        result_labels = np.unique(result[result != 0])
        assert len(result_labels) == 1

    @timeit
    def test_split_two_merged_blobs(self):
        """Two touching round blobs centered in adjacent cells should split."""
        shape = (100, 200)
        # Two blobs centered at (50, 40) and (50, 160) with overlap near col=100
        blob1 = _make_disk(50, 40, 20, shape)
        blob2 = _make_disk(50, 160, 20, shape)

        # Create a merged object: two blobs connected by a bridge
        merged = blob1 | blob2
        # Add a connecting bridge so they're one connected component
        merged[45:55, 55:145] = True

        labeled, _ = ndimage.label(merged)
        assert labeled.max() == 1  # Confirm they're one object

        row_edges = np.array([0, 100])
        col_edges = np.array([0, 100, 200])

        result = GridInferenceMixin._split_merged_objects(labeled, row_edges, col_edges)

        result_labels = np.unique(result[result != 0])
        assert len(result_labels) >= 2, "Merged blobs spanning two cells should be split"

    @timeit
    def test_no_split_single_large_colony(self):
        """Large blob with one dominant EDT peak should not be split."""
        shape = (100, 200)
        # Single large blob centered at (50, 50) that spills slightly into adjacent cell
        blob = _make_disk(50, 50, 40, shape)

        labeled = np.zeros(shape, dtype=np.int32)
        labeled[blob] = 1

        row_edges = np.array([0, 100])
        col_edges = np.array([0, 80, 200])

        result = GridInferenceMixin._split_merged_objects(labeled, row_edges, col_edges)

        result_labels = np.unique(result[result != 0])
        # Should remain one label (all peaks in same cell or only one peak)
        assert len(result_labels) == 1


class TestComputeObjectCentroids:
    """Tests for _compute_object_centroids."""

    @timeit
    def test_geometric_centroid(self):
        """Single centered blob should have centroid at its center."""
        labeled = np.zeros((100, 100), dtype=np.int32)
        labeled[40:60, 40:60] = 1  # 20x20 block centered at (50, 50)

        centroids = GridInferenceMixin._compute_object_centroids(labeled)

        assert 1 in centroids
        cr, cc = centroids[1]
        assert abs(cr - 49.5) < 1.0  # Center of 40:60 is 49.5
        assert abs(cc - 49.5) < 1.0

    @timeit
    def test_intensity_weighted_centroid(self):
        """Blob with asymmetric intensity should shift centroid toward high-intensity side."""
        labeled = np.zeros((100, 100), dtype=np.int32)
        labeled[40:60, 40:60] = 1

        intensity = np.zeros((100, 100), dtype=np.float64)
        # High intensity on right side of the blob
        intensity[40:60, 50:60] = 10.0
        intensity[40:60, 40:50] = 1.0

        centroids_weighted = GridInferenceMixin._compute_object_centroids(labeled, intensity)
        centroids_geometric = GridInferenceMixin._compute_object_centroids(labeled)

        # Weighted centroid should be shifted right (higher col)
        assert centroids_weighted[1][1] > centroids_geometric[1][1]

    @timeit
    def test_empty_labeled(self):
        """All-zero labeled should return empty dict."""
        labeled = np.zeros((50, 50), dtype=np.int32)

        centroids = GridInferenceMixin._compute_object_centroids(labeled)

        assert centroids == {}


class TestAssignGridObjectsCentroidBased:
    """Tests for centroid-based _assign_grid_objects."""

    @timeit
    def test_whole_object_not_cleaved(self):
        """Object overlapping two cells should be assigned whole to the cell containing its centroid."""
        labeled = np.zeros((100, 200), dtype=np.int32)
        # Object centered at (50, 90) — centroid in left cell — spills into right cell
        labeled[40:60, 80:110] = 1

        row_edges = np.array([0, 100])
        col_edges = np.array([0, 100, 200])

        result = GridInferenceMixin._assign_grid_objects(
            labeled, row_edges, col_edges, "dominant", np.int32, split_merged=False,
        )

        # The entire object should have one label (no boundary cleaving)
        object_pixels = result[labeled == 1]
        assert np.all(object_pixels == object_pixels[0])
        assert object_pixels[0] > 0

        # Pixels in the right cell that were part of the original object should also be labeled
        right_cell_obj_pixels = result[40:60, 100:110]
        assert np.any(right_cell_obj_pixels > 0), "Right-cell spillover pixels should be labeled"

    @timeit
    def test_multiple_objects_same_cell_dominant(self):
        """Two objects with centroids in same cell: dominant picks larger."""
        labeled = np.zeros((100, 100), dtype=np.int32)
        labeled[10:30, 10:30] = 1   # 20x20 = 400 pixels
        labeled[40:50, 40:50] = 2   # 10x10 = 100 pixels

        row_edges = np.array([0, 100])
        col_edges = np.array([0, 100])

        result = GridInferenceMixin._assign_grid_objects(
            labeled, row_edges, col_edges, "dominant", np.int32, split_merged=False,
        )

        # Only the larger object should survive
        assert result.max() == 1
        assert np.all(result[labeled == 1] > 0)
        assert np.all(result[labeled == 2] == 0)

    @timeit
    def test_intensity_none_falls_back_geometric(self):
        """intensity=None should still produce valid assignment using geometric centroids."""
        labeled = np.zeros((100, 200), dtype=np.int32)
        labeled[20:30, 20:30] = 1
        labeled[20:30, 120:130] = 2

        row_edges = np.array([0, 100])
        col_edges = np.array([0, 100, 200])

        result = GridInferenceMixin._assign_grid_objects(
            labeled, row_edges, col_edges, "dominant", np.int32,
            intensity=None, split_merged=False,
        )

        assert result.max() == 2
        assert np.any(result[labeled == 1] > 0)
        assert np.any(result[labeled == 2] > 0)

    @timeit
    def test_split_merged_true_splits_touching_colonies(self):
        """Two touching round blobs should be split then assigned to correct cells."""
        shape = (100, 200)
        blob1 = _make_disk(50, 40, 20, shape)
        blob2 = _make_disk(50, 160, 20, shape)
        merged = blob1 | blob2
        merged[45:55, 55:145] = True  # Bridge

        labeled, _ = ndimage.label(merged)
        assert labeled.max() == 1

        row_edges = np.array([0, 100])
        col_edges = np.array([0, 100, 200])

        result = GridInferenceMixin._assign_grid_objects(
            labeled, row_edges, col_edges, "dominant", np.int32, split_merged=True,
        )

        # Should have assigned objects to both cells
        assert result.max() >= 2, "Split + assign should produce objects in both cells"

    @timeit
    def test_split_merged_false_skips_splitting(self):
        """split_merged=False should keep merged blob as one object."""
        shape = (100, 200)
        blob1 = _make_disk(50, 40, 20, shape)
        blob2 = _make_disk(50, 160, 20, shape)
        merged = blob1 | blob2
        merged[45:55, 55:145] = True

        labeled, _ = ndimage.label(merged)
        assert labeled.max() == 1

        row_edges = np.array([0, 100])
        col_edges = np.array([0, 100, 200])

        result = GridInferenceMixin._assign_grid_objects(
            labeled, row_edges, col_edges, "dominant", np.int32, split_merged=False,
        )

        # Merged blob assigned as one object (centroid-based, whole object)
        assert result.max() == 1


class TestSinePeakDetectorSelectionMode:
    """Tests for SinePeakDetector selection_mode parameter."""

    @timeit
    def test_selection_mode_parameter_accepted(self):
        """Verify SinePeakDetector(selection_mode='centered') instantiates."""
        detector = SinePeakDetector(selection_mode="centered")
        assert detector.selection_mode == "centered"

    @timeit
    def test_split_merged_parameter_accepted(self):
        """Verify SinePeakDetector(split_merged=False) instantiates."""
        detector = SinePeakDetector(split_merged=False)
        assert detector.split_merged is False

    @timeit
    def test_default_selection_mode_is_dominant(self):
        """Default selection_mode should be 'dominant'."""
        detector = SinePeakDetector()
        assert detector.selection_mode == "dominant"
        assert detector.split_merged is True
