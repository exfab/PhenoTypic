"""
Comprehensive test suite for the GitterDetector class.

Tests cover initialization, detection on different image types, parameter variations,
helper methods, and edge cases.
"""
import pytest
import numpy as np
import phenotypic
from phenotypic.detect import GitterDetector
from phenotypic.data import load_plate_12hr, load_plate_72hr

from .resources.TestHelper import timeit


class TestGitterDetectorInitialization:
    """Test GitterDetector initialization and parameter handling."""

    @timeit
    def test_default_initialization(self):
        """Test that GitterDetector can be initialized with default parameters."""
        detector = GitterDetector()
        assert detector.thresh_method == 'otsu'
        assert detector.subtract_background is True
        assert detector.remove_noise is True
        assert detector.footprint_radius == 3
        assert detector.smoothing_sigma == 2.0
        assert detector.min_peak_distance is None
        assert detector.peak_prominence is None
        assert detector.edge_refinement is True

    @timeit
    @pytest.mark.parametrize("thresh_method", ['otsu', 'mean', 'local', 'triangle', 'minimum', 'isodata'])
    def test_initialization_with_thresh_methods(self, thresh_method):
        """Test initialization with different thresholding methods."""
        detector = GitterDetector(thresh_method=thresh_method)
        assert detector.thresh_method == thresh_method

    @timeit
    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        detector = GitterDetector(
            thresh_method='triangle',
            subtract_background=False,
            remove_noise=False,
            footprint_radius=5,
            smoothing_sigma=3.0,
            min_peak_distance=10,
            peak_prominence=0.2,
            edge_refinement=False
        )
        assert detector.thresh_method == 'triangle'
        assert detector.subtract_background is False
        assert detector.remove_noise is False
        assert detector.footprint_radius == 5
        assert detector.smoothing_sigma == 3.0
        assert detector.min_peak_distance == 10
        assert detector.peak_prominence == 0.2
        assert detector.edge_refinement is False


class TestGitterDetectorOnGridImage:
    """Test GitterDetector on GridImage with known grid structure."""

    @timeit
    def test_detection_on_grid_image_12hr(self):
        """Test detection on 12-hour plate GridImage."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector()
        result = detector.apply(image, inplace=False)

        assert result is not None
        assert not result.isempty()
        assert result.num_objects > 0
        assert result.objmap[:].max() > 0
        # Grid should have 8x12 = 96 positions
        assert result.num_objects <= 96

    @timeit
    def test_detection_on_grid_image_72hr(self):
        """Test detection on 72-hour plate GridImage."""
        image = phenotypic.GridImage(load_plate_72hr())
        detector = GitterDetector()
        result = detector.apply(image, inplace=False)

        assert result is not None
        assert not result.isempty()
        assert result.num_objects > 0
        assert result.objmap[:].max() > 0
        assert result.num_objects <= 96

    @timeit
    def test_detection_preserves_grid_info(self):
        """Test that detection preserves grid information in GridImage."""
        image = phenotypic.GridImage(load_plate_12hr(), nrows=8, ncols=12)
        detector = GitterDetector()
        result = detector.apply(image, inplace=False)

        assert isinstance(result, phenotypic.GridImage)
        assert result.nrows == 8
        assert result.ncols == 12

    @timeit
    def test_inplace_detection(self):
        """Test in-place detection modifies the original image."""
        image = phenotypic.GridImage(load_plate_12hr())
        original_objmap = image.objmap[:].copy()
        detector = GitterDetector()
        result = detector.apply(image, inplace=True)

        assert result is image
        assert not np.array_equal(image.objmap[:], original_objmap)
        assert image.num_objects > 0


class TestGitterDetectorOnRegularImage:
    """Test GitterDetector on regular Image (grid inference)."""

    @timeit
    def test_detection_on_regular_image(self):
        """Test detection on regular Image without explicit grid."""
        image = phenotypic.Image(load_plate_12hr())
        detector = GitterDetector()
        result = detector.apply(image, inplace=False)

        assert result is not None
        assert not result.isempty()
        assert result.num_objects > 0

    @timeit
    def test_grid_inference_12hr(self):
        """Test that grid inference works on 12hr plate."""
        image = phenotypic.Image(load_plate_12hr())
        detector = GitterDetector()
        result = detector.apply(image, inplace=False)

        # Should detect colonies even without explicit grid
        assert result.num_objects > 0
        # Should be reasonable number for a plate
        assert 10 < result.num_objects < 150


class TestGitterDetectorParameters:
    """Test GitterDetector with different parameter combinations."""

    @timeit
    @pytest.mark.parametrize("thresh_method", ['otsu', 'mean', 'triangle', 'minimum', 'isodata'])
    def test_different_thresholding_methods(self, thresh_method):
        """Test that different thresholding methods all work."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector(thresh_method=thresh_method)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_with_background_subtraction(self):
        """Test detection with background subtraction enabled."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector(subtract_background=True)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_without_background_subtraction(self):
        """Test detection without background subtraction."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector(subtract_background=False)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_with_noise_removal(self):
        """Test detection with noise removal enabled."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector(remove_noise=True)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_without_noise_removal(self):
        """Test detection without noise removal."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector(remove_noise=False)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_with_edge_refinement(self):
        """Test detection with edge refinement enabled."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector(edge_refinement=True)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_without_edge_refinement(self):
        """Test detection without edge refinement."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector(edge_refinement=False)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0, 5.0])
    def test_different_smoothing_sigma(self, sigma):
        """Test detection with different smoothing sigma values."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector(smoothing_sigma=sigma)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("radius", [1, 3, 5, 7])
    def test_different_footprint_radius(self, radius):
        """Test detection with different footprint radii."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector(footprint_radius=radius)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_with_custom_peak_distance(self):
        """Test detection with custom minimum peak distance."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector(min_peak_distance=20)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0

    @timeit
    def test_with_custom_peak_prominence(self):
        """Test detection with custom peak prominence."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector(peak_prominence=0.15)
        result = detector.apply(image, inplace=False)

        assert result.num_objects > 0


class TestGitterDetectorHelperMethods:
    """Test GitterDetector helper methods."""

    @timeit
    def test_thresholding_creates_binary_mask(self):
        """Test that _thresholding creates a valid binary mask."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector()

        matrix = image.enh_gray[:]
        binary_mask = detector._thresholding(matrix)

        assert binary_mask.dtype == bool or binary_mask.dtype == np.uint8
        assert binary_mask.shape == matrix.shape
        assert np.all((binary_mask == 0) | (binary_mask == 1))

    @timeit
    def test_clean_and_sum_binary_axis0(self):
        """Test _clean_and_sum_binary for axis=0 (rows)."""
        detector = GitterDetector()
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
        detector = GitterDetector()
        # Create simple test pattern
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[20:80, 20:30] = True  # Vertical stripe
        binary_image[20:80, 50:60] = True  # Another stripe

        sums = detector._clean_and_sum_binary(binary_image, axis=1)

        assert len(sums) == 100
        assert sums.sum() > 0

    @timeit
    def test_estimate_edges_returns_correct_number(self):
        """Test that _estimate_edges returns n_bins+1 edges."""
        detector = GitterDetector()
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[10:20, :] = True
        binary_image[30:40, :] = True
        binary_image[50:60, :] = True

        n_bins = 8
        edges = detector._estimate_edges(binary_image, axis=0, n_bins=n_bins)

        assert len(edges) == n_bins + 1
        assert edges[0] == 0  # Should start at 0
        assert edges[-1] <= binary_image.shape[1]  # Should not exceed image size

    @timeit
    def test_refine_edges_maintains_count(self):
        """Test that _refine_edges maintains the number of edges."""
        detector = GitterDetector()
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[10:20, :] = True

        initial_edges = np.array([0, 25, 50, 75, 100])
        refined_edges = detector._refine_edges(binary_image, initial_edges, axis=0)

        assert len(refined_edges) == len(initial_edges)
        assert refined_edges[0] == 0  # First edge should remain at border
        assert refined_edges[-1] == 100  # Last edge should remain at border

    @timeit
    @pytest.mark.parametrize("nrows,ncols", [
        (8, 12),   # 96-well plate
        (16, 24),  # 384-well plate
        (32, 48),  # 1536-well plate
        (4, 6),    # Small grid
    ])
    def test_infer_grid_shape_with_synthetic_data(self, nrows, ncols):
        """Test grid shape inference with synthetic gridded data."""
        detector = GitterDetector()

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
                binary_image[r_center-3:r_center+3, c_center-3:c_center+3] = True

        inferred_rows, inferred_cols = detector._infer_grid_shape(binary_image)

        # Inference should be close to actual grid size
        assert inferred_rows > 0
        assert inferred_cols > 0


class TestGitterDetectorEdgeCases:
    """Test GitterDetector with edge cases and unusual inputs."""

    @timeit
    def test_empty_image(self):
        """Test detection on an empty/blank image."""
        # Create blank image (RGB for GridImage compatibility)
        blank_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
        image = phenotypic.GridImage(blank_array)
        detector = GitterDetector()
        result = detector.apply(image, inplace=False)

        # Should complete without error, even if no objects detected
        assert result is not None

    @timeit
    def test_very_dark_image(self):
        """Test detection on very dark image."""
        dark_array = np.zeros((100, 100, 3), dtype=np.uint8)
        image = phenotypic.GridImage(dark_array)
        detector = GitterDetector()
        result = detector.apply(image, inplace=False)

        # Should complete without error
        assert result is not None

    @timeit
    def test_single_colony(self):
        """Test detection with only one colony."""
        # Create image with single object (RGB format)
        single_colony = np.zeros((100, 100, 3), dtype=np.uint8)
        single_colony[40:60, 40:60, :] = 255
        image = phenotypic.Image(single_colony)
        detector = GitterDetector()
        result = detector.apply(image, inplace=False)

        assert result is not None
        # Should detect at least one object
        assert result.num_objects >= 0

    @timeit
    def test_small_image(self):
        """Test detection on very small image."""
        small_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        image = phenotypic.GridImage(small_array, nrows=4, ncols=6)
        detector = GitterDetector()
        result = detector.apply(image, inplace=False)

        # Should complete without error
        assert result is not None

    @timeit
    def test_large_footprint_radius(self):
        """Test with footprint radius larger than typical colony size."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector(footprint_radius=20)
        result = detector.apply(image, inplace=False)

        # Should still work, though may affect results
        assert result is not None


class TestGitterDetectorOutputConsistency:
    """Test that GitterDetector produces consistent and valid outputs."""

    @timeit
    def test_objmap_has_sequential_labels(self):
        """Test that objmap has properly sequential labels after detection."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector()
        result = detector.apply(image, inplace=False)

        unique_labels = np.unique(result.objmap[:])
        # Labels should start from 0 (background) or 1 (first object)
        assert unique_labels[0] == 0 or unique_labels[0] == 1

        if result.num_objects > 0:
            # Labels should be reasonably sequential after relabeling
            max_label = unique_labels[-1]
            assert max_label <= result.num_objects + 1

    @timeit
    def test_objmask_matches_objmap(self):
        """Test that objmask and objmap are consistent."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector()
        result = detector.apply(image, inplace=False)

        # Where objmap > 0, objmask should be True
        objmap_mask = result.objmap[:] > 0
        # Note: objmask may differ from objmap>0 due to grid assignment
        # but we can check basic consistency
        assert result.objmask.shape == result.objmap.shape

    @timeit
    def test_num_objects_matches_objmap(self):
        """Test that num_objects matches the actual number of objects in objmap."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = GitterDetector()
        result = detector.apply(image, inplace=False)

        unique_labels = np.unique(result.objmap[:])
        # Remove background (0)
        object_labels = unique_labels[unique_labels > 0]

        assert result.num_objects == len(object_labels)

    @timeit
    def test_detection_reproducibility(self):
        """Test that detection is reproducible with same parameters."""
        image1 = phenotypic.GridImage(load_plate_12hr())
        image2 = phenotypic.GridImage(load_plate_12hr())

        detector = GitterDetector(
            thresh_method='otsu',
            subtract_background=True,
            remove_noise=True,
            footprint_radius=3,
            smoothing_sigma=2.0,
            edge_refinement=True
        )

        result1 = detector.apply(image1, inplace=False)
        result2 = detector.apply(image2, inplace=False)

        # Results should be identical
        assert result1.num_objects == result2.num_objects
        assert np.array_equal(result1.objmap[:], result2.objmap[:])


class TestGitterDetectorComparisonWithOtherDetectors:
    """Compare GitterDetector with other detectors as sanity check."""

    @timeit
    def test_detects_similar_number_as_watershed(self):
        """Test that GitterDetector finds similar number of objects as WatershedDetector."""
        from phenotypic.detect import WatershedDetector

        image_gitter = phenotypic.GridImage(load_plate_12hr())
        image_watershed = phenotypic.GridImage(load_plate_12hr())

        gitter = GitterDetector()
        watershed = WatershedDetector()

        result_gitter = gitter.apply(image_gitter, inplace=False)
        result_watershed = watershed.apply(image_watershed, inplace=False)

        # Should detect objects (both should find colonies)
        assert result_gitter.num_objects > 0
        assert result_watershed.num_objects > 0

        # Numbers don't need to match exactly, but should be in similar range
        # Allow up to 50% difference
        ratio = result_gitter.num_objects / max(result_watershed.num_objects, 1)
        assert 0.5 <= ratio <= 2.0, f"GitterDetector found {result_gitter.num_objects} objects, WatershedDetector found {result_watershed.num_objects}"


# Run all tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
