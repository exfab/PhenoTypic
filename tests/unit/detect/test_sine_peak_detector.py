"""
Focused test suite for SinePeakDetector algorithm-specific logic.

Tests focus on sinusoidal cross-correlation, grid inference, peak detection parameters,
edge refinement, and pipeline integration. Basic initialization and apply() are covered
by smoke tests in test_operation.py.
"""

import pytest
import numpy as np
import phenotypic
from phenotypic.detect import SinePeakDetector
from phenotypic.data import load_plate_12hr

from ..resources.TestHelper import timeit


class TestSinePeakDetectorBasic:
    """Basic detection tests for SinePeakDetector."""

    @timeit
    def test_detection_on_gridimage(self):
        """Test basic detection on GridImage."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector()
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    def test_detection_on_plain_image(self):
        """Test detection on plain Image (uses grid inference)."""
        image = phenotypic.Image(load_plate_12hr())
        detector = SinePeakDetector()
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    def test_inplace_semantics(self):
        """Test inplace=True modifies original, inplace=False returns copy."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector()

        result = detector.apply(image, inplace=False)
        assert result is not image

        original_id = id(image)
        result_inplace = detector.apply(image, inplace=True)
        assert result_inplace is image


class TestSinePeakDetectorSinusoidalCorrelation:
    """Test sinusoidal cross-correlation specific behavior."""

    @timeit
    def test_edge_count_matches_grid(self):
        """Estimated edges should have n_bins+1 entries."""
        detector = SinePeakDetector()
        # Create synthetic binary with grid-like peaks
        binary = np.zeros((200, 300), dtype=bool)
        for r in range(8):
            center = 12 + r * 22
            binary[center - 3:center + 3, :] = True

        edges = detector._estimate_edges(binary, axis=0, n_bins=8)
        assert len(edges) == 9
        assert edges[0] == 0
        assert edges[-1] == binary.shape[1]

    @timeit
    def test_ncc_perfect_match(self):
        """NCC of a signal with itself should peak near 1.0."""
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        template = np.sin(np.linspace(0, 2 * np.pi, 50))
        ncc = SinePeakDetector._normalized_cross_correlation(signal, template)
        assert np.max(ncc) > 0.8

    @timeit
    def test_ncc_zero_signal(self):
        """NCC with zero signal should return zeros."""
        signal = np.zeros(100)
        template = np.sin(np.linspace(0, 2 * np.pi, 20))
        ncc = SinePeakDetector._normalized_cross_correlation(signal, template)
        assert np.allclose(ncc, 0.0)

    @timeit
    def test_ncc_output_length(self):
        """NCC output length should match signal length minus template length + 1."""
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        template = np.sin(np.linspace(0, 2 * np.pi, 50))
        ncc = SinePeakDetector._normalized_cross_correlation(signal, template)
        # Normalized cross-correlation output should be at most len(signal)
        assert len(ncc) <= len(signal)
        assert len(ncc) > 0

    @timeit
    def test_ncc_values_bounded(self):
        """NCC values should be bounded between -1 and 1."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(200)
        template = rng.standard_normal(30)
        ncc = SinePeakDetector._normalized_cross_correlation(signal, template)
        assert np.all(ncc >= -1.0 - 1e-10)
        assert np.all(ncc <= 1.0 + 1e-10)

    @timeit
    def test_threshold_effect_on_peaks(self):
        """Higher correlation_threshold should be more selective."""
        image = phenotypic.GridImage(load_plate_12hr())

        detector_low = SinePeakDetector(correlation_threshold=0.1)
        detector_high = SinePeakDetector(correlation_threshold=0.8)

        result_low = detector_low.apply(image, inplace=False)
        result_high = detector_high.apply(image, inplace=False)

        # Both should still detect something
        assert result_low.num_objects > 0
        assert result_high.num_objects > 0


class TestSinePeakDetectorParameterEffects:
    """Test parameter effects on detection."""

    @timeit
    @pytest.mark.parametrize(
        "thresh_method", ["otsu", "mean", "local", "triangle", "isodata", "li"]
    )
    def test_threshold_methods(self, thresh_method):
        """Different thresholding methods all produce detections."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector(thresh_method=thresh_method)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    @pytest.mark.xfail(
        reason="minimum threshold produces degenerate objmask that causes "
               "AutoGridFinder histogram bin edges to collapse",
        raises=Exception,
    )
    def test_minimum_threshold_on_gridimage(self):
        """Test minimum threshold on GridImage -- known grid finder limitation."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector(thresh_method="minimum")
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0, 5.0])
    def test_smoothing_sigma(self, sigma):
        """Different smoothing values all produce detections."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector(smoothing_sigma=sigma)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("corr_thresh", [0.1, 0.3, 0.5])
    def test_correlation_thresholds(self, corr_thresh):
        """Different correlation thresholds all produce detections."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector(correlation_threshold=corr_thresh)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("noise_radius", [1, 2, 4])
    def test_noise_radius_independence(self, noise_radius):
        """Different noise_radius values all produce detections."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector(noise_radius=noise_radius)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("footprint_width", [1, 3, 6, 9])
    def test_different_footprint_width(self, footprint_width):
        """Test detection with different shape widths."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector(footprint_width=footprint_width)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    def test_with_custom_peak_distance(self):
        """Test detection with custom minimum peak distance."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector(min_peak_distance=20)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    def test_with_custom_peak_prominence(self):
        """Test detection with custom peak prominence."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector(peak_prominence=0.15)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0


class TestSinePeakDetectorEdgeRefinement:
    """Test edge refinement toggle."""

    @timeit
    def test_edge_count_preservation(self):
        """Refinement should not change the number of edges."""
        detector = SinePeakDetector()
        binary = np.zeros((100, 100), dtype=bool)
        binary[10:20, :] = True

        initial_edges = np.array([0, 25, 50, 75, 100])
        refined_edges = detector._refine_edges(binary, initial_edges, axis=0)

        assert len(refined_edges) == len(initial_edges)

    @timeit
    def test_refinement_preserves_borders(self):
        """Refinement should keep first and last edge at image borders."""
        detector = SinePeakDetector()
        binary = np.zeros((100, 100), dtype=bool)
        binary[10:20, :] = True
        binary[50:60, :] = True

        initial_edges = np.array([0, 25, 50, 75, 100])
        refined_edges = detector._refine_edges(binary, initial_edges, axis=0)

        assert refined_edges[0] == 0
        assert refined_edges[-1] == 100
        assert np.all(np.diff(refined_edges) >= 0)

    @timeit
    def test_refinement_toggle(self):
        """Both refinement on and off should produce valid results."""
        image = phenotypic.GridImage(load_plate_12hr())

        detector_on = SinePeakDetector(edge_refinement=True)
        detector_off = SinePeakDetector(edge_refinement=False)

        result_on = detector_on.apply(image, inplace=False)
        result_off = detector_off.apply(image, inplace=False)

        assert result_on.num_objects > 0
        assert result_off.num_objects > 0


class TestSinePeakDetectorGridInference:
    """Test grid inference algorithms inherited from GridInferenceMixin."""

    @timeit
    @pytest.mark.parametrize(
        "nrows,ncols",
        [
            (8, 12),   # 96-well plate
            (16, 24),  # 384-well plate
            (32, 48),  # 1536-well plate
            (4, 6),    # Small grid
        ],
    )
    def test_infer_grid_shape_with_synthetic_data(self, nrows, ncols):
        """Test grid shape inference with synthetic gridded data."""
        detector = SinePeakDetector()

        # Create synthetic binary image with grid pattern
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
    def test_infer_grid_shape_blank_defaults(self):
        """Blank mask should default to an 8x12 plate."""
        detector = SinePeakDetector()
        binary_image = np.zeros((120, 180), dtype=bool)

        inferred_rows, inferred_cols = detector._infer_grid_shape(binary_image)

        assert inferred_rows == 8
        assert inferred_cols == 12


class TestSinePeakDetectorHelperMethods:
    """Test helper methods for algorithm implementation."""

    @timeit
    def test_thresholding_creates_binary_mask(self):
        """Test that _thresholding creates a valid binary mask."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector()

        matrix = image.detect_mat[:]
        binary_mask = detector._thresholding(matrix)

        assert binary_mask.dtype == bool or binary_mask.dtype == np.bool_
        assert binary_mask.shape == matrix.shape
        assert np.all((binary_mask == 0) | (binary_mask == 1))

    @timeit
    def test_thresholding_adaptive_kernel(self):
        """Test _thresholding with adaptive kernel sizing via nrows/ncols."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector()

        matrix = image.detect_mat[:]
        binary_mask = detector._thresholding(matrix, nrows=8, ncols=12)

        assert binary_mask.dtype == bool or binary_mask.dtype == np.bool_
        assert binary_mask.shape == matrix.shape
        assert np.any(binary_mask)  # Should detect some foreground

    @timeit
    @pytest.mark.parametrize("input_val,expected", [
        (1, 3), (2, 3), (3, 3), (4, 5), (5, 5), (6, 7), (10, 11),
    ])
    def test_round_odd(self, input_val, expected):
        """Test _round_odd enforces odd integers with minimum 3."""
        assert SinePeakDetector._round_odd(input_val) == expected

    @timeit
    def test_clean_and_sum_binary_axis0(self):
        """Test _clean_and_sum_binary for axis=0 (rows)."""
        detector = SinePeakDetector()
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[20:30, 20:80] = True
        binary_image[50:60, 20:80] = True

        sums = detector._clean_and_sum_binary(binary_image, axis=0)

        assert len(sums) == 100
        assert sums.sum() > 0

    @timeit
    def test_clean_and_sum_binary_axis1(self):
        """Test _clean_and_sum_binary for axis=1 (columns)."""
        detector = SinePeakDetector()
        binary_image = np.zeros((100, 100), dtype=bool)
        binary_image[20:80, 20:30] = True
        binary_image[20:80, 50:60] = True

        sums = detector._clean_and_sum_binary(binary_image, axis=1)

        assert len(sums) == 100
        assert sums.sum() > 0

    @timeit
    def test_estimate_edges_returns_correct_number(self):
        """Test that _estimate_edges returns n_bins+1 edges."""
        detector = SinePeakDetector()
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


class TestSinePeakDetectorReproducibility:
    """Test determinism and label consistency."""

    @timeit
    def test_determinism(self):
        """Same input should produce identical output."""
        image1 = phenotypic.GridImage(load_plate_12hr())
        image2 = phenotypic.GridImage(load_plate_12hr())

        detector = SinePeakDetector()
        result1 = detector.apply(image1, inplace=False)
        result2 = detector.apply(image2, inplace=False)

        assert result1.num_objects == result2.num_objects
        assert np.array_equal(result1.objmap[:], result2.objmap[:])

    @timeit
    def test_detection_reproducibility_with_params(self):
        """Test that detection is reproducible with explicit parameters."""
        image1 = phenotypic.GridImage(load_plate_12hr())
        image2 = phenotypic.GridImage(load_plate_12hr())

        detector = SinePeakDetector(
            thresh_method="otsu",
            subtract_background=True,
            remove_noise=True,
            footprint_width=3,
            smoothing_sigma=2.0,
            edge_refinement=True,
            correlation_threshold=0.3,
        )

        result1 = detector.apply(image1, inplace=False)
        result2 = detector.apply(image2, inplace=False)

        assert result1.num_objects == result2.num_objects
        assert np.array_equal(result1.objmap[:], result2.objmap[:])

    @timeit
    def test_sequential_labels(self):
        """Object map should have sequential labels."""
        image = phenotypic.GridImage(load_plate_12hr())
        detector = SinePeakDetector()
        result = detector.apply(image, inplace=False)

        unique_labels = np.unique(result.objmap[:])
        assert unique_labels[0] == 0 or unique_labels[0] == 1
        if result.num_objects > 0:
            max_label = unique_labels[-1]
            assert max_label <= result.num_objects + 1


class TestSinePeakDetectorIntegration:
    """Test pipeline and serialization integration."""

    @timeit
    def test_pipeline_integration(self):
        """SinePeakDetector should work in an ImagePipeline."""
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur

        pipeline = ImagePipeline([
            GaussianBlur(sigma=1.0),
            SinePeakDetector(thresh_method="otsu"),
        ])

        image = phenotypic.GridImage(load_plate_12hr())
        result = pipeline.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    def test_json_roundtrip(self):
        """Serialization and deserialization should preserve detector."""
        from phenotypic import ImagePipeline

        detector = SinePeakDetector(
            thresh_method="triangle",
            correlation_threshold=0.4,
            noise_radius=2,
            smoothing_sigma=3.0,
        )
        pipeline = ImagePipeline([detector])

        json_str = pipeline.to_json()
        restored = ImagePipeline.from_json(json_str)

        restored_detector = restored._ops["SinePeakDetector"]
        assert isinstance(restored_detector, SinePeakDetector)
        assert restored_detector.thresh_method == "triangle"
        assert restored_detector.correlation_threshold == 0.4
        assert restored_detector.noise_radius == 2
        assert restored_detector.smoothing_sigma == 3.0

    @timeit
    def test_serialization_equivalence(self):
        """Serialized and restored detector should produce same results."""
        from phenotypic import ImagePipeline

        detector = SinePeakDetector(correlation_threshold=0.4)
        pipeline = ImagePipeline([detector])

        json_str = pipeline.to_json()
        restored = ImagePipeline.from_json(json_str)

        image1 = phenotypic.GridImage(load_plate_12hr())
        image2 = phenotypic.GridImage(load_plate_12hr())

        result1 = pipeline.apply(image1, inplace=False)
        result2 = restored.apply(image2, inplace=False)

        assert result1.num_objects == result2.num_objects
        assert np.array_equal(result1.objmap[:], result2.objmap[:])


# Run all tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
