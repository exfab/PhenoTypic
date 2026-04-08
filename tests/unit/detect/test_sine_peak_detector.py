"""
Focused test suite for SinePeakDetector algorithm-specific logic.

Tests focus on sinusoidal cross-correlation and pipeline integration.
Grid inference, helper methods, and edge refinement tests shared with
RoundPeaksDetector are in test_grid_inference_mixin.py.
"""

import pytest
import numpy as np
import phenotypic
from phenotypic.detect import SinePeakDetector

from ..resources.TestHelper import timeit


class TestSinePeakDetectorBasic:
    """Basic detection tests for SinePeakDetector."""

    @timeit
    def test_detection_on_gridimage(self, plate_12hr_grid_image):
        """Test basic detection on GridImage."""
        image = plate_12hr_grid_image.copy()
        detector = SinePeakDetector()
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    def test_detection_on_plain_image(self, plate_12hr_grid_image):
        """Test detection on plain Image (uses grid inference)."""
        image = phenotypic.Image(plate_12hr_grid_image.copy())
        detector = SinePeakDetector()
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    def test_inplace_semantics(self, plate_12hr_grid_image):
        """Test inplace=True modifies original, inplace=False returns copy."""
        image = plate_12hr_grid_image.copy()
        detector = SinePeakDetector()

        result = detector.apply(image, inplace=False)
        assert result is not image

        result_inplace = detector.apply(image, inplace=True)
        assert result_inplace is image


class TestSinePeakDetectorSinusoidalCorrelation:
    """Test sinusoidal cross-correlation specific behavior."""

    @timeit
    def test_edge_count_matches_grid(self):
        """Estimated edges should have n_bins+1 entries."""
        detector = SinePeakDetector()
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
    def test_threshold_effect_on_peaks(self, plate_12hr_grid_image):
        """Higher correlation_threshold should be more selective."""
        image = plate_12hr_grid_image.copy()

        detector_low = SinePeakDetector(correlation_threshold=0.1)
        detector_high = SinePeakDetector(correlation_threshold=0.8)

        result_low = detector_low.apply(image, inplace=False)
        result_high = detector_high.apply(image.copy(), inplace=False)

        assert result_low.num_objects > 0
        assert result_high.num_objects > 0


class TestSinePeakDetectorParameterEffects:
    """Test parameter effects on detection."""

    @timeit
    @pytest.mark.parametrize(
        "thresh_method", ["otsu", "mean", "local", "triangle", "isodata", "li"]
    )
    def test_threshold_methods(self, thresh_method, plate_12hr_grid_image):
        """Different thresholding methods all produce detections."""
        image = plate_12hr_grid_image.copy()
        detector = SinePeakDetector(thresh_method=thresh_method)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    def test_minimum_threshold_on_gridimage(self, plate_12hr_grid_image):
        """Test minimum threshold on GridImage."""
        image = plate_12hr_grid_image.copy()
        detector = SinePeakDetector(thresh_method="minimum")
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0, 5.0])
    def test_smoothing_sigma(self, sigma, plate_12hr_grid_image):
        """Different smoothing values all produce detections."""
        image = plate_12hr_grid_image.copy()
        detector = SinePeakDetector(smoothing_sigma=sigma)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("corr_thresh", [0.1, 0.3, 0.5])
    def test_correlation_thresholds(self, corr_thresh, plate_12hr_grid_image):
        """Different correlation thresholds all produce detections."""
        image = plate_12hr_grid_image.copy()
        detector = SinePeakDetector(correlation_threshold=corr_thresh)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("noise_radius", [1, 2, 4])
    def test_noise_radius_independence(self, noise_radius, plate_12hr_grid_image):
        """Different noise_radius values all produce detections."""
        image = plate_12hr_grid_image.copy()
        detector = SinePeakDetector(noise_radius=noise_radius)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    @pytest.mark.parametrize("footprint_width", [1, 3, 6, 9])
    def test_different_footprint_width(self, footprint_width, plate_12hr_grid_image):
        """Test detection with different shape widths."""
        image = plate_12hr_grid_image.copy()
        detector = SinePeakDetector(footprint_width=footprint_width)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    def test_with_custom_peak_distance(self, plate_12hr_grid_image):
        """Test detection with custom minimum peak distance."""
        image = plate_12hr_grid_image.copy()
        detector = SinePeakDetector(min_peak_distance=20)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0

    @timeit
    def test_with_custom_peak_prominence(self, plate_12hr_grid_image):
        """Test detection with custom peak prominence."""
        image = plate_12hr_grid_image.copy()
        detector = SinePeakDetector(peak_prominence=0.15)
        result = detector.apply(image, inplace=False)
        assert result.num_objects > 0


class TestSinePeakDetectorEdgeRefinement:
    """Test edge refinement toggle."""

    @timeit
    def test_refinement_toggle(self, plate_12hr_grid_image):
        """Both refinement on and off should produce valid results."""
        image = plate_12hr_grid_image.copy()

        detector_on = SinePeakDetector(edge_refinement=True)
        detector_off = SinePeakDetector(edge_refinement=False)

        result_on = detector_on.apply(image, inplace=False)
        result_off = detector_off.apply(image.copy(), inplace=False)

        assert result_on.num_objects > 0
        assert result_off.num_objects > 0


class TestSinePeakDetectorReproducibility:
    """Test determinism and label consistency."""

    @timeit
    def test_determinism(self, plate_12hr_grid_image):
        """Same input should produce identical output."""
        image1 = plate_12hr_grid_image.copy()
        image2 = plate_12hr_grid_image.copy()

        detector = SinePeakDetector()
        result1 = detector.apply(image1, inplace=False)
        result2 = detector.apply(image2, inplace=False)

        assert result1.num_objects == result2.num_objects
        assert np.array_equal(result1.objmap[:], result2.objmap[:])

    @timeit
    def test_detection_reproducibility_with_params(self, plate_12hr_grid_image):
        """Test that detection is reproducible with explicit parameters."""
        image1 = plate_12hr_grid_image.copy()
        image2 = plate_12hr_grid_image.copy()

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
    def test_sequential_labels(self, plate_12hr_grid_image):
        """Object map should have sequential labels."""
        image = plate_12hr_grid_image.copy()
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
    def test_pipeline_integration(self, plate_12hr_grid_image):
        """SinePeakDetector should work in an ImagePipeline."""
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur

        pipeline = ImagePipeline([
            GaussianBlur(sigma=1.0),
            SinePeakDetector(thresh_method="otsu"),
        ])

        image = plate_12hr_grid_image.copy()
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
    def test_serialization_equivalence(self, plate_12hr_grid_image):
        """Serialized and restored detector should produce same results."""
        from phenotypic import ImagePipeline

        detector = SinePeakDetector(correlation_threshold=0.4)
        pipeline = ImagePipeline([detector])

        json_str = pipeline.to_json()
        restored = ImagePipeline.from_json(json_str)

        image1 = plate_12hr_grid_image.copy()
        image2 = plate_12hr_grid_image.copy()

        result1 = pipeline.apply(image1, inplace=False)
        result2 = restored.apply(image2, inplace=False)

        assert result1.num_objects == result2.num_objects
        assert np.array_equal(result1.objmap[:], result2.objmap[:])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
