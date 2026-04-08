"""Tests for MadHysteresisDetector.

Covers MAD-based noise estimation, hysteresis thresholding, parameter effects,
edge cases, reproducibility, pipeline integration, and serialization.
"""

import pytest
import numpy as np

from phenotypic import Image, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import MadHysteresisDetector


def _make_response_image(seed=42):
    """Create a synthetic filter-response-like image with Gaussian noise + signal.

    Returns an Image whose detect_mat resembles a filter output (e.g. CED/LoG):
    background noise near zero with bright signal blobs.
    """
    rng = np.random.default_rng(seed)
    # Background: low-level Gaussian noise
    arr = np.abs(rng.normal(0, 5, (200, 200))).astype(np.float32)
    # Signal blobs well above noise floor
    arr[30:60, 30:60] += 120   # blob 1 (interior)
    arr[80:100, 80:100] += 150  # blob 2 (interior)
    arr[140:170, 130:170] += 100  # blob 3 (interior)
    return Image(arr.astype(np.uint8))


class TestMadHysteresisDetectorBasic:
    """Basic detection behavior."""

    def test_basic_detection_on_response_image(self):
        """Applying to a response-like image produces a non-empty mask."""
        image = _make_response_image()
        detector = MadHysteresisDetector()
        result = detector.apply(image, inplace=False)
        mask = result.objmask[:]

        assert mask.shape == image.detect_mat[:].shape
        assert mask.any(), "Expected non-empty mask on response image"

    def test_basic_detection_on_synth_plate(self):
        """Applying to synth plate runs without error."""
        image = Image(load_synth_yeast_plate())
        detector = MadHysteresisDetector()
        result = detector.apply(image, inplace=False)

        # Synth plate is raw intensity (not a filter response), so mask may be
        # empty after clear_border. Just verify it ran and produced a valid mask.
        mask = result.objmask[:]
        assert mask.shape == image.detect_mat[:].shape

    def test_inplace_false_preserves_original(self):
        """inplace=False should not modify the original image."""
        image = _make_response_image()
        result = MadHysteresisDetector().apply(image, inplace=False)

        assert result is not image

    def test_inplace_true_modifies_original(self):
        """inplace=True should modify the original image."""
        image = _make_response_image()
        result = MadHysteresisDetector().apply(image, inplace=True)

        assert result is image
        assert result.objmask[:].any()


class TestMadHysteresisDetectorParameterEffects:
    """Parameter sensitivity and effects."""

    def test_higher_k_high_reduces_detections(self):
        """Increasing k_high should reduce or maintain detected pixels."""
        image_low = _make_response_image()
        image_high = _make_response_image()

        result_low = MadHysteresisDetector(k_high=3.0, k_low=1.5).apply(
            image_low, inplace=False
        )
        result_high = MadHysteresisDetector(k_high=8.0, k_low=1.5).apply(
            image_high, inplace=False
        )

        pixels_low = result_low.objmask[:].sum()
        pixels_high = result_high.objmask[:].sum()

        assert pixels_low >= pixels_high, (
            f"Higher k_high should detect fewer pixels: {pixels_low} vs {pixels_high}"
        )

    def test_min_size_filters_small_components(self):
        """Larger min_size should remove small components."""
        image_small = _make_response_image()
        image_large = _make_response_image()

        result_small = MadHysteresisDetector(min_size=1).apply(
            image_small, inplace=False
        )
        result_large = MadHysteresisDetector(min_size=500).apply(
            image_large, inplace=False
        )

        pixels_small = result_small.objmask[:].sum()
        pixels_large = result_large.objmask[:].sum()

        assert pixels_small >= pixels_large

    def test_ignore_borders_clears_edge_objects(self):
        """ignore_borders=True should remove objects touching the image edge."""
        # Create image with object touching the border
        rng = np.random.default_rng(42)
        arr = np.abs(rng.normal(0, 3, (100, 100))).astype(np.float32)
        # Bright region touching left border
        arr[20:80, 0:30] += 200
        # Bright region in center (not touching border)
        arr[40:60, 50:70] += 200

        image_borders = Image(arr.astype(np.uint8))
        image_no_borders = Image(arr.astype(np.uint8))

        result_borders = MadHysteresisDetector(
            ignore_borders=True, k_high=3.0, k_low=1.5
        ).apply(image_borders, inplace=False)
        result_no_borders = MadHysteresisDetector(
            ignore_borders=False, k_high=3.0, k_low=1.5
        ).apply(image_no_borders, inplace=False)

        # With ignore_borders, fewer pixels should be detected
        assert result_no_borders.objmask[:].sum() >= result_borders.objmask[:].sum()


class TestMadHysteresisDetectorEdgeCases:
    """Edge cases and error handling."""

    def test_k_low_ge_k_high_raises(self):
        """k_low >= k_high should raise an error."""
        image = _make_response_image()

        with pytest.raises(Exception, match="k_low.*must be less than k_high"):
            MadHysteresisDetector(k_high=3.0, k_low=3.0).apply(image)

        with pytest.raises(Exception, match="k_low.*must be less than k_high"):
            MadHysteresisDetector(k_high=3.0, k_low=5.0).apply(image)

    def test_uniform_image_returns_empty_mask(self):
        """Uniform image (zero sigma_noise) should return empty mask."""
        arr = np.full((100, 100), 128, dtype=np.uint8)
        image = Image(arr)
        result = MadHysteresisDetector(ignore_zeros=False).apply(image, inplace=False)

        assert not result.objmask[:].any(), "Uniform image should produce empty mask"

    def test_all_zeros_with_ignore_zeros_returns_empty_mask(self):
        """All-zero image with ignore_zeros=True should return empty mask."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        image = Image(arr)
        result = MadHysteresisDetector(ignore_zeros=True).apply(image, inplace=False)

        assert not result.objmask[:].any(), "All-zero image should produce empty mask"


class TestMadHysteresisDetectorReproducibility:
    """Determinism and reproducibility."""

    def test_deterministic_output(self):
        """Same input should produce identical output across runs."""
        image1 = _make_response_image(seed=99)
        image2 = _make_response_image(seed=99)
        detector = MadHysteresisDetector(k_high=5.0, k_low=2.5, min_size=20)

        result1 = detector.apply(image1, inplace=False)
        result2 = detector.apply(image2, inplace=False)

        np.testing.assert_array_equal(result1.objmask[:], result2.objmask[:])


class TestMadHysteresisDetectorIntegration:
    """Pipeline integration and serialization."""

    def test_pipeline_integration(self):
        """MadHysteresisDetector works within an ImagePipeline."""
        pipeline = ImagePipeline([
            MadHysteresisDetector(k_high=5.0, k_low=2.5),
        ])
        image = _make_response_image()
        result = pipeline.apply(image, inplace=False)

        assert result.objmask[:].any()

    def test_json_serialization_roundtrip(self):
        """Serialization to JSON and back preserves parameters."""
        original = MadHysteresisDetector(
            k_high=4.0, k_low=2.0, min_size=50,
            connectivity=1, ignore_zeros=False, ignore_borders=False,
        )
        pipeline = ImagePipeline([original])
        json_str = pipeline.to_json()
        restored_pipeline = ImagePipeline.from_json(json_str)

        restored = list(restored_pipeline._ops.values())[0]
        assert isinstance(restored, MadHysteresisDetector)
        assert restored.k_high == original.k_high
        assert restored.k_low == original.k_low
        assert restored.min_size == original.min_size
        assert restored.connectivity == original.connectivity
        assert restored.ignore_zeros == original.ignore_zeros
        assert restored.ignore_borders == original.ignore_borders

    def test_serialization_functional_equivalence(self):
        """Serialized and restored detector produces identical results."""
        original = MadHysteresisDetector(k_high=4.0, k_low=2.0, min_size=30)
        pipeline = ImagePipeline([original])
        restored_pipeline = ImagePipeline.from_json(pipeline.to_json())

        image1 = _make_response_image()
        image2 = _make_response_image()

        result1 = pipeline.apply(image1, inplace=False)
        result2 = restored_pipeline.apply(image2, inplace=False)

        np.testing.assert_array_equal(result1.objmask[:], result2.objmask[:])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
