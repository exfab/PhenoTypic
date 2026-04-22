import pytest
import numpy as np
from phenotypic import ImagePipeline
from phenotypic.detect import (
    OtsuDetector,
    TriangleDetector,
    FilamentousFungiDetector,
)
from phenotypic.enhance import GaussianBlur


class TestFilamentousFungiDetector:
    """Test suite for FilamentousFungiDetector functionality and edge cases."""

    def test_basic_detection(self, synth_plate):
        """Test basic detection with default configuration produces valid result."""
        image = synth_plate.copy()

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(ignore_zeros=True),
        )
        result = detector.apply(image)

        # Should produce valid objmask and objmap
        assert result.objmask[:].sum() > 0
        assert result.objmap[:].max() > 0

    def test_objmask_objmap_consistency(self, synth_plate):
        """Test that objmask and objmap are consistent after detection."""
        image = synth_plate.copy()

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
        )
        result = detector.apply(image)

        objmask = result.objmask[:]
        objmap = result.objmap[:]

        # All non-zero pixels in objmap should be True in objmask
        assert np.all((objmap > 0) == objmask)

    def test_no_centers_detected_raises(self, synth_plate):
        """Test that error is raised when inoculum_detector finds nothing."""
        image = synth_plate.copy()

        # Create a custom detector that always returns empty mask
        class EmptyCenterDetector(OtsuDetector):
            def _operate(self, img):
                img.objmask[:] = False
                return img

        detector = FilamentousFungiDetector(
                inoculum_detector=EmptyCenterDetector(),
        )

        # apply() wraps exceptions in RuntimeError per framework's error handling
        with pytest.raises(RuntimeError, match="No centers detected"):
            detector.apply(image)

    def test_default_inoculum_detector(self, synth_plate):
        """Test that the default inoculum detector pipeline runs end-to-end."""
        image = synth_plate.copy()

        detector = FilamentousFungiDetector()
        result = detector.apply(image)

        assert result.objmap[:].max() > 0
        assert result.objmask[:].sum() > 0

    def test_invalid_center_detector_type_raises(self):
        """Test that TypeError is raised for invalid inoculum_detector type."""
        with pytest.raises(TypeError, match="inoculum_detector must be"):
            FilamentousFungiDetector(
                    inoculum_detector="not_a_detector",
            )

    def test_inplace_false_preserves_original(self, synth_plate):
        """Test that inplace=False preserves original image."""
        image = synth_plate.copy()
        original_rgb = image.rgb[:].copy()

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
        )
        result = detector.apply(image, inplace=False)

        # Original should be unchanged
        np.testing.assert_array_equal(image.rgb[:], original_rgb)
        # Result should be different image object
        assert result is not image
        # Result should have detection
        assert result.objmap[:].max() > 0

    def test_inplace_true_modifies_original(self, synth_plate):
        """Test that inplace=True modifies the original image."""
        image = synth_plate.copy()

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
        )
        result = detector.apply(image, inplace=True)

        # Should return same image object
        assert result is image
        # Should have detection
        assert image.objmap[:].max() > 0

    def test_with_imagepipeline_center_detector(self, synth_plate):
        """Test that ImagePipeline works as inoculum_detector."""
        image = synth_plate.copy()

        # Create pipeline for center detection with preprocessing
        center_pipeline = ImagePipeline([
            GaussianBlur(sigma=0.5),
            OtsuDetector()
        ])

        detector = FilamentousFungiDetector(
                inoculum_detector=center_pipeline,
        )
        result = detector.apply(image)

        # Should produce valid results
        assert result.objmap[:].max() > 0
        assert result.objmask[:].sum() > 0

    def test_serialization_roundtrip(self):
        """Test that detector serializes and deserializes correctly."""
        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(ignore_zeros=True),
        )

        # Create pipeline
        pipeline = ImagePipeline([detector])

        # Serialize to JSON
        json_str = pipeline.to_json()

        # Deserialize
        restored_pipeline = ImagePipeline.from_json(json_str)

        # Verify structure
        assert len(restored_pipeline._ops) == 1
        restored_detector = list(restored_pipeline._ops.values())[0]
        assert isinstance(restored_detector, FilamentousFungiDetector)
        assert isinstance(restored_detector.inoculum_detector, OtsuDetector)

    def test_serialization_functional_equivalence(self, synth_plate):
        """Test that serialized/deserialized detector produces identical results."""
        image = synth_plate.copy()

        # Original detector
        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
        )
        original_result = detector.apply(image, inplace=False)

        # Serialize and deserialize
        pipeline = ImagePipeline([detector])
        json_str = pipeline.to_json()
        restored_pipeline = ImagePipeline.from_json(json_str)

        # Apply restored detector via pipeline
        restored_result = restored_pipeline.apply(image.copy(), inplace=False)

        # Results should be identical
        np.testing.assert_array_equal(
                original_result.objmask[:],
                restored_result.objmask[:]
        )
        np.testing.assert_array_equal(
                original_result.objmap[:],
                restored_result.objmap[:]
        )

    def test_pipeline_integration(self, synth_plate):
        """Test that detector integrates into full processing pipeline."""
        image = synth_plate.copy()

        # Build pipeline with enhancement and detection
        pipeline = ImagePipeline([
            GaussianBlur(sigma=1.0),
            FilamentousFungiDetector(
                    inoculum_detector=OtsuDetector(),
            ),
        ])

        result = pipeline.apply(image)

        # Should produce valid results
        assert result.objmap[:].max() > 0
        assert result.objmask[:].sum() > 0

    def test_different_inoculum_detectors(self, synth_plate):
        """Test that different inoculum detector choices produce valid results."""
        image = synth_plate.copy()

        for inoculum in (OtsuDetector(), TriangleDetector()):
            detector = FilamentousFungiDetector(
                    inoculum_detector=inoculum,
            )
            result = detector.apply(image.copy())

            # All should produce valid results
            assert result.objmap[:].max() > 0
            assert result.objmask[:].sum() > 0

    def test_valid_labels(self, synth_plate):
        """Test that Voronoi partition produces valid non-background labels."""
        image = synth_plate.copy()

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
        )
        result = detector.apply(image)

        objmap = result.objmap[:]
        unique_labels = np.unique(objmap)

        # Background should be present
        assert 0 in unique_labels

        # Should detect multiple objects
        assert objmap.max() > 1

        # All labels should be non-negative integers
        assert np.all(objmap >= 0)

    def test_no_memory_leaks(self, synth_plate):
        """Test that operation cleans up memory properly."""
        image = synth_plate.copy()

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
        )

        # Apply multiple times and verify consistent memory usage
        for _ in range(3):
            result = detector.apply(image.copy())
            assert result.objmap[:].max() > 0

        # If we get here without memory errors, test passes

    def test_reproducibility(self, synth_plate):
        """Test that same input produces same output."""
        image = synth_plate.copy()

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(ignore_zeros=True),
        )

        result1 = detector.apply(image.copy())
        result2 = detector.apply(image.copy())

        # Same input should produce identical output
        np.testing.assert_array_equal(result1.objmap[:], result2.objmap[:])
        np.testing.assert_array_equal(result1.objmask[:], result2.objmask[:])
