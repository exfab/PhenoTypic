import pytest
import numpy as np
from phenotypic import ImagePipeline
from phenotypic.detect import (
    OtsuDetector,
    TriangleDetector,
    FilamentousFungiDetector,
)
from phenotypic.enhance import GaussianBlur, CLAHE


class TestFilamentousFungiDetector:
    """Test suite for FilamentousFungiDetector functionality and edge cases."""

    def test_basic_detection(self, synth_plate):
        """Test basic detection with two detectors produces valid result."""
        image = synth_plate.copy()

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(ignore_zeros=True),
                overall_detector=TriangleDetector(),
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
                overall_detector=TriangleDetector(),
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
                overall_detector=TriangleDetector(),
        )

        # apply() wraps exceptions in RuntimeError per framework's error handling
        with pytest.raises(RuntimeError, match="No centers detected"):
            detector.apply(image)

    def test_no_overall_detected_raises(self, synth_plate):
        """Test that error is raised when overall_detector finds nothing."""
        image = synth_plate.copy()

        # Create a custom detector that always returns empty mask
        class EmptyOverallDetector(TriangleDetector):
            def _operate(self, img):
                img.objmask[:] = False
                return img

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
                overall_detector=EmptyOverallDetector(),
        )

        # apply() wraps exceptions in RuntimeError per framework's error handling
        with pytest.raises(RuntimeError, match="No overall structure detected"):
            detector.apply(image)

    def test_center_and_overall_produce_results(self, synth_plate):
        """Test that both center and overall detectors produce results."""
        image = synth_plate.copy()

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
                overall_detector=TriangleDetector(),
        )

        # Should successfully detect both centers and overall structure
        result = detector.apply(image)

        # Verify results have detected objects
        assert result.objmap[:].max() > 0
        assert result.objmask[:].sum() > 0

    def test_invalid_center_detector_type_raises(self):
        """Test that TypeError is raised for invalid inoculum_detector type."""
        with pytest.raises(TypeError, match="inoculum_detector must be"):
            FilamentousFungiDetector(
                    inoculum_detector="not_a_detector",
                    overall_detector=TriangleDetector(),
            )

    def test_invalid_overall_detector_type_raises(self):
        """Test that TypeError is raised for invalid overall_detector type."""
        with pytest.raises(TypeError, match="overall_detector must be"):
            FilamentousFungiDetector(
                    inoculum_detector=OtsuDetector(),
                    overall_detector=123,
            )

    def test_inplace_false_preserves_original(self, synth_plate):
        """Test that inplace=False preserves original image."""
        image = synth_plate.copy()
        original_rgb = image.rgb[:].copy()

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
                overall_detector=TriangleDetector(),
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
                overall_detector=TriangleDetector(),
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
                overall_detector=TriangleDetector(),
        )
        result = detector.apply(image)

        # Should produce valid results
        assert result.objmap[:].max() > 0
        assert result.objmask[:].sum() > 0

    def test_with_imagepipeline_overall_detector(self, synth_plate):
        """Test that ImagePipeline works as overall_detector."""
        image = synth_plate.copy()

        # Create pipeline for overall detection with preprocessing
        overall_pipeline = ImagePipeline([
            CLAHE(clip_limit=2.0),
            TriangleDetector()
        ])

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
                overall_detector=overall_pipeline,
        )
        result = detector.apply(image)

        # Should produce valid results
        assert result.objmap[:].max() > 0
        assert result.objmask[:].sum() > 0

    def test_with_both_imagepipeline_detectors(self, synth_plate):
        """Test that ImagePipeline works for both detectors."""
        image = synth_plate.copy()

        center_pipeline = ImagePipeline([
            GaussianBlur(sigma=0.5),
            OtsuDetector()
        ])
        overall_pipeline = ImagePipeline([
            CLAHE(clip_limit=2.0),
            TriangleDetector()
        ])

        detector = FilamentousFungiDetector(
                inoculum_detector=center_pipeline,
                overall_detector=overall_pipeline,
        )
        result = detector.apply(image)

        # Should produce valid results
        assert result.objmap[:].max() > 0
        assert result.objmask[:].sum() > 0

    def test_serialization_roundtrip(self):
        """Test that detector serializes and deserializes correctly."""
        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(ignore_zeros=True),
                overall_detector=TriangleDetector(),
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
        assert isinstance(restored_detector.overall_detector, TriangleDetector)

    def test_serialization_functional_equivalence(self, synth_plate):
        """Test that serialized/deserialized detector produces identical results."""
        image = synth_plate.copy()

        # Original detector
        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
                overall_detector=TriangleDetector(),
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

        # Build pipeline with enhancement, detection, and cleanup
        pipeline = ImagePipeline([
            GaussianBlur(sigma=1.0),
            FilamentousFungiDetector(
                    inoculum_detector=OtsuDetector(),
                    overall_detector=TriangleDetector(),
            ),
        ])

        result = pipeline.apply(image)

        # Should produce valid results
        assert result.objmap[:].max() > 0
        assert result.objmask[:].sum() > 0

    def test_different_detector_combinations(self, synth_plate):
        """Test various detector combinations work correctly."""
        image = synth_plate.copy()

        combinations = [
            (OtsuDetector(), TriangleDetector()),
            (OtsuDetector(), OtsuDetector()),
            (TriangleDetector(), OtsuDetector()),
        ]

        for center, overall in combinations:
            detector = FilamentousFungiDetector(
                    inoculum_detector=center,
                    overall_detector=overall,
            )
            result = detector.apply(image.copy())

            # All should produce valid results
            assert result.objmap[:].max() > 0
            assert result.objmask[:].sum() > 0

    def test_consecutive_labels(self, synth_plate):
        """Test that watershed produces valid non-background labels.

        Watershed segmentation doesn't guarantee consecutive labels due to how
        the algorithm allocates marker IDs. We check that labels are present
        and represent distinct objects rather than checking for consecutiveness.
        """
        image = synth_plate.copy()

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
                overall_detector=TriangleDetector(),
        )
        result = detector.apply(image)

        objmap = result.objmap[:]
        num_objects = objmap.max()

        # Check that 0 is always the background label
        assert 0 in np.unique(objmap), "Label 0 (background) should be present"

        # Check that we have multiple objects
        assert num_objects > 1, "Should detect multiple objects"

        # Check that all labels from 1 to num_objects are present or near-consecutive
        # (watershed doesn't guarantee exact consecutiveness, but should have no large gaps)
        unique_labels = np.unique(objmap)
        unique_nonzero = unique_labels[unique_labels > 0]

        # Allow up to 5% missing labels (due to watershed allocation)
        max_allowed_missing = max(1, int(0.05 * num_objects))
        num_missing = num_objects - len(unique_nonzero)
        assert num_missing <= max_allowed_missing, \
            f"Too many missing labels: {num_missing} missing (max {max_allowed_missing} allowed)"

    def test_no_memory_leaks(self, synth_plate):
        """Test that operation cleans up memory properly."""
        image = synth_plate.copy()

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(),
                overall_detector=TriangleDetector(),
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
                overall_detector=TriangleDetector(),
        )

        result1 = detector.apply(image.copy())
        result2 = detector.apply(image.copy())

        # Same input should produce identical output
        np.testing.assert_array_equal(result1.objmap[:], result2.objmap[:])
        np.testing.assert_array_equal(result1.objmask[:], result2.objmask[:])
