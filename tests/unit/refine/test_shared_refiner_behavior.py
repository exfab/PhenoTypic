"""Shared tests for refiner behavior across GridAlignmentRefiner and SineAlignmentRefiner.

These tests verify common behavior that is identical for both refiners,
avoiding duplication.
"""

from __future__ import annotations

import pytest
import numpy as np
from phenotypic import Image, GridImage
from phenotypic.detect import OtsuDetector, RoundPeaksDetector
from phenotypic.refine import GridAlignmentRefiner, SineAlignmentRefiner
REFINERS = [GridAlignmentRefiner, SineAlignmentRefiner]


class TestRefinerBasicsShared:
    """Shared basic functionality tests."""

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_refiner_creation(self, RefinerClass):
        """Test that refiner can be instantiated with default parameters."""
        refiner = RefinerClass()
        assert refiner.smoothing_sigma == 2.0
        assert refiner.min_peak_distance is None
        assert refiner.peak_prominence is None
        assert refiner.edge_refinement is True

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_grid_alignment_with_gridimage(self, RefinerClass, synth_plate_detected):
        """Test refiner with explicit GridImage (known grid dimensions)."""
        detected = synth_plate_detected.copy()
        assert isinstance(detected, GridImage)

        initial_count = detected.objmap[:].max()
        assert initial_count > 0

        refiner = RefinerClass()
        refined = refiner.apply(detected)

        refined_count = refined.objmap[:].max()
        assert refined_count > 0

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_grid_alignment_with_regular_image(self, RefinerClass, synth_plate):
        """Test refiner with regular Image (grid inference)."""
        image = Image(synth_plate.copy())

        detector = RoundPeaksDetector()
        detected = detector.apply(image)

        initial_count = detected.objmap[:].max()
        assert initial_count > 0

        refiner = RefinerClass(smoothing_sigma=2.0, edge_refinement=True)
        refined = refiner.apply(detected)

        refined_count = refined.objmap[:].max()
        assert refined_count > 0

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_objmask_objmap_consistency(self, RefinerClass, synth_plate_detected):
        """Test that objmask and objmap remain consistent after refinement."""
        detected = synth_plate_detected.copy()

        refiner = RefinerClass()
        refined = refiner.apply(detected)

        objmap = refined.objmap[:]
        objmask = refined.objmask[:]

        mask_from_map = objmap > 0
        np.testing.assert_array_equal(objmask, mask_from_map)

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_inplace_vs_copy(self, RefinerClass, synth_plate_detected):
        """Test inplace vs copy behavior."""
        detected = synth_plate_detected.copy()
        original_objmap = detected.objmap[:].copy()

        refiner = RefinerClass()
        result_copy = refiner.apply(detected, inplace=False)

        np.testing.assert_array_equal(detected.objmap[:], original_objmap)
        assert not np.array_equal(result_copy.objmap[:], original_objmap)

        detected2 = synth_plate_detected.copy()
        refiner.apply(detected2, inplace=True)
        assert not np.array_equal(detected2.objmap[:], original_objmap)

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_protected_image_data(self, RefinerClass, synth_plate_detected):
        """Test that rgb, gray, and detect_mat are protected from modification."""
        detected = synth_plate_detected.copy()

        original_rgb = detected.rgb[:].copy()
        original_gray = detected.gray[:].copy()
        original_detect_mat = detected.detect_mat[:].copy()

        refiner = RefinerClass()
        refined = refiner.apply(detected)

        np.testing.assert_array_equal(refined.rgb[:], original_rgb)
        np.testing.assert_array_equal(refined.gray[:], original_gray)
        np.testing.assert_array_equal(refined.detect_mat[:], original_detect_mat)


class TestRefinerEdgeCasesShared:
    """Shared edge case tests."""

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_no_objects_detected(self, RefinerClass):
        """Test behavior when no objects are detected."""
        image = Image(np.ones((100, 100, 3), dtype=np.uint8) * 255)

        detector = OtsuDetector()
        detected = detector.apply(image)

        # OtsuDetector on a uniform image labels everything as one object.
        # Manually clear the objmap to test the true "no objects" path.
        detected.objmap[:] = np.zeros_like(detected.objmap[:])

        refiner = RefinerClass()
        refined = refiner.apply(detected)

        assert refined.objmap[:].max() == 0

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_single_object(self, RefinerClass, synth_plate):
        """Test refinement with a single detected object."""
        grid_image = synth_plate.copy()

        objmap = np.zeros_like(grid_image.objmap[:])
        objmap[50:100, 50:100] = 1
        grid_image.objmap[:] = objmap

        refiner = RefinerClass()
        refined = refiner.apply(grid_image)

        assert refined.objmap[:].max() >= 1

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_multiple_objects_per_cell(self, RefinerClass, synth_plate_detected):
        """Test that refiner keeps only dominant object per cell."""
        detected = synth_plate_detected.copy()

        objmap_before = detected.objmap[:].copy()
        cells_with_objects_before = np.sum(objmap_before > 0)

        refiner = RefinerClass()
        refined = refiner.apply(detected)

        objmap_after = refined.objmap[:].copy()
        cells_with_objects_after = np.sum(objmap_after > 0)

        assert cells_with_objects_after <= cells_with_objects_before


class TestRefinerPipelineShared:
    """Shared pipeline integration tests."""

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_pipeline_integration(self, RefinerClass, synth_plate):
        """Test refiner in a complete processing pipeline."""
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur, CLAHE

        pipeline = ImagePipeline([
            GaussianBlur(sigma=1),
            CLAHE(clip_limit=2),
            RoundPeaksDetector(),
            RefinerClass(),
        ])

        grid_image = synth_plate.copy()
        result = pipeline.apply(grid_image)

        assert result.objmap[:].max() > 0
        assert result.rgb is not None

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_multiple_refiners_chained(self, RefinerClass, synth_plate_detected):
        """Test chaining multiple refinement operations."""
        from phenotypic.refine import SmallObjectRemover

        detected = synth_plate_detected.copy()

        refiner = RefinerClass()
        refined = refiner.apply(detected)

        small_remover = SmallObjectRemover(min_size=100)
        small_removed = small_remover.apply(refined)

        assert small_removed.objmap[:].max() >= 0
        np.testing.assert_array_equal(
                small_removed.rgb[:], detected.rgb[:]
        )


class TestRefinerLabelingShared:
    """Shared labeling consistency tests."""

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_contiguous_labels(self, RefinerClass, synth_plate_detected):
        """Test that refined labels are contiguous (1, 2, 3, ...)."""
        detected = synth_plate_detected.copy()

        refiner = RefinerClass()
        refined = refiner.apply(detected)

        objmap = refined.objmap[:]
        max_label = objmap.max()

        if max_label > 0:
            unique_labels = np.unique(objmap)
            expected_labels = np.arange(max_label + 1)
            np.testing.assert_array_equal(unique_labels, expected_labels)

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_label_relabeling(self, RefinerClass, synth_plate):
        """Test that objects are relabeled contiguously after refinement."""
        grid_image = synth_plate.copy()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        initial_labels = set(np.unique(detected.objmap[:]))
        initial_labels.discard(0)

        refiner = RefinerClass()
        refined = refiner.apply(detected)

        refined_labels = set(np.unique(refined.objmap[:]))
        refined_labels.discard(0)

        assert len(refined_labels) <= len(initial_labels)

        if refined_labels:
            max_label = max(refined_labels)
            expected_labels = set(range(1, max_label + 1))
            assert refined_labels == expected_labels


class TestRefinerSelectionModeShared:
    """Shared selection mode tests."""

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_default_selection_mode(self, RefinerClass):
        """Default selection_mode is 'dominant'."""
        refiner = RefinerClass()
        assert refiner.selection_mode == "dominant"

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_centered_mode_produces_results(self, RefinerClass, synth_plate_detected):
        """Centered mode produces valid refinement results."""
        detected = synth_plate_detected.copy()

        refiner = RefinerClass(selection_mode="centered")
        refined = refiner.apply(detected)

        assert refined.objmap[:].max() > 0

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_regularized_mode_produces_results(self, RefinerClass, synth_plate_detected):
        """Regularized mode produces valid refinement results."""
        detected = synth_plate_detected.copy()

        refiner = RefinerClass(selection_mode="regularized")
        refined = refiner.apply(detected)

        assert refined.objmap[:].max() > 0

    @pytest.mark.parametrize("RefinerClass", REFINERS)
    def test_json_roundtrip_with_selection_mode(self, RefinerClass):
        """JSON serialization preserves selection_mode."""
        from phenotypic import ImagePipeline

        pipeline = ImagePipeline([
            RoundPeaksDetector(),
            RefinerClass(selection_mode="centered"),
        ])

        json_str = pipeline.to_json()
        restored = ImagePipeline.from_json(json_str)

        ops = restored.get_ops()
        refiner = ops[RefinerClass.__name__]
        assert isinstance(refiner, RefinerClass)
        assert refiner.selection_mode == "centered"
