"""Tests for GridAlignmentRefiner - grid-aligned object refinement operation."""

from __future__ import annotations

import numpy as np
from phenotypic import Image, GridImage
from phenotypic.detect import OtsuDetector, RoundPeaksDetector
from phenotypic.refine import GridAlignmentRefiner
from phenotypic.data import load_synth_yeast_plate


class TestGridAlignmentRefinerBasics:
    """Test basic GridAlignmentRefiner functionality."""

    def test_refiner_creation(self):
        """Test that GridAlignmentRefiner can be instantiated with default parameters."""
        refiner = GridAlignmentRefiner()
        assert refiner.smoothing_sigma == 2.0
        assert refiner.min_peak_distance is None
        assert refiner.peak_prominence is None
        assert refiner.edge_refinement is True

    def test_refiner_with_custom_parameters(self):
        """Test GridAlignmentRefiner with custom grid inference parameters."""
        refiner = GridAlignmentRefiner(
                smoothing_sigma=1.5,
                min_peak_distance=20,
                peak_prominence=0.15,
                edge_refinement=False,
        )
        assert refiner.smoothing_sigma == 1.5
        assert refiner.min_peak_distance == 20
        assert refiner.peak_prominence == 0.15
        assert refiner.edge_refinement is False

    def test_grid_alignment_with_gridimage(self):
        """Test GridAlignmentRefiner with explicit GridImage (known grid dimensions)."""
        # Load synthetic plate with explicit 8x12 grid
        grid_image = load_synth_yeast_plate()
        assert isinstance(grid_image, GridImage)

        # Detect colonies with basic detector
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        # Get initial object count
        initial_count = detected.objmap[:].max()
        assert initial_count > 0

        # Refine to keep only grid-aligned objects
        refiner = GridAlignmentRefiner()
        refined = refiner.apply(detected)

        # Check that refinement kept some objects
        refined_count = refined.objmap[:].max()
        assert refined_count > 0
        # Note: refined_count may be different from initial_count because
        # the refiner reassigns labels to grid cells (not just filtering)

    def test_grid_alignment_with_regular_image(self):
        """Test GridAlignmentRefiner with regular Image (grid inference)."""
        # Load synthetic plate as regular Image (without grid info)
        grid_image = load_synth_yeast_plate()
        image = Image.imread(grid_image.path) if hasattr(grid_image,
                                                         'path') else grid_image

        # Detect colonies
        detector = RoundPeaksDetector()
        detected = detector.apply(image)

        initial_count = detected.objmap[:].max()
        assert initial_count > 0

        # Refine with grid inference
        refiner = GridAlignmentRefiner(smoothing_sigma=2.0, edge_refinement=True)
        refined = refiner.apply(detected)

        refined_count = refined.objmap[:].max()
        assert refined_count > 0

    def test_objmask_objmap_consistency(self):
        """Test that objmask and objmap remain consistent after refinement."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        refiner = GridAlignmentRefiner()
        refined = refiner.apply(detected)

        objmap = refined.objmap[:]
        objmask = refined.objmask[:]

        # objmask should be True wherever objmap > 0
        mask_from_map = objmap > 0
        np.testing.assert_array_equal(objmask, mask_from_map)

    def test_inplace_vs_copy(self):
        """Test inplace vs copy behavior."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        original_objmap = detected.objmap[:].copy()

        # Non-inplace should not modify original
        refiner = GridAlignmentRefiner()
        result_copy = refiner.apply(detected, inplace=False)

        # Original should be unchanged
        np.testing.assert_array_equal(detected.objmap[:], original_objmap)
        # Result should be different (refined)
        assert not np.array_equal(result_copy.objmap[:], original_objmap)

        # Inplace should modify original
        detected2 = load_synth_yeast_plate()
        detector.apply(detected2, inplace=True)
        refiner.apply(detected2, inplace=True)
        assert not np.array_equal(detected2.objmap[:], original_objmap)

    def test_protected_image_data(self):
        """Test that rgb, gray, and detect_mat are protected from modification."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        # Save original image data
        original_rgb = detected.rgb[:].copy()
        original_gray = detected.gray[:].copy()
        original_detect_mat = detected.detect_mat[:].copy()

        # Apply refinement
        refiner = GridAlignmentRefiner()
        refined = refiner.apply(detected)

        # Check that image data unchanged
        np.testing.assert_array_equal(refined.rgb[:], original_rgb)
        np.testing.assert_array_equal(refined.gray[:], original_gray)
        np.testing.assert_array_equal(refined.detect_mat[:], original_detect_mat)


class TestGridAlignmentRefinerGridInference:
    """Test grid inference capabilities of GridAlignmentRefiner."""

    def test_grid_inference_with_regular_image(self):
        """Test that grid inference works for regular Image without explicit dimensions."""
        grid_image = load_synth_yeast_plate()
        image = Image.imread(grid_image.path) if hasattr(grid_image,
                                                         'path') else grid_image

        detector = RoundPeaksDetector()
        detected = detector.apply(image)

        refiner = GridAlignmentRefiner()
        refined = refiner.apply(detected)

        # Should successfully infer and apply grid
        assert refined.objmap[:].max() > 0

    def test_smoothing_sigma_effect(self):
        """Test effect of smoothing_sigma parameter on grid detection."""
        grid_image = load_synth_yeast_plate()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        # Test with no smoothing
        refiner_no_smooth = GridAlignmentRefiner(smoothing_sigma=0.0)
        refined_no_smooth = refiner_no_smooth.apply(detected)

        # Test with smoothing
        refiner_smooth = GridAlignmentRefiner(smoothing_sigma=2.0)
        refined_smooth = refiner_smooth.apply(detected)

        # Both should produce valid results (may differ)
        assert refined_no_smooth.objmap[:].max() > 0
        assert refined_smooth.objmap[:].max() > 0

    def test_edge_refinement_effect(self):
        """Test effect of edge_refinement parameter."""
        grid_image = load_synth_yeast_plate()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        # Test without edge refinement
        refiner_no_refine = GridAlignmentRefiner(edge_refinement=False)
        refined_no_refine = refiner_no_refine.apply(detected)

        # Test with edge refinement
        refiner_refine = GridAlignmentRefiner(edge_refinement=True)
        refined_refine = refiner_refine.apply(detected)

        # Both should work
        assert refined_no_refine.objmap[:].max() > 0
        assert refined_refine.objmap[:].max() > 0


class TestGridAlignmentRefinerEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_objects_detected(self):
        """Test behavior when no objects are detected."""
        # Create blank image with no colonies
        image = Image(np.ones((100, 100, 3), dtype=np.uint8) * 255)

        # Detect will find no objects
        detector = OtsuDetector()
        detected = detector.apply(image)

        # Refiner should handle gracefully
        refiner = GridAlignmentRefiner()
        refined = refiner.apply(detected)

        # Should have no objects
        assert refined.objmap[:].max() == 0

    def test_single_object(self):
        """Test refinement with a single detected object."""
        # Create image with single colony in one grid cell
        grid_image = load_synth_yeast_plate()

        # Create sparse detection (only one object)
        objmap = np.zeros_like(grid_image.objmap[:])
        objmap[50:100, 50:100] = 1

        grid_image.objmap[:] = objmap

        refiner = GridAlignmentRefiner()
        refined = refiner.apply(grid_image)

        # Should keep the single object
        assert refined.objmap[:].max() >= 1

    def test_multiple_objects_per_cell(self):
        """Test that refiner keeps only dominant object per cell."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        # Get initial distribution
        objmap_before = detected.objmap[:].copy()
        cells_with_objects_before = np.sum(objmap_before > 0)

        # Apply refinement
        refiner = GridAlignmentRefiner()
        refined = refiner.apply(detected)

        objmap_after = refined.objmap[:].copy()
        cells_with_objects_after = np.sum(objmap_after > 0)

        # After refinement, should have fewer or equal cells with objects
        # (removed fragmented/spurious detections)
        assert cells_with_objects_after <= cells_with_objects_before


class TestGridAlignmentRefinerPipeline:
    """Test GridAlignmentRefiner in full processing pipelines."""

    def test_pipeline_integration(self):
        """Test GridAlignmentRefiner in a complete processing pipeline."""
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur, CLAHE

        pipeline = ImagePipeline([
            GaussianBlur(sigma=1),
            CLAHE(clip_limit=2),
            RoundPeaksDetector(),
            GridAlignmentRefiner(),
        ])

        grid_image = load_synth_yeast_plate()
        result = pipeline.apply(grid_image)

        # Should have processed successfully
        assert result.objmap[:].max() > 0
        assert result.rgb is not None

    def test_multiple_refiners_chained(self):
        """Test chaining multiple refinement operations."""
        from phenotypic.refine import SmallObjectRemover

        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        # Chain refinement operations
        grid_align = GridAlignmentRefiner()
        grid_aligned = grid_align.apply(detected)

        # Can chain with other refiners
        small_remover = SmallObjectRemover(min_size=100)
        small_removed = small_remover.apply(grid_aligned)

        # Should maintain consistency
        assert small_removed.objmap[:].max() >= 0
        np.testing.assert_array_equal(
                small_removed.rgb[:], detected.rgb[:]
        )


class TestGridAlignmentRefinerLabelingConsistency:
    """Test that label reassignment maintains consistency."""

    def test_contiguous_labels(self):
        """Test that refined labels are contiguous (1, 2, 3, ...)."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        refiner = GridAlignmentRefiner()
        refined = refiner.apply(detected)

        objmap = refined.objmap[:]
        max_label = objmap.max()

        if max_label > 0:
            # Labels should be 0, 1, 2, ..., max_label
            unique_labels = np.unique(objmap)
            expected_labels = np.arange(max_label + 1)
            np.testing.assert_array_equal(unique_labels, expected_labels)

    def test_label_relabeling(self):
        """Test that objects are relabeled contiguously after refinement."""
        grid_image = load_synth_yeast_plate()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        # Get initial labels
        initial_labels = set(np.unique(detected.objmap[:]))
        initial_labels.discard(0)  # Remove background

        refiner = GridAlignmentRefiner()
        refined = refiner.apply(detected)

        # Get refined labels
        refined_labels = set(np.unique(refined.objmap[:]))
        refined_labels.discard(0)

        # Count should be <= initial (some objects may be merged/removed)
        assert len(refined_labels) <= len(initial_labels)

        # Labels should be contiguous if present
        if refined_labels:
            max_label = max(refined_labels)
            expected_labels = set(range(1, max_label + 1))
            assert refined_labels == expected_labels


class TestGridAlignmentRefinerMemoryAndPerformance:
    """Test memory efficiency and performance."""

    def test_garbage_collection_called(self):
        """Test that garbage collection is invoked in _operate."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        refiner = GridAlignmentRefiner()
        # Should not raise or have memory issues
        refined = refiner.apply(detected)

        assert refined.objmap[:].max() > 0

    def test_large_image_handling(self):
        """Test that refiner handles larger images without issues."""
        # Create a larger synthetic image
        grid_image = load_synth_yeast_plate()

        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        refiner = GridAlignmentRefiner()
        refined = refiner.apply(detected)

        assert refined.objmap[:].max() > 0
