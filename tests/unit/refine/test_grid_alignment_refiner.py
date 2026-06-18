"""Tests for GridAlignmentRefiner — grid inference and mixin-direct tests.

Basic refiner behavior (creation, inplace, edge cases, pipeline, labeling,
selection mode) is tested in test_shared_refiner_behavior.py.
"""

from __future__ import annotations

import pytest
import numpy as np
from phenotypic.detect import RoundPeaksDetector
from phenotypic.refine import GridAlignmentRefiner
from phenotypic.sdk_.mixin import GridInferenceMixin


class TestGridAlignmentRefinerGridInference:
    """Test grid inference capabilities of GridAlignmentRefiner."""

    def test_grid_inference_with_regular_image(self, synth_plate):
        """Test that grid inference works for regular Image without explicit dimensions."""
        from phenotypic import Image

        image = Image(synth_plate.copy())

        detector = RoundPeaksDetector()
        detected = detector.apply(image)

        refiner = GridAlignmentRefiner()
        refined = refiner.apply(detected)

        assert refined.objmap[:].max() > 0

    def test_smoothing_sigma_effect(self, synth_plate):
        """Test effect of smoothing_sigma parameter on grid detection."""
        grid_image = synth_plate.copy()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        refiner_no_smooth = GridAlignmentRefiner(smoothing_sigma=0.0)
        refined_no_smooth = refiner_no_smooth.apply(detected)

        refiner_smooth = GridAlignmentRefiner(smoothing_sigma=2.0)
        refined_smooth = refiner_smooth.apply(detected.copy())

        assert refined_no_smooth.objmap[:].max() > 0
        assert refined_smooth.objmap[:].max() > 0

    def test_edge_refinement_effect(self, synth_plate):
        """Test effect of edge_refinement parameter."""
        grid_image = synth_plate.copy()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        refiner_no_refine = GridAlignmentRefiner(edge_refinement=False)
        refined_no_refine = refiner_no_refine.apply(detected)

        refiner_refine = GridAlignmentRefiner(edge_refinement=True)
        refined_refine = refiner_refine.apply(detected.copy())

        assert refined_no_refine.objmap[:].max() > 0
        assert refined_refine.objmap[:].max() > 0


class TestSelectCellObject:
    """Test _select_cell_object static method."""

    def test_empty_region_returns_none(self):
        """All-zero region returns None."""
        region = np.zeros((10, 10), dtype=np.int32)
        assert GridInferenceMixin._select_cell_object(region, "dominant") is None
        assert GridInferenceMixin._select_cell_object(region, "centered") is None

    def test_single_object_dominant(self):
        """Single-object region returns that label in dominant mode."""
        region = np.zeros((10, 10), dtype=np.int32)
        region[3:7, 3:7] = 5
        assert GridInferenceMixin._select_cell_object(region, "dominant") == 5

    def test_single_object_centered(self):
        """Single-object region returns that label in centered mode."""
        region = np.zeros((10, 10), dtype=np.int32)
        region[3:7, 3:7] = 5
        assert GridInferenceMixin._select_cell_object(region, "centered") == 5

    def test_dominant_picks_largest(self):
        """Dominant mode picks the label with the most pixels."""
        region = np.zeros((20, 20), dtype=np.int32)
        region[0:8, 0:8] = 1
        region[9:11, 9:11] = 2
        assert GridInferenceMixin._select_cell_object(region, "dominant") == 1

    def test_centered_picks_closest_to_center(self):
        """Centered mode picks the label whose centroid is nearest cell center."""
        region = np.zeros((20, 20), dtype=np.int32)
        region[0:8, 0:8] = 1
        region[9:11, 9:11] = 2
        assert GridInferenceMixin._select_cell_object(region, "centered") == 2

    def test_centered_vs_dominant_differ(self):
        """Centered and dominant should disagree when large corner vs small center."""
        region = np.zeros((20, 20), dtype=np.int32)
        region[0:8, 0:8] = 1
        region[9:11, 9:11] = 2
        dominant = GridInferenceMixin._select_cell_object(region, "dominant")
        centered = GridInferenceMixin._select_cell_object(region, "centered")
        assert dominant != centered

    def test_invalid_mode_raises(self):
        """Unknown selection mode raises ValueError."""
        region = np.zeros((10, 10), dtype=np.int32)
        region[3:7, 3:7] = 1
        with pytest.raises(ValueError, match="Unknown selection_mode"):
            GridInferenceMixin._select_cell_object(region, "bogus")  # type: ignore[arg-type]


class TestAssignGridObjects:
    """Test _assign_grid_objects static method."""

    def test_regularized_produces_valid_output(self):
        """Regularized mode on a synthetic grid produces labels > 0."""
        labeled = np.zeros((100, 100), dtype=np.int32)
        lbl = 1
        for r in range(4):
            for c in range(4):
                r0, c0 = r * 25 + 10, c * 25 + 10
                labeled[r0:r0 + 5, c0:c0 + 5] = lbl
                lbl += 1
        row_edges = np.array([0, 25, 50, 75, 100])
        col_edges = np.array([0, 25, 50, 75, 100])

        result = GridInferenceMixin._assign_grid_objects(
            labeled, row_edges, col_edges, "regularized", np.int32
        )
        assert result.max() > 0
        assert result.max() <= 16

    def test_regularized_vs_dominant_on_clean_data(self):
        """On clean data, regularized and dominant produce the same count."""
        labeled = np.zeros((100, 100), dtype=np.int32)
        lbl = 1
        for r in range(4):
            for c in range(4):
                r0, c0 = r * 25 + 10, c * 25 + 10
                labeled[r0:r0 + 5, c0:c0 + 5] = lbl
                lbl += 1
        row_edges = np.array([0, 25, 50, 75, 100])
        col_edges = np.array([0, 25, 50, 75, 100])

        dominant = GridInferenceMixin._assign_grid_objects(
            labeled, row_edges, col_edges, "dominant", np.int32
        )
        regularized = GridInferenceMixin._assign_grid_objects(
            labeled, row_edges, col_edges, "regularized", np.int32
        )
        assert dominant.max() == regularized.max()

    def test_centered_mode(self):
        """Centered mode selects per-cell centered objects."""
        labeled = np.zeros((100, 100), dtype=np.int32)
        lbl = 1
        for r in range(4):
            for c in range(4):
                r0, c0 = r * 25 + 10, c * 25 + 10
                labeled[r0:r0 + 5, c0:c0 + 5] = lbl
                lbl += 1
        row_edges = np.array([0, 25, 50, 75, 100])
        col_edges = np.array([0, 25, 50, 75, 100])

        result = GridInferenceMixin._assign_grid_objects(
            labeled, row_edges, col_edges, "centered", np.int32
        )
        assert result.max() == 16
