"""Tests for ExtractColonyCore and the module-level extract_gmm_cores function."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.refine._extract_colony_core import ExtractColonyCore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bright_core_data():
    """Synthetic labelled region with a clearly bright core and dim surround.

    Region 1 occupies rows 20:60, cols 20:60.
    The inner 20x20 block (30:50, 30:50) is bright (0.9) while the outer
    frame is dim (0.3).  The contrast is strong enough for a 2-component
    GMM to separate core from surround.
    """
    intensity = np.zeros((100, 100), dtype=np.float64)
    label_map = np.zeros((100, 100), dtype=np.int32)

    label_map[20:60, 20:60] = 1
    intensity[20:60, 20:60] = 0.3  # dim surround
    intensity[30:50, 30:50] = 0.9  # bright core

    return intensity, label_map


@pytest.fixture()
def multi_label_data():
    """Two labelled regions with distinct bright cores."""
    intensity = np.zeros((100, 200), dtype=np.float64)
    label_map = np.zeros((100, 200), dtype=np.int32)

    # Region 1 (left)
    label_map[20:60, 20:60] = 1
    intensity[20:60, 20:60] = 0.3
    intensity[30:50, 30:50] = 0.9

    # Region 2 (right)
    label_map[20:60, 120:160] = 2
    intensity[20:60, 120:160] = 0.25
    intensity[30:50, 130:150] = 0.85

    return intensity, label_map


# ---------------------------------------------------------------------------
# Module-level function: extract_gmm_cores
# ---------------------------------------------------------------------------


class TestExtractGMMCores:
    """Tests for the public module-level function."""

    def test_empty_label_map_returns_empty(self):
        """An all-zero label map should produce an all-zero output."""
        intensity = np.random.default_rng(0).random((50, 50))
        label_map = np.zeros((50, 50), dtype=np.int32)

        result = ExtractColonyCore._extract_cores(intensity, label_map)

        assert result.shape == label_map.shape
        assert result.dtype == label_map.dtype
        assert (result == 0).all()

    def test_two_component_core_extraction(self, bright_core_data):
        """A region with a strong bright centre should be shrunk to its core."""
        intensity, label_map = bright_core_data

        result = ExtractColonyCore._extract_cores(
            intensity,
            label_map,
            separation_threshold=0.5,
            min_core_area=10,
        )

        # The refined mask should still contain label 1
        assert 1 in result

        # The core should be smaller than the original region
        original_area = (label_map == 1).sum()
        refined_area = (result == 1).sum()
        assert refined_area < original_area

        # The core should overlap with the bright centre
        bright_zone = np.zeros_like(label_map, dtype=bool)
        bright_zone[30:50, 30:50] = True
        core_mask = result == 1
        overlap = (core_mask & bright_zone).sum()
        assert overlap > 0, "Core should overlap with the bright centre"

        # Most of the core should be in the bright zone
        fraction_in_bright = overlap / core_mask.sum()
        assert fraction_in_bright > 0.5

    def test_uniform_region_unchanged(self):
        """A region with near-uniform intensity should be left unchanged."""
        intensity = np.full((50, 50), 0.5, dtype=np.float64)
        label_map = np.zeros((50, 50), dtype=np.int32)
        label_map[10:40, 10:40] = 1

        result = ExtractColonyCore._extract_cores(intensity, label_map)

        np.testing.assert_array_equal(result, label_map)

    def test_small_region_kept_as_is(self):
        """Regions smaller than min_core_area should be preserved."""
        intensity = np.zeros((50, 50), dtype=np.float64)
        label_map = np.zeros((50, 50), dtype=np.int32)

        # A 3x3 region = 9 pixels, well below default min_core_area=30
        label_map[20:23, 20:23] = 1
        intensity[20:23, 20:23] = 0.3
        intensity[21, 21] = 0.9

        result = ExtractColonyCore._extract_cores(intensity, label_map, min_core_area=30)

        np.testing.assert_array_equal(result, label_map)

    def test_multiple_labels_processed_independently(self, multi_label_data):
        """Each label should be refined independently."""
        intensity, label_map = multi_label_data

        result = ExtractColonyCore._extract_cores(
            intensity,
            label_map,
            separation_threshold=0.5,
            min_core_area=10,
        )

        # Both labels should still be present
        assert 1 in result
        assert 2 in result

        # Both should have been refined (smaller than original)
        for lbl in [1, 2]:
            original_area = (label_map == lbl).sum()
            refined_area = (result == lbl).sum()
            assert refined_area < original_area, (
                f"Label {lbl} should have been refined to a smaller core"
            )

    def test_morph_open_radius_zero_disables_opening(self, bright_core_data):
        """Setting morph_open_radius=0 should skip morphological opening."""
        intensity, label_map = bright_core_data

        result_with = ExtractColonyCore._extract_cores(
            intensity,
            label_map,
            separation_threshold=0.5,
            min_core_area=10,
            morph_open_radius=2,
            morph_close_radius=0,
        )
        result_without = ExtractColonyCore._extract_cores(
            intensity,
            label_map,
            separation_threshold=0.5,
            min_core_area=10,
            morph_open_radius=0,
            morph_close_radius=0,
        )

        # With opening should generally be equal or smaller than without
        assert (result_with == 1).sum() <= (result_without == 1).sum()

    def test_morph_close_radius_zero_disables_closing(self, bright_core_data):
        """Setting morph_close_radius=0 should skip morphological closing."""
        intensity, label_map = bright_core_data

        result_with = ExtractColonyCore._extract_cores(
            intensity,
            label_map,
            separation_threshold=0.5,
            min_core_area=10,
            morph_open_radius=0,
            morph_close_radius=3,
        )
        result_without = ExtractColonyCore._extract_cores(
            intensity,
            label_map,
            separation_threshold=0.5,
            min_core_area=10,
            morph_open_radius=0,
            morph_close_radius=0,
        )

        # With closing should generally be equal or larger than without
        assert (result_with == 1).sum() >= (result_without == 1).sum()

    def test_output_dtype_matches_input(self, bright_core_data):
        """Output label map should have the same dtype as input."""
        intensity, label_map = bright_core_data
        result = ExtractColonyCore._extract_cores(intensity, label_map)
        assert result.dtype == label_map.dtype

    def test_output_shape_matches_input(self, bright_core_data):
        """Output label map should have the same shape as input."""
        intensity, label_map = bright_core_data
        result = ExtractColonyCore._extract_cores(intensity, label_map)
        assert result.shape == label_map.shape


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_build_ellipse_kernel_positive_radius(self):
        kernel = ExtractColonyCore._build_ellipse_kernel(2)
        assert kernel is not None
        assert kernel.shape == (5, 5)
        assert kernel.dtype == np.uint8

    def test_build_ellipse_kernel_zero_radius(self):
        assert ExtractColonyCore._build_ellipse_kernel(0) is None

    def test_build_ellipse_kernel_negative_radius(self):
        assert ExtractColonyCore._build_ellipse_kernel(-1) is None

    def test_normalized_separation_well_separated(self):
        """Two well-separated Gaussians should have high separation."""
        from sklearn.mixture import GaussianMixture

        rng = np.random.default_rng(42)
        data = np.concatenate([
            rng.normal(0.2, 0.05, 500),
            rng.normal(0.8, 0.05, 500),
        ]).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(data)
        sep = ExtractColonyCore._normalized_separation(gmm)
        assert sep > 2.0, "Well-separated means should give high separation"

    def test_normalized_separation_overlapping(self):
        """Overlapping Gaussians should have low separation."""
        from sklearn.mixture import GaussianMixture

        rng = np.random.default_rng(42)
        data = rng.normal(0.5, 0.2, 1000).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(data)
        sep = ExtractColonyCore._normalized_separation(gmm)
        assert sep < 1.0, "Overlapping distributions should give low separation"


# ---------------------------------------------------------------------------
# ExtractColonyCore class integration
# ---------------------------------------------------------------------------


class TestGMMCoreExtractorPipeline:
    """Test ExtractColonyCore through the pipeline / apply interface."""

    def test_apply_on_detected_image(self, synth_plate_detected):
        """Apply ExtractColonyCore after OtsuDetector on synthetic data."""
        image = synth_plate_detected.copy()

        # Should have some detections
        assert image.objmap[:].max() > 0

        original_labels = np.unique(image.objmap[:])
        original_labels = original_labels[original_labels != 0]

        refiner = ExtractColonyCore(
            separation_threshold=0.5,
            min_core_area=5,
        )
        refined = refiner.apply(image)

        # The refined image should still have detections
        assert refined.objmap[:].max() > 0

    def test_serialization_roundtrip(self):
        """ExtractColonyCore should survive JSON serialization in a pipeline."""
        pipe = ImagePipeline(ops=[
            OtsuDetector(),
            ExtractColonyCore(
                n_components=2,
                separation_threshold=0.7,
                min_core_area=20,
                morph_open_radius=0,
                morph_close_radius=3,
            ),
        ])

        json_str = pipe.to_json()
        loaded = ImagePipeline.from_json(json_str)

        ops = list(loaded._ops.values())
        assert len(ops) == 2
        op = ops[1]
        assert isinstance(op, ExtractColonyCore)
        assert op.n_components == 2
        assert op.separation_threshold == 0.7
        assert op.min_core_area == 20
        assert op.morph_open_radius == 0
        assert op.morph_close_radius == 3

    def test_is_object_refiner(self):
        """ExtractColonyCore should be an ObjectRefiner subclass."""
        from phenotypic.abc_ import ObjectRefiner

        op = ExtractColonyCore()
        assert isinstance(op, ObjectRefiner)

    def test_inplace_false_preserves_original(self, bright_core_data):
        """apply(inplace=False) should not modify the original image."""
        intensity, label_map = bright_core_data

        # Build an image with detections
        rgb = np.stack([intensity] * 3, axis=-1)
        rgb = (rgb * 255).astype(np.uint8)
        image = Image(rgb)
        image.objmask[:] = label_map > 0
        image.objmap[:] = label_map

        original_objmap = image.objmap[:].copy()

        refiner = ExtractColonyCore(
            separation_threshold=0.5,
            min_core_area=10,
        )
        refined = refiner.apply(image, inplace=False)

        # Original should be unchanged
        np.testing.assert_array_equal(image.objmap[:], original_objmap)

        # Refined may differ
        assert refined is not image
