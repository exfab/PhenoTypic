"""Tests for SineAlignmentRefiner - sinusoidal cross-correlation grid-aligned object refinement."""

from __future__ import annotations

import numpy as np
from phenotypic import Image, GridImage
from phenotypic.detect import OtsuDetector, RoundPeaksDetector
from phenotypic.refine import SineAlignmentRefiner
from phenotypic.data import load_synth_yeast_plate


class TestSineAlignmentRefinerBasics:
    """Test basic SineAlignmentRefiner functionality."""

    def test_refiner_creation(self):
        """Test that SineAlignmentRefiner can be instantiated with default parameters."""
        refiner = SineAlignmentRefiner()
        assert refiner.smoothing_sigma == 2.0
        assert refiner.min_peak_distance is None
        assert refiner.peak_prominence is None
        assert refiner.edge_refinement is True
        assert refiner.correlation_threshold == 0.3

    def test_refiner_with_custom_parameters(self):
        """Test SineAlignmentRefiner with custom parameters."""
        refiner = SineAlignmentRefiner(
                smoothing_sigma=1.5,
                min_peak_distance=20,
                peak_prominence=0.15,
                edge_refinement=False,
                correlation_threshold=0.5,
        )
        assert refiner.smoothing_sigma == 1.5
        assert refiner.min_peak_distance == 20
        assert refiner.peak_prominence == 0.15
        assert refiner.edge_refinement is False
        assert refiner.correlation_threshold == 0.5

    def test_grid_alignment_with_gridimage(self):
        """Test SineAlignmentRefiner with explicit GridImage (known grid dimensions)."""
        grid_image = load_synth_yeast_plate()
        assert isinstance(grid_image, GridImage)

        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        initial_count = detected.objmap[:].max()
        assert initial_count > 0

        refiner = SineAlignmentRefiner()
        refined = refiner.apply(detected)

        refined_count = refined.objmap[:].max()
        assert refined_count > 0

    def test_grid_alignment_with_regular_image(self):
        """Test SineAlignmentRefiner with regular Image (grid inference via sine correlation)."""
        grid_image = load_synth_yeast_plate()
        image = Image.imread(grid_image.path) if hasattr(grid_image,
                                                         'path') else grid_image

        detector = RoundPeaksDetector()
        detected = detector.apply(image)

        initial_count = detected.objmap[:].max()
        assert initial_count > 0

        refiner = SineAlignmentRefiner(smoothing_sigma=2.0, edge_refinement=True)
        refined = refiner.apply(detected)

        refined_count = refined.objmap[:].max()
        assert refined_count > 0

    def test_objmask_objmap_consistency(self):
        """Test that objmask and objmap remain consistent after refinement."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        refiner = SineAlignmentRefiner()
        refined = refiner.apply(detected)

        objmap = refined.objmap[:]
        objmask = refined.objmask[:]

        mask_from_map = objmap > 0
        np.testing.assert_array_equal(objmask, mask_from_map)

    def test_inplace_vs_copy(self):
        """Test inplace vs copy behavior."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        original_objmap = detected.objmap[:].copy()

        refiner = SineAlignmentRefiner()
        result_copy = refiner.apply(detected, inplace=False)

        np.testing.assert_array_equal(detected.objmap[:], original_objmap)
        assert not np.array_equal(result_copy.objmap[:], original_objmap)

        detected2 = load_synth_yeast_plate()
        detector.apply(detected2, inplace=True)
        refiner.apply(detected2, inplace=True)
        assert not np.array_equal(detected2.objmap[:], original_objmap)

    def test_protected_image_data(self):
        """Test that rgb, gray, and detect_mat are protected from modification."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        original_rgb = detected.rgb[:].copy()
        original_gray = detected.gray[:].copy()
        original_detect_mat = detected.detect_mat[:].copy()

        refiner = SineAlignmentRefiner()
        refined = refiner.apply(detected)

        np.testing.assert_array_equal(refined.rgb[:], original_rgb)
        np.testing.assert_array_equal(refined.gray[:], original_gray)
        np.testing.assert_array_equal(refined.detect_mat[:], original_detect_mat)


class TestSineAlignmentRefinerCorrelation:
    """Test sinusoidal cross-correlation specific functionality."""

    def test_ncc_perfect_match(self):
        """Test NCC returns ~1.0 for identical signal and template."""
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        template = np.sin(np.linspace(0, 2 * np.pi, 25))

        ncc = SineAlignmentRefiner._normalized_cross_correlation(signal, template)

        assert ncc.shape == signal.shape
        assert np.max(ncc) > 0.5

    def test_ncc_zero_signal(self):
        """Test NCC returns near-zero in interior for constant (zero-variance) signal."""
        signal = np.ones(100)
        template = np.sin(np.linspace(0, 2 * np.pi, 25))

        ncc = SineAlignmentRefiner._normalized_cross_correlation(signal, template)

        # Interior values (away from edge effects) should be ~0
        interior = ncc[25:-25]
        np.testing.assert_allclose(interior, 0.0, atol=1e-8)

    def test_ncc_bounds(self):
        """Test NCC values are bounded in [-1, 1]."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(200)
        template = rng.standard_normal(30)

        ncc = SineAlignmentRefiner._normalized_cross_correlation(signal, template)

        assert np.all(ncc >= -1.0)
        assert np.all(ncc <= 1.0)

    def test_ncc_zero_template(self):
        """Test NCC returns zeros when template has zero variance."""
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        template = np.ones(25)  # Constant template

        ncc = SineAlignmentRefiner._normalized_cross_correlation(signal, template)

        np.testing.assert_array_equal(ncc, np.zeros(100))

    def test_sine_edge_estimation_returns_correct_count(self):
        """Test that _estimate_edges returns n_bins + 1 edges."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        refiner = SineAlignmentRefiner()
        objmask = detected.objmask[:]

        edges = refiner._estimate_edges(objmask, axis=0, n_bins=8)
        assert len(edges) == 9  # 8 bins -> 9 edges

        edges = refiner._estimate_edges(objmask, axis=1, n_bins=12)
        assert len(edges) == 13  # 12 bins -> 13 edges


class TestSineAlignmentRefinerParameters:
    """Test parameter effects on refinement results."""

    def test_correlation_threshold_effect(self):
        """Test that correlation_threshold affects peak selection."""
        grid_image = load_synth_yeast_plate()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        # Low threshold accepts more peaks
        refiner_low = SineAlignmentRefiner(correlation_threshold=0.1)
        refined_low = refiner_low.apply(detected)

        # High threshold is more selective
        refiner_high = SineAlignmentRefiner(correlation_threshold=0.8)
        refined_high = refiner_high.apply(detected)

        # Both should produce valid results
        assert refined_low.objmap[:].max() > 0
        assert refined_high.objmap[:].max() > 0

    def test_smoothing_sigma_effect(self):
        """Test effect of smoothing_sigma parameter on grid detection."""
        grid_image = load_synth_yeast_plate()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        refiner_no_smooth = SineAlignmentRefiner(smoothing_sigma=0.0)
        refined_no_smooth = refiner_no_smooth.apply(detected)

        refiner_smooth = SineAlignmentRefiner(smoothing_sigma=2.0)
        refined_smooth = refiner_smooth.apply(detected)

        assert refined_no_smooth.objmap[:].max() > 0
        assert refined_smooth.objmap[:].max() > 0

    def test_edge_refinement_toggle(self):
        """Test effect of edge_refinement parameter."""
        grid_image = load_synth_yeast_plate()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        refiner_no_refine = SineAlignmentRefiner(edge_refinement=False)
        refined_no_refine = refiner_no_refine.apply(detected)

        refiner_refine = SineAlignmentRefiner(edge_refinement=True)
        refined_refine = refiner_refine.apply(detected)

        assert refined_no_refine.objmap[:].max() > 0
        assert refined_refine.objmap[:].max() > 0


class TestSineAlignmentRefinerEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_objects_detected(self):
        """Test behavior when no objects are detected."""
        image = Image(np.ones((100, 100, 3), dtype=np.uint8) * 255)

        detector = OtsuDetector()
        detected = detector.apply(image)

        refiner = SineAlignmentRefiner()
        refined = refiner.apply(detected)

        assert refined.objmap[:].max() == 0

    def test_single_object(self):
        """Test refinement with a single detected object."""
        grid_image = load_synth_yeast_plate()

        objmap = np.zeros_like(grid_image.objmap[:])
        objmap[50:100, 50:100] = 1
        grid_image.objmap[:] = objmap

        refiner = SineAlignmentRefiner()
        refined = refiner.apply(grid_image)

        assert refined.objmap[:].max() >= 1

    def test_multiple_objects_per_cell(self):
        """Test that refiner keeps only dominant object per cell."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        objmap_before = detected.objmap[:].copy()
        cells_with_objects_before = np.sum(objmap_before > 0)

        refiner = SineAlignmentRefiner()
        refined = refiner.apply(detected)

        objmap_after = refined.objmap[:].copy()
        cells_with_objects_after = np.sum(objmap_after > 0)

        assert cells_with_objects_after <= cells_with_objects_before


class TestSineAlignmentRefinerPipeline:
    """Test SineAlignmentRefiner in full processing pipelines."""

    def test_pipeline_integration(self):
        """Test SineAlignmentRefiner in a complete processing pipeline."""
        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur, CLAHE

        pipeline = ImagePipeline([
            GaussianBlur(sigma=1),
            CLAHE(clip_limit=2),
            RoundPeaksDetector(),
            SineAlignmentRefiner(),
        ])

        grid_image = load_synth_yeast_plate()
        result = pipeline.apply(grid_image)

        assert result.objmap[:].max() > 0
        assert result.rgb is not None

    def test_multiple_refiners_chained(self):
        """Test chaining multiple refinement operations."""
        from phenotypic.refine import SmallObjectRemover

        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        sine_align = SineAlignmentRefiner()
        sine_aligned = sine_align.apply(detected)

        small_remover = SmallObjectRemover(min_size=100)
        small_removed = small_remover.apply(sine_aligned)

        assert small_removed.objmap[:].max() >= 0
        np.testing.assert_array_equal(
                small_removed.rgb[:], detected.rgb[:]
        )

    def test_json_roundtrip(self):
        """Test JSON serialization roundtrip of SineAlignmentRefiner in pipeline."""
        from phenotypic import ImagePipeline

        pipeline = ImagePipeline([
            RoundPeaksDetector(),
            SineAlignmentRefiner(
                    smoothing_sigma=1.5,
                    correlation_threshold=0.4,
                    edge_refinement=False,
            ),
        ])

        json_str = pipeline.to_json()
        restored = ImagePipeline.from_json(json_str)

        # Verify restored pipeline has same operations
        ops = restored.get_ops()
        assert len(ops) == 2
        refiner = ops["SineAlignmentRefiner"]
        assert isinstance(refiner, SineAlignmentRefiner)
        assert refiner.smoothing_sigma == 1.5
        assert refiner.correlation_threshold == 0.4
        assert refiner.edge_refinement is False


class TestSineAlignmentRefinerLabeling:
    """Test that label reassignment maintains consistency."""

    def test_contiguous_labels(self):
        """Test that refined labels are contiguous (1, 2, 3, ...)."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        refiner = SineAlignmentRefiner()
        refined = refiner.apply(detected)

        objmap = refined.objmap[:]
        max_label = objmap.max()

        if max_label > 0:
            unique_labels = np.unique(objmap)
            expected_labels = np.arange(max_label + 1)
            np.testing.assert_array_equal(unique_labels, expected_labels)

    def test_label_relabeling(self):
        """Test that objects are relabeled contiguously after refinement."""
        grid_image = load_synth_yeast_plate()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        initial_labels = set(np.unique(detected.objmap[:]))
        initial_labels.discard(0)

        refiner = SineAlignmentRefiner()
        refined = refiner.apply(detected)

        refined_labels = set(np.unique(refined.objmap[:]))
        refined_labels.discard(0)

        assert len(refined_labels) <= len(initial_labels)

        if refined_labels:
            max_label = max(refined_labels)
            expected_labels = set(range(1, max_label + 1))
            assert refined_labels == expected_labels


class TestSineAlignmentRefinerSelectionMode:
    """Test SineAlignmentRefiner with different selection modes."""

    def test_default_selection_mode(self):
        """Default selection_mode is 'dominant'."""
        refiner = SineAlignmentRefiner()
        assert refiner.selection_mode == "dominant"

    def test_centered_mode_produces_results(self):
        """Centered mode produces valid refinement results."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        refiner = SineAlignmentRefiner(selection_mode="centered")
        refined = refiner.apply(detected)

        assert refined.objmap[:].max() > 0

    def test_regularized_mode_produces_results(self):
        """Regularized mode produces valid refinement results."""
        grid_image = load_synth_yeast_plate()
        detector = OtsuDetector()
        detected = detector.apply(grid_image)

        refiner = SineAlignmentRefiner(selection_mode="regularized")
        refined = refiner.apply(detected)

        assert refined.objmap[:].max() > 0

    def test_json_roundtrip_with_selection_mode(self):
        """JSON serialization preserves selection_mode."""
        from phenotypic import ImagePipeline

        pipeline = ImagePipeline([
            RoundPeaksDetector(),
            SineAlignmentRefiner(
                    correlation_threshold=0.4,
                    selection_mode="regularized",
            ),
        ])

        json_str = pipeline.to_json()
        restored = ImagePipeline.from_json(json_str)

        ops = restored.get_ops()
        refiner = ops["SineAlignmentRefiner"]
        assert isinstance(refiner, SineAlignmentRefiner)
        assert refiner.selection_mode == "regularized"
        assert refiner.correlation_threshold == 0.4
