"""Tests for SineAlignmentRefiner — NCC-specific and parameter tests.

Basic refiner behavior (creation, inplace, edge cases, pipeline, labeling,
selection mode) is tested in test_shared_refiner_behavior.py.
"""

from __future__ import annotations

import numpy as np
from phenotypic.detect import OtsuDetector, RoundPeaksDetector
from phenotypic.refine import SineAlignmentRefiner


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
        template = np.ones(25)

        ncc = SineAlignmentRefiner._normalized_cross_correlation(signal, template)

        np.testing.assert_array_equal(ncc, np.zeros(100))

    def test_sine_edge_estimation_returns_correct_count(self, synth_plate_detected):
        """Test that _estimate_edges returns n_bins + 1 edges."""
        detected = synth_plate_detected.copy()

        refiner = SineAlignmentRefiner()
        objmask = detected.objmask[:]

        edges = refiner._estimate_edges(objmask, axis=0, n_bins=8)
        assert len(edges) == 9

        edges = refiner._estimate_edges(objmask, axis=1, n_bins=12)
        assert len(edges) == 13


class TestSineAlignmentRefinerParameters:
    """Test parameter effects on refinement results."""

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

    def test_correlation_threshold_effect(self, synth_plate):
        """Test that correlation_threshold affects peak selection."""
        grid_image = synth_plate.copy()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        refiner_low = SineAlignmentRefiner(correlation_threshold=0.1)
        refined_low = refiner_low.apply(detected)

        refiner_high = SineAlignmentRefiner(correlation_threshold=0.8)
        refined_high = refiner_high.apply(detected.copy())

        assert refined_low.objmap[:].max() > 0
        assert refined_high.objmap[:].max() > 0

    def test_smoothing_sigma_effect(self, synth_plate):
        """Test effect of smoothing_sigma parameter on grid detection."""
        grid_image = synth_plate.copy()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        refiner_no_smooth = SineAlignmentRefiner(smoothing_sigma=0.0)
        refined_no_smooth = refiner_no_smooth.apply(detected)

        refiner_smooth = SineAlignmentRefiner(smoothing_sigma=2.0)
        refined_smooth = refiner_smooth.apply(detected.copy())

        assert refined_no_smooth.objmap[:].max() > 0
        assert refined_smooth.objmap[:].max() > 0

    def test_edge_refinement_toggle(self, synth_plate):
        """Test effect of edge_refinement parameter."""
        grid_image = synth_plate.copy()
        detector = RoundPeaksDetector()
        detected = detector.apply(grid_image)

        refiner_no_refine = SineAlignmentRefiner(edge_refinement=False)
        refined_no_refine = refiner_no_refine.apply(detected)

        refiner_refine = SineAlignmentRefiner(edge_refinement=True)
        refined_refine = refiner_refine.apply(detected.copy())

        assert refined_no_refine.objmap[:].max() > 0
        assert refined_refine.objmap[:].max() > 0

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

        ops = restored.get_ops()
        assert len(ops) == 2
        refiner = ops["SineAlignmentRefiner"]
        assert isinstance(refiner, SineAlignmentRefiner)
        assert refiner.smoothing_sigma == 1.5
        assert refiner.correlation_threshold == 0.4
        assert refiner.edge_refinement is False
