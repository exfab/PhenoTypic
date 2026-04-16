"""Tests for the branch pathfinding subpackage.

Unit tests for the modules under ``phenotypic.tools_.branch_pathfinding``
(cost surface composition, fragment prescreening, path quality filtering,
Dijkstra kernels, dataclasses) and integration tests that exercise them
via ``FilamentousFungiDetector(enable_reconnection=True)``.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from phenotypic.tools_.branch_pathfinding import (
    # Cost surface
    compute_anisotropy,
    compute_orientation_coherence,
    compute_local_mad_map,
    assemble_composite_cost,
    apply_structure_mask,
    apply_border_penalty,
    apply_distance_gap_penalty,
    # Fragment prescreening
    compute_min_cost_envelope,
    calibrate_screening_threshold,
    prescreen_fragments,
    # Path quality
    compute_path_metrics,
    extract_calibration_branches,
    calibrate_thresholds,
    apply_filter_cascade,
    # Dijkstra kernels
    run_multisource_dijkstra,
    assign_fragments_to_colonies,
    extract_fragment_paths,
    assemble_connected_mask,
    backtrack_path,
    # Dataclasses
    DijkstraResult,
    FragmentAssignment,
    FragmentPath,
    PrescreenResult,
    PathMetrics,
    CalibrationData,
    FilterThresholds,
    FilterResult,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(scope="session")
def two_colony_scene():
    """Synthetic 100x100 scene with two small colonies and one fragment.

    Colony 1: circle at (20, 20) with radius 10
    Colony 2: circle at (80, 80) with radius 10
    Fragment: small 5x5 blob at rows [25:30], cols [35:40], near colony 1
    Cost surface: uniform 1.0 everywhere, near-zero inside colonies.
    """
    rr, cc = np.ogrid[:100, :100]

    colony_labels = np.zeros((100, 100), dtype=np.int32)
    colony_labels[((rr - 20) ** 2 + (cc - 20) ** 2) < 100] = 1
    colony_labels[((rr - 80) ** 2 + (cc - 80) ** 2) < 100] = 2

    fragment_labels = np.zeros((100, 100), dtype=np.int32)
    fragment_labels[25:30, 35:40] = 1

    cost_surface = np.ones((100, 100), dtype=np.float32)
    cost_surface[colony_labels > 0] = 1e-6

    return colony_labels, fragment_labels, cost_surface


@pytest.fixture(scope="session")
def dijkstra_result(two_colony_scene):
    """Pre-computed Dijkstra result for the two-colony scene."""
    colony_labels, _, cost_surface = two_colony_scene
    return run_multisource_dijkstra(cost_surface, colony_labels, delta=1.0)


class _DuckPath:
    """Lightweight duck-typed path object for compute_path_metrics tests."""

    def __init__(self, coords, cost_profile, total_cost, path_length):
        self.coords = np.asarray(coords, dtype=np.int32)
        self.cost_profile = np.asarray(cost_profile, dtype=np.float64)
        self.total_cost = float(total_cost)
        self.path_length = int(path_length)


# =====================================================================
# TestCostSurface
# =====================================================================


class TestCostSurface:
    """Tests for _cost_surface.py functions."""

    # -- compute_anisotropy --

    def test_anisotropy_equal_moments_near_zero(self):
        """Equal M and m produces anisotropy near 0."""
        M = np.full((10, 10), 5.0, dtype=np.float32)
        m = np.full((10, 10), 5.0, dtype=np.float32)
        result = compute_anisotropy(M, m)
        assert result.dtype == np.float32
        assert_allclose(result, 0.0, atol=1e-4)

    def test_anisotropy_large_M_near_one(self):
        """M >> m produces anisotropy near 1."""
        M = np.full((10, 10), 1000.0, dtype=np.float32)
        m = np.full((10, 10), 1.0, dtype=np.float32)
        result = compute_anisotropy(M, m)
        assert np.all(result > 0.99)

    def test_anisotropy_output_range(self):
        """Anisotropy should be in [0, 1] for non-negative inputs."""
        rng = np.random.default_rng(42)
        M = rng.uniform(0.0, 10.0, size=(20, 20)).astype(np.float32)
        m = rng.uniform(0.0, 10.0, size=(20, 20)).astype(np.float32)
        # Ensure M >= m for meaningful anisotropy
        M, m = np.maximum(M, m), np.minimum(M, m)
        result = compute_anisotropy(M, m)
        assert np.all(result >= -1e-5)
        assert np.all(result <= 1.0 + 1e-5)

    # -- compute_orientation_coherence --

    def test_coherence_uniform_theta_is_one(self):
        """Perfectly aligned orientations yield coherence ~1.0."""
        theta = np.full((50, 50), 0.5, dtype=np.float64)
        result = compute_orientation_coherence(theta, r_coh=5)
        assert result.dtype == np.float32
        assert_allclose(result, 1.0, atol=0.05)

    def test_coherence_random_theta_is_low(self):
        """Random orientations yield low coherence."""
        rng = np.random.default_rng(99)
        theta = rng.uniform(-np.pi, np.pi, size=(80, 80))
        result = compute_orientation_coherence(theta, r_coh=12)
        # Centre pixels should have low coherence
        centre = result[20:60, 20:60]
        assert float(np.mean(centre)) < 0.3

    # -- compute_local_mad_map --

    def test_mad_constant_image_is_zero(self):
        """Constant image gives zero MAD everywhere."""
        img = np.full((30, 30), 7.0, dtype=np.float32)
        result = compute_local_mad_map(img, window_size=5)
        assert result.dtype == np.float32
        assert_allclose(result, 0.0, atol=1e-6)

    def test_mad_raises_for_3d_input(self):
        """3-D array raises ValueError."""
        img = np.zeros((10, 10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="must be 2-D"):
            compute_local_mad_map(img)

    def test_mad_raises_for_even_window(self):
        """Even window_size raises ValueError."""
        img = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="must be odd"):
            compute_local_mad_map(img, window_size=4)

    def test_mad_nonzero_for_noisy_image(self):
        """Noisy image has nonzero MAD."""
        rng = np.random.default_rng(7)
        img = rng.uniform(0, 1, size=(30, 30)).astype(np.float32)
        result = compute_local_mad_map(img, window_size=5)
        assert float(np.mean(result)) > 0.0

    # -- assemble_composite_cost --

    def test_high_features_low_cost(self):
        """High P*A*O produces low cost."""
        shape = (10, 10)
        P = np.ones(shape, dtype=np.float32)
        A = np.ones(shape, dtype=np.float32)
        O = np.ones(shape, dtype=np.float32)
        MAD = np.zeros(shape, dtype=np.float32)
        result = assemble_composite_cost(P, A, O, MAD)
        # denominator = 1*1*1 + eps ~1, numerator = 1
        assert result.dtype == np.float32
        assert np.all(result < 2.0)

    def test_low_features_high_cost(self):
        """Low denominator produces high cost."""
        shape = (10, 10)
        P = np.full(shape, 1e-6, dtype=np.float32)
        A = np.full(shape, 1e-6, dtype=np.float32)
        O = np.full(shape, 1e-6, dtype=np.float32)
        MAD = np.ones(shape, dtype=np.float32)
        result = assemble_composite_cost(P, A, O, MAD)
        # numerator = 1 + 1 = 2, denominator ~ eps -> high cost
        assert np.all(result > 1e3)

    def test_composite_cost_dtype(self):
        """Output is float32."""
        shape = (5, 5)
        result = assemble_composite_cost(
                np.ones(shape, np.float32),
                np.ones(shape, np.float32),
                np.ones(shape, np.float32),
                np.zeros(shape, np.float32),
        )
        assert result.dtype == np.float32

    # -- apply_structure_mask --

    def test_masked_pixels_get_eps(self):
        """Pixels inside colony_mask get eps_free cost."""
        cost = np.full((10, 10), 100.0, dtype=np.float32)
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[2:5, 2:5] = 1
        result = apply_structure_mask(cost, mask, eps_free=1e-6)
        assert_allclose(result[2:5, 2:5], 1e-6)

    def test_unmasked_pixels_unchanged(self):
        """Pixels outside colony_mask remain unchanged."""
        cost = np.full((10, 10), 100.0, dtype=np.float32)
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[2:5, 2:5] = 1
        result = apply_structure_mask(cost, mask)
        assert_allclose(result[0, 0], 100.0)

    def test_returns_copy(self):
        """Result is a copy; input is not modified."""
        cost = np.full((10, 10), 50.0, dtype=np.float32)
        mask = np.ones((10, 10), dtype=np.int32)
        result = apply_structure_mask(cost, mask)
        assert result is not cost
        assert_allclose(cost, 50.0)  # input unchanged


# =====================================================================
# TestFragmentPrescreening
# =====================================================================


class TestFragmentPrescreening:
    """Tests for _fragment_prescreening.py functions."""

    # -- compute_min_cost_envelope --

    def test_min_cost_envelope_picks_local_minima(self):
        """Minimum filter picks up nearby low values within radius."""
        cost = np.full((50, 50), 10.0, dtype=np.float32)
        cost[25, 25] = 0.1  # low-cost spike
        result = compute_min_cost_envelope(cost, r_screen=5)
        # Pixels within 5 of (25, 25) should see the 0.1
        assert result[25, 28] == pytest.approx(0.1, abs=1e-6)
        # Pixels far away should not see it
        assert result[0, 0] == pytest.approx(10.0, abs=1e-6)

    def test_min_cost_envelope_preserves_shape(self):
        """Output has same shape as input."""
        cost = np.ones((30, 40), dtype=np.float32)
        result = compute_min_cost_envelope(cost, r_screen=3)
        assert result.shape == (30, 40)

    # -- calibrate_screening_threshold --

    def test_calibrate_returns_threshold_at_percentile(self):
        """Threshold matches the requested percentile of boundary costs."""
        cost = np.ones((50, 50), dtype=np.float32)
        mask = np.zeros((50, 50), dtype=np.int32)
        mask[10:20, 10:20] = 1  # small colony block
        tau, values = calibrate_screening_threshold(
                cost, mask, r_screen=5, percentile=50.0
        )
        assert isinstance(tau, float)
        assert tau > 0
        assert values.size > 0

    def test_calibrate_raises_for_empty_mask(self):
        """Empty mask raises ValueError (no boundary pixels)."""
        cost = np.ones((20, 20), dtype=np.float32)
        mask = np.zeros((20, 20), dtype=np.int32)
        with pytest.raises(ValueError, match="No boundary pixels"):
            calibrate_screening_threshold(cost, mask)

    # -- prescreen_fragments --

    def test_prescreen_rejects_high_cost_fragment(self):
        """Fragment in uniformly high-cost region is rejected."""
        cost = np.full((50, 50), 100.0, dtype=np.float32)
        frags = np.zeros((50, 50), dtype=np.int32)
        frags[5:10, 5:10] = 1  # fragment in expensive area
        result = prescreen_fragments(
                cost, frags, r_screen=3, tau_screen=1.0
        )
        assert 1 in result.rejected_ids
        assert 1 not in result.passed_ids

    def test_prescreen_keeps_low_cost_fragment(self):
        """Fragment near a low-cost corridor passes screening."""
        cost = np.full((50, 50), 100.0, dtype=np.float32)
        cost[5:10, 5:10] = 0.01  # low-cost patch
        frags = np.zeros((50, 50), dtype=np.int32)
        frags[6:9, 6:9] = 1  # fragment sitting on low-cost patch
        result = prescreen_fragments(
                cost, frags, r_screen=3, tau_screen=50.0
        )
        assert 1 in result.passed_ids

    def test_prescreen_raises_shape_mismatch(self):
        """Mismatched cost/fragment shapes raise ValueError."""
        cost = np.ones((30, 30), dtype=np.float32)
        frags = np.zeros((40, 40), dtype=np.int32)
        frags[5:10, 5:10] = 1
        with pytest.raises(ValueError, match="Shape mismatch"):
            prescreen_fragments(cost, frags, r_screen=3, tau_screen=1.0)

    def test_prescreen_raises_empty_fragments(self):
        """Fragment array with no labels raises ValueError."""
        cost = np.ones((20, 20), dtype=np.float32)
        frags = np.zeros((20, 20), dtype=np.int32)
        with pytest.raises(ValueError, match="no labeled fragments"):
            prescreen_fragments(cost, frags, r_screen=3, tau_screen=1.0)

    def test_prescreen_raises_no_threshold(self):
        """Neither tau nor calibration data raises ValueError."""
        cost = np.ones((20, 20), dtype=np.float32)
        frags = np.zeros((20, 20), dtype=np.int32)
        frags[5:10, 5:10] = 1
        with pytest.raises(ValueError, match="Must provide"):
            prescreen_fragments(cost, frags, r_screen=3)

    def test_prescreen_screened_labels_zeroed(self):
        """Rejected fragments are zeroed in screened_fragment_labels."""
        cost = np.full((50, 50), 100.0, dtype=np.float32)
        frags = np.zeros((50, 50), dtype=np.int32)
        frags[5:10, 5:10] = 1
        result = prescreen_fragments(
                cost, frags, r_screen=3, tau_screen=1.0
        )
        assert result.screened_fragment_labels.max() == 0

    def test_prescreen_with_calibration_values(self):
        """Screening threshold derived from calibration_cost_values works."""
        cost = np.full((50, 50), 5.0, dtype=np.float32)
        frags = np.zeros((50, 50), dtype=np.int32)
        frags[5:10, 5:10] = 1
        cal_values = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        result = prescreen_fragments(
                cost, frags, r_screen=3,
                calibration_cost_values=cal_values,
                calibration_percentile=99.0,
        )
        assert isinstance(result, PrescreenResult)


# =====================================================================
# TestPathQuality
# =====================================================================


class TestPathQuality:
    """Tests for _path_quality.py functions."""

    # -- compute_path_metrics --

    def test_uniform_cost_path_median(self):
        """Path on uniform cost surface has median_raw_cost equal to that cost."""
        cost_surface = np.full((100, 100), 5.0, dtype=np.float32)
        coords = np.column_stack([np.arange(50), np.zeros(50)]).astype(np.int32)
        path = _DuckPath(coords, np.ones(50), total_cost=50.0, path_length=50)
        metrics = compute_path_metrics(path, cost_surface, window_cost=30)
        assert_allclose(metrics.median_raw_cost, 5.0)

    def test_short_path_window_cost_equals_median(self):
        """Path shorter than window_cost uses whole-path median."""
        cost_surface = np.full((100, 100), 3.0, dtype=np.float32)
        coords = np.column_stack([np.arange(10), np.zeros(10)]).astype(np.int32)
        path = _DuckPath(coords, np.ones(10), total_cost=10.0, path_length=10)
        metrics = compute_path_metrics(path, cost_surface, window_cost=30)
        assert_allclose(metrics.max_window_cost, metrics.median_raw_cost)

    def test_metrics_returns_all_five_fields(self):
        """PathMetrics has all five structure-based fields."""
        cost_surface = np.ones((100, 100), dtype=np.float32)
        coords = np.column_stack([np.arange(20), np.zeros(20)]).astype(np.int32)
        path = _DuckPath(coords, np.ones(20), total_cost=20.0, path_length=20)
        metrics = compute_path_metrics(path, cost_surface)
        assert hasattr(metrics, "median_raw_cost")
        assert hasattr(metrics, "max_window_cost")
        assert hasattr(metrics, "band_cost_variance")
        assert hasattr(metrics, "pct_energy_band_median")
        assert hasattr(metrics, "gray_band_snr")

    def test_pct_energy_zero_when_not_provided(self):
        """PCT energy band median is 0 when pct_energy is None."""
        cost_surface = np.ones((100, 100), dtype=np.float32)
        coords = np.column_stack([np.arange(20), np.zeros(20)]).astype(np.int32)
        path = _DuckPath(coords, np.ones(20), total_cost=20.0, path_length=20)
        metrics = compute_path_metrics(path, cost_surface, pct_energy=None)
        assert metrics.pct_energy_band_median == 0.0
        assert metrics.gray_band_snr == 0.0

    # -- calibrate_thresholds --

    def test_calibrate_thresholds_from_data(self):
        """Produces valid FilterThresholds from calibration data."""
        cal = CalibrationData(
                median_cost_values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                max_window_cost_values=np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
                band_variance_values=np.array([0.01, 0.02, 0.05, 0.1, 0.2]),
                pct_energy_median_values=np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
                gray_snr_values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        thresholds = calibrate_thresholds(cal, k=3.0)
        assert isinstance(thresholds, FilterThresholds)
        assert thresholds.tau_median_cost > 0
        assert thresholds.tau_window_cost > 0
        assert thresholds.k_iqr == 3.0

    # -- apply_filter_cascade --

    def test_filter_rejects_high_cost_path(self):
        """Path on high-cost surface is rejected by F1."""
        cost_surface = np.full((100, 100), 1000.0, dtype=np.float32)
        coords = np.column_stack([np.arange(50), np.zeros(50)]).astype(np.int32)
        path = _DuckPath(coords, np.ones(50), total_cost=5000.0, path_length=50)
        thresholds = FilterThresholds(
                tau_median_cost=10.0,
                tau_window_cost=10.0,
                tau_band_variance=1e6,
                tau_pct_energy_median=-1e6,
                tau_gray_snr=-1e6,
                k_iqr=3.0,
        )
        result = apply_filter_cascade({1: path}, cost_surface, thresholds)
        assert 1 in result.rejected_ids
        assert "F1_median_cost" in result.per_filter_rejections
        assert 1 in result.per_filter_rejections["F1_median_cost"]

    def test_filter_passes_low_cost_path(self):
        """Path on low-cost surface passes all filters."""
        cost_surface = np.full((100, 100), 0.1, dtype=np.float32)
        coords = np.column_stack([np.arange(50), np.zeros(50)]).astype(np.int32)
        path = _DuckPath(coords, np.ones(50), total_cost=50.0, path_length=50)
        thresholds = FilterThresholds(
                tau_median_cost=100.0,
                tau_window_cost=100.0,
                tau_band_variance=1e6,
                tau_pct_energy_median=-1e6,
                tau_gray_snr=-1e6,
                k_iqr=3.0,
        )
        result = apply_filter_cascade({1: path}, cost_surface, thresholds)
        assert 1 in result.passed_ids
        assert 1 not in result.rejected_ids


# =====================================================================
# TestDijkstraKernels
# =====================================================================


class TestDijkstraKernels:
    """Tests for _dijkstra_kernels.py public functions."""

    # -- run_multisource_dijkstra --

    def test_cost_distance_zero_inside_colonies(self, dijkstra_result):
        """Cost distance is 0 inside colony pixels."""
        two_colony_scene_data = dijkstra_result
        # Colony pixels were seeded at 0
        assert_allclose(
                two_colony_scene_data.cost_distance[
                    two_colony_scene_data.colony_id > 0
                    ].min(),
                0.0,
        )

    def test_cost_distance_positive_outside_colonies(
            self, two_colony_scene, dijkstra_result
    ):
        """Cost distance is >0 outside colony labels."""
        colony_labels, _, _ = two_colony_scene
        outside = colony_labels == 0
        outside_costs = dijkstra_result.cost_distance[outside]
        # Some pixels outside colonies should have positive cost
        assert np.any(outside_costs > 0)

    def test_colony_id_propagated(self, two_colony_scene, dijkstra_result):
        """Colony ID propagates to all reachable pixels."""
        colony_labels, _, _ = two_colony_scene
        # Fragment region (25:30, 35:40) should be reached by colony 1
        region_ids = dijkstra_result.colony_id[25:30, 35:40]
        assert np.all(region_ids > 0), "Fragment region should be reached"

    def test_colony_centroids_present(self, dijkstra_result):
        """Colony centroids dict has entries for both colonies."""
        assert 1 in dijkstra_result.colony_centroids
        assert 2 in dijkstra_result.colony_centroids

    def test_predecessor_minus_one_inside_colonies(
            self, two_colony_scene, dijkstra_result
    ):
        """Colony interior pixels have predecessor -1."""
        colony_labels, _, _ = two_colony_scene
        # Pixels deep inside colony 1 (not on boundary) should have pred = -1
        # The center pixel (20, 20) should definitely be interior
        assert dijkstra_result.predecessor[20, 20] == -1

    # -- assign_fragments_to_colonies --

    def test_fragment_assigned_to_nearest_colony(
            self, two_colony_scene, dijkstra_result
    ):
        """Fragment near colony 1 gets assigned to colony 1."""
        _, fragment_labels, _ = two_colony_scene
        assignments = assign_fragments_to_colonies(
                fragment_labels,
                dijkstra_result.colony_id,
                dijkstra_result.cost_distance,
        )
        assert 1 in assignments
        assert assignments[1].colony_id == 1

    def test_unreached_fragment_gets_minus_one(self, dijkstra_result):
        """Fragment at unreachable location gets colony_id=-1."""
        # Create a fragment where colony_id is -1 everywhere
        fake_frags = np.zeros((100, 100), dtype=np.int32)
        fake_frags[0, 0] = 1
        fake_colony_id = np.full((100, 100), -1, dtype=np.int32)
        fake_cost_dist = np.full((100, 100), np.inf, dtype=np.float64)
        assignments = assign_fragments_to_colonies(
                fake_frags, fake_colony_id, fake_cost_dist
        )
        assert assignments[1].colony_id == -1
        assert assignments[1].majority_fraction == 0.0

    # -- backtrack_path --

    def test_backtrack_unreached_pixel_returns_none(self, dijkstra_result):
        """Backtracking from an unreached pixel returns None."""
        # Use a pixel that is unreached (cost_distance = inf)
        pred = dijkstra_result.predecessor
        cd = dijkstra_result.cost_distance
        cost = np.ones((100, 100), dtype=np.float32)
        # Set up a guaranteed unreached pixel
        cd_copy = cd.copy()
        cd_copy[99, 0] = np.inf
        result = backtrack_path(99, 0, pred, cd_copy, cost)
        assert result is None

    def test_backtrack_reached_pixel_returns_path(
            self, two_colony_scene, dijkstra_result
    ):
        """Backtracking from a reached pixel yields a valid path to colony."""
        _, _, cost_surface = two_colony_scene
        # Pick a pixel in the fragment region that should be reached
        seed_r, seed_c = 27, 37
        if dijkstra_result.cost_distance[seed_r, seed_c] < np.inf:
            result = backtrack_path(
                    seed_r,
                    seed_c,
                    dijkstra_result.predecessor,
                    dijkstra_result.cost_distance,
                    cost_surface,
            )
            assert result is not None
            coords, cost_profile = result
            assert coords.shape[1] == 2
            assert len(cost_profile) == len(coords)
            # Path should end at a colony (cost_distance = 0)
            assert_allclose(cost_profile[-1], 0.0)

    # -- extract_fragment_paths --

    def test_extract_paths_returns_valid_paths(
            self, two_colony_scene, dijkstra_result
    ):
        """Extract paths from fragments to colonies."""
        colony_labels, fragment_labels, cost_surface = two_colony_scene
        assignments = assign_fragments_to_colonies(
                fragment_labels,
                dijkstra_result.colony_id,
                dijkstra_result.cost_distance,
        )
        paths, unconnected = extract_fragment_paths(
                fragment_labels, assignments, dijkstra_result, cost_surface
        )
        # Fragment 1 should have a path to colony 1
        assert 1 in paths
        assert paths[1].colony_id == 1
        assert paths[1].path_length > 0
        assert len(unconnected) == 0

    # -- assemble_connected_mask --

    def test_assemble_paints_fragment_and_path(
            self, two_colony_scene, dijkstra_result
    ):
        """Fragment and path pixels painted with colony label."""
        colony_labels, fragment_labels, cost_surface = two_colony_scene
        assignments = assign_fragments_to_colonies(
                fragment_labels,
                dijkstra_result.colony_id,
                dijkstra_result.cost_distance,
        )
        paths, _ = extract_fragment_paths(
                fragment_labels, assignments, dijkstra_result, cost_surface
        )
        connected = assemble_connected_mask(
                colony_labels, fragment_labels, assignments, paths
        )
        # Fragment pixels should now have colony 1 label
        assert np.all(connected[25:30, 35:40] == 1)
        # Original colony pixels should be unchanged
        assert_array_equal(
                connected[colony_labels == 2],
                colony_labels[colony_labels == 2],
        )


# =====================================================================
# TestDataclasses
# =====================================================================


class TestDataclasses:
    """Basic instantiation tests for reconnection dataclasses."""

    def test_dijkstra_result_creation(self):
        """DijkstraResult can be created with valid data."""
        dr = DijkstraResult(
                cost_distance=np.zeros((5, 5), dtype=np.float64),
                colony_id=np.ones((5, 5), dtype=np.int32),
                predecessor=np.full((5, 5), -1, dtype=np.int32),
                colony_centroids={1: (2.5, 2.5)},
        )
        assert dr.cost_distance.shape == (5, 5)

    def test_fragment_assignment_creation(self):
        """FragmentAssignment can be created with valid data."""
        fa = FragmentAssignment(
                fragment_id=1,
                colony_id=3,
                is_bridge=False,
                majority_fraction=0.95,
                mean_cost=1.5,
        )
        assert fa.fragment_id == 1
        assert fa.colony_id == 3

    def test_fragment_path_creation(self):
        """FragmentPath can be created with valid data."""
        fp = FragmentPath(
                fragment_id=1,
                colony_id=2,
                coords=np.array([[0, 0], [1, 1]], dtype=np.int32),
                cost_profile=np.array([1.0, 0.0]),
                total_cost=1.0,
                path_length=2,
        )
        assert fp.path_length == 2

    def test_prescreen_result_creation(self):
        """PrescreenResult can be created with valid data."""
        pr = PrescreenResult(
                screened_fragment_labels=np.zeros((5, 5), dtype=np.int32),
                passed_ids={1, 2},
                rejected_ids={3},
                threshold_used=0.5,
        )
        assert len(pr.passed_ids) == 2

    def test_path_metrics_creation(self):
        """PathMetrics can be created with valid data."""
        pm = PathMetrics(
                median_raw_cost=1.5,
                max_window_cost=2.0,
                band_cost_variance=0.05,
                pct_energy_band_median=0.7,
                gray_band_snr=3.0,
        )
        assert pm.median_raw_cost == 1.5

    def test_calibration_data_creation(self):
        """CalibrationData can be created with valid data."""
        cd = CalibrationData(
                median_cost_values=np.array([1.0, 2.0]),
                max_window_cost_values=np.array([1.5, 2.5]),
                band_variance_values=np.array([0.01, 0.02]),
                pct_energy_median_values=np.array([0.8, 0.9]),
                gray_snr_values=np.array([3.0, 4.0]),
        )
        assert len(cd.median_cost_values) == 2

    def test_filter_thresholds_creation(self):
        """FilterThresholds can be created with valid data."""
        ft = FilterThresholds(
                tau_median_cost=5.0,
                tau_window_cost=6.0,
                tau_band_variance=0.5,
                tau_pct_energy_median=0.1,
                tau_gray_snr=0.5,
                k_iqr=3.0,
        )
        assert ft.k_iqr == 3.0

    def test_filter_result_creation(self):
        """FilterResult can be created with valid data."""
        fr = FilterResult(
                passed_ids={1, 2},
                rejected_ids={3},
                per_filter_rejections={"F1_median_cost": {3}},
                metrics={},
                thresholds=FilterThresholds(5.0, 6.0, 0.5, 0.1, 0.5, 3.0),
        )
        assert 3 in fr.rejected_ids


# =====================================================================
# TestExtractCalibrationBranches
# =====================================================================


class TestExtractCalibrationBranches:
    """Tests for extract_calibration_branches from _path_quality.py."""

    def test_returns_calibration_data(self):
        """Produces CalibrationData from a synthetic colony with branches."""
        # Create a colony with a long linear branch (skeleton-like)
        colony = np.zeros((80, 80), dtype=np.int32)
        colony[30:50, 30:50] = 1  # solid block
        colony[40, 50:75] = 1  # horizontal branch (25 px)
        cost = np.ones((80, 80), dtype=np.float32)
        cal = extract_calibration_branches(
                colony, cost, min_branch_length=5
        )
        assert isinstance(cal, CalibrationData)

    def test_empty_when_branches_too_short(self):
        """Returns empty arrays when all branches are shorter than minimum."""
        colony = np.zeros((30, 30), dtype=np.int32)
        colony[10:15, 10:15] = 1  # small blob
        cost = np.ones((30, 30), dtype=np.float32)
        cal = extract_calibration_branches(
                colony, cost, min_branch_length=100
        )
        assert cal.median_cost_values.size == 0


# =====================================================================
# TestFilamentousFungiDetectorReconnection
# =====================================================================


class TestFilamentousFungiDetectorReconnection:
    """Integration and backward-compatibility tests for reconnection."""

    def test_enable_reconnection_true_runs_without_error(self, synth_plate):
        """Reconnection mode runs without errors on synth plate.

        The synth plate has circular yeast colonies, so there will not be
        realistic filamentous reconnection, but the full code path must
        execute without raising.
        """
        from phenotypic.detect import (
            FilamentousFungiDetector,
            OtsuDetector,
            TriangleDetector,
        )

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(ignore_zeros=True),
                overall_detector=TriangleDetector(),
                enable_reconnection=True,
        )
        result = detector.apply(synth_plate.copy())

        assert result.objmask[:].sum() > 0
        assert result.objmap[:].max() > 0

    def test_reconnection_produces_valid_objmap(self, synth_plate):
        """Reconnection output has consistent objmask/objmap."""
        from phenotypic.detect import (
            FilamentousFungiDetector,
            OtsuDetector,
            TriangleDetector,
        )

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(ignore_zeros=True),
                overall_detector=TriangleDetector(),
                enable_reconnection=True,
        )
        result = detector.apply(synth_plate.copy())

        objmask = result.objmask[:]
        objmap = result.objmap[:]
        # All non-zero objmap pixels should be True in objmask
        assert_array_equal(objmap > 0, objmask)

    def test_reconnection_serialization_roundtrip(self):
        """Detector with reconnection serializes and restores correctly."""
        from phenotypic import ImagePipeline
        from phenotypic.detect import (
            FilamentousFungiDetector,
            OtsuDetector,
            TriangleDetector,
        )

        detector = FilamentousFungiDetector(
                inoculum_detector=OtsuDetector(ignore_zeros=True),
                overall_detector=TriangleDetector(),
                enable_reconnection=True,
                delta=2.0,
                r_screen=15,
        )
        pipeline = ImagePipeline([detector])
        json_str = pipeline.to_json()
        restored = ImagePipeline.from_json(json_str)

        restored_det = list(restored._ops.values())[0]
        assert isinstance(restored_det, FilamentousFungiDetector)
        assert restored_det.enable_reconnection is True
        assert restored_det.delta == 2.0
        assert restored_det.r_screen == 15
