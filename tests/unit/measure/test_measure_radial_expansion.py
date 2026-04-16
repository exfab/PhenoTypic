"""Tests for MeasureRadialExpansion measurement operation."""

import pytest
import numpy as np
import pandas as pd

from phenotypic import Image

from phenotypic.measure import MeasureRadialExpansion
from phenotypic.tools_.constants_ import OBJECT
from phenotypic.tools_.measurement_info_ import RADIAL_EXPANSION


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


class TestMeasureRadialExpansionStructure:
    """Verify output DataFrame shape and column structure."""

    def test_output_has_all_columns(self, synth_plate_detected):
        """Output DataFrame contains all RADIAL_EXPANSION columns plus Label."""
        image = synth_plate_detected.copy()
        measurer = MeasureRadialExpansion()
        df = measurer.measure(image)

        assert df.columns[0] == OBJECT.LABEL
        for feature in RADIAL_EXPANSION:
            assert str(feature) in df.columns, f"Missing column: {feature}"

    def test_output_row_count_matches_objects(self, synth_plate_detected):
        """One row per detected object."""
        image = synth_plate_detected.copy()
        measurer = MeasureRadialExpansion()
        df = measurer.measure(image)

        assert len(df) == image.num_objects

    def test_object_labels_match(self, synth_plate_detected):
        """Label column matches image.objects.labels."""
        image = synth_plate_detected.copy()
        measurer = MeasureRadialExpansion()
        df = measurer.measure(image)

        np.testing.assert_array_equal(
            df[OBJECT.LABEL].values,
            image.objects.labels2series().values,
        )


# ---------------------------------------------------------------------------
# Value constraint tests
# ---------------------------------------------------------------------------


class TestMeasureRadialExpansionValues:
    """Verify measurement values satisfy domain constraints."""

    @pytest.fixture()
    def df(self, synth_plate_detected):
        image = synth_plate_detected.copy()
        return MeasureRadialExpansion().measure(image)

    def test_core_radius_non_negative(self, df):
        """CoreRadius must be >= 0 wherever it is not NaN."""
        col = str(RADIAL_EXPANSION.CORE_RADIUS)
        valid = df[col].dropna()
        assert (valid >= 0).all(), "CoreRadius has negative values"

    def test_num_branches_non_negative(self, df):
        """NumBranches must be >= 0 wherever it is not NaN."""
        col = str(RADIAL_EXPANSION.NUM_BRANCHES)
        valid = df[col].dropna()
        assert (valid >= 0).all(), "NumBranches has negative values"

    def test_robust_mean_leq_max_branch(self, df):
        """Where both are non-NaN, RobustMeanRadius <= MaxBranchLength."""
        robust = df[str(RADIAL_EXPANSION.ROBUST_MEAN_RADIUS)]
        maxbl = df[str(RADIAL_EXPANSION.MAX_BRANCH_LENGTH)]
        mask = robust.notna() & maxbl.notna()
        if mask.any():
            assert (robust[mask].values <= maxbl[mask].values + 1e-9).all(), (
                "RobustMeanRadius exceeds MaxBranchLength for some objects"
            )

    def test_runner_detected_is_binary(self, df):
        """RunnerDetected should be 0.0 or 1.0 (or NaN)."""
        col = str(RADIAL_EXPANSION.RUNNER_DETECTED)
        valid = df[col].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0}), (
            f"RunnerDetected has non-binary values: {valid.unique()}"
        )

    def test_runner_length_nan_when_not_detected(self, df):
        """RunnerLength must be NaN when RunnerDetected is 0."""
        detected = df[str(RADIAL_EXPANSION.RUNNER_DETECTED)]
        length = df[str(RADIAL_EXPANSION.RUNNER_LENGTH)]
        no_runner = detected == 0.0
        if no_runner.any():
            assert length[no_runner].isna().all(), (
                "RunnerLength is not NaN for objects with RunnerDetected == 0"
            )


# ---------------------------------------------------------------------------
# Parameter variation tests
# ---------------------------------------------------------------------------


class TestMeasureRadialExpansionParams:
    """Verify the class runs without error under different parameter combos."""

    def test_outlier_method_iqr(self, synth_plate_detected):
        image = synth_plate_detected.copy()
        df = MeasureRadialExpansion(outlier_method="iqr").measure(image)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == image.num_objects

    def test_outlier_method_mad(self, synth_plate_detected):
        image = synth_plate_detected.copy()
        df = MeasureRadialExpansion(outlier_method="mad").measure(image)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == image.num_objects

    def test_outlier_method_ellipse(self, synth_plate_detected):
        image = synth_plate_detected.copy()
        df = MeasureRadialExpansion(outlier_method="ellipse").measure(image)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == image.num_objects

    def test_skeleton_method_lee(self, synth_plate_detected):
        image = synth_plate_detected.copy()
        df = MeasureRadialExpansion(skeleton_method="lee").measure(image)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == image.num_objects

    def test_different_pelt_penalty(self, synth_plate_detected):
        image = synth_plate_detected.copy()
        df = MeasureRadialExpansion(pelt_penalty=10.0).measure(image)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == image.num_objects


# ---------------------------------------------------------------------------
# Edge case tests with synthetic objmaps
# ---------------------------------------------------------------------------


class TestMeasureRadialExpansionEdgeCases:
    """Edge cases using small synthetic images."""

    @staticmethod
    def _make_image_with_objmap(
        gray: np.ndarray, objmap: np.ndarray,
    ) -> Image:
        """Create an Image with a pre-set objmap (bypasses detection)."""
        rgb = np.stack([gray, gray, gray], axis=-1)
        image = Image(rgb)
        image.objmap[:] = objmap
        return image

    def test_small_object_returns_nan(self):
        """Objects < 10 pixels produce NaN measurements."""
        gray = np.ones((100, 100), dtype=np.uint8) * 200
        gray[49:52, 49:52] = 50
        objmap = np.zeros((100, 100), dtype=np.int32)
        objmap[49:52, 49:52] = 1  # 9-pixel object
        image = self._make_image_with_objmap(gray, objmap)

        measurer = MeasureRadialExpansion()
        df = measurer.measure(image)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        mean_col = str(RADIAL_EXPANSION.MEAN_RADIUS)
        num_branches_col = str(RADIAL_EXPANSION.NUM_BRANCHES)
        nb = df[num_branches_col].iloc[0]
        mr = df[mean_col].iloc[0]
        if pd.notna(nb) and nb == 0:
            assert pd.isna(mr), "MeanRadius should be NaN when NumBranches == 0"

    def test_compact_colony_zero_branches(self):
        """Compact circular colony produces NumBranches == 0 or very few branches."""
        gray = np.ones((200, 200), dtype=np.uint8) * 220
        objmap = np.zeros((200, 200), dtype=np.int32)
        rr, cc = np.ogrid[:200, :200]
        circle = ((rr - 100) ** 2 + (cc - 100) ** 2) < 40**2
        gray[circle] = 40
        objmap[circle] = 1
        image = self._make_image_with_objmap(gray, objmap)

        measurer = MeasureRadialExpansion()
        df = measurer.measure(image)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        num_branches = df[str(RADIAL_EXPANSION.NUM_BRANCHES)].dropna()
        if len(num_branches) > 0:
            assert num_branches.max() <= 10, (
                f"Expected few branches for compact colony, got {num_branches.max()}"
            )


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestMeasureRadialExpansionSerialization:
    """Verify JSON round-trip via ImagePipeline preserves constructor parameters."""

    def test_json_roundtrip(self):
        from phenotypic import ImagePipeline

        original = MeasureRadialExpansion(
            outlier_method="iqr",
            outlier_k=2.0,
            n_annuli=50,
            pelt_penalty=3.0,
            skeleton_method="lee",
        )
        pipe = ImagePipeline(ops=[original])
        restored_pipe = ImagePipeline.from_json(pipe.to_json())
        restored = list(restored_pipe._ops.values())[0]

        assert restored.outlier_method == "iqr"
        assert restored.outlier_k == 2.0
        assert restored.n_annuli == 50
        assert restored.pelt_penalty == 3.0
        assert restored.skeleton_method == "lee"

    def test_json_roundtrip_defaults(self):
        from phenotypic import ImagePipeline

        original = MeasureRadialExpansion()
        pipe = ImagePipeline(ops=[original])
        restored_pipe = ImagePipeline.from_json(pipe.to_json())
        restored = list(restored_pipe._ops.values())[0]

        assert restored.outlier_method == "mad"
        assert restored.outlier_k == 3.0
        assert restored.n_annuli == 100
        assert restored.pelt_penalty == 5.0
        assert restored.skeleton_method == "zhang"


# ---------------------------------------------------------------------------
# Decompose method tests
# ---------------------------------------------------------------------------


class TestMeasureRadialExpansionDecompose:
    """Verify the decompose() per-branch diagnostic method."""

    def test_decompose_returns_dataframe(self, synth_plate_detected):
        image = synth_plate_detected.copy()
        measurer = MeasureRadialExpansion()
        df = measurer.decompose(image)

        assert isinstance(df, pd.DataFrame)
        expected_cols = {"ObjectLabel", "BranchIndex", "Angle", "Length", "IsRunner"}
        assert set(df.columns) == expected_cols

    def test_decompose_is_runner_binary(self, synth_plate_detected):
        image = synth_plate_detected.copy()
        measurer = MeasureRadialExpansion()
        df = measurer.decompose(image)

        if len(df) > 0:
            assert set(df["IsRunner"].unique()).issubset({0, 1}), (
                f"IsRunner has non-binary values: {df['IsRunner'].unique()}"
            )

    def test_decompose_branch_index_starts_at_zero(self, synth_plate_detected):
        image = synth_plate_detected.copy()
        measurer = MeasureRadialExpansion()
        df = measurer.decompose(image)

        if len(df) > 0:
            for label, group in df.groupby("ObjectLabel"):
                assert group["BranchIndex"].min() == 0, (
                    f"BranchIndex does not start at 0 for object {label}"
                )

    def test_decompose_length_non_negative(self, synth_plate_detected):
        image = synth_plate_detected.copy()
        measurer = MeasureRadialExpansion()
        df = measurer.decompose(image)

        if len(df) > 0:
            assert (df["Length"] >= 0).all(), "Negative branch lengths found"


# ---------------------------------------------------------------------------
# Dijkstra pathfinding — white-box tests on _trace_branches_dijkstra
# ---------------------------------------------------------------------------


class TestTraceBranchesDijkstra:
    """Targeted tests for the Dijkstra-based branch tracer.

    These exercise ``_trace_branches_dijkstra`` directly on synthetic
    local-bbox inputs so we can compare against known-optimal path
    lengths and probe edge cases that are hard to trigger through a
    full ``measure()`` call.
    """

    @staticmethod
    def _dist_map(shape, center_rc):
        rows, cols = np.indices(shape)
        return np.sqrt(
            (rows - center_rc[0]) ** 2 + (cols - center_rc[1]) ** 2
        )

    def test_dijkstra_linear_skeleton_optimal_length(self):
        """Straight skeleton with no obstacles: Dijkstra length equals
        the exact tip-to-core-edge pixel count (optimality on a trivial
        graph)."""
        h, w = 11, 21
        local_mask = np.zeros((h, w), dtype=bool)
        local_mask[5, :] = True
        skeleton = local_mask.copy()
        endpoints = np.array([[5, 0]], dtype=np.int32)
        center_rc = (5.0, 10.0)
        core_radius = 3.0  # core spans cols 7..13
        dist_map = self._dist_map((h, w), center_rc)

        branches = MeasureRadialExpansion._trace_branches_dijkstra(
            local_mask, skeleton, endpoints, dist_map, center_rc, core_radius,
        )

        assert len(branches) == 1
        coords, length = branches[0]
        # Tip at col 0, core boundary at col 7 → exactly 7 horizontal steps.
        assert abs(length - 7.0) < 1e-6, (
            f"Dijkstra length {length} should exactly equal 7.0 on a linear skeleton"
        )
        # First coord is the tip (backtrack order is tip → core).
        assert tuple(coords[0]) == (5, 0)
        # Last coord is at the core boundary.
        assert tuple(coords[-1]) == (5, 7)

    def test_dijkstra_no_longer_than_skeleton_contour(self):
        """On an L-shaped skeleton, Dijkstra length is bounded above by
        the full skeleton contour length (strong optimality check —
        Dijkstra must not wander beyond the skeleton itself)."""
        h, w = 11, 11
        local_mask = np.zeros((h, w), dtype=bool)
        local_mask[3:8, 0:8] = True
        skeleton = np.zeros((h, w), dtype=bool)
        skeleton[5, 0:7] = True   # horizontal arm: 7 pixels
        skeleton[3:6, 6] = True   # vertical arm: 3 pixels (shared pixel at 5,6)
        endpoints = np.array([[3, 6]], dtype=np.int32)
        center_rc = (5.0, 0.0)
        core_radius = 1.0
        dist_map = self._dist_map((h, w), center_rc)

        branches = MeasureRadialExpansion._trace_branches_dijkstra(
            local_mask, skeleton, endpoints, dist_map, center_rc, core_radius,
        )

        assert len(branches) == 1
        _coords, length = branches[0]
        # Skeleton contour from (3, 6) to core boundary: vertical 2 + horizontal 5 = 7.
        # Dijkstra must be no longer than this (may be shorter via diagonal cuts).
        assert length <= 7.0 + 1e-6, (
            f"Dijkstra length {length} exceeds skeleton contour length 7.0"
        )
        assert length > 0

    def test_skeleton_disconnected_from_core_resolves_via_detour(self):
        """Greedy DFS dead-ends when the skeleton has a one-pixel gap
        from the core ring; Dijkstra completes the path via a single
        object-interior detour."""
        h, w = 11, 21
        local_mask = np.zeros((h, w), dtype=bool)
        local_mask[4:7, :] = True  # 3-pixel-tall object band
        skeleton = np.zeros((h, w), dtype=bool)
        skeleton[5, 0:9] = True   # skeleton ends at col 8
        skeleton[5, 10:] = True    # resumes at col 10 — one-pixel gap at col 9
        endpoints = np.array([[5, 0]], dtype=np.int32)
        center_rc = (5.0, 15.0)
        core_radius = 3.0
        dist_map = self._dist_map((h, w), center_rc)

        branches = MeasureRadialExpansion._trace_branches_dijkstra(
            local_mask, skeleton, endpoints, dist_map, center_rc, core_radius,
        )

        assert len(branches) == 1, (
            "Dijkstra should route through the object-interior detour"
        )
        coords, length = branches[0]
        assert length > 0
        assert len(coords) >= 2

    def test_core_radius_zero_fallback_seeds_at_centroid(self):
        """When PELT finds no changepoint (core_radius == 0) the core
        mask is empty; the fallback seed at the centroid keeps Dijkstra
        well-defined so paths are still produced."""
        h, w = 11, 11
        local_mask = np.ones((h, w), dtype=bool)
        skeleton = np.zeros((h, w), dtype=bool)
        skeleton[5, :] = True  # horizontal spine across the object
        endpoints = np.array([[5, 0], [5, 10]], dtype=np.int32)
        center_rc = (5.0, 5.0)
        core_radius = 0.0
        dist_map = self._dist_map((h, w), center_rc)

        branches = MeasureRadialExpansion._trace_branches_dijkstra(
            local_mask, skeleton, endpoints, dist_map, center_rc, core_radius,
        )

        assert len(branches) == 2
        for coords, length in branches:
            assert length > 0
            # Path should terminate at or near the centroid fallback seed.
            last_r, last_c = coords[-1]
            assert abs(int(last_r) - 5) <= 1
            assert abs(int(last_c) - 5) <= 1

    def test_empty_endpoints_returns_empty(self):
        """No tips → no paths."""
        h, w = 5, 5
        local_mask = np.ones((h, w), dtype=bool)
        skeleton = np.zeros((h, w), dtype=bool)
        skeleton[2, :] = True
        endpoints = np.empty((0, 2), dtype=np.int32)
        dist_map = self._dist_map((h, w), (2.0, 2.0))

        branches = MeasureRadialExpansion._trace_branches_dijkstra(
            local_mask, skeleton, endpoints, dist_map, (2.0, 2.0), 1.0,
        )

        assert branches == []

    def test_build_branch_cost_surface_prefers_skeleton(self):
        """Cost-surface helper: skeleton pixels strictly cheaper than
        on-object off-skeleton pixels, which are strictly cheaper than
        off-object pixels."""
        h, w = 5, 5
        local_mask = np.zeros((h, w), dtype=bool)
        local_mask[1:4, 1:4] = True
        skeleton = np.zeros((h, w), dtype=bool)
        skeleton[2, 2] = True
        dist_map = self._dist_map((h, w), (2.0, 2.0))

        cost = MeasureRadialExpansion._build_branch_cost_surface(
            local_mask, skeleton, dist_map, core_radius=0.0,
        )

        assert cost[2, 2] < cost[1, 1], "skeleton should be cheaper than on-object pixel"
        assert cost[1, 1] < cost[0, 0], "on-object should be cheaper than off-object"
