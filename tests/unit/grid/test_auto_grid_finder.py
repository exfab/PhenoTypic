"""Unit and integration tests for the center-based AutoGridFinder."""

import warnings

import numpy as np
import pandas as pd
import pytest

from phenotypic.grid import AutoGridFinder
from phenotypic.tools_.measurement_info import BBOX, GRID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_info_table(row_centers, col_centers):
    """Build a minimal info table for ``_fit_axis_edges`` unit tests.

    Only includes DIST_WEIGHTED_CENTER columns used by AutoGridFinder's
    fitting logic. Integration tests that call ``measure()`` use real
    images which provide the full schema (including CENTER_RR/CC for grid
    assignment).
    """
    return pd.DataFrame({
        str(BBOX.DIST_WEIGHTED_CENTER_RR): row_centers,
        str(BBOX.DIST_WEIGHTED_CENTER_CC): col_centers,
    })


# ===========================================================================
# _extract_axis_centers
# ===========================================================================


class TestExtractAxisCenters:

    def test_axis0_returns_sorted_row_centers(self):
        table = _make_info_table([50, 10, 30], [100, 200, 300])
        result = AutoGridFinder._extract_axis_centers(table, axis=0)
        np.testing.assert_array_equal(result, [10, 30, 50])

    def test_axis1_returns_sorted_col_centers(self):
        table = _make_info_table([10, 20, 30], [300, 100, 200])
        result = AutoGridFinder._extract_axis_centers(table, axis=1)
        np.testing.assert_array_equal(result, [100, 200, 300])

    def test_invalid_axis_raises(self):
        table = _make_info_table([10], [20])
        with pytest.raises(ValueError, match="axis must be 0 or 1"):
            AutoGridFinder._extract_axis_centers(table, axis=2)


# ===========================================================================
# _estimate_pitch
# ===========================================================================


class TestEstimatePitch:

    def test_uniform_spacing(self):
        centers = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
        # 5 centers spanning 4 intervals → pitch = 80/4 = 20
        assert AutoGridFinder._estimate_pitch(centers, n_expected=5) == pytest.approx(20.0)

    def test_range_based_with_many_objects(self):
        # Multiple objects per cell — range / (n_expected - 1) still works
        centers = np.sort(np.array([
            10.0, 11.0, 12.0,  # cell 0
            30.0, 31.0, 32.0,  # cell 1
            50.0, 51.0, 52.0,  # cell 2
            70.0, 71.0, 72.0,  # cell 3
        ]))
        pitch = AutoGridFinder._estimate_pitch(centers, n_expected=4)
        # range = 72-10 = 62; 62/3 ≈ 20.67 — close to true 20
        assert 18.0 < pitch < 25.0

    def test_too_few_centers_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            AutoGridFinder._estimate_pitch(np.array([42.0]), n_expected=8)


# ===========================================================================
# _assign_grid_indices
# ===========================================================================


class TestAssignGridIndices:

    def test_uniform_centers(self):
        centers = np.array([10.0, 30.0, 50.0, 70.0])
        indices = AutoGridFinder._assign_grid_indices(centers, pitch=20.0)
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])

    def test_gap_in_centers(self):
        # Missing one position in the middle
        centers = np.array([10.0, 30.0, 70.0])
        indices = AutoGridFinder._assign_grid_indices(centers, pitch=20.0)
        np.testing.assert_array_equal(indices, [0, 1, 3])

    def test_jittery_centers(self):
        centers = np.array([10.0, 31.0, 49.0, 72.0])
        indices = AutoGridFinder._assign_grid_indices(centers, pitch=20.0)
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])


# ===========================================================================
# _fit_pitch_and_offset
# ===========================================================================


class TestFitPitchAndOffset:

    def test_perfect_grid(self):
        centers = np.array([10.0, 30.0, 50.0, 70.0])
        indices = np.array([0, 1, 2, 3])
        pitch, offset = AutoGridFinder._fit_pitch_and_offset(centers, indices)
        assert pitch == pytest.approx(20.0)
        assert offset == pytest.approx(10.0)

    def test_noisy_grid(self):
        rng = np.random.default_rng(42)
        true_pitch, true_offset = 25.0, 15.0
        indices = np.arange(8)
        centers = true_pitch * indices + true_offset + rng.normal(0, 1, 8)
        pitch, offset = AutoGridFinder._fit_pitch_and_offset(centers, indices)
        assert pitch == pytest.approx(true_pitch, abs=2.0)
        assert offset == pytest.approx(true_offset, abs=3.0)

    def test_all_same_index_returns_zero_pitch(self):
        """When all indices are identical, pitch is 0 and offset is the mean center."""
        centers = np.array([10.0, 11.0, 12.0])
        indices = np.array([0, 0, 0])
        pitch, offset = AutoGridFinder._fit_pitch_and_offset(centers, indices)
        assert pitch == 0.0
        assert offset == pytest.approx(11.0)


# ===========================================================================
# _identify_inliers
# ===========================================================================


class TestIdentifyInliers:

    def test_all_inliers(self):
        centers = np.array([10.0, 30.0, 50.0, 70.0])
        indices = np.array([0, 1, 2, 3])
        mask = AutoGridFinder._identify_inliers(
            centers, indices, pitch=20.0, offset=10.0, threshold=5.0,
        )
        assert mask.all()

    def test_one_outlier(self):
        centers = np.array([10.0, 30.0, 50.0, 100.0])  # last is outlier
        indices = np.array([0, 1, 2, 3])
        mask = AutoGridFinder._identify_inliers(
            centers, indices, pitch=20.0, offset=10.0, threshold=5.0,
        )
        np.testing.assert_array_equal(mask, [True, True, True, False])


# ===========================================================================
# _compute_grid_edges
# ===========================================================================


class TestComputeGridEdges:

    def test_correct_count(self):
        edges = AutoGridFinder._compute_grid_edges(
            pitch=20.0, offset=10.0, n_bins=8, image_dim=200,
        )
        assert len(edges) == 9

    def test_clipping(self):
        # Offset that would produce negative first edge
        edges = AutoGridFinder._compute_grid_edges(
            pitch=20.0, offset=5.0, n_bins=8, image_dim=100,
        )
        assert edges[0] >= 0
        assert edges[-1] <= 100

    def test_sorted_ascending(self):
        edges = AutoGridFinder._compute_grid_edges(
            pitch=25.0, offset=15.0, n_bins=12, image_dim=400,
        )
        assert np.all(np.diff(edges) >= 0)

    def test_negative_offset_no_duplicate_zeros(self):
        """Deeply negative offset must not produce duplicate 0 edges."""
        edges = AutoGridFinder._compute_grid_edges(
            pitch=437, offset=-229.5, n_bins=12, image_dim=4776,
        )
        assert len(edges) == len(set(edges)), f"Duplicate edges: {edges}"
        assert np.all(np.diff(edges) > 0)

    def test_pitch_too_large_shrinks_to_fit(self):
        """When pitch * n_bins > image_dim, pitch is reduced to fit."""
        edges = AutoGridFinder._compute_grid_edges(
            pitch=500, offset=0, n_bins=12, image_dim=4776,
        )
        assert len(edges) == len(set(edges)), f"Duplicate edges: {edges}"
        assert edges[0] == 0
        assert edges[-1] == 4776
        assert np.all(np.diff(edges) > 0)

    def test_offset_clamped_left(self):
        """Offset that would push first edge negative gets clamped."""
        edges = AutoGridFinder._compute_grid_edges(
            pitch=100, offset=-50, n_bins=4, image_dim=500,
        )
        assert edges[0] >= 0
        assert len(edges) == len(set(edges))
        assert np.all(np.diff(edges) > 0)

    def test_offset_clamped_right(self):
        """Offset that would push last edge past image_dim gets clamped."""
        edges = AutoGridFinder._compute_grid_edges(
            pitch=100, offset=400, n_bins=4, image_dim=450,
        )
        assert edges[-1] <= 450
        assert len(edges) == len(set(edges))
        assert np.all(np.diff(edges) > 0)


# ===========================================================================
# _fit_axis_edges  (instance method)
# ===========================================================================


class TestFitAxisEdges:

    def _make_finder(self, **kwargs):
        return AutoGridFinder(nrows=8, ncols=12, **kwargs)

    def test_perfect_grid(self):
        finder = self._make_finder()
        pitch, offset = 25.0, 12.5
        row_centers = [offset + pitch * i for i in range(8)]
        col_centers = list(range(8))  # dummy
        table = _make_info_table(row_centers, col_centers)
        edges = finder._fit_axis_edges(table, axis=0, n_expected=8, image_dim=200)
        assert len(edges) == 9
        assert edges[0] >= 0
        assert edges[-1] <= 200

    def test_with_outlier(self):
        finder = self._make_finder()
        pitch, offset = 25.0, 12.5
        row_centers = [offset + pitch * i for i in range(8)]
        row_centers[-1] += 50  # big outlier
        col_centers = list(range(8))
        table = _make_info_table(row_centers, col_centers)
        edges = finder._fit_axis_edges(table, axis=0, n_expected=8, image_dim=250)
        # Should still produce valid edges
        assert len(edges) == 9
        assert np.all(np.diff(edges) >= 0)

    def test_few_objects_fallback(self):
        finder = self._make_finder()
        table = _make_info_table([50.0], [100.0])
        edges = finder._fit_axis_edges(table, axis=0, n_expected=8, image_dim=200)
        assert len(edges) == 9
        # Uniform spacing fallback
        expected_pitch = 200 / 8
        actual_diffs = np.diff(edges)
        assert np.all(actual_diffs > 0)
        np.testing.assert_allclose(actual_diffs, expected_pitch, atol=1.5)

    def test_span_within_tolerance_no_anchor(self):
        """One entirely-missing edge cell (n_missing == 1) is within _SPAN_TOLERANCE.

        No ``[span-coverage-*]`` warning should fire; the fitted offset
        flows through unchanged.
        """
        finder = AutoGridFinder(nrows=12, ncols=12)
        # 11 of 12 expected cells detected, missing the last one.
        # pitch=100, offset=50; cells at 50, 150, ..., 1050.
        row_centers = [50.0 + 100.0 * i for i in range(11)]
        col_centers = list(range(11))
        table = _make_info_table(row_centers, col_centers)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            edges = finder._fit_axis_edges(
                table, axis=0, n_expected=12, image_dim=1200,
            )
        assert len(edges) == 13
        span_warnings = [
            x for x in w if "span-coverage" in str(x.message)
        ]
        assert len(span_warnings) == 0, (
            f"Expected no span-coverage warning, got: "
            f"{[str(x.message) for x in span_warnings]}"
        )

    def test_span_coverage_low_anchor_keeps_fit(self):
        """Direct unit test of [span-coverage-low] branch.

        After moving the iterative seed to span-based pitch
        (:meth:`_estimate_pitch` + :meth:`_choose_seed_anchor`), the
        fit places ``c_min`` at the predicted cell-0 center *and*
        ``c_max`` at the predicted cell-(n-1) center by construction
        — so the one-sided low/high anchor branches are unreachable
        through ``_fit_axis_edges`` for in-distribution inputs. They
        remain valid defensive code paths for post-OLS drift, so we
        test the low branch directly: pass a synthetic fit where
        ``c_min`` lines up with predicted cell-0 but ``c_max`` is far
        from predicted cell-(n-1).
        """
        centers = np.array([50.0, 150.0, 250.0, 350.0, 450.0, 550.0])
        pitch = 100.0
        offset = 50.0  # predicted cell-0 at 50; predicted cell-11 at 1150
        span = 6
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            new_offset = AutoGridFinder._resolve_span_anchor(
                centers, pitch, offset, n_expected=12, span=span,
                axis_label="rows", span_word="row",
            )
        assert new_offset == offset  # fit preserved (Option (i))
        msgs = [str(x.message) for x in w]
        low_count = sum("[span-coverage-low]" in m for m in msgs)
        other_count = sum(
            ("[span-coverage-high]" in m or "[span-coverage-symmetric]" in m)
            for m in msgs
        )
        assert low_count == 1, f"Expected 1 low-anchor warning, got: {msgs}"
        assert other_count == 0, f"Unexpected other span warning: {msgs}"

    def test_span_coverage_high_anchor_keeps_fit(self):
        """Direct unit test of [span-coverage-high] branch.

        Mirror of :meth:`test_span_coverage_low_anchor_keeps_fit`: a
        synthetic fit where ``c_max`` aligns with predicted cell-(n-1)
        but ``c_min`` is far from predicted cell-0.
        """
        centers = np.array([650.0, 750.0, 850.0, 950.0, 1050.0, 1150.0])
        pitch = 100.0
        offset = 50.0  # predicted cell-0 at 50; predicted cell-11 at 1150
        span = 6
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            new_offset = AutoGridFinder._resolve_span_anchor(
                centers, pitch, offset, n_expected=12, span=span,
                axis_label="rows", span_word="row",
            )
        assert new_offset == offset
        msgs = [str(x.message) for x in w]
        high_count = sum("[span-coverage-high]" in m for m in msgs)
        other_count = sum(
            ("[span-coverage-low]" in m or "[span-coverage-symmetric]" in m)
            for m in msgs
        )
        assert high_count == 1, f"Expected 1 high-anchor warning, got: {msgs}"
        assert other_count == 0, f"Unexpected other span warning: {msgs}"

    def test_span_coverage_symmetric_keeps_fit(self):
        """Direct unit test of [span-coverage-symmetric] branch.

        After the iterative-seed refactor (median-diff pitch +
        ``_choose_seed_anchor``), the iterative fit virtually always
        lands with either ``c_min`` aligning at predicted cell-0 or
        ``c_max`` aligning at predicted cell-(n-1), so reaching the
        "neither end aligned" branch through ``_fit_axis_edges`` with
        normal inputs is no longer practical. The branch is still
        defensible code for pathological post-OLS drift, so we test it
        directly via :meth:`AutoGridFinder._resolve_span_anchor` with
        synthetic pitch/offset that put both predicted boundary cells
        more than ``pitch / 2`` away from the centroid extremes.
        """
        centers = np.array(
            [400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        )
        pitch = 100.0
        offset = 0.0  # predicted cell-0 at 0, cell-11 at 1100 — both far
        span = 7
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            new_offset = AutoGridFinder._resolve_span_anchor(
                centers, pitch, offset, n_expected=12, span=span,
                axis_label="rows", span_word="row",
            )
        assert new_offset == offset  # fit preserved (Option (i))
        msgs = [str(x.message) for x in w]
        sym_count = sum("[span-coverage-symmetric]" in m for m in msgs)
        other_count = sum(
            ("[span-coverage-low]" in m or "[span-coverage-high]" in m)
            for m in msgs
        )
        assert sym_count == 1, f"Expected 1 symmetric warning, got: {msgs}"
        assert other_count == 0, f"Unexpected other span warning: {msgs}"

    def test_empty_table_fallback(self):
        finder = self._make_finder()
        table = _make_info_table([], [])
        edges = finder._fit_axis_edges(table, axis=0, n_expected=8, image_dim=200)
        assert len(edges) == 9


# ===========================================================================
# Constructor
# ===========================================================================


class TestAutoGridFinderConstructor:

    def test_default_params(self):
        finder = AutoGridFinder()
        assert finder.nrows == 8
        assert finder.ncols == 12
        assert finder.residual_fraction == 0.25

    def test_custom_residual_fraction(self):
        finder = AutoGridFinder(residual_fraction=0.5)
        assert finder.residual_fraction == 0.5

    def test_tol_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AutoGridFinder(tol=0.01)
            assert any("tol" in str(warning.message) for warning in w)
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_max_iter_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AutoGridFinder(max_iter=100)
            assert any("max_iter" in str(warning.message) for warning in w)


# ===========================================================================
# Integration with real sample data
# ===========================================================================


class TestAutoGridFinderIntegration:

    @pytest.fixture
    def detected_image(self):
        from phenotypic.data import load_synth_yeast_plate
        from phenotypic.detect import OtsuDetector
        import phenotypic

        image = phenotypic.GridImage(load_synth_yeast_plate())
        return OtsuDetector().apply(image, inplace=False)

    def test_full_pipeline(self, detected_image):
        finder = AutoGridFinder(nrows=8, ncols=12)
        result = finder.measure(detected_image)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert str(GRID.ROW_NUM) in result.columns
        assert str(GRID.COL_NUM) in result.columns
        assert str(GRID.ROW_MAJOR_IDX) in result.columns

    def test_row_edges_count(self, detected_image):
        finder = AutoGridFinder(nrows=8, ncols=12)
        row_edges = finder.get_row_edges(detected_image)
        assert len(row_edges) == 9
        assert np.all(np.diff(row_edges) >= 0)

    def test_col_edges_count(self, detected_image):
        finder = AutoGridFinder(nrows=8, ncols=12)
        col_edges = finder.get_col_edges(detected_image)
        assert len(col_edges) == 13
        assert np.all(np.diff(col_edges) >= 0)

    def test_edges_within_image_bounds(self, detected_image):
        finder = AutoGridFinder(nrows=8, ncols=12)
        row_edges = finder.get_row_edges(detected_image)
        col_edges = finder.get_col_edges(detected_image)
        assert row_edges[0] >= 0
        assert row_edges[-1] <= detected_image.shape[0]
        assert col_edges[0] >= 0
        assert col_edges[-1] <= detected_image.shape[1]

    def test_json_roundtrip(self, detected_image):
        from phenotypic import ImagePipeline
        from phenotypic.detect import OtsuDetector

        # AutoGridFinder is a MeasureFeatures (GridFinder -> GridMeasureFeatures
        # -> MeasureFeatures), so it belongs in the pipeline's ``meas`` queue,
        # not ``ops`` — the pydantic-typed fields now enforce that placement.
        pipeline = ImagePipeline(
            ops=[OtsuDetector()],
            meas=[AutoGridFinder(nrows=8, ncols=12, residual_fraction=0.4)],
        )
        json_str = pipeline.to_json()
        restored = ImagePipeline.from_json(json_str)
        restored_finder = restored._meas["AutoGridFinder"]
        assert isinstance(restored_finder, AutoGridFinder)
        assert restored_finder.nrows == 8
        assert restored_finder.ncols == 12
        assert restored_finder.residual_fraction == 0.4


# ===========================================================================
# _aggregate_to_cell_medians
# ===========================================================================


class TestAggregateToCellMedians:

    def test_single_object_per_cell(self):
        centers = np.array([10.0, 30.0, 50.0])
        indices = np.array([0, 1, 2])
        medians, unique_idx = AutoGridFinder._aggregate_to_cell_medians(
            centers, indices,
        )
        np.testing.assert_array_equal(unique_idx, [0, 1, 2])
        np.testing.assert_array_equal(medians, [10.0, 30.0, 50.0])

    def test_multiple_objects_per_cell(self):
        centers = np.array([9.0, 10.0, 11.0, 29.0, 31.0, 50.0])
        indices = np.array([0, 0, 0, 1, 1, 2])
        medians, unique_idx = AutoGridFinder._aggregate_to_cell_medians(
            centers, indices,
        )
        np.testing.assert_array_equal(unique_idx, [0, 1, 2])
        assert medians[0] == pytest.approx(10.0)  # median of 9,10,11
        assert medians[1] == pytest.approx(30.0)  # median of 29,31
        assert medians[2] == pytest.approx(50.0)

    def test_unequal_counts_produce_equal_output(self):
        """500 objects in one cell and 1 in another → 2 medians."""
        rng = np.random.default_rng(42)
        cell0 = rng.normal(10.0, 1.0, 500)
        cell1 = np.array([50.0])
        centers = np.concatenate([cell0, cell1])
        indices = np.concatenate([np.zeros(500, dtype=int), np.array([2])])
        medians, unique_idx = AutoGridFinder._aggregate_to_cell_medians(
            centers, indices,
        )
        assert len(medians) == 2
        assert len(unique_idx) == 2


# ===========================================================================
# _assign_grid_indices with anchor
# ===========================================================================


class TestAssignGridIndicesWithAnchor:

    def test_anchor_none_uses_first_center(self):
        centers = np.array([10.0, 30.0, 50.0])
        idx_default = AutoGridFinder._assign_grid_indices(centers, 20.0)
        idx_none = AutoGridFinder._assign_grid_indices(centers, 20.0, anchor=None)
        np.testing.assert_array_equal(idx_default, idx_none)

    def test_median_anchor(self):
        centers = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
        anchor = float(np.median(centers))  # 50.0
        indices = AutoGridFinder._assign_grid_indices(centers, 20.0, anchor=anchor)
        # (centers - 50) / 20 → [-2, -1, 0, 1, 2], shifted to [0, 1, 2, 3, 4]
        np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])

    def test_outlier_first_center_with_anchor(self):
        """Median anchor should handle an outlier at the start."""
        centers = np.array([0.0, 50.0, 70.0, 90.0, 110.0])
        # Without anchor: reference is 0.0 (the outlier)
        # With median anchor (70.0): indices are well-behaved
        anchor = float(np.median(centers))
        indices = AutoGridFinder._assign_grid_indices(centers, 20.0, anchor=anchor)
        assert indices.min() == 0
        # The outlier at 0.0 should be far from the grid → large negative index shifted up
        assert len(np.unique(indices)) == 5


# ===========================================================================
# High object count integration test
# ===========================================================================


class TestHighObjectCountRobustness:

    def test_many_objects_per_cell(self):
        """Simulate ~50 objects per grid cell + noise, verify edges are sane."""
        rng = np.random.default_rng(123)
        true_pitch = 25.0
        true_offset = 12.5
        n_rows = 8
        image_dim = 220

        # ~50 objects per cell, jittered around true centers
        row_centers = []
        for i in range(n_rows):
            true_center = true_offset + true_pitch * i
            row_centers.extend(rng.normal(true_center, 2.0, 50))

        # Add scattered noise
        row_centers.extend(rng.uniform(0, image_dim, 100))
        col_centers = rng.uniform(0, 300, len(row_centers))

        table = _make_info_table(row_centers, col_centers)
        finder = AutoGridFinder(nrows=n_rows, ncols=12)
        edges = finder._fit_axis_edges(table, axis=0, n_expected=n_rows, image_dim=image_dim)

        assert len(edges) == n_rows + 1
        assert edges[0] >= 0
        assert edges[-1] <= image_dim
        assert np.all(np.diff(edges) >= 0)

        # Edges should produce roughly correct pitch
        diffs = np.diff(edges)
        nonzero_diffs = diffs[diffs > 0]
        if len(nonzero_diffs) > 0:
            median_spacing = np.median(nonzero_diffs)
            assert 15.0 < median_spacing < 35.0, (
                f"Expected spacing ~25, got median {median_spacing}"
            )


# ===========================================================================
# Object-count guard
# ===========================================================================


class TestObjectCountGuard:

    def test_guard_triggers_above_threshold(self):
        """Uniform edges returned and warning emitted when objects >> cells."""
        rng = np.random.default_rng(42)
        n_expected = 8
        image_dim = 200
        n_objects = AutoGridFinder._MAX_OBJECTS_PER_CELL * n_expected + 1

        table = _make_info_table(
            rng.uniform(0, image_dim, n_objects),
            rng.uniform(0, 300, n_objects),
        )
        finder = AutoGridFinder(nrows=n_expected, ncols=12)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            edges = finder._fit_axis_edges(
                table, axis=0, n_expected=n_expected, image_dim=image_dim,
            )
            assert len(w) == 1
            assert "Falling back to uniform" in str(w[0].message)

        expected = finder._uniform_edges(n_expected, image_dim)
        np.testing.assert_array_equal(edges, expected)

    def test_guard_does_not_trigger_below_threshold(self):
        """No warning when object count is below the threshold."""
        rng = np.random.default_rng(42)
        n_expected = 8
        image_dim = 200
        n_objects = AutoGridFinder._MAX_OBJECTS_PER_CELL * n_expected - 1

        table = _make_info_table(
            rng.uniform(0, image_dim, n_objects),
            rng.uniform(0, 300, n_objects),
        )
        finder = AutoGridFinder(nrows=n_expected, ncols=12)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            finder._fit_axis_edges(
                table, axis=0, n_expected=n_expected, image_dim=image_dim,
            )
            user_warnings = [x for x in w if "Falling back to uniform" in str(x.message)]
            assert len(user_warnings) == 0

    def test_existing_low_and_zero_paths_unaffected(self):
        """The < 2 and < 2*n_expected paths still work correctly."""
        finder = AutoGridFinder(nrows=4, ncols=6)
        image_dim = 200

        # Zero centers → uniform edges
        empty = _make_info_table([], [])
        edges = finder._fit_axis_edges(empty, axis=0, n_expected=4, image_dim=image_dim)
        assert len(edges) == 5

        # 1 center → uniform edges (< 2 guard)
        single = _make_info_table([100.0], [150.0])
        edges = finder._fit_axis_edges(single, axis=0, n_expected=4, image_dim=image_dim)
        assert len(edges) == 5


# ===========================================================================
# Section number dtype
# ===========================================================================


class TestSectionNumberDtype:

    def test_section_columns_are_categorical_uint16(self):
        """ROW_MAJOR_IDX and COL_MAJOR_IDX should be categorical with UInt16 codes."""

        # Build a concrete GridFinder to test the base-class method
        finder = AutoGridFinder(nrows=2, ncols=3)
        table = pd.DataFrame({
            str(GRID.ROW_NUM): pd.Categorical([0, 1, 0, 1]),
            str(GRID.COL_NUM): pd.Categorical([0, 1, 2, 0]),
        })
        result = finder._add_section_number_info(
            table,
            row_edges=np.array([0, 100, 200]),
            col_edges=np.array([0, 80, 160, 240]),
            imshape=(200, 240),
        )

        for col_name in [str(GRID.ROW_MAJOR_IDX), str(GRID.COL_MAJOR_IDX)]:
            assert result[col_name].dtype.name == "category"
            assert result[col_name].cat.categories.dtype == pd.UInt16Dtype()


class TestInspectDashboardRename:
    """Regression guard for the inspect()→dashboard() rename.

    The ``inspect()`` name is reserved across the codebase for methods
    returning a saveable matplotlib or plotly figure consumed by the
    CLI's ``--save-inspect`` flag. AutoGridFinder's diagnostic surface
    returns a ``panel.Column`` (interactive, not flattenable to a
    static raster) so it is exposed under ``dashboard()`` instead.
    These checks catch an accidental revert of the rename, which would
    cause the CLI to auto-save a non-figure return type and emit
    confusing warnings.
    """

    def test_dashboard_method_exists(self):
        assert hasattr(AutoGridFinder, "dashboard"), (
            "AutoGridFinder.dashboard() must exist after the inspect→dashboard rename"
        )
        assert callable(AutoGridFinder.dashboard)

    def test_inspect_method_does_not_exist(self):
        assert not hasattr(AutoGridFinder, "inspect"), (
            "AutoGridFinder.inspect() must NOT exist — it would be auto-picked "
            "up by the CLI's --save-inspect dispatch and produce a panel.Column "
            "(unsupported figure type). Use dashboard() for the interactive "
            "diagnostic surface."
        )
