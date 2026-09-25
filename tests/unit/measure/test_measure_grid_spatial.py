"""Tests for MeasureNeighborDist measurement operation."""

import pytest
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

from phenotypic import GridImage, Image
from phenotypic.grid import ManualGridFinder
from phenotypic.measure import MeasureNeighborDist
from phenotypic.measure._measure_neighbor_dist import (
    _nearest_objects,
    _nearest_relation,
)
from phenotypic.schema import OBJECT
from phenotypic.schema import NEIGHBOR_DIST, GRID


def _circle(objmap: np.ndarray, label: int, rr: int, cc: int, radius: int) -> None:
    """Stamp a filled disc with the given label into ``objmap`` in place."""
    rr_grid, cc_grid = np.ogrid[:objmap.shape[0], :objmap.shape[1]]
    mask = (rr_grid - rr) ** 2 + (cc_grid - cc) ** 2 <= radius ** 2
    objmap[mask] = label


def _make_synthetic_grid_image(
        height: int,
        width: int,
        row_edges: np.ndarray,
        col_edges: np.ndarray,
        circles: list[tuple[int, int, int, int]],
) -> GridImage:
    """Build a GridImage with a ManualGridFinder and a hand-crafted objmap.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        row_edges: Manual grid row edges.
        col_edges: Manual grid column edges.
        circles: List of (label, center_row, center_col, radius) discs to stamp.

    Returns:
        A GridImage with the synthetic objmap injected and the grid finder
        configured so ``grid.info()`` returns the expected grid assignments.
    """
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    finder = ManualGridFinder(row_edges=row_edges, col_edges=col_edges)
    image = GridImage(arr=arr, grid_finder=finder)

    objmap = np.zeros((height, width), dtype=np.uint16)
    for label, rr, cc, radius in circles:
        _circle(objmap, label, rr, cc, radius)
    image.objmap[:] = objmap
    return image


class TestMeasureGridSpatial:
    """Tests for MeasureNeighborDist measurement operation."""

    @pytest.fixture
    def sample_image(self, synth_plate):
        """Reuse session-scoped synth_plate from tests/unit/conftest.py.

        Tests below either read sample_image directly or do .copy() before
        mutating, so sharing a single instance is safe.
        """
        return synth_plate

    @pytest.fixture
    def measurer(self):
        """Create MeasureNeighborDist instance."""
        return MeasureNeighborDist()

    def test_output_has_required_columns(self, sample_image, measurer):
        """Verify all expected columns are present in output."""
        df = measurer.measure(sample_image)

        # First column must be Object_Label
        assert df.columns[0] == OBJECT.LABEL

        # All NEIGHBOR_DIST columns must be present
        expected_columns = [
            NEIGHBOR_DIST.LEFT_NEIGHBOR_OBJ_LABEL,
            NEIGHBOR_DIST.LEFT_DISTANCE,
            NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL,
            NEIGHBOR_DIST.RIGHT_DISTANCE,
            NEIGHBOR_DIST.ABOVE_NEIGHBOR_OBJ_LABEL,
            NEIGHBOR_DIST.ABOVE_DISTANCE,
            NEIGHBOR_DIST.UNDER_NEIGHBOR_OBJ_LABEL,
            NEIGHBOR_DIST.UNDER_DISTANCE,
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_output_row_count_matches_objects(self, sample_image, measurer):
        """Verify output has one row per detected object."""
        df = measurer.measure(sample_image)
        grid_info = sample_image.grid.info(include_metadata=False)
        assert len(df) == len(grid_info)

    def test_object_labels_match(self, sample_image, measurer):
        """Verify Object_Label column matches grid info labels."""
        df = measurer.measure(sample_image)
        grid_info = sample_image.grid.info(include_metadata=False)
        np.testing.assert_array_equal(
                df[OBJECT.LABEL].values,
                grid_info[OBJECT.LABEL].values
        )

    def test_edge_cells_have_nan_left_neighbors(self, sample_image, measurer):
        """Colonies in leftmost column should have NaN left neighbor."""
        df = measurer.measure(sample_image)
        grid_info = sample_image.grid.info(include_metadata=False)

        # Get objects in leftmost column (col 0)
        col_0_labels = grid_info[grid_info[GRID.COL_NUM] == 0][OBJECT.LABEL]

        for label in col_0_labels:
            row = df[df[OBJECT.LABEL] == label]
            assert pd.isna(row[NEIGHBOR_DIST.LEFT_NEIGHBOR_OBJ_LABEL].iloc[0]), \
                f"Object {label} in col 0 should have NaN left neighbor"

    def test_edge_cells_have_nan_above_neighbors(self, sample_image, measurer):
        """Colonies in top row should have NaN above neighbor."""
        df = measurer.measure(sample_image)
        grid_info = sample_image.grid.info(include_metadata=False)

        # Get objects in top row (row 0)
        row_0_labels = grid_info[grid_info[GRID.ROW_NUM] == 0][OBJECT.LABEL]

        for label in row_0_labels:
            row = df[df[OBJECT.LABEL] == label]
            assert pd.isna(row[NEIGHBOR_DIST.ABOVE_NEIGHBOR_OBJ_LABEL].iloc[0]), \
                f"Object {label} in row 0 should have NaN above neighbor"

    def test_distance_is_non_negative(self, sample_image, measurer):
        """All valid distances should be >= 0."""
        df = measurer.measure(sample_image)

        distance_cols = [
            NEIGHBOR_DIST.LEFT_DISTANCE,
            NEIGHBOR_DIST.RIGHT_DISTANCE,
            NEIGHBOR_DIST.ABOVE_DISTANCE,
            NEIGHBOR_DIST.UNDER_DISTANCE,
        ]

        for col in distance_cols:
            valid_distances = df[col].dropna()
            if len(valid_distances) > 0:
                assert (valid_distances >= 0).all(), \
                    f"Column {col} has negative distances"

    def test_neighbor_labels_are_valid(self, sample_image, measurer):
        """Neighbor labels should reference existing objects."""
        df = measurer.measure(sample_image)
        all_labels = set(df[OBJECT.LABEL].values)

        label_cols = [
            NEIGHBOR_DIST.LEFT_NEIGHBOR_OBJ_LABEL,
            NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL,
            NEIGHBOR_DIST.ABOVE_NEIGHBOR_OBJ_LABEL,
            NEIGHBOR_DIST.UNDER_NEIGHBOR_OBJ_LABEL,
        ]

        for col in label_cols:
            valid_labels = df[col].dropna().astype(int)
            for label in valid_labels:
                assert label in all_labels, \
                    f"Neighbor label {label} in {col} is not a valid object label"


class TestWindowBbox:
    """Unit tests for the pure-numpy window union helper."""

    def test_single_bbox_unchanged(self):
        bbox = (10, 20, 30, 40)
        assert MeasureNeighborDist._window_bbox([bbox]) == bbox

    def test_union_takes_extremes(self):
        bboxes = [
            (10, 20, 30, 40),
            (5, 25, 35, 50),
            (8, 18, 28, 45),
        ]
        # min of mins, max of maxs over (min_rr, max_rr, min_cc, max_cc)
        assert MeasureNeighborDist._window_bbox(bboxes) == (5, 25, 28, 50)


class TestEdtDistance:
    """End-to-end tests of the EDT-based algorithm with synthetic GridImages."""

    def test_two_circles_in_adjacent_cells_match_edge_to_edge_distance(self):
        """Distance between two circles in adjacent cells equals center-to-center
        minus the two radii (true pixel-to-pixel)."""
        # 1x2 grid, 100x100 image, one circle per cell at the cell center
        image = _make_synthetic_grid_image(
                height=100,
                width=100,
                row_edges=np.array([0, 100]),
                col_edges=np.array([0, 50, 100]),
                circles=[
                    (1, 50, 25, 5),  # left cell, center (50, 25), r=5
                    (2, 50, 75, 5),  # right cell, center (50, 75), r=5
                ],
        )
        df = MeasureNeighborDist().measure(image)

        left = df[df[OBJECT.LABEL] == 1].iloc[0]
        right = df[df[OBJECT.LABEL] == 2].iloc[0]

        # Center-to-center column distance = 50; subtract two radii of 5
        expected = 50.0 - 5 - 5
        # Allow ±1 px slack for rasterization
        assert abs(left[NEIGHBOR_DIST.RIGHT_DISTANCE] - expected) <= 1.0
        assert int(left[NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL]) == 2

        # Reciprocal
        assert abs(right[NEIGHBOR_DIST.LEFT_DISTANCE] - expected) <= 1.0
        assert int(right[NEIGHBOR_DIST.LEFT_NEIGHBOR_OBJ_LABEL]) == 1

        # Edge-of-plate NaNs hold
        assert pd.isna(left[NEIGHBOR_DIST.LEFT_DISTANCE])
        assert pd.isna(right[NEIGHBOR_DIST.RIGHT_DISTANCE])
        assert pd.isna(left[NEIGHBOR_DIST.ABOVE_DISTANCE])
        assert pd.isna(left[NEIGHBOR_DIST.UNDER_DISTANCE])

    def test_diagonal_circles_match_true_mask_distance(self):
        """For diagonally separated circles, EDT gives the true mask-to-mask
        distance (≈ center-to-center − 2·radius), which differs measurably
        from the bbox-corner approximation the old method used."""
        # 1x2 grid; left circle in top region of left cell, right circle in
        # bottom region of right cell — diagonal even within the left/right
        # neighbour relation
        image = _make_synthetic_grid_image(
                height=100,
                width=100,
                row_edges=np.array([0, 100]),
                col_edges=np.array([0, 50, 100]),
                circles=[
                    (1, 20, 20, 5),
                    (2, 80, 80, 5),
                ],
        )
        df = MeasureNeighborDist().measure(image)
        right_dist = df[df[OBJECT.LABEL] == 1].iloc[0][NEIGHBOR_DIST.RIGHT_DISTANCE]

        # True closest-pixel distance ≈ ||(80,80) − (20,20)|| − 2·r
        center_dist = np.sqrt(60 ** 2 + 60 ** 2)
        expected = center_dist - 2 * 5
        assert abs(right_dist - expected) <= 1.5

        # The old bbox-corner geometry connected the inner corners of the
        # two boxes — (25, 25) and (75, 75) — giving sqrt(50^2 + 50^2)
        # ≈ 70.71, which is the wrong answer for these circles.
        bbox_corner_dist = np.sqrt(50 ** 2 + 50 ** 2)
        assert abs(right_dist - bbox_corner_dist) > 2.0

    def test_multi_object_target_section_per_object_distances(self):
        """Two target objects in the same cell each get their own RightDistance,
        attributed via the Voronoi partition of the window."""
        # Target cell at (0, 0) holds two circles stacked vertically.
        # Right neighbour cell at (0, 1) holds two circles stacked vertically.
        # The top target's nearest right neighbour should be the top-right;
        # the bottom target's should be the bottom-right.
        image = _make_synthetic_grid_image(
                height=100,
                width=100,
                row_edges=np.array([0, 100]),
                col_edges=np.array([0, 50, 100]),
                circles=[
                    (1, 20, 20, 4),  # target top
                    (2, 80, 20, 4),  # target bottom
                    (3, 20, 80, 4),  # neighbour top
                    (4, 80, 80, 4),  # neighbour bottom
                ],
        )
        df = MeasureNeighborDist().measure(image)

        top = df[df[OBJECT.LABEL] == 1].iloc[0]
        bot = df[df[OBJECT.LABEL] == 2].iloc[0]

        # Each target attributes to the closer neighbour (same vertical level)
        assert int(top[NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL]) == 3
        assert int(bot[NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL]) == 4

        # Both gaps are along the same row, both ≈ 60 - 4 - 4 = 52
        expected = 60.0 - 4 - 4
        assert abs(top[NEIGHBOR_DIST.RIGHT_DISTANCE] - expected) <= 1.0
        assert abs(bot[NEIGHBOR_DIST.RIGHT_DISTANCE] - expected) <= 1.0

    def test_shielded_target_returns_nan(self):
        """When a second target object sits between the first target and the
        right neighbour cell, the first target owns no Voronoi territory in
        the right cell and gets NaN for that direction."""
        # Target cell (0,0) holds two circles aligned horizontally:
        #   circle 1 at (50, 10) — far from right cell
        #   circle 2 at (50, 40) — close to right cell, "in the way"
        # Right cell (0,1) holds one circle at (50, 75).
        # From any pixel in the right cell, circle 2 is closer than circle 1,
        # so circle 1 has no closest-attribution to right-cell pixels.
        image = _make_synthetic_grid_image(
                height=100,
                width=100,
                row_edges=np.array([0, 100]),
                col_edges=np.array([0, 50, 100]),
                circles=[
                    (1, 50, 10, 3),
                    (2, 50, 40, 3),
                    (3, 50, 75, 3),
                ],
        )
        df = MeasureNeighborDist().measure(image)

        shielded = df[df[OBJECT.LABEL] == 1].iloc[0]
        front = df[df[OBJECT.LABEL] == 2].iloc[0]

        # Circle 1 is shielded by circle 2 from the right neighbour
        assert pd.isna(shielded[NEIGHBOR_DIST.RIGHT_DISTANCE])
        assert pd.isna(shielded[NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL])

        # Circle 2 reports the real gap to circle 3
        assert int(front[NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL]) == 3
        expected = (75 - 40) - 3 - 3
        assert abs(front[NEIGHBOR_DIST.RIGHT_DISTANCE] - expected) <= 1.0

    def test_empty_neighbor_cell_yields_nan(self):
        """An in-bounds neighbour cell with no detected objects → NaN."""
        # 1x3 grid; only the leftmost and rightmost cells have circles
        image = _make_synthetic_grid_image(
                height=100,
                width=180,
                row_edges=np.array([0, 100]),
                col_edges=np.array([0, 60, 120, 180]),
                circles=[
                    (1, 50, 30, 5),  # left cell
                    (2, 50, 150, 5),  # right cell — middle is empty
                ],
        )
        df = MeasureNeighborDist().measure(image)

        left = df[df[OBJECT.LABEL] == 1].iloc[0]
        right = df[df[OBJECT.LABEL] == 2].iloc[0]

        # Each colony's immediate neighbour cell is empty → NaN
        assert pd.isna(left[NEIGHBOR_DIST.RIGHT_DISTANCE])
        assert pd.isna(left[NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL])
        assert pd.isna(right[NEIGHBOR_DIST.LEFT_DISTANCE])
        assert pd.isna(right[NEIGHBOR_DIST.LEFT_NEIGHBOR_OBJ_LABEL])


class TestMeasureGridSpatialIntegration:
    """Integration tests for MeasureNeighborDist with real data patterns."""

    @pytest.fixture
    def sample_image(self, synth_plate):
        # Reuse session-scoped synth_plate from tests/unit/conftest.py.
        return synth_plate

    def test_reciprocal_neighbors(self, sample_image):
        """For single-object cells, A's right neighbor B implies B's left
        neighbor is A. (Multi-object cells can pick different closest objects
        in each direction, so strict reciprocity only holds for the 1:1
        regime.)"""
        measurer = MeasureNeighborDist()
        df = measurer.measure(sample_image)
        grid_info = sample_image.grid.info(include_metadata=False)

        # Labels in cells that contain exactly one object
        cell_counts = grid_info.groupby(
                [GRID.ROW_NUM, GRID.COL_NUM], observed=True
        ).size()
        single_object_cells = set(cell_counts[cell_counts == 1].index)
        label_to_cell = {
            int(row[OBJECT.LABEL]): (int(row[GRID.ROW_NUM]), int(row[GRID.COL_NUM]))
            for _, row in grid_info.iterrows()
        }

        checked = 0
        for _, row in df.iterrows():
            obj_label = int(row[OBJECT.LABEL])
            if label_to_cell[obj_label] not in single_object_cells:
                continue
            right_neighbor = row[NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL]
            if pd.isna(right_neighbor):
                continue
            r_label = int(right_neighbor)
            if label_to_cell[r_label] not in single_object_cells:
                continue

            neighbor_row = df[df[OBJECT.LABEL] == r_label].iloc[0]
            left_of_neighbor = neighbor_row[NEIGHBOR_DIST.LEFT_NEIGHBOR_OBJ_LABEL]
            assert pd.notna(left_of_neighbor), \
                f"Object {r_label} should have a left neighbor (us, {obj_label})"
            assert int(left_of_neighbor) == obj_label, \
                f"Reciprocity broken: {obj_label} -> right -> {r_label}, but " \
                f"{r_label} -> left -> {int(left_of_neighbor)}"
            checked += 1

        assert checked > 0, \
            "Sample plate had no single-object adjacent pairs to verify"

    def test_consistent_distances(self, sample_image):
        """Distance from A to B should equal distance from B to A."""
        measurer = MeasureNeighborDist()
        df = measurer.measure(sample_image)

        for _, row in df.iterrows():
            obj_label = row[OBJECT.LABEL]
            right_neighbor = row[NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL]
            right_dist = row[NEIGHBOR_DIST.RIGHT_DISTANCE]

            if pd.notna(right_neighbor) and pd.notna(right_dist):
                # Find neighbor's left distance back to us
                neighbor_row = df[df[OBJECT.LABEL] == int(right_neighbor)]
                if len(neighbor_row) > 0:
                    left_neighbor = \
                        neighbor_row[NEIGHBOR_DIST.LEFT_NEIGHBOR_OBJ_LABEL].iloc[0]
                    left_dist = neighbor_row[NEIGHBOR_DIST.LEFT_DISTANCE].iloc[0]

                    # If the neighbor's left neighbor is us, distances should match
                    if pd.notna(left_neighbor) and int(left_neighbor) == int(obj_label):
                        assert np.isclose(right_dist, left_dist), \
                            f"Distance mismatch: {obj_label} -> {right_neighbor} = {right_dist}, " \
                            f"but {right_neighbor} -> {obj_label} = {left_dist}"


class TestPublicGridApiOnly:
    """MeasureNeighborDist uses only public image.grid members and fits the
    grid once per measurement, not once per section."""

    def test_module_uses_no_private_grid_members(self):
        import inspect
        import phenotypic.measure._measure_neighbor_dist as mod
        assert "grid._" not in inspect.getsource(mod)

    @staticmethod
    def _assert_windows_match_oracle(image):
        # Oracle: the accessor's private window helper, called from the TEST
        # only, pins that the public-API reimplementation is exact.
        grid = image.grid
        info = grid.info(include_metadata=False)
        got = MeasureNeighborDist._section_bboxes(
                info, grid.get_row_edges(), grid.get_col_edges(),
                height=image.shape[0], width=image.shape[1],
        )
        assert len(got) > 0
        for (r, c), bbox in got.items():
            (min_rr, min_cc), (max_rr, max_cc) = grid._adv_get_grid_section_slices(
                    r * grid.ncols + c, info
            )
            want = tuple(int(np.asarray(v).item())
                         for v in (min_rr, max_rr, min_cc, max_cc))
            assert bbox == want, (r, c)

    def test_section_bboxes_match_grid_accessor_windows(self, synth_plate):
        # The synth plate exercises only the column-widening terms.
        self._assert_windows_match_oracle(synth_plate)

    def test_section_bboxes_widen_rows_and_clip_to_image(self):
        # Colonies spill past row line 50 in both directions (widening
        # min_rr/max_rr), and the last grid edges sit at the image size, so
        # every edge cell's window is clipped to height-1 / width-1.
        image = _make_synthetic_grid_image(
                height=100, width=100,
                row_edges=np.array([0, 50, 100]),
                col_edges=np.array([0, 50, 100]),
                circles=[
                    (1, 45, 25, 8),   # row 0, spills down to 53
                    (2, 56, 75, 8),   # row 1, spills up to 48
                    (3, 94, 25, 5),   # row 1, touches the bottom edge
                    (4, 20, 94, 5),   # col 1, touches the right edge
                ],
        )
        self._assert_windows_match_oracle(image)

    def test_edges_fetched_once_per_measurement(self, synth_plate, monkeypatch):
        from phenotypic._core._image_parts.accessors import GridAccessor
        calls = {"row": 0, "col": 0}
        orig_row, orig_col = GridAccessor.get_row_edges, GridAccessor.get_col_edges

        def row_spy(self):
            calls["row"] += 1
            return orig_row(self)

        def col_spy(self):
            calls["col"] += 1
            return orig_col(self)

        monkeypatch.setattr(GridAccessor, "get_row_edges", row_spy)
        monkeypatch.setattr(GridAccessor, "get_col_edges", col_spy)
        MeasureNeighborDist().measure(synth_plate)
        # Before: 402 each (one pair per section lookup). After: exactly 1.
        # grid.info() fits through the finder's _operate, not these getters.
        assert calls == {"row": 1, "col": 1}


class TestNearestSchema:
    """NEIGHBOR_DIST declares the three nearest-object members, in order."""

    def test_headers_append_nearest_members_after_directional(self):
        headers = NEIGHBOR_DIST.get_headers()
        assert headers[-3:] == [
            "NeighborDist_NearestObjLabel",
            "NeighborDist_NearestDistance",
            "NeighborDist_NearestRelation",
        ]
        assert len(headers) == 11

    def test_category_is_neighbor_dist_for_every_header(self):
        assert NEIGHBOR_DIST.category() == "NeighborDist"
        assert all(h.startswith("NeighborDist_")
                   for h in NEIGHBOR_DIST.get_headers())

    def test_nearest_members_have_no_authored_bio_desc(self):
        for member in (NEIGHBOR_DIST.NEAREST_OBJ_LABEL,
                       NEIGHBOR_DIST.NEAREST_DISTANCE,
                       NEIGHBOR_DIST.NEAREST_RELATION):
            assert member.bio_desc == ""

    def test_relation_desc_documents_every_code(self):
        desc = NEIGHBOR_DIST.NEAREST_RELATION.desc
        for code in ("0", "1", "2", "3"):
            assert code in desc


def _brute_nearest(objmap: np.ndarray, eligible: np.ndarray):
    """All-pairs nearest over FULL masks (no boundary shortcut, no pruning)."""
    coords = {int(lab): np.argwhere(objmap == lab) for lab in eligible}
    out_label, out_dist = [], []
    for a in eligible:
        best, best_lab = np.inf, np.nan
        for b in sorted(int(x) for x in eligible if x != a):
            d = float(cKDTree(coords[b]).query(coords[int(a)], k=1)[0].min())
            if d < best:
                best, best_lab = d, b
        out_label.append(best_lab)
        out_dist.append(best if np.isfinite(best) else np.nan)
    return np.asarray(out_label, float), np.asarray(out_dist, float)


class TestNearestObjectsHelper:
    """Exact nearest-object search on raw label maps (no grid)."""

    def test_two_discs_edge_to_edge(self):
        objmap = np.zeros((60, 120), np.int32)
        _circle(objmap, 1, 30, 30, 5)
        _circle(objmap, 2, 30, 90, 5)
        lab, dist = _nearest_objects(objmap, np.array([1, 2]))
        assert lab.tolist() == [2.0, 1.0]
        # centre gap 60 minus two radii; ±1 px rasterisation slack
        assert np.all(np.abs(dist - 50.0) <= 1.0)

    def test_fewer_than_two_eligible_is_nan(self):
        objmap = np.zeros((20, 20), np.int32)
        _circle(objmap, 4, 10, 10, 3)
        lab, dist = _nearest_objects(objmap, np.array([4]))
        assert np.isnan(lab).all() and np.isnan(dist).all()
        lab0, dist0 = _nearest_objects(objmap, np.array([], dtype=np.int64))
        assert lab0.shape == (0,) and dist0.shape == (0,)

    def test_ineligible_label_is_never_returned(self):
        # Label 2 is physically closest to 1 but not eligible (an off-grid
        # object on a GridImage); 1 must fall through to 3.
        objmap = np.zeros((40, 120), np.int32)
        _circle(objmap, 1, 20, 20, 4)
        _circle(objmap, 2, 20, 40, 4)
        _circle(objmap, 3, 20, 100, 4)
        lab, _ = _nearest_objects(objmap, np.array([1, 3]))
        assert lab.tolist() == [3.0, 1.0]

    def test_tie_resolves_to_smaller_label_even_when_visited_second(self):
        # Target 5 is a single pixel at (20, 20).
        # Label 2: one pixel at (20, 30): distance 10, bbox lower bound 10.
        # Label 9: pixels (26, 28) and (12, 40): distance 10 via (6, 8), but
        #          bbox rows 12..26 x cols 28..40 gives a lower bound of 8, so 9 is
        #          visited FIRST. The strict `lb > best` stop must still visit 2,
        #          and the tie-break must prefer it.
        objmap = np.zeros((50, 50), np.int32)
        objmap[20, 20] = 5
        objmap[20, 30] = 2
        objmap[26, 28] = 9
        objmap[12, 40] = 9
        lab, dist = _nearest_objects(objmap, np.array([5, 2, 9]))
        assert lab[0] == 2.0
        assert dist[0] == 10.0

    def test_non_contiguous_labels(self):
        objmap = np.zeros((40, 160), np.int32)
        _circle(objmap, 1, 20, 20, 4)
        _circle(objmap, 3, 20, 60, 4)
        _circle(objmap, 7, 20, 140, 4)
        lab, _ = _nearest_objects(objmap, np.array([1, 3, 7]))
        assert lab.tolist() == [3.0, 1.0, 3.0]

    def test_border_touching_objects(self):
        objmap = np.zeros((30, 30), np.int32)
        objmap[0:5, 0:5] = 1        # touches top-left corner of the array
        objmap[25:30, 20:30] = 2    # touches bottom-right edges
        lab, dist = _nearest_objects(objmap, np.array([1, 2]))
        # closest pixels (4, 4) and (25, 20): offset (21, 16)
        assert dist[0] == np.sqrt(21 ** 2 + 16 ** 2)
        assert lab.tolist() == [2.0, 1.0]

    def test_satellite_inside_ring_hole(self):
        objmap = np.zeros((80, 80), np.int32)
        rr, cc = np.ogrid[:80, :80]
        r2 = (rr - 40) ** 2 + (cc - 40) ** 2
        objmap[(r2 <= 30 ** 2) & (r2 > 15 ** 2)] = 1   # ring, hole radius 15
        objmap[r2 <= 3 ** 2] = 2                        # satellite in the hole
        lab, dist = _nearest_objects(objmap, np.array([1, 2]))
        # gap across the hole is ~15 - 3; the outer rim would be ~27 away
        assert lab.tolist() == [2.0, 1.0]
        assert abs(dist[0] - 12.0) <= 1.0

    def test_touching_objects_distance_one(self):
        objmap = np.zeros((20, 20), np.int32)
        objmap[5:10, 5:10] = 1
        objmap[5:10, 10:15] = 2   # shares an edge with label 1
        _, dist = _nearest_objects(objmap, np.array([1, 2]))
        assert dist.tolist() == [1.0, 1.0]

    def test_eligible_label_missing_from_objmap_raises(self):
        objmap = np.zeros((10, 10), np.int32)
        objmap[2, 2] = 1
        with pytest.raises(ValueError, match="label 4"):
            _nearest_objects(objmap, np.array([1, 4]))

    @pytest.mark.parametrize("seed", range(20))
    def test_matches_brute_force_on_random_blobs(self, seed):
        rng = np.random.default_rng(seed)
        objmap = np.zeros((64, 64), np.int32)
        label = 1
        for _ in range(40):
            if label > 9:
                break
            r, c, rad = rng.integers(4, 60), rng.integers(4, 60), rng.integers(1, 5)
            rr, cc = np.ogrid[:64, :64]
            disc = (rr - r) ** 2 + (cc - c) ** 2 <= rad ** 2
            if (objmap[disc] != 0).any():
                continue
            objmap[disc] = label
            label += 1
        eligible = np.unique(objmap[objmap > 0]).astype(np.int64)
        got = _nearest_objects(objmap, eligible)
        want = _brute_nearest(objmap, eligible)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])


class TestNearestRelation:
    def test_codes(self):
        self_rc = np.array([[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]], float)
        near_rc = np.array([[2, 2], [2, 3], [1, 2], [3, 3], [2, 4]], float)
        assert _nearest_relation(self_rc, near_rc).tolist() == [0, 1, 1, 2, 3]

    def test_empty(self):
        empty = np.empty((0, 2))
        assert _nearest_relation(empty, empty).shape == (0,)


@pytest.fixture(scope="module")
def synth_neighbor_df(synth_plate):
    """One MeasureNeighborDist run on the synth plate, shared by the
    read-only assertions below (they only read the frame)."""
    return MeasureNeighborDist().measure(synth_plate)


def _row(df: pd.DataFrame, label: int) -> pd.Series:
    return df[df[OBJECT.LABEL] == label].iloc[0]


class TestNearestColumns:
    """End-to-end nearest-object columns on synthetic plates."""

    def test_same_cell_satellite(self):
        image = _make_synthetic_grid_image(
                height=100, width=100,
                row_edges=np.array([0, 100]), col_edges=np.array([0, 50, 100]),
                circles=[(1, 50, 25, 8), (2, 50, 40, 2), (3, 50, 75, 8)],
        )
        df = MeasureNeighborDist().measure(image)
        colony, satellite, other = _row(df, 1), _row(df, 2), _row(df, 3)
        assert colony[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == 2
        assert colony[NEIGHBOR_DIST.NEAREST_RELATION] == 0
        assert abs(colony[NEIGHBOR_DIST.NEAREST_DISTANCE] - (15 - 8 - 2)) <= 1.0
        assert satellite[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == 1
        assert other[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == 2
        assert other[NEIGHBOR_DIST.NEAREST_RELATION] == 1

    def test_diagonal_when_adjacent_cells_empty(self):
        image = _make_synthetic_grid_image(
                height=100, width=100,
                row_edges=np.array([0, 50, 100]), col_edges=np.array([0, 50, 100]),
                circles=[(1, 25, 25, 5), (2, 75, 75, 5)],
        )
        df = MeasureNeighborDist().measure(image)
        a = _row(df, 1)
        assert a[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == 2
        assert a[NEIGHBOR_DIST.NEAREST_RELATION] == 2
        assert abs(a[NEIGHBOR_DIST.NEAREST_DISTANCE]
                   - (np.hypot(50, 50) - 10)) <= 1.5
        # the directional columns can't see a diagonal neighbour
        assert pd.isna(a[NEIGHBOR_DIST.RIGHT_DISTANCE])
        assert pd.isna(a[NEIGHBOR_DIST.UNDER_DISTANCE])

    def test_distant_when_middle_cell_empty(self):
        image = _make_synthetic_grid_image(
                height=100, width=180,
                row_edges=np.array([0, 100]),
                col_edges=np.array([0, 60, 120, 180]),
                circles=[(1, 50, 30, 5), (2, 50, 150, 5)],
        )
        df = MeasureNeighborDist().measure(image)
        for label, other in ((1, 2), (2, 1)):
            r = _row(df, label)
            assert r[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == other
            assert r[NEIGHBOR_DIST.NEAREST_RELATION] == 3
            assert abs(r[NEIGHBOR_DIST.NEAREST_DISTANCE] - 110.0) <= 1.0

    def test_adjacent_nearest_equals_directional_distance(self):
        image = _make_synthetic_grid_image(
                height=100, width=100,
                row_edges=np.array([0, 100]), col_edges=np.array([0, 50, 100]),
                circles=[(1, 50, 25, 5), (2, 50, 75, 5)],
        )
        df = MeasureNeighborDist().measure(image)
        a = _row(df, 1)
        assert a[NEIGHBOR_DIST.NEAREST_RELATION] == 1
        # same pixel-centre convention, same pair → bit-identical (spec C4)
        assert a[NEIGHBOR_DIST.NEAREST_DISTANCE] == a[NEIGHBOR_DIST.RIGHT_DISTANCE]

    def test_single_object_nearest_is_nan(self):
        image = _make_synthetic_grid_image(
                height=100, width=100,
                row_edges=np.array([0, 100]), col_edges=np.array([0, 50, 100]),
                circles=[(1, 50, 25, 5)],
        )
        df = MeasureNeighborDist().measure(image)
        r = _row(df, 1)
        for col in (NEIGHBOR_DIST.NEAREST_OBJ_LABEL,
                    NEIGHBOR_DIST.NEAREST_DISTANCE,
                    NEIGHBOR_DIST.NEAREST_RELATION):
            assert pd.isna(r[col])

    def test_plain_image_emits_all_columns(self):
        image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
        objmap = np.zeros((100, 100), dtype=np.uint16)
        _circle(objmap, 1, 50, 25, 5)
        _circle(objmap, 2, 50, 75, 5)
        image.objmap[:] = objmap
        assert not hasattr(image, "grid")

        df = MeasureNeighborDist().measure(image)

        assert [str(c) for c in df.columns] == (
                [str(OBJECT.LABEL)] + NEIGHBOR_DIST.get_headers()
        )
        a = _row(df, 1)
        assert a[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == 2
        assert abs(a[NEIGHBOR_DIST.NEAREST_DISTANCE] - 40.0) <= 1.0
        directional = NEIGHBOR_DIST.get_headers()[:8]
        assert df[directional].isna().all().all()
        assert df[str(NEIGHBOR_DIST.NEAREST_RELATION)].isna().all()

    def test_grid_output_columns_match_schema_exactly(
            self, synth_plate, synth_neighbor_df
    ):
        df = synth_neighbor_df
        assert [str(c) for c in df.columns] == (
                [str(OBJECT.LABEL)] + NEIGHBOR_DIST.get_headers()
        )
        assert len(df) == synth_plate.num_objects

    def test_include_meta_merges_grid_info(self, synth_plate):
        df = MeasureNeighborDist().measure(synth_plate, include_meta=True)
        assert len(df) == synth_plate.num_objects
        assert str(GRID.ROW_NUM) in [str(c) for c in df.columns]
        assert [str(c) for c in df.columns].count(str(OBJECT.LABEL)) == 1


class TestNearestProperty:
    """Invariants on the real synthetic yeast plate."""

    def test_nearest_never_exceeds_any_directional_distance(self, synth_neighbor_df):
        df = synth_neighbor_df
        nearest = df[NEIGHBOR_DIST.NEAREST_DISTANCE].to_numpy()
        checked = 0
        for col in (NEIGHBOR_DIST.LEFT_DISTANCE, NEIGHBOR_DIST.RIGHT_DISTANCE,
                    NEIGHBOR_DIST.ABOVE_DISTANCE, NEIGHBOR_DIST.UNDER_DISTANCE):
            d = df[col].to_numpy()
            ok = ~np.isnan(d)
            # exact: both are sqrt of the same integer squared distance (C4)
            assert np.all(nearest[ok] <= d[ok])
            checked += int(ok.sum())
        assert checked > 0

    def test_matches_brute_force_on_synth_plate(self, synth_plate, synth_neighbor_df):
        df = synth_neighbor_df
        objmap = synth_plate.objmap[:]
        labels = df[OBJECT.LABEL].to_numpy().astype(np.int64)
        want_label, want_dist = _brute_nearest(objmap, labels)
        np.testing.assert_array_equal(
                df[NEIGHBOR_DIST.NEAREST_OBJ_LABEL].to_numpy(), want_label)
        np.testing.assert_array_equal(
                df[NEIGHBOR_DIST.NEAREST_DISTANCE].to_numpy(), want_dist)


class TestNearestEdgeCases:
    """Spec §4 edge cases driven through measure(), plus the large-object cost."""

    _COLUMNS = [str(OBJECT.LABEL)] + NEIGHBOR_DIST.get_headers()

    def test_empty_plain_image_returns_no_rows(self):
        image = Image(arr=np.zeros((50, 50, 3), dtype=np.uint8))
        df = MeasureNeighborDist().measure(image)
        assert len(df) == 0
        assert [str(c) for c in df.columns] == self._COLUMNS

    def test_empty_grid_image_returns_no_rows(self):
        image = _make_synthetic_grid_image(
                height=100, width=100,
                row_edges=np.array([0, 50, 100]), col_edges=np.array([0, 50, 100]),
                circles=[],
        )
        df = MeasureNeighborDist().measure(image)
        assert len(df) == 0
        assert [str(c) for c in df.columns] == self._COLUMNS

    def test_off_grid_object_keeps_row_and_is_never_nearest(self, monkeypatch):
        # No shipped grid finder yields a NaN grid position (spec §2), so
        # blank one row of the real grid.info() frame. Label 2 sits 7 px from
        # label 1 and is NOT the last row, so both "search everything" and a
        # misaligned scatter back into the frame change the result.
        from phenotypic._core._image_parts.accessors import GridAccessor
        orig_info = GridAccessor.info

        def info_with_off_grid(self, include_metadata=True):
            df = orig_info(self, include_metadata=include_metadata)
            df.loc[df[OBJECT.LABEL] == 2, GRID.ROW_NUM] = np.nan
            return df

        monkeypatch.setattr(GridAccessor, "info", info_with_off_grid)
        image = _make_synthetic_grid_image(
                height=100, width=150,
                row_edges=np.array([0, 100]),
                col_edges=np.array([0, 50, 100, 150]),
                circles=[(1, 50, 25, 5), (2, 50, 40, 3), (3, 50, 125, 5)],
        )
        df = MeasureNeighborDist().measure(image)

        assert sorted(df[OBJECT.LABEL].tolist()) == [1, 2, 3]
        off = _row(df, 2)
        for col in (NEIGHBOR_DIST.NEAREST_OBJ_LABEL,
                    NEIGHBOR_DIST.NEAREST_DISTANCE,
                    NEIGHBOR_DIST.NEAREST_RELATION):
            assert pd.isna(off[col])
        assert _row(df, 1)[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == 3
        assert _row(df, 3)[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] == 1
        on_grid = df[df[OBJECT.LABEL] != 2]
        assert on_grid[NEIGHBOR_DIST.NEAREST_RELATION].tolist() == [3.0, 3.0]

    def test_large_object_boundary_is_never_the_query_side(self, monkeypatch):
        # A plate-rim ring encloses nine small colonies, so every colony's box
        # lower bound against the rim is 0 and the rim visits all nine. The
        # rim's long boundary must go into a tree, never be queried nine times.
        import phenotypic.measure._measure_neighbor_dist as mod
        queried: list[int] = []

        class SpyTree(cKDTree):
            def query(self, x, *args, **kwargs):
                queried.append(len(x))
                return super().query(x, *args, **kwargs)

        monkeypatch.setattr(mod, "cKDTree", SpyTree)
        objmap = np.zeros((200, 200), np.int32)
        rr, cc = np.ogrid[:200, :200]
        r2 = (rr - 100) ** 2 + (cc - 100) ** 2
        objmap[(r2 <= 98 ** 2) & (r2 > 94 ** 2)] = 1
        label = 2
        for r in (70, 100, 130):
            for c in (70, 100, 130):
                _circle(objmap, label, r, c, 5)
                label += 1
        eligible = np.arange(1, label, dtype=np.int64)

        got = _nearest_objects(objmap, eligible)

        rim = objmap == 1
        from scipy import ndimage as ndi
        rim_boundary = int((rim & ~ndi.binary_erosion(rim)).sum())
        assert max(queried) < rim_boundary
        want = _brute_nearest(objmap, eligible)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])
