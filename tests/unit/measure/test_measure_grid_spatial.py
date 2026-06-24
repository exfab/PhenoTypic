"""Tests for MeasureNeighborDist measurement operation."""

import pytest
import pandas as pd
import numpy as np

from phenotypic import GridImage
from phenotypic.grid import ManualGridFinder
from phenotypic.measure import MeasureNeighborDist
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
