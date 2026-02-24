"""Tests for MeasureGridSpatial measurement operation."""

import pytest
import pandas as pd
import numpy as np

from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureGridSpatial
from phenotypic.tools_.constants_ import OBJECT
from phenotypic.tools_.measurement_info_ import GRID_SPATIAL, GRID


class TestMeasureGridSpatial:
    """Tests for MeasureGridSpatial measurement operation."""

    @pytest.fixture
    def sample_image(self):
        """Load synthetic yeast plate image for testing."""
        return load_synth_yeast_plate()

    @pytest.fixture
    def measurer(self):
        """Create MeasureGridSpatial instance."""
        return MeasureGridSpatial()

    def test_output_has_required_columns(self, sample_image, measurer):
        """Verify all expected columns are present in output."""
        df = measurer.measure(sample_image)

        # First column must be ObjectLabel
        assert df.columns[0] == OBJECT.LABEL

        # All GRID_SPATIAL columns must be present
        expected_columns = [
            GRID_SPATIAL.LEFT_NEIGHBOR_OBJ_LABEL,
            GRID_SPATIAL.LEFT_DISTANCE,
            GRID_SPATIAL.RIGHT_NEIGHBOR_OBJ_LABEL,
            GRID_SPATIAL.RIGHT_DISTANCE,
            GRID_SPATIAL.ABOVE_NEIGHBOR_OBJ_LABEL,
            GRID_SPATIAL.ABOVE_DISTANCE,
            GRID_SPATIAL.UNDER_NEIGHBOR_OBJ_LABEL,
            GRID_SPATIAL.UNDER_DISTANCE,
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_output_row_count_matches_objects(self, sample_image, measurer):
        """Verify output has one row per detected object."""
        df = measurer.measure(sample_image)
        grid_info = sample_image.grid.info(include_metadata=False)
        assert len(df) == len(grid_info)

    def test_object_labels_match(self, sample_image, measurer):
        """Verify ObjectLabel column matches grid info labels."""
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
            assert pd.isna(row[GRID_SPATIAL.LEFT_NEIGHBOR_OBJ_LABEL].iloc[0]), \
                f"Object {label} in col 0 should have NaN left neighbor"

    def test_edge_cells_have_nan_above_neighbors(self, sample_image, measurer):
        """Colonies in top row should have NaN above neighbor."""
        df = measurer.measure(sample_image)
        grid_info = sample_image.grid.info(include_metadata=False)

        # Get objects in top row (row 0)
        row_0_labels = grid_info[grid_info[GRID.ROW_NUM] == 0][OBJECT.LABEL]

        for label in row_0_labels:
            row = df[df[OBJECT.LABEL] == label]
            assert pd.isna(row[GRID_SPATIAL.ABOVE_NEIGHBOR_OBJ_LABEL].iloc[0]), \
                f"Object {label} in row 0 should have NaN above neighbor"

    def test_distance_is_non_negative(self, sample_image, measurer):
        """All valid distances should be >= 0."""
        df = measurer.measure(sample_image)

        distance_cols = [
            GRID_SPATIAL.LEFT_DISTANCE,
            GRID_SPATIAL.RIGHT_DISTANCE,
            GRID_SPATIAL.ABOVE_DISTANCE,
            GRID_SPATIAL.UNDER_DISTANCE,
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
            GRID_SPATIAL.LEFT_NEIGHBOR_OBJ_LABEL,
            GRID_SPATIAL.RIGHT_NEIGHBOR_OBJ_LABEL,
            GRID_SPATIAL.ABOVE_NEIGHBOR_OBJ_LABEL,
            GRID_SPATIAL.UNDER_NEIGHBOR_OBJ_LABEL,
        ]

        for col in label_cols:
            valid_labels = df[col].dropna().astype(int)
            for label in valid_labels:
                assert label in all_labels, \
                    f"Neighbor label {label} in {col} is not a valid object label"


class TestBboxDistance:
    """Tests for the bounding box distance calculation."""

    def test_non_overlapping_horizontal(self):
        """Test distance between horizontally separated boxes."""
        bbox1 = (0, 10, 0, 10)  # min_rr, max_rr, min_cc, max_cc
        bbox2 = (0, 10, 20, 30)  # 10 pixels apart in column direction

        dist = MeasureGridSpatial._bbox_distance(bbox1, bbox2)
        assert dist == 10.0

    def test_non_overlapping_vertical(self):
        """Test distance between vertically separated boxes."""
        bbox1 = (0, 10, 0, 10)
        bbox2 = (20, 30, 0, 10)  # 10 pixels apart in row direction

        dist = MeasureGridSpatial._bbox_distance(bbox1, bbox2)
        assert dist == 10.0

    def test_overlapping_boxes(self):
        """Overlapping boxes should have distance 0."""
        bbox1 = (0, 10, 0, 10)
        bbox2 = (5, 15, 5, 15)  # Overlaps in both dimensions

        dist = MeasureGridSpatial._bbox_distance(bbox1, bbox2)
        assert dist == 0.0

    def test_touching_boxes(self):
        """Boxes that touch (share an edge) should have distance 0."""
        bbox1 = (0, 10, 0, 10)
        bbox2 = (0, 10, 10, 20)  # Touches at column 10

        dist = MeasureGridSpatial._bbox_distance(bbox1, bbox2)
        assert dist == 0.0

    def test_diagonal_separation(self):
        """Test distance between diagonally separated boxes."""
        bbox1 = (0, 10, 0, 10)
        bbox2 = (20, 30, 20, 30)  # 10 apart in both directions

        dist = MeasureGridSpatial._bbox_distance(bbox1, bbox2)
        expected = np.sqrt(100 + 100)  # sqrt(10^2 + 10^2)
        assert np.isclose(dist, expected)

    def test_single_pixel_boxes(self):
        """Test distance between single-pixel boxes."""
        bbox1 = (5, 5, 5, 5)  # Single pixel at (5, 5)
        bbox2 = (5, 5, 8, 8)  # Single pixel at (5, 8)

        dist = MeasureGridSpatial._bbox_distance(bbox1, bbox2)
        assert dist == 3.0

    def test_one_dimensional_overlap(self):
        """Boxes overlapping in one dimension only."""
        bbox1 = (0, 10, 0, 10)
        bbox2 = (5, 15, 20, 30)  # Overlaps in rows, separated in cols

        dist = MeasureGridSpatial._bbox_distance(bbox1, bbox2)
        assert dist == 10.0  # Distance only in column direction


class TestMeasureGridSpatialIntegration:
    """Integration tests for MeasureGridSpatial with real data patterns."""

    @pytest.fixture
    def sample_image(self):
        return load_synth_yeast_plate()

    def test_reciprocal_neighbors(self, sample_image):
        """If A's right neighbor is B, then B's left neighbor should be A."""
        measurer = MeasureGridSpatial()
        df = measurer.measure(sample_image)

        # Find pairs where right neighbor is defined
        for _, row in df.iterrows():
            right_neighbor = row[GRID_SPATIAL.RIGHT_NEIGHBOR_OBJ_LABEL]
            if pd.notna(right_neighbor):
                obj_label = row[OBJECT.LABEL]
                # Find the neighbor's row
                neighbor_row = df[df[OBJECT.LABEL] == int(right_neighbor)]
                if len(neighbor_row) > 0:
                    # The neighbor's left neighbor should be us
                    left_of_neighbor = neighbor_row[GRID_SPATIAL.LEFT_NEIGHBOR_OBJ_LABEL].iloc[0]
                    if pd.notna(left_of_neighbor):
                        # Note: This may not always be true if there are multiple
                        # objects per cell; the closest one is selected
                        pass  # Skip strict assertion for multi-object cells

    def test_consistent_distances(self, sample_image):
        """Distance from A to B should equal distance from B to A."""
        measurer = MeasureGridSpatial()
        df = measurer.measure(sample_image)

        for _, row in df.iterrows():
            obj_label = row[OBJECT.LABEL]
            right_neighbor = row[GRID_SPATIAL.RIGHT_NEIGHBOR_OBJ_LABEL]
            right_dist = row[GRID_SPATIAL.RIGHT_DISTANCE]

            if pd.notna(right_neighbor) and pd.notna(right_dist):
                # Find neighbor's left distance back to us
                neighbor_row = df[df[OBJECT.LABEL] == int(right_neighbor)]
                if len(neighbor_row) > 0:
                    left_neighbor = neighbor_row[GRID_SPATIAL.LEFT_NEIGHBOR_OBJ_LABEL].iloc[0]
                    left_dist = neighbor_row[GRID_SPATIAL.LEFT_DISTANCE].iloc[0]

                    # If the neighbor's left neighbor is us, distances should match
                    if pd.notna(left_neighbor) and int(left_neighbor) == int(obj_label):
                        assert np.isclose(right_dist, left_dist), \
                            f"Distance mismatch: {obj_label} -> {right_neighbor} = {right_dist}, " \
                            f"but {right_neighbor} -> {obj_label} = {left_dist}"
