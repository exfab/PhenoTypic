"""Measure spatial relationships between neighboring colonies in grid cells."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

import pandas as pd
import numpy as np

from phenotypic.abc_ import GridMeasureFeatures
from phenotypic.tools_.constants_ import OBJECT
from phenotypic.tools_.measurement_info_ import GRID_SPATIAL, BBOX, GRID


class MeasureGridSpatial(GridMeasureFeatures):
    """Measure edge-to-edge distances to neighbors in adjacent grid cells.

    For each detected colony, identify the nearest object in the left,
    right, above, and below grid cells and compute the minimum
    bounding-box edge-to-edge distance. Edge and corner colonies report
    ``NaN`` for directions beyond the plate boundary.

    Returns:
        pd.DataFrame: Object-level neighbor measurements with columns:

            - Label: unique object identifier.
            - LeftNeighborObjLabel, LeftDistance.
            - RightNeighborObjLabel, RightDistance.
            - AboveNeighborObjLabel, AboveDistance.
            - UnderNeighborObjLabel, UnderDistance.
            - ``NaN`` where no neighbor exists in a given direction.

    Best For:
        - Quantifying colony spacing and crowding to assess nutrient
          competition risk on arrayed plates.
        - Flagging closely spaced colonies that may cross-contaminate.
        - Enabling neighbor-aware paired statistical comparisons for
          competition or cooperation studies.

    Consider Also:
        - :class:`MeasureGridLinRegStats` for regression-based
          positional quality metrics.
        - :class:`MeasureGridSpread` for within-well colony dispersion
          rather than between-well distances.
        - :class:`MeasureBounds` for raw bounding boxes and centroids
          without neighbor lookups.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of grid-level measurements.
    """

    _measurement_info_class = GRID_SPATIAL

    def _operate(self, image: GridImage) -> pd.DataFrame:
        gs_table = image.grid.info(include_metadata=False)

        # Build lookup: (row, col) -> list of (label, min_rr, max_rr, min_cc, max_cc)
        cell_lookup = self._build_cell_lookup(gs_table)

        # Initialize result arrays (MeasurementInfo members are strings, use directly)
        n_objects = len(gs_table)
        results = {
            GRID_SPATIAL.LEFT_NEIGHBOR_OBJ_LABEL : np.full(n_objects, np.nan),
            GRID_SPATIAL.LEFT_DISTANCE           : np.full(n_objects, np.nan),
            GRID_SPATIAL.RIGHT_NEIGHBOR_OBJ_LABEL: np.full(n_objects, np.nan),
            GRID_SPATIAL.RIGHT_DISTANCE          : np.full(n_objects, np.nan),
            GRID_SPATIAL.ABOVE_NEIGHBOR_OBJ_LABEL: np.full(n_objects, np.nan),
            GRID_SPATIAL.ABOVE_DISTANCE          : np.full(n_objects, np.nan),
            GRID_SPATIAL.UNDER_NEIGHBOR_OBJ_LABEL: np.full(n_objects, np.nan),
            GRID_SPATIAL.UNDER_DISTANCE          : np.full(n_objects, np.nan),
        }

        # Process each object
        for i, (idx, row) in enumerate(gs_table.iterrows()):
            obj_row = row[GRID.ROW_NUM]
            obj_col = row[GRID.COL_NUM]
            obj_bbox = self._extract_bbox(row)

            if pd.isna(obj_row) or pd.isna(obj_col):
                continue

            obj_row, obj_col = int(obj_row), int(obj_col)

            # Check each direction: (direction_name, neighbor_row, neighbor_col, label_col, dist_col)
            directions = [
                (obj_row, obj_col - 1, GRID_SPATIAL.LEFT_NEIGHBOR_OBJ_LABEL,
                 GRID_SPATIAL.LEFT_DISTANCE),
                (obj_row, obj_col + 1, GRID_SPATIAL.RIGHT_NEIGHBOR_OBJ_LABEL,
                 GRID_SPATIAL.RIGHT_DISTANCE),
                (obj_row - 1, obj_col, GRID_SPATIAL.ABOVE_NEIGHBOR_OBJ_LABEL,
                 GRID_SPATIAL.ABOVE_DISTANCE),
                (obj_row + 1, obj_col, GRID_SPATIAL.UNDER_NEIGHBOR_OBJ_LABEL,
                 GRID_SPATIAL.UNDER_DISTANCE),
            ]

            for n_row, n_col, label_col, dist_col in directions:
                neighbor_label, neighbor_dist = self._find_closest_neighbor(
                        cell_lookup, n_row, n_col, obj_bbox
                )
                if neighbor_label is not None:
                    results[label_col][i] = neighbor_label
                    results[dist_col][i] = neighbor_dist

        # Build output DataFrame
        df = pd.DataFrame(results)
        df.insert(0, OBJECT.LABEL, gs_table[OBJECT.LABEL].values)
        return df

    def _build_cell_lookup(self, gs_table: pd.DataFrame) -> dict:
        """Build lookup mapping (row, col) -> list of (label, bbox).

        Args:
            gs_table: DataFrame from image.grid.info() with object positions.

        Returns:
            Dictionary mapping (grid_row, grid_col) tuples to lists of
            (object_label, bbox_tuple) for all objects in that cell.
        """
        lookup = {}
        for _, row in gs_table.iterrows():
            grid_row = row[GRID.ROW_NUM]
            grid_col = row[GRID.COL_NUM]
            if pd.isna(grid_row) or pd.isna(grid_col):
                continue
            key = (int(grid_row), int(grid_col))
            bbox = self._extract_bbox(row)
            label = row[OBJECT.LABEL]
            lookup.setdefault(key, []).append((label, bbox))
        return lookup

    def _extract_bbox(self, row: pd.Series) -> tuple:
        """Extract bounding box tuple from DataFrame row.

        Args:
            row: Single row from grid info DataFrame.

        Returns:
            Tuple of (min_rr, max_rr, min_cc, max_cc) bounding box coordinates.
        """
        return (
            row[BBOX.MIN_RR],
            row[BBOX.MAX_RR],
            row[BBOX.MIN_CC],
            row[BBOX.MAX_CC],
        )

    def _find_closest_neighbor(
            self, lookup: dict, n_row: int, n_col: int, obj_bbox: tuple
    ) -> tuple:
        """Find closest object in neighbor cell.

        Args:
            lookup: Cell lookup dictionary from _build_cell_lookup.
            n_row: Grid row index of neighbor cell.
            n_col: Grid column index of neighbor cell.
            obj_bbox: Bounding box of the current object.

        Returns:
            Tuple of (neighbor_label, distance) if neighbor exists,
            otherwise (None, None).
        """
        if n_row < 0 or n_col < 0:
            return None, None

        neighbors = lookup.get((n_row, n_col), [])
        if not neighbors:
            return None, None

        closest_label = None
        closest_dist = np.inf

        for label, n_bbox in neighbors:
            dist = self._bbox_distance(obj_bbox, n_bbox)
            if dist < closest_dist:
                closest_dist = dist
                closest_label = label

        return closest_label, closest_dist

    @staticmethod
    def _bbox_distance(bbox1: tuple, bbox2: tuple) -> float:
        """Compute minimum edge-to-edge Euclidean distance between bounding boxes.

        This computes the minimum distance between any point on the first bounding
        box and any point on the second bounding box. If the boxes overlap, the
        distance is 0.

        Args:
            bbox1: First bounding box as (min_rr, max_rr, min_cc, max_cc).
            bbox2: Second bounding box as (min_rr, max_rr, min_cc, max_cc).

        Returns:
            Minimum Euclidean distance between the bounding boxes (0 if overlapping).
        """
        min_rr1, max_rr1, min_cc1, max_cc1 = bbox1
        min_rr2, max_rr2, min_cc2, max_cc2 = bbox2

        # Distance in row direction (0 if overlapping)
        if max_rr1 < min_rr2:
            rr_dist = min_rr2 - max_rr1
        elif max_rr2 < min_rr1:
            rr_dist = min_rr1 - max_rr2
        else:
            rr_dist = 0

        # Distance in column direction (0 if overlapping)
        if max_cc1 < min_cc2:
            cc_dist = min_cc2 - max_cc1
        elif max_cc2 < min_cc1:
            cc_dist = min_cc1 - max_cc2
        else:
            cc_dist = 0

        return np.sqrt(rr_dist ** 2 + cc_dist ** 2)


MeasureGridSpatial.__doc__ = GRID_SPATIAL.append_rst_to_doc(MeasureGridSpatial)
