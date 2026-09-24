"""Measure spatial relationships between neighboring colonies in grid cells."""

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from phenotypic.abc_ import GridMeasureFeatures
from phenotypic.schema import OBJECT
from phenotypic.schema import NEIGHBOR_DIST, GRID, BBOX


class MeasureNeighborDist(GridMeasureFeatures):
    """Measure pixel-to-pixel distances to neighbors in adjacent grid cells.

    For each detected colony, identify the nearest object in the left,
    right, above, and below grid cells and report the minimum Euclidean
    distance between their pixel masks. Distances are computed via a
    per-section distance transform over a local window covering the target
    cell and its immediate neighbors, so round colonies are not
    over-estimated by their bounding boxes. Edge and corner colonies report
    ``NaN`` for directions beyond the plate boundary.

    Returns:
        pd.DataFrame: Object-level neighbor measurements with columns:

            - Label: unique object identifier.
            - LeftNeighborObjLabel, LeftDistance.
            - RightNeighborObjLabel, RightDistance.
            - AboveNeighborObjLabel, AboveDistance.
            - UnderNeighborObjLabel, UnderDistance.
            - ``NaN`` where no neighbor exists (out of plate, empty
              neighbor cell, or the colony is shielded by another colony
              in the same cell).

    Best For:
        - Quantifying colony spacing and crowding to assess nutrient
          competition risk on arrayed plates, especially for round or
          irregular colony shapes where bounding-box geometry overstates
          proximity.
        - Flagging closely spaced colonies that may cross-contaminate.
        - Enabling neighbor-aware paired statistical comparisons for
          competition or cooperation studies.

    Consider Also:
        - :class:`MeasureGridLinRegStats` for regression-based positional
          quality metrics.
        - :class:`MeasureGridSpread` for within-well colony dispersion
          rather than between-well distances.
        - :class:`MeasureBounds` for raw bounding boxes and centroids
          without neighbor lookups.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of grid-level measurements.
    """

    _measurement_infoclass: ClassVar[type] = NEIGHBOR_DIST

    # (d_row, d_col, label_col, dist_col)
    _DIRECTIONS: ClassVar[tuple] = (
        (0, -1, NEIGHBOR_DIST.LEFT_NEIGHBOR_OBJ_LABEL,
         NEIGHBOR_DIST.LEFT_DISTANCE),
        (0, +1, NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL,
         NEIGHBOR_DIST.RIGHT_DISTANCE),
        (-1, 0, NEIGHBOR_DIST.ABOVE_NEIGHBOR_OBJ_LABEL,
         NEIGHBOR_DIST.ABOVE_DISTANCE),
        (+1, 0, NEIGHBOR_DIST.UNDER_NEIGHBOR_OBJ_LABEL,
         NEIGHBOR_DIST.UNDER_DISTANCE),
    )

    def _operate(self, image: GridImage) -> pd.DataFrame:
        grid_info = image.grid.info(include_metadata=False)
        # Densify once; subsequent windowing is pure numpy slicing on this view.
        objmap_full = image.objmap[:]
        nrows, ncols = image.grid.nrows, image.grid.ncols
        # Each public edge getter re-fits the grid, so fetch both once and
        # memoize every occupied cell's window up front.
        section_bbox = self._section_bboxes(
                grid_info,
                image.grid.get_row_edges(),
                image.grid.get_col_edges(),
                height=image.shape[0],
                width=image.shape[1],
        )

        labels = grid_info[OBJECT.LABEL].to_numpy()
        n_objs = len(labels)
        label_to_row = {int(label): i for i, label in enumerate(labels)}

        results: dict = {
            NEIGHBOR_DIST.LEFT_NEIGHBOR_OBJ_LABEL : np.full(n_objs, np.nan),
            NEIGHBOR_DIST.LEFT_DISTANCE           : np.full(n_objs, np.nan),
            NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL: np.full(n_objs, np.nan),
            NEIGHBOR_DIST.RIGHT_DISTANCE          : np.full(n_objs, np.nan),
            NEIGHBOR_DIST.ABOVE_NEIGHBOR_OBJ_LABEL: np.full(n_objs, np.nan),
            NEIGHBOR_DIST.ABOVE_DISTANCE          : np.full(n_objs, np.nan),
            NEIGHBOR_DIST.UNDER_NEIGHBOR_OBJ_LABEL: np.full(n_objs, np.nan),
            NEIGHBOR_DIST.UNDER_DISTANCE          : np.full(n_objs, np.nan),
        }

        # Build (row, col) -> section_df once so neighbour lookups are O(1)
        # instead of an O(n_objects) DataFrame scan per direction per target.
        section_groups: dict[tuple[int, int], pd.DataFrame] = {}
        for (g_row, g_col), sec_df in grid_info.groupby(
                [GRID.ROW_NUM, GRID.COL_NUM], observed=True
        ):
            if pd.isna(g_row) or pd.isna(g_col):
                continue
            section_groups[(int(g_row), int(g_col))] = sec_df

        for (target_row, target_col), section_df in section_groups.items():
            target_labels = section_df[OBJECT.LABEL].to_numpy().astype(np.int64)
            if target_labels.size == 0:
                continue

            target_bbox = section_bbox[(target_row, target_col)]

            valid_neighbors = self._collect_neighbors(
                    section_groups, section_bbox,
                    target_row, target_col, nrows, ncols,
            )
            if not valid_neighbors:
                continue

            # Include target bbox so any target-object pixels that spilled past
            # the section's grid edges are still seeded into target_mask (the
            # object-fitting bounds may extend beyond the naive grid edges).
            all_bboxes = [target_bbox] + [n[2] for n in valid_neighbors]
            win = self._window_bbox(all_bboxes)

            objmap_win = objmap_full[win[0]:win[1] + 1, win[2]:win[3] + 1]
            target_mask = np.isin(objmap_win, target_labels)
            if not target_mask.any():
                continue

            # Distance from each window pixel to the nearest target-section
            # object pixel, with index back-pointers so we can attribute every
            # pixel to a specific target object.
            dt, indices = ndi.distance_transform_edt(
                    ~target_mask, return_indices=True
            )
            nearest_target_label = objmap_win[indices[0], indices[1]]

            for (_n_row, _n_col, n_bbox, n_labels,
                 label_col, dist_col) in valid_neighbors:
                lo_rr = n_bbox[0] - win[0]
                hi_rr = n_bbox[1] - win[0] + 1
                lo_cc = n_bbox[2] - win[2]
                hi_cc = n_bbox[3] - win[2] + 1

                objmap_n = objmap_win[lo_rr:hi_rr, lo_cc:hi_cc]
                neighbor_pixel_mask = np.isin(objmap_n, n_labels)
                if not neighbor_pixel_mask.any():
                    continue

                dt_n = dt[lo_rr:hi_rr, lo_cc:hi_cc]
                nearest_n = nearest_target_label[lo_rr:hi_rr, lo_cc:hi_cc]

                for target_label in target_labels:
                    cand = neighbor_pixel_mask & (nearest_n == target_label)
                    if not cand.any():
                        continue
                    cand_dt = dt_n[cand]
                    cand_obj = objmap_n[cand]
                    argmin = int(np.argmin(cand_dt))
                    row_idx = label_to_row[int(target_label)]
                    results[label_col][row_idx] = int(cand_obj[argmin])
                    results[dist_col][row_idx] = float(cand_dt[argmin])

        df = pd.DataFrame(results)
        df.insert(0, OBJECT.LABEL, labels)
        return df

    @staticmethod
    def _section_bboxes(
            grid_info: pd.DataFrame,
            row_edges: np.ndarray,
            col_edges: np.ndarray,
            *,
            height: int,
            width: int,
    ) -> dict[tuple[int, int], tuple[int, int, int, int]]:
        """Window (min_rr, max_rr, min_cc, max_cc) for every occupied grid cell.

        The cell's grid rectangle, widened to cover every object assigned to
        it (colonies may spill past a grid line), then clipped to the image.
        Built from public grid members only, once per measurement.
        """
        bboxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        for (g_row, g_col), sec in grid_info.groupby(
                [GRID.ROW_NUM, GRID.COL_NUM], observed=True
        ):
            if pd.isna(g_row) or pd.isna(g_col):
                continue
            r, c = int(g_row), int(g_col)
            min_rr = max(min(row_edges[r], sec[BBOX.MIN_RR].min()), 0)
            max_rr = min(max(row_edges[r + 1], sec[BBOX.MAX_RR].max()), height - 1)
            min_cc = max(min(col_edges[c], sec[BBOX.MIN_CC].min()), 0)
            max_cc = min(max(col_edges[c + 1], sec[BBOX.MAX_CC].max()), width - 1)
            bboxes[(r, c)] = (int(min_rr), int(max_rr), int(min_cc), int(max_cc))
        return bboxes

    @staticmethod
    def _window_bbox(
            bboxes: list[tuple[int, int, int, int]]
    ) -> tuple[int, int, int, int]:
        """Union of (min_rr, max_rr, min_cc, max_cc) bboxes."""
        arr = np.asarray(bboxes, dtype=np.int64)
        return (
            int(arr[:, 0].min()),
            int(arr[:, 1].max()),
            int(arr[:, 2].min()),
            int(arr[:, 3].max()),
        )

    def _collect_neighbors(
            self,
            section_groups: dict[tuple[int, int], pd.DataFrame],
            section_bbox: dict[tuple[int, int], tuple[int, int, int, int]],
            target_row: int,
            target_col: int,
            nrows: int,
            ncols: int,
    ) -> list[tuple[int, int, tuple[int, int, int, int], np.ndarray, str, str]]:
        """Enumerate valid (in-bounds, non-empty) neighbor sections."""
        valid: list[tuple[
            int, int, tuple[int, int, int, int], np.ndarray, str, str
        ]] = []
        for d_row, d_col, label_col, dist_col in self._DIRECTIONS:
            n_row, n_col = target_row + d_row, target_col + d_col
            if not (0 <= n_row < nrows and 0 <= n_col < ncols):
                continue
            n_section_df = section_groups.get((n_row, n_col))
            if n_section_df is None or n_section_df.empty:
                continue
            n_labels = n_section_df[OBJECT.LABEL].to_numpy().astype(np.int64)
            valid.append((n_row, n_col, section_bbox[(n_row, n_col)],
                          n_labels, label_col, dist_col))
        return valid


MeasureNeighborDist.__doc__ = NEIGHBOR_DIST.append_rst_to_doc(MeasureNeighborDist)
