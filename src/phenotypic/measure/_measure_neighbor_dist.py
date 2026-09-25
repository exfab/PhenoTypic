"""Measure spatial relationships between neighboring colonies in grid cells."""

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from phenotypic import Image
    from phenotypic._core._grid_image import GridImage

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from phenotypic.abc_ import MeasureFeatures
from phenotypic.schema import OBJECT
from phenotypic.schema import NEIGHBOR_DIST, GRID, BBOX


def _boundary_coords(
        objmap: np.ndarray, label: int, sl: tuple[slice, slice]
) -> np.ndarray:
    """Full-image (row, col) coordinates of ``label``'s 4-connected boundary.

    A pixel is boundary when at least one 4-neighbour is outside the object;
    hole edges count. The minimum distance between two disjoint masks is always
    attained on these pixels (spec §3.6 C1). Pixels on the crop edge are
    boundary because ``binary_erosion`` treats outside-the-crop as background,
    which is correct for the tight ``find_objects`` slice.
    """
    mask = objmap[sl] == label
    edge = mask & ~ndi.binary_erosion(mask)
    coords = np.argwhere(edge)
    coords[:, 0] += sl[0].start
    coords[:, 1] += sl[1].start
    return coords


def _nearest_objects(
        objmap: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact nearest other object for every eligible label.

    Branch and bound: candidates are visited in ascending (bounding-box lower
    bound, label) order, and the search stops at the first candidate whose
    lower bound strictly exceeds the best exact distance found. The box gap
    never exceeds the mask distance (C2), so no nearer object is skipped, and
    equal-distance candidates are still visited for the smaller-label
    tie-break (C3).

    Args:
        objmap: 2-D integer label map.
        eligible: 1-D labels to search among; only these are targets or
            candidates.

    Returns:
        ``(nearest_label, nearest_distance)`` float arrays aligned to
        ``eligible``; ``NaN`` where no other eligible object exists.

    Raises:
        ValueError: If an eligible label has no pixels in ``objmap``.
    """
    eligible = np.asarray(eligible, dtype=np.int64)
    n = eligible.size
    nearest_label = np.full(n, np.nan)
    nearest_dist = np.full(n, np.nan)
    if n < 2:
        return nearest_label, nearest_dist

    slices = ndi.find_objects(objmap)
    boundaries: list[np.ndarray] = []
    bboxes = np.empty((n, 4), dtype=np.int64)  # inclusive min_r, max_r, min_c, max_c
    for i, label in enumerate(eligible):
        sl = slices[label - 1] if 0 < label <= len(slices) else None
        if sl is None:
            raise ValueError(f"eligible label {label} has no pixels in objmap")
        coords = _boundary_coords(objmap, int(label), sl)
        boundaries.append(coords)
        bboxes[i] = (coords[:, 0].min(), coords[:, 0].max(),
                     coords[:, 1].min(), coords[:, 1].max())

    trees: dict[int, cKDTree] = {}
    for i in range(n):
        gap_r = np.maximum(0, np.maximum(bboxes[:, 0] - bboxes[i, 1],
                                         bboxes[i, 0] - bboxes[:, 1]))
        gap_c = np.maximum(0, np.maximum(bboxes[:, 2] - bboxes[i, 3],
                                         bboxes[i, 2] - bboxes[:, 3]))
        # sqrt of an exact integer sum, the same rounding path cKDTree uses, so
        # a lower bound equal to a true distance compares equal rather than
        # 1 ulp high (np.hypot is not guaranteed correctly rounded on every libm),
        # and tied candidates are never skipped.
        lower = np.sqrt((gap_r * gap_r + gap_c * gap_c).astype(np.float64))
        lower[i] = np.inf
        best, best_j = np.inf, -1
        for j in np.lexsort((eligible, lower)):
            if j == i or lower[j] > best:
                break
            if j not in trees:
                trees[j] = cKDTree(boundaries[j])
            d = float(np.min(trees[j].query(boundaries[i], k=1)[0]))
            if d < best or (d == best and eligible[j] < eligible[best_j]):
                best, best_j = d, int(j)
        nearest_label[i] = eligible[best_j]
        nearest_dist[i] = best
    return nearest_label, nearest_dist


def _nearest_relation(self_rc: np.ndarray, nearest_rc: np.ndarray) -> np.ndarray:
    """Grid relation code per spec §3.4 from (row, col) cell positions."""
    dr = np.abs(nearest_rc[:, 0] - self_rc[:, 0])
    dc = np.abs(nearest_rc[:, 1] - self_rc[:, 1])
    step = dr + dc
    return np.select(
            [step == 0, step == 1, (dr == 1) & (dc == 1)],
            [0.0, 1.0, 2.0],
            default=3.0,
    )


class MeasureNeighborDist(MeasureFeatures):
    """Measure distances to grid neighbours and to the nearest object.

    For each detected colony, report (a) the nearest object in the left,
    right, above, and below grid cells with the minimum Euclidean distance
    between their pixel masks, and (b) the single closest other object
    anywhere on the plate, with its distance and its grid relation (same cell,
    adjacent, diagonal, or further). Directional distances use a per-section
    distance transform over a local window; the nearest-object search is an
    exact branch and bound over mask boundaries. Works on a ``GridImage``
    (objects without a grid cell are excluded from the nearest search) and on a
    plain ``Image``, where only the nearest label and distance are populated.

    Returns:
        pd.DataFrame: Object-level neighbor measurements with columns:

            - Label: unique object identifier.
            - LeftNeighborObjLabel, LeftDistance.
            - RightNeighborObjLabel, RightDistance.
            - AboveNeighborObjLabel, AboveDistance.
            - UnderNeighborObjLabel, UnderDistance.
            - NearestObjLabel, NearestDistance, NearestRelation.
            - ``NaN`` where no neighbor exists (out of plate, empty
              neighbor cell, shielded by a cellmate, no grid, or fewer than
              two eligible objects).

    Best For:
        - Quantifying colony spacing and crowding to assess nutrient
          competition risk on arrayed plates, especially for round or
          irregular colony shapes where bounding-box geometry overstates
          proximity.
        - Screening for satellite colonies, fragments, and contaminants:
          a small ``NearestDistance`` with a same-cell ``NearestRelation``.
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

    def _operate(self, image: Image) -> pd.DataFrame:
        # Densify once; subsequent windowing is pure numpy slicing on this view.
        objmap = image.objmap[:]
        directional = [col for _, _, lab, dist in self._DIRECTIONS
                       for col in (lab, dist)]
        if hasattr(image, "grid"):
            info = image.grid.info(include_metadata=False)
            results = self._measure_grid_directions(
                    cast("GridImage", image), info, objmap
            )
            grid_rc = info[[GRID.ROW_NUM, GRID.COL_NUM]].to_numpy(dtype=float)
            eligible_rows = np.flatnonzero(~np.isnan(grid_rc).any(axis=1))
        else:
            info = image.objects.info(include_metadata=False)
            results = {col: np.full(len(info), np.nan) for col in directional}
            grid_rc = None
            eligible_rows = np.arange(len(info))

        labels = info[OBJECT.LABEL].to_numpy().astype(np.int64)
        n_objs = labels.size
        nearest_label = np.full(n_objs, np.nan)
        nearest_dist = np.full(n_objs, np.nan)
        relation = np.full(n_objs, np.nan)

        e_label, e_dist = _nearest_objects(objmap, labels[eligible_rows])
        nearest_label[eligible_rows] = e_label
        nearest_dist[eligible_rows] = e_dist

        if grid_rc is not None:
            label_to_row = {int(lab): i for i, lab in enumerate(labels)}
            found = ~np.isnan(e_label)
            rows_self = eligible_rows[found]
            rows_near = np.array(
                    [label_to_row[int(lab)] for lab in e_label[found]],
                    dtype=np.int64,
            )
            relation[rows_self] = _nearest_relation(
                    grid_rc[rows_self], grid_rc[rows_near]
            )

        results[NEIGHBOR_DIST.NEAREST_OBJ_LABEL] = nearest_label
        results[NEIGHBOR_DIST.NEAREST_DISTANCE] = nearest_dist
        results[NEIGHBOR_DIST.NEAREST_RELATION] = relation

        df = pd.DataFrame(results)
        df.insert(0, OBJECT.LABEL, info[OBJECT.LABEL].to_numpy())
        return df

    def _measure_grid_directions(
            self,
            image: GridImage,
            grid_info: pd.DataFrame,
            objmap_full: np.ndarray,
    ) -> dict:
        """The eight directional columns, keyed in enum order."""
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

        return results

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
        for key, sec in grid_info.groupby(
                [GRID.ROW_NUM, GRID.COL_NUM], observed=True
        ):
            g_row, g_col = cast("tuple[float, float]", key)
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
