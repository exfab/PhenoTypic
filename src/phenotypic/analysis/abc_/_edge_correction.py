from __future__ import annotations

import abc
from abc import ABC
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pydantic import PrivateAttr, field_validator

from phenotypic.sdk_ import ColumnRef

from ._set_analyzer import SetAnalyzer


class EdgeCorrection(SetAnalyzer, ABC):
    """Abstract base for grid-aware edge-effect correction strategies.

    Holds the grid-layout configuration (``nrows``/``ncols``/
    ``connectivity``/``time_label``) and the neighbor-topology machinery
    (:meth:`_surrounded_positions`) shared by every edge corrector, and
    drives the standard grouped-correction template in :meth:`analyze`.
    Concrete subclasses provide the per-group correction by implementing
    the static :meth:`_apply2group_func` worker plus :meth:`_group_config`,
    which supplies the kwargs forwarded to it.

    Attributes:
        time_label (str): Column holding the time point.
        nrows (int): Grid rows.
        ncols (int): Grid columns.
        connectivity (int): Neighbor pattern: 4 (orthogonal) or 8 (with
            diagonals).
    """

    time_label: ColumnRef = "Metadata_Time"
    nrows: int = 8
    ncols: int = 12
    connectivity: int = 4

    _original_data: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)

    @field_validator("connectivity")
    @classmethod
    def _validate_connectivity(cls, value: int) -> int:
        """Reject connectivity patterns other than 4 or 8."""
        if value not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {value}")
        return value

    @field_validator("nrows", "ncols")
    @classmethod
    def _validate_grid_dim(cls, value: int) -> int:
        """Reject non-positive grid dimensions."""
        if value <= 0:
            raise ValueError(f"nrows and ncols must be positive, got {value}")
        return value

    @staticmethod
    def _surrounded_positions(
            active_idx: np.ndarray | list[int],
            shape: tuple[int, int],
            connectivity: int = 4,
            min_neighbors: int | None = None,
            return_counts: bool = False,
            dtype: np.dtype = np.int64,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Find grid cells that are surrounded by active neighbors.

        This function identifies cells in a 2D grid that have a sufficient number
        of active neighbors based on the specified connectivity pattern. Input uses
        flattened indices in C-order (row-major).

        Args:
            active_idx: Flattened indices of active cells. Will be deduplicated.
            shape: Grid dimensions as (rows, cols).
            connectivity: Neighbor pattern. Must be 4 (N,S,E,W) or 8 (adds diagonals).
            min_neighbors: Minimum number of active neighbors required. If None,
                requires all neighbors in the connectivity pattern to be active
                (fully surrounded). Border cells cannot qualify when None.
            return_counts: If True, also return the neighbor counts for selected indices.
            dtype: Data type for output arrays.

        Returns:
            If return_counts is False:
                Sorted array of flattened indices meeting the neighbor criterion.
            If return_counts is True:
                Tuple of (indices, counts) where counts[i] is the number of active
                neighbors for indices[i].

        Raises:
            ValueError: If connectivity is not 4 or 8, if any active_idx is out of
                bounds, if min_neighbors is invalid, or if shape is invalid.

        Notes:
            - Flattening uses C-order: idx = row * cols + col
            - When min_neighbors=None, border cells are geometrically excluded since
              they cannot have all neighbors active
            - Results are always sorted for deterministic output

        Examples:
            Finding fully surrounded and partially surrounded cells on an 8x12 grid:

            >>> import numpy as np
            >>> # 8x12 plate; 3x3 active block centered at (4,6)
            >>> rows, cols = 8, 12
            >>> block_rc = [(r, c) for r in range(3, 6) for c in range(5, 8)]
            >>> active = np.array([r*cols + c for r, c in block_rc], dtype=np.int64)
            >>> # Fully surrounded (default, since min_neighbors=None -> all)
            >>> res_all = EdgeCorrection._surrounded_positions(active, (rows, cols), connectivity=4)
            >>> assert np.array_equal(res_all, np.array([4*cols + 6], dtype=np.int64))
            >>> # Threshold: at least 3 of 4 neighbors
            >>> idxs, counts = EdgeCorrection._surrounded_positions(
            ...     active, (rows, cols), connectivity=4, min_neighbors=3, return_counts=True
            ... )
            >>> assert (counts >= 3).all()
            >>> assert (4*cols + 6) in idxs  # center has 4
        """
        # Validate connectivity
        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

        # Validate shape
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"shape must be two positive integers, got {shape}")

        rows, cols = shape
        total_cells = rows * cols

        # Coerce active_idx to 1D unique array
        active_idx = np.asarray(active_idx, dtype=dtype).ravel()
        active_idx = np.unique(active_idx)

        # Validate bounds
        if len(active_idx) > 0:
            if active_idx.min() < 0 or active_idx.max() >= total_cells:
                raise ValueError(
                        f"All active_idx must be in [0, {total_cells}), "
                        f"got range [{active_idx.min()}, {active_idx.max()}]"
                )

        # Determine max_neighbors and validate min_neighbors
        max_neighbors = connectivity
        if min_neighbors is None:
            min_neighbors = max_neighbors
        else:
            if not (1 <= min_neighbors <= max_neighbors):
                raise ValueError(
                        f"min_neighbors must be in [1, {max_neighbors}], got {min_neighbors}"
                )

        # Handle empty input
        if len(active_idx) == 0:
            if return_counts:
                return np.array([], dtype=dtype), np.array([], dtype=dtype)
            return np.array([], dtype=dtype)

        # Build active mask
        active_mask = np.zeros((rows, cols), dtype=bool)
        rows_idx = active_idx // cols
        cols_idx = active_idx % cols
        active_mask[rows_idx, cols_idx] = True

        # Define neighbor offsets based on connectivity
        if connectivity == 4:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:  # connectivity == 8
            offsets = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),  # cardinal
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),  # diagonal
            ]

        # Accumulate neighbor counts using aligned slicing
        neighbor_count = np.zeros((rows, cols), dtype=np.int32)

        for dr, dc in offsets:
            # Calculate slice bounds for source (active_mask)
            src_r_start = max(0, -dr)
            src_r_end = rows - max(0, dr)
            src_c_start = max(0, -dc)
            src_c_end = cols - max(0, dc)

            # Calculate slice bounds for destination (neighbor_count)
            dst_r_start = max(0, dr)
            dst_r_end = rows - max(0, -dr)
            dst_c_start = max(0, dc)
            dst_c_end = cols - max(0, -dc)

            # Extract views
            src_view = active_mask[src_r_start:src_r_end, src_c_start:src_c_end]
            dst_view = neighbor_count[dst_r_start:dst_r_end, dst_c_start:dst_c_end]

            # Accumulate
            dst_view += src_view.astype(np.int32)

        # Select cells that are active AND have sufficient neighbors
        sufficient_neighbors = neighbor_count >= min_neighbors
        selected_mask = active_mask & sufficient_neighbors

        # Convert back to flattened indices
        selected_rows, selected_cols = np.where(selected_mask)
        result_idx = (selected_rows * cols + selected_cols).astype(dtype)
        result_idx = np.sort(result_idx)

        if return_counts:
            # Get counts for selected indices
            counts = neighbor_count[selected_rows, selected_cols].astype(dtype)
            # Sort counts to match sorted indices
            sort_order = np.argsort(selected_rows * cols + selected_cols)
            counts = counts[sort_order]
            return result_idx, counts

        return result_idx

    @abc.abstractmethod
    def _group_config(self) -> dict[str, Any]:
        """Per-group kwargs forwarded to :meth:`_apply2group_func`."""

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the edge-correction strategy group-by-group.

        Validates the frame, stores the pre-correction copy on
        ``self._original_data``, aggregates to one row per well per group,
        then dispatches each group to the static :meth:`_apply2group_func`
        worker (serial for a single group, joblib-parallel otherwise) using
        the kwargs from :meth:`_group_config`.
        """
        from phenotypic.schema import GRID

        if data is None or len(data) == 0:
            raise ValueError("Input data cannot be empty")

        self._original_data = data

        section_col = str(GRID.ROW_MAJOR_IDX)
        required_cols = set(self.groupby + [section_col, self.on])
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        groupby_cols = self.groupby + [section_col]
        if self.time_label in data:
            groupby_cols = groupby_cols + [self.time_label]

        agg_dict: dict[str, Any] = {}
        for col in data.columns:
            if col not in groupby_cols:
                agg_dict[col] = self.agg_func if col == self.on else "first"

        agg_data = data.groupby(by=groupby_cols, as_index=False).agg(agg_dict)

        config = self._group_config()
        if len(self.groupby) == 0:
            corrected_data = [self.__class__._apply2group_func(agg_data, **config)]
        else:
            grouped = agg_data.groupby(by=self.groupby, as_index=False)
            corrected_data = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.__class__._apply2group_func)(group, **config)
                    for _, group in grouped
            )

        if corrected_data:
            self._latest_measurements = pd.concat(corrected_data, ignore_index=True)
        else:
            self._latest_measurements = pd.DataFrame()
        return self._latest_measurements

    def show(self):
        """Visualize edge correction results.

        Raises:
            NotImplementedError: Subclasses must provide their own visualization.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement show(). "
            "Override in a concrete subclass."
        )

    def results(self) -> pd.DataFrame:
        """Return the corrected measurements from the last :meth:`analyze` call.

        Returns:
            pd.DataFrame: Edge-corrected measurement DataFrame. Empty if
                :meth:`analyze` has not been called.
        """
        return self._latest_measurements
