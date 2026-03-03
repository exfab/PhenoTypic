from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import pandas as pd
import numpy as np

from phenotypic.abc_ import GridFinder
from phenotypic.tools_.measurement_info_ import BBOX, GRID


class AutoGridFinder(GridFinder):
    """
    Automatically determines grid row and column edges from detected object
    centers using a deterministic robust-fit algorithm.

    Unlike histogram or optimizer-based approaches, this class fits a regular
    grid model directly to the weighted centroids of detected objects. Outlier
    rejection ensures that protruding colonies (e.g., filamentous fungi with
    extended hyphae) do not pull grid boundaries away from the true positions.

    Args:
        nrows: Number of rows in the grid (default 8 for 96-well plates).
        ncols: Number of columns in the grid (default 12 for 96-well plates).
        residual_fraction: Outlier threshold as a fraction of pitch. Centers
            whose fit residual exceeds ``pitch * residual_fraction`` are
            excluded from the refined fit (default 0.25).
        tol: Deprecated. Accepted for backward compatibility but ignored.
        max_iter: Deprecated. Accepted for backward compatibility but ignored.
    """

    def __init__(
            self,
            nrows: int = 8,
            ncols: int = 12,
            residual_fraction: float = 0.25,
            *,
            tol: float | None = None,
            max_iter: int | None = None,
    ):
        super().__init__(nrows=nrows, ncols=ncols)
        self.residual_fraction: float = residual_fraction

        if tol is not None:
            warnings.warn(
                "The 'tol' parameter is deprecated and has no effect. "
                "AutoGridFinder now uses a deterministic robust-fit algorithm.",
                DeprecationWarning,
                stacklevel=2,
            )
        if max_iter is not None:
            warnings.warn(
                "The 'max_iter' parameter is deprecated and has no effect. "
                "AutoGridFinder now uses a deterministic robust-fit algorithm.",
                DeprecationWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Static helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_axis_centers(info_table: pd.DataFrame, axis: int) -> np.ndarray:
        """Return sorted weighted centers along *axis* (0=rows, 1=cols)."""
        if axis == 0:
            col = str(BBOX.WEIGHTED_CENTER_RR)
        elif axis == 1:
            col = str(BBOX.WEIGHTED_CENTER_CC)
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")
        centers = info_table.loc[:, col].values.astype(float)
        centers.sort()
        return centers

    @staticmethod
    def _estimate_pitch(centers: np.ndarray, n_expected: int) -> float:
        """Estimate grid pitch from sorted centers and expected grid count.

        Uses ``(max - min) / (n_expected - 1)`` which is robust when multiple
        objects share a grid cell (common in colony imaging where fragments or
        sub-colonies yield many detections per well).
        """
        if len(centers) < 2:
            raise ValueError("Need at least 2 centers to estimate pitch")
        return float((centers[-1] - centers[0]) / max(n_expected - 1, 1))

    @staticmethod
    def _assign_grid_indices(centers: np.ndarray, pitch: float) -> np.ndarray:
        """Assign integer grid indices: ``round((c - min_c) / pitch)``."""
        return np.rint((centers - centers[0]) / pitch).astype(int)

    @staticmethod
    def _fit_pitch_and_offset(
            centers: np.ndarray, indices: np.ndarray,
    ) -> tuple[float, float]:
        """Closed-form linear fit ``center = pitch * idx + offset``.

        Returns:
            Tuple of (pitch, offset).
        """
        idx_mean = indices.mean()
        ctr_mean = centers.mean()
        idx_dev = indices - idx_mean
        denom = float(idx_dev @ idx_dev)
        if denom == 0.0:
            return 0.0, ctr_mean
        pitch = float(idx_dev @ (centers - ctr_mean)) / denom
        offset = ctr_mean - pitch * idx_mean
        return pitch, offset

    @staticmethod
    def _identify_inliers(
            centers: np.ndarray,
            indices: np.ndarray,
            pitch: float,
            offset: float,
            threshold: float,
    ) -> np.ndarray:
        """Return boolean mask where ``|residual| <= threshold``."""
        predicted = pitch * indices + offset
        residuals = np.abs(centers - predicted)
        return residuals <= threshold

    @staticmethod
    def _compute_grid_edges(
            pitch: float,
            offset: float,
            n_bins: int,
            image_dim: int,
    ) -> np.ndarray:
        """Compute ``n_bins + 1`` edge coordinates clipped to ``[0, image_dim]``.

        Edges are placed at ``offset + pitch * i - pitch / 2`` for
        ``i = 0 .. n_bins``.
        """
        edges = offset + pitch * np.arange(n_bins + 1) - pitch / 2
        np.clip(edges, 0, image_dim, out=edges)
        np.round(edges, out=edges)
        return edges.astype(int)

    # ------------------------------------------------------------------
    # Axis-level orchestrator
    # ------------------------------------------------------------------

    @staticmethod
    def _uniform_edges(n_expected: int, image_dim: int) -> np.ndarray:
        """Fallback: uniform spacing centered in image."""
        pitch = image_dim / n_expected
        return AutoGridFinder._compute_grid_edges(
            pitch, pitch / 2, n_expected, image_dim,
        )

    def _fit_axis_edges(
            self,
            info_table: pd.DataFrame,
            axis: int,
            n_expected: int,
            image_dim: int,
    ) -> np.ndarray:
        """Full pipeline: extract centers → fit → reject → refit → edges.

        Falls back to uniform spacing when fewer than 2 objects are found.
        Uses symmetry anchoring when detected span is less than expected.
        """
        try:
            centers = self._extract_axis_centers(info_table, axis)
        except (KeyError, IndexError):
            centers = np.array([])

        if len(centers) < 2:
            return self._uniform_edges(n_expected, image_dim)

        # Step 1: initial pitch estimate
        pitch = self._estimate_pitch(centers, n_expected)
        if pitch <= 0:
            return self._uniform_edges(n_expected, image_dim)

        # Step 2: assign grid indices + initial fit
        indices = self._assign_grid_indices(centers, pitch)
        pitch, offset = self._fit_pitch_and_offset(centers, indices)
        if pitch <= 0:
            return self._uniform_edges(n_expected, image_dim)

        # Step 3: outlier rejection + refit
        threshold = pitch * self.residual_fraction
        inlier_mask = self._identify_inliers(
            centers, indices, pitch, offset, threshold,
        )
        if inlier_mask.sum() >= 2:
            pitch, offset = self._fit_pitch_and_offset(
                centers[inlier_mask], indices[inlier_mask],
            )
        if pitch <= 0:
            return self._uniform_edges(n_expected, image_dim)

        # Step 4: symmetry anchoring when detected span < expected
        span = int(indices.max() - indices.min()) + 1
        if span < n_expected:
            image_center = image_dim / 2.0
            grid_center_idx = (n_expected - 1) / 2.0
            offset = image_center - pitch * grid_center_idx

        return self._compute_grid_edges(pitch, offset, n_expected, image_dim)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_row_edges(self, image: Image) -> np.ndarray:
        """Return row edge coordinates for *image*.

        Args:
            image: Image with detected objects (``image.objects.info()``).

        Returns:
            Integer array of length ``nrows + 1``.
        """
        info_table = image.objects.info(include_metadata=False)
        return self._fit_axis_edges(
            info_table, axis=0, n_expected=self.nrows, image_dim=image.shape[0],
        )

    def get_col_edges(self, image: Image) -> np.ndarray:
        """Return column edge coordinates for *image*.

        Args:
            image: Image with detected objects (``image.objects.info()``).

        Returns:
            Integer array of length ``ncols + 1``.
        """
        info_table = image.objects.info(include_metadata=False)
        return self._fit_axis_edges(
            info_table, axis=1, n_expected=self.ncols, image_dim=image.shape[1],
        )

    def _operate(self, image: Image) -> pd.DataFrame:
        """Compute grid edges and assign each detected object to a grid cell.

        Args:
            image: Image with detected objects.

        Returns:
            DataFrame with grid assignments (ROW_NUM, COL_NUM, ROW_MAJOR_IDX).
        """
        info_table = image.objects.info(include_metadata=False)
        row_edges = self._fit_axis_edges(
            info_table, axis=0, n_expected=self.nrows, image_dim=image.shape[0],
        )
        col_edges = self._fit_axis_edges(
            info_table, axis=1, n_expected=self.ncols, image_dim=image.shape[1],
        )
        return super()._get_grid_info(
            image=image, row_edges=row_edges, col_edges=col_edges,
        )


AutoGridFinder.measure.__doc__ = AutoGridFinder._operate.__doc__
AutoGridFinder.__doc__ = GRID.append_rst_to_doc(AutoGridFinder.__doc__)
