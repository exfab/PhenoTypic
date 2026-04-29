from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import pandas as pd
import numpy as np

from phenotypic.abc_ import GridFinder
from phenotypic.tools_.measurement_info import BBOX, GRID


class AutoGridFinderFallbackWarning(UserWarning):
    """Warning category for fallbacks and geometry overrides in
    :class:`AutoGridFinder`.

    Emitted whenever the grid fitter takes a fallback (uniform spacing,
    switch to simple pipeline, span-based pitch) or overrides the
    fitted offset (symmetry anchoring, pitch shrink). Use this category
    to filter only AutoGridFinder diagnostics in batch runs::

        import warnings
        from phenotypic.grid._auto_grid_finder import (
            AutoGridFinderFallbackWarning,
        )
        warnings.filterwarnings(
            "ignore", category=AutoGridFinderFallbackWarning,
        )
    """


class AutoGridFinder(GridFinder):
    """
    Automatically determines grid row and column edges from detected object
    centers using a deterministic robust-fit algorithm.

    Unlike histogram or optimizer-based approaches, this class fits a regular
    grid model directly to the per-object distance-transform maximum centers
    (deepest interior point of each object's mask). These centers are
    anchored in the dense colony body and are unaffected by thin filamentous
    extensions (e.g., fungal hyphae) that would otherwise pull
    intensity-weighted centroids off-body and bias the grid fit. Outlier
    rejection further protects against atypical objects pulling boundaries
    away from the true positions.

    Args:
        nrows: Number of rows in the grid (default 8 for 96-well plates).
        ncols: Number of columns in the grid (default 12 for 96-well plates).
        residual_fraction: Outlier threshold as a fraction of pitch. Centers

            whose fit residual exceeds ``pitch * residual_fraction`` are
            excluded from the refined fit (default 0.25).

        warn: Whether to emit :class:`AutoGridFinderFallbackWarning`
            diagnostics for fallbacks and geometry overrides. Defaults
            to ``False`` (silent); set to ``True`` to surface them.
        tol: Deprecated. Accepted for backward compatibility but ignored.
        max_iter: Deprecated. Accepted for backward compatibility but ignored.

    Notes:
        Diagnostic warnings of category
        :class:`AutoGridFinderFallbackWarning` are emitted at every
        fallback (uniform spacing, simple-pipeline switch, span-based
        pitch) and geometry override (symmetry anchoring, pitch shrink).
        Suppress them in batch runs with::

            import warnings
            from phenotypic.grid._auto_grid_finder import (
                AutoGridFinderFallbackWarning,
            )
            warnings.filterwarnings(
                "ignore", category=AutoGridFinderFallbackWarning,
            )
    """

    _MAX_OBJECTS_PER_CELL: int = 250

    def __init__(
            self,
            nrows: int = 8,
            ncols: int = 12,
            residual_fraction: float = 0.25,
            *,
            warn: bool = False,
            tol: float | None = None,
            max_iter: int | None = None,
    ):
        super().__init__(nrows=nrows, ncols=ncols)
        self.residual_fraction: float = residual_fraction
        self.warn: bool = warn

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

    @contextmanager
    def _warning_filter(self):
        """Suppress :class:`AutoGridFinderFallbackWarning` when ``self.warn`` is False."""
        if self.warn:
            yield
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AutoGridFinderFallbackWarning)
                yield

    # ------------------------------------------------------------------
    # Static helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_axis_centers(info_table: pd.DataFrame, axis: int) -> np.ndarray:
        """Return sorted weighted centers along *axis* (0=rows, 1=cols)."""
        if axis == 0:
            col = str(BBOX.DIST_WEIGHTED_CENTER_RR)
        elif axis == 1:
            col = str(BBOX.DIST_WEIGHTED_CENTER_CC)
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
    def _assign_grid_indices(
            centers: np.ndarray,
            pitch: float,
            anchor: float | None = None,
    ) -> np.ndarray:
        """Assign integer grid indices: ``round((c - anchor) / pitch)``.

        Args:
            centers: Sorted 1-D array of object center coordinates.
            pitch: Estimated grid pitch.
            anchor: Reference coordinate for index 0.  When *None*,
                ``centers[0]`` is used (original behaviour).
        """
        ref = centers[0] if anchor is None else anchor
        indices = np.rint((centers - ref) / pitch).astype(int)
        indices -= indices.min()
        return indices

    @staticmethod
    def _aggregate_to_cell_medians(
            centers: np.ndarray, indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collapse multiple centers per grid index to one median each.

        Returns:
            Tuple of (median_centers, unique_indices), one entry per
            occupied grid slot.
        """
        grouped = pd.Series(centers).groupby(indices).median()
        return grouped.values, grouped.index.values

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
            warnings.warn(
                f"AutoGridFinder._fit_pitch_and_offset: degenerate indices "
                f"(all {len(indices)} centers map to the same grid index); "
                f"pitch set to 0.0, offset to centers mean ({ctr_mean:.2f}).",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
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

        The offset is clamped so that the first edge is >= 0 and the
        last edge is <= image_dim, preventing duplicate edges after
        clipping.  When the fitted pitch is too large for the image,
        the pitch is shrunk to ``image_dim / n_bins`` so the grid
        fills the available space.
        """
        half = pitch / 2.0
        min_offset = half                              # first edge >= 0
        max_offset = image_dim - pitch * n_bins + half  # last edge <= image_dim

        if min_offset > max_offset:
            warnings.warn(
                f"AutoGridFinder._compute_grid_edges: fitted pitch "
                f"({pitch:.2f}) too large for image_dim={image_dim} with "
                f"n_bins={n_bins}; shrinking pitch to "
                f"{image_dim / n_bins:.2f} and recentering.",
                AutoGridFinderFallbackWarning,
                stacklevel=3,
            )
            # Pitch is too large for the image — shrink to fit
            pitch = image_dim / n_bins
            half = pitch / 2.0
            offset = half
        else:
            offset = float(np.clip(offset, min_offset, max_offset))

        edges = offset + pitch * np.arange(n_bins + 1) - half
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

    @staticmethod
    def _image_pitch_indices(
            centers: np.ndarray, n_expected: int, image_dim: int,
    ) -> np.ndarray:
        """Assign cell indices using image-pitch prior (sparse-robust).

        Independent of any fitted pitch, so usable as diagnostic ground
        truth even when the fitter has failed. Centers are rounded to
        the nearest cell using ``image_dim / n_expected`` as pitch with
        cell 0 anchored at ``image_pitch / 2``. Indices are clipped to
        ``[0, n_expected - 1]``.
        """
        if len(centers) == 0:
            return np.empty(0, dtype=int)
        image_pitch = image_dim / n_expected
        anchor = image_pitch / 2.0
        idx = np.rint((centers - anchor) / image_pitch).astype(int)
        return np.clip(idx, 0, n_expected - 1)

    @staticmethod
    def _classify_axis_pipeline(
            centers: np.ndarray, n_expected: int, image_dim: int,
    ) -> str:
        """Return ``"iterative"`` or ``"simple"`` per the per-axis gate.

        The simple pipeline runs only when BOTH the raw count and the
        per-axis occupancy (via :meth:`_image_pitch_indices`) are too
        low for iterative refinement to seed reliably:

        - ``count_low``: ``len(centers) < 1.5 * n_expected``
        - ``occupancy_low``: ``n_occupied < 0.5 * n_expected``

        Either condition alone routes to ``"iterative"``. This helper
        is the single source of truth for the gate, used by
        :meth:`_fit_axis_edges` (dispatch) and
        :meth:`_run_timed_pipeline` (dashboard label).
        """
        if len(centers) < 2:
            return "iterative"  # caller handles len < 2 separately
        axis_indices = AutoGridFinder._image_pitch_indices(
            centers, n_expected, image_dim,
        )
        n_occupied = int(np.unique(axis_indices).size)
        count_low = len(centers) < 1.5 * n_expected
        occupancy_low = n_occupied < 0.5 * n_expected
        return "simple" if (count_low and occupancy_low) else "iterative"

    def _fit_axis_edges_simple(
            self,
            centers: np.ndarray,
            n_expected: int,
            image_dim: int,
            axis_label: str = "axis",
    ) -> np.ndarray:
        """Simple pipeline used when the object count is low.

        Uses span-based pitch estimation and fits all centers directly.
        """
        pitch = self._estimate_pitch(centers, n_expected)
        if pitch <= 0:
            warnings.warn(
                f"AutoGridFinder ({axis_label}, simple): span-based pitch "
                f"<= 0 ({pitch}); falling back to uniform edges.",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
            return self._uniform_edges(n_expected, image_dim)

        indices = self._assign_grid_indices(centers, pitch)
        pitch, offset = self._fit_pitch_and_offset(centers, indices)
        if pitch <= 0:
            warnings.warn(
                f"AutoGridFinder ({axis_label}, simple): initial fit pitch "
                f"<= 0 ({pitch}); falling back to uniform edges.",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
            return self._uniform_edges(n_expected, image_dim)

        threshold = pitch * self.residual_fraction
        inlier_mask = self._identify_inliers(
            centers, indices, pitch, offset, threshold,
        )
        n_inliers = int(inlier_mask.sum())
        if n_inliers >= 2:
            pitch, offset = self._fit_pitch_and_offset(
                centers[inlier_mask], indices[inlier_mask],
            )
        else:
            warnings.warn(
                f"AutoGridFinder ({axis_label}, simple): only {n_inliers} "
                f"inliers of {len(centers)} centers within residual threshold "
                f"({threshold:.2f}); skipping refit and using initial fit.",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
        if pitch <= 0:
            warnings.warn(
                f"AutoGridFinder ({axis_label}, simple): refit pitch <= 0 "
                f"({pitch}); falling back to uniform edges.",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
            return self._uniform_edges(n_expected, image_dim)

        span = int(indices.max() - indices.min()) + 1
        if span < n_expected:
            warnings.warn(
                f"AutoGridFinder ({axis_label}, simple): detected span "
                f"({span}) < n_expected ({n_expected}); recentering grid on "
                f"image (symmetry anchoring overrides fitted offset).",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
            image_center = image_dim / 2.0
            grid_center_idx = (n_expected - 1) / 2.0
            offset = image_center - pitch * grid_center_idx

        return self._compute_grid_edges(pitch, offset, n_expected, image_dim)

    def _fit_axis_edges_iterative(
            self,
            centers: np.ndarray,
            n_expected: int,
            image_dim: int,
            axis_label: str = "axis",
    ) -> np.ndarray:
        """Iterative cell-median refinement seeded from image-pitch indices.

        Each iteration: aggregate centers to one median per occupied cell,
        closed-form fit ``center = pitch * idx + offset``, reassign
        indices using the new fit. Terminates when index assignments
        stop changing (or only a single boundary cell flickers after
        an initial stabilization phase) or after ``MAX_ITER``
        iterations. The image-pitch seed makes this robust to sparse
        plates and within-cell fragmentation, since the initial
        assignment uses only the structural prior, not the diff
        distribution.
        """
        # MAX_ITER=16 covers the empirical worst case observed on dense
        # plates (~1500 centers, convergence at iter 10). Each iteration
        # is cheap (one closed-form fit + index reassignment).
        MAX_ITER = 16
        # After a stabilization phase, accept a single border-cell
        # flicker as converged: the resulting fit drift is sub-pixel
        # (e.g. on km-plate-12hr at iter 7, pitch=161.22 vs true
        # convergence pitch=161.16, a 0.04% difference irrelevant
        # post-rounding to integer edges).
        STABILITY_PHASE = 4
        STABILITY_TOL = 1

        image_pitch = image_dim / n_expected
        indices = self._image_pitch_indices(centers, n_expected, image_dim)
        pitch, offset = image_pitch, image_pitch / 2.0

        converged = False
        for iter_num in range(MAX_ITER):
            medians, unique_idx = self._aggregate_to_cell_medians(
                centers, indices,
            )
            new_pitch, new_offset = self._fit_pitch_and_offset(
                medians, unique_idx,
            )
            if new_pitch <= 0:
                warnings.warn(
                    f"AutoGridFinder ({axis_label}, iterative): degenerate "
                    f"fit pitch <= 0 ({new_pitch}) at iter {iter_num}; "
                    f"falling back to uniform edges.",
                    AutoGridFinderFallbackWarning,
                    stacklevel=4,
                )
                return self._uniform_edges(n_expected, image_dim)

            new_indices = np.rint(
                (centers - new_offset) / new_pitch,
            ).astype(int)
            new_indices = np.clip(new_indices, 0, n_expected - 1)

            n_changed = int(np.sum(new_indices != indices))
            pitch, offset = new_pitch, new_offset
            if n_changed == 0:
                converged = True
                indices = new_indices
                break
            if iter_num >= STABILITY_PHASE and n_changed <= STABILITY_TOL:
                # Single boundary cell flickering after stabilization
                converged = True
                indices = new_indices
                break
            indices = new_indices

        if not converged:
            warnings.warn(
                f"AutoGridFinder ({axis_label}, iterative): index "
                f"assignments did not stabilize after {MAX_ITER} iterations; "
                f"using last fit.",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )

        # Final outlier rejection + refit on inlier cells
        medians, unique_idx = self._aggregate_to_cell_medians(centers, indices)
        threshold = pitch * self.residual_fraction
        inlier_mask = self._identify_inliers(
            medians, unique_idx, pitch, offset, threshold,
        )
        n_inliers = int(inlier_mask.sum())
        if n_inliers >= 2:
            pitch, offset = self._fit_pitch_and_offset(
                medians[inlier_mask], unique_idx[inlier_mask],
            )
        else:
            warnings.warn(
                f"AutoGridFinder ({axis_label}, iterative): only {n_inliers} "
                f"cell-median inliers of {len(medians)} cells within residual "
                f"threshold ({threshold:.2f}); skipping refit and using last "
                f"fit.",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
        if pitch <= 0:
            warnings.warn(
                f"AutoGridFinder ({axis_label}, iterative): refit pitch "
                f"<= 0 ({pitch}); falling back to uniform edges.",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
            return self._uniform_edges(n_expected, image_dim)

        # Symmetry anchoring when detected span < expected.
        #
        # NOTE (latent design choice): unique_idx is post-clip to
        # [0, n_expected - 1], so a single spurious detection past the
        # image edge inflates the span and suppresses anchoring even
        # when real coverage is genuinely sparse. The deleted robust
        # path had the same blind spot. Tightening this would require
        # an occupancy-based gate or a pitch-drift sanity guard
        # (deferred per plan; see "Future enhancement" in
        # /Users/alex/.claude/plans/lively-baking-plum.md).
        span = int(unique_idx.max() - unique_idx.min()) + 1
        if span < n_expected:
            warnings.warn(
                f"AutoGridFinder ({axis_label}, iterative): detected span "
                f"({span}) < n_expected ({n_expected}); recentering grid on "
                f"image (symmetry anchoring overrides fitted offset).",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
            image_center = image_dim / 2.0
            grid_center_idx = (n_expected - 1) / 2.0
            offset = image_center - pitch * grid_center_idx

        return self._compute_grid_edges(pitch, offset, n_expected, image_dim)

    def _fit_axis_edges(
            self,
            info_table: pd.DataFrame,
            axis: int,
            n_expected: int,
            image_dim: int,
    ) -> np.ndarray:
        """Full pipeline: extract centers → fit → edges.

        Falls back to uniform spacing when fewer than 2 objects are found.
        Uses the simple pipeline for low-count *and* low-occupancy axes;
        otherwise routes to the iterative pipeline (image-pitch seed +
        cell-median refinement).
        """
        axis_label = (
            "rows" if axis == 0 else "cols" if axis == 1 else f"axis {axis}"
        )
        try:
            centers = self._extract_axis_centers(info_table, axis)
        except (KeyError, IndexError) as exc:
            warnings.warn(
                f"AutoGridFinder ({axis_label}): could not extract centers "
                f"({type(exc).__name__}); falling back to uniform edges.",
                AutoGridFinderFallbackWarning,
                stacklevel=3,
            )
            return self._uniform_edges(n_expected, image_dim)

        if len(centers) < 2:
            warnings.warn(
                f"AutoGridFinder ({axis_label}): {len(centers)} centers "
                f"available (< 2); falling back to uniform edges.",
                AutoGridFinderFallbackWarning,
                stacklevel=3,
            )
            return self._uniform_edges(n_expected, image_dim)

        # Per-axis sparsity gate: shared with `_run_timed_pipeline` via
        # `_classify_axis_pipeline` so the dashboard label and the
        # actual dispatch can never drift apart.
        if self._classify_axis_pipeline(
            centers, n_expected, image_dim,
        ) == "simple":
            n_occupied = int(np.unique(self._image_pitch_indices(
                centers, n_expected, image_dim,
            )).size)
            warnings.warn(
                f"AutoGridFinder ({axis_label}): {len(centers)} centers "
                f"(< 1.5 * n_expected = {1.5 * n_expected:.1f}) and "
                f"{n_occupied}/{n_expected} occupied cells (< 50%); using "
                f"simple pipeline instead of iterative pipeline.",
                AutoGridFinderFallbackWarning,
                stacklevel=3,
            )
            return self._fit_axis_edges_simple(
                centers, n_expected, image_dim, axis_label=axis_label,
            )

        # High-N guard: skip fitting for pathological object counts
        if len(centers) > self._MAX_OBJECTS_PER_CELL * n_expected:
            warnings.warn(
                f"AutoGridFinder ({axis_label}): detected {len(centers)} "
                f"objects for {n_expected} expected grid positions "
                f"(>{self._MAX_OBJECTS_PER_CELL} per cell). Falling back to "
                f"uniform grid spacing.",
                AutoGridFinderFallbackWarning,
                stacklevel=3,
            )
            return self._uniform_edges(n_expected, image_dim)

        return self._fit_axis_edges_iterative(
            centers, n_expected, image_dim, axis_label=axis_label,
        )

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
        with self._warning_filter():
            if image.num_objects == 0:
                warnings.warn(
                    "AutoGridFinder.get_row_edges: no objects detected; "
                    "falling back to uniform row edges.",
                    AutoGridFinderFallbackWarning,
                    stacklevel=2,
                )
                return self._uniform_edges(self.nrows, image.shape[0])
            info_table = image.objects.info(include_metadata=False)
            return self._fit_axis_edges(
                info_table, axis=0, n_expected=self.nrows,
                image_dim=image.shape[0],
            )

    def get_col_edges(self, image: Image) -> np.ndarray:
        """Return column edge coordinates for *image*.

        Args:
            image: Image with detected objects (``image.objects.info()``).

        Returns:
            Integer array of length ``ncols + 1``.
        """
        with self._warning_filter():
            if image.num_objects == 0:
                warnings.warn(
                    "AutoGridFinder.get_col_edges: no objects detected; "
                    "falling back to uniform column edges.",
                    AutoGridFinderFallbackWarning,
                    stacklevel=2,
                )
                return self._uniform_edges(self.ncols, image.shape[1])
            info_table = image.objects.info(include_metadata=False)
            return self._fit_axis_edges(
                info_table, axis=1, n_expected=self.ncols,
                image_dim=image.shape[1],
            )

    def _operate(self, image: Image) -> pd.DataFrame:
        """Compute grid edges and assign each detected object to a grid cell.

        Args:
            image: Image with detected objects.

        Returns:
            DataFrame with grid assignments (ROW_NUM, COL_NUM, ROW_MAJOR_IDX).
        """
        with self._warning_filter():
            if image.num_objects == 0:
                warnings.warn(
                    "AutoGridFinder._operate: no objects detected; falling "
                    "back to uniform grid edges for both axes.",
                    AutoGridFinderFallbackWarning,
                    stacklevel=2,
                )
                return super()._get_grid_info(
                    image=image,
                    row_edges=self._uniform_edges(self.nrows, image.shape[0]),
                    col_edges=self._uniform_edges(self.ncols, image.shape[1]),
                )
            info_table = image.objects.info(include_metadata=False)
            row_edges = self._fit_axis_edges(
                info_table, axis=0, n_expected=self.nrows,
                image_dim=image.shape[0],
            )
            col_edges = self._fit_axis_edges(
                info_table, axis=1, n_expected=self.ncols,
                image_dim=image.shape[1],
            )
            return super()._get_grid_info(
                image=image, row_edges=row_edges, col_edges=col_edges,
                info_table=info_table,
            )

    # ------------------------------------------------------------------
    # Diagnostic inspect() method
    # ------------------------------------------------------------------

    _OI_NAVY = "#003660"
    _OI_ORANGE = "#E69F00"
    _OI_SKY = "#56B4E9"
    _OI_GREEN = "#009E73"
    _OI_VERMILION = "#D55E00"
    _OI_BLUE = "#0072B2"
    _OI_PURPLE = "#CC79A7"
    _OI_GREY = "#BBBBBB"

    @staticmethod
    def _dashboard_rcparams() -> dict:
        """Return standard dashboard matplotlib rcParams."""
        return {
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#f5f7fa",
            "axes.edgecolor": "#dde3ed",
            "axes.grid": True,
            "grid.color": "#e8ecf2",
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlecolor": "#003660",
            "axes.titleweight": "600",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.labelcolor": "#2e3a4e",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": "#8892a4",
            "ytick.color": "#8892a4",
            "font.family": "sans-serif",
            "font.sans-serif": ["DM Sans", "Helvetica Neue", "Arial"],
            "axes.prop_cycle": __import__("matplotlib").cycler(color=[
                "#003660", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#CC79A7",
            ]),
        }

    @staticmethod
    def _in_jupyter() -> bool:
        """Detect if running inside a Jupyter notebook."""
        try:
            get_ipython()  # type: ignore  # noqa: F821
            return True
        except NameError:
            return False

    def _run_timed_pipeline(
        self, image: Image, show_progress: bool = True,
    ) -> dict:
        """Run the grid pipeline with per-step timing and optional progress bar.

        Args:
            image: Image with detected objects.
            show_progress: Whether to display a progress bar.

        Returns:
            Dict with keys: timings, info_table, row_edges, col_edges, grid_df,
            pipeline_path.
        """
        import time

        steps = [
            "regionprops",
            "fit rows",
            "fit cols",
            "grid assignment",
        ]
        timings: dict[str, float] = {}
        pbar = None
        pipeline_path = "uniform (no objects)"

        if show_progress:
            if self._in_jupyter():
                try:
                    from ipywidgets import IntProgress
                    from IPython.display import display
                    pbar = IntProgress(
                        min=0, max=len(steps), description="Grid inspect:",
                    )
                    display(pbar)
                except ImportError:
                    pass

            if pbar is None:
                try:
                    from tqdm import tqdm
                    pbar = tqdm(total=len(steps), desc="Grid inspect")
                except ImportError:
                    pass

        def _tick(step_name: str, start: float) -> None:
            timings[step_name] = time.perf_counter() - start
            if pbar is not None:
                if hasattr(pbar, "value"):  # ipywidgets
                    pbar.value += 1
                else:  # tqdm
                    pbar.update(1)

        with self._warning_filter():
            # Step 1: regionprops
            t0 = time.perf_counter()
            if image.num_objects == 0:
                info_table = pd.DataFrame()
            else:
                info_table = image.objects.info(include_metadata=False)
            _tick("regionprops", t0)

            # Step 2: fit rows
            t0 = time.perf_counter()
            if image.num_objects == 0:
                row_edges = self._uniform_edges(self.nrows, image.shape[0])
            else:
                n_centers = len(info_table)
                if n_centers < 2:
                    pipeline_path = "uniform (< 2 objects)"
                elif n_centers > self._MAX_OBJECTS_PER_CELL * self.nrows:
                    pipeline_path = "uniform (object count guard)"
                else:
                    # Per-axis labels via the shared classifier; reported as
                    # "rows: X / cols: Y" so divergent decisions are visible.
                    row_centers = AutoGridFinder._extract_axis_centers(
                        info_table, 0,
                    )
                    col_centers = AutoGridFinder._extract_axis_centers(
                        info_table, 1,
                    )
                    row_label = AutoGridFinder._classify_axis_pipeline(
                        row_centers, self.nrows, image.shape[0],
                    )
                    col_label = AutoGridFinder._classify_axis_pipeline(
                        col_centers, self.ncols, image.shape[1],
                    )
                    if row_label == col_label:
                        pipeline_path = row_label
                    else:
                        pipeline_path = (
                            f"rows: {row_label} / cols: {col_label}"
                        )
                row_edges = self._fit_axis_edges(
                    info_table, axis=0, n_expected=self.nrows,
                    image_dim=image.shape[0],
                )
            _tick("fit rows", t0)

            # Step 3: fit cols
            t0 = time.perf_counter()
            if image.num_objects == 0:
                col_edges = self._uniform_edges(self.ncols, image.shape[1])
            else:
                col_edges = self._fit_axis_edges(
                    info_table, axis=1, n_expected=self.ncols,
                    image_dim=image.shape[1],
                )
            _tick("fit cols", t0)

            # Step 4: grid assignment
            t0 = time.perf_counter()
            grid_df = super()._get_grid_info(
                image=image, row_edges=row_edges, col_edges=col_edges,
                info_table=info_table if not info_table.empty else None,
            )
            _tick("grid assignment", t0)

        if pbar is not None and hasattr(pbar, "close"):
            pbar.close()

        return {
            "timings": timings,
            "info_table": info_table,
            "row_edges": row_edges,
            "col_edges": col_edges,
            "grid_df": grid_df,
            "pipeline_path": pipeline_path,
        }

    @classmethod
    def _plot_timing_waterfall(cls, timings: dict[str, float]):
        """Horizontal bar chart of per-step timing."""
        import panel as pn
        import matplotlib.pyplot as plt

        with plt.rc_context(cls._dashboard_rcparams()):
            fig, ax = plt.subplots(figsize=(5, 2.5))
            steps = list(timings.keys())
            times = [timings[s] for s in steps]
            total = sum(times)

            bars = ax.barh(steps, times, color=cls._OI_NAVY, height=0.6)
            for bar, t in zip(bars, times):
                ax.text(
                    bar.get_width() + total * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{t:.3f}s", va="center", fontsize=8,
                    fontfamily="monospace", color="#2e3a4e",
                )
            ax.set_xlabel("Time (s)")
            ax.set_title(f"Step Timing (total: {total:.3f}s)")
            ax.invert_yaxis()
            fig.tight_layout()
            pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
            plt.close(fig)
            return pane

    @classmethod
    def _plot_object_size_dist(
        cls, info_table: pd.DataFrame, nrows: int, ncols: int,
        image_shape: tuple[int, ...],
    ):
        """Histogram of object bounding box areas with expected cell size."""
        import panel as pn
        import matplotlib.pyplot as plt

        with plt.rc_context(cls._dashboard_rcparams()):
            fig, ax = plt.subplots(figsize=(5, 3.5))

            if info_table.empty:
                ax.text(
                    0.5, 0.5, "No objects detected", ha="center", va="center",
                    fontsize=10, color="#8892a4", transform=ax.transAxes,
                )
                ax.set_title("Object Size Distribution")
                fig.tight_layout()
                pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
                plt.close(fig)
                return pane

            heights = (
                info_table[str(BBOX.MAX_RR)].values
                - info_table[str(BBOX.MIN_RR)].values
            )
            widths = (
                info_table[str(BBOX.MAX_CC)].values
                - info_table[str(BBOX.MIN_CC)].values
            )
            areas = heights * widths

            expected_cell_area = (
                (image_shape[0] / nrows) * (image_shape[1] / ncols)
            )
            oversized_mask = areas > expected_cell_area

            ax.hist(
                areas[~oversized_mask], bins=50, color=cls._OI_NAVY,
                alpha=0.8, label="Normal",
            )
            if oversized_mask.any():
                ax.hist(
                    areas[oversized_mask], bins=max(1, oversized_mask.sum() // 2),
                    color=cls._OI_VERMILION, alpha=0.8,
                    label=f"Oversized ({oversized_mask.sum()})",
                )
            ax.axvline(
                expected_cell_area, ls="--", color=cls._OI_GREY, lw=1.5,
                label="Expected cell area",
            )
            ax.set_xlabel("Bbox Area (px\u00b2)")
            ax.set_ylabel("Count")
            ax.set_title("Object Size Distribution")
            ax.legend(fontsize=7, framealpha=0.8)
            fig.tight_layout()
            pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
            plt.close(fig)
            return pane

    @classmethod
    def _plot_center_scatter(
        cls, info_table: pd.DataFrame, row_edges: np.ndarray,
        col_edges: np.ndarray, image_shape: tuple[int, ...],
    ):
        """Scatter plot of weighted centroids with grid edge overlay."""
        import panel as pn
        import matplotlib.pyplot as plt

        with plt.rc_context(cls._dashboard_rcparams()):
            aspect = image_shape[1] / image_shape[0]
            fig_h = 4.0
            fig, ax = plt.subplots(figsize=(fig_h * aspect, fig_h))

            if info_table.empty:
                ax.text(
                    0.5, 0.5, "No objects detected", ha="center", va="center",
                    fontsize=10, color="#8892a4", transform=ax.transAxes,
                )
            else:
                cc = info_table[str(BBOX.DIST_WEIGHTED_CENTER_CC)].values
                rr = info_table[str(BBOX.DIST_WEIGHTED_CENTER_RR)].values
                ax.scatter(
                    cc, rr, s=4, alpha=0.5, color=cls._OI_NAVY,
                    edgecolors="none",
                )

            for edge in row_edges:
                ax.axhline(edge, color=cls._OI_VERMILION, lw=0.8, alpha=0.7)
            for edge in col_edges:
                ax.axvline(edge, color=cls._OI_VERMILION, lw=0.8, alpha=0.7)

            ax.set_xlim(0, image_shape[1])
            ax.set_ylim(image_shape[0], 0)
            ax.set_xlabel("Column (px)")
            ax.set_ylabel("Row (px)")
            ax.set_title("Centroids with Grid Overlay")
            fig.tight_layout()
            pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
            plt.close(fig)
            return pane

    @classmethod
    def _plot_successive_diffs(
        cls, info_table: pd.DataFrame, nrows: int, ncols: int,
        image_shape: tuple[int, ...],
    ):
        """Histogram of successive center diffs with image-pitch markers.

        Two subplots (rows, cols) showing ``np.diff(sorted(centers))`` per
        axis, with vertical reference lines at 1x, 2x, 3x ``image_pitch``.
        A peak at 1x means dense, well-separated detections; secondary
        peaks at 2x or 3x indicate sparse plates with missing cells; a
        peak below 1x indicates within-cell fragmentation.
        """
        import panel as pn
        import matplotlib.pyplot as plt

        with plt.rc_context(cls._dashboard_rcparams()):
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))

            for ax, axis, n_expected, dim, label in [
                (axes[0], 0, nrows, image_shape[0], "Row"),
                (axes[1], 1, ncols, image_shape[1], "Col"),
            ]:
                image_pitch = dim / n_expected

                if info_table.empty:
                    ax.text(
                        0.5, 0.5, "No objects detected",
                        ha="center", va="center", fontsize=10,
                        color="#8892a4", transform=ax.transAxes,
                    )
                    ax.set_title(f"{label} diffs")
                    continue

                centers = cls._extract_axis_centers(info_table, axis)
                if len(centers) < 2:
                    ax.text(
                        0.5, 0.5, "<2 centers",
                        ha="center", va="center", fontsize=10,
                        color="#8892a4", transform=ax.transAxes,
                    )
                    ax.set_title(f"{label} diffs")
                    continue

                diffs = np.diff(centers)
                diffs = diffs[diffs > 0]
                if len(diffs) == 0:
                    ax.text(
                        0.5, 0.5, "No positive diffs",
                        ha="center", va="center", fontsize=10,
                        color="#8892a4", transform=ax.transAxes,
                    )
                    ax.set_title(f"{label} diffs")
                    continue

                bin_width = image_pitch / 8.0
                upper = max(float(diffs.max()), 3.5 * image_pitch)
                n_bins = max(int(np.ceil(upper / bin_width)), 1)
                ax.hist(
                    diffs, bins=n_bins, range=(0, upper + bin_width),
                    color=cls._OI_NAVY, alpha=0.85,
                )
                ax.axvline(
                    image_pitch, color=cls._OI_VERMILION, ls="--", lw=1.2,
                    label=f"1x ({image_pitch:.0f})",
                )
                ax.axvline(
                    2 * image_pitch, color=cls._OI_GREY, ls="--", lw=1.0,
                    label="2x",
                )
                ax.axvline(
                    3 * image_pitch, color=cls._OI_GREY, ls=":", lw=1.0,
                    label="3x",
                )

                ax.set_xlabel("Δ between adjacent centers (px)")
                ax.set_ylabel("count")
                ax.set_title(f"{label} diffs (image_pitch={image_pitch:.0f})")
                ax.legend(fontsize=7, framealpha=0.8)

            fig.tight_layout()
            pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
            plt.close(fig)
            return pane

    @classmethod
    def _plot_axis_occupancy(
        cls, info_table: pd.DataFrame, nrows: int, ncols: int,
        image_shape: tuple[int, ...],
    ):
        """Bar chart of detection counts per cell index per axis.

        Cells assigned via ``_image_pitch_indices`` so the result is
        independent of the fitted pitch. Empty cells (count 0) are drawn
        in vermilion to make missing rows/columns visually obvious.
        """
        import panel as pn
        import matplotlib.pyplot as plt

        with plt.rc_context(cls._dashboard_rcparams()):
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))

            for ax, axis, n_expected, dim, label in [
                (axes[0], 0, nrows, image_shape[0], "Row"),
                (axes[1], 1, ncols, image_shape[1], "Col"),
            ]:
                if info_table.empty:
                    ax.text(
                        0.5, 0.5, "No objects detected",
                        ha="center", va="center", fontsize=10,
                        color="#8892a4", transform=ax.transAxes,
                    )
                    ax.set_title(f"{label} occupancy")
                    continue

                centers = cls._extract_axis_centers(info_table, axis)
                indices = cls._image_pitch_indices(centers, n_expected, dim)
                counts = np.bincount(indices, minlength=n_expected)
                occupied = int((counts > 0).sum())

                colors = [
                    cls._OI_VERMILION if c == 0 else cls._OI_NAVY
                    for c in counts
                ]
                bars = ax.bar(
                    range(n_expected), counts, color=colors, alpha=0.85,
                )
                for bar, c in zip(bars, counts):
                    if c > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), str(int(c)),
                            ha="center", va="bottom",
                            fontsize=7, fontfamily="monospace",
                            color="#2e3a4e",
                        )

                ax.set_xticks(range(n_expected))
                ax.set_xlabel(f"{label} index")
                ax.set_ylabel("# detections")
                ax.set_title(
                    f"{label} occupancy ({occupied}/{n_expected} cells)"
                )

            fig.tight_layout()
            pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
            plt.close(fig)
            return pane

    @classmethod
    def _build_inspect_summary(
        cls, result: dict, nrows: int, ncols: int,
        image_shape: tuple[int, ...],
    ):
        """Markdown summary panel with grid diagnostics."""
        import panel as pn

        info_table = result["info_table"]
        timings = result["timings"]
        grid_df = result["grid_df"]

        n_objects = len(info_table)
        total_time = sum(timings.values())

        # Objects per cell stats
        if not grid_df.empty and str(GRID.ROW_MAJOR_IDX) in grid_df.columns:
            counts = grid_df[str(GRID.ROW_MAJOR_IDX)].value_counts()
            min_per_cell = int(counts.min()) if len(counts) > 0 else 0
            med_per_cell = float(counts.median()) if len(counts) > 0 else 0
            max_per_cell = int(counts.max()) if len(counts) > 0 else 0
            occupied = len(counts)
        else:
            min_per_cell = med_per_cell = max_per_cell = occupied = 0

        # Oversized objects
        if not info_table.empty:
            heights = (
                info_table[str(BBOX.MAX_RR)].values
                - info_table[str(BBOX.MIN_RR)].values
            )
            widths = (
                info_table[str(BBOX.MAX_CC)].values
                - info_table[str(BBOX.MIN_CC)].values
            )
            expected_cell_area = (
                (image_shape[0] / nrows) * (image_shape[1] / ncols)
            )
            n_oversized = int((heights * widths > expected_cell_area).sum())
        else:
            n_oversized = 0

        # Pitch from edges
        row_edges = result["row_edges"]
        col_edges = result["col_edges"]
        row_pitch = float(np.median(np.diff(row_edges)))
        col_pitch = float(np.median(np.diff(col_edges)))

        # Sparse-plate diagnostics: image-pitch indices give occupancy
        # and span coverage independent of the fitted pitch
        if not info_table.empty:
            row_centers = AutoGridFinder._extract_axis_centers(info_table, 0)
            col_centers = AutoGridFinder._extract_axis_centers(info_table, 1)
            row_idx = AutoGridFinder._image_pitch_indices(
                row_centers, nrows, image_shape[0],
            )
            col_idx = AutoGridFinder._image_pitch_indices(
                col_centers, ncols, image_shape[1],
            )
            row_counts = np.bincount(row_idx, minlength=nrows)
            col_counts = np.bincount(col_idx, minlength=ncols)
            occupied_rows = int((row_counts > 0).sum())
            occupied_cols = int((col_counts > 0).sum())
            row_span = (
                int(row_idx.max() - row_idx.min() + 1) if len(row_idx) else 0
            )
            col_span = (
                int(col_idx.max() - col_idx.min() + 1) if len(col_idx) else 0
            )
        else:
            occupied_rows = occupied_cols = row_span = col_span = 0

        md = (
            f"### Summary\n\n"
            f"| Metric | Value |\n"
            f"|---|---|\n"
            f"| Objects | {n_objects} |\n"
            f"| Grid | {nrows} x {ncols} ({nrows * ncols} cells) |\n"
            f"| Occupied cells | {occupied} |\n"
            f"| Obj/cell (min / med / max) | {min_per_cell} / "
            f"{med_per_cell:.1f} / {max_per_cell} |\n"
            f"| Oversized objects | {n_oversized} |\n"
            f"| Row pitch | {row_pitch:.1f} px |\n"
            f"| Col pitch | {col_pitch:.1f} px |\n"
            f"| Pipeline path | {result['pipeline_path']} |\n"
            f"| Row occupancy | {occupied_rows}/{nrows} "
            f"({occupied_rows / nrows:.0%}) |\n"
            f"| Col occupancy | {occupied_cols}/{ncols} "
            f"({occupied_cols / ncols:.0%}) |\n"
            f"| Row span coverage | {row_span}/{nrows} "
            f"({row_span / nrows:.0%}) |\n"
            f"| Col span coverage | {col_span}/{ncols} "
            f"({col_span / ncols:.0%}) |\n"
            f"| Total time | {total_time:.3f} s |\n"
        )
        return pn.pane.Markdown(
            md, styles={"font-family": "'DM Sans', sans-serif"},
        )

    def inspect(self, image: Image, show_progress: bool = True):
        """Interactive diagnostic dashboard for grid fitting.

        Profiles the grid-fitting pipeline and displays timing breakdown,
        object size distribution, centroid scatter with grid overlay, and
        summary statistics. Useful for identifying bottlenecks when
        ``grid.info()`` is slow (e.g., with filamentous fungi images).

        Uses an ipywidgets progress bar in Jupyter, tqdm otherwise.

        Args:
            image: Image with detected objects (must have objmap).
            show_progress: Whether to display a progress bar during
                profiling. Defaults to True.

        Returns:
            Panel Column layout with 4 diagnostic panels.

        Examples:
            >>> from phenotypic.data import load_synth_yeast_plate
            >>> from phenotypic.detect import OtsuDetector
            >>> from phenotypic.grid import AutoGridFinder
            >>> image = load_synth_yeast_plate()
            >>> image = OtsuDetector().apply(image)
            >>> finder = AutoGridFinder(nrows=8, ncols=12)
            >>> dashboard = finder.inspect(image)
        """
        import panel as pn

        result = self._run_timed_pipeline(image, show_progress=show_progress)

        header = pn.pane.Markdown(
            f"## Grid Fitting Diagnostics -- {image.num_objects} objects, "
            f"{self.nrows}x{self.ncols} grid",
            styles={
                "font-family": "'DM Sans', sans-serif",
                "color": self._OI_NAVY,
            },
        )

        p1 = self._plot_timing_waterfall(result["timings"])
        p2 = self._plot_object_size_dist(
            result["info_table"], self.nrows, self.ncols, image.shape,
        )
        p3 = self._plot_center_scatter(
            result["info_table"], result["row_edges"],
            result["col_edges"], image.shape,
        )
        p4 = self._build_inspect_summary(
            result, self.nrows, self.ncols, image.shape,
        )
        p5 = self._plot_successive_diffs(
            result["info_table"], self.nrows, self.ncols, image.shape,
        )
        p6 = self._plot_axis_occupancy(
            result["info_table"], self.nrows, self.ncols, image.shape,
        )

        return pn.Column(
            header,
            pn.Row(p1, p4),
            pn.Row(p3, p2),
            pn.Row(p5, p6),
        )


AutoGridFinder.measure.__doc__ = AutoGridFinder._operate.__doc__
AutoGridFinder.__doc__ = GRID.append_rst_to_doc(AutoGridFinder.__doc__)
