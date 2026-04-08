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

    _MAX_OBJECTS_PER_CELL: int = 250

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
    def _robust_pitch_estimate(centers: np.ndarray, n_expected: int) -> float:
        """Estimate grid pitch robustly using the mode of successive differences.

        Falls back to span-based estimation when the mode is outside a
        plausible range (0.5x–2.0x the span estimate).

        Args:
            centers: Sorted 1-D array of object center coordinates.
            n_expected: Number of expected grid positions along this axis.

        Returns:
            Estimated pitch in pixels.
        """
        span_pitch = AutoGridFinder._estimate_pitch(centers, n_expected)
        if span_pitch <= 0:
            return span_pitch

        diffs = np.diff(centers)
        diffs = diffs[diffs > 0]
        if len(diffs) < 2:
            return span_pitch

        bin_width = span_pitch / 4.0
        n_bins = max(int(np.ceil(diffs.max() / bin_width)), 1)
        counts, bin_edges = np.histogram(diffs, bins=n_bins, range=(0, diffs.max() + bin_width))
        mode_bin = np.argmax(counts)
        mode_pitch = float((bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2.0)

        if mode_pitch < 0.5 * span_pitch or mode_pitch > 2.0 * span_pitch:
            return span_pitch
        return mode_pitch

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

    def _fit_axis_edges_simple(
            self,
            centers: np.ndarray,
            n_expected: int,
            image_dim: int,
    ) -> np.ndarray:
        """Simple pipeline used when the object count is low.

        Uses span-based pitch estimation and fits all centers directly.
        """
        pitch = self._estimate_pitch(centers, n_expected)
        if pitch <= 0:
            return self._uniform_edges(n_expected, image_dim)

        indices = self._assign_grid_indices(centers, pitch)
        pitch, offset = self._fit_pitch_and_offset(centers, indices)
        if pitch <= 0:
            return self._uniform_edges(n_expected, image_dim)

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

        span = int(indices.max() - indices.min()) + 1
        if span < n_expected:
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
        """Full pipeline: extract centers → fit → reject → refit → edges.

        Falls back to uniform spacing when fewer than 2 objects are found.
        Uses the simple pipeline for low object counts and the robust
        pipeline (median aggregation + robust pitch) for high counts.
        """
        try:
            centers = self._extract_axis_centers(info_table, axis)
        except (KeyError, IndexError):
            centers = np.array([])

        if len(centers) < 2:
            return self._uniform_edges(n_expected, image_dim)

        # Low-N guard: use simple pipeline when few objects
        if len(centers) < 2 * n_expected:
            return self._fit_axis_edges_simple(
                centers, n_expected, image_dim,
            )

        # High-N guard: skip fitting for pathological object counts
        if len(centers) > self._MAX_OBJECTS_PER_CELL * n_expected:
            warnings.warn(
                f"Detected {len(centers)} objects for {n_expected} expected "
                f"grid positions (>{self._MAX_OBJECTS_PER_CELL} per cell). "
                f"Falling back to uniform grid spacing.",
                stacklevel=2,
            )
            return self._uniform_edges(n_expected, image_dim)

        # --- Robust pipeline for high object counts ---

        # Step 1: robust pitch estimate (mode of successive differences)
        pitch = self._robust_pitch_estimate(centers, n_expected)
        if pitch <= 0:
            return self._uniform_edges(n_expected, image_dim)

        # Step 2: assign grid indices with median anchor
        anchor = float(np.median(centers))
        indices = self._assign_grid_indices(centers, pitch, anchor=anchor)

        # Step 3: aggregate to one median per grid cell
        medians, unique_idx = self._aggregate_to_cell_medians(
            centers, indices,
        )

        # Step 4: fit on aggregated medians (equal weight per cell)
        pitch, offset = self._fit_pitch_and_offset(medians, unique_idx)
        if pitch <= 0:
            return self._uniform_edges(n_expected, image_dim)

        # Step 5: outlier rejection + refit on cells
        threshold = pitch * self.residual_fraction
        inlier_mask = self._identify_inliers(
            medians, unique_idx, pitch, offset, threshold,
        )
        if inlier_mask.sum() >= 2:
            pitch, offset = self._fit_pitch_and_offset(
                medians[inlier_mask], unique_idx[inlier_mask],
            )
        if pitch <= 0:
            return self._uniform_edges(n_expected, image_dim)

        # Step 6: symmetry anchoring when detected span < expected
        span = int(unique_idx.max() - unique_idx.min()) + 1
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
        if image.num_objects == 0:
            return self._uniform_edges(self.nrows, image.shape[0])
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
        if image.num_objects == 0:
            return self._uniform_edges(self.ncols, image.shape[1])
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
        if image.num_objects == 0:
            return super()._get_grid_info(
                image=image,
                row_edges=self._uniform_edges(self.nrows, image.shape[0]),
                col_edges=self._uniform_edges(self.ncols, image.shape[1]),
            )
        info_table = image.objects.info(include_metadata=False)
        row_edges = self._fit_axis_edges(
            info_table, axis=0, n_expected=self.nrows, image_dim=image.shape[0],
        )
        col_edges = self._fit_axis_edges(
            info_table, axis=1, n_expected=self.ncols, image_dim=image.shape[1],
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
            elif n_centers < 2 * self.nrows:
                pipeline_path = "simple"
            elif n_centers > self._MAX_OBJECTS_PER_CELL * self.nrows:
                pipeline_path = "uniform (object count guard)"
            else:
                pipeline_path = "robust"
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
                cc = info_table[str(BBOX.WEIGHTED_CENTER_CC)].values
                rr = info_table[str(BBOX.WEIGHTED_CENTER_RR)].values
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

        return pn.Column(
            header,
            pn.Row(p1, p4),
            pn.Row(p3, p2),
        )


AutoGridFinder.measure.__doc__ = AutoGridFinder._operate.__doc__
AutoGridFinder.__doc__ = GRID.append_rst_to_doc(AutoGridFinder.__doc__)
