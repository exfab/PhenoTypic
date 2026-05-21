from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar

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
    switch to simple pipeline, span-based pitch), overrides geometry
    (pitch shrink in :meth:`AutoGridFinder._compute_grid_edges`), or
    surfaces a span-coverage decision
    (``[span-coverage-low]``, ``[span-coverage-high]``,
    ``[span-coverage-symmetric]`` — note that span-coverage warnings now
    document the chosen anchor; they no longer indicate that the fitted
    offset was overridden). Use this category to filter only
    AutoGridFinder diagnostics in batch runs::

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
            diagnostics for fallbacks, geometry overrides, and
            span-coverage anchoring decisions. Defaults to ``False``
            (silent); set to ``True`` to surface them.
        tol: Deprecated. Accepted for backward compatibility but ignored.
        max_iter: Deprecated. Accepted for backward compatibility but ignored.

    Notes:
        Span-coverage handling: when the detected span of occupied grid
        indices is less than ``n_expected`` by more than
        :attr:`_SPAN_TOLERANCE` (default ``1`` — a single entirely-empty
        edge row/column is silently tolerated), the fitter compares the
        leftmost and rightmost centroids to the fit's predicted cell-0
        and cell-(n-1) centers. When one end aligns within ``pitch / 2``
        of its predicted boundary cell, that side is treated as the
        anchored edge and the fitted offset is preserved (no override).
        The three diagnostic tags ``[span-coverage-low]`` /
        ``[span-coverage-high]`` / ``[span-coverage-symmetric]`` document
        which decision was made; ``warn=True`` surfaces them. To loosen
        or tighten the tolerance, subclass and override
        :attr:`_SPAN_TOLERANCE`.

        Diagnostic warnings of category
        :class:`AutoGridFinderFallbackWarning` are emitted at every
        fallback (uniform spacing, simple-pipeline switch, span-based
        pitch), geometry override (pitch shrink), and span-coverage
        anchor decision. Suppress them in batch runs with::

            import warnings
            from phenotypic.grid._auto_grid_finder import (
                AutoGridFinderFallbackWarning,
            )
            warnings.filterwarnings(
                "ignore", category=AutoGridFinderFallbackWarning,
            )
    """

    # ``ClassVar`` so pydantic treats these as class-level constants rather
    # than model fields (the project convention for operation constants --
    # see ``_GATSupportMixin._GAT_NOISE_PARAMS``). ``_SPAN_TOLERANCE`` stays
    # overridable by subclasses, as the class docstring documents.
    _MAX_OBJECTS_PER_CELL: ClassVar[int] = 250
    # Entire missing edge rows/cols tolerated before span-coverage anchoring
    # fires. With the default of 1, a 96-well plate with one fully-empty
    # edge column keeps its fitted offset; only ≥ 2 missing edge cells
    # trigger the anchoring check.
    _SPAN_TOLERANCE: ClassVar[int] = 1

    # ``nrows`` / ``ncols`` are required fields on the ``GridFinder`` ABC;
    # ``AutoGridFinder`` historically defaulted them to 8 / 12 (96-well
    # plate). Pydantic allows a subclass to add a default to an inherited
    # required field, so redeclare them here with the legacy defaults.
    nrows: int = 8
    ncols: int = 12
    residual_fraction: float = 0.25
    warn: bool = False
    # ``tol`` / ``max_iter`` are deprecated, accepted-but-ignored
    # constructor parameters retained for backward compatibility. They are
    # declared as optional fields (rather than dropped) so that, under
    # ``extra="forbid"``, legacy callers passing them still construct; a
    # non-``None`` value triggers a ``DeprecationWarning`` in
    # ``model_post_init``. The fitter never reads them.
    tol: float | None = None
    max_iter: int | None = None

    def model_post_init(self, __context: Any) -> None:
        """Warn on deprecated ``tol`` / ``max_iter`` parameters.

        Reproduces the legacy ``__init__`` deprecation warnings, then
        delegates to :meth:`BaseOperation.model_post_init` for logger and
        memory-tracking setup.

        Args:
            __context: Pydantic post-init context (unused).
        """
        if self.tol is not None:
            warnings.warn(
                "The 'tol' parameter is deprecated and has no effect. "
                "AutoGridFinder now uses a deterministic robust-fit algorithm.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.max_iter is not None:
            warnings.warn(
                "The 'max_iter' parameter is deprecated and has no effect. "
                "AutoGridFinder now uses a deterministic robust-fit algorithm.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().model_post_init(__context)

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
            axis_label: str = "axis",
    ) -> tuple[float, float]:
        """Closed-form linear fit ``center = pitch * idx + offset``.

        Args:
            centers: 1-D array of center coordinates.
            indices: 1-D array of integer grid indices, same length as
                ``centers``.
            axis_label: Diagnostic label (``"rows"`` or ``"cols"``) used
                only to disambiguate the degenerate-fit warning.

        Returns:
            Tuple of (pitch, offset).
        """
        idx_mean = indices.mean()
        ctr_mean = centers.mean()
        idx_dev = indices - idx_mean
        denom = float(idx_dev @ idx_dev)
        if denom == 0.0:
            warnings.warn(
                f"AutoGridFinder [degenerate-indices] ({axis_label}): all "
                f"{len(indices)} centers map to the same grid index, so the "
                f"closed-form pitch/offset fit is ill-defined; pitch set to "
                f"0.0, offset to centers mean ({ctr_mean:.2f}).",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
            return 0.0, ctr_mean
        pitch = float(idx_dev @ (centers - ctr_mean)) / denom
        offset = ctr_mean - pitch * idx_mean
        return pitch, offset

    @staticmethod
    def _choose_seed_anchor(
            centers: np.ndarray,
            pitch_seed: float,
            n_expected: int,
            image_dim: int,
    ) -> float:
        """Pick the iterative-seed offset (pixel position of cell 0)
        without using image **margins**.

        The classic ``image_pitch / 2`` anchor assumes the plate fills
        the image edge-to-edge. When the plate sits inside the image
        with empty margins (common with letterboxed cameras or wide
        sensors), that anchor pulls the initial index assignment
        inward, causing the OLS refinement to converge on a compressed
        grid that mis-assigns every detection by one index. This helper
        replaces that anchor with a data-driven choice that relies only
        on the image **midpoint** (a structural property) and the
        centroid distribution.

        Algorithm:

        1. ``expected_n = round((c_max - c_min) / pitch_seed) + 1`` —
           how many cells the detected span implies at the seed pitch.
        2. ``expected_n >= n_expected``: all (or more than all) cells
           appear detected → anchor at ``c_min`` (cell 0 = leftmost
           centroid).
        3. ``expected_n < n_expected``: some edge cells are likely
           undetected. Decide which side using centroid-midpoint vs
           image-midpoint:

           - ``centroid_mid <= image_mid - pitch_seed/2`` (centroids
             lean low): high-side cells are missing; anchor at
             ``c_min``.
           - ``centroid_mid >= image_mid + pitch_seed/2`` (centroids
             lean high): low-side cells are missing; anchor cell
             ``n_expected - 1`` at ``c_max``, i.e.
             ``offset = c_max - (n_expected - 1) * pitch_seed``.
           - Otherwise (roughly centered): fall back to ``c_min``;
             the post-fit ``_resolve_span_anchor`` check will surface
             any genuine ambiguity.

        The image **center** (``image_dim / 2``) is used only as a
        midpoint reference; the image **margins** (the distance from
        ``c_max`` to the right edge, etc.) are never consulted.

        Args:
            centers: Sorted 1-D array of detected centroid coordinates.
            pitch_seed: Seed pitch (typically from
                :meth:`_estimate_pitch`).
            n_expected: Number of expected grid cells.
            image_dim: Image extent along this axis (used only via its
                midpoint).

        Returns:
            The pixel coordinate where cell 0 should be anchored.
        """
        if pitch_seed <= 0 or len(centers) == 0:
            return float(centers[0]) if len(centers) else 0.0

        c_min = float(centers[0])
        c_max = float(centers[-1])
        expected_n = int(round((c_max - c_min) / pitch_seed)) + 1

        if expected_n >= n_expected:
            return c_min

        centroid_mid = (c_min + c_max) / 2.0
        image_mid = image_dim / 2.0
        half_pitch = pitch_seed / 2.0
        if centroid_mid >= image_mid + half_pitch:
            # Centroids skewed high → low-side cells missing.
            return c_max - (n_expected - 1) * pitch_seed
        # Centroids lean low or are roughly centered: anchor at c_min.
        return c_min

    @staticmethod
    def _resolve_span_anchor(
            centers: np.ndarray,
            pitch: float,
            offset: float,
            n_expected: int,
            span: int,
            axis_label: str,
            span_word: str,
    ) -> float:
        """Decide how to anchor the grid when ``span < n_expected``.

        Compares the fit's predicted cell-0 and cell-(n-1) centers against the
        actual centroid extremes ``c_min`` / ``c_max``. When one end of the
        centroid range aligns (within ``pitch/2``) with its predicted boundary
        cell and the other does not, the fit is already correctly anchored to
        the populated edge — the fitted offset is preserved (Option (i)). The
        empty cells simply sit at indices ``[0, idx_min - 1]`` or
        ``[idx_max + 1, n_expected - 1]``.

        When neither end aligns, the fit is uncertain; a
        ``[span-coverage-symmetric]`` warning is emitted but the offset is
        still preserved (no symmetric override).

        Args:
            centers: Sorted 1-D array of detected centroid coordinates.
            pitch: Fitted grid pitch (must be positive).
            offset: Fitted grid offset (pixel position of cell 0).
            n_expected: Number of expected grid cells along this axis.
            span: Span of occupied grid indices (``max - min + 1``).
            axis_label: ``"rows"`` / ``"cols"`` for warning text.
            span_word: ``"row"`` / ``"column"`` for warning text.

        Returns:
            The offset to use for edge computation. Under Option (i) this
            equals the input ``offset`` in every branch; the branching exists
            so the per-mode diagnostic warning is correct and so future
            tweaks need only change the per-branch return value.
        """
        n_missing = n_expected - span
        if n_missing <= AutoGridFinder._SPAN_TOLERANCE:
            return offset

        half_pitch = pitch / 2.0
        c_min = float(centers[0])     # centers are pre-sorted
        c_max = float(centers[-1])
        predicted_lo = offset                              # cell 0 center
        predicted_hi = offset + pitch * (n_expected - 1)   # cell n-1 center
        lower_anchored = abs(c_min - predicted_lo) <= half_pitch
        upper_anchored = abs(c_max - predicted_hi) <= half_pitch

        if lower_anchored and upper_anchored:
            # span < n_expected but both ends aligned → purely interior gap.
            return offset
        if lower_anchored and not upper_anchored:
            warnings.warn(
                f"AutoGridFinder [span-coverage-low] ({axis_label}): "
                f"detected {span_word} span = {span} cells (expected "
                f"{n_expected}); leftmost centroid ({c_min:.1f}) aligns with "
                f"predicted cell-0 ({predicted_lo:.1f}); keeping fitted "
                f"offset (low-anchored).",
                AutoGridFinderFallbackWarning, stacklevel=4,
            )
            return offset
        if upper_anchored and not lower_anchored:
            warnings.warn(
                f"AutoGridFinder [span-coverage-high] ({axis_label}): "
                f"detected {span_word} span = {span} cells (expected "
                f"{n_expected}); rightmost centroid ({c_max:.1f}) aligns with "
                f"predicted cell-{n_expected - 1} ({predicted_hi:.1f}); "
                f"keeping fitted offset (high-anchored).",
                AutoGridFinderFallbackWarning, stacklevel=4,
            )
            return offset

        # Neither end anchored — fit may be poorly placed but symmetric
        # override has been shown to bias densely-detected plates with one
        # missing edge column. Keep the fit and surface a diagnostic.
        warnings.warn(
            f"AutoGridFinder [span-coverage-symmetric] ({axis_label}): "
            f"detected {span_word} span = {span} cells (expected "
            f"{n_expected}); neither end of the centroid range "
            f"({c_min:.1f}-{c_max:.1f}) aligns with the fitted boundary "
            f"cells ({predicted_lo:.1f}, {predicted_hi:.1f}); keeping fit "
            f"(no override).",
            AutoGridFinderFallbackWarning, stacklevel=4,
        )
        return offset

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
            axis_label: str = "axis",
    ) -> np.ndarray:
        """Compute ``n_bins + 1`` edge coordinates clipped to ``[0, image_dim]``.

        Edges are placed at ``offset + pitch * i - pitch / 2`` for
        ``i = 0 .. n_bins``.

        The offset is clamped so that the first edge is >= 0 and the
        last edge is <= image_dim, preventing duplicate edges after
        clipping.  When the fitted pitch is too large for the image,
        the pitch is shrunk to ``image_dim / n_bins`` so the grid
        fills the available space.

        Args:
            pitch: Fitted grid pitch in pixels.
            offset: Fitted grid offset in pixels.
            n_bins: Number of grid cells along this axis.
            image_dim: Image extent along this axis in pixels.
            axis_label: Diagnostic label (``"rows"`` or ``"cols"``) used
                only to disambiguate the pitch-overflow warning.
        """
        half = pitch / 2.0
        min_offset = half                              # first edge >= 0
        max_offset = image_dim - pitch * n_bins + half  # last edge <= image_dim

        if min_offset > max_offset:
            warnings.warn(
                f"AutoGridFinder [pitch-overflow] ({axis_label}): fitted pitch "
                f"({pitch:.2f} px) too large for image_dim={image_dim} with "
                f"n_bins={n_bins}; shrinking pitch to "
                f"{image_dim / n_bins:.2f} px and recentering.",
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
    def _indices_from_edges(
            centers: np.ndarray, edges: np.ndarray,
    ) -> np.ndarray:
        """Recover per-cell indices from fitted grid edges.

        The dashboard counterpart to :meth:`_image_pitch_indices`:
        instead of binning centroids using the structural prior
        ``image_dim / n_expected``, this helper bins them against the
        **fitted** grid edges (which reflect the actual pitch and
        offset converged by the iterative pipeline). The two assign
        identical indices when the plate fills the image edge-to-edge;
        they disagree when the plate sits inside a wider image with
        empty margins — and that disagreement is itself a useful
        diagnostic.

        Uses the median pitch and cell-0 center derived from
        ``edges``::

            pitch = median(diff(edges))
            cell_0_center = edges[0] + pitch / 2
            idx = round((center - cell_0_center) / pitch)

        Args:
            centers: 1-D array of centroid coordinates along one axis.
            edges: ``n_bins + 1`` edge coordinates produced by
                :meth:`_compute_grid_edges`.

        Returns:
            Integer array of same length as ``centers``, clipped to
            ``[0, n_bins - 1]``. Returns an empty array if ``centers``
            is empty.
        """
        if len(centers) == 0:
            return np.empty(0, dtype=int)
        if len(edges) < 2:
            # No diffs available → can't recover pitch. Defensive
            # guard for the helper's documented empty-array contract;
            # not reachable from _compute_grid_edges (which always emits
            # n_bins + 1 ≥ 2 edges), but called directly by tests and
            # potentially by future code paths.
            return np.zeros(len(centers), dtype=int)
        n_bins = len(edges) - 1
        pitch = float(np.median(np.diff(edges)))
        if pitch <= 0:
            return np.zeros(len(centers), dtype=int)
        cell_0_center = float(edges[0]) + pitch / 2.0
        idx = np.rint((centers - cell_0_center) / pitch).astype(int)
        return np.clip(idx, 0, max(n_bins - 1, 0))

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
    def _compute_axis_dashboard_stats(
            info_table: pd.DataFrame,
            axis: int,
            n_expected: int,
            image_dim: int,
            edges: np.ndarray,
    ) -> dict:
        """Compute one axis's diagnostic stats in a single shot.

        Single source of truth for every dashboard panel. All
        per-axis numbers the summary table, occupancy plot, and
        successive-diffs plot need are computed here exactly once, so
        no panel can re-derive them inconsistently. Adding a new
        diagnostic panel? Read from the returned dict; don't recompute
        from ``info_table`` or ``edges``.

        Computes BOTH:

        - **Fitted** quantities (from the edges actually produced by
          :meth:`_compute_grid_edges`) — the algorithm's decision.
        - **Image-pitch** quantities (from the structural prior
          ``image_dim / n_expected``) — useful as an independent
          cross-check, but biased when the plate doesn't fill the
          image edge-to-edge.

        When the two disagree, ``"agree"`` is ``False`` — surface this
        in panel titles / summary rows so the user sees the
        structural-prior mismatch directly.

        Args:
            info_table: ``image.objects.info(include_metadata=False)``
                output (may be empty).
            axis: 0 for rows, 1 for cols.
            n_expected: Number of expected grid cells along this axis.
            image_dim: Image extent along this axis.
            edges: ``n_expected + 1`` integer edge coordinates produced
                by the fitter.

        Returns:
            Dict with keys::

                axis              0 or 1
                label             "Row" or "Col"
                n_expected        echo
                image_dim         echo
                edges             echo (same array)
                centers           sorted axis centroids (or empty)
                image_pitch       image_dim / n_expected
                fit_pitch         median(diff(edges))
                fit_indices       per-centroid cell, from edges
                ip_indices        per-centroid cell, from image_pitch
                fit_counts        bincount(fit_indices, minlength=n)
                ip_counts         bincount(ip_indices,  minlength=n)
                fit_occupied      sum(fit_counts > 0)
                ip_occupied       sum(ip_counts  > 0)
                fit_span          max - min + 1 of fit_indices
                ip_span           max - min + 1 of ip_indices
                agree             counts/span/occupied all match
        """
        label = "Row" if axis == 0 else "Col" if axis == 1 else f"Axis {axis}"
        empty_centers = np.empty(0, dtype=float)
        empty_idx = np.empty(0, dtype=int)
        zero_counts = np.zeros(n_expected, dtype=int)
        fit_pitch = (
            float(np.median(np.diff(edges)))
            if edges is not None and len(edges) >= 2 else 0.0
        )
        image_pitch = (
            image_dim / n_expected if n_expected > 0 else 0.0
        )

        if info_table.empty:
            return {
                "axis": axis,
                "label": label,
                "n_expected": n_expected,
                "image_dim": image_dim,
                "edges": edges,
                "centers": empty_centers,
                "image_pitch": image_pitch,
                "fit_pitch": fit_pitch,
                "fit_indices": empty_idx,
                "ip_indices": empty_idx,
                "fit_counts": zero_counts,
                "ip_counts": zero_counts.copy(),
                "fit_occupied": 0,
                "ip_occupied": 0,
                "fit_span": 0,
                "ip_span": 0,
                "agree": True,
            }

        centers = AutoGridFinder._extract_axis_centers(info_table, axis)
        fit_indices = AutoGridFinder._indices_from_edges(centers, edges)
        ip_indices = AutoGridFinder._image_pitch_indices(
            centers, n_expected, image_dim,
        )
        fit_counts = np.bincount(fit_indices, minlength=n_expected)
        ip_counts = np.bincount(ip_indices, minlength=n_expected)
        fit_occupied = int((fit_counts > 0).sum())
        ip_occupied = int((ip_counts > 0).sum())
        fit_span = (
            int(fit_indices.max() - fit_indices.min() + 1)
            if len(fit_indices) else 0
        )
        ip_span = (
            int(ip_indices.max() - ip_indices.min() + 1)
            if len(ip_indices) else 0
        )
        agree = bool(
            np.array_equal(fit_counts, ip_counts)
            and fit_span == ip_span
            and fit_occupied == ip_occupied
        )
        return {
            "axis": axis,
            "label": label,
            "n_expected": n_expected,
            "image_dim": image_dim,
            "edges": edges,
            "centers": centers,
            "image_pitch": image_pitch,
            "fit_pitch": fit_pitch,
            "fit_indices": fit_indices,
            "ip_indices": ip_indices,
            "fit_counts": fit_counts,
            "ip_counts": ip_counts,
            "fit_occupied": fit_occupied,
            "ip_occupied": ip_occupied,
            "fit_span": fit_span,
            "ip_span": ip_span,
            "agree": agree,
        }

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
        span_word = (
            "row" if axis_label == "rows"
            else "column" if axis_label == "cols"
            else axis_label
        )
        pitch = self._estimate_pitch(centers, n_expected)
        if pitch <= 0:
            warnings.warn(
                f"AutoGridFinder [span-pitch-degenerate] ({axis_label}, "
                f"simple): span-based {span_word} pitch estimate "
                f"({pitch:.2f}) is non-positive (centers max - min collapsed); "
                f"falling back to uniform {span_word} edges.",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
            return self._uniform_edges(n_expected, image_dim)

        indices = self._assign_grid_indices(centers, pitch)
        pitch, offset = self._fit_pitch_and_offset(
            centers, indices, axis_label=axis_label,
        )
        if pitch <= 0:
            warnings.warn(
                f"AutoGridFinder [initial-fit-degenerate] ({axis_label}, "
                f"simple): initial OLS {span_word} pitch ({pitch:.2f}) is "
                f"non-positive after fitting all centers; falling back to "
                f"uniform {span_word} edges.",
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
                axis_label=axis_label,
            )
        else:
            warnings.warn(
                f"AutoGridFinder [insufficient-inliers] ({axis_label}, "
                f"simple): only {n_inliers} of {len(centers)} {span_word} "
                f"centers fall within the residual threshold "
                f"({threshold:.2f} px = pitch * {self.residual_fraction}); "
                f"skipping refit and using the initial fit.",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
        if pitch <= 0:
            warnings.warn(
                f"AutoGridFinder [refit-degenerate] ({axis_label}, simple): "
                f"refit {span_word} pitch ({pitch:.2f}) is non-positive after "
                f"inlier filtering; falling back to uniform {span_word} edges.",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
            return self._uniform_edges(n_expected, image_dim)

        # Simple pipeline keeps raw (unclipped) indices, so indices.min/max
        # already reflect the data faithfully.
        span = int(indices.max() - indices.min()) + 1
        offset = self._resolve_span_anchor(
            centers, pitch, offset, n_expected, span,
            axis_label, span_word,
        )
        return self._compute_grid_edges(
            pitch, offset, n_expected, image_dim, axis_label=axis_label,
        )

    def _fit_axis_edges_iterative(
            self,
            centers: np.ndarray,
            n_expected: int,
            image_dim: int,
            axis_label: str = "axis",
    ) -> np.ndarray:
        """Iterative cell-median refinement with a data-driven seed.

        Each iteration: aggregate centers to one median per occupied cell,
        closed-form fit ``center = pitch * idx + offset``, reassign
        indices using the new fit. Terminates when index assignments
        stop changing (or only a single boundary cell flickers after
        an initial stabilization phase) or after ``MAX_ITER``
        iterations.

        Seeding (image-margin-independent):

        - **Pitch**: :meth:`_estimate_pitch` (span-based
          ``(c_max - c_min) / (n_expected - 1)``). Robust to any
          number of detections per axis position (a 6×10 plate sorted
          on the column axis has 6 detections at each column position;
          a 1-row plate with fragmentation has K per cell; both look
          identical to span-based pitch since ``c_min`` and ``c_max``
          are the outermost colonies' DT-maxima). Biased only when
          edge cells are physically missing; the post-fit
          :meth:`_resolve_span_anchor` check surfaces those cases via
          ``[span-coverage-low]`` / ``[span-coverage-high]``.
        - **Offset**: :meth:`_choose_seed_anchor` (``c_min`` when the
          detected span looks complete; otherwise picks the anchored
          side using centroid-midpoint vs image-midpoint).

        Neither the pitch nor the offset estimate depends on image
        margins (the distance from the plate to the image edge), so
        plates sitting inside a wider image with letterboxing are
        handled correctly.
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

        span_word = (
            "row" if axis_label == "rows"
            else "column" if axis_label == "cols"
            else axis_label
        )
        # Data-driven seed: span-based pitch (robust to any number of
        # detections per axis position — 2D grids with multiple rows
        # per column, fragmentation, etc.); offset from
        # _choose_seed_anchor (avoids the image-margin bias of
        # image_pitch / 2). image_pitch is no longer used at seed time;
        # the post-fit _resolve_span_anchor surfaces edge-missing cases
        # where span-based pitch is mildly biased low.
        pitch = self._estimate_pitch(centers, n_expected)
        offset = self._choose_seed_anchor(
            centers, pitch, n_expected, image_dim,
        )
        indices = np.clip(
            np.rint((centers - offset) / pitch).astype(int),
            0, n_expected - 1,
        )

        converged = False
        for iter_num in range(MAX_ITER):
            medians, unique_idx = self._aggregate_to_cell_medians(
                centers, indices,
            )
            new_pitch, new_offset = self._fit_pitch_and_offset(
                medians, unique_idx, axis_label=axis_label,
            )
            if new_pitch <= 0:
                warnings.warn(
                    f"AutoGridFinder [iter-fit-degenerate] ({axis_label}, "
                    f"iterative): {span_word} pitch ({new_pitch:.2f}) became "
                    f"non-positive at iter {iter_num} of refinement; falling "
                    f"back to uniform {span_word} edges.",
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
                f"AutoGridFinder [non-convergence] ({axis_label}, iterative): "
                f"{span_word} index assignments did not stabilize after "
                f"{MAX_ITER} iterations; using the last fit (no fallback).",
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
                axis_label=axis_label,
            )
        else:
            warnings.warn(
                f"AutoGridFinder [insufficient-inliers] ({axis_label}, "
                f"iterative): only {n_inliers} of {len(medians)} {span_word} "
                f"cell-medians fall within the residual threshold "
                f"({threshold:.2f} px = pitch * {self.residual_fraction}); "
                f"skipping refit and using the last fit.",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
        if pitch <= 0:
            warnings.warn(
                f"AutoGridFinder [refit-degenerate] ({axis_label}, "
                f"iterative): refit {span_word} pitch ({pitch:.2f}) is "
                f"non-positive after inlier filtering; falling back to "
                f"uniform {span_word} edges.",
                AutoGridFinderFallbackWarning,
                stacklevel=4,
            )
            return self._uniform_edges(n_expected, image_dim)

        # Compute span from raw (unclipped) indices: unique_idx is clipped
        # to [0, n_expected - 1], so a spurious detection past the image
        # edge would otherwise inflate the span and mask genuinely sparse
        # coverage. Using round((centers - offset) / pitch) without clip
        # makes the span reflect the actual data extent.
        raw_indices = np.rint((centers - offset) / pitch).astype(int)
        span = int(raw_indices.max() - raw_indices.min()) + 1
        offset = self._resolve_span_anchor(
            centers, pitch, offset, n_expected, span,
            axis_label, span_word,
        )
        return self._compute_grid_edges(
            pitch, offset, n_expected, image_dim, axis_label=axis_label,
        )

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
        otherwise routes to the iterative pipeline (data-driven seed +
        cell-median refinement).
        """
        axis_label = (
            "rows" if axis == 0 else "cols" if axis == 1 else f"axis {axis}"
        )
        span_word = (
            "row" if axis_label == "rows"
            else "column" if axis_label == "cols"
            else axis_label
        )
        try:
            centers = self._extract_axis_centers(info_table, axis)
        except (KeyError, IndexError) as exc:
            warnings.warn(
                f"AutoGridFinder [extract-centers] ({axis_label}): could not "
                f"read {span_word} centers from info_table "
                f"({type(exc).__name__}); falling back to uniform "
                f"{span_word} edges.",
                AutoGridFinderFallbackWarning,
                stacklevel=3,
            )
            return self._uniform_edges(n_expected, image_dim)

        if len(centers) < 2:
            warnings.warn(
                f"AutoGridFinder [insufficient-centers] ({axis_label}): only "
                f"{len(centers)} {span_word} centers available (need >= 2 to "
                f"fit a pitch); falling back to uniform {span_word} edges.",
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
                f"AutoGridFinder [pipeline-simple] ({axis_label}): "
                f"{len(centers)} {span_word} centers "
                f"(< 1.5 * n_expected = {1.5 * n_expected:.1f}) and "
                f"{n_occupied}/{n_expected} occupied {span_word} cells "
                f"(< 50%); routing to simple pipeline instead of iterative.",
                AutoGridFinderFallbackWarning,
                stacklevel=3,
            )
            return self._fit_axis_edges_simple(
                centers, n_expected, image_dim, axis_label=axis_label,
            )

        # High-N guard: skip fitting for pathological object counts
        if len(centers) > self._MAX_OBJECTS_PER_CELL * n_expected:
            warnings.warn(
                f"AutoGridFinder [high-object-count] ({axis_label}): detected "
                f"{len(centers)} {span_word} centers for {n_expected} "
                f"expected {span_word} positions "
                f"(> {self._MAX_OBJECTS_PER_CELL} per cell). Falling back to "
                f"uniform {span_word} grid spacing.",
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
                    "AutoGridFinder [no-objects] (rows): no objects detected "
                    "in image; falling back to uniform row edges.",
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
                    "AutoGridFinder [no-objects] (cols): no objects detected "
                    "in image; falling back to uniform column edges.",
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
                    "AutoGridFinder [no-objects] (rows + cols): no objects "
                    "detected in image; falling back to uniform grid edges "
                    "on both axes.",
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

    # Okabe-Ito palette for the inspect() diagnostic plots. Annotated
    # ``ClassVar`` so pydantic keeps them as plain class constants — an
    # un-annotated underscore name is otherwise collected as a
    # ``ModelPrivateAttr`` and class-level access returns the wrapper.
    _OI_NAVY: ClassVar[str] = "#003660"
    _OI_ORANGE: ClassVar[str] = "#E69F00"
    _OI_SKY: ClassVar[str] = "#56B4E9"
    _OI_GREEN: ClassVar[str] = "#009E73"
    _OI_VERMILION: ClassVar[str] = "#D55E00"
    _OI_BLUE: ClassVar[str] = "#0072B2"
    _OI_PURPLE: ClassVar[str] = "#CC79A7"
    _OI_GREY: ClassVar[str] = "#BBBBBB"

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
        cls, row_stats: dict, col_stats: dict,
        info_table_empty: bool,
    ):
        """Histogram of successive center diffs with pitch markers.

        Reads ``centers``, ``image_pitch``, and ``fit_pitch`` from each
        axis's stats dict (:meth:`_compute_axis_dashboard_stats`) — does
        not recompute. Reference lines:

        - **Green solid**: fitted pitch.
        - **Vermilion dashed**: 1× image_pitch (structural prior).
          When this disagrees with the green line, the plate doesn't
          fill the image edge-to-edge.
        - **Grey dashed/dotted**: 2× and 3× image_pitch for spotting
          sparse-coverage peaks.
        """
        import panel as pn
        import matplotlib.pyplot as plt

        with plt.rc_context(cls._dashboard_rcparams()):
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))

            for ax, stats in [(axes[0], row_stats), (axes[1], col_stats)]:
                label = stats["label"]
                image_pitch = stats["image_pitch"]
                fit_pitch = stats["fit_pitch"]
                centers = stats["centers"]

                if info_table_empty:
                    ax.text(
                        0.5, 0.5, "No objects detected",
                        ha="center", va="center", fontsize=10,
                        color="#8892a4", transform=ax.transAxes,
                    )
                    ax.set_title(f"{label} diffs")
                    continue

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
                    fit_pitch, color=cls._OI_GREEN, ls="-", lw=1.5,
                    label=f"fit ({fit_pitch:.0f})",
                )
                ax.axvline(
                    image_pitch, color=cls._OI_VERMILION, ls="--", lw=1.2,
                    label=f"1x ip ({image_pitch:.0f})",
                )
                ax.axvline(
                    2 * image_pitch, color=cls._OI_GREY, ls="--", lw=1.0,
                    label="2x ip",
                )
                ax.axvline(
                    3 * image_pitch, color=cls._OI_GREY, ls=":", lw=1.0,
                    label="3x ip",
                )

                ax.set_xlabel("Δ between adjacent centers (px)")
                ax.set_ylabel("count")
                ax.set_title(
                    f"{label} diffs (fit={fit_pitch:.0f}, "
                    f"image_pitch={image_pitch:.0f})"
                )
                ax.legend(fontsize=7, framealpha=0.8)

            fig.tight_layout()
            pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
            plt.close(fig)
            return pane

    @classmethod
    def _plot_axis_occupancy(
        cls, row_stats: dict, col_stats: dict,
        info_table_empty: bool,
    ):
        """Bar chart of detection counts per cell index per axis.

        Reads everything from the per-axis stats dicts produced by
        :meth:`_compute_axis_dashboard_stats` — does **not** recompute
        indices/counts. Primary bars (navy / vermilion) use ``fit_counts``
        (the algorithm's decision); when ``stats["agree"]`` is False,
        ``ip_counts`` is overlaid as horizontal grey markers for
        comparison and the title reports both occupancies.
        """
        import panel as pn
        import matplotlib.pyplot as plt

        with plt.rc_context(cls._dashboard_rcparams()):
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))

            for ax, stats in [(axes[0], row_stats), (axes[1], col_stats)]:
                label = stats["label"]
                n_expected = stats["n_expected"]
                fit_counts = stats["fit_counts"]
                ip_counts = stats["ip_counts"]
                fit_occupied = stats["fit_occupied"]
                ip_occupied = stats["ip_occupied"]

                if info_table_empty:
                    ax.text(
                        0.5, 0.5, "No objects detected",
                        ha="center", va="center", fontsize=10,
                        color="#8892a4", transform=ax.transAxes,
                    )
                    ax.set_title(f"{label} occupancy")
                    continue

                colors = [
                    cls._OI_VERMILION if c == 0 else cls._OI_NAVY
                    for c in fit_counts
                ]
                bars = ax.bar(
                    range(n_expected), fit_counts, color=colors,
                    alpha=0.85, label="fitted",
                )
                for bar, c in zip(bars, fit_counts):
                    if c > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), str(int(c)),
                            ha="center", va="bottom",
                            fontsize=7, fontfamily="monospace",
                            color="#2e3a4e",
                        )

                # Image-pitch counts as horizontal grey markers — only
                # shown when they disagree with fitted (otherwise the
                # plot stays clean).
                if not stats["agree"]:
                    ax.scatter(
                        range(n_expected), ip_counts,
                        marker="_", s=80, color=cls._OI_GREY, alpha=0.85,
                        label="image-pitch",
                    )
                    ax.legend(fontsize=7, framealpha=0.8, loc="upper right")

                ax.set_xticks(range(n_expected))
                ax.set_xlabel(f"{label} index")
                ax.set_ylabel("# detections")
                if stats["agree"]:
                    ax.set_title(
                        f"{label} occupancy "
                        f"({fit_occupied}/{n_expected} cells)"
                    )
                else:
                    ax.set_title(
                        f"{label} occupancy "
                        f"(fit: {fit_occupied}/{n_expected}, "
                        f"ip: {ip_occupied}/{n_expected})"
                    )

            fig.tight_layout()
            pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
            plt.close(fig)
            return pane

    @classmethod
    def _build_inspect_summary(
        cls, result: dict, row_stats: dict, col_stats: dict,
        image_shape: tuple[int, ...],
    ):
        """Markdown summary panel with grid diagnostics.

        Reads all per-axis numbers from the supplied stats dicts
        (:meth:`_compute_axis_dashboard_stats`) — does not recompute.
        """
        import panel as pn

        info_table = result["info_table"]
        timings = result["timings"]
        grid_df = result["grid_df"]
        nrows = row_stats["n_expected"]
        ncols = col_stats["n_expected"]

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

        def _row_pair(label: str, fit: int, ip: int, total: int) -> str:
            """Render a 'fit / ip' summary row, marking disagreements."""
            same = fit == ip
            marker = "" if same else " ⚠"
            if same:
                return (
                    f"| {label} | {fit}/{total} ({fit / total:.0%}) |\n"
                )
            return (
                f"| {label}{marker} | fit: {fit}/{total} "
                f"({fit / total:.0%}), ip: {ip}/{total} "
                f"({ip / total:.0%}) |\n"
            )

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
            f"| Row pitch | {row_stats['fit_pitch']:.1f} px |\n"
            f"| Col pitch | {col_stats['fit_pitch']:.1f} px |\n"
            f"| Pipeline path | {result['pipeline_path']} |\n"
            + _row_pair(
                "Row occupancy", row_stats["fit_occupied"],
                row_stats["ip_occupied"], nrows,
            )
            + _row_pair(
                "Col occupancy", col_stats["fit_occupied"],
                col_stats["ip_occupied"], ncols,
            )
            + _row_pair(
                "Row span coverage", row_stats["fit_span"],
                row_stats["ip_span"], nrows,
            )
            + _row_pair(
                "Col span coverage", col_stats["fit_span"],
                col_stats["ip_span"], ncols,
            )
            + f"| Total time | {total_time:.3f} s |\n"
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

        # Compute per-axis dashboard stats ONCE; every downstream panel
        # consumes from these dicts so the numbers cannot drift between
        # panels. See _compute_axis_dashboard_stats for the schema.
        row_stats = self._compute_axis_dashboard_stats(
            result["info_table"], axis=0, n_expected=self.nrows,
            image_dim=image.shape[0], edges=result["row_edges"],
        )
        col_stats = self._compute_axis_dashboard_stats(
            result["info_table"], axis=1, n_expected=self.ncols,
            image_dim=image.shape[1], edges=result["col_edges"],
        )
        info_table_empty = bool(result["info_table"].empty)

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
            result, row_stats, col_stats, image.shape,
        )
        p5 = self._plot_successive_diffs(
            row_stats, col_stats, info_table_empty,
        )
        p6 = self._plot_axis_occupancy(
            row_stats, col_stats, info_table_empty,
        )

        return pn.Column(
            header,
            pn.Row(p1, p4),
            pn.Row(p3, p2),
            pn.Row(p5, p6),
        )


AutoGridFinder.measure.__doc__ = AutoGridFinder._operate.__doc__
AutoGridFinder.__doc__ = GRID.append_rst_to_doc(AutoGridFinder.__doc__)
