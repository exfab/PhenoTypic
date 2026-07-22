from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Annotated, ClassVar

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd

from phenotypic.abc_ import GridFinder
from phenotypic.schema import BBOX
from phenotypic.sdk_.typing_ import TuneSpec


class CenteredAutoGridFinderFallbackWarning(UserWarning):
    """Warning category for fallbacks and bounded-ambiguous fits in
    :class:`CenteredAutoGridFinder` (degenerate comb-response, ICP failure,
    bound contradiction, low colony count). Filter in batch runs::

        import warnings
        from phenotypic.grid import CenteredAutoGridFinderFallbackWarning
        warnings.filterwarnings("ignore", category=CenteredAutoGridFinderFallbackWarning)
    """


class CenteredAutoGridFinder(GridFinder):
    """Center-anchored grid finder for sparse arrayed plates.

    Fits a regular axis-aligned grid (single isotropic pitch + center) to
    detected colony centers by their *periodicity* rather than their *span*,
    so it survives empty edge/interior rows that break span-based fitting.
    Assumes the plate is roughly centered in the (de-rotated) frame. See the
    design spec for the algorithm.

    Args:
        nrows: Number of grid rows (default 8 — 96-well plate).
        ncols: Number of grid columns (default 12 — 96-well plate).
        residual_fraction: ICP robust-trim threshold as a fraction of pitch
            (default 0.25).
        n_pitch_samples: Comb-response scan resolution (default 512).
        response_floor: Fundamental-selection threshold as a fraction of the
            peak comb-response (default 0.8).
        max_iter: ICP iteration cap per multi-start candidate (default 6).
        min_fit_objects: Below this colony count the fit is treated as
            bounded-ambiguous (default 6).
        warn: Emit :class:`CenteredAutoGridFinderFallbackWarning` (default False).

    Notes:
        nrows/ncols must match the physical plate; a mismatch produces a wrong
        grid silently (no internal guard). For multiple colonies per well use a
        downstream refiner (KeepNearestCenter / KeepSectionLargest /
        MergeWithinSection); this finder assigns faithfully, many-to-one. A fitted
        grid may have one outer edge clipped to the image boundary, but placements
        that would collapse an entire cell are rejected before assignment.

    Examples:
        Default 96-well fit on the bundled synthetic plate:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.grid import CenteredAutoGridFinder
        >>> image = OtsuDetector().apply(load_synth_yeast_plate())
        >>> finder = CenteredAutoGridFinder(nrows=8, ncols=12)
        >>> grid_df = finder.measure(image)
        >>> len(finder.get_row_edges(image)) == 9
        True
        >>> len(finder.get_col_edges(image)) == 13
        True
    """

    SPAN_PCT_LOW: ClassVar[float] = 5.0
    SPAN_PCT_HIGH: ClassVar[float] = 95.0
    ABSOLUTE_FLOOR: ClassVar[float] = 0.6   # pooled comb response (max 2.0) below which "no periodicity"
    DET_EPS: ClassVar[float] = 1e-6
    # Sub-pixel residual tie tolerance: a later multi-start candidate must beat the
    # incumbent by more than this (px) to replace it, so translation-invariant ties
    # resolve to the nearest-image-center candidate (tried first) rather than to
    # floating-point noise.
    _RESIDUAL_TIE_TOL: ClassVar[float] = 1e-6

    nrows: Annotated[int, TuneSpec(tunable=False)] = 8
    ncols: Annotated[int, TuneSpec(tunable=False)] = 12
    residual_fraction: Annotated[float, TuneSpec(0.1, 0.5)] = 0.25
    n_pitch_samples: Annotated[int, TuneSpec(tunable=False)] = 512
    response_floor: Annotated[float, TuneSpec(0.5, 0.95)] = 0.8
    max_iter: Annotated[int, TuneSpec(tunable=False)] = 6
    min_fit_objects: Annotated[int, TuneSpec(tunable=False)] = 6
    warn: bool = False

    # ---- helpers (filled in by later tasks) ----
    def _uniform_edges(self, n: int, image_dim: int) -> np.ndarray:
        """Evenly spaced edges spanning the full axis (length n+1)."""
        return np.linspace(0, image_dim, n + 1)

    def _compute_bounds(self, x: np.ndarray, y: np.ndarray, H: int, W: int) -> tuple[float, float]:
        """Object-derived pitch floor (percentile span) + image-derived ceiling
        (outermost cell centers fit the frame). NEVER uses image_dim/n as a floor."""
        x_span = np.percentile(x, self.SPAN_PCT_HIGH) - np.percentile(x, self.SPAN_PCT_LOW)
        y_span = np.percentile(y, self.SPAN_PCT_HIGH) - np.percentile(y, self.SPAN_PCT_LOW)
        p_min = max(x_span / max(self.ncols - 1, 1), y_span / max(self.nrows - 1, 1))
        p_max = min(H / max(self.nrows - 1, 1), W / max(self.ncols - 1, 1))
        return float(p_min), float(p_max)

    @staticmethod
    def _comb_mag(coords: np.ndarray, p: float) -> float:
        return float(np.abs(np.exp(1j * 2.0 * np.pi * coords / p).mean()))

    def _estimate_pitch(self, x: np.ndarray, y: np.ndarray,
                        p_min: float, p_max: float) -> tuple[float, bool]:
        """Pooled comb-response over [p_min, p_max]; pick the FUNDAMENTAL (largest p
        among strict local maxima >= response_floor*peak). Returns (pitch, ok)."""
        if not (p_max > p_min > 0):
            return float(p_max), False
        ps = np.linspace(p_min, p_max, self.n_pitch_samples)
        Rr = np.array([self._comb_mag(x, p) + self._comb_mag(y, p) for p in ps])
        peak = float(Rr.max())
        if peak < self.ABSOLUTE_FLOOR:
            return float(ps[int(np.argmax(Rr))]), False
        # Local maxima above the relative floor; choose the largest p (fundamental).
        # Boundary samples count as candidates so a true pitch landing exactly on the
        # p_min floor (e.g. the outermost columns fully span the frame, making the
        # percentile span == (C-1)*p) is recoverable — otherwise the peak at index 0
        # would be excluded by a strict-interior-only check. The ABSOLUTE_FLOOR guard
        # above still rejects genuinely non-periodic layouts.
        n = len(ps)
        floor_val = self.response_floor * peak
        idx = []
        for i in range(n):
            left = Rr[i] > Rr[i - 1] if i > 0 else True
            right = Rr[i] > Rr[i + 1] if i < n - 1 else True
            if left and right and Rr[i] >= floor_val:
                idx.append(i)
        if not idx:
            return float(ps[int(np.argmax(Rr))]), False
        p0 = float(ps[max(idx)])
        return p0, True

    @staticmethod
    def _phase(coords: np.ndarray, p: float) -> float:
        return float(np.angle(np.exp(1j * 2.0 * np.pi * coords / p).mean()))

    def _center_candidates(self, coords: np.ndarray, p: float,
                           n_cells: int, axis_len: int) -> list[float]:
        """Integer placements of the grid center consistent with the comb phase,
        kept if within the FULL in-frame offset box, ordered nearest-image-center first."""
        base = (self._phase(coords, p) / (2.0 * np.pi)) * p      # cell-center phase, in (-p/2, p/2]
        grid_extent = (n_cells - 1) * p
        half = (axis_len - grid_extent) / 2.0 + p                # full in-frame offset + 1 pitch slack
        img_c = axis_len / 2.0
        cands = []
        for m in range(-n_cells, n_cells + 1):
            c = base + (n_cells - 1) / 2.0 * p + m * p
            if abs(c - img_c) <= half:
                cands.append(float(c))
        return sorted(cands, key=lambda c: abs(c - img_c))

    @staticmethod
    def _bounded_solution(
            solution: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            a: np.ndarray,
            b: np.ndarray,
            p_min: float,
            p_max: float,
    ) -> tuple[float, float, float] | None:
        """Apply pitch bounds to a lattice solve and re-optimize its center."""
        cx, cy, p = (float(value) for value in solution)
        if not np.all(np.isfinite([cx, cy, p])):
            return None

        bounded_p = float(np.clip(p, p_min, p_max))
        if bounded_p != p:
            cx = float(np.mean(x - a * bounded_p))
            cy = float(np.mean(y - b * bounded_p))
        return cx, cy, bounded_p

    def _icp_refine(
            self,
            x: np.ndarray,
            y: np.ndarray,
            cx: float,
            cy: float,
            p: float,
            p_min: float,
            p_max: float,
    ) -> tuple[float, float, float, float] | None:
        """Closed-form assign->solve ICP from one seed. Returns (cx,cy,p,mean_residual)
        or None if the design matrix is singular (cannot constrain pitch).

        The fitted pitch is constrained to ``[p_min, p_max]`` after every solve.
        When a bound is active, the grid center is re-optimized for that fixed pitch.
        """
        if not (np.isfinite(p_min) and np.isfinite(p_max) and 0 < p_min <= p_max):
            return None

        R, C, N = self.nrows, self.ncols, len(x)
        a = b = None
        p = float(np.clip(p, p_min, p_max))
        for _ in range(self.max_iter):
            jx = np.clip(np.round((x - cx) / p + (C - 1) / 2.0), 0, C - 1)
            iy = np.clip(np.round((y - cy) / p + (R - 1) / 2.0), 0, R - 1)
            a = jx - (C - 1) / 2.0
            b = iy - (R - 1) / 2.0
            A = np.array([[N, 0.0, a.sum()],
                          [0.0, N, b.sum()],
                          [a.sum(), b.sum(), (a * a + b * b).sum()]])
            if abs(np.linalg.det(A)) < self.DET_EPS:
                return None
            rhs = np.array([x.sum(), y.sum(), (a * x + b * y).sum()])
            bounded = self._bounded_solution(
                np.linalg.solve(A, rhs), x, y, a, b, p_min, p_max
            )
            if bounded is None:
                return None
            cx, cy, p = bounded
            # one-pass robust trim then re-solve on inliers
            res = np.hypot(x - (cx + a * p), y - (cy + b * p))
            inl = res <= self.residual_fraction * p
            if 3 <= inl.sum() < N:
                ai, bi, xi, yi, ni = a[inl], b[inl], x[inl], y[inl], int(inl.sum())
                A2 = np.array([[ni, 0.0, ai.sum()],
                               [0.0, ni, bi.sum()],
                               [ai.sum(), bi.sum(), (ai * ai + bi * bi).sum()]])
                if abs(np.linalg.det(A2)) >= self.DET_EPS:
                    bounded = self._bounded_solution(
                        np.linalg.solve(
                            A2,
                            np.array([
                                xi.sum(),
                                yi.sum(),
                                (ai * xi + bi * yi).sum(),
                            ]),
                        ),
                        xi,
                        yi,
                        ai,
                        bi,
                        p_min,
                        p_max,
                    )
                    if bounded is None:
                        return None
                    cx, cy, p = bounded
        if a is None or b is None:
            # max_iter <= 0: the loop never ran, so nothing was fitted.
            return None
        res = np.hypot(x - (cx + a * p), y - (cy + b * p))
        return float(cx), float(cy), float(p), float(res.mean())

    def _multi_start_refine(
            self,
            x: np.ndarray,
            y: np.ndarray,
            p0: float,
            p_min: float,
            p_max: float,
            cx_cands: list[float],
            cy_cands: list[float],
            H: int,
            W: int,
    ) -> tuple[tuple[float, float, float, float] | None, bool]:
        """Run ICP from every candidate and keep the best feasible registration.

        Residual is the primary score. Fits tied within ``_RESIDUAL_TIE_TOL``
        explicitly prefer the grid center nearest the image center. Candidates whose
        clipped edges collapse a cell are discarded before scoring. The returned
        boolean records whether ICP produced any result, including infeasible ones.
        """
        best: tuple[float, float, float, float] | None = None
        best_center_distance = np.inf
        saw_refined_candidate = False
        for cx0 in cx_cands:
            for cy0 in cy_cands:
                out = self._icp_refine(x, y, cx0, cy0, p0, p_min, p_max)
                if out is None:
                    continue
                saw_refined_candidate = True
                cx, cy, p, residual = out
                if not np.all(np.isfinite(out)) or not (p_min <= p <= p_max):
                    continue
                row_edges = self._axis_edges(cy, p, self.nrows, H)
                col_edges = self._axis_edges(cx, p, self.ncols, W - 1)
                if not (
                    self._axis_edges_are_valid(row_edges, self.nrows, H)
                    and self._axis_edges_are_valid(col_edges, self.ncols, W - 1)
                ):
                    continue

                center_distance = float(
                    np.hypot(cx - (W - 1) / 2.0, cy - H / 2.0)
                )
                residual_tied = (
                    best is not None
                    and abs(residual - best[3]) <= self._RESIDUAL_TIE_TOL
                )
                if (
                    best is None
                    or residual < best[3] - self._RESIDUAL_TIE_TOL
                    or (residual_tied and center_distance < best_center_distance)
                ):
                    best = out
                    best_center_distance = center_distance
        return best, saw_refined_candidate

    # ---- centers -> edges ----
    def _axis_edges(self, center: float, p: float, n_cells: int, image_dim: int) -> np.ndarray:
        """Return clipped cell midlines without silently repairing collapsed cells.

        A single outer edge may clip to the frame while the partition remains valid.
        Callers use :meth:`_axis_edges_are_valid` to reject placements where two or
        more clipped edges coincide.
        """
        first_center = center - (n_cells - 1) / 2.0 * p
        edges = first_center - p / 2.0 + np.arange(n_cells + 1) * p
        return np.clip(edges, 0, image_dim)

    @staticmethod
    def _axis_edges_are_valid(
            edges: np.ndarray,
            n_cells: int,
            image_dim: int,
    ) -> bool:
        """Return whether edges form a finite, bounded, strict cell partition."""
        edges = np.asarray(edges)
        return bool(
            edges.ndim == 1
            and len(edges) == n_cells + 1
            and np.all(np.isfinite(edges))
            and np.all(edges >= 0)
            and np.all(edges <= image_dim)
            and np.all(np.diff(edges) > 0)
        )

    @staticmethod
    def _extract_centers(image: "Image"):
        info = image.objects.info(include_metadata=False)
        x = info[str(BBOX.DIST_WEIGHTED_CENTER_CC)].to_numpy(dtype=float)
        y = info[str(BBOX.DIST_WEIGHTED_CENTER_RR)].to_numpy(dtype=float)
        return x, y, info

    def _warn(self, msg: str) -> None:
        if self.warn:
            warnings.warn(f"CenteredAutoGridFinder {msg}",
                          CenteredAutoGridFinderFallbackWarning, stacklevel=2)

    def _centered_uniform(self, p: float, H: int, W: int):
        """Return a strict image-centered grid, safely bounding degenerate pitches."""
        centers_fit_max = min(
            H / max(self.nrows - 1, 1),
            (W - 1) / max(self.ncols - 1, 1),
        )
        full_cells_pitch = min(
            H / max(self.nrows, 1),
            (W - 1) / max(self.ncols, 1),
        )
        if np.isfinite(p) and p > 0:
            safe_p = min(float(p), centers_fit_max)
        else:
            safe_p = full_cells_pitch

        row_edges = self._axis_edges(H / 2.0, safe_p, self.nrows, H)
        col_edges = self._axis_edges((W - 1) / 2.0, safe_p, self.ncols, W - 1)
        if (
            self._axis_edges_are_valid(row_edges, self.nrows, H)
            and self._axis_edges_are_valid(col_edges, self.ncols, W - 1)
        ):
            return row_edges, col_edges

        row_edges = self._axis_edges(H / 2.0, full_cells_pitch, self.nrows, H)
        col_edges = self._axis_edges(
            (W - 1) / 2.0, full_cells_pitch, self.ncols, W - 1
        )
        if (
            self._axis_edges_are_valid(row_edges, self.nrows, H)
            and self._axis_edges_are_valid(col_edges, self.ncols, W - 1)
        ):
            return row_edges, col_edges

        # Pathological dimensions or floating-point resolution: preserve the edge
        # contract independently on each axis rather than returning collapsed bins.
        return (
            self._uniform_edges(self.nrows, H),
            self._uniform_edges(self.ncols, W - 1),
        )

    def _finalize_grid_edges(
            self,
            row_edges: np.ndarray,
            col_edges: np.ndarray,
            fallback_pitch: float,
            H: int,
            W: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Enforce the strict-edge postcondition before delegating to ``pd.cut``."""
        if (
            self._axis_edges_are_valid(row_edges, self.nrows, H)
            and self._axis_edges_are_valid(col_edges, self.ncols, W - 1)
        ):
            return row_edges, col_edges
        self._warn(
            "[invalid-geometry] fitted edges do not form a strict partition; "
            "centered uniform fallback."
        )
        return self._centered_uniform(fallback_pitch, H, W)

    def _fit_grid_from_centers(self, x: np.ndarray, y: np.ndarray, H: int, W: int):
        """Full pipeline on raw center arrays -> (row_edges, col_edges), with the
        sparse-tail fallback ladder. Returns axis-aligned edge arrays."""
        N = len(x)
        # N in {0,1}: no inferable pitch -> centered grid at the max-fitting pitch
        if N < 2:
            self._warn(f"[few-objects] N={N}; centered uniform grid at max pitch.")
            p_max = min(H / max(self.nrows - 1, 1), W / max(self.ncols - 1, 1))
            return self._centered_uniform(p_max, H, W)

        p_min, p_max = self._compute_bounds(x, y, H, W)
        if p_min >= p_max:
            self._warn(f"[bound-inversion] p_min={p_min:.1f} >= p_max={p_max:.1f}; "
                       "centered uniform grid at p_max.")
            return self._centered_uniform(p_max, H, W)

        p0, ok = self._estimate_pitch(x, y, p_min, p_max)
        if not ok:
            self._warn("[degenerate-response] no clear periodicity; centered uniform at p_min.")
            return self._centered_uniform(p_min, H, W)
        if N <= self.min_fit_objects:
            self._warn(f"[few-objects] N={N}: bounded-ambiguous fit (best-effort, not confident).")

        cx_c = self._center_candidates(x, p0, self.ncols, W - 1)
        cy_c = self._center_candidates(y, p0, self.nrows, H)
        best, saw_refined_candidate = self._multi_start_refine(
            x, y, p0, p_min, p_max, cx_c, cy_c, H, W
        )
        if best is None:
            if saw_refined_candidate:
                self._warn(
                    "[invalid-geometry] ICP registrations collapse one or more "
                    "grid cells; centered uniform at comb pitch."
                )
            else:
                self._warn(
                    "[icp-failed] no registration could be refined; "
                    "centered uniform at comb pitch."
                )
            return self._centered_uniform(p0, H, W)
        if best[3] > self.residual_fraction * best[2]:
            self._warn("[icp-failed] no acceptable registration; centered uniform at comb pitch.")
            return self._centered_uniform(p0, H, W)

        cx, cy, p, _res = best
        return self._finalize_grid_edges(
            self._axis_edges(cy, p, self.nrows, H),
            self._axis_edges(cx, p, self.ncols, W - 1),
            p0,
            H,
            W,
        )

    # ---- GridFinder overrides ----
    def get_row_edges(self, image: "Image") -> np.ndarray:
        return self._fit_grid(image)[0]

    def get_col_edges(self, image: "Image") -> np.ndarray:
        return self._fit_grid(image)[1]

    def _fit_grid(self, image: "Image"):
        """(row_edges, col_edges) for *image*, applying the fallback ladder (Task 7)."""
        x, y, _ = self._extract_centers(image)
        return self._fit_grid_from_centers(x, y, image.shape[0], image.shape[1])

    def _operate(self, image: "Image") -> pd.DataFrame:
        x, y, info = self._extract_centers(image)
        row_edges, col_edges = self._fit_grid_from_centers(x, y, image.shape[0], image.shape[1])
        return super()._get_grid_info(image=image, row_edges=row_edges,
                                      col_edges=col_edges, info_table=info)
