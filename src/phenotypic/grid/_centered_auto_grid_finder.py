from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Annotated, ClassVar

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd

from phenotypic.abc_ import GridFinder
from phenotypic.schema import BBOX
from phenotypic.tools_.typing_ import TuneSpec


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
        MergeWithinSection); this finder assigns faithfully, many-to-one.
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

    def _icp_refine(self, x: np.ndarray, y: np.ndarray,
                    cx: float, cy: float, p: float):
        """Closed-form assign->solve ICP from one seed. Returns (cx,cy,p,mean_residual)
        or None if the design matrix is singular (cannot constrain pitch)."""
        R, C, N = self.nrows, self.ncols, len(x)
        a = b = None
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
            cx, cy, p = np.linalg.solve(A, rhs)
            # one-pass robust trim then re-solve on inliers
            res = np.hypot(x - (cx + a * p), y - (cy + b * p))
            inl = res <= self.residual_fraction * p
            if 3 <= inl.sum() < N:
                ai, bi, xi, yi, ni = a[inl], b[inl], x[inl], y[inl], int(inl.sum())
                A2 = np.array([[ni, 0.0, ai.sum()],
                               [0.0, ni, bi.sum()],
                               [ai.sum(), bi.sum(), (ai * ai + bi * bi).sum()]])
                if abs(np.linalg.det(A2)) >= self.DET_EPS:
                    cx, cy, p = np.linalg.solve(A2, np.array([xi.sum(), yi.sum(),
                                                              (ai * xi + bi * yi).sum()]))
        res = np.hypot(x - (cx + a * p), y - (cy + b * p))
        return float(cx), float(cy), float(p), float(res.mean())

    def _multi_start_refine(self, x: np.ndarray, y: np.ndarray, p0: float,
                            cx_cands: list[float], cy_cands: list[float]):
        """Run ICP from every (cx,cy) candidate; keep the lowest-residual result.

        Candidates are supplied nearest-image-center first (the ordering prior). A
        later candidate replaces the current best only if it is *meaningfully* lower
        (by ``_RESIDUAL_TIE_TOL`` px), so a row/column-translation-invariant sparse
        layout — where several integer placements all fit with ~0 residual and differ
        only by floating-point noise — resolves to the placement nearest the image
        center rather than to whichever tie computed a marginally smaller float.
        """
        best = None
        for cx0 in cx_cands:
            for cy0 in cy_cands:
                out = self._icp_refine(x, y, cx0, cy0, p0)
                if out is None:
                    continue
                if best is None or out[3] < best[3] - self._RESIDUAL_TIE_TOL:
                    best = out
        return best

    # ---- centers -> edges ----
    def _axis_edges(self, center: float, p: float, n_cells: int, image_dim: int) -> np.ndarray:
        """n+1 edges = cell-center midlines with outer edges at +/- p/2, clipped to [0, image_dim]."""
        first_center = center - (n_cells - 1) / 2.0 * p
        edges = first_center - p / 2.0 + np.arange(n_cells + 1) * p
        return np.clip(edges, 0, image_dim)

    @staticmethod
    def _extract_centers(image: "Image"):
        info = image.objects.info(include_metadata=False)
        x = info[str(BBOX.DIST_WEIGHTED_CENTER_CC)].to_numpy(dtype=float)
        y = info[str(BBOX.DIST_WEIGHTED_CENTER_RR)].to_numpy(dtype=float)
        return x, y, info

    def _fit_grid_from_centers(self, x: np.ndarray, y: np.ndarray, H: int, W: int):
        """Full pipeline on raw center arrays -> (row_edges, col_edges).
        Fallback ladder lives in Task 7; here assume the happy path (>= 2 colonies,
        valid bounds, periodic). Returns axis-aligned edge arrays."""
        p_min, p_max = self._compute_bounds(x, y, H, W)
        p0, ok = self._estimate_pitch(x, y, p_min, p_max)
        cx_c = self._center_candidates(x, p0, self.ncols, W)
        cy_c = self._center_candidates(y, p0, self.nrows, H)
        best = self._multi_start_refine(x, y, p0, cx_c, cy_c)
        cx, cy, p, _res = best
        row_edges = self._axis_edges(cy, p, self.nrows, H)
        col_edges = self._axis_edges(cx, p, self.ncols, W)
        return row_edges, col_edges

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
