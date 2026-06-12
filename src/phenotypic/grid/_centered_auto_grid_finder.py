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

    # ---- GridFinder overrides ----
    def get_row_edges(self, image: "Image") -> np.ndarray:
        return self._uniform_edges(self.nrows, image.shape[0])

    def get_col_edges(self, image: "Image") -> np.ndarray:
        return self._uniform_edges(self.ncols, image.shape[1])

    def _operate(self, image: "Image") -> pd.DataFrame:
        row_edges = self.get_row_edges(image)
        col_edges = self.get_col_edges(image)
        return super()._get_grid_info(image=image, row_edges=row_edges, col_edges=col_edges)
