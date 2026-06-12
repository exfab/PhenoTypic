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

    # ---- GridFinder overrides ----
    def get_row_edges(self, image: "Image") -> np.ndarray:
        return self._uniform_edges(self.nrows, image.shape[0])

    def get_col_edges(self, image: "Image") -> np.ndarray:
        return self._uniform_edges(self.ncols, image.shape[1])

    def _operate(self, image: "Image") -> pd.DataFrame:
        row_edges = self.get_row_edges(image)
        col_edges = self.get_col_edges(image)
        return super()._get_grid_info(image=image, row_edges=row_edges, col_edges=col_edges)
