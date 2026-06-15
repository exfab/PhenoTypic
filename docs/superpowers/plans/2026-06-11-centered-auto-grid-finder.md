# CenteredAutoGridFinder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `CenteredAutoGridFinder`, a sparse-robust `GridFinder` that fits a center-anchored regular grid to colony centers via comb-response pitch + multi-start ICP, and make it the default finder for grid images.

**Architecture:** After upstream de-rotation, the grid is a 3-parameter model (single isotropic pitch `p` + center `cx,cy`). Pitch comes from the periodicity of object centers (Kuramoto comb-response over a bounded `[p_min,p_max]`), the center from the comb phase enumerated as a few integer placements across the full in-frame offset, and the final geometry from a closed-form 3×3 multi-start ICP that selects the placement with lowest residual. Output is axis-aligned `row_edges`/`col_edges` consumed by the existing `GridFinder._get_grid_info` machinery (faithful many-to-one assignment, no collision handling).

**Tech Stack:** Python, pydantic v2 (operation fields), numpy, pandas; project test runner `uv run pytest`; `phenotypic.schema.BBOX`/`GRID` column enums; `phenotypic.tools_.typing_.TuneSpec`.

**Spec:** `docs/superpowers/specs/2026-06-11-centered-auto-grid-finder-design.md` (read it first).

---

## File Structure

- **Create** `src/phenotypic/grid/_centered_auto_grid_finder.py` — the `CenteredAutoGridFinder` class, `CenteredAutoGridFinderFallbackWarning`, and all private fit helpers. One responsibility: fit a centered regular grid to object centers and emit edges.
- **Modify** `src/phenotypic/grid/__init__.py` — export the new class + warning.
- **Modify** `src/phenotypic/_core/_image_parts/_grid_image_handler.py` — flip the `grid_finder is None` default.
- **Modify** `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py` — flip the injected default finder.
- **Create** `tests/unit/grid/test_centered_auto_grid_finder.py` — full unit suite (synthetic lattices) + integration (decimated synth plate).
- **Modify** `src/phenotypic/measure/_measure_bounds.py` — replace the DT-argmax center with the DT-weighted centroid (Task 0; fixes budding-yeast doublets, improves both finders).
- **Modify** existing tests that assume `AutoGridFinder` is the default (found during Task 9 sweep).

**Conventions to mirror** (read `src/phenotypic/grid/_auto_grid_finder.py` lines 1–140, 1079–1162):
- pydantic fields are class-level `Annotated[..., TuneSpec(...)]`; **no `__init__`**; keyword-only construction.
- `ClassVar` for internal constants.
- Warning subclass of `UserWarning`.
- `_operate`/`get_row_edges`/`get_col_edges` are the three `GridFinder` overrides; `_operate` ends by calling `super()._get_grid_info(image=..., row_edges=..., col_edges=..., info_table=...)`.
- Centers come from `image.objects.info(include_metadata=False)`, columns `str(BBOX.DIST_WEIGHTED_CENTER_CC)` (= `"Bbox_DistWeightedCenterCC"`, the x / column axis) and `str(BBOX.DIST_WEIGHTED_CENTER_RR)` (y / row axis).
- Axis convention: **axis 1 = columns = x = CC = width `W = image.shape[1]`**; **axis 0 = rows = y = RR = height `H = image.shape[0]`**.

---

## Task 0: DT-weighted centroid center (fixes budding-yeast doublets) — do first

**Why first:** `DIST_WEIGHTED_CENTER` is currently `ndi.maximum_position(dt)` (DT argmax), which lands on one lobe of a two-peak (budding) colony. Switching to the DT-weighted centroid `ndi.center_of_mass(dt)` fixes this and is read by `CenteredAutoGridFinder` (and `AutoGridFinder`). Doing it first means Tasks 9–10 (default flip, integration, regression) run against the final center. Tasks 1–7 use synthetic lattice arrays directly and are unaffected either way.

**Files:**
- Modify: `src/phenotypic/measure/_measure_bounds.py:~116`
- Test: `tests/unit/measure/test_measure_bounds.py`

- [ ] **Step 1: Write the failing test** (dumbbell colony → center at the neck, not a lobe)

```python
# add to tests/unit/measure/test_measure_bounds.py, in class TestMeasureBounds
def test_dist_center_is_weighted_centroid_not_argmax_on_dumbbell(self, sample_image, measurer):
    """A two-lobe (budding) colony: the DT-weighted centroid sits at the neck
    (~midway), NOT on one lobe as DT-argmax (maximum_position) would."""
    image = sample_image.copy()
    objmap = np.zeros_like(image.objmap[:])
    objmap[100:140, 100:140] = 1     # lobe A, center cc=120
    objmap[100:140, 200:240] = 1     # lobe B, center cc=220
    objmap[118:122, 140:200] = 1     # thin neck joining them
    image.objmap[:] = objmap
    df = measurer.measure(image)
    cc = float(df[str(BBOX.DIST_WEIGHTED_CENTER_CC)].iloc[0])
    rr = float(df[str(BBOX.DIST_WEIGHTED_CENTER_RR)].iloc[0])
    assert 155 < cc < 185      # between the lobes (~170), not ~120 or ~220
    assert 110 < rr < 130
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/measure/test_measure_bounds.py::TestMeasureBounds::test_dist_center_is_weighted_centroid_not_argmax_on_dumbbell -q`
Expected: FAIL — current argmax gives `cc ≈ 120` (or 220), outside `(155, 185)`.

- [ ] **Step 3: Replace argmax with the DT-weighted centroid (+ degenerate guard)**

In `src/phenotypic/measure/_measure_bounds.py`, replace the `positions = ndi.maximum_position(...)` line (~L116) and its `np.asarray` with:

```python
            # DT-weighted centroid (Sum(dt*pos)/Sum(dt)) — robust to filament hyphae
            # (low DT weight) AND budding doublets (two lobes balance to the neck),
            # replacing the former DT-argmax which snapped onto a single lobe.
            positions = np.asarray(
                ndi.center_of_mass(dt, labels=objmap, index=labels), dtype=float
            )
            # Degenerate guard: objects whose pixels were all zeroed as inter-object
            # boundary have zero DT mass -> center_of_mass returns NaN. Fall back to
            # the unweighted (geometric) mask centroid for those labels.
            nan_rows = np.isnan(positions).any(axis=1)
            if nan_rows.any():
                geom = np.asarray(
                    ndi.center_of_mass(nonzero.astype(float), labels=objmap, index=labels),
                    dtype=float,
                )
                positions[nan_rows] = geom[nan_rows]
```

(Keep the surrounding `binary`/`dt`/`labels` computation unchanged. `ndi` is already imported in this module.)

- [ ] **Step 4: Run test to verify it passes + no MeasureBounds regression**

Run: `uv run pytest tests/unit/measure/test_measure_bounds.py -q`
Expected: PASS (new test + all existing MeasureBounds tests; existing tests only assert the center is inside the bbox, which the centroid still satisfies).

- [ ] **Step 5: Verify AutoGridFinder still fits (it now reads the centroid)**

Run: `uv run pytest tests/unit/grid/test_auto_grid_finder.py -q`
Expected: PASS. If a tolerance-tight assertion shifts because round-colony argmax≈centroid moved by <1px, widen the tolerance or update the expected value (confirm the new value is correct), and note it in the commit.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/measure/_measure_bounds.py tests/unit/measure/test_measure_bounds.py
git commit -m "fix(measure): DIST_WEIGHTED_CENTER uses DT-weighted centroid (handles budding doublets)"
```

---

## Task 1: Scaffold class, warning, fields, exports (working uniform-grid finder)

**Files:**
- Create: `src/phenotypic/grid/_centered_auto_grid_finder.py`
- Modify: `src/phenotypic/grid/__init__.py`
- Test: `tests/unit/grid/test_centered_auto_grid_finder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/grid/test_centered_auto_grid_finder.py
import numpy as np
import pytest

from phenotypic.abc_ import GridFinder
from phenotypic.grid import CenteredAutoGridFinder, CenteredAutoGridFinderFallbackWarning


def test_is_gridfinder_and_constructs_keyword_only():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    assert isinstance(f, GridFinder)
    assert (f.nrows, f.ncols) == (8, 12)
    with pytest.raises(Exception):
        CenteredAutoGridFinder(8, 12)  # positional construction is rejected


def test_warning_class_is_userwarning():
    assert issubclass(CenteredAutoGridFinderFallbackWarning, UserWarning)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py -q`
Expected: FAIL — `ImportError: cannot import name 'CenteredAutoGridFinder'`.

- [ ] **Step 3: Create the module skeleton**

```python
# src/phenotypic/grid/_centered_auto_grid_finder.py
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

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
```

- [ ] **Step 4: Add exports**

```python
# src/phenotypic/grid/__init__.py  — replace the import block + __all__
from ._auto_grid_finder import AutoGridFinder
from ._centered_auto_grid_finder import (
    CenteredAutoGridFinder,
    CenteredAutoGridFinderFallbackWarning,
)
from ._manual_grid_finder import ManualGridFinder
from ._grid_apply import GridApply

__all__ = [
    "GridApply",
    "AutoGridFinder",
    "CenteredAutoGridFinder",
    "CenteredAutoGridFinderFallbackWarning",
    "ManualGridFinder",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/grid/_centered_auto_grid_finder.py src/phenotypic/grid/__init__.py tests/unit/grid/test_centered_auto_grid_finder.py
git commit -m "feat(grid): scaffold CenteredAutoGridFinder (uniform-grid stub)"
```

---

## Task 2: Bounds — percentile span floor, centers-fit ceiling

**Files:**
- Modify: `src/phenotypic/grid/_centered_auto_grid_finder.py`
- Test: `tests/unit/grid/test_centered_auto_grid_finder.py`

- [ ] **Step 1: Write the failing test**

```python
def test_compute_bounds_centers_fit_ceiling_and_percentile_floor():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    # 12 columns at pitch 404, 8 rows; occupied x spans cols 0..11, y spans rows 1..6
    H, W = 3152, 5066
    x = np.array([311 + 404 * j for j in range(12)], dtype=float)          # full column span
    y = np.array([162 + 404 * i for i in (1, 2, 3, 5, 6)], dtype=float)     # rows 1..6 occupied
    p_min, p_max = f._compute_bounds(x, y, H, W)
    # ceiling = min(H/(R-1), W/(C-1)) = min(450.3, 460.5) = 450.3
    assert p_max == pytest.approx(min(H / 7, W / 11), rel=1e-6)
    # floor uses percentile span; must be <= true pitch 404 and < p_max (valid window)
    assert p_min < p_max
    assert p_min <= 404.0 + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py::test_compute_bounds_centers_fit_ceiling_and_percentile_floor -q`
Expected: FAIL — `AttributeError: ... has no attribute '_compute_bounds'`.

- [ ] **Step 3: Implement `_compute_bounds`**

```python
    def _compute_bounds(self, x: np.ndarray, y: np.ndarray, H: int, W: int) -> tuple[float, float]:
        """Object-derived pitch floor (percentile span) + image-derived ceiling
        (outermost cell centers fit the frame). NEVER uses image_dim/n as a floor."""
        x_span = np.percentile(x, self.SPAN_PCT_HIGH) - np.percentile(x, self.SPAN_PCT_LOW)
        y_span = np.percentile(y, self.SPAN_PCT_HIGH) - np.percentile(y, self.SPAN_PCT_LOW)
        p_min = max(x_span / max(self.ncols - 1, 1), y_span / max(self.nrows - 1, 1))
        p_max = min(H / max(self.nrows - 1, 1), W / max(self.ncols - 1, 1))
        return float(p_min), float(p_max)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py::test_compute_bounds_centers_fit_ceiling_and_percentile_floor -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(grid): CenteredAutoGridFinder pitch bounds (percentile floor, centers-fit ceiling)"
```

---

## Task 3: Comb-response pitch with fundamental selection

**Files:**
- Modify: `src/phenotypic/grid/_centered_auto_grid_finder.py`
- Test: `tests/unit/grid/test_centered_auto_grid_finder.py`

- [ ] **Step 1: Write the failing test** (recovers fundamental from sparse layout w/ empty rows; rejects octave)

```python
def _lattice_points(p, cx, cy, R, C, occupied):
    """occupied: iterable of (i,j) cells. Returns x (cols/CC), y (rows/RR) arrays."""
    xs, ys = [], []
    for (i, j) in occupied:
        xs.append(cx + (j - (C - 1) / 2) * p)
        ys.append(cy + (i - (R - 1) / 2) * p)
    return np.array(xs, float), np.array(ys, float)


def test_estimate_pitch_recovers_fundamental_with_empty_rows():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    p_true, cx, cy = 404.0, W / 2, H / 2
    # occupy rows {1,2,3,5,6} (empty edge + interior rows), scattered columns
    occ = [(1, 0), (1, 4), (1, 8), (2, 3), (2, 7), (2, 11),
           (3, 1), (3, 5), (3, 9), (5, 0), (5, 4), (5, 8), (6, 0), (6, 6), (6, 11)]
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    p_min, p_max = f._compute_bounds(x, y, H, W)
    p0, ok = f._estimate_pitch(x, y, p_min, p_max)
    assert ok
    assert p0 == pytest.approx(p_true, abs=3.0)   # not the p/2=202 octave


def test_estimate_pitch_flags_degenerate_on_random_points():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    rng = np.random.default_rng(0)
    H, W = 3152, 5066
    x = rng.uniform(0, W, 40); y = rng.uniform(0, H, 40)
    p_min, p_max = f._compute_bounds(x, y, H, W)
    _, ok = f._estimate_pitch(x, y, p_min, p_max)
    assert ok is False  # no real periodicity -> caller will fall back
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py -k estimate_pitch -q`
Expected: FAIL — no attribute `_estimate_pitch`.

- [ ] **Step 3: Implement `_comb_mag` + `_estimate_pitch`**

```python
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
        # strict interior local maxima above the relative floor, choose the largest p
        idx = [i for i in range(1, len(ps) - 1)
               if Rr[i] > Rr[i - 1] and Rr[i] > Rr[i + 1] and Rr[i] >= self.response_floor * peak]
        if not idx:
            return float(ps[int(np.argmax(Rr))]), False
        p0 = float(ps[max(idx)])
        return p0, True
```

Note: `n_pitch_samples=512` over a wide `[p_min,p_max]` resolves the fundamental for plate pitches; golden-section sub-sample refinement is unnecessary because ICP (Task 5) refines `p` to sub-pixel from the seed.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py -k estimate_pitch -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(grid): comb-response pitch with fundamental selection"
```

---

## Task 4: Phase → integer center candidates over full in-frame offset

**Files:**
- Modify: `src/phenotypic/grid/_centered_auto_grid_finder.py`
- Test: `tests/unit/grid/test_centered_auto_grid_finder.py`

- [ ] **Step 1: Write the failing test** (the true center is among the candidates)

```python
def test_center_candidates_include_true_center():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    p_true, cx, cy = 404.0, 2545.0, 1575.0   # offset from image center (2533,1576)
    occ = [(1, 0), (2, 3), (3, 5), (5, 8), (6, 11), (3, 1), (1, 8), (5, 4)]
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    cx_c = f._center_candidates(x, p_true, f.ncols, W)
    cy_c = f._center_candidates(y, p_true, f.nrows, H)
    assert min(abs(np.array(cx_c) - cx)) < 0.5 * p_true
    assert min(abs(np.array(cy_c) - cy)) < 0.5 * p_true
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py::test_center_candidates_include_true_center -q`
Expected: FAIL — no attribute `_center_candidates`.

- [ ] **Step 3: Implement `_phase` + `_center_candidates`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py::test_center_candidates_include_true_center -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(grid): phase-based center candidates over in-frame offset"
```

---

## Task 5: Multi-start closed-form ICP with singularity guard + residual selection

**Files:**
- Modify: `src/phenotypic/grid/_centered_auto_grid_finder.py`
- Test: `tests/unit/grid/test_centered_auto_grid_finder.py`

- [ ] **Step 1: Write the failing tests** (recovers params; one-cell-shift seed rejected; singular guard)

```python
def test_icp_recovers_params_from_good_seed():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    p_true, cx, cy = 404.0, 2545.0, 1575.0
    occ = [(1, 0), (1, 4), (2, 3), (3, 5), (5, 8), (6, 11), (3, 1), (5, 4), (2, 7), (6, 0)]
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    out = f._icp_refine(x, y, cx + 30, cy - 25, p_true + 6)  # seed within ~0.1 cell
    assert out is not None
    rcx, rcy, rp, res = out
    assert rp == pytest.approx(p_true, abs=1.0)
    assert rcx == pytest.approx(cx, abs=2.0) and rcy == pytest.approx(cy, abs=2.0)
    assert res < 0.05 * p_true


def test_multistart_rejects_one_cell_shift():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    p_true, cx, cy = 404.0, 2545.0, 1575.0
    occ = [(1, 0), (1, 4), (2, 3), (3, 5), (5, 8), (6, 11), (3, 1), (5, 4), (2, 7), (6, 0)]
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    cx_c = f._center_candidates(x, p_true, f.ncols, W)
    cy_c = f._center_candidates(y, p_true, f.nrows, H)
    best = f._multi_start_refine(x, y, p_true, cx_c, cy_c)
    assert best is not None
    bcx, bcy, bp, res = best
    assert bcx == pytest.approx(cx, abs=2.0) and bcy == pytest.approx(cy, abs=2.0)


def test_icp_singular_returns_none_all_one_cell():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    # all points in a tiny cluster -> every assignment rounds to one cell -> det ~ 0
    x = np.array([2500.0, 2503.0, 2498.0]); y = np.array([1570.0, 1572.0, 1569.0])
    out = f._icp_refine(x, y, 2533.0, 1576.0, 404.0)
    assert out is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py -k "icp or multistart" -q`
Expected: FAIL — no attribute `_icp_refine`.

- [ ] **Step 3: Implement `_icp_refine` + `_multi_start_refine`**

```python
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
        """Run ICP from every (cx,cy) candidate; keep the lowest-residual result."""
        best = None
        for cx0 in cx_cands:
            for cy0 in cy_cands:
                out = self._icp_refine(x, y, cx0, cy0, p0)
                if out is None:
                    continue
                if best is None or out[3] < best[3]:
                    best = out
        return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py -k "icp or multistart" -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(grid): multi-start closed-form ICP with singularity guard"
```

---

## Task 6: Centers→edges + joint `_fit_grid` + wire the three overrides

**Files:**
- Modify: `src/phenotypic/grid/_centered_auto_grid_finder.py`
- Test: `tests/unit/grid/test_centered_auto_grid_finder.py`

- [ ] **Step 1: Write the failing test** (edge contract + end-to-end assignment on a synthetic Image)

```python
from phenotypic.data import load_synth_yeast_plate  # used in later integration test
from phenotypic.schema import GRID


def _fake_image_with_centers(x, y, H, W):
    """Minimal stand-in exercising get_row/col_edges via a synthetic objects table is
    heavy; instead test _fit_grid + _axis_edges directly on arrays."""
    return x, y, H, W


def test_axis_edges_length_sorted_clipped():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    edges = f._axis_edges(center=1575.0, p=404.0, n_cells=8, image_dim=3152)
    assert len(edges) == 9                  # n+1
    assert np.all(np.diff(edges) > 0)       # sorted ascending
    assert edges[0] >= 0 and edges[-1] <= 3152


def test_fit_grid_returns_edges_and_lands_on_lattice():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    p_true, cx, cy = 404.0, 2545.0, 1575.0
    occ = [(1, 0), (1, 4), (1, 8), (2, 3), (2, 7), (2, 11), (3, 1), (3, 5),
           (3, 9), (5, 0), (5, 4), (5, 8), (6, 0), (6, 6), (6, 11)]
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    row_edges, col_edges = f._fit_grid_from_centers(x, y, H, W)
    assert len(row_edges) == 9 and len(col_edges) == 13
    # every colony falls strictly inside the bin of its true cell
    for (i, j), xi, yi in zip(occ, x, y):
        assert row_edges[i] <= yi <= row_edges[i + 1]
        assert col_edges[j] <= xi <= col_edges[j + 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py -k "axis_edges or fit_grid" -q`
Expected: FAIL — no attribute `_axis_edges` / `_fit_grid_from_centers`.

- [ ] **Step 3: Implement `_axis_edges`, `_extract_centers`, `_fit_grid_from_centers`, and rewire overrides**

```python
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
```

(Delete the Task-1 stub `get_row_edges`/`get_col_edges`/`_operate`; keep `_uniform_edges`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py -k "axis_edges or fit_grid" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(grid): centers->edges and joint _fit_grid wiring"
```

---

## Task 7: Fallback ladder (degenerate tail)

**Files:**
- Modify: `src/phenotypic/grid/_centered_auto_grid_finder.py`
- Test: `tests/unit/grid/test_centered_auto_grid_finder.py`

- [ ] **Step 1: Write the failing tests** (0/1 colony; bound inversion; ICP-fail → no crash, valid edges)

```python
def _edges_ok(row_edges, col_edges, R, C, H, W):
    return (len(row_edges) == R + 1 and len(col_edges) == C + 1
            and np.all(np.diff(row_edges) > 0) and np.all(np.diff(col_edges) > 0)
            and row_edges[0] >= 0 and row_edges[-1] <= H
            and col_edges[0] >= 0 and col_edges[-1] <= W)


def test_zero_and_one_colony_uniform_no_crash():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    for x, y in [(np.array([]), np.array([])), (np.array([2500.0]), np.array([1570.0]))]:
        re, ce = f._fit_grid_from_centers(x, y, H, W)
        assert _edges_ok(re, ce, 8, 12, H, W)


def test_degenerate_response_falls_back_without_exception():
    f = CenteredAutoGridFinder(nrows=8, ncols=12, warn=True)
    rng = np.random.default_rng(1)
    H, W = 3152, 5066
    x = rng.uniform(0, W, 30); y = rng.uniform(0, H, 30)
    with pytest.warns(CenteredAutoGridFinderFallbackWarning):
        re, ce = f._fit_grid_from_centers(x, y, H, W)
    assert _edges_ok(re, ce, 8, 12, H, W)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py -k "colony or degenerate" -q`
Expected: FAIL (uniform-on-empty raises in `_compute_bounds`/`np.percentile` on empty, or no warning emitted).

- [ ] **Step 3: Implement the ladder inside `_fit_grid_from_centers`**

```python
    def _warn(self, msg: str) -> None:
        if self.warn:
            warnings.warn(f"CenteredAutoGridFinder {msg}",
                          CenteredAutoGridFinderFallbackWarning, stacklevel=2)

    def _centered_uniform(self, p: float, H: int, W: int):
        """Centered uniform grid at pitch p (image-centered)."""
        re = self._axis_edges(H / 2.0, p, self.nrows, H)
        ce = self._axis_edges(W / 2.0, p, self.ncols, W)
        return re, ce

    def _fit_grid_from_centers(self, x: np.ndarray, y: np.ndarray, H: int, W: int):
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

        cx_c = self._center_candidates(x, p0, self.ncols, W)
        cy_c = self._center_candidates(y, p0, self.nrows, H)
        best = self._multi_start_refine(x, y, p0, cx_c, cy_c)
        if best is None or best[3] > self.residual_fraction * best[2]:
            self._warn("[icp-failed] no acceptable registration; centered uniform at comb pitch.")
            return self._centered_uniform(p0, H, W)

        cx, cy, p, _res = best
        return self._axis_edges(cy, p, self.nrows, H), self._axis_edges(cx, p, self.ncols, W)
```

- [ ] **Step 4: Run the FULL new-file suite**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py -q`
Expected: PASS (all tasks 1–7 tests green).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(grid): CenteredAutoGridFinder fallback ladder"
```

---

## Task 8: Docstring doctests on the real synth plate

**Files:**
- Modify: `src/phenotypic/grid/_centered_auto_grid_finder.py`
- Test: doctest via pytest `--doctest-modules`

- [ ] **Step 1: Add two runnable doctests to the class docstring** (append inside the docstring, before the closing `"""`)

```python
    """
    ... (existing docstring above) ...

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
```

- [ ] **Step 2: Run the doctest**

Run: `uv run pytest --doctest-modules src/phenotypic/grid/_centered_auto_grid_finder.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "docs(grid): CenteredAutoGridFinder doctests"
```

---

## Task 9: Make it the default GridFinder for grid images

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_grid_image_handler.py:~97`
- Modify: `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py:~1136-1144`
- Test: `tests/unit/grid/test_centered_auto_grid_finder.py` (+ fallout sweep across the suite)

- [ ] **Step 1: Write the failing test**

```python
def test_grid_image_default_finder_is_centered():
    import numpy as np
    from phenotypic import GridImage
    from phenotypic.grid import CenteredAutoGridFinder
    img = GridImage(arr=np.zeros((400, 600, 3), dtype=np.uint8), nrows=8, ncols=12)
    assert isinstance(img.grid_finder, CenteredAutoGridFinder)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py::test_grid_image_default_finder_is_centered -q`
Expected: FAIL — default is still `AutoGridFinder`.

- [ ] **Step 3: Flip the default in `_grid_image_handler.py`**

Change the import + the `grid_finder is None` branch:

```python
# top of file: add
from phenotypic.grid import CenteredAutoGridFinder
# (keep the existing AutoGridFinder import if present; it stays available)

# in GridImageHandler.__init__, the default branch (~L96-97):
        elif grid_finder is None:
            grid_finder = CenteredAutoGridFinder(nrows=nrows, ncols=ncols)
```

- [ ] **Step 4: Flip the injected default in `_image_pipeline_core.py`**

```python
# add import near the other finder import
from phenotypic.grid import CenteredAutoGridFinder

# the injection block (~L1136-1144) becomes:
        injected_key = (
            "CenteredAutoGridFinder"
            if "CenteredAutoGridFinder" not in self._meas
            else "_CenteredAutoGridFinder_preset"
        )
        run_order: Dict[str, MeasureFeatures] = {
            injected_key: CenteredAutoGridFinder(nrows=self._nrows, ncols=self._ncols),
        }
```

- [ ] **Step 5: Run the targeted test**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py::test_grid_image_default_finder_is_centered -q`
Expected: PASS.

- [ ] **Step 6: Fallout sweep — find and fix tests assuming the old default**

```bash
grep -rn "AutoGridFinder" tests | grep -v "Centered"
```

Run the broad suites and fix each failure (update default-type assertions to `CenteredAutoGridFinder`; for grid-output snapshots/goldens, regenerate only where the change is the intended default flip — confirm the new grid is correct before accepting):

Run: `uv run pytest tests/unit/grid tests/unit/_core -q`
Expected: PASS after fixes. Record any golden regenerations in the commit message.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(grid): default GridImage finder -> CenteredAutoGridFinder (+test fallout)"
```

---

## Task 10: Annotation-coverage gate + integration + dense-plate regression

**Files:**
- Test: `tests/unit/grid/test_centered_auto_grid_finder.py`
- Verify: `tests/unit/tune/test_annotation_coverage.py` (no code change expected — just must pass)

- [ ] **Step 1: Annotation-coverage gate passes**

The numeric fields (`residual_fraction`, `n_pitch_samples`, `response_floor`, `max_iter`, `min_fit_objects`) all carry `TuneSpec(...)` annotations (Task 1). Confirm the gate sees them:

Run: `uv run pytest tests/unit/tune/test_annotation_coverage.py -q`
Expected: PASS. If it flags a field, add the intended `TuneSpec(...)`/`TuneSpec(tunable=False)` per the spec §8 (do **not** silence by deleting the field).

- [ ] **Step 2: Write the integration + dense-plate regression test**

```python
def test_integration_decimated_synth_plate():
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    image = OtsuDetector().apply(load_synth_yeast_plate())
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    df = f.measure(image)
    assert str(GRID.ROW_NUM) in df.columns and str(GRID.COL_NUM) in df.columns
    # rows/cols within range, no NaN section for detected objects
    assert df[str(GRID.ROW_NUM)].dropna().between(0, 7).all()
    assert df[str(GRID.COL_NUM)].dropna().between(0, 11).all()


def test_dense_plate_pitch_above_naive_ceiling_is_found():
    """Regression for the bound-inversion: true pitch exceeds min(H/R,W/C)."""
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    p_true, cx, cy = 404.0, W / 2, H / 2          # 404 > min(H/8,W/12)=394
    occ = [(i, j) for i in range(8) for j in range(12)]   # fully dense
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    re, ce = f._fit_grid_from_centers(x, y, H, W)
    # recovered column pitch ~ 404 (not capped at 394)
    assert np.mean(np.diff(ce[1:-1])) == pytest.approx(p_true, abs=2.0)
```

- [ ] **Step 3: Run the integration tests**

Run: `uv run pytest tests/unit/grid/test_centered_auto_grid_finder.py -k "integration or dense_plate" -q`
Expected: PASS.

- [ ] **Step 4: Full regression + lint/type**

Run:
```bash
uv run pytest tests/unit/grid tests/unit/_core tests/unit/tune -q
uv run ruff check --fix src/phenotypic/grid/_centered_auto_grid_finder.py
uv run mypy src/phenotypic/grid/_centered_auto_grid_finder.py
```
Expected: tests PASS; ruff clean; mypy clean (add precise types if it complains).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "test(grid): integration + dense-plate regression for CenteredAutoGridFinder"
```

---

## Out of scope (tracked separately)

- GridAligner sparse-plate robustness (spec risk #7) — assume de-rotated input.
- GUI builder registration / bespoke dashboard — follow-up if adopted.
- The 127 MB `SaltTolerantSparsePlate.png` is **not** committed; the real-plate validation is recorded in spec §13 and reproduced by `/tmp/grid_exp/full_fit.py`. The committed tests use synthetic lattices + `load_synth_yeast_plate()`.
