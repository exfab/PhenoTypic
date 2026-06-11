# Robust colorimetric measures for `MeasureColor` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `MeasureColor`'s per-channel "kitchen-sink" statistics for L\*a\*b\* and HSV with compact, robust, multivariate colorimetric summaries (ΔE76 geometric median + ΔE2000 medoid centers, ΔE2000 consistency scalars, total variance, cone-embedded robust HSV, and a Lab-medoid hex swatch), and demote CIE XYZ / xy chromaticity to opt-in columns hidden from the reference doc.

**Architecture:** Pure, unit-testable math lives in a new `util/_robust_color_stats.py`. The robust center **reuses the existing, verified `phenotypic.util.geometric_median`** (pinned to `method='weiszfeld'`); the rest (ΔE2000 medoid, ΔE2000 spread, HSV cone transform, Lab→hex) are new pure functions. `MeasureColor._operate` extracts per-object L\*a\*b\* and HSV pixel vectors in one pass and calls those helpers; the legacy 8-stat suites survive only for opt-in XYZ/xy. New columns are declared as `MeasurementInfo` members in `schema/_color_lab.py` and `schema/_color_hsv.py`, which automatically flow into `get_headers()` and the generated reference table.

**Reuse note (verified):** `phenotypic.util.geometric_median(points, method='weiszfeld', eps=tol, max_iter=..., verbose=False) -> (median, info)` is correct on all our cases (symmetric cloud→origin, single-outlier resistance, collinear→middle, n=2 midpoint, identical points via zero-distance guard, n=1). **Caveat:** its *default* `method='cohen'` is unimplemented and raises `ValueError("Method 'cohen' is not implemented yet.")`, and it raises `ValueError("Need at least one point")` on empty input. Therefore our wrapper ALWAYS passes `method='weiszfeld'` and guards empty/`n==1` itself. (Pre-existing `cohen` default bug is out of scope — noted, not fixed here.)

**Tech Stack:** Python, NumPy, pandas, `colour-science` (`colour.difference.delta_E_CIE2000`, `colour.Lab_to_XYZ`, `colour.XYZ_to_sRGB`), pydantic v2 (operation fields), pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-06-10-robust-lab-color-measures-design.md`

**Conventions for every task below:**
- Run everything via `uv run` (never bare `python`/`pip`).
- Run pytest with a Qt binding present and offscreen: prefix with `QT_QPA_PLATFORM=offscreen`.
- Work only inside the worktree `/Users/alex/Projects/PhenoTypic/.claude/worktrees/robust-lab-color-measures` (use worktree-relative paths; do NOT `cd` to the main repo).
- Doctests/tests load images via `from phenotypic.data import load_synth_yeast_plate`.

---

## File Structure

**Create:**
- `src/phenotypic/util/_robust_color_stats.py` — pure functions: `robust_color_center` (thin wrapper reusing `util.geometric_median`, weiszfeld), `medoid_ciede2000`, `delta_e2000_spread`, `hsv_to_cone`, `cone_to_hsv`, `lab_to_srgb_hex`.
- `tests/unit/util/test_robust_color_stats.py` — unit tests for the pure helpers.
- `tests/unit/measure/test_measure_color.py` — behavior tests for the rewritten `MeasureColor` (no dedicated file exists today).

**Modify:**
- `src/phenotypic/util/__init__.py` — export the new helpers.
- `src/phenotypic/schema/_color_lab.py` — drop per-channel 8-stat members + `ChromaEstimated*`; add robust members + `robust_headers()`.
- `src/phenotypic/schema/_color_hsv.py` — drop per-channel 8-stat members; add robust members + `robust_headers()`.
- `src/phenotypic/measure/_measure_color.py` — new `_operate`, `include_xy` field, robust per-object helpers, opt-in XYZ/xy, updated docstring RST chain.
- `docs/source/_extensions/measurements_ref.py` — drop `ColorXYZ`/`Colorxy` from the `MeasureColor` doc entry.
- `src/phenotypic/_cli/_cli_readme_generator.py` — update color schema mapping (~lines 196, 213).
- `docs/source/explanation/measurement_metrics_biological_meaning.md` — update Color Metrics prose.
- `tests/migration/_scenarios.py` — add `with_xy` curated extra; keep `with_xyz`.
- `tests/migration/_goldens/measure.MeasureColor.parquet`, `…with_xyz.parquet` (+ new `…with_xy.parquet`) — regenerate (targeted).
- `tests/unit/cli/test_cli_output_manager.py`, `tests/unit/util/test_measurement_outputs.py` — update any references to removed columns.

**Leave intact:** `schema/_color_xyz.py`, `schema/_color_xy.py` (used by the opt-in paths).

---

## Task 1: Util helper module — `robust_color_center` (reuses `util.geometric_median`)

Do NOT reimplement Weiszfeld. Reuse the existing, verified
`phenotypic.util.geometric_median` (pinned to `method='weiszfeld'`) behind a thin
wrapper that adds empty/`n==1` guards and returns just the array.

**Files:**
- Create: `src/phenotypic/util/_robust_color_stats.py`
- Modify: `src/phenotypic/util/__init__.py`
- Test: `tests/unit/util/test_robust_color_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/util/test_robust_color_stats.py
import numpy as np
import pytest
from phenotypic.util._robust_color_stats import robust_color_center


def test_robust_center_symmetric_cloud():
    pts = np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    assert np.allclose(robust_color_center(pts), [0.0, 0.0, 0.0], atol=1e-3)


def test_robust_center_resists_single_outlier():
    cluster = np.tile([50.0, 10.0, 20.0], (99, 1))
    pts = np.vstack([cluster, [50_000.0, 50_000.0, 50_000.0]])
    assert np.allclose(robust_color_center(pts), [50.0, 10.0, 20.0], atol=1.0)


def test_robust_center_single_point_returns_it():
    assert np.allclose(robust_color_center(np.array([[3.0, 4.0, 5.0]])), [3.0, 4.0, 5.0])


def test_robust_center_identical_points():
    assert np.allclose(robust_color_center(np.tile([7.0, 7.0, 7.0], (5, 1))), [7.0, 7.0, 7.0])


def test_robust_center_empty_returns_nan():
    out = robust_color_center(np.empty((0, 3)))
    assert out.shape == (3,) and np.isnan(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/util/test_robust_color_stats.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.util._robust_color_stats'`.

- [ ] **Step 3: Write the module**

```python
# src/phenotypic/util/_robust_color_stats.py
"""Pure, unit-testable robust colorimetric estimators used by MeasureColor.

Free of Image/accessor dependencies so they can be tested in isolation. The
robust center reuses the verified ``phenotypic.util.geometric_median`` (the
``cohen`` method is unimplemented, so we always pin ``method='weiszfeld'``).
See docs/superpowers/specs/2026-06-10-robust-lab-color-measures-design.md.
"""
from __future__ import annotations

import numpy as np

from phenotypic.util._geometric_median import geometric_median as _geometric_median


def robust_color_center(
    points: np.ndarray, max_iter: int = 50, tol: float = 1e-4
) -> np.ndarray:
    """Euclidean geometric median of ``points`` (N, D), as a bare (D,) array.

    Reuses ``phenotypic.util.geometric_median`` (Weiszfeld). Returns all-NaN for
    empty input and the sole point for ``N == 1`` (the underlying solver requires
    ``N >= 1`` and a defined centroid).

    Args:
        points: (N, D) coordinates (Lab pixels, or HSV cone coordinates).
        max_iter: Weiszfeld iteration cap.
        tol: Convergence tolerance (forwarded as ``eps``).

    Returns:
        (D,) geometric-median coordinate.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must be 2-D (N, D)")
    n, d = points.shape
    if n == 0:
        return np.full(d, np.nan)
    if n == 1:
        return points[0].copy()
    center, _info = _geometric_median(
        points, method="weiszfeld", eps=tol, max_iter=max_iter, verbose=False
    )
    return np.asarray(center, dtype=np.float64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/util/test_robust_color_stats.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Export from the util package**

Add to `src/phenotypic/util/__init__.py` (extend the existing imports and
`__all__`):

```python
from ._robust_color_stats import (
    robust_color_center,
    medoid_ciede2000,
    delta_e2000_spread,
    hsv_to_cone,
    cone_to_hsv,
    lab_to_srgb_hex,
)
```

and add those six names to the `__all__` list. (The `medoid_ciede2000` etc.
symbols are added in Tasks 2–3; importing them now will fail, so either add this
export block at the END of Task 3, or add names incrementally as each is
implemented. Simplest: do the `__init__.py` export edit as the final step of
Task 3, not here. For Task 1, only export `robust_color_center`.)

For Task 1, add just:

```python
from ._robust_color_stats import robust_color_center
```
and `"robust_color_center"` to `__all__`.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/util/_robust_color_stats.py src/phenotypic/util/__init__.py tests/unit/util/test_robust_color_stats.py
git commit -m "feat(util): robust_color_center reusing geometric_median (weiszfeld)"
```

---

## Task 2: Pure helpers — `medoid_ciede2000` + `delta_e2000_spread`

**Files:**
- Modify: `src/phenotypic/util/_robust_color_stats.py`
- Test: `tests/unit/util/test_robust_color_stats.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/util/test_robust_color_stats.py
from phenotypic.util._robust_color_stats import (
    medoid_ciede2000,
    delta_e2000_spread,
)


def test_medoid_is_an_actual_input_pixel():
    rng = np.random.default_rng(0)
    lab = rng.uniform([20, -10, -10], [80, 40, 40], size=(50, 3))
    center, deltas = medoid_ciede2000(lab, max_pixels=1000, seed=0)
    assert any(np.allclose(center, p) for p in lab)  # center IS a real pixel
    assert deltas.shape == (50,)
    assert np.all(deltas >= 0)


def test_medoid_central_for_one_outlier():
    cluster = np.tile([50.0, 10.0, 20.0], (40, 1))
    lab = np.vstack([cluster, [10.0, -50.0, 60.0]])
    center, _ = medoid_ciede2000(lab, max_pixels=1000, seed=0)
    assert np.allclose(center, [50.0, 10.0, 20.0])


def test_medoid_subsample_is_reproducible():
    rng = np.random.default_rng(1)
    lab = rng.uniform([20, -10, -10], [80, 40, 40], size=(5000, 3))
    c1, d1 = medoid_ciede2000(lab, max_pixels=500, seed=7)
    c2, d2 = medoid_ciede2000(lab, max_pixels=500, seed=7)
    assert np.allclose(c1, c2)
    assert d1.shape == (5000,)  # spread uses ALL pixels, not the subsample


def test_medoid_single_and_empty():
    c, d = medoid_ciede2000(np.array([[40.0, 5.0, -5.0]]))
    assert np.allclose(c, [40.0, 5.0, -5.0]) and np.allclose(d, [0.0])
    c0, d0 = medoid_ciede2000(np.empty((0, 3)))
    assert np.isnan(c0).all() and d0.size == 0


def test_delta_e2000_spread_values():
    deltas = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    med, mean, p95 = delta_e2000_spread(deltas)
    assert med == 2.0
    assert mean == pytest.approx(21.2)
    assert p95 == pytest.approx(np.percentile(deltas, 95))


def test_delta_e2000_spread_empty_is_nan():
    med, mean, p95 = delta_e2000_spread(np.array([]))
    assert np.isnan(med) and np.isnan(mean) and np.isnan(p95)
```

- [ ] **Step 2: Run to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/util/test_robust_color_stats.py -q`
Expected: FAIL — `ImportError: cannot import name 'medoid_ciede2000'`.

- [ ] **Step 3: Add the implementation**

```python
# add to src/phenotypic/util/_robust_color_stats.py (module-level import + funcs)
import colour


def medoid_ciede2000(
    lab_points: np.ndarray, max_pixels: int = 1000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """ΔE2000 medoid center and per-pixel ΔE2000 distances to it.

    The medoid (real pixel minimizing total ΔE2000) is selected from a seeded
    subsample of at most ``max_pixels`` (the selection is O(m^2)); the returned
    distances are computed from the chosen medoid to **all** input pixels.

    Args:
        lab_points: (N, 3) CIE L*a*b* pixel vectors.
        max_pixels: Subsample cap for medoid selection.
        seed: RNG seed for reproducible subsampling.

    Returns:
        (center (3,), all_deltas (N,)). center is all-NaN and all_deltas empty
        when ``lab_points`` is empty.
    """
    lab = np.asarray(lab_points, dtype=np.float64)
    n = lab.shape[0]
    if n == 0:
        return np.full(3, np.nan), np.empty(0)
    if n == 1:
        return lab[0].copy(), np.zeros(1)

    if n > max_pixels:
        rng = np.random.default_rng(seed)
        sample = lab[rng.choice(n, size=max_pixels, replace=False)]
    else:
        sample = lab

    pairwise = colour.difference.delta_E_CIE2000(
        sample[:, None, :], sample[None, :, :]
    )
    medoid = sample[pairwise.sum(axis=1).argmin()]
    all_deltas = np.asarray(colour.difference.delta_E_CIE2000(lab, medoid))
    return medoid, all_deltas


def delta_e2000_spread(deltas: np.ndarray) -> tuple[float, float, float]:
    """Return (median, mean, P95) of a ΔE2000 distance array; NaNs if empty."""
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(deltas)),
        float(np.mean(deltas)),
        float(np.percentile(deltas, 95)),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/util/test_robust_color_stats.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/util/_robust_color_stats.py tests/unit/util/test_robust_color_stats.py
git commit -m "feat(util): add ciede2000 medoid + spread helpers"
```

---

## Task 3: Pure helpers — HSV cone transform + Lab→hex

**Files:**
- Modify: `src/phenotypic/util/_robust_color_stats.py`
- Modify: `src/phenotypic/util/__init__.py` (export all six helpers)
- Test: `tests/unit/util/test_robust_color_stats.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/util/test_robust_color_stats.py
from phenotypic.util._robust_color_stats import (
    hsv_to_cone,
    cone_to_hsv,
    lab_to_srgb_hex,
)


def test_cone_roundtrip_recovers_hsv():
    hsv = np.array([[0.0, 1.0, 1.0], [0.25, 0.5, 0.8], [0.99, 0.3, 0.6]])
    back = cone_to_hsv(hsv_to_cone(hsv))
    assert np.allclose(back, hsv, atol=1e-6)


def test_cone_collapses_unreliable_hue_at_zero_saturation():
    # Two grays with different (meaningless) hues map to the same cone point.
    a = hsv_to_cone(np.array([0.1, 0.0, 0.5]))
    b = hsv_to_cone(np.array([0.7, 0.0, 0.5]))
    assert np.allclose(a, b)


def test_cone_handles_hue_wraparound():
    # Hues near 0 and near 1 are adjacent, not opposite.
    near_zero = hsv_to_cone(np.array([0.001, 1.0, 1.0]))
    near_one = hsv_to_cone(np.array([0.999, 1.0, 1.0]))
    assert np.linalg.norm(near_zero - near_one) < 0.05


def test_lab_to_srgb_hex_format():
    h = lab_to_srgb_hex(np.array([60.0, 20.0, 30.0]))
    assert h == "#c1825d"


def test_lab_to_srgb_hex_nan_returns_empty():
    assert lab_to_srgb_hex(np.array([np.nan, 0.0, 0.0])) == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/util/test_robust_color_stats.py -q`
Expected: FAIL — `ImportError: cannot import name 'hsv_to_cone'`.

- [ ] **Step 3: Add the implementation**

```python
# add to src/phenotypic/util/_robust_color_stats.py
def hsv_to_cone(hsv: np.ndarray) -> np.ndarray:
    """Embed HSV (H,S,V in [0,1]) into Cartesian cone coords (S*V*cosθ, S*V*sinθ, V)."""
    hsv = np.asarray(hsv, dtype=np.float64)
    theta = 2.0 * np.pi * hsv[..., 0]
    chroma = hsv[..., 1] * hsv[..., 2]
    x = chroma * np.cos(theta)
    y = chroma * np.sin(theta)
    z = hsv[..., 2]
    return np.stack([x, y, z], axis=-1)


def cone_to_hsv(cone: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hsv_to_cone`; returns H,S,V in [0,1]."""
    cone = np.asarray(cone, dtype=np.float64)
    x, y, z = cone[..., 0], cone[..., 1], cone[..., 2]
    hue = (np.arctan2(y, x) / (2.0 * np.pi)) % 1.0
    chroma = np.sqrt(x * x + y * y)
    value = z
    sat = np.where(value > _EPS, np.clip(chroma / np.where(value > _EPS, value, 1.0), 0.0, 1.0), 0.0)
    return np.stack([hue, sat, value], axis=-1)


def lab_to_srgb_hex(lab: np.ndarray) -> str:
    """Convert a single CIE L*a*b* (D65) color to an sRGB ``#RRGGBB`` string.

    Returns ``""`` if any coordinate is NaN (e.g. an empty object).
    """
    lab = np.asarray(lab, dtype=np.float64)
    if np.isnan(lab).any():
        return ""
    xyz = colour.Lab_to_XYZ(lab)
    srgb = np.clip(colour.XYZ_to_sRGB(xyz), 0.0, 1.0)
    r, g, b = (np.round(srgb * 255.0).astype(int))
    return f"#{r:02x}{g:02x}{b:02x}"
```

- [ ] **Step 4: Run to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/util/test_robust_color_stats.py -q`
Expected: PASS (15 passed). If `test_lab_to_srgb_hex_format` asserts a different hex than your `colour` version yields, update the expected value to the printed actual (compute once with `QT_QPA_PLATFORM=offscreen uv run python -c "import numpy as np; from phenotypic.util._robust_color_stats import lab_to_srgb_hex; print(lab_to_srgb_hex(np.array([60.,20.,30.])))"`). (Verified value at plan-writing time: `#c1825d`.)

- [ ] **Step 5: Export all six helpers from the util package**

Replace the Task-1 single-symbol export in `src/phenotypic/util/__init__.py` with
the full block, and ensure all six names are in `__all__`:

```python
from ._robust_color_stats import (
    robust_color_center,
    medoid_ciede2000,
    delta_e2000_spread,
    hsv_to_cone,
    cone_to_hsv,
    lab_to_srgb_hex,
)
```

Verify the package imports cleanly:
Run: `QT_QPA_PLATFORM=offscreen uv run python -c "from phenotypic.util import robust_color_center, medoid_ciede2000, delta_e2000_spread, hsv_to_cone, cone_to_hsv, lab_to_srgb_hex; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/util/_robust_color_stats.py src/phenotypic/util/__init__.py tests/unit/util/test_robust_color_stats.py
git commit -m "feat(util): add HSV cone transform + Lab->sRGB hex helpers; export all"
```

---

## Task 4: Schema — rewrite `ColorLab`

**Files:**
- Modify: `src/phenotypic/schema/_color_lab.py`
- Test: new `tests/unit/schema/test_color_schema.py` (schema assertions; create the `tests/unit/schema/` dir + empty `__init__.py` if absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/schema/test_color_schema.py
from phenotypic.schema import ColorLab, ColorHSV


def test_colorlab_has_robust_headers_and_no_legacy_suite():
    headers = ColorLab.get_headers()
    # New robust columns present (prefixed with category "ColorLab_").
    for expected in [
        "ColorLab_L*GeoMedian", "ColorLab_a*GeoMedian", "ColorLab_b*GeoMedian",
        "ColorLab_L*Medoid", "ColorLab_a*Medoid", "ColorLab_b*Medoid",
        "ColorLab_DeltaE2000MedianFromMedoid",
        "ColorLab_DeltaE2000MeanFromMedoid",
        "ColorLab_DeltaE2000P95FromMedoid",
        "ColorLab_LabTotalVariance",
        "ColorLab_MedoidColorHex",
    ]:
        assert expected in headers
    # Legacy per-channel + chroma columns gone.
    assert not any("Mean" in h and "Robust" not in h and "Geo" not in h for h in headers)
    assert not any("ChromaEstimated" in h for h in headers)
    assert len(ColorLab.robust_headers()) == 11
```

- [ ] **Step 2: Run to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/schema/test_color_schema.py -q`
Expected: FAIL — `ColorLab` still has legacy members and no `robust_headers`.

- [ ] **Step 3: Replace the body of `src/phenotypic/schema/_color_lab.py`**

```python
"""Per-object robust colorimetric statistics in the CIE L*a*b* color space."""

from ._measurement_info import MeasurementInfo


class ColorLab(MeasurementInfo):
    """Robust CIE L*a*b* colorimetric summary for a colony.

    Reports two robust center colors -- the ΔE76 (Euclidean) geometric median
    and the ΔE2000 medoid -- plus ΔE2000 within-colony consistency scalars, the
    total Euclidean color variance, and an sRGB hex swatch (plot-only) derived
    from the medoid.
    """

    @classmethod
    def category(cls):
        return "ColorLab"

    # -- ΔE76 geometric-median center (continuous, 0.5 breakdown) --
    L_STAR_GEOMEDIAN = ("L*GeoMedian", "L* of the ΔE76 (Euclidean) geometric-median center color of the object")
    A_STAR_GEOMEDIAN = ("a*GeoMedian", "a* of the ΔE76 (Euclidean) geometric-median center color of the object")
    B_STAR_GEOMEDIAN = ("b*GeoMedian", "b* of the ΔE76 (Euclidean) geometric-median center color of the object")

    # -- ΔE2000 medoid center (real pixel, perceptually-corrected) --
    L_STAR_MEDOID = ("L*Medoid", "L* of the ΔE2000 medoid center color (real pixel minimizing total ΔE2000)")
    A_STAR_MEDOID = ("a*Medoid", "a* of the ΔE2000 medoid center color (real pixel minimizing total ΔE2000)")
    B_STAR_MEDOID = ("b*Medoid", "b* of the ΔE2000 medoid center color (real pixel minimizing total ΔE2000)")

    # -- ΔE2000 within-colony consistency, measured from the medoid --
    DELTA_E2000_MEDIAN = ("DeltaE2000MedianFromMedoid", "Median ΔE2000 of object pixels from the ΔE2000 medoid center (robust perceptual MAD)")
    DELTA_E2000_MEAN = ("DeltaE2000MeanFromMedoid", "Mean ΔE2000 of object pixels from the ΔE2000 medoid center (color-uniformity standard)")
    DELTA_E2000_P95 = ("DeltaE2000P95FromMedoid", "95th-percentile ΔE2000 of object pixels from the ΔE2000 medoid center (worst-case / sectoring flag)")

    # -- classical Euclidean spread --
    LAB_TOTAL_VARIANCE = ("LabTotalVariance", "Trace of the 3x3 L*a*b* covariance (var L* + var a* + var b*); mean-squared ΔE76 about the arithmetic mean")

    # -- plot-only swatch --
    MEDOID_COLOR_HEX = ("MedoidColorHex", "sRGB hex string of the ΔE2000 medoid color; for plot visualization only (not a numeric measurement)")

    @classmethod
    def robust_headers(cls):
        return [
            str(cls.L_STAR_GEOMEDIAN),
            str(cls.A_STAR_GEOMEDIAN),
            str(cls.B_STAR_GEOMEDIAN),
            str(cls.L_STAR_MEDOID),
            str(cls.A_STAR_MEDOID),
            str(cls.B_STAR_MEDOID),
            str(cls.DELTA_E2000_MEDIAN),
            str(cls.DELTA_E2000_MEAN),
            str(cls.DELTA_E2000_P95),
            str(cls.LAB_TOTAL_VARIANCE),
            str(cls.MEDOID_COLOR_HEX),
        ]
```

- [ ] **Step 4: Run to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/schema/test_color_schema.py::test_colorlab_has_robust_headers_and_no_legacy_suite -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/schema/_color_lab.py tests/unit/schema/test_color_schema.py
git commit -m "feat(schema): robust ColorLab columns; drop per-channel suite + chroma"
```

---

## Task 5: Schema — rewrite `ColorHSV`

**Files:**
- Modify: `src/phenotypic/schema/_color_hsv.py`
- Test: `tests/unit/schema/test_color_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/schema/test_color_schema.py
def test_colorhsv_has_robust_headers_and_no_legacy_suite():
    headers = ColorHSV.get_headers()
    for expected in [
        "ColorHSV_HueRobustMean",
        "ColorHSV_SaturationRobustMean",
        "ColorHSV_ValueRobustMean",
        "ColorHSV_HSVConeVariance",
    ]:
        assert expected in headers
    assert not any(h.endswith("Min") or h.endswith("Max") or h.endswith("Q1") for h in headers)
    assert len(ColorHSV.robust_headers()) == 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/schema/test_color_schema.py::test_colorhsv_has_robust_headers_and_no_legacy_suite -q`
Expected: FAIL.

- [ ] **Step 3: Replace the body of `src/phenotypic/schema/_color_hsv.py`**

```python
"""Per-object robust summary in the HSV color space (cone-embedded)."""

from ._measurement_info import MeasurementInfo


class ColorHSV(MeasurementInfo):
    """Robust HSV summary for a colony.

    HSV hue is circular and HSV is not perceptually uniform, so the robust
    center is computed as the geometric median of cone-Cartesian coordinates
    (S*V*cosθ, S*V*sinθ, V) and converted back to H,S,V. ``HSVConeVariance`` is
    the trace of the cone-Cartesian covariance.
    """

    @classmethod
    def category(cls):
        return "ColorHSV"

    HUE_ROBUST_MEAN = ("HueRobustMean", "Hue of the cone-embedded geometric-median robust center (circular-correct)")
    SATURATION_ROBUST_MEAN = ("SaturationRobustMean", "Saturation of the cone-embedded geometric-median robust center")
    VALUE_ROBUST_MEAN = ("ValueRobustMean", "Value (brightness) of the cone-embedded geometric-median robust center")
    HSV_CONE_VARIANCE = ("HSVConeVariance", "Trace of the HSV cone-Cartesian covariance (single 3D HSV spread scalar)")

    @classmethod
    def robust_headers(cls):
        return [
            str(cls.HUE_ROBUST_MEAN),
            str(cls.SATURATION_ROBUST_MEAN),
            str(cls.VALUE_ROBUST_MEAN),
            str(cls.HSV_CONE_VARIANCE),
        ]
```

- [ ] **Step 4: Run to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/schema/test_color_schema.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/schema/_color_hsv.py tests/unit/schema/test_color_schema.py
git commit -m "feat(schema): robust ColorHSV columns; drop per-channel suite"
```

---

## Task 6: Rewrite `MeasureColor._operate`

**Files:**
- Modify: `src/phenotypic/measure/_measure_color.py`
- Test: `tests/unit/measure/test_measure_color.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/measure/test_measure_color.py
import numpy as np
import pandas as pd
import pytest

from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureColor
from phenotypic.schema import OBJECT, ColorLab, ColorHSV


@pytest.fixture(scope="module")
def detected_image():
    img = load_synth_yeast_plate()
    return OtsuDetector().operate(img)


def test_default_output_is_robust_only(detected_image):
    df = MeasureColor().measure(detected_image)
    cols = set(df.columns)
    # robust Lab + HSV present
    assert set(ColorLab.robust_headers()).issubset(cols)
    assert set(ColorHSV.robust_headers()).issubset(cols)
    # XYZ/xy absent by default
    assert not any(c.startswith("ColorXYZ_") for c in cols)
    assert not any(c.startswith("Colorxy_") for c in cols)
    # one row per object
    assert len(df) == detected_image.num_objects


def test_hex_column_is_string(detected_image):
    df = MeasureColor().measure(detected_image)
    hexcol = df[str(ColorLab.MEDOID_COLOR_HEX)]
    assert hexcol.dtype == object
    assert hexcol.iloc[0].startswith("#") and len(hexcol.iloc[0]) == 7


def test_deltae_scalars_nonnegative(detected_image):
    df = MeasureColor().measure(detected_image)
    for col in [ColorLab.DELTA_E2000_MEDIAN, ColorLab.DELTA_E2000_MEAN, ColorLab.DELTA_E2000_P95]:
        vals = df[str(col)].to_numpy()
        assert np.all(vals[~np.isnan(vals)] >= 0)


def test_opt_in_xyz_and_xy(detected_image):
    df = MeasureColor(include_XYZ=True, include_xy=True).measure(detected_image)
    assert any(c.startswith("ColorXYZ_") for c in df.columns)
    assert any(c.startswith("Colorxy_") for c in df.columns)


def test_serialization_roundtrip(detected_image):
    op = MeasureColor(medoid_max_pixels=300, random_seed=3)
    restored = MeasureColor.from_json(op.to_json())
    assert restored.medoid_max_pixels == 300
    assert restored.random_seed == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/measure/test_measure_color.py -q`
Expected: FAIL — new columns/fields not present yet.

- [ ] **Step 3: Rewrite `src/phenotypic/measure/_measure_color.py`**

Replace the class body. Keep `_compute_color_metrics` (used by the opt-in XYZ/xy paths). Add the `include_xy` field, the robust per-object helpers, and rebuild `_operate`. The XYZ/xy blocks are the *existing* code, now guarded by their flags.

```python
from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd
import logging

from phenotypic.abc_ import MeasureFeatures
from phenotypic.schema import OBJECT
from phenotypic.schema import ColorXYZ, Colorxy, ColorLab, ColorHSV
from phenotypic.util import (
    robust_color_center,
    medoid_ciede2000,
    delta_e2000_spread,
    hsv_to_cone,
    cone_to_hsv,
    lab_to_srgb_hex,
)

logger = logging.getLogger(__name__)


class MeasureColor(MeasureFeatures):
    """Measure robust colorimetric statistics for each colony.

    Default output (always on):

    - **CIE L*a*b*** -- ΔE76 geometric-median center, ΔE2000 medoid center,
      ΔE2000 within-colony consistency (median/mean/P95 from the medoid),
      ``LabTotalVariance``, and an sRGB hex swatch (plot-only).
    - **HSV** -- a cone-embedded robust center (circular-correct) and
      ``HSVConeVariance``.

    Opt-in, hidden from the reference doc:

    - **CIE XYZ** (``include_XYZ=True``) and **xy chromaticity**
      (``include_xy=True``) -- legacy per-channel min/Q1/mean/median/Q3/max/
      stddev/CoeffVar suites.

    Args:
        include_XYZ: Emit the legacy CIE XYZ per-channel suite. Default ``False``.
        include_xy: Emit the legacy xy chromaticity per-channel suite. Default
            ``False``.
        geomedian_max_iter: Weiszfeld iteration cap for the L*a*b* geometric
            median. Default ``50``.
        geomedian_tol: Weiszfeld convergence tolerance. Default ``1e-4``.
        medoid_max_pixels: Subsample cap for the O(N^2) ΔE2000 medoid selection;
            consistency scalars still use all pixels. Default ``1000``.
        random_seed: Seed for reproducible medoid subsampling. Default ``0``.
    """

    _measurement_infoclasses: ClassVar[list[type]] = [
        ColorXYZ, Colorxy, ColorLab, ColorHSV]

    include_XYZ: bool = False
    include_xy: bool = False
    geomedian_max_iter: int = 50
    geomedian_tol: float = 1e-4
    medoid_max_pixels: int = 1000
    random_seed: int = 0

    def _operate(self, image: Image):
        objmap = image.objmap[:]
        data = {OBJECT.LABEL: image.objects.labels2series()}

        if self.include_XYZ:
            data.update(self._legacy_xyz_metrics(image, objmap))
        if self.include_xy:
            data.update(self._legacy_xy_metrics(image, objmap))

        data.update(self._robust_lab_hsv_metrics(image, objmap))
        return pd.DataFrame(data=data)

    # ------------------------------------------------------------------
    # Robust default block
    # ------------------------------------------------------------------
    def _robust_lab_hsv_metrics(self, image: Image, objmap: np.ndarray) -> dict:
        lab = image.color.Lab[:]
        hsv = image.color.hsv[:]
        labels = np.unique(objmap)
        labels = labels[labels != 0]

        rows: list[dict] = []
        for label in labels:
            mask = objmap == label
            rows.append({**self._robust_lab_row(lab[mask]),
                         **self._robust_hsv_row(hsv[mask])})

        # Assemble column-major dict; preserve header order.
        columns = ColorLab.robust_headers() + ColorHSV.robust_headers()
        return {col: [row[col] for row in rows] for col in columns}

    def _robust_lab_row(self, lab_px: np.ndarray) -> dict:
        gm = robust_color_center(
            lab_px, max_iter=self.geomedian_max_iter, tol=self.geomedian_tol
        )
        medoid, deltas = medoid_ciede2000(
            lab_px, max_pixels=self.medoid_max_pixels, seed=self.random_seed
        )
        de_median, de_mean, de_p95 = delta_e2000_spread(deltas)
        total_var = (
            float(lab_px.var(axis=0, ddof=0).sum()) if lab_px.shape[0] else float("nan")
        )
        return {
            str(ColorLab.L_STAR_GEOMEDIAN): float(gm[0]),
            str(ColorLab.A_STAR_GEOMEDIAN): float(gm[1]),
            str(ColorLab.B_STAR_GEOMEDIAN): float(gm[2]),
            str(ColorLab.L_STAR_MEDOID): float(medoid[0]),
            str(ColorLab.A_STAR_MEDOID): float(medoid[1]),
            str(ColorLab.B_STAR_MEDOID): float(medoid[2]),
            str(ColorLab.DELTA_E2000_MEDIAN): de_median,
            str(ColorLab.DELTA_E2000_MEAN): de_mean,
            str(ColorLab.DELTA_E2000_P95): de_p95,
            str(ColorLab.LAB_TOTAL_VARIANCE): total_var,
            str(ColorLab.MEDOID_COLOR_HEX): lab_to_srgb_hex(medoid),
        }

    def _robust_hsv_row(self, hsv_px: np.ndarray) -> dict:
        cone = hsv_to_cone(hsv_px)
        center_cone = robust_color_center(
            cone, max_iter=self.geomedian_max_iter, tol=self.geomedian_tol
        )
        center_hsv = cone_to_hsv(center_cone)
        cone_var = (
            float(cone.var(axis=0, ddof=0).sum()) if cone.shape[0] else float("nan")
        )
        return {
            str(ColorHSV.HUE_ROBUST_MEAN): float(center_hsv[0]),
            str(ColorHSV.SATURATION_ROBUST_MEAN): float(center_hsv[1]),
            str(ColorHSV.VALUE_ROBUST_MEAN): float(center_hsv[2]),
            str(ColorHSV.HSV_CONE_VARIANCE): cone_var,
        }

    # ------------------------------------------------------------------
    # Legacy opt-in blocks (8-stat suites)
    # ------------------------------------------------------------------
    def _legacy_xyz_metrics(self, image: Image, objmap: np.ndarray) -> dict:
        fg = image.color.XYZ.foreground()
        out = {}
        for ch, headers in (
            (0, ColorXYZ.cieX_headers()),
            (1, ColorXYZ.cieY_headers()),
            (2, ColorXYZ.cieZ_headers()),
        ):
            metrics = MeasureColor._compute_color_metrics(foreground=fg[..., ch], objmap=objmap)
            out.update({k: v for k, v in zip(headers, metrics)})
        return out

    def _legacy_xy_metrics(self, image: Image, objmap: np.ndarray) -> dict:
        fg = image.color.xy.foreground()
        out = {}
        for ch, headers in ((0, Colorxy.x_headers()), (1, Colorxy.y_headers())):
            metrics = MeasureColor._compute_color_metrics(foreground=fg[..., ch], objmap=objmap)
            out.update({k: v for k, v in zip(headers, metrics)})
        return out

    @staticmethod
    def _compute_color_metrics(foreground: np.ndarray, objmap: np.ndarray):
        """Per-object 8-stat suite for the legacy opt-in XYZ/xy paths."""
        return [
            MeasureFeatures._calculate_minimum(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_q1(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_mean(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_median(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_q3(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_maximum(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_stddev(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_coeff_variation(array=foreground, objmap=objmap),
        ]


# Reference-doc RST: only the default colorimetric spaces (Lab, HSV).
MeasureColor.__doc__ = ColorHSV.append_rst_to_doc(
    ColorLab.append_rst_to_doc(MeasureColor)
)
```

> Note: confirm `ColorXYZ.cieX_headers()`/`cieY_headers()`/`cieZ_headers()` and `Colorxy.x_headers()`/`y_headers()` names match the current schema (grep before editing); adjust the calls if the helper names differ.

- [ ] **Step 4: Run to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/measure/test_measure_color.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/measure/_measure_color.py tests/unit/measure/test_measure_color.py
git commit -m "feat(measure): robust-only MeasureColor with opt-in XYZ/xy"
```

---

## Task 7: Hex string column — numeric-safety audit

**Files:**
- Test: `tests/unit/measure/test_measure_color.py`
- Modify (only if a break is found): the offending aggregation/post path.

- [ ] **Step 1: Write the failing/guard test**

```python
# append to tests/unit/measure/test_measure_color.py
def test_hex_column_survives_numeric_aggregation(detected_image):
    df = MeasureColor().measure(detected_image)
    # Simulate the master-aggregation numeric reduction: must not raise on the
    # string hex column and must skip it.
    numeric_means = df.drop(columns=[OBJECT.LABEL]).mean(numeric_only=True)
    assert str(ColorLab.MEDOID_COLOR_HEX) not in numeric_means.index
    # group-mean (replicate aggregation shape) also tolerates the string column
    grouped = df.groupby(OBJECT.LABEL).mean(numeric_only=True)
    assert str(ColorLab.MEDOID_COLOR_HEX) not in grouped.columns
```

- [ ] **Step 2: Run to verify current behavior**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/measure/test_measure_color.py::test_hex_column_survives_numeric_aggregation -q`
Expected: PASS if the codebase already uses `numeric_only=True`. If it FAILS, continue to Step 3.

- [ ] **Step 3: Audit and fix real aggregation paths**

Search the aggregation/post/analysis paths that reduce measurement frames:

```bash
grep -rn "\.mean(\|\.std(\|\.median(\|groupby\|aggregate_measurements\|numeric_only" \
  src/phenotypic/_cli/_cli_output_manager.py \
  src/phenotypic/_cli/_cli_chunk_writer.py \
  src/phenotypic/post/ src/phenotypic/analysis/ | grep -i "mean\|std\|median\|numeric"
```

For any reduction over the full measurement frame that would include `MedoidColorHex`, add `numeric_only=True` (pandas) or `select_dtypes(include="number")` before the reduction. Keep the hex column flowing through to the per-image and mirror outputs unchanged (it is viz-only). Do **not** add it to analysis inputs.

- [ ] **Step 4: Re-run the guard test + the touched modules' tests**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/measure/test_measure_color.py tests/unit/cli/test_cli_output_manager.py tests/unit/util/test_measurement_outputs.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "fix(measure): keep hex swatch out of numeric measurement reductions"
```

---

## Task 8: Reference-table doc — hide XYZ/xy

**Files:**
- Modify: `docs/source/_extensions/measurements_ref.py:44-50`

- [ ] **Step 1: Edit the MeasureColor doc entry**

Change the `MeasureColor` infoclass list to only the default spaces:

```python
    ("phenotypic.measure._measure_color.MeasureColor",
        [
            "phenotypic.schema.ColorLab",
            "phenotypic.schema.ColorHSV",
        ]),
```

- [ ] **Step 2: Build the schema page locally to verify no XYZ/xy rows**

Run: `uv run --group docs sphinx-build -b html docs/source docs/_build/html -q 2>&1 | tail -5`
Expected: build completes; the generated MeasureColor schema table shows only ColorLab + ColorHSV rows. (If `sphinx-build` is heavy, instead assert via the extension's table builder in a quick `uv run python -c` that the rendered tables contain `L*GeoMedian` and not `CieXMean`.)

- [ ] **Step 3: Commit**

```bash
git add docs/source/_extensions/measurements_ref.py
git commit -m "docs: hide CIE XYZ/xy from MeasureColor reference table"
```

---

## Task 9: CLI README generator, prose doc, and sample fixture

**Files:**
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py` (~196, ~213)
- Modify: `docs/source/explanation/measurement_metrics_biological_meaning.md`
- Modify (if referenced by tests): `src/phenotypic/data/meas/all_meas.csv`

- [ ] **Step 1: Update the README generator color mapping**

The deliverables README generator lists color schema classes. Keep `ColorLab` and `ColorHSV` for the default surface; only include `ColorXYZ`/`Colorxy` when those opt-in flags are documented. Concretely, set:

```python
            "MeasureColor": [ColorLab, ColorHSV],
```

and remove `ColorXYZ`/`Colorxy`/unused imports if they are now unreferenced (let ruff flag them in Task 11).

- [ ] **Step 2: Update the prose doc**

In `docs/source/explanation/measurement_metrics_biological_meaning.md`, replace the "Color Metrics (MeasureColor)" section's per-channel description with the robust scheme: two centers (ΔE76 geometric median, ΔE2000 medoid), ΔE2000 consistency (median/mean/P95 from the medoid), `LabTotalVariance`, robust cone-embedded HSV + `HSVConeVariance`, and the plot-only `MedoidColorHex`. Note XYZ/xy are opt-in.

- [ ] **Step 3: Check whether the sample fixture references removed columns**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/analysis/test_icc.py tests/unit/analysis/test_icc_degenerate.py tests/smoke/test_serialization.py -q`
Expected: PASS. The ICC/serialization tests use `all_meas.csv` as generic numeric input. If any test asserts specific color column names that were removed, regenerate the fixture by running a pipeline with the new `MeasureColor` over the sample image and overwriting `all_meas.csv` (mirror the existing generation in `src/phenotypic/data/_sample_image_data.py`). If they pass untouched, leave the CSV as-is.

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/_cli/_cli_readme_generator.py docs/source/explanation/measurement_metrics_biological_meaning.md
# add all_meas.csv only if regenerated
git commit -m "docs/cli: reflect robust color schema in README + metrics guide"
```

---

## Task 10: Migration scenarios + targeted golden regeneration

**Files:**
- Modify: `tests/migration/_scenarios.py:411-414`
- Regenerate: `tests/migration/_goldens/measure.MeasureColor.parquet`, `…with_xyz.parquet`, new `…with_xy.parquet`

- [ ] **Step 1: Add the `with_xy` curated extra**

In the `-- measure --` section, alongside the existing `with_xyz` entry:

```python
    # -- measure --
    _CuratedExtra(
            "MeasureColor", "with_xyz", {"include_XYZ": True}
    ),
    _CuratedExtra(
            "MeasureColor", "with_xy", {"include_xy": True}
    ),
```

- [ ] **Step 2: Regenerate ONLY the MeasureColor goldens (avoid env-drift on others)**

Run this targeted snippet (regenerates just the `measure.MeasureColor*` parquet goldens):

```bash
QT_QPA_PLATFORM=offscreen uv run python - <<'PY'
from tests.migration._scenarios import build_scenarios
from tests.migration._runner import run_scenario, golden_path
targets = [s for s in build_scenarios()
           if s.scenario_id.startswith("measure.MeasureColor")]
assert targets, "no MeasureColor scenarios found"
for s in targets:
    golden = run_scenario(s)
    golden.save(golden_path(s))
    print("wrote", golden_path(s))
PY
```

> Verified APIs: `build_scenarios() -> list[Scenario]` in `tests/migration/_scenarios.py`;
> `run_scenario`, `golden_path`, and the golden object's `.save(path)` in
> `tests/migration/_runner.py` (FrameGolden.save at line 190). Each scenario has
> a `.scenario_id` slug (e.g. `measure.MeasureColor`, `measure.MeasureColor.with_xyz`).

- [ ] **Step 3: Verify migration equivalence for MeasureColor only**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/migration/test_equivalence.py -k MeasureColor -q`
Expected: PASS. (Do NOT run/commit regeneration of unrelated goldens — fresh-worktree float drift would corrupt them.)

- [ ] **Step 4: Commit**

```bash
git add tests/migration/_scenarios.py tests/migration/_goldens/measure.MeasureColor.parquet tests/migration/_goldens/measure.MeasureColor.with_xyz.parquet tests/migration/_goldens/measure.MeasureColor.with_xy.parquet
git commit -m "test(migration): regenerate MeasureColor goldens; add with_xy scenario"
```

---

## Task 11: Quality gates + relevant regression

**Files:** none (verification).

- [ ] **Step 1: Lint + format**

Run: `uv run ruff check --fix src/phenotypic tests`
Expected: clean (fix any unused imports from removed schema usage).

- [ ] **Step 2: Type check**

Run: `uv run mypy src/phenotypic/util/_robust_color_stats.py src/phenotypic/measure/_measure_color.py src/phenotypic/schema/_color_lab.py src/phenotypic/schema/_color_hsv.py`
Expected: no errors. (`colour` is untyped; add `# type: ignore[import-untyped]` on the `import colour` line only if mypy flags it, matching existing repo usage.)

- [ ] **Step 3: Run the full relevant suite**

Run:
```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/util/test_robust_color_stats.py \
  tests/unit/measure/test_measure_color.py \
  tests/unit/schema/test_color_schema.py \
  tests/unit/core/test_pipeline_serialization.py \
  tests/unit/util/test_measurement_outputs.py \
  tests/unit/cli/test_cli_output_manager.py \
  tests/migration/test_equivalence.py -k MeasureColor \
  -q
```
Expected: all PASS.

- [ ] **Step 4: Doctest the docstrings touched**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest --doctest-modules src/phenotypic/measure/_measure_color.py src/phenotypic/util/_robust_color_stats.py -q`
Expected: PASS (add at least one runnable `>>>` example using `load_synth_yeast_plate()` to `MeasureColor` if none exists).

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore(measure): lint/type/doctest pass for robust color measures"
```

---

## Self-Review (completed against the spec)

- **§3.1 robust centers** → Tasks 1, 2, 6. **§3.2 consistency + LabTotalVariance** → Tasks 2, 6. **§3.3 cone HSV** → Tasks 3, 6. **§3.4 hex** → Tasks 3, 6. **§4.1/§4.2 schema** → Tasks 4, 5. **§4.3 opt-in XYZ/xy + doc-hide** → Tasks 6, 8. **§5 params** → Task 6. **§7 hex numeric risk** → Task 7. **§7 golden risk** → Task 10. **§7 downstream refs** → Tasks 8, 9. **§8 testing** → Tasks 1–3, 6, 7, 10, 11.
- **Naming consistency:** `robust_color_center` (reuses `util.geometric_median`), `medoid_ciede2000`, `delta_e2000_spread`, `hsv_to_cone`, `cone_to_hsv`, `lab_to_srgb_hex` are used identically in Tasks 1–3 and Task 6, all living in `phenotypic.util`. Schema member names match between Tasks 4/5 and their use in Task 6.
- **Open items deferred to review (spec §9):** column naming, opt-in suites staying classical, hex provenance — flagged for the user, defaults chosen.
- **Verify-before-edit reminders** embedded where schema helper names (`cieX_headers`, scenario accessor) must be confirmed against the live code.
