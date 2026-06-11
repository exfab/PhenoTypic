# Error-Triage Cutoffs — Phase 3: Cutoff Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and unit-test the pure statistical engine that, for one error category vs. a good baseline, screens every measurement and ranks them by how cleanly they separate the two — emitting a per-measurement ANOVA F/p, separability (AUC), a ROC/Youden cutoff with recall/precision, and a BH-FDR-adjusted p — so the user can read off candidate filter cutoffs.

**Architecture:** A focused `pydantic.BaseModel` analyzer `ErrorCutoffFinder` with `analyze(good, error) -> pd.DataFrame`. It is **mode-agnostic** — it consumes a *good* frame and an *error* frame (the caller decides all-unlabeled vs verified-only good, per the spec's good-baseline modes), which keeps it fully unit-testable with synthetic frames and free of any GUI/IO. Stats via `scipy.stats` (`f_oneway`, `false_discovery_control`) + `sklearn.metrics` (`roc_curve`, `roc_auc_score`).

**Tech Stack:** Python 3.12, pydantic v2 `BaseModel`, pandas/numpy, scipy 1.16, scikit-learn 1.7, pytest, `uv`.

**Depends on:** Phase 1 (the `ERROR_ANALYSIS_*` io_constants already exist; not needed by the engine itself). No GUI, no CLI — Phase 4 wires this into the Error-analysis tab and Phase 5 persists its output.

**Spec:** `docs/superpowers/specs/2026-06-10-error-category-triage-cutoff-finder-design.md` (§7 Cutoff engine; §2 ranking by effect size; the verified-good baseline is the caller's concern, not the engine's).

---

## Conventions for this plan

- `uv run` for everything. Single test: `uv run pytest <path>::<test> -v`. No Qt, no browser — these are pure numeric unit tests.
- Commit per task, scoped `git add <paths>`. Worktree: `/Users/alex/Projects/PhenoTypic/.claude/worktrees/error-triage-cutoffs`.
- ⚠️ NEVER `git stash` / `git checkout <ref>` / branch-switch — unrelated user stashes live in this worktree.
- Google-style docstrings; runnable doctests use `load_synth_yeast_plate()` only where natural (the engine takes raw frames, so its doctests build tiny pandas frames inline — acceptable for a pure analyzer).
- Follow `analysis/` conventions: analyzers are pydantic models exported from `analysis/__init__.py`; the memory note "new analyzers must be re-exported to be discoverable" applies.

## Design decisions settled here (the spec left these to the plan)

- **Not a `SetAnalyzer` subclass.** `SetAnalyzer` is built around `on` (one column) + `groupby` (per-group iteration) + `_apply2group_func`; the cutoff engine screens *all* measurement columns across a *good-vs-error* split with no grouping. Forcing those fields would be misleading. So `ErrorCutoffFinder(pydantic.BaseModel)` is a standalone analyzer that keeps the project's `.analyze()` entry-point name. (If the reviewer prefers SetAnalyzer, that's a one-task refactor — but the fit is poor.)
- **Frames are pandas** (consistent with `analysis/` — `MADOutlierRemover`, the QC checks all use `pd.DataFrame`). The Phase-4 GUI converts its polars master/curated frames to pandas at the boundary.
- **Engine takes good + error frames** (not a labeled master + category). The good/error split — and the verified-only restriction — are the caller's responsibility (spec §7).
- **Ranking key = separability** = `max(auc_raw, 1 - auc_raw)` (direction-agnostic), recorded with a `direction` (`">"` = flag when measurement is above the cutoff, `"<"` = below). Effect size, not raw p (p is sample-size driven).
- **Insufficient data → empty frame.** When `len(error) < min_error_n` or `len(good) < min_good_n`, `analyze` returns an empty (0-row) frame with the right columns; a public `enough_data(good, error)` predicate lets the Phase-4 panel show the "review more / mark more" state without catching exceptions.

## File structure (Phase 3)

- Create: `src/phenotypic/analysis/_error_cutoffs.py` — `ErrorCutoffFinder` + the per-measurement scorer.
- Modify: `src/phenotypic/analysis/__init__.py` — export `ErrorCutoffFinder`.
- Create: `tests/unit/analysis/test_error_cutoffs.py`.

---

### Task 1: `ErrorCutoffFinder` skeleton + measurement detection + export

**Files:**
- Create: `src/phenotypic/analysis/_error_cutoffs.py`
- Modify: `src/phenotypic/analysis/__init__.py`
- Test: `tests/unit/analysis/test_error_cutoffs.py`

**Why:** Establish the pydantic model, its config fields, the measurement-column auto-detection (the same prefixes the colony grid + spec use), and the public export — before the stats land.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/analysis/test_error_cutoffs.py`:

```python
"""Tests for the error-cutoff finder (good-vs-category measurement screen)."""

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis import ErrorCutoffFinder


def _frame(values: dict[str, list[float]], n: int) -> pd.DataFrame:
    """Build a frame with the given measurement columns + filler metadata."""
    base = {
        "Metadata_ImageFile": ["p.tif"] * n,
        "Object_Label": list(range(1, n + 1)),
    }
    base.update(values)
    return pd.DataFrame(base)


def test_measurement_columns_detects_only_numeric_measurements():
    finder = ErrorCutoffFinder()
    df = _frame(
        {
            "Size_Area": [1.0, 2.0, 3.0],
            "Shape_Circularity": [0.1, 0.2, 0.3],
            "Intensity_MeanIntensity": [10.0, 11.0, 12.0],
            "Grid_RowNum": [1, 1, 2],  # grid context, not a measurement
        },
        n=3,
    )
    cols = finder.measurement_columns(df)
    assert "Size_Area" in cols
    assert "Shape_Circularity" in cols
    assert "Intensity_MeanIntensity" in cols
    # Metadata / object-id / grid-context columns are excluded.
    assert "Metadata_ImageFile" not in cols
    assert "Object_Label" not in cols
    assert "Grid_RowNum" not in cols


def test_default_min_n_fields():
    finder = ErrorCutoffFinder()
    assert finder.min_error_n == 8
    assert finder.min_good_n == 8
    # keyword-only pydantic construction; bad kwargs raise.
    with pytest.raises(Exception):
        ErrorCutoffFinder(min_error_n=5, bogus=1)


def test_enough_data_predicate():
    finder = ErrorCutoffFinder(min_error_n=3, min_good_n=3)
    good = _frame({"Size_Area": [1.0] * 5}, n=5)
    error = _frame({"Size_Area": [9.0] * 2}, n=2)  # below min_error_n
    assert finder.enough_data(good, error) is False
    error2 = _frame({"Size_Area": [9.0] * 4}, n=4)
    assert finder.enough_data(good, error2) is True
```

- [ ] **Step 2: Run → fail** (`ImportError: cannot import name 'ErrorCutoffFinder'`).
Run: `uv run pytest tests/unit/analysis/test_error_cutoffs.py -v`

- [ ] **Step 3: Implement the skeleton**

Create `src/phenotypic/analysis/_error_cutoffs.py`:

```python
"""Good-vs-error-category measurement screen with per-measurement cutoffs.

For one error category, :class:`ErrorCutoffFinder` compares the *good* baseline
distribution against the *error* distribution on every measurement column and
ranks the measurements by how cleanly they separate the two (AUC). Each
discriminative measurement gets a ROC/Youden's-J cutoff with the recall and
precision it achieves, plus the one-way ANOVA F/p and a Benjamini-Hochberg
FDR-adjusted p. The result is the table the Error-analysis tab reads so the
user can adopt a cutoff to filter similar bad data.

The engine is deliberately **GUI/IO-free and mode-agnostic**: it takes a *good*
frame and an *error* frame and does not know whether the good baseline is
"all unlabeled" or the verified-only set — the caller decides (spec §7).
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

#: Column-name prefixes that mark a numeric measurement (mirrors the spec's
#: measurement-selection list and the colony grid's ``_MEASUREMENT_PREFIXES``).
MEASUREMENT_PREFIXES: tuple[str, ...] = (
    "Size_",
    "Shape_",
    "Intensity_",
    "TextureGray_",
    "SymZones_",
    "GridSpatial_",
    "Bbox_",
    "RadialExpansion_",
)

#: Output columns of :meth:`ErrorCutoffFinder.analyze`, in order.
RESULT_COLUMNS: tuple[str, ...] = (
    "measurement",
    "auc",
    "direction",
    "cutoff",
    "recall",
    "precision",
    "f_stat",
    "p_value",
    "p_bh",
    "good_n",
    "error_n",
)


class ErrorCutoffFinder(BaseModel):
    """Rank measurements by good-vs-error separability with suggested cutoffs.

    Args:
        min_error_n: Minimum error-class sample size; below it, :meth:`analyze`
            returns an empty frame (the statistics would be unstable).
        min_good_n: Minimum good-class sample size; same behaviour.
        measurement_prefixes: Column-name prefixes treated as numeric
            measurements. Defaults to :data:`MEASUREMENT_PREFIXES`.
    """

    model_config = ConfigDict(extra="forbid")

    min_error_n: int = 8
    min_good_n: int = 8
    measurement_prefixes: tuple[str, ...] = MEASUREMENT_PREFIXES

    def measurement_columns(self, df: pd.DataFrame) -> list[str]:
        """Return the numeric measurement columns of ``df`` in column order.

        A column qualifies iff its name starts with one of
        :attr:`measurement_prefixes` and its dtype is numeric.

        Args:
            df: A measurement frame (good or error).

        Returns:
            The qualifying measurement column names.
        """
        out: list[str] = []
        for col in df.columns:
            if not col.startswith(self.measurement_prefixes):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                out.append(col)
        return out

    def enough_data(self, good: pd.DataFrame, error: pd.DataFrame) -> bool:
        """Return whether both classes meet their minimum sample sizes."""
        return len(good) >= self.min_good_n and len(error) >= self.min_error_n
```

- [ ] **Step 4: Export from `analysis/__init__.py`**

Add `from ._error_cutoffs import ErrorCutoffFinder` (next to the other analyzer imports) and `"ErrorCutoffFinder"` to `__all__` (keep alphabetical within its grouping).

- [ ] **Step 5: Run → pass; commit.**
Run: `uv run pytest tests/unit/analysis/test_error_cutoffs.py -v` (3 pass), `uv run mypy src/phenotypic/analysis/_error_cutoffs.py`, `uv run ruff check`.
```bash
git add src/phenotypic/analysis/_error_cutoffs.py src/phenotypic/analysis/__init__.py tests/unit/analysis/test_error_cutoffs.py
git commit -m "feat(analysis): ErrorCutoffFinder skeleton + measurement detection"
```

---

### Task 2: per-measurement stats + cutoff + FDR + ranking (`analyze`)

**Files:**
- Modify: `src/phenotypic/analysis/_error_cutoffs.py`
- Test: `tests/unit/analysis/test_error_cutoffs.py`

**Why:** The core. For each measurement: one-way ANOVA, AUC + direction (sklearn), ROC/Youden cutoff with recall/precision, then BH-FDR across measurements and a separability ranking.

- [ ] **Step 1: Write the failing tests**

Append:

```python
def _separating(n_good=40, n_err=20, seed=0):
    """Good ~ N(0,1); error ~ N(4,1) on Size_Area (clearly separable),
    plus a non-separating Shape_Circularity ~ N(0.5,0.05) in both."""
    rng = np.random.default_rng(seed)
    good = pd.DataFrame(
        {
            "Metadata_ImageFile": ["p.tif"] * n_good,
            "Object_Label": list(range(1, n_good + 1)),
            "Size_Area": rng.normal(0.0, 1.0, n_good),
            "Shape_Circularity": rng.normal(0.5, 0.05, n_good),
        }
    )
    error = pd.DataFrame(
        {
            "Metadata_ImageFile": ["p.tif"] * n_err,
            "Object_Label": list(range(1, n_err + 1)),
            "Size_Area": rng.normal(4.0, 1.0, n_err),
            "Shape_Circularity": rng.normal(0.5, 0.05, n_err),
        }
    )
    return good, error


def test_separating_measurement_ranks_first_with_high_auc():
    good, error = _separating()
    res = ErrorCutoffFinder().analyze(good, error)
    assert list(res.columns) == [
        "measurement", "auc", "direction", "cutoff", "recall",
        "precision", "f_stat", "p_value", "p_bh", "good_n", "error_n",
    ]
    # Size_Area separates; it ranks first with high AUC.
    assert res.iloc[0]["measurement"] == "Size_Area"
    assert res.iloc[0]["auc"] > 0.9
    # Error is the HIGH side -> flag when measurement is ABOVE the cutoff.
    assert res.iloc[0]["direction"] == ">"
    # The cutoff sits between the two means.
    assert 0.5 < res.iloc[0]["cutoff"] < 4.0
    # Recall/precision are sane fractions.
    assert 0.5 <= res.iloc[0]["recall"] <= 1.0
    assert 0.5 <= res.iloc[0]["precision"] <= 1.0
    # The non-separating measurement has AUC near 0.5.
    circ = res[res["measurement"] == "Shape_Circularity"].iloc[0]
    assert abs(circ["auc"] - 0.5) < 0.15
    # n columns reflect inputs.
    assert res.iloc[0]["good_n"] == 40
    assert res.iloc[0]["error_n"] == 20


def test_direction_below_when_error_is_low_side():
    # Error LOWER than good -> flag when measurement is BELOW the cutoff.
    rng = np.random.default_rng(1)
    good = pd.DataFrame({"Object_Label": range(40), "Intensity_MeanIntensity": rng.normal(5, 0.5, 40)})
    error = pd.DataFrame({"Object_Label": range(20), "Intensity_MeanIntensity": rng.normal(1, 0.5, 20)})
    res = ErrorCutoffFinder().analyze(good, error)
    row = res.iloc[0]
    assert row["measurement"] == "Intensity_MeanIntensity"
    assert row["direction"] == "<"
    assert 1.0 < row["cutoff"] < 5.0


def test_bh_adjusted_p_is_monotone_and_ge_raw():
    good, error = _separating()
    res = ErrorCutoffFinder().analyze(good, error)
    # BH-adjusted p >= raw p for every measurement.
    assert (res["p_bh"] >= res["p_value"] - 1e-9).all()


def test_insufficient_error_returns_empty_frame():
    good, error = _separating(n_good=40, n_err=3)
    res = ErrorCutoffFinder(min_error_n=8).analyze(good, error)
    assert res.empty
    assert list(res.columns) == list(__import__("phenotypic.analysis._error_cutoffs",
                                                fromlist=["RESULT_COLUMNS"]).RESULT_COLUMNS)
```

- [ ] **Step 2: Run → fail** (`AttributeError: ... has no attribute 'analyze'`).

- [ ] **Step 3: Implement `analyze` + the per-measurement scorer**

Add to `_error_cutoffs.py` (imports at top: `import numpy as np`, `from scipy.stats import f_oneway, false_discovery_control`, `from sklearn.metrics import roc_auc_score, roc_curve`):

```python
    def analyze(self, good: pd.DataFrame, error: pd.DataFrame) -> pd.DataFrame:
        """Screen every measurement for good-vs-error separation.

        Args:
            good: The good-baseline frame (caller chooses all-unlabeled vs
                verified-only — the engine is agnostic).
            error: The frame of objects labelled with the target error
                category.

        Returns:
            A frame with one row per measurement, columns
            :data:`RESULT_COLUMNS`, sorted by ``auc`` (separability) descending.
            Empty (0 rows, same columns) when :meth:`enough_data` is ``False``
            or no measurement column has enough non-NaN values in both classes.
        """
        empty = pd.DataFrame(columns=list(RESULT_COLUMNS))
        if not self.enough_data(good, error):
            return empty

        rows: list[dict[str, object]] = []
        for col in self.measurement_columns(good):
            if col not in error.columns:
                continue
            g = pd.to_numeric(good[col], errors="coerce").dropna().to_numpy()
            e = pd.to_numeric(error[col], errors="coerce").dropna().to_numpy()
            scored = self._score_measurement(g, e)
            if scored is None:
                continue
            scored["measurement"] = col
            rows.append(scored)

        if not rows:
            return empty

        res = pd.DataFrame(rows)
        # Benjamini-Hochberg across the screened measurements.
        res["p_bh"] = false_discovery_control(res["p_value"].to_numpy(), method="bh")
        res = res.sort_values("auc", ascending=False, ignore_index=True)
        return res[list(RESULT_COLUMNS)]

    @staticmethod
    def _score_measurement(g: "np.ndarray", e: "np.ndarray") -> dict | None:
        """Score one measurement: ANOVA F/p, AUC + direction, Youden cutoff.

        Args:
            g: Good-class values (NaN-free 1-D array).
            e: Error-class values (NaN-free 1-D array).

        Returns:
            A dict of the per-measurement statistics, or ``None`` when either
            class has < 2 values or zero variance makes the test degenerate.
        """
        if len(g) < 2 or len(e) < 2:
            return None
        scores = np.concatenate([g, e])
        if np.ptp(scores) == 0:  # all identical -> nothing to separate
            return None
        y = np.concatenate([np.zeros(len(g)), np.ones(len(e))])  # 1 = error

        f_stat, p_value = f_oneway(g, e)
        auc_raw = roc_auc_score(y, scores)  # P(error score > good score)

        if auc_raw >= 0.5:
            direction = ">"  # error is the HIGH side
            separability = auc_raw
            fpr, tpr, thr = roc_curve(y, scores)
        else:
            direction = "<"  # error is the LOW side
            separability = 1.0 - auc_raw
            fpr, tpr, thr = roc_curve(y, -scores)
            thr = -thr  # map thresholds back to the measurement scale

        # Youden's J optimal operating point. roc_curve prepends an
        # +/-inf threshold (the "classify nothing as positive" point); skip it.
        valid = np.isfinite(thr)
        j = tpr[valid] - fpr[valid]
        k = int(np.argmax(j))
        cutoff = float(thr[valid][k])
        recall = float(tpr[valid][k])
        # precision = TP / (TP + FP) at the operating point.
        tp = tpr[valid][k] * len(e)
        fp = fpr[valid][k] * len(g)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")

        return {
            "auc": float(separability),
            "direction": direction,
            "cutoff": cutoff,
            "recall": recall,
            "precision": precision,
            "f_stat": float(f_stat),
            "p_value": float(p_value),
            "good_n": int(len(g)),
            "error_n": int(len(e)),
        }
```

- [ ] **Step 4: Run → pass** (all Task-2 tests). Fix any boundary issues (e.g. `roc_curve`'s inf threshold handling) until green.

- [ ] **Step 5: Commit.**
```bash
git add src/phenotypic/analysis/_error_cutoffs.py tests/unit/analysis/test_error_cutoffs.py
git commit -m "feat(analysis): ErrorCutoffFinder.analyze — ANOVA + AUC + Youden cutoff + BH-FDR"
```

---

### Task 3: edge cases, NaN robustness, doctest + final gate

**Files:**
- Modify: `src/phenotypic/analysis/_error_cutoffs.py`
- Test: `tests/unit/analysis/test_error_cutoffs.py`

**Why:** Lock down the degenerate paths a real run will hit (a measurement that's all-NaN in one class, a constant measurement, perfectly-separated classes, missing columns) and add a runnable doctest so the public API is self-documenting.

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_all_nan_measurement_is_skipped_not_crashed():
    good = pd.DataFrame({"Object_Label": range(20), "Size_Area": np.r_[np.full(20, np.nan)],
                         "Shape_Circularity": np.random.default_rng(0).normal(0.5, 0.1, 20)})
    error = pd.DataFrame({"Object_Label": range(10), "Size_Area": np.full(10, np.nan),
                          "Shape_Circularity": np.random.default_rng(1).normal(0.7, 0.1, 10)})
    res = ErrorCutoffFinder(min_good_n=10, min_error_n=8).analyze(good, error)
    assert "Size_Area" not in set(res["measurement"])  # all-NaN -> skipped
    assert "Shape_Circularity" in set(res["measurement"])


def test_constant_measurement_is_skipped():
    good = pd.DataFrame({"Object_Label": range(20), "Size_Area": np.full(20, 3.0)})
    error = pd.DataFrame({"Object_Label": range(10), "Size_Area": np.full(10, 3.0)})
    res = ErrorCutoffFinder(min_good_n=10, min_error_n=8).analyze(good, error)
    assert res.empty  # no separable measurement


def test_perfect_separation_recall_precision_one():
    good = pd.DataFrame({"Object_Label": range(20), "Size_Area": np.linspace(0, 1, 20)})
    error = pd.DataFrame({"Object_Label": range(10), "Size_Area": np.linspace(10, 11, 10)})
    res = ErrorCutoffFinder(min_good_n=10, min_error_n=8).analyze(good, error)
    row = res.iloc[0]
    assert row["auc"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["precision"] == pytest.approx(1.0)
    assert 1.0 < row["cutoff"] < 10.0


def test_measurement_only_in_good_is_ignored():
    good = pd.DataFrame({"Object_Label": range(20), "Size_Area": np.random.default_rng(0).normal(0, 1, 20),
                         "Shape_OnlyHere": np.random.default_rng(0).normal(0, 1, 20)})
    error = pd.DataFrame({"Object_Label": range(10), "Size_Area": np.random.default_rng(1).normal(3, 1, 10)})
    res = ErrorCutoffFinder(min_good_n=10, min_error_n=8).analyze(good, error)
    assert "Shape_OnlyHere" not in set(res["measurement"])
```

- [ ] **Step 2: Run → confirm which fail.** Most should already pass from Task 2's implementation (the scorer skips `<2` / constant / NaN). Fix any that don't (e.g. ensure `min_good_n`/`min_error_n` count rows, and the all-NaN column drops to `<2` after `dropna` → returns `None`).

- [ ] **Step 3: Add a runnable doctest to `ErrorCutoffFinder.analyze`** (or the class docstring) so `--doctest-modules` covers the public API:

```python
        Examples:
            >>> import numpy as np, pandas as pd
            >>> rng = np.random.default_rng(0)
            >>> good = pd.DataFrame({"Size_Area": rng.normal(0, 1, 40)})
            >>> error = pd.DataFrame({"Size_Area": rng.normal(5, 1, 12)})
            >>> res = ErrorCutoffFinder().analyze(good, error)
            >>> res.iloc[0]["measurement"], bool(res.iloc[0]["auc"] > 0.9)
            ('Size_Area', True)
```

- [ ] **Step 4: Final gate + commit.**
Run: `uv run pytest tests/unit/analysis/test_error_cutoffs.py -v` (all pass), `uv run pytest --doctest-modules src/phenotypic/analysis/_error_cutoffs.py`, `uv run mypy src/phenotypic/analysis/_error_cutoffs.py`, `uv run ruff check --fix` on the touched files.
```bash
git add src/phenotypic/analysis/_error_cutoffs.py tests/unit/analysis/test_error_cutoffs.py
git commit -m "test(analysis): ErrorCutoffFinder edge cases + doctest"
```

---

## Self-review (against spec §7)

- ANOVA F/p per measurement → Task 2 (`f_oneway`). ✅
- Effect size + AUC ranking → Task 2 (`roc_auc_score`, separability = `max(auc,1-auc)`, sorted desc). ✅
- ROC/Youden cutoff + direction + recall/precision → Task 2 (`roc_curve` + Youden's J). ✅
- BH-FDR across measurements → Task 2 (`false_discovery_control`). ✅
- Output frame `[measurement, auc, f, p, p_bh, cutoff, direction, recall, precision, good_n, error_n]` → `RESULT_COLUMNS`. ✅
- Measurement auto-detection by prefix → Task 1. ✅
- min-n guard → Task 1 (`enough_data`) + Task 2 (empty frame). ✅
- Mode-agnostic good/error inputs (verified-only is the caller's concern) → the whole design. ✅
- Exported/discoverable → Task 1 Step 4. ✅

Not in Phase 3 (deferred): the good/error frame construction + verified-only derivation (Phase 4), the Error-analysis tab + toggle + draggable cutoff (Phase 4), persisting `error_analysis.*`/`verified.parquet` (Phases 4–5).
