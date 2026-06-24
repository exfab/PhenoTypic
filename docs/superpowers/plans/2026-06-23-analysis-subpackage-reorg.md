# Analysis Subpackage Reorg + Edge Correction GUI Section — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `src/phenotypic/analysis/` into `filter/`, `edge/`, and private `_helper/` subpackages (mirroring `qc/`), introduce an `EdgeCorrection` abstract base, and wire a dedicated "Edge Correction" section into the analysis GUI sub-app.

**Architecture:** Pure file moves via `git mv` keep history; the public API (`from phenotypic.analysis import X`) is preserved by keeping every re-export in `analysis/__init__.py`, so pipeline.json (de)serialization and the GUI registry — both name-based — keep working. `EdgeCorrection(SetAnalyzer, ABC)` factors the grid-config + topology out of `EdgeCorrector`. The GUI gains an `"edge"` section kind; because the pipeline stores all non-model analyzers in one `_filters` dict, a single partition helper splits that dict by registry category for the filter vs edge stacks.

**Tech Stack:** Python 3, pydantic v2, Dash, joblib, pytest, `uv` runner, mypy, ruff, sphinx-apidoc.

## Global Constraints

- **Runner:** `uv` only. Never bare `python`/`pip`. Run tests/tools via `uv run …`.
- **Qt tests:** GUI/analysis tests need a Qt binding + offscreen: prefix with `QT_QPA_PLATFORM=offscreen` and run under `uv run --group dev --group qt-test --extra gui …`.
- **Keyword-only construction:** all operations/analyzers are pydantic models — `EdgeCorrector(on=..., groupby=...)`, never positional.
- **Public API unchanged:** every class currently importable as `from phenotypic.analysis import X` must remain so. Class **names** never change.
- **`EdgeCorrection` is `abc_`-only:** export from `analysis/abc_/__init__.py`; do **not** re-export from `analysis/__init__.py` (keeps it out of the registry walk).
- **No re-export shims:** hard cutover at old private paths.
- **GUI category string is exact:** `"Edge Correction"` must match between the registry branch and `_choices_for_category("Edge Correction")`.
- **FEATURES.md gate:** any change under `src/phenotypic/gui/` requires a `FEATURES.md` edit (CI `features-md-gate` + pre-commit), with a resolvable `Test ref` on `✅ shipping` rows.
- **`bio_desc`/`image` on MeasurementInfo:** never author — out of scope here.
- **COMMIT SCOPING (overrides every commit/lint step below):** the branch
  `refactor/output-and-docs` carries large UNRELATED uncommitted work (schema/,
  measure/, docs rst, …) and a concurrent SDD run. **Never `git add -A` / `git
  add .` / `git add tests/`.** Stage only this task's own files via explicit
  paths. `git mv` already stages the moves. The exclusive "safe roots" for this
  refactor are: `src/phenotypic/analysis/`, `src/phenotypic/gui/analysis/`,
  `src/phenotypic/gui/_operation_registry.py`, `src/phenotypic/gui/FEATURES.md`,
  `src/phenotypic/gui/WORKFLOWS.md`, `docs/source/api_reference/phenotypic.analysis*.rst`,
  `docs/source/tutorials/gui/`, and the **specific** test files each task names.
  Any `ruff`/`mypy` run MUST be path-scoped to `src/phenotypic/analysis` and/or
  `src/phenotypic/gui` — never the whole repo (a repo-wide `ruff --fix` would
  rewrite the unrelated uncommitted files). Before committing, run `git status`
  and confirm no file outside your task's list is staged.

---

## File Structure

**New files**
- `src/phenotypic/analysis/_helper/__init__.py` — re-export public error-report funcs.
- `src/phenotypic/analysis/filter/__init__.py` — export `MADOutlierRemover`, `TukeyOutlierRemover`.
- `src/phenotypic/analysis/edge/__init__.py` — export `EdgeCorrector`.
- `src/phenotypic/analysis/abc_/_edge_correction.py` — `EdgeCorrection(SetAnalyzer, ABC)`.
- `docs/source/api_reference/phenotypic.analysis.filter.rst`, `…edge.rst`.

**Moved files (`git mv`)**
- `_qc_math.py`, `_error_report.py`, `_inoculum_prior.py` → `_helper/`.
- `_mad_outlier.py`, `_tukey_outlier.py` → `filter/`.
- `_edge_correction.py` → `edge/`.

**Modified files**
- `analysis/__init__.py`, `analysis/abc_/__init__.py`.
- `analysis/qc/_max_modz.py`, `_relative_mad.py`, `_tukey_fraction.py`.
- `analysis/abc_/_linear_softplus_base.py`.
- `gui/_operation_registry.py`.
- `gui/analysis/_ids.py`, `_layout.py`, `_callbacks.py`.
- `gui/FEATURES.md`, `gui/WORKFLOWS.md`.
- `docs/source/api_reference/phenotypic.analysis.rst`.
- Tests listed per task.

---

## Phase 1 — `_helper/` subpackage

### Task 1: Move helper modules into `analysis/_helper/`

**Files:**
- Create: `src/phenotypic/analysis/_helper/__init__.py`
- Move: `_qc_math.py`, `_error_report.py`, `_inoculum_prior.py` → `_helper/`
- Modify: `analysis/__init__.py`, `analysis/qc/_max_modz.py`, `_relative_mad.py`, `_tukey_fraction.py`, `analysis/abc_/_linear_softplus_base.py`, `analysis/_mad_outlier.py`, `analysis/_tukey_outlier.py`, the moved `_qc_math.py`'s own docstring/doctests
- Test: `tests/unit/analysis/test_max_modz.py`, `test_relative_mad.py`, `test_tukey_fraction.py`

**Interfaces:**
- Produces: `phenotypic.analysis._helper._qc_math` (module with `median_abs_deviation`, `modified_z_scores`, `tukey_fences`, `tukey_outlier_mask`, `tukey_outlier_fraction`, `MAD_CONSISTENCY`); `phenotypic.analysis._helper` re-exports `render_error_analysis_html`, `render_error_analysis_report`, `filter_spec_json`, `filter_spec_query`; `phenotypic.analysis._helper._inoculum_prior._InoculumPrior`.

- [ ] **Step 1: Write the failing test** (new import surface)

Append to `tests/unit/analysis/test_max_modz.py` (top-level, after existing imports):

```python
def test_qc_math_moved_to_helper():
    from phenotypic.analysis._helper._qc_math import modified_z_scores
    from phenotypic.analysis._helper import render_error_analysis_report
    import importlib
    import pytest
    # old private path is gone (hard cutover)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phenotypic.analysis._qc_math")
    assert callable(modified_z_scores)
    assert callable(render_error_analysis_report)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/analysis/test_max_modz.py::test_qc_math_moved_to_helper -q`
Expected: FAIL (`ModuleNotFoundError: phenotypic.analysis._helper`).

- [ ] **Step 3: Create the package and move the files**

```bash
mkdir -p src/phenotypic/analysis/_helper
git mv src/phenotypic/analysis/_qc_math.py src/phenotypic/analysis/_helper/_qc_math.py
git mv src/phenotypic/analysis/_error_report.py src/phenotypic/analysis/_helper/_error_report.py
git mv src/phenotypic/analysis/_inoculum_prior.py src/phenotypic/analysis/_helper/_inoculum_prior.py
```

- [ ] **Step 4: Write `_helper/__init__.py`**

Create `src/phenotypic/analysis/_helper/__init__.py`:

```python
"""Private helper modules for the analysis package.

Pure utilities with no public operation class: robust-statistics math
(:mod:`._qc_math`), error-report rendering (:mod:`._error_report`), and the
softplus inoculum-prior helper (:mod:`._inoculum_prior`). The public
error-report functions are re-exported here so :mod:`phenotypic.analysis`
can surface them from a single private home.
"""

from ._error_report import (
    filter_spec_json,
    filter_spec_query,
    render_error_analysis_html,
    render_error_analysis_report,
)

__all__ = [
    "filter_spec_json",
    "filter_spec_query",
    "render_error_analysis_html",
    "render_error_analysis_report",
]
```

- [ ] **Step 5: Update the importers (exact old → new)**

In `src/phenotypic/analysis/__init__.py`, change the `_error_report` import block (currently `from ._error_report import (`) to:

```python
from ._helper import (
    filter_spec_json,
    filter_spec_query,
    render_error_analysis_html,
    render_error_analysis_report,
)
```

In `src/phenotypic/analysis/qc/_max_modz.py` line 25:
`from phenotypic.analysis._qc_math import median_abs_deviation, modified_z_scores`
→ `from phenotypic.analysis._helper._qc_math import median_abs_deviation, modified_z_scores`

In `src/phenotypic/analysis/qc/_relative_mad.py` line 26:
`from phenotypic.analysis._qc_math import median_abs_deviation`
→ `from phenotypic.analysis._helper._qc_math import median_abs_deviation`

In `src/phenotypic/analysis/qc/_tukey_fraction.py` line 23:
`from phenotypic.analysis._qc_math import tukey_fences, tukey_outlier_mask`
→ `from phenotypic.analysis._helper._qc_math import tukey_fences, tukey_outlier_mask`

In `src/phenotypic/analysis/abc_/_linear_softplus_base.py` line 23:
`from phenotypic.analysis._inoculum_prior import _InoculumPrior`
→ `from phenotypic.analysis._helper._inoculum_prior import _InoculumPrior`

In `src/phenotypic/analysis/_mad_outlier.py` line 11:
`from . import _qc_math` → `from ._helper import _qc_math`

In `src/phenotypic/analysis/_tukey_outlier.py` line 11:
`from . import _qc_math` → `from ._helper import _qc_math`

- [ ] **Step 6: Fix the moved `_qc_math.py`'s own Sphinx refs + doctests**

In `src/phenotypic/analysis/_helper/_qc_math.py`:
- Module docstring (≈lines 4–5): `:class:`~phenotypic.analysis._mad_outlier.MADOutlierRemover`` → `:class:`~phenotypic.analysis.filter._mad_outlier.MADOutlierRemover``, and `._tukey_outlier.TukeyOutlierRemover` → `.filter._tukey_outlier.TukeyOutlierRemover`. (These point at the filter package the move lands in Phase 2.)
- Every doctest line `>>> from phenotypic.analysis._qc_math import …` → `>>> from phenotypic.analysis._helper._qc_math import …` (use `replace_all`).
In `src/phenotypic/analysis/qc/_max_modz.py` line 59 docstring `:func:`~phenotypic.analysis._qc_math.modified_z_scores`` → `:func:`~phenotypic.analysis._helper._qc_math.modified_z_scores``; and any `>>> from phenotypic.analysis._qc_math import …` doctest lines in `_max_modz.py`, `_relative_mad.py`, `_tukey_fraction.py` → `._helper._qc_math`.

- [ ] **Step 7: Update the three direct-import tests**

`tests/unit/analysis/test_max_modz.py:17`, `tests/unit/analysis/test_relative_mad.py:15`, `tests/unit/analysis/test_tukey_fraction.py:15`:
`from phenotypic.analysis._qc_math import …` → `from phenotypic.analysis._helper._qc_math import …`.

- [ ] **Step 8: Run the analysis unit tests**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/analysis -q`
Expected: PASS (including `test_qc_math_moved_to_helper`).

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/analysis tests/unit/analysis/test_max_modz.py tests/unit/analysis/test_relative_mad.py tests/unit/analysis/test_tukey_fraction.py
git status   # confirm ONLY the above are staged — nothing under schema/ measure/ etc.
git commit -m "refactor(analysis): move helper modules into analysis._helper"
```

---

## Phase 2 — `filter/` subpackage

### Task 2: Move the outlier removers into `analysis/filter/`

**Files:**
- Create: `src/phenotypic/analysis/filter/__init__.py`
- Move: `_mad_outlier.py`, `_tukey_outlier.py` → `filter/`
- Modify: `analysis/__init__.py`, `analysis/_helper/_qc_math.py` (already pointed to filter paths in Task 1 — verify), the moved files' `_helper` import
- Test: `tests/unit/analysis/test_mad_outlier.py`, `test_tukey_outlier.py`, `test_log_growth_model.py`

**Interfaces:**
- Consumes: `phenotypic.analysis._helper._qc_math` (Task 1).
- Produces: `phenotypic.analysis.filter` exports `MADOutlierRemover`, `TukeyOutlierRemover`; both remain importable as `from phenotypic.analysis import MADOutlierRemover, TukeyOutlierRemover`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/analysis/test_mad_outlier.py`:

```python
def test_filter_subpackage_paths():
    from phenotypic.analysis.filter import MADOutlierRemover, TukeyOutlierRemover
    from phenotypic.analysis import MADOutlierRemover as PublicMAD
    import importlib, pytest
    assert MADOutlierRemover is PublicMAD
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phenotypic.analysis._mad_outlier")
    assert TukeyOutlierRemover is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/analysis/test_mad_outlier.py::test_filter_subpackage_paths -q`
Expected: FAIL (`ModuleNotFoundError: phenotypic.analysis.filter`).

- [ ] **Step 3: Create the package and move the files**

```bash
mkdir -p src/phenotypic/analysis/filter
git mv src/phenotypic/analysis/_mad_outlier.py src/phenotypic/analysis/filter/_mad_outlier.py
git mv src/phenotypic/analysis/_tukey_outlier.py src/phenotypic/analysis/filter/_tukey_outlier.py
```

- [ ] **Step 4: Write `filter/__init__.py`**

Create `src/phenotypic/analysis/filter/__init__.py`:

```python
"""Set-analyzer filters that prune outlier rows from measurement frames.

Each class is a :class:`~phenotypic.analysis.abc_.SetAnalyzer` subclass that
removes colony measurements whose statistics mark them as outliers
(MAD-based or Tukey-fence-based), so downstream comparisons are not skewed
by a handful of extreme values.
"""

from ._mad_outlier import MADOutlierRemover
from ._tukey_outlier import TukeyOutlierRemover

__all__ = [
    "MADOutlierRemover",
    "TukeyOutlierRemover",
]
```

- [ ] **Step 5: Fix the moved files' helper import depth**

In `src/phenotypic/analysis/filter/_mad_outlier.py` line 11:
`from ._helper import _qc_math` → `from .._helper import _qc_math`
In `src/phenotypic/analysis/filter/_tukey_outlier.py` line 11:
`from ._helper import _qc_math` → `from .._helper import _qc_math`

- [ ] **Step 6: Update `analysis/__init__.py`**

Replace `from ._mad_outlier import MADOutlierRemover` and `from ._tukey_outlier import TukeyOutlierRemover` with a single:

```python
from .filter import MADOutlierRemover, TukeyOutlierRemover
```

(Keep them in `__all__` — unchanged.)

- [ ] **Step 7: Update the one direct-submodule test**

`tests/unit/analysis/test_log_growth_model.py:554`:
`from phenotypic.analysis._tukey_outlier import TukeyOutlierRemover`
→ `from phenotypic.analysis.filter._tukey_outlier import TukeyOutlierRemover`

- [ ] **Step 8: Run the analysis tests**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/analysis -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/analysis tests/unit/analysis/test_mad_outlier.py tests/unit/analysis/test_log_growth_model.py
git status   # confirm ONLY the above are staged
git commit -m "refactor(analysis): move outlier filters into analysis.filter"
```

---

## Phase 3 — `EdgeCorrection` ABC + `edge/` subpackage

### Task 3: Create the `EdgeCorrection` abstract base

**Files:**
- Create: `src/phenotypic/analysis/abc_/_edge_correction.py`
- Modify: `src/phenotypic/analysis/abc_/__init__.py`
- Test: `tests/unit/analysis/abc_/test_edge_correction_base.py` (new)

**Interfaces:**
- Consumes: `phenotypic.analysis.abc_._set_analyzer.SetAnalyzer`.
- Produces: `phenotypic.analysis.abc_.EdgeCorrection` — abstract `SetAnalyzer` subclass with fields `time_label`/`nrows`/`ncols`/`connectivity`, validators `_validate_connectivity`/`_validate_grid_dim`, `_original_data` PrivateAttr, staticmethod `_surrounded_positions(...)`, concrete template `analyze(data)`, and abstract `_group_config(self) -> dict`. (The per-group worker `_apply2group_func` stays abstract via `SetAnalyzer`.)

**Design note — why `_group_config`, not a per-group hook:** `EdgeCorrector.analyze` sets `self._original_data = data` (the full frame) and dispatches per-group via the **staticmethod** `_apply2group_func` precisely so joblib does not pickle `self` (and the full frame) into every task. The template therefore keeps the static-worker dispatch and factors out only the config dict via the abstract `_group_config()`. Do **not** route the parallel dispatch through a bound instance method.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/analysis/abc_/test_edge_correction_base.py`:

```python
"""Contract tests for the EdgeCorrection abstract base."""
import abc

import pytest

from phenotypic.analysis.abc_ import EdgeCorrection, SetAnalyzer


def test_edge_correction_is_abstract_setanalyzer():
    assert issubclass(EdgeCorrection, SetAnalyzer)
    assert issubclass(EdgeCorrection, abc.ABC)
    with pytest.raises(TypeError):
        EdgeCorrection(on="Shape_Area", groupby=["Metadata_Strain"])


def test_edge_correction_validates_grid():
    class _Concrete(EdgeCorrection):
        def _group_config(self):
            return {}

        @staticmethod
        def _apply2group_func(group, **config):
            return group

    with pytest.raises(ValueError):
        _Concrete(on="Shape_Area", groupby=["g"], connectivity=5)
    with pytest.raises(ValueError):
        _Concrete(on="Shape_Area", groupby=["g"], nrows=0)
    ok = _Concrete(on="Shape_Area", groupby=["g"], nrows=8, ncols=12, connectivity=8)
    assert ok.ncols == 12
```

- [ ] **Step 2: Run it to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/analysis/abc_/test_edge_correction_base.py -q`
Expected: FAIL (`ImportError: cannot import name 'EdgeCorrection'`).

- [ ] **Step 3: Write `abc_/_edge_correction.py`**

Create `src/phenotypic/analysis/abc_/_edge_correction.py`:

```python
from __future__ import annotations

import abc
from abc import ABC
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pydantic import PrivateAttr, field_validator

from phenotypic.sdk_ import ColumnRef

from ._set_analyzer import SetAnalyzer


class EdgeCorrection(SetAnalyzer, ABC):
    """Abstract base for grid-aware edge-effect correction strategies.

    Holds the grid-layout configuration (``nrows``/``ncols``/
    ``connectivity``/``time_label``) and the neighbor-topology machinery
    (:meth:`_surrounded_positions`) shared by every edge corrector, and
    drives the standard grouped-correction template in :meth:`analyze`.
    Concrete subclasses provide the per-group correction by implementing
    the static :meth:`_apply2group_func` worker plus :meth:`_group_config`,
    which supplies the kwargs forwarded to it.

    Attributes:
        time_label (str): Column holding the time point.
        nrows (int): Grid rows.
        ncols (int): Grid columns.
        connectivity (int): Neighbor pattern: 4 (orthogonal) or 8 (with
            diagonals).
    """

    time_label: ColumnRef = "Metadata_Time"
    nrows: int = 8
    ncols: int = 12
    connectivity: int = 4

    _original_data: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)

    @field_validator("connectivity")
    @classmethod
    def _validate_connectivity(cls, value: int) -> int:
        """Reject connectivity patterns other than 4 or 8."""
        if value not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {value}")
        return value

    @field_validator("nrows", "ncols")
    @classmethod
    def _validate_grid_dim(cls, value: int) -> int:
        """Reject non-positive grid dimensions."""
        if value <= 0:
            raise ValueError(f"nrows and ncols must be positive, got {value}")
        return value

    @staticmethod
    def _surrounded_positions(
            active_idx: np.ndarray | list[int],
            shape: tuple[int, int],
            connectivity: int = 4,
            min_neighbors: int | None = None,
            return_counts: bool = False,
            dtype: np.dtype = np.int64,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        # MOVE VERBATIM from the original EdgeCorrector._surrounded_positions
        # body (former _edge_correction.py lines 135-286), changing ONLY the
        # two docstring example lines that read
        # ``EdgeCorrector._surrounded_positions`` to
        # ``EdgeCorrection._surrounded_positions``.
        ...

    @abc.abstractmethod
    def _group_config(self) -> dict[str, Any]:
        """Per-group kwargs forwarded to :meth:`_apply2group_func`."""

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the edge-correction strategy group-by-group.

        Validates the frame, stores the pre-correction copy on
        ``self._original_data``, aggregates to one row per well per group,
        then dispatches each group to the static :meth:`_apply2group_func`
        worker (serial for a single group, joblib-parallel otherwise) using
        the kwargs from :meth:`_group_config`.
        """
        from phenotypic.schema import GRID

        if data is None or len(data) == 0:
            raise ValueError("Input data cannot be empty")

        self._original_data = data

        section_col = str(GRID.ROW_MAJOR_IDX)
        required_cols = set(self.groupby + [section_col, self.on])
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        groupby_cols = self.groupby + [section_col]
        if self.time_label in data:
            groupby_cols = groupby_cols + [self.time_label]

        agg_dict: dict[str, Any] = {}
        for col in data.columns:
            if col not in groupby_cols:
                agg_dict[col] = self.agg_func if col == self.on else "first"

        agg_data = data.groupby(by=groupby_cols, as_index=False).agg(agg_dict)

        config = self._group_config()
        if len(self.groupby) == 0:
            corrected_data = [self.__class__._apply2group_func(agg_data, **config)]
        else:
            grouped = agg_data.groupby(by=self.groupby, as_index=False)
            corrected_data = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.__class__._apply2group_func)(group, **config)
                    for _, group in grouped
            )

        if corrected_data:
            self._latest_measurements = pd.concat(corrected_data, ignore_index=True)
        else:
            self._latest_measurements = pd.DataFrame()
        return self._latest_measurements
```

> Fill the `_surrounded_positions` body by moving it verbatim from the original `_edge_correction.py` (lines 135–286) per the inline comment; that whole method body, including its examples, is unchanged except the two `EdgeCorrector.` → `EdgeCorrection.` doctest references.

- [ ] **Step 4: Export from `abc_/__init__.py`**

In `src/phenotypic/analysis/abc_/__init__.py`, add after the `QualityCheck` import:

```python
from ._edge_correction import EdgeCorrection
```

and add `"EdgeCorrection"` to `__all__`.

- [ ] **Step 5: Run the base contract test**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/analysis/abc_/test_edge_correction_base.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/analysis/abc_ tests/unit/analysis/abc_/test_edge_correction_base.py
git status   # confirm ONLY the above are staged
git commit -m "feat(analysis): add EdgeCorrection abstract base in abc_"
```

### Task 4: Move `EdgeCorrector` into `analysis/edge/` and subclass `EdgeCorrection`

**Files:**
- Create: `src/phenotypic/analysis/edge/__init__.py`
- Move: `_edge_correction.py` → `edge/_edge_correction.py`
- Modify: the moved file (`EdgeCorrector(EdgeCorrection)`, delete factored-up members, add `_group_config`), `analysis/__init__.py`
- Test: `tests/unit/analysis/test_edge_correction.py`

**Interfaces:**
- Consumes: `phenotypic.analysis.abc_.EdgeCorrection` (Task 3).
- Produces: `phenotypic.analysis.edge.EdgeCorrector`; still importable as `from phenotypic.analysis import EdgeCorrector`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/analysis/test_edge_correction.py`:

```python
def test_edge_subpackage_and_base():
    from phenotypic.analysis.edge import EdgeCorrector
    from phenotypic.analysis import EdgeCorrector as PublicEC
    from phenotypic.analysis.abc_ import EdgeCorrection
    import importlib, pytest
    assert EdgeCorrector is PublicEC
    assert issubclass(EdgeCorrector, EdgeCorrection)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phenotypic.analysis._edge_correction")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/analysis/test_edge_correction.py::test_edge_subpackage_and_base -q`
Expected: FAIL (`ModuleNotFoundError: phenotypic.analysis.edge`).

- [ ] **Step 3: Create the package and move the file**

```bash
mkdir -p src/phenotypic/analysis/edge
git mv src/phenotypic/analysis/_edge_correction.py src/phenotypic/analysis/edge/_edge_correction.py
```

- [ ] **Step 4: Rebase `EdgeCorrector` onto `EdgeCorrection`**

In `src/phenotypic/analysis/edge/_edge_correction.py`:
- Change the import `from .abc_ import SetAnalyzer` → `from ..abc_ import EdgeCorrection` (deeper package; `EDGE_CORRECTION` import line `from phenotypic.schema import EDGE_CORRECTION` is absolute — leave it).
- Change `class EdgeCorrector(SetAnalyzer):` → `class EdgeCorrector(EdgeCorrection):`.
- **Delete** the now-inherited members: the field declarations `time_label`, `nrows`, `ncols`, `connectivity`; the `_original_data` PrivateAttr; the `_validate_connectivity` and `_validate_grid_dim` validators; and the entire `_surrounded_positions` staticmethod. **Keep** `top_n`, `pvalue`, `_validate_top_n`, `_measurement_infoclass = EDGE_CORRECTION`.
- **Delete** the existing `analyze` method (now provided by the base template).
- **Add** a `_group_config` implementation (place it just above the retained `_apply2group_func`):

```python
    def _group_config(self) -> dict[str, Any]:
        """Per-group kwargs for :meth:`_apply2group_func` (the capping strategy)."""
        return {
            "nrows"       : self.nrows,
            "ncols"       : self.ncols,
            "top_n"       : self.top_n,
            "connectivity": self.connectivity,
            "on"          : self.on,
            "pvalue"      : self.pvalue,
            "time_label"  : self.time_label,
        }
```

- **Keep** `_apply2group_func`, `_calculate_group_stats`, `_perm_test`, `show`, `_show_collapsed`, `_show_individual`, `results` unchanged. Ensure `Any` is imported (it already is via `from typing import Any`) and that `np`/`pd`/`Parallel`/`delayed` imports that were only used by deleted methods are pruned if now unused (run ruff in Step 7 to catch).

- [ ] **Step 5: Write `edge/__init__.py`**

Create `src/phenotypic/analysis/edge/__init__.py`:

```python
"""Grid-aware edge-effect correction analyzers.

:class:`EdgeCorrector` caps edge-colony measurements that are inflated by
missing orthogonal neighbors, using the grid topology and abstract template
provided by :class:`~phenotypic.analysis.abc_.EdgeCorrection`.
"""

from ._edge_correction import EdgeCorrector

__all__ = ["EdgeCorrector"]
```

- [ ] **Step 6: Update `analysis/__init__.py`**

`from ._edge_correction import EdgeCorrector` → `from .edge import EdgeCorrector`.

- [ ] **Step 7: Lint, then run the edge + serialization tests**

Run: `uv run ruff check --fix src/phenotypic/analysis/edge/_edge_correction.py`
Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/analysis/test_edge_correction.py tests/unit/core/test_pipeline_analyze.py -q`
Expected: PASS (edge-correction algorithm output unchanged; pipeline analyze round-trip intact).

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/analysis tests/unit/analysis/test_edge_correction.py
git status   # confirm ONLY the above are staged
git commit -m "refactor(analysis): move EdgeCorrector into analysis.edge on EdgeCorrection base"
```

---

## Phase 4 — Registry category

### Task 5: Add the `"Edge Correction"` registry category

**Files:**
- Modify: `src/phenotypic/gui/_operation_registry.py`
- Modify (CI gate): `src/phenotypic/gui/FEATURES.md`
- Test: `tests/unit/gui/test_operation_registry.py`, `tests/unit/gui/test_param_forms.py`

**Interfaces:**
- Consumes: `phenotypic.analysis.abc_.EdgeCorrection`.
- Produces: `registry.get("EdgeCorrector").category == "Edge Correction"`; `EdgeCorrector` excluded from `"Filter"`/`"Model"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/gui/test_operation_registry.py`:

```python
class TestEdgeCorrectionCategory:
    def test_edge_corrector_is_edge_category(self):
        reg = OperationRegistry()
        reg.discover()
        info = reg.get("EdgeCorrector")
        assert info is not None
        assert info.category == "Edge Correction"
        filter_names = {i.name for i in reg.get_by_category("Filter")}
        model_names = {i.name for i in reg.get_by_category("Model")}
        assert "EdgeCorrector" not in filter_names
        assert "EdgeCorrector" not in model_names
        assert "EdgeCorrector" in {i.name for i in reg.get_by_category("Edge Correction")}
```

(`OperationRegistry` is already imported at the top of that test module; match the existing import name.)

- [ ] **Step 2: Run it to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/gui/test_operation_registry.py::TestEdgeCorrectionCategory -q`
Expected: FAIL (`EdgeCorrector` is `"Filter"`).

- [ ] **Step 3: Add the registry branch**

In `src/phenotypic/gui/_operation_registry.py`, inside `_discover_analyzers`, update the import line (≈246):

`from phenotypic.analysis.abc_ import ModelFitter, QualityCheck, SetAnalyzer`
→ `from phenotypic.analysis.abc_ import EdgeCorrection, ModelFitter, QualityCheck, SetAnalyzer`

And update the category cascade (≈257–262) to:

```python
            if issubclass(obj, QualityCheck):
                category = "quality_check"
            elif issubclass(obj, ModelFitter):
                category = "Model"
            elif issubclass(obj, EdgeCorrection):
                category = "Edge Correction"
            else:
                category = "Filter"
```

(`EdgeCorrection` is abstract and not re-exported at the `analysis` top level, so it never appears in the `inspect.getmembers(analysis_module)` walk — no exclusion-tuple change needed.)

- [ ] **Step 4: Update the existing category-assertion tests**

`tests/unit/gui/test_param_forms.py` `test_edge_corrector_registered` (≈line 217): `assert info.category == "Filter"` → `assert info.category == "Edge Correction"`.

In `tests/unit/gui/test_operation_registry.py`, find any assertion that `EdgeCorrector` is in `get_by_category("Filter")` (≈line 396 area) and move the expectation to `"Edge Correction"`; the parametrized param-existence rows (`("EdgeCorrector", "on", False)` etc., ≈227–229) are category-agnostic and stay.

- [ ] **Step 5: Add the FEATURES.md registry-discovery note**

In `src/phenotypic/gui/FEATURES.md`: (a) find the registry-discovery row that enumerates analyzer categories (currently names Filter/Model/quality_check) and add `Edge Correction`; (b) if the existing "Filter section stack" row's description names `EdgeCorrector` as an example filter, remove that mention (it is now an edge corrector). (Full section-stack row is added in Task 8; this step keeps the gate satisfied for this registry-only change. This Task touches `gui/` so FEATURES.md must change in the same commit.)

- [ ] **Step 6: Run the registry + param-form tests**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/gui/test_operation_registry.py tests/unit/gui/test_param_forms.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/gui/_operation_registry.py src/phenotypic/gui/FEATURES.md tests/unit/gui/test_operation_registry.py tests/unit/gui/test_param_forms.py
git status   # confirm ONLY the above are staged
git commit -m "feat(gui): register EdgeCorrection subclasses under 'Edge Correction' category"
```

---

## Phase 5 — GUI Edge Correction section

### Task 6: Add `edge` ids + partition helper

**Files:**
- Modify: `src/phenotypic/gui/analysis/_ids.py`, `src/phenotypic/gui/analysis/_layout.py`
- Test: `tests/unit/gui/analysis/test_edge_partition.py` (new)

**Interfaces:**
- Produces: `ids.ANALYSIS_EDGE_STACK`, `ids.ANALYSIS_EDGE_ADD_DROPDOWN`, `ids.edge_section_id(index)`; widened `SectionKind`/`InstantiationKind`/`PlotSectionKind` Literals; `_layout.filter_items_for_kind(pipeline, kind, registry=None) -> list[tuple[str, Any]]`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/gui/analysis/test_edge_partition.py`:

```python
"""The filter/edge GUI partition over the shared pipeline _filters dict."""
from phenotypic.analysis import EdgeCorrector, TukeyOutlierRemover
from phenotypic import ImagePipeline
from phenotypic.gui.analysis import _ids as ids
from phenotypic.gui.analysis._layout import filter_items_for_kind


def _pipeline_with_both():
    p = ImagePipeline()
    p.set_filters({
        "t0": TukeyOutlierRemover(on="Shape_Area", groupby=["Metadata_Strain"]),
        "e0": EdgeCorrector(on="Shape_Area", groupby=["Metadata_Strain"]),
        "t1": TukeyOutlierRemover(on="Shape_Area", groupby=["Metadata_Strain"]),
    })
    return p


def test_partition_splits_by_category():
    p = _pipeline_with_both()
    filt = filter_items_for_kind(p, "filter")
    edge = filter_items_for_kind(p, "edge")
    assert [k for k, _ in filt] == ["t0", "t1"]
    assert [k for k, _ in edge] == ["e0"]


def test_edge_ids_exist():
    assert ids.ANALYSIS_EDGE_STACK == "analysis-edge-stack"
    assert ids.ANALYSIS_EDGE_ADD_DROPDOWN == "analysis-edge-add-dropdown"
    assert ids.edge_section_id(2)["type"] == "analysis-edge-section"
```

(Confirm the `ImagePipeline` import path against an existing analysis GUI test, e.g. `tests/integration/gui/test_analysis_column_dropdowns.py`, and match it.)

- [ ] **Step 2: Run it to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/gui/analysis/test_edge_partition.py -q`
Expected: FAIL (`AttributeError: ANALYSIS_EDGE_STACK` / `ImportError: filter_items_for_kind`).

- [ ] **Step 3: Widen the Literals + add ids in `_ids.py`**

```python
SectionKind = Literal["post", "filter", "edge"]
InstantiationKind = Literal["post", "filter", "model", "edge"]
PlotSectionKind = Literal["filter", "model", "edge"]
```

Add constants next to the filter ones:

```python
#: Container holding the edge-correction section accordion stack.
ANALYSIS_EDGE_STACK = "analysis-edge-stack"

#: Dropdown to add a new edge corrector to the chain.
ANALYSIS_EDGE_ADD_DROPDOWN = "analysis-edge-add-dropdown"
```

Add the id function next to `filter_section_id`:

```python
def edge_section_id(index: int) -> dict[str, str | int]:
    """Pattern-matching ID for one edge-correction section in the stack."""
    return {"type": "analysis-edge-section", "index": index}
```

Add `"ANALYSIS_EDGE_STACK"`, `"ANALYSIS_EDGE_ADD_DROPDOWN"`, `"edge_section_id"` to `__all__`.

- [ ] **Step 4: Add the partition helper in `_layout.py`**

After `_choices_for_category` (≈line 55) add:

```python
#: Registry category whose analyzers render in the dedicated edge stack.
_EDGE_CATEGORY = "Edge Correction"


def filter_items_for_kind(
    pipeline: Any,
    kind: str,
    registry: OperationRegistry | None = None,
) -> list[tuple[str, Any]]:
    """Split the shared ``pipeline._filters`` dict by GUI section kind.

    The pipeline stores every non-model ``SetAnalyzer`` in one
    ``get_filters()`` dict; the GUI shows outlier filters and edge
    correctors in separate stacks. This returns the ordered
    ``(key, instance)`` sublist whose registry category maps to *kind*
    (``"edge"`` for ``"Edge Correction"``, ``"filter"`` for everything
    else). Local list position is the section's stable index for the
    remove/edit/preview callbacks.
    """
    registry = registry or get_registry()
    out: list[tuple[str, Any]] = []
    for key, inst in pipeline.get_filters().items():
        info = registry.get(type(inst).__name__)
        category = info.category if info is not None else "Filter"
        item_kind = "edge" if category == _EDGE_CATEGORY else "filter"
        if item_kind == kind:
            out.append((key, inst))
    return out
```

Confirm `Any` and `OperationRegistry` are already imported in `_layout.py` (they are — `build_section_stack` uses both).

- [ ] **Step 5: Run the partition test**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/gui/analysis/test_edge_partition.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/analysis/_ids.py src/phenotypic/gui/analysis/_layout.py tests/unit/gui/analysis/test_edge_partition.py
git status   # confirm ONLY the above are staged
git commit -m "feat(gui): edge section ids + filter/edge partition helper"
```

### Task 7: Render the edge stack in the layout

**Files:**
- Modify: `src/phenotypic/gui/analysis/_layout.py`
- Test: `tests/unit/gui/analysis/test_edge_partition.py` (extend)

**Interfaces:**
- Consumes: `filter_items_for_kind` (Task 6).
- Produces: `_build_edge_panel(recipe, *, columns_provider=None)`; `build_section_stack(..., "edge", ...)` yields edge cards; the pipeline header reports a separate edge count.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/gui/analysis/test_edge_partition.py`:

```python
class _Recipe:
    """Minimal stand-in: build_section_stack only reads recipe.pipeline."""
    def __init__(self, pipeline):
        self.pipeline = pipeline


def test_build_section_stack_edge_vs_filter():
    from phenotypic.gui.analysis._layout import build_section_stack

    recipe = _Recipe(_pipeline_with_both())
    edge_cards = build_section_stack(ids.ANALYSIS_EDGE_STACK, "edge", recipe)
    filter_cards = build_section_stack(ids.ANALYSIS_FILTER_STACK, "filter", recipe)
    assert len(edge_cards) == 1
    assert len(filter_cards) == 2
```

(`build_section_stack` reads only `recipe.pipeline` — verified at `_layout.py:432` — so the stub is sufficient and avoids depending on the `RecipeState` dataclass constructor.)

- [ ] **Step 2: Run it to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/gui/analysis/test_edge_partition.py::test_build_section_stack_edge_vs_filter -q`
Expected: FAIL (`build_section_stack` returns `[]` for `"edge"` and all 3 for `"filter"`).

- [ ] **Step 3: Update `build_section_stack` partition + branches**

In `_layout.py` `build_section_stack`, replace the item-selection block (≈432–437):

```python
    if kind == "post":
        items = list(pipeline.get_post().items())
    elif kind == "filter":
        items = list(pipeline.get_filters().items())
    else:
        return []
```

with:

```python
    if kind == "post":
        items = list(pipeline.get_post().items())
    elif kind in ("filter", "edge"):
        items = filter_items_for_kind(pipeline, kind, registry)
    else:
        return []
```

Update the plot-controls branch (≈461): `if kind == "filter":` → `if kind in ("filter", "edge"):`, and inside it pass `kind` (not the literal `"filter"`):

```python
        if kind in ("filter", "edge"):
            body = [
                body,
                plot_controls_form(kind, index, instance, plot_prefs),
            ]
```

Update the section-id selector (≈494–496):

```python
                id=ids.post_section_id(index)
                if kind == "post"
                else (
                    ids.edge_section_id(index)
                    if kind == "edge"
                    else ids.filter_section_id(index)
                ),
```

Note `registry` is resolved a few lines above via `if registry is None: registry = get_registry()`; pass that same `registry` into `filter_items_for_kind` (move the `registry` resolution above the item-selection block so it is non-None when the partition runs).

- [ ] **Step 4: Add `_build_edge_panel` and mount it**

After `_build_filter_panel` add:

```python
def _build_edge_panel(
    recipe: "RecipeState",
    *,
    columns_provider: Optional[ColumnsProvider] = None,
) -> html.Div:
    return _build_section_panel(
        title="Edge Correction",
        section_label="edge",
        choices=_choices_for_category("Edge Correction"),
        add_dropdown_id=ids.ANALYSIS_EDGE_ADD_DROPDOWN,
        stack_id=ids.ANALYSIS_EDGE_STACK,
        recipe=recipe,
        columns_provider=columns_provider,
    )
```

In `build_app_layout`, insert it after the filter panel (line 95):

```python
            _build_filter_panel(recipe, columns_provider=columns_provider),
            _build_edge_panel(recipe, columns_provider=columns_provider),
            _build_model_panel(recipe, columns_provider=columns_provider),
```

- [ ] **Step 5: Split the header count**

In `pipeline_header_children` (≈226–233), replace:

```python
    n_filters = len(pipeline.get_filters())
```

with:

```python
    n_edge = len(filter_items_for_kind(pipeline, "edge"))
    n_filters = len(filter_items_for_kind(pipeline, "filter"))
```

and update the summary string to add the edge chip:

```python
    summary = (
        f"{len(pipeline.get_ops())} ops · {len(pipeline.get_meas())} meas · "
        f"{n_post} post · {n_filters} filters · {n_edge} edge · "
        f"model: {model_name}"
    )
```

- [ ] **Step 6: Run the layout tests**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/unit/gui/analysis -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/gui/analysis/_layout.py tests/unit/gui/analysis/test_edge_partition.py
git status   # confirm ONLY the above are staged
git commit -m "feat(gui): render dedicated Edge Correction section stack + header chip"
```

### Task 8: Wire the edge callbacks

**Files:**
- Modify: `src/phenotypic/gui/analysis/_callbacks.py`
- Modify (CI gate): `src/phenotypic/gui/FEATURES.md`, `src/phenotypic/gui/WORKFLOWS.md`
- Test: `tests/integration/gui/test_analysis_edge_section.py` (new)

**Interfaces:**
- Consumes: `filter_items_for_kind`, `ids.ANALYSIS_EDGE_*`, the `"edge"` kind (Tasks 6–7).
- Produces: add/remove/edit/preview of edge sections persist into the shared `_filters` dict; `_resolve_preview_node(recipe, "edge", i)` returns the i-th edge corrector.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/gui/test_analysis_edge_section.py`:

```python
"""Edge-section add/remove/preview wiring over the shared _filters dict."""
from phenotypic.analysis import EdgeCorrector, TukeyOutlierRemover
from phenotypic import ImagePipeline
from phenotypic.gui.analysis._callbacks import _resolve_preview_node


class _Recipe:
    def __init__(self, pipeline):
        self.pipeline = pipeline


def test_resolve_preview_node_partitions_edge_and_filter():
    p = ImagePipeline()
    p.set_filters({
        "t0": TukeyOutlierRemover(on="Shape_Area", groupby=["Metadata_Strain"]),
        "e0": EdgeCorrector(on="Shape_Area", groupby=["Metadata_Strain"]),
    })
    recipe = _Recipe(p)
    edge = _resolve_preview_node(recipe, "edge", 0)
    filt = _resolve_preview_node(recipe, "filter", 0)
    assert isinstance(edge, EdgeCorrector)
    assert isinstance(filt, TukeyOutlierRemover)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/integration/gui/test_analysis_edge_section.py -q`
Expected: FAIL (`_resolve_preview_node` returns `None` for `"edge"`).

- [ ] **Step 3: Split `_FILTER_DEFAULTS`; add `_EDGE_DEFAULTS`; extend kind dicts**

In `_callbacks.py`, change `_FILTER_DEFAULTS` (66–69) to drop `EdgeCorrector`, and add `_EDGE_DEFAULTS`:

```python
_FILTER_DEFAULTS: dict[str, dict[str, Any]] = {
    "TukeyOutlierRemover": {"on": "Shape_Area", "groupby": ["Metadata_Strain"]},
}
_EDGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "EdgeCorrector": {"on": "Shape_Area", "groupby": ["Metadata_Strain"]},
}
```

> Keep `_FILTER_DEFAULTS` to just `TukeyOutlierRemover` — only remove the `EdgeCorrector` entry. Do **not** add `MADOutlierRemover`: `_instantiate` returns `None` for any class with no defaults entry (`_callbacks.py:668-671`), and `MADOutlierRemover` has no entry today either — adding one is out of scope for this refactor.

In `_KIND_DEFAULTS` (≈653) add `"edge": _EDGE_DEFAULTS`; in `_KIND_MODULES` (≈659) add `"edge": ModulePath.ANALYSIS`.

- [ ] **Step 4: Update `_resolve_preview_node`**

Replace the function (432–441) with the partitioned form:

```python
def _resolve_preview_node(recipe: Any, kind: Any, index: Any) -> Any:
    """Return the analyzer instance for a ``(kind, index)`` section, or None."""
    from phenotypic.gui.analysis._layout import filter_items_for_kind

    if kind in ("filter", "edge"):
        items = filter_items_for_kind(recipe.pipeline, kind)
        if isinstance(index, int) and 0 <= index < len(items):
            return items[index][1]
        return None
    if kind == "model":
        return recipe.pipeline.get_model()
    return None
```

- [ ] **Step 5: Add the `_add_edge` callback**

After `_add_filter` (after line 159) add a sibling (writes into the shared dict, rebuilds only the edge stack — adding an edge does not reindex the filter sublist):

```python
    @app.callback(
        Output(ids.ANALYSIS_EDGE_STACK, "children"),
        Output(ids.ANALYSIS_PIPELINE_HEADER, "children", allow_duplicate=True),
        Output(ids.ANALYSIS_PIPELINE_STORE, "data", allow_duplicate=True),
        Input(ids.ANALYSIS_EDGE_ADD_DROPDOWN, "value"),
        State(ids.ANALYSIS_PLOT_PREFS_STORE, "data"),
        prevent_initial_call=True,
    )
    def _add_edge(class_name: str | None, plot_prefs: dict | None):
        if not class_name:
            return no_update, no_update, no_update
        recipe = server.config[CFG_RECIPE_STATE]
        instance = _instantiate("edge", class_name)
        if instance is None:
            return no_update, no_update, no_update
        filters_dict = recipe.pipeline.get_filters()
        filters_dict[_unique_key(filters_dict, class_name)] = instance
        recipe.pipeline.set_filters(filters_dict)
        recipe.save()
        return (
            build_section_stack(
                ids.ANALYSIS_EDGE_STACK, "edge", recipe,
                columns_provider=_columns_provider,
                plot_prefs=plot_prefs,
            ),
            _pipeline_summary(recipe),
            recipe.last_json,
        )
```

- [ ] **Step 6: Extend `_remove_section` for `"edge"`**

Add the edge-stack output to the callback decorator (after the FILTER_STACK output, ≈203):

```python
        Output(ids.ANALYSIS_EDGE_STACK, "children", allow_duplicate=True),
```

**Arity fix (required):** the decorator now has **5** outputs (POST, FILTER, EDGE, HEADER, STORE). Update **every** early-return guard in `_remove_section` (the three at ≈218–219, 221–222, 224–225 that currently return `no_update, no_update, no_update, no_update`) to return **five** `no_update` values. Dash raises `ValueError` on any return whose length ≠ 5.

Replace the body's selection + write-back (231–261). The new body partitions the shared dict by kind, maps the local index to the global key, mutates the full dict, and rebuilds post + filter + edge:

```python
        recipe = server.config[CFG_RECIPE_STATE]
        if kind == "post":
            post_dict = recipe.pipeline.get_post()
            items = list(post_dict.items())
            if not (0 <= index < len(items)):
                return no_update, no_update, no_update, no_update, no_update
            items.pop(index)
            recipe.pipeline.set_post(dict(items))
        elif kind in ("filter", "edge"):
            from phenotypic.gui.analysis._layout import filter_items_for_kind

            sub = filter_items_for_kind(recipe.pipeline, kind)
            if not (0 <= index < len(sub)):
                return no_update, no_update, no_update, no_update, no_update
            key = sub[index][0]
            full = recipe.pipeline.get_filters()
            del full[key]
            recipe.pipeline.set_filters(full)
        else:
            return no_update, no_update, no_update, no_update, no_update
        recipe.save()

        return (
            build_section_stack(
                ids.ANALYSIS_POST_STACK, "post", recipe,
                columns_provider=_columns_provider,
            ),
            build_section_stack(
                ids.ANALYSIS_FILTER_STACK, "filter", recipe,
                columns_provider=_columns_provider,
                plot_prefs=plot_prefs,
            ),
            build_section_stack(
                ids.ANALYSIS_EDGE_STACK, "edge", recipe,
                columns_provider=_columns_provider,
                plot_prefs=plot_prefs,
            ),
            _pipeline_summary(recipe),
            recipe.last_json,
        )
```

(The return tuple now has 5 values — order: POST, FILTER, EDGE, HEADER, STORE — matching the decorator's output order.)

- [ ] **Step 7: Extend `_on_param_edit` + `_apply_param_edit` for `"edge"`**

In `_on_param_edit`, add the edge output to the decorator (after FILTER_STACK, ≈271):

```python
        Output(ids.ANALYSIS_EDGE_STACK, "children", allow_duplicate=True),
```

**Arity fix (required):** the decorator now has **6** outputs (POST, FILTER, EDGE, MODEL, HEADER, STORE). Update **every** early-return guard in `_on_param_edit` (the three at ≈295, 298, 303 that currently return five `no_update` values) to return **six** `no_update` values.

Add `edge_out: Any = no_update` beside the other `*_out` locals, an `elif kind == "edge":` arm that rebuilds the edge stack (mirror the `"filter"` arm with `ids.ANALYSIS_EDGE_STACK`, `"edge"`), and add `edge_out` to the return tuple in POST, FILTER, EDGE, MODEL, HEADER, STORE order. The new return:

```python
        return (
            post_out,
            filter_out,
            edge_out,
            model_out,
            _pipeline_summary(recipe),
            recipe.last_json,
        )
```

In `_apply_param_edit`, replace the `elif kind == "filter":` selection arm (493–495) and the write-back arm (565–567) to use the partition helper:

```python
    elif kind in ("filter", "edge"):
        from phenotypic.gui.analysis._layout import filter_items_for_kind

        section_dict = pipeline.get_filters()
        items = filter_items_for_kind(pipeline, kind)
```

and write-back:

```python
    elif kind in ("filter", "edge"):
        section_dict[section_key] = new_instance  # type: ignore[index]
        pipeline.set_filters(section_dict)  # type: ignore[arg-type]
```

(`section_key` already comes from `items[index]`, which is now the partitioned sublist, so the key maps correctly into the full `get_filters()` dict.)

- [ ] **Step 8: Update FEATURES.md + WORKFLOWS.md**

In `gui/FEATURES.md`, add an "Edge Correction section stack" row mirroring the "Filter section stack" row: id `#analysis-edge-stack` + `_choices_for_category("Edge Correction")`, status `✅ shipping`, `Test ref` `tests/integration/gui/test_analysis_edge_section.py::test_resolve_preview_node_partitions_edge_and_filter`. Update the pipeline-header-summary row to mention the edge chip.
In `gui/WORKFLOWS.md`, widen the analysis workflow row description to "post / filter / edge / model section authoring".

- [ ] **Step 9: Lint + run the GUI analysis suite**

Run: `uv run ruff check --fix src/phenotypic/gui/analysis/`
Run: `uv run mypy src/phenotypic/gui/analysis`
Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/integration/gui/test_analysis_edge_section.py tests/unit/gui/analysis -q`
Expected: PASS; mypy clean (the widened Literals flow through).

- [ ] **Step 10: Commit**

```bash
git add src/phenotypic/gui/analysis/_callbacks.py src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md tests/integration/gui/test_analysis_edge_section.py
git status   # confirm ONLY the above are staged
git commit -m "feat(gui): wire add/remove/edit/preview for the Edge Correction section"
```

### Task 9: Update integration/e2e tests that assumed EdgeCorrector in the filter stack

**Files:**
- Modify: `tests/integration/gui/test_analysis_column_dropdowns.py`, `tests/integration/gui/test_analysis_plot_preview.py`
- Test: those files

- [ ] **Step 1: Run them to see the failures**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/integration/gui/test_analysis_column_dropdowns.py tests/integration/gui/test_analysis_plot_preview.py -q`
Expected: FAIL — `prefix.startswith("analysis-filter")` and `_resolve_preview_node(..., "filter", 0)`/`kinds == {"filter","model"}` assertions no longer hold for the EdgeCorrector fixture.

- [ ] **Step 2: Update the assertions**

In `test_analysis_column_dropdowns.py`: where the EdgeCorrector-backed section's widget prefix is asserted (≈124, 166), change `analysis-filter` → `analysis-edge`. If the fixture also adds a real outlier filter, split the assertions per kind.
In `test_analysis_plot_preview.py`: change `_resolve_preview_node(recipe, "filter", 0)` (≈182) for the EdgeCorrector to `"edge"`; update the kinds set (≈140) from `{"filter", "model"}` to the set the fixture now produces (`{"edge", "model"}`, or `{"filter","edge","model"}` if a filter is also present); update the `p["kind"] == "filter"` filter-name extraction (≈128) accordingly.

- [ ] **Step 3: Run them to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest tests/integration/gui/test_analysis_column_dropdowns.py tests/integration/gui/test_analysis_plot_preview.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/gui/test_analysis_column_dropdowns.py tests/integration/gui/test_analysis_plot_preview.py
git status   # confirm ONLY the above are staged
git commit -m "test(gui): move EdgeCorrector section expectations to the edge stack"
```

> **Verified no change needed:** `tests/e2e/gui/test_analysis_app.py::test_analysis_standalone_renders_pipeline_header` only asserts `#analysis-page` is visible (no header-count text), and `test_pipeline_json_seeded_on_disk`'s `assert "filters" in config` stays valid because edge correctors serialize into the `filters` dict. Run that file in the Task 12 sweep to confirm.

### Task 10: Refresh GUI tutorial screenshots

**Files:**
- Modify: regenerated PNGs under `docs/source/tutorials/gui/` (commit all)

- [ ] **Step 1: Regenerate**

Run: `QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui python scripts/capture_gui_tutorial_screenshots.py`

- [ ] **Step 2: Commit the full set**

Per CLAUDE.md, commit **all** regenerated PNGs (do not cherry-pick the collateral font-noise churn):

```bash
git add docs/source/tutorials/gui
git status   # confirm ONLY tutorial PNGs are staged
git commit -m "docs(gui): refresh tutorial screenshots for Edge Correction section"
```

---

## Phase 6 — Docs + final verification

### Task 11: API-reference rst stubs + toctree

**Files:**
- Create: `docs/source/api_reference/phenotypic.analysis.filter.rst`, `docs/source/api_reference/phenotypic.analysis.edge.rst`
- Modify: `docs/source/api_reference/phenotypic.analysis.rst`

- [ ] **Step 1: Create `phenotypic.analysis.filter.rst`**

```rst
phenotypic.analysis.filter package
==================================

.. automodule:: phenotypic.analysis.filter
   :members:
   :show-inheritance:
   :undoc-members:
```

- [ ] **Step 2: Create `phenotypic.analysis.edge.rst`**

```rst
phenotypic.analysis.edge package
================================

.. automodule:: phenotypic.analysis.edge
   :members:
   :show-inheritance:
   :undoc-members:
```

- [ ] **Step 3: Add both to the `phenotypic.analysis.rst` toctree**

Under `Subpackages` → `.. toctree::`, add (keep alphabetical; **do not** add a private `_helper` page):

```rst
   phenotypic.analysis.abc_
   phenotypic.analysis.edge
   phenotypic.analysis.filter
   phenotypic.analysis.qc
```

- [ ] **Step 4: Commit**

```bash
git add docs/source/api_reference/phenotypic.analysis.edge.rst docs/source/api_reference/phenotypic.analysis.filter.rst docs/source/api_reference/phenotypic.analysis.rst
git commit -m "docs(api): add filter/edge subpackage reference stubs"
```

### Task 12: Final repo-wide verification

**Files:** none (verification only)

- [ ] **Step 1: Grep for stale private paths (must return nothing)**

Run:
```bash
rg -n "analysis\._(qc_math|mad_outlier|tukey_outlier|edge_correction|error_report|inoculum_prior)\b" src tests docs scripts
# Broader sweep to catch docstring/rst cross-refs the first pattern can miss:
rg -n "phenotypic\.analysis\._(qc_math|mad_outlier|tukey_outlier|edge_correction|error_report|inoculum_prior)" src tests docs
```
Expected: no matches (every reference now points at `_helper`/`filter`/`edge`, or the `_error_cutoffs` paths which are intentionally untouched). If `_error_cutoffs` appears, that is expected (it did not move) — confirm only `_error_cutoffs` remains.

- [ ] **Step 2: Lint + type-check the whole package**

Run (path-scoped — do NOT run repo-wide `ruff --fix`, it would rewrite the unrelated uncommitted files): `uv run ruff check --fix src/phenotypic/analysis src/phenotypic/gui tests/unit/analysis tests/unit/gui tests/integration/gui`
Run: `uv run mypy src/phenotypic/analysis src/phenotypic/gui`
Expected: clean.

- [ ] **Step 3: Full targeted test run**

Run:
```bash
QT_QPA_PLATFORM=offscreen uv run --group dev --group qt-test --extra gui pytest \
  tests/unit/analysis tests/unit/gui tests/integration/gui/test_analysis_edge_section.py \
  tests/integration/gui/test_analysis_column_dropdowns.py \
  tests/integration/gui/test_analysis_plot_preview.py \
  tests/e2e/gui/test_analysis_app.py \
  tests/unit/core/test_pipeline_analyze.py tests/unit/core/test_pipeline_qc_serialization.py -q
```
Expected: PASS.

- [ ] **Step 4: Final commit (if any lint/format churn)**

```bash
git add src/phenotypic/analysis src/phenotypic/gui/analysis src/phenotypic/gui/_operation_registry.py docs/source/api_reference/phenotypic.analysis.edge.rst docs/source/api_reference/phenotypic.analysis.filter.rst docs/source/api_reference/phenotypic.analysis.rst
git status   # confirm ONLY my refactor files are staged
git commit -m "chore(analysis): lint + format after subpackage reorg" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- Filters → `analysis.filter` (Task 2) ✓; helpers → `analysis._helper` (Task 1) ✓; edge → `analysis.edge` + `EdgeCorrection` ABC (Tasks 3–4) ✓; registry category (Task 5) ✓; full GUI edge section (Tasks 6–8) ✓; missed surfaces — `_qc_math` self-refs/doctests + `_max_modz` docref (Task 1 Step 6) ✓; rst stubs + toctree (Task 11) ✓; tests partition + integration/e2e (Tasks 6–9) ✓; FEATURES.md/WORKFLOWS.md (Tasks 5, 8) ✓; screenshots (Task 10) ✓; verification gates (Task 12) ✓.
- Decisions honored: models stay top-level (untouched); `ErrorCutoffFinder` untouched; `EdgeCorrection` abc_-only (Task 3 Step 4, not added to `analysis/__init__.py`); hard cutover (Task 12 grep) ✓.

**Placeholder scan:** the only `...` is the `_surrounded_positions` body in Task 3 Step 3, which carries an exact move instruction (verbatim from former lines 135–286, two doctest-prefix edits). No TBD/TODO/"handle edge cases".

**Type consistency:** `filter_items_for_kind(pipeline, kind, registry=None) -> list[tuple[str, Any]]` defined in Task 6, consumed identically in Tasks 7–8; `_group_config(self) -> dict` abstract in Task 3, implemented in Task 4; widened Literals (`SectionKind`/`InstantiationKind`/`PlotSectionKind`) defined in Task 6 and relied on in Tasks 7–8; `edge_section_id`/`ANALYSIS_EDGE_*` defined in Task 6 and used in Tasks 7–8. Return-tuple arities updated in lockstep with each callback's `Output` list (Task 8 Steps 6–7).
