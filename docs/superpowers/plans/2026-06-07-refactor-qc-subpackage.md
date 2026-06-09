# Refactor QC Subpackage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `phenotypic.qc` into `phenotypic.tools_._qc_recipe` and relocate all six `QualityCheck` subclasses from flat files in `phenotypic.analysis` into a new `phenotypic.analysis.qc` subpackage.

**Architecture:** Two independent structural moves. Phase A creates `tools_/_qc_recipe/` by copying the existing `qc/` module files verbatim, then rewires all import sites. Phase B creates `analysis/qc/` by moving the six `_*.py` QualityCheck modules there, updating their internal docstring examples, then updating `analysis/__init__.py` and all test imports. Both phases preserve every public name — no external API changes, only module paths.

**Tech Stack:** Python, uv (`uv run pytest`), ruff (`uv run ruff check --fix`)

---

## File Map

### Phase A — move phenotypic.qc → phenotypic.tools_._qc_recipe

| Action | Path |
|--------|------|
| Create | `src/phenotypic/tools_/_qc_recipe/__init__.py` |
| Create | `src/phenotypic/tools_/_qc_recipe/_recipe.py` |
| Create | `src/phenotypic/tools_/_qc_recipe/_runner.py` |
| Delete | `src/phenotypic/qc/__init__.py` |
| Delete | `src/phenotypic/qc/_recipe.py` |
| Delete | `src/phenotypic/qc/_runner.py` |
| Modify | `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py` |
| Modify | `src/phenotypic/_core/_pipeline_parts/_serializable_pipeline.py` |
| Modify | `src/phenotypic/_cli/_cli_output_manager.py` |
| Modify | `src/phenotypic/gui/results_viewer/_app.py` |
| Modify | `src/phenotypic/gui/results_viewer/_layout.py` |
| Modify | `src/phenotypic/gui/results_viewer/_qc_tab/_callbacks.py` |
| Modify | `src/phenotypic/gui/results_viewer/_qc_tab/_check_card.py` |
| Modify | `src/phenotypic/gui/results_viewer/_qc_tab/_layout.py` |
| Modify | `src/phenotypic/gui/results_viewer/_qc_tab/review/_callbacks.py` |
| Modify | `tests/e2e/gui/test_qc_review_splitter.py` |
| Modify | `tests/integration/cli/test_finalize_qc.py` |
| Modify | `tests/integration/gui/test_qc_review_recompute.py` |
| Modify | `tests/unit/core/test_pipeline_qc_serialization.py` |
| Modify | `tests/unit/gui/results_viewer/test_qc_review_layout.py` |
| Modify | `tests/unit/qc/test_qc_recipe.py` |
| Modify | `tests/unit/qc/test_run_qc.py` |

### Phase B — move QualityCheck subclasses → phenotypic.analysis.qc

| Action | Path |
|--------|------|
| Create | `src/phenotypic/analysis/qc/__init__.py` |
| Move   | `src/phenotypic/analysis/_expected_vs_detected.py` → `src/phenotypic/analysis/qc/_expected_vs_detected.py` |
| Move   | `src/phenotypic/analysis/_icc.py` → `src/phenotypic/analysis/qc/_icc.py` |
| Move   | `src/phenotypic/analysis/_max_modz.py` → `src/phenotypic/analysis/qc/_max_modz.py` |
| Move   | `src/phenotypic/analysis/_relative_mad.py` → `src/phenotypic/analysis/qc/_relative_mad.py` |
| Move   | `src/phenotypic/analysis/_replicate_agreement.py` → `src/phenotypic/analysis/qc/_replicate_agreement.py` |
| Move   | `src/phenotypic/analysis/_tukey_fraction.py` → `src/phenotypic/analysis/qc/_tukey_fraction.py` |
| Modify | `src/phenotypic/analysis/__init__.py` |
| Modify | `tests/unit/analysis/test_expected_vs_detected.py` |
| Modify | `tests/unit/analysis/test_icc.py` |
| Modify | `tests/unit/analysis/test_max_modz.py` |
| Modify | `tests/unit/analysis/test_qc_risk_scenarios.py` |
| Modify | `tests/unit/analysis/test_relative_mad.py` |
| Modify | `tests/unit/analysis/test_replicate_agreement.py` |
| Modify | `tests/unit/analysis/test_tukey_fraction.py` |

---

## Phase A: Move phenotypic.qc → phenotypic.tools_._qc_recipe

### Task A1: Create the new subpackage files

**Files:**
- Create: `src/phenotypic/tools_/_qc_recipe/__init__.py`
- Create: `src/phenotypic/tools_/_qc_recipe/_recipe.py`
- Create: `src/phenotypic/tools_/_qc_recipe/_runner.py`

- [ ] **Step 1: Create `_recipe.py`** — copy `src/phenotypic/qc/_recipe.py` verbatim (no import changes needed; its imports of `phenotypic.analysis.abc_`, `phenotypic.tools_`, and stdlib are all absolute and remain valid in the new location)

```bash
cp src/phenotypic/qc/_recipe.py src/phenotypic/tools_/_qc_recipe/_recipe.py
```

- [ ] **Step 2: Create `_runner.py`** — copy `src/phenotypic/qc/_runner.py` verbatim (all its imports are absolute or from `._recipe`, which is the relative sibling — no change needed)

```bash
cp src/phenotypic/qc/_runner.py src/phenotypic/tools_/_qc_recipe/_runner.py
```

- [ ] **Step 3: Create `__init__.py`** — re-export the same public surface as the old `phenotypic.qc`:

```python
# src/phenotypic/tools_/_qc_recipe/__init__.py
"""Pipeline-backed QC recipe types and runner, shared by CLI and GUI.

Moved from ``phenotypic.qc`` into ``phenotypic.tools_`` so the recipe
types live alongside other pipeline-support utilities.
"""

from ._recipe import (
    QC_RECIPE_FILENAME,
    QC_RECIPE_VERSION,
    QcRecipe,
    QcRecipeEntry,
    QcRecipeLoadWarning,
)

__all__ = [
    "QC_RECIPE_FILENAME",
    "QC_RECIPE_VERSION",
    "QcRecipe",
    "QcRecipeEntry",
    "QcRecipeLoadWarning",
]
```

- [ ] **Step 4: Verify new package is importable**

```bash
uv run python -c "from phenotypic.tools_._qc_recipe import QcRecipe, QcRecipeEntry, QcRecipeLoadWarning; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Commit the new files**

```bash
git add src/phenotypic/tools_/_qc_recipe/
git commit -m "refactor: add tools_._qc_recipe package (copy of phenotypic.qc)"
```

---

### Task A2: Update source-tree import sites

**Files:**
- Modify: `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py`
- Modify: `src/phenotypic/_core/_pipeline_parts/_serializable_pipeline.py`
- Modify: `src/phenotypic/_cli/_cli_output_manager.py`
- Modify: `src/phenotypic/gui/results_viewer/_app.py`
- Modify: `src/phenotypic/gui/results_viewer/_layout.py`
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/_callbacks.py`
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/_check_card.py`
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/_layout.py`
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/review/_callbacks.py`

- [ ] **Step 1: `_image_pipeline_core.py` line 38**

Old:
```python
from phenotypic.qc._recipe import QcRecipeEntry
```
New:
```python
from phenotypic.tools_._qc_recipe import QcRecipeEntry
```

- [ ] **Step 2: `_serializable_pipeline.py` — two sites**

Line 12 (TYPE_CHECKING block):
```python
# old
from phenotypic.qc._recipe import QcRecipeEntry
# new
from phenotypic.tools_._qc_recipe import QcRecipeEntry
```

Line 805 (inside `_deserialize_qc`):
```python
# old
from phenotypic.qc._recipe import QcRecipeEntry, QcRecipeLoadWarning
# new
from phenotypic.tools_._qc_recipe import QcRecipeEntry, QcRecipeLoadWarning
```

- [ ] **Step 3: `_cli_output_manager.py` line 621** (lazy import inside `finalize_post_master_outputs`)

```python
# old
from phenotypic.qc._runner import run_qc
# new
from phenotypic.tools_._qc_recipe._runner import run_qc
```

- [ ] **Step 4: `gui/results_viewer/_app.py` line 67**

```python
# old
from phenotypic.qc import QcRecipe
# new
from phenotypic.tools_._qc_recipe import QcRecipe
```

- [ ] **Step 5: `gui/results_viewer/_layout.py` line 45**

```python
# old
from phenotypic.qc import QcRecipe
# new
from phenotypic.tools_._qc_recipe import QcRecipe
```

- [ ] **Step 6: `gui/results_viewer/_qc_tab/_callbacks.py` line 70**

```python
# old
from phenotypic.qc import QcRecipe
# new
from phenotypic.tools_._qc_recipe import QcRecipe
```

- [ ] **Step 7: `gui/results_viewer/_qc_tab/_check_card.py` line 29**

```python
# old
from phenotypic.qc import QcRecipeEntry
# new
from phenotypic.tools_._qc_recipe import QcRecipeEntry
```

- [ ] **Step 8: `gui/results_viewer/_qc_tab/_layout.py` line 37**

```python
# old
from phenotypic.qc import QcRecipe, QcRecipeLoadWarning
# new
from phenotypic.tools_._qc_recipe import QcRecipe, QcRecipeLoadWarning
```

- [ ] **Step 9: `gui/results_viewer/_qc_tab/review/_callbacks.py` line 594** (lazy import)

```python
# old
from phenotypic.qc._runner import run_qc
# new
from phenotypic.tools_._qc_recipe._runner import run_qc
```

- [ ] **Step 10: Verify import graph is clean**

```bash
uv run python -c "
from phenotypic._core._pipeline_parts._image_pipeline_core import ImagePipelineCore
from phenotypic._cli._cli_output_manager import finalize_post_master_outputs
print('core + cli ok')
"
uv run ruff check --fix src/phenotypic/
```

Expected: no import errors, ruff exits 0.

- [ ] **Step 11: Commit**

```bash
git add src/phenotypic/
git commit -m "refactor: rewire src import sites from phenotypic.qc to tools_._qc_recipe"
```

---

### Task A3: Update test import sites

**Files:**
- Modify: `tests/e2e/gui/test_qc_review_splitter.py` (lines 28–29)
- Modify: `tests/integration/cli/test_finalize_qc.py` (line 24)
- Modify: `tests/integration/gui/test_qc_review_recompute.py` (lines 35–36)
- Modify: `tests/unit/core/test_pipeline_qc_serialization.py` (line 29)
- Modify: `tests/unit/gui/results_viewer/test_qc_review_layout.py` (line 34)
- Modify: `tests/unit/qc/test_qc_recipe.py` (line 24)
- Modify: `tests/unit/qc/test_run_qc.py` (lines 28–29)

Apply these replacements in each file:

| Old | New |
|-----|-----|
| `from phenotypic.qc import QcRecipeEntry` | `from phenotypic.tools_._qc_recipe import QcRecipeEntry` |
| `from phenotypic.qc import QcRecipe` | `from phenotypic.tools_._qc_recipe import QcRecipe` |
| `from phenotypic.qc import QcRecipe, QcRecipeLoadWarning` | `from phenotypic.tools_._qc_recipe import QcRecipe, QcRecipeLoadWarning` |
| `from phenotypic.qc._recipe import QcRecipe, QcRecipeEntry` | `from phenotypic.tools_._qc_recipe import QcRecipe, QcRecipeEntry` |
| `from phenotypic.qc._recipe import QcRecipeEntry` | `from phenotypic.tools_._qc_recipe import QcRecipeEntry` |
| `from phenotypic.qc._runner import run_qc` | `from phenotypic.tools_._qc_recipe._runner import run_qc` |

- [ ] **Step 1: Run the tests to confirm they fail only on import, not logic**

```bash
uv run pytest tests/unit/qc/ tests/unit/core/test_pipeline_qc_serialization.py -x --tb=short 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'phenotypic.qc'` (confirms old path is gone — only fails once old package is deleted; until then, both exist and tests should still pass). If the old `phenotypic.qc` package is still present, tests pass here — which is fine, proceed to update imports.

- [ ] **Step 2: Apply all import replacements**

In each file listed above, make the replacements from the table.

- [ ] **Step 3: Run affected tests**

```bash
uv run pytest tests/unit/qc/ tests/unit/core/test_pipeline_qc_serialization.py tests/unit/gui/results_viewer/test_qc_review_layout.py -x --tb=short
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "refactor: rewire test import sites from phenotypic.qc to tools_._qc_recipe"
```

---

### Task A4: Delete the old phenotypic.qc package

**Files:**
- Delete: `src/phenotypic/qc/__init__.py`
- Delete: `src/phenotypic/qc/_recipe.py`
- Delete: `src/phenotypic/qc/_runner.py`
- Delete: `src/phenotypic/qc/` (directory)

- [ ] **Step 1: Remove the old package**

```bash
git rm -r src/phenotypic/qc/
```

- [ ] **Step 2: Confirm nothing still imports from phenotypic.qc**

```bash
grep -rn "phenotypic\.qc" src/ tests/ --include="*.py" | grep -v "__pycache__" | grep -v "tools_._qc_recipe"
```

Expected: no output (or only comment/docstring references).

- [ ] **Step 3: Run the full relevant test suite**

```bash
uv run pytest tests/unit/qc/ tests/unit/core/test_pipeline_qc_serialization.py tests/integration/cli/test_finalize_qc.py -x --tb=short
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: delete phenotypic.qc (fully replaced by tools_._qc_recipe)"
```

---

## Phase B: Move QualityCheck subclasses → phenotypic.analysis.qc

### Task B1: Create phenotypic.analysis.qc subpackage

**Files:**
- Create: `src/phenotypic/analysis/qc/__init__.py`
- Create: `src/phenotypic/analysis/qc/_expected_vs_detected.py`
- Create: `src/phenotypic/analysis/qc/_icc.py`
- Create: `src/phenotypic/analysis/qc/_max_modz.py`
- Create: `src/phenotypic/analysis/qc/_relative_mad.py`
- Create: `src/phenotypic/analysis/qc/_replicate_agreement.py`
- Create: `src/phenotypic/analysis/qc/_tukey_fraction.py`

- [ ] **Step 1: Copy all six module files into the new subpackage**

```bash
mkdir -p src/phenotypic/analysis/qc
cp src/phenotypic/analysis/_expected_vs_detected.py src/phenotypic/analysis/qc/_expected_vs_detected.py
cp src/phenotypic/analysis/_icc.py                 src/phenotypic/analysis/qc/_icc.py
cp src/phenotypic/analysis/_max_modz.py             src/phenotypic/analysis/qc/_max_modz.py
cp src/phenotypic/analysis/_relative_mad.py         src/phenotypic/analysis/qc/_relative_mad.py
cp src/phenotypic/analysis/_replicate_agreement.py  src/phenotypic/analysis/qc/_replicate_agreement.py
cp src/phenotypic/analysis/_tukey_fraction.py       src/phenotypic/analysis/qc/_tukey_fraction.py
```

- [ ] **Step 2: Update docstring `from` examples inside each moved file**

Each file contains a doctest `>>>` line that references its old path. Update them:

In `qc/_expected_vs_detected.py`:
```python
# old: >>> from phenotypic.analysis._expected_vs_detected import (
# new: >>> from phenotypic.analysis.qc import (
```

In `qc/_icc.py`:
```python
# old: >>> from phenotypic.analysis._icc import ICC
# new: >>> from phenotypic.analysis.qc import ICC
```

In `qc/_max_modz.py`:
```python
# old: >>> from phenotypic.analysis._max_modz import MaxModifiedZScore
# new: >>> from phenotypic.analysis.qc import MaxModifiedZScore
```

In `qc/_relative_mad.py`:
```python
# old: >>> from phenotypic.analysis._relative_mad import RelativeMAD
# new: >>> from phenotypic.analysis.qc import RelativeMAD
```

In `qc/_replicate_agreement.py`:
```python
# old: >>> from phenotypic.analysis._replicate_agreement import (
# new: >>> from phenotypic.analysis.qc import (
```

In `qc/_tukey_fraction.py`:
```python
# old: >>> from phenotypic.analysis._tukey_fraction import (
# new: >>> from phenotypic.analysis.qc import (
```

- [ ] **Step 3: Create `src/phenotypic/analysis/qc/__init__.py`**

```python
"""QualityCheck implementations for smart-QC pipeline analysis.

Each class is a :class:`~phenotypic.analysis.abc_.QualityCheck` subclass
that flags groups of colony measurements whose statistical properties
indicate data quality problems (outliers, replicate disagreement, count
mismatches, etc.).
"""

from ._expected_vs_detected import ExpectedVsDetectedCount
from ._icc import ICC
from ._max_modz import MaxModifiedZScore
from ._relative_mad import RelativeMAD
from ._replicate_agreement import ReplicateAgreement
from ._tukey_fraction import TukeyOutlierFraction

__all__ = [
    "ExpectedVsDetectedCount",
    "ICC",
    "MaxModifiedZScore",
    "RelativeMAD",
    "ReplicateAgreement",
    "TukeyOutlierFraction",
]
```

- [ ] **Step 4: Verify new subpackage is importable**

```bash
uv run python -c "
from phenotypic.analysis.qc import (
    ExpectedVsDetectedCount, ICC, MaxModifiedZScore,
    RelativeMAD, ReplicateAgreement, TukeyOutlierFraction,
)
print('analysis.qc ok')
"
```

Expected: `analysis.qc ok`

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/analysis/qc/
git commit -m "refactor: add analysis.qc subpackage (copy of analysis QualityCheck modules)"
```

---

### Task B2: Update analysis/__init__.py and test imports

**Files:**
- Modify: `src/phenotypic/analysis/__init__.py`
- Modify: `tests/unit/analysis/test_expected_vs_detected.py`
- Modify: `tests/unit/analysis/test_icc.py`
- Modify: `tests/unit/analysis/test_max_modz.py`
- Modify: `tests/unit/analysis/test_qc_risk_scenarios.py`
- Modify: `tests/unit/analysis/test_relative_mad.py`
- Modify: `tests/unit/analysis/test_replicate_agreement.py`
- Modify: `tests/unit/analysis/test_tukey_fraction.py`

- [ ] **Step 1: Update `analysis/__init__.py`** — change the six QualityCheck imports to come from the new subpackage:

```python
# old (six separate lines)
from ._expected_vs_detected import ExpectedVsDetectedCount
from ._icc import ICC
from ._max_modz import MaxModifiedZScore
from ._relative_mad import RelativeMAD
from ._replicate_agreement import ReplicateAgreement
from ._tukey_fraction import TukeyOutlierFraction

# new (one block from the subpackage)
from .qc import (
    ExpectedVsDetectedCount,
    ICC,
    MaxModifiedZScore,
    RelativeMAD,
    ReplicateAgreement,
    TukeyOutlierFraction,
)
```

The `__all__` list and every other import in `__init__.py` stay unchanged.

- [ ] **Step 2: Confirm `phenotypic.analysis` still exports all names**

```bash
uv run python -c "
from phenotypic.analysis import (
    ExpectedVsDetectedCount, ICC, MaxModifiedZScore,
    RelativeMAD, ReplicateAgreement, TukeyOutlierFraction,
)
print('analysis top-level ok')
"
```

Expected: `analysis top-level ok`

- [ ] **Step 3: Update test imports** — apply these replacements in the seven test files:

| File | Old import | New import |
|------|-----------|------------|
| `test_expected_vs_detected.py:12` | `from phenotypic.analysis._expected_vs_detected import ExpectedVsDetectedCount` | `from phenotypic.analysis.qc import ExpectedVsDetectedCount` |
| `test_icc.py:19` | `from phenotypic.analysis._icc import ICC` | `from phenotypic.analysis.qc import ICC` |
| `test_max_modz.py:16` | `from phenotypic.analysis._max_modz import MaxModifiedZScore` | `from phenotypic.analysis.qc import MaxModifiedZScore` |
| `test_qc_risk_scenarios.py:37` | `from phenotypic.analysis._replicate_agreement import ReplicateAgreement` | `from phenotypic.analysis.qc import ReplicateAgreement` |
| `test_relative_mad.py:16` | `from phenotypic.analysis._relative_mad import RelativeMAD` | `from phenotypic.analysis.qc import RelativeMAD` |
| `test_replicate_agreement.py:17` | `from phenotypic.analysis._replicate_agreement import ReplicateAgreement` | `from phenotypic.analysis.qc import ReplicateAgreement` |
| `test_tukey_fraction.py:16` | `from phenotypic.analysis._tukey_fraction import TukeyOutlierFraction` | `from phenotypic.analysis.qc import TukeyOutlierFraction` |

- [ ] **Step 4: Run the analysis tests**

```bash
uv run pytest tests/unit/analysis/ -x --tb=short
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/analysis/__init__.py tests/unit/analysis/
git commit -m "refactor: rewire analysis imports and tests to use analysis.qc subpackage"
```

---

### Task B3: Delete the old flat QualityCheck modules

**Files:**
- Delete: `src/phenotypic/analysis/_expected_vs_detected.py`
- Delete: `src/phenotypic/analysis/_icc.py`
- Delete: `src/phenotypic/analysis/_max_modz.py`
- Delete: `src/phenotypic/analysis/_relative_mad.py`
- Delete: `src/phenotypic/analysis/_replicate_agreement.py`
- Delete: `src/phenotypic/analysis/_tukey_fraction.py`

- [ ] **Step 1: Remove old files**

```bash
git rm src/phenotypic/analysis/_expected_vs_detected.py \
       src/phenotypic/analysis/_icc.py \
       src/phenotypic/analysis/_max_modz.py \
       src/phenotypic/analysis/_relative_mad.py \
       src/phenotypic/analysis/_replicate_agreement.py \
       src/phenotypic/analysis/_tukey_fraction.py
```

- [ ] **Step 2: Confirm nothing imports the old paths**

```bash
grep -rn "phenotypic\.analysis\._expected_vs_detected\|phenotypic\.analysis\._icc\|phenotypic\.analysis\._max_modz\|phenotypic\.analysis\._relative_mad\|phenotypic\.analysis\._replicate_agreement\|phenotypic\.analysis\._tukey_fraction" src/ tests/ --include="*.py" | grep -v __pycache__
```

Expected: no output (docstring examples are now updated to `analysis.qc`).

- [ ] **Step 3: Run the full QC-related test suite**

```bash
uv run pytest tests/unit/analysis/ tests/unit/qc/ tests/unit/core/test_pipeline_qc_serialization.py -x --tb=short
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: delete flat QualityCheck modules (moved to analysis.qc subpackage)"
```

---

### Task B4: Final verification

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest tests/unit/ tests/integration/ -x --tb=short -q
```

Expected: all pass (or same failures as baseline before this refactor).

- [ ] **Step 2: Confirm class-resolution still works for QcRecipe**

The `_resolve_check_class` function in `tools_._qc_recipe._recipe` walks `inspect.getmembers(phenotypic.analysis, isclass)` to find check classes by name. Since `analysis/__init__.py` still re-exports all six classes, resolution must still work.

```bash
uv run python -c "
from phenotypic.tools_._qc_recipe._recipe import _resolve_check_class
for name in ['ExpectedVsDetectedCount', 'ICC', 'MaxModifiedZScore', 'RelativeMAD', 'ReplicateAgreement', 'TukeyOutlierFraction']:
    cls = _resolve_check_class(name)
    assert cls is not None, f'FAIL: {name} not resolved'
    print(f'  {name}: {cls}')
print('all resolved ok')
"
```

Expected: all six names print their class and `all resolved ok`.

- [ ] **Step 3: Run ruff**

```bash
uv run ruff check --fix src/ tests/
```

Expected: exits 0.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "refactor: verify and clean up after qc subpackage restructure"
```
