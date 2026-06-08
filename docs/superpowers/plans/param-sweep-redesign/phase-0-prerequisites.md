# Tune Engine — Phase 0: Cross-Module Prerequisites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the four zero-new-dependency enablers that unblock the tuning engine's Phase 1 — without touching `tune/` internals (which don't exist yet) and without breaking any existing behavior.

**Architecture:** (A) generalize the existing `OperationField` serialization marker into a `polymorphic_field(base=...)` factory so scorers/strategies will later serialize exactly like operations; (B) teach the class registry to find `phenotypic.tune` classes; (C) create a small shared `LocalExecutor` parallel-map primitive in a new top-level `_execution/`; (D) freeze a golden `generate_sweep_manifest` fixture (while `sweep` still exists) for Phase 1's grid byte-compat lock. Each task is independent and self-contained.

**Tech Stack:** Python 3, pydantic v2 (`Annotated`, `BeforeValidator`/`AfterValidator`/`PlainSerializer`), `joblib` (already a dependency), `pytest`. No new third-party deps. `uv` for everything.

**Spec:** `docs/superpowers/specs/param-sweep-redesign/engine-architecture.md` §6, §7, §14a + master `2026-06-01-…-design.md` §9 (hard-cutover deprecation).

**Conventions:** `uv run pytest`, `uv run mypy src/phenotypic`, `uv run ruff check --fix`; Google-style docstrings; tests under `tests/unit/`.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/phenotypic/tools_/typing_.py` | the `polymorphic_field(base, *, marker)` factory; `OperationField` becomes its alias | **Modify** |
| `src/phenotypic/_core/_pipeline_parts/_serializable_pipeline.py` | `_find_class_in_phenotypic` gains `"phenotypic.tune"` | **Modify** (`:591-602`) |
| `src/phenotypic/tune/__init__.py` | minimal package marker so the registry can import it | **Create** |
| `src/phenotypic/_execution/__init__.py` | re-exports `Executor`, `LocalExecutor` | **Create** |
| `src/phenotypic/_execution/_protocol.py` | the `Executor` Protocol | **Create** |
| `src/phenotypic/_execution/_local.py` | `LocalExecutor` (joblib parallel-map) | **Create** |
| `scripts/capture_grid_golden_manifest.py` | one-shot golden-fixture generator | **Create** |
| `tests/fixtures/tune/grid_golden_manifest.json` | the frozen golden manifest | **Create (generated)** |
| `tests/unit/tools_/test_polymorphic_field.py` | factory + OperationField back-compat | **Create** |
| `tests/unit/core/test_registry_finds_tune.py` | registry finds `phenotypic.tune` classes | **Create** |
| `tests/unit/util/test_local_executor.py` | `LocalExecutor` parallel-map | **Create** |
| `tests/unit/tune/__init__.py` + `tests/unit/tune/test_grid_golden_manifest.py` | golden sanity lock | **Create** |

> The `tune/` package is a *stub* in Phase 0 — only `__init__.py`, no scorers/strategies (those are Phase 1). `_execution/` is created here and used by `tune` in Phase 1; `sweep` is **not** modified (it is deleted at the end of Phase 1, master §9).

---

### Task A: `polymorphic_field(base=...)` factory

Generalize `OperationField`. The serialize/deserialize helpers are already type-agnostic; only the `AfterValidator` guard (`_require_operation_value`, which hard-asserts `BaseOperation`) is base-specific. Turn it into a parameterized factory; `OperationField` becomes `polymorphic_field(base=<lazy BaseOperation>, marker=_OperationFieldMarker())`.

**Files:**
- Modify: `src/phenotypic/tools_/typing_.py` (the `OperationField` block, ~`:234-315`)
- Test: `tests/unit/tools_/test_polymorphic_field.py`

- [ ] **Step 1: Write the failing test (factory + guard parameterization + OperationField back-compat)**

```python
# tests/unit/tools_/test_polymorphic_field.py
from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from phenotypic.tools_.typing_ import OperationField, polymorphic_field


# --- A live, non-operation pydantic base to prove base-parameterization ---
class _Animal(BaseModel):
    name: str = "?"


class _Dog(_Animal):
    legs: int = 4


_AnimalField = polymorphic_field(base=_Animal)


class _AnimalHost(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pet: _AnimalField  # type: ignore[valid-type]


def test_guard_accepts_base_subclass_instance():
    # A live instance of the declared base (passed through, not dict-deserialized)
    host = _AnimalHost(pet=_Dog(name="rex"))
    assert isinstance(host.pet, _Dog)
    assert host.pet.name == "rex"


def test_guard_rejects_non_base_instance():
    class _Rock(BaseModel):
        pass

    with pytest.raises(ValidationError):
        _AnimalHost(pet=_Rock())


# --- OperationField back-compat: it must still round-trip a real operation ---
class _OpHost(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    op: OperationField  # type: ignore[valid-type]


def test_operationfield_roundtrips_real_operation():
    from phenotypic.detect import OtsuDetector

    host = _OpHost(op=OtsuDetector(ignore_zeros=True))
    dumped = host.model_dump(mode="json")
    assert dumped["op"]["class"] == "OtsuDetector"

    restored = _OpHost.model_validate(dumped)
    assert type(restored.op).__name__ == "OtsuDetector"
    assert restored.op.ignore_zeros is True


def test_operationfield_keeps_gui_marker():
    # The GUI OperationRegistry detects operation params via _OperationFieldMarker
    # in the Annotated chain. OperationField must keep it after the refactor.
    from phenotypic.tools_.typing_ import _OperationFieldMarker

    meta = OperationField.__metadata__
    assert any(isinstance(m, _OperationFieldMarker) for m in meta)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tools_/test_polymorphic_field.py -v`
Expected: FAIL — `ImportError: cannot import name 'polymorphic_field'`.

- [ ] **Step 3: Implement the factory in `typing_.py`**

In `src/phenotypic/tools_/typing_.py`, **replace** the standalone guard `_require_operation_value` (`:234-262`) and the `OperationField = Annotated[...]` definition (`:309-315`) with a base-parameterized guard + the factory + the alias. Keep `_serialize_operation_value`, `_deserialize_operation_value`, and `_OperationFieldMarker` exactly as they are (already generic).

> **Required import (do this first):** `typing_.py` currently imports `from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Tuple` — **no `Callable`**. Add `Callable` to that line. `from __future__ import annotations` keeps the string annotations runtime-safe, so the tests pass either way, but **`mypy` (Step 6) fails with `Name "Callable" is not defined`** without it.

```python
# src/phenotypic/tools_/typing_.py  (replaces _require_operation_value + OperationField)

def _make_require_value(base: "type | Callable[[], type]"):
    """Build an ``AfterValidator`` guard asserting a value is a ``base`` instance.

    ``base`` may be a concrete type or a zero-arg callable that returns the
    type (resolved lazily, so ``OperationField`` can name ``BaseOperation``
    without importing it at ``tools_`` load time — avoiding the import cycle).
    """

    def _require(value):
        resolved = base if isinstance(base, type) else base()
        if not isinstance(value, resolved):
            raise ValueError(
                f"expected an instance of {resolved.__name__}, got "
                f"{type(value).__name__}"
            )
        return value

    return _require


def polymorphic_field(base: "type | Callable[[], type]", *, marker=None):
    """A pydantic field for a polymorphic model subtree (the concrete subclass
    survives a JSON round-trip via the ``phenotypic`` class registry).

    Serializes to ``{"class": <name>, "params": {...}}`` (or the pipeline-tagged
    form for an ``ImagePipeline``) and reconstructs the concrete subclass on
    load. ``base`` constrains the accepted/validated type (a class or a lazy
    callable). Pass ``marker`` to attach a sentinel to the ``Annotated`` chain
    (e.g. the GUI's ``_OperationFieldMarker``).

    Host models must set ``model_config`` with ``arbitrary_types_allowed=True``.
    """
    core = Annotated[
        Any,
        BeforeValidator(_deserialize_operation_value),
        AfterValidator(_make_require_value(base)),
        PlainSerializer(_serialize_operation_value),
    ]
    if marker is None:
        return core
    # Annotated flattens a nested Annotated (PEP 593): the marker joins the chain.
    return Annotated[core, marker]


def _lazy_base_operation() -> type:
    from phenotypic.abc_ import BaseOperation

    return BaseOperation


#: Back-compat alias — an operation/pipeline-valued field. Identical behavior to
#: the previous ``OperationField`` (concrete subclass round-trips; the GUI
#: registry detects it via ``_OperationFieldMarker``).
OperationField = polymorphic_field(
    base=_lazy_base_operation, marker=_OperationFieldMarker()
)
```

> Note: keep the existing `_serialize_operation_value` / `_deserialize_operation_value` docstrings, but you may add one line noting they are now generic (used by any `polymorphic_field`, not just operations).

- [ ] **Step 4: Run the new test to verify it passes**

Run: `uv run pytest tests/unit/tools_/test_polymorphic_field.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the back-compat lock (existing serialization + GUI registry suites stay green)**

Run: `uv run pytest tests/unit/tools_ tests/unit/core tests/unit/gui -k "serial or registry or operation or pipeline" -q`
Expected: PASS — no regressions in operation/pipeline round-trips or the GUI `OperationRegistry`. If anything references the removed `_require_operation_value` symbol, repoint it to `_make_require_value(_lazy_base_operation)`. Search first: `grep -rn '_require_operation_value' src tests` (expected: no hits outside `typing_.py` itself).

- [ ] **Step 6: Type-check, lint, commit**

```bash
uv run mypy src/phenotypic/tools_/typing_.py && uv run ruff check --fix src/phenotypic/tools_/typing_.py tests/unit/tools_/test_polymorphic_field.py
git add src/phenotypic/tools_/typing_.py tests/unit/tools_/test_polymorphic_field.py
git commit -m "feat(tools_): generalize OperationField into polymorphic_field(base=...) factory"
```

---

### Task B: registry learns `phenotypic.tune`

`_find_class_in_phenotypic` walks a hardcoded submodule list. Add `tune` + create the stub package so `TuningSpec` can later reconstruct `Scorer`/`StrategyConfig` subclasses by name.

**Files:**
- Modify: `src/phenotypic/_core/_pipeline_parts/_serializable_pipeline.py:591-602`
- Create: `src/phenotypic/tune/__init__.py`
- Test: `tests/unit/core/test_registry_finds_tune.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_registry_finds_tune.py
from __future__ import annotations

from phenotypic._core._pipeline_parts._serializable_pipeline import (
    SerializablePipeline,
)


def test_registry_searches_phenotypic_tune(monkeypatch):
    import phenotypic.tune as tune_pkg

    class _ProbeClass:
        pass

    # Export a class under the phenotypic.tune namespace; the registry must find it.
    monkeypatch.setattr(tune_pkg, "_ProbeClass", _ProbeClass, raising=False)
    found = SerializablePipeline._find_class_in_phenotypic("_ProbeClass")
    assert found is _ProbeClass


def test_tune_package_imports():
    import phenotypic.tune  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/core/test_registry_finds_tune.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.tune'` (and the probe is not found).

- [ ] **Step 3: Create the stub package + add `tune` to the registry list**

```python
# src/phenotypic/tune/__init__.py
"""Parameter-tuning engine (in progress).

Phase 0 ships only this package marker so the serialization registry
(``_find_class_in_phenotypic``) can resolve tune classes by name once they
land in Phase 1. Public symbols are re-exported here as they are implemented.
"""
from __future__ import annotations

__all__: list[str] = []
```

In `src/phenotypic/_core/_pipeline_parts/_serializable_pipeline.py`, add `"phenotypic.tune"` to the `submodules` list (`:591-602`):

```python
        submodules = [
            "phenotypic.detect",
            "phenotypic.measure",
            "phenotypic.enhance",
            "phenotypic.refine",
            "phenotypic.grid",
            "phenotypic.correction",
            "phenotypic.analysis",
            "phenotypic.prefab",
            "phenotypic.post",
            "phenotypic.nn",
            "phenotypic.tune",
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/core/test_registry_finds_tune.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/__init__.py tests/unit/core/test_registry_finds_tune.py
git add src/phenotypic/tune/__init__.py src/phenotypic/_core/_pipeline_parts/_serializable_pipeline.py tests/unit/core/test_registry_finds_tune.py
git commit -m "feat(core): registry resolves phenotypic.tune classes + stub tune package"
```

---

### Task C: `LocalExecutor` (shared parallel-map primitive)

A tiny joblib wrapper in a new top-level `_execution/`. `sweep` is **not** refactored (it is deleted at the end of Phase 1, master §9); this is created fresh for `tune`'s Evaluator.

**Files:**
- Create: `src/phenotypic/_execution/__init__.py`, `_protocol.py`, `_local.py`
- Test: `tests/unit/util/test_local_executor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/util/test_local_executor.py
from __future__ import annotations

from phenotypic._execution import Executor, LocalExecutor


def _square(x: int) -> int:
    return x * x


def test_local_executor_maps_in_order():
    ex = LocalExecutor(n_jobs=1)
    assert ex.run(_square, [1, 2, 3, 4]) == [1, 4, 9, 16]


def test_local_executor_parallel_results_ordered():
    ex = LocalExecutor(n_jobs=2)
    assert ex.run(_square, list(range(10))) == [i * i for i in range(10)]


def test_local_executor_empty():
    assert LocalExecutor(n_jobs=1).run(_square, []) == []


def test_local_executor_satisfies_protocol():
    assert isinstance(LocalExecutor(), Executor)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/util/test_local_executor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic._execution'`.

- [ ] **Step 3: Implement the module**

```python
# src/phenotypic/_execution/_protocol.py
"""The Executor seam: run a work-fn over items in parallel.

A low-level parallel-map primitive shared by callers that need to fan a
pure function over many inputs (the tuning Evaluator over calibration
images). Orchestration (saving, logging, scoring) lives in the caller's
injected ``work`` function.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence, TypeVar, runtime_checkable

T = TypeVar("T")
R = TypeVar("R")


@runtime_checkable
class Executor(Protocol):
    """Runs ``work(item)`` for every item, returning results in input order."""

    def run(self, work: Callable[[T], R], items: Sequence[T]) -> list[R]: ...
```

```python
# src/phenotypic/_execution/_local.py
"""Local (joblib) Executor."""
from __future__ import annotations

from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class LocalExecutor:
    """Map ``work`` over ``items`` with joblib, preserving input order.

    Args:
        n_jobs: Worker count (``-1`` = all cores). Default ``-1``.

    Examples:
        >>> from phenotypic._execution import LocalExecutor
        >>> LocalExecutor(n_jobs=1).run(lambda x: x + 1, [1, 2, 3])
        [2, 3, 4]
    """

    def __init__(self, n_jobs: int = -1) -> None:
        self.n_jobs = n_jobs

    def run(self, work: Callable[[T], R], items: Sequence[T]) -> list[R]:
        if not items:
            return []
        from joblib import Parallel, delayed

        return list(
            Parallel(n_jobs=self.n_jobs)(delayed(work)(item) for item in items)
        )
```

```python
# src/phenotypic/_execution/__init__.py
"""Shared execution primitives (Local now; Slurm in tune Phase 2)."""
from __future__ import annotations

from ._local import LocalExecutor
from ._protocol import Executor

__all__ = ["Executor", "LocalExecutor"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/util/test_local_executor.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Doctest + type-check + lint + commit**

```bash
uv run pytest --doctest-modules src/phenotypic/_execution/_local.py -q
uv run mypy src/phenotypic/_execution && uv run ruff check --fix src/phenotypic/_execution tests/unit/util/test_local_executor.py
git add src/phenotypic/_execution tests/unit/util/test_local_executor.py
git commit -m "feat(_execution): add shared LocalExecutor parallel-map primitive"
```

---

### Task D: capture the grid golden fixture

Freeze `generate_sweep_manifest`'s output over a **conditional (`Presence`) config** as a committed JSON golden — **while `sweep` still exists** — so Phase 1's `GridStrategy` byte-compat lock has a target after `sweep` is deleted.

**Files:**
- Create: `scripts/capture_grid_golden_manifest.py`
- Create (generated + committed): `tests/fixtures/tune/grid_golden_manifest.json`
- Create: `tests/unit/tune/__init__.py`, `tests/unit/tune/test_grid_golden_manifest.py`

- [ ] **Step 1: Write the golden generator script**

```python
# scripts/capture_grid_golden_manifest.py
"""One-shot capture of the grid golden manifest (the Phase-1 GridStrategy lock).

Run once, WHILE `phenotypic.sweep` still exists (it is deleted at the end of
tune Phase 1). Writes a frozen `generate_sweep_manifest` output over a
representative conditional (Presence) config. Re-run only to intentionally
regenerate the golden.
"""
from __future__ import annotations

import json
from pathlib import Path

from phenotypic.sweep import Presence, Sweep, generate_sweep_manifest
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector

GOLDEN = Path(__file__).resolve().parents[1] / (
    "tests/fixtures/tune/grid_golden_manifest.json"
)


def build_golden_config():
    """A conditional config: a Presence (present/absent) + a swept detector."""
    return [
        Presence(GaussianBlur, sigma=(1.0, 2.0)),     # 2 sigmas + absent = 3
        Sweep(OtsuDetector, ignore_zeros=(True, False)),  # 2
    ]  # → 3 * 2 = 6 pipelines


def main() -> None:
    manifest = generate_sweep_manifest(build_golden_config())
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote {GOLDEN} (total_pipelines={manifest['total_pipelines']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the golden + confirm shape**

Run: `uv run python scripts/capture_grid_golden_manifest.py`
Expected output: `Wrote .../tests/fixtures/tune/grid_golden_manifest.json (total_pipelines=6)`.

- [ ] **Step 3: Write the golden sanity lock**

```python
# tests/unit/tune/test_grid_golden_manifest.py
from __future__ import annotations

import json
from pathlib import Path

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "fixtures/tune/grid_golden_manifest.json"
)


def test_golden_exists_and_is_stable():
    assert GOLDEN.exists(), "run scripts/capture_grid_golden_manifest.py"
    manifest = json.loads(GOLDEN.read_text())
    # The Phase-1 GridStrategy regression lock compares against this exact file.
    assert manifest["total_pipelines"] == 6


def test_golden_matches_fresh_generation_while_sweep_exists():
    # Belt-and-suspenders: while sweep still exists, the committed golden must
    # equal a fresh generation (so we know it wasn't hand-edited). Phase 1's
    # GridStrategy will be locked to this golden after sweep is deleted.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from capture_grid_golden_manifest import build_golden_config
    from phenotypic.sweep import generate_sweep_manifest

    fresh = generate_sweep_manifest(build_golden_config())
    committed = json.loads(GOLDEN.read_text())
    assert json.loads(json.dumps(fresh, sort_keys=True)) == committed
```

> Import mechanics (the fix is shown above): `scripts/` is not an importable package, so the test inserts the `scripts/` dir on `sys.path` and imports the generator module **by basename** (`capture_grid_golden_manifest`). This keeps `build_golden_config` defined once (in the script) — no shared-helper module, no repo-root `conftest.py` hack, and no `ModuleNotFoundError` at collection time.

- [ ] **Step 4: Create the test package marker + run the lock**

```python
# tests/unit/tune/__init__.py
```

Run: `uv run pytest tests/unit/tune/test_grid_golden_manifest.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit the golden + script + test**

```bash
uv run ruff check --fix scripts/capture_grid_golden_manifest.py tests/unit/tune
git add scripts/capture_grid_golden_manifest.py tests/fixtures/tune/grid_golden_manifest.json tests/unit/tune
git commit -m "test(tune): capture grid golden manifest fixture (Phase-1 GridStrategy lock)"
```

---

### Task E: Phase-0 green gate

**Files:** none (verification only).

- [ ] **Step 1: Full relevant suite green**

Run: `uv run pytest tests/unit/tools_ tests/unit/core tests/unit/util tests/unit/tune tests/unit/sweep tests/unit/gui -q`
Expected: PASS — the new Phase-0 tests pass and **`sweep` + GUI suites are unaffected** (Phase 0 touched neither's behavior).

- [ ] **Step 2: Type-check + lint the whole Phase-0 surface**

Run: `uv run mypy src/phenotypic/tools_/typing_.py src/phenotypic/_execution src/phenotypic/tune && uv run ruff check src/phenotypic/_execution src/phenotypic/tune src/phenotypic/tools_/typing_.py`
Expected: clean.

- [ ] **Step 3: Commit (if any lint/type fixups were needed)**

```bash
git add -A && git commit -m "chore(tune): Phase-0 mypy/ruff green" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage (engine-architecture §14a prereqs 1–5):**
- §14a.1 registry += `phenotypic.tune` + re-export discipline → **Task B**. ✓ (re-export discipline is exercised in Phase 1 when classes exist; Task B proves the search-path inclusion.)
- §14a.2 `polymorphic_field(base=...)` factory + back-compat test → **Task A**. ✓
- §14a.3 `QCScorer` path contract → **deferred to Phase 1** (it's a usage rule for the `QCScorer`, which doesn't exist yet; no Phase-0 code). Correctly out of scope here.
- §14a.4 `LocalExecutor` (created for tune, sweep not refactored) → **Task C**. ✓
- §14a.5 capture the grid golden fixture → **Task D**. ✓
- master §9 "sweep not refactored / deleted end of Phase 1" → honored: no task touches `sweep` (only Task D *reads* `generate_sweep_manifest`). ✓

**Placeholder scan:** none — every code step is complete. The two soft notes (the `grep` for `_require_operation_value`, the script-import fallback) give concrete instructions, not TODOs.

**Type consistency:** `polymorphic_field(base, *, marker=None)` and `OperationField = polymorphic_field(base=_lazy_base_operation, marker=_OperationFieldMarker())` are consistent across Task A. `Executor.run(work, items) -> list[R]` and `LocalExecutor.run(...)` match across Task C. `build_golden_config()` / `GOLDEN` path / `total_pipelines == 6` are consistent across Task D (script + test). The golden lives at `tests/fixtures/tune/grid_golden_manifest.json` in both the script and the test.

---

## Sequencing note

Tasks A, B, C are fully independent (any order / parallelizable). Task D depends only on `sweep` existing (true now). **None depends on `tune/` internals.** After Phase 0, Phase 1 builds the engine core against `engine-architecture.md`, adds `ScorerField`/`StrategyConfigField = polymorphic_field(base=...)`, and (at its end) deletes `sweep` + locks `GridStrategy` to the golden.
