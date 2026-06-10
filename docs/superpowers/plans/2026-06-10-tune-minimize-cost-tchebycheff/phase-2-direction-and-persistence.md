# Phase 2 — Optuna direction cutover + study persistence

> **For agentic workers:** REQUIRED SUB-SKILL — use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Read
> [`README.md`](README.md) (the shared contract) and spec §7 "Phase 2", §10,
> §11 pitfalls #7/#8 **before** starting.

---

## ⚠️ ATOMIC WITH PHASE 1 — READ THIS FIRST

**Phase 1 and Phase 2 are a single atomic cutover.** They share one branch / one
PR and the suite is only green once *both* land. The coupling is the ASHA pruner:

- The Evaluator reports the running value to the ASHA
  `SuccessiveHalvingPruner` (`OptunaPruningChannel.report → trial.report`,
  `_strategies/_optuna.py`). The pruner is **direction-aware** — it reads
  `study.direction` to decide which interim values survive a rung.
- **Phase 1** flips the *reported value* (goodness → cost, lower-is-better).
- **Phase 2** (this file) flips the *study direction* (`maximize → minimize`).

Flip one without the other and ASHA prunes the **best** candidates. So:

- Do **not** open a separate PR for Phase 2.
- Do **not** run the full `test_optuna_pruning.py` end-to-end pruning assertion
  as "green" until Phase 1's reflected report value is also in the tree.
- The §11.5 escape hatch ("Phases 0–1 are behavior-preserving if pruning is
  disabled between them") is **not** used here — we land 1+2 together.

The other invariant this phase must guarantee (cross-cutting invariant #5,
README): **No silent maximize.** Optuna 4.9.0
`create_study(load_if_exists=True, direction="minimize")` against an existing
`maximize` study **does NOT raise — it silently loads the old study and keeps
`direction = MAXIMIZE`** (verified empirically; spec §11 pitfall #8). So the
**study-name bump is the correctness mechanism**; the `tune_convention` stamp and
the legacy-study detector are UX on top of it, not the safety net.

---

## Scope (this phase only)

1. `_strategies/_optuna_support.py`: `_MAXIMIZE → _MINIMIZE = "minimize"`;
   `study_objective_kwargs` returns `{"direction": _MINIMIZE}` /
   `{"directions": ["minimize"] * n}`.
2. `_multi_objective.py`: `objective_directions` returns `["minimize"] * n`.
3. Best-selection: `_study/_optuna_store.py::OptunaStudyStore.best()` `max → min`;
   `_study_store.py::StudyStore.best()` `max → min`. (Optuna native
   `best_trials` for the multi-objective Pareto path is direction-aware —
   confirm no change needed.)
4. Study-name bump: `_tune_cli/_run.py::_STUDY_NAME "tune" → "tune_cost_v1"`;
   keep all readers using the one constant. Fix GUI desync:
   `gui/tune/_run_root.py::_DEFAULT_STUDY_NAME` (+ alignment test) and the
   `gui/tune/_winner.py` doctest's hardcoded `study_name="tune"`.
5. Stamp + detector in `OptunaStudyStore.__init__`: after `create_study`,
   `study.set_user_attr("tune_convention", "minimize-cost-v1")`; before/around
   create, if a legacy `"tune"` study exists in the storage, log an actionable
   message via `optuna.load_study` in `try/except KeyError` (multi-objective:
   read `.directions` not `.direction`).

**Out of scope (later phases):** the `_run_root.py`
`_MULTI_OBJECTIVE_PLACEHOLDER_DIRECTIONS` flip and other GUI relabel work is
**Phase 4**; Pareto `_dominates` / `_screening_freeze` flips are **Phase 4**; the
Evaluator cost math, scorer migration, gap fix, `_is_suspicious` are **Phase 1**.
Do not touch them here.

---

## Conventions & commands

- Package manager / runner is **`uv` only**. Never bare `python`/`pip`.
- One-time in this worktree: `uv sync --group dev --extra tune`.
- Tune tests need the `tune` extra (Optuna):
  `uv run --extra tune pytest tests/unit/tune/<file>.py -v`.
- Type + lint once at the end of the phase (after the last task, before the
  phase-wrap commit):
  `uv run mypy src/phenotypic/tune` and
  `uv run ruff check --fix src/phenotypic/tune`.
- Commit after every green task with the exact `git add` / `git commit` shown.
- **Line numbers below are main-branch ±1–2.** Re-resolve every cited line by
  reading the file (symbols are authoritative, not line numbers).

---

## Pre-flight (no commit)

- [ ] Confirm the env: `uv run --extra tune python -c "import optuna; print(optuna.__version__)"`
      — expect `4.9.x`. If optuna is missing, run `uv sync --group dev --extra tune`.
- [ ] Confirm Phase 1 is already in the working tree (the reflected report value
      and the cost Evaluator math). If Phase 1 is **not** present, stop — these two
      phases land together (see the ATOMIC banner). You may still author Phase 2
      tests, but the `test_optuna_pruning.py` end-to-end assertion will only pass
      once Phase 1's report flip is also present.
- [ ] Read these symbols in the worktree to re-resolve line numbers:
      `_strategies/_optuna_support.py::_MAXIMIZE` (~L97),
      `study_objective_kwargs` (~L115),
      `_multi_objective.py::objective_directions` (~L91),
      `_study/_optuna_store.py::OptunaStudyStore.best` (~L249) and `__init__`
      (~L57), `_study_store.py::StudyStore.best` (~L97),
      `_tune_cli/_run.py::_STUDY_NAME` (~L70),
      `gui/tune/_run_root.py::_DEFAULT_STUDY_NAME` (~L38),
      `gui/tune/_winner.py` doctest (~L72).

---

## Task 1 — `_MAXIMIZE → _MINIMIZE` + `study_objective_kwargs` direction flip

The single canonical direction literal lives in
`src/phenotypic/tune/_strategies/_optuna_support.py`. `study_objective_kwargs`
maps it onto `create_study`'s mutually-exclusive `direction` /`directions` kwargs
and is the one place every study-creation site shares.

### 1a — Failing test

- [ ] Add the direction-literal tests to the existing
      `tests/unit/tune/test_optuna_pruning.py` companion, or create a focused new
      file `tests/unit/tune/test_optuna_direction.py`. Create the new file:

```python
# tests/unit/tune/test_optuna_direction.py
"""Phase 2: the tuner minimizes cost (lower-is-better), one ``_MINIMIZE`` literal.

The canonical direction literal and the ``study_objective_kwargs`` mapping it
feeds ``create_study`` are the single source of the optimizer's direction. After
the cost cutover every study (and every axis of a multi-objective one) minimizes.
"""
from __future__ import annotations

from phenotypic.tune._strategies._optuna_support import (
    _MINIMIZE,
    study_objective_kwargs,
)


def test_minimize_is_the_canonical_literal():
    assert _MINIMIZE == "minimize"


def test_single_objective_kwargs_minimize():
    # None or a single-axis directions list → the scalar minimize study.
    assert study_objective_kwargs(None) == {"direction": "minimize"}
    assert study_objective_kwargs(["minimize"]) == {"direction": "minimize"}


def test_multi_objective_kwargs_all_minimize():
    kwargs = study_objective_kwargs(["minimize", "minimize"])
    assert kwargs == {"directions": ["minimize", "minimize"]}
```

- [ ] Run (expect failure — `_MINIMIZE` does not exist yet, `study_objective_kwargs`
      still emits `"maximize"`):
      `uv run --extra tune pytest tests/unit/tune/test_optuna_direction.py -v`

### 1b — Minimal implementation

- [ ] In `src/phenotypic/tune/_strategies/_optuna_support.py`, rename the constant
      and update its docstring. Replace the `_MAXIMIZE` block (~L93–97):

```python
#: Every objective in a tuning study is normalized to a bounded ``[0,1]`` **cost**
#: (lower-is-better, ``0`` perfect, ``1`` worst — cost convention §4), so a
#: single-objective study (and every axis of a multi-objective one) **minimizes**.
#: The one canonical ``"minimize"`` literal the strategy, the study store, and the
#: multi-objective inference all share.
_MINIMIZE: Final[str] = "minimize"
```

- [ ] In the same file, update `study_objective_kwargs` (~L115–137): change the
      return docstring and the scalar branch to use `_MINIMIZE`:

```python
    Returns:
        ``{"directions": list(directions)}`` when multi-objective, else
        ``{"direction": _MINIMIZE}``.
    """
    if is_multi_objective_directions(directions):
        assert directions is not None  # narrowed by is_multi_objective_directions
        return {"directions": list(directions)}
    return {"direction": _MINIMIZE}
```

- [ ] Run (expect pass):
      `uv run --extra tune pytest tests/unit/tune/test_optuna_direction.py -v`

### 1c — Commit

- [ ] `git add src/phenotypic/tune/_strategies/_optuna_support.py tests/unit/tune/test_optuna_direction.py`
- [ ] `git commit -m "tune(phase2): flip canonical direction literal _MAXIMIZE -> _MINIMIZE"`

---

## Task 2 — `objective_directions → ["minimize"] * n`

`_multi_objective.py::objective_directions` is the single inference point for the
multi-objective NSGA-II axes (used by `_engine.py`, `_tune_cli/_worker.py`, and
`_tune_cli/_run.py`). It imports the canonical literal from `_optuna_support`.

### 2a — Failing test

- [ ] Append to `tests/unit/tune/test_optuna_direction.py`:

```python
def test_objective_directions_all_minimize():
    from phenotypic.tune._multi_objective import objective_directions

    class _MultiScorer:
        multi_objective = True

        def objective_names(self):
            return ["s0", "s1", "s2"]

    assert objective_directions(_MultiScorer()) == ["minimize", "minimize", "minimize"]


def test_objective_directions_single_objective_is_none():
    from phenotypic.tune._multi_objective import objective_directions

    class _ScalarScorer:
        multi_objective = False

    assert objective_directions(_ScalarScorer()) is None
```

- [ ] Run (expect failure — still emits `"maximize"`):
      `uv run --extra tune pytest tests/unit/tune/test_optuna_direction.py -v`

### 2b — Minimal implementation

- [ ] In `src/phenotypic/tune/_multi_objective.py`, update the import (~L19) and
      the `objective_directions` body + docstring (~L91–113). Change the import:

```python
from ._strategies._optuna_support import _MINIMIZE
```

- [ ] Update the docstring line and the return:

```python
    """The per-objective Optuna ``directions`` for a multi-objective ``scorer``.

    ``["minimize"] * n`` over the scorer's objective axes — every tuning
    objective is bounded cost, lower-is-better (cost convention §4). ``None`` for
    a single-objective scorer (a scalar study), and also ``None`` when a
    multi-objective scorer resolves to fewer than two named axes (a single axis
    is not a Pareto problem — fall back to the scalar path rather than build a
    degenerate one-objective "multi-objective" study).
    ...
    """
    if not is_multi_objective(scorer):
        return None
    names = objective_names(scorer)
    if len(names) < 2:
        return None
    return [_MINIMIZE] * len(names)
```

- [ ] Run (expect pass):
      `uv run --extra tune pytest tests/unit/tune/test_optuna_direction.py -v`

### 2c — Commit

- [ ] `git add src/phenotypic/tune/_multi_objective.py tests/unit/tune/test_optuna_direction.py`
- [ ] `git commit -m "tune(phase2): objective_directions emit minimize axes"`

---

## Task 3 — best-selection `max → min` (both stores)

Both `StudyStore` (the in-memory / parquet-journal backend) and
`OptunaStudyStore` pick the winning trial by `max(valid, key=t.score)`. Under
cost (lower-is-better) the winner is the **minimum** score. The Pareto
multi-objective path (`OptunaStudyStore.pareto_front` via `study.best_trials`) is
direction-aware in Optuna itself — it reads the study's `directions`, so flipping
`study_objective_kwargs` (Task 1) already corrects it; **no change to
`pareto_front`**.

### 3a — Failing test (`StudyStore`)

- [ ] Edit `tests/unit/tune/test_study_store.py`. Update the two best-selection
      assertions to reflect minimize. Replace `test_best_picks_max_score_ignoring_failures`
      (~L22–28):

```python
def test_best_picks_min_cost_ignoring_failures():
    # Cost convention: lower score is better; best() returns the minimum.
    store = StudyStore()
    store.append(_trial(0, 0.3, a=1))
    store.append(_trial(1, 0.9, a=2))
    store.append(_trial(2, 0.05, a=3, failed=True))  # failed → excluded
    best = store.best()
    assert best is not None and best.number == 0 and best.score == 0.3
```

- [ ] In the same file, the parquet round-trip asserts the *best* params; update
      `test_parquet_round_trip` (~L38–47) so the winner is the lower-cost trial:

```python
def test_parquet_round_trip(tmp_path):
    store = StudyStore()
    store.append(_trial(0, 0.3, a=1, mode="x"))
    store.append(_trial(1, 0.9, a=2, mode="y"))
    path = tmp_path / "trials.parquet"
    store.to_parquet(path)
    back = StudyStore.from_parquet(path)
    assert len(back) == 2
    # Lower cost wins under minimize → trial 0 (score 0.3).
    assert back.best().params == {"a": 1, "mode": "x"}
    assert back.best().terms == {"Count": 0.3}
```

- [ ] Run (expect failure — `best()` still returns the max-score trial):
      `uv run --extra tune pytest tests/unit/tune/test_study_store.py -v`

### 3b — Minimal implementation (`StudyStore`)

- [ ] In `src/phenotypic/tune/_study_store.py`, change `StudyStore.best()`
      (~L97–102):

```python
    def best(self) -> Optional[Trial]:
        """The non-failed trial with the lowest cost score, or ``None``."""
        valid = [t for t in self._trials if not t.failed]
        if not valid:
            return None
        return min(valid, key=lambda t: t.score)
```

- [ ] Run (expect pass):
      `uv run --extra tune pytest tests/unit/tune/test_study_store.py -v`

### 3c — Failing test (`OptunaStudyStore`)

- [ ] Add an Optuna-backed best-selection test. Create
      `tests/unit/tune/test_optuna_store_best.py`:

```python
# tests/unit/tune/test_optuna_store_best.py
"""Phase 2: ``OptunaStudyStore.best()`` returns the lowest-cost trial (minimize)."""
from __future__ import annotations

import importlib.util

import pytest

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")


def _store(url: str):
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    return OptunaStudyStore(storage_url=url, study_name="tune_cost_v1")


def _trial(n, score, *, failed=False):
    from phenotypic.tune._study_store import Trial

    return Trial(
        number=n, params={"a": n}, score=score,
        terms={"Count": score}, n_images=2, failed=failed,
    )


def test_best_returns_lowest_cost(tmp_path):
    store = _store(f"sqlite:///{tmp_path / 'study.db'}")
    store.append(_trial(0, 0.3))
    store.append(_trial(1, 0.9))
    store.append(_trial(2, 0.05, failed=True))  # failed → excluded
    best = store.best()
    assert best is not None and best.number == 0 and best.score == 0.3


def test_best_none_when_all_failed(tmp_path):
    store = _store(f"sqlite:///{tmp_path / 'study.db'}")
    store.append(_trial(0, 0.1, failed=True))
    assert store.best() is None
```

- [ ] Run (expect failure — `best()` still `max`):
      `uv run --extra tune pytest tests/unit/tune/test_optuna_store_best.py -v`

### 3d — Minimal implementation (`OptunaStudyStore`)

- [ ] In `src/phenotypic/tune/_study/_optuna_store.py`, change
      `OptunaStudyStore.best()` (~L249–254):

```python
    def best(self) -> Optional[Trial]:
        """The non-failed trial with the lowest cost score, or ``None``."""
        valid = [t for t in self.trials if not t.failed]
        if not valid:
            return None
        return min(valid, key=lambda t: t.score)
```

- [ ] **Confirm `pareto_front` needs no change** — read
      `OptunaStudyStore.pareto_front` (~L289–300). It delegates to
      `self._study.best_trials`, which Optuna computes against the study's
      `directions` (now `["minimize"]*n` after Task 1). No edit. Note this in the
      commit message.

- [ ] Run (expect pass):
      `uv run --extra tune pytest tests/unit/tune/test_optuna_store_best.py tests/unit/tune/test_study_store.py -v`

### 3e — Commit

- [ ] `git add src/phenotypic/tune/_study_store.py src/phenotypic/tune/_study/_optuna_store.py tests/unit/tune/test_study_store.py tests/unit/tune/test_optuna_store_best.py`
- [ ] `git commit -m "tune(phase2): best-selection min(cost); Pareto front unchanged (direction-aware)"`

---

## Task 4 — Study-name bump `_STUDY_NAME "tune" → "tune_cost_v1"` (correctness mechanism)

This is the **correctness** mechanism for the silent-maximize hazard: new code
only ever opens the bumped-name study, so the pre-cutover `"tune"` (maximize)
study is **never reopened**. Every reader (CLI forward path, SLURM submit, worker
launch, run marker) already shares the single `_tune_cli/_run.py::_STUDY_NAME`
constant (verified call sites: `_run.py` L206, L458, L758, L787 — the marker
field, the local store, the SLURM pre-create, and the `SlurmExecutor`).

### 4a — Failing test

- [ ] Create `tests/unit/tune/test_study_name_cutover.py`:

```python
# tests/unit/tune/test_study_name_cutover.py
"""Phase 2: the study name is bumped to ``tune_cost_v1`` (hard cutover, OQ7).

Bumping the single ``_STUDY_NAME`` constant makes the silent direction-mismatch
load (optuna ``load_if_exists`` keeps the old ``maximize``) impossible by
construction: new code never opens the pre-cutover ``"tune"`` study.
"""
from __future__ import annotations


def test_study_name_is_bumped():
    from phenotypic.tune._tune_cli._run import _STUDY_NAME

    assert _STUDY_NAME == "tune_cost_v1"


def test_gui_default_study_name_matches_cli_constant():
    # The GUI fallback constant must stay in lockstep with the CLI constant so a
    # spec-discovered run resolves the bumped study, not the inert legacy one.
    from phenotypic.gui.tune._run_root import _DEFAULT_STUDY_NAME
    from phenotypic.tune._tune_cli._run import _STUDY_NAME

    assert _DEFAULT_STUDY_NAME == _STUDY_NAME
```

- [ ] Run (expect failure — both still `"tune"`):
      `uv run --extra tune pytest tests/unit/tune/test_study_name_cutover.py -v`

### 4b — Minimal implementation

- [ ] In `src/phenotypic/tune/_tune_cli/_run.py`, bump the constant (~L67–70):

```python
#: The study name every tune run uses (the Optuna ``study_name`` + the marker's
#: ``study_name`` field). A single constant keeps the store, the SLURM fleet, and
#: the marker in lockstep. **Bumped from ``"tune"`` for the minimize-cost cutover
#: (spec §7 Phase 2, OQ7):** new code only ever opens this study, so a pre-cutover
#: ``"tune"`` (maximize) study is never reopened — the silent-maximize hazard is
#: impossible by construction, not contingent on a runtime guard.
_STUDY_NAME: Final[str] = "tune_cost_v1"
```

- [ ] In `src/phenotypic/gui/tune/_run_root.py`, bump the GUI fallback constant
      and update its mirror comment (~L35–38). Keep it a *mirrored* literal (not a
      top-level import of `_run._STUDY_NAME`, which would pull the heavy
      `_tune_cli._run` import chain — including `TuningEngine` — at GUI import
      time); the alignment test in 4a is the lockstep guard (matches the project's
      enum/literal-alignment convention):

```python
#: The study name every tune run uses (mirrors ``_tune_cli._run._STUDY_NAME``;
#: kept in lockstep by ``test_study_name_cutover.py``). Used as the fallback when
#: discovering from a ``tuning_spec.json`` (which, unlike the ``run.json`` marker,
#: does not record a study name).
_DEFAULT_STUDY_NAME: str = "tune_cost_v1"
```

- [ ] Run (expect pass):
      `uv run --extra tune pytest tests/unit/tune/test_study_name_cutover.py -v`

### 4c — Sweep for stray `"tune"` re-spellings

- [ ] Grep for any remaining hardcoded study name that should be the constant
      (excludes the GUI mirror, the spec/docs, and the `_winner.py` doctest fixed
      in Task 6):
      `grep -rn 'study_name *= *["'"'"']tune["'"'"']\|"tune"\|'"'"'tune'"'"'' src/phenotypic/tune src/phenotypic/gui/tune`
- [ ] Confirm the live readers (`_run.py` marker write, the local
      `OptunaStudyStore(...)` create, the SLURM pre-create, the `SlurmExecutor`
      `study_name=` arg, `_tune_cli/_worker.py` `--study-name` passthrough) all
      flow from `_STUDY_NAME` — no edit needed if they already reference the
      constant (they do at L206/L458/L758/L787). Note any newly-found re-spelling
      in the commit; fix it to import `_STUDY_NAME`.

### 4d — Commit

- [ ] `git add src/phenotypic/tune/_tune_cli/_run.py src/phenotypic/gui/tune/_run_root.py tests/unit/tune/test_study_name_cutover.py`
- [ ] `git commit -m "tune(phase2): bump _STUDY_NAME -> tune_cost_v1 (hard cutover); sync GUI fallback"`

---

## Task 5 — Stamp `tune_convention` + friendly legacy-study detector

In `OptunaStudyStore.__init__` (the `create=True` engine path), after the study is
created/loaded, stamp `study.set_user_attr("tune_convention", "minimize-cost-v1")`
for observability and future cutovers. Separately, **before** create, detect a
pre-cutover `"tune"` study sitting in the same storage and emit an actionable log
message (UX only — correctness is already the name bump from Task 4).

Detector contract:
- Use `optuna.load_study(storage=..., study_name="tune")` inside
  `try/except KeyError` (Optuna raises `KeyError` when the named study is
  absent). Any other exception must not abort startup — swallow + debug-log.
- A pre-cutover study is multi-objective-agnostic: read `.directions` (a list),
  **not** `.direction` (which raises on a multi-objective study). If any axis is
  `MAXIMIZE` (or simply: the legacy study exists at all), log the friendly note.
- Do **not** raise — the spec scopes this as UX, not correctness (§7 Phase 2 #3).
- Skip the detector when the new study name *is* `"tune"` (defensive: never warn
  about ourselves) — after Task 4 the name is `"tune_cost_v1"`, so this is just a
  guard against a misconfigured caller.

### 5a — Failing test (silent-maximize regression + stamp)

- [ ] Create `tests/unit/tune/test_study_persistence_cutover.py`:

```python
# tests/unit/tune/test_study_persistence_cutover.py
"""Phase 2: persistence hard cutover — no silent maximize, stamp + detector.

Verified optuna 4.9.0 hazard: ``create_study(load_if_exists=True,
direction="minimize")`` against an existing ``maximize`` study does NOT raise; it
silently keeps ``MAXIMIZE``. The name bump (``tune`` → ``tune_cost_v1``) makes
reopening the legacy study impossible by construction. These tests assert: the
new store opens the bumped minimize study; a pre-existing legacy ``"tune"``
maximize study in the SAME storage stays inert; the friendly detector fires; and
a fresh minimize study carries the ``tune_convention`` stamp.
"""
from __future__ import annotations

import importlib.util

import pytest

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")

_CONVENTION_KEY = "tune_convention"
_CONVENTION_VALUE = "minimize-cost-v1"


def _seed_legacy_maximize_study(url: str) -> None:
    """Write a pre-cutover ``"tune"`` MAXIMIZE study into the storage."""
    import optuna

    legacy = optuna.create_study(
        storage=url, study_name="tune", direction="maximize"
    )
    legacy.add_trial(
        optuna.trial.create_trial(
            value=0.95, params={}, distributions={},
            state=optuna.trial.TrialState.COMPLETE,
        )
    )


def test_fresh_study_minimizes_and_is_stamped(tmp_path):
    import optuna

    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = f"sqlite:///{tmp_path / 'study.db'}"
    store = OptunaStudyStore(storage_url=url, study_name="tune_cost_v1")
    assert store.study.direction == optuna.study.StudyDirection.MINIMIZE
    assert store.study.user_attrs.get(_CONVENTION_KEY) == _CONVENTION_VALUE


def test_legacy_maximize_study_left_inert(tmp_path, caplog):
    import logging

    import optuna

    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = f"sqlite:///{tmp_path / 'study.db'}"
    _seed_legacy_maximize_study(url)

    with caplog.at_level(logging.WARNING):
        store = OptunaStudyStore(storage_url=url, study_name="tune_cost_v1")

    # The new store opened the BUMPED, MINIMIZE study — never the legacy one.
    assert store.study.study_name == "tune_cost_v1"
    assert store.study.direction == optuna.study.StudyDirection.MINIMIZE

    # The legacy study is still present and still MAXIMIZE (inert, not reopened).
    legacy = optuna.load_study(storage=url, study_name="tune")
    assert legacy.direction == optuna.study.StudyDirection.MAXIMIZE

    # Friendly detector fired with an actionable message.
    assert any(
        "pre-cutover" in rec.getMessage().lower()
        or "tune_cost_v1" in rec.getMessage()
        for rec in caplog.records
    )


def test_no_detector_warning_without_legacy_study(tmp_path, caplog):
    import logging

    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = f"sqlite:///{tmp_path / 'study.db'}"
    with caplog.at_level(logging.WARNING):
        OptunaStudyStore(storage_url=url, study_name="tune_cost_v1")
    assert not any("pre-cutover" in rec.getMessage().lower() for rec in caplog.records)
```

- [ ] Run (expect failure — no `tune_convention` stamp, no detector, possibly an
      `AttributeError`/import error for the new module-level logger):
      `uv run --extra tune pytest tests/unit/tune/test_study_persistence_cutover.py -v`

### 5b — Minimal implementation

- [ ] In `src/phenotypic/tune/_study/_optuna_store.py`, add a module-level logger
      and the two constants near the top (after the existing
      `_HEARTBEAT_INTERVAL_S` / `_GRACE_PERIOD_S` block, ~L37–38):

```python
import logging

_logger = logging.getLogger(__name__)

#: Stamped on every freshly-created/loaded cost-convention study for
#: observability and future cutovers (spec §7 Phase 2 #2).
_CONVENTION_ATTR: str = "tune_convention"
_CONVENTION_VALUE: str = "minimize-cost-v1"
#: The pre-cutover study name. Detecting it in the same storage is UX only —
#: correctness is the ``_STUDY_NAME`` bump (a legacy study is never reopened).
_LEGACY_STUDY_NAME: str = "tune"
```

  (Add `import logging` to the existing top-of-file imports; do not duplicate.)

- [ ] In `OptunaStudyStore.__init__`, inside the `if create:` block, call the
      detector before `create_study` and stamp after it. Replace the
      `if create:` body (~L71–86) so it reads:

```python
        if create:
            storage = optuna.storages.RDBStorage(
                url=storage_url,
                heartbeat_interval=_HEARTBEAT_INTERVAL_S,
                grace_period=_GRACE_PERIOD_S,
            )
            if storage_url.startswith("sqlite"):
                self._enable_sqlite_wal(storage)

            # UX-only (correctness is the _STUDY_NAME bump): if a pre-cutover
            # "tune" study still sits in this storage, it cannot be resumed under
            # the cost convention — say so with an actionable message instead of
            # silently starting fresh beside it.
            self._warn_if_legacy_study_present(storage)

            create_kwargs: dict[str, Any] = {
                "storage": storage,
                "study_name": study_name,
                "load_if_exists": True,
                **study_objective_kwargs(directions),
            }
            self._study = optuna.create_study(**create_kwargs)
            self._study.set_user_attr(_CONVENTION_ATTR, _CONVENTION_VALUE)
        else:
            self._study = optuna.load_study(
                storage=storage_url,
                study_name=study_name,
            )
```

- [ ] Add the detector helper method to `OptunaStudyStore` (place it next to
      `_enable_sqlite_wal`, ~L117):

```python
    def _warn_if_legacy_study_present(self, storage: Any) -> None:
        """Log an actionable note when a pre-cutover ``"tune"`` study exists.

        UX only — the ``_STUDY_NAME`` bump already makes the silent direction
        mismatch impossible (optuna ``load_if_exists`` would otherwise keep the
        legacy ``maximize`` direction, verified 4.9.0). A pre-cutover study cannot
        be resumed under the cost convention; this points the user at a fresh run.

        ``optuna.load_study`` raises ``KeyError`` when the legacy study is absent
        (the common case) — swallowed. Any other error is non-fatal and
        debug-logged: detection must never abort study startup.
        """
        import optuna

        if self._study_name == _LEGACY_STUDY_NAME:
            return  # never warn about ourselves
        try:
            legacy = optuna.load_study(
                storage=storage, study_name=_LEGACY_STUDY_NAME
            )
        except KeyError:
            return  # no legacy study — nothing to warn about
        except Exception:  # noqa: BLE001 - detection is best-effort UX
            _logger.debug(
                "could not probe for a legacy %r study", _LEGACY_STUDY_NAME,
                exc_info=True,
            )
            return
        # ``.directions`` is multi-objective-safe (``.direction`` raises on a
        # multi-objective study); we only need its presence for the message.
        _logger.warning(
            "a pre-cutover %r study (directions=%s) is present in this storage; "
            "it cannot be resumed under the minimize-cost convention. Starting a "
            "fresh %r study beside it (or use a new output dir).",
            _LEGACY_STUDY_NAME,
            [d.name.lower() for d in legacy.directions],
            self._study_name,
        )
```

- [ ] Run (expect pass):
      `uv run --extra tune pytest tests/unit/tune/test_study_persistence_cutover.py -v`

### 5c — Commit

- [ ] `git add src/phenotypic/tune/_study/_optuna_store.py tests/unit/tune/test_study_persistence_cutover.py`
- [ ] `git commit -m "tune(phase2): stamp tune_convention + friendly legacy-study detector"`

---

## Task 6 — Fix `gui/tune/_winner.py` doctest study name + reflect end-to-end pruning

Two cleanups that depend on the earlier tasks.

### 6a — `_winner.py` doctest

The `write_winner` doctest hardcodes `study_name="tune"` and a winner with
`score=0.9`. The study name should be the bumped value; `score=0.9` is fine for
the doctest (it asserts on the override landing, not on direction), but the name
must not re-spell the legacy `"tune"`.

- [ ] In `src/phenotypic/gui/tune/_winner.py`, the `TuneRunRoot(...)` doctest call
      (~L71–75), change `study_name="tune"` → `study_name="tune_cost_v1"`:

```python
        >>> root = TuneRunRoot(
        ...     path=d, trials_path=None, storage_url=None,
        ...     study_name="tune_cost_v1",
        ...     directions=None, images_dir=None,
        ...     best_pipeline_path=best_pipeline_path(d),
        ... )
```

- [ ] Run the doctest:
      `uv run --extra tune python -m pytest --doctest-modules src/phenotypic/gui/tune/_winner.py -v`
      (or `uv run --extra tune python -m doctest src/phenotypic/gui/tune/_winner.py -v`).
      Expect pass.

### 6b — Reflect `test_optuna_pruning.py` end-to-end (PRUNED under minimize)

`test_bad_candidate_pruned_end_to_end` (~L114–151) seeds *good* trials at `1.0`
and a *bad* one at `0.0` and asserts PRUNED. Under minimize cost, **good = low
cost, bad = high cost** — invert the reported values and the `EvaluationResult`
scores or the ASHA pruner would prune the *best* candidate.

> **Atomic-with-Phase-1 note:** this end-to-end test only goes green once Phase
> 1's reflected report value is in the tree (the report feeds the
> direction-aware ASHA pruner). If Phase 1 is not yet present, expect this single
> test to still fail until both phases are integrated; the other Phase 2 tests
> are independent and pass now.

- [ ] In `tests/unit/tune/test_optuna_pruning.py`, rewrite
      `test_bad_candidate_pruned_end_to_end` to use cost values (good = low):

```python
def test_bad_candidate_pruned_end_to_end(tmp_path):
    """A deliberately-bad candidate reporting a HIGH cost at rung 1 is PRUNED.

    Cost convention (minimize): seed several good trials at a low first-rung cost,
    then a clearly inferior one at a high cost; the direction-aware ASHA pruner
    marks the inferior (high-cost) trial PRUNED at the first rung.
    """
    import optuna

    space = SearchSpace(knobs=(
        Knob(key="0.f", domain=FloatRange(low=0.0, high=1.0)),
    ))
    strat = _strategy(space, sampler="tpe", n_trials=50, rung_floor=1, rung_factor=2)

    # Strong trials report a LOW cost at the first rung (good). They must check
    # should_prune() so ASHA registers their values into the rung's pool.
    for _ in range(6):
        params, channel = strat.suggest()
        channel.report(0.0, 1)
        channel.should_prune()
        strat.register_result(params, EvaluationResult(
            score=0.0, terms={"t": 0.0}, n_images=2,
        ))

    # A clearly-inferior trial: reports a HIGH cost at the first rung (bad).
    params, channel = strat.suggest()
    channel.report(1.0, 1)
    pruned = channel.should_prune()
    assert pruned is True
    strat.register_result(
        params,
        EvaluationResult(score=1.0, terms={"t": 1.0}, n_images=1, pruned=True),
        pruned=True,
    )
    states = [t.state for t in strat._study.get_trials(deepcopy=False)]
    assert optuna.trial.TrialState.PRUNED in states
```

- [ ] Also update the multi-objective directions fixture in the same file —
      `test_multi_objective_gets_noop_channel` (~L105–111) passes
      `directions=["maximize", "maximize"]`. The channel choice is
      direction-independent, but use the cost-convention directions for
      consistency: `directions=["minimize", "minimize"]`.
- [ ] Run (expect pass once Phase 1 is integrated):
      `uv run --extra tune pytest tests/unit/tune/test_optuna_pruning.py -v`

### 6c — Commit

- [ ] `git add tests/unit/tune/test_optuna_pruning.py src/phenotypic/gui/tune/_winner.py`
- [ ] `git commit -m "tune(phase2): reflect end-to-end pruning to cost; fix winner doctest study name"`

---

## Task 7 — Shared-study-handle test alignment + full regression

The `test_shared_study_handle_and_retry.py` helpers spell `_STUDY = "tune"` and
construct a multi-objective strategy with `directions=["maximize", "maximize"]`
in `test_multi_objective_gets_noop_channel` (that one is in
`test_optuna_pruning.py`, handled in 6b). The shared-handle tests do not assert
direction, but the literal `_STUDY = "tune"` would now collide with the legacy
detector's warning (harmless) — bump it to the new name for clarity and to keep
the suite on the cutover convention.

### 7a — Update the shared-handle test fixture

- [ ] In `tests/unit/tune/test_shared_study_handle_and_retry.py`, bump the module
      constant (~L22):

```python
_STUDY = "tune_cost_v1"
```

- [ ] Run (expect pass — these tests assert handle identity / sampler re-attach,
      not direction):
      `uv run --extra tune pytest tests/unit/tune/test_shared_study_handle_and_retry.py -v`

### 7b — Full tune-suite regression

- [ ] Run the whole tune unit suite:
      `uv run --extra tune pytest tests/unit/tune -v`
- [ ] If any test still references the old direction/study-name and fails, it is
      either (a) a Phase-1-owned cost-math test that should already be reflected
      in the shared branch, or (b) a stray re-spelling — re-resolve and fix in
      the smallest scope, then re-run. Do **not** "fix" by reverting a Phase 2
      change.

### 7c — Type + lint (phase boundary)

- [ ] `uv run mypy src/phenotypic/tune`
- [ ] `uv run ruff check --fix src/phenotypic/tune`
- [ ] Re-run the tune suite if ruff applied any auto-fixes:
      `uv run --extra tune pytest tests/unit/tune -v`

### 7d — Commit

- [ ] `git add tests/unit/tune/test_shared_study_handle_and_retry.py`
- [ ] (Include any ruff auto-fixed source files in the add.)
- [ ] `git commit -m "tune(phase2): align shared-study-handle test to bumped study name; mypy/ruff clean"`

---

## Phase 2 done-criteria (verify before claiming complete)

- [ ] `_MINIMIZE = "minimize"` is the only direction literal in
      `_optuna_support.py`; `study_objective_kwargs` emits `"minimize"` for both
      single- and multi-objective shapes (`grep -rn '"maximize"' src/phenotypic/tune`
      returns only Phase-4-owned GUI placeholder sites, if any — note them).
- [ ] `objective_directions` returns `["minimize"] * n`.
- [ ] Both `best()` implementations use `min(...)`; `pareto_front` unchanged.
- [ ] `_STUDY_NAME == "tune_cost_v1"`; `_DEFAULT_STUDY_NAME == _STUDY_NAME`
      (alignment test green); `_winner.py` doctest uses the bumped name.
- [ ] A fresh study is `MINIMIZE` and carries `tune_convention="minimize-cost-v1"`.
- [ ] A legacy `"tune"` maximize study in the same storage is left inert; the new
      store opens `"tune_cost_v1"` minimize; the friendly detector logs.
- [ ] `uv run --extra tune pytest tests/unit/tune -v` is green **with Phase 1
      integrated** (the end-to-end pruning test is the one that requires Phase 1's
      reflected report value).
- [ ] `uv run mypy src/phenotypic/tune` and `uv run ruff check src/phenotypic/tune`
      are clean.

## Cross-cutting invariant coverage (README §"Cross-cutting invariants")

- **#5 No silent maximize** — Tasks 4 (name bump = correctness) + 5 (stamp +
  detector = UX); regression test in `test_study_persistence_cutover.py`.
- **Pruner ↔ direction coupling** — Task 6b's end-to-end PRUNED test exercises
  the direction-aware ASHA pruner against the flipped study direction; the
  ATOMIC-with-Phase-1 banner documents the coupling.

## Files touched by Phase 2

Source:
`src/phenotypic/tune/_strategies/_optuna_support.py`,
`src/phenotypic/tune/_multi_objective.py`,
`src/phenotypic/tune/_study_store.py`,
`src/phenotypic/tune/_study/_optuna_store.py`,
`src/phenotypic/tune/_tune_cli/_run.py`,
`src/phenotypic/gui/tune/_run_root.py`,
`src/phenotypic/gui/tune/_winner.py`.

Tests:
`tests/unit/tune/test_optuna_direction.py` (new),
`tests/unit/tune/test_optuna_store_best.py` (new),
`tests/unit/tune/test_study_name_cutover.py` (new),
`tests/unit/tune/test_study_persistence_cutover.py` (new),
`tests/unit/tune/test_study_store.py` (edit),
`tests/unit/tune/test_optuna_pruning.py` (edit),
`tests/unit/tune/test_shared_study_handle_and_retry.py` (edit).

> **Docs note:** `tune/CLAUDE.md` ("Higher-is-better everywhere; the single
> `_MAXIMIZE` literal") and `explain/tune-with-optuna.md` are updated in
> **Phase 5**, not here — but the wording flip is mandated by this phase's change.
> Do not let the CLAUDE.md mandate block the Phase 2 commit; it is tracked as a
> Phase 5 task.
