# Tune Engine — Phase 1d: Engine + CLI + Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Phase 1 — wire the Phase-1a/1b/1c pieces into a runnable engine: `Budget` + `Trial` + `StudyStore` (journal fallback), `TuningSpec` (embedding the base `pipeline`), `TuningEngine` (ask-and-tell loop + resume), RF-permutation `param_importance`, the `python -m phenotypic.tune` CLI (`-i/-o` + the `deliverables/` output layout), the **grid byte-compat lock** against the Phase-0 golden, and the **hard cutover** (delete `sweep` + ship the `manifest→spec` migration script).

**Architecture:** A `TuningSpec` is one self-contained pydantic model carrying the **embedded base `ImagePipeline`** (custom `to_json`/`from_json` field (de)serializer — plain pydantic can't round-trip polymorphic ops), the `SearchSpace`, a polymorphic `Scorer`, an `Evaluator`, a `StrategyConfig`, and a `Budget`. The `TuningEngine` drives the loop the engine-arch §8 sketches: `strategy = spec.strategy.build(space, store)`; `while not exhausted: params,_ = suggest(); result = evaluator.evaluate(spec.pipeline, spec.scorer, params, images); store.append(Trial); register_result`. The `StudyStore` is the **homegrown journal** (`trials.parquet`) — Optuna SQLite is Phase 2 — and powers **resume** (fast-forward a deterministic strategy past recorded trials). Importance falls back to **sklearn RandomForest + permutation_importance** (fANOVA is Phase 2). The CLI loads images from `-i`, runs the engine, and writes `deliverables/{best_pipeline,tuning_spec,param_importance}.json` + `trials.parquet`. `--strategy grid` reproduces the deleted sweep's exhaustive grid, locked against the Phase-0 golden.

**Tech Stack:** pydantic v2 (custom `field_serializer`/`field_validator`), `scikit-learn` + `joblib` + `pyarrow` (**already deps** — no new third-party deps), `argparse`. Reuses Phase 0 (`polymorphic_field`, registry `+= tune`, golden fixture), 1a (`SearchSpace`), 1b (`StrategyConfig`/`enumerate_grid`), 1c (`Scorer`/`QCScorer`/`Evaluator`/`build_pipeline`).

**Spec:** `engine-architecture.md` §5–§8 (Budget/TuningSpec/Executor/TuningEngine, incl. the §6 *embedded-pipeline* refinement applied this phase); master §6 (CLI `-i/-o`), §8 (output layout), §9 (hard cutover + golden lock). **Depends on:** Phases 0, 1a, 1b, 1c. **Closes:** Phase 1.

**Conventions:** `uv run pytest`, `uv run mypy src/phenotypic/tune`, `uv run ruff check --fix`; Google docstrings; tests under `tests/unit/tune/`; resolve output paths via `phenotypic.tools_` helpers (never hand-joined).

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/phenotypic/tune/_study_store.py` | `Trial`, `StudyStore` (journal + best + parquet round-trip) | Create |
| `src/phenotypic/tune/_spec.py` | `Budget`, `ScorerField`, `TuningSpec` (embedded pipeline) | Create |
| `src/phenotypic/tune/_engine.py` | `TuningEngine` (ask-and-tell loop + resume) | Create |
| `src/phenotypic/tune/_screening.py` | `compute_param_importance` (RF-permutation fallback) | Create |
| `src/phenotypic/tune/_tune_cli/__init__.py` | re-export `run_tuning` | Create |
| `src/phenotypic/tune/_tune_cli/_run.py` | `run_tuning` orchestration + `_load_images` + output writes | Create |
| `src/phenotypic/tune/__main__.py` | the `argparse` CLI (`python -m phenotypic.tune`) | Create |
| `src/phenotypic/tune/__init__.py` | export the Phase-1d public surface | Modify |
| `src/phenotypic/tools_/_io_constants.py` | tune output-path helpers | Modify |
| `scripts/migrate_sweep_manifest.py` | legacy `manifest → tuning_spec.json` | Create |
| `src/phenotypic/sweep/` | **deleted** (hard cutover) | Delete |
| `tests/unit/tune/test_study_store.py` … `test_migrate_manifest.py` | per-task tests | Create |

---

### Task 1: `Trial` + `StudyStore` (the journal)

**Files:**
- Create: `src/phenotypic/tune/_study_store.py`
- Test: `tests/unit/tune/test_study_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_study_store.py
from __future__ import annotations

from phenotypic.tune._study_store import StudyStore, Trial


def _trial(n: int, score: float, *, failed: bool = False, **params) -> Trial:
    return Trial(
        number=n, params=params, score=score,
        terms={"Count": score}, n_images=2, failed=failed,
    )


def test_append_and_len():
    store = StudyStore()
    store.append(_trial(0, 0.3, a=1))
    store.append(_trial(1, 0.9, a=2))
    assert len(store) == 2


def test_best_picks_max_score_ignoring_failures():
    store = StudyStore()
    store.append(_trial(0, 0.3, a=1))
    store.append(_trial(1, 0.9, a=2))
    store.append(_trial(2, 0.99, a=3, failed=True))  # failed → excluded
    best = store.best()
    assert best is not None and best.number == 1 and best.score == 0.9


def test_best_none_when_empty_or_all_failed():
    assert StudyStore().best() is None
    store = StudyStore()
    store.append(_trial(0, 0.0, failed=True))
    assert store.best() is None


def test_parquet_round_trip(tmp_path):
    store = StudyStore()
    store.append(_trial(0, 0.3, a=1, mode="x"))
    store.append(_trial(1, 0.9, a=2, mode="y"))
    path = tmp_path / "trials.parquet"
    store.to_parquet(path)
    back = StudyStore.from_parquet(path)
    assert len(back) == 2
    assert back.best().params == {"a": 2, "mode": "y"}
    assert back.best().terms == {"Count": 0.9}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_study_store.py -v`
Expected: FAIL — `ModuleNotFoundError: ..._study_store`.

- [ ] **Step 3: Implement `Trial` + `StudyStore`**

```python
# src/phenotypic/tune/_study_store.py
"""The trial journal — Phase-1 homegrown persistence (Optuna SQLite is Phase 2).

A ``StudyStore`` accumulates ``Trial`` records, reports the ``best`` (max score
among non-failed trials), and round-trips through ``trials.parquet`` (params and
terms persisted as JSON columns — lossless across heterogeneous/conditional
param sets). Reloading a store powers CLI resume (``_engine`` fast-forwards a
deterministic strategy past the recorded trials).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict


class Trial(BaseModel):
    """One evaluated candidate: its params, score, per-term scores, and status."""

    model_config = ConfigDict(frozen=True)

    number: int
    params: dict[str, Any]
    score: float
    terms: dict[str, float]
    n_images: int
    failed: bool = False


class StudyStore:
    """An append-only journal of trials with best-tracking + parquet I/O."""

    def __init__(self, trials: Optional[list[Trial]] = None) -> None:
        self._trials: list[Trial] = list(trials or [])

    def append(self, trial: Trial) -> None:
        self._trials.append(trial)

    @property
    def trials(self) -> list[Trial]:
        return list(self._trials)

    def __len__(self) -> int:
        return len(self._trials)

    def best(self) -> Optional[Trial]:
        """The non-failed trial with the highest score, or ``None``."""
        valid = [t for t in self._trials if not t.failed]
        if not valid:
            return None
        return max(valid, key=lambda t: t.score)

    def to_dataframe(self) -> pd.DataFrame:
        """One row per trial; ``params``/``terms`` serialized as JSON strings."""
        return pd.DataFrame(
            {
                "number": t.number,
                "score": t.score,
                "n_images": t.n_images,
                "failed": t.failed,
                "params_json": json.dumps(t.params, sort_keys=True),
                "terms_json": json.dumps(t.terms, sort_keys=True),
            }
            for t in self._trials
        )

    def to_parquet(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: Path) -> "StudyStore":
        df = pd.read_parquet(path)
        trials = [
            Trial(
                number=int(row.number),
                params=json.loads(row.params_json),
                score=float(row.score),
                terms=json.loads(row.terms_json),
                n_images=int(row.n_images),
                failed=bool(row.failed),
            )
            for row in df.itertuples(index=False)
        ]
        return cls(trials)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_study_store.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_study_store.py tests/unit/tune/test_study_store.py
git add src/phenotypic/tune/_study_store.py tests/unit/tune/test_study_store.py
git commit -m "feat(tune): Trial + StudyStore journal (parquet round-trip + best)"
```

---

### Task 2: `Budget` + `TuningSpec` (embedded pipeline)

**Files:**
- Create: `src/phenotypic/tune/_spec.py`
- Test: `tests/unit/tune/test_tuning_spec.py`

`TuningSpec.pipeline` embeds the base `ImagePipeline` (engine-arch §6). Because a pipeline
does **not** round-trip through plain pydantic (its polymorphic ops fail to reconstruct
against the abstract `ImageOperation`), the field uses a custom serializer/validator
delegating to the pipeline's own `to_json`/`from_json` (verified). `scorer` uses
`polymorphic_field(base=Scorer)` (Phase-0 factory + registry `+= tune` + the `QCScorer`
export from 1c), so any `Scorer` subclass round-trips; `strategy` uses the Phase-1b
`StrategyConfigUnion` (grid/random discriminated union).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_tuning_spec.py
from __future__ import annotations

import pandas as pd

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tune import (
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    QCScorer,
    SearchSpace,
)
from phenotypic.tune._spec import Budget, TuningSpec


def _spec(tmp_path) -> TuningSpec:
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["p"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(csv), groupby=["Metadata_ImageName"]
            )
        ),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def test_budget_defaults():
    b = Budget()
    assert b.n_trials is None  # grid → run until exhausted


def test_spec_round_trips_pipeline_and_scorer(tmp_path):
    spec = _spec(tmp_path)
    back = TuningSpec.model_validate_json(spec.model_dump_json())
    # embedded pipeline reconstructed (polymorphic ops survive)
    assert [type(o).__name__ for o in back.pipeline.get_ops().values()] == [
        "GaussianBlur", "OtsuDetector",
    ]
    assert back.pipeline.get_ops()["GaussianBlur"].sigma == 2.0
    # polymorphic scorer reconstructed; path-configured check still scores
    assert isinstance(back.scorer, QCScorer)
    assert back.scorer.score_image(
        None, pd.DataFrame({"Metadata_ImageName": ["p"] * 96,
                            "Object_Label": list(range(96))})
    )["Count"] == 1.0
    assert isinstance(back.strategy, GridConfig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_tuning_spec.py -v`
Expected: FAIL — `ModuleNotFoundError: ..._spec`.

- [ ] **Step 3: Implement `Budget` + `TuningSpec`**

```python
# src/phenotypic/tune/_spec.py
"""The tuning_spec.json model — one self-contained, round-trippable recipe."""
from __future__ import annotations

import json
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
)

from phenotypic import ImagePipeline
from phenotypic.tools_.typing_ import polymorphic_field

from ._evaluation import Evaluator
from ._scoring import Scorer
from ._search_space import SearchSpace
from ._strategies._config import StrategyConfigUnion

#: A ``Scorer``-valued field that round-trips any subclass via the registry
#: (Phase-0 ``polymorphic_field`` + ``_find_class_in_phenotypic`` += ``phenotypic.tune``).
ScorerField = polymorphic_field(base=Scorer)


class Budget(BaseModel):
    """Stopping criteria (engine-arch §5). Phase 1: trial count + failure cap."""

    model_config = ConfigDict(frozen=True)

    n_trials: Optional[int] = None     # engine-level cap; None → run until the strategy exhausts
    max_failures: Optional[int] = None  # abort after this many failed candidates; None → never


class TuningSpec(BaseModel):
    """A complete tuning run: base pipeline + space + scorer + strategy + budget."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline: ImagePipeline           # the base being tuned (embedded; custom (de)serializer)
    search_space: SearchSpace
    scorer: ScorerField               # any Scorer subclass
    evaluator: Evaluator
    strategy: StrategyConfigUnion     # grid | random (Phase 1)
    budget: Budget

    @field_validator("pipeline", mode="before")
    @classmethod
    def _coerce_pipeline(cls, value: object) -> ImagePipeline:
        """Accept a live pipeline, its JSON string, or its embedded dict."""
        if isinstance(value, ImagePipeline):
            return value
        if isinstance(value, str):
            return ImagePipeline.from_json(value)
        if isinstance(value, dict):
            return ImagePipeline.from_json(json.dumps(value))
        raise TypeError(
            f"pipeline must be an ImagePipeline, JSON string, or dict; "
            f"got {type(value).__name__}"
        )

    @field_serializer("pipeline")
    def _dump_pipeline(self, value: ImagePipeline) -> dict:
        return json.loads(value.to_json())
```

> The `scorer: ScorerField` round-trip requires the Phase-0 registry edit (`_find_class_in_phenotypic` includes `phenotypic.tune`) and the `QCScorer` export from `tune/__init__.py` (1c). Both are prerequisites; if the round-trip test errors with "class not found", that edit is missing.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_tuning_spec.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_spec.py tests/unit/tune/test_tuning_spec.py
git add src/phenotypic/tune/_spec.py tests/unit/tune/test_tuning_spec.py
git commit -m "feat(tune): TuningSpec (embedded pipeline) + Budget"
```

---

### Task 3: `TuningEngine` (ask-and-tell loop + resume)

**Files:**
- Create: `src/phenotypic/tune/_engine.py`
- Test: `tests/unit/tune/test_engine.py`

The loop builds the strategy from the spec, fast-forwards past any trials already in the
store (resume — deterministic strategies reproduce their prefix), then suggests → evaluates
→ records until the strategy exhausts or the budget caps. A candidate the `Evaluator` marks
failed (empty terms) is recorded `failed=True`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_engine.py
from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune import (
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    Scorer,
    SearchSpace,
)
from phenotypic.tune._engine import TuningEngine
from phenotypic.tune._spec import Budget, TuningSpec
from phenotypic.tune._study_store import StudyStore


class _ConstScorer(Scorer):
    def score_image(self, image, measurements) -> dict[str, float]:
        return {"Count": 1.0}


def _grid_space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.GaussianBlur.__enabled__", domain=Categorical(choices=(True, False))),
        Knob(key="0.sigma", domain=Categorical(choices=(1.0, 2.0)),
             conditional_on=(("0.GaussianBlur.__enabled__", True),)),
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def _spec(budget: Budget, store_pipeline) -> TuningSpec:
    return TuningSpec(
        pipeline=store_pipeline,
        search_space=_grid_space(),
        scorer=_ConstScorer(),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=budget,
    )


def _base():
    from phenotypic.enhance import GaussianBlur
    return ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()])


def test_engine_runs_full_grid():
    spec = _spec(Budget(), _base())
    engine = TuningEngine(spec)
    best = engine.optimize([load_synth_yeast_plate()])
    assert len(engine.store) == 6           # the conditional Cartesian product
    assert best is not None
    # all six param-combos are distinct
    seen = {tuple(sorted(t.params.items())) for t in engine.store.trials}
    assert len(seen) == 6


def test_engine_budget_caps_trials():
    spec = _spec(Budget(n_trials=3), _base())
    engine = TuningEngine(spec)
    engine.optimize([load_synth_yeast_plate()])
    assert len(engine.store) == 3


def test_engine_resumes_from_store():
    img = [load_synth_yeast_plate()]
    # first run: 3 of 6
    e1 = TuningEngine(_spec(Budget(n_trials=3), _base()))
    e1.optimize(img)
    # resume: same store, no cap → completes to 6 with no duplicates
    e2 = TuningEngine(_spec(Budget(), _base()), store=e1.store)
    e2.optimize(img)
    assert len(e2.store) == 6
    seen = {tuple(sorted(t.params.items())) for t in e2.store.trials}
    assert len(seen) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: ..._engine`.

- [ ] **Step 3: Implement `TuningEngine`**

```python
# src/phenotypic/tune/_engine.py
"""The orchestrator — the ask-and-tell loop over a strategy + evaluator.

Drives ``suggest → evaluate → register_result`` until the strategy exhausts or
the budget caps, journaling every trial. Resumes by fast-forwarding a
deterministic strategy past the trials already in the store.
"""
from __future__ import annotations

from typing import Optional

from ._evaluation import build_pipeline
from ._spec import TuningSpec
from ._study_store import StudyStore, Trial


class TuningEngine:
    """Runs a ``TuningSpec`` over a calibration image set, journaling to a store."""

    def __init__(self, spec: TuningSpec, store: Optional[StudyStore] = None) -> None:
        self._spec = spec
        self._store = store if store is not None else StudyStore()

    @property
    def store(self) -> StudyStore:
        return self._store

    def best_pipeline(self):
        """Build the winning ``ImagePipeline`` from the best trial (or ``None``)."""
        best = self._store.best()
        if best is None:
            return None
        return build_pipeline(self._spec.pipeline, best.params)

    def optimize(self, images: list) -> Optional[Trial]:
        """Run the loop; return the best trial.

        Args:
            images: The calibration images (non-empty).

        Returns:
            The best :class:`Trial`, or ``None`` if none succeeded.
        """
        spec = self._spec
        strategy = spec.strategy.build(spec.search_space, self._store)

        # Resume: replay the deterministic strategy past recorded trials.
        completed = len(self._store)
        for _ in range(completed):
            if strategy.is_exhausted():
                break
            strategy.suggest()

        failures = 0
        number = completed
        while not strategy.is_exhausted():
            if spec.budget.n_trials is not None and number >= spec.budget.n_trials:
                break
            params, channel = strategy.suggest()
            params = dict(params)
            result = spec.evaluator.evaluate(
                spec.pipeline, spec.scorer, params, images
            )
            failed = len(result.terms) == 0
            self._store.append(
                Trial(
                    number=number,
                    params=params,
                    score=result.score,
                    terms=result.terms,
                    n_images=result.n_images,
                    failed=failed,
                )
            )
            strategy.register_result(params, result)
            number += 1
            if failed:
                failures += 1
                if (
                    spec.budget.max_failures is not None
                    and failures >= spec.budget.max_failures
                ):
                    break

        return self._store.best()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_engine.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_engine.py tests/unit/tune/test_engine.py
git add src/phenotypic/tune/_engine.py tests/unit/tune/test_engine.py
git commit -m "feat(tune): TuningEngine ask-and-tell loop + resume fast-forward"
```

---

### Task 4: RF-permutation `param_importance`

**Files:**
- Create: `src/phenotypic/tune/_screening.py`
- Test: `tests/unit/tune/test_param_importance.py`

The Phase-1 importance fallback (fANOVA needs Optuna, Phase 2): fit a `RandomForestRegressor`
on the trials' (encoded) params → score, then `permutation_importance`. Non-numeric params
are one-hot encoded (per-key prefix) and the encoded importances summed back to the original
key; absent conditional params fill to 0.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_param_importance.py
from __future__ import annotations

from phenotypic.tune._screening import compute_param_importance
from phenotypic.tune._study_store import StudyStore, Trial


def test_importance_finds_the_driving_param():
    # score depends entirely on `a` (True→1.0, False→0.0); `b` is noise.
    store = StudyStore()
    for i in range(24):
        a = i % 2 == 0
        b = (i // 2) % 3  # irrelevant
        store.append(Trial(
            number=i, params={"a": a, "b": b},
            score=1.0 if a else 0.0, terms={"Count": 1.0 if a else 0.0},
            n_images=2,
        ))
    imp = compute_param_importance(store)
    assert set(imp) == {"a", "b"}
    assert imp["a"] > imp["b"]


def test_importance_empty_below_two_trials():
    store = StudyStore()
    store.append(Trial(number=0, params={"a": 1}, score=0.5, terms={}, n_images=1))
    assert compute_param_importance(store) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_param_importance.py -v`
Expected: FAIL — `ModuleNotFoundError: ..._screening`.

- [ ] **Step 3: Implement `compute_param_importance`**

```python
# src/phenotypic/tune/_screening.py
"""Parameter importance — Phase-1 RF + permutation fallback (fANOVA is Phase 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from ._study_store import StudyStore


def compute_param_importance(
    store: StudyStore, *, random_state: int = 0
) -> dict[str, float]:
    """Rank tuned parameters by permutation importance against the objective.

    Args:
        store: The journal of completed trials.
        random_state: Seed for the forest + permutation (reproducibility).

    Returns:
        ``{param_key: importance}`` sorted descending. Empty when fewer than two
        non-failed trials (nothing to fit).
    """
    trials = [t for t in store.trials if not t.failed]
    if len(trials) < 2:
        return {}

    raw = pd.DataFrame([t.params for t in trials])
    y = np.asarray([t.score for t in trials], dtype=float)
    original_keys = list(raw.columns)

    numeric = raw.select_dtypes(include="number")
    non_numeric = raw.drop(columns=list(numeric.columns))

    parts: list[pd.DataFrame] = []
    col_to_key: dict[str, str] = {}

    for col in numeric.columns:
        series = numeric[col].astype(float)
        fill = float(series.median()) if series.notna().any() else 0.0
        parts.append(series.fillna(fill).to_frame(name=col))
        col_to_key[col] = col

    if not non_numeric.empty:
        dummies = pd.get_dummies(
            non_numeric.astype("object"), prefix_sep="=", dummy_na=False
        )
        for col in dummies.columns:
            col_to_key[col] = col.split("=", 1)[0]
        parts.append(dummies)

    features = pd.concat(parts, axis=1).fillna(0.0)
    if features.shape[1] == 0:
        return {}

    forest = RandomForestRegressor(n_estimators=200, random_state=random_state)
    forest.fit(features.to_numpy(), y)
    perm = permutation_importance(
        forest, features.to_numpy(), y, n_repeats=10, random_state=random_state
    )

    importances: dict[str, float] = {key: 0.0 for key in original_keys}
    for col, value in zip(features.columns, perm.importances_mean):
        importances[col_to_key[col]] += float(value)

    return dict(
        sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_param_importance.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_screening.py tests/unit/tune/test_param_importance.py
git add src/phenotypic/tune/_screening.py tests/unit/tune/test_param_importance.py
git commit -m "feat(tune): RF-permutation param importance fallback"
```

---

### Task 5: tune output-path helpers + the CLI

**Files:**
- Modify: `src/phenotypic/tools_/_io_constants.py`
- Create: `src/phenotypic/tune/_tune_cli/__init__.py`, `src/phenotypic/tune/_tune_cli/_run.py`, `src/phenotypic/tune/__main__.py`
- Test: `tests/unit/tune/test_tune_cli.py`

`run_tuning(spec, images, output_dir)` runs the engine and writes the `deliverables/`
artifacts; the CLI (`__main__`) parses `-i/-o`, loads images, and calls it. The output-path
helpers live beside the forward CLI's in `tools_/_io_constants.py` (never hand-join).

- [ ] **Step 1: Add the output-path helpers**

Append to `src/phenotypic/tools_/_io_constants.py` (next to `deliverables_dir`,
`pipeline_json_path`, etc.):

```python
def tuning_spec_path(output_dir: Path) -> Path:
    """Resolved ``tuning_spec.json`` (under ``deliverables/``)."""
    return deliverables_dir(output_dir) / "tuning_spec.json"


def best_pipeline_path(output_dir: Path) -> Path:
    """The winning ``best_pipeline.json`` (under ``deliverables/``)."""
    return deliverables_dir(output_dir) / "best_pipeline.json"


def param_importance_path(output_dir: Path) -> Path:
    """The ``param_importance.json`` report (under ``deliverables/``)."""
    return deliverables_dir(output_dir) / "param_importance.json"


def trials_parquet_path(output_dir: Path) -> Path:
    """The trial journal ``trials.parquet`` (at the output-dir root)."""
    return Path(output_dir) / "trials.parquet"
```

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/tune/test_tune_cli.py
from __future__ import annotations

import json

import pandas as pd

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tools_ import _io_constants as io
from phenotypic.tune import (
    Categorical, Evaluator, GridConfig, Knob, QCScorer, SearchSpace,
)
from phenotypic.tune._spec import Budget, TuningSpec
from phenotypic.tune._tune_cli._run import run_tuning


def _spec(tmp_path) -> TuningSpec:
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["Synthetic96PlateWithObjects"] * 96,
         "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=QCScorer(check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"])),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def test_run_tuning_writes_deliverables(tmp_path):
    out = tmp_path / "run"
    best = run_tuning(_spec(tmp_path), [load_synth_yeast_plate()], out)

    assert io.best_pipeline_path(out).exists()
    assert io.tuning_spec_path(out).exists()
    assert io.param_importance_path(out).exists()
    assert io.trials_parquet_path(out).exists()
    # the written best pipeline reloads as a runnable ImagePipeline
    winner = ImagePipeline.from_json(io.best_pipeline_path(out).read_text())
    assert "OtsuDetector" in winner.get_ops()
    # importance covers the tuned knob
    imp = json.loads(io.param_importance_path(out).read_text())
    assert "1.ignore_zeros" in imp
    assert best is not None


def test_cli_main_invokes_run(tmp_path, monkeypatch):
    from phenotypic.tune import __main__ as cli

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    out = tmp_path / "out"

    # patch image loading (no PNG fixtures needed)
    monkeypatch.setattr(cli, "_load_images", lambda _p: [load_synth_yeast_plate()])
    cli.main([str(spec_path), "-i", str(tmp_path), "-o", str(out)])

    assert io.best_pipeline_path(out).exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_tune_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: ..._tune_cli`.

- [ ] **Step 4: Implement `run_tuning` + the CLI**

```python
# src/phenotypic/tune/_tune_cli/_run.py
"""Run-a-tuning-spec orchestration + the ``deliverables/`` writes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from phenotypic import GridImage
from phenotypic.tools_ import _io_constants as io

from .._engine import TuningEngine
from .._screening import compute_param_importance
from .._spec import TuningSpec
from .._study_store import StudyStore, Trial

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".h5"}


def _load_images(input_dir: Path) -> list:
    """Load every image file under ``input_dir`` as a ``GridImage``.

    Mirrors the forward CLI's directory scan; tuning targets arrayed plates, so
    images load as ``GridImage`` via ``imread``.
    """
    paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    return [GridImage.imread(p) for p in paths]


def run_tuning(
    spec: TuningSpec, images: list, output_dir: Path
) -> Optional[Trial]:
    """Run ``spec`` over ``images`` and write the ``deliverables/`` artifacts.

    Writes ``trials.parquet`` (root), and under ``deliverables/``:
    ``tuning_spec.json`` (resolved spec), ``best_pipeline.json`` (the winner),
    ``param_importance.json``. Resumes if ``trials.parquet`` already exists.

    Args:
        spec: The tuning recipe (embeds the base pipeline + scorer + strategy).
        images: The calibration images.
        output_dir: The run directory.

    Returns:
        The best :class:`Trial`, or ``None`` if none succeeded.
    """
    output_dir = Path(output_dir)
    io.deliverables_dir(output_dir).mkdir(parents=True, exist_ok=True)

    trials_path = io.trials_parquet_path(output_dir)
    store = StudyStore.from_parquet(trials_path) if trials_path.exists() else StudyStore()

    engine = TuningEngine(spec, store=store)
    best = engine.optimize(images)

    store.to_parquet(trials_path)
    io.tuning_spec_path(output_dir).write_text(spec.model_dump_json(indent=2))
    io.param_importance_path(output_dir).write_text(
        json.dumps(compute_param_importance(store), indent=2)
    )
    winner = engine.best_pipeline()
    if winner is not None:
        io.best_pipeline_path(output_dir).write_text(winner.to_json())
    return best
```

```python
# src/phenotypic/tune/_tune_cli/__init__.py
"""The tune CLI (private)."""
from __future__ import annotations

from ._run import run_tuning

__all__ = ["run_tuning"]
```

```python
# src/phenotypic/tune/__main__.py
"""``python -m phenotypic.tune`` — run a tuning spec over an image directory."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from ._spec import TuningSpec
from ._tune_cli._run import _load_images, run_tuning


def _default_output(input_dir: str) -> Path:
    return Path(f"./{Path(input_dir).name}_tune")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m phenotypic.tune",
        description="Tune an ImagePipeline's parameters over an image set.",
    )
    parser.add_argument("spec", help="path to a tuning_spec.json")
    parser.add_argument("-i", "--input", required=True, help="image directory")
    parser.add_argument("-o", "--output", default=None, help="output directory")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point. See ``--help``."""
    args = _parse_args(argv)
    spec = TuningSpec.model_validate_json(Path(args.spec).read_text())
    output_dir = Path(args.output) if args.output else _default_output(args.input)
    images = _load_images(Path(args.input))
    if not images:
        raise SystemExit(f"no images found under {args.input!r}")
    run_tuning(spec, images, output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_tune_cli.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/tools_/_io_constants.py src/phenotypic/tune/_tune_cli src/phenotypic/tune/__main__.py tests/unit/tune/test_tune_cli.py
git add src/phenotypic/tools_/_io_constants.py src/phenotypic/tune/_tune_cli src/phenotypic/tune/__main__.py tests/unit/tune/test_tune_cli.py
git commit -m "feat(tune): run_tuning orchestration + python -m phenotypic.tune CLI"
```

---

### Task 6: The grid byte-compat lock (vs. the Phase-0 golden)

**Files:**
- Test: `tests/unit/tune/test_grid_byte_compat_lock.py`

The lock that lets `sweep` be deleted: the tune grid (`enumerate_grid` → `build_pipeline`)
must reproduce the **same set of operation pipelines** as the frozen golden
`generate_sweep_manifest` output. (Legacy pipelines carry deterministic `Pipeline_N`
names; tune clones carry the base's uuid — so the invariant is *operation-combination
equivalence*, not literal manifest-byte equality. The golden is reconstructed via
`ImagePipeline.from_json` from the committed JSON, so this test stands after `sweep` is
deleted in Task 7.)

The golden config (Phase 0): `Presence(GaussianBlur, sigma=(1.0, 2.0))` +
`Sweep(OtsuDetector, ignore_zeros=(True, False))` → 6 pipelines. Its tune equivalent: base
`[GaussianBlur(sigma=1.0), OtsuDetector()]` (the blur's `sigma` is always overlaid, so the
base value is irrelevant) + the presence/sweep `SearchSpace`.

- [ ] **Step 1: Write the lock test**

```python
# tests/unit/tune/test_grid_byte_compat_lock.py
from __future__ import annotations

import json
from pathlib import Path

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tune import Categorical, Knob, SearchSpace
from phenotypic.tune._evaluation import build_pipeline
from phenotypic.tune._strategies._enumerate import enumerate_grid

GOLDEN = (
    Path(__file__).resolve().parents[3]
    / "tests/fixtures/tune/grid_golden_manifest.json"
)


def _signature(pipe: ImagePipeline) -> tuple:
    """Order-sensitive op signature, name/uuid-independent."""
    return tuple(
        (type(op).__name__, json.dumps(op.model_dump(mode="json"),
                                       sort_keys=True, default=str))
        for op in pipe.get_ops().values()
    )


def _golden_signatures() -> set:
    manifest = json.loads(GOLDEN.read_text())
    sigs = set()
    for cfg in manifest["configs"].values():
        for pipe_dict in cfg["pipelines"].values():
            pipe = ImagePipeline.from_json(json.dumps(pipe_dict))
            sigs.add(_signature(pipe))
    return sigs


def _tune_space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.GaussianBlur.__enabled__",
             domain=Categorical(choices=(True, False)), source="presence_optin"),
        Knob(key="0.sigma", domain=Categorical(choices=(1.0, 2.0)),
             conditional_on=(("0.GaussianBlur.__enabled__", True),)),
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def test_tune_grid_reproduces_golden_op_combinations():
    base = ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()])
    combos = enumerate_grid(_tune_space())
    assert len(combos) == 6
    tune_sigs = {_signature(build_pipeline(base, c)) for c in combos}
    assert tune_sigs == _golden_signatures()
```

- [ ] **Step 2: Run the lock (while `sweep`/golden exist)**

Run: `uv run pytest tests/unit/tune/test_grid_byte_compat_lock.py -v`
Expected: PASS (1 test) — the tune grid equals the frozen golden's op-combinations.

> If this fails, **do not** edit the golden — the divergence means `build_pipeline` /
> `enumerate_grid` changed observable output. Investigate before proceeding to Task 7.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/tune/test_grid_byte_compat_lock.py
git commit -m "test(tune): grid byte-compat lock vs Phase-0 golden manifest"
```

---

### Task 7: Hard cutover — public exports, delete `sweep`, migration script

**Files:**
- Modify: `src/phenotypic/tune/__init__.py`
- Delete: `src/phenotypic/sweep/`
- Create: `scripts/migrate_sweep_manifest.py`
- Test: `tests/unit/tune/test_migrate_manifest.py`
- Delete: `tests/unit/tune/test_grid_golden_manifest.py`'s sweep-importing test (the
  belt-and-suspenders `test_golden_matches_fresh_generation_while_sweep_exists`); keep
  `test_golden_exists_and_is_stable`.

- [ ] **Step 1: Export the Phase-1d public surface**

```python
# src/phenotypic/tune/__init__.py  (extend; preserve 1a/1b/1c exports)
from ._engine import TuningEngine
from ._screening import compute_param_importance
from ._spec import Budget, TuningSpec
from ._study_store import StudyStore, Trial
from ._tune_cli import run_tuning

# add to __all__:
#   "TuningEngine", "TuningSpec", "Budget", "StudyStore", "Trial",
#   "compute_param_importance", "run_tuning"
```

- [ ] **Step 2: Write the migration test (before deleting sweep)**

```python
# tests/unit/tune/test_migrate_manifest.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune import QCScorer
from phenotypic.tune._evaluation import build_pipeline
from phenotypic.tune._strategies._enumerate import enumerate_grid

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from migrate_sweep_manifest import migrate_manifest_to_spec  # noqa: E402

GOLDEN = (
    Path(__file__).resolve().parents[3]
    / "tests/fixtures/tune/grid_golden_manifest.json"
)


def _scorer(tmp_path) -> QCScorer:
    csv = tmp_path / "layout.csv"
    pd.DataFrame({"Metadata_ImageName": ["p"] * 96,
                  "Object_Label": list(range(96))}).to_csv(csv, index=False)
    return QCScorer(check=ExpectedVsDetectedCount(
        metadata=str(csv), groupby=["Metadata_ImageName"]))


def _sig(pipe):
    return tuple((type(o).__name__,
                  json.dumps(o.model_dump(mode="json"), sort_keys=True, default=str))
                 for o in pipe.get_ops().values())


def test_migrated_spec_grid_matches_manifest(tmp_path):
    manifest = json.loads(GOLDEN.read_text())
    spec = migrate_manifest_to_spec(manifest, scorer=_scorer(tmp_path))
    combos = enumerate_grid(spec.search_space)
    migrated = {_sig(build_pipeline(spec.pipeline, c)) for c in combos}
    # the migrated grid reproduces the manifest's op-combinations
    golden = set()
    for cfg in manifest["configs"].values():
        for pd_ in cfg["pipelines"].values():
            golden.add(_sig(ImagePipeline.from_json(json.dumps(pd_))))
    assert migrated == golden
```

- [ ] **Step 3: Implement the migration script**

```python
# scripts/migrate_sweep_manifest.py
"""Migrate a legacy ``generate_sweep_manifest`` JSON to a ``tuning_spec.json``.

The hard cutover (master §9) deletes ``sweep``; this converts an existing
manifest into the new ``TuningSpec``. MVP scope: a single config of flat +
presence sweeps (the shape ``generate_sweep_manifest`` produced) — it derives
the base pipeline (the op-richest variant), a ``SearchSpace`` (Categorical knobs
over the per-op param values observed; a ``__enabled__`` presence knob for ops
absent in some variants), and a ``GridConfig``. The user supplies the
``Scorer``. Nested-op manifests raise ``NotImplementedError``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune import (
    Categorical, Evaluator, GridConfig, Knob, QCScorer, SearchSpace,
)
from phenotypic.tune._scoring import Scorer
from phenotypic.tune._spec import Budget, TuningSpec


def _pipelines(manifest: dict) -> list[ImagePipeline]:
    pipes: list[ImagePipeline] = []
    for cfg in manifest["configs"].values():
        for pipe_dict in cfg["pipelines"].values():
            pipes.append(ImagePipeline.from_json(json.dumps(pipe_dict)))
    return pipes


def migrate_manifest_to_spec(manifest: dict, *, scorer: Scorer) -> TuningSpec:
    """Convert a legacy manifest into a ``TuningSpec`` (see module docstring)."""
    pipes = _pipelines(manifest)
    if not pipes:
        raise ValueError("manifest has no pipelines")

    # ops keyed by class name, in first-seen order; the op-richest variant is base.
    base_pipe = max(pipes, key=lambda p: len(p.get_ops()))
    base_ops = list(base_pipe.get_ops().values())
    base_classes = [type(op).__name__ for op in base_ops]

    knobs: list[Knob] = []
    for position, op in enumerate(base_ops):
        cls = base_classes[position]
        # which variants contain this position's class?
        present = [p for p in pipes if cls in {type(o).__name__ for o in p.get_ops().values()}]
        optional = len(present) < len(pipes)
        enabled_key = f"{position}.{cls}.__enabled__"
        if optional:
            knobs.append(Knob(
                key=enabled_key,
                domain=Categorical(choices=(True, False)),
                source="presence_optin",
            ))
        # per-field varying values across the present variants
        fields: dict[str, set] = {}
        for p in present:
            op_i = next(o for o in p.get_ops().values() if type(o).__name__ == cls)
            for fname in type(op_i).model_fields:
                fields.setdefault(fname, set()).add(_hashable(getattr(op_i, fname)))
        for fname, values in fields.items():
            if len(values) <= 1:
                continue  # constant → not a knob
            knob_kwargs: dict[str, Any] = dict(
                key=f"{position}.{fname}",
                domain=Categorical(choices=tuple(sorted(values, key=repr))),
            )
            if optional:
                knob_kwargs["conditional_on"] = ((enabled_key, True),)
            knobs.append(Knob(**knob_kwargs))

    return TuningSpec(
        pipeline=base_pipe,
        search_space=SearchSpace(knobs=tuple(knobs)),
        scorer=scorer,
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def _hashable(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        raise NotImplementedError(
            f"non-hashable swept param value {value!r}; nested-op manifests are "
            "not supported by the MVP migration"
        )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate a sweep manifest to a tuning_spec.json")
    parser.add_argument("manifest", help="legacy manifest JSON")
    parser.add_argument("-o", "--output", required=True, help="tuning_spec.json to write")
    parser.add_argument("--metadata", required=True, help="layout CSV/Parquet for the Count scorer")
    parser.add_argument("--groupby", nargs="+", default=["Metadata_ImageName"])
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    scorer = QCScorer(check=ExpectedVsDetectedCount(
        metadata=args.metadata, groupby=list(args.groupby)))
    spec = migrate_manifest_to_spec(manifest, scorer=scorer)
    Path(args.output).write_text(spec.model_dump_json(indent=2))
    print(f"Wrote {args.output} ({len(spec.search_space.knobs)} knobs)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the migration test (sweep still present)**

Run: `uv run pytest tests/unit/tune/test_migrate_manifest.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Delete `sweep` + its sweep-importing test, then prove the suite is green**

```bash
git rm -r src/phenotypic/sweep
# remove the belt-and-suspenders test that imports sweep (keep the golden-exists test):
#   edit tests/unit/tune/test_grid_golden_manifest.py — delete
#   test_golden_matches_fresh_generation_while_sweep_exists
```

Edit `tests/unit/tune/test_grid_golden_manifest.py`: delete the
`test_golden_matches_fresh_generation_while_sweep_exists` function (it imports
`phenotypic.sweep`); keep `test_golden_exists_and_is_stable`. Then grep for stragglers:

```bash
grep -rn "phenotypic.sweep\|from ..sweep\|import sweep" src/ tests/ docs/ pyproject.toml || echo "no sweep references"
```

Resolve any remaining references (e.g. a `[project.scripts]` `phenotypic-sweep` entry in
`pyproject.toml`, README mentions). The byte-compat lock (Task 6) and golden-exists test
must still pass **without** `sweep` (they read the frozen JSON via core `from_json`).

- [ ] **Step 6: Full Phase-1 gate**

Run:
```bash
uv run pytest tests/unit/tune -q
uv run pytest --doctest-modules src/phenotypic/tune -q
uv run mypy src/phenotypic/tune
uv run ruff check src/phenotypic/tune tests/unit/tune scripts/migrate_sweep_manifest.py
```
Expected: all green; no `phenotypic.sweep` importable.

- [ ] **Step 7: Commit the cutover**

```bash
git add -A
git commit -m "feat(tune): hard cutover — delete sweep, ship manifest->spec migration, export engine API"
```

---

## Self-Review

**Spec coverage:**
- engine-arch §5 `Budget` → Task 2; §6 `TuningSpec` *with embedded pipeline* (the applied refinement) + polymorphic `scorer` → Task 2; §8 `TuningEngine` ask-and-tell loop → Task 3; §10 layout (`_study_store`/`_spec`/`_engine`/`_screening`/`_tune_cli`) → Tasks 1–5.
- master §6 CLI `-i/-o` + default `./<input>_tune/` + resume-by-repointing → Tasks 3 (resume), 5 (CLI); §8 `deliverables/` layout (best_pipeline/tuning_spec/param_importance + root trials.parquet) via `tools_` helpers → Task 5; §9 hard cutover (delete sweep) + golden lock + migration → Tasks 6–7.
- screening-importance RF+permutation fallback → Task 4.

**Deferred (correctly, with later phases noted in code/docstrings):** Optuna SQLite `study.db`, ASHA pruning, fANOVA, `SlurmExecutor`, multi-objective `pareto/`, `tuning_report.html`, `screening/`/`splits/` dirs, metadata-stratified calibration split + held-out guard, `--auto-space`/`--screen`/`--multi-objective` CLI flags. Phase 1d ships the **CV-only-MVP happy path**: load images → grid/random → journal → best + importance.

**Placeholder scan:** none — every code/test step is complete. The embedded-pipeline custom serializer, the `ScorerField` round-trip, the builder/measure stack, the 6-combo grid, and the golden op-set were all verified against the live codebase before writing.

**Type consistency:** `Trial(number, params, score, terms, n_images, failed)`; `StudyStore.append/best/to_parquet/from_parquet`; `Budget(n_trials, max_failures)`; `TuningSpec(pipeline, search_space, scorer, evaluator, strategy, budget)`; `TuningEngine(spec, store).optimize(images) -> Trial | None` + `.best_pipeline()`; `compute_param_importance(store) -> dict[str,float]`; `run_tuning(spec, images, output_dir) -> Trial | None`; `migrate_manifest_to_spec(manifest, *, scorer) -> TuningSpec`. These reuse 1a (`SearchSpace`/`Knob`/`Categorical`), 1b (`GridConfig`/`StrategyConfigUnion`/`enumerate_grid`/`build`), and 1c (`Scorer`/`QCScorer`/`Evaluator`/`build_pipeline`) names exactly.

## Phase 1 complete

After 1d, `phenotypic.tune` is a runnable engine (`python -m phenotypic.tune spec.json -i … -o …`), `sweep` is gone, and the grid path is locked to the frozen golden. **Next:** Phase 2 (Optuna backend — `OptunaStrategy`/`OptunaConfig`, ASHA, SQLite `study.db`, fANOVA, `SlurmExecutor`) and the parallel operation-annotations workstream — both planned as structured outlines.
