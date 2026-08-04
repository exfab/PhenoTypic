# Tune Engine — Phase 1b: Search Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The swappable optimizer seam — the `SearchStrategy` Protocol plus the two zero-dependency strategies (`GridStrategy`, `RandomStrategy`), their serializable `StrategyConfig`s, and the `PruningChannel` Protocol + `NoOpChannel`.

**Architecture:** `SearchStrategy` is a runtime Protocol (engine-architecture.md §4); strategies hold runtime state and are constructed from a pydantic `StrategyConfig` via `config.build(space, store)`. `GridStrategy` enumerates the conditional `SearchSpace` Cartesian product (absent-collapses-to-one, mirroring legacy `Presence`); `RandomStrategy` samples it under a fixed seed. Both return a `NoOpChannel` (pruning is Phase 2). The **byte-compat lock against the golden manifest** lands in **Phase 1d** (it needs the params→pipeline builder); 1b tests enumeration against an independently-computed expected combo set.

**Tech Stack:** pydantic v2, `typing.Protocol`/`runtime_checkable`, stdlib `random`. No new deps.

**Spec:** `engine-architecture.md` §4; `optuna-integration.md` §3 (the ask-and-tell shape the Protocol mirrors); master §9 (the grid regression lock). **Depends on:** Phase 1a (`SearchSpace`/`Knob`/`Domain`/`Categorical`/`IntRange`/`FloatRange`/`Fixed`).

**Conventions:** `uv run pytest`, `uv run mypy src/phenotypic/tune`, `uv run ruff check --fix`; Google docstrings; tests under `tests/unit/tune/`.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/phenotypic/tune/_strategies/__init__.py` | internal re-exports | Create |
| `src/phenotypic/tune/_strategies/_pruning.py` | `PruningChannel` Protocol + `NoOpChannel` | Create |
| `src/phenotypic/tune/_strategies/_protocol.py` | `SearchStrategy` Protocol | Create |
| `src/phenotypic/tune/_strategies/_enumerate.py` | `grid_values(domain)` + `enumerate_grid(space)` (the conditional Cartesian) | Create |
| `src/phenotypic/tune/_strategies/_grid.py` | `GridStrategy` | Create |
| `src/phenotypic/tune/_strategies/_random.py` | `RandomStrategy` | Create |
| `src/phenotypic/tune/_strategies/_config.py` | `StrategyConfig` ABC + `GridConfig`/`RandomConfig` | Create |
| `tests/unit/tune/test_pruning_channel.py` | `NoOpChannel` | Create |
| `tests/unit/tune/test_grid_enumerate.py` | conditional Cartesian enumeration | Create |
| `tests/unit/tune/test_strategies.py` | Protocol conformance, Grid/Random behavior, configs | Create |

---

### Task 1: `PruningChannel` Protocol + `NoOpChannel`

**Files:**
- Create: `src/phenotypic/tune/_strategies/_pruning.py`, `src/phenotypic/tune/_strategies/__init__.py`
- Test: `tests/unit/tune/test_pruning_channel.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_pruning_channel.py
from __future__ import annotations

from phenotypic.tune._strategies import NoOpChannel, PruningChannel


def test_noop_channel_never_prunes():
    ch = NoOpChannel()
    ch.report(0.5, step=3)  # no-op, must not raise
    assert ch.should_prune() is False


def test_noop_satisfies_protocol():
    assert isinstance(NoOpChannel(), PruningChannel)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_pruning_channel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.tune._strategies'`.

- [ ] **Step 3: Implement**

```python
# src/phenotypic/tune/_strategies/_pruning.py
"""The pruning channel: how the Evaluator reports + checks early-stop.

Phase 1 ships only the no-op (Grid/Random never prune). Phase 2 adds an
Optuna-trial-backed channel. Keeping this a Protocol keeps the Evaluator
Optuna-free (robust-evaluation.md §7, optuna-integration.md §6).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PruningChannel(Protocol):
    def report(self, value: float, step: int) -> None: ...
    def should_prune(self) -> bool: ...


class NoOpChannel:
    """Never prunes; ``report`` is a no-op. Used by Grid/Random strategies."""

    def report(self, value: float, step: int) -> None:
        return None

    def should_prune(self) -> bool:
        return False
```

```python
# src/phenotypic/tune/_strategies/__init__.py
"""Search strategies (private)."""
from __future__ import annotations

from ._pruning import NoOpChannel, PruningChannel

__all__ = ["PruningChannel", "NoOpChannel"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_pruning_channel.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_strategies tests/unit/tune/test_pruning_channel.py
git add src/phenotypic/tune/_strategies tests/unit/tune/test_pruning_channel.py
git commit -m "feat(tune): PruningChannel Protocol + NoOpChannel"
```

---

### Task 2: Conditional grid enumeration

The reusable core: turn a `SearchSpace` into the list of param dicts (the conditional Cartesian product). Used by `GridStrategy` and tested in isolation.

**Files:**
- Create: `src/phenotypic/tune/_strategies/_enumerate.py`
- Test: `tests/unit/tune/test_grid_enumerate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_grid_enumerate.py
from __future__ import annotations

import pytest

from phenotypic.tune import (
    Categorical,
    FloatRange,
    IntRange,
    Knob,
    SearchSpace,
)
from phenotypic.tune._strategies._enumerate import enumerate_grid, grid_values


def test_grid_values_per_domain():
    assert grid_values(Categorical(choices=(True, False))) == [True, False]
    assert grid_values(IntRange(low=2, high=8, step=2)) == [2, 4, 6, 8]
    from phenotypic.tune import Fixed
    assert grid_values(Fixed(value="x")) == ["x"]


def test_grid_values_rejects_floatrange():
    with pytest.raises(ValueError, match="continuous|FloatRange"):
        grid_values(FloatRange(low=0.0, high=1.0))


def test_enumerate_conditional_absent_collapses():
    # Mirrors the golden config: Presence(BlurGauss, sigma=(1,2)) + Sweep(Otsu, ignore_zeros=(T,F))
    space = SearchSpace(knobs=(
        Knob(key="0.BlurGauss.__enabled__",
             domain=Categorical(choices=(True, False)), source="presence_optin"),
        Knob(key="0.BlurGauss.sigma",
             domain=Categorical(choices=(1.0, 2.0)),
             conditional_on=(("0.BlurGauss.__enabled__", True),)),
        Knob(key="1.OtsuDetector.ignore_zeros",
             domain=Categorical(choices=(True, False))),
    ))
    combos = enumerate_grid(space)
    # absent: enabled=False → sigma omitted → 2 (× ignore_zeros); present: 2 sigmas × 2 = 4 → total 6
    assert len(combos) == 6
    absent = [c for c in combos if c["0.BlurGauss.__enabled__"] is False]
    assert len(absent) == 2
    assert all("0.BlurGauss.sigma" not in c for c in absent)
    present = [c for c in combos if c["0.BlurGauss.__enabled__"] is True]
    assert len(present) == 4
    assert all("0.BlurGauss.sigma" in c for c in present)


def test_enumerate_unconditional_only():
    space = SearchSpace(knobs=(
        Knob(key="a", domain=Categorical(choices=(1, 2))),
        Knob(key="b", domain=IntRange(low=1, high=2)),
    ))
    assert len(enumerate_grid(space)) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_grid_enumerate.py -v`
Expected: FAIL — `ModuleNotFoundError: ..._enumerate`.

- [ ] **Step 3: Implement enumeration**

```python
# src/phenotypic/tune/_strategies/_enumerate.py
"""Conditional Cartesian-product enumeration of a SearchSpace (grid search).

Respects ``conditional_on`` — a child knob is included only when its parent
presence knob takes the matching value, so an absent op collapses to a single
combination (the legacy ``Presence`` semantics, master §9).
"""
from __future__ import annotations

import itertools
from typing import Any

from .._search_space import Categorical, Domain, Fixed, IntRange, SearchSpace


def grid_values(domain: Domain) -> list[Any]:
    """The discrete grid values for a domain. ``FloatRange`` is not enumerable."""
    if isinstance(domain, Categorical):
        return list(domain.choices)
    if isinstance(domain, IntRange):
        return list(range(domain.low, domain.high + 1, domain.step))
    if isinstance(domain, Fixed):
        return [domain.value]
    raise ValueError(
        "GridStrategy cannot enumerate a continuous FloatRange; use "
        "Categorical / IntRange, or a non-grid strategy."
    )


def _is_active(knob, chosen: dict[str, Any]) -> bool:
    if knob.conditional_on is None:
        return True
    return all(chosen.get(pkey) == pval for pkey, pval in knob.conditional_on)


def enumerate_grid(space: SearchSpace) -> list[dict[str, Any]]:
    """All param dicts in the conditional Cartesian product.

    Unconditional knobs (including presence ``__enabled__`` knobs) form the
    outer product; each conditional knob is only assigned when its parent
    value is present in the combination.
    """
    roots = [k for k in space.knobs if k.conditional_on is None]
    conditionals = [k for k in space.knobs if k.conditional_on is not None]

    combos: list[dict[str, Any]] = []
    root_values = [grid_values(k.domain) for k in roots]
    for root_combo in itertools.product(*root_values):
        base = {k.key: v for k, v in zip(roots, root_combo)}
        active = [k for k in conditionals if _is_active(k, base)]
        if not active:
            combos.append(dict(base))
            continue
        cond_values = [grid_values(k.domain) for k in active]
        for cond_combo in itertools.product(*cond_values):
            full = dict(base)
            full.update({k.key: v for k, v in zip(active, cond_combo)})
            combos.append(full)
    return combos
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_grid_enumerate.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_strategies/_enumerate.py tests/unit/tune/test_grid_enumerate.py
git add src/phenotypic/tune/_strategies/_enumerate.py tests/unit/tune/test_grid_enumerate.py
git commit -m "feat(tune): conditional grid enumeration"
```

---

### Task 3: `SearchStrategy` Protocol + `GridStrategy` + `RandomStrategy`

**Files:**
- Create: `src/phenotypic/tune/_strategies/_protocol.py`, `_grid.py`, `_random.py`
- Modify: `src/phenotypic/tune/_strategies/__init__.py`
- Test: `tests/unit/tune/test_strategies.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_strategies.py
from __future__ import annotations

from phenotypic.tune import Categorical, FloatRange, Knob, SearchSpace
from phenotypic.tune._strategies import (
    GridStrategy,
    NoOpChannel,
    RandomStrategy,
    SearchStrategy,
)


def _conditional_space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="g.__enabled__", domain=Categorical(choices=(True, False))),
        Knob(key="g.sigma", domain=FloatRange(low=0.5, high=5.0),
             conditional_on=(("g.__enabled__", True),)),
        Knob(key="d.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def _grid_space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="g.__enabled__", domain=Categorical(choices=(True, False))),
        Knob(key="g.sigma", domain=Categorical(choices=(1.0, 2.0)),
             conditional_on=(("g.__enabled__", True),)),
        Knob(key="d.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def test_strategies_satisfy_protocol():
    assert isinstance(GridStrategy(_grid_space()), SearchStrategy)
    assert isinstance(RandomStrategy(_conditional_space(), n_trials=3, seed=0),
                      SearchStrategy)


def test_grid_exhausts_after_enumeration():
    strat = GridStrategy(_grid_space())
    seen = []
    while not strat.is_exhausted():
        params, channel = strat.suggest()
        assert isinstance(channel, NoOpChannel)
        strat.register_result(params, result=None)  # grid ignores results
        seen.append(params)
    assert len(seen) == 6  # the conditional Cartesian product


def test_random_respects_conditionals_and_seed():
    a = RandomStrategy(_conditional_space(), n_trials=20, seed=42)
    seq_a = []
    while not a.is_exhausted():
        p, _ = a.suggest()
        a.register_result(p, result=None)
        seq_a.append(p)
        # sigma present iff blur enabled
        assert ("g.sigma" in p) == (p["g.__enabled__"] is True)
    assert len(seq_a) == 20

    b = RandomStrategy(_conditional_space(), n_trials=20, seed=42)
    seq_b = []
    while not b.is_exhausted():
        p, _ = b.suggest()
        b.register_result(p, result=None)
        seq_b.append(p)
    assert seq_a == seq_b  # seeded determinism


def test_grid_rejects_floatrange():
    import pytest
    with pytest.raises(ValueError):
        GridStrategy(_conditional_space())  # has a FloatRange knob
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_strategies.py -v`
Expected: FAIL — `ImportError: cannot import name 'GridStrategy'`.

- [ ] **Step 3: Implement the Protocol + strategies**

```python
# src/phenotypic/tune/_strategies/_protocol.py
"""The SearchStrategy seam (engine-architecture.md §4)."""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from ._pruning import PruningChannel


@runtime_checkable
class SearchStrategy(Protocol):
    def suggest(self) -> "tuple[Mapping[str, Any], PruningChannel]": ...
    def register_result(self, params: Mapping[str, Any], result: Any,
                        *, pruned: bool = False) -> None: ...
    def is_exhausted(self) -> bool: ...
```

```python
# src/phenotypic/tune/_strategies/_grid.py
"""GridStrategy — exhaustive enumeration (one trial per combo)."""
from __future__ import annotations

from typing import Any, Mapping

from .._search_space import SearchSpace
from ._enumerate import enumerate_grid
from ._pruning import NoOpChannel, PruningChannel


class GridStrategy:
    """Walks the full conditional Cartesian product of the SearchSpace.

    The degenerate "strategy" that reproduces exhaustive grid search. Raises
    on a continuous ``FloatRange`` knob (not enumerable).
    """

    def __init__(self, space: SearchSpace) -> None:
        self._combos = enumerate_grid(space)  # raises on FloatRange
        self._cursor = 0

    def suggest(self) -> "tuple[Mapping[str, Any], PruningChannel]":
        params = dict(self._combos[self._cursor])  # defensive copy: callers must not mutate the stored combo
        self._cursor += 1
        return params, NoOpChannel()

    def register_result(self, params: Mapping[str, Any], result: Any,
                        *, pruned: bool = False) -> None:
        return None  # grid does not learn

    def is_exhausted(self) -> bool:
        return self._cursor >= len(self._combos)
```

```python
# src/phenotypic/tune/_strategies/_random.py
"""RandomStrategy — seeded random sampling over the SearchSpace."""
from __future__ import annotations

import math
import random
from typing import Any, Mapping

from .._search_space import (
    Categorical,
    Fixed,
    FloatRange,
    IntRange,
    SearchSpace,
)
from ._pruning import NoOpChannel, PruningChannel


class RandomStrategy:
    """Samples ``n_trials`` random configurations under a fixed seed.

    Respects ``conditional_on``: a child knob is sampled only when its parent
    presence value was sampled to match.
    """

    def __init__(self, space: SearchSpace, *, n_trials: int, seed: int = 0) -> None:
        self._space = space
        self._n = n_trials
        self._rng = random.Random(seed)
        self._count = 0

    def _sample_domain(self, domain: Any) -> Any:
        if isinstance(domain, Categorical):
            return self._rng.choice(list(domain.choices))
        if isinstance(domain, IntRange):
            vals = list(range(domain.low, domain.high + 1, domain.step))
            return self._rng.choice(vals)
        if isinstance(domain, FloatRange):
            if domain.log:
                lo, hi = math.log(domain.low), math.log(domain.high)
                return math.exp(self._rng.uniform(lo, hi))
            return self._rng.uniform(domain.low, domain.high)
        if isinstance(domain, Fixed):
            return domain.value
        raise TypeError(f"unsupported domain {type(domain).__name__}")

    def suggest(self) -> "tuple[Mapping[str, Any], PruningChannel]":
        chosen: dict[str, Any] = {}
        # Sample knobs in order; a conditional knob is sampled only if active.
        for knob in self._space.knobs:
            if knob.conditional_on is not None and not all(
                chosen.get(pk) == pv for pk, pv in knob.conditional_on
            ):
                continue
            chosen[knob.key] = self._sample_domain(knob.domain)
        self._count += 1
        return chosen, NoOpChannel()

    def register_result(self, params: Mapping[str, Any], result: Any,
                        *, pruned: bool = False) -> None:
        return None  # random does not learn

    def is_exhausted(self) -> bool:
        return self._count >= self._n
```

> Ordering note: `RandomStrategy` relies on a parent presence knob appearing **before** its conditional children in `space.knobs` (true for inferred/hand-authored spaces — `__enabled__` is emitted first). A topological pass is unnecessary at depth-cap-1; document this precondition in the `RandomStrategy` docstring.

Update the subpackage `__init__.py`:

```python
# src/phenotypic/tune/_strategies/__init__.py  (replace file)
"""Search strategies (private)."""
from __future__ import annotations

from ._grid import GridStrategy
from ._protocol import SearchStrategy
from ._pruning import NoOpChannel, PruningChannel
from ._random import RandomStrategy

__all__ = [
    "SearchStrategy", "PruningChannel", "NoOpChannel",
    "GridStrategy", "RandomStrategy",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_strategies.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_strategies tests/unit/tune/test_strategies.py
git add src/phenotypic/tune/_strategies tests/unit/tune/test_strategies.py
git commit -m "feat(tune): SearchStrategy Protocol + GridStrategy + RandomStrategy"
```

---

### Task 4: `StrategyConfig` (serializable, with `build()`)

**Files:**
- Create: `src/phenotypic/tune/_strategies/_config.py`
- Modify: `src/phenotypic/tune/_strategies/__init__.py`, `src/phenotypic/tune/__init__.py`
- Test: `tests/unit/tune/test_strategy_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_strategy_config.py
from __future__ import annotations

from phenotypic.tune import Categorical, GridConfig, Knob, RandomConfig, SearchSpace
from phenotypic.tune._strategies import GridStrategy, RandomStrategy


def _space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="a", domain=Categorical(choices=(1, 2))),
    ))


def test_grid_config_builds_grid_strategy():
    strat = GridConfig().build(_space(), store=None)
    assert isinstance(strat, GridStrategy)


def test_random_config_builds_random_strategy():
    cfg = RandomConfig(n_trials=7, seed=3)
    strat = cfg.build(_space(), store=None)
    assert isinstance(strat, RandomStrategy)
    assert strat._n == 7


def test_config_roundtrips_via_discriminator():
    from pydantic import TypeAdapter
    from phenotypic.tune._strategies._config import StrategyConfigUnion

    adapter = TypeAdapter(StrategyConfigUnion)
    cfg = RandomConfig(n_trials=5, seed=1)
    back = adapter.validate_json(adapter.dump_json(cfg))
    assert back == cfg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_strategy_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'GridConfig'`.

- [ ] **Step 3: Implement the configs**

```python
# src/phenotypic/tune/_strategies/_config.py
"""Serializable strategy configs; each builds its live SearchStrategy.

These are a closed set in Phase 1 (grid/random) → a discriminated union.
Phase 2 adds ``OptunaConfig``; the polymorphic-field path (engine-architecture
§6) lets the open Scorer/Strategy sets extend, but the in-spec config field uses
this union for the built-in kinds.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .._search_space import SearchSpace
from ._grid import GridStrategy
from ._protocol import SearchStrategy
from ._random import RandomStrategy


class StrategyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    seed: int = 0

    @abstractmethod
    def build(self, space: SearchSpace, store: Optional[Any]) -> SearchStrategy: ...


class GridConfig(StrategyConfig):
    kind: Literal["grid"] = "grid"

    def build(self, space: SearchSpace, store: Optional[Any]) -> SearchStrategy:
        return GridStrategy(space)


class RandomConfig(StrategyConfig):
    kind: Literal["random"] = "random"
    n_trials: int

    def build(self, space: SearchSpace, store: Optional[Any]) -> SearchStrategy:
        return RandomStrategy(space, n_trials=self.n_trials, seed=self.seed)


#: Discriminated union of the built-in (Phase 1) strategy configs.
StrategyConfigUnion = Annotated[
    Union[GridConfig, RandomConfig], Field(discriminator="kind")
]
```

> Note: `StrategyConfig` is an abstract base; pydantic allows `BaseModel` + `@abstractmethod` (a subclass is required to instantiate). `GridConfig`/`RandomConfig` are concrete. The `TuningSpec.strategy` field (Phase 1d) is typed `StrategyConfigUnion` for Phase-1 kinds; Phase 2 widens it to add `OptunaConfig`.

Update `_strategies/__init__.py` and `tune/__init__.py` to export `StrategyConfig`, `GridConfig`, `RandomConfig`.

```python
# add to src/phenotypic/tune/_strategies/__init__.py __all__ + imports
from ._config import GridConfig, RandomConfig, StrategyConfig, StrategyConfigUnion
# __all__ += ["StrategyConfig", "GridConfig", "RandomConfig", "StrategyConfigUnion"]
```

```python
# add to src/phenotypic/tune/__init__.py
from ._strategies import GridConfig, RandomConfig, StrategyConfig
# __all__ += ["StrategyConfig", "GridConfig", "RandomConfig"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_strategy_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Types/lint/commit**

```bash
uv run mypy src/phenotypic/tune/_strategies && uv run ruff check --fix src/phenotypic/tune/_strategies tests/unit/tune/test_strategy_config.py
git add src/phenotypic/tune/_strategies/_config.py src/phenotypic/tune/_strategies/__init__.py src/phenotypic/tune/__init__.py tests/unit/tune/test_strategy_config.py
git commit -m "feat(tune): StrategyConfig (grid/random) with build() factory"
```

---

## Self-Review

**Spec coverage:** `SearchStrategy` Protocol + `suggest`/`register_result`/`is_exhausted` → Task 3; `GridStrategy` conditional enumeration (absent-collapses) → Tasks 2–3; `RandomStrategy` seeded + conditional-aware → Task 3; `PruningChannel` + `NoOpChannel` → Task 1; `StrategyConfig.build()` factory → Task 4 (engine-architecture §4). The **byte-compat golden lock** is deferred to **Phase 1d** (needs the params→pipeline builder) — Task 2 here proves the enumeration count/structure against an independent expectation.

**Placeholder scan:** none. The two notes (RandomStrategy ordering precondition; `StrategyConfig` abstract-base behavior) are concrete clarifications.

**Type consistency:** `SearchStrategy.suggest() -> (Mapping[str, Any], PruningChannel)` and `register_result(params, result, *, pruned=False)` are consistent across `GridStrategy`/`RandomStrategy`/the Protocol/the configs, and match what Phase 1d's `TuningEngine` loop calls. `StrategyConfig.build(space, store)` matches engine-architecture §4. `enumerate_grid(space) -> list[dict]` is consistent between Task 2 and `GridStrategy`.

## Hand-off to 1c / 1d

1c (`Scoring & evaluation`) defines the `Scorer` ABC + Count-only `QCScorer` + the `Evaluator` (which calls `strategy.suggest()` indirectly through the engine, and uses the `PruningChannel`). 1d wires the `TuningEngine` ask-and-tell loop over these strategies, adds the params→pipeline builder, and lands the **golden byte-compat lock** (enumerate → build pipelines → equal the Phase-0 golden fixture).
