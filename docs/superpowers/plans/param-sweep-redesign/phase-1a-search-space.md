# Tune Engine — Phase 1a: SearchSpace Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the hand-authorable search-space value types — the pydantic discriminated-union **domains** (`Categorical`/`IntRange`/`FloatRange`/`Fixed`) and the **`Knob`** / **`SearchSpace`** containers — that every later Phase-1 component (strategies, evaluator, engine, spec) consumes.

**Architecture:** All search-space data are **frozen pydantic value-models** (engine-architecture.md §5) so they round-trip through one serialization mechanism into `TuningSpec` (Phase 1d). Domains are a `Field(discriminator="kind")` union (a *closed* set). `Knob` carries provenance fields (`source`/`needs_review`/`description`) defaulted for hand-authoring; Phase 3's `infer_search_space` populates them. `InferredSearchSpace`, `TuneSpec`, and inference itself are **Phase 3** — Phase 1 hand-authors a `SearchSpace`.

**Tech Stack:** pydantic v2 (`Field(discriminator=...)`, `ConfigDict(frozen=True)`), `typing`. No new deps.

**Spec:** `engine-architecture.md` §5; `search-space-inference.md` §7 (the proposal types; note these are pydantic per engine-arch). **Depends on:** Phase 0 (the `tune/` stub package + the registry).

**Conventions:** `uv run pytest`, `uv run mypy src/phenotypic/tune`, `uv run ruff check --fix`; Google docstrings; tests under `tests/unit/tune/`.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/phenotypic/tune/_search_space/__init__.py` | internal re-exports | Create |
| `src/phenotypic/tune/_search_space/_domains.py` | `Categorical`/`IntRange`/`FloatRange`/`Fixed` + the `Domain` discriminated union | Create |
| `src/phenotypic/tune/_search_space/_space.py` | `Knob`, `SearchSpace` | Create |
| `src/phenotypic/tune/__init__.py` | public exports (extend the Phase-0 stub) | Modify |
| `tests/unit/tune/test_domains.py` | domain construction/validation/round-trip | Create |
| `tests/unit/tune/test_search_space.py` | `Knob`/`SearchSpace` + conditional + round-trip | Create |

> Naming: tune's `Fixed` (a domain) is the only `Fixed` after the Phase-0/Phase-1 hard cutover deletes `phenotypic.sweep.Fixed` (master §9). No clash.

---

### Task 1: The domain types (discriminated union)

**Files:**
- Create: `src/phenotypic/tune/_search_space/_domains.py`
- Create: `src/phenotypic/tune/_search_space/__init__.py`
- Test: `tests/unit/tune/test_domains.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_domains.py
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from phenotypic.tune._search_space._domains import (
    Categorical,
    Domain,
    Fixed,
    FloatRange,
    IntRange,
)


def test_construction_and_defaults():
    assert Categorical(choices=(True, False)).choices == (True, False)
    r = IntRange(low=2, high=20)
    assert (r.low, r.high, r.step, r.log) == (2, 20, 1, False)
    assert FloatRange(low=1e-3, high=1.0, log=True).log is True
    assert Fixed(value=4.0).value == 4.0


def test_list_choices_coerced_to_tuple():
    assert Categorical(choices=["disk", "square"]).choices == ("disk", "square")


def test_frozen():
    with pytest.raises(ValidationError):
        IntRange(low=2, high=20).low = 3  # type: ignore[misc]


def test_range_validation():
    with pytest.raises(ValidationError):
        IntRange(low=20, high=2)
    with pytest.raises(ValidationError):
        FloatRange(low=2.0, high=1.0)


def test_discriminated_union_roundtrip():
    adapter: TypeAdapter[Domain] = TypeAdapter(Domain)
    for dom in [
        Categorical(choices=(1, 2, 3)),
        IntRange(low=2, high=20, step=2),
        FloatRange(low=0.5, high=8.0, log=True),
        Fixed(value="reflect"),
    ]:
        blob = adapter.dump_json(dom)
        back = adapter.validate_json(blob)
        assert back == dom
        assert json.loads(blob)["kind"] == dom.kind
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_domains.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.tune._search_space'`.

- [ ] **Step 3: Implement the domains**

```python
# src/phenotypic/tune/_search_space/_domains.py
"""Search-space domain types — a frozen pydantic discriminated union.

A tunable parameter's domain is one of ``Categorical`` / ``IntRange`` /
``FloatRange`` / ``Fixed``; each carries a ``kind`` literal so a ``Knob``'s
``domain`` field serializes and deserializes to the concrete type.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _DomainBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Categorical(_DomainBase):
    """A finite set of choices (bools, enum/literal members, …)."""

    kind: Literal["categorical"] = "categorical"
    choices: tuple[Any, ...]

    @model_validator(mode="after")
    def _non_empty(self) -> "Categorical":
        if len(self.choices) == 0:
            raise ValueError("Categorical requires at least one choice")
        return self


class IntRange(_DomainBase):
    """An integer range ``[low, high]`` with an optional step / log scale."""

    kind: Literal["int_range"] = "int_range"
    low: int
    high: int
    step: int = 1
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "IntRange":
        if self.high < self.low:
            raise ValueError(f"IntRange high ({self.high}) < low ({self.low})")
        return self


class FloatRange(_DomainBase):
    """A float range ``[low, high]`` with an optional log scale."""

    kind: Literal["float_range"] = "float_range"
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "FloatRange":
        if self.high < self.low:
            raise ValueError(f"FloatRange high ({self.high}) < low ({self.low})")
        return self


class Fixed(_DomainBase):
    """A pinned (non-tunable / frozen) value."""

    kind: Literal["fixed"] = "fixed"
    value: Any


#: The discriminated union a ``Knob``'s ``domain`` field uses.
Domain = Annotated[
    Union[Categorical, IntRange, FloatRange, Fixed],
    Field(discriminator="kind"),
]
```

```python
# src/phenotypic/tune/_search_space/__init__.py
"""Internal search-space value types (private)."""
from __future__ import annotations

from ._domains import Categorical, Domain, Fixed, FloatRange, IntRange

__all__ = ["Categorical", "IntRange", "FloatRange", "Fixed", "Domain"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_domains.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_search_space tests/unit/tune/test_domains.py
git add src/phenotypic/tune/_search_space tests/unit/tune/test_domains.py
git commit -m "feat(tune): search-space domain types (discriminated union)"
```

---

### Task 2: `Knob` and `SearchSpace`

**Files:**
- Create: `src/phenotypic/tune/_search_space/_space.py`
- Modify: `src/phenotypic/tune/_search_space/__init__.py`
- Test: `tests/unit/tune/test_search_space.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_search_space.py
from __future__ import annotations

import pytest

from phenotypic.tune._search_space import (
    Categorical,
    FloatRange,
    IntRange,
    Knob,
    SearchSpace,
)


def _space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.sigma", domain=FloatRange(low=0.5, high=8.0)),
        Knob(key="1.size", domain=IntRange(low=4, high=400)),
        Knob(
            key="0.GaussianBlur.mode",
            domain=Categorical(choices=("reflect", "nearest")),
            conditional_on=(("0.GaussianBlur.__enabled__", True),),
        ),
    ))


def test_knob_defaults():
    k = Knob(key="x", domain=IntRange(low=1, high=9))
    assert k.source == "manual"
    assert k.needs_review is False
    assert k.description == ""
    assert k.conditional_on is None


def test_searchspace_keys_and_domain_lookup():
    s = _space()
    assert s.keys() == ["0.sigma", "1.size", "0.GaussianBlur.mode"]
    assert s.domain("1.size") == IntRange(low=4, high=400)
    with pytest.raises(KeyError):
        s.domain("nope")


def test_searchspace_iterates_knobs():
    assert [k.key for k in _space()] == [
        "0.sigma", "1.size", "0.GaussianBlur.mode",
    ]


def test_searchspace_roundtrip_with_conditional_and_mixed_domains():
    s = _space()
    blob = s.model_dump_json()
    back = SearchSpace.model_validate_json(blob)
    assert back == s
    # conditional_on survives (list↔tuple coercion) and the domain discriminator routes
    cond = next(k for k in back if k.key.endswith(".mode"))
    assert cond.conditional_on == (("0.GaussianBlur.__enabled__", True),)
    assert isinstance(cond.domain, Categorical)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_search_space.py -v`
Expected: FAIL — `ImportError: cannot import name 'Knob'`.

- [ ] **Step 3: Implement `Knob` + `SearchSpace`**

```python
# src/phenotypic/tune/_search_space/_space.py
"""The optimizer-facing search space: knobs + their domains."""
from __future__ import annotations

from typing import Any, Iterator, Optional

from pydantic import BaseModel, ConfigDict

from ._domains import Domain


class Knob(BaseModel):
    """One tunable parameter: a key, a domain, and optional provenance.

    ``conditional_on`` makes the knob active only when a parent presence knob
    holds the given value, e.g.
    ``(("0.GaussianBlur.__enabled__", True),)`` — define-by-run conditional
    nesting. The provenance fields (``source`` / ``needs_review`` /
    ``description``) default for hand-authoring and are populated by
    ``infer_search_space`` (Phase 3).
    """

    model_config = ConfigDict(frozen=True)

    key: str
    domain: Domain
    conditional_on: Optional[tuple[tuple[str, Any], ...]] = None
    source: str = "manual"
    needs_review: bool = False
    description: str = ""


class SearchSpace(BaseModel):
    """The clean, optimizer-facing collection of tunable knobs."""

    model_config = ConfigDict(frozen=True)

    knobs: tuple[Knob, ...]

    def keys(self) -> list[str]:
        return [k.key for k in self.knobs]

    def domain(self, key: str) -> Domain:
        for k in self.knobs:
            if k.key == key:
                return k.domain
        raise KeyError(key)

    def __iter__(self) -> Iterator[Knob]:  # type: ignore[override]
        return iter(self.knobs)
```

Update the subpackage `__init__.py` to export `Knob`, `SearchSpace`:

```python
# src/phenotypic/tune/_search_space/__init__.py  (replace file)
"""Internal search-space value types (private)."""
from __future__ import annotations

from ._domains import Categorical, Domain, Fixed, FloatRange, IntRange
from ._space import Knob, SearchSpace

__all__ = [
    "Categorical", "IntRange", "FloatRange", "Fixed", "Domain",
    "Knob", "SearchSpace",
]
```

> Note: overriding `__iter__` on a pydantic `BaseModel` is supported but shadows pydantic's field iteration; that's intentional here (a `SearchSpace` iterates its knobs). mypy needs the `# type: ignore[override]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_search_space.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_search_space tests/unit/tune/test_search_space.py
git add src/phenotypic/tune/_search_space tests/unit/tune/test_search_space.py
git commit -m "feat(tune): Knob + SearchSpace value types"
```

---

### Task 3: Public exports + doctest

**Files:**
- Modify: `src/phenotypic/tune/__init__.py`
- Test: the package doctest

- [ ] **Step 1: Export the public surface + a doctest**

```python
# src/phenotypic/tune/__init__.py  (replace the Phase-0 stub)
"""Parameter-tuning engine — public API (in progress).

Hand-author a search space:

    >>> from phenotypic.tune import SearchSpace, Knob, FloatRange, Categorical
    >>> space = SearchSpace(knobs=(
    ...     Knob(key="0.sigma", domain=FloatRange(low=0.5, high=8.0)),
    ...     Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ... ))
    >>> space.keys()
    ['0.sigma', '1.ignore_zeros']
    >>> space.domain("0.sigma").high
    8.0
"""
from __future__ import annotations

from ._search_space import (
    Categorical,
    Domain,
    Fixed,
    FloatRange,
    IntRange,
    Knob,
    SearchSpace,
)

__all__ = [
    "Categorical",
    "IntRange",
    "FloatRange",
    "Fixed",
    "Domain",
    "Knob",
    "SearchSpace",
]
```

- [ ] **Step 2: Run the doctest + the suite + types/lint**

Run: `uv run pytest --doctest-modules src/phenotypic/tune/__init__.py tests/unit/tune -q`
Expected: PASS.

Run: `uv run mypy src/phenotypic/tune && uv run ruff check src/phenotypic/tune tests/unit/tune`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add src/phenotypic/tune/__init__.py
git commit -m "feat(tune): export the Phase-1a search-space public API"
```

---

## Self-Review

**Spec coverage (engine-architecture §5):** domains as a pydantic discriminated union → Task 1; `Knob`/`SearchSpace` as frozen pydantic value-models with `conditional_on` → Task 2; public exports → Task 3. `InferredSearchSpace` / `Excluded` / `TuneSpec` / `infer_search_space` are **Phase 3** (inference) and correctly absent. `TuningSpec` is **Phase 1d** (needs `Scorer`/`StrategyConfig`).

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `Domain = Annotated[Union[...], Field(discriminator="kind")]`, `Knob(key, domain, conditional_on, source, needs_review, description)`, and `SearchSpace(knobs).keys()/.domain(key)` are consistent across Tasks 1–3 and match what Phase 1b (`GridStrategy` enumerates `SearchSpace`) and Phase 1d (`TuningSpec.search_space`) will import.

## Hand-off to 1b

Phase 1b (`Strategies`) imports `SearchSpace`/`Knob`/`Domain` to enumerate (`GridStrategy`) and sample (`RandomStrategy`), and adds the `SearchStrategy` Protocol + `StrategyConfig`. The `conditional_on` tags drive the define-by-run skipping there.
