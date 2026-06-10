# Phase 0 — orientation machinery (`_orient.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Ship the orientation primitives (`Sense`, `to_cost`, `clamp01`) as a new, self-contained module with full unit tests — and **nothing else**. This phase changes no behavior (it has no callers yet), so it can land on its own PR ahead of the cutover.

**Why dark:** Activating orientation means converting `Scorer` to a template method, which forces every scorer to migrate in the same change — that is Phase 1. Phase 0 only adds the building blocks Phase 1 imports.

**Files:**
- Create: `src/phenotypic/tune/_scoring/_orient.py`
- Create: `tests/unit/tune/test_orient.py`

**Read first:** the "Shared contract" section of [`README.md`](README.md) (the exact `Sense` / `to_cost` / `clamp01` definitions this phase ships). Confirm there is an existing private `_clamp01` in `src/phenotypic/tune/_scoring/_reference_free_scorer.py:53` — we are introducing a *public, shared* `clamp01` in `_orient.py`; Phase 1 will repoint the evaluator to it (do not touch the reference-free copy in this phase).

---

### Task 1: `to_cost` + `Sense` — bounded cases

**Files:**
- Create: `src/phenotypic/tune/_scoring/_orient.py`
- Test: `tests/unit/tune/test_orient.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/tune/test_orient.py`:
```python
"""Unit tests for the cost-orientation primitives."""
import math

import pytest

from phenotypic.tune._scoring._orient import Sense, clamp01, to_cost


class TestSense:
    def test_two_members_with_string_values(self):
        assert {s.value for s in Sense} == {"lower_better", "higher_better"}

    def test_str_enum_is_value_comparable(self):
        # str, Enum → member is its string value, robust for ClassVar use.
        assert Sense.LOWER_BETTER == "lower_better"


class TestToCostBounded:
    def test_lower_better_bounded_is_identity(self):
        # A value already in [0,1] that is a cost (lower better) passes through.
        assert to_cost(0.3, sense=Sense.LOWER_BETTER) == pytest.approx(0.3)

    def test_higher_better_bounded_is_complement(self):
        # A [0,1] goodness (Dice/IoU/ICC) is complemented to cost.
        assert to_cost(0.3, sense=Sense.HIGHER_BETTER) == pytest.approx(0.7)

    def test_perfect_goodness_maps_to_zero_cost(self):
        assert to_cost(1.0, sense=Sense.HIGHER_BETTER) == pytest.approx(0.0)

    def test_worst_goodness_maps_to_unit_cost(self):
        assert to_cost(0.0, sense=Sense.HIGHER_BETTER) == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra tune pytest tests/unit/tune/test_orient.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.tune._scoring._orient'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/phenotypic/tune/_scoring/_orient.py`:
```python
"""Cost orientation — map a scorer's natural per-term value to bounded cost.

A *cost* is in ``[0, 1]`` where ``0`` is perfect and ``1`` is worst (lower is
better; the optimizer minimizes). Scorers declare a :class:`Sense` and emit
their natural per-term values; :func:`to_cost` orients them. This is the one
place orientation happens, replacing the per-scorer hand-rolled flips.
"""
from __future__ import annotations

import math
from enum import Enum


class Sense(str, Enum):
    """Direction of a scorer's natural per-term values.

    ``LOWER_BETTER`` — a larger value is *worse* (a loss/divergence); maps to a
    QC check's ``_HIGHER_IS_BAD=True``. ``HIGHER_BETTER`` — a larger value is
    *better* (Dice, IoU, ICC, solidity).
    """

    LOWER_BETTER = "lower_better"
    HIGHER_BETTER = "higher_better"


def clamp01(value: float) -> float:
    """Clamp ``value`` into ``[0, 1]``.

    Used on the robust-aggregated cost: ``median + λ·IQR`` can reach ``~1+λ``
    (B1), so the term/child cost must be clamped to keep the ``0 ≤ cost ≤ 1``
    invariant the composite relies on.

    Args:
        value: Any float.

    Returns:
        ``0.0`` if ``value < 0``, ``1.0`` if ``value > 1``, else ``float(value)``.
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def to_cost(value: float, *, sense: Sense, anchor: float | None = None) -> float:
    """Map a scorer's natural per-term value to cost in ``[0, 1]``.

    Args:
        value: The scorer's natural per-term value.
        sense: Whether larger values are better or worse.
        anchor: ``None`` when ``value`` is already bounded in ``[0, 1]``;
            a positive float (the half-cost scale, e.g. a check's
            ``fail_threshold``) when ``value`` is an unbounded magnitude.

    Returns:
        The cost in ``[0, 1]`` (``0`` perfect, ``1`` worst).

    Examples:
        >>> to_cost(0.3, sense=Sense.LOWER_BETTER)
        0.3
        >>> to_cost(0.3, sense=Sense.HIGHER_BETTER)
        0.7
        >>> round(to_cost(0.1, sense=Sense.LOWER_BETTER, anchor=0.1), 3)
        0.5
        >>> to_cost(float("inf"), sense=Sense.LOWER_BETTER, anchor=0.1)
        1.0
    """
    if anchor is None:
        return value if sense is Sense.LOWER_BETTER else 1.0 - value
    if not math.isfinite(value):
        return 1.0
    goodness = math.exp(-math.log(2.0) * value / anchor)
    return (1.0 - goodness) if sense is Sense.LOWER_BETTER else goodness
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra tune pytest tests/unit/tune/test_orient.py -v`
Expected: PASS (4 tests in `TestToCostBounded`, 2 in `TestSense`).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_scoring/_orient.py tests/unit/tune/test_orient.py
git commit -m "feat(tune): add Sense + to_cost orientation primitives (bounded cases)"
```

---

### Task 2: `to_cost` — unbounded (anchored) cases + `clamp01`

**Files:**
- Modify: `tests/unit/tune/test_orient.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/tune/test_orient.py`:
```python
class TestToCostAnchored:
    def test_zero_divergence_is_zero_cost(self):
        assert to_cost(0.0, sense=Sense.LOWER_BETTER, anchor=0.1) == pytest.approx(0.0)

    def test_at_anchor_is_half_cost(self):
        assert to_cost(0.1, sense=Sense.LOWER_BETTER, anchor=0.1) == pytest.approx(0.5)

    def test_large_divergence_approaches_unit_cost(self):
        assert to_cost(10.0, sense=Sense.LOWER_BETTER, anchor=0.1) > 0.99

    def test_inf_divergence_is_worst(self):
        assert to_cost(float("inf"), sense=Sense.LOWER_BETTER, anchor=0.1) == 1.0

    def test_higher_better_unbounded_zero_is_worst(self):
        # higher-better unbounded: value 0 = worst → cost 1.0.
        assert to_cost(0.0, sense=Sense.HIGHER_BETTER, anchor=0.1) == pytest.approx(1.0)

    def test_higher_better_unbounded_at_anchor_is_half(self):
        assert to_cost(0.1, sense=Sense.HIGHER_BETTER, anchor=0.1) == pytest.approx(0.5)


class TestClamp01:
    @pytest.mark.parametrize(
        "value,expected", [(-0.5, 0.0), (0.0, 0.0), (0.3, 0.3), (1.0, 1.0), (1.5, 1.0)]
    )
    def test_clamps_to_unit_interval(self, value, expected):
        assert clamp01(value) == pytest.approx(expected)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra tune pytest tests/unit/tune/test_orient.py -v`
Expected: PASS for the anchored cases (already implemented in Task 1) — **but** confirm `TestClamp01` passes too. If `clamp01` was somehow omitted in Task 1, it FAILs with `ImportError`. (Task 1's implementation already includes `clamp01`, so this should pass; the purpose of this task is to lock the anchored + clamp behavior with explicit tests.)

- [ ] **Step 3: (no new implementation needed)**

`to_cost` anchored branch and `clamp01` were implemented in Task 1. If any test fails, fix `_orient.py` to match the README contract exactly.

- [ ] **Step 4: Run the doctest too**

Run: `uv run --extra tune pytest --doctest-modules src/phenotypic/tune/_scoring/_orient.py -v`
Expected: PASS (the `to_cost` docstring examples).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/tune/test_orient.py
git commit -m "test(tune): lock to_cost anchored cases and clamp01"
```

---

### Task 3: type + lint gate

- [ ] **Step 1: Type-check**

Run: `uv run mypy src/phenotypic/tune/_scoring/_orient.py`
Expected: `Success: no issues found`.

- [ ] **Step 2: Lint**

Run: `uv run ruff check --fix src/phenotypic/tune/_scoring/_orient.py tests/unit/tune/test_orient.py`
Expected: no remaining errors.

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -A && git commit -m "style(tune): lint orientation module" || echo "nothing to commit"
```

---

## Phase 0 done-criteria
- `_orient.py` exists with `Sense`, `clamp01`, `to_cost`; all `test_orient.py` tests + the module doctest pass.
- `mypy` + `ruff` clean.
- **No other file changed** — `git diff --name-only origin/main...HEAD` for this phase lists only `_orient.py` and `test_orient.py`. Behavior is unchanged (no callers yet).
