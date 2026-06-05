# Structured Knob Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the search space's stringly-typed knob keys (`"0.sigma"`) with a typed `KnobTarget` union (`Param`/`Presence`/`Nested`) + a `TuningSpec` cross-validator + a discovery catalog, so programmatic/MCP construction is hard to get wrong — while the `.key` bridge keeps `build_pipeline`, the engine, and the golden lock untouched.

**Architecture:** Each target renders a `.key` property equal to today's canonical string; `build_pipeline` keeps parsing `.key`, so the change is additive. `Knob.target` replaces `Knob.key` but a `key="…"` string is coerced via `parse_key` (string-preserving round-trip), so existing `key=`-based call sites and fixtures keep working **without migration**. A `TuningSpec` `model_validator(after)` cross-checks every target against the pipeline (op-range, `op_class`, field/leaf, did-you-mean). Discovery (`pipeline_targets`/`TunableParam`) and inference always stamp `op_class` (posture C). The design spec is `docs/superpowers/specs/param-sweep-redesign/structured-knob-targets-design.md`.

**Tech Stack:** Python 3.12, pydantic v2 (discriminated unions, `model_validator`), `uv run pytest -n 8` (NEVER `-n auto` — it OOM-kills the HPCC node), `mypy`, `ruff`. Branch: `redesign/param-sweep`.

---

## File structure

| File | Responsibility |
|------|----------------|
| `src/phenotypic/tune/_search_space/_targets.py` (NEW) | `Param`/`Presence`/`Nested`, `KnobTarget` union, `.key` renders, `parse_key`, `with_op_class`. No `_space`/`_infer` import (avoids a cycle). |
| `src/phenotypic/tune/_search_space/_discovery.py` (NEW) | `TunableParam`, `pipeline_targets` (imports `_infer` + `_targets`; NOT imported by `_space`). |
| `src/phenotypic/tune/targets/__init__.py` (NEW) | Public `phenotypic.tune.targets` subpackage re-exporting the above. |
| `src/phenotypic/tune/_search_space/_space.py` (MODIFY) | `Knob.target` field + `key=`/`conditional_on` coercion + `.key` property; `SearchSpace.targets()`. |
| `src/phenotypic/tune/_spec.py` (MODIFY) | `TuningSpec` `model_validator(after)` cross-check helpers. |
| `src/phenotypic/tune/_search_space/_infer.py` (MODIFY) | One-line `with_op_class` post-pass so inferred targets carry `op_class`. |
| Tests | `tests/unit/tune/test_targets.py` (new), `test_discovery.py` (new), `test_targets_subpackage.py` (new); extend `test_search_space.py`, `test_tuning_spec.py`, `test_infer*.py`. |

**Untouched (the `.key` bridge):** `_evaluation/_builder.py`, the strategies, `_engine.py`, `trials.parquet`, the grid byte-compat golden lock.

---

## Task 1: The target union + `.key` renders

**Files:**
- Create: `src/phenotypic/tune/_search_space/_targets.py`
- Test: `tests/unit/tune/test_targets.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_targets.py
from __future__ import annotations

from phenotypic.tune._search_space._targets import Param, Presence, Nested


def test_param_key():
    assert Param(op=0, field="sigma").key == "0.sigma"


def test_presence_key_bare_and_classed():
    assert Presence(op=0).key == "0.__enabled__"
    assert Presence(op=0, op_class="GaussianBlur").key == "0.GaussianBlur.__enabled__"


def test_nested_key():
    t = Nested(op=1, field="detectors", index=0, leaf="ignore_zeros")
    assert t.key == "1.detectors[0].ignore_zeros"


def test_targets_are_frozen_and_discriminated():
    import pytest
    t = Param(op=0, field="sigma")
    with pytest.raises(Exception):
        t.op = 5  # frozen
    assert t.kind == "param"
    assert Presence(op=0).kind == "presence"
    assert Nested(op=0, field="r", index=0, leaf="x").kind == "nested"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tune/test_targets.py -n 8`
Expected: FAIL — `ModuleNotFoundError: ..._targets`.

- [ ] **Step 3: Write the minimal implementation**

```python
# src/phenotypic/tune/_search_space/_targets.py
"""Typed parameter-reference targets for search-space knobs.

A ``Knob`` addresses a pipeline parameter by a structured ``KnobTarget`` — one
of ``Param`` (a flat field on a top-level op), ``Presence`` (an op on/off
toggle), or ``Nested`` (a depth-1 nested-op leaf). Each renders the canonical
``.key`` string the engine already consumes (``build_pipeline`` is untouched),
so targets are a typed authoring/serialization layer over the existing keys.

The public surface lives in the ``phenotypic.tune.targets`` subpackage; these
classes are accessed as ``targets.Param(...)``. This module imports neither
``_space`` nor ``_infer`` (it sits below both, so there is no import cycle).
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

#: The presence-toggle dunder segment (an op on/off flag in the key grammar).
_ENABLED = "__enabled__"


class _TargetBase(BaseModel):
    """Shared config for every target value-model (frozen, no extra fields)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Param(_TargetBase):
    """A flat scalar field on the top-level op at ``op`` — key ``"<op>.<field>"``.

    Args:
        op: The op's position index in the pipeline.
        field: The scalar field name on that op.
        op_class: Optional class-name cross-check — when set, the ``TuningSpec``
            validator asserts the op at ``op`` is this class (posture C: always
            populated by discovery / inference).
    """

    kind: Literal["param"] = "param"
    op: int
    field: str
    op_class: Optional[str] = None

    @property
    def key(self) -> str:
        """The canonical ``"<op>.<field>"`` key string."""
        return f"{self.op}.{self.field}"


class Presence(_TargetBase):
    """An op on/off toggle — key ``"<op>[.<Class>].__enabled__"``.

    Args:
        op: The op's position index.
        op_class: The op's class name; when set, renders the classed key form
            and is cross-checked by the ``TuningSpec`` validator.
    """

    kind: Literal["presence"] = "presence"
    op: int
    op_class: Optional[str] = None

    @property
    def key(self) -> str:
        """The canonical presence key (classed when ``op_class`` is set)."""
        if self.op_class:
            return f"{self.op}.{self.op_class}.{_ENABLED}"
        return f"{self.op}.{_ENABLED}"


class Nested(_TargetBase):
    """A depth-1 nested-op leaf — key ``"<op>.<field>[<index>].<leaf>"``.

    Args:
        op: The parent (top-level) op's position index.
        field: The parent's operation-valued list field.
        index: The slot in that list.
        leaf: The scalar field on the nested op.
        op_class: Optional class-name cross-check of the *parent* op at ``op``.
    """

    kind: Literal["nested"] = "nested"
    op: int
    field: str
    index: int
    leaf: str
    op_class: Optional[str] = None

    @property
    def key(self) -> str:
        """The canonical ``"<op>.<field>[<index>].<leaf>"`` key string."""
        return f"{self.op}.{self.field}[{self.index}].{self.leaf}"


#: The discriminated union a ``Knob.target`` holds.
KnobTarget = Annotated[Union[Param, Presence, Nested], Field(discriminator="kind")]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/tune/test_targets.py -n 8`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_search_space/_targets.py tests/unit/tune/test_targets.py
git commit -m "tune(targets): Param/Presence/Nested union with .key renders"
```

---

## Task 2: `parse_key` (string → target)

**Files:**
- Modify: `src/phenotypic/tune/_search_space/_targets.py`
- Test: `tests/unit/tune/test_targets.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/tune/test_targets.py
import pytest
from phenotypic.tune._search_space._targets import parse_key


@pytest.mark.parametrize("key", [
    "0.sigma", "0.__enabled__", "0.GaussianBlur.__enabled__", "1.detectors[0].ignore_zeros",
])
def test_parse_key_round_trips(key):
    assert parse_key(key).key == key          # string-preserving


def test_parse_key_recovers_op_class_only_for_classed_presence():
    assert parse_key("0.GaussianBlur.__enabled__").op_class == "GaussianBlur"
    assert parse_key("0.sigma").op_class is None
    assert parse_key("0.__enabled__").op_class is None


def test_parse_key_rejects_malformed():
    with pytest.raises(ValueError):
        parse_key("notanint.sigma")
    with pytest.raises(ValueError):
        parse_key("0")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tune/test_targets.py -k parse_key -n 8`
Expected: FAIL — `cannot import name 'parse_key'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# append to src/phenotypic/tune/_search_space/_targets.py
def parse_key(key: str) -> "KnobTarget":
    """Parse a canonical key string into a structured ``KnobTarget``.

    The pipeline-free inverse of ``KnobTarget.key`` — a purely *structural*
    parse; op-range / field-existence checks happen later against the pipeline
    in the ``TuningSpec`` validator. ``op_class`` is recovered only from the
    classed presence form (``"0.GaussianBlur.__enabled__"``); flat and nested
    keys do not encode a class, so their ``op_class`` is ``None``.

    Args:
        key: A canonical key, e.g. ``"0.sigma"`` /
            ``"0.GaussianBlur.__enabled__"`` / ``"0.refiners[1].min_size"``.

    Returns:
        The matching ``Param`` / ``Presence`` / ``Nested``.

    Raises:
        ValueError: When the first segment is not an int position, or the key
            matches none of the three grammars.
    """
    parts = key.split(".")
    try:
        op = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"key {key!r} does not start with an int position") from exc

    # Nested: a "<field>[<index>]" segment selects the nested grammar.
    for i, segment in enumerate(parts[1:], start=1):
        if "[" in segment and segment.endswith("]"):
            field, _, idx = segment[:-1].partition("[")
            leaf = ".".join(parts[i + 1:])
            if not field or not idx.isdigit() or not leaf:
                raise ValueError(f"key {key!r} has a malformed nested segment")
            return Nested(op=op, field=field, index=int(idx), leaf=leaf)

    # Presence: trailing "__enabled__" (classed three-part or bare two-part).
    if parts[-1] == _ENABLED:
        if len(parts) == 3:
            return Presence(op=op, op_class=parts[1])
        if len(parts) == 2:
            return Presence(op=op)
        raise ValueError(f"presence key {key!r} is malformed")

    # Flat: "<op>.<field>".
    if len(parts) == 2:
        return Param(op=op, field=parts[1])
    raise ValueError(f"key {key!r} is not a recognised flat/presence/nested key")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/tune/test_targets.py -n 8`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_search_space/_targets.py tests/unit/tune/test_targets.py
git commit -m "tune(targets): parse_key (string -> target, op_class from classed presence)"
```

---

## Task 3: `with_op_class` helper (posture C fill)

**Files:**
- Modify: `src/phenotypic/tune/_search_space/_targets.py`
- Test: `tests/unit/tune/test_targets.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/tune/test_targets.py
from phenotypic.tune._search_space._targets import with_op_class
from phenotypic import ImagePipeline
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector


def test_with_op_class_fills_from_pipeline():
    ops = list(ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]).get_ops().values())
    assert with_op_class(Param(op=0, field="sigma"), ops).op_class == "GaussianBlur"
    assert with_op_class(Param(op=1, field="ignore_zeros"), ops).op_class == "OtsuDetector"


def test_with_op_class_leaves_out_of_range_untouched():
    assert with_op_class(Param(op=9, field="x"), []).op_class is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tune/test_targets.py -k with_op_class -n 8`
Expected: FAIL — `cannot import name 'with_op_class'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# append to src/phenotypic/tune/_search_space/_targets.py
def with_op_class(target: "KnobTarget", ordered_ops: list) -> "KnobTarget":
    """Return ``target`` with ``op_class`` filled from the op at ``target.op``.

    Posture C: discovery / inference always stamp ``op_class`` so every
    programmatic target is wrong-op cross-checked. An out-of-range ``op`` is
    returned untouched (the ``TuningSpec`` validator reports the range error).

    Args:
        target: The target to enrich.
        ordered_ops: The pipeline's ops in position order.

    Returns:
        A copy with ``op_class = type(ordered_ops[target.op]).__name__``, or the
        unchanged ``target`` when ``op`` is out of range.
    """
    if not 0 <= target.op < len(ordered_ops):
        return target
    return target.model_copy(
        update={"op_class": type(ordered_ops[target.op]).__name__}
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/tune/test_targets.py -n 8`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_search_space/_targets.py tests/unit/tune/test_targets.py
git commit -m "tune(targets): with_op_class fill helper (posture C)"
```

---

## Task 4: `Knob.target` + `key=` coercion + `.key` property

**Files:**
- Modify: `src/phenotypic/tune/_search_space/_space.py`
- Test: `tests/unit/tune/test_search_space.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/tune/test_search_space.py
from phenotypic.tune._search_space._targets import Param, Presence
from phenotypic.tune._search_space._space import Knob
from phenotypic.tune._search_space._domains import FloatRange, Categorical


def test_knob_accepts_target_and_string_key_equivalently():
    a = Knob(target=Param(op=0, field="sigma"), domain=FloatRange(low=0.5, high=8.0))
    b = Knob(key="0.sigma", domain=FloatRange(low=0.5, high=8.0))
    assert a.target == b.target
    assert a.key == b.key == "0.sigma"          # .key property reads through


def test_knob_serializes_target_structurally_and_loads_legacy_string():
    k = Knob(key="0.GaussianBlur.__enabled__", domain=Categorical(choices=(True, False)))
    dumped = k.model_dump()
    assert dumped["target"]["kind"] == "presence"
    assert "key" not in dumped                  # structured, not the string
    # legacy string still loads:
    again = Knob.model_validate({"key": "0.GaussianBlur.__enabled__",
                                 "domain": {"kind": "categorical", "choices": [True, False]}})
    assert again.key == "0.GaussianBlur.__enabled__"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tune/test_search_space.py -k "target_and_string or serializes_target" -n 8`
Expected: FAIL — `Knob` rejects `target=` / `extra="forbid"` on the old `key` field.

- [ ] **Step 3: Write the minimal implementation**

Replace the `Knob` class body in `_space.py`. Change the imports at the top of the file:

```python
# src/phenotypic/tune/_search_space/_space.py  (imports)
from pydantic import BaseModel, ConfigDict, model_validator
from ._domains import Domain
from ._targets import KnobTarget, parse_key
```

Replace the `key: str` field + add the validator and property (keep `source`/`needs_review`/`description`/`is_active`):

```python
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: KnobTarget
    domain: Domain
    conditional_on: Optional[tuple[tuple[KnobTarget, Any], ...]] = None
    source: KnobSource = "manual"
    needs_review: bool = False
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_strings(cls, data: Any) -> Any:
        """Accept a legacy string ``key=`` and string ``conditional_on`` parents.

        ``key="0.sigma"`` is coerced to ``target=parse_key("0.sigma")`` and
        removed from the input (so ``extra="forbid"`` does not trip). A
        ``conditional_on`` entry whose parent is a string is parsed the same way;
        a dict (structured JSON) or a target instance passes through to the
        discriminated-union validator unchanged.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "key" in data and "target" not in data:
            data["target"] = parse_key(data.pop("key"))
        cond = data.get("conditional_on")
        if cond is not None:
            data["conditional_on"] = tuple(
                (parse_key(parent) if isinstance(parent, str) else parent, value)
                for parent, value in cond
            )
        return data

    @property
    def key(self) -> str:
        """The canonical key string of this knob's target (back-compat read)."""
        return self.target.key
```

Update the `Knob` docstring `Args:` (replace the `key:` entry with `target:` — a `KnobTarget`; note `key=` is still accepted and coerced).

Update `is_active` to read through the parent target's `.key` (it currently compares against `pkey`):

```python
    def is_active(self, chosen: Mapping[str, Any]) -> bool:
        # ... docstring unchanged ...
        if self.conditional_on is None:
            return True
        return all(chosen.get(ptarget.key) == pval for ptarget, pval in self.conditional_on)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/tune/test_search_space.py -n 8`
Expected: PASS (existing `key=`-based tests still pass via coercion; new tests pass).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_search_space/_space.py tests/unit/tune/test_search_space.py
git commit -m "tune(space): Knob.target + key= coercion + .key property + conditional_on targets"
```

---

## Task 5: `SearchSpace.targets()` accessor

**Files:**
- Modify: `src/phenotypic/tune/_search_space/_space.py`
- Test: `tests/unit/tune/test_search_space.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/tune/test_search_space.py
from phenotypic.tune._search_space._space import SearchSpace


def test_search_space_targets_and_keys():
    space = SearchSpace(knobs=(
        Knob(target=Param(op=0, field="sigma"), domain=FloatRange(low=0.5, high=8.0)),
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))
    assert space.keys() == ["0.sigma", "1.ignore_zeros"]      # via .key property
    assert [t.kind for t in space.targets()] == ["param", "param"]
    assert space.domain("0.sigma").high == 8.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tune/test_search_space.py -k targets_and_keys -n 8`
Expected: FAIL — `SearchSpace` has no `targets`.

- [ ] **Step 3: Write the minimal implementation**

Add to `SearchSpace` (its `.keys()`/`.domain()` already use `k.key`, now the property — no change needed there):

```python
    def targets(self) -> list[KnobTarget]:
        """Return the knob targets in declaration order."""
        return [k.target for k in self.knobs]
```

Add `KnobTarget` to the `_space.py` import from `._targets` if not already present.

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/tune/test_search_space.py -n 8`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_search_space/_space.py tests/unit/tune/test_search_space.py
git commit -m "tune(space): SearchSpace.targets() accessor"
```

---

## Task 6: `TuningSpec` cross-validator

**Files:**
- Modify: `src/phenotypic/tune/_spec.py`
- Test: `tests/unit/tune/test_tuning_spec.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/tune/test_tuning_spec.py
import pytest
from phenotypic import ImagePipeline
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector
from phenotypic.tune import (TuningSpec, SearchSpace, Knob, FloatRange, Categorical,
                             QCScorer, Evaluator, GridConfig, Budget)
from phenotypic.tune._search_space._targets import Param


def _spec_with(knobs):
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=knobs),
        scorer=QCScorer(check=...),   # replace ... with a real ExpectedVsDetectedCount fixture
        evaluator=Evaluator(), strategy=GridConfig(), budget=Budget(),
    )


def test_valid_targets_pass():
    _spec_with((Knob(target=Param(op=0, field="sigma"), domain=FloatRange(low=0.5, high=8.0)),))


def test_out_of_range_op_rejected():
    with pytest.raises(Exception, match="op 5"):
        _spec_with((Knob(target=Param(op=5, field="sigma"), domain=FloatRange(low=0.5, high=8.0)),))


def test_op_class_mismatch_rejected():
    with pytest.raises(Exception, match="OtsuDetector"):
        _spec_with((Knob(target=Param(op=0, field="sigma", op_class="OtsuDetector"),
                         domain=FloatRange(low=0.5, high=8.0)),))


def test_missing_field_suggests():
    with pytest.raises(Exception, match="did you mean 'sigma'"):
        _spec_with((Knob(target=Param(op=0, field="sigam"), domain=FloatRange(low=0.5, high=8.0)),))
```

(Replace the `check=...` placeholder with the project's existing `QCScorer` fixture builder — see `tests/unit/tune/test_qc_scorer.py` for the `ExpectedVsDetectedCount(metadata=..., groupby=[...])` construction; reuse the same `_layout`/csv helper.)

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tune/test_tuning_spec.py -k "out_of_range or op_class_mismatch or missing_field" -n 8`
Expected: FAIL — the invalid specs construct without error (no validator yet).

- [ ] **Step 3: Write the minimal implementation**

Add to `_spec.py`. Imports at the top:

```python
import difflib
from ._search_space._targets import KnobTarget, Nested, Param
```

Module-level helpers (above `TuningSpec`):

```python
def _check_field(op_obj: Any, field: str, op: int, cls_name: str) -> None:
    """Assert ``field`` exists on ``op_obj``; raise with a did-you-mean otherwise."""
    if field in type(op_obj).model_fields:
        return
    available = sorted(type(op_obj).model_fields)
    suggestion = difflib.get_close_matches(field, available, n=1)
    hint = f" — did you mean {suggestion[0]!r}?" if suggestion else ""
    raise ValueError(
        f"op {op} ({cls_name}) has no field {field!r}{hint}; available: {available}"
    )


def _validate_target(target: KnobTarget, ordered_ops: list) -> None:
    """Validate one knob target against the live pipeline ops (cross-check)."""
    op = target.op
    if not 0 <= op < len(ordered_ops):
        raise ValueError(
            f"knob target {target.key!r} addresses op {op}, but the pipeline has "
            f"{len(ordered_ops)} op(s)"
        )
    actual = ordered_ops[op]
    actual_cls = type(actual).__name__
    if target.op_class is not None and target.op_class != actual_cls:
        raise ValueError(
            f"knob target {target.key!r} names class {target.op_class!r}, but op "
            f"{op} is a {actual_cls!r}"
        )
    if isinstance(target, Param):
        _check_field(actual, target.field, op, actual_cls)
    elif isinstance(target, Nested):
        nested = getattr(actual, target.field, None)
        if not isinstance(nested, list):
            raise ValueError(
                f"nested target {target.key!r}: {actual_cls}.{target.field} is not "
                "a list-of-ops field"
            )
        if not 0 <= target.index < len(nested):
            raise ValueError(
                f"nested target {target.key!r}: index {target.index} out of range "
                f"({len(nested)} slot(s))"
            )
        slot = nested[target.index]
        if slot is None:
            raise ValueError(
                f"nested target {target.key!r}: slot {target.index} is empty (None)"
            )
        _check_field(slot, target.leaf, op, type(slot).__name__)
    # Presence: op-range + op_class already checked above.
```

Add the validator to `TuningSpec` (alongside `_reject_multi_objective_without_optuna`):

```python
    @model_validator(mode="after")
    def _validate_targets_against_pipeline(self) -> "TuningSpec":
        """Cross-check every knob target (and conditional parent) vs the pipeline.

        Catches targeting mistakes — out-of-range op, ``op_class`` mismatch,
        missing field/leaf (with a did-you-mean), unresolvable nesting — at spec
        construction (where an MCP submits), rather than deep in
        ``build_pipeline`` at evaluation time. Complements the apply-time ``⊆``
        backstop, which still catches validator-enforced value bounds.

        Returns:
            ``self`` when every target resolves.

        Raises:
            ValueError: With an actionable message naming the offending target.
        """
        ordered_ops = list(self.pipeline.get_ops().values())
        for knob in self.search_space.knobs:
            _validate_target(knob.target, ordered_ops)
            for parent, _ in (knob.conditional_on or ()):
                _validate_target(parent, ordered_ops)
        return self
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/tune/test_tuning_spec.py -n 8`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_spec.py tests/unit/tune/test_tuning_spec.py
git commit -m "tune(spec): TuningSpec cross-validator (op range/class/field, did-you-mean)"
```

---

## Task 7: Inference stamps `op_class` (posture C)

**Files:**
- Modify: `src/phenotypic/tune/_search_space/_infer.py`
- Test: `tests/unit/tune/test_infer_search_space.py` (or the existing inference test file — confirm the name with `ls tests/unit/tune | grep -i infer`)

- [ ] **Step 1: Write the failing test**

```python
# append to the inference test file
from phenotypic import ImagePipeline
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector
from phenotypic.tune import infer_search_space


def test_inferred_targets_carry_op_class():
    pipe = ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])
    proposal = infer_search_space(pipe)
    sigma = next(k for k in proposal.knobs if k.target.op == 0 and k.target.key == "0.sigma")
    assert sigma.target.op_class == "GaussianBlur"
    assert all(k.target.op_class for k in proposal.knobs)   # every knob stamped
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tune/test_infer_search_space.py -k op_class -n 8`
Expected: FAIL — inferred targets have `op_class is None` (built from `key=` coercion).

- [ ] **Step 3: Write the minimal implementation**

In `_infer.py`, import `with_op_class` and apply it as a post-pass at the end of `infer_search_space` (the existing `Knob(key=...)` helper calls are unchanged — they coerce to `op_class=None` targets, which this pass fills):

```python
# _infer.py imports
from ._targets import with_op_class
```

```python
# _infer.py — replace the final return of infer_search_space (currently ~line 722)
    knobs = [
        k.model_copy(update={"target": with_op_class(k.target, ops)}) for k in knobs
    ]
    return InferredSearchSpace(knobs=tuple(knobs), excluded=tuple(excluded))
```

(`ops = list(pipeline.get_ops().values())` is already in scope at the top of the function.)

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/tune/test_infer_search_space.py -n 8`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_search_space/_infer.py tests/unit/tune/test_infer_search_space.py
git commit -m "tune(infer): stamp op_class on every inferred target (posture C)"
```

---

## Task 8: Discovery catalog — `TunableParam` + `pipeline_targets`

**Files:**
- Create: `src/phenotypic/tune/_search_space/_discovery.py`
- Test: `tests/unit/tune/test_discovery.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_discovery.py
from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector
from phenotypic.tune._search_space._discovery import pipeline_targets, TunableParam


def test_pipeline_targets_catalog():
    pipe = ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])
    cat = pipeline_targets(pipe)
    assert cat and all(isinstance(t, TunableParam) for t in cat)
    sigma = next(t for t in cat if t.target.key == "0.sigma")
    assert sigma.op_class == "GaussianBlur"           # always populated
    assert sigma.value_type == "float"
    assert sigma.default == 2.0                       # current value
    assert sigma.suggested_domain is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tune/test_discovery.py -n 8`
Expected: FAIL — `..._discovery` does not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
# src/phenotypic/tune/_search_space/_discovery.py
"""Discovery catalog: the structured set of tunable parameters in a pipeline.

``pipeline_targets`` re-surfaces ``infer_search_space``'s mining as per-parameter
descriptors (each target ``op_class``-stamped) for the GUI 6c form and the MCP
"what can I tune?" tool — the agent *selects* a target rather than authoring a
key. Imports ``_infer`` (and is NOT imported by ``_space``), so the
``_space -> _targets`` edge stays cycle-free.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from ._domains import Categorical, Domain, FloatRange, IntRange
from ._infer import infer_search_space
from ._targets import KnobTarget, Nested, Param, Presence

#: The value-type of a tunable parameter (named ``value_type`` — not ``kind`` —
#: to avoid colliding with the target union's ``kind`` discriminator).
ValueType = Literal["float", "int", "bool", "categorical"]


class TunableParam(BaseModel):
    """One tunable parameter a pipeline exposes (a discovery descriptor).

    Args:
        target: The structured ``KnobTarget`` to reference it by (``op_class``
            always set).
        op_class: The class of the op at ``target.op``.
        value_type: The parameter's value type.
        default: The field's current value on the pipeline.
        suggested_domain: The inferred domain, or ``None`` if inference excluded
            it.
        description: The field's docstring / ``TuneSpec`` text.
        needs_review: Whether inference flagged the suggested domain for review.
    """

    model_config = ConfigDict(frozen=True)

    target: KnobTarget
    op_class: str
    value_type: ValueType
    default: Any
    suggested_domain: Optional[Domain]
    description: str
    needs_review: bool


def _value_type(domain: Domain) -> ValueType:
    if isinstance(domain, FloatRange):
        return "float"
    if isinstance(domain, IntRange):
        return "int"
    if isinstance(domain, Categorical) and all(isinstance(c, bool) for c in domain.choices):
        return "bool"
    return "categorical"


def _current_value(target: KnobTarget, ordered_ops: list) -> Any:
    op = ordered_ops[target.op]
    if isinstance(target, Param):
        return getattr(op, target.field, None)
    if isinstance(target, Presence):
        return True  # an op present in the base pipeline is enabled
    if isinstance(target, Nested):
        nested = getattr(op, target.field, None)
        if isinstance(nested, list) and 0 <= target.index < len(nested) and nested[target.index] is not None:
            return getattr(nested[target.index], target.leaf, None)
    return None


def pipeline_targets(pipeline: Any) -> list[TunableParam]:
    """The structured catalog of tunable parameters in ``pipeline``.

    Built on ``infer_search_space`` (each knob's target already ``op_class``-
    stamped); each knob becomes a ``TunableParam`` carrying the field's current
    value, inferred domain, value type, description, and review flag.

    Args:
        pipeline: A live ``ImagePipeline``.

    Returns:
        One ``TunableParam`` per inferred knob, in proposal order.
    """
    proposal = infer_search_space(pipeline)
    ordered_ops = list(pipeline.get_ops().values())
    return [
        TunableParam(
            target=knob.target,
            op_class=knob.target.op_class or type(ordered_ops[knob.target.op]).__name__,
            value_type=_value_type(knob.domain),
            default=_current_value(knob.target, ordered_ops),
            suggested_domain=knob.domain,
            description=knob.description,
            needs_review=knob.needs_review,
        )
        for knob in proposal.knobs
    ]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/tune/test_discovery.py -n 8`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_search_space/_discovery.py tests/unit/tune/test_discovery.py
git commit -m "tune(discovery): TunableParam + pipeline_targets catalog"
```

---

## Task 9: The public `phenotypic.tune.targets` subpackage

**Files:**
- Create: `src/phenotypic/tune/targets/__init__.py`
- Test: `tests/unit/tune/test_targets_subpackage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_targets_subpackage.py
from __future__ import annotations


def test_targets_subpackage_surface():
    from phenotypic.tune import targets
    assert set(targets.__all__) == {
        "Param", "Presence", "Nested", "KnobTarget", "parse_key",
        "TunableParam", "pipeline_targets",
    }
    from phenotypic.tune.targets import Param, pipeline_targets  # importable
    assert Param(op=0, field="sigma").key == "0.sigma"


def test_targets_subpackage_is_optuna_free():
    import sys
    sys.modules.pop("optuna", None)
    import importlib
    importlib.import_module("phenotypic.tune.targets")
    assert "optuna" not in sys.modules


def test_top_level_all_unchanged_count():
    import phenotypic.tune as t
    assert len(t.__all__) == 39   # the targets surface lives in the subpackage
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tune/test_targets_subpackage.py -n 8`
Expected: FAIL — `phenotypic.tune.targets` does not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
# src/phenotypic/tune/targets/__init__.py
"""Public typed parameter-reference surface for the tune search space.

``from phenotypic.tune import targets`` → ``targets.Param(op=0, field="sigma")``.
Groups the param-reference + discovery cluster so the top-level
``phenotypic.tune.__all__`` stays lean (these symbols live here, not there).
"""
from __future__ import annotations

from .._search_space._discovery import TunableParam, pipeline_targets
from .._search_space._targets import KnobTarget, Nested, Param, Presence, parse_key

__all__ = [
    "Param", "Presence", "Nested", "KnobTarget", "parse_key",
    "TunableParam", "pipeline_targets",
]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/tune/test_targets_subpackage.py -n 8`
Expected: PASS. (If `test_top_level_all_unchanged_count` fails, confirm no top-level export was added — the subpackage is the only new surface.)

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/targets/__init__.py tests/unit/tune/test_targets_subpackage.py
git commit -m "tune(targets): public phenotypic.tune.targets subpackage"
```

---

## Task 10: Full regression + standing locks + types/lint

**Files:**
- Possibly modify: any `tests/unit/tune/*` fixture that constructed a `TuningSpec` with a **non-resolving** key (the cross-validator now rejects those — they were latent bugs `build_pipeline` would have failed on anyway). Fix by using a real op/field.

- [ ] **Step 1: Run the full tune suite**

Run: `uv run pytest tests/unit/tune tests/unit/tools_/test_io_constants.py -n 8`
Expected: PASS. If a pre-existing test fails on the **new cross-validator** (a synthetic key that doesn't resolve to a real op/field), fix that test's `Knob` to address a real field — do **not** weaken the validator. If it fails on `extra="forbid"` because a test passed both `key=` and `target=`, drop the `key=`.

- [ ] **Step 2: Standing locks (must stay green)**

Run:
```bash
uv run pytest tests/unit/tune/test_grid_golden_manifest.py \
  tests/unit/tune/test_grid_byte_compat_lock.py \
  tests/unit/tune/test_lazy_import_lock.py \
  tests/unit/tune/test_study_store_objectives.py -n 8
```
Expected: PASS. The golden lock proves `.key` round-trips (build_pipeline unchanged); the lazy-import lock proves `tune.targets` pulls in no optuna.

- [ ] **Step 3: Doctests + types + lint**

Run:
```bash
uv run pytest --doctest-modules src/phenotypic/tune/_search_space/_targets.py -n 8
uv run mypy src/phenotypic/tune
uv run ruff check --fix src/phenotypic/tune
```
Expected: no mypy issues; ruff clean.

- [ ] **Step 4: Commit any fixture fixes**

```bash
git add -p   # stage only the fixture/test fixes you made (NEVER git add -A)
git commit -m "tune: migrate fixtures to resolving knob targets; full regression green"
```

- [ ] **Step 5: Final gate confirmation**

Run: `uv run pytest tests/unit/tune tests/unit/tools_/test_io_constants.py -n 8`
Expected: all green (the new `test_targets.py` / `test_discovery.py` / `test_targets_subpackage.py` included).

---

## Self-Review

**Spec coverage** — every spec section maps to a task: §3 union → T1; `parse_key` → T2; §5 `op_class` fill (posture C) → T3/T7; §4 `Knob` dual constructor → T4; `SearchSpace.targets` → T5; §5 cross-validator → T6; §6 discovery → T8; §8 namespacing → T9; §2/§9 untouched + locks → T10. §7 serialization (structured emit + legacy load) → T4 test. Depth>1 / op-identity are explicitly deferred (not tasks).

**Placeholder scan** — the only `...` is the `QCScorer(check=...)` in the T6 test, flagged with explicit instructions to reuse the existing `test_qc_scorer.py` fixture builder; everything else is complete code.

**Type consistency** — `KnobTarget`, `parse_key`, `with_op_class`, `Param`/`Presence`/`Nested` (fields `op`/`field`/`index`/`leaf`/`op_class`/`kind`), `Knob.target`/`.key`, `SearchSpace.targets()`, `TunableParam`(`target`/`op_class`/`value_type`/`default`/`suggested_domain`/`description`/`needs_review`), `pipeline_targets` — names/signatures are consistent across tasks. `Knob.is_active` reads `ptarget.key`; the strategies/`build_pipeline` consume `knob.key` (the property) unchanged.
