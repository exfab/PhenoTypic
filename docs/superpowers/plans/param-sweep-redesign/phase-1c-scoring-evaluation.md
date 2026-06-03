# Tune Engine — Phase 1c: Scoring + Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the objective-scoring layer (`Scorer` ABC + the no-ground-truth Count-only `QCScorer`) and the candidate `Evaluator` — the params→pipeline builder + the uniform 3-step robust-evaluation loop that turns a sampled parameter combo into a single comparable score.

**Architecture:** A `Scorer` is a **pydantic ABC** (engine-architecture.md §5) with `score_image` (abstract, per-image term scores, *higher = better*), a default `finalize` (combine robust-aggregated terms → scalar), and `availability`. The MVP concrete scorer is `QCScorer`, which wraps the existing `analysis.ExpectedVsDetectedCount` (path-configured so it round-trips) and maps its `|detected−expected|/expected` metric through a threshold-anchored smooth normalizer into a `{"Count": t}` term (`t∈[0,1]`). The `Evaluator` runs the **uniform 3-step loop** (qc/robust-eval docs): `score_image` per calibration image → robust-aggregate each term as `median − λ·IQR` (λ=0.5) → `finalize`. Its `build_pipeline(base, params)` **clones the base pipeline embedded in the `TuningSpec`** (the decision locked in engine-arch §6), overlays the combo onto the addressed ops by **fresh reconstruction** (full validation, byte-compatible with legacy `operation_class(**merged)`), and **drops** ops whose `__enabled__=False`.

**Tech Stack:** pydantic v2 (`BaseModel`+`ABC`, `ConfigDict(frozen=True)`), numpy (median/IQR), pandas (the measurement frame), `math`. No new deps. Reuses Phase-1a (`SearchSpace`/`Knob`) conceptually and the existing `ImagePipeline` / `ExpectedVsDetectedCount` APIs.

**Spec:** `engine-architecture.md` §5 (pydantic ABCs); `qc-objective-mapping.md` (the `QCScorer`, threshold-anchored normalizer, Count term); `robust-evaluation.md` §"3-step loop" + §"`median − λ·IQR`". **Depends on:** Phase 0 (registry `+= tune`, `polymorphic_field`), Phase 1a (`tune/` package + value types). **Hands off to:** Phase 1d (`TuningEngine` + `TuningSpec` embedding `pipeline`/`scorer`/`evaluator`; the golden byte-compat lock).

**Conventions:** `uv run pytest`, `uv run mypy src/phenotypic/tune`, `uv run ruff check --fix`; Google docstrings; tests under `tests/unit/tune/`. Doctests/tests use `load_synth_yeast_plate()`.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/phenotypic/tune/_scoring/__init__.py` | internal re-exports | Create |
| `src/phenotypic/tune/_scoring/_scorer.py` | `Scorer` pydantic ABC (`score_image`/`finalize`/`availability`) | Create |
| `src/phenotypic/tune/_scoring/_qc_scorer.py` | `_threshold_anchored` normalizer + Count-only `QCScorer` | Create |
| `src/phenotypic/tune/_evaluation/__init__.py` | internal re-exports | Create |
| `src/phenotypic/tune/_evaluation/_builder.py` | `build_pipeline(base, params)` + key-path parsing + op rebuild | Create |
| `src/phenotypic/tune/_evaluation/_evaluator.py` | `_robust_aggregate`, `EvaluationResult`, `Evaluator` | Create |
| `src/phenotypic/tune/__init__.py` | export the Phase-1c public surface (extend) | Modify |
| `tests/unit/tune/test_scorer.py` | `Scorer` ABC contract | Create |
| `tests/unit/tune/test_qc_scorer.py` | normalizer + `QCScorer` (hand-built frames + path round-trip) | Create |
| `tests/unit/tune/test_builder.py` | overlay / drop / class-validation / clone-isolation | Create |
| `tests/unit/tune/test_evaluator.py` | `_robust_aggregate` + the 3-step loop + failure policy | Create |
| `tests/unit/tune/test_evaluation_integration.py` | real `QCScorer`+`measure`+synth plate end-to-end | Create |

---

## Preamble — the worked example the tests build against

Phase 1 hand-authors a `SearchSpace` (the **grammar** + a full worked example live in
**master §5 "Hand-authoring a `SearchSpace`"** — the canonical reference; don't re-explain
it here). The `Evaluator` consumes one **combo** sampled from such a space — a flat
`dict[str, value]` keyed by the same root-relative paths — against a **base pipeline**:

```python
# base pipeline embedded in the TuningSpec (engine-arch §6); positions are 0-indexed
from phenotypic import ImagePipeline
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector

base = ImagePipeline(ops=[
    GaussianBlur(sigma=2.0),          # position 0  (optional → can carry __enabled__)
    OtsuDetector(ignore_zeros=False), # position 1
])

# one combo a strategy.suggest() produces from the space:
combo = {
    "0.GaussianBlur.__enabled__": True,   # presence: keep the blur
    "0.sigma": 3.2,                        # overlay onto position 0
    "1.ignore_zeros": True,                # overlay onto position 1
}
```

`build_pipeline(base, combo)` → clone `base` → rebuild position 0 with `sigma=3.2`,
position 1 with `ignore_zeros=True`, keep both (blur enabled) → a runnable
`ImagePipeline`. Setting `"0.GaussianBlur.__enabled__": False` instead drops the blur.

**Key-path grammar the builder honors (MVP):**

| Key shape | Handling |
|-----------|----------|
| `"<pos>.<field>"` (2 segments) | overlay `field` on the op at position `<pos>` |
| `"<pos>.<ClassName>.__enabled__"` (3 segments, `__enabled__` tail) | presence toggle; **validates** position `<pos>` is a `<ClassName>` |
| anything else (e.g. nested `"1.detectors[0].x"`) | `NotImplementedError` — nested overlay is **Phase 3** (inference emits it) |

---

### Task 1: The `Scorer` pydantic ABC

**Files:**
- Create: `src/phenotypic/tune/_scoring/_scorer.py`
- Create: `src/phenotypic/tune/_scoring/__init__.py`
- Test: `tests/unit/tune/test_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_scorer.py
from __future__ import annotations

import pandas as pd
import pytest

from phenotypic.tune._scoring._scorer import Scorer


class _FixedScorer(Scorer):
    """Concrete test double: returns preset terms, ignores its inputs."""

    terms: dict[str, float]

    def score_image(self, image, measurements) -> dict[str, float]:
        return dict(self.terms)


def test_scorer_is_abstract():
    # score_image is abstract — the bare base cannot be instantiated.
    with pytest.raises(TypeError):
        Scorer()  # type: ignore[abstract]


def test_concrete_scorer_score_image():
    s = _FixedScorer(terms={"Count": 0.8})
    assert s.score_image(None, pd.DataFrame()) == {"Count": 0.8}


def test_default_finalize_is_mean_of_terms():
    s = _FixedScorer(terms={})
    assert s.finalize({"Count": 0.8}) == pytest.approx(0.8)          # single term passes through
    assert s.finalize({"a": 0.2, "b": 0.8}) == pytest.approx(0.5)    # mean
    assert s.finalize({}) == 0.0                                      # empty → floor


def test_default_availability_true():
    assert _FixedScorer(terms={}).availability() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.tune._scoring'`.

- [ ] **Step 3: Implement the ABC**

```python
# src/phenotypic/tune/_scoring/_scorer.py
"""The pluggable scoring objective — a pydantic ABC.

A ``Scorer`` turns one image's measurement frame into a dict of **named term
scores** (``score_image``), where *higher is better* and each term is a clean,
comparable signal (typically normalized to ``[0, 1]``). The ``Evaluator``
collects the per-image terms across a calibration set, robust-aggregates each
term, then asks the scorer to ``finalize`` the aggregated terms into the single
scalar objective the optimizer maximizes. ``availability`` lets a scorer report
that it cannot run (e.g. missing metadata) so the engine can degrade gracefully.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import pandas as pd
from pydantic import BaseModel


class Scorer(BaseModel, ABC):
    """Base class for tuning objectives (no-GT, supervised, reference-free, …)."""

    @abstractmethod
    def score_image(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Score one image's measurements as named terms (higher = better).

        Args:
            image: The (already-processed) image — duck-typed; reference-free
                scorers read its mask/objmap, the ``QCScorer`` ignores it.
            measurements: The measurement frame the candidate pipeline produced
                for ``image`` (the output of ``ImagePipeline.measure``).

        Returns:
            A mapping of term name → score for this image. Keys must be stable
            across images so the ``Evaluator`` can aggregate per term.
        """
        raise NotImplementedError

    def availability(self) -> bool:
        """Whether this scorer can run as configured (default: yes).

        Returns:
            ``True`` if scoring is possible; subclasses override to report a
            missing prerequisite (e.g. a layout frame the ``QCScorer`` needs).
        """
        return True

    def finalize(self, terms: Mapping[str, float]) -> float:
        """Combine robust-aggregated per-term scores into one scalar objective.

        The default is the arithmetic mean of the term values (a single term
        passes through unchanged); composite scorers override to weight terms.

        Args:
            terms: Term name → robust-aggregated score (already reduced across
                the calibration set by the ``Evaluator``).

        Returns:
            The scalar objective (higher = better). ``0.0`` for no terms.
        """
        values = list(terms.values())
        if not values:
            return 0.0
        return float(sum(values) / len(values))
```

```python
# src/phenotypic/tune/_scoring/__init__.py
"""Internal scoring value types (private)."""
from __future__ import annotations

from ._scorer import Scorer

__all__ = ["Scorer"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_scorer.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_scoring tests/unit/tune/test_scorer.py
git add src/phenotypic/tune/_scoring tests/unit/tune/test_scorer.py
git commit -m "feat(tune): Scorer pydantic ABC (score_image/finalize/availability)"
```

---

### Task 2: The threshold-anchored normalizer + Count-only `QCScorer`

**Files:**
- Create: `src/phenotypic/tune/_scoring/_qc_scorer.py`
- Modify: `src/phenotypic/tune/_scoring/__init__.py`
- Test: `tests/unit/tune/test_qc_scorer.py`

The `QCScorer` wraps `analysis.ExpectedVsDetectedCount`. That check exposes
`name="Count"`, `fail_threshold` (default `0.10`), `groupby`, `metric_col()` (the
`QC_Count_Metric` column), and `metadata`/`metadata_source` (configure from a **path**
so it round-trips — an in-memory-frame check cannot be rebuilt from JSON; see its
docstring). Its metric is `|detected − expected| / expected`, `inf` when `expected==0`.

The normalizer anchors on `fail_threshold`: `t = exp(−ln2 · m / f)` → perfect `m=0` maps
to `1.0`, the fail boundary `m=f` maps to `0.5`, and `m=∞` (unmatched group) maps to `0.0`
— a smooth, monotone-decreasing, *higher-is-better* score (qc-objective-mapping §"threshold-anchored normalizer").

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_qc_scorer.py
from __future__ import annotations

import math

import pandas as pd
import pytest

from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune._scoring._qc_scorer import QCScorer, _threshold_anchored


def test_threshold_anchored_anchors():
    assert _threshold_anchored(0.0, 0.10) == pytest.approx(1.0)        # perfect
    assert _threshold_anchored(0.10, 0.10) == pytest.approx(0.5)       # at fail boundary
    assert _threshold_anchored(0.20, 0.10) == pytest.approx(0.25)      # 2× boundary
    assert _threshold_anchored(float("inf"), 0.10) == 0.0             # unmatched group
    # monotone decreasing in the metric
    assert _threshold_anchored(0.05, 0.10) > _threshold_anchored(0.15, 0.10)


def _layout(n: int, name: str = "p1") -> pd.DataFrame:
    return pd.DataFrame(
        {"Metadata_ImageName": [name] * n, "Object_Label": list(range(n))}
    )


def _measurements(n: int, name: str = "p1") -> pd.DataFrame:
    return pd.DataFrame(
        {"Metadata_ImageName": [name] * n, "Object_Label": list(range(n))}
    )


def test_score_image_perfect_match_is_one():
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout(96), groupby=["Metadata_ImageName"]
        )
    )
    out = scorer.score_image(None, _measurements(96))
    assert set(out) == {"Count"}
    assert out["Count"] == pytest.approx(1.0)


def test_score_image_at_fail_threshold_is_half():
    # expected 100, detected 90 → metric 0.10 == fail_threshold → t 0.5
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout(100), groupby=["Metadata_ImageName"]
        )
    )
    assert scorer.score_image(None, _measurements(90))["Count"] == pytest.approx(0.5)


def test_score_image_unmatched_group_is_zero():
    # measurement group "p2" has no metadata counterpart → metric inf → t 0
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout(96, "p1"), groupby=["Metadata_ImageName"]
        )
    )
    assert scorer.score_image(None, _measurements(10, "p2"))["Count"] == 0.0


def test_score_image_empty_measurements_is_zero():
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout(96), groupby=["Metadata_ImageName"]
        )
    )
    assert scorer.score_image(None, pd.DataFrame())["Count"] == 0.0


def test_availability_reflects_metadata():
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout(96), groupby=["Metadata_ImageName"]
        )
    )
    assert scorer.availability() is True


def test_path_configured_scorer_round_trips(tmp_path):
    # Configure the check from a CSV path so metadata_source persists through JSON.
    csv = tmp_path / "layout.csv"
    _layout(96).to_csv(csv, index=False)
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"]
        )
    )
    reloaded = QCScorer.model_validate_json(scorer.model_dump_json())
    assert reloaded.check.metadata_source == str(csv)
    # the reloaded scorer scores identically (re-read the layout from disk)
    assert reloaded.score_image(None, _measurements(96))["Count"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_qc_scorer.py -v`
Expected: FAIL — `ImportError: cannot import name 'QCScorer'`.

- [ ] **Step 3: Implement the normalizer + `QCScorer`**

```python
# src/phenotypic/tune/_scoring/_qc_scorer.py
"""The no-ground-truth Count objective.

Wraps :class:`phenotypic.analysis.ExpectedVsDetectedCount` — for each ``groupby``
unit it compares the detected colony count against an expected count from a
layout frame, yielding ``QC_Count_Metric = |detected - expected| / expected``
(``inf`` when a measurement group has no metadata counterpart). The metric is a
*lower-is-better* divergence in ``[0, ∞)``; ``_threshold_anchored`` flips and
normalizes it to a *higher-is-better* term in ``[0, 1]`` anchored on the check's
``fail_threshold``.
"""
from __future__ import annotations

import math
from typing import Any, ClassVar

import pandas as pd

from phenotypic.analysis import ExpectedVsDetectedCount

from ._scorer import Scorer


def _threshold_anchored(metric: float, fail_threshold: float) -> float:
    """Map a lower-is-better divergence to a higher-is-better score in ``[0, 1]``.

    ``t = exp(-ln2 * metric / fail_threshold)`` — so ``metric == 0`` → ``1.0``,
    ``metric == fail_threshold`` → ``0.5``, ``metric == inf`` → ``0.0``.

    Args:
        metric: The non-negative divergence (``|detected-expected|/expected``);
            ``inf`` for an unmatched group.
        fail_threshold: The metric value the check treats as a hard fail; the
            half-score anchor.

    Returns:
        The normalized score in ``[0, 1]`` (higher = better).
    """
    if not math.isfinite(metric):
        return 0.0
    if metric <= 0.0:
        return 1.0
    return math.exp(-math.log(2.0) * metric / fail_threshold)


class QCScorer(Scorer):
    """Count-only quality objective backed by ``ExpectedVsDetectedCount``.

    Args:
        check: A configured count check. **Configure it from a metadata path**
            (``metadata="layout.csv"``) so ``metadata_source`` persists and the
            scorer round-trips through ``tuning_spec.json``; a check built from
            an in-memory frame cannot be rebuilt from JSON.

    Examples:
        >>> import pandas as pd
        >>> from phenotypic.analysis import ExpectedVsDetectedCount
        >>> from phenotypic.tune import QCScorer
        >>> layout = pd.DataFrame(
        ...     {"Metadata_ImageName": ["p1"] * 96, "Object_Label": list(range(96))}
        ... )
        >>> scorer = QCScorer(
        ...     check=ExpectedVsDetectedCount(
        ...         metadata=layout, groupby=["Metadata_ImageName"]
        ...     )
        ... )
        >>> measured = pd.DataFrame(
        ...     {"Metadata_ImageName": ["p1"] * 96, "Object_Label": list(range(96))}
        ... )
        >>> round(scorer.score_image(None, measured)["Count"], 3)
        1.0
    """

    term_name: ClassVar[str] = "Count"

    check: ExpectedVsDetectedCount

    def availability(self) -> bool:
        """``True`` when the check resolved a non-empty layout frame."""
        return not self.check.metadata.empty

    def score_image(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Return ``{"Count": t}`` — the normalized per-image count score.

        Runs the count check on ``measurements``, normalizes each group's
        ``QC_Count_Metric`` via :func:`_threshold_anchored`, and averages across
        groups (a single-plate frame has one group, so the mean is that group's
        score). An empty frame scores ``0.0``.

        Args:
            image: Unused (the count objective reads only the frame).
            measurements: The candidate pipeline's measurement frame.

        Returns:
            ``{"Count": <score in [0, 1]>}`` (higher = better).
        """
        if measurements is None or len(measurements) == 0:
            return {self.term_name: 0.0}
        augmented = self.check.analyze(measurements)
        metric_col = self.check.metric_col()
        per_group = augmented.groupby(self.check.groupby, dropna=False)[
            metric_col
        ].first()
        fail = float(self.check.fail_threshold)
        score = float(
            per_group.map(lambda m: _threshold_anchored(float(m), fail)).mean()
        )
        return {self.term_name: score}
```

Update the subpackage `__init__.py`:

```python
# src/phenotypic/tune/_scoring/__init__.py  (replace file)
"""Internal scoring value types (private)."""
from __future__ import annotations

from ._qc_scorer import QCScorer, _threshold_anchored
from ._scorer import Scorer

__all__ = ["Scorer", "QCScorer", "_threshold_anchored"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_qc_scorer.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_scoring tests/unit/tune/test_qc_scorer.py
git add src/phenotypic/tune/_scoring tests/unit/tune/test_qc_scorer.py
git commit -m "feat(tune): Count-only QCScorer + threshold-anchored normalizer"
```

---

### Task 3: The params→pipeline builder

**Files:**
- Create: `src/phenotypic/tune/_evaluation/_builder.py`
- Create: `src/phenotypic/tune/_evaluation/__init__.py`
- Test: `tests/unit/tune/test_builder.py`

The builder clones the base, overlays each combo key onto the addressed op by **fresh
reconstruction** (`type(op)(**{**fields, **overrides})` — re-runs every validator, so the
result's `to_json()` is byte-identical to legacy `operation_class(**merged)`, which the
Phase-1d golden lock depends on), and drops `__enabled__=False` ops. Cloning via
`base.model_copy(deep=True)` then `set_ops(...)` preserves the base's `meas`/`post`/`qc`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_builder.py
from __future__ import annotations

import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tune._evaluation._builder import build_pipeline


def _base() -> ImagePipeline:
    return ImagePipeline(ops=[
        GaussianBlur(sigma=2.0),           # position 0
        OtsuDetector(ignore_zeros=False),  # position 1
    ])


def test_overlay_scalar_field_rebuilds_op_and_leaves_base_untouched():
    base = _base()
    candidate = build_pipeline(base, {"1.ignore_zeros": True, "0.sigma": 4.0})
    cops = candidate.get_ops()
    assert cops["OtsuDetector"].ignore_zeros is True
    assert cops["GaussianBlur"].sigma == 4.0
    # base is unmutated
    assert base.get_ops()["OtsuDetector"].ignore_zeros is False
    assert base.get_ops()["GaussianBlur"].sigma == 2.0


def test_no_overlay_yields_equivalent_pipeline():
    base = _base()
    candidate = build_pipeline(base, {})
    assert list(candidate.get_ops().keys()) == ["GaussianBlur", "OtsuDetector"]


def test_presence_false_drops_the_op():
    base = _base()
    candidate = build_pipeline(base, {"0.GaussianBlur.__enabled__": False})
    assert list(candidate.get_ops().keys()) == ["OtsuDetector"]


def test_presence_true_keeps_the_op():
    base = _base()
    candidate = build_pipeline(base, {"0.GaussianBlur.__enabled__": True, "0.sigma": 1.5})
    assert list(candidate.get_ops().keys()) == ["GaussianBlur", "OtsuDetector"]
    assert candidate.get_ops()["GaussianBlur"].sigma == 1.5


def test_presence_class_mismatch_raises():
    base = _base()
    # position 0 is a GaussianBlur, not an OtsuDetector
    with pytest.raises(ValueError, match="OtsuDetector"):
        build_pipeline(base, {"0.OtsuDetector.__enabled__": False})


def test_position_out_of_range_raises():
    base = _base()
    with pytest.raises(IndexError):
        build_pipeline(base, {"5.sigma": 1.0})


def test_nested_key_not_supported_in_phase_1():
    base = _base()
    with pytest.raises(NotImplementedError):
        build_pipeline(base, {"1.detectors[0].block_size": 7})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_builder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.tune._evaluation'`.

- [ ] **Step 3: Implement the builder**

```python
# src/phenotypic/tune/_evaluation/_builder.py
"""Turn a sampled parameter combo into a runnable ``ImagePipeline``.

The combo is a flat ``{root-relative-key: value}`` mapping (the same keys a
``SearchSpace`` knob carries; see master §5). ``build_pipeline`` clones the base
pipeline, overlays each key onto the op it addresses by **fresh reconstruction**
(full validation — byte-compatible with the legacy sweep's
``operation_class(**merged)``), and drops ops toggled off via ``__enabled__``.
"""
from __future__ import annotations

from typing import Any

from phenotypic import ImagePipeline


def _parse_key(key: str, ordered_ops: list) -> tuple[int, str]:
    """Resolve a combo key to ``(position, field)``.

    ``field`` is ``"__enabled__"`` for a presence toggle, otherwise a scalar
    field name. Presence keys carry the class name (``"0.GaussianBlur.__enabled__"``)
    which is validated against the op actually at that position.

    Args:
        key: A root-relative combo key.
        ordered_ops: The base pipeline's ops in order (for bounds + class checks).

    Returns:
        ``(position, field)``.

    Raises:
        IndexError: If the position is out of range.
        ValueError: If a presence key's class name does not match the op there.
        NotImplementedError: For nested keys (Phase 3).
    """
    parts = key.split(".")
    position = int(parts[0])
    if not 0 <= position < len(ordered_ops):
        raise IndexError(
            f"combo key {key!r} targets position {position}, but the base "
            f"pipeline has {len(ordered_ops)} op(s)"
        )

    if parts[-1] == "__enabled__":
        if len(parts) == 3:
            expected_cls = parts[1]
            actual_cls = type(ordered_ops[position]).__name__
            if actual_cls != expected_cls:
                raise ValueError(
                    f"presence key {key!r} targets class {expected_cls!r}, but "
                    f"position {position} holds a {actual_cls!r}"
                )
        return position, "__enabled__"

    if len(parts) == 2:
        return position, parts[1]

    raise NotImplementedError(
        f"nested overlay key {key!r} is not supported in Phase 1 "
        "(nested-op tuning lands with Phase 3 search-space inference)"
    )


def _rebuild_op(op: Any, overrides: dict[str, Any]) -> Any:
    """Return a fresh op of the same type with ``overrides`` applied.

    Reconstructs through the constructor (re-running validators) rather than
    mutating in place, so the result serializes byte-identically to a freshly
    constructed op — operations are immutable/keyword-only.

    Args:
        op: The base operation instance.
        overrides: Field name → new value.

    Returns:
        A new operation instance.
    """
    fields = {name: getattr(op, name) for name in type(op).model_fields}
    fields.update(overrides)
    return type(op)(**fields)


def build_pipeline(base: ImagePipeline, params: dict[str, Any]) -> ImagePipeline:
    """Clone ``base``, overlay ``params``, and drop ``__enabled__=False`` ops.

    Args:
        base: The base pipeline embedded in the ``TuningSpec``.
        params: A flat combo (``{root-relative-key: value}``) from a strategy.

    Returns:
        A new ``ImagePipeline`` carrying the base's measurements/post/qc with the
        tuned operations.

    Raises:
        IndexError / ValueError / NotImplementedError: Propagated from key parsing.
    """
    ordered_ops = list(base.get_ops().values())

    overrides: dict[int, dict[str, Any]] = {}
    enabled: dict[int, bool] = {}
    for key, value in params.items():
        position, field = _parse_key(key, ordered_ops)
        if field == "__enabled__":
            enabled[position] = bool(value)
        else:
            overrides.setdefault(position, {})[field] = value

    new_ops = []
    for position, op in enumerate(ordered_ops):
        if not enabled.get(position, True):
            continue  # presence toggled off → drop the op
        op_overrides = overrides.get(position)
        new_ops.append(_rebuild_op(op, op_overrides) if op_overrides else op)

    candidate = base.model_copy(deep=True)  # preserves meas/post/qc/name
    candidate.set_ops(new_ops)
    return candidate
```

```python
# src/phenotypic/tune/_evaluation/__init__.py
"""Internal evaluation machinery (private)."""
from __future__ import annotations

from ._builder import build_pipeline

__all__ = ["build_pipeline"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_builder.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_evaluation tests/unit/tune/test_builder.py
git add src/phenotypic/tune/_evaluation tests/unit/tune/test_builder.py
git commit -m "feat(tune): params->pipeline builder (overlay + presence drop)"
```

---

### Task 4: `EvaluationResult` + the `Evaluator` (3-step loop)

**Files:**
- Create: `src/phenotypic/tune/_evaluation/_evaluator.py`
- Modify: `src/phenotypic/tune/_evaluation/__init__.py`
- Test: `tests/unit/tune/test_evaluator.py`

The `Evaluator` runs the uniform loop: build the candidate, `score_image` per
calibration image, robust-aggregate each term as `median − λ·IQR` (λ = `stability_weight`,
default 0.5 — penalizes cross-image instability), then `finalize`. A candidate whose
pipeline/scorer raises is assigned `failure_score` (default 0.0 — the floor for the
higher-is-better `[0,1]` objective), never crashing the sweep. (This is the **CV-only MVP**:
one pass over the calibration set; multi-fold group-aware CV, multi-fidelity pruning, and
the adaptive held-out guard from robust-eval §6–§8 layer on in later phases.)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_evaluator.py
from __future__ import annotations

import pandas as pd
import pytest
from pydantic import PrivateAttr

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune._scoring._scorer import Scorer
from phenotypic.tune._evaluation._evaluator import (
    EvaluationResult,
    Evaluator,
    _robust_aggregate,
)


def test_robust_aggregate_penalizes_spread():
    # median 2.5, IQR (3.25 - 1.75) = 1.5 → 2.5 - 0.5*1.5 = 1.75
    assert _robust_aggregate([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(1.75)


def test_robust_aggregate_single_value_is_that_value():
    assert _robust_aggregate([0.8], 0.5) == pytest.approx(0.8)  # IQR 0


class _SequenceScorer(Scorer):
    """Returns preset per-call values (term ``"X"``), ignoring its inputs."""

    values: list[float]
    _cursor: int = PrivateAttr(default=0)

    def score_image(self, image, measurements) -> dict[str, float]:
        value = self.values[self._cursor % len(self.values)]
        self._cursor += 1
        return {"X": float(value)}


class _RaisingScorer(Scorer):
    def score_image(self, image, measurements) -> dict[str, float]:
        raise RuntimeError("scoring blew up")


def test_evaluate_runs_3_step_loop_and_aggregates():
    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()
    scorer = _SequenceScorer(values=[1.0, 2.0, 3.0])
    result = Evaluator().evaluate(base, scorer, {}, [img, img, img])
    assert isinstance(result, EvaluationResult)
    assert result.n_images == 3
    # term X aggregated: median 2.0, IQR (2.5 - 1.5)=1.0 → 2.0 - 0.5*1.0 = 1.5
    assert result.terms == {"X": pytest.approx(1.5)}
    # default finalize = mean of one term → 1.5
    assert result.score == pytest.approx(1.5)


def test_evaluate_requires_images():
    with pytest.raises(ValueError):
        Evaluator().evaluate(ImagePipeline(ops=[OtsuDetector()]), _SequenceScorer(values=[1.0]), {}, [])


def test_evaluate_failure_assigns_failure_score():
    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()
    result = Evaluator(failure_score=0.0).evaluate(base, _RaisingScorer(), {}, [img])
    assert result.score == 0.0
    assert result.terms == {}
    assert result.n_images == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_evaluator.py -v`
Expected: FAIL — `ImportError: cannot import name 'Evaluator'`.

- [ ] **Step 3: Implement `_robust_aggregate`, `EvaluationResult`, `Evaluator`**

```python
# src/phenotypic/tune/_evaluation/_evaluator.py
"""The candidate evaluator — the uniform 3-step robust-evaluation loop.

For one parameter combo: build the candidate pipeline, ``score_image`` over the
calibration set, robust-aggregate each term as ``median - λ·IQR`` (the spread
penalty rewards parameters that are stable across images, not just good on
average), then ``finalize`` to the scalar objective the optimizer maximizes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from .._scoring._scorer import Scorer
from ._builder import build_pipeline


def _robust_aggregate(values: list[float], stability_weight: float) -> float:
    """Reduce a term's per-image scores to ``median - stability_weight·IQR``.

    Args:
        values: The per-image scores for one term (higher = better).
        stability_weight: λ — how hard cross-image spread is penalized.

    Returns:
        The stability-penalized central tendency. For a single value the IQR is
        ``0`` and the result is that value.
    """
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    return median - stability_weight * float(q75 - q25)


class EvaluationResult(BaseModel):
    """The outcome of evaluating one candidate over the calibration set."""

    model_config = ConfigDict(frozen=True)

    score: float                  # the finalized scalar objective (higher = better)
    terms: dict[str, float]       # robust-aggregated per-term scores
    n_images: int                 # calibration images evaluated


class Evaluator(BaseModel):
    """Score a candidate combo over a calibration set (CV-only MVP)."""

    model_config = ConfigDict(frozen=True)

    stability_weight: float = 0.5   # λ in median - λ·IQR
    failure_score: float = 0.0      # assigned when a candidate fails to run

    def evaluate(
        self,
        base: Any,
        scorer: Scorer,
        params: dict[str, Any],
        images: list,
    ) -> EvaluationResult:
        """Build, score, robust-aggregate, and finalize one candidate.

        Args:
            base: The base pipeline embedded in the ``TuningSpec``.
            scorer: The objective.
            params: The sampled combo (``{root-relative-key: value}``).
            images: The calibration images (must be non-empty).

        Returns:
            The candidate's :class:`EvaluationResult`.

        Raises:
            ValueError: If ``images`` is empty.
        """
        if not images:
            raise ValueError(
                "Evaluator.evaluate requires at least one calibration image"
            )

        candidate = build_pipeline(base, params)

        per_term: dict[str, list[float]] = {}
        try:
            for image in images:
                measurements = candidate.measure(image, apply_post=False)
                for term, value in scorer.score_image(image, measurements).items():
                    per_term.setdefault(term, []).append(float(value))
        except Exception:
            # A broken candidate scores worst, never crashing the sweep.
            return EvaluationResult(
                score=self.failure_score, terms={}, n_images=len(images)
            )

        aggregated = {
            term: _robust_aggregate(values, self.stability_weight)
            for term, values in per_term.items()
        }
        score = float(scorer.finalize(aggregated))
        return EvaluationResult(
            score=score, terms=aggregated, n_images=len(images)
        )
```

Update the subpackage `__init__.py`:

```python
# src/phenotypic/tune/_evaluation/__init__.py  (replace file)
"""Internal evaluation machinery (private)."""
from __future__ import annotations

from ._builder import build_pipeline
from ._evaluator import EvaluationResult, Evaluator, _robust_aggregate

__all__ = ["build_pipeline", "Evaluator", "EvaluationResult", "_robust_aggregate"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_evaluator.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix src/phenotypic/tune/_evaluation tests/unit/tune/test_evaluator.py
git add src/phenotypic/tune/_evaluation tests/unit/tune/test_evaluator.py
git commit -m "feat(tune): Evaluator 3-step loop + median-λ·IQR aggregation"
```

---

### Task 5: Integration (real `QCScorer` + `measure` + synth plate) + public exports

**Files:**
- Modify: `src/phenotypic/tune/__init__.py`
- Test: `tests/unit/tune/test_evaluation_integration.py`

Proves the three real pieces compose: `build_pipeline` → `ImagePipeline.measure` →
`QCScorer.score_image`. `load_synth_yeast_plate()` is a `GridImage` of 96 colonies; an
`OtsuDetector` recovers all 96, so a layout expecting 96 scores `Count == 1.0`, and a
layout expecting 120 scores `Count == exp(−ln2·(24/120)/0.10) == 0.25`.

- [ ] **Step 1: Export the Phase-1c public surface**

```python
# src/phenotypic/tune/__init__.py  (extend the Phase-1a exports)
"""Parameter-tuning engine — public API (in progress)."""
from __future__ import annotations

from ._evaluation import EvaluationResult, Evaluator, build_pipeline
from ._scoring import QCScorer, Scorer
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
    # search space (Phase 1a)
    "Categorical",
    "IntRange",
    "FloatRange",
    "Fixed",
    "Domain",
    "Knob",
    "SearchSpace",
    # scoring (Phase 1c)
    "Scorer",
    "QCScorer",
    # evaluation (Phase 1c)
    "Evaluator",
    "EvaluationResult",
    "build_pipeline",
]
```

> Preserve any Phase-1b strategy exports already present in this file — merge, don't
> overwrite. (1b added `GridStrategy`/`RandomStrategy`/`StrategyConfig`/… to `__all__`;
> keep them.)

- [ ] **Step 2: Write the failing integration test**

```python
# tests/unit/tune/test_evaluation_integration.py
from __future__ import annotations

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune import Evaluator, QCScorer


def _layout_csv(tmp_path, n: int):
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {
            "Metadata_ImageName": ["Synthetic96PlateWithObjects"] * n,
            "Object_Label": list(range(n)),
        }
    ).to_csv(csv, index=False)
    return str(csv)


def test_perfect_count_scores_one(tmp_path):
    base = ImagePipeline(ops=[OtsuDetector()])
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout_csv(tmp_path, 96), groupby=["Metadata_ImageName"]
        )
    )
    result = Evaluator().evaluate(base, scorer, {}, [load_synth_yeast_plate()])
    assert result.n_images == 1
    assert result.terms["Count"] == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0)


def test_count_mismatch_scores_below_one(tmp_path):
    # layout expects 120, detector finds 96 → metric 24/120 = 0.2 → t = exp(-ln2*2) = 0.25
    base = ImagePipeline(ops=[OtsuDetector()])
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout_csv(tmp_path, 120), groupby=["Metadata_ImageName"]
        )
    )
    result = Evaluator().evaluate(base, scorer, {}, [load_synth_yeast_plate()])
    assert result.score == pytest.approx(0.25, abs=1e-6)
```

- [ ] **Step 3: Run test to verify it fails, then passes**

Run: `uv run pytest tests/unit/tune/test_evaluation_integration.py -v`
Expected (before Step 1 export merge): FAIL — `ImportError`. After: PASS (2 tests).

> If `OtsuDetector()` with defaults recovers ≠96 objects on the synth plate, the
> `0.25`/`1.0` anchors shift. The synth fixture is stable (96 colonies, full recovery);
> if a future detector default changes recovery, update the expected count in the layout,
> not the normalizer.

- [ ] **Step 4: Full Phase-1c gate — suite + types + lint + doctests**

Run:
```bash
uv run pytest tests/unit/tune -q
uv run pytest --doctest-modules src/phenotypic/tune/_scoring/_qc_scorer.py -q
uv run mypy src/phenotypic/tune
uv run ruff check src/phenotypic/tune tests/unit/tune
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/__init__.py tests/unit/tune/test_evaluation_integration.py
git commit -m "feat(tune): wire QCScorer+Evaluator public API + integration test"
```

---

## Self-Review

**Spec coverage:**
- `engine-architecture.md` §5 (pydantic ABCs) → `Scorer(BaseModel, ABC)` (Task 1); value-model `EvaluationResult` (Task 4).
- `qc-objective-mapping.md` (Count-only `QCScorer`, threshold-anchored normalizer, per-group reduction, path-configured round-trip, graceful degradation on empty) → Task 2.
- `robust-evaluation.md` (uniform 3-step loop, `median − λ·IQR` with λ=0.5, Scorer-owned normalization, failure policy) → Task 4. **CV-only MVP**: single calibration pass; multi-fold group-aware CV / multi-fidelity pruning / adaptive held-out guard explicitly deferred (noted in Task 4).
- engine-arch §6 base-pipeline decision (clone base → overlay → drop disabled) → Task 3 `build_pipeline`.

**Deferred (correctly out of Phase 1c):** the batch reliability panel + hybrid geometric fusion + anti-gaming guard (qc §3–§7) are Phase-2+/multi-objective; `ReferenceFreeScorer`/`SupervisedScorer`/`CompositeScorer` are Phases 3–4; nested-op overlay is Phase 3.

**Placeholder scan:** none — every code step is complete and runnable. The two empirical anchors (96 objects → `Count=1.0`; expected-120 → `0.25`) were verified against the live `load_synth_yeast_plate()` + `OtsuDetector()` + `measure()` stack.

**Type consistency:** `Scorer.score_image(image, measurements) -> dict[str, float]`, `Scorer.finalize(terms) -> float`, `Scorer.availability() -> bool`; `QCScorer.check: ExpectedVsDetectedCount`, `term_name="Count"`; `build_pipeline(base, params) -> ImagePipeline`; `Evaluator(stability_weight=0.5, failure_score=0.0).evaluate(base, scorer, params, images) -> EvaluationResult(score, terms, n_images)`; `_robust_aggregate(values, stability_weight) -> float`. These names are what Phase 1d (`TuningEngine` calls `evaluator.evaluate(spec.pipeline, spec.scorer, combo, images)`; `TuningSpec` embeds `pipeline`/`scorer`/`evaluator`) imports.

## Hand-off to 1d

Phase 1d builds `TuningEngine` (the ask-and-tell loop pairing a Phase-1b `SearchStrategy` with this `Evaluator`), `TuningSpec` (embedding `pipeline: ImagePipeline` per the engine-arch §6 refinement, plus `search_space`/`scorer`/`strategy`/`evaluator`/budget), the StudyStore + RF-permutation importance fallback, the CLI (`-i/-o`, output layout), and the **golden byte-compat lock** (enumerate the grid → `build_pipeline` each combo → compare `to_json()` against `tests/fixtures/tune/grid_golden_manifest.json` from Phase 0) — then deletes `sweep`.
