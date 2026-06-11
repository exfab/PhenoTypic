# Phase 1 — Evaluator cost math + scorer migration + gap + suspicious

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking. Each task is **TDD**: write a failing test, run it red, write the minimal implementation, run it green, commit.

**Goal:** Flip the **per-term and per-child math** of the tuner from "higher-is-better `[0,1]` goodness" to "lower-is-better `[0,1]` **cost**" (`0` = perfect, `1` = worst). This phase (1) converts the `Scorer` base into the `to_cost` template method, (2) migrates every leaf scorer to emit cost (via `_score_terms` + `_TERM_SENSE = HIGHER_BETTER` + `to_cost` complement — OQ1=A keeps the internal `[0,1]` folds), (3) rewrites the Evaluator aggregate/floor/suspicious math into cost-space, and (4) fixes the generalization-gap sign + relative-IQR blowups.

**This is the high-risk phase.** It is **ATOMIC with Phase 2** — they share one branch and one PR. The ASHA `SuccessiveHalvingPruner` (`_strategies/_optuna.py`) is direction-aware: it reads `study.direction`. The Evaluator reports `running_score` to it (`Evaluator.evaluate` → `channel.report(running_score, scored)`). **Phase 1 flips the reported value into cost; Phase 2 flips `study.direction` to `"minimize"`.** Either alone breaks pruning and best-selection.

> ⚠️ **DONE-CRITERIA / EXPECTED RED.** Phase 1's *unit* tests — the cost-value asserts, the gap sign, the `[0,1]` clamp, and `_is_suspicious` — all pass **standalone** after this phase. But **end-to-end study / pruner / best-selection tests stay RED until Phase 2 flips `study.direction`** (e.g. `tests/unit/tune/test_optuna_pruning.py`, any test that runs a full `TuningEngine.optimize` and asserts on the winner). **Do not try to make those green in Phase 1.** Phases 1 + 2 land together; the suite is only fully green once both are merged.

**Dependencies (from Phase 0, already merged):** `src/phenotypic/tune/_scoring/_orient.py` exports `Sense` (a `str, Enum` with `LOWER_BETTER` / `HIGHER_BETTER`), `to_cost(value, *, sense, anchor=None)`, and `clamp01(value)`. Confirm with `uv run --extra tune python -c "from phenotypic.tune._scoring._orient import Sense, to_cost, clamp01; print('ok')"` before starting. If it errors, Phase 0 has not landed — stop.

**Conventions:**
- Test command: `uv run --extra tune pytest <path> -v`.
- Type/lint at the END of the phase (after all tasks): `uv run mypy src/phenotypic/tune` and `uv run ruff check --fix src/phenotypic/tune`.
- All `file:line` refs below are from the worktree at planning time and drift ±1–2 — **re-resolve by reading the file and matching the cited symbol name** before editing.
- Commit after every green task with the exact `git add` / `git commit` shown. (The orchestrator owns the final PR.)

**Co-author trailer for every commit:**
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 1 — `Scorer` becomes the `to_cost` template method

Convert the base `Scorer` so `score_image` is the **one orientation boundary**: subclasses implement `_score_terms` (natural values) + declare `_TERM_SENSE`, and the base wraps each term via `to_cost`. This is the exact README "Shared contract" `Scorer` template method.

**Files:**
- Modify: `src/phenotypic/tune/_scoring/_scorer.py` (the `Scorer` class — re-resolve; `score_image` `@abstractmethod` is near `:55`, `finalize` near `:82`)
- Modify: `tests/unit/tune/test_evaluator.py` (the in-file `_SequenceScorer` / `_RaisingScorer` test doubles override `score_image`; they must move to `_score_terms` — re-resolve, near `:27`–`:41`)
- Create: `tests/unit/tune/test_scorer_template.py`

### Step 1.1 — write the failing test

Create `tests/unit/tune/test_scorer_template.py`:

```python
from __future__ import annotations

import pytest

from phenotypic.tune._scoring._orient import Sense
from phenotypic.tune._scoring._scorer import Scorer


class _HigherBetterLeaf(Scorer):
    """Emits a bounded [0,1] goodness term; base must complement it to cost."""

    _TERM_SENSE = Sense.HIGHER_BETTER

    def _score_terms(self, image, measurements) -> dict[str, float]:
        return {"G": 0.8}


class _LowerBetterLeaf(Scorer):
    """Emits a bounded [0,1] cost term directly (the cost-native default)."""

    # _TERM_SENSE defaults to Sense.LOWER_BETTER

    def _score_terms(self, image, measurements) -> dict[str, float]:
        return {"L": 0.2}


def test_default_term_sense_is_lower_better():
    assert Scorer._TERM_SENSE is Sense.LOWER_BETTER


def test_higher_better_leaf_is_complemented_to_cost():
    # to_cost(0.8, HIGHER_BETTER) == 1 - 0.8 == 0.2
    assert _HigherBetterLeaf().score_image(None, None) == {"G": pytest.approx(0.2)}


def test_lower_better_leaf_passes_through_as_cost():
    # to_cost(0.2, LOWER_BETTER) == 0.2 (identity)
    assert _LowerBetterLeaf().score_image(None, None) == {"L": pytest.approx(0.2)}


def test_term_anchor_defaults_to_none():
    assert _LowerBetterLeaf()._term_anchor("L") is None


def test_score_terms_is_abstract():
    # A subclass that forgets _score_terms cannot be instantiated.
    class _Incomplete(Scorer):
        pass

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]
```

### Step 1.2 — run it red

```
uv run --extra tune pytest tests/unit/tune/test_scorer_template.py -v
```
Expected: FAIL — `AttributeError: type object 'Scorer' has no attribute '_TERM_SENSE'` (and the leaves error because `Scorer` still declares `score_image` abstract, not `_score_terms`).

### Step 1.3 — minimal implementation

In `src/phenotypic/tune/_scoring/_scorer.py`, add the imports near the top (after the existing `from pydantic import BaseModel, ConfigDict`):

```python
from typing import Any, ClassVar, Mapping, TypeAlias

from ._orient import Sense, to_cost
```
(Keep the existing `import pandas as pd` and `from phenotypic.tools_.typing_ import polymorphic_field`; add `ClassVar` to the existing `typing` import line rather than duplicating it.)

Replace the `Scorer` class body's `score_image` abstractmethod with the template method. The class currently reads (re-resolve):

```python
class Scorer(BaseModel, ABC):
    """..."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def score_image(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Score one image's measurements as named terms (higher = better)."""
        raise NotImplementedError
```

Change it to:

```python
class Scorer(BaseModel, ABC):
    """Base class for tuning objectives (no-GT, supervised, reference-free, …).

    **Orientation is a base-class template method (the one cost boundary).** A
    scorer emits its *natural* per-term values in :meth:`_score_terms` (a
    divergence stays a divergence; Dice stays Dice), declares the *sense* of
    those values once via :attr:`_TERM_SENSE`, and the base :meth:`score_image`
    wraps each term into **cost ∈ [0,1]** (``0`` perfect, ``1`` worst, lower is
    better — the optimizer minimizes) via the shared :func:`to_cost` helper.

    To add a scorer:

    1. Subclass :class:`Scorer` and implement
       :meth:`_score_terms` → ``dict[str, float]`` returning your **natural**
       per-term values — do **not** flip or normalize by hand.
    2. Declare :attr:`_TERM_SENSE`: ``Sense.LOWER_BETTER`` (the default — larger
       value = worse, a loss/divergence) or ``Sense.HIGHER_BETTER`` (larger =
       better, e.g. Dice / IoU / ICC / solidity).
    3. Override :meth:`_term_anchor` **only if a term is unbounded** (return the
       half-cost scale, e.g. a QC check's ``fail_threshold``); bounded ``[0,1]``
       terms need nothing.
    4. Do **not** add scalarization parameters (``ε`` / ``ρ`` / weights /
       normalization are framework-derived).
    5. Register: re-export from ``tune/__init__.py`` and the class registry, or
       the GUI and ``from_json`` cannot see it.

    Production scorers must be **stateless** across :meth:`score_image` calls:
    the engine reuses one scorer instance for every trial, so per-trial mutable
    state would bleed across candidates. (A test double that deliberately returns
    a preset sequence via a private cursor is the documented exception.)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: Sense of this scorer's natural per-term values (v1: uniform per scorer).
    #: ``LOWER_BETTER`` is cost-native — a raw-loss scorer needs no annotation; a
    #: goodness-emitting scorer must declare ``HIGHER_BETTER``.
    _TERM_SENSE: ClassVar[Sense] = Sense.LOWER_BETTER

    @abstractmethod
    def _score_terms(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """This scorer's **natural** per-term values (its own convention).

        Args:
            image: The (already-processed) image — duck-typed; reference-free
                scorers read its mask/objmap, the ``QCScorer`` ignores it.
            measurements: The measurement frame the candidate pipeline produced
                for ``image`` (the output of ``ImagePipeline.measure``).

        Returns:
            A mapping of term name → natural value for this image (its own
            sense, not yet oriented to cost). Keys must be stable across images
            so the ``Evaluator`` can aggregate per term.
        """
        raise NotImplementedError

    def _term_anchor(self, term: str) -> float | None:
        """The half-cost anchor for an unbounded term, else ``None``.

        Args:
            term: The term name from :meth:`_score_terms`.

        Returns:
            ``None`` when the term is already bounded in ``[0,1]`` (the default —
            no anchoring); a positive float (the half-cost scale) for an
            unbounded magnitude that :func:`to_cost` should threshold-anchor.
        """
        return None

    def score_image(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Orient this scorer's natural terms into **cost ∈ [0,1]** (lower = better).

        The single orientation point: each natural term from
        :meth:`_score_terms` is mapped to cost via :func:`to_cost`, using this
        scorer's :attr:`_TERM_SENSE` and :meth:`_term_anchor`.

        Args:
            image: The processed image (passed through to :meth:`_score_terms`).
            measurements: The candidate pipeline's measurement frame.

        Returns:
            A mapping of term name → cost in ``[0,1]`` (``0`` perfect, ``1``
            worst). Keys are stable across images for per-term aggregation.
        """
        return {
            term: to_cost(
                value, sense=self._TERM_SENSE, anchor=self._term_anchor(term)
            )
            for term, value in self._score_terms(image, measurements).items()
        }
```

Then fix the two in-file test doubles in `tests/unit/tune/test_evaluator.py` so they implement `_score_terms` (they previously overrode `score_image`). Re-resolve the class bodies, near `:27`–`:41`:

```python
class _SequenceScorer(Scorer):
    """Returns preset per-call values (term ``"X"``), ignoring its inputs.

    Emits its values as **cost** directly (the ``LOWER_BETTER`` default), so the
    base ``score_image`` passes them through unchanged.
    """

    values: list[float]
    _cursor: int = PrivateAttr(default=0)

    def _score_terms(self, image, measurements) -> dict[str, float]:
        value = self.values[self._cursor % len(self.values)]
        self._cursor += 1
        return {"X": float(value)}


class _RaisingScorer(Scorer):
    def _score_terms(self, image, measurements) -> dict[str, float]:
        raise RuntimeError("scoring blew up")
```
(Do **not** change the numeric `test_robust_aggregate_*` / `test_evaluate_*` asserts in this task — `_SequenceScorer` emits `LOWER_BETTER` cost, so `to_cost` is identity and the existing `1.0/2.0/3.0 → 1.5` arithmetic is unchanged. The `_WORST_TERM` / `failure_score` value flips are Task 5/6, which update those asserts.)

### Step 1.4 — run it green

```
uv run --extra tune pytest tests/unit/tune/test_scorer_template.py -v
```
Expected: PASS (5 passed). Also re-run the evaluator file to confirm the doubles still work:
```
uv run --extra tune pytest tests/unit/tune/test_evaluator.py -v
```
Expected: PASS (the value flips for `_WORST_TERM`/`failure_score` come in Tasks 5–6; the arithmetic asserts here are LOWER_BETTER-identity and unchanged).

### Step 1.5 — commit

```
git add src/phenotypic/tune/_scoring/_scorer.py tests/unit/tune/test_scorer_template.py tests/unit/tune/test_evaluator.py
git commit -m "$(cat <<'EOF'
refactor(tune): make Scorer a to_cost template method

score_image now orients each scorer's natural _score_terms via to_cost
(_TERM_SENSE + _term_anchor); _score_terms is the new abstractmethod.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — migrate `QCScorer` to emit cost

`QCScorer` keeps its internal `_threshold_anchored` fold (OQ1=A → bounded `[0,1]` goodness), renames `score_image` to `_score_terms`, and declares `_TERM_SENSE = HIGHER_BETTER`. The base then complements `1 − goodness` → the scorer now emits **cost**.

**Files:**
- Modify: `src/phenotypic/tune/_scoring/_qc_scorer.py` (the `QCScorer` class — `score_image` near `:113`, doctest near `:101`)
- Modify: `tests/unit/tune/test_qc_scorer.py` (invert the cost asserts)
- Modify: `tests/unit/tune/test_qc_gaming_regression.py` (invert the gaming direction)

### Step 2.1 — write the failing test (update existing asserts to cost)

In `tests/unit/tune/test_qc_scorer.py`, invert the four `score_image` value asserts (re-resolve line numbers; the helper functions `_layout`/`_measurements` are unchanged):

- `test_score_image_perfect_match_is_one` → rename to `test_score_image_perfect_match_is_zero_cost`, body:
  ```python
  out = scorer.score_image(None, _measurements(96))
  assert set(out) == {"Count"}
  assert out["Count"] == pytest.approx(0.0)  # perfect match = zero cost
  ```
- `test_score_image_at_fail_threshold_is_half` body assert → `== pytest.approx(0.5)` stays `0.5` (the complement of `0.5` is `0.5` — fail-boundary is the cost midpoint; keep the test, update its comment to "metric 0.10 == fail_threshold → goodness 0.5 → cost 0.5").
- `test_score_image_unmatched_group_is_zero` → rename to `test_score_image_unmatched_group_is_worst_cost`, body assert → `== pytest.approx(1.0)` (inf divergence → goodness 0 → cost 1).
- `test_score_image_empty_measurements_is_zero` → rename to `test_score_image_empty_measurements_is_worst_cost`, body assert → `== pytest.approx(1.0)` (empty frame floors goodness to 0 → cost 1).
- `test_path_configured_scorer_round_trips`: the final assert `["Count"] == pytest.approx(1.0)` → `== pytest.approx(0.0)`.

(Leave `test_threshold_anchored_anchors` and `test_availability_reflects_metadata` unchanged — `_threshold_anchored` is the **internal goodness** fold and is not flipped; availability is direction-agnostic.)

In `tests/unit/tune/test_qc_gaming_regression.py`, invert `test_under_detect_scores_strictly_lower` → the faithful candidate now has **lower cost** than the under-detector:

```python
def test_under_detect_scores_strictly_higher_cost():
    # SAME layout (96 expected); a faithful frame detects all 96, an
    # under-detecting one detects far fewer (24). Under cost, faithful detection
    # must score STRICTLY LOWER (better) than under-detection.
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=_layout(96), groupby=["Metadata_ImageName"]
        )
    )
    faithful = scorer.score_image(None, _detected(96))["Count"]
    under_detect = scorer.score_image(None, _detected(24))["Count"]
    assert faithful < under_detect
```

### Step 2.2 — run it red

```
uv run --extra tune pytest tests/unit/tune/test_qc_scorer.py tests/unit/tune/test_qc_gaming_regression.py -v
```
Expected: FAIL — the renamed cost asserts fail because `QCScorer.score_image` still returns goodness (`1.0` for a perfect match, not `0.0`).

### Step 2.3 — minimal implementation

In `src/phenotypic/tune/_scoring/_qc_scorer.py`:

1. Add the import (after the existing `from ._scorer import Scorer`):
   ```python
   from ._orient import Sense
   ```
2. In the `QCScorer` class, add the sense ClassVar right after the existing `term_name` ClassVar (re-resolve, near `:105`):
   ```python
   #: The folded count term is a bounded [0,1] goodness; the base complements
   #: it (1 - value) into cost.
   _TERM_SENSE = Sense.HIGHER_BETTER
   ```
3. Rename `def score_image(` to `def _score_terms(` (the body is **unchanged** — it still folds to goodness via `fold_expected_vs_detected_count`). Update the docstring's `Returns:` line from "(higher = better)" to "natural goodness in ``[0, 1]`` (the base complements it to cost)".
4. Update the class-level doctest (near `:101`):
   ```python
   >>> round(scorer.score_image(None, measured)["Count"], 3)
   0.0
   ```
   and the doctest's prose line above the layout from "scores a perfect 96-well count match" → "a perfect 96-well count match → zero cost".

### Step 2.4 — run it green

```
uv run --extra tune pytest tests/unit/tune/test_qc_scorer.py tests/unit/tune/test_qc_gaming_regression.py -v
uv run --extra tune pytest --doctest-modules src/phenotypic/tune/_scoring/_qc_scorer.py -v
```
Expected: PASS (both files + the doctest).

### Step 2.5 — commit

```
git add src/phenotypic/tune/_scoring/_qc_scorer.py tests/unit/tune/test_qc_scorer.py tests/unit/tune/test_qc_gaming_regression.py
git commit -m "$(cat <<'EOF'
refactor(tune): QCScorer emits cost via _score_terms + HIGHER_BETTER

Keeps the internal _threshold_anchored goodness fold (OQ1=A); the base
to_cost complements it. Perfect match -> 0.0 cost; unmatched -> 1.0.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — migrate `SupervisedScorer` to emit cost

Both runnable tiers emit goodness today (`Region` = Dice/IoU ∈ [0,1] macro-average; `CountMAE` = folded count goodness). Rename `score_image` → `_score_terms`, declare `_TERM_SENSE = HIGHER_BETTER`. The tier dispatch + helpers are unchanged.

**Files:**
- Modify: `src/phenotypic/tune/_scoring/_supervised.py` (the `SupervisedScorer` class — `score_image` near `:202`, `_TERM_SENSE` insert near the ClassVars at `:143`–`:145`)
- Create: `tests/unit/tune/test_supervised_cost.py`

### Step 3.1 — write the failing test

The existing `test_supervised_scorer.py` (if present) pins construction / availability / term shape, not numeric region values (the numeric-vs-real-GT is deferred per the module TODO). Add a focused cost-orientation test that uses the `_score_terms` → `score_image` complement on the **count tier** (which is numerically exercisable via the count check, like QCScorer).

Create `tests/unit/tune/test_supervised_cost.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune import GroundTruthMasks, SupervisedScorer
from phenotypic.tune._scoring._orient import Sense


def _counts_csv(tmp_path):
    csv = tmp_path / "counts.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["p1"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return csv


def _measured(n: int, name: str = "p1") -> pd.DataFrame:
    return pd.DataFrame(
        {"Metadata_ImageName": [name] * n, "Object_Label": list(range(n))}
    )


def test_supervised_declares_higher_better():
    assert SupervisedScorer._TERM_SENSE is Sense.HIGHER_BETTER


def test_count_tier_perfect_match_is_zero_cost(tmp_path):
    csv = _counts_csv(tmp_path)
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=csv),
        count_check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"]
        ),
    )
    # score_image orients the goodness fold to cost: perfect 96-vs-96 -> 0.0.
    assert scorer.score_image(None, _measured(96)) == {
        "CountMAE": pytest.approx(0.0)
    }


def test_count_tier_under_detect_is_higher_cost(tmp_path):
    csv = _counts_csv(tmp_path)
    scorer = SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=csv),
        count_check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"]
        ),
    )
    faithful = scorer.score_image(None, _measured(96))["CountMAE"]
    under = scorer.score_image(None, _measured(24))["CountMAE"]
    assert faithful < under
```

### Step 3.2 — run it red

```
uv run --extra tune pytest tests/unit/tune/test_supervised_cost.py -v
```
Expected: FAIL — `test_supervised_declares_higher_better` errors (`_TERM_SENSE` missing) and the count-tier asserts get goodness (`1.0`), not cost (`0.0`).

### Step 3.3 — minimal implementation

In `src/phenotypic/tune/_scoring/_supervised.py`:

1. Add the import (after `from ._scorer import Scorer`):
   ```python
   from ._orient import Sense
   ```
2. Add the sense ClassVar right after the two existing term-name ClassVars (`region_term_name` / `count_term_name`, re-resolve near `:143`–`:145`):
   ```python
   #: Both tier terms (Region = Dice/IoU; CountMAE = folded count goodness) are
   #: bounded [0,1] goodness; the base complements them into cost.
   _TERM_SENSE = Sense.HIGHER_BETTER
   ```
3. Rename `def score_image(` to `def _score_terms(` (the tier-dispatch body is **unchanged**). In the docstring `Returns:` change "fold it to a higher-is-better ``[0, 1]`` score" / "macro-average per image" wording to note the values are **natural goodness the base complements to cost** (one line is enough; do not rewrite the tier prose).

### Step 3.4 — run it green

```
uv run --extra tune pytest tests/unit/tune/test_supervised_cost.py -v
uv run --extra tune pytest --doctest-modules src/phenotypic/tune/_scoring/_supervised.py -v
```
Expected: PASS. (The class doctest only asserts `modality()` / `availability()`, which are direction-agnostic, so it stays green.)

### Step 3.5 — commit

```
git add src/phenotypic/tune/_scoring/_supervised.py tests/unit/tune/test_supervised_cost.py
git commit -m "$(cat <<'EOF'
refactor(tune): SupervisedScorer emits cost via _score_terms + HIGHER_BETTER

Region (Dice/IoU) and CountMAE stay bounded [0,1] goodness internally;
the base to_cost complements them. Count tier: perfect -> 0.0 cost.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — migrate `ReferenceFreeScorer` to emit cost

All four terms (`ShapeRegularity`, `Contrast`, `SizeCV`, optional `Count`) are bounded `[0,1]` goodness (via `_clamp01` / `_bounded_inverse` / the count fold). Rename `score_image` → `_score_terms`, declare `_TERM_SENSE = HIGHER_BETTER`. The `_last_rho` diagnostic (`float("-inf")`, higher = better correlation) is **not** the objective — leave it, the meta-validation gate, `_ENABLE_RHO`/`_UNATTENDED_RHO`, and `availability()` untouched.

**Files:**
- Modify: `src/phenotypic/tune/_scoring/_reference_free_scorer.py` (`ReferenceFreeScorer` class — `score_image` near `:294`, `_TERM_SENSE` insert near the `count_term_name` ClassVar at `:149`)
- Modify: `tests/unit/tune/test_reference_free_scorer.py` (invert the value asserts; keep the gate / ddof / round-trip tests)

### Step 4.1 — write the failing test (update existing value asserts to cost)

In `tests/unit/tune/test_reference_free_scorer.py` (re-resolve line numbers):

- `test_score_image_returns_stable_proxy_terms`: keys unchanged; the bound check stays `0.0 <= value <= 1.0` (cost is also in `[0,1]`) — leave as-is.
- `test_score_image_includes_count_term_when_count_check_configured`: keys unchanged; the perfect-count assert `terms["Count"] == pytest.approx(1.0)` → `== pytest.approx(0.0)` (perfect count = zero cost). Update the comment.
- `test_keys_are_stable_across_images`: unchanged (keys only).
- `test_shape_regularity_reuses_schema_columns_no_recompute`: the missing-shape floor assert `terms["ShapeRegularity"] == pytest.approx(0.0)` → `== pytest.approx(1.0)` (goodness floor 0 → cost 1 = worst). Update the comment from "neutral floor" to "worst cost".
- `test_shape_regularity_clamps_solidity_quirk_into_unit_interval`: bound check `0.0 <= ... <= 1.0` unchanged.
- `test_contrast_term_reads_image_foreground_background`: the assert `["Contrast"] > 0.5` (high goodness) → `["Contrast"] < 0.5` (well-separated plate = LOW cost). Update the comment.
- `test_size_cv_term_is_high_for_uniform_sizes` → rename `..._is_low_cost_for_uniform_sizes`; assert `["SizeCV"] == pytest.approx(1.0)` → `== pytest.approx(0.0)` (CV 0 → goodness 1 → cost 0).
- `test_size_cv_uses_replicate_groups_when_configured`: `grouped == pytest.approx(1.0)` → `== pytest.approx(0.0)`; the relation `grouped > pooled` (within-group better than pooled, in goodness) → `grouped < pooled` (lower cost = better). Update the comment.
- `test_empty_measurements_floor_to_zero` → rename `..._floor_to_worst_cost`; `terms["ShapeRegularity"] == 0.0` → `== 1.0` and `terms["SizeCV"] == 0.0` → `== 1.0` (keys unchanged).
- `test_size_cv_term_reflects_ddof1_fold`: the fold value flips — the score is now the **cost** complement: `== pytest.approx(_bounded_inverse(0.5))` → `== pytest.approx(1.0 - _bounded_inverse(0.5))`. (`_bounded_inverse` is the internal goodness fold and stays unflipped; the complement happens at the boundary.)

Leave **unchanged** (direction-agnostic internal helpers / gate machinery): `test_clamp01_clamps_out_of_range`, `test_bounded_inverse_folds_dispersion_to_unit_interval`, all `test_meta_validate_*` / `test_availability_*` / `test_is_unattended_safe_*` / `test_enable_bar_*` / `test_unattended_*`, both `test_coefficient_of_variation_*`, and the registry / spec round-trip tests.

### Step 4.2 — run it red

```
uv run --extra tune pytest tests/unit/tune/test_reference_free_scorer.py -v
```
Expected: FAIL — the flipped value asserts fail because `score_image` still returns goodness.

### Step 4.3 — minimal implementation

In `src/phenotypic/tune/_scoring/_reference_free_scorer.py`:

1. Add the import (after `from ._scorer import Scorer`):
   ```python
   from ._orient import Sense
   ```
2. Add the sense ClassVar right after the existing `count_term_name` ClassVar (re-resolve near `:149`):
   ```python
   #: Every proxy term is bounded [0,1] goodness (fixed-normalized); the base
   #: complements each into cost.
   _TERM_SENSE = Sense.HIGHER_BETTER
   ```
3. Rename `def score_image(` to `def _score_terms(` (the body building `{"ShapeRegularity": ..., "Contrast": ..., "SizeCV": ...}` + optional `Count` is **unchanged**). In the docstring `Returns:` add one line: the listed terms are natural goodness the base complements to cost.
4. Update the **class-level doctest** (near `:140`–`:142`): `sorted(scorer.score_image(image, measurements))` returns the same keys (`['Contrast', 'ShapeRegularity', 'SizeCV']`) — unchanged; `scorer.availability()` → `False` — unchanged. (No numeric value is asserted in the doctest, so it stays green.)

### Step 4.4 — run it green

```
uv run --extra tune pytest tests/unit/tune/test_reference_free_scorer.py -v
uv run --extra tune pytest --doctest-modules src/phenotypic/tune/_scoring/_reference_free_scorer.py -v
```
Expected: PASS.

### Step 4.5 — commit

```
git add src/phenotypic/tune/_scoring/_reference_free_scorer.py tests/unit/tune/test_reference_free_scorer.py
git commit -m "$(cat <<'EOF'
refactor(tune): ReferenceFreeScorer emits cost via _score_terms + HIGHER_BETTER

All four proxy terms stay bounded [0,1] goodness internally; the base
to_cost complements them. _last_rho diagnostic is unchanged (not the objective).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — `CompositeScorer` `_score_terms` stub + doctest cost flip

`CompositeScorer` overrides `score_image` (the per-child merge — it must **not** re-orient already-cost children). But `_score_terms` is now abstract on the base, so the composite needs a stub that raises `NotImplementedError` to stay instantiable. The composite's children now emit cost, so its `score_image` output values flip — its **doctest** must be updated (the `finalize` *combiner* itself is Phase 3's job; in Phase 1 the geometric-mean of cost `0.0`s is just `0.0`, which is what we assert).

**Files:**
- Modify: `src/phenotypic/tune/_scoring/_composite.py` (`CompositeScorer` class — add `_score_terms` stub; doctest near `:86`–`:96`)

### Step 5.1 — write the failing test

Add to `tests/unit/tune/test_scorer_template.py` (from Task 1):

```python
def test_composite_score_terms_stub_raises():
    from phenotypic.tune import CompositeScorer

    with pytest.raises(NotImplementedError):
        CompositeScorer(scorers=[])._score_terms(None, None)
```

### Step 5.2 — run it red

```
uv run --extra tune pytest tests/unit/tune/test_scorer_template.py::test_composite_score_terms_stub_raises -v
```
Expected: FAIL — `CompositeScorer` is currently **uninstantiable** (abstract `_score_terms` from the base, not yet stubbed) → `TypeError: Can't instantiate abstract class CompositeScorer`. (This same `TypeError` would already be breaking every CompositeScorer construction across the suite the moment Task 1 landed — Step 5.3 is what makes the class instantiable again.)

### Step 5.3 — minimal implementation

In `src/phenotypic/tune/_scoring/_composite.py`, add the stub to the `CompositeScorer` class (place it just above the existing `score_image` override, re-resolve near `:178`):

```python
    def _score_terms(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Not used — the composite overrides :meth:`score_image` instead.

        ``_score_terms`` is abstract on the base, but a composite's children
        already returned **cost** (each child's own ``score_image`` oriented its
        terms), so the composite must **not** re-orient: it overrides the merge
        in :meth:`score_image` directly. This stub satisfies the abstract base.

        Raises:
            NotImplementedError: Always — call :meth:`score_image`.
        """
        raise NotImplementedError(
            "CompositeScorer overrides score_image (it merges already-cost "
            "children); _score_terms is not used."
        )
```

Then update the **class doctest** (re-resolve near `:86`–`:96`). The children now emit cost, so the perfect 96-well match is `0.0` cost per child:

- The prose line "scores a perfect 96-well count match" → "a perfect 96-well count match → **zero cost** per child".
- `>>> round(comp.finalize(terms), 3)  # geometric mean of the two child scalars` → expected output `0.0` (geometric mean of `[0.0, 0.0]` is `0.0`). Update the trailing comment to "geometric mean of the two child cost scalars (0.0 each → 0.0); Phase 3 replaces this combiner".
- `>>> {k: round(v, 3) for k, v in comp_mo.finalize(terms).items()}` → expected output `{'s0': 0.0, 's1': 0.0}`.

(Do **not** touch the `finalize` / `_geometric_mean` / `_weighted_mean` bodies — they are Phase 3. Phase 1 only makes the class instantiable and the doctest cost-consistent.)

### Step 5.4 — run it green

```
uv run --extra tune pytest tests/unit/tune/test_scorer_template.py -v
uv run --extra tune pytest --doctest-modules src/phenotypic/tune/_scoring/_composite.py -v
```
Expected: PASS (the stub test + the cost-flipped doctest).

### Step 5.5 — commit

```
git add src/phenotypic/tune/_scoring/_composite.py tests/unit/tune/test_scorer_template.py
git commit -m "$(cat <<'EOF'
refactor(tune): CompositeScorer _score_terms stub + cost-flipped doctest

The composite overrides score_image (merges already-cost children); the
abstract _score_terms raises NotImplementedError. Children now emit cost,
so the perfect-match doctest is 0.0 (geomean combiner is replaced in Phase 3).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — Evaluator robust-aggregate flips to cost + clamp (B1)

`_robust_aggregate` becomes `median + λ·IQR` (reflected) **then clamped to `[0,1]`** (pitfall #3 / B1 — the unclamped reflected cost ranges `[0, 1+λ]` and would break the `bᵢ ∈ [0,1]` invariant). Use the shared `clamp01` from `_orient`.

**Files:**
- Modify: `src/phenotypic/tune/_evaluation/_evaluator.py` (`_robust_aggregate` near `:53`–`:65`; import near `:15`–`:17`)
- Modify: `tests/unit/tune/test_evaluator.py` (invert the two `_robust_aggregate` asserts; add a clamp test)

### Step 6.1 — write the failing test

In `tests/unit/tune/test_evaluator.py`, invert the two aggregate asserts and add a clamp test (re-resolve, near `:18`–`:24`):

```python
def test_robust_aggregate_penalizes_spread():
    # cost = median + λ·IQR: median 2.5, IQR 1.5 → 2.5 + 0.5*1.5 = 3.25, clamped to 1.0
    assert _robust_aggregate([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(1.0)


def test_robust_aggregate_single_value_is_that_value():
    assert _robust_aggregate([0.2], 0.5) == pytest.approx(0.2)  # IQR 0


def test_robust_aggregate_clamps_to_unit_interval():
    # A high-variance bad term: np.percentile([0.1,0.8,0.9],[75,25]) = [0.85,0.45]
    # → median 0.8, IQR 0.40 → 0.8 + 0.5*0.40 = 1.0 (exactly at the ceiling; a
    # slightly worse term would exceed 1 and be clamped — B1: bᵢ ∈ [0,1] holds).
    assert _robust_aggregate([0.1, 0.8, 0.9], 0.5) == pytest.approx(1.0)


def test_robust_aggregate_in_unit_interval_is_not_clamped():
    # cost stays < 1: np.percentile([0.3,0.4,0.5],[75,25]) = [0.45,0.35], IQR 0.10
    # → median 0.4 + 0.5*0.10 = 0.45 (no clamp).
    assert _robust_aggregate([0.3, 0.4, 0.5], 0.5) == pytest.approx(0.45)


def test_robust_aggregate_above_one_is_clamped():
    # Genuinely > 1 before clamp: np.percentile([0.0,0.9,1.0],[75,25]) = [0.95,0.45],
    # IQR 0.50 → median 0.9 + 0.5*0.50 = 1.15 → clamped to 1.0.
    assert _robust_aggregate([0.0, 0.9, 1.0], 0.5) == pytest.approx(1.0)
```

(Re-resolve `test_robust_aggregate_penalizes_spread`'s arithmetic against the actual `_median_iqr` interpolation — `np.percentile([1,2,3,4],[75,25])` is `[3.25, 1.75]`, IQR `1.5`, median `2.5`, so `2.5 + 0.75 = 3.25` → clamped `1.0`. The single-value test input changed from `0.8` to `0.2` only to read as a cost; either value passes since IQR is 0. The `[0.1,0.8,0.9]` case lands *exactly* at `1.0` pre-clamp — `test_robust_aggregate_above_one_is_clamped` is the strict pre-clamp `> 1` proof.)

### Step 6.2 — run it red

```
uv run --extra tune pytest tests/unit/tune/test_evaluator.py -k robust_aggregate -v
```
Expected: FAIL — `_robust_aggregate` still computes `median - λ·IQR` (unclamped), so `[1,2,3,4]` returns `1.75`, not `1.0`.

### Step 6.3 — minimal implementation

In `src/phenotypic/tune/_evaluation/_evaluator.py`:

1. Add `clamp01` to the imports. The current import block (re-resolve near `:15`–`:17`) is:
   ```python
   from .._scoring._scorer import Scorer, project_objectives_to_scalar
   from .._strategies._pruning import NoOpChannel, PruningChannel
   from ._aggregate_math import _median_iqr, _relative
   from ._builder import build_pipeline
   ```
   Add after the `_scorer` import:
   ```python
   from .._scoring._orient import clamp01
   ```
2. Rewrite `_robust_aggregate` (re-resolve near `:53`):
   ```python
   def _robust_aggregate(values: list[float], stability_weight: float) -> float:
       """Reduce a term's per-image **costs** to ``clamp01(median + λ·IQR)``.

       Cost convention (lower = better): the spread penalty *adds* to the central
       tendency, so an unstable term is penalized toward the worst cost (``1``).
       The reflected aggregate ranges ``[0, 1+λ]``, so it is clamped to ``[0,1]``
       (B1) — the clamp is monotone and only bites on *terrible* terms (cost > 1,
       i.e. unstable **and** bad), so it is winner-preserving.

       Args:
           values: The per-image costs for one term (lower = better).
           stability_weight: λ — how hard cross-image spread is penalized.

       Returns:
           The clamped stability-penalized central tendency in ``[0,1]``. For a
           single value the IQR is ``0`` and the result is that value.
       """
       median, iqr = _median_iqr(values)
       return clamp01(median + stability_weight * iqr)
   ```

### Step 6.4 — run it green

```
uv run --extra tune pytest tests/unit/tune/test_evaluator.py -k robust_aggregate -v
```
Expected: PASS (4 tests).

### Step 6.5 — commit

```
git add src/phenotypic/tune/_evaluation/_evaluator.py tests/unit/tune/test_evaluator.py
git commit -m "$(cat <<'EOF'
refactor(tune): robust aggregate -> clamp01(median + λ·IQR) cost (B1)

Reflects the spread penalty into cost-space and clamps to [0,1] so the
bᵢ ∈ [0,1] invariant holds (winner-preserving).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — Evaluator worst-floor + failure flips (`0.0 → 1.0`)

`_WORST_TERM` (the per-image-exception pad) and `failure_score` (the won't-build / all-erroring floor) are the **worst** value — under cost that is `1.0`, not `0.0`. No separate `_FAILURE_COST` constant (the B1 clamp makes `1.0` a valid ceiling).

**Files:**
- Modify: `src/phenotypic/tune/_evaluation/_evaluator.py` (`_WORST_TERM` near `:23`; `failure_score` default near `:202`; docstrings)
- Modify: `tests/unit/tune/test_evaluator.py` (`test_evaluate_failure_assigns_failure_score`; the failure-floor / worst-pad asserts)

### Step 7.1 — write the failing test

In `tests/unit/tune/test_evaluator.py`, update `test_evaluate_failure_assigns_failure_score` (re-resolve near `:65`) so the **default** failure floor is the worst cost `1.0`:

```python
def test_evaluate_failure_assigns_worst_cost():
    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()
    # Default failure_score is now the worst cost (1.0): a candidate that won't
    # score floors to the worst, not the best.
    result = Evaluator().evaluate(base, _RaisingScorer(), {}, [img])
    assert result.score == pytest.approx(1.0)
    assert result.terms == {}
    assert result.n_images == 1
    assert result.failed is True
```

Add a worst-term-pad test that proves a per-image exception drags the aggregate **up** (toward worst cost):

```python
def test_per_image_exception_pads_worst_cost():
    from phenotypic.tune._scoring._scorer import Scorer

    class _OneGoodOneRaise(Scorer):
        """First call returns cost 0.0, the second raises."""

        _n: int = PrivateAttr(default=0)

        def _score_terms(self, image, measurements) -> dict[str, float]:
            self._n += 1
            if self._n == 1:
                return {"X": 0.0}
            raise RuntimeError("second image blew up")

    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()
    result = Evaluator(stability_weight=0.0).evaluate(
        base, _OneGoodOneRaise(), {}, [img, img]
    )
    # term X = aggregate of [0.0 (good), 1.0 (worst-term pad)] with λ=0
    # → median 0.5 (not 0.0); the failing plate drags the cost UP.
    assert result.terms["X"] == pytest.approx(0.5)
    assert result.failed is False  # not ALL images errored
```

### Step 7.2 — run it red

```
uv run --extra tune pytest tests/unit/tune/test_evaluator.py -k "worst_cost or worst_pad" -v
```
Expected: FAIL — `failure_score` defaults to `0.0` so the failed candidate scores `0.0` not `1.0`, and `_WORST_TERM=0.0` pads `[0.0, 0.0]` → median `0.0` not `0.5`.

### Step 7.3 — minimal implementation

In `src/phenotypic/tune/_evaluation/_evaluator.py`:

1. Flip `_WORST_TERM` (re-resolve near `:20`–`:23`):
   ```python
   #: The worst possible per-image term **cost** (lower-is-better objective
   #: ceiling). A per-image exception contributes this to every term so it
   #: honestly drags the aggregate toward the worst (robust-eval §10) rather than
   #: dodging a bad plate by crashing.
   _WORST_TERM = 1.0
   ```
2. Flip the `failure_score` default in the `Evaluator` model (re-resolve near `:202`):
   ```python
       failure_score: float = 1.0
   ```
   and update its docstring (re-resolve near `:181`–`:183`) from "the floor of the higher-is-better objective" to "the worst-cost ceiling assigned when a candidate fails to build, measure, or score (lower-is-better objective)".
3. Update the `EvaluationResult.score` / `terms` field docstrings (re-resolve near `:137`–`:141`) and the module docstring (re-resolve `:1`–`:7`) from "the scalar objective the optimizer maximizes" / "higher = better" to "the cost the optimizer **minimizes** (lower = better)". (Wording only — no logic.)

### Step 7.4 — run it green

```
uv run --extra tune pytest tests/unit/tune/test_evaluator.py -v
```
Expected: PASS for the new worst-cost / worst-pad tests. (The `_SequenceScorer` arithmetic tests from Task 1 still pass — they never hit the failure/worst-pad path.)

### Step 7.5 — commit

```
git add src/phenotypic/tune/_evaluation/_evaluator.py tests/unit/tune/test_evaluator.py
git commit -m "$(cat <<'EOF'
refactor(tune): worst-term + failure floors flip 0.0 -> 1.0 (cost ceiling)

A per-image exception / won't-build candidate floors to the worst cost
(1.0), dragging the aggregate up. No separate _FAILURE_COST (B1 clamp
makes 1.0 a valid ceiling).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 — `_is_suspicious` full reflection (pitfall #9 / OQ9)

Reflect **both** halves of the gaming flag and the `Count` default. Today: `score >= score_floor AND count <= count_floor`, default `terms.get("Count", 1.0)`. Under cost: a **low** finalized cost (great score) paired with a **high** `Count` cost (under-detection) → `score <= (1 − suspicious_score_floor) AND count_cost >= (1 − suspicious_count_floor)`, default `terms.get("Count", 0.0)` (missing Count = faithful = best cost).

**Files:**
- Modify: `src/phenotypic/tune/_evaluation/_evaluator.py` (`_is_suspicious` near `:104`–`:131`; the `evaluate` call site near `:330`–`:335` — the args stay the same; the floor fields stay the same names but the comparison reflects)
- Modify: `tests/unit/tune/test_evaluator.py` (add a `_is_suspicious` reflection test)

> **Reflection note:** the public `Evaluator.suspicious_score_floor` (default `0.7`) and `suspicious_count_floor` (default `0.3`) keep their **field names and defaults** (they are user-facing thresholds expressed against the *intuitive* "great score" / "under-detection" notions). The reflection `(1 − floor)` happens **inside** `_is_suspicious`, mapping the floors into cost-space. So `suspicious_score_floor=0.7` means "cost ≤ 0.3" and `suspicious_count_floor=0.3` means "Count cost ≥ 0.7".

### Step 8.1 — write the failing test

Add to `tests/unit/tune/test_evaluator.py`:

```python
from phenotypic.tune._evaluation._evaluator import _is_suspicious


def test_is_suspicious_flags_low_cost_with_high_count_cost():
    # Cost convention: a GREAT finalized cost (0.1 <= 1 - 0.7 = 0.3) paired with a
    # HIGH Count cost (0.8 >= 1 - 0.3 = 0.7, i.e. under-detection) is suspicious.
    assert _is_suspicious(
        0.1, {"Count": 0.8}, score_floor=0.7, count_floor=0.3
    ) is True


def test_is_suspicious_not_flagged_when_count_is_faithful():
    # Low Count cost (faithful detection) -> not suspicious even at a great score.
    assert _is_suspicious(
        0.1, {"Count": 0.2}, score_floor=0.7, count_floor=0.3
    ) is False


def test_is_suspicious_not_flagged_when_cost_is_mediocre():
    # A mediocre cost (0.6 > 0.3) is never flagged, regardless of Count.
    assert _is_suspicious(
        0.6, {"Count": 0.9}, score_floor=0.7, count_floor=0.3
    ) is False


def test_is_suspicious_missing_count_defaults_faithful():
    # A non-count objective: missing Count term defaults to 0.0 (faithful = best
    # cost) so it is NEVER flagged.
    assert _is_suspicious(
        0.0, {}, score_floor=0.7, count_floor=0.3
    ) is False
```

### Step 8.2 — run it red

```
uv run --extra tune pytest tests/unit/tune/test_evaluator.py -k is_suspicious -v
```
Expected: FAIL — the current `_is_suspicious` uses `score >= score_floor and count <= count_floor` with default `1.0`, so the cost-space cases are all wrong (e.g. `0.1 >= 0.7` is False → not flagged when it should be).

### Step 8.3 — minimal implementation

Rewrite `_is_suspicious` in `src/phenotypic/tune/_evaluation/_evaluator.py` (re-resolve near `:104`):

```python
def _is_suspicious(
    score: float,
    terms: Mapping[str, float],
    *,
    score_floor: float,
    count_floor: float,
) -> bool:
    """Flag the qc §5 "great cost on under-detection" gaming signature.

    A candidate is suspicious when a **low** finalized ``score`` (great cost) is
    paired with a **high** aggregated ``Count`` cost — the signature of a pipeline
    that scores well precisely *because* it under-detects (detecting fewer
    colonies dodges the spread/quality penalties). Read from already-computed
    aggregates; a heuristic review flag, not a hard rejection. The intuitive
    floors are mapped into cost-space here (``1 - floor``): a missing ``Count``
    term defaults to ``0.0`` (faithful = best cost) so a non-count objective is
    never flagged.

    Args:
        score: The finalized scalar cost (lower = better).
        terms: The robust-aggregated per-term costs; ``terms["Count"]`` is read.
        score_floor: The "great score" threshold expressed intuitively; the cost
            half fires when ``score <= 1 - score_floor``.
        count_floor: The "under-detection" threshold expressed intuitively; the
            cost half fires when ``terms["Count"] >= 1 - count_floor``.

    Returns:
        ``True`` when ``score <= (1 - score_floor)`` **and**
        ``terms["Count"] >= (1 - count_floor)``.
    """
    count_cost = float(terms.get("Count", 0.0))
    return score <= (1.0 - score_floor) and count_cost >= (1.0 - count_floor)
```

(The `evaluate` call site is unchanged — it already passes `final_score`, `aggregated`, `score_floor=self.suspicious_score_floor`, `count_floor=self.suspicious_count_floor`.)

### Step 8.4 — run it green

```
uv run --extra tune pytest tests/unit/tune/test_evaluator.py -k is_suspicious -v
```
Expected: PASS (4 tests).

### Step 8.5 — commit

```
git add src/phenotypic/tune/_evaluation/_evaluator.py tests/unit/tune/test_evaluator.py
git commit -m "$(cat <<'EOF'
refactor(tune): _is_suspicious reflects both halves + Count default into cost

Low finalized cost + high Count cost (under-detection) is the gaming
signature; floors map via (1 - floor); missing Count defaults to 0.0
(faithful). Pitfall #9 / OQ9.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9 — raise `_GAP_EPS` (1e-12 → 0.02)

The shared `_relative(x, central)` floors its denominator at `_GAP_EPS`. Under cost a great candidate's central tendency `≈ 0`, so the relative value explodes. Raise the floor to a meaningful `0.02` (defensive cap for the residual bad-end case after the `1 − cost` change in Tasks 10–11). One shared-helper constant fixes both call sites' denominator floor.

**Files:**
- Modify: `src/phenotypic/tune/_evaluation/_aggregate_math.py` (`_GAP_EPS` near `:22`; docstring)
- Modify (or create): `tests/unit/tune/test_aggregate_math.py` (pin the new floor)

### Step 9.1 — write the failing test

Create `tests/unit/tune/test_aggregate_math.py` (if it exists, append):

```python
from __future__ import annotations

import pytest

from phenotypic.tune._evaluation._aggregate_math import (
    _GAP_EPS,
    _median_iqr,
    _relative,
)


def test_gap_eps_is_a_meaningful_floor():
    # Raised 1e-12 -> 0.02 so a near-zero denominator cannot explode the ratio.
    assert _GAP_EPS == pytest.approx(0.02)


def test_relative_floors_tiny_denominator_at_gap_eps():
    # numerator 0.01 / max(0.0, 0.02) = 0.5, not a blow-up.
    assert _relative(0.01, 0.0) == pytest.approx(0.5)


def test_relative_uses_true_denominator_above_floor():
    # denominator 0.5 > floor → 0.1 / 0.5 = 0.2 (floor does not bite).
    assert _relative(0.1, 0.5) == pytest.approx(0.2)
```

### Step 9.2 — run it red

```
uv run --extra tune pytest tests/unit/tune/test_aggregate_math.py -v
```
Expected: FAIL — `_GAP_EPS` is `1e-12`, so `test_gap_eps_is_a_meaningful_floor` and `test_relative_floors_tiny_denominator_at_gap_eps` (which yields `0.01 / 1e-12 = 1e10`) fail.

### Step 9.3 — minimal implementation

In `src/phenotypic/tune/_evaluation/_aggregate_math.py`, change `_GAP_EPS` (re-resolve near `:19`–`:22`):

```python
#: Denominator floor for every relative ratio in the evaluation layer. Under the
#: cost convention a great candidate's central tendency is ≈ 0, so the relative
#: ratio is computed on the goodness-equivalent (``1 - cost``, see
#: ``_per_trial_dispersion`` / ``compute_generalization_gap``); this floor is the
#: defensive cap for the residual bad-end case (a few percent of the [0,1] scale,
#: small enough not to materially shift the gap for normal candidates).
_GAP_EPS: Final[float] = 0.02
```

### Step 9.4 — run it green

```
uv run --extra tune pytest tests/unit/tune/test_aggregate_math.py -v
```
Expected: PASS (3 tests).

### Step 9.5 — commit

```
git add src/phenotypic/tune/_evaluation/_aggregate_math.py tests/unit/tune/test_aggregate_math.py
git commit -m "$(cat <<'EOF'
refactor(tune): raise _GAP_EPS 1e-12 -> 0.02 (meaningful relative-ratio floor)

A near-zero cost central tendency would explode the relative ratio; the
new floor is the defensive cap (pitfall #2 / OQ4).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10 — `_per_trial_dispersion` computes relative IQR on `1 − median_cost`

The per-trial `gap` signal divides the IQR by the central tendency. Under cost, a good candidate's median `≈ 0` blows up the ratio. Fix (OQ4 part i): compute the relative quantity against the **goodness-equivalent** `1 − median_cost`, so a good candidate's denominator is `≈ 1`. Reflection-clean (it moves the singularity to the harmless bad end) and keeps the calibrated `GAP_FLAG_THRESHOLD=0.15` valid.

**Files:**
- Modify: `src/phenotypic/tune/_evaluation/_evaluator.py` (`_per_trial_dispersion` near `:68`–`:101`)
- Modify: `tests/unit/tune/test_evaluator.py` (add a dispersion-blowup-guard test)

### Step 10.1 — write the failing test

Add to `tests/unit/tune/test_evaluator.py`:

```python
from phenotypic.tune._evaluation._evaluator import _per_trial_dispersion


def test_per_trial_dispersion_does_not_blow_up_for_good_candidate():
    # A near-perfect candidate: median cost ≈ 0 with a small IQR. Computing the
    # relative IQR against (1 - median_cost ≈ 1) keeps it finite/small, not huge.
    scores = {"Count": [0.0, 0.0, 0.05, 0.05]}  # median 0.025, IQR 0.05
    gap = _per_trial_dispersion(scores, min_n=4)
    # 0.05 / (1 - 0.025) ≈ 0.0513 — finite and small (NOT 0.05 / 0.025 = 2.0).
    assert gap == pytest.approx(0.05 / (1.0 - 0.025))
    assert gap < 0.15  # below the calibrated flag threshold


def test_per_trial_dispersion_single_image_is_zero():
    assert _per_trial_dispersion({"Count": [0.0]}, min_n=4) == pytest.approx(0.0)


def test_per_trial_dispersion_below_min_n_is_none():
    assert _per_trial_dispersion({"Count": [0.0, 0.1]}, min_n=4) is None
```

### Step 10.2 — run it red

```
uv run --extra tune pytest tests/unit/tune/test_evaluator.py -k per_trial_dispersion -v
```
Expected: FAIL — `_per_trial_dispersion` currently calls `_relative(iqr, median)` so it returns `0.05 / max(0.025, 0.02) = 2.0`, not `0.05 / 0.975`.

### Step 10.3 — minimal implementation

Rewrite the tail of `_per_trial_dispersion` in `src/phenotypic/tune/_evaluation/_evaluator.py` (re-resolve near `:91`–`:101`). The guard logic (`not per_term_scores → None`, `n == 1 → 0.0`, `n < min_n → None`) is **unchanged**; only the final relative computation flips to the goodness-equivalent denominator:

```python
    if not per_term_scores:
        return None
    primary = next(iter(per_term_scores))
    values = per_term_scores[primary]
    n = len(values)
    if n == 1:
        return 0.0
    if n < min_n:
        return None
    median, iqr = _median_iqr(values)
    # Cost convention: a good candidate's median ≈ 0, so divide by the
    # goodness-equivalent (1 - median) to keep the relative IQR finite (the
    # singularity moves to the harmless bad end). Keeps GAP_FLAG_THRESHOLD valid.
    return _relative(iqr, 1.0 - median)
```

Update the function docstring's formula line (re-resolve near `:73`) from `(q75 - q25) / max(|median|, eps)` to `(q75 - q25) / max(1 - median, eps)` and note the cost-convention reasoning in one line.

### Step 10.4 — run it green

```
uv run --extra tune pytest tests/unit/tune/test_evaluator.py -k per_trial_dispersion -v
```
Expected: PASS (3 tests).

### Step 10.5 — commit

```
git add src/phenotypic/tune/_evaluation/_evaluator.py tests/unit/tune/test_evaluator.py
git commit -m "$(cat <<'EOF'
fix(tune): per-trial dispersion uses 1 - median_cost denominator (no blow-up)

A near-perfect candidate's median cost ≈ 0 would explode the relative IQR;
dividing by the goodness-equivalent (1 - median) keeps it finite (OQ4 part i).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11 — generalization gap: standard loss-space form (pitfall #1)

`compute_generalization_gap` keeps its formula + doctest (it is the textbook accuracy-space form `train − test`). The **caller** (`run_held_out`) must pass **goodness-equivalents** `1 − cal_cost` and `1 − heldout_cost`, so `cal_g − heldout_g = heldout_cost − cal_cost` = the standard loss-space gap `test − train` (positive = overfit). No bespoke sign flip, no margin re-derivation, and the doctest stays valid (read its inputs as goodness-equivalents). The relative drop is then divided by `cal_g = 1 − cal_cost` (a good winner's denominator ≈ 1 — the OQ4 fix), which the raised `_GAP_EPS` (Task 9) backstops.

**Files:**
- Modify: `src/phenotypic/tune/_evaluation/_generalization.py` (`run_held_out` — the `cal_score` capture near `:239`, the `compute_generalization_gap` call near `:277`; `compute_generalization_gap` itself is **unchanged** other than a docstring note)
- Modify: `tests/unit/tune/test_run_tuning_generalization.py` (the report-shape integration is unchanged; add a sign-direction assertion via a focused unit test on `run_held_out`)
- Create: `tests/unit/tune/test_generalization_gap_sign.py`

> **Sign direction (the bug this guards):** under cost, an **overfit** winner has a **higher** held-out cost than calibration (`heldout_cost > cal_cost`). Passing `1 − cal_cost` and `1 − heldout_cost` into the unchanged `absolute_drop = cal_g − heldout_g` yields `(1 − cal_cost) − (1 − heldout_cost) = heldout_cost − cal_cost > 0` → **flagged**. If you instead passed the raw costs, an overfit winner would give a *negative* drop and the gate would never fire — the silent failure pitfall #1 describes.

### Step 11.1 — write the failing test

Create `tests/unit/tune/test_generalization_gap_sign.py`:

```python
from __future__ import annotations

import pytest

from phenotypic.tune._evaluation._generalization import (
    compute_generalization_gap,
    run_held_out,
)


def test_compute_gap_doctest_inputs_are_goodness_equivalents():
    # The formula is unchanged (accuracy-space train - test). Read its inputs as
    # goodness-equivalents (1 - cost): cal_g 0.9, heldout_g 0.5 -> drop 0.4, flagged.
    rel, absolute, flagged = compute_generalization_gap(
        0.9, 0.5, rel_margin=0.15, abs_margin=0.05
    )
    assert (round(rel, 3), round(absolute, 3), flagged) == (0.444, 0.4, True)


class _FakeSplit:
    kind = "group"
    held_out = ["h1"]
    group_key = None
    within_group_caveat = False
    dataset_identity = "id-1"


class _FakeResult:
    def __init__(self, score: float) -> None:
        self.score = score


class _FakeEvaluator:
    def __init__(self, heldout_cost: float) -> None:
        self._heldout_cost = heldout_cost

    def evaluate(self, pipeline, scorer, params, images):
        return _FakeResult(self._heldout_cost)


class _FakeHeldOutCfg:
    gap_margin_relative = 0.15
    gap_margin_absolute = 0.05


class _FakeSpec:
    def __init__(self, heldout_cost: float) -> None:
        self.evaluator = _FakeEvaluator(heldout_cost)
        self.pipeline = object()
        self.scorer = object()
        self.held_out = _FakeHeldOutCfg()


class _FakeWinner:
    def __init__(self, cal_cost: float) -> None:
        self.params = {}
        self.score = cal_cost
        self.gap = 0.0


def test_overfit_winner_is_flagged_under_cost():
    # Overfit: held-out cost (0.6) >> calibration cost (0.1). The standard
    # loss-space gap heldout_cost - cal_cost = 0.5 > 0 must FLAG.
    report = run_held_out(
        _FakeSpec(heldout_cost=0.6),
        _FakeWinner(cal_cost=0.1),
        _FakeSplit(),
        {"h1": object()},
    )
    assert report.gap == pytest.approx(0.5)
    assert report.flagged is True


def test_good_generaliser_is_not_flagged_under_cost():
    # Good generaliser: held-out cost (0.12) ≈ calibration cost (0.10). Gap 0.02
    # is below the absolute margin -> NOT flagged (and not mis-flagged as overfit).
    report = run_held_out(
        _FakeSpec(heldout_cost=0.12),
        _FakeWinner(cal_cost=0.10),
        _FakeSplit(),
        {"h1": object()},
    )
    assert report.gap == pytest.approx(0.02)
    assert report.flagged is False
```

### Step 11.2 — run it red

```
uv run --extra tune pytest tests/unit/tune/test_generalization_gap_sign.py -v
```
Expected: FAIL — `run_held_out` currently passes the **raw** scores (`cal_score = winner.score`, `heldout_score = result.score`), so for the overfit case `absolute_drop = 0.1 − 0.6 = −0.5` → `gap == -0.5` and `flagged is False`. (`test_compute_gap_doctest_inputs_are_goodness_equivalents` passes already — the formula is unchanged — that one is the regression lock.)

### Step 11.3 — minimal implementation

In `src/phenotypic/tune/_evaluation/_generalization.py`, in `run_held_out` (re-resolve near `:236`–`:279`):

1. The data-poor branch (`split.kind == "none" or not held_out_images`) is **unchanged** — it reports `calibration_score=cal_score` (the raw cost) and never computes a gap.
2. In the **real held-out** branch, convert both scores to goodness-equivalents before the gap call. The current code reads:
   ```python
   heldout_score = float(result.score)
   relative_drop, absolute_drop, flagged = compute_generalization_gap(
       cal_score, heldout_score, rel_margin=rel_margin, abs_margin=abs_margin
   )
   ```
   Change to:
   ```python
   heldout_score = float(result.score)
   # Cost convention: pass goodness-equivalents (1 - cost) so the unchanged
   # accuracy-space formula (cal_g - heldout_g) equals the standard loss-space
   # gap (heldout_cost - cal_cost), positive = overfit. No bespoke sign flip.
   relative_drop, absolute_drop, flagged = compute_generalization_gap(
       1.0 - cal_score,
       1.0 - heldout_score,
       rel_margin=rel_margin,
       abs_margin=abs_margin,
   )
   ```
   The `GeneralizationReport(...)` construction is **unchanged**: `calibration_score=cal_score` and `heldout_score=heldout_score` still report the **raw costs** (the user-facing scores); only the gap math uses the goodness-equivalents. `gap=absolute_drop` is now `heldout_cost − cal_cost` (positive = overfit), which is correct.
3. Add a one-line note to `compute_generalization_gap`'s docstring (re-resolve near `:65`): the function is direction-agnostic; under the cost convention the caller passes goodness-equivalents (`1 − cost`) so the unchanged formula is the standard loss-space gap. Do **not** change the formula or the doctest.

### Step 11.4 — run it green

```
uv run --extra tune pytest tests/unit/tune/test_generalization_gap_sign.py -v
uv run --extra tune pytest --doctest-modules src/phenotypic/tune/_evaluation/_generalization.py -v
```
Expected: PASS (the sign tests + the unchanged doctest).

### Step 11.5 — commit

```
git add src/phenotypic/tune/_evaluation/_generalization.py tests/unit/tune/test_generalization_gap_sign.py
git commit -m "$(cat <<'EOF'
fix(tune): generalization gap is standard loss-space heldout_cost - cal_cost

run_held_out passes goodness-equivalents (1 - cost) into the unchanged
accuracy-space formula, so the gap is the textbook test - train (positive
= overfit). No bespoke sign flip; doctest stays valid (pitfall #1).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12 — phase boundary: type, lint, and full-suite sweep

After all code tasks, run type-checking + lint + the full tune unit suite. Fix any fallout (e.g. a stray `score_image` override left in another test double, an import of `_threshold_anchored` that needs no change, a doctest elsewhere).

**Files:** any touched by fixes (no new feature work).

### Step 12.1 — type + lint

```
uv run mypy src/phenotypic/tune
uv run ruff check --fix src/phenotypic/tune
```
Expected: mypy clean (re-resolve any `ClassVar[Sense]` annotation issue, any `clamp01` import); ruff auto-fixes formatting. Re-run `ruff check` to confirm zero remaining.

### Step 12.2 — full tune unit suite (note the expected reds)

```
uv run --extra tune pytest tests/unit/tune -v
```
Expected: **the Phase-1 unit tests pass**, but **end-to-end study/pruner/best-selection tests are RED** (they need Phase 2's `study.direction` flip). Triage each failure:
- A failure whose root cause is a cost-value assert this phase owns → **fix it here** (it was missed in the per-scorer task).
- A failure that runs `TuningEngine.optimize` / opens an Optuna study / asserts a winner / asserts the ASHA pruner pruned a specific trial → **leave it RED** and record it in the commit message as Phase-2-blocked. Examples to expect: `tests/unit/tune/test_optuna_pruning.py` (the pruner inversion — good trials at low cost, bad at high), any `test_engine*` / `test_run_tuning*` winner assertion, any `_study` best-selection test.

Document the exact list of intentionally-red tests in the final commit body so the Phase-2 worker knows what to turn green.

### Step 12.3 — commit any fixes

```
git add -A
git commit -m "$(cat <<'EOF'
chore(tune): phase-1 type/lint/test sweep; record Phase-2-blocked tests

mypy + ruff clean on src/phenotypic/tune. Phase-1 unit tests green.
End-to-end study/pruner/best-selection tests stay RED until Phase 2
flips study.direction (atomic cutover) — listed below:
<paste the intentionally-red test node ids here>

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1 done-criteria checklist

- [ ] `Scorer` is the `to_cost` template method (`_TERM_SENSE` ClassVar, `_score_terms` abstractmethod, `_term_anchor` → `None`, `score_image` wraps via `to_cost`).
- [ ] `QCScorer`, `SupervisedScorer`, `ReferenceFreeScorer` each rename `score_image` → `_score_terms`, declare `_TERM_SENSE = HIGHER_BETTER`, keep their internal folds (OQ1=A); their value tests + doctests assert **cost** (perfect = `0.0`, worst = `1.0`).
- [ ] `CompositeScorer` has a `_score_terms` stub that raises `NotImplementedError`; its doctest reflects cost children (`0.0`); `finalize`/`_geometric_mean` untouched (Phase 3 owns the combiner).
- [ ] `_robust_aggregate` = `clamp01(median + λ·IQR)`; `_WORST_TERM = 1.0`; `failure_score = 1.0`; no `_FAILURE_COST`.
- [ ] `_is_suspicious` reflects both halves + the `Count` default (`0.0`), floors mapped via `1 − floor`.
- [ ] `_GAP_EPS = 0.02`; `_per_trial_dispersion` divides by `1 − median`; `run_held_out` passes `1 − cost` into the unchanged `compute_generalization_gap` (gap = `heldout_cost − cal_cost`, positive = overfit; doctest still valid).
- [ ] All cost-value / gap-sign / clamp / suspicious **unit** tests pass standalone.
- [ ] `uv run mypy src/phenotypic/tune` and `uv run ruff check src/phenotypic/tune` are clean.
- [ ] **End-to-end study/pruner/best-selection tests remain RED** — this is EXPECTED. Phase 1 + Phase 2 are one atomic PR; the suite is fully green only after Phase 2 flips `study.direction` to `"minimize"`. The intentionally-red node ids are recorded in the Task 12 commit body.
