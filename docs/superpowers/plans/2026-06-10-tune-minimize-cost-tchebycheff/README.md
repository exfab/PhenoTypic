# Tune: minimize-cost + augmented Tchebycheff — Implementation Plan (index)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flip the `phenotypic.tune` module from "maximize a higher-is-better `[0,1]` score" to "minimize a bounded `[0,1]` **cost** (0 = perfect, 1 = worst)", and replace the geometric-mean composite with an **augmented Tchebycheff** scalarization.

**Architecture:** Orientation is a base-class template method on `Scorer` (a scorer emits its natural per-term values via `_score_terms`, declares a `_TERM_SENSE`, and the base wraps each term into cost via one shared `to_cost` helper). The Evaluator aggregates per-term cost (`median + λ·IQR`, clamped to `[0,1]`), each child reduces to a cost scalar, and the composite combines per-child cost scalars with augmented Tchebycheff (utopia `z* = −ε`, augmentation `ρ`, normalized to `[0,1]`). Optuna **minimizes**. Persistence cutover is a study-name bump (collision-impossible).

**Tech Stack:** Python 3.11+, pydantic v2, Optuna ≥4 (lazy-imported, behind the `tune` extra), numpy, pytest. Package manager/runner is **`uv`** only.

**Source spec:** [`docs/superpowers/specs/2026-06-09-tune-minimize-badness-augmented-tchebycheff-design.md`](../../specs/2026-06-09-tune-minimize-badness-augmented-tchebycheff-design.md) — read it before starting; this plan implements it.

---

## Phase files & execution order

Implement in order. **Phases 1 and 2 are a single atomic cutover** (they share one branch/PR and the test suite is only green once *both* land — the ASHA pruner reads `study.direction`, so the reported value and the study direction must flip together).

> **Operational consequence (Gap 3 — expected red suite mid-cutover):** during Phase 1, only each task's **own new/edited tests** are required to pass. The *full* `uv run --extra tune pytest tests/unit/tune` green-bar is **deferred to the end of Phase 2** — between the two phases the suite is legitimately RED (the reported objective is cost but Optuna still maximizes the old `"tune"` study). Do **not** treat that mid-cutover red suite as a task failure, and do **not** "fix" it by partially reverting Phase 1. The review gate that matters for Phase 1 is *its own tests green + mypy/ruff clean on touched files*; the suite-wide gate fires once at the Phase 2 boundary. The Phase-1↔2 code review agent (see *Execution & review protocol*) is dispatched once, after **both** phases land, not between them.

| File | Phase | Lands | Risk |
|------|-------|-------|------|
| [`phase-0-orientation-machinery.md`](phase-0-orientation-machinery.md) | 0 — `Sense`, `to_cost`, clamp helper (additive, dark) | own PR | Low |
| [`phase-1-evaluator-and-scorers.md`](phase-1-evaluator-and-scorers.md) | 1 — Evaluator cost math + scorer migration + gap + suspicious | **with Phase 2** | High |
| [`phase-2-direction-and-persistence.md`](phase-2-direction-and-persistence.md) | 2 — Optuna minimize + best-selection + study-name bump | **with Phase 1** | High |
| [`phase-3-tchebycheff-composite.md`](phase-3-tchebycheff-composite.md) | 3 — augmented Tchebycheff combiner + active set | own PR | Med |
| [`phase-4-pareto-screening-gui.md`](phase-4-pareto-screening-gui.md) | 4 — Pareto domination, screening freeze, GUI relabel | own PR | Med |
| [`phase-5-docs-and-tests.md`](phase-5-docs-and-tests.md) | 5 — explainer, CLAUDE.md, contrib guide, cross-phase regressions, **e2e minimize-cost smoke (Task 7)**, **whole-package final gate (Task 8)** | own PR | Low |

---

## Shared contract & conventions (ALL phases depend on this — do not diverge)

These names/signatures are fixed in Phase 0 and reused verbatim by later phases. If you change one, change it everywhere.

### The cost convention
Every per-term and per-child value the optimizer sees is a **cost** `∈ [0,1]`: `0` = perfect, `1` = worst, **lower is better, optimizer minimizes**. The word in code/docs/field-names is **"cost"** (never "badness", never "score" for the new quantity). The QC flag `_HIGHER_IS_BAD` is **unchanged** (`True` ⟺ the metric is a loss ⟺ `Sense.LOWER_BETTER`).

### `Sense` (new, `src/phenotypic/tune/_scoring/_orient.py`)
```python
from enum import Enum

class Sense(str, Enum):
    """Direction of a scorer's natural per-term values."""
    LOWER_BETTER = "lower_better"    # larger value = worse (a loss/divergence); maps to _HIGHER_IS_BAD=True
    HIGHER_BETTER = "higher_better"  # larger value = better (Dice, IoU, ICC, solidity)
```
A `str, Enum` so it is robust and readable (`Sense.LOWER_BETTER`). It is an internal `ClassVar` on `Scorer`, **never a serialized pydantic field**, so it needs no `Literal` partner and no alignment test.

### `to_cost` (new, `src/phenotypic/tune/_scoring/_orient.py`)
```python
import math
from phenotypic.tune._scoring._orient import Sense  # same module

def to_cost(value: float, *, sense: Sense, anchor: float | None = None) -> float:
    """Map a scorer's natural per-term value to cost in [0,1] (0 perfect, 1 worst).

    - anchor is None  → value is already bounded in [0,1]:
        LOWER_BETTER  → value           (already a [0,1] cost)
        HIGHER_BETTER → 1.0 - value     (complement a [0,1] goodness)
    - anchor is a positive float → value is an UNBOUNDED magnitude; map via the
      threshold-anchored transform exp(-ln2 * value / anchor) (1 at 0, .5 at anchor):
        LOWER_BETTER  → 1.0 - exp(...)  (a divergence: 0 cost at 0, →1 at ∞)
        HIGHER_BETTER → exp(...)
      A non-finite value (inf divergence) → cost 1.0 (worst).
    """
    if anchor is None:
        return value if sense is Sense.LOWER_BETTER else 1.0 - value
    if not math.isfinite(value):
        return 1.0
    goodness = math.exp(-math.log(2.0) * value / anchor)
    return (1.0 - goodness) if sense is Sense.LOWER_BETTER else goodness
```
For the **shipped roster** every term is already `[0,1]` (scorers keep their internal folds — OQ1=A), so `anchor` is always `None` and `to_cost` is identity/complement. The `anchor` branch is the contract for **future raw-loss scorers**; it is implemented and unit-tested in Phase 0 but has no live caller in v1.

### `clamp01` (new, `src/phenotypic/tune/_scoring/_orient.py`)
```python
def clamp01(value: float) -> float:
    """Clamp to [0,1]. Used on the robust-aggregated cost (B1: median + λ·IQR can reach ~1+λ)."""
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)
```

### `Scorer` template method (added AND activated in Phase 1, atomically with scorer migration — Phase 0 ships only `_orient.py` so it stays behavior-dark)
```python
class Scorer(BaseModel, ABC):
    _TERM_SENSE: ClassVar[Sense] = Sense.LOWER_BETTER   # cost-native default

    @abstractmethod
    def _score_terms(self, image, measurements) -> dict[str, float]:
        """This scorer's NATURAL per-term values (its own convention)."""

    def _term_anchor(self, term: str) -> float | None:
        """Anchor for an unbounded term, else None (already in [0,1])."""
        return None

    def score_image(self, image, measurements) -> dict[str, float]:
        return {
            term: to_cost(value, sense=self._TERM_SENSE, anchor=self._term_anchor(term))
            for term, value in self._score_terms(image, measurements).items()
        }
```
`CompositeScorer` overrides `score_image` (the merge), and provides a `_score_terms` stub that raises `NotImplementedError` (it is abstract on the base).

### Optimizer direction (Phase 2, `_strategies/_optuna_support.py`)
```python
_MINIMIZE: Final[str] = "minimize"   # replaces _MAXIMIZE
```
`objective_directions` returns `["minimize"] * n`; `study_objective_kwargs` returns `{"direction": _MINIMIZE}` / `{"directions": ["minimize"]*n}`. Best-selection becomes `min(valid, key=lambda t: t.score)`.

### Study-name bump (Phase 2, `_tune_cli/_run.py`)
```python
_STUDY_NAME: Final[str] = "tune_cost_v1"   # was "tune"
```
Every reader imports this constant — fix the two GUI desync sites (`gui/tune/_run_root.py` `_DEFAULT_STUDY_NAME`, `gui/tune/_winner.py` doctest).

### `CompositeBlend` (Phase 3, `tools_/typing_.py`)
```python
CompositeBlend = Literal["tchebycheff", "weighted_mean"]   # default "tchebycheff"
```
Serialized field value → `Literal` alias (no enum needed). Add `rho: float = 0.05` and `blend: CompositeBlend = "tchebycheff"` fields on `CompositeScorer`; `_UTOPIA_EPS: Final[float] = 1e-3`.

### `_GAP_EPS` (Phase 1, `_evaluation/_aggregate_math.py`)
Raise `1e-12 → 0.02` and compute the relative dispersion / generalization gap on the goodness-equivalent (`1 - cost`).

### Cross-phase symbol registry (pin these names — more than one phase shares them)
- **Legacy-study detection** — two real, importable helpers in ONE module, `_strategies/_optuna_support.py` (defined in Phase 2), plus the constant `_LEGACY_STUDY_NAME: Final[str] = "tune"`:
  - `is_legacy_study_name(study_name: str) -> bool` — **name-only** (`study_name == _LEGACY_STUDY_NAME`). Used by Phase 4's **read-only GUI monitor** (`gui/tune/_callbacks.py`), which must classify a run from its recorded `study_name` *without* connecting to a (legacy) study. Phase 4 imports it from `_optuna_support.py` — **not** from `_tune_cli/_run`.
  - `is_legacy_study_present(storage, *, study_name=_LEGACY_STUDY_NAME) -> bool` — **storage-probing** via `optuna.load_study` in `try/except KeyError` (absent → `False`; any other error → `False`, best-effort). Used by Phase 2's `OptunaStudyStore.__init__` guard (`_warn_if_legacy_study_present` calls it).
  Both live in the one module so the CLI and GUI cannot disagree about "legacy". A legacy study is **never reopened** — the `_STUDY_NAME` bump is the correctness mechanism; these helpers are UX (friendly message) only.
- **Active set** — `CompositeScorer.set_active_set(handles: tuple[str, ...]) -> None` storing a `_active_handles: tuple[str, ...] | None` PrivateAttr (Phase 3). The engine pins it once **after meta-validation**, before the trial loop; `finalize` reads it for BOTH the Tchebycheff `max` and the normalizer roster. `None` (never pinned) → use the in-call roster (keeps direct-`finalize` unit tests working).
- **Convention tag** — `study.set_user_attr("tune_convention", "minimize-cost-v1")`, stamped on every newly-created study (Phase 2).

---

## Cross-cutting invariants (assert / test these)
1. **Every term cost and every per-child cost scalar is in `[0,1]`** (B1 clamp). `_robust_aggregate` clamps; the composite asserts `0 ≤ bᵢ ≤ 1`.
2. **Generalization gap is the standard loss-space `heldout_cost − cal_cost`** (positive = overfit) — adopt the textbook definition, no custom sign flip.
3. **Reflection winner-equivalence:** for single-term / arithmetic-mean / Pareto paths the new minimize winner == the old maximize winner (winner-level, not bit-level; the composite is the one *intended* behavior change).
4. **One study-global active set** for both the Tchebycheff `max` numerator and the normalizer (children available study-wide); empty → cost `1.0`.
5. **No silent maximize:** a pre-cutover (`"tune"`, maximize) study is never reopened (name bump); a fresh `"tune_cost_v1"` study is stamped `tune_convention`.

---

## Dev environment & commands
```bash
# one-time, in this worktree:
uv sync --group dev --extra tune
# tune tests need the `tune` extra (Optuna):
uv run --extra tune pytest tests/unit/tune/<file>.py -v
# type + lint at phase boundaries:
uv run mypy src/phenotypic/tune
uv run ruff check --fix src/phenotypic/tune
```
Commit after every green task (the steps below show the exact `git add` / `git commit`). Run `mypy` + `ruff` once per phase before the final phase commit.

---

## Execution & review protocol (orchestration-level — runs *around* the phases)

The per-phase TDD steps and boundary gates (mypy/ruff/tune-suite) are the **inner** verification loop. Layered on top, the orchestrator runs these review gates:

### After every phase (per-phase review gate)
1. The phase's own done-criteria must be met (its tests green; mypy/ruff clean on the touched subtree).
2. **Dispatch a code-review agent** scoped to that phase's diff. Brief it with: the phase file, the spec section(s) it implements (see *Spec coverage map* below), and the *Cross-cutting invariants* list. It must flag: direction/sign errors, broken `[0,1]` invariants, missed reflection-equivalence, and any scorer that emits goodness instead of cost. Read its findings, fix blockers, re-run the phase tests, then proceed.
   - **Exception (Phase 1↔2 atomic):** run **one** code-review agent after **both** Phase 1 and Phase 2 land (not between them), because the suite is intentionally red mid-cutover (Gap 3). Brief it on the combined Phase 1+2 diff.

### After ALL phases land (final acceptance gate, in this order)
1. **Simplify agent** — one pass over the full cutover diff (all phases) to remove dead code (e.g. the removed `_geometric_mean`, any leftover goodness-era comments), de-duplicate the orientation logic, and tighten the new Tchebycheff/active-set code. Apply its fixes.
2. **Code-review agent** — a final whole-diff review (not per-phase) over the simplified tree. Fix blockers.
3. **Spec-adherence agent** — brief it with the **spec** (`docs/superpowers/specs/2026-06-09-…`) and the final tree; it walks the spec section-by-section (§4 cost convention, §5 Sense/template, §6 Tchebycheff/ε/ρ/active-set, §7 phase plan, §10 testing, §11 pitfalls/decisions) and reports any requirement that shipped differently or not at all. Reconcile every gap (fix the code or record an explicit, justified deviation).
4. After (1)–(3): re-run the **whole-package** final gate (below) one last time; only then is the cutover complete.

> Per the user's global orchestration rules: subagents are for executing this approved multi-step plan; brief each like a colleague (goal + why + file paths + the spec/invariant constraints); never trust a subagent's summary — read the changed files and re-run tests before marking a gate complete.

## Spec coverage map
- §5 Sense + orientation template method → Phase 0 + Phase 1 (scorer migration)
- §6.1–6.2 Tchebycheff formula + normalization, §6.3 active set → Phase 3
- §6.4 ε/ρ, §6.5 blend opt-out, §6.6 zero-parameter surface → Phase 3 (fields) + Phase 5 (docs)
- §7 Phase 1 Evaluator math + gap + suspicious + clamp → Phase 1
- §7 Phase 2 direction + persistence → Phase 2
- §7 Phase 4 Pareto/screening/GUI → Phase 4
- §7 Phase 5 docs → Phase 5
- §10 tests → distributed across each phase + the cross-phase regressions, the end-to-end minimize-cost smoke (Phase 5 Task 7), and the whole-package final gate (Phase 5 Task 8)
