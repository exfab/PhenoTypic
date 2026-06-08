# Tune Config Builder — Backend Prerequisites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the two backend (non-GUI) changes the Tune Config Builder depends on — `FloatRange.step` (quantized floats, wired into grid/random/Optuna) and `TuningSpec.phenotypic_version` (a provenance stamp) — so the later GUI plans have a complete, tested foundation.

**Architecture:** Pure changes to the `phenotypic.tune` module. `FloatRange` gains an optional `step` that mirrors `IntRange.step`: a quantizing `values()` method makes a stepped float grid-enumerable, and the three strategies (grid enumerate, random sample, Optuna suggest) each learn to use it, with the Optuna step↔log guard that already exists for `IntRange`. `TuningSpec` gains a `phenotypic_version` field stamped via `default_factory` and a load-time mismatch warning.

**Tech Stack:** Python, pydantic v2, Optuna, pytest, `uv` (sole runner — never bare `python`/`pip`).

**Plan series:** This is **Plan 1 of 4**. Downstream (separate plans, not in scope here):
- Plan 2 — GUI scaffold: hamburger Setup/Run/Monitor IA, Setup empty-state gate, `SANDBOX_TUNE_PRESETS_SUBDIR`, the scorer-param-form spike (Q9).
- Plan 3 — GUI Run/deploy: strategy/budget/compute form, image source from the shared source-image-root, deploy via the run-console runner, pre-flight + blocked-deploy validation.
- Plan 4 — GUI Monitor: run switcher, Local/SLURM live view, cancel, single-best + Pareto export.

Reference: the spec set in `docs/superpowers/spec/tune-config-builder/` (esp. docs 02 and 04) and the mockup `mockups/tune-config-builder.html`.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/phenotypic/tune/_search_space/_domains.py` | Domain value-models | Add `step` + `values()` to `FloatRange` |
| `src/phenotypic/tune/_strategies/_enumerate.py` | Grid Cartesian product | Enumerate a stepped `FloatRange` |
| `src/phenotypic/tune/_strategies/_random.py` | Random sampling | Sample a stepped `FloatRange` from its grid |
| `src/phenotypic/tune/_strategies/_optuna.py` | Optuna `suggest_*` mapping | Pass `step` to `suggest_float`; step↔log guard |
| `src/phenotypic/tune/_spec.py` | `TuningSpec` root model | Add `phenotypic_version` field + mismatch warning |
| `tests/unit/tune/test_domains.py` | Domain unit tests | `FloatRange.step` / `values()` cases |
| `tests/unit/tune/test_grid_enumerate.py` | Grid tests | Stepped-float enumeration case |
| `tests/unit/tune/test_strategies.py` | Random strategy tests | Stepped-float sampling case |
| `tests/unit/tune/test_optuna_strategy.py` | Optuna tests | Stepped-float suggest + guard |
| `tests/unit/tune/test_tuning_spec.py` | Spec round-trip tests | `phenotypic_version` stamp + warning |

---

## Task 1: `FloatRange.step` field and `values()`

**Files:**
- Modify: `src/phenotypic/tune/_search_space/_domains.py` (the `FloatRange` class, currently lines 77–95)
- Test: `tests/unit/tune/test_domains.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/tune/test_domains.py`:

```python
import pytest

from phenotypic.tune._search_space import FloatRange


def test_floatrange_step_defaults_to_none_continuous():
    fr = FloatRange(low=1.0, high=6.0)
    assert fr.step is None


def test_floatrange_stepped_values_are_clean_and_inclusive():
    fr = FloatRange(low=0.0, high=1.0, step=0.25)
    assert fr.values() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_floatrange_continuous_values_raises():
    fr = FloatRange(low=0.0, high=1.0)  # step is None
    with pytest.raises(ValueError, match="not enumerable"):
        fr.values()


def test_floatrange_nonpositive_step_rejected():
    with pytest.raises(ValueError, match="step"):
        FloatRange(low=0.0, high=1.0, step=0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/tune/test_domains.py -k floatrange_step -v`
Expected: FAIL — `FloatRange` has no `step` / `values` and accepts `step=0.0`.

- [ ] **Step 3: Implement the field, validator extension, and `values()`**

In `src/phenotypic/tune/_search_space/_domains.py`, replace the `FloatRange` class body (currently lines 77–95) with:

```python
class FloatRange(_DomainBase):
    """A float range ``[low, high]`` with an optional step / log scale.

    Args:
        low: Inclusive lower bound.
        high: Inclusive upper bound; must be ``>= low``.
        step: Quantization stride. ``None`` (default) means continuous; a
            positive value quantizes the range into a uniform grid that the
            grid and random strategies can enumerate (Optuna ``quniform``).
        log: Whether to sample on a logarithmic scale (default ``False``).
            Mutually exclusive with ``step`` at suggest time (Optuna forbids
            ``suggest_float(step=..., log=True)``); the Optuna strategy drops
            the step under log and logs a note.
    """

    kind: Literal["float_range"] = "float_range"
    low: float
    high: float
    step: float | None = None
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "FloatRange":
        if self.high < self.low:
            raise ValueError(f"FloatRange high ({self.high}) < low ({self.low})")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"FloatRange step ({self.step}) must be > 0")
        return self

    def values(self) -> list[float]:
        """The quantized grid in ``[low, high]`` stepped by :attr:`step`.

        Uses ``linspace``-style generation (``low + i*step``) rather than
        ``numpy.arange`` to avoid float-accumulation endpoint drift. Only
        defined for a stepped (non-continuous) range.

        Returns:
            The stepped floats from ``low`` to ``high`` inclusive.

        Raises:
            ValueError: When the range is continuous (``step is None``).
        """
        if self.step is None:
            raise ValueError(
                "continuous FloatRange (step=None) is not enumerable"
            )
        num = round((self.high - self.low) / self.step) + 1
        return [round(self.low + i * self.step, 12) for i in range(num)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/tune/test_domains.py -k floatrange_step -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full domains suite to catch serialization snapshots**

Run: `uv run pytest tests/unit/tune/test_domains.py -v`
Expected: PASS. If any test asserts an exact serialized `FloatRange` dict, add `"step": None` to the expected payload (the new field now serializes). Fix such assertions inline, then re-run.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/tune/_search_space/_domains.py tests/unit/tune/test_domains.py
git commit -m "feat(tune): add optional FloatRange.step with quantized values()"
```

---

## Task 2: Enumerate a stepped `FloatRange` in grid search

**Files:**
- Modify: `src/phenotypic/tune/_strategies/_enumerate.py` (the import line and `grid_values`)
- Test: `tests/unit/tune/test_grid_enumerate.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/tune/test_grid_enumerate.py`:

```python
import pytest

from phenotypic.tune._search_space import FloatRange
from phenotypic.tune._strategies._enumerate import grid_values


def test_grid_values_stepped_floatrange_enumerates():
    assert grid_values(FloatRange(low=0.0, high=1.0, step=0.5)) == [0.0, 0.5, 1.0]


def test_grid_values_continuous_floatrange_still_raises():
    with pytest.raises(ValueError, match="continuous FloatRange"):
        grid_values(FloatRange(low=0.0, high=1.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/tune/test_grid_enumerate.py -k floatrange -v`
Expected: FAIL — `grid_values` raises for *any* `FloatRange`, including a stepped one.

- [ ] **Step 3: Implement the stepped-float branch**

In `src/phenotypic/tune/_strategies/_enumerate.py`:

Change the import (currently `from .._search_space import Categorical, Domain, Fixed, IntRange, SearchSpace`) to include `FloatRange`:

```python
from .._search_space import (
    Categorical,
    Domain,
    Fixed,
    FloatRange,
    IntRange,
    SearchSpace,
)
```

Replace the `grid_values` function body with:

```python
def grid_values(domain: Domain) -> list[Any]:
    """The discrete grid values for a domain.

    A continuous ``FloatRange`` (``step is None``) is not enumerable; a stepped
    ``FloatRange`` quantizes via :meth:`FloatRange.values`.
    """
    if isinstance(domain, Categorical):
        return list(domain.choices)
    if isinstance(domain, IntRange):
        return domain.values()
    if isinstance(domain, FloatRange) and domain.step is not None:
        return domain.values()
    if isinstance(domain, Fixed):
        return [domain.value]
    raise ValueError(
        "GridStrategy cannot enumerate a continuous FloatRange; give it a step, "
        "or use Categorical / IntRange, or a non-grid strategy."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/tune/test_grid_enumerate.py -v`
Expected: PASS (existing tests + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_strategies/_enumerate.py tests/unit/tune/test_grid_enumerate.py
git commit -m "feat(tune): grid-enumerate a stepped FloatRange"
```

---

## Task 3: Sample a stepped `FloatRange` in random search

**Files:**
- Modify: `src/phenotypic/tune/_strategies/_random.py` (the `_sample_domain` `FloatRange` branch)
- Test: `tests/unit/tune/test_strategies.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/tune/test_strategies.py`:

```python
from phenotypic.tune._search_space import (
    FloatRange,
    Knob,
    SearchSpace,
)
from phenotypic.tune._search_space._targets import Param
from phenotypic.tune._strategies._random import RandomStrategy


def test_random_samples_stepped_float_on_grid():
    knob = Knob(
        target=Param(op=0, field="sigma"),
        domain=FloatRange(low=0.0, high=1.0, step=0.25),
    )
    strat = RandomStrategy(SearchSpace(knobs=(knob,)), n_trials=50, seed=0)
    seen = set()
    while not strat.is_exhausted():
        params, _ = strat.suggest()
        seen.update(params.values())
    assert seen.issubset({0.0, 0.25, 0.5, 0.75, 1.0})
    assert len(seen) > 1  # actually varies across the grid
```

(The `Knob` target uses `Param`; the knob's `key` is derived from the target, matching how inferred spaces build knobs.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tune/test_strategies.py -k stepped_float -v`
Expected: FAIL — random samples continuous `uniform`, producing values off the grid.

- [ ] **Step 3: Implement the stepped-float branch**

In `src/phenotypic/tune/_strategies/_random.py`, replace the `FloatRange` branch inside `_sample_domain` (currently the `if isinstance(domain, FloatRange):` block) with:

```python
        if isinstance(domain, FloatRange):
            if domain.step is not None:
                return self._rng.choice(domain.values())
            if domain.log:
                lo, hi = math.log(domain.low), math.log(domain.high)
                return math.exp(self._rng.uniform(lo, hi))
            return self._rng.uniform(domain.low, domain.high)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/tune/test_strategies.py -k stepped_float -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_strategies/_random.py tests/unit/tune/test_strategies.py
git commit -m "feat(tune): random-sample a stepped FloatRange from its grid"
```

---

## Task 4: Pass `step` to Optuna `suggest_float` with the step↔log guard

**Files:**
- Modify: `src/phenotypic/tune/_strategies/_optuna.py` (the `FloatRange` branch of `_suggest_domain`, currently lines 348–349)
- Test: `tests/unit/tune/test_optuna_strategy.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/tune/test_optuna_strategy.py`:

```python
import optuna

from phenotypic.tune._search_space import FloatRange
from phenotypic.tune._strategies._optuna import OptunaStrategy


def _fresh_trial() -> "optuna.trial.Trial":
    return optuna.create_study().ask()


def test_optuna_stepped_float_suggests_on_grid():
    # Bypass __init__: _suggest_domain only uses the module logger + the domain.
    strat = OptunaStrategy.__new__(OptunaStrategy)
    val = strat._suggest_domain(
        _fresh_trial(), "x", FloatRange(low=0.0, high=1.0, step=0.25)
    )
    assert val in {0.0, 0.25, 0.5, 0.75, 1.0}


def test_optuna_stepped_float_with_log_warns_and_drops_step(caplog):
    strat = OptunaStrategy.__new__(OptunaStrategy)
    with caplog.at_level("WARNING"):
        strat._suggest_domain(
            _fresh_trial(), "x", FloatRange(low=0.1, high=10.0, step=0.5, log=True)
        )
    assert "unsupported by Optuna" in caplog.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/tune/test_optuna_strategy.py -k stepped_float -v`
Expected: FAIL — `suggest_float` is called without `step`, so a stepped value isn't honored and no guard warning is emitted.

- [ ] **Step 3: Implement the step pass-through + guard**

In `src/phenotypic/tune/_strategies/_optuna.py`, replace the `FloatRange` branch of `_suggest_domain` (currently lines 348–349) with:

```python
        if isinstance(domain, FloatRange):
            step, log = domain.step, domain.log
            if step is not None and log:
                # Optuna forbids suggest_float(step=..., log=True): drop the
                # step under log scale, mirroring the IntRange guard above.
                _logger.warning(
                    "FloatRange %r: step=%s with log=True is unsupported by "
                    "Optuna; normalizing to continuous (log scale).",
                    key,
                    step,
                )
                step = None
            return trial.suggest_float(
                key, domain.low, domain.high, step=step, log=log
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/tune/test_optuna_strategy.py -k stepped_float -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tune/_strategies/_optuna.py tests/unit/tune/test_optuna_strategy.py
git commit -m "feat(tune): pass FloatRange.step to Optuna with step/log guard"
```

---

## Task 5: `TuningSpec.phenotypic_version` provenance stamp

**Files:**
- Modify: `src/phenotypic/tune/_spec.py` (imports + the `TuningSpec` class)
- Test: `tests/unit/tune/test_tuning_spec.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/tune/test_tuning_spec.py` (reuse the module's existing helper for building a minimal valid spec; this example assumes a `make_spec()`-style fixture — if the module names it differently, use that builder):

```python
import importlib.metadata
import json

from phenotypic.tune._spec import TuningSpec, _running_phenotypic_version


def test_fresh_spec_is_stamped_with_running_version(minimal_spec: TuningSpec):
    assert minimal_spec.phenotypic_version == _running_phenotypic_version()


def test_loaded_spec_preserves_its_stamp_and_warns_on_mismatch(minimal_spec):
    payload = json.loads(minimal_spec.to_json())
    payload["phenotypic_version"] = "0.0.1-ancient"
    with pytest.warns(UserWarning, match="authored with phenotypic 0.0.1-ancient"):
        loaded = TuningSpec.model_validate_json(json.dumps(payload))
    assert loaded.phenotypic_version == "0.0.1-ancient"  # provenance preserved


def test_legacy_spec_without_stamp_still_loads(minimal_spec):
    payload = json.loads(minimal_spec.to_json())
    payload.pop("phenotypic_version", None)
    loaded = TuningSpec.model_validate_json(json.dumps(payload))  # no raise
    assert loaded.phenotypic_version == _running_phenotypic_version()
```

If `tests/unit/tune/test_tuning_spec.py` has no `minimal_spec` fixture, add one at the top of the file built from `load_synth_yeast_plate()` + a one-op `ImagePipeline` + a `SearchSpace` with a single knob + the default `QCScorer`/`Evaluator`/`GridConfig`/`Budget`, mirroring the fixture in `tests/unit/tune/fixtures/phase1_tuning_spec.json`. (Look at the existing tests in this file for the exact builder they already use and prefer reusing it.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/tune/test_tuning_spec.py -k version -v`
Expected: FAIL — `_running_phenotypic_version` and the `phenotypic_version` field do not exist.

- [ ] **Step 3: Implement the field, factory, and mismatch warning**

In `src/phenotypic/tune/_spec.py`:

Add to the top-level imports (after the existing stdlib imports `import difflib`, `import json`):

```python
import importlib.metadata
import warnings
from typing import Any
```

(`Any` is already imported via `from typing import Any, Optional, TypeAlias` — keep that line; do not duplicate.)

Add `Field` to the pydantic import block (it currently imports `BaseModel, ConfigDict, TypeAdapter, field_serializer, field_validator, model_validator`):

```python
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)
```

Add a module-level helper (place it above `class TuningSpec`):

```python
def _running_phenotypic_version() -> str:
    """The installed ``phenotypic`` version, or ``"unknown"`` if undiscoverable."""
    try:
        return importlib.metadata.version("phenotypic")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"
```

Add the field to `TuningSpec` (immediately after `held_out: HeldOutConfig = HeldOutConfig()`):

```python
    phenotypic_version: str = Field(default_factory=_running_phenotypic_version)
```

Add a `mode="before"` model validator to `TuningSpec` (place it near the other validators, e.g. just above `_coerce_pipeline`):

```python
    @model_validator(mode="before")
    @classmethod
    def _warn_version_mismatch(cls, data: Any) -> Any:
        """Warn (advisory) when a loaded spec was authored on a different build.

        Only fires for dict-shaped input carrying an explicit
        ``phenotypic_version``; fresh in-memory construction (no key) is
        stamped by the field's ``default_factory`` and never warns.
        """
        if isinstance(data, dict):
            stamped = data.get("phenotypic_version")
            running = _running_phenotypic_version()
            if stamped is not None and stamped != running:
                warnings.warn(
                    f"tuning spec was authored with phenotypic {stamped}; "
                    f"running {running}",
                    stacklevel=2,
                )
        return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/tune/test_tuning_spec.py -k version -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full tuning-spec suite to catch round-trip snapshots**

Run: `uv run pytest tests/unit/tune/test_tuning_spec.py tests/unit/tune/test_tune_spec_marker.py -v`
Expected: PASS. The serialized spec now contains `phenotypic_version`; if any test asserts an exact full-spec JSON/dict, add the key (or assert with the field excluded). Fix inline and re-run.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/tune/_spec.py tests/unit/tune/test_tuning_spec.py
git commit -m "feat(tune): stamp TuningSpec with phenotypic_version + load warning"
```

---

## Task 6: Full-suite verification + lint/type gate

**Files:** none (verification only)

- [ ] **Step 1: Run the whole tune unit suite**

Run: `uv run pytest tests/unit/tune -q`
Expected: PASS. Investigate any failure caused by the two new serialized fields (`FloatRange.step`, `phenotypic_version`) appearing in snapshot/round-trip assertions; update those expectations and re-run until green.

- [ ] **Step 2: Type-check the changed module**

Run: `uv run mypy src/phenotypic/tune`
Expected: no new errors. (`step: float | None` and `phenotypic_version: str` are simple annotations; `values()` returns `list[float]`.)

- [ ] **Step 3: Lint/format**

Run: `uv run ruff check --fix src/phenotypic/tune tests/unit/tune`
Expected: clean (auto-fixes applied).

- [ ] **Step 4: Commit any fixups**

```bash
git add -A src/phenotypic/tune tests/unit/tune
git commit -m "test(tune): update snapshots for FloatRange.step + phenotypic_version" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage (doc 02 / doc 04 prerequisites):**
- `FloatRange.step` field + validator → Task 1. ✓
- `FloatRange.values()` linspace-style (not arange) → Task 1, Step 3. ✓
- Stepped float grid-enumerable → Task 2. ✓
- Random sampling honors step → Task 3. ✓
- Optuna `suggest_float(step=)` + step↔log guard → Task 4. ✓
- `TuningSpec.phenotypic_version` via `default_factory` → Task 5. ✓
- Load-time mismatch warning, provenance preserved, legacy loads → Task 5. ✓
- `TuningSpec.to_json` already exists (no new wrapper) → confirmed in doc 04; not a task. ✓
- `SANDBOX_TUNE_PRESETS_SUBDIR` → **GUI constant, deferred to Plan 2** (correctly out of scope here). ✓

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N" — every code step shows complete code. The only soft spot is the `minimal_spec` fixture in Task 5, which is gated with explicit instructions to reuse the file's existing spec builder and a concrete fallback recipe; this is a deliberate "reuse what's there" pointer, not a placeholder.

**Type consistency:** `step: float | None` used identically in Tasks 1–4. `values()` returns `list[float]` and is called by `grid_values` (Task 2) and `_sample_domain` (Task 3). `_running_phenotypic_version()` defined once (Task 5, Step 3) and referenced by the test (Task 5, Step 1) and the validator. `phenotypic_version` field name consistent across field, validator, and tests.

---

## Execution Handoff

This is Plan 1 of 4. After it is green, Plans 2–4 (GUI) build on the now-tested
`FloatRange.step` and `phenotypic_version`. Recommended: execute this plan, then
return to writing-plans for Plan 2.
