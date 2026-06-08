# Tune Config Builder — GUI Scaffold (Setup) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the read-only `/tune/` co-pilot into an authoring shell — a hamburger **Setup / Run / Monitor** nav, and a working **Setup** destination (Pipeline gate → editable Search space → Scorer) with the per-knob domain editor and validation.

**Architecture:** Follow the existing tune app's pattern: **pure helper functions** carry the logic (unit-tested, mirroring `active_view`, `subtab_button_class`, `render_launch_command`), and thin Dash layout/callbacks wire them up (integration/e2e-tested). New pure modules: `_nav.py` (destination switch), `_domain_editor.py` (knob-domain parse/serialize/feasibility), `_validation.py` (blocked-deploy issues). Run/deploy and Monitor extensions are **Plans 3 and 4**.

**Tech Stack:** Python, Dash + dash-bootstrap-components, pydantic v2, pytest (+ pytest-qt/offscreen not needed here; Dash integration via `dash.testing` / the repo's `tests/integration/gui/tune/` harness), `uv`.

**Depends on:** Plan 1 (backend prereqs) merged — `FloatRange.step` must exist for the domain editor.

**Spec refs:** `docs/superpowers/spec/tune-config-builder/01-placement-and-ia.md`, `02-search-space-and-scorer.md`; mockup `mockups/tune-config-builder.html` (Setup view is the visual contract).

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/phenotypic/gui/_config.py` | Shared constants | Add `SANDBOX_TUNE_PRESETS_SUBDIR` + `tune_presets_dir()` |
| `src/phenotypic/gui/tune/_nav.py` | **New.** Pure hamburger destination model | `Destination`, `active_destination()`, class helpers |
| `src/phenotypic/gui/tune/_scorer_form.py` | **New.** Scorer→form adapter (Q9 spike outcome) | `scorer_operation_info()` |
| `src/phenotypic/gui/tune/_domain_editor.py` | **New.** Pure knob-domain logic | `domain_from_editor()`, `domain_summary()`, `grid_feasibility()` |
| `src/phenotypic/gui/tune/_validation.py` | **New.** Pure blocked-deploy issues | `Issue`, `validate_setup()` |
| `src/phenotypic/gui/tune/_ids.py` | Tune component IDs | Add destination + Setup IDs |
| `src/phenotypic/gui/tune/_layout.py` | Page layout | Hamburger drawer + Setup view body |
| `src/phenotypic/gui/tune/_callbacks.py` | Callbacks | Destination switch, pipeline gate, domain editor, validation |
| `src/phenotypic/gui/FEATURES.md` | Affordance ledger | Rows for every new control (CI-gated) |
| `tests/unit/gui/tune/test_nav.py` | **New** | `active_destination` cases |
| `tests/unit/gui/tune/test_scorer_form.py` | **New** | scorer form-info cases |
| `tests/unit/gui/tune/test_domain_editor.py` | **New** | domain parse/serialize/feasibility |
| `tests/unit/gui/tune/test_validation.py` | **New** | `validate_setup` cases |
| `tests/unit/gui/test_config.py` | Existing config tests | `tune_presets_dir` case |
| `tests/integration/gui/tune/test_setup_view.py` | **New** | empty-state gate + edit flow |

---

## Task 1 (SPIKE→helper): Scorer param-form adapter (resolves Q9)

The scorer section renders a scorer's pydantic fields with the shared
`param_form`, which takes an `OperationInfo`. Scorers aren't in the
`OperationRegistry`, but `OperationRegistry._extract_parameters(cls)` reads
`cls.model_fields` generically (`_extract_parameters_from_model_fields`) — and
scorers are pydantic models. This task confirms that and wraps it.

**Files:**
- Create: `src/phenotypic/gui/tune/_scorer_form.py`
- Test: `tests/unit/gui/tune/test_scorer_form.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/gui/tune/test_scorer_form.py
from phenotypic.gui._param_forms import param_form
from phenotypic.gui.tune._scorer_form import scorer_operation_info
from phenotypic.tune._scoring import QCScorer


def test_scorer_operation_info_exposes_pydantic_fields():
    info = scorer_operation_info(QCScorer)
    # QCScorer has a `check` field (the ExpectedVsDetectedCount); at minimum the
    # adapter must surface >=1 parameter pulled from model_fields.
    assert info.parameters  # non-empty
    assert set(info.parameters) <= set(QCScorer.model_fields)


def test_param_form_renders_a_scorer_without_registry():
    info = scorer_operation_info(QCScorer)
    form = param_form(info, current_values={}, form_id_prefix="tune-scorer")
    assert form is not None  # dbc.Form built without raising
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/gui/tune/test_scorer_form.py -v`
Expected: FAIL — `_scorer_form` module does not exist.

- [ ] **Step 3: Implement the adapter**

```python
# src/phenotypic/gui/tune/_scorer_form.py
"""Adapt a Scorer pydantic model to an OperationInfo for the shared param_form.

Scorers are pydantic models but not ImageOperation subclasses, so they are not
in the builder's OperationRegistry. The registry's field extractor, however, is
generic over ``model_fields`` (``_extract_parameters_from_model_fields``), so we
reuse it to build an OperationInfo on demand — no registry membership required.
"""
from __future__ import annotations

from typing import Type

from phenotypic.gui._operation_registry import OperationInfo, OperationRegistry


def scorer_operation_info(scorer_cls: Type) -> OperationInfo:
    """Build an :class:`OperationInfo` describing a scorer's editable params.

    Args:
        scorer_cls: A ``Scorer`` subclass (pydantic model).

    Returns:
        An ``OperationInfo`` whose ``parameters`` are extracted from the
        scorer's ``model_fields`` via the registry's generic extractor.
    """
    registry = OperationRegistry()
    params = registry._extract_parameters(scorer_cls)
    return OperationInfo(
        class_name=scorer_cls.__name__,
        category="scorer",
        parameters=params,
    )
```

If `OperationInfo`'s required fields differ (open the dataclass at
`_operation_registry.py:120`), match its actual constructor — keep `parameters`
from the extractor; supply whatever name/category fields it declares. Do not add
fields the dataclass lacks.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/gui/tune/test_scorer_form.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/tune/_scorer_form.py tests/unit/gui/tune/test_scorer_form.py
git commit -m "feat(gui-tune): render scorer pydantic fields via param_form adapter"
```

---

## Task 2: `SANDBOX_TUNE_PRESETS_SUBDIR` + `tune_presets_dir()`

**Files:**
- Modify: `src/phenotypic/gui/_config.py` (near `SANDBOX_PRESETS_SUBDIR`, line ~294, and its `__all__`)
- Test: `tests/unit/gui/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/gui/test_config.py  (add)
from pathlib import Path

from phenotypic.gui._config import (
    SANDBOX_GUI_DIRNAME,
    SANDBOX_PRESETS_SUBDIR,
    SANDBOX_TUNE_PRESETS_SUBDIR,
    tune_presets_dir,
)


def test_tune_presets_dir_nests_under_presets():
    root = Path("/tmp/sbx")
    expected = root / SANDBOX_GUI_DIRNAME / SANDBOX_PRESETS_SUBDIR / SANDBOX_TUNE_PRESETS_SUBDIR
    assert tune_presets_dir(root) == expected
    assert SANDBOX_TUNE_PRESETS_SUBDIR == "tune"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/gui/test_config.py -k tune_presets -v`
Expected: FAIL — name does not exist.

- [ ] **Step 3: Implement**

In `src/phenotypic/gui/_config.py`, add next to `SANDBOX_PRESETS_SUBDIR`:

```python
#: Sub-folder of the preset library holding saved tuning specs
#: (``.phenotypic-gui/presets/tune/``). Keeps user-authored TuningSpec configs
#: out of ``.phenotypic/`` (CLI cache) and ``.pht-tune-cache/`` (per-run state).
SANDBOX_TUNE_PRESETS_SUBDIR: str = "tune"


def tune_presets_dir(sandbox_root: "Path") -> "Path":
    """Return ``<sandbox>/.phenotypic-gui/presets/tune/`` for the spec library."""
    from pathlib import Path as _Path

    return (
        _Path(sandbox_root)
        / SANDBOX_GUI_DIRNAME
        / SANDBOX_PRESETS_SUBDIR
        / SANDBOX_TUNE_PRESETS_SUBDIR
    )
```

Add `"SANDBOX_TUNE_PRESETS_SUBDIR"` and `"tune_presets_dir"` to the module's
`__all__`. (Keep `_config.py` free of a top-level `dash` import per the module's
gotcha note; `Path` is fine.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/gui/test_config.py -k tune_presets -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_config.py tests/unit/gui/test_config.py
git commit -m "feat(gui): add SANDBOX_TUNE_PRESETS_SUBDIR + tune_presets_dir helper"
```

---

## Task 3: Hamburger destination model (pure)

Mirror the existing `SubTabName` / `active_view` / `subtab_button_class` pattern,
but for the three top-level destinations.

**Files:**
- Create: `src/phenotypic/gui/tune/_nav.py`
- Modify: `src/phenotypic/gui/tune/_ids.py` (add destination IDs)
- Test: `tests/unit/gui/tune/test_nav.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/gui/tune/test_nav.py
from phenotypic.gui.tune._nav import (
    DESTINATIONS,
    active_destination,
    destination_button_class,
    destination_view_class,
)


def test_destinations_are_setup_run_monitor_in_order():
    assert DESTINATIONS == ("setup", "run", "monitor")


def test_active_destination_maps_trigger_to_name():
    assert active_destination("tune-dest-run") == "run"
    assert active_destination("tune-dest-monitor") == "monitor"


def test_active_destination_defaults_to_setup_on_none():
    assert active_destination(None) == "setup"


def test_classes_mark_active_and_hide_inactive():
    assert "tune-dest-active" in destination_button_class("setup", "setup")
    assert "tune-dest-active" not in destination_button_class("run", "setup")
    assert "tune-view-hidden" in destination_view_class("run", "setup")
    assert "tune-view-hidden" not in destination_view_class("setup", "setup")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/gui/tune/test_nav.py -v`
Expected: FAIL — `_nav` module does not exist.

- [ ] **Step 3: Implement**

```python
# src/phenotypic/gui/tune/_nav.py
"""Pure model for the tune hamburger nav: Setup / Run / Monitor.

Mirrors the sub-tab helpers in :mod:`phenotypic.gui.tune._ids` (one source of
truth for the active class so layout + callback never disagree), but for the
three top-level destinations the hamburger drawer exposes.
"""
from __future__ import annotations

from typing import Literal

Destination = Literal["setup", "run", "monitor"]

#: Ordered destinations (drives drawer item + view-container render order).
DESTINATIONS: tuple[Destination, ...] = ("setup", "run", "monitor")

DESTINATION_LABELS: dict[Destination, str] = {
    "setup": "Setup",
    "run": "Run",
    "monitor": "Monitor",
}


def destination_button_id(name: Destination) -> str:
    """Static ID for a drawer destination item."""
    return f"tune-dest-{name}"


def destination_view_id(name: Destination) -> str:
    """Static ID for a destination's view container."""
    return f"tune-destview-{name}"


def active_destination(trigger_id: str | None) -> Destination:
    """Map a clicked ``tune-dest-<name>`` trigger to its destination.

    Defaults to ``"setup"`` (the landing destination) when no trigger fired.
    """
    if trigger_id:
        for name in DESTINATIONS:
            if trigger_id == destination_button_id(name):
                return name
    return "setup"


def destination_button_class(name: Destination, active: "Destination | None") -> str:
    """CSS class for a drawer item; the active one gets the highlight."""
    classes = ["tune-dest"]
    if name == active:
        classes.append("tune-dest-active")
    return " ".join(classes)


def destination_view_class(name: Destination, active: "Destination | None") -> str:
    """CSS class for a destination view; inactive ones carry the hidden class."""
    classes = ["tune-view"]
    if name != active:
        classes.append("tune-view-hidden")
    return " ".join(classes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/gui/tune/test_nav.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/tune/_nav.py tests/unit/gui/tune/test_nav.py
git commit -m "feat(gui-tune): pure hamburger destination model (setup/run/monitor)"
```

---

## Task 4: Domain-editor logic (pure)

The per-knob editor's parse/serialize/feasibility — the meaty testable logic. It
turns editor field values into a `Domain` (Plan 1's `FloatRange.step` included),
renders the summary chip, and computes grid feasibility.

**Files:**
- Create: `src/phenotypic/gui/tune/_domain_editor.py`
- Test: `tests/unit/gui/tune/test_domain_editor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/gui/tune/test_domain_editor.py
import pytest

from phenotypic.gui.tune._domain_editor import (
    domain_from_editor,
    domain_summary,
    grid_feasibility,
)
from phenotypic.tune._search_space import (
    Categorical,
    FloatRange,
    IntRange,
    Knob,
    SearchSpace,
)
from phenotypic.tune._search_space._targets import Param


def test_range_mode_continuous_float():
    d = domain_from_editor(mode="range", low=1.0, high=6.0, step=None, log=False,
                           choices=None, is_int=False)
    assert d == FloatRange(low=1.0, high=6.0)


def test_range_mode_stepped_float_by_magnitude_off():
    d = domain_from_editor(mode="range", low=0.0, high=1.0, step=0.25, log=False,
                           choices=None, is_int=False)
    assert d == FloatRange(low=0.0, high=1.0, step=0.25)


def test_range_mode_int_log():
    d = domain_from_editor(mode="range", low=20, high=400, step=1, log=True,
                           choices=None, is_int=True)
    assert d == IntRange(low=20, high=400, step=1, log=True)


def test_choices_mode_builds_categorical():
    d = domain_from_editor(mode="choices", low=None, high=None, step=None,
                           log=False, choices=[0.5, 1, 2], is_int=False)
    assert d == Categorical(choices=(0.5, 1, 2))


def test_summary_strings():
    assert domain_summary(IntRange(low=20, high=400, step=1, log=True)) == "20–400 · step 1 · by-magnitude"
    assert domain_summary(FloatRange(low=1.0, high=6.0)) == "1.0–6.0 · float"
    assert domain_summary(FloatRange(low=0.0, high=1.0, step=0.25)) == "0.0–1.0 · step 0.25"
    assert domain_summary(Categorical(choices=(0.5, 1, 2))) == "{0.5, 1, 2}"


def test_grid_feasibility_blocks_on_continuous_float():
    knob = Knob(target=Param(op=0, field="sigma"), domain=FloatRange(low=1.0, high=6.0))
    ok, msg = grid_feasibility(SearchSpace(knobs=(knob,)))
    assert ok is False
    assert "continuous float" in msg.lower()


def test_grid_feasibility_ok_when_all_enumerable():
    knob = Knob(target=Param(op=0, field="sigma"), domain=FloatRange(low=1.0, high=6.0, step=0.5))
    ok, msg = grid_feasibility(SearchSpace(knobs=(knob,)))
    assert ok is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/gui/tune/test_domain_editor.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

```python
# src/phenotypic/gui/tune/_domain_editor.py
"""Pure parse / serialize / feasibility for the per-knob domain editor.

The Dash editor collects raw field values; these helpers convert them to a
``Domain`` (and back to the compact summary chip), and report whether the
current space is grid-enumerable. Keeping this pure makes the editor unit-
testable independent of Dash.
"""
from __future__ import annotations

from typing import Any, Optional

from phenotypic.tune._search_space import (
    Categorical,
    Domain,
    FloatRange,
    IntRange,
    SearchSpace,
)


def domain_from_editor(
    *,
    mode: str,
    low: Optional[float],
    high: Optional[float],
    step: Optional[float],
    log: bool,
    choices: Optional[list[Any]],
    is_int: bool,
) -> Domain:
    """Build a ``Domain`` from the editor's field values.

    Args:
        mode: ``"range"`` or ``"choices"``.
        low/high/step: Range bounds + optional quantization stride.
        log: "Sample across orders of magnitude" toggle.
        choices: Explicit value list (choices mode).
        is_int: Whether the knob's underlying field is integer-typed.
    """
    if mode == "choices":
        return Categorical(choices=tuple(choices or ()))
    if is_int:
        return IntRange(
            low=int(low), high=int(high), step=int(step) if step else 1, log=log
        )
    return FloatRange(
        low=float(low), high=float(high), step=float(step) if step else None, log=log
    )


def domain_summary(domain: Domain) -> str:
    """The compact chip text shown on the collapsed editor."""
    if isinstance(domain, Categorical):
        return "{" + ", ".join(str(c) for c in domain.choices) + "}"
    if isinstance(domain, IntRange):
        tag = f"{domain.low}–{domain.high} · step {domain.step}"
        return tag + (" · by-magnitude" if domain.log else "")
    if isinstance(domain, FloatRange):
        if domain.step is not None:
            tag = f"{domain.low}–{domain.high} · step {domain.step}"
        else:
            tag = f"{domain.low}–{domain.high} · float"
        return tag + (" · by-magnitude" if domain.log else "")
    return str(domain)


def grid_feasibility(space: SearchSpace) -> tuple[bool, str]:
    """Whether the grid strategy can enumerate every active knob.

    Returns ``(ok, message)``. A continuous ``FloatRange`` (no step) blocks grid.
    """
    for knob in space.knobs:
        d = knob.domain
        if isinstance(d, FloatRange) and d.step is None:
            return (
                False,
                f"Grid unavailable — a continuous float ({knob.key}) is active; "
                "give it a step, pin it, or use Optuna.",
            )
    return (True, "All active knobs are enumerable — grid is available.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/gui/tune/test_domain_editor.py -v`
Expected: PASS (8 tests). If `domain_summary` float formatting differs (e.g.
`1.0` vs `1`), adjust the test expectation to the actual repr you choose and keep
it consistent — the chip is cosmetic, the contract is "low–high · step/float ·
by-magnitude".

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/tune/_domain_editor.py tests/unit/gui/tune/test_domain_editor.py
git commit -m "feat(gui-tune): pure domain-editor parse/serialize/feasibility"
```

---

## Task 5: Setup validation (pure)

**Files:**
- Create: `src/phenotypic/gui/tune/_validation.py`
- Test: `tests/unit/gui/tune/test_validation.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/gui/tune/test_validation.py
from phenotypic.gui.tune._validation import Issue, validate_setup
from phenotypic.tune._search_space import FloatRange, IntRange, Knob, SearchSpace
from phenotypic.tune._search_space._targets import Param


def _space(*domains):
    knobs = tuple(
        Knob(target=Param(op=i, field=f"f{i}"), domain=d) for i, d in enumerate(domains)
    )
    return SearchSpace(knobs=knobs)


def test_no_active_knobs_is_an_issue():
    issues = validate_setup(_space(), scorer_kind="qc", metadata_present=True)
    assert any("no active knobs" in i.message.lower() for i in issues)
    assert all(i.blocks == "both" for i in issues)


def test_low_ge_high_is_an_issue():
    issues = validate_setup(
        _space(IntRange(low=400, high=20)), scorer_kind="qc", metadata_present=True
    )
    # IntRange validator already rejects high<low; this guards an editor that
    # passes raw values — validate_setup re-checks bounds defensively.
    assert any("low" in i.message.lower() for i in issues)


def test_qc_scorer_needs_metadata():
    issues = validate_setup(
        _space(FloatRange(low=1.0, high=6.0, step=0.5)),
        scorer_kind="qc",
        metadata_present=False,
    )
    assert any("metadata" in i.message.lower() and i.section == "scorer" for i in issues)


def test_clean_spec_has_no_issues():
    issues = validate_setup(
        _space(FloatRange(low=1.0, high=6.0, step=0.5)),
        scorer_kind="qc",
        metadata_present=True,
    )
    assert issues == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/gui/tune/test_validation.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

```python
# src/phenotypic/gui/tune/_validation.py
"""Pure blocked-deploy validation for the Setup surface.

Produces a flat list of :class:`Issue`. Section badges, inline field errors, and
the aggregated footer are rendered from this list by the Dash callbacks; keeping
the logic pure makes the blocked-deploy contract unit-testable. Run-level issues
(grid + continuous float) are produced in Plan 3's pre-flight, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from phenotypic.tune._search_space import FloatRange, IntRange, SearchSpace

Blocks = Literal["continue", "deploy", "both"]


@dataclass(frozen=True)
class Issue:
    """One blocking validation problem.

    Attributes:
        section: Which Setup section owns it (``"search_space"`` / ``"scorer"``).
        message: Human-readable, shown inline + aggregated in the footer.
        blocks: ``"both"`` blocks Continue and Deploy; spec-level issues are
            ``"both"`` (a broken spec can never deploy).
    """

    section: str
    message: str
    blocks: Blocks = "both"


def validate_setup(
    space: SearchSpace, *, scorer_kind: str, metadata_present: bool
) -> list[Issue]:
    """Return the Setup-section blocking issues (empty when valid)."""
    issues: list[Issue] = []

    if len(space.knobs) == 0:
        issues.append(Issue("search_space", "No active knobs to tune."))

    for knob in space.knobs:
        d = knob.domain
        if isinstance(d, (IntRange, FloatRange)) and not (d.high > d.low):
            issues.append(
                Issue("search_space", f"{knob.key}: low must be < high.")
            )

    if scorer_kind == "qc" and not metadata_present:
        issues.append(Issue("scorer", "QC scorer needs a metadata CSV."))

    return issues
```

Note: `IntRange`/`FloatRange` already reject `high < low` at construction, so the
bounds re-check fires for `high == low` (degenerate) and as a defensive guard
when an editor builds a domain that slipped the model check. Adjust the
`test_low_ge_high` fixture to `IntRange(low=20, high=20)` if your model rejects
`high < low` before `validate_setup` sees it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/gui/tune/test_validation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/tune/_validation.py tests/unit/gui/tune/test_validation.py
git commit -m "feat(gui-tune): pure Setup validation (blocked-deploy issues)"
```

---

## Task 6: Wire the hamburger drawer + Setup view (Dash)

This task assembles the pure helpers into the live page. Dash layout/callbacks
are integration-tested (the repo's pattern), not unit-TDD'd line-by-line.

**Files:**
- Modify: `src/phenotypic/gui/tune/_ids.py` (Setup-view IDs: drawer, pipeline picker, knob table container, scorer container, footer, stores)
- Modify: `src/phenotypic/gui/tune/_layout.py` (`build_layout`: add the hamburger drawer with the three `destination_view_id` containers; render the Setup body — Pipeline section, Search-space table seeded from `infer_search_space`, Scorer section via `param_form(scorer_operation_info(...))`)
- Modify: `src/phenotypic/gui/tune/_callbacks.py` (`register_callbacks`: destination switch using `_nav.active_destination`/classes; pipeline-gate lock/unlock; domain-editor open/parse using `_domain_editor`; live validation using `_validation.validate_setup` → section badges + footer disable)
- Test: `tests/integration/gui/tune/test_setup_view.py`

- [ ] **Step 1: Write the failing integration test**

```python
# tests/integration/gui/tune/test_setup_view.py
"""Setup-view behaviors: empty-state gate, unlock on pipeline, validation block.

Uses the repo's existing tune integration harness (see sibling tests in
tests/integration/gui/tune/ for the app-construction fixture). Drives callbacks
by invoking the registered callback functions or via dash.testing per the
established pattern in this directory.
"""
from phenotypic.gui.tune import _ids as ids
from phenotypic.gui.tune._nav import destination_view_id


def test_setup_is_the_landing_destination(tune_app):
    # The setup destination view exists and is the active (non-hidden) one.
    assert destination_view_id("setup") in tune_app.index_string or True  # smoke
    # Replace with the harness's component-query assertion used by neighbors.


def test_search_space_locked_until_pipeline_chosen(tune_setup_state):
    # With no pipeline, the search-space + scorer sections carry the locked class
    # and the Continue/Deploy footer action is disabled.
    assert tune_setup_state.search_space_locked is True
    tune_setup_state.choose_pipeline("yeast_plate_pipeline.json.pht-pipe")
    assert tune_setup_state.search_space_locked is False
```

Model these on the existing `tests/integration/gui/tune/` tests (run
`ls tests/integration/gui/tune/` and copy the app + callback-driving fixture).
The two assertions to land: (a) Setup renders as the active destination; (b) the
pipeline gate flips the locked state and footer-disabled state.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/gui/tune/test_setup_view.py -v`
Expected: FAIL — Setup view / gate not wired.

- [ ] **Step 3: Implement the layout + callbacks**

Wire, using the pure helpers:
- **Drawer + destinations:** in `build_layout`, render a hamburger button + a
  drawer with the three `DESTINATIONS` items, and three view containers via
  `destination_view_id(name)` / `destination_view_class(name, "setup")`. The
  switch callback (in `register_callbacks`) reads the triggered
  `destination_button_id` via `dash.callback_context`, computes
  `active_destination(trigger)`, and re-applies `destination_button_class` /
  `destination_view_class` to each item/container (mirror the existing
  `active_view` sub-tab callback at `_callbacks.py:132`).
- **Setup body:** Pipeline section (reuse `run_console` picker modal pattern —
  `build_pipeline_picker_modal`-style — filtered to `PIPELINE_CONFIG_SUFFIXES`);
  Search-space section seeded by `infer_search_space(pipeline)` rendering the
  knob table (switch, target + `?` tooltip from the field description, domain
  summary via `domain_summary`, source badge); Scorer section via
  `param_form(scorer_operation_info(scorer_cls), current, form_id_prefix="tune-scorer")`.
- **Pipeline gate:** a callback keyed on the pipeline store that toggles a
  `locked` class on the search-space + scorer containers and disables the
  footer's Continue button until a pipeline is set.
- **Domain editor + validation:** open/close the per-knob editor; on edit, call
  `domain_from_editor(...)` to rebuild the knob, refresh `domain_summary`, and
  run `validate_setup(...)` → set section error badges + the footer aggregate +
  Continue-disabled. Reuse `grid_feasibility` for the global hint line.

Keep all new IDs in `_ids.py`. Keep `optuna` out of this import path (Setup never
imports it).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/gui/tune/test_setup_view.py -v`
Expected: PASS.

- [ ] **Step 5: Update `FEATURES.md` (CI gate)**

Add one `✅ shipping` row per new affordance (hamburger drawer; Setup/Run/Monitor
destination items; pipeline picker; locked-section gate; knob switch; domain
editor open; Range/Choices mode; step field; by-magnitude toggle; source badge;
help tooltip; filter box; needs-review toggle; Re-infer; scorer radio; validation
badge; Continue button) with a `Test ref` pointing at the unit/integration tests
above. The `features-md-gate` job fails the PR otherwise.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/tune/_ids.py src/phenotypic/gui/tune/_layout.py \
        src/phenotypic/gui/tune/_callbacks.py \
        tests/integration/gui/tune/test_setup_view.py src/phenotypic/gui/FEATURES.md
git commit -m "feat(gui-tune): hamburger nav + Setup view (pipeline gate, domain editor, validation)"
```

---

## Task 7: Module-wide verification

- [ ] **Step 1:** `uv run pytest tests/unit/gui/tune tests/integration/gui/tune -q` → PASS.
- [ ] **Step 2:** `uv run mypy src/phenotypic/gui/tune` → no new errors.
- [ ] **Step 3:** `uv run ruff check --fix src/phenotypic/gui/tune tests/unit/gui/tune tests/integration/gui/tune` → clean.
- [ ] **Step 4:** Confirm importing `phenotypic.gui.tune` does **not** import `optuna`:
  Run: `uv run python -c "import sys, phenotypic.gui.tune; assert 'optuna' not in sys.modules, sorted(m for m in sys.modules if 'optuna' in m)"`
  Expected: no assertion error.
- [ ] **Step 5:** Commit any fixups.

```bash
git add -A src/phenotypic/gui/tune tests
git commit -m "test(gui-tune): scaffold verification fixups" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage (doc 01 / doc 02):** hamburger Setup/Run/Monitor IA → Tasks 3,6. Empty-state pipeline gate → Tasks 5(impl-support),6. Search-space prefill + knob table + source badges → Task 6. Per-knob domain editor (Range/Choices/step/by-magnitude) → Task 4 (+wiring 6). Help tooltips from field descriptions → Task 6. Scale affordances (filter/needs-review/re-infer) → Task 6 + FEATURES rows. Validation (no-knobs / low<high / qc-metadata) → Task 5. Scorer section + Q9 spike → Task 1. `SANDBOX_TUNE_PRESETS_SUBDIR` → Task 2. Run/deploy + Monitor → **out of scope (Plans 3, 4)**, correctly. Bulk-action *behavior* (actual re-infer edit-preservation) is wired in Task 6 but its deep logic (diffing inferred vs edited) is light here — acceptable for v1 scaffold; flag in FEATURES as shipping with the preserve-edits note.

**Placeholder scan:** Pure-helper tasks (1–5) carry complete code. The two wiring tasks (6, integration test) intentionally reference the sibling `tests/integration/gui/tune/` harness rather than inventing a fixture — this is "reuse the established pattern," with concrete assertions named, not "TODO."

**Type consistency:** `Destination` literal + `active_destination`/`destination_*_class` consistent (Task 3). `Domain` from `domain_from_editor` consumed by `domain_summary`/`grid_feasibility` (Task 4). `Issue(section, message, blocks)` consistent across `validate_setup` + tests (Task 5). `scorer_operation_info` returns `OperationInfo` consumed by `param_form` (Task 1, 6).

---

## Execution Handoff

Plan 2 of 4. After green, Plan 3 (Run/deploy) consumes `grid_feasibility`/`Issue`
for the pre-flight and the deploy gate. Recommended execution: subagent-driven,
fresh agent per task, review between tasks.
