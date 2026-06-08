# Tune Config Builder — Monitor & Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Monitor destination — a run switcher over every live/finished study, a Local-vs-SLURM live view (with degrade when the study store is unreachable), mode-aware run cancellation, and the result zone that closes the loop: **Export best pipeline** (single-objective) or **Pareto export** (multi-objective).

**Architecture:** Same pure-helper pattern. New pure module `_monitor.py` (switcher view-models, live-view selection, cancel prompts) and `_export.py` (build + write the tuned pipeline). Reuses `RunRegistry`/`RunRecord` (`shell/_runs_registry.py`), `LocalRunner.stop` for local cancel, `scancel` for SLURM, and the tune engine's `build_pipeline` + `is_multi_objective`/`objective_names`. The existing Monitor/Curate charts stay as-is; this plan adds the switcher, the live-view swap, cancel, and the export zone.

**Tech Stack:** Python, Dash, pydantic v2, pytest, `uv`. (Optuna stays lazy — the live study read already gates on the `tune` extra inside the poll callback; this plan adds no eager optuna import.)

**Depends on:** Plans 1–3 merged. Plan 3's deploy callback writes `RunRegistry` entries (mode + `slurm_job_id` + storage) that this plan reads.

**Spec refs:** `docs/superpowers/spec/tune-config-builder/03-run-deploy-and-monitor.md` (Monitor section); mockup Monitor view (run switcher, SLURM fleet card, best-result zone).

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/phenotypic/gui/tune/_monitor.py` | **New.** Pure switcher / live-view / cancel logic | `run_switcher_items()`, `live_view_kind()`, `cancel_prompt()` |
| `src/phenotypic/gui/tune/_export.py` | **New.** Build + write the tuned pipeline | `export_winning_pipeline()`, `export_pareto_pipeline()` |
| `src/phenotypic/gui/tune/_ids.py` | Tune IDs | Switcher, live-view containers, cancel dialog, export zone |
| `src/phenotypic/gui/tune/_layout.py` | Layout | Monitor: switcher + Local/SLURM live containers + export zone |
| `src/phenotypic/gui/tune/_callbacks.py` | Callbacks | Switcher select, live-view swap, cancel confirm/exec, export |
| `src/phenotypic/gui/FEATURES.md` | Ledger | Monitor affordance rows |
| `tests/unit/gui/tune/test_monitor.py` | **New** | switcher / live-view / cancel-prompt cases |
| `tests/unit/gui/tune/test_export.py` | **New** | export single + pareto cases |
| `tests/integration/gui/tune/test_monitor_view.py` | **New** | switcher swap + cancel + export (mocked runner) |

---

## Task 1: Run switcher + live-view selection + cancel prompt (pure)

**Files:**
- Create: `src/phenotypic/gui/tune/_monitor.py`
- Test: `tests/unit/gui/tune/test_monitor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/gui/tune/test_monitor.py
from types import SimpleNamespace

from phenotypic.gui.tune._monitor import (
    cancel_prompt,
    live_view_kind,
    run_switcher_items,
)


def _rec(run_id, mode, status, job=None):
    return SimpleNamespace(run_id=run_id, mode=mode, status=status, slurm_job_id=job)


def test_switcher_marks_active_and_killable():
    recs = [
        _rec("a", "local", "running"),
        _rec("b", "slurm", "running", job="4815162"),
        _rec("c", "local", "complete"),
    ]
    items = run_switcher_items(recs, active_id="b")
    by_id = {it.run_id: it for it in items}
    assert by_id["b"].active is True
    assert by_id["a"].active is False
    assert by_id["a"].killable is True       # running → cancellable
    assert by_id["c"].killable is False      # complete → not cancellable


def test_live_view_kind_selects_local_vs_slurm_and_degrades():
    assert live_view_kind("local", store_reachable=True) == "local-log"
    assert live_view_kind("slurm", store_reachable=True) == "slurm-fleet"
    assert live_view_kind("slurm", store_reachable=False) == "slurm-detached"


def test_cancel_prompt_is_mode_aware():
    local = cancel_prompt("yeast_qc_tpe", "local")
    assert "SIGTERM" in local and "resumed" in local
    slurm = cancel_prompt("ecoli_grid", "slurm")
    assert "scancel" in slurm and "shared" in slurm
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/gui/tune/test_monitor.py -v`
Expected: FAIL — `_monitor` module does not exist.

- [ ] **Step 3: Implement**

```python
# src/phenotypic/gui/tune/_monitor.py
"""Pure Monitor logic: run-switcher view-models, live-view selection, cancel text.

Decoupled from the Dash layer and from ``RunRegistry``'s exact shape — the
switcher takes any objects exposing ``run_id`` / ``mode`` / ``status`` /
``slurm_job_id`` (``RunRecord`` qualifies), so the callback adapts registry rows
to these helpers and the unit tests use lightweight stand-ins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

LiveViewKind = Literal["local-log", "slurm-fleet", "slurm-detached"]

#: Statuses for which a run can still be cancelled.
_CANCELLABLE = {"running", "submitting"}


@dataclass(frozen=True)
class SwitcherItem:
    """One run-switcher pill's view-model."""

    run_id: str
    mode: str
    status: str
    active: bool
    killable: bool


def run_switcher_items(records: list[Any], *, active_id: str | None) -> list[SwitcherItem]:
    """Build switcher pills from registry rows (preserving input order)."""
    return [
        SwitcherItem(
            run_id=r.run_id,
            mode=r.mode,
            status=r.status,
            active=(r.run_id == active_id),
            killable=(r.status in _CANCELLABLE),
        )
        for r in records
    ]


def live_view_kind(mode: str, *, store_reachable: bool) -> LiveViewKind:
    """Pick the live view for the selected run.

    Local runs tail stdout; SLURM runs show a fleet card when the study store is
    reachable for polling, else a detached card (job id + task count only).
    """
    if mode == "slurm":
        return "slurm-fleet" if store_reachable else "slurm-detached"
    return "local-log"


def cancel_prompt(name: str, mode: str) -> str:
    """Mode-aware confirmation body for cancelling a run."""
    if mode == "slurm":
        return (
            f"Issue scancel for {name}'s array job? Completed trials already "
            "written to the shared study store are kept, and the study can be "
            "resumed later."
        )
    return (
        f"Send SIGTERM to {name}? The trials already recorded in the journal are "
        "kept, and the study can be resumed."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/gui/tune/test_monitor.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/tune/_monitor.py tests/unit/gui/tune/test_monitor.py
git commit -m "feat(gui-tune): pure run-switcher / live-view / cancel-prompt logic"
```

---

## Task 2: Export the tuned pipeline (single + Pareto)

**Files:**
- Create: `src/phenotypic/gui/tune/_export.py`
- Test: `tests/unit/gui/tune/test_export.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/gui/tune/test_export.py
from phenotypic import ImagePipeline
from phenotypic.enhance import GaussianBlur
from phenotypic.gui.tune._export import (
    export_pareto_pipeline,
    export_winning_pipeline,
)
from phenotypic.tools_._io_constants import (
    CONFIG_SUFFIX_PIPELINE,
    best_pipeline_path,
    pareto_best_pipeline_path,
)


def _base() -> ImagePipeline:
    # ImagePipeline is keyword-only constructed; there is no .add() method.
    return ImagePipeline(ops=[GaussianBlur(sigma=1.0)])


def test_export_winning_writes_typed_pipeline(tmp_path):
    out = export_winning_pipeline(_base(), {"0.sigma": 2.5}, tmp_path)
    assert out == best_pipeline_path(tmp_path)
    assert str(out).endswith(CONFIG_SUFFIX_PIPELINE)
    assert out.exists()
    # round-trips: the exported file loads back as a pipeline
    reloaded = ImagePipeline.from_json(out)
    assert reloaded is not None


def test_export_pareto_writes_per_objective(tmp_path):
    out = export_pareto_pipeline(_base(), {"0.sigma": 3.0}, tmp_path, objective="s0")
    assert out == pareto_best_pipeline_path(tmp_path, "s0")
    assert out.exists()
    assert str(out).endswith(CONFIG_SUFFIX_PIPELINE)
```

(Construction is keyword-only: `ImagePipeline(ops=[...])` — confirmed in
`tests/unit/tune/test_build_pipeline_nested.py`. The contract under test is
"`build_pipeline(base, params)` → write to the canonical typed path".)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/gui/tune/test_export.py -v`
Expected: FAIL — `_export` module does not exist.

- [ ] **Step 3: Implement**

```python
# src/phenotypic/gui/tune/_export.py
"""Export a tuned pipeline from a trial's winning params — closes the loop.

The run produces ``best_params`` (knob values); the *usable* artifact is a
pipeline. We apply the params to the base via the tune engine's ``build_pipeline``
and write the canonical typed config path. Single-objective studies write
``best_pipeline.json.pht-pipe``; a Pareto trial writes the per-objective
``pareto/best_<objective>.json.pht-pipe``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from phenotypic import ImagePipeline
from phenotypic.tools_._io_constants import (
    best_pipeline_path,
    pareto_best_pipeline_path,
)
from phenotypic.tune._evaluation import build_pipeline


def export_winning_pipeline(
    base: ImagePipeline, params: dict[str, Any], output_dir: Path
) -> Path:
    """Write the single-objective winner to ``best_pipeline.json.pht-pipe``."""
    pipe = build_pipeline(base, params)
    path = best_pipeline_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pipe.to_json(path)  # ImagePipeline.to_json normalizes to the typed suffix
    return path


def export_pareto_pipeline(
    base: ImagePipeline, params: dict[str, Any], output_dir: Path, *, objective: str
) -> Path:
    """Write a Pareto-front trial to ``pareto/best_<objective>.json.pht-pipe``."""
    pipe = build_pipeline(base, params)
    path = pareto_best_pipeline_path(output_dir, objective)
    path.parent.mkdir(parents=True, exist_ok=True)
    pipe.to_json(path)
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/gui/tune/test_export.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/tune/_export.py tests/unit/gui/tune/test_export.py
git commit -m "feat(gui-tune): export tuned pipeline (single + pareto) to typed config path"
```

---

## Task 3: Wire the Monitor switcher + live-view swap (Dash)

**Files:**
- Modify: `src/phenotypic/gui/tune/_ids.py` (switcher container, per-pill ids, live-view containers `tune-live-local` / `tune-live-slurm`)
- Modify: `src/phenotypic/gui/tune/_layout.py` (Monitor body: run-switcher row above the existing charts; two live-view containers)
- Modify: `src/phenotypic/gui/tune/_callbacks.py` (build switcher from `RunRegistry.list()` via `run_switcher_items`; select-run swaps the live view via `live_view_kind`)
- Test: `tests/integration/gui/tune/test_monitor_view.py`

- [ ] **Step 1: Write the failing integration test (switcher half)**

```python
# tests/integration/gui/tune/test_monitor_view.py
def test_selecting_a_slurm_run_shows_the_fleet_card(tune_monitor_state):
    tune_monitor_state.seed_runs(local="run-a", slurm="run-b")
    tune_monitor_state.select("run-b")
    assert tune_monitor_state.live_view == "slurm-fleet"   # reachable store
    tune_monitor_state.select("run-a")
    assert tune_monitor_state.live_view == "local-log"
```

Build `tune_monitor_state` on the existing tune integration harness; seed the
shared `RunRegistry` (`app.server.config[CFG_RUN_REGISTRY]`) with a local + a
SLURM `RunRecord`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/gui/tune/test_monitor_view.py -k fleet -v`
Expected: FAIL — switcher/live-view not wired.

- [ ] **Step 3: Implement**

- Render the run-switcher row in the Monitor body from
  `run_switcher_items(registry.list(), active_id=…)`; each pill shows
  name/mode/status and (when `killable`) a ✕.
- Two live-view containers: `tune-live-local` (the existing log tail / progress)
  and `tune-live-slurm` (the fleet card: array-task chips + job id + the polling
  note; for `slurm-detached`, the degraded "submitted N tasks — store
  unreachable" variant). A select-run callback computes
  `live_view_kind(record.mode, store_reachable=…)` and toggles which container is
  visible (reuse the hidden-class pattern). `store_reachable` is derived by
  attempting the existing study-read open in a try/except (already gated on the
  `tune` extra) — unreachable → `slurm-detached`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/gui/tune/test_monitor_view.py -k fleet -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/tune/_ids.py src/phenotypic/gui/tune/_layout.py \
        src/phenotypic/gui/tune/_callbacks.py tests/integration/gui/tune/test_monitor_view.py
git commit -m "feat(gui-tune): Monitor run switcher + Local/SLURM live-view swap"
```

---

## Task 4: Wire cancellation (Dash)

**Files:**
- Modify: `src/phenotypic/gui/tune/_ids.py` (cancel confirm dialog ids)
- Modify: `src/phenotypic/gui/tune/_layout.py` (the confirm dialog)
- Modify: `src/phenotypic/gui/tune/_callbacks.py` (✕ → confirm → stop/scancel → registry + counter)
- Test: `tests/integration/gui/tune/test_monitor_view.py` (cancel half)

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/gui/tune/test_monitor_view.py  (add)
def test_cancel_local_run_stops_runner_and_decrements_counter(tune_monitor_state, fake_runner):
    tune_monitor_state.seed_runs(local="run-a")
    tune_monitor_state.live_runs = 1
    tune_monitor_state.ask_cancel("run-a")
    assert "SIGTERM" in tune_monitor_state.confirm_text
    tune_monitor_state.confirm_cancel()
    assert fake_runner.stopped == ["run-a"]      # LocalRunner.stop called
    assert tune_monitor_state.live_runs == 0
    assert tune_monitor_state.record_status("run-a") == "cancelled"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/gui/tune/test_monitor_view.py -k cancel -v`
Expected: FAIL — cancel not wired.

- [ ] **Step 3: Implement**

- The ✕ on a killable pill opens the confirm dialog with body
  `cancel_prompt(name, record.mode)`.
- Confirm:
  - Local → `runner.stop(run_id)` (`runner = app.server.config[CFG_RUNNER]`).
  - SLURM → run `scancel <record.slurm_job_id>` via the SLURM helper / subprocess
    (reuse `run_console/_slurm.py` if it exposes a cancel; else a small
    `subprocess.run(["scancel", job_id])`).
  - Update the `RunRecord.status` to `"cancelled"` in the registry, decrement the
    live-runs session store, re-render the switcher.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/gui/tune/test_monitor_view.py -k cancel -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/tune/_ids.py src/phenotypic/gui/tune/_layout.py \
        src/phenotypic/gui/tune/_callbacks.py tests/integration/gui/tune/test_monitor_view.py
git commit -m "feat(gui-tune): cancel a run (SIGTERM/scancel) with mode-aware confirm"
```

---

## Task 5: Wire the export zone (Dash) — single + Pareto

**Files:**
- Modify: `src/phenotypic/gui/tune/_ids.py` (best-result zone + export buttons + Pareto picker ids)
- Modify: `src/phenotypic/gui/tune/_layout.py` (best-result card / Pareto front view)
- Modify: `src/phenotypic/gui/tune/_callbacks.py` (export buttons → `_export`; multi-objective branch)
- Test: `tests/integration/gui/tune/test_monitor_view.py` (export half)

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/gui/tune/test_monitor_view.py  (add)
def test_single_objective_shows_best_card_and_exports(tune_monitor_state, tmp_path):
    tune_monitor_state.seed_single_objective_best(params={"0.sigma": 2.8})
    assert tune_monitor_state.result_zone == "single-best"
    path = tune_monitor_state.click_export_best()
    assert path.name == "best_pipeline.json.pht-pipe"
    assert path.exists()


def test_multi_objective_shows_pareto_and_exports_picked(tune_monitor_state, tmp_path):
    tune_monitor_state.seed_multi_objective(front={"s0": {"0.sigma": 2.0}})
    assert tune_monitor_state.result_zone == "pareto"
    path = tune_monitor_state.click_export_pareto("s0")
    assert "pareto" in str(path) and path.name == "best_s0.json.pht-pipe"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/gui/tune/test_monitor_view.py -k export -v`
Expected: FAIL — export zone not wired.

- [ ] **Step 3: Implement**

- Choose the result zone from `is_multi_objective(spec.scorer)` (import from
  `phenotypic.tune._multi_objective`, alongside `objective_names`; **not**
  `_evaluation`):
  - **single** → a "Best so far" card (trial id, score, params, "may still
    improve" while running); **Export best** → `export_winning_pipeline(base,
    best_params, output_dir)`; plus **Open in Builder** / **Send to Run Console**
    handing the exported path to the other mounts via the shell-level store.
  - **multi** → a Pareto view listing front trials per `objective_names(scorer)`;
    a picker selects a trial; **Export** → `export_pareto_pipeline(base, params,
    output_dir, objective=…)`.
- Read the best/front params from the bound run's deliverables (the engine writes
  `best_params.json` / the Pareto sidecar); `base` is the spec's pipeline.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/gui/tune/test_monitor_view.py -k export -v`
Expected: PASS.

- [ ] **Step 5: Update `FEATURES.md`**

Rows for: run switcher, each pill, ✕ cancel + confirm dialog, Local/SLURM/detached
live views, best-result card, Export best, Open in Builder, Send to Run Console,
Pareto front view + picker + per-objective export — each with a `Test ref`.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/tune/_ids.py src/phenotypic/gui/tune/_layout.py \
        src/phenotypic/gui/tune/_callbacks.py tests/integration/gui/tune/test_monitor_view.py \
        src/phenotypic/gui/FEATURES.md
git commit -m "feat(gui-tune): Monitor export zone — best pipeline + Pareto export"
```

---

## Task 6: Module-wide verification

- [ ] **Step 1:** `uv run pytest tests/unit/gui/tune tests/integration/gui/tune -q` → PASS.
- [ ] **Step 2:** `uv run mypy src/phenotypic/gui/tune` → no new errors.
- [ ] **Step 3:** `uv run ruff check --fix src/phenotypic/gui/tune tests/unit/gui/tune tests/integration/gui/tune` → clean.
- [ ] **Step 4:** Lazy-optuna guard:
  `uv run python -c "import sys, phenotypic.gui.tune; assert 'optuna' not in sys.modules"` → no error.
- [ ] **Step 5:** Refresh GUI tutorial screenshots (the author→deploy→monitor flow now ends in export):
  `uv run python scripts/capture_gui_tutorial_screenshots.py` and commit the full PNG set.
- [ ] **Step 6:** Commit fixups.

```bash
git add -A src/phenotypic/gui/tune tests docs/source/tutorials/gui
git commit -m "test(gui-tune): monitor verification + refreshed tutorial screenshots" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage (doc 03 Monitor):** run switcher → Tasks 1,3. Local vs SLURM live
view + degrade-when-unreachable → Tasks 1 (`live_view_kind`), 3. Cancel (SIGTERM
local / scancel SLURM, trials-kept text) → Tasks 1 (`cancel_prompt`), 4. Single
best card + Export best → Tasks 2,5. **Pareto** export (per D13/Q5) → Tasks 2,5.
Live-runs decrement on cancel → Task 4. Monitor/Curate charts unchanged → not
touched. Notifications-when-backgrounded (a minor surfaced item) → not in scope;
the live-runs counter + switcher cover return-to-it.

**Placeholder scan:** Pure tasks (1,2) carry complete code. Wiring tasks (3–5)
name exact helpers/signatures (`run_switcher_items`, `live_view_kind`,
`cancel_prompt`, `runner.stop`, `is_multi_objective`, `objective_names`,
`export_*`) and reuse the integration harness — no "TODO". The `ImagePipeline`
one-op builder in Task 2 is gated with "copy the neighbor's builder if the API
differs".

**Type consistency:** `SwitcherItem(run_id, mode, status, active, killable)`
consistent across helper + test. `LiveViewKind` literal consistent. `export_*`
return `Path` to the canonical helpers (`best_pipeline_path`,
`pareto_best_pipeline_path`) — matching their verified signatures. `runner.stop`
used per its real signature (`stop(run_id, *, grace_seconds=...)`).

---

## Execution Handoff

Plan 4 of 4 — completes the feature. After all four plans are green, the tune tab
authors a spec, deploys it (local/SLURM), monitors it, and exports a runnable
tuned pipeline. Recommended execution across the series: subagent-driven, fresh
agent per task, review gates between tasks; execute Plan 1 → 2 → 3 → 4 in order
(each depends on the prior).
