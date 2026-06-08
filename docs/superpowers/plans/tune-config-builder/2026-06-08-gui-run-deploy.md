# Tune Config Builder — Run & Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the **Run** destination — strategy / budget / advanced / compute config, image source from the hub's shared source-image-root, a pre-flight that blocks grid+continuous-float, and a **Deploy** that serializes the spec, launches `python -m phenotypic.tune run` via the run-console runner (Local or SLURM), and auto-advances to Monitor.

**Architecture:** Same pure-helper pattern as Plan 2. New pure module `_run_argv.py` (the tune CLI argv builder, mirroring `run_console/_state.to_argv`), a pre-flight in `_validation.py` (reusing Plan 2's `grid_feasibility`/`Issue`), and an image-source resolver. Deploy reuses `run_console`'s `LocalRunner` (`start(run_id, argv, *, output_dir)`) and `submit_slurm`, **imported in place** (no move — Plan/spec decision D13/Q3). Monitor extensions are **Plan 4**.

**Tech Stack:** Python, Dash, pydantic v2, Optuna (deploy path only, lazy import), pytest, `uv`.

**Depends on:** Plan 1 (`FloatRange.step`, `phenotypic_version`) and Plan 2 (hamburger nav, `Issue`, `grid_feasibility`, the Setup spec assembly) merged.

**Spec refs:** `docs/superpowers/spec/tune-config-builder/03-run-deploy-and-monitor.md`; mockup Run view.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/phenotypic/gui/tune/_run_argv.py` | **New.** Pure tune CLI argv builder | `tune_run_argv()` |
| `src/phenotypic/gui/tune/_command.py` | Existing command preview renderer | Add the same Run-form override flags as `tune_run_argv()` |
| `src/phenotypic/gui/tune/_validation.py` | Plan 2 module | Add `preflight_issues()`, `can_deploy()` |
| `src/phenotypic/gui/tune/_run_image_source.py` | **New.** Run `-i` resolver | `resolve_run_images()` (separate from the Curate `_image_source.py`) |
| `src/phenotypic/gui/tune/_ids.py` | Tune IDs | Run-view + deploy IDs |
| `src/phenotypic/gui/tune/_layout.py` | Layout | Run view body |
| `src/phenotypic/gui/tune/_callbacks.py` | Callbacks | Pre-flight, deploy, auto-advance |
| `src/phenotypic/gui/FEATURES.md` | Ledger | Run/deploy affordance rows |
| `src/phenotypic/gui/WORKFLOWS.md` | Flow ledger | author→deploy→monitor flow (+ capture fn + tutorial) |
| `tests/unit/gui/tune/test_run_argv.py` | **New** | argv builder cases |
| `tests/unit/gui/tune/test_validation.py` | Plan 2 test | preflight + can_deploy cases |
| `tests/unit/gui/tune/test_run_image_source.py` | **New** | `resolve_run_images` cases |
| `tests/integration/gui/tune/test_run_deploy.py` | **New** | deploy gating + launch (mocked runner) |

---

## Task 1: Tune CLI argv builder (pure)

Mirror `run_console/_state.to_argv`, but for `python -m phenotypic.tune run`.

**Files:**
- Create: `src/phenotypic/gui/tune/_run_argv.py`
- Test: `tests/unit/gui/tune/test_run_argv.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/gui/tune/test_run_argv.py
import pytest

from phenotypic.gui.tune._run_argv import tune_run_argv


def test_minimal_local_argv():
    argv = tune_run_argv(
        spec_path="/sbx/spec.json.pht-tune",
        images_dir="/data/imgs",
        output_dir="/out/run1",
        strategy="tpe",
        n_trials=50,
        storage_url="sqlite:///out/run1/.pht-tune-cache/study.db",
        n_workers=8,
        slurm_partition=None,
        slurm_mem=None,
        slurm_time=None,
        held_out_fraction=0.2,
        cv_group="plate_id",
        slurm=False,
        screen=False,
        python="python",
    )
    assert argv[:4] == ["python", "-m", "phenotypic.tune", "run"]
    assert "/sbx/spec.json.pht-tune" in argv
    assert argv[argv.index("-i") + 1] == "/data/imgs"
    assert argv[argv.index("-o") + 1] == "/out/run1"
    assert argv[argv.index("--strategy") + 1] == "tpe"
    assert argv[argv.index("--n-trials") + 1] == "50"
    assert argv[argv.index("--n-workers") + 1] == "8"
    assert argv[argv.index("--held-out-fraction") + 1] == "0.2"
    assert argv[argv.index("--cv-group") + 1] == "plate_id"
    assert "--slurm" not in argv


def test_grid_omits_n_trials_and_slurm_flag_present():
    argv = tune_run_argv(
        spec_path="s", images_dir="i", output_dir="o", strategy="grid",
        n_trials=50, storage_url=None, n_workers=None,
        slurm_partition="batch", slurm_mem="8G", slurm_time="04:00:00",
        held_out_fraction=None, cv_group=None, slurm=True, screen=True,
        python="python",
    )
    assert "--n-trials" not in argv      # grid is exhaustive; budget ignored
    assert "--slurm" in argv
    assert "--screen" in argv
    assert argv[argv.index("--slurm-partition") + 1] == "batch"
    assert argv[argv.index("--slurm-mem") + 1] == "8G"
    assert argv[argv.index("--slurm-time") + 1] == "04:00:00"
    assert "--storage-url" not in argv   # None omitted


def test_missing_required_slot_raises():
    with pytest.raises(ValueError, match="spec_path"):
        tune_run_argv(
            spec_path="", images_dir="i", output_dir="o", strategy="tpe",
            n_trials=None, storage_url=None, n_workers=None,
            slurm_partition=None, slurm_mem=None, slurm_time=None,
            held_out_fraction=None, cv_group=None, slurm=False, screen=False,
            python="python",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/gui/tune/test_run_argv.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

```python
# src/phenotypic/gui/tune/_run_argv.py
"""Pure builder for the ``python -m phenotypic.tune run …`` argv.

Mirrors :func:`phenotypic.gui.run_console._state.to_argv` but for the tune CLI.
Returns the **full** argv (including the python executable + ``-m
phenotypic.tune``) so the deploy callback can hand it straight to
``LocalRunner.start``. The executable is injectable for deterministic tests.
"""
from __future__ import annotations

import sys
from typing import Optional


def tune_run_argv(
    *,
    spec_path: str,
    images_dir: str,
    output_dir: str,
    strategy: str,
    n_trials: Optional[int],
    storage_url: Optional[str],
    n_workers: Optional[int],
    slurm_partition: Optional[str],
    slurm_mem: Optional[str],
    slurm_time: Optional[str],
    held_out_fraction: Optional[float],
    cv_group: Optional[str],
    slurm: bool,
    screen: bool,
    python: str | None = None,
) -> list[str]:
    """Build the full launch argv for a tune run.

    Raises:
        ValueError: If ``spec_path``, ``images_dir``, or ``output_dir`` is empty.
    """
    missing = [
        name
        for name, val in (
            ("spec_path", spec_path),
            ("images_dir", images_dir),
            ("output_dir", output_dir),
        )
        if not val
    ]
    if missing:
        raise ValueError("tune_run_argv missing required field(s): " + ", ".join(missing))

    exe = python or sys.executable
    argv: list[str] = [exe, "-m", "phenotypic.tune", "run", spec_path]
    argv += ["-i", images_dir, "-o", output_dir, "--strategy", strategy]
    # grid is exhaustive — the budget flag is meaningless and rejected upstream.
    if n_trials is not None and strategy != "grid":
        argv += ["--n-trials", str(n_trials)]
    if storage_url:
        argv += ["--storage-url", storage_url]
    if n_workers is not None:
        argv += ["--n-workers", str(n_workers)]
    if slurm_partition:
        argv += ["--slurm-partition", slurm_partition]
    if slurm_mem:
        argv += ["--slurm-mem", slurm_mem]
    if slurm_time:
        argv += ["--slurm-time", slurm_time]
    if held_out_fraction is not None:
        argv += ["--held-out-fraction", str(held_out_fraction)]
    if cv_group:
        argv += ["--cv-group", cv_group]
    if slurm:
        argv.append("--slurm")
    if screen:
        argv.append("--screen")
    return argv
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/gui/tune/test_run_argv.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/tune/_run_argv.py tests/unit/gui/tune/test_run_argv.py
git commit -m "feat(gui-tune): pure tune_run_argv builder for deploy"
```

---

## Task 2: Pre-flight + deploy gate (pure)

**Files:**
- Modify: `src/phenotypic/gui/tune/_validation.py` (add `preflight_issues`, `can_deploy`)
- Test: `tests/unit/gui/tune/test_validation.py` (add cases)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/gui/tune/test_validation.py  (add)
from phenotypic.gui.tune._validation import Issue, can_deploy, preflight_issues
from phenotypic.tune._search_space import FloatRange, Knob, SearchSpace
from phenotypic.tune._search_space._targets import Param


def _space(domain):
    return SearchSpace(knobs=(Knob(target=Param(op=0, field="sigma"), domain=domain),))


def test_grid_with_continuous_float_is_a_run_issue():
    issues = preflight_issues(_space(FloatRange(low=1.0, high=6.0)), strategy="grid")
    assert len(issues) == 1
    assert issues[0].section == "strategy"
    assert issues[0].blocks == "deploy"


def test_grid_with_stepped_float_is_clean():
    assert preflight_issues(_space(FloatRange(low=1.0, high=6.0, step=0.5)), strategy="grid") == []


def test_optuna_with_continuous_float_is_clean():
    assert preflight_issues(_space(FloatRange(low=1.0, high=6.0)), strategy="optuna") == []


def test_can_deploy_only_when_no_blocking_issues():
    assert can_deploy([], []) is True
    assert can_deploy([Issue("scorer", "x")], []) is False
    assert can_deploy([], [Issue("strategy", "x", blocks="deploy")]) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/gui/tune/test_validation.py -k "preflight or can_deploy" -v`
Expected: FAIL — `preflight_issues` / `can_deploy` not defined.

- [ ] **Step 3: Implement (append to `_validation.py`)**

```python
from phenotypic.gui.tune._domain_editor import grid_feasibility


def preflight_issues(space: SearchSpace, *, strategy: str) -> list[Issue]:
    """Run-level issues from the strategy×search-space interaction.

    The canonical conflict: grid cannot enumerate a continuous ``FloatRange``.
    Blocks Deploy only (it is not a spec defect — a different strategy fixes it).
    """
    issues: list[Issue] = []
    if strategy == "grid":
        ok, msg = grid_feasibility(space)
        if not ok:
            issues.append(Issue("strategy", msg, blocks="deploy"))
    return issues


def can_deploy(setup_issues: list[Issue], run_issues: list[Issue]) -> bool:
    """Deploy is allowed only when neither Setup nor Run has blocking issues."""
    return not setup_issues and not run_issues
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/gui/tune/test_validation.py -k "preflight or can_deploy" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/tune/_validation.py tests/unit/gui/tune/test_validation.py
git commit -m "feat(gui-tune): pre-flight (grid+float) + deploy gate"
```

---

## Task 3: Image source resolution (pure)

The new-run image dir comes from the hub's shared source-image-root, with a
per-run override. **Use a new module** (`_run_image_source.py`) rather than the
existing `_image_source.py`, which is the Curate-view plate picker — a different
concern. The real `resolve_source_image_root(sandbox, payload)` takes a sandbox
**and** the store payload (verified at `shell/_source_context.py:91`), so the
wrapper must thread the sandbox through.

**Files:**
- Create: `src/phenotypic/gui/tune/_run_image_source.py`
- Test: `tests/unit/gui/tune/test_run_image_source.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/gui/tune/test_run_image_source.py
from phenotypic.gui.tune._run_image_source import resolve_run_images


def test_override_wins_when_set():
    assert resolve_run_images(sandbox=None, store_payload=None, override="/explicit") == "/explicit"


def test_falls_back_to_shared_root(monkeypatch):
    # No override → resolve from the shared-source store payload (sandbox-bounded).
    monkeypatch.setattr(
        "phenotypic.gui.tune._run_image_source.resolve_source_image_root",
        lambda sandbox, payload: "/shared/imgs",
    )
    assert resolve_run_images(
        sandbox=object(), store_payload={"path": "/shared/imgs"}, override=None
    ) == "/shared/imgs"


def test_none_when_neither_available(monkeypatch):
    monkeypatch.setattr(
        "phenotypic.gui.tune._run_image_source.resolve_source_image_root",
        lambda sandbox, payload: None,
    )
    assert resolve_run_images(sandbox=object(), store_payload=None, override=None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/gui/tune/test_run_image_source.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

```python
# src/phenotypic/gui/tune/_run_image_source.py
"""Resolve the ``-i`` images dir for a tune run.

Precedence: an explicit per-run override wins; otherwise resolve the hub's
shared source-image-root (sandbox-bounded) from its store payload.
"""
from __future__ import annotations

from phenotypic.gui.shell._source_context import resolve_source_image_root


def resolve_run_images(*, sandbox, store_payload, override):
    """Return the images dir string, or ``None`` when neither source resolves.

    Args:
        sandbox: The frozen sandbox boundary (required by
            ``resolve_source_image_root(sandbox, payload)``).
        store_payload: The value from ``SHELL_SOURCE_IMAGE_ROOT_STORE``.
        override: An explicit per-run image dir (wins when set).
    """
    if override:
        return override
    resolved = resolve_source_image_root(sandbox, store_payload)
    return str(resolved) if resolved else None
```

The deploy callback (Task 5) threads its `sandbox` (from
`app.server.config[CFG_SANDBOX_ROOT]` / the bound `SandboxRoot`) into this.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/gui/tune/test_run_image_source.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/tune/_run_image_source.py tests/unit/gui/tune/test_run_image_source.py
git commit -m "feat(gui-tune): resolve run images from shared source-root + override"
```

---

## Task 4: Wire the Run view (Dash)

**Files:**
- Modify: `src/phenotypic/gui/tune/_ids.py` (Run-view IDs: strategy/sampler/budget/seed inputs, advanced inputs, Local/SLURM toggle, image-source row, output picker, storage input, command-preview, pre-flight banner, deploy button, footer)
- Modify: `src/phenotypic/gui/tune/_layout.py` (`build_layout`: the Run destination body)
- Modify: `src/phenotypic/gui/tune/_callbacks.py` (pre-flight banner + command preview + footer gating)
- Modify: `src/phenotypic/gui/tune/_command.py` (preview parity for every argv flag)
- Test: `tests/integration/gui/tune/test_run_deploy.py` (pre-flight + gate half)

- [ ] **Step 1: Write the failing integration test (pre-flight half)**

```python
# tests/integration/gui/tune/test_run_deploy.py
"""Run-view pre-flight + deploy gating. Mirrors the tune integration harness."""

def test_grid_with_continuous_float_disables_deploy(tune_run_state):
    tune_run_state.set_strategy("grid")          # active sigma is continuous float
    assert tune_run_state.deploy_disabled is True
    assert "continuous float" in tune_run_state.preflight_text.lower()


def test_optuna_clears_preflight_and_enables_deploy(tune_run_state):
    tune_run_state.set_strategy("optuna")
    assert tune_run_state.deploy_disabled is False
```

Build `tune_run_state` on the existing integration fixture (copy from a sibling
in `tests/integration/gui/tune/`). The assertions: strategy flip toggles the
pre-flight banner + the deploy-disabled state via `preflight_issues`/`can_deploy`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/gui/tune/test_run_deploy.py -k preflight -v`
Expected: FAIL — Run view not wired.

- [ ] **Step 3: Implement the Run layout + non-deploy callbacks**

In `build_layout`, render the Run destination body:
- **Strategy & budget**: strategy dropdown (`grid`/`random`/`optuna`), sampler
  dropdown (shown only for optuna), ASHA pruning checkbox, trials, max-failures,
  seed. The trials field shows a runtime estimate caption.
- **Advanced** (collapsed): held-out fraction, group key, stability λ, rung floor.
- **Compute target & output**: Local/SLURM radio (SLURM reveals partition/mem/time);
  **image source** row rendering `resolve_run_images(...)` (the shared root +
  a `build_output_picker_modal`-style override picker); output-dir picker;
  workers; storage URL; SLURM fields.
- **Command preview**: a read-only block fed by
  `render_launch_command(spec_path, input_dir, output_dir, strategy=…, n_trials=…,
  storage_url=…, n_workers=…, slurm_partition=…, slurm_mem=…,
  slurm_time=…, held_out_fraction=…, cv_group=…, screen=…, slurm=…)`.

Callbacks (no deploy yet): on strategy/space change, compute
`preflight_issues(space, strategy=…)` → render the banner + set the deploy
button's `disabled` from `can_deploy(setup_issues, run_issues)`; on any field
change, re-render the command preview via `render_launch_command`. The preview
must stay in parity with `tune_run_argv`; add a unit test that compares
`shlex.split(render_launch_command(...))` to `tune_run_argv(..., python="python")`
for the shared flags.

Lazy `optuna`: this view must not import optuna at module load — only the deploy
callback (next task) touches the engine, and even there via the CLI subprocess.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/gui/tune/test_run_deploy.py -k preflight -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/tune/_ids.py src/phenotypic/gui/tune/_layout.py \
        src/phenotypic/gui/tune/_callbacks.py tests/integration/gui/tune/test_run_deploy.py
git commit -m "feat(gui-tune): Run view (strategy/budget/compute, image source, pre-flight, command preview)"
```

---

## Task 5: Wire Deploy (Dash) — launch + auto-advance

**Files:**
- Modify: `src/phenotypic/gui/tune/_app.py` (`create_app` gains `runner=None`, `registry=None` params and sets `app.server.config[CFG_RUNNER]` / `[CFG_RUN_REGISTRY]`, mirroring `run_console/_app.py:103-104`)
- Modify: `src/phenotypic/gui/shell/_app.py` (the hub composer passes the **shared** runner + registry into `tune.create_app(runner=…, registry=…)` — the same instances it already gives run-console, so a tune run and a pipeline run share one registry)
- Modify: `src/phenotypic/gui/tune/_callbacks.py` (deploy callback)
- Test: `tests/integration/gui/tune/test_run_deploy.py` (deploy half, mocked runner)

**Why injection is required:** the tune app is a *separate* Flask app behind
`DispatcherMiddleware`. `app.server.config[CFG_RUNNER]` inside a tune callback
reads the **tune** app's config, which is empty unless `create_app` sets it. So
`create_app` must accept and store the shared instances, and the shell composer
must pass them — exactly as run-console does.

- [ ] **Step 1: Write the failing test (deploy half)**

```python
# tests/integration/gui/tune/test_run_deploy.py  (add)
def test_deploy_writes_spec_builds_argv_and_starts_runner(tune_run_state, fake_runner):
    tune_run_state.set_strategy("optuna")
    tune_run_state.deploy()
    # canonical run copy written with the typed suffix
    assert tune_run_state.written_spec_path.name == "tuning_spec.json.pht-tune"
    # runner started with a tune argv
    argv = fake_runner.last_start.argv
    assert argv[:4] == [argv[0], "-m", "phenotypic.tune", "run"]
    # auto-advanced to Monitor + counter bumped
    assert tune_run_state.active_destination == "monitor"
    assert tune_run_state.live_runs == 1


def test_deploy_blocked_when_invalid_is_a_noop(tune_run_state, fake_runner):
    tune_run_state.set_strategy("grid")  # continuous-float conflict → blocked
    tune_run_state.deploy()
    assert fake_runner.last_start is None
    assert tune_run_state.active_destination == "run"
```

`fake_runner` is a stub exposing `last_start` (records `start(run_id, argv,
output_dir=…)`); inject it via `app.server.config[CFG_RUNNER]` in the fixture.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/gui/tune/test_run_deploy.py -k deploy -v`
Expected: FAIL — deploy callback not wired.

- [ ] **Step 3: Implement the deploy callback**

In `register_callbacks`, add the Deploy callback. Logic (hard-guarded):

1. Re-validate: `setup = validate_setup(space, scorer_kind=…, metadata_present=…)`;
   `run = preflight_issues(space, strategy=…)`. If `not can_deploy(setup, run)`,
   return without launching (no-op; the button is already disabled, this is the
   belt-and-braces guard).
2. Assemble the authored `TuningSpec` from the Setup state and write a launch
   input spec under the GUI tune scratch/library area (not under
   `deliverables/`). The CLI owns the canonical resolved run copy at
   `deliverables/tuning_spec.json.pht-tune`; it writes that after applying
   strategy/budget/held-out argv overrides.
3. Resolve images: `images = resolve_run_images(store_payload=…, override=…)`.
4. Build argv: `argv = tune_run_argv(spec_path=…, images_dir=images,
   output_dir=…, strategy=…, n_trials=…, storage_url=…, n_workers=…,
   slurm_partition=…, slurm_mem=…, slurm_time=…, held_out_fraction=…,
   cv_group=…, slurm=(target=="slurm"), screen=…)`.
5. Launch — **both targets use `LocalRunner.start`**, because the tune CLI's own
   `--slurm` flag (already appended by `tune_run_argv` when `slurm=True`) makes
   the subprocess *submit the SLURM worker fleet itself* (`phenotypic.tune run
   … --slurm` → `sbatch`), then exit. So there is **no** dependency on
   `run_console/_slurm.submit_slurm` (whose signature takes a `RunConsoleState`
   and would not fit). Concretely:
   - `runner = app.server.config[CFG_RUNNER]`;
     `runner.start(run_id, argv, output_dir=Path(output_dir))`.
   - For SLURM, the subprocess is short-lived (dispatches sbatch then returns).
     v1 does **not** parse or persist the array job id because GUI-side SLURM
     cancellation is deferred.
   - Register the run in the shared `RunRegistry`
     (`app.server.config[CFG_RUN_REGISTRY]`) as a `RunRecord` with
     `mode` ("local"/"slurm") and `output_dir`, so Monitor (Plan 4) renders the
     right live view even when the study store is unreachable.
6. Bump the live-runs `dcc.Store(storage_type="session")` and set the active
   destination to `monitor` (auto-advance).

Import `optuna` nowhere here — deploy shells out to the CLI subprocess, which
owns the engine (and, for SLURM, the sbatch submission).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/gui/tune/test_run_deploy.py -k deploy -v`
Expected: PASS.

- [ ] **Step 5: Update `FEATURES.md` + `WORKFLOWS.md`**

- `FEATURES.md`: rows for strategy/sampler/pruning/budget/seed inputs, advanced
  inputs, Local/SLURM toggle, image-source row + override picker, output picker,
  storage input, command preview, pre-flight banner, **Deploy button** — each
  with a `Test ref`.
- **Save / open-from-library (D12).** Each FEATURES row needs a backing test, so
  add two tiny pure helpers (TDD, before the buttons): `save_spec_to_library(
  spec, *, sandbox, name) -> Path` (writes to `tune_presets_dir(sandbox)/name`
  via `spec.to_json` + `ensure_typed_json_suffix`) and `list_library_specs(
  sandbox) -> list[Path]` (globs the library, matched with
  `matches_any_suffix(p, TUNING_CONFIG_SUFFIXES)` so both `.json.pht-tune` and
  legacy `.json` show). Wire the **Save spec** button → `save_spec_to_library`,
  **Open from library** → `list_library_specs` + `TuningSpec.model_validate_json`,
  and **Browse…** → the directory-picker modal. Put the helpers in
  `tune/_spec_library.py` with `tests/unit/gui/tune/test_spec_library.py`. (Only
  then add the corresponding FEATURES rows — the `features-md-gate` requires a
  real `Test ref` per row.)
- `WORKFLOWS.md`: add the end-to-end **author → deploy → monitor** flow row. This
  REQUIRES a matching `_capture_tune_author_deploy` function in
  `scripts/capture_gui_tutorial_screenshots.py` and a walkthrough page under
  `docs/source/tutorials/gui/`. Run
  `uv run python scripts/capture_gui_tutorial_screenshots.py` and commit the full
  refreshed PNG set (font-render churn across other tutorials is expected — commit
  them all, do not cherry-pick).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/tune/_callbacks.py src/phenotypic/gui/tune/_app.py \
        tests/integration/gui/tune/test_run_deploy.py \
        src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md \
        scripts/capture_gui_tutorial_screenshots.py docs/source/tutorials/gui \
        docs/source/tutorials/gui/_screenshots 2>/dev/null
git commit -m "feat(gui-tune): Deploy a tuning run (local/SLURM) + auto-advance to Monitor"
```

---

## Task 6: Module-wide verification

- [ ] **Step 1:** `uv run pytest tests/unit/gui/tune tests/integration/gui/tune -q` → PASS.
- [ ] **Step 2:** `uv run mypy src/phenotypic/gui/tune` → no new errors.
- [ ] **Step 3:** `uv run ruff check --fix src/phenotypic/gui/tune tests/unit/gui/tune tests/integration/gui/tune` → clean.
- [ ] **Step 4:** Lazy-optuna guard still holds:
  `uv run python -c "import sys, phenotypic.gui.tune; assert 'optuna' not in sys.modules"` → no error.
- [ ] **Step 5:** Commit fixups.

```bash
git add -A src/phenotypic/gui/tune tests
git commit -m "test(gui-tune): run/deploy verification fixups" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage (doc 03):** Run sections (strategy/budget/advanced/compute) → Task 4. Image source from shared source-root + override → Task 3 (+wiring 4). Command preview via `render_launch_command` → Task 4. Pre-flight grid×float → Task 2. Deploy writes an authored launch spec, builds a full CLI argv, launches local/SLURM, registers mode/output in `RunRegistry`, and auto-advances with a session-store counter → Task 5. Runner imported in place (no move) → stated in Architecture (D13/Q3). Lazy optuna → Tasks 4,5 + verification 6. SLURM degrade-when-unreachable & the Monitor live views, run switcher, Local cancel, single-best + **Pareto** export → **Plan 4**. Save-to-library button → wired here (FEATURES row) using `tune_presets_dir` (Plan 2); load/library browser detail can extend in Plan 4 polish.

**Placeholder scan:** Pure tasks (1–3) have complete code. Wiring tasks (4,5) reference the established integration harness + name exact helpers/signatures to call — no "TODO". The `fake_runner`/`tune_run_state` fixtures are described concretely (record `start` args; inject via `CFG_RUNNER`) and follow the run-console test pattern.

**Type consistency:** `tune_run_argv(**kwargs) -> list[str]` consumed by the deploy callback (Task 5) exactly as defined (Task 1). `Issue`/`can_deploy`/`preflight_issues` consistent with Plan 2's `Issue` dataclass. `resolve_run_images(store_payload, override)` consistent across test + impl + callback. `tuning_spec_path`/`render_launch_command`/`LocalRunner.start(run_id, argv, *, output_dir)`/`submit_slurm` used per their real signatures (verified in source).

---

## Execution Handoff

Plan 3 of 4. Plan 4 (Monitor: run switcher, Local/SLURM live view + degrade,
Local cancel, single-best + Pareto export) consumes the `RunRegistry` entries
this plan writes. Recommended: subagent-driven, fresh agent per task, review
between tasks.
