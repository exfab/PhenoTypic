# Phase 1 — `.phenotypic` Machine-State Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate the CLI's machine-state sidecars (`progress/`, `processing_state.json`, `processing_events.log`) from the output-dir root into a single hidden `<output>/.phenotypic/` directory, for both the forward CLI and (later) process-only mode, with backward compatibility for pre-migration runs.

**Architecture:** `phenotypic.tools_._io_constants` is the single source of truth for artifact paths. Re-root the *machine-state* path helpers under a new `.phenotypic/` cache dir; convert the ~40 hand-joined call sites across the CLI/dashboard/tools to use the helpers; add `resolve_*` read-fallback helpers + a one-time `migrate_legacy_machine_state` so legacy runs resume and stay GUI-discoverable. Forward-run behavior is unchanged except where state lives on disk.

**Tech Stack:** Python 3.12, `uv` (runner/package manager), `click` CLI, `pytest`, Dash GUI. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-03-cli-process-only-and-phenotypic-cache-design.md` (§5.1, §5.2; decisions D1, D6, D7, D11, D14).

**Refinements over the spec discovered while planning (apply these):**
- Writers use the **pure** re-rooted helpers (always `.phenotypic/`); the CLI calls `migrate_legacy_machine_state(output_dir)` once before resuming/operating on an existing dir so a pre-migration run is moved in (no split-brain), rather than writing new state beside legacy state.
- GUI/read-only consumers use `resolve_*` (first-existing) and never mutate a run dir.
- The GUI **classifier is unaffected** by this phase (`is_cli_output` keys on `results/` + `deliverables/`, both unchanged); the only GUI edits are status/manifest reads switching to `resolve_manifest_json_path`, plus the mandatory `FEATURES.md` row.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/phenotypic/tools_/_io_constants.py` | `DIR_PHENOTYPIC`, `phenotypic_cache_dir`, re-rooted `progress_dir`/`processing_state_path`/`event_log_path`, `resolve_*` readers, `migrate_legacy_machine_state` | Modify |
| `src/phenotypic/tools_/__init__.py` | Export the new symbols | Modify |
| `src/phenotypic/_cli/_cli_state_management.py` | Route save/load/event through helpers + `migrate_legacy_machine_state` on load | Modify |
| `src/phenotypic/_cli/*.py`, `_cli/_dashboard/*.py` | Convert hand-joined `output_dir / DIR_PROGRESS` etc. to helpers | Modify (≈12 files) |
| `src/phenotypic/tools_/generate_report.py`, `tools_/monitor_slurm_jobs.py` | Read state via `resolve_*` | Modify |
| `src/phenotypic/gui/shell/_runs_registry.py`, `gui/run_console/_recent_runs.py` | Read status via `resolve_manifest_json_path` | Modify |
| `src/phenotypic/gui/_config.py` | Re-export `DIR_PHENOTYPIC` as `PHENOTYPIC_CACHE_DIRNAME` | Modify |
| `src/phenotypic/gui/FEATURES.md` | CI-gated row for the state-location change | Modify |
| `tests/unit/tools_/test_io_constants.py` | Layout + resolver + migration + grep-gate tests | Modify |
| `tests/integration/cli/` | Legacy-resume + new-layout regression | Modify/Create |

---

## Task 1: `.phenotypic` constants, re-rooted helpers, resolvers, migrator

**Files:**
- Modify: `src/phenotypic/tools_/_io_constants.py`
- Modify: `src/phenotypic/tools_/__init__.py`
- Test: `tests/unit/tools_/test_io_constants.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/tools_/test_io_constants.py`:

```python
from pathlib import Path

from phenotypic.tools_ import (
    DIR_PHENOTYPIC,
    phenotypic_cache_dir,
    progress_dir,
    processing_state_path,
    event_log_path,
    manifest_json_path,
    resolve_progress_dir,
    resolve_processing_state_path,
    resolve_manifest_json_path,
    migrate_legacy_machine_state,
    deliverables_dir,
    results_dir,
    qc_dir,
    logs_dir,
    slurm_scripts_dir,
)


class TestPhenotypicCacheLayout:
    def test_machine_state_roots_under_phenotypic(self):
        out = Path("/tmp/run")
        assert DIR_PHENOTYPIC == ".phenotypic"
        assert phenotypic_cache_dir(out) == out / ".phenotypic"
        assert progress_dir(out) == out / ".phenotypic" / "progress"
        assert processing_state_path(out) == out / ".phenotypic" / "processing_state.json"
        assert event_log_path(out) == out / ".phenotypic" / "processing_events.log"
        # manifest composes from progress_dir, so it follows the re-root
        assert manifest_json_path(out) == out / ".phenotypic" / "progress" / "manifest.json"

    def test_user_facing_dirs_unchanged(self):
        out = Path("/tmp/run")
        assert deliverables_dir(out) == out / "deliverables"
        assert results_dir(out) == out / "results"
        assert qc_dir(out) == out / "qc"
        assert logs_dir(out) == out / "logs"
        assert slurm_scripts_dir(out) == out / "slurm_scripts"


class TestBackCompatResolvers:
    def test_resolver_prefers_new_then_legacy_then_new_default(self, tmp_path):
        out = tmp_path
        # Neither exists -> default to new location
        assert resolve_processing_state_path(out) == processing_state_path(out)
        assert resolve_progress_dir(out) == progress_dir(out)
        # Legacy exists, new does not -> resolver returns legacy
        legacy_state = out / "processing_state.json"
        legacy_state.write_text("{}", encoding="utf-8")
        (out / "progress").mkdir()
        assert resolve_processing_state_path(out) == legacy_state
        assert resolve_progress_dir(out) == out / "progress"
        assert resolve_manifest_json_path(out) == out / "progress" / "manifest.json"
        # New exists -> new wins over legacy
        processing_state_path(out).parent.mkdir(parents=True, exist_ok=True)
        processing_state_path(out).write_text("{}", encoding="utf-8")
        progress_dir(out).mkdir(parents=True, exist_ok=True)
        assert resolve_processing_state_path(out) == processing_state_path(out)
        assert resolve_progress_dir(out) == progress_dir(out)


class TestMigrateLegacyMachineState:
    def test_moves_legacy_into_cache_dir(self, tmp_path):
        out = tmp_path
        (out / "progress").mkdir()
        (out / "progress" / "manifest.json").write_text("{}", encoding="utf-8")
        (out / "processing_state.json").write_text("{}", encoding="utf-8")
        (out / "processing_events.log").write_text("x\n", encoding="utf-8")
        moved = migrate_legacy_machine_state(out)
        assert moved is True
        assert (out / ".phenotypic" / "progress" / "manifest.json").is_file()
        assert (out / ".phenotypic" / "processing_state.json").is_file()
        assert (out / ".phenotypic" / "processing_events.log").is_file()
        assert not (out / "progress").exists()
        assert not (out / "processing_state.json").exists()

    def test_noop_when_already_migrated(self, tmp_path):
        out = tmp_path
        progress_dir(out).mkdir(parents=True)
        assert migrate_legacy_machine_state(out) is False

    def test_noop_when_nothing_present(self, tmp_path):
        assert migrate_legacy_machine_state(tmp_path) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/tools_/test_io_constants.py -k "PhenotypicCache or BackCompat or MigrateLegacy" -v`
Expected: FAIL with `ImportError: cannot import name 'DIR_PHENOTYPIC'` (symbols not defined yet).

- [ ] **Step 3: Implement constants, helpers, resolvers, migrator**

In `src/phenotypic/tools_/_io_constants.py`, add the directory constant near `DIR_DELIVERABLES`:

```python
#: ``<output>/.phenotypic/`` — hidden machine-state cache root. Holds the
#: run's progress/, processing_state.json, and processing_events.log. Hidden
#: so it does not clutter the user-facing output folder and is skipped by the
#: GUI's run/dataset candidate scan. NOTE: distinct from the GUI's
#: ``.phenotypic-gui`` sandbox dir (presets/state) — different root, different
#: purpose.
DIR_PHENOTYPIC: Final[str] = ".phenotypic"
```

Add the cache-dir helper and **re-root** the three machine-state helpers (replace the existing `progress_dir`, `processing_state_path`, `event_log_path` bodies):

```python
def phenotypic_cache_dir(output_dir: Path) -> Path:
    """Return ``<output>/.phenotypic/`` — the hidden machine-state root.

    Pure path expression; callers ``mkdir`` when they intend to write.
    """
    return output_dir / DIR_PHENOTYPIC


def progress_dir(output_dir: Path) -> Path:
    """Return ``<output>/.phenotypic/progress/``."""
    return phenotypic_cache_dir(output_dir) / DIR_PROGRESS


def event_log_path(output_dir: Path) -> Path:
    """Return ``<output>/.phenotypic/processing_events.log``."""
    return phenotypic_cache_dir(output_dir) / PROCESSING_EVENTS_LOG


def processing_state_path(output_dir: Path) -> Path:
    """Return ``<output>/.phenotypic/processing_state.json``."""
    return phenotypic_cache_dir(output_dir) / PROCESSING_STATE_JSON
```

Add the back-compat readers + migrator (place after `processing_state_path`):

```python
def _legacy_progress_dir(output_dir: Path) -> Path:
    """Pre-migration location: ``<output>/progress/``."""
    return output_dir / DIR_PROGRESS


def _legacy_processing_state_path(output_dir: Path) -> Path:
    """Pre-migration location: ``<output>/processing_state.json``."""
    return output_dir / PROCESSING_STATE_JSON


def resolve_progress_dir(output_dir: Path) -> Path:
    """Return the progress dir that exists, preferring ``.phenotypic/``.

    Read-only helper for resume/discovery so a pre-migration run (progress
    at the output root) is still found. Falls back to the new location when
    neither exists (the default for fresh writes).
    """
    new = progress_dir(output_dir)
    if new.exists():
        return new
    legacy = _legacy_progress_dir(output_dir)
    if legacy.exists():
        return legacy
    return new


def resolve_processing_state_path(output_dir: Path) -> Path:
    """Return the processing-state file that exists, preferring ``.phenotypic/``."""
    new = processing_state_path(output_dir)
    if new.exists():
        return new
    legacy = _legacy_processing_state_path(output_dir)
    if legacy.exists():
        return legacy
    return new


def resolve_manifest_json_path(output_dir: Path) -> Path:
    """Return ``<progress>/manifest.json`` resolving the progress dir for legacy runs."""
    return resolve_progress_dir(output_dir) / MANIFEST_JSON


def migrate_legacy_machine_state(output_dir: Path) -> bool:
    """Move a pre-migration run's machine-state into ``.phenotypic/`` once.

    If ``<output>/.phenotypic/`` does not yet exist but legacy machine-state
    (``progress/``, ``processing_state.json``, ``processing_events.log``) is
    present at the output root, move those into the cache dir so the run
    proceeds coherently against a single location. Idempotent: a no-op once
    ``.phenotypic/`` exists or when no legacy state is present.

    Returns:
        ``True`` if anything was moved, else ``False``.
    """
    import shutil

    cache = phenotypic_cache_dir(output_dir)
    if cache.exists():
        return False
    legacy_progress = _legacy_progress_dir(output_dir)
    legacy_state = _legacy_processing_state_path(output_dir)
    legacy_events = output_dir / PROCESSING_EVENTS_LOG
    if not (legacy_progress.exists() or legacy_state.exists() or legacy_events.exists()):
        return False
    cache.mkdir(parents=True, exist_ok=True)
    if legacy_progress.exists():
        shutil.move(str(legacy_progress), str(cache / DIR_PROGRESS))
    if legacy_state.exists():
        shutil.move(str(legacy_state), str(cache / PROCESSING_STATE_JSON))
    if legacy_events.exists():
        shutil.move(str(legacy_events), str(cache / PROCESSING_EVENTS_LOG))
    return True
```

In `src/phenotypic/tools_/__init__.py`, add to the import block from `._io_constants` and to `__all__`: `DIR_PHENOTYPIC`, `phenotypic_cache_dir`, `resolve_progress_dir`, `resolve_processing_state_path`, `resolve_manifest_json_path`, `migrate_legacy_machine_state`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/tools_/test_io_constants.py -k "PhenotypicCache or BackCompat or MigrateLegacy" -v`
Expected: PASS (all 7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tools_/_io_constants.py src/phenotypic/tools_/__init__.py tests/unit/tools_/test_io_constants.py
git commit -m "feat(io): add .phenotypic cache dir, re-root machine-state helpers, back-compat resolvers"
```

---

## Task 2: Route `_cli_state_management` through helpers + migrate-on-load

**Files:**
- Modify: `src/phenotypic/_cli/_cli_state_management.py:17,37,79,92,190`
- Test: `tests/unit/cli/test_cli_state_management.py` (create if absent)

- [ ] **Step 1: Write the failing test**

Create/append `tests/unit/cli/test_cli_state_management.py`:

```python
from datetime import datetime
from pathlib import Path

from phenotypic._cli._cli_state_management import save_processing_state, load_processing_state
from phenotypic._cli._cli_types import ProcessingState
from phenotypic.tools_ import processing_state_path


def _make_state(out: Path) -> ProcessingState:
    now = datetime(2026, 6, 3, 12, 0, 0)
    return ProcessingState(
        version="2", pipeline_path=Path("p.json"), input_path=Path("in"),
        output_dir=out, timestamp=now, execution_mode="local",
        last_updated=now, datasets={}, config={},
    )


def test_save_writes_under_phenotypic(tmp_path):
    save_processing_state(_make_state(tmp_path), tmp_path)
    assert processing_state_path(tmp_path).is_file()
    assert not (tmp_path / "processing_state.json").exists()


def test_load_migrates_and_reads_legacy_run(tmp_path):
    # Simulate a pre-migration run: state at the output root.
    save_processing_state(_make_state(tmp_path), tmp_path)
    new = processing_state_path(tmp_path)
    legacy = tmp_path / "processing_state.json"
    (tmp_path / ".phenotypic").rename(tmp_path / "_tmp")  # extract
    (tmp_path / "_tmp" / "processing_state.json").rename(legacy)
    (tmp_path / "_tmp").rmdir()
    assert legacy.is_file() and not new.exists()
    loaded = load_processing_state(tmp_path)
    assert loaded is not None
    # migrate-on-load moved it into .phenotypic
    assert new.is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_cli_state_management.py -v`
Expected: FAIL — `test_save_writes_under_phenotypic` fails because save still writes to the root.

- [ ] **Step 3: Apply the edits**

In `src/phenotypic/_cli/_cli_state_management.py`:

Line 17 — extend the import:
```python
from phenotypic.tools_ import (
    PROCESSING_STATE_JSON, PROCESSING_EVENTS_LOG, ProcessingStateKey,
    processing_state_path, event_log_path,
    resolve_processing_state_path, resolve_progress_dir,
    migrate_legacy_machine_state,
)
```

Line 37 (`save_processing_state`) — replace and ensure the parent dir exists:
```python
    state_file = processing_state_path(output_dir)
    state_file.parent.mkdir(parents=True, exist_ok=True)
```

Lines 79 & 92 (`load_processing_state`) — migrate first, then read the resolved paths:
```python
    migrate_legacy_machine_state(output_dir)
    state_file = resolve_processing_state_path(output_dir)
```
and
```python
    event_log = resolve_progress_dir(output_dir).parent / PROCESSING_EVENTS_LOG
```
(`resolve_progress_dir(...).parent` is the cache dir for a migrated/new run, the output root for a not-yet-migrated read — keeping the event log a sibling of `progress/`, per D14.)

Line 190 (in `get_datasets_with_remaining_images` or sibling) — replace `output_dir / PROCESSING_EVENTS_LOG` with:
```python
    event_log = resolve_progress_dir(output_dir).parent / PROCESSING_EVENTS_LOG
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/cli/test_cli_state_management.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_state_management.py tests/unit/cli/test_cli_state_management.py
git commit -m "feat(cli): route processing-state IO through .phenotypic helpers + migrate-on-load"
```

---

## Task 3: Convert remaining hand-joined writer sites to helpers

These are mechanical: replace each hand-joined machine-state path with the matching helper. **Re-run the audit grep first — line numbers drift.**

Run: `grep -rn --include='*.py' -E '/ ?"progress"|/ ?PROCESSING_STATE_JSON|/ ?"processing_state\.json"|/ ?PROCESSING_EVENTS_LOG|/ ?"processing_events\.log"|/ ?DIR_PROGRESS' src/phenotypic | grep -v _io_constants.py | grep -v '/sweep/'`

- [ ] **Step 1: Add a failing grep-gate test**

Append to `tests/unit/tools_/test_io_constants.py`:

```python
import subprocess


class TestNoHandJoinedStatePaths:
    def test_machine_state_paths_only_in_io_constants(self):
        """No module outside _io_constants (sweep excepted) hand-joins
        machine-state paths; everything must go through the helpers."""
        pattern = r'/ ?"progress"|/ ?PROCESSING_STATE_JSON|/ ?"processing_state\.json"|/ ?PROCESSING_EVENTS_LOG|/ ?"processing_events\.log"|/ ?DIR_PROGRESS'
        proc = subprocess.run(
            ["grep", "-rn", "--include=*.py", "-E", pattern, "src/phenotypic"],
            capture_output=True, text=True,
        )
        offenders = [
            ln for ln in proc.stdout.splitlines()
            if "_io_constants.py" not in ln
            and "/sweep/" not in ln          # D11: sweep intentionally excluded
            and "checkpoint_handler" not in ln  # progress_dir.parent / EVENTS — correct by design (D14)
        ]
        assert offenders == [], "Hand-joined machine-state paths remain:\n" + "\n".join(offenders)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tools_/test_io_constants.py -k NoHandJoined -v`
Expected: FAIL listing the offender lines (the inventory below).

- [ ] **Step 3: Convert each site**

For every offender, apply the corresponding replacement. Import the helper(s) at the top of each file from `phenotypic.tools_`.

Replacement rules:
- `output_dir / DIR_PROGRESS`  →  `progress_dir(output_dir)`
- `output_dir / PROCESSING_STATE_JSON`  →  `processing_state_path(output_dir)`
- `output_dir / PROCESSING_EVENTS_LOG`  →  `event_log_path(output_dir)`
- `output_dir / DIR_PROGRESS / DIR_RECOMPILE / X`  →  `recompile_dir(progress_dir(output_dir)) / X`
- `output_dir / DIR_PROGRESS / DIR_RECOMPILE / DIR_RECOMPILE_SHARDS`  →  `recompile_dir(progress_dir(output_dir)) / DIR_RECOMPILE_SHARDS`

Per-file checklist (write path = pure helper):
- [ ] `phenotypicCLI.py` — `output_dir / PROCESSING_STATE_JSON` (×2) → `processing_state_path(output_dir)`; `output_dir / DIR_PROGRESS` (×2) → `progress_dir(output_dir)`; the recompile-manifest and other `… / DIR_PROGRESS / …` joins → compose from `progress_dir(output_dir)` / `recompile_dir(...)`.
- [ ] `_cli_execution_strategies.py` — `event_log = output_dir / PROCESSING_EVENTS_LOG` (×2) → `event_log_path(output_dir)`; `progress_dir = output_dir / DIR_PROGRESS` (×3) → `progress_dir(output_dir)` (rename the local var to avoid shadowing the import, e.g. `prog_dir = progress_dir(output_dir)`).
- [ ] `_cli_recompile_worker.py` — the `… / DIR_PROGRESS` and shard joins → `progress_dir(output_dir)` / `recompile_dir(progress_dir(output_dir)) / DIR_RECOMPILE_SHARDS`.
- [ ] `_cli_process_single.py:465` — `progress_dir = output_dir / DIR_PROGRESS` → `progress_dir(output_dir)` (rename local var).
- [ ] `_cli_output_manager.py:303` — `output_dir / PROCESSING_STATE_JSON` → `processing_state_path(output_dir)`.
- [ ] `_cli_chunk_writer.py:67` — `output_dir / DIR_PROGRESS` → `progress_dir(output_dir)`.
- [ ] `_cli_checkpoint_handler.py:48` — `output_dir / DIR_PROGRESS` → `progress_dir(output_dir)`. **Leave L200** (`progress_dir.parent / PROCESSING_EVENTS_LOG`) as-is — correct once `progress_dir` is the re-rooted value (D14).
- [ ] `_cli_slurm_scripts.py:187`, `_cli_slurm_array_scripts.py:213` — `output_dir / PROCESSING_EVENTS_LOG` → `event_log_path(output_dir)`.
- [ ] `_cli_recompile_slurm_scripts.py:117` — `output_dir / DIR_PROGRESS / DIR_RECOMPILE` → `recompile_dir(progress_dir(output_dir))`.
- [ ] `_dashboard/_manifest_builder.py:311,469` — `output_dir / PROCESSING_EVENTS_LOG` → `event_log_path(output_dir)`.
- [ ] `_dashboard/_analysis_data.py:52`, `_dashboard/_generator.py:121,145` — `output_dir / DIR_PROGRESS` → `progress_dir(output_dir)` (rename local var).

> The grep-gate test allowlists `checkpoint_handler` (L200 is intentional) and `/sweep/` (D11). If your conversion leaves an unrelated match in `checkpoint_handler`, tighten the edit — only L200's `progress_dir.parent` form is permitted there.

- [ ] **Step 4: Run the gate + the existing CLI unit suite**

Run: `uv run pytest tests/unit/tools_/test_io_constants.py -k NoHandJoined -v`
Expected: PASS (offenders == []).
Run: `uv run pytest tests/unit/cli -q`
Expected: PASS (no regressions; these are pure path-helper swaps).

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/tools_/test_io_constants.py
git commit -m "refactor(cli): route all machine-state writers through .phenotypic helpers"
```

---

## Task 4: Convert `tools_` reporters to resolver reads

**Files:**
- Modify: `src/phenotypic/tools_/generate_report.py:50,118`
- Modify: `src/phenotypic/tools_/monitor_slurm_jobs.py:55,56`
- Test: `tests/unit/tools_/test_io_constants.py` (covered by the grep gate) + a focused read test

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/tools_/test_io_constants.py`:

```python
class TestReporterReadsResolve:
    def test_generate_report_finds_state_in_phenotypic(self, tmp_path):
        from phenotypic.tools_ import processing_state_path, event_log_path
        processing_state_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        processing_state_path(tmp_path).write_text(
            '{"version":"2","datasets":{}}', encoding="utf-8"
        )
        event_log_path(tmp_path).write_text("", encoding="utf-8")
        from phenotypic.tools_.generate_report import _load_state_for_report  # see Step 3
        state = _load_state_for_report(tmp_path)
        assert state is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/tools_/test_io_constants.py -k ReporterReads -v`
Expected: FAIL — reporter still reads `output_dir / "processing_state.json"` (root), or `_load_state_for_report` doesn't exist.

- [ ] **Step 3: Apply the edits**

In `src/phenotypic/tools_/generate_report.py`, replace the two hand-joins (and extract a tiny helper the test targets):
```python
from phenotypic.tools_ import resolve_processing_state_path, resolve_progress_dir, PROCESSING_EVENTS_LOG

def _load_state_for_report(output_dir):
    import json
    state_file = resolve_processing_state_path(output_dir)
    if not state_file.is_file():
        return None
    return json.loads(state_file.read_text(encoding="utf-8"))
```
and replace `event_log = output_dir / "processing_events.log"` with
`event_log = resolve_progress_dir(output_dir).parent / PROCESSING_EVENTS_LOG`,
and `state_file = output_dir / "processing_state.json"` with
`state_file = resolve_processing_state_path(output_dir)`.

In `src/phenotypic/tools_/monitor_slurm_jobs.py`, replace both root joins:
```python
from phenotypic.tools_ import resolve_processing_state_path, resolve_progress_dir, PROCESSING_EVENTS_LOG
...
    event_log = resolve_progress_dir(output_dir).parent / PROCESSING_EVENTS_LOG
    state_file = resolve_processing_state_path(output_dir)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/tools_/test_io_constants.py -k "ReporterReads or NoHandJoined" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tools_/generate_report.py src/phenotypic/tools_/monitor_slurm_jobs.py tests/unit/tools_/test_io_constants.py
git commit -m "refactor(tools): reporters read run state via .phenotypic resolvers"
```

---

## Task 5: GUI status/manifest reads via resolver + FEATURES.md

**Files:**
- Modify: `src/phenotypic/gui/shell/_runs_registry.py:41,297`
- Modify: `src/phenotypic/gui/run_console/_recent_runs.py` (any manifest read)
- Modify: `src/phenotypic/gui/_config.py` (add `PHENOTYPIC_CACHE_DIRNAME`)
- Modify: `src/phenotypic/gui/FEATURES.md`
- Test: `tests/unit/gui/test_runs_registry_layout.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/gui/test_runs_registry_layout.py`:

```python
from phenotypic.tools_ import manifest_json_path
from phenotypic.gui.shell._runs_registry import RunRegistry  # adjust to real class/path


def _write_manifest(path, payload='{"is_complete": true, "execution_mode": "local"}'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_status_read_finds_new_layout(tmp_path):
    _write_manifest(manifest_json_path(tmp_path))
    mode, status, _ = RunRegistry._read_status_from_manifest(tmp_path)
    assert status != "unknown"


def test_status_read_finds_legacy_layout(tmp_path):
    _write_manifest(tmp_path / "progress" / "manifest.json")
    mode, status, _ = RunRegistry._read_status_from_manifest(tmp_path)
    assert status != "unknown"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/gui/test_runs_registry_layout.py -v`
Expected: `test_status_read_finds_legacy_layout` FAILs (uses `manifest_json_path`, which now points only at `.phenotypic/`).

- [ ] **Step 3: Apply the edits**

In `_runs_registry.py`: change the import `manifest_json_path` → `resolve_manifest_json_path` (line 41) and the call site (line 297) `manifest_path = resolve_manifest_json_path(output_dir)`. Apply the same swap to any `manifest_json_path` / `read_run_manifest` call in `run_console/_recent_runs.py`.

In `_config.py`, add `PHENOTYPIC_CACHE_DIRNAME: str = DIR_PHENOTYPIC` (import `DIR_PHENOTYPIC` from `phenotypic.tools_`) next to `PROGRESS_DIRNAME`.

In `FEATURES.md`, update/add the row describing where run state is read from (machine-state now under `.phenotypic/`, status read via `resolve_manifest_json_path` with legacy fallback), with `Test ref` = `tests/unit/gui/test_runs_registry_layout.py::test_status_read_finds_legacy_layout`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/gui/test_runs_registry_layout.py -v`
Expected: PASS (both new + legacy).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/shell/_runs_registry.py src/phenotypic/gui/run_console/_recent_runs.py src/phenotypic/gui/_config.py src/phenotypic/gui/FEATURES.md tests/unit/gui/test_runs_registry_layout.py
git commit -m "feat(gui): read run status via .phenotypic resolver (legacy fallback) + FEATURES row"
```

---

## Task 6: Integration regression — new-layout run + legacy-resume

**Files:**
- Test: `tests/integration/cli/test_phenotypic_cache_layout.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/integration/cli/test_phenotypic_cache_layout.py` (model the fixture/invocation on the existing `tests/integration/cli/` runner, e.g. `test_cli_hdf_output.py`):

```python
import json
from pathlib import Path

from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.tools_ import progress_dir, processing_state_path


def test_forward_run_writes_state_under_phenotypic(tmp_path, synth_plate_dir, simple_pipeline_json):
    out = tmp_path / "out"
    res = CliRunner().invoke(phenotypic_cli, [
        "--pipeline", str(simple_pipeline_json),
        "--input", str(synth_plate_dir),
        "--output-dir", str(out),
        "--force-local", "--n-jobs", "1",
    ])
    assert res.exit_code == 0, res.output
    assert progress_dir(out).is_dir()
    assert not (out / "progress").exists()          # not at the root anymore
    assert (out / "deliverables").exists()           # user-facing dirs unchanged
    assert (out / "results").exists()


def test_resume_of_legacy_layout_migrates_and_completes(tmp_path, synth_plate_dir, simple_pipeline_json):
    out = tmp_path / "out"
    CliRunner().invoke(phenotypic_cli, [
        "--pipeline", str(simple_pipeline_json), "--input", str(synth_plate_dir),
        "--output-dir", str(out), "--force-local", "--n-jobs", "1",
    ])
    # Simulate a pre-migration layout by moving state back to the root.
    cache = out / ".phenotypic"
    (cache / "progress").rename(out / "progress")
    (cache / "processing_state.json").rename(out / "processing_state.json")
    (cache / "processing_events.log").rename(out / "processing_events.log")
    cache.rmdir()
    res = CliRunner().invoke(phenotypic_cli, [
        "--pipeline", str(simple_pipeline_json), "--input", str(synth_plate_dir),
        "--output-dir", str(out), "--force-local", "--n-jobs", "1", "--resume",
    ])
    assert res.exit_code == 0, res.output
    assert processing_state_path(out).is_file()       # migrated into .phenotypic
    assert not (out / "processing_state.json").exists()
```

> If `synth_plate_dir` / `simple_pipeline_json` fixtures don't exist, add them to `tests/integration/cli/conftest.py` mirroring an existing integration test's fixtures (write `load_synth_yeast_plate()` to a tmp dir; serialize a minimal `ImagePipeline` with one detector via `pipeline.to_json()`).

- [ ] **Step 2: Run it to verify it fails (or errors on missing fixtures)**

Run: `uv run pytest tests/integration/cli/test_phenotypic_cache_layout.py -v`
Expected: PASS once Tasks 1–5 are in (this task is the end-to-end safety net; if it fails, a writer site was missed — re-run the Task 3 grep gate).

- [ ] **Step 3: (No new impl)** — if it fails, fix the missed site, not the test.

- [ ] **Step 4: Re-run**

Run: `uv run pytest tests/integration/cli/test_phenotypic_cache_layout.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/cli/test_phenotypic_cache_layout.py tests/integration/cli/conftest.py
git commit -m "test(cli): new .phenotypic layout + legacy-resume migration regression"
```

---

## Task 7: Full verification + lint/type pass

- [ ] **Step 1: Type check**

Run: `uv run mypy src/phenotypic`
Expected: no new errors introduced by this phase (compare against a pre-change baseline if needed).

- [ ] **Step 2: Lint/format**

Run: `uv run ruff check --fix`
Expected: clean.

- [ ] **Step 3: Run the CLI + GUI + tools test suites**

Run: `uv run pytest tests/unit/tools_ tests/unit/cli tests/unit/gui tests/integration/cli -q`
Expected: PASS. Investigate any test asserting `<output>/processing_state.json` or `<output>/progress/...` at the root — update it to the helper/`.phenotypic` location (these are expected breakages from the move).

- [ ] **Step 4: Commit any test fixups**

```bash
git add -A
git commit -m "test: update root-level state-path assertions to .phenotypic layout"
```

---

## Self-Review (completed)

- **Spec coverage:** D1/D7 (migrate progress + state + events) → Tasks 1–4. D6 (re-root + read-fallback) → Task 1 resolvers + migrator. D11 (sweep excluded) → grep-gate allowlist. D14 (event log sibling) → Task 1 + leave `checkpoint_handler:200`. §5.2 GUI reads → Task 5. Back-compat resume/discovery → Tasks 2,5,6.
- **Placeholder scan:** none — every code step shows the code; mechanical edits are exact old→new with a verifying grep gate.
- **Type consistency:** helper names used downstream (`progress_dir`, `processing_state_path`, `event_log_path`, `resolve_*`, `migrate_legacy_machine_state`) match Task 1 definitions and the existing signatures (`output_dir: Path`).
