# Private GUI Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `src/phenotypic/gui` to `src/phenotypic/_gui` and make the `phenotypic-gui` console script the only user entry to the hub.

**Architecture:** One atomic, scripted move (`git mv` + a word-bounded textual rewrite + a short list of hand edits for spellings no regex can see). That is followed by two disjoint follow-ups: the entry-point seam (delete the hub `__main__`, route every hub launch through the console script) and the documentation sweep (user docs, contributor docs, API reference removal).

**Tech Stack:** Python 3.12, uv, pytest (+ xdist, pytest-qt, Playwright), Dash, Sphinx.

**Spec:** `docs/superpowers/specs/2026-09-10-private-gui-module/spec.md`

**Plan review:** `docs/superpowers/reports/2026-09-10-private-gui-module/plan-review.md` (all findings applied in this revision)

## Global Constraints

- `uv` only — never bare `python`/`pip`.
- The env re-sync after touching `[project.scripts]` matches CI and keeps every installed extra: `uv sync --group dev --group test-qt --group docs --all-extras`. `uv sync` is exact — a narrower extras list uninstalls optuna/torch/sam2/transformers/gudhi/psycopg and turns surface tests into silent skips. Diff `uv pip list` before and after.
- The textual rewrite excludes `docs/superpowers/**` (spec D4) and `docs/source/api_reference/gui/**` (deleted in Task 3).
- The "Must not change" runtime paths in the spec (`phenotypic/gui/viewer_cache`, `.phenotypic/logs/gui`, `.phenotypic-gui`, the `phenotypic-gui` script name, test directory names) stay byte-identical.
- `ruff check --fix` only with explicit paths.
- Pytest invocations: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg`, `-o addopts= -m "not slow"`, explicit `-n`, never `-x` for a measurement (`run-phenotypic-test` skill).
- A test that cannot run must fail, not skip.
- Each task's implementer commits its own work. Stage by explicit path — a scoped `git add -A <pathspecs>` only where a step says so, never an unscoped `git add -A` or `git commit -a` — and check `git diff --cached --stat` before committing. Every commit message ends with:

  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01GxzwZcwTtBDMiX1Tk5Dzg5
  ```

- Tasks run strictly one at a time (one checkout, one git index).

## Execution DAG

```
Task 0 (baseline) → Task 1 (Keystone: the move) → Task 2 (Seam: entry points) → Task 3 (Sweep: docs) → Task 4 (gates) → Task 5 (review, simplify, regression)
```

Tasks 2 and 3 share no files, but they still run one after the other: one checkout, one git index. Model: Tasks 1, 2 and all review gates on the session (Opus-tier) model; Task 3 on a Sonnet-tier model at medium effort.

## Test surface (derived mechanically)

```bash
{ git grep -lE "phenotypic\.gui|phenotypic/gui|phenotypic\._gui|phenotypic/_gui|_cli_error_outputs|reemit_error_deliverables|install_smart_grid|napari_pipeline_viewer" -- tests
  git ls-files tests/unit/gui tests/integration/gui tests/gui
  printf '%s\n' tests/unit/schema/test_no_metadata_literals.py tests/unit/test_ome_zarr_invariants.py \
    tests/integration/packaging/test_package_contents.py tests/integration/cli/test_finalize_qc.py
} | grep -E '/test_[^/]*\.py$' | grep -v '^tests/e2e/' | sort -u | while read -r f; do [ -f "$f" ] && echo "$f"; done > /tmp/private-gui-surface.txt

QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest $(cat /tmp/private-gui-surface.txt) \
  -q --no-header -p no:randomly -p no:cacheprovider -o addopts= -m "not slow" -n 6 \
  --junitxml=/tmp/private-gui-<label>.xml
```

---

### Task 0: Baseline (orchestrator)

**Files:**
- Create: `docs/superpowers/reports/2026-09-10-private-gui-module/baseline.md`

- [x] **Step 1:** mypy `src/phenotypic` → `Found 418 errors in 121 files`; ruff `src/phenotypic scripts tests` → `Found 65 errors`.
- [x] **Step 2:** Run the test surface with `<label>=baseline` on the untouched branch; record the summary line and the failing node IDs in `baseline.md`. → 262 files: `3009 passed, 16 skipped, 3 xfailed`, no failures.
- [x] **Step 3:** Commit the spec, this plan, and `baseline.md`. → `676ae0d14`.

---

### Task 1: Move the package (Keystone)

**Files:**
- Create: `tests/unit/gui/test_private_package.py`
- Move: `src/phenotypic/gui/` → `src/phenotypic/_gui/` (all tracked files)
- Modify (textual rewrite): every tracked text file matching `phenotypic[./]gui`, excluding `docs/superpowers/**` and `docs/source/api_reference/gui/**`
- Modify (hand edits — spellings invisible to the rewrite):
  - `pyproject.toml:253-255`
  - `scripts/check_features_md.py:31`
  - `scripts/check_workflows_md.py:41`
  - `tests/integration/packaging/test_package_contents.py:40`
  - `tests/unit/gui/shell/test_source_chrome.py:286`
  - `tests/e2e/gui/test_tune_launch_command.py:13-16`
  - `tests/unit/schema/test_no_metadata_literals.py:37-44`
  - `tests/unit/test_ome_zarr_invariants.py:150`
  - `tests/unit/gui/test_optional_deps.py:25`

**Interfaces:**
- Produces: package `phenotypic._gui` (same submodule tree as `phenotypic.gui` today); console script `phenotypic-gui = "phenotypic._gui.shell._launcher:main"`; test file `tests/unit/gui/test_private_package.py` (Task 2 appends to it).

- [ ] **Step 1: Write the failing guard test** — `tests/unit/gui/test_private_package.py`:

```python
"""The GUI ships as the private ``phenotypic._gui`` package.

Users start the hub only through the ``phenotypic-gui`` console script. The
public ``phenotypic.gui`` import path is gone; the five sub-app debug launchers
remain for contributors.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

SUB_APP_DEBUG_LAUNCHERS = ("analysis", "browse", "builder", "results_viewer", "run_console")


def test_public_gui_import_path_is_gone() -> None:
    """``phenotypic.gui`` must not resolve -- not even as a namespace package."""
    assert importlib.util.find_spec("phenotypic.gui") is None


def test_private_gui_package_resolves() -> None:
    assert importlib.util.find_spec("phenotypic._gui") is not None


@pytest.mark.parametrize("sub_app", SUB_APP_DEBUG_LAUNCHERS)
def test_sub_app_debug_launchers_remain(sub_app: str) -> None:
    assert importlib.util.find_spec(f"phenotypic._gui.{sub_app}.__main__") is not None


def test_console_script_targets_private_launcher() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'phenotypic-gui = "phenotypic._gui.shell._launcher:main"' in pyproject
```

- [ ] **Step 2: Run it — expect 8 failures** (the old path resolves; `_gui` does not, so its `find_spec` calls raise `ModuleNotFoundError`; pyproject still names `phenotypic.gui`):

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/test_private_package.py -q -o addopts= -p no:cacheprovider
```

- [ ] **Step 3: Move the tracked tree**

```bash
git mv src/phenotypic/gui src/phenotypic/_gui
```

- [ ] **Step 4: Confirm nothing is left at the old path, and clear the bytecode-only directories** — `git mv` renames the directory on disk, so ignored contents (`__pycache__`, `.DS_Store`) move with it. Confirm the old path is gone (expected: exit 0, no output):

```bash
test ! -e src/phenotypic/gui
```

Three directories that moved hold only bytecode from long-deleted modules (`sweep`, `_shared/timeline`, `results_viewer/timeline_view`), and each would resolve as a namespace package. Verify they contain nothing outside `__pycache__` (expected output: nothing), then remove them:

```bash
for d in src/phenotypic/_gui/sweep src/phenotypic/_gui/_shared/timeline src/phenotypic/_gui/results_viewer/timeline_view; do
  find "$d" -type f -not -path '*/__pycache__/*'
done
rm -rf src/phenotypic/_gui/sweep src/phenotypic/_gui/_shared/timeline src/phenotypic/_gui/results_viewer/timeline_view
```

If the `find` loop prints anything, stop and report it — do not delete.

- [ ] **Step 5: Textual rewrite** (word-bounded; `phenotypic-gui` is untouched because `-` is not in `[./]`):

```bash
git grep -lIz -E 'phenotypic[./]gui' -- . ':!docs/superpowers' ':!docs/source/api_reference/gui' \
  | xargs -0 perl -pi -e 's/\bphenotypic([.\/])gui\b/phenotypic${1}_gui/g'
```

- [ ] **Step 6: Hand edits**
  - `pyproject.toml`: `"gui/**/*.css"`, `"gui/**/*.js"`, `"gui/**/*.png"` → `"_gui/**/*.css"`, `"_gui/**/*.js"`, `"_gui/**/*.png"`.
  - `scripts/check_features_md.py:31`, `scripts/check_workflows_md.py:41`, `tests/integration/packaging/test_package_contents.py:40`, `tests/unit/gui/shell/test_source_chrome.py:286`: `/ "phenotypic" / "gui"` → `/ "phenotypic" / "_gui"`.
  - `tests/e2e/gui/test_tune_launch_command.py`: the `/ "gui"` line directly after `/ "phenotypic"` → `/ "_gui"`.
  - `tests/unit/schema/test_no_metadata_literals.py`: the four `_ALLOWED` keys `"gui/results_viewer/_curation_labels.py"`, `"gui/results_viewer/_compatibility.py"`, `"gui/shell/_metadata_context.py"`, `"gui/results_viewer/_scatter_tab/_facets.py"` → `"_gui/…"`.
  - `tests/unit/test_ome_zarr_invariants.py:150`: `"gui/builder/_preview_cache.py"` → `"_gui/builder/_preview_cache.py"`.
  - `tests/unit/gui/test_optional_deps.py:25`: `from phenotypic import gui` → `from phenotypic import _gui as gui` (attribute-style import; it contains no `phenotypic.gui` substring for the rewrite to match).

- [ ] **Step 7: Prove nothing was missed**

```bash
git grep -nIE 'phenotypic[./]gui([^_a-zA-Z0-9]|$)|from phenotypic import gui' -- . ':!docs/superpowers' ':!docs/source/api_reference/gui'
```
Expected: no output.

```bash
git grep -lE "[\"']gui[\"'/]" -- '*.py' pyproject.toml
```
Expected: **exactly** these 13 files. Each is a runtime path from the spec's "Must not change" list, or one of the two generator scripts deleted in Task 3:
- `scripts/generate_dispatch_reference.py`, `scripts/generate_validation_reference.py` (deleted in Task 3)
- `src/phenotypic/_gui/results_viewer/_output_root.py` (viewer cache root)
- `src/phenotypic/_gui/run_console/_callbacks.py`
- `src/phenotypic/_gui/run_console/_slurm.py`
- `src/phenotypic/_gui/run_console/_slurm_observer.py`
- `src/phenotypic/phenotypicCLI.py` (`gui_logs.name != "gui"`)
- `tests/integration/gui/test_run_console_callbacks.py`
- `tests/unit/cli/test_cli_output_freshness.py`
- `tests/unit/gui/run_console/test_slurm.py`
- `tests/unit/gui/run_console/test_slurm_observer.py`
- `tests/unit/gui/test_check_workflows_md.py` (`tutorials / "gui"`)
- `tests/unit/gui/test_viewer_cache_ownership.py`

Any other file is a missed package path: fix it and add it to your report.

- [ ] **Step 8: Re-sync the env so the console script is regenerated**

```bash
uv pip list > /tmp/private-gui-pip-before.txt
uv sync --group dev --group test-qt --group docs --all-extras
uv pip list | diff /tmp/private-gui-pip-before.txt -                  # expect: no package added or removed
grep -c "phenotypic._gui.shell._launcher" .venv/bin/phenotypic-gui    # expect 1
uv run phenotypic-gui --help | head -3                                # expect "usage: phenotypic-gui"
```

- [ ] **Step 9: Guard test green** — rerun the Step 2 command; expect 8 passed.

- [ ] **Step 10: Mutation proof** — `mkdir -p src/phenotypic/gui/__pycache__` and rerun; `test_public_gui_import_path_is_gone` must FAIL (the namespace-package trap). Then `rm -rf src/phenotypic/gui` and confirm it passes again.

- [ ] **Step 11: Task surface** — run the test surface with `<label>=task1`. Every failure must either be in the baseline (there are none) or be attributed by rerunning it alone. Compare the skip count against the baseline's 16; any new skip needs its reason. Report the summary line.

- [ ] **Step 12: Commit** — the move spans ~470 files, so a scoped `git add -A` over these pathspecs is required here. First confirm nothing outside them changed (expected: no output):

```bash
git status --short -- . ':!src/phenotypic' ':!tests' ':!scripts' ':!tools' ':!pyproject.toml' ':!.github' ':!.pre-commit-config.yaml' ':!.claude/skills' ':!README.md' ':!CLAUDE.md' ':!NOTICE' ':!docs/diagrams' ':!docs/source'
git add -A src/phenotypic tests scripts tools pyproject.toml .github .pre-commit-config.yaml .claude/skills README.md CLAUDE.md NOTICE docs/diagrams docs/source
```

Commit with subject `refactor(gui): move phenotypic.gui to the private phenotypic._gui package` and the trailer from Global Constraints.

---

### Task 2: Console script is the only hub entry (Seam)

**Files:**
- Delete: `src/phenotypic/_gui/__main__.py`
- Modify:
  - `src/phenotypic/_gui/__init__.py` (module docstring)
  - `src/phenotypic/_gui/shell/_launcher.py` (docstring lines 1-21)
  - `src/phenotypic/_gui/run_console/__main__.py` (docstring line 6)
  - `src/phenotypic/_gui/FEATURES.md` ("Entry points" and "Documentation" tables)
  - `tests/e2e/gui/conftest.py` (`_start_live_server`)
  - `scripts/capture_gui_tutorial_screenshots.py` (`boot_gui`)
- Test:
  - `tests/unit/gui/test_private_package.py`
  - `tests/integration/gui/test_console_script.py`

**Interfaces:**
- Consumes: `phenotypic._gui`, the regenerated console script, and `tests/unit/gui/test_private_package.py` from Task 1.
- Produces: `_phenotypic_gui_executable() -> str`, a private helper defined separately in `tests/e2e/gui/conftest.py` and `scripts/capture_gui_tutorial_screenshots.py` (the script is standalone and imports nothing from `tests/`). It looks only in `sysconfig.get_path("scripts")`, never on `PATH`: seven other worktree venvs in this repo install their own `phenotypic-gui`, each importing its own checkout.

- [ ] **Step 1: Failing test** — append to `tests/unit/gui/test_private_package.py`:

```python
def test_hub_has_no_module_entry() -> None:
    """``python -m phenotypic._gui`` is not a way to start the hub."""
    assert importlib.util.find_spec("phenotypic._gui.__main__") is None
```

Run the file; expect exactly this test to fail.

- [ ] **Step 2: Make the console-script test require the script** — in `tests/integration/gui/test_console_script.py`, add `import sysconfig` to the imports and replace `_phenotypic_gui_argv`:

```python
def _phenotypic_gui_argv() -> list[str]:
    """Return argv for ``phenotypic-gui``, the hub's only entry point.

    Looks only in this interpreter's scripts directory -- never ``PATH`` -- so
    the test exercises this checkout's install. A missing script fails the
    test: there is no module fallback to hide behind.
    """
    scripts_dir = sysconfig.get_path("scripts")
    binary = shutil.which("phenotypic-gui", path=scripts_dir)
    if binary is None:
        pytest.fail(f"phenotypic-gui console script is not installed in {scripts_dir}")
    return [binary]
```

- [ ] **Step 3: Delete the hub module entry**

```bash
git rm src/phenotypic/_gui/__main__.py
```

- [ ] **Step 4: Route the e2e fixture through the console script** — in `tests/e2e/gui/conftest.py`, add `import shutil` and `import sysconfig` to the imports if absent, and add this helper above `_start_live_server`:

```python
def _phenotypic_gui_executable() -> str:
    """Return the ``phenotypic-gui`` console script of the running environment.

    Looks only in this interpreter's scripts directory -- never ``PATH`` -- so
    the hub boots the checkout under test rather than another environment's
    install, and raises rather than skipping when the script is missing.
    """
    scripts_dir = sysconfig.get_path("scripts")
    found = shutil.which("phenotypic-gui", path=scripts_dir)
    if found is None:
        raise RuntimeError(
            f"phenotypic-gui console script not found in {scripts_dir}; run "
            "`uv sync --group dev --group test-qt --all-extras`"
        )
    return found
```

and replace the first three elements of `cmd`:

```python
    cmd = [
        _phenotypic_gui_executable(),
        "--root",
        str(sandbox),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
```

- [ ] **Step 5: Same change in `scripts/capture_gui_tutorial_screenshots.py::boot_gui`** — add `import sysconfig` to the imports (the script already imports `shutil`), add the identical `_phenotypic_gui_executable` helper above `boot_gui`, and replace `sys.executable, "-m", "phenotypic._gui",` with `_phenotypic_gui_executable(),`. The two standalone-viewer captures (`-m phenotypic._gui.analysis`, `-m phenotypic._gui.results_viewer`) stay — spec D1 keeps those launchers.

- [ ] **Step 6: Docstrings**
  - `src/phenotypic/_gui/__init__.py` — replace the module docstring's first paragraph with: `"""Private implementation of the PhenoTypic GUI hub.` + blank line + `Not a public API: import paths under ``phenotypic._gui`` may change without notice. Users start the hub with the ``phenotypic-gui`` console script. Components are lazy-loaded so that optional dependencies (Dash, napari) are only imported when used.`. Keep the "Sub-packages" / "Utilities" sections, with `gui.` → `_gui.`.
  - `src/phenotypic/_gui/shell/_launcher.py` — `:func:`launch_gui`` bullet: "programmatic boot used by downstream tests" (drop "the ``__main__`` module and"); `:func:`main`` bullet: "argparse front-end wired into ``[project.scripts]`` (``phenotypic-gui = phenotypic._gui.shell._launcher:main``), the hub's only entry point." (drop the `python -m` clause).
  - `src/phenotypic/_gui/run_console/__main__.py` — "The unified hub entry point is ``python -m phenotypic._gui``." → "The unified hub entry point is the ``phenotypic-gui`` console script."

- [ ] **Step 7: FEATURES.md** (the ledger must describe the entry points as they now are)
  - Delete the row `| \`python -m phenotypic._gui\` | Module entry | Argparse …`.
  - Row `| \`phenotypic-gui\` console script |` — Expected behaviour → `Sole hub entry point; \`--help\` + \`--root\` work`.
  - Documentation table, row `CLAUDE.md update` — Expected behaviour → `Names \`phenotypic-gui\` as the only hub entry and lists the private sub-app debug launchers`.

- [ ] **Step 8: Green + seam checks**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/test_private_package.py tests/integration/gui/test_console_script.py -q -o addopts= -p no:cacheprovider
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
```
Expected: 12 passed; both checkers exit 0.

- [ ] **Step 9: Mutation proof** — temporarily make `_phenotypic_gui_argv` return `[sys.executable, "-m", "phenotypic._gui"]`; `test_phenotypic_gui_help_succeeds` must FAIL. Revert.

- [ ] **Step 10: Boot the hub through the new e2e path** (live seam; Playwright chromium is installed locally):

```bash
PLAYWRIGHT=1 QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -m "not ci_flaky" -q -o addopts= -p no:cacheprovider -n 4
```
Without `PLAYWRIGHT=1`, `tests/e2e/gui/conftest.py:50` skips every test and the run reads green (CI sets it at `gui-checks.yml:208`). Report the summary line. To attribute a failure, rerun it alone on this branch **and** on `main`:

```bash
git worktree add /tmp/pht-main main
(cd /tmp/pht-main && uv sync --group dev --group test-qt --group docs --all-extras)   # a fresh worktree has no venv
```

Leave `/tmp/pht-main` in place; Task 4 reuses it.

- [ ] **Step 11: Capture-script boot smoke**

```bash
uv run python - <<'PY'
import sys, tempfile
from pathlib import Path
sys.path.insert(0, "scripts")
from capture_gui_tutorial_screenshots import boot_gui, shutdown_gui
proc, url = boot_gui(Path(tempfile.mkdtemp()))
print("ready", url)
shutdown_gui(proc)
PY
```
Expected: `[gui] booting: …/phenotypic-gui --root …` then `ready http://127.0.0.1:<port>`.

- [ ] **Step 12: Commit** — stage only this task's files by name (the Step 3 `git rm` is already staged):

```bash
git add tests/unit/gui/test_private_package.py tests/integration/gui/test_console_script.py tests/e2e/gui/conftest.py scripts/capture_gui_tutorial_screenshots.py src/phenotypic/_gui/__init__.py src/phenotypic/_gui/shell/_launcher.py src/phenotypic/_gui/run_console/__main__.py src/phenotypic/_gui/FEATURES.md
git diff --cached --stat    # expect exactly these 8 paths plus the deleted src/phenotypic/_gui/__main__.py
```

Commit with subject `refactor(gui): make phenotypic-gui the only hub entry point` and the trailer from Global Constraints.

---

### Task 3: Documentation (Sweep)

**Files:**
- Delete:
  - `docs/source/api_reference/gui/` (15 files)
  - `scripts/generate_dispatch_reference.py`
  - `scripts/generate_validation_reference.py`
  - `scripts/_reference_generator.py`
  - `tests/unit/gui/test_reference_generators.py`
- Modify:
  - `docs/source/api_reference/index.rst:36-43`
  - `README.md:178-212`
  - `docs/source/tutorials/getting_started.rst:53,274-280,314`
  - `docs/source/how_to/pages/gui_hub.md:27-32,143,290,316-333`
  - `docs/source/tutorials/gui/02_file_explorer.md:27`
  - `docs/source/tutorials/gui/06_view_results.md:20-33`
  - `docs/source/tutorials/gui/08_analysis.md:23-30`
  - `CLAUDE.md:237-241,249,348`
  - `src/phenotypic/_gui/CLAUDE.md` (top-level note, plus its `gui/…` shorthand)
- Modify (package-relative `gui/…` shorthand in prose and comments, Step 11; the list is derived by grep, 28 files at `a19478769`): `.github/workflows/package-integrity.ci.yml`, `DESIGN.md`, `docs/source/contrib_guide/tracked_state.md`, `pyproject.toml` (a comment), `src/phenotypic/_gui/FEATURES.md`, `src/phenotypic/_gui/_param_forms.py`, `src/phenotypic/_gui/_shared/tiles.py`, `src/phenotypic/_gui/builder/_param_form.py`, `src/phenotypic/_gui/builder/assets/builder.css`, `src/phenotypic/_gui/results_viewer/_assets/results_viewer.css`, `src/phenotypic/_gui/results_viewer/_output_root.py`, `src/phenotypic/_gui/results_viewer/_picker_navigation.py`, `src/phenotypic/_gui/shell/_assets/shell.css`, `src/phenotypic/_gui/shell/_runs_registry.py`, `src/phenotypic/sdk_/_verification_cache.py`, and ten test files (comments and docstrings only), plus the two CLAUDE.md files above

- [ ] **Step 1: Confirm the deletions have no other consumers** — expected: no output.

```bash
git grep -nE "generate_(dispatch|validation)_reference|_reference_generator|api_reference/gui" -- ':!docs/superpowers' ':!docs/source/api_reference/gui' ':!scripts/generate_dispatch_reference.py' ':!scripts/generate_validation_reference.py' ':!scripts/_reference_generator.py' ':!tests/unit/gui/test_reference_generators.py'
```

(The toctree entry in `docs/source/api_reference/index.rst` spells the page `gui/index`, not `api_reference/gui`, so it does not show up here; Step 3 removes it.)

Also record N, the number of tests collected from the test file Step 2 deletes (Step 12 needs it):

```bash
QT_QPA_PLATFORM=offscreen uv run pytest --collect-only -q -o addopts= -p no:cacheprovider tests/unit/gui/test_reference_generators.py | tail -1
```

- [ ] **Step 2: Delete**

```bash
git rm -r docs/source/api_reference/gui scripts/generate_dispatch_reference.py scripts/generate_validation_reference.py scripts/_reference_generator.py tests/unit/gui/test_reference_generators.py
```

- [ ] **Step 3: `docs/source/api_reference/index.rst`** — remove the block:

```rst
GUI internals
-------------

.. toctree::
   :maxdepth: 1

   gui/index

```

- [ ] **Step 4: `README.md`** — replace

````markdown
under one URL. Two equivalent entry points:

```bash
# Console script (preferred)
uv run phenotypic-gui --root ./images --port 8050

# Module entry (works in environments without the console script on PATH)
uv run python -m phenotypic._gui --root ./images --port 8050
```
````

with

````markdown
under one URL. Start it with the `phenotypic-gui` console script:

```bash
uv run phenotypic-gui --root ./images --port 8050
```
````

and replace "`phenotypic-gui` or `python -m phenotypic._gui`." with "`phenotypic-gui`."

- [ ] **Step 5: `docs/source/tutorials/getting_started.rst`**
  - Lines 52-53: the `sweep` viewer was deleted long ago (no tracked module, and no napari sweep code under `src/phenotypic`). Replace these two lines

    ```rst
      Required for ``image.rgb.napari()`` and related viewer methods, the point
      picker, and the napari sweep viewer (``python -m phenotypic._gui.sweep``).
    ```

    with

    ```rst
      Required for ``image.rgb.napari()`` and related viewer methods, and the point
      picker.
    ```
  - Lines 276-280: keep only `   uv run phenotypic-gui --root ./images --port 8050` in the code block (drop both comments and the module line).
  - Line 314: "Use ``phenotypic-gui`` or ``python -m phenotypic._gui``." → "Use ``phenotypic-gui``."

- [ ] **Step 6: `docs/source/how_to/pages/gui_hub.md`**
  - "Two equivalent entry points boot the same server:" → "The `phenotypic-gui` console script boots the hub:" and drop the `uv run python -m phenotypic._gui …` line from that code block.
  - Warning block: "Always use the hyphenated form `phenotypic-gui` or the module form `python -m phenotypic._gui`." → "Always use the hyphenated form `phenotypic-gui`."
  - Slurm walkthrough: `uv run python -m phenotypic._gui --root <project-dir> --port 8050` → `uv run phenotypic-gui --root <project-dir> --port 8050`.
  - Delete the whole `## Standalone tools` section (the heading through "…the same defaults as the hub launcher."). Spec D1: the debug launchers are contributor-only.

- [ ] **Step 7: `docs/source/tutorials/gui/06_view_results.md`** — replace

````markdown
evidence, you may inspect it but mutation controls for QC, Error, curation,
Analysis, rebuild, and publication stay disabled. Two other ways to get a
populated viewer are:

1. **Standalone launch** (recommended for now). Run
   `phenotypic._gui.results_viewer` directly with `--output-root` pointing
   at the CLI output:

   ```bash
   uv run python -m phenotypic._gui.results_viewer \
       --output-root gui_tutorial_dataset/results --port 8051
   ```

2. **Open `deliverables/dashboard.html`**
````

with

````markdown
evidence, you may inspect it but mutation controls for QC, Error, curation,
Analysis, rebuild, and publication stay disabled. You can also:

1. **Open `deliverables/dashboard.html`**
````

- [ ] **Step 8: The other two tutorials**
  - `docs/source/tutorials/gui/08_analysis.md` — delete the note block:

    ````markdown
    ```{note}
    The standalone launcher is still useful for headless workflows or
    long-running fits where you don't need the rest of the hub:

        uv run python -m phenotypic._gui.analysis \
            --root <path-to-cli-output> --port 8051
    ```

    ````

  - `docs/source/tutorials/gui/02_file_explorer.md:27` — replace ``Every entry is run through a small classifier (`phenotypic._gui.shell._classifier`)`` with ``Every entry is run through a small classifier``. User docs never name private modules (spec criterion 6).

- [ ] **Step 9: Root `CLAUDE.md`** — replace the three bullets

```markdown
- `uv run python -m phenotypic._gui --root ./images` — equivalent module entry.
- Standalone tools still work: `python -m phenotypic._gui.builder`,
  `python -m phenotypic._gui.results_viewer`, `python -m phenotypic._gui.run_console`.
- Note: `phenotypic gui` (no hyphen, as a subcommand of the existing CLI) is NOT
  supported. Use `phenotypic-gui` or `python -m phenotypic._gui`.
```

with

```markdown
- `phenotypic-gui` is the hub's **only** entry point. The GUI is the private
  package `phenotypic._gui` — no public import path, no `python -m` form of
  the hub.
- Sub-app debug launchers (contributors only, not in user docs):
  `uv run python -m phenotypic._gui.{builder,results_viewer,run_console,browse,analysis}`.
- Note: `phenotypic gui` (no hyphen, as a subcommand of the existing CLI) is NOT
  supported. Use `phenotypic-gui`.
```

and change both link texts `[gui/CLAUDE.md](src/phenotypic/_gui/CLAUDE.md)` → `[_gui/CLAUDE.md](src/phenotypic/_gui/CLAUDE.md)`.

- [ ] **Step 10: `src/phenotypic/_gui/CLAUDE.md`** — directly under the title, add:

```markdown
> **Private package.** `phenotypic._gui` is not a public API. Users start the
> hub only with the `phenotypic-gui` console script (`shell/_launcher.py:main`).
> The per-sub-app `__main__.py` launchers are contributor debugging tools; do
> not document them in user-facing docs, and do not add a `__main__.py` to the
> package root.
```

- [ ] **Step 11: Package-relative `gui/…` shorthand** — prose and comments still point at `gui/<subpath>`, which no longer exists (for example the file:line citation ``gui/builder/_point_picker.py:549`` at `src/phenotypic/sdk_/_verification_cache.py:104`). Task 1's rewrite and greps could not see these, because they have no `phenotypic` prefix. At `a19478769` the grep below finds 69 lines in 28 files, all package pointers and none a runtime path (a dry run of this exact substitution on committed copies changed 69 lines, left no match, and doubled no prefix). Record the file list, rewrite, then verify:

```bash
SUBS='builder|shell|results_viewer|run_console|browse|analysis|tune|sweep|_shared|_smart_grid|_config\.py|_design\.py|_operation_registry\.py|_param_forms\.py|_plot_refresh\.py|_schema_cache\.py|_snapshot_status\.py|_url_prefix\.py|_async_binding_client\.py|_binding_generation\.py|CLAUDE\.md|FEATURES\.md|WORKFLOWS\.md|__init__\.py|__main__\.py'
git grep -lE "(^|[^_./a-zA-Z0-9-])gui/($SUBS)" -- . ':!docs/superpowers' > /tmp/private-gui-shorthand-files.txt
wc -l < /tmp/private-gui-shorthand-files.txt   # expect 28, or 27 once Step 9 has changed the root CLAUDE.md link labels
tr '\n' '\0' < /tmp/private-gui-shorthand-files.txt | xargs -0 perl -pi -e "s{(^|[^_./a-zA-Z0-9-])gui/($SUBS)}{\${1}_gui/\${2}}g"
git grep -nE "(^|[^_./a-zA-Z0-9-])gui/($SUBS)" -- . ':!docs/superpowers'   # expect: no output
git diff --stat -- $(cat /tmp/private-gui-shorthand-files.txt)            # expect small per-file line counts, no whole-file rewrites
```

The pattern cannot touch runtime paths: those are built from separate string components (`"phenotypic" / "gui"`, `"logs" / "gui"`) or are followed by names outside the list (`tutorials/gui/01_setup.md`, `tests/unit/gui/…`).

- [ ] **Step 12: Verify** — both greps expected to print nothing:

```bash
git grep -nE "phenotypic[./]_gui" -- README.md docs/source
git grep -nE "api_reference/gui|gui/index" -- docs/source/api_reference
```

Then the ledger checkers (FEATURES.md changed in Step 11) and the Test surface with `<label>=task3` (Step 11 changed source comments and test docstrings):

```bash
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
```

Expected: both checkers exit 0. The surface has no failures and 16 skipped, and passed = 3009 + 9 − N: Task 1's 3009, plus the 9 tests of `tests/unit/gui/test_private_package.py` (untracked during Task 1's run), minus the N tests of the deleted generator test file (Step 1).

- [ ] **Step 13: Commit** — stage the edited files by name (the Step 2 deletions are already staged):

```bash
git add README.md docs/source/api_reference/index.rst docs/source/tutorials/getting_started.rst docs/source/how_to/pages/gui_hub.md docs/source/tutorials/gui/02_file_explorer.md docs/source/tutorials/gui/06_view_results.md docs/source/tutorials/gui/08_analysis.md CLAUDE.md src/phenotypic/_gui/CLAUDE.md
tr '\n' '\0' < /tmp/private-gui-shorthand-files.txt | xargs -0 git add --
git diff --cached --stat    # expect only this task's paths: the 9 named edits, the Step 11 shorthand files, and the Step 2 deletions
```

Commit with subject `docs(gui): phenotypic-gui is the only documented entry; drop the GUI API reference` and the trailer from Global Constraints.

---

### Task 4: Phase gates (orchestrator)

- [ ] **Step 1:** Test surface with `<label>=final` — compare against `baseline.xml` node by node, including the skip count (baseline: 16). Rerun each new failure alone before attributing it.
- [x] **Step 2:** `uv run mypy --cache-dir <fresh dir> src/phenotypic` ≤ 418 errors; `uv run ruff check src/phenotypic scripts tests --output-format concise` ≤ 65 errors. Diff the finding sets, not just the counts: `uv run python docs/superpowers/logic_validation_scripts/2026-09-10-private-gui-module/compare_findings.py mypy|ruff <main file> <branch file>`, with main's lists produced in the `/tmp/pht-main` worktree. Use a fresh mypy cache on both sides: at `da352a50e` the branch's persistent `.mypy_cache` hid one pre-existing finding (a dangling `TYPE_CHECKING` import in `_cli/_cli_finalize_run.py:19`), which proves cache state can change the finding set. → mypy 418 = 418 and ruff 65 = 65, identical finding multisets (0 new, 0 gone); ruff re-confirmed at `c9cacf8b0` after a fix round touched a test file.
- [x] **Step 3:** Wheel contents: `uv run --no-project --with pytest pytest tests/integration/packaging/test_package_contents.py -m slow -v -o addopts=`. → 5 passed at `da352a50e`.
- [ ] **Step 4:** Docs build exactly as CI runs it (`docs.yml:87` → `docs/Makefile:40`, `sphinx-build -n`), on this branch and in the `/tmp/pht-main` worktree (create and sync it as in Task 2 Step 10 if absent). For each `<label>` in `branch`, `main`:

```bash
uv run make -C docs html > /tmp/pht-docs-<label>.log 2>&1; echo "exit=$?" >> /tmp/pht-docs-<label>.log
```

then `uv run python docs/superpowers/logic_validation_scripts/2026-09-10-private-gui-module/compare_findings.py docs /tmp/pht-docs-main.log /tmp/pht-docs-branch.log` (exit 1 on any branch-only warning). A `sed` that strips only line numbers leaves each checkout's absolute path in every warning, so a `diff` of the two lists would differ on every line and prove nothing; the script also maps the package path and dotted name, and its `--selftest` includes negative controls. Write the build log straight to a file: piping a Sphinx build through `tee` into a background task's output produced a 619 KB transcript. Expected: exit 0 on both, and no new warning — in particular none naming `api_reference/gui`, `phenotypic._gui` or `phenotypic.gui`. The rewrite turns `:class:` targets in `sdk_/_qc_recipe/_recipe.py:25,272` and `_assets/__init__.py:22` into `phenotypic._gui…`, which `-n` checks.
- [ ] **Step 5:** Amend spec acceptance criterion 4 to except `tests/unit/gui/test_private_package.py`, which must name the removed path to assert it is gone (controller ruling during Task 1). Then check criteria 1-7 one by one, and record the results in `docs/superpowers/reports/2026-09-10-private-gui-module/acceptance.md`.
- [ ] **Step 6:** `git worktree remove /tmp/pht-main`.

---

### Task 5: Review, simplify, regression

- [ ] **Step 1:** Dispatch `xander-local:implementation-test-reviewer` (Opus) over `main..refactor/private-gui`, with the spec. Its report goes to `docs/superpowers/reports/2026-09-10-private-gui-module/implementation-test-review.md`.
- [ ] **Step 2:** Apply confirmed findings; rerun the test surface.
- [ ] **Step 3:** Run `/simplify` over the branch diff; apply its fixes; rerun the test surface.
- [ ] **Step 4:** Full regression once: `tests/unit tests/integration tests/gui tests/smoke` with the `run-phenotypic-test` settings (in the background locally, or the committed sbatch on HPCC). Attribute each failure by rerunning it alone here and on `main`.
- [ ] **Step 5:** Hand off with the `superpowers:finishing-a-development-branch` skill.
