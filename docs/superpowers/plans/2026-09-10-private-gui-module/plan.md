# Private GUI Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `src/phenotypic/gui` to `src/phenotypic/_gui` and make the `phenotypic-gui` console script the only user entry to the hub.

**Architecture:** One atomic, scripted move (`git mv` + a word-bounded textual rewrite + a short list of hand edits for path components that no regex can see). That is followed by two disjoint follow-ups: the entry-point seam (delete the hub `__main__`, route every hub launch through the console script) and the documentation sweep (user docs, contributor docs, API reference removal).

**Tech Stack:** Python 3.12, uv, pytest (+ xdist, pytest-qt, Playwright), Dash, Sphinx.

**Spec:** `docs/superpowers/specs/2026-09-10-private-gui-module/spec.md`

## Global Constraints

- `uv` only — never bare `python`/`pip`.
- The env re-sync after touching `[project.scripts]` is the full dev env: `uv sync --group dev --group test-qt --group docs --extra gui --extra napari`. Plain `uv sync` removes the GUI/napari extras.
- The textual rewrite excludes `docs/superpowers/**` (spec D4) and `docs/source/api_reference/gui/**` (deleted in Task 3).
- The "Must not change" runtime paths in the spec (`phenotypic/gui/viewer_cache`, `.phenotypic/logs/gui`, `.phenotypic-gui`, the `phenotypic-gui` script name, test directory names) stay byte-identical.
- `ruff check --fix` only with explicit paths.
- Pytest invocations: `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg`, `-o addopts= -m "not slow"`, explicit `-n`, never `-x` for a measurement (`run-phenotypic-test` skill).
- A test that cannot run must fail, not skip.
- Subagents do **not** commit; the orchestrator reviews each task's diff and commits it.

## Execution DAG

```
Task 0 (baseline, orchestrator) → Task 1 (Keystone: the move) ─┬→ Task 2 (Seam: entry points)  ─┬→ Task 4 (gates) → Task 5 (review, simplify, regression)
                                                                └→ Task 3 (Sweep: docs)        ─┘
```

Tasks 2 and 3 have zero file overlap and may run in parallel in the same checkout (no commits by agents). Model: Tasks 1, 2 and all review gates on the session (Opus-tier) model; Task 3 on a Sonnet-tier model at medium effort.

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
- [ ] **Step 3:** Commit the spec, this plan, and `baseline.md`.

---

### Task 1: Move the package (Keystone)

**Files:**
- Create: `tests/unit/gui/test_private_package.py`
- Move: `src/phenotypic/gui/` → `src/phenotypic/_gui/` (all tracked files)
- Modify (textual rewrite): every tracked text file matching `phenotypic[./]gui`, excluding `docs/superpowers/**` and `docs/source/api_reference/gui/**`
- Modify (hand edits — separate path components, invisible to the rewrite):
  - `pyproject.toml:253-255`
  - `scripts/check_features_md.py:31`
  - `scripts/check_workflows_md.py:41`
  - `tests/integration/packaging/test_package_contents.py:40`
  - `tests/unit/gui/shell/test_source_chrome.py:286`
  - `tests/e2e/gui/test_tune_launch_command.py:13-16`
  - `tests/unit/schema/test_no_metadata_literals.py:37-44`
  - `tests/unit/test_ome_zarr_invariants.py:150`

**Interfaces:**
- Produces: package `phenotypic._gui` (same submodule tree as `phenotypic.gui` today); console script `phenotypic-gui = "phenotypic._gui.shell._launcher:main"`.

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

- [ ] **Step 2: Run it — expect 8 failures** (the old path resolves; `_gui` does not; pyproject still names `phenotypic.gui`):

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/test_private_package.py -q -o addopts= -p no:cacheprovider
```

- [ ] **Step 3: Move the tracked tree**

```bash
git mv src/phenotypic/gui src/phenotypic/_gui
```

- [ ] **Step 4: Remove the untracked leftovers** — these would keep `phenotypic.gui` importable as a namespace package. First verify that only bytecode and `.DS_Store` remain (expected output: nothing), then delete:

```bash
find src/phenotypic/gui -type f -not -name '*.pyc' -not -name '.DS_Store'
rm -rf src/phenotypic/gui
```

If the `find` prints anything, stop and report it — do not delete.

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

- [ ] **Step 7: Prove nothing was missed**

```bash
git grep -nIE 'phenotypic[./]gui([^_a-zA-Z0-9]|$)' -- . ':!docs/superpowers' ':!docs/source/api_reference/gui'
```
Expected: no output.

```bash
git grep -nE "[\"']gui[\"'/]" -- src tests scripts tools pyproject.toml
```
Expected: hits **only** in these files, all runtime paths from the spec's "Must not change" list, or the doomed generators:
- `src/phenotypic/_gui/results_viewer/_output_root.py` (cache root)
- `src/phenotypic/_gui/run_console/_slurm.py`
- `src/phenotypic/_gui/run_console/_slurm_observer.py`
- `src/phenotypic/_gui/run_console/_callbacks.py`
- `tests/unit/gui/run_console/test_slurm.py`
- `tests/unit/cli/test_cli_output_freshness.py`
- `tests/unit/gui/test_viewer_cache_ownership.py`
- `tests/unit/gui/test_check_workflows_md.py` (`tutorials / "gui"`)
- `scripts/generate_dispatch_reference.py`, `scripts/generate_validation_reference.py` (deleted in Task 3)

- [ ] **Step 8: Re-sync the env so the console script is regenerated**

```bash
uv sync --group dev --group test-qt --group docs --extra gui --extra napari
grep -c "phenotypic._gui.shell._launcher" .venv/bin/phenotypic-gui   # expect 1
uv run phenotypic-gui --help | head -3                               # expect "usage: phenotypic-gui"
```

- [ ] **Step 9: Guard test green** — rerun the Step 2 command; expect 8 passed.

- [ ] **Step 10: Mutation proof** — `mkdir -p src/phenotypic/gui/__pycache__` and rerun; `test_public_gui_import_path_is_gone` must FAIL (the namespace-package trap). Then `rm -rf src/phenotypic/gui` and confirm it passes again.

- [ ] **Step 11: Task surface** — run the test surface with `<label>=task1`. Every failure must either be in the baseline or be attributed in isolation. Report the summary line.

- [ ] **Step 12: Commit (orchestrator)**

```bash
git add -A src/phenotypic tests scripts tools pyproject.toml .github .pre-commit-config.yaml .claude/skills README.md CLAUDE.md NOTICE docs/diagrams docs/source
git commit -m "refactor(gui): move phenotypic.gui to the private phenotypic._gui package"
```

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
- Consumes: `phenotypic._gui` and the regenerated console script from Task 1.
- Produces: `_phenotypic_gui_executable() -> str`, a private helper defined separately in `tests/e2e/gui/conftest.py` and `scripts/capture_gui_tutorial_screenshots.py` (the script is standalone and imports nothing from `tests/`).

- [ ] **Step 1: Failing test** — append to `tests/unit/gui/test_private_package.py`:

```python
def test_hub_has_no_module_entry() -> None:
    """``python -m phenotypic._gui`` is not a way to start the hub."""
    assert importlib.util.find_spec("phenotypic._gui.__main__") is None
```

Run the file; expect exactly this test to fail.

- [ ] **Step 2: Make the console-script test require the script** — in `tests/integration/gui/test_console_script.py`, replace `_phenotypic_gui_argv`:

```python
def _phenotypic_gui_argv() -> list[str]:
    """Return argv for ``phenotypic-gui``, the hub's only entry point.

    Prefers the script installed beside ``sys.executable`` so the test uses
    the environment it runs in. A missing script fails the test: there is no
    module fallback to hide behind.
    """
    binary = shutil.which(
        "phenotypic-gui", path=str(Path(sys.executable).parent)
    ) or shutil.which("phenotypic-gui")
    if binary is None:
        pytest.fail("phenotypic-gui console script is not installed in this environment")
    return [binary]
```

- [ ] **Step 3: Delete the hub module entry**

```bash
git rm src/phenotypic/_gui/__main__.py
```

- [ ] **Step 4: Route the e2e fixture through the console script** — in `tests/e2e/gui/conftest.py`, add the helper above `_start_live_server` (add `import shutil` to the imports if absent):

```python
def _phenotypic_gui_executable() -> str:
    """Return the ``phenotypic-gui`` console script of the running environment.

    Prefers the script beside ``sys.executable`` so the hub boots from the
    same environment as the tests; raises rather than skipping when the
    script is missing.
    """
    found = shutil.which(
        "phenotypic-gui", path=str(Path(sys.executable).parent)
    ) or shutil.which("phenotypic-gui")
    if found is None:
        raise RuntimeError(
            "phenotypic-gui console script not found; run "
            "`uv sync --group dev --group test-qt --extra gui`"
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

- [ ] **Step 5: Same change in `scripts/capture_gui_tutorial_screenshots.py::boot_gui`** — add the identical `_phenotypic_gui_executable` helper above `boot_gui` (the script already imports `shutil`, `sys`, `Path`) and replace `sys.executable, "-m", "phenotypic._gui",` with `_phenotypic_gui_executable(),`. The two standalone-viewer captures (`-m phenotypic._gui.analysis`, `-m phenotypic._gui.results_viewer`) stay — spec D1 keeps those launchers.

- [ ] **Step 6: Docstrings**
  - `src/phenotypic/_gui/__init__.py` — replace the module docstring's first paragraph with: `"""Private implementation of the PhenoTypic GUI hub.` + blank line + `Not a public API: import paths under ``phenotypic._gui`` may change without notice. Users start the hub with the ``phenotypic-gui`` console script. Components are lazy-loaded so that optional dependencies (Dash, napari) are only imported when used.`. Keep the "Sub-packages" / "Utilities" sections, with `gui.` → `_gui.`.
  - `src/phenotypic/_gui/shell/_launcher.py` — `:func:`launch_gui`` bullet: "programmatic boot used by downstream tests" (drop "the ``__main__`` module and"); `:func:`main`` bullet: "argparse front-end wired into ``[project.scripts]`` (``phenotypic-gui = phenotypic._gui.shell._launcher:main``), the hub's only entry point." (drop the `python -m` clause).
  - `src/phenotypic/_gui/run_console/__main__.py` — "The unified hub entry point is ``python -m phenotypic._gui``." → "The unified hub entry point is the ``phenotypic-gui`` console script."

- [ ] **Step 7: FEATURES.md** (the CI `features-md-gate` requires this file to change)
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

- [ ] **Step 10: Boot the hub through the new e2e path** (live seam; needs Playwright chromium, which is installed locally):

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -m "not ci_flaky" -q -o addopts= -p no:cacheprovider -n 4
```
Report the summary line; attribute any failure by rerunning it alone on this branch **and** on `main` (`git stash` is not needed — use `git worktree add /tmp/pht-main main`).

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

- [ ] **Step 12: Commit (orchestrator)** — `refactor(gui): make phenotypic-gui the only hub entry point`.

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
  - `docs/source/tutorials/gui/06_view_results.md:20-33`
  - `docs/source/tutorials/gui/08_analysis.md:23-30`
  - `CLAUDE.md:236-241,249,348`
  - `src/phenotypic/_gui/CLAUDE.md` (top-level note)

- [ ] **Step 1: Confirm the deletions have no other consumers** — expected: no output.

```bash
git grep -nE "generate_(dispatch|validation)_reference|_reference_generator|api_reference/gui" -- ':!docs/superpowers' ':!docs/source/api_reference/gui' ':!scripts/generate_dispatch_reference.py' ':!scripts/generate_validation_reference.py' ':!scripts/_reference_generator.py' ':!tests/unit/gui/test_reference_generators.py'
```

(The toctree entry in `docs/source/api_reference/index.rst` spells the page `gui/index`, not `api_reference/gui`, so it does not show up here; Step 3 removes it.)

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
  - Line 53: delete ` (``python -m phenotypic._gui.sweep``)` — `_gui/sweep` has no tracked files.
  - Lines 276-280: keep only `   uv run phenotypic-gui --root ./images --port 8050` in the code block (drop both comments and the module line).
  - Line 314: "Use ``phenotypic-gui`` or ``python -m phenotypic._gui``." → "Use ``phenotypic-gui``."

- [ ] **Step 6: `docs/source/how_to/pages/gui_hub.md`**
  - "Two equivalent entry points boot the same server:" → "The `phenotypic-gui` console script boots the hub:" and drop the `uv run python -m phenotypic._gui …` line from that code block.
  - Warning block: "Always use the hyphenated form `phenotypic-gui` or the module form `python -m phenotypic._gui`." → "Always use the hyphenated form `phenotypic-gui`."
  - Slurm walkthrough: `uv run python -m phenotypic._gui --root <project-dir> --port 8050` → `uv run phenotypic-gui --root <project-dir> --port 8050`.
  - Delete the whole `## Standalone tools` section (the heading through "…the same defaults as the hub launcher."). Spec D1: debug launchers are contributor-only.

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

- [ ] **Step 8: `docs/source/tutorials/gui/08_analysis.md`** — delete the note block:

````markdown
```{note}
The standalone launcher is still useful for headless workflows or
long-running fits where you don't need the rest of the hub:

    uv run python -m phenotypic._gui.analysis \
        --root <path-to-cli-output> --port 8051
```

````

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

- [ ] **Step 11: Verify** — both expected to print nothing:

```bash
git grep -nE "python -m phenotypic\._gui" -- README.md docs/source
git grep -nE "api_reference/gui|gui/index" -- docs/source/api_reference
```

- [ ] **Step 12: Commit (orchestrator)** — `docs(gui): phenotypic-gui is the only documented entry; drop the GUI API reference`.

---

### Task 4: Phase gates (orchestrator)

- [ ] **Step 1:** Test surface with `<label>=final` — compare against `baseline.xml` node by node. New failures must be rerun alone before being attributed.
- [ ] **Step 2:** `uv run mypy src/phenotypic` ≤ 418 errors; `uv run ruff check src/phenotypic scripts tests` ≤ 65 errors (a drop is expected if the deleted scripts carried findings). Diff the finding sets, not just the counts.
- [ ] **Step 3:** Wheel contents: `uv run --no-project --with pytest pytest tests/integration/packaging/test_package_contents.py -m slow -v -o addopts=`.
- [ ] **Step 4:** Docs build in the background: `uv run sphinx-build -b html docs/source /tmp/pht-docs -q 2>&1 | grep -iE "gui|toctree|not found"`. Expected: no warning that names `api_reference/gui` or `phenotypic._gui`/`phenotypic.gui`.
- [ ] **Step 5:** Spec acceptance criteria 1-7 checked one by one; record in `docs/superpowers/reports/2026-09-10-private-gui-module/acceptance.md`.

---

### Task 5: Review, simplify, regression

- [ ] **Step 1:** Dispatch `xander-local:implementation-test-reviewer` (Opus) over `main..refactor/private-gui`, with the spec. It writes `docs/superpowers/reports/2026-09-10-private-gui-module/implementation-test-review.md`.
- [ ] **Step 2:** Apply confirmed findings; rerun the test surface.
- [ ] **Step 3:** Run `/simplify` over the branch diff; apply; rerun the test surface.
- [ ] **Step 4:** Full regression once: `tests/unit tests/integration tests/gui tests/smoke` with the `run-phenotypic-test` settings (background locally, or the committed sbatch on HPCC). Attribute each failure by rerunning it alone here and on `main`.
- [ ] **Step 5:** Hand off with the `superpowers:finishing-a-development-branch` skill.
