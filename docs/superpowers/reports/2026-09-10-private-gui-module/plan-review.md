# Plan review — private GUI module

**Reviewer:** `xander-local:plan-reviewer` (Opus) · **Reviewed:** `plan.md` + `spec.md` at `676ae0d14` (read-only) · **Date:** 2026-09-10
Saved by the controller: the reviewer has no write tool.

**Verdict: FEASIBLE WITH CONCERNS.** The move itself is sound, but two steps would halt or mislead the run, and several verification steps can pass while something is broken.

## Critical

- **C1. Task 1 Step 8's `uv sync` uninstalls extras that the baseline ran with.** `uv sync` is exact: it removes anything not named. `uv pip list` shows optuna 4.9.0, psycopg 3.3.4, torch 2.11.0, sam2 1.1.0, transformers 5.12.1 and gudhi 3.13.0. In `pyproject.toml` these come only from the `tune`, `torch`, `foundation` and `topology` extras.
  - Surface tests switch on optuna: `tests/integration/gui/test_tune_monitor.py:104` (skipif) and `test_tune_live_timeout.py:20`. Once optuna is gone, the Task 1 Step 11 and Task 4 comparisons drift into extra skips that read as green. Task 5's full regression also loses the torch and transformers suites.
  - CI uses `uv sync --group dev --group test-qt --all-extras` (`gui-checks.yml`, `run-pytest-full.yml`). The error hint in Task 2 Step 4 repeats the problem.
  - **Fix:** use `uv sync --group dev --group test-qt --group docs --all-extras`, or no explicit sync (`uv run` syncs without removing packages). Diff `uv pip list` before and after.
- **C2. Step 4 halts on a correct state.** `git mv` renames the whole directory, untracked contents included. Tested in `/tmp`: after the move, `src/pkg/gui` did not exist and `__pycache__` plus an untracked file were under `_gui`. Step 4's `find` then prints `No such file or directory` (exit 1). The plan says "if the find prints anything, stop", so the executor stops.
  - The three directories that hold only `.pyc` files (`sweep`, `_shared/timeline`, `results_viewer/timeline_view`) move into `_gui`, where they resolve as namespace packages.
  - **Fix:** replace Step 4 with `test ! -e src/phenotypic/gui` (expect exit 0). Optionally delete `__pycache__`-only directories under `_gui`.

## Important

- **I1. One import escapes the rewrite.** `tests/unit/gui/test_optional_deps.py:25` has `from phenotypic import gui`. Neither regex, the Step 7 grep, nor acceptance criterion 4 matches it, so it becomes an ImportError at Step 11. **Fix:** add it to the Step 6 hand edits (`from phenotypic import _gui as gui`), and add `phenotypic import gui` to the Step 7 grep.
- **I2. Step 7's list of expected `"gui"` hits is missing three runtime paths.** They are `src/phenotypic/phenotypicCLI.py:948` (`gui_logs.name != "gui"`), `tests/integration/gui/test_run_console_callbacks.py:1000` and `tests/unit/gui/run_console/test_slurm_observer.py:1383`, all `.phenotypic/logs/gui`. Without them the executor stops on hits it wasn't told to expect. **Fix:** add all three to the list; also add `phenotypicCLI.py:948` to the spec's "Must not change" list.
- **I3. A user tutorial keeps a private module name.** `docs/source/tutorials/gui/02_file_explorer.md:27` (`phenotypic.gui.shell._classifier`) is rewritten to `phenotypic._gui…`, which breaks criterion 6. Task 3 doesn't cover it, and Task 3 Step 11 only greps for `python -m`. **Fix:** add a step that rewords it without the module path, and widen Step 11 to `git grep -nE 'phenotypic[./]_gui' -- README.md docs/source` (expect no output).
- **I4. The docs check in Task 4 Step 4 differs from CI and hides failures.**
  - CI runs `uv run make html` (`docs.yml:87`), which calls `sphinx-build -n` (`docs/Makefile:40`). The plan runs without `-n`, so unresolved `:class:`/`:mod:` targets never warn. The rewrite creates such targets in `sdk_/_qc_recipe/_recipe.py:25,272` and `_assets/__init__.py:22`.
  - Piping into `grep` also discards sphinx's exit status.
  - **Fix:** run `make -C docs html` with `set -o pipefail`, and diff the warning list against one from `main`.
- **I5. Parallel Tasks 2 and 3 can end up in the wrong commit.** Task 3 Step 2's `git rm` stages its deletions straight away. Task 2 Step 12 gives no pathspec, so a plain `git commit` would pull Task 3's deletions into the Task 2 commit. **Fix:** give each task's commit an explicit file list (`git commit -- <files>`), or commit Task 3 before Task 2.

## Minor

- **M1. Console-script lookup (the Windows question).** It works in any venv: `.venv/bin` equals `sysconfig.get_path('scripts')` here, and in a Windows venv `which` finds `Scripts\phenotypic-gui.exe` through PATHEXT. Two weak points:
  - With a non-venv Windows Python, `sys.executable`'s folder is the install root, not `Scripts`, so the first lookup misses.
  - The fallback to `PATH` can then pick up another environment's script. Seven worktree venvs in this repo have one, and each imports its own checkout. That would give green e2e results for the wrong code.
  - **Fix:** use `sysconfig.get_path("scripts")` with no `PATH` fallback, in all three copies.
- **M2.** Task 2 Step 10's `/tmp/pht-main` worktree needs its own `uv sync … --all-extras` before it can attribute anything; say so in the step.
- **M3.** The CLAUDE.md bullets are at lines 237-241, not 236-241.
- **M4.** Task 2 Step 7's reason is off: the FEATURES gate passes on the rename alone. In a `/tmp` test, `git diff --name-only` limited to the new path listed `_gui/FEATURES.md` as added. The step is harmless.

## Areas that check out

- **Rewrite on macOS:** `git grep -lIz | xargs -0 perl -pi` works as written. It touches 467 files, all under the Step 12 pathspecs, and every pathspec exists. No tracked symlink is hit (the 4 are `AGENTS.md` links) and there are no submodules. No hits for `.phenotypic/gui`, `phenotypic/gui/viewer_cache`, `phenotypic[./]gui-`, escaped `phenotypic\.gui`, `../gui/` or `from ..gui`. The viewer cache root is built from separate components (`_output_root.py:1044`), so the rewrite can't touch it.
- **Step 6 hand edits:** all 8 locations match the cited lines. The only miss is I1.
- **`find_spec` counts:** a missing parent package raises `ModuleNotFoundError` (tested), which pytest counts as a failure, so Step 2 has 8 failures. The new test file has 9 tests and `test_console_script.py` has 3, so Task 2 Step 8 expects 12. Step 10's mutation works: a `gui/` directory holding only `__pycache__` returns a namespace spec (`loader=None`), so the guard test fails as intended (tested). The editable install is a `.pth` pointing at `src`, with no static finder map.
- **Script regeneration:** setuptools editable install; `uv_cache.json` tracks a timestamp; no `cache-keys` override. The current `.venv/bin/phenotypic-gui` contains `from phenotypic.gui.shell._launcher import main`, so the `grep -c` check is valid. `uv sync` was not run; this relies on uv's default cache keys, which include `pyproject.toml`.
- **Other launch points:** the only hub launches via `python -m` are `tests/e2e/gui/conftest.py:230-232`, `scripts/capture_gui_tutorial_screenshots.py:537-539` and `tests/integration/gui/test_console_script.py:29`, all covered. Neither `docs/source/conf.py` nor `docs/Makefile` references the package. The rewrite covers `package-integrity.ci.yml:88-89`, `.pre-commit-config.yaml:12,21`, the `gui-checks.yml` path filters and `.claude/skills/gui-tutorial-capture`.
- **Ledger gates:** `check_features_md.py --strict` exits 0 today (295 shipping rows, 0 in progress); `check_workflows_md.py -v` exits 0 (14/14 workflows). Neither ledger references the files Task 3 deletes.
- **Task 3 docs removal:** 15 files; `docs/source/api_reference/index.rst:36-43` is the only toctree entry; the deleted pages define no labels and nothing else links to them. The `gui/index` entries in `tutorials/index.rst` point at the tutorials. `test_reference_generators.py` only writes to `tmp_path`.
- **Packaging test:** Task 4 Step 3's `--no-project` command matches CI (`package-integrity.ci.yml:73-74`); the root `tests/conftest.py` imports nothing from the project.
- **Windows CI:** `run-pytest-full.yml` (windows-latest) installs into a uv venv with `--all-extras`, so the console-script lookup works there.
