# Phase gates: affected-surface runs

Each phase ends with one run of its affected surface (CLAUDE.md, "Focused between
phases"). All runs: `QT_QPA_PLATFORM=offscreen uv run pytest <paths> -q --no-header
-o addopts= -m "not slow" -n 4` on a 4-core container (`len(os.sched_getaffinity(0))
== 4`). Every failure was re-run on the pre-change checkout (`425eb66`, whose `src/`
equals `81d19ec`) before attribution.

## Phase A (after `e13247d`)

- **Surface:** 60 test files matching `phenotypic_cli|execute_dry_run|full_validation|
  mint_run_identity|dry.run|clear_machine_state|_io_constants` under `tests/unit/cli`,
  `tests/unit/plotting`, `tests/unit/sdk_`, `tests/integration/cli`.
  **1396 passed, 6 skipped, 20 xfailed** (1422 collected, 7:47).
- **Wider importers:** 32 further files importing a changed module (excluding
  Playwright e2e). **444 passed, 44 skipped, 17 failed.** All 17 are in
  `tests/unit/tune/test_distributed_{finalize,lifecycle}_task2.py` and raise
  `ModuleNotFoundError: No module named 'optuna'` (the `tune` extra is not installed);
  the same 17 fail at `425eb66`. Not attributable to this change.
- **Review:** `phase-a-adherence.md` (pass with changes; fixed in `76c7437`).

## Phase B (after `344025d`)

- **Surface:** `BaseOperation` reaches nearly every operation test, so the operation
  packages were run whole: `tests/unit/{abc_,enhance,correction,measure,detect,refine,
  grid,post,prefab,core,schema}`, `tests/unit/sdk_/mixin`, `tests/unit/test_pickleable.py`.
  **3092 passed, 51 skipped, 12 failed** (19:42).
- All 12 failures are `tests/unit/core/test_napari_pipeline_viewer.py`,
  `ModuleNotFoundError: No module named 'napari'` (the `napari` extra is not
  installed); the same 12 fail at `425eb66`. Not attributable to this change.
- Guards run with Task 4: `tests/unit/ci/test_startup_imports.py`,
  `test_deferred_imports.py`, `tests/unit/tune/test_annotation_coverage.py`: 240 passed.

## Phase C (after `a18211a`; Tasks 5-8, plus Task 9 already committed)

- **Surface:** `tests/unit/cli`, `tests/integration/cli`, `tests/unit/gui/run_console`,
  `tests/integration/gui`, `tests/unit/detect/nn`, `tests/unit/core/test_pipeline_serialization.py`,
  `tests/unit/ci`, `tests/unit/test_docs_staged_cli.py`: every place the new checks, the
  duplicate-key refusal, the preload hook or the GPU-detector changes can reach a CLI run.
  **4680 passed, 35 skipped, 16 xfailed, 1 failed** (9:29).
- **The one failure is an artifact of this session, not of the change.**
  `tests/integration/cli/test_migrate_end_to_end.py::test_fixture_shaped_run_completes_32_measured_and_four_zero_object_images`
  failed with `NameError: name 'read_metadata_csv' is not defined`. The test spawns fresh
  worker processes (`--njobs 2`), and one of them imported
  `_embedded_measurement_tables.py` in the seconds between two edits Task 11 made to it
  while the gate ran (the call site changed before its import was added). Re-run in
  isolation on the finished code it passes (`1 passed`), and again in the Task 11 neighbor
  run (`155 passed`). Lesson recorded: never edit a module a running gate can still
  import fresh; a long-lived xdist worker is safe, a newly spawned process is not.
- **Also verified in Phase C:** placeholder-image CLI tests (10 files) against the input
  checks: 388 passed.

## Phase D (after `8191fa5`; Tasks 10-12)

- **Surface:** `tests/unit/cli`, `tests/integration/cli`, `tests/unit/gui/run_console`,
  `tests/unit/post`, `tests/unit/sdk_`, `tests/unit/core`, `tests/unit/ci`
  (`-m "not slow" -n 4 -o addopts=`): the input-header, metadata-join and post-column
  checks, the shared metadata reader's importers, and the RAW routing in `imread`.
  **6364 passed, 13 skipped, 21 xfailed, 12 failed, 3 errors** (14:52).
- The 12 failures are the known `test_napari_pipeline_viewer.py` baseline (`napari` not
  installed), identical to Phase B.
- The 3 errors are `tests/unit/sdk_/test_label_editor_widget.py` and
  `test_point_picker_widget.py` `TestRealPanelConstruction`: `fixture 'qtbot' not found`.
  `pytest-qt` (the `test-qt` group) is not installed in this environment; the same files
  give the same 3 errors run alone. Not attributable to this change.
- **What this surface did not cover, found afterwards.** `tests/e2e` is outside
  `testpaths` and needs `PLAYWRIGHT=1`, so no gate so far has run it.
  `tests/e2e/gui/test_run_console_fake_slurm.py::test_ordinary_slurm_submit_and_cancel_is_generation_fenced`
  fails from Phase C on: its only input was `b"not-read-by-submitter"`, and the run
  preflight (Task 8, `PF-HEADER-UNREADABLE`, every image affected, so an error) now refuses
  the submit before anything is written. The finding is correct and the premise in the
  fixture's name is the thing that changed, so Task 15 replaces the bytes with a real
  32x32 TIFF; the test is about submit/cancel fencing, not image content. Run with the
  preinstalled headless shell, all 5 tests in the file pass. The Phase E and final
  surfaces include `tests/e2e/gui/test_run_console*.py`.

## Phase C review fixes (`f508094`, C1-C16)

- **Surface:** class resolution now preloads on every first lookup, so the surface is every
  deserializing path: `tests/unit/cli`, `tests/integration/cli`, `tests/unit/core`,
  `tests/unit/detect`, `tests/unit/sdk_`, `tests/unit/refine`, `tests/unit/ci`,
  `tests/unit/gui/analysis`, `tests/integration/gui` (`-m "not slow" -n 4 -o addopts=`).
  **7343 passed, 68 skipped, 21 xfailed, 12 failed, 3 errors** (13:34).
- The 12 failures and 3 errors are the environment baseline recorded under Phase D
  (`napari` and `pytest-qt` not installed). Nothing attributable to the fixes.

## Phase E (after `2197543`; Tasks 13-15)

- **Surface, derived from importers** of `_cli_execution_strategies`, `_cli_interactive`,
  `_cli_staged_slurm`, `sdk_.slurm` and `run_console._callbacks`: `tests/unit/cli`,
  `tests/integration/cli`, `tests/unit/gui`, `tests/integration/gui`, `tests/unit/sdk_`,
  `tests/unit/tune`, `tests/unit/abc_`, `tests/unit/ci`, `tests/integration/packaging`,
  `tests/unit/test_docs_preflight_codes.py`. **8744 passed, 151 skipped, 23 xfailed,
  21 failed, 3 errors** (19:53).
- All 21 failures are `tests/unit/tune` (`test_distributed_finalize_task2.py` 16,
  `test_engine.py` 3, `test_distributed_lifecycle_task2.py` 1, `test_journal_backend_task1.py`
  1), each `ModuleNotFoundError: No module named 'optuna'` (the `tune` extra is not
  installed). The 3 errors are the `qtbot` baseline. Nothing attributable to this change.
- **Browser tests** (first gate to run them; `PLAYWRIGHT=1`, the preinstalled headless shell
  aliased through a scratch `PLAYWRIGHT_BROWSERS_PATH`): `tests/e2e/gui/test_run_console.py`
  and `test_run_console_fake_slurm.py`, **20 passed**.

## Docs build (Task 16 Step 4, at `d3bd6ee`)

- `sphinx-build -b html -j 4 -q -D nbsphinx_execute=never`, in a scratch worktree with its
  own virtual environment (`uv sync --group dev --group docs --extra gui`), so the shared
  environment was not changed under a running gate. The first attempt stopped at
  `PandocMissing`; with a `pypandoc_binary` pandoc on `PATH` it exits 0.
- 680 warnings outside the unreachable intersphinx inventories (the proxy refuses those
  hosts); none names a page this change touched. The five new `{ref}` links to *Run
  Preflight Checks* resolve in the HTML, and the finding-code table renders.

## Phase D review fixes (D1-D9)

- **Surface:** `tests/unit/cli`, `tests/integration/cli`, `tests/unit/post`,
  `tests/unit/gui/run_console`, `tests/integration/gui/test_run_console_callbacks.py`,
  `tests/unit/test_docs_preflight_codes.py`, `tests/unit/ci`. First run: **4155 passed,
  1 failed**: `test_invalid_metadata_never_replaces_existing_snapshot`, a real regression
  from the D8 change (Polars accepts `b'"unterminated'`, which the removed pandas parse
  refused). The pandas parse was restored; the file then passes (21 passed with the
  metadata preflight tests).

## Phase E review fixes (`ac0ba95`, E1-E13)

- **Surface:** `tests/unit/cli`, `tests/integration/cli`, `tests/unit/sdk_`,
  `tests/unit/gui/run_console`, `tests/integration/gui`, `tests/unit/ci`,
  `tests/unit/test_docs_preflight_codes.py` (`-m "not slow" -n 4 -o addopts=`).
  **6155 passed, 20 skipped, 21 xfailed, 3 errors** (10:36); the errors are the `qtbot`
  baseline. Browser run-console tests: **20 passed**.
