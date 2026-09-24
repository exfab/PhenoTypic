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
