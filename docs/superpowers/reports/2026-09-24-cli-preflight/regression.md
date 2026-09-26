# Full regression (plan Tasks 18 and 19)

- **Commit under test:** `ac0ba95` (all implementation and review fixes).
- **Where:** locally in the implementation container, not through the committed batch
  script. `run_unit_suite.sbatch` needs Slurm, which this environment does not have. The
  invocation mirrors the script's: the same file set (`tests/unit`, `tests/gui`,
  `tests/integration`, `tests/smoke`, and the two `tests/migration` files it names), `-q
  --no-header -p no:randomly -o addopts= -m "not slow"`, no `-x`, with an explicit `-n 4`
  (the container has 4 cores; `-n auto` is never used), `QT_QPA_PLATFORM=offscreen`.
- **Task 18:** the union of every phase's affected surface is contained in this file set,
  so this one run covers it rather than repeating it.
- **Duration:** 28:09 (2026-09-25 00:45 to 01:14 UTC).

## Result

**13663 passed, 236 skipped, 23 xfailed, 37 failed, 25 errors.**

Every failure and error was classified by rerunning it in isolation. One failure was this
change's and is fixed; every other one comes from an optional dependency or browser build
that this environment lacks.

| Group | Count | Cause | Isolation result |
|---|---|---|---|
| `tests/unit/schema/test_no_metadata_literals.py::test_legacy_metadata_names_are_confined_to_compatibility_and_migration_docs` | 1 failed | **This change.** The M31 test added in `26bf6e3` (`tests/unit/post/test_required_columns.py`) spelled a legacy metadata header as a literal, which this repository guard forbids outside compatibility code. | Fixed by taking the legacy name from `LEGACY_HEADER_TO_MEMBER`; the guard and the post tests then pass (13 passed). No phase gate had run `tests/unit/schema` after that test was added. |
| `tests/unit/tune/*` (`test_distributed_finalize_task2.py` 16, `test_engine.py` 3, `test_distributed_lifecycle_task2.py` 1, `test_journal_backend_task1.py` 1) | 21 failed | `ModuleNotFoundError: No module named 'optuna'` (the `tune` extra is not installed). | Same failures at every gate in `phase-gates.md`; not related to this change. |
| `tests/unit/core/test_napari_pipeline_viewer.py` | 12 failed | `No module named 'napari'` (the `napari` extra is not installed). | Identical at the base commit `425eb66` (Phase B record). |
| `tests/smoke/test_operation.py`, the three `FilFinderDetector` cases | 3 failed | `No module named 'astropy'` (the `topology` extra is not installed). The refusal message is from `ba0db2d` (2026-09-16); this change only declared the detector's packages for the preflight. | Same three fail in isolation. |
| `tests/unit/sdk_/test_label_editor_widget.py` (2), `test_point_picker_widget.py` (1) | 3 errors | `fixture 'qtbot' not found` (`pytest-qt`, the `test-qt` group, is not installed). | Same at every gate. |
| `tests/gui/results_viewer/test_splitter_browser.py` (13), `tests/gui/browse/test_viv_controller_browser.py` (4), `tests/gui/results_viewer/test_viv_source_epoch_browser.py` (3), `tests/gui/test_viv_consumer_epoch_browser.py` (2) | 22 errors | The installed Playwright expects `chromium_headless_shell-1217`; the container has build 1194. | **22 passed** when rerun with the preinstalled headless shell aliased through a scratch `PLAYWRIGHT_BROWSERS_PATH`. |

## Also run

- The run-console browser tests (`tests/e2e/gui/test_run_console.py`,
  `test_run_console_fake_slurm.py`), outside `testpaths`: 20 passed at the Phase E
  review-fix gate on `ac0ba95`.

## Not covered here

- A real Slurm cluster: the cluster checks are tested against scripted scheduler output
  and a fake `sbatch` on `PATH`. `slurm_behavior_probe.sh` is the procedure for
  confirming the behavior on a cluster (`slurm-behavior.md`).
- The optional-dependency tests above, which need the `tune`, `napari`, `topology` and
  `test-qt` installs; the batch script run on the cluster environment covers them.
