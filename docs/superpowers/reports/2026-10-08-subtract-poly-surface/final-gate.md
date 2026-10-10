# SubtractPolySurface final gate

Commit under test: `2bf26eaf3`. Merge-base with `main`: `8159b8819`. All counts below were
read from the Slurm logs in `/bigdata/exfab/anguy344/slurm_logs/`.

## Verdict

No regression from this branch. Every failure in both gates is `ModuleNotFoundError` for an
optional extra (`optuna` from `--extra tune`, `astropy` from `--extra topology`) that the gate
environment did not sync. Each failing test fails with the identical set of names at the
merge-base, and passes on the branch once the extras are installed.

## Step 1: lint and types

- `uv run ruff check` on the 8 touched files: `All checks passed!`
- `uv run mypy src/phenotypic`: 438 errors on the branch and 438 at `8159b8819`. The
  line-number-normalised error lists are identical (empty diff), and none is in a touched file.
  The `run-phenotypic-test` skill's 417 figure is stale.

## Step 2: logic-validation scripts

`basis_equivalence.py`, `level_conventions.py`, `robust_and_subsample.py`
(under `docs/superpowers/logic_validation_scripts/2026-10-08-subtract-poly-surface/`):
each printed `0 failure(s)`.

## Step 3: affected surface (job 29669152)

`tests/unit/enhance tests/unit/abc_ tests/unit/tune tests/unit/ci tests/gui/builder`,
`-n 16 -o addopts= -m "not slow"`, partition `short`:
`21 failed, 2246 passed, 116 skipped` in 149 s. All 21 are in `tests/unit/tune/` (the same
16 + 1 + 3 + 1 tests as the shards below). Nothing in `enhance`, `abc_`, `ci` or
`gui/builder` failed.

## Step 4: full sharded regression (array 29669157, 24 tasks, partition `short`)

Tasks 2, 4-18, 20-22 passed with exit 0. Tasks 0, 1, 3, 19, 23 exited 1. No shard died without
a summary line, and no `ERROR` lines appeared. Summed over the 24 summary lines:
13,994 passed, 24 failed, 223 skipped, 23 xfailed, 590 deselected.

| Shard | Failed | Tests | Cause (from the log) |
|---|---|---|---|
| 0 | 16 | `tune/test_distributed_finalize_task2.py` | `No module named 'optuna'` |
| 1 | 1 | `tune/test_distributed_lifecycle_task2.py::test_run_tuning_interpreter_failure_terminalizes_claimed_generation` | `No module named 'optuna'` |
| 3 | 3 | `tune/test_engine.py` (3 optuna-hook tests) | `Optuna is required for this strategy. Install the 'tune' extra` |
| 19 | 1 | `tune/test_journal_backend_task1.py::test_noncreating_rdb_schema_probe_rejects_empty_catalog_without_mutation` | `No module named 'optuna'` |
| 23 | 3 | `smoke/test_operation.py` x `FilFinderDetector` (`test_operation`, `test_inplace_contract`, `test_detector_objmap_objmask_consistency`) | `No module named 'astropy'` |

## Attribution (job 29674929, focused re-runs)

The four tune files plus `tests/smoke/test_operation.py -k FilFinderDetector`:

| Tree | Environment | Result |
|---|---|---|
| `8159b8819` (merge-base) | gate env, no `tune`/`topology` extras | tune files: 21 failed, 108 passed, 4 skipped. FilFinder smoke: 3 failed. |
| `2bf26eaf3` (branch) | `uv sync ... --extra tune --extra topology` | tune files: 133 passed. FilFinder smoke: 3 passed. |

The 21 + 3 failing test names at the merge-base are the same 24 names that failed on the branch
in the gate. With the extras installed, all 24 pass on the branch. Classification: environment
(missing optional extras), identical on `main`, not caused by this branch.

This matches the 2026-09-04 baseline note (same 16/3/1/1/3 tune and smoke failures) in the
regression-baseline memory, and the note that FilFinder's smoke cases fail without `astropy`.
The later main baselines (0 failed) were evidently run with those extras synced; the sync
command in the Task 9 brief omits `--extra tune --extra topology`.

Not claimed: a count comparison with the 13,462-test main baseline. Scopes differ (this
branch adds tests, and skip counts move with the extras), so names were compared, not counts.

## Commands and jobs

- `uv run ruff check <8 files>`; `uv run mypy src/phenotypic`; the three validation scripts.
- Detached worktree at `2bf26eaf3`, `uv sync --group dev --group test-qt --extra gui --extra napari`.
- Step 3: job 29669152, `plans/2026-10-08-subtract-poly-surface/run_affected_surface.sbatch`.
- Step 4: array 29669157, `run_unit_suite.sbatch` (24-way); cleanup job 29669196 (`afterany`).
- Triage: array 29674929, `triage_failures.sbatch` (task 0 merge-base, task 1 branch with extras).
