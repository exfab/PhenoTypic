# Phase 0 regression gate — `2539b06d` vs base `a8b6e17c`

**Result: no regressions.** One name differs from base, and it is a load flake that passes alone.

| | Gate (Phase 0 head) | Base |
|---|---|---|
| Commit | `2539b06d` (rename) | `a8b6e17c` (branch base, `origin/main`) |
| Worktree | `/bigdata/exfab/anguy344/gate-worktrees/mcat-p0-2539b06d` (detached, own venv, provenance OK) | `…/mcat-base-a8b6e17c` (same) |
| Array | 29102745 (shards 0–22) + 29104072 (shard 23 re-run) | 29102744 |
| Tests | 13,853 · 1 failed · 0 errors · 72 skipped | 0 failed · 72 skipped |

Harness: `docs/superpowers/plans/2026-09-25-measurement-categories/submit_regression_gate.sh`, which wraps
`2026-09-03-cli-gui-state-tracking/regression_shard.sbatch` (24 × 8 CPU on `short`). Compared by
**name** with `collect_results.py --baseline`.

## Why there is a shard 23 re-run

The original shard 23 (29102745_23) landed on `c03`, one of the old `c` nodes, where the polars
compatibility build runs slowly. After 46 minutes it was about 30% through its files, while the same
shard on base (`r43`) finished in 24:41. It would have hit the 2-hour wall. I held the cleanup
finalizer, re-submitted shard 23 alone with `--exclude=c[01-24,26-30]` against the same worktree
(29104072, ran on `r41`, 28:54, 530 passed), re-chained the finalizer with `afterany` on both jobs,
released it, then cancelled the slow original. The detached worktree was unchanged throughout, so
this is still one tree.

## The one differing name

`tests.integration.gui.test_run_console_callbacks::test_immediate_local_exit_terminalizes_allocated_generation`
failed in shard 16 with `Failed: local exit observer did not terminalize the run`. It **passed when
run alone** in the same gate worktree at `2539b06d` (1 passed, 12.6 s). It is a wall-clock observer
test, and Phase 0 did not touch the run console. Classified as a load flake, not a regression.

# Phase 2 regression gate — `55cf7965` vs base `a8b6e17c`

**Result: no regressions.** Array 29104945 (24 × 8 CPU on `short`), worktree
`/bigdata/exfab/anguy344/gate-worktrees/mcat-p2-55cf7965` (detached, own venv, provenance OK).
`collect_results.py --baseline`: 13,920 tests · 0 failed · 0 errors · 72 skipped;
REGRESSIONS (0), pre-existing (0). Shard 23 ran on `r42` (37:45) and completed normally; no re-run
was needed. The Phase 0 load flake (`test_immediate_local_exit_terminalizes_allocated_generation`)
did not recur.

# Task 10 final regression gate — `f1c5dea1` (branch merged with `origin/main` @ `8159b881`)

**Result: one failure, caused by this branch and fixed in `140a790f`.** Array 29220948 (24 × 8 CPU on
`short`), worktree `/bigdata/exfab/anguy344/gate-worktrees/mcat-final-f1c5dea1` (detached, own venv,
provenance OK). Baseline array 29220972 at `main` `8159b881`, same harness
(`2026-09-03-cli-gui-state-tracking/regression_shard.sbatch`).

| | Gate (`f1c5dea1`) |
|---|---|
| Tests | 14,244 · 1 failed · 0 errors · 73 skipped (all 24 shards, from the per-shard JUnit files) |
| Failure | `tests.unit.ci.test_pytest_shard_manifest::test_browser_tests_run_only_in_shards_that_install_a_browser` |

**Why it failed, and why no earlier gate caught it.** The shard guard reads any function parameter
named `page` as Playwright's fixture and requires that module to run in a shard that installs
Chromium. Phase 3 (`47efe825`) added the plain helper `_category_sections(page: str)` to
`tests/unit/docs/test_measurements_ref_extension.py`, where `page` is generated RST. Phase 3 had no
full regression, so the end-of-branch gate was the first to see it. Fix: the parameter is `rst`
(`140a790f`); the guard file plus the docs-extension file then pass, 24 of 24 (job 29221552).

**Baseline.** Shard 16 of the baseline failed
`tests.integration.gui.test_run_console_callbacks::test_immediate_local_exit_terminalizes_allocated_generation`,
the wall-clock observer test classified as a load flake in the Phase 0 gate above. It did not fail on
the gate side. No other failure on either side.

**mypy** (Task 10 Step 3), `schema`, `util`, `_cli_output_manager.py` and `_cli_readme_generator.py`:
6 errors in 3 files on both `f1c5dea1` and `8159b881`, identical once line numbers are dropped, so no
new errors (jobs 29221052 and 29221054).
