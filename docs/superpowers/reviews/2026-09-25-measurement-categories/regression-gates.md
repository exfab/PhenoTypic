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
