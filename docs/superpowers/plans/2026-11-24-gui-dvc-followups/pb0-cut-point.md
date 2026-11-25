# PB0 · PR B's cut point and #282's rebase gate

**Spec:** §1 step 4–5, I2, R13

---

**Goal:** PR B starts on a base that holds #281, PR A and #282, with #282's waiter rewrite
merged over PA1 T2's.

**Depends on:** PR A merged. **Stop 2:** #282 rebased or merged onto that `main` (its own
branch's work, not this plan's).

### Task 1: cut and gate (C16, orchestrator)

- [ ] **Step 1: Confirm the base.** `origin/main` contains PR A's merge, and
  `origin/feat/gpu-embedding-staging` contains `origin/main`
  (`git merge-base --is-ancestor origin/main origin/feat/gpu-embedding-staging`).
- [ ] **Step 2: The rebase-gate tests pass on #282's tip** (they are #282's obligation; this
  step only checks them):
  - `finished_run_outcome(out, generation)` returns `ENDED_UNPUBLISHED` for a same-generation
    fence closed by its finalizer **with** `terminal_status: "failed"` (`mark_generation_failed`,
    `_cli_slurm_lifecycle.py:721-751`);
  - `wait_for_run` still drains `_ledgered_jobs` after its outcome is decided (#282 U15);
  - `RunSnapshot.finished_with_failures` and the pure `decide_wait_outcome` survive #282's
    rewrite (`tests/unit/cli/test_cli_wait.py` green).

  A missing test is written in #282's branch, not PR B's.
- [ ] **Step 3: Cut** `feat/gui-stage4-measure` from `origin/feat/gpu-embedding-staging` in a
  new worktree `.claude/worktrees/gui-stage4`; `uv sync` with the full dev env (README).
- [ ] **Step 4: Re-find every PR B anchor** (spec §5, PB1–PB3) by name; record moves in
  `reports/2026-11-24-gui-dvc-followups/pr-b-anchors.md`. A changed behaviour is stop 4.
