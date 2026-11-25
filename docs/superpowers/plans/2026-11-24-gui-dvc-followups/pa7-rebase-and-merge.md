# PA7 · Rebase onto `main` and merge back to back (D9)

**Spec:** §1 sequencing, D5, D9, R9, R10

---

**Goal:** PR A lands on `main` immediately after #281, re-verified on the rebased tree.

**Depends on:** PA6. **Stop 1:** the user holds #281 until PA6's gates pass, then merges
#281. Nothing here pushes, opens a PR or merges without the user's go.

### Task 1: rebase and re-verify (C15)

**Recorded #281 base: `36b9e840e`** (`origin/feat/dvc-embeddings-refactor` when PR A was cut,
2026-11-24). If #281 gains commits before it merges, PR A is first rebased onto the new #281
tip and this SHA is updated here, in the same commit. `main` has already moved independently:
#284 (`59e7792ab`, 2026-11-24) touches `phenotypicCLI.py` and `_cli_failure_tracker.py`, and
`origin/main` was at `ff6912134` on 2026-11-25, so anchors in those files will shift.

- [ ] **Step 1: Confirm #281 merged.** `gh pr view 281 --repo exfab/PhenoTypic --json state,mergeCommit`
  reports `MERGED` with a merge commit. (`git merge-base --is-ancestor` is not used: it fails
  after a squash merge.) Then `git fetch origin`, with the `github_agent` key in a per-command
  `GIT_SSH_COMMAND`.
- [ ] **Step 2: Rebase only PR A's commits:**
  `git rebase --onto origin/main 36b9e840e feat/gui-dvc-followups`. This never replays #281's
  own commits, whether #281 reached `main` by merge commit or squash. Resolve conflicts by
  keeping `main`'s code and re-applying PR A's change on top; a conflict that changes a spec'd
  behaviour stops the task (stop 4).
- [ ] **Step 3: Re-verify every anchor.** For each `file:line` in the spec and this plan,
  re-find the named symbol on the rebased tree. Record moved anchors in
  `reports/2026-11-24-gui-dvc-followups/rebase-anchors.md`; a symbol whose behaviour changed
  is a spec change (stop 4).
- [ ] **Step 4: Re-run the 13 inherited reds by node id** on the rebased tree; all 13 pass.
- [ ] **Step 5: Live CPU SLURM lane:** `PHENOTYPIC_RUN_LIVE_SLURM=1 $PT -m slow tests/integration/gui/test_run_console_slurm_live.py`
  (from an interactive allocation, not a head node), plus one GUI-equivalent submit via
  `submit_slurm` on a 2-image tree: exit `4`, record `queued`, then `complete`.

### Task 2: final gates and the PR (C15)

- [ ] **Step 1: Full sharded regression** on the rebased SHA: the committed
  `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch` recipe, env
  synced with `--extra tune --extra topology`, detached worktree, never `preempt`. Every red is
  run alone before it is attributed. Result in `reports/2026-11-24-gui-dvc-followups/pr-a-regression.md`.
- [ ] **Step 2: Whole-branch review** (`implementation-test-reviewer`, Opus) over the rebased
  diff. Report `pr-a-final-review.md`; fix high-signal findings, re-run their surface.
- [ ] **Step 3: User go-ahead (stop 1),** then push and open PR A against `main`, body ending
  with the session's PR attribution; it closes #283. **Push from the root checkout:** a
  worktree session's `GIT_SSH_COMMAND` push is refused, so leave the worktree with
  `ExitWorktree(keep)` and run
  `GIT_SSH_COMMAND="ssh -i /rhome/anguy344/.ssh/github_agent -o IdentitiesOnly=yes -o BatchMode=yes" git push origin feat/gui-dvc-followups`
  from `/bigdata/exfab/anguy344/PhenoTypic`. Merge immediately after #281 (D9).
