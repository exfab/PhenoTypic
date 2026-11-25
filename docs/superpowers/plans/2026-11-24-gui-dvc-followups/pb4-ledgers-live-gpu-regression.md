# PB4 · Ledgers, the live GPU launch, simplify and the full regression

**Spec:** PB4; D7; §6

---

**Goal:** PR B's chrome is in the ledgers and tutorials; one real GPU Stage-4 launch from the
GUI succeeds; the branch passes its full regression.

**Depends on:** PB3.

### Task 1: ledgers, frames, tutorial text (C20)

**Files:**
- `src/phenotypic/_gui/FEATURES.md`: rows for the Embed-stage control, the Auto hint, the GPU
  section (`rc-staged-gpu-section`, `rc-input-gpu-slurm`, `rc-input-gpu-shards`), the Stage-4
  refusals, the measure selector, disabled Validate and form rules; update `:455`.
- `src/phenotypic/_gui/WORKFLOWS.md:40-41` (`run_local`, `run_slurm` descriptions).
- `src/phenotypic/_gui/CLAUDE.md`, "Importing readers": survivor-table rows for the new GUI
  imports of `_cli_embed_stage` (PB1, PB2) and `_cli_reference` (PB3) (Minor 13).
- `scripts/capture_gui_tutorial_screenshots.py`: one frame in `_capture_run_slurm` (`:1085`)
  of the GPU section for a DINO-only pipeline with Auto → GPU; one frame in
  `_capture_run_local` (`:1058`) of the selector set to Measure.
- `docs/source/tutorials/gui/04_run_local.md`, `05_run_slurm.md`: a measure-mode section with
  spec PB5's refusals and its six "cannot do" items.

- [ ] **Step 1: Edit.**
- [ ] **Step 2: `check_features_md.py --strict` and `check_workflows_md.py -v` exit 0.**
- [ ] **Step 3: Regenerate the full screenshot set** (Slurm, `batch`); commit every PNG.
- [ ] **Step 4: Playwright** run-console specs pass (`PLAYWRIGHT=1`).
- [ ] **Step 5: Commit.**

### Task 2: the live GPU-stage launch (C21, orchestrator, **stop 3**)

Runs only on the user's go-ahead.

**Files:**
- Modify: `tests/_support/live_slurm.py` (the live harness): admit one bounded GPU profile under
  `PHENOTYPIC_RUN_LIVE_SLURM_GPU=1` (today it admits "bounded CPU-only resources",
  `tests/integration/gui/test_run_console_slurm_live.py:1-9`).
- Create: `tests/integration/gui/test_run_console_slurm_live_gpu.py`, marked `slow`, skipped
  unless `PHENOTYPIC_RUN_LIVE_SLURM_GPU=1` and `sbatch` is on PATH.

**The case:** a DINO-only pipeline (DINOv2 `small`, weights cached), one 96×96 TIFF, launched
through `submit_slurm` with the form state the GUI builds:
- `--slurm partition=batch mem=8G time=00:30:00 cpus_per_task=2` (or `intel`);
- `--gpu-slurm` either `partition=exfab account=exfab gpus_per_node=1` or
  `partition=short_gpu constraint=gpu_latest time=01:55:00`, whichever `sbatch --test-only`
  predicts sooner (measured in the step, both printed). Never `preempt`.

```python
def test_a_gui_launched_stage4_run_completes():
    result = submit_slurm(gui_state_for_dino_only(...), sandbox_root=..., record_generation=gen)
    assert isinstance(result, SlurmSubmitResult)                     # exit 4
    assert registry.get(run_id).status == "queued"
    wait_until_terminal(observer, run_id, timeout_s=3 * 3600)
    assert registry.get(run_id).status == "complete"
    entry = embedding_entry(store_of(out), "dinov2-vits14/rgb/objects")
    assert entry["model"]["device_name"]                              # GPU recorded
```

- [ ] **Step 1: Ask the user** with the predicted starts; wait for the go-ahead.
- [ ] **Step 2: Submit** from an interactive allocation; logs under
  `/bigdata/exfab/anguy344/gate-results/guiplan/live-gpu/`.
- [ ] **Step 3: Record** job ids, partition, GPU model and the assertions in
  `reports/2026-11-24-gui-dvc-followups/live-gpu.md`.
- [ ] **Step 4: Commit** the harness and test.

### Task 3: simplify, full regression, PR (C22)

- [ ] **Step 1:** `code-simplifier` (Opus) over PR B's `src/` diff; apply; run the affected
  surface.
- [ ] **Step 2:** full sharded regression on PR B's tip (the `run_unit_suite.sbatch` recipe,
  detached worktree). Result in `pr-b-regression.md`.
- [ ] **Step 3:** whole-branch review (`implementation-test-reviewer`, Opus); fix high-signal
  findings.
- [ ] **Step 4: User go-ahead,** then push and open PR B against `main`.
