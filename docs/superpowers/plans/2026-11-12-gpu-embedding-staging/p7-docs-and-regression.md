# P7 · Docs, the simplify pass, and the full regression

**Spec:** §4.2 (the "One writer" section and parent D5 gain the Stage-4
rewrite), §6 P7, §5 "The end-of-change full sharded regression, once"

---

**Goal:** The module guide, the tracked-state contributor page, the batch
tutorial and the embeddings how-to describe Stage 4 as built; one simplify pass
runs over the change; then one full sharded regression, both lanes, at the final
SHA.

**Depends on:** P6, and every deep review's findings fixed or recorded.

### Task 1: Docs

**Files:**
- Modify: `src/phenotypic/_cli/CLAUDE.md`
  - "Execution strategies" table (`:9-56`): `StagedSlurmStrategy` also serves a
    SLURM run with `embed_stage == "gpu"` (`uses_staged_controller`);
  - "Staged GPU engine" (`:385-445`): a **Stage 4** item after Stage 3:
    deferral, the signal path, the commit order (store → record → signals), the
    backbone-major loop, the determinism settings, the batch rule, and
    `PHENOTYPIC_STAGE4_DEVICE=cpu` as test-only;
  - "Continuation" paragraph (`:618-646`): `stage4` in the classification, the
    three interrupted states, `--retry-failures` for a Stage-4 failure;
  - "SLURM chaining" (`:648-880`): controller config v2, `phases`, the Stage-4
    rounds and their own signal-aware counters, the worklist, a stall ending
    unfinished with exit `5` (U7), the prelude, the no-detector plan, the
    measure role and the measure-mode finalizer (U10);
  - "One writer per artifact, per pass" (`:1181-1197`): restated as **"nothing
    is rewritten after the record that certifies it"** (plan review), with the
    Stage-4 rewrite as the case that needs the restatement: Stage 3 promotes the
    root, Stage 4 promotes it again, and only then is the record written;
  - "Known limits" (`:1403`): a walltime kill between `promote_store`'s two
    renames can lose a store. Pre-existing, but Stage 4 crosses that window once
    more per image than an inline run (plan review, Minor);
  - "Environment variables" (`:1473`): `PHENOTYPIC_STAGE4_DEVICE`,
    `CUBLAS_WORKSPACE_CONFIG`.
- Modify: root `CLAUDE.md`, the "GPU detectors stage automatically" bullet: one
  sentence pointing at Stage 4 and `--embed-stage`.
- Modify: `docs/source/contrib_guide/tracked_state.md`: Stage-4 pending as a
  derived state (no new tracked state), the per-run terminal stage, and one row
  each for the Stage-4 signals (`progress/stage4_payload/`, their fence fields
  and when they are deleted), the per-round worklist
  (`slurm_scripts/stage4_worklist.json`) and the Stage-4 round counters in
  `staged_orchestration.json` (plan review, Minor).
- Modify: `docs/source/tutorials/pages/cli_batch_processing.md`: an
  "Embedding on GPUs" section (`--embed-stage`, the resolution table, R1–R3 as
  the user sees them, `--gpu-slurm`/`--gpu-shards` for Stage 4, pre-staged
  weights and `HF_HUB_OFFLINE=1`; that a local `--embed-stage gpu` checks CUDA
  before Stage 1 and exits `1` without one, and that this check imports torch,
  so `--dry-run` with that flag pays the import too; and that
  `CUBLAS_WORKSPACE_CONFIG` may be `:4096:8` (the default the CLI sets) or
  `:16:8`).
- Create or modify: the embeddings how-to page. If #281 merged none, create
  `docs/source/how_to/pages/embeddings.md` and add it to the how-to toctree.
  It covers model sizes and the size/version pairs, `compute_dtype` (float32
  recommended for every size, giant included; bf16 only for memory-bound models,
  with P2 T4's bf16 cross-GPU and batch-halving distances, per H7), what `compute` records, mixed GPU models (U6), the
  measured device distances (spec §2), P2 T4's batch-invariance result stated
  as S1 restates goal 3 (within one GPU model and physical batch), and U8's
  measure-mode failure behaviour.
- Modify: `docs/superpowers/specs/2026-11-12-dvc-embeddings-refactor/design.md:634`,
  parent D5's writer list: add `replace_run_embeddings` (spec §4.2).

- [ ] **Step 1: Write the docs.** Every claim names the code it describes; every
  number comes from a report under `docs/superpowers/reports/2026-11-12-gpu-embedding-staging/`.
- [ ] **Step 2: Build the docs as a Slurm job** (global CLAUDE.md, doc builds):
  `sphinx-build -j "$SLURM_CPUS_PER_TASK" -D nbsphinx_execute=never` on `short`,
  8 CPU, 16 GB, 1 h, from a detached worktree. Expected: no new warning against
  the #281 count, compared by text, not total.
- [ ] **Step 3: Read the generated HTML** for each changed section; an exit code
  of 0 is not evidence it rendered.
- [ ] **Step 4: Run the docs drift tests.**

  ```bash
  eval $PT tests/unit/test_docs_preflight_codes.py
  ```

- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/_cli/CLAUDE.md CLAUDE.md docs/source/contrib_guide/tracked_state.md docs/source/tutorials/pages/cli_batch_processing.md docs/source/how_to/pages/embeddings.md docs/source/how_to/index.md docs/superpowers/specs/2026-11-12-dvc-embeddings-refactor/design.md
  git commit -m "docs: Stage 4 in the module guide, tracked state, the tutorial and the embeddings how-to (#282 P7)"
  ```

  (List only the files actually changed; the how-to index file is whichever
  toctree holds the how-to pages.)

### Task 2: Simplify, then the full regression

**Files:**
- Report: `docs/superpowers/reports/2026-11-12-gpu-embedding-staging/final-regression.md`
- Gate scripts: `/bigdata/exfab/anguy344/gate-results/c19-final/` (not committed)

- [ ] **Step 1: The simplify pass.** One `code-simplifier` agent (Opus) over
  every `src/` file this change touched (`git diff --name-only <merge-base>..HEAD -- src/`).
  Quality only; each fix its own commit; each reason not to simplify written
  into the report, as #281's `final-regression.md` did.
- [ ] **Step 2: Focused re-run** of every test file the simplify commits touch
  the importers of, both lanes. Expected: green apart from the 13 expected-red
  GUI tests.
- [ ] **Step 3: mypy and ruff.** `uv run mypy src/phenotypic` count before and
  after this change, by file, against the #281 merge commit; `uv run ruff check`
  over the changed paths only. Expected: no new mypy error in a touched file;
  ruff clean.
- [ ] **Step 4: The full regression, once**, at the final SHA, with the
  `make_gate.sh` shape (`/bigdata/exfab/anguy344/gate-results/final/make_gate.sh`):
  - files: every `test_*.py` under `tests/unit`, `tests/smoke`,
    `tests/integration`, `tests/gui`, collected mechanically;
  - a detached worktree at the SHA; 24 shards on `short`, 8 CPU, 24 GB, 2 h,
    `-n $SLURM_CPUS_PER_TASK`; the main worktree's venv, checked for `optuna` and
    `astropy` first (else 24 false failures);
  - the foundation lane: every file that imports torch or uses `stub_backbone`
    (`grep -l 'stub_backbone\|importorskip("torch")'`), plus `tests/unit/measure`,
    `tests/unit/detect/nn` and every test file this plan created;
  - an `afterany` cleanup that removes the worktree.
- [ ] **Step 5: Classify every red** by name: the 13 expected-red GUI tests are
  expected; anything else is run in isolation first, then attributed. A failure
  seen only in the combined run is order dependence until shown otherwise.
- [ ] **Step 6: Write the report and commit it.**

  ```bash
  git add docs/superpowers/reports/2026-11-12-gpu-embedding-staging/final-regression.md
  git commit -m "docs(report): #282 simplify pass and full regression"
  ```
