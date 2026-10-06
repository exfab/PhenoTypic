# Phase 2 review: CLI (Tasks 4–7)

Reviewer: `implementation-test-reviewer` (independent, analysis only).
Scope: `git diff 6d1a2a62..d343a927` (commits `90c3532d`, `b289aed6`, `bedb2159`, `d343a927`),
against spec §5.2/§12, plan Tasks 4–7 and S1–S13, `_cli/CLAUDE.md` and `tracked_state.md`.

## Verdict

**Ready, with fixes recommended before Phase 3 for F1 and F2.** In a single invocation,
the core continuation design holds. Selection, the SLURM identity check, the staged
manifest and the workers all read one per-image digest through one function, and that
read happens after startup publishes the manifest. Dataset names and image stems agree
across the planner and every apply site. A changed digest re-derives the image on every
path, staged included. All six apply sites enter the context.

The defects that remain sit at the edges:

- Workers re-read the mutable manifest at several points in one image's life, so a
  concurrent invocation can certify an output under the wrong plan (F1). The CLI does not
  refuse that concurrency for CLI-submitted ordinary SLURM runs.
- An environmental error is recorded as a terminal scientific failure (F2).
- The tests prove less than they appear to in five named places (F5–F7). Five mutations
  that break real behaviour survive all four test files (M1–M5), and the positive control
  (M6) is killed.

## Findings (most severe first)

### F1 — Medium — workers re-derive the reference digest from the live manifest at several points in one image's life

**Where.**
- `_cli_process_single.py:839-863`: the SLURM identity check reads the manifest.
- `_cli_process_single.py:343` and `_cli_process_only.py:352`: the apply reads it
  again, through `worker_reference_context`.
- `_cli_process_single.py:908-927`: process mode recomputes the work-id **after** the
  apply, and publishes success under it.
- `_cli_process_single.py:1121-1139`: the failure path recomputes it again.
- Local process mode does the same: `_cli_execution_strategies.py` calls
  `_publish_local_image_success` without a `work_identity`, so the work-id is recomputed
  after `process_single_apply_only_core`.

Local full mode is the exception and computes it once (`:477`).

**Why it is new.** Before this change, every input to a work-id was immutable during a
run. Input and pipeline SHA are re-checked against the worklist; config travels on the
command line. Recomputing the work-id was therefore idempotent. The manifest is the first
work-id input that a later invocation rewrites in place (`publish_reference_inputs`
runs at every forward startup). In full mode the table snapshot is the second
(`_snapshot_metadata_csv`).

**Reachability.** The concurrent-run guard (`phenotypicCLI.py:2532-2552`) consults only
`active_ledger_job_ids`. That function reads the lifecycle ledger, and only the staged
orchestration and the GUI run console write to that ledger (`append_lifecycle_entry`
callers). A CLI-submitted ordinary SLURM array (`AutonomousSLURMStrategy`, no `--wait`) is
therefore not refused when the user re-runs the command while it is live. Neither is a
local continuation over such a tree.

**Failure scenario (process mode, SLURM).**
1. A user submits `--mode process --layer detect_mat` with blank map v1 and leaves it
   running.
2. The user notices a wrong blank and re-runs with `--metadata v2`. Startup replaces
   `.phenotypic/reference_metadata.csv` and republishes the manifest. Image `t01`'s digest
   changes.
3. An old-array worker for `t01` is mid-apply. It entered the context before step 2, so
   it holds the v1 table and the v1 blank map.
4. It finishes and recomputes its work-id at `:908`, from the v2 manifest.
5. It publishes success for `t01` under the **v2** work-id, with v1-blank pixels.
6. If that publication lands after the new invocation's own `t01` worker (or the new
   invocation has not yet selected), the v1 output stands, certified as v2. Every later
   continuation then skips `t01`.

That is a silent wrong answer. A narrower window also exists in full mode, between the
identity check (`:839`) and the context entry inside `process_single_image_core`.

**Fix.**
- **(a)** Compute the identity once per image, before the apply, and reuse it for the
  success and failure records. This is the pattern local full mode already uses. In the
  SLURM worker, publish under the `expected_work_id` that the identity check has just
  validated.
- **(b)** Pass the digest the identity was computed from into
  `worker_reference_context(output_dir, dataset, expected_digest=...)`. Have it refuse
  with a non-terminal error when the manifest entry differs, so the context and the
  work-id cannot describe different plans.
- **(c)**, optional and broader: refuse a forward invocation while
  `load_slurm_lifecycle(...)["active"] is True` for any strategy, not only for ledgered
  jobs.

(a) and (b) are small and local.

### F2 — Low-Medium — "table changed since planning" is recorded as a terminal scientific failure

**Where.**
- `_cli_reference.py:414-418`: `_manifest_base_context` raises `ReferenceTableError`.
  That happens at `worker_reference_context` entry, inside each apply site's scientific
  wrapper:
  - `_cli_process_single.py:409` (`PerImageScientificError("full", …)`)
  - `_cli_process_only.py:391`
  - `_cli_staged_workers.py:427`, `:475` (the Stage-2 `try`) and `:717`
- These are recorded through `append_terminal_failure`
  (`_cli_process_single.py:1118-1139`, `_record_local_terminal_failure`) and through the
  staged `_record_terminal_scientific_failure`.

**Failure scenario.** The table bytes change after publication. That happens with
concurrent invocations (F1), or when someone edits `deliverables/metadata.csv` or the
process snapshot by hand. Every image that enters the context then fails terminally. The
per-image digest deliberately excludes the table SHA (so that editing one plate re-runs
only that plate). The next continuation therefore re-plans to the **same** work-id for
every image whose own row did not change. Those images are skipped as "recorded
failure(s)" unless the user passes `--retry-failures`. The error message says "run the
same command again", but doing that does not re-run them.

**Fix.** Raise this condition as a non-scientific error, so no terminal record is
written and the image stays pending. For example, add a distinct
`ReferencePlanStaleError` and let it escape the scientific wrappers, as
`SlurmGenerationInactiveError` already does at `_cli_process_single.py:399`. A missing
dataset name (`ValueError` at `_cli_reference.py:440-441`) is a programming error, and
classifying it as terminal is acceptable.

### F3 — Low — the un-skippable early check misses the no-table and bad-snapshot cases

**Where.** `phenotypicCLI.py:676-705` (`_refuse_unusable_reference_table`) returns
silently when `--metadata` is absent and `--overwrite` is not set. It never looks for the
fallback snapshot. Spec §5.2 says: "The reference table is parsed early, before any
output change and regardless of `--skip-validation`."

**Failure scenario.** Run with `--skip-validation`, a reference pipeline, no
`--metadata`, and either no snapshot or a snapshot without `Metadata_ImageName`. The run
passes the read-only half. Before failing, it reaches the `--restart` clear, the
`mint_run_identity`, and `output_dir.mkdir`. It then fails inside
`_prepare_incremental_startup` as `Cannot prepare incremental startup state: …`. A
missing *column* is still caught only by the skippable preflight. Nothing is lost, but the
promise that nothing under `--output` changes before a refusal is broken.

**Fix.** When `metadata_csv is None`, resolve the fallback snapshot exactly as
`resolve_reference_table_path` does (full: `deliverables/metadata.csv`; process:
`.phenotypic/reference_metadata.csv`). Refuse if it is missing; parse it with
`ReferenceContext` if it exists. Both steps are read-only.

### F4 — Low — one `UNPLANNED_DIGEST` for every failure reason pins stale terminal failures

**Where.** `_cli_reference.py:32` and `:465`.

**Scenario.**
1. Image `t07` is unmatched. Its work-id is W(`"unplanned"`). The run records a terminal
   failure.
2. The user adds a row for it, but the blank it names does not exist (unresolved).
3. The image is still `"unplanned"`, so it gets the same work-id. The old "no row"
   failure is skipped as recorded, and the new reason never surfaces at run time.

The preflight does warn `PF-REF-UNRESOLVED`, so this is not silent. The skip message is
misleading, though.

**Fix.** Give unplanned images a digest of their failure class plus whatever was resolved,
for example `canonical_digest({"unplanned": "unresolved", "values": values})`. Keep a
non-hex marker in it, so it can never equal a real digest.

### F5 — Low (test gap) — the e2e same-named-blanks test does not exercise the dataset key of the lookup

**Where.** `test_cli_reference_e2e.py:151-181`. Both table rows say `BlankImage=blank`.

The test proves that each dataset resolves the **file** within its own directory (the
manifest's per-dataset image map). It cannot detect a planner or worker that drops the
dataset from the **lookup**: without a dataset, both `t01` rows collapse to the one value
`"blank"`. The probe below confirms this. Mutation M3 (planner `narrow` without
`dataset=`) is predicted to survive all four files. The worker-side equivalent is killed
only by the unit assertion `active.dataset == "plate1"` (`test_cli_reference.py:140`), not
behaviourally.

**Fix.** Give the rows different blank names, for example plate1 → `blank` and plate2 →
`blank_b`, with both files present in both directories at different levels.

### F6 — Low (test gap) — four wiring paths have no behavioural test

- **Stage 1.** The most direct staged placement is a top-level `SubtractBlank` before a
  top-level `GpuDetector`. That runs `SubtractBlank` in **Stage 1** (`pre_pipeline`). The
  staged fixture (`test_cli_reference_e2e.py:208-214`) nests both inside `branch`, so
  `pre_pipeline` is empty. Stage 1's context is guarded only by the source-string tripwire
  (`test_cli_reference_wiring.py:56-65`). **Fix:** parametrize the `staged` fixture over
  `{"branch": ImagePipeline(sb, gpu)}` and `{"sb": …, "gpu": …}`.
- **Stage 3 measure context** (`_cli_staged_workers.py:602-603`). No test has a reference
  operation inside a measurer in a staged run, so M1 is predicted to survive. The
  tripwire is still satisfied by the apply context two lines above.
- **SLURM worker `main` passing `output_dir=`** (`_cli_process_single.py:857`, `:926`,
  `:1048`, `:1139`). `test_selection_and_worker_agree_on_the_reference_work_id` calls
  `_worker_work_identity` directly, and no test drives `main` with a reference pipeline.
  Dropping any one of those `output_dir=` arguments (M4) is predicted to survive.
  - In production, dropping the one at `:857` makes every SLURM task fail closed:
    "work identity does not match worklist".
  - Dropping the one at `:926` or `:1048` publishes success under a digest-less work-id
    that selection never matches. The run then re-selects the image forever.

  Harnesses exist (`test_embedded_measurement_publication.py`,
  `test_cli_figures_run_date.py:291` invoke `_cli_process_single.main`). **Fix:** add a
  test that computes `expected_work_id` with `work_id_for_image` after
  `publish_reference_inputs` and runs `main` in both full and process mode.
- **Startup scoping** (`publish_reference_inputs` passing `operations=`). This is tested
  only at preflight level (`test_columns_only_an_out_of_scope_operation_reads_are_not_required`).
  Take a process run with an `ops` `SubtractBlank` plus a measurer-private one that reads a
  column the table lacks. Without `operations=` (M5), startup would raise
  `ReferenceTableError`. M5 is predicted to survive.

### F7 — Low (test gap) — the preflight's no-hash contract is pinned on the planner, not on the check

`test_plan_without_hashing_never_reads_a_reference_image` calls `plan_references`
directly. Flipping `hash_images=False` to `True` in `check_reference_metadata`
(`_cli_preflight.py:1352`, M2) is predicted to survive. **Fix:** run
`check_reference_metadata` under the same `reference_file_digest` monkeypatch.

Also weak: `test_process_mode_without_metadata_is_refused_before_any_write`
(`test_cli_reference_e2e.py:112`) asserts `not exists or no outputs`. The preflight
promises that nothing under `--output` changed, so the assertion should be
`assert not (base / "out").exists()`.

### F8 — Low (docs) — `_cli/CLAUDE.md` is not updated, and Task 11 does not list it

`_cli/CLAUDE.md` names the un-skippable refusals as "the `--metadata`, `--bit-depth` and
`--gpu-slurm` parses". The reference-table check now belongs on that list. The file also
needs two new contracts:

- **Every apply site must enter `worker_reference_context`.** Name the six sites and the
  tripwire test, so that a seventh site is not added bare.
- **Measure mode never touches the manifest.**

`tracked_state.md` is updated correctly.

## Measure-mode open item: recommendation

**Recommendation: refuse it early, in the read-only half and un-skippably, like the GPU
placement refusal.** In `--mode measure`, refuse when
`reference_operations_in_scope(pipeline, "measure")` is non-empty. Name each operation
path and point to `--mode full`.

Reasons:

1. **There is no plan measure mode could legitimately use.** Measure mode has no
   `--input` requirement and plans nothing. The only manifest on disk belongs to the last
   *forward* run. That manifest may be absent: it is removed when a later forward run's
   pipeline read no references. It may also be stale: the blank files may have changed
   since.
2. **Entering a context would make the measurements depend on an input their work-id
   does not carry.** Measure work-ids exclude the digest by design: they come out of
   `image_reference_digest` as `None`, and `test_measure_work_ids_ignore_a_forward_runs_manifest`
   pins that. Changing that would contradict the "measure never touches the manifest"
   rule (S13), and the plan would also need a measure-time manifest.
3. **Leaving the current behaviour is the worst option.**
   - Every image fails one at a time, after a full store read.
   - Measure failures are keyed by work-ids that no table fix can change, so after the
     user corrects the pipeline they persist as "recorded failures" unless the user passes
     `--retry-failures`.
   - `check_reference_metadata` returns `[]` for measure mode, so nothing warns
     beforehand.
4. **A refusal is cheap and truthful.** `reference_operations_in_scope` already scopes by
   `MODE_SLOTS["measure"]`, and `test_reference_operations_follow_the_mode` already proves
   it finds the measurer-private operation.

(Spec note, not a defect: in full mode such an operation runs on
`center_source = image.copy()` with `reset=False` (`_canonical_zone_measure.py:280-291`).
The freshness guard therefore refuses it unless the private pipeline begins with
`SetDetectMode`. The how-to should say so.)

## Mutation experiments (run by the orchestrator, results verbatim)

The orchestrator applied each mutation alone with a byte-restore harness. Each anchor had
to match exactly once; M4 was anchored on the call directly above
`if actual_work_id != expected_work_id:`. Each run covered the four test files with
`-p no:randomly -o addopts= -m "not slow"`, and the full-tree sha256 matched afterwards.

| ID | Mutation | Result |
|---|---|---|
| M1 | Stage-3 `measure` outside the context (`_cli_staged_workers.py:602`) | `exit=0 :: 69 passed`: **SURVIVED** (F6) |
| M2 | preflight `hash_images=True` (`_cli_preflight.py:1352`) | `exit=0 :: 69 passed`: **SURVIVED** (F7) |
| M3 | planner `narrow` without `dataset=` (`_cli_reference.py:121`) | `exit=0 :: 69 passed`: **SURVIVED** (F5) |
| M4 | SLURM identity check without `output_dir=` (`_cli_process_single.py:857`) | `exit=0 :: 69 passed`: **SURVIVED** (F6) |
| M5 | startup plans without `operations=` (`_cli_reference.py:247`) | `exit=0 :: 69 passed`: **SURVIVED** (F6) |
| M6 | Stage-2 prefix outside the context (`_cli_staged_workers.py:464`) | `exit=1 :: 3 failed, 66 passed`: **KILLED** by `test_every_worker_core_enters_the_reference_context`, `test_staged_objmap_export_runs_the_blank_in_stage_2` and `test_staged_full_run_with_subtract_blank_in_the_detector_branch` (positive control) |

The source-inspection tripwire cannot see M1, because `stage3_merge_measure_core` still
contains a second `worker_reference_context(`, the one around the apply. A per-call-site
tripwire would need to count occurrences, or better, be replaced by the behavioural test
proposed in F6.

Probe `.scratch/phase2_probe_dataset_scope.py`, output verbatim:

```
e2e table, no dataset: {'Metadata_BlankImage': 'blank'}
e2e table, plate2   : {'Metadata_BlankImage': 'blank'}
distinct, no dataset: ReferenceLookupError ambiguous
distinct, plate2    : {'Metadata_BlankImage': 'blank_b'}
```

So the e2e table cannot tell a dataset-scoped lookup from an unscoped one. With distinct
per-dataset blank names, an unscoped lookup fails as ambiguous, which is what the
proposed F5 fixture would catch.

Context: the Phase 2 Slurm gate 29450672 over `tests/unit/cli` and `tests/integration/cli`
(134 files; tree sha256-verified unchanged) gave 3690 passed and 0 failed or errored.

## Verified OK

- **Work-id ordering.** Every `work_id_for_image` call in `phenotypicCLI.py` runs after
  the `publish_reference_inputs` call in `_prepare_incremental_startup` within the same
  invocation:
  - `:828` and `:914` (`_migrate_legacy_success_evidence`, called after startup at
    `:3004`)
  - `:3156` and `:3406`
  - the selectors at `:3132`, `:3250` and `:3315`
  - the strategies (`_cli_execution_strategies.py`, `_cli_staged_strategy.py`,
    `_cli_staged_slurm.py:575`, `_cli_slurm_array_scripts.py:395`)

  `validate_resume_compatibility` and `_validate_resume_input_images` compute none.
  `--dry-run` computes none.
- **One digest reader for selection and SLURM workers.** `image_reference_digest`
  (`_cli_failure_tracker.py:347`) serves both, and measure mode gets `None` on both sides.
  Staged SLURM workers never recompute a work-id: they use `StagedManifestEntry.work_id`.
  No `sdk_` or GUI code produces a work-id.
- **A changed digest re-derives staged images.** The local `_stage1` checks
  `staged_store_matches_work_id` and calls `clear_downstream_artifacts_for_stage1`
  (`_cli_staged_strategy.py:174-183`), so a Stage-2 raw array inferred under the old blank
  is never replayed. The work-id-free `_terminal_output_exists` branch runs only when
  `staged_stage3_markers` is False, which happens only on legacy runs.
- **Image identity.** The planner, both work-id readers and `resolve_image` all use
  `source_image_stem`. That matches `Image.imread` (`_image_io_handler.py:797-798`, `:887`
  for stores), including multi-dot names. Stage 2 and Stage 3 look up by the store's
  restored `image.name`, and the staged e2e tests pass on it.
- **Dataset names agree.** The planner uses `Dataset.name`. The apply sites use:
  - local `dataset.name` (`_cli_execution_strategies.py:612`)
  - SLURM worker `dataset_name`
  - staged `ds.name` / `item.dataset`

  Datasets are one directory each (`scan_directory_structure`), so "the dataset's input
  directory" is well defined.
- **The manifest covers the selection universe.** It is published for `full_datasets`,
  after `--image-manifest` and `--sample`. A fixed table changes the digest of an
  "unplanned" image, so the image is re-derived.
- **Process-mode table precedence.** `--metadata` takes precedence over the snapshot, in
  both the preflight (`resolve_reference_table_path`) and startup. The snapshot survives
  `--restart`; the manifest is cleared by it. `--overwrite` without `--metadata` is
  refused before the rmtree. `--metadata` inside `--output` is protected in process mode
  for reference pipelines.
- **All six apply sites enter the context:** `process_single`, `process_only`, Stage 1,
  the Stage-2 prefix (inside the `try`), Stage 3 (apply and measure), and the objmap
  export. No other site applies operations to an image:
  - recompile and migrate apply nothing
  - the SLURM finalizer (`_cli_checkpoint_handler._run_finalize`) aggregates only
  - the GUI run console launches `python -m phenotypic`
  - tune is refused at spec load
  - the dry run applies nothing
- **Preflight behaviour.**
  - It runs in the read-only half, before `--overwrite` and `--dry-run`.
  - It is mode-scoped through `operations_in_scope`, for the gate and the columns alike.
  - It escalates on the union of failing images (`_severity_for`).
  - It plans with `hash_images=False`.
  - Parse failures are wrapped as `ReferenceTableError`
    (`_reference_context.py:170-176`), so neither the check nor the early refusal can
    traceback.
- **No phantom-row false green.** Every failed image makes the run exit non-zero (`:3553`,
  `:3743`). The e2e `exit_code == 0` assertions therefore rule out the case where
  metadata-only phantom rows in `measurements.csv` mimic one row per image.
- **The manifest memo is sound.** It is invalidated by inode as well as mtime and size,
  and `test_a_manifest_replaced_with_the_same_size_and_mtime_is_reread` kills the removal
  of `st_ino`. The table-SHA refusal is pinned by
  `test_worker_refuses_a_table_changed_after_planning`.

Accepted limitation, by spec decision S9: a store blank's digest is its root `zarr.json`,
so a third-party chunk rewrite that leaves the root unchanged does not re-run its
frames.
