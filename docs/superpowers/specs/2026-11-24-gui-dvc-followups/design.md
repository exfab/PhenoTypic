# Design: GUI follow-ups for #281 (DVC, TSV, per-pipeline tables) and #282 (Stage 4)

**Status:** draft for review · 2026-11-24 · revised 2026-11-24 with the user's answers to
the eight open questions (§10 lists what they settled)
**Issue:** #283 (PR A); the #282 GUI items (PR B)
**Bases, read for every citation below:**

- **PR A:** `origin/feat/dvc-embeddings-refactor` @ `36b9e840e` (#281). This branch is
  `feat/gui-dvc-followups`, worktree `.claude/worktrees/gui-followups`. PR A is built here
  and targets `main` after #281 merges (§1, D5).
- **PR B:** `origin/feat/gpu-embedding-staging` @ `0993c8e06` (#282).
  - #281 is an ancestor of #282.
  - #282 changes nothing under `src/phenotypic/_gui/` or in `sdk_/_exit_codes.py`
    relative to #281, so a PR A citation of GUI code holds on #282 too.
  - Line numbers differ between the bases. Each PR B citation below is at `0993c8e06`.

**Scope** was decided by the user on 2026-11-24: two PRs, plus one follow-up issue that is
drafted here but not opened. The user's answers the same day widened both PRs (§10).

Each judgement call of mine is marked **[call]** and collected in §9. The user's decisions
are cited as **D1–D8** (§10). New questions only the user can answer are in §11.

---

## 1. Intent

#281 changed the CLI's contract in ways the GUI still reads the old way. It also deliberately
made no change under `src/phenotypic/_gui/` (its spec, §2.2). #282 added a GPU Stage 4 for
embeddings that the GUI cannot reach.

**PR A** makes the GUI read #281's contract:
- exit codes `3` and `4`;
- per-pipeline table folders;
- TSV;
- the `*.pht-*.json` suffixes.

It also:
- turns the 13 GUI tests #281 left red back to green;
- retires `measurements.csv`, which means a CLI change (D1);
- lets a run that finished with failed images be curated (D2);
- lets every metadata picker take a `.tsv` (D8).

**PR B** lets the run console reach Stage 4:
- an `--embed-stage` control;
- a GPU section shown whenever the CLI would use it;
- Stage-4 refusals shown the way the staged-GPU refusal already is;
- `--mode measure` (D6).

**Sequencing (D5).**
1. **Now.** PR A is built on #281's branch.
2. **#281 merges to `main` first.** PR A targets `main` after that.
3. **Before PR A is opened:**
   - `git fetch origin` and rebase `feat/gui-dvc-followups` onto the latest `origin/main`,
     which will by then contain #281;
   - re-verify every PR A citation in this spec against the rebased tree. Line numbers will
     move. A cited behaviour that changed is a spec change, not a plan detail.
4. **PR B is cut only after #282 has been rebased or merged onto a `main` that contains
   PR A.** PR B needs PA1, because a Stage-4 SLURM submission exits `4` like any other, and
   it needs PA4's status helper (PB5).

This reverses #281's own merge gate ("#281 does not merge to `main` before #283 fixes
these", #281 spec §2.2). Between the two merges `main` carries the breaks in §3. See §7 R9
and §11 Q1.

---

## 2. Goals, non-goals, anti-goals

### 2.1 Goals

- **G1.** A SLURM launch the CLI queued is shown as queued, in every case.
  - No queued run reads as failed.
  - The GUI never cancels a generation because the submitter exited `4`.
- **G2.** A run that exits `3` (`COMPLETE_WITH_FAILURES`) reads "complete, N images failed"
  wherever the GUI states a run's outcome:
  - Recent Runs;
  - the run banner;
  - the results-viewer and analysis snapshot badge (D3).

  It reads the same whether the run was local or SLURM.
- **G2b (D2).** Such a run can be curated. Its failed images have no rows, and curating it
  leaves every enforced run-proof descriptor valid.
- **G3.** The Colony picker and the measurement routes read a v2 store's latest measuring run,
  `tables/<n>-<sha12>/measurements/table.parquet`, and still read a v1 store.
- **G4.** The 13 expected-red tests (#282 plan README) are green, through the fix each one
  needs. No `xfail` and no skip.
- **G5.** Every tabular text file PhenoTypic writes for the mirror or the GUI is TSV:
  - `measurements.csv` is no longer written (D1);
  - every reader prefers `.tsv` and falls back to `.csv` in an older tree;
  - an older tree's generated `measurements.csv` is retired after the TSV is published.
- **G5b (D8).** Every metadata picker and table reader in the GUI accepts `.tsv` wherever the
  CLI does.
- **G6.** Every GUI surface that lists, filters, loads or writes a typed config agrees with
  the sdk_ suffix sets:
  - the new `.pht-*.json` names are written;
  - the old `.json.pht-*` names are still read;
  - a tune spec is never listed as a pipeline.
- **G7 (PR B).** A pipeline that can run Stage 4 can be launched with Stage 4 from the GUI,
  with the same resolution the CLI applies, in full and measure mode.
- **G8 (PR B).** Every Stage-4 refusal the GUI can know before launch is shown in
  `rc-staged-gpu-refusal`, with Run disabled. The text is the CLI's own message.
- **G9.** `FEATURES.md`, `WORKFLOWS.md`, the tutorial text and the tutorial screenshots match
  the shipped chrome.

### 2.2 Non-goals

- **The follow-up issue (§8)**, drafted here, not opened:
  - the results-viewer run picker;
  - the DVC `dvc commit` hint;
  - an embeddings view.
- **Tune runs' exit codes.** `phenotypic.tune` does not use `RunExitCode` at #281: there is
  no reference under `src/phenotypic/tune/`, and its exits are `SystemExit(message)` at
  `tune/__main__.py:236` and `tune/_tune_cli/_worker.py:97,102`. The tune monitor's reap
  logic (`_gui/tune/_callbacks.py:741-745`) is therefore unchanged.
- **Claiming a CLI-launched run that has failures, for `--retry-failures`.** It stays refused
  (D2). The refusal is `_foreign_run_conflict`, `shell/_runs_registry.py:1407-1480`.
- **`--mode recompile`, `process` and `migrate` in the run console.** PR B adds `measure`
  only (D6).
- **Changing `resolve_run_state`'s `Completion` literals, or the CLI's exit codes.**
- **A tutorial screenshot of the new status (D4).** It is documented in tutorial text instead.

### 2.3 Anti-goals (failure even if everything "works")

- A queued SLURM run shown as failed, or a submission's generation cancelled by the GUI.
- A run shown as complete when an accepted image has neither a success nor a terminal-failure
  record.
- A curation write that makes an enforced proof descriptor (master, `metrics.json`,
  `schema.json`) stop matching, or that changes the run's verdict.
- A curated `measurements.tsv` that disagrees with `measurements.parquet` after a curation
  write that reported success.
- A GUI-side copy of a CLI decision rule (Stage-4 resolution, a refusal condition, the
  "finished with failures" verdict, measure's argument rules) that can drift from the CLI's.
- A test made green by deleting or weakening the property it pins.
- A legacy `.json.pht-*` config, or a legacy `measurements.csv` tree, that the GUI used to
  open and no longer can.
- An argv the GUI builds that the CLI refuses as a usage error.

---

## 3. What the code does today (verified)

### 3.1 Exit code 4 does not fail every SLURM launch. Its effect is narrower, and worse

#283 says `_slurm.py:826` "treats any non-zero exit as a failed submission". That is true of
the branch at `:826`, but not of the outcome. Every non-zero exit sets `ambiguous_error`
(`run_console/_slurm.py:826-831`), and `submit_slurm` then runs the recovery path that was
built for timeouts: `_reconcile_submission` (`:642-743`, called at `:837-841`).

I probed `submit_slurm` at `36b9e840e` with the submitter mocked:

| Case | Submitter exit | Result |
|---|---|---|
| Legacy `chunk_job_ids` manifest | 4 | `SlurmSubmitResult`, `reconciled=True` |
| Lifecycle active, job ledgered | 4 | `SlurmSubmitResult`, `reconciled=True` |
| Lifecycle **inactive** | 4 | **`cancel_generation` called**, then `SlurmSubmitError("…exited with code 4")` |
| No job recorded | 4 | `SlurmSubmitError("…exited with code 4")` |
| Nothing submitted (all done) | 3 | `SlurmSubmitError("…exited with code 3")` |
| Nothing submitted (all done) | 0 | `SlurmSubmitError("…exited cleanly but … contain no submitted jobs")` |

**What that means in practice:**

- **The ordinary case reaches `queued`.** But every submission is now labelled `reconciled`,
  which erases the signal that something went wrong.
- **If the scheduler evidence is slow to land, a queued run reads `failed`.** The registry
  takes `status="failed"` from the `except Exception` arm at `run_console/_callbacks.py:1086-1114`.
- **If the fence reads inactive, the GUI cancels its own generation.** The branch is
  `_slurm.py:652-656`.
- **A SLURM-mode run with nothing left to do never submits, and exits `0` or `3`.** Today it
  either fails or binds to the *previous* generation's job set:
  - The exits come from `phenotypicCLI.py:3897-3899`, `:3964` and `:4165-4225`. The code at
    `:4165` reads `if results.submitted: … sys.exit(RunExitCode.SUBMITTED)`, and otherwise
    the run finished in-process.
  - The stale binding is in `read_submitted_job_set`. With no `expected_generation`
    (`_slurm.py:372-432`), `_generation_for` (`:234-261`) prefers the lifecycle on disk.
  - The exit-`0` half predates #281. The exit-`3` half is new: the earlier run's fence is
    inactive, so the same branch calls `cancel_generation` on that finished generation.

**Every place the GUI reads a process exit code:**

| Site | Reads | At #281 |
|---|---|---|
| `run_console/_slurm.py:813-831` | the `python -m phenotypic` submitter | **changes** (PA1) |
| `shell/_runs_registry.py:636-688` `observe_local_exit` (`:661`) | local and validate `python -m phenotypic` | **changes** (PA4) |
| `run_console/_callbacks.py:2081-2090, 2192-2197` | forward to `observe_local_exit` | no logic |
| `run_console/_runner.py:410, 460-466` | process mechanics only | none |
| `run_console/_slurm_observer.py:189` | `squeue`'s exit, not the CLI's | none |
| `tune/_callbacks.py:691-697, 741-745` | `phenotypic.tune` | unchanged (§2.2) |

- **Validate is unaffected.** It always runs locally through the runner, even in SLURM mode
  (`run_console/_callbacks.py:2064-2104`, `:392-408`). A dry run exits `0`
  (`phenotypicCLI.py:3474-3482`), and a refusal exits `2`.
- **The fixed codes** are in `RunExitCode` (`sdk_/_exit_codes.py:42-47`): `0` COMPLETE,
  `1` ERROR, `3` COMPLETE_WITH_FAILURES, `4` SUBMITTED, `5` ENDED_UNPUBLISHED and
  `6` WAIT_TIMEOUT. `5` and `6` come only from `phenotypic-wait`, which the GUI never runs.

### 3.2 "Complete with failures" has no representation anywhere in the GUI

- **Local.** `observe_local_exit` maps every non-zero exit to `failed`
  (`shell/_runs_registry.py:661-663`).
- **SLURM.** A run with any terminal failure never gets a run proof.
  `publish_run_completion_evidence` raises unless every accepted image succeeded
  (`_cli/_cli_completion.py:1304-1342`). The observer therefore reaches the
  "inactive, no marker" branch and reports `failed`, with "lifecycle ended terminal-incomplete
  without a current run-completion marker" (`run_console/_slurm_observer.py:661-671`).
- **Boot rehydration.** `resolve_run_state(...).completion` is `"failed"` for such a run
  (`sdk_/_run_state.py:1922-1934`), and `_rehydrated_status` passes it through
  (`shell/_runs_registry.py:1026-1104`).
- **The viewer and analysis snapshot badge** reads "Run failed" (`_gui/_snapshot_status.py:47-50`).
- **Curation is refused.** Every write-authority check asks `completion == "complete"`:
  - `_mutation_guard.py:38-45`, used at `:121, 140, 170, 216`;
  - `OutputRoot.run_is_complete`, `_output_root.py:700-713`;
  - `mutation_snapshot_is_safe`, `:780-797`;
  - the processing-inventory reuse rule, `_processing_inventory.py:97`. Only a reusable
    inventory is `"exhaustive"` (`:130-145`), and the guard refuses a non-exhaustive one
    (`_mutation_guard.py:127-131`).

`phenotypic-wait` already decides "complete with failures" from disk, in
`decide_wait_outcome` (`_cli/_cli_wait.py:309-366`), rule 3: completion `failed`, no
unresolved image, and published. Its seams are:
- `_run_snapshot` (`:116-130`);
- `_published` (`:182-242`);
- `_cancelled` (`:245-266`);
- `_generation_failures` (`:269-285`);
- `_is_process_run` (`:288-301`).

**Two aggregate-proof checks disagree about the curated mirror, and that matters for D2:**
- **`valid_aggregate_snapshot`** (`_cli/_cli_completion.py:1271-1301`) hashes **every**
  `required_outputs` descriptor, so it returns `None` after any curation write.
- **`aggregate_proof_is_current`** (`sdk_/_run_state.py:1415-1433`, exported) skips the
  GUI-written descriptors in `_GUI_WRITTEN_PROOF_DESCRIPTORS` (`:1331-1333`, checked at
  `:1369-1371`).

The waiter's `_published` uses the strict one as its third disjunct (`_cli_wait.py:238-242`).

### 3.3 Two of the 13 reds are stale tests, and one needs a test edit as well as code

I ran all 13 at `36b9e840e` (Slurm job 32434046, log
`/bigdata/exfab/anguy344/gate-results/guispec/reds-32434046.log`). All 13 fail as listed.
`test_measurement_join_migration_run.py` **skipped** 5 tests: the live run at
`…/ucr_029_e_d_Maresca/data/results/2026-08-11-migration-test` is absent or mid-republish.

| # | Test | Failure observed | What makes it green |
|---|---|---|---|
| 1 | `tests/unit/gui/tune/test_validation.py::test_tuning_spec_path_is_deployable` | `spec_path_issue("tuning_spec.json.pht-tune")` returns the Deploy issue | **code**: `_gui/tune/_validation.py:61` accepts the legacy tune suffix (PA6) |
| 2 | `…/test_setup_authoring.py::test_write_authored_setup_spec_preserves_existing_spec_search_space` | `ValueError: No active knobs to tune.` A legacy `existing.json.pht-tune` is loaded as an `ImagePipeline` | **code**: `_setup_authoring.py:356` `load_pipeline_or_spec` recognises the legacy tune suffix (PA6) |
| 3 | `…/test_setup_authoring.py::test_existing_spec_replaces_scorer_only_when_requested` | same cause, surfacing as the in-memory-layout scorer error | **code**: same |
| 4–6 | `tests/integration/gui/tune/test_setup_view.py::{test_existing_spec_controls_preserve_seed_budget_and_storage, test_existing_spec_trial_override_preserves_authored_strategy, test_existing_grid_spec_never_emits_trial_override}` | the Deploy issue names `.json.pht-tune` | **code**: #1 and #2 |
| 7 | `tests/unit/gui/test_curation_does_not_unfence_the_run.py::test_the_proof_still_records_every_published_artifact` | `required_outputs` holds `measurements_tsv`; the test pins the pre-TSV set (`:135-139`) | **test**: the set becomes `{"master_parquet", "measurements_tsv", "measurements_parquet"}`. `measurements_csv` leaves it because PA8 stops writing that file. The test's own docstring says the proof is "a complete record of what finalization published" |
| 8 | `tests/unit/gui/results_viewer/test_measurement_routes.py::test_an_undeclared_column_never_opens_the_parquet` | `FileNotFoundError` from unlinking `store / MEASUREMENT_TABLE_RELATIVE_PATH` (`:205`), a v1 path no v2 store has | **test**: unlink the latest run's payload. The route already reads through `read_embedded_measurement_column` (`_measurement_routes.py:60,107`) |
| 9–12 | `…/test_measurement_source.py::{test_the_picker_offers_the_stores_own_numeric_columns, test_one_unmeasured_image_does_not_empty_the_picker, test_values_are_keyed_by_dataset_image_and_object_label, test_an_unmeasured_image_contributes_no_keys}` | the picker offers nothing (`'Size_Area' in ()`), and no values come back | **code**: `_measurement_source.py:107-108` (PA2) |
| 13 | `…/test_measurement_source.py::test_a_rewritten_table_is_read_fresh` | the same, and the test itself rewrites the v1 path (`:247-249`) | **code and test**: PA2, plus the test locates the payload the way the code does |

Probe at `36b9e840e`: `Image.save2zarr(store, measurement_table=…)` writes
`tables/1-unknown/measurements/table.parquet`. The v1 `tables/measurements/table.parquet`
does not exist, and `latest_measurement_run(store).run_id == "1-unknown"`.

**[call]** For #7 and #8 the test's expectation is stale and the code is correct. The brief
asks for "the code fix that each red really needs", and here that is the test edit. Each edit
must keep the property its test pins:
- #7 still pins the exact set;
- #8 still deletes the payload before the request, so read-before-validate is still caught.

### 3.4 Measure mode in the CLI (for PB5), at `0993c8e06`

**Refusals, all `click.UsageError`, exit 2:**
- **`--input`:** "does not accept --input; it discovers inputs from the existing output
  directory" (`phenotypicCLI.py:2801-2806`).
- **`--dry-run`:** "--dry-run cannot be combined with --mode measure" (`:2811-2814`).
- **`--restart`, `--retry-failures`, `--overwrite`, `--sample`:** refused at `:3089-3119`
  (messages at `:3095, 3101, 3108, 3115`).
- **No `--pipeline`:** refused at `:3121-3123`.
- **An `--output` that does not exist:** refused at `:3124-3130`.
- **A pipeline whose measurers read reference metadata:** refused, un-skippably
  (`_refuse_measuring_reference_pipeline`, `:761-773`, called at `:3150-3151`).
- **Both `metadata.csv` and `metadata.tsv` in the output:** refused
  (`_refuse_ambiguous_metadata_snapshot`, `:3227-3230`).

**What it accepts:**
- **`--metadata`.** It is used for the join, but never snapshotted: the snapshot step skips
  `measure_only` (`:1331`).
- **`--nrows`, `--ncols`, `--image-type`, `--njobs`, `--slurm`, `--gpu-slurm`, `--gpu-shards`
  and `--embed-stage`.** None of these is refused for measure.

**What it runs on.** It discovers stores with `scan_store_outputs(output_dir)`; a
`ValueError` there exits `1` (`:3513-3520`). It is never a continuation:
`continuing = not measure_only and …` (`:3242-3248`), and `resume=continuing` (`:3287`). So
R3 and the U12 refusal, which sit under `if config.resume:`, never apply to measure.

**The unstageable-`GpuDetector` refusal does not apply in measure.**
`uses_staged_gpu_strategy` returns `False` for `measure_only` before it looks at the pipeline
(`_cli_execution_strategies.py:1546-1549`). A SLURM measure that embeds on the GPU runs on the
staged controller (`uses_staged_controller`, `:1561-1574`, #282 U10).

**Outcomes:**
- **Exit `0`:** "Exit 0 means every image really was re-measured and re-embedded" (#282 U16).
- **Exit `3`:** measure failures. A Stage-4 failure keeps the old embeddings and is journalled
  (U8).
- **Exit `1`:** local leftovers (U16).
- **Exit `4`:** at SLURM submission.
- **A SLURM pass with leftovers** ends with the finalizer closing its fence with
  `closed_by: finalizer` plus `terminal_status: "failed"` and `terminal_error` in one write
  (`mark_generation_failed`, `_cli_slurm_lifecycle.py:721-751`). The error line is
  `unfinished_measure_line` (`_cli_staged_controller.py:530-538`). `phenotypic-wait` then
  returns `5`. The GUI observer never reads `terminal_status`: `grep terminal_status
  _gui/run_console/_slurm_observer.py` finds nothing.

---

## 4. PR A: GUI follow-up for #281 (closes #283)

Each item gives the current code, the exact behaviour PR A gives it, and its tests.
Citations are at `36b9e840e` and are re-verified after the rebase onto `origin/main` (§1).

### PA1. The SLURM submission exit code

**Behaviour.** `submit_slurm` (`run_console/_slurm.py:746-872`) classifies the submitter's
exit by `RunExitCode`, imported from `phenotypic.sdk_`, before it decides anything else. No
literal codes appear in GUI code.

| Exit | Meaning | What `submit_slurm` does |
|---|---|---|
| `SUBMITTED` (4) | queued | The normal path. No `ambiguous_error`. Read `read_submitted_job_set(output_dir)`; when `None`, reconcile once, as a clean exit does today (`:842-866`). Return `reconciled=False` unless reconciliation found the jobs. |
| `COMPLETE` (0) / `COMPLETE_WITH_FAILURES` (3) | the CLI finished in-process; nothing was submitted | Return a new `SlurmFinishedWithoutSubmission(returncode, output_dir, stdout, stderr)` result. Read no job set, bind no generation, never reconcile or cancel. |
| `-1` (timeout), stream error, any other code | ambiguous | Unchanged: reconcile, then raise unless durable jobs prove a submission. |

- **The 0/3 result in the callback.** `_complete_slurm_submission`
  (`run_console/_callbacks.py:1030-1201`) sets the record terminal. The status is `complete`,
  or `complete_with_failures` with N (PA4). It is decided by the same evidence check as a
  local exit, with detail "nothing needed SLURM; the run finished in this process". The
  registry already carries the GUI generation through `PHENOTYPIC_GUI_RECORD_GENERATION`
  (`_slurm.py:790`), so the local evidence check applies unchanged.
- **The `active is False → cancel_generation` branch** (`_slurm.py:652-656`) stays, for the
  timeout and stream-error cases it was written for. Exit 4 no longer reaches it.

**Approaches:**

| | Approach | Pro | Con |
|---|---|---|---|
| a | Treat `4` like `0`: `returncode not in (0, 4)` at `:826` | one line | Leaves the 0/3 no-op path wrong (stale binding at 0, failure and possible cancel at 3). Exit 0 still means "submitted" in the GUI, which #281 made false |
| **b** | **Classify by `RunExitCode` (recommended)** | Matches #281 C10 exactly, and removes the stale-binding bug | One new result type and one callback arm |

**[call]** Pick (b). In SLURM mode, exit `0` means "finished without submitting" even when an
earlier generation's job set is still on disk. At #281 no SLURM submission exits `0`
(`sdk_/_exit_codes.py:25-28`).

**Tests** (`tests/unit/gui/run_console/test_slurm.py`, `tests/integration/gui/test_run_console_callbacks.py`):
- **Exit 4:**
  - with a durable job set it returns `reconciled=False`;
  - with an inactive fence it does not call `cancel_generation`;
  - with no job ever recorded it reports one error naming "SUBMITTED".
- **Exit 0 and 3**, with a previous generation's job set on disk: no binding, no cancel, and a
  terminal record (`complete` / `complete_with_failures`).
- **Existing tests stay as they are.** `test_submit_slurm_raises_on_non_zero_exit` (exit 2) and
  `test_abnormal_exit_cancels_recovered_inactive_generation` (exit 2) are unchanged.
- **A probe that cannot pass vacuously.** Each mock-based test asserts the
  `cancel_generation` mock's call count, so a regression to (a) fails the inactive-fence test.
- **Live lane:** `tests/integration/gui/test_run_console_slurm_live.py`
  (`PHENOTYPIC_RUN_LIVE_SLURM=1`, `slow`) is the only end-to-end proof that a real exit 4
  lands `queued`. Run it once before PR A is opened, after the rebase.

### PA2. The results viewer table path

**Today.** `_store_and_identity` (`results_viewer/_measurement_source.py:95-111`) stats
`store / MEASUREMENT_TABLE_RELATIVE_PATH` (`:107-108`), the v1 flat path. A v2 store has no
file there, so every image returns `None`, the picker is empty and no values are read.

The two readers already target the right table: `embedded_measurement_columns` and
`read_embedded_measurement_column` (`sdk_/_measurement_tables.py:2056, 2084`) read the
latest measuring run, or a v1 store's flat table.

**Behaviour.** The payload whose identity keys the caches is the payload those readers read:
- the highest-order run holding `measurements` for a v2 store (#281 D6a);
- the flat table for a v1 store;
- `None` when there is neither.

Nothing else in the module changes. Its docstrings stop naming the v1 path
(`_measurement_source.py:8-10, 60`).

**Approaches:**

| | Approach | Pro | Con |
|---|---|---|---|
| a | `latest_measurement_run(store)` (`sdk_/_table_runs.py:474`, exported), then `.measurements_path()`; fall back to the v1 path when it returns `None` | Uses exported names only | A second copy of "v1 when there are no runs", in the GUI |
| **b** | **Export one sdk_ helper over `record_less_measurement_table` that returns `DeclaredMeasurementTable.relative_path` (`_table_runs.py:504-546, 586`), the one place that knows both layouts (recommended)** | One producer of "which payload does a record-less reader read"; the GUI cannot diverge from the reader | A new sdk_ export |

**[call]** Pick (b). It is a few lines in sdk_ plus an `__init__` entry, and it is the
"one producer" rule this codebase applies elsewhere.

**Safety, unchanged and to be pinned.** The pixel route allows only the first resolved path
component (`_gui/_shared/tiles.py:975-1010`), and `tables` is never in the allow-list
(`results_viewer/_zarr_routes.py:24-32, 270`). So `tables/<n>-<sha12>/…` stays off the wire
exactly as the v1 path did. Add one route test for a v2 payload path returning 404/400. Update
the docstrings that name the v1 path (`_zarr_routes.py:31`, `tiles.py:986, 1001`).

**Tests.**
- Reds #9–13.
- One new test for a store with **two** runs, where the picker offers the higher-order run's
  columns.
- One for a newer **grid-only** run (no `measurements`), which is skipped.
- A v1-store test, so the fallback is not dead code.
- `test_measurement_join_migration_run.py` reads the v1 path too (`:105, 209`). It gets the
  same resolver, but it can only be verified against the live run (§7 R5).

### PA3. The 13 expected-red tests

The table in §3.3 is the plan of record. PA6 turns #1–6 green, PA2 turns #9–13 green, and #7
and #8 are test edits.

**Gate.** All 13 must pass with no `xfail`, skip or `-k` exclusion. The affected surface is
`tests/unit/gui/tune/`, `tests/integration/gui/tune/`, `tests/unit/gui/results_viewer/` and
`tests/unit/gui/test_curation_does_not_unfence_the_run.py`, and it must be green.

### PA4. Exit 3 reads "complete, N images failed"

**Behaviour.** A run whose every accepted image is resolved, with at least one terminal
failure and its deliverables published, gets the terminal registry status
`complete_with_failures` and a count `failed_images = N`. N is the run state's
`diagnostics.failed`, the count the CLI's exit code uses (`phenotypicCLI.py:1493-1518`).

It is shown in four places:
- **Recent Runs:** `complete · N failed`.
- **The status banner** (`update_log_tail`, `run_console/_callbacks.py:2581-2622`): the same
  text.
- **The no-dashboard message** (`:2549-2560`): the same text.
- **The results-viewer and analysis snapshot badge** (D3) (`_gui/_snapshot_status.py`):
  "Run complete · N images failed", in the warning colour. It is a new arm in
  `snapshot_refresh_status` (`:53-128`), taken when completion is `failed` and the helper
  confirms. `_UNFINISHED_BADGE` (`:47-50`) keeps its two literals, so
  `test_every_completion_literal_has_a_badge` still binds it to `Completion`.

**One verdict, four entry points.** A single helper decides "finished with failures":

1. **A local or SLURM-no-op exit `3`.** `observe_local_exit` (`shell/_runs_registry.py:661`)
   branches on `RunExitCode.COMPLETE_WITH_FAILURES`:
   - when the helper confirms, the status is `complete_with_failures`;
   - otherwise it is `failed`, with detail "exit 3, but the run's evidence does not show a
     finished run".
   Other non-zero codes are unchanged.
2. **SLURM observation.** In `_observe_record`, at the "inactive, no marker" branch
   (`run_console/_slurm_observer.py:661-671`), the helper's outcome decides. A fence closed
   by its finalizer with no `terminal_status`, plus a helper confirmation, gives
   `complete_with_failures`. Everything else stays `failed`, except PB5's measure case.
3. **Boot rehydration.** `_rehydrated_status` (`shell/_runs_registry.py:1026-1104`) gives
   `complete_with_failures` for completion `failed` when the helper confirms.
4. **The snapshot badge and the write-authority checks**, through PA10.

**The helper's rule** is `decide_wait_outcome`'s (`_cli/_cli_wait.py:309-366`). For "finished
with failures" that is rule 3:
- completion `failed`;
- `accepted - verified - failed == 0`;
- `failed > 0`;
- published, and not cancelled.

**[call]** Expose the decision from `_cli_wait.py` as one public function,
`finished_run_outcome(output_dir, generation) -> RunExitCode | None`. It applies
`decide_wait_outcome` to the existing seams, with the live-jobs check left to the caller (the
observer has already established that nothing is live). Make the waiter call it too, so the
GUI cannot drift from what a DVC stage sees. The GUI maps the outcome:
- `COMPLETE_WITH_FAILURES` gives `complete_with_failures` with N;
- `COMPLETE` gives `complete`;
- `ENDED_UNPUBLISHED` gives `failed`, or `incomplete` for a measure run (PB5);
- `None` means "not finished".

Two refinements, both documented in the code:
- **A generation-less caller** (local, rehydration, the viewer) gets `True` from `_published`
  (`:206-207`). For those callers the function additionally requires
  `aggregate_proof_is_current(output_dir)` (`sdk_/_run_state.py:1415-1433`), or a process run.
  Otherwise a run killed after its last image but before finalize would read as complete.
- **It uses the GUI-tolerant check**, `aggregate_proof_is_current`, **never
  `valid_aggregate_snapshot`**. The strict check (§3.2) returns `None` after the first
  curation, which would turn a curated `complete_with_failures` run back into `failed` (PA10).

**Status representation:**

| | Approach | Pro | Con |
|---|---|---|---|
| **a** | **New `RunStatus` literal `complete_with_failures`, plus a `failed_images: int \| None` field on `RunRecord` (recommended)** | Honest and terminal. Gating code that tests `== "complete"` keeps its meaning | Touches the status sets (below) and the persisted record |
| b | Keep `complete`, carry N in `status_detail` | Smallest change | Recent Runs shows a green "complete" over failed images, and the claim gate and the observer read it as a clean completion |
| c | Keep `failed`, improve the detail | No schema change | Not what the user asked for (G2) |

**Edits (a) requires:**
- **The status sets:**
  - the `RunStatus` literal and `_RUN_STATUSES` (`shell/_runs_registry.py:103-130`);
  - `_TERMINAL_STATUSES` (`:133-135`);
  - `run_console/_callbacks.py:862`;
  - `run_console/_slurm_observer.py:62`.
- **The persisted record.** `failed_images` is serialised next to `returncode`
  (`_runs_registry.py:1244-1245, 1501`) and read tolerantly, so an older record has `None`.
- **Older readers.** An older GUI reading a newer owner record sees an unknown status.
  `run_status_is_nonterminal` (`:141-147`) returns `False` for it, which is the safe reading.
  sdk_ reads owner status only through `_OWNER_STATUSES_IN_FLIGHT`
  (`sdk_/_run_state.py:1137-1138`), which the new status is not in.
- **The claim gate is unchanged (D2).** A CLI-launched run with failures still meets
  `_foreign_run_conflict`'s "failed non-GUI processing state" refusal
  (`_runs_registry.py:1469-1475`). A GUI-owned `complete_with_failures` record is terminal, so
  its output is claimable as any terminal GUI record is.
- **The CSS.** Add `.run-console-recents-status-complete_with_failures` in `--oi-orange-text`.

**[call]** Use `--oi-orange-text`, the colour `incomplete` and `cancelled` share
(`run_console/_assets/run_console.css:338-350`): finished, but not everything you asked for.
Not `--oi-green-text`, because that is a clean success. Not vermilion, because DESIGN.md
reserves vermilion for error and alert.

**Tests:**
- `tests/unit/gui/shell/test_runs_registry.py`:
  - exit 3 with confirming evidence;
  - exit 3 with an unresolved image, which stays `failed`;
  - exit 3 with no aggregate proof, which stays `failed`;
  - exit 1, unchanged;
  - an old record with no `failed_images`;
  - a CLI-launched run with failures is still refused by the claim gate.
- `test_slurm_observer.py`:
  - a finalizer-closed fence with failures gives `complete_with_failures`;
  - a cancelled fence stays `failed`.
- The rehydration test.
- `tests/unit/gui/test_snapshot_status.py`: the new badge arm, and a curated
  `complete_with_failures` run still reads "Run complete · N images failed".
- The waiter tests (`tests/unit/cli/`): pass unchanged through the new public function.

### PA5. TSV exports and mirror readers

**Writers:**

| Writer | Today | PR A |
|---|---|---|
| Curated mirror, `_write_curated_mirror` (`results_viewer/_curation_labels.py:839-850`) | `.parquet`, then a best-effort `.csv` (`_atomic_write_csv`, `:961-966`) | `.parquet`, then `measurements.tsv` via `sdk_.write_tsv` (`sdk_/_delimited.py:102-148`), then retire a stale `measurements.csv` (PA8). `_atomic_write_csv` is deleted |
| `FilteredMeasurements._save_locked` (`results_viewer/_filtered_state.py:543-600`, CSV at `:586`) | `.parquet`, then `.csv` | `.parquet`, then `.tsv`, then retire the `.csv`. Only tests construct this class (`tests/gui/results_viewer/test_filtered_state.py:64`), so it is kept consistent rather than redesigned |
| Error analysis, `_stage_generation` (`results_viewer/_error_tab/_publication.py:576-600`); the CLI re-emit shares it (`_cli/_cli_error_outputs.py:67-95`) | `error_analysis.csv` (`computation_path_name`, `:849-851`) | `error_analysis.tsv` |

**[call]** The TSV stays secondary to the Parquet, which is what every GUI reader reads. A
failed TSV write **unlinks `measurements.tsv`** and logs, rather than leaving it beside a newer
Parquet. A text reader then finds nothing, and never reads stale rows. A stale CSV is retired
only after the TSV write succeeds, so a failure never costs the older file (the
`remove_stale_csv_sibling` contract, `sdk_/_delimited.py:151-185`).

`measurements_tsv` is already a GUI-written proof descriptor (`sdk_/_run_state.py:1331-1333`),
so curation does not unfence the run (PA10).

**Fingerprints must include the TSV**, or a concurrent TSV write is invisible to the lock and
retry logic:
- `_source_fingerprint` (`_curation_labels.py:761-771`) swaps `layout.mirror_csv` for
  `layout.mirror_tsv`;
- so does `_consumed_state_snapshot_paths` (`results_viewer/_output_root.py:1119-1133`).

**Error analysis publication** has a strict versioned contract. Exact artifact name sets are
checked at `_publication.py:490-495, 638-644, 662-676, 710-714`, under
`schema_version == 1` (journal `:471`, receipt `:717`).
- **New publications.** They write `error_analysis.tsv` under `schema_version: 2`.
- **Readers and rollback.** Both accept version 1 (`.csv`) for a publication already on disk.
- **The stale CSV.** Publishing a version-2 generation retires a live `error_analysis.csv`:
  publish first, then unlink, the B6 pattern (`sdk_.remove_stale_csv_sibling`). That needs
  `"error_analysis"` out of `_KEPT_CSV_STEMS` (`_delimited.py:146-148`; see PA8).
- **New sdk_ names.** Add `ERROR_ANALYSIS_TSV` and `BundleLayout.error_analysis_tsv` beside
  the CSV names, which stay defined as legacy read names (`sdk_/_io_constants.py:1090, 3469-3471`).

**Readers:**
- **`_schema_cache.py:46`**, the only GUI reader of the text mirror. It is the no-pyarrow
  fallback; everything else reads Parquet. It resolves through
  `BundleLayout.measurements_table_path()` (`sdk_/_io_constants.py:3379-3388`): `.tsv`, then
  `.csv`. It reads with `read_delimited`, so the delimiter follows the suffix.
- **The re-export.** `_gui/_config.py:68, 136, 462-466` adds `MEASUREMENTS_TSV`.
  `MEASUREMENTS_CSV` stays as a legacy read name, because #281's import-walk test (#281 spec
  B8; `tests/unit/sdk_/test_gui_sdk_imports.py`, `gui_sdk_signatures.json`) pins every sdk_
  name the GUI imports.

**Tests:**
- curation rewrites `measurements.tsv`, and its bytes equal `write_tsv(parquet frame)`;
- curation on a pre-PA8 tree retires its `measurements.csv`;
- a failed TSV write leaves no `.tsv` and keeps the old `.csv`;
- `_schema_cache` prefers the TSV and falls back to a CSV-only tree;
- error analysis:
  - writes a version-2 TSV publication;
  - reads a version-1 one;
  - retires the old CSV;
  - a rollback over a version-1 journal still validates;
- red #7's test (§3.3).

### PA6. The typed config suffixes

**Builder save is already right.** `_write_pipeline_config` (`builder/_callbacks.py:659-664`)
goes through `ensure_typed_json_suffix`, so the gap is in the listing and filtering surfaces
and the ledger text.

**The defect class: plain `.json` is a legacy pipeline suffix.** `PIPELINE_CONFIG_SUFFIXES`
(`sdk_/_io_constants.py:135-141`) contains `.json`, so any surface filtering by it lists
`x.pht-tune.json`, `x.pht-op.json` and `x.pht-cc.json` as pipelines. Before the flip those
names did not end in `.json`.

**[call]** Add one sdk_ predicate, `is_pipeline_config_path(path)`: it matches
`PIPELINE_CONFIG_SUFFIXES` and matches no non-pipeline typed suffix, canonical or legacy.
Every surface below uses it. The CLI side may adopt it later.

| Surface | Today (file:line) | PR A |
|---|---|---|
| Builder directory browser | `PIPELINE_EXTS = PIPELINE_CONFIG_SUFFIXES` (`builder/_directory_browser.py:32`), matched at `:259` | `directory_tree` (`:151-158`) accepts a predicate as well as `extensions`; the pipeline modal passes `is_pipeline_config_path` |
| Shell classifier | `_classify_file` (`shell/_classifier.py:220-235`). The content peek (`:348-360`) matches `"pipe_cfgs"`, which a tune spec also contains, because it embeds its pipeline | suffix test is `is_pipeline_config_path`, so no `has_pipeline_json` for a tune spec |
| Run console pipeline check | `run_console/_callbacks.py:2897-2899` | `is_pipeline_config_path` |
| Tune Deploy validator | `CONFIG_SUFFIX_TUNING` only (`tune/_validation.py:61`); message names `.json.pht-tune` (`:65`) | accept the canonical and legacy typed tune suffixes; message names `.pht-tune.json` |
| Tune spec loader | `load_pipeline_or_spec` (`tune/_setup_authoring.py:353-358`) treats only the canonical suffix as a spec | canonical **and** legacy typed tune suffixes load as `TuningSpec` |
| Tune pickers | `PIPELINE_CONFIG_SUFFIXES \| {CONFIG_SUFFIX_TUNING}` at `tune/_setup_authoring.py:47`, `tune/_layout.py:273-275`, `tune/_callbacks.py:310, 1243` | the union of the pipeline set and both typed tune suffixes |
| GUI tune preset writer | writes `…setup.json.pht-tune` (`tune/_setup_authoring.py:535`), a legacy name | `…setup.pht-tune.json` |
| Tune winner readers | `best_pipeline_path(path)` at `tune/_run_root.py:120, 137, 150` (canonical only) | read through `resolve_best_pipeline_path` (`sdk_/_io_constants.py:1866`); `TuneRunRoot` keeps a separate canonical write path |
| Tune writers | `_export.py:36-39, 51-54, 86`, `_winner.py:100-106`, `tune/_callbacks.py:2424-2428` write canonical beside any old twin | publish, then unlink `old_typed_config_sibling(path)`, as the CLI does (`tune/_tune_cli/_run.py:89-101`) |
| Analysis `RecipeState.save` | writes canonical (`analysis/_recipe_state.py:1159`), and an old `pipeline.json.pht-pipe` survives | retire the old typed sibling after publishing, as `QcRecipe._retire_old_typed_sibling` does (`sdk_/_qc_recipe/_recipe.py:546-560`); a plain `pipeline.json` is never removed (#281 A4) |
| Capture script | writes `pipeline.json.pht-pipe` (`scripts/capture_gui_tutorial_screenshots.py:52, 229, 1112`) | canonical name, so the screenshots show the name users will see |
| Docs | `_gui/CLAUDE.md:257`; `FEATURES.md` (19 mentions of `.json.pht-*`, e.g. `:123, 433, 486, 496, 512, 522, 527, 604-622`); 10 in `docs/source/tutorials/gui/` | canonical names; legacy names described as read-only |
| E2E fixtures | `tests/e2e/gui/` writes `.json.pht-*` (#281 hand-off P0) | canonical, plus one legacy-name read test per surface family |

**Tests:**
- a unit test per row;
- a parametrised test of `is_pipeline_config_path` over every canonical, legacy and plain name;
- a classifier test showing a `.pht-tune.json` holding `pipe_cfgs` is not a pipeline;
- the tune writers' retire step: the old twin is removed, a plain `.json` is kept, and a
  publish failure leaves the old file.

### PA7. Ledgers, tutorial text and screenshots (the `gui-tutorial-capture` skill)

- **`FEATURES.md`** (the CI gate requires it whenever `_gui/` changes):
  - new rows:
    - the "complete with failures" status;
    - the SLURM no-op exit 0/3;
    - the curated TSV mirror;
    - the error-analysis TSV;
    - the `is_pipeline_config_path` listing;
    - curation of a run with failures (PA10);
    - TSV metadata pickers (PA9);
  - rows to update:
    - `:356` "Curation on-disk format": `measurements.{tsv,parquet}`, no CSV;
    - `:371` "Colony measurement picker": the latest measuring run, not
      `tables.measurements`;
    - `:53, 54, 92, 114, 451`, the metadata picker and handoff rows, which name `.csv`;
    - `:585` and `:586`, the status rows;
    - `:626`, which names `measurements.{parquet,csv}`;
    - every `.json.pht-*` mention.
  - Each `✅ shipping` row's `Test ref` must resolve (pre-commit and `check_features_md.py`).
- **`WORKFLOWS.md`.** No new workflow. `run_local`, `run_slurm`, `view_results` and `analysis`
  descriptions change only if their text names a changed state or file.
- **Tutorial text (D4).** The new status is **documented in text, with no new frame**:
  - `docs/source/tutorials/gui/04_run_local.md` and `05_run_slurm.md` gain one paragraph on
    `complete · N failed`, how it differs from `failed`, and that the failed images have no
    rows;
  - `06_view_results.md` gains one on the badge and on curating such a run (PA10);
  - the tutorials' `measurements.csv` and `.json.pht-*` mentions are updated (`01_setup.md`
    among them).
- **Screenshots.** Regenerate the **full** set with
  `uv run python scripts/capture_gui_tutorial_screenshots.py` after the capture script's
  suffix flip, and commit everything. No capture scenario is added for
  `complete_with_failures` (D4).

### PA8. `measurements.csv` retirement (D1)

**Today.** Finalize writes the mirror three times: `.tsv`, `.csv` and `.parquet`
(`_seed_measurements`, `_cli/_cli_output_manager.py:968-1003`, called at `:1215` from the
shared finalize path). The CSV is kept only because GUI curation rewrote it (#281 B9).

**Behaviour.** No PhenoTypic writer produces `deliverables/measurements.csv` any more.
`measurements.tsv` is the text mirror, and `measurements.parquet` stays the GUI's read table.

**`_cli/` changes:**

| Site | Today | PR A |
|---|---|---|
| `_cli_output_manager.py:968-1003` `_seed_measurements` | writes `.tsv`, `.csv`, `.parquet` | writes `.tsv` and `.parquet`. After the TSV is published, `remove_stale_csv_sibling(measurements_tsv_path(output_dir))` retires an older run's generated `measurements.csv` (the B6 pattern, as for the analysis TSVs at `:838` and `:1502`). The docstring (`:975-983`) stops citing B9 |
| `_cli_completion.py:1205-1210` `required_paths` | records `measurements_csv` | drops `measurements_csv`; the proof records `master_parquet`, `measurements_tsv`, `measurements_parquet`, plus `metrics_json`/`schema_json` (`:1215-1220`). The comment at `:1183-1203` is updated |
| `_cli_interactive.py:257, 274-279` dry-run inventory | names `measurements.csv` "for the results viewer" | names only the TSV and Parquet |
| `_cli_readme_generator.py:116, 148-150` README tree and paragraph | lists `measurements.csv` "kept until #283" | removed; one sentence says older runs had a `measurements.csv` |
| `_dashboard/_generator.py:826, 1017-1019` download command | `wget …/deliverables/measurements.csv` | `…/deliverables/measurements.tsv`; the element id `cmd-csv` is renamed with its CSS/JS |
| `_cli_preflight.py:182-185` hint text | "dropped from measurements.csv" | "dropped from measurements.tsv" |

**`sdk_/` changes:**

| Site | Today | PR A |
|---|---|---|
| `_delimited.py:146-148` `_KEPT_CSV_STEMS` | `{"measurements", "error_analysis", "metadata"}` | `{"metadata"}`. The user's snapshot is never swept (the root `CLAUDE.md` rule); the docstring at `:155-162` follows |
| `_run_state.py:1331-1333` `_GUI_WRITTEN_PROOF_DESCRIPTORS` | `{measurements_csv, measurements_tsv, measurements_parquet}` | **unchanged**. It is reader-side: a proof written before PA8 still records `measurements_csv`, and a curation or a PA8 retirement must not unfence it |
| `_io_constants.py:415, 1698-1700, 3371-3377` `MEASUREMENTS_CSV`, `measurements_csv_path`, `BundleLayout.mirror_csv` | the curation write target | **kept as legacy read names**, with "read-only legacy; no writer" docstrings. The GUI import-walk test pins them (#281 B8) |
| `_io_constants.py:1708-1730, 3379-3388` `measurements_table_path` | "unwired until #283" | wired (PA5); docstrings drop the caveat |
| `_io_constants.py:403-430` constant comments | describe the CSV as the human-opened file | describe the TSV |

**An existing run's CSV on its next finalize.** It is retired after the TSV is published:
publish first, then unlink. This holds for a forward continuation, `--mode measure`,
`--mode recompile` and a full-run `--mode migrate`, all of which reach `_seed_measurements`
through the shared finalize (root `CLAUDE.md`, "Output layout"). Curation on a pre-PA8 tree
retires it the same way (PA5). A tree nobody finalizes again keeps its CSV, and every reader
falls back to it.

**Byte stability (#281 C11).** Removing a file from `deliverables/` is a one-time change in a
DVC stage's out hash, after which runs are byte-stable again. `tests/integration/cli/test_deliverables_byte_stable.py`
keeps comparing, with the CSV gone from both runs.

**Docs that name the file:**
- **The DVC how-to**, `docs/source/how_to/pages/dvc.md:91-94` and `:363-364`: the "until
  #283" caveat is removed, and curated edits now reach the TSV. Its template's `plots:` entry
  already names `measurements.tsv` (`:76`), and its `outs` are directories, so no DVC `outs`
  entry names the CSV.
- **The root `CLAUDE.md`**, "Output layout": `measurements.{csv,parquet}` becomes
  `measurements.{tsv,parquet}`.
- **`src/phenotypic/_cli/CLAUDE.md`, `src/phenotypic/_gui/CLAUDE.md`**: the same.
- **The other pages that name it:** `docs/source/api_reference/cli_reference.rst`,
  `how_to/pages/{reference_metadata,migrate_ome_zarr,rembi_metadata,generate_reports}.md`,
  `tutorials/gui/01_setup.md`, `tutorials/notebooks/07_measuring_and_exporting.ipynb`.
  The notebook is edited as text, not re-executed (§6).
- **Docstrings** that say "a delivered `measurements.csv`":
  `analysis/qc/_expected_vs_detected.py:298, 410`.

**Tests that name the CSV.** 30 files under `tests/` mention it. The largest are:
- `tests/unit/cli/test_tsv_cutover_old_tree.py` (19 lines);
- `test_cli_v2.py` (10);
- `tests/unit/sdk_/test_io_constants.py` (7);
- `test_deliverables_tsv.py` (6).

The shared helper `tests/_output_layout.py:82-105` `write_measurements_mirror(csv=True)`
has **62 importers**. Its default flips to `csv=False`; the keyword stays for old-tree tests.
Each assertion is reviewed one by one:
- an assertion that the CSV **exists** after a finalize becomes one that it **does not**, and
  that the TSV does;
- an assertion about an **old tree** keeps writing a CSV and checks the fallback or the
  retirement;
- `tests/fixtures/tables_v1/v1_run/` is a legacy fixture and stays byte-for-byte.

**New tests:**
- a finalize writes no `measurements.csv`;
- a finalize over a tree holding one retires it only after the TSV exists;
- a TSV publish failure keeps the CSV;
- `remove_stale_csv_sibling` now sweeps `measurements` and `error_analysis` and still refuses
  `metadata`;
- a proof written before PA8, holding `measurements_csv`, still validates after the CSV is
  retired.

### PA9. TSV-capable metadata pickers and readers (D8)

#281 B4 makes every user-supplied metadata table accept `.csv` and `.tsv`, with the delimiter
chosen by suffix (`sdk_.INPUT_TABLE_SUFFIXES`, `_delimited.py:37`; `delimiter_for`, `:42`).
These GUI sites still take `.csv` (plus `.parquet` where the consumer reads it) only:

| Site | Today | PR A |
|---|---|---|
| `shell/_source_picker.py:13, 109` shared metadata picker | `_CSV_EXTS = {".csv"}` | `INPUT_TABLE_SUFFIXES` |
| `shell/_metadata_context.py:722` payload validation | refuses any suffix but `.csv` | accepts `INPUT_TABLE_SUFFIXES`; the log text says "table" |
| `shell/_metadata_context.py:749-786` `read_metadata_csv_table` (Browse's metadata panel, called at `:466, 593`) | `csv.reader` with the default comma, so a TSV parses as one column | the delimiter comes from `delimiter_for(path)`. `utf-8-sig` and the duplicate-header handling stay |
| `builder/_reference_metadata.py:39-41, 44, 72` reference table | `(".csv", ".parquet")`; the refusal text names those two | adds `.tsv`. `ReferenceContext` already reads it (`_core/_reference_context.py:167-175`) |
| `tune/_setup_authoring.py:46, 372, 451, 583` QC layout path | `{".csv", ".parquet"}` | adds `.tsv` |
| `tune/_layout.py:330` and `tune/_callbacks.py:1254` tune metadata pickers | `{".csv", ".parquet"}` | adds `.tsv` |
| `run_console/_request_safety.py:403` metadata preflight | `read_metadata_csv` already accepts `.tsv`; a refused suffix surfaces as a bare `ValueError` | the blocked message names the allowed suffixes, as the CLI's does ("must be a .csv or .tsv table") |
| `analysis/_callbacks.py:95` `MergeMetadata` default `"metadata.csv"` | names a file a `.tsv` run does not have | resolved with `resolve_metadata_snapshot_path`. A tree holding both snapshot suffixes shows the resolver's message, not a traceback |

**[call]** The tune pickers are only useful if their consumer reads a `.tsv`. The QC layout's
`ExpectedVsDetectedCount._load_layout` reads `.csv` and `.parquet` only
(`analysis/qc/_expected_vs_detected.py:612-620`). PA9 extends it to `.tsv` through
`read_delimited`, and its field descriptions (`:79, 110`) follow. A picker that offers a file
its scorer then refuses would be a trap. This is the one non-GUI change PA9 makes.

**Tests:**
- each picker lists a `.tsv`;
- `read_metadata_csv_table` reads a TSV into the same columns and rows as the equivalent CSV;
- a `.txt` is still refused, with the message naming both suffixes;
- the reference table and the QC scorer load a `.tsv`;
- `MergeMetadata`'s default resolves to `metadata.tsv` in a TSV run.

### PA10. Curation of a `complete_with_failures` run (D2)

**Behaviour.** A run that `finished_run_outcome` (PA4) classifies as
`COMPLETE_WITH_FAILURES` is writable exactly as a `complete` run is. The failed images have no
rows in the master: aggregation reads only authorized embedded tables. So they offer nothing
to curate, and their metadata-only phantom rows carry a null `Object_Label`, which is already
excluded from curation keys (`FEATURES.md:356`).

**The rule.** One predicate, `run_permits_mutation(state, output_dir)`, is true for a bundle
(`state is None`), for `completion == "complete"`, and for a confirmed
`COMPLETE_WITH_FAILURES`. It replaces every `completion == "complete"` write-authority check:
- `_mutation_guard._run_is_complete` (`_mutation_guard.py:38-45`) and its three uses inside
  `authorize`: at bind (`:121`, through `OutputRoot.run_is_complete`), on the fresh state
  (`:140`), and on the verified state (`:170`);
- `output_mutations_disabled` and `output_read_only_diagnostic` (`:214-229`);
- `OutputRoot.run_is_complete` (`_output_root.py:700-713`), renamed `run_permits_mutation`.
  Its CAN-17 docstring now names the second allowed state;
- `mutation_snapshot_is_safe` (`_output_root.py:780-797`);
- the processing-inventory reuse rule (`_processing_inventory.py:97`). A `complete_with_failures`
  run is terminal, so it gets an `"exhaustive"` inventory (`:138-145`), and the guard's
  exhaustiveness check (`_mutation_guard.py:127-131`) passes for it as it does for a complete
  run.

**What curation must check** before each write. All of these come from `authorize`:
1. **The verdict, fresh, at all three points.** At bind, on the fresh deep state, and on the
   verified deep state, `run_permits_mutation` must hold. Any change, such as a retry that
   repaired an image or a new failure, refuses the write.
2. **The run identity digest** is unchanged across the three (`:146-150, 169-177`).
3. **No active owner** (`:151-154, 172`) and a current processing snapshot (`:155-159`).
4. **The failed count N is part of the verdict.** A write authorized against "N failed" is
   refused if the fresh verdict says otherwise. **[call]** The receipt records N beside the
   identity digest, so a test can assert it.

**Fencing after a curation write on such a run.** Curation rewrites `measurements.parquet`
and `measurements.tsv`, retires a stale `measurements.csv` (PA5), and writes the
`deliverables/qc/` and `errors/` state. Every descriptor it touches is in
`_GUI_WRITTEN_PROOF_DESCRIPTORS`, so:
- `aggregate_proof_is_current` stays `True`;
- `finished_run_outcome` still returns `COMPLETE_WITH_FAILURES` with the same N;
- `resolve_run_state` keeps completion `failed`, with the same identity digest and
  diagnostics;
- the enforced descriptors (`master_parquet`, `metrics_json`, `schema_json`) still match their
  bytes;
- the run has no run proof before or after; none is written for a run with failures
  (`_cli_completion.py:1304-1342`), so curation cannot create or break one.

`valid_aggregate_snapshot` returns `None` after the write (§3.2), as it already does for a
curated complete run. PA4's helper never asks it, so this does not move the verdict. It is a
pre-existing property of the strict check, recorded in §7 R4.

**Tests** (`tests/unit/gui/results_viewer/`, `tests/unit/gui/test_curation_does_not_unfence_the_run.py`):
- the guard authorizes a `complete_with_failures` run and still refuses a `failed` run with an
  unresolved image;
- a verdict that flips between bind and write (a retry repairs the failed image) is refused;
- after a curation write on such a run:
  - `aggregate_proof_is_current` is `True`;
  - `finished_run_outcome` returns the same N;
  - `resolve_run_state(...).identity.digest()` and `.diagnostics` are unchanged;
  - `master_parquet`, `metrics_json` and `schema_json` still match;
- a plain CLI re-run of the same command after the curation still exits `3` and does not
  rewrite the curated mirror's labels (`curation_labels.parquet` survives, per the root
  `CLAUDE.md`);
- the Results and Analysis read-only diagnostic no longer appears for such a run.

---

## 5. PR B: run-console Stage 4 and measure mode (#282 GUI items)

**Base:** `origin/feat/gpu-embedding-staging` @ `0993c8e06`, cut after #282 is rebased or
merged onto a `main` that contains PR A (§1). All citations in this section are at that base
and are re-verified when PR B is cut.

**What #282 requires of the GUI.** #282's spec names only the control ("A run-console control
for `--embed-stage` is a GUI follow-up. It needs the FEATURES.md and WORKFLOWS.md ledgers",
`design.md:418`). The rules the GUI must mirror are:
- **U2** (`:76`): `auto` is `gpu` on SLURM with a GPU profile, otherwise `inline`; a local run
  is `inline` unless `gpu`.
- **U4** (`:79`): `--mode measure` uses Stage 4.
- **U9** (`:83`): a `device="cpu"` pin wins under `auto`, and `gpu` with a pin is refused.
- **U12** (`:87`): a continuation under `auto` adopts the recorded stage. R3 refuses only a
  conflicting explicit flag, or an `auto` that would adopt `gpu` with no Stage 4.
- **U16** (`:91`): a measure pass with leftovers ends unfinished.
- **§4.1** (`:109-124`): in measure mode only `--gpu-slurm` counts as a GPU profile.
- **§4.7** (`:307-321`): R1–R3 are one usage-error line each, raised before `--dry-run`.

### PB1. An `--embed-stage` control

**Today.**
- No GUI code passes `--embed-stage`; `grep embed_stage src/phenotypic/_gui` finds nothing.
- The CLI option is `phenotypicCLI.py:2504-2515` (`click.Choice(["auto","gpu","inline"])`,
  default `auto`).
- `to_argv` builds the common argv tail (`run_console/_state.py:515-603`), and
  `_build_subprocess_argv` adds the SLURM-only `--gpu-*` options (`run_console/_slurm.py:197-210`).

**Behaviour.**
- **The control.** A three-way radio, `rc-radio-embed-stage`: Auto (default), GPU, Inline.
- **Where it lives.** In the always-visible run options, beside `--retry-failures`, not inside
  the SLURM-only GPU section, because `gpu` applies to a local run too (U2).
- **When it is enabled.** Only when the pipeline has an in-scope embedder
  (`pipeline_embedders`, `_cli/_cli_embed_stage.py:88`). Otherwise it is disabled, with the
  help text "This pipeline has no embedder; nothing runs in Stage 4" (the O5 note's meaning).
- **The state field.** `RunConsoleState.embed_stage: Literal["auto","gpu","inline"] = "auto"`.
  `from_json` reads it tolerantly, so older presets load as `auto` (pattern at
  `_state.py:146-147, 179-195`). It is passed into the positional
  `_action_control_states`/`values[...]` mapping (`run_console/_callbacks.py:411, 515-516, 552-553`).
- **The argv.** `to_argv` emits `--embed-stage <v>` only when `v != "auto"`, so existing
  argv, presets and the command digest are unchanged for every current user.
- **A local run with GPU chosen on a host with no CUDA** fails with the CLI's one-line
  `ClickException` (exit 1, `prepare_stage4_environment`, `_cli_embed_stage.py:526-562`). It
  surfaces as a failed run with that line, not in the refusal alert: it is a fact about the
  host, not the pipeline. **[call]**

### PB2. GPU section visibility matches the CLI's resolution

**Today.**
- `show_staged_gpu_controls` (`run_console/_callbacks.py:1840-1865`) shows
  `RC_STAGED_GPU_SECTION` only when `mode == "slurm" and uses_gpu`.
- `uses_gpu` comes from `_staged_gpu_capability` (`:255-277`), which calls
  `pipeline_requires_gpu` (`_cli/_cli_validation.py:442`), true for a `GpuDetector` only.
- So a DINO-only pipeline can never pass `--gpu-slurm`. SLURM `auto` then always resolves
  `inline` (`resolve_embed_stage`, `_cli_embed_stage.py:132-174`), and Stage 4 is unreachable
  from the GUI.

**Behaviour.** The section is shown when `mode == "slurm"` and any of these holds:
- **in full mode,** the pipeline holds a `GpuDetector` (today's rule);
- the pipeline can run Stage 4 (an in-scope `MeasureDinoEmbedding`, with no `device="cpu"`
  pin and no stageless embedder), and the control is not `inline`;
- the control is `gpu`.

In measure mode the first bullet does not apply: the detector does not run, and only
`--gpu-slurm` counts as a GPU profile (PB5).

Under `auto`, a hint under the control states what the CLI will resolve. It is computed by the
CLI's own resolver, for example:
- "Auto → GPU Stage 4 (GPU profile below)";
- "Auto → inline: meas 'dino' is pinned to device='cpu'", which is the U9 note text, verbatim.

The section's wording stops saying "detection":
- the header "Staged GPU detection" (`run_console/_form.py:513`) becomes "GPU stages";
- the shards help "One resident model per whole-GPU shard." (`:556`) covers Stage 2 and
  Stage 4;
- also update `_ids.py:136-137` and `_state.py:100`, and `docs/source/tutorials/gui/05_run_slurm.md:22, 44`.

**One resolver, not two:**

| | Approach | Pro | Con |
|---|---|---|---|
| a | Reimplement the rule in the GUI | no CLI change | A second copy of U2/U9/O3, which is anti-goal 4 |
| b | Call `has_slurm_gpu_profile` / `stage4_supported` / `resolve_embed_stage_for_config` (`_cli_embed_stage.py:350-468`) | the exact CLI code | They take an `ExecutionConfig` with about 20 required fields (`_cli/_cli_types.py:148+`), which the GUI cannot honestly build |
| **c** | **Factor `resolve_embed_stage_for_config`'s pipeline half into a public `resolve_embed_stage_for_pipeline(pipeline, requested, *, mode, is_slurm, gpu_slurm_given) -> (stage, notes)` in `_cli_embed_stage.py`, which the CLI then calls too (recommended)** | One producer; the GUI passes only what it has | A small refactor in #282's module, in PR B |

**[call]** Pick (c). The pure pieces already exist (`resolve_embed_stage` at `:132`,
`pipeline_embedders` at `:88`). Only `_cpu_pinned_key`, `_unstageable_embedder` and
`_has_gpu_detector` (`:109, 124, 471`) are private, and (c) keeps them private. The GUI
passes `mode="full"` or `mode="measure"` from PB5's selector. `resolve_embed_stage`'s
`gpu_profile = gpu_slurm_given or (mode == "full" and has_gpu_detector)` (`:173`) then applies
§4.1's measure rule with no GUI-side copy.

### PB3. Stage-4 refusals in `rc-staged-gpu-refusal`

**Today.** `_staged_gpu_capability` catches `UnstageableGpuDetectorError` only
(`_callbacks.py:268-271`). The alert reads "This pipeline cannot be run: {refusal}"
(`:1856-1858`). Run is disabled while it is open (`update_run_disabled`, `:2839-2855`). The
launch seam re-checks before allocating (`:2039-2041`).

A Stage-4 refusal today reaches the user only as an exit-2 failure with stderr: a
`SlurmSubmitError` on SLURM, "local process exited with status 2" locally.

| Refusal | Where (`_cli_embed_stage.py`) | Inputs | GUI |
|---|---|---|---|
| R1 `--embed-stage gpu` with `--mode process` | `refuse_embed_stage_for_process`, `:177-187` | mode | **cannot occur**: the GUI has no process mode. No GUI code |
| R2 `gpu` with a CPU-pinned or stageless embedder | `refuse_unstageable_embedders(pipeline, requested, resolved)`, `:190-226` | pipeline, control, mode | shown statically, in full and measure mode. Inputs gain `rc-radio-embed-stage` |
| R3 explicit flag conflicting with the recorded stage | `refuse_embed_stage_switch`, `:229-281` | recorded `embed_stage` (`EMBED_STAGE_CONFIG_KEY`, `:83`, in `processing_state.json`) and the control | full mode only, when the output already holds a run. Never in measure (§3.4) |
| U12 CPU-profile refusal (recorded `gpu`, `auto`, SLURM, no GPU profile) | inside `adopt_recorded_embed_stage`, `:284-347` (raised `:315-321`) | recorded stage, control, mode, and whether `--gpu-slurm` is set | full mode only, as R3 |

**Behaviour.**
- **The probe.** It becomes `_launch_refusal(pipeline_path, cli_mode, exec_mode, embed_stage,
  gpu_slurm_given, output_dir) -> str | None`. It runs the CLI's checks in the CLI's order
  (`phenotypicCLI.py:3310-3314, 3334, 3384-3402`), catching `click.UsageError`:
  - the `UnstageableGpuDetectorError` check, **in full mode only**, because
    `uses_staged_gpu_strategy` returns `False` for measure (§3.4);
  - R2;
  - R3/U12, in full mode only;
  - PB5's measure refusals, in measure mode.
- **The alert text** is the CLI's message, unchanged, under the existing "This pipeline cannot
  be run:" prefix.
- **Inputs.** The callback adds `RC_RADIO_EMBED_STAGE`, PB5's mode selector, the GPU-profile
  textarea and `RC_STORE_FORM_STATE`, for the output directory. That last pattern already
  exists in `show_reference_metadata_requirement`.
- **The recorded stage** is read with `load_processing_state` (`_cli_state_management.py:249`).
  A missing state means no R3/U12 check.
- **[call]** The run console has no `--restart` or `--overwrite` (`_state.py:67-113`), so in
  full mode an output holding a processing state is a continuation. That matches the CLI's
  `continuing` (`phenotypicCLI.py:3242-3248`) for every full-mode argv the GUI can produce.
- **The launch seam** (`_callbacks.py:2039-2041`) calls the same function, so a click racing
  a form change is refused before allocating.

**Approaches for R3/U12:**

| | Approach | Pro | Con |
|---|---|---|---|
| a | Static, with the CLI's functions (recommended) | instant, same text, Run disabled | must read `processing_state.json` on each form change, a small JSON read |
| b | Run `--dry-run` in the background on each change | the CLI decides everything | spawns a process per edit; slow on a large inventory; and impossible in measure mode (§3.4) |
| c | Leave R3/U12 to the launch failure | no work | the user learns after clicking Run, from a log |

Pick (a). The CLI stays the final authority: a refusal the probe misses still fails the launch
with exit 2, as today.

### PB4. Ledgers, tutorial text and screenshots

- **`FEATURES.md`:**
  - new rows:
    - the Embed-stage control;
    - the Auto-resolution hint;
    - the GPU section itself, which has no row today (`FEATURES.md` has only `:443` and
      `:455` around it);
    - the Stage-4 refusals;
    - the measure-mode selector, its disabled Validate, and its form rules (PB5);
  - update `:455` "Unstageable GPU pipeline refusal" to cover R2, R3 and U12.
- **`WORKFLOWS.md`.**
  - Update `run_local` (`:40`) and `run_slurm` (`:41`) to name measure mode and Stage 4.
  - `_capture_run_slurm` (`scripts/capture_gui_tutorial_screenshots.py:1085`) sets GPUs to 0
    (`:1139`) and never opens the GPU section.
  - **[call]** Add one frame of the GPU section for a DINO-only pipeline with Auto → GPU, and
    one frame of the mode selector set to Measure in `_capture_run_local`
    (`scripts/capture_gui_tutorial_screenshots.py:1058`). These show new chrome, which D4 (a
    status, not chrome) does not cover. No new workflow row: measure is a variant of the two
    run workflows, not a new end-to-end flow.
  - Regenerate the full set.
- **Tutorial text:** `04_run_local.md` and `05_run_slurm.md` gain a measure-mode section, with
  what it refuses (§3.4) and what the GUI cannot do (PB5).

### PB5. Measure mode in the run console (D6)

**Today.**
- `to_argv` always emits `--mode full`, with `--input` required (`run_console/_state.py:540-566`).
- `ExecutionMode` is `Literal["local","slurm"]` (`sdk_/typing_.py:212`); it is the execution
  mode, not the CLI mode.
- `RunMode` on the registry record is `Literal["local","slurm","validate","unknown"]`
  (`shell/_runs_registry.py:88`).
- No GUI site emits `--mode measure`.

**The selector.** A two-way radio, `rc-radio-cli-mode`: **Full** (default) or **Measure**.
It sits beside `rc-radio-mode` (local/SLURM, `_ids.py:105`), because the two are independent.
- **The state field.** `RunConsoleState.cli_mode: Literal["full","measure"] = "full"`. It is
  read tolerantly from older presets, and it is part of the command digest.
- **The argv.** `to_argv` emits `--mode <cli_mode>`.

**Inputs measure needs, and what the form does with each** (CLI rules in §3.4):

| Form slot | Full | Measure |
|---|---|---|
| Pipeline | required | required (`:3121-3123`) |
| Input folder | required | **hidden and never sent**: `--input` is refused (`:2801-2806`) |
| Output folder | any | **must exist and hold a forward run**. The form checks existence (`:3124-3130`) and `_output_holds_a_run` (`shell/_runs_registry.py:187-200`), and otherwise shows "Measure re-measures an existing output; choose a folder a previous run wrote" with Run disabled |
| Metadata | sent; snapshotted | sent; used for the join, **not snapshotted** (`:1331`). The help text says so. **[call]** Keep the handoff, because the CLI accepts and uses it |
| `--retry-failures` | offered | **disabled and never sent** (`:3100-3105`) |
| Advanced `sample` | offered | **disabled and never sent** (`:3113-3119`) |
| Advanced `nrows`/`ncols`/`image_type`/`workers` | sent | sent; the CLI accepts them (no measure refusal) |
| `--slurm`, `--gpu-slurm`, `--gpu-shards`, `--embed-stage` | sent | sent; Stage 4 is measure's re-embed path (U4) |

**[call]** Never send a refused flag; don't rely on the CLI to refuse it. The argv-builder test
asserts that no measure-mode argv contains `--input`, `--dry-run`, `--retry-failures`,
`--sample`, `--restart` or `--overwrite`.

**Static refusals in measure mode**, shown in `rc-staged-gpu-refusal` with Run disabled
(PB3):
- **A pipeline whose measurers read reference metadata.** Shown with the CLI's message from
  `_refuse_measuring_reference_pipeline` (`phenotypicCLI.py:761-773`). The existing
  `rc-reference-metadata-required` alert describes a *full* run's requirement, so it stays
  closed in measure mode.
- **An output with both `metadata.csv` and `metadata.tsv`.** Shown with the
  `resolve_metadata_snapshot_path` message (#281 B5).
- **R2**, from PB3.

**Validate in measure mode.** The CLI refuses `--dry-run` with measure (`:2811-2814`), so
there is no dry run to drive.

| | Approach | Pro | Con |
|---|---|---|---|
| **a** | **Disable Validate in measure mode, with the tooltip "Measure has no dry run (the CLI refuses --dry-run with --mode measure)". The static refusals above are the pre-launch check (recommended)** | Honest; no CLI change | No `sbatch --test-only` profile check before a SLURM measure |
| b | Validate runs `--mode full --dry-run` against the same output | exercises the SLURM profile preflight | Validates a different command: full mode's input scan and refusals, not measure's, and it needs an `--input` measure doesn't have |
| c | Add a measure dry run to the CLI | a real preflight | A CLI feature outside both PRs' scope, and #282 decided against it |

**[call]** Pick (a). The launch seam also refuses `action == "validate"` with
`cli_mode == "measure"` (`_callbacks.py:2063-2066`), so a stale browser cannot send it.

**How the registry records a measure run.**
- **The field.** `RunRecord` gains `cli_mode: Literal["full","measure"] = "full"`, persisted
  and read tolerantly. `mode` (local/SLURM/validate) keeps its meaning.
- **Display.** Recent Runs and the banner show `measure` beside the execution mode.
- **Local exit.** `observe_local_exit` reads the exit code as the authority for a measure run:
  - `0` gives `complete` (U16);
  - `3` gives `complete_with_failures`;
  - `1` gives `failed`, with the CLI's last stderr line, which names the unfinished images
    (U16).

  `_local_completion_evidence_conflict` (`_runs_registry.py:691-770`) is **not** applied,
  because a measure pass writes no forward completion marker. The marker it would read is the
  forward run's, and that says nothing about this pass.
- **N for a local measure exit `3`.** **[call]** It comes from the terminal-failure journal rows
  this measure pass wrote (measure work ids, #281 P2 I5). If the plan finds no per-pass count
  on disk, the status is `complete_with_failures` with N unknown, shown as
  "complete · images failed". It is never `resolve_run_state`'s forward count, which cannot
  see measure failures (`_cli_wait.py:269-285`).

**A SLURM measure: what the observer shows.** The GUI never runs `phenotypic-wait`, so it never
sees exit `5`. It derives the same outcome from disk through `finished_run_outcome` (PA4),
which already contains measure's rules: `_generation_failures` counts measure failures by
epoch, and a fence closed with a `terminal_status` is never "published" (`_cli_wait.py:182-242`).
At the inactive-fence branch (`_slurm_observer.py:661-671`):

| Outcome | Status | Detail |
|---|---|---|
| `COMPLETE` | `complete` | "measure pass complete" |
| `COMPLETE_WITH_FAILURES` | `complete_with_failures`, N from `_generation_failures` | "N images failed to re-measure; their previous measurements and embeddings are kept" (U8) |
| `ENDED_UNPUBLISHED`, with the fence `closed_by: finalizer` and `terminal_status: "failed"` | **`incomplete`** | the fence's `terminal_error`, which is `unfinished_measure_line`: "re-run the same command" |
| `ENDED_UNPUBLISHED` otherwise (cancelled, finalizer died) | `failed`, or `cancelled` per the existing cancel branch | as today |

**[call]** A U16 leftover is `incomplete`, not `failed`. `incomplete` already means "did not
finish; re-running resumes it" (`FEATURES.md:585`), which is U16's own remedy. The observer
gains a read of `terminal_status`/`terminal_error`; today it reads neither.

**The embed-stage resolver in measure mode.** `resolve_embed_stage_for_pipeline` (PB2) is
called with `mode="measure"`:
- only `--gpu-slurm` counts as a GPU profile, so the GPU section is shown for a Stage-4-capable
  pipeline (or the `gpu` control) and never for a `GpuDetector` alone;
- R2 applies;
- R3/U12 do not (§3.4).

**What measure mode in the GUI cannot do**, stated plainly in the tutorial text and the
FEATURES row:
1. **No Validate.** There is no dry run, so no `sbatch --test-only` check of a SLURM measure's
   profile before submission.
2. **No measure beside a live forward run.** The CLI allows a measure pass to run beside live
   forward workers (`_refuse_measuring_reference_pipeline`'s docstring, `:764-765`). The GUI
   holds one generation per output, and its claim gate refuses an output a live run claims.
3. **No measure over a CLI-launched run that has failures.** The claim gate's "failed non-GUI
   processing state" refusal stays (D2), so such a run is measured from the CLI.
4. **No reference-metadata pipelines.** The CLI refuses them.
5. **No per-image leftover list.** A SLURM pass's unfinished images are named in the
   finalizer's log, not on disk in a form the GUI reads. The GUI shows the one-line count and
   points at the log.
6. **No exit `5`.** The GUI derives the waiter's outcome rather than receiving it, so the
   "unfinished" state appears when the observer next polls the fence, not at a process exit.

**Tests:**
- **`test_state.py`:**
  - the measure argv is `--mode measure --pipeline … --output …` plus the allowed flags;
  - the full set of refused flags is absent;
  - a preset without `cli_mode` loads as full.
- **`test_run_console_callbacks.py`:**
  - the input slot is hidden in measure mode;
  - Validate is disabled, and the seam refuses a stale validate;
  - a missing or empty output disables Run with the message;
  - a reference-metadata pipeline shows the CLI's refusal;
  - the GPU section appears for a DINO-only pipeline in SLURM measure mode and not for a
    `GpuDetector`-only one;
  - R3 is never evaluated in measure mode.
- **`test_runs_registry.py`:**
  - a measure record round-trips `cli_mode`;
  - local exits 0, 3 and 1 map as specified, without the forward-marker check.
- **`test_slurm_observer.py`:** the four outcome rows, including a U16 fence giving
  `incomplete` with the `terminal_error` text.
- **An integration test against the real CLI:** each argv the form can build for measure mode
  is accepted by the CLI's argument parsing. Drive `phenotypicCLI` with Click's `CliRunner`
  up to the first non-usage step, so no flag the GUI sends is refused.

---

## 6. Test strategy

**Instruments by stage**, per the root `CLAUDE.md`:
- **Per task:** the directly touched test files, about a minute, run on Slurm
  (`--partition=batch` or `intel`, explicit `--mem`/`--time`, never `preempt`, never
  `-n auto`, `QT_QPA_PLATFORM=offscreen`, `-o addopts= -m "not slow"`).
- **Per phase:** the affected surface, derived from importers rather than directory names.
  - **PA1/PA4:** `tests/unit/gui/run_console/`, `tests/unit/gui/shell/`,
    `tests/integration/gui/test_run_console_callbacks.py`, `tests/unit/gui/test_runs_registry_layout.py`,
    `tests/unit/gui/test_snapshot_status.py`, and the `_cli_wait` tests.
  - **PA2/PA3:** `tests/unit/gui/results_viewer/`.
  - **PA5/PA8:**
    - `tests/unit/cli/`, `tests/integration/cli/`, `tests/unit/sdk_/`, `tests/gui/results_viewer/`
      and `tests/unit/gui/results_viewer/`;
    - **every importer of `tests/_output_layout.py`** (62 files), because its
      `write_measurements_mirror` default changes;
    - the error-tab and `_cli_error_outputs` tests.
  - **PA6:** `tests/unit/gui/{builder,shell,tune}/`, `tests/integration/gui/tune/`,
    `tests/integration/gui/test_tune_*.py`, `tests/unit/gui/analysis/`.
  - **PA9:** `tests/unit/gui/shell/`, `tests/unit/gui/builder/test_reference_metadata.py`,
    `tests/unit/gui/tune/`, `tests/unit/analysis/` (the `ExpectedVsDetectedCount` change),
    `tests/gui/browse/`.
  - **PA10:** `tests/unit/gui/results_viewer/`, `tests/unit/gui/analysis/`,
    `tests/unit/gui/test_curation_does_not_unfence_the_run.py`, and the curation tests under
    `tests/gui/results_viewer/`.
  - **PB1–PB5:** `tests/unit/gui/run_console/`, `tests/integration/gui/test_run_console_callbacks.py`,
    `tests/unit/gui/shell/`, `tests/unit/cli/` (the `_cli_embed_stage` refactor).
- **The GUI lanes:**
  - **`testpaths`:** `tests/unit`, `tests/integration` and `tests/gui`.
  - **The Playwright e2e:** `PLAYWRIGHT=1 uv run pytest tests/e2e/gui`. It is outside
    `testpaths` (`tests/e2e/conftest.py:51`), and `ci_flaky` is deselected as in CI. It runs
    once per PR, after the fixture-name flip (PA6), with at least the run-console, results
    viewer, analysis and tune specs.
- **The ledger gates:** `scripts/check_features_md.py`, `--strict` for the merge gate, and
  `scripts/check_workflows_md.py -v`. These are the `features-md-gate` and
  `workflows-md-gate` CI jobs.
- **The 13 reds:** all green, with no `xfail`, skip, deselect or `-k`, in the PR A per-phase
  run (§4 PA3). A red outside the 13 is a real failure.
- **The docs build** (PA8 edits how-to pages and a notebook): one Slurm job,
  `sphinx-build -j "$SLURM_CPUS_PER_TASK" -D nbsphinx_execute=never`. Then read the rendered
  HTML of the changed pages, since an exit code of 0 does not mean the pages rendered.
- **The live SLURM lane.**
  - **CPU:** `PHENOTYPIC_RUN_LIVE_SLURM=1`, `tests/integration/gui/test_run_console_slurm_live.py`,
    run once on PR A after its rebase and once on PR B. It is the only check that runs a real
    exit `4` through the GUI.
  - **GPU (D7):** one live GPU-stage SLURM launch from the GUI, in PR B:
    - **The pipeline.** A DINO-only pipeline (DINOv2 `small`, weights cached), over the live
      harness's one 96×96 TIFF. Auto resolves to GPU Stage 4 through the GUI's GPU section.
    - **Profiles.** The CPU stages use `--slurm partition=batch` (or `intel`) with explicit
      `mem`/`time`. The GPU stage's `--gpu-slurm` profile is either:
      - `partition=exfab account=exfab gpus_per_node=1`; or
      - `partition=short_gpu constraint=gpu_latest time=≤01:55:00` (no account).

      Pick between them by running `sbatch --test-only` for both and taking the earlier
      predicted start. **Never `preempt`.** `short_gpu` rejects a job with no GPU, so it is
      never the CPU profile.
    - **Assertions.** The submitter exits `4` and the record is `queued`. The observer reaches
      `complete`. The store holds the embedding table under its run folder, with
      `device_name` recorded.
    - **Harness work.** The harness (`tests/_support/live_slurm`) admits only "bounded
      CPU-only resources" (`test_run_console_slurm_live.py:1-9`), so this case needs a
      bounded GPU profile added to it. That is plan work in PR B, opt-in under its own flag
      (`PHENOTYPIC_RUN_LIVE_SLURM_GPU=1`).
    - **When.** It runs once before PR B is opened, after PR B is cut on the new base.
- **End of each PR:** the full sharded regression once (the committed
  `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch` recipe,
  env synced with `--extra tune --extra topology`). For PR A, run it **after the rebase onto
  `origin/main`**; a pre-rebase run is not the PR's baseline. For PR B, run it on the base
  that contains PR A.

---

## 7. Risks

- **R1. PR A edits two tests to green them (#7, #8).** A reviewer could read that as
  weakening. The mitigation is the property-preservation rule in §3.3, stated in each
  commit message.
- **R2. A new terminal status reaches code that enumerates statuses** without going through
  the shared sets:
  - `results_viewer/_qc_tab/_rebuild.py:171-172`;
  - `results_viewer/_discovery_contracts.py:16`;
  - any JS that keys on the CSS class.

  The mitigation is a grep gate in the plan (every `"complete"` / `"failed"` literal under
  `_gui/` is reviewed) and a test that renders each `RunStatus` in Recent Runs.
- **R3. The error-analysis schema bump touches a rollback-capable publisher** shared with the
  CLI's recompile re-emit. A version-1 journal on disk mid-rollback must still validate. Both
  directions are pinned with tests.
- **R4. The two aggregate-proof checks disagree after curation** (§3.2). PA4 and PA10 avoid
  `valid_aggregate_snapshot`.
  - **Where the strict check still applies.** The waiter's `_published` uses it as its third
    disjunct (`_cli_wait.py:238-242`). A SLURM generation whose fence its finalizer did not
    close, curated before a `phenotypic-wait` polls it, would read unpublished.
  - **Why the risk is small.** The first disjunct (fence closed by finalizer) covers every
    normal SLURM finish.
  - **Scope.** This predates PR A, but PA10 makes curation reachable on more runs. Recorded,
    not changed.
- **R5. `test_measurement_join_migration_run.py`** was skipped at #281's gate and again here
  (the live run is absent). Its v2 state is unverified until that run is republished.
- **R6. PA8 removes a deliverable users open in Excel**, and that a dashboard download
  command and outside scripts may name. Excel opens a `.tsv`, but by import, not by
  double-click on every platform. The README and the how-to pages say so, and an old tree
  keeps its CSV until it is finalized again.
- **R7 (PR B). The probe reads `processing_state.json` on every form change.** It is bounded
  to one small JSON read and memoised on mtime, like the classifier (`shell/_classifier.py:201-216`).
- **R8. The shared test helper's default flips** (`write_measurements_mirror(csv=False)`),
  reaching 62 importers. Mitigation: run every importer at the PA5/PA8 phase gate, and
  attribute each red in isolation before acting on it.
- **R9. The merge-order window (D5).** #281 merges to `main` before PR A. Until PR A merges,
  `main` carries every break in §3:
  - a SLURM launch that can read `failed`, or cancel its own generation;
  - an empty Colony picker;
  - a tune Deploy refusal;
  - 13 red GUI tests.

  #281's own spec forbids that order (§2.2), and #283 says the same. See §11 Q1.
- **R10. Rebase drift.** PA citations are at `36b9e840e`. #281 may change again before it
  merges, and `main` will hold other merges. Every citation, and the 13-red table, is
  re-verified after the rebase, and the reds are re-run there (§1).
- **R11. PA9 changes a public analyzer's accepted input** (`ExpectedVsDetectedCount` gains
  `.tsv`). It widens the input set and refuses nothing it accepted before.

---

## 8. Follow-up issue (draft, not to be opened by this work)

> **Title:** GUI: run picker, DVC commit hint, embeddings view (follow-up to #283)
>
> - **Run picker.** The viewer reads a store's latest measuring run
>   (`tables/<n>-<sha12>/`, #281 D6a). Let users pick another run, and always say which
>   run is shown. That run comes from `latest_measurement_run`, and is `<n>-unknown` when the
>   pipeline digest is unknown.
> - **DVC hint.** After curation edits `deliverables/` in a DVC-tracked output, say that
>   `dvc commit` is needed (#281 C9 documents the step).
> - **Embeddings view.** Show which embedding tables a store holds
>   (`tables/<run>/embeddings/<model>/<layer>/<region>/`, `TableRun.embedding_path`).
>   Viewing vectors is out of scope there too, unless decided otherwise.
> - **Restored fence.** A `dvc checkout`-restored active lifecycle fence reads `active` in the
>   GUI launch check until the next CLI run retires it (#281 hand-off, P2). Ask the scheduler,
>   as `_refuse_live_slurm_run` does, or call the same retirement.

---

## 9. My judgement calls (for review)

1. **The two stale tests.** #7 and #8 are fixed by test edits that keep each test's property
   (§3.3).
2. **SLURM exit `0`/`3` means "finished without submitting"**, and is never bound to an
   earlier generation's job set (PA1 b).
3. **One sdk_ helper names the payload a record-less reader reads**, for both layouts (PA2 b).
4. **One public `finished_run_outcome` in `_cli_wait.py`**, used by the waiter and every GUI
   entry point. Its generation-less callers also require `aggregate_proof_is_current`, and it
   never asks `valid_aggregate_snapshot` (PA4).
5. **A new terminal status `complete_with_failures` plus `failed_images`**, coloured
   `--oi-orange-text` (PA4 a).
6. **A failed TSV write unlinks the TSV.** A stale CSV is retired only after a successful TSV
   write (PA5).
7. **Error analysis moves to TSV** under a version-2 receipt. Version 1 is still read, and the
   live CSV is retired after publishing (PA5).
8. **One sdk_ predicate, `is_pipeline_config_path`, for every listing surface** (PA6).
9. **PA9 extends `ExpectedVsDetectedCount` to `.tsv`**, so the tune pickers' new suffix is
   usable.
10. **The mutation receipt records the failed count N** (PA10).
11. **The embed-stage control** sits outside the SLURM-only section, is disabled without an
    embedder, and emits nothing for `auto` (PB1).
12. **A local no-CUDA `gpu` run surfaces as a failed run**, not in the refusal alert (PB1).
13. **One resolver.** A public `resolve_embed_stage_for_pipeline` is factored out of
    `resolve_embed_stage_for_config`. The GUI passes `mode` from PB5's selector (PB2 c).
14. **R3/U12 are predicted statically** from `processing_state.json`, in full mode only (PB3 a).
15. **Two new frames in PR B** (the GPU section, the mode selector), and no new workflow row
    (PB4).
16. **Measure mode never sends a flag the CLI refuses**, and it keeps the `--metadata` handoff
    (PB5).
17. **Validate is disabled in measure mode** (PB5 a).
18. **A measure run's local exit code is authoritative.** N comes from the pass's own journal
    rows, or is shown as unknown (PB5).
19. **A U16 leftover SLURM measure reads `incomplete`** with the fence's `terminal_error`
    (PB5).

---

## 10. Decisions (user, 2026-11-24)

These settle the eight questions of the first draft.

- **D1. Retire `measurements.csv`.** PR A includes the CLI change (PA8). Curation writes
  `.parquet`, then `.tsv` (PA5). Readers still fall back to `.csv` in older trees.
- **D2. Curation of a `complete_with_failures` run is allowed** (PA10). Claiming a
  CLI-launched run for `--retry-failures` stays refused.
- **D3. The snapshot badge is in scope:** "Run complete · N images failed" (PA4).
- **D4. No new tutorial frame for the status.** It is documented in text (PA7).
- **D5. PR A targets `main` after #281 merges.** It is built on #281's branch now, and before
  it is opened it is rebased onto the latest `origin/main`, with every citation re-verified.
  PR B is cut after #282 has been rebased or merged onto a `main` that contains PR A (§1).
- **D6. Measure mode joins the run console in PR B** (PB5).
- **D7. One live GPU-stage SLURM launch from the GUI in PR B,** on `exfab` or `short_gpu`,
  never `preempt` (§6).
- **D8. TSV-capable metadata pickers join PR A** (PA9).
- **D9. Hold #281, then merge back to back** (settles R9, the `main` window). PR A is built and
  reviewed on #281's branch first. When PR A is ready, #281 merges to `main`; PR A is then
  rebased onto the fresh `origin/main`, its citations, the 13-red table and its checks are
  re-run there, and it merges at once. #281's merge gate ("not before #283's fixes") is
  thereby kept in substance: `main` never sits on the §3 breaks beyond that short gap.

---

## 11. Open questions (only the user can answer)

None. D9 settled the last one (the `main` window).
