# Design: GUI follow-ups for #281 (DVC, TSV, per-pipeline tables) and #282 (Stage 4)

**Status:** draft for review · 2026-11-24
- **Revision 2:** the user's answers to the eight open questions (D1–D8).
- **Revision 3:** the independent spec review
  (`docs/superpowers/reports/2026-11-24-gui-dvc-followups/spec-review.md`) and the user's
  answers to its questions (D10–D13). §12 maps each finding to its fix.

**Issue:** #283 (PR A); the #282 GUI items (PR B)
**Bases, read for every citation below:**

- **PR A:** `origin/feat/dvc-embeddings-refactor` @ `36b9e840e` (#281). This branch is
  `feat/gui-dvc-followups`, worktree `.claude/worktrees/gui-followups`. PR A is built here
  and targets `main` after #281 merges (§1, D5).
- **PR B:** `origin/feat/gpu-embedding-staging` @ `0993c8e06` (#282).
  - #281 is an ancestor of #282.
  - #282 changes nothing under `src/phenotypic/_gui/` or in `sdk_/_exit_codes.py`
    relative to #281, so a PR A citation of GUI code holds on #282 too.
  - #282 does change `_cli/_cli_wait.py` (123 lines changed: +99/−24), which PA4 also
    refactors (§1, R13).
  - Line numbers differ between the bases. Each PR B citation below is at `0993c8e06`.

**Scope** was decided by the user on 2026-11-24: two PRs, plus one follow-up issue that is
drafted here but not opened. The user's answers the same day widened both PRs (§10).

Each judgement call of mine is marked **[call]** and collected in §9. The user's decisions
are cited as **D1–D14** (§10). §11 records that no question is open.

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
- lets a run that finished with failed images be curated (D2) and relaunched from the GUI,
  whoever launched it (D10);
- lets every metadata picker take a `.tsv` (D8).

**PR B** lets the run console reach Stage 4:
- an `--embed-stage` control;
- a GPU section shown whenever the CLI would use it;
- Stage-4 refusals shown the way the staged-GPU refusal already is;
- `--mode measure` (D6).

**Sequencing (D5, D9).**
1. **Now.** PR A is built on #281's branch.
2. **#281 is held until PR A is reviewed and ready (D9), then merges to `main`.** PR A
   targets `main` and merges right after it.
3. **Before PR A is opened:**
   - `git fetch origin` and rebase `feat/gui-dvc-followups` onto the latest `origin/main`,
     which will by then contain #281;
   - re-verify every PR A citation in this spec against the rebased tree. Line numbers will
     move. A cited behaviour that changed is a spec change, not a plan detail.
4. **#282 is rebased or merged onto a `main` that contains PR A.** That is #282's obligation,
   and it includes one real conflict (I2, R13):
   - #282 rewrites `_cli/_cli_wait.py`: `_published` gains the U16 `terminal_status`
     override (`0993c8e06 _cli_wait.py:204-279`, override at `:228-257`), `_ledgered_jobs` is added (`:182-195`), and
     `wait_for_run` gains a frozen-outcome drain (`:506+`).
   - PA4 makes `decide_wait_outcome` compose the new sdk_ verdict, and adds the
     generation-bound wrapper the SLURM observer uses.
   - The rebase keeps both. A test after it pins two things: the wrapper returns
     `ENDED_UNPUBLISHED` for a same-generation fence carrying `terminal_status`, and the
     waiter still drains `_ledgered_jobs` (§6).
5. **PR B is cut only after step 4.** PR B needs PA1, because a Stage-4 SLURM submission
   exits `4` like any other, and it needs PA4's verdict (PB5).

#281's own merge gate says "#281 does not merge to `main` before #283 fixes these" (#281
spec §2.2). D9 keeps its intent: the two merge back to back, so `main` carries the §3 breaks
only for the gap between the merges. See §7 R9 and D9.

---

## 2. Goals, non-goals, anti-goals

### 2.1 Goals

- **G1.** A SLURM launch the CLI queued is shown as queued, in every case.
  - No queued run reads as failed.
  - The GUI never cancels a generation because the submitter exited `4` or `2`.
- **G2.** A run that exits `3` (`COMPLETE_WITH_FAILURES`) reads "complete, N images failed"
  wherever the GUI states a run's outcome:
  - Recent Runs;
  - the run banner;
  - the results-viewer and analysis snapshot badge (D3).

  It reads the same whether the run was local or SLURM. It is shown only while the run's
  aggregate publication still covers the run as it now stands (C1), or, when every image
  failed, while none exists (D16).
- **G2b (D2, D10).** Such a run can be curated, and relaunched from the GUI for Run,
  `--retry-failures`, Validate and Measure, whoever launched it. Its failed images have no
  rows, and curating it leaves every enforced run-proof descriptor valid.
- **G3.** The Colony picker and the measurement routes read a v2 store's latest measuring run,
  `tables/<n>-<sha12>/measurements/table.parquet`, and still read a v1 store.
- **G4.** The 13 expected-red tests (#282 plan README) are green, through the fix each one
  needs. No `xfail` and no skip.
- **G5.** Every tabular text file PhenoTypic writes for the mirror or the GUI is TSV:
  - `measurements.csv` is no longer written (D1);
  - every reader prefers `.tsv` and falls back to `.csv` in an older tree;
  - an older tree's generated `measurements.csv` is retired after the TSV is published;
  - `error_analysis.csv` stays CSV for now (D18; §8).
- **G5b (D8).** Every metadata picker and table reader in the GUI accepts `.tsv` wherever the
  CLI does.
- **G6.** Every *mounted* GUI surface that lists, filters, loads or writes a typed config
  agrees with the sdk_ suffix sets:
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
  - an embeddings view;
  - the unmounted tune app's suffix surfaces.
- **Tune runs' exit codes.** `phenotypic.tune` does not use `RunExitCode` at #281: there is
  no reference under `src/phenotypic/tune/`, and its exits are `SystemExit(message)` at
  `tune/__main__.py:236` and `tune/_tune_cli/_worker.py:97,102`. The tune monitor's reap
  logic (`_gui/tune/_callbacks.py:741-745`) is therefore unchanged.
- **Unmounted chrome.** The hub does not compose `_gui/tune/`, so `/tune/` 404s
  (`_gui/CLAUDE.md:20-26`). PR A changes tune code only where:
  - a red test needs it (#1–6);
  - the user named the site (D8's tune pickers).
- **Reapplying curation in the CLI's finalize (D11).** A CLI re-run of a run with failures
  reseeds the mirror without curation labels; see PA10.
- **Claiming an unfinished or live run from the GUI.** It stays refused (PA11).
- **`--mode recompile`, `process` and `migrate` in the run console.** PR B adds `measure`
  only (D6).
- **Changing `resolve_run_state`'s `Completion` literals, or the CLI's exit codes.**
- **A tutorial screenshot of the new status (D4).** It is documented in text instead.
- **Editing anything outside the repository (D12).**

### 2.3 Anti-goals (failure even if everything "works")

- A queued SLURM run shown as failed, or a submission's generation cancelled by the GUI.
- A run shown as complete, or made writable, when:
  - an accepted image has neither a success nor a terminal-failure record; or
  - its aggregate publication no longer covers the run's inventory, configuration, metadata
    or verified set (C1).
- A curation write that makes an enforced proof descriptor (master, `metrics.json`,
  `schema.json`) stop matching, or that changes the run's verdict.
- A curated `measurements.tsv` that disagrees with `measurements.parquet` after a curation
  write that reported success, or a stale text mirror a reader can fall back to.
- A persisted non-terminal status that no process will ever move, which locks the output.
- A GUI-side copy of a CLI decision rule (Stage-4 resolution, a refusal condition, the
  "finished with failures" verdict, measure's argument rules) that can drift from the CLI's.
- A test made green by deleting or weakening the property it pins.
- A legacy `.json.pht-*` config, or a legacy `measurements.csv` tree, that the GUI used to
  open and no longer can.
- An argv the GUI builds that the CLI refuses as a usage error, or one carrying a flag the
  user cannot see.

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

- **The ordinary case reaches `queued`.** But every submission is now labelled `reconciled`
  (`reconciled=ambiguous_error is not None`, `:871`), which erases the signal that something
  went wrong.
- **If the scheduler evidence is slow to land, a queued run reads `failed`.** The registry
  takes `status="failed"` from the `except Exception` arm at `run_console/_callbacks.py:1086-1114`.
- **If the fence reads inactive, the GUI cancels its own generation.** The branch is
  `_slurm.py:652-656`, and `_reconcile_submission` reaches it from every caller, the clean
  exit with no job set (`:837-841`) included.
- **Exit 2 also reconciles and can cancel.** A usage error is raised in the CLI's read-only
  half, before anything is submitted, yet `test_abnormal_exit_cancels_recovered_inactive_generation`
  (`tests/unit/gui/run_console/test_slurm.py:815-838`) pins a `cancel_generation` call on a
  *previous*, inactive generation. A deactivated fence is never relabelled
  (`_cli/_cli_slurm_lifecycle.py:700-704`), so the effect is mostly harmless. But the cancel
  can `scancel` ids left without a terminal ledger row, and PR B adds more usage errors.
- **When a SLURM-mode re-run exits `0` or `3` without submitting** (review C4):
  - A **complete** run with nothing left to do exits `0` (`phenotypicCLI.py:3897-3899`, `:3964`).
  - A run with failures and **at least one verified image** never takes that path. Its
    startup verdict is never `complete`, so `publication_refresh_required` is set
    (`:3765-3790`). With nothing left to process it goes to `_handle_recompile_slurm`
    (`:3925-3944`), which submits whenever a measurement source exists (`:4672-4760`) and
    exits `4`.
  - Exit `3` without a submission comes only from:
    - a run with failures and **no** verified image, where nothing needs refreshing (`:3964`);
    - `_handle_recompile_slurm`'s two local fallbacks (`:4673-4688`, `:4700-4713`), which
      run `_handle_recompile` inside the submitter process.
  - Today these exits either fail or bind to the *previous* generation's job set. The stale
    binding is in `read_submitted_job_set`: with no `expected_generation` (`_slurm.py:372-432`),
    `_generation_for` (`:234-261`) prefers the lifecycle on disk. The exit-`0` half predates
    #281. The exit-`3` half is new: the earlier run's fence is inactive, so the same branch
    calls `cancel_generation` on that finished generation.

**Every place the GUI reads a process exit code:**

| Site | Reads | At #281 |
|---|---|---|
| `run_console/_slurm.py:813-831` | the `python -m phenotypic` submitter | **changes** (PA1) |
| `shell/_runs_registry.py:636-688` `observe_local_exit` (`:661`, mode arm `:665-675`) | local and validate `python -m phenotypic` | **changes** (PA4) |
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
- **The review confirmed** that exit `0` never means "submitted" on any path the GUI can
  build. Every SLURM submission exits `SUBMITTED` (`phenotypicCLI.py:4177, 4194, 4800`).
  `--force-local`, the only flag that turns `--slurm` local (`_cli/_cli_types.py:298-302`),
  cannot be produced by the GUI's argv builder.

### 3.2 "Complete with failures" has no representation anywhere in the GUI

- **Local.** `observe_local_exit` maps every non-zero exit to `failed`
  (`shell/_runs_registry.py:661-663`).
- **SLURM.** A run with any terminal failure never gets a run proof.
  `publish_run_completion_evidence` raises unless every accepted image succeeded
  (`_cli/_cli_completion.py:1304-1342`). The observer therefore reaches the
  "inactive, no marker" branch and reports `failed`, with "lifecycle ended terminal-incomplete
  without a current run-completion marker" (`run_console/_slurm_observer.py:661-671`).
  - A *staged* run with failures reaches that branch only because the finalizer closes its
    orchestration with the phase `"terminal_incomplete"` (`_cli/_cli_checkpoint_handler.py:398-401`).
  - That phase is absent from `_STAGED_FAILURE_PHASES` (`_slurm_observer.py:92-94`).
    Otherwise `_staged_terminal_observation` would return `failed` first (`:642-649`).
- **Boot rehydration.** `resolve_run_state(...).completion` is `"failed"` for such a run
  (`sdk_/_run_state.py:1922-1934`), and `_rehydrated_status` passes it through
  (`shell/_runs_registry.py:1026-1104`).
- **The viewer and analysis snapshot badge** reads "Run failed". The unfinished arm returns
  *before* the currency check (`_gui/_snapshot_status.py:84-91` vs `:92-98`).
- **Curation is refused.** Every write-authority check asks `completion == "complete"`:
  - `_mutation_guard.py:38-45`, used at `:121, 140, 170, 216`;
  - `OutputRoot.run_is_complete`, `_output_root.py:700-713`;
  - `mutation_snapshot_is_safe`, `:780-797`;
  - the processing-inventory reuse rule, `_processing_inventory.py:97`. Only a reusable
    inventory is `"exhaustive"` (`:130-145`), and the guard refuses a non-exhaustive one
    (`_mutation_guard.py:127-131`).
- **Every relaunch is refused, whoever launched the run (review C2).**
  - `_assert_output_claimable_locked` (`shell/_runs_registry.py:1275-1300`) lets a terminal
    owner record pass (`:1288`). It then calls `_foreign_run_conflict(output_dir)`
    unconditionally (`:1298-1300`).
  - That function never looks at the owner, and refuses `completion == "failed"` with
    "output has failed non-GUI processing state …" (`:1469-1474`).
  - `allocate` reaches the gate on every launch, Validate included (`:412-413`).
  - It is pinned by `test_a_failed_image_refuses_the_claim`
    (`tests/unit/gui/shell/test_runs_registry.py:363-390`).
- **A CLI re-run of such a run re-finalizes and reseeds the mirror (review C4).**
  - The re-run reaches finalize through the refresh in §3.1.
  - Finalize reseeds `measurements.{tsv,csv,parquet}` from the post-applied master
    (`_seed_measurements`, `_cli/_cli_output_manager.py:968-1003`).
  - From `curation_labels.parquet` it re-emits only `errors/` and `error_analysis.*`
    (`_cli/_cli_error_outputs.py:56-65`; "(no mirror)" at `:63`).
  - So a curated mirror returns to uncurated, while the labels file survives. A complete
    run's re-run is a no-op and keeps it.

**What already decides "finished with failures".** `phenotypic-wait` decides it from disk, in
`decide_wait_outcome` (`_cli/_cli_wait.py:309-366`), rule 3: completion `failed`, no
unresolved image, and published. Its seams are:
- `_run_snapshot` (`:116-130`);
- `_published` (`:182-242`);
- `_cancelled` (`:245-266`);
- `_generation_failures` (`:269-285`);
- `_is_process_run` (`:288-301`).

Rule 3 never compares the aggregate proof to the run as it now stands.

**The aggregate proof carries the digests rule 1 compares** (`_cli/_cli_completion.py:1244-1265`):
`inventory_digest`, `finalization_input_digest`, `scientific_config_digest`,
`source_set_digest` and `source_image_count`.

For a complete run, `_run_proof_covers_current_inventory` (`sdk_/_run_state.py:1464-1541`)
compares those against current state. Its docstring says why each comparison is
load-bearing. A failed run has no run proof, so nothing makes those comparisons for it (C1).

**Two aggregate-proof checks disagree about the curated mirror:**
- **`valid_aggregate_snapshot`** (`_cli/_cli_completion.py:1271-1301`) hashes **every**
  `required_outputs` descriptor, so it returns `None` after any curation write.
- **`aggregate_proof_is_current`** (`sdk_/_run_state.py:1415-1433`, exported) skips the
  GUI-written descriptors in `_GUI_WRITTEN_PROOF_DESCRIPTORS` (`:1331-1333`, checked at
  `:1369-1371`).

The waiter's `_published` uses the strict one as its third disjunct (`_cli_wait.py:238-242`).

**The readers rule.** `_gui/CLAUDE.md` ("Importing readers: `sdk_`, not `_cli`") makes a new
`_cli` import that answers "is this run finished?" a regression, and
`docs/source/contrib_guide/tracked_state.md:496` says a GUI consumer imports from `sdk_` only.
Its **Known consumers** table (`:433-450`) must gain a row for every new reader in the same
change (`:18, 525`).

### 3.3 Two of the 13 reds are stale tests, and one needs a test edit as well as code

I ran all 13 at `36b9e840e` (Slurm job 32434046, log
`/bigdata/exfab/anguy344/gate-results/guispec/reds-32434046.log`: "13 failed, 17 passed,
5 skipped"). `test_measurement_join_migration_run.py` **skipped** 5 tests: the live run at
`…/ucr_029_e_d_Maresca/data/results/2026-08-11-migration-test` is absent or mid-republish.

| # | Test | Failure observed | What makes it green |
|---|---|---|---|
| 1 | `tests/unit/gui/tune/test_validation.py::test_tuning_spec_path_is_deployable` | `spec_path_issue("tuning_spec.json.pht-tune")` returns the Deploy issue | **code**: `_gui/tune/_validation.py:61` accepts the legacy tune suffix (PA6) |
| 2 | `…/test_setup_authoring.py::test_write_authored_setup_spec_preserves_existing_spec_search_space` | `ValueError: No active knobs to tune.` A legacy `existing.json.pht-tune` is loaded as an `ImagePipeline` | **code**: `_setup_authoring.py:356` `load_pipeline_or_spec` recognises the legacy tune suffix (PA6) |
| 3 | `…/test_setup_authoring.py::test_existing_spec_replaces_scorer_only_when_requested` | same cause, surfacing as the in-memory-layout scorer error | **code**: same |
| 4–6 | `tests/integration/gui/tune/test_setup_view.py::{test_existing_spec_controls_preserve_seed_budget_and_storage, test_existing_spec_trial_override_preserves_authored_strategy, test_existing_grid_spec_never_emits_trial_override}` | the Deploy issue names `.json.pht-tune` | **code**: #1 and #2 |
| 7 | `tests/unit/gui/test_curation_does_not_unfence_the_run.py::test_the_proof_still_records_every_published_artifact` | `required_outputs` holds `measurements_tsv`; the test pins the pre-TSV set (`:135-139`) | **test**: the set becomes `{"master_parquet", "measurements_tsv", "measurements_parquet"}`. `measurements_csv` leaves it because PA8 stops writing that file. The test's own docstring says the proof is "a complete record of what finalization published" |
| 8 | `tests/unit/gui/results_viewer/test_measurement_routes.py::test_an_undeclared_column_never_opens_the_parquet` | `FileNotFoundError` from unlinking `store / MEASUREMENT_TABLE_RELATIVE_PATH` (`:205`), a v1 path no v2 store has | **test**: unlink `declared_measurement_table_path(store)`. The route already reads through `read_embedded_measurement_column` (`_measurement_routes.py:60,107`) |
| 9–12 | `…/test_measurement_source.py::{test_the_picker_offers_the_stores_own_numeric_columns, test_one_unmeasured_image_does_not_empty_the_picker, test_values_are_keyed_by_dataset_image_and_object_label, test_an_unmeasured_image_contributes_no_keys}` | the picker offers nothing (`'Size_Area' in ()`), and no values come back | **code**: `_measurement_source.py:107-108` (PA2) |
| 13 | `…/test_measurement_source.py::test_a_rewritten_table_is_read_fresh` | the same, and the test itself rewrites the v1 path (`:247-249`) | **code and test**: PA2, plus the test rewrites `declared_measurement_table_path(store)` |

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
R3 and the U12 refusal, which sit under `if config.resume:` (`:3346-3402`), never apply to
measure.

**The unstageable-`GpuDetector` refusal does not apply in measure.**
`uses_staged_gpu_strategy` returns `False` for `measure_only` before it looks at the pipeline
(`_cli_execution_strategies.py:1546-1549`). A SLURM measure that embeds on the GPU runs on the
staged controller (`uses_staged_controller`, `:1561-1574`, #282 U10).

**Outcomes:**
- **Exit `0`:** "Exit 0 means every image really was re-measured and re-embedded" (#282 U16).
- **Exit `3`:** `_run_failure_count(output_dir, results.total_failed)` (`:1528-1550`, called
  at `:4622`). That is this pass's failure count when positive, and otherwise the **forward**
  run's `diagnostics.failed`. So a clean pass over a forward run with failures also exits `3`
  (review I3).
- **Exit `1`:** local leftovers (U16).
- **Exit `4`:** at SLURM submission.
- **The GUI completion marker.** A GUI-launched local measure with `run_failed == 0` does
  publish the GUI completion marker (`publish_local_gui_completion`, `:4622-4628`).
- **A SLURM pass with leftovers** ends with the finalizer closing its fence with
  `closed_by: finalizer` plus `terminal_status: "failed"` and `terminal_error` in one write
  (`mark_generation_failed`, `_cli_slurm_lifecycle.py:721-751`). The error line is
  `unfinished_measure_line` (`_cli_staged_controller.py:530-538`).
  - `phenotypic-wait` then returns `5`, through #282's `_published` override
    (`_cli_wait.py:228-257`).
  - The GUI observer never reads `terminal_status`: `grep terminal_status
    _gui/run_console/_slurm_observer.py` finds nothing.
- **A SLURM measure's `COMPLETE_WITH_FAILURES`** can also mean "this pass was clean, and the
  forward run has failures". Rule 3 fires on the forward verdict, whatever `_generation_failures`
  says (`_cli_wait.py:389-403`; `_generation_failures` at `:304-322`).

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
| `SUBMITTED` (4) | queued | The normal path. No `ambiguous_error`. Read `read_submitted_job_set(output_dir)`. When `None`, call `_reconcile_submission(output_dir, cancel_inactive=False)` once; an inactive fence then returns "no jobs" without reconciling (N2). If that also finds nothing, raise one error naming "SUBMITTED". Return `reconciled=False` unless reconciliation found the jobs. |
| `COMPLETE` (0) / `COMPLETE_WITH_FAILURES` (3) | the CLI finished in-process; nothing was submitted (§3.1) | Return a new `SlurmFinishedWithoutSubmission(returncode, output_dir, stdout, stderr)`. Read no job set, bind no generation, never reconcile or cancel. |
| `2` (usage error) | refused before anything was submitted | **Does not reconcile.** Raise the refusal with the CLI's stderr; never cancel, never bind. A usage error is raised in the CLI's read-only half, before any submission (`_cli/CLAUDE.md`, "The run preflight, and the read-only/mutating split"), so there is no job of this generation to attach. **[call]** (review Minor 15, N2) |
| `-1` (timeout), stream error, any other code | ambiguous | Unchanged: reconcile with `cancel_inactive=True`, then raise unless durable jobs prove a submission. |

**The no-cancel flag (I5, N2).** `_reconcile_submission` gains a keyword `cancel_inactive: bool`.
- **`True`** (the timeout and stream-error arms only): an inactive fence is cancelled as today
  (`_slurm.py:652-656`).
- **`False`** (exit `4` with no job set): an inactive fence returns "no jobs" at once, **without
  reconciling its ledger or scheduler comments**. Falling through instead would reconcile
  whatever previous generation the fence names, return its unresolved tokens, raise
  `SlurmSubmitPending` (`:844-856`), and let the callback bind the GUI record to that **old**
  generation (`run_console/_callbacks.py:1048-1084`).
- **Exit `2`** never calls `_reconcile_submission`.

Today the inactive→cancel branch is reached from every caller, including the clean-exit arm
(`:837-841`); with the flag, exit `4` and exit `2` can never reach it, and neither can bind.

**The 0/3 result (I6).**
- **Type.** `SlurmFinishedWithoutSubmission` is its own frozen dataclass, **not** a subclass
  of `SlurmSubmitResult`. A subclass carrying no job id would pass every downstream
  `isinstance(SlurmSubmitResult)` check and be bound. **[call]** These sites change to accept
  the union:
  - the type check in `_complete_slurm_submission` (`run_console/_callbacks.py:1042-1047`);
  - the submitter seam's annotation (`run_console/_app.py:74`, `_callbacks.py:1310`);
  - the fake the browser tests inject through `shell/_app.py:245-246`;
  - the shell's two further seam annotations (`shell/_app.py:215, 681`; plan review I5);
  - the presentation event: `_SlurmCompletion` (`run_console/_callbacks.py:849-855`) gains
    `finished: SlurmFinishedWithoutSubmission | None`, and the banner that reads
    `completion.result` (`:2672`) shows "nothing needed SLURM; the run finished in this
    process" for it.
- **Status.** A new registry method, `observe_finished_without_submission(run_id, generation,
  returncode)`, decides it. It does **not** reuse `observe_local_exit`, which maps
  `mode == "slurm"` to `failed`, "unsupported run mode" (`_runs_registry.py:665-675`).
  - Exit `0`: `_local_completion_evidence_conflict` (`:691-770`).
  - Exit `3`: `run_finished_with_failures` (PA4).
  - Detail: "nothing needed SLURM; the run finished in this process".
- **Cancel race.** A record in `cancelling` when the 0/3 result lands becomes `cancelled`, as
  `observe_local_exit` does (`:658-660`). The cancel arrived after the CLI had already
  finished, and nothing was queued to stop. The method's expected-status set includes
  `cancelling`; the existing sets at `_callbacks.py:1094-1101` and `:1125` exclude it.

**The 60-second submitter timeout (Minor 14).** The two local recompile fallbacks
(`phenotypicCLI.py:4673-4688`, `:4700-4713`) run real work inside the submitter, which is
killed at 60 s (`run_console/_slurm.py:751`). A killed submitter takes the ambiguous path.
This predates the spec. **[call]** PR A leaves the timeout alone and records the risk (R12),
because raising it would hold the browser's acknowledgement path for every SLURM launch.

**Approaches:**

| | Approach | Pro | Con |
|---|---|---|---|
| a | Treat `4` like `0`: `returncode not in (0, 4)` at `:826` | one line | Leaves the 0/3 path wrong (stale binding at 0, failure and possible cancel at 3), and exit 4 with no job set still reaches the cancel. Exit 0 still means "submitted" in the GUI, which #281 made false |
| **b** | **Classify by `RunExitCode` (recommended)** | Matches #281 C10 exactly; removes the stale-binding bug and every cancel on a non-timeout exit | One new result type, one registry method, a reconciliation flag |

**[call]** Pick (b). In SLURM mode, exit `0` means "finished without submitting" even when an
earlier generation's job set is still on disk (§3.1).

**Tests** (`tests/unit/gui/run_console/test_slurm.py`, `tests/integration/gui/test_run_console_callbacks.py`):
- **Exit 4:**
  - with a durable job set it returns `reconciled=False`;
  - with an inactive fence and a durable job set, it does not call `cancel_generation`;
  - with an inactive fence and **no** job set (`jobs is None`), it does not call
    `cancel_generation`;
  - with no job ever recorded, it reports one error naming "SUBMITTED".
- **Exit 0 and 3**, with a previous generation's job set on disk: no binding, no cancel, and a
  terminal record (`complete` / `complete_with_failures`). One synthetic exit-3 test is
  enough, because the GUI rarely reaches it (§3.1; review Minor 9).
- A `cancelling` record meeting a 0/3 result becomes `cancelled`.
- **No stale binding (N2).** An inactive previous generation with an unresolved `intent` row,
  plus exit 2 or exit 4 with no job set: `SlurmSubmitPending` is never raised, and the GUI
  record is never bound to that generation.
- **Exit 2.** `test_submit_slurm_raises_on_non_zero_exit` keeps its message assertions.
  `test_abnormal_exit_cancels_recovered_inactive_generation` is changed to pin the opposite:
  exit 2 never calls `cancel_generation`. A new test pins the cancel on the **timeout** arm,
  which keeps it. This is a deliberate behaviour change, so the commit says so.
- **A probe that cannot pass vacuously.** Every mock-based test asserts the
  `cancel_generation` mock's call count, so a regression to (a) fails.
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

**Behaviour.** `_store_and_identity` stats `declared_measurement_table_path(store)`
(`sdk_/_table_runs.py:613-635`, exported at `sdk_/__init__.py:508, 630`). That function
returns `store / record_less_measurement_table(...).relative_path`, or `None`, so it covers:
- the highest-order run holding `measurements` for a v2 store (#281 D6a);
- the flat table for a v1 store;
- `None` when there is neither, which is treated as "no table".

It is the same function the two readers' layout logic sits on, so the identity and the read
cannot diverge. **No new sdk_ export is needed** (review Minor 6). Nothing else in the module
changes, and its docstrings stop naming the v1 path (`_measurement_source.py:8-10, 60`).

**Safety, unchanged and to be pinned.** The pixel route allows only the first resolved path
component (`_gui/_shared/tiles.py:975-1010`), and `tables` is never in the allow-list
(`results_viewer/_zarr_routes.py:24-32, 270`). So `tables/<n>-<sha12>/…` stays off the wire
exactly as the v1 path did. Add one route test for a v2 payload path returning 404/400.
Update the docstrings that name the v1 path (`_zarr_routes.py:31`, `tiles.py:986, 1001`).

**Tests.**
- Reds #9–13. Reds #8 and #13's edits use `declared_measurement_table_path`.
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
`tests/unit/gui/test_curation_does_not_unfence_the_run.py`, and it must be green. After the
rebase onto `origin/main` the 13 are re-run there, because the expected-red list is #281's.

### PA4. Exit 3 reads "complete, N images failed"

**Behaviour.** A run that the verdict below confirms gets the terminal registry status
`complete_with_failures` and a count `failed_images = N`. The verdict requires that:
- every accepted image is resolved;
- at least one failed terminally;
- the run's aggregate publication still covers the run as it now stands, or, for an
  all-failed run, none exists (D16, PA4 step 4).

N is the number of `failed` verdicts in `state.images` (the count `diagnostics.failed`
reports, which nothing branches on). It agrees with the CLI's exit code on
zero/non-zero, but it can differ in value: the CLI uses this invocation's count when positive
(`_run_failure_count`, `phenotypicCLI.py:1493-1518`). The GUI shows the **run's** count
(review Minor 4).

It is shown in four places:
- **Recent Runs:** `complete · N failed`.
- **The status banner** (`update_log_tail`, `run_console/_callbacks.py:2581-2622`): the same
  text.
- **The no-dashboard message** (`:2549-2560`): the same text.
- **The results-viewer and analysis snapshot badge (D3).** In `snapshot_refresh_status`
  (`_gui/_snapshot_status.py:53-98`) the new arm does **not** return before the currency
  check, as the unfinished arm does (`:84-91`). It falls through to `snapshot_is_current`
  (`:92-98`). A confirmed run with a current snapshot reads "Run complete · N images failed"
  (warning); one that is not current reads "Changed on disk", as a complete run does (review
  I4). This keeps the module's CAN-18 rule (`:3-9`). `_UNFINISHED_BADGE` (`:47-50`) keeps its
  two literals, so `test_every_completion_literal_has_a_badge` still binds it.

**The verdict: one sdk_ function, the failed-run twin of rule 1 (C1, I1).**
`run_finished_with_failures(state: RunState, output_dir: Path) -> int | None`, in
`sdk_/_run_state.py` beside `_run_proof_covers_current_inventory` (`:1464-1541`) and exported
from `phenotypic.sdk_` (N1). It returns N, or `None`, from **the caller's own `RunState`**
and the aggregate proof beside it. It does no `resolve_run_state` of its own, so each caller
judges the state it already resolved, at its own depth:

1. **Completion.** `completion == "failed"`: no live authority, and at least one image
   failed (`:1922-1934`).
2. **Resolution, over `state.images`.** Every accepted image's verdict is `verified` or
   `failed`; none is unresolved. It is stated over the per-image verdicts, never over
   `diagnostics`, which nothing may branch on (`sdk_/_state_types.py:138-150`; plan review I1).
   N is the number of `failed` verdicts.
3. **The proof arm** (at least one image verified). `_valid_aggregate_proof(output_dir)`, the
   GUI-tolerant check behind `aggregate_proof_is_current` (`:1397-1433`), is not `None`.
   **Never `valid_aggregate_snapshot`**, whose strict hashing a curation write defeats (§3.2).
   Then clause 2 of rule 1, against that proof, the same comparisons
   `_run_proof_covers_current_inventory` makes against the run proof:
   - `proof["inventory_digest"] == state.identity.inventory_digest`;
   - `proof["scientific_config_digest"] == state.identity.scientific_config_digest`;
   - `proof["finalization_input_digest"] in state.finalization_digests` (see "Where the
     config comes from" below);
   - `proof["source_set_digest"] == canonical_digest(sorted(verified work ids))`, and
     `proof["source_image_count"] == len(verified)`. The aggregate proof carries both
     directly (`_cli_completion.py:1258-1259`).
4. **The no-proof arm (D16)** (no image verified). An all-failed run can have no aggregate
   proof: `publish_aggregate_snapshot` refuses zero successes (`_cli_completion.py:1179-1181`).
   In place of the proof's digests the arm requires:
   - **every** accepted image's verdict is `failed`. `resolve_run_state` matches failure
     records against the **current** inventory's work ids, which bind the processing
     configuration and the input (`_cli/_cli_failure_tracker.py:276, 507`), so a changed
     configuration or input set leaves an image unresolved rather than failed, and step 2
     already refuses that;
   - `completion == "failed"` (step 1), so no liveness authority claims the run. A first run
     killed mid-image leaves that image unresolved. A killed `--retry-failures` relaunch that
     verified nothing leaves every earlier failure row in the append-only journal
     (`_run_state.py:995-996`), so the run reads confirmed again, unchanged; that is benign, and
     it stays claimable (PA11, N7);
   - **no aggregate publication marker file exists**, tested by path:
     `aggregate_publication_marker_path(output_dir).exists()` is false. Never through
     `_valid_aggregate_proof`, which returns `None` alike for an absent, wrong-version, empty or
     tampered marker (`_run_state.py:1350-1377`; plan re-review R2-I2). A marker of any kind
     beside zero verified images is not a finished all-failed run;
   - **no deliverable from another generation:** neither `master_measurements.parquet` nor
     `measurements.parquet` exists under `deliverables/`. An all-failed finalize writes no
     deliverable: locally it returns before writing ("No valid measurements found",
     `_cli_finalize_run.py:883-885`), and a SLURM finalizer logs and closes the lifecycle
     terminal-incomplete (`_cli_checkpoint_handler.py:317-331`). So either file beside zero
     verified images was left by an earlier generation: `--restart` clears `.phenotypic/`,
     journal and marker included, but keeps `deliverables/` (`phenotypicCLI.py:3493-3500`).
     Such a tree reads `failed`, not finished, and is re-run from the CLI or with `--overwrite`;
   - **a fence, when one exists, was closed by its finalizer:** if `slurm_lifecycle_path(output_dir)`
     exists, it is inactive, `closed_by == "finalizer"`, and has no `terminal_status`. A
     cancelled all-failed SLURM run therefore reads `failed` here as it does in the observer,
     instead of flipping to `complete_with_failures` at the next GUI restart (plan re-review
     R2-M4). `FENCE_CLOSED_BY_FINALIZER` moves to `sdk_/_io_constants.py`, and
     `_cli/_cli_slurm_lifecycle.py:47` re-exports it, so sdk_ need not import `_cli`.

   No finalization input is compared: nothing was published, so nothing can be stale. N is
   the number of accepted images.
5. **Not a process run.** A run whose config records `process_only_layer` returns `None` (N4).
   No caller needs a process arm: the waiter keeps its own rule 4 (`_cli_wait.py:356-364`),
   the GUI has no process mode, and the viewer does not open process trees. A process tree
   with failures therefore stays unclaimable (PA11), which is right, because the CLI would
   refuse a full-mode continuation of it as incompatible.

**Where the config comes from (plan review I1).** `RunState` carries no config
(`sdk_/_state_types.py:157-166`), yet step 3's finalization comparison and step 5 need it: the
aggregate proof writes the unversioned finalization digest (`_cli_completion.py:1249-1256`),
while `identity.finalization_input_digest` is the versioned one (`_run_state.py:183-196,
325-327`). `RunState` therefore gains two fields, filled from `resolve_run_state`'s own single
config read (`_read_state_config`, `:134`):
- `finalization_digests: frozenset[str] = frozenset()`, the value of
  `accepted_finalization_digests(config)` (`:1436`);
- `process_only_layer: str | None = None`.

The defaults keep every existing `RunState(...)` construction valid. Two pinned surfaces change
deliberately (plan re-review R2-I1): `test_the_demoted_sources_live_only_under_diagnostics`
asserts `RunState`'s exact field set (`tests/unit/sdk_/test_run_state.py:56-75`), and
`docs/source/contrib_guide/tracked_state.md:43-53` tables the fields. Both gain the two names,
with a docstring line saying they are configuration the resolver already read, not demoted
evidence. **[call]** This is
chosen over re-reading `processing_state.json` inside the verdict and comparing identity
digests: with one read the verdict can never describe a different instant from the caller's
state, which is the point of N1. The verdict's one read is the aggregate proof itself, whose
fields are self-consistent.

**The shared helpers.** Rule 1 compares the inventory and scientific digests before its process
branch (`:1508-1520`), and takes its source set from `_source_set_binding(output_dir, proof)`
(`:1528`), which for a pre-P4 run proof is the aggregate's, not `proof`. So the comparisons are
two private helpers, `_identity_digests_match(proof, identity)` (inventory, scientific) and
`_source_set_matches(binding, verified)` (source set and count, given a binding mapping). Both
arms call both, so they cannot drift. **[call]**

**Who calls it, and with which state (N1):**
- **GUI, generation-less** (these import from `phenotypic.sdk_` only, per the readers rule):
  - `observe_local_exit`, for a local exit `3`, and `observe_finished_without_submission`
    (PA1): one shallow `resolve_run_state`, then the verdict over it;
  - `_rehydrated_status` (`shell/_runs_registry.py:1026-1104`): the shallow state it already
    resolves;
  - the snapshot badge: the shallow state `_run_completion` resolves
    (`_snapshot_status.py:101-121`), returned whole instead of only its completion;
  - the claim gate (PA11): the shallow state `_foreign_run_conflict` resolves under the owner
    lock (`:1436-1446`, `:1460`);
  - the write-authority checks (PA10): each passes the state it judges. The mutation guard
    passes its **fresh** and its **verified deep** states (`_mutation_guard.py:136-177`), so
    the verdict and the identity digest it is compared beside describe the same instant.
- **The waiter.** `_run_snapshot` (`_cli_wait.py:116-130`) computes the verdict in its single
  pass, and `RunSnapshot` gains `finished_with_failures: int | None` and
  `excluded_stores: tuple[str, ...]` (R2-I4), so "the verdict and the
  counts describe the same instant" still holds. `decide_wait_outcome` (`:309-366`) takes it as
  a **value** and stays a pure function (`tests/unit/cli/test_cli_wait.py:3-8`). Rule 3 becomes
  "`published` for this generation, and `finished_with_failures` is not `None`". Rule 4
  (process) keeps its own arm. **[call]** This tightens the waiter (D14):
  - before, a publication by this generation that no longer covers the run returned `3`;
  - now it returns `5`, `ENDED_UNPUBLISHED`.
  - **The realistic trigger (N3)** is a finalize that **excluded an unprojectable store**
    (`project_embedded_measurement_table`, `_cli/_cli_parquet_agg.py:277`; the root
    `CLAUDE.md`, "Output layout"). Its proof's `source_set_digest` omits a verified image, so
    such a run with failures moves from `3` to `5` deterministically. A metadata change after
    finalize almost never falls inside the waiter's window, because the live-run guard
    refuses a second invocation while the run is live.
  - **The exit-5 line gains a variant.** `_unpublished_line` (`_cli_wait.py:404-424`) prints
    "this submission published nothing, so its finalizer failed or never ran". The variant is
    printed **only when this generation published an aggregate proof** (its marker's
    `execution_epoch` is the generation; with no generation, a marker exists) and the verdict
    is `None`. It is never printed on `generation is None` alone, and never for an all-failed
    run, which D16 keeps at `3` (plan review C1).
    - **When a store is demonstrably unprojectable (D15).** The names come from the per-image
      stage facts `_advisories` already uses, never from a set difference: a verified image
      whose record authorizes an embedded table (`declares_measurements is True`) but whose
      store declares no column list to project it onto (`projectable_measurements is False`)
      (`_run_state.py:965-975, 1680-1695`). That predicate becomes one exported sdk_ helper,
      `excluded_store_names(images: Mapping[str, ImageState]) -> tuple[str, ...]`, which
      `_advisories` and the waiter both call. The waiter computes it in `_run_snapshot`'s one
      pass and carries it on `RunSnapshot` (plan re-review R2-I4). The line: "phenotypic-wait:
      the published outputs leave out <list>, because their stores declare no table the
      finalizer can project (its log under <logs> says why). Fix those stores by re-measuring
      them, or remove them from the input, then re-run the same command." **[call]**
    - **Otherwise, the gap wording, which names no cause:** "phenotypic-wait: this submission
      published, but its published outputs no longer cover the run: images verified after it
      published, or the metadata, pipeline or input set changed. Re-run the same command."
      A rolling input reaches this with nothing wrong, so no store is named
      (`test_a_master_that_merely_predates_an_image_claims_no_exclusion`,
      `tests/unit/sdk_/test_run_state.py:1351`, stays green).
    - **No marker field is added** (plan re-review R2-C1): a field derived as "verified minus
      `source_work_ids`" measures that gap, not an exclusion, and would name valid stores
      (`_cli_completion.py:1132-1150`, `_cli_finalize_fanout.py:700-706`).
    - `--accept-failures` still maps only a final `3` to `0` (D15).

  It changes what #281's exit `3` promises in one edge case; D14 adopts it.
- **The SLURM observer, generation-bound.** A thin wrapper,
  `finished_run_outcome(output_dir, generation) -> RunExitCode | None`, in `_cli/_cli_wait.py`.
  It is `decide_wait_outcome` over one `_run_snapshot` and the other existing seams, with the
  live-jobs check left to the caller. Only the SLURM observer calls it; it already imports lifecycle readers from `_cli`
  (`run_console/_slurm_observer.py:27-29`). At the "inactive, no marker" branch (`:661-671`),
  the outcome maps as follows:
  - `COMPLETE_WITH_FAILURES` gives `complete_with_failures` with N;
  - `COMPLETE` gives `complete`;
  - `ENDED_UNPUBLISHED` gives `failed`, with PB5's measure detail where it applies;
  - `None` keeps observing.
- **The Known consumers table.** Every new reader above gets a row in
  `docs/source/contrib_guide/tracked_state.md` (`:433-450`), in the same change (`:18, 525`).

**Status representation:**

| | Approach | Pro | Con |
|---|---|---|---|
| **a** | **New `RunStatus` literal `complete_with_failures`, plus a `failed_images: int \| None` field on `RunRecord` (recommended)** | Honest and terminal. Gating code that tests `== "complete"` keeps its meaning | Touches the status sets (below) and the persisted record |
| b | Keep `complete`, carry N in `status_detail` | Smallest change | Recent Runs shows a green "complete" over failed images, and the observer reads it as a clean completion |
| c | Keep `failed`, improve the detail | No schema change | Not what the user asked for (G2) |

**Edits (a) requires:**
- **The status sets:**
  - the `RunStatus` literal and `_RUN_STATUSES` (`shell/_runs_registry.py:103-130`);
  - `_TERMINAL_STATUSES` (`:133-135`);
  - `run_console/_callbacks.py:862`;
  - `run_console/_slurm_observer.py:62`.

  The existing guard `test_every_status_vocabulary_in_the_gui_agrees_with_the_registry`
  (`tests/unit/gui/shell/test_runs_registry.py:1268`) already catches a vocabulary that is
  missed, so no new grep gate is needed (review Minor 2).
- **The plumbing for N** (plan review I2, I3):
  - `_Observation` (`run_console/_slurm_observer.py:327-335`) gains `failed_images: int | None = None`, and `_apply` (`:861-896`) passes it;
  - `RunRegistry.compare_and_set` (`shell/_runs_registry.py:502-518`) gains a `failed_images` keyword, and resets it to `None` for any status other than `complete_with_failures`, so a later `complete` cannot carry a stale N;
  - `_rehydrated_status` (`:1026-1104`) returns N beside `(status, detail)`, and both its consumers persist it: `rehydrate_from_sandbox` (`:829`) and `_release_dead_owner_locked` (`:1357-1397`).
- **The persisted record.** `failed_images` is serialised next to `returncode`
  (`_runs_registry.py:1244-1245, 1501`) and read tolerantly, so an older record has `None`.
- **Older readers** (review Minor 3). An older GUI's `_read_owner_record` returns `None` for
  an unknown status (`_runs_registry.py:1209-1210`). Its claim gate then raises "output has an
  invalid generation owner" (`:1283-1287`). That fails closed: an older GUI cannot relaunch
  into an output a newer GUI marked `complete_with_failures`. sdk_ reads owner status only
  through `_OWNER_STATUSES_IN_FLIGHT` (`sdk_/_run_state.py:1137-1138`), which the new status is
  not in.
- **The CSS.** Add `.run-console-recents-status-complete_with_failures` in `--oi-orange-text`.

**[call]** Use `--oi-orange-text`, the colour `incomplete` and `cancelled` share
(`run_console/_assets/run_console.css:338-350`): finished, but not everything you asked for.
Not `--oi-green-text`, because that is a clean success. Not vermilion, because DESIGN.md
reserves vermilion for error and alert.

**Tests:**
- **`tests/unit/sdk_/`, for `run_finished_with_failures`:**
  - a finished run with failures returns N;
  - an unresolved image returns `None`;
  - no aggregate proof returns `None`;
  - **a `--retry-failures` continuation killed after its last image and before finalize**
    returns `None`. More images now verify, so `source_set_digest` and `source_image_count`
    no longer match (C1);
  - **a re-run with a changed `--metadata`, killed before finalize**, returns `None`. The
    `finalization_input_digest` no longer matches (C1);
  - an edited pipeline (`scientific_config_digest`) and a new input image
    (`inventory_digest`) each return `None`;
  - a curation write leaves the result unchanged (PA10);
  - the shared helper is used by both arms. A mutation of one comparison fails tests of both.
- **`tests/unit/gui/shell/test_runs_registry.py`:**
  - exit 3 with confirming evidence;
  - exit 3 with an unresolved image, which stays `failed`;
  - exit 3 with no aggregate proof, which stays `failed`;
  - exit 1, unchanged;
  - an old record with no `failed_images`.
- **`test_slurm_observer.py`:**
  - a finalizer-closed fence with failures gives `complete_with_failures`;
  - a cancelled fence stays `failed`;
  - a staged run closed with phase `"terminal_incomplete"` reaches the hook, and is not
    returned `failed` by `_staged_terminal_observation` (review Minor 16).
- The rehydration test.
- **`tests/unit/gui/test_snapshot_status.py`:**
  - the new badge arm on a current snapshot;
  - "Changed on disk" for a re-finalized master, using the pattern of
    `test_a_rewritten_processing_product_shows_changed_on_disk`;
  - a curated `complete_with_failures` run still reads "Run complete · N images failed".
- **The waiter tests (`tests/unit/cli/`):**
  - pass through the composed rule 3;
  - a new test for the tightened case (D14): a finalize that excluded an unprojectable store
    returns `ENDED_UNPUBLISHED`, and prints the exit-5 variant line.

### PA5. TSV exports and mirror readers

**Writers:**

| Writer | Today | PR A |
|---|---|---|
| Curated mirror, `_write_curated_mirror` (`results_viewer/_curation_labels.py:839-850`) | `.parquet`, then a best-effort `.csv` (`_atomic_write_csv`, `:961-966`) | `.parquet`, then `measurements.tsv` via `sdk_.write_tsv` (`sdk_/_delimited.py:102-148`), then retire a stale `measurements.csv` (PA8). `_atomic_write_csv` is deleted |
| `FilteredMeasurements._save_locked` (`results_viewer/_filtered_state.py:543-600`, CSV at `:575-600`) | `.parquet`, then `.csv` | the CSV write is replaced by a TSV write in place, and the `measurements_csv_path` import (`:44`) goes. Nothing else changes |
| Error analysis, `_stage_generation` (`results_viewer/_error_tab/_publication.py:576-600`); the CLI re-emit shares it (`_cli/_cli_error_outputs.py:67-95`) | `error_analysis.csv` (`computation_path_name`, `:849-851`) | **unchanged** (D18): `error_analysis.csv` stays, deferred to §8 |

**[call] On `FilteredMeasurements` (review Minor 8).** `_gui/CLAUDE.md:629-637` calls it a
"utility / constants module only … do not extend it", and only tests construct it
(`tests/gui/results_viewer/test_filtered_state.py:64, 352`). The review suggested leaving its
CSV write, or deleting the method. I do neither:
- **Leaving it** would keep a writer of `measurements.csv` in the codebase, against D1.
- **Deleting it** would remove a tested save path, which is out of scope.

A one-writer swap does not extend the class.

**[call] When the TSV write fails (review I7).** The Parquet stays authoritative; it is what
every GUI reader reads. On a failed curated TSV write:
- unlink `measurements.tsv` **and** a pre-PA8 `measurements.csv`, then log;
- both text files predate this curation, so a text reader that fell back to the CSV would
  read exactly the stale rows the unlink exists to hide;
- a text reader then finds nothing.

On a successful TSV write, a stale CSV is retired after it (the `remove_stale_csv_sibling`
contract, `sdk_/_delimited.py:151-185`).

`measurements_tsv` is already a GUI-written proof descriptor (`sdk_/_run_state.py:1331-1333`),
so curation does not unfence the run (PA10).

**Fingerprints must include the TSV:**
- **`_source_fingerprint`** (`_curation_labels.py:761-771`) swaps `layout.mirror_csv` for
  `layout.mirror_tsv`.
- **`_consumed_state_snapshot_paths`** (`results_viewer/_output_root.py:1119-1133`)
  fingerprints **both** `mirror_tsv` and `mirror_csv` during the transition (review Minor 11).
  A CLI retirement of the CSV during a discovery then forces a retry rather than a torn
  binding.

**Error analysis stays CSV (D18).** Its publication has a strict versioned contract (name sets
at `_publication.py:490-495, 638-644, 662-676, 710-714`, `schema_version == 1` at `:369, 471,
717`), and a version bump also has to move `_generation_digest`'s pinned version (`:799-827`),
or an unchanged v1 tree returns `already_published` early (`:331-338`; plan review I6). The
Error tab is unmounted, so the change is deferred to the follow-up issue (§8).
`"error_analysis"` stays in `_KEPT_CSV_STEMS` (PA8).

**Readers:**
- **`_schema_cache.py:46`**, the only GUI reader of the text mirror. It is the no-pyarrow
  fallback; everything else reads Parquet. It resolves through
  `BundleLayout.measurements_table_path()` (`sdk_/_io_constants.py:3379-3388`): `.tsv`, then
  `.csv`. It reads with `read_delimited`, so the delimiter follows the suffix.
- **The re-export.** `_gui/_config.py:68, 136, 462-466` adds `MEASUREMENTS_TSV`.
- **The GUI import-walk test** (review Minor 13). `test_the_walk_finds_the_gui_imports_the_spec_names`
  (`tests/unit/sdk_/test_gui_sdk_imports.py:89-92`) pins that the GUI *imports*
  `MEASUREMENTS_CSV` and `measurements_csv_path`. After PA5:
  - `measurements_csv_path`'s only GUI import (`_filtered_state.py:44`) is gone;
  - `MEASUREMENTS_CSV` remains only as `_config.py`'s re-export.

  So the test is edited to assert `MEASUREMENTS_TSV` only (N6). `_schema_cache` resolves
  through the `BundleLayout.measurements_table_path()` *method*, and the walk records only
  imported names (`:54-86`), so the free function `measurements_table_path` would never appear
  in it. Its module docstring anticipates the edit (`:4, 15`). `gui_sdk_signatures.json` is regenerated the same way. The sdk_ *exports*
  stay defined as legacy read names (PA8). That is a separate fact; the walk does not pin it.

**Tests:**
- curation rewrites `measurements.tsv`, and `read_delimited(tsv)` reads back the same frame
  as the Parquet (review Minor 17, a read-back rather than a bytes check);
- curation on a pre-PA8 tree retires its `measurements.csv`;
- a failed TSV write leaves neither a `.tsv` nor a `.csv`, and keeps the `.parquet`;
- `_schema_cache` prefers the TSV and falls back to a CSV-only tree;
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

**The tune app is unmounted** (`_gui/CLAUDE.md:20-26`; review Minor 7). PR A changes tune code
only where:
- a red test needs it: the Deploy validator, the spec loader, and the preset writer that
  reds #2 and #3 drive through `write_authored_setup_spec`;
- the user named the site: D8's pickers (PA9).

The other tune surfaces move to §8, to be fixed when `/tune/` is re-mounted.

| Surface | Today (file:line) | PR A |
|---|---|---|
| Builder directory browser | `PIPELINE_EXTS = PIPELINE_CONFIG_SUFFIXES` (`builder/_directory_browser.py:32`), matched at `:259` | `directory_tree` (`:151-158`) accepts a predicate as well as `extensions`; the pipeline modal passes `is_pipeline_config_path`, threaded through the wrappers `_render_tree_body` (`builder/_callbacks.py:4019`) and `_initial_tree_body` (`builder/_modal_browser.py:189`) |
| Run console pipeline picker (plan review I4) | `render_pipeline_tree` (`run_console/_form.py:58-73`) calls `directory_tree(..., extensions=PIPELINE_EXTS)` (`:67-73`), so a `.pht-tune.json` is selectable as the Run pipeline | passes `predicate=is_pipeline_config_path` |
| Shell classifier | `_classify_file` (`shell/_classifier.py:220-235`). The content peek (`:348-360`) matches `"pipe_cfgs"`, which a tune spec also contains, because it embeds its pipeline | suffix test is `is_pipeline_config_path`, so no `has_pipeline_json` for a tune spec |
| Run console pipeline check | `run_console/_callbacks.py:2897-2899` | `is_pipeline_config_path` |
| Tune Deploy validator (red #1, #4–6) | `CONFIG_SUFFIX_TUNING` only (`tune/_validation.py:61`); message names `.json.pht-tune` (`:65`) | accept the canonical and legacy typed tune suffixes; message names `.pht-tune.json` |
| Tune spec loader (reds #2–3) | `load_pipeline_or_spec` (`tune/_setup_authoring.py:353-358`) treats only the canonical suffix as a spec | canonical **and** legacy typed tune suffixes load as `TuningSpec` |
| GUI tune preset writer (reds #2–3) | writes `…setup.json.pht-tune` (`tune/_setup_authoring.py:535`), a legacy name | `…setup.pht-tune.json` |
| Analysis `RecipeState.save` | writes canonical (`analysis/_recipe_state.py:1159`), and an old `pipeline.json.pht-pipe` survives | retire the old typed sibling after publishing, as `QcRecipe._retire_old_typed_sibling` does (`sdk_/_qc_recipe/_recipe.py:546-560`); a plain `pipeline.json` is never removed (#281 A4) |
| Capture script | writes `pipeline.json.pht-pipe` (`scripts/capture_gui_tutorial_screenshots.py:52, 229, 1112`) | canonical name, so the screenshots show the name users will see |
| Docs | `_gui/CLAUDE.md:257` (`PIPELINE_JSON # "pipeline.json.pht-pipe"`, already stale); `FEATURES.md` (19 mentions of `.json.pht-*`, e.g. `:123, 433, 604-622`); 10 in `docs/source/tutorials/gui/` | canonical names; legacy names described as read-only |
| E2E fixtures | `tests/e2e/gui/` writes `.json.pht-*` (#281 hand-off P0) | canonical, plus one legacy-name read test per mounted surface family |

**Tests:**
- a unit test per row;
- a parametrised test of `is_pipeline_config_path` over every canonical, legacy and plain name;
- a classifier test showing a `.pht-tune.json` holding `pipe_cfgs` is not a pipeline;
- the `RecipeState` retire step: the old twin is removed, a plain `.json` is kept, and a
  publish failure leaves the old file.

### PA7. Ledgers, tutorial text and screenshots (the `gui-tutorial-capture` skill)

- **`FEATURES.md`** (the CI gate requires it whenever `_gui/` changes):
  - new rows:
    - the "complete with failures" status;
    - the SLURM no-op exit 0/3;
    - the curated TSV mirror;
    - the `is_pipeline_config_path` listing;
    - curation of a run with failures (PA10);
    - relaunch of a run with failures (PA11);
    - TSV metadata pickers (PA9);
  - rows to update:
    - `:356` "Curation on-disk format": `measurements.{tsv,parquet}`, no CSV;
    - `:371` "Colony measurement picker": the latest measuring run, not
      `tables.measurements`;
    - `:53, 54, 92, 114, 451, 456`, the metadata picker, handoff and reference rows, which
      name `.csv` or `deliverables/metadata.csv`;
    - `:537` and `:585-586`, the claim and status rows;
    - `:626`, which names `measurements.{parquet,csv}`;
    - every `.json.pht-*` mention on a mounted surface.
  - Each `✅ shipping` row's `Test ref` must resolve (pre-commit and `check_features_md.py`).
- **`WORKFLOWS.md`.** No new workflow. `run_local`, `run_slurm`, `view_results` and `analysis`
  descriptions change only if their text names a changed state or file.
- **Tutorial text (D4, D11).** The new status is **documented in text, with no new frame**:
  - **`docs/source/tutorials/gui/04_run_local.md` and `05_run_slurm.md`:**
    - one paragraph on `complete · N failed`: how it differs from `failed`, that the failed
      images have no rows, and that the output can be relaunched with `--retry-failures`
      (PA11). If that relaunch is killed before it finalizes having verified at least one
      image, the GUI refuses the output, and it is resumed by re-running the same command from
      the CLI (N7); one that verified nothing leaves the output as it was, still relaunchable. Never re-run the same
      output from the CLI and the GUI at once: nothing records a local CLI run's liveness, so
      the GUI cannot see it (D17);
  - **`06_view_results.md`:**
    - one paragraph on the badge and on curating such a run (PA10);
    - **curate after the last re-run (D11).** A CLI re-run or `dvc repro` of a run with
      failures re-finalizes and reseeds the mirror without curation. Your labels survive in
      `deliverables/qc/curation_labels.parquet`, and the viewer re-applies them at its next
      curation write. A complete run's re-run is a no-op and keeps the curated mirror;
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
| `_cli_output_manager.py:968-1003` `_seed_measurements` | writes `.tsv`, `.csv`, `.parquet`, each through `_guarded_terminal_best_effort` (`:99-115`), which swallows errors | writes `.tsv` and `.parquet`. The old CSV is retired by `remove_stale_csv_sibling(measurements_tsv_path(output_dir))` only if both hold: (1) the TSV write **succeeded**, a fact the plan must carry out of the best-effort wrapper; (2) the unlink runs **under `commit_guard` when one is given**; with `commit_guard=None` (which `_guarded_terminal_call`, `:90-97`, allows) it runs unguarded, still after the TSV, like the sweeps at `:838` and `:1502` (review Minor 10; plan re-review R2-M5). The docstring (`:975-983`) stops citing B9 |
| `_cli_completion.py:1205-1210` `required_paths` | records `measurements_csv` | drops `measurements_csv`. The proof records `master_parquet`, `measurements_tsv` and `measurements_parquet`, plus `metrics_json` and `schema_json` (`:1215-1220`). The comment at `:1183-1203` is updated |
| `_cli_interactive.py:257, 274-279` dry-run inventory | names `measurements.csv` "for the results viewer" | names only the TSV and Parquet |
| `_cli_readme_generator.py:116, 147-150` README tree and paragraph | lists `measurements.csv` "kept until #283", and names `error_analysis.csv` (`:147`) | removed; `error_analysis.csv` stays (D18); one sentence says older runs had a `measurements.csv` |
| `_dashboard/_generator.py:826, 1017-1019` download command | `wget …/deliverables/measurements.csv` | `…/deliverables/measurements.tsv`; the element id `cmd-csv` is renamed with its CSS/JS |
| `_cli_preflight.py:182-185` hint text | "dropped from measurements.csv" | "dropped from measurements.tsv" |

**`sdk_/` changes:**

| Site | Today | PR A |
|---|---|---|
| `_delimited.py:146-148` `_KEPT_CSV_STEMS` | `{"measurements", "error_analysis", "metadata"}` | `{"error_analysis", "metadata"}`: only `measurements` leaves it. `error_analysis.csv` stays CSV (D18), and the user's snapshot is never swept (the root `CLAUDE.md` rule); the docstring at `:155-162` follows |
| `_run_state.py:1331-1333` `_GUI_WRITTEN_PROOF_DESCRIPTORS` | `{measurements_csv, measurements_tsv, measurements_parquet}` | **unchanged**. It is reader-side: a proof written before PA8 still records `measurements_csv`, and a curation or a PA8 retirement must not unfence it |
| `_io_constants.py:415, 1698-1700, 3371-3377` `MEASUREMENTS_CSV`, `measurements_csv_path`, `BundleLayout.mirror_csv` | the curation write target | **kept as legacy read names**, with "read-only legacy; no writer" docstrings |
| `_io_constants.py:1708-1730, 3379-3388` `measurements_table_path` | "unwired until #283" | wired (PA5); docstrings drop the caveat |
| `_io_constants.py:403-430` constant comments | describe the CSV as the human-opened file | describe the TSV |

**An existing run's CSV on its next finalize.** It is retired after the TSV is published:
publish first, then unlink. This holds for a forward continuation, `--mode measure`,
`--mode recompile` and a full-run `--mode migrate`, all of which reach `_seed_measurements`
through the shared finalize (root `CLAUDE.md`, "Output layout").
- Every plain re-run of a run with failures re-finalizes (§3.2), so such trees lose their CSV
  at their next re-run.
- Curation on a pre-PA8 tree retires it the same way (PA5).
- A tree nobody finalizes again keeps its CSV, and every reader falls back to it.

**Byte stability (#281 C11).** Removing a file from `deliverables/` is a one-time change in a
DVC stage's out hash, after which runs are byte-stable again.
`tests/integration/cli/test_deliverables_byte_stable.py` keeps comparing, with the CSV gone
from both runs.

**Docs that name the file:**
- **The DVC how-to**, `docs/source/how_to/pages/dvc.md:91-94` and `:363-364`:
  - the "until #283" caveat is removed, because curated edits now reach the TSV;
  - a new paragraph says to **curate after the last `dvc repro` (D11)**: a re-run of a stage
    whose run has failed images re-finalizes and reseeds the mirror without curation, while
    the labels survive and the GUI re-applies them at its next write.

  The template's `plots:` entry already names `measurements.tsv` (`:76`), and its `outs` are
  directories, so no DVC `outs` entry names the CSV.
- **The root `CLAUDE.md`**, "Output layout": `measurements.{csv,parquet}` becomes
  `measurements.{tsv,parquet}`.
- **`src/phenotypic/_cli/CLAUDE.md`:**
  - the bullet "The GUI run console reads `4` as a failed submission" (`:777`) is removed or
    restated as fixed;
  - the paragraph "Two generated CSVs and one input snapshot stay CSV" (`:1659`) becomes
    "one input snapshot stays CSV";
  - its output inventory follows.
- **`src/phenotypic/_gui/CLAUDE.md`:**
  - the state-table row "Measurements mirror … `_curation_labels.py:846,848`" (`:324`) and
    the paragraph at `:427` name the TSV;
  - the `PIPELINE_JSON` comment (`:257`) is fixed (PA6);
  - the readers rule gains `run_finished_with_failures` as the sdk_ answer to "did this run
    finish with failures?".
- **`docs/source/contrib_guide/tracked_state.md`**: the Known consumers rows (PA4).
- **Every description of exit `5` (D14, N3)**, which now also covers "published, but the
  published outputs no longer cover the run":
  - `RunExitCode.ENDED_UNPUBLISHED`'s docstring, "published nothing" (`sdk_/_exit_codes.py:29-30`);
  - `_cli/CLAUDE.md`'s exit-code table (`:663`) and its `phenotypic-wait` bullets (`:697-745`,
    `:1549`);
  - the DVC how-to's exit-code table and its text about `3` (`docs/source/how_to/pages/dvc.md:370-392`).
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

**The shared helper** `tests/_output_layout.py:82-105` `write_measurements_mirror(csv=True)`
is **called by 28 test files**: `grep -rl write_measurements_mirror tests`, excluding the
helper itself. The review counted 29. **62 files import `tests._output_layout`** (review
Minor 19). The default flips to `csv=False`, and the keyword stays for old-tree tests.

Each assertion is reviewed one by one:
- an assertion that the CSV **exists** after a finalize becomes one that it **does not**, and
  that the TSV does;
- an assertion about an **old tree** keeps writing a CSV and checks the fallback or the
  retirement;
- `tests/fixtures/tables_v1/v1_run/` is a legacy fixture and stays byte-for-byte.

**New tests:**
- a finalize writes no `measurements.csv`;
- a finalize over a tree holding one retires it only after the TSV exists, and only inside
  the commit guard;
- a TSV publish failure keeps the CSV;
- `remove_stale_csv_sibling` now sweeps `measurements`, and still refuses `error_analysis`
  and `metadata`;
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
| `run_console/_callbacks.py:305-327` `reference_metadata_requirement` (review I8) | the requirement is met only when `metadata_csv_deliverable_path(output_dir)` is a file (`:324-326`), so a run snapshotted to `metadata.tsv` (#281 B5) shows "reference metadata required" and disables Run | resolve with `resolve_metadata_snapshot_path` (`sdk_/_io_constants.py:1764`). When the output holds both suffixes, the resolver's `ValueError` becomes the alert's text, with Run disabled |
| `run_console/_request_safety.py:403` metadata preflight | `read_metadata_csv` already accepts `.tsv`; a refused suffix surfaces as a bare `ValueError` | the blocked message names the allowed suffixes, as the CLI's does ("must be a .csv or .tsv table") |
| `analysis/_callbacks.py:95` `MergeMetadata` default `"metadata.csv"` | names a file a `.tsv` run does not have | resolved with `resolve_metadata_snapshot_path`. A tree holding both snapshot suffixes shows the resolver's message, not a traceback |
| `tune/_setup_authoring.py:46, 372, 451, 583` QC layout path (unmounted, D8) | `{".csv", ".parquet"}` | adds `.tsv` |
| `tune/_layout.py:330` and `tune/_callbacks.py:1254` tune metadata pickers (unmounted, D8) | `{".csv", ".parquet"}` | adds `.tsv` |

**The QC layout reader.** `ExpectedVsDetectedCount._load_layout` reads `.csv` and `.parquet`
only (`analysis/qc/_expected_vs_detected.py:612-620`). PA9 extends it to `.tsv` through
`read_delimited`, and its field descriptions (`:79, 110`) follow.

**[call]** The review notes the tune GUI is unmounted (Minor 7), so this change is justified
on **CLI parity**, not on the pickers. `ExpectedVsDetectedCount` is a `QualityCheck`
(`:256`) that the `phenotypic-tune` CLI's QC scorer wraps (`tune/score/_qc_scorer.py:20, 52`).
Its layout path is a user-supplied table, which #281 B4 says accepts `.tsv` everywhere. D8's
tune pickers stay in, because the user named them.

**Tests:**
- each mounted picker lists a `.tsv`;
- `read_metadata_csv_table` reads a TSV into the same columns and rows as the equivalent CSV;
- a `.txt` is still refused, with the message naming both suffixes;
- the reference table and the QC scorer load a `.tsv`;
- `reference_metadata_requirement` is met by `metadata.tsv`, and shows the resolver's message
  when both snapshots exist;
- `MergeMetadata`'s default resolves to `metadata.tsv` in a TSV run.

### PA10. Curation of a `complete_with_failures` run (D2, D11)

**Behaviour.** A run that `run_finished_with_failures` (PA4) confirms is writable exactly as
a `complete` run is.
- **No rows for the failed images.** Aggregation reads only authorized embedded tables, so
  the failed images contribute no rows to the master and offer nothing to curate.
- **Their metadata-only phantom rows** carry a null `Object_Label`, which is already excluded
  from curation keys (`FEATURES.md:356`).
- **The currency check comes with it.** A run whose publication no longer covers it (C1) is
  not writable.

**The rule.** One predicate, `run_permits_mutation(state, output_dir)`, is true for:
- a bundle (`state is None`);
- `completion == "complete"`;
- a run that `run_finished_with_failures` confirms.

It replaces every `completion == "complete"` write-authority check:
- `_mutation_guard._run_is_complete` (`_mutation_guard.py:38-45`) and its three uses inside
  `authorize`: at bind (`:121`, through `OutputRoot.run_is_complete`), on the fresh state
  (`:140`), and on the verified state (`:170`);
- `output_mutations_disabled` and `output_read_only_diagnostic` (`:214-229`);
- `OutputRoot.run_is_complete` (`_output_root.py:700-713`), renamed `run_permits_mutation`.
  Its CAN-17 docstring now names the second allowed state;
- `mutation_snapshot_is_safe` (`_output_root.py:780-797`);
- the processing-inventory reuse rule (`_processing_inventory.py:97`). A confirmed run is
  terminal, so it gets an `"exhaustive"` inventory (`:138-145`), and the guard's
  exhaustiveness check (`_mutation_guard.py:127-131`) passes for it as it does for a complete
  run.

**What curation must check** before each write. All of these come from `authorize`:
1. **The verdict, fresh, at all three points.** At bind, on the fresh deep state, and on the
   verified deep state, `run_permits_mutation` must hold. Any change refuses the write: a
   retry that repaired an image, a new failure, or a stale publication (C1).
2. **The run identity digest** is unchanged across the three (`:146-150, 169-177`).
3. **No active owner** (`:151-154, 172`) and a current processing snapshot (`:155-159`).
4. **The failed count N is part of the verdict.** A write authorized against "N failed" is
   refused if the fresh verdict says otherwise. **[call]** The receipt records N beside the
   identity digest, so a test can assert it.

**Fencing after a curation write on such a run.** Curation rewrites `measurements.parquet`
and `measurements.tsv`, retires a stale `measurements.csv` (PA5), and writes the
`deliverables/qc/` and `errors/` state. Every descriptor it touches is in
`_GUI_WRITTEN_PROOF_DESCRIPTORS`, and none of the comparisons in PA4 step 3 reads a mirror, so:
- `aggregate_proof_is_current` stays `True`;
- `run_finished_with_failures` still returns the same N;
- `resolve_run_state` keeps completion `failed`, with the same identity digest and
  diagnostics;
- the enforced descriptors (`master_parquet`, `metrics_json`, `schema_json`) still match their
  bytes;
- the run has no run proof before or after. None is written for a run with failures
  (`_cli_completion.py:1304-1342`), so curation cannot create or break one.

`valid_aggregate_snapshot` returns `None` after the write (§3.2), as it already does for a
curated complete run. The verdict never asks it, so this does not move the verdict (R4).

**A CLI re-run afterwards (D11, review C4).** A plain re-run, or a `dvc repro`, of the same
command re-finalizes such a run (§3.2):
- it reseeds the mirror from the post-applied master, without curation;
- it re-emits `errors/` and `error_analysis.*` from `curation_labels.parquet`
  (`_cli_error_outputs.py:56-65`);
- it publishes a new aggregate proof.

The labels file survives, and the GUI re-applies the labels at its next curation write. There
is no CLI change (D11). The tutorial and the DVC how-to say: curate after the last re-run
(PA7, PA8).

**Tests** (`tests/unit/gui/results_viewer/`, `tests/unit/gui/test_curation_does_not_unfence_the_run.py`):
- the guard authorizes a confirmed run;
- it still refuses:
  - a `failed` run with an unresolved image;
  - a run whose aggregate proof no longer covers it (the C1 cases);
- a verdict that flips between bind and write (a retry repairs the failed image) is refused;
- **after a curation write on such a run:**
  - `aggregate_proof_is_current` is `True`;
  - `run_finished_with_failures` returns the same N;
  - `resolve_run_state(...).identity.digest()` and `.diagnostics` are unchanged;
  - `master_parquet`, `metrics_json` and `schema_json` still match;
- **a plain CLI re-run after that curation (rewritten for D11):**
  - exits `3` locally;
  - `curation_labels.parquet` and `errors/` survive;
  - `measurements.parquet` and `measurements.tsv` are reseeded uncurated: the curated-removed
    rows are back;
  - after the viewer refreshes, the next GUI curation write re-applies the surviving labels,
    so the removed rows are gone again. The refresh is required: the re-finalize changed the
    session's fingerprint, so `_publish_if_current` refuses the write until the viewer reloads
    (`_curation_labels.py:773-800`; re-review C4 note);
- the Results and Analysis read-only diagnostic no longer appears for such a run.

### PA11. Relaunching a run that finished with failures (D10, review C2)

**Today** (§3.2). `_assert_output_claimable_locked` (`shell/_runs_registry.py:1275-1300`)
calls `_foreign_run_conflict` (`:1407-1479`) unconditionally. It refuses every
`completion == "failed"` output (`:1469-1474`), GUI-launched or not, so no output with a
failed image can be relaunched from the GUI.

**Behaviour (D10).** An output whose run `run_finished_with_failures` (PA4) confirms is
claimable, whoever launched it, for:
- **Run**, which continues the run and skips the recorded failures;
- **`--retry-failures`**;
- **Validate**;
- **Measure** (PB5).

`_foreign_run_conflict` returns `None` for it, as it does for `completion == "complete"`. Its
docstring's "Fires when" list (`:1418-1423`) says so.

**What stays refused, and why:**

| State | Where | Why it stays refused |
|---|---|---|
| An owner record that does not parse, or names an unknown status | `:1283-1287` (via `_read_owner_record`, `:1209-1210`) | No owner can be established, so nothing proves the output is free. This includes a newer GUI's `complete_with_failures` read by an older GUI (PA4) |
| A non-terminal GUI owner with no proof of death, which includes **every SLURM launch** | `:1288-1294`, `_dead_owner_reason` (`:1302+`) | A launch that may still be running; the scheduler, not a pid, owns its liveness |
| `completion == "active"` | `:1462-1466` | A liveness authority (a SLURM fence, a live pid) claims the output; a second launch would be a second writer |
| `completion == "failed"` that the verdict does **not** confirm | `:1469-1474` | An unfinished run: an unresolved image, no aggregate proof although an image verified, an aggregate marker beside zero verified images, or a publication that no longer covers the run (C1, D16). Its state is mid-way, and a new launch would mix with it |
| `completion == "incomplete"` | `:1475-1479` | Accepted work that never verified, or was never published over |

**A killed relaunch can only be resumed from the CLI (N7), if it verified something.**
Suppose a GUI-launched `--retry-failures` (or plain Run) relaunch on a confirmed output is
killed before it finalizes. If it verified at least one image, the source set no longer matches
the proof (or, for an all-failed run, an image is now verified with no proof), so the output is
an unconfirmed `failed` run, and this gate refuses it, as D10's "still refuses … unfinished"
requires. If it verified nothing, every earlier failure row is still in the journal and the
proof (if any) still covers the run, so the output reads confirmed again, unchanged, and stays
claimable (plan re-review R2-I2). The GUI's own relaunch is what strands the user on the
CLI, so the refusal message and the tutorial (PA7) both say to re-run the same command from
the CLI.

**A local CLI run is invisible to the gate (D17).** The liveness authorities are the SLURM
fence and a GUI owner with a live pid (`sdk_/_run_state.py:1078-1140`); a local CLI process
holds neither, and the CLI takes no output-wide lock (its one `exclusive_path_lock` guards the
metadata snapshot, `phenotypicCLI.py:1255-1268`). So the GUI can claim a confirmed output that
a local CLI process is re-running at that moment. This is accepted and documented: the
tutorial (PA7) and the run docs (PA8) say not to run the same output from the CLI and the GUI
at once.

The refusal message for an unconfirmed `failed` run no longer calls a GUI-launched run
"non-GUI". It says "output has an unfinished run with N failed image(s) of M accepted; re-run
it from the CLI or wait for it to finish".

**Tests** (`tests/unit/gui/shell/test_runs_registry.py`):
- `test_a_failed_image_refuses_the_claim` (`:363-390`) is split into two tests:
  - a confirmed run with failures is claimable, whether CLI-launched (no owner record) or
    GUI-launched (a terminal owner record);
  - a failed image beside an **unresolved** image, or with a stale publication, still
    refuses.
- Each refused state in the table keeps or gains a test.
- A Validate and a `--retry-failures` launch into a confirmed output both allocate.

---

## 5. PR B: run-console Stage 4 and measure mode (#282 GUI items)

**Base:** `origin/feat/gpu-embedding-staging` @ `0993c8e06`. PR B is cut after #282 is
rebased or merged onto a `main` that contains PR A (§1). All citations in this section are
at that base and are re-verified when PR B is cut.

**What #282 requires of the GUI.** #282's spec names only the control ("A run-console control
for `--embed-stage` is a GUI follow-up. It needs the FEATURES.md and WORKFLOWS.md ledgers",
`design.md:418`). The rules the GUI must mirror are:
- **U2** (`:77`): `auto` is `gpu` on SLURM with a GPU profile, otherwise `inline`; a local run
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

### PB2. GPU section visibility and the Auto hint match the CLI's resolution

**Today.**
- `show_staged_gpu_controls` (`run_console/_callbacks.py:1840-1865`) shows
  `RC_STAGED_GPU_SECTION` only when `mode == "slurm" and uses_gpu`.
- `uses_gpu` comes from `_staged_gpu_capability` (`:255-277`), which calls
  `pipeline_requires_gpu` (`_cli/_cli_validation.py:442`), true for a `GpuDetector` only.
- So a DINO-only pipeline can never pass `--gpu-slurm`. SLURM `auto` then always resolves
  `inline` (`resolve_embed_stage`, `_cli_embed_stage.py:132-174`), and Stage 4 is unreachable
  from the GUI.
- `--gpu-slurm` tokens come straight from the textarea (`_callbacks.py:1964-1968`).
  `_build_subprocess_argv` emits every token **whether or not the section is visible**
  (`_slurm.py:197-210`).

**The resolver: one, not two.**

| | Approach | Pro | Con |
|---|---|---|---|
| a | Reimplement the rule in the GUI | no CLI change | A second copy of U2/U9/O3, which is anti-goal 6 |
| b | Call `has_slurm_gpu_profile` / `stage4_supported` / `resolve_embed_stage_for_config` (`_cli_embed_stage.py:350-468`) | the exact CLI code | They take an `ExecutionConfig` with about 20 required fields (`_cli/_cli_types.py:148+`), which the GUI cannot honestly build |
| **c** | **Factor `resolve_embed_stage_for_config`'s pipeline half into a public `resolve_embed_stage_for_pipeline(pipeline, requested, *, mode, is_slurm, gpu_slurm_given) -> (stage, notes)` in `_cli_embed_stage.py`, which the CLI then calls too (recommended)** | One producer; the GUI passes only what it has | A small refactor in #282's module, in PR B |

**[call]** Pick (c). The pure pieces already exist (`resolve_embed_stage` at `:132`,
`pipeline_embedders` at `:88`). Only `_cpu_pinned`, `_cpu_pinned_key`, `_has_stage4_encoder`,
`_unstageable_embedder` and `_has_gpu_detector` (`:104-129, 471`) are private, and (c) keeps
them private. The GUI passes `mode="full"` or `mode="measure"` from PB5's selector.
`resolve_embed_stage`'s `gpu_profile = gpu_slurm_given or (mode == "full" and has_gpu_detector)`
(`:173`) then applies §4.1's measure rule with no GUI-side copy.

**Visibility, derived from the resolver** (review I10). With
`P = resolve_embed_stage_for_pipeline(pipeline, control, mode=cli_mode, is_slurm=True,
gpu_slurm_given=True)`, the section is shown when `mode == "slurm"` and either:
- `cli_mode == "full"` and the pipeline holds a `GpuDetector` (today's rule, Stage 2); or
- `P.stage == "gpu"`: Stage 4 would run if a GPU profile were given.

The second bullet needs no GUI copy of "Stage-4 capable". The pin, stageless-embedder and
no-embedder cases all resolve `inline` inside the resolver.

**No hidden flags.** `_build_subprocess_argv` sends `--gpu-slurm` and `--gpu-shards` **only
while the section is shown**. `gpu_slurm_given`, for the hint and for PB3's U12 check, is
computed from the same normalized token list the builder sends: the stripped, non-empty lines
of `_callbacks.py:1964-1968`. So the CLI never counts a GPU profile the user cannot see. The
tokens stay in the form state, and are sent again when the section reappears.

**The Auto hint, with U12 adoption** (review I9). Under `auto`, the hint below the control
states what the CLI will run. It is computed in the CLI's order:
1. `resolve_embed_stage_for_pipeline(..., gpu_slurm_given=<sent tokens non-empty>)`;
2. then, in full mode on a continuation (PB3's rule: the output holds a processing state),
   `adopt_recorded_embed_stage(recorded, "auto", resolved, is_slurm=…, stage4_available=True,
   gpu_profile=…)` (`_cli_embed_stage.py:284-347`), as the CLI does at
   `phenotypicCLI.py:3384-3402`.

The hint shows the final stage and the CLI's own note:
- the resolver's U9/O3 note, for example "meas 'dino' is pinned to device='cpu', so
  --embed-stage auto resolves to inline";
- or the adoption note, when a pre-#282 run recorded no stage, read as `inline`, and the
  continuation keeps embedding inline.

A refusal from either step goes to PB3's alert.

The section's wording stops saying "detection":
- the header "Staged GPU detection" (`run_console/_form.py:513`) becomes "GPU stages";
- the shards help "One resident model per whole-GPU shard." (`:556`) covers Stage 2 and
  Stage 4;
- also update `_ids.py:136-137` and `_state.py:100`, and `docs/source/tutorials/gui/05_run_slurm.md:22, 44`.

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
| U12 CPU-profile refusal (recorded `gpu`, `auto`, SLURM, no GPU profile) | inside `adopt_recorded_embed_stage`, `:284-347` (raised `:315-321`) | recorded stage, control, mode, and the sent `--gpu-slurm` tokens (PB2) | full mode only, as R3 |

**Behaviour.**
- **The probe.** It becomes `_launch_refusal(pipeline_path, cli_mode, exec_mode, embed_stage,
  gpu_slurm_tokens, output_dir) -> str | None`. It runs the CLI's checks in the CLI's order
  (`phenotypicCLI.py:3310-3314, 3334, 3384-3402`), catching `click.UsageError`:
  - the `UnstageableGpuDetectorError` check, **in full mode only**, because
    `uses_staged_gpu_strategy` returns `False` for measure (§3.4);
  - R2;
  - R3/U12, in full mode only;
  - PB5's measure refusals, in measure mode.
- **The alert text** is the CLI's message, unchanged, under the existing "This pipeline cannot
  be run:" prefix.
- **Inputs.** The callback adds:
  - `RC_RADIO_EMBED_STAGE`;
  - PB5's mode selector;
  - the GPU-profile textarea, normalized and gated by visibility as in PB2;
  - `RC_STORE_FORM_STATE`, for the output directory. That last pattern already exists in
    `show_reference_metadata_requirement`.
- **The recorded stage** is read with `load_processing_state` (`_cli_state_management.py:249`).
  A missing state means no R3/U12 check.
- **[call]** The run console has no `--restart` or `--overwrite` (`_state.py:67-113`), so in
  full mode an output holding a processing state is a continuation. That matches the CLI's
  `continuing` (`phenotypicCLI.py:3242-3248`) for every full-mode argv the GUI can produce.
- **The launch seam** (`_callbacks.py:2039-2041`) calls the same function, so a click racing
  a form change is refused before allocating.
- **The claim gate** (PA11) still runs at `allocate`. A refusal there is a separate message,
  not this alert.

**Approaches for R3/U12:**

| | Approach | Pro | Con |
|---|---|---|---|
| a | Static, with the CLI's functions (recommended) | instant, same text, Run disabled | must read `processing_state.json` on each form change, a small JSON read |
| b | Run `--dry-run` in the background on each change | the CLI decides everything | spawns a process per edit; slow on a large inventory; and impossible in measure mode (§3.4) |
| c | Leave R3/U12 to the launch failure | no work | the user learns after clicking Run, from a log |

Pick (a). The CLI stays the final authority: a refusal the probe misses still fails the launch
with exit 2. After PA1 that exit never cancels anything.

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
| `--slurm`, `--gpu-slurm`, `--gpu-shards`, `--embed-stage` | sent (GPU flags only while the section is shown, PB2) | the same; Stage 4 is measure's re-embed path (U4) |

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
- **Why not the forward evidence check** (review Minor 5). `_local_completion_evidence_conflict`
  (`_runs_registry.py:691-770`) is not applied. It judges whether the *forward* run's
  accepted images succeeded and its completion marker is complete. A measure pass's success
  is a different question, which only its exit code answers, U16 having made exit `0` mean
  every image was re-measured. A GUI-launched measure with no failures does publish the GUI
  completion marker (`phenotypicCLI.py:4622-4628`), but the marker does not distinguish a pass
  from the forward run, so it adds nothing.
- **N for a measure exit `3`** (review I3), defined the way the CLI does, and worded by which
  source applied:
  - **Local.** If this pass's failure count is positive, N is that count. The source is the
    terminal-failure journal rows this pass wrote, with measure work ids (#281 P2 I5): "N
    images failed to re-measure; their previous measurements and embeddings are kept" (U8).
    Otherwise N is the forward run's `diagnostics.failed`: "measure pass complete; the run
    still has N failed images, which a measure pass does not retry". This matches
    `_run_failure_count` (`phenotypicCLI.py:1528-1550`).
  - **SLURM.** If `_generation_failures(output_dir, generation)` (`0993c8e06 _cli_wait.py:304-322`) is
    positive, N is that count, with the first wording. Otherwise N is the forward
    `diagnostics.failed`, with the second.
  - **[call]** If the plan finds no per-pass count on disk for a local pass, the first branch
    is unavailable, and the second is used only when the forward run has failures. Otherwise
    the detail says "complete with failures; see the log".

**A SLURM measure: what the observer shows.** The GUI never runs `phenotypic-wait`, so it never
sees exit `5`. At the inactive-fence branch (`_slurm_observer.py:661-671`) it derives the
same outcome through `finished_run_outcome` (PA4). On #282's base that wrapper includes the
`terminal_status` override (`_cli_wait.py:228-257`; §1 step 4):

| Outcome | Status | Detail |
|---|---|---|
| `COMPLETE` | `complete` | "measure pass complete" |
| `COMPLETE_WITH_FAILURES` | `complete_with_failures`, N as defined above | the wording of whichever N applied |
| `ENDED_UNPUBLISHED`, with the fence `closed_by: finalizer` and `terminal_status: "failed"` (U16) | **`failed`**, terminal (D13) | the fence's `terminal_error` (`unfinished_measure_line`), which already ends "re-run the same command" |
| `ENDED_UNPUBLISHED` otherwise (cancelled, finalizer died) | `failed`, or `cancelled` per the existing cancel branch | as today |

**D13 (review C3).** A U16 leftover is `failed`, a terminal status, **not** `incomplete`.
`incomplete` is non-terminal by design (`_runs_registry.py:89-100`; `_TERMINAL_STATUSES`
`:133-135`). A persisted `incomplete` would have two effects:
- the observer would never stop polling the record;
- every relaunch would meet "output already has a durable nonterminal launch generation"
  (`:1288-1294`), because `_dead_owner_reason` returns `None` for every SLURM launch.

That is a permanent lock against the "re-run the same command" U16 prescribes. With
`failed`, the observer stops and the output is claimable for the re-run: the forward run
still verifies, so its completion is `complete` or a confirmed `failed` (PA11).

The observer gains a read of the fence's `terminal_status`/`terminal_error` for the detail;
today it reads neither.

**The embed-stage resolver in measure mode.** `resolve_embed_stage_for_pipeline` (PB2) is
called with `mode="measure"`:
- only `--gpu-slurm` counts as a GPU profile, so the GPU section is shown for a pipeline that
  resolves to GPU with a profile, and never for a `GpuDetector` alone;
- R2 applies;
- R3/U12 do not (§3.4).

**What measure mode in the GUI cannot do**, stated plainly in the tutorial text and the
FEATURES row:
1. **No Validate.** There is no dry run, so no `sbatch --test-only` check of a SLURM measure's
   profile before submission.
2. **No measure beside a live forward run.** The CLI allows a measure pass to run beside live
   forward workers (`_refuse_measuring_reference_pipeline`'s docstring, `:764-765`). The GUI
   holds one generation per output, and its claim gate refuses an output a live run claims
   (PA11, `completion == "active"`).
3. **No measure over an unfinished forward run.** A forward run with unresolved images, or a
   publication that no longer covers it, is refused by the claim gate (PA11). A forward run
   that **finished** with failures, CLI- or GUI-launched, can be measured (D10).
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
  - no `--gpu-slurm` is sent while the section is hidden;
  - a preset without `cli_mode` loads as full.
- **`test_run_console_callbacks.py`:**
  - the input slot is hidden in measure mode;
  - Validate is disabled, and the seam refuses a stale validate;
  - a missing or empty output disables Run with the message;
  - a reference-metadata pipeline shows the CLI's refusal;
  - the GPU section appears for a DINO-only pipeline in SLURM measure mode and not for a
    `GpuDetector`-only one;
  - R3 is never evaluated in measure mode;
  - the Auto hint shows the adoption note on a pre-#282 continuation (I9).
- **`test_runs_registry.py`:**
  - a measure record round-trips `cli_mode`;
  - local exits 0, 3 and 1 map as specified, without the forward-marker check;
  - exit 3 with no pass failure uses the forward-count wording (I3).
- **`test_slurm_observer.py`:**
  - the four outcome rows;
  - a U16 fence gives `failed`, with the `terminal_error` text;
  - **after a U16 observation the observer stops polling the record, and `allocate` on the
    same output is not refused for non-termination** (C3);
  - a clean pass over a forward run with failures shows the forward-count wording, not
    "0 images failed" (I3).
- **An integration test against the real CLI.** Each argv the form can build for measure mode
  is accepted by the CLI's argument parsing.
  - **The stopping point** (review Minor 18). Monkeypatch `scan_store_outputs` (called at
    `phenotypicCLI.py:3518`) to raise a sentinel exception. The test asserts it was reached,
    which proves every usage check above it passed, and that the exit is not `2`.
  - It drives `phenotypicCLI` through Click's `CliRunner`.

---

## 6. Test strategy

**Instruments by stage**, per the root `CLAUDE.md`:
- **Per task:** the directly touched test files, about a minute, run on Slurm
  (`--partition=batch` or `intel`, explicit `--mem`/`--time`, never `preempt`, never
  `-n auto`, `QT_QPA_PLATFORM=offscreen`, `-o addopts= -m "not slow"`).
- **Per phase:** the affected surface, derived from importers rather than directory names.
  - **PA1/PA4/PA11:** `tests/unit/gui/run_console/`, `tests/unit/gui/shell/`,
    `tests/integration/gui/test_run_console_callbacks.py`, `tests/unit/gui/test_runs_registry_layout.py`,
    `tests/unit/gui/test_snapshot_status.py`, `tests/unit/sdk_/` (the new verdict), and the
    `_cli_wait` tests under `tests/unit/cli/`.
  - **PA2/PA3:** `tests/unit/gui/results_viewer/`.
  - **PA5/PA8:**
    - `tests/unit/cli/`, `tests/integration/cli/`, `tests/unit/sdk_/`, `tests/gui/results_viewer/`
      and `tests/unit/gui/results_viewer/`;
    - **every importer of `tests/_output_layout.py`** (62 files; 28 of them call
      `write_measurements_mirror`), because its default changes.
  - **PA6:** `tests/unit/gui/{builder,shell}/`, the red tune tests, `tests/unit/gui/analysis/`.
  - **PA9:** `tests/unit/gui/shell/`, `tests/unit/gui/builder/test_reference_metadata.py`,
    `tests/unit/gui/run_console/test_reference_requirement.py`, `tests/unit/gui/tune/`,
    `tests/unit/analysis/` (the `ExpectedVsDetectedCount` change), `tests/gui/browse/`.
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
    viewer and analysis specs. The tune app is unmounted, so its specs do not count toward
    this gate.
- **The ledger gates:** `scripts/check_features_md.py`, `--strict` for the merge gate, and
  `scripts/check_workflows_md.py -v`. These are the `features-md-gate` and
  `workflows-md-gate` CI jobs.
- **The 13 reds:** all green, with no `xfail`, skip, deselect or `-k`, in the PR A per-phase
  run, and again after the rebase (§4 PA3). A red outside the 13 is a real failure.
- **The #282 rebase gate (I2).** After #282 is rebased onto a `main` holding PR A, and before
  PR B is cut, two tests pass:
  - `finished_run_outcome` returns `ENDED_UNPUBLISHED` for a same-generation fence carrying
    `terminal_status`;
  - `wait_for_run` still drains `_ledgered_jobs` after the outcome is decided.
- **The docs build** (PA7 and PA8 edit how-to pages, tutorials and a notebook): one Slurm job,
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

- **R1. PR A edits test expectations to make them green**:
  - reds #7 and #8;
  - `test_a_failed_image_refuses_the_claim`, which D10 changes;
  - `test_abnormal_exit_cancels_recovered_inactive_generation`, which PA1 changes.

  A reviewer could read that as weakening. The mitigation is the property-preservation rule
  in §3.3, and the deliberate behaviour changes named in each commit message.
- **R2. A new terminal status reaches code that enumerates statuses** without going through
  the shared sets. Two examples are `results_viewer/_qc_tab/_rebuild.py:171-172` and any JS
  that keys on the CSS class. The existing vocabulary guard
  (`tests/unit/gui/shell/test_runs_registry.py:1268`) catches a missed spelling. A Recent Runs
  render test covers each `RunStatus`.
- **R3.** *Withdrawn: D18 defers the error-analysis schema bump to §8.*
- **R4. The two aggregate-proof checks disagree after curation** (§3.2). The verdict avoids
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
- **R6. PA8 removes a deliverable users open in Excel**, and that a dashboard download command
  and outside scripts name. Excel opens a `.tsv`, but by import, not by double-click on every
  platform. The README and the how-to pages say so.
- **R6a. Known external consumers that break at the next finalize (D12).** The user's own
  project scripts read `deliverables/measurements.csv`:
  - `projects/ucr_029_e_d_Maresca/scripts/plot_strain_growth_scatter.py:22, 65`, pointing at
    `data/results/2026-10-26-gc-ome-zarr/deliverables/measurements.csv`;
  - `projects/ucr_033_e_d_Linzer_Ganoderma/scripts/plot_strain_growth_scatter.py:37, 90`;
  - `projects/ucr_029_e_d_Maresca/scripts/r_plots/check_new_columns.py:14`.

  The user will update them; PR A edits nothing outside the repository. A tree's CSV goes at
  its next finalize, and for a run with failures that is its next plain re-run (§3.2). So
  these scripts break as soon as such a tree is re-run under PR A.
- **R7 (PR B). The probe reads `processing_state.json` on every form change.** It is bounded
  to one small JSON read and memoised on mtime, like the classifier (`shell/_classifier.py:201-216`).
- **R8. The shared test helper's default flips** (`write_measurements_mirror(csv=False)`).
  28 files call it, and 62 import its module. Mitigation: run every importer at the PA5/PA8
  phase gate, and attribute each red in isolation before acting on it.
- **R9. The merge-order window (D5).** #281 merges to `main` before PR A. Until PR A merges,
  `main` carries every break in §3:
  - a SLURM launch that can read `failed`, or cancel its own generation;
  - an empty Colony picker;
  - a tune Deploy refusal;
  - 13 red GUI tests.

  #281's own spec forbids that order (§2.2), and #283 says the same. D9 settles it: #281 is
  held until PR A is ready, and the two merge back to back, so the window is the gap between
  the merges.
- **R10. Rebase drift.** PA citations are at `36b9e840e`. #281 may change again before it
  merges, and `main` will hold other merges. Every citation, and the 13-red table, is
  re-verified after the rebase, and the reds are re-run there (§1).
- **R11. PA9 changes a public analyzer's accepted input** (`ExpectedVsDetectedCount` gains
  `.tsv`). It widens the input set and refuses nothing it accepted before.
- **R12. The 60-second submitter timeout** (PA1, review Minor 14). A SLURM recompile that falls
  back to local work inside the submitter, and takes longer than 60 s, is killed and treated
  as ambiguous. It predates the spec and is not changed here.
- **R13. #282's waiter rewrite conflicts with PA4's** (§1 step 4, review I2). The conflict is
  resolved in #282's rebase, not in PR A, and is pinned by the rebase gate tests (§6).
- **R14. D10 opens a concurrent-writer window against a local CLI run (corrected; D17).** A
  local CLI process holds no liveness authority (`sdk_/_run_state.py:1078-1140`) and takes no
  output-wide lock, and a re-run of a run with failures always re-finalizes (§3.2). So the GUI
  can claim and launch into a confirmed output while a local CLI run is re-running it. Before
  D10 the gate refused every `failed` output; the same window already existed for `complete`
  outputs, whose re-run is a no-op. A SLURM CLI run is still fenced. Accepted and documented
  (D17); a CLI liveness record would be a CLI change outside both PRs.

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
> - **Tune app suffixes, when `/tune/` is re-mounted.** The pickers'
>   `PIPELINE_CONFIG_SUFFIXES | {CONFIG_SUFFIX_TUNING}` sets (`_gui/tune/_setup_authoring.py:47`,
>   `_layout.py:273-275`, `_callbacks.py:310, 1243`) omit the legacy tune suffix.
>   - The winner readers (`_run_root.py:120, 137, 150`) should read through
>     `resolve_best_pipeline_path`.
>   - The writers (`_export.py:36-39, 51-54, 86`, `_winner.py:100-106`,
>     `_callbacks.py:2424-2428`) should retire the old typed twin, as
>     `tune/_tune_cli/_run.py:89-101` does.
> - **Error analysis as TSV (D18).** `error_analysis.tsv` under a version-2 publication: the
>   name-set validators (`_error_tab/_publication.py:490-495, 638-644, 662-676, 710-714`), the
>   journal writer (`:369`), `_generation_digest`'s pinned version (`:799-827`, so an unchanged
>   v1 tree is not "already published", `:331-338`), version-1 reads and rollback, and
>   retiring `error_analysis.csv` after publishing (drop `"error_analysis"` from
>   `_KEPT_CSV_STEMS`).

---

## 9. My judgement calls (for review)

1. **The two stale tests.** #7 and #8 are fixed by test edits that keep each test's property
   (§3.3).
2. **SLURM exit `0`/`3` means "finished without submitting"**, and is never bound to an
   earlier generation's job set (PA1 b).
3. **Exit `2` and exit `4` never cancel or bind an old generation.** Exit 2 does not reconcile;
   exit 4 reconciles with `cancel_inactive=False`, under which an inactive fence returns no
   jobs. Only the timeout and stream-error arms pass `True` (PA1).
4. **The 0/3 result is its own type**, not a `SlurmSubmitResult` subclass. A `cancelling`
   record meeting it becomes `cancelled` (PA1).
5. **PR A leaves the 60-second submitter timeout alone** (PA1, R12).
6. **The verdict `run_finished_with_failures(state, output_dir)` lives in `sdk_/_run_state.py`**,
   as the failed-run twin of rule 1. It takes the caller's `RunState`, and shares one private
   helper with `_run_proof_covers_current_inventory`. The waiter computes it inside
   `_run_snapshot` and passes a value to the pure `decide_wait_outcome`; a thin
   `finished_run_outcome` wrapper in `_cli_wait.py` serves only the SLURM observer (PA4).
7. **The verdict returns `None` for a process run**; no caller needs a process arm (PA4, N4).
8. **A new terminal status `complete_with_failures` plus `failed_images`**, coloured
   `--oi-orange-text` (PA4 a).
9. **When a curated TSV write fails, unlink the TSV and a stale CSV.** A stale CSV is retired
   only after a successful TSV write (PA5).
10. **`FilteredMeasurements` gets a one-writer swap, CSV to TSV**: neither left as is nor
    deleted (PA5).
11. *Withdrawn (D18): error analysis stays CSV.*
12. **One sdk_ predicate, `is_pipeline_config_path`, for every mounted listing surface** (PA6).
13. **The `ExpectedVsDetectedCount` `.tsv` extension is justified on CLI parity** (PA9).
14. **The mutation receipt records the failed count N** (PA10).
15. **The embed-stage control** sits outside the SLURM-only section, is disabled without an
    embedder, and emits nothing for `auto` (PB1).
16. **A local no-CUDA `gpu` run surfaces as a failed run**, not in the refusal alert (PB1).
17. **One resolver.** A public `resolve_embed_stage_for_pipeline` is factored out of
    `resolve_embed_stage_for_config`. Visibility and the hint derive from it, and the hint
    applies U12 adoption. GPU flags are sent only while the section is shown (PB2 c).
18. **R3/U12 are predicted statically** from `processing_state.json`, in full mode only (PB3 a).
19. **Two new frames in PR B** (the GPU section, the mode selector), and no new workflow row
    (PB4).
20. **Measure mode never sends a flag the CLI refuses**, and it keeps the `--metadata` handoff
    (PB5).
21. **Validate is disabled in measure mode** (PB5 a).
22. **A measure run's exit code is authoritative.** N is defined and worded the way the CLI
    counts it (PB5).
23. **`RunState` gains `finalization_digests` and `process_only_layer`** from
    `resolve_run_state`'s one config read, so the verdict needs no second read (PA4).
24. **The verdict's comparisons are two shared helpers**, `_identity_digests_match` and
    `_source_set_matches`, so rule 1 can call them with its own binding (PA4).
25. *Withdrawn (plan re-review R2-C1): no `excluded_work_ids` field.*
26. **`excluded_store_names(images)` is one sdk_ helper** shared by `_advisories` and the waiter,
    and the waiter carries its result on `RunSnapshot` (PA4, D15).
27. **The no-proof arm refuses stale deliverables and a non-finalizer fence** (PA4 step 4).

---

## 10. Decisions (user, 2026-11-24)

These settle the first draft's eight questions, the merge-order question (D9), the spec review's four (D10–D13) and the Revision 3 question (D14).

- **D1. Retire `measurements.csv`.** PR A includes the CLI change (PA8). Curation writes
  `.parquet`, then `.tsv` (PA5). Readers still fall back to `.csv` in older trees.
- **D2. Curation of a `complete_with_failures` run is allowed** (PA10). *Its original second
  sentence, "claiming a CLI-launched run for `--retry-failures` stays refused", is superseded
  by D10.*
- **D3. The snapshot badge is in scope:** "Run complete · N images failed" (PA4).
- **D4. No new tutorial frame for the status.** It is documented in text (PA7).
- **D5. PR A targets `main` after #281 merges.** It is built on #281's branch now, and before
  it is opened it is rebased onto the latest `origin/main`, with every citation re-verified.
  PR B is cut after #282 has been rebased or merged onto a `main` that contains PR A (§1).
- **D6. Measure mode joins the run console in PR B** (PB5).
- **D7. One live GPU-stage SLURM launch from the GUI in PR B,** on `exfab` or `short_gpu`,
  never `preempt` (§6).
- **D8. TSV-capable metadata pickers join PR A** (PA9).
- **D10 (review Q1). A run with failed images is claimable from the GUI even when it is not
  GUI-owned,** for Run, `--retry-failures`, Validate and Measure. The claim gate still
  refuses a live or unfinished run, and every other state it refuses for safety (PA11).
- **D11 (review Q2). A CLI re-run or `dvc repro` of a run with failures re-finalizes and
  reseeds the mirror without curation; this is accepted and documented.** The labels file
  survives, and the GUI re-applies the labels at its next write. There is no CLI change.
  Curate after the last re-run (PA7, PA8, PA10).
- **D12 (review Q3). The user's ucr_029/ucr_033 plotting scripts are left alone.** The user
  will update them. They are a known external consumer that breaks at the next finalize
  (R6a).
- **D13 (review Q4). A SLURM measure run that ends with unfinished images reads `failed`,
  terminal,** with the detail "re-run the same command" (PB5).
- **D9. Hold #281, then merge back to back** (settles R9, the `main` window). PR A is built and
  reviewed on #281's branch first. When PR A is ready, #281 merges to `main`; PR A is then
  rebased onto the fresh `origin/main`, its citations, the 13-red table and its checks are
  re-run there, and it merges at once. `main` never sits on the §3 breaks beyond that short gap.
- **D14. `phenotypic-wait` adopts the currency check** (user, 2026-11-24; settles the Revision 3
  question). The waiter's rule 3 goes through the shared sdk_ verdict, so a run whose published
  outputs no longer cover the current run (e.g. the metadata snapshot changed after finalize)
  exits `5` (`ENDED_UNPUBLISHED`), not `3`. Exit-code values are unchanged; exit `3` now
  always means the published outputs are current. The GUI and a DVC stage give one answer.
  The waiter's tests gain that case. *The realistic trigger (re-review N3) is a finalize that
  excluded an unprojectable store, whose proof omits a verified image; the metadata example
  above rarely falls inside the waiter's window. PA4 names it, tests it, and gives the exit-5
  line a variant; PA8 lists the docs that describe exit 5.*
- **D15 (user, 2026-11-25; plan finding 1). `--accept-failures` stays strict.** It maps only a
  final `3` to `0`. D14's exit `5` for an excluded unreadable store stays `5`, and the waiter's
  line names the excluded stores and says to fix or remove them (PA4). *The names come from the
  per-image unprojectable-store predicate, never from a set difference (plan re-review R2-C1).*
- **D16 (user, 2026-11-25; plan review Q1). An all-failed run is `complete_with_failures`**,
  through the verdict's explicit no-proof arm (PA4 step 4). The waiter stays at `3`, Recent
  Runs reads "complete · N failed", and the GUI can relaunch it with `--retry-failures` (D10).
- **D17 (user, 2026-11-25; plan review Q3). The local-CLI window is accepted and documented.**
  D10 can let the GUI claim an output a local CLI process may be re-running, because nothing
  records local CLI liveness. R14 is corrected; the tutorial and docs say not to run the same
  output from the CLI and the GUI at once (PA7, PA8, PA11).
- **D18 (user, 2026-11-25; plan review Q4). The error-analysis TSV is deferred** to the
  follow-up issue (§8). `error_analysis.csv` stays, and `"error_analysis"` stays in
  `_KEPT_CSV_STEMS`; only `measurements` leaves it (PA5, PA8).

---

## 11. Open questions (only the user can answer)

None. D9 settled the `main` window, D14 the waiter's exit 3, and D15–D18 the plan review's
questions.

---

## 12. Revision 3 (spec review)

The review is `docs/superpowers/reports/2026-11-24-gui-dvc-followups/spec-review.md`.

| Finding | Fix |
|---|---|
| **C1** The verdict accepted a stale aggregate proof | PA4 defines `run_finished_with_failures` in `sdk_/_run_state.py` as the failed-run twin of rule 1. It checks the inventory, configuration, finalization and source-set digests against the aggregate proof, through a helper shared with `_run_proof_covers_current_inventory`. The badge (PA4), the claim gate (PA11) and the guard (PA10) all go through it. Tests cover the killed `--retry-failures` and metadata-change continuations. Anti-goal 2 is widened |
| **C2** The claim gate refused every failed output | §3.2 states the real gate. New **PA11** implements D10, with a table of what stays refused and why. The false PA4 sentence is removed. PB5 limit 3 is rewritten, and the D2 text is superseded. `test_a_failed_image_refuses_the_claim` is split |
| **C3** `incomplete` wedged the output | D13: the U16 row in PB5 is `failed`, terminal. The test pins that the observer stops and `allocate` is not refused. Anti-goal 5 is added |
| **C4** A re-run re-finalizes and resets the mirror | §3.1 corrects the SLURM no-op claim: a run with failures and a verified image submits a recompile and exits `4`. §3.2 states the reseed. D11 is recorded. PA10's re-run test is rewritten. PA7 (tutorial) and PA8 (DVC how-to) say to curate after the last re-run |
| **I1** The `_cli` import broke the readers rule | The verdict lives in sdk_. Only the SLURM observer uses the `_cli_wait` wrapper. The Known consumers rows are added to the docs lists (PA4, PA8) |
| **I2** #282 rewrites `_cli_wait.py` | §1 step 4 names #282's rebase obligation. §6 adds the rebase gate tests. R13 |
| **I3** Measure exit 3 does not mean this pass failed | PB5 defines N the CLI's way (the pass's count when positive, else the forward count) and words the detail by source. Tests added |
| **I4** The badge arm skipped the currency check | PA4: the arm falls through to `snapshot_is_current`, with a "Changed on disk" test |
| **I5** Exit 4 with no job set could still cancel | PA1: `_reconcile_submission(cancel_inactive=False)`, with a `jobs is None` test |
| **I6** The 0/3 arm could not reuse `observe_local_exit` | PA1: a separate result type, the three seam sites and the fake updated, a new registry method, and the cancel race (gives `cancelled`) |
| **I7** Unlinking the TSV re-exposed the stale CSV | PA5: unlink both text files on a failed TSV write. The test is changed |
| **I8** PA9 missed the reference-metadata snapshot check | PA9 row for `reference_metadata_requirement`, with its test |
| **I9** The Auto hint ignored U12 adoption | PB2: the hint is computed as resolver, then adoption, with the adoption note. Test added |
| **I10** The visibility copy, and hidden `--gpu-slurm` | PB2: visibility comes from the resolver. GPU flags are sent only while the section is shown, and `gpu_slurm_given` comes from the sent tokens |
| **I11** External CSV consumers | D12; R6a lists the three scripts. No edit outside the repo |
| **Minor 1** U2 citation | `:77` |
| **Minor 2** R2's status list; a gate already exists | `_discovery_contracts.py` is dropped from R2. PA4 and R2 cite the existing vocabulary test |
| **Minor 3** The older-reader claim | PA4 corrected: `_read_owner_record` returns `None`, and the claim gate fails closed |
| **Minor 4** Which N | PA4: the run's `diagnostics.failed`; it agrees with the exit on zero/non-zero only |
| **Minor 5** PB5's reason for skipping the evidence check | PB5 restates the reason and cites the GUI completion marker |
| **Minor 6** PA2 needs no new export | PA2 uses `declared_measurement_table_path`. Reds #8 and #13 use it |
| **Minor 7** Tune is unmounted | PA6 keeps only the red-test sites. The rest move to §8. PA9 keeps D8's pickers, and justifies `ExpectedVsDetectedCount` on CLI parity. §6 drops the tune e2e specs from the gate |
| **Minor 8** `FilteredMeasurements` | Not as suggested. A one-writer swap (§9 call 10); reason in PA5 |
| **Minor 9** SLURM exit 3 is nearly unreachable | PA1: one synthetic test |
| **Minor 10** CSV retirement under the commit guard, only after the TSV | PA8 states both conditions |
| **Minor 11** Fingerprint both mirrors | PA5: `_consumed_state_snapshot_paths` fingerprints both |
| **Minor 12** Missed docs | PA8: `_cli_readme_generator.py:147`, `_cli/CLAUDE.md:777, 1659`, `_gui/CLAUDE.md:257, 324, 427` |
| **Minor 13** The import-walk test | PA5 states its edit, and corrects the reasoning |
| **Minor 14** The 60 s timeout | PA1 note; R12 |
| **Minor 15** Exit 2 still cancels | PA1: exit 2 reconciles without cancelling. The pinned test is changed (R1) |
| **Minor 16** The staged phase PA4 depends on | §3.2 states it. A PA4 test pins it |
| **Minor 17** A tautological TSV bytes test | PA5: a read-back check |
| **Minor 18** A stopping point for the CliRunner test | PB5: a `scan_store_outputs` sentinel |
| **Minor 19** 62 importers vs callers | PA8, §6 and R8: 28 callers, 62 importers |

**Re-review of revision 3** (appended to the same report):

| Finding | Fix |
|---|---|
| **N1** The verdict resolved its own state | PA4: `run_finished_with_failures(state, output_dir)`. Each caller passes the state it resolved at its own depth; the guard passes its fresh and verified deep states. The waiter computes the verdict in `_run_snapshot`'s single pass and passes a value to the pure `decide_wait_outcome`. §9 call 6 |
| **N2** `cancel_inactive=False` could bind an old generation; the exit-2 wording | PA1: an inactive fence under `cancel_inactive=False` returns no jobs without reconciling. Exit 2 does not reconcile at all. A no-stale-binding test is added. §9 call 3 |
| **N3** D14's exit-5 surfaces | PA4 names the realistic trigger (an excluded unprojectable store), tests it, and gives the exit-5 line a variant. PA8 lists the four surfaces. D14's text records the trigger |
| **N4** The unused process arm | PA4 step 5: the verdict returns `None` for a process run; a process tree with failures stays unclaimable. §9 call 7 |
| **N5** Stale §11 cross-references | The header, §1, PA4's waiter bullet and test, and R9 now cite D9 and D14 |
| **N6** The import-walk edit | PA5: assert `MEASUREMENTS_TSV` only |
| **N7** A killed GUI relaunch strands the user on the CLI | PA11 and PA7's tutorial text say so; the refusal message names the CLI re-run |
| **C4 note** A Refresh before the re-applying write | PA10's re-run test refreshes the viewer first |

**Plan review** (`reports/2026-11-24-gui-dvc-followups/plan-review.md`), the findings that
changed the spec:

| Finding | Fix |
|---|---|
| **C1** An all-failed run decided by accident | D16: PA4 step 4, the no-proof arm. The exit-5 variant prints only when this generation published a proof |
| **I1** The verdict had no config source | PA4: `RunState.finalization_digests` and `.process_only_layer`; step 2 over `state.images`; two shared helpers. §9 calls 23–24 |
| **I2, I3** No slot for N | PA4 "Edits (a) requires": `_Observation`, `_apply`, `compare_and_set`, `_rehydrated_status` and its two consumers |
| **I4** The run console's pipeline picker | PA6: a row for `run_console/_form.py:58-73`, and the builder wrappers |
| **I5** More consumers of the 0/3 result | PA1: `_SlurmCompletion.finished`, the banner, `shell/_app.py:215, 681` |
| **I6** The error-analysis v2 early return | D18: deferred to §8, with I6's two extra sites recorded there |
| **I12** R14 was wrong | D17: R14 corrected; PA11, PA7 and PA8 document the window |
| **Plan finding 1** `--accept-failures` and exit 5 | D15: strict; the line names the excluded stores (§9 call 26) |

**Plan re-review of revision 2** (`plan-review.md`, "Re-review of revision 2"), the findings
that changed the spec:

| Finding | Fix |
|---|---|
| **R2-C1** `excluded_work_ids` measured a gap | Dropped. PA4: `excluded_store_names(images)` from the per-image stage facts, shared with `_advisories`; otherwise the gap wording. §9 call 25 withdrawn, call 26 |
| **R2-I1** The pinned `RunState` field set | PA4: the field-set test and the `tracked_state.md` table change deliberately |
| **R2-I2** The no-proof arm | PA4 step 4: the marker tested by path; no master or mirror Parquet from another generation; the killed-retry sentence corrected; PA11's N7 paragraph and PA7's tutorial line follow |
| **R2-I4** The waiter keeps no `RunState` | PA4: `RunSnapshot` carries the names from the same pass |
| **R2-M4** A cancelled all-failed run read two ways | PA4 step 4: a fence must be finalizer-closed; `FENCE_CLOSED_BY_FINALIZER` moves to sdk_ |
| **R2-M5** "under `commit_guard`" | PA8: "when one is given" |
