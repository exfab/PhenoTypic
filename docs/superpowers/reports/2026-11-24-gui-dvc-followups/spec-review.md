# Spec review: GUI follow-ups for #281 and #282

**Reviewed:** `docs/superpowers/specs/2026-11-24-gui-dvc-followups/design.md` @ `fbe9ffcf6`
**Bases checked:** PR A at `36b9e840e` (this worktree); PR B at `0993c8e06` (read with `git show`).
**Reviewer:** independent plan reviewer, analysis only. This report is my only write.

Unless a line says otherwise, every `file:line` is at the PR A base `36b9e840e`, and every path is under `src/phenotypic/`.

## Verdict

**NEEDS REVISION before a plan is written.** Most citations are accurate, and the PA1 exit-code reading is safe. But four findings break a claim the spec relies on:

- **C1.** The "finished with failures" helper would certify stale deliverables.
- **C2.** The claim gate refuses *every* output with a failed image, including GUI-launched ones. The spec says a GUI-owned run stays claimable, and PB5 depends on that.
- **C3.** PB5's `incomplete` status for a U16 measure leftover is non-terminal. On a SLURM record it locks the output against the very re-run U16 prescribes.
- **C4.** A plain re-run of a `complete_with_failures` run always re-finalizes and resets the curated mirror, so PA10's re-run test cannot pass.

C1 and C3 are spec fixes. C2 and C4 also need a decision only the user can make (listed at the end).

---

## Critical

### C1. The generation-less verdict accepts an aggregate proof that no longer covers the run (PA4, PA10)

**Evidence.**
- **What the spec's check tests.** For local callers, rehydration, the viewer and the mutation guard, PA4 requires `aggregate_proof_is_current(output_dir)`. That function only re-hashes the descriptors, skipping the GUI-written ones (`sdk_/_run_state.py:1338-1378`, loop at `:1369-1376`). It never compares the proof to current state.
- **What the proof carries.** It records `inventory_digest`, `finalization_input_digest`, `scientific_config_digest`, `source_set_digest` and `source_image_count` (`_cli/_cli_completion.py:1239-1262`).
- **The check a complete run gets.** `_run_proof_covers_current_inventory` (`sdk_/_run_state.py:1464-1541`) makes five comparisons. Its docstring says why each is load-bearing: without `inventory_digest` a new image never invalidates completion; without `finalization_input_digest` a metadata edit goes unnoticed. A failed run has no run proof, so nothing in the spec's rule makes those comparisons.

**Concrete reachable case.**
1. A run R1 finalizes with failures. Its aggregate proof P1 is current.
2. The user re-runs with `--retry-failures`, or with a new `--metadata`.
3. Some images now verify, or the snapshot changes.
4. The run is killed (OOM, walltime, `kill -9`) after the last image but before finalize.

Then:
- `completion` is `failed`, because one image still has a terminal record;
- `unresolved == 0`;
- P1's bytes still match, because the master was not rewritten.

`finished_run_outcome` returns `COMPLETE_WITH_FAILURES`. The badge says complete, and PA10 authorizes curation of a mirror that is missing the newly verified images or carries the old metadata join. That is anti-goal 2 in spirit, and D2's write authority over stale data.

**Fix to the spec.**
- Define the generation-less arm as a failed-run twin of rule 1's clause 2, all of:
  - `aggregate_proof_is_current`;
  - `proof.inventory_digest == identity.inventory_digest`;
  - `scientific_config_digest` matches;
  - `finalization_input_digest ∈ accepted_finalization_digests(config)`;
  - `source_set_digest == canonical_digest(sorted(verified work ids))` and `source_image_count == len(verified)`.
- Put it in `sdk_` beside `_run_proof_covers_current_inventory` (see I1).
- Add a test for each case: the killed `--retry-failures` continuation, and the killed metadata change. Each must stay `failed`.

### C2. No output with a failed image is claimable from the GUI, whoever launched it (PA4, PB5, D2)

**Evidence.**
- `_assert_output_claimable_locked` (`_gui/shell/_runs_registry.py:1275-1300`) checks the owner record. A terminal owner passes (`:1288`). It then calls `self._foreign_run_conflict(output_dir)` **unconditionally** (`:1298-1300`).
- `_foreign_run_conflict` (`:1407-1475`) never looks at the owner. For `completion == "failed"` it returns `"output has failed non-GUI processing state …"` (`:1469-1474`).
- `allocate` reaches the gate on every launch, Validate included (`:412-413`).
- Pinned by `tests/unit/gui/shell/test_runs_registry.py:363-390`.

**So:**
- PA4's "A GUI-owned `complete_with_failures` record is terminal, so its output is claimable as any terminal GUI record is" is false.
- After a GUI run ends with failures, the GUI refuses Validate, Run (continuation), `--retry-failures` and PB5's measure on that output. The message calls the GUI's own run "non-GUI".
- PB5's limitation 3 ("No measure over a *CLI-launched* run that has failures") understates this: measure over **any** run with a failed image is unreachable.
- D2 says claiming a CLI-launched run for `--retry-failures` "stays refused", which implies a GUI-launched one is claimable. The code makes that impossible as specified.

**Fix to the spec.**
- State the real gate.
- Either:
  - (a) keep the gate, and amend PA4, PB5's limits, the tutorial text and the FEATURES rows to say no failed output can be relaunched from the GUI; or
  - (b) make `_foreign_run_conflict` stand down when a terminal GUI owner record exists for this output. That needs a fence: the owner's generation matching the run identity, or the marker's `gui_record_generation`.
- (b) is a scope change, so it is a user decision (Q1).

### C3. PB5 gives a U16 leftover the non-terminal status `incomplete`, which wedges the output

**Evidence.**
- **`incomplete` is non-terminal by design.** The comment at `_runs_registry.py:89-100` says "It is NOT terminal, and that is the trap". `_TERMINAL_STATUSES` is `{"complete","failed","cancelled"}` (`:133-135`). The observer's own set is the same (`_gui/run_console/_slurm_observer.py:62`).
- **Today it only appears in memory.** Rehydration produces it with `persist=False`, and `_release_dead_owner_locked` coerces any non-terminal verdict to `failed` before persisting, "precisely so a repaired record can never carry it" (`:96-100`).
- **What a persisted `incomplete` does.**
  1. The observer never stops polling the record.
  2. The next launch meets `persisted.status not in _TERMINAL_STATUSES` (`:1288`) and asks `_dead_owner_reason`. That returns `None` for every SLURM launch (`:1302+`), so the launch raises "output already has a durable nonterminal launch generation".

U16's remedy is "re-run the same command" (`unfinished_measure_line`). The GUI would refuse it permanently, with no UI affordance to clear it.

**Fix to the spec.** Pick one:
- a terminal status: `failed`, with `terminal_error` as the detail and the "re-run the same command" text;
- a new terminal literal.

Do not persist `incomplete`. Add a test that, after a U16 observation, the observer stops and `allocate` on the same output is not refused for non-termination.

### C4. A plain re-run of a `complete_with_failures` run re-finalizes and resets the curated mirror (PA10, §3.1)

**Evidence (36b9e840e).**
- **The refresh condition.** For a markers-era tree with at least one verified image, `publication_refresh_required` is set whenever `startup_state.completion != "complete"` (`phenotypicCLI.py:3765-3790`). A run with failures is never `complete`, because no run proof is written over a failure (`_cli/_cli_completion.py:1304-1342`). So every plain re-run re-finalizes. It takes the `:3925-3950` path, not the `:3964` "already processed" exit.
- **What finalize writes.** It reseeds the mirror from the post-applied master (`_seed_measurements`, `_cli/_cli_output_manager.py:968-1003`). It re-emits only `errors/` and `error_analysis.*` from `curation_labels.parquet`; `_cli/_cli_error_outputs.py:63` says "(no mirror)". **A curated mirror is reset to uncurated** on every CLI re-run or `dvc repro` of such a run. A complete run's re-run is a no-op and keeps it.
- **SLURM.** The same re-run goes to `_handle_recompile_slurm` (`phenotypicCLI.py:3925-3944`). It submits whenever any measurement source exists (`:4672-4760`), so it exits `4`, not `0`/`3`.

**So:**
- PA10's test "a plain CLI re-run … still exits `3` and does not rewrite the curated mirror's labels" cannot pass as worded. The labels file survives; the mirror does not.
- §3.1's "A SLURM-mode run with nothing left to do never submits, and exits `0` or `3`" is wrong for a run with failures that has a verified image: it submits a recompile and exits `4`.

**Fix to the spec.**
- Correct both statements.
- Rewrite the PA10 test: labels and `errors/` survive; the mirror is reseeded; and the GUI re-applies labels at the next curation write. Or change behaviour (Q2).
- Record the asymmetry with complete runs in PA7's tutorial text, since D2 makes it reachable.

---

## Important

### I1. `finished_run_outcome` in `_cli/_cli_wait.py`, imported by the viewer, guard and badge, breaks the readers rule (PA4, PA10)

- `_gui/CLAUDE.md` ("Importing readers: `sdk_`, not `_cli`") says: "A **new** `_cli` import that answers 'is this run finished?' is a regression."
- `docs/source/contrib_guide/tracked_state.md:496` says: "A GUI or notebook consumer imports from `phenotypic.sdk_` only."
- The spec's helper would be imported by `_snapshot_status.py`, `_mutation_guard.py`, `_output_root.py`, `_processing_inventory.py` and the registry.

**Fix.**
- Put the generation-less verdict (C1's predicate plus rule 3) in `sdk_/_run_state.py` and export it.
- Let `_cli_wait.decide_wait_outcome` and a thin generation-bound wrapper compose it. Only the SLURM observer, which already imports lifecycle readers from `_cli`, uses the generation-bound wrapper.
- Add the new consumer rows to the **Known consumers** table (`tracked_state.md:433-450`). That page requires it "in the same change" (`:18`, `:525`), and the spec's docs lists omit it.

### I2. #282 rewrites the function PA4 refactors

Between `36b9e840e` and `0993c8e06`, `_cli/_cli_wait.py` changes by +123/−41 lines:
- `_published` gains the U16 `terminal_status` override;
- `_ledgered_jobs` is added;
- `wait_for_run` gets a frozen-outcome drain loop.

PA4 refactors the same code on #281. PB5's measure table relies on the override ("a fence closed with a `terminal_status` is never published"), and that exists only on #282.

**Fix.**
- Name the conflict in §1's sequencing as #282's rebase obligation.
- After #282 lands, require a test that `finished_run_outcome` returns `ENDED_UNPUBLISHED` for a same-generation fence carrying `terminal_status`, and that the waiter still drains `_ledgered_jobs`.

### I3. In measure mode, exit 3 and outcome `COMPLETE_WITH_FAILURES` do not mean this pass failed images (PB5)

**SLURM.** `decide_wait_outcome` rule 3 (`_cli_wait.py:356-364`) returns `COMPLETE_WITH_FAILURES` when the **forward** run is `failed`, everything is resolved, and the pass published. A clean measure pass over a forward run with failures takes this branch with `_generation_failures == 0`. PB5's row would then show "0 images failed to re-measure".

**Local.** The exit code comes from `_run_failure_count(output_dir, results.total_failed)` (0993c8e06 `phenotypicCLI.py:1528-1550, 4622`). That is this pass's failures when positive, else the forward run's. So exit 3 with no measure failure is normal, and PB5's "N from this pass's journal rows, or unknown" shows 0 or unknown for it.

**Fix.**
- Define N the way the CLI does: `generation_failures` (SLURM) or the pass's count when positive, else the forward `diagnostics.failed`.
- Word the detail by which one applied.
- Test both cases.

### I4. The new badge arm skips the currency axis (PA4, D3)

`snapshot_refresh_status` returns an unfinished badge **before** it asks `snapshot_is_current` (`_gui/_snapshot_status.py:88-96`). A new arm placed with it would read "Run complete · N images failed" over a re-finalized master. By C4 that happens on every CLI re-run of such a run. This is the CAN-18 failure the module docstring exists to prevent (`:5-9`).

**Fix.** Let the arm fall through to the currency check: "Run complete · N failed" when current, otherwise "Changed on disk". Test it with `test_a_rewritten_processing_product_shows_changed_on_disk`'s pattern.

### I5. On exit 4 with no job set, PA1 can still reach the cancel branch

PA1 says that on exit 4, when `read_submitted_job_set` is `None`, `submit_slurm` should "reconcile once, as a clean exit does today", and that "Exit 4 no longer reaches" the inactive→cancel branch. But today's clean exit with `jobs is None` calls `_reconcile_submission` (`run_console/_slurm.py:837-841`), and that function contains the cancel branch (`:652-656`).

**Fix.**
- Give the exit-4 reconciliation a flag that never cancels, or name the narrower function.
- The test list's "with an inactive fence it does not call `cancel_generation`" must cover `jobs is None`.

### I6. The 0/3 callback arm cannot reuse `observe_local_exit` as it stands (PA1)

- **Mode.** `observe_local_exit` maps any mode other than `local`/`validate` to `failed` with "unsupported run mode" (`_runs_registry.py:665-675`). A SLURM record is `mode == "slurm"`. So "decided by the same evidence check as a local exit" must call `_local_completion_evidence_conflict` and the PA4 helper directly. The spec should say so.
- **Result type.** `_complete_slurm_submission` raises `TypeError` for any result that is not a `SlurmSubmitResult` (`run_console/_callbacks.py:1042-1047`), and that lands in the `failed` arm. The submitter seam is typed `Callable[..., SlurmSubmitResult]` (`run_console/_app.py:74`, `_callbacks.py:1310`), and browser tests inject a fake (`shell/_app.py:246`). Either make the new result a subclass, or update those three sites and the fake.
- **Cancel race.** A `cancelling` record that meets a 0/3 result needs a stated outcome. The except arm’s expected-status set (`:1094-1101`) and the success arm’s (`:1125`) exclude `cancelling`.

### I7. A failed TSV write unlinks the TSV, which re-exposes a stale CSV (PA5)

PA5's [call] says a failed curated TSV write unlinks `measurements.tsv`, so that "a text reader then finds nothing, and never reads stale rows". On a pre-PA8 tree the CSV is still there, and `measurements_table_path` / `BundleLayout.measurements_table_path` fall back to it (`sdk_/_io_constants.py:1708-1723, 3379-3388`). So the reader gets exactly those stale rows. The spec's own test, "a failed TSV write leaves no `.tsv` and keeps the old `.csv`", enshrines the contradiction.

**Fix.** On TSV failure, retire the CSV as well. The Parquet is authoritative, and the CSV predates the curation.

### I8. PA9 misses the run console's reference-metadata snapshot check

`reference_metadata_requirement` (`run_console/_callbacks.py:305-327`) treats the requirement as met only when `metadata_csv_deliverable_path(output_dir)` (`deliverables/metadata.csv`) is a file (`:326`). A run snapshotted to `metadata.tsv` (#281 B5, `sdk_/_io_constants.py:1732-1739`) shows "reference metadata required" and disables Run for a valid continuation.

**Fix.**
- Add a PA9 row: resolve with `resolve_metadata_snapshot_path`.
- When the output holds both snapshot suffixes, show the resolver's refusal.

### I9. The Auto hint ignores U12 adoption on continuations (PB2)

On a full-mode continuation the CLI calls `adopt_recorded_embed_stage` after resolving (0993c8e06 `phenotypicCLI.py:3384-3402`). Under `auto` it **runs the recorded stage**. A hint computed from `resolve_embed_stage_for_pipeline` alone is wrong for every continuation whose recorded stage differs, for example a pre-#282 run (recorded none, read as `inline`) with a DINO pipeline and a GPU profile. The hint would promise GPU, and the run embeds inline.

**Fix.** PB3 already reads the recorded stage. Compute the hint as the resolver, then the adoption, and show adoption's note text.

### I10. GPU-section visibility is a second copy of the CLI rule, and `--gpu-slurm` leaks from a hidden section (PB2)

- **A copy of the rule.** PB2's second bullet restates "Stage-4 capable" (DINO, no CPU pin, no stageless embedder). That is the GUI-side copy of `_has_stage4_encoder`, `_cpu_pinned` and `_unstageable_embedder` (0993c8e06 `_cli/_cli_embed_stage.py:104-129`) that anti-goal 4 forbids. Derive visibility from the resolver instead, for example `resolve(..., gpu_slurm_given=True) == "gpu"`, plus the full-mode GpuDetector rule.
- **Hidden tokens still sent.** `_build_subprocess_argv` emits every `state.gpu_slurm_args` token whether or not the section is visible (0993c8e06 `run_console/_slurm.py:197-210`). Those tokens come straight from the textarea (`_callbacks.py:1964-1968`). The CLI then counts a GPU profile the user cannot see. That changes `has_slurm_gpu_profile` for U12 and `gpu_slurm_given` for `auto`.

**Fix.** Send `--gpu-slurm` only when the section is shown. Compute `gpu_slurm_given` from the same normalized token list the argv builder sends.

### I11. External consumers of `measurements.csv` are concrete, and retirement reaches live trees soon (PA8, R6)

The user's project scripts take `deliverables/measurements.csv` as input:
- `projects/ucr_029_e_d_Maresca/scripts/plot_strain_growth_scatter.py:22,65`, which points at `data/results/2026-10-26-gc-ome-zarr/deliverables/measurements.csv`;
- `projects/ucr_033_e_d_Linzer_Ganoderma/scripts/plot_strain_growth_scatter.py:37,90`;
- `projects/ucr_029_e_d_Maresca/scripts/r_plots/check_new_columns.py:14`.

PA8 deletes the file at the tree's next finalize. By C4, that includes every plain re-run of a tree with failures. R6 names "outside scripts" only generically.

**Fix.** List these scripts in R6. Add an explicit step, or a user decision (Q3), on updating them before PR A merges.

---

## Minor

**Citations**

1. **The U2 line.** The citation is `design.md:76`; U2 is at `:77` in #282's spec. U4 `:79`, U9 `:83`, U12 `:87` and U16 `:91` are correct.
2. **R2's status list.** It lists `results_viewer/_discovery_contracts.py:16`. That is the `DiscoveryPhase` literal (`"complete"` is a discovery phase), not a run status. The gate R2 proposes already exists: `test_every_status_vocabulary_in_the_gui_agrees_with_the_registry` (`tests/unit/gui/shell/test_runs_registry.py:1268`), cited by `tracked_state.md`'s "Status vocabulary is hand-copied". Use it instead of a new grep gate.
3. **The older-reader claim.** PA4 says an older GUI reaches `run_status_is_nonterminal`. It does not: `_read_owner_record` returns `None` for an unknown status (`_runs_registry.py:1209-1210`), and the claim gate then raises "output has an invalid generation owner" (`:1283-1287`). That fails closed, but the stated reason is wrong.
4. **N for local exit 3.** PA4 says N is "the count the CLI's exit code uses". `_run_failure_count` uses this invocation's count when it is positive (`phenotypicCLI.py:1493-1518`), so only the zero/non-zero decision agrees. Showing the run's `diagnostics.failed` is fine. Say which.
5. **PB5's reason for skipping the evidence check.** It says "a measure pass writes no forward completion marker". A GUI-launched local measure with `run_failed == 0` does call `publish_local_gui_completion` (0993c8e06 `phenotypicCLI.py:4622-4628`). The decision (the exit code is authoritative) stands; the reason does not.

**Simplifications**

6. **PA2 (b) needs no new export.** It proposes a new `sdk_` export that already exists: `declared_measurement_table_path(store)` (`sdk_/_table_runs.py:613-635`, exported at `sdk_/__init__.py:508, 630`). It returns `store / record_less_measurement_table(...).relative_path`, or `None`. Use it in `_store_and_identity`, and in reds #8 and #13's test edits.
7. **Tune is unmounted.** The hub does not compose `_gui/tune/`, and `/tune/` returns 404 (`_gui/CLAUDE.md`, "Unmounted surfaces"). Beyond what reds #1–6 need, PA6's tune rows (pickers, preset writer, winner readers and writers) and PA9's tune pickers change chrome no user can reach. PA9's one non-GUI change, extending `ExpectedVsDetectedCount` to `.tsv`, is justified only by those pickers. Cut them, or justify the `ExpectedVsDetectedCount` change on CLI parity alone. The §6 e2e lane's "tune specs" would exercise an unmounted app.
8. **`FilteredMeasurements`.** `_gui/CLAUDE.md` says it is "a utility / constants module only … do not extend it". PA5 rewrites its `_save_locked`. Consider leaving its CSV write as is, or deleting the method, rather than porting it.
9. **The SLURM exit-3 branch is nearly unreachable from the GUI.** C2 refuses a failed output before any submit. With a verified image, a re-run of a run with failures submits a recompile and exits 4 (C4). Keep the classification, but its tests are synthetic; one is enough.

**Plan details the spec should state**

10. **Two citations:**
    - **The `observe_local_exit` arm.** It is at `:665-675`, a span the `observe_local_exit` citation `:636-688` covers; this is the arm I6 relies on.
    - **The CSV retirement in PA8.** It must run under `commit_guard` and only when the TSV write succeeded. `_guarded_terminal_best_effort` swallows errors (`_cli/_cli_output_manager.py:99-115`), and the existing sweeps at `:838` and `:1502` are unguarded. State both conditions.
11. **`_consumed_state_snapshot_paths`** (`results_viewer/_output_root.py:1119-1133`). During the transition, fingerprint both `mirror_tsv` and `mirror_csv`, rather than swapping one for the other. A CLI retirement during discovery then forces a retry.
12. **Docs PA5/PA8 miss:**
    - `_cli/_cli_readme_generator.py:147`, which names `error_analysis.csv`;
    - `_cli/CLAUDE.md`'s "The GUI run console reads `4` as a failed submission" bullet and its "Two generated CSVs … stay CSV" paragraph;
    - `_gui/CLAUDE.md`'s state table row "Measurements mirror … `_curation_labels.py:846,848`", and its `PIPELINE_JSON # "pipeline.json.pht-pipe"` comment, which is already stale.
13. **The import-walk test hard-codes two names.** `test_the_walk_finds_the_gui_imports_the_spec_names` (`tests/unit/sdk_/test_gui_sdk_imports.py:89-92`) asserts the GUI imports both `MEASUREMENTS_CSV` and `measurements_csv_path`. The only GUI import of `measurements_csv_path` is `results_viewer/_filtered_state.py:44`, which PA5 rewrites. PA5's reasoning ("stays … because #281's import-walk test pins every sdk_ name the GUI imports") is inverted: the walk pins what the GUI imports, not what `sdk_` exports. State the intended edit to this test. Its docstring allows that when #283 lands.
14. **The 60-second submitter timeout.** PA1's "0/3: finished without submitting" can include real in-process work. The two local fallbacks in `_handle_recompile_slurm` run `_handle_recompile` inside the submitter (`phenotypicCLI.py:4673-4688, 4700-4713`), and the submitter is killed at 60 s (`run_console/_slurm.py:751`). The killed run then goes down the ambiguous path. This predates the spec. Note it, or raise the timeout for that case.
15. **Exit 2 still cancels.** Exit 2 still reconciles and can call `cancel_generation` on a *previous*, inactive generation. `test_abnormal_exit_cancels_recovered_inactive_generation` pins this (`tests/unit/gui/run_console/test_slurm.py:815-838`). A usage error is raised in the CLI's read-only half before anything is submitted, and PB adds more of them (R2, R3, U12 when the probe misses one). Consider classifying 2 as "refused, nothing submitted". The cancel is mostly harmless, because a deactivated fence is never rewritten (`_cli/_cli_slurm_lifecycle.py:700-704`), but it can `scancel` ids left without a terminal ledger row.
16. **The staged phase PA4's hook depends on.** Staged runs with failures reach PA4's observer hook only because the finalizer's phase `"terminal_incomplete"` (`_cli/_cli_checkpoint_handler.py:400`) is absent from `_STAGED_FAILURE_PHASES` (`run_console/_slurm_observer.py:92-94`). Otherwise `_staged_terminal_observation` returns `failed` before the inactive branch (`:642-649`). Pin it with a test.

**Tests**

17. **PA5's TSV bytes test is tautological.** "Bytes equal `write_tsv(parquet frame)`" passes whenever the implementation calls `write_tsv`. Prefer a read-back check: `read_delimited(tsv)` equals the Parquet frame.
18. **PB5's CliRunner test needs a stopping point.** "Up to the first non-usage step" needs an explicit sentinel. For example, monkeypatch `scan_store_outputs` to raise, or measure runs real work over the fixture output.
19. **§6 and R8 conflate two counts.** They say 62 importers of `write_measurements_mirror`. That is the number of files importing `tests/_output_layout.py`. **29** call `write_measurements_mirror`. Running all 62 is fine; the count should say which.

---

## Validated

**PA1: exit 0 never means "submitted" on a path the GUI can build.**
- Every SLURM submission exits `RunExitCode.SUBMITTED`: `phenotypicCLI.py:4177`, `:4194`, and `_handle_recompile_slurm`'s `:4800`.
- The forward path refuses a SLURM strategy that did not submit (`:4159-4163`).
- The remaining `0` exits are the dry-run (`:3482`), the no-op continuations (`:3898`, `:3964`), and the recompile-local fallbacks.
- `--force-local`, the only thing that turns `--slurm` into a local run (`_cli/_cli_types.py:298-302`), cannot be produced by the GUI's argv builder.

**Other confirmed claims:**
- The §3.1 mechanics are right: `ambiguous_error` on any non-zero exit (`_slurm.py:826-831`), `reconciled=ambiguous_error is not None` (`:871`), and the inactive cancel (`:652-656`).
- All 13 reds fail as listed. The log is `/bigdata/exfab/anguy344/gate-results/guispec/reds-32434046.log`: "13 failed, 17 passed, 5 skipped".
- The test edits for #7 and #8 keep their properties.
- PA2's readers already target the record-less table (`sdk_/_measurement_tables.py:2056, 2084`).
- `_GUI_WRITTEN_PROOF_DESCRIPTORS` covers every mirror file curation writes (`sdk_/_run_state.py:1331-1333`).
- PB3's [call] holds. `continuing` excludes measure and requires a processing state (0993c8e06 `phenotypicCLI.py:3242-3248`), and R3/U12 sit under `if config.resume:` (`:3346-3402`).
- The measure-mode refusals in §3.4 are at the cited lines.
- `uses_staged_gpu_strategy` precedes `_resolve_embed_stage` (`:3310-3334`).
- A deactivated fence is never relabelled (`_cli_slurm_lifecycle.py:700-704`), so curation and cancellation cannot erase a finalizer close.
- The ancestry claim holds: #281 is an ancestor of #282, and #282 changes nothing under `_gui/` or in `_exit_codes.py`.
- #283's checklist is fully covered. Run picker, DVC hint and embeddings go to §8; every other item maps to PA1–PA7.

**Sound design choices:**
- PA1 (b) over (a).
- PA4 (a): a new terminal literal.
- PB2 (c): one resolver.
- PB3 (a): static R3/U12, in full mode only.
- PB5 (a): Validate disabled in measure.

---

## Decisions only the user can make

- **Q1 (from C2).** Should a GUI-launched output with failed images be claimable from the GUI, for Run, `--retry-failures`, Validate and measure?
  - Today the claim gate refuses it like a CLI-launched one.
  - Option (a): keep the refusal and document it.
  - Option (b): make `_foreign_run_conflict` honour a terminal GUI owner record with a matching generation.
- **Q2 (from C4).** Accept that every CLI re-run or `dvc repro` of a curated run with failures resets the mirror?
  - The labels survive, and the GUI re-applies them at its next write.
  - The alternative is a CLI change: finalize applies `curation_labels.parquet` to the mirror it seeds.
- **Q3 (from I11).** Update the ucr_029 and ucr_033 plotting scripts to `measurements.tsv` before PR A merges, or accept that they break at the next finalize of a live tree?
- **Q4 (from C3).** Show the U16 measure leftover as `failed` with the "re-run the same command" detail, or add a new terminal status for it?
