# Plan review: GUI follow-ups (#283 PR A, #282 GUI items PR B)

**Reviewed:** `docs/superpowers/plans/2026-11-24-gui-dvc-followups/` (uncommitted; README + pa1–pa7 + pb0–pb4)
against spec `design.md` @ `53c36d251`.
**Code read at:** PR A anchors in worktree `gui-followups` (src identical to `36b9e840e`; `git diff --stat 36b9e840e HEAD` touches docs only);
PR B anchors via `git show 0993c8e06:<path>`.
**Reviewer:** plan-reviewer, 2026-11-24. Analysis only; this file is the sole write.

## Verdict

**Needs revision before C1 is dispatched.** 2 critical, 12 important, 14 minor. Most anchors are right and the
architecture holds up: one sdk_ verdict, exit classification by `RunExitCode`, a separate 0/3 result type,
server-side `gpu_section_shown`, and a static R3/U12 probe. But two things would make tests pass while the code is
wrong or undecided:
- an all-failed run (no verified image) is outside the verdict by construction, and the plan's tests for it accept either answer;
- the verdict's `(state, output_dir)` signature cannot get the config it needs without a second, unsynchronised read.

Several Interfaces also omit plumbing that their own tests depend on.

Measured counts (re-derived here):
- `grep -rln 'measurements\.csv|measurements_csv|MEASUREMENTS_CSV|mirror_csv' tests`: **30** files;
- `grep -rl write_measurements_mirror tests` (excluding the helper): **28**;
- importers of `_output_layout`: **62**;
- the union of the 30 and the 28 is **56** distinct files (PA3 T2's sweep size);
- 29 tasks in total (19 in PR A, 10 in PR B), as the README says.

---

## Critical

### C1. A run where every image failed is decided by accident, and the plan's tests accept either outcome

**Where:**
- README Review Focus 1;
- PA1 T1 `test_a_run_whose_every_image_failed_is_judged_once`;
- PA1 T2 `test_the_waiter_and_the_verdict_agree_on_an_all_failed_run`;
- the helper `build_run_with_failures(failed_stems=("img-a","img-b"))`.

**Evidence:**
- `publish_aggregate_snapshot` raises `RuntimeError("No marker-authorized measurements to publish")` when the
  success count is 0 (`_cli/_cli_completion.py:1179-1181`). An all-failed run therefore **never** has a current
  aggregate proof, and spec PA4 step 3 makes `run_finished_with_failures` return `None` for it every time.
- The CLI exits `3` for such a run (`phenotypicCLI.py:3955-3964`, `_run_failure_count` → `_finished_run_exit`,
  `:1493-1532`).
- Today the waiter also returns `3`: `_published` returns `True` for `generation is None`
  (`_cli/_cli_wait.py:204-205`), and a SLURM finalizer-closed fence satisfies its first disjunct (`:221-228`).
  Rule 3 needs only `failed`, `unresolved == 0` and `published` (`:309-366`).

**Consequences of the plan as written:**
1. The helper, as specified (publish the aggregate over the verified stems), raises at fixture build time.
   Whoever fixes the helper will skip the proof, and both tests then pass.
2. The T1 test asserts `verdict in (2, None)` and that the result matches whether the marker is present. That is a
   restatement of the implementation, so it can never fail. The T2 test compares `finished_run_outcome` with the
   verdict it is computed from, which is the same tautology.
3. **D14 silently changes the outcome of every all-failed run, not just the N3 edge case.** `phenotypic-wait` moves
   from `3` to `5`, and the new `_unpublished_line(published=True)` variant prints "this submission published, but
   its published outputs no longer cover the run". That is false: nothing was published. A DVC stage using
   `--accept-failures` over an all-failed tree goes from passing to failing.
4. **D10 is not met for this case.** `_foreign_run_conflict` still refuses an all-failed CLI run, so the GUI cannot
   launch `--retry-failures` into exactly the run that needs it most.
5. G2 says "a run that exits `3` reads complete, N images failed". For an all-failed local run, the GUI shows
   `failed`.

**Fix:** settle the semantics first (user question Q1 below), then pin a single answer with a test that can fail.

The most coherent option is an explicit arm in the verdict: `verified == 0` and every accepted image failed means
"finished with failures" without a proof, because there is nothing to publish. That keeps the waiter at `3`,
makes the run claimable for `--retry-failures`, and leaves nothing to curate (the viewer cannot open it anyway).

Whatever the decision:
- the helper must handle zero verified images explicitly;
- the exit-5 variant line must only print when the generation actually published an aggregate proof, never on
  `generation is None` alone.

### C2. The spec's mutation check would let the resolution and inventory comparisons be deleted undetected

**Where:** PA1 T1 Step 5 ("invert each of the four comparisons…, the resolution check and the process check"), and
the tests meant to isolate them.

**Evidence:**
- **Inverting** a comparison (`==` → `!=`) is killed by `test_a_finished_run_with_failures_returns_n` alone. The
  base fixture is equal on every field, so the step "passes" without showing that each comparison is pinned. The
  mutant that matters is **deleting** a conjunct.
- `test_an_unresolved_image_returns_none` (drop img-a's record) also breaks the source set. The proof covers
  {img-a} but no image is verified now. So with the resolution check deleted, the source-set comparison still
  returns `None` and the mutant survives.
- `test_a_new_input_image_returns_none` (`add_accepted_work_id("img-c")`) creates an **unresolved** image, so the
  resolution check returns `None` before `inventory_digest` is compared. A deleted inventory comparison survives.
- The real-world case for "unresolved with a proof" is exactly P2 review I1/F5: an array task killed before writing
  its record while the `afterany` finalizer still publishes over the verified set. That is what rule 3's
  `unresolved == 0` exists for, and no test builds it.
- The plan says "a mutant that survives stops the cluster", so C1's cluster would stop on these. Better to fix the
  tests now.

**Fix:**
- Use deletion mutants, one per conjunct.
- Give the helper an `unresolved_stems=` parameter: an accepted stem with no record, published over the verified
  set. That isolates the resolution check.
- Add a "new image that failed terminally" case: inventory moves, every image resolved, the proof is older. That
  isolates `inventory_digest`.
- `source_image_count` cannot be isolated from `source_set_digest`. Say so in the mutation table rather than claim
  it is covered.

---

## Important

### I1. `run_finished_with_failures(state, output_dir)` has no source for `config`, and the shared helper does not fit rule 1

**Where:** PA1 T1 Interfaces (`_proof_binding_matches(proof, *, identity, config, verified)`).

**Evidence:**
- `RunState` carries no config: its fields are `completion`, `identity`, `images`, `advisories`, `diagnostics`,
  `depth` and `verified_at` (`sdk_/_state_types.py:157-166`). But step 4 needs `accepted_finalization_digests(config)`:
  - the aggregate proof writes the **unversioned** finalization digest (`_cli_completion.py:1249-1256`);
  - `identity.finalization_input_digest` is the **versioned** one (`_run_state.py:325-327`, `_finalization_inputs` `:183-196`);
  - so the identity alone cannot be compared.
- Step 5 needs `config["process_only_layer"]`, which the identity does not carry either.
- The verdict therefore has to re-read `processing_state.json` (`_read_state_config`, `:134`) at a different
  instant from the caller's state. That is the torn read N1 was written to remove, and
  `test_the_verdict_never_resolves_state` passes regardless.
- Rule 1 cannot simply call the helper:
  - it compares inventory and scientific digests **before** its process branch (`:1508-1520`);
  - it takes the source set from `_source_set_binding(output_dir, proof)` (`:1528`), which for a pre-P4 run proof
    is the *aggregate*, not `proof`.
  - A single `proof` parameter cannot serve both arms.

**Fix:** choose one of these and write it into Interfaces:
- (a) add the config, or a frozen `finalization_digests: frozenset[str]` plus `process_only_layer`, to `RunState`,
  filled in `resolve_run_state`'s own single read;
- (b) have the verdict re-read the config and return `None` unless `_identity_from(output_dir, config).digest() ==
  state.identity.digest()`, with a test that rewrites the config between resolve and verdict.

Then split the helper so rule 1 can share it: (inventory, scientific) and (finalization, source set given a binding
mapping).

Separately, step 2 should be stated over `state.images` verdicts, not `diagnostics`:
- `RunDiagnostics` says "Nothing branches on these" (`_state_types.py:138-150`);
- `_output_holds_a_run` calls a predicate that reaches into `diagnostics` "visibly wrong in review"
  (`shell/_runs_registry.py:214-217`).

### I2. PA1 T5 cannot persist `failed_images`: the observer and the CAS have no slot for it

**Where:** PA1 T5 Interfaces ("`_Observation("complete_with_failures", …)` with `failed_images=N`"), and PA1 T3's
file list.

**Evidence:**
- `_Observation` has no such field (`run_console/_slurm_observer.py:327-335`).
- `_apply` compares and CASes a fixed 6-tuple (`:861-896`).
- `RunRegistry.compare_and_set` has no `failed_images` keyword (`shell/_runs_registry.py:502-518`).

**Fix:**
- Add all three to T3/T5's **Files**.
- Make `compare_and_set` reset `failed_images` to `None` on any non-`complete_with_failures` status, so a later
  `complete` cannot carry a stale N.

### I3. Rehydration and dead-owner release lose N

**Where:** PA1 T3 (`_rehydrated_status`), and the test `test_rehydration_reads_a_cli_run_with_failures`, which
checks `[0]` only.

**Evidence:**
- `_rehydrated_status` returns `(status, detail)` (`shell/_runs_registry.py:1026-1103`).
- `rehydrate_from_sandbox` builds records from it.
- `_release_dead_owner_locked` persists its status (`:1383-1397`).
- Both would write `complete_with_failures` with `failed_images=None`, which `status_label` renders as "complete ·
  None failed".

**Fix:**
- Return N from `_rehydrated_status`, or have it return the `RunState` and N.
- Thread N through both consumers.
- Assert `failed_images == 1` in the rehydration test, and add a dead-owner-release test.

### I4. The run console's own pipeline picker still lists tune specs as pipelines (G6 gap, in both the spec and the plan)

**Where:** PA4 T1 (Files list only the builder callers, `builder/_callbacks.py:7028` and `_modal_browser.py:140`).

**Evidence:**
- `run_console/_form.py:67-72` (`render_pipeline_tree`) calls `directory_tree(..., extensions=PIPELINE_EXTS)`.
- `PIPELINE_EXTS = PIPELINE_CONFIG_SUFFIXES` (`builder/_directory_browser.py:32`) contains `.json` (`sdk_/_io_constants.py:135-141`).
- So `x.pht-tune.json` is selectable as the Run pipeline. That is a mounted surface.
- The two builder call sites also go through wrappers with fixed keyword lists, `_render_tree_body`
  (`builder/_callbacks.py:4019-4043`) and `_initial_tree_body` (`builder/_modal_browser.py:189-200`). Both need the
  new `predicate` keyword threaded through.

**Fix:**
- Add `_form.py:67` and both wrappers to Files.
- Add a run-console picker test beside `test_the_pipeline_browser_hides_a_tune_spec`.

### I5. The 0/3 result has more consumers than the three the spec names

**Where:** PA1 T4.

**Evidence:**
- `_SlurmCompletion.result: SlurmSubmitResult | None` (`run_console/_callbacks.py:849-855`) is the presentation
  event, and `result = completion.result` at `:2672` reads it for the banner.
- The plan never says what the browser shows for a finished-without-submission completion.
- The annotations at `shell/_app.py:215` and `:681` (`slurm_submitter: Callable[..., SlurmSubmitResult]`) are two
  further seam sites beyond `_app.py:74` and `_callbacks.py:1310`.

**Fix:**
- Define the `_SlurmCompletion` variant and the banner text ("nothing needed SLURM; the run finished in this
  process").
- Add both shell annotations to Files.
- Add an integration assertion on the rendered banner.

### I6. The error-analysis v2 publish is a no-op on a v1 tree: the idempotency check returns early

**Where:** PA3 T4 (Files lists `:471, 490-495, 638-644, 662-676, 710-717`).

**Evidence:**
- `publish_error_analysis` returns `already_published=True` when the receipt's `generation` equals
  `computation.generation` and `_validate_canonical_generation` passes (`_error_tab/_publication.py:331-337`).
- The generation is `_generation_digest`, whose payload pins `"schema_version": 1` and does not depend on artifact
  format (`:799-827`).
- Once validators accept v1, which the plan requires, a v1 tree whose labels have not changed is "already current".
  It is never rewritten as v2 and its CSV is never retired, so `test_a_v2_publish_retires_the_live_csv` fails.
- The journal writer `"schema_version": 1` at `:369` is not in the list either.

**Fix:** add `_generation_digest` (bump its `schema_version`) and `:369` to Files, or make the early return require a
v2 receipt. Keep the test, which is a good one.

### I7. `test_a_pre_pa8_proof_still_validates_after_the_csv_goes` is vacuous after T1

**Where:** PA3 T1.

**Evidence:**
- The test uses `build_complete_run`, whose proof comes from `publish_aggregate_snapshot`.
- After T1 removes `measurements_csv` from `required_paths` (`_cli_completion.py:1205-1210`), that proof no longer
  records `measurements_csv`. The comment "its proof records measurements_csv" is false, and the test passes
  whether or not `_GUI_WRITTEN_PROOF_DESCRIPTORS` still lists it.

**Fix:**
- Build the pre-PA8 proof explicitly: inject a `measurements_csv` descriptor into the marker with the real bytes'
  size and sha, or use a fixture tree written before T1.
- Then delete the CSV and assert `aggregate_proof_is_current`.
- Mutation: drop `measurements_csv` from `_GUI_WRITTEN_PROOF_DESCRIPTORS`; this test must fail.

### I8. Lanes V and T are not file-disjoint from the main line

**Where:** README "Parallel lanes".

**Evidence:** the README's own shared-files table lists `tests/_output_layout.py` for both PA1 T1 (the new helper)
and PA3 T2 (the default flip). Lane T's sweep also touches:
- `tests/unit/gui/test_snapshot_status.py`, which PA1 T6 edits on the main line;
- `tests/unit/sdk_/test_io_constants.py`, PA4 T1's test file.

PA3 T1 edits `sdk_/_io_constants.py` (docstrings), which PA4 T1 also edits. PA4 T1 waits only for PA1 T4, not for
Lane T's merge.

There is also a semantic coupling. On the main line before Lane T merges, `publish_aggregate_snapshot` still
`resolve(strict=True)`s `measurements.csv`. So PA1 T1's helper must write the CSV, which the default does. A helper
written with `csv=False` breaks there.

**Fix:**
- Either run PA3 T1–T2 after PA1 T1 and T6, or accept the merges and name the three overlapping files with the
  expected conflict.
- Make PA4 T1 also wait for Lane T's merge.

### I9. PA5's "verdict flips between bind and write" test probably passes for the wrong reason, and no mutation check covers the guard's three points

**Where:** PA5 T1 `test_a_verdict_that_flips_between_bind_and_write_is_refused`.

**Evidence:**
- `repair_failed_image` ("a retry verified img-b and re-finalized") rewrites the master and the processing products.
  `authorize` refuses that through the processing-snapshot currency check (`_mutation_guard.py:155-159`), or reads
  `complete` and allows it.
- Either way the test does not show that `run_permits_mutation` is asked on the fresh and verified states. The
  README's mutation-check list covers only the verdict, the waiter and `submit_slurm`.

**Fix:**
- Flip only the verdict: add a success record for img-b without re-finalizing, which changes the source set but not
  the identity digest.
- Assert that the refusal message names the completion.
- Add deletion mutants for each of the three `run_permits_mutation` calls in `authorize` (`:121, 140, 170`), for the
  PA11 condition in `_foreign_run_conflict`, and for PA3 T1's "unlink only after the TSV succeeded".

### I10. The Review Focus 4 test cannot fail

**Where:** PA1 T3 `test_a_retry_that_fixes_every_image_clears_failed_images`.

**Evidence:** the relaunch goes through `allocate`, which constructs a fresh `RunRecord` whose `failed_images`
defaults to `None` (`shell/_runs_registry.py:387-397`), so the assertion holds by construction.

**Fix:** the real hazard is an N that survives a status change on one record. Test it on the same record: through
`compare_and_set`/the observer (`complete_with_failures` → `complete`) and through `observe_local_exit(0)`. See I2.

### I11. The D9 rebase step (PA7) is fragile, as written, to squash merges and to #281 moving on

**Where:** PA7 T1 Steps 1–2.

**Evidence:**
- `git merge-base --is-ancestor origin/feat/dvc-embeddings-refactor origin/main` fails after a squash merge.
- A plain `git rebase origin/main` replays #281's own commits if #281 reaches `main` other than by merge commit.
  This repo currently merges with merge commits (`git log origin/main --first-parent`), but that is a convention,
  not a guarantee.
- `main` has already moved: #284 merged on 2026-11-24 and touches `phenotypicCLI.py` and `_cli_failure_tracker.py`
  (`gh pr view 284`). So anchors in both will shift.
- The plan files themselves are uncommitted. `git status` shows `?? docs/superpowers/plans/2026-11-24-gui-dvc-followups/`.

**Fix:**
- Use `git rebase --onto origin/main <recorded-#281-base> feat/gui-dvc-followups`, with the base being the #281 SHA
  PR A was last rebased on (`36b9e840e` today; update it if #281 gains commits).
- Confirm the merge with `gh pr view 281 --json state,mergeCommit`.
- Commit the plan before C1.
- The push has to come from the root checkout. A worktree session's `GIT_SSH_COMMAND` push is refused, so leave
  the worktree with `ExitWorktree(keep)` and push from the root. State that in Step 3.

### I12. The spec's R14 is wrong: D10 opens a concurrent-writer window against a local CLI run

**Where:** spec R14, which the plan inherits (PA1 T3).

**Evidence:**
- `_live_authority` recognises only the SLURM fence and a GUI owner with a live pid
  (`sdk_/_run_state.py:1078-1140`).
- A non-GUI **local** CLI process holds no liveness authority, and `phenotypicCLI.py` has no output-wide local
  exclusivity lock. Its only `exclusive_path_lock` is the metadata-snapshot lock at `:1255-1268`.
- A confirmed `failed` run stays confirmed while a CLI user re-runs it locally, and that re-run always re-finalizes
  (spec §3.1). So after D10 the GUI can claim and launch into it concurrently.
- Before D10 the gate refused every `failed` output. The same window already existed for `complete` outputs, but
  their re-run is a no-op.

**Fix:** this is the user's call (Q3). At minimum, correct R14's text. A cheap mitigation: the claim gate refuses
when `processing_state.json`'s mtime is within the last N seconds, or the CLI writes a pid-liveness record for local
runs (a CLI change, outside both PRs).

---

## Minor

1. **PA1 T1, the excluded-store fixture.** `omit_from_proof` omits img-a from the proof but not from the master or
   mirror. A real exclusion drops the store's rows from the master too (`project_embedded_measurement_table`). Make
   them match.
2. **PA1 T1, `clear_terminal_failure`.** The journal is append-only, and nothing in the code clears a row. A success
   record outranks a failure (`_run_state.py:986-996`), so the helper is unnecessary. Writing it as a hand edit of
   the journal breaks the "real writers" rule.
3. **PA1 T1, the process fixture.** `build_run_with_failures(process_only_layer=…)` must skip the master, the mirror
   and the aggregate, as `build_complete_run` does (`tests/_output_layout.py:503-517`). As specified, the helper
   publishes an aggregate unconditionally.
4. **PA1 T4, `jobs="durable"` with an inactive fence on exit 4.** The plan returns `read_submitted_job_set(output_dir)`
   with no `expected_generation`. Assert which generation the result names, so a stale binding through this arm
   cannot pass the test that only counts `cancel_generation` calls.
5. **PA1 T4, call counts.** The spec's "every mock-based test asserts the `cancel_generation` mock's call count" is
   not met: `test_exit_4_with_a_durable_job_set_is_not_reconciled`, `test_no_stale_binding_…` and
   `test_exit_2_never_reconciles` do not assert it.
6. **PA1 T3, render coverage.** Spec R2's "A Recent Runs render test covers each `RunStatus`" has no task.
7. **PA3 T1, the guard when there is none.** The "inside `commit_guard`" test is vacuous when the finalize path
   passes `commit_guard=None`, which `_guarded_terminal_call` allows (`_cli_output_manager.py:95-115`). Pin that
   path explicitly, or assert that the guarded call is used.
8. **PA3 T2, the sweep's size.** The sweep is **56** distinct files (the union of the 30 and the 28). The README and
   Task 2 read as if they were separate sets. Also, `run_is_complete` (PA5's rename) is referenced in tests at
   `tests/gui/results_viewer/test_mutation_guard.py:259` and `test_output_discovery_contracts.py:228, 250, 298`,
   which PA5's Files omit. Its "10 sites" grep is over `src/` only and finds 11 lines.
9. **PA3 T3, Review Focus 2.** The test asserts only `schema_columns == parquet_columns`. In an old-GUI-curated tree
   the uncurated TSV has the same columns, so the assertion cannot see the hazard. `_schema_cache` reads TSV-first
   after T3. Assert row counts or keys against the Parquet for any reader of the text mirror. Also,
   `CurationLabels.measurements_csv` (`_curation_labels.py:284`) and `_filtered_state.py:255` are further CSV-path
   uses outside the listed lines.
10. **PA4 T2, `MergeMetadata`'s default.** It is a static module dict (`analysis/_callbacks.py:90-96`). Resolving it
    per output means computing it at callback time with the bound output. Say where.
11. **PB1, the `RunMode` annotation.** `mode: RunMode` collides by name with the registry's `RunMode`
    (`shell/_runs_registry.py:88`). The intended type is `_cli_preflight.RunMode` (`0993c8e06
    _cli_embed_stage.py:74`). `notes` is a `list[str]` today (`:390-468`), not a tuple.
12. **PB1–PB3, test API names.** `RunConsoleState.from_json` does not exist: the functions are
    `run_state_from_json` / `run_state_to_json` (`0993c8e06 run_console/_state.py:121, 151`). `obs.is_tracking` does
    not exist either; `tracked_generation_counts` does (`_slurm_observer.py:409`).
13. **PB3 T1, `measure_reference_refusal`.** It goes in `phenotypicCLI.py`, which is missing from PB3 T1's Files. It
    needs `_reference_operation_paths_for` (`0993c8e06 phenotypicCLI.py:3148`). Several further points:
    - Importing the 4800-line CLI module into a run-console callback is costly. Put the wrapper in a `_cli` module.
    - The new GUI imports of `_cli_embed_stage` and the CLI should get rows in `_gui/CLAUDE.md`'s "Importing readers"
      survivor table. PB4 does not list that edit.
    - The README shared-files table lists `phenotypicCLI.py` under PB1 T1, but PB1 T1's Files do not include it.
14. **PB3, measure-mode details.**
    - T2's "`1` → failed with the CLI's last stderr line": `observe_local_exit(run_id, generation, returncode)` has
      no access to stderr. Name the log reader.
    - T1 lacks the spec's "both `metadata.csv` and `metadata.tsv`" measure refusal test and the "empty output" case.
    - The CliRunner fixture must hold no lifecycle fence: `_refuse_live_slurm_run` runs for measure too
      (`0993c8e06 phenotypicCLI.py:3343`) and would query the scheduler.

---

## Validated (credit where due)

- PA1 T4's classifier is right, as is the separate `SlurmFinishedWithoutSubmission` type.
  `SlurmSubmitPending ⊂ SlurmSubmitError` (`_slurm.py:75-79`), so the N2 test's `not isinstance(…, SlurmSubmitPending)`
  is meaningful. `_SubmissionReconciliation`'s field order matches the plan's constructor (`:167-174`).
- The verdict's comparisons are available on the aggregate proof as the spec says: `inventory_digest`,
  `scientific_config_digest`, `finalization_input_digest`, `source_set_digest` and `source_image_count`
  (`_cli_completion.py:1244-1260`). Only the config access (I1) is missing.
- `decide_wait_outcome` stays pure. Rule 4 is kept by `(published and fwf is not None) or (process_run and not
  cancelled)`, which the plan's `test_a_process_run_keeps_rule_four` pins.
- The vocabulary guards (`test_runs_registry.py:1268`, plus the terminal-set guard that requires ≥3 copies) will
  catch a missed terminal set. All three copies are named (`_runs_registry.py:133`, `_callbacks.py:862`,
  `_slurm_observer.py:62`).
- The red #7 expectation matches the gate log, and the reds #8–13 route through `declared_measurement_table_path`
  as the spec says.
- PB1's `gpu_section_shown`, set server-side at `_state_from_action_controls`, covers Validate too, because
  `_validate_argv_for` reuses `_build_subprocess_argv` (`0993c8e06 _callbacks.py:392-408`). That is a good
  catch-all for I10.
- The PB2/PB3 refusal anchors exist at `0993c8e06` as cited: `refuse_unstageable_embedders` `:190`,
  `refuse_embed_stage_switch` `:229`, `adopt_recorded_embed_stage` `:284`, `load_pipeline_or_none` `:483`.
  `scan_store_outputs` is a module-level import (`phenotypicCLI.py:171`, called at `:3518`), so the sentinel
  monkeypatch works.
- PA3 T1's ordering design (a TSV success sentinel out of `_guarded_terminal_best_effort`, then the guarded unlink)
  is feasible against `_cli_output_manager.py:99-115, 968-1003`.

## Unnecessary complexity (candidates to cut)

- **PA3 T4 (error-analysis v2).** The Error tab is unmounted (`_gui/CLAUDE.md`). The only live writer is the CLI
  re-emit. A schema bump with dual-version validators, rollback compatibility, generation-digest changes (I6) and
  CSV retirement is the largest single risk in PA3, for a file few users open. Deferring it (keeping
  `error_analysis` in `_KEPT_CSV_STEMS`) is a spec change, so it is listed as Q4.
- **PA5's `run_is_complete` → `run_permits_mutation` rename.** Changing the property's body, docstring and CAN-17
  note without renaming would avoid the test churn (Minor 8). The spec asks for the rename; this is optional.
- **`TUNING_SPEC_TYPED_SUFFIXES`** (PA4 T1) duplicates `TUNING_CONFIG_SUFFIXES - {".json"}`. Derive it from the sdk_
  set rather than spelling it a second time.

## Questions only the user can decide

1. **Q1 (from C1). What is an all-failed run?** There are no verified images and no aggregate proof is possible.
   Options:
   - (a) `complete_with_failures`, through an explicit no-proof arm. The waiter stays at `3`, the run is claimable
     for `--retry-failures` from the GUI, and Recent Runs reads "complete · N failed";
   - (b) `failed`, and the waiter moves to `5` as D14 now implies, with an accurate exit-5 line ("nothing was
     published: every image failed"). Under (b), D10 does not cover this case.
2. **Q2 (the plan's finding 1, extended).** Should `--accept-failures` also cover D14's new `5`, for the
   excluded-store case and, under Q1(b), the all-failed case?
3. **Q3 (from I12).** D10 lets the GUI claim a confirmed run that a local CLI process may be re-running, because
   nothing records local CLI liveness. Options: accept it and document it (correcting R14), or add a mitigation.
4. **Q4 (simplicity).** Keep the error-analysis TSV v2 in PR A, or defer it to the follow-up issue while the Error
   tab stays unmounted?
