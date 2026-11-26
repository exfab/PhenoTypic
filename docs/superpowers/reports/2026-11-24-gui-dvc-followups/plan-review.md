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

---

## Re-review of revision 2 (`9f7f91543`, 2026-11-25)

**Scope:** the plan README's "Revision 2" table, every changed plan file, and the spec diff
`53c36d251..9f7f91543` (D15–D18). I checked it against the code at `36b9e840e` (the worktree's
`src/` is unchanged) and at `0993c8e06`.

### Verdict

**Not ready for C1. One new blocker, introduced by the revision.** All the earlier findings are
resolved on paper, and most of the fixes are good. But the new `excluded_work_ids` call (D15)
reverses a design decision that a test already pins, and it would tell users to remove valid
stores.

Counts:
- **1 critical:** R2-C1.
- **4 important:** R2-I1 to R2-I4.
- **6 minor:** R2-M1 to R2-M6.

The other findings are resolved: C1, C2, I1–I8, I10–I12 and the fixable minors. I9 is resolved
only in part, and R2-I3 says why.

### The original findings

| Finding | Status | Evidence |
|---|---|---|
| C1 All-failed run | **Resolved** (D16). Its rationale is partly wrong: see R2-I2 | PA1 T1 pins a single answer for each case: `== 2`, a stray marker gives `None`, an unresolved image gives `None`. T2 pins exit `3`, T3 pins claimability, and the helper skips the aggregate when nothing verified |
| C2 Mutants and isolation | **Resolved** | Deletion-mutant table. `unresolved_stems=` isolates step 2. `test_a_new_image_that_failed_returns_none` isolates inventory. `test_an_excluded_store_returns_none` isolates the source set (completion stays `failed`), and the inseparable count is named |
| I1 Config source | **Resolved**, but one consumer is missed: R2-I1 | `RunState.finalization_digests` and `.process_only_layer` come from the one config read. The two helpers fit rule 1's two mappings (`_run_state.py:1508-1539`) |
| I2, I3 N plumbing | **Resolved** | `_Observation`, `_apply`, the `compare_and_set` keyword and its reset, and `_rehydrated_status` returning N, persisted by both consumers |
| I4 Run-console picker | **Resolved** | PA4 T1 lists `_form.py:58-73` and both wrappers |
| I5 0/3 consumers | **Resolved** | `_SlurmCompletion.finished`, `:2672`, `shell/_app.py:215, 681`, and a banner assertion |
| I6 Error-analysis early return | **Resolved by deferral** (D18). `_KEPT_CSV_STEMS == {"error_analysis","metadata"}` is consistent across the README, PA3 and spec PA8 | `grep` finds no remaining `error_analysis.tsv`, PA3 T4 or Lane T task in the plan |
| I7 Vacuous pre-PA8 test | **Resolved**, with a mutation check | One note: `inject_descriptor` hand-edits the proof. This is the only way to fake a pre-PA8 proof after T1, so it is acceptable, but say so in the test |
| I8 Lanes | **Resolved** for Lane T (removed). Lane V has a residual risk: R2-M1 | |
| I9 Flip test, guard mutants | **Partly resolved.** The test as written fails for the wrong reason: R2-I3 | |
| I10 Review Focus 4 | **Resolved** | The same record goes through `compare_and_set` and `observe_local_exit(0)` |
| I11 Rebase | **Resolved** | `git rebase --onto origin/main 36b9e840e`, `gh pr view 281`, the push from the root checkout, and the plan committed before C1 |
| I12 R14 | **Resolved** (D17) | R14 corrected; the docs and tutorial say not to run the same output from the CLI and the GUI at once |

### Critical

#### R2-C1. `excluded_work_ids` blames late images on the projection, contradicting a design decision that is already pinned

**Where:**
- PA1 T2 Interfaces: `"excluded_work_ids": sorted(<currently verified work ids> - set(source_work_ids))`, read with `_current_success_work_ids`;
- spec PA4's D15 line;
- `test_an_excluded_store_exits_5_naming_it`.

**Evidence:**
- **The diff measures a gap, not an exclusion.** Under a rolling input, images keep verifying
  between the merge and the proof. `publish_aggregate_snapshot`'s own docstring says this is why
  `source_work_ids` is passed in rather than derived live (`_cli_completion.py:1132-1150`), and the
  fan-out writer says "`planned` is a strict subset of `authorized` with nothing wrong"
  (`_cli_finalize_fanout.py:700-706`). On such a correct run, the D15 line would print "leave out
  X, because the finalizer could not project their stores… Fix or remove those stores".
- **The sdk_ reader made the opposite decision on purpose, and a test pins it.**
  - `_advisories` reports the certified-vs-verified gap with the comment "REPORTS THE GAP, NEVER
    DIAGNOSES IT. The same shortfall is produced by an excluded store and by the wholly ordinary
    case of an image finishing after the master was published" (`sdk_/_run_state.py:1657-1679`).
  - It names a store as excluded **only** from per-image stage facts that prove the store cannot
    be projected: `declares_measurements is True and projectable_measurements is False`
    (`:1680-1695`; recorded at `:965-975`).
  - The test that pins this is `test_a_master_that_merely_predates_an_image_claims_no_exclusion`
    (`tests/unit/sdk_/test_run_state.py:1351`).
- **The remedy is destructive and inconsistent.** "Fix or remove those stores" invites deleting
  valid results, which then costs a full reprocess on the GPU. The existing advisory says "the
  store has to be re-measured".
- **The test fixture models the benign case.** `omit_from_proof=("img-a",)` leaves img-a's store
  valid and projectable, so it builds the rolling-input gap. The test then asserts the
  misdiagnosis.
- **It adds cost.** One more `valid_image_success` pass, with per-artifact hashing, on every
  finalize and every migrate republish (`_hdf_to_zarr.py:800`).

**Fix: drop the marker field.** No CLI write is needed. Build the exit-5 line from what `RunState`
already knows:
- names from the same predicate `_advisories` uses, factored into one sdk_ helper that both
  share;
- otherwise the gap wording, which names no cause.

That keeps D15 (the line names the excluded stores) and removes the misdiagnosis. Then:
- give the fixture a store that declares no projectable table, the existing advisory test's
  pattern (`:1272`);
- add a rolling-input case that must **not** name any store.

If you do keep a marker field, it must be passed in from what the projection actually excluded
(`build_master_frame`'s authorized sources minus `aggregated_sources`), never derived live.

### Important

#### R2-I1. Adding fields to `RunState` breaks a pinned test, and the plan does not say so

**Where:** PA1 T1 Step 4 ("run … `test_run_state.py` … all pass").

**Evidence:**
- `test_the_demoted_sources_live_only_under_diagnostics` asserts `RunState`'s exact field set,
  `{completion, identity, images, advisories, diagnostics, depth, verified_at}`
  (`tests/unit/sdk_/test_run_state.py:56-75`), so Step 4 goes red.
- `docs/source/contrib_guide/tracked_state.md:43-53` documents the same field table.

**Not a problem:**
- No `RunState` is serialized: there is no `asdict`, pickle or key-set check on it.
- The only constructions outside the resolver use keyword arguments and get the defaults
  (`tests/unit/gui/test_snapshot_status.py:71`).

**Fix:** list both in PA1 T1's Files. Change the pinned set deliberately, with a docstring line
saying why these are configuration rather than demoted evidence, so the edit is not taken as
weakening. Add the two rows to the docs table.

#### R2-I2. The no-proof arm's marker check guards the wrong case, and the spec's reasoning for it is wrong

**Where:**
- spec PA4 step 4 ("A marker beside zero verified images means … a restart that cleared success records");
- PA1 T1 Interfaces ("`_valid_aggregate_proof` … also tells the no-proof arm whether a marker exists").

**Evidence:**

1. **`--restart` removes the marker together with the records.** The aggregate marker lives under
   `.phenotypic/`, and `--restart` clears all machine state (`phenotypicCLI.py:1882-1905`,
   `:3493-3500`) but **keeps `deliverables/`**. So the case the spec names cannot happen.
2. **The case that can happen passes the arm.** A `--restart` over a once-complete tree whose
   images now all fail gives "no marker, every image failed": the run is confirmed
   `complete_with_failures` beside the **previous run's** master, mirror, `metrics.json` and
   dashboard. An all-failed finalize writes no deliverable (`_cli_finalize_run.py:883-885`, "No
   valid measurements found"), so nothing replaces them.
   - The viewer cannot open such a tree, because `core_readable` needs the proof.
   - Recent Runs would show "complete · N failed" beside a dashboard link from the previous run.
3. **`_valid_aggregate_proof` cannot tell "absent" from "invalid".** It returns `None` for no
   marker, a wrong version, an empty `required_outputs` and a tampered master alike
   (`_run_state.py:1350-1377`). An implementation that follows the plan's wording confirms an
   all-failed run beside a stale or tampered marker. The stray-marker test uses a *valid* marker,
   so it cannot tell the two implementations apart.
4. **A killed `--retry-failures` relaunch also reads finished.** "A run killed mid-image leaves that
   image unresolved" is false for a retry: the image's earlier failure row is still in the
   append-only journal (`_run_state.py:995-996`). The outcome is benign (the run stays
   claimable), but the spec's claim and N7's "a killed relaunch … is an unconfirmed `failed` run"
   are both untrue when the retry verified nothing.

**Fix:**
- Test existence by path: `aggregate_publication_marker_path(output_dir).exists()`, or
  `_aggregate_proof_state`'s "no readable … marker" reason.
- Add an invalid-version marker case to the stray-marker test.
- Correct the spec's reasoning.
- Decide the restart case, and pin the decision in a test: either the arm also requires no
  master deliverable, or Recent Runs hides the dashboard link for a run confirmed with no proof.
- Correct the killed-retry sentence in PA4 and N7.

#### R2-I3. The PA5 flip test fails for the wrong reason, and it does not isolate the verdict

**Where:** PA5 T1 `test_a_verdict_that_flips_between_bind_and_write_is_refused`.

**Evidence:**
- `publish_image_success_for(out, "img-b")` removes the only failure: a success outranks the
  failure row (`_run_state.py:986-996`). The fresh completion therefore becomes **`incomplete`**
  (there is no run proof), not `failed`.
- The guard's message is `f"… run state is {_completion(fresh_state)}"`
  (`_mutation_guard.py:140-144`), so `match="run state is failed"` never matches, before or after
  the implementation.
- Because completion itself flips, even a predicate that skipped the verdict and checked
  `completion in {"complete", "failed"}` would still refuse, so the test does not show that the
  verdict is asked.
- The same applies to the T1 label on `test_a_killed_retry_continuation_returns_none`. That test is
  killed by step 1, not by the source set. The excluded-store test is the one that really isolates
  the source set.

**Fix:**
- Use three stems: one verified and two failed. Make one failed image succeed; completion stays
  `failed` and only the source set moves.
- Match "run state is failed".
- Add the missing test for "a fresh verdict whose N differs from the bound one refuses" (spec call 14).

#### R2-I4. The exit-5 line needs the `RunState`, which the waiter does not keep

**Where:** PA1 T2: `_unpublished_line(..., state: RunState)`.

**Evidence:**
- `_run_snapshot` keeps only four counts in a `NamedTuple` (`_cli_wait.py:89-130`).
- The reporting path (`_report_outcome`, `:427+`) has no `RunState` to hand.
- So either the snapshot grows the state, or the line re-resolves, which breaks the "one pass"
  rule that `test_the_verdict_comes_from_the_same_pass` pins.

**Fix:** carry what the line needs on `RunSnapshot` (the names from R2-C1's helper), computed in
the same pass.

### Minor

- **R2-M1 (Lane V's real disjointness).** Lane V's source files are disjoint from the main line, and
  its four test files (the README says three) are not in PA3 T2's 56-file sweep. I checked the
  union against `test_measurement_{source,routes,join_migration_run}` and `test_zarr_routes`.
  - The residual risk is helper placement. PA2's tests call `add_second_measuring_run`,
    `add_grid_only_run`, `rewrite_latest_table_with_zero_rows`, `run_from_fixture` and
    `v1_pairs`, none of which exists in a shared module. A module-local `add_grid_only_run`
    exists in `tests/unit/sdk_/test_tables_v2_state_readers.py:29`.
  - The README's convention says "new shared helpers live in `tests/_output_layout.py`", which
    PA1 T1 edits on the main line.
  - Fix: state in PA2 that these helpers stay in its own test modules, and correct the README
    count to four.
- **R2-M2 (fixture names).** The fixtures use `FIXTURE_DATASET = "plate"` and
  `FIXTURE_STEMS = ("a", "b")` (`tests/_output_layout.py:190-191`). The plan's tests use
  `img-a`/`img-b`/`img-c` and expect `"d1/img-a"`. Also, `build_run_with_failures` has no
  `stems` parameter, so how `img-c` enters the inventory is undefined. Define `stems` as
  `FIXTURE_STEMS` ∪ failed ∪ unresolved, and assert the spelling the fixture actually produces.
- **R2-M3 (the SLURM all-failed waiter case).** D16's waiter test is local only
  (`generation is None`). On SLURM, an all-failed finalizer logs and closes "terminal-incomplete"
  instead of raising, when nothing verified (`_cli_checkpoint_handler.py:317-331`). Add one
  generation-bound test that the fence it leaves satisfies `_published`, so exit `3` holds on
  SLURM as D16 says.
- **R2-M4 (one cancelled all-failed run, two answers).** For a cancelled all-failed SLURM run, the
  observer goes through `finished_run_outcome` (not published, so `5`, so `failed`), while
  rehydration goes through the bare verdict (`complete_with_failures`). One GUI restart flips its
  status. Either pin the disagreement deliberately, or have rehydration's no-proof arm require a
  finalizer-closed fence when a fence exists.
- **R2-M5 (the `commit_guard=None` path).** The plan unlinks unguarded when no guard is given. That
  is sensible, but spec PA8 still says the unlink runs "under `commit_guard`" without the "when
  one is given". Align the spec text.
- **R2-M6 (PB3's helper move).** Moving `_refuse_measuring_reference_pipeline` and
  `_reference_operation_paths_for` into `_cli/_cli_reference.py` changes their import path. Check
  for monkeypatch targets before the move; there are none today (`grep` over `tests/`). Leave a
  re-export only if the rebased #282 tree has gained one.

### Verified in this round

- **The marker's byte stability.** `test_deliverables_byte_stable.py` compares `deliverables/`
  only, and its docstring exempts `.phenotypic/` (`:1-6`). The marker already carries
  `published_at`, and no test or reader checks its exact key set (I grepped for `set(marker)`,
  `marker.keys()` and the like across `src/` and `tests/`). So an additive field is safe, should
  R2-C1 keep one.
- **What `--restart` clears.** It clears `.phenotypic/` (journal and marker included) and keeps
  `deliverables/` (`phenotypicCLI.py:3493-3500`). So stale failure rows cannot survive a restart;
  only stale deliverables can (R2-I2).
- **What failure rows are keyed by.** `_terminal_failures` keys rows by `work_id` without an epoch
  (`_run_state.py:1042-1075`). A config or input change moves the work ids, so the spec's "a
  changed configuration … leaves an image unresolved" holds.
- **The rule-1 refactor.** `_identity_digests_match` and `_source_set_matches` fit rule 1
  unchanged: inventory and scientific run before the process branch, and the source set goes
  through `_source_set_binding`. The rule-1 tests `test_each_of_rule_ones_comparisons_is_load_bearing`
  (`test_run_state.py:1034`) and `test_clause_one_is_load_bearing` (`:953`) exist to kill the
  helper mutants, as T1 Step 5 requires.
- **The task count.** 28 tasks: 18 in PR A, 10 in PR B. PA1 6, PA2 1, PA3 3, PA4 2, PA5 1, PA6 3,
  PA7 2; PB0 1, PB1 2, PB2 1, PB3 3, PB4 3.

---

## Re-review of revision 3 (`88a86b669`, 2026-11-25)

**Scope:**
- the plan README's "Revision 3" table;
- the diff `9a31a5361..88a86b669` across the spec, PA1, PA2, PA3, PA5 and PB3.

Checked against the code at `36b9e840e` (worktree `src/`) and `0993c8e06`.

### Verdict

**Ready to execute once one finding is fixed (R3-I1). Nothing is critical.** Every R2 finding is
resolved, and the two calls the team lead asked about hold up, subject to R3-I2 and R3-M1.

The fix for R2-M4 (the finalizer-closed-fence clause) introduces a regression. That clause
should be dropped. If it stays, it must be bound to the fence's generation.

Counts:
- **0 critical.**
- **2 important:** R3-I1 and R3-I2.
- **4 minor:** R3-M1 to R3-M4.

### The R2 findings

| Finding | Status | Evidence |
|---|---|---|
| R2-C1 `excluded_work_ids` | **Resolved** | No marker field remains. `excluded_store_names(images)` is the `_advisories` predicate (`_run_state.py:1680-1695`) unchanged in wording. `late_stems` must name no store, and `unprojectable_stems` drives the store through the real "no column list" shape. Both pinned advisory tests (`:1272`, `:1351`) stay green unedited, because a pure factoring of a predicate cannot change either answer |
| R2-I1 Pinned `RunState` field set | **Resolved** | `test_run_state.py:56-75` and `tracked_state.md:43-53` are listed and edited deliberately |
| R2-I2 No-proof arm | **Resolved**, apart from the fence clause (R3-I1) | The marker is tested by path, with `[wrong_version]` and `[empty_outputs]` killing the `_valid_aggregate_proof` mutant. The other-generation deliverable check holds on every path (below). The killed-retry case is pinned both ways, and N7 is corrected |
| R2-I3 PA5 flip test | **Resolved** | Three stems; completion is asserted to stay `failed`; the N-mismatch test is added with a message |
| R2-I4 `RunSnapshot` | **Resolved** | `excluded_stores` is computed in the same pass, and `_report_outcome` passes it on |
| R2-M1 to R2-M3, R2-M5, R2-M6 | **Resolved** | PA2's helpers stay local. The fixtures are `plate/<stem>`. The SLURM all-failed test exists, and the ordinary finalizer does close the fence with `closed_by=FENCE_CLOSED_BY_FINALIZER` and no `terminal_status` (`_cli_checkpoint_handler.py:491-499`), so `_published` holds and exit `3` is right. The spec says "when one is given". PB3 greps before the move |
| R2-M4 Cancelled all-failed run | **Fixed, but the fix regresses:** R3-I1 | |

### Verified: the deliverables-existence refusal (the team lead's question)

**Claim: an all-failed finalize writes no `master_measurements.parquet` and no
`measurements.parquet`.** This is true on every path, at both bases.

- **The mirror has one writer, and it runs after a non-`None` master.** The only writer of
  `measurements.parquet` is `_seed_measurements` (`_cli_output_manager.py:1002`). Its only
  caller is `finalize_post_master_outputs` (`:1215`), whose only caller is `finalize_run:952`.
  That is after `if master_df is None: return None` (`_cli_finalize_run.py:883-885`). The same
  holds at `0993c8e06`: `git grep` finds the same two call sites.
- **`master_measurements.parquet`** is written by:
  - `finalize_run:914`, which is past the same early return;
  - the chunk writer (`_cli_chunk_writer.py:229-236`), and only on the legacy
    non-`embedded_authority` arm, from chunk rows of successful images. An all-failed run has
    none.
- **Local, SLURM finalizer, measure, recompile and migrate** all reach the master through
  `finalize_run` / `aggregate_measurements`:
  - a SLURM all-failed finalizer gets `aggregate_path is None`, logs, and closes the fence
    (`_cli_checkpoint_handler.py:317-331, 491-499`);
  - migrate's `republish_aggregate` refuses zero successes (`_hdf_to_zarr.py:773-800`).

**Can a legitimate all-failed run have one of these files?** Only as leftovers from another
generation:
- a `--restart` keeps `deliverables/` and clears `.phenotypic/` (`phenotypicCLI.py:3493-3500`);
- a `dvc checkout` mixing commits;
- a hand-copied tree.

Images that once verified cannot later become `failed` without a restart: a success outranks a
failure row (`_run_state.py:986-996`). A deleted record or store makes an image *unresolved*,
which step 2 already refuses. GUI curation cannot write either file into such a run, because
the viewer cannot open a tree with no aggregate proof. **Call 1 is sound as a rule**, but see
R3-I2 for the remedy it tells the user.

### Verified: the `FENCE_CLOSED_BY_FINALIZER` move and its import direction

**Sound.**
- `_cli/_cli_slurm_lifecycle.py` already imports `slurm_lifecycle_path` from sdk_'s
  `_io_constants` (`:26`), so taking the constant from there adds no new direction.
- sdk_ already has `slurm_lifecycle_path` (`sdk_/_io_constants.py:2738`), so the verdict needs
  no `_cli` import.
- The re-export keeps every current importer working: `_cli_wait.py:212, 256`,
  `_cli_staged_orchestration.py:32`, `_cli_checkpoint_handler.py:406, 420`, and
  `_cli_slurm_lifecycle.__all__` (`:1072`).

**Moot if R3-I1 drops the fence clause.** Then the move is not needed at all.

### Verified: moving `authorize_measurements` into `tests/_output_layout.py`

**Safe for the 62 importers.**
- The change is additive. No importer does `from tests._output_layout import *`, and no test
  module defines a colliding `authorize_measurements`, `build_run_with_failures` or
  `excluded_store_names` (I grepped `tests/` and `src/`).
- The helper's imports are function-local, as `_authorize_measurements`' are today
  (`test_run_state.py:531-590`).
- The 5 call sites in `test_run_state.py` (`:669, 670, 685, 686, 1339`) switch to the import.

One caveat, R3-M3: it brings a hand-edit of a store's root `zarr.json` into the module whose
rule is "real publishers only".

### Important

#### R3-I1. The finalizer-closed-fence clause is not tied to a generation, so an earlier fence vetoes a later local run

**Where:** spec PA4 step 4's last bullet; PA1 T1 `test_an_all_failed_run_reads_its_fence`.

**Evidence:**
- The clause reads whatever `slurm_lifecycle_path(output_dir)` holds. A local run never writes or
  closes a fence: "A local run is not refused: it never publishes a fence"
  (`phenotypicCLI.py:1152-1168`). `_retire_dead_fence` closes only an *active* fence (`:1176+`).
- So take an SLURM generation that the user cancelled (fence `active: false`, `closed_by` unset),
  then a local continuation of the same output whose images all fail.
  - The CLI exits `3` (`:3955-3964`).
  - The verdict returns `None` because of the stale cancelled fence.
  - The GUI then reads `failed` and refuses `--retry-failures`, against D10 and D16.
  - The local waiter (`generation is None`, so `_published` is `True`) exits `5`, where D16
    says `3`.
- The same veto comes from a later `--mode measure` SLURM pass that closes its fence with
  `terminal_status: "failed"` (`0993c8e06 _cli_checkpoint_handler.py:414-430`, U16). That
  becomes reachable in PR B.
- The disagreement R2-M4 named (observer `failed` against rehydration `complete_with_failures`
  for a cancelled all-failed SLURM run) is the milder problem. Even there,
  `complete_with_failures` is arguably the true state, because every accepted image did fail.

**Fix (pick one):**
- **(a)** Drop the clause, and pin R2-M4's disagreement with a test and a sentence in spec PA4.
  This is my recommendation: it also makes the constant move unnecessary.
- **(b)** Bind the clause to the generation that produced the failures. It applies only when
  some current failure row's `lifecycle_epoch` equals the fence's `generation`. SLURM workers
  write that field (`_cli_wait.py:269-285` docstring); local rows say `"local"`.
  - This needs the epoch in the verdict's inputs: carry it on `ImageState`, or read the journal
    in the no-proof arm only.
  - Add a "cancelled SLURM generation, then a local all-failed continuation" test expecting `2`.

#### R3-I2. In the stale-deliverables case, the refusal tells the user to do something that does not help

**Where:** spec PA4 step 4 ("re-run from the CLI or with `--overwrite`"); PA1 T3's PA11 message
("re-run it from the CLI or wait for it to finish").

**Evidence:**
- After an all-failed `--restart` beside an old master, these do **not** clear the state:
  - a plain re-run is a no-op: no image verified, so no refresh, and the CLI exits `3`;
  - another `--restart` keeps `deliverables/` again;
  - a `--retry-failures` that verifies nothing.
- Only these do:
  - `--overwrite`;
  - deleting the two Parquet files;
  - a retry that verifies at least one image, which then publishes a proof and switches to the
    proof arm.
- The current claim-gate text sends the user round the no-op loop.
- **Call 1 itself is acceptable.** The trade is that the GUI cannot retry this tree, while the
  CLI's `--retry-failures` or `--overwrite` can.

**Fix:** give this case its own refusal text, decided inside `_foreign_run_conflict` from the
same existence check. For example: "output has N failed images and a master from an earlier run;
re-run with `--retry-failures` from the CLI, or start over with `--overwrite`". Pin it in PA1 T3,
and match it in the tutorial line.

### Minor

- **R3-M1 (call 2's remedy wording).** "Fix those stores by re-measuring them, or remove them from
  the input". Both halves are plausible:
  - `--mode measure` rewrites the table run with a descriptor;
  - a continuation rebuilds `work_ids` from the current input scan (`phenotypicCLI.py:4072-4080`),
    so a removed input leaves the inventory.

  But neither is pinned by a test, and the existing advisory says only "the store has to be
  re-measured" (`_run_state.py:1689-1695`). Name the command (`--mode measure`). Say whether
  the measure pass re-finalizes on its own: it reaches the shared finalize, so "then re-run" may
  be unnecessary. Add one integration assertion, or drop "remove them from the input".
- **R3-M2 (shared-files table).** PA1 T1 now edits `sdk_/_io_constants.py` (the constant), but the
  README shared-files row for `_io_constants.py` (`:238`) lists only PA3 T1, T2 and PA4 T1.
  The main line is sequential, so this is a bookkeeping gap only.
- **R3-M3 (the moved helper's hand-edit).** `authorize_measurements(projectable=False)` pops a
  table descriptor from the store's root `zarr.json` by hand (`test_run_state.py:575-581`).
  `tests/_output_layout.py`'s rule is real publishers, never hand-written JSON (`:448-470`).
  Note the helper there as the one sanctioned mutation, citing
  `test_finalize_run.py::_strip_measurement_descriptor`, the shape it reproduces.
- **R3-M4 (a test that cannot fail).** `test_a_killed_retry_that_verified_nothing_stays_confirmed`
  appends "a retry attempt without result". No writer produces a verdict-relevant row for an
  attempt with no outcome, so the test is equal to the base case and cannot fail. Keep it as
  documentation, but don't count it toward the mutation table.

---

## Confirm of revision 4 (`eb552348b`, 2026-11-25)

This is a narrow confirm of the three points the team lead asked about, not a full re-review.

### Verdict

**R3-I2 and R3-M1 are confirmed. R3-I1 is confirmed for the verdict and the GUI, but one of its
three pins is wrong:** the waiter does not exit `3` in that case. Fix that one test and its
spec sentence; nothing else blocks.

### 1. R3-I1: the fence clause is gone; can any other path in the arm read a stale generation?

**Confirmed for the verdict.** The no-proof arm no longer reads the fence. Its remaining inputs are:
- `state.images` verdicts;
- `state.completion`;
- the marker path;
- the two Parquet paths.

Each of these can be stale only in the **refusing** direction:
- **An earlier generation's marker or Parquet makes the arm return `None`, which is safe.** The
  marker is cleared together with the journal by `--restart` (`phenotypicCLI.py:3493-3500`), so
  "a marker beside zero verified images" arises only from a mixed `dvc checkout` or a copied
  tree.
- **Through step 1, an earlier generation's still-*active* fence makes the run `active`, so the
  arm refuses.**
  - `_live_authority` fences the lifecycle only by `restart_epoch`.
  - A local run does not retire an active fence that has no ledger row (`phenotypicCLI.py:1176-1192`).
  - This predates the change, applies to every verdict rather than this arm, and refuses rather
    than confirms. I list it as a known residual, not a regression.
- **Failure rows from an earlier generation still count.** They are keyed by content-derived
  `work_id` (`_run_state.py:1042-1075`). That is by design and agrees with the CLI's
  `_run_failure_count` (`:1493-1518`).

The pins in T1 (`test_a_cancelled_slurm_generation_then_a_local_all_failed_run_is_confirmed`,
`test_an_all_failed_run_ignores_any_fence`) and T3 (`…_is_claimable`) are right.

**Not confirmed for the waiter.** `test_a_cancelled_slurm_then_local_all_failed_run_exits_3`
(PA1 T2) and spec PA4 step 4's "the waiter exits `3` (`generation is None`, so `_published` is
true)" are both wrong:
- `wait_for_run` binds its generation by reading **whatever fence is on disk at the first poll**
  (`_cli_wait.py:513`; the same code at `0993c8e06:_cli_wait.py:571`).
- A local run writes no fence, so the waiter binds to the earlier, cancelled SLURM generation:
  - `_published` returns `False`: the fence was not closed by its finalizer, and no marker or
    proof carries that epoch;
  - `_cancelled` returns `True`;
  - so the waiter exits **`5`**, "this submission was cancelled before it published".
- This predates PR A, which does not touch how the waiter binds.
- The test passes only if `no_live_jobs_published` (undefined in the plan) stubs `_published`.
  In that case the test is vacuous.

**Fix:**
- Pin the true outcome: exit `5` with the "cancelled" line. The CLI's own exit is `3`. The
  verdict is confirmed, and the GUI is claimable.
- Correct the spec sentence.
- Make sure the waiter tests' helper never stubs `_published` for this case.

Making the waiter ignore a fence that a later local run outlives would be a waiter change.
That is outside both PRs unless the user asks for it.

### 2. R3-I2: is the stale-master remedy true at both bases?

**Confirmed: both named remedies work, and the excluded ones do not.**

- **`--overwrite`** deletes every entry under the output (`phenotypicCLI.py:3317-3318` sets
  `will_overwrite`; `:3520` deletes; the same at `0993c8e06:3325-3330`). That includes
  `deliverables/`.
  - One caveat: it refuses when a run input lives inside the output, for example a `--metadata`
    that points at the run's own snapshot (`:644-731`, `:1009`). The message could add "copy
    any input out of the output first", but that is optional.
- **A CLI `--retry-failures` run that verifies at least one image** is a continuation with work
  left to do (`:3794`), so it finalizes normally.
  - `finalize_run` rewrites the master (`_cli_finalize_run.py:914`) and the mirror (`:952` →
    `_seed_measurements`), then publishes a proof over the verified set (`:1013-1024`).
  - The run therefore moves to the proof arm, with no stale Parquet left.
- **The cases the plan says do nothing** are correct:
  - a plain re-run is a no-op that exits `3` (`:3945-3964`);
  - another `--restart` keeps `deliverables/`;
  - a retry that verifies nothing finalizes nothing (`:883-885`).

### 3. R3-M1: is the `--mode measure` remedy true for an unprojectable store?

**Confirmed, with one caveat.**

- `process_single_store_measure_core` re-measures the store and rewrites the pipeline's own run
  folder with a full descriptor (`replace_image_store_measurements`,
  `_cli_process_single.py:572-584`).
- It then republishes the forward record with `measurements=` set to the **new** table
  (`_republish_table_marker`, `:640-656`).
- The projection reads the run that the record certifies (`_cli_parquet_agg.py:316-320`), so the
  store becomes projectable.
- The measure pass reaches the shared finalize, which republishes the master, the mirror and the
  proof.

**Caveat:** `--mode measure` refuses a pipeline whose measurers read reference metadata, and
that refusal cannot be skipped (`_refuse_measuring_reference_pipeline`, `0993c8e06
phenotypicCLI.py:761-773, 3148-3151`). For such a pipeline the line names a command that cannot
run. Add the clause "(a pipeline that reads reference metadata cannot be re-measured; re-run it
with `--restart`)", or pin the gap in a test.
