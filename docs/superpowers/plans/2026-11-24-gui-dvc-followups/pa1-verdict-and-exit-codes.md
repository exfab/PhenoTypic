# PA1 · The verdict, the waiter, exit codes and the new status

**Spec:** §3.1, §3.2, PA1, PA4, PA11; D10, D13–D16; re-review N1–N4, N7; plan review C1, C2, I1–I5, I10

---

**Goal:** One sdk_ verdict decides "finished with failures". The waiter, the registry, the
observer, the snapshot badge and the claim gate use it; `submit_slurm` classifies the
submitter's exit by `RunExitCode`; `complete_with_failures` is a terminal status.

**Depends on:** nothing. Lane V (PA2) runs beside it; PA3 follows it on the main line.

### Task 1: `run_finished_with_failures(state, output_dir)` (C1)

**Files:**
- Modify: `src/phenotypic/sdk_/_state_types.py:155-166`: `RunState` gains
  `finalization_digests: frozenset[str] = frozenset()` and `process_only_layer: str | None = None`.
- Modify: `src/phenotypic/sdk_/_run_state.py`:
  - `resolve_run_state` (`:1848+`) fills both from its one config read (`_read_state_config`,
    `:134`; `accepted_finalization_digests`, `:1436`);
  - add the verdict and the two shared helpers beside `_run_proof_covers_current_inventory`
    (`:1464-1541`), which calls them;
  - factor the exclusion predicate out of `_advisories` (`:1680-1695`) into
    `excluded_store_names`, which `_advisories` then calls (R2-C1).
- Modify: `src/phenotypic/sdk_/_io_constants.py`: add `FENCE_CLOSED_BY_FINALIZER = "finalizer"`;
  `src/phenotypic/_cli/_cli_slurm_lifecycle.py:47` imports it from sdk_ instead of defining it.
- Modify: `src/phenotypic/sdk_/__init__.py`: export `run_finished_with_failures`,
  `excluded_store_names`, `FENCE_CLOSED_BY_FINALIZER` beside `aggregate_proof_is_current`
  (`:455, 645`).
- Modify (pinned, deliberately; R2-I1): `tests/unit/sdk_/test_run_state.py:56-75`
  (`test_the_demoted_sources_live_only_under_diagnostics`: the field set gains the two names,
  and its docstring says they are configuration the resolver already read, not demoted
  evidence); `docs/source/contrib_guide/tracked_state.md:43-53` (two rows).
- Create (helpers), in `tests/_output_layout.py`:
  - `authorize_measurements(root: Path, stem: str, *, projectable: bool = True) -> None`:
    moved from `tests/unit/sdk_/test_run_state.py`'s module-local `_authorize_measurements`
    (used by `_build_excluded_store_run`, `:654-690`), which then imports it.
  - `build_run_with_failures(tmp_path: Path, *, failed_stems: Sequence[str] = ("b",), unresolved_stems: Sequence[str] = (), unprojectable_stems: Sequence[str] = (), late_stems: Sequence[str] = (), process_only_layer: str | None = None) -> Path`.
    - **Stems (R2-M2):** the inventory is `FIXTURE_STEMS` (`("a", "b")`, `:191`) ∪ every
      named stem, in one dataset, `FIXTURE_DATASET` (`"plate"`, `:190`). A store is named
      `plate/<stem>`.
    - `write_processing_state` (`:288`) over every stem; `append_terminal_failure`
      (`_cli/_cli_failure_tracker.py:562`) for each failed stem; nothing for an unresolved stem.
    - Every other stem is **verified** (`publish_image_success`, `_cli_completion.py:217`).
      An **unprojectable** stem is verified through `authorize_measurements(projectable=False)`:
      its record authorizes a table its store declares no column list for, as a real exclusion
      leaves it (R2-C1). A **late** stem is verified and projectable, but published after the
      proof: the rolling-input gap.
    - With `process_only_layer`: no master, mirror or aggregate (as `build_complete_run`,
      `:503-517`). With no verified stem: no aggregate (`_cli_completion.py:1179-1181`).
    - Otherwise the master, the mirror and `publish_aggregate_snapshot(source_work_ids=...)`
      over the verified stems **minus** unprojectable and late stems.
    - It uses `write_measurements_mirror`'s default, so it works before and after PA3 T2.
- Test: `tests/unit/sdk_/test_run_finished_with_failures.py` (new, shared env)

**Interfaces:**
- Consumes: `resolve_run_state(output_dir, depth=...) -> RunState`; `_valid_aggregate_proof`
  (`:1397`); `_source_set_binding` (`:1542`); `aggregate_publication_marker_path`,
  `master_measurements_parquet_path`, `measurements_parquet_path`, `slurm_lifecycle_path`;
  `canonical_digest` (`sdk_/_digests.py:46`); `RunState.images[work_id].verdict`, `.identity`.
- Produces:
  - `run_finished_with_failures(state: RunState, output_dir: Path) -> int | None`: spec PA4
    steps 1–5. Its disk reads are the aggregate proof, and, in the no-proof arm only, three
    existence checks (the marker path, the two Parquet deliverables) and the lifecycle file. It
    never reads `processing_state.json`.
  - `_identity_digests_match(proof: Mapping[str, object], identity: RunIdentity) -> bool`;
    `_source_set_matches(binding: Mapping[str, object], verified: Sequence[str]) -> bool`.
  - `excluded_store_names(images: Mapping[str, ImageState]) -> tuple[str, ...]`: sorted
    `"<dataset>/<image_stem>"` of verified images with `declares_measurements is True` and
    `projectable_measurements is False`. `_advisories`' wording is unchanged.
  - `RunState.finalization_digests`, `RunState.process_only_layer`.

- [ ] **Step 1: Write the failing tests.**

```python
def _state(out): return resolve_run_state(out, depth="deep")

def test_a_finished_run_with_failures_returns_n(tmp_path):
    out = build_run_with_failures(tmp_path)                       # a verified, b failed
    assert run_finished_with_failures(_state(out), out) == 1

def test_a_live_run_returns_none(tmp_path):                       # step 1
    out = build_run_with_failures(tmp_path); hold_active_fence(out)
    assert _state(out).completion == "active" and run_finished_with_failures(_state(out), out) is None

def test_an_unresolved_image_beside_a_current_proof_returns_none(tmp_path):     # step 2
    out = build_run_with_failures(tmp_path, unresolved_stems=("c",))
    assert run_finished_with_failures(_state(out), out) is None

def test_no_aggregate_proof_with_a_verified_image_returns_none(tmp_path):
    out = build_run_with_failures(tmp_path); aggregate_publication_marker_path(out).unlink()
    assert run_finished_with_failures(_state(out), out) is None

def test_a_retry_that_verified_one_image_returns_none(tmp_path):  # source set (R2-I3 label)
    out = build_run_with_failures(tmp_path, failed_stems=("b", "c"))
    publish_image_success_for(out, "b")                           # c still failed
    assert _state(out).completion == "failed"                     # so step 1 does not kill it
    assert run_finished_with_failures(_state(out), out) is None

def test_a_killed_retry_that_verified_nothing_stays_confirmed(tmp_path):   # R2-I2, N7
    out = build_run_with_failures(tmp_path)
    append_retry_attempt_without_result(out, "b")                 # the journal row stays
    assert run_finished_with_failures(_state(out), out) == 1

def test_a_killed_metadata_change_returns_none(tmp_path):         # finalization digest
    out = build_run_with_failures(tmp_path); bump_metadata_snapshot_digest(out)   # :769
    assert run_finished_with_failures(_state(out), out) is None

def test_an_edited_pipeline_returns_none(tmp_path):               # scientific digest
    out = build_run_with_failures(tmp_path); bump_scientific_config_digest(out)   # :360
    assert run_finished_with_failures(_state(out), out) is None

def test_a_new_image_that_failed_returns_none(tmp_path):          # inventory digest
    out = build_run_with_failures(tmp_path); accept_and_fail(out, "c")
    assert run_finished_with_failures(_state(out), out) is None

def test_an_unprojectable_store_returns_none_and_is_named(tmp_path):      # R2-C1
    out = build_run_with_failures(tmp_path, unprojectable_stems=("c",))
    state = _state(out)
    assert run_finished_with_failures(state, out) is None
    assert excluded_store_names(state.images) == ("plate/c",)

def test_a_late_image_returns_none_and_names_no_store(tmp_path):         # rolling input
    out = build_run_with_failures(tmp_path, late_stems=("c",))
    state = _state(out)
    assert run_finished_with_failures(state, out) is None
    assert excluded_store_names(state.images) == ()

def test_a_process_run_returns_none(tmp_path):                    # N4
    out = build_run_with_failures(tmp_path, process_only_layer="gray")
    assert run_finished_with_failures(_state(out), out) is None

def test_a_curated_mirror_does_not_move_the_verdict(tmp_path):
    out = build_run_with_failures(tmp_path); rewrite_mirror_rows(out)
    assert run_finished_with_failures(_state(out), out) == 1

def test_an_all_failed_run_is_finished_with_failures(tmp_path):   # D16, Review Focus 1
    out = build_run_with_failures(tmp_path, failed_stems=("a", "b"))
    assert run_finished_with_failures(_state(out), out) == 2

@pytest.mark.parametrize("marker", ["valid", "wrong_version", "empty_outputs"])    # R2-I2: by path
def test_an_all_failed_run_beside_any_marker_returns_none(tmp_path, marker):
    out = build_run_with_failures(tmp_path, failed_stems=("a", "b"))
    write_marker(out, kind=marker)                                # _valid_aggregate_proof is None for two of these
    assert run_finished_with_failures(_state(out), out) is None

def test_an_all_failed_restart_beside_an_old_master_returns_none(tmp_path):    # R2-I2: another generation
    out = build_run_with_failures(tmp_path)                       # a once-published tree
    restart_and_fail_everything(out)                              # clears .phenotypic/, keeps deliverables/
    assert master_measurements_parquet_path(out).exists()
    assert run_finished_with_failures(_state(out), out) is None

def test_an_all_failed_restart_with_no_deliverables_is_confirmed(tmp_path):
    out = build_run_with_failures(tmp_path); restart_and_fail_everything(out, clear_deliverables=True)
    assert run_finished_with_failures(_state(out), out) == 2

@pytest.mark.parametrize("fence,expected", [("finalizer", 2), ("cancelled", None), ("finalizer_failed", None)])
def test_an_all_failed_run_reads_its_fence(tmp_path, fence, expected):    # R2-M4
    out = build_run_with_failures(tmp_path, failed_stems=("a", "b")); close_fence(out, how=fence)
    assert run_finished_with_failures(_state(out), out) == expected

def test_an_all_failed_run_with_an_unresolved_image_returns_none(tmp_path):
    out = build_run_with_failures(tmp_path, failed_stems=("a",), unresolved_stems=("b",))
    assert run_finished_with_failures(_state(out), out) is None

def test_the_verdict_reads_neither_state_nor_config(monkeypatch, tmp_path):    # N1, I1
    out = build_run_with_failures(tmp_path); state = _state(out)
    monkeypatch.setattr(run_state_module, "resolve_run_state", _explode)
    monkeypatch.setattr(run_state_module, "_read_state_config", _explode)
    assert run_finished_with_failures(state, out) == 1

def test_resolve_fills_the_finalization_digests(tmp_path):
    out = build_run_with_failures(tmp_path)
    assert _state(out).finalization_digests == accepted_finalization_digests(read_config(out))

def test_rule_one_still_proves_a_complete_run(tmp_path):
    assert _state(build_complete_run(tmp_path)).completion == "complete"
```

The small helpers (`hold_active_fence`, `publish_image_success_for`,
`append_retry_attempt_without_result`, `accept_and_fail`, `rewrite_mirror_rows`, `write_marker`,
`restart_and_fail_everything`, `close_fence`, `read_config`, `_explode`) live in the test
module, each a call to a real writer (`initialize_slurm_lifecycle`, `cancel_generation`,
`mark_generation_failed`, the `--restart` clearing function) or an `unlink`. `write_marker`'s two
invalid kinds are the one hand-edit, because no writer produces an invalid marker; the test says
so. None edits the failure journal by hand.

`test_a_master_that_merely_predates_an_image_claims_no_exclusion`
(`tests/unit/sdk_/test_run_state.py:1351`) and
`test_a_store_the_master_excluded_is_named_and_the_run_stays_incomplete` (`:1272`) must stay
green unedited: they pin `excluded_store_names`' two answers through `_advisories`.
- [ ] **Step 2: Run them; every verdict test fails with `ImportError`.**
  `$PT tests/unit/sdk_/test_run_finished_with_failures.py`
- [ ] **Step 3: Implement** the two `RunState` fields, the two helpers, `excluded_store_names`,
  the constant move, the verdict; route `_run_proof_covers_current_inventory` through the
  helpers (its process branch and `_source_set_binding` unchanged).
- [ ] **Step 4: Run the new file plus `tests/unit/sdk_/test_run_state.py` (with the deliberate
  field-set edit) and `test_run_state_layering.py`; all pass.**
- [ ] **Step 5: Mutation check, by deletion** (plan review C2). Delete one conjunct at a time;
  each must fail the named test:

  | Deleted conjunct | Must fail |
  |---|---|
  | step 1 (`completion == "failed"`) | `test_a_live_run_returns_none` |
  | step 2 (no unresolved image) | `test_an_unresolved_image_beside_a_current_proof_returns_none` |
  | step 3 proof present | `test_no_aggregate_proof_with_a_verified_image_returns_none` |
  | inventory digest | `test_a_new_image_that_failed_returns_none` |
  | scientific digest | `test_an_edited_pipeline_returns_none` |
  | finalization digest | `test_a_killed_metadata_change_returns_none` |
  | source-set digest **and** count | `test_a_retry_that_verified_one_image_returns_none`, `test_an_unprojectable_store_returns_none_and_is_named`, `test_a_late_image_returns_none_and_names_no_store` |
  | step 4 marker-by-path | `test_an_all_failed_run_beside_any_marker_returns_none[valid]`, `[wrong_version]`, `[empty_outputs]` |
  | step 4 marker check rewritten through `_valid_aggregate_proof` | `[wrong_version]`, `[empty_outputs]` (the review's "absent vs invalid" mutant) |
  | step 4 no other-generation deliverable | `test_an_all_failed_restart_beside_an_old_master_returns_none` |
  | step 4 finalizer-closed fence | `test_an_all_failed_run_reads_its_fence[cancelled]`, `[finalizer_failed]` |
  | step 5 (process) | `test_a_process_run_returns_none` |
  | `excluded_store_names`' `projectable is False` clause | `test_a_late_image_returns_none_and_names_no_store`, `test_a_master_that_merely_predates_an_image_claims_no_exclusion` |

  `source_image_count` cannot be isolated from `source_set_digest` (both move with the verified
  set); the table says so rather than claiming separate coverage. Each mutant of the two shared
  helpers must also fail a rule-1 test (`test_each_of_rule_ones_comparisons_is_load_bearing`,
  `test_run_state.py:1034`; `test_clause_one_is_load_bearing`, `:953`). Back up by full path,
  restore from the saved bytes, hash-check; record the table, with each result, in the cluster
  report.
- [ ] **Step 6: Commit** the source files, the moved helper, and the test files.

### Task 2: the waiter composes the verdict (C1, D14, N3)

**Files:**
- Modify: `src/phenotypic/_cli/_cli_wait.py`: `RunSnapshot` (`:90-111`), `_run_snapshot`
  (`:116-130`), `decide_wait_outcome` (`:309-366`), `_unpublished_line` (`:404-424`); add
  `finished_run_outcome`.
- Modify: `src/phenotypic/sdk_/_exit_codes.py:29-30` docstring (ENDED_UNPUBLISHED).
- No change to `publish_aggregate_snapshot` (R2-C1: no marker field).
- Test: `tests/unit/cli/test_cli_wait.py`

**Interfaces:**
- Consumes: Task 1's `run_finished_with_failures`.
- Produces:
  - `RunSnapshot.finished_with_failures: int | None = None` and
    `RunSnapshot.excluded_stores: tuple[str, ...] = ()`, both computed in `_run_snapshot`'s one
    `resolve_run_state` pass (`excluded_store_names(state.images)`, Task 1; R2-I4).
  - `decide_wait_outcome(..., finished_with_failures: int | None = None) -> RunExitCode | None`:
    still pure. Rule 3 requires `published and finished_with_failures is not None`; rule 4
    (process) unchanged.
  - `finished_run_outcome(output_dir: Path, generation: str | None) -> RunExitCode | None`:
    `decide_wait_outcome` over one `_run_snapshot` and the other seams, with `live` empty.
    Only the SLURM observer calls it (Task 5).
  - `_proof_published_by(output_dir: Path, generation: str | None) -> Mapping[str, object] | None`:
    the aggregate marker when its `execution_epoch` is `generation` (with `None`, any marker),
    else `None`.
  - `_unpublished_line(output_dir, *, cancelled: bool, published_proof: Mapping[str, object] | None, excluded_stores: tuple[str, ...]) -> str`:
    with no `published_proof`, today's line. With one: when `excluded_stores` is non-empty,
    spec PA4's D15 line naming them; otherwise the gap line, which names no store.
    `<logs>` is `slurm_logs_dir(output_dir)`. `_report_outcome` (`:427+`) passes the snapshot's
    `excluded_stores`, never a fresh resolve.

- [ ] **Step 1: Write the failing tests.**

```python
def test_failed_published_and_current_is_3():
    assert decide_wait_outcome("failed", NO_LIVE, published=True, process_run=False,
                               unresolved=0, finished_with_failures=2) is RunExitCode.COMPLETE_WITH_FAILURES

def test_failed_published_but_not_current_is_5():                 # D14
    assert decide_wait_outcome("failed", NO_LIVE, published=True, process_run=False,
                               unresolved=0, finished_with_failures=None) is RunExitCode.ENDED_UNPUBLISHED

def test_a_process_run_keeps_rule_four():
    assert decide_wait_outcome("failed", NO_LIVE, published=False, process_run=True,
                               unresolved=0, finished_with_failures=None) is RunExitCode.COMPLETE_WITH_FAILURES

def test_the_verdict_comes_from_the_same_pass(monkeypatch, tmp_path):   # N1
    calls = count_calls(monkeypatch, "phenotypic.sdk_.resolve_run_state")
    _run_snapshot(build_run_with_failures(tmp_path))
    assert calls == [1]

def test_an_unprojectable_store_exits_5_naming_it(monkeypatch, tmp_path, capsys):   # N3, D15, R2-C1
    out = build_run_with_failures(tmp_path, unprojectable_stems=("c",))
    code = wait_for_run(out, **no_live_jobs_published(monkeypatch, out))
    text = capsys.readouterr().out
    assert code is RunExitCode.ENDED_UNPUBLISHED
    assert "leave out plate/c" in text and "re-measuring them" in text

def test_a_rolling_input_gap_exits_5_naming_no_store(monkeypatch, tmp_path, capsys):   # R2-C1
    out = build_run_with_failures(tmp_path, late_stems=("c",))
    assert wait_for_run(out, **no_live_jobs_published(monkeypatch, out)) is RunExitCode.ENDED_UNPUBLISHED
    text = capsys.readouterr().out
    assert "no longer cover the run" in text and "plate/" not in text

def test_a_stale_publication_exits_5_with_the_gap_line(monkeypatch, tmp_path, capsys):
    out = build_run_with_failures(tmp_path); bump_metadata_snapshot_digest(out)
    assert wait_for_run(out, **no_live_jobs_published(monkeypatch, out)) is RunExitCode.ENDED_UNPUBLISHED
    assert "no longer cover the run" in capsys.readouterr().out

def test_the_variant_needs_a_published_proof(monkeypatch, tmp_path, capsys):   # C1
    out = build_run_with_failures(tmp_path); aggregate_publication_marker_path(out).unlink()
    wait_for_run(out, **no_live_jobs_published(monkeypatch, out))
    assert "published nothing" in capsys.readouterr().out

def test_the_excluded_names_come_from_the_same_pass(monkeypatch, tmp_path):   # R2-I4
    calls = count_calls(monkeypatch, "phenotypic.sdk_.resolve_run_state")
    snap = _run_snapshot(build_run_with_failures(tmp_path, unprojectable_stems=("c",)))
    assert calls == [1] and snap.excluded_stores == ("plate/c",)

def test_an_all_failed_run_exits_3(monkeypatch, tmp_path):                    # D16, Review Focus 1
    out = build_run_with_failures(tmp_path, failed_stems=("a", "b"))
    assert finished_run_outcome(out, None) is RunExitCode.COMPLETE_WITH_FAILURES
    assert wait_for_run(out, **no_live_jobs_published(monkeypatch, out)) is RunExitCode.COMPLETE_WITH_FAILURES

def test_an_all_failed_slurm_run_exits_3(tmp_path):                           # D16 on SLURM, R2-M3
    out = build_run_with_failures(tmp_path, failed_stems=("a", "b"))
    gen = finish_slurm_generation_all_failed(out)   # the real finalizer path: no aggregate,
                                                    # closes "terminal_incomplete" (_cli_checkpoint_handler.py:317-331, 395-401)
    assert _published(out, gen) is True
    assert finished_run_outcome(out, gen) is RunExitCode.COMPLETE_WITH_FAILURES

def test_accept_failures_stays_strict(monkeypatch, tmp_path):                  # D15
    out = build_run_with_failures(tmp_path, unprojectable_stems=("c",))
    assert wait_for_run(out, accept_failures=True, **no_live_jobs_published(monkeypatch, out)) is RunExitCode.ENDED_UNPUBLISHED
```

Every existing `decide_wait_outcome` test keeps its assertions; those that reach rule 3 gain
`finished_with_failures=<N>`.
- [ ] **Step 2: Run `$PT tests/unit/cli/test_cli_wait.py`; the new tests fail.**
- [ ] **Step 3: Implement** the field, the parameter, the wrapper and the variant line; update
  the `ENDED_UNPUBLISHED` docstring to "published nothing, or published outputs that no
  longer cover the run".
- [ ] **Step 4: Run `tests/unit/cli/test_cli_wait.py` and `tests/unit/cli/test_cli_wait_*`
  (`ls tests/unit/cli | grep wait`); all pass.**
- [ ] **Step 5: Mutation check.** Drop the `finished_with_failures` conjunct from rule 3:
  `test_failed_published_but_not_current_is_5` must fail. Restore from bytes.
- [ ] **Step 6: Commit.**

### Task 3: the status, local exits, rehydration and the claim gate (C2)

**Files:**
- Modify: `src/phenotypic/_gui/shell/_runs_registry.py`:
  - `RunStatus` / `_RUN_STATUSES` (`:103-130`), `_TERMINAL_STATUSES` (`:133-135`);
  - `RunRecord` (`:242+`): field `failed_images: int | None = None`, serialised beside
    `returncode` (`:1244-1245, 1501`), read tolerantly;
  - `observe_local_exit` (`:636-688`): the exit-3 arm, and `failed_images = None` on every
    other outcome;
  - `compare_and_set` (`:502-518`): a `failed_images` keyword, reset to `None` for any status
    other than `complete_with_failures` (plan review I2);
  - `_rehydrated_status` (`:1026-1104`) returns `(status, detail, failed_images)`, and both
    consumers persist N: `rehydrate_from_sandbox` (`:829`) and `_release_dead_owner_locked`
    (`:1357-1397`) (I3);
  - `_foreign_run_conflict` (`:1407-1479`) and its "Fires when" docstring (`:1418-1423`).
- Modify: `src/phenotypic/_gui/run_console/_callbacks.py:862` and
  `_gui/run_console/_slurm_observer.py:62` (terminal sets).
- Modify: `src/phenotypic/_gui/run_console/_recent_runs.py:37-66` (`RecentRunRow` gains
  `failed_images: int | None`) and `_layout.py:168-172` (status cell text via one
  `status_label(status: RunStatus, failed_images: int | None) -> str` helper in `_runs_registry.py`);
  `_callbacks.py:2549-2560, 2591-2622` (banner and no-dashboard text use it).
- Modify: `src/phenotypic/_gui/run_console/_assets/run_console.css:338-350`: add
  `.run-console-recents-status-complete_with_failures { color: var(--oi-orange-text); }`.
- Modify: `docs/source/contrib_guide/tracked_state.md:433-450`: Known consumers rows for
  `observe_local_exit`, `_rehydrated_status`, `_foreign_run_conflict`.
- Test: `tests/unit/gui/shell/test_runs_registry.py`, `tests/unit/gui/test_runs_registry_layout.py`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `RunStatus` gains `"complete_with_failures"` (terminal).
  - `RunRecord.failed_images: int | None`.
  - `status_label(status: RunStatus, failed_images: int | None) -> str`: `"complete · N failed"`
    for the new status, else `status`. Used by the Recent Runs cell, the banner and the
    no-dashboard text.
  - `_foreign_run_conflict` returns `None` for a confirmed run; for an unconfirmed `failed`
    run it returns spec PA11's message: "output has an unfinished run with N failed image(s)
    of M accepted; re-run it from the CLI or wait for it to finish".

- [ ] **Step 1: Write the failing tests.**

```python
def test_exit_3_with_confirming_evidence_is_complete_with_failures(tmp_path):
    out = build_run_with_failures(tmp_path); reg, rec = allocate_local(out)
    assert reg.observe_local_exit(rec.run_id, rec.generation, RunExitCode.COMPLETE_WITH_FAILURES)
    got = reg.get(rec.run_id)
    assert (got.status, got.failed_images) == ("complete_with_failures", 1)
    assert status_label(got.status, got.failed_images) == "complete · 1 failed"

def test_exit_3_with_an_unresolved_image_stays_failed(tmp_path): ...   # status "failed",
def test_exit_3_without_an_aggregate_proof_stays_failed(tmp_path): ...  # detail names "exit 3"
def test_exit_1_is_unchanged(tmp_path): ...                             # status "failed"

def test_an_old_record_without_failed_images_loads(tmp_path):
    write_owner_record_without(tmp_path, "failed_images")
    assert RunRegistry.load(...).get(...).failed_images is None

@pytest.mark.parametrize("via", ["compare_and_set", "observe_local_exit_0"])
def test_n_never_survives_a_status_change_on_one_record(tmp_path, via):       # Review Focus 4, I10
    out = build_run_with_failures(tmp_path); reg, rec = observe_exit_3(out)
    move_same_record_to_complete(reg, rec, via=via)   # CAS to "complete", or exit 0 on that generation
    assert (reg.get(rec.run_id).status, reg.get(rec.run_id).failed_images) == ("complete", None)

def test_rehydration_reads_a_cli_run_with_failures(tmp_path):                  # I3
    out = build_run_with_failures(tmp_path)
    assert RunRegistry._rehydrated_status(out)[::2] == ("complete_with_failures", 1)
    assert rehydrated_record(out).failed_images == 1

def test_a_dead_owner_release_keeps_n(tmp_path, dead_owner_pid):               # I3, fixture :1852
    out = build_run_with_failures(tmp_path); abandon_owner(out, dead_owner_pid)
    reg = registry_over(tmp_path); reg.allocate(...)        # the claim releases the dead owner
    assert persisted_owner(out)["failed_images"] == 1

def test_an_all_failed_cli_run_is_claimable_for_retry_failures(tmp_path):      # D16
    out = build_run_with_failures(tmp_path, failed_stems=("a", "b"))
    assert RunRegistry._foreign_run_conflict(out) is None

@pytest.mark.parametrize("status", sorted(_RUN_STATUSES))
def test_recent_runs_renders_every_status(status):                             # spec R2, Minor 6
    cell = render_recent_row(RecentRunRow(rel_path="r", status=status, has_dashboard=False,
                                          last_modified_seconds=0.0, mode="local",
                                          failed_images=1 if status == "complete_with_failures" else None))
    assert f"run-console-recents-status-{status}" in cell.className and "None" not in cell.children

def test_a_confirmed_run_with_failures_is_claimable(tmp_path):            # D10, CLI-launched
    out = build_run_with_failures(tmp_path)
    assert RunRegistry._foreign_run_conflict(out) is None

def test_a_gui_launched_run_with_failures_is_claimable(tmp_path):          # D10, terminal owner
    out = build_run_with_failures(tmp_path); reg, rec = observe_exit_3(out)
    assert reg.allocate(mode="local", output_dir=out, rel_path=..., command_digest="x", status="queued")

def test_a_failed_image_beside_an_unresolved_image_refuses(tmp_path):      # split of :363-390
    out = build_run_with_failures(tmp_path); drop_success_record(out, "a")
    msg = RunRegistry._foreign_run_conflict(out)
    assert "unfinished run with 1 failed image(s)" in msg and "from the CLI" in msg   # N7
    assert "non-GUI" not in msg

def test_a_stale_publication_refuses(tmp_path): ...       # bump_metadata_snapshot_digest
def test_a_process_tree_with_failures_refuses(tmp_path): ...   # N4
def test_an_active_run_still_refuses(tmp_path): ...       # live fence
def test_an_incomplete_run_still_refuses(tmp_path): ...   # build_incomplete_run (:797)
def test_validate_and_retry_failures_allocate_into_a_confirmed_output(tmp_path): ...
```

`test_a_failed_image_refuses_the_claim` (`:363-390`) is replaced by the two refusal tests
above plus the CLI-launched claimable test; the vocabulary guard
`test_every_status_vocabulary_in_the_gui_agrees_with_the_registry` (`:1268`) must pass
without edits once the three terminal sets carry the new literal.
- [ ] **Step 2: Run `$PT tests/unit/gui/shell/test_runs_registry.py tests/unit/gui/test_runs_registry_layout.py`; the new tests fail.**
- [ ] **Step 3: Implement.** Each call site resolves `depth="shallow"` once and passes that
  state to the verdict (N1).
- [ ] **Step 4: Run the two files, `tests/unit/gui/run_console/`, and
  `tests/integration/gui/test_run_console_callbacks.py`; all pass.**
- [ ] **Step 5: Mutation check.** Delete the PA11 condition in `_foreign_run_conflict` (so a
  confirmed run falls to the `failed` refusal): `test_a_confirmed_run_with_failures_is_claimable`
  must fail. Delete the `failed_images` reset in `compare_and_set`: the parametrised
  `test_n_never_survives_…[compare_and_set]` must fail. Restore from bytes.
- [ ] **Step 6: Commit**, including the CSS, the layout and `tracked_state.md`.

### Task 4: `submit_slurm` classifies the exit (C3)

**Files:**
- Modify: `src/phenotypic/_gui/run_console/_slurm.py`: `_reconcile_submission` (`:642-743`),
  `submit_slurm` (`:746-872`); add `SlurmFinishedWithoutSubmission`.
- Modify: `src/phenotypic/_gui/run_console/_callbacks.py`: `_SlurmCompletion` (`:849-855`),
  `_complete_slurm_submission` (`:1030-1201`), the seam annotation (`:1310`), the banner that
  reads `completion.result` (`:2672`); `run_console/_app.py:74`; `shell/_app.py:215, 681`
  (two more seam annotations; plan review I5).
- Modify: `src/phenotypic/_gui/shell/_runs_registry.py`: add
  `observe_finished_without_submission`.
- Modify: the browser-test fake behind `shell/_app.py:245-246` (find it with
  `grep -rn slurm_submitter tests/`).
- Test: `tests/unit/gui/run_console/test_slurm.py`,
  `tests/integration/gui/test_run_console_callbacks.py`, `tests/unit/gui/shell/test_runs_registry.py`

**Interfaces:**
- Consumes: Task 1, Task 3 (`complete_with_failures`, `failed_images`).
- Produces:
  - `@dataclass(frozen=True) class SlurmFinishedWithoutSubmission: returncode: int; output_dir: Path; stdout: str; stderr: str`
    (not a `SlurmSubmitResult` subclass).
  - `submit_slurm(...) -> SlurmSubmitResult | SlurmFinishedWithoutSubmission`.
  - `_reconcile_submission(output_dir: Path, *, cancel_inactive: bool) -> _SubmissionReconciliation`:
    with `False`, an inactive fence returns `_SubmissionReconciliation(None, generation, (), True)`
    without reading the ledger (N2).
  - `RunRegistry.observe_finished_without_submission(run_id: str, generation: UUID, returncode: int) -> bool`:
    `cancelling` → `cancelled`; `0` → `_local_completion_evidence_conflict` (`:691-770`);
    `3` → Task 1's verdict over one shallow state; detail "nothing needed SLURM; the run
    finished in this process".
  - `_SlurmCompletion.finished: SlurmFinishedWithoutSubmission | None = None`; the banner
    shows that detail for it.

- [ ] **Step 1: Write the failing tests** (mock target `_STREAM_TARGET`, `test_slurm.py:49`):

```python
def test_exit_4_with_a_durable_job_set_is_not_reconciled(tmp_path):
    r, cancel = submit_with_cancel_spy(tmp_path, rc=4, manifest={"0": "45_0"})
    assert isinstance(r, SlurmSubmitResult) and r.reconciled is False and r.job_id == "45"
    assert cancel.call_count == 0                                            # Minor 5

@pytest.mark.parametrize("jobs", ["durable", "none"])
def test_exit_4_with_an_inactive_fence_never_cancels(tmp_path, jobs):     # I5
    r, cancel = submit_with_inactive_fence(tmp_path, rc=4, jobs=jobs)
    assert cancel.call_count == 0
    if jobs == "durable":
        assert r.submitted_jobs.generation == THIS_GENERATION            # Minor 4: never the old one

def test_exit_4_with_no_job_names_submitted(tmp_path):
    with pytest.raises(SlurmSubmitError, match="SUBMITTED"):
        submit_with(tmp_path, rc=4)

@pytest.mark.parametrize("rc", [2, 4])
def test_no_stale_binding_to_an_inactive_generation(tmp_path, rc):         # N2
    seed_inactive_generation_with_intent_row(tmp_path)
    with pytest.raises(SlurmSubmitError) as exc, cancel_spy() as cancel:
        submit_with(tmp_path, rc=rc)
    assert not isinstance(exc.value, SlurmSubmitPending) and cancel.call_count == 0

def test_exit_2_never_reconciles(tmp_path, monkeypatch):
    spy = spy_on(monkeypatch, "phenotypic._gui.run_console._slurm._reconcile_submission")
    with pytest.raises(SlurmSubmitError, match="exited with code 2"), cancel_spy() as cancel:
        submit_with(tmp_path, rc=2, stderr="Error: --embed-stage gpu ...")
    assert spy.call_count == 0 and cancel.call_count == 0

@pytest.mark.parametrize("rc", [0, 3])
def test_finished_without_submitting_binds_nothing(tmp_path, rc):
    seed_previous_generation_job_set(tmp_path)
    r, cancel = submit_with_cancel_spy(tmp_path, rc=rc)
    assert isinstance(r, SlurmFinishedWithoutSubmission) and r.returncode == rc
    assert cancel.call_count == 0

def test_the_timeout_arm_still_cancels_an_inactive_generation(tmp_path):  # replaces :815-838's pin
    cancel = submit_with_inactive_fence(tmp_path, rc=-1, timed_out=True, jobs="durable")
    assert cancel.call_count == 1

def test_a_0_result_lands_terminal_and_says_so(...):   # integration: record "complete"; the
    # rendered banner contains "nothing needed SLURM; the run finished in this process" (I5)
def test_a_3_result_lands_complete_with_failures(...):   # one synthetic test (Minor 9)
def test_a_cancelling_record_meeting_a_0_3_result_is_cancelled(...):
```

`test_abnormal_exit_cancels_recovered_inactive_generation` (`test_slurm.py:815-838`) is
rewritten to assert `cancel.call_count == 0` for exit 2 (R1; the commit message names the
behaviour change). `test_submit_slurm_raises_on_non_zero_exit` keeps its assertions.
- [ ] **Step 2: Run `$PT tests/unit/gui/run_console/test_slurm.py`; the new tests fail.**
- [ ] **Step 3: Implement** the classifier, the flag, the result type, the union at the three
  seam sites and the fake, and the registry method.
- [ ] **Step 4: Run `tests/unit/gui/run_console/`, `tests/unit/gui/shell/`,
  `tests/integration/gui/test_run_console_callbacks.py`; all pass.**
- [ ] **Step 5: Mutation check.** (a) Re-route exit 4 through the ambiguous arm; (b) pass
  `cancel_inactive=True` on exit 4; (c) let `cancel_inactive=False` fall through to the ledger;
  (d) reconcile on exit 2. Each must fail a named test above. Restore from bytes.
- [ ] **Step 6: Commit.**

### Task 5: the observer hook (C4)

**Files:**
- Modify: `src/phenotypic/_gui/run_console/_slurm_observer.py`: `_Observation` (`:327-335`)
  gains `failed_images: int | None = None`; `_apply` (`:861-896`) passes it to
  `compare_and_set` (Task 3's keyword); the "inactive, no marker" branch (`:661-671`).
- Modify: `tracked_state.md` Known consumers: the observer's `finished_run_outcome` row.
- Test: `tests/unit/gui/run_console/test_slurm_observer.py`

**Interfaces:**
- Consumes: Task 2's `finished_run_outcome`; Task 3's status and `failed_images`.
- Produces: at that branch, after `all_confirmed_inactive`, `finished_run_outcome(record.output_dir, str(generation))`
  maps `COMPLETE_WITH_FAILURES` → `_Observation("complete_with_failures", …, terminal=True)`
  with `failed_images=N`, `COMPLETE` → `complete`, anything else → today's `failed`. The
  observer adds no `resolve_run_state` call, so `_resolve_run_state_call_sites` (`:1011`)
  is unchanged.

- [ ] **Step 1: Write the failing tests.**

```python
def test_a_finalizer_closed_fence_with_failures_is_complete_with_failures(tmp_path):
    reg, rec = _registered_slurm_run(tmp_path)               # :92
    make_it_finish_with_failures(rec.output_dir, closed_by_finalizer=True)
    obs = _bound_observer(reg, ...).observe_once()           # :202
    assert (reg.get(rec.run_id).status, reg.get(rec.run_id).failed_images) == ("complete_with_failures", 1)

def test_a_cancelled_fence_with_failures_stays_failed(tmp_path): ...
def test_a_staged_terminal_incomplete_reaches_the_hook(tmp_path):      # Minor 16
    # orchestration phase "terminal_incomplete" (_cli_checkpoint_handler.py:398-401)
    ...; assert reg.get(rec.run_id).status == "complete_with_failures"
```
- [ ] **Step 2: Run; they fail.** `$PT tests/unit/gui/run_console/test_slurm_observer.py`
- [ ] **Step 3: Implement the hook.**
- [ ] **Step 4: Run the file; all pass.**
- [ ] **Step 5: Commit.**

### Task 6: the snapshot badge arm (C4, D3, I4)

**Files:**
- Modify: `src/phenotypic/_gui/_snapshot_status.py`: `_run_completion` (`:101-121`) returns
  the `RunState | None`; `snapshot_refresh_status` (`:53-98`) adds the arm.
- Modify: `tracked_state.md` Known consumers: the badge row.
- Test: `tests/unit/gui/test_snapshot_status.py`

**Interfaces:**
- Consumes: Task 1.
- Produces: when completion is `failed` and the verdict over the same shallow state is `N`,
  the arm skips `_UNFINISHED_BADGE` and falls through to `snapshot_is_current` (`:92-98`):
  `("Run complete · N images failed", "warning", False)` when current (the standalone form
  ends "· restart app to refresh", `True`), else today's "Changed on disk" pair.

- [ ] **Step 1: Write the failing tests.**

```python
def test_a_current_run_with_failures_reads_complete_with_n(tmp_path):
    root = output_root_over(build_run_with_failures(tmp_path))
    assert snapshot_refresh_status(root, refresh_supported=True) == ("Run complete · 1 images failed", "warning", False)

def test_a_refinalized_run_with_failures_reads_changed_on_disk(tmp_path):
    # pattern of test_a_rewritten_processing_product_shows_changed_on_disk (:200)
    ...; assert label == "Changed on disk"

def test_a_curated_run_with_failures_still_reads_complete_with_n(tmp_path): ...
```
Keep `test_every_completion_literal_has_a_badge` (`:187`) and
`test_the_completion_question_is_asked_shallow_at_the_run_root` (`:442`) green unchanged.
"1 images" is the spec's fixed copy ("N images failed"); do not pluralise.
- [ ] **Step 2: Run; they fail.** `$PT tests/unit/gui/test_snapshot_status.py`
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the file and `tests/unit/gui/results_viewer/`; all pass.**
- [ ] **Step 5: Commit.**

### PA1 gate

- Per task: the files above.
- Per phase: `tests/unit/sdk_/`, `tests/unit/cli/test_cli_wait*.py`, `tests/unit/gui/`,
  `tests/integration/gui/`, `tests/gui/` (Slurm, detached worktree). Its red set must be a
  subset of the 13 inherited reds.
- **Deep review** (`implementation-test-reviewer`, Opus) over PA1's combined diff, with the
  mutation tables from T1, T2, T4. Report:
  `reports/2026-11-24-gui-dvc-followups/pa1-review.md`.
