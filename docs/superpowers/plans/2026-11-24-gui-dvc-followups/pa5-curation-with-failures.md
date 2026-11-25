# PA5 · Curation of a run that finished with failures

**Spec:** PA10 (D2, D11); re-review C4 note, N1

---

**Goal:** A run the verdict confirms is writable exactly as a complete run is, at every
write-authority site, and a curation write leaves its verdict and every enforced descriptor
unchanged.

**Depends on:** PA1 T1, T3 (verdict, status); PA3 T3 (curation writes the TSV; shares
`results_viewer/_output_root.py`).

### Task 1: `run_permits_mutation` at every write-authority site (C11)

**Files:**
- Modify: `src/phenotypic/_gui/results_viewer/_mutation_guard.py:38-45, 85-187, 214-229`.
- Modify: `src/phenotypic/_gui/results_viewer/_output_root.py:687-730` (rename
  `run_is_complete` → `run_permits_mutation`, CAN-17 docstring), `:780-797`
  (`mutation_snapshot_is_safe`); every caller of `run_is_complete`
  (`grep -rn run_is_complete src/phenotypic/_gui`: 11 lines), and its four test references:
  `tests/gui/results_viewer/test_mutation_guard.py:259`,
  `tests/gui/results_viewer/test_output_discovery_contracts.py:228, 250, 298` (Minor 8).
- Modify: `src/phenotypic/_gui/results_viewer/_processing_inventory.py:97`.
- Modify: `tracked_state.md` Known consumers: the guard, `OutputRoot`, the inventory rows.
- Test: `tests/gui/results_viewer/test_mutation_guard.py`, `tests/unit/gui/results_viewer/test_output_root.py`,
  `tests/gui/results_viewer/test_output_root.py`, `tests/unit/gui/test_curation_does_not_unfence_the_run.py`,
  `tests/unit/gui/results_viewer/test_curation_labels.py`.

**Interfaces:**
- Consumes: PA1 T1 `run_finished_with_failures(state, output_dir)`.
- Produces:
  - `run_permits_mutation(state: RunState | None, output_dir: Path | None) -> bool` in
    `_mutation_guard.py`: `True` for `state is None` (bundle), `completion == "complete"`, or
    a confirmed verdict over **that** `state` (N1).
  - `authorize` calls it on the bound state, the fresh deep state and the verified deep state.
  - `OutputMutationReceipt` gains `failed_images: int | None`, recorded beside
    `run_identity_digest` [spec call 14]; a fresh verdict whose N differs from the bound one
    refuses with "… blocked: the failed image count changed from N to M after this snapshot
    was bound. Refresh Results and Analysis."
  - `load_or_scan_processing_inventory`: `reusable` uses the same predicate.

- [ ] **Step 1: Write the failing tests.**

```python
def test_the_guard_authorizes_a_confirmed_run(tmp_path):
    root = output_root_over(build_run_with_failures(tmp_path))
    receipt = OutputMutationGuard(root, gen).authorize("curate", presented_generation=gen)
    assert receipt.failed_images == 1

def test_the_guard_refuses_an_unresolved_failed_run(tmp_path): ...        # drop_success_record
def test_the_guard_refuses_a_stale_publication(tmp_path): ...             # bump_metadata_snapshot_digest
def test_a_verdict_that_flips_between_bind_and_write_is_refused(tmp_path):   # I9, R2-I3
    root = output_root_over(out := build_run_with_failures(tmp_path, failed_stems=("b", "c")))
    publish_image_success_for(out, "b")      # c still failed: completion STAYS "failed"; only the
                                             # source set moves, so only the verdict flips
    assert resolve_run_state(out, depth="deep").completion == "failed"
    with pytest.raises(OutputMutationBlocked, match="run state is failed"):
        OutputMutationGuard(root, gen).authorize("curate", presented_generation=gen)

def test_a_fresh_verdict_whose_n_differs_refuses(tmp_path, monkeypatch):   # spec call 14, R2-I3
    root = output_root_over(build_run_with_failures(tmp_path, failed_stems=("b", "c")))   # bound N = 2
    monkeypatch.setattr(guard_module, "run_finished_with_failures", lambda state, out: 1)  # fresh N = 1
    with pytest.raises(OutputMutationBlocked, match="failed image count changed"):
        OutputMutationGuard(root, gen).authorize("curate", presented_generation=gen)

def test_a_confirmed_run_gets_an_exhaustive_inventory(tmp_path):
    assert output_root_over(build_run_with_failures(tmp_path)).has_exhaustive_processing_inventory

def test_the_read_only_diagnostic_is_gone_for_a_confirmed_run(tmp_path):
    assert output_read_only_diagnostic(output_root_over(build_run_with_failures(tmp_path))) is None

def test_a_curation_write_keeps_the_verdict_and_the_fence(tmp_path):
    out = build_run_with_failures(tmp_path); before = resolve_run_state(out, depth="deep")
    curate_one_colony(out)                                    # test file's _curate_one_colony (:57)
    after = resolve_run_state(out, depth="deep")
    assert aggregate_proof_is_current(out)
    assert run_finished_with_failures(after, out) == 1
    assert (after.identity.digest(), after.diagnostics) == (before.identity.digest(), before.diagnostics)
    assert enforced_descriptors_match(out, {"master_parquet", "metrics_json", "schema_json"})
    assert run_proof(out) is None

def test_a_cli_rerun_after_curation_reseeds_and_the_gui_reapplies(tiny_inputs_one_bad, tiny_pipeline_file, tmp_path):   # D11
    # a REAL run: tests/_tiny_run.py's tiny_inputs_one_bad (:91) gives one terminal failure
    out = tmp_path / "out"; argv = ["--pipeline", str(tiny_pipeline_file), "--input", str(tiny_inputs_one_bad), "--output", str(out)]
    assert run_cli(argv).exit_code == RunExitCode.COMPLETE_WITH_FAILURES     # run_cli, :191
    removed = curate_remove_one(out)
    result = run_cli(argv)
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES
    assert curation_labels_path(out).is_file() and errors_dir(out).is_dir()
    assert removed in mirror_keys(out)                         # reseeded uncurated
    viewer = refresh_viewer(out)                               # required: the fingerprint moved
    curate_any(viewer)                                         # (_curation_labels.py:773-800)
    assert removed not in mirror_keys(out)
```
- [ ] **Step 2: Run the test files listed; the new tests fail.**
- [ ] **Step 3: Implement**, renaming `run_is_complete` at all its callers in one commit.
- [ ] **Step 4: Run the listed files plus `tests/unit/gui/results_viewer/`, `tests/unit/gui/analysis/`,
  `tests/gui/`; all pass.**
- [ ] **Step 5: Mutation check** (I9). Delete each of the three `run_permits_mutation` calls in
  `authorize` (bound `:121`, fresh `:140`, verified `:170`) one at a time. Each must fail
  `test_a_verdict_that_flips_between_bind_and_write_is_refused` or a variant of it that flips the
  verdict between the named points (the test monkeypatches `resolve_output_run_state` to return
  the flipped state from the second, then the third, call). Restore from bytes.
- [ ] **Step 6: Commit.**

### PA4–PA5 gate

- Per phase: `tests/unit/gui/`, `tests/integration/gui/`, `tests/gui/` (Slurm, detached
  worktree). Red set: empty.
- **Deep review** (`implementation-test-reviewer`, Opus) over PA4 and PA5 together. Report:
  `pa4-pa5-review.md`.
