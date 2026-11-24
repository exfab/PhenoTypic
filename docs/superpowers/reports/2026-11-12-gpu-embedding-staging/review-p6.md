# Deep review P6: `--mode measure` through a republish Stage 4

**Range:** `577c59740..47678fffa` (6 commits: `69426a3e5`, `4c06f7bd6`, `65dd273d4`,
`fcdd4d7b5`, `01ae2c0e1`, `47678fffa`), worktree `gpu-embedding-staging`.
**Reviewer:** implementation-test-reviewer (analysis only; this file is the one write).
**Date:** 2026-11-24.

## Summary

The local half holds up. Several pieces work as specified:

- U8's carry rule (`measure_embedding_carry`, `replace_image_tables(carry_embeddings=)`);
- the record identity on republish;
- the epoch-stamped "done" predicate;
- the measure-mode failure stage;
- the gate lift.

The tests for each are real, and two of them were strengthened after a surviving mutant
(M10).

**The SLURM half has one contract-level defect.** The waiter decides a measure pass's
exit code entirely from the *forward* run's state. A measure pass that leaves an image
un-measured or un-re-embedded keeps the old record, and that record still verifies. So
the pass exits `0`, "finished, no terminal failures", in three situations I reproduced by
execution (C1). C17's deviation 1 is one instance of it. The same mechanism makes a
killed same-pipeline measure step invisible (I1). Separately, U8's carry turns the
documented remedy for a *missing* embedding folder into a measure failure (I2).

**Evidence I ran myself.** I ran three probe files, 6 tests in all, 6 passed, against
`HEAD` in the worktree's `uv` env (`PYTHONPATH=.:src`, `-p tests.conftest`). They are in
the session scratchpad,
`/scratch/anguy344/32407366/claude-5188/-bigdata-exfab-anguy344-PhenoTypic/755c852f-f7be-4fc5-96a7-36f51adcec11/scratchpad/p6probe/`,
and their essentials are quoted under each finding.

**What I did not run.** I did not run the P6 test files, any gate or the full suite, and
I quote no pass count for them. The plan's optional live check (p6 T2 Step 5) is not in
`live-smoke.md`, which has no measure section.

**Counts:** Critical 1 · Important 3 · Minor 11 · User decisions 4.

---

## Critical

### C1. A SLURM measure pass that leaves images un-measured or un-re-embedded exits `0`

**Where:**
- `_cli_checkpoint_handler.py:387-404`: the measure finalizer always closes with
  `closed_by=FINALIZER` and no `terminal_status`.
- `_cli_wait.py:204-264` (`_published`) and `:331-388` (`decide_wait_outcome`).
- `_cli_staged_controller.py:680-684`: leftovers are "terminal" but get no journal row.

**Mechanism.** The waiter returns `COMPLETE` when the pass is published (the fence was
closed by the finalizer) and `resolve_run_state(...).completion == "complete"`. In a
measure pass, every image the pass failed to reach still holds its forward record. That
record verifies whenever the pass left the root and table untouched, or left them
byte-identical, which is the C17 finding. Those images therefore count as `verified`. The
measure finalizer never asks whether *this pass* republished each manifest entry.

**Reproduced by execution** (`test_probe_wait.py`, `test_probe_dead.py`, built on
`build_measure_epoch_tree` and `_run_finalize`, then fed to
`decide_wait_outcome(..., LiveSlurmJobs())`):

| Scenario | Snapshot | Waiter |
|---|---|---|
| Same pipeline (`prior_pass=True`), Stage 4 republished image 0, image 1 stalled (U7) | `complete, accepted=2, verified=2` | **`0`** |
| Differing pipeline, Stage 4 republished image 0 only | `incomplete, verified=1` | `5` |
| Differing pipeline, both measure steps died before writing, and U15 gave up | `complete, verified=2` | **`0`** |

**What goes wrong.**

1. **Deviation 1, the case you asked about.** `--mode measure --embed-stage gpu` is the
   remedy R3 now names: "re-embed it on the GPU …" (`_cli_embed_stage.py:264-272`). A
   Stage-4 stall in that pass leaves the stalled images with their CPU-inline embeddings,
   carried under the same `params_sha256`, and the pass exits `0`. The run then silently
   mixes inline and Stage-4 embeddings, which is exactly what intent goal 3 and U3 exist to
   prevent. A DVC stage treats it as current. **Exit `0` here is misleading, not
   acceptable.**
2. **Row 3 contradicts U15b.** U15b says "a second unchanged round leaves the image
   unrecorded, exit 5". A measure image cannot be "unrecorded", because its forward record
   remains. So the measure role's dead-task retry gives up and then reports success for
   images it never measured. This holds whatever the pipeline, as long as the step dies
   before its table write.
3. Exit `5` for a measure pass is reached only by accident: a *differing* pipeline whose
   measure step did write, so the root changed. No test pins it end to end.

**Fix.**

- In the measure branch of `_run_finalize`, before closing, compute the leftovers: the
  manifest entries (`load_staged_manifest(config["manifest_path"])`) that have a published
  forward record (`_has_published_record`), are not `stage4_status(..., republish=True,
  epoch=epoch) == "done"`, and are not in `terminal_failure_work_ids_under_epoch(out,
  epoch)`.
- If any remain, log their stems with `recorded_stall_lines`, and close the generation so
  that `_published` is false. One way: `mark_generation_failed(out, epoch, "N image(s)
  were not re-measured/re-embedded by this pass; re-run the same command")` and
  `deactivate_orchestration(out, "terminal_incomplete")`. The waiter then returns `5` with
  the documented "re-run the same command".
- Add the end-to-end test:

  ```python
  # same-pipeline MeasureEpochTree(prior_pass=True); republish(0) only;
  # _run_finalize(...); decide_wait_outcome(...) == RunExitCode.ENDED_UNPUBLISHED
  ```

  Add a twin with `measure=False` (dead steps), and a control in which every entry is
  republished, which must give `0`. A journalled failure must give `3`.

The ordinary pre-P6 SLURM measure array likely has the killed-task half of this defect
too: a killed task leaves a verifying record, and `_publish_run_completion_marker` still
publishes. That is inferred from the code; I did not execute it.

---

## Important

### I1. A killed same-pipeline measure step is indistinguishable from a finished one, and Stage 4 then certifies the un-re-measured table (deviation 5)

**Where:** `_cli_staged_controller.py:441-467` (`_dead_measure_indices`), and
`_cli_staged_resume.py:252-317` (`_republish_stage4_status`, where "pending" needs only
that the measure pipeline's folder exists).

**Scenario.**

1. A SLURM same-pipeline measure, run after a code or schema change, so the table would
   change. A measure task is OOM-killed or loses its node before its table write.
2. The forward folder *is* the measure pipeline's folder, so the store reads `pending`,
   not `not_ready`. `_dead_measure_indices` does not resubmit it.
3. Stage 4 re-embeds the store and republishes its record over the **forward** table.
4. The image reads `done` and the pass exits `0`, with stale measurements and nothing
   logged.

The docstring at `:447-451` says the forward table "is the table the step would have
rewritten". That holds only when the measure is a no-op, and a no-op is the one case in
which nobody runs measure for the measurements.

Confirmed by execution (`test_probe_wait.py::test_same_pipeline_killed_measure_step_is_not_dead`):
with `measure=False` and the forward pipeline, `_dead_stage3_indices == []`, and
`_classify_stage4` puts both entries in `retryable`.

**Fix.** Give the measure step durable per-epoch evidence and make both readers ask for
it:

- `run_measure_step` writes `progress/measure_done/<epoch>/<ds>/<stem>` (atomic, under
  `commit_guard`) after `process_single_store_measure_core` returns.
- `_dead_measure_indices` counts as dead any entry with no such marker, no journal row
  under the epoch, and no `done` status.
- `_republish_stage4_status` returns `not_ready` without that marker when `epoch` is not
  None, so Stage 4 never republishes a store this pass did not measure.

The markers go with the pass, so a later pass with a new epoch ignores them. This also
closes C1's row 3 for both pipelines. Test: same pipeline with `measure=False` gives dead
`[0, 1]`, and after `run_measure_step` gives dead `[]`.

### I2. U8's carry turns the E5 remedy into a failure when an embedding folder is missing, and carries a dangling entry when only the file is

**Where:**
- `_cli_process_single.py:509-579`: `measure_embedding_carry` selects on
  `producer.params_sha256` alone and never checks the entry's files.
- `sdk_/_measurement_tables.py:1530-1533`: `_link_carried_embedding` raises
  `FileNotFoundError` when the source folder is absent.

**Scenario.** Parent E5 (dvc spec, "Remedy for an unreadable embedding") sends the user,
for an unreadable embedding, to a same-pipeline `--mode measure` that "re-embeds every
image". On SLURM with `--gpu-slurm`, `auto` now resolves `gpu`. The deferring core
selects the entry, because its params match, and then:

- **Folder deleted** (a partial rsync, or a hand cleanup): the store transaction raises
  `FileNotFoundError` inside the part. `measure_scientific_boundary` journals
  `MEASURE_FAILURE_STAGE`, and the image is never re-embedded. Every retry repeats this.
  The remedy can never succeed under `--embed-stage gpu`, and the user must know to pass
  `--embed-stage inline`.
- **File deleted, folder present:** `_link_tree` links nothing, and the new root declares
  a carried entry whose file does not exist. If Stage 4 then fails for that image, its
  republished record certifies a root with a dangling entry. The finalizer's embedding
  aggregation ("a failure fails finalization", E5) then fails the whole pass over one
  image.

Both cases are confirmed by execution (`test_probe_carry.py`, 2 passed):

```python
shutil.rmtree(store / embedding_table_relative_path(run.run_id, key).parent)
with pytest.raises(FileNotFoundError):
    replace_image_tables(store, tables_for_store(store), pipeline_sha256=run.pipeline_sha256,
                         carry_embeddings=[(run.run_id, key)])
# file-only deletion: the carried entry is declared, and its file does not exist
```

**Fix.** Carry only what a reader would accept. In `measure_embedding_carry`, skip a
candidate entry unless `read_embedding_entry(store, key, entry)` succeeds: present, its
sha256 matching, and not stale. Carrying an unreadable entry buys nothing, since its only
purpose is a valid fallback after a Stage-4 failure. Under rule 3 the image then carries
nothing for that key, as the spec already allows. Keep the raise in
`_link_carried_embedding` as the internal invariant.

The cost is one read of each carried table per image. Embedding tables are small next to
the store, but say so in the docstring. Tests: folder deleted gives no pair, Stage 4
re-embeds, and the exit is `0`; file corrupted gives no pair.

### I3. A measure resolves the hub's current commit instead of the forward run's, so a partial failure mixes model commits under one `params_sha256` (deviation 6)

**Where:** `_cli_embed_stage.py:724`: `recorded = ... if config.resume else None`. Measure
mode is never `resume`, so it always resolves afresh. Also `_cli_staged_slurm_worker.py:898-910`
and `_cli_stage4_worker.py:1767`.

Reading the revisions from job metadata only is correct for the *measure submission's*
pin, because the state's revisions are the forward run's. The question is which pin. For
an unpinned `revision`, `params_sha256` holds `revision: None`, while the entry's
`model.revision` is the resolved commit, and that commit is descriptor identity
(`_measure_dino_embedding.py:440-455`; dvc spec E5 "Mixed descriptors").

**Scenario.**

1. The forward run pinned commit X. The hub has since moved to Y.
2. A same-pipeline measure re-embeds with Y.
3. Each image Stage 4 fails keeps its carried X entry, because U8 matches on
   `params_sha256` and so treats X as comparable.
4. The pass exits `3`, and the deliverable carries two descriptors for one key. Under C1
   that happens at exit `0`.

U5's "one commit per run" is broken by the measure, not by sharding. See decision **D2**.

**Fix, if D2 says reuse.** For a measure, seed `pin_embed_revisions` with
`recorded_embed_revisions(output_dir)` (the forward state), exactly as a continuation
does, and resolve only the models recorded `None`.

---

## Minor

**M1. The orphan sweep is skipped in republish mode, which leaks signals (deviation 4).**
Where: `_cli_staged_controller.py:906`. An image republished and then killed before its
signals are consumed reads `done`, so no later round and no sweep in this pass removes
its signals. They are root-bound to the pre-commit root, so they are stale and can never
be reused: disk only, until a later forward sweep. The local path still sweeps in
republish mode (`_cli_stage4_worker.py:1747`) against forward records. That costs only
re-embedding, but it is inconsistent with the controller. Fix: in republish mode, sweep
only the entries whose `stage4_status(republish=True, epoch=…) == "done"`, in both places.

**M2. With the publish-only shortcut off, a committed but unrepublished image is
re-embedded.** Where: `_cli_stage4_worker.py:851`. A walltime between
`replace_run_embeddings` and `_republish_record` makes the next round's signals stale,
because the root moved. The image re-embeds, and its embeddings are written twice in one
pass. That contradicts p6 Step 3 ("Stage 4 writes the embeddings once"). The record is
still written once and last, so the verification cache stays sound. Fix, optional:
recognise "this pass committed" by provenance, for example stamp the pass epoch in the
entry's `compute` block, which is already excluded from grouping, and allow the shortcut
when every key carries it.

**M3. Local fencing (deviation 2).** `_local_record_lock(..., republish=True)` holds the
processing-state lock with no generation compare (`_cli_stage4_worker.py:1672-1688`).

- *Is local measure still fenced against a concurrent second invocation?* **No**, and it
  was not before P6 either: the inline core republished under no lock at all. What holds:
  `replace_run_embeddings` is root-fenced (`expected_root_sha256`), so a concurrent
  rewrite leaves the image pending (exit 1); `_republish_table_marker` rehashes
  artifacts at write; and the state lock serialises the record writes. The worst outcome
  is a non-verifying record plus exit `1`, never a record that certifies a mixed root.
- **The journal is asymmetric.** Under a differing pipeline, a measure-step failure is
  not journalled, because `_record_local_terminal_failure` uses `_local_epoch_ownership`
  (`_cli_execution_strategies.py:145`). A Stage-4 failure in the same invocation is
  journalled. Exit codes are unaffected (both come from invocation counts), but the
  journal is incomplete.
- Fix: in measure mode, have `_record_local_terminal_failure` take `_local_state_lock` as
  Stage 4 does. A per-invocation lease, a non-blocking `.phenotypic/measure.lock`, is a
  cheap real fence (D4).

**M4. `_fail_image`'s republish can crash the shard after journalling.** Where:
`_cli_stage4_worker.py:709-710`. `_republish_table_marker` resolves every artifact
`strict=True`. A missing overlay, or a corrupt store (the very failure being journalled),
raises out of `_BackboneStream._fail` or the commit thread. That kills every image of the
shard, or the local invocation (exit 1 with a traceback). The next round sees the journal
row and treats the image as terminal, so this does not loop. Fix: wrap that republish,
log it, and leave the image journalled.

**M5. The controller's `finalizing` branch can relabel a finished measure pass as
`failed`.** Where: `_cli_staged_controller.py:1037-1043`. It requires
`staged_completion_matches(epoch)`, and a measure pass never writes that marker. A
controller that loaded the state as `finalizing` just before the finalizer's
`deactivate_orchestration` saved `complete` (`deactivate_orchestration` takes no
orchestration lock), and that saw the finalizer inactive afterwards, saves `failed` over
it. The fence still says `closed_by: finalizer`, so the waiter is unaffected, but the
state and the GUI observer are wrong. Fix: also accept the fence closed by the finalizer
for this epoch.

**M6. The measure pipeline's sha has four producers.** They are the submission
(`_cli_staged_slurm.py:834`), `run_measure_step`'s early return, the core
(`_cli_process_single.py:~550`), and `run_stage4` (`_measure_pipeline_sha256`). Each
re-reads the user's pipeline file by absolute path. An edit to that file mid-run splits
them: the core writes a folder under the new sha, while the controller asks for the
recorded one. The result is "not_ready", then dead-task rounds, then C1. Fix: the workers
take `config["pipeline_sha256"]` and refuse when the file's sha differs ("one producer
per derived value").

**M7. Manifest v5 is also written for every forward run.** Where:
`_cli_staged_orchestration.py:306`. Compatibility is fine in the direction that matters:
P6 code reads an in-flight #281/P5 v4 manifest, and `staged_manifest_republishes` reads
it as forward (pinned). But #281/P5 code refuses any manifest P6 writes, forward ones
included, so downgrading or mixing venvs mid-run fails the tasks. The bump is required
for republish manifests, because a v4 reader would classify one as forward. It is not
needed for forward manifests. Optional: write `version: 4` when `republish` is false.

**M8. Every store is hashed once at submission (deviation 7).** This is acceptable, and
fewer passes than the ordinary SLURM array, which calls `file_sha256` twice per store
(`_cli_slurm_array_scripts.py:400-403`). But `StagedSlurmStrategy.execute` hashes
serially (`_cli_staged_slurm.py:768`), whole store trees, on the submitting node. The
local path uses threads (`stage4_entries(n_jobs=)`). Fix: reuse `stage4_entries` for the
manifest.

**M9. Tests.**

- `test_a_carried_stale_flag_is_kept_as_recorded`
  (`tests/unit/sdk_/test_replace_image_tables_carry.py`) cannot tell "kept" from
  "recomputed": the source is `stale: false` and the objmap is unchanged. Plant
  `stale: true` on the source and assert that it survives.
- Two test names now assert acceptance rather than refusal:
  `test_the_slurm_measure_stage4_gate_is_refused_before_the_mutating_half` and
  `test_the_measure_stage4_gate_is_refused_before_the_mutating_half`.
- Missing negative cases:
  - the measure exit-`5` end-to-end test (C1);
  - node-state failure in republish mode (OOM, `Stage4ShardError`), which must journal
    nothing and republish nothing;
  - identity kept on the *failure* republish (`test_a_stage4_failure…` checks only that
    the digest changed);
  - `carry_embeddings` naming a key that `embeddings` also writes;
  - a local measure where the core fails for one store and Stage 4 fails for another,
    which must count 2, not 3;
  - a same-pipeline dead measure step (I1).

**M10. Out of P6 scope, traced but not executed.** On SLURM, `_run_finalize` calls
`aggregate_measurements(pipeline=None)` (`_cli_checkpoint_handler.py:~300`).
`finalize_run` then posts with `_load_pipeline_from_output_dir`, which is
`deliverables/pipeline.pht-pipe.json`, the forward run's persisted pipeline. So a
differing measure pipeline's `post` chain would not be applied on SLURM. This looks
pre-existing in #281's measure finalizer. Worth a follow-up check.

**M11. Deviation 3: the measure finalizer marks the orchestration complete, writing no staged-completion marker.**

**Safe for the measure pass itself**: the trailing controller exits on `complete`
(`_QUIET_PHASES`), and the waiter reads the fence.

**Not a no-op for the forward run.**

- `initialize_orchestration` deletes the forward run's
  `staged_finalization_complete.json` (`_cli_staged_orchestration.py:178`).
- A later plain forward re-run of a `GpuDetector` pipeline therefore takes the
  "Continuing staged GPU finalization" branch (`phenotypicCLI.py:4070-4080`) instead of
  "All images already processed!". On SLURM that means a new submission, exit `4`.
- That pass aggregates the certified, measure-written tables (D6a), so it certifies
  nothing wrong.
- But it rewrites `deliverables/`, the run proof and `metrics.json`. The measure pass's
  `failed` count disappears, because those rows sit under the measure epoch.

**Fix:** keep the forward marker across a measure submission: a `keep_completion_marker`
flag on `initialize_orchestration` for the measure role, or restore its bytes. Its
`epoch` never matches the measure epoch, so nothing the measure pass checks changes.

---

## Checked and sound

- **U8 rule against `replace_image_tables`.** Own run first, then the certified run,
  matched on key plus `params_sha256`. The pairs are validated in the plan before any
  part exists (no `.part` left, pinned). Files are hard-linked and paths re-rooted. The
  same-folder carry links from the live store, after the part's folder is cleared. And
  `stale` is consistent with D7: every objmap writer refreshes every run, and measure
  writes none.
- **Republish keeps the record's identity.** `_republish_table_marker` keeps `work_id`,
  `attempt_id`, `relative_image_path`, `mode`, `provenance` and `source_provenance`
  (pinned locally and on SLURM).
- **Ordering within the pass.** Locally and on SLURM: table (root-last), then
  embeddings (root-last), then the record last, then the signals consumed. On failure:
  the journal, then the record, then the signals dropped. On SLURM a failure republish
  stamps the epoch, so the image reads `done` and is counted through the journal (exit
  `3`, pinned locally).
- **Deviation 6.** Job metadata as the shard's revision source is right, given that the
  state belongs to the forward run. The only open question is which commit (I3, D2).
- **The inline measure path is unchanged.** `defer_embeddings=False`, no entries, and
  `work_identity=None`.
- **The gate lift.** `stage4_supported` admits `measure`, the flipped tests assert the
  new behaviour, and R3's second remedy appears only where Stage 4 can run (pinned over
  the whole matrix).

---

## User decisions

- **D1 (C1).** Should a SLURM measure pass that leaves any forward-published manifest
  image neither republished under its epoch nor journalled exit `5`, rather than `0`?
  *Recommendation: yes.* Exit `0` currently asserts a re-embed, or a re-measure, that did
  not happen.
- **D2 (I3).** Should an unpinned `revision` in `--mode measure` reuse the forward run's
  recorded commit (U5 extended to a re-embed of the same run), or resolve the hub
  afresh?
  - *Reuse:* one commit across the run, and U8's carry stays truly comparable.
  - *Afresh:* picks up hub updates, but a failure leaves two descriptors.
  *Recommendation: reuse by default.* An explicit `revision` remains the way to move.
- **D3 (I1).** Should a killed same-pipeline measure step be detected (a per-epoch
  measure marker), or accepted and documented as a limitation beside U14?
  *Recommendation: detect it.* It costs one marker file per image, and it also closes
  C1's dead-step row.
- **D4 (M3).** Keep local measure at #281's level of fencing, which is none beyond root
  fences and the state lock, or add a per-invocation lease?
  *Recommendation:* the lease, if cheap. It is not blocking for #282.
