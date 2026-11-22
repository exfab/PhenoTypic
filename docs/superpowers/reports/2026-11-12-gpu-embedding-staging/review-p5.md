# #282 P5 deep review: SLURM Stage 4 (controller v2, rounds, worklist, no-detector plan, U5, preflight)

**Range:** `3a62ad6eb..8160629af`, on `feat/gpu-embedding-staging`: b203fa721, 8cae08925 (T1),
050671fa7 (T2), e484bf777, e82dedefa (T3), 8160629af (T4 smoke).
**Method:** I read the code, the tests and the smoke report against spec §4.3, §4.7, §4.8, U3,
U5, U6, U7 and O6, plan p5 and README, `_cli/CLAUDE.md` and `tracked_state.md`. This is
analysis only: **I ran no tests and no jobs.** Every line number is at `8160629af`.

## Verdict

**There is no Critical finding.** The controller's Stage-4 logic is sound and matches §4.3.
Rounds sit on their own counters. The worklist partitions only the retryable set. No sidecar is
submitted. A stall records nothing. A v1 config still drives an in-flight #281 run's jobs.

The four Important findings are about what sits around that logic:
- **I1.** A plain re-run of an in-flight pre-P5 SLURM run is now refused by default (R3).
- **I2.** C14's `stage3_markers_required=False` override is safe today, but it is a latent hazard.
- **I3.** The trailing controller exits FAILED after every `terminal_incomplete` close (smoke F4).
  This is pre-existing, but U7 makes it the normal end of a stall.
- **I4.** U7's "exit 5" is not pinned end to end.

## The four checks the plan asked for

| Check | Result | Evidence |
|---|---|---|
| Sidecar rule | **Pass** | The Stage-4 array is submitted only by the controller, from `_submit_stage4_round` (`_cli_staged_controller.py:423-488`). That happens only after the previous work array is terminal (`run_staged_controller` returns while `active_job_id` is live, `:606-616`). There are no trigger entries and no parallel job. Pinned by `test_no_sidecar_job` and `test_no_sidecar_job_in_any_round`. |
| Stage-2 / Stage-4 counter separation | **Pass in code**; one test cannot fail (M6) | Stage 4 reads and writes only `stage4_round`, `stage4_last_progress_digest` and `stage4_zero_progress_rounds` (`:441-450`). Stage 2's `round`, `last_retryable_digest` and `zero_progress_rounds` are untouched. `initialize_orchestration` zeroes the Stage-4 keys per epoch. |
| U7: no record, no journal row | **Pass in code**; the exit-5 half is unpinned (I4) | The stall path logs, saves the state and submits the finalizer (`:452-462`). Nothing journals. A record with only stages and no failure row reads `"unverified"` (`sdk_/_run_state.py:995-996`), so it counts as unresolved, and the waiter's rule gives 5. |
| A v1 config drives an in-flight #281 run | **Pass** for the run's own jobs; **its re-run is refused** (I1) | See the walk-through below. |

**How a v1 config still works:**
- `_load_controller_config` gives v1 `phases = ["stage1","stage2","stage3"]` (`:101-103`), so
  `_advance_stage4` is unreachable.
- `_classify_stage2` reads `config["detector_slot"]` (`:151`), which v1 carries.
- An old `stage3.sh` passes no `--stage3-role`, which defaults to `staged`
  (`_cli_staged_slurm_worker.py` `main`).
- Workers read deferral from `job_metadata.json` `EMBED_STAGE` (`_cli_staged_slurm_worker.py:224-237`).
  A #281 run has none, so it stays inline.
- The finalizer's `run_terminal_stage` defaults to Stage 3 (`_stages.py:41-43`).
- The new orchestration keys are read with `.get`.
- One caveat on the test: the v1 fixture builds its state with today's `initialize_orchestration`
  (`tests/unit/cli/conftest.py` `_build_controller`). That state carries the Stage-4 keys, which
  a real #281 state lacks (M11).

## Critical

None.

## Important

### I1. P5's gate lift makes R3 refuse the plain re-run of every in-flight pre-P5 SLURM GPU-profile + DINO run (#282 defect, newly reachable; the fix is a user decision, D1)

- **Where:**
  - `_cli_embed_stage.py:224-265` (`refuse_embed_stage_switch`) and `:268-284` (`stage4_supported`, now `run_mode_of(config) == "full"`);
  - the call at `phenotypicCLI.py:3387-3392`;
  - pinned at unit level by `tests/unit/cli/test_cli_embed_stage.py:129-130`, where a recorded `None` with a resolved `gpu` is refused.
- **Scenario:** a #281 run on SLURM with a `GpuDetector` (or `--gpu-slurm`) and a `MeasureDinoEmbedding`.
  1. Its state has no `embed_stage`, which reads `"inline"`. A P4-era run recorded `"inline"`
     through the interim gate.
  2. After upgrading to P5, the documented recovery is "re-run the same command", with the
     default `--embed-stage auto`. A DVC stage does this, as does a user after a walltime.
  3. `auto` now resolves `gpu`. R3 raises a `UsageError`, exit 2: *"this run was started with
     --embed-stage inline … Re-run with --embed-stage inline"*.
  4. Before T2 the gate kept SLURM `auto` inline, so these continuations passed.
  - So the run's already-submitted jobs proceed (v1 works), but its recovery command now fails.
    A DVC `cmd` with no `--embed-stage` breaks on upgrade.
- **Fix:** one of these. They are listed under D1 because they change U3's behaviour.
  - (a) Under `auto`, a continuation adopts the recorded stage, and R3 refuses only an explicit
    `--embed-stage` that conflicts. Resolve `auto` against the recorded value before
    `refuse_embed_stage_switch`, and print one note line.
  - (b) Keep U3 strict, and add a release note plus a `_cli/CLAUDE.md` line in P7 that names
    `--embed-stage inline` for in-flight runs.

  Either way, add a CLI-level test: a recorded-`None` SLURM GpuDetector + DINO state, re-run
  with no `--embed-stage`, and assert the chosen behaviour.

### I2. C14's `stage3_markers_required=False` answers the finalizer's question by inverting a flag that means something else (#282, safe today; fix now, cheap)

**This is the answer to review question 1.**

- **Where:**
  - `_cli_staged_slurm.py:837-841` sets the orchestration state's `stage3_markers_required = False` for a no-detector run.
  - The same run's controller config has `"stage3_markers_required": markers_required`, which is `True` (`:515`).
  - Its stage-3 script still gets `--stage3-markers-required` (`:207`).
  - The reader is `_cli_checkpoint_handler.py:264-281`.
- **Is it safe?** Yes, today, for these reasons:
  - The orchestration-state key has exactly one reader, the finalizer gate at
    `_cli_checkpoint_handler.py:264`. I grepped `src/` and `tests/`; the only other hit is the
    test pinning the override (`test_staged_slurm_scripts_stage4.py:194`).
  - Skipping `reconcile_stage3_publications` there loses nothing for a no-detector run. There
    are no Stage-2 signals to consume (`_cli_staged_resume.py:629-636`). The legacy-parquet
    quarantine (`:637-646`) has no forward writer.
  - Without the override, `staged_detector_slot` would raise on a pipeline with no `GpuDetector`.
- **Does any other reader change behaviour for `single_pass`?** No.
  - The config's key is read only by `_classify_stage2` (`_cli_staged_controller.py:144`), which
    a `["stage3","stage4"]` run never reaches.
  - `run_single_pass_step` takes no `markers_required`.
  - The GUI observer reads `phase`, not this key (`_slurm_observer.py:1158`).
- **The hazard:** the flag's documented meaning is "Stage-3 markers are the completion
  contract". A no-detector Stage-4 run depends on exactly those markers: `stages.stage3` is what
  `stages_1_to_3_done` reads (`_cli_staged_resume.py:193`). The run now carries
  `markers_required: False` in one document and `True` in the other.
  - `sdk_/_run_state.py:625-689` (`staged_image_is_complete`) is the planned F12 consolidation
    target. Its docstring says no caller has migrated yet.
  - With `markers_required=False`, that function returns "complete" on table presence alone.
  - A future change that feeds it this state flag would treat a Stage-4-pending image (table
    present, no embeddings, no record) as finished. That is the "certify with a silent hole"
    that spec §4.2 forbids, and nothing would fail to say so.
  - The comment at `_cli_checkpoint_handler.py:268` ("set only by the staged GPU orchestration,
    so the recorded pipeline has a GpuDetector") is now true only because of the inversion.
- **Fix:**
  - Gate the reconcile on the recorded detector slot instead. `staged_controller.json` already
    carries `detector_slot` if and only if `"stage1"` is in `phases`
    (`_cli_staged_slurm.py:528-538`).
  - Pass that slot to `reconcile_stage3_publications`, rather than re-deriving it with
    `staged_detector_slot(PIPELINE_PATH)`. That is also one producer for the slot, not two.
  - Drop the override at `:837-841` and flip `test_staged_slurm_scripts_stage4.py:194` to assert
    the key's absence or `True`.

### I3. Smoke F4: the trailing controller raises `SlurmGenerationInactiveError` after a `terminal_incomplete` close (pre-existing; recommend fixing in #282)

**This is the answer to review question 4.**

- **Where:** `_cli_staged_controller.py:571` and `:581`. The quiet set is
  `{"complete", "cancelled", "failed"}`. `deactivate_orchestration(..., "terminal_incomplete")`
  (`_cli_checkpoint_handler.py:404`) writes that phase (`_cli_staged_orchestration.py:995,1004`).
- **Is it new in #282?** No. It is the same at the base: `git show 3a62ad6eb:…/_cli_staged_controller.py`
  has the set at `:301` and `:311`, and the base finalizer already closes `"terminal_incomplete"`.
  `git log -S'"terminal_incomplete"'` dates the close to `379acee42` (2026-08-17).
- **Why it matters more now:** U7 makes `terminal_incomplete` the *designed* end of every
  Stage-4 stall, as well as of every run with one terminal failure (smoke attempt 3). So every
  such SLURM run leaves a FAILED controller in `sacct`, which reads as a crash. A
  `sacct`-watching orchestrator, the GUI's failure-state heuristics or a user will misread it.
- **Fix:** add `"terminal_incomplete"` to both sets. Add one test: a state in phase
  `terminal_incomplete` with a fenced epoch, where `run_staged_controller` returns without
  raising.

### I4. U7's exit 5 is not pinned end to end, and the test named for it cannot fail against P5 (#282 test gap)

**This is part of the answer to review question 9.**

- **Where:** `test_staged_controller_stage4.py:202-207` (`test_a_stall_reads_5_to_the_waiter`).
  It calls the pre-existing `decide_wait_outcome` with a hand-fed `unresolved=2`. No P5 code is
  on its path, so it passes whatever the controller, the finalizer or the run state do.
- **What is unpinned:** the chain "stall → finalizer → leftover images have a record with only
  stages, and no journal row → `resolve_run_state` counts them unresolved → waiter 5".
  - I verified the run-state link by reading `sdk_/_run_state.py:995-996`: no valid record and
    no failure gives `"unverified"`.
  - Nothing fails if someone later makes a stages-only record read `failed`, or journals
    leftovers in the finalizer. Then exit 3 would silently replace exit 5.
- **Fix:** after `test_two_unchanged_rounds_finalize_and_record_nothing`'s three rounds, assert:
  - `resolve_run_state(out, depth="deep").diagnostics` has `unresolved == 2` and `failed == 0`;
  - and, through `_cli_wait`'s seams, that the outcome is `ENDED_UNPUBLISHED`. Use the real
    `_run_snapshot`, not a literal.

## Minor

- **M1. Smoke F7: `--dry-run` says "SLURM (autonomous)" for staged runs.** This is pre-existing:
  the label dates from `ac0ba9537` and `4ac129dd7`, and it is wrong for GpuDetector runs too.
  #282 adds a second mislabelled path.
  - Where: `_cli_interactive.py:49`.
  - Fix: `'SLURM (staged controller)' if uses_staged_controller(config) else 'SLURM (autonomous)'`.
    This is one line, so it can land in #282.
- **M2. Smoke F1: `phenotypic-wait` exits before the trailing controller has run.** This is a
  pre-existing #281 waiter behaviour. A closed fence makes `live_slurm_job_ids` answer empty
  without asking the scheduler (`_cli/CLAUDE.md`, "How `live_slurm_job_ids` answers").
  - It is not a #282 defect.
  - Docs now: a line beside `phenotypic-wait` saying that teardown must wait for every ledgered
    job, as `live_smoke.sh cleanup` does.
  - The behavioural fix is D4.
- **M3. Smoke F5: no staged event carries a SLURM job or task id.** This is pre-existing in
  every stage: `stage_event` never passes them, though `append_event` takes them
  (`_cli_update_state.py:76`).
  - Not a #282 defect.
  - A cheap follow-up: default both from `SLURM_ARRAY_JOB_ID`/`SLURM_JOB_ID` and
    `SLURM_ARRAY_TASK_ID` inside `stage_event`. That would make "not re-embedded in round 2" a
    direct read, as F5 says.
- **M4. Smoke det a3: Stage-2 signals are retained for Stage-3 failures.** This is pre-existing,
  and the docs need it.
  - `reconcile_stage3_publications` consumes the token and raw array only for an image that is
    complete or has a Stage-3 marker (`_cli_staged_resume.py:629-636`). A Stage-3 scientific
    failure therefore keeps both halves. That matches the smoke's 44 files for 22 failures.
  - So `--retry-failures` replays Stage 3 from the raw array, with no GPU.
  - It is not stated in `_cli/CLAUDE.md`. Add one sentence under "Stage-2 signals …" in P7.
- **M5. Smoke F2: an OOM-killed Stage-3 (and Stage-1) task is never resubmitted.** This is
  pre-existing.
  - `_submit_next_stage3_or_finalizer` submits each chunk once (`_cli_staged_controller.py:287-309`).
    The autonomous drip-feed path also has no retry: grepping
    `_cli_execution_strategies.py`, `sdk_/slurm/_dispatcher.py` and `_cli_slurm_submission.py`
    for requeue, resubmit or retry finds nothing.
  - #282 inherits it on the no-detector path, unchanged from the inline path it replaces.
  - In `_classify_stage4` such an image is `not_ready`, so terminal for the run, and nothing is
    logged. Exit 5 is the **correct** code (review question 9). C10 defines 5 as "accepted
    images never attempted; re-run the same command", and a re-run does retry those images,
    since there is no terminal row.
  - But the operator gets no line saying four images never finished Stage 3, and a host-OOM
    image will OOM again on that re-run.
  - Fix: in `_advance_stage4` (`:510-512`), log how many terminal entries are `not_ready` and
    how many are this epoch's failures. Policy is D5.
- **M6. `test_a_stage2_stall_does_not_count_against_stage4` cannot fail under shared counters.**
  - Where: `test_staged_controller_stage4.py:120-128`.
  - It plants `zero_progress_rounds=2` with no matching digest. Under a shared-counter
    implementation, the digest mismatch resets the count to 0, and the test still sees two
    Stage-4 rounds.
  - Fix: also plant `last_retryable_digest` equal to the Stage-4 digest the first transition
    will compute, using `_stage4_progress_digest(config, entries)[0]`. Then shared counters
    would stall at once.
- **M7. `test_the_loader_workers_come_from_the_allocation` uses one image.**
  - Where: `test_stage4_slurm_worker.py:65-80`.
  - `resolve_local_worker_count(-1, 1)` is 1 whatever the allocation. The test does catch a bare
    `SLURM_CPUS_PER_TASK` read (5 − 1 = 4), but not an affinity-blind `os.cpu_count()`.
  - Fix: use 3 or more entries, and patch `os.sched_getaffinity` to a set smaller than both
    `cpu_count` and the environment variable.
- **M8. Stale test text after the gate lift.**
  - `test_staged_controller_stage4.py:421-426` says phase `stage4` is "unreachable today … the
    interim gate never resolves".
  - `:491-502` monkeypatches `stage4_supported` to `True` "until P5 T2". After T2 the patch hides
    whether the real lifted gate admits the run.
  - Fix: drop the patch and the stale sentences.
- **M9. `PF-EMBED-SHARD-AMORTIZE` counts every input, not the images left to embed.**
  - Where: `_cli_preflight.py:1202`.
  - A continuation with 10 images left over 4 shards stays silent if the run had 4,000.
  - Fix: use the continuation's remaining count when it is available, or say "inputs" in the
    message.
- **M10. U5's offline fallback is reused for ever, and offline-with-a-resolved-commit fails
  silently into a stall.**
  - `pin_embed_revisions` reuses a recorded `None` (`_cli_embed_stage.py:624`). So a run first
    submitted offline never pins, even when a continuation is online.
  - Also: a submitter that is online resolves `main`'s current sha. Shards with
    `HF_HUB_OFFLINE=1` (spec §4.8 recommends it) and an older pre-staged snapshot cannot load
    that sha. Every shard then fails in every round, the digest is unchanged, and the run stalls
    at exit 5. Only the shard logs say why.
  - Fix:
    - (a) Re-resolve a recorded `None` on a continuation; a `None` was never a pin.
    - (b) In the `PF-EMBED-REVISION-UNPINNED` docs row, say that pinning `revision=` is the only
      safe pairing with pre-staged offline weights.
- **M11. The fake scheduler does not model these real behaviours.**
  - **A TIMEOUT mid-commit, at the controller level.** The store is promoted with embeddings,
    there is no record, and the signals are still present. It should classify `pending`, go back
    into the worklist, and publish only, with no re-embedding. P4 T4 covers the worker half
    locally, but no controller test plants this state.
  - **`scheduler_job_is_active` answering `None`** for the active Stage-4 array. The controller
    must wait (`:610-616`), and no Stage-4 test covers it.
  - **A controller that dies after `sbatch`, before saving the state,** for a `stage4-round-<n>`
    token. See M12.
  - **The v1 fixture's state.** It is built by today's `initialize_orchestration`, so it carries
    the Stage-4 keys; a real #281 state does not. Pop them in `v1_controller_config`.
- **M12. The worklist is one file, rewritten in every round, with the same name under the same
  token.**
  - Where: `write_stage4_worklist` (`_cli_staged_orchestration.py:314-349`), called before
    `submit_with_intent` (`_cli_staged_controller.py:464-479`).
  - Scenario:
    1. Controller A writes round n's worklist and `sbatch` succeeds, but A dies before saving
       `active_job_id`.
    2. Successor B sees no active job and re-classifies. Some of A's shards may already have
       committed.
    3. B rewrites the worklist with a different partition, and `submit_with_intent` returns A's
       job from the intent row or comment.
    4. A's not-yet-started tasks then read B's partition, while its started tasks hold A's. One
       image can be in two running shards.
  - The damage is limited: the loser meets `StoreRootChangedError`, and signal writes are atomic.
    But GPU time is spent twice, and the one-partition-per-round invariant breaks.
  - Fix: when the ledger already holds an `intent` or `submitted` row for `stage4-round-<n>`
    under this epoch, keep the existing worklist and skip the rewrite.
- **M13. Empty shards still allocate a GPU each.**
  - `stage4.sh` is always `--array=0-(n_shards-1)` (`_cli_staged_slurm.py` `generate_staged_scripts`,
    stage-4 branch).
  - A late round with 1 image and `--gpu-shards 8` queues 8 GPU jobs, and 7 of them exit 0.
    On `exfab`, with 2 GPUs, those empty tasks queue ahead of the real one.
  - Fix: pass `--array=0-(min(n_shards, len(retryable)) - 1)` on the `sbatch` command line when
    submitting the round. Command-line options override `#SBATCH`.
- **M14. A stall caused by too short a walltime gives advice that cannot work.**
  - If cold load plus the first image exceeds the GPU `time=`, every round writes nothing, the
    run stalls and exits 5, and "re-run the same command" repeats it.
  - The only message is a `logger.warning` in the controller log (`:452-457`).
  - Fix: word the warning to name `--gpu-slurm time=` and the shard logs, and mirror one line
    into the finalizer log, which exit 5 already points at.
- **M15. The recorded revisions have two readers with different precedence.**
  - `_cli_staged_slurm_worker.py:801-807` reads `job_metadata.json` only. `recorded_embed_revisions`
    (`_cli_embed_stage.py:587-609`) reads `processing_state.json`, then job metadata.
  - They agree today, because both files are written from `config.embed_revisions` in one
    invocation. Still, the worker should call `recorded_embed_revisions`, so there is one reader.
- **M16. Docs debt P7 must carry.**
  - `_cli/CLAUDE.md`'s strategy table still defines a "GPU run" as one with a `GpuDetector`.
  - It does not describe config v2, `phases`, the worklist, `stage4.sh`'s prelude, Stage-4
    rounds and their counters, or the U7 stall.
  - "Walltime survival" covers Stage 2 only.
  - None of M2, M4 or I3 is written down.

## Smoke findings, classified

| Item | Class | Action |
|---|---|---|
| **C14's markers flag (question 1)** | #282, safe today | I2: gate the finalizer on the recorded `detector_slot` and drop the override |
| **F1:** the waiter exits before the trailing controller | Pre-existing (#281 waiter) | Docs now (M2); behaviour is D4 |
| **F2:** Stage-3 OOM not resubmitted | Pre-existing (staged and autonomous) | Log a count (M5); policy is D5 |
| **F3:** sprawling crops make prep about 100× slower | Not a defect; efficiency | D3 (round cap); a crop-area preflight is a follow-up |
| **F4:** the trailing controller exits 1 | Pre-existing (`379acee42`), made routine by U7 | I3: fix in #282 |
| **F5:** no SLURM ids on events | Pre-existing | Follow-up (M3) |
| **F6:** cold load 40–300 s vs 750 s | #282 constant; the model also ignores the loader | D2; Minor otherwise |
| **F7:** the dry-run label | Pre-existing, extended by #282 | One-line fix (M1) |
| **F8:** inputs hashed about three times | Pre-existing (P3–P4 review) | Already listed there |
| **det a3:** 44 Stage-2 files left | Pre-existing, by design (`--retry-failures` replay) | Docs (M4) |
| **The 4 OOM images exit 5 (question 9)** | Correct code under C10 | Unpinned (I4); add a remedy line (M5) |

**More on F6.** `_COLD_LOAD_S = 750` (`_cli_preflight.py:1113`) is spec §2's cold first call,
with imports and weights from GPFS. The smoke measured dinov2-small with cached weights and
`HF_HUB_OFFLINE=1`; both are real numbers, for different conditions.

The larger modelling error is in the denominator. `_WARM_GPU_S_PER_IMAGE` (`:1116`) counts
GPU-bound seconds only: 1.71 s for base. The smoke's steady state was loader-bound, at 2.74–4.60 s
per image with one worker.

With the smoke's own numbers, the load took roughly 7–50% of each shard (for example 297.9 s
against about 189 × 4.6 s on gpu09). So the "more than half" warning that fired at 200 per shard
was a false positive.

If the constant changes, four things change in lockstep: spec §2's note,
`logic_validation_scripts/…/shard_amortization.py` (`COLD_LOAD_S`, and the 439/297 checks), the
docs row in `cli_batch_processing.md`, and `test_preflight_embed_stage.py:151,186-187`. The test
pins `"439"` and `"750 s"` and holds the script and the constant together.

## User decisions

- **D1 (U3, from I1).** Should `--embed-stage auto` on a continuation adopt the recorded
  `embed_stage`, refusing only an explicit conflicting flag? Or does U3 stay strict, so that
  every in-flight pre-P5 SLURM GPU-profile + DINO run needs `--embed-stage inline` on its
  re-run, with a release note?
  - **Recommendation: adopt under `auto`.** U3's purpose, no mixed embeddings, still holds,
    because the recorded stage is the one used.
- **D2 (O6 and spec §2, from F6).** What should `PF-EMBED-SHARD-AMORTIZE` assume?
  - (a) Keep 750 s as an explicit worst case, and reword the message to "up to about 750 s cold;
    40–300 s measured with pre-staged weights".
  - (b) Re-derive the threshold from a measured cached-load figure, and divide by
    `max(GPU s/image, loader s/image ÷ workers)`.
  - (c) Drop the warning.
  - **Recommendation: (b).** The warning fired falsely on the one live run we have.
- **D3 (U7, from F3).** Should Stage 4 have an absolute cap on rounds, or a per-image
  "terminal after N node-state failures" rule? Under U7 two cases never end:
  - an image whose prep is ~35 min makes progress in every round, so the run goes on for days;
  - a "poison" image that host-OOMs its shard in every round reaches exit 5 on every re-run, for
    ever, and "re-run the same command" never helps it.
- **D4 (from F1).** Should `phenotypic-wait` keep polling this generation's ledgered jobs after
  the fence closes, until they are terminal, so a waiter-triggered teardown cannot race the
  trailing controller? Or is that docs only? This is a #281 waiter change, not #282 scope.
- **D5 (from F2).** Should the controller re-classify and resubmit Stage-1 and Stage-3 chunks
  whose tasks died without a record or journal row (OOM, node failure), as Stage 2 and Stage 4
  do? Today the remedy is a plain re-run (exit 5). That works for a node failure, but not for a
  host OOM unless `mem_gb` is raised. This is a pre-existing gap and outside #282's scope.

## Files read

- `src/phenotypic/_cli/_cli_staged_controller.py`, `_cli_staged_orchestration.py`,
  `_cli_staged_slurm.py`, `_cli_staged_slurm_worker.py`, `_cli_embed_stage.py`,
  `_cli_preflight.py`, `_cli_execution_strategies.py`, `_cli_state_management.py`,
  `_cli_stage4_signal.py`, `_cli_stage4_worker.py`, `_cli_checkpoint_handler.py`,
  `_cli_staged_resume.py`, `_cli_interactive.py`, `phenotypicCLI.py`, `_stages.py`;
- `src/phenotypic/sdk_/_run_state.py`;
- `src/phenotypic/_gui/run_console/_slurm_observer.py`;
- `tests/unit/cli/test_staged_controller_stage4.py`, `test_staged_slurm_scripts_stage4.py`,
  `test_stage4_slurm_worker.py`, `test_cli_embed_stage.py`, `test_cli_preflight_ordering.py`,
  `conftest.py`; `tests/integration/cli/test_embed_stage_refusals.py`;
- the spec, plan p5, the README, `live-smoke.md`, `_cli/CLAUDE.md` and `tracked_state.md`.
