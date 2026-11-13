# Review: P2 (DVC run contract) of #281

**Scope.** `git diff a47e92b1b 9d86bd565 -- src tests docs/source`: Tasks 1, 2,
2a and 3 to 6, in commits `11ceb5255` to `9d86bd565`. I checked them against
spec §C1–C14 and `plans/…/p2-dvc-run-contract.md`. This is a read-only review:
I edited no worktree file except this report. I wrote one scratch DVC config
in the session scratchpad to probe `find_dvc_cache_type`.

## Verdict

**Approve after I1–I4. I5 and I6 can wait for P3 if they are recorded.**
There are no critical defects. The core contracts hold:

- **No SLURM submission exits 0.** Every submit site I found exits `4`:
  full, measure, staged, process, SLURM recompile, the forward run with nothing
  left whose only work is a SLURM recompile, a submitted migrate, and a SLURM
  migrate `--dry-run`. Every site that exits `0` submitted nothing. See the
  exit-site table below.
- **The waiter never publishes.** `_cli_wait.py` imports only readers, and
  `subprocess.run` of the child is its only side effect. Within one poll it
  reads the scheduler before the run state (`_cli_wait.py:273-274`).
- **The generation binding works for the cases the spec names.** Those are a
  dead finalizer over an active fence, a measure resubmission whose finalizer
  died, a cancelled resubmission, and an unstamped proof. Each is pinned
  end-to-end through the real seams in `test_cli_wait.py:465-568`.
- **A close can never relabel another close.** `_deactivate_generation_locked`
  writes `closed_by` only on the transition, so a later cancel cannot erase a
  finalizer's close, and a late finalizer cannot relabel a cancel. That holds
  by construction.
- **`retire_dead_lifecycle_fence` never closes a fence whose jobs merely cannot
  be queried.** `LiveSlurmJobs` is truthy for `unverified` and `unresolved`,
  and `test_a_fence_that_may_still_own_a_job_is_kept` pins both answers. For
  the one hole that remains (a fence with **no** ledger rows yet), see I4.
- **`metrics.json` and `schema.json` are canonical.** Keys are sorted, the JSON
  is UTF-8 with a trailing newline, and neither file holds a clock or host
  value. The aggregate proof declares both, and `require_run_summaries=True`
  enforces them on the `finalize_run` path only.
- **`find_dvc_cache_type` parses DVC's own indented config.** I checked a
  scratch `.dvc/config` holding `[core]`, `['remote "x"']` and an indented
  `type = hardlink,symlink`. It returned `'hardlink,symlink'`.

**Tests run** (`-p no:randomly -o addopts= -m "not slow"`, `QT_QPA_PLATFORM=offscreen`, serial):

- `test_cli_wait.py`, `test_preflight_dvc_cache.py`, `test_run_metrics.py`:
  **91 passed in 51 s**.
- `test_deliverables_byte_stable.py`, `test_cli_exit_codes.py`,
  `test_dvc_outs_at_submission.py`, `test_dvc_layout_guards.py`,
  `test_run_summaries.py`, `test_output_schema.py`: **45 passed in 182 s**.

Everything shipped is green. The findings below are behaviours the tests do
not exercise.

**Counts:** Critical 0 · Important 6 · Minor 11 · Design forks 5.

---

## Exit-site audit

This table covers every process exit reachable from `python -m phenotypic`
(`phenotypicCLI.py`, plus `handle_migrate_mode`) and from `phenotypic-wait`. It
leaves out the internal SLURM worker CLIs (`_cli_process_single`, the staged
worker, `_cli_recompile_worker`): no submitting process returns their codes,
and their `0`/`1` are unchanged.

### `python -m phenotypic`

| Site | Condition | Code | Matches C10? |
|---|---|---|---|
| Click parsing (missing/unknown option, `--metadata` suffix callback `:1012`, `--revert` outside migrate) | usage | 2 | yes |
| `_refuse_extra_positionals` `:1869`; mode/option combos `:2380-2506`, `:2593-2601`, `:2690-2733`, `:2893`, `:2910`, `:2998`; GPU placement / grid preset / reference-table / input-under-output refusals | `click.UsageError` | 2 | yes |
| `:2519`, `:2575` (`--slurm`/`--gpu-slurm` `BadParameter` caught), `:2546`, `:2588` (bad `time=`), `:2784` (`--ext` `BadParameter` caught) | option-value error | **1** | defensible: a usage-class error converted to 1 (M1) |
| `:2367`, `:2638`, `:3295`, `:3377` `click.ClickException`; outer handler `:4098-4100` | runtime refusal | 1 | yes |
| `_refuse_live_slurm_run` `:973` | a live/unverified/unresolved job owns `--output` | 1 | yes (documented as a refusal, not a usage error) |
| `:2802` `error_exit` (metadata unreadable), `:2941-3170` (non-empty output, unreadable/incompatible state, scan error, config validation, pipeline load, preflight errors, display error), `:3313`, `:3327`, `:3368` | config/runtime error | 1 | yes |
| migrate, local | `report.ok` / not | 0 / 1 | yes |
| migrate, SLURM attempt or SLURM `--dry-run` (`_cli_migrate.py:2594`) | submitted | 4 | yes |
| migrate `--revert` (`_cli_migrate.py:2189`, `:2193`) | local | 0 / 1 (bare ints) | value yes, style no (M2) |
| migrate refusals (`_cli_migrate.py` `ClickException` ×9, `UsageError` ×1) | refusal | 1 / 2 | yes |
| recompile, SLURM, chain submitted (`:4476`) | submitted | 4 | yes |
| recompile, SLURM, nothing needs SLURM → local fallback (`:4364`, `:4390`) | ran locally | **0 always** | **no when the run has terminal failures (I2)** |
| recompile, local (`:2676`) | ran locally | **0 always** | **no when the run has terminal failures (I2)** |
| recompile `error_exit` `:4311`, `:4406`, `:4710` | no datasets / attempt init failed | 1 | yes |
| forward/process `--dry-run` `:3184` | nothing written or submitted | 0 | yes |
| staged nothing left `:3600` | already processed | 0 / 3 (run failure count) | yes |
| forward SLURM nothing left + new metadata or refresh `:3636` | `_handle_recompile_slurm` | 4, or 0 on its local fallback | 4 yes; fallback is I2 |
| forward nothing left `:3666` (local **or** SLURM) | already processed | 0 / 3 | yes. Nothing is submitted, and the launch form returns it unpolled |
| process SLURM `:3879`; full/measure/staged SLURM `:3896` | submitted | 4 | yes |
| SLURM strategy returned `submitted=False` `:3861` | `RuntimeError` → outer handler | 1 | yes |
| process local `:3927` | finished | 0 / 3 | yes |
| staged **local**, finalization incomplete `:4062` | includes "only terminal failures" | **1** | **no (I3)** |
| full/measure local `:4088` | finished | 0 / 3 (`_run_failure_count`) | yes |
| `KeyboardInterrupt` `:4092` | interrupted | 130 | yes |
| `UnstageableGpuDetectorError` backstop `:4104` | refusal | 2 | yes |
| any other exception `:4110` | crash | 1 | yes |

### `phenotypic-wait`

| Site | Condition | Code | Matches C3/C10? |
|---|---|---|---|
| Click parsing (`--output` missing, `--poll-seconds 0`, bad float) | usage | 2 | yes |
| wait-only form, `--output` not a directory (`_cli_wait.py:505`) | usage | 2 | yes |
| positional not introduced by `--` (`:490`) | usage | 2 | yes |
| `check_submit_argv` (`:378`, `:384`, `:390`) | empty / migrate / other output | 2 | yes |
| child exit ≠ 4 (`:498-504`) | passthrough | child's code | yes |
| child killed by signal N (`:331`) | passthrough | 128+N | yes |
| child program missing / not executable (`:325-330`) | passthrough | 127 / 126 | yes (beyond spec, documented) |
| `wait_for_run` (`:510-513`) | run ended | 0 / 3 / 5 / 6 | yes, except I1 and I2 |
| **`KeyboardInterrupt`** during the wait (Click's standalone `Abort`) | interrupted | **1** | **ambiguous with `ERROR` (M4)** |
| **`KeyboardInterrupt`** during `_run_submit` | `subprocess.run` SIGKILLs the child 0.25 s later, then Abort | **1** | **M4** |
| an uncaught exception in a seam (e.g. `PermissionError` from `squeue`, a malformed fence `KeyError`) | traceback | **1** | **ambiguous with the run's `ERROR` (M5)** |
| SIGTERM to the waiter (wrapper job walltime / `scancel`) | default action | 143 | outside the contract. See fork F2 |

---

## Critical

None.

## Important

### I1. `COMPLETE_WITH_FAILURES` is returned while some accepted images never ran

**Where:** `_cli_wait.py:207-208` (`decide_wait_outcome`), together with
`_run_state.py:1841-1842` (`completion = "failed"` when **any** image is
`failed`, whatever the others are) and `_cli_checkpoint_handler.py:492-500`
(the ordinary finalizer closes the fence `closed_by: "finalizer"` on **any**
"not complete" run, so `_published` is true).

**Scenario.** A SLURM full run over 1,000 plates has one plate that fails
terminally (a `NoObjectsError` plate). One array task is OOM-killed, or a
node dies before its images write a record, so those images have neither a
success nor a failure record and read `unverified`. The finalizer runs, finds
the run incomplete, and closes the fence as `finalizer`. With no job alive,
the waiter computes `completion == "failed"` and `published == True`, and
exits **3**. A DVC orchestrator reads that as "complete with failures", even
though part of the inventory was never processed and a plain re-run would
process it. The process-mode fallback (`published or process_run`) has the
same hole, and it is wider: a **cancelled** process run (`closed_by` unset)
with one recorded failure also exits 3, because `process_run` alone licenses
the code.

A local run cannot produce this state, since every image it attempts gets a
record. So the waiter is not returning "the code the run would have returned
locally" (spec C3).

**Fix.** Make `COMPLETE_WITH_FAILURES` require that every accepted image is
resolved. Have `_completion` return the `RunState`, or add a seam returning
`diagnostics`, and require `verified + failed == accepted` for code 3. Use
`ENDED_UNPUBLISHED` otherwise. For the process fallback, also require
`published`, or at least that the fence was not cancelled. The test that
would have caught it is a `RunState` with `{failed: 1, unverified: 1}` and a
finalizer-closed fence, which must give 5. `test_a_finalized_resubmission_with_a_failure_exits_3`
uses a fully resolved tree, so it cannot fail on this bug. See fork F5 for
whether a dedicated code is wanted.

### I2. A local recompile exits 0 over a run with terminal failures, while the waiter returns 3 for the same run on SLURM

**Where:** `phenotypicCLI.py:2668-2677` (`recompile_exit = RunExitCode.COMPLETE`)
and `_handle_recompile_slurm`'s two local fallbacks `:4364`, `:4390`. The
latter is also reached from the forward "nothing left, new metadata" route at
`:3636`.

**Scenario.** A run finished with one terminal failure (it exited 3), and the
user runs `--mode recompile` with corrected metadata.

- **Locally** it exits **0**.
- **On SLURM** with the launch form, the recompile finalizer stamps and closes
  as `finalizer`, `resolve_run_state` still reads `failed`, and the waiter
  exits **3**.

The same run therefore gets two codes, depending on where it ran. That breaks
C3's "the code the run would have returned locally" and C10's "0 = no terminal
failures". The forward path already fixed this class of bug for re-runs with
`_run_failure_count` (`test_rerunning_a_run_that_exited_3_still_exits_3`).
Recompile was left out because the plan's T1 table prescribed `COMPLETE` for
it.

**Fix.** At all three sites, use `_finished_run_exit(_run_failure_count(output_dir))`.
Add a `test_a_local_recompile_of_a_run_with_a_failure_exits_3` beside
`test_a_local_recompile_exits_0` (`test_cli_exit_codes.py:155`). Because the
plan table said otherwise, this is also listed as fork F4.

### I3. A local staged GPU run with one terminal failure exits 1, forever

**Where:** `phenotypicCLI.py:4031-4062`. The `STAGED FINALIZATION FAILED`
branch fires whenever `results.total_failed != 0`, or when
`full_stage3_complete` is false, and a terminally failed image never gets a
Stage-3 marker.

**Scenario.** A local staged run (a `GpuDetector` pipeline) has one plate that
fails terminally.

- **The first run** exits **1** (`ERROR`) and prints "Run the same command
  again to continue".
- **The re-run** finds no staged completion. It takes "Continuing staged GPU
  finalization (image stages complete)" (`:3584-3593`), and `results.total_failed`
  is 0 this time, but `full_stage3_complete` is still false for the failed
  plate. So it exits **1** again, on every re-run.
- **The same pipeline on SLURM** under the waiter exits **3**: the staged
  finalizer closes the fence `terminal_incomplete` as `finalizer` and
  completion reads `failed`.

An orchestrator therefore sees a permanent `ERROR` for a run that finished
with failures, and the message tells the user to do something that cannot
help. `_cli/CLAUDE.md` documents this as pre-#281 behaviour kept unchanged,
and the plan table row `:3894` says the same, but it contradicts C10 for the
one mode most likely to run under DVC on a GPU node.

**Fix.** Separate "finalization did not publish" from "every unfinished image
is a terminal failure". When `finalization_succeeded` holds, and every image
lacking a Stage-3 marker has a current terminal failure record, exit
`_finished_run_exit(run_failed)`. Do not mark the run staged-complete then,
just as the SLURM path closes `terminal_incomplete`. Keep `ERROR` for a real
publication failure. This is listed as fork F3 because the plan kept the old
behaviour on purpose.

### I4. `retire_dead_lifecycle_fence` closes the fence of a submission still being prepared

**Where:** `_cli_staged_orchestration.py:776-820`, called at
`phenotypicCLI.py:3193` and `:4523`. `live_slurm_job_ids`
(`_cli_staged_orchestration.py:726-766`) returns an empty `LiveSlurmJobs` for
an **active fence with no ledger rows**.

**Scenario.** Invocation A publishes its fence: `initialize_slurm_lifecycle`
at `_cli_execution_strategies.py:1076`, or `initialize_orchestration` for
staged runs. It is now writing job metadata and chunk scripts, and has not
written its first `intent` row. Invocation B, a GUI re-click, a DVC retry, or
a second shell on the same `--output`, passes `_refuse_live_slurm_run` (no
rows, so nothing is live) and reaches the mutating half. There
`_retire_dead_fence` closes A's fence. A's `submit_with_lifecycle` then fails
with `SlurmGenerationInactiveError`, so A dies after its mutating half began.

Before T5, B hit `initialize_slurm_lifecycle`'s refusal and A was unharmed.
The window is short on the ordinary path. On the staged path it covers
manifest and controller-state writing, which grows with the image count.

**Fix.** Do not retire a generation that has **no** ledger rows for it. A
submitter that died before its first `intent` row leaves exactly that state,
which is rare and recoverable by hand, and it can keep the old refusal with a
message naming the fence. Alternatively, hold `lifecycle_lock_path` from fence
publication through the first `intent` row, and take it non-blocking in
`retire`. Add a test: an active fence with zero rows must not be retired.

### I5. On SLURM, a measure-mode failure is invisible to the waiter and to `metrics.json`

**Where:** `_cli_failure_tracker.py:444-453` (measure work ids carry mode
`"measure"`), `_run_state.py:1824-1830` (the inventory is the forward run's
`work_ids`), `_cli_finalize_run.py:574-580` (`failed` is the journal ∩
the accepted work ids).

**Scenario.** `--mode measure --slurm` over a finished run, where one image's
re-measure fails terminally. Measure keeps no run state, so its failure record
is keyed by a measure-mode work id outside the forward inventory. That has
three effects:

- **The waiter:** `resolve_run_state` still reads `complete`, from the old run
  proof, because the image's record and store are untouched, and the new
  aggregate is stamped with this generation. The waiter exits **0**.
- **A local measure** exits **3**, because `failed_now > 0`
  (`phenotypicCLI.py:4067`).
- **`metrics.json`** says `failed: 0` and `succeeded: accepted`. The aggregate
  also still carries that image's **old** measurements, which was true before
  this change but is now certified by a `0`.

Confidence is medium: I traced the work-id domains but did not run a SLURM
measure resubmission end to end.

**Fix.** Do one of these. Have the measure finalizer stamp its own failure
count, for example `measure_failed` in the aggregate proof or in
`metrics.json`, which the waiter reads under the generation binding. Or have
`_published` return the count of terminal failures recorded under this
generation. A test that resolves the question: a `finished_tiny_run`, a
measure-mode terminal record added for one image, a stamped aggregate. The
waiter must give 3.

### I6. In a DVC stage, any terminal failure fails the stage permanently, and the page offers no way out

**Where:** `docs/source/how_to/pages/dvc.md:328-329`, which states that
`3` fails the stage, and C10.

**Scenario.** A routine run with a single `NoObjectsError` plate exits 3. DVC
treats any non-zero exit as a failed stage: it does not write `dvc.lock`, and
`dvc repro` re-runs the stage next time. The continuation then exits 3 again
("All images already processed" with failures). So a dataset containing one
empty plate can never reproduce cleanly. The page notes that the stage fails
and stops there. A user cannot add `|| test $? -eq 3` to the launch form
without a shell wrapper, and a naive `; test $? -eq 0 -o $? -eq 3` is wrong,
because `$?` is reset by the first test.

**Fix (doc, minimum).** Give a copy-ready, correct idiom:
`cmd: sh -c 'phenotypic-wait --output out -- python -m phenotypic …; rc=$?; [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ]'`.
Say what each choice costs. The real decision is fork F1.

---

## Minor

- **M1.** `phenotypicCLI.py:2517-2519`, `:2573-2575`, `:2782-2784` catch a
  `click.BadParameter` (exit 2) and exit **1**. `:2546`/`:2588` likewise exit 1
  for a bad `time=`. This predates P2, but the exit-code contract now
  documents `2` as "a refused option", so a malformed `--slurm` exiting 1
  contradicts the table in `_cli/CLAUDE.md` and `dvc.md:318`. Either re-raise
  as `click.UsageError`, or note in the table that option *values* checked
  after parsing exit 1.
- **M2.** `_run_migration_revert` returns bare `0`/`1`
  (`_cli_migrate.py:2189`, `:2193`). `_cli/CLAUDE.md` says "use the enum member
  at every exit site, never a bare literal".
- **M3.** `_cli_wait._option_values` (`_cli_wait.py:334-353`) scans **every**
  token after `argv[0]`, wrappers included. A wrapper that has its own `-o`,
  such as `srun -o slurm.log python -m phenotypic -o out …`, is refused with
  exit 2, because `slurm.log` ≠ `out`. Also, `python -m phenotypic` makes
  `phenotypic` a "mode value". That is harmless, but it shows the parser is
  not aware of the program. Document that the check reads every token, or
  scan only after the token `phenotypic` / `-m phenotypic`.
- **M4.** Ctrl-C, or the SIGINT DVC sends on interrupt, exits the waiter with
  Click's `Abort`, code **1**. During `_run_submit`, `subprocess.run` also
  SIGKILLs the child `_sigint_wait_secs` (0.25 s) after the interrupt, which
  can land between the child's `sbatch` and its `submitted` ledger row. Comment
  recovery covers that row, but the child's own `KeyboardInterrupt` → 130 path
  is cut short. In `_run_submit`, catch `KeyboardInterrupt`, `wait()` for the
  child, and return its code. Map an interrupted wait to 130, the CLI's own
  interrupt code, rather than 1.
- **M5.** An exception inside a poll ends days of waiting with a traceback and
  exit **1**, indistinguishable from the run's `ERROR`. Examples are an
  `OSError` other than `FileNotFoundError` from `squeue`, which
  `scheduler_job_is_active` does not catch (`_cli_staged_orchestration.py:650-659`),
  and a fence missing `generation` (`_cli_wait.py:87`). Treat a poll exception
  as "keep waiting", echoing the reason, with a bounded retry count, or give it
  a distinct code.
- **M6.** Each poll runs one `squeue`, and for a purged job one `sacct`, for
  **every** ledgered job of the generation that has no terminal row
  (`_unterminated_ledger_job_ids`). Ordinary chunks and dispatchers never get
  terminal rows. A long drip-feed chain therefore costs dozens of scheduler
  calls per minute, for days. The waiter may keep an in-memory set of job ids
  already answered `False`, since terminal is final. That writes nothing and
  changes no verdict.
- **M7.** **C13 restored ledger with purged accounting.** A `dvc checkout` of
  an old revision that restores an `active` fence whose job ids SLURM has
  since purged (past `MinJobAge` and `PurgeJobAfter`) makes `squeue` report
  "Invalid job id", and `sacct` return no rows. `scheduler_job_is_active`
  returns `None`, so the job is `unverified` forever. The guard then refuses
  every run, and there is no override. `test_restored_liveness_state_fails_safe`
  monkeypatches `scheduler_job_is_active → False`, so it cannot see this. It
  is fail-safe, but it is a dead end. Document the manual recovery, deleting
  `.phenotypic/slurm_lifecycle.json` after checking `sacct`, in `dvc.md`'s C13
  paragraph, or fold it into F2.
- **M8.** **`dvc.md` plot column.** `dvc.md:63` plots `x: Metadata_ImageFile`.
  No schema member spells `ImageFile` (`schema/_metadata/_image.py` has
  `ImageName`). The only `Metadata_ImageFile` in `src/` is a legacy constant
  in the GUI. **I confirmed this with a tiny run.** The mirror's
  `Metadata_*` header is `Metadata_Dataset, Metadata_ImageName,
  Metadata_ImageType, Metadata_BitDepth, Metadata_FileSuffix`, so the
  copy-ready `plots:` block names a column that does not exist. Use
  `x: Metadata_ImageName`.
- **M9.** `dvc.md:34-37` says PhenoTypic "creates all three outs in every
  mode", and `:15` says "nothing else at the root". Process mode creates
  neither `deliverables/` nor `results/`, and writes its mirror at the root.
  The page explains this later (`:199-215`), but the earlier sentences are
  false as written. Add "(process mode: see below)".
- **M10.** `metrics.json`'s `pipeline_sha256` is `null` for every P2 run
  (`_cli_finalize_run.py:409-434`). That deviates from plan T3 Step 3, which
  said to use the run config's sha until P3 T4, but it matches spec C4's "not
  the run config's", and `dvc.md:102-104` says so. Make sure P3 T4's
  replacement is tracked as a P3 must-do; as shipped, the field gives
  `dvc metrics diff` nothing.
- **M11.** **Process and test gaps.**
  - T2a Step 4 required a DVC 3.67.1 `dvc repro` through a submit stub that
    exits 4. The commit message (`9d86bd565`) does not record that it was run.
  - Plan T4 Step 4 required three runs of the byte-stability test, with no
    record either.
  - `_cli_readme_generator.py:121` has its `#` column one space left of the
    others, a cosmetic defect in the README.
  - The README sentence "Both are rewritten by every finalization" is true
    only on the `finalize_run` path. The migrate re-publication does not write
    them, which `require_run_summaries=False` reflects.

## Test coverage gaps

These are mutations I reasoned about and the tests' answer to each:

| Mutant | Caught? |
|---|---|
| `_published` accepts any closed fence (drop the `closed_by` check) | yes: `test_a_fence_closed_without_a_reason_is_not_published`, `test_cancelled_resubmission_exits_5` |
| `wait_for_run` re-reads the generation each poll | yes: `test_measure_resubmission_waits_then_binds_to_its_own_generation` |
| `retire_dead_lifecycle_fence` passes `closed_by=FINALIZER` | yes: `test_restored_liveness_state_fails_safe` (`"closed_by" not in fence`). `test_a_restored_active_fence_does_not_block_a_slurm_resubmission:63` asserts this on the **new** fence, so it alone would not catch it |
| `_retire_dead_fence` removed from `_initialize_recompile_slurm_attempt` (`phenotypicCLI.py:4523`) | **no.** No test drives a SLURM recompile over a dead fence |
| `retire` skips `staged-*` fences | **no.** No test retires a staged fence or checks that `deactivate_orchestration` journals it |
| `retire` retires an `owner_kind` / `migrate` / `tune` fence | **no.** The negative branches are untested |
| `decide_wait_outcome` returns 3 for `{failed, unverified}` (I1) | **no**: that *is* the shipped behaviour |
| the recompile and staged-local exits (I2, I3) | **no**: no failure-bearing recompile or staged-local exit test |
| measure SLURM submission exits 4 | only through the shared `:3896` site; no measure-specific pin. A staged SLURM submission is pinned in `test_cli_slurm_submission_summary.py` |

Byte stability (C11, `test_deliverables_byte_stable.py`) compares one
in-process run against one fresh-interpreter run, with `--njobs 1` vs `2` and
distinct hash seeds. That is good design. It covers one dataset with **no
`--metadata`**, though, so the mirror's join, the phantom-row append and their
row order are never compared. Those are the ordering most exposed to a
hash-seed or worker-count effect, and the configuration a DVC user will
actually run. Add a `metadata_bytes` variant, and a two-dataset variant so the
dataset order of the concatenation is pinned.

## Design forks and open questions (for the user)

- **F1. How should a DVC stage treat `3`?** As shipped, a run with any
  terminal failure fails its DVC stage forever (I6). The options:
  1. Keep it, and document the shell idiom that accepts 3.
  2. Add `phenotypic-wait --accept-failures`, which maps 3 → 0 after
     printing the count.
  3. Have DVC users rely on `metrics.json`'s `images.failed` and treat 3 as
     success by default.

  The first is the smallest change. The second keeps the contract while
  giving DVC a clean stage.
- **F2. Should the launch form attach to a run already in flight?** If the
  wrapper job that runs `dvc repro` hits its walltime, or DVC is killed, the
  SLURM run keeps going. The next `dvc repro` re-launches the submit, and
  `_refuse_live_slurm_run` exits 1 because the run is live. The stage fails
  until the run ends; then the user must run the wait-only form by hand and
  `dvc commit`. The launch form could, on a live run of the **same**
  `--output`, skip the submit and wait on the existing generation. That is the
  natural recovery, but it changes "the child's non-4 code is returned as-is".
- **F3. The staged-local exit with failures (I3).** The plan kept `ERROR`
  on purpose. Should it become `COMPLETE_WITH_FAILURES` like every other
  mode?
- **F4. The recompile exit with failures (I2).** The plan table says
  `COMPLETE`. Spec C3 requires the waiter's code to match the local one. One
  of the two must give: 3 for both, or the waiter must special-case recompile.
- **F5. A code for "ended with images never attempted" (I1).** `5`
  (`ENDED_UNPUBLISHED`) is the nearest existing code, but its documented
  meaning is "published nothing", and this finalizer did publish a partial
  aggregate. The options are to reuse 5 and widen its meaning in the docs, or
  to add `7 = ENDED_INCOMPLETE`. The enum promises that values never change,
  so this should be settled before the merge to `main`.
- **Open (no fork).** The doc's `sbatch` line runs the **submit** itself
  inside a 1-CPU, 4 GB job, because the launch form runs it as a child. The
  submit scans the inputs, hashes every input for work ids and runs the
  preflight. Nothing checked that this fits 1 CPU / 4 GB at a 33,923-image
  scale, or that an `sbatch` from inside a job, which inherits `SLURM_*`, gives
  the same chain as one from a login shell. Worth one timed check before the
  page is relied on.
