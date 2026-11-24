# #282 P3–P4 review fixes (C12r)

**Input:** `review-p3-p4.md` (1 Critical, 4 Important, 7 Minor), with the
orchestrator's rulings. **Tree:** worktree `gpu-embedding-staging`, branch
`feat/gpu-embedding-staging`, from `cfe15cc3c`. **Final code SHA:**
`c0fb687f2`; this report is a docs-only commit on top.

## Commits, by finding

| Finding | Commit | What changed |
|---|---|---|
| C1 | `a5b5c92b8` | `stage4_supported(config)` (`_cli_embed_stage.py`): only a local full run. SLURM (any mode) and `--mode measure` (local or SLURM): `auto` resolves `inline` with one note; explicit `gpu` is one usage-error line. R3's remedy names only a runnable command (`--embed-stage <recorded>` when this invocation could honour it, else `--restart`; the `--mode measure --embed-stage gpu` alternative is dropped until P6 T1). P5 T2 and P6 T1 each gain the line that lifts their path and names the tests to flip; README Global Constraints states the gate. |
| I1 | `5b9dba896` | The byte test runs `batch_size=1` on both sides and asserts every entry ran at `m = 1`. A new test is always on: per-row cosine `>= 1 − 1e-6` at the default batch. Spec §5 amended. `design.html` and `plan.html` re-rendered. |
| I2 | `f5019b9a0` | `replace_run_embeddings(durable=ctx.output_manager.durable_writes)`. The tri-state is passed through, so `None` stays `None`. |
| I3 | `9cce6edff` | `_ImageWork.release()`: `_commit_image` releases in a `finally`, and a stream failure releases at once. |
| I4 | `7caf7371e` | `split_stage4_pending`/`run_local_stage4` take entries built from an identity the caller holds (`manifest_entry_for_identity`, which hashes nothing). `LocalParallelStrategy` computes each identity once under deferral (`stage4_entries`, threads), then hands it to the core and to Stage 4. `StagedGpuStrategy` hands Stage 4 Stage 1's identities, and `_terminal_output_exists` reads them instead of re-hashing. The CLI count is removed; the local strategy prints it from its own split. |
| M1 | `2c2f783b5` | Docs only: `run_stage4` docstring and `_cli/CLAUDE.md`. A new generation is fenced; the same generation is not. |
| M2–M6 | `c0fb687f2` | See below. |

The M2–M6 commit (`c0fb687f2`) changes, finding by finding:

- **M2.** A digest-read failure re-raises an out-of-memory error, whether `MemoryError` or torch's. An `OSError` leaves the image pending (`PENDING_UNREADABLE`) and keeps its signals. Only other errors are journalled.
- **M3.**
  - `_loader_items` turns a failure of the loader itself (a dead worker) into `Stage4ShardError`.
  - `run_local_stage4` classifies by `is_out_of_memory`, so a torch OOM is also the one-line exit 1.
  - `check_commits` runs after each loader item.
  - `_log_commit_failures` logs every commit's exception on the error path.
- **M4.**
  - `_publish` also deletes the image's signal temps.
  - A store with no root is pending (`PENDING_NO_ROOT`).
  - A root with no forward journal run is a terminal `stage4` failure.
  - The pending message names its reasons.
- **M5.**
  - `core_done` holds only pending images that Stage 4 will take (`without_terminal_failures`, shared with `run_local_stage4`).
  - One deep `resolve_run_state` feeds both `_unpublished_images` and `_all_current_terminal_failures`.
- **M6.** These are tests. `FaultyDinoEmbedding`'s trigger file now selects `fault`, `oom-once` or `die`, and `run_cli_subprocess` gains `crash_inside="replace_run_embeddings"`.

## I1: the measurement behind the spec amendment and the floor

- **GEMM row dependence.** Measured with the reviewer's script (copy:
  `gate-results/c12r-282/proofs/gemm_rows.py`, log `gemm_rows.log`).
  - Environment: `i29` (Xeon E5-2683 v4), torch 2.11.0+cu130.
  - Setup: row 0 of the stub's `Linear(8, 8)` over 257 tokens, run alone and as one row of passes of 2, 8, 64 and 512 rows.
  - Result: **5 of 2,056 float32 values differ** in every case, at 1 and 4 threads (max 5.96e-8). After `layer_norm` and a float16 cast, 0 differ.
- **The cosine floor.** Measured on `i29` (log `i1-cosine-floor.log`), Stage 4 against inline at the default `batch_size` (64), both pipeline shapes:
  - 0 float16 values differ, and the minimum cosine is 1 − 2.2e-16.
  - Moving every stored value of these rows by one float16 ULP gives at most 1 − cos = 3.9e-7.
  - So the floor 1 − 1e-6 admits a one-ULP drift in every value. It is the README's per-row floor, now with a measured reason.
- **What the floor cannot do.** The stub's closest distinct rows (Otsu objects) are 5.7e-8 apart, so on the stub the floor cannot tell two near-identical colonies apart. It guards numerical agreement, not row identity. The identity columns are asserted equal separately (`deliverable_row_cosines`).

## Failure proofs

Each mutant was applied with `gate-results/c12r-282/proofs/mutate.py`, which works in four steps:

1. back up the target file's bytes;
2. apply one exact-text replacement (the `<name>.old` / `<name>.new` files sit beside it);
3. run the tests;
4. restore from the bytes and compare the sha256.

Every row below restored with `sha OK`. The logs are `proofs/<name>.log`.

| Mutant | What it breaks | Result |
|---|---|---|
| `c1-gate` | gate removed (`if False:`) | 14 failed (unit + ordering, PT) |
| `c1-gate-torch` | same | `test_a_slurm_detector_dino_run_resolves_inline_and_finishes` failed (TPT) |
| `c1-r3` | R3 always offers `--embed-stage <recorded>` | 2 failed (`test_r3_never_names_a_command_that_cannot_run[gated-*-gpu-inline]`) |
| `i1-perturb` | Stage-4 vectors perturbed | 4 failed (byte test and cosine test, both shapes) |
| `i1-m-not-1` | Stage 4 ignores `batch_size` (m = 64) | 2 failed (byte test) |
| `i2-durable` | `durable=` dropped | 3 failed (durable, no-durable, auto) |
| `i3-release` | no release in the `finally` | 2 failed (embedded, reused signals) |
| `i4-stage4-rehash` | `run_local_stage4` re-hashes | 4 failed |
| `i4-staged-rehash` | `_terminal_output_exists` re-hashes | 2 failed (staged) |
| `i4-core-rehash` | the core ignores the carried identity | 1 failed (single-pass first run) |
| `m2-oom` | digest OOM swallowed | 2 failed (host, torch) |
| `m2-oserror` | `OSError` journalled | 1 failed |
| `m3-loader` | plain `for item in loader` | 1 failed (`loader-worker-death`: a traceback, no `Error:` line) |
| `m3-classify` | no `is_out_of_memory` classification | 1 failed (`torch-oom`) |
| `m3-check` | no per-item commit check | 1 failed |
| `m3-log` | no commit-failure log | 1 failed |
| `m4-temps` | temps not deleted at publish | 1 failed |
| `m4-norun` | no journal run → pending | 1 failed |
| `m4-noroot` | no root → generic reason | 1 failed |
| `m5-core` | old `core_done` (every skipped image) | 1 failed |
| `m5-snapshot` | helper re-resolves | 1 failed |

**One test of mine was wrong at first, and a mutant caught it.**
`m3-loader` first *survived*. The `loader-worker-death` case was spelled
`"loader-death"` in the parametrisation but matched as
`"loader-worker-death"` in its `if` chain. It therefore fell through to the
encoder-OOM branch and passed for the wrong reason. I fixed the selector,
asserted the branch, and asserted that the message names the loader. The
mutant was then caught.

## Commands, with their lanes and summary lines

- `PT` = shared env. `TPT` = foundation venv. Both are as in the README. The `-n` values are within my allocation (8 CPUs).
- These runs are local to the per-step and per-task stage. The phase gate is the Slurm job below.

| Lane | Command | Summary |
|---|---|---|
| PT | `tests/unit/cli/test_cli_embed_stage.py` | 61 passed |
| PT | `tests/unit/cli/test_cli_preflight_ordering.py` | 47 passed |
| TPT | `tests/integration/cli/test_embed_stage_refusals.py` | 3 passed |
| TPT | `test_local_stage4.py -k "matches_inline or agrees_with_inline"` | 4 passed |
| TPT | `test_stage4_worker.py` (after I3), `-n 8` | 39 passed |
| TPT | `test_local_stage4.py test_stage4_deferral.py test_stage4_worker.py test_staged_gpu_local.py test_embed_stage_refusals.py` (after I4), `-n 8` | 116 passed |
| TPT | `test_local_stage4.py test_stage4_worker.py test_stage4_deferral.py test_embed_stage_refusals.py` (after M2–M6), `-n 8` | 103 passed |
| — | `uv run ruff check --fix <changed paths>` | All checks passed |

**Gate**, at `c0fb687f2`, set up the way C12 set up its gate:

- the P4 surface, `files.txt`: 310 files, as 24 shards on `intel`;
- plus the foundation lane, `foundation_files.txt`: 7 files.

The gate runs in a detached worktree, `gate-worktrees/c12r-c0fb687f2`, and an `afterany` cleanup removes it. Its script is `gate-results/c12r-282/make_gate.sh`. The job ids:

- array `32342415`;
- foundation `32342416`;
- cleanup `32342417`.

The results go to the orchestrator once the jobs end. A finished job does not wake an agent, so this report does not hold them.

## Not fixed now, and why

- **The CLI's own startup hashing (I4 residual).** From the strategy through Stage 4, each input is now hashed once (pinned). The whole invocation still hashes each input before the strategy runs: 3× on a first run and 4× on a continuation, measured in `proofs/i4-whole-invocation-hash-count.log` (strategy + Stage 4: 1×). The extra hashes come from CLI sites that predate #282:
  - `_prepare_incremental_startup`'s legacy-failure migration (`phenotypicCLI.py` `work_id_for_image` in `valid_work_ids`);
  - the resume selection (`get_remaining_images_for_datasets`);
  - `state.config["work_ids"]`.

  Carrying one CLI-computed identity into the strategy would remove them. Orchestrator ruling: not widened in #282, since these sites predate it and it does not make them worse. Recorded as a follow-up in the plan README's "Follow-ups" section: "The CLI's startup re-hashes every input before the strategy, so a forward invocation hashes each input 3× in all on a first run and 4× on a continuation, against 1× from the strategy through Stage 4 (`proofs/i4-whole-invocation-hash-count.log`)."
- **M7, assigned to P7 and left as it is:**
  - `_cli/CLAUDE.md` "Progress events" still says Stage 3 only. `aggregate_state_from_events` now takes `terminal_stage`.
  - `_cli/CLAUDE.md` "Durability" still says "exactly three write sites". Stage 4's `replace_run_embeddings` is a fourth (I2).
  - `tracked_state.md` has no rows for the Stage-4 signal directory, its writer, its consumers and its sweep.
- **M1** needed no code change, by the ruling. A local run lease would be a separate change.
- **The artifacts are re-rendered, not republished.** `design.html` and `plan.html` match their Markdown at `5b9dba896` and after. Republishing to the artifact URLs is the orchestrator's call.
