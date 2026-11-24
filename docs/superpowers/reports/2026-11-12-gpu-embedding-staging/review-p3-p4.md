# #282 P3–P4 deep review

**Reviewer:** implementation-test-reviewer (Opus), analysis only.
**Tree:** worktree `gpu-embedding-staging`, branch `feat/gpu-embedding-staging`, HEAD `f25643315`.
**Range:** `20d6f6178^..f25643315` (P3 T1, P3 T2, P4 T1 ×3, P4 T2, P4 T3, P4 T4); docs commits skipped.
**Read:** spec §4.1–4.8, §5, U2–U11, O3; plan P3, P4, README (Review Focus, Revision 2/3,
P1/P2 fixes); `plan-review.md`; root and `_cli/CLAUDE.md`; `tracked_state.md`.

**Verdict.** P3 is sound, and its tests are thorough and able to fail. On the local
path, P4 does what the plan says on every point I was asked to look at hardest (see
"Checked and holds" at the end). I found one Critical, but it is a state of the branch
rather than a bug in local Stage 4. At HEAD, on SLURM and in measure mode, `--embed-stage`
already resolves to `gpu`, and no stage there runs Stage 4. I also found four Important
issues: a byte-equality test whose premise is false, an inert `--durable-writes`, memory
that grows with the shard, and repeated input hashing. There are seven Minor issues.

Counts: **1 Critical, 4 Important, 7 Minor.**

---

## Critical

### C1. At HEAD, SLURM and measure runs resolve `gpu`, but nothing on those paths embeds them

- **Where:**
  - `_cli_embed_stage.py:154-157`: `auto` gives `gpu` on SLURM with a `GpuDetector` or `--gpu-slurm`, and `MeasureDinoEmbedding.device` defaults to `"auto"` (`_measure_dino_embedding.py:304`).
  - `_cli_staged_slurm_worker.py:521-527`: the SLURM Stage 3 now defers, and it reads the deferral from `job_metadata.json` `EMBED_STAGE`, which `_cli_staged_slurm.py:551` writes.
  - `_cli_staged_controller.py`: the controller has no `stage4` phase.
  - `_cli_process_single.py`: `main`, the AutonomousSLURM per-image worker, never passes `defer_embeddings`.
  - `_cli_process_single.py:554`: the measure core still calls `pipeline.embed` inline.
- **Failure scenarios, on HEAD today:**
  1. **A SLURM run of a pipeline with a `GpuDetector` and a DINO embedder, all defaults.** This is the most common production shape. Every Stage-3 task defers: it writes the store and `stages.stage3`, and no record. No Stage 4 exists, and the finalizer reads records (`_all_accepted_images_succeeded`, `_cli_checkpoint_handler.py:394-405`). It therefore closes `terminal_incomplete`, so no image is ever verified. A plain re-run makes R3 pin `gpu` again, every Stage-3 step returns early, and the run cannot finish. Nothing is falsely certified, but the run never completes.
  2. **A SLURM run with no detector and `--gpu-slurm`.** It resolves `gpu`, and state and job metadata record `embed_stage: "gpu"` (`_cli_execution_strategies.py:1241`). But `AutonomousSLURMStrategy` embeds **inline on the CPU** and publishes every record. After P5 lands, a continuation of that run passes R3 (`gpu == gpu`). New images then get GPU vectors and the old ones keep CPU vectors. That is the mixing U3 exists to refuse, and no check can see it, because the record says `gpu`.
  3. **Local `--mode measure --embed-stage gpu`.** It passes the CUDA check (`prepare_stage4_environment`) and then embeds on the CPU. R3's own remedy text tells users to run exactly this (`_cli_embed_stage.py:236`).
- **About C12's note (an all-`stage4` plan keeps phase `"stage3"`):**
  - The remap itself is safe. The controller has no `"stage4"` phase (`_cli_staged_controller.py:301-387` dispatches only on `stage1` / `stage2` / `stage3` / `finalizing`), so passing `"stage4"` through would be worse.
  - Each Stage-3 step returns early on `stages_1_to_3_done`, before the prereq `SystemExit`.
  - The finalizer is record-based, so it never calls `mark_staged_complete` over record-less images.
  - So the remap is not where the risk is; scenario 1 is.
- **Fix:** gate the three paths on their executors until P5 and P6 land. In
  `resolve_embed_stage_for_config`, resolve SLURM `auto` to `inline`, and refuse an explicit
  SLURM or measure `gpu` with one usage line ("Stage 4 on SLURM / measure lands in #282 P5 / P6").
  Remove each gate in the commit that builds its executor, with a test that fails if the gate
  is lifted early: for example, a SLURM `--embed-stage gpu` resolution test that P5 flips.
  The alternative is to state in the README that no commit between `c91232d62` and P5 T2 may be
  pushed, run on SLURM, or used for the live `exfab` smoke. The gate is cheaper and makes the
  branch safe at every commit.

---

## Important

### I1. The inline-equality test does not run "the same pass shape"; its byte equality is float16 rounding, measured

- **Where:** `tests/integration/cli/test_local_stage4.py:67-85`, `tests/_tiny_run.py:2994` (`ONE_PASS_BATCH = 512`), `_batch_encoder.py:619, 780-786` (the last pass is zero-padded to `m`).
- **The premise is false.** With `batch_size=512`, the inline side runs one pass per image and key, of *n* rows, unpadded: 1 object, or the 4 grid cells. Stage 4 builds one queue for both keys (both 224-px `crop_cls`). It runs one pass holding **both images' rows**, zero-padded to `m = min(512, physical_batch_for(...)) = 512`. The GEMM shapes differ on every row.
- **Measured on this node (`i29`, the foundation venv):**
  - Setup: the stub's own `Linear(8, 8)` (seed 20261119) over 257 tokens. Row 0 alone was compared with row 0 inside a batch of 2, 8, 64 and 512, at 1 and 4 threads.
  - Result: **5 of 2,056 float32 values differ** (max 5.96e-8) in every case.
  - After `layer_norm` and a float16 cast, 0 values differed.
  - Script: `/scratch/anguy344/32338975/claude-5188/-bigdata-exfab-anguy344-PhenoTypic/755c852f-f7be-4fc5-96a7-36f51adcec11/scratchpad/gemm_rows.py`.
- **What that means.**
  - Stage 4 and inline compute different float32 vectors, and the deliverables are byte-equal only because the float16 storage rounding absorbs the difference. Each value then sits a few float32 ULPs from a float16 rounding boundary. Another CPU family, a different BLAS kernel or another thread count can flip one bit, and the test then fails although nothing is wrong.
  - Spec §5's own condition ("`batch_size` at least its region count, so each side runs one pass") does not give equal shapes either, because Stage 4 pads to `m` and batches across images. Spec and test share the error.
- **Fix:** make the shapes truly equal: `batch_size=1` on every embedder, so inline runs `(1, 3, 224, 224)` passes and Stage 4 runs `m = 1`, unpadded, one row per pass. Also correct spec §5's sentence, which says one pass each side is enough. Keep an always-on cosine assertion (`>= 1 − 1e-6` per row) beside the byte check. A fall-back assertion that cannot fail is not an alternative: the cosine check is the guard, and the byte check holds only where the shapes truly match.

### I2. Stage 4's store commit ignores `--durable-writes` / `--no-durable-writes`

- **Where:** `_cli_stage4_worker.py:839-847`. `replace_run_embeddings(...)` is called without `durable=`, so it auto-detects (`_measurement_tables.py:1657`). `output_manager` is in `_Stage4Context`, but its `durable_writes` is never read.
- **The rule this breaks:** `_cli/CLAUDE.md` "Durability": *"Durability lives on the `OutputManager` … which is what makes it structurally impossible for a write site to be silently inert."* Stage 4 is a fourth store write site, and it bypasses that.
- **Scenarios:**
  - A local `--durable-writes` run promotes un-fsynced stores. A crash then loses the commit while the record, written after it, survives.
  - A SLURM `--no-durable-writes` run (P5) is fsynced anyway.
- **Fix:** pass `durable=ctx.output_manager.durable_writes`. Add a test that monkeypatches `replace_run_embeddings` and asserts the value it receives for each flag. Add the site to the `_cli/CLAUDE.md` count in P7.

### I3. Every image's payloads and reused signals stay in memory until `run_stage4` returns

- **Where:**
  - `_cli_stage4_worker.py:590-600`: `_ImageWork.payloads` and `.reused`.
  - `:1166-1175`: `to_embed` holds every `_ImageWork` for the whole call.
  - `:1158-1163`: `futures` and `identities`.
  - `_commit_image` never clears `work.payloads` or `work.reused`.
- **Scenario:**
  - Each payload is a pandas frame whose `embedding` column is one numpy object per row (`_cli_stage4_signal.py:239`).
  - At the spec's scale (66 colonies plus 96 wells at 768-d, float16), that is about 250 kB of vectors per image, plus per-row object overhead.
  - A local or shard run over N images keeps all N images' vectors alive after they have been committed and published: gigabytes for a 33k-image run, or for a large SLURM shard in P5.
  - The README's memory model (`stage4_memory.py`) bounds the encoder's in-flight bytes and does not count this.
- **Fix:**
  - At the end of `_commit_image` (in a `finally` after `_publish`, `_leave_pending` or `_fail_image`), clear `work.payloads` and `work.reused`.
  - Drop `to_embed` entries once an image is complete. The backbone filter needs only `todo`.
  - Test: run over K images with a spy on payload construction, and assert the committed `_ImageWork`s hold no payloads by the end. A `weakref` on one payload frame, checked after its commit, is the simplest form.

### I4. Each invocation re-hashes every input up to three more times, and Stage 4 recomputes the image's identity

- **Where:** `split_stage4_pending` calls `staged_manifest_entry`, then `work_identity_for_image`, then `file_sha256` of the input and of the pipeline file. It does this per image, uncached (`_cli_failure_tracker.py:91`). It runs:
  1. in the CLI's "N of them need Stage 4 only" count (`phenotypicCLI.py:4112-4121`), for a message only;
  2. in `LocalParallelStrategy.execute` (`_cli_execution_strategies.py:430-437`);
  3. again in `run_local_stage4` (`_cli_stage4_worker.py:1362`), which also runs on a first run.

  The staged resume plan and the cores have already hashed each input once.
- **Scenarios:**
  - **Cost.** A no-detector continuation reads every raw input from GPFS three extra times, and a first run reads them once extra. The task brief cites about 33,000 images, and the spec says ~4012 × 6016 plates, so that is terabytes of avoidable I/O. P5 T2 plans to reuse `split_stage4_pending` for the SLURM no-detector continuation.
  - **Identity.** `_cli/CLAUDE.md` "Reference metadata": *"The staged engine holds one identity per image for the whole run, not one per stage … Never recompute a work-id after the apply."* Stage 4 recomputes it after Stages 1–3. If a reference manifest is re-planned mid-run, Stage 3 bound the store to the old work id. Stage 4's new work id then makes `stages_1_to_3_done` false, so the image silently drops out of Stage 4 (`not_ready`) and the run exits 1. This narrow case is only reachable with reference metadata, but it breaks a stated rule.
- **Fix:**
  - Compute the split once per invocation and carry it. `LocalParallelStrategy` can hand its `_stage4_items` to `run_local_stage4`. `StagedGpuStrategy` already holds `identities` (`_cli_staged_strategy.py:283`) and can build the entries from them.
  - Drop the CLI count, or compute it from the same split.
  - Test: count `file_sha256` calls over a continuation, and assert one per input.

---

## Minor

### M1. `record_lock` (C11): a local run is fenced exactly as the local strategies are, no more and no less

This answers the first of the "look hardest" questions.

- `_local_epoch_ownership` (`_cli_execution_strategies.py:87-107`) takes the processing-state lock and compares `processing_generation`. Stage 4 holds it around each record (`:767`) and each journal entry (`:660`), as `_publish_local_image_success` and `_record_local_terminal_failure` do. So a `--restart` or `--overwrite` invocation (a new generation) fences an older process at its next record. That process then raises `RuntimeError("Local lifecycle epoch is stale")` out of the commit thread, and exits 1 with a traceback.
- **What is not fenced, in either path:** two concurrent invocations of the *same* generation, because a continuation keeps its content-derived generation (`phenotypicCLI.py:3872-3881`).
  - In the core path, both re-run and both publish one image.
  - In Stage 4, the root check makes the store write safer than the core's. A second process meets `StoreRootChangedError` and leaves the image pending, or sees stale signals.
  - Stage 4 has one new double-publish path: process B classifies the image *after* A's `replace_run_embeddings` and *before* A's record. B sees fresh embeddings and takes "publish only", so A and B both write the record in one pass.
  - No deadlock: lock order is state lock, then event-log lock, and an event append takes only the event-log lock.
- **Fix:** no code change is needed for parity. State the limit in the P7 docs ("concurrent local invocations over one `--output` are unsupported"). A local run lease would be a separate change.

### M2. A failure to read the live digests is always terminal, and swallows an out-of-memory error

- **Where:** `_cli_stage4_worker.py:823-826`. `except Exception` goes to `_fail_image`.
- **Scenarios:**
  - `_live_store_digests` reads the full level-0 objmap and each bound series, about 100–200 MB per plate. A `MemoryError` there is caught. `append_terminal_failure` refuses to journal it, yet the image still has its signals deleted, a `failed` event, and an entry in `outcome.failed`. The run then exits 1 with "STAGED FINALIZATION FAILED" and re-embeds next time.
  - A transient GPFS `OSError` becomes a terminal record, which only `--retry-failures` clears.
- **Fix:**
  - Re-raise `is_out_of_memory(exc)` and `MemoryError`.
  - Leave the image pending on `OSError` (as for a root change, never terminal).
  - Journal only decode and format errors.

### M3. Several errors escape the one-line exit-1 mapping, and a commit error can be lost

- **Where:** `run_local_stage4` catches `(Stage4DeviceError, Stage4OutOfMemory, Stage4ShardError, MemoryError)` (`:1380`).
- **Errors that are not mapped:**
  - A dead DataLoader worker raises `RuntimeError("DataLoader worker … exited unexpectedly")` from `for item in loader` (`:913`), outside `_encoder_call`. Spec §4.8 counts this "host OOM (loader)" case as node state.
  - A `torch.OutOfMemoryError` re-raised from `vectors_from` or `payload_from` (`:959-960`, `:987-988`) is a `RuntimeError`, not a `MemoryError`.
  - Both end as "Unexpected error" with a traceback. The exit code (1) and the journal (nothing written) are still right.
- **Lost commit errors:**
  - If `_embed_backbone` raises, `pool.shutdown(wait=True)` runs, the exception propagates, and any commit future's own exception is never read.
  - A commit error raised mid-backbone is noticed only before the *next* backbone (`:1181`), so the main thread embeds the rest of the backbone first.
- **Fix:**
  - Wrap the loader iteration so that a worker death becomes `Stage4ShardError`.
  - Classify by `is_out_of_memory` in `run_local_stage4`.
  - Log every commit future's exception on the error path.
  - Check `_raise_commit_failure` after each loader item. It is cheap.

### M4. Two kinds of leftover state

- **Temp files.** `_publish` removes the image's final signal files (`delete_stage4_signals`, `:781`) but not its temp files. A temp left by an earlier crash therefore survives the run in which the image publishes, and goes only at the next invocation's sweep. Fix: call `delete_stage4_signal_temps` in `_publish`, after the signals; no writer can still hold that image.
- **A misleading pending message.** `_classify_image` returns `"pending"` when the store has no root or no journal run (`:725-727`). `run_local_stage4` then reports "store changed during the run; run the same command again" (`:1386-1389`). That message is wrong for this case, and every re-run repeats it with exit 1. `stages_1_to_3_done` makes the state unlikely, since it implies a table, but it is not impossible. Fix: treat it as the terminal per-image failure spec §4.3 gives a store that does not match its work id, or give it its own message.

### M5. Local counts and the completion predicate do some needless or misleading work

- `core_done` (`_cli_execution_strategies.py:440-444`) counts every image that skips the core as a core success. That includes images already `done`, and images whose Stage 4 will later be filtered out as terminal failures, so they are not in `stage4`'s counts.
- `_unpublished_images` (`phenotypicCLI.py`) and `_all_current_terminal_failures` each call `resolve_run_state(depth="deep")`, two deep passes back to back.
- This is display and cost only: exit codes come from `resolve_run_state`. Fix: build `core_done` from `pending` only, and share one deep resolve.

### M6. Test gaps (no false greens found; these are missing cases)

- **No test of node-state failures at the CLI.**
  - Nothing tests `Stage4OutOfMemory`, a loader `FailedImage(node_state=True)`, or a host `MemoryError`, each giving exit 1 with no journal row and every image still pending. `test_stale_signals_are_deleted_before_the_first_load` stops at `run_stage4`. The `node_state` path (`:939-942`) is never executed.
  - Add an OOM-once fault to `FaultyDinoEmbedding`'s trigger file, so a spawned worker sees it, and assert exit 1, `terminal_failures == []`, and `pending` everywhere.
- **No test of a DataLoader worker death** (M3).
- **`test_a_local_run_left_pending_exits_1` is incomplete.** It asserts neither that nothing was journalled nor that a plain re-run then exits 0.
- **The crash tests have one key per image** (the `FakeGpuDetector` pipeline with `emb` only). A crash between two keys' signals, the C4 resume case, is covered only by planted signals (`test_a_second_round_embeds_only_the_keys_it_lacks`), never by a real crash. Add `grid=True` to one crash parameter.
- **There is no crash *inside* `replace_run_embeddings`** (after the part, before the root). It should leave a `.part` and a resumable image.
- **There is no negative publish-only test:** a store whose embeddings block has a different `params_sha256` or `compute_dtype` must be re-embedded, not published (`_store_holds_fresh_embeddings`, `:699-703`).
- **There is no `--durable-writes` propagation test** (I2).

The crash tests themselves are real. `os._exit(137)` from the commit thread ends the process, and the test asserts return code 137 and reads the crashed image from disk. Fault injection is on disk, and both `num_workers=0` and `num_workers=2` are parametrised. Nothing relies on a monkeypatch that a spawned worker must see: the workers only open stores and call `prepare_regions`.

### M7. Documents now stale (assigned to P7, but the root rule says "same change")

- `_cli/CLAUDE.md` "Progress events" still says `aggregate_state_from_events` "counts overall completion at Stage 3 only". It now takes `terminal_stage`.
- `tracked_state.md` gained the `_unpublished_images` consumer row. It has no rows for the Stage-4 signal directory, its writer and consumers, or its sweep. `_cli/CLAUDE.md` asks for those "in the same change".
- The plan assigns all three to P7 T1, so this is a note, not a request to move the work. If P7 is long after P5, consider landing the signal rows with P5, which adds the controller as a consumer.

---

## Checked and holds

The items the brief asked me to look at hardest, verified in code:

- **`stages_1_to_3_done` gates every shortcut, and none publishes.**
  - Local `_terminal_output_exists` returns early under deferral (`_cli_staged_strategy.py:143-146`). It gates both Stage 2 (`:255-264`) and Stage 3 (`:332`).
  - Stage 1 skips a matching store (`:208`).
  - SLURM `run_stage3_step` returns before the prereq check (`_cli_staged_slurm_worker.py:521-527`). Its old resume publish is now `not deferring` (`:528`), and its first `terminal` branch publishes nothing.
  - The SLURM Stage-2 shard filters waiting images (`:370-378`).
  - `classify_staged_image` returns `"stage4"` only after `valid_image_success` has said no (`_cli_staged_resume.py:365-371, 400-409`).
  - `split_stage4_pending` keeps pending images away from the single-pass core.
  - Both cores drop `stages.stage3` and the image's signals before their first write, and write `stages.stage3` last.
  - `clear_downstream_artifacts_for_stage1` deletes the signals. `publish_image_record` merges `stages`, so Stage 4's record keeps `stage3`.
- **Completion and exit codes.**
  - Locally, a Stage-4 terminal failure exits 3: it is journalled with `failed_stage="stage4"`, and `_unpublished_images` plus `_all_current_terminal_failures` read it.
  - A pending leftover gives a `ClickException`, exit 1, before finalization.
  - OOM is refused by `append_terminal_failure`; a root change goes through `_leave_pending`. Neither is terminal, apart from M2's digest-read exception.
  - `staged_complete` needs `not unfinished` over records.
  - The SLURM finalizer is record-based (`_cli_checkpoint_handler.py:394-405`).
- **The commit path.**
  - Signals are classified per key against the live root, and stale ones are deleted before any load. The test `test_stale_signals_are_deleted_before_the_first_load` pins this.
  - Valid signals are reused per key (C4); commit-only and publish-only are routed.
  - The binds come from the live store (I11), and a payload bound elsewhere fails the image before any commit.
  - Order is `replace_run_embeddings`, then the record, then the signal deletion, on one thread. A spy checks the order per image.
  - Signals from two roots never commit.
  - A poisoned encoder raises `Stage4ShardError`, and nothing is journalled. `test_a_failed_forward_pass_fails_the_shard_and_journals_nothing` can fail.
  - `epoch or "local-unfenced"` matches the local strategies' stamping.
- **C12's bug fix.**
  - `run_local_stage4` now drops exact terminal failures unless `--retry-failures` (`:1363-1365`). This is needed because a finalizer-only continuation hands it `full_datasets` (`phenotypicCLI.py:4015-4027`).
  - The core skip uses `core_tasks` only when `config.resume`. A first run has nothing pending, and `--restart`/`--overwrite` are not continuations (`phenotypicCLI.py:3196-3202`).
  - `test_retry_failures_reruns_only_stage4` is parametrised over both pipeline shapes.
- **C9 and R3.**
  - The resume rewrite keeps `embed_stage` (`phenotypicCLI.py:4197`). `test_a_gpu_run_keeps_its_embed_stage_across_continuations` runs two continuations, so it would catch the dropped key.
  - R3 sits after the state load, before `validate_resume_compatibility` and before `prepare_stage4_environment`, so a refused option exits 2 rather than 1.
  - It does not fire on `--restart` or `--overwrite`.
- **The DataLoader.**
  - It uses `spawn` with `persistent_workers=False`, `batch_size=None`, identity `collate_fn`, `prefetch_factor=2`, and in-order iteration.
  - `num_workers = resolve_local_worker_count(n_jobs, n) − 1`, so `--njobs 1` gives 0 workers.
  - The processor is loaded in the main process and pickled in; a worker loads nothing.
  - Workers materialise their crops.
  - `phenotypic/__main__.py` is `__main__`-guarded, so a spawn does not re-run the CLI.
  - Tests cover 0, 1 and 2 workers, and pickling.
- **P3.**
  - `replace_run_embeddings` refuses an unknown run or a non-v2 store before any part exists. It rewrites only the run's entry in place, so every other run and `upstream` stay byte-equal. It recomputes `stale` from the caller's digests, and its root-change refusals leave no part.
  - The signal fences on work id, run, key directory and root. It is atomic, its delete joins the stem rather than globbing it, and its temp sweep matches by prefix and suffix.
  - The tests include controls (a consolidated-fixture assert, a second run, an independent zarr + pyarrow reader), and they can fail.
