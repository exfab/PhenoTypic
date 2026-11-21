# #282 P1 + P2 deep review: the fixes (cluster C5r)

**Review:** `review-p1-p2.md` (this folder), 6 Important and 17 Minor items.
**Tree:** worktree `gpu-embedding-staging`, branch `feat/gpu-embedding-staging`.
**Range:** `f70578175..fc131d9ff`, six commits:
- four code commits;
- one docs commit;
- one test fix.

Every number below is read from a log under
`/bigdata/exfab/anguy344/gate-results/c5r-282/`. The log is named beside the
number.

## Summary

- **Status.** Every Important item is fixed. Every Minor item is fixed, except
  M12, which the orchestrator assigned to P4 (P4 T1 Step 3a).
- **The bottomed-out OOM ruling.** `Stage4OutOfMemory` now poisons the encoder
  too, as the orchestrator ruled.
- **Mutants.** 26 mutants, each breaking one fix. Every one was killed except the
  one expected survivor, which shows the gap I5 closes.
- **Gates at `d73a165cb`:**
  - the shared lane: `1 failed, 781 passed, 153 skipped`;
  - the torch lane: `1 failed, 976 passed, 6 skipped`;
  - the Broadwell golden on i29: `25 passed`, no skips.

  The one red in both lanes is a test that patched a seam the M11 change had
  bypassed. It is fixed in `fc131d9ff`, a test-only commit, and both lanes pass
  it alone.
- **RSS.** I4 restores bounded host memory, by about the whole crop stack at
  448 px. At 224 px, one run showed the fixed tree 52400 KiB higher. A second
  job showed that this is glibc's dynamic mmap threshold, not a cost that
  grows with the region count.

## Item to commit

| Item | Commit | What changed |
|---|---|---|
| I1 mixed m within one image | `74781a7e3` | `drain()` yields `DrainedTicket(ticket, outputs, physical_batches)`. `compute_block(physical_batch=int \| Collection[int])` records one size as `physical_batch`; several as `physical_batch: null` plus `physical_batches`, largest first; none as `physical_batch: null` |
| I2 one m per encoder | `74781a7e3` | `BatchEncoder(physical_batch=int \| (h, w) -> m)`, called once per queue; an OOM halves only that queue's m. `physical_batches` and `halved` replace the `physical_batch` property and the public `halve()` |
| I3 no single producer of m | `74781a7e3`, `93c049d39` | `physical_batch_inputs` / `physical_batch_for` in `_batch_encoder.py` read the weight bytes, tokens and heads from the loaded model. `tokens_for` takes a tile's `(h, w)`. The T4 script's `m_for` calls both |
| I4 inline host memory | `9100a4df2` | `MeasureEmbedding._prepare_inline`. `MeasureDinoEmbedding` defers its crops (`_DeferredCrops`), cut one slice at a time by the same `_crop` calls in the same order. `prepare_regions` still materialises by default |
| I5 non-default digests only on Broadwell | `b5d9a4ab0` | An always-on, torch-free check of all 9 golden rows' `params_sha256`, plus a check that every golden row has one |
| I6 a non-OOM error loses rows | `74781a7e3` | Any failed pass re-queues its rows, then poisons the encoder: `submit`, `flush` and `drain` raise `RuntimeError` chained from it |
| OOM ruling (orchestrator) | `dcf88149f` | A `Stage4OutOfMemory` poisons the encoder too |
| M1 byte bound `>` / `>=` | `74781a7e3` | The test asserts that nothing runs at the bound, then that all three tickets finish once it is crossed |
| M2 device-field second producer | `74781a7e3` | `backbone_device_name` is patched with a counter; the test asserts one call and equal fields |
| M3 masked_dense relies on GEMM luck | `dcf88149f` | Both sides compared at one tile a pass (`physical_batch=1`) |
| M4 two vacuous-PASS paths | `93c049d39` | `stored_invariant` needs rows in every region. A torch-free `verdict()` fails when no row ran. Repeat mode with nothing to repeat exits 1 |
| M5 shared worktree per SHA | `93c049d39` | One tree per submission (`c6-<sha>-<timestamp>-<pid>`); an existing tree is refused |
| M6 `empty_cache` inside `except` | `74781a7e3` | Runs after the `except` block. No test: it runs on CUDA only |
| M7 duplicated definitions | `dcf88149f` | `_is_device_oom` is `abc_.is_out_of_memory` without `MemoryError`. The dtype list is `get_args(ComputeDType)`. `COMPUTE_DTYPE_DEFAULT` (`sdk_.typing_`) is read by the field default, `_DIGEST_NEUTRAL_DEFAULTS` and the deliverables |
| M8 manifest `compute_dtype` for every kind | `dcf88149f` | The absent-field default applies to `producer.class == "MeasureDinoEmbedding"` only |
| M9 `"cuda:N"` treated as the CPU | `dcf88149f` | `compute_block`, `backbone_device_name` and `configure_determinism` read the index |
| M10 inline `physical_batch` is the cap | `dcf88149f` | An inline payload records the passes it ran, through the ABC hook `_inline_entry_descriptor` and one `_batch_bounds` |
| M11 inline masked_dense loads a second processor | `dcf88149f` | The inline tiles use the cached backbone's processor, read only after the regions (`_prepare_regions(processor_of=...)`) |
| M12 a late `CUBLAS_WORKSPACE_CONFIG` | deferred | P4 T1 Step 3a (orchestrator ruling) |
| M13 global determinism never restored | `dcf88149f` | Documented as process-global and never restored |
| M14 missing negative tests | `dcf88149f` | `_admit`'s refusals (6 cases), the tile squeeze, the sync adaptor, the state after `Stage4OutOfMemory`, and bf16 masked_dense at one pass shape. `_admit` refuses a non-float32 tile |
| M15 in-flight bound docstring | `dcf88149f` | It counts outputs only, not queued tiles or undrained tickets; the caller drains after every submit |
| M16 float16 unmeasured | `dcf88149f` | Stated in the field's docstring |
| M17 loose typing | `dcf88149f` | `BackboneKey.size: EmbeddingDinoSize`, `compute_dtype: ComputeDType`; `backbone_key() -> BackboneKey` |
| Docs | `d73a165cb` | Spec §4.4/§4.5, README Global Constraints and a review section, P4 T1/T3 interfaces, one P5 line. Both artifacts re-rendered, and each `cmp`-equal to a fresh render |
| The seam a test patched | `fc131d9ff` | See "The `fc131d9ff` root cause" |

The six P4 hand-offs are explicit steps in the P4 plan:
- T1 Step 3a: refuse a late cuBLAS setting (M12).
- T3 Step 3a: `_check_geometry` after each load.
- T3 Step 3b: an explicit device for `configure_determinism`.
- T3 Step 3c: the per-geometry rule from `physical_batch_for`.
- T3 Step 3d: the loader hands materialised crops only.
- T3 Step 3e: a poisoned encoder fails the shard.

## Failure proofs

**How the proofs ran.** Slurm job 32326840 ran on c23, in a private detached
worktree at `d73a165cb`. The tree was dirty=0 at its start and end. Log:
`c5r-mutants_32326840.log`.

**What each mutant does.**
- Each mutant (`mutant.py`) replaces one string that occurs exactly once,
  breaking one fix.
- It runs the test that guards the fix, then restores the file from a byte
  copy.
- It compares the file's sha256 before and after.

**The baseline.** Before any mutant, the guarding tests were green: `39 passed,
113 deselected`. `m4_proof.py` and `m5_proof.sh` passed on the fixed scripts.

**The sha check.** All 26 shas matched after restore (`MATCH 26`, `MISMATCH 0`).

| Mutant | Breaks | Guarding test | Result |
|---|---|---|---|
| I1-route-overwrites | a ticket keeps only its last pass size | `test_a_halving_mid_ticket_records_every_pass_size` | caught |
| I1-block-keeps-one-size | several sizes recorded as one | `test_the_compute_block_states_every_pass_size` | caught (`1 failed, 4 passed`) |
| I2-one-m-per-encoder | every queue takes the first queue's m | `test_embedders_sharing_a_backbone_get_an_m_per_crop_px` | caught |
| I2-halving-every-queue | an OOM halves every queue | `test_an_oom_halves_only_its_own_queue` | caught |
| I3-float32-dtype-bytes | `dtype_bytes=4` for every dtype | `test_physical_batch_for_gives_the_scripts_m_for_every_combination` | caught |
| I4-materialised-inline | the inline path materialises every crop | `test_the_inline_path_cuts_crops_one_batch_at_a_time` | caught |
| I5-old-test-misses | the digest drops a field whatever its value | the default-only `test_default_float32_keeps_the_281_params_digest` | **survived (expected)** |
| I5-drop-non-neutral | the same mutant | the new per-config digest test | caught (`1 failed, 8 passed`) |
| I6-no-requeue | a failed pass loses its rows | `test_a_non_oom_forward_error_requeues_its_rows_and_poisons_the_encoder` | caught |
| I6-no-poison | a non-OOM error does not poison | the same test | caught |
| M1-bound-ge | `>` becomes `>=` | `test_dense_grids_flush_on_the_byte_bound` | caught |
| M2-second-producer | `model.device_name` computed separately | `test_model_and_compute_device_fields_agree` | caught |
| M4-empty-region-passes | an empty region counts as invariant | `m4_proof.py` | caught (`FAIL zero colonies`, `FAIL nothing at all`) |
| M4-no-rows-passes | no rows gives PASS | `m4_proof.py` | caught (`FAIL no rows (repeat)`, `(measure)`) |
| M5-shared-tree | the old shared tree name | `m5_proof.sh` | caught (`FAIL the two submissions share one tree`) |
| OOM-no-poison | a bottomed-out OOM does not poison | `test_an_oom_that_cannot_halve_requeues_and_poisons_the_encoder` | caught |
| M3-tiles-shifted | tiles shifted by 0.001 in the encoder | `test_the_batch_encoder_matches_the_inline_encoder[masked_dense]` | caught |
| M7-host-oom-halves | a host `MemoryError` counts as a GPU OOM | `test_only_a_device_oom_halves` | caught |
| M7-default-constant | `COMPUTE_DTYPE_DEFAULT = "bfloat16"` | `tests/unit/cli/test_embedding_grouping_provenance.py` | caught (`1 failed, 3 passed`) |
| M8-every-producer | every producer gains `compute_dtype` | `test_only_a_dino_entry_reads_an_absent_compute_dtype` | caught |
| M9-indexed-is-cpu | `"cuda:1"` read as the CPU | `test_an_indexed_cuda_device_is_cuda` | caught |
| M10-records-the-cap | the inline block records `batch_size` | `test_inline_entries_record_their_compute` | caught |
| M11-second-processor | inline tiles load a second processor | `test_inline_dense_tiles_use_the_backbones_own_processor` | caught |
| M14-float64-tile | float64 tiles admitted | `test_inputs_that_are_not_a_backbones_are_refused` | caught (`1 failed, 5 passed`) |
| M14-no-squeeze | a `(1, 3, h, w)` tile not squeezed | `test_a_batched_tile_is_squeezed` | caught |
| M14-sync-no-refusal | the sync adaptor runs beside a held ticket | `test_the_sync_adaptor_refuses_an_encoder_holding_another_ticket` | caught |

The log's tally is `survived=1 (expected 1)`.

**Items with no mutant.**
- M13, M15 and M16 are documentation.
- M17 is annotations only.
- M6 runs on CUDA only.
- M12 is P4's.

**The `fc131d9ff` fix.** It has its own mutant. Log: `oom-test-mutant.log`.
- The mutant breaks the out-of-memory re-raise in `payload()`
  (`if is_out_of_memory(exc):` becomes `if False:`), so a `MemoryError` is
  wrapped as the image's failure.
- It ran in a private detached worktree at `fc131d9ff`, which was removed
  afterwards.
- In both lanes the test failed. Each plate recorded a terminal
  `PerImageScientificError` ("`<class 'MemoryError'>: the node ran out`"): `1 failed in 5.12s` in the shared env and `1 failed in 4.16s` in the torch env.
- The file's sha256 was `5dad6cc2…877d` before and after, a match in both lanes.
  The tree was dirty=0.

## Gates

**Local runs, from the worktree root** (the README's `$TPT` and `$PT`):

| Run | Files | Result | Log |
|---|---|---|---|
| torch lane | the 5 files touched by I1–I6 | `124 passed, 6 skipped, 2 deselected in 209.65s` | `touched-tpt-1.log` |
| shared env | test_batch_size, test_embedding_inline_golden, test_measure_dino_embedding_sizes | `48 passed, 39 skipped, 2 deselected in 1.17s` | `touched-pt-1.log` |
| torch lane | the 11 files touched by `dcf88149f` | `315 passed, 6 skipped, 3 deselected, 1 warning in 334.11s` | `touched-tpt-2.log` |
| shared env | the grouping, batch_size, golden and sizes files, plus both ci import guards | `301 passed, 42 skipped, 2 deselected in 455.41s` | `touched-pt-2.log` |

- **The skips.** The 6 torch-lane skips are the golden's byte tests, which run
  only on the capture CPU.
- **The warning.** It is the deliberate float16 overflow in
  `test_payload_from_refuses_non_finite_float16`.
- **ruff.** Clean on every changed path.
- **mypy.** On the changed source modules, mypy reports 2 errors, both
  `import-not-found` for torch: `_dino_support.py:471` and
  `_batch_encoder.py:207` (`mypy-2.log`). mypy reports this once per module,
  at its first `import torch`. A like-for-like run at the base was not
  obtained, so this report does not claim they are pre-existing.

**Slurm gate at `d73a165cb`.** The gate ran in a detached worktree. Each job
printed dirty=0 at its start and end. The file list is the 31 importers of the
changed modules (`gate_files.txt`); no GUI file is among them.

| Job | Lane | Node | Result |
|---|---|---|---|
| 32326837 | shared env, 8 CPUs, `-n 8` | c27 | `1 failed, 781 passed, 153 skipped, 3 warnings in 308.83s` |
| 32326836 | torch lane, 16 CPUs, `-n 16` | c05 | `1 failed, 976 passed, 6 skipped, 8 warnings in 1942.61s` |
| 32326838 | Broadwell golden, all rows including the slow real dinov2-small ones, `OMP_NUM_THREADS=4` | i29 (`Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz`) | `25 passed, 1 warning in 354.03s` |
| 32326840 | mutants | c23 | 25 caught, 1 expected survivor |

- **The one red.** In both lanes it is
  `test_embedding_aggregation.py::test_an_embedder_that_runs_out_of_memory_records_no_terminal_failure`,
  fixed in `fc131d9ff`.
- **The golden.** It ran with no skips, so every row's bytes matched the #281
  golden. That includes M11's processor change on `real_small_objects_dense`.
  The warning is transformers' non-writable-array `UserWarning`.
  `fc131d9ff` changes a test only, so this result holds at HEAD.
- **Cleanup.** Job 32326841 removed the three trees.

## The `fc131d9ff` root cause

The cause was M11 (`dcf88149f`), not I4.

- At I4 (`9100a4df2`), `_prepare_inline` still called the public
  `prepare_regions(image, materialise=False)`. The test's patch,
  `_oom(self, image, **kwargs)`, accepts that call, so it still fired.
- M11 moved the body into a private
  `_prepare_regions(image, *, processor_of, materialise)`, and
  `_prepare_inline` calls that directly. `measure()` and `payload()` no longer
  reach the patched public method, so the embedder really ran.

The test run alone (`oom-test-isolated.log`, `oom-test-fixed.log`):

| Commit | Shared env | Torch env |
|---|---|---|
| `d73a165cb` | failed: `ModuleNotFoundError: No module named 'torch'`, recorded as a terminal failure | failed in 437.25 s: it embedded with the cached dinov2-base and exited `RunExitCode.COMPLETE` |
| `f70578175` | `1 passed in 5.31s` | `1 passed in 3.93s` |
| `fc131d9ff` | `1 passed in 8.81s` | `1 passed in 3.33s` |

- **The fix.** The test patches `MeasureDinoEmbedding._prepare_regions`: inline
  `measure()` and `payload()` reach it through `_prepare_inline`, and Stage 4
  through the public `prepare_regions`.
- **No other test.** No other test patches `prepare_regions`.
- **Why the local runs missed it.** `test_embedding_aggregation.py` was in the
  gate's importer list but not in the local touched-file runs.

## Host memory (I4)

The probe (`rss_probe.py`, and `rss_probe2.py` with a region count):
- **What it runs:** one inline `MeasureDinoEmbedding(size="small", device="cpu")` payload on the Otsu-detected synthetic plate, which has 552 regions.
- **How it measures:** after the backbone loads, it resets the peak by writing
  5 to `/proc/self/clear_refs`. "Growth" is `VmHWM` after the payload minus
  `VmRSS` before it, in KiB.
- **Isolation:** each run is a fresh process.

**Job 32326839** (c21, 4 threads, `before f70578175 / after d73a165cb`). It hit
its 45-minute limit after one run:

| Tree | crop | Growth | all_crops |
|---|---|---|---|
| before | 448 | 1960700 | 324576 |
| after | 448 | (killed, time limit) | |

**Job 32330273** (c16, `AMD Opteron(tm) Processor 6376`, 8 threads,
`before f70578175 / after fc131d9ff`). One run per tree per crop:

| crop | Growth before | Growth after | Change | Seconds before → after | all_crops |
|---|---|---|---|---|---|
| 448 | 1974984 | 1684052 | −290932 | 1036 → 647 | 324576 |
| 224 | 651684 | 704084 | +52400 | 137 → 130 | 81144 |

**Job 32335179** (c16, the same CPU and trees). All runs at 224 px, one run
per tree per row:

| Run | Regions | Growth before | Growth after |
|---|---|---|---|
| dynamic mmap threshold | 552 | 638920 | 699532 |
| dynamic mmap threshold | 138 | 701844 | 689520 |
| `MALLOC_MMAP_THRESHOLD_=131072` (fixed) | 552 | 564524 | 492656 |

**Ruled out first, from the code.** The deferred crops keep the layer pixels
and the objmap alive through the forward passes, so I measured both on this
plate:
- the layer pixels are a view of `image.rgb[:]`, 600 × 800 × 3 uint8 (1.4 MiB);
- the objmap copy is uint16 (0.9 MiB).

That is about 2.3 MiB, and it grows with the image's size, not with its
region count.

**Reading.**
- **At 448 px, the saving is the crop stack less one batch.**
  - The saving is 290932 KiB.
  - The crop stack less one deferred slice is 324576 − 37632 = 286944 KiB,
    where one slice is 64 × 448 × 448 × 3 bytes = 37632 KiB.
- **At 224 px, with the dynamic threshold off, the same arithmetic holds.**
  - The fixed tree is 71868 KiB below the base tree.
  - The crop stack less one slice is 81144 − 9408 = 71736 KiB.
- **The +52 MB at 224 px is glibc's dynamic mmap threshold.**
  - Each per-batch slice, 9408 KiB at 224 px, is mmapped and then freed.
  - Freeing it raises the threshold to that size.
  - Later slices and torch's smaller buffers then come from a heap that
    fragments and is not given back to the OS, so peak RSS counts freed
    blocks.
- **Allocator variance.** With the dynamic threshold on, the base tree itself is
  not monotonic: 701844 KiB at 138 regions against 638920 KiB at 552. The
  allocator alone moves a run by about the size of the +52 MB.
- **No growth with the region count.** The fixed tree is flat from 552 to 138
  regions (699532 against 689520 KiB). A per-region cost would have made it
  fall.
- **The extra cost is bounded.** It is a bounded allocator effect, never more
  than the old full crop stack. Each payload row is one run, so these
  differences carry single-run noise.
- **The orchestrator closed I4** on this evidence.

## Housekeeping

- **Cancelled gate.** Jobs 32324973–78, the first gate at `93c049d39`, were
  cancelled while still pending: the orchestrator's answers made that tree
  stale. Their trees were removed.
- **Moved jobs.** The orchestrator moved four gate jobs from `short` to `batch`,
  with their ids unchanged, and moved the golden to `intel`.
- **A cleanup that removed only one tree.** Cleanup 32335180 removed the rss3
  "before" tree but not the "after" tree, which it had only partly deleted.
  That tree, `c5r-282-rss3-after-fc131d9ff`, was removed by hand afterwards.
  No `c5r` worktree remains.
