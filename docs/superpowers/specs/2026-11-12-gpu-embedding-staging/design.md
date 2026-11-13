# Follow-up: GPU execution of embeddings

**Status:** deferred · scoping stub, not a design · 2026-11-12
**Parent:** `docs/superpowers/specs/2026-11-12-dvc-embeddings-refactor/design.md`
(the parent spec defers this explicitly: §2.2 non-goals, §6 "Later").

## Why this is a separate change

The parent change ships `MeasureDinoEmbedding` on **CPU**, inside the `meas`
slot. It needs no change to the staged GPU engine, because it is not a
`GpuDetector`, so `refuse_cpu_only_slot` (`_cli/_cli_validation.py`) never fires.
The GPU route is the largest piece of the work, and choosing between its options
needs a number we don't have yet: **CPU embedding time per image** for crops and
grids at realistic colony counts.

**Start condition:** the parent's P4 is merged, and its CPU timings are measured
as a Slurm job on a fresh node and recorded beside the parent plan.

## Options carried from the parent spec (§4.4)

| Option | Where DINO runs | What changes | Gains | Costs |
|---|---|---|---|---|
| **B · Dense features from Stage 2** | GPU Stage 2 computes an objmap-independent patch-feature grid; Stage 3 pools it under the final objmap (crops) or the grid cells | a new Stage-2 signal kind beside `stage2_raw/<ds>/<slot>/`; a feature-extractor role next to `GpuDetector`; `find_gpu_detectors` and the one-GPU-op-per-pipeline limit extended (one detector + one extractor) | one resident model; reuses Stage-2 sharding, tokens, epoch fencing and recovery controllers; Stage 3 replay stays exact because the features don't depend on the objmap | `masked_dense` pooling only, no crop CLS; a transient signal of roughly 23–187 MB per image fp16 (½ to full resolution; parent feasibility page §6) on top of the detector's raw output |
| **C · GPU Stage 4** | a new GPU stage after Stage 3 crops colonies (or grid cells) from the final objmap/store | a fourth stage in the epoch-fenced controller: manifest classification, chunking against `MaxArraySize`/`MaxSubmitJobs`, ledger rows, a Stage-4 completion marker, continuation and `--restart` semantics | best embeddings (crop CLS) at GPU speed; works for pipelines with no `GpuDetector` | the largest change; a second GPU round trip per run; Stage 4 rewrites the store after Stage 3 published it, which must respect the root-last transaction and the per-image record |
| A′ · GPU inside Stage 3 | Stage 3 workers request a GPU when the pipeline has a `MeasureEmbedding` with `device="cuda"` | the Stage-3 SLURM profile gains a GPU delta | smallest change | GPUs held for the whole CPU-bound measure stage; model reloaded per worker process |

## Constraints any option must keep

- **Process mode (AutoConvertRaw-GC)** runs on CPU Spot VMs and Slurm CPU arrays.
  GPU execution there is out of scope unless AutoConvertRaw-GC gains a GPU side.
- **Per-pipeline tables** (parent Part D): output lands in the producing
  pipeline's `tables/<seq>-<sha12>/embeddings/<model>/{crops,grid}/`, written
  inside a root-last store transaction and covered by the completion marker.
- **Stage-2 signals are keyed by detector slot** (`_cli_stage2_token.py`). A
  feature signal needs its own slot namespace, so a detector and an extractor in
  one pipeline cannot collide.
- **No scheduler sidecar jobs** beside an active array. Route any helper work
  through the array or a terminal `afterany` finalizer (`_cli/CLAUDE.md`).
- **Byte-stable deliverables** (parent C11). GPU inference must not make
  `deliverables/embeddings/` differ run to run. Pin deterministic kernels, or
  measure and document the tolerance.
- **Partition choice:** `exfab` vs `short_gpu` is picked by measured predicted
  start, with `--constraint=gpu_latest` on `short_gpu` (user HPCC notes).

## Open questions for when this starts

1. Is `masked_dense` pooling (option B) good enough, or is crop CLS required
   (option C)? Answering this needs a comparison of the two on real plates, for
   example nearest-neighbour agreement between the two embeddings of the same
   colonies.
2. Should grid embeddings use the GPU route at all, given that their regions are
   fixed and known before Stage 1?
3. What CPU time per image makes GPU worth its queue wait on `exfab` and
   `short_gpu`?
