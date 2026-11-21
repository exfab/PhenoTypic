# #282 P2 Task 4: GPU batch invariance (DINOv2, three GPU models)

**Verdict: stop rule 3 is not triggered.** On every GPU model and row:

- every stored float16 vector is bit-identical to the reference in all 8 perturbation cases;
- the separate-process repeat is bit-identical;
- there is no determinism warning and no float16 overflow.

P2 passes its batch-invariance gate, and P4 may start.

- **Code under test:** `2effad485`, in a detached worktree at
  `/bigdata/exfab/anguy344/gate-worktrees/c6-2effad485`. Every job printed
  `dirty=0` at start and end, and every result JSON records `"dirty": false`.
  The finalize job removed the worktree (`cleanup exit 0`).
- **Scripts:**
  - `docs/superpowers/plans/2026-11-12-gpu-embedding-staging/batch_invariance.py`
  - `batch_invariance.sbatch`
  - `submit_batch_invariance.sh`
- **Results and logs:** `/bigdata/exfab/anguy344/gate-results/c6-282/`.
  - Per GPU: `<gpu>.json`/`.npz` and `<gpu>-repeat.json`/`.npz`.
  - The cross-GPU table is `cross_gpu.json`.
  - Logs are named `<job-name>-<id>.log`.
- **Source of the numbers:** every number below is read from those JSONs. Runtimes are the JSONs' `row_s` and `total_s`.

## Setup

- **Inputs.** The three corrected Salt_081 stores from spec §2, read only.
  - Regions: the control pipeline's colonies (`objects`, 66 per image) and the
    8 × 12 rig1 grid (`wells`, 96 per image).
  - That gives 486 stored vectors per row: 198 colonies and 288 wells.
  - Encoder inputs: crop rows run 486 crops at 224 px. `masked_dense` runs
    2562 tiles in 7 geometries, the largest being 2316 tiles of 224 × 224.
- **Rows** (DINOv2 only, U11):
  - base float32 `crop_cls`;
  - base float32 `masked_dense`;
  - large float32 `crop_cls`;
  - giant bfloat16 `crop_cls`.
- **Physical batch m.** m is `physical_batch_size` of the GPU's total memory,
  with the embedder's `batch_size = 64` as the cap. It is 64 for every GPU and
  row, so m/2 = 32.
- **Determinism settings**, as `configure_determinism` reported them in every row:
  - `cudnn_deterministic: true` and `cudnn_benchmark: false`;
  - both TF32 flags false;
  - `float32_matmul_precision: highest`;
  - `deterministic_algorithms: true`, with `warn_only: true`;
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Software:** torch 2.11.0+cu130, CUDA 13.0, the foundation venv; HF offline.
- **Cases**, each at the same m. The reference submits one ticket per
  (region, image), in order, as Stage 4 does.

  | Case | What changes |
  |---|---|
  | `reversed` | row order reversed |
  | `shuffled` | rows in a seeded random order |
  | `interleaved` | each row's neighbours come from the other plates |
  | `position` | every row once at position 0 and once at position m−1 of a full batch; every other row in those batches is checked too |
  | `pad_1` | the row's last batch is zero-padded with 1 zero |
  | `pad_m_minus_1` | the row's last batch is zero-padded with m−1 zeros |
  | `shards` | images split across two `BatchEncoder` instances on one model |
  | `in_flight_1` | `max_in_flight_bytes=1`, so every submission flushes every queue |

  In every case except `shards` and `in_flight_1`, each row is its own ticket,
  so its batch and position are known exactly.

- **What "invariant" means.** Every stored float16 vector is bit-identical to
  the reference.
  - For crops, this is checked on every occurrence. That is 31,104
    occurrences in `position` alone.
  - For `masked_dense`, each full pass is pooled and compared. Every tile
    occurrence's float32 patch grid was also compared bit for bit: 163,968
    in `position`, with 0 mismatches on every GPU.

## Ada: NVIDIA RTX 6000 Ada Generation (capability 8.9)

- Measure job 32297727 (`exfab`, gpu12); total 1210.59 s.
- Repeat job 32297776 (`exfab`, gpu12); total 150.63 s.

| Row | Invariant (8 cases) | Repeat identical | m | m/2 result | m/2 max cos distance | Determinism warnings | f16 max abs | f16 overflow | Row time |
|---|---|---|---|---|---|---|---|---|---|
| base `crop_cls` fp32 | yes | yes | 64 | identical | 0 | none | 10.84 | no | 101.88 s |
| base `masked_dense` fp32 | yes | yes | 64 | identical | 0 | none | 9.98 | no | 441.31 s |
| large `crop_cls` fp32 | yes | yes | 64 | identical | 0 | none | 8.30 | no | 291.89 s |
| giant `crop_cls` bf16 | yes | yes | 64 | 486/486 rows differ | 3.18e-4 | none | 7.69 | no | 254.34 s |

## A100: NVIDIA A100-SXM4-80GB (capability 8.0)

- Measure job 32297789 (`short_gpu`, gpu08); total 1430.27 s.
- Repeat job 32297801 (`short_gpu`, gpu08); total 212.0 s.

| Row | Invariant (8 cases) | Repeat identical | m | m/2 result | m/2 max cos distance | Determinism warnings | f16 max abs | f16 overflow | Row time |
|---|---|---|---|---|---|---|---|---|---|
| base `crop_cls` fp32 | yes | yes | 64 | identical | 0 | none | 10.84 | no | 143.09 s |
| base `masked_dense` fp32 | yes | yes | 64 | identical | 0 | none | 9.98 | no | 591.38 s |
| large `crop_cls` fp32 | yes | yes | 64 | identical | 0 | none | 8.30 | no | 361.6 s |
| giant `crop_cls` bf16 | yes | yes | 64 | identical | 0 | none | 7.69 | no | 182.66 s |

## Blackwell: NVIDIA RTX PRO 6000 Blackwell Server Edition (capability 12.0)

- Measure job 32297813 (`short_gpu`, gpu16); total 901.31 s.
- Repeat job 32297823 (`short_gpu`, gpu17, a different node from the measure job); total 332.48 s.

| Row | Invariant (8 cases) | Repeat identical | m | m/2 result | m/2 max cos distance | Determinism warnings | f16 max abs | f16 overflow | Row time |
|---|---|---|---|---|---|---|---|---|---|
| base `crop_cls` fp32 | yes | yes | 64 | 486/486 rows differ | 4.86e-9 | none | 10.84 | no | 216.03 s |
| base `masked_dense` fp32 | yes | yes | 64 | 485/486 rows differ | 6.27e-9 | none | 9.98 | no | 236.82 s |
| large `crop_cls` fp32 | yes | yes | 64 | identical | 0 | none | 8.30 | no | 129.74 s |
| giant `crop_cls` bf16 | yes | yes | 64 | 486/486 rows differ | 4.29e-4 | none | 7.69 | no | 138.73 s |

The m/2 float32 max abs difference on Blackwell:

| Row | m/2 f32 max abs |
|---|---|
| base `crop_cls` | 2.60e-5 |
| base `masked_dense` | 1.24e-5 |
| giant bf16 | 0.156 |

For giant bf16 on Ada it is 0.139. Row times include the model load: the
first load in each job was cold, 119.8 s for base on Blackwell.

**Within a row, across all three GPUs and every case:**

- `f32_bit_mismatch_occurrences` is 0;
- `f16_mismatch_occurrences` is 0 for crops;
- `halved` is false;
- `other_warnings` is empty.

## Cross-GPU distance (`cross_gpu.json`)

The cosine distance is between stored float16 vectors, over 486 rows per row.
A row whose stored bits are identical counts as 0.

| Row | Pair | Median cos | Max cos | Bit-identical rows | Max abs |
|---|---|---|---|---|---|
| base `crop_cls` | Ada vs A100 | 3.68e-10 | 4.63e-9 | 0 | 0.0039 |
| | Ada vs Blackwell | 2.74e-10 | 4.80e-9 | 0 | 0.0039 |
| | A100 vs Blackwell | 4.21e-10 | 5.09e-9 | 0 | 0.0039 |
| base `masked_dense` | Ada vs A100 | 9.62e-11 | 6.76e-9 | 1 | 0.0039 |
| | Ada vs Blackwell | 5.94e-11 | 3.29e-9 | 1 | 0.0020 |
| | A100 vs Blackwell | 1.18e-10 | 6.66e-9 | 0 | 0.0039 |
| large fp32 | Ada vs A100 | 7.71e-10 | 2.27e-7 | 0 | 0.0039 |
| | Ada vs Blackwell | 5.28e-10 | 3.71e-8 | 0 | 0.0039 |
| | A100 vs Blackwell | 8.69e-10 | 1.46e-7 | 0 | 0.0039 |
| giant bf16 | Ada vs A100 | 3.14e-5 | 1.77e-4 | 0 | 0.133 |
| | Ada vs Blackwell | 3.30e-5 | 3.16e-4 | 0 | 0.141 |
| | A100 vs Blackwell | 3.23e-5 | 2.14e-4 | 0 | 0.128 |

## Reading

1. **Batch invariance holds as restated (S1).** At one GPU model and one m, a
   row's stored float16 vector does not depend on any of these:
   - its neighbours;
   - its position (0 or m−1);
   - the padding of its batch;
   - the shard split;
   - forced early flushes;
   - the process: the Blackwell repeat ran on a different node.

   So the spec's statement that vectors do not depend on sharding stands, and
   the §5 fallback (m = 1 per crop) is not needed.
2. **Halving m (OOM) changes bits on some models.**
   - float32 rows:
     - Ada and A100: unchanged (identical);
     - Blackwell, base rows: at most 6.27e-9 in cosine distance.
   - giant bf16:
     - Ada: up to 3.18e-4;
     - Blackwell: up to 4.29e-4;
     - A100: identical.

   Recording the halving per shard (spec §4.4) is therefore needed provenance
   for bf16.
3. **Across GPU models**, float32 differs by at most 2.27e-7 in cosine
   distance (large, Ada vs A100). bf16 giant differs by up to 3.16e-4. The
   float32 figure agrees with spec §2's "about 1e-7".
4. **No float16 range problem for ViT-g.** The largest stored value is 7.69,
   against 65504.
5. **The user's decision**, recorded in spec H7 by `5b54e0b30`: float32 stays
   the default and the recommendation for every size. bf16 is an opt-in,
   documented with the bf16 numbers above.

## Deviations from the Task 4 text

1. **Results location.** Results are under `gate-results/c6-282/`, per the
   cluster brief, not under `gate-results/c6-batch-invariance/`.
2. **Repeat dependency.** The repeat depends `afterany` on its measure job and
   checks for itself, instead of `afterok`.
   - This cluster does not set `kill_invalid_depend`. After a failed measure
     job, an `afterok` repeat would pend forever, and the `afterany` finalize
     job would pend behind it.
   - The repeat runs the reference only. It fails if the measure job's `.npz`
     is missing, or if its GPU name differs.
3. **Warnings.** They are recorded (`catch_warnings(record=True)`) rather than
   raised (`simplefilter("error")`). A row still fails on any determinism
   warning, and every warning is listed. The row finishes instead of aborting
   at the first one. No warning of any kind was recorded in any row.
4. **No pre-stage job (Step 3).** dinov2-base, -large and -giant were already
   cached in `~/.cache/huggingface`, so the jobs ran with `HF_HUB_OFFLINE=1`
   and no separate weight pre-stage job ran.
5. **How m is computed.** m is the shipped `physical_batch_size`, with
   `weight_bytes` from the loaded parameters and `per_sample_bytes` from the
   model config. The embedder's `batch_size = 64` caps it, so m = 64 on every
   GPU. Task 4 does not fix the per-sample inputs, so P4's worker should
   compute m the same way.

**Pre-run smoke.** A one-image smoke at `a52f86ed7` ran on all three GPUs
(jobs 32296059, 32296164 and 32296166). Its `masked_dense` row stopped on an
assertion in the script, not in shipped code. The script had assumed both
regions cut the same tiles, but the control pipeline crops the plate before
detection. `5b5a5a38f` made each (region, image) its own unit. The smoke's
`base_crop_cls` row was invariant in every case.
