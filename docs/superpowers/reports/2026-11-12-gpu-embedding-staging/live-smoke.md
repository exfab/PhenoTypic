# #282 P5 Task 4: live SLURM Stage-4 smoke (C16)

**Verdict: pass.** The no-detector run (attempt 2) was walltime-killed in Stage-4 round 1
and finished in round 2, with no image embedded twice. The detector run (attempt 3)
completed Stages 1–4. In both, every image is accounted for by an independently
read record or a journalled failure, and `phenotypic-wait` returned the run's own code.
The smoke found no defect in Stage 4 itself. It found four product issues around it
(F1–F4), and four observations for the P5 review and the docs (F5–F8).

- **Code under test:** `e82dedefacb23e9f9198f20a8836febf4187e050`, one detached worktree
  per attempt or probe under `/bigdata/exfab/anguy344/gate-worktrees/c16-e82dedefa-*`.
  Every job printed `phenotypic.__file__` inside its worktree, with `dirty=0`.
- **Environment:** the foundation venv
  (`/bigdata/exfab/anguy344/gate-results/c18/venv-foundation`, torch 2.11.0+cu130), with
  `PYTHONPATH=<worktree>/src:<worktree>`. The venv's editable install points at another
  tree, so PYTHONPATH is what selects the code. `HF_HUB_OFFLINE=1` and
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`; no token file was read.
- **Script:** `docs/superpowers/plans/2026-11-12-gpu-embedding-staging/live_smoke.sh`
  (`setup`, `pipelines`, `probe`, `launch`, `verify`, `cleanup`).
- **Results, logs and every number below:**
  `/bigdata/exfab/anguy344/gate-results/c16-live-smoke-20261121/`, a new folder that `setup`
  refuses to reuse. Each run has its own subfolder. The checks are
  `attempt-*/verify-*.json`, written by `live_smoke.sh verify`. It is an independent reader
  (stdlib, zarr 3.1.5, pyarrow) and never imports `phenotypic`. The final run of it over all
  three runs, job 32368949, exited 0 for each.

## Inputs

- **Source:** the Maresca migrated copy,
  `/bigdata/exfab/anguy344/projects/ucr_029_e_d_Maresca/data/results/2026-08-11-ome-zarr/results/7-24-26_redo_full/zarr/`,
  which holds 6657 `*.ome.zarr` stores.
- **Subset:** the first 400 by name (Python code-point sort), `d000466_280_001_2026-07-24_18-32-48`
  to `d000466_280_007_2026-07-31_06-38-09`. The exact list is `input/subset.txt`; its sha256
  is in `input/subset.txt.sha256`, and its summary in `input/subset_summary.json`.
  - All 400 have an rgb level 0 of `3 × 3132 × 5086` uint16.
  - Together they hold 77.2 GB (`du -sb`).
- **How the CLI gets the subset, read only:** `--input <zarr dir> --image-manifest input/subset.txt`.
  The CLI's own `--image-manifest` selects exactly those entries from the scan, and the
  preflight printed `selected 400 of 6657 image(s)`. No store was copied or linked, so
  symlinked stores never came up.
- **Untouched:** every `verify` checks that no subset store's root `zarr.json` changed after
  `setup`. The newest is 2026-09-25T16:18:54.

## Pipelines (`pipelines/v1`–`v3`, written by the code under test)

Both runs measure `MeasureSize` and two `MeasureDinoEmbedding(size="small",
revision="ed25f3a31f01632728cabb09d1542f84ab7b0056")`:
- `emb_obj` on objects;
- `emb_grid` on a 2 × 2 `ManualGridFinder`, with rows `[0, 1566, 3132]` and columns `[0, 2543, 5086]`.

Both use `--image-type gridimage`. The first probe, with `--image-type image`, was refused
by `PF-GRID-IMAGE`, because a grid embedder needs a GridImage.

| Version | nodet ops | det ops | Used by |
|---|---|---|---|
| v1 | `OtsuDetector` | `FakeGpuDetector(threshold=0.5)` | probe-2 only |
| v2 | `BlurGauss(5)`, `OtsuDetector`, `RemoveSmallObjects(600)`, `RemoveBorderObjects(20)` | the same with `FakeGpuDetector(threshold=0.5)` | attempts 1 and 2 |
| v3 | as v2 | `FakeGpuDetector(input_layer="detect_mat", threshold=0.6)` in Otsu's place | attempt 3 (det) |

- **v1 → v2:** on these plates `OtsuDetector` alone labels **65535** objects, the uint16
  ceiling (probe-2, diag 32346026). The blur and the two refiners are the plates' own
  pipeline's, from `UCR_029_E_D_Maresca_v14.json.pht-pipe`. With them, Otsu gives 23 objects
  on the first plate.
- **v2 → v3:** see F3.

`job_metadata.json` records `embed_revisions: {}` in every run. That is correct:
`resolve_embed_revisions` (`_cli_embed_stage.py:536`) records only unpinned embedders, and
both are pinned.

## Probes (CPU, before any GPU job)

| Probe | What | Result |
|---|---|---|
| probe-1 | v1, `--image-type image` | refused: `PF-GRID-IMAGE` |
| probe-2 | v1, gridimage | the dry-run passed; the 2-image local run was cancelled after 29 min (65535 objects) |
| probe-3 | v2 | both 400-image SLURM `--dry-run`s exit 0 and create no output; both 2-image local CPU runs exit 0, `PROCESSING COMPLETE`, in 2:45 and 3:02, at peak RSS 7.46 and 7.80 GB |
| probe-4-threshold | v3 threshold sweep | see F3 |

**Loader cost on CPU** (`probe-3/loader_timing.json`, Stage-4's per-image work):

| | Load | Objects prepare | Grid prepare |
|---|---|---|---|
| plate 1 | 1.42 s | 0.54 s | 3.61 s |
| plate 2 | 1.19 s | 0.46 s | 3.59 s |

That is about 5.3–5.6 s per image. In the one-image timing, Stage-2-like fake detection
took 3.6–4.8 s per image locally.

## Profiles

- **CPU stages, controllers and finalizer** (`--slurm`): `intel`, 2 CPU, `mem_gb=16`, `time=60`.
- **GPU** (`--gpu-slurm`): `slurm_cpus_per_task=2`, so one loader worker, `mem_gb=24`, and
  `--gpu-shards 2`.
  - Attempts 1–2: `slurm_partition=exfab slurm_account=exfab`.
  - Attempt 3: `slurm_partition=short_gpu slurm_constraint=gpu_latest slurm_qos=short_gpu`.
- **Partition choice:** each launch took the partition whose `sbatch --test-only` predicted
  the earlier start.
  - 02:09: exfab 03:19, short_gpu next day 10:12.
  - 08:46: exfab 2026-11-26T20:32, short_gpu 2026-11-22T05:55.
- **Manual move in attempt 3:** det's Stage-2 array 32354705 sat PENDING (Priority) on
  short_gpu. At 18:39 the orchestrator moved it to exfab with
  `scontrol update JobId=32354705 Partition=exfab Account=exfab QOS=normal Features=""`,
  when `--test-only` predicted exfab at 19:20 and short_gpu at 12:16 the next day. Its Stage-4
  array was scheduled on short_gpu (gpu09) by the scheduler, unmoved.
  So **det's Stage 2 ran on exfab after a manual move**, and its Stage 4 ran on short_gpu.
- Never `preempt`.

## Walltime sizing (Step 2)

- **Attempt 1:** sized from the spec §2 cold first call (650–750 s) and
  `batch-invariance.md`'s cold load (119.8 s on Blackwell). With the probe-3 loader cost,
  that gave about 1100 s of work per 200-image shard.
  - For round 2 to finish, round 1 must commit at least half of each shard, which needs
    `T ≥ L + W/2` and `T < L + W`.
  - Since L was uncertain, I hedged: nodet 17 min, det 24 min.
- **The measurement** (nodet attempt 1, `processing_events.log` against each shard's `sacct`
  start, on gpu12, RTX 6000 Ada):

  | Shard | First commit | Last commit | Per image |
  |---|---|---|---|
  | 0 | 112.7 s | 670.9 s | 2.81 s |
  | 1 | 39.7 s | 584.6 s | 2.74 s |

  Both shards finished well inside 17 min (11:16 and 9:46), so attempt 1 had no walltime kill.
- **Attempt 2** (nodet only) used `time=8` (480 s):
  - expected round 1: (480 − 113)/2.8 ≈ 130 to (480 − 40)/2.8 ≈ 157 images per shard;
  - expected round 2: 43–70 images per shard, which fits.
  - **Measured:** round 1 committed **335** (both shards TIMEOUT, at 8:14 and 8:19), and
    round 2 committed **65** (33 + 32, COMPLETED in 2:04 and 1:57).
- **Attempt 3** (det only) used `time=30`, sized to finish without a kill, since attempt 2
  already covers the recovery.

## The three attempts

### Attempt 1: nodet passes, but round 1 did not time out; det v2 cancelled

- **nodet** (`attempt-1/verify-nodet.json`):
  - waiter 32346264 exit 0;
  - `stage4-round-1` 32347872 COMPLETED;
  - 400 of 400 records valid;
  - no Stage-4 round 2, so it does not show the walltime recovery.
- **det v2** (waiter 32346265): Stages 1–3 ran, then Stage-4 rounds 1, 2 and 3
  (32349217, 32350858, 32351970) each hit TIMEOUT at 24 min.
  - Commits per shard-round, by time window: 11 and 2; 2 and 1; 2 and 1. That is 19 of 357 images.
  - The orchestrator cancelled the run during round 4 (32353469), at 08:42. Cause: F3.
  - Stage 3 also showed F2.

### Attempt 2: nodet, the walltime recovery (spec §4.8)

From `attempt-2/verify-nodet.json`, `failed_checks: []`:

- **Ledger** (`.phenotypic/progress/slurm_jobs.jsonl` with `sacct`):

  | Role | Token | Job | Result |
  |---|---|---|---|
  | controller-initial | | 32350966 | COMPLETED |
  | stage3 | `stage3-chunk-0` | 32350982 | COMPLETED |
  | stage4 | **`stage4-round-1`** | 32351573 | **TIMEOUT** ×2 (8:14, 8:19), gpu12 |
  | stage4 | **`stage4-round-2`** | 32352307 | **COMPLETED** ×2 (2:04, 1:57), gpu12 |
  | finalizer | | 32353554 | COMPLETED |
  | controller (trailing) | | 32353546 | FAILED, see F1 |

- **Round 2's worklist** on disk is round 2 with shards `[33, 32]`. The shard logs say
  `33 published, 0 failed, 0 left for the next round, 64 signal(s) written` and
  `32 published, …`. Round 1's shards were killed before writing a summary.
- **No re-embedding:** Stage-4 completions were 335 in round 1 and 65 in round 2.
  - `images_completed_twice: 0`.
  - `started_again_after_completion: 0`: no image that committed in round 1 has a Stage-4
    event inside a round-2 shard's run window.
- **Records:** 400 of 400 valid. Each record:
  - has `version` 1;
  - has a `work_id` equal to the manifest's;
  - has a `lifecycle_epoch` equal to the ledger epoch;
  - has `stages.stage3`;
  - has every artifact's sha256 and size equal to the bytes on disk;
  - has a store digest equal to sha256 of the store's root `zarr.json`.
- **Stores:**
  - opened with `zarr.open_group(mode="r")`;
  - the latest run carries exactly `dinov2-vits14/rgb/grid` and `dinov2-vits14/rgb/objects`,
    both `stale: false`;
  - every file sha256 matches;
  - pyarrow reads 384-d finite vectors;
  - the objects table's labels equal the measurement table's, and the grid has 4 rows.
- **Compute:** all 800 entries have:
  - `stage: stage4` and `device: cuda`;
  - `device_name` "NVIDIA RTX 6000 Ada Generation", `capability` 8.9;
  - torch 2.11.0+cu130 and cuda 13.0;
  - **`physical_batch` 64**, with no `physical_batches` (no halving).
- **Signals:** 0 Stage-4 signal files left.
- **No-detector shape:**
  - `slurm_scripts/` holds `controller.sh`, `finalizer.sh`, `stage3.sh`, `stage4.sh`, the
    worklist, the config and the manifest, and **no `stage1.sh`/`stage2.sh`**;
  - `staged_controller.json` is `version: 2`, **`phases: ["stage3", "stage4"]`**, with
    `stage3_role: single_pass`.
- **Waiter:** 32350610 `phenotypic-wait exit=0`.
- **Cleanup:** 32350611 printed `worktree removed: …/c16-e82dedefa-a2`. But see F1.

### Attempt 3: det v3, all four stages

From `attempt-3/verify-det.json`, `failed_checks: []`:

- **Ledger:**

  | Role | Token | Job | Result |
  |---|---|---|---|
  | stage1 | `stage1-chunk-0` | 32354222 | 400 COMPLETED |
  | stage2 | `stage2-round-1` | 32354705 | COMPLETED (14:43, 7:22), gpu12, **exfab after the manual move** |
  | stage3 | `stage3-chunk-0` | 32362992 | 400 COMPLETED |
  | stage4 | `stage4-round-1` | 32363499 | COMPLETED (19:57, 13:57), gpu09, short_gpu |
  | finalizer | | 32368355 | COMPLETED |
  | controller (trailing) | | 32368285 | FAILED, see F4 |

- **Events:**
  - Stage 1: 400 started, 400 completed.
  - Stage 2: 400 started, 400 completed.
  - Stage 3: 400 started, 378 completed, 22 failed.
  - Stage 4: 378 started, 378 completed.
- **Shard logs:** `189 published, 0 failed, 0 left for the next round` on each.
  - First commit at 297.9 s and 66.4 s; then 4.60 and 4.09 s per image.
  - The two shards ran one after the other on gpu09.
- **Exit 3 is the journalled failures:**
  - `terminal_failures.jsonl` holds **22** rows, all `failed_stage: stage3` with
    `NoObjectsError` from MeasureSize: plates on which the v3 fake detector found nothing.
  - Each row's `work_id` names a manifest image with no full record.
  - **378 verified + 22 failed = 400, 0 unresolved.**
  - The waiter printed `22 image(s) failed terminally` and `COMPLETE_WITH_FAILURES (exit 3)`.
  - `metrics.json` has `accepted 400, failed 22, succeeded 378`.
- **Records:** 378 of 378 non-failed images valid, under the same checks as attempt 2.
- **Compute:** 756 entries, all "NVIDIA RTX 6000 Ada Generation", 8.9, `physical_batch` 64.
- **Deliverables:** `deliverables/embeddings/dinov2-vits14/rgb/objects.parquet` has 13952 rows,
  equal to `objects.measured`. `grid.parquet` has 1512 rows, which is 378 × 4.
- **Signals:**
  - 0 Stage-4 signals left.
  - **44 Stage-2 signal files left**: a raw array and a token for each of exactly the 22
    Stage-3 failures (set equality checked). I read this as the GPU result kept for a
    `--retry-failures` replay, but I did not find it documented; worth a line in
    `_cli/CLAUDE.md`.
- **Shape:** all of `stage1.sh`–`stage4.sh` are present; `phases` holds all four stages and
  `stage3_role: staged`.
- **No walltime kill on det, by design.** Attempt 2 covers the recovery, which is the same
  controller logic with or without a detector. det's job here was Stages 1, 2 and 4.
- **Waiter:** 32353844 `phenotypic-wait exit=3`; `sacct` shows FAILED because the exit is
  non-zero.
- **Cleanup:** 32353854 is ledger-aware (F1). It logged
  `waiting on ledgered jobs: 32368285_0` until the trailing controller ended, then
  `worktree removed: …/c16-e82dedefa-a3`.

## Findings (product; not fixed in this cluster)

**F1. `phenotypic-wait` exits before the run's last recovery controller has run.**
- Every controller pre-arms an `afterany` successor, including the one that submits the
  finalizer.
- Once the finalizer closes the fence (`closed_by: finalizer`), `live_slurm_job_ids` answers
  empty without asking the scheduler, so the waiter exits. The trailing controller may still
  be queued.
- In attempt 2:
  - the finalizer finished at 08:25:19, and the waiter exited 0 at 08:25:33;
  - my `afterany`-on-the-waiter cleanup removed the worktree from 08:25:51 to 08:25:56;
  - trailing controller 32353546 started at 08:25:51 and died at 08:25:54 with
    `ModuleNotFoundError: No module named 'phenotypic.sdk_.mixin._input_layer_mixin'`.
- The outputs were already published, so this was harmless here. But any teardown triggered
  by the waiter can race the trailing controller: a DVC stage's environment, a venv swap, a
  worktree cleanup.
- Any cleanup must wait on every ledgered job, which `live_smoke.sh cleanup` now does.
  Document it beside `phenotypic-wait`, or have the waiter also wait for ledgered jobs that
  are still queued.

**F2. A Stage-3 task killed by OOM is never resubmitted, and leaves no record and no journal row.**
- In det v2, four Stage-3 tasks (32348265_177, _178, _222, _224) ended `OUT_OF_MEMORY` at
  16 GB after 9–12 min.
- The ledger shows `stage3-chunk-0` was never resubmitted. The four images have no
  measurement table, no record and no row in `terminal_failures.jsonl`.
- Stage-4 round 1's events cover 357 images, which is 400 − 39 journalled `NoObjectsError` − 4,
  consistent with §4.3's "terminal on missing `stages.stage3`".
- The run was cancelled before finalizing, so I cannot show what it would have reported.
  By the waiter's documented rule, `3` needs `verified + failed == accepted`, so these four
  would most likely have made it **exit 5**, `ENDED_UNPUBLISHED`. That is an inference, not
  an observation.
- A host OOM is node state, not a per-image verdict (compare §4.8 for Stage 4). Whether the
  controller should retry such a Stage-3 task is for the P5 review.

**F3. Objects that sprawl make crop prep about 100× slower, on both paths, and no stop rule catches it.**
- v2's det thresholds raw `rgb`; its `input_layer` defaults to rgb, so the blur never reaches
  it. It labels thin networks that span the plate.
- Padded crop sides, read from the stores' measurement tables (1.5 × the longer bbox side):

  | | Objects | Median padded side | p90 | Max | Objects ≥ 2000 px | Crop area per image, median | Max |
  |---|---|---|---|---|---|---|---|
  | det v2 | 18,873 | 4171 px | 6556 | 7534 | 13,528 | 163 Mpx | 5,502 Mpx |
  | nodet v2 | 17,335 | 151 px | 357 | 1725 | 0 | 1.7 Mpx | 7.1 Mpx |

- `_crop` pads each box and squares it, then resizes it to 224 with skimage
  (`_measure_dino_embedding.py:744-803`). The inline path makes the same calls.
- **Measured cost:**
  - A det v2 shard used one core: `sstat` AveCPU 8:04 at 8:18 elapsed.
  - A CPU timing of one det v2 store (195 objects, diag 32353653) ran for 35:53 before it
    failed on an import, after the object crops were cut. Its tree had been removed
    underneath it. So one image's load and object-crop prep took up to about 35 min.
- Every round made some progress, so the "two unchanged retryable sets" rule never fires.
  At 1–11 images per shard-round, such a run goes on for days.
- **Why v3:** a CPU sweep (`probe-4-threshold/threshold_sweep.json`, plates 1, 150 and 300 of
  the subset) compared the options:
  - `input_layer="detect_mat", threshold=0.6`: 57 / 49 / 19 objects, 5.39 / 2.18 / 0.57 Mpx,
    largest padded side 2017 px;
  - Otsu: 23 / 48 / 18 objects, 0.17 / 1.89 / 0.53 Mpx;
  - 0.3–0.5 gave 0 objects on two or three of the plates, and 0.7 gave 0 on plate 1.
- This is a robustness and efficiency note, not a defect. A preflight or a per-image warning
  on crop area, or a cap on the padded side before resizing, would turn "days" into one
  visible line.

**F4. The trailing recovery controller exits 1 after a run with terminal failures.**
- `run_staged_controller` returns quietly only when the orchestration phase is in
  `{"complete", "cancelled", "failed"}` (`_cli_staged_controller.py:571`). Otherwise it calls
  `assert_active_epoch` (`:573`).
- A run that finalizes with terminal failures is closed as `"terminal_incomplete"`
  (`_cli_checkpoint_handler.py:404`).
- So in attempt 3, trailing controller 32368285 raised
  `SlurmGenerationInactiveError: Staged orchestration 5f67ab5f… is fenced inactive` and
  ended FAILED after 9 s.
- In nodet attempt 1 the same controller found `phase: complete` and COMPLETED in 5 s.
- This is benign for the outputs. But every SLURM run with even one per-image failure leaves a
  FAILED controller in `sacct`, which reads as a crash. `"terminal_incomplete"` probably
  belongs in the quiet set.

**F5. No staged event carries a SLURM job or task id.**
- In every run, the 6th and 7th fields of `processing_events.log` are empty on every row:
  `stage1`, `stage2`, `stage3` and `stage4` alike (all 4 runs, every row). So an event cannot
  be tied to a round or a shard.
- `append_event` accepts `slurm_job_id`/`slurm_array_task_id` (`_cli_update_state.py:76`), but
  these callers pass none.
- This smoke therefore assigns events to rounds by time window against `sacct`, and records
  `stage4_events_outside_any_task_window: 0`.
- Recording the ids would make "not re-embedded in round 2" a direct read. Flagged for the
  P5 review.

**F6. The cold load is far below spec §2.**
- On gpu12 the first Stage-4 commit came 112.7 s and 39.7 s after the shard started. On gpu09
  (attempt 3) it came 297.9 s and 66.4 s after.
- Steady state was 2.74–2.81 s per image (nodet) and 4.09–4.60 s (det v3), with one loader
  worker, which is the loader's CPU cost.
- Spec §2's "650–750 s" is not what dinov2-small costs on a GPU node with cached weights and
  `HF_HUB_OFFLINE=1`.
- `PF-EMBED-SHARD-AMORTIZE` quotes "about 750 s". It fired for 200 images per shard, against a
  threshold of about 439, on runs whose load was under 5 min.
- The docs, and that warning's constant, should be revisited against these numbers.

**F7. `--dry-run` labels a staged run "SLURM (autonomous)".**
- Both probe-3 dry-runs printed `Selected backend: SLURM (autonomous)`, the det one with a
  `GpuDetector` included (`_cli_interactive.py:49`).
- The real submissions took the staged controller, so the preview is cosmetic but misleading.

**F8. Submission over store inputs reads the input about three times.**
- Each waiter spent 31–37 min before its first controller was ledgered:
  - attempt 1: 02:09:39 → 02:46:48;
  - attempt 2: 05:57:31 → 06:29:13;
  - attempt 3: 08:47:17 → 09:20:14.
- `sacct` MaxDiskRead for attempt 1's two waiters was 224947.89M and 221019.87M, against
  77.2 GB of input. `file_sha256` digests the whole store tree
  (`_cli_failure_tracker.py:94-125`).
- This is the repeated input hashing P3–P4 review already lists as Important. Here it is
  measured on real stores.

## Deviations from the Task 4 text

1. **Input:** an existing OME-Zarr store tree, chosen by the user, and not a raw dataset.
2. **Pipelines:** each detector is bracketed by the plates' own blur and refiners (v2), and
   det uses a detect_mat fake (v3). The reasons are under "Pipelines" and F3.
3. **Three attempts, not one run:**
   - attempt 1 was sized from spec §2 and did not time out;
   - attempt 2 re-ran nodet alone, sized from attempt 1's measured load;
   - det v2 was cancelled and re-run as attempt 3, without a walltime target.
   The ledger's TIMEOUT-then-round-2 is shown on nodet (no detector) only.
4. **GPU partition:** attempt 3's Stage 2 was moved to exfab by hand (above). Its Stage 4 ran
   on short_gpu.
5. **Cleanup:** the plan asks for an `afterany` cleanup. Attempts 1 and 2 used one on the
   waiter, which F1 shows is not enough. Attempt 3's waited on every ledgered job.
6. **Commit timing:** the script and this report were committed together after the last run.
   The script gained `pipelines v3`, `verify`, ledger-aware `cleanup` and `-` (skip a run)
   during the smoke. The copies that ran are kept beside the results
   (`attempt-*/live_smoke.verify.sh`, `live_smoke.sh`, `live_smoke.verify-final.sh`).

## #282 P6 Task 2 Step 5: SLURM `--mode measure --embed-stage gpu` over attempt 2 (2026-11-24)

**Verdict: pass.** A same-pipeline SLURM measure resubmission over attempt 2's nodet
output measured and re-embedded all 400 images on GPUs. It republished every record under
the measure epoch, and `phenotypic-wait` returned 0. `measure-verify` (job 32427337) exited
0 with `failed_checks: []`. I found no product defect.

- **Code under test:** `d2730b345db268c2448a71a8aa924bef0ef75c52`, in one detached
  worktree, `/bigdata/exfab/anguy344/gate-worktrees/c16-d2730b345-measure`. The waiter
  printed `phenotypic=<tree>/src/phenotypic/__init__.py` and `dirty=0`.
- **The forward run** was written by `e82dedefa` (attempt 2, manifest version 4). So this
  is also a v4-manifest run measured by newer code.
- **Environment:** as in P5, with `PYTHONSAFEPATH=1` added. The cwd was
  `attempt-5-measure/`, which is user-owned. `HF_HUB_OFFLINE=1` and
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` were set, and no token file was read.
- **Script:** `live_smoke.sh` gains `measure-snapshot`, `measure-dryrun`, `measure-launch`
  and `measure-verify`. The copy that ran, and every file below, are in
  `/bigdata/exfab/anguy344/gate-results/c16-live-smoke-20261121/attempt-5-measure/`.
  The checks are in `verify-measure.json`.

### What ran

- **Command:** `phenotypic-wait --output attempt-2/nodet -- python -m phenotypic --mode measure
  --pipeline pipelines/v2/nodet.json --output attempt-2/nodet --image-type gridimage
  --embed-stage gpu`, with `--gpu-shards 2`.
  - CPU profile: `--slurm` intel, 2 CPU, `mem_gb=16`, `time=60`.
  - GPU profile: `--gpu-slurm slurm_partition=exfab slurm_account=exfab time=30
    slurm_cpus_per_task=2 mem_gb=24`.
  - **Same pipeline:** the file's sha256 is `8bf15eec937d…`, which names the forward
    run's folder `1-8bf15eec937d`.
- **Partition:** at 08:07, `--test-only` predicted exfab 08:56:49 and short_gpu
  14:13:48. The Stage-4 array started on exfab at 08:31:48 and was never moved.
- **No preview:** `--mode measure` refuses `--dry-run` (`phenotypicCLI.py:2812`; probe job
  32426674 exited 2). So the first measure invocation was the real one.
- **Before the submit:**
  - `measure-launch` copied the forward run's `slurm_scripts/`, records, `progress/*.json`,
    `slurm_jobs.jsonl` and `processing_state.json` to `before-state/`, since the pass
    rewrites them.
  - It then wrote `before.json`: for every image, the record's sha256 and identity, the
    root `zarr.json` sha256, and each embedding entry's `compute` block, revision,
    `params_sha256` and file inode. It also holds the measurement table's sha256 and inode.
  - `before.json` read 400 images, manifest v4, `embed_revisions` `None` in
    `processing_state.json` and `{}` in `job_metadata.json`. Both embedders are pinned.

### Ledger (epoch `97e8d426fdcd4f569a1d86ae45a451b7`, rows appended to the forward run's)

| Role | Token | Job | Result |
|---|---|---|---|
| controller-initial | | 32426792 | COMPLETED (0:23) |
| controller | `controller-after-32426792` | 32426803 | COMPLETED |
| stage3 (measure) | `stage3-chunk-0` | 32426804 | 400 COMPLETED, 08:14:04–08:19:28, 13–156 s each, MaxRSS 1722.50M |
| controller | `controller-after-32426803` | 32427225 | COMPLETED |
| stage4 | `stage4-round-1` | 32427226 | COMPLETED ×2 (15:41, 11:41), gpu12, exfab |
| finalizer | | 32427329 | COMPLETED (2:39) |
| controller (trailing) | `controller-after-32427225` | 32427328 | COMPLETED (0:03) |

- **Submit:** the waiter launched at 08:08:56, and the initial controller was submitted at
  08:13:40, which is 4 min 44 s. P5's three forward submissions took 31–37 min (F8). This
  is not a like-for-like comparison: the inputs are the run's own 400 stores rather than
  the Maresca ones, the waiter had 8 CPUs rather than 2, and P6 M8 now hashes on threads.
- **Stage 4:** the two shards ran one after the other on gpu12. Each logged `200
  published, 0 failed, 0 left for the next round, 400 signal(s) written`.
  - From the event log: shard 0's first commit came 303 s after it started, and shard 1's
    87 s after. Both then ran at 3.1 s per image. MaxRSS was 16188688K and 15948772K,
    against `mem_gb=24`.
- **The waiter** exited `COMPLETE (exit 0)` at 09:04:05, which is after the trailing
  controller ended (09:03:43). This is U15/F1 working live: the waiter outlasts the last
  ledgered job.
  - Every poll read `completion=complete`, from the first, 6 s in. The forward records
    verify throughout, so the outcome came from the fence and the live jobs, as U16
    intends.
- **The cleanup** (32426726, `afterany` on the waiter, ledger-aware, run from the copy in
  `attempt-5-measure/`) logged `worktree removed: …/c16-d2730b345-measure`. It had no
  ledgered job left to wait on.

### Checks (`verify-measure.json`; stdlib, zarr 3.1.5, pyarrow, no `phenotypic` import)

- **`phenotypic-wait` exit 0.**
- **Controller and manifest:** the controller has `stage3_role: measure` and `phases:
  ["stage3", "stage4"]`. The manifest is version 5 with `republish: true` and 400
  images.
- **Fence:** `slurm_lifecycle.json` has the generation of the measure epoch,
  `closed_by: finalizer`, `active: false` and no `terminal_status`.
- **Job metadata and journal:** `job_metadata.json` has `run_mode: measure`. There are 0
  terminal-failure rows under the epoch.
- **Records:** 400 of 400 are valid.
  - Each record's sha256 differs from `before.json`, and its `lifecycle_epoch` is the
    measure epoch.
  - Each record's identity is unchanged: `work_id`, `attempt_id`, `relative_image_path`,
    `mode`, `provenance` and `source_provenance`.
  - Every artifact's sha256 and size match the bytes on disk, and the store digest equals
    sha256 of the root `zarr.json`.
  - A direct record comparison with `before-state/` shows that exactly `completed_at`,
    `lifecycle_epoch` and `stages.measured.at` changed, in all 400.
- **Stores:**
  - All 400 have the measure pipeline's run as their latest run (`8bf15eec937d`), with
    exactly `dinov2-vits14/rgb/grid` and `dinov2-vits14/rgb/objects`.
  - Every entry has `stale: false`, `compute.stage == "stage4"` and `device: cuda`.
  - All 800 entries ran on "NVIDIA RTX 6000 Ada Generation", 8.9, torch 2.11.0+cu130,
    cuda 13.0, with `physical_batch` 64.
  - **Every embedding file has a new inode** compared with `before.json`, and its sha256
    matches its bytes.
  - The vectors are 384-d and finite. Each objects table's labels equal its measurement
    table's, and each grid table has 4 rows.
- **U17:** every entry's `model.revision` equals the revision `before.json` recorded,
  which is the pinned `ed25f3a31f01632728cabb09d1542f84ab7b0056`. Its
  `producer.params_sha256` is also unchanged. After the pass, `job_metadata.json` still
  has `embed_revisions: {}`.
- **U18:** there are 400 measure-done markers, all under
  `progress/measure_done/97e8d426…/`, one per image, each naming the epoch.
- **Signals:** 0 Stage-4 signal files are left.
- **Events after the snapshot:** 400 each of `stage3 started`, `stage3 completed`,
  `stage4 started` and `stage4 completed`.
- **Deliverables:** `metrics.json` has `accepted 400, failed 0, succeeded 400`, objects
  `measured 17335`, and embedding rows 17335 (objects) and 1600 (grid).
- **Inputs untouched:** the newest Maresca subset root `zarr.json` mtime is still
  2026-09-25T16:18:54.

### Observations (no defect)

**M1. The pass rewrote the files and changed none of their bytes. That is the C17 case,
and it was told apart from a no-op by the epoch and the markers.**
- All 400 root `zarr.json` were byte-identical before and after (`root_changed: 0`).
- All 800 embedding files were byte-identical to the forward run's
  (`embedding_bytes_identical_to_before: 800`), and so were all 400 measurement tables.
- Yet every embedding file and every measurement table has a new inode, and 400
  `stage4 completed` events fall inside the shards' windows. So the images were re-measured
  and re-embedded, and the same model on the same GPU type produced the same bytes.
- This is the case the C17 ruling describes, a same-pipeline core leaving the root and
  table byte-identical. Only the record's `lifecycle_epoch` and the per-epoch markers
  distinguish "republished by this pass" from "still the forward record".
- A check on root or file digests alone would have read this pass as having done nothing.
  The inode check is what shows that new files were written.

**M2. `--mode measure` has no `--dry-run`.** This is refused by design
(`phenotypicCLI.py:2812`). Because the refusal comes before the other usage checks, a
first live measure cannot be previewed. Any later refusal would still come back as code 2
before anything is submitted.
