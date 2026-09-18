# Real-GPU smoke run: nested Sam2, staged vs single pass (2026-09-16)

**Verdict: staging reproduces the single pass exactly. The first comparison
failed because `DenoiseBlockMatch` (BM3D) gives different results on different
CPUs, and at different thread counts, not because of staging.**

Code: frozen checkout at `fc05e066`. Pipeline: Linzer `F1gfd5.json.pht-pipe`,
where `Sam2` sits at `CompositeDetector/ops[0]`. Inputs: 6 plates, 3 from each
dataset. Scripts: `docs/superpowers/plans/2026-09-15-nested-gpu-staging/gpu_smoke/`.
Outputs: `ucr_033_e_d_Linzer_Ganoderma/data/pipeline_dev/runs/2026-09-16_nested_staging_smoke/`.

## 1. The staged production path runs end to end

`submit_staged.sbatch` ran `python -m phenotypic --slurm ... --gpu-slurm ...`.

- The CLI detected the nested detector at `('CompositeDetector', 'ops[0]')`
  and routed the run to the staged engine.
- All five generated worker scripts named the frozen checkout's interpreter.
- Every job finished `COMPLETED 0:0`:
  - 4 controllers;
  - Stage 1 (6 tasks, on `x05`, Intel Cascade Lake, 4 CPUs);
  - Stage 2 (one `exfab` GPU task, 2m14s for all six images);
  - Stage 3 (6 tasks, on `c17`/`c20`/`c21`, AMD Opteron);
  - the finalizer.
- It published six stores plus the deliverables.

## 2. The first comparison disagreed

`compare_arms.py` compared the staged run with a single-pass reference
(`run_single_pass.sbatch`, on `gpu12`). On all six images:

- objmaps differed by 354–1,395 px;
- object counts matched;
- shape and zone measurements differed (for example, `Shape_Area` by up to
  1.1e3 px on one plate).

## 3. Isolating the cause (plate `d000415_300_028_2026-05-24_14-08-31`)

| Check | Result |
|---|---|
| Stage-1 layers: in-memory rerun on `c07` vs the staged store written on `x05` | `rgb` identical. `gray` differs (max 5.4e-5), `detect_mat` differs (max 0.023). Sam2's uint8 input differs in 1.96 M values (max 6). |
| Stage 1 run twice in one process | identical |
| `save2zarr` → `load_zarr`, after the crop and after Stage 1 | identical, bit for bit |
| Sam2 twice on the same in-memory input | identical |
| Sam2 on in-memory vs store input (the inputs differ) | raw masks differ; foreground differs by 106 px |
| **From ONE preprocessed image: live post-pipeline vs a `ReplayDetector` holding Sam2's raw result** | **objmap identical; measurements identical (largest difference 0)** |
| Live post-pipeline run twice | identical |
| Two single passes in separate jobs, both on `gpu12` | objmaps differ by 242 px |

The first four rows rule out the store and any randomness within one process.
The bold row is the staging mechanism itself, and it is exact.

## 4. Where the variation between jobs comes from

`diag_env_digests.py` ran on several nodes and at two thread counts
(sha256 prefixes):

| Node | CPU | Threads | gray after read / crop | gray after Stage 1 | detect_mat after Stage 1 |
|---|---|---|---|---|---|
| `c07` | AMD Opteron 6376 | 4 | `ef9a…` / `dfa4…` | `bd11…` | `8e93…` |
| `c07` | AMD Opteron 6376 | 8 | same | `bd11…` | `8e93…` |
| `x02` | Intel Xeon Gold 5220R | 4 | same | `5e01…` | `8676…` |
| `x02` | Intel Xeon Gold 5220R | 8 | same | `416a…` | `fe4d…` |
| `r01` | AMD EPYC 7502 | 4 | same | `eddb…` | `ab04…` |

- `gray` is identical everywhere until Stage 1.
- After the crop, the only Stage-1 operation that writes `gray` is
  `DenoiseBlockMatch` (`CompositeEnhance` writes `detect_mat` only), which
  calls the `bm3d` package. Its output depends on the CPU model, and on
  Cascade Lake also on the thread count.
- Sam2 on one fixed input, in two separate `gpu12` jobs: identical raw output
  (`f300…`, 945 labels).

The two single passes that differed on `gpu12` both had 8 CPUs, but not
necessarily the same cores on that shared hyperthreaded node. That fits the
thread- and topology-dependence above, but this run did not isolate it.

## Implications

- **For this PR:** the nested staging path is correct. No code change follows
  from this run. `compare_arms.py` is only a valid equality test when Stage 1
  and the reference ran on the same kind of node with the same core allocation.
- **For analysis (applies to runs with or without staging):** any pipeline
  that uses `DenoiseBlockMatch` gives node-dependent results on a mixed
  cluster, including the current `--force-local` F1gfd5 deployment spread
  across GPU nodes. The effect here was a few hundred boundary pixels per
  plate. Pinning Stage 1 to a single CPU type (`--slurm slurm_constraint=...`)
  and a fixed CPU count would reduce the effect. It would not eliminate it on
  a shared node, per the `gpu12` observation.
- `compare_arms.py` counted an image twice when both its objmap and its table
  differed ("12 with differences" for 6 images); fixed.
