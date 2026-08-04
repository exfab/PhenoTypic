# Spec 1 — Staged Batched GPU Detection Architecture

- **Status:** Draft — plan-reviewer pass applied 2026-06-16 (B1/B2/S1/S4/S7/C2 folded
  in); pending user review
- **Date:** 2026-06-16
- **Feature folder:** `docs/superpowers/specs/gpu-detection-optimization/`
- **Companion:** Spec 2 — *GPU Detector Models* (
  `2026-06-16-gpu-detector-models-design.md`)

---

## 1. Problem & Goal

When a CLI pipeline contains a `GpuDetector`, detection is currently executed as a
**batch of short-lived per-image jobs**:

- **Local:** the presence of a `GpuDetector` forces `n_jobs=1`
  (`LocalParallelStrategy` in `_cli_execution_strategies.py`), so images are
  processed one at a time, each call re-running the full read → enhance → GPU
  detect → measure → save chain.
- **SLURM:** `AutonomousSLURMStrategy` auto-requests `--gpus-per-node=1` but still
  submits an **array where each task is a separate process that re-initialises
  CUDA, reloads the model onto the GPU, runs one image, and exits.** Model load +
  cuDNN/CUDA init is paid *per image*; the GPU sits idle during disk I/O and the
  CPU-side enhance/measure work.

This is the documented inefficiency. Documentation research (PyTorch tuning guide;
SAM2/micro-sam source) confirms the correct pattern is **load the model once and
stream many pre-processed images through a `torch.utils.data.DataLoader`**, keeping
the GPU saturated. The dominant, reliable win is **amortising model load + CUDA/cuDNN
init and overlapping I/O with compute**; true multi-image batching is a secondary
win that matters for some models (Spec 2) and not others.

**Goal:** Split a `GpuDetector` pipeline at the detector boundary into three stages —
**CPU preprocess → GPU detect (resident model) → CPU measure** — chained
automatically from a single CLI command, with full resume, reusing the existing
per-image HDF + measure-only machinery.

**Non-goals (→ Spec 2):** SAM3, DINOv2/DINOv3+SAM2, INSID3, FSSDINO detectors, the
`foundation`/`gpu` dependency extras, and the gated-weights download path. Spec 1
only **refactors the existing `Sam2` / `MicroSamDetector`** onto the new
interface and proves the engine with them.

---

## 2. Locked Decisions (from brainstorming)

| #   | Decision                | Choice                                                                                                                                                                                                                                            |
|-----|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| D1  | Orchestration           | **Single auto-staged command** — one `python -m phenotypic` run splits internally into 3 chained stages.                                                                                                                                          |
| D2  | Trigger                 | **Default** for any CLI run with a `GpuDetector`, covering **both** `--mode full` **and** `--mode process --layer objmap`. No opt-in flag; notebook `_operate()` stays single-image.                                                              |
| D3  | Staging artifact        | **Full per-image HDF now**, with a documented hook for an optional compact pre-resize cache later ("HDF now, compact later").                                                                                                                     |
| D4  | `GpuDetector` interface | **Full batched API now** — `input_layer`, `supports_batching`, `output_kind`, `preprocess`, `collate`, `infer_batch`; default `infer_batch` loops single-image inference.                                                                         |
| D5  | Stage-2 topology        | **Small array of resident-model shard-workers** (`G` GPUs × `W` workers/GPU); two fill knobs — `gpu_batch_size` (in-worker batch) and `workers_per_gpu` (GPU packing).                                                                            |
| D6  | SLURM resources         | **Per-stage** — CPU partition for stages 1 & 3, GPU partition for stage 2; a 3-link **`afterany`** dependency chain (per-image failures don't block the next stage; content-defined work-lists).                                                  |
| D7  | GPU-fill defaults       | **Safe defaults (`workers_per_gpu=1`, `gpu_batch_size=1`) + explicit flags now; auto-tuner a documented future hook.**                                                                                                                            |
| D8  | Output routes           | Detectors declare `output_kind ∈ {instance, semantic}`; **instance** writes `objmap`, **semantic** writes only `objmask` (like a threshold detector).                                                                                             |
| D9  | Input layer             | Detectors declare `input_layer ∈ {rgb, gray, detect_mat}` (pydantic field); single-channel layers are stacked to `(H,W,3)` by the base `preprocess`.                                                                                              |
| D10 | Stage tracking          | Content-defined artifacts = truth/resume; the existing event log with **stage-tagged** statuses = live progress; existing dashboard reused.                                                                                                       |
| D11 | Staged HDF              | The staged `.h5` **is** the normal `results/<ds>/hdf/` output (GPU objmap merged in at Stage 3 from a sidecar); no separate staging area, no clean flag.                                                                                          |
| D12 | objmap-clear prep       | **Optional under D13** — the sidecar is the GPU objmap's source of truth and Stage 3 overwrites the HDF objmap from it, so a stale pre-stage-detector objmap can neither leak nor false-trigger resume. Kept only as tidiness.                    |
| D13 | objmap write-back       | **Sidecar** — Stage 2 writes the GPU result to a per-image `results/<ds>/objmap/<stem>.npy` (HDF opened read-only); Stage 3 merges it into the final HDF on an atomic re-save. The GPU stage never mutates the HDF → crash-safe, no HDF5 locking. |

---

## 3. Pipeline Split — a CLI concern, **not** an `ImagePipeline` change

**The split is orchestration owned by the CLI; `ImagePipeline` is unchanged.**
`ImagePipeline` stays a plain ordered container with `apply()` / `measure()` and gains
**no** stage/split awareness (no `apply_until`/`apply_from`, no split methods). A **new
CLI-side splitter** (`_cli/_cli_pipeline_split.py`, used by the new staging execution
strategy alongside `LocalParallelStrategy` / `AutonomousSLURMStrategy`) reads the
**public ordered `pipeline.get_ops()`** dict and partitions it at the **first
`GpuDetector`**:

- **pre-ops** — `ImageCorrector`, `ImageEnhancer`, `GridFinder`, etc. → **Stage 1**.
- the **`GpuDetector`** → **Stage 2**.
- **post-ops** — `ObjectRefiner` (incl. the user's watershed/separation refiner) →
  **Stage 3**, followed by `measure()` / post / aggregate.

**How each stage runs its slice.** The CLI builds **throwaway sub-`ImagePipeline`s**
from
the slices and reuses the existing primitives — there is no new execution machinery on
the pipeline:

- Stage 1: `ImagePipeline(ops=pre_ops).apply(image)` → save HDF.
- Stage 2: the batched engine drives the `GpuDetector` directly (§4–§7).
- Stage 3: `ImagePipeline(ops=post_ops, meas=…, post=…, filters=…, model=…, qc=…,
  nrows=…, ncols=…).apply(...)` then `measure(...)`.

The Stage-3 sub-pipeline **carries the original pipeline's `meas`/`post`/`filters`/
`model`/`qc` and grid presets**, so measurement (including any auto-injected
`AutoGridFinder` via `_build_measurement_run_order`) behaves identically to a
single-pass
run.

**Layer routing.** The detector declares `input_layer ∈ {rgb, gray, detect_mat}` (a
pydantic field; §4); Stage 1 guarantees that layer is valid before staging. When
`input_layer` is a single-channel 2D layer (`gray`/`detect_mat`), the base `preprocess`
**stacks it into a 3-channel `(H, W, 3)` block** so 3-channel models (SAM-family)
consume
it unchanged (§4). The CLI splitter uses `_layers_modified_by()` to validate that **no
post-detector op feeds back into the detector's input layer** (else fail fast with a
clear
message). (Caveat per review S3: `_layers_modified_by` does not descend into nested
sub-pipeline ops.)

**Multiple GPU detectors.** Spec 1 supports the common **single-`GpuDetector`** case
and **explicitly rejects `> 1` `GpuDetector`** in one pipeline with a clear
"not yet supported" error. (Revisit only if a real need appears.)

---

## 4. `GpuDetector` Batched/Streaming Interface (D4)

The interface is built **in full now** even though Spec 1's models do not truly batch,
so Spec 2 models only implement overrides — **no engine changes later**.

```text
GpuDetector(ObjectDetector, ABC):
  # capability / routing markers (pydantic FIELDS — see note below)
  input_layer:       Literal["rgb","gray","detect_mat"] = "rgb"  # which layer the model consumes
  supports_batching: bool = False                               # true (N,C,H,W) forward available?
  output_kind:       Literal["instance","semantic"] = "instance"

  # batched inference API — the engine drives these
  def preprocess(self, array) -> sample          # CPU; runs in DataLoader workers.
                                                 # DEFAULT: 2D layer (gray/detect_mat) -> stack (H,W,3),
                                                 # then model-specific normalization.
  def collate(self, samples) -> batch            # default: stack / list passthrough
  def infer_batch(self, batch) -> list[result]   # GPU; result is objmap (instance) or mask (semantic)

  # notebook path — unchanged behavior
  def _operate(self, image) -> image             # builds a 1-elem batch, writes objmask/objmap into the Image
```

- **Default `infer_batch`** loops the existing single-image inference, so
  `supports_batching=False` models (SAM2, micro-sam) are correct with zero new GPU
  code. Spec 2's SAM3 / DINO-based models set `supports_batching=True` and override
  `infer_batch` with a real `(N,C,H,W)` forward.
- `infer_batch` returns **lightweight arrays** (objmaps or masks), **never `Image`
  objects** — so DataLoader worker IPC stays cheap.
- **Channel adaptation (default `preprocess`, decision D9):** when `input_layer` is
  `gray` or `detect_mat` (a single-channel 2D array), the base `preprocess` **stacks it
  to a `(H, W, 3)` block** (replicate the channel ×3) so 3-channel models consume it
  unchanged; `rgb` passes through. Model-specific normalization (e.g. uint8 coercion)
  runs after. `input_layer` is a pydantic field reusing a
  `Literal["rgb","gray","detect_mat"]` alias (per the project's `tools_/typing_.py`
  convention). **Every detector scoped so far (SAM2, micro-sam, SAM3, DINO+SAM2, INSID3,
  FSSDINO) is 3-channel** — SAM image encoders and DINO ViT backbones are inherently
  RGB —
  so `input_layer="rgb"` is the right default for all of them; the stacking adapter lets
  `gray`/`detect_mat` *work* but replicates one channel into a color-pretrained encoder
  (discarding its color prior), so it is a deliberate/edge choice here (it matters more
  for
  a future grayscale-native model).
- **`output_kind` routes the write-back** (see §5, §8): `instance` → `objmap`;
  `semantic` → `objmask` only.
- `_operate` is preserved and re-expressed in terms of `preprocess` + `infer_batch`
  so the single-image (notebook) and batched (CLI) paths share one core per detector.

*Pydantic mechanics (decided per review S4):* `input_layer`, `output_kind`, and
`supports_batching` are **pydantic fields** with class-set defaults — **not `ClassVar`**
— so they serialize, round-trip through `to_json`/`from_json`, and stay per-instance
overrideable. (`ClassVar` would not serialize, breaking the pipeline JSON contract.)

---

## 5. Staging Artifact & Lifecycle (D3)

Per-image `.h5` is the **canonical cross-stage carrier**:

1. **Stage 1** writes the `.h5` *pre-detection* via the existing
   `OutputManager.save_image_hdf` path — **unchanged HDF contract**. Whatever the
   pre-ops
   leave in `objmap` (zeros if no detector ran, or a stale map if a **non-GPU
   `ObjectDetector`** was placed before the `GpuDetector`) is **irrelevant under the
   sidecar design**: Stage 3 overwrites the HDF's objmap from the sidecar, and the
   Stage-2
   resume signal is *sidecar presence* (§9), not the HDF's objmap. The **objmap-clear
   prep
   step (D12) is therefore optional tidiness, not load-bearing** here. HDF already
   embeds
   `HdfAttr.PHENOTYPIC_CLASS` + grid state, so `GridImage` rehydrates correctly in Stage
   3
   (the existing `--mode measure` path relies on this).
2. **Stage 2 (sidecar write-back, decision D13)** opens the `.h5` **read-only**, does a
   **partial read** of just `input_layer` (h5py reads one dataset; it never decodes
   `detect_mat`/`rgb` it doesn't need), runs `infer_batch`, and writes the raw GPU
   result
   to a small **per-image sidecar** — `results/<ds>/objmap/<stem>.npy` — via `np.save`
   (to `*.tmp` then `os.rename` so it appears atomically). **It never opens the HDF for
   writing**, so there is no HDF5 write-lock on the shared filesystem and no way to
   corrupt
   the staged input. The sidecar holds exactly what `infer_batch` returns (a labeled
   uint16
   map for `output_kind="instance"`, or a binary mask for `output_kind="semantic"`); the
   route is known from the detector's `output_kind` (so Stage 3 applies it via the right
   accessor — §8), so the sidecar itself is just a plain array.
3. **Stage 3** reloads the full `.h5` (existing `load_hdf5`), loads the sidecar
   (`np.load`), and **applies it via the Image accessor** — `image.objmap[:] = arr`
   (instance) or `image.objmask[:] = arr` (semantic; auto-labels into `objmap`, see
   §8) —
   then runs the existing measure path (`process_single_hdf_measure_core`):
   post-detector
   refiners (incl. watershed), `measure()`, aggregation, deliverables. Stage 3 also
   **re-saves the final `.h5`** (existing `save_image_hdf`, written to a temp file then
   `os.rename`d atomically over the original) so the output HDF carries the GPU objmap —
   matching the forward path, which already saves the HDF after detection. Stage 3 then
   **deletes the sidecar** (cleanup is **non-optional** — the
   sidecar is a transient cross-stage handoff, not an output).

A **documented hook** is left for an optional compact pre-resize cache if profiling
later shows the GPU starving on HDF reads — added as a pure optimisation **without
changing the contract**. The staged `.h5` **is** the run's normal
`results/<ds>/hdf/<stem>.h5` output (created in Stage 1; GPU objmap merged in at Stage 3
from the sidecar), so after Stage 3 the output folder is **identical to a single-pass
run** — there is **no** separate staging area and **no** `--clean-stage-hdf` flag. The
only extra artifact is the transient `results/<ds>/objmap/<stem>.npy` sidecar, which
**Stage 3 deletes after the merge** (mandatory cleanup).

**Storage fact (verified in review):** the HDF `/layers/` group stores only `rgb`,
`gray`, `detect_mat`, `objmap`; **`objmask` is a derived binary view of the same
`sparse_object_map` backend as `objmap`**, not an independent dataset. A semantic
detector therefore cannot "write only `objmask`" at the storage layer —
`image.objmask[:] = mask` auto-labels into `objmap`, which is what the HDF persists. The
round-trip works: at the **Stage-3 merge**, `image.objmask[:] = mask` (semantic)
auto-labels into `objmap`, and `save_image_hdf` persists it under the **v2 grouped**
`layers/objmap` dataset — the existing save path, no new HDF layout. The sidecar (a
plain
`.npy`) carries the GPU result between stages and is **outside** the HDF contract
entirely.

---

## 6. Three-Stage Data Flow

```text
        ┌── Stage 1: CPU parallel ──┐   ┌── Stage 2: GPU resident ───┐   ┌── Stage 3: CPU parallel ──────┐
 raw →  │ read → pre-ops → save .h5 │ → │ read .h5 (input, RO) →     │ → │ reload .h5 + objmap sidecar → │ → deliverables
        │                           │   │ infer_batch → write objmap │   │ apply → post-ops → measure →  │
        │                           │   │ .npy sidecar               │   │ aggregate → atomic re-save .h5│
        └───────────────────────────┘   └────────────────────────────┘   └───────────────────────────────┘

 --mode process --layer objmap = Stage 1 + Stage 2, then export objmap PNGs (stop).
 --mode full                    = Stage 1 + Stage 2 + Stage 3.
```

`--mode process --layer objmap` auto-routes through Stages 1–2 (the GPU work) and
then the existing per-layer export; no measurement/deliverables. (Note: the current
CLI flag is `--mode process --layer {...}`; the old `--process-only` is removed.)

---

## 7. Execution Topology (D5, D6, D7)

### Local

Stages run sequentially in-process. Stage 1/3 use joblib `n_jobs`. Stage 2 is a
resident-model loop fed by a torch `DataLoader`:
`num_workers ≈ cpus`, `pin_memory=True`, `prefetch_factor`, `persistent_workers=True`,
`shuffle=False`, `drop_last=False`, **a per-sample id carried through for result
reassembly**, wrapped in `torch.inference_mode()` with `model.eval()`. GPU→CPU result
copies are synchronised (non-blocking H2D is safe; D2H is not without a sync).

### SLURM

A **3-link `afterany` dependency chain** (not `afterok` — a few per-image failures must
not block the next stage; each stage's work-list is content-defined, §9), each link with
its **own partition/resources**: Stage 1/3 on a **CPU partition**, Stage 2 on the **GPU
partition**.

**Scope note (review S1) — this is a non-trivial refactor, not a parameter addition.**
The current `submit_slurm_script_chain` / `generate_all_array_job_scripts` apply a
**single flat `slurm_args`** to every generated script and chain **chunks within one
stage** (a drip-feed serial chunk chain in `generate_dispatcher_chain`); there is no
concept of "stage" or per-stage resources. The plan must introduce a **stage
abstraction**: three distinct script sets (Stage 1 CPU array, Stage 2 GPU shard-workers,
Stage 3 CPU array), each with its own SBATCH resource profile, wired
`afterany:stage_{n-1}` (per-image failures must not block the next stage; work-lists are
content-defined) — and Stage 2's worker model (a few resident workers streaming shards)
is
**structurally different** from today's one-task-per-image array. Budget for this as
real
surgery.

### Stage-2 GPU pool

A *small* array of **resident-model shard-workers** — 1 worker locally; on SLURM a
small GPU array where **each task loads the model once and streams a shard** of the
staged HDFs. Two fill knobs layered per-GPU:

- **`gpu_batch_size`** — real `(N,C,H,W)` forward, one model copy, N images/pass.
  Fills the GPU for `supports_batching=True` models (Spec 2). Default **1**.
- **`workers_per_gpu`** — pack K resident workers on one physical GPU (K model
  copies, VRAM-bounded). Where the cluster enables **NVIDIA MPS** their kernels run
  concurrently; without MPS, plain time-slicing still overlaps one worker's CPU-side
  HDF read / post-processing with another's kernels. Fills the GPU for
  non-batchable models (SAM2/micro-sam). Default **1**.

Defaults `1×1` are correct for every model and already capture the load-amortisation +
I/O-overlap win. An **auto-tuner** (probe VRAM + model footprint) is a documented
future hook, not built now.

**Precision:** **fp32 by default** (project is accuracy-first). AMP/bf16 is opt-in and
documented as "validate mask/label parity vs fp32 first" (numerical drift is a
measurement-correctness risk). `torch.compile` / `cudnn.benchmark` are off by default
(they re-autotune/recompile on variable image sizes).

---

## 8. Output Routes — Instance vs. Semantic (D8)

**Verified storage reality (review):** `objmask` is a *derived binary view* of the same
`sparse_object_map` backend as `objmap` — not a separate dataset.
`image.objmask[:] = mask`
routes through the accessor's `__setitem__`, which connected-component-labels the mask
(`skimage.measure.label`) and stores the result in that shared backend. `OtsuDetector`
relies on exactly this: it sets `image.objmask = mask` and the auto-label produces the
`objmap`; a downstream separation/watershed refiner (`refine/_separate_objects.py`) then
splits touching colonies.

- **Instance route** (`output_kind="instance"`, default — SAM2/micro-sam today):
  `infer_batch` returns labeled uint16 maps; the engine sets `image.objmap[:] = labeled`
  so the **detector controls labeling** (e.g. largest-first painting to preserve
  small-colony identity at overlaps).
- **Semantic route** (`output_kind="semantic"` — Spec 2 INSID3/FSSDINO): `infer_batch`
  returns **binary masks**; the engine sets `image.objmask[:] = binary_mask`, which
  **auto-labels via connected components** into the shared `objmap` backend — *exactly
  the `OtsuDetector` path*. **No watershed inside the detector**; touching colonies that
  merge into one component are split by the user's downstream `SeparateObjects`
  /watershed
  refiner, which (being a post-detector op) runs in **Stage 3**.

Both routes therefore persist a labeled `objmap` (the semantic route's labels are naive
connected components pending the downstream watershed). This is the *identical image
state a threshold detector produces today*, so HDF staging, resume, and measure handle
it
with **zero new machinery** — the only route-dependent difference is **which accessor
the
Stage-3 merge calls when applying the sidecar** (`objmap[:]` for instance vs
`objmask[:]`
for semantic; §5). The `output_kind` marker is added in Spec 1 (per D4), defaulting to
`instance` so existing behavior is unchanged.

---

## 9. Resume, Idempotency & Stage Tracking

Stage boundaries are **content-defined**:

- Stage 1 done for an id ⇔ its `.h5` exists with `input_layer` valid and readable.
- Stage 2 done for an id ⇔ its **objmap sidecar** (`results/<ds>/objmap/<stem>.npy`)
  exists. Cheap to check (a file stat — no need to open the HDF) and **immune to any
  pre-stage detector's objmap** in the `.h5`, which is why the D12 objmap-clear is
  unnecessary under the sidecar design.
- Stage 3 done for an id ⇔ its atomic per-image Stage-3 completion marker exists.
  The marker is published after the parquet, HDF, and plot; Stage 3 then deletes
  the now-merged sidecar.

The **sidecar-presence check is the primary Stage-2 completion predicate** (review C2):
`np.save` to `*.tmp` + `os.rename` makes the sidecar appear atomically, so a worker that
dies mid-batch leaves each id either done (sidecar present → skip) or not (absent →
reprocess), with no double-counting. A dependent, epoch-fenced controller waits for the
array master to terminate and then classifies every image as done, terminal, or
retryable.
It submits another Stage-2 round only while retryable images remain; workers never
self-requeue. Legacy parquet-only runs are migrated to Stage-3 markers when no
sidecar remains. Current
epoch terminal failures and missing Stage-1 HDFs count as terminal.

The controller stores an atomic state record and append-only job ledger. Every scheduler
transition has a deterministic token and SLURM comment, allowing a successor controller
to
discover a submitted job if its predecessor died before recording the job ID. One
unchanged
retryable-set round is retried; a second unchanged round terminalizes the remainder and
advances to Stage 3. Every worker verifies its orchestration epoch immediately before
publishing an artifact, so restart and cancellation fence stale jobs.

**Stage tracking — the progress mechanism (decision D10).** Two layers, both reusing
existing infrastructure:

1. **Truth = content-defined artifacts** (above). The source of truth for resume *and*
   for each stage worker's work-list. Robust to lost/partial logs: any worker re-derives
   an image's furthest stage by checking valid HDF → **objmap sidecar presence** →
   atomic Stage-3 completion marker. Legacy parquet-only completions are migrated
   before the marker contract is enabled. This is what makes `afterany` chaining safe —
   a stage
   processes exactly the artifacts the prior stage actually produced.
2. **Live progress = the existing append-only event log** (`event_log_path`,
   `append_event` / `append_completion_event`), extended with **stage-tagged statuses**:
   each stage worker emits `stage{N}_started` / `stage{N}_completed` / `stage{N}_failed`
   keyed by `(dataset, image)`. `aggregate_state_from_events` is extended to compute,
   per
   image, the **furthest completed stage** (and the failure stage, if any) plus
   per-stage
   counts; `job_metadata.json` records that the run is a 3-stage GPU run and its stage
   list so the dashboard knows to render three stages.

An image's **current stage = max stage with a `completed` event** (live view) or the
highest stage whose artifact predicate holds (authoritative view). The existing
dashboard
(OQ5 = reuse) reads this aggregate and renders per-stage progress, e.g.
`Stage 1 ✓ 100/100 · Stage 2 63/100 · Stage 3 40/100`, plus each image's furthest stage.
Events drive the low-latency UI; artifacts drive correctness/resume — the same split the
current single-pass run already uses between its event log and `processing_state.json` /
parquets.

---

## 10. CLI / Config Surface

No new *mode* flag (batched is simply *how* a `GpuDetector` runs). New options:

- **`--gpu-slurm key=value`** (repeatable; mirrors `--slurm`) — the **GPU-stage (Stage
  2)**
  SBATCH profile. **Inherit/delta:** effective Stage-2 args =
  `{**slurm_args, **gpu_slurm_args}` (+ auto `slurm_gpus_per_node=1` if absent). Shared
  keys (account, qos, time) set once in `--slurm` carry over; stage-specific keys — a
  **separate GPU partition and its own account/gres/time** — are overridden in
  `--gpu-slurm`. Stages 1 & 3 use `--slurm` (the CPU profile).
- **`--gpu-shards N`** (default **1**) — number of parallel **Stage-2 GPU tasks** (each
  shard = one whole-GPU array task; ≈ the GPUs to spread images across). **SLURM-only —
  ignored locally**. Set to your
  concurrent-GPU count for real runs.
- **`--gpu-workers-per-gpu W`** (default 1) — reserved for future replica packing.
  The current staged implementation launches one resident model per GPU shard.
- **`--gpu-batch-size N|auto`** (default 1) — images per forward pass within one replica
  (**effective only for batchable detectors** — SAM2/micro-sam are one-image-at-a-time).
  `auto` runs a one-time **VRAM-probe warmup** (try 1,2,4,… on a representative tile,
  pick
  the largest fitting under a safety margin of free VRAM ÷ `workers_per_gpu`,
  OOM-backoff).
  The probe is **implemented in Spec 2** (where batchable detectors arrive); in Spec 1
  it
  is parsed but inert.

The active fill knob in this implementation is **`--gpu-shards`** (across GPUs).
Per-GPU replicas and detector batching remain reserved for later work.

`--mode process --layer objmap` requires no new flags; it routes through Stages 1–2
automatically when a `GpuDetector` is present.

---

## 11. Refactor of Existing Detectors

`Sam2` and `MicroSamDetector`:

- Extract their single-image core into the shared `infer_batch` default path.
- Set `input_layer="rgb"`, `supports_batching=False`, `output_kind="instance"`.
- `_operate` stays working (delegates to a 1-element batch). **Pure refactor — no
  behavior change for notebook users**; doctest/serialization round-trips preserved.

---

## 12. Licensing Scaffolding (Spec 1 share)

Spec 1 establishes the **attribution scaffolding** the feature needs, populated for the
(ungated, permissive) components it already touches:

- Add root `NOTICE` + `licenses/` directory; seed with **SAM2 (Apache-2.0)** and
  **micro-sam** entries. `NOTICE` states plainly that PhenoTypic **does not redistribute
  model weights** — they are downloaded by the user under each model's license.
- Add the **license-acceptance gate hook** to the checkpoint-manager surface (a no-op
  for the ungated SAM2/micro-sam, but the hook Spec 2's gated SAM3/DINOv3 managers
  call). Include `NOTICE`/`licenses/` in the sdist/wheel via build config.

The gated HF download path + `foundation`/`gpu` extras + per-model managers land in
**Spec 2** (that's where `huggingface_hub` and the gated models arrive).

---

## 13. Backward Compatibility

- Notebook `op.apply(image)` / `pipeline.apply(image)` — unchanged.
- `--mode process --layer {rgb|gray|detect_mat}` (non-objmap) — unchanged.
- The **only** behavior change: a CLI run containing a `GpuDetector` now stages
  internally instead of running per-image. That *is* the feature.

---

## 14. Testing Strategy

- **Unit:** pipeline-split partitioning; the `>1 GpuDetector` and input-layer-feedback
  guards; `infer_batch` default-loop equivalence to `_operate`; **sidecar write →
  Stage-3
  merge → atomic HDF re-save** round-trip (both `objmap` instance and `objmask` semantic
  routes, asserting the final HDF objmap matches); **mandatory sidecar cleanup after
  Stage
  3**; resume skip logic (sidecar-presence predicate).
- **Integration:** a tiny **fake CPU `GpuDetector`** with both `supports_batching ∈
  {True, False}` and `output_kind ∈ {instance, semantic}` variants, driven end-to-end
  through all 3 stages locally; plus the `--mode process --layer objmap` path.
- **SLURM:** script-generation tests for the 3-link per-stage chain (generation/parse,
  not live submission).

---

## 15. Risks / Open Questions

- **Semantic write-back via the accessor** (review B1/B2): `objmask` is not an
  independent HDF dataset — it shares the `objmap` backend, and `objmask[:] = mask`
  auto-labels into it. The **Stage-3 merge** applies the sidecar via `image.objmask[:]`
  (semantic) / `image.objmap[:]` (instance); the subsequent `save_image_hdf` persists it
  under the **v2 grouped** `layers/objmap` dataset (the existing save path). No in-place
  HDF dataset patching and no v1 flat-layout writes.
- **SLURM 3-stage chaining is a real refactor** of the flat single-partition drip-feed
  machinery (see §7) — not a parameter addition.
- **Pipeline splitter** finds the *first-`GpuDetector` position* by iterating `_ops`
  (review S2 — `pipeline_requires_gpu` only returns a boolean). `_layers_modified_by`
  does **not** descend into nested sub-pipeline ops (review S3), so the input-layer
  feedback check is blind to ops wrapped in a nested `ImagePipelineCore` — document the
  limitation.
- **objmap write-back = sidecar (chosen, D13).** The GPU stage never opens the HDF for
  writing — it writes a per-image `.npy` sidecar — so there is **no HDF5 write-locking
  on
  the shared filesystem and no way to corrupt the staged input**. Residual items for the
  plan: (a) Stage 2's `np.save` uses `*.tmp` + `os.rename` for atomic sidecar
  appearance;
  (b) Stage 3's HDF re-save is **atomic** (`save_image_hdf` to `*.tmp` then `os.rename`)
  so a crash there can't corrupt the output; (c) the big HDF is written twice (Stage 1 +
  the Stage-3 re-save) — accepted I/O cost; (d) **Stage 3 deletes the sidecar after the
  merge — cleanup is non-optional.**
- **All scoped detectors are 3-channel** (Q1): `input_layer="rgb"` is the default; the
  `gray`/`detect_mat` → `(H,W,3)` stacking adapter (D9) works but discards the color
  prior of these RGB-pretrained backbones.
- **MPS availability on UCR HPCC** GPU nodes is unconfirmed; packing degrades
  gracefully to time-slicing without it.
- **DataLoader IPC of arrays** — ensure per-sample id + array (not `Image`) crosses
  worker boundaries; confirm h5py partial-read is process-safe under `num_workers`.

---

## 16. Spec 2 Handoff

Spec 1 delivers: the **CLI-side pipeline splitter** (no `ImagePipeline` change), the *
*full** `GpuDetector` interface (`input_layer`,
`supports_batching`, `output_kind`, `preprocess`/`collate`/`infer_batch`), the staged
HDF engine, per-stage SLURM chaining, shard-workers + fill knobs, resume, and the
licensing scaffolding. Spec 2 implements **SAM3, DINOv2/v3+SAM2, INSID3, FSSDINO** as
overrides on this interface (true batching, semantic route, gated weights), with **no
engine changes**.
