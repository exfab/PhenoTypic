# CLI Execution (`_cli/`)

The batch-processing engine behind `python -m phenotypic`. `../phenotypicCLI.py`
(at the package root, not in `_cli/`) parses options into an `ExecutionConfig`
(`_cli_types.py`, a mutable `@dataclass`), then
`create_execution_strategy(config, output_manager)`
(`_cli_execution_strategies.py`) dispatches to one strategy.

## Execution strategies (the dispatch)

`create_execution_strategy` picks by `(is_slurm_mode, measure_only,
process_only_layer, pipeline_requires_gpu)`:

| Strategy | When | File |
|---|---|---|
| `LocalParallelStrategy` | local CPU run (joblib) | `_cli_execution_strategies.py` |
| `AutonomousSLURMStrategy` | SLURM CPU run (array + drip-feed chunk chain) | `_cli_execution_strategies.py` |
| `StagedGpuStrategy` | **local** forward GPU run, or `--mode process --layer objmap` | `_cli_staged_strategy.py` |
| `StagedSlurmStrategy` | **SLURM** forward GPU run (`process_only_layer is None`) | `_cli_staged_slurm.py` |

A "GPU run" = the pipeline contains a `GpuDetector` (`pipeline_requires_gpu`),
**at any depth in the operation tree**. The staged strategies are
forward-oriented; measure-only and non-objmap process-layer exports stay on the
Local/Autonomous strategies.

**`--mode process --layer objmap` runs the post-detector op chain.**
`_export_objmap_layer` (`_cli_staged_strategy.py`) runs after Stages 1–2 and
substitutes a `ReplayDetector` at `plan.gpu_path` inside `post_pipeline`, then
applies that residual pipeline — so the export is *the objmap the pipeline
produces*, not the detector's raw output. For a **nested** detector the raw
array is one branch of a composite and not an objmap at all, so the previous
behaviour exported the wrong thing rather than merely an unrefined one. Three
details that are load-bearing rather than incidental:

- It reads `load_stage2_raw`, **never the store** — Stage 2 writes nothing into
  the store, so the store's objmap here is still Stage 1's zeros and a store
  read would silently export an all-zeros PNG for every image.
- The apply is wrapped in `continuing_provenance_application` and **no**
  `provenance_success_sink`. Stage 1 left the application `"staged"`, which is
  not terminal, so a plain apply at owner-depth 0 raises; and a success sink
  would write to the store *after* the image success marker, invalidating the
  descriptor the marker just recorded. Do not "restore" the sink, and do not
  reach for `set_provenance_status(image, "in_progress")` — that status is also
  non-terminal and raises the very error it looks like it prevents.
- The change of meaning is fenced in the work id:
  `PROCESS_LAYER_SEMANTICS_REVISION` (`_cli_failure_tracker.py`, currently `3`;
  2 → 3 is the per-image figures, see **Per-image figures** below)
  rides beside `process_format` in the process-only branch of
  `processing_configuration_digest_from_values`, **not** in the base payload —
  a base placement would cold-start every in-flight `full` and `measure`
  continuation for no correctness gain. It is one integer for all layers, so a
  bump also invalidates `--layer gray` continuations; invalidating too much is
  safe, so that is a cost, not a bug.

## Staged GPU engine

When a CLI pipeline contains a `GpuDetector`, the **CLI** (not `ImagePipeline`)
splits it at the detector boundary (`split_pipeline_at_gpu` in
`_cli_pipeline_split.py` →
`StagePlan{pre_pipeline, gpu_path, gpu_detector, stage2_prefix, post_pipeline}`)
and runs three content-defined stages. The per-image stage cores live in
`_cli_staged_workers.py` and are shared by both staged strategies:

1. **Stage 1** `stage1_preprocess_core` — apply pre-detector ops → publish the
   staged OME-Zarr store `results/<ds>/zarr/<stem>.ome.zarr/` (objmap included,
   as zeros, because `valid_staged_store` requires it).
2. **Stage 2** `stage2_detect_core` — load the input layer (store **read-only**),
   apply `plan.stage2_prefix` to a provenance-detached copy when the detector is
   nested, run the resident detector, and drop its **Stage-2 signal**: the
   retained **raw** detector output at
   `.phenotypic/progress/stage2_raw/<ds>/<slot>/<stem>.npy` — the name is
   unchanged but the file is a **compressed** archive (`np.savez_compressed`),
   because the dataset-wide barrier keeps every image's raw output live at
   once: ~32 MB each uncompressed is ~1 TiB over a 33,923-image run.
   `load_stage2_raw` reads a pre-compression bare `.npy` too, so a run
   interrupted across that change replays instead of re-inferring — then a
   consumable
   **token** at
   `.phenotypic/progress/stage2_done/<ds>/<slot>/<stem>.json`
   (`_cli_stage2_token.py`,
   both atomic temp+`os.replace`; raw first, so a crash between them leaves no
   "done" signal). **Stage 2 does not write into the store** — only the final
   store needs third-party interop, and an in-store write would be visible to
   the uncached crop route as raw pre-`drop_frame_background` labels.
3. **Stage 3** `stage3_merge_measure_core` — substitute a `ReplayDetector`
   (`_cli_replay_detector.py`) at `plan.gpu_path` inside `post_pipeline`, so the
   **raw** array is written through the real detector's `_write_object_output`
   and the enclosing operation runs normally. Never the store's own objmap:
   Stage 3 re-promotes over it, so a retry would refine already-refined labels.
   Then apply post-ops + `measure(apply_post=False)`, re-promote the store, and
   consume the signal — **token first, then the raw array** (mandatory).

### Nested detectors: found tree-wide, cut at the top-level ancestor

**`GpuDetector`s are found tree-wide**, not at the top level.
`find_gpu_detectors` (`_cli_validation.py`) walks the whole operation tree via
`sdk_._operation_tree.walk_operations` and returns `(path, detector)` pairs,
where `path` is a tuple of non-empty strings (`"ops[0]"` for a list entry). That
same path is the detector's recorded `pipeline_step_path`, the argument to
`substitute_at_path`, and the input to `detector_slot` — **one addressing
scheme, not three.**

That identity is only true because container operations now push a per-branch
step: `CompositeDetector`, `CompositeEnhance`, `FilamentousFungiDetector` and
`TwoKFilamentousDetector` drive their children through
`_core/_provenance.apply_child(child, image, segment=...)` instead of calling
`.apply()` directly, so each branch records its own segment rather than
inheriting the container's path. **Recorded `pipeline_step_path`s are therefore
deeper for every pipeline using a container op**, not only for staged ones.
`apply_child` is deliberately *not* used for a nested operation run by a
`MeasureFeatures` — those are private probes that stay out of the plate's
provenance (`measure/CLAUDE.md`).

- **The split cuts at the detector's top-level ancestor.** `pre_pipeline` holds
  the root ops before that ancestor; `post_pipeline` holds the ancestor onward
  **including the detector itself**. Stage 3 never applies `post_pipeline`
  as-is — it substitutes a `ReplayDetector` at `gpu_path` first, or the run
  re-executes live GPU inference on a CPU node.
- **`stage2_prefix` is the ops ahead of the detector *inside its own branch*.**
  `_branch_prefix` dispatches on `_child_contract`, never on `isinstance`: a
  `"sequence"` container contributes the ops preceding the step taken from it, a
  `"parallel"` one contributes nothing, and the **root** contributes nothing
  because Stage 1 already ran and stored its ops. Empty for a bare leaf and for
  a top-level detector.
- **Those prefix ops run twice and must be deterministic.** Stage 2 applies them
  to a provenance-detached copy purely to produce the detector's model input,
  then discards it; Stage 3 runs the same ops again inside the enclosing
  operation and records *those* in the journal. A non-deterministic prefix means
  Stage 2 infers from an image Stage 3 never reconstructs, and nothing fails to
  say so.
- **Unstageable placements are refused, on the production path.** All refusals
  live in `find_gpu_detectors`, which `pipeline_requires_gpu` calls — a refusal
  reachable only from `split_pipeline_at_gpu` fires after the run has already
  been routed, which is how this narrowing was silently lost once already. They
  are: a container that is not a composition primitive; a subclass of a listed
  composite that **overrides `_operate`** (the contract was verified against the
  base's `_operate`, so a replacement has been verified against nothing); a
  detector reached through **any** pipeline's `meas`/`post`/`filters`/`model`,
  root or nested, at any depth (`refuse_cpu_only_slot`); and, under
  `strict=True` (only `split_pipeline_at_gpu`), more than one detector. All
  raise `UnstageableGpuDetectorError`, a `ValueError` subclass.
- **`_CHILD_CONTRACT` is the closed table**, `{CompositeDetector: "parallel",
  CompositeEnhance: "parallel"}`, with `ImagePipeline` handled separately as
  `"sequence"` in `_child_contract`. It is populated lazily by
  `_populate_child_contract` in a **single** `dict.update` — read it through
  that function, never directly, and never populate it key-by-key: the guard is
  `if _CHILD_CONTRACT: return`, so a half-populated table makes a thread refuse
  a `CompositeEnhance` the design permits, and the GUI (threaded Werkzeug) would
  show a refusal alert and disable Run for a valid pipeline. Entries match by
  `isinstance`. There is no `_UNSUPPORTED_CONTAINERS` map. Coverage is asserted
  on the table itself (`tests/unit/cli/test_gpu_detection_tree_wide.py`) and
  each `"parallel"` entry carries a behavioural probe
  (`tests/unit/detect/test_container_child_contracts.py`); a new container needs
  no entry, because refused-by-default is the right answer for it.

#### Pipeline slots, and how a refusal reaches the user

- **Every pipeline's four slots are walked, root and nested.**
  `iter_child_operations` yields an `ImagePipelineCore`'s `ops` *and* its
  `meas`/`post`/`filters`/`model` entries (`_iter_slot_children`). Slot
  children are yielded **unconditionally**, never filtered through
  `_is_operation`: that predicate rejects `ModelFitter`, `SetAnalyzer` and
  `PostMeasurement`, so filtering would stop the root `model` slot being found
  at all. `find_gpu_detectors` then refuses any hit whose path passes through a
  slot at any depth. There is no `_CPU_ONLY_SLOTS` any more; the root-only loop
  it drove is gone, and so is its doubled `model/model/...` path.
- **Slot segments are spelled as a colon namespace:** `meas:<key>`,
  `post:<key>`, `filters:<key>`, and `model:<ClassName>` for the single model
  (`sdk_/_operation_tree.py`, `PIPELINE_SLOTS`). A bare `meas` segment would
  collide with an `ops` key a user is free to choose, and the `ops[0]` bracket
  form admits only integers. `_child` checks the model's class name, so a
  recorded `model:` path stops resolving once the model it addressed is
  replaced. **Never classify a segment by parsing it:** a user may key an op
  `"meas:X"`, so `pipeline_slot_of` decides against the live node, and a
  segment that is both an `ops` key and a resolvable slot entry raises
  `KeyError` as ambiguous rather than picking one.
- **Why a slot placement is refused.** The slots run after the op chain, which
  in the staged engine is Stage 3 on a CPU node, after Stage 2's inference has
  finished; and a measurement may hand its nested detector a derived input
  (possibly once per object) that Stage 2, which records one array per image
  per slot, cannot reproduce. **Do not reason "unrefused would be silently
  wrong" for ordinary pipelines — that was measured false.** Every shipped slot
  type (`MeasureFeatures`, `PostMeasurement`, `SetAnalyzer`, `ModelFitter`) is
  not a composition primitive, so `validate_ancestor_contracts` refuses those
  shapes anyway, just blaming the entry's class ("cannot be nested inside
  MeasureSymZones"). The slot check runs *first* so the message names the real
  reason. **It is the only guard** for a class that is both a slot type and a
  composition primitive — `class MeasuringComposite(CompositeDetector,
  MeasureFeatures)` inherits `_operate`, passes `_child_contract`, and without
  the slot refusal would be staged and run by Stage 2 on the stored image with
  no error. Pinned by `test_a_slot_entry_that_is_also_a_composite_is_refused`,
  which asserts the ancestor check admits the entry so it cannot pass on that
  check alone.
- **CLI: a usage error, before anything touches `--output`.** Right after
  `ExecutionConfig` is built, `phenotypic_cli` calls
  `uses_staged_gpu_strategy(config)` and converts `UnstageableGpuDetectorError`
  into `click.UsageError`; any *other* exception is ignored there so a corrupt
  pipeline still reaches the existing "Pipeline loading failed" line. The
  preflight sits above the manifest load, `--overwrite` clearing,
  `mint_run_identity`, the `--dry-run` exit and `create_execution_strategy`.
  Both placements matter: below the clearing, a refused pipeline pointed at an
  existing run directory **deleted that run's results** before refusing, and
  `--dry-run` (the GUI's Validate) never reached the check, so Validate passed
  what Run refused. The outer handler converts a later refusal (e.g. from the
  splitter) the same way. Pinned by `tests/unit/cli/test_cli_gpu_refusal.py`.
  Two consequences of routing through `uses_staged_gpu_strategy`:
  `--mode measure` returns before scanning and is never refused, while
  `--mode process --layer rgb|gray` **is** refused although those layers come
  from pre-detector ops — `pipeline_requires_gpu` raises before the layer is
  consulted. The second predates this change and is unresolved.
- **GUI: the message, in either mode, and no Run.**
  `_gui/run_console/_callbacks.py:_staged_gpu_capability(path)` returns
  `(uses_gpu, refusal | None)` and catches `UnstageableGpuDetectorError`
  *before* its generic `(OSError, ValueError, TypeError)` clause — the refusal
  **is** a `ValueError`, so clause order is the whole fix. The
  `rc-staged-gpu-refusal` alert sits outside the staged-GPU section, so it
  shows in Local mode too; `update_run_disabled`, the sole owner of Run's
  `disabled`, takes the alert's `is_open`; and `click_action` refuses Validate
  and Run before `registry.allocate`, covering a click that races the
  pipeline-change repaint. An unreadable pipeline returns `(False, None)` — no
  alert; the CLI's validation reports it.
- **The previous text here was wrong about the mechanism.** It said the GUI
  swallowing the refusal routed a refused run to CPU; in fact the old
  `_pipeline_uses_staged_gpu` had one consumer, which only toggled the GPU form
  section's `display`, and the run itself is a `python -m phenotypic`
  subprocess, which raised — a hidden section, then a raw traceback.

### Stage-2 signals are keyed by detector slot

Both halves of the signal live under `<ds>/<slot>/`, where `slot` is
`detector_slot(plan.gpu_path)` — each path segment with runs of
non-alphanumerics collapsed to `-`, joined by `__`, plus 8 hex of the **exact**
path so `ops[0]` and `ops-0` cannot collide
(`CompositeDetector__ops-0__dddf3676`). Every path helper in
`_cli_stage2_token.py` takes `slot` with **no default**: a default would let a
caller reach a shared path without being handed one, which is the structural
hole the keying closes. Sites holding a pipeline path rather than a `StagePlan`
use `staged_detector_slot(pipeline_path)`; where a `StagePlan` is in scope, call
`detector_slot(plan.gpu_path)` directly.

At the shipped one-detector limit the slot level changes no behaviour. It exists
so that supporting N detectors is an additive feature rather than a migration of
the on-disk layout of a signal a 33,923-image run depends on.

Two deliberate exceptions to "always pass a slot":

- `find_stage2_token(output_dir, dataset, stem)` — the one legitimate slot-free
  **read**, for `--mode migrate`, which has no pipeline and therefore no
  `StagePlan`. It searches the legacy flat path, then each slot directory, and
  never moves or unlinks anything.
- `relocate_legacy_stage2_signal(...)` — a one-time move of a pre-slot-keying
  signal into its slot directory, so a run interrupted before the change and
  resumed after it does not repay Stage 2 in GPU time. Moves **both** halves,
  raw first, never overwrites a current signal, and **refuses** when the caller
  declares `slot_count != 1` (a per-image legacy signal is one detector's output
  and cannot be attributed among several).

**Continuation is automatic and content-defined.** Run the same command again;
there is no `--resume` flag. Exact terminal failures remain skipped unless
`--retry-failures` is supplied.
A missing or invalid store selects Stage 1; a valid store without a complete
Stage-2 signal selects Stage 2; and a valid store with one selects Stage 3.
**Every prereq probe tests BOTH halves of the signal** —
`stage2_result_replayable()` is the one function all **six call sites in three
modules** call (`_cli_staged_strategy.py` ×3, `_cli_staged_slurm_worker.py` ×2,
`_cli_staged_controller.py` ×1) — a count a `grep` can confirm or refute, which
"five sites" could not, because it never said sites of what. The token
is only a flag; Stage 3's actual input is the raw `.npy`, so a
token-present/raw-missing image is routed back to Stage 2, not into a Stage 3
that would raise `FileNotFoundError` and be recorded as a terminal *scientific*
failure. Stage 3 embeds the authoritative Parquet inside the final store transaction,
publishes the image marker over both the store root and table, then consumes the
signal. The output is byte-identical to a
single-pass run, with one exception: a `PlotImage` bound to a pre-GPU `ops`
entry goes to `pre_pipeline`, so Stage 1 draws it on the image after the
pre-GPU ops only, where a single-pass run draws it after the whole op chain.
Stage 3 keeps that figure (see **Per-image figures**). Today the only shipped
`PlotImage` that is also an operation is `CalibrateColorRpcc`, whose overlay is
the same either way because it draws from its own `apply()`.

**Progress events.** Stages emit stage-tagged events via the `stage` field on
the event log (`_cli_update_state.py`: `append_event(..., stage="stage1|2|3")`).
`status` stays the closed `{started, completed, failed}` set;
`aggregate_state_from_events` counts overall completion at Stage 3 only, and
`aggregate_stage_state_from_events` gives the per-`(dataset, stage)` breakdown.
Use the shared `stage_event` context manager + `emit_missing_prereq` helpers in
`_cli_staged_workers.py` rather than hand-writing the started/completed/failed
trio.

## SLURM chaining (`_cli_staged_slurm.py` + `_cli_staged_slurm_worker.py`)

### Array trigger routing, not scheduler sidecar jobs

Do not add a **scheduler sidecar job**, meaning an extra `sbatch` job intended
to run in parallel beside an already active ordinary array. The cluster's
allocation and submission bounds are consumed by the array cohort, so an
outside sidecar may remain pending, starve the work it is meant to accompany,
or exceed the bounded submission topology.

Route ancillary work that must run with an ordinary array **through the array
itself**. Insert a reserved trigger token into the array task-entry list and
dispatch that token inside the generated array script. Follow the existing
`_CHECKPOINT_SENTINEL = "__PHENOTYPIC_CHECKPOINT__"` and
`_MANIFEST_SENTINEL = "__PHENOTYPIC_MANIFEST__"` pattern in
`_cli_slurm_array_scripts.py`; do not submit a parallel helper job. A new
trigger must:

- use a collision-resistant reserved `__PHENOTYPIC_<ROLE>__` token;
- be routed by an explicit array-worker branch rather than treated as an image;
- be included when calculating the task-entry count and chunk size, so the
  final `#SBATCH --array` length remains within `MaxArraySize`; and
- have tests proving both the trigger routing and the absence of a standalone
  parallel submission.

This scheduler rule is unrelated to the staged GPU Stage-2 **signal files**
(the retained raw `.npy` and its token). It also does not convert a terminal `afterany` finalizer into an array
entry: a finalizer runs after the array becomes terminal and is not a parallel
sidecar. The existing staged-GPU controller topology is a specialized,
explicitly capacity-reserved design; do not generalize it into new ordinary
array sidecars.

> **Queue ordinary SLURM work through the drip-feed dispatcher, and staged GPU
> work through its recoverable controller.** The CPU autonomous strategy and
> `--recompile` funnel their
> ordered chunk scripts through `submit_slurm_script_chain`
> (`_cli_slurm_submission.py`) → `generate_dispatcher_chain` +
> `submit_drip_feed_start` (`sdk_/slurm/_dispatcher.py`). The dispatcher submits
> **only chunk 0 + a tiny dispatcher job** up front; when chunk N ends, its
> dispatcher submits chunk N+1. Peak queue occupancy stays at ~1 chunk + 1
> dispatcher, so a run's full task count (which for the staged path is
> ~2 × n_images) never trips the per-user `MaxSubmitJobs`. A new SLURM
> submission site MUST reuse this helper, not loop `submit_script` over all
> chunks — eager submission is what caused the `AssocMaxSubmitJobLimit` failures.

`StagedSlurmStrategy` writes per-stage SBATCH scripts plus a controller script.
The dispatcher paragraph above applies only to ordinary CPU/recompile chains;
the staged GPU path no longer flattens its whole lifecycle into that dispatcher.
It creates an orchestration UUID, versioned image manifest, atomic controller
state, and append-only job ledger before submitting Controller 0. That controller
pre-arms its recovery controller before submitting Stage-1 chunk 0. Each controller
thereafter pre-submits its recovery
controller, then either launches the next Stage-1/Stage-3 chunk, launches a
Stage-2 round, or launches the finalizer. Deterministic SLURM comments let a
successor discover a job when its predecessor died after `sbatch` but before
persisting the returned ID.

- Stage 1 / Stage 3 = arrays over **images**; Stage 2 = an array over
  **shards** (`--gpu-shards`, `partition_shards`), each a resident-model
  `run_stage2_shard`. Controller jobs run on `config.slurm_args` (CPU).
- **Array chunking (`min(MaxArraySize, MaxSubmitJobs - 2)`):** the image count is
  split into `ceil(n_images / chunk_limit)` chunk scripts. The two reserved slots
  are for the running controller and its dependent recovery controller.
  (`calculate_optimal_array_chunks` → `_write_image_stage_chunks`), where
  `get_slurm_max_submit_jobs()` conservatively uses the smallest configured QoS
  or user-association limit, and `chunk_limit` reserves one submission slot for
  the dependent dispatcher queued alongside the active array. A single chunk
  must fit **both** the array-index cap and the remaining per-user submit
  capacity. Known limits below three are rejected.
  Each chunk is a 0-based `--array=0-(k-1)` whose `TASK_INDICES` window holds the
  **absolute** manifest indices and whose worker reads
  `--index $CURRENT_TASK_INDEX`, so no array index ever reaches the limit. A
  single chunk keeps the plain `stage1.sh`/`stage3.sh` name; multiple become
  `stageN_chunk{i}.sh`. Stage 2 is **never chunked** (a shard worker streams its
  whole shard on one GPU); `--gpu-shards > chunk_limit` raises. `generate_staged_scripts`
  returns stage arrays plus controller, finalizer, config, and manifest paths.
  Image stages are always lists. The finalizer is a
  one-task CPU job that reloads the canonical pipeline and runs the same
  aggregate/finalize path as ordinary SLURM, including named analysis and plot
  publication. The controller records every dynamically submitted ID and keeps
  only one work array plus its recovery controller active at a time.
- Per-stage resources: Stages 1 & 3 use `config.slurm_args` (CPU); Stage 2 uses
  `resolve_stage_slurm_args(gpu_slurm_args, slurm_args)` — inherit/delta over the
  CPU profile, auto-add `slurm_gpus_per_node=1` (explicit `=0` **omits** the
  directive so a CPU partition can run the GPU stage, e.g. tests).
- **Walltime survival:** Stage 2 does not install a signal handler or self-requeue.
  After each array reaches a terminal scheduler state, its dependent controller
  reclassifies the manifest from valid stores, complete Stage-2 signals,
  Stage-3 markers, and terminal failures. Remaining images launch another array round. One unchanged
  retryable-set round is retried; a second unchanged round terminalizes the
  remainder and advances to Stage 3.
- Every worker checks the active epoch immediately before publishing the store,
  the Stage-2 signal, parquet, plot, or deletion changes. Restart and cancellation fence
  stale workers before clearing or cancelling ledgered jobs.
- Without `--wait`, staged submission returns `PROCESSING SUBMITTED` and remote
  finalization owns aggregation, reports, README, and the completion marker.
  With `--wait`, the CLI monitors that marker and never duplicates publication.
- Per-image isolation: a missing prereq (S6) is recorded and skipped, never an
  unhandled raise that aborts a shard.

## Legacy-tree migration (`--mode migrate`)

Migration runs locally without `--slurm`. With one or more `--slurm key=value`
arguments, the public CLI creates a fresh generation-scoped
`MigrationSlurmPlan` and submits its ordered metadata → image → seal → optional
reclaim → finalizer chain through `submit_migration_slurm_plan`. An explicit
`--njobs` is invalid on that path — including an explicit value equal to the
Click default — because the scheduler controls worker parallelism; determine
that distinction at the Click boundary with `ParameterSource`, never by
inspecting the numeric value.

Validate incompatible migration options before generating a plan or initializing
lifecycle state. A SLURM `--dry-run` writes its manifest, config, scripts,
statuses, lifecycle, and logs only below the external cache control root, never
inside the scientific output tree. It submits the metadata preflight and
throttled per-image validation chain; without `--wait`, report its durable job
IDs, generation, and control references. With `--wait`, dry and non-dry attempts
wait until the finalizer has closed the matching lifecycle, then validate that
generation's typed terminal status. A failed or missing terminal authority is a
Click error carrying the durable failure reason. Each rerun is a new
generation/attempt, rather than reuse of a prior failed scheduler token.

Local mutation and the complete plan/initialize/submit critical section hold a
shared-filesystem migration attempt lease. Recovery must acquire that lease
nonblocking before declaring an owner dead or a submission abandoned; hostname
and PID fields are diagnostic only across hosts. For a ledgered SLURM attempt,
only an explicit known terminal scheduler state permits interrupted
terminalization. Every active/held/requeue/signaling state preserves the attempt,
and an unknown state fails closed.

## Per-image figures

Every `PlotImage` binding's figures are written **into the image's store**, in
one folder per run, `figures/<date>-<pipeline sha[:12]>/`. `deliverables/plots/`
is then filled by copying that folder out. The store layout and descriptor are
in the `working-with-ome-zarr` skill. The build is
`build_image_figures` (`plotting/_pipeline/_store_figures.py`). It writes
nothing, and no figure error can fail the image. The store write is
`write_image_figures` (`sdk_/_image_figures.py`), and the copy-out is
`publish_store_figures` (`plotting/_pipeline/_store_copyout.py`).
`PlotCoordinator.emit_image` is **retired**. Nothing renders an image plot
straight to `deliverables/` any more.

| Mode | Build | Store write | Copy-out |
|---|---|---|---|
| Full (`_cli_process_single.py`) | after `apply_and_measure` | `save_image_store(figures=)`, inside the root-last transaction | yes |
| Staged Stage 1 | only the bindings `split_pipeline_at_gpu` gave to `pre_pipeline`, after the pre-GPU ops | Stage 1's `save_image_store(figures=)` | no; Stage 3 publishes |
| Staged Stage 3 | `keep_image_figures` for Stage 1's bindings, merged with `build_image_figures(post_pipeline, keep_from=<Stage-1 store>)` | `save_image_store(figures=)` | yes |
| Measure | after `measure()` and **before** the table replace | `replace_image_tables(figures=)`: the same root-last transaction as the tables | yes |
| Process, `--process-format zarr` | after `pipeline.apply`, before the provenance status is closed | `write_process_only_layer(figures=)` | no `deliverables/` |
| Process, `tiff` (and the `--layer objmap` PNG) | none for the exported file | none | none |

- **Copy-out runs after the store is promoted and before anything marks the
  image complete.** Full mode: promotion → copy-out → the caller's completion
  record. Stage 3: promotion → copy-out → Stage-3 marker → token → raw. Measure:
  the table-and-figures transaction → copy-out → marker refresh. A crash between
  promotion and copy-out therefore re-runs the image. The copy-out is
  best-effort. It publishes only **this run's** folder. It verifies each file
  against its `sha256`. It writes an `.html` for each stored `plotly-json`, and
  it never renders a PNG. Every failure, including each descriptor `failed`
  entry, becomes a `.failures.jsonl` line; only `PlotPublicationBlocked`
  propagates.
- **One run date for the whole run.** The CLI captures the initial call once
  (`mint_run_initiation`): its UTC date, UTC timestamp and pid. It records them
  in `state.config` as `figures_run_date`, `initiated_at_utc` and
  `initiated_pid` before any worker starts, and a resume reuses them
  (`_run_initiation` in `phenotypicCLI.py`). `--restart` and `--overwrite` mint
  a new call. Measure mode keeps no run state. Locally it mints its own call; on
  SLURM its submitter writes the same three keys into `job_metadata.json` before
  fan-out. In-process workers get the call on the `OutputManager`. A worker in
  another process reads it back with `recorded_run_initiation` (state) or
  `metadata_run_initiation` (measure on SLURM). **It never travels on a command
  line**: no worker argument and no generated script carries it. It is not in
  `processing_configuration_digest`, so a new day never invalidates
  continuation, and a run that crosses midnight stays in one folder.
- **Measure mode reuses a folder.** If the store already has a run with this
  pipeline's sha256, measure mode takes the most recent such run's date
  (`latest_run_date`). It then overwrites that folder's measurer figures and
  keeps its §3a figures. Otherwise it writes a folder dated by its own call. The
  digest is read from the pipeline file measure mode runs, not from the store's
  journal.
- **§3a figures (`FigureInputUnavailable`).** A binding whose `inspect()` raises
  it is kept from the same run's folder, sha-verified and all or nothing, when
  the store being replaced has one. Otherwise it is listed in the run's
  `unavailable`, not in `failed`. It is never copied between run folders.
- **Nothing is wiped.** Every store rewrite carries the other runs' folders
  (`carry_figure_runs` in `_write_store_part`; the measure rewrite keeps them as
  hard links). Only `--overwrite`, which deletes the output folder, removes
  them.
- **A run folder that cannot be named** (no pipeline digest, a malformed date)
  gives the image no figures. It records one `.failures.jsonl` line with
  binding id `"<run>"` (`name_figure_run`), and the image still completes.
- **Continuation.** Process mode bumped `PROCESS_LAYER_SEMANTICS_REVISION` from
  2 to 3, so a process tree resumed across the upgrade re-derives, `tiff`
  included. Full mode has no revision: a full run resumed across the upgrade
  reuses figure-less stores, and `--overwrite` fills them in. `--mode migrate`
  never adds a figure.

## Durability (`--durable-writes` / `--no-durable-writes`)

Whether each per-image store is `fsync`ed before its promote. The flag is a
**tri-state**: unset means *auto-detect* — on under SLURM, off locally — and
`None` is carried end to end, resolved to a bool in exactly one place,
`ngff_._resolve_durability`, which also produces the sentence logged at run
start so the flag and its description cannot drift. **Do not resolve it
earlier**: that would freeze the submitting node's environment into a value a
worker on a different node then reuses.

Durability lives on the **`OutputManager`**, not on each call.
`save_image_store(durable=None)` defers to `self.durable_writes`, which is what
makes it structurally impossible for a write site to be silently inert. There
are exactly **three** write sites — `_cli_process_single.py`,
`_cli_staged_workers.py` Stage 1 and Stage 3. Everything else that appears in a
grep is a **transport** site: a fresh process that must be handed the flag on
its command line, namely the staged SLURM worker, the ordinary per-image SLURM
array (`_cli_slurm_array_scripts.py` → `python -m phenotypic._cli._cli_process_single`),
and the staged script generator. A new spawn site needs the flag threaded, or
the option is inert on that path alone.

**That is the rule for this one flag, not the pattern for a new per-run
value.** A value that must be the same for every worker, stage and resume
belongs in the run's recorded state, and workers read it from there. The figure
run date and the initial call's timestamp and pid work this way (see
**Per-image figures**). They are recorded in `state.config`, or in
`job_metadata.json` for measure mode on SLURM, and they are never handed to a
worker on its command line.

`durable_writes` is deliberately **not** part of
`processing_configuration_digest` (`_cli_failure_tracker.py`, an explicit
allowlist). Durability is a storage guarantee, not a scientific parameter —
folding it into the work id would make `--no-durable-writes` restart a finished
run from zero.

Rejected with `--mode recompile` and `migrate`: those
modes write no image store from a pipeline, so the flag could only mislead.

## One writer per artifact, one producer per derived value

**The rule the state-tracking design rests on, and it is load-bearing in two
different places.** Both were learned the expensive way in the
`cli-gui-state-tracking` change; the reasoning is in
`docs/superpowers/reports/2026-09-03-cli-gui-state-tracking/`.

### One writer per artifact, per pass

**Each worker owns exactly one image and writes its artifacts once.** No path
rewrites a tracked artifact within a pass. Retries, `--restart` and `--mode
measure` are separate invocations, seconds to hours apart.

That is not a convention — **it is what makes the verification cache correct.**
The cache fences on `(size, mtime_ns)`, which cannot detect a same-size rewrite
landing inside one filesystem `mtime` tick. It does not have to, because no such
rewrite occurs. Measured on GPFS `/bigdata`: 0 of 200 back-to-back same-size
writes shared an `mtime_ns` (~81 µs resolution); node-local scratch is 181/200 at
~1 ms, which is one more reason run output may never live there.

**So a change that makes any component rewrite a tracked artifact twice within a
pass silently invalidates the cache's soundness argument** — and nothing will
fail, because the cache will simply keep answering from a stale verdict. If you
need a second write, say so where the cache is defined, not only where you write.

### One producer per derived value

**A value derived from state has exactly one producer: the reader of that
state.** Do not compute it a second time somewhere earlier and carry it.

`inventory_digest` is the worked example. It is `canonical_digest(work_ids)` —
a pure function of `processing_state.json` — and four sites compute it that way
and agree: `sdk_/_run_state.py` and the three proof writers in
`_cli_completion.py`. A fifth producer in `_cli_identity.py` derived it from the
`--image-manifest` digest instead, which is `None` by default. So the field whose
job is answering *"did the accepted scope change?"* answered **"no"**
unconditionally, and could never equal the reader's value for any input.

**The tell was the ordering.** `mint_run_identity` runs before
`state.config["work_ids"]` exists. A value that cannot be computed at the point
you are standing is a value you do not own — the fix was to leave it empty and
document the reader as its owner, not to invent a stand-in.

> **Before adding a producer, ask what already computes this and whether you can
> reach it.** If you cannot reach it because of ordering, you are not its
> producer. If you cannot reach it because of layering, the shared definition
> needs a home in `sdk_` — the layer the CLI, the GUI and the readers may all
> import. It is never a reason to restate it.

## Per-image completion markers

`publish_image_success` certifies the artifacts an image produced;
`valid_image_success` is the first conjunct of resume classification. Per-image publications are **versioned**, and there are **two constants, not
one renamed**: `RECORD_VERSION` (`sdk_/_image_record.py:87`, currently **1**)
stamps the per-image record `images/<ds>/<stem>.json`, and
`SUCCESS_MARKER_VERSION` (`sdk_/_io_constants.py:741`, currently **2**) is the
legacy `image_complete/` marker's, still written by the HDF→Zarr migrator
(`sdk_/_hdf_to_zarr.py:607,645`). A version mismatch invalidates rather than
migrates. Do not document either as having replaced the other; both have live
writers.

Never hand-declare the per-image data artifact. Call
**`image_data_artifact(output_dir, output_manager, dataset, image_stem)`**,
which returns `("store", <store dir>)` when a store exists and `("hdf", ….h5)`
otherwise. Three separate clusters each broke a run by declaring `.h5`
directly after the writer had moved to the store — `publish_image_success`
resolves every artifact `strict=True`, so the image fails *after* completing
all of its work.

Descriptors dispatch on `kind`. A **store** is fingerprinted by its root
`zarr.json` alone, not recursively: the root is written **last** by the promote
protocol, so a valid root implies a complete store. An absent `kind` reads as
`"file"` (v1 shape); an unknown `kind` **fails closed**.

> **"Nothing writes into the store after publication" is FALSE, and root-only
> fingerprinting is sound anyway.** The stronger claim was written here and is
> withdrawn — the table replacers have **three** call sites, not the one
> (`_cli_migrate_image.py:281`) an earlier note named. The third,
> `OutputManager.replace_image_store_measurements` ←
> `_cli_process_single.py`, runs on the ordinary **`--mode measure`** path and
> writes into a store that is already published. *(P4 split the replacer in
> two: that path now calls `replace_image_tables`, which moves the measurement
> table, the metadata table and the root's `metadata_table` block together;
> `replace_embedded_measurement_table` survives for `--mode migrate` until its
> producer is repointed. Both go through the same root-last transaction.
> `--mode recompile` was the third writer until 2026-09-11; its per-store
> rewrite was removed rather than repointed, so recompile is now aggregate +
> finalize and writes no store byte.)*
>
> What actually holds is narrower and is what the fingerprint needs: **every
> table replacement is a root-last store transaction**, so the root is rewritten
> last exactly as at promote time.
>
> **There used to be a second write shape, and P4 deleted it.** When the
> descriptor was unchanged, `replace_embedded_measurement_table` took a
> same-directory atomic file replacement and returned **without touching the
> root** at all. That preserved *"a valid root implies a complete store"* —
> which is all the fingerprint needs — but it also meant the per-image proof's
> store digest still matched after the table's bytes had changed underneath it,
> and it left the root's recorded metadata snapshot stale. Both failures were
> silent. After P4's inversion the descriptor became a pure function of the
> measurement schema and the objmap target, so *every* metadata-driven
> re-measure would have taken that branch. The fast path is gone; the cost is
> that a `--mode measure` re-promote now runs on every image it touches.
>
> So the root never observes a partial store. The root fingerprint is
> a **completeness** check, not a content-version one, and it was never asked to
> distinguish two tables carrying the same columns — the descriptor holds
> `schema_version`, `type`, `format`, `path`, `measurement_columns` and `target`,
> and deliberately no row count or content digest.
>
> The measure path then re-publishes the marker through
> `_republish_table_marker` (`_cli_process_single.py:462`), which rehashes every
> artifact and writes the marker last, so no store write outlives the
> publication that certifies it.
>
> **Why the wording mattered even though the conclusion did not.** A reader
> checking the guarantee would have found a counterexample on the first grep and
> had no way to tell whether the fingerprint was unsound or merely
> mis-described. Overstating a true invariant costs the next reader the ability
> to verify it — and the overstatement came from a citation that named one call
> site when there were three, which is the third time in this change a dismissal
> rested on an under-counted citation.

The `"hdf"` branch is **not** dead code, though no forward path writes an
`.h5` any more. `_migrate_legacy_success_evidence` promotes trees from older
releases, which have `results/<ds>/hdf/<stem>.h5` and no store; dropping the
branch would make it name a nonexistent store and silently refuse to promote
the very trees it exists to rescue — reprocessing all of them.

---

## Tracked state, and how everything else is derived

**The full reference now lives in the contributor guide:
[`docs/source/contrib_guide/tracked_state.md`](../../../docs/source/contrib_guide/tracked_state.md)**
— *Run state: what is tracked, and how to read it*.

It is the single source for the four tracked states and their writers, the
three content proofs and their publication order, the derived-value table
naming the function behind each fact, the two retained-but-unread artifacts,
the table of **known consumers** and what each decides, and the two consumers
that deliberately ask a different question.

Read it before adding any counter, flag, marker or cached count to
`.phenotypic/`.

The three things worth carrying in your head without opening it:

- **Four things are written down**, and every other answer is computed from
  them plus the artifacts on disk. The organising principle is *move state that
  is tracked to state that is checked*. **A fifth tracked value appearing is a
  design regression** — the page's last section is the checklist that decides
  whether a proposed one is really derived.
- **`resolve_run_state(output_dir, depth=...)` is the one call.** It never
  raises; an unreadable or foreign tree degrades toward `incomplete`. Use
  `depth="deep"` for anything that writes, `depth="shallow"` for listing and
  polling.
- **Readers live in `sdk_`; writers stay in `_cli`** (INV-LAYER: `sdk_` may
  never import `phenotypic._cli`).

> **If you add, remove or change a tracked state, a proof, or a consumer,
> update that page in the same change.** It carries a checklist for exactly
> this, including adding a row to the consumer table and adding an
> `sdk_/_io_constants.py` path helper rather than hand-joining a name. A
> reference updated one change later is a reference nobody can trust in
> between, and a row that describes something no longer true is worse than a
> missing row — it carries the authority of documentation while being wrong.

---

## What `--mode migrate` asserts, and what it does not

**It converts schema. It does not finish a run.** Everything below follows from
that one sentence, and most of the surprises in this area come from expecting
otherwise.

- **A migrated-but-incomplete tree reports `incomplete`, and that is correct.**
  `publish_run_completion_evidence` refuses to write a run proof when some
  accepted image has no success record, and migrate does not override it.
  Writing one there would certify a run that never finished.
- **`stage2_done/` is read, never moved.** It holds a *consumable token* that
  Stage 3 replays and then `unlink`s. Migrate reads it to enrich a record and
  leaves it exactly where it is; renaming it into `legacy-v2/`, where nothing
  reads it, would orphan the work of any staged run live across the migrate.
  Only `image_complete/` and `stage3_complete/` move.
- **`deliverables/metadata.csv` is byte-exact provenance**, with no exception for
  migrate. The canonical view is emitted *alongside* it as
  `metadata.canonical.csv`; `metadata.original.csv` does not exist and must not
  be created.
- **The master is parquet-only** (D8). `master_measurements.csv` is deleted on
  sight, not rewritten.
- **`work_id`s are not re-minted** (D-C), so every existing one stays valid.
  `record_rejection` (`sdk_/_image_record.py:207`) skips the `work_id`
  comparison for `PROVENANCE_MIGRATED` records — a migrated tree's identity
  cannot be re-derived, so it is marked unavailable rather than fabricated.
  **Absent means `"forward"`**, so a writer that forgets the field produces a
  fenced record rather than an accepted one, and `resolve_run_state` advises
  when the relaxation is in effect.
- **Not every record written during a migration carries `"migrated"`**, and the
  difference is the code path rather than the mode: the state migrator stamps
  it (`_cli_migrate_state.py:350`, `:1080`), while migrating the image artifact
  itself goes through `publish_image_success` (`_cli_migrate_image.py:589`),
  which has **no `provenance` parameter** and so takes `publish_image_record`'s
  `PROVENANCE_FORWARD` default. **Do not use record provenance to ask whether a
  tree was migrated** — `_output_was_migrated` keys on the migration manifest
  for exactly this reason. Full table in
  [`tracked_state.md`](../../../docs/source/contrib_guide/tracked_state.md)
  under *Provenance has two writers*.

### A migrated tree is not continuable, and `--restart` is the remedy

User ruling (2026-09-09). Migration rebuilds the admitted image set from the tree
rather than from the original run's record of it, so continuation compares
against a set the inputs no longer match. **That is accepted rather than
repaired**: such a tree *"should be considered as if not ran then, and do a full
restart"*.

The refusal names the command — keyed on `.phenotypic/migration_manifest.json`,
which is a declared fact rather than an inference. **Told, never done**: clearing
machine state and reprocessing every image is hours of compute and a destructive
step, and firing it automatically from a condition the user did not ask about is
the hidden state transition this change exists to remove.

Not every migrated tree is affected — a legacy tree **with markers** carries real
filenames in `work_ids`, so nothing is rebuilt and it continues normally.

---

## Known limits — read these before trusting the above

A register that records only what works is the shape this change exists to
remove.

- **`work_id` does not fold in `restart_epoch`** (`_cli_failure_tracker.py:331-339`).
  So a pre-migration run's journalled failures still key-match after a
  `--restart`, and a later plain re-run reports *"N recorded failure(s)
  skipped"* for images the restart just reprocessed successfully. Inherited, not
  created by migration; `--retry-failures` is the escape.
- **Migration pollutes `initial_images` on a pre-markers tree**, adding bare
  stems beside the real filenames, because `_ensure_migration_processing_state`
  falls back to the stem when `work_ids` carries no filename for it. **Tolerated,
  not fixed**, and pinned by an `xfail(strict=True)` in
  `tests/integration/cli/test_migrate_end_to_end.py` so a future repair is loud.
  It is inert only because a migrated tree refuses continuation *by rule* — if
  that refusal is ever relaxed, this becomes live again.
- **O-3 and O-4** are recorded in
  `docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/OPEN-QUESTIONS.md`.

### Process trees, and the one target kind that is not a tree

**Machine-state conversion follows a shared predicate, not the dispatch.**
`target_kind_owns_machine_state` (`_cli_migrate_provenance.py:56`) is true for
`full_run`, `process_tree` and `pre_markers_process`, and **false for
`direct_store`**. The older `kind != "full_run"` test asks *is this not a full
run?* and was read as *is this per-store provenance work?* — which is false for
two of the three provenance-only kinds.

`direct_store` is excluded **by kind, not by luck**: a single store keeps its
lifecycle state in a hashed sibling, never inside itself, so converting there
would write `.phenotypic/` where that kind's own contract forbids it.

**A pre-markers process tree used to migrate to nothing, successfully.**
`execute_provenance_migration` iterates `target.stores`, and such a tree has
`stores=()` by construction — its outputs are process layers, not stores — so
the local arm did no work, reported `provenance_upgraded=0`, and **exited 0**.
The user saw success and got none. It now converts at
`_cli_migrate.py:1647`, after a clean store upgrade and only when there are no
failures.

**`--slurm` on a storeless provenance-only target is refused, by design.** It is
a `click.UsageError` naming the remedy, not a missing feature: the SLURM chain
has no vocabulary for such a tree. `seal_provenance_migration` barriers store
statuses and the finalizer reports upgrade counts, so the tree would come back
*"0 upgraded, succeeded"* with the only real work — the machine-state conversion
— invisible in its own terminal report. **Run it without `--slurm`.** Two
independent guards already refuse the shape, which is what makes it a topology
mismatch rather than an oversight in one place: the manifest writer raises on
zero tasks, and the worker config loader's `target_kind` cannot even parse a
config naming this kind.

---

## Readers live in `sdk_`; writers stay in `_cli`

`resolve_run_state` and the run-state readers are in
`phenotypic.sdk_._run_state`. Writers — anything that publishes a proof, bumps an
epoch or appends to a journal — stay in `phenotypic._cli`, because `sdk_` may not
import `_cli`. `tests/unit/sdk_/test_run_state_layering.py` enforces the
direction with an AST walk, so a violation fails there rather than at import.

**There is no migration version floor.** An earlier draft named v0.17.3 and it
was withdrawn (U-6): `state.version` is a *state-schema* version, not a package
one — `"2.0.0"` is its value both at v0.17.3 and immediately before `"3.0.0"`
arrived — so there is no version string to refuse on. Detection is by **shape**,
the pre-markers signal is an absent `work_ids` key, and a pre-markers tree is
supported however old. `ConversionVerdict` has deliberately no `BELOW_FLOOR`
member (`sdk_/_schema_shape.py:192`).

## Environment variables (important for future work)

- `PHENOTYPIC_PRELOAD_MODULES` — comma list of modules staged SLURM **workers and
  the finalizer** import before `ImagePipeline.from_json`
  (`_cli_preload.py:preload_custom_operation_modules`). Fresh remote processes
  can't see op classes defined outside the `phenotypic` namespace; list a
  self-registering module here so a pipeline with **custom operations**
  deserializes on compute nodes and during final publication. `sbatch
  --export=ALL` propagates it. (Tests use
  `tests/_fakes/register_fake_gpu.py`.)
- `PHENOTYPIC_SLURM_PYTHONPATH` — internal submission snapshot of the caller's
  `PYTHONPATH`. Generated batch scripts restore it before invoking Python. This
  keeps custom-operation modules and the reviewed source checkout importable on
  clusters that filter raw `PYTHONPATH` even when `sbatch --export=ALL` is used.
  Callers set `PYTHONPATH`; PhenoTypic owns the namespaced snapshot.
- `PHENOTYPIC_ACCEPT_MODEL_LICENSE` — comma list of model names accepted for
  gated-weight downloads; checked by `require_license_acceptance`
  (`detect/nn/_checkpoint_manager.py`). SAM2/micro-sam are ungated and never call
  it; the hook exists for Spec 2's gated models (SAM3, DINOv3). Licensing
  scaffolding: root `NOTICE` + `licenses/*.txt` + `MANIFEST.in`.

## Gotchas

- **GPU detectors stage automatically in the CLI** — `op.apply(image)` in a
  notebook is unchanged; a `GpuDetector` in a *CLI* run triggers the staged
  engine, not per-image processing.
- **Output layout** — see the **Output layout & deliverables** section below for
  the full inventory and master-vs-mirror rules.
- **HPCC SLURM heterogeneity (polars build)** — the cluster has pre-AVX2 nodes where the
  stock `polars` wheel SIGILLs ("Illegal instruction"). The project depends on
  `polars[rtcompat]` (a runtime-CPU-dispatch build that runs on pre-AVX2 nodes without a
  per-node wheel swap); numpy/scipy use runtime SIMD dispatch and are unaffected. See
  `docs/source/how_to/pages/polars_cpu_build.md`. Stage 2's GPU work runs on GPU nodes.

---

## Output layout & deliverables

User-facing run outputs live under `<output>/deliverables/` (hard cutover):
`master_measurements.parquet` (**parquet only** since D8), `measurements.{csv,parquet}`,
`measurements_by_feature/<feature>.{csv,parquet}`,
`<AnalysisClass>.{csv,parquet}`, `analysis_manifest.json`,
`plots/<plot-id>/...`,
`dashboard.html`, `processing_report.html`, `README.md`,
`pipeline.json.pht-pipe`, and `overlays/<ds>/<stem>.png` (detection overlay PNGs). The
pipeline config is **seeded** from the `--pipeline` bytes when a full or
staged run starts (`_seed_pipeline_config`, only when no config -- canonical
or legacy -- exists, under `pipeline_publication_lock`) and rewritten from
the loaded pipeline at finalize; nothing is copied to the output root any
more. The `--restart --image-manifest` freshness check treats that seeded
file and its lock file as scaffolding. The
dashboard is progress-only: local runs render progress directly, while SLURM
runs add Progress and Download tabs. Use the Results Viewer or the GUI
`/analysis/` app for interactive exploration. Each per-image **OME-Zarr store** stays at
`results/<ds>/zarr/<stem>.ome.zarr/`. Its authoritative object measurements
are embedded at `tables/measurements/table.parquet`, described by
`attributes.phenotypic.tables.measurements` in the store root. Forward,
staged Stage 3, and measure runs do not create
`results/<ds>/measurements/<stem>.parquet`; that directory is legacy migration
input only. There is no per-image `.h5` on any forward path: `results/<ds>/hdf/` appears only in a tree written by a pre-store
release, or in one migrated with the default `keep_source=True`, and the only
things that read it are `--mode migrate`, `datasets_needing_migration` (the
predicate every writing mode refuses on), and the `"hdf"` completion-marker
fallback described above. Machine state lives under
`.phenotypic/`: `progress_dir(output)` resolves
`<output>/.phenotypic/progress/` and `processing_state_path(output)` resolves
`<output>/.phenotypic/processing_state.json`; the corresponding `resolve_*`
helpers retain legacy root-level reads. The durable
**QC + curation state** lives under `deliverables/qc/` (`qc.duckdb`,
`review_state.json`, `curation_labels.parquet`, `custom_categories.json`) so a
`deliverables/` bundle is self-contained and GUI-openable standalone;
`BundleLayout.qc_dir` / `migrate_legacy_qc` still read/move a pre-relocation
root `qc/`. `run_qc` writes the
single `deliverables/qc/qc.duckdb` (one self-describing table per QC module plus a
`qc_modules` catalog, atomic full rebuild). Resolve these paths via the
`phenotypic.sdk_` helpers (`deliverables_dir`, `master_measurements_parquet_path`,
`qc_dir`, `qc_duckdb_path`, …), never by hand-joining names.

**Master vs. mirror.** *(Rewritten by P4 — spec §7.3. The paragraph this
replaced described the pre-inversion contract in every particular, and each
particular is now false.)*

Each embedded table carries **measurements alone**; the image's own rows of the
run's metadata snapshot sit beside it at `tables/metadata/pht-metadata.parquet`,
and the store root records which snapshot it was built against in
`attributes.phenotypic.metadata_table.snapshot_sha256`.
`master_measurements.parquet` is the pre-post concatenation of
marker-authorized embedded tables, each **projected onto its own store's
descriptor** first, so it is **un-joined**: intrinsic identity
(`Metadata_Dataset`, `Metadata_ImageName`, the `IMAGE`-owned provenance block)
plus measurements, and no user metadata at all.

**Not every store is inverted, and the projection is what makes that safe**
(P7 Task 4). `--mode migrate` leaves each store's embedded table as the
pre-inversion producer wrote it: the metadata snapshot right-joined in, each
measured row repeated once per matching metadata row. Aggregated as-is, those
tables make a v1-shaped master, and the one global join below then joins a
second time on the user columns — on a real 6,657-image migration that dropped
every measured row from the mirror. So every read path into a master —
`build_master_frame`, the P5 fan-out shards, and the recompile shards — goes
through `project_embedded_measurement_table` (`_cli_parquet_agg.py`): keep the
descriptor's `measurement_columns`, collapse join fan-out on `target.column`
for a table recorded `joined`, and **exclude** a store with no descriptor or
whose same-label rows disagree. An excluded store is logged and left out of the
source set the aggregate proof certifies, so the run reads `incomplete` rather
than certifying an image the master does not carry. Both shard producers report
the set they merged to their finalizer as `planned_work_ids` — the P5 fan-out
through `resolve_finalizer_shard_inputs`, and recompile through the
`source_work_ids` its measurement statuses record — so an exclusion in a shard
is not counted by the proof.

**The join happens once, at finalization**, in `finalize_run`
(`_cli_finalize_run.py`) → `finalize_post_master_outputs` →
`join_metadata(master_df, metadata_csv, how="left")`. That one call identifies
its own common columns, so **nothing reads the stores' recorded join keys** —
which is why D-A is free to leave them inconsistent across snapshot
generations. Finalization no longer rejects mixed metadata digests; divergence
is an advisory, and an advisory is never a gate. It *does* still refuse mixed
**authority** — a tree holding both embedded tables and legacy external
Parquets (`refuse_mixed_measurement_authority`).

`measurements.{csv,parquet}` is the post-applied, metadata-joined mirror the GUI
reads and curates. Metadata is the **left** frame, deliberately: a metadata
identity that matched no measured object survives as a phantom row with
`QC_MetadataOnly=true`, its metadata values kept and its measurement/info
columns null, while a measured object whose key appears in **no** metadata row
is dropped — an object outside the described experiment. The master keeps that
object; the mirror does not. That asymmetry is the master/mirror distinction the
"feed analysis and dashboards from the mirror" rule rests on. Per-feature splits
and named analysis artifacts derive from the mirror. Analysis consumers resolve
tables through `analysis_manifest.json`, never by constructing filenames.

**Reading a master written before the inversion.** Nothing stamps the file — a
v1 master carries **user** metadata because the join happened per image, a v2
carries only **intrinsic identity** metadata, and that *is* the discrimination.
Ask `phenotypic.sdk_.master_carries_user_metadata(frame)`, which is its one home
and carries its retirement condition; never re-derive the check at a reader.

**Ownership decides, never the `Metadata_` prefix** — both shapes carry it. A v2
master carries `Metadata_Dataset` and `Metadata_ImageName`, and `Metadata_Strain`
is itself a schema member (`GENETIC.STRAIN`), so neither the prefix nor
"is it in the schema" separates the two. A header is user metadata when
`is_metadata_header` accepts it and `metadata_owner_for_header` returns neither
`IMAGE` nor `EXPERIMENT.DATASET`. This is the general rule in root `CLAUDE.md`
("Metadata queries use schema ownership, never string prefixes"), and the master
is the case where getting it wrong classifies *every* v2 master as v1.

The limit that leaves: column provenance is not recoverable from a column name,
so a master carrying a non-`IMAGE` metadata column that a **custom operation**
produced reads as v1 though no CSV was joined into it. The forward pipeline emits
no such column, which is what keeps this a limit rather than a defect.

Ordinary SLURM checkpoints read only marker-authorized embedded tables and write
rolling cache state below `.phenotypic/progress/`. They do not recreate
visible per-image Parquets, dataset aggregates, or a partial deliverable master.

**Metadata snapshot authority.** Before local processing or SLURM submission,
full runs and recompile atomically copy the configured `--metadata` bytes to
`deliverables/metadata.csv`, verify the copy, and use that stable path for
finalization. The snapshot is source provenance, not a generated schema table:
legacy headers normalize only in memory, while finalization, recompile, and
explicit `--mode migrate` must leave its bytes unchanged. Recompile performs
no metadata migration preflight or mutation. All generated measurement,
analysis, QC, and REMBI outputs use canonical flat `Metadata_<Label>` headers.

`QC_MetadataOnly` is a **user-facing output column, not internal machinery** — it is how a
user filters the mirror for "which strains went undetected". Analysis/QC/post code must
**not** branch on it. Those ops are public API (a notebook calls them on frames that never
saw the CLI and carry no flag), so they detect a phantom the same way they detect any
missing value: **drop/ignore NaN**. A phantom row is null in every measurement/info column,
so NaN-native math (`notna()`, `nanpercentile`, `np.isfinite`, `dropna`) handles it, needs
no flag column to exist, and is automatically a no-op on frames that have none.
Feed configured analysis and GUI result exploration from `measurements.parquet`, not
`master_measurements.*`.

**There is one FINAL master writer, and it is `finalize_run`.** Every mode
reaches it: the forward CLI and `--mode measure` through `aggregate_measurements`
(which is `finalize_run` under the publication lock), and the recompile SLURM
finalizer through `_run_post_master_steps`, which hands its per-shard Parquets
in as `shard_paths` rather than merging and writing a master of its own. That
collapse is the point of §7.4 — recompile is *"call `finalize_run` again"*, not
a second implementation to keep in sync — and it is also what keeps **one
writer per artifact, per pass** true of `master_measurements.parquet`. A new
code path that needs a final master calls `finalize_run`; it does not write the
file and then call `finalize_post_master_outputs` itself.

Mid-run checkpoint writers (`_aggregate_chunks_locked` in
`_cli_chunk_writer.py`) intentionally bypass it and keep their rolling state
under `.phenotypic/progress/`; post, per-feature splits, analysis, and
`pipeline.json` persistence are deferred to final aggregation. Do not add
`finalize_post_master_outputs` to the chunk writer.
