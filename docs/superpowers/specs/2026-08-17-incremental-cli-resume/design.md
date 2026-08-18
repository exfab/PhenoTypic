# Incremental CLI Auto-Continue and Crash-Safe Completion

**Date:** 2026-08-17
**Status:** Approved for implementation; implementation in progress.
**Scope:** `python -m phenotypic` forward runs in local and SLURM execution.

## 1. Purpose

An experiment may add images to the same input directory over hours or days. The
same PhenoTypic command must be safe to invoke repeatedly against the same input
and output paths without a continuation flag:

```bash
uv run python -m phenotypic \
  --pipeline pipeline.json \
  --input ./experiment \
  --output ./results
```

The first call starts a run. Later calls accept append-only input additions,
process only incomplete images, rebuild aggregate measurements from the complete
accepted inventory, and leave the output caught up with the experiment.

The same contract must survive Python exceptions, parent-process crashes, worker
crashes, OOM kills, SLURM timeouts, controller loss, and finalizer failure. A
failure may delay an update, but it must not make incomplete work look complete
or require destructive recovery. Supported readers must never consume an
aggregate whose publication evidence does not match its bytes. A failed
aggregate promotion may make the newest snapshot temporarily unavailable until
recovery, but it cannot silently present a mixed snapshot as valid.

Local and SLURM execution are independent. One invocation selects exactly one
backend, and local and SLURM work do not execute against the same output at the
same time. A later invocation may select the other backend after the preceding
invocation is terminal, provided both environments can access the same input and
output state.

## 2. Scope

### 2.1 In scope

- Default auto-continue behavior for `--mode full`.
- Append-only additions to existing datasets and addition of new dataset
  directories.
- A no-op result when no work or finalization repair is required.
- Input readiness checks and persisted input identity.
- A common per-image completion contract for local, ordinary SLURM, local staged
  GPU, and staged GPU SLURM execution.
- Backend-specific active-execution protection.
- Crash-safe, independently resumable finalization.
- A best-effort progress manifest derived from durable evidence.
- Durable, append-only terminal-failure suppression keyed to the exact image
  computation, with explicit retry through `--retry-failures`.
- Backward-compatible migration from processing-state schema 2 and staged Stage
  3 markers.
- `--mode process` using the same auto-continue lifecycle with a mode-specific
  per-image output contract.
- Local and SLURM integration, unit, failure-injection, and end-to-end tests.

### 2.2 Out of scope

- Watching the input directory continuously within a long-lived daemon.
- Accepting additions while a local or SLURM invocation is active. They are
  accepted by the next invocation.
- Simultaneous local and SLURM consumption of one work queue.
- Editing, replacing, renaming, or deleting an already accepted input image.
- Automatically invalidating only the downstream artifacts of a changed input.
  Use `--restart` or a new output directory instead.
- Recursive input layouts deeper than the scanner's existing one-level dataset
  structure.
- Making `--mode measure` incremental. Measure and recompile remain explicit,
  one-shot modes over an existing output.
- A transactional generation directory for every optional user-facing
  deliverable. Core aggregate files use staging plus marker-last publication;
  supported readers reject them while their aggregate-publication marker is
  absent or invalid.

## 3. Current behavior and gaps

The design changes several current contracts rather than only changing the
Click default:

1. Earlier CLI behavior exposed an opt-in continuation flag. The new contract
   removes that flag instead of preserving a second lifecycle entry point.
2. Automatic continuation must distinguish a fresh output from a compatible
   output with processing state; the first invocation has no state.
3. Resume currently requires exact filename-set equality within every existing
   dataset. Added files are rejected (`src/phenotypic/phenotypicCLI.py:264-318`,
   `:1555-1566`).
4. `initial_images` is a fixed set used to fence event aggregation
   (`src/phenotypic/_cli/_cli_state_management.py:106-127`, `:239-250`). A new
   image or dataset must be committed to the accepted inventory before its
   events can contribute to state.
5. The remaining-work helper admits an entirely new dataset but does not itself
   persist that dataset into processing state
   (`src/phenotypic/_cli/_cli_state_management.py:359-393`). Without explicit
   reconciliation, the dataset can be rediscovered as new on later calls.
6. Ordinary CPU resume finalizes from the filtered worklist, whereas staged GPU
   resume substitutes the full dataset list
   (`src/phenotypic/phenotypicCLI.py:1784-1786`). Incremental finalization must
   always use the full accepted inventory.
7. Final aggregation may prefer `_dataset_aggregated.parquet` over individual
   per-image sources (`src/phenotypic/_cli/_measurement_sources.py:62-129`). A
   healthy but stale aggregate can therefore omit newly added or newly repaired
   images.
8. The current manifest treats completed plus failed equal to total as complete
   (`src/phenotypic/_cli/_dashboard/_manifest_builder.py:632`). A failed image
   must remain incomplete, but a caught terminal scientific failure must not be
   retried again unless its computation identity changes or the user supplies
   `--retry-failures`.
9. Local ordinary processing records success in the event log after writing
   image artifacts but has no durable per-image terminal marker
   (`src/phenotypic/_cli/_cli_execution_strategies.py:307-383`). A hard kill can
   occur between artifact writes and the event append.
10. Staged GPU processing already writes a per-image Stage 3 marker last and
    resumes from artifact state (`src/phenotypic/_cli/_cli_staged_resume.py:99-190`,
    `src/phenotypic/_cli/_cli_staged_workers.py:199-226`). This is the precedent
    generalized by this specification.
11. Finalization currently resets GUI QC review progress
    (`src/phenotypic/_cli/_cli_output_manager.py:1014-1020`). A no-op incremental
    invocation must not reset it, and a data update needs an explicit review
    invalidation policy.
12. Manifest completion currently gates ordinary SLURM publication, sentinel
    aggregation, local GUI publication, SLURM observation, Results Viewer
    consistency, and recent-run classification
    (`src/phenotypic/_cli/_cli_checkpoint_handler.py:365`,
    `src/phenotypic/_cli/_cli_sentinel.py:150`,
    `src/phenotypic/_cli/_cli_gui_lifecycle.py:93`,
    `src/phenotypic/gui/run_console/_slurm_observer.py:1310`,
    `src/phenotypic/gui/results_viewer/_output_consistency.py:126-148`,
    `src/phenotypic/gui/shell/_runs_registry.py:873`). All must migrate together
    for the manifest to become a true cache.
13. Measure mode rewrites the same per-image parquets terminal aggregation reads
    (`src/phenotypic/_cli/_cli_process_single.py:159-166`). It cannot remain a
    state-free mutator of schema-3 marker-authorized artifacts.
14. Current output helpers downgrade or swallow failures for required master,
    mirror, pipeline, post, and metadata behavior
    (`src/phenotypic/_cli/_cli_output_manager.py:421-454`, `:681-767`,
    `:934-958`, `:1318-1335`). Marker-backed core publication therefore needs a
    strict API rather than interpreting current best-effort returns as success.
15. Ordinary SLURM deactivation currently follows successful completion
    publication (`src/phenotypic/_cli/_cli_checkpoint_handler.py:330-408`). A
    terminal array with missing image markers needs a separate
    terminal-incomplete lifecycle transition.
16. Sampling currently occurs from the freshly scanned directory before resume
    filtering (`src/phenotypic/phenotypicCLI.py:1512`,
    `src/phenotypic/_cli/_cli_interactive.py:257`). Stable continuation requires
    persisting the sample policy, seed, and exact cohort.

## 4. Decisions

| ID | Decision |
|---|---|
| D1 | An ordinary forward invocation uses lifecycle policy `auto`: create a run when no state exists, otherwise continue a compatible run. |
| D2 | The public continuation flag is removed. Repeating a compatible command is the sole continuation entry point. |
| D3 | Accepted inputs are append-only. Missing or changed accepted inputs are errors. |
| D4 | `processing_state.json` is authoritative for accepted inventory and scientific compatibility. |
| D5 | A valid per-image completion marker, not the latest event, is authoritative for per-image completion. |
| D6 | `aggregate_publication.json` authorizes one internally consistent aggregate snapshot. `run_completion.json`, matched to current inventory and finalization-input digests, additionally proves that every accepted image completed. |
| D7 | `manifest.json` is a best-effort derived snapshot. Failure to write it never invalidates completed scientific work and never prevents later recovery. |
| D8 | One invocation selects one backend. Local and SLURM never consume the same active worklist. |
| D9 | Work submitted to SLURM is an immutable snapshot. Files arriving after submission wait for the next invocation. |
| D10 | A valid success marker wins. Otherwise an exact matching record in `.phenotypic/terminal_failures.jsonl` suppresses that `work_id` by default. Otherwise the image is pending. `--retry-failures` ignores matching terminal records for that invocation only. |
| D11 | New and ordinary pending images are processed before terminal failures selected by `--retry-failures`, so a persistent bad input does not prevent live acquisition updates. |
| D12 | Aggregate publication is idempotent. It publishes all currently marker-authorized successes after an invocation becomes terminal, even when some accepted images failed. An aggregate-publication marker is written for every valid non-empty snapshot; a run-completion marker is reserved for the all-success case. |
| D13 | Terminal aggregation reads only individual measurement parquets authorized by valid completion markers. Checkpoint aggregates remain caches for mid-run publication and are not terminal authority. |
| D14 | Backend resource settings are not scientific identity. A terminal run may continue later using the other backend if state and artifacts are shared. |
| D15 | No worker signal handler, cleanup hook, or failure event is required for correctness. Scheduler state is never completion authority, but it is required as liveness and takeover evidence when durable SLURM lifecycle state is nonterminal or submission acknowledgment is ambiguous. |
| D16 | A valid completion marker is written only after all required artifacts for its contract have been atomically published and validated. |
| D17 | No-op invocation refreshes the manifest but does not rerun finalization, rewrite deliverables, or reset QC review state. |
| D18 | `.phenotypic/terminal_failures.jsonl` is the sole terminal-failure authority. It is append-only, file-locked, flushed, and `fsync`ed. Progress events, `progress/failures.jsonl`, scheduler state, and legacy staged journals are diagnostic only. |
| D19 | Only a caught exception inside an explicitly bounded per-image scientific operation may become terminal. OOM, `MemoryError`, timeout, node loss, preemption, cancellation, lifecycle or lock failure, artifact publication failure, and aggregate finalization failure remain pending and retryable. |
| D20 | When metadata is configured, startup atomically snapshots it to `deliverables/metadata.csv` before processing or SLURM submission. That stable copy, not the external source path, is the finalization input. |

## 5. Terminology and authority

### 5.1 Accepted inventory

The set of dataset and image identities committed to `processing_state.json`.
Discovery alone does not make a file accepted. Reconciliation validates the file
and atomically commits it before workers may emit events or artifacts for it.

### 5.2 Invocation

One execution of `python -m phenotypic`. It selects either `local` or `slurm`.
It has a unique `invocation_id`, but belongs to the stable processing generation
of the experiment.

### 5.3 Worklist

The immutable set of accepted images selected for one invocation. It contains
new and interrupted images whose completion evidence is missing or invalid,
plus exact matching terminal failures only when that invocation specifies
`--retry-failures`.

### 5.4 Work identity

Every accepted image computation has one `work_id`, calculated as SHA-256 over
canonical JSON containing the schema version, dataset, normalized input-relative
path, accepted input SHA-256, pipeline fingerprint, per-image scientific
processing-configuration digest, and mode. Dataset and path prevent identical
bytes in distinct logical images from sharing an outcome. Backend resources,
worker counts, checkpoint settings, metadata, study, QC, and aggregate-only
finalization inputs are excluded.

The same helper supplies `work_id` to immutable worklists, lifecycle
assignments, image completion markers, terminal-failure records, and manifest
reconstruction. The failed stage is diagnostic and is not part of `work_id`.

### 5.5 Image completion

An image is complete only when its mode-specific marker exists, parses, matches
the accepted input and pipeline identities, and all marker-declared required
artifacts validate.

### 5.6 Aggregate and run completion

`aggregate_publication.json` proves that its declared aggregate snapshot was
published consistently from marker-authorized per-image inputs. A run is
complete only when every accepted image is complete and a matching
`run_completion.json` proves that the publication covers the current inventory
and finalization inputs.

### 5.7 Authority table

| Question | Authority | Non-authoritative evidence |
|---|---|---|
| Which images belong to the run? | `processing_state.json` accepted inventory | Current directory listing, events, manifest |
| Did one image finish? | Valid image completion marker plus declared artifacts | Completed event, parquet existence alone, SLURM state |
| Is one exact image computation terminally failed? | Valid matching record in `.phenotypic/terminal_failures.jsonl`, unless a valid success marker exists | Event status, `progress/failures.jsonl`, scheduler state, legacy failed sets |
| Is work currently active? | Local run lock or backend-specific SLURM lifecycle | Manifest status, recent mtime |
| Is an aggregate snapshot safe to read? | Valid `aggregate_publication.json` matching every required aggregate artifact | Master existence, mirror existence, or manifest alone |
| Did all accepted work and finalization finish? | Valid `run_completion.json` for current inventory and finalization-input digests, referencing the valid aggregate publication | Aggregate publication alone or `manifest.is_complete` alone |
| What should the dashboard display? | Derived manifest | None; the manifest is itself a cache |

## 6. Persistent layout

New paths are provided through `phenotypic.sdk_` helpers and are never hand-joined
at call sites:

```text
<output>/.phenotypic/
├── processing_state.json
├── processing_events.log
├── terminal_failures.jsonl            # authoritative, append-only
├── staging/
│   └── <invocation-id>/               # never authoritative
└── progress/
    ├── manifest.json
    ├── failures.jsonl                  # legacy/rebuildable display detail only
    ├── aggregate_publication.json
    ├── run_completion.json
    ├── active_execution.json           # current backend and execution epoch
    ├── publication_history/
    │   └── <publication-id>.json
    ├── batches/
    │   └── <invocation-id>.json        # persisted for SLURM; optional local audit
    ├── image_complete/
    │   └── <dataset>/
    │       └── <stem>.json
    └── stage3_complete/                 # legacy read/migration only
```

QC review archives live under
`<output>/deliverables/qc/review_state_history/` beside the durable QC state.

`image_complete/<dataset>/<stem>.json` requires unique `(dataset, stem)` output
identity. Reconciliation rejects inputs such as `plate.tif` and `plate.png` in
the same dataset because both map to the same HDF, parquet, and marker paths.

Legacy `stage3_complete` remains readable during migration but receives no new
writes after all staged writers adopt the general marker helper.

## 7. Processing-state schema 3

The state format advances from `2.0.0` to `3.0.0`:

```json
{
  "version": "3.0.0",
  "pipeline_path": "/abs/pipeline.json",
  "input_path": "/abs/experiment",
  "output_dir": "/abs/results",
  "timestamp": "2026-08-17T09:00:00",
  "last_updated": "2026-08-17T12:00:00",
  "processing_generation": "stable-uuid",
  "scientific_config_digest": "sha256",
  "inventory_digest": "sha256",
  "finalization_input_digest": "sha256",
  "cohort": {"policy": "all", "requested_size": null, "seed": null},
  "datasets": {
    "day_03": {
      "images": {
        "plate_001.tif": {
          "relative_path": "day_03/plate_001.tif",
          "stem": "plate_001",
          "size": 48199230,
          "mtime_ns": 1786901123456789000,
          "sha256": "sha256",
          "accepted_at": "2026-08-17T12:00:00"
        }
      },
      "errors": {}
    }
  },
  "config": {
    "pipeline_sha256": "sha256",
    "phenotypic_version": "0.18.1",
    "image_type": "GridImage",
    "nrows": 8,
    "ncols": 12,
    "bit_depth": null,
    "detect_mode": "gray",
    "process_only_layer": null,
    "include_dataset_column": true,
    "finalization_inputs": {
      "metadata": {"path": "/abs/metadata.csv", "sha256": "sha256"},
      "study": null,
      "no_qc": false
    }
  }
}
```

The serialized form may retain `completed`, `failed`, and `errors` compatibility
views during migration, but resume selection does not trust those sets after
schema 3. They are derived from markers and events when needed.

### 7.1 Inventory digest

`inventory_digest` is SHA-256 over canonical JSON containing:

- state schema version;
- processing generation;
- pipeline and scientific-configuration digest;
- sorted dataset names;
- for each dataset, sorted image names and accepted SHA-256 digests.

Paths outside the input root, timestamps, backend, `n_jobs`, SLURM resources,
and invocation ID are excluded. The canonical encoder uses UTF-8, sorted keys,
compact separators, and no platform-native path separators.

The digest changes whenever accepted scientific input changes. It is copied into
aggregate and run markers so old evidence becomes nonmatching immediately after
the atomic state update, without requiring marker deletion.

Per-image markers deliberately do **not** carry the full inventory digest. They
carry the stable processing generation, scientific-configuration digest, and
their own input digest. Otherwise accepting one new image would invalidate every
previously completed image marker and force the whole experiment to reprocess.

### 7.2 Scientific compatibility

Continuation rejects changes to:

- pipeline content digest;
- PhenoTypic version unless an explicit future migration policy permits it;
- image type;
- grid dimensions after preset resolution;
- bit depth and detection mode;
- process-only layer;
- output identity settings that change artifact meaning;
- input and output roots after normalized resolution.

Continuation may change:

- local versus SLURM backend, when no invocation is active;
- local worker count;
- SLURM partition, memory, walltime, account, array sizing, and wait behavior;
- checkpoint interval;
- logging and display options.

Post, analysis, plot, and QC configuration are handled by the pipeline digest.
External metadata and study files are recorded by normalized path and SHA-256.
`no_qc` and any other CLI option that changes finalization output are recorded
canonically beside them. Their combined digest is
`finalization_input_digest`. A change does not reprocess images, but it makes
the old run-completion evidence nonmatching and triggers finalization-only
publication.

## 8. Input discovery and reconciliation

### 8.1 Directory rules

The existing input layouts remain:

- one image file;
- one flat image directory;
- one input root containing one level of dataset directories.

A flat run may add more root images. A dataset-directory run may add images to
existing dataset directories or add new dataset directories. Transitioning from
flat to mixed root plus subdirectories remains invalid
(`src/phenotypic/_cli/_cli_directory_scanner.py:46-135`).

### 8.2 Readiness

An image is eligible for acceptance only when:

1. it matches a supported extension and existing dotfile exclusions;
2. its name does not indicate a temporary acquisition file;
3. it is at least `--input-settle-seconds` old, default 5 seconds;
4. size and `mtime_ns` are unchanged across two observations spanning the settle
   interval, unless the producer contract guarantees atomic rename into place;
5. it can be opened sufficiently to validate its image container; and
6. it does not collide on `(dataset, stem)`.

Unready images are not accepted and do not increase `total_images`. They are
reported as `discovered_unready` in the manifest and reconsidered on the next
invocation. They are not recorded as processing failures.

### 8.3 Identity and mutation detection

New accepted images receive size, `mtime_ns`, and SHA-256 identity. On later
invocations:

- missing accepted images are errors;
- changed size or `mtime_ns` triggers a SHA-256 comparison and then an error if
  content changed;
- unchanged stat identity is accepted without rehashing by default;
- the worker rechecks stat identity before reading and immediately before
  publishing its completion marker;
- a worker-detected change prevents marker publication and records an input
  mutation failure.

This is crash and accidental-mutation protection, not adversarial tamper
detection. A byte replacement that deliberately preserves both size and
`mtime_ns` is outside the default threat model.

### 8.4 Reconciliation transaction

Under the output mutation lock:

1. load or create processing state;
2. validate scientific compatibility;
3. classify discovered inputs as accepted-existing, accepted-new, unready,
   missing, or changed;
4. fail without mutation on missing or changed accepted inputs;
5. determine whether accepted inventory or finalization inputs changed;
6. union accepted-new images and datasets into an in-memory state candidate;
7. compute the new inventory and finalization-input digests;
8. atomically write processing state when it changed;
9. best-effort rebuild the manifest; and
10. derive each current `work_id`, read the terminal-failure journal once, and
    derive the invocation worklist from valid completion markers followed by
    exact matching terminal records and the invocation retry policy.

Reconciliation does not delete completion markers. Their embedded digests make
them nonmatching as soon as the expanded state commits. This removes a needless
crash window: before the atomic state replacement, old state and old markers
remain mutually valid; afterward, the old run marker is stale by construction.
An old aggregate-publication marker may still describe a safe last-published
snapshot, but it cannot establish completion for the expanded run.

## 9. CLI lifecycle

### 9.1 Default behavior

The forward CLI resolves one of these lifecycle actions:

| Condition | Action |
|---|---|
| Output absent or empty and no state | Create fresh state and process all accepted images |
| Compatible state exists | Reconcile additions and continue |
| Compatible state, no incomplete images, valid run marker | Refresh manifest and exit successful no-op |
| Compatible state, all images complete, run marker missing/invalid | Run finalization only |
| Compatible state, incomplete images | Process worklist, then finalize |
| Output non-empty without recoverable state | Fail; require `--overwrite`, `--restart`, or another output |
| Incompatible scientific identity | Fail; require `--restart` or another output |
| Active invocation exists | Report already active; do not mutate inventory or submit work |

### 9.2 Flags

- No lifecycle flag means `auto`.
- `--restart` clears current machine state, completion markers, transient staged
  sidecars, and active lifecycle evidence while preserving user-facing outputs
  according to its existing contract. It preserves the append-only terminal
  failure journal; the restarted computation receives a different `work_id`
  whenever input, pipeline, mode, or relevant processing configuration changed.
- `--overwrite` retains its destructive full-output replacement behavior.
- `--retry-failures` ignores matching terminal-failure records for this
  invocation only. It never deletes, truncates, compacts, or rewrites the
  journal. A retry killed by OOM, timeout, cancellation, or another
  infrastructure failure remains suppressed by the historical terminal record
  until a later explicit retry.

`--restart` and `--overwrite` remain mutually exclusive.

### 9.3 Ordering

An ordinary worklist is ordered:

1. newly accepted images;
2. interrupted and other nonterminal pending images;
3. terminal failures selected by `--retry-failures`;
4. deterministic dataset and filename order within each class.

This prioritizes live experimental updates while preventing a persistently
broken computation from consuming every subsequent invocation.

## 10. Backend selection

### 10.1 Common boundary

After reconciliation, `create_execution_strategy` selects exactly one executor
from the current local, ordinary SLURM, staged local GPU, or staged SLURM
strategies. The worklist is never split between local and SLURM.

All strategies receive:

- the full accepted inventory;
- the immutable current worklist;
- processing generation and inventory digest;
- finalization-input digest;
- scientific-configuration digest;
- invocation ID;
- current execution epoch;
- current mode and canonical per-image processing-configuration digest;
- each item's `work_id` and unique attempt ID; and
- completion-marker writer and validator.

Every invocation first acquires one short common admission lock. While holding
it, admission checks both backend guards: the local lifetime lock and durable
SLURM lifecycle. It then establishes a new execution epoch and the selected
backend's ownership before releasing the admission lock. The shared admission
protocol prevents a local start from racing a SLURM submission even though the
backends otherwise remain independent.

### 10.2 Local

Local execution holds an exclusive run lock from preflight through finalization.
Joblib workers may append events and publish through per-image locks, but another
top-level invocation of either backend cannot be admitted while that lock and
epoch are current.

If the parent process exits or is killed, the operating system releases the
lock. Under the admission lock, the next invocation supersedes the abandoned
local epoch. Any orphan worker from the old parent then fails its locked epoch
check before it can promote staged artifacts. The next invocation classifies
every missing or invalid marker as incomplete. It does not depend on a parent
cleanup handler.

### 10.3 SLURM

SLURM execution persists an immutable batch manifest before scheduler
submission. The lifecycle is first recorded as `submitting`, with the invocation
ID, epoch, deterministic scheduler job name, and scheduler comment. Array
workers read only that manifest and never rescan the live input directory.

The batch manifest binds the invocation ID, execution epoch, processing
generation, inventory digest, finalization-input digest, scientific-config
digest, selected mode/stage, ordered worklist, and full accepted inventory. The
array consumes only the worklist; the terminal observer uses the full inventory
to validate existing plus newly completed markers before aggregate publication.

If the submitter dies after `sbatch` accepts a job but before returned job IDs
are persisted, recovery queries the scheduler by the pre-recorded stable job
name and comment. It adopts exactly one matching job set, refuses takeover when
the result is ambiguous, and submits only when the scheduler proves no matching
job exists. Ordinary SLURM uses a recoverable drip-feed controller and an
`afterany` terminal finalizer so array exceptions, OOMs, and timeouts still
transition the durable lifecycle and allow partial publication. Controller
successors and the finalizer are allowed as separate jobs only when scheduler
dependencies make them strictly sequential with the array they reconcile.
Failed dependency setup leaves recovery for a later invocation; it never
launches an unfenced job beside active array work.

Checkpoint, manifest, monitoring, and any other ancillary work that must run in
parallel with image processing is represented by reserved
`__PHENOTYPIC_<ROLE>__` trigger entries in the assigning array. Routing occurs
before image-path handling. Trigger entries consume indices and count toward
`MaxArraySize`, chunk sizing, and submit limits. The implementation never
submits separate parallel scheduler-sidecar jobs for this work. This restriction
does not apply to Stage-2 `.npy` sidecar files.

The submitting CLI may exit after successful submission. Backend liveness is
therefore represented by the existing SLURM lifecycle and, for staged work, the
existing epoch-fenced orchestration ledger. A later call:

- reports already active when the scheduler or lifecycle proves work active;
- refuses duplicate submission when liveness is indeterminate;
- continues only after the prior lifecycle is terminal or safely recovered.

Ordinary SLURM adopts the same epoch-fencing principle already used by staged
SLURM. It does not need the staged controller's three-phase topology, but stale
workers and finalizers must be unable to publish into a superseding invocation.

### 10.4 Switching backend between invocations

A later invocation may select the other backend only when, under the common
admission protocol:

- the previous invocation is terminal;
- both environments see the same canonical output state;
- both environments can read the accepted input paths or an equivalent mounted
  path recorded through an explicit relocation mechanism; and
- scientific compatibility succeeds.

Backend selection is stored on the invocation and in diagnostic job metadata,
not in the inventory digest.

## 11. Per-image completion contract

### 11.1 Marker schema

```json
{
  "version": 1,
  "work_id": "sha256",
  "processing_generation": "uuid",
  "scientific_config_digest": "sha256",
  "invocation_id": "uuid",
  "execution_epoch": "uuid",
  "backend": "local",
  "dataset": "day_03",
  "image_name": "plate_001.tif",
  "image_sha256": "sha256",
  "mode": "full",
  "required_artifacts": {
    "hdf": {"path": "results/day_03/hdf/plate_001.h5", "sha256": "sha256", "size": 123},
    "measurements": {"path": "results/day_03/measurements/plate_001.parquet", "sha256": "sha256", "size": 123},
    "overlay": {"path": "deliverables/overlays/day_03/plate_001.png", "sha256": "sha256", "size": 123}
  },
  "completed_at": "2026-08-17T12:03:00"
}
```

Paths are output-relative and containment-checked. `overlay` is omitted when its
contract disables it. Configured per-image plot artifacts are included when the
pipeline declares them required.

### 11.2 Publication sequence

For ordinary full processing:

1. validate input identity;
2. run the pipeline and measurements into an invocation-unique staging path;
3. validate every staged required artifact and calculate its SHA-256 and size;
4. revalidate input identity;
5. acquire the per-image publication lock;
6. while holding the lock, check the current processing generation, input
   identity, and local or SLURM execution epoch;
7. atomically promote each staged artifact to its canonical path;
8. atomically write the completion marker containing every promoted artifact
   identity before releasing the lock; and
9. append a completed event best-effort.

A worker never removes or overwrites a canonical marker or artifact before the
locked fence check. A stale worker deletes only its own staging directory. A
crash during locked promotion can leave canonical files without a matching
marker, so the next invocation safely regenerates them; it cannot leave a valid
marker authorizing changed bytes.

Events may precede marker publication only for `started`. A `completed` event is
diagnostic and follows the marker. A failure to append it does not change
completion.

### 11.3 Marker validation

A marker is valid only if:

- JSON and schema parse;
- `work_id` equals the canonical identity recomputed from current accepted
  state;
- dataset, image, mode, processing generation, scientific config, and input
  digest match current state;
- every required path is contained in the output root and exists;
- the measurement parquet is readable and contains the expected image identity;
- the HDF passes the existing HDF validity checks appropriate to the mode;
- every required artifact's mandatory SHA-256 and size match.

Invalid markers are quarantined or ignored, never repaired in place. Their image
is returned to the worklist.

### 11.4 Staged GPU integration

Stage 1 and Stage 2 remain intermediate and content-defined. Stage 3 publishes
the general image completion marker after its current HDF, parquet, overlay, and
strict per-image plot publication sequence succeeds.

Staged `--mode process --layer objmap` publishes the general process-only image
marker after atomically promoting the exported layer and before deleting its
Stage 2 sidecar. It does not rely on Stage 3, which that path never executes.

Existing `stage3_complete` markers are promoted to general markers during schema
migration only after validating their corresponding artifacts and accepted
input identity. The existing legacy migration flag remains diagnostic.

The staged-specific terminal-failure journal receives no new writes after
schema-3 activation. All stages use the general terminal journal and the same
`work_id`. Missing Stage-1 HDFs, missing Stage-2 sidecars, shard-wide failure,
and publication failure remain pending; only caught per-image exceptions inside
the active stage's scientific boundary may append a terminal record.

New staged local and staged SLURM finalizers publish the same general aggregate
and run markers as ordinary execution. `staged_finalization_complete.json`
becomes schema-2 legacy evidence only and receives no new writes after cutover.

### 11.5 Process-only integration

`--mode process --layer <layer>` uses a marker whose required artifact is the
single mirrored layer output. It has no measurement aggregation. After all
image markers validate, it publishes a mode-specific run-completion marker and
then attempts the manifest.

## 12. Events and failures

The event log, scheduler ledger, and `progress/failures.jsonl` remain diagnostic.
The sole terminal-failure authority is:

```text
<output>/.phenotypic/terminal_failures.jsonl
```

### 12.1 Terminal record

Each complete JSON line contains at least schema version, `work_id`, dataset,
normalized input-relative image path, failed processing stage, exception type,
exception message, attempt ID, lifecycle epoch, and UTC timestamp. Optional
traceback and scheduler-task fields are diagnostic. Duplicate records are valid
and readers index them by `work_id`; the journal is never compacted or rewritten.

The writer uses the existing cross-process file lock. While holding the
exclusive lock it seeks to the end, inserts a newline if a killed writer left a
non-newline tail, appends exactly one JSON record and newline, flushes, and calls
`fsync`. It has no unlocked fallback. Lock, write, flush, or `fsync` failure
means this attempt did not establish a terminal outcome and the image remains
pending. Readers take the shared lock, ignore blank or malformed lines, and
continue so a killed append cannot hide later valid records.

### 12.2 Conservative classification

A worker may append a terminal record only for an `Exception` caught inside an
explicit per-image scientific boundary while its lifecycle epoch remains
current and no valid success marker exists. This boundary covers image decoding
and configured pipeline, detector, measurement, or process-only application
code. An apparent runtime bug thrown there is terminal for that exact `work_id`.

The following are infrastructure outcomes and never terminal image failures:

- `MemoryError`, OOM, hard kill, timeout, node loss, preemption, or cancellation;
- stale lifecycle or publication fence and lock failure;
- staging, artifact validation, canonical promotion, or marker-publication
  failure;
- missing staged HDF, objmap sidecar, or other prerequisite;
- aggregate, metadata, post-processing, and finalization failure; and
- any outcome inferred only from `sacct` or other scheduler state.

They are recorded in existing diagnostic attempt or scheduler ledgers when
possible and leave markerless work pending. SLURM controllers never manufacture
terminal records. In staged GPU execution, only a caught per-image scientific
exception is terminal; shard-wide OOM or timeout leaves unresolved images
pending.

### 12.3 Selection and state projection

Work selection uses this order:

1. a valid current `work_id` success marker means successful and skipped;
2. otherwise a matching terminal record means failed and skipped by default;
3. otherwise the image is pending; and
4. `--retry-failures` ignores step 2 for that invocation.

A successful retry writes its ordinary marker and does not alter the journal.
Success therefore supersedes every matching historical failure. If a retry is
killed or suffers infrastructure failure, its old terminal record still
suppresses it on the next default invocation.

For display, categories are disjoint in this order: successful, active current
lifecycle assignment, terminally failed, then pending. Active retry temporarily
hides its historical failure; the failure reappears if that retry ends without
a success marker.

## 13. Manifest semantics

### 13.1 Role

`manifest.json` is an atomic, replaceable cache for humans and GUI polling. It is
rebuilt from processing state, marker validation, the terminal-failure journal,
and current lifecycle assignments. Diagnostic events provide detail only. No
worker, resume planner, finalizer, or completion publisher uses it as scientific
authority.

A manifest write failure is logged and reported when possible, but it does not:

- fail a completed image;
- delete a marker;
- cancel work;
- block aggregate or run-completion publication; or
- require restart.

The next invocation, checkpoint observer, sentinel, or finalizer attempts to
rebuild it.

### 13.2 Schema

The existing fields remain where possible:

```json
{
  "version": 3,
  "execution_mode": "slurm",
  "total_images": 100,
  "successful": 93,
  "terminal_failed": 2,
  "active": 1,
  "pending": 4,
  "remaining": 7,
  "is_complete": false,
  "publication_available": true,
  "published_images": 93,
  "finalization_required": false,
  "discovered_unready": 1,
  "input_path": "experiment",
  "last_updated": "...",
  "last_scanned_at": "...",
  "datasets": {}
}
```

Invariants:

```text
total_images = successful + terminal_failed + active + pending
remaining = terminal_failed + active + pending
is_complete = valid run-completion marker matches current inventory and
              finalization-input digests and references a valid aggregate marker
publication_available = aggregate-publication marker and all declared bytes validate
published_images = source_image_count from the valid aggregate marker, else 0
```

When all images are complete but the run marker is missing,
`finalization_required=true` and `is_complete=false`.

For one compatibility window, `completed`, `failed`, and `started` may be
emitted as exact aliases of `successful`, `terminal_failed`, and `active`.
They are not calculated through a second code path.

### 13.3 Failure-safe transitions

When inputs are added:

1. old run-completion evidence is invalidated;
2. state expands atomically;
3. manifest best-effort moves from complete to incomplete;
4. work begins.

If step 3 fails, the stale manifest cannot establish completion because the GUI
and run classifiers require the matching run-completion marker. The manifest is
repaired on the next safe opportunity.

### 13.4 GUI readers

GUI run classification must not treat `manifest.is_complete` alone as terminal
publication evidence. A completed claim requires the shared completion
inspector to validate state, `run_completion.json`, and its referenced aggregate
publication against current processing state. Aggregate readers use the same
inspector and refuse canonical files if the aggregate marker or any declared
artifact identity is invalid.

Legacy outputs without the new marker retain their existing best-effort
classification path, explicitly labeled as legacy evidence. New schema-3 runs
never fall back from a missing marker to manifest-only completion.

## 14. Finalization

### 14.1 Entry conditions

Aggregate publication starts after the selected backend's invocation becomes
terminal, even when some accepted images failed. This preserves the live-update
goal: one bad image must not prevent successful new measurements from appearing
in the master and mirror.

Publication may be entered:

- immediately after local work becomes terminal;
- by the SLURM dependent finalizer after its work arrays become terminal;
- by a later invocation when every image is complete but run completion is
  missing, in which case it is finalization-only; or
- by explicit recompile logic that intentionally republishes derived outputs.

The source set is every currently valid image completion marker. Failed and
interrupted images are absent from the aggregate and remain visible in progress.
Interrupted and infrastructure-failed images remain eligible automatically;
terminal failures require `--retry-failures` or a different `work_id`. If no
image has completed, the publisher writes no empty authoritative master and
leaves any prior published snapshot untouched.

### 14.2 Source selection

Terminal finalization enumerates accepted images from processing state, validates
their markers, and reads the exact individual measurement parquet declared by
each marker. It does not prefer `_dataset_aggregated.parquet`.

Checkpoint aggregates remain useful for mid-run downloads and can continue to
be rebuilt append-only. They are caches and cannot introduce a row into terminal
outputs without corresponding valid per-image completion evidence.

### 14.3 Startup metadata snapshot

When full mode or recompile configures metadata, the CLI validates and stages a
byte-for-byte copy before local processing or SLURM submission. Under the
lifecycle mutation lock it records the expected digest and canonical path in
state, atomically promotes the staged bytes to
`deliverables/metadata.csv`, and verifies the promoted hash. A copy or
validation failure aborts before work is assigned.

Workers and finalizers use this canonical copy rather than the external source
path. An identical source does not rewrite the copy. A changed metadata digest
invalidates finalization evidence and requires finalization only; it does not
change per-image `work_id`. Existing compatible runs reuse their valid canonical
copy when no new path is supplied. The aggregate and run markers bind its
descriptor whenever metadata is configured. Process-only continues to ignore
metadata and creates no deliverables directory.

The old best-effort metadata copy during final output generation is removed.
Schema migration adopts a valid existing deliverable copy, otherwise it copies
the recorded or newly supplied source. Migration fails safely when configured
metadata has neither a valid copy nor an available source.

### 14.4 Strict core finalization

A new strict core-finalization API returns validated artifact descriptors or
raises. It does not swallow, downgrade, or silently fall back from failures in:

- `deliverables/master_measurements.csv`;
- `deliverables/master_measurements.parquet`;
- `deliverables/measurements.csv`;
- `deliverables/measurements.parquet`;
- the canonical persisted pipeline contract returned by `pipeline_json_path`;
- configured post transforms; or
- the configured metadata join.

The master remains clean and pre-post. The mirror must represent the configured
post and metadata behavior exactly; an unjoined or clean fallback cannot be
authorized as success. Each required descriptor contains output-relative path,
SHA-256, and size.

Per-feature splits, named analysis outputs, plots, HTML reports, README, REMBI,
and QC remain optional publishers. Their failures are isolated, collected as
warnings, and handled consistently by local, ordinary SLURM, and staged
finalizers. They do not cause image reprocessing.

### 14.5 Aggregate-publication marker

Core outputs are first built and validated in an invocation-unique staging
directory. The finalizer then acquires the finalization-publication lock,
revalidates the execution epoch while holding it, moves any prior aggregate
marker into publication history, atomically replaces each canonical core file,
validates the promoted bytes, and writes this marker last:

```json
{
  "version": 1,
  "publication_id": "uuid",
  "processing_generation": "uuid",
  "inventory_digest": "sha256",
  "finalization_input_digest": "sha256",
  "scientific_config_digest": "sha256",
  "source_set_digest": "sha256",
  "source_image_count": 93,
  "finalizer_invocation_id": "uuid",
  "backend": "slurm",
  "execution_epoch": "uuid",
  "required_outputs": {
    "master_csv": {"path": "...", "sha256": "sha256", "size": 123},
    "master_parquet": {"path": "...", "sha256": "sha256", "size": 123},
    "measurements_csv": {"path": "...", "sha256": "sha256", "size": 123},
    "measurements_parquet": {"path": "...", "sha256": "sha256", "size": 123},
    "pipeline": {"path": "...", "sha256": "sha256", "size": 123}
  },
  "warnings": [],
  "published_at": "..."
}
```

`source_set_digest` covers the sorted image-marker identities actually included,
so a partial snapshot is explicit. A crash during canonical promotion leaves no
current valid aggregate marker. Supported GUI, SDK, download, analysis, and
recompile readers must then refuse the mixed files and report publication repair
required. The next finalizer rebuilds them from marker-authorized per-image
sources. Direct third-party reads that ignore publication evidence are outside
this guarantee.

### 14.6 Run-completion marker

When every accepted image has a valid completion marker, the finalizer writes a
small `run_completion.json` after the aggregate marker:

```json
{
  "version": 2,
  "processing_generation": "uuid",
  "inventory_digest": "sha256",
  "finalization_input_digest": "sha256",
  "scientific_config_digest": "sha256",
  "publication_id": "uuid",
  "finalizer_invocation_id": "uuid",
  "execution_epoch": "uuid",
  "gui_record_generation": null,
  "status": "complete",
  "completed_at": "..."
}
```

Validation requires the referenced aggregate marker to be valid and to match
the same inventory and finalization-input digests. Partial publication after an
invocation containing failures never writes `run_completion.json`.

`processing_generation` identifies the scientific run across incremental
invocations. `execution_epoch` fences the one invocation publishing these
markers. `gui_record_generation` optionally carries the exact private GUI launch
generation when the invocation was GUI-owned.

Execution epoch is a publication-time fence and audit field, not durable
scientific identity. A valid marker from a terminal prior invocation remains
valid after that epoch deactivates. Validation does not require it to equal the
next invocation's epoch.

Neither publication marker depends on a successful manifest write. After marker
publication, manifest regeneration is attempted. A later reader reconstructs
aggregate validity and run completion directly from state and markers.

### 14.7 QC and curation

A no-op invocation never calls core or optional finalization, so it cannot clear
review state.

The initial implementation uses one explicit policy when accepted inventory or
finalization inputs change: atomically move `qc/review_state.json`, if present,
to `qc/review_state_history/<timestamp>-<prior-publication-id>.json` before
rebuilding QC, then begin the new publication with no review cursor. The CLI and
manifest warnings report that review progress was archived and reset. If a
later step fails, the archived file remains recoverable.

`curation_labels.parquet`, `custom_categories.json`, and the verified baseline
are preserved and re-keyed through their existing contracts. Incremental
finalization intentionally reseeds an uncurated measurements mirror; the Results
Viewer applies durable curation labels at load time. An already-open GUI detects
the changed publication ID and reloads or displays a stale-session warning
instead of writing review progress against the new snapshot.

## 15. Active-execution safety

### 15.1 Mutual exclusion contract

The product contract says local and SLURM do not execute simultaneously against
one output. PhenoTypic enforces that contract through the common admission lock
and both backend guards. It does not implement a mixed-backend distributed
queue or divide one invocation across backends.

### 15.2 Local guard

An exclusive output run lock is held by the top-level local process through
reconciliation, worker execution, and finalization. SLURM admission checks that
lock while holding the common admission lock. Worker-side publication locks
remain narrower and do not replace it.

### 15.3 SLURM guard

An active SLURM generation remains durable after the submitting CLI exits. Job
metadata and lifecycle state are written before or within the shared admission
and locked submission protocols. Local admission consults this lifecycle and
the scheduler when required. A second call does not overwrite active job
metadata.

Invocation lifecycle is distinct from scientific run completion. Once all
scheduler roles are terminal, the `afterany` observer marks the invocation
`terminal_complete`, `terminal_incomplete`, or `terminal_finalizer_failed`,
records missing image identities, and deactivates its execution epoch even when
`run_completion.json` was not written. The next invocation can then retry
marker-missing images or finalization without treating the old array as active.

### 15.4 Indeterminate state

If lifecycle state is nonterminal and the scheduler cannot prove whether work is
active or terminal, PhenoTypic refuses another invocation of either backend. It
reports the known stable scheduler identity and asks the user to retry after
scheduler visibility returns or use the existing explicit cancel/restart
workflow. Scheduler state is liveness evidence only and never proves image or
run completion.

## 16. Crash-recovery matrix

| Failure | Durable evidence | Next invocation |
|---|---|---|
| Crash before state reconciliation | Old state and marker remain | Re-scan and continue normally |
| Crash after state write but before manifest | Expanded inventory, stale manifest, invalid/missing run marker | Rebuild manifest and process delta |
| Caught per-image scientific exception | Durable terminal record, no image marker | Skip by default; retry only with `--retry-failures` or a different `work_id` |
| Terminal-journal lock/write/flush failure | No committed terminal record | Leave pending and retry automatically |
| Local worker hard kill | Partial/atomic artifacts possible, no marker | Validate/overwrite artifacts and retry |
| Local parent crash | Completed image markers survive; local lock releases | Resume missing markers |
| SLURM task caught per-image scientific exception | Durable terminal record, no marker | Skip by default; explicit retry required |
| SLURM task OOM or timeout | Marker absent; event may be absent; `afterany` observer closes lifecycle | Retry after lifecycle terminal |
| Submitter dies before `sbatch` | Durable `submitting` identity; no scheduler match | Recover submission safely |
| Submitter dies after scheduler acceptance but before job-ID write | Durable `submitting` identity; scheduler match by name/comment | Adopt matching jobs; do not duplicate |
| SLURM submission process exits normally | Active lifecycle and batch manifest | Report active; do not resubmit |
| SLURM controller or dispatcher loss | Ledger, scheduler state, completed markers | Existing recovery controller or later invocation resumes |
| Stale worker reaches publication after restart | Locked epoch check fails before canonical promotion | Delete only its staging data; preserve newer artifacts |
| Manifest generation bug | State and markers remain authoritative | Log warning and regenerate later |
| Terminal aggregate cache stale | Individual marker-authorized parquets remain | Ignore cache during finalization |
| Finalizer exception before entering locked core promotion | Prior aggregate marker and canonical snapshot remain | Retry publication only when all images are complete, or after remaining work otherwise |
| Finalizer exception after marker retirement or during core promotion | No valid current aggregate marker; canonical files may be old or mixed | Supported readers refuse; rebuild and republish idempotently |
| Optional plot/QC/report failure | Aggregate marker may publish with warnings | Do not reprocess images; allow explicit repair/recompile |

## 17. Mode interactions

### 17.1 Full mode

Receives the complete behavior specified here.

### 17.2 Process mode

Uses auto-continue and mode-specific image markers. It writes no measurement
master and therefore publishes a mode-specific run marker after all
exported-layer markers validate, then attempts the manifest. It does not use an
aggregate-publication marker because it has no aggregate contract.

Changing `--layer` is scientifically incompatible with the saved process run.

### 17.3 Measure mode

The initial implementation rejects `--mode measure` when schema-3 processing
state exists because measure rewrites the per-image parquets authorized by image
markers. Legacy one-shot behavior remains available only on outputs without
schema-3 authority. A future marker-aware measure design must fence the whole
rewrite, republish affected image markers, and republish aggregate and run
markers; silently retaining old markers is forbidden. Existing lifecycle flags
remain invalid in this mode (`src/phenotypic/phenotypicCLI.py:1150-1187`).

### 17.4 Recompile mode

Remains explicit finalization from existing outputs. It must use the same
marker-authorized source selection for schema-3 runs so it cannot publish orphan
or stale per-image parquets. Both local and SLURM recompile use the
`terminal_authorized` discovery policy, never the `checkpoint_cache` policy.
When recompile changes external finalization inputs, it updates their digest
under the output mutation lock before publication.

### 17.5 Sample mode

`--sample` is not a live-ingestion feature. A sampled run persists
`cohort_policy="sampled"`, requested size, random seed, and the exact selected
identities. Auto-continue reconciles only those identities and ignores later
non-cohort discoveries. A continuation that omits or changes the saved sample
semantics fails and requests `--restart` or another output directory. An
unseeded first selection persists the generated seed so later behavior is
deterministic.

### 17.6 Dry run

Dry run reports whether the invocation would create, continue, finalize-only,
or no-op, plus counts for new, incomplete, failed, unready, missing, and changed
images. It does not remove markers or write state.

## 18. Migration

### 18.1 Schema-2 state

On first continuation of a schema-2 run:

1. load and event-aggregate the legacy state;
2. rescan the exact recorded input path;
3. require every legacy `initial_images` entry to remain present;
4. fingerprint those inputs;
5. reconstruct the saved mode-specific required-artifact contract from schema-2
   config and the persisted pipeline, including process-only layer, overlay
   policy, and configured strict image plots;
6. for every legacy completed image, validate every required artifact for that
   reconstructed contract and calculate its identity;
7. promote safely validated completions to general markers with
   `legacy_migration=true`;
8. leave failed and ambiguous images incomplete;
9. promote valid existing Stage 3 markers only through the same contract;
10. examine legacy `DatasetState.failed`, events, `progress/failures.jsonl`, and
    staged terminal journals only as migration evidence; append a terminal
    record only when a row already carries the complete exact identity,
    attempt/epoch/stage fields, and explicit caught-scientific classification,
    and otherwise retain it as diagnostics;
11. adopt or create the startup metadata snapshot when metadata was configured;
12. atomically write schema-3 state; and
13. rebuild the manifest.

Migration fails without modifying state if pipeline compatibility, input
presence, artifact identity, or stem uniqueness cannot be established.

### 18.2 Legacy run completion

An existing completion marker without an inventory digest is not copied blindly.
If all migrated image markers validate, the migrator runs finalization or
validates the required authoritative outputs and then writes new aggregate and
schema-3 run markers carrying the new digests. A legacy
`staged_finalization_complete.json` may support migration diagnostics, but it
never becomes current completion evidence by itself.

### 18.3 Legacy layout

The existing machine-state relocation from root-level `processing_state.json`
and `progress/` remains before schema migration. New writers use only the hidden
`.phenotypic/` paths.

## 19. Required implementation boundaries

The implementation introduces focused helpers rather than expanding
`phenotypicCLI.py` further:

- `_cli_inventory.py`: input fingerprint model, reconciliation, inventory
  digest, schema migration, worklist ordering.
- `_cli_completion.py`: image, aggregate, and run marker paths, schemas,
  validation, publication, legacy promotion, and the shared completion
  inspector.
- `_cli_failure_tracker.py`: canonical `work_id`, durable terminal-journal
  append/read, conservative classification, and legacy diagnostic projection.
- `_cli_lifecycle.py`: auto/create/restart decision and
  finalization-only/no-op classification.
- Existing backend strategies: consume immutable worklists and publish through
  the shared completion helper.
- `_measurement_sources.py`: marker-authorized terminal source discovery while
  retaining a distinct cache discovery policy for checkpoint use.
- `_cli_output_manager.py`: strict core-finalization entry point, isolated
  optional publishers, and aggregate marker-last publication.
- `_manifest_builder.py`: derive counts from inventory, markers, the terminal
  journal, and current lifecycle assignments; never authorize completion.
- `sdk_/_io_constants.py`: canonical paths and serialized key constants.

Exact module names may change during implementation, but authority boundaries do
not.

### 19.1 Completion-consumer migration

The shared completion inspector replaces manifest-only or legacy-marker
authority in every current consumer:

| Consumer | Required change |
|---|---|
| Ordinary SLURM checkpoint handler | Publish completion from state, image markers, strict core outputs, and aggregate marker; manifest is best-effort afterward |
| SLURM sentinel | Trigger terminal aggregation from scheduler/lifecycle terminality, not manifest arithmetic |
| Local GUI lifecycle publisher | Publish from the shared inspector without requiring a manifest |
| Run Console SLURM observer | Derive terminal status from lifecycle plus shared completion inspection |
| Results Viewer output-consistency check | Require a valid aggregate marker before loading core outputs; require the run marker only for all-complete status |
| GUI recent-runs registry | Use shared completion inspection; treat manifest fields as display cache only |

Both local and SLURM recompile also use the inspector's
`terminal_authorized` source policy. No schema-3 consumer may promote a
manifest-only or `staged_finalization_complete.json`-only success.

## 20. Verification

### 20.1 Unit tests

- Auto lifecycle decision for absent, empty, compatible, incompatible,
  finalization-only, no-op, and active outputs.
- Inventory reconciliation for additions to an existing dataset.
- Inventory reconciliation for a new dataset.
- Missing accepted input rejection.
- Changed accepted input rejection.
- Unready file exclusion and later acceptance.
- Same-stem collision rejection.
- Canonical inventory digest stability across ordering and platforms.
- Image-marker validation for missing, malformed, stale-generation,
  stale-configuration, wrong-input, digest/size mismatch, unreadable-parquet,
  and invalid-HDF cases.
- Aggregate-marker validation for every required artifact, partial source-set
  digest, and changed finalization inputs.
- Run-marker validation for stale inventory, stale finalization inputs, and
  mismatched publication ID; publication-time rejection of stale execution
  epochs.
- Canonical `work_id` changes for input, pipeline, mode, and relevant processing
  configuration, but remains stable across backend and finalization-only changes.
- Worklist suppression of matching terminal failures, explicit retry selection,
  and ordering after new and ordinary pending images.
- Durable concurrent journal appends, duplicate rows, malformed/partial rows,
  killed append recovery, and lock/write/flush/`fsync` failure.
- Success-marker precedence over matching historical failures.
- Conservative exception classification, including `MemoryError` and every
  infrastructure/publication exclusion.
- Manifest arithmetic and run-marker-derived `is_complete`.
- Schema-2 and Stage-3-marker migration.
- Mode-specific schema-2 migration for full, every process-only layer, staged
  full, overlay-disabled, and strict image-plot contracts.
- Sample policy persistence for seeded, initially unseeded, omitted-flag,
  changed-size, new-file, and new-dataset cases.
- No-op path does not call finalization or QC reset.

### 20.2 Local integration tests

- First ordinary call processes all images.
- Identical second call is a no-op and preserves deliverable mtimes and QC review
  state.
- Adding one image processes only that image and republishes a master containing
  old plus new measurements.
- Adding a dataset republishes a master containing every prior dataset.
- A caught per-image scientific exception writes a terminal record and is
  skipped by the next default call.
- `--retry-failures` selects the terminal computation, and a successful retry
  marker supersedes its retained historical failure.
- An explicit retry killed or OOM-terminated retains the old suppression.
- A subprocess killed between each artifact-publication step recovers without
  claiming completion.
- A stale orphan worker cannot alter canonical artifacts or markers after its
  epoch is superseded.
- Replacing any marker-authorized artifact after marker publication invalidates
  the marker and excludes it from terminal aggregation.
- A parent killed after some workers complete resumes only missing markers.
- A finalizer killed before and after each authoritative write resumes
  finalization only.
- Soft failures in every required write, post transform, metadata join, and
  pipeline persistence withhold aggregate and run markers; optional publisher
  failures produce warnings instead.
- Changed metadata, study, and `no_qc` inputs republish without reprocessing
  images.
- Manifest writer failure does not fail completed image work or core
  finalization.
- Process-only auto-continue follows its mode-specific contract.
- Measure mode rejects schema-3 outputs before rewriting any parquet.

### 20.3 Ordinary SLURM integration tests

- Submitted array uses a frozen batch manifest while later input additions are
  ignored until the next invocation.
- Second invocation during active jobs reports active and submits nothing.
- A caught per-image scientific exception is terminal; OOM-like hard exit,
  timeout-like hard exit, node loss, cancellation, and missing events remain
  pending and the invocation lifecycle becomes terminal-incomplete.
- Scheduler reconciliation never creates a terminal record from `sacct` state.
- Kill points before `sbatch`, after scheduler acceptance, and before job-ID
  persistence recover by stable scheduler identity without duplicate arrays.
- A terminal array with missing markers publishes successful images without a
  run-completion marker; the next invocation resubmits only missing images.
- Stale-generation worker and finalizer publications are rejected before they
  modify canonical artifacts.
- Failed dependent finalizer leads to finalization-only recovery.
- Switching a terminal run from local to SLURM and from SLURM to local preserves
  completed evidence when paths are shared.

### 20.4 Staged GPU regression tests

- Stage 1, Stage 2, Stage 3 content-defined resume remains unchanged.
- General markers are written only after successful Stage 3 publication.
- Existing Stage 3 markers promote safely.
- Staged process-only objmap export writes a general marker before sidecar
  deletion.
- New images enter Stage 1 while completed old images remain skipped.
- Staged local and staged SLURM produce the same general marker schema.
- Existing epoch fencing, controller recovery, shard behavior, and sidecar
  cleanup remain intact.
- Per-image caught Stage-1, Stage-2, and Stage-3 scientific exceptions use the
  general terminal journal; missing prerequisites and shard-wide OOM or timeout
  remain pending.
- Legacy staged finalization is readable only for migration; new staged runs
  publish general aggregate and run markers.

### 20.5 GUI and output tests

- Schema-3 run is never classified complete from manifest alone.
- Manifest `successful`, `terminal_failed`, `active`, and `pending` counts are
  reconstructed after journal corruption or manifest-write failure without
  overlap.
- Checkpoint handler, sentinel, local GUI publisher, Run Console observer,
  Results Viewer consistency, and recent-runs registry all remain correct when
  manifest writes fail.
- Removing or invalidating the run marker reopens the run even when an old
  manifest says complete.
- A stale manifest plus a valid matching run marker rehydrates as complete and
  is repaired on refresh.
- Added inventory changes the digest and invalidates old completion evidence.
- Terminal aggregation ignores an unmarked orphan parquet and a stale dataset
  aggregate.
- Required master and mirror outputs contain all marker-authorized datasets.
- A crash-created mixture of core files has no valid aggregate marker and every
  supported reader refuses it until repair.
- Optional finalization failures appear as warnings without reprocessing images.
- QC review state is archived and reset on data change; curation labels, custom
  categories, and verified baseline survive and stale GUI sessions cannot write
  against the new publication.

### 20.6 Failure injection standard

Every load-bearing atomic sequence has tests that raise or hard-exit immediately
before and after publication. Assertions are made from a fresh process using
only disk state. An in-process object retained across the injected failure is
not acceptable recovery evidence.

## 21. Acceptance criteria

The change is complete only when all statements below hold:

1. The same ordinary full command can be run repeatedly without a continuation flag.
2. The first invocation starts fresh; later compatible invocations continue.
3. New stable images and datasets are accepted without permitting mutation of
   accepted inputs.
4. An unchanged invocation exits zero without rewriting final deliverables or
   QC review progress.
5. Local and SLURM select one independent backend per invocation, never share an
   active worklist, and cannot execute simultaneously against one output.
6. Every completed image has marker-backed, content-bound evidence written last.
7. Missing markers after infrastructure failure, including OOM or timeout with
   no event, are retried automatically. Exact matching terminal scientific
   failures are skipped unless `--retry-failures` is supplied.
8. A finalizer crash never causes completed images to rerun. It is repaired by
   finalization-only when all images are complete, or after retrying remaining
   images when they are not.
9. Manifest generation failure cannot invalidate scientific completion or make
   incomplete work complete.
10. `manifest.is_complete` requires a matching run-completion marker for
    schema-3 runs.
11. A valid aggregate-publication marker authorizes terminal master and mirror
    outputs containing every and only its declared marker-authorized accepted
    images, independent of checkpoint aggregates.
12. Active local or SLURM work cannot be duplicated by another invocation.
13. Stale-generation workers and finalizers cannot modify canonical artifacts or
    publish terminal evidence.
14. Existing schema-2 and staged-GPU outputs either migrate with validated
    evidence or fail safely without state mutation.
15. Local, ordinary SLURM, staged local GPU, and staged GPU SLURM pass the same
    completion and crash-recovery conformance suite.
16. A terminal-incomplete SLURM invocation releases backend ownership so a later
    invocation can retry missing markers without manual state repair.
17. Required finalization failures cannot authorize fallback content, and every
    supported aggregate reader refuses unmarked or hash-mismatched core files.
18. `.phenotypic/terminal_failures.jsonl` is the only terminal-failure
    authority, is never rewritten, and loses no complete concurrent append.
19. A valid success marker supersedes every historical failure for its
    `work_id`; a changed admitted computation is never suppressed by an older
    identity.
20. A run with terminal failures may publish marker-authorized successes but
    remains `terminal_incomplete` and has no run-completion marker.
21. Configured metadata is durably copied to `deliverables/metadata.csv` before
    local work or scheduler submission, and downstream work uses that copy.

## 22. Implementation sequence

1. Add schema-3 inventory, canonical `work_id`, and digest primitives with
   migration tests.
2. Add the durable terminal-failure journal and conservative classifier using
   the existing cross-process lock.
3. Add general image, aggregate-publication, and run markers with the shared
   completion inspector and validation tests.
4. Change ordinary local workers to staged, locked, fence-before-promotion
   publication.
5. Reconcile staged Stage 3 markers and all staged terminal outcomes into the
   general contracts.
6. Add default auto lifecycle, explicit retry, no-op, and finalization-only paths.
7. Add the startup metadata snapshot and strict marker-authorized core
   finalization.
8. Migrate every manifest-dependent completion consumer to the shared inspector.
9. Integrate ordinary SLURM immutable batches, ambiguous-submission recovery,
   terminal-incomplete lifecycle, and epoch fences.
10. Apply process-mode and recompile behavior.
11. Complete failure injection, cross-backend sequential continuation, and live
   acquisition end-to-end tests.

## 23. Explicit non-guarantees

- PhenoTypic does not observe new files while no invocation or future watcher is
  running. A cron job or acquisition system may invoke the same command
  repeatedly for near-live operation.
- New inputs are not merged into an active SLURM batch. They wait for the next
  invocation.
- A user who bypasses the CLI and manually edits machine state or completion
  markers can invalidate guarantees.
- A third-party process that reads canonical aggregate files without validating
  `aggregate_publication.json` can observe files during a failed promotion;
  supported PhenoTypic readers refuse such a snapshot.
- Default stat-based rechecks do not detect adversarial content replacement that
  preserves size and `mtime_ns` exactly.
- Local-to-SLURM continuation cannot work when the environments do not share or
  deliberately transfer the same canonical state and artifacts.
