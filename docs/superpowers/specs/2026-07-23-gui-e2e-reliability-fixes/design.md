# GUI E2E Reliability and Compatibility Fixes

Date: 2026-07-23

## Status

Draft implementation specification based on the remote end-to-end GUI audit
performed on 2026-07-22.

This document specifies production and test changes. It does not itself change
runtime behavior.

Audit environment:

- Deployed repository commit:
  `79d0b879d7985710abd358fadcff829488e05b43`.
- Local specification commit:
  `6a50857174a80c54baf07291615ff2aae2948cf5`.
- The only local commit after the deployed revision fixes
  `OperationFailedError` pickling; it does not change the GUI findings in this
  specification.
- The active remote output `data/results/2026-07-16` was not selected or
  modified during the audit.
- Pipeline and SLURM tests used isolated one-image outputs under
  `gui_e2e_runs/2026-07-22_codex/`.

## 1. Summary

The audit found that most GUI surfaces render and their primary navigation
works, but several workflows do not preserve the state or safety guarantees
that their UI communicates:

- Run Console can make a fresh output non-fresh before the CLI starts.
- Run Console actions can consume a stale derived form store instead of the
  controls visible when the user clicks Run.
- Local and SLURM lifecycle state is optimistic, active-row-only, and can
  remain `running` or `submitting` after terminal failure.
- Ordinary CPU SLURM cancellation does not fence later dispatcher submissions.
- Staged SLURM submission can be reported as failed even after a live
  controller was submitted.
- Builder preview data can exist in cache while a remounted inspector shows
  the empty placeholder.
- Tune Setup visually unlocks static placeholders and can let stale shell
  handoff state override a newly typed pipeline path.
- Opening Results can migrate source files or create source-local caches,
  despite appearing to be a read operation.
- Historical QC recipes fail late instead of receiving a compatibility
  preflight, while legacy QC parquet outputs require a deliberate rebuild.
- Error-tab activation can overwrite canonical deliverables from a frozen
  viewer snapshot.
- Reader-facing commands, accepted image extensions, and tutorials have
  drifted from the executable contracts.

The recommended architecture has four rules:

1. A stable server-side run record is authoritative for execution lifecycle.
   Browser stores are presentation state, not execution authority.
2. Loading, binding, browsing, or switching tabs is read-only. Any migration,
   publication, or durable cache write requires an explicit action.
3. Compatibility checks are pure and complete before a write begins.
   Migrations are explicit, backed up, atomic, and idempotent.
4. A displayed or copied command is rendered from the same validated token
   list used for deployment.

## 2. Issue inventory and disposition

| ID | Priority | Finding | Disposition |
| --- | --- | --- | --- |
| GUI-RUN-001 | P0 | `.gui_log` makes a fresh output fail the CLI nonempty-output guard | Fix in Phase 1 |
| GUI-RUN-002 | P0 | Run/Validate can read stale `RC_STORE_FORM_STATE` and choose the wrong mode | Fix in Phase 1 |
| GUI-RUN-003 | P0 | Pending SLURM futures depend on the currently active browser record | Fix in Phase 1 |
| GUI-RUN-004 | P0 | Local exits do not produce terminal registry state | Fix in Phase 1 |
| GUI-SLURM-001 | P0 | Ordinary SLURM cancellation does not fence or cancel the complete job chain | Fix in Phase 2 |
| GUI-SLURM-002 | P1 | SLURM status and logs do not observe scheduler or durable output state | Fix in Phase 2 |
| GUI-SLURM-003 | P1 | Time field says `HH:MM:SS`, while CLI currently requires integer minutes | Fix in Phase 1 |
| GUI-SLURM-004 | P1 | Staged GPU controls and controller job-id handling are incomplete | Fix in Phase 2 |
| GUI-BUILDER-001 | P1 | Inspector remount can erase a valid cached preview | Fix in Phase 3 |
| GUI-TUNE-001 | P1 | Setup unlocks static Search space and Scorer placeholders | Fix in Phase 3 |
| GUI-TUNE-002 | P1 | Stale shell pipeline handoff can override explicit typed input | Fix in Phase 3 |
| GUI-TUNE-003 | P1 | Command preview can diverge from deploy argv or expose storage credentials | Fix in Phase 3 |
| GUI-RESULTS-001 | P0 | Binding or viewing can mutate the selected source output | Fix in Phase 4 |
| GUI-RESULTS-002 | P1 | Results and Analysis retain a frozen snapshot without an explicit refresh contract | Fix in Phase 4 |
| GUI-QC-001 | P1 | Legacy QC recipe fields fail late and warnings can be delayed or duplicated | Fix in Phase 4 |
| GUI-QC-002 | P1 | Legacy QC parquet artifacts require a guided `qc.duckdb` rebuild | Fix in Phase 4 |
| GUI-ERROR-001 | P0 | Opening Error can publish partial, stale canonical deliverables | Fix in Phase 4 |
| GUI-ANALYSIS-001 | P1 | Saving after a tolerant load can silently drop unknown analyzer nodes | Fix in Phase 4 |
| GUI-PATH-001 | P2 | Sidebar, pickers, and localStorage labels can show stale path state | Fix in Phase 5 |
| GUI-PATH-002 | P2 | Run and Tune image extensions drift from canonical `IMAGE_EXTS` | Fix in Phase 5 |
| GUI-DOCS-001 | P2 | Tutorials and displayed commands describe retired controls or unsupported runners | Fix in Phase 5 |

The first full-resolution SLURM smoke job reached the cluster and ran on a
compute node. Its image task was OOM-killed under a deliberately low 4 GB
allocation. That resource outcome is not a product defect. The GUI lifecycle
and observability failures around it are in scope.

## 3. Goals

### 3.1 Execution correctness

- The mode and fields visible at click time determine the exact command.
- Selecting SLURM can never silently execute a local pipeline.
- Every launched or submitted run reaches a durable terminal or explicitly
  unknown state.
- Cancellation prevents future scheduler submissions for the cancelled run.
- A run has one stable identity from pre-submit through terminal state.

### 3.2 Data safety

- Merely binding an output, opening a tab, requesting a tile, or refreshing a
  page does not alter the selected output tree.
- Compatibility migrations and derived-output publication are explicit.
- Stale snapshots cannot publish canonical artifacts.
- Unsupported serialized nodes are preserved or block saving; they are never
  silently dropped.

### 3.3 Usable compatibility

- Historical recipes are classified as compatible, safely migratable, or
  blocked before instantiation.
- Intentional format cutovers remain cutovers. The GUI provides an explicit
  rebuild when a lossless reader migration is not available.
- Source and metadata labels show the currently resolved path or a visible
  unavailable state.

### 3.4 Verifiable UI contracts

- Builder and Tune browser tests execute the complete workflows, not only
  inspect callback registration.
- Displayed commands match deploy tokens modulo intentional secret redaction.
- All image pickers consume the canonical extension set.

## 4. Non-goals

- Do not make the GUI a filesystem watcher for the entire sandbox.
- Do not make the Builder preview cache multi-process in this change. The
  supported launcher remains a single GUI process. Multi-worker cache storage
  is a separate deployment feature.
- Do not add SLURM replica packing. `gpu_workers_per_gpu` remains internal at
  its implemented value of `1`.
- Do not infer a new scientific meaning for ambiguous historical QC fields.
- Do not synthesize `qc.duckdb` by guessing a mapping from the retired flat
  QC parquets. Rebuild it from the validated recipe and measurements.
- Do not auto-refresh active Results while a user is curating.
- Do not change pipeline algorithms or numerical defaults.

## 5. System-wide invariants

These invariants are acceptance requirements, not implementation suggestions.

### 5.1 Visible-state invariant

Run, Validate, Save Preset, Tune Continue, and Tune Deploy construct their
request from the raw controls captured as Dash `State` in that same callback.
A previously computed aggregate browser store is not an action authority.

### 5.2 Stable-identity invariant

The canonical run id is the sandbox-relative resolved output path plus a
durable UUID launch generation. The generation is persisted with machine state
before execution so GUI restart and registry rehydration cannot reuse an old
identity. Local PID, transient submission token, SLURM job ids, and staged
controller ids are attributes of that record. They are not replacement record
ids.

### 5.3 Terminal-state invariant

`complete`, `failed`, and `cancelled` require terminal evidence. `running`
requires a live local process, active scheduler work, or durable nonterminal
machine state. A deactivated cancellation fence moves a run to `cancelling`;
it becomes `cancelled` only after every recovered scheduler id is confirmed
inactive. If the scheduler cannot be queried and no durable terminal evidence
exists, status is `unknown`, not `running`, `failed`, or `cancelled`.

### 5.4 No-silent-fallback invariant

Any SLURM request must contain at least one validated `--slurm key=value`
token. `submit_slurm` rejects `mode != "slurm"` and rejects an empty SLURM
profile. It never invokes the CLI and hopes the CLI will infer the target.

### 5.5 Read-only-view invariant

The following actions produce no writes under the bound source:

- `OutputRoot.discover`;
- Results or Analysis app construction;
- opening any Results tab;
- first tile or preview request;
- refreshing the browser;
- compatibility preflight.

External GUI cache writes under the configured GUI cache root are allowed.

### 5.6 Explicit-write invariant

Migration, QC rebuild, curation save, analysis save, and error publication:

- name every target before starting;
- validate all inputs first;
- capture source fingerprints;
- write to temporary generation paths;
- atomically publish;
- preserve the old generation until success;
- record a backup or migration receipt when source configuration changes.

### 5.7 Command-parity invariant

Command preview and deployment use one authoritative token object. The preview
may replace a secret token with an environment-variable reference, but all
non-secret tokens and their ordering are identical.

## 6. Architecture decision

### 6.1 Chosen approach

Use a stable server-side lifecycle registry for execution, pure preflight
objects for compatibility, and explicit command objects for launch.

This preserves the existing Dash apps and CLI entry points while moving
authority out of asynchronous derived stores and tab callbacks.

### 6.2 Rejected approach: patch each observed callback

Ignoring `.gui_log`, adding a longer polling interval, and forcing preview
rerenders would address individual symptoms but leave:

- stale form-state races;
- stranded futures after navigation;
- multiple identities for one run;
- no cancellation fence;
- view-time source mutation;
- preview/deploy command divergence.

The audit reproduced combinations of these failures, so isolated callback
patches are not sufficient.

### 6.3 Rejected approach: automatic migration and automatic refresh

Automatic conversion on bind would preserve the current convenience but
continue making read operations destructive. Automatic Results refresh would
also swap data underneath curation and error analysis. Both behaviors violate
the data-safety goals.

## 7. Phase 1: Run Console launch correctness

### 7.1 Build action state directly from controls

Add one pure helper:

```python
def state_from_controls(
    *,
    pipeline_path: object,
    input_dir: object,
    output_dir: object,
    mode: object,
    flags: object,
    sample: object,
    nrows: object,
    ncols: object,
    image_type: object,
    workers: object,
    log_level: object,
    slurm_partition: object,
    slurm_time: object,
    slurm_mem: object,
    slurm_cpus: object,
    slurm_gpus: object,
    slurm_extra: object,
    metadata_payload: object,
    sandbox: SandboxRoot,
) -> RunConsoleState:
    ...
```

Run, Validate, and Save Preset take every field as callback `State` and call
this helper. `RC_STORE_FORM_STATE` may remain as a presentation or preset
preview cache, but actions do not read it.

Additional guards:

- Resolve all paths through `SandboxRoot`.
- Reject two nonterminal generations targeting the same output.
- Register the run record before starting a process or submission future.
- Atomically create a generation-owner record under `.phenotypic` before
  launch. It contains the GUI launch UUID, command digest, mode, creation time,
  and lifecycle epoch.
- Validate any existing nonterminal machine state, including work started by a
  different GUI or CLI process, before claiming the output. Resume may attach
  only when the stored owner, pipeline identity, and requested resume policy
  are compatible.
- Require every worker, finalizer, and publication marker to name the same
  generation or orchestration epoch. Evidence from an older generation cannot
  complete the new record.
- A SLURM state must contain a nonempty validated SLURM profile.
- Preserve the final visible mode during rapid toggle and click sequences.

### 7.2 Reconcile GUI logs with the fresh-output contract

Move `RUN_LOG_DIRNAME` and `STDOUT_LOG` to the canonical SDK I/O constants and
re-export them through `gui._config`.

The CLI fresh-output guard ignores exactly the reserved GUI log directory when
determining whether the output is fresh. The exception applies only when
`.gui_log` is a real, non-symlink directory containing allowlisted GUI log
files:

```python
def is_safe_gui_log_entry(entry: Path) -> bool:
    return (
        entry.name == RUN_LOG_DIRNAME
        and entry.is_dir()
        and not entry.is_symlink()
        and all(
            child.is_file()
            and not child.is_symlink()
            and child.name in GUI_LOG_FILENAMES
            for child in entry.iterdir()
        )
    )
```

It does not ignore arbitrary dot-directories, `.phenotypic`, existing
deliverables, results, or unrelated files.

Required behavior:

- an output containing only `.gui_log/stdout.log` is fresh;
- an output containing `.gui_log` plus any other entry is non-fresh;
- a symlinked `.gui_log`, nested directory, or unrecognized file is
  non-fresh;
- Validate followed by Run can reuse the reserved log directory;
- rerunning a completed output without resume, restart, or overwrite remains
  rejected.

### 7.3 Stable run records and generation fencing

Extend `RunRecord` with:

```python
@dataclass
class RunRecord:
    run_id: str
    generation: UUID
    mode: Literal["validate", "local", "slurm"]
    output_dir: Path
    rel_path: str
    status: RunStatus
    pid: int | None = None
    scheduler_ids: tuple[str, ...] = ()
    primary_scheduler_id: str | None = None
    log_paths: tuple[Path, ...] = ()
    submitted_at: datetime | None = None
    terminal_at: datetime | None = None
    returncode: int | None = None
    status_detail: str | None = None
```

The registry allocates and persists a new UUID generation before launch. An
older future or process-exit callback may update the record only if its
generation still matches. Rehydration reads the persisted generation rather
than deriving a new integer from process memory.

SLURM submission futures receive a completion callback immediately after they
are created. The callback:

1. removes the future from the pending map;
2. checks the record generation;
3. updates the stable record to `running` or `failed`;
4. records scheduler ids, logs, and failure detail;
5. bumps a thread-safe registry revision.

Completion does not depend on `RC_STORE_ACTIVE_RUN_ID`, the Run page being
mounted, or a browser interval firing.

Failed submission records remain visible. They are not removed with their
diagnostic evidence.

Cancellation is defined during `submitting`, not only after the first job id is
known:

- create and persist the inactive/active epoch before starting the submitter;
- Cancel first deactivates that epoch;
- request cancellation or termination of the pending submitter process;
- if submission returns job ids after cancellation, immediately ledger and
  cancel them;
- a submitter must recheck the epoch immediately before every `sbatch`.

A submitter timeout, killed GUI process, or exception after `sbatch` is not
immediately classified as a clean failure. Recovery first reads incomplete
submission intents, versioned metadata, the append-only ledger, and scheduler
comments containing the launch generation. Any recovered ids are attached to
the record or cancelled according to the epoch fence. The record becomes
`failed` only after this reconciliation proves that no submitted work remains.

### 7.4 Local process terminal observation

Extend `LocalRunner.start` with an exit callback or a registered lifecycle
observer. The registration order is:

1. allocate stable record and generation;
2. create and validate the log directory;
3. reserve runner id;
4. start the subprocess;
5. attach PID and log path;
6. start the exit observer;
7. on exit, update the matching generation.

Every launch boundary has a terminal cleanup path:

- log-directory or `Popen` failure marks the matching record `failed`;
- exit-observer startup failure terminates and reaps the child before marking
  the record `failed`;
- an exception after a PID is attached never leaves an unobserved child;
- all partial handles and reservations are released without deleting the
  diagnostic record or log.

Exit mapping:

- return code `0` -> `complete`;
- nonzero return code -> `failed`;
- a record already marked `cancelled` remains `cancelled`;
- missing handle plus no terminal evidence -> `unknown`.

Keep the finished handle and ring buffer available until the user starts a new
generation for that output or the retention limit evicts it.

### 7.5 Time-limit parser

Add a shared parser in the SLURM SDK that accepts:

- positive integer minutes, for example `10`;
- `HH:MM:SS`, for example `00:10:00`;
- `D-HH:MM:SS`, for example `1-04:00:00`;
- empty input as unset.

It returns a canonical SLURM time string or `None`. Invalid or nonpositive
values fail before any subprocess or scheduler command.

Relabel the field to:

> Time limit
>
> Minutes or SLURM duration (`HH:MM:SS`, `D-HH:MM:SS`)

CLI and GUI validation call the same parser.

### 7.6 Phase 1 files

Files to modify:

- `src/phenotypic/gui/run_console/_callbacks.py`
- `src/phenotypic/gui/run_console/_runner.py`
- `src/phenotypic/gui/run_console/_state.py`
- `src/phenotypic/gui/run_console/_form.py`
- `src/phenotypic/gui/run_console/_ids.py`
- `src/phenotypic/gui/shell/_runs_registry.py`
- `src/phenotypic/gui/_config.py`
- `src/phenotypic/sdk_/_io_constants.py`
- `src/phenotypic/sdk_/slurm/_sbatch.py`
- `src/phenotypic/phenotypicCLI.py`

Tests to add or extend:

- `tests/unit/gui/run_console/test_runner.py`
- `tests/unit/gui/run_console/test_state.py`
- `tests/unit/gui/run_console/test_slurm.py`
- `tests/unit/gui/shell/test_runs_registry.py`
- `tests/integration/gui/test_run_console_callbacks.py`
- `tests/e2e/gui/test_run_console.py`
- a focused SLURM time-parser unit test under `tests/unit/sdk_/`.

## 8. Phase 2: Complete SLURM lifecycle

### 8.1 Generalize the append-only job ledger

The ordinary CPU dispatcher must use the same two safety concepts as staged
GPU orchestration:

- an append-only record of every submitted job id and role;
- an epoch-active fence checked before any later submission.

The ordinary dispatcher becomes a small Python entry point that:

1. acquires the generation's submit/cancel coordination lock;
2. checks that the generation epoch is active;
3. records a uniquely identified submission intent;
4. invokes `sbatch`;
5. records the returned job id and role before releasing the lock;
6. mirrors all ids into `job_metadata[slurm_job_ids]`.

The intent is durable before `sbatch` and records a generation-specific
scheduler comment. On startup and after any submitter timeout or abnormal exit,
the reconciler resolves each incomplete intent by querying that comment,
checking metadata and the ledger, then either attaching the recovered job id
or cancelling it when the epoch is inactive. This applies to the initial
array, every chunk, staged controllers, recovery jobs, and finalizers.

Cancellation acquires the same coordination lock, writes the inactive fence,
then runs `scancel` for every active ledgered id. It reconciles incomplete
submission intents and rescans scheduler jobs by generation comment until no
in-flight intent can produce an unledgered job. A continuation that starts
after the fence was written must exit without submitting more work.

Calling `scancel` for only the initial array id is insufficient because a
continuation may already have submitted or may later submit additional work.

### 8.2 Submission metadata contract

Version the scheduler metadata so each stored id includes a role and launch
generation. The GUI reads:

- `chunk_job_ids` for ordinary data arrays and chunks;
- `slurm_job_ids` for all scheduler roles, including staged controllers,
  finalizers, and recovery jobs.

A staged run with empty `chunk_job_ids` and a `controller-initial` id in
`slurm_job_ids` is a successful submission. That role is the deterministic
initial primary scheduler handle. Do not choose a primary id from mapping
iteration order.

For pre-versioned metadata, merge ids with the append-only ledger. If neither
source can determine roles, preserve the ids with role `unknown` rather than
inventing a role.

The metadata reader returns a typed result:

```python
@dataclass(frozen=True)
class SubmittedJobSet:
    primary_id: str
    all_ids: tuple[str, ...]
    roles: Mapping[str, tuple[str, ...]]
    generation: UUID
```

### 8.3 Background lifecycle observer

Add a bounded server-side observer for nonterminal SLURM records. It does not
run scheduler commands inside the one-second Dash callback.

Status precedence:

1. inactive cancellation fence with possibly active or unreconciled work ->
   `cancelling`;
2. inactive cancellation fence with every recovered id confirmed inactive ->
   `cancelled`;
3. staged orchestration terminal state;
4. CLI completion or failure marker;
5. active `squeue` state -> `running` or `queued`;
6. all jobs `COMPLETED` but publication marker not yet visible ->
   `reconciling`;
7. explicit scheduler failure state -> `failed`;
8. exhausted reconciliation grace plus verified incomplete output -> `failed`;
9. scheduler unavailable with no durable terminal evidence -> `unknown`.

`RunStatus` explicitly includes `queued`, `reconciling`, and `cancelling`.
No single terminal scheduler job may terminalize a record while another
ledgered id or unresolved submission intent remains nonterminal.

Successful durable evidence is mode-specific:

- Validate: the generation-matched local process exits `0`; validation does
  not claim output publication.
- Local non-staged: the generation-matched process exits `0` and the atomic
  dashboard manifest reports `is_complete=true` for the expected inventory.
- Local staged: the generation-matched process exits `0`, every required
  Stage-3 per-image marker exists, and the staged completion marker matches
  the local pipeline identity and launch generation.
- Ordinary SLURM: every ledgered job and submission intent is terminal, the
  authoritative finalizer succeeds, and an atomic finalization marker for the
  launch generation accompanies an `is_complete=true` manifest.
- Staged SLURM: orchestration state is `complete`, every required Stage-3
  per-image marker exists, and `staged_completion_matches(output, epoch)` is
  true for the generation's epoch.
- Process/export mode: the authoritative process or finalizer succeeds, the
  expected exported-image inventory is complete, and its atomic publication
  marker matches the launch generation.

Where an existing mode lacks a generation-bearing top-level marker, add one
under `.phenotypic/progress/` and publish it only after all final outputs are
atomically visible. Scheduler `COMPLETED` alone is not sufficient because
shared-filesystem visibility and final publication can lag job exit.

The observer updates registry revision only when the effective state changes.
Recent Runs redraws from that revision instead of rescanning the whole sandbox
once per second.

### 8.4 SLURM log model

Persist GUI submitter stdout and stderr under
`.phenotypic/logs/gui/`. Scheduler logs remain under
`.phenotypic/logs/slurm/`.

The active log view reads incrementally with:

- one cursor per file;
- a fixed byte and line budget;
- file-role headings;
- rotation or truncation detection;
- no full-file reread on each interval.

Before scheduler logs exist, show submission output. Once scheduler logs land,
show both sources.

### 8.5 Staged GPU controls

When a pipeline contains a `GpuDetector`, show a staged-GPU section:

- CPU profile, using the existing common SLURM fields;
- GPU-stage delta profile, serialized as repeated `--gpu-slurm key=value`;
- GPU shards, serialized as `--gpu-shards`;
- a read-only note that the current implementation runs one resident model
  per shard.

Do not expose `gpu_workers_per_gpu` until replica packing exists.

Remove or relabel the ambiguous common `GPUs` field. If retained, label it
`CPU-stage GPUs` so it cannot be mistaken for Stage 2 allocation.

Generated-resource invariants:

- controller, Stage 1, Stage 3, and finalizer use the CPU profile;
- Stage 2 inherits the CPU profile and applies the GPU delta;
- Stage 2 defaults to one GPU unless explicitly overridden with zero;
- `gpu_shards` reaches the CLI unchanged.

### 8.6 Dashboard polling

Do not stop dashboard polling after the current short fixed window. Continue
while the record is nonterminal, using a bounded backoff. Bound polling
frequency, not the lifetime of a queued or running job. Stop on:

- dashboard found;
- terminal state with no dashboard;
- user unbind;

After a terminal state with no dashboard, show the terminal reason and a
manual Refresh action.

### 8.7 Phase 2 files

Files to modify:

- `src/phenotypic/gui/run_console/_slurm.py`
- `src/phenotypic/gui/run_console/_callbacks.py`
- `src/phenotypic/gui/run_console/_state.py`
- `src/phenotypic/gui/run_console/_layout.py`
- `src/phenotypic/gui/shell/_runs_registry.py`
- `src/phenotypic/sdk_/slurm/_dispatcher.py`
- `src/phenotypic/_cli/_cli_execution_strategies.py`
- `src/phenotypic/_cli/_cli_staged_slurm.py`
- `src/phenotypic/_cli/_cli_staged_orchestration.py`
- `src/phenotypic/phenotypicCLI.py`

Files to add:

- a shared CLI SLURM ledger/fence module if the staged module cannot be
  generalized without staged-only naming;
- a Dash-free GUI lifecycle observer module;
- focused tests for ordinary cancellation fencing and incremental log reads.

The existing opt-in live SLURM test remains the final scheduler integration
gate. Unit and integration tests use fake `sbatch`, `squeue`, `sacct`, and
`scancel` executables.

## 9. Phase 3: Builder and Tune authoring coherence

### 9.1 Keep Builder preview DOM stable

The current inspector is remounted with placeholder children during state
fan-in redraws. Separately, `render_inspector_preview` memoizes
`(selected_id, id(cached))` and returns `no_update` for an unchanged cache
entry. A redraw can therefore erase a valid preview without triggering a
restore.

Refactor the inspector into:

- a stable inspector shell mounted once;
- a dynamic header and parameter-form container;
- a stable `INSPECTOR_PREVIEW` container owned only by the preview callback.

No callback may replace an ancestor that contains `INSPECTOR_PREVIEW`. Refactor
every current `INSPECTOR_CONTAINER.children` writer, including load, prefab,
delete, undo, state fan-in, and navigation paths, to target stable child
containers instead.

Replace process-global object-identity memoization with a pipeline-state
revision plus preview generation:

```python
PreviewKey = tuple[str, str, str, int]
# session_id, selected_node_id, pipeline_revision, preview_generation
```

Every semantics-changing pipeline mutation invalidates the current preview
revision and shows `Preview stale - run again`. Run preview bakes all
intermediates into a temporary generation, then atomically publishes that
generation only after the complete bake succeeds. Partial cache writes never
become current. The preview renderer always restores cached content when the
selected node, pipeline revision, or preview generation changes.

The single-process cache remains supported. Document multi-worker GUI serving
as unsupported until an external cache is designed.

### 9.2 Make Tune Setup content real

Search space and Scorer currently remain literal placeholder children after
their locked CSS class is removed.

After pipeline resolution:

- load the pipeline or existing `TuningSpec`;
- render inferred knobs using existing `infer_search_space`, `space_to_spec`,
  and domain-editor components;
- preserve an existing spec's search space, scorer, strategy, and budget by
  default;
- for a pipeline input, require metadata and propose the default QC scorer;
- for an existing `TuningSpec`, make metadata optional and render its existing
  scorer;
- offer an explicit `Replace scorer with metadata-backed QC scorer` control
  before changing an existing spec's scorer;
- keep unsupported nested knobs visible and read-only;
- show all validation issues before Continue.

Continue is disabled until the full authored spec validates, not merely until
two path strings are truthy. Existing-spec mode does not require metadata
unless the user selects Replace scorer.

On click:

1. capture raw Setup controls as callback `State`;
2. resolve the pipeline/spec and any required or selected metadata path through
   the sandbox;
3. construct and validate the full `TuningSpec` in memory;
4. write the authored spec atomically under the GUI preset directory;
5. store its descriptor;
6. navigate to Run;
7. show a success or full-size failure alert.

### 9.3 Explicit path precedence

A path explicitly typed by the user wins over an older shell handoff.

Add pipeline and metadata picker buttons plus current-session selection stores
to Tune Setup. The stores carry the same sandbox-bounded path payload used by
the shared picker.

Resolution order:

1. nonempty typed path;
2. current-session picker selection;
3. valid shared shell handoff;
4. unset.

The UI labels the source of the resolved value. A stale handoff cannot silently
replace a typed value.

### 9.4 One Tune command object

Create one validated launch-command object containing:

- actual argv tokens;
- sandbox-resolved pipeline/spec, images, and output;
- execution target;
- redacted display tokens;
- copy eligibility;
- validation issues.

Deploy executes `argv`. Preview and Copy render `display_tokens`.

Reader-facing command modes:

- `GUI-equivalent`: the actual interpreter token used by the running GUI;
- `portable project command`: `uv run python -m phenotypic.tune ...`.

Credential-bearing storage URLs never appear in the DOM. Render an environment
variable token such as `$PHENOTYPIC_STORAGE_URL`.

The storage control has two explicit modes:

- local SQLite path, which is sandbox-resolved and may be displayed;
- server environment variable name, which defaults to
  `PHENOTYPIC_STORAGE_URL`.

In environment-variable mode the browser sends only the variable name. The
server resolves its value at deploy time, and the value is never returned to a
browser store, preview, callback error, or DOM node.

Copy is disabled while any placeholder such as `<images>` remains or any path
fails sandbox resolution.

### 9.5 Phase 3 files

Files to modify:

- `src/phenotypic/gui/builder/_callbacks.py`
- `src/phenotypic/gui/builder/_layout.py`
- `src/phenotypic/gui/builder/_linear_layout.py`
- `src/phenotypic/gui/builder/_session.py`
- `src/phenotypic/gui/tune/_layout.py`
- `src/phenotypic/gui/tune/_ids.py`
- `src/phenotypic/gui/tune/_callbacks.py`
- `src/phenotypic/gui/tune/_setup_authoring.py`
- `src/phenotypic/gui/tune/_command.py`
- `src/phenotypic/gui/tune/_run_argv.py`
- `src/phenotypic/gui/tune/_space.py`

Browser acceptance tests:

- Run Builder preview, assert the selected node renders, click the same node,
  and assert the preview survives. Modify a parameter and assert the exact
  `Preview stale - run again` state until a new atomic preview generation
  publishes. Repeat through load, prefab, delete, and undo redraw paths.
- Type a pipeline and metadata path in Tune, assert inferred Setup content,
  Continue, assert the authored file, and assert navigation to Run.
- Start from an existing `TuningSpec` and prove scorer, strategy, budget, and
  search space round-trip unchanged unless Replace scorer is explicitly
  selected.
- Seed a stale shared handoff, type a different path, and prove the typed path
  is authored.
- Compare the Tune display and deploy CLI tails after the documented
  launcher-prefix transform. Independently assert deploy uses the actual
  interpreter and Copy uses the portable `uv run python` prefix. Include paths
  with spaces, missing images, and server-side storage credential resolution.

## 10. Phase 4: Results, QC, Error, and Analysis safety

### 10.1 Pure output compatibility preflight

Add a pure compatibility service returning:

```python
CompatibilityStatus = Literal["compatible", "migratable", "blocked"]

@dataclass(frozen=True)
class CompatibilityIssue:
    code: str
    status: CompatibilityStatus
    location: str
    message: str
    proposed_change: str | None = None

@dataclass(frozen=True)
class OutputCompatibilityReport:
    status: CompatibilityStatus
    source_fingerprint: str
    issues: tuple[CompatibilityIssue, ...]
    migrated_pipeline_payload: dict[str, object] | None = None
```

Preflight:

- parses but does not write;
- preserves raw entries that fail typed instantiation;
- validates every proposed migration before returning `migratable`;
- never mutates `QcRecipe.load_warnings`;
- produces each warning once.

### 10.2 Versioned GridOccupancy migration

Support the exact observed historical shape:

```json
{
  "class": "GridOccupancy",
  "params": {
    "metadata": "",
    "metadata_source": "/path/layout.csv",
    "cell_label": null
  }
}
```

Deterministic migration:

- if `metadata` is empty and `metadata_source` is nonempty, copy
  `metadata_source` to `metadata`;
- if both are nonempty and unequal, block as ambiguous;
- remove `metadata_source`;
- omit `cell_label` when it is null so the current default applies;
- map retired metadata column spellings only through an explicit, tested alias
  table and only when the target column exists;
- block missing metadata files or unmappable group columns.

`Migrate recipe`:

1. reruns preflight and verifies the source fingerprint;
2. creates a timestamped backup of the pipeline payload;
3. atomically replaces the pipeline only after every QC entry instantiates;
4. writes a migration receipt containing old/new version and hashes;
5. is idempotent.

Do not silently migrate during bind.

### 10.3 Preserve the `qc.duckdb` cutover

The new QC database is catalog-driven and cannot be reconstructed losslessly by
reading only the retired flat summary/member parquets. Keep the hard cutover.

Replace the passive message with an explicit `Rebuild QC database` action when:

- the pipeline recipe is compatible;
- the measurements mirror is complete;
- no active CLI writer owns the output;
- the output is writable.

The action calls the existing atomic `run_qc` writer. Legacy parquets remain
untouched until the new database has been written and validated. If recipe
compatibility blocks rebuild, show the exact recipe blocker first.

### 10.4 Make discovery and viewing source-preserving

`OutputRoot.discover` becomes pure. It does not:

- move legacy `qc/`;
- fold a legacy viewer sidecar into the pipeline;
- create source-local DZI directories;
- write any viewer cache.

Generated DZI and viewer caches default to:

```text
<sandbox>/.phenotypic-gui/viewer_cache/<source-key>/
```

where `source-key` includes the canonical source path and content fingerprint.
Persistent source-local cache is an opt-in advanced action, not the default.

Legacy topology and sidecar migrations move behind the compatibility panel and
use the explicit-write invariant.

### 10.5 Explicit snapshot freshness

The shell stores the bound output path and snapshot descriptor, not only a
frozen `OutputRoot` object.

The header displays:

- snapshot timestamp;
- source fingerprint or revision;
- `Current`, `Changed on disk`, or `Active run snapshot`.

`Refresh snapshot`:

1. checks that there are no unsaved curation edits;
2. captures pre-read source fingerprints;
3. rediscovers Results data;
4. captures post-read fingerprints;
5. retries or refuses if files changed during discovery;
6. atomically swaps the Results and Analysis sessions to the same revision.

Active outputs default to read-only. Mutation controls remain disabled until a
stable completed snapshot is selected or the user explicitly acknowledges the
active-run risk.

### 10.6 Make Error activation compute-only

Opening or switching to Error computes an in-memory preview only. It does not
write canonical `error_analysis.parquet`, CSV, verified baseline, or any
deliverable.

Add an explicit `Publish all categories` action. Publication:

- computes all configured categories, not only the focused category;
- fingerprints master, mirror, labels, QC DB, and review state;
- refuses if any fingerprint changed since computation;
- stages parquet, CSV, and a generation manifest together;
- atomically publishes the complete generation;
- never overwrites an all-category result with a focused preview.

The CLI finalizer remains the normal authoritative publisher.

### 10.7 Preserve unknown Analysis nodes

Analysis tolerant loading retains opaque JSON for every unknown or invalid
analyzer node.

On save, choose one of two safe outcomes:

- merge scoped edits into the original payload while preserving opaque nodes;
- block Save and require the user to explicitly migrate or drop the unknown
  nodes.

An unrelated edit never serializes only the successfully instantiated subset.

### 10.8 Phase 4 files

Files to modify:

- `src/phenotypic/sdk_/_qc_recipe/_recipe.py`
- `src/phenotypic/analysis/qc/_expected_vs_detected.py`
- `src/phenotypic/analysis/qc/_grid_occupancy.py`
- `src/phenotypic/gui/results_viewer/_output_root.py`
- `src/phenotypic/gui/results_viewer/_app.py`
- `src/phenotypic/gui/results_viewer/_layout.py`
- `src/phenotypic/gui/results_viewer/_qc_tab/_callbacks.py`
- `src/phenotypic/gui/results_viewer/_qc_tab/review/_db.py`
- `src/phenotypic/gui/results_viewer/_qc_tab/review/_callbacks.py`
- `src/phenotypic/gui/results_viewer/_error_tab/_callbacks.py`
- `src/phenotypic/gui/results_viewer/_tile_routes.py`
- `src/phenotypic/gui/shell/_routes.py`
- `src/phenotypic/gui/shell/_app.py`
- `src/phenotypic/gui/analysis/_recipe_state.py`
- `src/phenotypic/sdk_/_io_constants.py`

Files to add:

- a Dash-free output-compatibility module;
- compatibility fixtures for each supported historical schema;
- generation/fingerprint helpers if existing atomic-write utilities do not
  provide them.

## 11. Phase 5: Paths, extensions, and documentation

### 11.1 One image-extension contract

All GUI image discovery and picker code imports
`phenotypic.gui._config.IMAGE_EXTS`.

Current canonical behavior:

- includes `.cr3`;
- excludes ambiguous `.raw`;
- includes the existing raster and supported camera RAW formats.

Remove hand-written Run Console and Tune Curate extension sets. Add identity
tests so consumers cannot drift.

### 11.2 Versioned and resolved shared path state

Introduce version 2 source and metadata payloads:

```json
{
  "version": 2,
  "kind": "image_source",
  "relative_path": "data/subset",
  "absolute_path_at_selection": "/old/root/data/subset",
  "sandbox_fingerprint": "...",
  "validation": {
    "exists": true,
    "is_directory": true
  },
  "selected_at": "..."
}
```

Resolution accepts `relative_path` under the current sandbox only when the
stored sandbox fingerprint matches. A mismatch is unavailable pending explicit
user confirmation, even if the new sandbox happens to contain the same
relative name. The old absolute path is diagnostic only.

Version 2 preserves the existing source/metadata kind and validation fields.
Selection recomputes them against the current sandbox rather than trusting
stored booleans.

The displayed label comes from the resolved current path. If resolution fails,
show:

> Previous source unavailable in this sandbox

with Pick and Clear actions. Do not display the stale path as if it were active.

Read version 1 payloads for backward compatibility and rewrite them to version
2 only after a successful explicit selection.

### 11.3 Refresh contract

Keep Refresh explicit rather than adding a recursive watcher.

One shared refresh revision must:

- invalidate classifier cache;
- rerender the root sidebar snapshot;
- invalidate open directory-picker snapshots;
- re-resolve shared source and metadata labels;
- refresh capability badges for nested active outputs.

For a bound nonterminal run, the lifecycle observer can bump the targeted path
capability revision without recursively scanning the sandbox.

### 11.4 Reader-facing commands and tutorials

Update:

- Browse tutorials to use Settings -> Input folder -> Pick;
- Browse Timeline tutorial and `WORKFLOWS.md` to match the same path;
- Tune metadata examples to canonical metadata headers;
- Tune installation guidance to `uv sync --extra ...` or the repository's
  exact supported extra command;
- copied project commands to `uv run python -m ...`;
- Analysis recompile guidance to the same project command convention;
- Results header and output-root recompile guidance;
- QC Review callback/layout recompile guidance;
- `docs/source/tutorials/gui/15_qc_review.md`.

Actual subprocess execution continues to use the current interpreter from the
validated command object. Reader-facing portable copy and internal execution
are labeled separately.

Any visible GUI change also updates:

- `src/phenotypic/gui/FEATURES.md`;
- `src/phenotypic/gui/WORKFLOWS.md` when the flow changes;
- the relevant GUI tutorials;
- tutorial screenshot capture code and committed screenshots.

### 11.5 Phase 5 files

Files to modify include:

- `src/phenotypic/gui/_config.py`
- `src/phenotypic/gui/run_console/_callbacks.py`
- `src/phenotypic/gui/tune/_callbacks.py`
- `src/phenotypic/gui/tune/_command.py`
- `src/phenotypic/gui/shell/_layout.py`
- `src/phenotypic/gui/shell/_callbacks.py`
- `src/phenotypic/gui/shell/_source_context.py`
- `src/phenotypic/gui/shell/_metadata_context.py`
- `src/phenotypic/gui/results_viewer/_output_root.py`
- `src/phenotypic/gui/results_viewer/_qc_tab/review/_callbacks.py`
- `src/phenotypic/gui/results_viewer/_qc_tab/review/_layout.py`
- `src/phenotypic/gui/analysis/_layout.py`
- `src/phenotypic/gui/FEATURES.md`
- `src/phenotypic/gui/WORKFLOWS.md`
- `docs/source/tutorials/gui/15_qc_review.md`
- `docs/source/tutorials/gui/16_tune_copilot.md`
- `docs/source/tutorials/gui/18_browse.md`
- `docs/source/tutorials/gui/19_browse_timeline.md`
- `scripts/capture_gui_tutorial_screenshots.py`

Tests cover the shared context payloads, extension identity, displayed command
strings, and browser Refresh behavior across every consumer named above.

## 12. Compatibility and migration policy

### 12.1 Version support

- Current payloads are read natively.
- Explicitly enumerated historical payloads can be `migratable`.
- Unknown fields or ambiguous mappings are `blocked`, not guessed.
- Raw unknown entries remain available for export and diagnosis.

### 12.2 Backups

Configuration migration backups live adjacent to the canonical configuration
or under a dedicated migration-backup directory resolved by an SDK helper.
Names include timestamp and source hash.

Do not place ad hoc backup names in GUI callback code.

### 12.3 Idempotence

Running compatibility preflight repeatedly makes no changes. Running an
already-applied migration reports `compatible` and does not create another
backup.

### 12.4 Rollback

The migration receipt names the exact backup and resulting hash. A separate
explicit Restore action may be added, but the implementation must at minimum
make manual rollback unambiguous.

## 13. Test strategy

### 13.1 Unit tests

Pure tests cover:

- raw-controls to `RunConsoleState`;
- output freshness with the exact `.gui_log` exception;
- launch generation fencing, output ownership, and stale-generation rejection;
- local exit-code transitions and cleanup at every launch boundary;
- SLURM time parsing;
- versioned `chunk_job_ids`, `slurm_job_ids`, ledger-role, and deterministic
  primary-id resolution;
- scheduler/durable-state precedence, including `queued`, `reconciling`,
  `cancelling`, scheduler-unavailable, and reconciliation-grace exhaustion;
- exact completion evidence for validate, local, staged local, ordinary SLURM,
  staged SLURM, and process/export modes;
- incremental log cursors;
- Builder preview-generation keys;
- Tune path precedence and command parity;
- compatibility classification and exact legacy migrations;
- source-store v1 to v2 resolution;
- extension-set identity.

### 13.2 Integration tests

Integration tests cover:

- a real local GUI runner subprocess with a tiny pipeline;
- immediate process exit before callback registration completes;
- log-directory, `Popen`, and exit-observer startup failures with no leaked
  process or reservation;
- two concurrent SLURM submission futures with navigation and active-row
  changes;
- cancellation fencing with fake continuation jobs;
- submitter timeout after `sbatch` but before its result reaches the GUI;
- submitter crash between `sbatch` and ledger commit, recovered by the
  generation scheduler comment;
- a chunk submission crash before its continuation is recorded;
- cancellation remaining `cancelling` while scheduler state is unavailable,
  then becoming `cancelled` only after every recovered id is inactive;
- one completed job while another ledgered job or unresolved intent is still
  pending, which must not terminalize the run;
- staged metadata with only a controller id;
- Tune Setup authoring and atomic spec write;
- QC migration backup, validation, and idempotence;
- `qc.duckdb` rebuild success and rollback on failure;
- Results snapshot refresh across Results and Analysis;
- Analysis preservation of unknown nodes;
- Error explicit publication with fingerprint mismatch refusal.

### 13.3 Browser E2E tests

Browser tests cover:

1. Local Validate success and failure reach terminal states.
2. Local Run on a fresh output does not fail because of `.gui_log`.
3. Rapid Local -> SLURM -> Run uses SLURM.
4. Rapid SLURM -> Local -> Run uses Local.
5. SLURM submit failure leaves a failed diagnostic row.
6. Cancelling SLURM remains visibly nonterminal until all generation jobs are
   confirmed inactive.
7. Builder preview survives same-node selection and inspector redraw.
8. Tune Setup renders actual content, authors a spec, and advances.
9. Results bind, tab navigation, and first tile request leave the source tree
   byte-for-byte unchanged.
10. Error tab activation produces no file changes.
11. Compatibility and rebuild actions require explicit confirmation.
12. Run and Tune Curate can select `.cr3`, while ambiguous `.raw` is absent.
13. Version 1 and version 2 localStorage payloads show unavailable after a
    sandbox-fingerprint change until the user explicitly reselects.
14. Create a nested output after page load, click Refresh, and assert sidebar,
    open picker, source/metadata labels, and capability badges all update.

### 13.4 Live SLURM test

The existing opt-in live SLURM test is extended or accompanied by one GUI
submission test using:

- one small image;
- one CPU-only pipeline;
- one worker;
- a unique output;
- a short partition and bounded time;
- scheduler cleanup in a `finally` path;
- job ledger assertions;
- terminal manifest verification.

Add a second opt-in live cancellation case that waits until at least one
continuation is submitted, cancels, and proves no ledgered or
generation-commented scheduler job survives or appears afterward.

The live tests:

- are marked `slow`;
- require `PHENOTYPIC_RUN_LIVE_SLURM=1`;
- skip when scheduler tools are unavailable;
- create a unique temporary input and output;
- assert cleanup in `finally`.

The implementation adds the exact GUI live-test file shown here:

```bash
PHENOTYPIC_RUN_LIVE_SLURM=1 \
  uv run pytest tests/integration/gui/test_run_console_slurm_live.py -m slow -v
```

The live test is opt-in and never targets an existing output.

### 13.5 Source-tree mutation assertions

Results safety tests hash or tree-diff the source before and after:

- discovery;
- app construction;
- every tab activation;
- tile generation;
- browser reload.

Only the explicit migration, rebuild, curation, analysis, or publication action
may alter the expected target set.

## 14. Implementation sequence

### Bundle A: Local and submission correctness

- Phase 1 controls, fresh-output rule, stable run identity, exit callbacks, and
  time parser.
- This bundle is first because it restores trustworthy small local tests and
  prevents silent Local/SLURM target changes.

### Bundle B: Scheduler lifecycle

- Phase 2 generalized ledger, cancellation fence, status observer, logs, and
  staged GPU controls.
- Land ordinary and staged tests together so cancellation semantics cannot
  diverge again.

### Bundle C: Authoring UI coherence

- Phase 3 Builder stable preview container and Tune functional Setup/command
  object.

### Bundle D: Read-only Results and compatibility

- Phase 4 pure discovery, external viewer cache, compatibility preflight,
  explicit migration/rebuild, snapshot refresh, safe Error publication, and
  Analysis opaque-node preservation.
- Characterization tests must land before moving existing migrations out of
  discovery.

### Bundle E: Path and documentation convergence

- Phase 5 canonical extensions, path payload v2, shared refresh, ledgers,
  tutorials, and screenshots.

Each bundle is independently reviewable and must leave all earlier invariants
passing.

## 15. Definition of done

The work is complete when:

- all P0 and P1 rows in the issue inventory have an automated regression test;
- Run and Validate terminal states match process exit;
- selected SLURM mode cannot invoke a local CLI command;
- every submitted scheduler job is recovered, ledgered, and cancellable even
  when the submitter exits between `sbatch` and ledger commit;
- cancellation remains `cancelling` until every generation job is confirmed
  inactive;
- output publication and Resume reject stale or conflicting generation-owner
  state;
- staged controller-only metadata is accepted;
- Builder preview remains visible after inspector redraw;
- Tune Setup renders and authors a real spec through a browser test;
- binding and viewing Results changes no source files;
- historical QC compatibility is classified before instantiation;
- `qc.duckdb` rebuild is explicit and atomic;
- opening Error changes no files;
- copied commands and deployed argv share one CLI-tail token source after the
  documented launcher-prefix transform;
- all GUI image pickers use canonical `IMAGE_EXTS`;
- `FEATURES.md`, applicable `WORKFLOWS.md`, tutorials, and screenshots are
  synchronized;
- targeted unit, integration, browser, and opt-in live SLURM tests pass.

## 16. Verification commands

Exact test filenames may expand during implementation. The minimum focused
verification set is:

```bash
uv run pytest tests/unit/gui/run_console
uv run pytest tests/integration/gui/test_run_console_callbacks.py
PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_run_console.py
uv run pytest tests/gui/builder tests/unit/gui/builder
uv run pytest tests/integration/gui/tune tests/unit/gui/tune
uv run pytest tests/gui/results_viewer tests/unit/gui/results_viewer
uv run pytest tests/unit/qc tests/integration/gui/test_qc_review_recompute.py
PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_qc_tab.py
PLAYWRIGHT=1 uv run pytest tests/e2e/gui
uv run pytest tests/unit/gui/shell
uv run ruff check <explicit changed paths>
uv run mypy src/phenotypic
uv run python scripts/check_features_md.py
uv run python scripts/check_workflows_md.py
uv run pytest tests/integration/cli/test_staged_slurm_live.py -m slow -v
PHENOTYPIC_RUN_LIVE_SLURM=1 \
  uv run pytest tests/integration/gui/test_run_console_slurm_live.py -m slow -v
```

Run live SLURM tests only in an isolated scheduler environment with their
required markers and cleanup contracts. If the existing CLI live test also
uses an opt-in environment gate, set its documented gate in the command rather
than weakening its skip protection.

## 17. Numerical validation

This design does not introduce a load-bearing numerical invariant. No
`docs/superpowers/logic_validation_scripts/` script is required.

Scientific QC behavior is not re-derived or changed. Compatibility tests prove
serialization mapping, validation, atomicity, and preservation of existing
configured values.
