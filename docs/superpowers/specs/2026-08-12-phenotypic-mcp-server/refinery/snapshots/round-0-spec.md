# PhenoTypic MCP Server — §1 Architecture and Process Model

Status: **draft, reviewed once, revised**
Date: 2026-08-12

## 1.1 Purpose

Expose PhenoTypic to an LLM agent so that the agent can:

1. Build candidate `ImagePipeline` configurations from the operation catalog.
2. Tune them with `phenotypic.tune` (grid / random / Optuna), locally or on SLURM.
3. Deploy a winning pipeline over a full dataset via `python -m phenotypic`.

The server is a **new surface over existing engines**, not a new engine. Every
numeric result it returns is produced by the same code the CLI and GUI already
run.

## 1.2 Prior intent in the codebase

This server is not speculative. Four sites already name it as the intended
consumer, and the design must honour the contracts they imply:

| Site | Contract it fixes |
|---|---|
| `abc_/_base_operation.py:192` | `model_json_schema()` — with docstring-derived field descriptions — **is** the operation contract handed to the agent. |
| `sdk_/_docstring_params.py:7` | Google-style `Args:` blocks are the source of those descriptions. Docstring quality is API quality. |
| `tune/_search_space/_discovery.py:4` | `pipeline_targets()` powers a "what can I tune?" tool where the agent **selects a structured target, never authors a string key**. |
| `tune/_spec.py:293` | `TuningSpec`'s model validators are the submit-time gate — "where an MCP submits". Spec rejection happens at construction, not at evaluation. |

## 1.3 Process model

One stdio server per agent session, co-located with the agent on the cluster
login node.

```
workstation ──ssh──> login node
                      ├─ claude code (orchestrator)
                      │    ├─ subagent A ─┐
                      │    ├─ subagent B ─┼─ share ONE stdio connection
                      │    └─ subagent C ─┘
                      └─ phenotypic-mcp (stdio child process)
                           ├─ workspace on shared FS
                           └─ sbatch / squeue / sacct
```

Consequences, all load-bearing:

- **Subagents do not get their own server.** In Claude Code a subagent inherits
  the parent's MCP connections, so N subagents produce interleaved calls into a
  *single* process. Every tool must be safe under concurrent invocation.
- **No auth layer.** The server runs as the user, with the user's filesystem and
  scheduler rights. Its security boundary is the workspace sandbox (§2), not
  authentication.
- **All paths are local paths.** No staging, no file transfer, no URL scheme.
- **The server may be killed and restarted at any time** (agent restart, session
  end). It therefore holds no authoritative state — see §2. Where this spec says
  a subagent "holds a pipeline", it means the agent holds an **id** and the
  server re-reads that artifact from disk on each call; no live `ImagePipeline`
  object survives between tool invocations. What a restart *does* cost is
  in-flight local work, which §1.5 reconciles rather than ignores.

## 1.4 Layering

```
┌─────────────────────────────────────────────────────────┐
│ phenotypic/mcp/            MCP tool layer (thin)         │
│   _server.py               transport, dispatch, limits   │
│   _tools/{catalog,pipeline,tune,deploy,workspace}.py     │
│   _errors.py               structured error envelope     │
│   _routing.py              work-class → executor + slot  │
└────────────────────────┬────────────────────────────────┘
                         │ imports only
┌────────────────────────▼────────────────────────────────┐
│ phenotypic/_services/      shared, Dash-free service tier│
│   registry.py   ← gui/_operation_registry.py             │
│   sandbox.py    ← gui/shell/_sandbox.py                  │
│   runs.py       ← gui/shell/_runs_registry.py,           │
│                   gui/run_console/_runner.py             │
│   tune_spec.py  ← gui/tune/{_setup_authoring,_command,   │
│                   _validation,_export}.py                │
│                 ← gui/tune/_space.py  **PURE HALF ONLY** │
│   argv.py       ← gui/run_console/_state.py::to_argv     │
│                   AND its RunConsoleState dataclass      │
│                 ← gui/tune/_run_argv.py                  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│ Existing engines (unchanged)                             │
│   _core/_image_pipeline.py     ImagePipeline             │
│   tune/                        run_tuning, TuningEngine  │
│   _cli/, phenotypicCLI.py      execution strategies      │
│   sdk_/_io_constants.py        every path helper         │
└──────────────────────────────────────────────────────────┘

phenotypic/gui/*  →  re-export shims onto _services (GUI behaviour unchanged)
```

### P2 is nine moves and one split — not a pure move

An earlier draft of this section claimed the promotion was "a move, not a
rewrite". **That is false for `gui/tune/_space.py`**, which carries module-level
`import dash_bootstrap_components as dbc` and `from dash import html` at
`_space.py:33-34`. The file mixes two separable halves:

| Half | Symbols | Destination |
|---|---|---|
| Pure | `space_to_spec` (`:209`), `apply_space_edits` (`:161`), `_build_search_space` (`:134`) | `_services/tune_spec.py` |
| Dash view | `_knob_form` (`:396`), `setup_knob_forms` (`:468`), `build_space_view` (`:503`) | stays in `gui/tune/_space_view.py`, importing the pure half back |

The split is forced, not optional: `gui/tune/_setup_authoring.py:27` does
`from phenotypic.gui.tune._space import apply_space_edits, space_to_spec`, so
`_setup_authoring.py` cannot be promoted without either splitting `_space.py` or
dragging Dash into `_services`. This counts as a **fourth genuinely new piece**
of work in §1.6, not as mechanical churn.

**A second, non-obvious Dash edge runs through `RunRegistry` itself.**
`rehydrate_from_sandbox` — the boot-recovery method §2.4 depends on — calls
`classify()` (`gui/shell/_runs_registry.py:59,712,823`), and
`gui/shell/_classifier.py:34` does
`from phenotypic.gui.builder._directory_browser import IMAGE_EXTS`, where
`_directory_browser.py:20-21` imports `dash_bootstrap_components` and `dash`.

So promoting `runs.py` is a **three-file** job, not two: `_runs_registry.py`,
`run_console/_runner.py`, **and** the image-extension constant that
`_classifier.py` needs. `IMAGE_EXTS` is a bare set of suffixes with no Dash
content; it belongs in `sdk_/_io_constants.py` alongside the other filename
constants, with `_directory_browser.py` importing it from there. That relocation
is a prerequisite of P2, listed in §7.

The general lesson, which the import-purity test in §6.5 exists to enforce: a
module being *content*-clean does not make its import graph clean.

`to_argv` likewise cannot travel alone: its signature is
`to_argv(state: RunConsoleState)` and `RunConsoleState` is defined in the same
file (`run_console/_state.py:70`). The dataclass is clean — plain, already
JSON-serializable, no Dash coupling — so it moves with the function. Leaving it
behind would make `_services/argv.py` import back up into `gui/`, inverting the
layering the diagram asserts.

### What "Dash-free" does and does not buy

`_services` will be genuinely Dash-free: after the split, every promoted module's
own imports are stdlib + phenotypic-internal.

But note what that does *not* fix. `gui/shell/__init__.py:17` and
`gui/tune/__init__.py:18` are **eager** — each does
`from ._app import create_app` at package-import time, unlike the deliberately
lazy `gui/__init__.py`. So importing `phenotypic.gui.shell._sandbox` today drags
`dash`, `dash_bootstrap_components`, `flask`, and `werkzeug` into `sys.modules`
purely by executing the parent package's `__init__`, even though `_sandbox.py`'s
own content is clean.

The MCP server sidesteps this entirely by importing `phenotypic._services.*`
and never touching `phenotypic.gui`. Existing GUI call sites — 43 for
`SandboxRoot`, 15 for `RunRegistry` — keep paying the Dash import cost until
they are repointed at `_services`. **Making those `__init__.py`s lazy is
deferred cleanup, not a prerequisite**, and this spec does not require it.

(An earlier draft cited `gui/tune/__init__.py:1-15` as evidence of
Dash-freedom. That docstring asserts *optuna*-freedom, and the file eagerly
imports the Dash app factory three lines below. Citation withdrawn.)

### Why promote rather than import in place

The MCP server and the GUI need the same six capabilities. Importing
`phenotypic.gui._operation_registry` from `phenotypic.mcp` would make a private
module of one user-facing surface the de-facto API of another, with no test
protecting the boundary. Promotion costs one refactor and buys a layer that can
be tested and versioned on its own terms.

### Object lifecycle in the server

Two patterns exist today for holding an `OperationRegistry`: a lazy module
global (`get_registry()` / `_REGISTRY` at `gui/_operation_registry.py:811-823`)
and per-Dash-app caching in `app.server.config[CFG_OPERATION_REGISTRY]`. The
second has no analogue in a stdio server. **The MCP server uses the module
global** — a function re-export shim preserves it correctly, since the global
lives in the promoted module's own namespace.

`RunRegistry` is instance-based, not a singleton. The server constructs
**exactly one**, bound to the workspace root, at startup, and runs
`rehydrate_from_sandbox` before serving the first request.

## 1.5 Smart routing: work classes × environment

The server never asks the agent "should this run locally or on SLURM?" It
classifies the *work* and routes it. This is what keeps N concurrent subagents
from OOMing a shared login node without making the server useless off-cluster.

**Work classes**

| Class | Meaning | Examples |
|---|---|---|
| `W0` introspect | Pure computation over metadata; no image I/O | list operations, describe schema, validate a spec, infer a search space, diff two pipelines |
| `W1` probe | Bounded image compute, interactive latency | apply a pipeline to 1–N images and return measurements + benchmark |
| `W2` study | Unbounded optimization | a tune study |
| `W3` deploy | Full-dataset processing | `python -m phenotypic` over a dataset tree |

**Environment** defaults to `slurm` when **(a)** `sbatch` resolves on `PATH`,
**(b)** a liveness probe `squeue -h --me` exits 0 within a short timeout, and
**(c)** at least one SLURM profile is configured (§5). Otherwise `local`.

A bare `shutil.which("sbatch")` is deliberately *not* sufficient: a binary on
`PATH` proves nothing about a live controller, valid credentials, or a usable
allocation. Note also that the repo's existing gate is explicit opt-in, not
detection — `ExecutionConfig.is_slurm_mode()` (`_cli/_cli_types.py:176-180`) is
just `force_local ? False : bool(slurm_args)`. Auto-detection is therefore a
*convenience default this server adds*, and it must be conservative.

The probe runs at startup and is cached; `workspace_info` reports both the
verdict and its basis, and a `refresh` flag re-probes. When a submission fails
because the scheduler went away mid-session, the error names the environment
verdict and its staleness rather than surfacing a raw `sbatch` error.
`execution.environment` in config overrides detection in both directions.

### One arbiter, not two

**Invariant: at most one image-touching local computation runs at a time,
process-wide.**

An earlier draft claimed an `asyncio` lock enforced this while simultaneously
giving `W2`/`W3` a *separate* "sequential queue". Two uncoordinated mechanisms
do not produce one invariant: a `W1` probe holding the lock in-process and a
locally-routed `W3` deploy in a subprocess could run concurrently — and that
subprocess is `LocalParallelStrategy` with joblib `n_jobs=-1`, i.e. every core
(`phenotypicCLI.py:788-792`, resolved via `os.cpu_count()`). That is precisely
the thrash this section exists to prevent.

There is **one** `LocalComputeSlot`: a process-wide semaphore of capacity 1.

| Work | Acquires the slot? | Held for |
|---|---|---|
| `W0` any environment | no | — |
| `W1` probe | **yes** | the duration of the in-process compute |
| `W2`/`W3` routed `local` | **yes** | the entire subprocess lifetime, released on reap |
| `W2`/`W3` routed `slurm` | no | — (the scheduler is the arbiter) |

`LocalRunner` offers nothing to reuse here — it is a multi-handle subprocess
tracker with no exclusivity guard, because the GUI never needed one (a human
clicks Run once). The slot is new code, listed as such in §1.6.

**Routing table**

| Class | `local` env | `slurm` env |
|---|---|---|
| `W0` | in-process, no slot | in-process, no slot |
| `W1` | in-process, **holds slot** | in-process, **holds slot** |
| `W2` | subprocess, **holds slot** | `sbatch` fleet, no slot |
| `W3` | subprocess, **holds slot** | `sbatch` array, no slot |

### Blocking work never blocks the event loop

`ImagePipeline.apply()` is synchronous, CPU-bound, and copies the image
(`_image_pipeline_core.py:943-966`). Running it directly in an async handler
would block the entire event loop — stalling `W0` calls from *other* subagents
and silently falsifying "agent-side fan-out is free". **`W1` compute runs via
`run_in_executor`**; the handler holds the slot and awaits, so `W0` dispatch
stays responsive throughout.

`W1` also drops its `Image` objects before releasing the slot, so peak residency
is one probe rather than a sawtooth that grows across a long-lived session.

`W1` carries a second, independent guard: a hard cap on scope (default 4 images)
and a wall-clock timeout. A mis-set slot still cannot melt the node.

`W2`/`W3` in a `local` environment are *not* refused — that would make the
server useless on a workstation — but they serialize on the same slot, and the
tool result says so, so the agent learns the run is queued rather than
mysteriously slow.

### Restart reconciliation

The server may be killed without running `LocalRunner`'s `atexit` cleanup
(`run_console/_runner.py:141-145`), orphaning a local `W2`/`W3` subprocess. A
fresh server that simply reset its slot would admit a second local job beside
the orphan, doubling contention.

So on startup, **before serving any request**, the server runs
`rehydrate_from_sandbox` and reconciles: for every nonterminal `RunRecord`, it
checks whether the recorded PID is still live. A live orphan **claims the slot**;
a dead one is CAS'd to `failed` with `status_detail` naming the lost server.
Only then does the server accept work.

### What the agent sees when routing bites

Routing is never silent. Every submit-class tool result carries the decision:

```json
{
  "run_id": "runs/2026-08-12-plateA-b3f2",
  "routed_to": "slurm",
  "reason": "class=W3, environment=slurm, profile=cpu-bulk",
  "queue_position": null
}
```

and a queued local run reports `"routed_to": "local", "queue_position": 2`.

## 1.6 Reuse inventory

What exists today versus what this design adds:

| Capability | Status | Source |
|---|---|---|
| Operation discovery + param introspection | **exists** | `gui/_operation_registry.py` — `get_registry()`, `OperationInfo`, `ParamInfo` |
| Field descriptions in JSON schema | **exists** | `BaseOperation.__pydantic_init_subclass__` → `apply_docstring_descriptions` |
| Pipeline (de)serialization | **exists** | `ImagePipeline.to_json` / `from_json`; accepts a pre-parsed dict |
| Flat-key param overlay onto a pipeline | **exists** | `tune/_evaluation/_builder.py:366` `build_pipeline` |
| Tunable-target discovery | **exists** | `tune/_search_space/_discovery.py:73` `pipeline_targets` |
| Search-space inference | **exists** | `tune/_search_space/_infer.py:685` `infer_search_space` |
| Tune spec authoring + validation | **exists** | `gui/tune/_setup_authoring.py:534`, `gui/tune/_validation.py` |
| Tune launch (in-process) | **exists** | `tune/_tune_cli/_run.py:483` `run_tuning` |
| Best-trial → pipeline export | **exists** | `gui/tune/_export.py`, `tune/_evaluation/_builder.py` |
| argv construction for both CLIs | **exists** | `gui/run_console/_state.py:515`, `gui/tune/_run_argv.py` |
| Run registry with on-disk records, generation fencing | **exists** | `gui/shell/_runs_registry.py` — `RunRegistry.allocate` / `compare_and_set` |
| Local subprocess supervision | **exists** | `gui/run_console/_runner.py` `LocalRunner` |
| Machine-readable progress | **exists** | `.phenotypic/progress/manifest.json`, `processing_events.log` |
| Path helpers for every artifact | **exists** | `sdk_/_io_constants.py` |
| Filesystem sandbox | **exists** | `gui/shell/_sandbox.py` `SandboxRoot` |
| — | — | — |
| JSON-serializable operation descriptor | **new** | projection over `OperationInfo`/`ParamInfo` (~40 lines) |
| Declarative pipeline put/patch | **new** | §3 |
| SLURM profile config + caps enforcement | **new** | §5 |
| Work-class routing, `LocalComputeSlot`, restart reconciliation | **new** | §1.5 — nothing to reuse; `LocalRunner` has no exclusivity guard |
| MCP tool layer and error envelope | **new** | §3, §6 |
| `_services` promotion — 9 moves | **new** (mechanical) | §1.4 |
| `gui/tune/_space.py` pure/view split | **new** (real refactor) | §1.4 — forced by `import dash` at `_space.py:33-34` |
| `build_array_script_spec` extraction (pure sbatch render) | **new** (real refactor) | §5.3 — `generate_array_job_script` writes files under `output_dir`, so `deploy_plan` cannot reuse it |
| Directory-level digest helper | **new** | §7 P3 — every existing fingerprint helper is single-file |
| Persistent probe worker subprocess | **new** | §3.2 — nothing bounds an in-process `apply()` by wall clock; `LocalRunner` is fire-and-forget `Popen`, not a request/response worker |
| Killable store-open subprocess | **new** | §4.4, §7 P7 — the nearest analogue (the GUI Monitor's live-open pool) is documented as deliberately *non*-killable |
| Subset staging (materialize a file list as a directory) | **new** | §7 P6 — neither engine accepts a file list |
| Plan / promotion token records | **new** | §5.4 — opaque ids over persisted records, not forgeable digests |

Roughly: the server is a **thin adapter plus nine genuinely new pieces** —
descriptor projection + column derivation, profile governance, routing + slot,
the `_space.py` split, the pure sbatch-spec extraction, subset staging, the token
store, the persistent probe worker, and the killable store-open subprocess. The
`_services` promotion itself is mechanical; nothing else on that list is.

The count went **3 → 4 → 5 → 7 → 9** as successive reviews traced what the design
actually requires. The estimate was optimistic every time, and twice this table
went stale because a later section grew a prerequisite that was never carried
back here — which is the same drift the reviews kept finding in the worked
examples, one level up. **This table and `README.md`'s summary are part of the
edit whenever §7 gains a prerequisite.**

## 1.6.1 Performance requirements

Stated because the design has one shared process serving N subagents (§1.3), so
"slow" and "blocks everyone" are the same failure here.

| Class | Requirement |
|---|---|
| **`W0`** | Returns in **under one second**, and **must never block the event loop**. A `W0` tool that performs real I/O — a filesystem walk, a subprocess, a store open, a directory digest — runs in the executor. `W0` means *takes no compute slot*; it does **not** mean *is instant*, and the two must not be conflated. |
| **`W1`** | Bounded by `limits.probe_max_images` (default 4) and `limits.probe_timeout_s` (default 300), inclusive of slot wait. The timeout must be reconciled with the host's own tool-call timeout — a server that outlives the host's patience holds the slot after the caller has given up. |
| **`W2` / `W3`** | **No latency requirement.** Submit-and-poll: the tool returns on submission and progress is polled. |
| **Connection** | The `tools/list` payload is spent every turn by every subagent; it is a budgeted resource, not free. |

The binding consequence: under §1.3's single shared connection, any handler that
blocks stalls **every** subagent, not just its caller. §5.5 already carves
`deploy_status {detail:"results"}` into the executor for exactly this reason;
that carve-out is the rule, not an exception.

## 1.7 Non-goals

- **No new science.** No new scorer, detector, or metric.
- **No agent-authored sbatch.** Raw `--slurm key=value` passthrough is
  deliberately not exposed. `parse_slurm_args` (`_cli/_cli_utils.py:336-375`)
  constrains **neither** dimension: any non-empty key is accepted and translated
  to a directive by `key.replace("slurm_","").replace("_","-")`, and values go
  through `ast.literal_eval` with a silent fallback to the raw string. The
  profile layer in §5 must therefore bound keys *and* values.
- **No in-process full-dataset execution.** `W3` always goes through the CLI as
  a subprocess, matching how the GUI launches runs.
- **No pipeline topology search.** Tuning optimizes parameters of a fixed
  topology; "which operations" stays the agent's decision.
- **No HTTP transport in v1.** stdio only. The tool layer is written
  transport-agnostically so HTTP remains addable, but nothing in v1 depends on
  it.
- **No SLURM cancellation of another session's jobs.** Cancel is scoped to runs
  the server allocated, via the existing generation fencing.
# PhenoTypic MCP Server — §2 State, Identity, and Workspace Layout

Status: **draft, reviewed once, revised**
Date: 2026-08-12

## 2.1 The rule

**Disk is the authority. The server holds no state that matters if lost.**

Every tool call resolves what it needs from the filesystem, acts, and writes
back. Killing the server loses no artifact; restarting it recovers its view by
reading the workspace. This is what makes agent-side fan-out safe — subagents
interleaving calls into one process cannot corrupt shared in-memory state,
because nothing in memory is authoritative.

Four things live in memory, each derived or immutable:

| Held | Lifetime | Why it is safe |
|---|---|---|
| `OperationRegistry` singleton (`get_registry()`) | process | Immutable after `discover()`; a pure function of installed code |
| `LocalComputeSlot` + queue depth (§1.5) | process | Process-local scheduling; reconciled against disk at startup |
| Parsed server config | process | Read-only, reloaded on restart |
| **One `RunRegistry` instance** | process, rehydrated at startup | **Not** authoritative — the authority is `gui_launch_owner.json` on disk plus an interprocess lock. The instance is a cache and a mutation gateway. |

The `RunRegistry` deserves the explicit callout because an earlier draft of this
section listed only three items and claimed "there is none to corrupt". That
overstated it. The registry *is* in-memory state; what rescues the claim is that
every correctness-critical operation (`allocate`, `compare_and_set`) takes an
interprocess file lock and re-reads the on-disk owner record inside it, so two
processes — or one process after a restart — converge on the same truth.

**Rehydration policy.** The server constructs exactly one `RunRegistry` bound to
the workspace and runs `rehydrate_from_sandbox(max_depth=3)` **once at startup**,
before accepting requests, then keeps it warm. It is not rebuilt per call: that
walk is a synchronous filesystem scan whose own docstring
(`gui/shell/_runs_registry.py:770-776`) warns it can be slow "on a sandbox with
thousands of plate folders". Since §1.3 says the server may be restarted often,
this cost recurs per restart — `workspace_info` reports the scan duration so a
pathological workspace is visible rather than mysterious.

## 2.2 Identity is a sandbox-relative path

**Not** a synthetic id, not a UUID, not a database row.

```
pipeline_id  =  "pipelines/edge-v3.json.pht-pipe"
study_id     =  "studies/edge-v3-tpe"
run_id       =  "runs/2026-08-12-plateA"
```

This follows the convention the GUI established: `RunRegistry` uses the
sandbox-relative output path as `run_id` (`gui/tune/_deploy.py:43-44`), and the
builder treats a config file on disk as the unit of persistence, with no project
record at all.

Why this and not synthetic ids:

- The agent can `ls` the workspace and see its own work. So can you.
- A restarted server re-derives every id by walking the tree.
- Ids paste cleanly between sessions, into the GUI, and into a shell.
- No id-allocation table means no id-allocation race between subagents.

### What path-identity costs — stated plainly

Path identity is **not** free of consequence, and two cases matter:

**Renaming a run directory invalidates its owner record.**
`_read_owner_record` (`gui/shell/_runs_registry.py:932-957`) rejects a record
when the persisted `rel_path` disagrees with the freshly computed one (line
952), and `rel_path` is recomputed from the current directory name on every scan
(`_discover_output_dirs`, line 709). So renaming `runs/plateA` →
`runs/plateA-v2` discards the record and the run degrades to the manifest-only
fallback: status limited to `{complete, failed, unknown}`, no PID, no
generation, no CAS. **Renaming a run or study directory is therefore
unsupported while it is live**, and `workspace_list` flags any directory whose
owner record failed identity checks rather than silently showing it as
`unknown`.

**Already-submitted SLURM work pins absolute paths.** A generated sbatch script
embeds `Path(pipeline_path).absolute()` (`_cli/_cli_staged_slurm.py:129`), and
`job_metadata.json` stores the absolute pipeline path
(`_cli_staged_slurm.py:462-464`). A job that sits in the queue for hours or days
will fail at execution if the workspace was moved or renamed meanwhile —
independent of anything the MCP id scheme does. **The workspace root must be
immutable while any submitted job is in flight.** `workspace_info` reports
whether in-flight work exists, and the server refuses to start when its
configured root differs from the root recorded in a live run's `job_metadata`.

So the accurate claim is narrower than "ids survive anything": ids are stable
across server restarts and across sessions, and they are *not* stable across
directory renames or workspace relocation.

**Collision policy.** `pipeline_put` and `tune_put_spec` take an explicit `name`
and **fail loudly** if the target exists, unless `overwrite: true`.
Auto-suffixing is deliberately rejected: a subagent that silently writes
`candidate_1` when it believes it wrote `candidate` will then tune the wrong
artifact. When the orchestrator fans out, it hands each subagent a distinct name
— the same discipline as handing each a distinct branch.

## 2.3 Workspace layout

The workspace root is a `SandboxRoot` (`gui/shell/_sandbox.py:62-90`). Every
path argument in every tool resolves through it; anything landing outside is
rejected before the tool does work.

**Root selection:** `phenotypic-mcp --workspace <path>`, defaulting to the
server process's CWD. Because subagents share one server (§1.3), there is
exactly one CWD and the default is unambiguous. `workspace_info` always echoes
the resolved root, and the server logs a startup warning when the root contains
`.git` — a source checkout is a plausible launch directory and a poor place to
accumulate run outputs.

```
<workspace>/
├── pipelines/
│   └── <name>.json.pht-pipe          # ImagePipeline.to_json()
├── tune/
│   └── <name>.setup.json.pht-tune    # authored TuningSpec
├── assays/
│   └── <dataset>.assay.json          # §9.3 assay profile
├── subsets/
│   └── <name>.subset.json            # §10.2 development subset
├── campaigns/
│   └── <name>/campaign.json          # §8.2 the agreed plan
├── .phenotypic-mcp/
│   ├── lineage.jsonl                 # §2.5
│   ├── plans/<token>.json            # §5.4 plan + promotion token records
│   ├── probes/<pipeline>/            # §3.2 probe outputs
│   └── subset-staging/<digest>/      # §10.3.1 materialized subset dirs
├── studies/
│   └── <name>/                       # a tune output_dir
│       ├── trials.parquet            # NOTE: output root, not deliverables/
│       ├── deliverables/
│       │   ├── tuning_spec.json.pht-tune     # RESOLVED spec (CLI-owned)
│       │   ├── best_pipeline.json.pht-pipe
│       │   ├── best_params.json
│       │   ├── param_importance.json
│       │   ├── generalization.json
│       │   └── pareto/…                      # multi-objective only
│       ├── .pht-tune-cache/{run.json,study.db,splits/split.json}
│       └── .phenotypic/progress/gui_launch_owner.json
└── runs/
    └── <name>/                       # a `python -m phenotypic` output_dir
        ├── deliverables/…            # measurements, dashboard, qc, README
        ├── results/<dataset>/{hdf,measurements}/
        └── .phenotypic/
            ├── processing_state.json
            ├── processing_events.log
            └── progress/{manifest.json,failures.jsonl,job_metadata.json,
                          slurm_jobs.jsonl,run_completion.json,
                          gui_launch_owner.json}
```

Only `pipelines/`, `tune/`, `assays/`, `subsets/`, `campaigns/`, `studies/`,
`runs/` are this server's invention.
**Everything inside `studies/<name>/` and `runs/<name>/` is written by the
existing engines**, at paths owned by `sdk_/_io_constants.py`. The server never
hand-joins a filename; it calls the helper (`tuning_spec_path`,
`best_pipeline_path`, `manifest_json_path`, and the `resolve_*` variants for
legacy tolerance).

Typed suffixes are mandatory and never spelled literally — `.json.pht-pipe`
(`CONFIG_SUFFIX_PIPELINE`) and `.json.pht-tune` (`CONFIG_SUFFIX_TUNING`),
applied via `ensure_typed_json_suffix` and matched via `matches_any_suffix`
(never `Path.suffix`, which sees only the trailing `.pht-tune`).

**Nested workspaces are safe but unadvertised.** Two `SandboxRoot`s that overlap
(an MCP workspace inside a broader GUI sandbox) will both enumerate the same run
directories, since `SandboxRoot.root` is frozen per-instance with no
cross-instance awareness. Launch correctness is preserved regardless, because
`exclusive_path_lock` and the owner-record path key off the **absolute**
`output_dir`, not the sandbox root. Overlap therefore duplicates *listings*, not
*claims*. The server does not forbid it; `workspace_info` reports the root so
overlap is diagnosable.

## 2.4 Run records: reuse `RunRegistry`

Both `studies/` and `runs/` entries are runs in the existing sense, and both
register through `RunRegistry`:

```python
record = registry.allocate(
    run_id=rel_path, mode="slurm"|"local", output_dir=…, rel_path=…,
    command_digest=sha256(argv), status="submitting"|"queued",
)
handle = runner.start(run_id, argv, output_dir=…, generation=record.generation)
registry.compare_and_set(run_id, record.generation,
                         expected_statuses=…, status=…, pid=…, log_paths=…)
```

This buys four properties the server would otherwise build:

1. **Interprocess locking** on allocation (`exclusive_path_lock`), so two
   subagents cannot both claim one output directory.
2. **Nonterminal-generation rejection** — a second launch against a live output
   directory is refused rather than racing it.
3. **Generation fencing** (`compare_and_set`) — a stale caller cannot mutate a
   record that has since been re-launched.
4. **Boot recovery** — `rehydrate_from_sandbox` rebuilds the view by scanning
   for owner records and manifests.

### The limit of boot recovery, stated honestly

`gui_launch_owner.json` is written **only** by code paths that call
`RunRegistry.allocate` — today `gui/run_console/_callbacks.py:1915,2027,2121`
and `gui/tune/_deploy.py:46`, and tomorrow this server. Neither
`phenotypic._cli` nor `run_tuning` writes one.

So a run you start by typing `python -m phenotypic` yourself on the login node —
an expected coexistence, since §1.3 puts you and the agent on the same node — is
**not** fully recoverable. It surfaces only through the manifest fallback
(`_read_status_from_manifest`), capped to `{complete, failed, unknown}` with no
PID, generation, or CAS. Full-fidelity recovery covers runs this server (or the
GUI) launched; everything else is read-only observation. `workspace_list` labels
each row with its evidence source so the difference is visible.

**Consequence, and it is a good one:** an agent-launched run appears in the
GUI's Recent Runs, and a GUI-launched run is visible to the agent. That interop
is free and worth preserving.

## 2.5 Lineage

"Build several pipelines in parallel, tune them, deploy the winner" is only
useful if the agent can answer *which pipeline produced which study produced
which winner* — including after a context compaction or a server restart.

Most of that chain is already reconstructible from existing artifacts, and the
journal should not pretend otherwise:

| Hop | Recoverable today? | From |
|---|---|---|
| pipeline → study | **yes** | `TuningSpec.pipeline` is *embedded*, not referenced (`tune/_spec.py:165`); the resolved spec at `deliverables/tuning_spec.json.pht-tune` can be content-hashed and matched against `pipelines/*` |
| study → its winner | **yes** | `deliverables/best_pipeline.json.pht-pipe` |
| run → pipeline | **yes** | `job_metadata.json` `PIPELINE_PATH` |
| **dataset → study** | **no** | `TuningSpec` has **no dataset field** (`tune/_spec.py:162-171`); `--images` is a launch-time CLI argument recorded nowhere in the resolved spec |
| exported winner → the `pipelines/` copy the agent then deployed | **no** | nothing records the copy |

Two hops are genuinely unrecoverable, and both are ones an agent traverses
constantly. The journal exists for those, plus cheap chronology — not because
the whole chain is otherwise lost.

**The dataset hop is load-bearing for `campaign_status.comparable` (§8.3)**: two
arms tuned against different image sets cannot be honestly ranked, and nothing
in the existing artifacts records which images a study used. So the `tune.start`
lineage event carries the **subset** it ran on — which §10.3.1 also makes the
tool argument, so the two cannot disagree:

```json
{"ts":"…","event":"tune.start","id":"studies/edge-v3-tpe",
 "parent":"pipelines/edge-v3.json.pht-pipe",
 "subset":{"id":"subsets/plates-dev-24.subset.json","digest":"sha256:77b2…","n_images":24}}
```

`digest` needs a **directory-level fingerprint helper, which does not exist** —
`bytes_fingerprint` / `file_fingerprint` (`sdk_/_io_constants.py:154,166`) and
`pipeline_content_digest` are all single-file. A stable digest over
`(relative path, size, mtime_ns)` for each image, sorted, is sufficient and
cheap; it is listed as new work in §7 P3 rather than assumed.

`<workspace>/.phenotypic-mcp/lineage.jsonl`, append-only:

```json
{"ts":"2026-08-12T14:02:11Z","event":"pipeline.put","id":"pipelines/edge-v3.json.pht-pipe","digest":"sha256:9c1e…","parent":null,"agent":"subagent-B"}
{"ts":"2026-08-12T14:07:40Z","event":"tune.start","id":"studies/edge-v3-tpe","parent":"pipelines/edge-v3.json.pht-pipe"}
{"ts":"2026-08-12T15:31:02Z","event":"tune.export_best","id":"pipelines/edge-v3-tuned.json.pht-pipe","parent":"studies/edge-v3-tpe","trial":47,"score":0.081}
{"ts":"2026-08-12T15:44:19Z","event":"deploy.start","id":"runs/2026-08-12-plateA","parent":"pipelines/edge-v3-tuned.json.pht-pipe"}
```

**Digest format.** `digest` is `f"sha256:{...}"`, matching `bytes_fingerprint` /
`file_fingerprint` (`sdk_/_io_constants.py:154-179`). Note that
`pipeline_content_digest` (`_cli/_cli_staged_resume.py:64-66`) returns a **bare**
hexdigest with no prefix — the two are the same hash over the same bytes, but a
consumer comparing a lineage `digest` against a resume digest must strip the
`sha256:` prefix first. They do not string-compare equal as written.

**Writes are offloaded.** The append reuses the repo's `atomic_append`
(`_cli/_cli_file_locking.py:167-193`), whose `file_lock` spins in a synchronous
`while True: flock / time.sleep(0.01)` retry with a 30 s timeout — deliberately,
for slow NFS/Lustre. Calling that from a request-handling coroutine would stall
**every** concurrent tool call, including the supposedly-unbounded `W0` ones, for
up to the timeout. Lineage writes therefore go through `asyncio.to_thread`, never
inline.

`agent` is an **optional, self-declared** label. It is provenance for humans
reading the journal, never an authorization or routing input — a subagent that
lies about its name changes nothing about what it may do.

Worst case the journal is truncated and the artifacts still stand alone, with
three of the four hops reconstructible.

## 2.6 Concurrency summary

| Shared resource | Guard | Provided by |
|---|---|---|
| Output directory claim | interprocess file lock + nonterminal-generation check | `RunRegistry.allocate` |
| Run record mutation | generation-fenced CAS | `RunRegistry.compare_and_set` |
| Local image compute | one process-wide `LocalComputeSlot` | new, §1.5 |
| Artifact writes | `atomic_write_text` + explicit-overwrite policy | `sdk_` helpers, §2.2 |
| Plan / promotion tokens | `atomic_write_text`; single-use CAS on `consumed_by` **under `exclusive_path_lock`** — `allocate`'s interprocess idiom, **not** `compare_and_set`'s in-process `threading.Lock`, since §2.3 treats overlapping server instances over one workspace as anticipated | new, §5.4 |
| Subset staging dirs | Keyed by subset digest, so concurrent arms share one directory instead of racing; created idempotently | new, §10.3.1 |
| `campaign.json` mutation | `atomic_write_text` + a status transition guard: `approve` and any amendment CAS on `status`, and **`campaign_start` snapshots the campaign it launched** rather than re-reading mid-fan-out | new, §8.3 |
| Lineage journal | `atomic_append` under file lock, **via `asyncio.to_thread`** | existing pattern, §2.5 |
| Operation registry | immutable after discovery | `get_registry()` |

No tool mutates another tool's in-flight artifact. The one cross-tool write is
`tune_export_best`, which writes a *new* pipeline file rather than editing the
base — matching `build_pipeline`, which deep-copies rather than mutating
(`tune/_evaluation/_builder.py:384`).

## 2.7 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-2.1 GUI preset interop~~ → **visible `tune/` only**. Agent-authored specs
  keep readable names at `<workspace>/tune/<name>.setup.json.pht-tune`; the GUI
  reaches them through its Browse… escape hatch. Writing additionally into
  `.phenotypic-gui/presets/tune/` would double every artifact, and writing only
  there would hand the agent content-addressed filenames (`<stem>-<sha256[:20]>`),
  losing the readable-id property §2.2 is built on.

- ~~OQ-2.2 workspace root~~ → `--workspace`, defaulting to CWD (unambiguous
  because subagents share one server), with the root always echoed by
  `workspace_info` and a warning when it is a git checkout (§2.3).
# PhenoTypic MCP Server — §3 Tool Contract: Catalog, Pipeline, Workspace

Status: **draft, reviewed once, revised**
Date: 2026-08-12

## 3.0 Conventions binding every tool

**Naming.** `<group>_<verb>`, flat, no dots — `pipeline_put`, never `pipeline.put`.
32 tools in nine groups: catalog (3), pipeline (5), workspace (4),
assay (2, §9.3), subset (3, §10.3), tune (5, §4), deploy (3, §5),
**campaign (5, §8)**, promotion (2, §10.5).

**Every tool returns the same envelope.**

```json
{
  "ok": true,
  "data": { },
  "issues": [ {"severity":"error|warning|advisory","code":"…","message":"…","path":"…","hint":"…"} ],
  "routed": {"class":"W1","routed_to":"local","reason":"…","queue_position":null}
}
```

`issues` appears even on success — `severity: "advisory"` carries the GUI's
non-blocking hints without failing the call. `routed` appears only on `W1`+.

**Errors are values, not exceptions.** A failed call returns `ok: false` with
issues so the agent can correct itself. Protocol errors are reserved for
malformed calls. Codes are enumerated in §6.

**Paths** resolve through `SandboxRoot`; escapes are rejected before any work.

**Ids are sandbox-relative paths** (§2.2). Bare-stem sugar is accepted **only
for the typed-suffix artifacts**, where `matches_any_suffix` can resolve it —
never `Path.suffix`, which sees only `.pht-tune`:

| Id | Bare stem? | Resolution |
|---|---|---|
| `pipeline_id` | yes | `pipelines/<stem>` + `PIPELINE_CONFIG_SUFFIXES` |
| `spec_id` | yes | `tune/<stem>` + `TUNING_CONFIG_SUFFIXES` |
| `subset_id`, `assay` | yes | fixed `.subset.json` / `.assay.json` suffix |
| `study_id`, `run_id`, `campaign_id` | **no** | directories, no suffix to match — the full sandbox-relative form is required |

**Token discipline.** List tools return compact rows; full JSON schemas come
only from the detail tool, one operation at a time. No tool returns an unbounded
measurement table — dataframes are summarized, with a parquet path for more.

---

## 3.1 Catalog group (all `W0`)

### `catalog_operations`

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `category` | `str?` | `null` | `Enhancer, Detector, Refiner, Corrector, Measure, Grid, Post, Filter, Model, Edge Correction, quality_check, Prefab, **Scorer**, **Strategy**` |
| `limit` | `int` | `100` | Row cap; the response reports `truncated` and the total |
| `query` | `str?` | `null` | Substring over name + docstring summary |

```json
{"ok":true,"data":{"operations":[
  {"name":"OtsuDetector","category":"Detector","summary":"Global Otsu threshold on detect_mat.",
   "n_params":3,"has_nested_operations":false},
  {"name":"FilamentousFungiDetector","category":"Detector","summary":"Two-stage fungal colony detector.",
   "n_params":20,"has_nested_operations":true}]}}
```

`summary` is the **first sentence** of `OperationInfo.docstring`.
`has_nested_operations` is `any(p.is_operation or p.is_pipeline …)` — it tells
the agent a param needs another operation as its value, which the JSON schema
cannot say (below).

**Discovery breadth (resolved OQ-3.1).** `OperationRegistry.discover()` walks
`enhance, detect, refine, correction, measure, grid, post, analysis`, but
`_find_class_in_phenotypic` *also* resolves `prefab`, `tune`, `tune.score`,
`tune.strategy`, and `detect.nn`. **The two lists are reconciled to one shared
constant**, so the agent can reach `MicroSamDetector` and the prefab pipelines.
Without `detect.nn` the entire staged-GPU path would be unreachable from the
server, which would silently amputate a headline CLI capability. Reconciling is
new work, listed in §7 P3.

**Scorers and strategies are catalog citizens too.** `catalog_operations
{category:"Scorer"}` lists `QCScorer`, `SupervisedScorer`,
`ReferenceFreeScorer`, `CompositeScorer`; `catalog_operation_detail
{name:"QCScorer"}` returns its `model_json_schema()` like any operation. Without
this an agent authoring `tune_put_spec` has no way to learn the
`{"check":{"metadata": <path>}}` shape except by guessing and reading
did-you-mean errors — `tune_space`'s `scorers_available` gives a class name and
an English requirement, not a schema.

### `catalog_operation_detail`

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | — | Operation, **scorer, or strategy** class name |
| `verbose_descriptions` | `bool` | `false` | Full docstring text per param instead of first sentence |

```json
{"ok":true,"data":{
  "name":"BlurGauss","category":"Enhancer","module":"phenotypic.enhance",
  "doc":"Gaussian blur on detect_mat …",
  "json_schema":{ },
  "params":[
    {"name":"sigma","type":"float","default":2.0,"required":false,
     "description":"Standard deviation of the Gaussian kernel in pixels.",
     "constraints":{"exclusiveMinimum":0.0},
     "is_operation":false,"is_pipeline":false,"is_list":false,"is_optional":false,
     "choices":null,"column_ref":null}],
  "layers_modified":["detect_mat"]}}
```

`json_schema` is `cls.model_json_schema()` verbatim — the contract
`_base_operation.py:192` designates. Field `description`s land there
automatically via `apply_docstring_descriptions`, **verified working**. So
docstring quality is API quality: a param with no `Args:` entry reaches the agent
undocumented.

Three corrections over the first draft, each verified:

- **`constraints` uses JSON Schema keywords, not pydantic `Field()` kwargs.**
  `FlattenIllumination.sigma`, declared `Field(200.0, gt=0.0)`, reports
  `"exclusiveMinimum": 0.0`. The projection passes the real keywords through
  rather than inventing a `gt`/`le` spelling.
- **Descriptions are long.** `BlurGauss.sigma`'s real description runs ~180
  characters across four sentences, and `FilamentousFungiDetector` has 20 params.
  The default is therefore the **first sentence**, with the full text behind
  `verbose_descriptions` — otherwise one detail call can return several KB of
  prose, contradicting §3.0.
- **No `tunable` field.** An earlier draft advertised a per-param suggested
  domain here. `ParamInfo` carries no such data, and `infer_search_space` /
  `pipeline_targets` both require a *positioned pipeline* — knobs are keyed
  `"<position>.<field>"`. Tunability is a property of an operation *in a
  pipeline*, not of a class, so it belongs to `tune_space` (§4.1) and is removed
  from here.

Two gaps the raw schema cannot express, filled from `OperationInfo`/`ParamInfo`:

1. **`OperationField` erases to `Any`**, so operation-valued params appear as
   untyped `{}` in the schema. `is_operation`/`is_pipeline` come from the
   `_OperationFieldMarker` walk instead.
2. **`NdArrayField`'s schema is a bare `{"type":"array","items":{}}`** with no
   shape or dtype. Flagged `type: "ndarray"`; not agent-authorable in practice.

### `catalog_measurements`

| Arg | Type | Meaning |
|---|---|---|
| `measurer` | `str?` | Restrict to one `MeasureFeatures` class |

**Header derivation must dispatch on `header_scheme()`.** A blanket
`get_headers()` call is wrong, and for at least one measurer it raises:

```
SIZE.header_scheme()     -> "static"   -> get_headers()  -> ['Size_Area', …]
TEXTURE.header_scheme()  -> "texture"  -> get_headers()  -> TypeError:
                                          missing 1 required positional argument: 'scale'
TEXTURE.get_headers(5)   -> ['Texture_AngularSecondMoment-deg000-scale05', …]
```

`TEXTURE` (`schema/_texture.py:144-181`) overrides `get_headers(scale,
matrix_name=None)` with **no default for `scale`**, and `MeasureTexture` emits
one header per (member × angle × scale) — 130 columns for `scale=[5,10]`, not the
13 base labels. So the projection:

1. reads `header_scheme()` per `MeasurementInfo` class (`static` /
   `metric_qualified` / `texture`, per `schema/CLAUDE.md`);
2. for `static`, calls `get_headers()`;
3. for `texture`, calls `get_headers(scale, matrix_name)` once per entry in the
   **live measurer instance's** `scale` list and merges;
4. for `metric_qualified`, uses the qualified-header path.

The source of the class list is the public instance method
`MeasureFeatures.get_measurement_infoclasses()` (`abc_/_measure_features.py:333`),
which is genuinely instance-dependent — `MeasureColor` includes or excludes
members based on `self.include_XYZ` / `self.include_xy`.

**Do not model this on the README generator.** An earlier draft cited "the same
measurer→`MeasurementInfo` mapping the README generator uses". No such reusable
mapping exists: `_cli_readme_generator.py:140-235` iterates `pipeline._meas` and
renders `member.value` directly, never expanding texture headers — so it
under-reports texture columns in generated READMEs today. Reusing it would
inherit the bug. (Worth fixing there separately; out of scope here.)

`desc` per column is the enum member's `desc` — the text users see. `bio_desc`
is **never** returned; it is human-authored and frequently empty.

---

## 3.2 Pipeline group

### `pipeline_put` (`W0`)

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | — | Workspace name |
| `pipeline` | `object?` | — | Spec below. Mutually exclusive with `from_prefab` |
| `from_prefab` | `str?` | `null` | Materialize a shipped prefab (e.g. `"FilamentousFungiPipeline"`) as a new workspace pipeline |
| `overwrite` | `bool` | `false` | Required to replace |
| `dry_run` | `bool` | `false` | Validate and return issues without writing |

```json
{"name":"edge-v3",
 "pipeline":{
   "desc":"Otsu on blurred detect_mat, size+shape measured.",
   "nrows":8,"ncols":12,
   "ops":[{"class":"BlurGauss","params":{"sigma":2.0}},
          {"class":"OtsuDetector","params":{"ignore_zeros":true}}],
   "meas":[{"class":"MeasureSize"},{"class":"MeasureShape"}],
   "post":[],"filters":[],"model":null}}
```

A param whose value is itself an operation nests the same `{class, params}`
envelope. Note the **write** shape carries `params` at both levels, while the
**error** path does not — `OperationField`'s `BeforeValidator` unwraps the inner
`params` before validating, so a bad nested field reports as
`ops[3].params.inoculum_detector.sigmaa` (§6.2, reproduced there):

```json
{"class":"FilamentousFungiDetector",
 "params":{"inoculum_detector":{"class":"OtsuDetector","params":{"ignore_zeros":true}}}}
```

**Prefab-first needs a materialization step.** §9.4's procedure opens with
"probe the candidate prefabs", and §2.2 requires a `pipeline_id` to be a
sandbox-relative path — so a bare class name from `catalog_operations` is not
probeable. `pipeline_put {name, from_prefab}` instantiates the prefab and writes
it to `pipelines/<name>.json.pht-pipe`, which is then an ordinary pipeline:
patchable, probeable, tunable. Without it the flagship workflow has no valid
first call.

**Construction path — public constructor, no private imports.** For each entry
the server resolves the class and validates params, then hands the resulting
*instances* to the public constructor:

```python
cls  = SerializablePipeline._find_class_in_phenotypic(entry["class"])
inst = cls.model_validate(entry["params"])
pipe = ImagePipeline(ops=[…instances…], meas=[…], post=[…], nrows=…, ncols=…)
```

`ops` declares `validation_alias=AliasChoices("ops","pipe_cfgs")` with a
`field_validator(mode="before")` that runs `_normalize_operation_collection` /
`_make_unique` **transparently**, so duplicate classes are deduped to
`BlurGauss`, `BlurGauss_1`, … with the caller importing nothing private.

An earlier draft proposed calling `_make_unique` directly and round-tripping a
synthetic `pipe_cfgs` envelope through `from_json`. That reached into a private
staticmethod in a private module — exactly the coupling §1.4 promotes
`_services` to avoid. Verified: both paths produce byte-identical `to_json()`.
Verified also that the naive shortcut *does not* work — passing raw
`{"class":…,"params":…}` dicts straight to the constructor raises
`ValidationError` (`extra_forbidden`), so per-entry resolution is unavoidable
and the two calls above are the minimum.

**Ordering** is insertion order = execution order within `ops`. The engine
enforces no stage ordering — `ops`/`meas`/`post` are three dicts run in three
phases. The server runs the GUI's DAG validator, which requires the adapter
`from_pipeline_dag(pipeline) -> BuilderState` (`gui/builder/_conversion_dag.py:676`)
before `validate(state)` (`gui/builder/_validation.py:98`), and surfaces
`stage_order_hint` as **advisory**, matching the GUI. It does not block.

```json
{"ok":true,
 "data":{"pipeline_id":"pipelines/edge-v3.json.pht-pipe","digest":"sha256:9c1e…",
         "n_ops":2,"n_meas":2,"execution_order":["BlurGauss","OtsuDetector"],
         "produces_columns":["Size_Area","Shape_Circularity","…"],
         "requires_gpu":false}}
```

`produces_columns` uses the `header_scheme()` dispatch of §3.1 over the **built
pipeline's live measurer instances** — which is why it is computed after
construction, when `MeasureTexture.scale` is actually known. It lets the agent
confirm a pipeline yields the columns its scorer needs before running anything.
`requires_gpu` comes from `pipeline_requires_gpu` and drives routing (§5).

**Validation errors** are structured, using `extra="forbid"` plus
`difflib.get_close_matches` — the same three-line stdlib technique used at
`tune/_spec.py:80-87`. (That is a pattern to copy, not a shared utility; there is
no exported did-you-mean helper in the repo.)

```json
{"ok":false,"issues":[
  {"severity":"error","code":"unknown_param","path":"ops[0].params.sigmaa",
   "message":"BlurGauss has no parameter 'sigmaa'.","hint":"Did you mean 'sigma'?"}]}
```

### `pipeline_patch` (`W0`)

| Arg | Type | Meaning |
|---|---|---|
| `pipeline_id` | `str` | Target |
| `edits` | `array` | Ordered edits, applied atomically |
| `dry_run` | `bool` | Validate only |

Edit kinds: `insert_op {slot, index, class, params}`, `remove_op {slot, index}`,
`move_op {slot, from, to}`, `set_params {slot, index, params, merge=true}`,
`set_grid {nrows, ncols}`, `set_model {class, params|null}`;
`slot ∈ {ops, meas, post, filters}`.

**Index semantics, pinned** — ambiguity here produces silently-wrong pipelines
rather than loud errors:

- `index` is 0-based over the slot's current ordered list.
- `insert_op.index` follows Python `list.insert`: clamped to `[0, len]`; `-1`
  means "before the last element"; omitting it appends.
- `remove_op.index` and `set_params.index` must be in `[0, len)` — out of range
  is an **error**, never a clamp.
- `move_op.from` and `move_op.to` must both be in `[0, len)` and `[0, len-1)`
  respectively — **out of range is an error, matching `remove_op`, never a
  clamp.** `to` is the index in the list **after** the source is removed. A move
  is remove-then-insert, so both a clamping and an erroring reading were
  defensible; erroring is chosen because an agent computing `to` against a
  pre-edit list (easy once an earlier edit in the same array has shifted indices)
  should get a hard stop, not a silently reordered pipeline.
- Edits apply in array order, each seeing the previous one's result.

All edits apply to an in-memory deep copy; **the file is written only if every
edit validates**, so a failing third edit leaves the artifact untouched.

```json
{"ok":true,"data":{
  "pipeline_id":"pipelines/edge-v3.json.pht-pipe","digest":"sha256:1d77…",
  "n_ops":3,"execution_order":["BlurGauss","FocusEdgePhase","OtsuDetector"],
  "produces_columns":["Size_Area","Shape_Circularity"],"requires_gpu":false,
  "diff":[{"kind":"insert_op","slot":"ops","index":1,"class":"FocusEdgePhase"}],
  "exploration":{"steps":3,"cap":12,"no_improvement_streak":0,
                 "tracked_signal":"objmap.num_objects",
                 "budget_note":"3 of 12 patches used"}}}
```

**`exploration` is what makes §8.7's loop runnable.** That section states step
and no-improvement caps "reported in the response", and the lineage journal
already records a `step` counter — but without this block an agent would have to
poll `workspace_lineage` and count rows to know it was on patch 11 of 12, which
nothing instructs it to do. `tracked_signal` names the metric the streak is
measured against, so "no improvement" is a defined claim rather than a vibe.

### `pipeline_diff` (`W0`)

| Arg | Type | Meaning |
|---|---|---|
| `a`, `b` | `str` | Two pipeline ids |

Structural diff between two pipelines: ops added/removed/reordered, and
per-parameter value changes. §1.5 lists "diff two pipelines" as a `W0` example,
and comparing candidates before probing them saves serialized `LocalComputeSlot`
time. Also the natural way to see what tuning actually changed:
`pipeline_diff {a:"edge-v3", b:"edge-v3-tuned"}`.

### `pipeline_get` (`W0`)

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `pipeline_id` | `str` | — | Target |
| `format` | `"summary" \| "envelope" \| "raw"` | `"summary"` | Compact rendering, the agent-facing envelope, or the literal file |

`summary` lists ops with **non-default params only**, plus `produces_columns`
and `requires_gpu`.

**`envelope` is projected, not the file verbatim** — this matters because the two
shapes differ. On disk, `_serialize_pipeline_config`
(`_serializable_pipeline.py:153-155`) writes operations under **`pipe_cfgs`** as a
**name-keyed dict** (`{"BlurGauss": {...}, "BlurGauss_1": {...}}`), whereas every
agent-facing example in this spec — and `pipeline_put`'s input contract — uses
**`ops` as an ordered array**. `envelope` returns the `ops`-array form, so
`pipeline_get` → edit → `pipeline_put` round-trips without the agent having to
know about `pipe_cfgs` or about internal instance names.

`raw` returns the file bytes for the rare case someone wants exactly what is on
disk. It is not the default precisely because handing an agent `pipe_cfgs` when
every documented example says `ops` invites malformed edits.

**Internal op names are not stable identifiers.** `_make_unique`
(`_image_pipeline_core.py:779-810`) recomputes suffixes from list position every
time a collection is normalized, so removing an earlier `BlurGauss` renames a
later `BlurGauss_1` to `BlurGauss`. Nothing agent-facing exposes these names —
addressing is by `(slot, index)` — and nothing in the server may key state off
them.

### `pipeline_probe` (`W1`)

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `pipeline_id` | `str` | — | Target |
| `subset_id` | `str` | — | Registered subset (§10.3.1). A raw path is accepted **only** when no subset exists yet, and the response says which was used |
| `n_images` | `int` | `2` | Capped at `limits.probe_max_images` (default 4) |
| `sample` | `"first" \| "random"` | `"first"` | `random` uses a fixed seed |
| `stages` | `bool` | `false` | Per-operation evidence via `apply_with_intermediates` (§8.7) |
| `save_overlay` | `bool` | `false` | Render label overlays for a **human** reading the transcript |

**Executed in a persistent, killable probe worker subprocess** — not in the
server process.

This is a correction forced by review. The first draft said "in-process, bounded
by a wall-clock timeout", and that combination is not implementable safely:
`asyncio.wait_for` cannot preempt a CPU-bound or C-extension-blocked call, so
the timeout would not fire; and offloading to a thread lets the caller return
while the runaway thread keeps running — if the `LocalComputeSlot` releases only
when that future resolves, every later probe from every subagent deadlocks
behind an op Python cannot kill. There is no precedent in the codebase for
bounding an in-process `pipeline.apply()` by wall clock, because nothing has
needed to.

So:

- The server owns **one** probe worker subprocess, spawned lazily and kept warm
  (paying the `import phenotypic` cost once, not per probe).
- A probe sends **`{pipeline_path, image_paths, options}` as length-prefixed
  JSON over a dedicated pipe pair — never the worker's stdout.** The engine opens
  `tqdm` bars when `verbose`/`benchmark` is set, and at least one operation module
  does a bare `print()` (`detect/nn/_helper/_checkpoint_manager.py`); §6.4 already
  treats third-party operation output as untrusted content flowing through the
  server. Any of that on a stdout protocol channel corrupts the stream for every
  subsequent probe until the worker respawns. The worker's stdout/stderr are
  captured to a log instead.
  Note there is no precedent to inherit: `LocalRunner` is fire-and-forget `Popen`
  plus a log tee, not a persistent request/response worker. It does **not** send a pipeline envelope or a pickled object: the worker
  calls `ImagePipeline.from_json(path)` itself, so reconstruction goes through
  exactly the same code path — and inherits exactly the same round-trip
  guarantees — as every other consumer. Sending a payload instead would make the
  probe the one place a pipeline reaches an engine without passing through disk,
  contradicting §2.1, and would raise a pickle-vs-JSON encoding question for
  `NdArrayField` and nested `OperationField` values that nothing else in this
  design has to answer.
- Timeout is `SIGKILL` + respawn. The slot is released on process death, so a
  runaway probe cannot wedge the server.
- A probe that OOMs kills the worker, not the server — which also bounds the
  cumulative-memory concern, since worker RSS resets on respawn.
- Holding the `LocalComputeSlot` == the worker is busy, so `W1` and local
  `W2`/`W3` still serialize against one another exactly as §1.5 requires.

```json
{"ok":true,
 "data":{"n_images":2,
   "per_image":[{"name":"plateA_01.tif","num_objects":94,"elapsed_s":3.4,
                 "rss_after_mb":812}],
   "measurements":{"n_rows":188,"columns":["Size_Area","Shape_Circularity"],
     "describe":{"Size_Area":{"count":188,"mean":412.3,"std":88.1,"min":95,"max":901}},
     "parquet":".phenotypic-mcp/probes/edge-v3/measurements.parquet"},
   "benchmark":[{"process":"BlurGauss","type":"Operation","seconds":0.31,"rss_delta_mb":42.0}]},
 "routed":{"class":"W1","routed_to":"local","reason":"probe worker","queue_position":0}}
```

- **Never returns raw rows** — a `describe()` summary plus a parquet path.
- **Benchmark is opt-in in the engine**: `benchmark=True` at construction and an
  explicit `benchmark_results()` call; nothing persists it. The probe is the only
  place benchmarking is wired anywhere.
- `num_objects` reads `image.num_objects`, not the `objmap` accessor.
- **`rss_after_mb` is a snapshot, not a peak.** The only RSS instrumentation in
  the codebase is `_get_process_rss_mb()` (`_image_pipeline_core.py:813-824`), a
  point-in-time `psutil` read taken after each operation — there is no sampling
  thread anywhere. An OS peak (`ru_maxrss`) would be worse here, not better: it
  is a monotonic high-water mark for the **whole process since start**, and the
  probe worker is deliberately kept warm across images, so images 2–4 of a probe
  would all report image 1's peak. An earlier draft called this field
  `peak_rss_mb`, which promised a measurement nothing computes.
- Per-image timing is recorded to lineage so `deploy_plan` estimates have a real
  basis (§5.3).

**`stages: true`** runs `apply_with_intermediates` (`_image_pipeline_core.py:969`)
**with `output_dir=None`** — intermediates stay in memory as `Image` copies.
Passing an `output_dir` writes each snapshot to HDF5 and sets the dict value to
`None`, which would force a second pass re-reading from disk to compute any layer
statistic. With `n_images` capped at 4 the in-memory form is both simpler and
cheaper. It returns per-operation numeric evidence — the affordance
that makes incremental construction possible (§8.7):

```json
{"stages":[
  {"index":0,"op":"BlurGauss","layers_modified":["detect_mat"],
   "detect_mat":{"before":{"mean":0.41,"std":0.09,"p05":0.28,"p95":0.62},
                 "after" :{"mean":0.41,"std":0.04,"p05":0.35,"p95":0.48},
                 "pixels_changed_pct":99.8},
   "seconds":0.31},
  {"index":1,"op":"OtsuDetector","layers_modified":["objmap"],
   "objmap":{"num_objects":61,"area":{"mean":388,"std":141,"min":42,"max":2107}},
   "seconds":1.04}]}
```

Which layers each op touches comes from `_layers_modified_by`
(`_image_pipeline_core.py:100`), so the report shows only what actually changed.

**`meas`/`post` stages report a measurement diff, not a layer diff.**
`_layers_modified_by` returns `None` for a `MeasureFeatures` op — measurement
ops populate the table, not a layer — so a patched `MeasureShape` would otherwise
yield `layers_modified: []` and give the agent nothing to decide keep/revert on.
For those slots the stage row instead carries `columns_added`, `n_rows`, and a
`describe` of the new columns. §8.7's loop is uniform across slots only because
of this; without it, the loop would silently apply to `ops` alone.

**The evidence is numeric because the agent has no eyes.** It cannot look at a
`detect_mat` and see that the blur washed out the colony edges — but it can read
that `std` fell from 0.09 to 0.04 while `num_objects` dropped, and conclude the
enhancement destroyed the contrast the detector needed. Overlays serve a human
reading the transcript; **stage statistics serve the agent**, and confusing the
two produces a tool that looks informative and tells the agent nothing.

---

## 3.3 Workspace group (all `W0`)

### `workspace_info`

```json
{"ok":true,"data":{
  "workspace":"/scratch/alex/phenotypic-agent","workspace_source":"--workspace",
  "environment":"slurm",
  "environment_source":"auto: sbatch on PATH, squeue probe ok, 2 profiles",
  "slurm_profiles":[
    {"name":"cpu-bulk","partition":"batch","account":"exfab",
     "caps":{"max_time":"08:00:00","max_array":512,"max_cpus_per_task":32},
     "overridable":["time","cpus_per_task","mem_gb","njobs","n_workers"]}],
  "limits":{"probe_max_images":4,"probe_timeout_s":300,"local_slot_capacity":1},
  "tune":{"distributed_backend":"postgres","journal_backend_enabled":false},
  "in_flight":{"local":0,"slurm":2},
  "rehydrate_ms":184,
  "next_recommended":"pipeline_put",
  "workflow":{"assay":"assays/plates.assay.json","subset":"subsets/plates-dev-24.subset.json",
              "blocked":[],"note":"assay and subset exist; pipelines may be authored and probed"},
  "counts":{"pipelines":3,"tune_specs":3,"assays":1,"subsets":1,
             "campaigns":1,"studies":1,"runs":0},
  "active_subset":"subsets/plates-dev-24.subset.json"}}
```

`tune.journal_backend_enabled` reflects §7 P1: while `false`, a SLURM tune
request fails with an actionable error naming Postgres rather than submitting
into the SQLite corruption case. `in_flight` and `rehydrate_ms` make the
workspace-immutability rule (§2.2) and rehydration cost visible.

**`next_recommended` and `workflow` make the ordering discoverable from data.**
The workflow (§8.1) is otherwise taught only by the skills, so an agent that
never loads `phenotypic-assay-triage` has no contract-level signal that an assay
and a subset come before tuning. `workspace_info` is the natural first call, so
it answers *what should I do next* — `assay_put` when `counts.assays == 0`,
`subset_generate` when there is no subset, and so on, with `blocked` naming any
tool that would refuse right now and why. Skills teach judgment; this makes the
bare ordering legible without them.

**`blocked` covers only preconditions already computed for this response** —
artifact existence and configuration, never per-call validation:

| Blocked when | Tools |
|---|---|
| no subset registered | `tune_start`, `campaign_put`, `deploy_plan`, `deploy_start` |
| `tune.journal_backend_enabled: false` and no Postgres configured | SLURM-routed `tune_start` |
| `environment: "local"` with no profiles | anything requiring a `compute.profile` |
| a live run holds the workspace root (§2.2) | nothing — reported in `in_flight`, not blocked |

It does **not** pre-evaluate scorer `availability()`, profile caps, or token
staleness. Those need the call's own arguments, and running them speculatively
would turn a cheap `W0` orientation call into real work — contradicting §3.0's
token discipline for the one tool an agent calls first.

`next_recommended` is a single string and therefore a simplification: once an
assay and subset exist, several tools are equally reasonable. It is advisory
ordering, not a scheduler.

### `workspace_list`

| Arg | Type | Meaning |
|---|---|---|
| `kind` | `"pipelines" \| "tune_specs" \| "assays" \| "subsets" \| "campaigns" \| "studies" \| "runs" \| "all"` | Filter |
| `status` | `str?` | For studies/runs: `RunRegistry` status |

`tune_specs` was missing from the first draft, which left authored specs
(`<workspace>/tune/*.json.pht-tune`) unenumerable — an agent could write three
and then have no way to list them. Studies and runs come from `RunRegistry`
after startup rehydration, each row labelled with its evidence source so
owner-record rows are distinguishable from manifest-only ones (§2.4).

### `workspace_cancel`

`{id, reason?}`. Routes to `cancel_generation` / `cancel_staged_jobs` /
`LocalRunner.stop`. **Scoped to runs this server holds a `RunRegistry` record
for**, so an agent cannot reach another session's work.

### `assay_put` / `assay_get` (`W0`)

The assay profile (§9.3) needs a call path, or its validation contract describes
logic nothing invokes.

`assay_put {dataset, traits, overwrite?, dry_run?}` writes
`assays/<dataset>.assay.json`. It validates the **envelope only** (§9.3.5): each
trait has `value` and `source`; `source ∈ {human, probe, metadata, inferred}`;
`evidence` present and its `probe_ref` resolvable when `source: "probe"`. It
never validates a biological value, and it **preserves unknown trait keys
verbatim** rather than rejecting them — which means this is the one place in the
server that must *not* use pydantic's `extra="forbid"`. The model is an explicit
`traits: dict[str, TraitEnvelope]`, so unknown keys are ordinary data, not extra
fields.

`assay_get {dataset}` reads it back, echoing which traits are `human`-sourced
(and therefore uncheckable) versus `probe`-sourced.

### `workspace_lineage`

Reads `.phenotypic-mcp/lineage.jsonl` (§2.5), optionally filtered to one id's
ancestry — how an agent recovers "which pipeline produced this winner" after a
context compaction.

**Bounded by default**: optional `id` selects the ancestry **first**, then
`limit` (default 50, newest-first) bounds *that* result. The order matters: if
`limit` cut the journal globally before the filter, tracing an artifact older
than the newest 50 events would return nothing — indistinguishable from "no such
ancestry", and it would break the recovery path in §8.3. An unfiltered journal grows a row per `pipeline.put`, per
exploration step (up to 12 each, §8.7), per `tune.start`, per campaign and
deploy event — returning all of it would contradict §3.0's own token discipline.

---

## 3.4 Worked example: one subagent, one pipeline

```
catalog_operations       {category:"Detector"}
catalog_operation_detail {name:"OtsuDetector"}
pipeline_put             {name:"edge-v3", pipeline:{…}}
  -> produces_columns [Size_Area, Shape_Circularity, …], requires_gpu false
pipeline_probe           {pipeline_id:"edge-v3", subset_id:"subsets/plates-dev-24.subset.json", n_images:2}
  -> 94 objects/image, Size_Area mean 412 — blur may be too strong
pipeline_patch           {pipeline_id:"edge-v3",
                          edits:[{kind:"set_params", slot:"ops", index:0, params:{sigma:1.2}}]}
pipeline_probe           {pipeline_id:"edge-v3", subset_id:"subsets/plates-dev-24.subset.json", n_images:2}
  -> 96 objects/image, tighter spread — keep
```

Sibling subagents run this against different topologies. Their `catalog_*`,
`pipeline_put`, and `pipeline_patch` calls interleave freely (`W0`); their
probes serialize behind one worker, so peak memory is one probe, not three.

## 3.5 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-3.1 catalog breadth~~ → reconcile the two enumeration lists; expose
  `prefab` and `detect.nn` (§3.1).
- ~~OQ-3.2 probe overlays~~ → **opt-in via `save_overlay`**, off by default.
  They cost probe time and the agent cannot read them; they exist for a human
  reading the transcript. The agent's evidence is `stages` (§8.7).
# PhenoTypic MCP Server — §4 Tune Integration Contract

Status: **draft, reviewed once, revised**
Date: 2026-08-12

## 4.0 The governing constraint

`tune/_search_space/_discovery.py:4-6` states the contract this section must
honour:

> re-surfaces `infer_search_space`'s mining as per-parameter descriptors (each
> target `op_class`-stamped) for the GUI 6c form and the MCP "what can I tune?"
> tool — **the agent *selects* a target rather than authoring a key.**

So: the agent never writes `"0.sigma"`. It calls `tune_space`, reads back
structured `KnobTarget` descriptors, and refers to them by index. The legacy
string-key form still validates in `Knob` (coerced by `_coerce_legacy_strings`),
but the MCP tools do not expose it. This is not stylistic — a hand-authored key
that names a nonexistent op position or a misspelled field is a class of error
the selection model makes unrepresentable.

The second governing fact: `TuningSpec`'s model validators are the submit-time
gate (`_spec.py:293`, "where an MCP submits"). `tune_put_spec` validates by
**constructing a real `TuningSpec`**, so every existing check fires — op-index
range, `op_class` mismatch, missing field with a `difflib` did-you-mean,
unresolvable nesting, and the multi-objective/strategy rejection.

## 4.1 `tune_space` (`W0`) — what can I tune?

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `pipeline_id` | `str` | — | Target pipeline |
| `include_excluded` | `bool` | `false` | Also return params inference rejected, with reasons |
| `unbounded_factor` | `float` | `4.0` | Passed to `infer_search_space` |

Backed by **`infer_search_space(pipeline)` directly**, not by
`pipeline_targets`. The distinction matters: `pipeline_targets` returns a bare
`list[TunableParam]` and discards the `InferredSearchSpace` proposal object, so
it exposes neither `.excluded` (needed for `include_excluded`) nor
`.n_needs_review` / `.n_excluded` (which fold in inference-blind exclusions, not
just per-knob flags). The server calls `infer_search_space`, builds the
`TunableParam` rows the same way `pipeline_targets` does, and keeps the proposal
object for the rest of the payload.

```json
{"ok":true,"data":{
 "targets":[
  {"ref":0,
   "target":{"kind":"param","op":0,"field":"sigma","op_class":"BlurGauss"},
   "label":"BlurGauss.sigma",
   "value_type":"float","default":2.0,
   "suggested_domain":{"kind":"float_range","low":0.5,"high":8.0,"log":false},
   "source":"bounded","needs_review":false,
   "description":"Standard deviation of the Gaussian kernel in pixels."},
  {"ref":1,
   "target":{"kind":"param","op":1,"field":"ignore_zeros","op_class":"OtsuDetector"},
   "label":"OtsuDetector.ignore_zeros",
   "value_type":"categorical","default":true,
   "suggested_domain":{"kind":"categorical","choices":[true,false]},
   "source":"bool","needs_review":false},
  {"ref":2,
   "target":{"kind":"presence","op":1,"op_class":"OtsuDetector"},
   "label":"OtsuDetector.__enabled__",
   "value_type":"bool","default":true,"suggested_domain":null,
   "source":"presence_optin","needs_review":false}],
 "excluded":[{"key":"1.mask","reason":"ndarray","field_type":"np.ndarray"}],
 "pipeline_digest":"sha256:9c1e…",
 "n_needs_review":0,"n_excluded":1,
 "scorers_available":[
   {"class":"QCScorer","available":true,
    "requires":"a metadata CSV of expected counts","found":"data/tune_layout.csv"},
   {"class":"SupervisedScorer","available":false,
    "requires":"ground-truth masks or a count table",
    "hint":"no gt_masks_source found under the workspace"},
   {"class":"ReferenceFreeScorer","available":false,
    "requires":"meta_validate() to pass against a labelled subset"}]}}
```

Four deliberate affordances:

- **`ref` is the selection handle.** `tune_put_spec` takes `ref`s. The full
  `target` object is included so a power caller can construct one directly, and
  so the agent can reason about op positions.
- **`needs_review: true`** marks a domain inference guessed from an unbounded
  field (`[default/4, default*4]`, `source: "unbounded_heuristic"`). The agent
  should either narrow it or say it is guessing.
- **`excluded` rows carry `Excluded`'s real fields** — `key`, `reason`,
  `field_type` (`_search_space/_inferred.py:45-59`). There is no `label`;
  `field_type` is the field that actually tells an agent *why* it was excluded,
  which is the model's own documented purpose.
- **`scorers_available`** is the affordance that stops the most common failure.
  Every scorer has an `availability()` method and `run_tuning` hard-asserts it
  (`_assert_scorer_available`, `_run.py:419`) — but only *after* the agent has
  authored a whole spec. Surfacing availability at space-discovery time turns a
  late runtime abort into an upfront choice.

## 4.2 `tune_put_spec` (`W0`) — author the study

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | — | Workspace name |
| `pipeline_id` | `str` | — | Base pipeline |
| `pipeline_digest` | `str` | — | The digest `tune_space` returned; a mismatch is `stale_target_ref` |
| `select` | `array` | — | Chosen knobs (below) |
| `scorer` | `object` | — | `{class, params}` |
| `strategy` | `object` | — | `{kind, n_trials?, sampler?, seed?}` |
| `budget` | `object?` | `{}` | `{n_trials?, max_failures?}` |
| `evaluator` | `object?` | `{}` | Overrides on `Evaluator` defaults |
| `held_out` | `object?` | `{}` | `{held_out_fraction?, group_key?, …}` |
| `screen` | `bool` | `false` | Two-round screening freeze (OQ-4.1). Opt-in only; an agent enabling it must expect knobs it read from `tune_space` to stop varying mid-study. Rejected with `screening_unsupported_on_slurm` when the study routes to SLURM, which drops screening silently today (§7 P4) |
| `overwrite` / `dry_run` | `bool` | `false` | As §3 |

```json
{"name":"edge-v3-tpe",
 "pipeline_id":"edge-v3",
 "pipeline_digest":"sha256:9c1e…",
 "select":[
   {"ref":0},
   {"ref":1, "enabled":false},
   {"ref":2, "domain":{"kind":"float_range","low":1.0,"high":4.0,"step":0.25}}],
 "scorer":{"class":"QCScorer",
           // `metadata` is CORRECT here and deliberate: ExpectedVsDetectedCount
           // ships this field name with no alias. §10.3's `grouping_metadata`
           // rename applies only to the selector this spec introduces.
           "params":{"check":{"metadata":"data/tune_layout.csv"}}},
 "strategy":{"kind":"optuna","sampler":"tpe","n_trials":200,"seed":0},
 "held_out":{"held_out_fraction":0.2}}
```

`select` semantics: omitting `domain` accepts `suggested_domain`; `enabled:false`
drops the knob entirely; supplying `domain` overrides. A `ref` that no longer
resolves (because the pipeline changed since `tune_space`) is a hard error naming
the drift, **not** a silent re-index. `tune_space` returns `pipeline_digest`, and
`tune_put_spec` takes it back as `pipeline_digest` and rejects a mismatch with
`stale_target_ref`. Returning it is what makes the check **proactive**: an agent
that ran `pipeline_patch` between the two calls can compare digests itself rather
than discovering the staleness only by having its spec rejected.

### Pre-submit checks the server adds

`TuningSpec` construction catches target errors. These three it does not, and
each otherwise fails late and confusingly:

| Check | Why | Code |
|---|---|---|
| Scorer `availability()` | `_assert_scorer_available` (`_run.py:419`) fires only at `run_tuning` time, after run artifacts are written | `scorer_unavailable` |
| `grid` + a continuous `FloatRange` (`step: null`) | `grid_values` raises `ValueError` at enumeration time, deep in the run | `grid_needs_stepped_domain` |
| Empty active knob set | A study with nothing to vary | `no_active_knobs` |

Two rejections the server **does not** duplicate, because construction already
covers them: multi-objective + grid/random
(`reject_grid_random_multi_objective`), and `grid` + `n_trials` — the latter is
caught by `_coerce_strategy` plus `GridConfig`'s `extra="forbid"`, since
`GridConfig` has no `n_trials` field at all. The server surfaces those errors
rather than reimplementing them. (An earlier draft listed `grid_ignores_n_trials`
among "checks the server adds" and cited `resolve_strategy`, whose signature
takes a strategy *name*, not the structured `strategy` object a spec carries.)

### The `QCScorer` round-trip trap

`QCScorer` holds an `ExpectedVsDetectedCount` check. **It must be configured
from a metadata *path*, not an in-memory DataFrame** — a DataFrame-configured
check cannot be reloaded from JSON, and a SLURM worker reloads the spec from
disk. The server enforces this: `scorer.params.check.metadata` must be a
sandbox-resolvable path, rejected at put time with `code: "scorer_not_portable"`.

**And the server resolves it to an ABSOLUTE path before writing the resolved
spec.** `ExpectedVsDetectedCount._normalize_metadata`
(`analysis/qc/_expected_vs_detected.py:213-251`) keeps the string exactly as
given, and `model_post_init` re-resolves it on **every** reconstruction —
including inside a fresh SLURM worker reloading the spec from disk. No SLURM
script in this codebase sets `--chdir` or calls `os.chdir` (verified across
`_execution/_slurm.py` and `sdk_/slurm/*.py`), so a workspace-relative path works
today only by the accident that jobs inherit the submission CWD. Writing an
absolute path removes the CWD dependency entirely; leaving it relative means a
future `--chdir` added for unrelated reasons silently breaks every distributed
study, or worse resolves to an unrelated file that happens to exist there.
Every distributed worker reloads the resolved spec, so this is a correctness
requirement, not a nicety.

### Two payloads, one filename

`auto-space` writes an `InferredSearchSpace` to
`deliverables/tuning_spec.json.pht-tune`; `run` writes a full `TuningSpec` to the
same filename. Any tool loading a spec **must discriminate on shape** — a
`TuningSpec` has `pipeline`/`scorer`; an `InferredSearchSpace` has
`knobs`/`excluded` at top level. The server does so and reports which it found.

Note also that `TuningSpec` has **no `extra="forbid"`** (`_spec.py:162`), so
pydantic silently ignores unknown top-level keys. The MCP layer is stricter: an
unrecognized key in a `tune_put_spec` payload is an error, because silently
dropping an agent's intent is worse than rejecting it.

## 4.3 `tune_start` (`W2`) — launch

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `spec_id` | `str` | — | Authored spec |
| `subset_id` | `str` | — | Registered subset (§10.3.1). **A raw path is refused** — `tune_start` reaches fleet-scale compute |
| `study_name` | `str?` | derived from `spec_id` | Output directory under `studies/` |
| `compute` | `object?` | `{}` | `{profile, n_workers, time, mem}` — §5 rules |
| `strategy_override` | `object?` | `null` | `{strategy?, n_trials?}` |

**Launch mechanism: subprocess, exactly as the GUI does it.** The server builds
argv with `tune_run_argv`, then calls `deploy_tune_run(runner=…, registry=…,
sandbox=…, argv=…, output_dir=…, slurm=…)`, which allocates a `RunRegistry`
record, starts a `LocalRunner` process, and CASes the pid and log paths in.

A SLURM tune is *also* launched through the local runner — the spawned
`python -m phenotypic.tune` process owns the `--slurm` fleet submission itself.
This mirrors `gui/tune/_deploy.py:36-39` and means one code path covers both.

### Storage routing (the part that depends on §7 P1)

The server **always passes an explicit `--storage-url`** and never lets
`$PHENOTYPIC_TUNE_STORAGE_URL` be inherited implicitly. That env var is a
resolution fallback in `_resolve_storage_url`, and `_STUDY_NAME` is the hard
constant `"tune_cost_v1"` — so with the env var set, N parallel agent studies
would silently attach to **one** Optuna study and pool their trials (hazard H2,
§7).

| Situation | Storage passed |
|---|---|
| local study | `sqlite:///<study>/.pht-tune-cache/study.db` |
| SLURM, P1 landed | `journal:///<study>/.pht-tune-cache/journal.log` |
| SLURM, P1 not landed (or L1 unproven), Postgres configured | the configured password-less URL, with a per-study database — supported, not recommended once the journal lands (§7) |
| SLURM, P1 not landed, no Postgres | **refused**, `code: "distributed_storage_unavailable"` |

That last row matters: `_validate_slurm_request` does **not** check the storage
backend, so `--slurm` with SQLite submits happily into the documented NFS
corruption case (hazard H1). The server refuses rather than submitting.

The server additionally refuses to start a study whose resolved storage URL
matches that of another live study — the direct guard against H2.

Response:

```json
{"ok":true,
 "data":{"study_id":"studies/edge-v3-tpe","run_record":{"generation":"…","status":"queued"},
         "storage":"journal:///…/.pht-tune-cache/journal.log",
         "n_trials":200,"n_workers":8,"images":42},
 "routed":{"class":"W2","routed_to":"slurm","reason":"environment=slurm, profile=cpu-bulk",
           "queue_position":null}}
```

## 4.4 `tune_status` (`W0`) — poll

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `study_id` | `str` | — | Target |
| `detail` | `"progress" \| "results"` | `"progress"` | How much to read |

### What `progress` can actually report

An earlier draft claimed `progress` returns trial counts, the running best, and
the held-out gap while being "marker-only… it never opens Optuna", and cited the
GUI Monitor's tick as precedent. **Both halves were wrong**, and they were wrong
in opposite directions:

- `TuneRunRoot` (`gui/tune/_run_root.py:60-156`) is a **location resolver**. It
  returns `{path, trials_path, storage_url, study_name, directions, images_dir,
  best_pipeline_path}` — where things are, not how the run is going.
- The `run.json` marker (`_run.py:205-217`) holds
  `{version, study_name, storage_url, images_dir, strategy, n_trials,
  is_multi_objective, slurm, start_time}` — written **once at start**. No
  `completed`, no `failed`, no `best`, no `gap`.
- The GUI tick does **not** stop at discovery. `_poll_study`
  (`gui/tune/_callbacks.py:1911-1939`) calls `TuneRunRoot.discover(...)` and then
  immediately `read_study_for_monitor(root)` — which opens the live store. The
  precedent cited as proof of cheapness is the thing that opens Optuna.

So the two modes are redrawn along the line that actually exists — **does this
touch the trial store or not**:

| Mode | Reads | Returns | Cost |
|---|---|---|---|
| `progress` | `run.json`, the `.pht-tune-cache` markers, `RunRegistry`, and a **parquet row count** when `trials.parquet` exists | `status`, `strategy`, `n_trials` budget, `scheduler` job ids, `started_at`, and `trials_recorded` *only if the parquet is present* | genuinely cheap; no store |
| `results` | the live store, or `trials.parquet` on degradation | leaderboard, best trial, per-term costs, importances, Pareto front, generalization gap | **killable subprocess**, §4.4 below |

`completed` / `pruned` / `failed` / `best` / `gap` are trial-level facts and live
**only** in `results`. A distributed run writes no `trials.parquet` until
finalize (§4.5), so for those `progress` reports `trials_recorded: null` and says
so rather than implying zero.

**Consequence for `campaign_status` (§8.3):** its per-arm leaderboard is
trial-level, so it is a `results`-class call, not a free one. It runs one
store-open per arm through the same killable subprocess, and the orchestrator is
expected to poll it on a human timescale (minutes), not a UI tick.

`results` opens the study (or degrades to `trials.parquet` when the store is
unreachable) and returns the leaderboard, best trial, parameter importances,
Pareto front for multi-objective, and the generalization report.

**`results` opens the store in a killable subprocess.** This is not symmetry with
the probe worker for its own sake — it is forced by §7 B2/B3. Constructing a
`JournalFileBackend` creates the file as a side effect of a "read-only" open
(B2), and an `open()`/`os.path.exists()` against a stale NFS mount blocks in an
uncancellable syscall (B3). The GUI survives this only because a human retries;
this server does not have that luxury. **One stdio process serves every subagent
in the session** (§1.3), so a single wedged poll against a stale mount would
stall `tune_status(detail="results")` for *every* subagent for the rest of the
session, with nothing to notice or recover it.

The `progress` mode has no such exposure — it is genuinely marker-only and never
opens a store, which is what makes *that* mode safe to poll often. The first
draft extended "polling is safe" to both modes; it is only true of one.

```json
// detail: "progress"  — no store opened
{"ok":true,"data":{
  "status":"running","strategy":"tpe","budget":200,
  "trials_recorded":null,
  "trials_recorded_note":"distributed run; trials.parquet is not written until finalize (§4.5)",
  "started_at":"2026-08-12T14:07:40Z",
  "scheduler":{"job_ids":["4412331"],"reachable":true}}}

// detail: "results" — store opened in a killable subprocess
{"ok":true,"data":{
  "status":"running","completed":126,"pruned":14,"failed":3,"budget":200,
  "best":{"trial":47,"score":0.081,
          "params":{"BlurGauss.sigma":1.34,"OtsuDetector.__enabled__":true}},
  "gap":{"value":0.06,"verdict":"ok"}}}
```

Two honest degradations, both reported rather than hidden:

- **`--slurm` never writes `trials.parquet`.** The docstring at `_run.py:744`
  claims a later `--recompile` finalize does it, but **the tune CLI has no
  `--recompile` flag**. So a fire-and-forget distributed study's parquet does not
  appear on its own. `tune_status` reads the live store instead, and
  `tune_export_best` triggers finalization explicitly (§4.5).
- **Cost convention.** Every score is a cost in `[0,1]`, lower is better,
  minimized. `tune_status` labels it `"score (cost, lower is better)"` so an
  agent cannot mistake it for an accuracy.

## 4.5 `tune_export_best` (`W0`) — close the loop

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `study_id` | `str` | — | Source study |
| `name` | `str?` | `<study>-best` | Destination pipeline name |
| `objective` | `str?` | `null` | For multi-objective: export a per-axis winner instead of the knee |

Winner selection follows `_headline_winner`: the Pareto **knee point** when a
front exists, otherwise `best()`. The response states which rule applied
(`selection: "pareto_knee" | "single_best"`), because for a multi-objective study
"the best pipeline" is a choice, not a fact.

### Local and distributed studies need different paths

`export_best_from_run` → `prepare_best_from_run` (`gui/tune/_export.py:69-88`)
hard-requires `best_params_path(output_dir).is_file()`, raising
`FileNotFoundError` otherwise. That file is written by exactly one call site,
`_finalize_best_params` (`tune/_tune_cli/_run.py:637`) — which sits **after**
`if slurm: return _submit_slurm_fleet(...)` (`:593-609`).

**So a SLURM-launched study never writes `best_params.json`, and the plain
export path raises on every distributed study.** An earlier draft of §4.6 called
`tune_export_best` on a study it had shown two lines earlier as
`routed_to: slurm`; as written that example would have thrown.

| Study | Path |
|---|---|
| Local | `export_best_from_run(output_dir)` — the sidecar already exists |
| **Distributed** | **Finalize first, then export** |

The distributed finalize opens the live store, computes `_headline_winner` /
`_selection_label`, and writes the artifacts the SLURM branch skipped — **in the
local run's existing order, which is load-bearing**
(`_run.py:628-641`):

1. `_finalize_outputs` → `trials.parquet`, `param_importance.json`,
   `best_pipeline.json`
2. `_finalize_pareto_outputs` → Pareto front, per-axis winners, and it
   **overwrites** `best_pipeline.json` with the knee
3. `_finalize_best_params` → `best_params.json` — **last, deliberately**
4. `_finalize_generalization` → `generalization.json`

then the ordinary `prepare_best_from_run` → `publish_prepared_export`.

**`best_params.json` is written last because it is the de-facto completion
marker.** `prepare_best_from_run` gates on its existence
(`gui/tune/_export.py:75-77`, raising `FileNotFoundError` otherwise), so writing
it first — as an earlier draft of this section specified — would leave an
interrupted finalize looking exportable when it is not.

**Two interruption hazards the order does not fully close**, both reported rather
than hidden:

- A kill *inside* step 2 leaves `best_pipeline.json` holding the **scalar** best
  from step 1, never overwritten by the knee. Since `selection` is recomputed on
  each export call, a stale file can be labelled `pareto_knee` when it is not.
  The finalize therefore writes a `finalize_in_progress` marker at step 1 and
  clears it after step 4; an export finding that marker refuses with
  `finalize_incomplete` rather than trusting the directory.
- Finalize is **not** safe against a still-running study. Two concurrent
  `tune_export_best` calls would each compute a different `_headline_winner` as
  trials land, and overwrite each other. So finalize is gated on the study being
  terminal — budget drained or no live scheduler jobs — and refuses with
  `study_not_finished` otherwise. `_finalize_best_params` silently no-ops when
  the winner is `None` (`_run.py:712-713`), which would otherwise surface later
  as a misleading `FileNotFoundError`.

This also **resolves OQ-4.3 affirmatively**: `trials.parquet` *is* written for
distributed studies, because the finalize already holds the store open and the
marginal cost is one parquet write. The alternative left every distributed study
directory permanently unreadable offline and un-openable by the GUI's
parquet-only degradation path.

The store open runs in a **killable subprocess**, for the reason in §4.4.

The winner is materialized by `build_pipeline(spec.pipeline, trial.params)`,
which deep-copies the base and **rebuilds each op through its constructor** so
every validator re-runs — the apply-time ⊆ backstop that catches
validator-enforced bounds inference cannot see. A winner that fails to rebuild is
an error naming the offending knob, not a silent fallback.

The exported pipeline is written into `pipelines/` as a **new artifact**, and a
lineage row records `parent: <study_id>`, `trial`, and `score`. It is then an
ordinary pipeline id — probeable with `pipeline_probe`, deployable with
`deploy_start`.

## 4.6 Worked example: three subagents, one winner

```
# each subagent, independently and concurrently (all W0, no slot contention)
tune_space      {pipeline_id:"edge-v3"}          -> 7 targets, QCScorer available
tune_put_spec   {name:"edge-v3-tpe", pipeline_id:"edge-v3",
                 select:[{ref:0},{ref:2}], scorer:{QCScorer(metadata=…)},
                 strategy:{kind:"optuna",sampler:"tpe",n_trials:200}}
tune_start      {spec_id:"edge-v3-tpe", subset_id:"subsets/plates-dev-24.subset.json",
                 compute:{profile:"cpu-bulk", n_workers:8}}
                -> routed_to slurm, study_id studies/edge-v3-tpe

# orchestrator polls all three cheaply
tune_status {study_id:"studies/edge-v3-tpe"}   -> 126/200, best 0.081
tune_status {study_id:"studies/watershed-tpe"} -> 143/200, best 0.117
tune_status {study_id:"studies/canny-tpe"}     -> 98/200,  best 0.204

tune_export_best {study_id:"studies/edge-v3-tpe", name:"winner"}
                -> distributed study: finalize first (opens the store in a
                   killable subprocess, writes best_params.json + trials.parquet
                   + param_importance.json, which the SLURM branch skipped),
                   then export
                -> pipelines/winner.json.pht-pipe, selection single_best
```

Because each study's storage URL is derived from its own output directory, the
three studies are isolated by construction — the same property local SQLite
already has, now extended to the distributed case by P1.

## 4.7 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-4.1 screening~~ → **fix the no-op (§7 P4), then expose it, default off**.
  `tune_put_spec` takes `screen: false` by default; the agent opts in
  deliberately. An agent that enables it must expect knobs it read from
  `tune_space` to stop varying mid-study.

- ~~OQ-4.2 who picks the scorer~~ → **explicit always**. `tune_space` reports
  availability, but the agent names the scorer even when only one is available.
- ~~OQ-4.3 `trials.parquet` for distributed studies~~ → **yes, written**. The
  distributed finalize (§4.5) already holds the store open, so the marginal cost
  is one parquet write, and it is what makes a distributed study directory
  readable offline and openable by the GUI's parquet-only degradation path.
# PhenoTypic MCP Server — §5 Deploy and SLURM Contract

Status: **draft, reviewed once, revised**
Date: 2026-08-12

## 5.1 Why a governance layer exists at all

`parse_slurm_args` (`_cli/_cli_utils.py:336-375`) constrains nothing. Any
non-empty key is accepted and mechanically translated into a directive by
`key.replace("slurm_","").replace("_","-")`; values go through
`ast.literal_eval` with a silent fallback to raw string. The only guardrails in
the whole path are:

- four reserved keys dropped with a warning — `array`, `output`, `error`,
  `job-name` (`sdk_/slurm/_sbatch.py:20`);
- an **advisory-only** warning past `MAX_SLURM_TIME_MINUTES` (10080 = 7 days,
  `_cli_constants.py:24`) — it does not block.

So an unconstrained agent can name a partition that does not exist, request
seven days on a shared queue, or bill an account that is not yours. Exposing
`--slurm key=value` verbatim is not an option. The profile layer is the guard.

## 5.2 Server configuration

`~/.phenotypic/mcp.toml`, read-only at startup.

```toml
[workspace]
root = "/scratch/alex/phenotypic-agent"     # optional; --workspace wins, else CWD

[execution]
environment = "auto"          # auto | local | slurm

[limits]
probe_max_images  = 4
probe_timeout_s   = 300

# Distributed-tune storage. Referenced by §4.3's routing table; without this
# table a SLURM tune request returns `distributed_storage_unavailable`.
[tune.storage]
# "journal"  -> journal:///<study>/.pht-tune-cache/journal.log  (requires §7 P1
#               AND a passing L1 negative-control run on this cluster's mount)
# "postgres" -> the operator-provided server below
# "none"     -> refuse all distributed tune; local studies still work
distributed_backend = "postgres"

  [tune.storage.postgres]
  # Password-less ONLY. libpq resolves the secret from ~/.pgpass, $PGPASSWORD,
  # or a PGSERVICE entry; `_reject_password_in_url` rejects an inline password,
  # and the server never accepts one as an argument.
  url_template = "postgresql+psycopg://${USER}@pg-host:54399/${study_db}"
  # The server mints one database per study — H2's isolation unit is the
  # database, not the study name (§7).
  create_database = true

[slurm.profiles.cpu-bulk]
partition     = "batch"
account       = "exfab"
qos           = "normal"
cpus_per_task = 8
mem_gb        = 16
time          = "04:00:00"
overridable   = ["time", "cpus_per_task", "mem_gb", "njobs", "n_workers"]

  [slurm.profiles.cpu-bulk.caps]
  max_time          = "08:00:00"
  max_cpus_per_task = 32
  max_mem_gb        = 64
  max_array         = 512
  max_n_workers     = 32

[slurm.profiles.gpu-short]
partition        = "gpu"
gpus_per_node    = 1
time             = "02:00:00"
overridable      = ["time"]
  [slurm.profiles.gpu-short.caps]
  max_time  = "02:00:00"
  max_array = 16
```

**Rules, enforced before any argv is built:**

1. The agent names a `profile`. It cannot supply a partition, account, or QoS —
   those exist only in config.
2. It may set only keys listed in that profile's `overridable`.
3. Every override is checked against `caps`. A violation is a hard error naming
   the key, the request, and the cap — never a silent clamp, which would make
   the agent believe it got what it asked for.
4. Reserved SBATCH keys are rejected outright rather than warned-and-dropped, so
   an agent that tries to set `--array` learns it cannot.
5. `time` values pass through `parse_slurm_time` for canonicalization before
   comparison, so `240`, `04:00:00`, and `0-04:00:00` compare correctly.

Array width is additionally floored by the live cluster: `get_slurm_array_limit`
(`scontrol show config` → `MaxArraySize`, fallback 1000) and
`get_slurm_max_submit_jobs` (`sacctmgr`). The effective width is
`min(profile cap, cluster limit)`, and the plan reports which bound applied.

## 5.2.1 How an agent actually supplies compute

**It never writes an sbatch key.** It names a profile and, optionally,
overrides from that profile's `overridable` list:

```json
"compute": {"profile": "cpu-bulk", "time": "02:00:00", "n_workers": 8}
```

In the normal flow it does not even do that per call: `compute` is agreed
**once at campaign level** (§8.2) and every arm inherits it, so the subagents
executing arms choose no SLURM parameters at all. A per-call `compute` appears
only on a standalone `tune_start` or `deploy_plan` outside a campaign.

Three sites accept it — `campaign_put`, `tune_start`, `deploy_plan` — and
`deploy_start` takes none, inheriting from its `plan_token`.

### One `compute` object, two different CLI surfaces

The server, not the agent, translates. And the two engines do not accept SLURM
parameters the same way:

| Path | Surface |
|---|---|
| `python -m phenotypic` | **repeated `--slurm key=value`** (`phenotypicCLI.py:795`), free-form keys |
| `python -m phenotypic.tune` | **four discrete flags** — `--slurm-partition`, `--slurm-mem`, `--slurm-time`, `--slurm-constraint` (`tune/__main__.py:104-125`), plus `--n-workers` |

### The tune path cannot express a full profile — so the server refuses

`_submit_slurm_fleet` (`tune/_tune_cli/_run.py:797-805`) builds its `slurm_args`
from **only those four flags**. There is no `--slurm-account`, no `--slurm-qos`,
no `--slurm-cpus-per-task`, no `--slurm-gpus`.

So a profile like the `cpu-bulk` example above is **not fully expressible on the
tune path**:

| Profile key | deploy | tune |
|---|---|---|
| `partition` | ✅ | ✅ |
| `mem_gb` | ✅ | ✅ (`--slurm-mem`) |
| `time` | ✅ | ✅ |
| `constraint` | ✅ | ✅ |
| **`account`** | ✅ | ❌ **dropped** |
| **`qos`** | ✅ | ❌ **dropped** |
| **`cpus_per_task`** | ✅ | ❌ **dropped** |
| **`gpus_per_node`** | ✅ | ❌ **dropped** |

On a cluster where `account` is mandatory, a tune fleet submitted under this
profile is rejected by the scheduler — or, worse, silently billed to your
default account. This is exactly the silent-drop failure the profile layer
exists to prevent, so the server performs an **expressibility check**:

> Before submitting, the server checks every profile key against the target
> path's supported set. An inexpressible key is a hard error
> (`profile_not_expressible`) naming the key, the path, and the profile —
> **never a silent drop.**

Two ways out, and the config picks:

```toml
[slurm.profiles.cpu-bulk]
partition = "batch"; account = "exfab"; qos = "normal"; cpus_per_task = 8
paths = ["deploy"]                 # this profile is deploy-only

[slurm.profiles.tune-fleet]
partition = "batch"; mem_gb = 8; time = "04:00:00"
paths = ["tune"]                   # expressible on the narrow tune surface
```

`paths` defaults to both; the server validates it against each surface at
startup, so a mis-specified profile fails when you load config rather than at
2 a.m. when a fleet submits.

**The better fix is upstream**, and it belongs in §7: give the tune CLI the same
`--slurm key=value` surface the forward CLI already has, at which point one
profile serves both paths and the expressibility check becomes vestigial. Until
then the check is what keeps the drop from being silent.

## 5.3 `deploy_plan` (`W0`) — preview, never submit

Deploying to a cluster is the one place an agent can consume a large amount of
somebody else's compute. `deploy_plan` makes the ask inspectable first, and it
performs **no** submission and **no** writes under the run's output directory.

**This requires a refactor, not a call-through.** The only existing generator
that produces a full sbatch script, `generate_array_job_script`
(`_cli/_cli_slurm_array_scripts.py:116-368`), has real side effects under the
*real* output directory: `script_dir.mkdir(...)` (`:184-185`), `log_dir.mkdir(...)`
(`:198`), and `write_slurm_array_script` → `path.write_text(...)` +
`path.chmod(0o755)` (`sdk_/slurm/_script_rendering.py:133-147`).

Calling it for a "preview" would populate `<output_dir>/.phenotypic/slurm_scripts/`
and `logs/` **before you approve anything** — and would then trip
`deploy_start`'s own `output_not_empty` check (§5.4) on the directory the
preview swore it only looked at.

`SlurmArrayScriptSpec.render()` *is* already pure. What is entangled is the
~150 lines of argument, `cmd_parts`, and dispatch-block construction that build
the spec inside `generate_array_job_script` alongside the write. So P2 gains a
task: **extract `build_array_script_spec(...) -> SlurmArrayScriptSpec`** with no
I/O, and have both the real generator and `deploy_plan` call it. This is a fifth
genuinely new piece of work (§1.6), previously invisible in that accounting.

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `pipeline_id` | `str` | — | Pipeline to run |
| `subset_id` | `str` | — | Registered subset (§10.3.1) |
| `scope` | `"subset" \| "full"` | `"subset"` | `full` plans against `subset.parent`; the only way to obtain a full-scope `plan_token` |
| `run_name` | `str` | — | Output directory under `runs/` |
| `compute` | `object?` | `{}` | `{profile, …overridable keys}` |
| `mode` | `"full" \| "measure" \| "process"` | `"full"` | CLI mode |
| `layer` | `str?` | `null` | Required when `mode="process"` |
| `metadata_csv` | `str?` | `null` | Joined onto the measurements mirror |
| `sample` | `int?` | `null` | Pilot subset |

```json
{"ok":true,"data":{
  "routed_to":"slurm","profile":"cpu-bulk",
  "datasets":{"plateA":48,"plateB":52},"n_images":100,
  "argv":["python","-m","phenotypic","--mode","full","--pipeline","…","--slurm","slurm_partition=batch","…"],
  "sbatch_preview":"#!/bin/bash\n#SBATCH --partition=batch\n#SBATCH --account=exfab\n#SBATCH --array=0-99\n…",
  "array":{"requested":100,"chunks":1,"effective_limit":512,"limit_source":"profile cap"},
  "estimate":{"basis":"probe of 2 images at 3.4 s/image",
              "node_seconds":340,"wall_clock_hint":"~6 min at 100 concurrent"},
  "requires_gpu":false,"staged_gpu":false,
  "outputs":{"root":"runs/2026-08-12-plateA",
             "deliverables":"runs/2026-08-12-plateA/deliverables"},
  "plan_token":"pl_7f3a…","plan_expires":"2026-08-13T14:02:11Z","scope":"subset"},
 "issues":[{"severity":"warning","code":"no_probe_evidence",
            "message":"No probe recorded for this pipeline; the estimate is a default, not a measurement."}]}
```

The estimate is honest about its basis. If the agent has run `pipeline_probe`
on this pipeline, per-image timing comes from that lineage row; otherwise the
response says the number is a default. An estimate presented without provenance
is worse than none.

## 5.4 `deploy_start` (`W3`) — submit

Same arguments as `deploy_plan`, plus:

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `scope` | `"subset" \| "full"` | `"subset"` | `full` targets `subset.parent` and additionally requires `promotion_token` (§10.5) |
| `plan_token` | `str` | — | **Required.** From a matching `deploy_plan`, or from an approved campaign arm |
| `promotion_token` | `str?` | `null` | **Required when `scope: "full"`.** From `promotion_approve` |
| `resume` | `bool` | `false` | Continue an interrupted run |
| `retry_failures` | `bool` | `false` | Requires `resume` |
| `restart` | `bool` | `false` | Clear machine state and start over |

**Plan-then-submit is mandatory.** `deploy_start` refuses without a
`plan_token` whose recorded `(pipeline digest, images digest, compute)` matches
the request, returning `code: "plan_required"` or `"plan_stale"`. Every cluster
submission is therefore preceded by an inspectable preview.

### What a token *is*

A token is **an opaque random id naming a persisted record** — not a digest of
its own contents.

```
<workspace>/.phenotypic-mcp/plans/<token>.json
{"token":"pl_7f3a…","kind":"plan","created":"…","expires":"…",
 "scope":"subset","pipeline_digest":"sha256:9c1e…","subset_id":"subsets/…",
 "subset_digest":"sha256:77b2…","compute":{"profile":"cpu-bulk","time":"02:00:00"},
 "argv_digest":"sha256:4b0a…","consumed_by":null}
```

The alternative — a **self-describing hash** of `(pipeline_digest,
images_digest, compute)` — is tempting because it needs no storage and the
server could recompute it. It is also **forgeable**: §2.5 publishes the exact
digest format, so any agent could compute a valid token without ever calling
`deploy_plan`, and "plan-then-submit is mandatory" would be a fiction rather
than a gate. A random id cannot be guessed from public inputs.

Persisting rather than holding in memory follows from §1.3: the server may be
killed at any time and holds no authoritative state, so an in-memory token map
would make every restart silently invalidate approvals you had already given.

| Property | Rule |
|---|---|
| Storage | `<workspace>/.phenotypic-mcp/plans/<token>.json`, atomic write (added to §2.3's tree and §2.6's concurrency table) |
| Validation | Re-derive the digests from the *current* request and compare; any mismatch → `plan_stale` naming which field moved |
| Expiry | `expires` (default 24 h); an expired token → `plan_stale`, not silent acceptance |
| Single use | `consumed_by` is CAS'd to the `run_id` on a successful `deploy_start`; a second use → `plan_stale`. Re-running a deploy means re-planning, which is cheap and keeps the preview honest |

`promotion_token` records are identical in shape with `kind: "promotion"`, and
additionally bind `parent_digest` — so a parent that gained images between
review and submit invalidates the token (`promotion_stale`, §10.5) rather than
quietly deploying over a dataset you did not review.

The token is satisfied two ways: a direct `deploy_plan` call, or membership in
an **approved campaign** (§8), which stamps a token per arm at approval time.
That keeps the human checkpoint in the planning phase where you actually are,
rather than inserting one into autonomous Phase-2 execution.

**`overwrite` is deliberately not exposed.** The CLI's `--overwrite` does
`shutil.rmtree(output_dir)` — it destroys `deliverables/`, every per-image HDF,
and any QC curation a human did. That is not an action an agent should be able
to take from a tool call. A non-empty output directory returns
`code: "output_not_empty"` with three named options: pick a new `run_name`,
`resume`, or `restart`. Deleting data stays a human decision at a shell prompt.

`restart` **is** exposed, because `clear_machine_state` wipes only
`.phenotypic/` and leaves `deliverables/`, `results/`, and `qc/` intact. The
response states exactly what it cleared.

**Mechanism.** Build argv via `to_argv(RunConsoleState)` + the profile's
`--slurm` pairs, `RunRegistry.allocate`, `LocalRunner.start`, CAS in pid and log
paths — the same path the GUI Run Console uses. `--wait` is **never** passed:
MCP tool calls must not block for hours, so submission returns immediately and
status is polled (§5.5). This matches the CLI's own default.

`resume` preconditions are checked by the server *before* submitting, because
the CLI's failure mode is `sys.exit(1)` inside a subprocess, which reaches the
agent as an opaque non-zero exit.

**The server imports and calls `validate_resume_compatibility` directly** rather
than re-enumerating its field list. `ExecutionConfig` (`_cli/_cli_types.py:95`)
is a plain dataclass constructible without Click, and every field compared is a
raw CLI-supplied value rather than something derived from image inspection — so
the check genuinely replays before submission.

Re-enumerating was the first draft's approach and it was already wrong: the
published list omitted `include_dataset_column`, `overlay_alpha`, and
`save_overlays`, which `validate_resume_compatibility`
(`_cli/_cli_state_management.py:304-319`) also checks when present in the saved
state. A hand-maintained mirror of another module's validation drifts silently
and reintroduces exactly the opaque-exit failure this pre-check exists to
prevent. Calling the real function cannot drift.

One caveat the server must handle: `input_path` is compared by literal
`Path`/string equality with no `resolve_path=True` normalization on the `--input`
option, so the server must serialize the path exactly as the original run's
state recorded it or get a spurious mismatch.

It also refuses while ledgered SLURM jobs are live, which the CLI likewise
blocks.

### GPU staging is automatic, and the plan says so

A pipeline containing a `GpuDetector` (`pipeline_requires_gpu`) *usually*
triggers the staged engine — CPU preprocess → resident-model GPU detect → CPU
measure — not per-image processing. On SLURM that becomes an epoch-fenced
controller with Stages 1 & 3 on the CPU profile and Stage 2 as a GPU array.

**It is not unconditional, and `deploy_plan` must report the real answer.**
`uses_staged_gpu_strategy` (`_cli_execution_strategies.py:1058-1068`) routes to
the staged engine only when `process_only_layer` is `None` (any mode) or
`"objmap"` (local only). So `mode="process"` with a `layer` other than `objmap`
falls back to ordinary per-image `AutonomousSLURMStrategy` — which still
auto-adds `slurm_gpus_per_node=1` (`:705-710`) but loads the model per image
instead of once. `deploy_plan`'s `staged_gpu` flag reflects the actual dispatch,
because a user told "staged" who then gets per-image model loading will see
wildly different cost than the estimate promised.

The agent does not opt into staging and cannot opt out of it. What it must do is
name a **GPU profile** for Stage 2 via `compute.gpu_profile`, which maps to
`--gpu-slurm` and inherits/deltas over the CPU profile (`{**cpu, **gpu}`,
`_cli_staged_slurm.py:92`). `deploy_plan` shows both profiles.

## 5.5 `deploy_status` (`W0`) — poll

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `run_id` | `str` | — | Target |
| `detail` | `"progress" \| "results"` | `"progress"` | How much to read |
| `refresh` | `bool` | `false` | Force a manifest rebuild before reading |

**`deploy_status` overlays the `RunRegistry` record on the manifest read.** The
manifest alone can never report a cancellation: `is_complete` is computed purely
from completed/failed image counts (`_dashboard/_manifest_builder.py:632,646`)
and has no cancellation concept, and `staged_finalization_complete.json` is
written only on success. So a cancelled SLURM or staged run polled through the
manifest alone sits at `is_complete: false` **forever**. `RunRegistry`'s status
enum does carry a distinct `cancelled` (`gui/shell/_runs_registry.py:76-86`), so
the record is the authority for lifecycle state and the manifest is the
authority for progress counts. `deploy_status` reports both and lets the record
win on status.

**`manifest.json` is the designed polling surface for progress**, not the CLI
exit code.
Without `--wait` the CLI exits 0 the moment jobs are submitted, so exit status
says nothing about the run. Terminal truth is `manifest.json → is_complete`
plus `run_completion.json` (ordinary) or `staged_finalization_complete.json`
(staged).

```json
{"ok":true,"data":{
  "status":"running","execution_mode":"slurm",
  "total_images":100,"completed":63,"failed":2,"started":8,"pending":27,
  "success_rate":0.969,"is_complete":false,
  "datasets":{"plateA":{"total":48,"completed":40,"failed":1}},
  "failure_categories":{"DetectionError":2},
  "slurm":{"chunk_job_ids":{"0":"4412331"},"active_chunks":[0]},
  "terminal_marker":null,
  "last_updated":"2026-08-12T15:22:04.881Z"}}
```

This response is a **projection over `manifest.json`, not a verbatim relay.**
Most field names pass through unchanged, but `slurm` here renames the manifest's
`slurm_info` key (`DashboardManifestSlurmInfoKey`,
`sdk_/_io_constants.py:1785-1797`); its sub-keys (`chunk_job_ids`,
`active_chunks`, …) are unchanged. Naming the projection explicitly matters
because every *other* field does relay verbatim, so an unflagged rename reads as
a transcription slip.

On SLURM the manifest is refreshed mid-run by an in-array sentinel task. An
external process can force a refresh by invoking the same handler —
`python -m phenotypic._cli._cli_checkpoint_handler --output-dir <dir>
--checkpoint-type manifest` — which is what `refresh: true` does. Without it,
`last_updated` may lag, so the field is always returned and the agent is
expected to read it.

`detail: "results"` adds the deliverables inventory and a **bounded summary** of
`measurements.parquet`. Never raw rows — the parquet path is returned for
anything more.

The bound is not optional. §3.1 already establishes that one measurer
(`MeasureTexture` at `scale=[5,10]`) emits **130 columns**, and a full deploy
produces per-object rows across every image, so "per-column describe" is
unbounded in both dimensions as first written:

| Bound | Default |
|---|---|
| Columns described | numeric columns only, first 40 by schema order; `columns_truncated: true` and the full column list still returned |
| Rows scanned | full file via the parquet reader's column projection — only the described columns are read, not the whole table |
| `QC_MetadataOnly` rows | counted separately and excluded from the describe, so metadata-only phantoms never inflate a distribution |

**And this call is offloaded.** `deploy_status` is classified `W0`, and §1.5
runs `W0` inline on the event loop — but reading and describing a large parquet
is real I/O plus compute, and doing it inline would stall every other subagent's
`W0` call for its duration, which is the exact failure `run_in_executor` exists
to prevent for `W1`. `detail: "results"` therefore runs in the executor even
though it takes no `LocalComputeSlot`; it is `W0` in the sense of *not touching
the compute slot*, not in the sense of *being instant*.

Two facts the response surfaces because getting them wrong corrupts analysis:

- `master_measurements.*` is the clean, pre-post, metadata-free archive;
  `measurements.*` is the post-applied, metadata-joined **mirror**. Analysis and
  dashboards read the mirror. The response labels both.
- Rows with `QC_MetadataOnly = true` are metadata keys that matched no measured
  object. They are counted separately so an agent does not read them as
  detections.

Failures come from `failures.jsonl` via `categorize_failures`, with per-image
detail available on request rather than dumped by default.

## 5.6 Cancellation

`workspace_cancel` (§3.3) routes by mode: `cancel_generation` for ordinary SLURM
runs, `cancel_staged_jobs` for staged, `LocalRunner.stop` (SIGTERM → 10 s →
SIGKILL) for local. It is **scoped to runs this server holds a `RunRegistry`
record for**, so an agent cannot cancel another session's or another user's
jobs. Cancellation is generation-fenced, so a cancel aimed at a superseded
generation is refused rather than killing the replacement.

**Local cancellation races the exit observer.** `LocalRunner.stop()` does not
touch `RunRegistry`; the caller CASes afterwards. But `observe_local_exit`
(`gui/shell/_runs_registry.py:536-588`) runs concurrently and checks for
`{cancelling, cancelled}` *before* falling back to `returncode != 0 → failed`.
If it wins the race, the run is recorded `failed` with
`"exited with status -15"`, and the cancel path's own CAS is then rejected —
its `expected_statuses` excludes `failed`. So the server sets `cancelling`
**before** calling `stop()`, closing the window. Otherwise a cancellation you
asked for is reported back to you as a crash.

**`cancel_staged_jobs` is the least-exercised of the three paths.** It exists and
is exported (`_cli_staged_orchestration.py:633-655`) but has no non-test caller —
the GUI routes staged cancels through `cancel_generation` like any other SLURM
run. It is specified here as the staged path and flagged as unverified in
production rather than presented as equally proven.

## 5.7 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-5.1 plan gate~~ → **plan-then-submit is mandatory**, satisfied either by a
  direct `deploy_plan` or by an approved campaign arm (§5.4, §8.1).
- ~~OQ-5.2 pilot-first~~ → not mandated. The plan gate plus probe-derived
  estimates cover the wasted-allocation case without forcing a pilot on a
  workflow you already validated in the GUI. `sample` remains available.
- ~~OQ-5.3 profile defaults~~ → **no silent defaults**; `compute.profile` is
  always explicit, even with one profile configured. Compute is the one choice
  that spends shared allocation.
- ~~Missing storage config surface~~ → `[tune.storage]` added to §5.2; §4.3's
  routing table now has a config surface to reference.
# PhenoTypic MCP Server — §6 Errors, Limits, Safety, Testing

Status: **draft, reviewed once, revised**
Date: 2026-08-12

## 6.1 Error philosophy

**Errors are values.** A failed tool call returns `ok: false` with structured
issues — not an MCP protocol error. An agent that receives a protocol error can
only retry or give up; an agent that receives
`{"code":"unknown_param","path":"ops[0].params.sigmaa","hint":"Did you mean 'sigma'?"}`
can fix it. Protocol errors are reserved for malformed calls (bad JSON, missing
required argument).

Every issue carries `severity`, `code`, `message`, and where applicable `path`
and `hint`. `code` is a closed set — agents may branch on it; `message` is
human prose and may change.

## 6.2 Error codes

| Code | Severity | Raised when |
|---|---|---|
| `path_outside_workspace` | error | A path argument resolves outside `SandboxRoot` |
| `not_found` | error | An id names no artifact |
| `already_exists` | error | `put` without `overwrite` |
| `unknown_class` | error | Operation/scorer/strategy class not resolvable; carries did-you-mean |
| `unknown_param` | error | `extra="forbid"` rejected a field; carries did-you-mean |
| `invalid_param` | error | Pydantic validation failed (type, bound, validator) |
| `missing_param` | error | A required field was absent (`missing`) — distinct from `invalid_param` |
| `arm_artifact_drift` | error | An arm's `.pht-tune` or pipeline digest changed between `campaign_put` and `campaign_start` (§8.2) |
| `finalize_incomplete` | error | A `finalize_in_progress` marker is present; the study directory cannot be trusted (§4.5) |
| `study_not_finished` | error | `tune_export_best` on a distributed study whose budget is not drained and whose jobs are still live (§4.5) |
| `stage_order_hint` | **advisory** | GUI DAG validator hint; never blocks |
| `pipeline_empty` | error | Neither ops nor measurements — the CLI's own check |
| `stale_target_ref` | error | A `select` ref built against a superseded pipeline digest |
| `no_active_knobs` | error | Tune spec varies nothing |
| `grid_needs_stepped_domain` | error | `grid` strategy with a continuous `FloatRange` |
| `scorer_unavailable` | error | `scorer.availability()` is `False` |
| `scorer_not_portable` | error | `QCScorer` check not configured from a path |
| `distributed_storage_unavailable` | error | SLURM tune with no safe shared storage (§7 P1 not landed) |
| `storage_url_collision` | error | Another live study resolves to the same storage URL |
| `unknown_profile` | error | `compute.profile` names no configured profile |
| `param_not_overridable` | error | Override key absent from the profile's `overridable` |
| `cap_exceeded` | error | Override violates a cap; names key, request, cap |
| `reserved_sbatch_key` | error | Agent tried to set `array`/`output`/`error`/`job-name` |
| `profile_not_expressible` | error | A profile key the target CLI surface cannot carry — e.g. `account`/`qos`/`cpus_per_task` on the tune path, which accepts only four discrete SLURM flags (§5.2.1) |
| `plan_required` | error | `deploy_start` without a `plan_token` (§5.4) |
| `plan_stale` | error | Plan token's pipeline/images/compute digest no longer matches |
| `campaign_not_approved` | error | `campaign_start` on a `draft` campaign |
| `promotion_required` | error | `deploy_start {scope:"full"}` without a `promotion_token` (§10.5) |
| `promotion_stale` | error | Promotion token's pipeline or parent-dataset digest changed since the review |
| `campaign_arm_scope_full` | error | A campaign deploy arm targeting the full dataset — campaigns are subset-scoped (§10.4) |
| `amendment_exceeds_envelope` | error | A mid-campaign amendment outside the approved budget, profile, or scorer |
| `subset_required` | error | A subset-scoped tool given a raw parent path instead of a `subset_id` (§10.3.1) |
| `subset_too_small` | error | Subset below `min_heldout_plates` (6): `derive_split` returns an EMPTY held-out set, so there is no gap at all |
| `subset_too_small_for_heldout` | **warning** | Subset under ~15: a single held-out plate makes the gap a one-sample estimate |
| `selector_unavailable` | error | `SubsetSelector.availability()` is `False` — e.g. `EmbeddingSubsetSelector`, which raises rather than falling back to random (§10.3) |
| `group_key_not_in_metadata` | error | `MetadataGroupSubsetSelector.group_key` names no column in `grouping_metadata`. **Carries the CSV's actual column list and a did-you-mean** — the agent cannot open the file itself, so an error without the columns leaves it guessing |
| `arm_scorer_mismatch` | error | A campaign arm's scorer differs from the campaign scorer (§8.2) |
| `screening_unsupported_on_slurm` | error | `--screen` + SLURM, which silently drops screening today. Ships: OQ-4.1 resolved to expose screening (default off), so the guard is needed (§7 P4). |
| `output_not_empty` | error | Deploy target non-empty; names `run_name`/`resume`/`restart` |
| `resume_incompatible` | error | Pre-validated `validate_resume_compatibility` mismatch; names the field |
| `scheduler_jobs_active` | error | Resume/restart while ledgered jobs are live |
| `local_slot_timeout` | error | `W1` waited past `probe_timeout_s` for the slot. Carries `held_by` (work class + id), `held_for_s`, and `queue_position`, so an agent can distinguish "retry in 30 s" from "a 2-hour deploy holds it" instead of retrying blind |
| `probe_cap_exceeded` | error | `n_images` above `limits.probe_max_images` |
| `environment_mismatch` | error | SLURM work requested in a `local` environment with no profiles |
| `submission_failed` | error | A subprocess exited non-zero or `sbatch` rejected a script for a reason no pre-check models; carries exit code, stderr tail, and **`retryable: bool`** classified from known transient-vs-config `sbatch` patterns, so an agent does not retry a bad account name verbatim |
| `scheduler_unreachable` | error | Detection said `slurm`, submission found otherwise; reports probe staleness |
| `not_owned` | error | Cancel targeted a run with no `RunRegistry` record here |
| `version_drift` | **warning** | Spec `phenotypic_version` ≠ installed; matches the engine's warn-only posture |

### Mapping engine exceptions onto the envelope

The codes above are the target; these are the sources.

**`pydantic.ValidationError` → one `issues` row per `.errors()` entry.** A single
`ValidationError` routinely carries several, and collapsing them would hide all
but the first. Per entry: `type` selects the code (`extra_forbidden` → `unknown_param`, `missing` →
**`missing_param`**, everything else → `invalid_param`), and `loc` becomes
`path`. `missing` gets its own code because `invalid_param` is defined as
"type, bound, validator" — none of which describe an absent required field.

**`loc` → `path` uses the agent's own addressing, not pydantic's.** Integers
become `[i]`, names join with `.`. The **nested-operation case does not carry a
second `params` segment**, which an earlier draft got wrong. Reproduced against
the real field:

```python
FilamentousFungiDetector.model_validate(
    {'inoculum_detector': {'class': 'OtsuDetector', 'params': {'sigmaa': 1.0}}})
# loc == ('inoculum_detector', 'sigmaa')      <- no 'params' between them
```

`OperationField`'s `BeforeValidator` (`sdk_/typing_.py:302-343`) calls
`cls.model_validate(value.get("params", {}))` with the params dict already
unwrapped, so the inner error's `loc` is bare and pydantic prepends only the
outer field name. The correct rendering is
`ops[3].params.inoculum_detector.sigmaa`.

This matters beyond cosmetics: §6.5 mandates one test per code asserting `path`
is populated. Written against the fictitious shape, those tests would encode the
wrong contract permanently.

Note also that §3.2's per-entry resolution keeps errors out of the final
`ImagePipeline(ops=[...])` assembly. That is what makes `ops[0]` addressing valid
at all — `ImagePipeline.ops` is a `Dict[str, ...]`, so an integer-indexed `loc`
could not arise from the assembly step, and a union-discrimination failure there
would inject validator-chain tags into `loc` and multiply the rows. The one existing formatter in the repo,
`_validation_messages` (`gui/tune/_setup_authoring.py:524-531`), instead does
`".".join(...)`, yielding `0.sigma`; that is the GUI's convention and does not
match the bracketed paths every example here uses. The projection is new code,
not reuse.

**`hint`** is populated by `difflib.get_close_matches` for `unknown_param` /
`unknown_class` (against `model_fields`) **and for any closed-set check whose
candidate set the agent cannot see** — notably `group_key_not_in_metadata`,
where the candidates live in an external CSV no tool reads back. It is otherwise
absent; never a generic string. The rule is: **if the valid values exist and the
agent has no way to obtain them, the error carries them.**

**Subprocess and scheduler failures.** Not every failure is pre-checkable; §5.4
pre-validates resume compatibility precisely because the CLI's own failure mode
is `sys.exit(1)` inside a subprocess. Anything that escapes those checks maps to
`submission_failed`, carrying the captured stderr tail and the exit code, rather
than falling through the closed set. This covers an `sbatch` that passed the
liveness probe but rejected a specific script (an account or QoS the profile caps
did not model), and any CLI traceback the pre-checks did not anticipate.

## 6.3 Limits

| Limit | Default | Enforced by |
|---|---|---|
| `probe_max_images` | 4 | Hard cap on `pipeline_probe.n_images` |
| `probe_timeout_s` | 300 | Wall clock on `W1`, including slot wait |
| Local compute concurrency | 1 | `LocalComputeSlot` (§1.5) |
| Catalog list size | unbounded rows, compact fields | `catalog_operations` returns no schemas |
| Measurement payloads | summary only | `describe()` + column names; parquet path for the rest |
| Log tail | 200 lines | `LocalRunner.snapshot_log` |

## 6.4 Safety boundary

The server runs as the user with the user's filesystem and scheduler rights.
There is no authentication — its boundary is the workspace sandbox plus these
explicit refusals:

1. **No `--overwrite`.** `shutil.rmtree(output_dir)` is not reachable from any
   tool. Destroying measurements, HDFs, and human QC curation stays a shell
   decision (§5.4).
2. **No raw sbatch.** Partition, account, and QoS come only from config;
   overrides are allow-listed and capped (§5.2).
3. **No cross-session cancellation.** Cancel requires a local `RunRegistry`
   record (§5.6).
4. **No path escape.** Every path resolves through `SandboxRoot` first.
5. **No credential handling.** Storage URLs are password-less by construction —
   `_reject_password_in_url` is the engine's chokepoint and the server never
   accepts a password argument. Postgres secrets stay in `~/.pgpass`.
6. **No implicit env inheritance for storage.** `$PHENOTYPIC_TUNE_STORAGE_URL`
   is never allowed to select storage silently (§4.3).

**Untrusted content.** Image filenames, dataset directory names, metadata CSV
contents, and docstrings from third-party operation classes all flow into tool
results. They are data, not instructions. Docstrings in particular are rendered
into `catalog_operation_detail` — a hostile third-party operation package could
place directives there. The server does not sanitize prose (that would corrupt
legitimate documentation), so this is recorded as a known property: **an agent
must treat catalog text as documentation, not as instruction.**

## 6.5 Testing

The project's test-integrity rules bind here, and they bind **generally**, not
only where this section calls them out:

> **Every test below must be proven able to fail** — by reintroducing the bug it
> guards, or by a one-line mutation of the code under test — before it is
> trusted. A cap-enforcement test that would keep passing with the comparison
> deleted proves nothing. A check that cannot run must **fail**, not skip.

An earlier draft applied this only to the import-purity gate and the P1 script.
That was a misreading of a project-wide rule as a per-case one: the round-trip,
error-code, cap, and refusal tests below need the same treatment, and are the
ones most likely to rot into tautologies.

### Unit — the `_services` promotion (P2)

- **Import-purity gate.** A test that imports every `phenotypic._services.*`
  module in a subprocess and asserts `dash`, `dash_bootstrap_components`,
  `flask`, and `werkzeug` are absent from `sys.modules`. This is the test that
  makes the `_space.py` split (§1.4) permanent rather than aspirational. It must
  be shown to fail when a `import dash` is reintroduced.
- **Shim equivalence.** For each promoted symbol, assert the `gui.*` re-export
  and the `_services.*` original are the *same object* — catching the
  `_REGISTRY` double-singleton failure specifically.
- Existing GUI tests and the CI ledger gates (`FEATURES.md`, `WORKFLOWS.md`,
  smoke-capture) must stay green unchanged.

### Unit — tool layer

- **Round-trip:** agent-facing pipeline spec → envelope → `from_json` →
  `to_json` → the same envelope, including nested `OperationField` values and
  the `_make_unique` duplicate-class naming.
- **Every error code is reachable.** One test per code in §6.2, asserting the
  code *and* that `path`/`hint` are populated where the table promises them. A
  code with no test is a code that will regress silently.
- **Cap enforcement is exact:** at, one below, and one above each cap, including
  `time` in all three accepted formats (`240`, `04:00:00`, `0-04:00:00`).
- **Refusals are refusals:** no argument combination reaches `--overwrite`, a
  reserved sbatch key, or a non-overridable profile key.
- **The embedding placeholder actually raises.** `EmbeddingSubsetSelector`
  must be shown to raise `NotImplementedError` and to report
  `availability() == (False, ...)` — never to return a random selection. A
  placeholder that silently degrades would stamp `method: "EmbeddingSubsetSelector"`
  onto an artifact describing a subset with none of the claimed visual coverage,
  and nothing downstream could contradict it.
- **Sandbox escape, adversarially.** §6.4 has no authentication — `SandboxRoot`
  *is* the entire security boundary — so it gets an explicit hostile test rather
  than relying on the blanket one-test-per-code rule: `..` traversal, symlinks
  pointing outside, absolute paths, and the cross-platform cases this project
  must support (Windows drive letters, UNC paths, case-insensitive filesystems).
  Path-escape semantics are a classic cross-platform gap and a generic pass will
  under-cover them.
- **`deploy_plan` writes nothing under the output directory.** Assert the output
  directory is byte-identical before and after a plan call — this is what keeps
  the extracted `build_array_script_spec` (§5.3) from regressing back into the
  file-writing generator and silently breaking `output_not_empty`.

### Concurrency

- **`LocalComputeSlot` mutual exclusion:** interleave `W1` probes and a local
  `W3` submission from concurrent tasks; assert never more than one holds the
  slot. Must fail if the slot is removed from either path — this is the test
  that guards the §1.5 blocker fix.
- **Event loop stays responsive:** `W0` calls complete while a `W1` probe is in
  flight, proving the `run_in_executor` offload.
- **Restart reconciliation:** with a live orphan subprocess recorded in
  `RunRegistry`, a fresh server must claim the slot rather than admit a second
  local job; with a dead one, it must CAS to `failed`.

### Integration

- End-to-end on `load_synth_yeast_plate()`: put → probe → tune (local, grid,
  tiny budget) → export best → deploy (local, `--sample`) → status until
  terminal marker.
- **SLURM tests are marked** and require `sbatch` on `PATH`, following the
  existing `slurm` marker convention. They must **fail, not skip**, when the
  marker is selected and the scheduler is absent.

### The P1 gate (§7)

`optuna_journal_storage.py` must pass **with its temp directory on the target
cluster's shared filesystem** before the journal backend is enabled. Its C2
claim is currently proven only on local APFS. Additionally, the script must be
mutation-tested: swapping `JournalFileSymlinkLock` for a no-op lock, or removing
the pre-create step, must make it fail.

## 6.6 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-6.1 advisory volume~~ → **return every advisory kind the GUI DAG
  validator emits** (`fork`, `stub`, `required_aux`, `container_mode`,
  `missing_input`, `duplicate_input`, `stage_order_hint`, `unsupported_linear`,
  …). Curating them would mean the server deciding which of the engine's own
  findings an agent may see — a judgment call §9.1 places in a skill, not in the
  server. The skill teaches which advisories usually matter; the server reports
  all of them.
# PhenoTypic MCP Server — §7 Prerequisites and Rollout

Status: **draft, reviewed once, revised — P1 is significantly larger than first estimated**
Date: 2026-08-12

Seven changes land before the MCP server is fully useful. P1 is a substantial
engine change to `phenotypic.tune` needing its own spec; P2–P7 are smaller.

---

## P1 — NFS-safe `JournalStorage` backend for distributed tune

### The problem

A SLURM tune fleet needs storage every worker can write. Today both options are
unsatisfactory for an agent-driven workflow:

| Today | Problem |
|---|---|
| `sqlite:///<output>/.pht-tune-cache/study.db` (the default) | Documented-unsafe across nodes — `tune_distributed_hpcc.md:8-12`: "SQLite-WAL is unsafe on the network filesystems HPCC clusters use (NFS, Lustre) … several SLURM array workers writing the same `study.db` will corrupt it or lose trials." |
| `postgresql+psycopg://…` | Requires a user-space Postgres server as its own sbatch job, a `pgdata` allocation, `~/.pgpass`, and a `createdb` per study. Real operational weight for an agent trying four pipelines. |

**H1 — `--slurm` does not check which of the two you got.**
`_validate_slurm_request` (`tune/_tune_cli/_run.py:664`) validates strategy, spec
path, images dir, and worker count — never the storage backend. Verified by
grepping `sqlite`/`storage_url` across `tune/`, `_execution/`, and `gui/tune/`:
no path anywhere cross-checks scheme against `--slurm`. So `--slurm` with the
default SQLite URL submits straight into the documented corruption case.

**H2 — the study name is a hard constant.** `_STUDY_NAME = "tune_cost_v1"`
(`_run.py:73`), used by store, fleet, and worker with no parameterization by
output directory. The isolation unit for concurrent studies is therefore the
**storage URL**, not the study name — two studies on one Postgres database
silently attach to the same Optuna study and pool trials. And
`tune_distributed_hpcc.md:122` instructs `export PHENOTYPIC_TUNE_STORAGE_URL=…`,
which is a resolution fallback: with it set, *every* subsequent study merges.
That is exactly the agent fan-out case.

### The fix

Add Optuna's `JournalStorage` over a symlink-locked file backend, and make it
the default for distributed runs.

```python
from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock

lock = JournalFileSymlinkLock(str(journal_path))
storage = optuna.storages.JournalStorage(JournalFileBackend(str(journal_path), lock_obj=lock))
```

`JournalFileSymlinkLock` is the NFS-safe lock (symlink creation is atomic on
NFS; `JournalFileOpenLock` relies on `O_EXCL` semantics NFS does not reliably
provide).

**Default routing becomes:**

| Invocation | Storage |
|---|---|
| local, no `--slurm` | `sqlite:///<output>/.pht-tune-cache/study.db` — unchanged |
| `--slurm`, no explicit URL | `journal:///<output>/.pht-tune-cache/journal.log` — **new default** |
| explicit `postgresql+psycopg://…` | `RDBStorage` — unchanged, **supported but no longer recommended** (below) |
| explicit `sqlite://…` **with** `--slurm` | **hard error** — closes H1 |

Because the journal default derives from `output_dir`, **each study gets its own
journal file by construction** — the isolation local SQLite already has, now
extended to the distributed case. H2 stops being reachable by default. The MCP
server adds two belt-and-braces guards regardless (§4.3): it always passes
`--storage-url` explicitly, and refuses two live studies resolving to the same
URL.

### Why Postgres stays — and why it stops being the recommendation

**There is no "Postgres side" of the tune engine to remove.** Storage is one
call, `optuna.storages.RDBStorage(url=…)` (`_optuna_store.py:83-87`); sqlite and
Postgres are the *same line*, differing only by a WAL pragma applied when the URL
starts with `sqlite`. Every other mention of Postgres in `tune/` is a docstring
or an error message. Dropping Postgres would mean **adding** a rejection for
those URLs — active work to remove a capability, while the `RDBStorage` branch
has to stay anyway to serve the local sqlite default. The journal backend is a
third scheme *alongside* it, not a replacement for it.

So two reasons Postgres remains supported:

1. **L1 gating.** The journal default is not enabled until the negative control
   passes on the target cluster's shared mount. Until then Postgres is the only
   supported distributed path — a rollout reason with an end date.
2. **Existing users.** `tune_distributed_hpcc.md` already instructs standing one
   up, and `--storage-url` is generic by design: "PhenoTypic is backend-agnostic
   — it does not ship or depend on any particular database server."

**And one reason it does not remain the recommendation.** An earlier draft said
Postgres was "still right for very large fleets". That was asserted with no
measurement, and measurement does not support it (C6):

| workers | aggregate | per-worker |
|---|---|---|
| 1 | 143 trials/s | 143/s |
| 16 | 486 trials/s | **30/s** — a 4.7× collapse |

Per-worker write throughput *does* fall under lock contention. But that is the
wrong yardstick: **the lock is held for a journal append, not for image
evaluation.** A realistic PhenoTypic trial evaluates ≥2 images at seconds each,
so 8 workers produce roughly **1.1 trials/s** while the journal sustains
**376 appends/s** — about **330× headroom**. Even assuming NFS symlink-lock round
trips several orders of magnitude slower than APFS, the margin survives.

Contention is measurable and irrelevant at the timescale this workload runs at.
Postgres therefore stays *supported* — for the L1 window, for existing setups,
and because removing it would cost work — but the recommendation for a
distributed study becomes the journal backend once L1 passes.

### Blast radius — larger than first estimated

An earlier draft claimed the strategy layer's duck-typing made this "a small
blast radius". **That was wrong.** Optuna's string resolver wraps *any* storage
string in `RDBStorage`; there is no hook for a pseudo-scheme. Verified against
the pinned optuna 4.9.0:

```
optuna.storages.RDBStorage("journal:///tmp/x/journal.log")
  -> NoSuchModuleError: Can't load plugin: sqlalchemy.dialects:journal
optuna.storages.get_storage("journal:///tmp/x/journal.log")
  -> NoSuchModuleError: Can't load plugin: sqlalchemy.dialects:journal
```

So a **scheme-dispatch resolver is mandatory**, threaded through five
construction sites, not "a branch alongside RDBStorage":

| Site | Role |
|---|---|
| `tune/_tune_cli/_run.py:475` | Engine opens the store |
| `tune/_tune_cli/_run.py:785` | SLURM pre-create — the first thing a `--slurm` run does under the new default |
| `tune/_tune_cli/_worker.py:50` | Every SLURM worker — the entire point of P1 |
| `tune/_study/_optuna_store.py:106-109` and `gui/tune/_callbacks.py:871` | GUI Monitor, `create=False` |
| `tune/strategy/_optuna.py:239` | Fallback when the store exposes no `.study` |

Four further consequences, none of them in the first draft:

**B1 — the transient-retry predicate does not cover journal failures.**
`retry_on_transient_db_error` (`strategy/_optuna_support.py:260-319`) catches
only `sqlalchemy.exc.OperationalError`. Every `ask`/`tell`/user-attr call in the
loop (`strategy/_optuna.py:289,400,407,414,426,432`) depends on it for exactly
the condition P1 introduces: transient contention on a shared, jittery store. A
journal-backend transient — `OSError`/`PermissionError` from a flaky NFS,
`UpdateFinishedTrialError`, a `RuntimeError` from lock release — matches nothing
and crashes the worker on first occurrence. The predicate must become
backend-aware before P1 lands.

**B2 — a "read-only" Monitor open has a write side effect.**
`JournalFileBackend.__init__` does
`if not os.path.exists(file_path): open(file_path, "ab").close()` — verified.
So merely constructing the backend creates the file. A Monitor poll against a
not-yet-started study would touch a phantom `journal.log` into
`.pht-tune-cache/`, contradicting `OptunaStudyStore`'s documented `create=False`
contract ("load only an existing study"). It also loses the clean
`FileNotFoundError` UX SQLite gets, because `_open_live_study`'s pre-check
(`gui/tune/_callbacks.py:864`) is scheme-gated to sqlite and silently no-ops for
anything else. Both need an explicit existence check before construction.

**B3 — the Monitor's timeout bounding assumes file storage cannot network-hang.**
`_CONNECT_TIMEOUT_BACKENDS = {"postgresql"}` (`gui/tune/_callbacks.py:116-118`),
with the comment that SQLite is local and never network-hangs. P1's premise is
putting `journal.log` on the same NFS/Lustre mount the fleet writes. An
`os.path.exists()` or `open()` against a stale mount (default `hard` mounts,
a routine HPC failure) blocks at the syscall level indefinitely, and Python
cannot cancel a thread stuck there. `_LIVE_OPEN_POOL` is `max_workers=1`
(`:128-130`) — so one wedged probe poisons **every** subsequent Monitor poll for
**every** study for the life of the process. Either bound the file probe with an
OS-level killable mechanism or document and accept the risk explicitly.

**B4 — `journal.log` never compacts.** optuna 4.9.0 ships no compaction for the
file backend. A long study (hundreds of trials × full `pheno_params` /
`pheno_terms` payloads) grows monotonically, unlike Postgres which an operator
can vacuum. The P1 plan should state expected sizes for realistic MCP fan-out.

### What genuinely does not change

The strategy layer's heartbeat probes *are* storage-agnostic by duck-typing:

- `_heartbeat_interval` (`strategy/_optuna.py:57`) probes
  `getattr(storage, "get_heartbeat_interval", None)`, returns `None` when
  absent → `_start_heartbeat` returns `None` → no thread starts.
- `_record_heartbeat` (`:64`) returns early when `record_heartbeat` is absent.
- `_fail_stale_trials` wraps `optuna.storages.fail_stale_trials` in try/except.

Verified: `JournalStorage` has neither method; `RDBStorage` has both.

### Evidence, and the honest limit of it

`docs/superpowers/logic_validation_scripts/2026-08-12-phenotypic-mcp-server/optuna_journal_storage.py`
re-derives five claims directly from Optuna (never importing `phenotypic`).

```
ok   [C1]  optuna 4.9.0: JournalStorage + JournalFileSymlinkLock present
ok   [C2a] 4 processes x 15 trials — 60 trials persisted intact
ok   [C2b] negative control ran; see DISCRIMINATION note
ok   [C3]  journal: no heartbeat; RDB: interval=60 — stale reclamation LOST
ok   [C4]  fail_stale_trials is a safe no-op on a journal-backed study
ok   [C5]  journal:// rejected by RDBStorage AND get_storage; backend __init__ CREATES the file
```

**C2b is the important one, and on a local filesystem it reports
`DISCRIMINATION: NONE`.** An earlier version of this script asserted C2a alone
and was reported as passing. Mutation testing killed it: replacing the lock with
a no-op *also* passed, 60/60 trials, no loss. The reason is that
`JournalFileBackend.append_logs` does `open(path,"ab")` → one `write()` →
`fsync()`, and POSIX guarantees `O_APPEND` atomicity on a local filesystem
regardless of any application-level lock. On APFS/ext4 the test measures OS
write atomicity, not the lock — it could not fail, and a test that cannot fail is
worthless (project test-integrity rule).

The script now runs the negative control explicitly and says so.

**L1 — the gating step, as originally written.** Before P1 is implemented, run:

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-12-phenotypic-mcp-server/optuna_journal_storage.py \
    --dir /path/on/the/cluster/shared/mount --require-discrimination
```

`--require-discrimination` exits non-zero unless the no-op-lock control
**actually loses trials there**.

### L1 as written is unsatisfiable on a POSIX-coherent filesystem — RESOLVED by measurement

Run on UCR HPCC, that gate **cannot pass**, and the reason is not a defect in the
journal backend. Both `/bigdata` and `/rhome` are **GPFS**, not the NFS or Lustre
this section assumed. GPFS enforces POSIX byte-range semantics cluster-wide
through a distributed token manager, so `append_logs`' `open(ab) → write →
fsync()` is already indivisible and a no-op lock loses nothing. The gate demands
the control fail; on a filesystem that provides the guarantee itself, it never
will. **A gate that cannot be satisfied is not a gate — it is a blocker with a
misleading name.**

The original run (job 27466782) also could not have answered the real question:
`multiprocessing` places every worker on one host, so it measured the local
kernel's `O_APPEND` atomicity, never the distributed token manager a SLURM fleet
depends on.

**C7 (cross-node) settles it.** `run_l1_cross_node.sbatch` splits the suite into
`init` → N × `worker` → `verify`, lets `srun` place one worker per node, and
stamps each trial with its hostname so `--require-distinct-nodes` proves the run
was genuinely distributed. Result, job **27468703**, four nodes `c[07,09,12,14]`
on `/bigdata`:

```
ok [C7-symlink] 60 trials persisted intact across 4 nodes ['c07','c09','c12','c14']
ok [C7-noop]    60 trials persisted intact across 4 nodes ['c07','c09','c12','c14']
VERDICT: NO DISCRIMINATION, cross-node.
```

So: **journal storage is safe on this mount, and it is safe because of the
filesystem rather than because of the lock.** The symlink lock is redundant here,
not broken — and it ships enabled regardless, since it costs nothing and is what
makes the same code correct on an NFS deployment elsewhere.

**The corrected gate.** What P1 requires before shipping the journal default on a
given cluster is:

> The **symlink-locked** run survives a **cross-node** fan-out on that cluster's
> shared mount, with `--require-distinct-nodes` proving the workers really were
> on different hosts. The negative control's outcome is **informative, not
> required**: it losing trials shows the lock is load-bearing there; it surviving
> shows the filesystem is. Either way the locked run is what must pass.

Two honest limits on this evidence: it is 4 workers × 15 trials, not a 32-worker
fleet at scale (C6's ~65× throughput headroom is the argument that contention
stays irrelevant, and it is an argument, not this measurement); and absence of
loss over a finite sample is consistent with GPFS's architectural guarantee
rather than independent proof of it. **This result is filesystem-specific and
must be re-run on any cluster whose shared mount is not GPFS.**

### L2 — no heartbeat means no stale-trial reclamation

A worker killed by walltime or OOM leaves its trial `RUNNING` forever; under
`RDBStorage`, `fail_stale_trials` would reclaim it after the grace period.

Bounded, not fatal: `OptunaStrategy.is_exhausted()`
(`strategy/_optuna.py:441-456`) counts only `COMPLETE` and `PRUNED`, so orphaned
`RUNNING` trials do not stall budget drain — the fleet **reaches, and may
modestly overshoot,** its `n_trials`. (The overshoot is pre-existing and
identical under `RDBStorage`: `TuningEngine.optimize` (`_engine.py:109-116`)
does check-then-act on `is_exhausted()`, a TOCTOU race that concurrent workers
can overrun by up to N−1 trials. `is_exhausted`'s own docstring calls this
tolerated.) The cost is zombie rows and a slightly optimistic in-flight count.
Where that matters, Postgres remains available.

### Scope

P1 is an engine change to `phenotypic.tune`, not MCP work. It needs its own
spec, an implementation plan covering the five sites plus B1–B4, tests
(concurrent-worker, plus a mutation check that reintroducing SLURM+SQLite fails
the guard), and a rewrite of `tune_distributed_hpcc.md`. **The MCP server's
distributed-tune path is blocked on it; nothing else is.**

---

## P2 — Promote the Dash-free tier to `phenotypic/_services/`

Nine module moves plus one real split (§1.4). Watch items:

- **`gui/tune/_space.py` must be split** — it has `import dash` at `:33-34`.
- **`runs.py` is a three-file job.** `rehydrate_from_sandbox` → `classify()` →
  `gui/shell/_classifier.py:34` → `from phenotypic.gui.builder._directory_browser
  import IMAGE_EXTS` → `_directory_browser.py:20-21` imports Dash. `IMAGE_EXTS`
  is a bare suffix set; relocate it to `sdk_/_io_constants.py`.
- **`RunConsoleState` moves with `to_argv`**, or `_services/argv.py` imports back
  up into `gui/`.
- `gui/_operation_registry.py` holds a module singleton (`_REGISTRY`); the shim
  must re-export the *same* object, not construct a second.
- `discover()` is lazy today and must stay lazy.

Guarded by the import-purity test in §6.5, which must be shown to fail when an
`import dash` is reintroduced.

**Plus one extraction that is not a move.** `deploy_plan` (§5.3) must render an
sbatch preview without touching the run's output directory, but
`generate_array_job_script` (`_cli/_cli_slurm_array_scripts.py:116-368`) does
`mkdir` + `write_text` + `chmod` under `output_dir`. `SlurmArrayScriptSpec.render()`
is already pure; the ~150 lines that *build* the spec are entangled with the
write. Extract `build_array_script_spec(...) -> SlurmArrayScriptSpec` (no I/O)
and have both the real generator and `deploy_plan` call it.

## P3 — Catalog reconciliation and the JSON descriptor

Two pieces:

1. **One enumeration list.** `OperationRegistry.discover()` walks eight modules;
   `_find_class_in_phenotypic` resolves those plus `prefab`, `tune`,
   `tune.score`, `tune.strategy`, `detect.nn`. Reconcile to a shared constant so
   the agent can reach NN/GPU detectors and prefabs (§3.1).
2. **A JSON descriptor projection** over `OperationInfo`/`ParamInfo` — plus the
   `header_scheme()`-dispatching column derivation for `produces_columns` and
   `catalog_measurements`, which is *not* the ~40-line job first estimated
   (`TEXTURE.get_headers()` raises `TypeError` without a `scale`; see §3.1).
3. **The `phenotypic/subset/` subpackage** — `SubsetSelector` ABC plus
   `RandomSubsetSelector`, `MetadataGroupSubsetSelector`, and the
   `EmbeddingSubsetSelector` placeholder (§10.3). Must be added to
   `_find_class_in_phenotypic`'s submodule list in the same change that
   reconciles it with `OperationRegistry.discover()`, so selectors resolve and
   serialize by bare class name like every other extensible class.
4. **A directory-level digest helper.** `campaign_status.comparable` (§8.3) must
   detect two arms tuned against different image sets, and nothing today can:
   `bytes_fingerprint`, `file_fingerprint` (`sdk_/_io_constants.py:154,166`) and
   `pipeline_content_digest` are all single-file, and `TuningSpec` records no
   dataset at all. A stable digest over sorted `(relative path, size, mtime_ns)`
   is sufficient and cheap. Until it lands, `comparable` reports dataset
   comparability as **unknown** rather than assuming it.

## P4 — Close the `--screen` + `--slurm` silent no-op

`run_tuning` branches into `_submit_slurm_fleet` and returns **before** reaching
its `if screen:` block, and `_worker.py`'s `run_worker` constructs
`TuningEngine(...).optimize(...)` with no `ScreeningController` at all. So
`--screen --slurm` today silently drops screening — no error, no warning, the
full unscreened space runs on the fleet.

This must become an explicit error before the MCP server exposes screening at
all, since an agent requesting screening + SLURM compute would otherwise get
silently different behaviour than it asked for. It also settles OQ-4.1: the
question is not merely whether to expose `--screen`, but that the combination is
broken today.

---

## P5 — Give the tune CLI a `--slurm key=value` surface

`python -m phenotypic` accepts free-form repeated `--slurm key=value`
(`phenotypicCLI.py:795`). `python -m phenotypic.tune` accepts only four discrete
flags — `--slurm-partition`, `--slurm-mem`, `--slurm-time`,
`--slurm-constraint` (`tune/__main__.py:104-125`) — and `_submit_slurm_fleet`
(`_run.py:797-805`) builds its `slurm_args` from those alone.

So **`account`, `qos`, `cpus_per_task`, and `gpus_per_node` cannot reach a tune
fleet at all.** On a cluster where `account` is mandatory every tune submission
is rejected; where it is optional, the work is silently billed to the default
account. Both engines already funnel into the same
`format_sbatch_directives` (`sdk_/slurm/_sbatch.py:102`), which handles arbitrary
keys — the narrowing is purely in the tune CLI's argument surface.

Adding `--slurm key=value` to the tune CLI (keeping the four existing flags as
sugar) collapses one profile to both paths and makes §5.2.1's expressibility
check vestigial. Small, self-contained, and independent of P1–P4.

Until it lands, the MCP server refuses inexpressible profiles rather than
dropping keys (`profile_not_expressible`), and `paths` in the profile config
declares which surfaces a profile is valid for.

---

## P6 — Subset staging

Neither engine accepts a file list: `tune`'s `-i` is documented "image
directory" and `_load_images` does a non-recursive `iterdir`
(`_run.py:235-279`); the forward CLI's `-i` is a single `click.Path` with no
`multiple=True` and feeds `scan_directory_structure`
(`_cli_directory_scanner.py:28-117`). So §10's whole subset boundary depends on
the server **materializing** a staging directory that mirrors the parent's
dataset substructure — symlinks by default, copies where symlinks need
privileges (Windows), keyed by subset digest so concurrent arms share one.

Small but genuinely new, and it has a cross-platform edge the rest of the design
does not. §1.6's reuse inventory missed it.

---

## P7 — The distributed finalize

§4.5's distributed finalize has **no code to build from**. There is no
`finalize`/recompute entry point anywhere in `tune/` or `gui/tune/`;
`_finalize_outputs`, `_finalize_best_params`, and `_finalize_pareto_outputs`
have no call sites outside `run_tuning` itself, and the `--recompile` referenced
by a stale docstring (`_run.py:744`) does not exist on the tune CLI.

So it is a new entry point that opens a store, gates on the study being terminal,
writes the four artifact groups in the existing order behind a
`finalize_in_progress` marker, and is safe to re-run. Modest, but it is
engine-adjacent work rather than server glue, and it was previously described as
if it were assembling existing pieces.

Related new-build risk worth naming: the **killable subprocess** §4.4 requires
for store opens has no precedent either. The nearest analogue, the GUI Monitor's
live-open pool, is explicitly documented as *non*-killable — a timed-out thread
is abandoned and its single-worker pool stays poisoned for the process lifetime
(`gui/tune/_callbacks.py:915-930`). That is exactly why the spec calls for a
subprocess, and exactly why there is nothing to copy.

---

## Rollout order

```
P2 (_services promotion)   ──┐
P3 (catalog + descriptor)  ──┤
P4 (--screen guard)        ──┼──> MCP v1: catalog + pipeline + probe + campaigns
P6 (subset staging)        ──┤        + local tune + deploy (W0/W1/W3)
P7 (distributed finalize)  ──┘
                              (P5, tune --slurm k=v, is independent — it only
                               retires §5.2.1's expressibility check)

L1 (negative-control run on the real cluster mount)
      └──> P1 (JournalStorage: 5 sites + B1–B4)  ──> distributed tune (W2 on SLURM)
```

**P6 is v1-critical, not optional infrastructure.** An earlier version of this
diagram omitted it, which would have led an implementer to build P2–P4, believe
v1 was reachable, and find that *every* subset-scoped tool refuses: §10.3.1 makes
`subset_id` mandatory for `tune_start`, `campaign_put`, and `deploy_start`, and a
subset cannot reach either engine without the staging directories P6 builds.

P7 is listed under v1 because **a distributed study is reachable in v1 without
P1**: §4.3's storage-routing table admits a SLURM tune whenever Postgres is
configured, journal backend or not. That study takes the `if slurm: return
_submit_slurm_fleet(...)` early exit, so it never runs the inline finalize, and
without P7 it can never export a winner.

Not because a *local* study needs it — a local study's
`_finalize_outputs` / `_finalize_pareto_outputs` / `_finalize_best_params` run
inline inside `run_tuning`, so its `best_params.json` exists the moment the run
finishes and `export_best_from_run` works with no new code (§4.5). An earlier
version of this paragraph claimed the opposite.

MCP v1 ships without P1. Building pipelines, probing them, planning campaigns,
tuning **locally**, and deploying to SLURM all work with today's engines. Only
**distributed tuning** waits — and until then a SLURM tune request returns
`distributed_storage_unavailable` naming Postgres as the supported path, rather
than silently submitting into H1.
# PhenoTypic MCP Server — §8 Workflow, UX, and Campaigns

Status: **draft, reviewed once, revised**
Date: 2026-08-12

## 8.1 The intended UX

The server is not designed for an agent to wander a parameter space unattended.
The workflow is **collaborative planning, then delegated execution**:

```
Phase 0 — TRIAGE (human + agent)
  Characterize the ASSAY before any pipeline exists: organism morphology
  (filamentous / round / mixed), colony-vs-background contrast, colony
  separation, plate format, imaging modality. Some of this only you know;
  some is measurable from a probe. Produces assay.json (§9.3).

  ALSO establishes the DEVELOPMENT SUBSET (§10) — you name it, or the agent
  samples it with a recorded method. Everything below runs on the subset;
  the full dataset is touched only after a separate promotion gate.
        │
        ▼
Phase 1 — PLAN (human + agent, conversational, W0 + bounded W1)
  Driven by the assay profile: prefab pipelines FIRST (§9.4), probed and
  compared, tuned before anything custom is authored. Nothing is submitted;
  every tool here is read-only or writes only draft artifacts.

  Phase 1 has TWO modes, and the agent should say which it is in:

    1a EXPLORE (§8.7) — the next step depends on what the last one showed.
       patch → probe{stages} → read evidence → keep/revert. Bounded and
       trailed. Use when you cannot yet name the arms.

    1b CONVERGE — the candidates are known. Write them down as a campaign.

  Exploration ends exactly when the arms can be named; that transition is
  the agent's to declare and yours to accept.
        │
        ▼
  A CAMPAIGN: the agreed set of arms, written down, reviewed by you.
        │
        ▼  ← the human checkpoint lives HERE, once, not at every submission
Phase 2 — EXECUTE (agent, autonomous, W2/W3)
  ONE `campaign_start` call launches every arm. Subagents, where used, built
  the arms during Phase 1 — they do not launch or poll individually. The
  orchestrator polls `campaign_status`. You are not in this loop.
        │
        ▼
Phase 3 — REPORT (agent → you)
  One leaderboard across arms, with the winner's provenance and the evidence
  behind the recommendation.
```

This shapes three design choices that would otherwise look arbitrary:

1. **The `deploy_plan` → `deploy_start` gate is satisfied at campaign
   granularity**, not per submission. Requiring a fresh plan for every arm would
   put a checkpoint in Phase 2, where you are deliberately absent. An approved
   campaign carries the plan token for each of its arms.
2. **Phase 1 must be cheap and side-effect-free.** Every planning tool is `W0`
   or a `dry_run`, so exploring twenty ideas conversationally costs nothing and
   touches no allocation.
3. **Phase 3 needs one call, not N.** An orchestrator polling three studies
   through three separate `tune_status` calls has to reassemble the comparison
   itself, and will compare costs whose scorers differ without noticing.

## 8.2 The campaign artifact

`<workspace>/campaigns/<name>/campaign.json`

```json
{
  "schema_version": 1,
  "name": "fungal-edge-sweep",
  "status": "approved",
  "created": "2026-08-12T13:40:02Z",
  "approved_at": "2026-08-12T13:52:17Z",
  "question": "Does phase-based edge detection beat the filamentous prefab on the low-contrast plates?",
  "assay": "assays/plates.assay.json",
  "subset_id": "subsets/plates-dev-24.subset.json",
  "metadata_csv": "data/tune_layout.csv",
  "objective": {"scorer": {"class": "QCScorer",
                           // `metadata` is CORRECT — the shipped field name (§10.3)
                           "params": {"check": {"metadata": "data/tune_layout.csv"}}},
                "sense": "cost in [0,1], lower is better"},
  "budget": {"trials_per_arm": 200, "max_concurrent_arms": 3},
  "compute": {"profile": "cpu-bulk", "n_workers": 8},
  "arms": [
    {"id": "prefab-fil", "pipeline": "pipelines/filamentous-prefab.json.pht-pipe",
     "tune_spec": "tune/filamentous-prefab.setup.json.pht-tune",
     "pipeline_digest": "sha256:3e91…", "spec_digest": "sha256:c07b…",
     "rationale": "baseline — FilamentousFungiPipeline, the assay-matched prefab"},
    {"id": "phase", "pipeline": "pipelines/phase-edge.json.pht-pipe",
     "tune_spec": "tune/phase-edge.setup.json.pht-tune",
     "rationale": "the hypothesis",
     "prefab_baseline": {"pipeline": "FilamentousFungiPipeline", "best_cost": 0.31}},
    {"id": "watershed", "pipeline": "pipelines/watershed.json.pht-pipe",
     "tune_spec": "tune/watershed.setup.json.pht-tune",   "rationale": "control"}
  ]
}
```

**Arms reference the assay, and custom arms cite the prefab they beat.**
`prefab_baseline` is the §9.4 convention: an arm whose pipeline is not a prefab
or prefab derivative records which prefab came closest and how it scored. The
server validates the field's *shape* and that the referenced study exists, but
does not require it — "custom before prefab is usually premature" is judgment,
which §9.1 places in a skill, not in the server.

**One scorer for the whole campaign.** Arms are only comparable if they are
scored the same way; a campaign whose arms carry different scorers produces a
leaderboard that means nothing. `campaign_put` takes the scorer at campaign
level and **rejects** an arm whose tune spec disagrees, with
`code: "arm_scorer_mismatch"`. This is the single most valuable invariant the
campaign concept adds.

### How that comparison is actually computed

Naming the mechanism is not pedantry — **both obvious implementations are
wrong**, and each fails in a different direction. Reproduced against the real
classes:

```python
s1 = QCScorer(check=ExpectedVsDetectedCount(metadata=<DataFrame>, groupby=[...]))
s2 = QCScorer(check=ExpectedVsDetectedCount(metadata=<same DataFrame>, groupby=[...]))

s1 == s2
# ValueError: The truth value of a DataFrame is ambiguous.
```

`ExpectedVsDetectedCount.metadata` is typed `pd.DataFrame | str`
(`analysis/qc/_expected_vs_detected.py:42-58`) — a public pydantic field, so
pydantic's generated `__eq__` walks into it and pandas raises.

The natural workaround is worse:

```python
s1.model_dump(mode="json")   # {"check": {"metadata": None, ...}}
s3.model_dump(mode="json")   # s3 built from a COMPLETELY DIFFERENT layout
s1_dump == s3_dump           # True
```

`_serialize_metadata` (`_expected_vs_detected.py:253`) emits `None` for any
DataFrame-backed metadata, so a dict comparison collapses every DataFrame-backed
scorer to one sentinel and declares them all equal — **silently producing
exactly the meaningless leaderboard this invariant exists to prevent.**

So the mechanism is ordered, and the order is load-bearing:

1. **Reject non-portable scorers first.** Any scorer whose `model_dump(mode="json")`
   contains a round-trip-lossy `None` where a source was configured is rejected
   with `scorer_not_portable` (§4.2) — *before* any comparison is attempted. This
   is already required independently, because a SLURM worker reloads the spec
   from disk and a DataFrame-backed check cannot be reconstructed
   (`model_validate({"metadata": None, …})` raises with "a check serialized from
   an in-memory DataFrame has no source path to round-trip").
2. **Then compare `model_dump(mode="json")`** — safe, because every surviving
   scorer is path-configured and serializes faithfully.
3. **Never use `==` on scorer objects.** Not anywhere, not as a shortcut.

**Validation is bound to bytes, not to a moment.** `campaign_put` checks each
arm's `.pht-tune` file, but `tune_start` re-parses that file from disk at launch
(`tune/__main__.py:200`) and `tune_put_spec {overwrite: true}` can rewrite it in
between. Without a binding, a rewritten scorer would launch unchallenged and the
drift would surface only in `campaign_status` — after the compute was spent,
which is precisely what the invariant exists to prevent.

So `campaign_put` records **`spec_digest`** and **`pipeline_digest`** per arm on
the campaign artifact, and `campaign_start` re-hashes and refuses on mismatch
(`arm_artifact_drift`). This is the same stale-digest pattern already used for
`tune_space` → `tune_put_spec` refs and for plan tokens.

The pipeline digest matters independently: `TuningSpec.pipeline` is **embedded**
(`tune/_spec.py:165`), not referenced, so `campaign_put`'s two checks — "the
arm's `pipeline` path loads" and "the arm's tune spec constructs" — validate two
different objects that nothing compares. Editing `pipelines/<name>.json.pht-pipe`
after `tune_put_spec` snapshotted it means the campaign validates a pipeline the
arm will never run.

`campaign_put`'s validation order partly self-defends today — arms' tune specs
are reloaded from `.pht-tune` JSON, and MCP arguments arrive as JSON, so a live
DataFrame cannot reach the request path. But that is incidental, not designed:
any future code comparing a campaign's declared scorer against an already-parsed
`deliverables/tuning_spec.json.pht-tune` object would hit case 1 directly.

**`status` is provenance, not security.** The server cannot verify that a human
approved anything; `campaign_approve` is a call the agent makes *after* you say
so in chat. It is recorded so the transcript and the artifact agree, and so
Phase 2 has a checkable precondition — not because it authenticates you.

## 8.3 Campaign tools

Five tools, bringing the total to 32.

### `campaign_put` (`W0`) — draft the plan

Takes the campaign body above; defaults `status: "draft"`. Validates
**everything, submitting nothing**:

| Checked | How |
|---|---|
| Every arm's pipeline loads and is non-empty | `ImagePipeline.from_json` + the CLI's own emptiness check |
| Every arm's tune spec constructs | Real `TuningSpec` construction — all validators fire (§4.0) |
| All arms share the campaign scorer | Structural comparison (§8.2); `arm_scorer_mismatch`. **The arm's `.pht-tune` digest is recorded on the campaign**, and re-verified at `campaign_start` |
| Scorer is available and portable | `availability()` + the `QCScorer` path rule (§4.2) |
| Compute profile exists; overrides within caps | §5.2 |
| Arms resolve to distinct storage URLs | The H2 guard (§7) |
| `subset_id` resolves to a registered subset, non-empty, ≥ `min_heldout_plates` | The subset artifact (§10.2); a raw path is refused with `subset_required` |

The response is the **review document** — this is what you read before saying go:

```json
{"ok":true,"data":{
  "campaign_id":"campaigns/fungal-edge-sweep","status":"draft",
  "arms":[{"id":"otsu","n_knobs":4,"strategy":"tpe","trials":200,
           "pipeline_digest":"sha256:3e91…","spec_digest":"sha256:c07b…",
           "routed_to":"slurm","profile":"cpu-bulk",
           "estimate":{"node_seconds":6800,"basis":"probe: 3.4 s/image x 42 x 200/8"}}],
  "totals":{"arms":3,"trials":600,"est_node_hours":5.7,
            "concurrency":"3 arms x 8 workers = 24 tasks"},
  "objective":"QCScorer — cost in [0,1], lower is better",
  "pending_human_ack":true,
  "ack_prompt":"3 arms, 600 trials, ~5.7 node-hours on cpu-bulk. Approve?"},
 "issues":[{"severity":"warning","code":"needs_review_domain",
            "message":"phase arm: FocusEdgePhase.k has an inferred unbounded domain [0.5, 8.0]; inference guessed it.",
            "path":"arms[1].knobs[2]"}]}
```

`est_node_hours` is the number you actually want before agreeing to anything,
and its `basis` says whether it came from a real probe or a default.

### `campaign_approve` (`W0`)

`{campaign_id, human_response, note?}` → flips `draft` → `approved`, stamps
`approved_at`, **mints one `plan_token` per arm**, and appends a lineage row.

```json
{"ok":true,"data":{"campaign_id":"campaigns/fungal-edge-sweep","status":"approved",
  "approved_at":"2026-08-12T13:52:17Z",
  "plan_tokens":{"prefab-fil":"pl_7f3a…","phase":"pl_9c1e…","watershed":"pl_4b0a…"}}}
```

`human_response` is **required** and carries what the human actually said, which
is then recorded on the artifact and in lineage.

This does not authenticate anything — §8.2 is explicit that status is provenance,
not security, and an agent could fabricate the field. What it changes is the
failure mode: with `pending_human_ack: true` on the `campaign_put` response and a
required `human_response` here, approving without asking becomes an **explicit
fabrication** rather than an omission an agent can drift into. An agent that
never loaded the skill still gets a machine-readable signal that something is
waiting on a person. Refuses if any
blocking issue from `campaign_put` is unresolved, so approval cannot outrun
validation.

**Tokens are minted here, never by `campaign_put`.** An earlier draft showed
populated `plan_tokens` in `campaign_put`'s `draft` response — which would have
let an agent take a draft campaign's tokens straight to `deploy_start`, skipping
approval and the human checkpoint entirely. §8.2 is explicit that `status` is
provenance rather than security, so the gate has to be that the artifact an
unapproved campaign hands back contains nothing spendable.

### `campaign_start` (`W2`)

`{campaign_id, arms?}` → launches arms, honouring `max_concurrent_arms` and the
routing rules of §1.5. Refuses a `draft` campaign with
`code: "campaign_not_approved"`.

Each arm launches through the ordinary `tune_start` path — `RunRegistry.allocate`
→ `LocalRunner.start` → CAS — so campaign arms are ordinary studies, visible in
`workspace_list` and in the GUI. A campaign is an *organizing layer*, not a
parallel execution engine.

**Arm → study naming is explicit and persisted.** §2.2 forbids auto-suffixing,
so `campaign_start` does not invent names silently: each arm's study is
`studies/<campaign-name>-<arm-id>`, and the resolved `study_id` is **written back
into `campaign.json`** on the arm. A collision with an existing study is an
error naming both, not a silent rename. Without this, `campaign_status`'s
per-arm `study_id` would have no defined source.

**`campaign_start` snapshots the campaign it launched** rather than re-reading it
during fan-out, so a concurrent `campaign_approve` or an in-envelope amendment (§10.4) cannot
change the arm set mid-launch. Writes to `campaign.json` are atomic and CAS on
`status` (§2.6).

### `campaign_get` (`W0`) — read the stored campaign back

`{campaign_id}` → the `campaign.json` artifact verbatim: arms with their
`pipeline`, `tune_spec`, `study_id`, `rationale`, `prefab_baseline`, and the
`pipeline_digest` / `spec_digest` binding (§8.2), plus the objective, budget,
compute, subset, and assay references.

**This is the session-recovery entry point.** An agent resuming after a context
compaction typically holds one thing: a campaign id. `campaign_status` reports
*progress* per arm but not the artifact ids, so without `campaign_get` the only
route back to a winning arm's pipeline was to know, unprompted, to call
`workspace_lineage {id: study_id}` and trace `tune.start`'s `parent` — a path
this spec named nowhere.

**Recovery procedure**, stated once so it is not folklore:

```
campaign_get {campaign_id}        -> arms, their pipeline/tune_spec/study_id,
                                     subset_id, assay, objective
campaign_status {campaign_id}     -> where each arm actually got to
workspace_lineage {id: <study>}   -> only if you need the provenance chain
```

### `campaign_status` (`W0`) — one call, all arms

```json
{"ok":true,"data":{
 "campaign_id":"campaigns/fungal-edge-sweep","status":"running",
 "objective":"QCScorer — cost in [0,1], lower is better",
 "arms":[
  {"id":"phase","study_id":"studies/phase-edge","status":"running",
   "completed":126,"pruned":14,"failed":3,"budget":200,
   "best":{"trial":47,"score":0.081},"gap":{"value":0.06,"verdict":"ok"}},
  {"id":"otsu","study_id":"studies/otsu-base","status":"complete",
   "completed":200,"best":{"trial":180,"score":0.117},"gap":{"value":0.05,"verdict":"ok"}},
  {"id":"watershed","study_id":"studies/watershed","status":"failed",
   "completed":31,"failed":169,
   "error":"GridFinder found 0 rows on 169/200 trials"}],
 "leaderboard":[{"arm":"phase","score":0.081},{"arm":"otsu","score":0.117}],
 "comparable":true}}
```

**Polling economy.** `campaign_status` takes `since` (an opaque cursor from the
previous response). With it, arms whose state is unchanged collapse to
`{"arm":"otsu","unchanged":true}` and only movement is returned.

**The cursor is over the store artifact's stat, and it skips the store open.**
This is the load-bearing detail: §4.4 establishes that a per-arm leaderboard is a
`results`-class call requiring a killable subprocess per arm, and that this cost
is why polling must be infrequent. So the cursor embeds each arm's
`(path, mtime_ns, size)` for `trials.parquet` / `study.db` / `journal.log`, and an
arm whose stat is unchanged is reported `unchanged` **without opening its store at
all**.

Trimming only the response payload would have saved context tokens while leaving
the N-subprocess-opens-per-poll cost — and the wedged-mount exposure of §7 B3 —
exactly as before. Skipping the open is what makes `campaign_status {since}`
genuinely cheaper than a bare `campaign_status`, and therefore safe to call more
often. A multi-hour
campaign polled on a human timescale otherwise accumulates dozens of
near-identical multi-KB snapshots in the agent's context — the exact long-running
unattended workflow this design is built around is also the one most able to
exhaust context. The skill instructs retaining only the latest full snapshot.

`comparable` is false — with an explanation — when arms cannot be honestly
ranked. **A leaderboard that silently ranks incomparable things is worse than no
leaderboard**, so the field is mandatory in the response rather than inferred by
the reader.

Its three causes, and where each gets its data — because one of them needed
plumbing that did not exist:

| Cause | Data source |
|---|---|
| Scorer drift between arms | The campaign-level scorer vs each arm's resolved `deliverables/tuning_spec.json.pht-tune`, compared by the ordered mechanism in §8.2 |
| An arm failed too heavily for its best trial to mean anything | `failed` / `completed` counts from the study store |
| **Arms ran on different datasets** | The `tune.start` **lineage** event (§2.5) |

That last row is not free. `TuningSpec` has **no dataset field**
(`tune/_spec.py:162-171`) — `--images` is a launch-time CLI argument recorded
nowhere in the resolved spec — and no directory-level digest helper exists in the
codebase (`bytes_fingerprint`, `file_fingerprint`, and `pipeline_content_digest`
are all single-file). So §2.5 adds a `dataset` block to the `tune.start` lineage
event, and §7 P3 adds the directory-digest helper. Until both land,
`campaign_status` must report dataset comparability as **unknown**, not assume
it — claiming a comparison the artifacts cannot support is the failure this flag
exists to prevent.

`gap` surfaces the held-out generalization check, so an arm that won by
overfitting the calibration split is visible as such rather than crowned.

## 8.4 Phase 1 in practice

What the planning conversation actually looks like, tool by tool:

```
you:   "new Aspergillus set, low-contrast plates. Otsu is under-segmenting."

agent: [skill: phenotypic-assay-triage]
       → asks: morphology? expected colonies/plate? → you: "filamentous, 96"
       pipeline_put {name:"fil-prefab", from_prefab:"FilamentousFungiPipeline"}
       pipeline_probe {pipeline_id:"fil-prefab", …} to measure contrast/separation
       → fil-prefab now exists and is reused below; re-materializing it would
         return already_exists (§2.2 collision policy)
       → writes assay.json: morphology filamentous (human), contrast low (probe),
         separation touching (probe), 8x12 arrayed

agent: [skill: phenotypic-pipeline-construction — prefab-first]
       catalog_operations {category:"Prefab"}
       → assay says filamentous + touching → candidates:
         FilamentousFungiPipeline (3 ops), HeavyWatershedPipeline (15)
       pipeline_put {name:"watershed", from_prefab:"HeavyWatershedPipeline"}
       (fil-prefab already materialized during triage — reuse, do not re-put)
       pipeline_probe both on the same 2 subset images
       → filamentous prefab: 61 objects; watershed: 44. Neither near 96.
       → "I'd tune the filamentous prefab first rather than author anything new."

you:   "do that, but also try a phase-based edge arm as the hypothesis"

agent: pipeline_put {name:"phase-edge", …, dry_run:true}     # nothing written
       tune_space   {pipeline_id:"phase-edge"}               # 9 targets, QCScorer available
       pipeline_probe {pipeline_id:"phase-edge", subset_id:"subsets/plates-dev-24.subset.json", n_images:2}
       → "94 vs 61 objects on the two low-contrast plates; phase looks promising.
          Blur sigma domain is an inference guess — I'd narrow it to [1,3]."

you:   "agreed, and cap it at 200 trials each"

agent: campaign_put {…}
       → review document: 3 arms, 600 trials, ~5.7 node-hours, one warning

you:   "go"

agent: campaign_approve {campaign_id:"campaigns/fungal-edge-sweep",
                         human_response:"go"}
       campaign_start   {campaign_id:"campaigns/fungal-edge-sweep"}
```

Everything before `campaign_start` is `W0` or `W1` — no allocation consumed, one
`LocalComputeSlot` used briefly by the probe. The expensive, irreversible step
happens once, after you have seen the number.

## 8.5 Phase 2 fan-out

**Fan-out happens in Phase 1, not Phase 2.** The orchestrator spawns one
subagent per arm to *author* it — explore, probe, settle a pipeline and a tune
spec — and each owns exactly one arm's ids, which is why §2.2 requires distinct
explicit names rather than auto-suffixing.

**Launching is not fanned out.** A single `campaign_start` drives every arm
through `RunRegistry.allocate → LocalRunner.start → CAS`, and the orchestrator
polls one `campaign_status`. An earlier draft described Phase 2 as subagents
each launching and polling their own arm, which contradicted §8.3 and the
`phenotypic-tuning-campaign` skill's tool list (which omits `tune_start` and
`tune_status` entirely). A subagent handed only its own arm could not tell which
model applied, and would waste a premature `tune_start`. `tune_start` remains
available for a standalone study outside a campaign.

What keeps this safe is entirely in §1.5 and §2.4: the subagents' `W0` calls
interleave freely, their probes serialize on the one `LocalComputeSlot`, and
`RunRegistry.allocate` refuses two claims on one output directory. Nothing
about fan-out needs new machinery here.

The orchestrator polls `campaign_status`, not three `tune_status` calls, and
reports to you on completion or on the first arm that fails hard.

## 8.7 Incremental construction — the inner loop of Phase 1

Phase 1 as described so far assumes the plan is *knowable upfront*. Often it is
not: you add an enhancer, look at what it did to `detect_mat`, and **that result
determines what the next operation should be**. The plan is discovered, not
designed.

This is not a different workflow — it is the inner loop of Phase 1, and it must
be cheap, bounded, and auditable.

```
        ┌─────────────────────────────────────────────┐
        │  pipeline_patch   (add / tune one op)       │
        │        ↓                                    │
        │  pipeline_probe {stages: true}              │
        │        ↓                                    │
        │  read per-stage numeric evidence            │
        │        ↓                                    │
        │  decide: keep · revert · try different op   │
        └──────────────┬──────────────────────────────┘
                       │  exit when the arms can be named
                       ▼
              campaign_put  →  Phase 2
```

Every tool in the loop is `W0` except the probe, which is `W1`. So the loop costs
no allocation, and sibling subagents can each run their own loop — their patches
interleave freely and their probes serialize behind the one `LocalComputeSlot`.

### Why this needs stage evidence specifically

An agent cannot see an image. Given only a final object count, a failed step is
uninterpretable: 61 objects instead of 96 could mean the enhancer destroyed the
contrast, the detector's threshold is wrong, or the refiner merged neighbours.
`pipeline_probe {stages: true}` (§3.2) makes each hypothesis checkable
separately — `detect_mat.std` collapsing after a blur, `num_objects` before and
after a refiner — which is the difference between iterating and guessing.

### Bounds

Incremental construction is where an agent can most easily wander, so it is
bounded on three axes:

| Bound | Default | Why |
|---|---|---|
| Steps per exploration | 12 patches | Beyond this, the agent is guessing rather than converging; the tool result says so |
| Images per probe | `limits.probe_max_images` (4) | Two images is usually enough to see a step's direction |
| No-improvement streak | 3 | Three consecutive steps with no movement in the tracked signal ends exploration and reports what was tried |

These are **advisory limits reported in the response**, not refusals — the agent
is told it has exhausted its exploration budget and should either commit to a
campaign or ask you. Hard-refusing would strand a legitimately long exploration
mid-way with nothing written down.

### The construction trail

Every accepted step appends a lineage row, so the resulting pipeline explains
itself:

```json
{"event":"pipeline.step","id":"pipelines/edge-v3.json.pht-pipe","step":3,
 "edit":{"kind":"insert_op","slot":"ops","index":1,"class":"FocusEdgePhase"},
 "evidence":{"num_objects":{"before":61,"after":88},
             "detect_mat.std":{"before":0.04,"after":0.11}},
 "decision":"keep"}
```

This is what makes an incrementally-built pipeline defensible months later:
not "the agent produced this", but *which* step produced which improvement, on
what evidence. It is also what a `prefab_baseline` justification (§9.4) cites
when a custom pipeline finally does beat the prefab.

### Where it sits relative to prefab-first

The two compose rather than compete: **start from the assay-matched prefab, then
iterate from there.** The prefab is the starting point of the loop, not an
alternative to it — most explorations are "the prefab gets 61 of 96; what one
change closes the gap?", which is a far better-posed question than building from
an empty pipeline.

An exploration that begins from an empty pipeline should say why in its trail.

## 8.8 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-8.1 deploy arms~~ and ~~OQ-8.2 mid-campaign amendment~~ → **both granted,
  scoped to the development subset (§10.4)**. A campaign may carry a deploy arm
  and may replace a failed arm autonomously, provided the replacement stays
  inside the approved budget, profile, and scorer. Neither can reach the full
  dataset: that requires a separate human **promotion** (§10.5).

  Scoping development to a subset is what made both permissions safe to grant.
  The danger in "deploy arms" was never automation as such — it was unattended
  *full-dataset* compute on an unreviewed pipeline. Bound the loop to a subset
  and the campaign spends bounded, cheap compute, while the expensive
  irreversible step keeps its own gate.
# PhenoTypic MCP Server — §9 Separation of Responsibilities, and the Bundled Skills

Status: **draft, reviewed once, revised**
Date: 2026-08-12

## 9.1 The dividing principle

Sections 1–8 specify **mechanism**. They do not say what a *good* pipeline for a
filamentous fungus looks like, or that you should try a prefab before authoring
something new. That knowledge is real, it is what makes the difference between a
useful agent and a random-search machine — and it does not belong in the server.

The test for which layer a rule belongs to:

> **Could a well-intentioned expert reasonably want the opposite?**
> If **no** — the rule protects the cluster, the data, or correctness → it is
> **server enforcement**.
> If **yes** — it is domain judgment a knowledgeable person might override → it
> is **skill guidance**.

"Never expose `--overwrite`" passes the first test: no expert wants an agent
able to `rmtree` a run's deliverables from a tool call. "Try
`FilamentousFungiPipeline` before authoring a custom filamentous detector"
fails it: an expert who already knows this strain segments badly under that
prefab should skip it. So the first is a server refusal, the second is a skill
instruction.

Putting a rule in the wrong layer fails in a specific way:

| Mistake | Failure mode |
|---|---|
| Domain judgment encoded in the server | The server has opinions it cannot justify and you cannot override. Every new organism needs a server release. |
| Enforcement left to a skill | A subagent that ignores or never loads the skill can still melt the node or delete data. Skills are advice; advice is not a boundary. |

## 9.2 The four layers

| Layer | Owns | Never does |
|---|---|---|
| **Engines** (`_core`, `tune`, `_cli`) | Computation, serialization, scheduling, all numeric results | Know an agent exists |
| **MCP server** (§1–§8) | Mechanism, validation, routing, resource guards, refusals, structured errors | Encode domain judgment; pick an operation for you; decide a pipeline is "good" |
| **Skills** (this section) | Domain judgment, procedure, heuristics, what-to-try-first, how to read results | Enforce anything; bypass a server refusal; substitute for a validation |
| **Human** (you) | Assay knowledge the images cannot reveal, campaign approval, destructive operations | — |

The clean statement: **the server makes wrong things impossible; the skill makes
right things likely.**

## 9.3 Upfront assay characterization

Pipeline choice is driven by traits of the *assay*, not by the image file.
Several of the decisive ones cannot be measured from a plate image at all —
whether the organism is filamentous or yeast-like is something you know and the
agent does not.

So Phase 1 (§8.1) opens with an **assay triage**, producing a durable artifact:

### 9.3.0 The extensibility rule

Traits that narrow pipeline construction will keep being discovered — medium
opacity, incubation timepoint, dimorphic switching, plate lid glare. A schema
with a fixed field set would need a **server release per trait**, which is
precisely the coupling §9.1 exists to prevent.

So the artifact is an **open map of traits under a uniform envelope**:

> **The server validates the shape of a trait, never the set of traits.**

Adding a trait requires no server change, no schema bump, and no code — only a
new row in the skill-owned registry (§9.3.4).

`<workspace>/assays/<dataset>.assay.json`

```json
{
  "schema_version": 1,
  "name": "exfab-fungal-2026-08",
  "dataset": {"path": "data/plates", "digest": "sha256:1a4c…", "n_images": 480},
  "traits": {
    "organism.morphology": {
      "value": "filamentous",
      "source": "human",
      "note": "Aspergillus spp., hyphal spread expected by 72 h"
    },
    "colony.contrast_vs_background": {
      "value": "low",
      "source": "human",
      "note": "hyphal edges wash out against the opaque medium",
      "evidence": {"measure": "michelson_percell_median", "measured": 0.041,
                   "n_cells": 96,
                   "probe_ref": ".phenotypic-mcp/probes/filamentous-prefab/"}
    },
    "colony.separation": {
      "value": "touching",
      "source": "probe",
      "evidence": {"measure": "expected_vs_detected", "measured": 61, "expected": 96}
    },
    "plate.format":  {"value": "arrayed", "source": "human"},
    "plate.nrows":   {"value": 8,  "source": "human"},
    "plate.ncols":   {"value": 12, "source": "human"},
    "imaging.modality":  {"value": "flatbed_scanner", "source": "metadata"},
    "imaging.bit_depth": {"value": 16, "source": "metadata"},

    "medium.opacity": {"value": "opaque", "source": "human"}
  }
}
```

That last entry is the point: `medium.opacity` is a trait added *after* v1. It
required a registry row in the skill and **nothing else** — no server change, no
`schema_version` bump, no migration.

### 9.3.0.1 The trait envelope

Every entry in `traits` has the same shape, and this is the only structure the
server knows:

| Key | Required | Meaning |
|---|---|---|
| `value` | yes | Scalar or bool. The server does not interpret it. |
| `source` | yes | One of the four provenance values (§9.3.1). **The only closed enum.** |
| `evidence` | when `source: "probe"`; **optional corroboration otherwise** | `{measure, measured, …}` — the named measure and its number, so the claim is auditable and the bands recalibratable. A `human` value may carry evidence that *disagrees* with it; that disagreement is visible rather than silently resolved. |
| `note` | no | Free text for a human reader |
| `confidence` | no | Optional `[0,1]`; reserved for traits that later warrant it |

**Unknown trait keys round-trip verbatim.** The server preserves any trait it
does not recognize, rather than dropping or rejecting it — the opposite of
pydantic's `extra="forbid"` used elsewhere in this codebase, and a deliberate
inversion. An older server reading a newer skill's assay must not silently
discard a trait; silent loss of a trait that drove a pipeline decision would
make the artifact actively misleading. Forward compatibility is a correctness
property here, not a convenience.

**Trait keys are dotted and namespaced** (`<group>.<trait>`), so a new group
(`medium.*`, `growth.*`, `stress.*`) needs no structural change either.

### 9.3.1 Provenance vocabulary — the only enum the server enforces

Every field carries `source`, because the four ways of knowing are not equally
trustworthy and the agent must not blur them:

| `source` | Meaning |
|---|---|
| `human` | You told the agent. Authoritative; never overwritten by inference. |
| `probe` | Measured from images via `pipeline_probe` evidence. |
| `metadata` | Read from image metadata (bit depth, dimensions, EXIF). |
| `inferred` | The agent guessed. Must be stated as a guess wherever it drives a decision. |

**The agent asks for what it cannot measure and measures what it can.**

### 9.3.2 Domain vocabulary — what each term means and what it drives

This is the vocabulary the **skill** owns. The server never learns what any of
these words mean (§9.3.4).

**`organism.morphology`** — `filamentous | round | mixed | unknown`

| Value | Meaning |
|---|---|
| `filamentous` | Hyphal growth; irregular, non-convex, diffuse boundaries; colonies may merge into a mycelial mat |
| `round` | Yeast or bacterial colonies; approximately circular, convex, discrete |
| `mixed` | Both on the same plate — co-culture, dimorphic switching, or contamination |
| `unknown` | Not stated and not inferable |

*Determined by:* **the human, essentially always.** It could in principle be
inferred from a `Shape_Circularity` distribution, but only *after* a detection
exists — and the detector you would choose depends on the answer. That
circularity is why it must be asked, not guessed.

*Drives:* prefab choice (`FilamentousFungiPipeline` vs `RoundPeaks*`), and which
measurements are meaningful at all — filamentous assays want spatial/hyphal
metrics; round assays want size and shape.

**`colony.contrast_vs_background`** — `high | moderate | low`

*Determined by:* **the human in v1**, with a probe-measured number as
corroborating evidence. That split is not caution — it is what measurement
forced. See below.

*Measure:* **per-grid-cell Michelson contrast** at the Otsu split,
(μ_fg − μ_bg) / (μ_fg + μ_bg), reported as the median across cells.

```python
t   = skimage.filters.threshold_otsu(cell)     # per grid cell, not whole frame
fg  = cell >= t
mich = (cell[fg].mean() - cell[~fg].mean()) / (cell[fg].mean() + cell[~fg].mean())
```

### Why not Otsu's η — a measured refutation

An earlier draft specified η = σ²_B/σ²_T, "precisely the separability Otsu
maximizes, so it is principled rather than invented". The reasoning is correct
about what η *is* and wrong about what this trait *needs*. Measured on the three
bundled plate images (`contrast_trait_measure.py`, run against them):

| Claim | Result |
|---|---|
| **η is scale-invariant** | Reducing image contrast **20×** left η at `0.965` and Cohen's d at `10.435` — *numerically unchanged*. Both normalize by the very spread that reducing contrast shrinks. |
| **Michelson is not** | Same reduction: `0.2387 → 0.1443 → 0.0725 → 0.0364 → 0.0121`, linear in α. |
| **η has no dynamic range here** | Whole-frame `0.965–0.966` across 3 plates; per-cell p10–p90 = `0.945–0.963`, a span of **1.8% of the nominal [0,1] scale**. Three bands cannot be cut from that. |
| **Whole-frame Otsu measures the wrong thing** | It puts **46.1%** of pixels in "foreground" — far more than colonies occupy. The split is separating the *plate disc from the surround*, not colony from agar. |

That last row also explains something the review flagged as mere circularity:
`ReferenceFreeScorer._contrast` (`tune/score/_reference_free_scorer.py:377-409`)
needs `image.objmask` **because whole-image Otsu does not find colonies**. The
mask is not incidental to that implementation; it is what makes the number mean
anything. Measuring per grid cell — one colony and its local agar per cell —
removes the mask dependency without inheriting the plate-vs-surround artifact.

η would have shipped as a plausible-sounding trait that is *invariant to the
property it claims to measure*. Nothing downstream would have contradicted it:
every plate would have read `high`, and a genuinely low-contrast assay would too.

### Bands stay human-sourced (resolves OQ-9.4)

Every plate available in this repo is high-contrast — per-cell Michelson median
`0.233` (p10–p90 `0.225–0.239`). **One point does not calibrate a three-band
scale**, and inventing cut-points around a single anchor would repeat the η
mistake in a new coordinate system.

So in v1:

- `value` (`high | moderate | low`) is **`source: "human"`** — you know whether
  your plates are low-contrast, and §9.3.3 already establishes that
  human-sourced traits are the high-stakes uncheckable ones.
- `evidence` carries `{"measure": "michelson_percell_median", "measured": 0.233,
  "n_cells": 96}` as corroboration, so a human answer that contradicts the
  number is *visible* rather than silently overridden.
- `traits.yaml` records `calibration: uncalibrated` with the single known anchor,
  and the bands become derivable — as dataset-relative terciles or absolute
  cut-points — once a dataset spanning low contrast exists. That is a registry
  edit, not a server change (§9.3.4).

*Drives:* whether enhancement or detection is the bottleneck.

**`colony.separation`** — `well_separated | touching | confluent`

*Determined by:* **probe** — the fraction of detected objects sharing a boundary,
or expected count versus detected count on an arrayed plate.

*Drives:* watershed versus peak detection; how much refinement matters.

**`colony.pigmentation_informative`** — `bool`

*Determined by:* human, sometimes probe (channel separability).
*Drives:* `--detect-mode` (gray vs a colour channel or Lab), whether
`MeasureColor` earns its columns.

**`plate.format`** — `arrayed | unarrayed`, with `nrows` × `ncols`

*Determined by:* human or metadata.
*Drives:* `GridImage` vs `Image`, `--nrows/--ncols`, `GridSectionPipeline`, and —
critically — the expected counts that `QCScorer` scores against.

**`imaging.modality`** — `flatbed_scanner | camera | spimager | other`

*Determined by:* metadata or human.
*Drives:* `SpImagerPipeline`; and a camera implies vignetting, so
`FlattenIllumination` likely matters.

### 9.3.3 Failure modes, ranked by consequence

The reason this artifact exists. Note that the two worst failures are in the
fields the server **cannot** check and the human **must** supply — which is the
argument for asking rather than inferring.

| Wrong field | What happens | Caught by? |
|---|---|---|
| **`plate.nrows`/`ncols`** | `QCScorer` scores against wrong expected counts, so **the objective itself is wrong**. Tuning optimizes toward a false target and every arm's cost is meaningless — while looking perfectly healthy. | **Nothing.** The worst failure in the system. |
| **`morphology: round`** when filamentous | A peak detector finds one "colony" per dense region. Counts come out *plausible*, so QC may not flag it; the assay is silently under-counted. | Weakly — a size distribution with implausibly large objects |
| `morphology: filamentous` when round | Over-segmentation and hyphal metrics that measure noise | Probe object count far above expectation |
| `separation: well_separated` when touching | Merged colonies counted as one; size distribution skews high with a long tail | Probe: count below expected, size tail |
| `contrast: high` when low | Agent picks an Otsu prefab and tunes detector params, when the real fix was enhancement. Budget burned in the wrong subspace. | Probe: low object count, poor best-cost plateau |
| `pigmentation_informative` wrong | False negative loses signal; false positive adds noise columns | Low stakes either way |
| `imaging.modality` wrong | A worse prefab starting point | Low stakes; probe reveals it |

The pattern: **fields sourced `human` are high-stakes and uncheckable; fields
sourced `probe` are lower-stakes and self-correcting**, because the probe that
set them also surfaces the evidence that contradicts them. That asymmetry is why
`source` is mandatory and why a skill writing `source: "human"` for a value the
human never gave is the specific abuse to prevent.

### 9.3.4 The trait registry — the extension point

The skill owns a declarative registry, shipped beside it as **data, not prose**,
so traits and their routing rules can be added and audited without rewriting
procedure text.

`.claude/skills/phenotypic-assay-triage/traits.yaml`

```yaml
version: 3
traits:
  - key: organism.morphology
    values: [filamentous, round, mixed, unknown]
    determined_by: human           # ask; do not infer
    ask: "Is the organism filamentous (hyphal), round (yeast/bacterial), or mixed?"
    drives: [prefab_choice, measurement_family]
    failure: "round asserted for a filamentous organism yields plausible-looking
              counts from a peak detector — a silent under-count"
    stakes: critical

  - key: colony.contrast_vs_background
    values: [high, moderate, low]
    determined_by: human           # ask; the probe corroborates, it does not decide
    ask: "Are the colonies high, moderate, or low contrast against the agar?"
    measure:
      name: michelson_percell_median   # (mu_fg - mu_bg)/(mu_fg + mu_bg) per grid cell
      calibration: uncalibrated        # bands NOT derivable from one anchor
      anchors:
        - {dataset: "docs _dataset plates", value: 0.233, label: high, n_cells: 96}
      rejected:
        - {name: otsu_eta, why: "scale-invariant: unchanged across a 20x contrast
                                 reduction; per-cell span 1.8% of [0,1]"}
    drives: [enhancement_weight, detector_family]
    stakes: moderate

  # added in registry v3 — no server change, no schema_version bump
  - key: medium.opacity
    values: [clear, opaque, pigmented]
    determined_by: human
    ask: "Is the agar clear, opaque, or pigmented?"
    drives: [detect_mode, enhancement_weight]
    stakes: moderate

rules:                              # trait signals -> candidate prefabs (§9.4)
  - when: {organism.morphology: filamentous}
    prefer: [FilamentousFungiPipeline]
  - when: {organism.morphology: round, colony.separation: well_separated,
           colony.contrast_vs_background: high}
    prefer: [RoundPeaksPipeline]
  - when: {organism.morphology: round, colony.separation: well_separated}
    prefer: [RoundPeaksPipeline, HeavyRoundPeaksPipeline]
  - when: {organism.morphology: round, colony.separation: touching}
    prefer: [HeavyWatershedPipeline, HeavyRoundPeaksPipeline]
```

Three properties this buys:

1. **Adding a trait is a data change.** A registry row is reviewable in a diff
   and testable in isolation. Contrast a markdown table, where "add a trait"
   means editing prose the agent may or may not honour.
2. **Rules are separable from procedure.** The `rules:` block is the §9.4
   decision table in machine-readable form, so it can be extended, reordered, or
   contradicted by a site-specific overlay without touching the skill's method.
3. **Recalibration is a data change too.** Moving the η bands (OQ-9.4) edits one
   `bands:` line, not code and not prose.

The **server never reads this file.** It is skill data, exactly as the
biological vocabulary is skill knowledge.

### 9.3.5 What the server validates — envelope only

**The server validates the shape of a trait. It never validates biology, and it
never enumerates traits.**

| Server checks | Server does **not** check |
|---|---|
| File exists, parses, has `schema_version` | Whether `medium.opacity` is a real trait |
| Each `traits.*` entry has `value` and `source` | What `filamentous` means |
| `source` ∈ `{human, probe, metadata, inferred}` | Whether `low` contrast is plausible here |
| `evidence` present when `source: "probe"`, and its `probe_ref` resolves | Whether the probe supports the claim |
| Unknown trait keys are **preserved verbatim** | Whether 8×12 matches the plate |

**And the server never *acts* on a trait.** No trait value gates any tool's
behaviour anywhere in the catalog — not scorer choice, not subset requirements,
not GPU routing, not operation filtering. `assay_put` and `assay_get` are the
only tools that touch the artifact; `campaign_put` stores the `assay` reference
as a string without even checking the file resolves. **The assay is provenance
for humans and input for skills; it is not an interlock.**

That is deliberate (§9.1), but it should be read alongside §9.3.3's failure
table, which rates a wrong `plate.nrows` as the worst failure in the system with
"**Nothing**" catching it. The entire safety story for a `critical`-stakes trait
like `organism.morphology` rests on the skill being loaded and followed. There is
no server-side backstop if it is not — which is a materially different risk
posture than "the server validates the shape of a trait" might suggest on its
own.

This keeps §9.1 intact under extension: the only closed enum the server enforces
is `source` — **provenance, not biology**. Every biological vocabulary lives in
the registry and can grow, and the server's validation logic is *finite and
final*: it is written once against the envelope and never grows as traits do.

### 9.3.6 Adding a trait later — worked

Suppose plate-lid glare turns out to drive enhancement choice.

| Step | Where | Server change? |
|---|---|---|
| Add `imaging.lid_glare: [none, mild, severe]` to `traits.yaml` | skill | **no** |
| Add a `rules:` row preferring a glare-tolerant enhancer | skill | **no** |
| Skill starts asking about it in triage | skill | **no** |
| Existing assays without the trait keep validating | — | **no** — traits are individually optional |
| A newer skill's assay read by an older server | — | **no** — unknown keys round-trip |

The only thing that would force a server change is a new **provenance** kind —
say `source: "instrument"` — and that is a genuinely structural addition worth a
`schema_version` bump.

### 9.3.7 Scope

**One assay profile per dataset**, at `<workspace>/assays/<dataset>.assay.json`.
A workspace routinely holds more than one organism; a single per-workspace
profile is correct until the day you add a second, and then it is silently wrong
with no signal. Campaign arms reference the profile by path, so a leaderboard
stays interpretable months later: *these arms were chosen because the organism
was filamentous and contrast was low.*

## 9.4 Prefab-first construction

**Rule: try the relevant prefab pipelines before authoring a new one.**

Seven ship today (`phenotypic.prefab`), and they are validated, documented
chains rather than examples. Their real intents:

| Prefab | Intent (from its docstring) | ops |
|---|---|---|
| `RoundPeaksPipeline` | Round colonies, lightweight peak-based detection | **2** |
| `FilamentousFungiPipeline` | Filamentous fungi, `DenoiseBlockMatch` + spatial measurements | **3** |
| `SpImagerPipeline` | SpImager-sourced images | **4** |
| `GridSectionPipeline` | Per-section processing on grid plates | **13** |
| `HeavyWatershedPipeline` | Watershed segmentation for **touching** colonies | **15** |
| `HeavyRoundPeaksPipeline` | Round colonies, peak detection with full refinement | **18** |
| `HeavyOtsuPipeline` | Multi-stage Otsu thresholding with refinement | **19** |

Op counts are measured, not estimated, and they are ordered here because
**cost order is not obvious from the names.** `SpImagerPipeline` is labelled
"light" but includes `DenoiseBlockMatch` (BM3D) — the very op whose addition is
what marks the `Heavy*` variants heavy — so a probe of it is not cheap despite
the label. An agent ordering candidates by expected cost should use this column,
not the adjective in the docstring.

### Assay profile → candidate prefabs

This table is the human-readable rendering of the `rules:` block in
`traits.yaml` (§9.3.4). It is **skill data, not server logic** — exactly the kind
of judgment an expert may override, and extended by adding a rule rather than by
editing prose.

Rules are evaluated **most-specific-first**, and morphology dominates: it
constrains which detector family can work at all, whereas contrast and
separation only modulate how much enhancement and refinement are needed.

| Assay signal | First candidates |
|---|---|
| `morphology: filamentous` | `FilamentousFungiPipeline` (3) |
| `morphology: round` + `separation: well_separated` + `contrast: high` | `RoundPeaksPipeline` (2) — genuinely the cheapest that can work |
| `morphology: round` + `separation: well_separated` + `contrast: moderate\|low` | `RoundPeaksPipeline` (2), then `HeavyRoundPeaksPipeline` (18) |
| `morphology: round` + `separation: touching` | `HeavyWatershedPipeline` (15), then `HeavyRoundPeaksPipeline` (18) |
| `plate.format: arrayed` and dense | `GridSectionPipeline` (13) |
| `imaging.modality: spimager` | `SpImagerPipeline` (4) |
| `morphology: mixed` | Two arms — the filamentous and round candidates — rather than one compromise pipeline |
| `contrast: low` (modifier, not a rule) | Expect enhancement to matter more than detector choice; prefer the refinement-heavy variant of whichever family morphology selected |

`HeavyOtsuPipeline` (19 ops) is deliberately **not** a first candidate for any
signal. An earlier draft listed it under `contrast: high + well_separated` as
"cheapest that can work" — it is the *most* expensive of the seven, and that
same assay also matches the `RoundPeaksPipeline` row at 2 ops. Two overlapping
rows recommending pipelines 9× apart in cost is exactly the kind of error a
rules table makes visible and a prose paragraph hides. Reach for `HeavyOtsu`
when the cheaper family has been tried and failed, not first.

### The procedure

1. Pick candidate prefabs from the assay profile — usually one or two, three at
   most.
2. **Materialize each**: `pipeline_put {name:"fil-prefab",
   from_prefab:"FilamentousFungiPipeline"}`. A bare class name from the catalog
   is not a `pipeline_id` (§2.2 requires a sandbox path), so this step is not
   optional — it is what makes a prefab probeable.
3. `pipeline_probe` each on the same 2 images. Compare object counts, size
   distributions, and per-op timing.
4. Tune the best prefab **before** authoring anything custom. A prefab whose
   parameters are wrong for your assay is not a failed prefab; `tune_space` on a
   prefab is the cheapest large improvement available.
5. Author a custom pipeline **only** when the best tuned prefab still fails a
   stated bar — and record why.

**Custom pipelines carry a justification.** When an arm's pipeline is not a
prefab or a prefab derivative, the campaign records which prefab came closest and
how it failed:

```json
{"id":"custom-phase","pipeline":"pipelines/phase-edge.json.pht-pipe",
 "rationale":"FilamentousFungiPipeline tuned to 0.31 cost; under-segments hyphal
              edges on the low-contrast plates (probe: 61 vs ~95 expected objects).",
 "prefab_baseline":{"pipeline":"FilamentousFungiPipeline","best_cost":0.31}}
```

This is a **skill-enforced convention with server support**: the server supplies
the `prefab_baseline` field and validates its shape if present, but does not
refuse a campaign that omits it. The knowledge that a bare custom pipeline is
usually premature is judgment, not a boundary. What the server *does* guarantee
is that if the field is there, it is well-formed and its referenced study exists.

`catalog_operations` exposes prefabs (resolved OQ-3.1, §3.1), so the agent can
discover them rather than needing them memorized.

## 9.5 The bundled skills

Four skills ship with the server. Each maps to one phase and states its
tool sequence, so an agent that loads it knows both *what to do* and *which
tools do it*.

### `phenotypic-assay-triage`

**When:** at the start of any new dataset, before any pipeline exists.
**Produces:** `assays/<dataset>.assay.json` **and** `subsets/<name>.subset.json`.
**Tools:** `assay_put`, `assay_get`, `subset_put`, `subset_get`,
`pipeline_probe`, `catalog_operations`.
**Procedure:**

1. Ask the human for morphology and expected colony count per plate — these are
   not measurable (§9.3.3).
2. Read imaging metadata for modality and bit depth.
3. **Establish the development subset** (§10): ask the human to name one, or
   sample with a recorded method and seed. Everything downstream runs on it.
4. Probe 2–4 subset images to measure contrast (`michelson_percell_median`,
   §9.3.2 — **not** Otsu's η, which §9.3.2 refutes as scale-invariant) and
   separation.
5. `assay_put` with every trait carrying its `source`; `subset_put` with the
   measured `coverage` range.

**Hard rule it teaches:** never write `source: "human"` for something the human
did not say. A guess is `inferred`, and it must be stated as a guess wherever it
drives a decision.

### `phenotypic-pipeline-construction`

**When:** after triage, when choosing what to try.
**Procedure:** §9.4 — prefab-first, probe-compare, tune-before-authoring, and the
justification convention for custom pipelines.
**Tools:** `catalog_operations`, `catalog_operation_detail`, `pipeline_put`,
`pipeline_probe`, `pipeline_patch`, `pipeline_diff`.
**Hard rule it teaches:** a probe result is evidence about *those two images*.
Two probes are not a validation.

### `phenotypic-tuning-campaign`

**When:** turning candidates into a campaign.
**Procedure:** one scorer for all arms (§8.2); pick the scorer from what the
workspace actually has (`tune_space` reports availability); include a baseline
arm and a control arm, not only the hypothesis; size the budget from the probe
timing; narrow `needs_review` domains rather than accepting inferred bounds.
**Tools:** `tune_space`, `tune_put_spec`, `campaign_put`, `campaign_approve`,
`campaign_start`, `campaign_status`, `campaign_get`.

**Recovery:** resuming with only a campaign id, call `campaign_get` first — it
returns the arms' `pipeline`/`tune_spec`/`study_id`, which `campaign_status` does
not carry. Then `campaign_status` for progress, and `workspace_lineage {id}` only
if you need the provenance chain (§8.3).
**Hard rule it teaches:** cost is in `[0, 1]` and **lower is better**. Report the
held-out `gap`, never the calibration score alone — an arm that won by
overfitting the split is not a winner.

### `phenotypic-deploy-and-verify`

**When:** a winner exists and a full dataset is to be processed.
**Tools:** `deploy_plan`, `promotion_request`, `promotion_approve`,
`deploy_start`, `deploy_status`, `workspace_cancel`.
**Procedure:**

1. `promotion_request` — assemble the decision: winner provenance, subset score
   and held-out gap, measured full-dataset estimate, coverage warnings.
2. **Show the human that response and wait.** This is the gate; it is not
   optional and it is not something the agent can conclude on its own.
3. `promotion_approve` only after they say so, then
   `deploy_plan {scope:"full"}` → `deploy_start {scope:"full"}`.
4. Poll `manifest.json`, not exit codes — without `--wait` the CLI exits 0 on
   submission.
5. Verify against the mirror (`measurements.*`), never the master, and report
   `QC_MetadataOnly` rows separately from detections.

**Hard rules it teaches:** deletion and overwrite are the human's job at a
shell — if the output directory is occupied, ask rather than routing around it.
And a coverage warning on the promotion review is a reason to *say something*,
not a formality to pass through.

## 9.6 Skill/server boundary — worked cases

| Rule | Layer | Why |
|---|---|---|
| Path must resolve inside the workspace | **Server** | No expert wants otherwise |
| Only one local image computation at a time | **Server** | Protects a shared node |
| Named SLURM profile, capped overrides | **Server** | Protects a shared allocation |
| Campaign arms share one scorer | **Server** | Otherwise the leaderboard is meaningless — a correctness property |
| Try prefabs before custom pipelines | **Skill** | An expert may legitimately skip them |
| `HeavyWatershedPipeline` for touching colonies | **Skill** | Heuristic; assay-dependent |
| Include a baseline and a control arm | **Skill** | Good method, not a correctness rule |
| Ask the human for organism morphology | **Skill** | Procedural discipline |
| A `journal://` study must not share a URL with another live study | **Server** | Silent trial pooling is data corruption |

## 9.7 Packaging and installation

**Skills are authored in-repo** at `.claude/skills/phenotypic-*/SKILL.md`, so
they version in lockstep with the tool contract. This matters concretely: a skill
that instructs the agent to call `pipeline_diff` is *wrong* until that tool
exists, and only co-versioning makes that a reviewable diff rather than a
runtime surprise.

In-repo authoring alone does not reach anyone who installs PhenoTypic elsewhere,
so the server ships an installer:

```bash
uv run phenotypic-mcp setup            # detect harnesses, install skills + register server
uv run phenotypic-mcp setup --check    # report what is installed and whether it is current
uv run phenotypic-mcp setup --harness claude-code   # target one explicitly
uv run phenotypic-mcp setup --uninstall
```

This follows the pattern of other agent tools that pair a skill with an MCP
server (graphify is the reference here — one `SKILL.md` in the harness's skills
directory, alongside a stdio MCP server the skill drives).

**Behaviour:**

1. **Detect** installed harnesses rather than assuming one.
2. **Install skills** into each harness's own convention.
3. **Register the MCP server** in that harness's config, pointing at the
   `phenotypic-mcp` entry point from the current environment (an absolute
   interpreter path, matching how `get_python_command(for_slurm=True)` resolves
   `sys.executable` rather than a bare `python`).
4. **Idempotent and versioned** — re-running upgrades in place; each installed
   skill carries the `phenotypic` version it shipped with, and `--check` reports
   drift instead of silently serving a stale skill against a newer tool surface.
5. **Never clobber user edits** — a modified skill file is reported and skipped
   unless `--force`.

Claude Code's conventions are `~/.claude/skills/<name>/SKILL.md` plus an
`mcpServers` entry, and are the ones this spec states with confidence. **The
exact paths and config shapes for other harnesses must be verified against each
harness's current documentation at implementation time** rather than assumed
here — getting one wrong installs a skill nothing loads, which fails silently.
The implementation plan should treat per-harness support as one task each, with
`--check` as the acceptance test.

## 9.8 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-9.1 skill packaging~~ → in-repo authoring plus a
  `phenotypic-mcp setup` installer (§9.7).
- ~~OQ-9.2 assay scope~~ → **per-dataset**, at
  `<workspace>/assays/<dataset>.assay.json` (§9.3.7).
- ~~OQ-9.3 assay validation~~ → **structure and provenance only**. The server
  checks shape, required keys, and `source ∈ {human, probe, metadata,
  inferred}`; it never validates biological values, so the domain vocabulary can
  grow without a server release (§9.3.4).
# PhenoTypic MCP Server — §10 Development Subsets and the Promotion Gate

Status: **draft, reviewed once, revised**
Date: 2026-08-12

## 10.1 The subset is the unit of development

**Everything from triage through campaign execution runs on a subset. The full
dataset is touched exactly once, after an explicit human promotion.**

This single structural choice resolves the autonomy question recorded as OQ-8.1
and OQ-8.2 (§8.8). The
reason "let campaigns carry deploy arms" felt dangerous was that a full-dataset
SLURM run could launch on a pipeline nobody had looked at. Scope the development
loop to a subset and that risk disappears: an unattended campaign spends subset
compute, which is bounded and cheap by construction, and the expensive
irreversible step keeps its own gate.

```
Phase 0  TRIAGE     → assay + SUBSET                  (you + agent)
Phase 1  PLAN       → explore + converge ON THE SUBSET (you + agent)
   ▼  ── campaign approval ──
Phase 2  EXECUTE    → tune arms ON THE SUBSET          (agent alone, may amend,
Phase 3  REPORT     → leaderboard + winner              may carry deploy arms)
   ▼  ── ★ PROMOTION GATE ★ ── human, mandatory, separate ──
Phase 4  DEPLOY     → the full dataset                 (attended decision)
```

The two gates answer different questions, which is why one cannot substitute for
the other:

| Gate | Question it asks | When |
|---|---|---|
| Campaign approval (§8.2) | *Is this a sensible experiment to run?* | Before subset compute |
| **Promotion (§10.5)** | *Is this winner good enough to spend the full dataset on?* | Before full-dataset compute |

## 10.2 The subset artifact

`<workspace>/subsets/<name>.subset.json`

```json
{
  "schema_version": 1,
  "name": "plates-dev-24",
  "parent": {"path": "data/plates", "digest": "sha256:1a4c…", "n_images": 480},
  "selection": {
    "method": "MetadataGroupSubsetSelector",
    "params": {"n": 24, "seed": 0,
               "grouping_metadata": "data/plate_batches.csv",
               "group_key": "Metadata_Batch", "allocation": "equal"},
    "rationale": "8 batches x 3 plates each; equal allocation so the two rare
                  low-contrast batches are not swamped by the six common ones"
  },
  "images": ["plateA/plateA_01.tif", "plateA/plateA_07.tif", "plateB/plateB_03.tif", "…"],
  "n_images": 24,
  "digest": "sha256:77b2…",
  "coverage": {
    "measured_on": 4,
    "contrast_michelson": {"min": 0.031, "max": 0.094},
    "note": "spans low→moderate on the per-cell Michelson measure (§9.3.2);
             no high-contrast batch included"
  }
}
```

Three things it must record, because each one changes how much the results mean:

- **`parent` with a digest** — so a promotion can verify the full dataset has not
  changed since development, and so `campaign_status.comparable` (§8.3) has its
  dataset identity. `images` entries are **parent-relative paths**, not bare
  filenames: `scan_directory_structure` treats one level of subdirectories as
  separate datasets, so a bare name cannot disambiguate two datasets that both
  contain `plate_001.tif` (§10.3.1).
- **`selection`** — the selector class, its params, and its seed. A
  `RandomSubsetSelector` and a `MetadataGroupSubsetSelector` support very
  different confidence in the result, and a recorded seed makes either
  reproducible.
- **`coverage`** — what range of assay traits the subset actually spans, measured
  during triage. A subset that contains only easy plates will tune to a pipeline
  that fails on the hard ones, and nothing downstream can detect that from the
  cost alone.

## 10.3 Where subsets come from — the selector hierarchy

Subset selection is a **pluggable strategy**, following the same pattern as
every other extensible thing in this codebase: a pydantic ABC, concrete
subclasses, `{class, params}` serialization, and resolution by bare class name.

### `SubsetSelector` — the base class

New public subpackage `phenotypic/subset/`, added to
`_find_class_in_phenotypic`'s submodule list so selectors serialize and resolve
exactly like operations and scorers do.

```python
class SubsetSelector(BaseModel, ABC):
    """Choose a development subset from a parent image set.

    Args:
        n: Target subset size.
        seed: RNG seed; recorded on the artifact so a selection is reproducible.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    n: int = Field(..., ge=1)
    seed: int = 0

    @abstractmethod
    def _select(self, candidates: list[ImageRef]) -> list[str]: ...

    def availability(self) -> tuple[bool, str]: ...   # (usable?, why not)
    def cost_class(self) -> Literal["W0", "W1", "W2"]: ...

    def select(self, candidates: list[ImageRef]) -> SubsetSelection:
        """Template: check availability, delegate, then dedup, order, and
        record the rationale so the artifact explains itself."""
```

`SubsetSelection` is frozen and carries `images`, `method`, `params`, `seed`,
and a human-readable `rationale` — which becomes `selection` on the subset
artifact (§10.2).

Two methods worth calling out:

- **`availability()`** mirrors `Scorer.availability()` (§4.1), so
  `subset_generate` can report which selectors are usable *before* the agent
  commits — the same affordance that stops the most common tuning failure.
- **`cost_class()`** is what keeps an expensive selector from being smuggled
  into triage. Selection cost is not uniform, and the difference is structural:

  | Selector | Cost | Because |
  |---|---|---|
  | `RandomSubsetSelector` | `W0` | Needs only the file list |
  | `MetadataGroupSubsetSelector` | `W0` | Needs only the metadata CSV, which already exists |
  | `EmbeddingSubsetSelector` | `W2` | Must encode **every parent image** |

  An earlier draft deferred all stratification to a future iteration on the
  grounds that stratifying requires measuring the whole dataset. That was too
  blunt: it is true of *trait* and *embedding* stratification, and false of
  **metadata** stratification, where the grouping already exists on disk. So
  metadata sampling ships now.

### The three selectors

**`RandomSubsetSelector`** — uniform without replacement, seeded.

```json
{"class": "RandomSubsetSelector", "params": {"n": 24, "seed": 0}}
```

Honest and unstratified. The right default when no metadata exists, and the
right *baseline* even when it does.

**`MetadataGroupSubsetSelector`** — sample across metadata groups.

```json
{"class": "MetadataGroupSubsetSelector",
 "params": {"n": 24, "seed": 0,
            "grouping_metadata": "data/plate_batches.csv",
            "group_key": "Metadata_Batch",
            "allocation": "proportional"}}
```

| Param | Meaning |
|---|---|
| `grouping_metadata` | CSV supplying the grouping column. **Named distinctly on purpose** — three different CSVs appear in this spec: `deploy_plan.metadata_csv` (joined onto the output mirror), this one (subset stratification), and `QCScorer.check.metadata` (the expected counts the whole objective is scored against). Passing the wrong one at the scorer produces a meaningless objective rather than an error. **This naming choice does NOT extend to the other two** — see below. |
| `group_key` | The column in `grouping_metadata` naming each plate's group |
| `allocation` | `proportional` (mirror group sizes) or `equal` (same count per group, so a rare condition is not lost) |
| `min_per_group` | Floor per group; groups smaller than it are taken whole |

> **Do not "fix" `QCScorer.check.metadata` to match.** Verified by
> construction: `ExpectedVsDetectedCount(metadata=…)` succeeds and
> `ExpectedVsDetectedCount(expected_counts_csv=…)` raises `ValidationError`.
> A reviewer reading the disambiguation rationale above, without checking the
> code, proposed exactly this change — twice.

**Only a class this spec introduces may be renamed.** `grouping_metadata` is
this spec's choice because `MetadataGroupSubsetSelector` does not exist yet.
`ExpectedVsDetectedCount.metadata` **ships today with no alias**
(`analysis/qc/_expected_vs_detected.py:208`), so it keeps its name and is
disambiguated in prose only. A draft of this spec renamed it to
`expected_counts_csv` in two worked examples — a field that does not exist, which
would raise `missing` on `metadata` and `extra_forbidden` on the invention: the
exact failure §4.2's pre-submit checks exist to prevent, written into the
example.

**The selector performs its own CSV→filename join. It does *not* reuse
`_resolve_groups`.**

An earlier draft claimed it did — "the same vocabulary the tune split already
uses" — and that reusing it would guarantee the held-out split reached Tier 2
(whole-group hold-out) rather than the weaker within-group tier. **Both halves
are false, verified by reproduction.** `_resolve_groups`
(`tune/_evaluation/_split.py:114-133`) is a pure in-memory
`image.metadata.get(group_key)` lookup with no CSV and no join, and a freshly
read image carries only:

```
MetadataImage_BitDepth, MetadataImage_FileSuffix, MetadataImage_ImageName,
MetadataImage_ImageType, MetadataImage_UUID
```

`img.metadata.get("Metadata_Batch")` returns `None`. External CSV columns reach
data only through `join_metadata` (`_cli/_cli_output_manager.py:83-175`), which
operates on the **measurement DataFrame** inside `finalize_post_master_outputs`
— i.e. after a full pipeline run has measured every image, the exact opposite of
Phase-0 triage.

So the claimed payoff does not follow: `_resolve_groups` returns `{}` for any
externally-sourced key, and `derive_split` falls through to the within-group or
data-poor tier **silently** — no error, just a weaker generalization estimate
than the reader was promised.

What the selector actually does: read `grouping_metadata`, join rows to images
**by filename / parent-relative path**, and stratify on that. `group_key` names a
column in that CSV. It is a name shared with the tune split's vocabulary and
nothing more.

**If the held-out split should also benefit from the grouping**, something must
populate `image.metadata[group_key]` from the CSV at tune load time. That is an
engine change to `phenotypic.tune`, it is **not** in §7's P1–P7, and it is not
assumed anywhere in this design. Until it exists, stratifying a subset does not
change how the split is derived.

**`EmbeddingSubsetSelector`** — placeholder, and it **fails loudly**.

```json
{"class": "EmbeddingSubsetSelector",
 "params": {"n": 24, "seed": 0, "model": "<unset>", "strategy": "kmeans_medoids"}}
```

Intended shape: embed every parent image with a vision model, cluster, and take
medoids — giving visual coverage without any metadata or hand-labelling.

Until implemented, `availability()` returns
`(False, "EmbeddingSubsetSelector is not implemented; no embedding backend is configured")`
and `_select` **raises `NotImplementedError`**. It does **not** silently fall
back to random. A placeholder that quietly degrades to a different strategy is
the worst possible failure here: the artifact would record
`method: "embedding"`, the agent and the human would both believe the subset had
visual coverage, and nothing would contradict them. Per the project's
test-integrity rule, a check that cannot run must fail rather than skip — the
same logic applies to a selector that cannot select.

Its `cost_class()` returns `W2` even while unimplemented, so the routing story
is already correct when it lands: embedding a 480-image parent is a scheduled
job, not a planning step.

### `subset_generate` (`W0` or as `cost_class()` reports)

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | — | Workspace subset name |
| `parent` | `str` | — | Parent image directory |
| `selector` | `object` | — | `{class, params}` |
| `dry_run` | `bool` | `false` | Return the selection without writing |

Returns the chosen images, the per-group allocation when applicable, and the
recorded rationale.

### `subset_put` (`W0`) — a human-named subset

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | — | Workspace subset name |
| `parent` | `str` | — | Parent image directory |
| `images` | `array[str]` | — | Parent-relative paths (§10.2), or globs resolved against `parent` |
| `note` | `str?` | `null` | Why these — recorded as the selection rationale |
| `coverage` | `object?` | `null` | Measured trait ranges from triage (§9.3) |
| `overwrite` / `dry_run` | `bool` | `false` | As §3 |

Records `selection.method: "user_named"` with the note. A human-picked subset is
first-class — `user_named` is a selection method, not a lesser one — and this is
the call `phenotypic-assay-triage` step 3 makes when you name the images
yourself.

`subset_get {name}` returns the artifact plus whether staging used symlinks or
copies (§10.3.1).

Because selectors resolve by bare class name like every other extensible class,
adding a fourth is a new subclass plus one `__init__.py` export. No tool
signature changes, no schema bump.

## 10.3.1 How the boundary is *enforced*

"The full dataset is touched exactly once" is a claim about mechanism, and a
claim like that is worth nothing unless something refuses.

The refusal cannot be `deploy_start` alone. `tune_start` and `pipeline_probe`
both take a raw `images` path, and `W2` tune work is explicitly allowed to run
unattended and to route to a full `sbatch` fleet — so an ordinary `tune_start`
pointed at the parent directory would spend full-dataset compute without ever
approaching the promotion gate.

**So subset-scoped tools take a `subset_id`, not a path:**

| Tool | Before | Now |
|---|---|---|
| `pipeline_probe` | `images: str` | `subset_id: str` — raw path allowed **only** while the workspace has no subset at all (bounded `W1`: ≤4 images) |
| `tune_start` | `images: str` | `subset_id: str` |
| `campaign_put` | `dataset.images` path | `subset_id`, recorded on the campaign |
| `deploy_start` | `images` + `scope` | `subset_id` + `scope`; `scope:"full"` resolves to `subset.parent` |

A raw parent path in a subset-scoped phase is refused with
`code: "subset_required"`. This also gives `campaign_status.comparable` (§8.3)
its dataset identity for free: every arm in a campaign shares one `subset_id`,
so arms are comparable by construction rather than by after-the-fact digest
comparison.

### Neither engine accepts a file list — the subset must be materialized

An earlier draft said the server "resolves `subset_id` → the subset's recorded
images list and **passes those to the engine**". **That is not implementable.**
Both call surfaces take a single *path*, not a list:

| Engine | Input | Consumed by |
|---|---|---|
| `python -m phenotypic.tune run` | `-i/--input`, help text literally "image directory" (`tune/__main__.py:49`) | `_load_images(input_dir)` → `Path(input_dir).iterdir()`, **non-recursive directory scan** (`_run.py:235-279`) |
| `python -m phenotypic` | `-i/--input`, `click.Path(dir_okay=True, file_okay=True)`, **no `multiple=True`** (`phenotypicCLI.py:721-730`) | `scan_directory_structure(input_path)` — walks root images, or one level of subdirectories as separate datasets (`_cli_directory_scanner.py:28-117`) |

There is no manifest flag, no repeated `-i`, no file-list parameter on either.
`--sample N` only randomly *thins* datasets already discovered by the scan; it
cannot select named images.

So the server **materializes staging directories** and passes those. **Two
layouts, because the two engines want opposite things:**

```
<workspace>/.phenotypic-mcp/subset-staging/<subset-digest>/
├── flat/                     # for tune — _load_images is a NON-RECURSIVE iterdir
│   ├── plateA_01.tif -> …/data/plates/plateA/plateA_01.tif
│   ├── plateA_07.tif -> …
│   └── plateB_03.tif -> …
└── nested/                   # for deploy — Metadata_Dataset comes from subdir names
    ├── plateA/plateA_01.tif -> …
    ├── plateA/plateA_07.tif -> …
    └── plateB/plateB_03.tif -> …
```

**A single layout cannot serve both, and picking either one alone breaks the
other.** `_load_images` (`tune/_tune_cli/_run.py:259-262`) is
`Path(input_dir).iterdir()` filtered to files — at the root of a *nested*
staging directory it sees only subdirectories, matches zero images, and the run
dies on `SystemExit("no images found under …")` (`tune/__main__.py:202-204`).
Conversely `scan_directory_structure` derives `Metadata_Dataset` from subdirectory
names, so handing *deploy* a flat directory silently relabels every row's dataset
to the staging folder name — the exact corruption nesting exists to prevent.

The split is cheap and safe because the two engines genuinely differ in what they
need: **tune has no dataset concept at all** (`_load_images` returns a flat
filename-sorted list of `GridImage`s; grouping for scoring comes from the
scorer's own CSV, not from directories), while deploy's whole output schema keys
off dataset identity. Both layouts are symlink trees under one digest, so the
marginal cost is inodes, not bytes.

An earlier draft specified only the nested layout and listed `tune_start` among
the tools materializing through it — which would have failed outright on the
spec's own headline example, a `data/plates` parent with `plateA/`/`plateB/`
subdirectories.

**Fidelity is a check, not just a property.** The staging builder verifies that
the layout it produced round-trips: `nested/` must reproduce exactly the dataset
names `scan_directory_structure` would derive from the parent for those images.
Nothing in the engines can catch a mismatch — `scan_directory_structure` only
rejects *internally* inconsistent directories (root images **and** subdirectories
together, `_cli_directory_scanner.py:97-103`); it has no way to know what the
parent looked like. So the check lives in the builder or nowhere.

Four properties it must have:

1. **It mirrors the parent's dataset substructure.** `scan_directory_structure`
   treats one level of subdirectories as separate datasets and rejects mixed
   structures, so a flat staging dir would silently collapse a multi-dataset
   parent into one dataset and change every `Metadata_Dataset` value. The
   subset artifact's `images` entries are therefore **parent-relative paths**
   (`plateA/plateA_01.tif`), not bare filenames — an earlier §10.2 example
   showed bare names, which cannot disambiguate two datasets containing
   `plate_001.tif`.
2. **Symlinks by default, copies on fallback.** Symlinks are cheap and a
   subset may be staged repeatedly across an unattended campaign. But Windows
   symlink creation needs elevated privileges or Developer Mode, and this
   project supports Windows — so the server probes once, falls back to copying,
   and **reports which it used** in `subset_get`. A silent copy of a large
   subset is a surprise worth surfacing.
3. **Keyed by the subset digest**, so re-staging an unchanged subset is a no-op
   and two concurrent arms share one staging directory rather than racing.
4. **It lives under `.phenotypic-mcp/`**, not under `runs/` or the parent — it
   is server scratch, and `--restart`/`--overwrite` semantics must never reach
   the parent images through it.

This staging layer is **new work that §1.6's reuse inventory missed** and §7's
prerequisites did not list. It is tracked as P6.

**The raw-path fallback is bounded to cheap tools.** `pipeline_probe` may take a
path while no subset exists, because it is capped at 4 images and holds the
compute slot — it cannot reach fleet scale. `tune_start`, `campaign_put`, and
`deploy_start` have **no** fallback: they refuse with `subset_required`. An
agent must therefore create a subset before anything unattended or
fleet-scale, which is what makes §10.1's invariant structural rather than
opt-in.

The single exception is `scope: "full"`, which is the *point* of the promotion
gate and is guarded by `promotion_token`.

## 10.4 What runs unattended (resolves OQ-8.1 and OQ-8.2)

**Both permissions are granted, scoped to the subset.**

| Capability | Allowed? | Bound |
|---|---|---|
| Amend a campaign mid-flight (replace a failed arm) | **yes** | Must stay inside the approved budget, compute profile, and scorer. Logged with the reason. |
| Carry a deploy arm | **yes** | **Subset only.** A deploy arm targeting the full dataset is refused. |
| Deploy to the full dataset | **no** | Requires promotion (§10.5) |

So an overnight campaign can lose an arm at trial 31, substitute a replacement
inside the envelope you approved, finish, and even run the winner end-to-end
across the subset — producing real measurements, a real dashboard, and real QC
output for you to look at in the morning. What it cannot do is touch the other
456 plates.

An amendment that would exceed the approved budget, change the scorer, or switch
compute profile is **not** an amendment; it needs a fresh `campaign_approve`.
The envelope is what you actually agreed to, and it is checkable.

## 10.5 The promotion gate

```
promotion_request → [human says yes] → promotion_approve
                                            ↓
                          deploy_plan  {scope:"full"}   ← plan_token for the PARENT
                                            ↓
                          deploy_start {scope:"full"}   ← plan_token + promotion_token
```

**Both** `deploy_plan` and `deploy_start` take `scope`. This matters: a campaign
arm can mint a `plan_token` only for `scope:"subset"` (§10.4), so a full-dataset
run has no other way to obtain one — the plan must be drawn explicitly against
the parent, which is also what produces the sbatch preview and array sizing for
480 images rather than 24.

| `scope` | Requires | Runs against |
|---|---|---|
| `"subset"` (default) | `plan_token` | the subset's image list; reachable from a campaign arm |
| `"full"` | `plan_token` (scope=full) **and** `promotion_token` | `subset.parent` |

`promotion_request` assembles the decision you are actually making, in one
response:

```json
{"ok":true,"data":{
  "pipeline":"pipelines/edge-v3-tuned.json.pht-pipe",
  "provenance":{"from_study":"studies/phase-edge","trial":47,
                "prefab_baseline":{"pipeline":"FilamentousFungiPipeline","best_cost":0.31}},
  "subset":{"name":"plates-dev-24","n_images":24,
            "score":0.081,"gap":{"value":0.06,"verdict":"ok"}},
  "full":{"path":"data/plates","n_images":480,"digest_matches_parent":true},
  "estimate":{"node_hours":18.4,"basis":"subset run: 3.4 s/image measured"},
  "warnings":[
    {"code":"subset_coverage_unverified",
     "message":"Subset spans contrast_michelson 0.031–0.094 across 4 measured images. The
                parent's 480 images were NOT characterized, so whether the subset
                represents them is unknown, not confirmed. Selection was
                'user_named'."}]}}
```

Two properties this must have:

- **The estimate is measured, not guessed.** The subset run already produced real
  per-image timing, so the full-dataset node-hour figure has a basis. This is the
  strongest argument for subset-first development independent of safety: it makes
  the cost of the expensive step *knowable* before you commit.
- **Coverage is reported honestly, including its limits.** A winner tuned on 24
  easy plates may fail on the hard ones, and cost alone cannot reveal that. But
  v1 measures traits only on the subset — §10.3 rules full-parent
  characterization out of scope precisely because it is a substantial compute
  job. So the warning says the subset's range is **unverified against the
  parent**, not that a specific number of parent images fall outside it.
  Claiming the latter would require exactly the dataset-wide probing v1 defers,
  and asserting it anyway would be the more dangerous error: a false assurance
  of representativeness is worse than an admitted unknown.

**Full scope bypasses staging deliberately**, running against `subset.parent`
directly — the `flat/`+`nested/` split (§10.3.1) exists only for subset-scoped
work. The parent's *structure* is not re-scanned at promotion, and does not need
to be: a structural regression (say a stray image dropped beside the `plateA/`
subdirectories, which would trip `scan_directory_structure`'s mixed-structure
rejection at submit time) also changes the parent's file-set digest, so
`digest_matches_parent: false` catches it first and forces a fresh
`promotion_request`. Stated because §10.3.1 makes fidelity an explicit check for
staging, and a reader is entitled to ask why the full path has no equivalent.

`promotion_approve {promotion_id, human_response, note?}` records the decision,
mints the token, and appends a lineage row:

```json
{"ok":true,"data":{"promotion_id":"prom_2c81","status":"approved",
  "promotion_token":"pm_5d17…","expires":"2026-08-14T09:12:00Z"}}
```

`promotion_request`'s response carries `pending_human_ack: true` and an
`ack_prompt` summarizing the ask (winner, subset score, gap, node-hours,
coverage warnings), and `human_response` here is required — same reasoning as
§8.3: it cannot authenticate, but it makes skipping the human an explicit
fabrication rather than a silent default. The token is bound to `(pipeline digest, parent digest, scope)` — if the
full dataset gained images since the request, the token is stale and the review
happens again with `code: "promotion_stale"`.

**The server cannot verify a human approved.** As with campaign approval (§8.2),
`promotion_approve` is a call the agent makes after you say so in chat. It is
provenance so the artifact and the transcript agree, not authentication.

## 10.6 Risks worth stating

- **An unrepresentative subset is the dominant failure mode**, and it is silent:
  every score looks healthy while the pipeline is tuned to a easy slice.
  `coverage` and the promotion warning are the mitigations; neither is a
  guarantee, and `user_named` selection puts the responsibility with you.
- **A subset small enough to be cheap may be too small for the held-out split.**
  The real cliff is sharp and sits at 6: `derive_split`
  (`tune/_evaluation/_split.py:191-199`) returns `kind="none"` with an **empty**
  held-out set when `n_plates < min_heldout_plates` (default 6,
  `_evaluation/_held_out.py:48`) — every plate becomes calibration and the
  generalization gap is not merely noisy but absent. Above 6 there is no second
  discontinuity: Tier 3 sizing is `n_held = max(1, round(0.2 · n))`, growing
  smoothly (1 plate at n=6–9, 2 at 10–12, 3 at 13–17).

  So `subset_put` **errors** below 6 and **warns** below roughly 15, where a
  single held-out plate makes the gap a one-sample estimate.

  **Unless the parent itself is that small.** A 4-plate pilot workspace cannot
  produce a 6-image subset, and since `subset_id` is a hard requirement for
  `tune_start` / `campaign_put` / `deploy_start`, a hard floor would lock such a
  workspace out of tuning and deployment entirely with no stated next action.
  So when `n_images >= parent.n_images` — the subset *is* the parent — the error
  downgrades to `subset_too_small_for_heldout` and the run proceeds with
  `kind: "none"`: every plate calibrates, there is no held-out gap, and
  `tune_status` reports `gap: null` with that reason rather than a number that
  does not exist. Small pilots are a real workflow, not an edge case; what they
  cannot have is a generalization estimate, and saying so is better than
  refusing. An earlier draft
  cited "~12" as following from `min_heldout_plates = 6`; nothing in the split
  logic produces 12, and the real hazard is the hard zero at 6.
- **Subset compute is bounded but not free.** An unattended campaign with deploy
  arms still consumes an allocation. The campaign budget and profile caps (§5.2)
  are what bound it, and they bind on subset runs exactly as on any other.

## 10.6.1 Does promotion re-probe? Only when the headers disagree

The promotion estimate extrapolates subset per-image timing to the parent. That
extrapolation is wrong when the parent holds images the subset does not
represent *dimensionally* — larger frames, a different bit depth, a second
modality mixed in.

Always probing would add a `W1` step, and a `LocalComputeSlot` acquisition, to
every promotion. Never probing would let a silently-wrong estimate through. Both
are avoidable, because **the thing that breaks the extrapolation is readable
without decoding a single pixel.**

So promotion runs a two-tier check:

| Tier | Cost | What it does |
|---|---|---|
| **Always** — header sweep | `W0`, no decode, no slot | Read dimensions, bit depth, and channel count from every parent image header. Compare the distribution against the subset's. |
| **Only on mismatch** — re-probe | `W1`, 2 images | Probe 2 images drawn from `parent \ subset`, chosen from the *mismatching* stratum, and re-derive the estimate from that timing |

Header reads are cheap enough to run over a 480-image parent (TIFF/PNG headers,
not pixel data), and they catch the dominant failure directly: cost scales with
pixel count, so a parent whose images match the subset's dimensions and depth
extrapolates soundly, and one whose images do not is exactly the case worth
spending two probes on.

The promotion response reports which tier ran:

```json
"estimate":{"node_hours":18.4,"basis":"subset run: 3.4 s/image measured",
            "extrapolation_check":"headers match (1024x1536, 16-bit, 3ch across
                                   all 480); no re-probe needed"}
```

and on mismatch:

```json
"estimate":{"node_hours":41.7,
            "basis":"re-probed 2 images from the 4096x4096 stratum at 9.1 s/image",
            "extrapolation_check":"MISMATCH — 113 parent images are 4096x4096
                                   while every subset image is 1024x1536"}
```

Note that the header sweep also gives a *bounded, honest* version of the
coverage gap §10.5 warns about. It cannot tell you whether the parent spans a
biological trait range the subset misses — that would need the full-dataset
characterization §10.3 rules out of v1 — but it **can** state exactly how many
parent images differ dimensionally, which is a real fact rather than an
extrapolated one.

## 10.7 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-10.1 promotion re-probe~~ → **header sweep always, re-probe only on
  mismatch** (§10.6.1). What breaks the extrapolation is readable from headers
  without decoding, so the common case costs nothing and the failing case gets a
  measured estimate.
# PhenoTypic MCP Server — Design Spec

**Status:** draft. All ten sections reviewed by independent reviewers and
revised — six blockers found and fixed across five review passes.
**Date:** 2026-08-12

## What this is

A design for an MCP server that lets an LLM agent build `ImagePipeline`
configurations, tune them with `phenotypic.tune`, and deploy them over datasets
— locally or on SLURM.

The intended UX is **collaborative planning, then delegated execution, on a
subset**: you and the agent characterize the assay and pick a development
subset, decide what is worth trying, write that agreement down as a *campaign*,
and the agent executes it across parallel subagents without you in the loop —
**bounded to the subset**. The full dataset is touched once, after a separate
human promotion. §8 describes the flow, §9 the division of labour, §10 the
subset and the promotion gate; read those three first if you want the shape
before the mechanics.

Two gates, asking different questions: **campaign approval** ("is this a
sensible experiment?") before subset compute, and **promotion** ("is this winner
worth the full dataset?") before the expensive irreversible step.

**Mechanism and judgment are deliberately separated.** §1–§8 specify what the
server does and refuses. §9 specifies what the *agent* should know — how to
triage an organism's traits, why prefab pipelines come before custom ones, how to
read a leaderboard — and ships that as bundled skills. The rule dividing them:
the server makes wrong things impossible; the skills make right things likely.

## Sections

| § | File | Covers |
|---|---|---|
| 1 | [01-architecture.md](01-architecture.md) | Process model, layering, `_services` promotion, work-class routing, `LocalComputeSlot` |
| 2 | [02-state-and-identity.md](02-state-and-identity.md) | Disk-as-authority, path identity and its limits, workspace tree, `RunRegistry` reuse, lineage |
| 3 | [03-tool-catalog.md](03-tool-catalog.md) | Catalog, pipeline, and workspace tools; the probe worker |
| 4 | [04-tune-integration.md](04-tune-integration.md) | Structured knob targets, spec authoring, launch, polling, best-pipeline export |
| 5 | [05-deploy-and-slurm.md](05-deploy-and-slurm.md) | SLURM profiles and caps, plan-then-submit, deploy, status, cancellation |
| 6 | [06-errors-limits-testing.md](06-errors-limits-testing.md) | Error taxonomy, limits, safety boundary, test plan |
| 7 | [07-prerequisites.md](07-prerequisites.md) | P1 JournalStorage backend, P2 promotion, P3 catalog+descriptor, P4 `--screen` guard, rollout |
| 8 | [08-workflow-and-campaigns.md](08-workflow-and-campaigns.md) | The phased UX and the campaign artifact |
| 9 | [09-responsibilities-and-skills.md](09-responsibilities-and-skills.md) | Server-vs-skill boundary, assay triage, prefab-first construction, the four bundled skills |
| 10 | [10-subsets-and-promotion.md](10-subsets-and-promotion.md) | The development subset as the unit of work, the `SubsetSelector` hierarchy, and the promotion gate before full-dataset compute |

## Executable evidence

`docs/superpowers/logic_validation_scripts/2026-08-12-phenotypic-mcp-server/optuna_journal_storage.py`

Two scripts, both re-deriving their claims from the dependency or the data
directly rather than from `phenotypic`.

**`optuna_journal_storage.py`** — the claims behind §7 P1. **Read its `DISCRIMINATION` verdict, not just the ok lines** — on a
local filesystem the negative control also passes, meaning C2a there measures OS
`O_APPEND` atomicity rather than the lock. `--require-discrimination` is the gate
that must pass on the target cluster mount before P1 is implemented.

It also measures throughput headroom (C6), which is what retired the claim that
Postgres remains right for large fleets.

**`contrast_trait_measure.py`** — the choice of contrast measure for §9.3.2.
Establishes that Otsu's η and Cohen's d are scale-invariant (unchanged across a
20× contrast reduction), that Michelson tracks contrast linearly, that η's
per-cell span on real plates is 1.8% of its nominal range, and that whole-frame
Otsu splits plate-from-surround rather than colony-from-agar. It killed a
measure that would otherwise have shipped looking principled.

## Design commitments worth knowing before reading

- **The codebase already anticipates this server** in four places
  (`abc_/_base_operation.py:192`, `sdk_/_docstring_params.py:7`,
  `tune/_search_space/_discovery.py:4`, `tune/_spec.py:293`). Those sites fix
  parts of the contract — notably that the agent **selects a structured tuning
  target, never authors a string key**.
- **One shared stdio server per session.** Subagents inherit the parent's MCP
  connection; they do not get their own process. Hence one `LocalComputeSlot`.
- **Disk is the authority.** The server holds no state whose loss matters.
- **Roughly 80% of the substrate exists**, mostly as a Dash-free tier under
  `gui/`. The server is a thin adapter plus **nine** genuinely new pieces
  (§1.6) — descriptor projection + column derivation, profile governance,
  routing + the compute slot, the `_space.py` pure/view split, a pure sbatch-spec
  extraction, subset staging, the token store, the probe worker, and the killable
  store-open subprocess. All but the first three surfaced only under review; the
  count went 3 → 4 → 5 → 7 → 9.
- **Two hard refusals:** no `--overwrite` (it is `shutil.rmtree`), and no raw
  sbatch passthrough (`parse_slurm_args` constrains neither keys nor values).
- **Development happens on a subset.** The full dataset is touched once, behind
  a promotion gate separate from campaign approval (§10). Subset-scoped tools
  take a `subset_id`, not a path, so the boundary is enforced rather than
  merely asserted.

## Open questions

**None.** Every question raised during design or by the five independent review
passes has been resolved and recorded in the relevant section's "Resolved since
first draft" block.

The last two were closed by measurement rather than by decision:

- **OQ-9.4 (contrast bands)** did not need calibrating — it needed *refuting*.
  The proposed measure, Otsu's η, turns out to be **invariant to contrast**: a
  20× reduction left it numerically unchanged. Replaced with per-cell Michelson,
  which tracks contrast linearly; the categorical band stays human-sourced until
  a dataset spanning low contrast exists. Evidence:
  `logic_validation_scripts/.../contrast_trait_measure.py`.
- **OQ-10.1 (promotion re-probe)** resolved to a header sweep always, re-probe
  only on mismatch — what breaks the timing extrapolation is readable from image
  headers without decoding.

**Resolved:** topology (stdio on the login node) · parallelism (agent-side
fan-out) · state (on-disk workspace, `RunRegistry` reused rather than a new
index) · SLURM authority (named profiles + capped overrides) · coupling
(`_services` promotion) · catalog breadth (reconcile both enumeration lists) ·
workspace root (`--workspace`, defaulting to CWD) · defaulting (explicit always)
· deploy gate (plan-then-submit mandatory) · distributed storage (JournalStorage
backend, gated on L1) · skill packaging (in-repo + `phenotypic-mcp setup`) ·
assay scope (per-dataset) · assay validation (structure and provenance only —
never biology).
