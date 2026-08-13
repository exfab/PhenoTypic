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

Roughly: the server is a **thin adapter plus five genuinely new pieces**
(descriptor projection + column derivation, profile governance, routing + slot,
the `_space.py` split, and the pure sbatch-spec extraction). The promotion itself
is mechanical; the two extractions are not.

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
