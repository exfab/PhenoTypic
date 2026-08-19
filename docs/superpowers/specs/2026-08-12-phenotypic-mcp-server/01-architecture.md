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

**The transport is PyPI `fastmcp` 3.x**, added as an optional extra so the core
package gains no dependency. Deliberately **not** the FastMCP 1.0 frozen inside
the official `mcp` SDK: current guidance warns against it, and the capability
this design most depends on — elicitation for the human gates (§8.2, §10.5) — is
exactly what a frozen 1.0 is likely to lack. The tool layer stays
transport-agnostic, so this is a dependency choice, not an architectural one.

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
| `W0` introspect | Pure computation over metadata; no image I/O | list operations, describe schema, validate a spec, infer a search space, summarize a pipeline |
| `W1` probe | Bounded image compute, interactive latency — **or metadata-only image I/O that may escalate into it** | apply a pipeline to 1–N images and return measurements + benchmark; `deploy_plan {scope:"full"}`, which sweeps the parent's image **headers** and re-probes 2 images on a mismatch (§5.3) |
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

There is **one** `LocalComputeSlot`: a process-wide semaphore whose capacity is
`local_slot_capacity`, configuration defaulting to 1 (below, and §1.6.1).

| Work | Acquires the slot? | Held for |
|---|---|---|
| `W0` any environment | no | — |
| `W1` probe | **yes** | the duration of the in-process compute |
| `W2`/`W3` routed `local` | **yes** | the entire subprocess lifetime, released on reap |
| `W2`/`W3` routed `slurm` | no | — (the scheduler is the arbiter) |

**"While it computes" is not a hedge, and `deploy_plan {scope:"full"}` is why it
is written that way.** A `pipeline_probe` acquires the slot for its whole
handler, because its whole handler is image compute. `deploy_plan {full}` is
`W1` for the escalation it *may* perform, not for the work it always does: its
tier-1 path reads image headers — measured at 0.18 ms/image, 0.081 s for 460
images (§5.3) — and takes no slot, and it acquires the slot only when the headers
disagree and it re-probes 2 images. Read as unconditional, the row would make
every full-scope preview block every subagent's probing behind a running local
arm, and fail `local_slot_timeout` on the common path where the headers match.
The class is `W1` because the escalation exists; the acquisition is where the
compute is.

`LocalRunner` offers nothing to reuse here — it is a multi-handle subprocess
tracker with no exclusivity guard, because the GUI never needed one (a human
clicks Run once). The slot is new code, listed as such in §1.6.

**Second invariant, stated rather than derived: at most one `W1` probe is in
flight process-wide.** This is true today only as a *consequence* of the slot —
a probe holds it, so a second probe cannot start. A safety that exists only as a
consequence of another rule is one refactor away from vanishing with nothing
failing, and §3.2's single warm probe worker is written assuming it. So it is an
invariant in its own right, and it is what the worker's liveness check answers
to: before dispatching, the server verifies the warm worker is alive and
respawns if not, rather than assuming the previous holder left it usable.

**The invariant binds probe *execution*, not the `W1` label.** What may not
overlap is a pipeline being applied to images through §3.2's warm probe worker.
`deploy_plan {full}`'s tier-1 header sweep is `W1`-classed but is not a probe
and does not go through the worker — it is header I/O on the `blocking`
executor. Its escalation re-probe *is* a probe, dispatches through the same warm
worker, and is bound by the same invariant. Stating this stops the reasonable
reading in which every `W1` handler must queue behind the worker.

**Routing table**

| Class | `local` env | `slurm` env |
|---|---|---|
| `W0` | in-process, no slot | in-process, no slot |
| `W1` | in-process, **holds slot while it computes** | in-process, **holds slot while it computes** |
| `W2` | subprocess, **holds slot** | `sbatch` fleet, no slot |
| `W3` | subprocess, **holds slot** | `sbatch` array, no slot |

### A locally-routed batch job suspends interactive probing

Stated plainly because it is a real limitation, not an oversight. A `W2`/`W3`
routed **local** holds the slot for its entire subprocess lifetime, and `W1`'s
budget (`probe_timeout_s`, default 300 s) **includes slot wait** — so while a
local study or deploy runs, every `pipeline_probe` from every subagent fails with
`local_slot_timeout` rather than queueing. §8.7's exploration loop is unavailable
for that duration.

On a cluster this is nearly unreachable: `W2`/`W3` route to `sbatch` and take no
slot. It bites on a workstation with no scheduler, and the honest answer there is
that local batch work and interactive probing do not overlap.

**Which is why a local `W2`/`W3` child is detached, not reaped at session end.**
`LocalRunner` installs an `atexit` hook that SIGTERMs its children, and under it
the bargain above is not payable: probing is suspended for hours by design, and
then the run those hours were spent on dies with the session. So local batch
children start with `start_new_session=True` and are **not** registered with that
hook; restart reconciliation adopts them (§1.5, below). The suspension buys a
result rather than buying nothing. The alternative —
admitting `W1` alongside a reduced-worker local job — was considered and rejected
as machinery serving a deployment this design does not target.

**Locally, the slot *is* the cap — there is not a second mechanism.** Any
separately-stated arm count beside it would be a second owner for one invariant,
describing a parallelism the slot forbids.

So `LocalComputeSlot` owns the local-OOM invariant alone, and **its capacity is
configuration** (`local_slot_capacity`, default `1`). A workstation with memory
to spare may set `2`; nothing else changes, because every other rule here is
written against "the slot", not against a number. `budget.max_concurrent_arms`
is a *campaign* budget and never a memory guard.

**`executors.compute.max_workers` is not an independent number — it *is*
`local_slot_capacity`.** Fixing the pool at 1 while the slot is configurable
makes the two disagree at any capacity above the default, which is precisely the
disagreement the split was introduced to make impossible: the invariant below
would be false by configuration, and a probe holding a slot but starved on a
one-worker pool would burn `probe_timeout_s` and then report `local_slot_timeout`
— "waited past `probe_timeout_s` for the slot" — when it had the slot all along.
One number, stated once, and the pool and the slot cannot drift apart.

**Whenever more than one CPU-heavy holder is admitted, probe responses carry
`contended: true`, and such timings are not eligible as an estimate basis.** Two
paths admit one: capacity above 1, and the orphan rule, which admits a `W1`
beside a live orphan that is typically `LocalParallelStrategy` with `n_jobs=-1`
— every core, the exact thrash this section opens by citing. The trade is
defensible; staying silent about it is not, because probe timing is what §10.5
calls "measured, not guessed" in the estimate a human approves, and what §8.7's
keep/revert decisions are read from. A contended measurement that presents itself
as a clean one corrupts both.

**The server-wide arm ceiling is a queue, not a check.** USER-24 made per-group
work N independent campaigns, so there are now N background launchers contending
for one `limits.max_inflight_arms`. "Each launcher checks before launching" is
not an admission protocol: nothing orders the claimants and nothing owns the
counter, so one launcher can win repeatedly while another never does — and §9.3's
own two-column example yields six campaigns against eight slots, making that the
expected configuration rather than a corner.

So the ceiling is **one `asyncio.Semaphore(max_inflight_arms)` acquired by every
launcher**, which gives arrival-order admission for free. And because a starved
arm otherwise looks identical to a healthy one — it sits in `queued` while
`launch_state` reports `clean`, since a background task genuinely is alive —
`campaign_status` reports **`queued_reason: "campaign_budget" | "server_ceiling"
| "local_slot"`** and, for the ceiling, a queue position. The third value matters:
under local routing a `W2` arm holds the slot for its whole subprocess lifetime,
so at `local_slot_capacity=1` a three-arm campaign starts one arm and the other
two are waiting on the *slot* — which is neither of the first two reasons, and
without a name for it those arms sit in a state the design does not describe.

**The launcher's wake condition is the semaphore itself.** Acquiring
`asyncio.Semaphore(max_inflight_arms)` parks the launcher until a release wakes
it in arrival order — so "what wakes a blocked launcher" needs no poll interval
and no separate signal; the same primitive that bounds admission provides it.
On `workspace_cancel` the launcher is cancelled and stops launching further arms,
leaving already-launched arms to their own lifecycle; on server shutdown it exits
without launching more, and its lease expiry is what lets the next server see the
fan-out as incomplete rather than merely idle. Without those two, the ceiling introduces
a starvation mode the per-campaign budget never had, and hides it behind a
healthy-looking poll.

**A second local arm does not make its *caller* wait.** The handler returns
immediately with a `run_id`, a `queued` status and a `queue_position`; the *run* then waits for the
slot in the background launcher (below). What is forbidden is the handler parking
on the semaphore — a blocking acquire would stall the call for the hours an arm
actually takes, against a host timeout the server does not control, and an
abandoned coroutine would hold a reservation nothing will ever release.

So local batch work is **queued, not refused** — refusing would make the server
useless on a workstation, which is the case §1.5 exists to serve. `W1` probes are
the exception and genuinely fail rather than queue: `probe_timeout_s` includes
slot wait by design, so a probe that cannot start promptly is not worth starting
at all (`local_slot_timeout`, §6.2).

### Blocking work never blocks the event loop

**Every tool handler is `async def`, and everything that blocks is offloaded to a
worker thread.** Not only CPU-bound compute — the rule covers subprocess waits,
SLURM polling, and the lineage reads whose lock can spin for 30 s (§2.5). This is
the model `fastmcp` expects, and it is the only one under which §1.6.1's table is
satisfiable at all: a `W0` call promised to return in under a second cannot share
a thread with a synchronous `sbatch` poll.

The rule is stated once, here, and the rest of the spec relies on it rather than
re-deriving it per tool. Anywhere a handler touches the filesystem under a lock,
a subprocess, or the scheduler, assume the offload.

**Two executors, not one.** The offload target is not a single default pool. The
server owns two named module-level `ThreadPoolExecutor`s:

| Executor | Workers | Carries |
|---|---|---|
| `blocking` | 4 | filesystem reads and writes, the lineage journal's 30 s lock spin, subprocess waits, scheduler polling |
| `compute` | 1 | `W1` pipeline execution |

The split is not tidiness. With one shared pool, a burst of `campaign_status`
store-opens — N arms × N subagents, each a blocking call — can occupy every
worker and starve the probe **the compute slot has already admitted**: the slot
guarantees the probe exclusivity and the pool takes it away, so the probe waits
on a resource no rule in this section governs. Sizing `compute` at exactly one
worker makes the pool a *second expression of the same one-probe invariant*
rather than an independent scheduler, so the pool and the slot cannot disagree
about how many probes are running. Both numbers are stated bounds (§1.6.1).

`ImagePipeline.apply()` is synchronous, CPU-bound, and copies the image
(`_image_pipeline_core.py:943-966`). Running it directly in an async handler
would block the entire event loop — stalling `W0` calls from *other* subagents
and silently falsifying "agent-side fan-out is free".

**`W1` does not run in the server process at all** — not even on a worker thread
via `run_in_executor`. A runaway op in a thread cannot be killed, so the caller
returns while the thread keeps running and every later probe from every subagent
deadlocks behind something Python cannot interrupt. The probe therefore executes
in **a persistent, killable worker subprocess** (§3.2), and the server-side cost
is a bounded pipe wait — not a computation.

So `executors.compute` is **the probe-dispatch slot, not a compute pool**: one
in-flight probe *request*, whichever process runs it. That is what makes the
one-probe invariant hold across the slot and the pool for a single reason rather
than two coincidentally-agreeing ones, and it is why the wait does not go on
`blocking` despite being a subprocess wait — putting it there would make
`blocking`'s four workers the real gate on probe concurrency and re-create the
starvation the split exists to prevent.

Peak memory residency is likewise the subprocess's, not the server's: §3.2 makes
the equivalent claim correctly via RSS reset on respawn, which a thread could not
offer.

`W1` carries a second, independent guard: a hard cap on scope (default 4 images)
and a wall-clock timeout. A mis-set slot still cannot melt the node.

`W2`/`W3` in a `local` environment are *not* refused — that would make the
server useless on a workstation — but they serialize on the same slot, and the
tool result says so, so the agent learns the run is queued rather than
mysteriously slow.

### The slot primitive, and release symmetry

`LocalComputeSlot` is an **`asyncio.Semaphore(local_slot_capacity)`** (§1.5
above; default 1) owned by the event loop. Thread-side and
process-side code — the `compute` executor's worker, the probe worker's exit
observer, `LocalRunner`'s reap thread — **never touch it**. They signal through
`loop.call_soon_threadsafe`, so every acquire and release happens on the loop
thread and the semaphore needs no cross-thread synchronization of its own.

**Release lives in a `finally` at the innermost layer that acquired it.** There
are four exit paths — normal return, exception, cancellation, and subprocess
reap — and the defect this rule prevents is a design where each path releases
separately and one of them forgets. One acquiring layer, one `finally`, one
release statement that all four paths run through.

**The exit observer is release-first, record-second, and the record step is in
its own `try`/`except`.** Ordering here is load-bearing: recording the run's
terminal status takes `exclusive_path_lock`, which spins up to 30 s and can
raise `artifact_lock_timeout`. If the release sits after the record, that
timeout skips it and the slot is stranded for the life of the server — silently,
because the reap thread swallows callback exceptions. Releasing first costs
nothing and removes the whole class.

**And the slot is acquired with a wall-clock lease, unconditionally.** The lease
is `probe_timeout_s` for `W1` and, for a local `W2`/`W3`, the run's `--time` or
a configured maximum. On expiry the slot auto-releases and the holder's record
is marked `slot_lease_expired` (§6.2), which is a visible terminal state rather
than a stuck one. The lease is unconditional because a design whose only release
is a callback the runtime may never invoke has no recovery at all, and the
failure it produces — every later probe blocked, no error anywhere — is the least
diagnosable one this section can create.

### Restart reconciliation

The server may be killed without running `LocalRunner`'s `atexit` cleanup
(`run_console/_runner.py:141-145`), orphaning a local `W2`/`W3` subprocess. A
fresh server that simply reset its slot would admit a second local job beside
the orphan, doubling contention.

So on startup, **before serving any request**, the server runs
`rehydrate_from_sandbox` and reconciles: for every nonterminal `RunRecord`, it
decides whether that run is still alive. A dead one is CAS'd to `failed` with
`status_detail` naming the lost server. Only then does the server accept work.

**Liveness is decided on `(pid, create_time)`, and identity is the pair.** The
run record stores both — `create_time` from `psutil.Process.create_time()`, or
field 22 of `/proc/<pid>/stat` where `psutil` is unavailable — captured at spawn.
At reconciliation a pid that resolves to a process whose `create_time` differs
from the recorded one is **dead**: the pid was reused. A bare pid check gets this
wrong in both directions, and both are bad — a reused pid makes a finished run
look live and wedges the local path forever, and there is no version of the
mistake in which the server is merely conservative.

**A live orphan is refused, not watched.** Letting it *claim the slot* is a
reservation with no releaser: the `Popen` belonged to the dead server, so no exit
observer exists and the slot is held until this server dies too. Watching is not
available either, because Linux offers a non-parent no exit notification short
of polling, and polling reintroduces exactly the daemon-thread callback the
release-symmetry rule above just constrained.

So the server **records** the orphan and refuses to admit local `W2`/`W3` beside
it, with `local_slot_orphaned` (§6.2) naming the pid, the run id, and
`workspace_cancel` as the way out. `W1` is still admitted: a probe is capped at
`probe_max_images` and holds a wall-clock lease, so it is bounded contention
rather than the unbounded thrash the slot exists to prevent, and suspending the
agent's entire exploration loop on a stale record would be a worse trade.
Refusing is less machinery than watching and it makes the stuck state **visible
at the moment it bites**, with a named remedy, instead of leaving a server that
is inexplicably unable to start anything.

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
| Plan token records | **new** | §5.4 — opaque ids over persisted records, not forgeable digests. Since the promotion fold it also carries the material the `deploy_start` gate is rendered from (`ack_prompt`, `decision_content`) — but **not** the ack itself, which is taken at `deploy_start` (USER-18) |

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
| **`W0`** | Returns in **under one second**, and **must never block the event loop**. A `W0` tool that performs real I/O — a filesystem walk, a subprocess, a store open, a directory digest — runs in the executor. `W0` means *takes no compute slot*; it does **not** mean *is instant*, and the two must not be conflated. **One exemption: while a human-gate elicitation is outstanding.** That wait is an `await` on the loop, not a block of it — other subagents' calls are served throughout — and it is bounded by §8.2's single-flight rule. Both human gates (`campaign_approve`, `deploy_start`) rely on this; without it the `W0` row conflates latency with blocking and forbids a wait that costs the server nothing. |
| **`W1`** | **Probe execution** is bounded by `limits.probe_max_images` (default 4) and `limits.probe_timeout_s` (default 300), inclusive of slot wait. The timeout must be reconciled with the host's own tool-call timeout — a server that outlives the host's patience holds the slot after the caller has given up. **`deploy_plan {scope:"full"}` is the one `W1` whose bound is not the image cap**: its tier-1 header sweep is bounded by the parent's size, measured at 0.18 ms/image and flat to ~5,700 images (§5.3), and only its escalation — 2 images — is subject to the cap. Applying the 4-image cap to it would refuse the tool at any real dataset size. |
| **`W2` / `W3`** | **No latency requirement.** Submit-and-poll: the tool returns on submission and progress is polled. |
| **Connection** | The `tools/list` payload is spent every turn by every subagent; it is a budgeted resource, not free. |

The binding consequence: under §1.3's single shared connection, any handler that
blocks stalls **every** subagent, not just its caller. §5.5 already carves
`deploy_status {detail:"results"}` into the executor for exactly this reason;
that carve-out is the rule, not an exception.

### Stated bounds

Every number the concurrency design depends on, in one table, so none of them
lives only inside a paragraph:

| Bound | Value | Owner |
|---|---|---|
| `executors.blocking` workers | 4 | §1.5 — filesystem, journal, subprocess waits, scheduler polling |
| `executors.compute` workers | **= `local_slot_capacity`** (default 1) | §1.5 — the **probe-dispatch slot**, not a compute pool. Defined as the slot's capacity so the two cannot disagree (§1.5) |
| `local_slot_capacity` | 1 | §1.5 — the local-OOM invariant; configuration, not a promise |
| `limits.max_inflight_arms` | 8 | **server-wide**, across *all* campaigns — and the only bound on in-flight work, since an over-cap `sbatch` does not error but queues indefinitely on `Reason=AssocGrpCpuLimit`, so submission success is not backpressure |
| `limits.max_inflight_local_runs` | = `local_slot_capacity` | **server-wide** — not a second knob; the slot already is this bound |

**The arm ceiling is server-wide, and that is the whole point.**
`budget.max_concurrent_arms` (§8.2) is a *per-campaign* budget, so N campaigns ×
M arms has no aggregate ceiling anywhere — and §9.3's own worked example, one
campaign per metadata group, produces six campaigns from a two-column
cross-product without anybody deciding to fan out. The ceiling is checked by the
background launcher (§8.3) before each arm is launched, not by the tool handler,
because the handler returns before most arms exist.

`8` is a **policy default, not a derived number** — no measurement in this spec
implies it. It is set where it is because §4.4's per-arm store-open subprocess
is the cost that grows with in-flight arms, and because a ceiling an operator
raises deliberately is safer than one discovered by exhausting an account cap.
It is configuration; the invariant is that a ceiling exists and the launcher
checks it.

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
