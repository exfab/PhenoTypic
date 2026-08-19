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

**Root selection:** `phenotypic-mcp --workspace <path>`. **There is no default —
the root must be given explicitly, and it must contain the image data.**

That second clause is load-bearing and was missing from an earlier draft. Every
worked example in this spec reaches `data/plates`, `data/tune_layout.csv` and
`data/plate_batches.csv` as though they sat inside the workspace, while
`SandboxRoot.resolve()` follows symlinks and then rejects anything landing
outside the root. So either the data is inside the root or the flagship workflow
cannot start; there is no third option that does not introduce a second
containment concept.

The CWD default is dropped with it. A default that silently produces a workspace
without the data is worse than a required argument, and §1.3's "exactly one CWD"
argument only ever established that the default was *unambiguous*, never that it
was *correct*.

`workspace_info` always echoes the resolved root, and the server logs a startup
warning when the root contains `.git` — a source checkout is a plausible launch
directory and a poor place to accumulate run outputs.

```
<workspace>/
├── pipelines/
│   └── <name>.json.pht-pipe          # ImagePipeline.to_json()
├── tune/
│   └── <name>.setup.json.pht-tune    # authored TuningSpec
├── profiles/
│   └── <dataset>.experiment.json          # §9.3 experiment profile
├── subsets/
│   └── <name>.subset.json            # §10.2 development subset
├── campaigns/
│   └── <name>/campaign.json          # §8.2 the agreed plan
├── .phenotypic-mcp/
│   ├── lineage.jsonl                 # §2.5
│   ├── plans/<token>.json            # §5.4 plan token records
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

Only `pipelines/`, `tune/`, `profiles/`, `subsets/`, `campaigns/`, `studies/`,
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

**One field is added: the record stores `(pid, create_time)`, and process
identity is the pair.** `RunRecord` carries a bare `pid` today, which is not an
identity — pids are reused, and a reused pid is indistinguishable from a live run
under any check that looks at the number alone. `create_time` comes from
`psutil.Process.create_time()` (or `/proc/<pid>/stat` field 22) at spawn, and any
liveness decision — restart reconciliation (§1.5), `workspace_cancel`,
`workspace_list`'s status column — compares both. A pid whose `create_time`
differs is **dead**, full stop. This is what stops the server from either wedging
the local path behind a run that finished hours ago or signalling somebody else's
process.

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
{"ts":"2026-08-12T15:43:55Z","event":"deploy.approve","id":"runs/2026-08-12-plateA","parent":"pipelines/edge-v3-tuned.json.pht-pipe","scope":"full","token":"pl_7f3a…","group_filter":{"Metadata_Species":"A_nidulans"},"node_hours":18.4,"ack_source":"elicited","human_response":"yes, go ahead"}
{"ts":"2026-08-12T15:44:19Z","event":"deploy.start","id":"runs/2026-08-12-plateA","parent":"pipelines/edge-v3-tuned.json.pht-pipe","scope":"full","subset_id":"subsets/plates-dev-24.subset.json","group_filter":{"Metadata_Species":"A_nidulans"}}
```

**Both deploy rows carry `group_filter`** (`null` at `scope:"subset"`, and
`null` at full for an unfiltered subset). Without it a group-scoped full deploy
is not reconstructible from the journal: `scope:"full"` plus a `subset_id` reads
as "the whole parent", which is precisely what USER-21 ruled it is not. The
approve row is where the human's consent is recorded, so it is where the image
set that consent covers has to be legible — the token that bound it (§5.4) is
single-use and its record is collectable.

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
| Run record mutation | generation-fenced CAS, **never onto a terminal status** | `RunRegistry.compare_and_set` |
| Local image compute | one process-wide `LocalComputeSlot`: `asyncio.Semaphore(local_slot_capacity)` (default 1, configuration — USER-17), released in a `finally` at the innermost acquiring layer, under a wall-clock lease | new, §1.5 |
| Artifact writes | `atomic_write_text` + explicit-overwrite policy | `sdk_` helpers, §2.2 |
| Artifact **read-modify-write** | `exclusive_path_lock` on the artifact path + content-digest CAS; mismatch is `artifact_changed`, never `ok:true` | new, below |
| Plan tokens | `atomic_write_text`; single-use CAS on `consumed_by` **under `exclusive_path_lock`** — `allocate`'s interprocess idiom, **not** `compare_and_set`'s in-process `threading.Lock`, since §2.3 treats overlapping server instances over one workspace as anticipated | new, §5.4 |
| Subset staging dirs | Keyed by subset digest; idempotent **by completion** — temp dir → `os.replace` → `.complete` marker written last, and readers require the marker | new, §10.3.1 |
| `campaign.json` mutation | `atomic_write_text` + a transition guard CASing on **`(status, artifact_digest)`**, never `status` alone; `campaign_start` snapshots the campaign it launched rather than re-reading mid-fan-out | new, §8.3 |
| Lineage journal | `atomic_append` under file lock, **via `asyncio.to_thread`** | existing pattern, §2.5 |
| Operation registry | immutable after discovery, and the module-level `_REGISTRY` is **published only after `discover()` returns**, under a module-level lock | `get_registry()`, amended below |

No tool mutates another tool's in-flight artifact. The one cross-tool write is
`tune_export_best`, which writes a *new* pipeline file rather than editing the
base — matching `build_pipeline`, which deep-copies rather than mutating
(`tune/_evaluation/_builder.py:384`).

**`deploy.approve` exists because the fold deleted the only writer of it.** The
retired `promotion_approve` recorded the decision *and appended a lineage row*;
collapsing it into `deploy_start` kept the decision and dropped the row, which
would have left **the most consequential human decision in the system with no
durable record at all** — only a token file that gets consumed and a
`consumed_by` field that says a run happened, not that anyone agreed to it. It is
written before the submission it authorizes, so an approval followed by a crash
is still reconstructible.

### The CAS key is `(status, artifact_digest)`, never `status` alone

Status-only compare-and-set is the guard three separate defects slip past, and
they slip past it for one reason: **an amendment can change what the artifact
says while leaving `status` exactly where it was.** A §10.4 in-envelope amendment
edits the arm set of an `approved` campaign and it is `approved` before and
after, so a concurrent writer holding a stale snapshot passes the CAS and writes
the pre-amendment arm set back. Nothing fails, and the campaign silently reverts.

So every mutation of an artifact CASes on the pair: the expected `status` **and**
the content digest of the bytes the caller read. A digest mismatch is
`artifact_changed` (§6.2), and the caller re-reads rather than overwriting.

Two things fall out of the same fix, which is why it is one fix and not three:

- **Double launch needs no separate guard.** `campaign_start` moves
  `approved → launching → running`, and a concurrent second call fails the
  transition itself — `launching` is not `approved`. The state machine does the
  work that a bespoke "already started?" check would have done less reliably.
- **The elicitation window closes.** `campaign_approve` prompts a human, and a
  human takes minutes; the window between reading the campaign and writing
  `approved` is not milliseconds, it is however long somebody takes to read the
  summary. So the digest is captured **when the elicitation prompt is built**,
  and re-CAS'd after the answer comes back. If the artifact moved underneath, the
  call fails with `campaign_changed_during_approval` and re-prompts against the
  new content. Approving a summary that went stale while it was on screen is
  exactly the failure elicitation was adopted to prevent, and without the
  re-CAS elicitation *creates* it.

### Post-start CAS never resurrects a terminal record

Ordering, stated because the mechanism is already here and only the sequence was
missing: **`allocate` → record `launching` → spawn → CAS `launching → running`,
and that last CAS applies only if the record is still `launching`.**

A local subprocess can die before the launcher's next statement runs — a missing
module, a bad argument, an immediate OOM — and the exit observer will have
already CAS'd the record to `failed`. An unconditional write-back to `running`
then resurrects a dead run, and because `allocate` refuses a nonterminal
generation on that output directory, the directory is blocked permanently by a
process that does not exist. `compare_and_set`'s generation fence already
provides the mechanism; the rule is simply that `running` is never written over
a terminal status.

### Two locks, one order

**The in-process `threading.Lock` is never nested inside the interprocess file
lock.** `exclusive_path_lock` spins to 30 s by design (§2.5), and a thread
holding `RunRegistry`'s `threading.Lock` across that spin stalls every other
handler in the process for the full timeout — the event-loop offload of §1.5
does not help, because the contention is on the lock, not on the loop. This is
not hypothetical: `allocate` (`_services/runs.py:317-337`) opens `with
self._lock:` and takes `exclusive_path_lock` *inside* it today, which was
correct for a GUI where one human clicks Run and is not correct for a server
serving N subagents.

The order is: **take the file lock first; take the `threading.Lock` only around
the in-memory mutation, and release it before the file lock.** The in-memory
critical section is a dict update measured in microseconds; the file lock is
measured in seconds. Nesting them the other way round makes the cheap one wait
on the expensive one.

### The registry is published after it is populated

`get_registry()` assigns the module-level `_REGISTRY` **before** `discover()`
runs. A second thread arriving mid-discovery therefore gets a real object that is
silently incomplete — not an error, not an empty registry, just a catalog missing
whichever operations had not been imported yet. Discovery runs under a
module-level lock and `_REGISTRY` is assigned only after `discover()` returns,
so a loser thread waits and then sees the finished object.

## 2.7 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-2.1 GUI preset interop~~ → **visible `tune/` only**. Agent-authored specs
  keep readable names at `<workspace>/tune/<name>.setup.json.pht-tune`; the GUI
  reaches them through its Browse… escape hatch. Writing additionally into
  `.phenotypic-gui/presets/tune/` would double every artifact, and writing only
  there would hand the agent content-addressed filenames (`<stem>-<sha256[:20]>`),
  losing the readable-id property §2.2 is built on.

- ~~OQ-2.2 workspace root~~ → `--workspace`, **mandatory, no default**, and it
  must contain the image data — a silent CWD default risked producing a
  workspace without the data (§2.3). The root is always echoed by
  `workspace_info` and a warning is logged when it is a git checkout (§2.3).
