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
| `images` | `str` | — | File or directory tree |
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
             "deliverables":"runs/2026-08-12-plateA/deliverables"}},
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

**`manifest.json` is the designed polling surface**, not the CLI exit code.
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
