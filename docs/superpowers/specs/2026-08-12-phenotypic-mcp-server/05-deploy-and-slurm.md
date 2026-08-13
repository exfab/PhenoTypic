# PhenoTypic MCP Server — §5 Deploy and SLURM Contract

Status: **draft, pending review**
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
  `_cli_constants.py:23`) — it does not block.

So an unconstrained agent can name a partition that does not exist, request
seven days on a shared queue, or bill an account that is not yours. Exposing
`--slurm key=value` verbatim is not an option. The profile layer is the guard.

## 5.2 Server configuration

`~/.phenotypic/mcp.toml`, read-only at startup.

```toml
[workspace]
root = "/scratch/alex/phenotypic-agent"     # OQ-2.2

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

## 5.3 `deploy_plan` (`W0`) — preview, never submit

Deploying to a cluster is the one place an agent can consume a large amount of
somebody else's compute. `deploy_plan` makes the ask inspectable first, and it
performs **no** submission and **no** writes outside a scratch render.

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
| `plan_token` | `str` | — | **Required.** From a matching `deploy_plan`, or from an approved campaign arm |
| `resume` | `bool` | `false` | Continue an interrupted run |
| `retry_failures` | `bool` | `false` | Requires `resume` |
| `restart` | `bool` | `false` | Clear machine state and start over |

**Plan-then-submit is mandatory.** `deploy_start` refuses without a
`plan_token` whose recorded `(pipeline digest, images digest, compute)` matches
the request, returning `code: "plan_required"` or `"plan_stale"`. Every cluster
submission is therefore preceded by an inspectable preview.

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
agent as an opaque non-zero exit. The server pre-validates what
`validate_resume_compatibility` will check — the **`pipeline_sha256` content
digest** (not the path), `input_path`, `image_type`, grid `nrows`/`ncols`,
`bit_depth`, `detect_mode`, `process_only_layer` — and reports the specific
mismatch. It also refuses while ledgered SLURM jobs are still live, which the
CLI likewise blocks.

### GPU staging is automatic, and the plan says so

A pipeline containing a `GpuDetector` (`pipeline_requires_gpu`) triggers the
staged engine — CPU preprocess → resident-model GPU detect → CPU measure — not
per-image processing. On SLURM that becomes an epoch-fenced controller with
Stages 1 & 3 on the CPU profile and Stage 2 as a GPU array.

The agent does not opt into this and cannot opt out. What it must do is name a
**GPU profile** for Stage 2 via `compute.gpu_profile`, which maps to
`--gpu-slurm` and inherits/deltas over the CPU profile. `deploy_plan` reports
`staged_gpu: true` and shows both profiles, so the compute ask is never a
surprise.

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

On SLURM the manifest is refreshed mid-run by an in-array sentinel task. An
external process can force a refresh by invoking the same handler —
`python -m phenotypic._cli._cli_checkpoint_handler --output-dir <dir>
--checkpoint-type manifest` — which is what `refresh: true` does. Without it,
`last_updated` may lag, so the field is always returned and the agent is
expected to read it.

`detail: "results"` adds the deliverables inventory and a **summary** of
`measurements.parquet`: row count, column list, and per-column describe. Never
raw rows — the parquet path is returned for anything more.

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
