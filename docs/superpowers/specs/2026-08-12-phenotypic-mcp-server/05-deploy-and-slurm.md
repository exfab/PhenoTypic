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
# No `account` key. `--account=exfab` belongs only with the `exfab` partition;
# elsewhere the default account is correct. Setting it here does not fail at
# submit — the job QUEUES on `AssocGrpCpuLimit`, indefinitely and silently,
# which is the failure mode an agent cannot diagnose and this file exists to
# prevent.
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
partition        = "exfab"     # NOT the public `gpu` partition: it exists, but
account          = "exfab"     # its queue is long. `exfab` requires this account.
gpus_per_node    = 1
time             = "02:00:00"
overridable      = ["time"]
  [slurm.profiles.gpu-short.caps]
  max_time  = "02:00:00"
  max_array = 16
```

Note what the two corrections above have in common: **both are site facts an
agent cannot derive and cannot detect getting wrong.** A wrong account queues
rather than errors; a valid-but-congested partition runs correctly and simply
never starts. Neither produces a diagnosable failure, which is the argument for
naming them once in a file rather than deciding them per call.

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

## 5.2.2 The project layer — `<root>/phenotypic-mcp.toml`

`~/.phenotypic/mcp.toml` is a **per-user** file, and one user works across
several projects whose compute needs genuinely differ — a filamentous-fungi run
at ~30 min/image and a yeast plate sweep do not want the same walltime, and
nothing about the user is the reason. So a second layer sits in the workspace:

```
<workspace-root>/phenotypic-mcp.toml
```

**It is visible, at the root, deliberately.** Not under `.phenotypic-mcp/`, which
is machine state the server owns and rewrites. This file is **human-authored
input** — the same distinction that puts `pyproject.toml` at the root while
`.venv/` hides. A file that constrains what an agent may submit on your behalf is
one you must be able to find without being told it exists, and read without
being told the syntax. Discovery is unambiguous because USER-11 already makes the
workspace root mandatory: the server reads exactly `<root>/phenotypic-mcp.toml`
and does not walk parent directories.

```toml
# Everything is optional. A project with no compute opinions omits the file.
default_profile = "fungi-long"

[slurm.profiles.fungi-long]           # a profile only this project needs
partition     = "epyc"
cpus_per_task = 16
time          = "24:00:00"
overridable   = ["time", "n_workers"]
  [slurm.profiles.fungi-long.caps]
  max_time      = "48:00:00"
  max_n_workers = 16

[slurm.profiles.cpu-bulk.caps]        # NARROW an inherited profile
max_cpus_per_task = 8                 # site allows 32; this project says 8
```

### The rule that makes a project layer safe: narrow, never widen

| Layer | Owned by | May |
|---|---|---|
| `~/.phenotypic/mcp.toml` | the user/operator | set the **ceiling** |
| `<root>/phenotypic-mcp.toml` | the project | choose a default profile, add profiles, **lower** caps |
| the tool call | the agent | override `overridable` keys, within the **effective** caps |

**A project file may only make the effective cap smaller.** Raising
`max_cpus_per_task` above the site value is a **startup error naming both
values** — not a silent clamp, because a clamp teaches the author their file
works. A project-defined profile is capped by the site's caps for any key the
site constrains globally.

This is what lets the file adapt per project without becoming a way to escape the
operator's limits, and it is why the layering runs in this direction rather than
last-write-wins.

### Effective config is inspectable, not inferred

`workspace_info` reports the **effective** profile set with each value's
originating layer (`site` / `project` / `default`). A config system whose result
can only be discovered by submitting a job is one nobody will trust, and an agent
that cannot see the effective caps will guess at them — which is the behaviour
the whole mechanism exists to remove.

### Failure is loud

| Condition | Behaviour |
|---|---|
| File absent | Fine. Site config applies unchanged |
| Malformed TOML | **Startup error** naming file and line. Never "ignore and continue" — a config that silently does nothing is worse than none |
| Widens a cap | **Startup error** naming key, project value, site ceiling |
| Names an unknown profile as `default_profile` | Startup error listing the profiles that do exist |
| Sets a reserved SBATCH key | Rejected exactly as §5.2 rule 4 rejects it from an agent |

Startup, not first-use: a config error must surface when the server starts,
while a human is present, not three hours into an overnight campaign.

## 5.3 `deploy_plan` (`W0` at `subset`, `W1` at `full`) — preview, never submit

Deploying to a cluster is the one place an agent can consume a large amount of
somebody else's compute. `deploy_plan` makes the ask inspectable first, and it
performs **no** submission and **no** writes under the run's output directory.

**Its work class depends on `scope`.** At `scope:"subset"` it reads registered
state and renders a spec — genuinely `W0`. At `scope:"full"` it additionally runs
§10.6.1's parent header sweep, and on a mismatch escalates to a 2-image
re-probe.

**The sweep is not the reason.** Measured, it is comfortably inside `W0`
(`logic_validation_scripts/2026-08-12-phenotypic-mcp-server/header_sweep_cost.py`):
**460 real images on GPFS: 0.081 s, 0.18 ms/image** — twelve times under the
one-second ceiling, and flat from 480 images out to roughly 5,700. Tier 1 is
comfortably `W0` at the scale §10.6.1's worked example describes.

**The re-probe is the reason.** It takes the compute slot outright, and per §1.5
it can then wait out `probe_timeout_s` behind a running local arm — unbounded in
the way that actually matters. So `scope:"full"` is declared `W1`: bounded by the
§1.6.1 probe caps, offloaded to a worker thread per §1.5, and honest about the
slot. The response reports which path ran, and `estimate.basis` already says
whether a re-probe happened.

Two consequences of the measurement worth stating. First, **the sweep's class is
dataset-size-dependent**: at ~0.18 ms/image it crosses one second near 5,700
images and reaches ~9 s at 50,000, so a sufficiently large parent makes tier 1
alone non-`W0`. The `W1` declaration already covers that, which is the useful
outcome — the class is right for a reason the design did not originally have.
Second, the measured cold and warm figures were **identical**, which means
`posix_fadvise(DONTNEED)` does not evict from GPFS's own pagepool; the "cold"
number is therefore an upper bound on optimism, not a true cold read. It does not
change the conclusion at this scale — a 10× cold penalty still lands under the
ceiling at 480 images — but a first-touch sweep on a genuinely cold filesystem
has not been measured, and should not be claimed.

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
| `scope` | `"subset" \| "full"` | `"subset"` | `full` plans against `subset.parent`, **intersected with the subset's `group_filter` where it has one** (§10.2, §10.5); the only way to obtain a full-scope `plan_token` |
| `run_name` | `str` | — | Output directory under `runs/` |
| `compute` | `object?` | `{}` | `{profile, …overridable keys}` |
| `metadata_csv` | `str?` | `null` | Joined onto the measurements mirror |

**`mode`, `layer` and `sample` are deliberately absent.** They existed because
the CLI has them, not because any workflow section reaches for them: §8, §9.5 and
§10 all describe tuning on a subset and deploying the winner. Keeping them forced
§5.4's most intricate paragraph — `uses_staged_gpu_strategy` routes to the staged
engine only when `process_only_layer` is `None` or `"objmap"`, so `staged_gpu`
had to report a mode-dependent dispatch answer — and they are the spec's largest
coupling to a storage redesign that is adding `--mode migrate` and changing what
`--mode recompile` does. **v1 deploy is always the full pipeline.** `sample` goes
with them: the subset *is* the thinning mechanism now, and two ways to shrink an
input set is one too many. Add `mode` back when a workflow needs it, against
whatever the enum is by then.

```json
{"ok":true,"data":{
  "routed_to":"slurm","profile":"cpu-bulk",
  "datasets":{"plateA":48,"plateB":52},"n_images":100,
  "argv":["python","-m","phenotypic","--mode","full","--pipeline","…","--slurm","slurm_partition=batch","…"],
  "sbatch_preview":"#!/bin/bash\n#SBATCH --partition=batch\n#SBATCH --account=exfab\n#SBATCH --array=0-99\n…",
  "array":{"requested":100,"chunks":1,"effective_limit":512,"limit_source":"profile cap"},
  "estimate":{"basis":"probe of 2 images at 3.4 s/image",
              "node_seconds":340,"node_hours":0.094,
              "wall_clock_hint":"~6 min at 100 concurrent"},
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

**`estimate` reports both `node_seconds` and `node_hours`, and `node_hours` is
the bound one.** The two are the same number and the response carried only the
first while §5.4's token bound the second and §10.5's promotion response
reported the second — so the field the human approves was not in the schema of
the tool that produces it. `node_hours` is the human-facing unit (it is what
`ack_prompt` quotes); `node_seconds` stays because the per-arm campaign
estimates (§8.3) are seconds-scale and mixing units across §5 and §8 is worse
than carrying both.

**At `scope:"full"` the response also echoes `group_filter`** — the map copied
from the subset artifact, or `null` — so the agent can see at plan time which
images a full-scope run will touch, rather than discovering the intersection at
submission. It is the same value §5.4's token binds.

## 5.4 `deploy_start` (`W3`) — submit

Same arguments as `deploy_plan`, plus:

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `scope` | `"subset" \| "full"` | `"subset"` | `full` targets `subset.parent` — intersected with the subset's `group_filter` where it has one (§10.5). Its `plan_token` must have been minted at `scope:"full"`; **the human ack is taken here, not carried on the token** |
| `plan_token` | `str` | — | **Required.** From a matching `deploy_plan`, or from an approved campaign arm |
| `human_response` | `str?` | — | **Required when `plan_token.kind == "plan"`** — a fresh spending decision. **Omitted when `kind == "campaign_arm"`**, which carries consent forward from `campaign_approve` (below) |
| `note` | `str?` | `null` | Free-text context recorded alongside the approval |
| `resume` | `bool` | `false` | Continue an interrupted run |
| `retry_failures` | `bool` | `false` | Requires `resume` |
| `restart` | `bool` | `false` | Clear machine state and start over. **Refused against an output root already holding deliverables from a different approved manifest** (`restart_would_mix_approvals`, USER-32) — see below |

The response carries
**`ack_source: "elicited" | "agent_asserted" | "campaign_approved"`** (§8.2), which
is what makes the distinction auditable on the artifact rather than implicit in
host configuration.

**The prompt is rendered by the server from the token, never passed in by the
agent.** This is why the token carries `ack_prompt` and `decision_content` rather
than only the digests. §10.5 mints the prompt at `deploy_plan` and calls it "text
to show" — but `deploy_start` is where it is now shown, one call later, and if
the server cannot reproduce it there are only two options and both are wrong:
the agent supplies the text, which makes the numbers a human reads
agent-chosen — the single thing §8.2 adopted elicitation to prevent — or the
server re-renders from digests alone and silently drops the subset score,
held-out gap and coverage warnings that §10.5 calls the decision content.
Persisting both on the token closes it, and it is also what lets §8.2's rule
"every elicitation leads with the artifact id it approves" be satisfied by a
handler rather than by an agent's good manners.

**The elicitation is raised before the slot is acquired and before `allocate`.**
This ordering is the whole point of moving the gate here, and stating it in three
words elsewhere is not enough: ask first, then allocate, then acquire. Reversed,
a locally-routed `W3` would hold the exclusive slot across a human's think time —
which under §1.5's own suspension rule blocks every subagent's probing for as
long as the person takes to answer, reproducing the hazard the relocation was
made to remove, one call later.

**Plan-then-submit is mandatory.** `deploy_start` refuses without a
`plan_token` **whose every bound field re-derives equal from the current
request** — the binding set below is the single statement of what those fields
are, and nothing outside this section restates it. A missing token returns
`code: "plan_required"`; a token that no longer matches returns `"plan_stale"`
naming the field that moved. Every cluster submission is therefore preceded by
an inspectable preview.

### What a token *is*

A token is **an opaque random id naming a persisted record** — not a digest of
its own contents.

```
<workspace>/.phenotypic-mcp/plans/<token>.json
{"token":"pl_7f3a…","kind":"plan","created":"…","expires":"…",
 "scope":"subset","pipeline_digest":"sha256:9c1e…","subset_id":"subsets/…",
 "subset_digest":"sha256:77b2…","compute":{"profile":"cpu-bulk","time":"02:00:00"},
 "parent_digest":null,"group_filter":null,
 "run_name":"runs/2026-08-12-plateA",
 "array":{"requested":480,"chunks":1,"effective_limit":2500},
 "estimate":{"node_hours":18.4},
 "ack_prompt":"Deploy edge-v3-tuned across 480 images (~18.4 node-hours)? …",
 "decision_content":{"subset_score":0.081,"gap":0.06,"warnings":["subset_coverage_unverified"]},
 "argv_digest":"sha256:4b0a…","consumed_by":null}
```

`parent_digest` and `group_filter` are `null` at `scope:"subset"` and populated
at `scope:"full"`; the table below is what says which. They are **fields of the
record at both scopes**, not fields that appear only at one — an optional key an
implementer has to infer from an example is how `group_filter` came to have no
storage location at all.

**The binding set grew when the token absorbed the promotion gate.** While the
token only meant "a plan was drawn", a stale one cost a re-plan. Now it carries a
human's consent to a specific quantity of somebody else's compute, so anything
quoted to that human has to be inside it.

**The table below is that set, and it is the only statement of it.** Every bound
field is here with the scope it binds under; no other section states a binding
set, and any section that needs to talk about one cites this table. It is written
that way because the set had been stated four times in four incompatible ways —
three here and two in §10.5 — and the disagreement was invisible because each
restatement read as a summary of the others rather than as a competing claim.

| Field | Scope | Why it binds |
|---|---|---|
| `scope` | both | `subset` and `full` are different spends; a token minted for one is not the other |
| `pipeline_digest` | both | The pipeline the human was quoted a cost for |
| `subset_id` | both | Which registered subset the plan was drawn against |
| `subset_digest` | both | The subset's own image list can be re-cut under the same name |
| `compute` | both | The profile and its overrides — what the work costs per node |
| `run_name` | both | Otherwise an ack given for `runs/2026-08-12-plateA` is spendable against a different output directory with every digest still matching |
| `array` | both | §5.3 resolves the width live from `scontrol`/`sacctmgr` (§5.2), so the cluster can re-chunk between plan and start. `compute` binds the *profile*, not the resolved width |
| `estimate.node_hours` | both | The number quoted verbatim in `ack_prompt`. This is the figure the human actually approves, and it was outside the token entirely |
| `image_manifest_digest` | `full` + filter | **Content** digest of `.phenotypic-mcp/plans/<token>.images`. `argv_digest` cannot stand in — it covers the string that *names* the file |
| `argv_digest` | both | The rendered invocation, defined below |
| **`parent_digest`** | **`full` only** | A parent that gained images between the plan and the submission invalidates the token (`plan_stale`, §10.5) rather than quietly deploying over a dataset nobody reviewed. `null` at `subset` |
| **`group_filter`** | **`full` only** | USER-21: full scope on a group-filtered subset is `parent ∩ group_filter`, so an ack given for one group's images cannot be spent on another's. The filter is copied from the subset artifact's `group_filter` (§10.2) at plan time and re-compared at start. `null` at `subset`, and `null` at `full` for a subset with no filter |

`ack_prompt` and `decision_content` are **carried on the record but are not
bound**: they are what the server re-renders the elicitation from, not inputs
re-derived from the request, so there is nothing to compare them against. Their
integrity comes from the bound fields they were computed from.

**`argv_digest` is the SHA-256 of the rendered argv list**, joined with `\0`, as
produced by `to_argv` plus the profile's `--slurm` pairs — including `--output`.
It and the explicit `run_name` field are both kept: `argv_digest` also moves when
a compute key changes, and the two failures deserve different messages.

**An absent `group_filter` normalizes to `null` everywhere.** The subset artifact
records `{}` for "no filter", the token record carries `null`, and §5.4 copies the
artifact's value onto the token and **re-compares it** at `deploy_start`. Left
unstated, `{} != null` fails that comparison and **every unfiltered full-scope
deploy returns `plan_stale`** — the common path, broken by a representation
detail. Normalize on the way in: an empty map and `null` are the same value, and
the token stores `null`.

**`image_manifest_digest` binds the resolved image set (USER-26), and
`argv_digest` cannot stand in for it.** At `full` scope with a non-null
`group_filter`, `deploy_plan` resolves `parent ∩ group_filter` to a concrete list
and writes it to **`.phenotypic-mcp/plans/<token>.images`** — under the token, so
it shares the token's lifecycle. The record binds the file's content digest, and
`deploy_start` re-derives and compares it.

Binding the argv instead would be a null guard: `argv_digest` covers the argv
*string*, which merely **names** the file, so a manifest whose *contents* changed
between plan and start re-derives an identical `argv_digest` and passes every
check — across a 24 h token lifetime, for a file the server itself collects. The
property USER-26 was adopted to provide is that a human approves an image set
which cannot subsequently drift, and only a content digest carries it.

**Note the name.** This is *not* `manifest.json`, the run-status manifest
`deploy_status` polls (§5.5, `_dashboard/_manifest_builder.py`). Two unrelated
artifacts called "the manifest" in one section is a defect waiting to happen, so
the image list is always **`image_manifest`** and never bare "manifest".

**Collection must not race a live run.** Expired tokens are collected on the next
`deploy_plan` or server start, but a full deploy is a multi-hour SLURM array that
reads its `image_manifest` long after the token expired and was consumed —
pulling the input list out from under a running job. **Collection skips any token
whose `consumed_by` names a run that is not yet terminal**, and takes the
`.images` file with the record when it does collect.

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
| Collection | Expired tokens are deleted on the next `deploy_plan` or server start. Without this, every plan the agent draws and abandons accumulates under `.phenotypic-mcp/plans/` indefinitely — and once a token can carry a human's consent, an abandoned one is not merely litter but a standing approval waiting for a matching request. Expiry already bounds the exposure to 24 h; collection is what stops it accruing |

The token **does not record the human ack** — USER-18 moved the ack to
`deploy_start`, one call later, so the token carries the *material* the gate is
rendered from and the ack lands on the `deploy.approve` lineage row (§2.5) and
the response's `ack_source`. A token that recorded an ack would be a second
mutable state racing the gate, which is the contradiction the relocation
dissolved.

### Who supplies the human's words, and when

`human_response` is keyed on the **token kind**, and the reason is that §10.4
lets a campaign carry a **deploy arm** which the background launcher runs
unattended. At 3am there is no human to ask, so an unconditional requirement
leaves only two outcomes and both are bad: the launcher fabricates the field —
the precise thing USER-22's audit trail exists to prevent — and records
`ack_source: "agent_asserted"`, which is **false**, because a human *did*
approve, at `campaign_approve`, for this arm, inside the budget they agreed; or
campaign deploy arms are simply unlaunchable and §10.4's capability is specified
but unreachable.

So a `campaign_arm` token carries the consent forward. Its `deploy.approve`
lineage row records **`ack_source: "campaign_approved"`** together with
`campaign_id` and `arm_id`, which points at the approval that actually happened
rather than inventing one that did not.

**This is not USER-22 re-litigated.** USER-22's objection was to a required-field
rule that varies with *host capability* — unpredictable from `tools/list`, and
forcing a fallback branch into the tool contract. A rule keyed on the token kind
is statically documentable, visible in the request the caller already makes, and
leaves one signature. The elicitation still fires for every human decision; it
just does not fire twice for the same one.

### The token's two producers, and what each can bind

The token is satisfied two ways: a direct `deploy_plan` call, or membership in
an **approved campaign** (§8), which stamps a token per arm at approval time.
That keeps the human checkpoint in the planning phase where you actually are,
rather than inserting one into autonomous Phase-2 execution.

**But a campaign arm cannot bind everything a `deploy_plan` token binds, and a
binding table that does not say so leaves an implementer with four mandatory
fields and no value for them.** At `campaign_approve` time
the arm has no `run_name` (the study is named `studies/<campaign>-<arm>` later,
at fan-out), no resolved `array` (nothing has consulted `scontrol`), no
`estimate.node_hours` from a `deploy_plan` (the campaign's own per-arm estimate
is a *tune* estimate, §8.3), and no `argv_digest` (no argv has been rendered).
Running a full `deploy_plan` per arm at approval time is not what §8.3 does and
would put a cluster query inside a `W0` handler.

So a campaign-stamped token is a **distinct kind**, and the record says which:

| `kind` | Minted by | Binds | `scope` |
|---|---|---|---|
| `"plan"` | `deploy_plan` | the full table above | `subset` or `full` |
| `"campaign_arm"` | `campaign_approve` (§8.3) | `scope`, `pipeline_digest`, `subset_id`, `subset_digest`, `compute`, plus `campaign_id` and `arm_id` | **`subset` only** |

`run_name`, `array`, `estimate.node_hours` and `argv_digest` are **absent, not
null**, on a `campaign_arm` token, and validation does not look for them: the
arm's spend is bounded by the campaign the human approved, whose own budget
(`trials_per_arm`, `max_concurrent_arms`, `compute`) is the quantity that was
quoted. `campaign_arm_scope_full` (§6.2) already refuses a campaign arm at full
scope, which is what keeps the weaker binding set off the irreversible path —
the full-dataset spend is reachable only through a `"plan"` token, which binds
everything.

**`overwrite` is deliberately not exposed.** The CLI's `--overwrite` does
`shutil.rmtree(output_dir)` — it destroys `deliverables/`, every per-image store,
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

**`restart` requires a fresh output root when an approved manifest is involved.**
`--restart` sets `continuing = False` (`phenotypicCLI.py:1486-1491`), which
bypasses the manifest drift check **while keeping `results/`, `deliverables/` and
`qc/`**. So a re-deploy carrying a *different* manifest would be accepted with no
comparison at all, and deliverables would silently blend two separately-approved
image sets — a state nobody can untangle afterwards, because no row records which
approval it came from.

So `deploy_start {restart: true}` is refused when the output root already holds
deliverables produced under a **different** `image_manifest_digest`. One output
root, one approved set. The agent's remedy is a new `run_name`, which costs
nothing.

The alternative — making `--restart` participate in the drift guard — was
rejected because it would require auditing every other flag that sets
`continuing = False` for the same bypass, and a guard whose coverage depends on
an audit is not a guard.

**`sample` and `image_manifest` are mutually exclusive** (USER-33). `--sample N`
applies *after* the manifest, so a human could approve 312 images and have 20
run with the manifest and its digest both unchanged. The two express
contradictory intents — a set someone specifically approved is not a set to
sample — so the combination is a hard error at the CLI rather than an ordering
subtlety for an agent to reason about.

**The same pre-submit block also checks for a half-migrated output tree.** The
OME-Zarr store makes *every mode that consumes results* refuse against a tree
that is partly `.h5` and partly `.ome.zarr` — and because migration is itself
resumable, a half-migrated tree is the **expected** state after any interruption,
not a corner. That refusal is another `sys.exit(1)` inside the subprocess, which
is exactly the opaque failure this block exists to convert into something an
agent can act on. So `deploy_start` calls `datasets_needing_migration(output_dir)`
alongside `validate_resume_compatibility` and returns a structured
`migration_required` naming the datasets.

**And it must say what to do about it, because the server cannot do it.** USER-8
cut `mode` from the deploy tools, so there is no `deploy_start {mode:"migrate"}`
to offer — a refusal with no in-server escape is a dead end on the irreversible
path. The error therefore **carries the remedy as a shell command**:
`python -m phenotypic --mode migrate --output <run>`.

Surfacing the command is preferred over re-admitting `migrate` as a tool
argument: it keeps USER-8's cut intact, and migration is a one-time
storage-format operation a human should run knowingly rather than something an
agent triggers mid-campaign. If that proves too indirect in practice, re-admitting
a narrow `migrate` remains available — but it is a reversal of a user ruling and
would need one.

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

A pipeline containing a `GpuDetector` (`pipeline_requires_gpu`) *always*
triggers the staged engine — CPU preprocess → resident-model GPU detect → CPU
measure — not per-image processing. On SLURM that becomes an epoch-fenced
controller with Stages 1 & 3 on the CPU profile and Stage 2 as a GPU array.

**It is unconditional.** `uses_staged_gpu_strategy`
(`_cli_execution_strategies.py:1058-1068`) routes to the staged engine whenever
`process_only_layer` is `None`, and with `mode`/`layer`/`sample` cut from the
deploy tools (§5.3) that argument is always `None`. `deploy_plan`'s `staged_gpu`
flag is therefore derived rather than independently computed: it is simply
`requires_gpu`.

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
