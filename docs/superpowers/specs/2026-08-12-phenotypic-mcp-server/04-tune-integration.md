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
