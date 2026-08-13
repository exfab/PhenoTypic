# PhenoTypic MCP Server — §4 Tune Integration Contract

Status: **draft, pending review**
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

Backed by `pipeline_targets(pipeline)` → `list[TunableParam]`, which wraps
`infer_search_space`.

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
 "excluded":[{"label":"CannyDetector.mask","reason":"ndarray"}],
 "n_needs_review":0,
 "scorers_available":[
   {"class":"QCScorer","available":true,
    "requires":"a metadata CSV of expected counts","found":"data/tune_layout.csv"},
   {"class":"SupervisedScorer","available":false,
    "requires":"ground-truth masks or a count table",
    "hint":"no gt_masks_source found under the workspace"},
   {"class":"ReferenceFreeScorer","available":false,
    "requires":"meta_validate() to pass against a labelled subset"}]}}
```

Three deliberate affordances:

- **`ref` is the selection handle.** `tune_put_spec` takes `ref`s. The full
  `target` object is included so a power caller can construct one directly, and
  so the agent can reason about op positions.
- **`needs_review: true`** marks a domain inference guessed from an unbounded
  field (`[default/4, default*4]`, `source: "unbounded_heuristic"`). The agent
  should either narrow it or say it is guessing.
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
| `select` | `array` | — | Chosen knobs (below) |
| `scorer` | `object` | — | `{class, params}` |
| `strategy` | `object` | — | `{kind, n_trials?, sampler?, seed?}` |
| `budget` | `object?` | `{}` | `{n_trials?, max_failures?}` |
| `evaluator` | `object?` | `{}` | Overrides on `Evaluator` defaults |
| `held_out` | `object?` | `{}` | `{held_out_fraction?, group_key?, …}` |
| `overwrite` / `dry_run` | `bool` | `false` | As §3 |

```json
{"name":"edge-v3-tpe",
 "pipeline_id":"edge-v3",
 "select":[
   {"ref":0},
   {"ref":1, "enabled":false},
   {"ref":2, "domain":{"kind":"float_range","low":1.0,"high":4.0,"step":0.25}}],
 "scorer":{"class":"QCScorer",
           "params":{"check":{"metadata":"data/tune_layout.csv"}}},
 "strategy":{"kind":"optuna","sampler":"tpe","n_trials":200,"seed":0},
 "held_out":{"held_out_fraction":0.2}}
```

`select` semantics: omitting `domain` accepts `suggested_domain`; `enabled:false`
drops the knob entirely; supplying `domain` overrides. A `ref` that no longer
resolves (because the pipeline changed since `tune_space`) is a hard error naming
the drift, **not** a silent re-index — the server records the pipeline digest in
the `tune_space` response and rejects a `select` built against a stale digest.

### Pre-submit checks the server adds

`TuningSpec` construction catches target errors. These four it does not, and
each otherwise fails late and confusingly:

| Check | Why | Code |
|---|---|---|
| Scorer `availability()` | `run_tuning` asserts it only after writing run artifacts | `scorer_unavailable` |
| `grid` + a continuous `FloatRange` (`step: null`) | `grid_values` raises `ValueError` at enumeration time | `grid_needs_stepped_domain` |
| `grid` + `n_trials` | `resolve_strategy` rejects the combination | `grid_ignores_n_trials` |
| Empty active knob set | A study with nothing to vary | `no_active_knobs` |

Multi-objective + grid/random is already rejected by
`reject_grid_random_multi_objective`; the server surfaces that error rather than
duplicating it.

### The `QCScorer` round-trip trap

`QCScorer` holds an `ExpectedVsDetectedCount` check. **It must be configured
from a metadata *path*, not an in-memory DataFrame** — a DataFrame-configured
check cannot be reloaded from JSON, and a SLURM worker reloads the spec from
disk. The server enforces this: `scorer.params.check.metadata` must be a
sandbox-resolvable path, rejected at put time with `code: "scorer_not_portable"`.
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
| `images` | `str` | — | Image directory |
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
| SLURM, P1 not landed, Postgres configured | the configured password-less URL, with a per-study database |
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

`progress` is cheap and marker-only: `TuneRunRoot.discover(path)` reads
`.pht-tune-cache/run.json`, falling back to `deliverables/tuning_spec.json`, then
legacy `trials.parquet` — **it never opens Optuna**. That is what makes polling
safe to do often, and it is the GUI Monitor's own tick behaviour.

`results` opens the study (or degrades to `trials.parquet` when the store is
unreachable) and returns the leaderboard, best trial, parameter importances,
Pareto front for multi-objective, and the generalization report.

```json
{"ok":true,"data":{
  "status":"running","completed":126,"pruned":14,"failed":3,"budget":200,
  "best":{"trial":47,"score":0.081,
          "params":{"BlurGauss.sigma":1.34,"OtsuDetector.__enabled__":true}},
  "gap":{"value":0.06,"verdict":"ok"},
  "scheduler":{"job_ids":["4412331"],"reachable":true}}}
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

Backed by `export_best_from_run(output_dir)` / `export_pareto_pipeline`. Winner
selection follows `_headline_winner`: the Pareto **knee point** when a front
exists, otherwise `best()`. The response states which rule applied
(`selection: "pareto_knee" | "single_best"`), because for a multi-objective study
"the best pipeline" is a choice, not a fact.

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
tune_start      {spec_id:"edge-v3-tpe", images:"data/plates",
                 compute:{profile:"cpu-bulk", n_workers:8}}
                -> routed_to slurm, study_id studies/edge-v3-tpe

# orchestrator polls all three cheaply
tune_status {study_id:"studies/edge-v3-tpe"}   -> 126/200, best 0.081
tune_status {study_id:"studies/watershed-tpe"} -> 143/200, best 0.117
tune_status {study_id:"studies/canny-tpe"}     -> 98/200,  best 0.204

tune_export_best {study_id:"studies/edge-v3-tpe", name:"winner"}
                -> pipelines/winner.json.pht-pipe, selection single_best
```

Because each study's storage URL is derived from its own output directory, the
three studies are isolated by construction — the same property local SQLite
already has, now extended to the distributed case by P1.

## 4.7 Open questions

- **OQ-4.1 — screening.** `ScreeningController` (`_screening_freeze.py:244`) can
  freeze low-importance params mid-study, exposed as `--screen`/`--no-screen`.
  Should `tune_put_spec` expose it at all? Two reasons for caution: it changes
  the search space mid-run, which an agent reading `tune_space` output would not
  expect; and **`--screen` + `--slurm` is a silent no-op today** — `run_tuning`
  returns from `_submit_slurm_fleet` before reaching its `if screen:` block, and
  the worker builds no `ScreeningController` at all (§7 P4). Closing that hole is
  a prerequisite either way; whether to then expose the knob is the open part.
- **OQ-4.3 — `trials.parquet` for distributed studies.** Given no `--recompile`
  exists, should `tune_export_best` also write `trials.parquet` from the live
  store (making the study directory self-contained and GUI-readable offline), or
  is leaving it store-only acceptable?

**Resolved since first draft:**

- ~~OQ-4.2 who picks the scorer~~ → **explicit always**. `tune_space` reports
  availability, but the agent names the scorer even when only one is available.
