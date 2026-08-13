# PhenoTypic MCP Server — §3 Tool Contract: Catalog, Pipeline, Workspace

Status: **draft, reviewed once, revised**
Date: 2026-08-12

## 3.0 Conventions binding every tool

**Naming.** `<group>_<verb>`, flat, no dots — `pipeline_put`, never `pipeline.put`.
31 tools in nine groups: catalog (3), pipeline (5), workspace (4),
assay (2, §9.3), **subset (3, §10.3)**, tune (5, §4), deploy (3, §5),
campaign (4, §8), promotion (2, §10.5).

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

**Ids are sandbox-relative paths** (§2.2). The bare stem (`"edge-v3"`) is
accepted as sugar, resolved with `matches_any_suffix` against
`PIPELINE_CONFIG_SUFFIXES` — never `Path.suffix`, which sees only `.pht-pipe`.

**Token discipline.** List tools return compact rows; full JSON schemas come
only from the detail tool, one operation at a time. No tool returns an unbounded
measurement table — dataframes are summarized, with a parquet path for more.

---

## 3.1 Catalog group (all `W0`)

### `catalog_operations`

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `category` | `str?` | `null` | `Enhancer, Detector, Refiner, Corrector, Measure, Grid, Post, Filter, Model, Edge Correction, quality_check, Prefab` |
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

### `catalog_operation_detail`

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | — | Operation class name |
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
| `pipeline` | `object` | — | Spec below |
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
- `move_op.to` is the index in the list **after** the source is removed.
- Edits apply in array order, each seeing the previous one's result.

All edits apply to an in-memory deep copy; **the file is written only if every
edit validates**, so a failing third edit leaves the artifact untouched. Returns
the `pipeline_put` payload plus a `diff`.

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
| `images` | `str` | — | File or directory |
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
- A probe sends **`{pipeline_path, image_paths, options}` as JSON** over the
  pipe. It does **not** send a pipeline envelope or a pickled object: the worker
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
  "counts":{"pipelines":3,"tune_specs":3,"assays":1,"subsets":1,
             "campaigns":1,"studies":1,"runs":0},
  "active_subset":"subsets/plates-dev-24.subset.json"}}
```

`tune.journal_backend_enabled` reflects §7 P1: while `false`, a SLURM tune
request fails with an actionable error naming Postgres rather than submitting
into the SQLite corruption case. `in_flight` and `rehydrate_ms` make the
workspace-immutability rule (§2.2) and rehydration cost visible.

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

---

## 3.4 Worked example: one subagent, one pipeline

```
catalog_operations       {category:"Detector"}
catalog_operation_detail {name:"OtsuDetector"}
pipeline_put             {name:"edge-v3", pipeline:{…}}
  -> produces_columns [Size_Area, Shape_Circularity, …], requires_gpu false
pipeline_probe           {pipeline_id:"edge-v3", images:"data/plateA", n_images:2}
  -> 94 objects/image, Size_Area mean 412 — blur may be too strong
pipeline_patch           {pipeline_id:"edge-v3",
                          edits:[{kind:"set_params", slot:"ops", index:0, params:{sigma:1.2}}]}
pipeline_probe           {pipeline_id:"edge-v3", images:"data/plateA", n_images:2}
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
