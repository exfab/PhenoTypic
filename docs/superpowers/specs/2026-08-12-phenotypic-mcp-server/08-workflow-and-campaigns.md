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
                           "params": {"check": {"metadata": "data/tune_layout.csv"}}},
                "sense": "cost in [0,1], lower is better"},
  "budget": {"trials_per_arm": 200, "max_concurrent_arms": 3},
  "compute": {"profile": "cpu-bulk", "n_workers": 8},
  "arms": [
    {"id": "prefab-fil", "pipeline": "pipelines/filamentous-prefab.json.pht-pipe",
     "tune_spec": "tune/filamentous-prefab.setup.json.pht-tune",
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

Four tools, bringing the total to 22.

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
`{"arm":"otsu","unchanged":true}` and only movement is returned. A multi-hour
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
       pipeline_probe with FilamentousFungiPipeline to measure contrast/separation
       → writes assay.json: morphology filamentous (human), contrast low (probe),
         separation touching (probe), 8x12 arrayed

agent: [skill: phenotypic-pipeline-construction — prefab-first]
       catalog_operations {category:"Prefab"}
       → assay says filamentous + touching → candidates:
         FilamentousFungiPipeline, HeavyWatershedPipeline
       pipeline_probe both on the same 2 images
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

agent: campaign_approve {campaign_id:"campaigns/fungal-edge-sweep"}
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
