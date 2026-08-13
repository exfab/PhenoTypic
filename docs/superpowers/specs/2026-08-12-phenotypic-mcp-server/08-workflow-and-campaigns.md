# PhenoTypic MCP Server — §8 Workflow, UX, and Campaigns

Status: **draft, pending review**
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
        │
        ▼
Phase 1 — PLAN (human + agent, conversational, all W0)
  Driven by the assay profile: prefab pipelines FIRST (§9.4), probed and
  compared, tuned before anything custom is authored. You and the agent
  settle topologies, knobs, scorer, compute. Nothing is submitted; every
  tool here is read-only or writes only draft artifacts.
        │
        ▼
  A CAMPAIGN: the agreed set of arms, written down, reviewed by you.
        │
        ▼  ← the human checkpoint lives HERE, once, not at every submission
Phase 2 — EXECUTE (agent, autonomous, W2/W3)
  The orchestrator fans out one subagent per arm. Each builds its pipeline,
  authors its tune spec, launches, and polls. You are not in this loop.
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
  "assay": "assay.json",
  "dataset": {"images": "data/plates", "metadata_csv": "data/tune_layout.csv",
              "n_images": 42, "digest": "sha256:1a4c…"},
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
leaderboard that means nothing. `campaign_put` therefore takes the scorer at
campaign level and **rejects** an arm whose tune spec disagrees, with
`code: "arm_scorer_mismatch"`. This is the single most valuable invariant the
campaign concept adds.

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
| All arms share the campaign scorer | Structural comparison; `arm_scorer_mismatch` |
| Scorer is available and portable | `availability()` + the `QCScorer` path rule (§4.2) |
| Compute profile exists; overrides within caps | §5.2 |
| Arms resolve to distinct storage URLs | The H2 guard (§7) |
| Dataset resolves and is non-empty | `SandboxRoot` + directory scan |

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
  "plan_tokens":{"otsu":"…","phase":"…","watershed":"…"}},
 "issues":[{"severity":"warning","code":"needs_review_domain",
            "message":"phase arm: FocusEdgePhase.k has an inferred unbounded domain [0.5, 8.0]; inference guessed it.",
            "path":"arms[1].knobs[2]"}]}
```

`est_node_hours` is the number you actually want before agreeing to anything,
and its `basis` says whether it came from a real probe or a default.

### `campaign_approve` (`W0`)

`{campaign_id, note?}` → flips `draft` → `approved`, stamps `approved_at`,
appends a lineage row. Refuses if any blocking issue from `campaign_put` is
unresolved, so approval cannot outrun validation.

### `campaign_start` (`W2`)

`{campaign_id, arms?}` → launches arms, honouring `max_concurrent_arms` and the
routing rules of §1.5. Refuses a `draft` campaign with
`code: "campaign_not_approved"`.

Each arm launches through the ordinary `tune_start` path — `RunRegistry.allocate`
→ `LocalRunner.start` → CAS — so campaign arms are ordinary studies, visible in
`workspace_list` and in the GUI. A campaign is an *organizing layer*, not a
parallel execution engine.

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

`comparable` is false — with an explanation — when arms cannot be honestly
ranked: a scorer drifted, an arm ran on a different dataset digest, or an arm
failed too heavily for its best trial to be meaningful. **A leaderboard that
silently ranks incomparable things is worse than no leaderboard**, so the field
is mandatory in the response rather than inferred by the reader.

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
       pipeline_probe {pipeline_id:"phase-edge", images:"data/plates", n_images:2}
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

The orchestrator spawns one subagent per arm. Each subagent owns exactly one
arm and calls only that arm's ids, which is why §2.2 requires distinct explicit
names rather than auto-suffixing.

What keeps this safe is entirely in §1.5 and §2.4: the subagents' `W0` calls
interleave freely, their probes serialize on the one `LocalComputeSlot`, and
`RunRegistry.allocate` refuses two claims on one output directory. Nothing
about fan-out needs new machinery here.

The orchestrator polls `campaign_status`, not three `tune_status` calls, and
reports to you on completion or on the first arm that fails hard.

## 8.6 Open questions

- **OQ-8.1 — should a campaign be able to include deploy arms?** As written a
  campaign is a *tuning* comparison and deployment is a separate step after a
  winner is chosen. The alternative is a campaign that also carries "then deploy
  the winner to dataset X", making the whole study→deploy chain one approved
  unit. That is more automation past the human checkpoint — attractive for
  overnight work, but it means a full-dataset run launches without you seeing
  the winner first.
- **OQ-8.2 — re-planning mid-campaign.** If an arm fails early (as `watershed`
  does above), should the agent be able to amend the campaign and add a
  replacement arm autonomously, or does an amendment require a fresh
  `campaign_approve`? The strict reading — amendment needs approval — protects
  the checkpoint but may strand an overnight campaign on one bad arm.
