# PhenoTypic MCP Server — §9 Separation of Responsibilities, and the Bundled Skills

Status: **draft, pending review**
Date: 2026-08-12

## 9.1 The dividing principle

Sections 1–8 specify **mechanism**. They do not say what a *good* pipeline for a
filamentous fungus looks like, or that you should try a prefab before authoring
something new. That knowledge is real, it is what makes the difference between a
useful agent and a random-search machine — and it does not belong in the server.

The test for which layer a rule belongs to:

> **Could a well-intentioned expert reasonably want the opposite?**
> If **no** — the rule protects the cluster, the data, or correctness → it is
> **server enforcement**.
> If **yes** — it is domain judgment a knowledgeable person might override → it
> is **skill guidance**.

"Never expose `--overwrite`" passes the first test: no expert wants an agent
able to `rmtree` a run's deliverables from a tool call. "Try
`FilamentousFungiPipeline` before authoring a custom filamentous detector"
fails it: an expert who already knows this strain segments badly under that
prefab should skip it. So the first is a server refusal, the second is a skill
instruction.

Putting a rule in the wrong layer fails in a specific way:

| Mistake | Failure mode |
|---|---|
| Domain judgment encoded in the server | The server has opinions it cannot justify and you cannot override. Every new organism needs a server release. |
| Enforcement left to a skill | A subagent that ignores or never loads the skill can still melt the node or delete data. Skills are advice; advice is not a boundary. |

## 9.2 The four layers

| Layer | Owns | Never does |
|---|---|---|
| **Engines** (`_core`, `tune`, `_cli`) | Computation, serialization, scheduling, all numeric results | Know an agent exists |
| **MCP server** (§1–§8) | Mechanism, validation, routing, resource guards, refusals, structured errors | Encode domain judgment; pick an operation for you; decide a pipeline is "good" |
| **Skills** (this section) | Domain judgment, procedure, heuristics, what-to-try-first, how to read results | Enforce anything; bypass a server refusal; substitute for a validation |
| **Human** (you) | Assay knowledge the images cannot reveal, campaign approval, destructive operations | — |

The clean statement: **the server makes wrong things impossible; the skill makes
right things likely.**

## 9.3 Upfront assay characterization

Pipeline choice is driven by traits of the *assay*, not by the image file.
Several of the decisive ones cannot be measured from a plate image at all —
whether the organism is filamentous or yeast-like is something you know and the
agent does not.

So Phase 1 (§8.1) opens with an **assay triage**, producing a durable artifact:

`<workspace>/assay.json`

```json
{
  "schema_version": 1,
  "name": "exfab-fungal-2026-08",
  "organism": {
    "morphology": "filamentous",        // filamentous | round | mixed | unknown
    "source": "human",                  // human | inferred | probe
    "notes": "Aspergillus spp., hyphal spread expected by 72 h"
  },
  "colony": {
    "contrast_vs_background": "low",    // high | moderate | low
    "separation": "touching",           // well_separated | touching | confluent
    "pigmentation_informative": true,
    "source": "probe"
  },
  "plate": {"format": "arrayed", "nrows": 8, "ncols": 12},
  "imaging": {"modality": "flatbed_scanner", "bit_depth": 16, "source": "metadata"},
  "evidence": {"probe": "runs from pipeline_probe on 2 images", "images_seen": 2}
}
```

Every field carries `source`, because the three ways of knowing are not equally
trustworthy and the agent must not blur them:

| `source` | Meaning |
|---|---|
| `human` | You told the agent. Authoritative; never overwritten by inference. |
| `probe` | Measured from images via `pipeline_probe` evidence (contrast, object count, size distribution). |
| `metadata` | Read from image metadata (bit depth, dimensions, EXIF). |
| `inferred` | The agent guessed. Must be stated as a guess when it drives a decision. |

**The agent asks for what it cannot measure and measures what it can.** Morphology
is almost always `human`; contrast and separation are genuinely measurable from
a probe; bit depth and dimensions come from metadata. A skill that guesses
morphology from an image and does not say so is the specific failure this
artifact exists to prevent.

The assay profile is referenced by the campaign (§8.2), so a leaderboard is
always interpretable months later: *these arms were chosen because the organism
was filamentous and contrast was low.*

## 9.4 Prefab-first construction

**Rule: try the relevant prefab pipelines before authoring a new one.**

Seven ship today (`phenotypic.prefab`), and they are validated, documented
chains rather than examples. Their real intents:

| Prefab | Intent (from its docstring) |
|---|---|
| `FilamentousFungiPipeline` | Filamentous fungi detection with `DenoiseBlockMatch` denoising and spatial measurements |
| `RoundPeaksPipeline` | Round colonies, lightweight peak-based detection |
| `HeavyRoundPeaksPipeline` | Round colonies, peak detection with full refinement |
| `HeavyWatershedPipeline` | Watershed segmentation for **touching** colonies |
| `HeavyOtsuPipeline` | Multi-stage Otsu thresholding with refinement |
| `GridSectionPipeline` | Per-section processing on grid plates |
| `SpImagerPipeline` | Light processing for SpImager-sourced images |

### Assay profile → candidate prefabs

This table is **skill content, not server logic** — it is exactly the kind of
judgment an expert may override.

| Assay signal | First candidates |
|---|---|
| `morphology: filamentous` | `FilamentousFungiPipeline` |
| `morphology: round` + `separation: well_separated` | `RoundPeaksPipeline`, then `HeavyRoundPeaksPipeline` |
| `morphology: round` + `separation: touching` | `HeavyWatershedPipeline`, then `HeavyRoundPeaksPipeline` |
| `contrast: high` + `separation: well_separated` | `HeavyOtsuPipeline` (cheapest that can work) |
| `contrast: low` | Prefer the refinement-heavy variants; expect enhancement to matter more than detector choice |
| `plate.format: arrayed` and dense | `GridSectionPipeline` |
| `imaging.modality: spimager` | `SpImagerPipeline` |
| `morphology: mixed` | Two arms — the filamentous and round candidates — rather than one compromise pipeline |

### The procedure

1. Pick candidate prefabs from the assay profile — usually one or two, three at
   most.
2. `pipeline_probe` each on the same 2 images. Compare object counts, size
   distributions, and per-op timing.
3. Tune the best prefab **before** authoring anything custom. A prefab whose
   parameters are wrong for your assay is not a failed prefab; `tune_space` on a
   prefab is the cheapest large improvement available.
4. Author a custom pipeline **only** when the best tuned prefab still fails a
   stated bar — and record why.

**Custom pipelines carry a justification.** When an arm's pipeline is not a
prefab or a prefab derivative, the campaign records which prefab came closest and
how it failed:

```json
{"id":"custom-phase","pipeline":"pipelines/phase-edge.json.pht-pipe",
 "rationale":"FilamentousFungiPipeline tuned to 0.31 cost; under-segments hyphal
              edges on the low-contrast plates (probe: 61 vs ~95 expected objects).",
 "prefab_baseline":{"pipeline":"FilamentousFungiPipeline","best_cost":0.31}}
```

This is a **skill-enforced convention with server support**: the server supplies
the `prefab_baseline` field and validates its shape if present, but does not
refuse a campaign that omits it. The knowledge that a bare custom pipeline is
usually premature is judgment, not a boundary. What the server *does* guarantee
is that if the field is there, it is well-formed and its referenced study exists.

`catalog_operations` exposes prefabs (resolved OQ-3.1, §3.1), so the agent can
discover them rather than needing them memorized.

## 9.5 The bundled skills

Four skills ship with the server. Each maps to one phase and states its
tool sequence, so an agent that loads it knows both *what to do* and *which
tools do it*.

### `phenotypic-assay-triage`

**When:** at the start of any new dataset, before any pipeline exists.
**Produces:** `assay.json`.
**Procedure:** ask the human for morphology and expected colony count per plate;
read imaging metadata; run one `pipeline_probe` with a prefab candidate to
measure contrast and separation; record every field with its `source`; state
explicitly which fields are guesses.
**Hard rule it teaches:** never write `source: "human"` for something the human
did not say.

### `phenotypic-pipeline-construction`

**When:** after triage, when choosing what to try.
**Procedure:** §9.4 — prefab-first, probe-compare, tune-before-authoring, and the
justification convention for custom pipelines.
**Tools:** `catalog_operations`, `catalog_operation_detail`, `pipeline_put`,
`pipeline_probe`, `pipeline_patch`, `pipeline_diff`.
**Hard rule it teaches:** a probe result is evidence about *those two images*.
Two probes are not a validation.

### `phenotypic-tuning-campaign`

**When:** turning candidates into a campaign.
**Procedure:** one scorer for all arms (§8.2); pick the scorer from what the
workspace actually has (`tune_space` reports availability); include a baseline
arm and a control arm, not only the hypothesis; size the budget from the probe
timing; narrow `needs_review` domains rather than accepting inferred bounds.
**Tools:** `tune_space`, `tune_put_spec`, `campaign_put`, `campaign_approve`,
`campaign_start`, `campaign_status`.
**Hard rule it teaches:** cost is in `[0, 1]` and **lower is better**. Report the
held-out `gap`, never the calibration score alone — an arm that won by
overfitting the split is not a winner.

### `phenotypic-deploy-and-verify`

**When:** a winner exists and a full dataset is to be processed.
**Procedure:** `deploy_plan` and show the node-hour estimate *before* asking;
submit; poll `manifest.json` rather than trusting exit codes; on completion,
verify against the mirror (`measurements.*`), not the master, and report
`QC_MetadataOnly` rows separately from detections.
**Hard rule it teaches:** deletion and overwrite are the human's job at a shell.
If the output directory is occupied, ask — do not find a way around it.

## 9.6 Skill/server boundary — worked cases

| Rule | Layer | Why |
|---|---|---|
| Path must resolve inside the workspace | **Server** | No expert wants otherwise |
| Only one local image computation at a time | **Server** | Protects a shared node |
| Named SLURM profile, capped overrides | **Server** | Protects a shared allocation |
| Campaign arms share one scorer | **Server** | Otherwise the leaderboard is meaningless — a correctness property |
| Try prefabs before custom pipelines | **Skill** | An expert may legitimately skip them |
| `HeavyWatershedPipeline` for touching colonies | **Skill** | Heuristic; assay-dependent |
| Include a baseline and a control arm | **Skill** | Good method, not a correctness rule |
| Ask the human for organism morphology | **Skill** | Procedural discipline |
| A `journal://` study must not share a URL with another live study | **Server** | Silent trial pooling is data corruption |

## 9.7 Open questions

- **OQ-9.1 — where do the skills live?** Shipping them in the repo under
  `.claude/skills/` versions them with the tool contract, which matters because a
  skill naming `pipeline_diff` is wrong until that tool exists. Shipping them
  inside the installed `phenotypic` package instead means `pip install` brings
  them, but they then need a discovery mechanism. Which do you want?
- **OQ-9.2 — should `assay.json` be per-workspace or per-dataset?** One workspace
  may hold several organisms. Per-dataset is more correct and more bookkeeping;
  per-workspace is simpler and silently wrong when you add a second organism.
- **OQ-9.3 — should the server validate `assay.json` at all?** Currently it is a
  skill-authored artifact the server only stores and echoes. Giving it a schema
  makes it checkable and referenceable from a campaign; leaving it free-form
  keeps domain vocabulary out of the server, per §9.1.
