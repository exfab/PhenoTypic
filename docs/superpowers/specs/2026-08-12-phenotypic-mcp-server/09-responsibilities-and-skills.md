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

### 9.3.0 The extensibility rule

Traits that narrow pipeline construction will keep being discovered — medium
opacity, incubation timepoint, dimorphic switching, plate lid glare. A schema
with a fixed field set would need a **server release per trait**, which is
precisely the coupling §9.1 exists to prevent.

So the artifact is an **open map of traits under a uniform envelope**:

> **The server validates the shape of a trait, never the set of traits.**

Adding a trait requires no server change, no schema bump, and no code — only a
new row in the skill-owned registry (§9.3.4).

`<workspace>/assays/<dataset>.assay.json`

```json
{
  "schema_version": 1,
  "name": "exfab-fungal-2026-08",
  "dataset": {"path": "data/plates", "digest": "sha256:1a4c…", "n_images": 42},
  "traits": {
    "organism.morphology": {
      "value": "filamentous",
      "source": "human",
      "note": "Aspergillus spp., hyphal spread expected by 72 h"
    },
    "colony.contrast_vs_background": {
      "value": "low",
      "source": "probe",
      "evidence": {"measure": "otsu_eta", "measured": 0.31,
                   "probe_ref": ".phenotypic-mcp/probes/filamentous-prefab/"}
    },
    "colony.separation": {
      "value": "touching",
      "source": "probe",
      "evidence": {"measure": "expected_vs_detected", "measured": 61, "expected": 96}
    },
    "plate.format":  {"value": "arrayed", "source": "human"},
    "plate.nrows":   {"value": 8,  "source": "human"},
    "plate.ncols":   {"value": 12, "source": "human"},
    "imaging.modality":  {"value": "flatbed_scanner", "source": "metadata"},
    "imaging.bit_depth": {"value": 16, "source": "metadata"},

    "medium.opacity": {"value": "opaque", "source": "human"}
  }
}
```

That last entry is the point: `medium.opacity` is a trait added *after* v1. It
required a registry row in the skill and **nothing else** — no server change, no
`schema_version` bump, no migration.

### 9.3.0.1 The trait envelope

Every entry in `traits` has the same shape, and this is the only structure the
server knows:

| Key | Required | Meaning |
|---|---|---|
| `value` | yes | Scalar or bool. The server does not interpret it. |
| `source` | yes | One of the four provenance values (§9.3.1). **The only closed enum.** |
| `evidence` | when `source: "probe"` | `{measure, measured, …}` — the named measure and its number, so the claim is auditable and the bands recalibratable |
| `note` | no | Free text for a human reader |
| `confidence` | no | Optional `[0,1]`; reserved for traits that later warrant it |

**Unknown trait keys round-trip verbatim.** The server preserves any trait it
does not recognize, rather than dropping or rejecting it — the opposite of
pydantic's `extra="forbid"` used elsewhere in this codebase, and a deliberate
inversion. An older server reading a newer skill's assay must not silently
discard a trait; silent loss of a trait that drove a pipeline decision would
make the artifact actively misleading. Forward compatibility is a correctness
property here, not a convenience.

**Trait keys are dotted and namespaced** (`<group>.<trait>`), so a new group
(`medium.*`, `growth.*`, `stress.*`) needs no structural change either.

### 9.3.1 Provenance vocabulary — the only enum the server enforces

Every field carries `source`, because the four ways of knowing are not equally
trustworthy and the agent must not blur them:

| `source` | Meaning |
|---|---|
| `human` | You told the agent. Authoritative; never overwritten by inference. |
| `probe` | Measured from images via `pipeline_probe` evidence. |
| `metadata` | Read from image metadata (bit depth, dimensions, EXIF). |
| `inferred` | The agent guessed. Must be stated as a guess wherever it drives a decision. |

**The agent asks for what it cannot measure and measures what it can.**

### 9.3.2 Domain vocabulary — what each term means and what it drives

This is the vocabulary the **skill** owns. The server never learns what any of
these words mean (§9.3.4).

**`organism.morphology`** — `filamentous | round | mixed | unknown`

| Value | Meaning |
|---|---|
| `filamentous` | Hyphal growth; irregular, non-convex, diffuse boundaries; colonies may merge into a mycelial mat |
| `round` | Yeast or bacterial colonies; approximately circular, convex, discrete |
| `mixed` | Both on the same plate — co-culture, dimorphic switching, or contamination |
| `unknown` | Not stated and not inferable |

*Determined by:* **the human, essentially always.** It could in principle be
inferred from a `Shape_Circularity` distribution, but only *after* a detection
exists — and the detector you would choose depends on the answer. That
circularity is why it must be asked, not guessed.

*Drives:* prefab choice (`FilamentousFungiPipeline` vs `RoundPeaks*`), and which
measurements are meaningful at all — filamentous assays want spatial/hyphal
metrics; round assays want size and shape.

**`colony.contrast_vs_background`** — `high | moderate | low`

*Determined by:* **probe**, and this one needs an operational definition or it is
vibes. Proposed measure: Otsu's between-class variance ratio on `detect_mat`,
η = σ²_B / σ²_T ∈ [0, 1] — precisely the separability Otsu maximizes, so it is
principled rather than invented.

| Band | η | Status |
|---|---|---|
| `high` | ≥ 0.75 | **Provisional cut points.** The *measure* is principled; the *bands* need calibrating against real plates before they mean anything. |
| `moderate` | 0.45 – 0.75 | |
| `low` | < 0.45 | |

*Drives:* whether enhancement or detection is the bottleneck; whether
`HeavyOtsuPipeline` can work at all.

**`colony.separation`** — `well_separated | touching | confluent`

*Determined by:* **probe** — the fraction of detected objects sharing a boundary,
or expected count versus detected count on an arrayed plate.

*Drives:* watershed versus peak detection; how much refinement matters.

**`colony.pigmentation_informative`** — `bool`

*Determined by:* human, sometimes probe (channel separability).
*Drives:* `--detect-mode` (gray vs a colour channel or Lab), whether
`MeasureColor` earns its columns.

**`plate.format`** — `arrayed | unarrayed`, with `nrows` × `ncols`

*Determined by:* human or metadata.
*Drives:* `GridImage` vs `Image`, `--nrows/--ncols`, `GridSectionPipeline`, and —
critically — the expected counts that `QCScorer` scores against.

**`imaging.modality`** — `flatbed_scanner | camera | spimager | other`

*Determined by:* metadata or human.
*Drives:* `SpImagerPipeline`; and a camera implies vignetting, so
`FlattenIllumination` likely matters.

### 9.3.3 Failure modes, ranked by consequence

The reason this artifact exists. Note that the two worst failures are in the
fields the server **cannot** check and the human **must** supply — which is the
argument for asking rather than inferring.

| Wrong field | What happens | Caught by? |
|---|---|---|
| **`plate.nrows`/`ncols`** | `QCScorer` scores against wrong expected counts, so **the objective itself is wrong**. Tuning optimizes toward a false target and every arm's cost is meaningless — while looking perfectly healthy. | **Nothing.** The worst failure in the system. |
| **`morphology: round`** when filamentous | A peak detector finds one "colony" per dense region. Counts come out *plausible*, so QC may not flag it; the assay is silently under-counted. | Weakly — a size distribution with implausibly large objects |
| `morphology: filamentous` when round | Over-segmentation and hyphal metrics that measure noise | Probe object count far above expectation |
| `separation: well_separated` when touching | Merged colonies counted as one; size distribution skews high with a long tail | Probe: count below expected, size tail |
| `contrast: high` when low | Agent picks an Otsu prefab and tunes detector params, when the real fix was enhancement. Budget burned in the wrong subspace. | Probe: low object count, poor best-cost plateau |
| `pigmentation_informative` wrong | False negative loses signal; false positive adds noise columns | Low stakes either way |
| `imaging.modality` wrong | A worse prefab starting point | Low stakes; probe reveals it |

The pattern: **fields sourced `human` are high-stakes and uncheckable; fields
sourced `probe` are lower-stakes and self-correcting**, because the probe that
set them also surfaces the evidence that contradicts them. That asymmetry is why
`source` is mandatory and why a skill writing `source: "human"` for a value the
human never gave is the specific abuse to prevent.

### 9.3.4 The trait registry — the extension point

The skill owns a declarative registry, shipped beside it as **data, not prose**,
so traits and their routing rules can be added and audited without rewriting
procedure text.

`.claude/skills/phenotypic-assay-triage/traits.yaml`

```yaml
version: 3
traits:
  - key: organism.morphology
    values: [filamentous, round, mixed, unknown]
    determined_by: human           # ask; do not infer
    ask: "Is the organism filamentous (hyphal), round (yeast/bacterial), or mixed?"
    drives: [prefab_choice, measurement_family]
    failure: "round asserted for a filamentous organism yields plausible-looking
              counts from a peak detector — a silent under-count"
    stakes: critical

  - key: colony.contrast_vs_background
    values: [high, moderate, low]
    determined_by: probe
    measure:
      name: otsu_eta               # between-class variance ratio on detect_mat
      bands: {high: [0.75, 1.0], moderate: [0.45, 0.75], low: [0.0, 0.45]}
      calibration: provisional     # OQ-9.4
    drives: [enhancement_weight, detector_family]
    stakes: moderate

  # added in registry v3 — no server change, no schema_version bump
  - key: medium.opacity
    values: [clear, opaque, pigmented]
    determined_by: human
    ask: "Is the agar clear, opaque, or pigmented?"
    drives: [detect_mode, enhancement_weight]
    stakes: moderate

rules:                              # trait signals -> candidate prefabs (§9.4)
  - when: {organism.morphology: filamentous}
    prefer: [FilamentousFungiPipeline]
  - when: {organism.morphology: round, colony.separation: touching}
    prefer: [HeavyWatershedPipeline, HeavyRoundPeaksPipeline]
  - when: {colony.contrast_vs_background: high, colony.separation: well_separated}
    prefer: [HeavyOtsuPipeline]
```

Three properties this buys:

1. **Adding a trait is a data change.** A registry row is reviewable in a diff
   and testable in isolation. Contrast a markdown table, where "add a trait"
   means editing prose the agent may or may not honour.
2. **Rules are separable from procedure.** The `rules:` block is the §9.4
   decision table in machine-readable form, so it can be extended, reordered, or
   contradicted by a site-specific overlay without touching the skill's method.
3. **Recalibration is a data change too.** Moving the η bands (OQ-9.4) edits one
   `bands:` line, not code and not prose.

The **server never reads this file.** It is skill data, exactly as the
biological vocabulary is skill knowledge.

### 9.3.5 What the server validates — envelope only

**The server validates the shape of a trait. It never validates biology, and it
never enumerates traits.**

| Server checks | Server does **not** check |
|---|---|
| File exists, parses, has `schema_version` | Whether `medium.opacity` is a real trait |
| Each `traits.*` entry has `value` and `source` | What `filamentous` means |
| `source` ∈ `{human, probe, metadata, inferred}` | Whether `low` contrast is plausible here |
| `evidence` present when `source: "probe"`, and its `probe_ref` resolves | Whether the probe supports the claim |
| Unknown trait keys are **preserved verbatim** | Whether 8×12 matches the plate |

This keeps §9.1 intact under extension: the only closed enum the server enforces
is `source` — **provenance, not biology**. Every biological vocabulary lives in
the registry and can grow, and the server's validation logic is *finite and
final*: it is written once against the envelope and never grows as traits do.

### 9.3.6 Adding a trait later — worked

Suppose plate-lid glare turns out to drive enhancement choice.

| Step | Where | Server change? |
|---|---|---|
| Add `imaging.lid_glare: [none, mild, severe]` to `traits.yaml` | skill | **no** |
| Add a `rules:` row preferring a glare-tolerant enhancer | skill | **no** |
| Skill starts asking about it in triage | skill | **no** |
| Existing assays without the trait keep validating | — | **no** — traits are individually optional |
| A newer skill's assay read by an older server | — | **no** — unknown keys round-trip |

The only thing that would force a server change is a new **provenance** kind —
say `source: "instrument"` — and that is a genuinely structural addition worth a
`schema_version` bump.

### 9.3.7 Scope

**One assay profile per dataset**, at `<workspace>/assays/<dataset>.assay.json`.
A workspace routinely holds more than one organism; a single per-workspace
profile is correct until the day you add a second, and then it is silently wrong
with no signal. Campaign arms reference the profile by path, so a leaderboard
stays interpretable months later: *these arms were chosen because the organism
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

This table is the human-readable rendering of the `rules:` block in
`traits.yaml` (§9.3.4). It is **skill data, not server logic** — exactly the kind
of judgment an expert may override, and extended by adding a rule rather than by
editing prose.

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

## 9.7 Packaging and installation

**Skills are authored in-repo** at `.claude/skills/phenotypic-*/SKILL.md`, so
they version in lockstep with the tool contract. This matters concretely: a skill
that instructs the agent to call `pipeline_diff` is *wrong* until that tool
exists, and only co-versioning makes that a reviewable diff rather than a
runtime surprise.

In-repo authoring alone does not reach anyone who installs PhenoTypic elsewhere,
so the server ships an installer:

```bash
uv run phenotypic-mcp setup            # detect harnesses, install skills + register server
uv run phenotypic-mcp setup --check    # report what is installed and whether it is current
uv run phenotypic-mcp setup --harness claude-code   # target one explicitly
uv run phenotypic-mcp setup --uninstall
```

This follows the pattern of other agent tools that pair a skill with an MCP
server (graphify is the reference here — one `SKILL.md` in the harness's skills
directory, alongside a stdio MCP server the skill drives).

**Behaviour:**

1. **Detect** installed harnesses rather than assuming one.
2. **Install skills** into each harness's own convention.
3. **Register the MCP server** in that harness's config, pointing at the
   `phenotypic-mcp` entry point from the current environment (an absolute
   interpreter path, matching how `get_python_command(for_slurm=True)` resolves
   `sys.executable` rather than a bare `python`).
4. **Idempotent and versioned** — re-running upgrades in place; each installed
   skill carries the `phenotypic` version it shipped with, and `--check` reports
   drift instead of silently serving a stale skill against a newer tool surface.
5. **Never clobber user edits** — a modified skill file is reported and skipped
   unless `--force`.

Claude Code's conventions are `~/.claude/skills/<name>/SKILL.md` plus an
`mcpServers` entry, and are the ones this spec states with confidence. **The
exact paths and config shapes for other harnesses must be verified against each
harness's current documentation at implementation time** rather than assumed
here — getting one wrong installs a skill nothing loads, which fails silently.
The implementation plan should treat per-harness support as one task each, with
`--check` as the acceptance test.

## 9.8 Open questions

*(None outstanding.)*

**Resolved since first draft:**

- ~~OQ-9.1 skill packaging~~ → in-repo authoring plus a
  `phenotypic-mcp setup` installer (§9.7).
- ~~OQ-9.2 assay scope~~ → **per-dataset**, at
  `<workspace>/assays/<dataset>.assay.json` (§9.3.7).
- ~~OQ-9.3 assay validation~~ → **structure and provenance only**. The server
  checks shape, required keys, and `source ∈ {human, probe, metadata,
  inferred}`; it never validates biological values, so the domain vocabulary can
  grow without a server release (§9.3.4).
