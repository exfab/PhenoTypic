# PhenoTypic MCP Server — §9 Separation of Responsibilities, and the Bundled Skills

Status: **draft, reviewed once, revised**
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

## 9.3 Upfront experiment characterization

Pipeline choice is driven by traits of the *experiment*, not by the image file.
Several of the decisive ones cannot be measured from a plate image at all —
whether the organism is filamentous or yeast-like is something you know and the
agent does not.

So Phase 1 (§8.1) opens with an **experiment-profile triage**, producing a durable artifact:

### 9.3.0 The extensibility rule

Traits that narrow pipeline construction will keep being discovered — medium
opacity, incubation timepoint, dimorphic switching, plate lid glare. A schema
with a fixed field set would need a **server release per trait**, which is
precisely the coupling §9.1 exists to prevent.

So the artifact is an **open map of traits under a uniform envelope**:

> **The server validates the shape of a trait, never the set of traits.**

Adding a trait requires no server change, no schema bump, and no code — only a
new row in the skill-owned registry (§9.3.4).

`<workspace>/profiles/<dataset>.experiment.json`

```json
{
  "schema_version": 1,
  "name": "exfab-fungal-2026-08",
  "dataset": {"path": "data/plates", "digest": "sha256:1a4c…", "n_images": 480},
  "traits": {
    "organism.morphology": {
      "value": "filamentous",
      "source": "human",
      "note": "Aspergillus spp., hyphal spread expected by 72 h"
    },
    "colony.contrast_vs_background": {
      "value": "low",
      "source": "human",
      "note": "hyphal edges wash out against the opaque medium",
      "evidence": {"measure": "michelson_percell_median", "measured": 0.041,
                   "n_cells": 96,
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
| `evidence` | when `source: "probe"`; **optional corroboration otherwise** | `{measure, measured, …}` — the named measure and its number, so the claim is auditable and the bands recalibratable. A `human` value may carry evidence that *disagrees* with it; that disagreement is visible rather than silently resolved. |
| `note` | no | Free text for a human reader |
| `confidence` | no | Optional `[0,1]`; reserved for traits that later warrant it |

**Unknown trait keys round-trip verbatim.** The server preserves any trait it
does not recognize, rather than dropping or rejecting it — the opposite of
pydantic's `extra="forbid"` used elsewhere in this codebase, and a deliberate
inversion. An older server reading a newer skill's profile must not silently
discard a trait; silent loss of a trait that drove a pipeline decision would
make the artifact actively misleading. Forward compatibility is a correctness
property here, not a convenience.

**Trait keys are dotted and namespaced** (`<group>.<trait>`), so a new group
(`medium.*`, `growth.*`, `stress.*`) needs no structural change either.

### 9.3.0.2 Multi-group experiments — the agent groups, the server filters

One experiment routinely holds several **species × media groups** needing
different pipelines and parameters. An earlier draft assumed homogeneity in three
places at once: one profile per dataset, one `pipeline_id` per deploy (§5.4), one
scorer per campaign (§8.2).

**The grouping strategy belongs to the agent, not the server.** This section once
specified `group_by` on the profile, per-group trait overrides, and a per-group
cost breakdown on `campaign_status`. All three were removed, because the
capability turns out to fall out of primitives that already exist:

- **A campaign carries exactly one `subset_id`** (§8.3), and `user_named` is a
  first-class selection method (§10.3). So one subset per group gives one
  campaign per group, and **that campaign's ordinary aggregate cost already *is*
  the group's cost.** The breakdown had no producer because it needed none.
- **§8.2's one-scorer invariant then holds trivially**, since each campaign spans
  one group. Comparing an *Aspergillus* arm against a *Neurospora* arm never was
  meaningful, so nothing is lost by not enabling it.
- **Per-group trait overrides were already inert.** §9.3.5 says the server never
  acts on a trait, so an override map was skill data the server carried without
  reading. The skill can hold it directly.

Keeping the strategy out of the server also keeps §9.3.5's invariant whole rather
than carving the first exception into it — and avoids three joins that had no
owner, since the profile's grouping, the selector's `group_key`, and the scorer's
expected-count CSV were three independent notions of "group" with nothing
reconciling them. Under one-subset-per-group there is one: **the subset is the
group.**

**One primitive stays on the server**, because without it the agent must
enumerate image paths per group — workable at 480 images, brittle at 50,000, and
silently stale the moment the dataset changes:

> **`group_filter`** — a `{column: value}` map on the `SubsetSelector` **ABC**,
> applied to the candidate image set *before* any selector runs.

On the ABC rather than on `MetadataGroupSubsetSelector` deliberately. Restricting
a candidate set is one idea, stated once, and it composes with every selector —
random-within-one-group works, and `MetadataGroupSubsetSelector`'s `allocation`
and `min_per_group` keep their stratification meaning *inside* the filtered set
instead of becoming inert in a second mode. This is mechanism, not strategy,
which is the line §9.1 already draws.

**The strategy — general-first — lives in the skill.** Try one pipeline across
the whole experiment; descend to per-group only where evidence requires it. Same
discipline as §9.4's prefab-first rule and for the same reason: specializing
before you have evidence buys complexity you cannot justify later. Note the
premise for descending is weaker than it looks — heterogeneous expected counts
are **already** expressible under one scorer, because `QCScorer` reads expected
counts as per-image rows of a metadata CSV rather than as scorer configuration.
The general-first pass is genuinely runnable across heterogeneous groups.

**The descent stays in lineage.** A campaign artifact carries an optional
`derived_from: {campaign_id, reason}`. Without it, N sibling per-group campaigns
have no recorded relationship — the fact that they came from one experiment and
one general-first failure would live only in the agent's context, and vanish at
the next compaction. One field keeps the descent reconstructible months later,
which is the same standard §8.7 holds the construction trail to.

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

*Determined by:* **the human in v1**, with a probe-measured number as
corroborating evidence. That split is not caution — it is what measurement
forced. See below.

*Measure:* **per-grid-cell Michelson contrast** at the Otsu split,
(μ_fg − μ_bg) / (μ_fg + μ_bg), reported as the median across cells.

```python
t   = skimage.filters.threshold_otsu(cell)     # per grid cell, not whole frame
fg  = cell >= t
mich = (cell[fg].mean() - cell[~fg].mean()) / (cell[fg].mean() + cell[~fg].mean())
```

### Why not Otsu's η — a measured refutation

An earlier draft specified η = σ²_B/σ²_T, "precisely the separability Otsu
maximizes, so it is principled rather than invented". The reasoning is correct
about what η *is* and wrong about what this trait *needs*. Measured on the three
bundled plate images (`contrast_trait_measure.py`, run against them):

| Claim | Result |
|---|---|
| **η is scale-invariant** | Reducing image contrast **20×** left η at `0.965` and Cohen's d at `10.435` — *numerically unchanged*. Both normalize by the very spread that reducing contrast shrinks. |
| **Michelson is not** | Same reduction: `0.2387 → 0.1443 → 0.0725 → 0.0364 → 0.0121`, linear in α. |
| **η has no dynamic range here** | Whole-frame `0.965–0.966` across 3 plates; per-cell p10–p90 = `0.945–0.963`, a span of **1.8% of the nominal [0,1] scale**. Three bands cannot be cut from that. |
| **Whole-frame Otsu measures the wrong thing** | It puts **46.1%** of pixels in "foreground" — far more than colonies occupy. The split is separating the *plate disc from the surround*, not colony from agar. |

That last row also explains something the review flagged as mere circularity:
`ReferenceFreeScorer._contrast` (`tune/score/_reference_free_scorer.py:377-409`)
needs `image.objmask` **because whole-image Otsu does not find colonies**. The
mask is not incidental to that implementation; it is what makes the number mean
anything. Measuring per grid cell — one colony and its local agar per cell —
removes the mask dependency without inheriting the plate-vs-surround artifact.

η would have shipped as a plausible-sounding trait that is *invariant to the
property it claims to measure*. Nothing downstream would have contradicted it:
every plate would have read `high`, and a genuinely low-contrast assay would too.

### Bands stay human-sourced (resolves OQ-9.4)

Every plate available in this repo is high-contrast — per-cell Michelson median
`0.233` (p10–p90 `0.225–0.239`). **One point does not calibrate a three-band
scale**, and inventing cut-points around a single anchor would repeat the η
mistake in a new coordinate system.

So in v1:

- `value` (`high | moderate | low`) is **`source: "human"`** — you know whether
  your plates are low-contrast, and §9.3.3 already establishes that
  human-sourced traits are the high-stakes uncheckable ones.
- `evidence` carries `{"measure": "michelson_percell_median", "measured": 0.233,
  "n_cells": 96}` as corroboration, so a human answer that contradicts the
  number is *visible* rather than silently overridden.
- `traits.yaml` records `calibration: uncalibrated` with the single known anchor,
  and the bands become derivable — as dataset-relative terciles or absolute
  cut-points — once a dataset spanning low contrast exists. That is a registry
  edit, not a server change (§9.3.4).

*Drives:* whether enhancement or detection is the bottleneck.

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

`.claude/skills/phenotypic-experiment-triage/traits.yaml`

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
    determined_by: human           # ask; the probe corroborates, it does not decide
    ask: "Are the colonies high, moderate, or low contrast against the agar?"
    measure:
      name: michelson_percell_median   # (mu_fg - mu_bg)/(mu_fg + mu_bg) per grid cell
      calibration: uncalibrated        # bands NOT derivable from one anchor
      anchors:
        - {dataset: "docs _dataset plates", value: 0.233, label: high, n_cells: 96}
      rejected:
        - {name: otsu_eta, why: "scale-invariant: unchanged across a 20x contrast
                                 reduction; per-cell span 1.8% of [0,1]"}
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
  - when: {organism.morphology: round, colony.separation: well_separated,
           colony.contrast_vs_background: high}
    prefer: [RoundPeaksPipeline]
  - when: {organism.morphology: round, colony.separation: well_separated}
    prefer: [RoundPeaksPipeline, HeavyRoundPeaksPipeline]
  - when: {organism.morphology: round, colony.separation: touching}
    prefer: [HeavyWatershedPipeline, HeavyRoundPeaksPipeline]
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

**And the server never *acts* on a trait.** No trait value gates any tool's
behaviour anywhere in the catalog — not scorer choice, not subset requirements,
not GPU routing, not operation filtering. `experiment_profile_get` is the
only tool that touches the artifact; `campaign_put` stores the `experiment_profile` reference
as a string without even checking the file resolves. **The experiment profile is provenance
for humans and input for skills; it is not an interlock.**

That is deliberate (§9.1), but it should be read alongside §9.3.3's failure
table, which rates a wrong `plate.nrows` as the worst failure in the system with
"**Nothing**" catching it. The entire safety story for a `critical`-stakes trait
like `organism.morphology` rests on the skill being loaded and followed. There is
no server-side backstop if it is not — which is a materially different risk
posture than "the server validates the shape of a trait" might suggest on its
own.

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
| Existing profiles without the trait keep validating | — | **no** — traits are individually optional |
| A newer skill's profile read by an older server | — | **no** — unknown keys round-trip |

The only thing that would force a server change is a new **provenance** kind —
say `source: "instrument"` — and that is a genuinely structural addition worth a
`schema_version` bump.

### 9.3.7 Scope

**One experiment profile per dataset**, at `<workspace>/profiles/<dataset>.experiment.json`.
A workspace routinely holds more than one organism; a single per-workspace
profile is correct until the day you add a second, and then it is silently wrong
with no signal. Campaign arms reference the profile by path, so a leaderboard
stays interpretable months later: *these arms were chosen because the organism
was filamentous and contrast was low.*

## 9.4 Prefab-first construction

**Rule: try the relevant prefab pipelines before authoring a new one.**

Seven ship today (`phenotypic.prefab`), and they are validated, documented
chains rather than examples. Their real intents:

| Prefab | Intent (from its docstring) | ops |
|---|---|---|
| `RoundPeaksPipeline` | Round colonies, lightweight peak-based detection | **2** |
| `FilamentousFungiPipeline` | Filamentous fungi, `DenoiseBlockMatch` + spatial measurements | **3** |
| `SpImagerPipeline` | SpImager-sourced images | **4** |
| `GridSectionPipeline` | Per-section processing on grid plates | **13** |
| `HeavyWatershedPipeline` | Watershed segmentation for **touching** colonies | **15** |
| `HeavyRoundPeaksPipeline` | Round colonies, peak detection with full refinement | **18** |
| `HeavyOtsuPipeline` | Multi-stage Otsu thresholding with refinement | **19** |

Op counts are measured, not estimated, and they are ordered here because
**cost order is not obvious from the names.** `SpImagerPipeline` is labelled
"light" but includes `DenoiseBlockMatch` (BM3D) — the very op whose addition is
what marks the `Heavy*` variants heavy — so a probe of it is not cheap despite
the label. An agent ordering candidates by expected cost should use this column,
not the adjective in the docstring.

### Experiment profile → candidate prefabs

This table is the human-readable rendering of the `rules:` block in
`traits.yaml` (§9.3.4). It is **skill data, not server logic** — exactly the kind
of judgment an expert may override, and extended by adding a rule rather than by
editing prose.

Rules are evaluated **most-specific-first**, and morphology dominates: it
constrains which detector family can work at all, whereas contrast and
separation only modulate how much enhancement and refinement are needed.

| Profile signal | First candidates |
|---|---|
| `morphology: filamentous` | `FilamentousFungiPipeline` (3) |
| `morphology: round` + `separation: well_separated` + `contrast: high` | `RoundPeaksPipeline` (2) — genuinely the cheapest that can work |
| `morphology: round` + `separation: well_separated` + `contrast: moderate\|low` | `RoundPeaksPipeline` (2), then `HeavyRoundPeaksPipeline` (18) |
| `morphology: round` + `separation: touching` | `HeavyWatershedPipeline` (15), then `HeavyRoundPeaksPipeline` (18) |
| `plate.format: arrayed` and dense | `GridSectionPipeline` (13) |
| `imaging.modality: spimager` | `SpImagerPipeline` (4) |
| `morphology: mixed` | Two arms — the filamentous and round candidates — rather than one compromise pipeline |
| `contrast: low` (modifier, not a rule) | Expect enhancement to matter more than detector choice; prefer the refinement-heavy variant of whichever family morphology selected |

`HeavyOtsuPipeline` (19 ops) is deliberately **not** a first candidate for any
signal. An earlier draft listed it under `contrast: high + well_separated` as
"cheapest that can work" — it is the *most* expensive of the seven, and that
same assay also matches the `RoundPeaksPipeline` row at 2 ops. Two overlapping
rows recommending pipelines 9× apart in cost is exactly the kind of error a
rules table makes visible and a prose paragraph hides. Reach for `HeavyOtsu`
when the cheaper family has been tried and failed, not first.

### The procedure

1. Pick candidate prefabs from the experiment profile — usually one or two, three at
   most.
2. **Materialize each**: `pipeline_put {name:"fil-prefab",
   from_prefab:"FilamentousFungiPipeline"}`. A bare class name from the catalog
   is not a `pipeline_id` (§2.2 requires a sandbox path), so this step is not
   optional — it is what makes a prefab probeable.
3. `pipeline_probe` each on the same 2 images. Compare object counts, size
   distributions, and per-op timing.
4. Tune the best prefab **before** authoring anything custom. A prefab whose
   parameters are wrong for your assay is not a failed prefab; `tune_space` on a
   prefab is the cheapest large improvement available.
5. Author a custom pipeline **only** when the best tuned prefab still fails a
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

### `phenotypic-experiment-triage`

**When:** at the start of any new dataset, before any pipeline exists.
**Produces:** `profiles/<dataset>.experiment.json` **and** `subsets/<name>.subset.json`.
**Tools:** `experiment_profile_get`, `subset_put`, `subset_get`,
`pipeline_probe`, `catalog_operations`.
**Procedure:**

1. Ask the human for morphology and expected colony count per plate — these are
   not measurable (§9.3.3).
2. Read imaging metadata for modality and bit depth.
3. **Establish the development subset** (§10): ask the human to name one, or
   sample with a recorded method and seed. Everything downstream runs on it.
4. Probe 2–4 subset images to measure contrast (`michelson_percell_median`,
   §9.3.2 — **not** Otsu's η, which §9.3.2 refutes as scale-invariant) and
   separation.
5. Write the profile with every trait carrying its `source` (the skill writes the
   file directly — there is no `_put` tool); `subset_put` with the
   measured `coverage` range.

**Hard rule it teaches:** never write `source: "human"` for something the human
did not say. A guess is `inferred`, and it must be stated as a guess wherever it
drives a decision.

### `phenotypic-pipeline-construction`

**When:** after triage, when choosing what to try.
**Procedure:** §9.4 — prefab-first, probe-compare, tune-before-authoring, and the
justification convention for custom pipelines.
**Tools:** `catalog_operations`, `catalog_operation_detail`, `pipeline_put`,
`pipeline_probe`, `pipeline_patch`.
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

**Recovery:** resuming with only a campaign id, call `campaign_status {detail:"artifact"}` first — it
returns the arms' `pipeline`/`tune_spec`/`study_id`, which `campaign_status` does
not carry. Then `campaign_status` for progress, and `workspace_lineage {id}` only
if you need the provenance chain (§8.3).
**Hard rule it teaches:** cost is in `[0, 1]` and **lower is better**. Report the
held-out `gap`, never the calibration score alone — an arm that won by
overfitting the split is not a winner.

### `phenotypic-deploy-and-verify`

**When:** a winner exists and a full dataset is to be processed.
**Tools:** `deploy_plan`, `deploy_start`, `deploy_status`, `workspace_cancel`.
**Procedure:**

1. `deploy_plan {scope:"full"}` — assembles the decision: winner provenance,
   subset score and held-out gap, measured full-dataset estimate in
   `node_hours`, coverage warnings, and the header sweep (§10.6.1). It mints the
   `plan_token`. **No human is in this step** — it draws and binds, it does not
   ask.
2. **Read the response yourself and bring the human the parts that decide it** —
   the node-hour figure, the held-out gap, and any coverage warning. This is
   preparation for the gate, not the gate.
3. `deploy_start {scope:"full", plan_token, human_response}` — **this call is the
   gate.** The server raises the elicitation from inside it, rendering the prompt
   from the token (§5.4), so the numbers the human sees are the server's, not
   ours. `human_response` is **required whenever you hold a `plan` token**
   (USER-22, §5.4 — a `campaign_arm` token carries consent forward from
   `campaign_approve`, so an unattended deploy arm does not fabricate one): it carries
   what the human actually said. On a host with elicitation the server's own
   prompt is authoritative and the response records
   `ack_source: "elicited"`; on a host without it, the field is the whole record
   of the decision and comes back `agent_asserted`. **Never invent it, and never
   treat a timeout or a decline as a yes** — neither is a person agreeing.
4. Poll `manifest.json`, not exit codes — without `--wait` the CLI exits 0 on
   submission.
5. Verify against the mirror (`measurements.*`), never the master, and report
   `QC_MetadataOnly` rows separately from detections.

**Hard rules it teaches:** deletion and overwrite are the human's job at a
shell — if the output directory is occupied, ask rather than routing around it.
And a coverage warning on the `deploy_plan {scope:"full"}` response is a reason
to *say something*, not a formality to pass through.

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
that instructs the agent to call a tool is *wrong* until that tool
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
- ~~OQ-9.2 profile scope~~ → **per-dataset**, at
  `<workspace>/profiles/<dataset>.experiment.json` (§9.3.7).
- ~~OQ-9.3 profile validation~~ → **structure and provenance only**. The server
  checks shape, required keys, and `source ∈ {human, probe, metadata,
  inferred}`; it never validates biological values, so the domain vocabulary can
  grow without a server release (§9.3.4).
