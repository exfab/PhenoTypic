# Measurement Classification System — Conceptual Framework

- **Date:** 2026-06-18
- **Branch:** `metric-classification-system`
- **Status:** Design — conceptual framework (Sections 1–9) + Path 3 integration
  (Section 10), this document.
- **Scope of this document:** Define a vocabulary and a set of rules for classifying
  every column PhenoTypic can emit, so that a user can tell *how to apply a
  measurement* without learning its math or independently validating it (Sections
  1–9), and specify the **Path 3 (hybrid) integration** that lands this in the repo
  as a `schema/` intermediate-class spine + per-`Entry` overrides, a CI coverage
  gate, and a new `explanation/` docs page (Section 10). GUI surfacing remains a
  later, separate effort.

---

## 1. Motivation

PhenoTypic emits ~24 measurement-column enums plus metadata, QC, and derived-model
columns (see `phenotypic.schema`). Users repeatedly ask the same question in
different words: *"Which of these numbers can I report as a result, and which are
just inputs to a classifier?"*

The naive split — "shape/size/intensity/growth are phenotypes; texture/color are
for classification" — is a good instinct but breaks down on contact with practice:

1. The split is not about *which* measurement; it is about *how it is used*. In
   colony work, size, shape, edges, texture, opacity **and** color are routinely
   concatenated into a single feature vector for strain/species classification
   (Rattray et al., 2023; Huang, 2018; Gu et al., 2020). Circularity is
   interpretable yet is a classifier input.
2. "Unitless" is a useful *flag* but not the defining criterion: circularity and
   eccentricity are unitless yet interpretable.
3. Color is genuinely a middle ground, and it splits *internally* — opacity/lightness
   behaves like a biomass readout, whereas hue/pigmentation is both an interpretable
   trait and a classifier feature.
4. Roughly a third of the schema (metadata, QC, geometric bookkeeping, fit
   diagnostics) is not a "phenotype vs. classifier" question at all.

The framework below resolves these by separating two axes, splitting columns into
four coarse kinds first, and applying a three-tier spectrum only where it is
meaningful.

---

## 2. The engine: two orthogonal axes

Every measured column is located by two independent questions. Conflating them is
what made the original instinct feel slippery.

- **Axis 1 — Interpretability (semantic ↔ agnostic).** Does the number map to a
  *named biological thing* (a pigment, a diameter, biomass), or is it a *mathematical
  descriptor* with no one-to-one biological referent (a Haralick texture value, a
  chromaticity coordinate)? This is the radiomics "semantic vs. agnostic" distinction
  applied directly (Gillies et al., 2016; Alsoof et al., 2023): semantic features are
  "common language terms used to describe an ROI — for example, size, shape," while
  agnostic features are "mathematically calculated quantitative signifiers."
- **Axis 2 — Analytical role (readout ↔ fingerprint).** Do you use it *as a response
  variable* (measure it to quantify an effect: "this drug shrank colonies 30%"), or
  *as a feature* (pour it, with many others, into clustering/classification to
  separate groups)?

The three user-facing tiers (Section 4) are the **diagonal** of this 2×2. The
off-diagonal cells are where the "middle ground" lives.

```
                     READOUT (response var)        FINGERPRINT (feature vector)
SEMANTIC       │  Tier 1: Direct phenotype     │  Tier 2 (used as features):
(interpretable)│  size, growth, opacity        │  shape descriptors, pigmentation
               │                               │
AGNOSTIC       │  (rare / empty)               │  Tier 3: Discriminative feature
(mathematical) │                               │  texture, color moments / coords
```

`SEMANTIC × FINGERPRINT` is exactly the cell that is "interpretable but routinely
used as a classifier input" — which is why a descriptor like circularity feels like
both a phenotype and a feature. It is both.

---

## 3. Coarse pre-split: four kinds of column

The phenotype↔feature question is only meaningful for *measured signal*. About a
third of the schema is not signal, so columns are split into four kinds first.

| Kind | Families | What it is / why it is separate |
|---|---|---|
| **Identity / design factors** | `METADATA` + the 7 experimental-tag enums (`GENETIC_`, `SAMPLE_`, `PLATE_`, `CONDITION_`, `INCUBATION_`, `ACQUISITION_`, `EXPERIMENT_METADATA`); geometric bookkeeping `BBOX`, `OBJECT` | Not outcomes. These are the **independent variables** you analyze phenotypes *against* (group by replicate, regress on condition) plus provenance/locator columns. Reframing "metadata = nothing": metadata is the *X* in your model, not noise. |
| **Quality / trust** | `QUALITY_CHECK`, `QUALITY_COUNT`, `QUALITY_SE`, `QUALITY_MAD`, `QUALITY_ZMAX`, `QUALITY_TUKEY`, `QUALITY_ICC`, `CURATION`, `ErrorCategory` | Answer "can I believe this row/plate." They **gate** analysis; never a biological claim. |
| **Primary measurements** | `SHAPE`, `SIZE`, `INTENSITY`, `TEXTURE`, the 5 color enums (`ColorLab`, `ColorHSV`, `ColorXYZ`, `Colorxy`, `ColorComposition`), `RADIAL_EXPANSION`, `SYMMETRIC_ZONES` | The measured signal. **The three-tier spectrum (Section 4) applies only here.** |
| **Derived / model outputs** | `LOG_GROWTH_MODEL`, `LINEAR_SOFTPLUS_MODEL`, `DOUBLE_SOFTPLUS_MODEL`, `EDGE_CORRECTION`, `MODEL_METRICS`, `GRID`, `GRID_SPATIAL`, `GRID_SPREAD`, `GRID_LINREG_STATS` | Results computed *from* primary measurements (or from the array geometry). Not a fifth tier — a **resolution layer** (Section 5) that routes each column back into one of the four kinds / three tiers. |

---

## 4. The three-tier spectrum (Primary measurements only)

### Tier 1 — Direct phenotypes (readouts)

Semantic + readout. A real biological quantity, with units or a direct physical
referent; safe to interpret a *single value*; use as a response variable.

- **Size family** — `SIZE`, and the size-valued `SHAPE` members (`Area`,
  `ConvexArea`, `MeanRadius`/`MedianRadius`/`MaxRadius`, `MinFeretDiameter`/
  `MaxFeretDiameter`, `MajorAxisLength`/`MinorAxisLength`, `BboxArea`). *Basis:*
  colony area is a validated fitness proxy sensitive to ~1% differences
  (Baryshnikova et al., 2010; Fasanello et al., 2022; Bischof et al., 2016). The
  most defensible "real phenotype" in the library.
- **Intensity / opacity** — `INTENSITY`. *Basis:* opacity behaves as a biomass/
  density (OD-like) readout (Bär et al., 2020). **Caveat:** confounded by
  illumination and pigment, so it is a readout *with an asterisk* — Tier 1 by role,
  but condition-sensitive. Flag this explicitly wherever it surfaces.

*How to apply:* compare across conditions, dose-response, t-test/regression; report
with units; a single number is meaningful.

### Tier 2 — Descriptive traits (interpretable form; used both ways)

Semantic + (readout *or* fingerprint). A named morphological property a human would
describe — interpretable directionally, but usually unitless and comparative, so
typically anchored to a control and equally valuable as clustering input.

- **Shape form descriptors** — `Circularity`, `Solidity`, `Eccentricity`, `Extent`,
  `Compactness`, `Orientation`. *Basis:* each names a morphology phenotype (low
  solidity → spreading/invasive growth; high eccentricity → directional growth), yet
  all are unitless and routinely fed to classifiers (Rattray et al., 2023; Huang,
  2018). This is where "unitless ≠ phenotype" is *almost* right — they are unitless
  but interpretable, so unitlessness is a **flag, not the rule**.
- **Perceptual color / pigmentation** — `ColorLab` (L\* ≈ lightness/opacity, a\*/b\*
  ≈ pigmentation), `ColorHSV` (hue ≈ pigment). *Basis:* a pigment is a genuine trait,
  but the same values are classifier fuel — the literal middle ground, resolved here
  by *which color enum* (see Section 6).
- **Structured morphology** — `RADIAL_EXPANSION`, `SYMMETRIC_ZONES`. *Basis:*
  interpretable as zonation / radial-growth-pattern phenotypes, also discriminative.

*How to apply:* quantify a *specific named* change (spreading via solidity,
pigmentation via b\*); interpret the *direction* against a control; do not over-read
the absolute number; equally valid as clustering inputs.

### Tier 3 — Discriminative features (fingerprints)

Agnostic + fingerprint. Mathematically derived, no single biological referent,
unitless, meaningful only *in aggregate / relative to other samples*. Primary value
is classification, clustering, species/strain ID, biofilm typing, outlier detection.

- **Texture** — `TEXTURE` (Haralick-type). *Basis:* the canonical agnostic feature;
  the workhorse for species/strain/biofilm classification (Rattray et al., 2023;
  Gu et al., 2020).
- **Colorimetric / compositional color** — `ColorXYZ`, `Colorxy` (chromaticity
  coordinates), `ColorComposition`. *Basis:* device/colorimetric coordinates and
  color moments, not perceptual traits — used as fingerprint dimensions, not
  interpreted singly.

*How to apply:* never interpret one value biologically; standardize and use the
**whole block** together; supervised (needs labels) or unsupervised (clustering);
judge by discrimination performance, not by a per-feature biological story.

---

## 5. The resolution layer for derived / model outputs

Derived columns are **not** a fifth tier. Each is a function of upstream
measurements, and its placement follows the **transformation type**, not the source
column. The key evidence that "inherit from upstream" is too blunt: a single enum can
straddle. `LOG_GROWTH_MODEL` contains both biological kinetics (`r`, `K`, `µmax`,
`N0`) and fit machinery (`lambda`, `beta`, `Kmax`) — same model call, different homes.

So the classification unit is finer than the enum (sometimes **per member**), and each
derived column carries two orthogonal annotations from which its tier is **computed,
not asserted**:

- `derivation_type` ∈ { `parameterization`, `normalization`, `diagnostic`,
  `spatial_relational` } (primary columns are `raw`).
- `derives_from` — a provenance pointer to the upstream family/families, or to "the
  target measurement" for the parametric transforms.

### The four transformation archetypes

| Archetype | What it does | Referent of the output | Resolves to |
|---|---|---|---|
| **A. Parameterization** | Fit a dynamical/empirical model to a primary phenotype, extract its parameters | The *same biological process*, summarized | **Same tier as the input phenotype** (often sharper) |
| **B. Normalization** | De-confound a primary phenotype (remove an artifact) | The input quantity, cleaned | **The input's tier, exactly** (parametric in the target) |
| **C. Diagnostic** | Goodness-of-fit, optimizer state, regularization hyperparameters | The *model*, not the colony | **Quality kind** |
| **D. Spatial / relational** | Describe array layout or neighbor relations | The *grid geometry / inter-object relations* | **Quality**, or a measured **design-covariate** |

### Worked placement of every derived family

| Family · members | `derivation_type` | `derives_from` | Resolved placement | Why |
|---|---|---|---|---|
| `LOG_GROWTH_MODEL` · `r`, `K`, `N0`, `µmax` | A. parameterization | `SIZE` (size-vs-time) | **Tier 1** | kinetic parameters of the proliferation process |
| `LOG_GROWTH_MODEL` · `lambda`, `beta`, `Kmax` | C. diagnostic (hyperparam) | the fit | **Quality** | regularization / bounds of the optimizer, not biology |
| `LINEAR_SOFTPLUS_MODEL`, `DOUBLE_SOFTPLUS_MODEL` · fitted params | A. parameterization | the fitted phenotype-vs-x series | **Same tier as the fitted phenotype** | empirical-model parameters of a primary signal |
| `LINEAR_SOFTPLUS_MODEL`, `DOUBLE_SOFTPLUS_MODEL` · any fit knobs/bounds | C. diagnostic | the fit | **Quality** | fit machinery |
| `EDGE_CORRECTION` · `NewVal`, `Cap` | B. normalization | target measurement (e.g. `SIZE`) | **= target's tier** | de-confounded *same* quantity |
| `MODEL_METRICS` · `MAE`, `MSE`, `RMSE`, `R2`, `NumSamples`, `OptimizerLoss`, `OptimizerStatus` | C. diagnostic | the fit | **Quality** | describes fit quality, not the colony |
| `GRID_LINREG_STATS` · `ResidualError`, `RowM/B`, `ColM/B`, `PredRR`, `PredCC` | D. spatial diagnostic | centroids + grid | **Quality** | docstring: "evaluate grid alignment quality" |
| `GRID_SPATIAL` · `Left/Right/Above/UnderDistance` (+ neighbor labels) | D. relational | centroids | **Quality** or **design-covariate** | neighbor proximity = merge signal *and* a competition/edge-effect covariate |
| `GRID_SPREAD` · `ObjectSpread` | D. relational | centroids | **Quality** (lean) | over-segmentation flag; secondary "invasive spreading" signal |
| `GRID` · layout params | D. relational | array geometry | **Identity / design factor** | describes plate layout, not a colony trait |

Two consequences the coarse model cannot express:

1. **Per-member classification.** The `LOG_GROWTH_MODEL` split shows the unit of
   classification is sometimes the member, not the enum.
2. **A measured design-covariate home.** Neighbor distance (`GRID_SPATIAL`) is
   something you regress phenotypes *against* — like metadata, but *computed* rather
   than supplied. It belongs with Identity / design factors when used as a covariate.

### Why parameterization outputs match the upstream tier — the basis

Three legs, with a counter-example inside the same enum that proves the rule is
principled, not convenient:

1. **Mechanistic.** Colony size over time is the integral of the growth process; the
   logistic parameters `r`, `K`, `µmax` are the *generating parameters of that same
   trajectory*. Endpoint size and growth rate are two summaries of one underlying
   quantity — proliferation. A structure-preserving fit (extracting a rate constant
   from a monotone size–time curve) changes neither the **referent** (biomass/
   proliferation) nor the **role** (response variable), and *increases*
   interpretability — a rate constant is more mechanistic than a single endpoint.
   Same point on both axes → same tier.
2. **Empirical (strongest leg).** The field treats colony size and colony growth rate
   as **interchangeable proxies for one construct — fitness.** Baryshnikova et al.
   (2010) use "colony size as a proxy for fitness"; Fasanello et al. (2022) convert
   colony size → Malthusian growth rate → relative fitness; Lam et al. (2023) state
   growth is "evaluated based on their colony sizes or colony growth rates." When
   practitioners substitute A for B to measure the same thing, that *is* evidence
   they belong in the same tier.
3. **Operational (trust contract).** Tier 1's promise is "safe to interpret a single
   value as a biological claim." `µmax` passes the exact test `Area` passes — "the
   mutant grows 20% slower" is as defensible from one number as "the mutant is 20%
   smaller."

**The proof it is the transformation, not the model:** `R²`, `lambda`, and `beta`
come out of the *same fit call* but fail leg 3 — you cannot make a biological claim
from `R²=0.98`. Their referent is the model, so they route to Quality. If the rule
were "outputs of the growth model → Tier 1," they would be misfiled. Because the rule
is "parameterization of a Tier-1 process → Tier 1; diagnostics of the fit →
Quality," they sort correctly. That asymmetry is the entire basis.

---

## 6. Cross-cutting resolutions

- **Color is split, not one bucket.** Perceptual/interpretable spaces (`ColorLab`,
  `ColorHSV`) → Tier 2 (pigmentation phenotype + feature); colorimetric/compositional
  spaces (`ColorXYZ`, `Colorxy`, `ColorComposition`) → Tier 3 (fingerprint).
- **Intensity is a caveated Tier 1.** Tier 1 by role (biomass/opacity readout), but
  illumination- and pigment-confounded; surface the caveat wherever it appears.
- **Unitlessness is a flag, not a criterion.** It biases a measurement toward Tier 2/3
  but does not decide it (circularity is unitless yet interpretable).
- **The tiers double as a trust contract** (Section 7).

---

## 7. The trust contract (the user-facing payoff)

The tiers exist to let users apply measurements *without learning the math or
independently validating methods*. Each tier is a promise about what a single value
licenses:

- **Tier 1 — Direct phenotype:** pre-validated for direct biological claims
  (literature-backed). Safe to report a single number as a result.
- **Tier 2 — Descriptive trait:** interpret *directionally*, anchored to a control.
- **Tier 3 — Discriminative feature:** make **no** single-value biological claim; its
  job is discrimination, validated by how well groups separate — not by defending each
  feature's meaning.
- **Quality:** gates trust in a row/plate; never an outcome.
- **Identity / design factor:** the variables you analyze outcomes *against*.

This maps the original four named families cleanly: **size + growth → Tier 1; shape +
pigment-color → Tier 2; texture + colorimetric-color → Tier 3** — with color
deliberately split across Tiers 2/3 and intensity flagged as a caveated Tier 1.

---

## 8. Summary model (one picture)

```
Every column
   │
   ├─ Identity / design factor   → the X you analyze against (metadata, bbox, object, grid layout)
   ├─ Quality / trust            → gates analysis (QC_*, curation, errors, fit diagnostics)
   ├─ Primary measurement        → place on the spectrum:
   │      Tier 1  Direct phenotype     (semantic + readout)     size, intensity*
   │      Tier 2  Descriptive trait    (semantic + both)        shape descriptors, Lab/HSV color, radial/zones
   │      Tier 3  Discriminative feat. (agnostic + fingerprint) texture, XYZ/xy/composition color
   │
   └─ Derived / model output     → resolution layer (derivation_type, derives_from):
          A parameterization → same tier as input phenotype   (growth r/K/µmax → Tier 1)
          B normalization    → input's tier, exactly          (edge-corrected size → size's tier)
          C diagnostic       → Quality                        (R², RMSE, optimizer status, fit knobs)
          D spatial/relational → Quality or design-covariate  (grid residuals, neighbor distances)

   * intensity is Tier 1 by role but illumination/pigment-confounded — caveat always.
```

---

## 9. Scope, non-goals, and open questions

**In scope (this document):** the vocabulary, the two axes, the four kinds, the three
tiers, the derived resolution layer, the trust contract, the placement of every
existing `phenotypic.schema` family, and the **Path 3 integration** (Section 10).

**Out of scope (deferred to a later effort):**

- **GUI surfacing** — badging/grouping measurements by tier in the results viewer or
  builder. Depends on the `gui-checks` / `FEATURES.md` gate and should wait until the
  framework is stable. Path 3 deliberately makes the tiers `issubclass`-queryable so
  this is cheap to add later.
- Reconciling the pre-existing `Shape_Area` / `Size_Area` duplication (both Tier 1
  under this scheme; a separate cleanup if ever desired).
- The full Path 2 de-straddle (column renames, operation-layer moves) — explicitly
  *not* taken; see Section 10.5.

**Resolved design decisions (were open questions):**

1. **Storage granularity** → **hybrid.** Tier lives on the *intermediate base class*
   for tier-uniform enums; the two straddlers (`SHAPE`, `LOG_GROWTH_MODEL`) carry
   per-`Entry` overrides. A single `resolved_tier` accessor is the one read path.
2. **Human-authored vs. derived** → the *mechanical* tier/kind is auto-assigned;
   any *biological* caveat text (e.g. intensity's confound) inherits the `bio_desc`
   "human-authored only" guardrail.
3. **`derives_from` representation** → a **string token** (e.g. `"SIZE"`), never a
   typed class reference, to preserve the import-light `schema/` constraint.
4. **Design-covariate kind** → a **tag on Identity**, not a fifth first-class kind
   (keeps the coarse split to four).
5. **Stability contract** → a column's `resolved_tier` is part of the public docs/
   trust contract; re-tiering is a documented, reviewable change (the coverage gate
   makes silent drift impossible) but is **not** a column-name change, so it is not an
   API break.

---

## 10. Integration design (Path 3 — hybrid)

**Chosen approach:** an intermediate-class spine in `schema/` carries the tier/kind for
tier-uniform enums; the two straddlers keep their members but tag the minority via
per-`Entry` overrides. **No column is renamed**, so the operation layer, goldens, and
downstream code are untouched. This was selected over Path 2 (de-straddle) after
scoping showed `Shape_Area` alone appears as a literal in ~38 files and collides with
an existing `Size_Area` — a ~1-week migration vs. Path 3's ~1–2 days, low-risk.

### 10.1 The intermediate-class spine (`schema/`)

Member-less subclasses of `MeasurementInfo` (Enum subclassing is legal only when the
parent has no members), each carrying `kind()` / `tier()` classmethods (same idiom as
the existing `category()`) and the shared trust-contract docstring:

```
MeasurementInfo (str, Enum)        # naming via category() — unchanged
├── IdentityInfo                   # kind=identity
├── QualityInfo                    # kind=quality
├── PrimaryMeasure                 # kind=primary (no fixed tier)
│   ├── DirectPhenotype            # tier 1
│   ├── DescriptiveTrait           # tier 2
│   └── DiscriminativeFeature      # tier 3
└── DerivedMeasure                 # kind=derived; derivation_type()/derives_from()
```

### 10.2 Re-parenting (one line per enum)

- **Tier-uniform enums** re-parent to their leaf base: `SIZE(DirectPhenotype)`,
  `INTENSITY(DirectPhenotype)`, `ColorLab(DescriptiveTrait)`,
  `ColorHSV(DescriptiveTrait)`, `RADIAL_EXPANSION(DescriptiveTrait)`,
  `SYMMETRIC_ZONES(DescriptiveTrait)`, `TEXTURE(DiscriminativeFeature)`,
  `ColorXYZ(DiscriminativeFeature)`, `Colorxy(DiscriminativeFeature)`,
  `ColorComposition(DiscriminativeFeature)`, `METADATA`/experimental tags +
  `BBOX`/`OBJECT` → `IdentityInfo`, `QUALITY_*`/`CURATION`/`ErrorCategory` →
  `QualityInfo`, `MODEL_METRICS`/`GRID_LINREG_STATS`/`GRID_SPATIAL`/`GRID_SPREAD` →
  per Section 5 placement (mostly `QualityInfo`; `GRID` → `IdentityInfo`),
  `EDGE_CORRECTION`/`LINEAR_SOFTPLUS_MODEL`/`DOUBLE_SOFTPLUS_MODEL` →
  `DerivedMeasure`.
- **Straddlers** re-parent to the neutral parent and override the minority members:
  - `SHAPE(PrimaryMeasure)` — class default Tier 2; size-magnitude members (`Area`,
    `ConvexArea`, `MeanRadius`/`MedianRadius`/`MaxRadius`, `Min/MaxFeretDiameter`,
    `Major/MinorAxisLength`, `BboxArea`, `Perimeter`) get `Entry(..., tier=1)`.
  - `LOG_GROWTH_MODEL(DerivedMeasure)` — `r`/`K`/`µmax`/`N0` resolve to Tier 1;
    `lambda`/`beta`/`Kmax` get `Entry(..., derivation_type="diagnostic")` → Quality.

### 10.3 `Entry` + resolution

- Extend the `Entry` frozen dataclass with optional KW-only fields: `tier: int | None`,
  `derivation_type: str | None`, `derives_from: str | None`, validated in
  `__post_init__`.
- Add a single resolution accessor (e.g. `MeasurementInfo.resolved_tier` /
  `resolved_kind`): `Entry` override → else class `tier()`/`kind()` → else
  (`DerivedMeasure`) resolve from `derivation_type` + `derives_from`. One read path
  for all consumers.
- Optionally extend `rst_table()` with a conditional **Tier / Use** column (suppressed
  when empty, mirroring the existing Biology/Image columns).

### 10.4 Non-breaking verification (already scoped)

- `_cli_readme_generator.py` uses a **hardcoded** `measurer_to_info` map, not subclass
  discovery → unaffected by re-parenting.
- `util/_measurement_outputs.py` `_is_info_class` (`issubclass(x, MeasurementInfo)`)
  still holds transitively; `_measurement_descriptions()` iterates `schema.__all__`
  and member-less bases yield nothing. Add a "has members" guard if the bases are
  exported in `__all__`.
- No `__subclasses__()` reflection exists in the codebase.
- Enum **values** (header strings) are unchanged → serialization (`pipeline.json`,
  parquet headers), goldens, and `measure/` operations are untouched.

### 10.5 Blast radius summary

**Changed (additive / one-line):** `schema/_measurement_info.py` (or new
`schema/_tiers.py`) for the bases + `Entry` fields + `resolved_tier`; ~22 one-line
enum re-parents; 2 straddler files with per-member overrides; `schema/__init__.py`
exports; new CI **coverage gate** (mirroring `tests/unit/tune/test_annotation_coverage.py`)
+ resolution/straddler unit tests; new docs page (Section 10.6).

**Untouched:** no column renames (the ~38 `Shape_Area` literal files, `measure/`
operations, `tests/migration/_goldens`, `tests/unit/schema/_golden`, analysis/CLI/post
goldens, serialization) and no `Size_Area`/`Shape_Area` collision.

**Risks (all Low):** R1 tier bases iterated as column enums → member-less no-op / add
"has members" filter; R2 an intermediate base gains a member → test asserts emptiness;
R3 two-source tier for the 2 straddlers → single `resolved_tier` accessor + coverage
gate; R4 `derives_from` import coupling → string token.

### 10.6 New documentation

- **Conceptual page:** new `docs/source/explanation/measurement_classification_system.md`,
  wired into `explanation/index.rst` under the **"Measurement & Analysis"** caption
  (alongside `measurement_metrics_biological_meaning.md`, cross-linked both ways). It
  presents the two axes, four kinds, three tiers, the derived resolution layer, and —
  centrally — the **trust contract** (what a single value licenses per tier) so a user
  can apply measurements without the math.
- **Reference annotation:** the per-column **Tier / Use** badge renders into
  `measurements_ref/` automatically via the extended `rst_table()` — no per-page hand
  editing.

---

## References

Alsoof, D., McDonald, C. L., Durand, W. M., et al. (2023). Radiomics in spine
surgery. *International Journal of Spine Surgery, 17*(S1), S57–S64.
https://doi.org/10.14444/8501

Baryshnikova, A., Costanzo, M., Kim, Y., et al. (2010). Quantitative analysis of
fitness and genetic interactions in yeast on a genome scale. *Nature Methods, 7*(12),
1017–1024. https://doi.org/10.1038/nmeth.1534

Bär, J., Boumasmoud, M., Kouyos, R. D., et al. (2020). Efficient microbial colony
growth dynamics quantification with ColTapp, an automated image analysis application.
*Scientific Reports, 10*(1). https://doi.org/10.1038/s41598-020-72979-4

Bischof, L., Převorovský, M., Rallis, C., et al. (2016). Spotsizer: High-throughput
quantitative analysis of microbial growth. *BioTechniques, 61*(4), 191–201.
https://doi.org/10.2144/000114459

Fasanello, V. J., Liu, P., Longan, E., et al. (2022). Using colony size to measure
fitness in *Saccharomyces cerevisiae*. *PLOS ONE, 17*(10), e0271709.
https://doi.org/10.1371/journal.pone.0271709

Gillies, R. J., Kinahan, P. E., & Hricak, H. (2016). Radiomics: Images are more than
pictures, they are data. *Radiology, 278*(2), 563–577.
https://doi.org/10.1148/radiol.2015151169

Gu, P., Feng, Y. X., Zhu, L., et al. (2020). Unified classification of bacterial
colonies on different agar media based on hyperspectral imaging and machine learning.
*Molecules, 25*(8), 1797. https://doi.org/10.3390/molecules25081797

Huang, L. (2018). Novel neural network application for bacterial colony
classification. *Theoretical Biology and Medical Modelling, 15*(1).
https://doi.org/10.1186/s12976-018-0093-x

Lam, U. T.-F., Nguyen, T. T. T., Raechell, R., et al. (2023). A normalization protocol
reduces edge effect in high-throughput analyses of hydroxyurea hypersensitivity in
fission yeast. *Biomedicines, 11*(10), 2829. https://doi.org/10.3390/biomedicines11102829

Rattray, J. B., Lowhorn, R. J., Walden, R., et al. (2023). Machine learning
identification of *Pseudomonas aeruginosa* strains from colony image data. *PLOS
Computational Biology, 19*(12), e1011699. https://doi.org/10.1371/journal.pcbi.1011699
