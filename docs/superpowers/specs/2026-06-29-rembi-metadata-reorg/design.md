# Design: Reorganize Image Metadata & Outputs Around REMBI

**Date:** 2026-06-29
**Status:** Approved (design) — pending spec review, then implementation plan
**Topic:** Make PhenoTypic's metadata model explicitly REMBI-shaped in code and output

---

## 1. Motivation

[REMBI — Recommended Metadata for Biological Images](https://pmc.ncbi.nlm.nih.gov/articles/PMC8606015/)
(Sarkans et al., 2021) is the community metadata recommendation that feeds the
EMBL-EBI **BioImage Archive**. It organizes bioimaging provenance into modules:

> **Study → Biosample (Specimen) → Specimen Preparation → Image Acquisition →
> Image Data → (Image Correlation) → Analyzed Data**

PhenoTypic's metadata is **already ~70% REMBI-shaped** — it just never says so.
The seven `_experimental_tags/` enums plus the framework `METADATA` enum map
almost one-to-one onto REMBI modules, but the mapping is implicit: nothing in the
code names the modules, no module grouping surfaces in the output, and the one
genuinely missing module (Study) has no home.

This work makes the latent REMBI structure **explicit in the code and in the run
output**, without changing what the measurement pipeline computes and without
making any biological/experimental metadata mandatory.

### REMBI → PhenoTypic mapping (the conceptual backbone)

| REMBI module | PhenoTypic source | Notes |
|---|---|---|
| **Study** | `EXPERIMENT_METADATA` + **new** `STUDY_METADATA` + study config | Study descriptors (title/authors/license/…) are the one real gap |
| **Biosample / Specimen** | `GENETIC_METADATA` + `SAMPLE_METADATA` | Two enums collapse into one module |
| **Specimen Preparation** | `CONDITION_METADATA` + `CULTURE_METADATA` + `PLATE_METADATA` | Three enums → one module |
| **Image Acquisition** | `ACQUISITION_METADATA` | Closest existing match |
| **Image Data** | framework `METADATA` (UUID, BitDepth, ImageType, format, file) | Always-present technical image props |
| **Image Correlation** | — | N/A for single-modality plate photos; intentionally omitted |
| **Analyzed Data** | the 32 measurement enums (`SHAPE`, `SIZE`, …) | Already tabular; surfaced as a feature catalog + table pointer |

---

## 2. Goals / Non-Goals

### Goals
1. Each metadata enum **declares its REMBI module** structurally, mirroring the
   existing `kind()`/`tier()` classification machinery (`_tiers.py` + `_classify`).
2. Fill the missing **Study** module with a new `STUDY_METADATA` recommended-tag
   enum.
3. **Always** emit one self-describing **REMBI manifest** per run into
   `deliverables/`, built by folding the per-colony metadata up to each module's
   natural scope.
4. Surface the module grouping on the read side: order `Metadata_*` columns by
   REMBI module and add an `image.metadata.by_module(...)` view.

### Non-Goals (explicit YAGNI)
- **No validation / no required fields.** Nothing fails a run for missing
  metadata. This is the line that separates this work from full BioImage Archive
  conformance.
- **No coupling to BioImage Archive's submission format or schema.** We emit a
  REMBI-*flavored* document, not a BIA-ingestible package. (Submission-ready
  export is a clearly-scoped future tier, out of scope here.)
- **No Image Correlation module** (multimodal/correlative imaging is out of
  domain).
- **No changes to measurement computation.**
- ~~No renames~~ → **superseded (§10):** the metadata *category prefixes* are
  being renamed (`Metadata_` → `Metadata<Topic>_`) so headers are self-describing.
  This is an intentional namespace migration; enum *class names* and member labels
  are still unchanged.

### Guarantee: everything stays opt-in
The new `STUDY_METADATA` is just more *recommended vocabulary*, exactly like
`Metadata_Strain` today — supplying none of it is legal, supplying arbitrary
extra columns is legal. The **only** content ever guaranteed in the manifest is
the technical image data the pipeline already generates itself (UUID, bit depth,
file list). Every biological/experimental field is sparse-by-default.

---

## 3. Architecture

Four changes plus two small read-side ripples: three additive (module
classification, the `STUDY_METADATA` enum, the manifest) and one migration (the
category-prefix rename, §10). No measurement logic is removed.

### 3.1 REMBI module classification (schema layer) — detailed

This reuses the exact machinery already in `schema/`: a member-less concept,
a classmethod default on `MeasurementInfo`, an optional per-member `Entry`
override, a single resolver, and a CI coverage gate. Seven concrete pieces (a–g):

#### (a) `REMBI_MODULE` — a closed value set

New file `src/phenotypic/schema/_rembi.py`. A `str, Enum` (closed value set per
the project's closed-set convention), whose **definition order is the canonical
module order** used for manifest section order and column ordering:

```python
# src/phenotypic/schema/_rembi.py
from enum import Enum

class REMBI_MODULE(str, Enum):
    """REMBI metadata modules (Sarkans et al. 2021). Definition order is canonical."""
    STUDY              = "Study"
    BIOSAMPLE          = "Biosample"
    SPECIMEN_PREP      = "SpecimenPreparation"
    IMAGE_ACQUISITION  = "ImageAcquisition"
    IMAGE_DATA         = "ImageData"
    ANALYZED_DATA      = "AnalyzedData"
    UNCATEGORIZED      = "Uncategorized"   # sink for unknown Metadata_* columns
```

`UNCATEGORIZED` is a real member (not `None`) so every routing path lands on a
typed value. It is the **only** module not part of REMBI proper; it exists solely
so unknown user tags are *kept and labeled* rather than dropped or crashing.
Re-exported from `schema/__init__.py` and added to `__all__`.

#### (b) classmethod default on the base

In `_measurement_info.py`, next to `kind()`/`tier()`:

```python
    @classmethod
    def rembi_module(cls) -> "REMBI_MODULE | None":
        """REMBI module for this enum, or None until declared by a subclass."""
        return None
```

(Imported lazily / under `TYPE_CHECKING` to preserve the package's import-light
load order — `_rembi.py` imports only stdlib + this base, same rule as `_tiers.py`.)

#### (c) optional per-member `Entry` override (for straddlers)

Add a KW_ONLY field to `Entry`, symmetric with `tier`:

```python
    rembi_module: "REMBI_MODULE | None" = None
```

validated in `Entry.__post_init__` (must be a `REMBI_MODULE` or `None`). **Why
keep it if every enum maps cleanly to one module?** One realistic straddler,
grounded in how REMBI actually models time (see below): in `CULTURE_METADATA`
(→ `SPECIMEN_PREP`), the temporal members `TIME` / `TIMEPOINT` / `FRAME_INDEX`
are, in a time-course, the *experimental variable being sampled* — which REMBI
places under **Biosample → `experimental_variables`** (REMBI literally lists
"Time" as an example value there), not specimen prep. **Decided:** these three
members carry the override; the rest of `CULTURE_METADATA` stays
`SPECIMEN_PREP`:

```python
class CULTURE_METADATA(IdentityInfo):
    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:  return REMBI_MODULE.SPECIMEN_PREP
    TEMPERATURE = Entry("Temperature", "...")                # → SPECIMEN_PREP (default)
    TIME       = Entry("Time", "...",  rembi_module=REMBI_MODULE.BIOSAMPLE)
    TIME_UNIT  = Entry("TimeUnit", "...", rembi_module=REMBI_MODULE.BIOSAMPLE)
    TIMEPOINT  = Entry("Timepoint", "...", rembi_module=REMBI_MODULE.BIOSAMPLE)
    FRAME_INDEX= Entry("FrameIndex", "...", rembi_module=REMBI_MODULE.BIOSAMPLE)
    DAY = Entry("Day", "..."); GENERATION = Entry("Generation", "...")   # → SPECIMEN_PREP
    ...
```

This is the design's one real exercise of the per-member override — validating the
mechanism. (`TIME_UNIT` follows `TIME` so the value and its unit stay in one
module.) The coverage gate still passes: every member resolves to a real module
(`SPECIMEN_PREP` or `BIOSAMPLE`).

> **REMBI's treatment of time (authoritative — EMBL-EBI REMBI Model Reference;
> Sarkans et al. 2021).** REMBI has **no dedicated time field**. Temporal info is
> modeled by *kind*: (1) the time-course sampling point → `experimental_variables`
> on **Biosample**; (2) growth/culture duration & conditions → `growth_protocol`
> free-text on **Specimen** (≈ `SPECIMEN_PREP`); (3) acquisition date/time →
> `image_acquisition_parameters` on **Image Acquisition** (our
> `ACQUISITION.IMAGING_DATE`, already correct). REMBI does **not** put time-course
> time under Image Acquisition. The choice was `SPECIMEN_PREP` (growth-elapsed
> reading) vs `BIOSAMPLE` (experimental-variable reading); **decided: `BIOSAMPLE`**,
> matching REMBI's experimental-variable model.

#### (d) the resolver — total, never None

A property mirroring `resolved_kind`, with a **single decisive fallback so it is
total** (defined for every member of every enum):

```python
    @property
    def resolved_rembi_module(self) -> "REMBI_MODULE":
        # 1. explicit per-member override
        if self.rembi_module_override is not None:
            return self.rembi_module_override
        # 2. enum-level declaration
        mod = type(self).rembi_module()
        if mod is not None:
            return mod
        # 3. fallback: anything undeclared is analyzed-data (measurement &
        #    locator columns — SHAPE/SIZE/OBJECT/GRID/BBOX — need no edits)
        return REMBI_MODULE.ANALYZED_DATA
```

(`rembi_module_override` is the per-member value stashed in `__new__` from
`Entry.rembi_module`, exactly like `tier_override` ← `Entry.tier`.)

Consequence — **what gets edited and what does not**:
- **Declare** a module on the 8 `Metadata_`-namespace enums (below). They never
  rely on the fallback.
- **Untouched:** the 32 measurement enums *and* the `OBJECT`/`GRID`/`BBOX`
  locator enums (also `IdentityInfo`, but in `Object_`/`Grid_`/`BBox_`
  categories, **not** `Metadata_`). They fall through to `ANALYZED_DATA` — which
  is correct: they are columns of the analyzed measurements table.

#### (e) the 9 module declarations

Add a `rembi_module()` override to each metadata enum — the same one-liner idiom
as the `category()` override already on every one of these classes:

```python
class GENETIC_METADATA(IdentityInfo):
    @classmethod
    def category(cls) -> str:        return "MetadataGenetic"   # §10 Scheme-B value
    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:  return REMBI_MODULE.BIOSAMPLE
    ...
```

(The `category()` strings shown throughout §3 are the post-§10 Scheme-B values,
e.g. `MetadataGenetic`, `MetadataImage` — not the legacy `"Metadata"`.)

| Enum | `rembi_module()` |
|---|---|
| `METADATA` (framework) | `IMAGE_DATA` |
| `EXPERIMENT_METADATA` | `STUDY` |
| `STUDY_METADATA` (new, §3.2) | `STUDY` |
| `GENETIC_METADATA` | `BIOSAMPLE` |
| `SAMPLE_METADATA` | `BIOSAMPLE` |
| `CONDITION_METADATA` | `SPECIMEN_PREP` |
| `CULTURE_METADATA` | `SPECIMEN_PREP` |
| `PLATE_METADATA` | `SPECIMEN_PREP` |
| `ACQUISITION_METADATA` | `IMAGE_ACQUISITION` |

> Decision over the base-class alternative (a `BiosampleModule(IdentityInfo)`
> hierarchy like `_tiers.py`): **rejected.** The tier bases exist because tiers
> *straddle* within an enum; REMBI modules are a flat 1:1 enum→module mapping, so
> a parallel base hierarchy adds layering for no compositional benefit. The
> classmethod override matches the `category()` idiom already in these files and
> is the least-surprising change.

#### (f) the column→module reverse index

One helper in `_rembi.py`, built once from the schema:

```python
def header_to_module() -> dict[str, REMBI_MODULE]:
    """Map every known column header (e.g. 'MetadataGenetic_Strain', 'Shape_Area')
    to its REMBI module, by walking schema.__all__ enums and reading each member's
    resolved_rembi_module."""
```

The manifest builder routes each *table column string* with this precedence:
1. **in the index** → its module (covers all `Metadata<Topic>_*` tags + all known
   measurement/locator columns);
2. **else `is_metadata_header(col)`** (starts with the `Metadata` family,
   incl. a generic `Metadata_<label>` from the join's unknown-label fallback) →
   `UNCATEGORIZED` (unknown user tag, kept);
3. **else** → `ANALYZED_DATA` (custom/category-prefixed measurement column).

So the resolver is the single source of truth for *known* headers; the two prefix
fallbacks only handle *strings not present in any enum*.

#### (g) coverage gate (mirrors `test_classification_coverage.py`)

New test asserting the invariant **"every metadata-namespace enum declares a
real REMBI module"** — so a future metadata enum can't silently fall through to
`ANALYZED_DATA`. The namespace test is `category().startswith("Metadata")` (the
9 metadata enums; measurement categories are `Shape`/`Size`/`Object`/… and do not
match):

```python
def test_metadata_enums_declare_a_rembi_module():
    for enum in _column_enums():                  # reuse the existing helper
        if not enum.category().startswith("Metadata"):
            continue
        for m in enum:
            assert m.resolved_rembi_module is not REMBI_MODULE.ANALYZED_DATA, \
                f"{enum.__name__}.{m.name} did not declare a REMBI module"
            assert m.resolved_rembi_module is not REMBI_MODULE.UNCATEGORIZED
```

Plus a totality test (`resolved_rembi_module` never raises / always a
`REMBI_MODULE`) across all column enums.

### 3.2 New `STUDY_METADATA` enum (fills the Study gap)

`src/phenotypic/schema/_experimental_tags/_study.py`, an `IdentityInfo` subclass
(so `kind="identity"`, `category()=="MetadataStudy"`, members render
`MetadataStudy_*`) with `rembi_module() == REMBI_MODULE.STUDY`. Re-exported from
`_experimental_tags/__init__.py` and `schema/__init__.py`.

Members **mirror REMBI's Study component field names exactly** (one set per run):
`Title, Description, PrivateUntilDate, Keywords, Author, License, Funding,
Publications, Links, Acknowledgements`. All recommended, none required.

**Flat-representation note.** REMBI's `authors`, `publications`, and `links` are
*structured lists* (author objects with name/affiliation/ORCID, etc.). PhenoTypic's
metadata model is flat scalar tags, so each renders as a single `MetadataStudy_*`
column whose value may be a delimited string
(e.g. `MetadataStudy_Author = "Doe, J.; Roe, A."`).
Full structured authors/affiliations/ORCID is a **BioImage-Archive-conformance**
concern, explicitly out of scope here (§2). The flat tags preserve the REMBI
*vocabulary* without the structured schema.

> Per the repo guardrail, agents author only `label` + `desc`. `bio_desc` stays
> `""` and `image` stays `None` for human authoring.

### 3.3 REMBI manifest builder + writer (output layer)

**New module** `src/phenotypic/sdk_/_rembi_manifest.py`:

- `build_rembi_manifest(measurements_df, per_image_metadata, study_config) -> dict`
  — pure function, fully unit-testable, no I/O.
- `write_rembi_manifest(output_dir, ...)` — serializes the dict to YAML
  (matching the pipeline's existing YAML serialization; `image_data.files` etc.
  are plain scalars/lists). **Best-effort, wrapped so it never blocks finalize**,
  exactly like the existing best-effort `metadata.csv` co-located copy.

**Hook:** call the writer from `finalize_post_master_outputs` in
`src/phenotypic/_cli/_cli_output_manager.py`, sourcing from the **measurements
mirror** (per the master-vs-mirror rule) plus the per-image framework metadata.

**Output location (decided — flat):** `deliverables/rembi.yaml`, at the
deliverables root **beside the input `metadata.csv`** source copy
(`DELIVERABLES_METADATA_CSV`), so the manifest and the raw metadata it summarizes
sit together. New constants/helper in `sdk_/_io_constants.py`:
`REMBI_MANIFEST_YAML = "rembi.yaml"` and `rembi_manifest_path(output_dir)` rooted
at `deliverables_dir()`. No `metadata/` subfolder.

**Serialization (decided):** YAML — even though the pipeline itself serializes to
JSON (`to_json`), the human-facing run manifest is more readable as YAML. Written
with the project's existing YAML dependency; the dict is plain scalars/lists, so
round-trips trivially.

### 3.4 Read-side ripples (smaller, stage last)

- **Column ordering:** `MetadataAccessor.insert_metadata` (and the export path)
  order metadata-family (`Metadata<Topic>_*`) columns by canonical REMBI module
  order, then alpha within a module. Display-only; no column is added or dropped.
- **`image.metadata.by_module(module)`** — read-only grouping view over the
  existing `image._metadata` private/protected/public dicts. Framework
  private/protected keys map to `IMAGE_DATA`; public tags map via the schema
  reverse index; unknown keys → `uncategorized`.

---

## 4. The transformation: per-colony metadata → REMBI manifest

Two-part mechanism: a **static column→module map** (from the schema enums) and a
**per-module aggregation rule** (collapses rows to each module's scope).

### Inputs (all already exist in the finalize path)
- **Measurements mirror** — one row per colony, carrying every metadata-family
  (`Metadata<Topic>_*`) column from the `--metadata` join.
- **Per-image framework metadata** — UUID, bit depth, format, file name (from
  each image's HDF / object metadata).
- **Study config** (optional) — study-level fields not naturally per-image.

### Step A — classify each column (static, no data)
Build a reverse index once from the schema: `MetadataGenetic_Strain →
GENETIC_METADATA → BIOSAMPLE`. Any metadata-family column (`is_metadata_header`)
**not** found in any enum is retained in an `uncategorized:` bucket — never dropped
(permissive/opt-in stance preserved). Non-metadata columns are AnalyzedData feature
columns (grouped by their category prefix, e.g. `Size_*`, `Shape_*`).

### Step B — collapse to each module's cardinality

| REMBI module | Source enums | Aggregation rule (rows → manifest) |
|---|---|---|
| **Study** | `STUDY` + `EXPERIMENT` + study config | One value per field via distinct-collapse; if a study field has >1 distinct value, emit a sorted list **and a soft warning** (signals a mislabeled run). |
| **Biosample** | `GENETIC` + `SAMPLE` | **Scalar-or-list**: one distinct value → scalar; many → sorted distinct list. |
| **SpecimenPreparation** | `CONDITION` + `CULTURE` + `PLATE` | Scalar-or-list. |
| **ImageAcquisition** | `ACQUISITION` | Scalar-or-list. |
| **ImageData** | framework `METADATA` | **Per-image**: `files:` list (one entry per image: name, UUID, bit depth, type) + dataset rollups (`n_images`, bit depths seen). |
| **AnalyzedData** | the 32 measurement enums present | Feature catalog (column names grouped by category) + pointer to the measurements file. No per-colony values inlined. |

The manifest is therefore a **deterministic fold** of the per-colony table up to
module scope: regroup + distinct-collapse + static label map. No new measurement.

### Study-field sourcing (decided)
Accept study fields from **both** sources: constant `Metadata_*` columns in the
`--metadata` CSV **and** an optional study file (`--study study.yaml`), with the
**study file taking precedence** on conflict. Absent study file → Study section
is built purely from constant CSV columns (often sparse). New CLI flag `--study`.

### Multi-value representation (decided)
**Scalar-or-list** across Biosample / SpecimenPrep / ImageAcquisition. Compact
and readable; per-image/per-plate linkage is not lost to the project because the
measurements table still carries it row-by-row.

### Worked micro-example
Mirror has `MetadataGenetic_Strain`, `MetadataCondition_Media`,
`MetadataCulture_Temperature`, `Size_Area`, `Shape_Circularity` over 2 plates /
3072 colonies:

```yaml
# deliverables/rembi.yaml   (beside deliverables/metadata.csv)
study:
  title: "1536-density deletion screen, 30C"   # from study.yaml or constant CSV col
  license: CC-BY-4.0
  experiment_id: EXP-2026-014
biosample:
  organism: Saccharomyces cerevisiae
  strain: [BY4741, by4742]            # >1 distinct → list
specimen_preparation:
  media: YPD                           # 1 distinct → scalar
  temperature: 30
image_acquisition:
  instrument: Epson V850
  resolution: 600
image_data:
  n_images: 12
  bit_depth: [8]
  files:
    - {name: plate01.tif, uuid: ..., bit_depth: 8, image_type: rgb}
    # ...
analyzed_data:
  measurements: deliverables/measurements.parquet
  features:
    Size:  [Area]
    Shape: [Circularity]
uncategorized:                         # unknown Metadata_* columns, kept not dropped
  some_custom_tag: [...]
```

---

## 5. Failure / edge-case behavior

- **No metadata supplied:** `image_data` is still fully populated (always
  exists); every other section is empty/omitted. Run still completes; manifest
  still writes.
- **Manifest write fails** (I/O, serialization): caught and logged; **finalize
  proceeds** (same contract as the best-effort `metadata.csv` copy). Never blocks
  the run.
- **Study field ambiguity** (>1 distinct value where one expected): emit list +
  soft warning, do not error.
- **Unknown `Metadata_*` columns:** preserved under `uncategorized`.
- **NaN / non-scalar cell values:** dropped from distinct sets; a field that is
  entirely NaN is omitted from its module.
- **`--mode process` layer-export runs** (no measurement/deliverables): manifest
  is **not** emitted (decided — consistent with skipping deliverables). The hook
  lives only in `finalize_post_master_outputs`, which process-mode never calls.

---

## 6. Testing strategy

- **Schema unit tests** (`tests/unit/schema/`): every metadata enum resolves to a
  `REMBI_MODULE`; `STUDY_METADATA` members classify; reverse index round-trips
  label→module; extend the existing classification coverage gate to assert each
  metadata-namespace enum (category starts with `Metadata`) declares a module; the
  9 renamed `category()` strings match Scheme B (§10.1).
- **`build_rembi_manifest` unit tests** (pure function): empty input → only
  `image_data`; scalar-vs-list collapse; study file overrides constant CSV column;
  unknown column → `uncategorized`; NaN handling; study-ambiguity warning.
- **Integration:** a small CLI run (`load_synth_yeast_plate()` fixtures) asserts
  `deliverables/rembi.yaml` exists, parses, and has the expected module sections;
  a zero-metadata run still produces a valid sparse manifest; a `--mode process`
  run produces **no** manifest.
- **Read-side:** `by_module(...)` grouping and metadata-family column ordering.
- **Docs doctests** for new public surface (`STUDY_METADATA`, `REMBI_MODULE`).

---

## 7. Suggested build order (for the implementation plan)

1. **Schema core** — `REMBI_MODULE` enum, `Entry.rembi_module` field,
   `rembi_module()` classmethod + `resolved_rembi_module` resolver/fallback,
   `schema/__init__.py` re-exports. (Self-contained; unit-tested in isolation.)
2. **Declare modules** on the 8 existing metadata enums + add `STUDY_METADATA`
   (10 REMBI fields) + the 4 `CULTURE` temporal `Entry(rembi_module=BIOSAMPLE)`
   overrides. Add the REMBI coverage gate.
3. **Category-prefix migration (§10)** — rename the 9 `category()` returns to
   Scheme B; introduce the centralized `is_metadata_header()` /
   `metadata_category_prefixes()` helper and route all ~6 hardcoded `Metadata_`
   sites through it; make the `--metadata` join schema-aware (label→header index,
   generic fallback). Grep gate + regression across post/GUI/dashboard/timeline.
   *(Largest blast radius — its own stage; lands before the manifest so the
   reverse index uses final headers.)*
4. **Manifest builder** (`_rembi_manifest.py`, pure) + its unit tests.
5. **Writer + finalize hook** + `_io_constants.py` paths + `--study` CLI flag +
   integration test (incl. `--mode process` emits no manifest).
6. **Read-side ripples** — column ordering + `by_module(...)`.
7. **Docs** — short explanation page (REMBI mapping table, §9) +
   Measurements/schema reference updates; update `schema/CLAUDE.md` and the 9 enum
   docstrings for the new categories; doctests.

Stages 1–5 deliver the user-visible outcome (self-describing module-prefixed
headers + always-on manifest); 6–7 are polish and can land incrementally. Stage 3
is the riskiest (namespace migration) and should get its own review gate.

---

## 8. Resolved decisions (was: open questions)
- **`--mode process` skips the manifest.** ✓ (hook only in finalize; §5).
- **YAML serialization.** ✓ (human-facing readability; pipeline stays JSON; §3.3).
- **Flat output:** `deliverables/rembi.yaml` beside the input `metadata.csv`. ✓ (§3.3).

- **Timepoint straddler → BIOSAMPLE.** ✓ `TIME`/`TIME_UNIT`/`TIMEPOINT`/
  `FRAME_INDEX` carry `Entry(rembi_module=BIOSAMPLE)`; rest of `CULTURE` stays
  `SPECIMEN_PREP` (§3.1c). Matches REMBI's time-as-experimental-variable model.
- **`STUDY_METADATA` mirrors REMBI Study exactly.** ✓ `Title, Description,
  PrivateUntilDate, Keywords, Author, License, Funding, Publications, Links,
  Acknowledgements` (§3.2), flattened to scalar tags.

---

## 9. Mapping at a glance (the hard cutover)

The conceptual end-state: every metadata value mapped to a REMBI module. (The
*implementation* reaches this additively — a `rembi_module()` tag per enum + 4
per-member overrides — no enum is split or merged.)

```
 CURRENT METADATA SOURCE                                 REMBI MODULE (destination)
 ═══════════════════════                                 ══════════════════════════
 STUDY_METADATA   (new) ──────────────┐
 EXPERIMENT_METADATA ─────────────────┴───────────────▶  ▓▓ STUDY ▓▓

 GENETIC_METADATA ────────────────────┐
 SAMPLE_METADATA ─────────────────────┼───────────────▶  ▓▓ BIOSAMPLE ▓▓
 CULTURE_METADATA · Time/TimeUnit/  │  ← per-member
                       Timepoint/Frame ┘    override

 CONDITION_METADATA ──────────────────┐
 PLATE_METADATA ──────────────────────┼───────────────▶  ▓▓ SPECIMEN_PREP ▓▓
 CULTURE_METADATA · Temp/Day/Gen/   │
                       Humidity/Atmos. ┘

 ACQUISITION_METADATA ────────────────────────────────▶  ▓▓ IMAGE_ACQUISITION ▓▓
 METADATA (framework, auto) ──────────────────────────▶  ▓▓ IMAGE_DATA ▓▓

 SHAPE/SIZE/COLOR/TEXTURE/…  ┐
 OBJECT/GRID/BBOX (locators) ┼ (fallback, 0 edits) ───▶  ▓▓ ANALYZED_DATA ▓▓
 QC/quality columns          ┘
 unknown Metadata_* (custom) ─────────────────────────▶  ▓▓ UNCATEGORIZED ▓▓ (kept)
```

Field-level end-state:

```
┌─ STUDY ─────────────────────────────────────────────────────────────────────┐
│ STUDY_METADATA   Title · Description · PrivateUntilDate · Keywords · Author   │
│  (mirrors REMBI) License · Funding · Publications · Links · Acknowledgements  │
│ EXPERIMENT_META  ExperimentID · Project · Dataset · Protocol · Notes          │
├─ BIOSAMPLE ─────────────────────────────────────────────────────────────────┤
│ GENETIC_META   Organism · Strain · Genotype · Background · Allele · Plasmid   │
│                SelectionMarker · MatingType · Ploidy                          │
│ SAMPLE_META    SampleID · BioReplicate · CondReplicate · TechReplicate ·      │
│                Clone · LibraryID · SourcePlate · SourceWell · Barcode·Control │
│ CULTURE_META ⟵override  Time · TimeUnit · Timepoint · FrameIndex           │
├─ SPECIMEN_PREP ─────────────────────────────────────────────────────────────┤
│ CONDITION_META  Media · CarbonSource · NitrogenSource · pH · Supplement ·     │
│                 Antibiotic · Inducer · Treatment · Compound · Concentration · │
│                 Dose · Stress                                                 │
│ CULTURE_META Temperature · Day · Generation · Humidity · Atmosphere        │
│ PLATE_META      PlateID · Batch · ArrayDensity · IncubatorPosition            │
├─ IMAGE_ACQUISITION ─────────────────────────────────────────────────────────┤
│ ACQUISITION_META  ImagingDate · Instrument · Experimenter · Resolution ·      │
│                   ExposureTime                                                │
├─ IMAGE_DATA  (framework-populated, always present) ─────────────────────────┤
│ METADATA  UUID · ImageName · ParentImageName · ParentUUID · ImageFormat ·    │
│           ImageType · BitDepth · FileSuffix                                   │
├─ ANALYZED_DATA  (no edits — fallback) ──────────────────────────────────────┤
│ Size_* · Shape_* · Intensity_* · Color*_* · Texture_* · Object_* · Grid_* ·  │
│ BBox_* · Quality_*                                                           │
└─ UNCATEGORIZED · any Metadata_<X> not declared by an enum (preserved) ───────┘
```

---

## 10. Category-prefix namespace migration (NEW scope)

**Goal (user request):** make each metadata column header self-describing by
renaming the shared `category() == "Metadata"` into per-enum, topic-indicative
categories — e.g. `METADATA.category() == "MetadataImage"`, so the header becomes
`MetadataImage_ImageName`.

### 10.1 Naming scheme — DECIDED: **Scheme B (per-enum topic)**

| Enum | **Scheme B — per-enum topic** (recommended) | **Scheme A — per-REMBI-module** |
|---|---|---|
| `METADATA` | `MetadataImage` | `MetadataImage` |
| `STUDY_METADATA` | `MetadataStudy` | `MetadataStudy` |
| `EXPERIMENT_METADATA` | `MetadataExperiment` | `MetadataStudy` |
| `GENETIC_METADATA` | `MetadataGenetic` | `MetadataBiosample` |
| `SAMPLE_METADATA` | `MetadataSample` | `MetadataBiosample` |
| `CONDITION_METADATA` | `MetadataCondition` | `MetadataSpecimenPrep` |
| `CULTURE_METADATA` | `MetadataCulture` | `MetadataSpecimenPrep` |
| `PLATE_METADATA` | `MetadataPlate` | `MetadataSpecimenPrep` |
| `ACQUISITION_METADATA` | `MetadataAcquisition` | `MetadataAcquisition` |

- **Scheme B (per-enum topic):** distinct prefix per enum; e.g.
  `MetadataGenetic_Strain`, `MetadataCulture_Time`. REMBI module stays a
  *separate* `rembi_module()` layer (manifest grouping). Granular headers;
  **the Time straddler is graceful** — `MetadataCulture_Time` carries an
  `Entry(rembi_module=BIOSAMPLE)` and the category makes no module claim.
- **Scheme A (per-module):** the header *is* the REMBI module; reverse index
  becomes pure prefix-parsing. But several enums collapse to one prefix (losing
  genetic-vs-sample in the header), **and the straddler contradicts itself** —
  `MetadataSpecimenPrep_Time` would sit in the `biosample:` manifest section. To
  stay consistent, Scheme A forces relocating `Time/TimeUnit/Timepoint/FrameIndex`
  into a `MetadataBiosample`-category enum (an enum split).

**✓ Adopted: Scheme B** — self-describing headers, no enum split, clean
straddler. The REMBI module remains explicit via `rembi_module()` + the manifest.
The 9 category strings are: `MetadataImage, MetadataStudy, MetadataExperiment,
MetadataGenetic, MetadataSample, MetadataCondition, MetadataCulture,
MetadataPlate, MetadataAcquisition`.

### 10.2 What changes in code

1. **`category()` overrides** on the 9 metadata enums (the one-liner each already
   has, returning the new string).
2. **Centralize the prefix predicate.** Today ~6 sites hardcode `"Metadata_"`
   (`post/_utils._PREFIX`, `MetadataAccessor.insert_metadata`, GUI
   `_METADATA_PREFIX` ×3, dashboard `SCATTER_PREFIX_PRIORITY`) plus
   `timeline_view` time-name detection. Replace with one helper in `schema` /
   `sdk_`: `is_metadata_header(col) -> bool` (matches **any** `Metadata`-family
   prefix) and `metadata_category_prefixes() -> tuple[str, ...]` (derived from the
   enums). All call sites route through it. Net simplification: the rule lives in
   one place instead of six string literals.
3. **`--metadata` join becomes schema-aware.** Today the join blanket-prefixes
   every user column to `Metadata_`. New behavior: look up each incoming column
   label in a `label → full_header` index built from the enums
   (`Strain → MetadataGenetic_Strain`); **unknown labels fall back to a generic
   `Metadata_<label>`** (still accepted — opt-in preserved). One index, built once.
4. **Reverse index for the manifest** keys off the same `label → (header, module)`
   map; no change in §4's routing logic, only the header strings.

### 10.3 Back-compat — DECIDED: **clean break**

Every new prefix still starts with `Metadata`, so the centralized predicate
(`startswith("Metadata")`) keeps recognizing metadata columns. **✓ Adopted: clean
break** — new prefixes only. Previously written `Metadata_*` parquets/CSVs still
*read* (they match the generic `Metadata…` predicate) but their columns won't
auto-map to a module (→ `UNCATEGORIZED`); no aliasing layer. Ship one migration
note in the changelog. No compatibility shim.

### 10.4 Impact / test surface
- Update `schema/CLAUDE.md` + the 9 enums' docstrings (they currently state
  `category() == "Metadata"`).
- Grep gate: no `"Metadata_"` string literals remain outside the centralized
  helper (one test asserts the helper is the only definition).
- Regression: `insert_metadata`, the metadata join (`post/_merge_metadata`,
  `_expand_metadata`), GUI results-viewer metadata grouping, dashboard scatter
  prefixes, timeline time-detection.

---

## 11. Rename execution scope (from full repo scan)

Scoped by a repo-wide scan + three parallel Explore passes. **~49 source files +
~128 test files** hardcode metadata strings, in **three workstreams**. (Occurrence
counts are approximate; file:line inventories live in the scan, not duplicated
here.)

### 11.1 New centralized helpers (one source of truth, kills the literals)

In `sdk_` (new `_metadata_helpers.py`, re-exported from `sdk_/__init__.py`):

```python
def metadata_category_prefixes() -> tuple[str, ...]:
    """('MetadataImage_', 'MetadataStudy_', 'MetadataExperiment_',
        'MetadataGenetic_', 'MetadataSample_', 'MetadataCondition_',
        'MetadataCulture_', 'MetadataPlate_', 'MetadataAcquisition_')
    — derived from the schema enums at import, in canonical REMBI order."""

def is_metadata_header(col: str) -> bool:
    """True if col starts with any metadata-family prefix (replaces every
    hardcoded startswith('Metadata_') / _PREFIX / _METADATA_PREFIX)."""

def metadata_category_for_label(label: str) -> str | None:
    """'Strain' -> 'MetadataGenetic'; None if no enum owns the label.
    Powers schema-aware prefixing in the post ops (§11.4)."""
```

### 11.2 Workstream 1 — bare-prefix predicates (8 code sites)

Mostly mechanical: swap the literal for `is_metadata_header()` /
`metadata_category_prefixes()`.

| Site | Fix |
|---|---|
| `gui/results_viewer/_output_root.py:547`, `_viewer_card.py:105`, `colony_view/_grid.py:98` (`_METADATA_PREFIX`) | `is_metadata_header()`; `_viewer_card` const is doc-only (delete) |
| `gui/results_viewer/timeline_view/_grid.py:136` | `is_metadata_header(name) and _is_time_like_name(name)` — regex unchanged (label still `Time`) |
| `_cli/_dashboard/_analysis_helpers.py:22` `SCATTER_PREFIX_PRIORITY` | expand `"Metadata_"` → `metadata_category_prefixes()` (priority bucket) |
| `_cli/_dashboard/_analysis/_scatter_plot.py:89` (JS) | inject the prefixes from Python into the page so JS isn't a second source of truth |
| `_core/.../_metadata_accessor.py:322` (`insert_metadata`) | schema-aware prefix (→ §11.4) |
| `post/_utils.py:3` (`_PREFIX`) | schema-aware prefix (→ §11.4) — the hard one |

### 11.3 Workstream 2 — `Metadata_ImageFile` → `Metadata_ImageName`

**Confirmed:** all 4 creation sites write the **stem only** (`_cli_chunk_writer.py:237`
`pq_path.stem`; `_cli_output_manager.py:910`, `_dashboard/_analysis_data.py:150`,
`_cli_recompile_worker.py:160` regex-strip the extension), so `Metadata_ImageFile`
≡ `Metadata_ImageName` semantically. Plan:

- **Creation (4 sites):** emit `Metadata_ImageName` (route through
  `str(METADATA.IMAGE_NAME)`); also thread **`Metadata_FileSuffix`**
  (`METADATA.SUFFIX`, already populated at import `_image_io_handler.py:525` but
  not currently carried into per-image parquets) so the full filename is
  reconstructible.
- **Consumption:** the curation/QC **key constants** `KEY_IMAGE_FILE` /
  `_META_IMAGE_FILE` / `_KEY_IMAGE_FILE` (`_curation_labels.py:36`,
  `_filtered_state.py:53`, `_error_tab/_data.py:44`, `_heatmap_tab/_figure.py:69`,
  `_qc_tab/review/_data.py:31`) → `str(METADATA.IMAGE_NAME)`; QC pairs
  (`analysis/abc_/_quality_check.py:313-353`), heatmap picker, colony/timeline
  grids, and the image-viewer JS all read the column dynamically — only the
  constant changes.
- **Suffix decision:** **reuse `METADATA.SUFFIX` ("FileSuffix"); do NOT add a new
  `ImageSuffix`.** Overlay/HDF path builders already hardcode `.png`/`.h5` (no
  change); only consumers wanting the literal `plate1.tif` reconstruct
  `ImageName + FileSuffix`.
- **Persisted-state migration (the sharp edge of clean-break):** the key column is
  written durably to `deliverables/qc/curation_labels.parquet`, the curated
  `measurements.parquet` mirror, `errors/*.parquet`, and `qc.duckdb`. A pure
  rename **orphans curation/QC state in pre-existing output folders.** A GUI
  fallback already exists (`_output_root.py:437-445`: accept `Metadata_ImageName`
  when `ImageFile` absent). **Decided (§11.7):** a one-line **rename-on-load shim**
  for the curation parquet (`Metadata_ImageFile` → `Metadata_ImageName` if the old
  column is present) so curation labels survive; everything else stays clean-break.

### 11.4 Workstream 3 — specific-literal de-stringly-typing

~130 occurrences across ~30 files → enum references. Clean mappings:

| Literal | → enum member |
|---|---|
| `Metadata_Time` | `CULTURE_METADATA.TIME` |
| `Metadata_Dataset` | `EXPERIMENT_METADATA.DATASET` |
| `Metadata_ImageName` | `METADATA.IMAGE_NAME` |
| `Metadata_Strain` | `GENETIC_METADATA.STRAIN` |
| `Metadata_SampleID` / `Metadata_ID` | `SAMPLE_METADATA.SAMPLE_ID` |
| `Metadata_Temp` | `CULTURE_METADATA.TEMPERATURE` |
| `Metadata_Well` | `SAMPLE_METADATA.SOURCE_WELL` |

- **Plain swaps** (~45%): class-field defaults, module constants, list elements →
  `str(ENUM.MEMBER)`.
- **Context-aware** (~15%): polars `.alias("Metadata_Dataset")` and Dash dict
  literals → `str(ENUM.MEMBER)` (same value, no behavior change).
- **The post-op prefixing redesign (hardest):** `post/_utils._ensure_prefix()`
  (used by `_merge_metadata`, `_append_string`, `_prepend_string`,
  `_expand_metadata`) blindly prepends `"Metadata_"` to a bare user label. Replace
  with `metadata_category_for_label(label)` → correct prefix when the label is
  known, **generic fallback** (`Metadata_<label>`, kept/uncategorized) when not.
  This preserves the "recommended vocabulary, not validator" contract.
- **Doctest / example-only literals:** leave genuinely-arbitrary example columns
  (`Metadata_Condition` in a `_merge_metadata` doctest, `Metadata_Flag` in a
  `_post_measurement` doctest) **as-is** — they intentionally demonstrate that
  arbitrary (non-vocabulary) columns are accepted. Convert only schema-backed
  literals.
- **"Needs schema home" → decided:** `analysis/qc/_icc.py` defaults its replicate
  column to a generic `Metadata_Replicate`, which has no schema member. **Repoint
  the ICC default to `SAMPLE_METADATA.BIO_REPLICATE`** (no new vocabulary; user can
  override). `SAMPLE_METADATA` keeps its `BIO_/COND_/TECH_REPLICATE` trio.

### 11.5 Test surface (~128 files)

Mechanical column-name updates, concentrated in `tests/unit/tune` (22),
`tests/unit/analysis` (16), `tests/integration/gui` (13),
`tests/unit/gui/results_viewer` (11), `tests/unit/cli` (8), `tests/e2e/gui` (8).
Plus new `tests/migration/` cases for the curation rename-on-load shim.

### 11.6 Revised staging (expands the old single "stage 3")

- **3a — helpers + predicates:** add the three `sdk_` helpers; route the 8
  bare-prefix sites through them; inject prefixes into the dashboard JS.
- **3b — ImageFile consolidation:** thread `FileSuffix`; switch creation +
  `KEY_IMAGE_FILE` constants to `ImageName`; add the curation rename-on-load shim.
- **3c — de-stringly-typing:** replace specific literals with enum refs; redesign
  post-op prefixing; resolve the `Replicate` default.
- **3d — tests + docs:** update the ~128 test files; grep gate (no stray
  `"Metadata_"` literals outside the helper); migration tests; docstrings.

### 11.7 Decisions — all resolved
1. **Post-op unknown-label prefixing → schema-aware with generic fallback.** ✓
   Known label → its category prefix; unknown → generic `Metadata_<label>` (kept,
   uncategorized). Preserves "recommended vocabulary, not validator."
2. **Curation persisted-state → rename-on-load shim.** ✓ Reading
   `curation_labels.parquet` renames `Metadata_ImageFile` → `Metadata_ImageName`
   when the legacy column is present; rest of the run stays clean-break.
3. **Suffix naming → keep `FileSuffix`.** ✓ Reuse `METADATA.SUFFIX` as-is; thread
   it into aggregation. No new `ImageSuffix`.
4. **ICC `Replicate` default → repoint to `SAMPLE_METADATA.BIO_REPLICATE`.** ✓ No
   new vocabulary; user can still override.
