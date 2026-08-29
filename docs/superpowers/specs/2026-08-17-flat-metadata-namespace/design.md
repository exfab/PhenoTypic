# Flat metadata namespace with typed schema ownership

**Date:** 2026-08-17
**Status:** Approved
**Scope:** Public schema enum names, metadata column headers, ownership lookup,
metadata ordering, and compatibility migration

## Summary

PhenoTypic will store every metadata column in one concise physical namespace:

```text
Metadata_<Label>
```

The semantic metadata topic will no longer be encoded into the column prefix.
It will remain available through the concrete `MetadataInfo` enum that owns the
label. For example:

```text
GENETIC.STRAIN
    physical header     -> Metadata_Strain
    semantic owner      -> GENETIC
    classification kind -> identity
    REMBI module        -> BIOSAMPLE
```

Concrete metadata enums will use the same uppercase naming convention as the
other concrete `MeasurementInfo` enums. Their redundant `_METADATA` suffixes
will be removed. The current framework `METADATA` enum will become `IMAGE`, so
every concrete enum name describes its semantic topic and `MetadataInfo` remains
the unambiguous common base.

This is an internal schema and migration design. It requires no third-party
package or new runtime dependency.

## Locked decisions

The following compatibility and migration decisions are final for this design:

1. **Every recompile migrates automatically.** Local and SLURM recompiles run
   the same metadata-schema migration before consuming an existing per-image
   HDF. Migration is not restricted to a special command, opt-in flag, or one
   execution backend.

   > **Superseded (2026-08-18)** by
   > [2026-08-18-ome-zarr-image-store](../2026-08-18-ome-zarr-image-store/design.md).
   > Metadata-schema migration moves out of `--mode recompile` and into `--mode migrate`.
   > `recompile` stops rewriting legacy headers but keeps reading them — **decision #3
   > (permanent stored-data compatibility) is untouched**, so no existing output directory
   > breaks.
2. **HDF migration is copy-on-write.** Migration builds and validates a sibling
   replacement file, then atomically publishes it. The original HDF remains
   untouched if copying, normalization, validation, or publication fails.
3. **Stored-data compatibility is permanent.** Readers and recompile migration
   continue accepting historical per-topic metadata headers indefinitely. This
   promise is separate from the temporary Python-name compatibility window.
4. **Python aliases last one transition release.** The old public enum names
   resolve to their canonical topic names with a deprecation warning for one
   release, then are removed. Stored-header migration is not removed with them.
5. **External metadata is immutable.** Normalizing a caller-owned pandas or
   Polars frame always returns a new frame. Success and failure both leave the
   input frame, its column order, and its row order unchanged.
6. **HDF layout and metadata schema are versioned separately.** The existing HDF
   layout version continues to describe groups, datasets, and storage structure.
   A distinct metadata-schema marker records the header namespace version. A
   header-only migration advances the metadata marker without pretending that
   the HDF layout changed.
7. **The startup metadata snapshot is immutable provenance.** Full runs and
   recompile copy configured metadata bytes exactly to
   `deliverables/metadata.csv` before local processing or SLURM submission.
   Readers normalize its legacy headers in memory, but finalization and bundle
   migration never rewrite it. Canonical emission applies to generated
   scientific outputs, not this source snapshot.

## Context

`MeasurementInfo.__new__()` currently constructs a member's physical header as
`f"{cls.category()}_{entry.label}"`. Metadata enums therefore use different
`category()` values, such as `MetadataGenetic` and `MetadataCulture`, to preserve
their semantic topic in serialized column names. The result is explicit but
verbose headers such as `MetadataGenetic_Strain`.

This combines two independent concerns:

1. **Storage namespace:** whether a column is metadata.
2. **Semantic ownership:** whether known metadata is genetic, culture, sample,
   plate, acquisition, or another topic.

REMBI provenance is already a third independent concern. It is resolved through
`rembi_module()` and per-member overrides rather than solely from the physical
header. The new design makes that separation explicit for semantic ownership as
well.

The existing metadata vocabulary is compatible with a flat namespace. At the
time of this specification, the nine concrete metadata enums contain 74 labels
and all 74 are globally unique. The existing label-uniqueness test already
protects this property. Under this design, uniqueness becomes a load-bearing
schema invariant because duplicate labels would produce the same physical
header.

## Goals

- Emit concise, uniform metadata headers such as `Metadata_Strain`,
  `Metadata_Time`, and `Metadata_ImageName`.
- Preserve topic-aware, literal-free script checks through schema enum ownership.
- Rename concrete metadata enums so uppercase concrete names follow the existing
  `MeasurementInfo` convention without a redundant `_METADATA` suffix.
- Keep REMBI module classification independent from both physical headers and
  semantic topics.
- Preserve the curated bio-semantic metadata ordering even though every metadata
  enum now shares one `category()` string.
- Accept existing per-topic headers and old Python enum names during a defined
  compatibility period.
- Continue accepting arbitrary user metadata that is not in the recommended
  vocabulary.

## Non-goals

- Do not merge the metadata enums into one large `METADATA` enum.
- Do not infer semantic ownership by parsing label text.
- Do not introduce a second `MetadataTopic` enum merely to duplicate the
  concrete owner classes.
- Do not equate `IdentityInfo` with metadata. `BBOX`, `GRID`, and `OBJECT` remain
  identity/locator schemas but are not metadata.
- Do not change REMBI modules or move members between REMBI modules.
- Do not add biological claims, `bio_desc` content, or measurement definitions.
- Do not change measurement namespaces such as `Shape_*`, `Size_*`, or
  `Intensity_*`.

## Naming contract

### Concrete enum names

| Current public name | Canonical public name | Canonical category | Example header |
|---|---|---|---|
| `METADATA` | `IMAGE` | `Metadata` | `Metadata_ImageName` |
| `STUDY_METADATA` | `STUDY` | `Metadata` | `Metadata_Title` |
| `EXPERIMENT_METADATA` | `EXPERIMENT` | `Metadata` | `Metadata_Dataset` |
| `GENETIC_METADATA` | `GENETIC` | `Metadata` | `Metadata_Strain` |
| `SAMPLE_METADATA` | `SAMPLE` | `Metadata` | `Metadata_SampleID` |
| `CONDITION_METADATA` | `CONDITION` | `Metadata` | `Metadata_Media` |
| `CULTURE_METADATA` | `CULTURE` | `Metadata` | `Metadata_Time` |
| `PLATE_METADATA` | `PLATE` | `Metadata` | `Metadata_PlateID` |
| `ACQUISITION_METADATA` | `ACQUISITION` | `Metadata` | `Metadata_Instrument` |

`IMAGE` is chosen instead of retaining `METADATA` because `METADATA` would look
like the common namespace or base while actually owning only framework-populated
image bookkeeping. `IMAGE` gives that enum the same topic-noun shape as every
other concrete owner.

### Base class

A new memberless intermediate base separates metadata from other identity
schemas:

```python
class MetadataInfo(IdentityInfo):
    """Identity information stored in the shared Metadata namespace."""

    @classmethod
    def category(cls) -> str:
        return "Metadata"
```

All nine concrete metadata enums inherit `MetadataInfo` and remove their own
`category()` overrides. `MetadataInfo.kind()` continues to resolve to
`"identity"` through `IdentityInfo`.

Concrete enums remain uppercase because they are finite public
`MeasurementInfo` schemas, matching names such as `SHAPE`, `SIZE`, and
`INTENSITY`. `MetadataInfo` uses normal class casing because it is a memberless
classification base, matching `IdentityInfo` and `QualityInfo`.

## Orthogonal classification axes

The design defines four distinct properties:

| Axis | Question | Source of truth | Example for `GENETIC.STRAIN` |
|---|---|---|---|
| Namespace | Is this stored as metadata? | `MetadataInfo.category()` | `Metadata` |
| Owner | What semantic vocabulary owns it? | Concrete enum class | `GENETIC` |
| Kind | How should analysis classify it? | `IdentityInfo.kind()` | `identity` |
| Provenance | Where does REMBI place it? | Member/class REMBI resolver | `BIOSAMPLE` |

No axis is reconstructed by parsing another. In particular, the `Metadata_`
prefix establishes only the namespace, not the owner or REMBI module.

## Public script API

### Direct schema references

Users should obtain known column names from enum members:

```python
strain_column = str(GENETIC.STRAIN)
time_column = str(CULTURE.TIME)

result = frame.group_by(strain_column)
```

No direct column-name literal is required.

### Ownership predicates

The existing `MeasurementInfo.owns_header()` remains the preferred single-owner
predicate:

```python
if GENETIC.owns_header(candidate):
    handle_genetic_metadata(candidate)
```

`MetadataInfo` adds a cached finite header set for callers that explicitly need
set algebra or set membership:

```python
class MetadataInfo(IdentityInfo):
    @classmethod
    @cache
    def header_set(cls) -> frozenset[str]:
        return frozenset(member.value for member in cls)
```

Example:

```python
if candidate in CULTURE.header_set():
    handle_culture_metadata(candidate)

selected = set(frame.columns) & SAMPLE.header_set()
```

`header_set()` belongs on `MetadataInfo`, not general `MeasurementInfo`, because
some non-metadata schemas own dynamic runtime-qualified headers that cannot be
enumerated as a finite static set.

### Reverse lookup

The SDK exposes immutable, cached reverse lookup through four helpers:

```python
def metadata_member_for_header(header: str) -> MetadataInfo | None: ...

def metadata_owner_for_header(
    header: str,
) -> type[MetadataInfo] | None: ...

def metadata_member_for_label(label: str) -> MetadataInfo | None: ...

def metadata_owner_for_label(
    label: str,
) -> type[MetadataInfo] | None: ...
```

Examples:

```python
member = metadata_member_for_header(candidate)

if isinstance(member, GENETIC):
    handle_genetic_metadata(member)

if metadata_owner_for_header(candidate) is CULTURE:
    handle_culture_column(candidate)
```

Unknown `Metadata_*` headers return `None` from member and owner lookup while
remaining valid metadata according to `is_metadata_header()`.

### Registry construction

One registry derives all known metadata relationships from concrete
`MetadataInfo` subclasses:

```text
[concrete MetadataInfo enums]
               |
               v
      [validate unique labels]
               |
       +-------+--------+
       |                |
       v                v
[header -> member] [label -> member]
       |                |
       +-------+--------+
               |
               v
     [owner and REMBI queries]
```

Registry construction must raise `ValueError` on duplicate bare labels or
duplicate canonical headers. It must not use first-wins behavior. The existing
CI uniqueness test remains as a faster and more explanatory coverage gate.

## Input and output normalization

### Canonical emission

Newly written CSV, parquet, HDF metadata, measurement mirrors, generated
documentation, and examples use only `Metadata_<Label>`.

```text
GENETIC.STRAIN -> Metadata_Strain
CULTURE.TIME   -> Metadata_Time
IMAGE.UUID     -> Metadata_UUID
```

### User input

Metadata ingestion accepts three forms:

1. Bare known label, such as `Strain`.
2. Canonical flat header, such as `Metadata_Strain`.
3. Legacy per-topic header, such as `MetadataGenetic_Strain`.

The normalization flow is:

```text
[input name]
     |
     +-- bare known label ----------> [known member] --+
     |                                                  |
     +-- canonical Metadata_* --------------------------+--> [canonical header]
     |                                                  |
     +-- legacy per-topic header --> [legacy lookup] ---+
     |
     +-- bare unknown label ---------------------------> Metadata_<input>
```

Unknown labels remain supported as generic metadata. They have no schema member,
semantic owner, or declared REMBI module and therefore route to
`REMBI_MODULE.UNCATEGORIZED` on metadata surfaces.

## Ordering contract

The current ordering code uses per-topic category strings as semantic keys. That
cannot continue once every category equals `Metadata`. Ordering must instead use
the concrete owner classes:

```python
_METADATA_OWNER_ORDER: tuple[type[MetadataInfo], ...] = (
    SAMPLE,
    PLATE,
    GENETIC,
    CONDITION,
    CULTURE,
    EXPERIMENT,
    STUDY,
    ACQUISITION,
    IMAGE,
)
```

The canonical presentation order remains:

```text
[SAMPLE -> PLATE -> GENETIC -> CONDITION -> CULTURE
 -> EXPERIMENT -> STUDY -> ACQUISITION -> unknown metadata]
 -> [measurements]
 -> [IMAGE metadata]
 -> [Object_Label, Bbox_*, Grid_*]
```

Consequences:

- `_metadata_enums()` discovers `issubclass(obj, MetadataInfo)`, not a category
  prefix.
- `_cluster_ordered_enums()` ranks enum classes directly.
- `canonical_metadata_order()` uses the owner's position, not
  `enum.category()`.
- `metadata_category_prefixes()` is no longer an ordering API. Its only possible
  canonical result is `("Metadata_",)`, so it is deprecated in favor of owner
  and order helpers.
- `order_measurement_columns()` identifies framework image metadata through
  `IMAGE.owns_header(column)` rather than `column.startswith()`; otherwise the
  shared prefix would incorrectly move all metadata behind measurements.

## REMBI behavior

REMBI resolution continues to derive from the resolved schema member:

```python
member = metadata_member_for_header(header)
module = (
    member.resolved_rembi_module
    if member is not None
    else REMBI_MODULE.UNCATEGORIZED
)
```

This preserves member overrides such as culture time fields that resolve to a
different REMBI module from their containing enum. A flat header therefore does
not weaken REMBI classification for known fields.

`header_to_module()` emits canonical flat-header keys. Compatibility readers
normalize legacy headers before querying it.

## Compatibility and migration

### Python names

Canonical exports in `phenotypic.schema.__all__` are the new short names only.
For one compatibility release, module-level `__getattr__` resolves old names and
emits `DeprecationWarning`:

```text
METADATA             -> IMAGE
GENETIC_METADATA     -> GENETIC
SAMPLE_METADATA      -> SAMPLE
PLATE_METADATA       -> PLATE
CONDITION_METADATA   -> CONDITION
CULTURE_METADATA     -> CULTURE
ACQUISITION_METADATA -> ACQUISITION
EXPERIMENT_METADATA  -> EXPERIMENT
STUDY_METADATA       -> STUDY
```

Legacy names are omitted from `__all__`, generated documentation, schema
discovery, and GUI vocabulary. This prevents duplicate discovery while retaining
direct old imports during the compatibility window. Removal occurs in the next
declared breaking release. This is exactly one transition release. It does not
limit stored-data compatibility.

### Stored headers

A compatibility-only mapping records the old prefix owned by each new enum:

```python
_LEGACY_PREFIX_BY_OWNER = {
    IMAGE: "MetadataImage",
    STUDY: "MetadataStudy",
    EXPERIMENT: "MetadataExperiment",
    GENETIC: "MetadataGenetic",
    SAMPLE: "MetadataSample",
    CONDITION: "MetadataCondition",
    CULTURE: "MetadataCulture",
    PLATE: "MetadataPlate",
    ACQUISITION: "MetadataAcquisition",
}
```

The legacy-header map is generated from this table and current member labels.
Legacy strings remain isolated in the compatibility module and migration tests;
normal schema logic must not branch on them.

Readers normalize old headers to canonical headers permanently. Scientific
writers never emit old headers. The byte-exact `deliverables/metadata.csv`
startup snapshot is an input-provenance copy and is deliberately exempt from
header rewriting.

Every local or SLURM recompile also performs this normalization automatically on
existing per-image HDF inputs. Recompile migration is copy-on-write: create and
validate a sibling replacement, atomically publish it only after validation, and
leave the source file unchanged on failure. Both execution backends use the same
migration entry point and completion rules. Target discovery excludes the
startup metadata snapshot.

The migration records a dedicated metadata-schema version marker. It does not
reuse or increment the HDF layout version unless the physical group/dataset
layout independently changes.

Caller-provided pandas and Polars metadata frames are never renamed or coalesced
in place. Normalization returns a new frame of the same implementation and
preserves row order and surviving column order.

If a frame contains both legacy and canonical forms of the same known field:

- Coalesce them when each row has at most one non-null value or both non-null
  values compare equal.
- Raise a descriptive `ValueError` when any row has conflicting non-null values.
- Preserve the canonical column position and discard the legacy column only
  after conflict validation.

This rule avoids silent data loss while supporting partially migrated datasets.

### Deprecated helpers

`metadata_category_for_label()` is deprecated because all known labels would now
return the uninformative string `"Metadata"`. Callers migrate to
`metadata_owner_for_label()` or `metadata_member_for_label()`.

`metadata_category_prefixes()` is deprecated as a semantic ordering API. Callers
that only need namespace detection use `is_metadata_header()`. Callers that need
topic ordering use the owner registry.

## Alternatives considered

### 1. Display aliases only

**Flow:** bare/display label -> existing per-topic canonical header -> storage

**Pros:** No stored-data migration; exported headers remain self-describing.

**Cons:** Scripts and exported tables retain the verbosity this change is meant
to remove. Semantic topic remains coupled to physical spelling.

**Best for:** Input and GUI ergonomics without changing the data contract.

### 2. Flat namespace with enum ownership, selected

**Flow:** bare or legacy input -> member registry -> `Metadata_<Label>` -> storage

**Pros:** Concise output; enum-based, literal-free checks; no duplicate topic
taxonomy; REMBI remains independent.

**Cons:** Requires a breaking header migration and explicit owner-based ordering.

**Best for:** A concise public data contract that remains strongly scriptable.

### 3. Flat namespace plus `MetadataTopic`

**Flow:** input -> member registry -> separate topic enum -> flat storage

**Pros:** Provides small serializable topic tokens independent of Python classes.

**Cons:** Duplicates every concrete enum's identity and introduces another
coverage/mapping surface.

**Best for:** A future cross-language protocol that must serialize topic identity
separately from column names. It is not required now.

## Implementation sequence

### Phase 1: Schema hierarchy and names

1. Add and export `MetadataInfo`.
2. Rename the nine concrete classes to the canonical topic nouns.
3. Make them inherit `MetadataInfo` and remove per-class `category()` methods.
4. Add the transitional module `__getattr__` name resolver.
5. Update internal imports to canonical names.

### Phase 2: Registries and public predicates

1. Build fail-fast label-to-member and header-to-member registries.
2. Add the four member/owner lookup functions.
3. Add `MetadataInfo.header_set()`.
4. Replace string-prefix metadata discovery with `issubclass(..., MetadataInfo)`.
5. Deprecate category-based lookup APIs.

### Phase 3: Header migration

1. Add the isolated legacy-prefix map and generated legacy-header lookup.
2. Normalize CSV, parquet, HDF, metadata accessor, post-operation, and GUI input
   boundaries.
3. Implement dual-column coalescing and conflict rejection.
4. Change all writers to canonical flat headers.

### Phase 4: Ordering and downstream consumers

1. Replace `_METADATA_CLUSTER_ORDER` strings with `_METADATA_OWNER_ORDER` classes.
2. Rework canonical ordering to rank owners.
3. Detect the trailing framework block through `IMAGE.owns_header()`.
4. Update REMBI manifests, output splitting, analysis defaults, plotting defaults,
   GUI selectors, timeline logic, QC keys, and curation keys to use schema members
   or owner helpers.

### Phase 5: Documentation and cleanup

1. Update schema docs, examples, CLI help, generated README text, REMBI examples,
   and migration notes.
2. Replace internal direct header literals with schema references except in
   compatibility tests and migration fixtures.
3. Add a grep gate for legacy per-topic prefixes outside the compatibility module,
   migration tests, frozen historical specs, and explicit release notes.
4. Remove legacy Python-name resolution in the next declared breaking release.

## Verification plan

### Schema contracts

- Every concrete metadata enum subclasses `MetadataInfo`.
- Every concrete metadata enum returns `category() == "Metadata"`.
- Representative values render as `Metadata_Strain`, `Metadata_Time`, and
  `Metadata_ImageName`.
- All labels and canonical headers are globally unique.
- Duplicate registration raises at registry construction, independent of tests.
- Every metadata member still resolves to `resolved_kind == "identity"`.

### Ownership contracts

- `GENETIC.owns_header(str(GENETIC.STRAIN))` is true.
- The same header is absent from every other owner's `header_set()`.
- Header and label reverse lookup return the exact member and concrete owner.
- Unknown generic metadata passes `is_metadata_header()` but has no member or
  owner.

### Ordering contracts

- Known metadata follows `_METADATA_OWNER_ORDER` and enum definition order.
- Unknown generic metadata follows known front metadata.
- Measurement columns follow front metadata.
- Only `IMAGE`-owned metadata moves to the trailing framework block.
- `Object_Label`, `Bbox_*`, and `Grid_*` retain their existing final placement.

### Migration contracts

- Each legacy per-topic header normalizes to its canonical flat header.
- Bare known and canonical names are idempotent.
- Bare unknown labels receive the generic `Metadata_` prefix.
- Equal legacy/canonical duplicate columns coalesce.
- Complementary-null duplicate columns coalesce.
- Conflicting duplicate columns raise without mutating the input.
- Old output folders remain readable by curation, results, timeline, analysis,
  recompile, and REMBI-manifest paths.

### REMBI contracts

- Every known canonical header maps to its existing resolved REMBI module.
- Per-member overrides remain intact.
- Unknown metadata maps to `UNCATEGORIZED` on metadata manifests.
- Legacy headers resolve identically after normalization.

### Public API contracts

- Canonical short enum names are present in `phenotypic.schema.__all__`.
- Legacy names resolve with `DeprecationWarning` during the compatibility window.
- Legacy names do not appear in schema enumeration, generated docs, or GUI lists.
- Doctest and user examples use canonical enum members rather than direct header
  literals.

## Acceptance criteria

The change is complete when:

1. All new metadata outputs use only `Metadata_<Label>`.
2. Known metadata remains categorically queryable through its concrete enum with
   no direct string literal.
3. Set membership and reverse owner lookup are public, typed, and tested.
4. Bio-semantic ordering and the special trailing `IMAGE` block are unchanged in
   meaning.
5. REMBI output is unchanged except for canonical header spelling.
6. Existing per-topic outputs read safely, including explicit conflict handling.
7. Canonical concrete enum names have no `_METADATA` suffix.
8. No production consumer derives metadata topic from a category string or
   physical prefix.

## Recommendation

Adopt the flat namespace with concrete enum ownership. Use `MetadataInfo` only as
the shared namespace/classification base, concrete uppercase topic nouns as the
semantic types, and `Metadata_<Label>` as the sole canonical physical spelling.
Keep legacy names and headers at compatibility boundaries for one transition
release, then remove the Python aliases while retaining stored-header migration
for durable historical outputs.
