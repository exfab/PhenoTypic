# Bio-semantic cluster ordering for the metadata front-block

**Date:** 2026-07-02
**Status:** Design — awaiting review
**Depends on:** PR #180 (`MetadataImage_` relocated to the trailing region)

## Context

The measurement sheets place user/experimental metadata columns at the front of the
frame. Today that front block is sorted by **REMBI provenance module**
(`STUDY → BIOSAMPLE → SPECIMEN_PREP → IMAGE_ACQUISITION → IMAGE_DATA`), implemented in
`MetadataAccessor.insert_metadata()` and mirrored by
`metadata_category_prefixes()`. REMBI order is a *standards* order, not how a bench
scientist reads a plate — and it has a concrete readability defect: `CULTURE_METADATA`
carries per-member `rembi_module=BIOSAMPLE` overrides on `Time` / `Timepoint` /
`FrameIndex`, so those members **scatter away** from `Temperature` / `Atmosphere`,
splitting one conceptual enum across two REMBI groups.

We want a **bio-semantic narrative** order a reader scans left-to-right:

```
① Identity  →  ② Strain  →  ③ Condition  →  ④ Design & provenance
```

This is a **reorder only** change (decided during brainstorming): no new per-member
taxonomy, no `cluster()` classmethod. The ordering unit is the **whole category
prefix** — each of the metadata enums lands in exactly one cluster.

## Goals

- Replace the REMBI-module front-block sort with the bio-semantic cluster order.
- Keep each metadata enum **contiguous** (fixes the `MetadataCulture_` scatter).
- Order members **within** a category by curated enum **definition order** (not the
  incidental alphabetical tiebreak used today).
- Make the cluster order a **single source of truth** so every metadata-ordering
  consumer (the `measurements.*` mirror, per-feature splits, README bucketing,
  dashboard scatter-axis priority) clusters identically.

## Non-goals

- **REMBI stays** the provenance axis: `by_module()`, `header_to_module()`, and the
  REMBI manifest (`_rembi_manifest.py`) are unchanged. Cluster order is a *separate,
  human-facing* axis layered on top — it does not touch REMBI classification.
- **No per-member cluster tags.** A member cannot be split out of its enum's cluster
  (e.g. `MetadataSample_`'s replicate fields ride with the rest of Sample in Identity).
- **`master_measurements.*`** is metadata-free — unaffected.
- **The `MetadataImage_` framework block** is already relocated to the trailing region
  (after measurements) by PR #180; this spec governs only the **front user-metadata
  block**. See "Interaction with #180".

## The cluster taxonomy

Category-granularity mapping (locked during brainstorming):

| # | Cluster | Categories, in order | Rationale |
|---|---------|---------------------|-----------|
| ① | **Identity** (lead) | `MetadataSample`, `MetadataPlate` | Who/where is this colony — sample id, replicate, clone, well, then plate/batch/layout |
| ② | **Strain** | `MetadataGenetic` | Genetic identity: organism, strain, genotype, allele, marker, ploidy |
| ③ | **Condition** | `MetadataCondition`, `MetadataCulture` | Chemical environment (media/treatment/compound), then temporal/physical culture (temperature/time/atmosphere) |
| ④ | **Design & provenance** | `MetadataExperiment`, `MetadataStudy`, `MetadataAcquisition` | Run structure (experiment/project/dataset), study bibliographics, imaging provenance |
| — | *Framework bookkeeping* | `MetadataImage` | Per-image provenance (UUID/ImageName/BitDepth). Relocated to **trailing** by #180; last in the prefix list for non-relocated surfaces |
| — | *Uncategorized* | generic `Metadata_` fallback | Unknown user tags; sorts after all known categories |

**Within-cluster category order** (also locked): Identity = Sample → Plate;
Condition = Condition → Culture; Design = Experiment → Study → Acquisition.

**Within-category member order:** curated enum **definition order** (e.g. Genetic:
Organism, Strain, Genotype, Background, Allele, Plasmid, SelectionMarker, MatingType,
Ploidy).

## Canonical order (the contract)

The full metadata front-block order becomes:

```
MetadataSample_*  →  MetadataPlate_*                          (① Identity)
  →  MetadataGenetic_*                                        (② Strain)
  →  MetadataCondition_*  →  MetadataCulture_*                (③ Condition)
  →  MetadataExperiment_*  →  MetadataStudy_*  →  MetadataAcquisition_*   (④ Design)
  →  <uncategorized Metadata_*>
```

with member columns inside each category in enum definition order. In the
`measurements.*` mirror this front block is followed by `[measurements]`, then the
`MetadataImage_*` block, then the `[Object_Label, Bbox_*, Grid_*]` info block (the last
two placements courtesy of #180):

```
[① Identity → ② Strain → ③ Condition → ④ Design → uncategorized]
   → [measurements]
   → [MetadataImage_*]
   → [Object_Label, Bbox_*, Grid_*]
```

## Design (mechanism)

All ordering derives from one new constant + one resolver in
`src/phenotypic/sdk_/_metadata_helpers.py`; every consumer reads from there.

### 1. Cluster-order constant (single source of truth)

```python
# Bio-semantic cluster order for the metadata front-block (category granularity).
# REMBI (header_to_module/by_module/manifest) remains a separate provenance axis.
_METADATA_CLUSTER_ORDER: tuple[str, ...] = (
    # ① Identity — who/where is this colony
    "MetadataSample", "MetadataPlate",
    # ② Strain — genetic identity
    "MetadataGenetic",
    # ③ Condition — chemical + temporal/physical environment
    "MetadataCondition", "MetadataCulture",
    # ④ Design & provenance
    "MetadataExperiment", "MetadataStudy", "MetadataAcquisition",
    # Framework per-image bookkeeping (relocated to trailing in measure() by #180)
    "MetadataImage",
)
```

The literal category strings duplicate each enum's `category()`; a **coverage-gate
test** (below) keeps them honest: adding a metadata enum without placing it here fails
CI. This mirrors the repo's existing coverage-gate pattern (tune annotations,
classification kinds).

### 2. `canonical_metadata_order()` — global header rank

A cached helper mapping every *known* metadata header to a global integer rank,
built by walking the enums in cluster order, each in definition order:

```python
@lru_cache(maxsize=1)
def canonical_metadata_order() -> dict[str, int]:
    """Global rank for every known metadata header, cluster-then-definition ordered."""
    rank = {m.category(): i for i, m in enumerate(_cluster_ordered_enums())}
    out: dict[str, int] = {}
    for enum in _cluster_ordered_enums():          # cluster order
        for i, member in enumerate(enum):          # definition order
            out[member.value] = rank[enum.category()] * 1000 + i
    return out
```

Consumers sort columns by `(canonical_metadata_order().get(header, BIG), str(header))`
— known headers cluster/definition-ordered, unknown/uncategorized fall to the end
alphabetically, stable.

### 3. `metadata_category_prefixes()` — reorder to cluster order

Reimplement its sort key from `(REMBI-rank, category)` to the index of each enum's
category in `_METADATA_CLUSTER_ORDER`. Return value shape (`tuple[str, ...]` of
`"<Category>_"` prefixes) is unchanged, so `is_metadata_header` and the dashboard
`SCATTER_PREFIX_PRIORITY` keep working — they simply pick up the new order.

### 4. `insert_metadata()` — sort by canonical order

Replace the REMBI `_rank` closure with a sort on
`canonical_metadata_order()`:

```python
rank = canonical_metadata_order()
def _key(item):
    header = ensure_metadata_prefix(item[0])
    return (rank.get(header, len(rank)), str(item[0]))
items = sorted(self._public_protected_metadata.items(), key=_key, reverse=True)
# insert() at loc=0 in reverse -> lowest rank lands leftmost
```

No other logic in `insert_metadata` changes; `MetadataImage_` is still inserted here
and relocated downstream by `measure()`'s `_order_measurement_columns` (from #180).

## Interaction with PR #180

`#180` splits `MetadataImage_` to the trailing region **inside `measure()`**, so in the
`measurements.*` mirror `MetadataImage_` is never actually part of the front block —
the cluster order governs the user-metadata prefix ahead of the measurements. For
**non-`measure()` surfaces** that read `metadata_category_prefixes()` directly
(dashboard scatter priority, README/split bucketing), `MetadataImage_` is listed **last**
in the cluster order, giving the same "framework bookkeeping trails user data" intent
without a relocation step.

## Files to modify

| File | Change |
|------|--------|
| `src/phenotypic/sdk_/_metadata_helpers.py` | Add `_METADATA_CLUSTER_ORDER`, `canonical_metadata_order()`, a `_cluster_ordered_enums()` helper; reorder `metadata_category_prefixes()` |
| `src/phenotypic/sdk_/__init__.py` | Export `canonical_metadata_order` |
| `src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py` | `insert_metadata()` sort key → canonical order; docstring note (REMBI → cluster) |
| `docs/source/explanation/metadata_namespace.md` | Add a "Column order (bio-semantic clusters)" section; clarify REMBI is a separate axis |
| `src/phenotypic/schema/CLAUDE.md` | One line: front-block order is cluster-driven, REMBI is provenance-only |

## Consumers affected (expected, benign)

- **Dashboard scatter-axis priority** (`SCATTER_PREFIX_PRIORITY`) — default axis
  candidate order shifts to cluster order (Sample/Plate first). Arguably better UX.
- **Per-feature splits + README bucketing** — metadata buckets render in cluster order.
- **`measurements.*` mirror** — front-block column order changes (the visible goal).

## Open questions

1. **Uncategorized vs `MetadataImage_` in non-relocated surfaces.** In the `measure()`
   mirror this is moot (#180 relocates Image; uncategorized ends the front block). But
   in split/README/scatter surfaces the prefix list ends `… Acquisition, Image`, so an
   *unknown* `Metadata_Foo` (rank = end) sorts **after** `MetadataImage_`. If we'd
   rather unknown user tags always outrank framework bookkeeping everywhere, we'd special
   -case `MetadataImage_` to sort after the generic fallback in those surfaces.
   **Recommendation:** accept `… Acquisition → Image → uncategorized` for the secondary
   surfaces (simpler, and those surfaces rarely carry unknown tags).
2. **External `--metadata` CSV join.** Joined columns flow through
   `_cli_output_manager.py` / `_cli_chunk_writer.py` / `post/`. Need to confirm the join
   either routes through `insert_metadata` / a canonical re-sort, or append columns that
   should then be re-ordered. If the join bypasses ordering, add a single
   canonical-order pass after the join. **To confirm during implementation** — may be a
   no-op if the mirror is rebuilt via `insert_metadata`.

## Verification

1. **Empirical column dump** — a plate with tags spanning all four clusters plus an
   unknown tag; assert the front block matches the canonical order and each enum is
   contiguous (esp. `MetadataCulture_` no longer split).
2. **New unit tests**
   - `measure()` mirror: cluster contiguity + narrative order (Identity → Strain →
     Condition → Design → uncategorized) ahead of measurements.
   - `canonical_metadata_order()`: within-category definition order; unknown → end.
   - **Coverage gate**: `set(_METADATA_CLUSTER_ORDER) == {e.category() for e in
     _metadata_enums()}` so a new metadata enum forces a cluster placement.
3. **Update pinned tests** — `tests/unit/sdk_/test_metadata_helpers.py`
   (`_expected_metadata_prefixes` → cluster order); any REMBI-order assertions in
   `tests/unit/sdk_/test_metadata_io.py` / `test_metadata_by_module.py`.
4. **Regression** — `test_cli_output_manager`, `test_measurement_outputs`, split tests,
   dashboard scatter tests.
5. `uv run ruff check --fix` + `uv run mypy src/phenotypic` on changed files.

## Ship

- New branch off `main` (e.g. `feat/metadata-cluster-order`), commit, push, PR → `main`.
- PR notes: reorder-only, REMBI untouched, single-source-of-truth, `MetadataCulture_`
  contiguity fix; deferred follow-up to #180.
