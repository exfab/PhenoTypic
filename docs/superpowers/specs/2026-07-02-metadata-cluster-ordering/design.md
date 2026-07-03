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
and relocated downstream by the shared `order_measurement_columns` helper (§5).

### 5. Shared column-order helper + the polars mirror pass

There are **two** ordering surfaces, and today only the first is covered:

- **Per-image parquets** — the pandas `measure()` path
  (`_cli_process_single.py` → `apply_and_measure(..., apply_post=False)`,
  `include_metadata=True`). Ordered by #180's `_order_measurement_columns`; this spec
  makes its front-metadata block cluster-ordered via `insert_metadata` (§4).
- **The mirror `measurements.*` + per-feature splits + `analysis.*`** — built
  separately in `finalize_post_master_outputs` (`_cli_output_manager.py`) on a **polars**
  frame:
  `master_df` (clean) → `join_metadata(master_df, metadata_csv)` → `_apply_post_to_master`
  → `_seed_measurements`. **`join_metadata` puts the external CSV's columns on the left in
  CSV column order** (`metadata_df.join(df, on=common, how="inner")`, line 123), so joined
  metadata lands front-but-unordered and post-added metadata columns land wherever post
  put them. This path never sees the canonical order.

**Fix — one shared ordering function, two call sites.** Extract the partition-and-order
logic (currently inline in #180's pandas-only `_order_measurement_columns`) into a
framework-agnostic helper in `_metadata_helpers.py` that operates purely on column-name
strings:

```python
def order_measurement_columns(columns: Sequence[str]) -> list[str]:
    """Canonical column order: [front metadata] -> [measurements]
    -> [MetadataImage_] -> [Object_Label, Bbox_*, Grid_*].

    Front metadata is cluster/definition ordered via canonical_metadata_order();
    info-block columns are detected by name (OBJECT.LABEL / Bbox_ / Grid_)."""
    rank = canonical_metadata_order()
    image_prefix = f"{METADATA.category()}_"
    front, image_meta, info, meas = [], [], [], []
    for c in columns:
        if c.startswith(image_prefix):        image_meta.append(c)
        elif is_metadata_header(c):           front.append(c)
        elif c == OBJECT.LABEL or c.startswith("Bbox_") or c.startswith("Grid_"):
                                              info.append(c)
        else:                                 meas.append(c)
    front.sort(key=lambda c: (rank.get(c, len(rank)), str(c)))
    return front + meas + image_meta + info
```

Then:
- **pandas path** — `ImagePipelineCore._order_measurement_columns` becomes a thin wrapper:
  `return df[order_measurement_columns(list(df.columns))]`. (#180's `info_cols` param is
  dropped — info columns are now detected by name, matching the test classifier. This
  removes the only reason `_order_measurement_columns` needed the pre-merge `info_cols`
  capture.)
- **polars path** — in `finalize_post_master_outputs`, after `_apply_post_to_master`,
  before `_seed_measurements`:
  ```python
  post_df = _apply_post_to_master(working_df, pipeline)
  post_df = post_df.select(order_measurement_columns(post_df.columns))  # NEW
  _seed_measurements(output_dir, post_df)
  ```
  `post_df.columns` is a plain `list[str]`, so the same helper drives the reorder; the
  mirror, per-feature splits, and `analysis.*` (all derived from `post_df`) inherit the
  canonical cluster order. The **clean master on disk is untouched** — only the in-memory
  working/post frame is reordered.

This makes the cluster order a single source of truth across *both* the pandas per-image
parquets and the polars mirror, and folds #180's relocation into the same helper.

**Info-block detection is collision-free.** Only the `GRID` and `BBOX` schema enums emit
`Grid_*` / `Bbox_*` columns (categories `Grid` / `Bbox`). The neighboring quality enums
`GRID_LINREG_STATS` and `GRID_SPREAD` render as `GridLinReg_*` / `GridSpread_*`, which do
**not** match the `"Grid_"` prefix (the underscore guards them), so detecting the info
block by name is equivalent to #180's explicit `info_cols` — verified against the current
schema. A guard test (below) pins this so a future enum named `Grid…`/`Bbox…` can't
silently leak into the info block.

## Interaction with PR #180

`#180` splits `MetadataImage_` to the trailing region **inside `measure()`** (the pandas
path), which governs the **per-image parquets**. The `measurements.*` **mirror** is built
on the separate polars path and picks up the identical placement from the shared
`order_measurement_columns` helper (Design §5) — so both surfaces agree without
duplicating the rule. This spec **folds #180's pandas-only `_order_measurement_columns`
into that shared helper**: #180's method becomes a thin wrapper, the polars mirror gains
the same ordering, and the `MetadataImage_`-after-measurements placement plus the cluster
front-block order live in exactly one place.

For **prefix-list surfaces** that read `metadata_category_prefixes()` directly (dashboard
scatter priority, README/split bucketing), `MetadataImage_` is listed **last** in the
cluster order, giving the same "framework bookkeeping trails user data" intent without a
relocation step.

## Files to modify

| File | Change |
|------|--------|
| `src/phenotypic/sdk_/_metadata_helpers.py` | Add `_METADATA_CLUSTER_ORDER`, `canonical_metadata_order()`, a `_cluster_ordered_enums()` helper, and `order_measurement_columns()`; reorder `metadata_category_prefixes()` |
| `src/phenotypic/sdk_/__init__.py` | Export `canonical_metadata_order`, `order_measurement_columns` |
| `src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py` | `insert_metadata()` sort key → canonical order; docstring note (REMBI → cluster) |
| `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py` | `_order_measurement_columns` → thin wrapper over shared `order_measurement_columns` (drop `info_cols` param + its capture) |
| `src/phenotypic/_cli/_cli_output_manager.py` | Add the canonical-order `post_df.select(...)` pass after `_apply_post_to_master`, before `_seed_measurements` |
| `docs/source/explanation/metadata_namespace.md` | Add a "Column order (bio-semantic clusters)" section; clarify REMBI is a separate axis |
| `src/phenotypic/schema/CLAUDE.md` | One line: front-block order is cluster-driven, REMBI is provenance-only |

## Consumers affected (expected, benign)

- **Dashboard scatter-axis priority** (`SCATTER_PREFIX_PRIORITY`) — default axis
  candidate order shifts to cluster order (Sample/Plate first). Arguably better UX.
- **Per-feature splits + README bucketing** — metadata buckets render in cluster order.
- **`measurements.*` mirror** — front-block column order changes (the visible goal).

## Resolved decisions (formerly open questions)

1. **Uncategorized vs `MetadataImage_` in secondary surfaces — ACCEPTED as recommended.**
   In the mirror this is moot (`order_measurement_columns` puts `MetadataImage_` after
   the measurements and uncategorized user tags at the tail of the *front* block). In the
   secondary prefix-list surfaces (dashboard scatter priority, README/split bucketing) the
   list ends `… Acquisition, Image`, so an *unknown* `Metadata_Foo` sorts after
   `MetadataImage_`. We accept this — those surfaces rarely carry unknown tags, and no
   special-casing is added.
2. **External `--metadata` CSV join — RESOLVED (add the pass).** Traced: the join
   (`join_metadata`, `_cli_output_manager.py:92`) is a **polars** inner join with the
   external CSV on the left, so joined columns land front in CSV order, bypassing the
   canonical order; the mirror/splits/analysis derive from that frame. Fix specified in
   Design §5 — a single `post_df.select(order_measurement_columns(post_df.columns))` pass
   after post, before `_seed_measurements`, sharing the same ordering helper as the pandas
   path. The clean master on disk is untouched.

## Known limitations (deferred; surfaced by the implementation review)

`order_measurement_columns` classifies columns by name. One edge case remains deferred:

1. **Non-metadata columns literally named `Grid_*` / `Bbox_*`** are detected as info-block
   columns and moved to the trailing geometry block. Only the `GRID`/`BBOX` schema enums
   emit those prefixes (guarded by `test_info_block_prefix_is_collision_free`), so this can
   only occur if a `post/` op or a `--metadata` CSV introduces such a name. Closing it would
   need an explicit info-column allowlist rather than prefix inference; deferred.

**Resolved (was limitation #2):** bare (un-prefixed) `--metadata` CSV *attribute* columns
(e.g. `Strain`, `Replicate`) are now prefixed by `join_metadata` via `ensure_metadata_prefix`
(`Strain` → `MetadataGenetic_Strain`; unknown `Foo` → `Metadata_Foo`) before the join, so
they are recognized as metadata and order into the front block — matching the pandas
`insert_metadata` path. Join-key columns keep their raw names so the join still matches.

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
   - **Info-block guard**: assert no non-`GRID`/`BBOX` schema enum renders headers that
     match the `"Grid_"` / `"Bbox_"` prefix, so name-based info detection can't misclassify
     a future measurement column into the trailing info block.
3. **Update pinned tests** — `tests/unit/sdk_/test_metadata_helpers.py`
   (`_expected_metadata_prefixes` → cluster order); any REMBI-order assertions in
   `tests/unit/sdk_/test_metadata_io.py` / `test_metadata_by_module.py`.
4. **Mirror + external join (polars path)** — a small CLI/E2E test: run with a
   `--metadata` CSV whose columns are deliberately in a non-canonical order, then assert
   `deliverables/measurements.parquet` column order matches the canonical contract
   (cluster-ordered front metadata → measurements → `MetadataImage_` → info block) and
   that the clean `master_measurements.parquet` is unchanged.
5. **Shared helper parity** — `order_measurement_columns` unit test on a shuffled column
   list; assert the pandas `_order_measurement_columns` wrapper and a `polars.select`
   using the same helper produce identical column order.
6. **Regression** — `test_cli_output_manager`, `test_measurement_outputs`, split tests,
   dashboard scatter tests, and #180's `test_measure_column_order_*` (must still pass
   after the wrapper refactor).
7. `uv run ruff check --fix` + `uv run mypy src/phenotypic` on changed files.

## Ship

- New branch off `main` (e.g. `feat/metadata-cluster-order`), commit, push, PR → `main`.
- PR notes: reorder-only, REMBI untouched, single-source-of-truth, `MetadataCulture_`
  contiguity fix; deferred follow-up to #180.
