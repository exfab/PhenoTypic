# Metadata Cluster Ordering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder the metadata front-block of the measurement sheets from REMBI-provenance order to a bench-scientist narrative — Identity → Strain → Condition → Design — driven by a single source of truth shared by the pandas per-image path and the polars mirror path.

**Architecture:** Add a cluster-order constant + a `canonical_metadata_order()` rank map + a framework-agnostic `order_measurement_columns()` helper in `sdk_/_metadata_helpers.py`. `insert_metadata()` and `metadata_category_prefixes()` sort by the cluster order; the pandas `measure()` reorder and the polars mirror finalize both call the one shared `order_measurement_columns()`. REMBI stays a separate provenance axis (untouched).

**Tech Stack:** Python 3.12, pydantic v2 schema enums, pandas (per-image path), polars (mirror path), pytest, `uv` runner.

## Global Constraints

- **`uv` is the sole runner.** Every command is `uv run …`; never bare `python`/`pip`.
- **Prerequisite: PR #180 must be present in the branch.** This plan refactors
  `ImagePipelineCore._order_measurement_columns`, which #180 introduced. The working
  branch `feat/metadata-cluster-order` is currently based on `main` (pre-#180) — **rebase
  it onto #180** (or onto `main` after #180 merges) before Task 6. Tasks 1–5 do not touch
  #180 code and may proceed on the current base.
- **Reorder only.** No per-member cluster tags, no new `cluster()` classmethod on enums.
  The ordering unit is the whole category prefix.
- **REMBI is untouched.** `by_module()`, `header_to_module()`, `_rembi_manifest.py` keep
  their REMBI logic. Cluster order is a separate, human-facing axis.
- **Canonical cluster order (verbatim):**
  `MetadataSample → MetadataPlate → MetadataGenetic → MetadataCondition →
  MetadataCulture → MetadataExperiment → MetadataStudy → MetadataAcquisition →
  MetadataImage`. Within a category: enum **definition order**.
- **Full column contract:** `[front metadata] → [measurements] → [MetadataImage_*] →
  [Object_Label, Bbox_*, Grid_*]`.
- **Spec:** `docs/superpowers/specs/2026-07-02-metadata-cluster-ordering/design.md`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/phenotypic/sdk_/_metadata_helpers.py` | New: `_METADATA_CLUSTER_ORDER`, `_cluster_ordered_enums()`, `canonical_metadata_order()`, `order_measurement_columns()`; reordered `metadata_category_prefixes()` |
| `src/phenotypic/sdk_/__init__.py` | Export `canonical_metadata_order`, `order_measurement_columns` |
| `src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py` | `insert_metadata()` sorts by canonical order |
| `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py` | `_order_measurement_columns` → thin wrapper over the shared helper (drop `info_cols`) |
| `src/phenotypic/_cli/_cli_output_manager.py` | Canonical-order `post_df.select(...)` pass before `_seed_measurements` |
| `tests/unit/sdk_/test_metadata_helpers.py` | Update pinned prefix expectations to cluster order; new cluster/canonical/guard tests |
| `tests/unit/core/test_metadata_cluster_order.py` | New: `order_measurement_columns` + polars finalize integration |
| `docs/source/explanation/metadata_namespace.md`, `src/phenotypic/schema/CLAUDE.md` | Doc notes: cluster order is the human axis, REMBI is provenance |

---

## Execution: DAG & Clustering

Derived from the per-task Files/Interfaces blocks (→ = must-finish-before; shared files
in brackets).

```
T1 [_metadata_helpers.py] ─┬─→ T2 [+__init__] ─┬─→ T4 [+__init__, test_cluster_order] ─┬─→ T6 [_image_pipeline_core]*
                           └─→ T3             └─→ T5 [_metadata_accessor, test_by_module]└─→ T7 [_cli_output_manager]
T8 (docs) — independent                                          T9 (verify+ship) — depends on all
* T6 additionally requires the #180 rebase (see Global Constraints).
```

**Clusters** (right-sized by shape, not count):

| Cluster | Tasks | Shape | Rationale |
|---|---|---|---|
| **A — Ordering source of truth** | T1, T2, T3, T4, T8 | Keystone + Leaf | All four code tasks rewrite `_metadata_helpers.py` (shared file → cannot parallelize; cohesive intent). One reviewable diff; self-verifies via `test_metadata_helpers.py` + the helper tests in `test_metadata_cluster_order.py`. T8 docs describe what A defines → folded in. |
| **B — Consumer wiring** | T5, T7 | Seam ×2 | Wire the shared helper into the pandas `insert_metadata` and the polars mirror. Disjoint source files, shared append-only test file + intent → one agent, one diff. High-scrutiny gate (T7 held the plan-review blocker). |
| **C — #180 wrapper fold** | T6 | Seam | Risky wiring into `measure()` assembly; separate subsystem; carries the #180 rebase prereq. Own gate. |
| **Ship** | T9 | Gate | Orchestrator-run: ruff/mypy/regression sweep + simplify pass + push/PR. |

**Model/effort:** all clusters + gates → session model (Opus 4.8), high effort (Keystone/
Seam judgment work). No sub-frontier work.

**Sequencing:** `[rebase onto #180]` → **A** → light gate → **B** → high-scrutiny gate →
**C** → gate → **Ship**. B and C have zero source-file overlap (parallel-worktree
eligible) but are run **sequentially** — two tiny seams; sequential keeps gates clean and
history bisectable.

**Gates:** per-cluster = review diff + run that cluster's tests/lint before the next; after
B+C = a fresh code-review agent over the combined consumer diff; end = simplify pass +
regression sweep (Task 9).

---

## Task 1: Cluster-order constant + coverage gate

**Files:**
- Modify: `src/phenotypic/sdk_/_metadata_helpers.py`
- Test: `tests/unit/sdk_/test_metadata_helpers.py`

**Interfaces:**
- Produces: `_METADATA_CLUSTER_ORDER: tuple[str, ...]` and
  `_cluster_ordered_enums() -> tuple[type, ...]` (module-private, imported by tests and
  later tasks in the same module).

- [ ] **Step 1: Write the failing coverage-gate test**

Append to `tests/unit/sdk_/test_metadata_helpers.py`:

```python
def test_cluster_order_covers_every_metadata_enum():
    """Every metadata-namespace enum must be placed in the cluster order.

    A new enum added without a cluster slot fails here (coverage gate).
    """
    from phenotypic.sdk_._metadata_helpers import (
        _METADATA_CLUSTER_ORDER,
        _metadata_enums,
    )

    assert set(_METADATA_CLUSTER_ORDER) == {e.category() for e in _metadata_enums()}
    # No duplicate placements.
    assert len(_METADATA_CLUSTER_ORDER) == len(set(_METADATA_CLUSTER_ORDER))


def test_cluster_ordered_enums_follow_constant():
    from phenotypic.sdk_._metadata_helpers import (
        _METADATA_CLUSTER_ORDER,
        _cluster_ordered_enums,
    )

    cats = [e.category() for e in _cluster_ordered_enums()]
    assert cats == list(_METADATA_CLUSTER_ORDER)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/sdk_/test_metadata_helpers.py::test_cluster_order_covers_every_metadata_enum -v`
Expected: FAIL with `ImportError: cannot import name '_METADATA_CLUSTER_ORDER'`.

- [ ] **Step 3: Add the constant and the ordered-enums helper**

In `src/phenotypic/sdk_/_metadata_helpers.py`, immediately after the `_metadata_enums()`
function (before `metadata_category_prefixes`), add:

```python
#: Bio-semantic cluster order for the metadata front-block (category granularity).
#: The human-facing column axis. REMBI (header_to_module / by_module / manifest)
#: remains a SEPARATE provenance axis and is not affected by this ordering.
_METADATA_CLUSTER_ORDER: tuple[str, ...] = (
    # (1) Identity — who / where is this colony
    "MetadataSample",
    "MetadataPlate",
    # (2) Strain — genetic identity
    "MetadataGenetic",
    # (3) Condition — chemical then temporal/physical environment
    "MetadataCondition",
    "MetadataCulture",
    # (4) Design & provenance
    "MetadataExperiment",
    "MetadataStudy",
    "MetadataAcquisition",
    # Framework per-image bookkeeping — last (relocated to the trailing region
    # of the measurement frame by order_measurement_columns()).
    "MetadataImage",
)


@lru_cache(maxsize=1)
def _cluster_ordered_enums() -> tuple[type, ...]:
    """The metadata enums sorted by ``_METADATA_CLUSTER_ORDER`` (then stable).

    An enum whose category is absent from the cluster order sorts last; the
    coverage-gate test forbids that state, so it is a defensive fallback only.
    """
    rank = {cat: i for i, cat in enumerate(_METADATA_CLUSTER_ORDER)}
    return tuple(
        sorted(_metadata_enums(), key=lambda e: rank.get(e.category(), len(rank)))
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/sdk_/test_metadata_helpers.py::test_cluster_order_covers_every_metadata_enum tests/unit/sdk_/test_metadata_helpers.py::test_cluster_ordered_enums_follow_constant -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_metadata_helpers.py tests/unit/sdk_/test_metadata_helpers.py
git commit -m "feat(metadata): add bio-semantic cluster order constant + coverage gate"
```

---

## Task 2: `canonical_metadata_order()` rank map

**Files:**
- Modify: `src/phenotypic/sdk_/_metadata_helpers.py`
- Modify: `src/phenotypic/sdk_/__init__.py`
- Test: `tests/unit/sdk_/test_metadata_helpers.py`

**Interfaces:**
- Consumes: `_cluster_ordered_enums()` (Task 1).
- Produces: `canonical_metadata_order() -> dict[str, int]` — maps each known metadata
  **header** (e.g. `"MetadataGenetic_Strain"`) to a global integer rank, cluster-order
  major, enum definition-order minor. Exported from `phenotypic.sdk_`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/sdk_/test_metadata_helpers.py`:

```python
def test_canonical_order_clusters_then_definition_order():
    from phenotypic.schema import GENETIC_METADATA, SAMPLE_METADATA
    from phenotypic.sdk_ import canonical_metadata_order

    order = canonical_metadata_order()

    # Identity (Sample) ranks before Strain (Genetic): whole clusters ordered.
    assert order[SAMPLE_METADATA.SAMPLE_ID.value] < order[GENETIC_METADATA.ORGANISM.value]
    # Within Genetic: definition order (Organism declared before Strain).
    assert order[GENETIC_METADATA.ORGANISM.value] < order[GENETIC_METADATA.STRAIN.value]
    # MetadataImage_ ranks last among categories.
    from phenotypic.schema import METADATA
    assert order[METADATA.IMAGE_NAME.value] > order[GENETIC_METADATA.STRAIN.value]


def test_canonical_order_unknown_header_absent():
    from phenotypic.sdk_ import canonical_metadata_order

    assert "Metadata_TotallyUnknownTag" not in canonical_metadata_order()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/sdk_/test_metadata_helpers.py::test_canonical_order_clusters_then_definition_order -v`
Expected: FAIL with `ImportError: cannot import name 'canonical_metadata_order'`.

- [ ] **Step 3: Implement `canonical_metadata_order()`**

In `src/phenotypic/sdk_/_metadata_helpers.py`, add after `_cluster_ordered_enums()`:

```python
@lru_cache(maxsize=1)
def canonical_metadata_order() -> dict[str, int]:
    """Global rank for every known metadata header (cluster then definition order).

    Cluster-order major, enum definition-order minor. A header absent from this
    map is an unknown/uncategorized user tag; callers rank those last. The map is
    derived entirely from the import-time schema enums, so it is cached.
    """
    cat_rank = {e.category(): i for i, e in enumerate(_cluster_ordered_enums())}
    out: dict[str, int] = {}
    for enum in _cluster_ordered_enums():
        # Category stride (1000) must exceed the largest enum's member count
        # (~12 today) so per-category definition ranks never bleed into the
        # next category. Guard the invariant rather than let a silent collision
        # corrupt ordering.
        assert len(enum) < 1000, f"{enum.__name__} exceeds the category stride"
        base = cat_rank[enum.category()] * 1000
        for i, member in enumerate(enum):
            out[member.value] = base + i
    return out
```

- [ ] **Step 4: Export from `phenotypic.sdk_`**

In `src/phenotypic/sdk_/__init__.py`, next to the existing `metadata_category_prefixes`
import and `__all__` entry, add `canonical_metadata_order` to both. Find:

```python
    metadata_category_prefixes,
```
Add on the following line:
```python
    canonical_metadata_order,
```
And in the `__all__` list, next to `"metadata_category_prefixes",` add:
```python
    "canonical_metadata_order",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/sdk_/test_metadata_helpers.py -k canonical_order -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/_metadata_helpers.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_/test_metadata_helpers.py
git commit -m "feat(metadata): add canonical_metadata_order() rank map"
```

---

## Task 3: Reorder `metadata_category_prefixes()` to cluster order

**Files:**
- Modify: `src/phenotypic/sdk_/_metadata_helpers.py`
- Test: `tests/unit/sdk_/test_metadata_helpers.py` (update pinned expectations)

**Interfaces:**
- Consumes: `_cluster_ordered_enums()` (Task 1).
- Produces: `metadata_category_prefixes()` now returns prefixes in cluster order
  (same `tuple[str, ...]` shape).

- [ ] **Step 1: Update the pinned expectation test to cluster order**

In `tests/unit/sdk_/test_metadata_helpers.py`, **delete** the old
`test_prefixes_match_schema_derivation` (lines 51-52) and **replace** the
`_expected_metadata_prefixes` helper (lines 26-48) with an explicit cluster-ordered
expectation plus a new pinning test (do not leave the old REMBI-derivation helper body —
it would silently become a duplicate with a misleading name):

```python
def _expected_metadata_prefixes() -> tuple[str, ...]:
    """Expected prefixes in bio-semantic cluster order (independent of the helper)."""
    return (
        "MetadataSample_",
        "MetadataPlate_",
        "MetadataGenetic_",
        "MetadataCondition_",
        "MetadataCulture_",
        "MetadataExperiment_",
        "MetadataStudy_",
        "MetadataAcquisition_",
        "MetadataImage_",
    )


def test_prefixes_match_cluster_order():
    assert metadata_category_prefixes() == _expected_metadata_prefixes()
```

Then update `test_prefix_order_follows_rembi_module_rank` — rename it and reframe it on
the cluster axis (the Genetic-before-Image assertion still holds because Strain precedes
the trailing Image block):

```python
def test_prefix_order_follows_cluster_rank():
    """Strain (MetadataGenetic) sorts ahead of the trailing framework Image block."""
    prefixes = metadata_category_prefixes()
    assert prefixes.index("MetadataGenetic_") < prefixes.index("MetadataImage_")
    # Identity leads: Sample is the first prefix.
    assert prefixes[0] == "MetadataSample_"
```

Remove the now-unused `REMBI_MODULE` import from the test file if nothing else uses it
(the `is_metadata_header` / `metadata_category_for_label` tests do not).

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/sdk_/test_metadata_helpers.py::test_prefixes_match_cluster_order -v`
Expected: FAIL — the helper still returns REMBI order (`MetadataStudy_` first), not
`MetadataSample_` first.

- [ ] **Step 3: Reimplement `metadata_category_prefixes()`**

In `src/phenotypic/sdk_/_metadata_helpers.py`, replace the body of
`metadata_category_prefixes()` (the REMBI-sorted version) with a walk over
`_cluster_ordered_enums()`:

```python
@lru_cache(maxsize=1)
def metadata_category_prefixes() -> tuple[str, ...]:
    """All metadata category prefixes (e.g. ``'MetadataGenetic_'``) in cluster order.

    Ordered by the bio-semantic cluster order (``_METADATA_CLUSTER_ORDER``), then
    deduplicated, so callers building bucket-priority lists get a stable, canonical
    ordering. REMBI is a separate axis (see ``by_module`` / ``header_to_module``).
    """
    seen: set[str] = set()
    prefixes: list[str] = []
    for e in _cluster_ordered_enums():
        p = f"{e.category()}_"
        if p not in seen:
            seen.add(p)
            prefixes.append(p)
    return tuple(prefixes)
```

Leave the `REMBI_MODULE` import at the top of `_metadata_helpers.py` in place only if it
is still referenced elsewhere in the file; if this was its last use, remove it from the
`from phenotypic.schema import MeasurementInfo, REMBI_MODULE` line.

- [ ] **Step 4: Run the affected tests to verify they pass**

Run: `uv run pytest tests/unit/sdk_/test_metadata_helpers.py -v`
Expected: PASS (all, including `test_prefixes_are_well_formed_and_unique`,
`test_prefixes_cover_genetic_and_framework_enums`, `test_prefix_order_follows_cluster_rank`).

- [ ] **Step 5: Run the dashboard scatter-priority consumers**

Run: `uv run pytest tests/unit/ -k "scatter or analysis_helpers" -q`
Expected: PASS — `SCATTER_PREFIX_PRIORITY` picks up the new order transparently.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/_metadata_helpers.py tests/unit/sdk_/test_metadata_helpers.py
git commit -m "feat(metadata): order category prefixes by bio-semantic cluster"
```

---

## Task 4: `order_measurement_columns()` shared helper

**Files:**
- Modify: `src/phenotypic/sdk_/_metadata_helpers.py`
- Modify: `src/phenotypic/sdk_/__init__.py`
- Test: `tests/unit/core/test_metadata_cluster_order.py` (new file)

**Interfaces:**
- Consumes: `canonical_metadata_order()` (Task 2), `is_metadata_header()` (same module).
- Produces: `order_measurement_columns(columns: Sequence[str]) -> list[str]` — pure,
  framework-agnostic ordering used by both the pandas and polars paths. Exported from
  `phenotypic.sdk_`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/core/test_metadata_cluster_order.py`:

```python
"""Cluster-ordering: the shared order_measurement_columns helper + mirror wiring."""

from __future__ import annotations


def test_order_measurement_columns_full_contract():
    from phenotypic.sdk_ import order_measurement_columns

    # Deliberately shuffled input spanning every partition.
    cols = [
        "Grid_RowNum",                 # info
        "MetadataImage_ImageName",     # framework image (trailing)
        "Shape_Area",                  # measurement
        "MetadataGenetic_Strain",      # front metadata (Strain cluster)
        "Object_Label",                # info (leads info block by name, not position)
        "MetadataSample_SampleID",     # front metadata (Identity cluster, leads)
        "MetadataCondition_Media",     # front metadata (Condition cluster)
        "Metadata_UnknownTag",         # uncategorized user metadata -> end of front
    ]

    ordered = order_measurement_columns(cols)

    assert ordered == [
        # front metadata: Identity (Sample) -> Strain -> Condition -> uncategorized
        "MetadataSample_SampleID",
        "MetadataGenetic_Strain",
        "MetadataCondition_Media",
        "Metadata_UnknownTag",
        # measurements
        "Shape_Area",
        # framework image block
        "MetadataImage_ImageName",
        # per-object info block
        "Object_Label",
        "Grid_RowNum",
    ]


def test_order_measurement_columns_no_metadata():
    from phenotypic.sdk_ import order_measurement_columns

    cols = ["Shape_Area", "Object_Label", "Bbox_MinRR", "Intensity_MeanIntensity"]
    ordered = order_measurement_columns(cols)
    # Measurements keep relative order; info block trails.
    assert ordered == [
        "Shape_Area",
        "Intensity_MeanIntensity",
        "Object_Label",
        "Bbox_MinRR",
    ]


def test_info_block_prefix_is_collision_free():
    """Only GRID/BBOX emit Grid_/Bbox_ headers; GridLinReg_/GridSpread_ must not."""
    import phenotypic.schema as schema
    from phenotypic.schema import MeasurementInfo

    for name in schema.__all__:
        obj = getattr(schema, name)
        if not (isinstance(obj, type) and issubclass(obj, MeasurementInfo)
                and obj is not MeasurementInfo and list(obj)):
            continue
        if obj.category() in {"Grid", "Bbox"}:
            continue
        for member in obj:
            assert not member.value.startswith("Grid_"), member.value
            assert not member.value.startswith("Bbox_"), member.value
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/core/test_metadata_cluster_order.py::test_order_measurement_columns_full_contract -v`
Expected: FAIL with `ImportError: cannot import name 'order_measurement_columns'`.

- [ ] **Step 3: Implement `order_measurement_columns()`**

In `src/phenotypic/sdk_/_metadata_helpers.py`, add (top of file, extend the imports):

```python
from collections.abc import Sequence
```

Then add the helper after `canonical_metadata_order()`:

```python
def order_measurement_columns(columns: Sequence[str]) -> list[str]:
    """Canonical measurement-frame column order.

    ``[front metadata] -> [measurements] -> [MetadataImage_*] -> [info block]``.

    Front (user/experimental) metadata is cluster/definition ordered via
    :func:`canonical_metadata_order`; unknown/uncategorized ``Metadata_*`` tags fall
    to the end of the front block alphabetically. The framework ``MetadataImage_*``
    block is per-image provenance and trails the measurements. The per-object info
    block (``Object_Label`` + ``Bbox_*`` / ``Grid_*``) is detected by name and moves
    last. Measurements keep their incoming relative order.

    Pure over column-name strings, so both the pandas (``df[...]``) and polars
    (``df.select(...)``) paths reuse it.
    """
    from phenotypic.schema import METADATA, OBJECT

    rank = canonical_metadata_order()
    image_prefix = f"{METADATA.category()}_"
    label = str(OBJECT.LABEL)

    front: list[str] = []
    image_meta: list[str] = []
    info: list[str] = []
    meas: list[str] = []
    for c in columns:
        if c.startswith(image_prefix):
            image_meta.append(c)
        elif is_metadata_header(c):
            front.append(c)
        elif c == label or c.startswith("Bbox_") or c.startswith("Grid_"):
            info.append(c)
        else:
            meas.append(c)
    # Unknown/uncategorized Metadata_* tags sort AFTER every known header. Ranks
    # use a 1000-stride (Task 2), so `len(rank)` (~72) is NOT a valid "after
    # everything" sentinel — it would land unknown tags mid-front-block. Derive
    # the sentinel from the actual max rank.
    unknown_rank = max(rank.values(), default=0) + 1
    front.sort(key=lambda c: (rank.get(c, unknown_rank), str(c)))
    # Object_Label leads the info block; the Bbox_*/Grid_* geometry keeps its
    # incoming order (stable sort) so the trailing block matches the info-frame
    # geometry order that #180 produced.
    info.sort(key=lambda c: 0 if c == label else 1)
    return front + meas + image_meta + info
```

- [ ] **Step 4: Export from `phenotypic.sdk_`**

In `src/phenotypic/sdk_/__init__.py`, add `order_measurement_columns` to the import block
(next to `canonical_metadata_order`) and to `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/core/test_metadata_cluster_order.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/_metadata_helpers.py src/phenotypic/sdk_/__init__.py tests/unit/core/test_metadata_cluster_order.py
git commit -m "feat(metadata): add shared order_measurement_columns helper"
```

---

## Task 5: `insert_metadata()` sorts by canonical order

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py:317-354`
- Test: `tests/unit/sdk_/test_metadata_io.py`, `tests/unit/core/test_metadata_by_module.py`

**Interfaces:**
- Consumes: `canonical_metadata_order()` (Task 2), `ensure_metadata_prefix()`,
  `is_metadata_header()` (existing).
- Produces: `insert_metadata()` inserts metadata columns front-block in cluster order.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/core/test_metadata_cluster_order.py`:

```python
def test_insert_metadata_front_block_cluster_order():
    """insert_metadata places user tags in cluster order (Sample/Identity before Strain)."""
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.sdk_ import is_metadata_header
    import pandas as pd

    img = load_synth_yeast_plate()
    # Set tags out of cluster order on purpose.
    img.metadata["Strain"] = "BY4741"        # MetadataGenetic_ (Strain cluster)
    img.metadata["SampleID"] = "S1"          # MetadataSample_ (Identity cluster, leads)
    img.metadata["Media"] = "YPD"            # MetadataCondition_ (Condition cluster)

    df = img.metadata.insert_metadata(pd.DataFrame({"Object_Label": [1]}))
    meta_cols = [c for c in df.columns if is_metadata_header(c)]

    # Identity (Sample) precedes Strain precedes Condition.
    assert meta_cols.index("MetadataSample_SampleID") < meta_cols.index(
        "MetadataGenetic_Strain"
    )
    assert meta_cols.index("MetadataGenetic_Strain") < meta_cols.index(
        "MetadataCondition_Media"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/core/test_metadata_cluster_order.py::test_insert_metadata_front_block_cluster_order -v`
Expected: FAIL — under REMBI order Genetic (Biosample) currently precedes Sample only by
category alpha, and Condition (SpecimenPrep) lands after both, so the Sample-before-Strain
assertion fails (Genetic sorts before Sample today).

- [ ] **Step 3: Swap the `insert_metadata` sort key to canonical order**

In `src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py`, replace the REMBI
block inside `insert_metadata` (currently importing `REMBI_MODULE, header_to_module` and
building `idx`/`order`/`_rank`) with:

```python
        working_df = df if inplace else df.copy()
        # Insert metadata columns in canonical bio-semantic cluster order (then
        # definition order, then alpha for unknown tags). insert() places each
        # column at loc=0, so iterate in reverse rank to land the lowest-rank
        # category (Identity) at the leftmost position.
        from phenotypic.sdk_ import (
            canonical_metadata_order,
            ensure_metadata_prefix,
            is_metadata_header,
        )

        rank = canonical_metadata_order()
        # Unknown/uncategorized tags sort after every known header (1000-stride
        # ranks, so len(rank) is not a valid sentinel — mirrors
        # order_measurement_columns). reverse=True + insert(loc=0) lands the
        # lowest-rank category (Identity) leftmost and unknown tags at the tail
        # of the front block.
        unknown_rank = max(rank.values(), default=0) + 1

        def _rank(item):
            header = ensure_metadata_prefix(item[0])
            return (rank.get(header, unknown_rank), str(item[0]))

        items = sorted(
            self._public_protected_metadata.items(), key=_rank, reverse=True
        )
```

Leave the rest of the method (the `for key, value in items:` insert loop) unchanged.
Update the method docstring's Notes bullet that reads
"Columns are inserted … so iteration order determines final order" to note the order is
the bio-semantic cluster order (not REMBI).

- [ ] **Step 4: Update the known REMBI-order assertion in `test_metadata_by_module.py`**

`tests/unit/core/test_metadata_by_module.py::test_insert_metadata_orders_by_rembi_module`
(line 51) pins the old REMBI front-block order and **will** fail. Under the cluster order
the sequence is Strain (Genetic, cluster 2) → Media (Condition, 3) → Dataset (Experiment,
4) → ImageName (Image, last). Update the assertion and its comment (lines 50-51):

```python
    # Canonical cluster order: Strain (Genetic) < Media (Condition)
    # < Dataset (Experiment) < ImageName (framework Image, last).
    assert _pos("Strain") < _pos("Media") < _pos("Dataset") < _pos("ImageName")
```

Also update the setup comment (lines 36-39) so it no longer describes REMBI ranks — note
the cluster ranks instead. Rename the test function to
`test_insert_metadata_orders_by_cluster` for accuracy.

- [ ] **Step 5: Run the new test + the metadata-accessor suites**

Run: `uv run pytest tests/unit/core/test_metadata_cluster_order.py::test_insert_metadata_front_block_cluster_order tests/unit/sdk_/test_metadata_io.py tests/unit/core/test_metadata_by_module.py -q`
Expected: PASS. (`by_module()` grouping itself is REMBI-based and unchanged; only the
insert-order assertion above moves.)

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py tests/unit/core/test_metadata_cluster_order.py tests/unit/core/test_metadata_by_module.py
git commit -m "feat(metadata): insert_metadata sorts by bio-semantic cluster order"
```

---

## Task 6: Pandas `_order_measurement_columns` → shared helper wrapper

**Prerequisite:** PR #180 present in the branch (see Global Constraints). Rebase first.

**Files:**
- Modify: `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py` (`measure()` call
  site ~1127-1137 and the `_order_measurement_columns` static method ~1153)
- Test: `tests/unit/core/test_image_pipeline.py` (#180's `test_measure_column_order_*`
  must still pass unchanged)

**Interfaces:**
- Consumes: `order_measurement_columns()` (Task 4).
- Produces: `_order_measurement_columns(df) -> pd.DataFrame` (no `info_cols` param).

- [ ] **Step 1: Run #180's contract tests to establish the green baseline**

Run: `uv run pytest tests/unit/core/test_image_pipeline.py -k column_order -v`
Expected: PASS (2 passed) on the pre-refactor code.

- [ ] **Step 2: Replace the static method with a thin wrapper**

In `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py`, replace the whole
`_order_measurement_columns` static method body with:

```python
    @staticmethod
    def _order_measurement_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Order columns as ``[metadata] -> [measurements] -> [MetadataImage_] -> [info]``.

        Delegates to the single source of truth
        :func:`phenotypic.sdk_.order_measurement_columns`, shared with the polars
        mirror path, so both surfaces agree on the cluster/definition ordering and
        the ``MetadataImage_``-after-measurements placement.

        Args:
            df: The merged measurement DataFrame.

        Returns:
            pd.DataFrame: *df* with columns reordered; rows/index unchanged.
        """
        from phenotypic.sdk_ import order_measurement_columns

        return df[order_measurement_columns(list(df.columns))]
```

- [ ] **Step 3: Drop the `info_cols` capture and update the call site**

At the `measure()` assembly (~lines 1127-1137), remove the pre-merge `info_cols` capture
and the argument. Change:

```python
        # The info frame is appended last and is metadata-free, so its columns
        # (``Object_Label`` + ``Bbox_*`` / ``Grid_*``) form the trailing info block.
        info_cols = list(measurements[-1].columns)

        df = self._merge_on_object_labels(measurements)

        # Metadata (and, downstream, external-joined metadata) belongs at the
        # front, ahead of the measurements; the info block stays last.
        if include_metadata:
            df = image.metadata.insert_metadata(df)
        df = self._order_measurement_columns(df, info_cols)
```
to:
```python
        df = self._merge_on_object_labels(measurements)

        # Metadata (and, downstream, external-joined metadata) belongs at the
        # front, ahead of the measurements; the info block stays last. Ordering
        # is delegated to the shared canonical helper.
        if include_metadata:
            df = image.metadata.insert_metadata(df)
        df = self._order_measurement_columns(df)
```

- [ ] **Step 4: Run #180's contract tests + a broad pipeline sweep**

Run: `uv run pytest tests/unit/core/test_image_pipeline.py -q`
Expected: PASS (all, including both `test_measure_column_order_*`). The name-based info
detection is equivalent to the old `info_cols` (guarded by
`test_info_block_prefix_is_collision_free` from Task 4).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py
git commit -m "refactor(measure): delegate column order to shared canonical helper"
```

---

## Task 7: Polars mirror canonical-order pass

**Files:**
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:661-662`
- Test: `tests/unit/core/test_metadata_cluster_order.py`
- Modify (regression): `tests/unit/cli/test_cli_output_manager.py:284,286` — the mirror
  now legitimately diverges from the clean master in column *order*.

**Interfaces:**
- Consumes: `order_measurement_columns()` (Task 4), `finalize_post_master_outputs(...)`
  (existing, returns the post/mirror polars frame).
- Produces: the mirror `measurements.*` + splits + analysis carry the canonical order.

**Note:** this pass makes the mirror's column order differ from the clean master's. An
existing test (`test_split_runs_via_state_file_fallback`) asserts the two are byte-identical
in column order — that invariant is intentionally broken here and its assertions must be
relaxed to *set*-equality + data-equality (Step 4 below). This is the whole point of the
feature: master stays clean/stable, mirror gets the human ordering.

- [ ] **Step 1: Write the failing integration test**

Append to `tests/unit/core/test_metadata_cluster_order.py`:

```python
def test_finalize_mirror_applies_cluster_order(tmp_path):
    """The polars mirror frame from finalize is canonical-ordered, even after a
    --metadata join that lands external columns front-in-CSV-order."""
    import polars as pl
    from phenotypic._cli._cli_output_manager import finalize_post_master_outputs

    # Clean master (metadata-free) with a join anchor column present in both frames.
    master = pl.DataFrame(
        {
            "MetadataImage_ImageName": ["plateA"],
            "Object_Label": [1],
            "Grid_RowNum": [1],
            "Shape_Area": [123.0],
        }
    )
    # External metadata CSV with columns in NON-canonical order.
    meta_csv = tmp_path / "meta.csv"
    meta_csv.write_text(
        "MetadataImage_ImageName,MetadataCondition_Media,MetadataGenetic_Strain,MetadataSample_SampleID\n"
        "plateA,YPD,BY4741,S1\n"
    )

    out_dir = tmp_path / "run"
    out_dir.mkdir()

    post_df = finalize_post_master_outputs(
        out_dir, master, pipeline=None, metadata_csv=meta_csv, no_qc=True
    )

    assert post_df.columns == [
        # front metadata: Identity(Sample) -> Strain -> Condition
        "MetadataSample_SampleID",
        "MetadataGenetic_Strain",
        "MetadataCondition_Media",
        # measurements
        "Shape_Area",
        # framework image block
        "MetadataImage_ImageName",
        # per-object info block
        "Object_Label",
        "Grid_RowNum",
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/core/test_metadata_cluster_order.py::test_finalize_mirror_applies_cluster_order -v`
Expected: FAIL — without the pass, joined metadata columns stay in CSV order with the
image name leading (`MetadataImage_ImageName` first), so the assertion mismatches.

- [ ] **Step 3: Insert the canonical-order pass**

In `src/phenotypic/_cli/_cli_output_manager.py`, inside `finalize_post_master_outputs`,
find:

```python
    post_df = _apply_post_to_master(working_df, pipeline)
    _seed_measurements(output_dir, post_df)
```
and change to:

```python
    post_df = _apply_post_to_master(working_df, pipeline)
    # Reorder the mirror/splits/analysis frame to the canonical cluster contract
    # ([front metadata] -> [measurements] -> [MetadataImage_] -> [info block]),
    # the same helper the pandas per-image path uses. The clean master on disk is
    # untouched — only this in-memory working frame is reordered.
    from phenotypic.sdk_ import order_measurement_columns

    post_df = post_df.select(order_measurement_columns(post_df.columns))
    _seed_measurements(output_dir, post_df)
```

- [ ] **Step 4: Relax the seed-vs-master assertions that pinned identical column order**

In `tests/unit/cli/test_cli_output_manager.py::TestAggregateMeasurementsAutoResolve::test_split_runs_via_state_file_fallback`,
two assertions assume the seeded mirror equals the master byte-for-byte, which the reorder
pass now intentionally breaks (the mirror is cluster-ordered; the master archive is not).

Change line 284:
```python
        assert seed_df.columns == master_df.columns
```
to a set-equality check plus a pin on the mirror's new placement (framework `MetadataImage_`
trails the measurements):
```python
        assert set(seed_df.columns) == set(master_df.columns)
        assert seed_df.columns.index(str(METADATA.IMAGE_NAME)) > seed_df.columns.index(
            "Shape_Area"
        )
```

Change line 286 — polars `.equals()` is column-order sensitive, so align the seed to the
master's column order before comparing data:
```python
        assert seed_pq_df.select(master_pq.columns).equals(master_pq)
```

`METADATA` is already imported in this test module (used by the fixture at line 250); no
new import is needed.

- [ ] **Step 5: Run the integration test + the full output-manager suite**

Run: `uv run pytest tests/unit/core/test_metadata_cluster_order.py::test_finalize_mirror_applies_cluster_order tests/unit/cli/test_cli_output_manager.py -q`
Expected: PASS (the new integration test + all 19 output-manager tests, including the two
relaxed assertions).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli/_cli_output_manager.py tests/unit/core/test_metadata_cluster_order.py tests/unit/cli/test_cli_output_manager.py
git commit -m "feat(cli): apply canonical cluster order to the measurements mirror"
```

---

## Task 8: Documentation

**Files:**
- Modify: `docs/source/explanation/metadata_namespace.md`
- Modify: `src/phenotypic/schema/CLAUDE.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Add the column-order section to `metadata_namespace.md`**

Append a section after the existing REMBI-module table:

```markdown
## Column order (bio-semantic clusters)

Measurement sheets order the metadata front-block by a bench-scientist narrative,
**not** by REMBI module:

1. **Identity** — `MetadataSample_*`, `MetadataPlate_*`
2. **Strain** — `MetadataGenetic_*`
3. **Condition** — `MetadataCondition_*`, `MetadataCulture_*`
4. **Design & provenance** — `MetadataExperiment_*`, `MetadataStudy_*`, `MetadataAcquisition_*`

Unknown/uncategorized `Metadata_*` tags trail the four clusters. Within a category,
columns follow the enum's declaration order. The framework `MetadataImage_*` block is
per-image provenance and is placed **after** the measurements, before the per-object
`Object_Label` / `Bbox_*` / `Grid_*` info block. REMBI (`by_module`, the run manifest)
remains a separate provenance axis and is unaffected by this ordering.
```

- [ ] **Step 2: Add one line to `schema/CLAUDE.md`**

In `src/phenotypic/schema/CLAUDE.md`, near the metadata-enum description, add:

```markdown
- **Front-block column order** is the bio-semantic cluster order
  (Identity → Strain → Condition → Design; see `sdk_/_metadata_helpers.py`
  `_METADATA_CLUSTER_ORDER` / `order_measurement_columns`). REMBI
  (`rembi_module()` / `header_to_module`) is a *separate* provenance axis and does
  not drive column order.
```

- [ ] **Step 3: Commit**

```bash
git add docs/source/explanation/metadata_namespace.md src/phenotypic/schema/CLAUDE.md
git commit -m "docs(metadata): document bio-semantic cluster column order"
```

---

## Task 9: Full verification sweep

**Files:** none (verification only).

- [ ] **Step 1: Lint + type-check the changed files**

Run:
```bash
uv run ruff check --fix src/phenotypic/sdk_/_metadata_helpers.py src/phenotypic/sdk_/__init__.py src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py src/phenotypic/_cli/_cli_output_manager.py tests/unit/core/test_metadata_cluster_order.py tests/unit/sdk_/test_metadata_helpers.py
uv run mypy src/phenotypic/sdk_/_metadata_helpers.py
```
Expected: `All checks passed!`; mypy reports no **new** errors (the pre-existing
`cols_to_merge_on.append` error in `_image_pipeline_core.py` is unrelated).

- [ ] **Step 2: Empirical end-to-end column dump**

Run:
```bash
uv run python -c "
from phenotypic.data import load_synth_yeast_plate
from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize, MeasureShape
img = load_synth_yeast_plate()
img.metadata['Strain']='BY4741'; img.metadata['SampleID']='S1'; img.metadata['Media']='YPD'
pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize(), MeasureShape()])
print(list(pipe.apply_and_measure(img, apply_post=False).columns))
"
```
Expected: `MetadataSample_SampleID`, `MetadataGenetic_Strain`, `MetadataCondition_Media`
first (Identity → Strain → Condition), then `Size_*`/`Shape_*`, then `MetadataImage_*`,
then `Object_Label`, `Bbox_*`, `Grid_*`.

- [ ] **Step 3: Targeted regression suite**

Run:
```bash
uv run pytest tests/unit/sdk_/test_metadata_helpers.py tests/unit/sdk_/test_metadata_io.py tests/unit/core/test_metadata_by_module.py tests/unit/core/test_metadata_cluster_order.py tests/unit/core/test_image_pipeline.py tests/unit/cli/test_cli_output_manager.py tests/unit/util/test_measurement_outputs.py tests/unit/post/test_expand_metadata.py -q
```
Expected: all pass.

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin feat/metadata-cluster-order
gh pr create --base main --title "feat(metadata): bio-semantic cluster column ordering" --body "Implements docs/superpowers/specs/2026-07-02-metadata-cluster-ordering. Reorders the metadata front-block to Identity -> Strain -> Condition -> Design via a single source of truth (canonical_metadata_order + order_measurement_columns) shared by the pandas per-image path and the polars mirror. REMBI untouched. Follow-up to #180."
```
Expected: PR URL printed.

---

## Self-Review

**Spec coverage:**
- Cluster taxonomy + order → Task 1 (constant), Task 3 (prefixes), Task 5 (insert).
- Within-category definition order → Task 2 (`canonical_metadata_order`) + tests.
- Single source of truth (`canonical_metadata_order`, `order_measurement_columns`) →
  Tasks 2, 4; consumed by Tasks 5, 6, 7.
- `metadata_category_prefixes` reorder → Task 3.
- `insert_metadata` → Task 5.
- Shared helper + #180 fold → Task 4 + Task 6.
- Polars mirror pass (external `--metadata` join) → Task 7.
- Coverage gate + info-block guard → Task 1 + Task 4.
- Docs → Task 8. Verification → Task 9.

**Placeholder scan:** every code step shows full code; the one conditional
("if any test pins REMBI front-block order, update it") in Task 5 Step 4 references the
concrete transform defined in Task 3.

**Type consistency:** `canonical_metadata_order() -> dict[str, int]` and
`order_measurement_columns(Sequence[str]) -> list[str]` are used with matching signatures
in Tasks 5 (`rank.get(header, len(rank))`), 6 (`df[order_measurement_columns(list(df.columns))]`),
and 7 (`post_df.select(order_measurement_columns(post_df.columns))`).

**Independent plan review incorporated (2026-07-02):**
- **Blocker** — Task 7's reorder pass breaks
  `test_cli_output_manager.py::test_split_runs_via_state_file_fallback` (it pinned
  seed==master column order). Task 7 Step 4 now relaxes line 284 to set-equality + a mirror
  placement pin, and **also line 286** (`seed_pq_df.equals(master_pq)` — polars `.equals()`
  is order-sensitive, so it's aligned via `.select(master_pq.columns)` before comparing).
  The cli test file is added to Task 7's Files.
- **Should-fix** — Task 5 now names the guaranteed break at
  `test_metadata_by_module.py:51` with the corrected cluster-order assertion (Strain < Media
  < Dataset < ImageName) instead of a vague "if any test pins REMBI order" caveat.
- **Should-fix** — Task 3 now explicitly deletes the stale
  `test_prefixes_match_schema_derivation` (avoids a silent misnamed duplicate).
- **Should-fix** — Task 2's `canonical_metadata_order()` now asserts the ×1000 category
  stride exceeds any enum's member count.
