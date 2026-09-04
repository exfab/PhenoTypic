# Phase 4 — Embedded-table inversion and `finalize_run`

**Depends on:** P3. **Blocks:** P5–P7. *(P0 no longer gates this phase — S-4 was cut by CAN-25 and its one real question lives in Task 1.)*

**Spec:** §7 (measurement and metadata data flow), D8 — as amended by
[D-A](OPEN-QUESTIONS.md#d-a-per-store-metadata-is-written-at-promote-time-not-backfilled).

**Goal:** embedded per-image tables carry **measurements only**; each store's user metadata
is written as `tables/metadata/pht-metadata.parquet` **in the same `.part` as the
measurements**, before the root `zarr.json`; the metadata join moves to finalization; and
`finalize_run` becomes the one aggregation + join + publish path for `full`, `measure` and
`recompile`.

**Local path only.** SLURM and `--njobs` fan-out is P5.

### What D-A changes from spec §7

`finalize_run` is **six steps, not seven** — step 6 ("backfill `pht-metadata.parquet` per
store — certified re-promote") is cut. The metadata table is written at promote time
instead, so **no NEW path** writes into a promoted store (**INV-PROVEN**, first obligation).

§7.4's late-metadata guarantee narrows correspondingly, and the narrowing must be
documented where users read it, not only here:

> A `metadata.csv` edit changes `metadata_sha256`, invalidating
> `finalization_input_digest`, so the next invocation re-runs `finalize_run` — re-joining
> the mirror. **Stores keep the metadata snapshot they were built against**; each store's
> `phenotypic.metadata.snapshot_sha256` records which one, and `resolve_run_state` raises
> an advisory when they diverge (P1 Task 5).

---

## File Structure

| File | Responsibility |
|---|---|
| **Modify** `src/phenotypic/_cli/_embedded_measurement_tables.py:42` | `prepare_embedded_measurement_table` returns the **unjoined** baseline plus a separate metadata projection. |
| **Modify** `src/phenotypic/sdk_/_measurement_tables.py` | Write both tables into the `.part`; extend the root `tables` block. **And repair `replace_embedded_measurement_table`'s in-place branch (`:284-290`)** — see below, the first draft's requirement was not sufficient (CAN-3, flow-r2 C5). |

> **The in-place branch cannot be fixed by "refresh the root with the table" alone (C5).**
> `build_measurement_table_descriptor` (`:109-129`) returns exactly `{schema_version, type,
> format, path, measurement_columns, target}` — **no digest, no join keys, no join status** —
> and `measurement_columns` is the *pre-join baseline* tuple. So a metadata edit that changes
> values but not the measurement schema leaves `current == descriptor` **true**, the in-place
> branch fires, and the Parquet is rewritten inside a promoted store with no `.part` and no
> root rewrite.
>
> **After P4's inversion this gets worse, not better:** the descriptor becomes a pure
> function of the measurement schema and the objmap target, so *every* metadata-driven
> re-measure takes that branch.
>
> Two changes, both required:
> 1. **Put `metadata_snapshot_sha256` into the descriptor**, so `current == descriptor`
>    becomes a real test of "has anything the store certifies changed" rather than of
>    "did the column list change".
> 2. **Refresh the root whenever the payload changes**, not only on a descriptor change.
>
> **State the cost honestly.** (2) means the copytree/hardlink re-promote runs on **every**
> `--mode measure`, not just on a descriptor change. That is precisely spike S-1's cost —
> which D-A cut, on grounds the ledger already records as false (CAN-3): the mechanism
> already ships. So the cost is now *larger* than when S-1 was dropped and is still
> unmeasured. If `--mode measure` on a large tree becomes slow, this is the reason, and
> measuring it is a follow-up, not a blocker for P4.
| **Modify** `src/phenotypic/_cli/_cli_output_manager.py:1970-2001` | `replace_image_store_measurements` feeds the **joined** producer at `:1992-1995`. Bring it onto `prepare_image_tables`, or `--mode measure` silently un-inverts every image it touches. |
| **Create** `src/phenotypic/sdk_/_master_io.py` | `read_master_measurements(output_dir)` — the reader U-3 requires, raising on an unstamped or wrong-versioned master. Route every in-repo master read through it. |
| **Modify** `src/phenotypic/_cli/_embedded_measurement_tables.py:106-131` | `embedded_measurement_table_matches` is **reclaim authority**, not provenance (M1) — six migrator call sites (`_cli_migrate.py:1331,1337`; `_cli_migrate_image.py:278,314,777,796`). See below. |
| **Modify** `src/phenotypic/sdk_/_measurement_tables.py:216-227` | `_valid_embedded_measurement_contract` **rejects `join_status == "not_requested"` with a non-empty digest** — but the inverted producer must record the snapshot digest on an *unjoined* table, or D-A's advisory has no per-store source. Revise in the same commit. |
| **Modify** `src/phenotypic/_cli/_cli_recompile_slurm_scripts.py:186-202` | The **third** `_consistent_embedded_join_keys` call site (M2), at script-generation time, serialising `"metadata_join_keys"` into the finalizer task. Retiring the function from `finalize_run` alone leaves the abort firing **at submission**, before any worker runs. Delete the task-schema field in the same commit. |

> **M1 — the shape change reaches the migrator, and in two opposite directions.**
> `embedded_measurement_table_matches` builds its expected table from
> `prepare_embedded_measurement_table(...)` **including `parquet_metadata()`** — join status,
> join keys, snapshot digest, measurement columns — and asserts
> `actual.equals(expected, check_metadata=True)`. Its docstring calls row count *"reclaim
> authority rather than an incidental property"*, so this is a correctness gate, not a
> convenience check.
>
> - **`_cli_migrate_image.py:278-288`** re-writes the store when `not exact`. A shape change
>   makes `exact` False on **every pre-P4 store**, so P4 would create a **fourth**
>   post-proof store-write path — which INV-PROVEN's own wording forbids ("Do not write a
>   fourth"). Gate the comparison on the store's schema, so a pre-inversion store is
>   compared against pre-inversion expectations.
> - **`_cli_migrate.py:1329-1341`** and **`_cli_migrate_image.py:766-800`** under
>   `--delete-sources`: a mismatch keeps the sources, which is the safe direction — but
>   `--delete-sources` then becomes **permanently impossible** on any pre-P4 store. Say so
>   in P7's docs rather than letting a user discover it.
| **Create** `src/phenotypic/_cli/_cli_finalize_run.py` | `finalize_run(output_dir, …)` — the one path. ~260 lines. |
| **Modify** `src/phenotypic/_cli/_cli_output_manager.py:1351` | `_aggregate_measurements_unlocked` delegates aggregation to `finalize_run`. |
| **Modify** `src/phenotypic/_cli/_cli_recompile_worker.py:764` | `_run_post_master_steps` becomes a `finalize_run` call. |
| **Modify** `src/phenotypic/_cli/_cli_completion.py:868` | Aggregate proof's `required_outputs` drops `master_csv` (D8). |
| **Delete** | `MASTER_MEASUREMENTS_CSV`, `master_measurements_csv_path()`, `BundleLayout.master_csv`, `load_master_measurements()` (D8). |
| **Modify** `src/phenotypic/_cli/_cli_recompile_tables.py` (292 lines) | **Hard-`isinstance`-checks `PreparedEmbeddedMeasurementTable` at `:100`** — the exact type Task 1 replaces — raising `TypeError` on the new one. Reads the marker format by key. |
| **Modify** `src/phenotypic/_cli/_cli_recompile_recovery.py` (838 lines) | Reads the marker format by key; part of the same 15-site set. |
| **Test** `tests/unit/cli/test_finalize_run.py` *(new)* | INV-INPUTS, the six steps, the three entry points. |

> **These two modules are a three-round blind spot, and the largest one found (gen-r3).**
> 1,130 lines that read the marker format at ~15 sites and type-check the producer Task 1
> inverts — appearing in **zero plan documents, zero ledger entries, and zero reviewer
> reports across three rounds**. Recompile is one of the three entry points §7.4 routes
> through `finalize_run` and P4 Task 4 parametrizes a byte-identical master over, so its
> internals were in scope from the first draft; nobody opened them.
>
> `_cli_recompile_tables.py:100` fails **closed** — `TypeError("Recompile table preparation
> returned an invalid payload")` — so this surfaces as a crash rather than corruption, which
> is the good direction. But it means `--mode recompile` is broken from the moment Task 1
> lands until this is fixed, and no test in the plan covers it.
>
> **Measured: 33 marker-format reads across the two files**, not the ~15 first estimated.
> Enumerate them before editing either, the way P3 Step 3b now enumerates the staged-engine
> sites.

### What recompile actually needs from a marker, and why the collapsed form is *simpler*

Recompile uses markers for exactly two things:

1. **An identity round-trip after rewriting a table.** `_republish_table_marker`
   (`_cli_recompile_tables.py:58-82`) reads seven fields — `work_id`, `dataset`,
   `relative_image_path`, `image_stem`, `mode`, `attempt_id`, `lifecycle_epoch` — and hands
   every one straight back to `publish_image_success` with freshly-resolved artifacts. It
   must re-publish because rewriting the embedded table invalidates the marker's artifact
   digests. **The read-back exists only because today's publisher replaces the whole
   marker**, so recompile has to supply every field or lose it.
2. **Discovery fallback.** `_standalone_marker_sources` (`:135`) globs `image_complete/`
   for "valid embedded authority when no processing state is present".

The record is a **superset** of (1)'s seven fields, so a port would be mechanical. But
**P3's merge rule removes the need for the round trip entirely** — `publish_image_record`
merges rather than replaces (CAN-6), so recompile publishes the **new `artifacts` only** and
the merge preserves identity and `stages` untouched.

| Site | Change |
|---|---|
| `_republish_table_marker` (`:58-82`) | **Delete the seven-field read-back.** Publish updated `artifacts`; the merge preserves the rest. `_marker_artifacts` (`:39-56`) and its hand-rolled `relative_to` path validation go with it — the publisher already resolves artifacts under the output root. ~40 lines → one call. |
| `_standalone_marker_sources` (`:135-150`) | Glob `DIR_IMAGE_RECORDS` instead of `DIR_IMAGE_COMPLETE`; the per-entry field reads (`dataset`, `image_stem`, `work_id`, `artifacts.measurements`) are unchanged. |
| `:100` | `isinstance(prepared, PreparedEmbeddedMeasurementTable)` → `PreparedImageTables`. **This is the crash**: it fails closed with `TypeError("Recompile table preparation returned an invalid payload")`, so `--mode recompile` breaks on the first run after Task 1 lands. |
| `_cli_recompile_recovery.py:38,782` | `SUCCESS_MARKER_VERSION` → `RECORD_VERSION`. |
| `_cli_recompile_recovery.py:52,387,477,637,709` | `image_completion_marker_path` → `image_record_path`. |

**Test that `--mode recompile` still round-trips**, since nothing in the plan covered it
before: recompile a two-image tree and assert each record's identity fields and `stages` are
byte-identical to before, with only `artifacts` digests changed.
| **Test** `tests/unit/cli/test_embedded_table_inversion.py` *(new)* | Intrinsic/user metadata boundary; curation re-keying. |

---

## Interfaces

**Produces:**

```python
# phenotypic._cli._cli_finalize_run

def finalize_run(
    output_dir: Path,
    *,
    dataset_names: Sequence[str],
    pipeline: "ImagePipeline | None" = None,
    metadata_csv: Path | None = None,
    no_qc: bool = False,
    study_config: dict | None = None,
    shard_paths: Sequence[Path] | None = None,   # P5 supplies these; None = local concat
    commit_guard: "CommitGuard | None" = None,
) -> Path | None:
    """The one aggregation + join + publish path (spec §7.4)."""
```

```python
# phenotypic._cli._embedded_measurement_tables

@dataclass(frozen=True)
class PreparedImageTables:
    measurements: pd.DataFrame          # intrinsic identity only, NO user metadata
    metadata: pd.DataFrame | None       # user metadata rows + join keys, or None
    measurement_columns: tuple[str, ...]
    join_status: Literal["joined", "not_requested", "no_common_keys"]
    join_keys: tuple[str, ...]
    metadata_snapshot_sha256: str

def prepare_image_tables(
    measurements: pd.DataFrame, metadata_csv: Path | None
) -> PreparedImageTables: ...
```

**Consumes:** P3's `publish_image_record`; `phenotypic.sdk_.promote_store`,
`MEASUREMENT_TABLE_RELATIVE_PATH`.

---

## Task 1: Split the embedded table into measurements and metadata

**Files:**
- Modify: `src/phenotypic/_cli/_embedded_measurement_tables.py:42`
- Test: `tests/unit/cli/test_embedded_table_inversion.py`

**This is subtraction, not invention.** `prepare_embedded_measurement_table` already
computes `measurement_columns` from the baseline **before** joining
(`_embedded_measurement_tables.py:55`) and writes it as
`phenotypic.measurement_columns`. "Embedded table without user metadata" is exactly that
existing projection.

- [ ] **Step 1: Write the failing tests**

```python
def test_intrinsic_identity_stays_in_the_measurement_table(tmp_path):
    """Spec §7.1: a concatenated row that cannot say which image it came from is
    unusable. Metadata_ImageFile, Metadata_Dataset and the object label stay."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(_measurements_with_metadata(), _metadata_csv(tmp_path))
    assert "Metadata_ImageFile" in prepared.measurements.columns
    assert "Metadata_Dataset" in prepared.measurements.columns


def test_user_metadata_leaves_the_measurement_table(tmp_path):
    """§7.3's contract change. Metadata_Strain came from --metadata, not from the
    image, so it belongs in pht-metadata.parquet."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(_measurements_with_metadata(), _metadata_csv(tmp_path))
    assert "Metadata_Strain" not in prepared.measurements.columns
    assert "Metadata_Strain" in prepared.metadata.columns


def test_the_measurement_table_equals_the_pre_join_baseline_exactly(tmp_path):
    """The boundary already has a name: measurement_columns, computed from the
    baseline BEFORE joining (_embedded_measurement_tables.py:55). This asserts the
    new split IS that projection rather than a re-derivation of it."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    baseline = _measurements_with_metadata()
    prepared = prepare_image_tables(baseline, _metadata_csv(tmp_path))
    assert tuple(prepared.measurements.columns) == prepared.measurement_columns


def test_no_metadata_table_when_the_join_was_not_requested(tmp_path):
    """§7.2: absence is the honest signal."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(_measurements_with_metadata(), None)
    assert prepared.metadata is None
    assert prepared.join_status == "not_requested"


def test_no_metadata_table_when_no_columns_are_in_common(tmp_path):
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(
        _measurements_with_metadata(), _unrelated_metadata_csv(tmp_path)
    )
    assert prepared.metadata is None
    assert prepared.join_status == "no_common_keys"


def test_duplicate_metadata_keys_preserve_fan_out(tmp_path):
    """The behaviour prepare_embedded_measurement_table already warns about, and
    the one S-4 spiked. Losing it silently changes row counts in the mirror."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(
        _measurements_with_metadata(), _metadata_csv_with_duplicate_keys(tmp_path)
    )
    assert len(prepared.metadata) == 3
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`prepare_image_tables` keeps `prepare_embedded_measurement_table`'s normalization and its
`prepare_metadata_join_keys` call, and then **stops before the right join**
(`_embedded_measurement_tables.py:88-95`). `measurements` is the baseline; `metadata` is
the semi-join of the metadata frame onto that image's distinct join keys.

**S-4's verdict licenses this.** If S-4 returned `FAIL`, stop and report — a local
projection that diverges from a global one means the promote-time write cannot be correct
and D-A needs revisiting with the user.

Keep `prepare_embedded_measurement_table` as a thin wrapper for one release **only if** a
caller outside this change needs it; grep first, and delete it if not.

- [ ] **Step 4: Run the tests.** Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_embedded_measurement_tables.py \
        tests/unit/cli/test_embedded_table_inversion.py
git commit -m "feat(cli): split the embedded table into measurements and metadata

Spec §7.1-7.2. Subtraction, not invention: measurement_columns already recorded
this boundary, computed from the baseline before the join."
```

---

## Task 2: Write both tables at promote time

**Files:**
- Modify: `src/phenotypic/sdk_/_measurement_tables.py`
- Test: `tests/unit/cli/test_embedded_table_inversion.py`, `tests/unit/sdk_/`

Implements **D-A** and **INV-PROVEN**.

- [ ] **Step 1: Write the failing tests**

```python
def test_both_tables_land_in_the_same_part_before_the_root(tmp_path):
    """D-A / INV-PROVEN. The root zarr.json is written last and is the record's
    content anchor (_cli_completion.py:41-47), so anything written after it is a
    mutation of a proven artifact. Writing metadata in the same .part is what makes
    the backfill unnecessary."""
    store = _build_store_with_metadata(tmp_path)
    assert (store / "tables" / "measurements" / "table.parquet").is_file()
    assert (store / "tables" / "metadata" / "pht-metadata.parquet").is_file()
    root = json.loads((store / "zarr.json").read_text())
    assert "metadata" in root["attributes"]["phenotypic"]["tables"]


def test_the_store_records_the_metadata_snapshot_it_was_built_against(tmp_path):
    """D-A: stores keep the metadata they were built with, and say which one. That
    is what lets resolve_run_state DERIVE the divergence advisory instead of
    tracking a backfill stage."""
    store = _build_store_with_metadata(tmp_path)
    root = json.loads((store / "zarr.json").read_text())
    assert root["attributes"]["phenotypic"]["metadata"]["snapshot_sha256"]


def test_finalize_run_writes_no_byte_into_a_proven_store(tmp_path):
    """INV-PROVEN, first obligation: no NEW path writes into a promoted store.

    Publish a record, snapshot every mtime under the store, run finalize_run, and
    assert not one file moved. This is the test that would have caught the
    backfill if it had shipped."""
    from phenotypic._cli._cli_finalize_run import finalize_run

    store = _publish_a_successful_image(tmp_path, dataset="plate", stem="a")
    before = {p: p.stat().st_mtime_ns for p in sorted(store.rglob("*")) if p.is_file()}
    finalize_run(tmp_path, dataset_names=["plate"])
    after = {p: p.stat().st_mtime_ns for p in sorted(store.rglob("*")) if p.is_file()}
    assert before == after, "finalize_run mutated a store that carries a content proof"


def test_measure_mode_refreshes_the_table_and_the_root_together(tmp_path):
    """INV-PROVEN, second obligation -- and the reason the invariant is stated the
    way it is (CAN-3).

    The stronger claim ("nothing ever writes into a proven store") is FALSE and was
    false before this change. --mode measure re-measures from stores and calls
    replace_embedded_measurement_table (sdk_/_measurement_tables.py:242), whose
    IN-PLACE branch (:284-290) fires when the descriptor is unchanged: it rewrites
    tables/measurements/table.parquet directly in the promoted store, with no
    .part, no copytree, and NO ROOT REWRITE.

    Two things break as a result, and both are silent:
      1. the record's store digest still matches, so the proof certifies content
         that changed underneath it;
      2. `snapshot_sha256` lives in the root, so D-A's divergence advisory reads a
         value this branch never refreshes -- it reports stale metadata as current.
    """
    from phenotypic.sdk_ import STORE_ROOT_JSON

    store = _publish_a_successful_image(tmp_path, dataset="plate", stem="a", metadata=True)
    root_before = (store / STORE_ROOT_JSON).read_bytes()

    _run_measure_mode(tmp_path, metadata=_edited_metadata_csv(tmp_path))

    root_after = (store / STORE_ROOT_JSON).read_bytes()
    assert root_after != root_before, (
        "the embedded table was rewritten without refreshing the root, so the "
        "per-image proof still certifies the old digest and snapshot_sha256 is "
        "stale -- INV-PROVEN's second obligation"
    )
    assert _snapshot_sha256(store) == _sha256_of(_edited_metadata_csv(tmp_path))


def test_measure_mode_writes_the_metadata_table_not_a_joined_one(tmp_path):
    """INV-PROVEN, second obligation, other half.

    replace_image_store_measurements feeds prepare_embedded_measurement_table --
    the JOINED producer -- at _cli_output_manager.py:1992-1995. P4 Task 1 replaced
    that producer with prepare_image_tables everywhere else. If this call site is
    missed, --mode measure on an inverted tree writes joined tables and no
    pht-metadata.parquet, silently un-inverting every image it touches.
    """
    store = _publish_a_successful_image(tmp_path, dataset="plate", stem="a", metadata=True)
    _run_measure_mode(tmp_path, metadata=_metadata_csv(tmp_path))

    measurements = _read_embedded_measurements(store)
    assert "Metadata_Strain" not in measurements.columns
    assert (store / "tables" / "metadata" / "pht-metadata.parquet").is_file()
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

Extend the `.part` writer to emit `tables/metadata/pht-metadata.parquet` when
`prepared.metadata is not None`, before `OME/zarr.json` and the root. Extend the root's
`attributes.phenotypic.tables` with a `metadata` descriptor.

**The snapshot digest goes in a NEW root key, and the name matters (flow-r2 C5).** An
earlier draft said to add `attributes.phenotypic.metadata = {"snapshot_sha256": …}`. That
key is **already taken**: `PhenotypicAttr.METADATA` holds `{protected, public, imported}`
image-metadata sections (`sdk_/ngff_.py:569-580`), carrying things like bit depth
(`:1130-1138`). Writing a snapshot digest there would collide with per-image metadata.

Use `attributes.phenotypic.metadata_table`:

```json
"metadata_table": {"snapshot_sha256": "…", "join_keys": [...], "join_kind": "…"}
```

**And putting it in the root is the point, not a detail.** Today
`METADATA_SNAPSHOT_SHA256 = "phenotypic.metadata.snapshot_sha256"` (`ngff_.py:95`) is
**Arrow schema metadata on the Parquet**, one of `EMBEDDED_MEASUREMENT_PARQUET_METADATA_KEYS`.
D-A's divergence advisory reads it from `sdk_` on the deep path, and P1 Task 5 describes that
as "one attribute read per store … from a value the store already carries". Read from the
Parquet it is not an attribute read — it is **opening a Parquet footer per store**, a
different cost and a new dependency on the INV-LAYER plain-JSON path, and §9.2's numbers do
not include it.

Mirroring it into the root at promote time costs one JSON field, keeps the advisory a plain
`zarr.json` read, and keeps the Parquet copy as the authority the Parquet itself carries.
**Write both; never derive one from the other at read time.**

The Parquet KV keys ride along unchanged (§7.2): `phenotypic.join.keys`,
`phenotypic.join.kind`, `phenotypic.metadata.snapshot_sha256`. The join is self-describing
from the file itself, which is the property that makes the store useful to a third party at
all.

**Order is load-bearing and is INV-PROVEN's first obligation:** chunks → both tables →
`OME/zarr.json` → root `zarr.json` → `promote_store`. Any other order and an interrupted
store can read as present.

- [ ] **Step 4: Run the tests plus the NGFF conformance suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/sdk_ tests/_ngff_conformance.py -q
```

The store gains a table, so its NGFF conformance must be re-checked — a non-conforming
store is one `napari` cannot open, which is half of why it is OME-Zarr.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/sdk_ src/phenotypic/_cli tests/unit
git commit -m "feat(sdk): write pht-metadata.parquet in the store's original promote

D-A. No post-proof store mutation exists on any forward path, so §6.3's hardlink
re-promote and §6.4's receipt generalisation are both unnecessary. INV-PROVEN is
pinned by a property test over every file's mtime across a finalize_run."
```

---

## Task 3: `finalize_run` — the one path

**Files:**
- Create: `src/phenotypic/_cli/_cli_finalize_run.py`
- Test: `tests/unit/cli/test_finalize_run.py`

The seam already exists and is already shared: `finalize_post_master_outputs`
(`_cli_output_manager.py:969`) is called by both the forward path (`:1526`) and the
recompile worker (`_cli_recompile_worker.py:802`), whose own comment says it is "matching
the forward CLI path". What is **not** shared is aggregation. This task widens the seam to
own it.

- [ ] **Step 1: Write INV-INPUTS first — the phase's gate**

```python
def test_finalize_run_ignores_every_stale_intermediate(tmp_path):
    """INV-INPUTS / spec §7.5. Plant a stale chunk parquet, a stale shard, a stale
    _dataset_aggregated.parquet, a stale analysis_full.parquet and a stale master,
    each containing a row that exists in NO embedded table. Assert the new master
    matches a concat of the embedded tables exactly.

    Those files are outputs and intermediates of a PREVIOUS finalization, not inputs
    to this one. Under a rolling input, reusing any of them silently omits images
    that arrived since the cache was built, or retains rows for an image whose
    content changed and therefore has a new work_id.
    """
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import master_measurements_parquet_path

    _publish_two_successful_images(tmp_path)
    poison = pl.DataFrame({"Metadata_ImageFile": ["GHOST.tif"], "Shape_Circularity": [0.0]})
    _plant_stale_chunk_parquet(tmp_path, poison)
    _plant_stale_shard(tmp_path, poison)
    _plant_stale_dataset_aggregate(tmp_path, poison)
    _plant_stale_analysis_full(tmp_path, poison)
    _plant_stale_master(tmp_path, poison)

    finalize_run(tmp_path, dataset_names=["plate"])

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    assert "GHOST.tif" not in master["Metadata_ImageFile"].to_list()
    assert master.equals(_concat_of_embedded_tables(tmp_path))


def test_finalize_run_invalidates_the_intermediates_on_success(tmp_path):
    """§7.5: so a later invocation cannot mistake them for inputs."""
    from phenotypic._cli._cli_finalize_run import finalize_run

    _publish_two_successful_images(tmp_path)
    chunk = _plant_stale_chunk_parquet(tmp_path, _poison())
    finalize_run(tmp_path, dataset_names=["plate"])
    assert not chunk.exists()


def test_the_master_carries_no_user_metadata(tmp_path):
    """§7.3's contract change, stated as a test.

    The one genuinely dangerous failure mode in §7 is code that filters the master
    on a user-metadata column: it returns EMPTY rather than erroring. The schema
    version P7 stamps is what makes an old reader fail loudly instead."""
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import (
        master_measurements_parquet_path,
        measurements_parquet_path,
    )

    _publish_two_successful_images(tmp_path, metadata=True)
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    assert "Metadata_Strain" not in master.columns
    assert "Metadata_Strain" in mirror.columns


def test_curation_re_keying_still_works_against_the_intrinsic_master(tmp_path):
    """§7.3 names this as needing an explicit test rather than assumption.

    Curation deliberately reads the CLEAN master so labels survive for curated-out
    objects (_curation_labels.py:406). It keys on dataset / image / object-label --
    all intrinsic -- so it should be unaffected. Test it; do not assume it."""
    _publish_two_successful_images(tmp_path, metadata=True)
    _finalize_and_curate(tmp_path, curated_out=["a.tif::3"])
    assert _curated_label_survives(tmp_path, "a.tif::3")


def test_master_measurements_csv_is_gone(tmp_path):
    """D8: master is parquet-only. The un-joined master is no longer the file a
    human opens -- the mirror is."""
    from phenotypic._cli._cli_finalize_run import finalize_run

    _publish_two_successful_images(tmp_path)
    finalize_run(tmp_path, dataset_names=["plate"])
    assert not (tmp_path / "deliverables" / "master_measurements.csv").exists()
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement the six steps**

```python
def finalize_run(output_dir, *, dataset_names, pipeline=None, metadata_csv=None,
                 no_qc=False, study_config=None, shard_paths=None, commit_guard=None):
    """Aggregate, join, publish -- one path for `full`, `measure` and `recompile`.

    Six steps (spec §7.4, minus the backfill D-A cut):

    1. select marker-authorized embedded measurement tables
    2. concat  ->  master_measurements.parquet          (un-joined, D8: no CSV)
    3. join metadata + append metadata-only phantoms + apply post ops
    4. write  ->  deliverables/measurements.{parquet,csv}
    5. persist pipeline.json, analysis outputs, per-feature splits
    6. publish aggregate proof -> run proof

    INVARIANT (INV-INPUTS, §7.5) -- **step 1 selects exactly the marker-authorized
    embedded measurement tables.** It never reads a prior master, chunk parquet,
    measurement shard, ``analysis_full.parquet`` or ``_dataset_aggregated.parquet``
    as an aggregation input. Those are outputs and intermediates of a PREVIOUS
    finalization; under a rolling input, reusing one silently omits images that
    arrived since, or retains rows for an image whose content changed and therefore
    has a new ``work_id``. Master is a pure function of the currently authorized
    embedded tables -- which is the derivability property this whole design is for.

    ``shard_paths`` is P5's fan-out hook: when supplied, step 2 merges those instead
    of reading the tables directly. It does not weaken INV-INPUTS, because the shards
    were themselves produced from authorized embedded tables **in this invocation**,
    namespaced by ``scheduler_epoch`` so a prior run's shards can never be merged.
    """
```

> **Before rewriting anything in `finalize_post_master_outputs`, inventory what it already
> does.** Its docstring (`_cli_output_manager.py:1023-1050`) numbers **five** steps: the
> metadata handling this task changes, `_apply_post_to_master`, `_seed_measurements`, the
> per-feature splits, and the pipeline/analysis/QC block — plus the
> `order_measurement_columns` call at `:1104-1106` that is not in that numbered list at all.
> A rewrite that names only the step under discussion drops the others silently. That is not
> hypothetical: the column-ordering call was missing from this task until a reader asked
> whether it still existed.

Step 1 calls the existing `authorized_measurement_sources` (`_cli_completion.py:768`) —
already the right predicate, already marker-authorized, and **moved onto records by P3 Step
3b**; if that move was skipped it returns `{}` and this step writes an empty master with no
exception.

### Step 3 keeps metadata on the left, and deletes the other branch (CAN-1)

The first draft said *"Step 3 onward is `finalize_post_master_outputs`, unchanged."*
**It cannot be.** That function has exactly two branches
(`_cli_output_manager.py:1077-1086`), and after the inversion both lose something:

| Branch | Condition | Does | After the inversion |
|---|---|---|---|
| legacy | `metadata_join_keys is None` (`:1077`) | `join_metadata(working_df, metadata_csv, how="left")` (`:1081`) | **Correct — and it is the only branch that is.** Metadata on the left: joins user metadata onto every matched measured row **and** keeps metadata-unmatched rows as `QC_MetadataOnly` phantoms. Both halves, one call. |
| embedded | keys provided (`:1085`) | `_append_metadata_only_rows(...)` only | **Broken.** Its premise (`:1023-1026`) is *"Measured rows already carry their publication-time metadata from the embedded tables and are not joined again"* — which **P4 falsifies**. It appends phantoms and joins nothing, so every measured row's user metadata is null. It also raises `ValueError` at `:884-893` for any join key now absent from the master. |

> **The metadata-on-the-left orientation is deliberate and stays.** `join_metadata`'s
> docstring (`_cli_output_manager.py:143-153`) is explicit: *"a left join is asymmetric by
> design: it keeps metadata-unmatched rows but still drops measurement-unmatched rows"*, and
> *"Absence of a colony is data: a strain that failed to grow, or that detection missed, is
> exactly what the user needs to see."*
>
> A measured object whose key appears in no metadata row is an object outside the described
> experiment. Dropping it is the intended semantics, not a data-loss bug — **user ruling,
> round 2.** An earlier draft of this section proposed reversing the orientation so orphan
> measurements survived; that would have changed a deliberate scientific decision on the
> strength of a reviewer's framing. Reverted.

### The surviving branch has never run on a forward tree (flow-r3 C1)

`join_metadata` is the **legacy** branch — reached only when `metadata_join_keys is None`,
which on a modern tree it never is. Deleting the embedded branch promotes a code path that
has not executed on a forward run since embedded tables landed. Three behaviours it brings,
none of them wrong in isolation and all of them changes:

1. **It casts join keys to `String` unconditionally** (`:139-142`, "casts them to ``String``
   for a safe join"). The mirror's join-key dtype changes from whatever the measurements
   carried to `String`.
2. **Row order follows the metadata frame** (its docstring says so), not the master's. The
   mirror's row order changes.
3. **Under a heterogeneous master** — `diagonal_relaxed` concat over stores with differing
   columns — a key present in some stores and absent in others can drop measured rows the
   per-image joins kept.

None of this argues against the user's ruling; metadata stays the left frame. It argues that
**"reuse the existing call" is not the no-op it reads as**, and each behaviour needs a
pinned test rather than a discovery in production:

```python
def test_the_mirrors_join_key_dtype_and_row_order_are_pinned(tmp_path):
    """flow-r3 C1. join_metadata is the legacy branch and has not run on a forward
    tree since embedded tables landed. Promoting it changes dtype and row order --
    both observable by the GUI and by any user script reading the mirror."""
    import polars as pl

    from phenotypic.sdk_ import measurements_parquet_path

    _publish_two_successful_images(tmp_path, metadata=True, join_key_dtype=pl.Int64)
    _finalize(tmp_path)

    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    assert mirror.schema["Metadata_Well"] == pl.String, (
        "join_metadata casts join keys to String; if this changed, the GUI's "
        "filters and every downstream script keyed on the old dtype changed with it"
    )


def test_a_heterogeneous_master_loses_no_measured_row(tmp_path):
    """The dangerous third behaviour: a key present in some stores and absent in
    others, concatenated diagonal_relaxed, then joined globally."""
    import polars as pl

    from phenotypic.sdk_ import master_measurements_parquet_path, measurements_parquet_path

    _publish_image_with_columns(tmp_path, "a.tif", extra=["Grid_RowNum"])
    _publish_image_with_columns(tmp_path, "b.tif", extra=[])      # ragged
    _finalize(tmp_path)

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    measured = mirror.filter(pl.col("QC_MetadataOnly").fill_null(False).not_())
    assert set(measured["Metadata_ImageFile"]) == set(master["Metadata_ImageFile"])
```

### So step 3 is one existing call, not a new composite

The inversion does not need a third mode. It needs the **embedded branch deleted**:

```
step 3  =  join_metadata(master_df, metadata_csv, how="left")   →  post ops
```

That single call already does everything §7.4 step 3 asks for, in the orientation the
project intends:

- it **identifies the common columns itself** (`:139-142`), so it needs no recorded join
  keys — which is also why **CAN-2's `_consistent_embedded_join_keys` retirement falls out
  for free**: nothing downstream needs the stores' recorded keys, so their D-A-manufactured
  inconsistency stops being reachable rather than needing to be tolerated;
- it joins user metadata onto every matched measured row — the half the embedded branch lost
  once P4 made its premise false;
- it emits phantoms with `QC_MetadataOnly` — the half the embedded branch already had.

**Delete the `metadata_join_keys` parameter and its branch** (`:1077-1086`), and with it
`_consistent_embedded_join_keys`' two call sites (`:1435-1439`,
`_cli_recompile_worker.py:785`). The `measurement_sources`-vs-`metadata_join_keys` split in
`_run_post_master_steps` (`:777-787`) goes at the same time — §7.4 already predicted it
would, "because the two callers arrive with differently-shaped inputs", and after this they
arrive the same way.

Update the docstring at `:1023-1026`, which states the now-false premise, and
`_cli/CLAUDE.md`'s master-vs-mirror rules.

Test both halves **in one frame** — a measured row carrying a non-null user column, and a
phantom row present.

### Keep the column ordering — `join_metadata` returns metadata-first

`join_metadata`'s own docstring: *"Returns: DataFrame with metadata columns first, then
measurement columns … Row order follows the metadata frame."* What restores the canonical
frame shape is **`order_measurement_columns`** (`sdk_/_metadata_helpers.py:111`), applied at
`_cli_output_manager.py:1104-1106`:

```python
post_df = post_df.select(order_measurement_columns(post_df.columns))
```

**That call is inside the function this task rewrites and must survive.** An earlier draft
of this task specified step 3 without mentioning it at all, which is how a rewrite silently
drops a sibling step: `finalize_post_master_outputs` does five numbered things
(`:1023-1050`), and enumerating only the one under discussion is not a rewrite plan.

Canonical order is `[front metadata] → [measurements] → [IMAGE metadata] → [info block]`.
Verified against the real function on the two frames this change produces:

| Frame | Ordered columns |
|---|---|
| master (intrinsic only) | `Metadata_Dataset`, `Metadata_ImageFile`, `Shape_Circularity`, `Object_Label`, `Bbox_X` |
| mirror (joined) | `Metadata_Strain`, `Metadata_ImageFile`, `Shape_Circularity`, `QC_MetadataOnly`, `Object_Label`, `Bbox_X` |

Three things that follow, none of them obvious:

1. **The intrinsic identity columns are front-block, not trailing.** `Metadata_ImageFile`
   has `metadata_owner_for_header(...) is None` and `Metadata_Dataset` is `EXPERIMENT`-owned
   — **neither is `IMAGE`-owned**, so they lead the frame rather than trailing the
   measurements. §7.1's "intrinsic identity stays" therefore leaves the master's shape
   recognisable: identity, measurements, info block.
2. **The master is not ordered by this call**, because it is written before it. It inherits
   its order from the embedded tables, which the pipeline already orders through the same
   function (`_image_pipeline_core.py:1258,1275-1291`). Removing user metadata does not
   disturb the survivors' relative order — unknown-owner tags sort alphabetically at the end
   of the front block, so deleting some leaves the rest in place. **Assert that rather than
   assuming it.**
3. **`QC_MetadataOnly` sorts into the measurements block**, since it is not a metadata
   header, not the object label, and not `Bbox_`/`Grid_`. That is existing behaviour, it is
   **out of scope**, and it is recorded here only so a reviewer seeing it in the ordered
   mirror does not read it as a regression this change introduced.

```python
def test_the_mirror_keeps_canonical_column_order_after_the_join(tmp_path):
    """join_metadata returns metadata-first; order_measurement_columns restores the
    canonical shape. The call lives inside the function this phase rewrites."""
    import polars as pl

    from phenotypic.sdk_ import measurements_parquet_path, order_measurement_columns

    _publish_two_successful_images(tmp_path, metadata=True)
    _finalize(tmp_path)

    cols = pl.read_parquet(measurements_parquet_path(tmp_path)).columns
    assert cols == order_measurement_columns(cols), (
        "the mirror is not canonically ordered -- the order_measurement_columns "
        "call at _cli_output_manager.py:1106 was dropped in the rewrite"
    )


def test_the_master_inherits_canonical_order_from_the_embedded_tables(tmp_path):
    """The master is written BEFORE the ordering call, so it depends on its inputs
    already being ordered. The inversion removes columns from those inputs; assert
    that does not disturb the rest."""
    import polars as pl

    from phenotypic.sdk_ import master_measurements_parquet_path, order_measurement_columns

    _publish_two_successful_images(tmp_path, metadata=True)
    _finalize(tmp_path)

    cols = pl.read_parquet(master_measurements_parquet_path(tmp_path)).columns
    assert cols == order_measurement_columns(cols)
    assert cols[:2] == ["Metadata_Dataset", "Metadata_ImageFile"], (
        "intrinsic identity should lead the master -- neither column is IMAGE-owned, "
        "so both belong to the front block"
    )
```

**And `pht-metadata.parquet` gets the same treatment** (Task 2): order its columns with the
same function, so a third-party reader joining the two tables sees one convention rather
than two.

### Where the join keys come from (CAN-2)

`_consistent_embedded_join_keys` (`_cli_output_manager.py:914-966`) collects
`(metadata_snapshot_sha256, join_keys)` from every authorized embedded table and raises
`ValueError("Embedded measurement tables have mixed metadata digests or join keys")` at
`:962-965`. It is called unconditionally on the marker-authorized path (`:1435-1439`).

**D-A deliberately manufactures the state that trips it.** Stores keep the snapshot they
were built against, so any run that gains images after a `metadata.csv` edit has two
generations on disk, and the next aggregation aborts — while D-A's contract says divergence
is an advisory and *"an advisory is never a gate"*. This is a gate, in the finalizer, on the
normal rolling-input path.

**Retire it from the finalize path.** Derive the join keys once, from `metadata.csv` ∩
master columns; the stores' recorded keys become **provenance only**. That is what the
inversion implies: once the join is global, a per-store record of how *that store* was
joined is history, not input.

**The late-metadata case is the dangerous one, because it looks like it works.** A run with
no metadata records `join_status="not_requested"`, digest `""`, keys `()`
(`_embedded_measurement_tables.py:55-62`). Add `metadata.csv` and re-run: the recorded keys
are `()` — which is **not `None`** — so finalize takes the append-phantoms branch with an
empty key tuple and **joins no measured row at all**. Every measured row's user metadata is
null; the phantoms carry the column, so a membership assertion passes.

This **deletes recompile's separate master-merge** and collapses the `measurement_sources`
vs `metadata_join_keys` branch in `_run_post_master_steps`
(`_cli_recompile_worker.py:777-787`), which exists only because the two callers arrive with
differently-shaped inputs. After this task they arrive the same way.

- [ ] **Step 3b: Add the two tests that catch CAN-1 and CAN-2**

```python
def test_the_mirror_carries_both_joined_rows_and_phantoms(tmp_path):
    """CAN-1. Neither existing branch does both halves: one joins and drops every
    phantom, the other appends phantoms and joins nothing. Assert them in ONE
    frame, because each half passes a test that only looks at the other."""
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import measurements_parquet_path

    _publish_two_successful_images(tmp_path, metadata=True)
    _add_a_metadata_only_row(tmp_path, well="Z99")
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))

    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    measured = mirror.filter(pl.col("QC_MetadataOnly").fill_null(False).not_())
    phantoms = mirror.filter(pl.col("QC_MetadataOnly").fill_null(False))

    assert measured.height > 0 and phantoms.height == 1
    assert measured["Metadata_Strain"].null_count() == 0, "measured rows were not joined"
    assert "Z99" in phantoms["Metadata_Well"].to_list(), "phantoms were dropped"


def test_a_measured_row_absent_from_metadata_is_dropped_deliberately(tmp_path):
    """The asymmetry is by design (user ruling, round 2), so PIN it rather than
    leaving it as an accident of which frame is on the left.

    metadata.csv describes the experiment. A measured object whose key appears in
    no metadata row is an object outside that description, and `join_metadata`'s
    docstring states the intent: it keeps metadata-unmatched rows -- "a strain that
    failed to grow, or that detection missed, is exactly what the user needs to
    see" -- and drops measurement-unmatched ones.

    This test exists because an earlier draft proposed reversing the orientation.
    Without it, a future reader sees only "left join" and cannot tell which way
    round was intended.
    """
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import measurements_parquet_path

    _publish_two_successful_images(tmp_path, metadata=True)
    _add_an_object_whose_key_is_absent_from_metadata(tmp_path, image="b.tif", label=7)
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))

    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    orphan = mirror.filter(
        (pl.col("Metadata_ImageFile") == "b.tif") & (pl.col("Object_Label") == 7)
    )
    assert orphan.height == 0, (
        "an object outside the described experiment reached the mirror; the join "
        "orientation was reversed"
    )


def test_the_master_keeps_the_object_the_mirror_drops(tmp_path):
    """Where the dropped object DOES survive, and why that is the right split.

    §7.3: the master is the un-joined archival set -- intrinsic identity, every
    authorized measured row. The mirror is the post-applied, metadata-joined display
    frame. So an object outside the experiment is preserved in the master and absent
    from the mirror, which is exactly the master/mirror distinction CLAUDE.md's
    "feed analysis and dashboards from the mirror, not the master" rule rests on.
    """
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import master_measurements_parquet_path

    _publish_two_successful_images(tmp_path, metadata=True)
    _add_an_object_whose_key_is_absent_from_metadata(tmp_path, image="b.tif", label=7)
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    kept = master.filter(
        (pl.col("Metadata_ImageFile") == "b.tif") & (pl.col("Object_Label") == 7)
    )
    assert kept.height == 1, "the master must retain every authorized measured row"


def test_metadata_added_after_the_stores_still_joins_every_measured_row(tmp_path):
    """CAN-2, with DF-2's assertion verbatim.

    The `measured.height > 0` guard matters: without it the assertion is vacuously
    true on an all-phantom frame, which is the failure mode the first draft's
    version already had.
    """
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import measurements_parquet_path

    _publish_two_successful_images(tmp_path, metadata=False)   # keys recorded as ()
    _add_metadata_csv(tmp_path)
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))

    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    measured = mirror.filter(pl.col("QC_MetadataOnly").fill_null(False).not_())
    assert measured.height > 0, "fixture produced no measured rows to check"
    assert "Metadata_Strain" in measured.columns
    assert measured["Metadata_Strain"].null_count() == 0, (
        "user metadata reached the mirror only as metadata-only phantoms; every "
        "measured row is null. The join keys were () rather than None, so "
        "finalize took the append-phantoms branch and joined nothing."
    )


def test_stores_with_mixed_metadata_snapshots_do_not_abort_finalization(tmp_path):
    """CAN-2. D-A manufactures this state on the normal rolling-input path; the
    kept code raises on it."""
    _publish_two_successful_images(tmp_path, metadata=True)
    _edit_metadata_csv(tmp_path)
    _publish_one_more_image(tmp_path, metadata=True)    # different snapshot digest
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))
    # must not raise; divergence is an advisory, per D-A
```

- [ ] **Step 4: Run the tests.** Expected: PASS (8 passed).

- [ ] **Step 5: Prove INV-INPUTS can fail**

Add a `_dataset_aggregated.parquet` fast path to step 1 — the shape
`_aggregate_measurements_unlocked`'s docstring describes today ("Prefers pre-aggregated
`_dataset_aggregated.parquet` files when available"). Confirm
`test_finalize_run_ignores_every_stale_intermediate` fails, then remove it. **That fast
path is exactly the bug INV-INPUTS forbids, and it is in the current code.**

- [ ] **Step 6: Mint the master's schema stamp HERE, and give it a reader**

Two round-1 findings meet at this step.

**CAN-9 — the stamp belongs where the shape is produced.** The first draft stamped
`phenotypic.master_schema_version = "2"` during `--mode migrate`, which explicitly does
*not* re-run finalization — so the stamped file was still the legacy metadata-joined
master, and the stamp certified a shape the file would not have until the next
`finalize_run`. Its own two tests contradicted each other. Minting it here means the stamp
and the shape come out of one code path and cannot disagree; P7 then leaves a legacy master
**unstamped**, which correctly identifies it as pre-v2.

**U-3 — a stamp with no reader is the pattern P6 deletes.** §7.3 justifies it as making "an
old reader fail loudly", which a Parquet KV key cannot do: `pd.read_parquet`,
`pl.read_parquet` and `pq.read_table().to_pandas()` ignore KV metadata and raise nothing.
Meanwhile P6 deletes `DashboardManifestKey.VERSION` *as a finding* for being written at one
site and read at zero. Shipping a new instance of that pattern while deleting the old one
is indefensible.

```python
# phenotypic.sdk_._master_io

MASTER_SCHEMA_VERSION: Final[int] = 2

def read_master_measurements(output_dir: Path) -> "pl.DataFrame":
    """Read the master, refusing one written under a different schema.

    §7.3 calls a silent empty result "the one genuinely dangerous failure mode in
    §7": after the inversion, code filtering the master on a user-metadata column
    gets nothing back rather than an error. The stamp is what turns that into a
    loud failure -- **for readers that check it**, which is every reader inside
    this repository and none outside it. §7.3 is corrected to claim exactly that
    (U-3); do not restate the stronger claim.

    Raises:
        ValueError: the master carries no schema stamp (pre-v2, needs
            ``--mode migrate``) or an unrecognized one.
    """
```

Route every in-repo master read through it — `grep -rn 'master_measurements_parquet_path'
src/` and convert each call site.

```python
def test_an_unstamped_master_is_refused_loudly(tmp_path):
    """U-3. A pre-v2 master read by post-v2 code must raise, not return a frame
    whose user-metadata columns silently vanished."""
    import pytest

    from phenotypic.sdk_._master_io import read_master_measurements

    _write_legacy_joined_master(tmp_path)          # no stamp
    with pytest.raises(ValueError, match="migrate"):
        read_master_measurements(tmp_path)


def test_the_stamp_and_the_shape_are_minted_together(tmp_path):
    """CAN-9. There is no window in which a stamped master has the wrong shape."""
    import pyarrow.parquet as pq

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import master_measurements_parquet_path

    _publish_two_successful_images(tmp_path, metadata=True)
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))

    path = master_measurements_parquet_path(tmp_path)
    assert pq.read_schema(path).metadata[b"phenotypic.master_schema_version"] == b"2"
    assert "Metadata_Strain" not in pq.read_schema(path).names
```

- [ ] **Step 7: Publish `source_set_digest` in the run proof (U-4)**

`publication_id` is cut. It was `sha256(source_set_digest ‖ finalization_inputs)` — a pure
function of two values the binding check already compares — so the aggregate↔run binding is
stated **directly** instead of through an opaque hash. `finalize_run` step 6 publishes
`source_set_digest` and `source_image_count` into **both** proofs, and P1's rule 1 compares
them (CAN-4).

`source_set_digest` had no home in any phase before this step — it appeared only in the
README's digest table and two prose mentions in P5. This is that home.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/_cli/_cli_finalize_run.py src/phenotypic/sdk_/_master_io.py \
        tests/unit/cli/test_finalize_run.py
git commit -m "feat(cli): finalize_run -- one aggregation and publication path

Spec §7.4, §7.5, six steps (D-A cut the backfill). Step 3 is join_metadata(how="left") --
the one call that already does both halves in the orientation the project intends.
The embedded branch is deleted: P4 falsified its premise that measured rows already
carry their metadata (CAN-1). join_metadata identifies its own common columns, so
the stores' recorded join keys -- which D-A deliberately makes inconsistent -- stop
being read at all rather than needing to be tolerated (CAN-2). The master schema stamp is minted here so
the stamp and the shape come from one code path (CAN-9), and read_master_measurements
gives it the reader U-3 requires. source_set_digest is published into both proofs,
replacing the cut publication_id (U-4).

INV-INPUTS was confirmed to fail when the _dataset_aggregated.parquet fast path the
current aggregator documents is reintroduced."
```

---

## Task 4: Route all three entry points through `finalize_run`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:1351`, `:1545`
- Modify: `src/phenotypic/_cli/_cli_recompile_worker.py:764`
- Test: `tests/unit/cli/test_finalize_run.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("mode", ["full", "measure", "recompile"])
def test_every_mode_produces_a_byte_identical_master(tmp_path, mode):
    """§7.4: recompile becomes 'call finalize_run again', not a parallel
    implementation that must be kept in sync. Three modes, one master."""
    output = _run_mode(tmp_path, mode)
    assert _master_bytes(output) == _master_bytes(_run_mode(tmp_path / "ref", "full"))


def test_process_mode_skips_finalization_entirely(tmp_path):
    """§7.4's table: `process` writes one layer, no measurement, and
    process_only_layer already short-circuits the aggregate proof."""
    output = _run_mode(tmp_path, "process")
    assert not (output / "deliverables" / "master_measurements.parquet").exists()
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`_aggregate_measurements_unlocked` keeps its lock (`aggregate_measurements`'s
`.aggregate_publication.lock`, `_cli_output_manager.py:1552`) and delegates its body.
`_run_post_master_steps` becomes a `finalize_run` call, keeping its
`generation_publication_guard` wrapper.

- [ ] **Step 4: Delete the D8 surfaces**

`MASTER_MEASUREMENTS_CSV`, `master_measurements_csv_path()`, `BundleLayout.master_csv`,
`load_master_measurements()`, and the `master_csv` entry in the aggregate proof's
`required_outputs` (`_cli_completion.py:888`). The proof's `required_outputs` drops from
four artifacts to three.

Per [Q6](OPEN-QUESTIONS.md#q6-ten-test-files-depend-on-master_measurements_csv_path), ten
test files reference `master_measurements_csv_path`. Fix each: assert on the parquet, or on
`measurements.csv` where the test genuinely wanted a human-readable file.
`BundleLayout.detect` keys on `master_measurements.parquet` already, so bundle detection is
unaffected.

- [ ] **Step 5: Phase gate**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix src/phenotypic/_cli/_cli_finalize_run.py \
  src/phenotypic/_cli/_cli_output_manager.py src/phenotypic/_cli/_cli_recompile_worker.py \
  src/phenotypic/_cli/_embedded_measurement_tables.py src/phenotypic/sdk_/ tests/unit/
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit -q
```

This is the first phase where the full `tests/unit` suite is warranted rather than a
selection — the master's shape changed and it is read almost everywhere. **The suite is
~65 minutes and is a Slurm job**: use the **`run-phenotypic-test`** and **`slurm-job`**
skills, with the committed script at
`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`. Never
`-n auto` (it reads node cores, not the allocation) and never `-x` (it truncates a run that
then gets recorded as a baseline).

- [ ] **Step 6: Update the docs the contract change invalidates**

- `CLAUDE.md`'s "Output layout (`deliverables/`)" bullet: `master_measurements.*` is now
  `master_measurements.parquet`, un-joined and intrinsic-only.
- `src/phenotypic/_cli/CLAUDE.md`'s master-vs-mirror rules.
- `docs/source/how_to/pages/` wherever the master is described as metadata-joined.

- [ ] **Step 7: Commit**

```bash
git add -A src/phenotypic tests docs CLAUDE.md
git commit -m "refactor(cli): route full, measure and recompile through finalize_run

Spec §7.3, §7.4, D8. Deletes recompile's separate master-merge and the
measurement_sources/metadata_join_keys branch that existed only because the two
callers arrived with differently-shaped inputs. Master is parquet-only and carries
intrinsic identity only; the mirror carries the join."
```

---

## Task 5: Verify the promote-time metadata end to end

**Files:**
- Test: `tests/integration/` (a real single-image run)

- [ ] **Step 1: Run a real local run with `--metadata` and assert the store is self-describing**

```python
def test_a_real_run_leaves_stores_a_third_party_can_join(tmp_path):
    """D-A's whole justification: the store is self-describing WITHOUT any post-hoc
    rewrite. Read it back with plain pyarrow -- no phenotypic import in the assertion
    path -- and join it, the way a napari or QuPath user would."""
    import pyarrow.parquet as pq

    output = _run_full_pipeline(tmp_path, metadata=True)
    store = next(output.rglob("*.ome.zarr"))

    measurements = pq.read_table(store / "tables" / "measurements" / "table.parquet")
    metadata = pq.read_table(store / "tables" / "metadata" / "pht-metadata.parquet")
    keys = json.loads(metadata.schema.metadata[b"phenotypic.join.keys"])
    joined = measurements.to_pandas().merge(metadata.to_pandas(), on=keys, how="left")
    assert "Metadata_Strain" in joined.columns
```

- [ ] **Step 2: Run it.** Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration
git commit -m "test(cli): a promoted store is joinable by a third party with pyarrow alone

D-A. The assertion path imports no phenotypic code, which is the property that makes
'self-describing' mean something."
```
