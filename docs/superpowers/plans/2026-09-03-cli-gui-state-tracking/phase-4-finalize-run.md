# Phase 4 — Embedded-table inversion and `finalize_run`

**Depends on:** P3, P0 (S-4). **Blocks:** P5–P7.

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
instead, so no artifact carrying a content proof is ever mutated (**INV-IMMUTABLE**).

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
| **Modify** `src/phenotypic/sdk_/_measurement_tables.py` | Write both tables into the `.part`; extend the root `tables` block. |
| **Create** `src/phenotypic/_cli/_cli_finalize_run.py` | `finalize_run(output_dir, …)` — the one path. ~260 lines. |
| **Modify** `src/phenotypic/_cli/_cli_output_manager.py:1351` | `_aggregate_measurements_unlocked` delegates aggregation to `finalize_run`. |
| **Modify** `src/phenotypic/_cli/_cli_recompile_worker.py:764` | `_run_post_master_steps` becomes a `finalize_run` call. |
| **Modify** `src/phenotypic/_cli/_cli_completion.py:868` | Aggregate proof's `required_outputs` drops `master_csv` (D8). |
| **Delete** | `MASTER_MEASUREMENTS_CSV`, `master_measurements_csv_path()`, `BundleLayout.master_csv`, `load_master_measurements()` (D8). |
| **Test** `tests/unit/cli/test_finalize_run.py` *(new)* | INV-INPUTS, the six steps, the three entry points. |
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

Implements **D-A** and **INV-IMMUTABLE**.

- [ ] **Step 1: Write the failing tests**

```python
def test_both_tables_land_in_the_same_part_before_the_root(tmp_path):
    """D-A / INV-IMMUTABLE. The root zarr.json is written last and is the record's
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


def test_nothing_writes_into_a_store_after_its_record_is_published(tmp_path):
    """INV-IMMUTABLE, as a property test rather than a convention.

    Publish a record, snapshot every mtime under the store, run finalize_run, and
    assert not one file moved. This is the test that would have caught the backfill
    if it had shipped."""
    from phenotypic._cli._cli_finalize_run import finalize_run

    store = _publish_a_successful_image(tmp_path, dataset="plate", stem="a")
    before = {p: p.stat().st_mtime_ns for p in sorted(store.rglob("*")) if p.is_file()}
    finalize_run(tmp_path, dataset_names=["plate"])
    after = {p: p.stat().st_mtime_ns for p in sorted(store.rglob("*")) if p.is_file()}
    assert before == after, (
        "finalize_run mutated a store that already carries a content proof; "
        "INV-IMMUTABLE forbids it and D-A removed the only mechanism that did"
    )
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

Extend the `.part` writer to emit `tables/metadata/pht-metadata.parquet` when
`prepared.metadata is not None`, before `OME/zarr.json` and the root. Extend the root's
`attributes.phenotypic.tables` with a `metadata` descriptor, and add
`attributes.phenotypic.metadata = {"snapshot_sha256": …, "join_keys": [...],
"join_kind": …}`.

The Parquet KV keys ride along unchanged (§7.2): `phenotypic.join.keys`,
`phenotypic.join.kind`, `phenotypic.metadata.snapshot_sha256`. The join is self-describing
from the file itself, which is the property that makes the store useful to a third party at
all.

**Order is load-bearing and is the whole of INV-IMMUTABLE:** chunks → both tables →
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
re-promote and §6.4's receipt generalisation are both unnecessary. INV-IMMUTABLE is
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

Step 1 calls the existing `authorized_measurement_sources`
(`_cli_completion.py:768`) — already the right predicate, already marker-authorized. Step 3
onward is `finalize_post_master_outputs`, unchanged.

This **deletes recompile's separate master-merge** and collapses the `measurement_sources`
vs `metadata_join_keys` branch in `_run_post_master_steps`
(`_cli_recompile_worker.py:777-787`), which exists only because the two callers arrive with
differently-shaped inputs. After this task they arrive the same way.

- [ ] **Step 4: Run the tests.** Expected: PASS (5 passed).

- [ ] **Step 5: Prove INV-INPUTS can fail**

Add a `_dataset_aggregated.parquet` fast path to step 1 — the shape
`_aggregate_measurements_unlocked`'s docstring describes today ("Prefers pre-aggregated
`_dataset_aggregated.parquet` files when available"). Confirm
`test_finalize_run_ignores_every_stale_intermediate` fails, then remove it. **That fast
path is exactly the bug INV-INPUTS forbids, and it is in the current code.**

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli/_cli_finalize_run.py tests/unit/cli/test_finalize_run.py
git commit -m "feat(cli): finalize_run -- one aggregation and publication path

Spec §7.4, §7.5, six steps (D-A cut the backfill). INV-INPUTS was confirmed to fail
when the _dataset_aggregated.parquet fast path the current aggregator documents is
reintroduced."
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
