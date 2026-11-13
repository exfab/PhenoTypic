# P3 · Per-pipeline tables (`tables/` v2)

**Spec:** Part D (D1–D10)

---

**Goal:** Each store holds one folder per measuring pipeline,
`tables/<n>-<sha12>/`, and every writer carries the other folders across.

**Approach:** All 27 modules that touch today's flat layout reach a run folder
through one resolver. That makes the change a mechanical substitution at each
call site, not 27 separate path joins. v1 stores are converted by
`--mode migrate`, and forward modes refuse them.

**Folder numbers** are a store-local call-order counter (spec D2). A new folder
gets `1 +` the highest number among the store's run folders and its `upstream`
entries. The journal's `sequence` is never used: it is `len(applications) + 1`
of one image's journal (`_core/_provenance.py:374`), and a forward run builds
that journal from its input (`_cli_process_single.py:300-321`), so over a raw
input it is always `1`.

**Which folder a writer owns** (spec D6): an existing folder with the writer's
pipeline sha256 is replaced, keeping its number, with every file rewritten as a
new inode. Otherwise the writer adds a folder with the next number. A write with
no known sha replaces the highest-numbered `-unknown` folder.

**Which folder a reader uses** (spec D6a): finalize, recompile and measure
aggregate, per image, the folder that image's completion marker certifies.
Readers without a record default to the highest-numbered folder that holds
`measurements`.

**GUI:** the P3 rows of the README's expected-red table go red in Task 2 and
stay red until #283. `MEASUREMENT_TABLE_RELATIVE_PATH` stays defined as the v1
path.

### Task 1: v1 fixtures, run identity and the resolver

**Files:**
- Create: `tests/fixtures/tables_v1/_generate.py`, and the fixtures it writes:
  - `v1_run/`, a complete one-image full-run output tree (deliverables, records,
    run proof), written by the CLI at this task's commit;
  - `v1_store_null_pipeline.ome.zarr/`, a copy of `v1_run`'s store whose
    journal holds one `legacy` application with `pipeline: null`.
- Create: `src/phenotypic/sdk_/_table_runs.py`
- Modify: `src/phenotypic/sdk_/ngff_.py:78-100`. Add `TABLES_SCHEMA_VERSION = 2`
  and `EMBEDDINGS_TABLE_GROUP = "embeddings"`. Keep `MEASUREMENT_TABLE_RELATIVE_PATH`
  as the **v1** legacy path; the GUI imports it (#283).
- Modify: `src/phenotypic/_core/_provenance.py`. Add
  `current_pipeline_sha256(image) -> str | None`.
- Modify: `src/phenotypic/sdk_/__init__.py`. Export the resolver names; add them
  to `_LAZY_ATTRS` if they import pandas.
- Create (helpers), in `tests/_tiny_run.py`:
  - fixtures `v1_tree` (a `tmp_path` copy of `v1_run/`), `v1_store` (that
    copy's one store) and `v1_store_legacy_null_pipeline` (a `tmp_path` copy)
  - `v2_block(*, runs: Mapping[str, Mapping] = {}, upstream: Mapping[str, Mapping] = {}) -> dict`:
    a v2 `attributes.phenotypic` block with those entries
- Test: `tests/unit/sdk_/test_table_runs.py` (new)

**Interfaces:**
- Produces:
  - `TABLE_RUN_ID_RE = re.compile(r"^(?P<order>[1-9][0-9]*)-(?P<sha>[0-9a-f]{12}|unknown)$")`
  - `table_run_id(order: int, pipeline_sha256: str | None) -> str`
    (`None` → `"<order>-unknown"`)
  - `@dataclass(frozen=True) class TableRun`:
    - fields: `store: Path`, `run_id: str`, `order: int`,
      `pipeline_sha256: str | None`, `descriptor: Mapping[str, object]`
    - `has_measurements: bool` (the entry has a `measurements` block)
    - `measurements_path() -> Path` and `metadata_path() -> Path`
    - `embedding_path(model: str, region: Literal["crops", "grid"]) -> Path`
  - `store_tables_schema(store: Path) -> Literal[0, 1, 2]`: 0 means no tables
  - `list_table_runs(store: Path) -> tuple[TableRun, ...]`, ascending `order`;
    `()` for schema 0 or 1. Two runs with one `order` raise `ValueError`: the
    counter makes that state unreachable, so it means a corrupt descriptor.
  - `latest_table_run(store) -> TableRun | None`: the highest `order`
  - `latest_measurement_run(store) -> TableRun | None`: the highest `order`
    with `has_measurements` (the record-less reader default, D6a)
  - `table_run_for_pipeline(store, pipeline_sha256: str) -> TableRun | None`
  - `next_table_run_order(block: Mapping) -> int`: `1 +` the highest order among
    `runs` and the `<n>` of each `upstream` key; `1` when there are none
  - `choose_table_run_id(block: Mapping, *, pipeline_sha256: str | None) -> str`:
    1. a run with this sha exists → its id;
    2. `pipeline_sha256 is None` and an `-unknown` run exists → the
       highest-numbered one;
    3. otherwise → `table_run_id(next_table_run_order(block), pipeline_sha256)`.
  - `current_pipeline_sha256(image) -> str | None`: the last application's
    `pipeline["sha256"]`; `None` for an empty journal
    (`Image(arr=…)`, `_provenance.py:122-131`), a programmatic application
    (`pipeline=None`, `:660-666`) or a converted `legacy` one
  - `store_of_table_path(path: Path) -> Path`: the nearest ancestor ending in
    `.ome.zarr`. It replaces the `parents[len(MEASUREMENT_TABLE_RELATIVE_PATH.parts) - 1]`
    arithmetic in `_cli_parquet_agg.py:240` and `_cli_finalize_fanout.py:523`.

- [ ] **Step 1: Generate and commit the v1 fixtures.** After P3 T2 no code
  writes the flat layout, so these are captured now.
  - `_generate.py` follows `tests/fixtures/legacy_hdf/_generate.py`: a module
    docstring naming every fixture and what it pins, runnable as
    `uv run python tests/fixtures/tables_v1/_generate.py`.
  - `v1_run/` comes from `python -m phenotypic` over one
    `load_synth_yeast_plate()` image with the `tiny_pipeline`. The docstring
    names this task's commit and says the tree is rebuilt only from it.
  - `v1_store_null_pipeline` is edited from that store's root: one `legacy`
    application, `pipeline: null`.
  - The generator copies `v1_run/` to a temporary directory and asserts
    `resolve_run_state(copy, depth="deep").completion == "complete"`, so no
    absolute path in `.phenotypic/` survives the move. Keep the tree under
    5 MB.
- [ ] **Step 2: Write the failing tests**

```python
def test_run_id_format():
    assert table_run_id(2, "a71e09c2b4f5" + "0" * 52) == "2-a71e09c2b4f5"
    assert table_run_id(3, None) == "3-unknown"
    assert TABLE_RUN_ID_RE.match("0-abc") is None

def test_choose_reuses_the_same_pipeline_folder():
    block = v2_block(runs={"2-" + SHA_A[:12]: {"order": 2, "pipeline_sha256": SHA_A}})
    assert choose_table_run_id(block, pipeline_sha256=SHA_A) == "2-" + SHA_A[:12]

def test_a_new_pipeline_gets_the_next_number():
    block = v2_block(runs={"1-" + SHA_A[:12]: {"order": 1, "pipeline_sha256": SHA_A},
                           "2-" + SHA_B[:12]: {"order": 2, "pipeline_sha256": SHA_B}})
    assert choose_table_run_id(block, pipeline_sha256=SHA_C) == "3-" + SHA_C[:12]

def test_upstream_entries_count_toward_the_next_number():
    block = v2_block(upstream={"1-" + SHA_P[:12]: {"pipeline_sha256": SHA_P}})
    assert choose_table_run_id(block, pipeline_sha256=SHA_A) == "2-" + SHA_A[:12]

def test_an_unknown_sha_replaces_the_highest_unknown_folder():
    block = v2_block(runs={"1-unknown": {"order": 1, "pipeline_sha256": None},
                           "2-" + SHA_A[:12]: {"order": 2, "pipeline_sha256": SHA_A},
                           "3-unknown": {"order": 3, "pipeline_sha256": None}})
    assert choose_table_run_id(block, pipeline_sha256=None) == "3-unknown"
    assert choose_table_run_id(v2_block(), pipeline_sha256=None) == "1-unknown"

def test_two_runs_with_one_number_are_refused(tmp_path):
    store = write_root_block(tmp_path, v2_block(runs={"1-" + SHA_A[:12]: {"order": 1},
                                                      "1-" + SHA_B[:12]: {"order": 1}}))
    with pytest.raises(ValueError, match="order"):
        list_table_runs(store)

def test_an_empty_journal_has_no_pipeline_sha(tiny_pipeline_file):
    image = Image(arr=load_synth_yeast_plate().rgb[:])
    assert current_pipeline_sha256(image) is None
    initialize_cli_provenance(image, tiny_pipeline_file, kind="full", input_filename="p.tif")
    assert current_pipeline_sha256(image) == file_sha256(tiny_pipeline_file)

def test_store_of_table_path_handles_both_depths(tmp_path):
    s = tmp_path / "x.ome.zarr"
    assert store_of_table_path(s / "tables/measurements/table.parquet") == s
    assert store_of_table_path(s / "tables/2-a71e09c2b4f5/measurements/table.parquet") == s

def test_the_v1_fixtures_are_v1(v1_store, v1_store_legacy_null_pipeline):
    assert store_tables_schema(v1_store) == 1 == store_tables_schema(v1_store_legacy_null_pipeline)
```

  `SHA_A`, `SHA_B`, `SHA_C`, `SHA_P` are distinct 64-hex constants;
  `write_root_block(dir, block) -> Path` is a module-local helper that writes a
  store root holding `block`.
- [ ] **Step 3: Run them and confirm they fail.**
- [ ] **Step 4: Implement.** This is pure parsing of the root `zarr.json`
  block. Read it with the existing reader in `_measurement_tables.py:871`.
- [ ] **Step 5: Run again; it passes.**
- [ ] **Step 6: Commit** `_generate.py`, the fixture trees, and the paths
  above.

### Task 2: The writer, the carry step and stale binding

**Files:**
- Modify: `src/phenotypic/sdk_/_measurement_tables.py`:
  - :197-260, `write_embedded_measurement_table` and `write_metadata_table`
    write under `tables/<run>/`
  - :347-432, `build_image_tables_attributes`, `apply_image_tables_attributes`
    and `write_image_tables` produce the v2 block; `write_image_tables`
    delegates to `write_run_entry`
  - :141-149, :433-600, the contract validators read v2. They compare group
    documents with `_TABLES_GROUP_DOCUMENT` by exact equality (:141-149,
    :468-472, :539-542); a `measurements/` group's expected document is that
    document plus `ngff_feature_table_attrs(...)`.
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py:1451-1497`,
  the `write_image_tables` call (:1453-1459), the carry beside
  `carry_figure_runs` (:1470-1472), and the root block. `save2zarr`/`_save_store`
  gain `upstream: Mapping[str, Mapping] | None = None` (Task 6 passes it).
- Create (helpers), in `tests/_tiny_run.py`:
  - fixtures `pipeline_file_a` (the `tiny_pipeline`) and `pipeline_file_b`
    (the `tiny_pipeline` plus `MeasureShape()`); `SHA_A`/`SHA_B` in tests are
    their `file_sha256`. Use `pipeline_file_a` as `run_cli`'s default pipeline
    from here on.
  - fixture `measured_image`: `load_synth_yeast_plate()` with `OtsuDetector`
    applied and `initialize_cli_provenance(image, pipeline_file_a, kind="full", input_filename="plate.tif")`
  - `tables_for(image) -> PreparedImageTables`: the tables the forward worker
    builds for it, with an added `Pipeline_Marker` column naming the journal's
    pipeline file
  - `tree_digest(path) -> str`: sha256 over sorted relative paths and bytes
  - `resave_with_pipeline(store, pipeline_file, *, tables: bool = True)`: loads
    the store with `Image.load_zarr`, opens a `full` application for
    `pipeline_file`, and saves it back with or without tables
  - fixture `store_with_two_runs`: `measured_image` saved with A's tables, then
    `resave_with_pipeline(…, pipeline_file_b)`
  - fixture `store_with_crop_entry`: a measured store whose run entry holds a
    hand-built crop embedding entry bound to its objmap, written through
    `write_run_entry`'s `embeddings=` raw-entry seam
  - `resave_with_objmap(store, *, zeros: bool)` and `crop_entry(store) -> dict`
- Test: `tests/unit/sdk_/test_tables_v2_writer.py` (new)

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `write_run_entry(part: Path, run_id: str, *, order: int, pipeline_sha256: str | None, measurements: PreparedImageTables | None = None, objmap_target: str | None = None, embeddings: Mapping[str, Mapping] | None = None, objmap_sha256: str | None = None, array_sha256: Mapping[str, str] | None = None) -> dict`.
    The single owner of a run entry: it writes the run folder's tables into the
    part and returns the entry. Every subtable is optional, so a store with
    embeddings and no measurements (a process store, P5) has an entry too.
    In this task `embeddings` takes already-written raw entries only (the
    crop-entry fixture); P4 T2 adds the payload writer.
  - `carry_table_runs(final: Path, part: Path, *, exclude: str | None) -> dict`.
    It hard-links every run folder other than `exclude` into the part, and
    returns their entries plus `upstream`.
  - `refresh_embedding_staleness(runs: dict, *, objmap_sha256: str | None, array_sha256: Mapping[str, str]) -> None`.
    It sets each embedding entry's `stale` from its `binds` (D7). `array_sha256`
    is keyed by the entry's `input.series`.
  - `objmap_digest(objmap: np.ndarray) -> str`: sha256 over the dtype string,
    the shape and the C-order bytes of the objmap **as the store writes it**
    (after the writer's dtype cast), so one set of labels has one digest.
  - `ngff_feature_table_attrs(objmap_rel_path: str) -> dict`, the ngio
    attributes on each `measurements/` group (D4):
    `{"type": "feature_table", "table_version": "1", "region": {"path": ...},
    "backend": "parquet", "index_key": "Object_Label", "index_type": "int",
    "instance_key": "Object_Label"}`

- [ ] **Step 1: Write the failing tests**

```python
def test_save_writes_run_folder_and_v2_descriptor(tmp_path, measured_image, pipeline_file_a):
    store = measured_image.save2zarr(tmp_path / "p.ome.zarr", measurement_table=tables_for(measured_image))
    run = latest_table_run(store)
    assert run.run_id == f"1-{file_sha256(pipeline_file_a)[:12]}" and run.measurements_path().is_file()
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))["attributes"]["phenotypic"]["tables"]
    assert root["schema_version"] == 2 and list(root["runs"]) == [run.run_id]
    assert root["runs"][run.run_id]["order"] == 1

def test_a_programmatic_save_is_unknown_and_resaves_into_one_folder(tmp_path, measured_image):
    image = Image(arr=measured_image.rgb[:])                  # empty journal
    store = image.save2zarr(tmp_path / "q.ome.zarr", measurement_table=tables_for(measured_image))
    image.save2zarr(store, measurement_table=tables_for(measured_image))
    assert [r.run_id for r in list_table_runs(store)] == ["1-unknown"]

def test_a_second_pipeline_gets_folder_2(tmp_path, measured_image, pipeline_file_b):
    store = measured_image.save2zarr(tmp_path / "p.ome.zarr", measurement_table=tables_for(measured_image))
    resave_with_pipeline(store, pipeline_file_b)
    assert [r.order for r in list_table_runs(store)] == [1, 2]

def test_resave_carries_other_runs_byte_identical(store_with_two_runs, pipeline_file_b):
    first = list_table_runs(store_with_two_runs)[0]
    before = tree_digest(store_with_two_runs / "tables" / first.run_id)
    resave_with_pipeline(store_with_two_runs, pipeline_file_b)
    assert tree_digest(store_with_two_runs / "tables" / first.run_id) == before

def test_table_less_save_keeps_its_own_folder(store_with_two_runs, pipeline_file_b):
    # Review Focus 2: the initial-checkpoint and Stage-1 saves write no tables
    own = table_run_for_pipeline(store_with_two_runs, file_sha256(pipeline_file_b))
    before = tree_digest(store_with_two_runs / "tables" / own.run_id)
    resave_with_pipeline(store_with_two_runs, pipeline_file_b, tables=False)
    assert tree_digest(store_with_two_runs / "tables" / own.run_id) == before

def test_the_same_pipeline_replaces_its_folder_with_new_inodes(store_with_two_runs, pipeline_file_b):
    own = table_run_for_pipeline(store_with_two_runs, file_sha256(pipeline_file_b))
    inode = own.measurements_path().stat().st_ino
    resave_with_pipeline(store_with_two_runs, pipeline_file_b)
    again = table_run_for_pipeline(store_with_two_runs, file_sha256(pipeline_file_b))
    assert again.run_id == own.run_id and again.measurements_path().stat().st_ino != inode

def test_stale_is_recomputed_not_latched(store_with_crop_entry):
    # Review Focus 2: Stage 1 writes zeros, Stage 3 restores the same objmap
    resave_with_objmap(store_with_crop_entry, zeros=True)
    assert crop_entry(store_with_crop_entry)["stale"] is True
    resave_with_objmap(store_with_crop_entry, zeros=False)
    assert crop_entry(store_with_crop_entry)["stale"] is False

def test_independent_reader(tmp_path, measured_image):
    store = measured_image.save2zarr(tmp_path / "p.ome.zarr", measurement_table=tables_for(measured_image))
    run = latest_table_run(store)
    g = zarr.open_group(store, mode="r")
    assert g["tables"][run.run_id]["measurements"].attrs["type"] == "feature_table"
    assert pq.read_table(run.measurements_path()).num_rows == measured_image.num_objects
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - `_write_store_part` takes `pipeline_sha256` from
    `current_pipeline_sha256(self)`, and the run id from
    `choose_table_run_id` over the live store's block (when the store exists)
    merged with `upstream`.
  - It calls `carry_table_runs(final, part, exclude=run_id)` only when this
    write produces that run's tables (`measurement_table` or `embeddings` is
    given). A table-less write (`_cli_process_single.py:327`,
    `_cli_staged_workers.py:370,424`) passes `exclude=None` and carries every
    folder, its own included. This is the rule figures follow
    (`_image_io_handler.py:1470-1472`).
  - It computes `objmap_digest` from the in-memory objmap and calls
    `refresh_embedding_staleness`.
  - The root-level `metadata_table` block (`PhenotypicAttr.METADATA_TABLE`)
    moves into its run entry (spec D3). `_run_state.py`'s readers of it follow
    in Task 4.
  - This commit turns the README's P3 GUI rows red. That is expected.
- [ ] **Step 4: Run them, plus `tests/unit/sdk_/test_ngff_promote.py` and the
  existing `_measurement_tables` tests.** They pass.
- [ ] **Step 5: Run the ngio discovery check (D4).** This is manual, and adds no
  dependency:
  `uv run --with ngio python -c "import ngio; c = ngio.open_ome_zarr_container('<store>'); print(c.list_tables())"`.
  Record in the commit message whether the nested run's `measurements` table is
  listed. If it isn't, keep the attributes and record the limit in
  `docs/source/how_to/pages/zarr_storage.md`. Do not flatten the layout.
- [ ] **Step 6: Commit.**

### Task 3: The measure and recompile rewrite

**Files:**
- Modify: `src/phenotypic/sdk_/_measurement_tables.py:638-806`:
  - `_rewrite_store_tables` narrows `rmtree(part / "tables")` (:694) to
    `rmtree(part / "tables" / run_id)`, so that folder's files are rewritten as
    new inodes and every other folder is carried;
  - `replace_image_tables` takes `pipeline_sha256` and returns the `TableRun`
    it wrote. It picks the run with `choose_table_run_id` and writes through
    `write_run_entry`.
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:2016-2072`.
  `replace_image_store_measurements` passes the sha of the pipeline file
  measure mode runs, and returns the run's measurements path. The marker
  republish (`_republish_table_marker`, `_cli_recompile_tables.py:51`, called at
  `_cli_process_single.py:567-581`) declares that path.
- Create (helpers), in `tests/_tiny_run.py`: fixture `full_run_store`, the
  store of `measured_image` saved with A's tables; `other_tables(store)`, a
  `PreparedImageTables` with a different `Pipeline_Marker`
- Test: `tests/unit/sdk_/test_measure_rewrite_v2.py` (new)

**Interfaces:**
- Consumes: Task 1's `choose_table_run_id`, Task 2's `write_run_entry`.
- Produces: `replace_image_tables(store_path, tables, *, pipeline_sha256: str | None, figures=None, objmap_target=None, durable=None, commit_guard=None) -> TableRun`

- [ ] **Step 1: Write the failing tests**

```python
def test_measure_with_a_new_pipeline_adds_a_folder(full_run_store):
    # Review Focus 3
    first = latest_table_run(full_run_store)
    before = tree_digest(full_run_store / "tables" / first.run_id)
    run = replace_image_tables(full_run_store, other_tables(full_run_store), pipeline_sha256=SHA_B)
    assert run.order == 2 and len(list_table_runs(full_run_store)) == 2
    assert tree_digest(full_run_store / "tables" / first.run_id) == before

def test_measure_with_the_same_pipeline_twice_keeps_one_folder_and_its_rows(full_run_store):
    first = latest_table_run(full_run_store)
    rows = pq.read_table(first.measurements_path()).num_rows
    for _ in range(2):
        run = replace_image_tables(full_run_store, tables_for_store(full_run_store),
                                   pipeline_sha256=first.pipeline_sha256)
    assert run.run_id == first.run_id and len(list_table_runs(full_run_store)) == 1
    assert pq.read_table(run.measurements_path()).num_rows == rows

def test_the_measure_rewrite_leaves_stale_unchanged(store_with_crop_entry):
    resave_with_objmap(store_with_crop_entry, zeros=True)
    replace_image_tables(store_with_crop_entry, other_tables(store_with_crop_entry), pipeline_sha256=SHA_B)
    assert crop_entry(store_with_crop_entry)["stale"] is True     # D7: no objmap written
```

  `tables_for_store(store)` rebuilds the same tables from the stored image.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, plus the existing measure-mode tests
  (`grep -rl replace_image_tables tests/`).** They pass.
- [ ] **Step 5: Commit.**

### Task 4: Every reader through the resolver

**Files:** each of these changes a path join, a parents-count or a literal to a
resolver call.
- Finalizer and aggregation: `src/phenotypic/_cli/_cli_parquet_agg.py:171-240`,
  `_cli_finalize_fanout.py:500-523`, `_cli_finalize_run.py`,
  `_cli_chunk_writer.py:322-329` (`parents[2]`, `parents[4]`),
  `_embedded_measurement_tables.py`
- Per-image writers and resume: `_cli_process_single.py:74,1052`,
  `_cli_execution_strategies.py:68,183`, `_cli_staged_strategy.py:113-117`,
  `_cli_staged_resume.py:12,292-297,406`, `_cli_staged_slurm_worker.py:23`
- Recompile: `_cli_recompile_recovery.py:28,359,435,492`, `_cli_recompile_tables.py`,
  `_cli_recompile_worker.py:371,481-484,516,587` (including the literal
  `("tables", "measurements", "table.parquet")`),
  `_cli_recompile_slurm_scripts.py:532` (`parents[2]`), and
  `phenotypicCLI.py:4000-4010` (recompile overlay regeneration,
  `item.store / MEASUREMENT_TABLE_RELATIVE_PATH`)
- State: `_cli_migrate_state.py`, `src/phenotypic/sdk_/_run_state.py:665-678,740-760`
- Migrate's own readers and writers move in Task 5.
- `sdk_` readers the GUI calls, with signatures unchanged and
  `latest_measurement_run` by default: `read_embedded_measurement_descriptor`
  (:871), `embedded_measurement_columns` (:933), `read_embedded_measurement_column`
  (:961)
- `src/phenotypic/_cli/_cli_finalize_run.py`: `metrics.json`'s
  `pipeline_sha256` becomes the sha every aggregated run folder shares, or
  `null` when they differ (`build_run_metrics` takes `str | None`).
- Create (helpers), in `tests/_tiny_run.py`: `one_store(out) -> Path`, the
  single store under `out/results`
- Test: `tests/unit/cli/test_tables_v2_readers.py` (new), plus the existing
  importers

**Interfaces:**
- Consumes: Task 1.
- Produces: `certified_table_run(output_root: Path, marker_path: Path) -> TableRun | None`
  in `_cli_recompile_recovery.py`, built on `_marker_measurement_source`
  (:314). It is the run folder the image's marker certifies.
- Selection rule (D6a):
  - Finalize, measure mode and recompile aggregate `certified_table_run` per
    image. They never select by the run config's pipeline sha, which still
    names the full run's pipeline after `--mode measure`.
  - Generic `sdk_` readers read `latest_measurement_run(store)`.
  - The completion marker declares the run's parquet as the table artifact.

- [ ] **Step 1: Write the failing tests**

```python
def test_master_reads_the_folder_the_record_certifies(store_with_two_runs, tmp_path):
    marker = write_marker_certifying(tmp_path, list_table_runs(store_with_two_runs)[0])
    run = certified_table_run(tmp_path, marker)
    frame = project_embedded_measurement_table(run.measurements_path())
    assert set(frame["Pipeline_Marker"]) == {"A"}

def test_a_full_rerun_with_an_edited_pipeline_numbers_folders_1_and_2(tiny_inputs, tmp_path, pipeline_file_b):
    out = tmp_path / "out"
    run_cli(tiny_inputs, out)
    run_cli(tiny_inputs, out, pipeline=pipeline_file_b)
    assert [r.order for r in list_table_runs(one_store(out))] == [1, 2]
    master = pl.read_parquet(out / "deliverables" / "master_measurements.parquet")
    assert str(SHAPE.CIRCULARITY) in master.columns          # only pipeline B measures shape

def test_measure_twice_with_the_same_pipeline_keeps_one_folder_and_row_count(finished_tiny_run):
    rows = pl.read_parquet(finished_tiny_run / "deliverables" / "master_measurements.parquet").height
    run_cli(None, finished_tiny_run, mode="measure"); run_cli(None, finished_tiny_run, mode="measure")
    assert len(list_table_runs(one_store(finished_tiny_run))) == 1
    assert pl.read_parquet(finished_tiny_run / "deliverables" / "master_measurements.parquet").height == rows

def test_measure_with_another_pipeline_adds_a_folder_and_recompile_keeps_it(finished_tiny_run, pipeline_file_b):
    first = list_table_runs(one_store(finished_tiny_run))[0]
    before = tree_digest(one_store(finished_tiny_run) / "tables" / first.run_id)
    run_cli(None, finished_tiny_run, mode="measure", pipeline=pipeline_file_b)
    assert len(list_table_runs(one_store(finished_tiny_run))) == 2
    assert tree_digest(one_store(finished_tiny_run) / "tables" / first.run_id) == before
    run_cli(None, finished_tiny_run, mode="recompile")
    metrics = json.loads((finished_tiny_run / "deliverables" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["pipeline_sha256"] == file_sha256(pipeline_file_b)    # I17: B's folder, not A's

def test_marker_rejects_a_corrupted_run_table(finished_tiny_run):
    latest_measurement_run(one_store(finished_tiny_run)).measurements_path().write_bytes(b"corrupt")
    assert resolve_run_state(finished_tiny_run, depth="deep").completion != "complete"

def test_gui_facing_readers_default_to_the_latest_folder_with_measurements(store_with_two_runs):
    add_grid_only_run(store_with_two_runs)            # a folder with no `measurements`
    expected = list_table_runs(store_with_two_runs)[1]
    assert embedded_measurement_columns(store_with_two_runs) == tuple(pq.read_schema(expected.measurements_path()).names)
```

  `write_marker_certifying(output_root, run)` and `add_grid_only_run(store)`
  are module-local: the first writes a completion marker whose `measurements`
  artifact is `run.measurements_path()`, the second adds a run entry with no
  `measurements` block through `write_run_entry` in a rewrite. The shape
  header is spelled by the schema: `str(SHAPE.CIRCULARITY)` is the prefixed
  column name (`schema/_base/_measurement_info.py:635`).
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Substitute, module by module.** After each module, run
  `grep -nE 'MEASUREMENT_TABLE_RELATIVE_PATH|METADATA_TABLE_RELATIVE_PATH|parents\[|"tables"|"table\.parquet"' <module>`.
  Every hit outside the v1 migration and legacy-read code is a missed reader.
- [ ] **Step 4: Run the phase gate.**
  - Importers of `_measurement_tables`, `_table_runs`, `_cli_parquet_agg`,
    `_cli_recompile_*` and `_run_state` (`grep -rl` over `tests/`).
  - `tests/unit/gui` and `tests/gui` once. The red set equals the README's P0
    and P3 rows; any other red GUI test is a real failure.
  - Run this surface as a Slurm job if it exceeds a couple of minutes.
- [ ] **Step 5: Commit**, one commit per module group.

### Task 5: v1 stores — refuse forward, convert in migrate

**Files:**
- Create: `src/phenotypic/_cli/_cli_tables_v1.py`, the CLI-only v1 check.
- Modify: `src/phenotypic/phenotypicCLI.py`, the read-only half, beside the
  unconverted-tree refusal that follows the process overlap check (:2307).
  Every mode except `migrate` calls `first_v1_store` and raises
  `click.UsageError` naming the store and `--mode migrate`.
  `datasets_needing_migration` is **not** touched: it runs in
  `_run_state._advisories` (:1539) on every `resolve_run_state` call, GUI and
  `phenotypic-wait` polls included.
- Modify: `src/phenotypic/sdk_/_measurement_tables.py:807-868`.
  `replace_embedded_measurement_table` writes v2 directly, through
  `write_run_entry`, so migrate never leaves a tree in a mixed layout. Its
  callers (`_cli_migrate_image.py:298`, `_cli_migrate.py:1394`) keep their
  call shape.
- Modify: `src/phenotypic/_cli/_cli_migrate.py` and
  `_cli_migrate_provenance_worker.py`:
  - add a `tables_v2` step to the store array, ordered after every step that
    can still write a store's tables;
  - after the conversion, republish each converted image's record
    (`_republish_table_marker`, `_cli_recompile_tables.py:51`), then the run
    proof, as measure mode does;
  - replace the glob at :1057-1061
    (`*/zarr/*.ome.zarr/tables/measurements/table.parquet`, always empty on v2)
    with `latest_measurement_run` over the tree's stores, so the "aggregate
    rebuild produced no measurements" guard keeps firing.
- Test: `tests/unit/cli/test_migrate_tables_v2.py` (new); keep
  `tests/integration/cli/test_migrate_end_to_end.py` green.

**Interfaces:**
- Produces:
  - `first_v1_store(output_dir: Path) -> Path | None`: walks
    `results/*/zarr/*.ome.zarr` and returns the first store with
    `store_tables_schema == 1`, reading one root per store and stopping there.
  - `v1_table_pipeline_sha(block: Mapping) -> str | None` (spec D10): the most
    recent figure run's `pipeline_sha256` when the figures descriptor has one;
    otherwise the latest journal application of kind `full` or `legacy` with a
    pipeline digest; otherwise `None`. Measure mode never journals, so a v1
    table it rewrote is labelled correctly only through its figure run.
  - `convert_tables_v1_to_v2(store: Path, *, durable: bool | None) -> TableRun | None`.
    It moves `tables/measurements` and `tables/metadata` into
    `tables/1-<sha12>/` (or `1-unknown`, `pipeline_sha256: null`) through
    `_rewrite_store_tables`' hard-link copy and root-last promote. Pixel arrays
    are never rewritten. It returns `None` for a store already at v2.

- [ ] **Step 1: Write the failing tests**

```python
def test_forward_run_refuses_a_v1_store(v1_tree, tiny_inputs):
    # Review Focus 1
    before = tree_digest(v1_tree)
    result = run_cli(tiny_inputs, v1_tree)
    assert result.exit_code == 2 and "--mode migrate" in result.output
    assert tree_digest(v1_tree) == before                     # refused before anything is written

def test_migrate_converts_and_is_idempotent(v1_store):
    run = convert_tables_v1_to_v2(v1_store, durable=False)
    assert run.run_id.startswith("1-") and run.measurements_path().is_file()
    assert store_tables_schema(v1_store) == 2
    assert convert_tables_v1_to_v2(v1_store, durable=False) is None

def test_unknown_digest_is_never_fabricated(v1_store_legacy_null_pipeline):
    run = convert_tables_v1_to_v2(v1_store_legacy_null_pipeline, durable=False)
    assert run.run_id == "1-unknown" and run.pipeline_sha256 is None

def test_the_figure_run_sha_wins_over_the_journal():
    block = {"figures": {"runs": {"2026-11-01-" + SHA_B[:12]: {"pipeline_sha256": SHA_B}}},
             "provenance": journal_with_full_application(SHA_A)}
    assert v1_table_pipeline_sha(block) == SHA_B

def test_the_migrate_import_writes_v2(v1_store):
    convert_tables_v1_to_v2(v1_store, durable=False)
    replace_embedded_measurement_table(v1_store, prepared_from(v1_store))
    assert not (v1_store / MEASUREMENT_TABLE_RELATIVE_PATH).exists()
    assert len(list_table_runs(v1_store)) == 1

def test_a_completed_v1_tree_reads_complete_after_migrate(v1_tree):
    assert resolve_run_state(v1_tree, depth="deep").completion == "complete"
    assert run_cli(None, v1_tree, mode="migrate").exit_code == 0
    assert store_tables_schema(one_store(v1_tree)) == 2
    assert resolve_run_state(v1_tree, depth="deep").completion == "complete"
```

  `journal_with_full_application(sha)` and `prepared_from(store)` are
  module-local: the first builds a one-application provenance block, the
  second a `PreparedEmbeddedMeasurementTable` from the store's own table.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, plus the migrate integration test and
  `grep -rl "replace_embedded_measurement_table\|_cli_migrate" tests/`.** They
  pass.
- [ ] **Step 5: Commit.**

### Task 6: Upstream reference

**Files:**
- Modify: `src/phenotypic/sdk_/_table_runs.py`. Add the upstream functions.
- Modify: the per-image writers that know their input path:
  `_cli_process_single.py`, `_cli_staged_workers.py` (Stage 1/3), and
  `_cli_process_only.py`. Each passes `upstream=` to the store write.
- Create (helpers), in `tests/_tiny_run.py`:
  - fixture `input_store_v2`: `measured_image` saved under
    `<tmp>/in/ds/plate.ome.zarr`, so it holds `1-<A>`
  - `root_tables(store) -> dict`: the store's `attributes.phenotypic.tables`
  - `touch_root(store)`: rewrites the store's root `zarr.json` with one more
    byte of whitespace
- Test: `tests/unit/sdk_/test_upstream_reference.py` (new)

**Interfaces:**
- Produces:
  - `upstream_entries_for_input(input_path: Path, input_root: Path) -> dict[str, dict]`.
    Empty unless `input_path` is a store at v2. Entries are
    `{run_id: {"source": <path relative to input_root>, "root_sha256": <sha of
    the input's root zarr.json>, "pipeline_sha256": ...}}`, merged with the
    input's own `upstream`.
  - `follow_upstream(store: Path, run_id: str, *, input_root: Path) -> TableRun`.
    It raises `UpstreamChangedError(run_id, expected, actual)` on a root-digest
    mismatch, and `FileNotFoundError` when the input is unreachable.

- [ ] **Step 1: Write the failing tests**

```python
def test_result_store_records_input_runs_and_numbers_after_them(input_store_v2, tmp_path, pipeline_file_b):
    out = tmp_path / "out"
    run_cli(input_store_v2.parent.parent, out, pipeline=pipeline_file_b)
    tables = root_tables(one_store(out))
    assert set(tables["upstream"]) == {latest_table_run(input_store_v2).run_id}       # "1-<A>"
    assert tables["upstream"][next(iter(tables["upstream"]))]["root_sha256"] == file_sha256(input_store_v2 / "zarr.json")
    assert [r.run_id for r in list_table_runs(one_store(out))] == [f"2-{file_sha256(pipeline_file_b)[:12]}"]

def test_follow_upstream_detects_a_reprocessed_input(input_store_v2, tmp_path, pipeline_file_b):
    out = tmp_path / "out"
    run_cli(input_store_v2.parent.parent, out, pipeline=pipeline_file_b)
    touch_root(input_store_v2)                     # simulate an AutoConvertRaw-GC reprocess
    with pytest.raises(UpstreamChangedError):
        follow_upstream(one_store(out), next(iter(root_tables(one_store(out))["upstream"])),
                        input_root=input_store_v2.parent.parent)
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** Nothing is copied; only the reference is recorded.
- [ ] **Step 4: Run them; they pass.**
- [ ] **Step 5: Commit.** Then run the P3 phase gate once: Task 4 Step 4's
  surface, plus `tests/integration/cli`. The red GUI set equals the README's P0
  and P3 rows.
