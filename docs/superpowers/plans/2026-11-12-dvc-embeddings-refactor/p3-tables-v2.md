# P3 · Per-pipeline tables (`tables/` v2)

**Spec:** Part D (D1–D10)

---

**Goal:** Each store holds one folder per measuring pipeline,
`tables/<seq>-<sha12>/`, and every writer carries the other folders across.

**Approach:** All 27 modules that touch today's flat layout reach a run folder
through one resolver. That makes the change a mechanical substitution at each
call site, not 27 separate path joins. v1 stores are converted by
`--mode migrate`, and forward modes refuse them.

### Task 1: Run identity and the resolver

**Files:**
- Create: `src/phenotypic/sdk_/_table_runs.py`
- Modify: `src/phenotypic/sdk_/ngff_.py:78-100`. Add `TABLES_SCHEMA_VERSION = 2`
  and `EMBEDDINGS_TABLE_GROUP = "embeddings"`. Keep `MEASUREMENT_TABLE_RELATIVE_PATH`
  as the **v1** legacy path; the GUI imports it (#283).
- Modify: `src/phenotypic/_core/_provenance.py`. Add
  `current_application_identity(image) -> tuple[int, str | None]`, the last
  application's `sequence` and its `pipeline["sha256"]`.
- Modify: `src/phenotypic/sdk_/__init__.py`. Export the resolver names; add them
  to `_LAZY_ATTRS` if they import pandas.
- Test: `tests/unit/sdk_/test_table_runs.py` (new)

**Interfaces:**
- Produces:
  - `TABLE_RUN_ID_RE = re.compile(r"^(?P<seq>[1-9][0-9]*)-(?P<sha>[0-9a-f]{12}|unknown)$")`
  - `table_run_id(sequence: int, pipeline_sha256: str | None) -> str`
    (`None` → `"<seq>-unknown"`)
  - `@dataclass(frozen=True) class TableRun`:
    - fields: `store: Path`, `run_id: str`, `sequence: int`,
      `pipeline_sha256: str | None`, `descriptor: Mapping[str, object]`
    - `measurements_path() -> Path` and `metadata_path() -> Path`
    - `embedding_path(model: str, region: Literal["crops", "grid"]) -> Path`
  - `store_tables_schema(store: Path) -> Literal[0, 1, 2]`: 0 means no tables
  - `list_table_runs(store: Path) -> tuple[TableRun, ...]`, ascending
    `sequence`; `()` for schema 0 or 1
  - `latest_table_run(store) -> TableRun | None`
  - `table_run_for_pipeline(store, pipeline_sha256: str) -> TableRun | None`
  - `choose_table_run_id(root_block: Mapping, *, pipeline_sha256: str | None, journal_sequence: int | None) -> str`:
    1. an existing run with this sha → its id (D6);
    2. else, with `journal_sequence` given (forward runs) →
       `table_run_id(journal_sequence, sha)`;
    3. else (measure mode, which appends no application) →
       `max(last journal sequence, max existing run sequence) + 1`.
  - `store_of_table_path(path: Path) -> Path`: the nearest ancestor ending in
    `.ome.zarr`. It replaces the `parents[len(MEASUREMENT_TABLE_RELATIVE_PATH.parts) - 1]`
    arithmetic in `_cli_parquet_agg.py:240` and `_cli_finalize_fanout.py:523`.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_id_format():
    assert table_run_id(2, "a71e09c2b4f5" + "0" * 52) == "2-a71e09c2b4f5"
    assert table_run_id(3, None) == "3-unknown"
    assert TABLE_RUN_ID_RE.match("0-abc") is None

def test_choose_reuses_same_pipeline_folder():
    block = v2_block(runs={"2-a71e09c2b4f5": {"sequence": 2, "pipeline_sha256": SHA_A}})
    assert choose_table_run_id(block, pipeline_sha256=SHA_A, journal_sequence=5) == "2-a71e09c2b4f5"

def test_measure_mode_gets_next_sequence():
    block = v2_block(runs={"2-a71e09c2b4f5": {"sequence": 2, "pipeline_sha256": SHA_A}}, journal_len=2)
    assert choose_table_run_id(block, pipeline_sha256=SHA_B, journal_sequence=None) == f"3-{SHA_B[:12]}"

def test_store_of_table_path_handles_both_depths(tmp_path):
    s = tmp_path / "x.ome.zarr"
    assert store_of_table_path(s / "tables/measurements/table.parquet") == s
    assert store_of_table_path(s / "tables/2-a71e09c2b4f5/measurements/table.parquet") == s
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** This is pure parsing of the root `zarr.json`
  block. Read it with the existing reader in `_measurement_tables.py:871`.
- [ ] **Step 4: Run again; it passes.**
- [ ] **Step 5: Commit.**

### Task 2: The writer, the carry step and stale binding

**Files:**
- Modify: `src/phenotypic/sdk_/_measurement_tables.py`:
  - :197-260, `write_embedded_measurement_table` and `write_metadata_table`
    write under `tables/<run>/`
  - :347-432, `build_image_tables_attributes`, `apply_image_tables_attributes`
    and `write_image_tables` take `run_id` and produce the v2 block
  - :433-600, the contract validators read v2
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`, the
  `_write_store_part` call to `write_image_tables` (:1453) and the carry beside
  `carry_figure_runs` (:1468)
- Test: `tests/unit/sdk_/test_tables_v2_writer.py` (new)

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `write_image_tables(store_part: Path, tables: PreparedImageTables, *, run_id: str, sequence: int, pipeline_sha256: str | None, objmap_target: str) -> dict`.
    It returns the run's descriptor entry.
  - `carry_table_runs(final: Path, part: Path, *, exclude: str | None) -> dict`.
    It hard-links every other run folder into the part and returns their
    descriptor entries; it also carries `upstream`.
  - `refresh_embedding_staleness(runs: dict, *, objmap_sha256: str | None, array_sha256: Mapping[str, str]) -> None`.
    It sets each embedding entry's `stale` from its `binds` (D7).
  - `NGFF_FEATURE_TABLE_ATTRS(objmap_rel_path: str) -> dict`, the ngio attributes
    on each `measurements/` group (D4):
    `{"type": "feature_table", "table_version": "1", "region": {"path": ...},
    "backend": "parquet", "index_key": "Object_Label", "index_type": "int",
    "instance_key": "Object_Label"}`

- [ ] **Step 1: Write the failing tests**

```python
def test_save_writes_run_folder_and_v2_descriptor(tmp_path, measured_image):
    store = measured_image.save2zarr(tmp_path / "p.ome.zarr", measurement_table=tables_for(measured_image))
    run = latest_table_run(store)
    assert run.run_id == f"1-{SHA[:12]}" and run.measurements_path().is_file()
    root = json.loads((store / "zarr.json").read_text())["attributes"]["phenotypic"]["tables"]
    assert root["schema_version"] == 2 and list(root["runs"]) == [run.run_id]

def test_resave_carries_other_runs_byte_identical(tmp_path, store_with_two_runs):
    before = tree_digest(store_with_two_runs / "tables" / "1-aaaaaaaaaaaa")
    resave_with_pipeline(store_with_two_runs, SHA_B)
    assert tree_digest(store_with_two_runs / "tables" / "1-aaaaaaaaaaaa") == before

def test_stale_is_recomputed_not_latched(tmp_path, store_with_crop_entry):
    # Review Focus 2: Stage 1 writes zeros, Stage 3 restores the same objmap
    resave_with_objmap(store_with_crop_entry, zeros=True)
    assert crop_entry(store_with_crop_entry)["stale"] is True
    resave_with_objmap(store_with_crop_entry, zeros=False)
    assert crop_entry(store_with_crop_entry)["stale"] is False

def test_independent_reader(tmp_path, measured_image):
    store = measured_image.save2zarr(tmp_path / "p.ome.zarr", measurement_table=tables_for(measured_image))
    g = zarr.open_group(store, mode="r")
    assert g["tables"][latest_table_run(store).run_id]["measurements"].attrs["type"] == "feature_table"
    assert pq.read_table(latest_table_run(store).measurements_path()).num_rows == measured_image.num_objects
```

  `crop_entry` writes and reads a hand-built embedding entry dict. P4 adds real
  ones.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - `_write_store_part` takes `run_id`, `sequence` and `pipeline_sha256` from
    `current_application_identity(self)` and `choose_table_run_id`.
  - It calls `carry_table_runs(final, part, exclude=run_id)` before the root.
  - It computes `objmap_sha256` from the in-memory objmap and calls
    `refresh_embedding_staleness`.
  The root-level `metadata_table` block (`PhenotypicAttr.METADATA_TABLE`) moves
  into its run entry (spec D3). `_run_state.py`'s readers of it follow in Task 4.
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
- Modify: `src/phenotypic/sdk_/_measurement_tables.py:638-806`.
  `_rewrite_store_tables` narrows `rmtree(part / "tables")` (:694) to
  `rmtree(part / "tables" / run_id)`; `replace_image_tables` takes
  `pipeline_sha256`.
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:2016-2072`.
  `replace_image_store_measurements` passes the sha of the pipeline file measure
  mode runs, and returns the run's table path.
- Test: `tests/unit/cli/test_measure_mode_tables_v2.py` (new)

**Interfaces:**
- Consumes: Task 1's `choose_table_run_id` with `journal_sequence=None`.
- Produces: `replace_image_tables(store_path, tables, *, pipeline_sha256: str, figures=None, durable=None, commit_guard=None) -> TableRun`

- [ ] **Step 1: Write the failing tests**

```python
def test_measure_with_new_pipeline_adds_a_folder(full_run_store):
    # Review Focus 3
    first = tree_digest(full_run_store / "tables" / latest_table_run(full_run_store).run_id)
    run = replace_image_tables(full_run_store, other_tables, pipeline_sha256=SHA_B)
    assert run.sequence == 2 and len(list_table_runs(full_run_store)) == 2
    assert tree_digest(full_run_store / "tables" / list_table_runs(full_run_store)[0].run_id) == first

def test_measure_with_same_pipeline_replaces_its_folder(full_run_store):
    run_id = latest_table_run(full_run_store).run_id
    run = replace_image_tables(full_run_store, other_tables, pipeline_sha256=FULL_SHA)
    assert run.run_id == run_id and len(list_table_runs(full_run_store)) == 1
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** Rewrite only the run folder, as new inodes, which is
  the figures rule.
- [ ] **Step 4: Run them, plus the existing measure-mode tests
  (`grep -rl replace_image_tables tests/`).** They pass.
- [ ] **Step 5: Commit.**

### Task 4: Every reader through the resolver

**Files:** each of these changes a path join or a parents-count to a resolver
call.
- Finalizer and aggregation: `src/phenotypic/_cli/_cli_parquet_agg.py:171-240`,
  `_cli_finalize_fanout.py:500-523`, `_cli_finalize_run.py`,
  `_cli_chunk_writer.py`, `_embedded_measurement_tables.py`
- Per-image writers and resume: `_cli_process_single.py:74,1052`,
  `_cli_execution_strategies.py:68,183`, `_cli_staged_strategy.py:113-117`,
  `_cli_staged_resume.py:12,292-297,406`, `_cli_staged_slurm_worker.py:23`
- Recompile: `_cli_recompile_recovery.py:28,359,435,492`, `_cli_recompile_tables.py`,
  `_cli_recompile_worker.py:371,516,587`, `_cli_recompile_slurm_scripts.py`
- Migrate and state: `_cli_migrate.py`, `_cli_migrate_image.py`,
  `_cli_migrate_state.py`, `src/phenotypic/sdk_/_run_state.py:673-678,745-757`
- `sdk_` readers the GUI calls, with signatures unchanged and the **latest run**
  by default (#283): `read_embedded_measurement_descriptor` (:871),
  `embedded_measurement_columns` (:933), `read_embedded_measurement_column`
  (:961)
- Test: `tests/unit/cli/test_tables_v2_readers.py` (new), plus the existing
  importers

**Interfaces:**
- Consumes: Task 1.
- Selection rule:
  - The forward finalizer, measure mode and recompile read
    `table_run_for_pipeline(store, <the run's pipeline sha>)`.
  - Generic `sdk_` readers read `latest_table_run(store)`.
  - The completion marker declares the run's parquet as the table artifact.

- [ ] **Step 1: Write the failing tests**

```python
def test_master_reads_the_run_for_its_own_pipeline(store_with_two_runs, tmp_path):
    frame = project_embedded_measurement_table(
        table_run_for_pipeline(store_with_two_runs, SHA_A).measurements_path())
    assert set(frame["Pipeline_Marker"]) == {"A"}    # fixture writes a marker column per run

def test_marker_rejects_a_corrupted_run_table(finished_tiny_run):
    run = latest_table_run(one_store(finished_tiny_run))
    run.measurements_path().write_bytes(b"corrupt")
    assert resolve_run_state(finished_tiny_run, depth="deep").completion != "complete"

def test_gui_facing_readers_keep_signatures(store_with_two_runs):
    assert embedded_measurement_columns(store_with_two_runs) == latest_columns(store_with_two_runs)
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Substitute, module by module.** After each module, run
  `grep -n "MEASUREMENT_TABLE_RELATIVE_PATH\|METADATA_TABLE_RELATIVE_PATH" <module>`.
  It must return nothing outside the v1 migration and legacy-read code.
- [ ] **Step 4: Run the phase gate.**
  - Importers of `_measurement_tables`, `_table_runs`, `_cli_parquet_agg`,
    `_cli_recompile_*` and `_run_state` (`grep -rl` over `tests/`).
  - `tests/unit/gui` and `tests/gui`, unmodified, must stay green except the
    one #283 item; record that failure if it appears.
  - Run this surface as a Slurm job if it exceeds a couple of minutes.
- [ ] **Step 5: Commit**, one commit per module group.

### Task 5: v1 stores — refuse forward, convert in migrate

**Files:**
- Modify: the writing-mode refusal (`datasets_needing_migration` and its
  callers). A store with `store_tables_schema == 1` needs migration.
- Modify: `src/phenotypic/_cli/_cli_migrate.py` and
  `_cli_migrate_provenance_worker.py`. Add a `tables_v2` conversion step to the
  store array (D10).
- Test: `tests/unit/cli/test_migrate_tables_v2.py` (new); keep
  `tests/integration/cli/test_migrate_end_to_end.py` green.

**Interfaces:**
- Produces: `convert_tables_v1_to_v2(store: Path, *, durable: bool | None) -> TableRun | None`.
  It moves `tables/measurements` and `tables/metadata` into `tables/<seq>-<sha12>/`,
  root-last. `<seq>` and the sha come from the journal's latest application of
  kind `full` or `legacy`; with no digest it uses `<seq>-unknown` and
  `pipeline_sha256: null`. It returns `None` for a store already at v2 (a no-op).

- [ ] **Step 1: Write the failing tests**

```python
def test_forward_run_refuses_a_v1_store(v1_tree):
    # Review Focus 1
    result = run_cli_full(v1_tree)
    assert result.exit_code != 0 and "--mode migrate" in result.output

def test_migrate_converts_and_is_idempotent(v1_store):
    run = convert_tables_v1_to_v2(v1_store, durable=False)
    assert run.measurements_path().is_file() and store_tables_schema(v1_store) == 2
    assert convert_tables_v1_to_v2(v1_store, durable=False) is None

def test_unknown_digest_is_never_fabricated(v1_store_legacy_null_pipeline):
    run = convert_tables_v1_to_v2(v1_store_legacy_null_pipeline, durable=False)
    assert run.run_id.endswith("-unknown") and run.pipeline_sha256 is None
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** Use `_rewrite_store_tables`' hard-link copy and
  root-last promote. Pixel arrays are never rewritten.
- [ ] **Step 4: Run them, plus the migrate integration test.** They pass.
- [ ] **Step 5: Commit.**

### Task 6: Upstream reference

**Files:**
- Modify: `src/phenotypic/sdk_/_table_runs.py`. Add the upstream functions.
- Modify: the per-image writers that know their input path:
  `_cli_process_single.py`, `_cli_staged_workers.py` (Stage 1/3), and
  `_cli_process_only.py`. Each passes `upstream=` to the store write.
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
def test_result_store_records_input_runs(process_store_v2, tmp_path):
    out = run_full_over(process_store_v2, tmp_path)
    up = root_tables(one_store(out))["upstream"]
    assert set(up) == {latest_table_run(process_store_v2).run_id}
    assert up[next(iter(up))]["root_sha256"] == sha256_file(process_store_v2 / "zarr.json")

def test_follow_upstream_detects_a_reprocessed_input(process_store_v2, tmp_path):
    out = run_full_over(process_store_v2, tmp_path)
    touch_root(process_store_v2)                     # simulate an AutoConvertRaw-GC reprocess
    with pytest.raises(UpstreamChangedError):
        follow_upstream(one_store(out), next(iter(root_tables(one_store(out))["upstream"])),
                        input_root=process_store_v2.parent)
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** Nothing is copied; only the reference is recorded.
- [ ] **Step 4: Run them; they pass.**
- [ ] **Step 5: Commit.** Then run the P3 phase gate once (Task 4 Step 4's
  surface, plus `tests/integration/cli`).
