# PA2 · The results viewer reads per-pipeline tables

**Spec:** §3.3 (reds #8–13), PA2

---

**Goal:** The Colony picker and the measurement routes read a v2 store's latest measuring run
and still read a v1 store; reds #8–13 are green.

**Depends on:** nothing. Runs as **Lane V** beside PA1, in its own worktree.

**Helpers stay local (R2-M1).** `add_second_measuring_run`, `add_grid_only_run`,
`rewrite_latest_table_with_zero_rows`, `run_from_fixture` and `v1_pairs` are written in the two
test modules that use them (`test_measurement_source.py`, `test_zarr_routes.py`), never in
`tests/_output_layout.py`, which PA1 T1 edits on the main line. A module-local
`add_grid_only_run` already exists in `tests/unit/sdk_/test_tables_v2_state_readers.py:29`; copy
its approach, do not import across test modules.

### Task 1: `_store_and_identity` through `declared_measurement_table_path` (C5)

**Files:**
- Modify: `src/phenotypic/_gui/results_viewer/_measurement_source.py:95-111` (and docstrings
  `:8-10, 60`).
- Modify (docstrings only): `results_viewer/_zarr_routes.py:31`, `_gui/_shared/tiles.py:986, 1001`.
- Modify (tests): `tests/unit/gui/results_viewer/test_measurement_source.py:247-249` (red #13),
  `tests/unit/gui/results_viewer/test_measurement_routes.py:205` (red #8),
  `tests/unit/gui/results_viewer/test_measurement_join_migration_run.py:105, 209`.
- Test: the two files above, plus `tests/unit/gui/results_viewer/test_zarr_routes.py`.

**Interfaces:**
- Consumes: `phenotypic.sdk_.declared_measurement_table_path(store: Path) -> Path | None`
  (`sdk_/_table_runs.py:613-635`, exported `sdk_/__init__.py:508, 630`).
- Produces: `_store_and_identity(output_root, dataset, stem) -> tuple[Path, tuple[int, int, int, int]] | None`,
  same signature; it stats `declared_measurement_table_path(store)` and returns `None` when
  that is `None` or missing. No new sdk_ export.

- [ ] **Step 1: Edit reds #8 and #13 to locate the payload with
  `declared_measurement_table_path(store)`**, keeping each test's property (spec §3.3 [call]):
  #8 still unlinks the payload before the request; #13 still rewrites the payload in place and
  asserts the second read sees `99.0`.
- [ ] **Step 2: Add the new tests.**

```python
def test_the_picker_reads_the_highest_order_measuring_run(run):
    add_second_measuring_run(run, MEASURED, extra_column="Size_Perimeter")   # resave_with_pipeline
    assert "Size_Perimeter" in displayable_measurement_columns(_output_root(run), [(DATASET, MEASURED)])

def test_a_newer_grid_only_run_is_skipped(run):
    add_grid_only_run(run, MEASURED)                 # a run folder with no measurements
    assert "Size_Area" in displayable_measurement_columns(_output_root(run), [(DATASET, MEASURED)])

def test_a_v1_store_still_reads(tmp_path):
    root = run_from_fixture("tests/fixtures/tables_v1/v1_run")   # copied, never edited
    assert displayable_measurement_columns(_output_root(root), v1_pairs(root))

def test_an_empty_latest_table_offers_nothing_and_does_not_raise(run):
    rewrite_latest_table_with_zero_rows(run, MEASURED)
    assert measurement_values_for(_output_root(run), [(DATASET, MEASURED)], "Size_Area") == {}
```
And in `test_zarr_routes.py`:
```python
def test_a_v2_table_path_is_never_served(route):
    assert route.get_raw("tables/1-unknown/measurements/table.parquet").status_code in (400, 404)
```
- [ ] **Step 3: Run `$PT tests/unit/gui/results_viewer/test_measurement_source.py
  tests/unit/gui/results_viewer/test_measurement_routes.py tests/unit/gui/results_viewer/test_zarr_routes.py`;
  reds #8–13 and the new picker tests fail.**
- [ ] **Step 4: Implement** the one-call change in `_store_and_identity` and the docstrings.
- [ ] **Step 5: Run the three files; all pass, reds #8–13 included.**
  `test_measurement_join_migration_run.py` may skip (live run absent, spec R5); its edit is
  reviewed by reading.
- [ ] **Step 6: Commit.**

### PA2 gate

`tests/unit/gui/results_viewer/` and `tests/gui/results_viewer/` (light). The orchestrator
merges Lane V into the main line after PA1's gate.
