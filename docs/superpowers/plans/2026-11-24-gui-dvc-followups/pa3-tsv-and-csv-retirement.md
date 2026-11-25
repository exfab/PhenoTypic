# PA3 · TSV everywhere a writer exists; `measurements.csv` retired

**Spec:** PA5, PA8 (D1, D18), §3.3 red #7; review I7, Minors 8, 10–13, 17, 19; re-review N6

---

**Goal:** Finalize and curation write `measurements.{tsv,parquet}` only and retire an older
run's generated CSV after the TSV lands; readers prefer `.tsv` and fall back to `.csv`.
`error_analysis.csv` stays CSV (D18).

**Depends on:** PA1, on the main line (plan review I8). T2 flips the default of the helper PA1
T1 extends, and it edits `tests/unit/gui/test_snapshot_status.py` (PA1 T6's test file), so the
two cannot run as parallel lanes without conflicting edits.

### Task 1: the CLI stops writing `measurements.csv` (C6)

**Files:**
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:968-1003` (`_seed_measurements`).
- Modify: `src/phenotypic/_cli/_cli_completion.py:1183-1210` (`required_paths`, comment).
- Modify: `src/phenotypic/sdk_/_delimited.py:146-162` (`_KEPT_CSV_STEMS = frozenset({"error_analysis", "metadata"})`: only `measurements` leaves it, D18; docstring).
- Modify: `src/phenotypic/sdk_/_io_constants.py:403-430, 1698-1730, 3371-3393` (docstrings only:
  CSV names are "read-only legacy; no writer"; `measurements_table_path` is wired).
- Modify: `_cli/_cli_interactive.py:257, 274-279`; `_cli/_cli_readme_generator.py:116, 147-150`;
  `_cli/_dashboard/_generator.py:826, 1017-1019` (`cmd-csv` → `cmd-tsv`, JS and CSS);
  `_cli/_cli_preflight.py:182-185`; `analysis/qc/_expected_vs_detected.py:298, 410` (docstrings).
- Modify (test, red #7): `tests/unit/gui/test_curation_does_not_unfence_the_run.py:135-139`.
- Test: `tests/unit/cli/test_deliverables_tsv.py`, `tests/unit/cli/test_tsv_cutover_old_tree.py`,
  `tests/unit/sdk_/test_delimited.py`, `tests/unit/cli/test_cli_output_manager.py`,
  `tests/unit/cli/test_readme_output_structure.py`, `tests/unit/cli/test_cli_progress_dashboard.py`.

**Interfaces:**
- Consumes: `write_tsv`, `remove_stale_csv_sibling` (`sdk_/_delimited.py:102, 151`);
  `_guarded_terminal_best_effort` (`_cli_output_manager.py:99-115`).
- Produces: `_seed_measurements(output_dir, master_df, *, commit_guard=None) -> None` writes
  `.tsv` then `.parquet`; after a **successful** TSV write, `remove_stale_csv_sibling(measurements_tsv_path(output_dir))`
  runs **through `commit_guard`** (the same guard wrapper the writes use). To know the TSV
  succeeded, the TSV call returns a sentinel through `_guarded_terminal_best_effort(default=False)`
  and a `True` on success. On the finalize path that passes `commit_guard=None`
  (`_guarded_terminal_call`, `:90-97`, allows it), the unlink runs unguarded, after the TSV,
  exactly as the analysis sweeps at `:838, 1502` do; the test pins both paths (Minor 7). The
  aggregate proof's `required_outputs` is
  `{"master_parquet", "measurements_tsv", "measurements_parquet"}` plus `metrics_json`/`schema_json`
  when written.

- [ ] **Step 1: Write the failing tests.**

```python
def test_a_finalize_writes_no_measurements_csv(tiny_finalized_run):
    assert not measurements_csv_path(tiny_finalized_run).exists()
    assert measurements_tsv_path(tiny_finalized_run).is_file()

def test_a_finalize_retires_an_old_csv_after_the_tsv(tmp_path):
    out = tree_with_generated_csv(tmp_path)          # pre-PA8 layout, real writers
    order = record_fs_order(out)                      # wraps write_tsv and Path.unlink
    finalize(out)
    assert order.index(("write", "measurements.tsv")) < order.index(("unlink", "measurements.csv"))

def test_the_retirement_runs_inside_the_commit_guard(tmp_path): ...   # a real guard: enter seen before unlink
def test_with_no_guard_the_retirement_still_follows_the_tsv(tmp_path): ...   # commit_guard=None path (Minor 7)
def test_a_failed_tsv_write_keeps_the_old_csv(tmp_path, monkeypatch):
    fail(monkeypatch, "phenotypic._cli._cli_output_manager.write_tsv")
    out = tree_with_generated_csv(tmp_path); finalize(out)
    assert measurements_csv_path(out).is_file()

def test_the_sweep_now_takes_measurements(tmp_path):
    assert remove_stale_csv_sibling(write_pair(tmp_path / "deliverables", "measurements")) is True

@pytest.mark.parametrize("stem", ["error_analysis", "metadata"])           # D18
def test_the_sweep_still_refuses_error_analysis_and_metadata(tmp_path, stem):
    with pytest.raises(ValueError, match="never swept"):
        remove_stale_csv_sibling(write_pair(tmp_path / "deliverables", stem))

def test_a_pre_pa8_proof_still_validates_after_the_csv_goes(tmp_path):    # I7
    out = build_complete_run(tmp_path)
    inject_descriptor(out, "measurements_csv", measurements_csv_path(out))   # the real bytes' size and sha,
    measurements_csv_path(out).unlink()                                       # as a pre-PA8 proof records them
    assert aggregate_proof_is_current(out)

def test_the_dashboard_download_names_the_tsv(...): assert "deliverables/measurements.tsv" in html
def test_the_readme_lists_no_measurements_csv(...): ...
```
Red #7's set becomes `{"master_parquet", "measurements_tsv", "measurements_parquet"}` (the
gate log shows those plus `measurements_csv` today; no `metrics_json`/`schema_json`).
- [ ] **Step 2: Run the six test files and red #7; the new tests fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Independent reader.** In the finalize test, read `measurements.tsv` with
  `pandas.read_csv(sep="\t", dtype=str, keep_default_na=False)` and the Parquet with
  `pyarrow.parquet.read_table`; assert the same columns and row count.
- [ ] **Step 5: Run the six files and red #7; all pass.**
- [ ] **Step 6: Mutation check.** (a) Drop `measurements_csv` from
  `_GUI_WRITTEN_PROOF_DESCRIPTORS`: `test_a_pre_pa8_proof_still_validates_after_the_csv_goes`
  must fail (I7). (b) Retire the CSV before, or regardless of, the TSV write:
  `test_a_failed_tsv_write_keeps_the_old_csv` and the ordering test must fail (I9). Restore from
  bytes.
- [ ] **Step 7: Commit.**

### Task 2: the test sweep over the CSV mirror (C7)

**Files:**
- Modify: `tests/_output_layout.py:82-105`: `write_measurements_mirror(..., csv: bool = False, ...)`.
- Modify: the **56** distinct files that are the union of the 30 files naming the CSV mirror
  (`grep -rln 'measurements\.csv\|measurements_csv\|MEASUREMENTS_CSV\|mirror_csv' tests`) and
  the 28 callers of `write_measurements_mirror` (Minor 8). Never `tests/fixtures/tables_v1/`.

**Interfaces:**
- Consumes: Task 1's behaviour.
- Produces: no API. Per assertion, one of three edits:
  1. "the CSV exists after a finalize" → "it does not, and the TSV does";
  2. an old-tree test passes `csv=True` and checks the fallback or the retirement;
  3. a test whose subject is unrelated to the mirror drops the CSV expectation.

- [ ] **Step 1: List every hit with its file:line and the edit kind (1/2/3)** in the cluster
  report before editing.
- [ ] **Step 2: Flip the helper default and apply the edits.**
- [ ] **Step 3: Run every importer of `tests/_output_layout.py`**
  (`grep -rl _output_layout tests/`, 62 files) as one Slurm job in a detached worktree. Each
  red is run alone before it is attributed. The only expected reds are the inherited 13 not
  yet fixed.
- [ ] **Step 4: Opus verify pass** reads the diff against the Step 1 table: no assertion
  weakened beyond its edit kind.
- [ ] **Step 5: Commit.**

### Task 3: curation writes the TSV (C8)

**Files:**
- Modify: `src/phenotypic/_gui/results_viewer/_curation_labels.py`: `_write_curated_mirror`
  (`:839-850`), `_source_fingerprint` (`:761-771`); delete `_atomic_write_csv` (`:961-966`).
- Modify: `src/phenotypic/_gui/results_viewer/_output_root.py:1119-1133`
  (`_consumed_state_snapshot_paths`: both `mirror_tsv` and `mirror_csv`).
- Modify: `src/phenotypic/_gui/results_viewer/_filtered_state.py:44, 255, 575-600` (one-writer swap;
  `:255` reads `measurements_csv_path`); `_curation_labels.py:284` (`CurationLabels.measurements_csv`
  becomes `measurements_tsv`, its callers follow) (Minor 9).
- Modify: `src/phenotypic/_gui/_schema_cache.py:27, 46` (TSV-first, `read_delimited`).
- Modify: `src/phenotypic/_gui/_config.py:68, 136, 462-466` (re-export `MEASUREMENTS_TSV`).
- Modify: `tests/unit/sdk_/test_gui_sdk_imports.py:89-92` (assert `{"MEASUREMENTS_TSV"} <= names`, N6);
  regenerate `tests/unit/sdk_/gui_sdk_signatures.json` by the procedure in that test's
  docstring (`:4, 15`).
- Test: `tests/unit/gui/results_viewer/test_curation_labels.py`, `tests/unit/gui/test_schema_cache.py`,
  `tests/gui/results_viewer/test_filtered_state.py`, `tests/unit/sdk_/test_gui_sdk_imports.py`.

**Interfaces:**
- Consumes: `BundleLayout.mirror_tsv`, `.mirror_csv`, `.measurements_table_path()`
  (`sdk_/_io_constants.py:3371-3393`); `write_tsv`, `read_delimited`, `remove_stale_csv_sibling`.
- Produces: `_write_curated_mirror() -> None`: Parquet, then TSV; on TSV success retire the
  CSV; on TSV failure unlink the TSV and the CSV and log (I7). `_schema_cache` resolves the
  text mirror by `layout.measurements_table_path()`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_curation_writes_a_tsv_that_reads_back_as_the_parquet(curated_store):   # Minor 17
    curate_one(curated_store)
    tsv = read_delimited(curated_store.layout.mirror_tsv)
    pq_ = pyarrow.parquet.read_table(curated_store.layout.mirror_parquet).to_pandas()
    assert_frame_equal(normalise(tsv), normalise(pq_))

def test_curation_retires_a_pre_pa8_csv(curated_store_with_csv):
    curate_one(curated_store_with_csv); assert not curated_store_with_csv.layout.mirror_csv.exists()

def test_a_failed_tsv_write_leaves_no_text_mirror(curated_store_with_csv, monkeypatch):  # I7
    fail(monkeypatch, "phenotypic._gui.results_viewer._curation_labels.write_tsv")
    curate_one(curated_store_with_csv)
    assert not curated_store_with_csv.layout.mirror_tsv.exists()
    assert not curated_store_with_csv.layout.mirror_csv.exists()
    assert curated_store_with_csv.layout.mirror_parquet.is_file()

def test_an_old_gui_curated_tree_is_repaired_by_the_next_curation(tmp_path):   # Review Focus 2
    out = old_gui_curated_tree(tmp_path)      # curated .csv/.parquet, uncurated .tsv
    assert text_mirror_keys(out) == parquet_keys(out)   # rows and keys, not columns (Minor 9):
                                                         # every reader of the text mirror, TSV-first
    curate_one_at(out)
    assert read_delimited(mirror_tsv(out)).shape[0] == parquet_rows(out)
    assert not mirror_csv(out).exists()

def test_the_schema_cache_prefers_the_tsv_and_falls_back(tmp_path): ...
def test_a_cli_retirement_during_discovery_forces_a_retry(tmp_path): ...   # Minor 11
def test_filtered_state_writes_a_tsv(tmp_path): ...                         # Minor 8
```
- [ ] **Step 2: Run the four test files; the new tests fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the four files, `tests/unit/gui/results_viewer/`, `tests/gui/results_viewer/`
  and `tests/unit/gui/test_curation_does_not_unfence_the_run.py`; all pass.**
- [ ] **Step 5: Commit.**

### PA3 gate

- Per phase: every importer of `tests/_output_layout.py`, plus `tests/unit/cli/`,
  `tests/integration/cli/` (`test_deliverables_byte_stable.py` must pass: the CSV is gone from
  both runs), `tests/unit/sdk_/`, `tests/gui/results_viewer/`, `tests/unit/gui/results_viewer/`.
  Slurm, detached worktree.
- **Deep review** (`implementation-test-reviewer`, Opus) over PA3. Report: `pa3-review.md`.
