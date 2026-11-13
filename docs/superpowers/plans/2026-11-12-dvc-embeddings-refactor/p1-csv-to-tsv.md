# P1 · CSV → TSV

**Spec:** Part B (B1–B8)

---

**Goal:** Every tabular text file PhenoTypic generates is TSV, with one shared
writer and one shared delimiter rule. Inputs accept `.csv` or `.tsv`. The
metadata snapshot keeps the user's bytes and extension. This is a hard cutover
for external readers; PhenoTypic's own readers still fall back to `.csv` in
older trees.

### Task 1: The delimited-text module

**Files:**
- Create: `src/phenotypic/sdk_/_delimited.py`
- Modify: `src/phenotypic/sdk_/__init__.py`. These are pure-stdlib or lazily
  importing helpers; put a helper in `_LAZY_ATTRS` if it imports polars or pandas.
- Test: `tests/unit/sdk_/test_delimited.py` (new)

**Interfaces:**
- Produces:
  - `TSV_SUFFIX = ".tsv"`, `CSV_SUFFIX = ".csv"`
  - `INPUT_TABLE_SUFFIXES = frozenset({".csv", ".tsv"})`
  - `delimiter_for(path: Path) -> str`: `"\t"` for `.tsv` and `","` for `.csv`,
    case-insensitive; anything else raises `ValueError` naming both allowed
    suffixes
  - `write_tsv(frame: pl.DataFrame | pd.DataFrame, path: Path) -> None`: atomic,
    using `atomic_write_with_writer` with temp suffix `.tsv.tmp`
  - `read_delimited(path: Path, **pandas_kwargs) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

```python
def test_delimiter_for():
    assert delimiter_for(Path("m.tsv")) == "\t"
    assert delimiter_for(Path("m.CSV")) == ","
    with pytest.raises(ValueError, match=r"\.csv or \.tsv"):
        delimiter_for(Path("m.xlsx"))

@pytest.mark.parametrize("lib", ["polars", "pandas"])
def test_write_tsv_quotes_tabs_and_newlines(tmp_path, lib):
    frame = make_frame(lib, {"a": ["x\ty", "line1\nline2", 'q"q'], "b": [1, 2, 3]})
    out = tmp_path / "t.tsv"
    write_tsv(frame, out)
    back = read_delimited(out)
    assert back["a"].tolist() == ["x\ty", "line1\nline2", 'q"q']
    assert out.read_bytes().splitlines()[0] == b"a\tb"
```

- [ ] **Step 2: Run them and confirm they fail** (the module doesn't exist).
- [ ] **Step 3: Implement.** Polars uses `write_csv(path, separator="\t")`;
  pandas uses `to_csv(path, sep="\t", index=False, lineterminator="\n")`. Both
  quote by default.
- [ ] **Step 4: Run again; it passes.**
- [ ] **Step 5: Commit.**

### Task 2: Deliverable names, constants and the writers

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py`:
  - :331 add `MEASUREMENTS_TSV = "measurements.tsv"`. `MEASUREMENTS_CSV` stays
    defined and exported as the legacy read name, because the GUI imports it
    (#283).
  - :410 `named_analysis_csv_path` → `named_analysis_tsv_path`
  - :927 `ERROR_ANALYSIS_CSV` → `ERROR_ANALYSIS_TSV`
  - :1466 `measurements_csv_path` → `measurements_tsv_path`
  - :2514 the error-analysis path
  - `BundleLayout` (:2987, :3069)
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:752-782` (analysis),
  `:944-970` `_seed_measurements`, and `:1425-1450` `_write_split`
- Modify: `src/phenotypic/plotting/_pipeline/_analysis_artifacts.py:37-111`
  (manifest fields `csv`/`csv_sha256` → `tsv`/`tsv_sha256`,
  `schema_version` + 1)
- Modify: `src/phenotypic/_cli/_cli_completion.py:1133` (the aggregate proof's
  artifact key `measurements_csv` → `measurements_tsv`)
- Test: `tests/unit/cli/test_deliverables_tsv.py` (new), plus the existing
  analysis-manifest tests

**Interfaces:**
- Consumes: Task 1's `write_tsv`.
- Produces:
  - `measurements_tsv_path(output_dir)`
  - `named_analysis_tsv_path(base, analysis_id)`
  - `error_analysis_tsv_path(output_dir)`
  - `AnalysisManifestEntry.tsv` and `.tsv_sha256`
  - `ANALYSIS_MANIFEST_SCHEMA_VERSION` (current + 1)

- [ ] **Step 1: Write the failing tests**

```python
def test_finalize_writes_tsv_deliverables(finished_tiny_run):
    d = finished_tiny_run / "deliverables"
    assert (d / "measurements.tsv").is_file()
    assert not list(d.rglob("*.csv"))
    assert {p.suffix for p in (d / "measurements_by_feature").iterdir()} == {".tsv", ".parquet"}
    assert (d / "measurements.tsv").read_bytes().split(b"\n", 1)[0].count(b"\t") > 0

def test_analysis_manifest_records_tsv(finished_tiny_run_with_model):
    entry = read_analysis_manifest(finished_tiny_run_with_model / "deliverables").entries[0]
    assert entry.tsv.endswith(".tsv") and len(entry.tsv_sha256) == 64

def test_old_manifest_with_csv_fields_still_reads(tmp_path):
    write_v_prev_manifest(tmp_path, csv="Fit.csv", csv_sha256="0" * 64)
    assert read_analysis_manifest(tmp_path).entries[0].tsv == "Fit.csv"
```

  `finished_tiny_run` is a fixture that runs `python -m phenotypic` locally over
  `load_synth_yeast_plate()`. Reuse the existing CLI end-to-end fixture in
  `tests/unit/cli/conftest.py` if one exists.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - Every writer above calls `write_tsv`.
  - The manifest reader maps legacy `csv`/`csv_sha256` onto the new fields.
- [ ] **Step 4: Run again, plus the importers of `_analysis_artifacts` and
  `_cli_output_manager`.** They pass.
- [ ] **Step 5: Commit.**

### Task 3: User inputs and the byte-exact snapshots

**Files:**
- Modify: `src/phenotypic/_cli/_cli_reference.py:353-430`:
  - `metadata_csv_payload` accepts `INPUT_TABLE_SUFFIXES`;
  - the snapshot keeps the input's extension.
- Modify: `src/phenotypic/sdk_/_io_constants.py:343, :728, :1099, :1476`. The
  snapshot path functions take the suffix, and add
  `resolve_metadata_snapshot_path(output_dir) -> Path | None` and
  `resolve_reference_metadata_snapshot_path(output_dir) -> Path | None`.
- Modify: callers that use those paths:
  - `_cli_migrate.py:686,1049,1357,1722`
  - `_cli_identity.py:142`
  - `_hdf_to_zarr.py:507`
  - `_cli_output_manager.py` (`metadata_csv_deliverable_path` uses)
  - `phenotypicCLI.py` (`_snapshot_metadata_csv`)
  - the `--metadata` click option's help text
- Modify: `src/phenotypic/sdk_/_metadata_migration.py:469` and `:2935` (read by
  suffix; the canonical view is written as `metadata.canonical.tsv`)
- Modify: `src/phenotypic/tune/score/_gt_loader.py` (read by suffix)
- Test: `tests/unit/cli/test_metadata_snapshot_suffix.py` (new); keep
  `test_the_metadata_snapshot_is_byte_unchanged_by_a_full_migrate` green.

**Interfaces:**
- Consumes: Task 1's `delimiter_for` and `INPUT_TABLE_SUFFIXES`.
- Produces:
  - `metadata_snapshot_path(output_dir, suffix: str) -> Path`
  - `resolve_metadata_snapshot_path(output_dir) -> Path | None`, which looks for
    `.tsv` then `.csv`; exactly one may exist

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("ext", [".csv", ".tsv"])
def test_snapshot_keeps_bytes_and_extension(tmp_path, ext, metadata_bytes):
    src = tmp_path / f"meta{ext}"; src.write_bytes(metadata_bytes[ext])
    run_full(tmp_path / "out", metadata=src)
    snap = tmp_path / "out" / "deliverables" / f"metadata{ext}"
    assert snap.read_bytes() == src.read_bytes()
    assert resolve_metadata_snapshot_path(tmp_path / "out") == snap

def test_metadata_with_other_suffix_is_refused(tmp_path):
    bad = tmp_path / "meta.xlsx"; bad.write_bytes(b"x")
    result = invoke_cli(["--metadata", str(bad), ...])
    assert result.exit_code == 2 and ".csv or .tsv" in result.output

def test_tsv_metadata_joins_like_csv(tmp_path, metadata_bytes):
    for ext in (".csv", ".tsv"):
        (tmp_path / f"meta{ext}").write_bytes(metadata_bytes[ext])   # same rows, two delimiters
        run_full(tmp_path / f"out{ext}", metadata=tmp_path / f"meta{ext}")
    read = lambda ext: pl.read_parquet(tmp_path / f"out{ext}" / "deliverables" / "measurements.parquet")
    assert read(".tsv").equals(read(".csv"))
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** Snapshots copy bytes, so nothing is reparsed;
  readers call `read_delimited`.
- [ ] **Step 4: Run them, plus the importers of `_cli_reference` and
  `_metadata_migration`.** They pass.
- [ ] **Step 5: Commit.**

### Task 4: Old trees and docs

**Files:**
- Keep (no edit): everything under `src/phenotypic/_gui/` (issue #283).
  `MEASUREMENTS_CSV` stays exported from `phenotypic.sdk_` as the legacy read
  name, so `_gui/_config.py:68` still imports.
- Modify: non-GUI readers that open the mirror. Find them with
  `grep -rn "measurements_csv_path\|MEASUREMENTS_CSV\|measurements.csv" src/phenotypic --include=*.py | grep -v _gui/`.
  Each reader falls back to the legacy `.csv` name when no `.tsv` exists.
- Modify: the finalizer. After publishing `measurements.tsv` and the split
  `.tsv` files, it unlinks the stale generated `.csv` siblings in the B2 list.
  It never touches `metadata.*`.
- Modify: docs. Find them with
  `grep -rln "measurements\.csv\|by_feature/.*\.csv\|\.csv\b" docs/source src/phenotypic/_cli/_cli_readme_generator.py`.
  Also update the root and `_cli` CLAUDE.md output inventories.
- Modify: the release note. The deliverables `README.md` generator states the
  rename.
- Test: `tests/unit/cli/test_tsv_cutover_old_tree.py` (new). The existing GUI
  results-viewer tests must stay green, unmodified.

- [ ] **Step 1: Write the failing tests**

```python
def test_bundle_layout_finds_a_pre_cutover_mirror(old_tree_with_measurements_csv):
    layout = BundleLayout.detect(old_tree_with_measurements_csv)
    assert layout.measurements_table_path().name == "measurements.csv"

def test_refinalize_old_tree_replaces_generated_csv(old_tree_with_measurements_csv):
    run_recompile(old_tree_with_measurements_csv)
    d = old_tree_with_measurements_csv / "deliverables"
    assert (d / "measurements.tsv").is_file() and not (d / "measurements.csv").exists()
    assert (d / "metadata.csv").read_bytes() == ORIGINAL_METADATA_BYTES
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement** the fallback reads (`BundleLayout` gains
  `measurements_table_path()`: `.tsv` first, then `.csv`) and the stale-CSV
  removal.
- [ ] **Step 4: Run the phase gate.** Run the CLI surface plus `tests/unit/gui`
  and `tests/gui` once; the GUI tests must pass unmodified. Then run
  `tests/unit/test_docs_preflight_codes.py`.
- [ ] **Step 5: Commit.**
