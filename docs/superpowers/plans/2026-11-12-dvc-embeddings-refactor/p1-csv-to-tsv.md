# P1 · CSV → TSV

**Spec:** Part B (B1–B9)

---

**Goal:** Every tabular text file the CLI generates is TSV, with one shared
writer and one shared delimiter rule. Inputs accept `.csv` or `.tsv`. The
metadata snapshot keeps the user's bytes and extension. PhenoTypic's own
readers still fall back to `.csv` in older trees.

**What stays CSV until #283** (spec B9), because GUI code writes it:
- `error_analysis.csv`. `_cli/_cli_error_outputs.py:81-96` calls the GUI's
  `publish_error_analysis`, which names the file itself
  (`_gui/results_viewer/_error_tab/_publication.py:849`) and writes it with
  `combined.write_csv` (`:588`). P1 does not touch `ERROR_ANALYSIS_CSV` or
  `error_analysis_csv_path`.
- The curated mirror `measurements.csv`. GUI curation rewrites
  `BundleLayout.mirror_csv` on every save
  (`_gui/results_viewer/_curation_labels.py:839-850`). Finalize writes **both**
  `measurements.tsv` and `measurements.csv`, so the GUI keeps a current file to
  curate. Until #283, curated edits reach the mirror's `.parquet` and `.csv`
  but not its `.tsv`.

**Legacy names stay defined, with their signatures** (spec B8). New constants
and functions go **beside** the old ones; none is renamed or re-signed.

### Task 1: The delimited-text module and the GUI import guard

**Files:**
- Create: `src/phenotypic/sdk_/_delimited.py`
- Modify: `src/phenotypic/sdk_/__init__.py`. These are pure-stdlib or lazily
  importing helpers; put a helper in `_LAZY_ATTRS` if it imports polars or pandas.
- Create: `tests/unit/sdk_/test_gui_sdk_imports.py` and its baseline
  `tests/unit/sdk_/gui_sdk_signatures.json`
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
  - `gui_sdk_imports(gui_root: Path) -> dict[str, set[str]]` (test-only, in the
    guard test module): every name imported from `phenotypic.sdk_` or a
    `phenotypic.sdk_.<module>` by a file under `gui_root`, keyed by module

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

  `make_frame(lib, data)` is a module-local helper that builds a polars or
  pandas frame.

```python
# tests/unit/sdk_/test_gui_sdk_imports.py  (spec B8)
GUI_ROOT = Path(phenotypic.__file__).parent / "_gui"
BASELINE = json.loads((Path(__file__).parent / "gui_sdk_signatures.json").read_text(encoding="utf-8"))

def test_every_gui_import_from_sdk_still_resolves():
    for module, names in gui_sdk_imports(GUI_ROOT).items():
        mod = importlib.import_module(module)
        for name in names:
            assert hasattr(mod, name), f"{module}.{name} is imported by _gui/ and no longer exists"

def test_every_sdk_function_the_gui_calls_keeps_its_signature():
    for qualified, signature in BASELINE.items():
        module, _, name = qualified.rpartition(".")
        obj = getattr(importlib.import_module(module), name)
        assert str(inspect.signature(obj)) == signature, qualified
```

- [ ] **Step 2: Generate the signature baseline from `main`.** Check out
  `main` @ `3248a9173` in a detached worktree, and from it write
  `gui_sdk_signatures.json`: `{"<module>.<name>": str(inspect.signature(obj))}`
  for every **callable** the walk finds, classes excluded. Commit the JSON as
  generated. It records the signatures the GUI was written against.
- [ ] **Step 3: Run the tests and confirm they fail** (`_delimited` doesn't
  exist). The guard tests pass already; they exist to fail later.
- [ ] **Step 4: Implement.** Polars uses `write_csv(path, separator="\t")`;
  pandas uses `to_csv(path, sep="\t", index=False, lineterminator="\n")`. Both
  quote by default.
- [ ] **Step 5: Run again; it passes.**
- [ ] **Step 6: Commit.**

### Task 2: Deliverable names, constants and the writers

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py`:
  - :331 add `MEASUREMENTS_TSV = "measurements.tsv"` beside `MEASUREMENTS_CSV`
  - :410 add `named_analysis_tsv_path(base, analysis_id)` beside
    `named_analysis_csv_path`
  - :1466 add `measurements_tsv_path(output_dir)` beside `measurements_csv_path`
  - `BundleLayout` (:2985-2987): add `mirror_tsv`; `mirror_csv` keeps pointing
    at `measurements.csv`
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:752-782` (analysis),
  `:944-970` `_seed_measurements`, and `:1425-1450` `_write_split`. The mirror
  is written as `.tsv` **and** `.csv`; splits and analysis tables as `.tsv`
  only.
- Modify: `src/phenotypic/plotting/_pipeline/_analysis_artifacts.py:37-111`
  (manifest fields `csv`/`csv_sha256` → `tsv`/`tsv_sha256`,
  `schema_version` + 1)
- Modify: `src/phenotypic/_cli/_cli_completion.py:1131-1135`. The aggregate
  proof's `required_paths` gains `"measurements_tsv": measurements_tsv_path(output_dir)`;
  `measurements_csv` stays.
- Modify: `src/phenotypic/sdk_/_run_state.py:1236-1238`.
  `_GUI_WRITTEN_PROOF_DESCRIPTORS` gains `"measurements_tsv"`, so a curation
  does not invalidate the run proof.
- Modify: `tests/_output_layout.py:80-95`, `write_measurements_mirror`, gains
  `tsv: bool = True` and writes `measurements.tsv`, so the shared fixture
  matches the proof's required outputs. It has 61 importers, GUI tests
  included.
- Create (helpers), in `tests/_tiny_run.py`:
  - fixture `tiny_inputs`: `<tmp>/in/ds/plate_{1,2}.tif`, written from
    `load_synth_yeast_plate()`
  - `run_cli(inputs, out, *, mode="full", pipeline=None, slurm=False, extra=()) -> click.testing.Result`:
    invokes the CLI with `CliRunner`. `inputs=None` omits `--input`.
    `--pipeline` is passed for `full`, `measure` and `process`, defaulting to
    the `tiny_pipeline` written to a file. `slurm=True` adds
    `--slurm partition=intel`.
  - fixture `finished_tiny_run`: `run_cli(tiny_inputs, tmp_path / "out")`,
    asserts exit `0`, returns the output root
- Test: `tests/unit/cli/test_deliverables_tsv.py` (new), and the manifest tests
  in `tests/unit/plotting/test_analysis_artifacts.py`

**Interfaces:**
- Consumes: Task 1's `write_tsv`.
- Produces:
  - `MEASUREMENTS_TSV`, `measurements_tsv_path(output_dir)`
  - `named_analysis_tsv_path(base, analysis_id)`
  - `BundleLayout.mirror_tsv`
  - `AnalysisManifestEntry.tsv` and `.tsv_sha256`
  - `ANALYSIS_MANIFEST_SCHEMA_VERSION` (current + 1)

- [ ] **Step 1: Write the failing tests**

```python
CSV_THE_CLI_MAY_LEAVE = {"measurements.csv", "metadata.csv", "error_analysis.csv"}

def test_finalize_writes_tsv_deliverables(finished_tiny_run):
    d = finished_tiny_run / "deliverables"
    assert (d / "measurements.tsv").is_file() and (d / "measurements.csv").is_file()
    assert {p.name for p in d.rglob("*.csv")} <= CSV_THE_CLI_MAY_LEAVE
    assert {p.suffix for p in (d / "measurements_by_feature").iterdir()} == {".tsv", ".parquet"}
    assert (d / "measurements.tsv").read_bytes().split(b"\n", 1)[0].count(b"\t") > 0

def test_a_curation_does_not_stale_the_proof(finished_tiny_run):
    d = finished_tiny_run / "deliverables"
    for name in ("measurements.csv", "measurements.tsv"):
        (d / name).write_bytes((d / name).read_bytes() + b"\n")   # what a curation save changes
    assert aggregate_proof_refusal(finished_tiny_run) is None

# tests/unit/plotting/test_analysis_artifacts.py
def test_analysis_manifest_records_tsv(tmp_path):
    write_tsv(pl.DataFrame({"a": [1]}), tmp_path / "Fit.tsv")
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "Fit.parquet")
    entry = build_analysis_manifest_entry(analysis_id="Fit", producer_class="Fit",
                                          tsv_path=tmp_path / "Fit.tsv",
                                          parquet_path=tmp_path / "Fit.parquet", rows=1, columns=["a"])
    write_analysis_manifest(tmp_path, AnalysisManifest(analyses={"Fit": entry}))
    assert read_analysis_manifest(tmp_path).analyses["Fit"].tsv == "Fit.tsv"

def test_old_manifest_with_csv_fields_still_reads(tmp_path):
    old_entry = {"class": "Fit", "csv": "Fit.csv", "parquet": "Fit.parquet", "rows": 1,
                 "columns": ["a"], "csv_sha256": "0" * 64, "parquet_sha256": "0" * 64}
    payload = {"schema_version": ANALYSIS_MANIFEST_SCHEMA_VERSION - 1, "analyses": {"Fit": old_entry}}
    (tmp_path / "analysis_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert read_analysis_manifest(tmp_path).analyses["Fit"].tsv == "Fit.csv"
```

  The old entry's keys are today's `_ENTRY_FIELDS` (`_analysis_artifacts.py:34-44`).
  `build_analysis_manifest_entry` takes `tsv_path` in place of `csv_path`
  (`:274-290`).
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - Every writer above calls `write_tsv`; the mirror's `.csv` keeps its
    current writer.
  - The manifest reader maps legacy `csv`/`csv_sha256` onto the new fields.
- [ ] **Step 4: Run again, plus the test surface by grep.** It is every file
  under `tests/` that matches
  `grep -rlE "measurements\.csv|MEASUREMENTS_CSV|measurements_csv_path|_analysis_artifacts|_cli_output_manager" tests`
  (21 files match the first three names alone today). They pass, except the
  README's P0 expected-red rows.
- [ ] **Step 5: Commit.**

### Task 3: User inputs and the byte-exact snapshots

**Files:**
- Modify: `src/phenotypic/phenotypicCLI.py:2050-2060`. The `--metadata` option
  gains `callback=_metadata_suffix_callback`, which raises `click.BadParameter`
  for any suffix outside `INPUT_TABLE_SUFFIXES`. Today the option is a bare
  `click.Path` and checks no suffix.
- Modify: `src/phenotypic/_cli/_cli_reference.py:353-430`:
  - `metadata_csv_payload` accepts `INPUT_TABLE_SUFFIXES`, reading by
    `delimiter_for` (`:400` assumes comma today);
  - the snapshot keeps the input's extension.
- Modify: `src/phenotypic/sdk_/_io_constants.py:343, :728, :1099, :1476`. Add,
  beside the unchanged one-argument `metadata_csv_deliverable_path(output_dir)`
  and `reference_metadata_snapshot_path(output_dir)`:
  - `metadata_snapshot_path(output_dir, suffix: str) -> Path`
  - `reference_snapshot_path(output_dir, suffix: str) -> Path`
  - `resolve_metadata_snapshot_path(output_dir) -> Path | None`
  - `resolve_reference_metadata_snapshot_path(output_dir) -> Path | None`
- Modify: the non-GUI callers of the snapshot paths, which move to the
  resolvers or the suffixed writers:
  - `_cli_migrate.py:686,1049,1357,1722`
  - `_cli_identity.py:142`
  - `_hdf_to_zarr.py:507`
  - `_cli_output_manager.py` (`metadata_csv_deliverable_path` uses)
  - `phenotypicCLI.py:939` (`_snapshot_metadata_csv`)
  - the `--metadata` option's help text
- Modify: `src/phenotypic/sdk_/_metadata_migration.py:469` (read by suffix) and
  `:2935` (the canonical view is written as `metadata.canonical.tsv`)
- Modify: `src/phenotypic/tune/score/_gt_loader.py` (read by suffix)
- Create (helpers), in `tests/_tiny_run.py`: fixture `metadata_bytes`, a dict
  `{".csv": …, ".tsv": …}` holding the same rows with two delimiters, keyed to
  `tiny_inputs`' image names
- Test: `tests/unit/cli/test_metadata_snapshot_suffix.py` (new); keep
  `test_the_metadata_snapshot_is_byte_unchanged_by_a_full_migrate` green.

**Interfaces:**
- Consumes: Task 1's `delimiter_for` and `INPUT_TABLE_SUFFIXES`.
- Produces: the four functions above. Each resolver looks for `.tsv` then
  `.csv`; exactly one may exist.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("ext", [".csv", ".tsv"])
def test_snapshot_keeps_bytes_and_extension(tmp_path, tiny_inputs, ext, metadata_bytes):
    src = tmp_path / f"meta{ext}"; src.write_bytes(metadata_bytes[ext])
    assert run_cli(tiny_inputs, tmp_path / "out", extra=["--metadata", str(src)]).exit_code == 0
    snap = tmp_path / "out" / "deliverables" / f"metadata{ext}"
    assert snap.read_bytes() == src.read_bytes()
    assert resolve_metadata_snapshot_path(tmp_path / "out") == snap

def test_metadata_with_other_suffix_is_refused(tmp_path, tiny_inputs):
    bad = tmp_path / "meta.xlsx"; bad.write_bytes(b"x")
    result = run_cli(tiny_inputs, tmp_path / "out", extra=["--metadata", str(bad)])
    assert result.exit_code == 2 and ".csv or .tsv" in result.output

def test_tsv_metadata_joins_like_csv(tmp_path, tiny_inputs, metadata_bytes):
    for ext in (".csv", ".tsv"):
        (tmp_path / f"meta{ext}").write_bytes(metadata_bytes[ext])
        run_cli(tiny_inputs, tmp_path / f"out{ext}", extra=["--metadata", str(tmp_path / f"meta{ext}")])
    read = lambda ext: pl.read_parquet(tmp_path / f"out{ext}" / "deliverables" / "measurements.parquet")
    assert read(".tsv").equals(read(".csv"))

def test_one_argument_snapshot_path_is_unchanged(tmp_path):
    assert metadata_csv_deliverable_path(tmp_path) == tmp_path / "deliverables" / "metadata.csv"
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** Snapshots copy bytes, so nothing is reparsed;
  readers call `read_delimited`.
- [ ] **Step 4: Run them, plus the importers of `_cli_reference` and
  `_metadata_migration`, and Task 1's guard.** They pass.
- [ ] **Step 5: Commit.**

### Task 4: Old trees and docs

**Files:**
- Keep (no edit): everything under `src/phenotypic/_gui/` (issue #283).
- Modify: non-GUI readers that open the mirror text file. Find them with
  `grep -rn "measurements_csv_path\|MEASUREMENTS_CSV\|measurements.csv" src/phenotypic --include=*.py | grep -v _gui/`.
  Each reads `.tsv` and falls back to `.csv` when no `.tsv` exists.
- Modify: the finalizer. After publishing the TSVs, it unlinks the stale
  generated `.csv` siblings of the B2 files it rewrites as TSV only: the
  `measurements_by_feature/` and `measurements_by_category/` splits,
  `<AnalysisClass>.csv`, and `metadata.canonical.csv`. It never touches
  `measurements.csv` (finalize rewrites it, B9), `metadata.*` or
  `error_analysis.csv`.
- Modify: docs. Find them with
  `grep -rln "measurements\.csv\|by_feature/.*\.csv\|\.csv\b" docs/source src/phenotypic/_cli/_cli_readme_generator.py`.
  Update the root and `_cli` CLAUDE.md output inventories, including the two
  files that stay CSV and why.
- Modify: the release note. The deliverables `README.md` generator states the
  rename, and that curated edits reach `measurements.csv` but not
  `measurements.tsv` until #283.
- Create (helpers), in `tests/_tiny_run.py`: fixture
  `old_tree_with_measurements_csv`, built from `finished_tiny_run` with a
  `--metadata` CSV: it deletes every generated `.tsv` and writes each one's
  content back as the pre-cutover `.csv` name, leaving a tree as a pre-P1
  finalize would have.
- Test: `tests/unit/cli/test_tsv_cutover_old_tree.py` (new).

- [ ] **Step 1: Write the failing tests**

```python
def test_bundle_layout_finds_a_pre_cutover_mirror(old_tree_with_measurements_csv):
    layout = BundleLayout.detect(old_tree_with_measurements_csv)
    assert layout.measurements_table_path().name == "measurements.csv"

def test_refinalize_old_tree_replaces_generated_csv(old_tree_with_measurements_csv):
    d = old_tree_with_measurements_csv / "deliverables"
    metadata_before = (d / "metadata.csv").read_bytes()
    run_cli(None, old_tree_with_measurements_csv, mode="recompile")
    assert (d / "measurements.tsv").is_file() and (d / "measurements.csv").is_file()
    assert not list((d / "measurements_by_feature").glob("*.csv"))
    assert (d / "metadata.csv").read_bytes() == metadata_before
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement** the fallback reads (`BundleLayout` gains
  `measurements_table_path()`: `.tsv` first, then `.csv`) and the stale-CSV
  removal.
- [ ] **Step 4: Run the phase gate.** Run the CLI surface, plus `tests/unit/gui`
  and `tests/gui` once. The only red GUI tests are the README's P0 rows; any
  other red GUI test is a real failure. Then run
  `tests/unit/test_docs_preflight_codes.py`.
- [ ] **Step 5: Commit.**
