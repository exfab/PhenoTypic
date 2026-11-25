# PA4 · Typed config suffixes and TSV-capable pickers

**Spec:** PA6, PA9 (D8), §3.3 reds #1–6; review I8, Minor 7

---

**Goal:** Every mounted listing surface agrees with the sdk_ suffix sets and never lists a
tune spec as a pipeline; reds #1–6 are green; every metadata picker and reader takes a `.tsv`.

**Depends on:** PA3, on the main line. PA1 T4 and PA4 both edit `run_console/_callbacks.py`, and
PA3 T1 and PA4 T1 both edit `sdk_/_io_constants.py` and `tests/unit/sdk_/test_io_constants.py`
(plan review I8).

### Task 1: `is_pipeline_config_path` and the suffix surfaces (C9)

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (after `matches_any_suffix`, `:162-164`) and
  `sdk_/__init__.py`: add `is_pipeline_config_path`.
- Modify: `_gui/builder/_directory_browser.py:32, 151-158, 259` (`directory_tree(..., predicate=None)`);
  the pipeline modal callers (`builder/_callbacks.py:7028`, `builder/_modal_browser.py:140`),
  through the wrappers whose keyword lists are fixed: `_render_tree_body`
  (`builder/_callbacks.py:4019-4043`) and `_initial_tree_body` (`builder/_modal_browser.py:189-200`).
- Modify: `_gui/run_console/_form.py:58-73` (`render_pipeline_tree` passes
  `predicate=is_pipeline_config_path`; plan review I4).
- Modify: `_gui/shell/_classifier.py:220-235`; `_gui/run_console/_callbacks.py:2897-2899`.
- Modify: `_gui/tune/_validation.py:61-67`; `_gui/tune/_setup_authoring.py:353-358, 511-536` (`authored_setup_spec_path`).
- Modify: `_gui/analysis/_recipe_state.py:1159` (retire the old typed sibling after publish).
- Modify: `scripts/capture_gui_tutorial_screenshots.py:52, 229, 1112` (canonical name).
- Modify: `tests/e2e/gui/**` fixtures that write `.json.pht-*` (`grep -rln 'json\.pht-' tests/e2e`).
- Test: `tests/unit/sdk_/test_io_constants.py`, `tests/unit/gui/builder/`,
  `tests/unit/gui/shell/test_classifier.py`, `tests/unit/gui/tune/` (reds #1–3),
  `tests/integration/gui/tune/test_setup_view.py` (reds #4–6), `tests/unit/gui/analysis/`.

**Interfaces:**
- Consumes: `PIPELINE_CONFIG_SUFFIXES`, `TUNING_CONFIG_SUFFIXES`, `CONFIG_SUFFIXES`,
  `LEGACY_TYPED_CONFIG_SUFFIXES` (`_io_constants.py:110-148`); `old_typed_config_sibling`
  (`:211`); `_pipeline_config_candidates_in` (used by `QcRecipe._retire_old_typed_sibling`,
  `sdk_/_qc_recipe/_recipe.py:546-560`).
- Produces:
  - `is_pipeline_config_path(path: str | Path) -> bool`: true iff `path` matches
    `PIPELINE_CONFIG_SUFFIXES` and matches none of `.pht-op.json`, `.pht-cc.json`,
    `.pht-tune.json`, `.json.pht-op`, `.json.pht-cc`, `.json.pht-tune` (case-insensitive).
  - `directory_tree(root, current=None, *, extensions=IMAGE_EXTS, predicate: Callable[[Path], bool] | None = None, select_files=True, id_type="dir-entry")`:
    a file is selectable when it passes both.
  - `TUNING_SPEC_TYPED_SUFFIXES = TUNING_CONFIG_SUFFIXES - {LEGACY_JSON_SUFFIX}` (derived from
    the sdk_ set, never spelled again; module constant in `tune/_validation.py`), used by
    `spec_path_issue` and `load_pipeline_or_spec`.

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize("name,expected", [
    ("p.pht-pipe.json", True), ("p.json.pht-pipe", True), ("pipeline.json", True),
    ("x.pht-tune.json", False), ("x.json.pht-tune", False), ("x.pht-op.json", False),
    ("x.pht-cc.json", False), ("metrics.JSON", True), ("P.PHT-TUNE.JSON", False),
])
def test_is_pipeline_config_path(name, expected):
    assert is_pipeline_config_path(name) is expected

def test_the_pipeline_browser_hides_a_tune_spec(tmp_path):
    (tmp_path / "a.pht-pipe.json").write_text("{}"); (tmp_path / "s.pht-tune.json").write_text("{}")
    assert listed_names(directory_tree(tmp_path, extensions=PIPELINE_EXTS, predicate=is_pipeline_config_path)) == {"a.pht-pipe.json"}

@pytest.mark.parametrize("render", ["builder_render_tree_body", "builder_initial_tree_body", "run_console_pipeline_tree"])
def test_every_pipeline_picker_hides_a_tune_spec(tmp_path, render):          # I4
    seed(tmp_path, "a.pht-pipe.json", "s.pht-tune.json")
    assert listed_names(RENDERERS[render](tmp_path)) == {"a.pht-pipe.json"}

def test_a_tune_spec_holding_pipe_cfgs_is_not_a_pipeline(tmp_path):
    p = tmp_path / "s.pht-tune.json"; p.write_text('{"pipeline": {"pipe_cfgs": {}}}')
    assert classify(str(p)).has_pipeline_json is False

def test_the_deploy_message_names_the_canonical_suffix():
    assert ".pht-tune.json" in spec_path_issue("plain.json").message

def test_the_preset_writer_uses_the_canonical_suffix(tmp_path):
    assert authored_setup_spec_path(tmp_path, source_path=..., authored_content="").name.endswith(".setup.pht-tune.json")   # :511

def test_recipe_save_retires_the_old_typed_pipeline(tmp_path):
    out = run_with_old_typed_pipeline(tmp_path)            # deliverables/pipeline.json.pht-pipe
    RecipeState.load(out).save()
    assert (out / "deliverables/pipeline.pht-pipe.json").is_file()
    assert not (out / "deliverables/pipeline.json.pht-pipe").exists()

def test_recipe_save_never_removes_a_plain_pipeline_json(tmp_path): ...
def test_a_failed_recipe_publish_keeps_the_old_file(tmp_path, monkeypatch): ...
```
- [ ] **Step 2: Run the new tests and reds #1–6; all fail.**
- [ ] **Step 3: Implement.** The tune edits stop at the validator, the loader and the preset
  writer; every other tune surface is in the spec's §8 follow-up (Minor 7).
- [ ] **Step 4: Run every file in **Test** above; all pass, reds #1–6 included.**
- [ ] **Step 5: Run the Playwright run-console and builder specs once** after the e2e fixture
  flip: `PLAYWRIGHT=1 $PT tests/e2e/gui/test_run_console.py tests/e2e/gui/builder -m "not slow and not ci_flaky"`.
- [ ] **Step 6: Commit.**

### Task 2: TSV-capable pickers and readers (C10)

**Files:** every site in spec PA9's table:
- `_gui/shell/_source_picker.py:13, 109`; `_gui/shell/_metadata_context.py:722, 749-786`;
- `_gui/builder/_reference_metadata.py:39-44, 72`;
- `_gui/run_console/_callbacks.py:305-327` (`reference_metadata_requirement`, I8);
- `_gui/run_console/_request_safety.py:395-410` (the refusal message);
- `_gui/analysis/_callbacks.py:95` (`MergeMetadata` default);
- `_gui/tune/_setup_authoring.py:46`, `_gui/tune/_layout.py:330`, `_gui/tune/_callbacks.py:1254`
  (unmounted, D8);
- `src/phenotypic/analysis/qc/_expected_vs_detected.py:79, 110, 612-620`.
- Test: `tests/unit/gui/shell/test_metadata_context.py`, `tests/unit/gui/shell/test_source_chrome.py`,
  `tests/unit/gui/builder/test_reference_metadata.py`, `tests/unit/gui/run_console/test_reference_requirement.py`,
  `tests/unit/gui/run_console/test_request_safety.py`, `tests/unit/gui/analysis/`,
  `tests/unit/analysis/test_expected_vs_detected_optional_metadata.py`, `tests/gui/browse/`.

**Interfaces:**
- Consumes: `INPUT_TABLE_SUFFIXES`, `delimiter_for`, `read_delimited`
  (`sdk_/_delimited.py:37, 42, 224`); `resolve_metadata_snapshot_path` (`sdk_/_io_constants.py:1764`).
- Produces:
  - `read_metadata_csv_table(path: Path) -> tuple[list[str], list[dict[str, str]]]`, same
    signature; `csv.reader(handle, delimiter=delimiter_for(path))`.
  - `reference_metadata_requirement(...)` is met by `resolve_metadata_snapshot_path(output_dir)`;
    its `ValueError` (both suffixes) becomes the requirement text.
  - The `MergeMetadata` default is computed at instantiation, not in the static dict
    (`analysis/_callbacks.py:81-96`): `_instantiate(kind, class_name, *, output_dir: Path | None = None)`
    (`:1070`) replaces `metadata_path` with `resolve_metadata_snapshot_path(output_dir).name` for
    `MergeMetadata` when the bound output has a snapshot, else keeps `"metadata.csv"`. Its callers
    pass the bound output's root (Minor 10).

- [ ] **Step 1: Write the failing tests.**

```python
def test_each_mounted_picker_lists_a_tsv(tmp_path): ...    # shared picker, reference table
def test_read_metadata_csv_table_reads_a_tsv_like_its_csv(tmp_path):
    assert read_metadata_csv_table(write_tsv_of(tmp_path, ROWS)) == read_metadata_csv_table(write_csv_of(tmp_path, ROWS))

def test_an_excel_tsv_reads_like_its_csv(tmp_path):                         # Review Focus 3
    tsv = write_bytes(tmp_path / "plates.tsv", b"\xef\xbb\xbfMetadata_ImageName\tNote\nplate1\t\"a\tb\"\n")
    cols, rows = read_metadata_csv_table(tsv)
    assert cols[0] == "Metadata_ImageName" and rows[0]["Note"] == "a\tb"

def test_a_txt_table_is_refused_naming_both_suffixes(tmp_path):
    assert ".csv or .tsv" in metadata_preflight_message(tmp_path / "plates.txt")

def test_the_reference_requirement_is_met_by_a_tsv_snapshot(tmp_path): ...
def test_both_snapshots_show_the_resolvers_message(tmp_path): ...
def test_merge_metadata_defaults_to_the_tsv_snapshot(tmp_path): ...
def test_the_qc_layout_reads_a_tsv(tmp_path):
    check = ExpectedVsDetectedCount(metadata=str(write_tsv_of(tmp_path, LAYOUT)), groupby=["Metadata_ImageName"])
    assert check._load_layout(...).equals(_load_layout_of_csv_twin(...))
```
- [ ] **Step 2: Run the test files listed; the new tests fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run every listed file; all pass.**
- [ ] **Step 5: Commit.**

### PA4 gate

`tests/unit/gui/`, `tests/integration/gui/`, `tests/gui/`, `tests/unit/analysis/`,
`tests/unit/sdk_/` (Slurm, detached worktree). The red set must now be empty of #1–6 and
#8–13; only #7 may remain if PA3 T1 has not merged.
