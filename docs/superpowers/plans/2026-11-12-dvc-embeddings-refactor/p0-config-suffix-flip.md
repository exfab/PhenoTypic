# P0 · Typed config suffix flip

**Spec:** Part A (A1–A6)

---

**Goal:** Writers produce `*.pht-pipe.json` (and `-op`, `-cc`, `-tune`), so DVC
reads and writes them with its JSON handler. Every reader still accepts the old
`*.json.pht-*` names and plain `.json`.

**Why the last suffix matters:** DVC picks its params parser from the last,
lower-cased suffix, and an unknown suffix falls back to YAML
(`dvc/utils/serialize/__init__.py`).

**GUI:** no file under `src/phenotypic/_gui/` changes. The three P0 rows of the
README's expected-red table go red here and stay red until #283. Two more
GUI-side effects are #283's, and harmless until then:
- the GUI's `matches_any_suffix(path, PIPELINE_CONFIG_SUFFIXES)` callers
  (`_gui/builder/_directory_browser.py:32`, `_gui/shell/_classifier.py:222`)
  list a new-style `x.pht-tune.json` as a pipeline candidate, because it ends
  in plain `.json`;
- the GUI analysis writer (`_gui/analysis/_recipe_state.py:909`) writes the new
  name beside an old one, so a folder briefly holds two current files. The
  resolver prefers the new name, so readers stay correct.

### Task 1: Suffix constants, the helpers and the resolvers

T1 lands the names **and** the resolvers in one commit. `PIPELINE_JSON`,
`BEST_PIPELINE_JSON` and `TUNING_SPEC_JSON` are built from the suffix constants
(`_io_constants.py:446`), so flipping the constants renames every canonical
artifact at once. Without the resolvers in the same commit, an existing
`deliverables/pipeline.json.pht-pipe` would be unreadable.

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py`:
  - :103-154, the suffix constants and helpers
  - :244-260, `migration_lock_path` identity
  - :446-467, the `_LEGACY_*` names
  - :1486-1535, the resolvers
  - :1797, `phenotypic_cache_pipeline_json_path`
- Modify: `src/phenotypic/sdk_/__init__.py` (export
  `LEGACY_TYPED_CONFIG_SUFFIXES`)
- Test: `tests/unit/sdk_/test_io_constants.py:231-287`

**Interfaces:**
- Produces:
  - `CONFIG_SUFFIX_PIPELINE = ".pht-pipe.json"`
  - `CONFIG_SUFFIX_OPERATION = ".pht-op.json"`
  - `CONFIG_SUFFIX_COLOR_CHECKER = ".pht-cc.json"`
  - `CONFIG_SUFFIX_TUNING = ".pht-tune.json"`
  - `LEGACY_TYPED_CONFIG_SUFFIXES: Final[Mapping[str, str]]`, mapping each new
    suffix to its old form (`".pht-pipe.json": ".json.pht-pipe"`, …)
  - `PIPELINE_CONFIG_SUFFIXES = {".pht-pipe.json", ".json.pht-pipe", ".json"}`
  - `TUNING_CONFIG_SUFFIXES = {".pht-tune.json", ".json.pht-tune", ".json"}`
  - `ensure_typed_json_suffix(path, suffix) -> Path` (new rules below)
  - `PIPELINE_JSON == "pipeline.pht-pipe.json"`
  - `_OLD_TYPED_PIPELINE_JSON == "pipeline.json.pht-pipe"`
  - `_LEGACY_PIPELINE_JSON == "pipeline.json"`
  - `pipeline_config_candidates(output_dir) -> tuple[Path, Path, Path]`: new,
    then old typed, then plain
  - `resolve_pipeline_config_path(output_dir) -> Path`: the first that exists,
    else the new path. Same name and signature as today (the GUI calls it).
  - the same trio for `tuning_spec`, `best_pipeline` and the `.phenotypic`
    process copy

- [ ] **Step 1: Write the failing tests**

```python
def test_config_suffix_values():
    assert CONFIG_SUFFIX_PIPELINE == ".pht-pipe.json"
    assert CONFIG_SUFFIX_OPERATION == ".pht-op.json"
    assert CONFIG_SUFFIX_COLOR_CHECKER == ".pht-cc.json"
    assert CONFIG_SUFFIX_TUNING == ".pht-tune.json"
    assert LEGACY_TYPED_CONFIG_SUFFIXES[CONFIG_SUFFIX_PIPELINE] == ".json.pht-pipe"

@pytest.mark.parametrize(("raw", "expected"), [
    ("x", "x.pht-pipe.json"),
    ("x.json", "x.pht-pipe.json"),
    ("x.JSON", "x.pht-pipe.JSON"),
    ("x.pht-pipe.json", "x.pht-pipe.json"),
    ("x.json.pht-pipe", "x.json.pht-pipe"),        # explicit old suffix honoured
    ("x.JSON.PHT-PIPE", "x.JSON.PHT-PIPE"),
])
def test_ensure_typed_json_suffix(raw, expected):
    assert str(ensure_typed_json_suffix(raw, CONFIG_SUFFIX_PIPELINE)) == expected

def test_pipeline_suffixes_accept_new_old_and_plain():
    for name in ("p.pht-pipe.json", "p.json.pht-pipe", "p.json"):
        assert matches_any_suffix(name, PIPELINE_CONFIG_SUFFIXES)

def test_resolver_prefers_new_then_old_then_plain(tmp_path):
    d = tmp_path / "deliverables"; d.mkdir()
    (d / "pipeline.json").write_text("{}")
    assert resolve_pipeline_config_path(tmp_path).name == "pipeline.json"
    (d / "pipeline.json.pht-pipe").write_text("{}")
    assert resolve_pipeline_config_path(tmp_path).name == "pipeline.json.pht-pipe"
    (d / "pipeline.pht-pipe.json").write_text("{}")
    assert resolve_pipeline_config_path(tmp_path).name == "pipeline.pht-pipe.json"

def test_all_three_names_share_one_migration_lock(tmp_path):
    names = ("pipeline.pht-pipe.json", "pipeline.json.pht-pipe", "pipeline.json")
    assert len({migration_lock_path(tmp_path / n) for n in names}) == 1
```

- [ ] **Step 2: Run them and confirm they fail.** Run
  `eval $PT tests/unit/sdk_/test_io_constants.py`. Expect FAIL on the suffix
  values and the resolver order.
- [ ] **Step 3: Implement.**
  - `ensure_typed_json_suffix` checks, in order:
    1. ends with the new or the legacy typed suffix → return the path unchanged;
    2. ends with `.json` (any case) → insert the typed stem before it, keeping
       the original `.json` spelling;
    3. otherwise → append the new suffix.
  - `has_config_suffix` and `matches_any_suffix` are unchanged.
  - Resolvers walk `pipeline_config_candidates`.
- [ ] **Step 4: Run again; it passes.** Run
  `eval $PT tests/unit/sdk_/test_io_constants.py`, plus the existing
  pipeline-config tests
  (`grep -rl "pht-pipe\|PIPELINE_JSON" tests/unit/cli tests/unit/sdk_`).
- [ ] **Step 5: Commit** the paths above, listed explicitly.

### Task 2: Writers, the seed and the literal sweep

**Files:**
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:610`,
  `_persist_pipeline_to_output_dir(output_dir, pipeline)` (the finalize writer)
- Modify: `src/phenotypic/phenotypicCLI.py:4634`,
  `_seed_pipeline_config(pipeline_path, output_dir)`
- Modify: `src/phenotypic/_cli/_cli_migrate.py:676-684` (the glob)
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py:118` and
  `_cli_interactive.py:143` (help text)
- Create (helpers): `tests/_tiny_run.py` with fixtures `tiny_pipeline`
  (`ImagePipeline(ops=[OtsuDetector()], meas={"size": MeasureSize()})`) and
  `tiny_pipeline_file` (writes it to `tmp_path / "tiny.pht-pipe.json"`);
  re-export both from `tests/conftest.py`
- Test: `tests/unit/cli/test_pipeline_config_publication.py` (new)

**Interfaces:**
- Consumes: Task 1's constants and resolvers.

- [ ] **Step 1: Write the failing tests**

```python
def test_finalize_publishes_new_name_and_removes_old(tmp_path, tiny_pipeline):
    old = tmp_path / "deliverables" / "pipeline.json.pht-pipe"
    old.parent.mkdir(); old.write_text(tiny_pipeline.to_json())
    _persist_pipeline_to_output_dir(tmp_path, tiny_pipeline)
    assert (tmp_path / "deliverables" / "pipeline.pht-pipe.json").is_file()
    assert not old.exists()

def test_finalize_keeps_plain_legacy_json(tmp_path, tiny_pipeline):
    plain = tmp_path / "deliverables" / "pipeline.json"
    plain.parent.mkdir(); plain.write_text(tiny_pipeline.to_json())
    _persist_pipeline_to_output_dir(tmp_path, tiny_pipeline)
    assert plain.is_file()

def test_seed_skips_when_any_name_exists(tmp_path, tiny_pipeline_file):
    old = tmp_path / "deliverables" / "pipeline.json.pht-pipe"
    old.parent.mkdir(); old.write_text("{}")
    assert _seed_pipeline_config(tiny_pipeline_file, tmp_path) is None
    assert not (tmp_path / "deliverables" / "pipeline.pht-pipe.json").exists()
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - The writer publishes the new name, then unlinks the old typed name if
    present. It never touches plain `pipeline.json`, so a V1 reader still finds
    it.
  - Migrate's glob takes both `*.pht-pipe.json` and `*.pht-pipe`.
  - The help strings use the new name.
- [ ] **Step 4: Run Task 2's tests plus the existing pipeline-config tests.**
  They pass.
- [ ] **Step 5: Sweep literals.**
  - Run `grep -rln 'json\.pht-' src tests docs/source | grep -v '_gui/\|tests/unit/gui/\|tests/gui/'`.
  - Update every literal that names a file PhenoTypic **writes** to the new
    suffix.
  - Leave literals in tests that deliberately exercise a legacy read, and mark
    each with `# legacy name, read path`.
  - Never edit `src/phenotypic/_gui/`, `tests/unit/gui/` or `tests/gui/`.
- [ ] **Step 6: Run the P0 gate.** Run the importers of `_io_constants` under
  `tests/unit/cli` and `tests/unit/sdk_`, then `tests/unit/gui` and `tests/gui`
  once. The only red GUI tests are the README's three P0 rows.
- [ ] **Step 7: Commit** every changed path, listed explicitly.

### Task 3: AutoConvertRaw-GC rename (repo `software/AutoConvertRaw-GC`)

`PIPELINE_NAME` is also an object-storage key: runs planned before the rollout
hold `<prefix>pipeline.json.pht-pipe`. Readers therefore fall back to the old
key, and only the writer (`plan.py:262`) uses the new one.

**Files:**
- Rename: `pipelines/exfab_rig1.json.pht-pipe` → `pipelines/exfab_rig1.pht-pipe.json`
- Modify: `src/acr_core/plan.py:27`. `PIPELINE_NAME = "pipeline.pht-pipe.json"`,
  and add `LEGACY_PIPELINE_NAME = "pipeline.json.pht-pipe"`.
- Modify: the readers of a run prefix, each trying `PIPELINE_NAME` then
  `LEGACY_PIPELINE_NAME`:
  - `src/acr_core/worker.py:250,273` (download, then load)
  - `src/acr_core/runs.py:38` (the listed run's `pipeline_sha`)
  - `src/acr_core/preflight.py:20`
  - `src/acr_hpcc/cli.py:37,113`
- Modify: `configs/exfab.yaml:32`, `configs/smoke.yaml:33`, `configs/smoke-hpcc.yaml:32`
- Modify: `src/acr_core/pipeline_info.py:1` (docstring)
- Modify: docs `CLAUDE.md:39`, `README.md:112`, and `docs/how-it-works.md:72,294,339,639`
- Test: `tests/test_pipeline_name.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
def test_renamed_pipeline_keeps_its_hash():
    new = REPO / "pipelines/exfab_rig1.pht-pipe.json"
    assert hashlib.sha256(new.read_bytes()).hexdigest() == "<sha256 of the old file, captured before the rename>"
    assert PIPELINE_NAME == "pipeline.pht-pipe.json"

def test_a_run_planned_before_the_rename_still_lists_and_executes(tmp_path):
    storage = LocalStorage(tmp_path)                   # acr_core.storage
    storage.write_bytes("runs/r1/" + LEGACY_PIPELINE_NAME, b"{}")
    assert read_run_pipeline_bytes(storage, "runs/r1/") == b"{}"
```

  `read_run_pipeline_bytes(storage, prefix) -> bytes` lives in
  `acr_core/plan.py` and is the one helper every reader above calls.
- [ ] **Step 2: Capture the old hash, then run the tests and confirm they
  fail.** Get the hash with `sha256sum pipelines/exfab_rig1.json.pht-pipe` and
  paste it into the test. Run `uv run pytest tests/test_pipeline_name.py`.
  Expect FAIL: the file and the helper are missing.
- [ ] **Step 3: Rename with `git mv`, add the helper, and update the
  references and docs.**
- [ ] **Step 4: Confirm the vendored wheel still loads it, then run the suite.**
  - `uv run python -c "from phenotypic import ImagePipeline; ImagePipeline.from_json('pipelines/exfab_rig1.pht-pipe.json')"`
    (vendored 0.20.2).
  - `uv run pytest` (~544 tests, no PhenoTypic import beyond the check above).
- [ ] **Step 5: Commit** in AutoConvertRaw-GC with explicit paths. Do not push
  until P5 (rollout).
