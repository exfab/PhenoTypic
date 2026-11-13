# P0 · Typed config suffix flip

**Spec:** Part A (A1–A6)

---

**Goal:** Writers produce `*.pht-pipe.json` (and `-op`, `-cc`, `-tune`), so DVC
reads and writes them with its JSON handler. Every reader still accepts the old
`*.json.pht-*` names and plain `.json`.

**Why the last suffix matters:** DVC picks its params parser from the last,
lower-cased suffix, and an unknown suffix falls back to YAML
(`dvc/utils/serialize/__init__.py`).

### Task 1: Suffix constants and the two helpers

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py:103-154`
- Modify: `src/phenotypic/sdk_/__init__.py` (export `LEGACY_TYPED_CONFIG_SUFFIXES`)
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

- [ ] **Step 1: Rewrite the existing tests to the new values**

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
    assert not matches_any_suffix("t.json.pht-tune", PIPELINE_CONFIG_SUFFIXES)
```

- [ ] **Step 2: Run them and confirm they fail.** Run
  `eval $PT tests/unit/sdk_/test_io_constants.py`. Expect FAIL on the suffix
  values.
- [ ] **Step 3: Implement.**
  - `ensure_typed_json_suffix` checks, in order:
    1. ends with the new or the legacy typed suffix → return the path unchanged;
    2. ends with `.json` (any case) → insert the typed stem before it, keeping
       the original `.json` spelling;
    3. otherwise → append the new suffix.
  - `has_config_suffix` is unchanged.
- [ ] **Step 4: Run again; it passes.** Run
  `eval $PT tests/unit/sdk_/test_io_constants.py`.
- [ ] **Step 5: Commit** `src/phenotypic/sdk_/_io_constants.py`,
  `src/phenotypic/sdk_/__init__.py` and `tests/unit/sdk_/test_io_constants.py`.

### Task 2: Run-artifact names, resolvers and the one-current-file rule

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py`:
  - :244-260 `migration_lock_path` identity
  - :446-467 the `_LEGACY_*` names
  - :1486-1535 the resolvers
  - :1797 `phenotypic_cache_pipeline_json_path`
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:614` (the finalize writer)
- Modify: `src/phenotypic/phenotypicCLI.py`, `_seed_pipeline_config` (:4697)
- Modify: `src/phenotypic/_cli/_cli_migrate.py:676-684` (the glob)
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py:118` and `_cli_interactive.py:143` (help text)
- Test: `tests/unit/sdk_/test_io_constants.py`, `tests/unit/cli/test_pipeline_config_publication.py` (new)

**Interfaces:**
- Consumes: Task 1's constants.
- Produces:
  - `PIPELINE_JSON == "pipeline.pht-pipe.json"`
  - `_OLD_TYPED_PIPELINE_JSON == "pipeline.json.pht-pipe"`
  - `_LEGACY_PIPELINE_JSON == "pipeline.json"`
  - `pipeline_config_candidates(output_dir) -> tuple[Path, Path, Path]`: new,
    then old typed, then plain
  - `resolve_pipeline_config_path(output_dir) -> Path`: the first that exists,
    else the new path
  - the same trio for `tuning_spec`, `best_pipeline` and the `.phenotypic`
    process copy

- [ ] **Step 1: Write the failing tests**

```python
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

def test_finalize_publishes_new_name_and_removes_old(tmp_path, tiny_pipeline):
    old = tmp_path / "deliverables" / "pipeline.json.pht-pipe"
    old.parent.mkdir(); old.write_text(tiny_pipeline.to_json())
    persist_pipeline_config(tmp_path, tiny_pipeline)     # the :614 writer
    assert (tmp_path / "deliverables" / "pipeline.pht-pipe.json").is_file()
    assert not old.exists()

def test_seed_skips_when_any_name_exists(tmp_path, tiny_pipeline_file):
    old = tmp_path / "deliverables" / "pipeline.json.pht-pipe"
    old.parent.mkdir(); old.write_text("{}")
    _seed_pipeline_config(tmp_path, tiny_pipeline_file)
    assert not (tmp_path / "deliverables" / "pipeline.pht-pipe.json").exists()
```

  Use the writer's real name at `_cli_output_manager.py:614` in place of
  `persist_pipeline_config` if it differs.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - Resolvers walk `pipeline_config_candidates`.
  - The writer publishes the new name, then unlinks the old typed name if
    present. It never touches plain `pipeline.json`, so a V1 reader still finds
    it.
  - Migrate's glob takes both `*.pht-pipe.json` and `*.pht-pipe`.
  - The help strings use the new name.
- [ ] **Step 4: Run Task 2's tests plus the existing pipeline-config tests**
  (`grep -rl "pht-pipe\|PIPELINE_JSON" tests/unit/cli tests/unit/sdk_`). They
  pass.
- [ ] **Step 5: Sweep literals.**
  - Run `grep -rln 'json\.pht-' src tests docs/source`.
  - Update every literal that names a file PhenoTypic **writes** to the new
    suffix.
  - Leave literals in tests that deliberately exercise a legacy read, and mark
    each with `# legacy name, read path`.
  - Re-run the importers of `_io_constants` under `tests/unit/gui` and
    `tests/unit/cli`.
- [ ] **Step 6: Commit** every changed path, listed explicitly.

### Task 3: AutoConvertRaw-GC rename (repo `software/AutoConvertRaw-GC`)

**Files:**
- Rename: `pipelines/exfab_rig1.json.pht-pipe` → `pipelines/exfab_rig1.pht-pipe.json`
- Modify: `src/acr_core/plan.py:27` (`PIPELINE_NAME`)
- Modify: `configs/exfab.yaml:32`, `configs/smoke.yaml:33`, `configs/smoke-hpcc.yaml:32`
- Modify: `src/acr_core/pipeline_info.py:1` (docstring)
- Test: `tests/test_pipeline_name.py` (new)

- [ ] **Step 1: Write the failing test**

```python
def test_renamed_pipeline_keeps_its_hash():
    new = REPO / "pipelines/exfab_rig1.pht-pipe.json"
    assert sha256_file(new) == "<sha256 of the old file, captured before the rename>"
    assert PIPELINE_NAME == "pipeline.pht-pipe.json"
```

- [ ] **Step 2: Capture the old hash, then run the test and confirm it fails.**
  Get the hash with `sha256sum pipelines/exfab_rig1.json.pht-pipe` and paste it
  into the test. Run `uv run pytest tests/test_pipeline_name.py`. Expect FAIL:
  the file is missing.
- [ ] **Step 3: Rename with `git mv` and update the four references.**
- [ ] **Step 4: Confirm the vendored wheel still loads it, then run the suite.**
  - `uv run python -c "from phenotypic import ImagePipeline; ImagePipeline.from_json('pipelines/exfab_rig1.pht-pipe.json')"`
    (vendored 0.20.2).
  - `uv run pytest` (~544 tests, no PhenoTypic import beyond the check above).
- [ ] **Step 5: Commit** in AutoConvertRaw-GC with explicit paths. Do not push
  until P5 (rollout).
