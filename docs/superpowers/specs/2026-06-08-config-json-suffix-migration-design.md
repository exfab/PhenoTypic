# Design: PhenoTypic config JSON suffix migration

- **Date:** 2026-06-08
- **Branch:** current Codex worktree
- **Status:** Draft design spec
- **Author:** Alexander Nguyen (with Codex)

## 1. Summary

PhenoTypic currently writes several reusable configuration documents as plain
JSON filenames:

- `pipeline.json` for `ImagePipeline` and `PrefabPipeline` configurations.
- `best_pipeline.json` and `pareto/best_<objective>.json` for tuned
  `ImagePipeline` winners.
- `tuning_spec.json` for both full `TuningSpec` documents and
  `InferredSearchSpace` auto-space proposals.
- User-chosen JSON paths for standalone `BaseOperation` exports.
- User-chosen JSON paths for `ColorCheckerProfile` exports.

This migration adds project-specific terminal suffixes to those JSON config
files so they are visually identifiable and easy to query:

| Runtime object | Required saved-file suffix |
|---|---|
| `ImagePipeline` and `PrefabPipeline` | `.json.pht-pipe` |
| `BaseOperation` and subclasses | `.json.pht-op` |
| `ColorCheckerProfile` | `.json.pht-cc` |
| `TuningSpec` and tuning search-space proposals | `.json.pht-tune` |

The suffixes intentionally end with the PhenoTypic type marker rather than
`.json`. That makes shell queries such as `*.pht-pipe` and file-browser scans
unambiguous, but it also means callers must not rely on `Path.suffix == ".json"`
for these config files. All new matching code must use centralized suffix
helpers rather than ad hoc suffix checks.

## 2. Goals

- Add guard logic so save paths for each user-facing config type automatically
  receive the correct suffix when the caller omits it.
- Centralize every suffix string in `phenotypic.tools_._io_constants` and
  re-export it from `phenotypic.tools_`.
- Do not spell the same suffix literal in more than one source location.
- Keep old plain `.json` files readable.
- Update canonical CLI and GUI output names to the new suffixes.
- Update GUI file pickers so users can find both new typed config files and
  legacy plain `.json` pipeline files.
- Keep state and report JSON artifacts out of this migration unless they are
  reusable user-authored configuration documents.

## 3. Non-goals

- No change to the JSON payload schemas.
- No migration command that renames existing files in place.
- No support for YAML, TOML, or other config encodings.
- No suffix changes for measurement tables, dashboards, reports, tune journals,
  Optuna stores, or GUI session state.
- No removal of legacy plain `.json` read support.

## 4. User-facing scope

### 4.1 In scope

#### `ImagePipeline` and `PrefabPipeline`

`ImagePipeline.to_json(filepath=...)` is the main writer for user-saved
pipelines. `PrefabPipeline.from_json(...)` delegates through
`ImagePipeline.from_json(...)`, so prefab pipeline compatibility follows the
pipeline path. The canonical run output path is currently provided by
`pipeline_json_path(output_dir)`.

All of the following should move to the pipeline suffix:

- `deliverables/pipeline.json` to `deliverables/pipeline.json.pht-pipe`
- `.phenotypic/pipeline.json` process-only cache copy to
  `.phenotypic/pipeline.json.pht-pipe`
- `deliverables/best_pipeline.json` to
  `deliverables/best_pipeline.json.pht-pipe`
- `deliverables/pareto/best_<objective>.json` to
  `deliverables/pareto/best_<objective>.json.pht-pipe`
- GUI builder default save filename from `pipeline.json` to
  `pipeline.json.pht-pipe`

Legacy plain `.json` pipeline files must remain loadable through explicit file
paths and visible in GUI load pickers.

#### `BaseOperation`

`BaseOperation.to_json(filepath=...)` and `BaseOperation.from_json(...)` are the
standalone operation serializer pair. These are user-facing when users export or
share a single operation configuration outside a full pipeline.

Any path passed to `BaseOperation.to_json(filepath=...)` should be normalized to
end in `.json.pht-op` unless it already does.

#### `ColorCheckerProfile`

`ColorCheckerProfile.to_json(filepath=...)` and
`ColorCheckerProfile.from_json(...)` are the profile serializer pair. These are
user-facing calibration configs and should use `.json.pht-cc`.

Any path passed to `ColorCheckerProfile.to_json(filepath=...)` should be
normalized to end in `.json.pht-cc` unless it already does.

#### `TuningSpec` and `InferredSearchSpace`

The tuning workflow currently writes two distinct payload types to the same
logical tuning document path:

- `run_tuning(...)` writes a full `TuningSpec`.
- `run_auto_space(...)` writes an `InferredSearchSpace` proposal to the same
  `tuning_spec_path(output_dir)` so the user can inspect and finalize it.

Both are tuning workflow configuration documents, so both should use the same
`.json.pht-tune` suffix. Loaders must keep the current tolerant behavior:
attempt to validate as `TuningSpec`, and treat a validation miss as a proposal
or unavailable full spec where that is already the current contract.

### 4.2 Out of scope

These files are user-visible, but they are not reusable user-authored config
documents and should keep their current names:

- `param_importance.json`
- `generalization.json`
- `.pht-tune-cache/run.json`
- `.pht-tune-cache/splits/split.json`
- `qc/qc_config.json`
- `qc/review_state.json`
- `.viewer_cache/qc_recipe.json` legacy sidecar
- `progress/*.json`, `progress/*.jsonl`, and other machine-state sidecars
- `builder-state.json` downloads
- run-console UI state payloads

This distinction is important: the typed suffixes should identify portable
recipes and profiles, not every JSON blob PhenoTypic writes.

## 5. Naming source of truth

All suffix names and derived config filenames live in
`src/phenotypic/tools_/_io_constants.py`. The package-level
`src/phenotypic/tools_/__init__.py` must re-export them.

### 5.1 New constants

Add the suffix constants near the existing filename constants:

```python
# Config JSON typed suffixes. These are the only source locations for the
# literal suffix strings.
LEGACY_JSON_SUFFIX: Final[str] = ".json"
CONFIG_SUFFIX_PIPELINE: Final[str] = ".json.pht-pipe"
CONFIG_SUFFIX_OPERATION: Final[str] = ".json.pht-op"
CONFIG_SUFFIX_COLOR_CHECKER: Final[str] = ".json.pht-cc"
CONFIG_SUFFIX_TUNING: Final[str] = ".json.pht-tune"

CONFIG_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        CONFIG_SUFFIX_PIPELINE,
        CONFIG_SUFFIX_OPERATION,
        CONFIG_SUFFIX_COLOR_CHECKER,
        CONFIG_SUFFIX_TUNING,
    }
)
PIPELINE_CONFIG_SUFFIXES: Final[frozenset[str]] = frozenset(
    {CONFIG_SUFFIX_PIPELINE, LEGACY_JSON_SUFFIX}
)
TUNING_CONFIG_SUFFIXES: Final[frozenset[str]] = frozenset(
    {CONFIG_SUFFIX_TUNING, LEGACY_JSON_SUFFIX}
)
```

If implementation scope allows, convert GUI directory filters that currently
spell `".json"` to import `LEGACY_JSON_SUFFIX`.

### 5.2 Derived filename constants

Do not spell `.json.pht-pipe`, `.json.pht-tune`, `.json.pht-op`, or
`.json.pht-cc` outside the suffix constants above. Existing filename constants
should become derived values:

```python
PIPELINE_JSON: Final[str] = f"pipeline{CONFIG_SUFFIX_PIPELINE}"
TUNING_SPEC_JSON: Final[str] = f"tuning_spec{CONFIG_SUFFIX_TUNING}"
BEST_PIPELINE_JSON: Final[str] = f"best_pipeline{CONFIG_SUFFIX_PIPELINE}"
_PARETO_BEST_PIPELINE_FILENAME_TEMPLATE: Final[str] = (
    f"best_{{objective}}{CONFIG_SUFFIX_PIPELINE}"
)
```

The constant names can remain `PIPELINE_JSON`, `TUNING_SPEC_JSON`, and
`BEST_PIPELINE_JSON` for compatibility with existing imports. Their values
change. This is a smaller API break than renaming the symbols.

### 5.3 Path normalization helpers

Add path helpers to the same module. The guard should use user-friendly legacy
normalization: `x.json` becomes `x.json.pht-pipe`, not
`x.json.json.pht-pipe`.

```python
def has_config_suffix(path: str | Path, suffixes: frozenset[str]) -> bool:
    """Return True when path ends with any full config suffix."""
    text = str(path)
    return any(text.endswith(suffix) for suffix in suffixes)


def ensure_typed_json_suffix(path: str | Path, suffix: str) -> Path:
    """Return path with the typed JSON suffix, normalizing legacy .json."""
    candidate = Path(path)
    text = str(candidate)
    if text.endswith(suffix):
        return candidate
    if text.endswith(LEGACY_JSON_SUFFIX):
        return Path(text + suffix.removeprefix(LEGACY_JSON_SUFFIX))
    return Path(text + suffix)
```

`Path.with_suffix(...)` is not appropriate here because the desired suffix is a
compound terminal suffix appended to the caller's chosen filename. For example,
`Path("my_pipeline").with_suffix(".json.pht-pipe")` may work for a bare stem,
but `Path("my_pipeline.json").with_suffix(".json.pht-pipe")` would replace the
existing `.json` instead of preserving the user's requested base. The migration
contract is append the missing PhenoTypic marker while preserving a user-supplied
JSON base.

### 5.4 Suffix matching helper for file pickers

Add a filename matching helper so GUI trees stop relying on `Path.suffix`:

```python
def matches_any_suffix(path: str | Path, suffixes: frozenset[str]) -> bool:
    """Return True when path text ends with any suffix in suffixes."""
    text = str(path).lower()
    return any(text.endswith(suffix.lower()) for suffix in suffixes)
```

This helper should be used by directory browsers that surface typed config
files. `Path.suffix` returns only `.pht-pipe` for `pipeline.json.pht-pipe`, so
`Path.suffix` cannot answer whether the file is JSON-backed or pipeline-backed.

## 6. Serializer and writer behavior

### 6.1 `ImagePipeline.to_json`

When `filepath` is provided:

1. Normalize with `ensure_typed_json_suffix(filepath, CONFIG_SUFFIX_PIPELINE)`.
2. Write to the normalized path.
3. Return `None`, preserving the current overload behavior.

When `filepath` is `None`, return the JSON string unchanged.

`ImagePipeline.from_json(...)` should continue to accept any explicit file path
that exists, including old `.json` files and new `.json.pht-pipe` files. The
loader parses by content, not suffix.

### 6.2 `BaseOperation.to_json`

When `filepath` is provided:

1. Normalize with `ensure_typed_json_suffix(filepath, CONFIG_SUFFIX_OPERATION)`.
2. Write to the normalized path.
3. Return `None`.

`BaseOperation.from_json(...)` already delegates source coercion through
`read_json_source(...)`, so explicit path compatibility should remain.

### 6.3 `ColorCheckerProfile.to_json`

When `filepath` is provided:

1. Normalize with
   `ensure_typed_json_suffix(filepath, CONFIG_SUFFIX_COLOR_CHECKER)`.
2. Write to the normalized path.
3. Return `None`.

`ColorCheckerProfile.from_json(...)` already delegates source coercion through
`read_json_source(...)`, so explicit path compatibility should remain.

### 6.4 Tuning writes

Tuning has no `TuningSpec.to_json(filepath=...)` wrapper today. Do not introduce
bare ad hoc path logic at every writer. Route all tuning config writes through
the existing `tuning_spec_path(output_dir)` helper and the new
`TUNING_SPEC_JSON` value.

Affected writers include:

- `run_tuning(...)`, which echoes the resolved full spec.
- `run_auto_space(...)`, which writes an `InferredSearchSpace` proposal.
- GUI tune export, which writes the edited full spec.

Loaders should keep using `tuning_spec_path(output_dir)` when discovering a
run's canonical tuning document. CLI positional `SPEC` arguments should accept
both legacy `.json` and new `.json.pht-tune` paths because they are explicit
paths and validation is content-based.

## 7. Canonical output path changes

Change these constants and path helpers through `_io_constants.py` only:

| Helper | Old result | New result |
|---|---|---|
| `pipeline_json_path(output)` | `<output>/deliverables/pipeline.json` | `<output>/deliverables/pipeline.json.pht-pipe` |
| `phenotypic_cache_pipeline_json_path(output)` | `<output>/.phenotypic/pipeline.json` | `<output>/.phenotypic/pipeline.json.pht-pipe` |
| `best_pipeline_path(output)` | `<output>/deliverables/best_pipeline.json` | `<output>/deliverables/best_pipeline.json.pht-pipe` |
| `pareto_best_pipeline_path(output, objective)` | `<output>/deliverables/pareto/best_<objective>.json` | `<output>/deliverables/pareto/best_<objective>.json.pht-pipe` |
| `tuning_spec_path(output)` | `<output>/deliverables/tuning_spec.json` | `<output>/deliverables/tuning_spec.json.pht-tune` |

No caller should hand-join these filenames. If a caller needs one of these
files, it must use the path helper.

## 8. Backward compatibility and discovery

### 8.1 Explicit reads

Explicit reads remain path-based:

- `ImagePipeline.from_json(path)` loads whatever path the user passed.
- `BaseOperation.from_json(path)` loads whatever path the user passed.
- `ColorCheckerProfile.from_json(path)` loads whatever path the user passed.
- `TuningSpec.model_validate_json(Path(args.spec).read_text())` loads whatever
  path the user passed.

Suffix checks should not reject legacy files in these explicit paths.

### 8.2 Canonical fallback reads

Where code discovers canonical files under an output directory, add fallback
helpers rather than duplicating logic at call sites:

```python
def resolve_pipeline_config_path(output_dir: Path) -> Path:
    """Return new pipeline path when present, else legacy pipeline.json."""


def resolve_tuning_spec_path(output_dir: Path) -> Path:
    """Return new tuning path when present, else legacy tuning_spec.json."""


def resolve_best_pipeline_path(output_dir: Path) -> Path:
    """Return new best-pipeline path when present, else legacy best_pipeline.json."""
```

Use these only for read/discovery paths that need to tolerate old runs. Writers
must use the canonical new helpers.

### 8.3 GUI file pickers

The builder and run-console directory trees currently filter pipeline files by
`PIPELINE_EXTS = frozenset({".json"})`. That will not match
`pipeline.json.pht-pipe` when implemented with `Path.suffix`.

Required design:

- `PIPELINE_EXTS` should become `PIPELINE_CONFIG_SUFFIXES`.
- Directory filtering must use `matches_any_suffix(path, suffixes)`.
- Pipeline pickers should include both `.json.pht-pipe` and legacy `.json`.
- Tuning pickers, if any are added or already filtered, should include both
  `.json.pht-tune` and legacy `.json`.
- The default GUI save filename should import `PIPELINE_JSON`, whose value is
  now `pipeline.json.pht-pipe`.

### 8.4 Shell and results-viewer run discovery

Run discovery should prefer new typed config paths but tolerate old paths:

- Output-root summary should call `resolve_pipeline_config_path(...)`.
- Tune run roots should call `resolve_tuning_spec_path(...)`.
- Monitor/curate paths should call `resolve_best_pipeline_path(...)` only when
  reading existing outputs. Writers keep using `best_pipeline_path(...)`.

## 9. User-facing copy and docs

Update text that says "Pipeline JSON" or names `pipeline.json` as the primary
filename when it appears in user-facing CLI help, GUI labels, tutorials, or
examples.

Preferred wording:

- "Pipeline config file created with `pipeline.to_json()`."
- "Default filename: `pipeline.json.pht-pipe`."
- "Legacy `.json` pipeline files are still accepted."

Keep references to JSON payloads as JSON. The file is still JSON content even
though the terminal suffix is `.pht-pipe`.

## 10. Test plan

### 10.1 Unit tests for constants and helpers

Add tests under `tests/unit/tools_/test_io_constants.py`:

- Assert suffix constants equal the agreed values.
- Assert filename constants derive the new typed suffixes.
- Assert path helpers return the new canonical paths.
- Assert `ensure_typed_json_suffix(Path("x"), CONFIG_SUFFIX_PIPELINE)` returns
  `Path("x.json.pht-pipe")`.
- Assert `ensure_typed_json_suffix(Path("x.json"), CONFIG_SUFFIX_PIPELINE)`
  returns `Path("x.json.pht-pipe")`.
- Assert `ensure_typed_json_suffix(Path("x.json.pht-pipe"), CONFIG_SUFFIX_PIPELINE)`
  returns `Path("x.json.pht-pipe")`.

### 10.2 Serializer tests

Update/add tests:

- `ImagePipeline.to_json(tmp_path / "pipe")` writes
  `pipe.json.pht-pipe`.
- `ImagePipeline.to_json(tmp_path / "pipe.json")` writes
  `pipe.json.pht-pipe`.
- `ImagePipeline.to_json(tmp_path / "pipe.json.pht-pipe")` does not double
  append.
- Legacy `ImagePipeline.from_json(tmp_path / "pipe.json")` still works.
- Equivalent tests for `BaseOperation` and `ColorCheckerProfile`.

### 10.3 Tuning tests

Update tests that assert `tuning_spec_path(...)`, `best_pipeline_path(...)`, or
`pareto_best_pipeline_path(...)`.

Add tests that:

- `run_tuning(...)` writes `tuning_spec.json.pht-tune`.
- `run_auto_space(...)` writes `tuning_spec.json.pht-tune` even though the
  payload is an `InferredSearchSpace`.
- GUI tune space loading falls back from full `TuningSpec` validation when the
  typed tuning file contains an `InferredSearchSpace` proposal.
- Legacy `tuning_spec.json` still loads when discovered in an old output root.

### 10.4 GUI directory filter tests

Update `tests/unit/gui/test_directory_tree_filters.py`:

- New typed pipeline files are shown.
- Legacy `.json` pipeline files are shown.
- Non-config files such as `notes.txt` are hidden.
- A file named `report.json.pht-tune` is hidden from pipeline pickers.

### 10.5 CLI tests

Update CLI tests that assert copied pipeline paths, output path examples, or
deliverable names.

Add tests that:

- CLI `--pipeline old_pipeline.json` still runs.
- CLI output writes canonical `pipeline.json.pht-pipe`.
- Process-only cache copy writes `.phenotypic/pipeline.json.pht-pipe`.
- CLI help examples mention the new typed pipeline suffix or use neutral
  placeholder wording.

## 11. Implementation order

1. Add constants and helpers to `tools_/_io_constants.py`; re-export from
   `tools_/__init__.py`.
2. Update unit tests for the constants and path helpers.
3. Update serializer guards for `ImagePipeline`, `BaseOperation`, and
   `ColorCheckerProfile`.
4. Update canonical filename constants and path-helper expectations.
5. Add read-fallback helpers for legacy canonical files.
6. Update CLI/tune writers to use the new canonical paths through helpers.
7. Update GUI directory filtering to use full-suffix matching.
8. Update GUI/run discovery readers to use fallback helpers where needed.
9. Update user-facing copy and docs.
10. Run focused tests, then broader `ruff`, `mypy`, and relevant unit suites.

## 12. Risks and mitigations

### Risk: `Path.suffix` no longer sees `.json`

Mitigation: centralize `matches_any_suffix(...)` and update GUI filters to use
full-name suffix matching.

### Risk: old output roots disappear from GUI discovery

Mitigation: add explicit `resolve_*` fallback helpers and use them in
discovery/read paths.

### Risk: suffix strings drift across modules

Mitigation: suffix literals live only in `_io_constants.py`. Tests should scan
for forbidden repeated suffix literals outside that file:

```bash
rg '\\.json\\.pht-(pipe|op|cc|tune)' src tests docs
```

Expected after migration: only `_io_constants.py`, this design spec, and tests
that intentionally assert the public contract contain those literals.

### Risk: `TuningSpec` and `InferredSearchSpace` share a filename but not schema

Mitigation: keep the existing tolerant loader behavior. The suffix identifies
the tuning workflow document, not a single pydantic class.

### Risk: user confusion because files are JSON but do not end in `.json`

Mitigation: UI copy should say "JSON-backed PhenoTypic config" or "Pipeline
config", and docs should state that the content is still JSON. The suffix choice
prioritizes queryability by PhenoTypic config type.

## 13. Settled decisions and optional follow-ups

1. **Legacy `.json` normalization on save:** `pipeline.to_json("x.json")`
   writes `x.json.pht-pipe`.

   Rationale: this better matches user intent and avoids awkward doubled JSON
   stems. The `ensure_typed_json_suffix(...)` helper strips a terminal
   `LEGACY_JSON_SUFFIX` before appending the PhenoTypic marker.

2. **Constant names:** Keep legacy symbol names like `PIPELINE_JSON`, or rename
   to `PIPELINE_CONFIG_FILENAME`?

   Decision: keep existing names for now and add aliases only if the code
   becomes unclear. Existing imports are widespread, and the payload is still
   JSON.

3. **Standalone `TuningSpec` convenience API:** Should `TuningSpec` gain
   `to_json(filepath=None)` / `from_json(...)` wrappers?

   Decision: not required for this migration. The current tuning workflow
   uses centralized path helpers and Pydantic APIs. Adding wrappers is optional
   follow-up API polish.

## 14. Acceptance criteria

- All new reusable config saves append the configured typed suffix when needed.
- No code outside `phenotypic.tools_._io_constants` defines the typed suffix
  literals.
- Canonical output roots use:
  - `pipeline.json.pht-pipe`
  - `best_pipeline.json.pht-pipe`
  - `pareto/best_<objective>.json.pht-pipe`
  - `tuning_spec.json.pht-tune`
- `run_auto_space(...)` writes `tuning_spec.json.pht-tune` for
  `InferredSearchSpace` proposals.
- Legacy `.json` config files still load by explicit path.
- Legacy output roots with `pipeline.json`, `best_pipeline.json`, or
  `tuning_spec.json` are still discoverable where they were previously
  discoverable.
- GUI pipeline file pickers show new `.json.pht-pipe` files and old `.json`
  pipeline files.
- State/report JSON artifacts keep their existing names.
