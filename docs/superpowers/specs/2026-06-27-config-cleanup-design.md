# Config Cleanup Design

Date: 2026-06-27

## Status

Draft for review. This document is an architecture cleanup spec only. It does
not change production code.

## Problem

PhenoTypic has several separate concepts currently described as "config":

- process-wide runtime switches in the legacy `phenotypic.settings_` package
- CLI command defaults and normalized execution options
- SLURM and staged-GPU resource defaults
- GUI routing, cache, tile, and display defaults
- GUI design tokens
- SDK artifact layout constants and path helpers
- operation defaults and tuning metadata

The problem is not that reusable values exist in multiple files. The problem is
that the ownership boundary for each kind of value is inconsistent, so defaults
can drift between entry points, tests, docs, workers, and generated artifacts.

Concrete examples found during inspection:

- `phenotypic.settings_.VALIDATE_OPS` is intended to be runtime mutable, but
  `phenotypic.sdk_.funcs_` imports it by value, so toggling
  `phenotypic.settings_.VALIDATE_OPS` after import does not affect validation.
- `phenotypic.settings_.MPL.FIGSIZE` appears public but is not used by plotting
  code. Its local guide also says the default is `(8, 6)`, while the code sets
  `(10, 8)`.
- CLI GPU defaults are declared in Click options and repeated in
  `ExecutionConfig`.
- `gpu_batch_size` and `gpu_workers_per_gpu` are parsed and stored, but current
  production usage appears absent. `gpu_shards` and `gpu_slurm_args` are used by
  staged SLURM.
- SLURM fallback values are duplicated between `_cli_constants.py` and SDK SLURM
  code.
- `--detect-mode` accepts a broad mode set at the top-level CLI, while the
  per-image worker accepts only `gray`, `red`, `green`, and `blue`.
- GUI `_config.py` mixes routing constants, Flask config keys, filesystem names,
  URL segments, tile sizing policies, polling intervals, branding strings, and
  SDK artifact re-exports.
- `sdk_/_io_constants.py` is a strong single source of truth for artifact layout,
  but it risks becoming a dumping ground if unrelated defaults are added there.

## Goals

1. Make each reusable value owned by the narrowest sensible layer.
2. Preserve reproducibility of serialized pipelines and operations.
3. Eliminate drift between Click defaults, worker defaults, dataclass defaults,
   tests, and docs.
4. Make the new public settings import unambiguous.
5. Make future defaults discoverable without forcing one global registry for
   unrelated concepts.

## Non-Goals

- Do not move operation parameters into global settings.
- Do not add a pandas-style global option registry for all package defaults.
- Do not change output layout semantics in this cleanup.
- Do not keep a compatibility shim for `phenotypic.settings_`.
- Do not use global settings to alter scientific or algorithmic behavior that
  should be serialized with a pipeline.

## Recommended Architecture

Use a layered architecture with explicit ownership buckets.

### 1. Public Runtime Settings

Create a small public runtime settings module:

```text
src/phenotypic/settings.py
```

This should own only true process-wide runtime switches, such as debug
validation. It should not own algorithm defaults, operation parameters, output
paths, GUI design tokens, or CLI command defaults.

Recommended initial surface:

```python
VALIDATE_OPS: bool = False

def set_validate_ops(enabled: bool) -> None: ...

@contextmanager
def validation(enabled: bool) -> Iterator[None]: ...
```

Consumers must read from the module at call time:

```python
import phenotypic.settings as settings

if settings.VALIDATE_OPS:
    ...
```

They must not use:

```python
from phenotypic.settings import VALIDATE_OPS
```

That import form captures the bool and breaks runtime mutation.

### 2. Legacy Settings Hard Cutoff

Remove the legacy settings package:

```text
src/phenotypic/settings_/__init__.py
```

No compatibility shim should remain. `import phenotypic.settings_` should raise
`ModuleNotFoundError`.

Rationale:

- `settings_` is currently exported by `phenotypic.__init__`.
- The current `settings_` package has no submodules, so a package buys little.
- `phenotypic.settings` is clearer and more idiomatic for a public runtime
  settings surface.
- The user has explicitly requested a hard cutoff rather than a deprecation
  window.

Important caveat: moving from `settings_` to `settings.py` does not by itself
fix import-order problems. The real fix is to make consumers read the settings
module live at call time.

### 3. CLI Runtime Defaults

Add a CLI-local defaults module:

```text
src/phenotypic/_cli/_cli_defaults.py
```

This module should own defaults that are specific to CLI invocation and worker
dispatch, not package-wide scientific behavior.

Initial constants:

```python
DEFAULT_CLI_MODE = "full"
DEFAULT_CLI_IMAGE_TYPE = "GridImage"
DEFAULT_CLI_DETECT_MODE = "gray"
DEFAULT_CLI_N_JOBS = -1
DEFAULT_CLI_EXT = "tiff"
DEFAULT_OVERLAY_ALPHA = 0.3

DEFAULT_GPU_BATCH_SIZE = 1
DEFAULT_GPU_WORKERS_PER_GPU = 1
DEFAULT_GPU_SHARDS = 1
DEFAULT_GPU_SLURM_GPUS_PER_NODE = 1
DEFAULT_STAGE2_SIGNAL_GRACE_SECONDS = 120
```

Click options, `ExecutionConfig`, staged SLURM script generation, tests, and docs
should import these values instead of restating literals.

Values currently in scope:

- `--gpu-batch-size` defaults to `1`.
- `--gpu-workers-per-gpu` defaults to `1`.
- `--gpu-shards` defaults to `1`.
- `--gpu-slurm` defaults to an empty dict after parsing.
- staged SLURM Stage 2 auto-adds `slurm_gpus_per_node=1` when absent.
- explicit `slurm_gpus_per_node=0` disables the GPU directive for Stage 2.
- staged SLURM Stage 2 signal grace defaults to `120` seconds.
- staged dependency mode is `afterany`.

`gpu_batch_size` and `gpu_workers_per_gpu` should either be wired into staged
execution or documented as reserved/spec-forward. Leaving parsed-but-unused
options without that clarification is misleading.

### 4. CLI Execution Config

Keep `ExecutionConfig` as the normalized CLI execution object, but clean its
boundary.

Recommended direction:

- Keep raw user input parsing near Click.
- Normalize into a single config object once.
- Prefer immutable or copy-on-write updates for derived values.
- Separate user-provided values from derived values where practical.

Short-term:

- Use `_cli_defaults.py` for default values.
- Avoid mutating `config.slurm_args` in strategy code when adding derived GPU
  resources. Create a local `resolved_slurm_args` dict instead.
- Add tests that compare Click defaults to `ExecutionConfig` defaults.

Longer-term:

- Consider a frozen dataclass or pydantic model for execution config.
- Consider a small `ResolvedExecutionConfig` for post-validation values like
  resolved output directory, resolved SLURM profile, and resolved GPU profile.

### 5. SDK Constants and Artifact Layout

Keep `phenotypic.sdk_._io_constants` as the public source of truth for artifact
layout:

- config suffixes
- filenames
- directory names
- templated filename renderers
- path helpers
- JSON contract keys
- environment variable names
- output bundle discovery helpers

Do not move those values into runtime settings. They are package contracts, not
user preferences.

If `_io_constants.py` keeps growing, split it internally while preserving
public re-exports from `phenotypic.sdk_`:

```text
src/phenotypic/sdk_/_io/
  _names.py
  _paths.py
  _keys.py
  _bundle.py
```

This can reduce file size without weakening the single-source-of-truth contract.

### 6. Operation Defaults

Keep algorithm defaults on pydantic operation fields.

Do not promote operation defaults into global config. Examples include detector
thresholds, grid finder defaults, morphology parameters, denoising parameters,
and tune search metadata.

Rationale:

- Operation defaults are part of the serialized pipeline contract.
- Global overrides would make a pipeline's behavior depend on ambient state.
- Tuning ranges belong near fields because they describe the parameter being
  tuned.

`TuneSpec` should remain search metadata. It should not be treated as runtime
validation unless explicitly backed by `Field` bounds or validators.

### 7. GUI Config and Design Tokens

Keep `phenotypic.gui._config` and `phenotypic.gui._design`, but preserve a clear
boundary:

- `_config.py`: GUI routing, server config keys, sandbox/cache names, URL
  segments, GUI interaction defaults, lightweight helpers.
- `_design.py`: colors, fonts, type scale, spacing, radii, shadows, motion, CSS
  injection, design-token helpers.

Near-term cleanup:

- Import `IMAGE_EXTS` directly from `_config.py` in shell classifier code rather
  than through `builder._directory_browser`.
- Replace deprecated `FONT_SIZE_LABEL` call sites and docs with
  `FONT_SIZE_BODY_SM` or `FONT_SIZE_CAPTION`.
- Add CSS variables or design tokens for repeated hover/status colors that
  currently bypass `_design.py`.
- Extract a generic bounded step helper only if another tile-like control is
  added. Current duplication between colony and timeline tile sizing is small.

Do not route GUI design tokens through `phenotypic.settings`.

## Settings Module Decision

`phenotypic.settings.py` is a better public shape than the current
`phenotypic.settings_` subpackage if the settings surface stays small.

Reasons:

- A single module is enough. `settings_` currently has no submodules.
- `phenotypic.settings` is a cleaner public import than `phenotypic.settings_`.
- The underscore suffix makes the module look semi-private even though it is
  exported.
- A root module makes future docs read naturally:

```python
import phenotypic.settings as settings
settings.VALIDATE_OPS = True
```

This should be a hard migration:

1. Add `phenotypic/settings.py`.
2. Delete `phenotypic/settings_/__init__.py`.
3. Update internal imports to `import phenotypic.settings as settings`.
4. Remove `settings_` from `phenotypic.__init__`.
5. Update docs and examples.
6. Add a test that `import phenotypic.settings_` fails.

Do not make this migration the only cleanup. It is naming cleanup plus a better
public API shape. The runtime correctness fix still depends on live module reads.

## Implementation Plan

### Phase 1: Fix Runtime Settings Semantics

1. Add `src/phenotypic/settings.py`.
2. Move `VALIDATE_OPS` there.
3. Add `set_validate_ops()` and `validation()` context manager.
4. Delete `src/phenotypic/settings_/__init__.py`.
5. Update `sdk_/funcs_.py` to import the settings module and read
   `settings.VALIDATE_OPS` at call time.
6. Remove `MPL.FIGSIZE` unless a real plotting consumer is added.
7. Remove the old `settings_/CLAUDE.md` guide.

Tests:

- Toggling `phenotypic.settings.VALIDATE_OPS` after importing `sdk_.funcs_`
  changes validation behavior.
- `import phenotypic.settings_` raises `ModuleNotFoundError`.
- The validation context manager restores the previous value after exit.

### Phase 2: Centralize CLI and GPU Defaults

1. Add `_cli_defaults.py`.
2. Move CLI defaults there.
3. Update Click option defaults.
4. Update `ExecutionConfig` defaults.
5. Update staged SLURM defaults, including Stage 2 GPU request and signal grace.
6. Update tests to assert CLI defaults and `ExecutionConfig` defaults stay
   aligned.
7. Decide whether `gpu_batch_size` and `gpu_workers_per_gpu` are implemented now
   or explicitly documented as reserved.

Tests:

- Click help/defaults match `_cli_defaults.py`.
- `ExecutionConfig()` defaults match `_cli_defaults.py` where applicable.
- `resolve_stage_slurm_args({}, cpu_args)` adds one GPU.
- `resolve_stage_slurm_args({"slurm_gpus_per_node": 0}, cpu_args)` omits the GPU
  directive.
- Non-default `--gpu-shards` affects staged SLURM Stage 2 array size.

### Phase 3: Fix CLI Drift Bugs

1. Make per-image worker detect-mode choices use the same contract as the
   top-level CLI.
2. Add a test for a non-RGB mode such as `LabL` through SLURM script generation
   or worker argument validation.
3. Consolidate SLURM fallback constants currently duplicated as `1000`, `5`, and
   `30`.
4. Update generated README output layout to match typed pipeline suffixes and
   `.phenotypic/` machine state.
5. Update SLURM docs to say SLURM mode requires `--slurm`, unless behavior is
   intentionally changed.

Tests:

- Top-level CLI and worker agree on detect-mode accepted values.
- Generated README names `pipeline.json.pht-pipe` and `.phenotypic/` correctly.
- SLURM fallback values are imported from one owner.

### Phase 4: GUI Token and Config Cleanup

1. Fix shell classifier to import `IMAGE_EXTS` directly from `_config.py`.
2. Replace `FONT_SIZE_LABEL` call sites and docs.
3. Add missing design tokens for repeated CSS literals that should be shared.
4. Keep SDK artifact re-exports in `_config.py` for compatibility, but document
   that new artifact names must be added to SDK first.

Tests:

- `_config.py` stays Dash/Flask-free at import time.
- `IMAGE_EXTS` identity stays shared between builder, browse, and classifier.
- Design token CSS contains any new shared variables.
- Existing GUI design tests pass after `FONT_SIZE_LABEL` migration.

### Phase 5: Optional Internal SDK Split

Only do this if `_io_constants.py` continues to grow.

1. Split implementation into internal modules.
2. Preserve imports from `phenotypic.sdk_`.
3. Preserve tests as public contract tests.

Tests:

- Existing `tests/unit/sdk_/test_io_constants.py` remains the primary guard.
- No public import path breaks.

## Migration and Compatibility

- `phenotypic.settings` becomes the preferred import.
- `phenotypic.settings_` is removed.
- No operation JSON schema should change.
- No output layout should change.
- Existing pipeline JSON files should load unchanged.
- CLI behavior should remain unchanged except for drift bug fixes and clearer
  validation.

## Risks

- Import-order changes can accidentally pull in heavy modules earlier. Keep
  `settings.py` stdlib-only and cheap.
- Removing `settings_` is a breaking change. This is intentional for the hard
  cutoff and should be reflected in release notes.
- Centralizing CLI defaults may reveal tests that depend on duplicated literals.
  Update tests to assert defaults from the new owner instead of restating magic
  values.
- If `gpu_batch_size` and `gpu_workers_per_gpu` remain unused, centralization
  alone may make them look more complete than they are. Either wire them or mark
  them reserved.

## Acceptance Criteria

- There is a clear documented owner for each category of reusable value.
- Runtime settings are live, not captured by by-value imports.
- CLI defaults are defined once and consumed by Click and `ExecutionConfig`.
- Staged GPU defaults are defined once and covered by tests.
- Operation defaults remain on operation models.
- SDK artifact layout remains the cross-module path contract.
- GUI design tokens remain separate from GUI runtime config.
- `phenotypic.settings` is the only public settings import.
