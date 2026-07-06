# Medium Dedupe Phase 1 Implementation Plan

This plan implements Phase 1 from
`docs/superpowers/specs/2026-07-03-medium-priority-dedupe-cleanup/design.md`:
test and script utility dedupe for xdist worker selection, GUI ledger Markdown
table parsing, and generated-reference check/write behavior.

## Scope

- Add `tests/_support/xdist_workers.py` for shared `pytest -n auto` worker
  resolution.
- Add `scripts/_markdown_table.py` for escaped-pipe-aware Markdown table row
  splitting.
- Add `scripts/_reference_generator.py` for generated file check/write
  mechanics.
- Update the two pytest xdist hooks, GUI ledger validators, and reference
  generators to use the helpers.
- Add focused unit coverage for each helper and keep existing script behavior
  stable.

## Out Of Scope

- Moving production subpackage helpers under `_helpers/`.
- Cross-subpackage `sdk_` helper extraction.
- Any GUI behavior, docs content, or generated reference output changes.
- Any medium Phase 2 or Phase 3 dedupe work.

## Implementation Steps

### 1. Shared xdist worker resolver

- Add `tests/_support/__init__.py`.
- Add `tests/_support/xdist_workers.py` with
  `resolve_xdist_auto_workers(env, affinity_count, cpu_count)`.
- Preserve current behavior:
  - `SLURM_CPUS_PER_TASK` wins when present.
  - Invalid SLURM values still raise `ValueError`.
  - Affinity count is preferred when available.
  - Root `conftest.py` still returns `None` on platforms without affinity.
  - `tests/conftest.py` still falls back to `os.cpu_count() or 1`.
- Add `tests/unit/test_xdist_workers.py` covering SLURM precedence, invalid
  SLURM values, affinity fallback, zero clamping, and `None` fallback.
- Update both xdist hooks to delegate policy to the helper while keeping
  platform-specific `os` calls local.

### 2. Shared Markdown table row parsing

- Add `scripts/_markdown_table.py` with `split_markdown_row_cells(row)`.
- Move escaped `\|` handling from `scripts/check_features_md.py` into the
  helper.
- Update `scripts/check_features_md.py` and
  `scripts/check_workflows_md.py` to use the helper.
- Keep each validator's table recognition, warning policy, and status rules
  local.
- Add `tests/unit/gui/test_check_features_md.py` coverage for escaped pipes in
  feature rows and malformed feature-row warnings.
- Extend `tests/unit/gui/test_check_workflows_md.py` coverage for escaped pipes
  in workflow table cells.

### 3. Shared reference generator check/write runner

- Add `scripts/_reference_generator.py` with a small helper that accepts the
  output path, rendered content, check flag, script path, and regenerate
  command.
- Preserve existing user-visible messages and return codes:
  - `1` when `--check` sees a missing output file.
  - `1` when `--check` sees drift.
  - `0` when check passes.
  - Create parent directories and write the rendered output outside check mode.
- Update `scripts/generate_validation_reference.py` and
  `scripts/generate_dispatch_reference.py` to call the helper after their
  coverage checks and rendering.
- Add `tests/unit/gui/test_reference_generators.py` coverage for missing,
  drift, matching, and write-mode behavior through the helper, plus a light
  smoke check that both generator modules route through it.

## Verification

- `uv run pytest tests/unit/test_xdist_workers.py`
- `uv run pytest tests/unit/gui/test_check_features_md.py tests/unit/gui/test_check_workflows_md.py`
- `uv run pytest tests/unit/gui/test_reference_generators.py`
- `uv run python scripts/generate_validation_reference.py --check`
- `uv run python scripts/generate_dispatch_reference.py --check`
- `uv run ruff check conftest.py tests/conftest.py tests/_support/xdist_workers.py scripts/_markdown_table.py scripts/_reference_generator.py scripts/check_features_md.py scripts/check_workflows_md.py scripts/generate_validation_reference.py scripts/generate_dispatch_reference.py tests/unit/test_xdist_workers.py tests/unit/gui/test_check_features_md.py tests/unit/gui/test_check_workflows_md.py tests/unit/gui/test_reference_generators.py`
