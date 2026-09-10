# What this review slice is, and what it deliberately leaves out

**This branch is a REVIEW SLICE, not a feature branch.** It is `main` plus one
subset of the change on `cli-gui-state-tracking`, cut so the diff fits a review
budget. It is disposable and is never merged.

## Why you are seeing source without its unit tests

This slice carries **all source** for the run-state model and `--mode migrate`,
plus every `tests/integration/cli` test. The **unit** tests for this code are
real and extensive but live in the sibling slice
`review/state-and-migrate-tests` (6 files, 7,560 lines), because the two
together exceed the budget.

**So absence of a unit test in this diff is not evidence of missing coverage.**
Please do not report "this is untested" for anything in the table below; check
the named files first, or raise it as a question rather than a finding.

## Coverage that exists on the full branch

Counted by test files that reference each module by name, with the number of
`def test_` functions in them. It is a locator, not a coverage measurement --
it says where to look, not that every branch is exercised.

| Source file in this slice | tests | files | Primary test files |
|---|---|---|---|
| `src/phenotypic/_cli/_cli_completion.py` | 693 | 34 | `tests/unit/cli/test_migrate_state.py` (59)<br>`tests/unit/sdk_/test_run_state.py` (56) |
| `src/phenotypic/_cli/_cli_identity.py` | 96 | 2 | `tests/unit/cli/test_migrate_state.py` (59)<br>`tests/unit/cli/test_run_identity.py` (37) |
| `src/phenotypic/_cli/_cli_image_record.py` | 83 | 2 | `tests/unit/cli/test_migrate_state.py` (59)<br>`tests/unit/cli/test_image_record.py` (24) |
| `src/phenotypic/_cli/_cli_migrate.py` | 322 | 10 | `tests/unit/cli/test_cli_migrate_mode.py` (74)<br>`tests/unit/cli/test_migrate_state.py` (59) |
| `src/phenotypic/_cli/_cli_migrate_image.py` | 234 | 8 | `tests/unit/sdk_/test_run_state.py` (56)<br>`tests/unit/cli/test_cli_migrate_authority.py` (42) |
| `src/phenotypic/_cli/_cli_migrate_manifest.py` | 222 | 7 | `tests/unit/cli/test_migrate_state.py` (59)<br>`tests/unit/cli/test_cli_migrate_authority.py` (42) |
| `src/phenotypic/_cli/_cli_migrate_provenance.py` | 47 | 3 | `tests/unit/cli/test_schema_gate.py` (24)<br>`tests/unit/cli/test_cli_provenance_migration_slurm.py` (14) |
| `src/phenotypic/_cli/_cli_migrate_provenance_worker.py` | 75 | 3 | `tests/unit/cli/test_migrate_state.py` (59)<br>`tests/unit/cli/test_cli_provenance_migration_slurm.py` (14) |
| `src/phenotypic/_cli/_cli_migrate_state.py` | 133 | 2 | `tests/unit/cli/test_cli_migrate_mode.py` (74)<br>`tests/unit/cli/test_migrate_state.py` (59) |
| `src/phenotypic/_cli/_cli_migrate_worker.py` | 120 | 4 | `tests/unit/cli/test_migrate_state.py` (59)<br>`tests/unit/cli/test_cli_migrate_slurm.py` (30) |
| `src/phenotypic/_cli/_cli_state_management.py` | 532 | 20 | `tests/unit/cli/test_cli_v2.py` (81)<br>`tests/unit/cli/test_migrate_state.py` (59) |
| `src/phenotypic/sdk_/_digests.py` | 93 | 2 | `tests/unit/sdk_/test_run_state.py` (56)<br>`tests/unit/cli/test_run_identity.py` (37) |
| `src/phenotypic/sdk_/_hdf_to_zarr.py` | 533 | 16 | `tests/unit/sdk_/test_io_constants.py` (138)<br>`tests/unit/cli/test_cli_migrate_mode.py` (74) |
| `src/phenotypic/sdk_/_image_record.py` | 153 | 6 | `tests/unit/cli/test_migrate_state.py` (59)<br>`tests/unit/sdk_/test_run_state.py` (56) |
| `src/phenotypic/sdk_/_io_constants.py` | 462 | 29 | `tests/unit/sdk_/test_io_constants.py` (138)<br>`tests/unit/sdk_/test_run_state.py` (56) |
| `src/phenotypic/sdk_/_metadata_migration.py` | 260 | 7 | `tests/migration/test_metadata_migration_journal.py` (76)<br>`tests/unit/cli/test_cli_migrate_mode.py` (74) |
| `src/phenotypic/sdk_/_run_state.py` | 367 | 13 | `tests/unit/gui/shell/test_runs_registry.py` (69)<br>`tests/unit/cli/test_migrate_state.py` (59) |
| `src/phenotypic/sdk_/_state_types.py` | 123 | 5 | `tests/unit/sdk_/test_run_state.py` (56)<br>`tests/unit/cli/test_run_identity.py` (37) |
| `src/phenotypic/sdk_/_verification_cache.py` | 51 | 3 | `tests/unit/sdk_/test_verification_cache_disk.py` (29)<br>`tests/unit/sdk_/test_verification_cache.py` (20) |

## The three test files most worth opening

| File | tests | Proves |
|---|---|---|
| `tests/unit/cli/test_migrate_state.py` | 59 | every migrate planner/applier, retention, `--revert` |
| `tests/unit/sdk_/test_run_state.py` | 56 | verdict precedence, the liveness fence, schema advisories |
| `tests/unit/cli/test_run_identity.py` | 37 | the five identity tokens and what does/does not mint a new one |

## Full context

- Branch under review: `cli-gui-state-tracking` (241 commits, 40,694 lines of src+tests)
- The whole change passed a 24-shard regression at `92f55986`: **zero failing names, 12,535 tests**
- State model reference: `docs/source/contrib_guide/tracked_state.md`
- Every drift found during execution: `docs/superpowers/reports/2026-09-03-cli-gui-state-tracking/document-drift.md`
