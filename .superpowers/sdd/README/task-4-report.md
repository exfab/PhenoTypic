# Task 4 Fix Round 1 Report

## Scope

Reviewed and repaired Task 4 commit 16407547 (based on 286bd5a8) against the
binding provenance-schema-v2 design. Work was limited to migration runtime code,
focused migration tests, and this requested report.

## Finding mapping

### 1. Provenance-only seal summary could be forged

The provenance seal reader now re-derives clean, upgraded, and failures from the
current exact typed status records after checking the ordered status-byte digest.
The serialized seal summary must exactly equal the independently derived
summary. Editing only seal.json can no longer convert failed statuses into
terminal success.

Regression:
tests/unit/cli/test_cli_provenance_migration_slurm.py::
test_provenance_finalizer_rejects_edited_seal_summary

### 2. Full-run upgrade evidence vanished after downstream image failure

Each nondry full-run image worker now publishes a canonical typed provenance
receipt immediately after the root upgrade and before image migration. If image
migration fails, that receipt remains durable. When retry observes schema v2,
the publisher retains and returns the earlier schema_before=1/upgraded=true
receipt, and the completed image status carries that original evidence.

Regressions:
tests/unit/cli/test_cli_provenance_migration_slurm.py::
test_full_run_worker_preserves_sealed_upgrade_across_image_failure_retry
and the direct retry-preservation assertions in
test_full_run_image_seal_owns_typed_provenance_summary.

### 3. Full-run counts were mutable and semantically unchecked

Canonical image statuses and image seals are now schema version 2. Image
statuses contain an exact typed provenance payload bound to generation,
manifest, task identity, metadata authority, and store path. Semantic
validation enforces upgraded == (schema_before == 1), so contradictory
schema_before=2/upgraded=true evidence is rejected.

The canonical image seal owns re-derived provenance upgrade/failure totals.
Seal validation checks the current ordered status digest and independently
re-derives the summary from current typed statuses. The nondry finalizer ignores
mutable orchestration status counters and installs provenance totals only from a
validated canonical image seal. Tests cover forged seal counters and post-seal
status edits.

Regressions:
tests/unit/cli/test_cli_provenance_migration_slurm.py::
test_full_run_image_report_does_not_trust_unsealed_provenance_counts
and
test_full_run_image_seal_owns_typed_provenance_summary.

### 4. Provenance-only Slurm dry-run bypassed target lifecycle fencing

Before creating its external dry-run control root, provenance-only dry-run now
reads the target-owned lifecycle root and refuses an active generation. The
check is read-only and retains the no-scientific-write dry-run contract.

Regression:
tests/unit/cli/test_cli_provenance_migration_slurm.py::
test_provenance_dry_run_refuses_active_target_without_science_writes

### 5. Failed terminal publication closed the lifecycle without authority

The provenance-only finalizer now returns failure when typed terminal
publication fails and leaves the lifecycle active/recoverable. It closes the
lifecycle only after durable typed terminal publication succeeds.

Regression:
tests/unit/cli/test_cli_provenance_migration_slurm.py::
test_provenance_terminal_publication_failure_keeps_lifecycle_active

## RED evidence

All regressions were first run against the Task 4 implementation behavior and
failed as expected.

    uv run pytest -q tests/unit/cli/test_cli_provenance_migration_slurm.py::test_provenance_finalizer_rejects_edited_seal_summary
    FAILED: expected finalizer exit_code 1, observed 0
    1 failed

    uv run pytest -q tests/unit/cli/test_cli_provenance_migration_slurm.py::test_full_run_worker_preserves_sealed_upgrade_across_image_failure_retry
    FAILED: canonical pre-image provenance publication was missing and event order did not match
    1 failed

    uv run pytest -q tests/unit/cli/test_cli_provenance_migration_slurm.py::test_full_run_image_report_does_not_trust_unsealed_provenance_counts
    FAILED: expected provenance_upgraded 0, observed mutable orchestration value 1
    1 failed

    uv run pytest -q tests/unit/cli/test_cli_provenance_migration_slurm.py::test_full_run_image_seal_owns_typed_provenance_summary
    FAILED: ImportError for missing publish_migration_provenance_status
    1 failed

    uv run pytest -q tests/unit/cli/test_cli_provenance_migration_slurm.py::test_provenance_dry_run_refuses_active_target_without_science_writes
    FAILED: DID NOT RAISE click.ClickException
    1 failed

    uv run pytest -q tests/unit/cli/test_cli_provenance_migration_slurm.py::test_provenance_terminal_publication_failure_keeps_lifecycle_active
    FAILED: expected lifecycle active True, observed False
    1 failed

## GREEN evidence

The seal-summary, dry-run lifecycle, and terminal-publication regressions were
run individually after their fixes:

    uv run pytest -q tests/unit/cli/test_cli_provenance_migration_slurm.py::test_provenance_finalizer_rejects_edited_seal_summary
    1 passed

    uv run pytest -q tests/unit/cli/test_cli_provenance_migration_slurm.py::test_provenance_dry_run_refuses_active_target_without_science_writes
    1 passed

    uv run pytest -q tests/unit/cli/test_cli_provenance_migration_slurm.py::test_provenance_terminal_publication_failure_keeps_lifecycle_active
    1 passed

The three combined full-run typed authority/count regressions passed:

    uv run pytest -q       tests/unit/cli/test_cli_provenance_migration_slurm.py::test_full_run_worker_preserves_sealed_upgrade_across_image_failure_retry       tests/unit/cli/test_cli_provenance_migration_slurm.py::test_full_run_image_report_does_not_trust_unsealed_provenance_counts       tests/unit/cli/test_cli_provenance_migration_slurm.py::test_full_run_image_seal_owns_typed_provenance_summary
    3 passed in 0.35s

The strengthened seal test, including direct retry preservation, forged seal
summary rejection, and post-seal status mutation rejection, passed:

    uv run pytest -q tests/unit/cli/test_cli_provenance_migration_slurm.py::test_full_run_image_seal_owns_typed_provenance_summary
    1 passed in 0.29s

The complete Task 4 file passed:

    uv run pytest -q tests/unit/cli/test_cli_provenance_migration_slurm.py
    14 passed in 0.40s

The adjacent authority/Slurm test pass initially exposed one legacy
schema-version-1 test fixture:

    uv run pytest -q tests/unit/cli/test_cli_migrate_authority.py tests/unit/cli/test_cli_migrate_slurm.py
    1 failed, 108 passed in 8.65s

The fixture was updated to canonical image-seal schema version 2; its individual
rerun passed.

## Final focused verification

    uv run pytest -q tests/unit/cli/test_cli_provenance_migration_slurm.py tests/unit/cli/test_cli_migrate_authority.py tests/unit/cli/test_cli_migrate_slurm.py
    123 passed in 9.16s

    uv run ruff check --fix src/phenotypic/_cli/_cli_migrate.py src/phenotypic/_cli/_cli_migrate_manifest.py src/phenotypic/_cli/_cli_migrate_worker.py src/phenotypic/_cli/_cli_migrate_provenance_manifest.py src/phenotypic/_cli/_cli_migrate_provenance_worker.py tests/unit/cli/test_cli_provenance_migration_slurm.py tests/unit/cli/test_cli_migrate_slurm.py
    All checks passed!

    uv run mypy --follow-imports=skip src/phenotypic/_cli/_cli_migrate.py src/phenotypic/_cli/_cli_migrate_manifest.py src/phenotypic/_cli/_cli_migrate_worker.py src/phenotypic/_cli/_cli_migrate_provenance_manifest.py src/phenotypic/_cli/_cli_migrate_provenance_worker.py
    Success: no issues found in 5 source files

A normal focused mypy invocation followed the wider project import graph and
reported 17 existing errors in four unrelated modules:
gui/_smart_grid/_overlay_visuals.py,
correction/_color_correction/_helpers.py,
sdk_/mixin/_grid_inference_mixin.py, and util/image_metrics.py.
No error was reported in an affected migration module.

The repository knowledge graph was refreshed with graphify update .; it produced
no tracked diff.

## Files changed

- src/phenotypic/_cli/_cli_migrate.py
- src/phenotypic/_cli/_cli_migrate_manifest.py
- src/phenotypic/_cli/_cli_migrate_worker.py
- src/phenotypic/_cli/_cli_migrate_provenance_manifest.py
- src/phenotypic/_cli/_cli_migrate_provenance_worker.py
- tests/unit/cli/test_cli_provenance_migration_slurm.py
- tests/unit/cli/test_cli_migrate_slurm.py
- .superpowers/sdd/README/task-4-report.md

## Self-review

- Checked exact field sets, types, task identity, store identity, generation,
  manifest digest, metadata terminal digest, and provenance semantic invariants.
- Confirmed terminal counts come only from a fully validated canonical image
  seal for nondry full runs.
- Confirmed both seal-only summary edits and post-seal status mutations fail
  authority validation.
- Confirmed the original upgrade receipt survives a downstream image failure
  and a retry that observes schema v2.
- Confirmed provenance-only dry-run performs no target scientific writes while
  respecting the target lifecycle fence.
- Confirmed terminal-publication failure leaves lifecycle state recoverable.
- Ran git diff --check and reviewed the explicit Task 4 production/test diff.
- Did not run the full repository suite, per task instruction.

## Concerns

No known Task 4 correctness concern remains. The only verification limitation is
the explicitly requested focused-suite boundary; the full repository suite was
not run. The wider mypy import graph still contains the unrelated pre-existing
errors listed above.
