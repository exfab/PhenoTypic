# C15 gate: P3 Task 5, v1 refusal and migrate conversion

Classified by the C15 implementer; written here by the orchestrator. Per-test
list: `/bigdata/exfab/anguy344/gate-results/c15/classified.tsv`.

- **Job:** `32096296`, 24 shards on `short`, 564 files, detached worktree at
  `cbd8d914d` (dirty=0 at start and end; `phenotypic` imported from the gate
  tree). Cleanup `32096297` COMPLETED 0:0 and removed the worktree.
- **Totals:** 10,284 passed, 14 failed, 0 errors, 47 skipped, 23 xfailed
  (C14d: 44 failed, 11 errors).
- **Commits:** `9f6c148a4` (sdk: `convert_tables_v1_to_v2`, `v1_table_pipeline_sha`,
  `declared_measurement_table_path`; `replace_embedded_measurement_table` writes v2
  and converts a v1 store in the same transaction; the v1 writer is deleted),
  `d57eaa33b` (cli: `_cli_tables_v1.first_v1_store` and the forward refusal in
  `_refuse_unmigrated_output`; the tables_v2 step; migrate's flat-path readers,
  including the `_publish_migration_aggregate` glob, through the resolver;
  `tests/unit/cli/test_migrate_tables_v2.py`), `ad749fc80` (migrate tests read the
  run folder), `cbd8d914d` (docs), and after the gate `ecc2a97da` and `3a576baa5`
  (tests and docs only).

## Classification

- **Green now (42):** every C15 red — test_cli_migrate_mode 12,
  test_migrate_end_to_end 10, test_embedded_measurement_migration 7,
  test_migration_republishes_state 4, test_finalize_run `_build_v1_tree` 8 — plus
  the README test fixed in `b8ed191ab`. Root cause of most: the demotion fixture
  `tests/unit/sdk_/_migration_fixtures.py` read the flat path off a v2 store and so
  built an HDF-only tree with no legacy Parquet. The 8 test_finalize_run tests
  needed no change; they were red only because the replacer refused v2.
- **Still red (13), all GUI expected-red (README P0/P1/P3, #283):**
  test_setup_view ×3, test_measurement_routes::test_an_undeclared_column_never_opens_the_parquet,
  test_measurement_source ×5, test_curation_does_not_unfence_the_run ×1,
  test_setup_authoring ×2, test_validation ×1.
- **New (1), a timing flake:**
  `tests/integration/gui/test_run_console_callbacks.py::test_immediate_local_exit_terminalizes_allocated_generation`
  — "local exit observer did not terminalize the run", a 10 s deadline (:1832-1841)
  on a real `python -m phenotypic` subprocess, shard 18. The whole file passed 5 of 5
  isolated runs at HEAD (45 passed, 5.8–7.7 s); `first_v1_store` only stats a missing
  `results/` there, and no `_gui` file changed.
- **xfails (10 in the migrate files) predate C15** (`git log -S xfail 9f6c148a4^..HEAD -- tests/`
  is empty); they are reachable now that setup succeeds. Nine are P7 U-10
  (`_hdf_to_zarr._republish_image_marker` rewrites the legacy marker and writes no
  record, so `valid_image_success` is false for every migrated image); one is the
  strict "TOLERATED, not fixed (user ruling 2026-09-09)" pollution test.

## Decisions

- **Placement (accepted):** in a full-run migrate the tables_v2 step runs inside each
  image task, right after the table stage (local loop and SLURM image array), so it
  republishes a current record via `_republish_table_marker` before the marker stage
  re-validates it. Direct-store and process-tree targets run it in their own store
  array (`convert_provenance_store_tables`). D10 requires conversion then a record
  republish then the run proof; this satisfies it. Reason recorded in
  `_cli/CLAUDE.md`.
- **`--overwrite` does not bypass the v1 refusal** (orchestrator, reversing a first
  instinct): the existing unconverted-tree (`.h5`) refusal also ignores
  `--overwrite`, and `--overwrite` should never silently delete a tree that still
  needs migrating. Pinned: refusal before the `--overwrite` rmtree, before `--dry-run`
  exits, and under `--restart` — each exit 2, names `--mode migrate`, no traceback,
  tree byte-identical.
- The replacer writes the measurements table directly rather than through
  `write_run_entry`, to keep the run's metadata and embeddings as they were.

## Verification

- `test_migrate_tables_v2.py`: `24 passed`. With `first_v1_store` disabled, five
  refusal tests go red; removing either conversion step turns 4 tests red (files
  restored and hash-checked).
- The seven migrate files: `190 passed, 1 skipped, 10 xfailed`.

## For the P3 review

- C15 noted: if a later migrate pass finds a record stale, the marker stage's
  fallback (`_existing_marker_identity`) reads only legacy `image_complete/` markers,
  so a record-era tree would get default identity.
- The migrate report counts no conversions; a dry run reports a v1 store as work.
