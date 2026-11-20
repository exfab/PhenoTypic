# C16r gate: P3 review fixes

Classified by the C16r implementer; written here by the orchestrator. Per-test
list: `/bigdata/exfab/anguy344/gate-results/c16r/classified.tsv`. Findings are
those of `review-p3.md`.

- **Job:** `32096812`, 24 shards on `short`, 567 files (C15's 564 plus
  `tests/unit/cli/test_migrate_rerun_v2.py`, `tests/unit/sdk_/test_tables_v2_carry_rules.py`,
  `tests/unit/cli/test_finalize_mixed_pipelines.py`), detached worktree at
  `462128436`. Every shard printed `dirty=0` at start and end and imported
  `phenotypic` from the gate tree. All 24 COMPLETED 0:0.
- **Totals:** 10,374 tests: 10,314 passed, 13 failed, 0 errors, 47 skipped
  (C15: 14 failed). The orchestrator's independent junit sum agrees.

## Classification

- **Still red (13), all GUI expected-red (README P0/P1/P3, #283)**, the same
  13 rows as C15: test_setup_view ×3, test_measurement_routes::test_an_undeclared_column_never_opens_the_parquet,
  test_measurement_source ×5, test_curation_does_not_unfence_the_run ×1,
  test_setup_authoring ×2, test_validation ×1.
- **New red:** none.
- **Green now:** C15's timing flake
  `test_run_console_callbacks::test_immediate_local_exit_terminalizes_allocated_generation`.

## Commits

| Finding | Commit | Change |
|---|---|---|
| C1 | `330ddaa63` | A migrate re-run certifies the folder the record certifies; the latest folder is the fallback only with no record or a v1 one. |
| I1 | `5c8cd475f` | `publish_image_success(provenance=)`; `_republish_table_marker` (shared with measure) carries `provenance` and `source_provenance`. |
| I2 | `6c79e41c0` | `_existing_marker_identity` reads the record first, the legacy marker only when no record carries an identity. |
| OQ3 | `89eba92ab` | `journal_table_run` resolves a `legacy` last application through `v1_table_pipeline_sha`. |
| M2 | `92b97d45c` | One `import_table_run_id` for the importer and migrate's checks; `embedded_measurement_table_matches(run_id=)`. |
| M4 | `2ce9ef3d1` | A save refuses an unparseable root; only `migrate_hdf_to_zarr` passes `save2zarr(replace_unreadable=True)`. |
| M1, M7 | `df2125d96` | A writer given `upstream=` (even `{}`) replaces the block; `run_upstream_collisions` logs collisions; the M7 tests. |
| OQ4 | `804f4ee35` | `publish_run_summaries` warns naming each pipeline's sha12 and table count; never refuses. |
| M6 | `462128436` | `tracked_state.md`, `_cli/CLAUDE.md`, the OME-Zarr skill, one `_cli_migrate_state.py` docstring. |

## Decisions

- **I2 epoch (orchestrator, correcting its own triage):** the triage said to
  keep `lifecycle_epoch` verbatim. That was wrong: under SLURM
  `publish_image_success` refuses any epoch but the live one, so a SLURM migrate
  could never republish. A rebuilt record keeps `work_id` (when it matches, or
  the record is migrated), `relative_image_path`, `mode`, `attempt_id`,
  `provenance` and `source_provenance`; the epoch takes the live migration
  generation, else the record's. `_cli/CLAUDE.md` names the epoch as the one
  exception, with the reason.

## Verification

Each fix's tests were seen red with the pre-fix source restored temporarily:

- **C1:** `test_migrate_over_a_tree_certifying_an_older_folder_changes_nothing`,
  `test_the_certified_folder_is_the_one_migrate_keeps`.
- **I1:** `test_the_tables_v2_republish_keeps_provenance[migrated]`,
  `test_a_migrated_record_with_its_own_work_id_stays_valid`,
  `test_the_measure_republish_keeps_provenance[migrated]`.
- **I2:** both `test_a_crash_before_the_tables_v2_republish_keeps_identity`
  cases, `test_the_crash_path_and_the_clean_path_publish_the_same_identity`,
  `test_a_record_for_another_work_id_is_not_borrowed`.
- **OQ3:** `test_a_legacy_journal_resolves_to_the_folder_migrate_named`,
  `test_after_migrate_a_staged_continuation_reuses_the_migrated_image`.
- **M2:** `test_a_multi_run_store_with_a_legacy_parquet_converges_in_one_pass`
  ("embedded table does not exactly match the prepared source", as predicted).
- **M4:** `test_an_ordinary_save_refuses_a_store_whose_root_cannot_be_parsed`,
  `test_the_hdf_rebuild_replaces_an_unparseable_store_whole`,
  `test_migrate_hdf_to_zarr_passes_the_explicit_replace`.
- **M1:** `test_a_restart_drops_an_upstream_entry_the_input_no_longer_has`,
  `test_an_order_collision_with_the_input_is_logged`.
- **M7:** pin current behaviour, so green before and after:
  `test_restart_keeps_another_pipelines_folder_byte_identical`,
  `test_a_same_pipeline_replace_image_tables_drops_that_runs_embeddings`
  (intended until P4 adds `embeddings=`).
- **OQ4:** not run against the old source. The three tests in
  `test_finalize_mixed_pipelines.py` import `_warn_on_mixed_pipelines`, which
  only the fix adds, so before it they fail at import rather than on their
  assertions. The CLI-path test asserts the warning names both shas with their
  table counts, `metrics.json` has `pipeline_sha256: null`, and the run exits 0.
- **M6:** docs only; the five test files that read the edited pages:
  `183 passed, 2 skipped`.
