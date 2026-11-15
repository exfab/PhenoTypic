# C14a gate: P3 Task 4, finalizer and aggregation readers

Classified by the C14a implementer; written here by the orchestrator. Files in
`/bigdata/exfab/anguy344/gate-results/c14a/`: `classified.tsv` (each red with its
prior tag), `owners.tsv` (each still-red C14a test with its current owner),
`classify.py`, `subclassify.py`.

- **Job:** `32095455`, 24 shards on `short`, detached worktree at `0b0043a38`
  (clean at start and end; `phenotypic` imported from the gate tree). Cleanup
  `32095456` COMPLETED; the worktree is gone (checked by the orchestrator).
- **Totals:** 10,272 tests; 106 failed, 11 errors, 47 skipped → 117 reds
  (C14b: 348).
- **Commits:** `eccb770ee` (aggregation readers, `certified_table_run`),
  `65ab7a6cd` (`metrics.json` `pipeline_sha256`, new tests, fixed pipeline names in
  `tests/_tiny_run.py`), `0b0043a38` (fixtures read the run folder), `f5b36e400`
  (gate regression fix), `d3cb2879d` (forward test-side v1 reads).

## Outcome of the tests tagged for C14a

- **252 "blocked on C14a finalize":** 179 pass; 1 more passes under its new name
  (`test_run_summaries::test_metrics_pipeline_sha_is_null_while_v1_tables_name_no_pipeline`
  → `…_is_the_certified_run_folders_shared_digest`); 72 still red (61 failures,
  11 setup errors), all at another cluster's wall below.
- **55 "C14a's own readers":** 46 pass; 9 still red, at another cluster's wall.

## The 81 still red, by current owner

- **C15 (migrate), 41:** test_cli_migrate_mode 12, test_migrate_end_to_end 10,
  test_embedded_measurement_migration 7, test_migration_republishes_state 4,
  test_finalize_run 8 (all `_build_v1_tree`: `replace_embedded_measurement_table`
  refuses a v2 store until Task 5).
- **C14c (recompile), 33:** test_cli_recompile_slurm 13, test_cli_recompile 6,
  test_cli_exit_codes 6 (the 5 recompile ones plus
  `test_a_forward_slurm_run_with_nothing_left_exits_4`), test_dvc_layout_guards 2,
  test_metadata_snapshot_suffix 2, test_tsv_cutover_old_tree 2,
  test_dvc_outs_at_submission 1,
  `test_finalize_run::test_every_mode_produces_a_byte_identical_master[recompile]`,
  test_run_summaries 1. All go through `_cli_recompile_recovery.py:490` ("Cannot
  safely restore measurement authority") or its overlay twin.
- **C13, 4:** TestMeasureRerun ×3, and
  `test_cli_store_output::TestForwardRunStoreLayout::test_forward_run_produces_store_only`
  (a test-side v1 read; left to C13, which has the file open).
- **Fixed after the gate, 3 (`d3cb2879d`):** test_promote_time_metadata ×2,
  `test_cli_v2::TestEdgeCases::test_single_image_processing` — test-side v1
  reads. `4 passed`.

Unchanged from C14b's assignment: GUI expected-red 18 (README P0/P1/P3), C14d 7,
C14c 6, C13 4.

## New red: 1, fixed in `f5b36e400`

`test_embedded_measurement_migration::test_exact_embedded_table_comparison_includes_schema_order_nulls_and_fanout`.
`eccb770ee` made `embedded_measurement_table_matches` read the store's tables
schema first, so a store with no root refused the comparison. The resolver now
applies only to v2 stores; anything else is compared at the flat path as before.
After the fix it passes; `test_tables_v2_readers` is `10 passed, 1 failed`, the
failure being C13's uncommitted
`test_measure_with_another_pipeline_adds_a_folder_and_recompile_keeps_it`.

## Deviations

- `_measurement_sources.py:351`, a regex off the plan's list, matched only the flat
  path, so a v2 master's `Metadata_ImageName` became `"table"`. Fixed in
  `eccb770ee`. Task 4's Step 3 grep misses regexes.
- `tests/_tiny_run.py`'s pipeline builders take fixed names: the default
  `run_cli` pipeline is byte-identical to `pipeline_file_a`, as its docstring says.
- `_aggregated_pipeline_digests` resolves each run from the source path (the
  artifact its record certifies) instead of re-reading every record through
  `certified_table_run`; the two give the same run.

## Out of scope, not fixed

- `_cli_readme_generator.py:131,175` still documents `tables/measurements/table.parquet`
  in the user-facing run README. Assigned to C14d.
