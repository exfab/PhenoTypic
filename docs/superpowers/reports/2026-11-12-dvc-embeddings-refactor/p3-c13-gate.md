# C13 gate: P3 Task 3, the measure rewrite

Classified by the C13 implementer; written here by the orchestrator. Files in
`/bigdata/exfab/anguy344/gate-results/c13/`: `classified.tsv` (each red with its
C14a tag), `owners.tsv` (each red with its current owner), `classify.py`,
`owners.py`.

- **Job:** `32095630`, 24 shards on `short`, detached worktree at `d9cc6c4e9`
  (dirty=0 at start and end; `phenotypic` imported from the gate tree). Cleanup
  `32095655` exited 0 and removed the worktree.
- **Totals:** 10,126 passed, 94 failed, 11 errors, 48 skipped → 105 reds
  (C14a: 117). No new reds.
- **Commits:** `ee59426d2` (`replace_image_tables(..., *, pipeline_sha256)` picks
  its folder with `choose_table_run_id` and returns the `TableRun`), `d9cc6c4e9`
  (measure-mode record certifies that run; `_republish_table_marker(measurements=)`;
  tests onto the resolver), `8bedd5569` (`pipeline_sha256` required, keyword-only,
  on `OutputManager.replace_image_store_measurements`).

## Green now: 12

- C13's own: test_embedded_measurement_replacement ×3,
  `test_directory_scanner::test_measure_core_embeds_the_table_in_the_bare_stem_store`.
- Left to C13 by C14a: TestMeasureRerun ×3,
  `TestForwardRunStoreLayout::test_forward_run_produces_store_only`.
- C14a's post-gate fixes, now inside a gate: test_promote_time_metadata ×2,
  `test_cli_v2::TestEdgeCases::test_single_image_processing` (`d3cb2879d`),
  `test_embedded_measurement_migration::test_exact_embedded_table_comparison_…`
  (`f5b36e400`).

## Still red: 105, by owner

- **C15, migrate (41):** test_cli_migrate_mode 12, test_migrate_end_to_end 10,
  test_embedded_measurement_migration 7, test_migration_republishes_state 4,
  test_finalize_run 8 (the v1 replacer refuses a v2 store).
- **C14c, recompile (39):** test_cli_recompile_slurm 15, test_cli_recompile 6,
  test_cli_exit_codes 6, test_dvc_layout_guards 2, test_metadata_snapshot_suffix 2,
  test_tsv_cutover_old_tree 2, test_recompile_no_store_writes 2,
  test_dvc_outs_at_submission 1, `test_finalize_run[recompile]` 1,
  test_run_summaries 1, test_embedded_measurement_recompile 1.
- **GUI expected-red (18):** test_measurement_routes 6, test_measurement_source 5,
  test_setup_view 3, test_setup_authoring 2, test_curation_does_not_unfence_the_run 1,
  test_validation 1 — the README's P0/P1/P3 rows.
- **C14d (7):** test_run_state 7.

## New tests at the gate

- `tests/unit/sdk_/test_measure_rewrite_v2.py`: 5 passed.
- `test_measure_twice_with_the_same_pipeline_keeps_one_folder_and_row_count`: passed.
- `test_measure_with_another_pipeline_adds_a_folder_and_recompile_keeps_it`:
  xfailed, only on the C14c refusal message "Cannot safely restore measurement
  authority" (`_cli_recompile_recovery.py:468`); C14c removes the xfail.

## Mutation checks (detached tree, originals restored and hash-checked)

- selecting the latest run instead of the digest kills 9 tests;
- clearing the whole `tables/` kills 5;
- dropping `measurements=` from the republish kills 2.

## Since the gate

`8bedd5569` — the four touched files: `9 failed, 72 passed`, all in
test_finalize_run with names and signatures already in C14a's list (8 C15, 1 C14c).

## Notes

- Orchestrator decision: `pipeline_sha256` is required on the OutputManager (a
  forgotten sha would silently write an `-unknown` folder).
- A same-pipeline measure rewrite rebuilds that run's entry from its tables, so
  embeddings in that same folder are dropped; other folders' are carried.
  Consistent with D6; to be checked against P4's `embeddings=` keyword.
- Pre-existing mypy error, untouched: `_measurement_tables.py:1486`.
