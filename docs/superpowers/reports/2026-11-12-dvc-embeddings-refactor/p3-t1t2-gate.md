# C12 gate: P3 Tasks 1–2 (tables v2 writer)

Classified by the C12 implementer; written to this file by the orchestrator
(the harness refuses report-file writes from subagents).

- **Tree:** `1a634df93`, clean at the start and end of every shard.
- **Job:** `32076656`, 24 shards on `short`, all `COMPLETED 0:0`. Logs, junit,
  `tagged.tsv` (every red with its tags) and `reds/` (per-test traceback text)
  are in `/bigdata/exfab/anguy344/gate-results/c12-p3-t1t2/`.
- **Surface:** 559 test files in `tests/unit/{sdk_,core,cli,plotting,gui}`,
  `tests/integration/{cli,gui}` and `tests/gui`. Wider than the importers,
  because every store test calls `save2zarr`.
- **Command:**
  `uv run --no-sync pytest -q --no-header -p no:randomly -o addopts= -m "not slow" -rfE -n 8 --junitxml=…`
- **Totals:** 10,256 tests; 203 failures, 167 errors, 47 skipped. 370 reds:
  (a) 18, (b) 351, (c) 1.

## How the reds were attributed

The junit text of each red was tagged by its signature.

- **An image fails during a CLI run (316 reds).** The CLI run exits 3, or a
  fixture run prints `Failed: N`. Reproduced with a one-image local full run at
  `1a634df93`: every image fails with
  `FileNotFoundError: …/plate_1.ome.zarr/tables/measurements`. The per-image
  record still declares the v1 table path as its `measurements` artifact
  (`_cli_process_single.py:1090`, `_cli_execution_strategies.py:183`), and
  `publish_image_success` resolves it with `strict=True`. C14b.
- **Staged runs (18 reds)** fail the same way at Stage 3 (`STAGED FINALIZATION
  FAILED`, `completed=0, failed=1`, or a missing image record). The Stage-3
  record names the v1 path at `_cli_staged_strategy.py:113-117`,
  `_cli_staged_resume.py:292-297,406`, `_cli_staged_slurm_worker.py:342,433,565`.
  C14b. Two needed a closer look:
  - `test_slurm_stage3_worker_writes_and_backfills_overlay` ("pipeline
    reloaded"): the second `run_stage3_step` takes its already-complete
    shortcut only if the first published, and that publish failed on the v1
    path (`_cli_staged_slurm_worker.py:433`).
  - `test_local_stage3_crash_between_deletes_leaves_the_benign_survivor`: the
    Stage-2 token survives because Stage 3 failed before consuming it.
- **Direct v1 reads (99 reds).** The traceback opens `tables/measurements/…` or
  indexes `tables.measurements`, all at reader sites on T4's list. A few tests
  read the v1 path in their own helpers: `test_staged_store_stages`,
  `test_staged_nested_equivalence`, `test_run_state`.
- **Migrate (95 reds, overlapping the above).** Their fixtures do a forward run
  first, so they hit C14b first. Behind that, migrate's own readers use the flat
  path (`_cli_migrate.py:1384,1484`, `_cli_migrate_image.py:205,231,307,535`),
  and `replace_embedded_measurement_table` stays v1 until T5 and now refuses a
  v2 store. C15.
- **No red's traceback runs through the new writer code** (`carry_table_runs`,
  `_write_table_runs`, `write_run_entry`, `choose_table_run_id`, `_parse_runs`,
  `refresh_embedding_staleness`), except group (c).

## (c) Not explained by the v2 layout: 1, fixed in `af122f26e`

- **Test:**
  `tests/unit/cli/test_cli_migrate_image.py::test_invalid_store_is_replaced_from_the_explicit_hdf_source`,
  `MigrationImageStageError: KeyError: 'phenotypic'`.
- **Cause:** `--mode migrate` rebuilds an invalid store from its HDF source with
  `save2zarr` (`_hdf_to_zarr.py:335`). The new `carry_table_runs` read the
  replaced store's root with `read_phenotypic_attributes`, which raises
  `KeyError` on a root with no `phenotypic` block. Before T2 a save never read
  the replaced store's root this way.
- **Fix:** `_replaced_store_block` (`sdk_/_measurement_tables.py`) is the single
  read of the replaced store, used by the carry and by folder numbering in
  `_write_table_runs`. An absent store, a root with no `phenotypic` block, or an
  unparseable root (logged) carries nothing — what a save always did.
- **Pinned by:**
  `tests/unit/sdk_/test_tables_v2_writer.py::test_a_save_replaces_a_store_with_no_phenotypic_block`,
  parametrized over an empty-attributes root and a non-JSON root.
- **Verification:** the pin plus the migrate test, `-n 4`: `3 passed in 19.44s`.
  Touched surface after the fix (test_tables_v2_writer, test_table_runs,
  test_embedded_measurement_tables, test_embedded_table_inversion,
  test_cli_migrate_image, test_ngff_promote, test_image_figures_store, `-n 8`):
  `145 passed, 1 skipped in 40.77s`.

## (a) Expected red: 18, matching the README table

- **P0:** `tests/unit/gui/tune/test_validation.py::test_tuning_spec_path_is_deployable`;
  `tests/unit/gui/tune/test_setup_authoring.py::{test_write_authored_setup_spec_preserves_existing_spec_search_space, test_existing_spec_replaces_scorer_only_when_requested}`;
  `tests/integration/gui/tune/test_setup_view.py::{test_existing_spec_controls_preserve_seed_budget_and_storage, test_existing_spec_trial_override_preserves_authored_strategy, test_existing_grid_spec_never_emits_trial_override}`.
- **P1:** `tests/unit/gui/test_curation_does_not_unfence_the_run.py::test_the_proof_still_records_every_published_artifact`.
- **P3, `tests/unit/gui/results_viewer/test_measurement_routes.py` (6):**
  `test_an_undeclared_column_is_400`, `test_a_path_traversal_column_is_400`,
  `test_a_non_numeric_column_is_400`, `test_serves_one_column_keyed_by_object_label`,
  `test_serves_only_the_requested_column`, and one more over the `route` fixture
  whose traceback opens the v1 path. `test_an_undeclared_column_never_opens_the_parquet`
  stayed green, which the README allows.
- **P3, `tests/unit/gui/results_viewer/test_measurement_source.py` (5):**
  `test_the_picker_offers_the_stores_own_numeric_columns`,
  `test_values_are_keyed_by_dataset_image_and_object_label`,
  `test_an_unmeasured_image_contributes_no_keys`, `test_a_rewritten_table_is_read_fresh`,
  `test_one_unmeasured_image_does_not_empty_the_picker`.
- **Unverified:** every test in `test_measurement_join_migration_run.py` skipped
  ("migration-test run absent, or mid-republish"), so its P3 row is neither red
  nor green at this gate.

## (b) Closed by T3–T5: 351

| Module | Reds | First failure | Then |
|---|---|---|---|
| unit/cli/test_cli_migrate_mode | 48 | C14b | C15 |
| unit/cli/test_finalize_run | 35 | v1 path | C14a |
| unit/cli/test_finalize_fanout | 19 | v1 path | C14a |
| unit/cli/test_cli_recompile_slurm | 19 | C14b / v1 path | C14c |
| unit/cli/test_run_summaries | 18 | C14b | C14a |
| unit/cli/test_embedded_measurement_migration | 16 | C14b | C15 |
| integration/cli/test_staged_gpu_local | 15 | C14b (staged) | – |
| integration/cli/test_migrate_end_to_end | 13 | C14b | C15 |
| unit/cli/test_cli_wait | 13 | C14b | – |
| unit/cli/test_cli_reference_e2e | 13 | C14b | – |
| integration/cli/test_cli_store_output | 11 | C14b | C13 (TestMeasureRerun) |
| unit/cli/test_metadata_snapshot_suffix | 9 | C14b | – |
| unit/cli/test_cli_exit_codes | 9 | C14b | – |
| unit/sdk_/test_run_state | 7 | v1 path / `tables.measurements` | C14d |
| unit/sdk_/test_metadata_canonical_view | 7 | C14b | – |
| unit/sdk_/test_migration_republishes_state | 6 | C14b | C15 |
| unit/cli/test_deliverables_tsv | 6 | C14b | – |
| unit/cli/test_cli_recompile | 6 | C14b | C14c |

- **5 each:** test_tsv_cutover_old_tree (C14b), test_staged_stage1_reads_like_single_pass
  (C14b), test_recompile_no_longer_migrates (C14b, then C14c/C15),
  test_cli_grid_preset_refusal (C14b), integration test_dvc_layout_guards (C14b).
- **4 each:** test_migration_predicate (C14b, then C15), integration
  test_cli_provenance_original (C14b).
- **3 each:** test_hdf_to_zarr (fixture on C14b, then C15),
  test_embedded_measurement_replacement (C13), test_embedded_measurement_publication
  (C14b), integration test_rolling_input, test_phenotypic_cache_layout,
  test_calibration_figure_in_store (C14b).
- **2 each:** test_directory_scanner (C13 / C14b), test_recompile_no_store_writes
  (C14c), test_staged_nested_equivalence and integration test_staged_store_stages
  (the tests' own v1 helpers, C14b), test_cli_slurm_submission_summary,
  test_cli_v2.TestEdgeCases, integration test_promote_time_metadata,
  test_figures_run_date_cli, test_deliverables_byte_stable (C14b).
- **1 each:** test_embedded_measurement_recompile (C14c),
  test_embedded_measurement_checkpoint and test_completion_split (C14a),
  test_dvc_outs_at_submission, test_cli_reference_wiring,
  test_cli_processing_report_on_demand, test_cli_preload_everywhere,
  test_cli_preflight_ordering, integration test_rembi_manifest_emitted,
  test_process_store_consumers, test_lifecycle_publication_races,
  test_cli_metadata_deliverable (C14b).

Every name is a line in `tagged.tsv` (test id, kind, tags, first message line).
The README's new subsection "Intra-P3 red outside the GUI (closed within P3)"
records these by cluster; they are not accepted at the P3 phase gate.

## Dispatch order (orchestrator's decision)

C14b's record sites are the keystone: until they move, no CLI run completes an
image, so the gates of C13, C14a, C14c and C14d cannot separate their own reds
from C14b's. C14b and C13 also share `_cli_process_single.py`. Order:
**C14b → C13 → C14a → C14c → C14d → C15 → C16**, sequential, because every
cluster's gate runs in this one worktree.

## Not run

- **T2 Step 5, the ngio discovery check:** the subagent's permission classifier
  refused `uv run --with ngio …`; left for the user to decide.
