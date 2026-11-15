# C14c gate: P3 Task 4, recompile readers

Classified by the C14c implementer; written here by the orchestrator. Per-test
list: `/bigdata/exfab/anguy344/gate-results/c14c/classified.tsv` (from
`classify.py`, keyed on C13's `owners.tsv`).

- **Job:** `32095810`, 24 shards on `short`, detached worktree at `d0d716065`
  (dirty=0 at start and end; `phenotypic` imported from the gate tree). Cleanup
  `32095841` exited 0 and removed the worktree.
- **Totals:** 10,168 passed, 56 failed, 11 errors, 47 skipped → 67 reds (C13: 105).
- **Commits:** `1266bc0d2` (recompile reads the run folder each record certifies:
  `certified_measurement_table`, `store_holds_measurement_table`,
  `table_lies_in_store` in `_cli_recompile_recovery.py`; authority assert,
  transition readers, overlay planning, repair/refresh with `table_path=`, the
  worker, `phenotypicCLI._regenerate_missing_overlays`; three new tests; C13's
  xfail removed), `d0d716065` (recompile tests' own v1 joins through the resolver),
  `98633c2b4` (overlay-pass fix, after the gate — see below).

## Green now: 39, every red assigned to C14c

- Tagged recompile (32): test_cli_recompile_slurm 12, test_cli_recompile 6,
  test_cli_exit_codes 5, test_dvc_layout_guards 2, test_metadata_snapshot_suffix 2,
  test_tsv_cutover_old_tree 2, test_dvc_outs_at_submission 1,
  `test_finalize_run[recompile]` 1, test_run_summaries 1.
- C14b-era (6): test_cli_recompile_slurm 3, test_recompile_no_store_writes 2,
  test_embedded_measurement_recompile 1.
- Unassigned (1): `test_cli_exit_codes::test_a_forward_slurm_run_with_nothing_left_exits_4`.
- C13's `test_measure_with_another_pipeline_adds_a_folder_and_recompile_keeps_it`
  passes without its xfail: `metrics.json` names pipeline B's folder (I17).

## Still red: 67

- **C15, migrate (41):** test_cli_migrate_mode 12, test_migrate_end_to_end 10,
  test_embedded_measurement_migration 7, test_migration_republishes_state 4,
  test_finalize_run 8 (`_build_v1_tree`).
- **GUI expected-red (18):** README P0/P1/P3 rows.
- **C14d (7):** test_run_state.
- **New (1), a timing flake:**
  `tests/integration/gui/test_results_async_binding.py::test_latest_bind_advances_after_one_of_two_saturated_workers_checkpoints`
  — "binding job did not become terminal" from `_poll_terminal(..., timeout=3.0)`
  (:386), shard 15. No `_gui`/`OutputRoot` code was touched; the file passed
  `10 passed` in 5 of 5 isolated re-runs at HEAD (3.1–12.1 s). Recorded as a
  load-sensitive flake; re-check at the next gate.

## After the gate: `98633c2b4`

- **Defect (pre-existing):** `_regenerate_missing_overlays` asked the legacy
  completion marker whether an image claims a table. On a record-era tree that
  marker never exists, so when a record claimed a table missing from disk, local
  recompile rendered one overlay PNG into `deliverables/overlays/` before the later
  authority check refused the run (exit 1 either way; no store byte written; SLURM
  unaffected).
- **Fix:** pass `image_authority_path(output_dir, ds, stem)`.
- **Test:** `test_cli_recompile.py::test_local_overlay_pass_refuses_a_record_claiming_a_missing_table`
  — failed first (`DID NOT RAISE`), passes after. Recompile files: `84 passed`;
  files that drive a local recompile: `113 passed`.
- **Commit hygiene:** `98633c2b4` also carries C14d's four staged files
  (`sdk_/_measurement_tables.py`, `sdk_/_run_state.py`,
  `tests/unit/sdk_/test_run_state.py`, `tests/unit/sdk_/test_tables_v2_state_readers.py`)
  through an index race between the two agents. The content is C14d's, verified by
  C14d; the commit was left unrewritten. Agents now commit with `git commit -- <paths>`.

## Mutation check

Swapping `certified_measurement_table` for a latest-folder version was caught by
two of the three new tests; the end-to-end one passes because the authority check
reads the record directly. Original restored and hash-checked.
