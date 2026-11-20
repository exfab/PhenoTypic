# C14b gate: P3 Task 4, per-image record and resume sites

Classified by the C14b implementer; written here by the orchestrator. The full
per-test list (test, origin, tag, signature, first line) is
`/bigdata/exfab/anguy344/gate-results/c14b/classified.tsv`.

- **Job:** `32095306`, 24 shards on `short`, in a detached worktree at
  `0ff0d29f7` (dirty=0 at start and end; `phenotypic` imported from the gate
  tree). The afterany cleanup job `32095330` removed the worktree.
- **Totals:** 10,267 tests, 348 reds (C12: 370).
- **Commits:** `c6595773b` (sdk_ resolver: `journal_table_run`,
  `journal_measurement_table`, `owned_table_run_id`, `journal_pipeline_sha256`),
  `eef79f351` (full-mode records, `image_measurement_artifact`), `0ff0d29f7`
  (staged records, resume probes, tests' own v1 helpers), `07b78e58a` (two
  post-gate test files).

## Classification

- **Green now (27 C12 reds):** test_staged_gpu_local ×10, test_cli_reference_e2e ×4,
  test_staged_stage1_reads_like_single_pass ×3 (SLURM variants),
  test_staged_store_stages ×2, test_staged_nested_equivalence ×2, and one each in
  test_cli_provenance_original, test_lifecycle_publication_races,
  test_cli_reference_wiring,
  test_directory_scanner::test_process_single_cli_marker_certifies_the_store,
  test_embedded_measurement_checkpoint. (test_cli_migrate_image went green from
  `af122f26e`.)
- **Green after the gate (6), `07b78e58a`:** test_embedded_measurement_publication ×3,
  test_calibration_figure_in_store ×3 — their own assertions hand-joined the v1
  path. Green at that commit in a clean detached worktree.
- **Blocked on C14a finalize (252: 247 from C12, 5 new):** images complete, then
  the run dies with `not an OME-Zarr store directory: …/tables`
  (`_cli_finalize_fanout.py:523` depth arithmetic). Largest: test_cli_migrate_mode 48,
  test_run_summaries 18, test_embedded_measurement_migration 16,
  test_cli_recompile_slurm 16, test_cli_exit_codes 14, test_cli_wait 13,
  test_migrate_end_to_end 13. 13 name no cause in junit; 11 reproduced at
  `07b78e58a` (CLI exits 1), two end at the finalize message.
- **Still another cluster's (90):** C14a's own v1 readers 55 (test_finalize_run 35,
  test_finalize_fanout 19, test_completion_split 1); C14d 7 (test_run_state); C14c 6
  (test_cli_recompile_slurm 3, test_recompile_no_store_writes 2,
  test_embedded_measurement_recompile 1); C13 4 (test_embedded_measurement_replacement 3,
  test_directory_scanner::test_measure_core_embeds… 1); GUI expected-red 18, matching
  the README's P0/P1/P3 rows.
- **New red (5), all test_cli_exit_codes, not regressions:**
  test_rerunning_a_run_that_exited_3_still_exits_3,
  test_rerunning_with_retry_failures_retries_and_still_exits_3,
  test_a_local_recompile_of_a_run_with_a_failure_exits_3,
  test_a_slurm_recompile_run_locally_reports_the_runs_failures[no-measurement-tasks],
  test_a_local_staged_publication_failure_still_exits_1. Masked green at C12 (they
  expect exit 3 or `STAGED FINALIZATION FAILED`, which an all-images-fail run also
  gives); now they reach the finalize error. Tagged blocked on C14a.
- **Staged-resume files C14b moved onto `plant_measurement_table`:**
  test_staged_resume_equivalence, test_staged_resume_parity, test_staged_resume —
  all green.

## Verification

In a clean worktree at `07b78e58a` (main venv, `PYTHONPATH` = that tree):
`pytest -q --no-header -p no:randomly -o addopts= -m "not slow" -n 8` over
test_embedded_measurement_publication, test_calibration_figure_in_store,
test_tables_v2_readers, test_table_runs and the three staged-resume files →
`1595 passed in 180.08s (0:03:00)`. mypy on the 8 touched src modules →
`Success: no issues found in 8 source files`.

## Decisions and deviations

- The C14b tests pin the per-image record (valid, names the run folder's parquet),
  not the exit code, because finalize is C14a's.
- How the path is found: the save writes its journal into the root last and picks
  its folder from that journal's pipeline sha, so reading the sha back from the
  promoted root names the folder the save wrote. Measure mode adds no journal
  application, so C13's rewrite must return its own `TableRun`.
- A plain continuation refuses a changed pipeline
  (`_cli_state_management.py:473`). Orchestrator decision: plan drift, not a fork;
  the plan's edited-pipeline test uses `--restart`. Continuation semantics are
  unchanged.
- Dispatch order changed to C14b → C14a → C13 → C14c → C14d, because C13's gate
  would hit the same finalize wall.
