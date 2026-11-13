# Plan revision: DVC and embeddings (#281)

**Revised:** `docs/superpowers/plans/2026-11-12-dvc-embeddings-refactor/` (README + P0–P5),
against `plan-review.md` and the spec as of `bcc96d4ec`. Re-rendered to
`docs/superpowers/artifacts/2026-11-12-dvc-embeddings-refactor/plan.html`.
2026-11-12.

Every citation added was checked against the worktree with `sed`/`grep`. The spec
and the review were not edited.

---

## Where each finding landed

### Critical

| ID | Landed in | Notes |
|---|---|---|
| C1 | README Global Constraints; P3 intro; P3 T1 (`TABLE_RUN_ID_RE` `order` group, `TableRun.order`, `next_table_run_order`, `choose_table_run_id` without a journal input, `list_table_runs` refuses a tie); P3 T2 `test_a_second_pipeline_gets_folder_2`; P3 T4 `test_a_full_rerun_with_an_edited_pipeline_numbers_folders_1_and_2`; P3 T6 upstream test (`2-…` after an input's `1-…`) | User decision: store-local counter (spec D2). `current_application_identity` became `current_pipeline_sha256`. |
| C2 | P2 T2: decision table (live jobs wait first), "published" bound to the first-poll fence generation, process-mode row, migrate not served; new `aggregate_proof_matches_current_state` in `sdk_/_run_state.py`; tests for measure resubmission, stale proof (5), process failure (3), intent-only row | Spec C3 as written. |
| C3 | README Global Constraints and the **Expected-red GUI tests** table; README conventions (gates compare the red set with the table); P3 intro, T2 Step 3, T4 Step 4, T6 Step 5; P1 T4, P4 T4, P5 T3 gates; merge gate widened | User decision: let it fail. No xfail, no GUI test edits. |
| C4 | P1 intro; P1 T1 AST guard `tests/unit/sdk_/test_gui_sdk_imports.py` with a signature baseline generated from `main`; P1 T2 (new names beside old); P1 T3 (one-argument snapshot paths unchanged, new suffixed functions beside them, a test pinning the one-argument form) | |
| C5 | P1 intro; P1 T2 (no `error_analysis` change, narrowed `rglob` to `CSV_THE_CLI_MAY_LEAVE`); P1 T4 (sweep excludes `error_analysis.csv`, `measurements.csv`, `metadata.*`) | |
| C6 | P3 T5: `replace_embedded_measurement_table` writes v2; `tables_v2` step ordered after every table-writing step; records then run proof republished; glob at `_cli_migrate.py:1057-1061` replaced by the resolver; `test_a_completed_v1_tree_reads_complete_after_migrate` | Folder is `1-<sha>`, sha per D10 (Minor 9). |

### Important

| ID | Landed in | Option picked |
|---|---|---|
| I1 | P3 T2 Step 3 (exclude only when the write produces its run's tables) and `test_table_less_save_keeps_its_own_folder`; README Review Focus 2 | |
| I2 | P3 T1 `current_pipeline_sha256` and `choose_table_run_id` rule 2; P3 T2 `measured_image` fixture opens a CLI application; `test_a_programmatic_save_is_unknown_and_resaves_into_one_folder` | Spec D2/D6: no digest → `<n>-unknown`; an unknown write replaces the highest `-unknown`. |
| I3 | P2 T1 exit-site table (`:2528`, `:3434`, `:3477`, `:3497`, `:3712`, `:3729`, `:3759`, `:3916`), migrate (`_cli_migrate.py:2576`, `:2409-2410`), `_cli/CLAUDE.md:667`; Step 4 derives the test list by grep (31 files today, not 28); new tests for rerun-after-3, process local, recompile SLURM, no-work forward SLURM | A SLURM `--dry-run` migrate exits `4`: it submits the validation chain (`_cli/CLAUDE.md:660-663`). The review cited `:3492`; the exit is at `:3497`. |
| I4 | P2 T2 module-level `_completion`, `_live_jobs`, `_fence_generation`, `_published`, `_is_process_run` | |
| I5 | P2 T5 `test_restored_liveness_state_fails_safe`: restored active fence with a finished ledgered job, dead-pid owner record, guard lets a run proceed, state not `active` | |
| I6 | P2 T6 Step 1: one dataset, `hdf/` excluded, scratch repo on `/bigdata`, `--time=12:00:00`, extrapolate, delete afterwards; README Lane B | Bounded subset (spec C8). |
| I7 | P1 intro; P1 T2 (`mirror_csv` kept, `mirror_tsv` added, finalize writes both, `measurements_tsv` in the proof and in `_GUI_WRITTEN_PROOF_DESCRIPTORS`, `test_a_curation_does_not_stale_the_proof`); P1 T4 release note | |
| I8 | P0 T1 now lands the `_OLD_TYPED_*` names and the resolver trio with the constants | Resolvers in T1, rather than merging T1 and T2. |
| I9 | README expected-red table, P0 rows; P0 intro | Listed as known GUI-test breaks for #283 (spec §2.2), no legacy-inclusive export. |
| I10 | P0 T3: `LEGACY_PIPELINE_NAME` read fallback through one helper in every run-prefix reader; docs `CLAUDE.md:39`, `README.md:112`, `docs/how-it-works.md:72,294,339,639` | Read fallback, not "drain first". |
| I11 | P5 T4: lock-based CPU torch (`[[tool.uv.index]]` explicit + `[tool.uv.sources]` + `phenotypic[foundation]` + re-lock); `expected_embeddings` at `acr_gc/cli.py:199` and `acr_hpcc/launch.py:144`; pinned commit via `PHENOTYPIC_DINO_REVISION`; P4 T3 loader `revision=` | See spec problem 4. |
| I12 | P4 T1 (errors wrapped in `OperationFailedError`); P4 T3 tests assert `OperationFailedError` with `err_type`; license gate called in the measurer before loading, `RuntimeError` | Assert the wrapped error, so `MeasureEmbedding` keeps the `MeasureFeatures` contract. |
| I13 | P4 T3: `embed_crops` batched helper on `reshape_patch_tokens`, `resolved_revision` from `model.config._commit_hash`, `stub_backbone` calls `cache_clear()`, `extract_patch_features` kept for `masked_dense` | |
| I14 | P4 T4: split and README measurement section skip `MeasureEmbedding`; separate README section; `schema.json` resolves `embedding` through the embedding table | Column named `embedding` (spec E5). |
| I15 | P3 T2 `write_run_entry` as the single run-entry owner; `array_sha256: Mapping[str, str]` keyed by `input.series` in P3 T2 and P4 T2; P5 T2 uses it for an embeddings-only store | |
| I16 | P3 T4 file list (`phenotypicCLI.py:4000-4010`, `_cli_chunk_writer.py:322-329`, `_cli_recompile_slurm_scripts.py:532`, `_cli_recompile_worker.py:481-484,516,587`); grep extended to `parents\[`, `"tables"`, `"table\.parquet"`; migrate's glob and `replace_embedded_measurement_table` in P3 T5 | |
| I17 | P3 intro; P3 T4 `certified_table_run` and the selection rule; `test_measure_with_another_pipeline_adds_a_folder_and_recompile_keeps_it`; P4 T4 aggregates certified runs | |
| I18 | P3 T5 `first_v1_store` in `_cli/_cli_tables_v1.py`, read-only half; `datasets_needing_migration` untouched | |
| I19 | P5 T1 `process_meas_refusal`: `meas` refused unless the output is a store | |
| I20 | README Phases table (P4 T4 → P2 T3), Execution shared-file table (added `_run_state.py`, `_cli_readme_generator.py`, `_cli_migrate.py`, `_cli/CLAUDE.md`, test helpers; corrected phase lists) | |
| I21 | P3 T1 Step 1: `tests/fixtures/tables_v1/_generate.py` in the `legacy_hdf` style; `v1_run/` from the CLI at that commit, `v1_store_null_pipeline` edited from it; self-check that a moved copy reads `complete` | `old_tree_with_measurements_csv` is a TSV-cutover fixture, not a v1 one, so P1 T4 builds it at test time from a fresh run. |

### Minor

| # | Landed in |
|---|---|
| 1 | P0 T2: `_seed_pipeline_config(tiny_pipeline_file, tmp_path)` (signature `(pipeline_path, output_dir)`, `phenotypicCLI.py:4634`); writer `_persist_pipeline_to_output_dir` (`_cli_output_manager.py:610`) |
| 2 | P2 T3 test uses `UNITS.PX2` |
| 3 | P2 T3 Step 3: counts from `work_ids`, the terminal journal and the source set; no deep `resolve_run_state` |
| 4 | P1 T3: `_metadata_suffix_callback` on the `--metadata` option (`phenotypicCLI.py:2050-2060`) |
| 5 | P4 T2 Step 3: never through `_write_validated_frame`; `FixedSizeListArray.from_arrays` + explicit schema |
| 6 | P3 T2 Files: validators accept the `measurements/` document plus the ngio attributes |
| 7 | P4 T1: pipeline validation refuses two embedders with one payload key; test |
| 8 | P4 T4: `aggregate_embedding_tables` streams with `pq.ParquetWriter`, one row group per store |
| 9 | P3 T5 `v1_table_pipeline_sha`: most recent figure run's sha first; test |
| 10 | P3 T1 `latest_measurement_run`; P3 T4 test with a grid-only folder |
| 11 | P5 T1: `_cli/CLAUDE.md:123` bullet updated; the check also runs in `process_single_apply_only_core` |
| 12 | P5 T5 Step 4: measure bytes on `intel` vs `epyc`, record the limit if they differ (measure, not state) |
| 13 | README exit-code table: `1` runtime/config error, `2` Click usage refusal |
| 14 | P4 T3: `crop_px` choices `112, 224, 336, 448`; test |
| 15 | P2 T5: `config.local` overlays `config`; global/system ignored and said so |
| 16 | P2 T2 Step 3 and P2 T6 item 2 |
| 17 | P1 T2 Step 4: surface derived by grep (21 files match today) |
| 18 | P4 T1 cites `_image_pipeline_core.py:1139` |
| 19 | P4 T1 and P5 T1: both are properties |
| 20 | P3 T2 `objmap_digest`: dtype + shape + C-order bytes of the array as stored |
| 21 | P0 intro: noted as a harmless #283 item |

### New: spec E7 (in place)

| Item | Landed in |
|---|---|
| Task | P5 T3 "In place", between process stores (T2) and the AutoConvertRaw-GC rollout (now T4); pilot is T5 |
| Equality-only overlap relaxation | P5 T3 Files (`phenotypicCLI.py:2289-2306`); `test_strict_containment_stays_refused` |
| Read-only refusals | `in_place_refusal`, seven checks in order; parametrized `test_in_place_refusals_write_nothing` (one `Error:` line, tree unchanged) and `test_a_raw_input_is_refused` |
| Table-only transaction | `process_in_place_core` over `_rewrite_store_tables(expected_root_sha256=…, consolidate=True)`; `test_pixel_arrays_are_never_rewritten` (level-0 inodes and bytes) |
| Pre-swap root check | `StoreRootChangedError`, rebuild once, then fail; `test_root_changed_before_swap_rebuilds_once_then_fails` |
| Level-0 digest after `apply()` | `level0_digest`; `test_a_pixel_changing_op_the_class_check_misses_fails_the_image` |
| Continuation | `in_place_input_digest` excludes root, `tables/`, `figures/`; config digest gains `in_place: true`; `test_rerunning_does_no_work` |
| Independent reader | `test_an_independent_reader_sees_the_new_folder` (zarr consolidated + pyarrow) |
| ACR folder unchanged | `test_in_place_adds_folder_2_and_leaves_folder_1_byte_identical` |
| How-to notes | P5 T3 docs (`zarr_storage.md`, CLAUDE.md files); P2 T6 item 4 (not a DVC stage) |
| Review Focus | README item 5 replaces the grid-cell-outside-image item, the weakest of the five (a plain validation error with a direct test) |
| Execution | README clusters C20 (P5 T1–T2, light), C21 (P5 T3, light, then the P5 deep gate and the full regression), C22 rollout, C23 pilot |

---

## Expected-red GUI tests

Verified with `grep`/`ls` against the worktree, and by reading the GUI code each
test reaches.

| Phase | Test | Cause |
|---|---|---|
| P0 | `tests/unit/gui/tune/test_validation.py::test_tuning_spec_path_is_deployable` (:98-99) | `_gui/tune/_validation.py:61` |
| P0 | `tests/unit/gui/tune/test_setup_authoring.py::test_write_authored_setup_spec_preserves_existing_spec_search_space` (:147) | `_gui/tune/_setup_authoring.py:356` loads `existing.json.pht-tune` as a pipeline |
| P0 | `tests/unit/gui/tune/test_setup_authoring.py::test_existing_spec_replaces_scorer_only_when_requested` (:190) | same |
| P3 | `tests/unit/gui/results_viewer/test_measurement_routes.py` (fixture `:100`, unlink `:205`) | `_gui/results_viewer/_measurement_source.py:107-108` stats the v1 path |
| P3 | `tests/unit/gui/results_viewer/test_measurement_source.py` (fixture `:88`, `:249`) | same |
| P3 | `tests/unit/gui/results_viewer/test_measurement_join_migration_run.py` (`:105`, `:209`) | same |

The review listed `test_setup_authoring.py:60` as red. It is not: the GUI writes
the literal `.setup.json.pht-tune` (`_setup_authoring.py:535`), so the assert
holds. The two tests above, which load an `existing.json.pht-tune`, are the real
breaks.

Checked and expected green: the GUI SLURM tests (they mock the return code), the
zarr-route tests that hand-write a v1 table and expect `404`, and the GUI tests
that read `measurements.csv` (B9 keeps it; P1 T2 updates `tests/_output_layout.py`
so `publish_complete_run_over_outputs` still finds every proof output).

---

## Spec problems found (not edited)

1. **Stale citation.** §3.2 cites `_cli_process_single.py:462` for
   `_republish_table_marker`. It is defined at `_cli_recompile_tables.py:51` and
   called at `_cli_process_single.py:567-581`. The plan cites the real lines.
2. **B6 vs B9.** B6 removes the stale generated `.csv` sibling of every B2 file;
   `measurements.csv` is the CSV sibling of a B2 file, but B9 keeps finalize
   writing it. The plan reads B9 as the rule for the mirror, so the sweep skips
   it. Worth one clarifying clause in B6.
3. **C4 `pipeline_sha256` source.** C4 says metrics come from the run state, but
   after `--mode measure` the run config still names the full run's pipeline —
   the D6a problem. The plan takes the sha every aggregated run folder shares
   (P3 T4), `null` when they differ. The user should confirm.
4. **E3/E6 revision.** E3 keys the backbone cache by revision and D3 records a
   resolved revision, but E3's field table has no revision, and E6 needs a
   pinned commit offline. The plan adds a `PHENOTYPIC_DINO_REVISION` environment
   variable (P4 T3, P5 T4). That is a new user-facing knob the spec does not
   name; the user should confirm it, or choose a field instead.
5. **§2.2 known-break list is incomplete.** With plain `.json` still a legacy
   pipeline suffix, the GUI's `matches_any_suffix(path, PIPELINE_CONFIG_SUFFIXES)`
   (`_gui/builder/_directory_browser.py:32`, `_gui/shell/_classifier.py:222`)
   lists a new `x.pht-tune.json` as a pipeline. No test catches it. The plan
   records it under P0 for #283.
6. **E7 "table-only".** The plan writes the in-place run's figure run in the same
   transaction, as the measure rewrite already does. If "table-only" was meant to
   exclude figures, P5 T3's plan should drop `figures`.
7. **D10 with several figure runs.** "The figures descriptor's run sha" is
   ambiguous when a store has more than one figure run. The plan takes the most
   recent by run id date.

---

## Not fully resolved

- `v1_run/` size: the 5 MB bound in P3 T1 is a target; the generator shows the
  real size.
- `tiny_inputs_one_bad` relies on a truncated TIFF being recorded as a terminal
  per-image failure in full mode. P2 T1 tells the implementer to confirm it in
  Step 2; process mode records it (`_cli_process_only.py:396-398`).
- `zarr.open_consolidated(...).members(max_depth=None)` in P5 T3's reader test
  was not run against the installed zarr; adjust to its API if it differs.
- `tables_for(image)` names "the tables the forward worker builds" without a
  function name; the implementer picks the existing builder.
