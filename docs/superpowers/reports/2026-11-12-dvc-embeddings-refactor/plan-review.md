# Plan review: DVC and embeddings (#281)

**Reviewed:** `docs/superpowers/plans/2026-11-12-dvc-embeddings-refactor/` (README + P0–P5) against
`docs/superpowers/specs/2026-11-12-dvc-embeddings-refactor/design.md`, on worktree
`worktree-dvc-embeddings-refactor` (plan commit `d34afb069`, base `main` @ `3248a9173`), plus
AutoConvertRaw-GC @ `33046cb`.
**Reviewer:** plan-feasibility review, analysis only. 2026-11-12.

**Verdict: NEEDS REVISION.** The phase structure is sound, and most citations resolve. But six
findings would make a task fail or produce wrong behaviour:
- the `<seq>` rule breaks on an ordinary forward re-run;
- `phenotypic-wait` can return early or return the wrong code;
- three changes contradict the "no `_gui/` change, GUI tests green" constraint;
- the v1→v2 migration silently un-completes every converted tree.

Severity counts: **6 Critical, 21 Important, 21 Minor.**

---

## Critical

### C1. `<seq>` from the journal is not monotonic across forward re-runs, so two folders share one sequence

- **Plan:** P3 T1 `choose_table_run_id` rule 2 (`table_run_id(journal_sequence, sha)`), with
  `list_table_runs` "ascending `sequence`" and `latest_table_run`. P3 T2 Step 3.
- **Evidence:**
  - `_core/_provenance.py:374`: an application's `sequence` is `len(applications) + 1`, so it is
    local to one image's journal.
  - A forward full or single-pass worker builds that journal from the **input** file, not from the
    existing store: `_cli_process_single.py:300-321` (`initialize_cli_provenance` on the freshly
    read input). A raw TIFF input starts with `applications == []`, so every forward application
    over a raw input is `sequence == 1`.
  - The rewritten store's root then carries the new journal (`_image_io_handler.py:1477-1507`).
    The carried run folders refer to sequences of a journal that no longer exists.
  - Concrete failure:
    1. run full A → `tables/1-<A>`;
    2. edit the pipeline to A′ and re-run the same command (the work id changes, so every image is
       reprocessed);
    3. the store now holds `1-<A>` (carried) and `1-<A′>`.
    `latest_table_run` has a tie. The GUI-facing readers P3 T4 makes "latest by default" can then
    return the stale pipeline's table.
  - Staged runs are fine: Stage 1 starts one application and Stage 3 continues it
    (`_cli_staged_workers.py:100-108,385,423`; `_cli_staged_strategy.py:542-567`).
  - `resume_provenance_application` (`_provenance.py:840-925`) reuses the checkpoint's sequence and
    never appends one.
  - `--mode measure` appends none (`phenotypicCLI.py:3526-3528`, "measure mode, which never
    mutates processing state"), and the measure worker only loads and measures.
  - So the plan's rule 3 premise is right, but rule 2's is not.
- **Fix to the plan:**
  - When no existing run has this sha, rule 2 becomes
    `max(journal_sequence, max existing run sequence + 1)`.
  - Better: drop the journal from the folder name and use `max(existing) + 1` for every new
    folder. D2 leaves the staged/measure mapping to the builder, so record the deviation for the
    user.
  - Add a P3 T1 test: a full run with A, then a full run with A′ into the same `--output`, gives
    sequences 1 and 2.
  - State that `list_table_runs` can never tie.

### C2. `phenotypic-wait`'s decision table returns early, and returns the wrong code

- **Plan:** P2 T2, the decision table, `decide_wait_outcome`, and
  `test_failed_but_published_is_3`.
- **Evidence, part (a): "complete outranks live" is wrong for the waiter.**
  - The table returns `COMPLETE` for `complete` "any live jobs". `resolve_run_state` orders
    complete > active only because "a live worker … belongs to a new invocation that has already
    changed the inventory" (`sdk_/_run_state.py:1760-1766`).
  - That is false for `--mode measure` and same-metadata `--mode recompile`. Measure mode never
    touches `processing_state.json` (`phenotypicCLI.py:3526-3528`). So the previous full run's
    proof still matches `inventory_digest`, `scientific_config_digest` and
    `finalization_input_digest` (`_run_state.py:1408-1444`).
  - Consequence: `python -m phenotypic --mode measure --slurm …; phenotypic-wait` exits `0` on
    its first poll while the measure jobs are queued, and DVC then hashes outs that are about to
    be rewritten.
- **Evidence, part (b): "failed + aggregate proof current" is not bound to this submission.**
  - `aggregate_proof_refusal` (`_run_state.py:1255-1282`) checks only that the listed outputs'
    bytes still match. It never compares the proof's `inventory_digest`,
    `scientific_config_digest`, `source_set_digest` or any generation with current state.
  - The run proof is published only when every image succeeded
    (`_cli_completion.py:1221-1252` raises otherwise). So for a run with failures, the aggregate
    proof is the only evidence, and a proof left by an **earlier** finalization still passes.
  - Concrete failure:
    1. a run completes;
    2. new images are added and the run is resubmitted;
    3. one image fails and the finalizer is OOM-killed;
    4. the waiter exits `3`, "published", though nothing from this submission was published.
  - Process mode never publishes an aggregate proof
    (`_cli_completion.py:1254-1258`, `aggregate = None if process_only_layer`). So **every**
    process SLURM run with one terminal failure exits `5`.
- **Fix to the plan:**
  - Any `live`/`unverified`/`unresolved` job means keep waiting, **before** the `complete` row.
  - Bind "published" to the current lifecycle generation. Read the fence's generation at the
    first poll, and require either:
    - that generation's fence closed by its finalizer; or
    - a run proof whose `execution_epoch` equals that generation; or
    - an aggregate proof whose inventory, config and source-set digests match current state, via a
      new `sdk_` helper that reuses `_source_set_binding`.
  - Give process mode its own row: `failed`, no live job → `3`.
  - Add tests:
    - a measure resubmission over a completed run keeps waiting while a job is live;
    - a stale aggregate proof with a dead finalizer gives `5`;
    - a process run with one failure gives `3`.

### C3. P3 turns GUI tests red, and the plan says they stay green

- **Plan:** README Global Constraints ("The existing GUI tests must stay green"). P3 T4 Step 4
  ("must stay green except the one #283 item").
- **Evidence:**
  - `_gui/results_viewer/_measurement_source.py:107-108` stats
    `store / MEASUREMENT_TABLE_RELATIVE_PATH` (the v1 path). After P3 T2 a new store has no file
    there, so `_store_and_identity` returns `None` and the Colony picker offers nothing.
  - These tests write stores through the forward writer and then touch the v1 path:
    - `tests/unit/gui/results_viewer/test_measurement_routes.py:100` writes with
      `save2zarr(measurement_table=…)`, and `:205` calls
      `(… / MEASUREMENT_TABLE_RELATIVE_PATH).unlink()`, which raises `FileNotFoundError`;
    - `test_measurement_source.py:88,249`;
    - `test_measurement_join_migration_run.py:105,209`.
  - The spec's §2.2 admits "one direct table-path join in the results viewer", but not these
    failures.
- **Fix to the plan:** this needs a user decision.
  - Option 1: list these tests in P3 T4 as expected failures, `xfail(strict=True, reason="#283")`.
    That edits tests, not `src/phenotypic/_gui/`.
  - Option 2: make #283's table-path fix a second merge-gate item and run P3's gate against it.
  - Either way, replace "except the one #283 item" with the actual list.

### C4. P1 renames or re-signs `sdk_` functions the GUI imports

- **Plan:** P1 T2 (`:1466 measurements_csv_path → measurements_tsv_path`). P1 T3 ("The snapshot
  path functions take the suffix").
- **Evidence:**
  - `_gui/results_viewer/_filtered_state.py:44` imports `measurements_csv_path` from
    `phenotypic.sdk_`. A rename is an `ImportError` when the results viewer is imported.
  - `_gui/run_console/_callbacks.py:326` calls `metadata_csv_deliverable_path(output_dir)` with
    one argument. Adding a required `suffix` parameter is a `TypeError`.
  - The README constraint protects only `MEASUREMENTS_CSV` and `MEASUREMENT_TABLE_RELATIVE_PATH`
    by name.
- **Fix to the plan:**
  - Add the new functions beside the old ones, and keep `measurements_csv_path` and
    `metadata_csv_deliverable_path(output_dir)` unchanged as legacy names.
  - Add a P1 T1 guard test: an AST walk of `src/phenotypic/_gui/**` collects every name imported
    from `phenotypic.sdk_` and asserts each still resolves, with an unchanged signature for the
    functions it calls.

### C5. `error_analysis.tsv` cannot be produced without changing `_gui/`, and the stale-CSV sweep would break the GUI publisher

- **Plan:** P1 T2 (`:927 ERROR_ANALYSIS_CSV → ERROR_ANALYSIS_TSV`, `:2514`). P1 T4 (the finalizer
  "unlinks the stale generated `.csv` siblings in the B2 list"). P1 T2's test
  `assert not list(d.rglob("*.csv"))`.
- **Evidence:**
  - The CLI's error-analysis deliverable is written by GUI code.
    `_cli/_cli_error_outputs.py:81-96` imports `publish_error_analysis` from
    `_gui/results_viewer/_error_tab/_publication.py`.
  - That module names the file itself (`computation_path_name("csv")` → `error_analysis.csv`,
    `:849-851`) and writes it with `combined.write_csv` (`:588`). It never reads
    `ERROR_ANALYSIS_CSV`.
  - Renaming the constant therefore changes nothing. Deleting `error_analysis.csv` after publish
    would remove a file that the publisher's generation record lists.
- **Fix to the plan:**
  - Drop `error_analysis.*` from P1. Record it for #283 as a spec B2 deviation for the user.
  - Exclude it from the stale-CSV sweep.
  - Narrow the `rglob("*.csv")` assertion to the files the CLI itself writes.

### C6. The v1→v2 conversion un-completes every converted tree, and migrate keeps writing v1

- **Plan:** P3 T5 `convert_tables_v1_to_v2` and the `tables_v2` store-array step.
- **Evidence, part 1: records.**
  - A per-image record fingerprints a store by its root `zarr.json`
    (`_run_state.py:517-539`). The conversion rewrites that root "root-last", so every record goes
    stale.
  - Then `resolve_run_state` reads `incomplete`, because rule 1 needs every image `verified`
    (`_run_state.py:1405`). The next forward run sends every image back to work.
  - Measure mode solves the same problem with `_republish_table_marker`
    (`_cli_process_single.py:462`). P3 T5 never mentions records or the run proof.
- **Evidence, part 2: migrate still writes v1.**
  - `replace_embedded_measurement_table` (`sdk_/_measurement_tables.py:807-868`) writes the flat
    v1 path. It is called from `_cli_migrate_image.py:298` and `_cli_migrate.py:1394`, and P3
    does not mention it.
  - If it runs after the conversion, the store gets a mixed layout.
  - `_cli_migrate.py:1057-1061` globs `*/zarr/*.ome.zarr/tables/measurements/table.parquet`. On
    v2 that glob is always empty, so the "aggregate rebuild produced no measurements" guard
    silently stops firing.
- **Fix to the plan:**
  - Repoint `replace_embedded_measurement_table` to write v2 directly. That makes a separate
    conversion pass unnecessary for HDF-era trees.
  - Order `tables_v2` after every step that can still write v1.
  - Add "republish each converted image's record, then the run proof" as an explicit step.
  - Add a test: a completed v1 tree reads `complete` after `--mode migrate`.
  - Replace the glob with the resolver.

---

## Important

### I1. P3 T2: carry `exclude=run_id` unconditionally drops the run's own folder on table-less saves

- **Plan:** P3 T2 Step 3, "calls `carry_table_runs(final, part, exclude=run_id)`".
- **Evidence:**
  - Several writes carry no tables:
    - the initial-checkpoint saves (`_cli_process_single.py:327`, `_cli_staged_workers.py:370`);
    - Stage 1 (`_cli_staged_workers.py:424`).
  - With exclusion always on, each of them drops the same-pipeline folder, embeddings included,
    before the image's work starts. An image that then fails terminally loses that pipeline's
    previous tables for good.
  - Figures already get this right: `exclude=figures.run.run_id if figures is not None else None`
    (`_image_io_handler.py:1468-1470`).
- **Fix:** exclude only when this write produces its own run's tables. Pin it with a test.

### I2. P3 T1/T2: run ids for empty and programmatic journals are undefined

- **Evidence:**
  - `Image(arr=…).save2zarr(measurement_table=…)` has `applications == []`
    (`_provenance.py:122-131`). `current_application_identity` has no sequence to give, and seq 0
    fails `TABLE_RUN_ID_RE`. This exact call is common in existing tests, GUI tests included.
  - Programmatic applications record `pipeline=None` (`_provenance.py:660-666`), giving
    `<seq>-unknown`. D6's sha match can never reuse such a folder, so notebook re-saves pile up
    folders.
  - P3 T2's `test_save_writes_run_folder_and_v2_descriptor` expects `1-{SHA[:12]}` from a plain
    `measured_image.save2zarr`. Nothing provides a sha there unless the fixture opens a CLI
    application.
- **Fix:**
  - Define both cases. For example: an empty journal is `1-unknown`; a write with an unknown sha
    replaces the latest `-unknown` folder.
  - Make the test's fixture call `initialize_cli_provenance`.

### I3. P2 T1: exit sites and the test surface are under-counted

- **Missed exit sites:**
  - `phenotypicCLI.py:2528`: one `sys.exit(0)` after both local and SLURM recompile. It must
    branch.
  - `:3477`: a forward invocation with nothing left to do submits a SLURM recompile and exits `0`.
    It should exit `4`.
  - `:3434` and `:3492`: "All images already processed!" exits `0` even when `:3489` reports
    skipped recorded failures. Re-running a run that exited `3` must not report `0`.
- **Under-counted tests:** P2 T1 Step 4 names two test files. In fact **28 files** under
  `tests/unit/cli` and `tests/integration` mention SLURM and assert `exit_code == 0`, among them:
  - `test_cli_recompile_slurm.py`
  - `test_cli_migrate_slurm.py`
  - `test_cli_provenance_migration_slurm.py`
  - `test_staged_stage1_reads_like_single_pass.py`
  - `test_cli_reference_e2e.py`
  - `test_cli_image_manifest.py`
  - `test_cli_store_options.py`
  - `test_process_only_e2e.py`
- **Process-only local exit:** the plan does not say what `:3742-3760` becomes. Per C10 it should
  be `3`, not `1`.
- **Migrate SLURM:** `handle_migrate_mode` returns `0` for a submitted attempt
  (`_cli_migrate.py:2576`), and for a submitted dry run (`:2409-2410`). The plan must say whether
  a SLURM `--dry-run` submission is `4`.
- **`phenotypic-wait` scope:** migrate keeps its lifecycle under an external control root, so
  `live_slurm_job_ids(output_dir)` sees nothing. The waiter cannot serve migrate. Say so.
- **Fix:** list every site, and derive the test list by searching the SLURM tests for
  `exit_code == 0`. Update the `_cli/CLAUDE.md` sentence that says a submitted migration exits 0.

### I4. P2 T2: the tests monkeypatch names that the implementation imports inside functions

- **Evidence:**
  - Step 3 imports `live_slurm_job_ids` inside the function.
  - The tests call `monkeypatch.setattr(_cli_wait, "live_slurm_job_ids", …)`. With the default
    `raising=True` that raises `AttributeError`, because the module attribute does not exist.
- **Fix:** add module-level `_completion(d)` and `_live_jobs(d)` wrappers that import lazily, and
  patch those.

### I5. P2 T5: the C13 test passes without exercising C13

- **Evidence:**
  - `two_revision_run` is two local runs, so there is no lifecycle fence.
  - `assert not live_slurm_job_ids(out)` is then trivially true.
  - C13's risk is a restored `active: true` fence or a `gui_launch_owner.json` from another
    revision.
- **Fix:**
  - Restore a `.phenotypic/` with an active fence whose ledgered jobs the fake scheduler reports as
    finished, and with a dead-pid owner record.
  - Assert the guard allows the run, and that `resolve_run_state` is not `active`.

### I6. P2 T6 Step 1: the C8 measurement is a ~1.35 TB copy

- **Evidence:**
  - The migrated maresca copy is 1,354 GB, and its `results/` includes HDF hardlinks (memory
    `maresca-ome-zarr-migrated-copy`).
  - Adding `out/results` to DVC with a copy cache duplicates all of that into the DVC cache. A
    hardlinked working copy saves nothing on the cache side.
  - The hardlinked scratch repo must also sit on `/bigdata`, the same filesystem.
- **Fix:**
  - Measure on a bounded subset, for example one dataset with `hdf/` excluded, and extrapolate.
  - Or state the quota and walltime explicitly and get the user's go-ahead.
  - Give the job a `--time`.

### I7. P1: the GUI's mirror writer and the proof's GUI-written set

- **Evidence:**
  - GUI curation rewrites `BundleLayout.mirror_csv` (`_io_constants.py:2985-2987`) on every save
    (`_gui/results_viewer/_curation_labels.py:839-850`).
  - If P1 points `mirror_csv` at `.tsv`, the GUI writes comma CSV into `measurements.tsv`.
  - If it does not, each curation recreates `measurements.csv` beside a `measurements.tsv` that is
    now stale, and P1 T4's sweep deletes it on the next finalize.
  - The proof key rename `measurements_csv → measurements_tsv` (`_cli_completion.py:1133`) takes
    the mirror text file out of `_GUI_WRITTEN_PROOF_DESCRIPTORS`
    (`_run_state.py:1236-1238`), so it becomes enforced by default.
- **Fix:**
  - Keep `mirror_csv` pointing at the legacy `.csv` (the GUI's target until #283), and add
    `mirror_tsv`.
  - Add `measurements_tsv` to `_GUI_WRITTEN_PROOF_DESCRIPTORS` now, so #283 does not break
    `core_readable`.
  - Document that until #283, curated edits reach the `.parquet` and the `.csv` but not the
    `.tsv`.

### I8. P0 T1 leaves the tree red until T2

- **Evidence:**
  - `PIPELINE_JSON = f"pipeline{CONFIG_SUFFIX_PIPELINE}"` (`_io_constants.py:446`), and the same
    holds for `BEST_PIPELINE_JSON` and `TUNING_SPEC_JSON`. So T1 flips every canonical artifact
    name immediately.
  - The resolvers (`:1496-1510`) know only canonical and plain `.json`. After T1, an existing
    `deliverables/pipeline.json.pht-pipe` is unreadable.
  - T1's gate runs only `test_io_constants.py`.
- **Fix:** add the `_OLD_TYPED_*` names and the resolver trio to T1, or merge T1 and T2 into one
  commit.

### I9. P0 changes GUI behaviour and breaks GUI tests

- **Evidence:**
  - `_gui/tune/_validation.py:61` accepts a Deploy spec only if it ends in `CONFIG_SUFFIX_TUNING`.
    After the flip, a legacy `x.json.pht-tune` spec is refused.
  - `tests/unit/gui/tune/test_validation.py:99` therefore fails.
  - `test_setup_authoring.py:60` asserts that a written name ends with `.json.pht-tune`.
- **Fix:**
  - Enumerate these as known GUI-test changes for the user, and route them to #283 or xfail them.
  - Or keep the GUI's behaviour by exporting a legacy-inclusive `TUNING_TYPED_SUFFIXES` that #283
    adopts.

### I10. P0 T3: `PIPELINE_NAME` is an object-storage key, not just a local name

- **Evidence:** AutoConvertRaw-GC reads `prefix + PIPELINE_NAME` from existing run prefixes:
  - `src/acr_core/worker.py:250,273`
  - `runs.py:38`
  - `preflight.py:20`
  - `acr_hpcc/cli.py:37`

  After the rename, runs planned before the rollout can neither execute nor be listed.
- **Fix:**
  - Add a read fallback to `pipeline.json.pht-pipe` in the worker and `runs.py`, or require
    in-flight runs to drain first.
  - Add the docs the plan omits: `CLAUDE.md:39`, `README.md:112`, `docs/how-it-works.md:639`.

### I11. P5 T3: the Dockerfile and run-record steps do not match AutoConvertRaw-GC

- **Evidence:**
  - The image installs with a frozen `uv sync` from the lock file (`Dockerfile:30-33`). A
    `--index-url` flag cannot select CPU torch there. The lock itself must carry it:
    - a `[[tool.uv.index]]` for the PyTorch CPU wheels, marked explicit;
    - a `[tool.uv.sources]` entry that points `torch` at it;
    - `phenotypic[foundation]` in `pyproject.toml`;
    - then a re-lock.
  - `run.json` is built in `acr_gc/cli.py:199` and `acr_hpcc/launch.py:144`, not in `plan.py`.
  - A pinned revision under `HF_HUB_OFFLINE=1` cannot reach the model:
    `load_dino_backbone` calls `from_pretrained(dino_id)` with no `revision`
    (`_dino_support.py:240-243`), so offline it resolves `refs/main`.
- **Fix:**
  - Rewrite the Dockerfile step around the lock.
  - Put `expected_embeddings` where `expected_bindings` is written.
  - Give the loader a `revision=` parameter, from P4 T3, that records the resolved commit.

### I12. P4 T3: the error tests cannot pass as written

- **Evidence:**
  - `MeasureFeatures.measure` wraps every exception in `OperationFailedError`
    (`abc_/_measure_features.py:523-546`). So
    `pytest.raises(GridCellOutsideImageError)` fails.
  - The DINOv3 license gate is inside `load_dino_backbone` (`_dino_support.py:228-234`), so a
    stub that replaces that function also bypasses the gate.
  - The gate raises `RuntimeError` (`_helper/_checkpoint_manager.py:818`). There is no
    `LicenseNotAcceptedError`.
- **Fix:**
  - Either let `MeasureEmbedding.measure` re-raise `GridCellOutsideImageError` unwrapped (it
    overrides `measure`), or assert `OperationFailedError` with `err_type`.
  - Call `require_license_acceptance` in the measurer before loading, or stub one level lower.

### I13. P4 T3: the existing DINO helpers do not cover what the plan reuses them for

- **Evidence:**
  - `extract_patch_features` (`_dino_support.py:286-321`) runs one unbatched image with native
    processor kwargs and discards CLS.
  - So it cannot serve `crop_cls`, `batch_size`, or resized `crop_px` crops.
  - The loader exposes no revision, but D3 requires `model.revision` and the cache key is
    `(hf_id, revision, device)`.
- **Fix:**
  - Specify a new batched forward helper that reuses `reshape_patch_tokens` (`:248`), and a
    resolved-revision source such as `model.config._commit_hash`, or `huggingface_hub`.
  - Have the `stub_backbone` fixture call `cache_clear()` on the `lru_cache`.

### I14. P4: `_measurement_infoclass = DINO_EMBEDDING` makes the embedder a mirror producer

- **Evidence:**
  - `_cli_output_manager.py:531-545` builds a by-feature split for each `meas` entry with an
    infoclass.
  - `_cli_readme_generator.py:201-210` documents each `MeasureFeatures` entry's columns as
    mirror columns.
  - Both would advertise a `<Prefix>_Embedding` header that is in no TSV, which contradicts the
    anti-goal. The column is named `embedding`, unprefixed, so `_member_for_column("embedding")`
    is `None` in `schema.json`.
- **Fix:**
  - Skip `MeasureEmbedding` in the split and in the README's measurement section, and document the
    embedding deliverable separately.
  - Decide whether the column is `embedding` or the enum's header, and make `schema.json` resolve
    it.

### I15. Cross-task interface gaps between P3, P4 and P5

- **Evidence:**
  - `refresh_embedding_staleness(…, array_sha256: Mapping[str, str])` (P3 T2) does not match
    `write_embedding_tables(…, array_sha256: str)` (P4 T2).
  - No task owns the run entry for a store with embeddings but **no** measurements (process mode,
    P5 T2). `write_image_tables` requires a `PreparedImageTables` and an `objmap_target`
    (`_measurement_tables.py:406-430`), and a process store has neither (`write_objmap=False`,
    `_cli_process_only.py:223-247`).
- **Fix:**
  - Use one type, a per-series mapping keyed by `input.series`.
  - Add `write_run_entry(part, run_id, *, measurements=None, embeddings=None, …)` to P3 T2 as the
    single owner of a run entry.

### I16. P3 T4's reader list misses readers, and its grep check misses parent arithmetic

- **Missed readers:**
  - `phenotypicCLI.py:4000-4010`, recompile overlay regeneration
    (`item.store / MEASUREMENT_TABLE_RELATIVE_PATH`);
  - `_cli_migrate.py:1059`, the literal glob (C6);
  - `replace_embedded_measurement_table` and its two callers (C6).
- **Parent arithmetic that Step 3's grep cannot catch:**
  - `_cli_chunk_writer.py:326-327` (`parents[2]`, `parents[4]`);
  - `_cli_recompile_slurm_scripts.py:532` (`parents[2]`);
  - `_cli_recompile_worker.py:481-484,516,587`, including the literals
    `("tables", "measurements", "table.parquet")`.
- **Fix:** add these to the list. Extend Step 3's check to `parents[` and to the literals
  `"tables"` and `"table.parquet"`.

### I17. P3 T4: "the run's pipeline sha" is the wrong selector after `--mode measure`

- **Evidence:**
  - Measure mode keeps no run state, so `state.config["pipeline_sha256"]` stays the full run's sha
    A.
  - After `--mode measure` with B, each image's marker certifies B's parquet.
  - A later recompile that selects A's folder would revert the deliverables to A's measurements,
    and contradict the marker authority (`marker_claims_measurement_authority`).
  - The SLURM measure finalizer must also be handed B's sha. `deliverables/pipeline.pht-pipe.json`
    is re-serialized at finalize, so its bytes do not hash to B.
- **Fix:** select the run folder that each image's record certifies. That needs no sha passed
  around, and it is what the marker already decides.

### I18. P3 T5: the v1 refusal must not go in `datasets_needing_migration`

- **Evidence:**
  - The predicate is shared with the GUI and runs in `_run_state._advisories` (`:1539`) on every
    `resolve_run_state` call. That includes every GUI poll and every `phenotypic-wait` poll.
  - Checking v1 there costs one root read per store (6,657 on maresca), and puts a migration
    advisory on every existing tree in the GUI.
- **Fix:** use a CLI-only check in the read-only half that stops at the first v1 store, and leave
  the advisory to #283.

### I19. P5 T1: `meas` is admitted where process output cannot hold tables

- **Evidence:**
  - Flat exports (`--process-format tiff`, `--layer objmap`'s PNG, `detect_mat`) write no store
    (`_cli_process_only.py:251-262`). The embeddings would be computed and dropped.
- **Fix:** refuse `meas` in process mode unless the output is a zarr store (`rgb`/`gray`), in the
  same read-only refusal.

### I20. "P0, P1 and P3 touch disjoint files" is false

- **Evidence:** all three modify:
  - `sdk_/_io_constants.py`
  - `sdk_/__init__.py`
  - `_cli/_cli_migrate.py`
  - `_cli/_cli_output_manager.py`

  P1 and P2 both modify `_cli_completion.py`. P2 T3 and P3 both modify `_cli_finalize_run.py`.
  Parallel branches will conflict.
- **Fix:** sequence the shared files, or name one owner per file and rebase order. Correct the
  README table: P4 T4 also depends on P2 T3 (`build_run_metrics(embeddings=)`).

### I21. v1 fixtures must be captured before P3 T2 lands

- **Evidence:** after P3 T2 no code writes the flat layout. So `v1_tree`, `v1_store`,
  `v1_store_legacy_null_pipeline` and `old_tree_with_measurements_csv` cannot be built with the
  writer.
- **Fix:** commit generated v1 stores, with a generator script in the style of
  `tests/fixtures/legacy_hdf/_generate.py`, as P3 T1's first step.

---

## Minor

1. **P0 T2, `_seed_pipeline_config`.**
   - The test calls `_seed_pipeline_config(tmp_path, tiny_pipeline_file)`, but the signature is
     `(pipeline_path, output_dir)` (`phenotypicCLI.py:4634-4636`).
   - The citation is `:4634`, not `:4697`.
   - The writer is `_persist_pipeline_to_output_dir` (`_cli_output_manager.py:610`).
2. **P2 T3, unit.** `test_schema_names_producer_and_unit` expects `str(UNITS.PX)`. `Size_Area`'s
   unit is `px2` (verified: `UNITS.PX2`).
3. **P2 T3, counts.** Taking the counts from `resolve_run_state(...).diagnostics` inside
   `finalize_run` runs an O(N) deep verification in the finalizer, and writes the verification
   cache. Use the state's `work_ids`, the terminal journal, and the source set finalize already
   holds.
4. **P1 T3, `--metadata` suffix.** No suffix check exists for `--metadata` in full mode
   (`phenotypicCLI.py:2050-2060`, `click.Path` only). The exit-2 test needs a new click
   callback, or a `UsageError` in the read-only half. Name it.
5. **P4 T2, Arrow type.** `_write_validated_frame` uses `pa.Table.from_pandas(frame)`
   (`_measurement_tables.py:163`). That yields `list<halffloat>`, not `fixed_size_list`
   (verified). Say "do not route embeddings through it". `FixedSizeListArray.from_arrays` plus an
   explicit schema works and is deterministic (verified, pyarrow 23.0.1; polars 1.41.2 reads it as
   `Array(Float16, 4)`).
6. **P3 T2, group documents.** The contract validators compare group documents with
   `_TABLES_GROUP_DOCUMENT` by exact equality (`_measurement_tables.py:141-149,468-472,539-542`).
   The ngio attributes on `measurements/` need a matching validator change.
7. **P4 T1, payload keys.** `embed()` keys are `<model>/<region>`. Two embedders with the same
   model and region (say, a different `pooling`) collide. Refuse at pipeline validation.
8. **P4 T4, memory.** `aggregate_embedding_tables` over 33k stores is about 5 GB of float16
   vectors. Stream with `pq.ParquetWriter`, one row group per store.
9. **P3 T5, sha source.** The `<seq>`/sha comes from the latest `full`/`legacy` application.
   A v1 table rewritten by `--mode measure` (never journaled) is then labelled with the wrong
   pipeline. Use the figures descriptor's run sha when it is present, or record the limit.
10. **P3 T4, latest run.** "Latest" for the GUI readers should be the latest run **with**
    `measurements`. A grid-only process run has none.
11. **P5 T1, side effects.**
    - `MODE_SLOTS["process"]` gaining `meas` also widens `reference_operations_in_scope`.
      Update the `_cli/CLAUDE.md` bullet "A SubtractBlank inside a measurer needs the table in
      full mode and not in process mode".
    - AutoConvertRaw-GC calls `process_single_apply_only_core` directly
      (`acr_core/process_one.py:118`), bypassing the CLI refusal. Add the guard in the core too.
12. **P5 T2, determinism.** A real model's CPU float16 output may differ across CPU types, and
    process-store bytes feed downstream work ids. Measure it in the pilot, or state the limit.
13. **README, exit-code table.** "1 = refused" conflicts with the refusals that raise
    `click.UsageError` and exit `2`, such as P5 T1's. Reword the table.
14. **P4 T3, tune range.** The `crop_px` tune range has step 14. That produces values that are
    invalid for DINOv3 (patch 16).
15. **P2 T5, DVC config precedence.** `find_dvc_cache_type` should overlay `config.local` on
    `config`, and say that global and system DVC config are ignored.
16. **P2 T2, "reader only".** `resolve_run_state` rewrites `.phenotypic/verification_cache.json`,
    so the waiter is not strictly a reader. Say so in the how-to; it is harmless.
17. **P1 T2, test surface.** 21 test files reference `measurements.csv`, `MEASUREMENTS_CSV` or
    `measurements_csv_path`. Derive the surface by grep, as P2 should.
18. **P4 T1, citation.** `_image_pipeline_core.py:1122` is inside `apply()`. `measure()` is at
    `:1139`.
19. **P5 T1 / P4 T1, typing.** A `ClassVar[bool]` on `MeasureFeatures` overridden by a `@property`
    works at runtime (verified on pydantic v2). mypy will flag the override; add an ignore with a
    reason, or make both properties.
20. **P3 T2, objmap digest.** Hash `dtype + shape + C-order bytes` of the objmap. If Stage 1's
    zeros and Stage 3's labels differ in dtype, identical labels hash differently and the entry is
    latched stale.
21. **P0, two current files.** The GUI analysis writer (`_gui/analysis/_recipe_state.py:909`)
    will write the new name beside an old one, so a folder briefly has two current files (A4). This
    is #283's, and harmless because the resolver prefers the new name. Note it there.

---

## Verified as correct

- **DVC params path.** The `-S` key path is `pipe_cfgs.<OpKey>.params.<field>`, and
  `meas.<key>.params.grid.params.row_edges` is real. `to_json` nests an `OperationField` as
  `{"class", "params"}` (verified with `MeasureSymZones(center_detector=OtsuDetector())`).
- **Grid finder.** `ManualGridFinder`'s fields are `nrows, ncols, row_edges, col_edges`, and it
  re-derives `nrows`/`ncols` from edited edges.
- **Consolidation.** Tables written into the `.part` before the root are consolidated:
  `_consolidate_store_part` runs after the root (`_image_io_handler.py:1512-1513`), and
  `zarr.consolidate_metadata` walks nested groups, provided every level has its group document.
- **`LiveSlurmJobs`.** `LiveSlurmJobs(live, unverified, unresolved)` is positional and falsy when
  empty (`_cli_staged_orchestration.py:690-706`).
- **Waiter timing.** No gap found in which the waiter sees no job while a finalizer is queued but
  unledgered: every submission writes an `intent` row before `sbatch`, and the dispatcher stays
  live until its `submitted` row is written. Keep a test for the intent-only row.
- **Constants and files.** `PROCESS_LAYER_SEMANTICS_REVISION` is at
  `_cli_failure_tracker.py:214`; `MODE_SLOTS` is at `_cli_preflight.py:263`. All the test files
  and the sbatch script the plan names exist.

---

## Fixtures

**Exist today:**
- `synth_one_level_input`, `simple_pipeline_json`, `make_exec_config`, `zarr_world`,
  `legacy_run` / `legacy_format_run` (`tests/unit/cli/conftest.py`).
- The SLURM-submission stub harness in `tests/unit/cli/test_cli_slurm_submission_summary.py:90-117`.
  Reuse it for `fake_sbatch`.
- `detected_plate` (module-local, `tests/unit/util/test_output_headers.py`) and `plate`
  (module-local in several files). Neither is shared, so promote each to a conftest or
  re-declare it.

**Do not exist, so a task must create them:**
- `fake_sbatch`: in `test_cli_v2.py:1654` it is a nested function, **not** a fixture, contrary to
  P2 T1.
- `fake_scancel`
- `finished_tiny_run`, `finished_tiny_run_with_model`, `finished_tiny_run_with_embeddings`
- `tiny_inputs`, `tiny_inputs_one_blank`, `all_blank_inputs`
- `tiny_pipeline`, `tiny_pipeline_file`, `tiny_mirror`, `metadata_bytes`
- `two_revision_run`
- `measured_image`, `store_with_two_runs`, `store_with_crop_entry`, `full_run_store`
- `v1_tree`, `v1_store`, `v1_store_legacy_null_pipeline`, `old_tree_with_measurements_csv` (I21)
- `process_store_v2`
- `stub_backbone`, `toy_embedder`, `toy_grid_embedder`
- `image_with_crop_payload`, `image_with_two_models`, `tree_with_one_stale_store`

Helpers the tests call that are not fixtures, and do not exist either: `run_cli`,
`run_cli_measure`, `run_cli_process`, `run_full`, `invoke_cli`, `run_recompile`,
`run_cli_with_unstageable_gpu`, `tree_digest`, `one_store`, `root_tables`, `run_entry`,
`crop_entry`, `resave_with_pipeline`, `resave_with_objmap`, `touch_root`, `make_dvc_repo`,
`run_preflight`, `write_v_prev_manifest`, `read_run_record_without_field`, `fake_store` (ACR).
