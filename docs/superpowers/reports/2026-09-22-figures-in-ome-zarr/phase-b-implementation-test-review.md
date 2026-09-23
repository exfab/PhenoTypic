# Phase B gate: implementation and test review

- **Scope:** `3e200280..c63d6911`, `src/` and `tests/` only. Covers Task 6 (`8922a1bd`), Task 7 (`6332d268`), the MeasureOrientationZones fix (`5e8dd097`), and Task 7a/7b (`91c6e117`, `00b3e456`, `65931d20`, `c63d6911`). PR #238, which came in through the merge `ff738bb1`, is not reviewed. Its files were read only where this branch changed them afterwards.
- **Reviewed against:** spec §1, §1a, §3, §3a, §4, Revision 1–14; plan Tasks 6, 7, 7a, 7b; Phase A report MINOR-9.
- **Method:** I read the code and tests. I ran no pytest, probes or mutations. Every item marked UNVERIFIED is an inference from reading.

## Summary

| Severity | Count |
|---|---|
| BLOCKER | 0 |
| MAJOR | 1 |
| MINOR | 12 |

**Phase A MINOR-9 is resolved.**
- Measure mode now passes the real `figures` (`_cli_process_single.py:473-498`).
- `emit_image` no longer appears anywhere in `src/` (`grep -rn emit_image src` finds nothing).
- The preflight/emit mismatch in its second bullet went away with `emit_image`.

**What holds** (verified by reading, with evidence):

- **Never-wipe, store level.** Every rewrite path carries or keeps the other runs:
  - Full, Stage 1, Stage 3 and process mode all go through `_write_store_part`. It calls `carry_figure_runs(final, part, exclude=<this run>)` (`_image_io_handler.py:1415-1443`) before writing the root.
  - The measure rewrite removes only `figures/<this run>` from the hard-linked part (`_measurement_tables.py:692-700`). The root merge keeps every other run (`_image_figures.py:394-414`).
  - `write_provenance_checkpoint` is a read-modify-write of the root, so it preserves the `figures` key (`_provenance.py:1075-1093`).
  - No `_cli` code path deletes a store except `--overwrite`; see MAJOR-1.
- **Hard-link safety in the measure rewrite.** Everything in this run's folder is a new file:
  - `_ensure_group` never rewrites an existing `zarr.json`, and it writes through `atomic_write_json`, which renames rather than writing in place (`_image_figures.py:212-223`).
  - Kept §3a bytes are read from the live store before the part exists.
  - Other runs' folders in the part are never opened for writing.
  - `_carry_file` refuses any path outside its own run folder (`_image_figures.py:361-363`).
  - The residual risk, if the clear fails, is MINOR-6.
- **One run id per run:**
  - The call is minted once, after the `--overwrite` rmtree and the restart clear, and reused on resume (`phenotypicCLI.py:2428-2430`, `:826-852`).
  - It is recorded in `create_initial_state` (`_cli_state_management.py:388-392`) and rewritten unchanged by the resume branch (`phenotypicCLI.py:3023-3025`). Both happen before any strategy runs (`:3058`).
  - In-process workers get it on the CLI's `OutputManager` (`:3075-3083`). The local process worker is passed it explicitly (`_cli_execution_strategies.py:612`).
  - SLURM array, process and staged workers read it from state (`_cli_process_single.py:743-747`, `_cli_staged_slurm_worker.py:192,379`). Measure mode on SLURM reads `job_metadata.json`, written before fan-out (`_cli_execution_strategies.py:1062-1076`).
  - It never travels on a command line; the tests at `test_cli_figures_run_date.py:177-249` check this.
  - Stage 1 and Stage 3 take the pipeline digest from the same journal application (`_cli_staged_workers.py:384,608`).
- **Measure reuse:** `latest_run_date` takes the maximum ISO date among runs with a matching `pipeline_sha256` (`_image_figures.py:438-464`). The digest comes from the pipeline file itself, not the store's journal (`_cli_process_single.py:472`). Full mode's `resolved_pipeline_identity` and the worker snapshots are byte copies of the same file (`_cli_execution_strategies.py:956-969`), so the digests agree.
- **§3a keep:**
  - A binding is kept only from `keep_from`'s folder for the same `run_id`, and only if every file verifies against its sha256; the keep is all or nothing (`_store_figures.py:256-332`). Any other folder is never read.
  - A binding that cannot be kept is listed in `unavailable`, not in `failed` (`:142-145`).
  - A stored entry drawn by a different class is refused (`:298-303`).
- **Process mode omits `initiated_*`:** only the date is passed (`_cli_process_only.py:357-366`). The two-process byte-identity test uses a real, different timestamp and pid in each process (`test_process_only_zarr.py:689-706`), so it checks that they are omitted.
- **GPU-detector refusal** still holds for a plot on a top-level detector (`_cli_pipeline_split.py:141-151`), and `test_pipeline_split_nested.py:290` pins it.
- **Guards:**
  - `build_image_figures` and `keep_image_figures` re-raise `PlotPublicationBlocked` (`_store_figures.py:149-150,206-207`).
  - Full, Stage 1 and Stage 3 re-raise a fenced commit as `SlurmGenerationInactiveError` through `slurm_generation_inactive_cause` (`_cli_process_single.py:385-395`, `_cli_staged_workers.py:401-417,666-683`).
  - `_check_active` precedes both the build and the copy-out (`_cli_process_single.py:351,378`; `_cli_staged_workers.py:383,631`).
- **Copy-out order.** In every mode the copy-out runs after the store is promoted and before anything that marks the image complete:
  - Full: promotion → copy-out → the caller's completion record (`_cli_process_single.py:363-384`).
  - Stage 3: promotion → copy-out → Stage-3 marker → token → raw (`_cli_staged_workers.py:616-664`).
  - Measure: table and figure transaction → copy-out → marker refresh (`_cli_process_single.py:492-506`).
- **Continuation:** the process revision is 3 (`_cli_failure_tracker.py:205-208`). `run_initiation` is not in the digest; `test_cli_figures_run_date.py:149-169` pins this.
- **Lazy imports:** every new import in `_cli`/`phenotypicCLI` is inside a function or under `TYPE_CHECKING`, and `FigureInputUnavailable` is stdlib-only. UNVERIFIED: I did not re-run `test_startup_imports.py` / `test_deferred_imports.py`.
- **Windows:**
  - Run folder names are `[0-9-]` plus hex.
  - Descriptor paths are POSIX and are parsed with `PurePosixPath`.
  - Carry and write go through `ngff_.long_path`.
  - The one descriptor-held test is skipped on win32.
  - UNVERIFIED: none of this was run on Windows.

---

## MAJOR

### MAJOR-1: `--overwrite` deletes every run folder, but spec §1a says it keeps them. The test that claims to cover it does not run the flag.

- **Spec:** `design.md:291-297`: *"carry every other run folder across byte-for-byte … This holds even when full or process mode rewrites the store from scratch (`--overwrite`, or a re-derived process run)."* This is part of the user-decided §1a.
- **Code:** for a fresh full or process run, `--overwrite` runs `shutil.rmtree(output_dir)` on the whole output root (`phenotypicCLI.py:2372-2385`). That removes every `results/<ds>/zarr/<stem>.ome.zarr`, so every run folder and descriptor entry is gone before any store is written. The carry in `_write_store_part` never sees an old store.
- **Test:** `tests/unit/sdk_/test_image_figures_store.py:173-175` says it covers *"Full `--overwrite`, a re-derived process store, Stage 3 over Stage 1"*. It only calls `save2zarr` over an existing store, which is what the plan narrowed "overwrite" to (`plan.md:2639`). No test drives the CLI flag. The docstring therefore overstates what is covered.
- **Resolution** is a user decision, either:
  1. correct the spec: `--overwrite` deletes the output tree and with it the store history, as its help text already says; or
  2. change `--overwrite` so it keeps `results/**/zarr/*.ome.zarr/figures/`, which is a larger change.

  Option 1 is probably intended. Either way, fix the test docstring.

---

## MINOR

### MINOR-1: Stage 3 records a Stage-1 binding-level failure with `plot_class: "<unresolved>"`.
- Stage 3 copies out through `PlotCoordinator(plan.post_pipeline, …)` (`_cli_staged_workers.py:633-640`). `plot_classes` therefore lacks every pre-GPU binding.
- For a kept Stage-1 binding with pages, the class falls back to the descriptor (`_store_copyout.py:107-108`). For one that failed outright (`page: null`), the descriptor records no class, so `.failures.jsonl` gets `<unresolved>` (`_store_copyout.py:83-86`).
- Spec §3 requires `plot_class` to come from the pipeline's binding.
- Fix: give the coordinator the union of `plan.pre_pipeline.get_plots()` and `plan.post_pipeline.get_plots()`, or pass `plot_classes` for both.
- No test covers this.

### MINOR-2: Errors in naming the run folder can fail the image, although spec §3 step 1 says no figure error can.
- `figure_run_for` is evaluated outside every per-binding failure boundary. It builds a `FigureRun`, which raises `ValueError` on a malformed date (`_image_figures.py:104-117`), and `build_image_figures` raises `ValueError` when `run is None` and the pipeline has image bindings (`_store_figures.py:129-133`).
- In full mode and Stage 1/3 this becomes a `PerImageScientificError`, a terminal failure that later resumes skip (`_cli_process_single.py:390-395`).
- Triggers:
  - a hand-edited or corrupt `state.config["figures_run_date"]`: `_initiation_from` accepts any `str` (`_cli_state_management.py:153-167`);
  - a journal with no pipeline digest (Stage 1/3 and process mode take it from the journal).

  Neither happens with CLI-written state, which is why this is MINOR.
- Hardening: validate the date in `_initiation_from`, returning `None` for a bad one.

### MINOR-3: Stale docstrings still say `figures=None` "writes no `figures` key".
- It now means "add no run, and carry every run already at the path".
- Locations:
  - `_image_io_handler.py:1095-1097` (`save2zarr`, public API);
  - `_image_io_handler.py:1188-1190` (`_save_store`);
  - `_cli_output_manager.py:1896-1898` (`save_image_store`).
- `save2zarr`'s `path: … Created or replaced.` (`:1091`, and `:1168` for `_save_store`) is also misleading: the promote is still a rename, but the new store now contains the old store's figure runs.

### MINOR-4: Run entries record no input identity, so a store rewritten for a different image carries the old image's figures.
- `carry_figure_runs` carries from whatever store is at *final*.
- The CLI case: an input file replaced under the same stem, then `--restart`. More generally: any library call `save2zarr(existing_path)` with a different image.
- Afterwards the store holds another image's figures under older run ids. A run entry has `date`, `pipeline_sha256` and `initiated_*`, but no work id or input digest, so a consumer cannot tell.
- This follows the user's "never wiped" decision, so it is not a defect. It is a gap in the descriptor worth recording: adding `work_id`, or the input sha, to the run entry would close it.

### MINOR-5: Staged output now differs from a single-pass run for a pre-GPU `PlotImage` op that is not a §3a provider.
- Stage 1 draws such a plot on the image after the pre-GPU ops only (`_cli_staged_workers.py:371-385`). A single-pass run draws it after the whole pipeline.
- Root `CLAUDE.md:179` states *"The output folder is identical to a single-pass run"*.
- Spec §3a's table sanctions this: Stage 1 builds the bindings whose producer it ran.
- No shipped op besides `CalibrateColorRpcc` is a `PlotImage`, so for shipped code this cannot happen today. Before this branch such plots were refused. Task 8 should qualify the CLAUDE.md sentence.

### MINOR-6: A clear that fails silently would let the measure rewrite write through into the live store.
- `_rewrite_store_tables` clears this run's folder with `rmtree(..., ignore_errors=True)` (`_measurement_tables.py:698-700`). `write_image_figures` then writes with `Path(target).write_bytes(...)` (`_image_figures.py:258`).
- If the rmtree silently leaves a hard-linked file in place (permissions, or a Windows path over 260 characters, since that rmtree does not use `long_path`), the write truncates the shared inode, which changes the published store before its new root exists.
- The tables block uses the same `ignore_errors` pattern, but writes its files differently.
- Hardening, either:
  - unlink `target` before writing, or write to a temporary name and rename; or
  - assert that the run folder is absent before `write_image_figures` runs.

### MINOR-7: A future `schema_version` loses history in one path and is mislabelled in another.
- `carry_figure_runs` carries nothing when `schema_version != 1` (`_image_figures.py:328-334`). A full save over such a store therefore drops all of its runs, contrary to "never wiped".
- `apply_image_figures_attributes` overwrites the existing `schema_version` with the incoming `1` and keeps the old runs (`_image_figures.py:407-414`). A measure rewrite would therefore label newer-layout runs as v1, which is exactly what the carry comment refuses to do.
- This only matters once a v2 exists. Pick one behaviour for both paths.

### MINOR-8: `--mode measure` behaviour is described incompletely in `correction/CLAUDE.md`.
- `src/phenotypic/correction/CLAUDE.md:57-58` says measure *"lists the binding as `unavailable` in its own run folder"*.
- Under Revision 14, measure with the same pipeline reuses the run folder and keeps the overlay; `test_calibration_figure_in_store.py:139-155` pins this.

### MINOR-9: `_cli/CLAUDE.md` still says the process revision is "currently `2`".
- `src/phenotypic/_cli/CLAUDE.md:46`. The code says 3. This belongs to Task 8's scope, but it is a leftover in `src/`.

### MINOR-10: The three initiation keys are spelled twice.
- `RUN_INITIATION_STATE_KEYS` (`_cli_state_management.py:127-131`) restates the literals of `JobMetadataKey.FIGURES_RUN_DATE` / `INITIATED_AT_UTC` / `INITIATED_PID` (`_io_constants.py:2542-2544`).
- `sdk_/CLAUDE.md` says JSON contract keys come from `_io_constants.py`.
- Also, `phenotypicCLI._run_initiation` imports the private `_initiation_from` (`phenotypicCLI.py:840-843`); a public reader for `state.config` would be cleaner.

### MINOR-11: Two assertions are weaker than their tests' names claim.
- `test_calibration_figure_in_store.py:155`: `_deliverable(out) == data` after measure mode compares bytes the *full* run already copied out, so it passes even if measure mode's copy-out never ran. Compare mtime or inode, or remove `deliverables/plots` before the measure run.
- `test_image_figures.py:191-204` (`test_carry_links_every_other_run_byte_for_byte`) checks bytes, not links. A carry that silently always copied would pass. The same-inode check exists only on the measure path (`test_figures_in_store.py:156`, `test_image_figures_store.py:115`). That is acceptable, because the docstring says "linked (or copied)", but the test name says "links".

### MINOR-12: Two spec rows have no dedicated test.
- **Process re-derive over another day's store.** The claim is that the consolidated process writer carries the older run and consolidates its groups. `test_image_figures_store.py:92-99` writes a fresh store only. Carry is shared with `save2zarr` through `_write_store_part`, and consolidation over carried groups is plausible but unexercised. UNVERIFIED.
- **Local full mode across midnight.** Two images processed on different days should land in one folder. This is covered only structurally: `OutputManager.run_initiation` carries the value, and `test_figures_run_date_cli.py` checks the state. No test runs two images with the clock advanced between them.

---

## Test validity (false-green hunt)

- **Order dependence.** None found:
  - multi-run comparisons use `set(...)`;
  - single-run assertions use `list(...) == [one]`;
  - `unavailable` and `bindings` order come from pipeline order, which is deterministic.
- **Date and clock dependence.** None found:
  - every integration test pins its dates through `RunInitiation`;
  - the clock is patched only at `_image_figures._utc_now`, which is the one clock read;
  - the staged test's clock patches are meaningful: a Stage 3 that dropped `initiation` would fall back to `_utc_now()` and fail on `run_id`.
  - `test_cli_figures_run_date.py:199` asserts that the literal `2026-09-22` (today) is absent from the array script. It passed on the only day it could have failed for the wrong reason.
- **Fixtures that bypass the path they claim.**
  - `test_image_figures_store.py:173` is covered in MAJOR-1.
  - `test_staged_figures_keep.py` seeds the Stage-3 run folder through a real `save2zarr` over the Stage-1 store, which is a faithful seed.
  - `emit_image_via_store` builds a minimal store with no pixels. That is acceptable for copy-out tests; the CLI integration tests exercise the real store.
- **Tests ported from `emit_image`** (`test_coordinator.py`, `test_publication_end_to_end.py`, `test_embedded_measurement_replacement.py:155`) follow the plan's per-test table:
  - The deleted tests (`strict`, Chrome-rerun) have named replacements in `test_store_copyout.py`.
  - `test_the_flat_path_closes_its_matplotlib_figure` now pins the close in the build; after the move it can no longer detect a copy-out-side leak, and its docstring says so.
  - `test_embedded_measurement_replacement` still shows that a failure between the table write and the marker leaves the marker stale.

## Leftovers grep (`src/`)

- `emit_image`, `KEEP_FIGURES`, `_KeepFigures`, `carry_from`, `--figures-run-date`: none in `src/`.
- `figures_run_date` appears only as the state and job-metadata key, which is intended.
- The `emit_image` hits in `tests/` are the `emit_image_via_store` fixture and a parametrize id; they are intended.
- Stale prose: MINOR-3, MINOR-8 and MINOR-9.
- A terminology slip at `_cli_pipeline_split.py:135`: "Stage 3 **carries** it" should read "keeps", since "carry" is the between-stores term in §1a and "keep" is the same-run term in §3a.

## Unverified

- I ran no pytest, probe or mutation. The lead reports that the mutation checks and green baselines were run.
- Lazy-import guard tests were not re-run.
- Windows behaviour is reasoned, not executed.
- MINOR-6's write-through requires the rmtree to fail silently. I did not reproduce that.
