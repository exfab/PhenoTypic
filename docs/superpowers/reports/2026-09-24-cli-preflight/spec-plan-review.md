# CLI preflight: independent review of spec and plan

- **Reviewed:** `docs/superpowers/specs/2026-09-24-cli-preflight/design.md`, `DEFERRED.md`,
  `docs/superpowers/plans/2026-09-24-cli-preflight/plan.md`,
  `docs/superpowers/reports/2026-09-24-cli-preflight/claim-verification.md`, and
  `baseline_probes/`
- **Code base:** HEAD `145157e` (docs-only commit on top of `81d19ec`; no source differs)
- **Reviewer probes:** scratch scripts under the session scratchpad (`review/`), all run with
  `uv run python`; none are committed. Outputs are quoted where they are the evidence.

## Verdict

**Not ready.** The ordering fix (§1), the requirement-declaration design (§3), and most of
the cited evidence are sound, and the large majority of file:line citations check out. Three
problems would ship a regression or leave a claimed fix unclosed. First, the metadata
preflight (§9) computes join keys against a three-column source frame, so it refuses
legitimate per-colony and plate-layout joins that the production join handles correctly
(R1, reproduced). Second, the custom-op preload fix (§10.2, F13) does not reach local
joblib/loky workers, so a default local run with a custom operation still fails every image
after the fix (R2, reproduced). Third, the RAW work-id fence (§12) is specified for
`work_id_for_image` only, but the SLURM worker recomputes the identity through a second
producer and refuses any mismatch, so every RAW image on SLURM would fail (R3). Beyond
those, several error-severity checks are not scoped by mode or slot and will refuse valid
`process` and `measure` runs (R4), the `PF-GRID-PRESET` condition is wrong (R5), and the
§8 "complete column set" argument is false (R6). With those fixed the design is
implementable; the plan needs the listed test and scoping corrections.

Counts: **Blocking 3, Major 8, Minor 19, Nit 8.**

## Findings

### Blocking

**R1. §9 `PF-META-DUP-KEYS` and `PF-META-NO-KEYS` refuse valid per-well metadata joins.**
- *Section:* spec §9 table, §12 "Refusals that are new"; plan Task 11.
- *What is wrong:* the analysis joins the metadata CSV against a source frame holding only
  `Metadata_ImageName`, `Metadata_Suffix`, `Metadata_Dataset`
  (`_gui/run_console/_request_safety.py:348-364`). The production join runs against the
  measurement frame, which also carries measurement headers such as `Grid_RowNum` and
  `Grid_ColNum`, and `normalize_external_metadata_columns` deliberately preserves those
  headers as join keys (`_cli/_metadata_join.py:107-129`). A plate map keyed on
  `ImageName + Grid_RowNum + Grid_ColNum` therefore looks like duplicate keys to the
  preflight, and a single plate layout keyed only on `Grid_RowNum + Grid_ColNum` looks like
  no keys at all. The GUI treats both as advisory warnings for exactly this reason
  ("may fan out", "may skip it"); the spec promotes them to errors.
- *Evidence (reviewer probe `review/meta.py`, calling `prepare_metadata_join_keys`):*
  ```
  plate_map   | source-frame: ('Metadata_ImageName',) dups 2 | measurement-frame: ('Grid_ColNum', 'Grid_RowNum', 'Metadata_ImageName') dups 0 unmatched 0
  layout_only | source-frame: () dups 0 | measurement-frame: ('Grid_ColNum', 'Grid_RowNum') dups 0 unmatched 0
  ```
- *Recommendation:* make `PF-META-NO-KEYS` and `PF-META-DUP-KEYS` errors only when the
  metadata CSV has no column that could be a measurement-level key (i.e. when
  `unverified_join_columns` is empty); otherwise emit them as warnings alongside
  `PF-META-UNVERIFIED`. Add plan Task 11 tests for both plate-map shapes asserting no error.
  Also correct §9's sentence "Both produce tables that are wrong rather than incomplete":
  with no common columns `join_metadata` skips the join and logs a warning
  (`_cli_output_manager.py:336-341`), which is incomplete, not wrong.

**R2. F13 is not closed: local parallel workers never preload custom operations.**
- *Section:* spec §10.2, §5; plan Task 7.
- *What is wrong:* `LocalParallelStrategy` runs images through
  `Parallel(n_jobs=effective_n_jobs)` (`_cli/_cli_execution_strategies.py:375`), default
  backend loky, whose workers are fresh processes. Each worker calls
  `ImagePipeline.from_json` inside `process_single_image_core`
  (`_cli/_cli_process_single.py:261`). A preload in the main process does not register the
  class in those workers. The local staged strategy has the same shape
  (`_cli_staged_strategy.py:214`, `:351`). §10.2's worker list and Task 7's entry-point test
  cover `main` functions only, not the core functions loky calls.
- *Evidence (reviewer probe `review/loky/run.py`):* main process calls
  `preload_custom_operation_modules()` with `PHENOTYPIC_PRELOAD_MODULES=my_custom_ops_reg`
  (simulating the §10.2 fix), then `phenotypic_cli --njobs 2`:
  ```
  PerImageScientificError: Class 'MyThreshDetector' not found in phenotypic namespace. ...
  Completed: 0/2   Failed: 2   EXIT 1
  ```
- *Recommendation:* call the (idempotent) preload at the top of every function that
  deserializes a pipeline in a worker: `process_single_image_core`,
  `process_single_store_measure_core`, the process-only core, and the staged Stage-1/3
  callables; or pass a loky `initializer`. Add a Task 7 test that runs the CLI as a
  subprocess with `--njobs 2` and a custom op and asserts `Completed: 2/2`.

**R3. The RAW fence as planned breaks every RAW image on SLURM.**
- *Section:* spec §12; plan Task 9 Steps 2-3.
- *What is wrong:* the plan folds `RAW_DECODE_REVISION` into `work_id_for_image`. The array
  worker computes the identity independently through `_worker_work_identity`, which calls
  `compute_work_id` directly (`_cli/_cli_process_single.py:123-170`), and raises
  `"SLURM task work identity does not match worklist"` on any difference
  (`_cli_process_single.py:814-838`). Two producers of one derived value is the pattern
  `_cli/CLAUDE.md` ("One producer per derived value") forbids.
- *Recommendation:* put the revision inside `compute_work_id`, keyed on the suffix of
  `relative_image_path`, so both producers inherit it. Add a Task 9 test asserting
  `_worker_work_identity(...)[0] == work_id_for_image(...)[0]` for a `.nef` input and for a
  `.tiff` input.

### Major

**R4. Pipeline and input checks are not scoped by mode or slot, so valid runs are refused.**
- *Section:* spec §4 `PF-GRID-IMAGE`, `PF-GRID-PRESET`; §7 `PF-RGB-OP-GRAY`,
  `PF-DETECT-MODE-GRAY`; §9 metadata checks.
- *What is wrong:* `--mode process` runs only `pipeline.apply()` (`_cli_process_only.py:347`),
  never `measure()`, so a grid measurer or `MeasureColor` in `meas`, or a preset that injects
  a grid finder, cannot fail a process run; the spec refuses all of them because the walk
  covers "any operation in the tree". `--mode measure` runs only `pipeline.measure()` on a
  loaded store (`_cli_process_single.py:441-446`), takes the image class from the store with
  `--image-type` as a fallback only (`:414-416`), and never applies `--detect-mode`; the
  spec's `PF-GRID-IMAGE` and `PF-DETECT-MODE-GRAY` do not account for that. Process mode also
  ignores `--metadata` (`phenotypicCLI.py:1918-1926`), so §9 checks should not run there.
- *Recommendation:* add a per-mode slot map to `PreflightContext` (full: ops + meas + post;
  process: ops only; measure: meas + post) and run each requirement check over that subset
  only. For measure mode, read the image class from each store's `phenotypic.image_class`
  instead of `config.image_type`. Add one test per mode proving a slot outside the mode's
  scope produces no finding.

**R5. `PF-GRID-PRESET` fires on the wrong condition.**
- *Section:* spec §4 table; plan Task 5 Step 1.
- *What is wrong:* the spec says "the pipeline sets `nrows` or `ncols`". Injection requires
  **both**, and is skipped when `meas` already holds a `GridFinder`
  (`_image_pipeline_core.py:1308-1316`). Under `--image-type Image` the CLI does not apply
  `--nrows/--ncols` to the pipeline (`_cli_process_single.py:264-275`), so only the preset
  matters.
- *Evidence (reviewer probe `review/f5.py`):*
  ```
  {'nrows': 8, 'ncols': 12} RAISED RuntimeError [CenteredAutoGridFinder] ... | cause: GridImageInputError
  {'nrows': 8} OK (552, 16)
  {} OK (552, 16)
  ```
  This also settles open question 1: keep the code, with the corrected condition.
- *Recommendation:* condition becomes "`image_type == "Image"`, mode calls `measure()`,
  `pipeline.nrows is not None and pipeline.ncols is not None`, and no `GridFinder` in `meas`".
  Add a test for the one-of-two case producing no finding.

**R6. §8's "the set is complete for metadata columns" is false, so `PF-POST-COLUMN` can
refuse valid runs.**
- *Section:* spec §8; plan Task 12 Step 1.
- *What is wrong:* at least three more sources put metadata columns into the frame post
  runs on. (a) The CLI inserts `Metadata_Dataset` when the dataset column is on
  (`_cli_output_manager.py:1764-1767`, `:1918-1922`); Task 12's `apply_and_measure` baseline
  on the synthetic plate cannot see it. (b) `imread` restores a PhenoTypic-exported file's
  public metadata into the image (`_image_io_handler.py:797-824`), and `insert_metadata`
  writes every public and protected key as a column (`_metadata_accessor.py:366-376`).
  (c) A custom operation may set `image.metadata[...]`. Separately, `JoinMetadata.on` holds
  the *table's* spelling after validation (`post/_join_metadata.py:193-203`) and is
  re-spelled against the frame at run time (`:258-263`), so `required_columns()` returning
  `on` verbatim will misreport.
- *Recommendation:* add the CLI-inserted dataset column to the known set; downgrade
  `PF-POST-COLUMN` to a warning when the tree contains a class outside the `phenotypic`
  package or when any input is a PhenoTypic-exported file or store; derive `JoinMetadata`'s
  requirement through the same `external_metadata_preserved_columns` /
  `ensure_metadata_prefix` rule `_normalized_table` uses.

**R7. The SAM3 licence gate is a new refusal of runs that work today, and §12 omits it.**
- *Section:* spec §10.3, §12, F11.
- *What is wrong:* today `Sam3._ensure_model_loaded` loads through `from_pretrained` with no
  PhenoTypic gate (`detect/nn/_sam3.py:178-205`). A user whose Hugging Face access is granted
  and whose weights are cached runs successfully without `PHENOTYPIC_ACCEPT_MODEL_LICENSE`.
  After §10.3 plus `PF-LICENSE`, every such run is refused. That may be the intended
  contract (the class docstring says the gate applies), but it is a new refusal and must be
  listed, documented, and called out in the change log.
- *Recommendation:* add it to §12 "Refusals that are new" and to Task 16's docs, with the
  one-line remedy.

**R8. The weights-cache probe imports `torch` (and possibly `micro_sam`) on the login node.**
- *Section:* spec §5 "Weights are cached".
- *What is wrong:* §5 argues `find_spec` avoids importing `torch`, then reuses
  `Sam2CheckpointManager.is_cached`, whose `cache_dir()` does `import torch.hub`
  (`detect/nn/_helper/_checkpoint_manager.py:232`, `:241-253`).
  `MicroSamCheckpointManager.cache_dir()` imports `micro_sam.util` (`:409`), which imports
  torch. That is seconds and hundreds of MB in the submitting process, and when torch is
  absent `is_cached` raises instead of returning `None`. Plan Task 8 patches `is_cached`, so no
  test would notice.
- *Recommendation:* compute the torch hub directory without importing torch (`TORCH_HOME`,
  `XDG_CACHE_HOME`, default `~/.cache/torch/hub/checkpoints`), and resolve the micro-sam cache
  from `MICROSAM_CACHEDIR` or `platformdirs` only (the fallback branch at `:417-425`). Add a
  subprocess test asserting `run_preflight` on a `Sam2` pipeline leaves `torch` out of
  `sys.modules`.

**R9. §10.1 enables a RAW decode path that has never run, with no real-file validation.**
- *Section:* spec §10.1, §12; plan Task 9.
- *What is wrong:* the rawpy branch (`_image_io_handler.py:736-771`) is dead code today
  (claim-verification §5). The probe covered routing only, on a TIFF renamed `.dng`; no real
  camera file was decoded. Task 9 tests routing with rawpy patched. So the change that alters
  science output for every RAW user (linear gamma `(1, 1)`, `output_bps=16`, auto-brightness
  on) ships with no check that the result is a sane plate image. On Windows, where `rawpy` is
  excluded (`pyproject.toml:55`), RAW inputs that decode today (through Pillow) become a hard
  refusal; §12 does not mention it. Minor side issue: `rawpy_params.pop(...)` mutates a
  caller-supplied dict.
- *Recommendation:* add a Task 9 step that decodes at least one real RAW sample (a small
  public DNG, committed as a test fixture or fetched in a skipped-if-offline test) and asserts
  shape, dtype `uint16`, and a plausible intensity range; state the Windows behavior change in
  §12 and the docs; copy `rawpy_params` before popping.

**R10. `--overwrite` still deletes the pipeline, metadata or manifest when they live under
`--output`.**
- *Section:* spec §1 (F3 extension); omission.
- *What is wrong:* the overlap refusal is extended to `--input` only. A user re-running with
  `--pipeline out/deliverables/pipeline.json.pht-pipe --output out --overwrite` (or
  `--metadata out/deliverables/metadata.csv`) passes the new preflight, then the `rmtree`
  (`phenotypicCLI.py:2393-2399`) deletes the file before `_prepare_incremental_startup`
  snapshots it (`:2581`) and before workers read it. Under `--restart`,
  `clear_machine_state` deletes a `--pipeline` placed under `.phenotypic/`
  (`sdk_/_io_constants.py:1347-1356`). Both violate the objective "before the CLI has deleted
  ... anything" for inputs the run itself needs.
- *Recommendation:* extend the §1 refusal to `--pipeline`, `--metadata` and `--image-manifest`
  (under `--overwrite`: inside `--output`; under `--restart`: inside `.phenotypic/` and not in
  `_PRESERVED_ON_RESTART`), and add those cases to Task 1.

**R11. F19 has no test, and its severity is decided from one cluster.**
- *Section:* spec §6 "Time fits the partition"; plan Task 13.
- *What is wrong:* Task 13 Step 2 lists no test for `PF-TIME-OVER-PARTITION` (no `scontrol
  show partition` parsing, no `MaxTime=UNLIMITED`, no comma-separated partition list, no
  default partition when none is given). Step 1 fixes the shipped severity from the
  `EnforcePartLimits` value of the one cluster the implementer can reach, but the code runs
  on every user's cluster. A QOS carrying `Flags=PartitionTimeLimit` lets a job exceed the
  partition `MaxTime` (UNVERIFIED from documentation, see external claims below), which would
  make the error a false positive. The step also has no fallback when the implementer has no
  Slurm access.
- *Recommendation:* read `EnforcePartLimits` at run time from `scontrol show config` (same
  10 s timeout) and choose the severity from it, or make the finding a warning. Add unit tests
  for the parser cases above. State in Task 13 what to do without cluster access (keep the
  warning severity and leave open question 3 recorded as open).

### Minor

**R12. Task 1's fixture is not an image.** Plan Task 1 Step 1 calls the
`test_cli_gpu_refusal.py` `image_tree` "a one-image TIFF tree", but it writes
`b"never decoded: the refusal fires first"` to `img001.tiff`
(`tests/unit/cli/test_cli_gpu_refusal.py:57-63`). Once Task 10 lands, every input is
`PF-HEADER-UNREADABLE`, an error, so `test_overwrite_dry_run_previews_and_deletes_nothing`
(expects exit 0) and Task 3's tripwire context break. Write a real small TIFF with
`tifffile.imwrite` or `skimage.io.imsave`.

**R13. Task 4 cites a list the spec does not contain.** "Each of the 13 grid classes in the
spec's background (`FilamentousFungiDetector` ... `MeasureGridSpread`)": the background names
only the four ABCs. The concrete set at HEAD is 16 (reviewer probe `review/enum_ops.py`):
`FilamentousFungiDetector`, `TwoKFilamentousDetector`, `ManualGridPointDetector`,
`GridAligner`, `GridApply`, `GridOversizedObjectRemover`, `KeepSectionLargest`,
`MergeWithinSection`, `ReduceSectionsByLine`, `RemoveGridOutliers`, `MeasureGridSpread`,
`MeasureGridLinRegStats`, `MeasureNeighborDist`, plus the three finders. Put this list in the
plan.

**R14. §3 table says `MicroSamDetector` is "RGB never".** Its `input_layer` is a
`GpuInputLayer` field with default `"gray"` (`detect/nn/_microsam_detector.py:164`); a user may
set `"rgb"`. It should inherit the `GpuDetector` override, not hard-code `False`.

**R15. `sbatch --test-only` details.** (a) `format_sbatch_directives` emits no shebang, and
`sbatch` rejects a script whose first line is not `#!`; the rendered stdin script must
prepend one. (b) It requires `output_log`/`error_log` paths; say which. (c) Use
`sbatch_submission_environment()` as `submit_script` does (`sdk_/slurm/_sbatch.py:228-235`),
or `SBATCH_*` variables make test and real submission differ. (d) A transient controller
error (socket timeout) would become an error-severity `PF-SBATCH-REJECTED`; classify
communication failures as `PF-SBATCH-UNAVAILABLE`.

**R16. Test-scope rule not followed.** CLAUDE.md asks for the affected surface once per phase,
derived from importers. The plan runs one surface pass (Task 17) after Phase E, and Task 2
Step 6 hand-picks files, missing ones that reference the dry-run path it changes:
`tests/unit/cli/test_cli_provenance_original.py:32`, `tests/unit/cli/test_schema_gate.py`,
`tests/unit/cli/test_embedded_measurement_migration.py`, and
`tests/unit/plotting/test_backends.py:307` (docstring describes the double validation being
removed). Add a surface run at the end of each phase, derived by `grep -rl`.

**R17. Task 8's SAM3 test cannot be shown to fail on the base commit here.** It is skipped
when `transformers` is absent, and in this environment `transformers`, `torch`, `sam2`,
`huggingface_hub`, `micro_sam`, `fil_finder` and `astropy` are all absent (reviewer
`find_spec` probe). Inject a fake `transformers` module into `sys.modules` instead.

**R18. Task 9 Step 4 names test files that do not exist.**
`tests/unit/cli/test_cli_failure_tracker*.py` matches nothing; the relevant files are
`tests/unit/cli/test_work_id_semantics_revision.py` and `test_store_work_identity.py`. The
".tiff work id is byte-identical to the base commit's" test, computed "from
`compute_work_id`'s inputs", is tautological if it calls the modified function; pin a literal
digest computed at `81d19ec`.

**R19. Task 3 Step 3 "load the pipeline once" is underspecified.** `validate_pipeline` returns
`(bool, str)` and discards the pipeline (`_cli/_cli_validation.py:29-83`). Say whether its
signature changes or a second load is acceptable.

**R20. Task 7 Step 1's grep misses list-form entry points.** `grep "\-m phenotypic"` does not
match `_cli_slurm_array_scripts.py:218`, which spells the module as a separate list item
(`"phenotypic._cli._cli_process_single"`). Grep for `phenotypic._cli._cli_` module strings.

**R21. Output-location heuristics.** `$SCRATCH` is shared parallel storage on many clusters, so
the node-local warning misfires there; prefer the filesystem type of the output mount
(`tmpfs`, local `ext4`/`xfs` vs `gpfs`/`lustre`/`nfs`). The disk-space "lower bound" is not
one: `--mode process --layer objmap` writes a label PNG far smaller than its input, and in
measure mode the inputs are stores already inside `--output`. Either scope the warning to full
mode or drop the "lower bound" label (and then §13's reason for no validation script needs
revisiting).

**R22. Removing the pandas CSV check weakens `--skip-validation` runs.** Today the parse at
`phenotypicCLI.py:2149-2160` runs even under `--skip-validation`; after Task 11 an unreadable
CSV is found only per image. Keep a minimal parse through `read_metadata_csv` outside the
skippable block, or list this in §12.

**R23. §10.5 may change worker-embedded metadata dtypes.** Full-scan inference differs from
100-row inference not only where the latter raises (e.g. first 100 rows null then integers).
A continuation could then mix stores embedded under both rules. Impact on aggregation is
UNVERIFIED; state in §12 whether a fence is needed.

**R24. §10.4 also blocks reloading old outputs.** `--mode recompile` and the GUI reload the
pipeline from `deliverables/pipeline.json.pht-pipe`, seeded from the user's bytes. A run whose
original pipeline had a duplicate key can no longer be recompiled. Add to §12.

**R25. Header semantics for palette and multi-page files.** A palette PNG reports one band in
its header but decodes to RGB through imageio/Pillow; a multi-page TIFF's `pages[0]` is not
what `skimage.io.imread` returns. Task 10 Step 1 probes palette PNG; also probe multi-page
TIFF and state that `PF-DETECT-MODE-GRAY`/`PF-RGB-OP-GRAY` use the *decoded* channel count.

**R26. `--sample` is ignored by the preflight.** Headers of every input are read and the
all-versus-subset severity is computed over the full set, while only the sample runs. State
the intended behavior.

**R27. `F3`'s refusal should be listed as non-skippable.** §0's `--skip-validation` exceptions
name ordering, option types and the GPU refusal; the destructive-overlap refusal (and R10's
extension) should be named too.

**R28. `PF-CUSTOM-OP` is not a `PreflightFinding`.** Task 7 Step 3 emits it from
`validate_pipeline` as a string, before `run_preflight` can run (the preflight needs the loaded
pipeline). Either say so in §5 or exclude it from `FindingCode`, or the "every code has a hint
and is in the docs" test is covering a code that never appears in a report.

**R29. Omission: same-stem inputs collide.** `a.png` and `a.tif` in one dataset map to the same
`<stem>.ome.zarr` store and the same `Metadata_ImageName` join key. The scan result makes this
statically checkable. The reviewer found no guard by grep in `_cli_directory_scanner.py` or
`phenotypicCLI.py`; whether a later stage refuses it is UNVERIFIED.

**R30. Omission: compute-node visibility of the other inputs.** On SLURM, `--input`, the
`--pipeline` file, `--metadata`, and a `JoinMetadata` table under node-local storage fail on
every worker. Apply the §9 node-local check to them too.

### Nit

**R31.** claim-verification §8 cites `_serializable_pipeline.py:283`; the `json.loads` is at
`:279` (the spec has it right).

**R32.** Plan Task 2 Step 2 cites `sdk_/_io_constants.py:1318-1364` for the kept-names
constant; `_PRESERVED_ON_RESTART` is at `:1313-1315` and is private.

**R33.** Removing `full_validation` from `execute_dry_run` leaves the
`_ANNOUNCED_PLOT_WARNINGS` comment (`_cli_validation.py:23-26`) and the `test_backends.py:307`
docstring describing a double validation that no longer happens.

**R34.** "Preflight" already names three other things (`preflight_plot_backends`, the GUI's
`build_metadata_preflight`, and the unstageable-GPU check that `_cli/CLAUDE.md` calls "the
preflight"). Pick distinct names in the docs.

**R35.** §4 says "The probes showed that `GridImage` fails the same way"; the
claim-verification report probed plain `Image` only. The reviewer's probe (`review/f6.py`)
confirms it for `GridImage` (`RuntimeError` wrapping `NoObjectsError` with `meas`,
`OperationFailedError` from `CenteredAutoGridFinder` without), so only the attribution is off.

**R36.** `RAW_DECODE_REVISION = 2`: say why 2 (the implicit pre-fix decoding is revision 1).

**R37.** Task 16 Step 1 "generated from `HINTS`": say whether the table is generated by a script
or written by hand and checked by the test.

**R38.** The GUI's `_unverified_measurement_join_columns` (`_request_safety.py:367-388`), which
§9 moves into `_cli/`, filters on `"_" in column` before calling `metadata_member_for_header`.
It is a qualification test, not a metadata-prefix test, so it does not break the CLAUDE.md
ownership rule, but the moved docstring should say so, so a later reader does not "fix" it into
a prefix check.

## Coverage of F1-F26

| # | Closing task | Test that fails if the fix is reverted | Gap |
|---|---|---|---|
| F1 | 1, 2 | `test_overwrite_with_a_corrupt_pipeline_keeps_the_previous_run`, `test_overwrite_dry_run_previews_and_deletes_nothing` | Fixture must be a real image (R12) |
| F2 | 1, 2 | `test_restart_dry_run_leaves_machine_state_untouched` | None |
| F3 | 1, 2 | `test_overwrite_refuses_an_input_inside_the_output` | Pipeline/metadata/manifest not covered (R10) |
| F4 | 5 | `PF-GRID-IMAGE` tests | Mode scoping (R4) |
| F5 | 5 | `PF-GRID-PRESET` test | Wrong condition (R5) |
| F6 | 5 | `PF-NO-DETECTOR` tests | None |
| F7 | 10 | `--detect-mode red` all-gray error / subset warning | Measure-mode scoping (R4) |
| F8 | 4, 10 | `MeasureColor` pair; per-class requirement tests | Process-mode scoping (R4) |
| F9 | 8 | `PF-MISSING-MODULE` test with `find_spec` patched | None |
| F10 | 8 | DINOv3 `download(interactive=False)` test | None |
| F11 | 8 | SAM3 gate-before-`from_pretrained` test | Skipped without `transformers` (R17); new refusal unlisted (R7) |
| F12 | 8 | `PF-WEIGHTS-UNCACHED` test | Probe imports torch, untested (R8) |
| F13 | 7 | subprocess `--dry-run` with preload; entry-point order test | **Local loky workers still fail (R2)** |
| F14 | 6 | duplicate-key `from_json` tests | Recompile impact unlisted (R24) |
| F15 | 6 | `--bit-depth 12` usage error | None |
| F16 | 13 | `--gpu-slurm time=banana` parse-time error | None |
| F17 | 13 | `PF-SBATCH-REJECTED` with patched `subprocess.run` | Shebang/env details (R15) |
| F18 | 13 | limit function and GRES return-code tests | `sinfo` exit code on an unknown partition UNVERIFIED |
| F19 | 13 | **none listed** | **No test; severity from one cluster (R11)** |
| F20 | 15 | Validate argv contains Run's SLURM tokens | None |
| F21 | 11 | `PF-META-*` tests | **False positives (R1)** |
| F22 | 11 | 151-row CSV through the worker path | Dtype drift (R23) |
| F23 | 12 | `PF-POST-COLUMN` tests | Completeness claim false (R6) |
| F24 | 9, 10 | routing test with rawpy/skimage patched; `PF-RAW-NO-RAWPY` | **SLURM work-id mismatch (R3)**; no real-file decode (R9) |
| F25 | 14 | output-location tests | Heuristics (R21) |
| F26 | 13 | dry-run preview prints `format_sbatch_directives` output | None |

Every task's first test was checked for whether it fails at `81d19ec`. All do, except that
Task 8's SAM3 test is skipped in environments without `transformers` (R17), and Task 4's
"`import phenotypic.abc_` loads no deferred module" check already passes at base (it is a
guard, which is fine, but it is not the task's failing test).

## Claims verified as correct

- **§1 ordering and read-only status.** Line numbers for `uses_staged_gpu_strategy` (`:2229`),
  the `if restart:` clear (`:2363`/`:2370`), the fresh-run check (`:2386-2410`), the `rmtree`
  (`:2399`), `mint_run_identity` (`:2438`), scanning (`:2447-2484`), validation
  (`:2492-2530`), and the dry-run exit (`:2537-2552`) match HEAD. Every block §1 moves up is
  read-only: `scan_directory_structure`, `apply_image_manifest`, `organize_by_dataset`
  (plain `Dataset` dataclass, no `__post_init__`), `scan_store_outputs`,
  `validate_execution_config`, `validate_pipeline`, `_display_execution_config`,
  `execute_dry_run`. None of them reads `identity`, `config.run_initiation` or anything else
  computed between `:2363` and `:2447`; `config.resume`, `resume_state`,
  `config.output_dir`, `manifest_snapshot` and the `--restart --image-manifest` check are all
  computed above the move target. `_refuse_unmigrated_output` (`:1888-1889`) is already above
  every mutation. The report's §2 ordering defects are consistent with this code.
- **F2.** `_cli_identity.py:239-253` documents "gate finding F8" with the stated
  justification.
- **F3.** The overlap refusal (`:1866-1883`) is `process`-only.
- **§3 ABCs.** The four raise sites are at the cited lines; all 16 concrete grid classes derive
  from one of the four raising ABCs; `GridOperation` itself does not raise but has no direct
  concrete subclass. `RoundPeaksDetector`, `SinePeakDetector`, `RefineBySineFit`,
  `GridAlignmentRefiner`, `InoculumDetector` are not grid-ABC classes. The six unconditional
  RGB classes do require RGB (`ColorDenoise` raises at `_color_denoise.py:196-200`;
  `FocusEdgeColorPhase` at `:238-242`). The ratchet heuristic flags `BayesShrinkCorrector`,
  `VisuShrinkCorrector`, `DenoiseBlockMatch` and `MeasureSymZones` (plot-only RGB read), all
  gray-tolerant, which is what Step 2 anticipates; it does not flag `PadImage` or
  `ColorDenoise`, which the spec handles explicitly. A pydantic probe
  (`review/classvar.py`) confirms a subclass may reassign a parent's
  `_requires_rgb_input: ClassVar[bool]`, that the value lands in `cls.__dict__`, and that
  `model_json_schema()` is unchanged.
- **`InputLayerMixin`** raises `NoArrayError` on gray for `input_layer="rgb"`
  (`sdk_/mixin/_input_layer_mixin.py:53-88`) and precedes the operation base in the MRO, so an
  override there works.
- **F9-F12.** Lazy optional imports in the six GPU detectors and `_load_filfinder_runtime`
  (`_filfinder_detector.py:155-162`); DINOv3 `download()` calls at
  `_dinosam2_detector.py:392` and `_dino_support.py:232` with default `interactive=True`;
  `input()` at `_checkpoint_manager.py:831`; `Sam3` bypasses the gate
  (`_sam3.py:178-205`); `Sam2CheckpointManager.is_cached` at `:241`,
  `MicroSamCheckpointManager.list_cached` at `:460`.
- **F13 (main process).** `preload_custom_operation_modules` callers are exactly the four the
  report lists; `phenotypicCLI.py` never calls it.
- **F14.** `json.loads` without a hook at `_serializable_pipeline.py:279`.
- **F15.** `--bit-depth` is `type=int` (`:1535-1540`), and `Image` accepts only 8 and 16
  (`_image_data_manager.py:486-487`), so the `Choice` loses nothing.
- **F16-F18, F26.** `--slurm` time parse at `:1968-1981`; `--gpu-slurm` parsed without time
  validation at ExecutionConfig creation; `parse_slurm_time` inside
  `format_sbatch_directives`; staged limits at `_cli_staged_slurm.py:558-573` after the
  manifest hashes every input (`:540-555`); the only GRES check is in
  `AutonomousSLURMStrategy` (`_cli_execution_strategies.py:906-947`) and reads stdout without
  checking `returncode`; `_display_slurm_config` re-implements directive naming
  (`_cli_interactive.py:53-84`); `_slurm_headroom.py` subprocess calls have no timeout (D9).
- **F20.** `to_argv` never emits `--slurm`; Run adds SLURM and GPU tokens only in
  `_build_subprocess_argv` (`_gui/run_console/_slurm.py:197-210`).
- **F21-F22.** `join_metadata` behavior at `_cli_output_manager.py:338-418`; worker reads with
  default `pl.read_csv` at `_embedded_measurement_tables.py:85` vs `infer_schema_length=None`
  at `_cli_output_manager.py:333`; the GUI also uses the default (`_request_safety.py:446`).
- **F23 context.** Post runs after the metadata join (`_cli_output_manager.py:1163`, `:1174`).
- **F24.** `IO.ACCEPTED_FILE_EXTENSIONS` branch at `_image_io_handler.py:732-733` precedes the
  RAW branch at `:736`.
- **F25.** No `disk_usage`, `statvfs` or `os.access` under `_cli/` or in `phenotypicCLI.py`.
- **F5/F6 (reviewer probes).** Both-set preset under plain `Image` raises
  `GridImageInputError` via the injected `CenteredAutoGridFinder`; a no-detector pipeline
  fails under both `Image` and `GridImage`.
- **§6 staged profile.** `resolve_stage_slurm_args` (`_cli_staged_slurm.py:78-102`) behaves as
  described.
- **§7 tifffile.** Imported at module level in `_color_space_accessor.py:9` and lazily in
  `_accessor_io_handler.py:335`; not a declared dependency (`pyproject.toml:38-81`).
- **DEFERRED.md.** `_apply_post_to_master` (`:873`), the slot walker excluding `qc`/`plots`
  (`sdk_/_operation_tree.py:31`), and the dry-run size estimate (`_cli_interactive.py:236-250`)
  are as described.
- **CLAUDE.md rules.** The design keeps the metadata snapshot byte-for-byte, submits no
  scheduler job (`--test-only`), touches no vendored reference, and places the baseline probes
  (which import `phenotypic`) beside the plan rather than under `logic_validation_scripts/`.

## External Slurm claims

The reviewer could not reach `slurm.schedmd.com` (egress blocked). The following are therefore
**UNVERIFIED** against official documentation and must be confirmed in Task 13 as planned:

- `EnforcePartLimits` defaults to `NO`, and under `NO` a job exceeding the partition `MaxTime`
  is accepted and pends with reason `PartitionTimeLimit`. (Consistent with the reviewer's
  recollection.)
- `sbatch --test-only` validates the script and returns an estimated start time without
  submitting a job, and `sbatch` reads the script from standard input when no file is named.
  (Consistent with the reviewer's recollection; whether `--test-only` rejects a time limit
  over `MaxTime` when `EnforcePartLimits=NO` is not known.)
- A QOS with `Flags=PartitionTimeLimit` allows exceeding the partition limit (relevant to R11).
- `sinfo -p <unknown>` exit status (relevant to F18's premise).
