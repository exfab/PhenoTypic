# CLI Preflight: Implementation Plan

> **For agentic workers:** execute task by task, in order. Steps use checkbox (`- [ ]`)
> syntax for tracking. Every task starts with a test that fails on the base commit, and
> ends with a commit.

**Goal:** refuse incompatible configurations in the submitting shell, before the CLI
writes anything, using only the pipeline JSON, the options, the environment, the cluster
configuration and file headers. No image is run.

**Architecture:** `phenotypic_cli` is split into a read-only half, which ends with a new
`run_preflight` and the `--dry-run` exit, and a mutating half, which begins with the
restart clear and the overwrite delete. `run_preflight` collects `PreflightFinding`s from
small check functions and refuses on any error. Operations state their own requirements
through `BaseOperation.preflight_requirements()`. Six runtime defects the survey found are
fixed at their source.

**Tech stack:** Python 3.12, click, pydantic v2, polars, tifffile, Pillow, pytest; `uv`
as the sole runner.

**Spec:** `docs/superpowers/specs/2026-09-24-cli-preflight/design.md`. Read it first; this
plan cites its sections (§n) and findings (Fn) rather than restating them. Deferred items
are in `DEFERRED.md` beside it.

**Evidence:** `docs/superpowers/reports/2026-09-24-cli-preflight/claim-verification.md`.
The probe scripts it ran are in `baseline_probes/` beside this plan. They drive shipped
code, which is why they live here and not under `logic_validation_scripts/`.

## Global constraints

- **`uv` is the sole runner.** Never bare `python` or `pip`.
- **Explicit paths for ruff.** `uv run ruff check --fix <files you changed>`; never bare.
- **Explicit paths for git.** `git add <files>`; never `git add -A`.
- **Lazy entry points.** `_cli/_cli_preflight.py`, `_cli/_metadata_preflight.py` and
  `abc_/_requirements.py` import nothing from `HEAVY_STARTUP_MODULES` or
  `DEFERRED_RUNTIME_MODULES` at module level. PIL, tifffile, polars, huggingface_hub and
  every detector import go inside the function that uses them. Guards:
  `tests/unit/ci/test_startup_imports.py`, `tests/unit/ci/test_deferred_imports.py`.
- **The preflight writes nothing.** No check may create, open for writing, or delete any
  path. Task 3 adds a test that runs the whole preflight under a filesystem-write tripwire.
- **Metadata semantics by schema ownership.** Never test a header with
  `startswith("Metadata_")`; use set membership on canonical headers or the
  `metadata_*_for_header` helpers (`CLAUDE.md`, Gotchas).
- **Test scope follows `CLAUDE.md`'s table.** Per step: the step's own test. Per task: the
  files the task touches and their direct test files. Per phase: the affected surface, once,
  derived from importers (`grep -rl "<module>" tests/`). The full suite runs once, in Task
  18. Use the `run-phenotypic-test` skill for anything above a single file.
- **Never `-n auto`.** Pass `-n "$SLURM_CPUS_PER_TASK"` only for a run expected to exceed a
  minute.
- **Operations.** Tasks 4, 8 and 12 edit operation classes; use the `adding-an-operation`
  skill before editing any of them.
- **Commit after every task**, ending the message with the session's attribution lines.

## Phase A: nothing mutates before validation

### Task 1: pin the ordering defects (F1, F2, F3)

**Files:** create `tests/unit/cli/test_cli_preflight_ordering.py`.

- [ ] **Step 1.** Copy the `image_tree` fixture pattern from
  `tests/unit/cli/test_cli_gpu_refusal.py` (a one-image TIFF tree under `tmp_path`), and add
  a `valid_pipeline` fixture:
  `ImagePipeline(ops={"det": OtsuDetector()}, meas={"size": MeasureSize()})` written with
  `path.write_text(pipeline.to_json())`. Use `write_text`, not `to_json(path)`: the latter
  appends `.pht-pipe` (claim-verification report §1).
- [ ] **Step 2.** Write the tests. Each one asserts the exit code, the message, and the
  on-disk state, because each assertion alone passes on broken code:

  ```python
  def test_overwrite_with_a_corrupt_pipeline_keeps_the_previous_run(image_tree, tmp_path):
      output_dir = tmp_path / "out"; output_dir.mkdir()
      (output_dir / "previous-run.txt").write_text("keep me")
      corrupt = tmp_path / "p.json"; corrupt.write_text("{ not json")
      result = CliRunner().invoke(phenotypic_cli, [
          "--pipeline", str(corrupt), "--input", str(image_tree),
          "--output", str(output_dir), "--overwrite"])
      assert result.exit_code != 0, result.output
      assert "Pipeline loading failed" in result.output
      assert (output_dir / "previous-run.txt").read_text() == "keep me"

  def test_overwrite_dry_run_previews_and_deletes_nothing(valid_pipeline, image_tree, tmp_path): ...
      # exit 0; output mentions "would delete"; previous-run.txt intact

  def test_restart_dry_run_leaves_machine_state_untouched(valid_pipeline, image_tree, tmp_path): ...
      # run once for real with --njobs 1 to create .phenotypic/, snapshot every file's
      # bytes under it, run --restart --dry-run, assert the snapshot is identical,
      # including restart_epoch.json

  def test_overwrite_refuses_an_input_inside_the_output(valid_pipeline, tmp_path): ...
      # --input out/images, --output out, --overwrite: exit != 0, images still exist
  ```
- [ ] **Step 3.** Run `uv run pytest tests/unit/cli/test_cli_preflight_ordering.py -q`.
  All four must **fail** on the base commit. Record the failure lines in the commit message.
- [ ] **Step 4.** Commit: `test(cli): pin that validation and dry-run precede every mutation`.

### Task 2: reorder `phenotypic_cli` (§1)

**Files:** modify `src/phenotypic/phenotypicCLI.py`, `src/phenotypic/_cli/_cli_identity.py`
(docstring only), `src/phenotypic/_cli/_cli_interactive.py`.

- [ ] **Step 1.** In `phenotypic_cli`, move the following blocks, in this order, to sit
  immediately after the `--restart --image-manifest` freshness check and before the
  `if restart:` clear. Their bodies do not change.
  1. The refusal half of the fresh-run contents check (`:2386-2410`). Compute
     `will_overwrite = overwrite and <non-ignorable entries exist>` here, and refuse here
     when entries exist without `--overwrite`. Leave the `shutil.rmtree` where it is, now
     guarded by `will_overwrite`.
  2. Scanning (`:2447-2484`), including `scan_store_outputs` for measure mode and
     `apply_image_manifest`.
  3. Validation (`:2492-2530`).
  4. `_display_execution_config` and the `--dry-run` exit (`:2537-2552`).
- [ ] **Step 2.** Before the dry-run exit, when `restart` or `will_overwrite` is set, print
  what the real run would do: for `--overwrite`, "would delete N entries under <output>";
  for `--restart`, "would clear .phenotypic/ machine state (terminal_failures.jsonl,
  restart_epoch.json and legacy-v2/ are kept)". The list of kept names comes from
  `clear_machine_state`'s own constant (`sdk_/_io_constants.py:1318-1364`), not a copy.
- [ ] **Step 3.** Extend the overlap refusal at `:1865-1883` so that it also fires, in
  every mode, when `overwrite` is set and the canonical input equals or lies inside the
  canonical output. Keep the `process`-mode rule as it is. Word the new message after the
  existing one.
- [ ] **Step 4.** Rewrite the "gate finding F8" paragraph of `mint_run_identity`'s
  docstring. A dry run now exits before the mint, so the paragraph states that and drops
  the justification that relied on `clear_machine_state` having already run.
- [ ] **Step 5.** Remove the `full_validation` call from `execute_dry_run`
  (`_cli_interactive.py:206-217`); the main path has already validated. Keep
  `full_validation` itself only if another caller exists (`grep -rn full_validation src`);
  otherwise delete it.
- [ ] **Step 6.** Run Task 1's tests; all four pass. Then run the directly affected CLI
  surface once:
  `uv run pytest tests/unit/cli/test_cli_v2.py tests/unit/cli/test_cli_gpu_refusal.py
  tests/unit/cli/test_cli_image_manifest.py tests/unit/cli/test_cli_mode_contract.py
  tests/unit/cli/test_run_identity.py tests/unit/cli/test_scanner_stores.py
  tests/unit/cli/test_process_format_cli.py tests/unit/cli/test_cli_store_options.py
  tests/unit/cli/test_cli_metadata_startup.py -q`.
  A failure that asserts the *old* order (for example, a test that expects `--restart
  --dry-run` to bump the epoch) is updated to the new contract, and the commit message names
  each such test. Any other failure is a regression; fix it.
- [ ] **Step 7.** Commit: `fix(cli): validate and exit dry runs before any output mutation`.

## Phase B: the framework

### Task 3: `_cli_preflight.py` and wiring (§2)

**Files:** create `src/phenotypic/_cli/_cli_preflight.py`,
`tests/unit/cli/test_cli_preflight_core.py`; modify `src/phenotypic/phenotypicCLI.py`.

- [ ] **Step 1.** Write failing tests for the types and orchestrator:
  - `run_preflight` with a check that raises yields exactly one warning whose code is
    `PF-CHECK-CRASHED`, whose message names the check function, and nothing else fails;
  - `PreflightReport.errors` and `.warnings` partition `findings`;
  - `render` prints errors before warnings, and caps `subjects` at 20 with
    "… and N more";
  - every value of the `FindingCode` `Literal` has a non-empty entry in a module-level
    `HINTS` mapping (a coverage test over `typing.get_args`);
  - **write tripwire:** monkeypatch `builtins.open` (mode containing `w`, `a`, `x` or `+`),
    `os.open` with write flags, `Path.mkdir`, `Path.unlink`, `shutil.rmtree`, `os.remove`
    and `Path.write_bytes`/`write_text` to raise, then call `run_preflight` on a context
    built from Task 1's fixtures. It must complete. Later tasks re-run this test after
    adding their checks, which is what enforces "the preflight writes nothing".
- [ ] **Step 2.** Implement the dataclasses and `run_preflight` exactly as §2 specifies.
  Start with an empty `CHECKS: tuple[Callable[[PreflightContext], list[PreflightFinding]], ...]`.
- [ ] **Step 3.** Wire it into `phenotypic_cli` directly after `validate_pipeline` succeeds,
  inside the `if not config.skip_validation` block. Load the pipeline once, and pass the
  same object to `PreflightContext`. On errors, render the report and `sys.exit(1)`; on
  warnings, render and continue.
- [ ] **Step 4.** Run `tests/unit/cli/test_cli_preflight_core.py` and Task 1's file.
- [ ] **Step 5.** Commit: `feat(cli): preflight report, orchestrator and write tripwire`.

### Task 4: operations declare requirements (§3)

**Files:** create `src/phenotypic/abc_/_requirements.py`,
`tests/unit/abc_/test_preflight_requirements.py`; modify `abc_/_base_operation.py`,
`abc_/__init__.py` (export `OperationRequirements`, `WeightRequirement`),
`sdk_/mixin/_input_layer_mixin.py`, `abc_/_gpu_detector.py`,
`enhance/_set_detect_mode.py`, and the six unconditional RGB classes listed in §3.

- [ ] **Step 1.** Failing tests:
  - `OtsuDetector().preflight_requirements() == OperationRequirements()`;
  - each of the 13 grid classes in the spec's background (`FilamentousFungiDetector` …
    `MeasureGridSpread`) reports `grid_image=True`, and so do the three `GridFinder`
    subclasses; `RoundPeaksDetector`, `SinePeakDetector`, `RefineBySineFit`,
    `GridAlignmentRefiner` and `InoculumDetector` report `False`;
  - `ContrastGamma(input_layer="rgb")` reports `rgb_input=True` and the default reports
    `False`;
  - `SetDetectMode(mode="red")` reports `rgb_input=True`, `mode="gray"` reports `False`;
  - each unconditional RGB class reports `True`;
  - **ratchet:** iterate every concrete subclass of `BaseOperation` importable from the
    public subpackages. For each class whose defining module's source contains `.rgb[` or
    `.color.`, assert `"_requires_rgb_input" in cls.__dict__`. The docstring states that
    this is a heuristic;
  - `import phenotypic.abc_` in a fresh subprocess does not import `torch`, `PIL`, or any
    `DEFERRED_RUNTIME_MODULES` entry.
- [ ] **Step 2.** Implement §3. Set `_requires_rgb_input = False` explicitly on
  `BayesShrinkCorrector`, `VisuShrinkCorrector`, `PadImage`, and on any other class the
  ratchet names that tolerates gray; read each class's RGB branch before choosing, and say
  in the commit message which classes were set to `False`, and why.
- [ ] **Step 3.** Confirm that `model_json_schema()` output is unchanged for a sample of
  classes: class variables and methods are not fields. Assert it in the test file.
- [ ] **Step 4.** Run the new test file, `tests/unit/ci/test_startup_imports.py`,
  `tests/unit/ci/test_deferred_imports.py`, and the tune annotation-coverage gate named in
  the `adding-an-operation` skill.
- [ ] **Step 5.** Commit: `feat(abc): operations declare preflight requirements`.

## Phase C: pipeline and environment checks

### Task 5: grid, preset and detector checks (§4)

**Files:** create `tests/unit/cli/test_preflight_pipeline_checks.py`; modify
`_cli/_cli_preflight.py`.

- [ ] **Step 1: settle open question 1.** Run, with `uv run python`:
  `ImagePipeline(ops={"det": OtsuDetector()}, meas={"size": MeasureSize()}, nrows=8, ncols=12)`
  applied through `apply_and_measure` to a plain `Image` built from
  `load_synth_yeast_plate().rgb[:]`. If it raises `GridImageInputError` (possibly wrapped in
  `RuntimeError`), keep `PF-GRID-PRESET`. If it does not, delete that code from the spec's
  table and from the `FindingCode` `Literal`, and note the result in the commit message.
- [ ] **Step 2.** Failing tests, one per code, each with a passing counterpart:
  - `PF-GRID-IMAGE`: `--image-type Image` with `RemoveGridOutliers` nested inside a
    `CompositeDetector` branch (proves the tree walk), and with `MeasureGridSpread` in
    `meas`; no finding under `GridImage`;
  - `PF-NO-DETECTOR`: enhancer-only `ops` in `full` mode is an error, under both image
    types; the same pipeline in `measure` and `process` mode produces nothing; a detector
    nested in a `CompositeDetector` satisfies it;
  - `PF-GRID-PRESET` if kept.
- [ ] **Step 3.** Implement the three check functions and register them in `CHECKS`.
- [ ] **Step 4.** Run the new file and `test_cli_preflight_core.py` (tripwire included).
- [ ] **Step 5.** Commit: `feat(cli): preflight grid, preset and detector checks`.

### Task 6: duplicate JSON keys and `--bit-depth` (§10.4, §10.6)

**Files:** modify `_core/_pipeline_parts/_serializable_pipeline.py`,
`phenotypicCLI.py`; create `tests/unit/core/test_pipeline_duplicate_keys.py`.

- [ ] **Step 1.** Failing tests. The first uses
  `baseline_probes/`-style input: `pipe_cfgs` holding `det`, `blur`, `det`. `from_json`
  must raise `ValueError` naming `pipe_cfgs.det`. A second test puts a duplicate key inside
  an op's `params`. A third checks that `--bit-depth 12` is a click usage error, while
  `--bit-depth 16` parses to `int` 16.
- [ ] **Step 2.** Add an `object_pairs_hook` that tracks the JSON path and raises on a
  repeated key. Pass it at the `json.loads` call (`_serializable_pipeline.py:279`), and at
  any other `json.loads` of pipeline text found by
  `grep -n "json.loads" src/phenotypic/_core/_pipeline_parts/`.
- [ ] **Step 3.** Change `--bit-depth` to `type=click.Choice(["8", "16"])` with a
  callback converting to `int`. Check that the GUI never emits another value
  (`grep -rn "bit-depth\|bit_depth" src/phenotypic/_gui/run_console/`).
- [ ] **Step 4.** Run the new file plus `tests/unit/core/` files matching
  `*serializ*` and `*pipeline*json*`, and `tests/unit/cli/test_cli_v2.py`.
- [ ] **Step 5.** Commit: `fix(pipeline): refuse duplicate JSON keys; restrict --bit-depth`.

### Task 7: custom-op preload everywhere (§10.2, §5)

**Files:** modify `phenotypicCLI.py`, each worker entry point found in Step 1, and
`_cli/_cli_preflight.py`; create `tests/unit/cli/test_cli_preload_entrypoints.py`.

- [ ] **Step 1.** List every module under `src/phenotypic/_cli/` that calls
  `ImagePipeline.from_json`, `validate_pipeline`, or `pipeline_requires_gpu`, and every
  `python -m phenotypic._cli.<module>` entry point rendered into a SLURM script
  (`grep -rn "\-m phenotypic" src/phenotypic/_cli/`). A module needs the preload if it is
  an entry point and does not already call it. Write the list into the commit message.
- [ ] **Step 2.** Failing tests, using the self-registering module
  `baseline_probes/mods/my_custom_ops_reg.py` copied into `tests/_fakes/`:
  - `python -m phenotypic --dry-run` as a **subprocess** with
    `PHENOTYPIC_PRELOAD_MODULES` set validates. It must be a subprocess, because the
    existing live test patched the submitter in-process and hid the bug;
  - the same pipeline with the variable unset fails with `PF-CUSTOM-OP`, and the hint
    names both `PHENOTYPIC_PRELOAD_MODULES` and self-registration;
  - for each entry point from Step 1, `main` calls the preload before its first
    `from_json`. Assert this by patching `preload_custom_operation_modules` to record,
    and `from_json` to assert the record exists.
- [ ] **Step 3.** Implement. In `validate_pipeline`, catch the "not found in phenotypic
  namespace" `AttributeError` specifically and return the `PF-CUSTOM-OP` text.
- [ ] **Step 4.** Run the new file, `tests/unit/cli/test_cli_preload.py`,
  `tests/unit/cli/test_cli_runtime_preload.py`.
- [ ] **Step 5.** Commit: `fix(cli): honor PHENOTYPIC_PRELOAD_MODULES in every entry point`.

### Task 8: modules, licenses and weights (§5, §10.3)

**Files:** modify `detect/nn/_sam2.py`, `_sam3.py`, `_dinosam2_detector.py`,
`_fssdino_detector.py`, `_insid3_detector.py`, `_microsam_detector.py`,
`_helper/_dino_support.py`, `detect/_filfinder_detector.py`, `_cli/_cli_preflight.py`;
create `tests/unit/detect/nn/test_nn_preflight_requirements.py`,
`tests/unit/cli/test_preflight_environment_checks.py`.

- [ ] **Step 1.** Failing tests for the runtime fixes:
  - `Sam3()._ensure_model_loaded()` with the license variable unset raises the
    license `RuntimeError` **before** `from_pretrained` is reached (patch
    `transformers.Sam3Model.from_pretrained` to fail the test if called; skip if
    `transformers` is absent);
  - the DINOv3 paths call `download(interactive=False)`: patch
    `Dinov3CheckpointManager.download` to record its kwargs, and patch `builtins.input` to
    fail the test.
- [ ] **Step 2.** Failing tests for requirements: each detector in §3's table reports the
  modules, extra and weights listed there, and the model string follows its fields
  (`Sam2(model_size="large")` → `sam2:large`; `Sam2(checkpoint=...)` → no weights).
- [ ] **Step 3.** Failing tests for the three environment checks, with `find_spec` and
  `is_cached` patched: `PF-MISSING-MODULE` is an error that names the extra (or conda for
  `micro_sam`); `PF-LICENSE` is an error when the key is absent from
  `PHENOTYPIC_ACCEPT_MODEL_LICENSE`; `PF-WEIGHTS-UNCACHED` is a warning; and an
  `is_cached` returning `None` yields no weights finding.
- [ ] **Step 4.** Implement §10.3, the overrides, and the three checks. For Hugging Face
  cache probes use `huggingface_hub.try_to_load_from_cache(repo_id, "config.json")`,
  imported inside the probe.
- [ ] **Step 5.** Run the two new files, the existing `tests/unit/detect/nn/` files that
  import the touched modules (derive them by grep), and the tripwire test.
- [ ] **Step 6.** Commit: `fix(nn): never prompt in a pipeline; gate SAM3; preflight deps and weights`.

## Phase D: inputs, metadata and post

### Task 9: RAW decoding and its continuation fence (§10.1, §12)

**Files:** modify `_core/_image_parts/_image_io_handler.py`,
`_cli/_cli_failure_tracker.py`, `pyproject.toml` (add `tifffile`); create
`tests/unit/core/test_imread_raw_routing.py`,
`tests/unit/cli/test_work_id_raw_revision.py`.

- [ ] **Step 1.** Failing tests from `baseline_probes/p5/probe5.py`: for each RAW suffix,
  `Image.imread` calls `rawpy.imread` for pixels and does not call `skimage.io.imread`
  (patch both). With `rawpy` patched to `None`, a RAW suffix raises
  `UnsupportedFileTypeError` whose message names `rawpy`. PNG, JPEG and TIFF still route to
  skimage.
- [ ] **Step 2.** Failing tests for the fence: `work_id_for_image` for a `.nef` input
  differs from the value computed with `RAW_DECODE_REVISION` removed, while a `.tiff`
  input's work id is byte-identical to the base commit's. Compute that base value inside
  the test from `compute_work_id`'s inputs, not from a pasted string.
- [ ] **Step 3.** Implement. Reorder the branches so that RAW is tested first. Add
  `RAW_DECODE_REVISION = 2` beside `PROCESS_LAYER_SEMANTICS_REVISION`, and fold it into the
  digest input for RAW suffixes only. Add `tifffile` to `[project] dependencies` with a
  lower bound equal to the version currently in `uv.lock`, then run `uv lock`.
- [ ] **Step 4.** Run the new files, `tests/unit/sdk_/test_metadata_io.py`,
  `tests/unit/cli/test_cli_failure_tracker*.py` (whatever exists), and
  `tests/unit/cli/test_directory_scanner.py`.
- [ ] **Step 5.** Commit: `fix(io): decode RAW through rawpy; fence RAW work ids`.

### Task 10: header checks (§7)

**Files:** create `_cli/_cli_input_headers.py`,
`tests/unit/cli/test_preflight_input_headers.py`; modify `_cli/_cli_preflight.py`.

- [ ] **Step 1: settle open question 2.** With `uv run python`, write and `Image.imread`:
  an RGBA PNG, a palette PNG, a 2-channel TIFF, a 4-channel TIFF, a `uint16` TIFF read with
  `--bit-depth`-equivalent `bit_depth=8`, and a `uint8` PNG read with `bit_depth=16`.
  Record each outcome (shape, dtype, `bit_depth`, or the exception) in
  `docs/superpowers/reports/2026-09-24-cli-preflight/header-behavior.md`. `PF-CHANNELS` and
  `PF-BIT-DEPTH` then flag exactly the cases that raise or that change the data silently,
  and nothing else.
- [ ] **Step 2.** Failing tests for `read_input_header(path) -> InputHeader | HeaderError`
  over PNG, JPEG, TIFF, a zero-byte file, a truncated PNG, and an OME-Zarr store written
  with `Image.save2zarr`. For the store, patch `zarr.open_array` to fail the test: the
  header path must not open an array.
- [ ] **Step 3.** Failing tests for the six codes, covering both severities: every input
  gray under `--detect-mode red` is an error, while one gray input among RGB ones is a
  warning listing that file. Add the equivalent pair for `MeasureColor`.
- [ ] **Step 4.** Implement with a `ThreadPoolExecutor(max_workers=HEADER_SCAN_WORKERS)`,
  where `HEADER_SCAN_WORKERS = 16` is a module constant. Headers are read in `with` blocks,
  opened read-only.
- [ ] **Step 5.** Run the new file and the tripwire test.
- [ ] **Step 6.** Commit: `feat(cli): preflight input headers without decoding pixels`.

### Task 11: metadata (§9, §10.5)

**Files:** create `_cli/_metadata_preflight.py`,
`tests/unit/cli/test_metadata_preflight.py`; modify `_cli/_metadata_join.py`,
`_cli/_embedded_measurement_tables.py`, `_cli/_cli_output_manager.py`,
`_gui/run_console/_request_safety.py`, `phenotypicCLI.py`, `_cli/_cli_preflight.py`.

- [ ] **Step 1.** Failing test for F22: a CSV whose first 150 rows hold integers in a key
  column and whose row 151 holds `"A7"`. Through the worker path
  (`_embedded_measurement_tables`) it must not raise `ComputeError`.
- [ ] **Step 2.** Add `read_metadata_csv(path)` (`infer_schema_length=None`) to
  `_metadata_join.py`, and route the four readers named in §10.5 through it. The test from
  Step 1 passes.
- [ ] **Step 3.** Failing tests for `analyze_metadata_join` and the seven `PF-META-*`
  codes, each with a passing counterpart. For `PF-META-UNMATCHED`, assert that the listed
  subjects are the unmatched images.
- [ ] **Step 4.** Move the analysis core out of `build_metadata_preflight` into
  `analyze_metadata_join`, and make the GUI function delegate to it. The existing
  `tests/unit/gui/run_console/test_request_safety.py` must pass **unchanged**; that is the
  proof that the GUI's behavior did not move.
- [ ] **Step 5.** Delete the pandas parse at `phenotypicCLI.py:2149-2160`. Keep the
  zero-rows warning, re-expressed as a preflight warning.
- [ ] **Step 6.** Run the new file, `test_request_safety.py`,
  `tests/unit/cli/test_cli_output_manager.py`, `test_finalize_run.py`,
  `test_embedded_measurement_aggregation.py`, `test_metadata_namespace_compat.py`,
  `test_cli_metadata_startup.py`, and the tripwire test.
- [ ] **Step 7.** Commit: `feat(cli): metadata join preflight shared with the GUI; one CSV reader`.

### Task 12: post-measurement columns (§8)

**Files:** modify `abc_/_post_measurement.py` and the five post classes;
`_cli/_cli_preflight.py`; create `tests/unit/post/test_required_columns.py`,
`tests/unit/cli/test_preflight_post_columns.py`.

- [ ] **Step 1: establish the intrinsic metadata set.** Read what
  `image.metadata.insert_metadata` writes (`_image_pipeline_core.py:1256-1257`), and pin it
  with a test that runs `OtsuDetector` + `MeasureSize` on `load_synth_yeast_plate()` and
  compares the metadata headers in the frame against a function
  `intrinsic_metadata_headers()` that derives them from the schema. The function, not a
  literal list, is what the check uses.
- [ ] **Step 2.** Failing tests: each post class's `required_columns()`; then
  `PF-POST-COLUMN` fires for `AppendString(column="DoesNotExist")` and does not fire when
  the column comes from `--metadata`, from an intrinsic header, or from an earlier
  `ExpandMetadata` in the chain; and `JoinMetadata.on` naming an unknown header is a
  warning, not an error.
- [ ] **Step 3.** Implement. Compute measurement headers through
  `get_measurement_infoclasses()` and `get_headers()`, and skip any info class whose
  `get_headers` requires arguments (`TEXTURE`). Missing measurement coverage can only
  weaken the `JoinMetadata` warning; it never produces an error.
- [ ] **Step 4.** Run the two new files, `tests/unit/post/`, and the tripwire test.
- [ ] **Step 5.** Commit: `feat(cli): preflight post-measurement column requirements`.

## Phase E: cluster, output and GUI

### Task 13: cluster checks (§6)

**Files:** create `docs/superpowers/plans/2026-09-24-cli-preflight/slurm_behavior_probe.sh`,
`tests/unit/cli/test_preflight_slurm_checks.py`; modify `phenotypicCLI.py`,
`_cli/_cli_preflight.py`, `_cli/_cli_staged_slurm.py`, `_cli/_cli_execution_strategies.py`,
`_cli/_cli_interactive.py`.

- [ ] **Step 1: settle open question 3 on the cluster.** Commit a small script that runs,
  on a login node, `scontrol show config | grep -i EnforcePartLimits`, then
  `printf '#!/bin/bash\n#SBATCH --partition=short\n#SBATCH --time=99-00:00:00\ntrue\n' |
  sbatch --test-only`, then the same with `--partition=does-not-exist` and with
  `--gpus-per-node=1` on a CPU partition. Record the outputs in
  `docs/superpowers/reports/2026-09-24-cli-preflight/slurm-behavior.md`. Keep
  `PF-TIME-OVER-PARTITION` an error if `EnforcePartLimits` is `NO`; if it is enforced,
  make it a warning, since `--test-only` then covers it.
- [ ] **Step 2.** Failing tests, with `subprocess.run` patched as in
  `tests/unit/cli/test_cli_slurm_array.py:66-130`:
  - `--gpu-slurm time=banana` is a usage error at parse time, even with
    `--skip-validation`;
  - `sbatch --test-only` receiving the rendered script on stdin, with a nonzero exit, is
    `PF-SBATCH-REJECTED` carrying stderr; a missing `sbatch` is the
    `PF-SBATCH-UNAVAILABLE` warning; a timeout is the same warning;
  - a staged GPU run tests both profiles, and the GPU profile carries
    `--gpus-per-node=1`;
  - `staged_slurm_limit_errors(2, 1000, 1)` and `(10, 4, 8)` return the two existing
    messages verbatim, and the strategy now calls the function;
  - the shared GRES function reports an `sinfo` nonzero exit as an unknown partition, not
    as "no GPUs";
  - the dry-run preview prints `format_sbatch_directives` output.
- [ ] **Step 3.** Implement. Every subprocess call has an explicit timeout (30 s for
  `sbatch`, 10 s for `scontrol` and `sinfo`).
- [ ] **Step 4.** Run the new file, `test_cli_slurm_array.py`,
  `test_staged_slurm_scripts.py`, `test_slurm_process_only_scripts.py`, and
  `tests/unit/sdk_/test_slurm_time.py`.
- [ ] **Step 5.** Commit: `feat(cli): preflight SLURM profiles with sbatch --test-only`.

### Task 14: output location (§9)

**Files:** modify `_cli/_cli_preflight.py`; create
`tests/unit/cli/test_preflight_output_location.py`.

- [ ] **Step 1.** Failing tests: an unwritable nearest ancestor is an error (skip on
  Windows and when running as root, where `os.access` ignores mode bits); free space
  patched below the input total is a warning whose text says "lower bound"; a SLURM run
  whose output is under a patched `tempfile.gettempdir()` or `$SCRATCH` is a warning,
  while the same path on a local run produces nothing.
- [ ] **Step 2.** Implement; run the new file and the tripwire test.
- [ ] **Step 3.** Commit: `feat(cli): preflight output writability, space and node-local storage`.

### Task 15: GUI Validate forwards cluster options (§11)

**Files:** modify `_gui/run_console/_callbacks.py` and/or `_gui/run_console/_state.py`;
extend `tests/unit/gui/run_console/test_slurm.py` or `test_state.py`.

- [ ] **Step 1.** Failing test: for a state whose mode is SLURM, the Validate argv
  contains the same `--slurm`, `--gpu-slurm` and `--gpu-shards` tokens as the Run argv,
  plus `--dry-run`.
- [ ] **Step 2.** Implement by reusing Run's argv construction. Do not duplicate it.
- [ ] **Step 3.** Run the touched GUI test files and
  `tests/e2e/gui/test_run_console_fake_slurm.py` if the environment has Playwright. Its
  fake `sbatch` must accept `--test-only`: extend `_write_fake_slurm_bin` to exit 0 for
  it, and add `sinfo` and `scontrol show partition` fakes.
- [ ] **Step 4.** Confirm that no user-visible chrome changed, so `FEATURES.md` and
  `WORKFLOWS.md` need no edit. If anything did change, follow the `gui-tutorial-capture`
  skill.
- [ ] **Step 5.** Commit: `feat(gui): Validate checks the SLURM profile Run will submit`.

## Phase F: documentation and regression

### Task 16: documentation

**Files:** `docs/source/tutorials/pages/cli_batch_processing.md`,
`docs/source/tutorials/pages/cli_modes.md`,
`docs/source/how_to/pages/slurm_pipelines.md`,
`docs/source/how_to/pages/gpu_detection_setup.md`,
`docs/source/contrib_guide/gpu_detectors.md`,
`docs/source/tutorials/gui/04_run_local.md`, `docs/source/tutorials/gui/05_run_slurm.md`,
`src/phenotypic/_cli/CLAUDE.md`, the `--skip-validation` and `--dry-run` help strings.

- [ ] **Step 1.** Add a "Preflight checks" section to `cli_batch_processing.md` with one
  row per finding code, generated from `HINTS`. Add a test that fails when a code is
  missing from the page.
- [ ] **Step 2.** Document the self-registering contract for custom operations and
  `preflight_requirements()` for authors, and the fact that GPU detectors no longer prompt
  for licenses.
- [ ] **Step 3.** In `_cli/CLAUDE.md`, record the read-only/mutating split of
  `phenotypic_cli` and the rule that a new mutation goes below the dry-run exit.
- [ ] **Step 4.** Build the docs with the project's docs build command, as used by
  `docs/superpowers/plans/2026-09-15-nested-gpu-staging/run_docs_build.sbatch`.
- [ ] **Step 5.** Commit: `docs: CLI preflight checks and custom-op registration`.

### Task 17: phase-level surface run

- [ ] **Step 1.** Derive the affected surface from importers of every module this plan
  touched (`grep -rl` over `tests/` for each module path), deduplicate, and run it once with
  the `run-phenotypic-test` skill.
- [ ] **Step 2.** Run each failure in isolation before attributing it. Fix what is this
  change's; record anything else in the report below.
- [ ] **Step 3.** Re-run the four baseline probes that motivated a fix (`p1`, `p4`, `p5`,
  `p7`) and append their new outputs to the claim-verification report, under a heading
  naming the commit.

### Task 18: full regression, once

- [ ] **Step 1.** Submit the full sharded suite with the committed batch script
  `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`, after
  pointing its `WORKTREE` at this branch's checkout. Do not use `-x`.
- [ ] **Step 2.** Write the outcome to
  `docs/superpowers/reports/2026-09-24-cli-preflight/regression.md`: the commit, pass and
  fail counts, and each failure with its isolation result.
- [ ] **Step 3.** Commit the report.

## Review gates

After Phase A, after Phase C, and after Phase E, a reviewer who did not write the code
reads the diff against the spec and writes `spec-adherence.md` into
`docs/superpowers/reports/2026-09-24-cli-preflight/`. The report must name each finding
(Fn) the phase claims to close, and the test that fails when that fix is reverted. The
reviewer proves that last point by reverting the fix locally and running the test.
</content>
</invoke>
