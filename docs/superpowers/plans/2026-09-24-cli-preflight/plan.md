# CLI Run Preflight: Implementation Plan

> **For agentic workers:** execute task by task, in order. Steps use checkbox (`- [ ]`)
> syntax for tracking. Every task starts with a test that fails on the base commit, and
> ends with a commit.

**Goal:** refuse incompatible configurations in the submitting shell, before the CLI
writes anything, using only the pipeline JSON, the options, the environment, the cluster
configuration and file headers. No image is run.

**Architecture:** `phenotypic_cli` is split into a read-only half, which ends with a new
`run_preflight` and the `--dry-run` exit, and a mutating half, which begins with the
restart clear and the overwrite delete. `run_preflight` collects `PreflightFinding`s from
small check functions, scoped to the pipeline slots the mode executes, and refuses on any
error. Operations state their own requirements through
`BaseOperation.preflight_requirements()`. Several runtime defects the survey found are
fixed at their source.

**Tech stack:** Python 3.12, click, pydantic v2, polars, tifffile, Pillow, pytest; `uv`
as the sole runner.

**Spec:** `docs/superpowers/specs/2026-09-24-cli-preflight/design.md`. Read it first; this
plan cites its sections (§n), findings (Fn) and decisions (Dn) rather than restating them.
Deferred items are in `DEFERRED.md` beside it.

**Evidence and review:** `docs/superpowers/reports/2026-09-24-cli-preflight/` holds the
claim-verification report and the independent review (`spec-plan-review.md`, R1-R38). This
revision of the plan incorporates every review finding; where a task changed because of one,
the step cites it. The probe scripts behind the claim-verification report are in
`baseline_probes/` beside this plan; they drive shipped code, which is why they live here
and not under `logic_validation_scripts/`.

## Global constraints

- **`uv` is the sole runner.** Never bare `python` or `pip`.
- **Explicit paths for ruff.** `uv run ruff check --fix <files you changed>`; never bare.
- **Explicit paths for git.** `git add <files>`; never `git add -A`.
- **Lazy entry points.** `_cli/_cli_preflight.py`, `_cli/_cli_input_headers.py`,
  `_cli/_metadata_preflight.py` and `abc_/_requirements.py` import nothing from
  `HEAVY_STARTUP_MODULES` or `DEFERRED_RUNTIME_MODULES` at module level. PIL, tifffile,
  polars, huggingface_hub and every detector import go inside the function that uses them.
  **No preflight code path may import `torch` or `micro_sam`** (spec §5). Guards:
  `tests/unit/ci/test_startup_imports.py`, `tests/unit/ci/test_deferred_imports.py`, and the
  subprocess test added in Task 8.
- **The preflight writes nothing.** No check may create, open for writing, or delete any
  path. Task 3 adds a write-tripwire test; every later task that adds a check re-runs it.
- **Checks are scoped by mode** (spec §0, D10). Every requirement check iterates
  `operations_in_scope(context)`, never the whole tree, and every check that reads an option
  runs only in the modes that read that option. Each task that adds a check includes at
  least one test proving that an out-of-scope slot or mode produces no finding.
- **Metadata semantics by schema ownership.** Never test a header with
  `startswith("Metadata_")`; use set membership on canonical headers or the
  `metadata_*_for_header` helpers (`CLAUDE.md`, Gotchas).
- **Fixtures are real images.** Any CLI-level test fixture that is scanned as input must be
  a decodable image written with `tifffile.imwrite` or `skimage.io.imsave`, never placeholder
  bytes. Once Task 10 lands, placeholder bytes are `PF-HEADER-UNREADABLE` errors (review R12).
- **Test scope follows `CLAUDE.md`'s table** (review R16). Per step: the step's own test. Per
  task: the files the task touches and their direct test files. **Per phase: the affected
  surface, once, at the phase gate**, derived mechanically from importers:
  `grep -rlE "<module path>|<symbol>" tests/` for every source module the phase changed,
  deduplicated. The full suite runs once, in Task 19. Use the `run-phenotypic-test` skill for
  anything above a single file. Run each failing test in isolation before attributing it.
- **Never `-n auto`.** Pass `-n "$SLURM_CPUS_PER_TASK"` only for a run expected to exceed a
  minute.
- **Operations.** Tasks 4, 8 and 12 edit operation classes; use the `adding-an-operation`
  skill before editing any of them.
- **Optional packages absent.** In the development environment `transformers`, `torch`,
  `sam2`, `huggingface_hub`, `micro_sam`, `fil_finder` and `astropy` may be absent. A test
  that must fail on the base commit may not be skipped for want of one of them; inject a fake
  module into `sys.modules` with `monkeypatch.setitem` instead (review R17).
- **Commit after every task**, ending the message with the session's attribution lines.

## Phase A: nothing mutates before validation

### Task 1: pin the ordering and overlap defects (F1, F2, F3, F27)

**Files:** create `tests/unit/cli/test_cli_preflight_ordering.py`.

- [ ] **Step 1.** Fixtures. `image_tree`: one dataset holding one 16 by 16 `uint8` RGB TIFF
  written with `tifffile.imwrite` (not the placeholder-byte fixture in
  `test_cli_gpu_refusal.py`, review R12). `valid_pipeline`:
  `ImagePipeline(ops={"det": OtsuDetector()}, meas={"size": MeasureSize()})` written with
  `path.write_text(pipeline.to_json())`. Use `write_text`, not `to_json(path)`: the latter
  appends `.pht-pipe` (claim-verification report §1).
- [ ] **Step 2.** Write the tests. Each asserts the exit code, the message, and the on-disk
  state, because each assertion alone passes on broken code.

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
  ```

  and, in the same style:
  - `test_overwrite_dry_run_previews_and_deletes_nothing`: exit 0; output contains
    "would delete"; `previous-run.txt` intact.
  - `test_restart_dry_run_leaves_machine_state_untouched`: run once for real with
    `--njobs 1` to create `.phenotypic/`; snapshot the bytes of every file under it; run
    `--restart --dry-run`; assert the snapshot is identical, `restart_epoch.json` included.
  - `test_overwrite_refuses_each_run_input_inside_the_output`, parametrized over `--input`,
    `--pipeline`, `--metadata` and `--image-manifest` placed under `--output`: exit != 0 and
    the file still exists (F3, F27; review R10).
  - `test_restart_refuses_a_run_input_inside_machine_state`: `--pipeline` placed under
    `.phenotypic/`, `--restart`: exit != 0 and the file still exists.
  - `test_overlap_refusal_survives_skip_validation`: the `--input` case with
    `--skip-validation` still refuses (spec §0, review R27).
- [ ] **Step 3.** Run `uv run pytest tests/unit/cli/test_cli_preflight_ordering.py -q`. Every
  test must **fail** on the base commit. Record the failure lines in the commit message.
- [ ] **Step 4.** Commit: `test(cli): pin that validation and dry-run precede every mutation`.

### Task 2: reorder `phenotypic_cli` (§1)

**Files:** modify `src/phenotypic/phenotypicCLI.py`, `src/phenotypic/_cli/_cli_identity.py`
(docstring only), `src/phenotypic/_cli/_cli_interactive.py`,
`src/phenotypic/_cli/_cli_validation.py` (comment only), `tests/unit/plotting/test_backends.py`
(docstring only).

- [ ] **Step 1.** In `phenotypic_cli`, move the following blocks, in this order, to sit
  immediately after the `--restart --image-manifest` freshness check and before the
  `if restart:` clear. Their bodies do not change. (The review confirmed every one is
  read-only and that nothing they need is computed below them.)
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
  for `--restart`, "would clear .phenotypic/ machine state (kept: …)". The kept names come
  from `_PRESERVED_ON_RESTART` (`sdk_/_io_constants.py:1313-1315`, review R32). It is private,
  so expose it through a small public accessor in `sdk_/_io_constants.py` rather than
  importing the underscore name across modules.
- [ ] **Step 3.** Extend the overlap refusal at `:1865-1883` as spec §1 specifies: under
  `--overwrite`, refuse `--input`, `--pipeline`, `--metadata` and `--image-manifest` paths
  that equal or lie inside `--output`; under `--restart`, refuse those paths inside
  `.phenotypic/` unless preserved. Canonicalize with `resolve(strict=False)` as the existing
  process-mode rule does. Keep the process-mode rule unchanged. Place the refusal outside the
  `--skip-validation` gate.
- [ ] **Step 4.** Rewrite the "gate finding F8" paragraph of `mint_run_identity`'s
  docstring. A dry run now exits before the mint, so the paragraph states that and drops
  the justification that relied on `clear_machine_state` having already run.
- [ ] **Step 5.** Remove the `full_validation` call from `execute_dry_run`
  (`_cli_interactive.py:206-217`); the main path has already validated. Delete
  `full_validation` itself if `grep -rn full_validation src tests` finds no other caller.
  Update the two texts that describe the double validation: the `_ANNOUNCED_PLOT_WARNINGS`
  comment (`_cli_validation.py:23-26`) and the docstring at `tests/unit/plotting/test_backends.py:307`
  (review R33).
- [ ] **Step 6.** Run Task 1's tests; all pass. Then run the directly affected files, derived
  by `grep -rlE "phenotypic_cli|execute_dry_run|full_validation|mint_run_identity|dry.run" tests/unit/cli tests/unit/plotting tests/integration/cli`.
  That set includes at least `test_cli_v2.py`, `test_cli_gpu_refusal.py`,
  `test_cli_image_manifest.py`, `test_cli_mode_contract.py`, `test_run_identity.py`,
  `test_scanner_stores.py`, `test_process_format_cli.py`, `test_cli_store_options.py`,
  `test_cli_metadata_startup.py`, `test_cli_provenance_original.py`, `test_schema_gate.py`,
  `test_embedded_measurement_migration.py` and `tests/unit/plotting/test_backends.py`
  (review R16). A failure that asserts the *old* order (for example, a test expecting
  `--restart --dry-run` to bump the epoch) is updated to the new contract, and the commit
  message names each such test. Any other failure is a regression; fix it.
- [ ] **Step 7.** Commit: `fix(cli): validate and exit dry runs before any output mutation`.

### Phase A gate

- [ ] Run the Phase A affected surface once (global constraints).
- [ ] An independent reviewer writes `phase-a-adherence.md` into the reports folder (see
  "Review gates" at the end).

## Phase B: the framework

### Task 3: `_cli_preflight.py`, scoping and wiring (§2)

**Files:** create `src/phenotypic/_cli/_cli_preflight.py`,
`tests/unit/cli/test_cli_preflight_core.py`; modify `src/phenotypic/_cli/_cli_validation.py`,
`src/phenotypic/phenotypicCLI.py`.

- [ ] **Step 1.** Failing tests for the types, the orchestrator and scoping:
  - `run_preflight` with a check that raises yields exactly one warning, `PF-CHECK-CRASHED`,
    whose message names the check function; nothing else fails.
  - `PreflightReport.errors` and `.warnings` partition `findings`.
  - `render` prints errors before warnings, and caps `subjects` at 20 with "… and N more".
  - Every value of the `FindingCode` `Literal` has a non-empty entry in `HINTS` (coverage
    over `typing.get_args`).
  - `operations_in_scope` for a pipeline holding one op in each of `ops`, `meas`, `post`,
    `filters` returns exactly the `MODE_SLOTS[mode]` subset for each of the three modes,
    including ops nested inside a `CompositeDetector` in `ops` and inside a nested
    `ImagePipeline`.
  - `load_pipeline_for_validation` returns `(pipeline, None)` for a valid file and
    `(None, finding)` for invalid JSON; `validate_pipeline` keeps its `(bool, str)` contract
    for existing callers (review R19).
  - **Write tripwire:** monkeypatch `builtins.open` (any mode containing `w`, `a`, `x` or
    `+`), `os.open` with write flags, `Path.mkdir`, `Path.touch`, `Path.unlink`,
    `shutil.rmtree`, `os.remove`, `os.rename`, `os.replace` and `Path.write_bytes`/`write_text`
    to raise; then call `run_preflight` on a context built from Task 1's fixtures. It must
    complete.
- [ ] **Step 2.** Implement the dataclasses, `MODE_SLOTS`, `operations_in_scope`,
  `load_pipeline_for_validation` and `run_preflight` as spec §2 specifies. Start with an empty
  `CHECKS` tuple.
- [ ] **Step 3.** Wire it into `phenotypic_cli` inside the `if not config.skip_validation`
  block: replace the `validate_pipeline` call with `load_pipeline_for_validation`; on a load
  finding, render a one-finding report and exit 1; otherwise build the `PreflightContext`
  from the loaded pipeline and call `run_preflight`. On errors, render and `sys.exit(1)`; on
  warnings, render and continue. The existing "Pipeline loading failed" wording stays in the
  rendered message, because Task 1 and `test_cli_gpu_refusal.py` assert it.
- [ ] **Step 4.** Run `test_cli_preflight_core.py`, Task 1's file, and
  `tests/unit/plotting/test_backends.py`.
- [ ] **Step 5.** Commit: `feat(cli): run preflight report, mode scoping and write tripwire`.

### Task 4: operations declare requirements (§3)

**Files:** create `src/phenotypic/abc_/_requirements.py`,
`tests/unit/abc_/test_preflight_requirements.py`; modify `abc_/_base_operation.py`,
`abc_/__init__.py` (export `OperationRequirements`, `WeightRequirement`),
`sdk_/mixin/_input_layer_mixin.py`, `abc_/_gpu_detector.py`, `enhance/_set_detect_mode.py`,
the six unconditional RGB classes listed in spec §3, and the gray-tolerant classes the
ratchet names.

- [ ] **Step 1.** Failing tests:
  - `OtsuDetector().preflight_requirements() == OperationRequirements()`.
  - The sixteen grid classes listed in spec §3 report `grid_image=True` (review R13);
    `RoundPeaksDetector`, `SinePeakDetector`, `RefineBySineFit`, `GridAlignmentRefiner` and
    `InoculumDetector` report `False`.
  - `ContrastGamma(input_layer="rgb")` reports `rgb_input=True` and the default reports
    `False`; the same for the three other `InputLayerMixin` classes.
  - `SetDetectMode(mode="red")` reports `rgb_input=True`; `mode="gray"` reports `False`.
  - Each unconditional RGB class reports `True`.
  - **Ratchet:** for every concrete `BaseOperation` subclass importable from the public
    subpackages whose defining module's source contains `.rgb[` or `.color.`, assert
    `"_requires_rgb_input" in cls.__dict__`. At the base commit this names
    `BayesShrinkCorrector`, `VisuShrinkCorrector`, `DenoiseBlockMatch` and `MeasureSymZones`
    among others; the docstring states that the test is a heuristic.
  - `model_json_schema()` is unchanged for a sample of classes (class variables and methods
    are not fields; the review's probe confirmed this for pydantic).
- [ ] **Step 2.** Implement spec §3. Set `_requires_rgb_input = False` explicitly on each
  gray-tolerant class the ratchet names, and on `PadImage`; read each class's RGB branch
  before choosing, and list in the commit message which classes were set to `False`, and
  why. GPU-detector overrides, including `MicroSamDetector`'s, are Task 8's; this task adds
  only the `GpuDetector` base rule (`rgb_input = self.input_layer == "rgb"`, review R14).
- [ ] **Step 3.** Run the new test file, `tests/unit/ci/test_startup_imports.py`,
  `tests/unit/ci/test_deferred_imports.py`, and the tune annotation-coverage gate named in
  the `adding-an-operation` skill.
- [ ] **Step 4.** Commit: `feat(abc): operations declare preflight requirements`.

### Phase B gate

- [ ] Run the Phase B affected surface once. Changing `BaseOperation` reaches almost every
  operation test, so derive the set by importers of `abc_/_base_operation.py`,
  `abc_/__init__.py`, and each edited class, and run it with the `run-phenotypic-test`
  skill.

## Phase C: pipeline and environment checks

### Task 5: grid, preset and detector checks (§4)

**Files:** create `tests/unit/cli/test_preflight_pipeline_checks.py`; modify
`_cli/_cli_preflight.py`.

- [ ] **Step 1.** Failing tests, one per code, each with passing counterparts:
  - `PF-GRID-IMAGE`, `full` mode, `--image-type Image`: `RemoveGridOutliers` nested inside a
    `CompositeDetector` branch is an error (proves the tree walk), and so is
    `MeasureGridSpread` in `meas`. No finding under `GridImage`. In `process` mode,
    `MeasureGridSpread` in `meas` produces **no** finding (review R4). In `measure` mode the
    image class comes from each store's `phenotypic.image_class`: build two stores with
    `Image.save2zarr` and `GridImage.save2zarr`; with a grid measurer, the `Image` store is
    listed and the finding is a warning.
  - `PF-GRID-PRESET` (review R5): a preset with both `nrows` and `ncols` under
    `--image-type Image` in `full` mode is an error; only `nrows` set produces nothing; both
    set with a `GridFinder` already in `meas` produces nothing; `process` mode produces
    nothing.
  - `PF-NO-DETECTOR`: enhancer-only `ops` in `full` mode is an error under both image types;
    the same pipeline in `measure` and `process` mode produces nothing; a detector nested in
    a `CompositeDetector` satisfies it.
- [ ] **Step 2.** Implement the three check functions over `operations_in_scope`, and
  register them in `CHECKS`.
- [ ] **Step 3.** Run the new file and `test_cli_preflight_core.py` (tripwire included).
- [ ] **Step 4.** Commit: `feat(cli): preflight grid, preset and detector checks`.

### Task 6: duplicate JSON keys and `--bit-depth` (§10.4, §10.6)

**Files:** modify `_core/_pipeline_parts/_serializable_pipeline.py`, `phenotypicCLI.py`;
create `tests/unit/core/test_pipeline_duplicate_keys.py`.

- [ ] **Step 1.** Failing tests: `pipe_cfgs` holding `det`, `blur`, `det` makes `from_json`
  raise `ValueError` naming `pipe_cfgs.det`; a duplicate inside an op's `params` does the
  same; `--bit-depth 12` is a click usage error while `--bit-depth 16` parses to `int` 16.
- [ ] **Step 2.** Add an `object_pairs_hook` that tracks the JSON path and raises on a
  repeated key. Pass it at `_serializable_pipeline.py:279` and at any other `json.loads` of
  pipeline text that `grep -n "json.loads" src/phenotypic/_core/_pipeline_parts/` finds.
- [ ] **Step 3.** Change `--bit-depth` to `type=click.Choice(["8", "16"])` with a callback
  converting to `int`. Check that the GUI never emits another value
  (`grep -rn "bit-depth\|bit_depth" src/phenotypic/_gui/run_console/`).
- [ ] **Step 4.** Run the new file, the `tests/unit/core/` files that `grep -rl from_json`
  finds, and `tests/unit/cli/test_cli_v2.py`.
- [ ] **Step 5.** Commit: `fix(pipeline): refuse duplicate JSON keys; restrict --bit-depth`.

### Task 7: custom-op preload in every process (§10.2, §5)

**Files:** modify `_cli/_cli_preload.py`, `phenotypicCLI.py`, each worker function found in
Step 1, and `_cli/_cli_validation.py`; create `tests/_fakes/register_custom_detector.py`
(copied from `baseline_probes/mods/my_custom_ops_reg.py`, with its definition module) and
`tests/unit/cli/test_cli_preload_everywhere.py`.

- [ ] **Step 1.** List every function under `src/phenotypic/_cli/` that calls
  `ImagePipeline.from_json`, `load_pipeline_for_validation`, `validate_pipeline` or
  `pipeline_requires_gpu`, and every entry module rendered into a SLURM script. Find the
  latter with `grep -rn "phenotypic\._cli\._cli_" src/phenotypic/_cli/`, which also matches
  the list-form spelling at `_cli_slurm_array_scripts.py:218` that `grep "\-m phenotypic"`
  misses (review R20). A function needs the preload if it can run in a process other than
  the main CLI process: a SLURM entry `main`, or a callable handed to joblib (the review
  found `process_single_image_core`, the measure-mode core, the process-only core, and the
  staged Stage-1 and Stage-3 callables at `_cli_staged_strategy.py:214`, `:351`). Write the
  list into the commit message.
- [ ] **Step 2.** Failing tests:
  - **The main process**: `python -m phenotypic --dry-run` run as a **subprocess** with
    `PHENOTYPIC_PRELOAD_MODULES` set validates. It must be a subprocess, because the existing
    live test patched the submitter in-process and hid the bug.
  - **Local workers** (review R2): the same custom pipeline run as a subprocess with
    `--njobs 2` over two images reports `Completed: 2/2` and exits 0.
  - **Missing registration**: with the variable unset, the dry run exits 1 with
    `PF-CUSTOM-OP`, and the hint names both `PHENOTYPIC_PRELOAD_MODULES` and
    self-registration.
  - **Idempotence**: calling the preload twice imports each module once.
  - **Every listed function** calls the preload before its first `from_json`: patch
    `preload_custom_operation_modules` to record, and `from_json` to assert the record exists.
- [ ] **Step 3.** Implement: make the preload idempotent, then call it at each site from
  Step 1. In `load_pipeline_for_validation`, catch the "not found in phenotypic namespace"
  `AttributeError` specifically and return the `PF-CUSTOM-OP` finding.
- [ ] **Step 4.** Run the new file, `tests/unit/cli/test_cli_preload.py`,
  `tests/unit/cli/test_cli_runtime_preload.py`, and `tests/integration/cli/test_staged_gpu_local.py`.
- [ ] **Step 5.** Commit: `fix(cli): honor PHENOTYPIC_PRELOAD_MODULES in every worker process`.

### Task 8: modules, licenses and weights (§5, §10.3)

**Files:** modify `detect/nn/_sam2.py`, `_sam3.py`, `_dinosam2_detector.py`,
`_fssdino_detector.py`, `_insid3_detector.py`, `_microsam_detector.py`,
`_helper/_dino_support.py`, `_helper/_checkpoint_manager.py`,
`detect/_filfinder_detector.py`, `_cli/_cli_preflight.py`; create
`tests/unit/detect/nn/test_nn_preflight_requirements.py`,
`tests/unit/cli/test_preflight_environment_checks.py`.

- [ ] **Step 1.** Failing tests for the runtime fixes, using fake modules injected with
  `monkeypatch.setitem(sys.modules, ...)` so they cannot skip (review R17):
  - `Sam3()._ensure_model_loaded()` with the license variable unset raises the license
    `RuntimeError` **before** `transformers.Sam3Model.from_pretrained` is reached (the fake
    `from_pretrained` fails the test if called).
  - The DINOv3 paths call `download(interactive=False)`: patch
    `Dinov3CheckpointManager.download` to record its kwargs, and `builtins.input` to fail the
    test.
- [ ] **Step 2.** Failing tests for the cache resolver (review R8):
  - `torch_hub_checkpoint_dir()` follows `TORCH_HOME`, then `XDG_CACHE_HOME`, then
    `~/.cache`, and never imports `torch` (assert `"torch" not in sys.modules` in a
    subprocess).
  - When torch *is* installed (skip otherwise; this is a pin, not the task's failing test),
    it equals `Path(torch.hub.get_dir()) / "checkpoints"`.
  - `Sam2CheckpointManager.cache_dir()` and `MicroSamCheckpointManager`'s cache directory
    both call the shared resolver, so the probe and the runtime agree by construction.
  - A subprocess running `run_preflight` on a `Sam2` pipeline and on a `MicroSamDetector`
    pipeline leaves `torch` and `micro_sam` out of `sys.modules`.
- [ ] **Step 3.** Failing tests for requirements: each detector in spec §3's table reports
  the modules, extra and weights listed there, and the model string follows its fields
  (`Sam2(model_size="large")` gives `sam2:large`; `Sam2(checkpoint=...)` gives no weights).
  `MicroSamDetector(input_layer="rgb")` reports `rgb_input=True` (review R14).
- [ ] **Step 4.** Failing tests for the three environment checks with `find_spec` and the
  resolver patched: `PF-MISSING-MODULE` is an error naming the extra (or conda for
  `micro_sam`); `PF-LICENSE` is an error when the key is absent from
  `PHENOTYPIC_ACCEPT_MODEL_LICENSE`; `PF-WEIGHTS-UNCACHED` is a warning; an `is_cached`
  returning `None` yields no weights finding. Each check runs only over
  `operations_in_scope`.
- [ ] **Step 5.** Implement §10.3, the resolver, the overrides, and the three checks. For
  Hugging Face probes use `huggingface_hub.try_to_load_from_cache(repo_id, "config.json")`,
  imported inside the probe.
- [ ] **Step 6.** Run the new files, the existing `tests/unit/detect/nn/` files that import
  the touched modules (derive them by grep), and the tripwire test.
- [ ] **Step 7.** Commit: `fix(nn): never prompt in a pipeline; gate SAM3; preflight deps and weights`.

### Phase C gate

- [ ] Run the Phase C affected surface once, then the independent review (`phase-c-adherence.md`).

## Phase D: inputs, metadata and post

### Task 9: RAW decoding and its continuation fence (§10.1, §12)

**Files:** modify `_core/_image_parts/_image_io_handler.py`, `_cli/_cli_failure_tracker.py`,
`pyproject.toml` (add `tifffile`), `uv.lock`; create
`tests/unit/core/test_imread_raw_routing.py`, `tests/unit/cli/test_work_id_raw_revision.py`,
and, if a redistributable sample is available, `tests/fixtures/raw/` with a `README.md`
recording the file's source and license.

- [ ] **Step 1.** Failing routing tests, from `baseline_probes/p5/probe5.py`: for each RAW
  suffix, `Image.imread` calls `rawpy.imread` for pixels and does not call
  `skimage.io.imread` (patch both). With `rawpy` patched to `None`, a RAW suffix raises
  `UnsupportedFileTypeError` whose message names `rawpy`. PNG, JPEG and TIFF still route to
  skimage. A caller's `rawpy_params` dict is unchanged after the call.
- [ ] **Step 2.** Failing fence tests (review R3, R18):
  - `_worker_work_identity(...)[0] == work_id_for_image(...)[0]` for a `.nef` input and for
    a `.tiff` input: both producers agree.
  - The `.nef` work id differs from its value at `81d19ec`; the `.tiff` work id equals its
    value at `81d19ec`. Both expected values are **literal digests**, computed once by
    running `compute_work_id` at `81d19ec` with fixed inputs and pasted into the test with a
    comment naming the commit. Computing them inside the test would be tautological.
- [ ] **Step 3.** Real-file decode (review R9): obtain a small camera RAW sample whose
  license permits redistribution, commit it under `tests/fixtures/raw/` with its provenance,
  and assert that `Image.imread` returns shape `(H, W, 3)`, dtype `uint16`, and a median
  intensity strictly between 1% and 99% of the dtype range. If no redistributable sample
  can be found, the test reads a path from `PHENOTYPIC_RAW_SAMPLE` and skips when it is
  unset, and the implementer runs it once on a real file and records the output in
  `docs/superpowers/reports/2026-09-24-cli-preflight/raw-decode.md`. The fix does not merge
  without one of the two.
- [ ] **Step 4.** Implement: test the RAW branch first; copy `rawpy_params` before popping;
  add `RAW_DECODE_REVISION = 2` beside `PROCESS_LAYER_SEMANTICS_REVISION`, and inside
  `compute_work_id` add `"raw_decode_revision"` to the payload only when the suffix of
  `relative_image_path` is a RAW suffix. Add `tifffile` to `[project] dependencies` with a
  lower bound equal to the version in `uv.lock`, then run `uv lock`.
- [ ] **Step 5.** Run the new files, `tests/unit/sdk_/test_metadata_io.py`,
  `tests/unit/cli/test_work_id_semantics_revision.py`, `tests/unit/cli/test_store_work_identity.py`
  and `tests/unit/cli/test_directory_scanner.py`.
- [ ] **Step 6.** Commit: `fix(io): decode RAW through rawpy; fence RAW work ids in compute_work_id`.

### Task 10: header checks and stem collisions (§7)

**Files:** create `_cli/_cli_input_headers.py`, `tests/unit/cli/test_preflight_input_headers.py`,
`docs/superpowers/reports/2026-09-24-cli-preflight/header-behavior.md`; modify
`_cli/_cli_preflight.py`.

- [ ] **Step 1: settle open questions 2 and 5 by probe.** With `uv run python`, write and
  `Image.imread` each of: an RGBA PNG, a palette PNG, an `LA` PNG, a 2-channel TIFF, a
  4-channel TIFF, a 3-page grayscale TIFF, a `uint16` TIFF with `bit_depth=8`, and a `uint8`
  PNG with `bit_depth=16`. Record each outcome (shape, dtype, `bit_depth`, or the exception)
  in `header-behavior.md`. Then run the CLI once on a dataset holding `a.png` and `a.tif`
  and record what happens (F28, review R29).
- [ ] **Step 2.** From Step 1, write the decoded-channel table (spec §7). `PF-CHANNELS` flags
  exactly the header shapes that raise; `PF-BIT-DEPTH` flags exactly the cases that raise or
  change data silently. `PF-STEM-COLLISION` is an error, or, if Step 1 showed a later stage
  already refusing it, moves that refusal earlier with the same wording.
- [ ] **Step 3.** Failing tests for `read_input_header(path) -> InputHeader | HeaderError`
  over PNG, JPEG, TIFF, a zero-byte file, a truncated PNG, and an OME-Zarr store written with
  `Image.save2zarr`. For the store, patch `zarr.open_array` to fail the test: the header path
  must not open an array. `InputHeader` records the decoded channel count **and** whether
  the file carries PhenoTypic metadata under `IO.PHENOTYPIC_METADATA_KEY` (for §8). A test
  compares the table's predicted channel count with `imread`'s actual result for every file
  from Step 1.
- [ ] **Step 4.** Failing tests for the codes, covering both severities and the scoping
  rules: every input gray under `--detect-mode red` is an error, one gray input among RGB
  ones is a warning listing that file; the same pair for `MeasureColor` in `full` mode;
  `MeasureColor` in `meas` under `process` mode produces nothing; `--detect-mode` checks
  produce nothing in `measure` mode; with `--sample 1` the checks still cover every input
  (review R26).
- [ ] **Step 5.** Implement with a `ThreadPoolExecutor(max_workers=HEADER_SCAN_WORKERS)`,
  where `HEADER_SCAN_WORKERS = 16` is a module constant. Open headers read-only, in `with`
  blocks.
- [ ] **Step 6.** Run the new file and the tripwire test.
- [ ] **Step 7.** Commit: `feat(cli): preflight input headers and stem collisions without decoding pixels`.

### Task 11: metadata (§9, §10.5)

**Files:** create `_cli/_metadata_preflight.py`, `tests/unit/cli/test_metadata_preflight.py`;
modify `_cli/_metadata_join.py`, `_cli/_embedded_measurement_tables.py`,
`_cli/_cli_output_manager.py`, `_gui/run_console/_request_safety.py`, `phenotypicCLI.py`,
`_cli/_cli_preflight.py`.

- [ ] **Step 1: settle open question 4.** Determine whether a worker's embedded table
  carries metadata columns whose dtype comes from the CSV read, and whether aggregation
  compares those dtypes across stores (`_embedded_measurement_tables.py:83-86`,
  `project_embedded_measurement_table`, `_cli_parquet_agg.py`). Probe with a CSV whose first
  100 rows are null in one column and integers after. Record the answer in the commit
  message. If a continuation could mix the two rules, add a revision to the processing
  configuration digest for runs with `--metadata`, following Task 9's pattern.
- [ ] **Step 2.** Failing test for F22: a CSV whose first 150 rows hold integers in a key
  column and whose row 151 holds `"A7"`; through the worker path it must not raise
  `ComputeError`.
- [ ] **Step 3.** Add `read_metadata_csv(path)` (`infer_schema_length=None`) to
  `_metadata_join.py` and route the four readers of spec §10.5 through it. Replace the pandas
  parse at `phenotypicCLI.py:2149-2160` with a `read_metadata_csv` call in the same place,
  outside the skippable block (review R22), keeping its error text and zero-rows warning.
- [ ] **Step 4.** Failing tests for `analyze_metadata_join` and the seven `PF-META-*` codes,
  each with a passing counterpart. They must include the review's two plate-map shapes
  (review R1): a map keyed on `ImageName + Grid_RowNum + Grid_ColNum` and a layout keyed only
  on `Grid_RowNum + Grid_ColNum`. Both produce **no error**, only `PF-META-UNVERIFIED`
  warnings. A CSV with no shared column and no measurement-level column is a
  `PF-META-NO-KEYS` error; `process` mode produces no metadata findings.
- [ ] **Step 5.** Move the analysis core out of `build_metadata_preflight` into
  `analyze_metadata_join`, carrying `_unverified_measurement_join_columns` with a docstring
  explaining that its `"_" in column` filter is a qualification test, not a prefix test
  (review R38). The GUI function delegates to it. The existing
  `tests/unit/gui/run_console/test_request_safety.py` must pass **unchanged**; that is the
  proof that the GUI's behavior did not move.
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
  `image.metadata.insert_metadata` writes (`_image_pipeline_core.py:1256-1257`,
  `_metadata_accessor.py:366-376`), and pin it with a test that runs `OtsuDetector` +
  `MeasureSize` on a freshly constructed `Image` (not an imported PhenoTypic file) and
  compares the frame's metadata headers with a function `intrinsic_metadata_headers()` that
  derives them from the schema. Add the CLI's `Metadata_Dataset` column when the dataset
  column is on (`_cli_output_manager.py:1764-1767`, review R6).
- [ ] **Step 2.** Failing tests:
  - Each post class's `required_columns()`; `JoinMetadata`'s is derived through the same
    `external_metadata_preserved_columns` / `ensure_metadata_prefix` rule as its
    `_normalized_table`, and a test pins that it names the frame spelling, not the table's.
  - `PF-POST-COLUMN` is an **error** for `AppendString(column="DoesNotExist")` when the set is
    provably complete; the **same case is a warning** when the tree contains a class defined
    outside `phenotypic`, when any input is an OME-Zarr store, or when any input's header
    carries PhenoTypic metadata.
  - No finding when the column comes from `--metadata`, from an intrinsic header, from
    `Metadata_Dataset`, or from an earlier `ExpandMetadata` in the chain.
  - `JoinMetadata.on` naming an unknown measurement header is a warning.
  - `process` mode produces nothing.
- [ ] **Step 3.** Implement. Compute measurement headers through
  `get_measurement_infoclasses()` and `get_headers()`, and skip any info class whose
  `get_headers` requires arguments (`TEXTURE`). Missing measurement coverage can only weaken
  the `JoinMetadata` warning; it never produces an error.
- [ ] **Step 4.** Run the two new files, `tests/unit/post/`, and the tripwire test.
- [ ] **Step 5.** Commit: `feat(cli): preflight post-measurement column requirements`.

### Phase D gate

- [ ] Run the Phase D affected surface once, then the independent review (`phase-d-adherence.md`).

## Phase E: cluster, output and GUI

### Task 13: cluster checks (§6)

**Files:** create `docs/superpowers/plans/2026-09-24-cli-preflight/slurm_behavior_probe.sh`,
`tests/unit/cli/test_preflight_slurm_checks.py`; modify `phenotypicCLI.py`,
`_cli/_cli_preflight.py`, `_cli/_cli_staged_slurm.py`, `_cli/_cli_execution_strategies.py`,
`_cli/_cli_interactive.py`.

- [ ] **Step 1: settle open question 3.** Commit a script that, on a login node, prints
  `scontrol show config | grep -i EnforcePartLimits`, then pipes
  `printf '#!/bin/bash\n#SBATCH --partition=<p>\n#SBATCH --time=99-00:00:00\ntrue\n'` to
  `sbatch --test-only`, then repeats with `--partition=does-not-exist` and with
  `--gpus-per-node=1` on a CPU partition, and runs `sinfo -p does-not-exist --Format=gres
  --noheader; echo "exit=$?"`. Record the outputs in
  `docs/superpowers/reports/2026-09-24-cli-preflight/slurm-behavior.md`. **Without cluster
  access**, record that the step was not run, leave open question 3 open, and continue: the
  design reads the live setting and does not depend on the answer (review R11).
- [ ] **Step 2.** Failing tests, with `subprocess.run` patched as in
  `tests/unit/cli/test_cli_slurm_array.py:66-130`:
  - `--gpu-slurm time=banana` is a usage error at parse time, even with `--skip-validation`.
  - The script piped to `sbatch --test-only` starts with `#!/bin/bash`, uses `/dev/null` log
    paths, and is sent with `env=sbatch_submission_environment()` (review R15).
  - A nonzero exit is `PF-SBATCH-REJECTED` carrying stderr; a missing `sbatch`, a timeout, and
    stderr matching a controller-communication pattern are each the `PF-SBATCH-UNAVAILABLE`
    warning.
  - A staged GPU run tests both profiles, and the GPU profile carries `--gpus-per-node=1`.
  - `PF-TIME-OVER-PARTITION` (review R11): parser cases for `MaxTime=UNLIMITED`,
    `MaxTime=2-00:00:00`, a comma-separated partition list (the tightest limit wins), and no
    `slurm_partition` (the `Default=YES` partition); the finding is a warning, and its text
    differs between `EnforcePartLimits=NO` and `ALL`.
  - `staged_slurm_limit_errors(2, 1000, 1)` and `(10, 4, 8)` return the two existing messages
    verbatim, and the strategy now calls the function.
  - The shared GRES function reports an `sinfo` nonzero exit as an unknown partition, not as
    "no GPUs".
  - The dry-run preview prints `format_sbatch_directives` output.
- [ ] **Step 3.** Implement. Every subprocess call has an explicit timeout (30 s for
  `sbatch`, 10 s for `scontrol` and `sinfo`).
- [ ] **Step 4.** Run the new file, `test_cli_slurm_array.py`, `test_staged_slurm_scripts.py`,
  `test_slurm_process_only_scripts.py`, and `tests/unit/sdk_/test_slurm_time.py`.
- [ ] **Step 5.** Commit: `feat(cli): preflight SLURM profiles with sbatch --test-only`.

### Task 14: output location and node-local inputs (§9)

**Files:** modify `_cli/_cli_preflight.py`; create
`tests/unit/cli/test_preflight_output_location.py`.

- [ ] **Step 1.** Failing tests:
  - An unwritable nearest ancestor is an error (skip on Windows and as root, where
    `os.access` ignores mode bits).
  - In `full` mode, free space patched below the input total is a warning whose text says
    "heuristic"; in `process` and `measure` mode there is no space finding (review R21).
  - With `/proc/self/mounts` content patched, on a SLURM run: an output on `tmpfs` or `ext4`
    is a warning; on `gpfs`, `lustre` or `nfs4` it is not; `--input`, `--pipeline`,
    `--metadata` and a `JoinMetadata` table on `tmpfs` are each warned about (F29, review
    R30); a local run produces nothing; a non-Linux platform produces nothing.
- [ ] **Step 2.** Implement: the longest mount-point prefix of the resolved path decides its
  filesystem type. Run the new file and the tripwire test.
- [ ] **Step 3.** Commit: `feat(cli): preflight output writability, space and node-local paths`.

### Task 15: GUI Validate forwards cluster options (§11)

**Files:** modify `_gui/run_console/_callbacks.py` and/or `_gui/run_console/_state.py`;
extend `tests/unit/gui/run_console/test_slurm.py` or `test_state.py`;
`tests/e2e/gui/test_run_console_fake_slurm.py`.

- [ ] **Step 1.** Failing test: for a state whose mode is SLURM, the Validate argv contains
  the same `--slurm`, `--gpu-slurm` and `--gpu-shards` tokens as the Run argv, plus
  `--dry-run`.
- [ ] **Step 2.** Implement by reusing Run's argv construction (`_build_subprocess_argv`,
  `_gui/run_console/_slurm.py:177-210`); do not duplicate it.
- [ ] **Step 3.** Extend `_write_fake_slurm_bin` in the fake-SLURM e2e test so the fake
  `sbatch` exits 0 for `--test-only`, and add `sinfo` and `scontrol show partition`/`show
  config` fakes. Run the touched GUI unit files, and the e2e file if Playwright is available.
- [ ] **Step 4.** Confirm that no user-visible chrome changed, so `FEATURES.md` and
  `WORKFLOWS.md` need no edit. If anything did change, follow the `gui-tutorial-capture`
  skill.
- [ ] **Step 5.** Commit: `feat(gui): Validate checks the SLURM profile Run will submit`.

### Phase E gate

- [ ] Run the Phase E affected surface once, then the independent review (`phase-e-adherence.md`).

## Phase F: documentation and regression

### Task 16: documentation

**Files:** `docs/source/tutorials/pages/cli_batch_processing.md`,
`docs/source/tutorials/pages/cli_modes.md`, `docs/source/how_to/pages/slurm_pipelines.md`,
`docs/source/how_to/pages/gpu_detection_setup.md`, `docs/source/contrib_guide/gpu_detectors.md`,
`docs/source/tutorials/gui/04_run_local.md`, `docs/source/tutorials/gui/05_run_slurm.md`,
`src/phenotypic/_cli/CLAUDE.md`, and the `--skip-validation` and `--dry-run` help strings.

- [ ] **Step 1.** Add a "Run preflight checks" section to `cli_batch_processing.md`: a
  hand-written table with one row per finding code, its severity rule, and its remedy. Add a
  test that parses the page and fails when a `FindingCode` value is missing from the table or
  a table row names a code that no longer exists (review R37). Use the name "run preflight"
  throughout, and "GPU placement refusal" for the existing check (review R34).
- [ ] **Step 2.** Document, each with its one-line remedy: the self-registering contract for
  custom operations and `preflight_requirements()` for authors; that GPU detectors never
  prompt for licenses; **that SAM3 now requires `PHENOTYPIC_ACCEPT_MODEL_LICENSE=sam3`**
  (review R7); **that RAW input on Windows is now refused** (review R9); and **that a
  pipeline with a duplicate JSON key no longer loads, including for `--mode recompile`**
  (review R24).
- [ ] **Step 3.** In `_cli/CLAUDE.md`, record the read-only/mutating split of
  `phenotypic_cli`, the rule that a new mutation goes below the dry-run exit, the mode-scoping
  rule, and the rename of "the preflight" to "GPU placement refusal".
- [ ] **Step 4.** Build the docs with the project's docs build command, as used by
  `docs/superpowers/plans/2026-09-15-nested-gpu-staging/run_docs_build.sbatch`.
- [ ] **Step 5.** Commit: `docs: CLI run preflight, custom-op registration, license and RAW changes`.

### Task 17: close the evidence loop

- [ ] **Step 1.** Re-run the baseline probes that motivated a fix (`p1`, `p4`, `p5`, `p7`) and
  the review's reproductions for R1, R2 and R3, and append their new outputs to the
  claim-verification report under a heading naming the commit.
- [ ] **Step 2.** Correct the citation the review found in the claim-verification report
  (§8 cites `_serializable_pipeline.py:283`; the call is at `:279`, review R31), noting the
  correction in place.
- [ ] **Step 3.** Commit the updated reports.

### Task 18: final affected-surface run

- [ ] **Step 1.** Derive the union of all phase surfaces and run it once with the
  `run-phenotypic-test` skill. Fix what is this change's; record anything else in the
  regression report.

### Task 19: full regression, once

- [ ] **Step 1.** Submit the full sharded suite with the committed batch script
  `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`, after
  pointing its `WORKTREE` at this branch's checkout. Do not use `-x`.
- [ ] **Step 2.** Write the outcome to
  `docs/superpowers/reports/2026-09-24-cli-preflight/regression.md`: the commit, pass and
  fail counts, and each failure with its isolation result.
- [ ] **Step 3.** Commit the report.

## Review gates

At the end of Phases A, C, D and E, a reviewer who did not write the code reads the diff
against the spec and writes `phase-<x>-adherence.md` into
`docs/superpowers/reports/2026-09-24-cli-preflight/`. The report names each finding (Fn) the
phase claims to close and the test that fails when that fix is reverted. The reviewer proves
that last point by reverting the fix locally and running the test, and records the command
and its output.

## Coverage of F1-F29

| # | Task | Test that fails if the fix is reverted |
|---|---|---|
| F1 | 1, 2 | `test_overwrite_with_a_corrupt_pipeline_keeps_the_previous_run`, `test_overwrite_dry_run_previews_and_deletes_nothing` |
| F2 | 1, 2 | `test_restart_dry_run_leaves_machine_state_untouched` |
| F3, F27 | 1, 2 | `test_overwrite_refuses_each_run_input_inside_the_output`, `test_restart_refuses_a_run_input_inside_machine_state` |
| F4 | 5 | `PF-GRID-IMAGE` tests, with mode-scope counterparts |
| F5 | 5 | `PF-GRID-PRESET` tests, both-set and one-set cases |
| F6 | 5 | `PF-NO-DETECTOR` tests |
| F7 | 10 | `--detect-mode red` all-gray error and subset warning |
| F8 | 4, 10 | `MeasureColor` pair; per-class requirement tests |
| F9 | 8 | `PF-MISSING-MODULE` test |
| F10 | 8 | DINOv3 `download(interactive=False)` test |
| F11 | 8 | SAM3 gate-before-`from_pretrained` test, with a fake `transformers` |
| F12 | 8 | `PF-WEIGHTS-UNCACHED` test; no-torch-import subprocess test |
| F13 | 7 | subprocess `--dry-run` and `--njobs 2` tests |
| F14 | 6 | duplicate-key `from_json` tests |
| F15 | 6 | `--bit-depth 12` usage error |
| F16 | 13 | `--gpu-slurm time=banana` parse-time error |
| F17 | 13 | `PF-SBATCH-REJECTED` test |
| F18 | 13 | limit-function and GRES return-code tests |
| F19 | 13 | `PF-TIME-OVER-PARTITION` parser and severity tests |
| F20 | 15 | Validate argv test |
| F21 | 11 | `PF-META-*` tests, plate-map shapes included |
| F22 | 11 | 151-row CSV through the worker path |
| F23 | 12 | `PF-POST-COLUMN` error and warning cases |
| F24 | 9, 10 | routing tests; producer-equality and literal-digest tests; real-file decode; `PF-RAW-NO-RAWPY` |
| F25 | 14 | output-location tests |
| F26 | 13 | dry-run preview test |
| F28 | 10 | `PF-STEM-COLLISION` test |
| F29 | 14 | node-local input tests |
