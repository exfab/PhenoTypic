# Phase C adherence review: CLI run preflight (Tasks 5-8)

- **Reviewer:** independent (did not write the code)
- **Reviewed at:** `b16c525` (detached `origin/claude/modest-mccarthy-jz0ylw`)
- **Commits in scope:** `56dd369` (Task 5), `46a07ec` (Task 6), `d7ef296` (Task 7), `9040869` (Task 8)
- **Spec:** `docs/superpowers/specs/2026-09-24-cli-preflight/design.md` §3, §4, §5, §10.2, §10.3, §10.4, §10.6; F4-F6, F9-F15; D10
- **Plan:** `docs/superpowers/plans/2026-09-24-cli-preflight/plan.md`, Phase C
- **Environment:** `uv sync --group dev --group test-qt --extra gui`; `torch`, `transformers`,
  `sam2`, `micro_sam`, `huggingface_hub` absent. All pytest runs used
  `QT_QPA_PLATFORM=offscreen`, `-o addopts= -m "not slow"`, `-p no:pytest-qt` (no libEGL),
  and at most `-n 4`.

## Verdict

**Pass with changes.**

The four tasks do what the spec asks. Every check condition I traced matches the code it
guards, mode scoping goes through `operations_in_scope` everywhere, and every revert proof
failed the test it should. The Task 7 deviation (preloading inside class resolution instead
of at each call site) is sound, and better than the plan's approach, with the caveats in
C4-C6. No finding refuses a legitimate run.

Two Major findings need a fix before Phase F: the micro-sam cache probe looks in the wrong
directory, so every `MicroSamDetector` run with cached weights gets a false
`PF-WEIGHTS-UNCACHED` warning (C1). And the test that is supposed to keep `torch` and
`micro_sam` out of the submitting process passes against the very regression it exists to
catch whenever those packages are not installed (C2).

## Findings

### Blocking

None.

### Major

**C1. The micro-sam weight probe looks in the wrong directory, so the warning fires on every cached micro-sam run.**

- *Evidence.* `microsam_cache_dir_without_import()` (`detect/nn/_helper/_checkpoint_manager.py:871`)
  returns `MICROSAM_CACHEDIR` (or `platformdirs.user_cache_dir("micro_sam")`), and
  `microsam_weight_requirement.is_cached` (`:948`) looks for `<cache>/<model_type>/` or
  `<cache>/*<model_type>*`. Upstream micro-sam downloads models through pooch into
  `os.path.join(microsam_cachedir(), "models")`, as a file named after the registry key
  (`micro_sam/util.py:177`, current `master`, fetched during this review). That is
  `<cache>/models/vit_b_lm`, which neither pattern matches. Repro:
  `MICROSAM_CACHEDIR=<dir>` with `<dir>/models/vit_b_lm` present →
  `MicroSamDetector().preflight_requirements().weights[0].is_cached()` returns **False**.
  The glob also errs the other way: when only `vit_b_lm` is cached at top level,
  `*vit_b*` reports `vit_b` as cached.
- *Root cause.* Spec §5 (`design.md:410`) requires that "the existing managers'
  `cache_dir()` methods are changed to call the same resolver, so the probe and the runtime
  agree by construction rather than by parallel code". Task 8 kept parallel code and copied
  `MicroSamCheckpointManager.cache_dir`'s *fallback* branch (`:412-425`), which already
  disagrees with that method's *primary* branch (`_get_default_model_folder()`, the models
  subfolder). The Sam2 half has the same parallel-code shape, but its resolver is correct
  (C7).
- *Impact.* This is a warning, not a refusal. It still sends a user who has pre-downloaded
  weights off to fix a problem they do not have. It also teaches users to ignore the
  preflight's warnings.
- *Fix.* Resolve the models folder as `<MICROSAM_CACHEDIR or platformdirs cache>/models`
  and test `(models / model_type).is_file()` by exact name, with no glob. Have
  `MicroSamCheckpointManager.cache_dir()`'s fallback call the same function. Add a test
  that builds the real `<cache>/models/<model_type>` layout and asserts `True`, plus one
  asserting that `vit_b` is not reported cached when only `vit_b_lm` exists.

**C2. The "never import `torch` or `micro_sam`" guard is vacuous wherever those packages are absent.**

- *Evidence.* `test_probing_requirements_imports_neither_torch_nor_micro_sam`
  (`tests/unit/detect/nn/test_nn_preflight_requirements.py:180`) checks `sys.modules` in a
  subprocess. When a package cannot be imported it never enters `sys.modules`, so a
  regression wrapped in `try/except ImportError` passes. Revert proof: I replaced the
  micro-sam probe body with `return MicroSamCheckpointManager.cache_dir()`, which imports
  `micro_sam.util` when it can (R8's exact regression). The test **passed** (`1 passed`).
  With a stub `micro_sam` package on `PYTHONPATH`, the same test failed with
  `AssertionError: ['micro_sam']`. The guard therefore depends on the CI environment, and
  CI lacks these packages. The plan (Task 8 Step 2) also asked for a subprocess running
  `run_preflight` on a Sam2 and a MicroSam pipeline. The test only calls
  `preflight_requirements()` and `is_cached()`.
- *Mitigating evidence.* I ran the whole `run_preflight` (all 14 checks) on a pipeline
  holding `Sam2`, `Sam3`, `DinoSam2Detector(dino_version=3)`, `FssDinoDetector`,
  `Insid3Detector`, `MicroSamDetector` and `FilFinderDetector`. Importable stub `torch`,
  `transformers`, `sam2`, `micro_sam`, `fil_finder` and `astropy` packages were on
  `PYTHONPATH`, and none was imported (`HEAVY: []`). So the code is correct today, and
  only the guard is weak.
- *Fix.* In the test, write stub packages for `torch`, `micro_sam`, `transformers` and
  `sam2` into `tmp_path`, prepend that directory to the subprocess `PYTHONPATH`, and run
  `run_preflight` on a context holding every nn detector. Assert that none of the stubs is
  in `sys.modules`. The guard then fails on any import, installed or not.

### Minor

**C3. `PF-CUSTOM-OP` and `PF-PIPELINE-LOAD` never reach the user as findings: the code and the hint are not printed.**

- *Evidence.* On a load finding, `phenotypic_cli` prints `"✗ Pipeline loading failed:"`
  and then `load_finding.message` only (`phenotypicCLI.py:2624-2629`). Actual dry-run
  output for an unregistered custom op:
  `Failed to load pipeline: UnknownOperationClassError: Class 'CustomThresholdDetector' not found ...`,
  with no `[PF-CUSTOM-OP]` and no `→` hint. Spec §2 says "the CLI renders a report holding
  that single finding; this is how `PF-CUSTOM-OP` and a JSON error reach the user through
  the same report format". The remedy still reaches the user because
  `UnknownOperationClassError`'s own message states it, so the practical impact is small.
  Still, `HINTS["PF-CUSTOM-OP"]` and `HINTS["PF-PIPELINE-LOAD"]` are dead text on the CLI
  path. The wiring is Task 3's (Phase B), but `PF-CUSTOM-OP` is Phase C's code.
- *Fix.* After the "Pipeline loading failed" line, print
  `PreflightReport((load_finding,)).render_lines()` (the Task 1 and GPU-refusal tests
  assert only the line, which stays). Extend
  `test_an_unregistered_custom_op_is_refused_with_the_remedy` to assert `PF-CUSTOM-OP` in
  the output.

**C4. The resolution-time preload runs only on a miss, so the same variable means different things in different processes.**

- *Evidence.* `_find_class_in_phenotypic` (`_serializable_pipeline.py:625`) imports the
  `PHENOTYPIC_PRELOAD_MODULES` modules only when a name fails to resolve. Some processes
  import them eagerly: the main CLI (`phenotypicCLI.py:1933-1935`), the staged SLURM
  worker, the checkpoint handler and the finalizer. Loky workers and the ordinary per-image
  array worker (`_cli_process_single`, which has no explicit call) import them only if the
  pipeline names a class that does not resolve. Suppose a registration module does more
  than attach a new name, for example rebinding a built-in name (`phenotypic.OtsuDetector =
  Patched`) or registering something a built-in class looks up. Then the main process and
  those workers run different code, silently. The plan's per-site approach would have
  preloaded unconditionally.
- *Assessment of the deviation as a whole.* It is sound, and better than the plan. Class
  resolution is the single step every deserializing process passes through, so no list of
  sites can go stale. The revert proof shows it is what makes loky workers work. On the
  other hazards: I found no recursion (a module that deserializes during its own import
  gets its partially initialized module back from `import_module` and then a clean
  `UnknownOperationClassError`; established by reading, not exercised). Thread safety
  follows from the import lock and the absence of shared mutable state. The miss path
  costs one environment parse plus `sys.modules` lookups.
- *Fix.* Preload once per process on the *first* call, hit or miss, behind a module-level
  flag set before importing, which also rules out re-entry. Keep the retry on a miss. Then
  every process honors the variable the way the main CLI does. Say so in the docstring and
  in spec §10.2.

**C5. The miss-path preload changes the contract of three other callers of `_find_class_in_phenotypic`.**

- *Evidence.* `_metadata_migration._serialized_class` (`sdk_/_metadata_migration.py:546-553`)
  is documented as resolving "a public serialized class **without importing custom code**".
  It now imports every listed module on any unknown `class` envelope during
  `--mode migrate`, and if one of them fails to import, migrate raises `ImportError` where
  it used to treat the envelope as opaque. `RemoveByFeature._validate_feature`
  (`refine/_remove_by_feature.py:118`) can now raise `ImportError` from a pydantic
  validator, which pydantic does not wrap in `ValidationError`. The GUI's
  `_recipe_state._resolved_analyzer_class` has the same exposure. Each case needs the
  variable set to a broken module, so this is Minor.
- *Fix.* Probe-style callers should call `_search_phenotypic_namespace` (no preload), or
  take a `preload=False` flag. Update the `_serialized_class` docstring either way.

**C6. The explicit startup preload is untested, and a bad module name prints a raw traceback in every mode.**

- *Evidence.* With `phenotypicCLI.py:1935` replaced by `pass`,
  `test_cli_preload_everywhere.py`, `test_cli_preload.py` and `test_cli_runtime_preload.py`
  all still pass (`17 passed`). The commit claims "a broken module name fails at startup
  with its own ImportError", and no test pins that. The call also sits outside the `try`,
  so `PHENOTYPIC_PRELOAD_MODULES=no_such_mod_xyz ... --dry-run` ends in a
  `ModuleNotFoundError` traceback. That happens in every mode, including `--mode migrate`,
  which never needed the variable before. In the GUI, Validate shows the traceback.
- *Fix.* Catch `ImportError` there and raise a `click.ClickException` that names the
  variable and the module. Add a subprocess test asserting exit 1 and the variable's name.

**C7. The spec still describes two things Task 8 did differently.**

- §5 says the managers' `cache_dir()` methods call the shared resolver. They do not. The
  commit message gives a good reason for Sam2: only torch sees `torch.hub.set_dir`. The
  spec was not updated, and C1 is the price of the parallel micro-sam code.
- The §3 table says "every `GpuDetector`: ... `torch`". The base
  `GpuDetector.preflight_requirements` (`abc_/_gpu_detector.py:177`) adds only the RGB
  rule. That is the right call, because a custom or fake GPU detector (for example
  `tests/_fakes/register_fake_gpu.py`) needs no torch, and requiring it would refuse
  `test_staged_gpu_local.py` in a torch-free environment. It is still a silent deviation.
- *Fix.* Update §3 and §5 to describe what shipped. For Sam2, record the known limit that
  the probe cannot see `torch.hub.set_dir`.

**C8. Four behaviors have no test (each mutation below survived).**

| Mutation | Tests run | Result |
|---|---|---|
| `check_grid_preset`: `mode == "process"` → `mode != "full"` (drops measure mode) | `tests/unit/cli -k preflight` | 115 passed |
| `check_detector_present`: slot-skip `continue` → `pass` (a detector in `meas` satisfies the check) | `tests/unit/cli -k preflight` | 115 passed |
| `_loads_rejecting_duplicate_keys`: list branch disabled (duplicate inside `CompositeDetector.ops[0]` accepted) | `test_pipeline_duplicate_keys.py` | 7 passed |
| no out-of-scope test for `PF-LICENSE` / `PF-WEIGHTS-UNCACHED` (plan global constraint) | (inspection) | n/a |

- *Fix.* Add four tests: `PF-GRID-PRESET` on a plain-`Image` store in measure mode;
  `ops={"blur": BlurGauss()}` with `meas={"z": MeasureSymZones(center_detector=OtsuDetector())}`
  → `PF-NO-DETECTOR`; a duplicate at `pipe_cfgs.c.params.ops[0].params.ignore_zeros`
  (I confirmed ad hoc that the walker reports `a[0].x`, `a[0][0].y` and `[0].k.z`
  correctly); and `Insid3Detector` in `ops` under measure mode → no `PF-LICENSE`.

### Nit

- **C9.** `check_model_licenses` re-implements `require_license_acceptance`'s parse of
  `PHENOTYPIC_ACCEPT_MODEL_LICENSE` (`_cli_preflight.py:484-488` vs
  `_checkpoint_manager.py:822-826`). The two agree today. Extract
  `accepted_model_licenses()` so the preflight and the runtime gate cannot drift
  ("one producer per derived value").
- **C10.** `_loads_rejecting_duplicate_keys` parses the text twice. That is harmless at
  pipeline size, but the walk could return plain dicts instead. D7 says the refusal is
  "library-wide", yet `BaseOperation.from_json` (single-op JSON via `read_json_source`)
  and the tune spec's embedded pipeline (`tune/_spec.py:231-233`, already parsed by
  `json.load`) still drop duplicates silently. Either narrow D7's wording or route those
  paths through the same loader.
- **C11.** The six nn `preflight_requirements` overrides lack a return annotation
  (`-> OperationRequirements`) and a Google-style `Returns:` section, and each docstring's
  second sentence is one line of more than 100 characters.
- **C12.** `check_optional_modules` groups by `(module, extra)`, so a pipeline with
  `Sam2` (extra `torch`) and `Sam3` (extra `foundation`) reports a missing `torch` twice.
  Group by module and list each extra.
- **C13.** `test_preload_is_idempotent` says "imports the module once" but asserts only
  `sys.modules` identity, which `import_module` guarantees anyway. Count executions with a
  module-level counter in a fake, or drop the claim.
- **C14.** `src/phenotypic/_cli/CLAUDE.md` ("Environment variables") still says only
  staged SLURM workers and the finalizer import the preload modules. Presumably Task 16's
  job, but the root `CLAUDE.md` gotcha says the same.
- **C15.** The spec §4 table gives `PF-GRID-PRESET` severity "error". In measure mode with
  a mix of store classes, the code emits a warning listing the plain stores. That is the
  correct application of §0 and D3, so update the table rather than the code.
- **C16.** UNVERIFIED (torch not installed): `torch_hub_checkpoint_dir` treats an empty
  `TORCH_HOME` as unset (`or`), while torch's own resolution uses
  `os.getenv("TORCH_HOME", default)`, which honors an empty string. This is an edge case,
  and the pin test that would catch it skips here.

## Revert-proof table

Every row: the hunk was reverted in the worktree only (never committed), the named test was
run, and then the file was restored with `git checkout -- <file>`. After all restores,
`git status` was clean, and the Phase C spot-check set passed: **361 passed, 24 skipped**.
Command, unless noted:
`QT_QPA_PLATFORM=offscreen uv run pytest -o addopts= -m "not slow" -q -p no:cacheprovider -p no:pytest-qt <test file>`.

| Finding | Hunk reverted | Test file | Before (reverted) | After (restored) |
|---|---|---|---|---|
| F5 | `check_grid_preset`: `return []` before `pipeline = context.pipeline` | `tests/unit/cli/test_preflight_pipeline_checks.py` | 1 failed: `test_a_preset_with_both_dimensions_under_plain_image_is_an_error` | 14 passed |
| F5 | `_nrows is None or _ncols is None` → `and` | same | 1 failed: `test_a_preset_with_one_dimension_injects_nothing` | 14 passed |
| F5 | `GridFinder`-in-`meas` guard disabled | same | 1 failed: `test_a_preset_with_a_grid_finder_already_in_meas_injects_nothing` | 14 passed |
| F6 | `check_detector_present`: `if context.mode != "full"` → `if True` | same | 2 failed: `test_a_forward_run_without_a_detector_is_an_error[Image]`, `[GridImage]` | 14 passed |
| F4 | `check_grid_image`: `return []` before `grid_ops` | same | 3 failed: nested op, grid measurer, measure-mode store class | 14 passed |
| F4 | `_image_class_by_input`: measure-mode per-store read disabled | same | 1 failed: `test_measure_mode_reads_each_stores_recorded_image_class` | 14 passed |
| F14 | `_loads_rejecting_duplicate_keys(json_data)` → `json.loads(json_data)` | `tests/unit/core/test_pipeline_duplicate_keys.py` | 2 failed: op-key and params duplicate tests | 7 passed |
| F15 | `--bit-depth` `type=click.Choice(["8","16"])` → `type=int` | same | 3 failed: `test_bit_depth_accepts_only_8_or_16[12]`, `[0]`, `[32]` | 7 passed |
| F13 | resolution-time preload disabled (`if not preload_module_names()` → `if True`) | `tests/unit/cli/test_cli_preload_everywhere.py` (`-n 4`) | 2 failed: `test_a_custom_op_runs_in_local_parallel_workers` ("Processing failed for plate1/img00x.tiff"), `test_class_resolution_preloads_in_a_process_that_never_ran_the_cli` | 6 passed |
| F13 | the above **plus** the startup preload at `phenotypicCLI.py:1935` | same | 3 failed: adds `test_a_custom_op_validates_in_the_main_cli_process` ("Pipeline loading failed") | 6 passed |
| F13 | startup preload only (`phenotypicCLI.py:1935` → `pass`) | same + `test_cli_preload.py` + `test_cli_runtime_preload.py` | **17 passed: not caught** (C6) | 17 passed |
| F10 | `_dino_support.load_dino_backbone`: `download(interactive=False)` → `download()` | `tests/unit/detect/nn/test_nn_preflight_requirements.py` | 1 failed: `test_dinov3_paths_never_prompt[load_dino_backbone]` | 15 passed, 1 skipped |
| F10 | `DinoSam2Detector._ensure_model_loaded`: `download(interactive=False)` → `download()` | same | 1 failed: `test_dinov3_paths_never_prompt[DinoSam2Detector]` | 15 passed, 1 skipped |
| F11 | `Sam3._ensure_model_loaded`: the `require_license_acceptance(..., interactive=False)` block removed | same | 1 failed: `test_sam3_refuses_an_unaccepted_license_before_loading` | 15 passed, 1 skipped |
| F9 | `check_optional_modules`: `return []` before the loop | `tests/unit/cli/test_preflight_environment_checks.py` | 2 failed: `test_a_missing_package_is_an_error_naming_its_extra`, `test_micro_sam_names_no_extra` | 10 passed |
| F12 | `check_model_weights_cached`: `if weight.is_cached() is False` → `if False` | same | 1 failed: `test_uncached_weights_warn_and_unknown_is_silent[False-codes0]` | 10 passed |
| F11/§5 | `check_model_licenses`: `return []` in place of `findings = []` | same | 1 failed: `test_an_unaccepted_gated_license_is_an_error` | 10 passed |
| R8 | micro-sam probe → `MicroSamCheckpointManager.cache_dir()` | `test_nn_preflight_requirements.py -k imports_neither` | **1 passed: not caught** (C2); with stub `micro_sam` on `PYTHONPATH`: 1 failed, `AssertionError: ['micro_sam']` | 1 passed |
| C8 rows | see C8 | see C8 | **not caught** | n/a |

## Regression spot checks

- The ten files and directories named in the brief (`test_preflight_pipeline_checks.py`,
  `test_preflight_environment_checks.py`, `test_cli_preload_everywhere.py`,
  `tests/unit/detect/nn/`, `test_pipeline_duplicate_keys.py`,
  `tests/unit/abc_/test_preflight_requirements.py`, `test_cli_gpu_refusal.py`,
  `tests/integration/cli/test_staged_gpu_local.py`, `test_cli_preflight_core.py`):
  **361 passed, 24 skipped** (`-n 4`, 51 s; 29 s on the final re-run).
- `tests/unit/ci` (startup and deferred import guards), `test_cli_preload.py`,
  `test_cli_runtime_preload.py`, `test_cli_v2.py`: **340 passed** (`-n 4`, 115 s).
- `ruff check` on every source and test file the four commits touched: clean.
  `mypy src/phenotypic/_cli/_cli_preflight.py src/phenotypic/sdk_/_preload.py`: clean.
- I did not re-run the Phase C gate's full 4680-test surface. The gate report's one
  failure (`test_migrate_end_to_end.py`, `NameError: read_metadata_csv`, explained as a
  mid-edit artifact of Task 11) is **UNVERIFIED** by me.

## Verified as correct

- **PF-GRID-PRESET** mirrors `_build_measurement_run_order`
  (`_image_pipeline_core.py:1294-1327`) exactly: both `_nrows` and `_ncols` must be set,
  there must be no `GridFinder` instance among `meas` values, and the check skips process
  mode. The CLI never applies `--nrows`/`--ncols` to the pipeline under `--image-type
  Image` (`_cli_process_single.py:262-275`), so the preset is the only trigger.
- **PF-NO-DETECTOR** fires only in full mode. It iterates `operations_in_scope`, skips
  paths whose first segment is a root slot (`pipeline_slot_of`), and accepts any
  `ObjectDetector` in root `ops` at any depth, including inside `CompositeDetector` and
  nested `ImagePipeline.ops`. It excludes a nested pipeline's `meas`, which the parent
  never measures. Every shipped object-producing class I enumerated (28 subclasses,
  including `GpuDetector`, `GridObjectDetector`, `ManualPointDetector`,
  `FilFinderDetector` and the composites) derives from `ObjectDetector`. No non-detector
  op writes an objmap from nothing. The hint names the custom-op exception.
- **PF-GRID-IMAGE**: full and process mode use `--image-type`, and process mode builds
  `GridImage` from it (`_cli_process_only.py:303`). Measure mode reads each store's
  `phenotypic.image_class` with `--image-type` as the fallback, and compares against the
  literal `"GridImage"` just as `load_image_from_store` (`sdk_/_io_constants.py:2780-2786`)
  does. Error when every input is affected, warning with subjects for a subset.
- **Mode scoping (D10)**: every Phase C requirement check goes through
  `operations_in_scope` / `_requirements_in_scope`. `MODE_SLOTS` matches spec §2.
- **No false positive on a legitimate run** was found for PF-GRID-IMAGE, PF-GRID-PRESET,
  PF-NO-DETECTOR, PF-MISSING-MODULE or PF-LICENSE. C1 is the only false positive, and it
  is a warning.
- **Duplicate keys**: `object_pairs_hook` plus the path walk handle nested objects, lists,
  lists of lists, dicts inside lists and a top-level list (checked ad hoc). A duplicate is
  a `ValueError`, not a `JSONDecodeError`, so `from_json` does not rewrap it as "Invalid
  JSON", and `read_pipeline_file` reports it as `PF-PIPELINE-LOAD`. `from_json(dict)`
  bypasses the check by design.
- **`--bit-depth`**: `click.Choice(["8","16"])` with an `int` callback. The GUI run console
  never emits the flag. The SLURM array script forwards the already-validated integer to
  the worker's own `type=int` option.
- **nn requirements**: the overrides import only `dataclasses` and `_checkpoint_manager`,
  and never `torch`, `micro_sam`, `transformers` or `sam2` (verified with importable stubs,
  above). `find_spec` is called only on top-level names, so no parent package is imported.
  The DINO repo ids produced by `dino_weight_requirement` equal `hf_dino_id(version, size)`
  for every size each detector's `Literal` allows. `Sam2(checkpoint=...)` declares no
  weights. `DinoSam2Detector` declares both backbones. The `huggingface_hub` requirement
  is added only for DINOv3, whose `snapshot_download` needs it. The extras match
  `pyproject.toml` (`torch` provides `sam2`; `foundation` pulls `torch` plus
  `transformers` plus `huggingface_hub`; `topology` provides `fil-finder`).
- **SAM3 gate**: `require_license_acceptance(..., interactive=False)` runs after the
  `transformers` import (no network) and before the first `from_pretrained`. It is the
  only runtime path that loads SAM3 weights, since `_forward_tiles` and `_infer_batch`
  both go through `_ensure_model_loaded`. Both DINOv3 runtime download sites pass
  `interactive=False`. The only remaining interactive `download()` is the explicit
  `phenotypic.detect.nn` CLI, where a prompt is appropriate.
- **License keys**: the preflight's accepted set is parsed exactly as the runtime gate
  parses it (case-insensitive, comma-separated, whitespace-stripped), and uses the same
  `license_key` constants (`sam3`, `dinov3`).
- **Hugging Face probe** returns `None` when `huggingface_hub` is absent, and
  `isinstance(found, str)` correctly treats both `None` and the `_CACHED_NO_EXIST` sentinel
  as uncached.
- **Lazy imports**: `_cli_preflight.py`, `sdk_/_preload.py` and `_cli_preload.py` import
  only the standard library at module level, and the class-resolution hook imports
  `_preload` inside the function. `tests/unit/ci` passes.
- **Task 7's deviation** is documented in spec §10.2 and in the commit message, and I
  judge it sound (see C4 for the one semantic gap).
