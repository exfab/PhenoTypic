# CLI preflight: refuse incompatible runs before anything is written

- **Date:** 2026-09-24
- **Branch:** `claude/modest-mccarthy-jz0ylw`, off `81d19ec`
- **Status:** revised after independent review; awaiting approval
- **Review:** `docs/superpowers/reports/2026-09-24-cli-preflight/spec-plan-review.md`
  (findings R1-R38; the disposition of each is in the table at the end of this spec)
- **Evidence:** `docs/superpowers/reports/2026-09-24-cli-preflight/claim-verification.md`
  (empirical probes, scripts under
  `docs/superpowers/plans/2026-09-24-cli-preflight/baseline_probes/`)
- **Plan:** `docs/superpowers/plans/2026-09-24-cli-preflight/plan.md`

## Objective

A user who launches `python -m phenotypic` with a configuration that cannot work should
learn so in the submitting shell, in seconds, and before the CLI has deleted, cleared, or
written anything. Today most incompatibilities surface much later: inside a SLURM array task
as a per-image failure, at finalization as a WARNING-level log line, or not at all.

The preflight must not run the pipeline on a sample image. A sample run conflates "this
configuration is incompatible" with "this one image is bad", and it can refuse a good run
because it drew a corrupt file. Every check below reads only the pipeline JSON, the CLI
options, the environment, the cluster's configuration, and file *headers*. No pixel is
decoded and no operation is applied.

## Non-goals

- **A sample or smoke run.** Excluded by the objective, for the reason above.
- **Changing what a successful run computes.** Every change here either refuses earlier,
  warns earlier, or fixes a defect whose current behavior is already wrong (§10). The one
  change that alters output bytes for some inputs, RAW decoding, is fenced in the work id
  (§12).
- **Changing finalization's handling of post-op failures.** §8 catches the static causes of
  a swallowed post failure before the run starts. Making the runtime swallow itself visible
  touches the finalizer's publication contract and is recorded in `DEFERRED.md`.
- **A strict mode that promotes warnings to errors.** Recorded in `DEFERRED.md`.
- **Validating `qc` and `plots` slots.** `find_operations` does not walk them
  (`sdk_/_operation_tree.py`), and plot backends already have their own preflight
  (`preflight_plot_backends`). Recorded in `DEFERRED.md`.
- **GUI chrome.** The GUI inherits the new checks through its existing Validate button
  (§11). No new control is added, so the `gui-tutorial-capture` ledgers do not move.

## Background

### What is validated today, and when

`phenotypic_cli` (`phenotypicCLI.py:1720`) validates in two places. Near the top it parses
options and refuses flag combinations; after building `ExecutionConfig` it calls
`uses_staged_gpu_strategy` (`:2229`), which refuses an unstageable `GpuDetector` and
swallows every other exception. The substantive checks, `validate_execution_config` and
`validate_pipeline` (`_cli/_cli_validation.py:29-127`), run at `:2503` and `:2518`. They
confirm that the pipeline loads, that it is not empty, that `nrows`/`ncols` are positive,
and that declared plot backends import. The `validate_execution_config` SLURM block is a
no-op (`_cli_validation.py:120-127`).

Between those two points the CLI mutates the output tree. `--restart` runs
`clear_machine_state` at `:2370`; `--overwrite` runs `shutil.rmtree(output_dir)` at
`:2399`; `mint_run_identity` at `:2438` writes `.phenotypic/restart_epoch.json` under
`--restart`. The `--dry-run` exit sits below all of them at `:2547`. Nothing refuses
`--dry-run` combined with `--overwrite` or `--restart`.

### What the probes established

The claim-verification report ran each suspected failure against `81d19ec`. Four results
shape this design. First, `--overwrite` with an unparseable pipeline deletes the output
directory and only then prints "Pipeline loading failed"; `--overwrite --dry-run` with a
valid pipeline deletes it and exits 0 (report §2). Second, a grayscale input under an
RGB-requiring `--detect-mode` does not fall back silently, as the pre-spec survey had
claimed. Every CLI image path calls `set_detect_mode` after reading, which raises
`ValueError`, so each such image fails with `PerImageScientificError`, and `--dry-run` does
not notice (report §4). Third, a custom operation cannot load in `python -m phenotypic` at
all, because the main process never calls `preload_custom_operation_modules` (report §1).
Fourth, a post op naming an absent column is caught at finalization, logged at WARNING, and
discards the output of *every* post op, including ones that succeeded (report §7).

### Why the checks are static, and where static knowledge is missing

Most requirements are already expressed as types. Every operation that refuses a plain
`Image` derives from one of four ABCs that raise `GridImageInputError`
(`abc_/_grid_object_detector.py:200-206`, `_grid_object_refiner.py:288-295`,
`_grid_corrector.py:253-261`, `_grid_measure.py:261-270`). Detection modes declare
`requires_rgb` (`_core/_image_parts/detection_modes/_detection_mode.py:30`). Measurers
declare their output schema through `get_measurement_infoclasses()`
(`abc_/_measure_features.py:333`).

Other requirements exist only as a raise deep inside `_operate`. `MeasureColor` needs RGB
because `image.color.XYZ` raises on gray (`_xyz_accessor.py:106-107`), and six GPU
detectors need optional packages they import lazily. A checker that re-derives these facts
from outside the operation drifts as operations are added. §3 therefore gives each
operation one place to state them.

## Findings this spec closes

Evidence marked **P** was reproduced by a probe in the claim-verification report; **R**
was established by reading the cited code; **S** rests on Slurm's documented behavior and
must be confirmed on the target cluster (plan Task 13).

| # | Finding | Evidence | § |
|---|---|---|---|
| F1 | `--overwrite` deletes the output before the pipeline is validated; `--overwrite --dry-run` deletes it and exits 0 | P §2; `phenotypicCLI.py:2399`, `:2518`, `:2547` | 1 |
| F2 | `--restart --dry-run` clears `.phenotypic/` and bumps the restart epoch | R `:2370`, `:2438`; `_cli_identity.py` docstring "gate finding F8" | 1 |
| F3 | `--overwrite` with `--input` inside `--output` deletes the inputs in `full` mode; only `process` mode refuses the overlap | R `:1865-1883`, `:2399` | 1 |
| F4 | Grid operations under `--image-type Image` fail every image | R the four ABC raises above | 4 |
| F5 | A pipeline preset with **both** `nrows` and `ncols` under `--image-type Image` fails every image: `measure()` injects `CenteredAutoGridFinder` unless `meas` already holds a `GridFinder` | P (review R5); `_image_pipeline_core.py:1305-1316` | 4 |
| F6 | A `full`-mode pipeline with no object-producing op fails every image with `NoObjectsError` | P §3; `_image_pipeline_core.py:1268-1272` | 4 |
| F7 | An RGB-requiring `--detect-mode` fails every grayscale input, and `--dry-run` does not notice | P §4 | 7 |
| F8 | RGB-requiring operations fail every grayscale input | P §6 (`MeasureColor`, `CalibrateColorRpcc`) | 7 |
| F9 | A missing optional package (`sam2`, `transformers`, `huggingface_hub`, `fil_finder`, `micro_sam`, `torch`) fails every image with `ImportError` | R `detect/nn/*`, `detect/_filfinder_detector.py:156-163` | 5 |
| F10 | DINOv3 detectors call `download()` with its default `interactive=True`, so a batch worker without `PHENOTYPIC_ACCEPT_MODEL_LICENSE` reaches `input()` | R `_dinosam2_detector.py:392`, `_helper/_dino_support.py:232`, `_checkpoint_manager.py:831` | 5, 10 |
| F11 | `Sam3` loads gated weights through `from_pretrained` and never calls the license gate, contrary to its own docstring | R `detect/nn/_sam3.py:180-205` | 5, 10 |
| F12 | Uncached weights are downloaded on the compute node at first use | R `_checkpoint_manager.py:296-315` and the table in §5 | 5 |
| F13 | `PHENOTYPIC_PRELOAD_MODULES` is ignored by the main CLI process and by the ordinary array worker, so a custom operation cannot run | P §1; callers listed in §10.2 | 5, 10 |
| F14 | A duplicate key in pipeline JSON is silently dropped, and the survivor takes the *first* occurrence's position, reordering execution | P §8; `_serializable_pipeline.py:279` | 10 |
| F15 | `--bit-depth` accepts any integer; only 8 and 16 are meaningful | R `phenotypicCLI.py:1535-1540` | 10 |
| F16 | `--gpu-slurm` `time` is not parsed at startup, only at script rendering after state is written | R `:1970-1982`, `:2221`, `sdk_/slurm/_sbatch.py:146` | 6 |
| F17 | A misspelled SLURM key, an unknown partition or account, or an unsatisfiable GPU request is rejected only by `sbatch` at submission, after scripts, state and the metadata snapshot are written | R `sdk_/slurm/_sbatch.py:127-162`, `:242-245` | 6 |
| F18 | Array and submit limits are checked only inside the strategies; the staged path has no GRES check; the one GRES check misreads an `sinfo` error as "no GPUs" | R `_cli_execution_strategies.py:899-947`, `_cli_staged_slurm.py:558-573` | 6 |
| F19 | A time limit above the partition's `MaxTime` is accepted and the job pends indefinitely where `EnforcePartLimits=NO` | S | 6 |
| F20 | GUI Validate never forwards `--slurm`, `--gpu-slurm` or `--gpu-shards`, so it cannot validate a cluster run | R `_gui/run_console/_state.py:515-603` | 11 |
| F21 | Metadata join problems surface only at finalization as warnings; unmatched images are dropped from `measurements.csv` and duplicate keys fan rows out | R `_cli/_cli_output_manager.py:338-418` | 9 |
| F22 | Workers parse the metadata CSV with Polars' default schema inference, which finalization deliberately avoids; the startup check uses pandas and cannot see the difference | R `_embedded_measurement_tables.py:85` vs `_cli_output_manager.py:333` | 9, 10 |
| F23 | A post op naming an absent column is swallowed at finalization, and all post output is discarded | P §7 | 8 |
| F24 | RAW files never reach `rawpy`: the RAW suffixes are in `IO.ACCEPTED_FILE_EXTENSIONS`, which the first `imread` branch tests, and a 16-bit file read this way arrives as 8-bit | P §5; `sdk_/constants_.py:96-101`, `_image_io_handler.py:732-736` | 7, 10 |
| F25 | Nothing checks that the output location is writable, has space, or is shared storage on a SLURM run | R (no `disk_usage`, `statvfs` or `os.access` under `_cli/`) | 9 |
| F26 | The dry-run SBATCH preview re-implements directive formatting and diverges from what is submitted | R `_cli/_cli_interactive.py:53-84` vs `sdk_/slurm/_sbatch.py:102-165` | 6 |
| F27 | `--overwrite` deletes a `--pipeline`, `--metadata` or `--image-manifest` file stored under `--output` before the run reads it; `--restart` does the same to one under `.phenotypic/` | R (review R10); `phenotypicCLI.py:2393-2399`, `sdk_/_io_constants.py:1347-1356` | 1 |
| F28 | Two inputs with the same stem in one dataset (`a.png`, `a.tif`) map to one store and one join key | R (review R29); whether a later stage refuses it is unverified (plan Task 10) | 7 |
| F29 | On a SLURM run, an input, pipeline, metadata CSV or `JoinMetadata` table on node-local storage is invisible to every worker | R (review R30) | 9 |

## Design

### §0 Principles

**Read-only, and first.** The preflight performs no write anywhere. It runs after
`ExecutionConfig` exists and before the first mutation of the output tree. §1 moves the
mutations below it.

**Findings, not exceptions.** Each check returns zero or more `PreflightFinding` values
instead of raising. The CLI collects all of them and prints one report, so a user with
three problems learns about all three in one attempt instead of one per launch.

**Severity follows reach.** A finding that makes *every* image fail, or corrupts the run's
tables, is an **error** and refuses the run. A finding that affects a *subset* of inputs is
a **warning**: the CLI already isolates per-image failures (`PerImageScientificError`), so
refusing the whole run over a few bad files would reproduce exactly the sample-run problem
the objective rules out. A warning lists the affected inputs (the first 20, then a count).

**A broken check never blocks a run.** If a check raises, the preflight records a warning
(`PF-CHECK-CRASHED`) naming the check and the exception. It does not refuse on a defect in the checker.

**Checks are scoped to what the mode runs.** Each CLI mode executes a different part of the
pipeline. `full` applies `ops` and then measures (`meas`, `post`, `filters`, `model`);
`process` only applies `ops` (`_cli_process_only.py:347`) and ignores `--metadata`; `measure`
only measures a stored image (`_cli_process_single.py:441-446`), takes the image class from
the store, and never applies `--detect-mode`. A requirement is checked only over the slots
the mode will execute, and an option is checked only in the modes that read it. Without this
rule a valid `process` run carrying `MeasureColor` in `meas` would be refused for a
measurement it never takes (review R4).

**`--skip-validation` skips the preflight.** It keeps its meaning ("skip pipeline
validation, for advanced users") and extends to every check in this spec, with four
exceptions that are structural rather than advisory. The first is the ordering in §1. The
second is the destructive-overlap refusals of §1, which protect the run's own inputs. The
third is option-type validation (`--bit-depth`, `--gpu-slurm` time, and the minimal
metadata parse of §10.5). The fourth is the existing unstageable-GPU refusal.

**Naming.** "Preflight" already names three things in this codebase: `preflight_plot_backends`,
the GUI's `build_metadata_preflight`, and the unstageable-GPU check that `_cli/CLAUDE.md`
calls "the preflight". User-facing text calls this change the **run preflight**; code keeps
the `_cli_preflight` module name, and docs refer to the GPU check as the "GPU placement
refusal".

### §1 Ordering: nothing mutates before the preflight and the dry-run exit

`phenotypic_cli` is reordered into a read-only half and a mutating half. The read-only half
ends with the preflight and the `--dry-run` exit; the mutating half begins with the restart
clear and the overwrite delete. Concretely, these existing steps move up, unchanged in
behavior, to sit before `clear_machine_state` (`:2363`):

1. the fresh-run output-contents check (`:2386-2410`), split so that its *refusal* ("Output
   directory already contains files") moves up while its `shutil.rmtree` stays down;
2. input scanning, image-manifest application and `organize_by_dataset` (`:2447-2484`), and
   measure mode's `scan_store_outputs`. All of these are read-only: `organize_by_dataset`
   builds `Dataset` values without creating directories (`_cli_directory_scanner.py:190-222`);
3. `validate_execution_config`, `validate_pipeline` and the new `run_preflight`
   (`:2492-2530`);
4. `_display_execution_config` and the `--dry-run` exit (`:2537-2552`).

`clear_machine_state`, the `rmtree`, `mint_run_identity` and everything after them keep
their current relative order. `mint_run_identity` remains below the `rmtree`, which is the
placement its own comment identifies as the fix for an earlier bug.

Two consequences follow. A dry run never mutates, so `--dry-run --overwrite` and
`--dry-run --restart` become previews: the dry-run output states what the real run would
delete or clear, by count and path, and then exits. That also retires the "gate finding F8"
exception documented on `mint_run_identity` (`_cli_identity.py`). The exception was
justified by "`--restart` has already run `clear_machine_state` by this point, so a dry run
under it has written to the tree regardless", and after this change that premise is false.
The docstring is updated to say a dry run exits before the mint.

The overlap refusal that `process` mode already applies (`:1865-1883`) is extended to every
file the run itself reads (F3, F27). Under `--overwrite`, the run is refused when the
canonical `--input`, `--pipeline`, `--metadata` or `--image-manifest` path equals or lies
inside the canonical `--output`, because the delete would remove it before the run reads it.
Under `--restart`, the same paths are refused when they lie inside `.phenotypic/` and are not
among the entries `clear_machine_state` preserves (`_PRESERVED_ON_RESTART`,
`sdk_/_io_constants.py:1313-1315`). These refusals are not skippable (§0).

### §2 The preflight module

A new module `src/phenotypic/_cli/_cli_preflight.py` holds the report types and the
orchestrator. It imports nothing heavy at module level, which keeps it within the lazy-entry
rule guarded by `tests/unit/ci/test_startup_imports.py`.

```python
Severity = Literal["error", "warning"]

@dataclass(frozen=True)
class PreflightFinding:
    code: FindingCode         # a Literal of every stable identifier, e.g. "PF-GRID-IMAGE"
    severity: Severity
    message: str              # what is wrong, naming the op path / file / key
    hint: str                 # what to do about it
    subjects: tuple[str, ...] = ()   # affected paths, capped at 20 for display
    subject_count: int = 0           # total affected, when subjects is capped

@dataclass(frozen=True)
class PreflightReport:
    findings: tuple[PreflightFinding, ...]
    @property
    def errors(self) -> tuple[PreflightFinding, ...]: ...
    @property
    def warnings(self) -> tuple[PreflightFinding, ...]: ...
    def render(self, console) -> None: ...   # grouped by category, errors first

RunMode = Literal["full", "measure", "process"]

#: The pipeline slots each mode executes (§0, "Checks are scoped to what the mode runs").
MODE_SLOTS: dict[RunMode, frozenset[str]] = {
    "full": frozenset({"ops", "meas", "post", "filters", "model"}),
    "process": frozenset({"ops"}),
    "measure": frozenset({"meas", "post", "filters", "model"}),
}

@dataclass(frozen=True)
class PreflightContext:
    config: ExecutionConfig
    pipeline: ImagePipeline
    datasets: Sequence[Dataset]
    mode: RunMode

def run_preflight(context: PreflightContext) -> PreflightReport: ...
```

`run_preflight` calls each check in a fixed order (pipeline, environment, cluster, inputs,
metadata, output) and wraps each call as §0 describes. A shared helper,
`operations_in_scope(context)`, walks the tree with `walk_operations` and keeps only the
operations reached through a slot in `MODE_SLOTS[context.mode]`; every requirement check
iterates that, never the whole tree.

The preflight needs the loaded pipeline, which `validate_pipeline` currently discards
(`_cli_validation.py:29-83`, it returns `(bool, str)`). A new
`load_pipeline_for_validation(path) -> tuple[ImagePipeline | None, PreflightFinding | None]`
does the load and the existing emptiness and plot-backend checks once, and
`validate_pipeline` becomes a thin wrapper over it for its existing callers. When the load
fails, no other check can run, so the CLI renders a report holding that single finding;
this is how `PF-CUSTOM-OP` and a JSON error reach the user through the same report format. Checks are plain functions
`check_<subject>(context) -> list[PreflightFinding]`, one per finding family, so each is
unit-testable against a hand-built context without a CLI invocation.

When the report has errors, the CLI prints it and exits with status 1, the status the
current validation failures already use. When it has only warnings, the CLI prints them
and continues. `--dry-run` runs the same preflight before it exits, because §1 places the
dry-run exit after it, and exits 1 on errors. `execute_dry_run` currently repeats
`full_validation` (`_cli_interactive.py:208-217`); that repeat is removed, since the report
has already been printed. The GUI's Validate runs the CLI as a `--dry-run` subprocess, so it
sees every finding and the exit status.

The finding codes form a closed set, the `FindingCode` `Literal`, with one hint per code in
a module-level `HINTS` mapping. A test asserts that every code has a hint and that the user
docs list every code.

### §3 Operations declare their requirements

`BaseOperation` gains one method:

```python
def preflight_requirements(self) -> OperationRequirements: ...
```

`OperationRequirements` is a frozen dataclass in `abc_/_requirements.py`, exported from
`phenotypic.abc_` so that authors of custom operations can declare requirements too:

```python
@dataclass(frozen=True)
class WeightRequirement:
    model: str                              # e.g. "sam2:tiny", "dinov3:base"
    license_key: str | None                 # the PHENOTYPIC_ACCEPT_MODEL_LICENSE token, if gated
    is_cached: Callable[[], bool | None]    # None = cannot tell without the network

@dataclass(frozen=True)
class OperationRequirements:
    grid_image: bool = False
    rgb_input: bool = False
    modules: tuple[str, ...] = ()           # import names, checked with find_spec
    extra: str | None = None                # the pyproject extra that provides them
    weights: tuple[WeightRequirement, ...] = ()
```

The default implementation derives `grid_image` from the type, as
`isinstance(self, (GridOperation, GridMeasureFeatures))`, and reads three class variables
for the rest: `_requires_rgb_input: ClassVar[bool] = False`,
`_requires_modules: ClassVar[tuple[str, ...]] = ()`, and
`_requires_extra: ClassVar[str | None] = None`. An operation whose requirement is
unconditional sets the class variable. An operation whose requirement depends on a field
overrides the method:

| Operation | Requirement | Declared by |
|---|---|---|
| `MeasureColor`, `MeasureColorComposition`, `FocusEdgeColorPhase`, `ColorDenoise`, `ColorCorrector`, `CalibrateColorRpcc` | RGB, unconditional | class variable |
| `ContrastGamma`, `ContrastLog`, `ContrastSigmoid`, `ContrastStretching` | RGB when `input_layer == "rgb"` | override on `InputLayerMixin` |
| every `GpuDetector` | RGB when `input_layer == "rgb"` (no package: a custom or fake `GpuDetector` need not use `torch`, and requiring it would refuse `tests/_fakes/register_fake_gpu.py` pipelines in a torch-free environment; each shipped detector below declares its own packages) | override on `GpuDetector` |
| `SetDetectMode` | RGB when `get_detection_mode(self.mode).requires_rgb` | override |
| `Sam2` | `sam2`, `torch`; weights `sam2:<model_size>` unless `checkpoint` is set | override |
| `Sam3` | `transformers`, `torch`; gated weights `facebook/sam3` | override |
| `DinoSam2Detector` | `transformers`, `sam2`, `torch`; DINO weights for `dino_version`/`dino_size` (gated when v3), SAM2 weights for `sam2_model_size` | override |
| `FssDinoDetector`, `Insid3Detector` | `transformers`, `torch`; DINO weights (gated when v3) | override on a shared helper |
| `MicroSamDetector` | `micro_sam`; weights `<model_type>`; RGB through the inherited `GpuDetector` rule (its `input_layer` defaults to `"gray"` but a user may set `"rgb"`) | override, calling the `GpuDetector` one |
| `FilFinderDetector` | `fil_finder`, `astropy` (extra `topology`) | class variables |

A ratchet test prevents a new RGB-reading operation from shipping undeclared. It lists every
concrete operation class whose module source contains `.rgb[` or `.color.` and requires that
the class set `_requires_rgb_input` *explicitly*, in its own `__dict__`, even when the value
is `False`. At `81d19ec` it flags `BayesShrinkCorrector`, `VisuShrinkCorrector`,
`DenoiseBlockMatch` and `MeasureSymZones` (a plot-only RGB read), all gray-tolerant, which set
`False` explicitly (review, "Claims verified"). `PadImage` and `ColorDenoise` are not flagged
by the heuristic and are declared by hand. The test is a heuristic and says so in its
docstring: it catches the common spelling of an RGB read, not every one.

The grid rule covers sixteen concrete classes at `81d19ec`, each deriving from one of the four
raising ABCs: `FilamentousFungiDetector`, `TwoKFilamentousDetector`, `ManualGridPointDetector`,
`GridAligner`, `GridApply`, `GridOversizedObjectRemover`, `KeepSectionLargest`,
`MergeWithinSection`, `ReduceSectionsByLine`, `RemoveGridOutliers`, `MeasureGridSpread`,
`MeasureGridLinRegStats`, `MeasureNeighborDist`, `AutoGridFinder`, `CenteredAutoGridFinder`,
and `ManualGridFinder`. (Since `86be6906` on `main`, `MeasureNeighborDist` accepts a plain
`Image` and is no longer a grid measurer; the derived requirement followed it with no change
here.)

Declaring requirements on the operation instead of in a checker-side table follows the
distinction `_CHILD_CONTRACT` draws in `_cli_validation.py`. That table restates a
*definitional* contract of three composition primitives, while these requirements are
incidental facts about each operation's implementation and change when the implementation
does. They belong beside the code that creates them.

### §4 Pipeline checks

These read only the loaded `ImagePipeline` and `ExecutionConfig`, over
`operations_in_scope(context)` (§2).

| Code | Condition | Severity |
|---|---|---|
| `PF-GRID-IMAGE` | the effective image class is `Image` and an in-scope operation has `grid_image`. In `full` and `process` mode the class is `config.image_type`; in `measure` mode it is each store's recorded `phenotypic.image_class`, with `--image-type` only as the fallback the worker uses (`_cli_process_single.py:414-416`), so the finding lists affected stores | error when every image is affected, else warning |
| `PF-GRID-PRESET` | the effective image class is `Image`, the mode calls `measure()` (`full` or `measure`), `pipeline.nrows` **and** `pipeline.ncols` are both set, and no `GridFinder` is in `meas` (F5). Under `--image-type Image` the CLI does not apply `--nrows`/`--ncols` to the pipeline (`_cli_process_single.py:264-275`), so only the preset matters. In `measure` mode the class is each store's recorded one | error; warning listing the plain stores when only some are (§0) |
| `PF-NO-DETECTOR` | mode is `full` (forward, CPU or staged) and no `ObjectDetector` exists anywhere in `ops` | error |

`PF-NO-DETECTOR` does not fire in `measure` mode, which measures stored objmaps
(`_cli_process_single.py:404-449`), or in `process` mode, which never calls `measure()`
(`_cli_process_only.py:347`). Its hint names the one legitimate exception, a custom
non-detector operation that writes the objmap itself, and points to `--skip-validation`.
The review's probe showed that `GridImage` fails the same way as `Image` under the default
`CenteredAutoGridFinder` (`grid/_centered_auto_grid_finder.py:339`); the claim-verification
report had probed only `Image`. The check therefore applies to both image types.

### §5 Environment checks

**Custom operations resolve.** The preflight runs after `preload_custom_operation_modules()`,
which §10.2 adds to the main process. When `from_json` fails with the "not found in
phenotypic namespace" error, `load_pipeline_for_validation` (§2) returns the finding
(`PF-CUSTOM-OP`, error) naming the class, and the CLI renders it as a one-finding report. Its hint
explains both halves of the contract: the module must be listed in
`PHENOTYPIC_PRELOAD_MODULES`, and importing it must attach the class to the `phenotypic`
namespace, as `tests/_fakes/register_fake_gpu.py` does. A module that merely defines the
class still fails (report §1b), and the old message ("Make sure it's properly imported in
phenotypic.__init__.py") gave no hint of either.

**Optional packages are installed.** For each distinct module in the union of
`requirements.modules`, `importlib.util.find_spec` must succeed (`PF-MISSING-MODULE`,
error). `find_spec` locates a package without importing it, so this costs no `torch` import
on a login node. The hint names the `uv sync --extra <extra>` command, or conda for
`micro_sam`, which no extra provides.

**Gated weights are accepted.** For each `WeightRequirement` with a `license_key`, the key
must appear in `PHENOTYPIC_ACCEPT_MODEL_LICENSE` (`PF-LICENSE`, error). After §10.3 no
runtime path prompts, so an unaccepted license fails every image.

**Weights are cached.** For each `WeightRequirement`, `is_cached()` is consulted
(`PF-WEIGHTS-UNCACHED`, warning). The finding is a warning, not an error, because some
clusters give compute nodes network access. Its hint names the pre-download command in
`phenotypic.detect.nn`'s CLI.

The cache probes must not import `torch` or `micro_sam` in the submitting process: the
existing `Sam2CheckpointManager.cache_dir()` imports `torch.hub`
(`_checkpoint_manager.py:232`) and `MicroSamCheckpointManager.cache_dir()` imported
`micro_sam.util` (`:409`), which imports `torch` (review R8). The probes therefore resolve
directories from the environment alone.

*As implemented* (plan Task 8, corrected after the Phase C review, C1, C7, C16):

- **SAM2.** `torch_hub_checkpoint_dir()` transcribes `torch.hub._get_torch_home`
  (`os.getenv("TORCH_HOME", <$XDG_CACHE_HOME or ~/.cache>/torch)`, so a set-but-empty
  variable is honored as torch honors it) plus `hub/checkpoints`. This is **parallel code,
  not a shared resolver**: `Sam2CheckpointManager.cache_dir()` keeps asking `torch.hub` at
  run time, because only torch can see a `torch.hub.set_dir` call. That call is the known
  limit of the probe; a test pins the two equal whenever torch is installed.
- **micro-sam.** `microsam_models_dir()` is the shared resolver, and
  `MicroSamCheckpointManager.cache_dir()` calls it. It mirrors upstream
  `micro_sam.util.models()`: `<MICROSAM_CACHEDIR or pooch.os_cache("micro_sam")>/models`,
  one file per model named after its registry key. The first implementation copied the
  manager's old fallback (the cache root, not `models/`) and matched by glob, so every
  cached micro-sam model read as uncached (review C1); `_get_default_model_folder`, which
  the manager imported, no longer exists upstream. The probe, `list_cached` and `clear`
  now match exact names only, since `vit_b` is a prefix of `vit_b_lm`.
- **Hugging Face repos.** `huggingface_hub.try_to_load_from_cache` on the repo's
  `config.json`, which returns `None` from `is_cached` when `huggingface_hub` is absent;
  `PF-MISSING-MODULE` has then already fired.

### §6 Cluster checks

These run only when `config.is_slurm_mode()` (`_cli_types.py:223`).

**Both profiles parse.** `--gpu-slurm` `time` is validated at option-parse time with
`parse_slurm_time`, exactly as `--slurm` is at `:1970-1982` (F16). This is an option-type
check and runs even under `--skip-validation`.

**`sbatch` accepts each profile.** For each profile the run will submit (the CPU profile
from `--slurm`; for a staged GPU run also the GPU profile from `resolve_stage_slurm_args`),
the preflight renders a script of three parts: a `#!/bin/bash` line, which
`format_sbatch_directives` does not emit and `sbatch` requires; that function's directive
block, with `/dev/null` as both log paths; and a `true` body. It pipes the script to
`sbatch --test-only` on standard input, with `env=sbatch_submission_environment()` as
`submit_script` uses (`sdk_/slurm/_sbatch.py:228-235`), so that `SBATCH_*` variables affect
the test and the real submission identically. A nonzero exit is `PF-SBATCH-REJECTED`, an
error with `sbatch`'s stderr in the message. Slurm documents `--test-only` as validating the script
and estimating a start time without submitting a job, and `sbatch` reads the script from
standard input when no file is named. So this check writes no file and catches a misspelled
key, an unknown partition, account or QoS, and an unsatisfiable GPU or memory request (F17)
in one call per profile. The subprocess has a 30-second timeout. When `sbatch` is not on
`PATH`, times out, or fails with a controller-communication error (for example "Socket timed
out" or "Unable to contact slurm controller"), the finding is the warning
`PF-SBATCH-UNAVAILABLE`, because a transient controller fault says nothing about the
configuration and the strategies will report a real submission failure themselves. The list
of communication-error patterns is a module constant with its own test.

**Time fits the partition.** Where `EnforcePartLimits` is `NO`, which the author and the
reviewer both recall as Slurm's default, a job whose time limit exceeds the partition's
`MaxTime` is accepted and then pends with reason `PartitionTimeLimit` (F19); neither could
reach Slurm's documentation to confirm it, so it is UNVERIFIED until plan Task 13. The
severity is therefore decided at run time on the user's cluster, not fixed from one cluster
at implementation time (review R11). The preflight reads `EnforcePartLimits` from
`scontrol show config` and the partition's `MaxTime` from `scontrol show partition <p>`
(10 s timeout each). When the requested time exceeds `MaxTime`, the finding
`PF-TIME-OVER-PARTITION` is a warning, not an error: a QOS with `Flags=PartitionTimeLimit`
can legitimately override the partition limit (also UNVERIFIED), and the preflight does not
resolve QOS flags. The warning's text states whether `EnforcePartLimits` is off, in which
case the job would pend rather than be rejected. The parser handles `MaxTime=UNLIMITED`,
a comma-separated partition list, and the absence of `slurm_partition` (the default
partition, read from `scontrol show partition` output's `Default=YES`).

**Limits the strategies already enforce, enforced earlier.** The staged strategy refuses
`MaxSubmitJobs < 3` and `--gpu-shards` above the chunk limit (`_cli_staged_slurm.py:558-573`),
but only after building the manifest, which hashes every input. These comparisons move into
a pure function, `staged_slurm_limit_errors(max_submit, array_limit, gpu_shards)`, called by
both the preflight (`PF-SLURM-LIMIT`, error) and the strategy.

**GPU partitions have GPUs.** The `sinfo --Format=gres` check inside
`AutonomousSLURMStrategy` (`_cli_execution_strategies.py:906-947`) becomes a shared function
used by the preflight for the profile that carries `slurm_gpus_per_node > 0`, which is the
GPU profile on the staged path. Staged forward runs, the common GPU case, never reached the
old check. The shared function checks `returncode` first, so an `sinfo` error for an
unknown partition reports that error instead of "partition has no GPUs" (F18). The strategy
keeps calling it, so `--skip-validation` does not remove the guard.

**The dry-run preview shows the real directives.** `_display_slurm_config`
(`_cli_interactive.py:53-84`) is replaced with the output of `format_sbatch_directives`, so
the preview is the text that will be submitted (F26).

*As implemented* (plan Tasks 13-14, corrected after the Phase E review, E1-E3, E5-E9,
E13; Slurm behavior read from Slurm's source and manual pages, see `slurm-behavior.md`):

- **Which `--test-only` failures are errors (E1).** Only a message naming a configuration
  fault that a real submission rejects the same way: an invalid partition, account, QoS or
  GRES specification (`slurm_errno.c`) and `sbatch`'s own option-parsing errors
  (`SBATCH_REJECTION_PATTERNS`). Every other failure is `PF-SBATCH-UNAVAILABLE`, carrying
  `sbatch`'s message. The first implementation made every nonzero exit an error, but the
  will-run test ignores DOWN and DRAINED nodes, so during a maintenance drain it refused
  runs that a real submission queues. A job-submit plugin's site-specific message is
  therefore a warning too: it cannot be told apart from a transient one.
- **Which profiles (E2).** A GPU pipeline that `AutonomousSLURMStrategy` runs (`--mode
  process` on SLURM) submits `--slurm` with `slurm_gpus_per_node=1` added; the preflight
  tests that profile, through the one definition `with_default_gpu_request`, and runs the
  GPU-partition check on it. The dry-run preview shows every profile the run submits (E9).
- **Which partition and time (E3, E7).** Read back from the rendered directives
  (`effective_sbatch_option`), so the GUI's `partition=`/`time=` count and the last
  directive wins, as in `sbatch`; an `SBATCH_PARTITION` or `SBATCH_TIMELIMIT` in the
  submission environment overrides both (`sbatch.1`). For a partition list the rule is
  `EnforcePartLimits`'s: the tightest `MaxTime` under `ALL`, the loosest otherwise.
- **The GRES check (E5, E6).** It refuses only when `sinfo -h -o %G` (untruncated; the
  `--Format` default truncates at 20 characters) positively lists the partition's GRES and
  no entry is named `gpu`. Empty output (an unknown or hidden partition) and a failing
  `sinfo` are "cannot tell"; `sbatch --test-only` reports an unknown partition precisely.
  The strategy uses the same function, so this also stops the strategy refusing a GPU run
  for an unreadable partition.
- **Output location (E8, E13).** Three separately registered checks, so one's fault cannot
  hide another's finding. A later mount at the same point wins, every octal escape in
  `/proc/self/mounts` is decoded, and `overlay` (a container image's root) is not
  node-local.
- **Known limit, not changed (E4).** `PF-SLURM-LIMIT` inherits the staged strategy's
  `get_slurm_max_submit_jobs`, which takes the smallest `MaxSubmitJobsPerUser` over every
  QoS on the cluster. The strategy refuses the same runs later, so the preflight adds no
  refusal; resolving the job's own QoS changes the strategy's chunking and is left to a
  separate change.

The unused helpers in `sdk_/slurm/_slurm_headroom.py` are **not** adopted. Their
`subprocess.run` calls have no timeout, and `validate_submission` expects submitit-style
unprefixed keys (`partition`, `cpus_per_task`) that the CLI never produces.

### §7 Input checks: headers, not pixels

For each scanned input the preflight reads one header, in a bounded thread pool (16
workers, a module constant). It records the channel count, dtype, and shape:

| Input kind | How the header is read |
|---|---|
| PNG, JPEG | `PIL.Image.open(path)`, which parses the header lazily and decodes nothing until `load()` |
| TIFF | `tifffile.TiffFile(path).series[0]`: `shape` and `dtype` (tags only), then skimage's axis move and `Image`'s channel rule (see *As implemented* below) |
| RAW | no header read; after §10.1 a RAW file decodes to 16-bit RGB when `rawpy` is importable |
| OME-Zarr store | the root and series `zarr.json` documents via the existing `ngff_` helpers, stopping before `zarr.open_array`; channel count comes from `project_ngff_axes` |

*As implemented* (plan Task 10, corrected after the Phase D review, D1, D5, D9):

- **TIFF from the first series, not the first page.** `skimage.io.imread` returns
  `series[0]`, and a Fiji composite, an OME-TIFF with `CYX` axes, or any single-series
  `(3,H,W)` stack stores its channels as single-sample pages of one series, which decode
  to RGB. The first implementation read `pages[0].samplesperpixel` and called them
  grayscale, so `PF-RGB-OP-GRAY` and `PF-DETECT-MODE-GRAY` refused runs that complete
  (D1, reproduced end to end by the reviewer). The header now transcribes the two rules
  `imread` applies: skimage's axis move (`skimage/io/_io.py`) and
  `ImageDataManager._guess_image_format`. A 4-D series is reported unknown.
- **Stores.** A PhenoTypic store's channels come from its recorded `series` block; a
  third-party store's are reported unknown rather than re-derived through
  `project_ngff_axes`, which can only weaken a check. A Zarr v2 store is reported as
  unreadable with its format named.
- **No shape field.** `InputHeader` records channels, refused channel count, bits, and
  whether the input restores PhenoTypic metadata; nothing consumes a shape.

`tifffile` is imported directly in two modules already (`_color_space_accessor.py:9`,
`_accessor_io_handler.py:335`) but arrives only transitively through scikit-image. Adding it
to `[project] dependencies` was planned here and is **deferred**: `uv lock` could not
re-resolve the project in the implementation environment for a reason unrelated to
`tifffile` (see `DEFERRED.md`). It remains installed through scikit-image.

A header reports what the file stores, which is not always what `imread` returns: a
palette PNG stores one band but decodes to RGB, and a multi-page TIFF's first page need not
be the array `skimage.io.imread` returns (review R25). The checks below therefore use the
**decoded** channel count, which the header reader derives through a small table mapping
header facts to decoded shape (PIL mode `P` to three channels, and so on). Plan Task 10 builds
that table from probes of `imread` itself, and a test compares the table's prediction with
`imread`'s actual result on each probe file.

The input checks run in `full` and `process` mode, where inputs are read from `--input`. In
`measure` mode the inputs are stores already under `--output`, their image class and channel
layout are recorded in the store, and `--detect-mode` is not applied, so only
`PF-RGB-OP-GRAY` runs there, over the stores' recorded channel count. With `--sample`, the
checks still cover every input: a sample is a trial of the full run, and a user should learn
from it what the full run will meet (review R26).

The findings apply the severity rule of §0. Each is an error when every input is affected
and a warning listing the affected inputs otherwise:

| Code | Condition |
|---|---|
| `PF-DETECT-MODE-GRAY` | `--detect-mode` requires RGB and the input is single-channel (F7) |
| `PF-RGB-OP-GRAY` | some operation's requirements set `rgb_input` and the input is single-channel (F8) |
| `PF-HEADER-UNREADABLE` | the header cannot be parsed, or the file is zero bytes |
| `PF-RAW-NO-RAWPY` | the input is RAW and `rawpy` is not importable |
| `PF-CHANNELS` | the channel count is one the reader refuses; plan Task 10 establishes that set by probe before the check is written |
| `PF-BIT-DEPTH` | `--bit-depth` disagrees with the header dtype, or dtypes are mixed across inputs; the severity and wording follow from what plan Task 10's probe shows `imread` does with the mismatch |
| `PF-STEM-COLLISION` | two inputs in one dataset share a stem (`a.png`, `a.tif`) and would map to one store and one metadata key (F28). This is read from the scan, not from headers. It is an error because it corrupts outputs rather than failing an image, unless plan Task 10 finds that a later stage already refuses it, in which case the check moves that refusal earlier with the same wording |

The header pass costs one small read per input. On a forward run it is small next to what
startup already does: `_prepare_incremental_startup` computes `work_id_for_image` for every
input, which hashes each file in full (`_cli_failure_tracker.py:347-373`).

### §8 Post-measurement column checks

> **Removed after implementation (2026-09-26, user decision).** `PF-POST-COLUMN` was
> taken out before merge. It predicted the columns the final table would carry from each
> measurer's declared headers, the intrinsic metadata, and the `--metadata` CSV, and that
> prediction is brittle against changes to measurement outputs: moving the Feret
> diameters from `MeasureShape` to `MeasureSize` on `main` changed the column set it
> depended on, and every such change needs this check to keep up. The check, its
> finding code, `PostMeasurement.preflight_columns` with its overrides, and
> `post._utils.missing_metadata_columns` are deleted. F23 is therefore not closed: a post
> operation naming an absent column is still logged at finalization and discards the
> post output, as before this change. The text below is kept as the design record.

`PostMeasurement` gains `required_columns(self) -> tuple[str, ...]`, default `()`. It is
implemented by `AppendString`, `PrependString`, `ExpandMetadata` and `MergeMetadata` (their
already-normalized column fields) and by `JoinMetadata` (its `on` keys).

The preflight computes the set of columns the master table will carry when post runs. That
set has four known sources: the headers of every measurer's declared info classes, the
intrinsic image metadata that `measure()` inserts, the `Metadata_Dataset` column the CLI adds
when the dataset column is on (`_cli_output_manager.py:1764-1767`, `:1918-1922`), and the
headers of `--metadata` after normalization. It then walks the post chain in order, adding the
columns each op creates (`ExpandMetadata.labels`, `MergeMetadata.label`, `JoinMetadata`'s
joined columns) before checking the next op.

The set is not always complete for metadata (review R6). Reading a PhenoTypic-exported file or
store restores its public metadata into the image (`_image_io_handler.py:797-824`), and
`insert_metadata` writes every public and protected key as a column
(`_metadata_accessor.py:366-376`); a custom operation may also set `image.metadata[...]`. So
for the four metadata-string ops, a required column absent from the set is an error
(`PF-POST-COLUMN`) only when the set is provably complete: no in-scope operation is a class
defined outside the `phenotypic` package, and no input is an OME-Zarr store or carries
PhenoTypic metadata in its header (the header reader of §7 reports this from the TIFF tag or
PNG text chunk under `IO.PHENOTYPIC_METADATA_KEY`, without decoding pixels). Otherwise the
same finding is a warning.

*As implemented* (plan Task 12, review D5): the API is
`PostMeasurement.preflight_columns(available) -> (missing, produced)` rather than
`required_columns()`, with a `_preflight_reads_metadata_only` class variable. Each op answers
through its own resolution rules against the columns known so far, and reports the columns
it adds, so a later op in the chain finds them. Where this section says `required_columns()`
below, read `preflight_columns`. The error case matters because at run time the absence would
silently discard every post op's output (F23).

`JoinMetadata` keeps its `on` keys in the *table's* spelling after validation and re-spells
them against the frame at run time (`post/_join_metadata.py:193-203`, `:258-263`). Its
`required_columns()` therefore applies the same
`external_metadata_preserved_columns` / `ensure_metadata_prefix` rule its `_normalized_table`
uses, rather than returning `on` verbatim. For `JoinMetadata.on`, which may name a
measurement header, an absent column is a warning, because the measurement half of the set is
not always complete either:
`TEXTURE.get_headers` needs a `scale` argument (`schema/_texture.py:160`), grid finders
declare no info class, and custom measurers may declare nothing. Membership is a set test on
already-canonical headers, so no header string is parsed for a `Metadata_` prefix, as the
schema-ownership rule in `CLAUDE.md` requires.

### §9 Metadata and output checks

**Metadata join.** The GUI already computes everything the CLI needs, in
`build_metadata_preflight` (`_gui/run_console/_request_safety.py:401-552`), but that function
is bound to GUI types (`SandboxRoot`, the GUI metadata payload). Its core moves to a new
`_cli/_metadata_preflight.py`:

```python
def analyze_metadata_join(
    images: Sequence[tuple[str, Path]],   # (dataset, path), as scanned
    metadata_csv: Path,
) -> MetadataJoinAnalysis: ...
```

The metadata checks run only in `full` mode; `process` mode ignores `--metadata`
(`phenotypicCLI.py:1918-1926`), and `measure` mode joins nothing new. The analysis reads the
CSV with the shared reader of §10.5, normalizes headers with the SDK functions
the GUI's wrapper uses, builds the source key frame (`IMAGE.IMAGE_NAME` from
`source_image_stem`, `IMAGE.SUFFIX`, `EXPERIMENT.DATASET`), and calls
`prepare_metadata_join_keys`. The GUI's `build_metadata_preflight` keeps its signature and
its sandbox and fingerprint logic, and delegates the analysis. That removes the duplicated
source-key projection.

| Code | Condition | Severity |
|---|---|---|
| `PF-META-PARSE` | the CSV does not parse with full schema inference | error |
| `PF-META-ALIAS` | header normalization raises (conflicting legacy and canonical aliases); finalization would fail the same way | error |
| `PF-META-NO-KEYS` | no column is shared with the source key frame, so nothing joins | error only when the CSV has no measurement-level key column **and** the run's metadata set is complete (§8's rule); otherwise warning |
| `PF-META-DUP-KEYS` | duplicate join keys among the verifiable columns, which fan measured rows out | as `PF-META-NO-KEYS` |
| `PF-META-UNMATCHED` | images with no metadata row; their rows are dropped from `measurements.csv` | warning, listing images |
| `PF-META-ORPHANS` | metadata rows matching no image; they appear as metadata-only rows | warning |
| `PF-META-UNVERIFIED` | a join column that only a measurement can supply (e.g. `Grid_RowNum`), which the preflight cannot verify | warning |

The source key frame holds only `ImageName`, `Suffix` and `Dataset`, while the production
join runs against the measurement frame, which also carries headers such as `Grid_RowNum` and
`Grid_ColNum`, and `normalize_external_metadata_columns` deliberately keeps those as join keys
(`_cli/_metadata_join.py:107-129`). A per-well plate map keyed on
`ImageName + Grid_RowNum + Grid_ColNum` therefore *looks* duplicated to the source frame, and a
plate layout keyed only on `Grid_RowNum + Grid_ColNum` *looks* keyless, yet the real join
matches both without duplicates (review R1, reproduced). That is why the two findings are
errors only when no measurement-level key column exists. In that case they are real: with no
shared column `join_metadata` skips the join and publishes the table without metadata
(`_cli_output_manager.py:336-341`), and duplicate keys fan rows out. `--skip-validation`
remains the escape.

*As implemented* (plan Task 11, corrected after the Phase D review, D2, D4, D5):

- **Completeness (D2).** The production join intersects the CSV with the *measurement*
  frame, which carries every metadata key the images restore on read and any a custom
  operation sets. So the key findings are errors only when `_metadata_set_is_complete`
  holds, the same predicate §8 applies to post columns (review R6). A CSV keyed on `Strain`
  over PhenoTypic PNG exports that carry `Strain` is a warning; the reviewer showed such a
  run joins correctly.
- **Which qualified columns may be keys (D4).** A qualified CSV column counts as a possible
  measurement key only when it is a known schema header, which is exactly when
  `external_metadata_preserved_columns` keeps its raw spelling in the production join;
  `Strain_ID` is joined as an attribute and no longer softens the key errors. With a custom
  operation in scope, every qualified column still counts. This filter is applied in the
  CLI check; the shared helper and the GUI's direct use of it are unchanged, and the GUI's
  Validate runs the CLI dry-run, which applies it.
- **The GUI (D5).** `build_metadata_preflight` reuses `source_join_key_frame` and
  `unverified_measurement_join_columns` and calls `prepare_metadata_join_keys` after its own
  input normalization; it does not call `analyze_metadata_join`. The duplicated source-key
  projection is gone, which was the purpose; the reviewer compared the two analyses on seven
  CSV shapes and they agree.

The moved `_unverified_measurement_join_columns` (`_request_safety.py:367-388`) filters on
`"_" in column` before calling `metadata_member_for_header`. That is a test of whether a name
is qualified, not a metadata-prefix test, so it does not break the schema-ownership rule; the
moved docstring says so, so that a later reader does not "fix" it into a prefix check.

**Output location** (`PF-OUTPUT-*`). The preflight checks the nearest existing ancestor of
`--output`, since the directory itself may not exist yet, with three read-only probes:

- `os.access(ancestor, os.W_OK | os.X_OK)` must hold (error).
- In `full` mode only, `shutil.disk_usage(ancestor).free` below the total size of the inputs
  is a warning. It is a heuristic, not a bound: a forward run stores each input's pixels
  (unless `--drop-originals`) with compression, so the true footprint may be smaller or
  larger. `process --layer objmap` writes far less than its input, and `measure` mode's
  inputs are already inside `--output`, so neither mode runs this check (review R21). The hint
  also says that `disk_usage` cannot see GPFS user quotas.
- On a SLURM run, a path on node-local storage is a warning, because workers on other nodes
  cannot see it (`_cli/CLAUDE.md`, "One writer per artifact"). Node-local is decided by the
  filesystem type of the path's mount, read from `/proc/self/mounts` (Linux only; other
  platforms skip the check): `tmpfs` and local disk types (`ext4`, `xfs`, `btrfs`) are
  node-local, while `gpfs`, `lustre`, `nfs`, `nfs4`, `beegfs` and `cifs` are shared. A name
  heuristic such as `$SCRATCH` would misfire on clusters where scratch is a shared parallel
  filesystem (review R21). The check covers `--output`, and also `--input`, `--pipeline`,
  `--metadata` and every `JoinMetadata` table path, since workers read those too (F29,
  review R30).

### §10 Defects fixed alongside

Several findings are runtime defects, not missing checks. A preflight warning would only
describe them, so this change fixes each at its source.

**§10.1 RAW decoding (F24).** In `_image_io_handler.py` the RAW branch is tested before the
general branch, and `IO.ACCEPTED_FILE_EXTENSIONS` keeps its meaning for the scanner. A RAW
suffix with `rawpy` present decodes through `rawpy`; with `rawpy` absent it raises
`UnsupportedFileTypeError`, naming the missing package. The branch copies `rawpy_params`
before popping from it, instead of mutating the caller's dict.

The `rawpy` branch has never run in production, because it was unreachable (review R9). Its
parameters (linear gamma `(1, 1)`, `output_bps=16`, automatic brightness) were written but
never exercised, so enabling it without a real decode would ship an untested scientific
path. Plan Task 9 therefore decodes at least one real camera RAW file and checks shape,
`uint16` dtype and a plausible intensity range before the branch is enabled; if no redistributable
RAW sample can be committed, the test reads a path from an environment variable and the
change is gated on a recorded manual run. §12 covers continuation and the Windows
consequence.

**§10.2 Custom-op preload (F13).** Preloading must happen in every *process* that
deserializes a pipeline, not only in every entry point. Local runs execute images through
joblib's default loky backend (`_cli_execution_strategies.py:375`; the staged strategy at
`_cli_staged_strategy.py:214`, `:351`), whose workers are fresh processes that never pass
through `main`; the review reproduced a local `--njobs 2` run failing every image even with
the main process preloaded (R2).

*As implemented* (plan Task 7), the preload runs at the one step every such process passes
through: class resolution. `SerializablePipeline._find_class_in_phenotypic` searches the
`phenotypic` namespace. Before its first lookup in a process, hit or miss, it imports the
modules `PHENOTYPIC_PRELOAD_MODULES` names (`preload_custom_operation_modules_once`, keyed
on the variable's value and recorded before importing, so a module that resolves a class
during its own import does not re-enter); on a miss it imports them again, which also waits
for another thread's in-flight import, and searches once more. Preloading on the first call
rather than only on a miss is review C4: otherwise a process whose classes all resolve would
never import a module that rebinds or registers something, and would run different code from
the main CLI. This reaches the CLI, SLURM workers, loky workers and
any future call site by construction, where the originally planned per-site calls would have
needed a list that a new site could fall out of. Because `_core` may not import `_cli`, the
function moved to `phenotypic.sdk_._preload`, and `_cli/_cli_preload.py` re-exports it for
existing callers. The main CLI also calls it explicitly right after
`load_runtime_dependencies()`, so a broken module name fails at startup as one
`click.ClickException` line naming the variable (review C6). A caller that must not import
custom code, `_metadata_migration._serialized_class` in `--mode migrate`, uses
`_search_phenotypic_namespace` instead (review C5); `RemoveByFeature` and the GUI analysis
recipe keep the preloading lookup, because a custom measurer or analyzer must resolve there. An unresolved class raises `UnknownOperationClassError` (an `AttributeError`
subclass) whose message names the registration contract, and `load_pipeline_for_validation`
reports it as `PF-CUSTOM-OP`.

**§10.3 No prompt inside a pipeline (F10, F11).** The two DINOv3 runtime call sites pass
`interactive=False`, which is what `Dinov3CheckpointManager.download`'s own docstring says
batch callers do. `Sam3._ensure_model_loaded` calls
`require_license_acceptance(..., interactive=False)` before `from_pretrained`, which is what
its docstring already claims it does.

**§10.4 Duplicate JSON keys (F14).** `ImagePipeline.from_json` parses with an
`object_pairs_hook` that raises `ValueError` naming the duplicated key and its JSON path.
It lives in `from_json`, not in the CLI, because the silent reorder is equally wrong in a
notebook. Its reach is every pipeline read through `ImagePipeline.from_json` (a file or a
JSON string); two paths still parse with plain `json` and keep the last duplicate:
`BaseOperation.from_json` for single-operation JSON, and the tune spec's embedded pipeline
(`tune/_spec.py`), which arrives already parsed. `from_json(dict)` bypasses the check by
design (review C10). A saved pipeline carrying a duplicate stops loading; that is intended, since it
never ran as written.

**§10.5 One metadata reader (F22).** A single `read_metadata_csv(path) -> pl.DataFrame`
in `_cli/_metadata_join.py` reads with `infer_schema_length=None`. The worker
(`_embedded_measurement_tables.py:85`), `join_metadata` (`_cli_output_manager.py:333`), the
GUI preflight, and the new CLI preflight all call it. The pandas parse at
`phenotypicCLI.py:2149-2160` is replaced by a call to `read_metadata_csv` in the same place,
outside the skippable block, so that an unreadable CSV is still refused under
`--skip-validation` as it is today (review R22). `PF-META-PARSE` reports the same failure in
the report when validation runs.

Full-scan inference can also *change* a column's inferred dtype where the 100-row sample
would not have raised, for example a column whose first 100 rows are null. Whether a worker's
embedded table carries metadata dtypes that aggregation later compares across stores is not
yet established (review R23, open question 4); plan Task 11 settles it before the reader
changes, and adds a work-id fence like §12's if a continuation could mix the two rules.
`_snapshot_metadata_csv` keeps its byte-for-byte copy, as the metadata-snapshot rule in
`CLAUDE.md` requires.

**§10.6 `--bit-depth` (F15).** The option becomes `click.Choice(["8", "16"])`, converted to
`int`.

### §11 GUI

Validate forwards the same `--slurm`, `--gpu-slurm` and `--gpu-shards` tokens that Run
forwards, by reusing the argv construction Run already uses (`_slurm_argv_extension` and
the GPU tokens added in `_build_subprocess_argv`, `_gui/run_console/_slurm.py:177-210`). With
§1's ordering a `--dry-run` never submits, so forwarding is safe: a GUI Validate of a SLURM
run now reaches `sbatch --test-only`. The run console shows the CLI's output and exit status
as it does today, so no chrome changes.

### §12 Compatibility and continuation

**RAW inputs change meaning.** After §10.1, the same RAW file yields different pixels, so a
resumed run must not reuse an output decoded the old way. The work id has two producers:
`work_id_for_image`, and the SLURM worker's `_worker_work_identity`
(`_cli_process_single.py:123-170`), which calls `compute_work_id` directly and refuses any
mismatch ("SLURM task work identity does not match worklist", `:814-838`). A revision added to
only one producer would fail every RAW image on SLURM (review R3). The revision therefore lives
inside `compute_work_id` itself (`_cli_failure_tracker.py:293`): when the suffix of
`relative_image_path` is a RAW suffix, the digest payload gains
`"raw_decode_revision": RAW_DECODE_REVISION`; for any other suffix the payload, and so the
digest, is byte-identical to today's. `RAW_DECODE_REVISION` is `2` because the pre-fix
decoding is implicitly revision 1. This follows the precedent of
`PROCESS_LAYER_SEMANTICS_REVISION`, scoped per image instead of per run, so continuations of
non-RAW runs are untouched.

**RAW on Windows.** `rawpy` is not installed on Windows (`pyproject.toml:55`). Today a RAW
file there is read through Pillow, which returns an embedded preview or fails; after §10.1 it
raises `UnsupportedFileTypeError`, and `PF-RAW-NO-RAWPY` reports it before the run. This is a
deliberate change: the previous result was not the sensor data.

**Refusals that are new.** `PF-NO-DETECTOR`, `PF-GRID-IMAGE`, `PF-GRID-PRESET`,
`PF-STEM-COLLISION`, `PF-META-NO-KEYS` and `PF-META-DUP-KEYS` (each only in the cases §9 makes
errors), `PF-POST-COLUMN` (only when the column set is provably complete, §8), the §1 overlap
refusals, and the §10.4 duplicate-key error refuse runs that start today. Each of those runs
fails every image, publishes a wrong or incomplete table, or deletes its own inputs, so the
refusal changes when the user learns, not whether the run succeeds. `--skip-validation`
bypasses all but the §1 overlap refusals and §10.4, which is a load error.

Two further refusals change what a working run does, and are called out separately in the
user docs and the change log:

- **SAM3 now requires license acceptance** (review R7). Today `Sam3` loads gated weights
  through `from_pretrained` without PhenoTypic's gate (`detect/nn/_sam3.py:178-205`), so a user
  with Hugging Face access and cached weights runs without setting
  `PHENOTYPIC_ACCEPT_MODEL_LICENSE`. After §10.3 that run is refused by `PF-LICENSE`. This is
  the contract the class's own docstring already states. The remedy is one line:
  `export PHENOTYPIC_ACCEPT_MODEL_LICENSE=sam3`.
- **A pipeline with a duplicate JSON key can no longer be recompiled or reloaded** (review
  R24). `--mode recompile` and the GUI reload the pipeline from
  `deliverables/pipeline.json.pht-pipe`, which holds the user's original bytes. The error
  names the key; removing the duplicate that did not run restores the file.

**Dry runs that exit 1.** A `--dry-run` whose preflight finds an error exits 1, as a
dry run with an unloadable pipeline already does today.

### §13 Testing

Each check function is unit-tested against a hand-built `PreflightContext`, with a positive
case, a negative case, and, for warnings, a subset case asserting the listed subjects. SLURM
checks use the existing `subprocess.run` patching pattern
(`tests/unit/cli/test_cli_slurm_array.py:66-130`). The ordering change is pinned
end-to-end with the probe's own scenario: a sentinel file survives
`--overwrite` + bad pipeline, and survives `--overwrite --dry-run`
(`tests/unit/cli/test_cli_preflight_ordering.py`). Every probe in the claim-verification
report becomes a regression test that fails on `81d19ec` and passes after its task.

No logic-validation script is written. `CLAUDE.md` asks for one when a design rests on a
numeric invariant, and this design has none; its one quantity, the disk-space comparison, is
labelled a heuristic and only warns.

## Decisions

| # | Decision | Alternatives rejected, and why |
|---|---|---|
| D1 | Static checks only; no sample run | A sample run conflates a bad image with a bad configuration (objective) |
| D2 | Findings are collected, not raised | Raising reports one problem per launch |
| D3 | Error when every input is affected, warning for a subset | Refusing over a subset reproduces the sample-run failure mode |
| D4 | Requirements declared on operations (§3) | A checker-side table drifts from `_operate`; the `_CHILD_CONTRACT` precedent covers definitional contracts only |
| D5 | Dry run exits before any mutation, and previews deletions | Refusing `--dry-run --overwrite` outright loses a useful preview |
| D6 | `sbatch --test-only` over a key allow-list | An allow-list of `sbatch` options is a second copy of Slurm's documentation to maintain |
| D7 | Duplicate JSON keys refused in `ImagePipeline.from_json` (text input), not only in the CLI; single-op JSON and the tune spec are not covered (§10.4) | A CLI-only check leaves notebooks reordering silently |
| D8 | RAW decode fenced per image in the work id | A run-wide revision would cold-start every in-flight continuation |
| D9 | Headroom helpers not adopted | No timeouts, and they expect keys the CLI never produces |
| D10 | Checks scoped to the slots each mode executes (§0, §2) | Walking the whole tree refuses valid `process` and `measure` runs (review R4) |
| D11 | RAW revision inside `compute_work_id`, keyed on suffix | Adding it to one of two producers breaks SLURM (review R3); `_cli/CLAUDE.md` "One producer per derived value" |
| D12 | Metadata key findings are errors only when no measurement-level key exists | The source key frame cannot see measurement keys, so errors would refuse valid plate maps (review R1) |
| D13 | Partition-time finding is a warning whose text depends on the live `EnforcePartLimits` | Fixing severity from one cluster ships a false positive to others (review R11) |
| D14 | Node-local storage detected by filesystem type, not path name | `$SCRATCH` is shared storage on many clusters (review R21) |

## Open questions

1. **`PF-GRID-PRESET`.** Resolved by the review's probe (R5): the injection raises when both
   `nrows` and `ncols` are set, and not otherwise. Kept with the corrected condition.
2. **`PF-CHANNELS` and `PF-BIT-DEPTH`.** What `imread` does with RGBA, palette PNG, and a
   `--bit-depth` that disagrees with the dtype is not yet known. Plan Task 10 probes it and
   the checks mirror the observed behavior; the spec does not guess.
3. **Slurm behavior.** `--test-only` semantics, `sbatch` reading standard input, and
   `EnforcePartLimits` defaulting to `NO` are recalled by both author and reviewer but were
   not checked against Slurm's documentation (unreachable from this environment). Plan
   Task 13 checks them on a real cluster. The design no longer depends on the default,
   because §6 reads the live setting.
4. **Metadata dtype drift (§10.5).** Whether full-scan inference can change a stored table in
   a way aggregation notices across a continuation. Plan Task 11 Step 1 settles it.
5. **Same-stem inputs (F28).** Whether a later stage already refuses them. Plan Task 10
   settles it.

## Disposition of the review (R1-R38)

Every finding in `spec-plan-review.md` was accepted. The ones that changed this design are
cited inline as "review Rn". The rest change only the plan, and are listed here so that none
is lost:

| Finding | Where it is addressed |
|---|---|
| R1, R4, R5, R6, R7, R8, R9, R10, R11, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R34, R35, R36, R38 | this spec, at the sections citing them |
| R2, R3 | this spec (§10.2, §12) and plan Tasks 7 and 9 |
| R12, R13, R16, R17, R18, R19, R20, R31, R32, R33, R37 | plan only (fixtures, test names, test scope, grep patterns, citations) |
| R14, R15 | this spec (§3 table, §6) and plan Tasks 8 and 13 |

## Disposition of the Phase C adherence review (C1-C16)

`docs/superpowers/reports/2026-09-24-cli-preflight/phase-c-adherence.md`, verdict "pass
with changes". Every finding was accepted.

| Finding | Disposition |
|---|---|
| C1 (Major) | Fixed. `microsam_models_dir()` mirrors upstream `micro_sam.util.models()` (verified against `micro_sam/util.py` on `master`), the manager's `cache_dir()` calls it, and the probe, `list_cached` and `clear` match exact names. The old globbing `clear("vit_b")` would have deleted `vit_b_lm` once pointed at the real folder. §5 |
| C2 (Major) | Fixed. The guard runs `run_preflight` over every nn detector and `FilFinderDetector` with importable stub packages, plus a self-test that a caught import is still seen. Revert proof: R8's regression fails it with `['micro_sam']` |
| C3 | Fixed. A load finding prints as a one-finding report, code and hint included |
| C4 | Fixed. Resolution preloads before its first lookup, hit or miss, once per variable value. §10.2 |
| C5 | Fixed for `_serialized_class` (migrate), which now uses `_search_phenotypic_namespace`. `RemoveByFeature` and the GUI recipe keep the preloading lookup on purpose. §10.2 |
| C6 | Fixed. A broken name is a `click.ClickException` naming the variable; a subprocess test pins it |
| C7 | Spec updated (§3 table, §5), including the `torch.hub.set_dir` limit |
| C8 | Four tests added; each of the reviewer's mutations now fails one |
| C9 | Fixed. `accepted_model_licenses()` is the one parse |
| C10 | D7 and §10.4 narrowed to what ships |
| C11 | Fixed (return annotations, `Returns:` sections, wrapped lines) |
| C12 | Fixed. One finding per package, naming every extra |
| C13 | Fixed. The idempotence test counts module executions |
| C14 | Fixed in both `CLAUDE.md` files |
| C15 | §4 table updated to the implemented severity rule |
| C16 | Fixed. `torch_hub_checkpoint_dir` transcribes `torch.hub._get_torch_home` (verified against `torch/hub.py` on `main`); an empty `TORCH_HOME` is pinned by a test |

## Disposition of the Phase D adherence review (D1-D9)

`docs/superpowers/reports/2026-09-24-cli-preflight/phase-d-adherence.md`, verdict "changes
required". Every finding was accepted.

| Finding | Disposition |
|---|---|
| D1 (Blocking) | Fixed. TIFF headers predict from `series[0]` through skimage's axis move and `Image`'s channel rule; five stack cases and an RGBA TIFF join `CASES`, which compares every prediction with `imread`. §7 |
| D2 (Major) | Fixed. Key findings are errors only when the metadata set is complete. §9 |
| D3 (Major) | Fixed. JPEG is exempt from `PF-BIT-DEPTH`. `header-behavior.md` gains a JPEG row |
| D4 | Fixed in the CLI check (known schema headers only, unless a custom op is in scope). §9 |
| D5 | Spec updated (§7, §8, §9 *As implemented*) |
| D6 | Tests added for M3 (a long first data row, which only pandas accepts), M11, M14, M19, M22, M23 and M31; each now fails under its mutation |
| D7 | The write tripwire's context carries a metadata CSV |
| D8 | Partly accepted. Process mode no longer parses the `--metadata` it ignores. The snapshot keeps its pandas parse: the finding's premise, that the shared reader is stricter on every input, has a counterexample (`b'"unterminated'`, which Polars reads as a one-column header and pandas refuses), pinned by `test_invalid_metadata_never_replaces_existing_snapshot` |
| D9 | A Zarr v2 store's header error names the format |

## Disposition of the Phase E adherence review (E1-E13)

`docs/superpowers/reports/2026-09-24-cli-preflight/phase-e-adherence.md`, verdict "changes
required". All findings but E4 and E12 were fixed; see §6 *As implemented*.

| Finding | Disposition |
|---|---|
| E1 (Blocking) | Fixed. Errors only for named configuration faults; the reviewer's eight other messages are warnings, each pinned by a test, and a CLI test drives both verdicts through a fake `sbatch` on `PATH` |
| E2 (Major) | Fixed. `with_default_gpu_request` is shared by the strategy and the preflight |
| E3 (Major) | Fixed. `effective_sbatch_option` reads the rendered directives; the strategy's GRES check uses it too |
| E4 (Major) | Not fixed: a pre-existing limit of the staged strategy, which refuses the same runs; recorded in §6 and proposed as a separate change |
| E5, E6 | Fixed in `partition_gres_error` |
| E7 | Fixed: `EnforcePartLimits` rule for lists, `SBATCH_*` precedence, full-sentence wording test |
| E8 | Fixed |
| E9 | Fixed: the preview reuses the preflight's profile list |
| E10 | Tests added for M9, M13, M20 (patched `os.access`), M23b, M26b and M28b, plus the CLI `sbatch` test; the browser fake scheduler still accepts everything, and the CLI test covers the subprocess path instead |
| E11 | `slurm-behavior.md` records what the documentation settles; the probe gains cases 8-9 |
| E12 | Does not reproduce: the form-state validation (`_state.py:479`) already refuses an empty SLURM profile before any argv is built; pinned by a test |
| E13 | Fixed: three output checks; `partition_gres_error` catches `OSError` |
