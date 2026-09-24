# CLI preflight: refuse incompatible runs before anything is written

- **Date:** 2026-09-24
- **Branch:** `claude/modest-mccarthy-jz0ylw`, off `81d19ec`
- **Status:** design drafted, awaiting review
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
| F5 | A pipeline with `nrows`/`ncols` under `--image-type Image` fails every image: `measure()` injects `CenteredAutoGridFinder` | R `_image_pipeline_core.py:1293-1326`; to be confirmed by plan Task 5 | 4 |
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

**`--skip-validation` skips the preflight.** It keeps its meaning ("skip pipeline
validation, for advanced users") and extends to every check in this spec, with three
exceptions that are structural rather than advisory. The first is the ordering in §1. The
second is option-type validation (`--bit-depth`, `--gpu-slurm` time). The third is the
existing unstageable-GPU refusal.

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

The overlap refusal that `process` mode already applies (`:1865-1883`) is extended to
`--overwrite` in every mode. When `--overwrite` is set and the canonical input lies inside
the canonical output, or equals it, the run is refused, because the delete would remove the
inputs (F3).

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

@dataclass(frozen=True)
class PreflightContext:
    config: ExecutionConfig
    pipeline: ImagePipeline
    datasets: Sequence[Dataset]
    mode: Literal["full", "measure", "process"]

def run_preflight(context: PreflightContext) -> PreflightReport: ...
```

`run_preflight` calls each check in a fixed order (pipeline, environment, cluster, inputs,
metadata, output) and wraps each call as §0 describes. Checks are plain functions
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
| every `GpuDetector` | RGB when `input_layer == "rgb"`; `torch` | override on `GpuDetector` |
| `SetDetectMode` | RGB when `get_detection_mode(self.mode).requires_rgb` | override |
| `Sam2` | `sam2`, `torch`; weights `sam2:<model_size>` unless `checkpoint` is set | override |
| `Sam3` | `transformers`, `torch`; gated weights `facebook/sam3` | override |
| `DinoSam2Detector` | `transformers`, `sam2`, `torch`; DINO weights for `dino_version`/`dino_size` (gated when v3), SAM2 weights for `sam2_model_size` | override |
| `FssDinoDetector`, `Insid3Detector` | `transformers`, `torch`; DINO weights (gated when v3) | override on a shared helper |
| `MicroSamDetector` | `micro_sam`; weights `<model_type>`; RGB never (`input_layer="gray"`) | override |
| `FilFinderDetector` | `fil_finder`, `astropy` (extra `topology`) | class variables |

A ratchet test prevents a new RGB-reading operation from shipping undeclared. It lists every
concrete operation class whose module source contains `.rgb[` or `.color.` and requires that
the class set `_requires_rgb_input` *explicitly*, in its own `__dict__`, even when the value
is `False`. That is what `BayesShrinkCorrector`, `VisuShrinkCorrector` and `PadImage`
(conditional readers that tolerate gray) will do. The test is a heuristic and says so in its
docstring: it catches the common spelling of an RGB read, not every one.

Declaring requirements on the operation instead of in a checker-side table follows the
distinction `_CHILD_CONTRACT` draws in `_cli_validation.py`. That table restates a
*definitional* contract of three composition primitives, while these requirements are
incidental facts about each operation's implementation and change when the implementation
does. They belong beside the code that creates them.

### §4 Pipeline checks

These read only the loaded `ImagePipeline` and `ExecutionConfig`. The requirement walk uses
`walk_operations` / `find_operations` (`sdk_/_operation_tree.py`), which reach nested
pipelines, composites, and the `meas`, `post`, `filters` and `model` slots.

| Code | Condition | Severity |
|---|---|---|
| `PF-GRID-IMAGE` | `image_type == "Image"` and any operation in the tree has `grid_image` | error |
| `PF-GRID-PRESET` | `image_type == "Image"` and the pipeline sets `nrows` or `ncols` (F5) | error, once plan Task 5 confirms the injection raises; otherwise dropped |
| `PF-NO-DETECTOR` | mode is `full` (forward, CPU or staged) and no `ObjectDetector` exists anywhere in `ops` | error |

`PF-NO-DETECTOR` does not fire in `measure` mode, which measures stored objmaps
(`_cli_process_single.py:404-449`), or in `process` mode, which never calls `measure()`
(`_cli_process_only.py:347`). Its hint names the one legitimate exception, a custom
non-detector operation that writes the objmap itself, and points to `--skip-validation`.
The probes showed that `GridImage` fails the same way as `Image` under the default
`CenteredAutoGridFinder` (`grid/_centered_auto_grid_finder.py:339`), so the check applies to
both image types.

### §5 Environment checks

**Custom operations resolve.** The preflight runs after `preload_custom_operation_modules()`,
which §10.2 adds to the main process. When `from_json` fails with the "not found in
phenotypic namespace" error, the finding (`PF-CUSTOM-OP`, error) names the class. Its hint
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
`phenotypic.detect.nn`'s CLI. The cache probes reuse what exists:
`Sam2CheckpointManager.is_cached` (`_checkpoint_manager.py:241`),
`MicroSamCheckpointManager.list_cached` (`:460`), and, for Hugging Face repos,
`huggingface_hub.try_to_load_from_cache` on the repo's `config.json`. That last probe
returns `None` from `is_cached` when `huggingface_hub` is absent, in which case
`PF-MISSING-MODULE` has already fired.

### §6 Cluster checks

These run only when `config.is_slurm_mode()` (`_cli_types.py:223`).

**Both profiles parse.** `--gpu-slurm` `time` is validated at option-parse time with
`parse_slurm_time`, exactly as `--slurm` is at `:1970-1982` (F16). This is an option-type
check and runs even under `--skip-validation`.

**`sbatch` accepts each profile.** For each profile the run will submit (the CPU profile
from `--slurm`; for a staged GPU run also the GPU profile from `resolve_stage_slurm_args`),
the preflight renders a directive block with `format_sbatch_directives` and a trivial body,
then pipes it to `sbatch --test-only` on standard input (`PF-SBATCH-REJECTED`, error, with
`sbatch`'s stderr in the message). Slurm documents `--test-only` as validating the script
and estimating a start time without submitting a job, and `sbatch` reads the script from
standard input when no file is named. So this check writes no file and catches a misspelled
key, an unknown partition, account or QoS, and an unsatisfiable GPU or memory request (F17)
in one call per profile. The subprocess has a 30-second timeout. When `sbatch` is not on
`PATH`, or times out, the finding is a warning (`PF-SBATCH-UNAVAILABLE`), because the
strategies will report the submission failure themselves.

**Time fits the partition.** Slurm's `EnforcePartLimits` defaults to `NO`, under which a
job whose time limit exceeds the partition's `MaxTime` is accepted and then pends with
reason `PartitionTimeLimit` (F19). `--test-only` may therefore pass such a request. The
preflight reads `scontrol show partition <p>` (timeout 10 s) and compares `MaxTime` with the
requested time (`PF-TIME-OVER-PARTITION`, error). Plan Task 13 confirms both Slurm
behaviors on the target cluster before the severity is fixed; if `EnforcePartLimits` is set
there, the check still costs one call and cannot misfire.

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

The unused helpers in `sdk_/slurm/_slurm_headroom.py` are **not** adopted. Their
`subprocess.run` calls have no timeout, and `validate_submission` expects submitit-style
unprefixed keys (`partition`, `cpus_per_task`) that the CLI never produces.

### §7 Input checks: headers, not pixels

For each scanned input the preflight reads one header, in a bounded thread pool (16
workers, a module constant). It records the channel count, dtype, and shape:

| Input kind | How the header is read |
|---|---|
| PNG, JPEG | `PIL.Image.open(path)`, which parses the header lazily and decodes nothing until `load()` |
| TIFF | `tifffile.TiffFile(path).pages[0]`: `samplesperpixel`, `dtype`, `shape` |
| RAW | no header read; after §10.1 a RAW file decodes to 16-bit RGB when `rawpy` is importable |
| OME-Zarr store | the root and series `zarr.json` documents via the existing `ngff_` helpers, stopping before `zarr.open_array`; channel count comes from `project_ngff_axes` |

`tifffile` is imported directly in two modules already (`_color_space_accessor.py:9`,
`_accessor_io_handler.py:335`) but arrives only transitively through scikit-image. §10
adds it to `[project] dependencies` so the preflight does not depend on a transitive pin.

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

The header pass costs one small read per input. On a forward run it is small next to what
startup already does: `_prepare_incremental_startup` computes `work_id_for_image` for every
input, which hashes each file in full (`_cli_failure_tracker.py:347-373`).

### §8 Post-measurement column checks

`PostMeasurement` gains `required_columns(self) -> tuple[str, ...]`, default `()`. It is
implemented by `AppendString`, `PrependString`, `ExpandMetadata` and `MergeMetadata` (their
already-normalized column fields) and by `JoinMetadata` (its `on` keys).

The preflight computes the set of columns the master table will carry when post runs. That
set has three sources: the headers of every measurer's declared info classes, the intrinsic
image metadata that `measure()` inserts, and the headers of `--metadata` after
normalization. It then walks the post chain in order, adding the columns each op creates
(`ExpandMetadata.labels`, `MergeMetadata.label`, `JoinMetadata`'s joined columns) before
checking the next op.

For the four metadata-string ops, a required column absent from that set is an error
(`PF-POST-COLUMN`). Metadata columns come only from the three sources above, so the set is
complete for them, and at run time the absence would silently discard every post op's
output (F23). For `JoinMetadata.on`, which may name a measurement header, an absent column
is a warning, because the measurement half of the set is not always complete:
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

It reads the CSV with the shared reader of §10.5, normalizes headers with the SDK functions
the GUI's wrapper uses, builds the source key frame (`IMAGE.IMAGE_NAME` from
`source_image_stem`, `IMAGE.SUFFIX`, `EXPERIMENT.DATASET`), and calls
`prepare_metadata_join_keys`. The GUI's `build_metadata_preflight` keeps its signature and
its sandbox and fingerprint logic, and delegates the analysis. That removes the duplicated
source-key projection.

| Code | Condition | Severity |
|---|---|---|
| `PF-META-PARSE` | the CSV does not parse with full schema inference | error |
| `PF-META-ALIAS` | header normalization raises (conflicting legacy and canonical aliases); finalization would fail the same way | error |
| `PF-META-NO-KEYS` | no column is shared with the source key frame, so nothing joins | error |
| `PF-META-DUP-KEYS` | duplicate join keys, which fan measured rows out | error |
| `PF-META-UNMATCHED` | images with no metadata row; their rows are dropped from `measurements.csv` | warning, listing images |
| `PF-META-ORPHANS` | metadata rows matching no image; they appear as metadata-only rows | warning |
| `PF-META-UNVERIFIED` | a join column that only a measurement can supply (e.g. `Grid_RowNum`), which the preflight cannot verify | warning |

`PF-META-NO-KEYS` and `PF-META-DUP-KEYS` refuse runs the CLI accepts today. Both produce
tables that are wrong rather than incomplete, and `--skip-validation` remains the escape.

**Output location** (`PF-OUTPUT-*`). The preflight checks the nearest existing ancestor of
`--output`, since the directory itself may not exist yet, with three read-only probes:

- `os.access(ancestor, os.W_OK | os.X_OK)` must hold (error).
- `shutil.disk_usage(ancestor).free` below the total size of the inputs is a warning,
  labelled a lower bound: every run that writes per-image outputs writes at least one
  output per input. It is not
  an estimate of the run's footprint, and `disk_usage` cannot see GPFS user quotas, which
  the hint says.
- On a SLURM run, an output under `tempfile.gettempdir()`, `/dev/shm`, `$TMPDIR`, or
  `$SCRATCH` is a warning, because workers on other nodes cannot see node-local storage
  (`_cli/CLAUDE.md`, "One writer per artifact").

### §10 Defects fixed alongside

Several findings are runtime defects, not missing checks. A preflight warning would only
describe them, so this change fixes each at its source.

**§10.1 RAW decoding (F24).** In `_image_io_handler.py` the RAW branch is tested before the
general branch, and `IO.ACCEPTED_FILE_EXTENSIONS` keeps its meaning for the scanner. A RAW
suffix with `rawpy` present decodes through `rawpy`; with `rawpy` absent it raises
`UnsupportedFileTypeError`, naming the missing package. §12 covers continuation.

**§10.2 Custom-op preload (F13).** `preload_custom_operation_modules()` is called in the
main CLI immediately after `load_runtime_dependencies()` (`phenotypicCLI.py:1805`), and at
the top of every worker entry point that loads a pipeline and does not call it today:
`_cli_process_single.main`, `_cli_chunk_writer`, `_cli_recompile_worker`, and the
non-staged finalizer path in `_cli_checkpoint_handler`. The worker list is derived by
grepping for `from_json(` under `_cli/` (plan Task 7), not from this paragraph.

**§10.3 No prompt inside a pipeline (F10, F11).** The two DINOv3 runtime call sites pass
`interactive=False`, which is what `Dinov3CheckpointManager.download`'s own docstring says
batch callers do. `Sam3._ensure_model_loaded` calls
`require_license_acceptance(..., interactive=False)` before `from_pretrained`, which is what
its docstring already claims it does.

**§10.4 Duplicate JSON keys (F14).** `ImagePipeline.from_json` parses with an
`object_pairs_hook` that raises `ValueError` naming the duplicated key and its JSON path.
This is library-wide, not CLI-only, because the silent reorder is equally wrong in a
notebook. A saved pipeline carrying a duplicate stops loading; that is intended, since it
never ran as written.

**§10.5 One metadata reader (F22).** A single `read_metadata_csv(path) -> pl.DataFrame`
in `_cli/_metadata_join.py` reads with `infer_schema_length=None`. The worker
(`_embedded_measurement_tables.py:85`), `join_metadata` (`_cli_output_manager.py:333`), the
GUI preflight, and the new CLI preflight all call it. The pandas parse at
`phenotypicCLI.py:2149-2160` is removed, because `PF-META-PARSE` supersedes it.
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
resumed run must not reuse an output decoded the old way. `work_id_for_image` adds a
`RAW_DECODE_REVISION = 2` to the digest input for inputs with a RAW suffix only. This
follows the precedent of `PROCESS_LAYER_SEMANTICS_REVISION`, scoped per image instead of
per run, so continuations of non-RAW runs are untouched.

**Refusals that are new.** `PF-NO-DETECTOR`, `PF-GRID-IMAGE`, `PF-META-NO-KEYS`,
`PF-META-DUP-KEYS`, `PF-POST-COLUMN` and the §10.4 duplicate-key error refuse runs that
start today. Each of those runs either fails every image or publishes a wrong table, so the
refusal changes when the user learns, not whether the run succeeds. `--skip-validation`
bypasses all but §10.4, which is a load error.

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
numeric invariant, and this design has none; its one quantity, the disk-space bound, is
labelled a lower bound rather than an estimate.

## Decisions

| # | Decision | Alternatives rejected, and why |
|---|---|---|
| D1 | Static checks only; no sample run | A sample run conflates a bad image with a bad configuration (objective) |
| D2 | Findings are collected, not raised | Raising reports one problem per launch |
| D3 | Error when every input is affected, warning for a subset | Refusing over a subset reproduces the sample-run failure mode |
| D4 | Requirements declared on operations (§3) | A checker-side table drifts from `_operate`; the `_CHILD_CONTRACT` precedent covers definitional contracts only |
| D5 | Dry run exits before any mutation, and previews deletions | Refusing `--dry-run --overwrite` outright loses a useful preview |
| D6 | `sbatch --test-only` over a key allow-list | An allow-list of `sbatch` options is a second copy of Slurm's documentation to maintain |
| D7 | Duplicate JSON keys refused in `from_json`, library-wide | A CLI-only check leaves notebooks reordering silently |
| D8 | RAW decode fenced per image in the work id | A run-wide revision would cold-start every in-flight continuation |
| D9 | Headroom helpers not adopted | No timeouts, and they expect keys the CLI never produces |

## Open questions

1. **`PF-GRID-PRESET`.** F5 is traced, not run. Plan Task 5 runs it; if the injection does
   not raise, the code is dropped.
2. **`PF-CHANNELS` and `PF-BIT-DEPTH`.** What `imread` does with RGBA, palette PNG, and a
   `--bit-depth` that disagrees with the dtype is not yet known. Plan Task 10 probes it and
   the checks mirror the observed behavior; the spec does not guess.
3. **Slurm behavior on the target cluster.** `--test-only` semantics and the
   `EnforcePartLimits` setting are confirmed there in plan Task 13 before the severity of
   `PF-TIME-OVER-PARTITION` is fixed.
</content>
</invoke>
