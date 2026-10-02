# GPU detectors: fall back to CPU when no accelerator is present

Date: 2026-10-02. Status: **implemented** (same day), with the recommended
scope: only `device="auto"` falls back; an explicit accelerator still raises.

## Implementation notes (deviations from the plan below)

- **Step 3 needed a second change.** `with_default_gpu_request`
  (`sdk_/slurm/_sbatch.py`) passed an explicit `slurm_gpus_per_node=0` through,
  so the non-staged path would have emitted `--gpus-per-node=0`, which SLURM
  rejects (the staged resolver's docstring records this as OQ4). It now drops
  the key, mirroring `resolve_stage_slurm_args`. The run preflight shares that
  helper, so it still skips the partition check for that case.
- **The two CLI blocks became helpers** so they are testable without driving a
  whole strategy: `_report_gpu_pipeline_device` (local device report) and
  `gpu_pipeline_slurm_args` (SLURM profile + GRES refusal), both in
  `_cli/_cli_execution_strategies.py`.
- **Step 2's optional `StagedGpuStrategy` notice and Step 5 were not done.**
  The staged path logs the fallback through `resolve_device`'s log warning at
  model load.
- **Tests** run without PyTorch through a fake `torch`
  (`tests/_fakes/fake_torch.py`): `tests/unit/detect/nn/test_resolve_device_fallback.py`
  and `tests/unit/cli/test_gpu_cpu_fallback.py`. Each of the three fixes was
  reverted in turn and at least one test failed each time (4, 3 and 1
  failures). The CPU smoke run of real models listed under **Risks** is still
  outstanding.

## Problem

A pipeline containing any `GpuDetector` (`Sam2`, `Sam3`, `MicroSamDetector`,
`FssDinoDetector`, `Insid3Detector`, `DinoSam2Detector`) aborts on a machine
with no CUDA/MPS/XPU/HPU device. The user expectation is that the default
`device="auto"` uses an accelerator when one exists and otherwise runs on CPU.

## Root cause

The refusal is deliberate, not an accident, and it lives in one function.

1. `resolve_device(device="auto", allow_cpu=False)`
   (`src/phenotypic/detect/nn/_helper/_checkpoint_manager.py:121-180`) probes
   accelerators and, when none is found, falls back to CPU **only if**
   `allow_cpu=True`. Otherwise it raises
   `RuntimeError("No accelerator available. ... Set device='cpu' to force CPU")`.
2. Every detector calls it with the default `allow_cpu=False`, and every
   detector's `device` field defaults to `"auto"`:

   | Detector | Field | Call |
   |---|---|---|
   | `Sam2` | `_sam2.py:304` | `_sam2.py:347` |
   | `Sam3` | `_sam3.py:171` | `_sam3.py:235` |
   | `MicroSamDetector` | `_microsam_detector.py:163` | `_microsam_detector.py:217` |
   | `FssDinoDetector` | `_fssdino_detector.py:397` | `_fssdino_detector.py:482` |
   | `Insid3Detector` | `_insid3_detector.py:277` | `_insid3_detector.py:375` |
   | `DinoSam2Detector` | `_dinosam2_detector.py:344` | `_dinosam2_detector.py:426` |

   No production caller passes `allow_cpu=True`; the fallback branch is reached
   only by `tests/unit/detect/nn/test_checkpoint_manager.py:120`.
3. The contract is documented as intended behavior in three places, which must
   change with the code: the `GpuDetector` class docstring
   (`abc_/_gpu_detector.py:84-85`, "No GPU available: Raises RuntimeError"),
   each detector's `Raises:` section, and
   `docs/source/how_to/pages/gpu_detection_setup.md:710,756-765`.

### Where the failure surfaces

- **Local staged run** (`StagedGpuStrategy`, the route for every local forward
  GPU run and `--layer objmap`): Stage 1 preprocesses *every* image, then
  `plan.gpu_detector._ensure_model_loaded()` at
  `_cli/_cli_staged_strategy.py:230` raises outside the per-image `try`, so the
  whole run aborts after Stage 1's work is spent.
- **SLURM staged Stage-2 worker**: same call at
  `_cli/_cli_staged_slurm_worker.py:315`. This contradicts the documented
  escape hatch in `CLAUDE.md` ("explicit `slurm_gpus_per_node=0` runs the GPU
  stage on a CPU partition"): with zero GPUs the worker reaches the same raise.
- **Local non-staged run** (`LocalParallelStrategy`, reached by
  `--mode process --layer {rgb,gray,detect_mat}` with a GPU pipeline): an
  explicit pre-check at `_cli/_cli_execution_strategies.py:346-357` calls
  `resolve_device("auto")`, catches only `ImportError`, and lets the
  `RuntimeError` abort the run before any image.
- **Notebook** `detector.apply(image)`: raises on first inference.

### Secondary inconsistency found while tracing

`AutonomousSLURMStrategy` (`_cli_execution_strategies.py:914-934`) runs
`partition_gres_error` even when the user explicitly set
`slurm_gpus_per_node=0`, whereas the preflight twin `check_gpu_partition`
(`_cli/_cli_preflight.py:968-969`) skips that case. A GPU pipeline exported with
`--mode process` onto a CPU partition is therefore refused at submission even
after the user opted out of a GPU. Fixing the device fallback without fixing
this leaves the SLURM non-staged path still refused.

## Decision to confirm before implementing

**Scope of the fallback.** Recommended: fall back only for `device="auto"`;
keep raising for an explicitly requested accelerator (`"cuda"`, `"mps"`,
`"xpu"`). Rationale: `"auto"` already means "pick what is available", while an
explicit `"cuda"` is a statement the user wants a GPU, and silently ignoring it
hides a misconfigured node (for example, a SLURM job that landed without its
GRES). The alternative (fall back for explicit accelerators too) is friendlier
for pipeline JSON authored on a GPU workstation and replayed on a CPU node, at
the cost of masking that misconfiguration.

## Fix

### Step 1: make `"auto"` fall back in `resolve_device`

Change the default of `allow_cpu` to `True` in `resolve_device`, keeping the
existing `UserWarning`. Prefer changing the default over editing six call
sites: one place to keep correct, and any future detector inherits it. Keep the
`allow_cpu=False` branch for callers that need a hard requirement. Also log the
fallback at `WARNING` through the module `logger`, because a `UserWarning`
raised once per process inside a SLURM or loky worker is easy to lose.

Explicit-device branches (`_checkpoint_manager.py:166-180`) stay unchanged
under the recommended scope.

### Step 2: stop the CLI pre-check from refusing

`_cli_execution_strategies.py:346-357`: after Step 1, `resolve_device("auto")`
returns `"cpu"` instead of raising, so the run proceeds, but the console line
would read `✓ GPU detected: cpu`. Branch on the result: print the accelerator
when one is found, otherwise a yellow line stating that no accelerator was
found and the GPU stage will run on CPU (slow). Keep the forced `n_jobs=1`:
multiple loky workers each loading a foundation model on CPU would multiply
resident memory, and PyTorch already parallelizes a CPU forward across
intra-op threads.

Optionally add the same one-line notice to `StagedGpuStrategy` before Stage 1,
so a local staged run tells the user up front rather than after Stage 1.

### Step 3: honor `slurm_gpus_per_node=0` in `AutonomousSLURMStrategy`

Guard the `partition_gres_error` call at `_cli_execution_strategies.py:921`
with the same condition the preflight uses:
`effective_sbatch_option(slurm_args, "gpus-per-node") in (None, "0")` → skip.
This aligns submission with preflight and makes the documented CPU-partition
route work for the non-staged path too. The staged path already resolves its
profile through `resolve_stage_slurm_args`; verify it does not re-check GRES
when the GPU profile carries `=0`.

### Step 4: update the documented contract

- `abc_/_gpu_detector.py:76-85`: replace "No GPU available: Raises
  RuntimeError at pipeline validation time" with the fallback behavior. The
  sentence is also stale in a second way: no check runs "at pipeline
  validation time"; the raise happens at model load.
- Each detector's `device:` arg and `Raises:` entry (six files): `"auto"`
  falls back to CPU with a warning; only an unavailable *explicit* accelerator
  raises.
- `docs/source/how_to/pages/gpu_detection_setup.md:702-711` and the
  `RuntimeError: No accelerator available` troubleshooting entry (`:756-765`):
  rewrite for the new default; keep a note that CPU inference of large models
  is slow.
- `src/phenotypic/_cli/CLAUDE.md` and root `CLAUDE.md` where they describe GPU
  refusal, if any sentence still implies a hard GPU requirement.

### Step 5 (optional, recommended): record the resolved device

CPU and GPU forwards are not bit-identical (floating-point reduction order
differs across backends), so two runs of the same pipeline JSON can produce
slightly different masks depending on the node. Nothing currently records which
device ran. Store the resolved device string on the detector (several already
keep `self._device`) and surface it in the run log; recording it in the
per-image provenance journal is a larger change and should be its own decision.
[Based on general reasoning about floating-point determinism; not measured on
these models.]

## Tests

Per task, run only the directly touched files (per `CLAUDE.md`).

1. `tests/unit/detect/nn/test_checkpoint_manager.py`: monkeypatch the
   accelerator checks to all return `False`; assert `resolve_device("auto")`
   returns `"cpu"` and emits the `UserWarning`; assert
   `resolve_device("auto", allow_cpu=False)` still raises; assert
   `resolve_device("cuda")` still raises when CUDA is unavailable. Replace the
   current `test_auto_without_allow_cpu_returns_or_raises`, whose outcome
   depends on the host's hardware and so pins nothing.
2. One parametrized test across the six detectors: with accelerator probes
   patched to `False`, `_ensure_model_loaded()` reaches the model constructor
   with `device="cpu"` (stub the constructor; no weights download).
3. CLI: `LocalParallelStrategy` with a GPU pipeline and probes patched to
   `False` does not raise and prints the CPU notice.
4. `StagedGpuStrategy` local run with a stub `GpuDetector` and probes patched
   to `False` completes all three stages (extend `tests/unit/cli/` staged
   tests that already use a stub detector).
5. `AutonomousSLURMStrategy`: with `slurm_gpus_per_node=0` and a stubbed
   `partition_gres_error` that would refuse, submission is not refused; with the
   key absent, it still is.
6. `tests/unit/ci/test_startup_imports.py` and `test_deferred_imports.py`, as a
   guard that no `torch` import moved to module level.

## Risks and open items

- **Third-party CPU support is unverified here.** `torch` is not installed in
  this environment, so no detector was run on CPU during the investigation.
  The PhenoTypic code paths contain no CUDA-only constructs (no `autocast`,
  `.half()`, `.cuda()`, or `bfloat16`; checked by grep over
  `src/phenotypic/detect/nn/`), but the upstream `sam2`, `micro_sam` and
  `transformers` model code was not audited. Before merging, run each detector
  once with `device="cpu"` on `load_synth_yeast_plate()` on a CPU-only node.
- **Runtime.** CPU inference of the larger checkpoints may be slow enough to
  look like a hang on a full plate. No timing figures are claimed here; measure
  during the CPU smoke run and put the numbers in the docs rather than an
  adjective.
- **Continuation.** The change does not alter any serialized field, so work-id
  digests and continuation are unaffected; a run that previously failed at
  model load will resume from its completed Stage-1 stores.
