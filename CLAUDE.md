# CLAUDE.md

## Agentic AI File Rules

- **Design Specifications** go in @docs/superpowers/specs/ under their own dated folder
  with the topic name
- **Mult-doc design plans** go in @docs/superpowers/plans/ under their own dated folder
  with the topic name
- **html artifacts** go in @docs/superpowers/artifacts/ under their own dated folder
  with the topic name

## Quick Start

**`uv` is the sole package manager and runner.** Never use bare `python` or `pip`.

- `uv run <cmd>` — run commands
- `uv add <package>` (or `--group dev`) — add dependencies
- `uv sync` — sync env (after checkout or in new worktrees)
- `uv sync --group dev --group test-qt --group docs --extra gui --extra napari` — full
  dev env
  (`test-qt` + the `napari` extra are required for the napari/Qt widget tests)
- `source .venv/bin/activate` — manual venv activation

### Linting & Type Checking

- `uv run mypy src/phenotypic` — type checking
- `uv run ruff check --fix` — format and lint

### CLI

- `uv run python -m phenotypic` — single pipeline on images/directories (parallel,
  SLURM, resume)
- `uv run python -m phenotypic --mode process --layer {rgb|gray|detect_mat|objmap}` —
  apply-only export: runs `pipeline.apply()` and writes ONE image layer per input
  (via the accessor `imsave` — `rgb` integer TIFF, `gray`/`detect_mat` float TIFF,
  `objmap` 16-bit raw-label PNG), mirroring the input tree. Skips
  measurement/deliverables/QC/dashboard; machine-state lives under `.phenotypic/`.
  Full local + SLURM + resume reuse.
- **GPU detectors stage automatically:** when a pipeline contains a `GpuDetector`,
  `python -m phenotypic` runs detection as three internal stages — CPU preprocess →
  resident-model GPU detect → CPU measure — reusing the per-image HDF. Stage 2 writes a
  per-image `.npy` objmap **sidecar** (HDF opened read-only); Stage 3 merges it into the
  final HDF, measures, and deletes the sidecar. The output folder is identical to a
  single-pass run; resume is content-defined (HDF → sidecar → parquet) and progress is
  stage-tagged. `--mode process --layer objmap` exports objmaps after Stages 1–2.
  On SLURM, the three stages submit as a **3-link `afterany` dependency chain** with
  per-stage resources: Stages 1 & 3 on the CPU `--slurm` profile, Stage 2 as a GPU array
  of resident-model shard-workers. Stage 2 survives walltime — each sidecar write is
  atomic and the worker SIGTERM-resubmits its shard, so a `TIMEOUT` never loses work.
  Staged GPU flags (Spec 1 §10):
    - `--gpu-slurm key=value` — Stage-2 GPU SBATCH profile; **inherits/deltas over
      `--slurm`** (put a separate GPU partition/account here); auto-adds
      `slurm_gpus_per_node=1` (explicit `=0` runs the GPU stage on a CPU partition).
    - `--gpu-shards N` (default 1) — parallel whole-GPU Stage-2 tasks (SLURM-only).
    - `--gpu-workers-per-gpu W` (default 1) — replicas packed per GPU (small-model
      fill).
    - `--gpu-batch-size N|auto` (default 1) — images/forward (batchable models; `auto`
      VRAM-probe lands in Spec 2).
- `uv run python -m phenotypic.tune run spec.json -i <images> -o <out>` —
  hyperparameter tuning (grid/random + Optuna), distributed via `--slurm`/
  `--storage-url`

### GUI hub

- `uv run phenotypic-gui --root ./images --port 8050` — unified hub: builder +
  results viewer + run console mounted under one URL via Werkzeug
  `DispatcherMiddleware`. SSH-tunnel from a workstation:
  `ssh -L 8050:localhost:8050 user@cluster`.
  For Open OnDemand-style path-stripping proxies, pass only the browser-visible
  path as `--url-prefix`, e.g.
  `--url-prefix /node/hz01/30099/`, then open the full OOD URL
  `https://ondemand.hpcc.ucr.edu/node/hz01/30099/`.
- `uv run python -m phenotypic.gui --root ./images` — equivalent module entry.
- Standalone tools still work: `python -m phenotypic.gui.builder`,
  `python -m phenotypic.gui.results_viewer`, `python -m phenotypic.gui.run_console`.
- Note: `phenotypic gui` (no hyphen, as a subcommand of the existing CLI) is NOT
  supported. Use `phenotypic-gui` or `python -m phenotypic.gui`.

#### Adding GUI features

Two CI-gated ledgers (`FEATURES.md`, `WORKFLOWS.md`) plus the
tutorial-screenshot capture script must stay in sync when you change GUI
chrome — see the **`gui-tutorial-capture`** skill.

---

## Architecture

**Purpose:** Modular image processing for arrayed colony phenotyping on solid media (
agar plates).
**Philosophy:** Accuracy over speed. Be mindful of memory — images are large and
operations copy data; avoid unnecessary intermediate allocations.

### Five Layers

1. **Image Data** — `Image`/`GridImage` with accessor pattern, lazy evaluation, caching
   (`image.rgb[:]`, `image.detect_mat[:]`, `image.color.Lab[:]`).
   See [_core/CLAUDE.md](src/phenotypic/_core/CLAUDE.md).
2. **Operation ABCs** — `_operate(image) -> image` interface.
   See [abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md) for hierarchy and reference
   implementations.
3. **Pipeline** — `ImagePipeline` chains operations, batch execution, YAML/JSON
   serialization, automatic benchmarking.
4. **Enhancement** — preprocessing ops on `detect_mat`; RGB/gray unchanged.
5. **Post-Measurement** — `post/` transforms DataFrames in the final stage of
   `ImagePipeline.measure()`.
   `analysis/` provides standalone statistical tools (edge correction, growth curves,
   outlier removal) for exported data.

### Design Decisions

- **Operations are pydantic v2 models:** every operation and analyzer is a
  `pydantic.BaseModel` rooted at `BaseOperation`. Parameters are **annotated
  class-level fields** — there is no hand-written `__init__`; construction is
  **keyword-only**; invalid input raises `pydantic.ValidationError` (a `ValueError`
  subclass). Algorithm bodies (`_operate`/`apply`/`measure`) are unchanged. Every class
  exposes a machine-readable contract via `model_json_schema()`, with field
  descriptions auto-derived from the Google-style `Args:` docstring. To add a
  parameter, declare a typed field; put input normalization and guards in a
  `field_validator`, never an `__init__`. Raw-array params use the reusable
  `NdArrayField` type; operation-valued params use `OperationField` (both in
  `sdk_/typing_.py`).
- **Public API:** only `__init__.py` exports are public; `_implementation.py` files are
  private.
- **Immutability:** operations return copies; never modify `image.rgb`/`image.gray`
  directly.
- **Explicit:** use `ImagePipeline` for multi-step workflows; no hidden state.
- **Domain-specific:** built for microbe phenotyping; use microbiology context in
  docs/examples.
- **Duck typing** for type checks; **explicit matplotlib** (no implicit pyplot).
- **Reproducibility:** `to_json()`/`from_json()` serialization; fixed random seeds.
- **Cross-platform:** macOS, Windows, Linux; use try/except for platform-specific
  imports.

---

## Code Style

- **Google-style docstrings** everywhere. Order and ImageOperation conventions live
  in [abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md).
- **Measurement docstring split (`MeasureFeatures` vs `MeasurementInfo`):** the two
  carry different documentation. A `MeasureFeatures` op's docstring explains **what
  its parameters mean** and gives a **high-level overview** of the measurements it
  emits (what the operation does, when to use it). The `MeasurementInfo` enum members
  carry the **detailed, per-column explanation** of the measurements themselves —
  what each value is, how it is computed, and **how to read the measurement output**
  (units, range, column header). Don't duplicate the per-column detail onto the
  measurer or the operation overview onto the enum. The deliverables `README.md`
  generator reads the measurer→`MeasurementInfo` mapping and emits each member's
  `desc` as the public column documentation, so the enum `desc` is what users see.
  (Authoring rule for enum members — author `label`/`desc` only, never `bio_desc` —
  is in **Gotchas** below.)
- All doctest examples must be **runnable** using `load_synth_yeast_plate()`; use
  microbiology context (colony visibility, edge sharpness, mask quality).
- **Never create** separate example files/notebooks — examples go in docstrings.
- Don't create summary documents unless explicitly asked.
- **Explicit naming:** no generic `main()`, `run()`, `process()` — name after what it
  does.
- Break large functions into smaller, testable helpers with private methods.
- For batch processing, use the CLI (`python -m phenotypic`) not custom scripts.
- Toggle integrity validation via `phenotypic.settings` (`VALIDATE_OPS`,
  `set_validate_ops()`, or the `validation()` context manager).

---

## Module Guides

- [_core/CLAUDE.md](src/phenotypic/_core/CLAUDE.md) — Image class, accessors
- [_cli/CLAUDE.md](src/phenotypic/_cli/CLAUDE.md) — execution strategies, staged GPU
  engine, SLURM chaining
- [abc_/CLAUDE.md](src/phenotypic/abc_/CLAUDE.md) — ABC hierarchy, implementation
- [schema/CLAUDE.md](src/phenotypic/schema/CLAUDE.md) — public measurement schema (
  `MeasurementInfo` base + header enums)
- [sdk_/CLAUDE.md](src/phenotypic/sdk_/CLAUDE.md) — mixins, utilities
- [enhance/CLAUDE.md](src/phenotypic/enhance/CLAUDE.md) — enhancer conventions
- [gui/CLAUDE.md](src/phenotypic/gui/CLAUDE.md) — GUI sub-apps, shared `_config.py`
  constants, `_design.py` tokens
- [DESIGN.md](DESIGN.md) — dashboard & plot style guide
- `src/phenotypic/post/`, `src/phenotypic/analysis/` — no sub-CLAUDE.md

## Key Files

- `src/phenotypic/_core/_image.py` — `Image` class
- `src/phenotypic/_core/_image_pipeline.py` — Pipeline implementation
- `src/phenotypic/abc_/` — Operation interfaces
- `src/phenotypic/__main__.py` — CLI entry point
- `src/phenotypic/_cli/_cli_execution_strategies.py` — strategy dispatch (
  `create_execution_strategy`)
- `src/phenotypic/_cli/_cli_staged_strategy.py`, `_cli_staged_slurm.py`,
  `_cli_staged_workers.py` — staged GPU engine (local + SLURM);
  see [_cli/CLAUDE.md](src/phenotypic/_cli/CLAUDE.md)

## Closed Value Sets & Operation Parameters

Conventions for closed value sets (`Enum`/`Literal`), `MeasurementInfo` /
`ConstantLabels`, parameterized strings, and the tune annotation-coverage gate
live in the **`adding-an-operation`** skill — use it when adding or editing any
operation parameter.

## Gotchas

- Some packages excluded on Windows: `rawpy`, `pympler`, `jupyter` — use try/except.
- External tools: ExifTool (raw metadata), Pandoc (doc builds).
- **Operations use `.apply()`, not `__call__`:** `op.apply(image)` is correct;
  `op(image)` raises `TypeError`.
- **GPU pipelines stage internally:** a `GpuDetector` in a CLI run triggers the staged
  engine (preprocess → GPU → measure) with a per-image objmap **sidecar**, not per-image
  processing; the resident model loads once. Notebook `op.apply(image)` is unchanged.
  See [_cli/CLAUDE.md](src/phenotypic/_cli/CLAUDE.md) for the strategy dispatch +
  stages.
- **Staged-GPU env vars:** `PHENOTYPIC_PRELOAD_MODULES` lets a fresh SLURM worker
  resolve
  custom op classes defined outside the `phenotypic` namespace (a self-registering
  module
  it imports before `from_json`); `PHENOTYPIC_ACCEPT_MODEL_LICENSE` +
  `require_license_acceptance`
  (`detect/nn/_checkpoint_manager.py`) gate gated-weight downloads — the hook for Spec
  2's
  SAM3/DINOv3. Third-party licensing scaffolding: root `NOTICE` + `licenses/` +
  `MANIFEST.in`.
- **Operations are keyword-only constructed:** `OtsuDetector(ignore_zeros=True)`, not
  `OtsuDetector(True)` — pydantic models take no positional args. Unknown kwargs and
  invalid values raise `pydantic.ValidationError`.
- **Measurement columns are category-prefixed:** `Size_Area`, `Shape_Circularity`,
  `Intensity_MeanIntensity`, etc. The header enums are the **public**
  `phenotypic.schema` package (`from phenotypic.schema import SHAPE, SIZE, ...`);
  the old `phenotypic.sdk_.measurement_info` path was removed.
  `MeasurementInfo.get_labels()` returns unprefixed names; `get_headers()` returns the
  prefixed column names used in DataFrames.
- **Authoring `MeasurementInfo` members:** members are declared with
  `Entry(label, desc, *, bio_desc="", image=None)` (the `Entry` value type in
  `phenotypic.schema`). When adding a new member or editing one, only author/edit
  the **`label`** (name) and **`desc`** (the technical/algorithm description of
  what is computed). **Never author or auto-fill `bio_desc`**, and leave `image`
  unset — biological-relevance claims must be written and verified by a human
  domain author, not generated. Agents may scaffold the `Entry(...)` and populate
  `label`/`desc`, but must leave `bio_desc=""`/`image=None` for human authoring.
- **Analysis classes use `.analyze()`:** `EdgeCorrector.analyze(df)`,
  `LogGrowthModel.analyze(df)` — not `.fit()` or `.correct()`.
- **`num_objects` is on `Image`**, not on the `objmap` accessor: use
  `image.num_objects`.
- **Output layout (`deliverables/`):** user-facing run outputs live under
  `<output>/deliverables/` (measurements, analysis, dashboards, overlays, and
  the durable QC + curation state under `deliverables/qc/`); per-image
  parquets/HDF and run state stay at the output root. `master_measurements.*`
  is the clean pre-post, metadata-free archive; `measurements.*` is the
  post-applied mirror the GUI reads/curates — feed analysis and dashboards from
  the **mirror**, not the master. Always resolve paths via the
  `phenotypic.sdk_` helpers (never hand-join names), and route any FINAL master
  write through `finalize_post_master_outputs`. Full file inventory,
  master-vs-mirror rules, and the finalize/chunk-writer carve-out are in
  [_cli/CLAUDE.md](src/phenotypic/_cli/CLAUDE.md).
