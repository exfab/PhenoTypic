# CLAUDE.md

## Agentic AI File Rules

- **Design Specifications** go in @docs/superpowers/specs/ under their own dated folder
  with the topic name
- **Mult-doc design plans** go in @docs/superpowers/plans/ under their own dated folder
  with the topic name
- **html artifacts** go in @docs/superpowers/artifacts/ under their own dated folder
  with the topic name
- **Executable logic-validation scripts** go in @docs/superpowers/logic_validation_scripts/
  under the change's own dated topic folder (same name as the matching spec/plan). One
  runnable script per subject, named `<subject>.py`; it re-derives the load-bearing numeric
  claims from scratch, depends only on the stdlib + numpy/scipy (never imports `phenotypic`),
  exits non-zero on failure, and is committed alongside the spec. Write one whenever a design
  rests on a numeric invariant a reader would otherwise take on faith — it is what keeps
  specs, plans, and tests from drifting apart across a long change. See the
  **`porting-a-reference-algorithm`** skill for the surrounding procedure. Precedent:
  `docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py` — an existing
  script still co-located with its spec; `logic_validation_scripts/` is the going-forward home.

## Quick Start

**`uv` is the sole package manager and runner.** Never use bare `python` or `pip`.

- `uv run <cmd>` — run commands
- `uv add <package>` (or `--group dev`) — add dependencies
- `uv sync` — sync env (after checkout or in new worktrees)
- `uv sync --group dev --group test-qt --group docs --extra gui --extra napari` — full
  dev env
  (`test-qt` + the `napari` extra are required for the napari/Qt widget tests)
- `source .venv/bin/activate` — manual venv activation

### Running tests

Use the **`run-phenotypic-test`** skill before any non-trivial pytest invocation —
the full `tests/unit` suite, anything on a compute node, anything headless, or any
run whose numbers you intend to quote as a baseline. Four traps here produce a
**wrong answer** rather than a slow one: a missing `QT_QPA_PLATFORM=offscreen`
aborts the interpreter at 79% with no summary; `-n auto` reads the node's core
count instead of the allocation's and manufactures timeout failures; the default
`addopts` streams uncaptured output and can triple the runtime when stdout is a
file on shared storage; and `-x` silently truncates a run that then gets recorded
as a baseline. The suite is ~65 minutes, not two — so it is a Slurm job
(**`slurm-job`** skill), with a committed batch script at
`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`.

### Linting & Type Checking

- `uv run mypy src/phenotypic` — type checking
- `uv run ruff check --fix <paths you changed>` — lint + autofix. **Always pass explicit
  paths.** Bare `ruff check --fix` walks the entire repo and rewrites files you never
  touched, burying your change in unrelated churn (and, in a parallel session, clobbering
  someone else's in-flight work). `[tool.ruff] extend-exclude` keeps it off the vendored
  upstream sources under `docs/superpowers/**/refs`, but nothing protects the rest of the
  tree. If you already ran it bare: `git status`, then revert everything outside your
  change before committing.

### CLI

- `uv run python -m phenotypic` — single pipeline on images/directories (parallel,
  SLURM, automatic continuation)
- `uv run python -m phenotypic --mode process --layer {rgb|gray|detect_mat|objmap}` —
  apply-only export: runs `pipeline.apply()` and writes ONE image layer per
  input, mirroring the input tree. **Output is a single-series OME-Zarr store**
  (`<stem>.ome.zarr/`) for `rgb`/`gray`, a float TIFF for `detect_mat`, and a
  16-bit raw-label PNG for `objmap`. `--process-format {tiff,zarr}` overrides;
  `--layer objmap --process-format zarr` and `--layer detect_mat
  --process-format zarr` are both refused, for different reasons — NGFF has no
  standalone label-image form, and PhenoTypic's store writer requires a primary
  series (`rgb` or `gray`). The store carries the pipeline that produced it in
  `attributes.phenotypic.provenance` and omits `image_class`, so
  `Image.load_zarr` refuses it and points at `Image.imread`, which reads any
  OME-Zarr — PhenoTypic's or a third party's — as plain pixels. A published
  store is bit-reproducible: `applied_at_utc` and `duration_seconds` are
  omitted from its journal, so two identical runs write byte-identical stores.
  Provenance travels one hop — processing a store into another store resets the
  second store's journal to that second pipeline only, it does not chain. A
  tree of stores is valid `--input`. Skips measurement/deliverables/QC/
  dashboard; machine state lives under `.phenotypic/`. Full local + SLURM
  continuation reuse; switching `--process-format` invalidates continuation
  rather than reusing outputs of the other kind. Run the same command again
  after an interruption or when new compatible inputs appear; there is no
  `--resume` flag.
- `uv run python -m phenotypic --mode migrate --output <run>` — convert a legacy
  `.h5` output tree to OME-Zarr stores **in place**, in two passes: pass 1 migrates
  the non-image metadata targets (`csv`/`parquet`/`json`/`frame`, never `.h5`), pass 2
  converts each per-image `results/<ds>/hdf/<stem>.h5` to
  `results/<ds>/zarr/<stem>.ome.zarr` and re-publishes its marker. Sources are
  **kept** by default; `--delete-sources` is opt-in and gated on a value-level
  re-read. Re-running after an interruption *is* the recovery procedure.
  **Every other mode that writes or reprocesses (`full`, `measure`, `recompile`,
  `process`) refuses an unconverted tree** with a pointer to this command —
  conversion rewrites the whole results tree, so it is typed deliberately rather
  than triggered as a side effect. Per-image storage is OME-Zarr only: `save2zarr`
  / `load_zarr` / `load_layer_zarr` / `save_intermediate_zarr` replaced the HDF
  quartet outright, and there is no `Image.save2hdf5`. See
  `docs/source/how_to/pages/zarr_storage.md`.
- **GPU detectors stage automatically:** when a pipeline contains a `GpuDetector`,
  `python -m phenotypic` runs detection as three internal stages — CPU preprocess →
  resident-model GPU detect → CPU measure — reusing the per-image OME-Zarr store.
  Stage 2 reads that store **read-only** and never writes into it; its result is a
  **Stage-2 signal** under `.phenotypic/progress/`: the retained **raw** detector
  output `stage2_raw/<ds>/<stem>.npy` plus a consumable **token**
  `stage2_done/<ds>/<stem>.json`. Stage 3 replays the raw array, measures,
  re-promotes the store, and consumes the token and then the raw array. The output
  folder is identical to a single-pass run; continuation is content-defined
  (valid store → complete Stage-2 signal → atomic Stage-3
  completion marker) and progress is
  stage-tagged. `--mode process --layer objmap` exports objmaps after Stages 1–2.
  On SLURM, the stages submit through an **epoch-fenced recoverable controller**:
  Stages 1 & 3 use the CPU `--slurm` profile, and Stage 2 is a GPU array
  of resident-model shard-workers. Stages 1 & 3 auto-split into
  `ceil(n_images / min(MaxArraySize, MaxSubmitJobs - 2))` chunks (Stage 2 is never
  chunked). Only Controller 0 is submitted initially; it pre-arms a dependent recovery
  controller before launching Stage-1 chunk 0. Each controller records the next job in
  an append-only ledger. After a Stage-2 timeout, the controller derives remaining work
  from complete Stage-2 signals and submits another round. No worker signal handler or self-requeue
  is used. Without `--wait`, the CLI reports submission only; the dependent finalizer is
  the sole publisher of aggregated outputs and the completion marker.
  Staged GPU flags (Spec 1 §10):
    - `--gpu-slurm key=value` — Stage-2 GPU SBATCH profile; **inherits/deltas over
      `--slurm`** (put a separate GPU partition/account here); auto-adds
      `slurm_gpus_per_node=1` (explicit `=0` runs the GPU stage on a CPU partition).
    - `--gpu-shards N` (default 1) — parallel whole-GPU Stage-2 tasks (SLURM-only).
    - `--gpu-workers-per-gpu W` (default 1) — reserved for future replica packing;
      the current staged worker runs one resident model per GPU shard.
- `uv run python -m phenotypic.tune run spec.json -i <images> -o <out>` —
  hyperparameter tuning (grid/random + Optuna), distributed via `--slurm`/
  `--storage-url`

#### SLURM array auxiliary work

- Do not submit scheduler **sidecar jobs** in parallel beside an active ordinary
  array. Allocation/submission bounds are already consumed by the array cohort.
- Route ancillary work through reserved trigger entries inside the array task
  list, following the existing `__PHENOTYPIC_CHECKPOINT__` and
  `__PHENOTYPIC_MANIFEST__` dispatch pattern. Count every trigger entry when
  sizing chunks against `MaxArraySize`, and test that no standalone parallel job
  is submitted.
- This rule concerns scheduler jobs, not the staged GPU Stage-2 signal files
  (the retained raw `.npy` and its token). A terminal `afterany` finalizer is
  also not a parallel sidecar.
- See `src/phenotypic/_cli/CLAUDE.md` for the full routing contract. Root
  `AGENTS.md` is a symlink to this file and therefore carries the same rule.

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
- **Two pixel paths, not one.** The results viewer's Plate and Colony surfaces
  and the builder's node preview read per-image OME-Zarr chunks in the browser
  through Viv/deck.gl (`/zarr/...`, `/preview-zarr/...`); Browse and the
  builder's point picker keep libvips → DZI → OpenSeadragon. The results viewer
  renders no server-side pyramid and caches no rendered PNG. Which surface uses
  which, and the two rules that go with it (never hard-code the series or label
  path, never recompute the pyramid), are in
  [gui/CLAUDE.md](src/phenotypic/gui/CLAUDE.md).

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

## Porting a Reference Algorithm

Some operations here are **ports** of an external algorithm (a paper, a reference
implementation), not fresh code — e.g. `FocusEdgePhase` and `FocusEdgeMonogenicPhase`
transcribe Kovesi's `phasecong3` / `phasecongmono`. When matching code to an outside
source, use the **`porting-a-reference-algorithm`** skill *before* making any claim about
what that source does. It is a checkable procedure: assemble every reference locally →
cite `file:line` for each claim → diff line-by-line (never inspect-and-summarize) → pin
behaviour with a golden fixture (all outputs) **and** behavioural controls → mutation-test
the suite → prove the fixture fails when the bug it guards is reintroduced → one
drift-register row per deviation, however small. The executable check it produces belongs
under `docs/superpowers/logic_validation_scripts/` (see **Agentic AI File Rules**).

**Vendored reference sources are read-only.** The upstream copies under
`docs/superpowers/specs/*/refs/` are the artifact every `file:line` citation and
line-by-line diff resolves against. They must stay **byte-identical to upstream** — never
lint, format, autofix, "tidy", or fix a real bug in them. Their imports, style, and even
their mistakes are the evidence; edit one and every claim ever cited against it silently
stops meaning anything, with nothing failing to tell you. `[tool.ruff] extend-exclude`
enforces this for ruff, but the rule binds regardless of the tool.

## Gotchas

- **`imread` vs `load_zarr` on an OME-Zarr store:** the verb decides, never the
  file. `Image.imread(store)` always reads plain pixels — PhenoTypic's own
  output, or a napari/QuPath/`bioformats2raw` export — and refuses rather than
  guessing when a store cannot be projected onto a 2-D image (a real `t` or `z`
  axis, a channel count that is neither 1 nor 3, an HCS plate); pass
  `t=`/`z=`/`c=`/`series=` to choose explicitly. `Image.load_zarr(store)`
  always restores run state and raises on a store with no
  `phenotypic.image_class`. NGFF has no RGB type: `rgb` is a 3-length `channel`
  axis ordered **before** the space axes, so stores are planar `(3,H,W)` and
  `imread` transposes to `(H,W,3)`.
- Some packages excluded on Windows: `rawpy`, `pympler`, `jupyter` — use try/except.
- External tools: ExifTool (raw metadata), Pandoc (doc builds).
- **Operations use `.apply()`, not `__call__`:** `op.apply(image)` is correct;
  `op(image)` raises `TypeError`.
- **GPU pipelines stage internally:** a `GpuDetector` in a CLI run triggers the staged
  engine (preprocess → GPU → measure) with a per-image Stage-2 signal (a retained
  raw `.npy` plus a consumable token under `.phenotypic/progress/`), not per-image
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
- **Metadata queries use schema ownership, never string prefixes:** determine whether a
  header or label is metadata, and which metadata type owns it, with
  `metadata_member_for_header()`, `metadata_owner_for_header()`,
  `metadata_member_for_label()`, or `metadata_owner_for_label()`. When working with
  schema classes directly, check `MetadataInfo` inheritance. Do not use
  `startswith("Metadata_")`, prefix splitting, category-name comparisons, or other
  serialized-string parsing as a semantic metadata check. String handling belongs only
  in the centralized compatibility and canonicalization helpers.
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
  the durable QC + curation state under `deliverables/qc/`); per-image image
  state lives in `results/<ds>/zarr/<stem>.ome.zarr/`, with authoritative
  object measurements at `tables/measurements/table.parquet` inside each
  store. Forward runs do not create external per-image measurement Parquets.
  `master_measurements.*` is the exact pre-post concatenation of authorized
  embedded tables (already metadata-joined measured rows);
  `measurements.*` appends metadata-only phantoms once and is the post-applied
  mirror the GUI reads/curates — feed analysis and dashboards from the
  **mirror**, not the master. Always resolve paths via the
  `phenotypic.sdk_` helpers (never hand-join names), and route any FINAL master
  write through `finalize_post_master_outputs`. Full file inventory,
  master-vs-mirror rules, and the finalize/chunk-writer carve-out are in
  [_cli/CLAUDE.md](src/phenotypic/_cli/CLAUDE.md).
- **Metadata startup snapshot:** full runs and recompile copy a configured
  `--metadata` CSV byte-for-byte to `deliverables/metadata.csv` before local
  work or SLURM submission. Treat that file as input provenance: **never rewrite
  it as a side effect** of any other operation. Finalization, chunk writers, and
  `--mode recompile` normalize legacy headers **only in memory**.
  **There is no exception, including `--mode migrate`.** An earlier draft of
  this rule carved one out — migrate would rewrite `deliverables/metadata.csv`
  with canonical headers after copying the original to
  `deliverables/metadata.original.csv`. That was **withdrawn** (spec D9 /
  FLOW-4) and never implemented. A snapshot that is sometimes rewritten is not
  provenance, and "the original is recoverable over there" is a weaker
  guarantee than "the bytes you supplied are still the bytes on disk".
  `--mode migrate` instead **emits a canonical view alongside** the snapshot,
  at `deliverables/metadata.canonical.csv`, and leaves `metadata.csv`
  byte-identical (pinned by
  `test_the_metadata_snapshot_is_byte_unchanged_by_a_full_migrate`).
  `metadata.original.csv` does not exist and must not be created. Generated
  scientific tables still emit only the canonical flat `Metadata_<Label>`
  namespace.
