# Plotting Hard-Cut Implementation Plan

- **Date:** 2026-07-16
- **Status:** Implemented
- **Spec:** `docs/superpowers/specs/2026-07-16-plotting-subpackage/design.md`

Implement this plan cluster by cluster. Run focused tests and an independent diff
review after every cluster. Use `uv` for all Python commands and preserve unrelated
changes already present in the active worktree.

## 1. Outcome and pilot boundary

Build a mergeable plotting backend that exposes the real integration cost rather than
hiding it behind compatibility shims. The pilot includes:

- public plotting capabilities under `phenotypic.abc_.plotting`;
- runtime bindings, outputs, adapter, registry, coordinator, and concrete plots under
  `phenotypic.plotting`;
- identity-preserving `ImagePipeline.plots` serialization;
- named analysis artifacts and dynamic input resolution;
- automatic CLI plot emission into `deliverables/plots`;
- CLI-side QC plot subjects and analyzed-check reuse;
- hard removal of `FigureProvider`, plotting-object `dash()` / `dashboard()`,
  `Image.plot`, and `--save-inspect`;
- a deep `MeasureSymmetricZones` conversion, the `ModelFitter` analysis seam, and the
  new `PlotMeasTimeSeries` class.

The implementation includes event-driven GUI refresh for configured measurement,
analysis, and QC plots, plus deterministic multi-page GUI selectors. Live cluster
submission remains outside the verification boundary. Recompile emits aggregate plots
only because it does not rerun measurers or rebuild per-image caches.

## 2. Mandatory blast-radius gate

Before production edits, dispatch an independent read-only explorer to audit:

1. every `FigureProvider`, `dash()`, `dashboard()`, `Image.plot`, plotter-registry, and
   `--save-inspect` definition and consumer;
2. pipeline construction, serialization, nested pipelines, staged splitting, worker
   loading, recompile, and GUI pipeline loading;
3. fixed `analysis.*` constants, readers, writers, standalone-bundle resolution,
   README generation, docs, and tests;
4. local, measure-only, staged GPU, SLURM command-generation, aggregation, and
   recompile plot-dispatch seams;
5. `ImagePlotHandler` responsibilities that must survive `Image.plot` removal;
6. public exports, import-time dependency rules, notebooks, module guides, and
   generated documentation.

The explorer must return file-and-line evidence, protected `Image.dash()` allowlist
entries, the staged call graph, and ten essential files. The implementing agent reads
those files and reconciles missing work into this plan before Cluster 1.

## 3. Public interfaces

### 3.1 Plotting capabilities

Create `phenotypic.abc_.plotting` as the only public import path for:

```python
PhtPlot
PlotImage
PlotMeas
PlotAnalysis
PlotQc
Control
FigureSpec
BoundFigures
figure
```

The five mixins are fieldless, methods-only cooperative mixins. They add no Pydantic
fields, constructors, abstract methods, or persistent state.

```python
class PhtPlot:
    def inspect(
        self,
        subject: Any = None,
        *,
        for_save: bool = False,
        **overrides: Any,
    ) -> Any: ...

    def report(self, subject: Any = None, **overrides: Any) -> Any: ...
```

There are no `dash()` or `dashboard()` compatibility aliases.

### 3.2 Runtime plotting package

Create `phenotypic.plotting` with public exports for:

- `FigureLike`, `PlotPage`, `PlotOutput`, and `FigureAdapter`;
- `PipelineObjectRef`, `PlotBinding`, `MeasurementInput`, and `AnalysisInput`;
- `QcPlotSubject`, `AnalysisRegistry`, and the plotting coordinator;
- `PlotDiagnostics`, `PlotDetectModes`, and `PlotMeasTimeSeries`.

`PlotPage` and `PlotOutput` are frozen runtime dataclasses and never enter pipeline
JSON. `FigureLike` accepts Plotly `go.Figure` and Matplotlib `Figure`; add Matplotlib as
a direct dependency because it becomes part of the public contract.

### 3.3 Pipeline configuration

Add `plots: list[PhtPlot | PlotBinding] | None = None` to `ImagePipeline`. Normalize the
stored value to an ordered list of `PlotBinding` objects.

- Raw objects already present in `ops`, `meas`, `post`, `filters`, or `model` become
  references only when object identity matches with `is`.
- Recipe-backed QC uses an explicit `qc.<instance_id>` reference because the pipeline
  stores recipes rather than persistent check instances.
- Inline plots store `module`, `qualname`, and `model_dump(mode="json")` parameters.
- Local classes containing `<locals>` fail serialization. External plots must be in an
  importable or explicitly preloaded module.
- Deserialize normal slots first, then resolve plot references against the exact
  reconstructed objects.
- Duplicate plot IDs, unresolved references, incompatible lifecycle inputs, or
  non-plot-capable objects fail validation with the plot ID in the error.

Only `PlotAnalysis` and `PlotQc` bindings accept `input`. They default to
`MeasurementInput()` and may select `AnalysisInput(analysis_id=...)`. `PlotMeas`
always consumes the post-applied mirror and has no input setting.

## 4. Implementation clusters

### Cluster 0: reconcile the audit

- Run the mandatory blast-radius subagent.
- Read its essential files and update this plan/spec if it finds contradictions.
- Record a protected allowlist for `Image.dash()` and channel accessor `dash()` calls.

### Cluster 1: plotting foundations and report hard cut

- Move the complete `FigureProvider` implementation into
  `phenotypic.abc_.plotting.PhtPlot` and delete `FigureProvider` after all consumers
  migrate.
- Keep imports lazy so importing the ABC package does not import Plotly, Matplotlib,
  Dash, ipywidgets, or the runtime `phenotypic.plotting` package.
- Migrate custom report composers individually:
  - grid-fit reporting;
  - color-correction reporting and operation wrappers;
  - orientation-zone reporting and operation wrappers.
- Convert `MeasureSymmetricZones` to `MeasureFeatures, PlotImage`. Preserve its
  per-image private cache, controls, theming, `for_save` flattening, Pydantic schema,
  and operation JSON.
- Convert `ModelFitter` to `SetAnalyzer, PlotAnalysis, ABC`:
  - extract its existing Plotly logic into a private builder;
  - expose keyword-only `inspect(..., for_save=False)` and `report(**plot_kwargs)`;
  - prevent `tmax` from being mistaken for a subject;
  - reuse populated private analysis state when the plot is the producer;
  - keep Matplotlib `show()` for non-plotting fallback consumers.
- Rename plotting-object `dash()` / `dashboard()` to `report()` without changing
  `Image.dash()` or image-channel `dash()` methods.

### Cluster 2: output model and backend adapter

- Normalize a raw supported figure to one `default` page and preserve ordered
  multi-page `PlotOutput` objects.
- Provide canonical typed group keys independent from display labels and filenames.
- Save Plotly through Kaleido and Matplotlib through `savefig`; restore caller-visible
  figure state and close Matplotlib figures on success or failure.
- Convert Plotly figures to `dcc.Graph` and Matplotlib figures to in-memory PNG-backed
  `html.Img` without temporary files.
- Sanitize one path component at a time, reject traversal, detect case-folded
  collisions, and append a stable short hash derived from the canonical page key.
- Publish per-plot page manifests atomically and include only successfully written
  pages. A failed page does not suppress its siblings.

### Cluster 3: bindings and identity-preserving serialization

- Add pipeline field validation, binding normalization, getters/setters, JSON envelope
  encoding, and post-slot reference resolution.
- Update nested pipeline and pipeline-as-operation serialization.
- Reject a nested pipeline with non-empty `plots` when it is used as an operation, and
  reject top-level references into nested pipeline internals. Recursive plot execution
  and namespacing are outside this pilot.
- Extend class resolution to built-in `phenotypic.plotting` types and qualified custom
  classes.
- Preserve bindings during staged splitting:
  - measurer-bound and inline `PlotImage` plots run in Stage 3;
  - post-GPU operation-bound `PlotImage` plots also run in Stage 3;
  - aggregate bindings remain with finalization;
  - pre-GPU or GPU-detector `PlotImage` references fail early in this pilot.
- Ensure the canonical persisted pipeline contains the complete aggregate bindings,
  while worker pipelines contain only bindings valid in that stage.

### Cluster 4: named analyses and dynamic resolution

- Replace fixed `analysis.csv` / `analysis.parquet` writes with
  `<analysis-id>.csv` / `<analysis-id>.parquet`. The current model defaults to its
  concrete class name, such as `LinearLagModel`.
- Validate analysis IDs as 1-128 safe ASCII stem characters, starting with a letter;
  reject separators, whitespace, `.` / `..`, and unsafe IDs rather than sanitizing.
- Add path helpers and `analysis_manifest.json` with schema version, class, relative
  paths, columns, row count, and SHA-256 checksums.
- Before replacing either artifact, persist a recovery journal under the shared
  generation lock. Atomically replace each artifact, publish the manifest last, and
  remove the journal only after commit or rollback. Readers and later writers recover
  an interrupted journal before proceeding, preserving crash-atomic class-named pairs.
- Make `_emit_analysis_outputs` return a runtime analysis result containing the ID,
  table, artifacts, and manifest entry so finalization does not discard the in-memory
  result.
- Resolve `AnalysisInput` on every refresh from the in-memory registry first, then the
  authoritative manifest/Parquet artifact. Read legacy `analysis.parquet` only when no
  manifest exists.
- Update current GUI analysis writes, status messages, standalone-bundle paths, README
  generation, and other fixed-name consumers.
- Remove legacy analysis constants/helpers from public SDK and GUI exports. Keep their
  filenames private only for resolver fallback when no manifest exists.

### Cluster 5: CLI coordinator and QC subjects

- Replace the `--save-inspect` duck-typed loop with configured lifecycle dispatch:
  1. per-image apply and measure, then `PlotImage.inspect(for_save=True)` on the same
     `Image` instance;
  2. finalize the post-applied/metadata-joined mirror, then `PlotMeas`;
  3. run and publish analyses, register their tables, then `PlotAnalysis`;
  4. run and publish QC, then `PlotQc`.
- Change the QC runner to return successful modules containing `instance_id`, the
  exact analyzed check instance, and its table specification. Existing callers may
  ignore the return. Return only modules whose tables were actually published to the
  QC database.
- Construct `QcPlotSubject` with the resolved table/input, exact analyzed check,
  database path, and immutable review-state snapshot.
- Emit:
  - single-page image plots at
    `deliverables/plots/<plot-id>/<dataset>/<image-stem>-<stable-hash>.png`, with the
    hash derived from the original dataset/image-stem pair;
  - multi-page image plots under an invocation directory with a manifest;
  - aggregate plots at `deliverables/plots/<plot-id>/<page>.png` plus manifest.
- Keep plot failures best-effort and log plot ID, page key, lifecycle, and failure
  phase. Never suppress measurement, analysis, QC, or manifest publication.
- Remove `--save-inspect` from CLI options, config/state, SLURM command generation,
  run-console controls, tests, and feature ledgers.
- Cover local CPU, measure-only, Stage 3, generated SLURM command/split behavior, and
  aggregate recompile. Recompile does not regenerate cache-dependent `PlotImage`
  output.
- Append a staged-SLURM finalizer job after the last Stage 3 chunk. It reloads the
  canonical pipeline and runs the same aggregate/finalize path as other strategies so
  mirror, named analysis, aggregate plots, QC, splits, and README are produced.

### Cluster 6: new multi-page plot

Implement:

```python
class PlotMeasTimeSeries(BaseModel, PlotMeas):
    page_by: ColumnRefList = Field(
        default_factory=lambda: ["MetadataGenetic_Strain"]
    )
    environment_by: ColumnRefList
    replicate_by: ColumnRefList
    time: ColumnRef = "MetadataCulture_Time"
    measurements: ColumnRefList = Field(default_factory=list)
    connect: bool = True
```

- `measurements=[]` means automatic selection, not no output.
- Select numeric public primary/derived measurements plus numeric custom measurement
  columns after excluding metadata, grouping, time, identity, QC, and known analysis
  columns. Preserve DataFrame column order.
- Require non-empty environment and replicate roles. Reject missing columns,
  duplicate/overlapping roles, nonnumeric explicit measurements, unsupported grouping
  values, and infinite numeric group values with actionable errors.
- Produce one Plotly page per strain, measurement rows, environmental columns, and
  replicate scatter traces. Sort within each replicate by time, connect only within a
  replicate when requested, and never aggregate.
- Preserve null groups as a labeled category and leave the caller's DataFrame and row
  order unchanged.
- Empty input returns `PlotOutput(pages=())`; non-empty input with no eligible
  measurement columns raises.

### Cluster 7: remove `Image.plot` safely

- Delete `Image.plot`, plot accessors, dynamic plotter registry, public plotter
  registration exports, instance caches, and development-only plotters without live
  non-accessor consumers.
- Migrate `DiagnosticsPlotter` and `DetectModesPlotter` to standalone
  `PlotDiagnostics` and `PlotDetectModes` Pydantic `PlotImage` models that accept an
  image at call time.
- Move `Image.napari()` from `ImagePlotHandler` into a visualization-only handler that
  remains in the linear `Image` MRO.
- Remove stale docs/notebooks or rewrite them to use standalone plots.

## 5. Verification and acceptance

Required tests:

- `PhtPlot` discovery, primary selection, controls, report composition, lazy imports,
  Pydantic schema purity, and absence of `dash()` / `dashboard()` aliases.
- `MeasureSymmetricZones` cache identity, controls, static export, serialization, and
  automatic local/measure-only CLI output without a flag.
- `ModelFitter` report kwargs, GUI control introspection, analyze-once behavior, and
  `LinearLagModel` same-instance plotting.
- `plots=None`, duplicate IDs, inline built-in and external class loading, local-class
  rejection, unresolved references, identity round-trip, and equal-but-distinct
  serialization.
- Staged supported-slot success, pre/GPU reference rejection, worker pipeline
  persistence, generated SLURM arguments, and aggregate-only recompile.
- Named analysis output, manifest checksums, registry-first resolution, missing-ID
  diagnostics, legacy fallback only without a manifest, and standalone deliverables.
- Plotly and Matplotlib saving/adaptation, cleanup on every path, unsupported types,
  unsafe labels, case-insensitive collisions, and partial page failure.
- QC analyzed-instance reuse and measurement-bound/analysis-bound subjects.
- `PlotMeasTimeSeries` deterministic pages/subplots/traces, nulls, mixed scalar group
  types, duplicate timepoints, automatic/explicit columns, custom numeric columns, no
  aggregation, empty input, validation failures, and input immutability.
- `PlotColonyMetricOverTime` required `on` metric, schema-backed grouping defaults,
  multi-column condition subplots, raw replicate traces, per-strain publication, and
  pipeline serialization overrides.
- `Image.plot` and plotter-registry absence while `Image.napari()`, `Image.dash()`, and
  every channel accessor `dash()` behavior, including objmap/objmask, remains unchanged.
- Image-consumer lifetime tests proving report helpers store no image/data state and
  operation convenience caches use weak references rather than retaining whole images.
  Compact cached measurements remain supported, while retained NumPy crops must own
  their buffers so they cannot pin a full image allocation through a view base chain.
- A subprocess import test proving `phenotypic.abc_.plotting` does not populate
  `sys.modules` with Plotly, Matplotlib, Dash, ipywidgets, or `phenotypic.plotting`.

Verification commands:

```text
uv run pytest <focused plotting/pipeline/CLI/GUI test paths>
uv run ruff check <explicit changed paths>
uv run mypy src/phenotypic
uv run pytest <affected integration suites>
```

Finish with an independent code-review subagent as required by repository policy.
Review for functional correctness, simplicity/duplication, staged execution, public
API consistency, and stale legacy references. Address all high-confidence findings,
rerun affected tests, and report the changed-file count, tests added, integration seams
touched, and any deliberately deferred work in the final handoff.
