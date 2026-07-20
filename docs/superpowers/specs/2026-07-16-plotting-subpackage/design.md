# Design: Pipeline Plotting Subpackage and Report Lifecycle

- **Date:** 2026-07-16
- **Status:** Implemented
- **Scope:** New plotting capability API under `phenotypic.abc_.plotting`, new
  `phenotypic.plotting` runtime package, pipeline plot configuration, analysis artifact
  identity, CLI and GUI plot refresh, plotting API hard cutover, and removal of
  `Image.plot`
- **Related existing design:** `DESIGN.md`, especially sections 06, 07, 09, and 15

The independent review identified and this revision resolves four material gaps: QC
recipes do not retain configured check instances, `PlotQc` lacked a complete runtime
subject, GUI Matplotlib cleanup was underspecified, and composite page keys were
ambiguous. Sections 7.2, 8.2, 8.5, and 11 contain the resulting contracts.

## 0. Approved implementation slice

This branch implements the backend, CLI, and configured-plot GUI lifecycle. The slice
is intentionally large enough to expose the integration cost that would be hidden by
a compatibility shim:

- implement the public plotting ABCs, runtime outputs, adapter, bindings, registry,
  coordinator, and `ImagePipeline.plots`;
- delete `FigureProvider`, plotting-object `dash()` / `dashboard()`, `Image.plot`, and
  `--save-inspect` rather than carrying transitional aliases;
- deeply convert `MeasureSymmetricZones` as the representative `PlotImage` producer;
- convert the `ModelFitter` seam to `PlotAnalysis` so a configured
  `LinearLagModel` can be reused by identity in `plots`;
- implement `PlotMeasTimeSeries` as the representative standalone, multi-page plot;
- implement named analysis artifacts, dynamic input resolution, and CLI-side QC plot
  subjects;
- update GUI consumers for event-driven measurement, analysis, and QC plot refresh;
- render multi-page GUI outputs through a deterministic labeled page selector.

The pilot covers local CPU, measure-only, staged Stage 3, generated SLURM command and
pipeline-split behavior, and aggregate recompile finalization. It does not submit a
live cluster job. Recompile regenerates aggregate plots only because its per-image
worker does not rerun measurers or reconstruct their diagnostic caches.

For staged GPU pipelines, the pilot supports `PlotImage` references to measurers and
inline final-image plots in Stage 3. Aggregate bindings remain with finalization.
Bindings to pre-GPU operations or the GPU detector fail before dispatch with an
actionable unsupported-reference error. This avoids duplicating stateful producers or
claiming that an object removed by pipeline splitting can preserve identity.

Post-GPU operation-bound `PlotImage` objects are supported because the exact operation
instances live in Stage 3. Staged SLURM appends a finalizer job after the last Stage 3
chunk; that job performs the same aggregate/finalize workflow used by local and regular
SLURM runs so named analyses, aggregate plots, QC, the mirror, and the README are not
silently omitted.

## 1. Motivation

PhenoTypic currently has several plotting mechanisms that overlap without sharing a
single lifecycle or serialization contract:

- `FigureProvider` turns `@figure` methods into `inspect()` and `dash()` surfaces
  (`src/phenotypic/abc_/_figure_provider.py:288`, `:395`, and `:433`).
- selected `MeasureFeatures` implementations hand-write `inspect()` methods and rely
  on per-image caches (`src/phenotypic/measure/_measure_symmetric_zones.py:495` and
  `src/phenotypic/measure/_measure_orientation_zones.py:1469`);
- `ModelFitter` and three QC implementations expose `dash()` directly
  (`src/phenotypic/analysis/abc_/_model_fitter.py:603` and
  `src/phenotypic/analysis/qc/_*.py`);
- some image operations expose `dashboard()` wrappers;
- `Image.plot` dynamically dispatches to seven registered plotter classes through a
  separate registry (`src/phenotypic/_core/_image_parts/accessors/_plot_accessor.py:44`
  and `src/phenotypic/sdk_/register/_plotter_registry.py:25`);
- CLI analysis output is fixed to `analysis.csv` and `analysis.parquet`, so consumers
  cannot identify an analysis robustly when more than one analysis artifact exists
  (`src/phenotypic/sdk_/_io_constants.py:1156` and `:1161`).

The new plotting system must let a configured operation or analyzer also be listed as
a plot without wrapping it in a second class or duplicating its settings. For example,
the same `LinearLagModel` instance used by the pipeline model slot must be usable in
`plots`, serialize once, deserialize once, and retain the analysis state populated by
`analyze()`. The same rule applies to image-bound measurers such as
`MeasureSymmetricZones`.

This design consolidates renderer behavior into
`phenotypic.abc_.plotting.PhtPlot`, adds action-named lifecycle mixins beside it,
introduces identity-preserving plot bindings, gives analyses stable runtime identities,
and removes the separate `Image.plot` accessor.

The existing CLI writer already has separate Matplotlib and Plotly branches
(`src/phenotypic/_cli/_cli_output_manager.py:1470-1557`), while the GUI independently
implements the same split (`src/phenotypic/gui/analysis/_render.py:37-101`). Moving that
dispatch into one adapter is therefore consolidation of established behavior, not a new
renderer dependency.

## 2. Goals

1. Add a public `phenotypic.abc_.plotting` subpackage for plotting capabilities and a
   public `phenotypic.plotting` subpackage for bindings, runtime output, adapters, and
   concrete plot classes.
2. Add a methods-only `phenotypic.abc_.plotting.PhtPlot` base mixin that owns the
   current `FigureProvider`
   behavior and introduces:
   - `inspect()` for the primary saveable backend figure or multi-page output;
   - `report()` for the complete composed or interactive report.
3. Add action-named lifecycle subclasses:
   - `PlotImage`;
   - `PlotMeas`;
   - `PlotAnalysis`;
   - `PlotQc`.
4. Add `plots` to `ImagePipeline`, accepting existing configured objects directly.
5. Preserve object identity when a plot points to an operation, measurer, analyzer, or
   model already configured elsewhere in the pipeline. For recipe-backed QC, reuse the
   same runtime check instance for analysis and plotting rather than constructing a
   second check.
6. Add dynamic table inputs for `PlotAnalysis` and `PlotQc` without adding an input
   setting to `PlotMeas`.
7. Replace fixed `analysis.*` output names with analysis-identified artifacts such as
   `LinearLagModel.csv` and `LinearLagModel.parquet`.
8. Make consumers resolve analysis tables through an analysis registry and manifest,
   never by constructing a filename.
9. Emit configured plot inspections under `deliverables/plots/` in all supported CLI
   execution paths.
10. Refresh `PlotMeas`, `PlotAnalysis`, and `PlotQc` after the corresponding data changes
    in the GUI.
11. Hard-cut the plotting-object methods `dash()` and `dashboard()` to `report()` while
    preserving the separate `Image.dash()` and image-channel `dash()` methods.
12. Remove `Image.plot`, its dynamic accessor, and its plotter registry.
13. Support plot classes that return multiple deterministically named figure pages.
14. Let CLI and GUI consumers save or render either Matplotlib `Figure` objects or
    Plotly `go.Figure` objects through one backend-neutral adapter.

## 3. Non-goals

- Do not rename or remove `Image.dash()`, `image.rgb.dash()`, `image.gray.dash()`, or
  `image.detect_mat.dash()`. Also preserve `image.objmap.dash()` and
  `image.objmask.dash()`, which share the single-channel accessor implementation.
- Do not rename the durable `dashboard.html` deliverable solely because the Python
  plotting method becomes `report()`.
- Do not add an external plotting framework. Plotly, Kaleido, Matplotlib, Dash, and
  ipywidgets already cover the required rendering surfaces.
- Do not add numerical algorithms or measurement columns.
- Do not make every `ImageOperation` plot-capable. Plotting remains explicit and
  opt-in.
- Do not infer analysis inputs by scanning filenames or matching DataFrame columns.
- Do not treat arbitrary Dash components as figures. The backend-neutral figure input
  contract covers `matplotlib.figure.Figure` and `plotly.graph_objects.Figure`; Dash
  components are outputs created by the GUI adapter.

Dash itself does not define a distinct figure object. `dcc.Graph.figure` consumes a
Plotly figure or figure-shaped mapping. In this design, “Dash figure” therefore means a
Plotly `go.Figure`; accepting arbitrary Dash layout components would mix figure content
with GUI layout and is deliberately excluded.

## 4. Terminology and public names

| Name | Meaning |
|---|---|
| `PhtPlot` | Renderer-neutral plotting capability and `@figure` composition base in `phenotypic.abc_.plotting` |
| `PlotImage` | ABC mixin refreshed after a particular image completes its relevant pipeline stage |
| `PlotMeas` | ABC mixin refreshed from the current aggregate post-applied measurements mirror |
| `PlotAnalysis` | ABC mixin refreshed after a selected analysis table is updated |
| `PlotQc` | ABC mixin refreshed after measurements, analysis, or QC state used by the plot is updated |
| `PlotBinding` | Serialized wiring from a plot-capable object to a lifecycle and optional input |
| `AnalysisInput` | Stable reference to a named analysis table |
| `MeasurementInput` | Explicit reference to the current measurements mirror |
| `AnalysisRegistry` | Runtime mapping from analysis ID to current table, producer, and artifacts |
| `QcPlotSubject` | Runtime payload containing a resolved input table plus current QC state |
| `PlotOutput` | Runtime-only ordered collection of zero or more named figure pages |
| `PlotPage` | One deterministically named figure plus its display label and grouping metadata |

The action-first class names are intentional. They describe what the mixin makes the
class do, while the `Plot*` prefix groups them in API completion.

All five mixin classes, plus `Control`, `FigureSpec`, `figure`, and `BoundFigures`, are
public only from `phenotypic.abc_.plotting`. `phenotypic.plotting` does not re-export
them. This keeps the user-facing capability contracts beside the other operation ABCs
and prevents two competing canonical import paths.

Namespace placement does not turn these classes into abstract base classes with
required abstract methods. They remain fieldless cooperative mixins. The `abc_`
location is an API and documentation taxonomy decision: users discover operation and
plotting capabilities together.

`PlotQc` follows the repository's `QcRecipe` and `QcTableSpec` casing.

## 5. `PhtPlot` contract

### 5.1 Responsibilities absorbed from `FigureProvider`

`phenotypic.abc_.plotting.PhtPlot` owns the complete current `FigureProvider`
mechanism:

- `Control` validation;
- `FigureSpec` metadata;
- the `@figure` decorator;
- lazy application of the PhenoTypic Plotly theme;
- MRO-aware figure discovery and definition ordering;
- primary-figure selection;
- held-subject and call-time-subject binding;
- transient `BoundFigures` caching;
- control-free figure composition;
- ipywidgets report construction when controls are present.

`FigureProvider` is removed as a public and internal class. This is a hard cutover, not
a compatibility alias. Existing classes migrate to either `PhtPlot` or the appropriate
action-named subclass.

### 5.2 Methods-only MRO shape

`PhtPlot` and its four lifecycle subclasses have no Pydantic fields, no `__init__`, and
no persistent instance state. That preserves the current fieldless mixin safety of
`FigureProvider` and avoids a second `BaseModel` root in `BaseOperation` and
`SetAnalyzer` hierarchies.

Standalone serializable plots combine a Pydantic root with a lifecycle mixin:

```python
class PlotDiagnostics(BaseModel, PlotImage):
    model_config = ConfigDict(extra="forbid")
```

Existing Pydantic classes add only a capability mixin:

```python
class MeasureSymmetricZones(MeasureFeatures, PlotImage):
    ...


class ModelFitter(SetAnalyzer, PlotAnalysis, ABC):
    ...
```

### 5.3 Public rendering methods

```python
class PhtPlot:
    def inspect(
        self,
        subject: Any = None,
        *,
        for_save: bool = False,
        **overrides: Any,
    ) -> Any:
        """Return one primary figure or an ordered set of saveable pages."""

    def report(self, subject: Any = None, **overrides: Any) -> Any:
        """Return the complete composed or interactive report."""
```

The ABC mixin deliberately uses `Any` for these runtime return annotations. It cannot
import `FigureLike` or `PlotOutput` from the higher-level `phenotypic.plotting` package
without reversing the dependency boundary. Concrete runtime plot classes narrow their
overrides to `FigureLike` or `PlotOutput`, and consumers perform explicit normalization.

`ModelFitter` is a special producer-reuse seam. Its existing Plotly implementation is
extracted into a private builder. `inspect(*, for_save=False, tmax=..., criteria=...,
figsize=..., cmap=..., legend=...)` and `report(**plot_kwargs)` both delegate to that
builder. All plotting parameters are keyword-only so figure discovery cannot mistake
`tmax` for a subject. After `analyze()`, the coordinator calls a reused model without a
DataFrame subject so it reads the exact populated private state. A standalone
`PlotAnalysis` that consumes another producer receives the dynamically resolved table
as its subject.

`inspect()` retains the current primary-figure rules for a single page. A class that
produces multiple independently saveable pages overrides `inspect()` and returns a
`PlotOutput`. It does not return a bare list or dict.

`report()` replaces both plotting-object meanings currently spelled `dash()` or
`dashboard()`. It returns a composed Plotly figure when figures are control-free and an
ipywidgets object in notebook contexts when controls are declared. GUI adapters consume
`BoundFigures` directly when they need native Dash controls.

There is no `PhtPlot.dash()` or `PhtPlot.dashboard()` compatibility method.

### 5.4 Backend-neutral figures and multi-page output

There are two independent composition axes:

1. **Panels** are subplots or `@figure` sections inside one figure page.
2. **Pages** are separate named figures that become separate files or selectable GUI
   views.

The runtime-only output types are frozen dataclasses. They are never serialized into
pipeline JSON because they contain live backend figure objects.

```python
from dataclasses import dataclass, field
from typing import Mapping, TypeAlias

from matplotlib.figure import Figure as MplFigure
from plotly.graph_objects import Figure as PlotlyFigure

FigureLike: TypeAlias = MplFigure | PlotlyFigure


@dataclass(frozen=True)
class PlotPage:
    key: str
    figure: FigureLike
    label: str | None = None
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class PlotOutput:
    pages: tuple[PlotPage, ...]
```

The production implementation keeps backend imports lazy rather than importing both
libraries at `phenotypic.abc_.plotting` or `phenotypic.plotting` import time. The
type sketch above shows the public contract, not the required import strategy.

Rules:

- page order is deterministic;
- `key` is a logical key, not a caller-controlled relative path;
- keys must be non-empty and unique within one output;
- the output writer sanitizes `label` when present, otherwise `key`, into a safe
  filename stem;
- sanitization collisions receive a stable short hash derived from `key` rather than
  overwriting a page;
- plot IDs, dataset names, and image stems pass through the same safe path-component
  helper before directory construction;
- `label` preserves the unsanitized human-readable group label;
- `metadata` records grouping values for manifests and GUI selectors;
- an empty `PlotOutput` is a valid no-data result and writes no figure files;
- a raw `FigureLike` result is normalized to a one-page `PlotOutput` by the consumer.

`FigureAdapter` is the one backend switch used by CLI and GUI code:

```python
class FigureAdapter:
    @staticmethod
    def save_png(figure: FigureLike, path: Path, *, title_prefix: str = "") -> None:
        ...

    @staticmethod
    def to_dash_component(figure: FigureLike) -> Any:
        ...

    @staticmethod
    def close(figure: FigureLike) -> None:
        ...
```

- Matplotlib saving uses `Figure.savefig(..., format="png")` and closes the figure
  after successful or failed publication.
- Plotly saving uses `Figure.write_image(..., format="png")` through Kaleido.
- Plotly GUI rendering returns `dcc.Graph`.
- Matplotlib GUI rendering rasterizes to in-memory PNG bytes and returns `html.Img`.
- unsupported types fail with a message naming the plot ID, page key, and actual type.

This adapter replaces the duplicated backend dispatch currently present in
`OutputManager._build_inspect_writer` and `gui.analysis._render.render_plot`.

The output module also provides a reusable `canonical_group_key(pairs)` helper for
grouped multi-page plots. It implements the typed tuple encoding defined in section
8.5. Plot classes own grouping semantics, but they do not invent their own ambiguous
key concatenation or filename escaping.

### 5.5 Hard-cut boundary

The following calls are removed:

```python
operation.dash(...)
operation.dashboard(...)
model.dash(...)
qc.dash(...)
figure_provider.dash(...)
```

The following calls remain unchanged:

```python
image.dash(...)
image.rgb.dash(...)
image.gray.dash(...)
image.detect_mat.dash(...)
```

Implementation and review must not use a global text replacement for `.dash()`.

## 6. Lifecycle subclasses

The subclasses identify refresh timing. They do not own table paths and do not add
Pydantic fields.

| Class | Trigger | Default resolved subject |
|---|---|---|
| `PlotImage` | one image finalized | the same `Image` instance used by the operation or measurer |
| `PlotMeas` | measurement mirror updated | current post-applied measurements DataFrame |
| `PlotAnalysis` | selected analysis updated | input selected by the binding |
| `PlotQc` | selected input or QC state updated | a `QcPlotSubject` containing the selected input and current QC context |

The class's pipeline role does not determine its plot lifecycle. For example,
`MeasureSymmetricZones` is a measurer, but its `inspect(image=...)` consumes image state,
so it implements `PlotImage`, not `PlotMeas`.

## 7. Pipeline configuration and object identity

### 7.1 `ImagePipeline.plots`

```python
plots: list[PhtPlot | PlotBinding] = Field(default_factory=list)
```

Construction accepts `None` for compatibility with the requested ergonomic API and
normalizes it to an empty list. The stored form is an ordered list of normalized
`PlotBinding` objects.

Example:

```python
lag = LinearLagModel(
    on="Size_Area",
    groupby=["MetadataGenetic_Strain"],
)
zones = MeasureSymmetricZones()

pipeline = ImagePipeline(
    meas={"zones": zones},
    model=lag,
    plots=[zones, lag],
)
```

### 7.2 Reference normalization

During validation, the pipeline builds an object registry for:

- `ops.<key>`;
- `meas.<key>`;
- `post.<key>`;
- `filters.<key>`;
- `model`;
- `qc.<instance_id>` recipe entries.

If a raw plot object is identical, using `is`, to an object already in this registry,
the binding stores a pipeline reference. Equality or equal Pydantic dumps never imply
shared identity.

The `qc` slot is deliberately different. `ImagePipeline.qc` stores `QcRecipeEntry`
configuration records, and the QC runner lazily constructs checks. A
`qc.<instance_id>` plot reference therefore resolves to the recipe entry, not to a
persistent `QualityCheck` object. During each QC run, the runner retains the check it
instantiated, calls `analyze()` on it, publishes QC artifacts, and gives that same live
instance to the plot coordinator. The runner does not instantiate the check a second
time for plotting. QC classes that do not implement `PlotQc` cannot be used as QC plot
references.

Because no persistent QC object exists to place in two lists, recipe-backed QC uses an
explicit reference:

```python
PlotBinding(
    id=qc_entry.instance_id,
    ref=PipelineObjectRef(slot="qc", key=qc_entry.instance_id),
    input=AnalysisInput(analysis_id="LinearLagModel"),
)
```

This is the one exception to the raw-object `plots=[obj]` shorthand. Standalone
`PlotQc` objects that are not recipe entries may still be serialized inline.

If a plot object is not configured elsewhere, it is serialized inline using its class
and `model_dump(mode="json")` settings. The inline shape stores `module` and
`qualname`, not only a short class name. Deserialization imports the module, walks the
qualified name, and validates the resolved class before `model_validate`. Local
classes containing `<locals>` are rejected because they cannot be re-imported.
Externally defined plots must live in an importable module; staged workers may use the
existing `PHENOTYPIC_PRELOAD_MODULES` hook before pipeline loading.

Canonical JSON:

```json
{
  "model": {
    "class": "LinearLagModel",
    "params": {
      "on": "Size_Area",
      "groupby": ["MetadataGenetic_Strain"]
    }
  },
  "plots": [
    {
      "id": "zones",
      "ref": {"slot": "meas", "key": "zones"}
    },
    {
      "id": "LinearLagModel",
      "ref": {"slot": "model"},
      "input": {"kind": "measurements"}
    }
  ]
}
```

Deserialization reconstructs normal pipeline objects first, then resolves plot
references to those exact reconstructed instances. Private cached analyzer state is not
serialized; it is repopulated by normal measurement and analysis execution.

Nested pipelines with non-empty `plots` are rejected when used as operations in this
pilot. Plot dispatch requires the top-level CLI output context, and silently preserving
but not executing nested plots would be misleading. Top-level bindings cannot reference
objects inside a nested pipeline. A future recursive design must define namespacing and
output ownership before lifting this restriction.

### 7.3 Plot IDs

Every binding has a stable `id` used for output directories, GUI component identity,
and dependency tracking.

Defaults:

- referenced dict slot: its configured key;
- referenced single model: its concrete class name;
- referenced QC entry: its stable `instance_id`;
- inline plot: its concrete class name.

Duplicate IDs in one pipeline are validation errors. Callers may set an explicit ID to
disambiguate repeated plot classes.

## 8. Dynamic inputs

### 8.1 Allowed input types

Only `PlotAnalysis` and `PlotQc` bindings expose `input`:

```python
PlotInput = MeasurementInput | AnalysisInput


class MeasurementInput(BaseModel):
    kind: Literal["measurements"] = "measurements"


class AnalysisInput(BaseModel):
    kind: Literal["analysis"] = "analysis"
    analysis_id: str
```

The default for `PlotAnalysis` and `PlotQc` is `MeasurementInput()`.

`PlotImage` and `PlotMeas` reject a supplied `input` during pipeline validation.
`PlotMeas` always consumes the current post-applied measurement mirror, so storing an
input there would create two ways to express the same contract.

The input belongs to `PlotBinding`, not to the lifecycle mixin or reused producer.
Adding a Pydantic field to `PlotAnalysis` or `PlotQc` would change every mixed-in
class's schema and would prevent one plot-capable producer from being bound more than
once with different inputs. The explicit interface is:

```python
pipeline = ImagePipeline(
    model=lag,
    plots=[
        lag,  # defaults to MeasurementInput()
        PlotBinding(
            id="growth-qc",
            plot=growth_qc,
            input=AnalysisInput(analysis_id="LinearLagModel"),
        ),
    ],
)
```

This preserves the requested `plots=[lag]` shorthand while keeping input selection
dynamic and binding-specific.

### 8.2 QC invocation payload

`PlotQc.inspect()` always receives one explicit runtime subject:

```python
@dataclass(frozen=True)
class QcPlotSubject:
    input_table: pd.DataFrame
    input_ref: MeasurementInput | AnalysisInput
    qc_instance_id: str | None
    analyzed_check: QualityCheck | None
    qc_database: Path | None
    review_state: Mapping[str, Any]
```

The coordinator resolves `input_table` afresh from `input_ref` for each call.
Recipe-backed QC plots receive their stable entry ID and the exact analyzed check
instance retained by the QC runner. Standalone aggregate QC plots use `None` for those
two fields. CLI contexts may provide an empty read-only `review_state`; GUI contexts
provide the current review-state snapshot. The payload is runtime-only and never enters
pipeline JSON.

The QC runner changes from a write-only `None` result to a result that retains runtime
producer identity:

```python
@dataclass(frozen=True)
class QcRunModule:
    instance_id: str
    check: QualityCheck
    spec: QcTableSpec


@dataclass(frozen=True)
class QcRunResult:
    database: Path | None
    modules: tuple[QcRunModule, ...]
```

`run_qc()` publishes the database atomically, then returns the successful modules.
Existing callers may ignore the return. A failed or disabled module has no result
entry. A module is returned only after its tables were successfully published into the
QC database; a check that analyzed successfully but failed database ingestion is not a
plotting subject. Plot failures occur after QC publication and cannot invalidate the
database.
The GUI passes an immutable copy of review state so callbacks cannot mutate canonical
curation state through a plot payload.

### 8.3 Dynamic analysis resolution

An `AnalysisInput` stores only an `analysis_id`. It never stores or derives a file path.

Each refresh resolves in this order:

1. current in-memory `AnalysisRegistry` table;
2. current persisted analysis manifest entry;
3. manifest-selected Parquet artifact;
4. fail with an error listing available analysis IDs.

CSV is a human-readable mirror and is not the preferred programmatic input.

Resolution is performed on every refresh rather than cached on the binding. A GUI
analysis rerun can therefore replace the registry table and notify dependent plots
without rewriting pipeline configuration.

### 8.4 Producer reuse

When a `PlotAnalysis` object is the same object as an analysis producer, such as the
pipeline's `LinearLagModel`, the coordinator does not construct or fit a second model.
The normal analysis stage calls `analyze()` on that object, registers its returned table,
and only then invokes `inspect()` or `report()` on the already-populated producer.

A standalone `PlotAnalysis` consuming another analysis receives the dynamically
resolved input table through the binding adapter.

### 8.5 Reusable multi-page measurement time series

The first required multi-page plot is `PlotMeasTimeSeries(PlotMeas)`. It consumes the
measurement mirror automatically and therefore has no `input` field.

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

`measurements=[]` is the explicit automatic-selection sentinel. Automatic selection
preserves DataFrame column order and includes numeric public primary/derived
measurements plus numeric custom measurement columns after excluding metadata,
grouping, time, object identity, QC, and known analysis columns. An explicit non-empty
list limits or orders rows. Requesting no output is expressed by omitting the plot from
the pipeline rather than by an empty explicit measurement selection.

`PlotColonyMetricOverTime` is the ready-to-use, single-metric specialization of this
grouping model. Its required `on: ColumnRef` parameter selects any numeric measurement
column without exposing the generic class's multi-row `measurements` list. It defaults
to one page per `MetadataGenetic_Strain`, one environmental subplot per
`MetadataCondition_Media`, one trace per `MetadataSample_BioReplicate`, and time on
`MetadataCulture_Time`. All grouping fields remain configurable, including
multi-column environmental groups. For a radius-growth report, use a public column
such as `on="Shape_MeanRadius"`; there is no public `Shape_Radius` column.

The first implementation builds Plotly figures only. Backend neutrality belongs to
`FigureAdapter`, which accepts both Plotly and Matplotlib figures from any producer;
duplicating this plot's subplot algorithm across two backends is out of scope.

For each unique `page_by` combination, the class emits one `PlotPage`:

- page: one strain by default;
- subplot rows: measurement columns, each with its own y-axis;
- subplot columns: unique environmental-group combinations;
- traces within each subplot: individual replicate combinations;
- x-axis: time;
- marks: raw scatter time-series points, connected within each replicate in sorted time
  order when `connect=True`;
- aggregation: none by default, so biological and technical replicate disagreement
  remains visible.

Separating measurements into rows is required because unrelated measurements may have
different units and scales. They must not be overlaid on one y-axis.

Example output:

```text
deliverables/plots/PlotMeasTimeSeries/
|-- BY4741.png
|-- RM11-1a.png
|-- manifest.json
`-- ...
```

The default plot binding ID is the class name, which produces the requested class-named
subfolder. An explicit binding ID replaces the folder name when two configurations of
the same class are present.

Page construction is deterministic:

1. normalize grouping values, preserving null as a labeled category;
2. construct canonical typed tuples and sort page and environmental groups by those
   tuples, using labels only for display;
3. sort each replicate trace by time;
4. encode the page tuple as a canonical logical key;
5. sanitize the human label at the output boundary;
6. append a stable short hash if two logical keys sanitize to the same filename.

Canonical grouping identity is separate from labels and paths. For columns `c1..cn`,
the logical key is compact JSON encoding of this ordered structure:

```text
[[column_name, type_tag, canonical_value], ...]
```

Normalization unwraps NumPy and pandas scalar wrappers. Supported tags are `null`,
`bool`, `int`, `float`, `str`, `datetime`, `date`, and `timedelta_ns`. Integers use
decimal strings, finite floats use `float.hex()`, temporal values use ISO 8601 or
integer nanoseconds, and null-like values use the single `null` representation.
Unsupported grouping types and infinite floats fail validation with the column name.
Compact JSON uses UTF-8, preserves column order, and has no insignificant whitespace.
Consequently, `("a/b", "c")` differs from `("a", "b/c")`, integer `1` differs from
string `"1"`, and null does not collide with a literal missing-value label.

For one string `page_by` value, the human label and filename remain readable, such as
`BY4741` and `BY4741.png`. For multiple values, the label includes column names and
values. If two labels sanitize to the same filename, the suffix hash is derived from
the canonical logical key, not from the label.

Each multi-page plot directory contains its own manifest:

```json
{
  "schema_version": 1,
  "plot_id": "PlotMeasTimeSeries",
  "class": "PlotMeasTimeSeries",
  "pages": [
    {
      "key": "[[\"MetadataGenetic_Strain\",\"str\",\"BY4741\"]]",
      "label": "BY4741",
      "file": "BY4741.png",
      "backend": "plotly",
      "metadata": {"MetadataGenetic_Strain": "BY4741"}
    }
  ]
}
```

The manifest is authoritative for page discovery. GUI and other consumers do not glob
the folder and do not reconstruct page filenames from strain values.

The subplot matrix can become large when every numeric measurement is selected. The
plot class sizes the figure from row and column counts and exposes the explicit
`measurements` selector for bounded reports. Automatic measurement pagination is not
part of the first implementation because it would change the requested one-file-per-
strain contract. The generic `PlotOutput` type can support that later without changing
consumers.

## 9. Analysis artifact identity

### 9.1 Output names

The fixed files `deliverables/analysis.csv` and
`deliverables/analysis.parquet` are removed.

The current single model uses its concrete class name as its default analysis ID and
artifact stem:

```text
deliverables/LinearLagModel.csv
deliverables/LinearLagModel.parquet
```

Future support for multiple analysis producers must require unique configured analysis
IDs. Class-name-only output is not sufficient when two instances share a class.

Analysis IDs are validated before becoming artifact stems: 1-128 ASCII characters,
starting with a letter and followed only by letters, digits, `_`, `-`, or `.`. Path
separators, `.` / `..`, whitespace, and empty IDs are rejected rather than sanitized.

### 9.2 Analysis manifest

The CLI and GUI atomically maintain:

```text
deliverables/analysis_manifest.json
```

Schema:

```json
{
  "schema_version": 1,
  "analyses": {
    "LinearLagModel": {
      "class": "LinearLagModel",
      "csv": "LinearLagModel.csv",
      "parquet": "LinearLagModel.parquet",
      "rows": 96,
      "columns": ["MetadataGenetic_Strain", "Model_Lag"]
    }
  }
}
```

Paths are relative to the deliverables base so standalone deliverables bundles remain
portable. Artifact and manifest paths are resolved only through new helpers in
`phenotypic.sdk_._io_constants`.

The manifest is written after the Parquet and CSV artifacts have both been published.
Each file is atomically replaced and the manifest is published last. Manifest entries
include SHA-256 checksums for both artifacts. Before either canonical artifact moves,
the writer durably records a `.analysis-publication.json` recovery journal containing
the prior-file existence state, transaction token, and intended manifest entry. A
reader or later writer acquires the shared generation lock and resolves any journal
before reading or starting another publication. If the intended manifest entry is
visible, recovery completes the committed generation; otherwise it restores the
previous canonical pair. This makes process loss between the two file replacements
recoverable while retaining the requested class-named CSV and Parquet files.
Publication failure is logged and does not invalidate measurement outputs.

## 10. CLI lifecycle and outputs

### 10.1 Output tree

Configured plots opt in to primary inspection output without a separate CLI flag:

```text
deliverables/
|-- LinearLagModel.csv
|-- LinearLagModel.parquet
|-- analysis_manifest.json
`-- plots/
    |-- zones/
    |   `-- <dataset>/
    |       `-- <image-stem>.png
    |-- PlotMeasTimeSeries/
    |   |-- BY4741.png
    |   |-- RM11-1a.png
    |   `-- manifest.json
    |-- measurements-summary/
    |   |-- default.png
    |   `-- manifest.json
    |-- LinearLagModel/
    |   |-- default.png
    |   `-- manifest.json
    `-- growth-qc/
        |-- default.png
        `-- manifest.json
```

`PlotImage` outputs are per image. `PlotMeas`, `PlotAnalysis`, and `PlotQc` outputs are
aggregate artifacts under a directory keyed by plot ID. Every aggregate directory has
a page manifest even when the output contains one `default` page. This gives consumers
one discovery contract for single-page and multi-page plots.

A multi-page `PlotImage` uses an invocation directory instead of a flat file:

```text
deliverables/plots/<plot-id>/<dataset>/<sanitized-image-stem>-<stable-hash>/<page>.png
deliverables/plots/<plot-id>/<dataset>/<sanitized-image-stem>-<stable-hash>/manifest.json
```

A single-page `PlotImage` retains a compact flat file at
`deliverables/plots/<plot-id>/<dataset>/<sanitized-image-stem>-<stable-hash>.png` to
avoid one manifest per image in large runs. The hash derives from the original
dataset/image-stem pair so lossy sanitization and case-folding cannot overwrite a
different image. This is an output optimization only; the coordinator still normalizes
both cases to `PlotOutput` internally.

The CLI calls `inspect(for_save=True)` for durable plot output, normalizes a raw figure
or `PlotOutput`, then saves every page through `FigureAdapter`. `report()` is the full
interactive surface for notebooks and GUI use; the CLI does not attempt to serialize
an ipywidgets object.

### 10.2 Ordering

```text
per-image apply and measure
    -> PlotImage inspect on the same Image instance
aggregate clean master
    -> apply post and metadata to measurements mirror
    -> PlotMeas inspect
run analyses
    -> publish named artifacts and update AnalysisRegistry
    -> PlotAnalysis inspect
run QC
    -> publish QC state
    -> PlotQc inspect
```

The same ordering applies to local CPU, measure-only, staged GPU Stage 3, recompile, and
SLURM finalization paths. Stage 3 must invoke per-image plot dispatch after measurement
has populated operation caches and before the image object is discarded.

### 10.3 Existing `--save-inspect`

The `--save-inspect` flag and `OutputManager.save_inspects` state are removed. The
pipeline's `plots` list is the explicit opt-in configuration. The existing duck-typed
loop over every measurer with `hasattr(measurer, "inspect")` is replaced by lifecycle
dispatch over normalized plot bindings.

## 11. GUI lifecycle

The GUI uses the same coordinator and adapter as CLI finalization. Measurement mirror
curation, successful named-analysis writes, and successful QC rebuilds refresh their
configured plot lifecycles. Multi-page results use a labeled Dash tab selector; a
single-page result keeps the direct Graph or Img surface.

A shared plotting coordinator is used by GUI callbacks rather than having each tab
guess which rendering method or artifact to call.

Events:

| Event | Updated plots |
|---|---|
| measurement mirror loaded or curated | `PlotMeas`; measurement-bound standalone `PlotQc`; then analysis and QC reruns as configured |
| analysis table replaced | dependent `PlotAnalysis` and `PlotQc` |
| QC database rebuilt or review state changed | dependent `PlotQc` |
| selected image changed and image data is available | relevant `PlotImage` |

A recipe-backed `PlotQc` refresh waits for the QC rebuild so its `QcPlotSubject`
contains the matching analyzed check and database. A standalone measurement-bound
`PlotQc` that does not depend on recipe state may refresh directly on the measurement
event. Dependency matching uses `MeasurementInput` or the exact `analysis_id`; it does
not redraw unrelated plots.

The analysis GUI calls `report()`, not `dash()`, for Plotly renderers. Matplotlib-only
analyzers that do not implement a plotting mixin continue to use the existing `show()`
fallback.

All plotting GUI consumers normalize `FigureLike | PlotOutput` through the shared
backend adapter:

- a one-page Plotly output becomes one `dcc.Graph`;
- a one-page Matplotlib output becomes one rasterized `html.Img`;
- multi-page output becomes a deterministic page selector with one adapted page visible
  at a time;
- labels and selector values come from the plot manifest or live `PlotPage` metadata,
  never from sanitized filenames;
- every Matplotlib figure is closed after rasterization and on normalization, selection,
  or adaptation failure so repeated GUI refreshes do not accumulate live figures.

GUI changes must update `src/phenotypic/gui/FEATURES.md`, and any changed end-to-end
workflow must update `WORKFLOWS.md` and tutorial screenshots as required by the GUI
module guide.

## 12. Removal of `Image.plot`

### 12.1 Removed surface

The following are deleted with no compatibility property:

- `Image.plot` and `_image_plot_handler.py` wiring;
- `PlotAccessor`;
- `DashPlotAccessor`;
- the plotter portion of `phenotypic.sdk_.register`;
- `register_plotter`, `get_plotter`, and `available_plotters` public exports;
- dynamic plotter discovery and instance caching;
- documentation and tests for `image.plot.*` and `image.plot.dash.*`.

Operation class registration in `phenotypic.sdk_.register` remains. Only the plotter
registry is removed.

`Image.napari()` currently shares the `ImagePlotHandler` that owns `Image.plot`.
Before deleting that handler, move `napari()` into a visualization-only image handler
that remains in the linear `Image` MRO. Regression tests must prove `Image.plot` is
absent while `Image.napari()` and protected `Image.dash()` surfaces still work.

### 12.2 Retained diagnostics

Two existing Plotly-capable diagnostics are migrated because they exercise the new
`PlotImage` contract and remain useful in pipeline deliverables:

| Existing class | New class | New use |
|---|---|---|
| `DiagnosticsPlotter` | `PlotDiagnostics` | `PlotDiagnostics().report(image)` or pipeline `plots` |
| `DetectModesPlotter` | `PlotDetectModes` | `PlotDetectModes().report(image)` or pipeline `plots` |

They become standalone Pydantic plot models whose `@figure` methods take an `Image`
subject at call time rather than receiving an image through `BasePlotter.__init__`.

The remaining Matplotlib-only development plotters are removed unless the mandatory
blast-radius audit finds a current non-accessor production consumer:

- `AllDataPlotter`;
- `MorphologyPlotter`;
- `SizeDistributionPlotter`;
- `SpatialPlotter`;
- `ThresholdPlotter`.

`Image.show(...)` and `Image.dash(...)` remain the supported direct image display APIs.

## 13. Error handling

- Duplicate plot IDs fail pipeline validation.
- A referenced pipeline slot that cannot be resolved fails deserialization with the
  plot ID and missing slot in the message.
- `input` on `PlotImage` or `PlotMeas` fails validation.
- Missing analysis IDs fail resolution and list currently available IDs.
- A plotting failure is isolated from measurement, analysis, and QC artifact
  publication. The CLI logs the plot ID and continues.
- Static image export failures identify whether the failure came from inspection,
  Kaleido, or filesystem publication.
- Plot outputs use atomic replacement so readers never observe partial files.
- One failed page does not suppress successfully rendered sibling pages. The per-plot
  manifest lists only successfully published pages and records page failures in logs.
- Figure objects are runtime-only and never enter pipeline JSON or analysis manifests.

## 14. Compatibility and migration policy

This feature intentionally makes breaking API changes:

- `FigureProvider` -> `PhtPlot` or a lifecycle subclass;
- plotting-object `.dash()` / `.dashboard()` -> `.report()`;
- `Image.plot.*` removed;
- `analysis.*` -> analysis-identified output names;
- `--save-inspect` removed.

There are no runtime aliases or deprecation warnings. Documentation, tests, examples,
GUI callbacks, and internal call sites move in the same release.

Readers of older output bundles may retain a read-only compatibility lookup for legacy
`analysis.parquet` and `analysis.csv`. Fresh writes never produce legacy names. This
reader compatibility is independent of the Python API hard cut.

The legacy constants and path helpers become private implementation details of that
fallback. They are removed from `phenotypic.sdk_`, GUI config, and other public exports;
programmatic consumers resolve the manifest instead.

## 15. Testing requirements

### Core plotting

- `PhtPlot`, lifecycle mixins, and figure-composition helpers import only from
  `phenotypic.abc_.plotting`; `phenotypic.plotting` does not re-export them.
- importing `phenotypic.abc_.plotting` does not import
  `phenotypic.plotting`, Plotly, Matplotlib, Dash, or ipywidgets.
- `@figure` validation and inherited ordering survive the move.
- primary selection and explicit `inspect()` overrides behave as before.
- `report()` composes control-free figures and delegates controlled figures to the
  notebook adapter.
- Pydantic schemas and dumps are unchanged by adding a lifecycle mixin.
- raw Matplotlib and Plotly figures normalize to one-page outputs.
- multi-page outputs preserve deterministic order, reject duplicate keys, sanitize
  unsafe filenames, and disambiguate sanitization collisions.
- the shared adapter atomically saves both backends and produces valid Dash components
  for both.
- Matplotlib figures close on success and failure, and Plotly figures retain their
  original titles and layout after publication.

### Pipeline and serialization

- `plots=None` normalizes to `[]`.
- `plots=[model]` serializes as a model reference and resolves to the same reconstructed
  object after round-trip.
- `plots=[measurer]` does the same for a measurement slot.
- equal but non-identical Pydantic objects do not collapse into one reference.
- inline plot models round-trip by class and parameters.
- invalid inputs and duplicate IDs fail with actionable messages.
- QC plot references round-trip by recipe `instance_id`; a runtime check is never
  serialized, and one QC run uses the same live check for `analyze()` and plotting.

### Analysis artifacts

- `LinearLagModel` writes class-named CSV and Parquet files.
- manifest entries point to existing files and use relative paths.
- runtime registry values replace older values on GUI rerun.
- consumers resolve in-memory tables before disk artifacts.
- legacy analysis readers remain read-only if retained.

### CLI

- local CPU, measure-only, staged GPU Stage 3, recompile, and SLURM script paths all
  dispatch the appropriate plot lifecycle.
- per-image output uses the same cached image instance.
- plot consumers receive images at call time and must not strongly retain whole
  `Image` objects after rendering; compact derived measurement caches are allowed.
  Cached NumPy crops must own their buffers because a small view can otherwise
  retain the complete plate-sized backing array.
- aggregate plot ordering follows measurements -> analyses -> QC.
- plot failures do not suppress canonical data artifacts.
- `PlotMeasTimeSeries` writes one file per strain under its class-named folder and a
  manifest that maps raw strain labels to filenames.
- `PlotColonyMetricOverTime` exercises a concrete `on="Shape_MeanRadius"` case with
  multi-column environmental groups and unaggregated replicate traces.
- recipe-backed `PlotQc` receives the analyzed check returned by `run_qc()` plus its
  dynamically resolved table in one `QcPlotSubject`.

### GUI

- analysis rendering calls `report()`.
- measurement, analysis, and QC update events refresh only dependent plots.
- standalone deliverables bundles resolve analysis inputs through relative manifest
  paths.
- single-page and multi-page Matplotlib and Plotly results render through one adapter.
- repeated Matplotlib refreshes close all rasterized and abandoned figures.

### Removal guards

- no `Image.plot` property remains;
- no plotter-registry public exports remain;
- no plotting-object `.dash()` or `.dashboard()` calls remain;
- an explicit allowlist protects `Image.dash()` and channel accessor `.dash()` calls.

## 16. Documentation requirements

- Add `phenotypic.abc_.plotting` beside the other ABC capability classes in the ABC API
  documentation. Document `phenotypic.plotting` separately as the runtime and concrete
  plot package, with an explicit capability-versus-runtime boundary.
- Update the ABC hierarchy guide to present lifecycle mixins as orthogonal capabilities,
  and update the mixin guide with the canonical `phenotypic.abc_.plotting` imports and
  reuse examples.
- Document `PlotOutput`, `PlotPage`, backend support, deterministic page naming, and the
  `PlotMeasTimeSeries` grouping model.
- Update pipeline serialization and constructor documentation.
- Replace analysis artifact inventories and CLI tree diagrams.
- Remove the custom plotter extension page and `Image.plot` tutorials, or rewrite the
  diagnostics tutorial around `PlotDiagnostics`.
- Update all examples from `.dash()` / `.dashboard()` to `.report()` only when the
  receiver implements `PhtPlot`.
- Update module guides in `_core`, `_cli`, `abc_`, `sdk_`, and `gui`.
- Update older active design documents when they prescribe the superseded public API;
  historical completed specs may instead receive a short superseded note.

## 17. Acceptance criteria

The feature is complete when:

1. a `LinearLagModel` can be configured once, placed in `plots`, serialized once by
   reference, deserialized to one shared object, analyzed, and rendered;
2. `MeasureSymmetricZones` can be configured once in `meas` and reused as a
   `PlotImage` without a wrapper;
3. configured CLI plot inspections appear under `deliverables/plots/` in every
   supported execution strategy;
4. `PlotMeas` has no input configuration and always receives the measurement mirror;
5. `PlotAnalysis` and `PlotQc` dynamically resolve measurement or named-analysis input;
6. analysis output is named for its analysis identity and registered in an atomic
   manifest;
7. GUI measurement and analysis changes refresh dependent plots;
8. `PhtPlot` exposes `inspect()` and `report()` only;
9. `Image.dash()` remains unchanged;
10. `Image.plot`, the plotter registry, `FigureProvider`, plotting-object `dash()` and
    `dashboard()`, fixed analysis output names, and `--save-inspect` are absent from
    fresh APIs and writes;
11. affected tests, Ruff, and mypy pass;
12. a `PlotMeasTimeSeries` pipeline writes one deterministically named scatter
    time-series page per strain, with environmental groups as subplots and replicates as
    separate traces;
13. single-page and multi-page Matplotlib and Plotly results can be saved by the CLI and
    rendered by GUI consumers without plot-class-specific backend branching;
14. a recipe-backed `PlotQc` reuses the exact check instance analyzed by `run_qc()` and
    receives its dynamic input, QC database, and review-state snapshot through
    `QcPlotSubject`;
15. `PhtPlot`, `PlotImage`, `PlotMeas`, `PlotAnalysis`, and `PlotQc` are defined and
    publicly imported from `phenotypic.abc_.plotting`, documented in the ABC API, and
    not re-exported from `phenotypic.plotting`.
16. image-consuming reports are stateless with respect to whole `Image` objects;
    convenience reuse may use a weak reference but never extend image lifetime.
