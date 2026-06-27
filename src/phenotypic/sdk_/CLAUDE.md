# SDK Module

Utility mixins, helpers, and constants.

## Mixins (`phenotypic.sdk_.mixin`)

### FootprintMixin
Morphological structuring elements via `_make_footprint(shape, width)`.
Shapes: `"disk"`, `"square"`, `"diamond"`. Combine with `ImageEnhancer`/`ObjectRefiner`:
`class MyOp(FootprintMixin, ImageEnhancer)`. See [abc_/CLAUDE.md](../abc_/CLAUDE.md).
Location: `mixin/_footprint_mixin.py`.

### GridInferenceMixin
Infers grid structure from binary masks via peak detection. Provides
`_infer_grid_shape(mask)` and `_estimate_edges(mask, axis, n_bins)`.
Used by detectors/refiners on gridded plate images.
Location: `mixin/_grid_inference_mixin.py`.

### LazyWidgetMixin
Auto-generates Jupyter widgets for parameter tuning from the operation's pydantic `model_fields`.
Included in all `ImageOperation` subclasses automatically.
Location: `mixin/_lazy_widget_mixin.py`.

### ClipControlMixin
Controls output clipping in composite operations. Provides `_disable_clipping(operation)`
for operations in non-normalized domains.
Location: `mixin/_clip_control_mixin.py`.

---

## Other Utilities

- [`branch_pathfinding/`](branch_pathfinding/CLAUDE.md) — multi-source Dijkstra, cost-surface composition, fragment prescreening, path quality filtering, Voronoi partition. Used by `FilamentousFungiDetector`; cost surfaces are the caller's responsibility.
- `register/` — operation self-registration; underpins the `PHENOTYPIC_PRELOAD_MODULES` workflow for resolving custom op classes defined outside the `phenotypic` namespace.
- `constants_.py` — framework constants for image data (image modes, image
  types, gamma encodings). `ConstantLabels` and framework-config enums
  (`GAMMA_ENCODINGS`, `PIPE_STATUS`) live here; they subclass `MeasurementInfo`,
  which lives in the public `phenotypic.schema` package. The `METADATA` enum and
  the experimental-tag vocabulary (`SAMPLE_METADATA`, `CONDITION_METADATA`, …)
  now live in `phenotypic.schema` too, since they name `Metadata_*` columns/keys.
- `_io_constants.py` — CLI artifact filenames + directory names + JSON
  contract keys + environment variable names + path-builder helpers.
  Shared between the CLI and GUI; both should import from here rather than
  re-spelling. Templated paths (e.g. `chunk_{id:03d}.parquet`) live as
  private `Final[str]` constants paired with public render functions.
- `typing_.py` — Literal type aliases for closed value sets used at public
  boundaries (`FootprintShape`, `DetectMode`, `ExecutionMode`,
  `ImageTypeName`, `ProcessingStatus`, `RecompileTaskType`,
  `CheckpointType`, `FailureSource`).
- `funcs_.py` — timing decorators, mask validation.
- `hdf_.py` — HDF5 storage utilities.
- `slurm_.py` / `submitit_.py` / `monitor_slurm_jobs.py` — SLURM integration.
- `generate_report.py` — report generation.
- `viz/` — shared visualization layer: the centralized Plotly theme
  (`viz.figures.apply_theme`, `PHENOTYPIC_TEMPLATE_NAME`, Okabe-Ito palette),
  the matplotlib rcParams mirror (`phenotypic_mpl_context`/`phenotypic_rc`),
  and the ipywidgets notebook shell (`viz.notebook.build_notebook_dashboard`).
  UI toolkits stay lazily imported; the theme imports plotly but no toolkit
  (enforced by `tests/unit/viz/test_import_rules.py`).
