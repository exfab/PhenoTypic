# Tools Module

Utility mixins, helpers, and constants.

## Mixins (`phenotypic.tools_.mixin`)

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
Auto-generates Jupyter widgets for parameter tuning from `__init__` params.
Included in all `ImageOperation` subclasses automatically.
Location: `mixin/_lazy_widget_mixin.py`.

### ClipControlMixin
Controls output clipping in composite operations. Provides `_disable_clipping(operation)`
for operations in non-normalized domains.
Location: `mixin/_clip_control_mixin.py`.

---

## Other Utilities

- [`branch_pathfinding/`](branch_pathfinding/CLAUDE.md) — multi-source Dijkstra, cost-surface composition, fragment prescreening, path quality filtering, Voronoi partition. Used by `FilamentousFungiDetector` and `MeasureRadialExpansion`; cost surfaces are the caller's responsibility.
- `constants_.py` — framework constants for image data (image modes, image
  types, gamma encodings, metadata labels). `ConstantLabels` and framework-config
  enums (`GAMMA_ENCODINGS`, `PIPE_STATUS`, `METADATA`) live here; they subclass
  `MeasurementInfo`, which now lives in the public `phenotypic.schema` package
  (measurement-column enums live there too).
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
