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

### InputLayerMixin
Adds an appended `input_layer: InputLayer` field selecting the operation's source
array. Provides `_read_input_layer(image)` (read-only `detect_mat` view, or a
read-only float32 `rgb` array normalized to [0, 1]), `_project_to_detect_mat(image,
arr)` (collapses a 3-D result via the image's own `detect_mode`; 2-D passes through),
and `_guard_input_range(arr)` (rescales into [0, 1] only when the input strays out,
raises on NaN/inf, and is skipped entirely when `norm is None`). An operation that
mixes this in without `NormalizedOutputMixin` has no `norm` field and is guarded as
if `norm="clip"`. The only layer written is still `detect_mat`.
Location: `mixin/_input_layer_mixin.py`.

### LazyWidgetMixin
Auto-generates Jupyter widgets for parameter tuning from the operation's pydantic `model_fields`.
Included in all `ImageOperation` subclasses automatically.
Location: `mixin/_lazy_widget_mixin.py`.

### NormControlMixin
Disables output normalization in composite operations. Provides
`_disable_normalization(operation)`, which duck-types on `norm` and returns a shallow
copy with `norm=None` — for inner operations running in non-normalized domains (e.g.
inside a Generalized Anscombe Transform region). Operations with no `norm` field are
returned unchanged. Renamed from `ClipControlMixin` in 0.18.0; the old name is gone.
Location: `mixin/_norm_control_mixin.py`.

**Fail-open hazard.** Because the check is a duck-type on `.norm`, an operation that
normalizes its output but never inherits `NormalizedOutputMixin` — so carries no
`norm` field — is returned **unchanged, with no error**. Its normalization stays
active inside the GAT region, where the signal is deliberately outside [0, 1]:
the stabilized values run to ~30, and either policy (`clip` saturating or `rescale`
remapping) drives the inverse transform to all zeros. Declare the policy through
`NormalizedOutputMixin`, and list its inert value in `_GAT_DEFER_VALUES`.

### NormalizedOutputMixin
Supplies the appended `norm: NormOut` field (`"clip"` default / `"rescale"` / `None`)
and `_apply_norm(arr)`, upholding `detect_mat`'s [0, 1] contract. `"clip"` saturates
and preserves absolute intensity; `"rescale"` remaps the observed range and therefore
**divides out any purely multiplicative `gain`**; `None` passes through. A
`model_validator` rejects the legacy `clip=` key with a 0.18.0 migration message
rather than pydantic's opaque extra-inputs error.
Location: `mixin/_normalized_output_mixin.py`.

**Field-append pattern.** Both `NormalizedOutputMixin` and `InputLayerMixin` move
their field to the end of the subclass's field order in `__pydantic_init_subclass__`,
so an operation's own parameters keep their natural position in `model_json_schema()`
and `to_json()`. Each hook calls `super()` before popping, so the mixin **earliest in
the MRO ends up last**: `class ContrastGamma(InputLayerMixin, NormalizedOutputMixin,
ContrastAdjustment)` yields `['gamma', 'gain', 'norm', 'input_layer']`.

---

## Other Utilities

- [`branch_pathfinding/`](branch_pathfinding/CLAUDE.md) — multi-source Dijkstra, cost-surface composition, fragment prescreening, path quality filtering, Voronoi partition. Used by `FilamentousFungiDetector`; cost surfaces are the caller's responsibility.
- `PHENOTYPIC_PRELOAD_MODULES` imports custom operation modules before pipeline
  deserialization so classes defined outside the `phenotypic` namespace can
  self-register through their normal class-definition hooks.
- `constants_.py` — framework constants for image data (image modes, image
  types, gamma encodings). `ConstantLabels` and framework-config enums
  (`GAMMA_ENCODINGS`, `PIPE_STATUS`) live here; they subclass `MeasurementInfo`,
  which lives in the public `phenotypic.schema` package. The `IMAGE` enum and
  semantic metadata vocabulary (`SAMPLE`, `CONDITION`, …) live in
  `phenotypic.schema` too, since they name `Metadata_*` columns/keys.
- `_metadata_helpers.py` — the canonical metadata boundary. Use
  `ensure_metadata_prefix` for bare/canonical/exact-legacy spellings,
  `metadata_member_for_header` / `metadata_owner_for_header` for semantic
  routing, and `normalize_metadata_columns` at DataFrame ingress. Never infer an
  owner by parsing a prefix. `is_metadata_header` accepts canonical flat headers
  and the finite exact legacy registry, not arbitrary `MetadataFoo_*` lookalikes.
- `_metadata_migration.py` — explicit durable migration APIs:
  `preflight_metadata_schema`, `migrate_metadata_file`,
  `migrate_metadata_bundle`, and `rollback_metadata_migration`. Migration uses
  fingerprints, prepared/applied receipts, atomic replacement, and copy-on-write
  HDF updates. HDF layout `schema_version` is independent from the metadata
  namespace marker. Bundle migration owns authoritative sources only; an
  external `--metadata` CSV and its byte-exact
  `deliverables/metadata.csv` startup snapshot are normalized in memory and
  never rewritten by recompile.
- `_io_constants.py` — CLI artifact filenames + directory names + JSON
  contract keys + environment variable names + path-builder helpers.
  Shared between the CLI and GUI; both should import from here rather than
  re-spelling. Templated paths (e.g. `chunk_{id:03d}.parquet`) live as
  private `Final[str]` constants paired with public render functions.
- `typing_.py` — Literal type aliases for closed value sets used at public
  boundaries (`FootprintShape`, `DetectMode`, `FilFinderOutput`,
  `FilFinderPruneCriteria`, `FilamentousFungiReconnectStrategy`, `ExecutionMode`,
  `ImageTypeName`, `ProcessingStatus`, `RecompileTaskType`,
  `CheckpointType`, `FailureSource`, `NormOut`, `InputLayer`).
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
- `orientation_fields/` — branch-tracking-free literal skeleton-ring crossing
  transforms, equal-crossing outward profiles, and composable Matplotlib
  diagnostics. Computation accepts explicit masks, orientation/coherence fields,
  center, distance map, and radii; enhancement and object detection remain caller
  responsibilities.
