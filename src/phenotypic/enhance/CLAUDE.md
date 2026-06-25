# phenotypic.enhance

## Implementation Conventions

- Concrete enhancers subclass a **purpose-group marker ABC**, not `ImageEnhancer`
  directly. Pick the group that matches what `_operate` produces: `FocusEdge`
  (edge/ridge response map), `FocusBlob` (blob/scale-space response),
  `Smoothing` (kernel/diffusion blur), `ImageDenoiser` (noise-estimate-driven
  restoration), `BackgroundSubtraction` (removes slow-varying background),
  `MorphologicalFiltering` (structuring-element small-feature ops), or
  `ContrastAdjustment` (intensity/contrast remapping). All live in
  `phenotypic.abc_`, all subclass `ImageEnhancer`, and all add no methods/params —
  they only label intent. The markers are deliberately **not** re-exported from
  this package's `__init__`, which keeps them out of the GUI enhancer dropdown
  (the builder registry walks `phenotypic.enhance`). The taxonomy is pinned by
  `tests/unit/abc_/test_enhancer_taxonomy.py`.
- Hessian-based filters default to `black_ridges=False` (bright structures on dark
  background), matching the `detect_mat` convention where colonies appear bright.
- `SetDetectMode` is the only class inheriting `ImageOperation` instead of
  `ImageEnhancer` — it resets `detect_mat` rather than modifying it. Don't use
  it as a reference implementation for new enhancers.
- `CompositeEnhance` is a **meta-enhancer**: it subclasses `ImageEnhancer`
  directly (not a purpose-group marker), mirroring how `CompositeDetector`
  subclasses `ObjectDetector` directly. It applies a `List[OperationField |
  None]` of enhancers/pipelines to the same input and reduces their
  `detect_mat` maps pixel-wise (`max`/`mean`/`min`/`median`, optional `[0,1]`
  clip). Its produce-type depends on its children, so it has no single marker
  ABC and is exempt from the `test_enhancer_taxonomy.py` roster.
