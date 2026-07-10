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
- `ContrastGamma`, `ContrastLog`, `ContrastSigmoid`, and `ContrastStretching` inherit
  `InputLayerMixin`, so they can read the pristine `rgb` layer instead of `detect_mat`.
  The 3-D result is collapsed back to 2-D through the image's own `detect_mode` — they
  still write `detect_mat` and nothing else. `ContrastStretching` alone has no `norm`
  field: percentile rescaling to [0, 1] is its algorithm.
- `input_layer="rgb"` is a **no-op under the per-pixel-selection `detect_mode`s**. A
  monotonically increasing pointwise curve commutes with a per-pixel selection —
  `min(f(r), f(g), f(b)) == f(min(r, g, b))` — so `red`/`green`/`blue`/`MinRGB`/`HsvV`
  measure `max|Δ| = 0.000000` either way. It only changes the output under modes that
  *mix* channels (`gray`, `LabL`/`LabA`/`LabB`, `HsvS`). Two exceptions to keep in mind:
  `ContrastSigmoid(inv=True)` is the one **decreasing** curve, so it anti-commutes and
  `input_layer` becomes meaningful even under `MinRGB`/`HsvV`; and `InvS` (`= min/max`)
  commutes *exactly* with `ContrastGamma`, because a power law commutes with a ratio.
  `ContrastLog(inv=True)` is skimage's *inverse-log* (`(2**x - 1) * gain`) — still
  increasing, so it still commutes.
- `FocusEdgeLaplace` defaults to `norm="rescale"`, not `"clip"`: a Laplacian is signed,
  and clipping would discard the entire negative lobe.
- `CompositeEnhance` is a **meta-enhancer**: it subclasses `ImageEnhancer`
  directly (not a purpose-group marker), mirroring how `CompositeDetector`
  subclasses `ObjectDetector` directly. It applies its `ops` (a
  `List[OperationField | None]` of enhancers/pipelines — same field name as
  `CompositeDetector.ops`) to the same input and reduces their
  `detect_mat` maps pixel-wise (`max`/`mean`/`min`/`median`, optional `[0,1]`
  clip). Its produce-type depends on its children, so it has no single marker
  ABC and is exempt from the `test_enhancer_taxonomy.py` roster.
