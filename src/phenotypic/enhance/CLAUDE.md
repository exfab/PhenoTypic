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
  subclasses `ObjectDetector` directly. It applies its `ops` (a
  `List[OperationField | None]` of enhancers/pipelines — same field name as
  `CompositeDetector.ops`) to the same input and reduces their
  `detect_mat` maps pixel-wise (`max`/`mean`/`min`/`median`, optional `[0,1]`
  clip). Its produce-type depends on its children, so it has no single marker
  ABC and is exempt from the `test_enhancer_taxonomy.py` roster.

## Phase congruency: `FocusEdgePhase` and `FocusEdgeMonogenicPhase`

Both are **ports** of Peter Kovesi's code — `phasecong3` (oriented filter bank) and
`phasecongmono` (monogenic signal, no orientation sweep). They share
`_monogenic_kernels.py`. Before touching either, read the
**`porting-a-reference-algorithm`** skill and
`docs/superpowers/specs/2026-07-08-alt-phase-detection/drift-register.md` (rows `M1`–`M11`).

Three things that have each cost real time:

- **The three references disagree**, and in `M6`/`M8` the *source text looks identical*.
  numpy's `/` is not MATLAB's `./`; numpy's `np.histogram` is not MATLAB's `histc`.
  `phasepack` is the only one that runs under `import`, which makes it the one most often
  misread as Kovesi. **Runnability and authority are unrelated.** Settle disputes by
  *executing* the reference, not by reading it.
- **`epsilon` is a seam.** `_phasecong3` passes `1e-5` (phasecong3's) into `spread_weight`,
  whose module lives beside `EPSILON_MONOGENIC = 1e-4` (phasecongmono's). Substituting one
  for the other shifts `pc_sum` by 7.5% and no behavioural test can see it —
  `TestTheEpsilonSeamIsLocked` exists for exactly this.
- **`FocusEdgeMonogenicPhase` is the first `FocusEdge` whose output need not be a response
  map.** `output="pc"` is one; `output="orientation"` and `output="feature_type"` are angle
  fields, meaningful only where `pc` is high (on `load_synth_yeast_plate`, 89.6% of pixels
  have `pc < 0.02`, and the angle output over those is noise spanning the full `[0,1]`).
  Kovesi consumes `or` masked by `pc`. They are **diagnostic outputs**; do not feed them
  straight into a detector.

`monogenic_phase_congruency` raises on `n_scale < 2`, `sigma_onf >= 1.0`, `mult <= 1.0`
(`M9`, `M10`). Each of those returned an all-zero or all-NaN map before the guard. Keep it
that way: a plausible array of zeros is the worst possible answer, and an all-NaN one is
worse still — it passes a naive `0 <= detect_mat <= 1` check by comparing false to
everything.
