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

`monogenic_phase_congruency`, `monogenic_channel_response` **and**
`congruency_from_accumulators` all raise on `n_scale < 2`; the first two also raise on
`sigma_onf >= 1.0` and `mult <= 1.0` (`M9`, `M10`). Each returned an all-zero or all-NaN map
before the guard. Keep it that way: a plausible array of zeros is the worst possible answer,
and an all-NaN one is worse still — it passes a naive `0 <= detect_mat <= 1` check by
comparing false to everything. `congruency_from_accumulators` is worse than either: at
`n_scale=1` with real accumulators it silently **saturates the frequency-spread weight to
1.0**, disabling the penalty and returning a finite, in-range map *larger than the truth*
(`0.5789` against `0.5613`).

## Colour phase congruency: `FocusEdgeColorPhase`

Per-channel monogenic PC over three luminance-first colour channels, fused by `joint`
(default), `coherent`, or `l2` (the CMPCM paper's rule). Shares `_monogenic_kernels.py`
through the accumulator seam (`MonogenicChannel`, `monogenic_channel_response`,
`congruency_from_accumulators`) and adds `_color_phase_kernels.py`. Drift rows `C2`–`C17`.

Five things that will cost you time otherwise:

- **It reads `image.rgb`, not `detect_mat`.** The second operation in this package that is a
  pipeline *source* rather than a transform — `SetDetectMode` is the first — so **any
  enhancer placed before it has no effect on its output.** Drift `C2`.
- **On round-colony plates, colour buys nothing.** Measured under lateral chromatic
  aberration: at `δ=3` on `load_synth_yeast_plate`, plain `FocusEdgeMonogenicPhase` on
  luminance localizes to `1.143` px and beats *every* fusion mode (`joint 1.375`,
  `coherent 1.700`, `l2 1.776`). The operation is **scoped to filamentous plates**, where
  `joint` reaches `1.008` against luminance's `1.158`. CA *creates* chromatic edges, and
  `joint` asserts them coherently — so its edge follows the displaced chroma rather than
  merging it. Spec §7.2.1–§7.2.2.
- **Only `coherent` builds a fused monogenic vector.** `joint` sums scalar energies; `l2`
  sums three finished congruency maps. `output="orientation"` and
  `output="feature_type"` are available for all fusion modes, but under `joint` and `l2`
  they are diagnostic maps from the weighted fused vector, not maps that produced `pc`.
  Prefer `output="pc"` for detection. Drift `C15`.
- **`color_space="hsv"` band-passes raw hue across its wrap discontinuity** and manufactures a
  phantom edge at near-red boundaries — `115.7×` its own background, interior of the frame.
  Retained because CMPCM uses HSV; `lab` is the default and has no seam. Drift `C16`.
- **The output is not invariant to a global rescale of the weight vector**, and the operation
  cannot even express one (luminance is pinned at `1.0`). The `ε` in `A_max + ε` sits inside a
  `g = 10` sigmoid; masked at `pc > 0.05`, `c = 0.01` moves the output by `6.9%`. An earlier
  spec revision claimed `~1%` and named the wrong `ε`. Drift `C17`.

**Whether a call-site bug is visible depends on the parameter regime.** Hardcoding the
fusion's `n_scale`, dropping `deviation_gain`, swapping the chroma weights and un-pinning
luminance all survived a bit-identity test with every field off its default — at
`cutoff=0.35, g=14, deviation_gain=1.2` the spread sigmoid is saturated wherever the
phase-deviation term survives, so the fusion's `n_scale` changes `pc` by *exactly* `0.0`.
Test the forwarding with a spy on the kernel call, not with a number.
