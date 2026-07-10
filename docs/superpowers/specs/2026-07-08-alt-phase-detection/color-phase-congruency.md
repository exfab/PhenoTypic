# `FocusEdgeColorPhase` — design

**Status: IN PROGRESS.** `FocusEdgeMonogenicPhase` has landed. Execution plan:
[`plans/2026-07-09-focus-edge-color-phase/plan.md`](../../plans/2026-07-09-focus-edge-color-phase/plan.md).
**Companion:** [`references.md`](./references.md). **Deviations:** [`drift-register.md`](./drift-register.md).
**Optional lift:** [`conformal-lift.md`](./conformal-lift.md) — gated, not required.
**Numeric claims:** re-derived on demand by
[`fusion_algebra.py`](../../logic_validation_scripts/2026-07-09-focus-edge-color-phase/fusion_algebra.py).
Three of them were wrong; see §3.1, §4.2, §5.1.

**This is the colour phase-congruency variant.** Per-channel monogenic phase congruency, then a
cross-channel fusion. It reuses `_monogenic_kernels.py` wholesale and adds no new signal theory.

The conformal lift of the CMPCM paper is **not** part of this operation. It contributes nothing to the
congruency output — its `f_z` channel is even (a Laplacian) and correlates `−0.97` with the even
channel `c`, while a true odd channel correlates `0.00` (`references.md` §9). It is scoped separately
and gated on an experiment it may well fail.

---

## 1. Architectural drift: this operation reads `rgb`

Every existing `FocusEdge` subclass is a `detect_mat → detect_mat` transform. Colour phase congruency
is defined on colour, and `rgb` is not a supported `detect_mat` layer, so this operation sources from
`image.rgb[:]` (through `image.color.Lab` / `image.color.hsv`, which derive from `rgb` via
`color.XYZ`) and writes only `detect_mat`.

**Legal** under `@validate_operation_integrity`, which forbids *mutating* `rgb`/`gray` and says
nothing about reading them. But it has a user-visible consequence: **any enhancer placed before this
operation in an `ImagePipeline` has no effect on its output.** It is a *source*, like `SetDetectMode`.

Record in the class docstring (a `Note:` mirroring `SetDetectMode`'s warning), in
`enhance/CLAUDE.md` (which already carves out `SetDetectMode` and `CompositeEnhance`), and in
`abc_/_enhance_markers/_focus_edge.py`.

---

## 2. Data flow

```
rgb ──► color.Lab[:] or color.hsv[:] ──► three scalar channels
          │
          ├─ per channel i: monogenic PC (monogenic-phase-congruency.md §2)
          │     → Eᵢ (energy), A_Σ,ᵢ (summed amplitude), Wᵢ (spread weight), Tᵢ (Rayleigh threshold)
          │
          └─ fuse (§3) ──► detect_mat[:] = clip(out, 0, 1)
```

The per-channel machinery is **exactly** `FocusEdgeMonogenicPhase`, verbatim, including its
`ε = 1e-4`, its `acos`/`deviation_gain` numerator, and its geometric noise factor. It is a port; do
not vary it per channel.

The `clip` is load-bearing for `l2` and redundant for `joint`. Keep it in both. **The internal helper
returns the un-clipped value**, because §7's PFOM regression needs the paper's actual quantity.

---

## 3. Fusion

`w = (1, chroma_weight_1, chroma_weight_2)` in **luminance-first** order (§4.2), luminance pinned at
`1.0`.

**`fusion="joint"` — default.** One shared denominator, matched norms:

```
E_total = Σᵢ wᵢ·Eᵢ                    A_total = Σᵢ wᵢ·A_Σ,ᵢ
T_total = Σᵢ wᵢ·Tᵢ                    A_max   = Σᵢ wᵢ·maxₛ A_{i,s}
width   = (A_total/(A_max + ε) − 1)/(n_scale − 1)
W_total = 1/(1 + exp(g·(cutoff − width)))
out     = W_total · max(1 − dg·acos(E_total/(A_total+ε)), 0) · max(E_total − T_total, 0)/(E_total + ε)
```

**`fusion="coherent"`.** As `joint`, but `E_total = ‖Σᵢ wᵢ·vᵢ‖` over the stacked per-channel
3-vectors. Cancels opposite-phase responses across channels. **Hazard:** it also annihilates a genuine
anti-correlated chromatic edge — a colony boundary where lightness falls as yellowness rises. Opt-in,
never default.

**`fusion="l2"`.** The CMPCM paper's rule: per-channel congruency `Fᵢ`, then `out = √(Σᵢ (wᵢFᵢ)²)`,
range `[0, ‖w‖]`. **Do not divide by `‖w‖`** — the paper does not, and §7 must check the paper's actual
quantity. Clip only at the write site.

### 3.1 Why the numerator must match the denominator's norm

An L2 numerator over an L1 denominator is **wrong**, and badly so. For a perfect edge
(`Eᵢ = A_Σ,ᵢ` in every firing channel), with the shipped `deviation_gain = 1.5`:

| firing channels | `√(Σ(wE)²)/Σ(wA)` | response | `ΣwE/ΣwA` | response |
|---|---|---|---|---|
| one only | 1.0000 | **1.0000** | 1.0000 | 1.0000 |
| all three, equally | 0.5774 | **0.0000** | 1.0000 | 1.0000 |
| `(0.804, 0.013, 0.183)` | 0.8247 | **0.0983** | 1.0000 | 1.0000 |

A coherent edge in *all three channels* is annihilated **exactly**; a **single-channel** response
passes at full strength. That inverts §7.2's CA acceptance criterion: under chromatic aberration the
correctly registered edge fires in all channels and would be suppressed, while the displaced
single-channel fringe fires alone and passes. It is an amplifier of exactly the artefact we set out
to attack.

> **Re-derived 2026-07-09.** An earlier revision of this table printed responses of `0.0091` and
> `0.1425`, which **no single `deviation_gain` produces** — row 2 would need `1.0373`, row 3 would
> need `1.4265`. At the shipped `1.5`, `max(1 − 1.5·acos(0.5774), 0)` clamps to zero. The `ratio`
> column was always right; only the `response` column drifted. The conclusion is unchanged and
> strengthened. Re-derived on demand by
> `logic_validation_scripts/2026-07-09-focus-edge-color-phase/fusion_algebra.py`, checks 01 and 02.

### 3.2 Why `joint` is the default

**Not** because of a `[0,1]` bound — every variant is bounded once `l2` is clipped.

The real reason: **`l2` has no cross-channel interaction at all.** It is three independent detectors
combined after the fact, so incoherent chroma amplitude can never veto a spurious luminance edge. Only
a shared denominator can do that, and that veto is the sole mechanism by which colour is expected to
help here. §7 exists to measure whether it does.

### 3.3 What joint fusion actually does

Stated plainly, because the field notebook claims otherwise (`references.md` §3):

- Channel amplitude enters the **denominator** whether or not that channel's structure is coherent. A
  channel with amplitude but no phase agreement — grain, sensor noise, speckle — therefore **vetoes**
  the other channels' edges. Real, and useful in the low-SNR regime.
- A channel's coherent structure enters the **numerator**, so it can **assert** an edge.
- Joint fusion does **not** suppress chromatic-aberration fringes. Lateral CA is the same edge
  misregistered across channels, not a single-channel artefact. Coherent summation *merges* the
  displaced edges into one response near the amplitude-weighted centroid — useful, but a different
  mechanism.
- Joint fusion does **not** suppress an isolated single-channel feature over a flat background. If the
  other channels are locally flat, `A_total ≈ A₁` and `out ≈ E₁/A₁`. A colony pigmented in one channel
  only survives.

---

## 4. The channel prior

### 4.1 Raw CIELAB is already the common scale

CIE76 defines `ΔE*ab = √(ΔL² + Δa² + Δb²)`, so a Euclidean norm over **raw** `L*a*b*` is already a
perceptual distance. `(1,1,1)` on raw Lab **is** the ΔE metric; dividing by nominal axis ranges
(`/100`, `/128`, `/128`) would corrupt an already-normalized space, biasing against chroma by
`128/100 = 1.28×`.

> Caveat: CIELAB is only *approximately* uniform; ΔE76's known weakness is in saturated blues. Agar
> plates sit near the achromatic axis, where ΔE76 is well behaved.

So the measured luminance dominance (69–80% on **real** plates, `references.md` §6.1) is a physical
fact in perceptual units, not a units artifact. A weight can only override it — which is what a prior
does. Per-image data-dependent scaling is rejected: `std` hands 71.9% of the synthetic plate's vote to
a near-flat `b*`, and any data-dependent gain makes the operation something other than a pure function
of its fields.

### 4.2 Two degrees of freedom, not three

`E_total`, `A_total`, `T_total`, `A_max` are all 1-homogeneous in `w`, so **the ratio** is invariant to
a global rescale. **The output is not**, and by far more than an earlier revision of this section
claimed.

> **Corrected 2026-07-09.** That revision said the invariance held "up to `O(ε/A_total)`… ~1% over
> `c ∈ [0.01, 100]`". Both the bound and the number are wrong. The `ε` that breaks 1-homogeneity is
> **not** the one in `E_total + ε`. It is the one in `A_max + ε`, which sits inside
> `width = (A_total/(A_max + ε) − 1)/(n_scale − 1)` and is then fed to a sigmoid of sharpness
> `g = 10`. Near that sigmoid's knee a small shift in `width` moves `W` by a large *relative* amount,
> and `W` multiplies the entire output. Measured over `c ∈ {0.01, 100}`, 20 000 draws per regime:
>
> | amplitude regime | max relative change in the output |
> |---|---|
> | `A ∈ [0.5, 5.0]` (Lab `L*` scale) | **100.0%** |
> | `A ∈ [0.05, 0.5]` (edge-pixel scale) | **670.0%** |
> | `A ∈ [1e-3, 0.5]` | **778.9%** |
>
> Any invariance test must therefore mask low-response pixels and derive its tolerance from
> `ε/(c·A_total)` and `|dW/d(width)| ≤ g/4`. See §7 test 4, drift `C17`, and `fusion_algebra.py`
> check 04.

Pin luminance at `1.0`. The two chromatic axes carry the two real degrees of freedom, as **scalar**
fields so they enter the tune annotation-coverage gate.

| `color_space` | pinned at `1.0` | `chroma_weight_1` | `chroma_weight_2` |
|---|---|---|---|
| `lab` | `L*` | `a*` | `b*` |
| `hsv` | `V` | `H` | `S` |

**Luminance-first order.** The accessors are natively `Lab → (L*, a*, b*)` and `hsv → (H, S, V)`.
This operation reorders `hsv` to `(V, H, S)` so index `0` is always the pinned luminance axis. Read
literally, an earlier §3 (*"in `color_space` channel order"*) pinned **`H`** under `hsv` — the one
channel that is circular, ill-conditioned at low saturation, and carries no luminance at all. Pinned
by `test_focus_edge_color_phase.py::TestChannelOrderIsLuminanceFirst`.

### 4.3 Why the `TuneSpec` upper bound is 8.0

"Parity" = the weight at which a channel contributes as much to `A_total` as `L*` does. On the two
**real** samples: `b*` parity is `2.6` (Rhodotorula) and `4.4` (Neurospora); `a*` parity is `19` and
`61`. `[0, 8]` brackets "chroma off" through "chroma dominates" for the axis that carries signal, and
deliberately refuses to let `a*` reach parity.

> `load_synth_yeast_plate` needs `b*` parity **27.8**, far outside this range, because its `b*` is
> near-flat (`std = 0.158`). It is not representative — and it is the **doctest plate**, so doctests
> must not assert chroma behaviour.

### 4.4 Why two scalars and not one

A single shared chroma weight is multiplicative, so it preserves the measured `b*`:`a*` ratio (7.2× and
13.9×, same direction on both real plates) and can never promote the weaker axis. Two are specified
because the second degree of freedom is real: a **loud, incoherent** chroma axis vetoes without
asserting, and JPEG 4:2:0 subsampling and Bayer demosaic noise are both anisotropic between `a*` and
`b*`. And under `hsv`, `H` and `S` are not comparable at all — at `S = 0.01` an RGB perturbation of
`0.00036` swings hue by `0.01`.

---

## 5. Fields

| field | default | annotation | source |
|---|---|---|---|
| `color_space` | `"lab"` | `ColorSpaceName` | ours; CMPCM uses HSV |
| `fusion` | `"joint"` | `PhaseFusion` | ours; CMPCM uses L2 |
| `chroma_weight_1` | `1.0` | `TuneSpec(0.0, 8.0)`, `Field(ge=0.0)` | ours; §4 |
| `chroma_weight_2` | `1.0` | `TuneSpec(0.0, 8.0)`, `Field(ge=0.0)` | ours; §4 |
| `lift` | `"monogenic"` | `PhaseLift` | `"conformal"` is gated; see `conformal-lift.md` |
| *(all `FocusEdgeMonogenicPhase` fields)* | | | ported verbatim |
| `output` | `"pc"` | `ColorPhaseOutput` | §5.1 — `"pc"` is the only member |

```python
ColorSpaceName   = Literal["lab", "hsv"]
PhaseFusion      = Literal["joint", "coherent", "l2"]
PhaseLift        = Literal["monogenic", "conformal"]
ColorPhaseOutput = Literal["pc"]
```

`lift="conformal"` **must raise `NotImplementedError`** until `conformal-lift.md`'s gate passes. The
field exists so the surface is stable; the path does not exist yet.

### 5.1 Only `pc` is exposed

An earlier revision of this table declared
`ColorPhaseOutput = Literal["pc", "orientation", "feature_type"]` and then defined neither angle
anywhere — not in §2's data flow, not in §3's three fusion formulas, not in §7's tests, not in §8's
acceptance criteria.

It cannot be filled in by analogy, because **only `coherent` builds a fused monogenic vector.**
`joint` sums scalar energies `Σᵢ wᵢ‖vᵢ‖`; `l2` runs the whole congruency chain three times and
combines three finished maps. Under the *default* fusion, an angle read off `Σᵢ wᵢ·vᵢ` would describe
a quantity the response never touched. That is two more unreferenced inventions on top of `C10`, for
outputs nothing in the codebase consumes.

**Decided 2026-07-09 (user): ship `pc` only.** The angles are nevertheless **computed and returned**
by the protected helper `FocusEdgeColorPhase._color_phase_congruency()`, on its `ColorPhaseResult`,
taken from the fused vector `Σᵢ wᵢ·vᵢ` in every mode. This mirrors
`FocusEdgePhase._phasecong3()`, which returns `orientation` and `feature_type` on its
`_PhaseCong3Result` while `output` exposes only `M`/`m`/`pc_sum` — and whose `pc_sum` is likewise not
`‖energy_v‖`. A future consumer can reach them without a breaking change; adding an `output` value
later is non-breaking, removing one is not.

Because the fused vector collapses to `v_L` when both chroma weights are `0`, §7 test 1 pins the two
angles against `FocusEdgeMonogenicPhase` at no extra cost. Drift `C15`, which also records the
cancellation hazard: the odd pair's sign encodes edge polarity, so an anti-correlated chromatic edge
(`L*` falling as `b*` rises) cancels in the sum, and the angle is least reliable exactly where colour
is doing the most work.

---

## 6. Error handling

- **Achromatic input.** `ValueError` when all three RGB channels are identical: `a*`, `b*` are then
  identically zero and `joint` degenerates to a luminance congruency divided by itself.
- **`lift="conformal"`.** `NotImplementedError`, raised at **construction** by a
  `model_validator(mode="after")`. pydantic v2 does not trap `NotImplementedError`, so its type
  survives — unlike a `ValueError` raised inside `_operate`, which `ImageOperation.apply` wraps
  twice: into a `RuntimeError` at `abc_/_image_operation.py:423` and then a bare `Exception` at
  `:470`. Measured chain on `FocusEdgePhase(sigma_onf=1.0).apply(...)`:
  `RuntimeError → Exception → ValueError`. Any test asserting a `ValueError` from `.apply()` must
  therefore walk the whole `__cause__` chain.
- Everything else is ordinary pydantic bounds. `ε = 1e-4`, inherited from the monogenic port.

The `acos` in every fusion mode is safe without a clamp firing: `E_total ≤ A_total` holds for both
`joint` (`Σwᵢ‖vᵢ‖ ≤ Σwᵢ A_Σ,ᵢ` by the triangle inequality per channel) and `coherent`
(`‖Σwᵢvᵢ‖ ≤ Σwᵢ‖vᵢ‖`). Measured maxima of `E_total/(A_total + ε)` over 200 000 random draws:
`0.996436` and `0.970076`. So `n_clamped == 0` is an assertable invariant, as it is for
`FocusEdgeMonogenicPhase` (drift `M1`). `fusion_algebra.py` check 03.

---

## 7. Testing

1. **Per-channel fidelity.** With `chroma_weight_* = 0`, the output must equal
   `FocusEdgeMonogenicPhase` applied to the luminance channel, to `rtol=1e-10`. The fusion must not
   perturb the port. Because the fused vector collapses to `v_L` there, this also pins the
   unexposed `orientation` and `feature_type` (§5.1, drift `C15`).
2. **Fusion sanity.** An edge present in all three channels scores higher than the same edge in one
   channel, under `fusion="joint"`. §3.1 as a regression; the L2-over-L1 form fails it outright —
   at the shipped `deviation_gain` its three-channel response is exactly `0`.
3. **The `[0,1]` bound**, all three fusion modes, random non-negative chroma weights. Assert
   `np.isfinite` as well: an all-NaN map passes a naive `0 <= x <= 1` check, because NaN compares
   false to everything (drift `M10`). And `n_clamped == 0` (§6).
4. **Weight-vector scale invariance, over a masked pixel set, with a derived tolerance.** Compare
   `out(w)` against `out(c·w)` for `c ∈ {0.01, 100}` on `load_synth_yeast_plate()`, restricted to
   pixels where `out(w) > 0.05`. The tolerance is **derived, not chosen**: `ε/(c·A_total)` bounds the
   `E_total + ε` term and `|dW/d(width)| ≤ g/4` bounds the sigmoid term. Assert the measured
   deviation against that bound, **and** assert the bound is tighter than the `0.05` mask floor on
   this image — otherwise the test cannot fail. **Unmasked, this comparison moves by up to 779%**
   (§4.2, drift `C17`). The retracted `rtol=2e-2` was a guess, and a guess loose enough that the
   anchor did no work.
5. `detect_mat` in `[0,1]`; `rgb`/`gray` unmutated; `to_json`/`from_json` round-trip; constructible
   with no arguments; 90° rotation equivariance at `rtol=1e-8` (the FFT's own reproducibility, not a
   guess).
6. `ValueError` on achromatic input — walking the `__cause__` chain, per §6 — and
   `NotImplementedError` on `lift="conformal"`, at construction.
7. **The hue-wrap artefact is demonstrated, not asserted.** Build a flat image of constant `S` and
   `V` whose hue ramps smoothly *through* red. There is no edge in it. `color_space="hsv"` must
   respond anyway; `color_space="lab"` must not. Drift `C16`. If this test ever goes green on `hsv`,
   someone has started unwrapping hue and has silently diverged from CMPCM.
8. Doctests on `load_synth_yeast_plate()`. **They must not assert chroma behaviour** (§4.3).

### 7.1 Ranking regression

The CMPCM paper's Table 1 PFOM is computed on its Fig. 4a "geometry image" (176×298), **whose pixels
are published nowhere**; its §2 pixel spec (173×299) describes Fig. 1a1, a flat colour-model demo with
three ±1 stripes. Two different images. So do **not** attempt to reproduce Table 1.

Instead: build a synthetic geometric colour image with known ideal edges, compute Pratt's Figure of
Merit on the **un-normalized** `l2` output, and assert the *ranking* `colour PC > PC > Canny`, with
`color_space="hsv", fusion="l2"`.

### 7.2 The CA experiment — acceptance criterion for the colour design

`load_synth_filamentous_plate()` returns a `GridImage` (600×800) with an `objmap` of 60 objects. Inject
a radial chromatic aberration of `δ ∈ {0, 1, 2, 3}` px into R and B. Compare boundary localization
error against the `objmap` for `FocusEdgePhase` (baseline), `FocusEdgeMonogenicPhase` on luminance,
and `fusion ∈ {l2, joint, coherent}`.

> Do **not** use `make_synthetic_filamentous_plate()` — it returns a bare `np.ndarray` with no label
> map.

**Prediction:** `joint` merges the displaced edges, so its error stays roughly flat in `δ`, while `l2`
degrades. **If this fails, ship `l2`.** Record the result either way; a null result must not be buried.

---

## 8. Acceptance criteria

1. **§7 test 1** (per-channel fidelity) passes — the fusion does not perturb the port, `rtol=1e-10`,
   including the two unexposed angles.
2. **§7 test 2** (fusion sanity) passes for `joint` and `coherent`.
3. The `[0,1]` bound holds for all three fusion modes under random weights, the output is **finite**,
   and `n_clamped == 0`.
4. **§7.2's** CA experiment has been *run and recorded*, and the shipped `fusion` default matches it.
   A null result is acceptable and must not be buried.
5. `lift="conformal"` raises `NotImplementedError`.
6. `uv run mypy src/phenotypic` and `uv run ruff check` introduce no new errors.
7. The operation appears in the GUI builder's enhancer dropdown (the registry walks
   `phenotypic.enhance.__all__`; `ImageEnhancer` subclasses go 30 → 31).
8. `monogenic_phase_congruency` is **bit-identical** across the accumulator refactor its fusion
   requires, proven by a test that the golden fixture's `rtol=1e-6` demonstrably cannot replace.

> An earlier revision of this list cited "§7.1" for per-channel fidelity and "§7.2" for fusion
> sanity. Those are §7's *numbered items* 1 and 2; §7.1 is the PFOM ranking regression and §7.2 is the
> CA experiment. Corrected 2026-07-09.

---

## 9. Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | `joint` is unvalidated against any external reference. | §7.2 is its acceptance test, with a stated falsification and a fallback (`l2`). |
| 2 | Chroma is 69–80% outvoted by luminance on real plates, so colour may buy little. | §7.2 measures it against a luminance-only baseline. A null result is acceptable and must be recorded. |
| 3 | `T_total = Σ wᵢTᵢ`, `A_max = Σ wᵢ maxₛA` are inventions with no reference. | 1-homogeneous, so §4.2's argument survives. Flagged in the drift register (C10). **Untested against anything external — say so.** |
| 4 | `rgb` sourcing surprises pipeline authors. | §1. |
| 5 | Someone enables `lift="conformal"` before the gate passes. | §5: it raises at construction. §7 test 6. |
| 6 | The accumulator refactor `joint` requires silently moves a bit in the shipped monogenic port. | The two fixtures cannot see it: `rtol=1e-6` has 6.7 orders of slack and has already hidden `np.hypot` and a reciprocal-multiply on this branch. The refactor's oracle is `np.array_equal`, proven able to fail by exactly that `np.hypot` mutation. §8 criterion 8. |
| 7 | `color_space="hsv"` manufactures an edge at near-red boundaries by bandpassing hue across its wrap seam. | Drift `C16`. `lab` is the default. §7 test 7 *demonstrates* the artefact, so it goes red if someone "fixes" it by unwrapping and diverges from CMPCM. |
| 8 | A reader trusts §4.2's old "~1%" and writes a scale-invariance test that cannot fail. | Drift `C17`. §7 test 4 now masks and derives its tolerance, and asserts the bound is tighter than the mask floor. |
