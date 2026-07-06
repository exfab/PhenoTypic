# From Pixels to Orientation Metrics

**Gradients, the structure tensor, and quantifying hyphal direction (Regime B)**

- **Date:** 2026-07-03
- **Scope:** How the image gradient is computed, how gradients are pooled into an
  orientation field (the structure tensor), and how that field is reduced to a small
  set of quantifiable metrics — with a comparison of the Sobel, Scharr, and
  Gaussian-derivative kernels and why we use the Gaussian derivative.
- **Context:** Background for the "Turning & curvature" family (Family D) of the
  branch-phenotypes catalog — the tracing-free, mask/grayscale orientation proxies
  (`Field_Coherence`, `Field_OrientGrad |∇φ|`, `Field_OrientEntropy`) that estimate
  hyphal direction and directional change *without* skeletonizing individual strands.
- **Audience:** Someone comfortable with basic calculus, convolution, and Gaussian
  blurring who wants the *why*, not just the API call.

---

## Pipeline at a glance

```
image ──①──► gradients (Ix,Iy) ──②──► structure tensor J ──③──► {coherence, |∇φ|, entropy} ──④──► {R, turning}
       kernel choice          per-pixel      outer product + pool               field reductions        compact
       (σ_d)                  vectors        (σ_i)                                                       metrics
```

Two Gaussian scales thread through everything:

| Scale | Kernel | Job | Set to |
|---|---|---|---|
| **`σ_d`** (derivative) | `G'_σd` | *measure* each gradient (noise suppression, which edge widths respond) | ≈ hypha width |
| **`σ_i`** (integration) | `G_σi` | *pool* gradient products into a neighborhood statistic (the tensor) | a few × `σ_d` |

---

## ① The gradient — estimating slope from a grid

A continuous image has a gradient `∇I = (∂I/∂x, ∂I/∂y)`; we only have samples on a
pixel grid, so every derivative is an **estimate** from neighboring samples.

### Finite differences and the Taylor read-off

Three ways to estimate `I'(x)` (spacing `h`, `= 1` px for images):

```
forward:  I(x+1) − I(x)
central: [I(x+1) − I(x−1)] / 2
```

Substitute the Taylor series `I(x±h) = I(x) ± h·I' + ½h²·I'' ± ⅙h³·I''' + …` and read
off which derivative survives and what error tags along:

```
forward → I' + (h/2)·I''  + …    → O(h),   biased by I''
central → I' + (h²/6)·I''' + …    → O(h²),  I'' cancels
```

**"Reading off" =** express each neighbor sample as derivatives-at-`x` via Taylor, then
collect terms; the target derivative should have coefficient 1 and the next term(s)
cancel; the lowest surviving power of `h` is the order of accuracy. (Bonus: *adding*
the two series cancels the odd terms and yields the second-derivative stencil
`[1, −2, 1]/h²` — subtract → 1st derivative, add → 2nd derivative.)

Central difference wins for two linked reasons:

- **Symmetric placement** — uses `x±1` equidistantly, so the estimate is anchored on
  `x` (no half-pixel shift; forward difference secretly estimates the slope at `x+½`).
- **Antisymmetric weights** `[−1, 0, +1]` — an *odd* kernel: it sums to zero (kills
  constants → truly a derivative), cancels the even-order (`I''`) error terms, and
  flips sign under reflection (a directional derivative must). This is a general
  **parity law**: *odd kernels ↔ odd derivatives (differentiate); even kernels ↔ even
  derivatives (smooth / 2nd derivative).*

*Numeric check:* central difference is exact for polynomials up to degree 2, and for a
cubic `I=x³` at `h=1` returns `3x²+1`, whose error `1` equals the predicted leading term
`(h²/6)·I''' = (1/6)(6) = 1`.

### Why raw differencing fails: it destroys SNR

In smooth regions `I(x+1) ≈ I(x−1)`, so the *true* difference is tiny — but the two
**independent** noises don't cancel:

```
central diff = [I(x+1) − I(x−1)]/2  +  [ε(x+1) − ε(x−1)]/2
                └── tiny true slope ──┘   └── full noise, survives ──┘
```

You subtract near-equal noisy numbers: noise that was a small fraction of each *value*
becomes a large fraction of the *difference* (`σ_n/g` with `g ≪ V`). Equivalently, in
the frequency domain a derivative is a **high-pass** (`gain ∝ |ω|`); white noise is
broadband/high-frequency-heavy while the signal sits at low frequencies, so the operator
hands back mostly noise. **Fix:** smooth first (a low-pass). Fused, smooth-then-
differentiate becomes a single **band-pass** kernel.

### The three gradient kernels

All three encode the same idea — **difference across the edge, smooth along it** — as a
separable (rank-1) outer product `smoother ⊗ differencer`:

- **Sobel** — `[1,2,1] ⊗ [−1,0,+1]`:
  ```
  [ 1 ]                     [ −1  0  +1 ]
  [ 2 ] ⊗ [−1 0 +1]  =      [ −2  0  +2 ]
  [ 1 ]                     [ −1  0  +1 ]
  ```
  The `[1,2,1]` smoother is the binomial `[1,1]∗[1,1]` — the discrete Gaussian at
  `σ ≈ 0.71` (normalized `[1,2,1]/4` has variance ½). The differencer factors too:
  `[−1,0,+1] = [1,1]∗[−1,1]`. So **Sobel is a crude, fixed-scale derivative-of-Gaussian**
  built from the two atoms average `[1,1]` and difference `[−1,1]`.

- **Scharr** — `[3,10,3] ⊗ [−1,0,+1]`: same structure, but the smoother is **optimized
  to minimize the angular (rotational) error** of the gradient direction → better
  orientation accuracy on diagonal edges. (`Prewitt [1,1,1] → Sobel [1,2,1] →
  Scharr [3,10,3] → true Gaussian` is a ladder of increasing Gaussian fidelity/isotropy.)

- **Gaussian derivative** — `G'_σ(x)·G_σ(y)`, the *analytic* derivative of the
  Gaussian, at any chosen `σ_d`. It is the gradient because differentiation commutes
  with convolution:
  ```
  d/dx ( G_σ * I ) = ( dG_σ/dx ) * I = G'_σ * I        (one pass: smooth at σ_d AND differentiate)
  G'_σ(x) = −(x/σ²)·G_σ(x)                              (general order via Hermite:
                                                        dⁿ/dxⁿ G = (−1/σ)ⁿ·Heₙ(x/σ)·G_σ)
  ```
  Separable in 2-D exactly like Sobel, but with true Gaussians at a tunable scale.

> **Terminology — three "G-of-Gaussian" filters; do not confuse them.** The gradient
> operator used throughout this document is the **derivative of Gaussian**, `G'_σ` — an
> *odd* kernel whose weights sum to **0** (a *band-pass* first-derivative). It is **not**
> the **Difference of Gaussians** `G_σ1 − G_σ2` (the filter usually abbreviated "DoG"),
> nor the **Laplacian of Gaussian** `∇²G_σ`. Those two are *even*, sum-to-zero,
> second-derivative-like **blob/edge detectors** (in fact `G_σ1 − G_σ2 ≈ ∇²G` via the
> diffusion equation), used for feature detection (Marr–Hildreth edges, SIFT blobs) — and
> neither appears in this pipeline. Only the **plain Gaussian** `G_σ`, whose weights sum
> to **1**, is a *smoother*.

### Comparison — and why the Gaussian derivative

| Property | **Sobel** | **Scharr** | **Gaussian derivative** |
|---|---|---|---|
| Support / scale | 3×3, `σ≈0.71` (fixed) | 3×3, `σ≈0.7` (fixed) | `~6σ+1`, **`σ_d` tunable** |
| Noise suppression | minimal, fixed | minimal, fixed | **tunable** via `σ_d` |
| Angular error | a few ° (worst near diagonals) | **sub-degree** (optimized) | **→0 as σ grows** (well-sampled) |
| Fidelity to ideal `iω` | crude | better at 3×3 | clean band-pass `iω·e^{−σ²ω²/2}` |
| Scale-space / multiscale | no | no | **yes** (unique linear scale-space) |
| Steerable, one kernel family | 1st-deriv only | 1st-deriv only | **yes** (Hermite family) |
| Cost | cheapest | cheap | higher (grows with σ) |
| Best when | teaching | fixed 3×3 on clean hi-res | **noisy/thin, accuracy-critical orientation** |

At **equal 3×3 support, Scharr wins** — it was purpose-built as the optimal 3×3, and a
Gaussian derivative truncated to 3 taps at small σ is a poorly-sampled approximation.
The Gaussian derivative is better for our problem because it **removes the 3×3
constraint**:

1. **Scale knob = matched noise suppression.** Scharr is welded to `σ≈0.7` (almost no
   smoothing); on thin, low-SNR hyphae the dominant error is *noise*, not Scharr's
   sub-degree diagonal bias. `σ_d` lets you match hypha width and push the passband
   below the noise band. You attack the error that actually dominates.
2. **Isotropy for free from scale.** A 3×3's angular error is an *under-sampling*
   artifact; Scharr squeezes the max from 9 fixed coefficients but hits a ceiling. A
   Gaussian derivative at `σ_d ≳ 1.5–2` densely samples an intrinsically isotropic
   kernel, so its angular error falls *below* Scharr's — the isotropy Scharr chases
   comes as a byproduct of the scale you needed anyway.
3. **Principled scale-space.** The Gaussian is the unique kernel giving a linear
   scale-space (no spurious new extrema as σ grows), so "differentiate at scale `σ_d`"
   is well-defined; derivatives are steerable (`∂_θ = cosθ·Ix + sinθ·Iy`) and follow
   one Hermite family across orders and scales.

> **`σ_d`** = the std-dev of the Gaussian inside the derivative-of-Gaussian = the
> *derivative scale* = how wide a neighborhood you estimate slope over ≈ hypha width.
> Too small → noise-dominated gradients; too large → thin hyphae blur out / merge.

For a `σ=1` first-derivative-of-Gaussian the sampled kernel is a smooth, 5-tap odd
cousin of `[−1,0,+1]`: `x=−2..2 → [+0.108, +0.242, 0, −0.242, −0.108]`.

### How the Gaussian derivative is computed (and why it carries smoothing)

**The identity that makes it a gradient.** Convolution and differentiation are both
linear and shift-invariant, and LTI operators commute, so

```
∂/∂x ( G_σ * I ) = ( ∂/∂x G_σ ) * I = G'_σ * I
```

One-line proof: `∂/∂x ∫ G_σ(x−u)·I(u) du = ∫ [∂/∂x G_σ(x−u)]·I(u) du = (G'_σ * I)(x)` — the
derivative moves under the integral onto the smooth, analytically-differentiable *kernel*.
So convolving with `G'_σ` is **exactly "Gaussian-blur at σ, then differentiate."** The
smoothing isn't a side effect; it's the `G_σ` that the derivative is taken *of*.

**The discrete kernel, step by step:**

1. **Truncate.** Radius `R = ⌈t·σ⌉`, `t ≈ 3–4` (scipy default `truncate=4.0`); the kernel
   lives on integer taps `k = −R…R`. Too-tight truncation biases the derivative.
2. **Sample the analytic kernel.** First order: `G'_σ(k) = −(k/σ²)·G_σ(k)`. General order
   is `Gaussian × Hermite`: `dⁿ/dxⁿ G = (−1/σ)ⁿ·Heₙ(k/σ)·G_σ(k)` — one kernel family for
   gradient (`n=1`), Hessian/Laplacian (`n=2`), etc. (`scipy.ndimage.gaussian_filter1d`
   samples the normalized Gaussian and multiplies by the order-`n` polynomial.)
3. **Normalize / calibrate.** Two constraints define a derivative filter:
   - the *smoothing* (order-0) factor is normalized to **sum 1** (unit DC gain);
   - the *derivative* (order-1) factor is antisymmetric so it **sums to 0** (a constant →
     zero) and is scaled to return the *true* slope — it differentiates a unit ramp
     exactly (`Σ_k (−k)·G'_σ(k) = 1`). scipy applies this calibration to the sampled kernel.
4. **Apply separably.** For `Ix`: convolve `G'_σ` along `x`, then the plain `G_σ` along `y`
   (`gaussian_filter(I, σ, order=(0,1))`); swap axes for `Iy`. Borders by reflection.

**Why it inherits smoothing (the honest version).** `G'_σ` is *not itself a smoother* — it
is an odd, sum-to-zero **band-pass** (a derivative). It carries a Gaussian's noise
suppression only because of the factorization above: the data is band-limited by `G_σ`
*before* the derivative sees it. In the frequency domain,

```
Ĝ'_σ(ω) = i·ω · e^{−σ²ω²/2}
          └i·ω┘   └── the plain Gaussian's low-pass response, baked in ──┘
```

The `e^{−σ²ω²/2}` envelope *is* the plain-Gaussian smoother — it rolls the gain back to
zero at high frequencies (where noise lives), so unlike the bare central difference
(`Ĥ(ω) = i·sin ω`, which keeps rising toward Nyquist and passes the noise band) the
Gaussian derivative **cannot amplify high-frequency noise**. Larger `σ_d` closes the
envelope sooner → more smoothing fused into the same one-pass operator; and in 2-D the
*perpendicular* factor `G_σ(y)` is a literal plain-Gaussian smoother averaging along the
edge. That is the whole reason we differentiate with `G'_σ` rather than `[−1,0,+1]`: the
same derivative, with a tunable Gaussian low-pass built in.

---

## ② The structure tensor — gradients → an orientation field

Goal: pool the per-pixel gradient vectors `g=(Ix,Iy)` over a neighborhood into a
*dominant local orientation*. **You cannot average the vectors** — orientation is
mod-180°, so `g` and `−g` (opposite flanks of a ridge) cancel.

**The trick:** average the **outer products** `ggᵀ`. Because `(−g)(−g)ᵀ = ggᵀ`, the
sign ambiguity dies and opposite gradients reinforce. The windowed average (Gaussian at
`σ_i`) is the structure tensor:

```
J = ⟨ g gᵀ ⟩ = ⎡ ⟨Ix²⟩   ⟨IxIy⟩ ⎤        ⟨·⟩ = Gaussian pool at σ_i
                ⎣ ⟨IxIy⟩  ⟨Iy²⟩  ⎦
```

`J` is the (uncentered) **covariance of the local gradient cloud** — its
eigen-decomposition is PCA fitting an ellipse to that cloud (it is also the Harris
corner matrix):

| gradient cloud | `λ₁, λ₂` | meaning |
|---|---|---|
| cigar | `λ₁ ≫ λ₂ ≈ 0` | one clean edge/fiber — strong orientation |
| round | `λ₁ ≈ λ₂ > 0` | corner / **crossing** / texture — no single orientation |
| tiny | `λ₁ ≈ λ₂ ≈ 0` | flat region |

- **Orientation:** `φ = ½·atan2(2Jxy, Jxx−Jyy)` (the `½` and `2Jxy` are the double-angle
  structure of diagonalizing a symmetric 2×2). The fiber runs at `φ + 90°`.
- **Coherence:** `C = (λ₁−λ₂)/(λ₁+λ₂) = √((Jyy−Jxx)² + 4Jxy²)/(Jxx+Jyy) ∈ [0,1]` —
  "how cigar-shaped," i.e. confidence in `φ`.

**Crossings read as orientation-less.** Half-vertical + half-horizontal edges give
`J = diag(½,½)` → `λ₁=λ₂` → `C=0`, even though two strong edges are present — the exact
failure mode that corrupts per-strand turning, and why coherence *flags* it.

The two scales are distinct jobs: `σ_d` measures each gradient cleanly; `σ_i` (must be
> 0 — with no integration `J` is rank-1 and coherence is trivially 1) turns single
gradients into a neighborhood statistic.

---

## ③ Three metrics off the field

- **Coherence** `C` — *local concentration*: is there a dominant orientation here?
  (eigenvalue contrast; per-pixel map ∈ [0,1].)
- **Orientation gradient** `|∇φ|` — *spatial rate of change*: how fast does orientation
  rotate across space? `φ` is π-periodic, so a naive `∇φ` spikes at wraps; use the
  double-angle vector `v=(cos2φ, sin2φ)`:
  ```
  |∇φ| = ½·√( |∇(cos2φ)|² + |∇(sin2φ)|² )        (exact, since |v|=1 ⇒ |∇v| = |∇(2φ)|)
  ```
  Units rad/µm. Only meaningful where `C` is high (where `C→0`, `φ` is noise and `|∇φ|`
  blows up), so report it **coherence-weighted**: `⟨|∇φ|⟩_C = Σ C·|∇φ| / Σ C`.
- **Orientation entropy** `H = −Σ pᵢ·log pᵢ` (normalized `/log N`) — *distributional
  spread* of orientations over a region ∈ [0,1]; discards *where* they are.

---

## ④ Combining them into quantifiable metrics

Three numbers, but only **two independent axes** plus a redundancy:

| | **Concentration** (how aligned) | **Spatial change** (how it turns) |
|---|---|---|
| Coherence | ✅ local (eigenvalue) | — |
| Entropy | ✅ regional (histogram) | — |
| `\|∇φ\|` | — | ✅ (derivative of the field) |

- **Coherence and entropy estimate the *same* quantity** — angular dispersion — just
  local/quadratic vs regional/information-theoretic; at region level `mean(C)` and
  `1 − Hₙₒᵣₘ` track each other → **largely redundant** as reported scalars (their one
  real difference is *scale*: locally coherent everywhere but globally multi-oriented →
  high mean-`C` *and* high entropy).
- **`|∇φ|` is orthogonal** — the derivative axis neither dispersion measure sees. It
  *disambiguates* what entropy conflates: high entropy can mean a fine-scale random mix
  (low `C`, noisy `|∇φ|`) **or** a smoothly *rotating* field (high `C`, organized
  `|∇φ|`) — and for hyphal turning only the second is signal.

**Clean 3 → 2 reduction:**

- **Concentration** = the circular **resultant length**
  `R = |⟨(cos2φ, sin2φ)⟩| ∈ [0,1]`. Since `1−R ≈ circular variance` — exactly what both
  coherence and entropy approximate — **`R` subsumes coherence and entropy** into one
  principled scalar.
- **Turning** = coherence-weighted `⟨|∇φ|⟩`.

**Recommendation:** report the pair **{`R` (alignment), coherence-weighted `⟨|∇φ|⟩`
(turning)}**, using coherence internally as the confidence weight rather than as its own
feature. If a downstream model needs a single number, ship coherence-weighted `⟨|∇φ|⟩`
(the "orientation-turning density"), noting that it drops the alignment axis.

---

## The one-paragraph version

Estimate each pixel's slope with a **central-difference derivative fused with Gaussian
smoothing** — a **Gaussian derivative at `σ_d`** matched to hypha width (Sobel and Scharr
are the fixed-scale 3×3 teaching/optimized versions; the Gaussian one wins by giving a
*tunable scale* that suppresses the noise dominating thin hyphae while gaining isotropy
for free). **Square the gradients into outer products** to escape the ±180° cancellation,
**pool them with a Gaussian at `σ_i`** → the structure tensor, whose ellipse gives
**coherence** (alignment) and `φ` (orientation). From `φ` and `C` derive **`|∇φ|`** (how
the field turns) and unify **coherence + entropy into one concentration scalar `R`**,
leaving a compact, faithful pair **{alignment `R`, coherence-weighted turning `⟨|∇φ|⟩`}**
as the Regime-B, tracing-free readout of hyphal orientation and directional change.

---

## Appendix A — practical implementation

The principled gradient uses a **true Gaussian derivative at a tunable `σ_d`**, then
pools products at `σ_i`:

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def structure_tensor_metrics(I, sigma_d, sigma_i, eps=1e-12):
    # ① gradients: Gaussian-derivative at the derivative scale σ_d
    Ix = gaussian_filter(I, sigma=sigma_d, order=(0, 1))   # ∂/∂x
    Iy = gaussian_filter(I, sigma=sigma_d, order=(1, 0))   # ∂/∂y

    # ② structure tensor: pool the products at the integration scale σ_i
    Jxx = gaussian_filter(Ix * Ix, sigma=sigma_i)
    Jyy = gaussian_filter(Iy * Iy, sigma=sigma_i)
    Jxy = gaussian_filter(Ix * Iy, sigma=sigma_i)

    # eigen-structure (closed form for symmetric 2×2)
    tr   = Jxx + Jyy
    diff = np.hypot(Jxx - Jyy, 2 * Jxy)          # = λ1 − λ2
    coherence = diff / (tr + eps)                # C ∈ [0,1]
    phi = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)   # local orientation (mod π)

    # ③ orientation gradient via the doubled angle (π-periodicity-safe)
    c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
    gcx, gcy = np.gradient(c2)
    gsx, gsy = np.gradient(s2)
    grad_phi = 0.5 * np.sqrt(gcx**2 + gcy**2 + gsx**2 + gsy**2)   # |∇φ|, rad/px

    # ④ compact metrics
    w = coherence
    turning = float((w * grad_phi).sum() / (w.sum() + eps))       # coherence-weighted ⟨|∇φ|⟩
    R = float(np.hypot((w * c2).sum(), (w * s2).sum()) / (w.sum() + eps))  # resultant (alignment)
    return dict(coherence=coherence, phi=phi, grad_phi=grad_phi, turning=turning, R=R)
```

**Note on `skimage`.** `skimage.feature.structure_tensor(image, sigma=…)` uses a *fixed*
small-support derivative (a Sobel/central-difference operator, no scale knob), and its
`sigma` argument is the **integration** scale `σ_i` (the Gaussian weighting on the
products) — *not* a derivative scale. To control `σ_d` there you must pre-smooth the
image or compute the gradients yourself (as above). This is why the recipe builds the
tensor from `scipy.ndimage.gaussian_filter(order=1)` rather than relying on the `sigma`
argument for `σ_d`.

## Appendix B — picking the scales

- `σ_d ≈` hypha half-width to width. Below it, gradients are noise; above it, adjacent
  strands blur together before the tensor is formed.
- `σ_i ≈ 2–4 × σ_d`, matched to the neighborhood over which a single orientation is
  meaningful. It must be > 0 (else `J` is rank-1 and coherence is meaningless), and it
  must be *matched* to the entropy window size, or coherence and entropy compare
  concentration at different scales and the coherence↔entropy redundancy breaks.
