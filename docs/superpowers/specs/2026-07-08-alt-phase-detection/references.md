# Monogenic PC and CMPCM — recovered source, corrected math, citations, reference implementations

**Date:** 2026-07-08 (revised 2026-07-09 after independent review)
**Status:** shared reference document for [`monogenic-phase-congruency.md`](./monogenic-phase-congruency.md),
[`color-phase-congruency.md`](./color-phase-congruency.md) and [`conformal-lift.md`](./conformal-lift.md).
Deviations are catalogued in [`drift-register.md`](./drift-register.md).

This file exists because the calculations and background behind the "Monogenic / Riesz PC" and
"Colour / multichannel PC (CMPCM)" cards in
`docs/superpowers/artifacts/2026-06-26-fungi-detection-method-tree/index-fieldnotebook.html`
could not be located. They were never lost; they were in the repo under an unexpected name, and they
were **wrong about CMPCM's central mechanism**. Both facts are recorded here so the next reader does
not repeat the search or inherit the error.

**Notation warning.** The CMPCM paper uses `k` for its band-pass index (Eq. 10). Kovesi uses `k` for
the Rayleigh noise-σ multiplier. They are different quantities that happen to share a symbol and a
value of 2. This document writes the paper's band-pass index as **`j`** throughout.

---

## 1. Provenance of the field-notebook cards

| Notebook card | Source in this repo |
|---|---|
| Monogenic / Riesz PC (`CAVEAT`) | `docs/superpowers/reports/filamentous-fungi-detection/breadth-survey.md`, §2.3, line 127 |
| Colour / multichannel PC, CMPCM (`PORT IT`) | same file, §2.3, line 129 |
| Supporting claims for both | `docs/superpowers/reports/filamentous-fungi-detection/claims-all-2289.md`, lines 540–554 |

The notebook's `math` / `how` / `variables` blocks are near-verbatim restatements of those two survey
paragraphs. `claims-verified-66.md` contains **neither** method, so neither survived the survey's own
3-vote adversarial verification. Both are marked `(u)` (unverified) in `recommendations.md` lines 142
and 145.

`docs/superpowers/reports/filamentous-fungi-detection/README.md` explains why the material feels
orphaned: it came from a `deep-research-litmax` run whose synthesis stage never completed, so the
reports were reassembled from recovered claims in follow-up workflows.

---

## 2. Citations

**Monogenic / Riesz phase congruency.**
Wang Lijuan, Zhang Changsheng, Liu Ziyu, Sun Bin, Tian Haiyong. "Image feature detection based on
phase congruency by Monogenic filters." *The 26th Chinese Control and Decision Conference (2014
CCDC)*, IEEE, 2014, pp. 2033–2038. DOI [10.1109/CCDC.2014.6852502](https://doi.org/10.1109/CCDC.2014.6852502).
**Still not read directly.** The author names were originally reconstructed from the CMPCM paper's
reference [13] (surname-first); they are now **confirmed** against IEEE Xplore's own author list
(2026-07-09), which also supplies the two names the reconstruction had abbreviated to "et al."
The full text remains unobtained: IEEE Xplore serves `502` to every non-browser client, and the
paper is not open access. Nothing in `_monogenic_kernels.py` rests on it — the port's oracle is
Kovesi's three implementations plus the golden fixture — but no one has checked this spec's
monogenic-PC prose against its nominal source.

**CMPCM.**
Shi, Meihong; Zhao, Xueqing; Qiao, Dongdong; Xu, Bugao; Li, Chunmei. "Conformal monogenic phase
congruency model-based edge detection in color images." *Multimedia Tools and Applications* **78**,
10701–10716 (2019). DOI [10.1007/s11042-018-6617-x](https://doi.org/10.1007/s11042-018-6617-x).
**Read in full.** Originally read as Springer HTML, in which Tables 1–3 render as images — hence the
long-standing note that we had "the PFOM *ordering* only, not the numbers." **Superseded 2026-07-09:**
the PDF was obtained and carries Table 1 as machine-readable text.

> `Table 1  PFOM performance evaluation of different edge detection algorithms`
> | Method | Canny | Log | VPMM | PC | MPC | CMPCM |
> |---|---|---|---|---|---|---|
> | FOM | 0.8888 | 0.9008 | 0.9934 | 0.9099 | 0.9321 | **0.9989** |

Read off the row, then checked against the paper's own prose, which states the PFOM of Canny is the
minimum and CMPCM the maximum, "followed by VPMM and then MPC, PC and Log." Sorting the numbers
descending gives CMPCM > VPMM > MPC > PC > Log > Canny — the stated ordering exactly. That agreement
is the evidence the numbers came from the right row; without it they would be six digits copied out
of a PDF.

> **These numbers are not reproduction targets, and `color-phase-congruency.md` §7.1 must not be
> rewritten to treat them as such.** They are the PFOM of six operators on the paper's Fig. 4a
> "geometry image", which is `176 × 298` (the PDF states that size exactly once, confirming §7.1's
> claim) and **whose pixels are published nowhere**. Knowing the answer does not give us the input.
> §7.1's design — synthesise a geometric colour image with known ideal edges and regress the
> *ranking* — is unchanged and remains correct. What the table adds is a sharper ordering (the gap
> `CMPCM 0.9989` vs `VPMM 0.9934` is 0.0055, so a ranking regression that resolves them must be
> better than ~0.5% in PFOM) and a record of the claim we are testing.

Tables 2–3 have not yet been transcribed.

**Foundations.**

- Felsberg, M.; Sommer, G. "The Monogenic Signal." *IEEE Trans. Signal Process.* **49**(12),
  3136–3144 (2001).
- Felsberg, M.; Sommer, G. "The Monogenic Scale-Space." *J. Math. Imaging Vis.* **21**(1–2), 5–26 (2004).
- Wietzke, L.; Sommer, G. "The Conformal Monogenic Signal." *DAGM 2008*, LNCS 5096, pp. 527–536.
  DOI [10.1007/978-3-540-69321-5_53](https://doi.org/10.1007/978-3-540-69321-5_53).
- **Fleischmann, O.; Wietzke, L.; Sommer, G. "Image Analysis by Conformal Embedding."**
  *J. Math. Imaging Vis.* **40**(3), 305–325 (2011).
  DOI [10.1007/s10851-011-0263-5](https://doi.org/10.1007/s10851-011-0263-5).
  **This is the paper that makes CMPCM implementable.** CMPCM cites it as [7] and inherits its
  conformal construction without restating the computable form. **Read in full.**
- Kovesi, P. "Image features from phase congruency." *Videre* **1**(3), 1–26 (1999). Source of the
  spread weight `W` and the Rayleigh noise threshold `T`.

> Access note: the Springer PDFs were read through an authenticated institutional session. They are
> copyrighted and **must not be committed to this repository.** Everything needed to implement is
> transcribed below.

---

## 3. What the field notebook and breadth-survey get wrong

Both claim CMPCM computes phase congruency **jointly across RGB channels** in a single
conformal-monogenic frame, and that this suppresses a fringe present in only one channel. The
notebook draws this as "shared ridge ✓ / 1-channel fringe ✗".

The paper does none of that. Its Steps 1–14 (§3.3): input RGB → convert to **HSV** (it explicitly
rejects RGB for inter-component correlation, and rejects L\*a\*b\* only on transform cost) → run the
full CMPCM functional **independently on each of H, S, V** → fuse by L2 norm
`E(x,y) = sqrt(Σᵢ Fᵢ(x,y)²)`.

There is no cross-channel congruency anywhere in the method. An L2 fusion of three independent
per-channel congruencies cannot suppress a single-channel fringe.

Separately, the "chromatic aberration" framing is an extrapolation the survey itself flags. It is
also optically wrong-headed: lateral CA is not a fringe present in one channel and absent from the
others, it is **the same edge misregistered between channels**. `recommendations.md` line 145 already
reaches the correct conclusion — "CA must be corrected upstream" — and this spec does not claim
otherwise.

**Follow-up task (out of scope):** correct the `index-fieldnotebook.html` CMPCM card and
`breadth-survey.md` line 129.

---

## 4. CMPCM as published, with transcription defects resolved

The paper contains **six** defects. Implement the corrected forms below, not the printed equations.

| Paper | Defect | Corrected |
|---|---|---|
| Eq. (5) | sphere written `(z − 1/1)² = 1/4` | `(z − 1/2)² = 1/4` |
| ~~Eq. (10)~~ | ~~is the `n=2` Poisson kernel while the conformal space needs `n=3`~~ | **NOT a defect.** Eq. (12) sums `c(0) ∗ p_{s,t×s}(z)` over `s = 1..n` with **planar** `z`. The band-pass genuinely lives in the image plane, where `n=2` is correct. Retracted. |
| Eq. (10) | defines `p_z` as *already* a difference, then Eq. (11) differences it again | the primitive is the plain Poisson kernel (JMIV Eq. 17) |
| Eq. (12) | subtracts `T/(A_Σ+ε)` with no non-negativity clamp | wrap in `max(·, 0)` |
| Eq. (12) | cites Abdou & Pratt [1] for the spread weight `W` | `W` is Kovesi's construction |
| Step 13 | `if s ≤ 3 then go to Step 5`; the loop variable is `i` | `if i ≤ 3` |

Everything else checks out: the fixed parameters (`t=1.5, n=4, S_max=n, λ=0.5, j=2, b=0.3`), the
HSV/per-channel/L2 story, and the PFOM ordering (§5.1 prose:
`CMPCM > VPMM > MPC > PC > LOG > Canny`).

**The scale ladder is RESOLVED: planar, arithmetic.** Eq. (12) defines
`c_Σ = Σ_{s=1}^{n} c(0) ∗ p_{s,t×s}(z)` and likewise for `f_x, f_y, f_z` — a convolution with
`p_{s,t×s}`, a function of the **image-plane** radius `z = √(x²+y²)`, applied to the four
already-computed component *images*. Steps 9–11 iterate `s = 1,2,3,4` as scale **values** consumed by
Eq. (11), `p_{s,t×s} = p_{t×s} − p_s`, `t = 1.5`. So: planar `n=2` DoP, arithmetic ladder. The
orphaned `λ = 0.5, k = 2` in Eq. (10) contradict Eq. (11)'s `t` and are part of Eq. (10)'s corruption.

### 4.1 Intrinsic dimension (Eq. 1)

`i0D` constant, `i1D` linear (lines, edges), `i2D` everything else (curves, corners, crossings).
Plain PC and monogenic PC handle `i1D` only; the conformal lift is what buys `i2D`.

### 4.2 The conformal monogenic signal (Eqs. 2–9)

Stereographic projection onto the sphere of centre `(0,0,½)` and **radius ½** (not the unit sphere):

```
S⁻¹(x, y) = (x, y, x²+y²) / (1 + x² + y²)          # plane → sphere   (JMIV Eq. 31)
S(u)      = (u₁, u₂) / (1 − u₃)                     # sphere → plane   (JMIV Eq. 32)
```

The signal is embedded **with respect to each test point**: for pixel `x` the sphere's south pole is
translated onto `x`, and all operators are evaluated at that south pole (JMIV Eq. 63). The conformal
monogenic signal is the 4-vector `f_CMS = (c(0), f_x(0), f_y(0), f_z(0))`, with

```
A = √(c² + f_x² + f_y² + f_z²)                    # local amplitude          (paper Eq. 9)
φ = atan2( √(f_x² + f_y² + f_z²), c )             # local phase              (paper Eq. 9)
θ = atan2( f_y, f_x )                             # local orientation        (paper Eq. 9)
ϕ = arctan( √(f_x² + f_y²) / f_z )                # i1D/i2D angle; → π/2 for i1D
κ = 2·f_z / √(f_x² + f_y²)                        # isophote curvature       (JMIV Eq. 97 ONLY)
```

> `κ` is **not** in the paper's Eq. (9). It is JMIV Eq. (97), `κ = 2/tan(ϕ)`, which follows from
> JMIV Corollary 4's `ϕₘ = arctan(2rₘ)` and `κ = 1/rₘ`. The factor is `2/tan`, not `2·tan`.

### 4.3 The computable form — derived, not printed anywhere

The paper's Eq. (7) defines `f_x, f_y, f_z` through a 3D Radon transform and its inverse. That is a
definition, not a recipe. Composing it with the JMIV Poisson kernel (Eq. 17), conjugate Poisson
kernels (Eq. 20), the projection (Eq. 31) and the embedding (Eq. 63) collapses to a plain 2D
**correlation** with four fixed masks per scale.

For `n = 3`: `pₛ(v) = (1/π²)·s/(|v|²+s²)²` and `qⁱₛ(v) = (1/π²)·vᵢ/(|v|²+s²)²`.

Pull them back through `S⁻¹` and weight by the sphere's surface measure. With `r² = x² + y²`:

```
λ  = 1 / (1 + r²)                # conformal factor of S⁻¹
u  = (xλ, yλ, r²λ)               # the lifted point S⁻¹(x,y)
ρ² = r²λ = r² / (1 + r²)         # |S⁻¹(x,y)|²  — note ρ² < 1 always, and ρ² = u₃
J  = λ²                          # surface-measure pullback

P̃ₛ(x,y)  = (1/π²) ·  s  / (ρ² + s²)² · J
Q̃ⁱₛ(x,y) = (1/π²) · uᵢ / (ρ² + s²)² · J        i = 1,2,3
```

**Verified** (numerically, and analytically by an independent reviewer): the induced area element of
`S⁻¹` is exactly `λ²`. The often-quoted `4/(1+r²)²` is the **unit**-sphere factor; JMIV's sphere has
radius ½ and halving the radius divides area by 4. The Gram determinant gives `E = G = λ²`, `F = 0`.
Also verified: `ρ² = r²/(1+r²)`, the `1/π²` constant (`Γ[2]/π²`, `n=3`), the `(ρ²+s²)²` exponent, and
all four mask symmetries — `Q̃¹` odd in `x`, `Q̃²` odd in `y`, `Q̃³` even, `P̃` even and positive.

Because `ρ² < 1` always, the kernel tail decays as `r⁻⁴` (from `J`) and truncates cleanly.

> **Correlation, not convolution.** The derivation yields `∫ f(x+y)·M(y) dy`. Writing it as `∗` and
> calling `scipy.signal.convolve` flips the kernel and negates the odd masks. `A`, `E`, `pc` and `κ`
> magnitude are unaffected; `θ` rotates by π and `κ` flips sign. This is the sign error observed in
> `conformal-lift.md` §5.

### 4.4 Band-pass and the DC leak

**`Q̃¹` and `Q̃²` are odd**, so their coefficients cancel and they are DC-free (zero-mean)
automatically. **`P̃` and `Q̃³` are even and single-signed** (`u₃ = r²λ ≥ 0`), so they respond to a
constant image. Measured mask sums at `s = 0.5`, `mask_radius = 5`:

```
sum(P̃ ) = +9.435933     sum(Q̃¹) = 0     sum(Q̃²) = 0     sum(Q̃³) = +1.728485
```

Differencing across conformal scales does **not** cancel this, because the pulled-back Poisson
kernel's mass depends on `s` (`∫_{S²} pₛ dσ = 1/(πs(1+s²))`). Measured at `s=1.0, t=1.5`:

```
sum(DoP P̃) = +1.124607        sum(DoP Q̃³) = +0.339491
```

Consequences, both measured:

- `E/A_Σ ≈ 1` everywhere, because numerator and denominator are dominated by the same constant:
  `0.99032` at an edge versus `0.98984` on bare agar. No discriminative power.
- `κ = 2Q³/√(Q¹²+Q²²)` measures image brightness over edge strength, not geometry — hence the wrong
  sign and `κ ∝ r` instead of `1/r`.
- (historic) the `f → αf + β` invariance appeared to fail on the `β` half under the folded construction; it does not.

**Why the paper does not have this problem.** Its Eq. (10) is the **planar 2-D** Poisson kernel, and
the 2-D Poisson kernel integrates to exactly 1 for every `s` (it is a probability density). So its
difference `p_{t·s} − p_s` is exactly zero-mean. The paper's DoP is a *planar bandpass applied to the
four components*, not a difference of sphere-pulled-back kernels. Swapping to the correct `n = 3`
kernel for the conformal space silently inherits a leak the paper never had.

**Neither candidate fix is admissible. This is unresolved.**

*Candidate A — unit-mass-normalize `P̃ₛ` and `Q̃³ₛ` before differencing* (the Difference-of-Gaussians
trick). It does make the difference exactly zero-mean on the lattice. But it rescales `Q³` by an
arbitrary, `mask_radius`-dependent constant while leaving `Q̃¹`/`Q̃²` untouched, and
`κ = 2Q³/√(Q¹²+Q²²)` is valid **only** when all three share one scale factor (JMIV Eq. 89,
`Qⁱ = ωᵢ·T`). It manufactures a `1/r`-looking trend and destroys the proportionality `κ` needs. An
earlier revision of this document adopted it; that was wrong.

*Candidate B — band-pass the signal with a planar 2-D DoP, leaving the kernels untouched.* Preserves
the common scale factor, and the composite masks are DC-free to `1e-17` (since
`sum(A ⋆ B) = sum(A)·sum(B)`). But `Q̃³` still has nonzero mass, so it responds to the *local mean* of
the band-passed signal while `Q̃¹`/`Q̃²` respond to its gradient. `κ` converges cleanly in
`mask_radius` (four significant figures over `R = 8 … 32`) to a **constant ≈ 20**, not to `1/r`.

All eight variants `{n=2, n=3} × {with J, without J} × {raw, band-passed}` were enumerated at
`mask_radius = 16`, respecting JMIV §3.9's precondition that the mask contain the isophote. **None
reproduces `κ = 1/r`.** Magnitudes run 40–500× high; the diagnostic `κ(2)/κ(6)`, which must be
`3.000`, ranges over `0.49 … 2.20`. `n=2` versus `n=3` changes almost nothing; `J` changes a lot.

A structural element is therefore missing from §4.3, and it is not the kernel dimension, the
Jacobian, or the bandpass placement. The failing invariant is JMIV's `Qⁱ = ωᵢ·T`: our three odd
components are not proportional to a common `T`. That proportionality is what Theorem 2 and
Corollary 4 establish, and it is where the investigation must start (§7, item 1).

### 4.3.1 The one-sided support — the root cause of everything

**On this sphere `|v|² = v₃`, so the support is one-sided in `v₃` (`v₃ ≥ 0` always).** The `R³` Riesz
kernel `h₃` is *odd* in `v₃`, and that oddness is what makes `H₃` a quadrature operator. Restricted to
`S²`, it becomes an **even, single-signed** operator. Every symptom follows: the DC leak, `Q̃³` even,
`E/A ≈ 1` under the folded construction, and `κ` 40–500× high.

**The exact linear-response law.** For a signal `g` on `S²` whose lift approximates a plane wave with
unit normal `ω`, with `M = ∫_{S²} v vᵀ w(v₃) dσ = diag(A, A, B)` and `μ₃ = ∫_{S²} v₃ w dσ`:

```
Q¹ = T·A·ω₁          Q² = T·A·ω₂          Q³ = g(0)·μ₃ + T·B·ω₃
```

**JMIV Eq. (89) says `Qⁱ = ωᵢ·T`. It is wrong on both counts**: it omits the DC term `g(0)·μ₃`, and it
assumes `A = B`.

Closed forms follow from Archimedes' hat-box (`∫_{S²}F(v₃)dσ = π∫₀¹F(h)dh`, total area `π` for a
radius-½ sphere) and the identity `v₁² + v₂² = v₃(1 − v₃)`. With `a = s₀²`:

```
A = (π/2)(I₁ − I₂)      B = π·I₂
I₁ = ln((1+a)/a) + a/(1+a) − 1          I₂ = 1 − 2a·ln((1+a)/a) − a²/(1+a) + a
```

Verified against a direct lattice sum to six digits. `B/A → 0` as `s₀ → 0`, `→ 4` as `s₀ → ∞`, and
`B/A = 1` at exactly **`s₀* = 0.19269068`** (full sphere). So the isotropy Eq. (89) assumes holds at a
single point and nowhere else.

### 4.3.2 Two forced corrections

**(P1) Subtract the test-point value, in the `z` channel only.**

```
Q³(x) = Σ_y [ f(x+y) − f(x) ] · Q̃³(y)
```

`Q̃¹`, `Q̃²` have zero mass (odd), so they are untouched and the common factor `T` survives. This
cancels `g(0)·μ₃` exactly.

**It also removes the need for `s₀`.** With `g̃(0) = 0` and `g̃` Lipschitz, `|g̃| ≤ C·r` while
`M₃ ~ 1/r²`, so the integrand is `≤ 2πC·r·r⁻²·r dr = O(1)dr` and **converges**. Verified: the raw
integral over `ε < r < 1` diverges (`1.685 → 9.826` as `ε: 1e-1 → 1e-6`), while the value-removed
integral converges to `0.500000`.

> An earlier revision of this document claimed `s₀` was **intrinsic** and that the scale-free `hᵢ`
> route was impossible. **Both claims are retracted.** The divergence is a divergence of the *DC
> response*, not of the operator. `s₀` is a regulariser of a term that (P1) deletes.

**(P2) Correct the gain.** Since `Qⁱ = T·(Aω₁, Aω₂, Bω₃)`:

```
tan ϕ = (B/A) · √(Q¹² + Q²²) / |Q³|            κ = (2/σ) · (A/B) · |Q³| / √(Q¹² + Q²²)
```

`A` and `B` are pure functions of the mask geometry, not of the data.

**Both corrections are necessary.** Measured on the cone `f = |x|` (target `κ·r = 1`): raw `0.6445`;
DC-removed only `0.2657`; gain only `2.1868`; **both `0.9015`**. Tuned (`s₀ = s₀*`, `σ = 4r`, `R = 4r`)
`κ·r = 1.0139, 1.0118, 1.0104`.

### 4.3.3 (P3) A length scale, and a validity bound

`S⁻¹` is not scale-invariant (the `1 + |x|²`), so the sphere fixes a unit. Introduce **`σ` = pixels per
sphere diameter**, and lift `ŷ = y/σ`. No source states this.

**`κ` is scale-covariant but is not curvature.** `κ·r` depends only on `(r/σ, R/σ)` — verified to 4
digits by scaling `(r,σ,R)` together. An earlier revision claimed `κ` "is not scale-free", citing a
`2.04×` spread; that was measured at `R/σ ≈ 0.5`, a mask too small to contain the isophote.

The genuine defect: three radial profiles with **identical** isophote curvature `1/r` give
`κ·r = 1.256 / 2.077 / 3.136` for `f = r, r², r³`. Isophote curvature predicts `1,1,1`;
`Laplacian/|∇f| = f''/f' + 1/r` predicts `1,2,3`. The estimator tracks the latter, agreeing with
curvature only when `f'' = 0` — the cone, which is JMIV's own Fig. 13 test signal. And on JMIV's
oscillatory circular signal, where Eq. (89) implies the radial frequency cancels, `κ·r` varies **3×**
with wavelength at fixed `r`.

Reproduce: `verify_claims.py::check_09`.

### 4.4.1 The fold — a shortcut this document introduced, and must not

An earlier revision replaced the paper's two-step structure with a single pulled-back kernel per
scale, differencing the **conformal** kernels across the conformal scale `s`:

```
f_x,ⱼ  = ( Q̃¹_{sⱼ} − Q̃¹_{t·sⱼ} ) ⋆ channel        # WRONG — do not implement
```

It justified this by `qⁱₛ = hᵢ ∗ pₛ`, claiming the band-pass could be folded into the kernel
"identically, by linearity."

**The identity is real; the inference is invalid.** `qⁱₛ = hᵢ ∗ pₛ` is a convolution in **`R³`** — it
smooths across the sphere's ambient space. The paper's Step 10 band-pass is a convolution in
**`R²`** — it smooths across the image plane. The pullback does not commute across the two:
`pullback(hᵢ ∗_{R³} pₛ) ≠ pₛ ⋆_{R²} pullback(hᵢ)`. Linearity was never the obstacle; *which space you
convolve in* is.

The fold also conflates two distinct scale spaces. **JMIV's** scale is the `R³` Poisson scale `s`,
used to define a *single-scale* conformal signal (Eq. 88) — that is the object `κ` is derived from.
**CMPCM's** scale is a *planar* DoP ladder applied to scale-free components; CMPCM never computes
`κ` at all. Testing `κ` against a construction carrying CMPCM's ladder tests the wrong object.

Worst of all, the fold removed the one operation guaranteeing a DC-free input. **`Q̃³` is even and
single-signed; the only thing that makes it meaningful is that the signal reaching it is zero-mean,
and Step 10's planar band-pass is what makes it so.** That is the band-pass's necessary function.
Having folded it away, no kernel-level patch can restore it without breaking `Qⁱ = ωᵢ·T` (§4.4).

### 4.4.2 The construction (settled)

Per channel `f`, offsets `|y| ≤ R`, `ŷ = y/σ`, `λ = 1/(1+|ŷ|²)`, `u = S⁻¹(ŷ)`, `w = 1/u₃²` (or
`1/(u₃+s₀²)²` if a regulariser is retained), `J = λ²`:

```
Step A (components, one correlation each)
    c (x) = Σ_y  f(x+y)             · P̃(y)
    Qⁱ(x) = Σ_y  f(x+y)             · Q̃ⁱ(y)                 i = 1, 2
    Q³(x) = Σ_y [f(x+y) − f(x)]     · Q̃³(y)                 ← (P1), forced

Step B (scale space, PLANAR n=2 DoP over the four component images)
    bⱼ = p_{t·sⱼ} − p_{sⱼ}    on the lattice, each p unit-mass-normalized ⇒ bⱼ exactly zero-mean
    sⱼ = 1, 2, 3, 4     t = 1.5                             ← settled by Eq. (12)
    (cⱼ, f_x,ⱼ, f_y,ⱼ, f_z,ⱼ) = bⱼ ⋆ (c, Q¹, Q², Q³)
    Aⱼ = ‖(cⱼ, f_x,ⱼ, f_y,ⱼ, f_z,ⱼ)‖
```

DC-free by construction: `mass(bⱼ) = 0` and `mass(A ⋆ B) = mass(A)·mass(B)`. The **same** band-pass
hits all four components, so the common factor `T` survives. Measured composite mask sums: `≤ 1e-14`.

**This works.** On a step edge: `E/A_Σ = 0.995352` at the edge, `0.000000` on flat background
(contrast `0.995`), and `f → 3f + 7` leaves both values bit-identical. The edge detector was never
broken; the earlier "no discriminative power" and "the `β` half of the invariance fails" were
artifacts of the fold (§4.4.1).

### 4.4.3 Noise threshold `T` — Kovesi's principle, correctly instantiated

Kovesi's `phasecongmono` computes

```
totalTau = tau · (1 − (1/mult)^nscale)/(1 − 1/mult)
T        = totalTau·√(π/2) + k·totalTau·√((4−π)/2)
```

He did **not** recommend the geometric sum as a general rule. His comment states the principle —
"the response of the larger scale filters to noise can then be estimated from the smallest scale
filter response **according to their relative bandwidths**" — and separately flags the summation as
"a **simplistic overestimate**, however these two quantities should be related by some constant that
**will depend on the filter bank being used**. Appropriate tuning of the parameter `k` will allow you
to produce the desired output (though the value of `k` seems to be **not at all critical**)."

`(1/mult)^s` is the log-Gabor *instantiation* of that principle, and it works because log-Gabor
scales are geometric by construction. **Two distinct errors follow from copying it.**

**Error 1 — a constant bias, which `k` absorbs.** `τ = median(A)/√(log 4)` is the exact median of a
Rayleigh, i.e. of a *two*-component amplitude. Measured by pushing white noise through the real
filters:

| bank | components | Kovesi's `τ` ÷ true `τ` |
|---|---|---|
| `phasecong3` | 2 | **0.998×** (exact) |
| `phasecongmono` | 3 | 1.048× |
| conformal | 4 | 0.811× |

Constant per bank, so `k` soaks it up — precisely the constant Kovesi describes. **Inherit it.**

**Error 2 — a per-scale error, which `k` cannot absorb.** For white noise of variance `σ²`, a linear
filter with kernel `M` yields response variance `σ²‖M‖₂²`, so

```
τⱼ / τ₀  =  √( Σᵢ ‖Mᵢ,ⱼ‖₂²  /  Σᵢ ‖Mᵢ,₀‖₂² )
```

That *is* "according to their relative bandwidths," made computable.

> **It does NOT reduce to Kovesi's factor.** An earlier revision claimed it did. The *consecutive*
> norm ratios converge to `1/mult` from `j = 3` (`0.4763, 0.4762, 0.4762`), but the *cumulative*
> `τⱼ/τ₀` is a persistent **`1.551×`** Kovesi's `(1/mult)ʲ`, because the two already disagree at the
> anchor step (`0.7149` vs `0.4762`). `totalTau` therefore differs by a constant `1.25×` — which is
> exactly the sort of per-bank constant Kovesi says `k` absorbs.
>
> So the *recommendation* (use the kernel-norm rule for a new bank) survives, and its *justification*
> does not. Use it because it is Kovesi's stated principle, not because it reproduces his numbers.
> Reproduce: `verify_claims.py::check_14`.

It also exposes a wrinkle in his own code: the `s=0 → s=1` ratio is `0.715`, not `0.476`. **The
lowpass is not the cause** — with the lowpass removed the ratio is still `0.589`. The finest log-Gabor
simply is not band-limited on the lattice: ~30% of scale-0's energy sits above `|u| = 0.45` and ~15%
above Nyquist. His extrapolation is least accurate at the very scale he anchors `τ` on. Not worth
fixing in a port; worth knowing.

For a planar DoP bank the ratio has a **closed form**. Since `‖p_a‖₂² = 1/(8πa²)` and
`⟨p_a,p_b⟩ = 1/(2π(a+b)²)` (Poisson semigroup), `‖p_{ts} − p_s‖₂ ∝ 1/s`, hence

```
τⱼ / τ₀  =  s₀ / sⱼ          exactly, in the continuum
```

For `sⱼ = 1,2,3,4` that is `1, 0.5, 0.333, 0.25`, against Kovesi's `(1/1.5)ʲ = 1, 0.667, 0.444, 0.296`.

> An earlier revision printed `0.478266 / 0.332875 / 0.256580` here. **Those numbers were wrong** — a
> buggy composite-mask padding. The conclusion is unchanged; the arithmetic was not. On a lattice with
> unit-mass normalization the measured ratios are `0.4372 / 0.2924 / 0.2202`.

**And the argument in the earlier revision proved too much.** It claimed a 13–28% per-scale error was
fatal. But Kovesi's *own* bank has a **50%** per-scale error at the anchor step (`0.7149` measured vs
`0.4762` nominal), and we nevertheless port it verbatim. The correct reason to use the kernel-norm
rule for a new bank is that **it is Kovesi's stated principle**, not that some percentage is large.

**Do not estimate `τⱼ` empirically per scale.** Kovesi anchors on the smallest scale *because* "the
smallest scale filters spend most of their time responding to noise, and only occasionally responding
to features." A coarse-scale median is contaminated by real structure, inflating `T` and suppressing
the faint hyphae we are trying to detect.

**Adopted:** anchor `τ₀` at the finest scale exactly as Kovesi does, then propagate with the
kernel-norm ratio. `FocusEdgeMonogenicPhase` keeps the geometric factor **verbatim** — it is a port,
and the golden fixture must match.

### 4.5 The congruency measure (Eq. 12, corrected)

```
A_Σ = Σⱼ Aⱼ
E   = ‖ Σⱼ ( cⱼ, f_x,ⱼ, f_y,ⱼ, f_z,ⱼ ) ‖            # coherent sum over scales
F   = max( W · ( exp( −|E/(A_Σ+ε) − 1| / b² ) − T/(A_Σ+ε) ),  0 )
```

with `b = 0.3`. This is **not** Kovesi's phase-deviation numerator. It is a soft peak at
`E/A_Σ = 1`, and the paper's Fig. 3 argues it is what suppresses the "aftershock" double-response
that plain MPC exhibits at adjacent edge points.

Two properties worth stating because a previous draft of the design got them backwards:

- `exp(−|x−1|/b²) ≤ 1` for **every** real `x`. Combined with `W ∈ (0,1)` and `T/(A+ε) ≥ 0`, `F ≤ 1`
  unconditionally. The triangle inequality `E ≤ A_Σ` is **not** what bounds `F`.
- The functional is **non-monotonic about `x = 1`**: `x = 0.90` and `x = 1.10` both give `0.329193`.
  What `E ≤ A_Σ` buys is that the operating point stays on the increasing branch, so a
  "super-congruent" pixel is never penalised.

`W` (Kovesi's spread weight) and `T` (Kovesi's Rayleigh threshold) are carried over unchanged from
the log-Gabor formulation.

### 4.6 Fixed parameters (Step 4)

`n = 4` scales, `t = 1.5`, `λ = 0.5`, `j = 2`, `b = 0.3`, `S_max = n`.

**The ladder's form is UNRESOLVED** (§4.4.2). Steps 9–11 read `s = 1; … s = s + 1; if s ≤ S_max`,
which is arithmetic *if* `s` is the scale value, and geometric *if* `s` is a loop index whose value
is supplied by Eq. (10)'s `s·λʲ` with `λ = 0.5`. The degeneracy bound (§4.3.1) and `cms.m` both
favour the index reading; Eq. (10)'s planar notation favours the other. Do not freeze it.

### 4.7 Colour handling (Steps 2, 3, 14)

RGB → HSV; independent CMPCM on H, S, V; fuse `E = sqrt(Σᵢ Fᵢ²)`.

---

## 5. Monogenic / Riesz PC — the actual formula

Replace the oriented log-Gabor bank with an **isotropic** log-Gabor radial bandpass times the 2D
Riesz transform, whose frequency response is `i·(u,v)/|(u,v)|`. Per scale the monogenic triple
`(f, h1, h2)` gives `A = √(f² + h1² + h2²)` analytically, with **no orientation sweep**.

The congruency measure is **not** `phasecong3`'s. All three reference implementations (§8.1) agree
verbatim:

```
sumAn  = Σₛ Aₛ                    maxAn = maxₛ Aₛ
energy = √( (Σₛf_s)² + (Σₛh1_s)² + (Σₛh2_s)² )
width  = (sumAn/(maxAn + ε) − 1)/(nscale − 1)
weight = 1/(1 + exp((cutoff − width)·g))
tau    = median(sumAn)/√(log 4)          # at the FIRST scale only
T      = totalTau·√(π/2) + k·totalTau·√((4−π)/2),   totalTau = tau·(1 − (1/mult)^nscale)/(1 − 1/mult)

PC = weight · max(1 − deviationGain·acos(energy/(sumAn + ε)), 0) · max(energy − T, 0)/(energy + ε)
or = atan(−sumh2/sumh1)
ft = atan2(sumf, √(sumh1² + sumh2²))
```

Defaults: `deviationGain = 1.5` ("sensible values are from 1 to about 2"), `k = 3.0` in Kovesi's own
Julia and MATLAB (`phasepack` uses `2.0`), `nscale = 4`, `minWaveLength = 3`, `mult = 2.1`,
`sigmaOnf = 0.55`, `cutoff = 0.5`, `g = 10`.

**`ε = 1e-4`, not `1e-5`.** `phasecongmono` uses `1e-4` in all three references (Julia line 441,
MATLAB line 153, `phasepack` line 129). `1e-5` belongs to `phasecong3` (Julia line 1272), which is
why our `FocusEdgePhase` correctly uses it. This is not cosmetic: `ε` sits inside
`acos(energy/(sumAn + ε))` and is the only thing keeping the argument strictly below 1.

**No reference clamps the `acos` argument.** Mathematically `energy ≤ sumAn` (triangle inequality on
the scale sum), so with `ε > 0` the ratio is `< 1`. Roundoff can still push it above 1, giving `NaN`.
We clamp to `[-1, 1]` and treat that as a documented, numerically-necessary deviation, paired with a
test asserting the clamp never activates on real images — so it is provably a no-op, not a silent
correctness patch.

**Attribution.** The field-notebook card and `breadth-survey.md` cite Wang Lijuan et al., CCDC 2014.
**We have not read that paper.** What `FocusEdgeMonogenicPhase` implements is Kovesi's
`phasecongmono`. The docstring must say so; the CCDC formula may differ.

The noise threshold enters as a **multiplicative fraction** `max(E−T,0)/(E+ε)`, not subtracted from
the numerator. Kovesi's comment: subtracting it early "would interfere with the phase deviation
computation."

The Riesz multiplier is built as `H = (i·fx − fy)/f` (`packedmonogenicfilters`), packing both odd
channels into one complex array so a single inverse FFT yields `h1` in the real part and `h2` in the
imaginary part. Our existing `_construct_filter_grids` already returns `sintheta = fx/freq` and
`costheta = fy/freq`, so `H = 1j*sintheta − costheta` needs no new grid maths. The lowpass
`1/(1 + (w/0.45)^30)` is identical to ours.

---

## 6. Measured evidence for the channel prior

All figures use the conformal DoP masks of §4.4 (`s = 1.0`, `t = 1.5`, `mask_radius = 5`), raw Lab.

### 6.1 Per-channel shares of a shared joint denominator

| plate | `L*` | `a*` | `b*` |
|---|---|---|---|
| `load_synth_yeast_plate` (synthetic) | 94.0% | 2.6% | 3.4% |
| `load_yeast_plate` (Rhodotorula, **real**) | **69.2%** | 3.7% | 27.1% |
| `load_fungi_plate` (Neurospora, **real**) | 80.4% | 1.3% | 18.3% |

> A previous draft quoted "80–95% luminance", derived by accidentally mixing `load_synth_yeast_plate`
> into one table and `load_yeast_plate` into another. On real plates the band is **69–80%**.

Because raw Lab is the ΔE metric (`color-phase-congruency.md` §4.1), this dominance is **not a units artifact**. It
is a physical fact in perceptual units.

Alternative scalings, for the record — `std` is pathological because `std(L*)` is inflated *by the
colonies*, so dividing by it cancels the signal you want:

| scaling | synth_yeast (L\*, a\*, b\*) |
|---|---|
| raw Lab (= ΔE) | 94.0%, 2.6%, 3.4% |
| fixed nominal (`/100, /128, /128`) | 95.2%, 2.1%, 2.7% |
| per-image MAD | 84.7%, 2.5%, 12.8% |
| per-image std | **22.8%**, 5.3%, **71.9%** |

A data-dependent gain also makes the operation something other than a pure function of its fields,
breaking the `to_json`/`from_json` reproducibility contract in spirit.

### 6.2 Per-axis amplitudes and parity weights

Mean conformal-DoP amplitude, raw Lab (directly comparable, ΔE units). "Parity weight" = the weight
at which that channel contributes as much to `A_total` as `L*` does.

| sample | `L*` | `a*` | `b*` | `b*` parity | `a*` parity |
|---|---|---|---|---|---|
| Rhodotorula (`load_yeast_plate`) | 4.2867 | 0.2316 | 1.6754 | **2.6** | 19 |
| Neurospora (`load_fungi_plate`) | 5.0589 | 0.0830 | 1.1538 | **4.4** | 61 |
| `load_synth_yeast_plate` | — | — | — | **27.8** | — |

This is the derivation of `TuneSpec(0.0, 8.0)`. The synthetic plate needs `b*` parity 27.8 because
its `b*` is near-flat (`std = 0.158`); it is **not representative**, and it is the doctest plate.

`b*` dominates `a*` on both real samples (7.2× and 13.9×), in the same direction. Rhodotorula is a
red-pigmented yeast, but its carotenoids still load blue-yellow, not red-green. Because a weight is
multiplicative, a shared chroma weight preserves that ratio and can never promote the weaker axis.

### 6.3 HSV hue is ill-conditioned at low saturation

Within a hue sector, `|ΔRGB| = 6·V·S·Δh` exactly. So the table below is an identity, not evidence —
its values correspond to `V = 0.6` and a hue step of `0.01`:

| saturation | \|ΔRGB\| |
|---|---|
| 0.90 | 0.03240 |
| 0.05 | 0.00180 |
| 0.01 | 0.00036 |

Read backwards: at `S = 0.01`, an RGB perturbation of `0.00036` swings hue by a full `0.01`. Bare agar
is low-chroma, so sensor noise there produces enormous hue excursions. **No scalar weight fixes this —
the pathology is conditioning, not gain.** Hue is also circular, so linear bandpass filtering of it is
invalid across the `0/1` seam.

The paper's BSDS500 naturals are saturated, so this never bit them. It is the primary reason
`color-phase-congruency.md` defaults `color_space="lab"` and confines `hsv` to the `("hsv", "l2")` ranking regression.

### 6.4 The weight vector has two degrees of freedom, not three

`E_total`, `A_total`, `T_total`, `A_max` are all 1-homogeneous in `w`, so the ratio `E_total/A_total`
is invariant under a global rescale. **`ε` is not homogeneous**, so the fused output is *not*
bit-identical: over `c ∈ [0.01, 100]` it moves by ~1%, and by ~7% once `c` reaches `1e-4`. Invariance
holds only to `O(ε/A_total)`.

Luminance is therefore pinned at `1.0` at negligible cost, and the two chromatic axes carry the two
real degrees of freedom as the scalar (tunable) fields `chroma_weight_1`, `chroma_weight_2`.

| `color_space` | pinned at `1.0` | `chroma_weight_1` | `chroma_weight_2` |
|---|---|---|---|
| `lab` | `L*` | `a*` | `b*` |
| `hsv` | `V` | `H` | `S` |

---

## 7. Open questions the implementation must close

1. **The scale-ladder domain (§4.4.2). BLOCKING, and the spike's FIRST question.** Reading A (planar
   DoP) versus Reading B (geometric conformal ladder, as `cms.m`). Three evidences are recorded in
   §4.4.2. JMIV's `κ` oracle decides. Every other conformal question is downstream of this one.

2. **`κ` does not reproduce JMIV's analytic ground truth, and the cause is unknown. BLOCKING.**
   None of the eight structural variants works (§4.4). The failing invariant is JMIV Eq. (89),
   `Qⁱ = ωᵢ·T` — our three odd components are not proportional to a common scalar `T`, so the ratio
   `Q³/√(Q¹²+Q²²)` is not `cot(ϕ)` and `κ = 2/tan(ϕ)` does not follow.

   Where to start: JMIV Theorem 2 (the embedded signal `g^{x,s}` approximates a plane wave in `R³`)
   and Corollary 4 (which *derives* `Qⁱ = ωᵢ·T`). Theorem 2 carries preconditions we have not checked
   — notably JMIV §3.9's requirement that the mask be large enough for the isophote through the test
   point to project entirely onto `S²`. Then Wietzke & Sommer, DAGM 2008, and if needed Wietzke &
   Sommer, "The Signal Multi-Vector," *JMIV* **37**, 132–150 (2010).

   Specific suspicions, none verified:
   - `Q̃³`'s tail decays as `r⁻⁴` while `Q̃¹`/`Q̃²` decay as `r⁻⁵`, so `Q̃³` is disproportionately
     sensitive to mask truncation and cannot share a common `T` on a finite lattice.
   - The embedding may require per-test-point mean removal that neither paper states.

3. **The `κ` estimator itself is unsound; fix it before trusting any number.**
   `κ = 2Q³/√(Q¹²+Q²²)` is singular wherever the gradient vanishes. On JMIV's own oscillatory signal
   a ring median blends valid and singular points, giving `κ = −18.7, +0.59, +36.0, −27.9` across
   rings — that is measurement error, not method error. Mask by odd-energy magnitude, and respect
   JMIV §3.9's precondition that `mask_radius` exceed the isophote radius. **Some fraction of every
   `κ` number recorded in this document is contaminated by this.**

4. **Convergence in `s₀`.** `s₀` is intrinsic (§4.3.1) but its value is free, bounded above by
   `s₀ ≲ 1.5`. Results must be shown to stabilize as `s₀` shrinks; §4.3.1's log-divergence sets the
   floor.

5. **Fig. 4a's pixels.** The paper's Table 1 PFOM is computed on its Fig. 4a "geometry image"
   (176×298), which is published nowhere. Its §2 pixel spec (173×299, uniform 128 with three flat ±1
   stripes) describes Fig. 1a1, a colour-model demo with no geometry. These are **two different
   images**, not two reports of one. `color-phase-congruency.md` §7.1 therefore builds its own geometry image and
   asserts only the ranking.
6. **`mask_radius` and the `s₀` bound.** `s₀ ≲ 1.5` follows from the degeneracy measurement in
   §4.3.1 and is evidence-backed, not a tuned value. `mask_radius` must additionally be large enough
   to contain the isophote under test (JMIV §3.9).

---

## 8. Reference implementations

### 8.1 Monogenic PC — three, and they agree verbatim

| Implementation | Language | Status |
|---|---|---|
| [ImagePhaseCongruency.jl](https://github.com/peterkovesi/ImagePhaseCongruency.jl) `phasecongmono` | Julia | maintained, by Kovesi |
| [`phasecongmono.m`](https://www.peterkovesi.com/matlabfns/PhaseCongruency/phasecongmono.m) | MATLAB | canonical |
| [phasepack](https://github.com/alimuldal/phasepack) `phasecongmono.py` | Python | unmaintained since 2016, MIT |

`phasepack` is MIT-licensed and is the natural cross-check for the golden fixture in `monogenic-phase-congruency.md` §7.
Secondary: [CPBridge/monogenic](https://github.com/CPBridge/monogenic) (C++/OpenCV + Python bindings)
and [pinga-lab/paper-monogenic-signal](https://github.com/pinga-lab/paper-monogenic-signal)
(`Code/monogenic.py`, Poisson scale-space monogenic in Python).

### 8.2 Conformal monogenic — one, and it must not be copied

[Vivianyuwei/Image-Edge-Detection-Based-on-Conformal-Phase](https://github.com/Vivianyuwei/Image-Edge-Detection-Based-on-Conformal-Phase)
— MATLAB, pushed July 2018, 9 stars, **no license file** (therefore all rights reserved).

Read it to cross-check geometry; **do not copy it.** Our implementation derives from the papers.

What it confirms: `cms.m` builds exactly our `u = S⁻¹(y)`, our `uvw = ρ²`, and our `(s² + ρ²)²`
denominator. Independent corroboration of the projection, of `ρ²`, and of the exponent.

What it gets wrong or does differently:

- **Omits the surface-measure factor `J = λ²`** and uses the constant `1/(2π)` (the `n=2` constant)
  with the `n=3` exponent. It gets away with `J` only because it uses a 3×3 mask.
- It is **not the CMPCM paper**. It belongs to a precursor grayscale paper by the same Xi'an
  Polytechnic group. There is no colour, no HSV, no L2 fusion, and Eq. (12)'s
  `exp(−|E/A−1|/b²)` appears nowhere.
- `cmpc.m`'s final line is `edge = mat2gray(energy) .* phasecong3(image,...)` — it multiplies the
  conformal energy by *Kovesi's log-Gabor* congruency.
- It `mat2gray`-normalizes each channel per scale (data-dependent), mixes a log-Gabor even channel
  with Poisson odd channels, has its entire noise-threshold block commented out, drops `rz` from the
  final energy, and ignores its own `nscale` argument (`for k=1:3` is hardcoded).
- It handles the §4.4 DC leak with an ad-hoc 3×3 mean-removal high-pass applied to the `P` and `Q³`
  convolutions but **not** to `Q¹`/`Q²`.

Treat it as evidence about geometry, and as a warning about everything else.


---

## 9. What the conformal `f_z` channel actually is

Measured on band-passed structured signals, with the corrected construction:

```
|corr(c_bp, f_x_bp)| = 0.0008      proper quadrature: even vs odd, uncorrelated
|corr(c_bp, f_z_bp)| = 0.8928      f_z is EVEN, not odd  (1175x larger)
|corr(f_z, ∇²f)|     = 0.8994      f_z is, to leading order, a LAPLACIAN
```

> The magnitude of `|corr(c, f_z)|` is configuration-dependent — `0.65` at `s₀ = 0.5`, up to `0.92` at
> `s₀*` — so the *structure* is the claim, not the number. `f_z` is three orders of magnitude more
> correlated with `c` than `f_x` is. Reproduce: `verify_claims.py::check_10`.

**Mechanism.** `Q̃³` is a positive, radially symmetric kernel, so `f ⋆ Q̃³` is a smoothing. The P1
value-removal makes it `(blur − identity)·f`, a Laplacian. Within a band the Laplacian acts as
multiplication by `−|ω|²`, roughly constant, so `f_z_bp` is nearly a scalar multiple of `c_bp`.

**Consequences.**

1. **The conformal lift contributes nothing to the congruency output.** `f_z` adds a rescaled copy of
   `c` to both `E` and `A_Σ`. The redundancy holds at crossings too (`corr = −0.995` there), so it is
   not an artifact of averaging over uninteresting pixels.
2. **`φ = atan2(‖(f_x,f_y,f_z)‖, c)` is not a phase.** It folds an even channel into the odd
   magnitude. The i2D content survives only in the ratio `f_z/√(f_x²+f_y²)` — smoothed Laplacian over
   gradient magnitude, i.e. the classical isophote-curvature estimator. JMIV's "curvature without any
   derivatives" is a derivative ratio with the derivatives hidden in the kernel. That is why it needs a
   length scale and is not scale-free.

**`φ` pathologies**, measured (`σ = 8`, single-point, unmasked, synthetic):

- Undefined on ridges. At the centre of a bright line the odd response is `0.0000`; `φ` is `0/0`.
- Cannot separate a corner from a curve: a 90° corner reads `0.7997`, inside the `0.63 … 0.98` spread
  that disk boundaries of radius `3 … 6` already produce.
- Non-monotone outside `r/σ ≲ 0.35`: a disk of radius 6 reads `0.9804`, *higher* than a straight edge's
  `0.8585`.

**The junction problem is a different paper.** The conformal monogenic signal models **one circle**. A
crossing is **two superimposed lines** — out of model. For that, see Wietzke, L.; Sommer, G. "The
Signal Multi-Vector." *J. Math. Imaging Vis.* **37**, 132–150 (2010), which models superimposed i1D
signals.

---

## 10. The validation landscape: nobody numerically tests this algorithm

Surveyed 2026-07-09, across every public implementation of monogenic phase congruency:

| Implementation | Licence | Test suite |
|---|---|---|
| [ImagePhaseCongruency.jl](https://github.com/peterkovesi/ImagePhaseCongruency.jl) (Kovesi) | MIT | `test/test_phasecongruency.jl` — **0 assertions.** Header: *"Hard to test these functions other than visually. This script simply runs then all to make sure that they at least run."* |
| [phasepack](https://github.com/alimuldal/phasepack) | MIT (in-file; no `LICENSE`) | none |
| [CPBridge/monogenic](https://github.com/CPBridge/monogenic) | — | `example/python/monogenicImageTest.py` — **0 assertions**, a demo |
| [CPBridge/monogenic_signal_matlab](https://github.com/CPBridge/monogenic_signal_matlab) | — | none |
| [pinga-lab/paper-monogenic-signal](https://github.com/pinga-lab/paper-monogenic-signal) | — | none |
| [Vivianyuwei/…-Conformal-Phase](https://github.com/Vivianyuwei/Image-Edge-Detection-Based-on-Conformal-Phase) | **none** | none |

**Consequence.** The golden fixture proposed in `monogenic-phase-congruency.md` §7 — numeric agreement
with `phasepack` at `rtol=1e-6` — would be **stronger validation than the reference implementation
itself carries.** That raises the fixture's value and lowers the confidence one may place in "it
matches the reference" as an argument. It also explains how errors like JMIV Eq. (89) survive: nothing
downstream would catch them.

### 10.1 What IS reusable: Kovesi's synthetic test images (MIT)

> **Adopted 2026-07-09.** All four are ported into `verify_claims.py` (MIT notice retained) and
> back checks `09b`, `15`, `16`, `17`. What each one bought is recorded in §10.2.

`src/syntheticimages.jl` exports four generators, all under MIT and portable with attribution. They
supply ground-truth controls this spec currently lacks:

| Generator | What it guarantees | Use for |
|---|---|---|
| `step2line(sze; ampexponent, phasecycles)` | A **phase-congruent** image whose feature type sweeps step → line down the page, at constant congruency | Positive control: `pc` must stay high and roughly constant while `feature_type` sweeps `0 → π/2`. We have **no** positive control today. |
| `circsine(sze; wavelength, ampexponent, offset)` | A phase-congruent **concentric** sine grating | An independent `κ` oracle, written by the algorithm's own author. `ampexponent` varies the radial frequency content — precisely the axis on which `κ` fails (§4.3.3). |
| `starsine(sze; ncycles)` | A phase-congruent radial grating | Orientation and angular-response checks |
| `noiseonf(sze, p)` | `1/f^p` noise, no congruent structure | Negative control for the Rayleigh noise threshold `T`, which is currently untested |

`ampexponent = -1` yields step features; `-2` with `offset = π/2` yields line features. That is the
one knob that separates "is this a step or a line" from "is this congruent", and it is exactly what a
`feature_type` test needs.

**Recommended adoption.** Port the four generators into a test helper (MIT header retained), then use
them for: the monogenic port's behavioural tests (positive + negative controls, alongside the golden
fixture); a three-channel congruent/incongruent pair for `color-phase-congruency.md` §7's fusion
sanity; and `circsine` as a second, author-provided oracle in `verify_claims.py::check_09`.

### 10.2 What the port actually bought

Done. Three faithfulness details bite during the port, all recorded in `verify_claims.py`: only **odd**
harmonics are summed; Julia's `[f(x,y) for x=l:u, y=l:u]` puts `x` on the **first** axis (so
`theta = atan2(Y, X)` with `X` the row coordinate); and `circsine`'s `trim` flag multiplies by
`(r < c) + (r >= c) ≡ 1`, a no-op, so it is not ported.

| Check | Generator | Result |
|---|---|---|
| `09b` | `circsine` | **Confirms `κ` is not curvature, on the author's own image.** At fixed geometry (`r=40`, `σ=16`, `R=48`) and a fixed feature type — the probe sits at `r = m·λ/2`, a simultaneous zero crossing of every odd harmonic — the isophote is an exact circle of curvature `1/40` for every setting. `κ·r` nonetheless ranges `0.351–0.597` across radial waveforms (1.70×) and `0.056–0.597` across wavelengths (**10.59×**). Curvature predicts `1.0000`; not one of the eight settings comes within 30%. |
| `15` | `step2line` | **The spec's first positive control.** `pc` at the congruency column varies `1.47×` (step-row vs line-row endpoint ratio `1.077`) while `feature_type` sweeps `−4.3° → +86.9°` monotonically. Gradient magnitude *localises* the feature in 100% of rows but its **value** collapses `18.1×` (endpoint ratio `0.055`), so any fixed threshold on it drops the line rows. The `acos` argument never leaves `[−1,1]` (drift-register M1). |
| `16` | `noiseonf` | **The Rayleigh threshold `T` is load-bearing, and `T` alone.** With `T` disabled, `1/f` noise reaches `pc₀.₉₉₉ = 0.72–0.76` against `0.954` for a congruent image — a 1.3× margin, useless. `T` is estimated from the image's own amplitude median, so noise gets a `T` **6–61×** larger than `step2line`'s: it cuts noise `1.4–2.6×` and the signal `1.013×`. Honest caveat: `T` does not drive noise to zero (`0.29–0.41` survives). `k` counts standard deviations; it does not guarantee. |
| `17` | `starsine` | **Pins the orientation convention and exposes a trap §7 missed.** Recovered orientation matches the generator's own `theta` field to `0.98°` median / `6.04°` at the 90th percentile. |

**New trap, found by `starsine` (§7 of `monogenic-phase-congruency.md` must be amended).** There are
*two* axis bugs, not one, and the axis-aligned edge pair catches only the first:

| Bug | Vertical/horizontal edge pair sees | `starsine` sees |
|---|---|---|
| `fx`/`fy` swapped | `pc` identical to `1.5e-17` (blind); orientation `0° ↔ 90°` — **caught** | orientation shifts `44.2°` median |
| `atan2(+h₂, h₁)` instead of `−h₂` | `0°` and `90°` are their own mirror images mod `π` — **completely blind, `pc` and orientation both** | orientation shifts `45.8°` median |

The `−h₂` sign encodes a **y-up** convention: `starsine`'s recovered orientation equals `+theta`, not
`−theta`. Flipping it reflects every orientation about the x-axis, which no axis-aligned test can see.
A test suite built only from horizontal and vertical step edges would ship that bug.

### 10.3 The golden fixture, and what only it could catch

`phasepack` 1.5 was installed once (user-approved), used to generate
`tests/fixtures/phasecongmono_golden.npz`, and removed. Agreement: `max|Δpc| = 3.5e-14` across five
64×64 images — `log10(1e-6 / 3.52e-14) = 7.45`, so **7.5 orders** inside the `rtol=1e-6` target, and
`max|Δorientation| = 0°` exactly. `verify_claims.py::check_19`.

It immediately paid for itself by exposing **three things this spec had asserted without checking**,
none of which any behavioural control detects.

### The `perfft2` fork — the references disagree

| Source | Spectrum |
|---|---|
| MATLAB `phasecongmono.m` l.156 | `IM = perfft2(im);` |
| `phasepack` (ports the MATLAB) | `perfft2` |
| **Julia `phasecongruency.jl` l.444–446** | `# (IMG,) = perfft2(img)` … `IMG = fft(img)   # Use fft rather than perfft2` |

Neither source explains the change. The gap is material: `pc` differs by up to **0.67 absolute**, and
`0.13` even 16 px inside every edge, because `fft2` tiles the image and leaks the border discontinuity
across the whole spectrum.

**Decision:** ship the Julia branch (`fft2`), fixture the MATLAB one (`perfft2`). The two differ only
in which spectrum enters an otherwise identical chain, so the fixture still validates the machinery.
`perfft2` is `phasecongmono`-specific — MATLAB `phasecong3.m` l.146 and Julia `phasecong3` both use
plain `fft2`, so the shipped `FocusEdgePhase` is unaffected.

### Two claims that were simply wrong

| Claim as written | Truth |
|---|---|
| "Every reference bandpasses the periodic component." | Two of three. Kovesi's own Julia opts out, explicitly. |
| "`T = max(…, ε)` is Kovesi's floor." | It is **`phasepack`'s**. Neither MATLAB nor Julia floors `T`. It is inactive on every non-constant image — the smallest fixture `T` is `3.7e-3`, 37× the floor. Kept as a free guard, now attributed. |

### And, at odd sizes only, a `phasepack` bug — *not* a Kovesi divergence

An earlier revision of this section claimed Kovesi's two implementations disagree on the frequency
grid. They do not. Both divide an odd axis by `N`:

| Source | Odd axis |
|---|---|
| Julia `frequencyfilt.jl` l.73 | `(-(cols-1)/2:(cols-1)/2)/cols` |
| MATLAB `filtergrid.m` l.49 | `[-(cols-1)/2:(cols-1)/2]/cols` |
| **`phasepack` `filtergrid.py`** | `linspace(-0.5, 0.5, cols, endpoint=True)` ⇒ step `1/(cols−1)` |
| **`phasepack` `tools.lowpassfilter`** | `arange(-(cols-1)/2, (cols-1)/2+1) / (cols-1)` |

`k/N` is the true DFT bin frequency. `phasepack` is simply wrong here, in **two** places, and only at
odd sizes; Kovesi's MATLAB `lowpassfilter.m` doesn't even build its own grid — it calls `filtergrid`.

Consequence: our `construct_filter_grids` matches **both** Kovesi implementations, which is a stronger
claim than the one this document previously made. All three agree at even sizes (`0.0`), and
`phasepack` differs by `8.2e-3` on a 255² `starsine` — which is why the golden fixture is generated at
even sizes only and cannot inherit the bug.

> This was the **third** misattribution of the same species in this spec, and the second inside the
> commit that recorded the lesson about the first two. See `drift-register.md` S7. It was found by
> `curl`-ing `filtergrid.m` and reading line 49 — the file had been one fetch away the whole time.

**The lesson.** Behavioural controls answer "does it behave like a phase-congruency operator?" A golden
fixture answers "is it *this* operator?" `step2line`, `noiseonf` and `starsine` all said yes to the
first while the second was unanswered. Both kinds of test are necessary; neither substitutes for the
other. And note the converse, from §10: `phasepack` carries no tests at all, so the fixture pins
*transcription*, not *correctness* — checks `09b` and `15`–`18` are what speak to correctness. The
fixture told us where the references disagree; it cannot tell us which is right.
