# Drift register

Every place these specs depart from a validated reference, with the reason, the evidence, and the
status. **Faithfulness to validated reference logic beats a convenient shortcut.** A deviation is
admissible only if it is (a) forced, (b) recorded here, and (c) tested.

Four categories:

- **FORCED** — no admissible alternative exists. Prove it.
- **CONTRACT** — required by a PhenoTypic invariant (e.g. `detect_mat ∈ [0,1]`).
- **CAPABILITY** — new behaviour we chose to add. Must be opt-in or reachable-and-tested alongside the
  faithful configuration.
- **DEFECT** — a shortcut. Remove it.

---

## `FocusEdgeMonogenicPhase` (a port)

| # | Deviation | Reference | Category | Status |
|---|---|---|---|---|
| M1 | `acos` argument clamped to `[-1,1]` | none of the three references clamps | FORCED | Roundoff can push `energy/(sumAn+ε)` above 1 ⇒ `NaN`. Paired with a test asserting the clamp **never fires** on the three shipped plates, so it is provably inert. |
| M2 | `orientation` / `feature_type` mapped from radians to `[0,1]` | `phasecongmono` returns radians | CONTRACT | `detect_mat ∈ [0,1]`. The map is stated in the docstring and is invertible. |
| M3 | `FocusEdgePhase` tightened to `n_scale ≥ 2` | our own shipped code, not a reference | CAPABILITY | Fixes an all-zero output at `n_scale=1` (`_focus_edge_phase.py:320` divides by `n_scale − 1`; measured `max=0` versus `max=0.971004` at `n_scale=4` on `load_synth_yeast_plate`, with a `RuntimeWarning`). Kovesi's `phasecong3.m` divides by `nscale-1` too and is equally undefined there; we raise where he would warn and emit zeros. **Breaking**, and it fires a guardrail: `tests/fixtures/tune/back_compat_pipelines/enhance_features_edges.json` serializes `"n_scale": 1`, and that corpus deliberately parks each fixture at the tightest legal-today edge so a narrowing trips it (measured: `1 failed, 14 passed`). Decided 2026-07-09 (user) to narrow now and move the fixture to the new edge; `test_n_scale_one_is_rejected` replaces the guardrail we disarmed. There is no `CHANGELOG` in this repository — the test is the record. |
| M4 | `fft2`, not `perfft2` | **the references disagree** | *not a deviation — a recorded choice* | MATLAB `phasecongmono.m` l.156 uses `perfft2`; Julia `phasecongruency.jl` l.446 uses `fft` and says so (`# Use fft rather than perfft2`). We ship the Julia branch. Material: `pc` moves up to `0.67` absolute, `0.13` even 16 px inside the border. The golden fixture pins the **`perfft2`** branch, since that is what `phasepack` computes; the branches differ only in which spectrum enters an identical chain. Kernel takes `periodic: bool = False`; **the operation does not expose it.** `verify_claims.py::check_18`. |
| M5 | `T = max(…, ε)` | `phasepack` only; neither MATLAB nor Julia floors `T` | CAPABILITY | A free guard on the degenerate case. Inactive on every non-constant image — the smallest `T` across the five fixture images is `3.7e-3`, 37× the floor; only a literally constant image reaches it. Kept because the fixture encodes it and it costs nothing. |
| M6 | `rayleigh_mode` anchors histogram bins at `0` and retains zeros | our own shipped `_focus_edge_phase.py:512`, which does neither | **CORRECTED to the reference** | Kovesi (`phasecongmono.m:465`): `edges = 0:mx/nbins:mx; n = histc(data, edges)`. `phasepack` (`tools.py:86`) also retains zeros. Our port dropped zeros and let `np.histogram` place edges at `data.min()` — an **undeclared** deviation from all three, shifting `T` by `0.0130%` on `load_synth_yeast_plate`. Neither form is measurably more accurate on synthetic Rayleigh samples (relerr `0.0922` vs `0.0938` at `σ=1`); the reason is faithfulness. **This changes shipped `FocusEdgePhase` output at `noise_method = -2`.** Golden fixture cannot cover it (its `_params` record `noiseMethod = -1`), so the value is pinned by a dedicated test — the mutation audit showed a doubled `rayleigh_mode` was otherwise invisible to every test. |
| M7 | `noise_method` outside `{-1, -2} ∪ [0, ∞)` raises `ValueError` | no reference validates it | CAPABILITY | `-1.5` matched neither branch, left `tau = 0.0`, and silently returned `T = ε`. Kovesi's MATLAB errors on the undefined `tau`. The dispatch compares against `epsilon`, following **Julia only** (`phasecongruency.jl:512`, `abs(noisemethod + 1) < epsilon`). **Correction (2026-07-09):** an earlier version of this row also cited `phasecongmono.m:224` for the epsilon compare. That is wrong — `phasecongmono.m:225` is an exact `if noiseMethod == -1`. MATLAB and Julia genuinely differ here; we follow Julia. Same species as S7: a claim about a reference made without opening it. |
| M8 | `riesz_multiplier` divides **componentwise**: `(fx/radius)*1j - (fy/radius)` | `phasepack:156` — the generator of our golden fixture — computes the numpy form `(1j*fx - fy)/radius` | **CORRECTED to the reference** | The three source texts print the same glyphs, but the languages differ beneath them. MATLAB's `./` and Julia's `/(z::Complex, x::Real)` (`base/complex.jl:348` → `Complex(real(z)/x, imag(z)/x)`) do a **true division per component**. numpy promotes the real denominator and runs `nc_quot`; with a zero imaginary part that branch is `scl = 1/r` followed by a **multiply**. Verified bit-exactly: `(1j*a - b)/r == (a*(1/r))*1j - (b*(1/r))` on 200 000 samples, differing from componentwise division on **42.8%** of them by up to **1.41 ulp**. Settled by executing `frequencyfilt.jl:238` in Julia and comparing raw IEEE-754 bit patterns: componentwise is **bit-identical**, the numpy form is not. **Cost:** the fixture is phasepack's, so `check_19`'s agreement loosens `3.52e-14 → 5.32e-14` — still **7.27 orders** inside `rtol=1e-6`. Accuracy was never in question; provenance was. |

**Nothing else.** The congruency formula, `ε = 1e-4`, `k = 3.0`, `deviation_gain = 1.5`, the geometric
noise factor `(1/mult)^s` — including its known inaccuracy at the anchor scale — and Kovesi's Rayleigh
constant with its measured `1.048×` bias are all reproduced verbatim. **A port has a reference; use
it.** The golden fixture enforces this at `rtol=1e-6`.

**Where the references disagree (M4), a port has no reference.** The deliverable is then a *recorded
decision*, not a fix. Exactly **one** such fork exists: the periodic FFT (M4).

The odd-size frequency-grid difference is **not** a fork. Kovesi's Julia (`frequencyfilt.jl` l.73) and
his MATLAB (`filtergrid.m` l.49) both divide an odd axis by `N`; `phasepack` divides by `N−1`, in both
its `filtergrid` and its `lowpassfilter`. That is a **`phasepack` bug**. We match both Kovesi
implementations — a stronger position than an earlier revision of this register claimed — and the
fixture is generated at even sizes, so it cannot inherit the bug. `references.md` §10.3.

### Corrected before shipping

| Was | Should be | How it was caught |
|---|---|---|
| `ε = 1e-5`, "matching `_focus_edge_phase.py`" | `ε = 1e-4` | `1e-5` is `phasecong3`'s. `phasecongmono` uses `1e-4` in Julia (line 441), MATLAB (line 153), and `phasepack` (line 129). `ε` guards the `acos` argument. |
| congruency "accumulates over scales exactly as in `_phasecong3`" | the `acos`/`deviation_gain` form | All three references agree verbatim, and it is not `phasecong3`'s formula. |
| cited to Wang Lijuan et al., CCDC 2014 | Kovesi's `phasecongmono` | We never read CCDC. The docstring must not imply otherwise. |

---

## `FocusEdgeColorPhase` (fusion) and the gated conformal lift (a derivation)

| # | Deviation | Reference | Category | Status |
|---|---|---|---|---|
| ~~C1~~ | ~~Poisson scale `s₀` instead of the scale-free `hᵢ`~~ | — | **WITHDRAWN** | The divergence is a divergence of the **DC response**, not the operator. With P1 (`g̃(0) = 0`) the integrand is `O(1)dr` and converges to `0.500000`. **No `s₀`.** The scale-free route is faithful and available. |
| C1a | `Q³(x) = Σ[f(x+y) − f(x)]·Q̃³(y)` — test-point value removal, `z` channel only | no source states it | **FORCED (P1)** | On `S²`, `v₃ = \|v\|² ≥ 0`, so `h₃` becomes even and single-signed. JMIV Eq. (89) omits the resulting DC term `g(0)·μ₃`. `Q̃¹`, `Q̃²` have zero mass ⇒ untouched ⇒ the common factor `T` survives. Also makes the scale-free kernel convergent. |
| C1b | `tan ϕ = (B/A)·√(Q¹²+Q²²)/\|Q³\|` — gain correction | JMIV Eq. (89) assumes `A = B` | **FORCED (P2)** | `M = diag(A,A,B)`; closed forms verified to 6 digits against a lattice sum. `B/A → 0` as `s₀→0`, `→ 4` as `s₀→∞`, `= 1` at exactly `s₀* = 0.19269068`. Both C1a and C1b are necessary: on the oracle, raw `0.6445`, DC-only `0.2657`, gain-only `2.1868`, **both `0.9015`**. |
| C1c | `σ` = pixels per sphere diameter | no source states it | **FORCED (P3)** | `S⁻¹` is not scale-invariant (the `1 + \|x\|²`), so the sphere fixes a unit. A structural parameter, not a curvature knob. |
| C2 | Reads `image.rgb`, not `detect_mat` | every other `FocusEdge` subclass | FORCED | CMPCM is defined on colour; `rgb` is not a supported `detect_mat` layer. Legal under `@validate_operation_integrity`. Recorded in the class docstring, `enhance/CLAUDE.md`, and the ABC. |
| C3 | `l2` output clipped at the write site | paper leaves it in `[0, √3]` | CONTRACT | The internal helper returns the un-clipped value, and the PFOM regression uses that helper. |
| C4 | `output="curvature"` emits `ϕ/(π/2)`, masked by odd energy; **`κ` is not shipped** | JMIV Eq. (97) defines `κ` | **CONTRACT + FORCED** | **`κ` is not curvature.** Three radial profiles with identical isophote curvature `1/r` give `κ·r = 1.256 / 2.077 / 3.136` (`f = r, r², r³`); the estimator tracks `f''/f' + 1/r`, agreeing with curvature only when `f'' = 0` — the cone, which is JMIV's own test signal. On JMIV's oscillatory circular signal it varies `3×` with wavelength at fixed `r`. `κ` **is** scale-covariant; an earlier revision descoped it for the wrong reason ("not scale-free", measured at `R/σ ≈ 0.5`, a mask too small to contain the isophote). `ϕ` is bounded but **undefined where the odd energy vanishes**. The 2019 paper's Eq. (12) never consumes `ϕ` or `κ` either. `verify_claims.py::check_09`. |
| C14 | Corrected `ω₃` sign | JMIV Eqs. (41)/(67)/(89) | FORCED | The lifted isophote satisfies `u₃ = 2⟨u₁₂, m⟩`, so the plane normal is `(2m₁, 2m₂, −1)`, not `(2m₁, 2m₂, +1)`. `\|κ\|` unaffected; `κ`'s sign flips and `θ` rotates by π. |
| C5 | Noise threshold uses the kernel-norm ratio, not `(1/mult)ʲ` | Kovesi's `phasecongmono` | **FORCED, and *more* faithful** | Kovesi states the *principle* ("according to their relative bandwidths") and flags his sum as "a simplistic overestimate" whose correction "will depend on the filter bank being used." `(1/mult)ʲ` is the log-Gabor instantiation. For a planar DoP bank the exact law is `τⱼ/τ₀ = s₀/sⱼ` (closed form, from `‖p_{ts} − p_s‖₂ ∝ 1/s`). **The reason is that this is Kovesi's stated principle — not that some percentage is large.** An earlier revision argued "13–28% is fatal", which proves too much: Kovesi's *own* bank has a 50% per-scale error at the anchor step (`0.7149` vs `0.4762`) and we port it verbatim regardless. |
| C6 | `color_space="lab"` default | paper uses HSV | CAPABILITY | Hue is circular and ill-conditioned at low saturation: at `S = 0.01` an RGB perturbation of `0.00036` swings hue by `0.01`. Bare agar is low-chroma. `("hsv","l2")` remains reachable and is the PFOM-regression configuration. |
| C7 | `fusion="joint"` default | paper uses per-channel + L2 | CAPABILITY | `l2` has no cross-channel interaction, so incoherent chroma can never veto a spurious luminance edge. Unvalidated; §8.4.13 is its acceptance test, and a null result flips the default back to `l2`. |
| C8 | `fusion="coherent"` | no reference | CAPABILITY | Opt-in, never default. Annihilates genuine anti-correlated chromatic edges. |
| C9 | `chroma_weight_1`, `chroma_weight_2` | paper has no weights | CAPABILITY | Defaults `1.0`/`1.0` reproduce the paper exactly. `(1,1,1)` on raw Lab **is** the CIE76 ΔE metric. |
| C10 | `T_total = Σ wᵢTᵢ`, `A_max = Σ wᵢ·maxₖA` in joint mode | no reference | CAPABILITY | Follows from `joint` existing at all. 1-homogeneous, so the scale-invariance argument survives. Untested against anything external. |
| C11 | `ValueError` on achromatic input | no reference | CAPABILITY | `joint` degenerates to a luminance congruency divided by itself. Fail loudly. |
| C12 | Kovesi's spread weight `W` reused | paper mis-cites it to Abdou & Pratt and never validates it for DoP scales | **UNVALIDATED** | A reasonable guess, not a derivation. Flagged; no evidence either way. |
| C13 | Rayleigh median constant `√(log 4)` on a 4-component amplitude | exact only for 2 components | CONSTANT BIAS | Measured `0.811×`. Kovesi states `k` absorbs exactly such a per-bank constant. Inherit, record. |

### Resolved

| # | Question | Answer |
|---|---|---|
| ~~U1~~ | Scale-ladder domain | **Reading A: planar, arithmetic.** Eq. (12) sums `c(0) ∗ p_{s,t×s}(z)` over `s = 1..n`, `z` the image-plane radius, applied to the component *images*. The `n=2` constant and exponent in Eq. (10) are therefore **correct, not a defect** — that entry is withdrawn from the defect table. |
| ~~U2~~ | `κ` fails its oracle | **JMIV Eq. (89) is wrong.** It omits `g(0)·μ₃` and assumes `A = B`. Corrections C1a and C1b recover `κ·r = 1.0139 / 1.0118 / 1.0104` at matched scale. The estimator was *also* unsound (singular where the odd response vanishes); estimate bounded `ϕ`, masked by odd energy. |

### Open — the one that decides whether this operation exists

| # | Question | Evidence |
|---|---|---|
| ~~U3~~ | Is the conformal lift redundant for `pc`? | **YES, resolved.** `f_z` is an **even** channel: `corr(c_bp, f_z_bp) = −0.9679` against `corr(c_bp, f_x_bp) = −0.0000`, and `corr(f_z, ∇²f) = +0.8904`. It is a Laplacian, hence a rescaled copy of `c` within a band. Redundancy holds **at crossings too** (`−0.995`). The lift contributes nothing to the congruency output. `references.md` §9. |
| U4 | **Does `φ` earn the conformal lift as a junction feature?** | Open, and the *only* remaining justification. Prior evidence is against: `φ` is undefined on ridges (odd energy `0.0000` at a line centre); it cannot separate a 90° corner (`0.7997`) from disk boundaries (`0.63 … 0.98`); and it is non-monotone outside `r/σ ≲ 0.35`. The conformal signal models **one circle**; a crossing is **two superimposed lines**, out of model. Three-arm gate in `conformal-lift.md` §4, with Wietzke & Sommer's *Signal Multi-Vector* (JMIV 37:132–150, 2010) as arm C. |

---

## Shortcuts that were removed, and what they cost

Recorded because both looked faithful, and both were expensive.

**D1 — Folding the band-pass into the conformal kernel.** The spec replaced the paper's two steps
(scale-free components, then a *planar* DoP over the four component images) with a single kernel per
scale, differencing the conformal kernels across the conformal scale. Justified by `qⁱₛ = hᵢ ∗ pₛ`,
"identical by linearity."

The identity is real; the inference is invalid. That convolution lives in `R³`; Step 10's band-pass
lives in `R²`; the pullback does not commute across them. Linearity was never the obstacle — *which
space you convolve in* is.

It also deleted the band-pass's one necessary function. `Q̃³` is even and single-signed, so it
responds to whatever DC the signal carries; **the only thing that keeps it honest is that its input
is zero-mean, and the planar DoP is what makes it so.** Two commits were then spent trying to patch
DC back in at the kernel level, where it cannot be patched without breaking `Qⁱ = ωᵢ·T`.

**D2 — Copying Kovesi's `(1/mult)^s` into CMPCM.** It *is* his formula, verbatim, which is exactly why
it reads as faithful. But it is the log-Gabor instantiation of a principle, transplanted onto a filter
bank it was never derived for. See C5.

**S3 — trusting a published equation.** JMIV Eq. (89) (`Qⁱ = ωᵢ·T`) is **wrong**: it omits a DC term
and assumes an isotropy that holds at one point. Weeks of the `κ` failure trace to taking it on faith.
The lesson is not "distrust papers" but: **when an invariant is load-bearing, test the invariant, not
just the thing built on it.** Had we measured `Qⁱ/ωᵢ` directly on a synthetic plane wave, this would
have surfaced in an hour.

**S4 — asserting in prose what was never asserted in code.** Four separate claims in this spec were
measured once, written down, and then drifted from what the number actually showed: the `2.04×` spread
(measured in an invalid regime), the "kernel-norm rule reduces to Kovesi's" (it does not; the cumulative
offset is a persistent `1.551×`), the `f_z` correlation figures (config-dependent), and "both
corrections are necessary" (true, but shown at a point where the two errors cancelled). One of the
tests even **cherry-picked its own slice** to pass. `verify_claims.py` exists so that every number in
these documents is re-derived on demand, and so that a claim which stops being true fails loudly.

**S5 — marking our own homework.** Every check in `verify_claims.py` up to `14` used a test signal this
spec wrote, against a ground truth this spec derived. That is exactly the arrangement under which S3
and S4 happened. Porting Kovesi's four MIT test-image generators fixed it: `circsine` re-convicts `κ`
on the algorithm author's own image (`10.59×`), `step2line` supplied the spec's **first positive
control**, `noiseonf` showed that congruency alone cannot reject `1/f` noise (only the threshold `T`
can), and `starsine` exposed a trap the spec's own §7 had missed — an `h₂` sign flip that mirrors every
orientation and that **no axis-aligned test can detect**, because `0` and `π/2` are their own mirror
images mod `π`. Four generators, four findings, none of which our own fixtures would have produced.
`references.md` §10.2.

**S6 — behavioural controls cannot tell you *which* operator you built.** S5's four generators were
added, all four checks passed, and the work was committed. The very next step — running the newly
installed `phasepack` against the transcription those checks had just blessed — showed `pc` was wrong
by up to **0.67 absolute**, because §2 said "filter the image FFT" and every reference filters the
image's *periodic component* (`perfft2`). The spec had the bug too. Checks 15, 16 and 17 passed with
it present, and would have kept passing forever: `step2line` still sweeps, `noiseonf` still gets
thresholded, `starsine` still recovers its orientations. Kovesi's own images tested that the thing
behaved like *a* phase-congruency operator, not that it was *his*. Only the golden fixture could tell
the difference, and it did so in one run. A second omission — `T = max(…, ε)` — fell out at the same
time. **A behavioural control and a golden fixture answer different questions; neither substitutes for
the other.** `references.md` §10.3.

**S7 — three misattributions, one cause: `phasepack` was read as if it were Kovesi.**

`phasepack` was the only reference that was *installed and runnable*. Every claim about "the
references" got sourced from the file that was easiest to open, then generalised upward. Three times:

| Claimed | Actual | Cost of checking |
|---|---|---|
| "Every reference bandpasses the periodic component." | Kovesi's Julia comments `perfft2` out, l.444. | one `grep` of a file already in the scratchpad |
| "The `max(T, ε)` floor is Kovesi's." | It is `phasepack`'s. Neither `.m` nor `.jl` floors `T`. | one `grep` of `phasecongmono.m` |
| "Kovesi's MATLAB divides an odd frequency axis by `N−1`." | `filtergrid.m` l.49 divides by `cols`. `phasepack` is the outlier, in two functions. | one `curl` of `filtergrid.m` |

The first two were committed **in the same commit that recorded lesson S6**, about not generalising
from one reference. The third was committed in the commit that *retracted* the first two.

The failure is not carelessness; it is that **"the reference" is a category error when three exist.**
A port must name *which* implementation each claim comes from, with a file and a line, before the
claim is written down — and where they disagree, the deliverable is a recorded decision, not a fix.
Note the asymmetry the third error exposes: `phasepack` is the only one with a bug here, and it is the
only one this spec could execute. Runnability and authority are unrelated. `references.md` §10.3.

**The lesson, in one line:** copying a reference's *constants* is not faithfulness. Copying its
*principle*, correctly instantiated, is — and verifying the principle is part of copying it.
Where a reference also publishes its *test data*, take that too: it is the only ground truth in the
loop that this spec cannot have biased. And where it ships runnable code, diff against it — the
things a behavioural test cannot see are exactly the things that make it the *reference*.

---

## Claims retracted during review

Kept visible so nobody re-derives them.

| Claim | Why it was wrong |
|---|---|
| "Joint fusion is the only formulation that stays in `[0,1]`." | `exp(−\|x−1\|/b²) ≤ 1` for every real `x`, so the triangle inequality never entered. Every variant is bounded once `l2` is clipped. |
| The `ΔE = 5` table proves raw CIELAB is the common scale. | Circular. Same array, same linear filter, three times. Would give the same answer in any space. The a-priori CIE76 argument stands alone. |
| Nominal rescaling (`/100, /128, /128`) is a normalization. | It corrupts an already-normalized space, biasing against chroma by `128/100`. |
| "Exactly invariant" / "bit-identical" under a global weight rescale. | `ε` is not 1-homogeneous. ~1% over `c ∈ [0.01, 100]`. |
| The DC fix recovers `κ ∝ 1/r`. | Artifact of unit-mass-normalizing `Q̃³` alone, which breaks the common scale factor `κ` requires. |
| The single-scale construction gives `κ ≈ 1.335/r`. | The `1.335` came from a crude 3×3 DC removal, not from the construction. Single-scale on a cone gives `κ ∝ r`. |
| ~~`s₀` is our shortcut for the paper's `hᵢ`.~~ | **UN-RETRACTED.** It *was* a shortcut. The divergence is of the DC response; with P1 the scale-free `hᵢ` converges. There is no `s₀`. |
| ~~The scale ladder is arithmetic.~~ | **UN-RETRACTED.** Eq. (12) settles it: planar `n=2` DoP, `sⱼ = 1..4`. |
| Eq. (10)'s `n=2` constant/exponent is "the defect that matters". | Not a defect. The band-pass genuinely lives in the plane. |
| The DC fix recovers `κ ∝ 1/r`. | Retracted for the *right* reason (unit-mass-normalizing `Q̃³` alone breaks the common scale factor) — but the **correct** DC fix (P1, value removal) does recover it. |
| `κ ∝ 1/r` is the whole gate. | The gate was never on the edge detector. Contrast `0.995`, affine-invariance bit-identical. Only `κ` was blocked. |
| `E ≤ A_Σ` matters for the `[0,1]` bound. | It does not. But `E/A_Σ ≈ 1` was *also* a folded-construction artifact: the real value is `0.999948` at an edge, `0.0` on flat. |
| `references.md` §6's "80–95% luminance". | Mixed `load_synth_yeast_plate` into one table and `load_yeast_plate` into another. Real plates: 69–80%. |
| "`pc` is invariant to the `fx`/`fy` swap, so nothing else catches it" (§7). | True but incomplete, and the incompleteness was the dangerous half. There is a **second** axis bug — the `−sumh2` sign — and the vertical/horizontal edge pair is blind to *that* one in `pc` **and** `orientation`. Only an off-axis pattern (`starsine`) catches it. |
| "Every reference implementation bandpasses the periodic component." | **Two of three.** MATLAB `phasecongmono.m` l.156 and `phasepack` do; Kovesi's Julia comments `perfft2` out (`IMG = fft(img)   # Use fft rather than perfft2`). Asserted after checking only `phasepack` — S4 repeating itself inside the commit that added S6. We ship the Julia branch and fixture the MATLAB one. §2.0. |
| "The `max(…, ε)` floor is Kovesi's." | It is **`phasepack`'s**. Neither MATLAB nor Julia floors `T`. Inactive on every non-constant image (smallest fixture `T` = `3.7e-3`, 37× the floor). Kept as a free guard, correctly attributed. |
| Implied: the shipped `FocusEdgePhase` might share the `fft2` defect. | It does not. `perfft2` is `phasecongmono`-specific; MATLAB `phasecong3.m` l.146 and Julia `phasecong3` both use plain `fft2`. §3.2's scope guard holds. |
| "Kovesi's Julia, MATLAB and `phasepack` agree verbatim." | False, but not for the reason first given. **Kovesi's two implementations agree** on the frequency grid; `phasepack` is the odd one out, and it is a bug: `filtergrid.py` and `tools.lowpassfilter` both divide an odd axis by `N−1` where Kovesi divides by `N` (`8.2e-3` on a 255² `starsine`, `0.0` at even sizes). The one genuine Julia-vs-MATLAB fork is `perfft2`. |
| "The two reference frequency grids diverge at odd sizes; Kovesi's MATLAB divides by `N−1`." | **Wrong.** `filtergrid.m` l.49 is `[-(cols-1)/2:(cols-1)/2]/cols` — `/cols`, identical to the Julia. `lowpassfilter.m` doesn't build a grid at all; it calls `filtergrid`. The `N−1` came from reading `phasepack`'s `linspace(..., endpoint=True)` and attributing it upstream. **Third misattribution of the same species, and the second inside the commit that recorded the lesson about the first two.** One `curl` of `filtergrid.m` would have caught it. |
