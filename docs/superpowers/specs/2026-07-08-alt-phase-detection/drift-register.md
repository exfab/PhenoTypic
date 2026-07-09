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
| M3 | `FocusEdgePhase` tightened to `n_scale ≥ 2` | our own shipped code, not a reference | CAPABILITY | Fixes a silent all-zero output at `n_scale=1` (`_focus_edge_phase.py:320` divides by `n_scale − 1`). Unrelated to the port; may be split into its own commit. Changelog entry required. |

**Nothing else.** The congruency formula, `ε = 1e-4`, `k = 3.0`, `deviation_gain = 1.5`, the geometric
noise factor `(1/mult)^s` — including its known inaccuracy at the anchor scale — and Kovesi's Rayleigh
constant with its measured `1.048×` bias are all reproduced verbatim. **A port has a reference; use
it.** The golden fixture enforces this at `rtol=1e-6`.

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
| C4 | `output="curvature"` emits `ϕ/(π/2)`, masked by odd energy; **`κ` is not shipped** | JMIV Eq. (97) defines `κ` | **CONTRACT + FORCED** | `κ` is unbounded, singular on ridges, and — measured — **not scale-free**: at fixed `σ` it spreads `2.04×` over `r ∈ [2,16]`. Unbiased only at `σ ∝ r`, which is circular. The plane-wave approximation is valid only for `r/σ ≲ 0.35`. `ϕ` is bounded but **undefined where the odd energy vanishes** (a straight wave's crest reads `0.0000` unmasked). The 2019 paper's Eq. (12) never consumes `ϕ` or `κ` either. |
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

**The lesson, in one line:** copying a reference's *constants* is not faithfulness. Copying its
*principle*, correctly instantiated, is — and verifying the principle is part of copying it.

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
