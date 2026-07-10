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
| M3 | `FocusEdgePhase` tightened to `n_scale ≥ 2` | our own shipped code, not a reference | CAPABILITY | Fixes an all-zero output at `n_scale=1` (the `n_scale − 1` divisor — extracted in cluster B to `_monogenic_kernels.py`'s `spread_weight`, formerly inline at `_focus_edge_phase.py:320` — makes `width` diverge and `weight` collapse to `0`; measured `max=0` versus `max=0.971004` at `n_scale=4` on `load_synth_yeast_plate`, with a `RuntimeWarning`). Kovesi's `phasecong3.m` divides by `nscale-1` too and is equally undefined there; we raise where he would warn and emit zeros. **Breaking**, and it fires a guardrail: `tests/fixtures/tune/back_compat_pipelines/enhance_features_edges.json` serializes `"n_scale": 1`, and that corpus deliberately parks each fixture at the tightest legal-today edge so a narrowing trips it (measured: `1 failed, 14 passed`). Decided 2026-07-09 (user) to narrow now and move the fixture to the new edge; `test_n_scale_one_is_rejected` replaces the guardrail we disarmed. There is no `CHANGELOG` in this repository — the test is the record. |
| M4 | `fft2`, not `perfft2` | **the references disagree** | *not a deviation — a recorded choice* | MATLAB `phasecongmono.m` l.156 uses `perfft2`; Julia `phasecongruency.jl` l.446 uses `fft` and says so (`# Use fft rather than perfft2`). We ship the Julia branch. Material: `pc` moves up to `0.67` absolute, `0.13` even 16 px inside the border. The golden fixture pins the **`perfft2`** branch, since that is what `phasepack` computes; the branches differ only in which spectrum enters an identical chain. Kernel takes `periodic: bool = False`; **the operation does not expose it.** `verify_claims.py::check_18` measures the fork, but note it exercises that file's own parallel implementation, **not** `src/` — it never imports `phenotypic`, by design (project file rule). The shipped default is pinned by `test_monogenic_kernels.py::TestThePeriodicDefaultIsPinned`. |
| M5 | `T = max(…, ε)` | `phasepack` only; neither MATLAB nor Julia floors `T` | CAPABILITY | A free guard on the degenerate case. Inactive on every non-constant image — the smallest `T` across the five fixture images is `3.7e-3`, 37× the floor; only a literally constant image reaches it. Kept because the fixture encodes it and it costs nothing. |
| M6 | `rayleigh_mode` anchors histogram bins at `data.min()` and retains zeros | **the references disagree** | *not a deviation — a recorded choice*, plus one genuine correction | **Rewritten 2026-07-09 after the B+C review; the previous revision of this row was wrong twice over.** It claimed the shipped code was "corrected to the reference" and that min-anchoring was "an undeclared deviation from all three." Both false. The references split: **Julia** `phasecongruency.jl:648-652` calls Images.jl `build_histogram(X, nbins)`, whose `partition_interval(nbins, minval, maxval)` is `range(minval, …)` with `minval = minimum_finite(img)` — **min-anchored**. **`phasepack`** `tools.py:86` is `np.histogram(data, nbins)` — **min-anchored**, and it generated our golden fixture. **MATLAB** `phasecongmono.m:466-468` is `edges = 0:mx/nbins:mx` — **zero-anchored**, and the lone outlier. Settled by installing Images.jl into a scratch depot and executing Kovesi's Julia `rayleighmode` on the exact amplitude `_phasecong3` feeds it: Julia and `phasepack` both return `0.0009652525656640632`; MATLAB returns `0.0009652419842992787`. **We ship the Julia branch**, as M4 does for `perfft2` — the register cannot follow Julia on one fork and MATLAB on another without saying so. The two anchors *coincide* whenever the data contain an exact zero (Julia's `minimum_finite` sees it), and diverge only on strictly positive data, which is every real amplitude array. **The genuine correction:** our old port *dropped zeros* before histogramming. Neither Julia nor `phasepack` does. Zeros are now retained. Because the shipped plates' amplitude arrays contain **no** exact zeros, this correction is bit-identical there — so `FocusEdgePhase` is now bit-identical to its pre-refactor self at **both** `noise_method = -1` and `-2`. Pinned by `test_zeros_are_retained` (retention) and `test_bins_are_anchored_at_the_minimum_not_at_zero` (anchor, on strictly positive data), which an earlier single test conflated and could distinguish neither. |
| M7 | `noise_method` outside `{-1, -2} ∪ [0, ∞)` raises `ValueError` | no reference validates it | CAPABILITY | `-1.5` matched neither branch, left `tau = 0.0`, and silently returned `T = ε`. Kovesi's MATLAB errors on the undefined `tau`. The dispatch compares against `epsilon`, following **Julia only** (`phasecongruency.jl:512`, `abs(noisemethod + 1) < epsilon`). **Correction (2026-07-09):** an earlier version of this row also cited `phasecongmono.m:224` for the epsilon compare. That is wrong — `phasecongmono.m:225` is an exact `if noiseMethod == -1`. MATLAB and Julia genuinely differ here; we follow Julia. Same species as S7: a claim about a reference made without opening it. |
| M8 | `riesz_multiplier` divides **componentwise**: `(fx/radius)*1j - (fy/radius)` | `phasepack:156` — the generator of our golden fixture — computes the numpy form `(1j*fx - fy)/radius` | **CORRECTED to the reference** | The three source texts print the same glyphs, but the languages differ beneath them. MATLAB's `./` and Julia's `/(z::Complex, x::Real)` (`base/complex.jl:348` → `Complex(real(z)/x, imag(z)/x)`) do a **true division per component**. numpy promotes the real denominator and runs `nc_quot`; with a zero imaginary part that branch is `scl = 1/r` followed by a **multiply**. Verified bit-exactly: `(1j*a - b)/r == (a*(1/r))*1j - (b*(1/r))` on 200 000 samples, differing from componentwise division on **42.8%** of them by up to **1.41 ulp**. Settled by executing `frequencyfilt.jl:238` in Julia and comparing raw IEEE-754 bit patterns: componentwise is **bit-identical**, the numpy form is not. **Cost:** the fixture is phasepack's, so `check_19`'s agreement loosens `3.52e-14 → 5.32e-14` — still **7.27 orders** inside `rtol=1e-6`. Accuracy was never in question; provenance was. Guarded on shipped code by `TestRieszMultiplier::test_it_divides_componentwise_as_kovesi_does_not_as_numpy_would` and `::test_packs_the_two_odd_channels_into_one_complex_array`; `check_19` corroborates via the spec's parallel implementation, which is kept bit-identical to the kernel. |
| M9 | `n_scale < 2` raises `ValueError` at **all three** public entry points onto `spread_weight`'s `n_scale − 1` divisor: `monogenic_phase_congruency`, `monogenic_channel_response`, `congruency_from_accumulators` | no reference validates it; Kovesi's `phasecongmono` divides by `(nscale-1)` unguarded | CAPABILITY | The kernel's own docstring promised "must be at least 2" and did not enforce it, and **both** illegal values failed *silently*. Measured on a 64×64 step edge: `n_scale=1` divides by `n_scale − 1 == 0` inside `spread_weight` and returns an all-zero `pc` with only a `RuntimeWarning`; `n_scale=0` skips the scale loop, leaves `max_amplitude` all zero, and returns an all-zero `pc` with **no warning at all**. Both operations guard themselves with `Field(ge=2)` (M3), so no shipped path changes — but these are pure functions that `FocusEdgeColorPhase` calls directly, and a plausible array of zeros is the worst possible answer. Same family as M7. **Extended 2026-07-09**, when the accumulator refactor split the function in three and opened a *third* door. `congruency_from_accumulators` is the one the fusion kernels call, once per mode, and it fails **worse than the other two and in the opposite direction**: handed accumulators from a valid `n_scale=4` channel, `A_Σ > A_max`, so `width = (A_Σ/(A_max+ε) − 1)/0 → +∞`, the sigmoid **saturates to 1.0 everywhere**, the frequency-spread penalty is silently switched off, and `pc` returns finite, inside `[0,1]`, and *larger than the truth* — `0.578872` against `0.561266` — behind a single `RuntimeWarning`. `n_scale=0` returns zeros with no warning. Both pass a naive `0 ≤ pc ≤ 1` check. (The all-zero limit M9 originally recorded arises because with one scale `A_Σ == A_max`, making the numerator `−ε/(A_Σ+ε) < 0` and `width → −∞`. Same divisor, opposite limit, set by which accumulators reach it. The two measurements do not conflict.) Pinned by `test_n_scale_below_two_raises` on both functions (proven able to fail: removing either guard reddens all three parametrisations), `test_n_scale_two_is_accepted` against an off-by-one, and `test_the_unguarded_failure_is_plausible_not_obvious`, which asserts the rogue weight saturates to `1.0` rather than to `0` — the assertion that would have to be *deleted*, not adjusted, by anyone removing the guard on the grounds that "it returns zeros anyway". |
| M10 | `monogenic_phase_congruency` raises `ValueError` when `sigma_onf >= 1.0` or `mult <= 1.0` | no reference validates either; Kovesi divides unguarded | CAPABILITY | `sigma_onf = 1.0` makes `log(sigma_onf) = 0`, so `log_gabor_scale` divides by `2*0**2`; `mult = 1.0` makes the geometric noise sum divide by `1 - 1/mult = 0`. Both return an **all-NaN** `pc`, which is strictly worse than M9's all-zero: NaN compares false to everything, so it passes a naive `0 <= detect_mat <= 1` range check. Found by the B+C review. Both were legal and *tunable*: `FocusEdgeMonogenicPhase(sigma_onf=1.0).apply(...)` returned an all-NaN `detect_mat`, and `FloatRange` "appends `high` exactly" (`tune/_search_space/_domains.py:86`), so a grid run over `TuneSpec(0.1, 1.0)` would have evaluated it. **Decided 2026-07-09 (user):** guard the kernel; narrow only the new operation (`sigma_onf: lt=1.0`, `TuneSpec(0.1, 0.99)`). `FocusEdgePhase`'s `Field` bound is left at `le=1.0`, so `ImagePipeline.from_json` still loads every legacy pipeline — `enhance_features_sigma_onf_high.json` pins `sigma_onf: 1.0` — but `.apply()` raises. **Correction, same day:** the `sigma_onf` guard was first placed in `monogenic_phase_congruency`, which `FocusEdgePhase` never calls — it reaches `log_gabor_radial` directly. So `FocusEdgePhase(sigma_onf=1.0).apply()` still returned an **all-zero** map on a step edge (passing a naive `0 <= x <= 1` check, because that is what a zero does) and an **all-NaN** one on a real plate, raising neither time. Found by the simplify pass. The guard now lives in `log_gabor_scale`, at the division itself, so both operations fail loudly; `mult` stays guarded in `monogenic_phase_congruency`, where its geometric sum divides. `FocusEdgePhase`'s `TuneSpec` high also drops `1.0 → 0.99`: `FloatRange` appends `high` exactly, so a grid run would otherwise hit the raise every time. A `TuneSpec` is a search window, not a validity bound, and narrowing it breaks nothing that deserializes. Pinned by `test_sigma_onf_at_or_above_one_raises`, `test_mult_at_or_below_one_raises`, `test_sigma_onf_just_below_one_is_accepted`, and `TestSigmaOnfOneIsRejectedAtApplyTime` — which also records that `ImageOperation` wraps the `ValueError` into a bare `Exception` **twice** (`_image_operation.py:422` and `:469`), destroying the type, so the test walks the `__cause__` chain. |
| M11 | `orientation = arctan2(-sum_h2, sum_h1)`, folded into `(-π/2, π/2]` | Kovesi writes single-argument `atan(-sumh2/sumh1)` in both implementations (`phasecongmono.m:292`, `phasecongruency.jl:580`) | FORCED | `atan(y/x)` divides by zero wherever `sum_h1 == 0`, which happens on any perfectly vertical structure; `arctan2` does not. The two are equal mod π and differ on 44–48% of finite elements by at most `4.441e-16`. Recorded because M8 was recorded for a *strictly smaller* numeric difference, and the standard must apply to both or neither. The fold reproduces single-argument `atan`'s range exactly, verified at endpoints and signed zeros — including `arctan2(-0.0, 0.0) = -0.0`, which survives the fold and maps to `0.5`. **`-π/2` is unattainable**, so `orientation`'s true image under `(θ+π/2)/π` is `(0, 1]`, not `[0, 1]`; `feature_type` attains both endpoints. |
| M12 | `sintheta = fx / radius`, `costheta = fy / radius` | **Kovesi's Julia** `frequencyfilt.jl:430-431` builds exactly this. His **MATLAB** `phasecong3.m:189-190` builds `sin/cos` of `atan2(-y, x)` | *not a deviation — a recorded choice; the references disagree* | **Rewritten 2026-07-09 after the F review. The previous revision was badly wrong**: it called this "mathematically identical, numerically not… up to `2.22e-16`". The elementwise magnitude is `2.22e-16`, but the *pair is swapped and signed*: our `(sintheta, costheta)` equals Kovesi's `(costheta, -sintheta)`, i.e. `(sin(θ+π/2), cos(θ+π/2))`. **The whole oriented filter bank is rotated 90°.** Working it through `_phasecong3`'s angular filter: MATLAB's angular distance is `|φ + A|`, ours is `|π/2 − φ − A|`. We ship the Julia, as `M4` and `M6` do. Measured, switching to MATLAB's grid: `max|Δpc_sum|` = `2.2e-04` at `n_orient=6`, `2.4e-04` at `8`, and **`5.1e-02` at `n_orient=5`** — because for *even* `n_orient` a 90° shift permutes the same set of filter angles, while for *odd* `n_orient` (which `TuneSpec(4, 8)` admits) it is a genuinely different bank. `M`, `m` are eigenvalues of the covariance tensor and so rotation-invariant; only `pc_sum` and `orientation` move. The reported orientation is corrected separately — see `M13`. Predates this branch: `main`'s `_construct_filter_grids` divided identically. |
| M13 | `_phasecong3` **reflects** its reported orientation: `or -> π/2 - or`, folded into `(-π/2, π/2]` | Kovesi's Julia `phasecong3` reports `π/2 - φ`; his MATLAB reports `-φ`; his `phasecongmono` reports `φ` | **CORRECTED — the reference contradicts itself** | **Settled by running Kovesi's code, not reading it.** Installed `ImagePhaseCongruency.jl` into a scratch Julia depot and ran his own `phasecong3`: a vertical step edge reads **`-86.35°`**, a horizontal one **`-0.06°`** — while its docstring says *"0 corresponds to a vertical edge, +ve anticlockwise."* His `phasecongmono`, in the same package, reads **`-0.03°`** on that vertical edge. **Kovesi's Julia contradicts itself.** His MATLAB `phasecong3` satisfies "0 = vertical" but reports `-φ` — clockwise — violating the other half of the same sentence. No implementation of his satisfies both halves; only `phasecongmono` does. We reflect so this operation reports `φ`. It then matches our docstring, matches `FocusEdgeMonogenicPhase.orientation` (mean disagreement **`35.89° → 1.64°`**, the residual being the oriented bank's angular quantisation), and matches his MATLAB at `0°` and `90°`. **The relation is a reflection about 45°, not a rotation** — measured across a full angle sweep: rotation gives `39.90°` mean error, reflection `5.25°`. Testing only `0°` and `90°`, where the two coincide, would have hidden that. `orientation` is a pure **output**: nothing consumes it (`detect_mat` exposes only `M`/`m`/`pc_sum`; `_filamentous_fungi_detector.py:424` reads `pc_sum`). So `M`, `m`, `pc_sum`, `feature_type` and `T` are **bit-identical to the pre-refactor tree** — verified. Switching the *grid* instead would have moved `pc_sum` and still left the two operators `35.89°` apart. Pinned by `TestOrientationConvention` and by `TestPhaseCong3Characterization`; removing the reflection reddens 4 tests. |

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
| C7 | `fusion="joint"` default | paper uses per-channel + L2 | CAPABILITY | `l2` has no cross-channel interaction, so incoherent chroma can never veto a spurious luminance edge. Unvalidated; `color-phase-congruency.md` §7.2's chromatic-aberration experiment is its acceptance test, and a null result flips the default back to `l2`. (An earlier revision of this row cited a non-existent "§8.4.13".) |
| C8 | `fusion="coherent"` | no reference | CAPABILITY | Opt-in, never default. Annihilates genuine anti-correlated chromatic edges. |
| C9 | `chroma_weight_1`, `chroma_weight_2` | paper has no weights | CAPABILITY | Defaults `1.0`/`1.0` reproduce the paper exactly. `(1,1,1)` on raw Lab **is** the CIE76 ΔE metric. |
| C10 | `T_total = Σ wᵢTᵢ`, `A_max = Σ wᵢ·maxₖA` in joint mode | no reference | CAPABILITY | Follows from `joint` existing at all. 1-homogeneous, so the scale-invariance argument survives. Untested against anything external. |
| C11 | `ValueError` on achromatic input | no reference | CAPABILITY | `joint` degenerates to a luminance congruency divided by itself. Fail loudly. |
| C12 | Kovesi's spread weight `W` reused | paper mis-cites it to Abdou & Pratt and never validates it for DoP scales | **UNVALIDATED** | A reasonable guess, not a derivation. Flagged; no evidence either way. |
| C13 | Rayleigh median constant `√(log 4)` on a 4-component amplitude | exact only for 2 components | CONSTANT BIAS | Measured `0.811×`. Kovesi states `k` absorbs exactly such a per-bank constant. Inherit, record. |
| C15 | `_color_phase_congruency` returns `orientation` and `feature_type` from the fused vector `Σᵢwᵢ·vᵢ`; **neither is exposed** via `output`, which is `Literal["pc"]` | no reference — CMPCM emits a single edge map | CAPABILITY, internal | The spec declared `ColorPhaseOutput = Literal["pc","orientation","feature_type"]` and never defined either angle. It cannot be filled in by analogy: **only `coherent` builds a fused monogenic vector.** `joint` sums scalar energies `Σwᵢ‖vᵢ‖`; `l2` combines three finished congruency maps. So under the *shipped default* these angles describe a quantity the response never touched. **Decided 2026-07-09 (user):** ship `pc` only; compute the angles anyway and hang them on the protected helper's result, so a future consumer reaches them without a breaking change. Precedent, not novelty: `_phasecong3` already reports angles off an `energy_v` (`_focus_edge_phase.py:344-347`) that did not produce its `pc_sum`, and exposes only `M`/`m`/`pc_sum`. **Hazard:** the odd pair's sign encodes edge polarity, so an anti-correlated chromatic edge — `L*` falling as `b*` rises — cancels in the channel sum. The angle is least reliable exactly where colour is doing the most work. Pinned at `chroma_weight_* = 0`, where the fused vector collapses to `v_L` and both angles must equal `FocusEdgeMonogenicPhase`'s to `rtol=1e-10`. |
| C16 | `color_space="hsv"` bandpasses **raw** hue, across its wrap discontinuity | CMPCM uses HSV and does not unwrap | CAPABILITY, recorded hazard | `H` is circular on `[0, 1)`. A log-Gabor bandpass across the `0.99 → 0.01` seam sees a unit step where the colour is continuous, so a near-red boundary manufactures a phantom edge. We do **not** unwrap: the paper does not, and `("hsv", "l2")` is §7.1's PFOM-regression configuration, which must exercise the paper's actual quantity. Compounds `C6`'s conditioning problem — at `S = 0.01` an RGB perturbation of `0.00036` swings hue by `0.01`. `color_space="lab"` is the default and has no seam. **Demonstrated, not asserted**, by `TestHueWrapArtifactIsReal`: a flat image of constant `S` and `V` whose hue ramps through red contains no edge, `hsv` responds anyway, `lab` does not. That test goes green if someone "fixes" it by unwrapping — which would be a silent divergence from CMPCM. |
| C17 | The output is **not** invariant to a global rescale of `w` | an earlier §4.2 of this spec claimed "invariant up to `O(ε/A_total)` — ~1%" | **CORRECTED — the claim was false, and named the wrong `ε`** | The `ε` that breaks 1-homogeneity is **not** the one in `E_total + ε`. It is the one in `A_max + ε`, inside `width = (A_total/(A_max + ε) − 1)/(n_scale − 1)`, which is fed to a sigmoid of sharpness `g = 10`. Near the knee a small `width` shift moves `W` by a large *relative* amount, and `W` multiplies the whole output. Measured over `c ∈ {0.01, 100}`, 20 000 draws per regime: **100.0%** at `A ∈ [0.5, 5.0]` (Lab `L*` scale), **670.0%** at `A ∈ [0.05, 0.5]`, **778.9%** at `A ∈ [1e-3, 0.5]`. §4.2's two-degrees-of-freedom argument **survives** — the *ratio* `E_total/A_total` is still 1-homogeneous — but the retracted `rtol=2e-2` test was a guess, and loose enough that its anchor did no work. §7 test 4 now masks low-response pixels and derives its tolerance from `ε/(c·A_total)` and `|dW/d(width)| ≤ g/4`. Same species as `S4`: a number measured once, written down, and then drifted from. Re-derived by `fusion_algebra.py` check 04. |

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
| "Exactly invariant" / "bit-identical" under a global weight rescale. | `ε` is not 1-homogeneous. But the replacement figure was wrong too — see the next row. |
| "…invariant up to `O(ε/A_total)` — ~1% over `c ∈ [0.01, 100]`." | **Wrong twice: wrong `ε`, wrong number.** The culprit is `A_max + ε` inside a `g = 10` sigmoid, not `E_total + ε`. Measured: **100%** at Lab `L*` amplitudes, **779%** at low ones. The retraction that introduced this claim retracted a *stronger* claim and replaced it with a merely-false one. `C17`. |
| §3.1's `response` column: `0.0091` for three coherent channels, `0.1425` for the measured prior. | Matches **no single `deviation_gain`** — row 2 needs `1.0373`, row 3 needs `1.4265`. At the shipped `1.5` the responses are `0.0000` and `0.0982`. The `ratio` column was always right. The conclusion survives and strengthens: the annihilation is *exact*. `fusion_algebra.py` checks 01 and 02. |
| The **correction** to that column: `0.0983` for the measured prior. | Wrong in the fourth decimal, and wrong the same way the thing it replaced was wrong. It was obtained by feeding the *printed* four-decimal ratio `0.8247` back through the formula. The exact ratio is `0.824665992993527`; the exact response is `0.0982226557669601`. With `d(response)/d(ratio) = dg/√(1−ratio²) = 2.6520` and a ratio already rounded by `3.4e-05`, the round-trip moves the response by `9.0e-05`. **A four-decimal intermediate cannot determine a four-decimal result.** This is `S4` for the fourth time, and the first time it was caught *by machine, in minutes*, rather than by a later reviewer — the whole argument for `logic_validation_scripts/`. The check's expected values are now full-`float64` literals at `1e-12`; reintroducing `0.0983` reddens it (verified). |
| `E_total/(A_total+ε)` maxima of `0.996436` (joint) and `0.970076` (coherent). | **Sampled, not determinate**, and quoted from an uncommitted seed nobody could reproduce. The committed `fusion_algebra.py` (seed `20260709`) reports `0.996808` and `0.966120`. The load-bearing statement is the *analytic* bound `E_coherent ≤ E_joint ≤ A_total`, which is what the check asserts; the maxima only say how close the draw got. Cite a sampled number with its seed or not at all. |
| `ColorPhaseOutput = Literal["pc","orientation","feature_type"]`. | Declared in §5's field table and defined nowhere — not in the data flow, not in any fusion formula, not in any test. **Only `coherent` builds a fused monogenic vector**, so under the shipped default both angles would have been unreferenced inventions. Narrowed to `Literal["pc"]`; the angles moved to the protected helper's result. `C15`. |
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
