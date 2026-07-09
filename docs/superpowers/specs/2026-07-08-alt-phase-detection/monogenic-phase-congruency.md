# `FocusEdgeMonogenicPhase` — design

**Status: CLOSED. Buildable now.**
**Companion:** [`references.md`](./references.md) §5 and §8.1. **Deviations:** [`drift-register.md`](./drift-register.md).

Phase congruency via an isotropic log-Gabor bandpass and the Riesz transform, instead of Kovesi's
oriented log-Gabor bank. A drop-in alternative to the existing `FocusEdgePhase` that eliminates the
orientation sweep.

**This is a port, not a derivation.** Kovesi's Julia (`ImagePhaseCongruency.jl`), Kovesi's MATLAB
(`phasecongmono.m`), and the MIT-licensed `phasepack` agree verbatim. Where this document and a
reference disagree, the reference wins.

---

## 1. Attribution

The field-notebook card and `breadth-survey.md` cite Wang Lijuan et al., *Image feature detection
based on phase congruency by Monogenic filters*, CCDC 2014. **We have not read that paper.**

What this operation implements is **Kovesi's `phasecongmono`**. The class docstring must say so, and
must not imply we have reproduced the CCDC formulation, which may differ.

---

## 2. Algorithm

Per scale `s`, filter the image FFT with the isotropic log-Gabor radial, take the inverse FFT for the
even channel `f_s`, and apply the packed Riesz multiplier for the two odd channels. Then
`Aₛ = √(f_s² + h1_s² + h2_s²)`.

### 2.0 `fft2` or `perfft2`? The references disagree. We ship `fft2`.

```
IM = fft2(img)           # Julia, phasecongruency.jl l.446: "Use fft rather than perfft2"
IM = perfft2(img)        # MATLAB, phasecongmono.m l.156 — and phasepack, which ports it
```

Kovesi's MATLAB bandpasses the image's **periodic component** (Moisan's periodic/smooth
decomposition). His later Julia explicitly comments that out. No reason is given in either source.

**The choice is material, not cosmetic.** `fft2` treats the image as tiled, so the intensity jump
between opposite borders leaks a cross-shaped artifact into every log-Gabor band: `pc` differs by up
to **0.67 absolute**, and 16 px inside every edge still by `0.13` on `step2line`.

**Decision:** implement `fft2` — Kovesi's most recent word. The golden fixture (§7) pins the
`perfft2` branch, because that is what `phasepack` computes. The two branches differ only in *which
spectrum enters an otherwise identical chain*, so fixturing one validates the other's machinery. The
kernel takes `periodic: bool = False`; the **operation does not expose it**.

`perfft2` is specific to `phasecongmono`. Both MATLAB (`phasecong3.m` l.146) and Julia `phasecong3`
use plain `fft2`, so the shipped `FocusEdgePhase` is unaffected and §3.2's scope guard holds.

`verify_claims.py::check_18`, `references.md` §10.3.

### 2.1 The congruency formula is NOT `_phasecong3`'s

```
sumAn  = Σₛ Aₛ                      maxAn = maxₛ Aₛ
energy = √( (Σₛf_s)² + (Σₛh1_s)² + (Σₛh2_s)² )
width  = (sumAn/(maxAn + ε) − 1)/(n_scale − 1)
W      = 1/(1 + exp(g·(cutoff − width)))

PC = W · max(1 − deviation_gain·acos(energy/(sumAn + ε)), 0) · max(energy − T, 0)/(energy + ε)
or = atan(−sumh2/sumh1)
ft = atan2(sumf, √(sumh1² + sumh2²))
```

Two things differ from `_phasecong3` and both are load-bearing. The noise threshold is applied as a
**multiplicative fraction** `max(E−T,0)/(E+ε)`, not subtracted from the numerator — Kovesi's comment
says subtracting it early "would interfere with the phase deviation computation." And the
phase-deviation term uses `acos(E/ΣA)` scaled by `deviation_gain`, which sharpens edge localization.

### 2.2 Noise threshold — keep it verbatim

```
tau      = median(sumAn)/√(log 4)                    # at the FIRST scale only
totalTau = tau · (1 − (1/mult)^n_scale)/(1 − 1/mult)
T        = max( totalTau·√(π/2) + k·totalTau·√((4−π)/2),  ε )   # the floor is phasepack's
```

The `max(…, ε)` floor is **`phasepack`'s, not Kovesi's** — neither MATLAB nor Julia has it. It is
inactive on every non-constant image (the smallest `T` across the five fixture images is `3.7e-3`,
37× the floor; only a literally constant image reaches it). Kept because it costs nothing, guards the
degenerate case, and the fixture encodes it. Recorded in `drift-register.md`.

`(1/mult)^s` is the log-Gabor instantiation of Kovesi's "relative bandwidths" principle, and it is
**exact for this bank** (`references.md` §4.4.3: the kernel-norm ratios converge to `1/mult` to six
digits). It has one known wrinkle — the `s=0 → s=1` ratio is `0.715`, not `0.476`. The lowpass is
**not** the cause; without it the ratio is still `0.589`. The finest log-Gabor simply is not
band-limited on the lattice (~30% of scale-0's energy sits above `|u| = 0.45`, ~15% above Nyquist), and
that is the scale `τ` is anchored on. **Do not fix this.** The golden fixture must match, and the
reference is the definition.

Likewise `median(A)/√(log 4)` is the exact Rayleigh median, i.e. of a *two*-component amplitude,
while `A` here has three components. Measured bias: `1.048×`. It is a constant per bank, and Kovesi
states that `k` absorbs exactly such a constant. **Inherit it.**

### 2.3 `ε = 1e-4`, and clamp the `acos`

All three references use `ε = 1e-4` for `phasecongmono` (Julia line 441, MATLAB line 153, `phasepack`
line 129). `1e-5` belongs to **Julia's** `phasecong3` (MATLAB's `phasecong3.m` uses `1e-4`), which is
why `FocusEdgePhase` — a Julia port — correctly uses it. Not
cosmetic: `ε` is the only thing keeping `energy/(sumAn + ε)` strictly below 1 before it enters
`acos`.

**No reference clamps the `acos` argument.** Mathematically `energy ≤ sumAn` by the triangle
inequality on the scale sum, so with `ε > 0` the ratio is `< 1`; roundoff can still exceed 1 and
yield `NaN`. We clamp to `[-1, 1]` — a documented, numerically-necessary deviation, paired with a
test asserting the clamp **never activates** on real images, so it is provably a no-op.

---

## 3. Architecture

| File | Exports | Reads | Writes |
|---|---|---|---|
| `enhance/_monogenic_kernels.py` | nothing (private, shared) | — | — |
| `enhance/_focus_edge_monogenic_phase.py` | `FocusEdgeMonogenicPhase(FocusEdge)` | `detect_mat` | `detect_mat` |

### 3.1 `_monogenic_kernels.py`

Pure functions, no `Image` dependency, unit-testable without fixtures:

- `construct_filter_grids(rows, cols)` — quadrant-shifted frequency grids.
- `log_gabor_radial(radius, n_scale, min_wavelength, mult, sigma_onf)` — isotropic radial bandpass.
- `riesz_multiplier(fx, fy, radius)` → `(1j*fx - fy)/radius`. **One division**, as
  `phasecongmono.m:183` and `frequencyfilt.jl:234-241` both do. An earlier revision specified
  `1j*sintheta - costheta`, which divides twice and reassociates: ~1 ulp per bin, moving golden
  agreement from `3.52e-14` to `5.32e-14`. Harmless, but a shortcut, and this is a port.
- `rayleigh_mode(amplitude)` — Rayleigh σ from the amplitude histogram.
- `spread_weight(sum_amplitude, max_amplitude, n_scale, cutoff, g)` — Kovesi's `W`.

`riesz_multiplier` is Kovesi's `packedmonogenicfilters` (`H = (i·fx − fy)/f`), which packs both odd
channels into one complex array so a single `ifft2` yields `h1` in the real part and `h2` in the
imaginary part. Our existing `construct_filter_grids` already returns `sintheta = fx/freq` and
`costheta = fy/freq`, so no new grid maths is needed. Its lowpass `1/(1 + (radius/0.45)**30)` is
already identical to Kovesi's `lowpassfilter(freq, 0.45, 15)` (the order is doubled).

`FocusEdgeColorPhase` shares this module wholesale; see [`color-phase-congruency.md`](./color-phase-congruency.md).

### 3.2 Bounded refactor of `FocusEdgePhase`

Four helpers already exist inside `enhance/_focus_edge_phase.py` (542 lines). Move these, and only
these, into `_monogenic_kernels.py`, and import them back:

- `_construct_filter_grids` → `construct_filter_grids`
- `_construct_log_gabor_filters` → `log_gabor_radial`
- `_rayleigh_mode` → `rayleigh_mode`
- the inline `width` / `weight` block → `spread_weight`

**Scope guard:** `_compute_angular_spread` and `_phasecong3` stay put. Nothing else is touched.
Behaviour must be bit-identical; `tests/unit/enhance/test_phase_congruency.py` (292 lines) is the
gate.

### 3.3 Latent bug found while specifying this

`FocusEdgePhase` declares `n_scale: Annotated[int, TuneSpec(3, 6)] = Field(4, ge=1)`, but
`_phasecong3` computes

```python
width = (sum_amplitude / (max_amplitude + epsilon) - 1) / (self.n_scale - 1)   # line 320
```

At `n_scale=1` this divides by zero, emits a `RuntimeWarning`, and the operation **silently returns
an all-zero `detect_mat`** (verified on `load_synth_yeast_plate()`; no NaN, no Inf, just zeros).
`TuneSpec(3, 6)` means the optimizer never lands there, so it has gone unnoticed.

This spec tightens `FocusEdgePhase` to `ge=2` as part of the §3.2 refactor of the very block that
contains the division.

> Behaviour change: `FocusEdgePhase(n_scale=1)` currently returns zeros and will now raise
> `pydantic.ValidationError`. Trading a silent wrong answer for a loud one is right, but it belongs
> in the changelog. It is unrelated to this port and may be split into its own commit.

---

## 4. Fields

| field | default | annotation | source |
|---|---|---|---|
| `n_scale` | `4` | `TuneSpec(3, 6)`, `Field(ge=2)` | Kovesi; `ge=2` per §3.3 |
| `min_wavelength` | `3.0` | `TuneSpec(2.0, 10.0)`, `Field(ge=2.0)` | Kovesi |
| `mult` | `2.1` | `TuneSpec(1.5, 3.0)`, `Field(gt=1.0)` | Kovesi |
| `sigma_onf` | `0.55` | `TuneSpec(0.1, 1.0)`, `Field(ge=0.1, le=1.0)` | Kovesi |
| `k` | `3.0` | `TuneSpec(0.5, 20.0)`, `Field(ge=0.0)` | **`phasecongmono`'s default is 3.0**, not `phasecong3`'s 2.0 |
| `deviation_gain` | `1.5` | `TuneSpec(1.0, 2.0)`, `Field(gt=0.0)` | Kovesi: "sensible values are from 1 to about 2" |
| `cutoff` | `0.5` | `TuneSpec(0.3, 0.7)`, `Field(gt=0.0, lt=1.0)` | Kovesi |
| `g` | `10.0` | `TuneSpec(2.0, 20.0)`, `Field(gt=0.0)` | Kovesi |
| `noise_method` | `-1.0` | `TuneSpec(tunable=False)` | Kovesi |
| `output` | `"pc"` | `MonogenicOutput` | matches `phasecongmono`'s return tuple |

`MonogenicOutput = Literal["pc", "orientation", "feature_type"]`, declared once as a `TypeAlias` in
`sdk_/typing_.py` alongside `FootprintShape` and `DetectMode`. No `Enum` — no user-visible
documentation surface, so per the `adding-an-operation` skill a bare `Literal` alias suffices.

`orientation` and `feature_type` are angles. Map them into `[0,1]` before writing `detect_mat`
(`(θ + π/2)/π` and `(ft + π/2)/π`), and say so in the docstring — `detect_mat`'s contract forces it,
and the raw angle is recoverable by inverting the map.

`ε = 1e-4` (§2.3), a module constant, not a field.

---

## 5. Error handling

Ordinary pydantic bounds; invalid input raises `pydantic.ValidationError` (a `ValueError` subclass).
The `acos` clamp (§2.3) is the only numeric guard, and it is tested to be inert.

---

## 6. Testing

Test-first, per `superpowers:test-driven-development`.

1. `test_phase_congruency.py` passes unchanged after the §3.2 refactor. Bit-identical.
2. **Golden fixture.** Generate `phasecongmono`'s output once from `phasepack` on a fixed synthetic
   image, commit the arrays as a `.npz`, and assert `rtol=1e-6`. See §7 — this needs a one-off
   approved install; `phasepack` does **not** become a runtime or CI dependency.
3. **`ε` regression.** Assert the module constant is `1e-4`, with a comment naming the three
   references. Cheap, and it prevents someone "unifying" it with `FocusEdgePhase`'s `1e-5`.
4. **The `acos` clamp is a no-op.** Instrument it; run over `load_synth_yeast_plate()`,
   `load_yeast_plate()`, `load_fungi_plate()`; assert it never fired.
5. **Contrast and illumination invariance.** `f → αf + β` leaves `pc` unchanged within tolerance.
   The defining property of phase congruency.
6. `detect_mat` output in `[0,1]`; `rgb`/`gray` unmutated (free from the integrity validator);
   `to_json`/`from_json` round-trip; constructible with no arguments; 90° rotation equivariance.
7. **Axis convention — two bugs, and axis-aligned edges catch only one.** A vertical edge must give
   `orientation = 0` and a horizontal edge `π/2`. That guards an `fx`/`fy` swap, which rotates every
   orientation by 90° while leaving `pc` untouched (measured: `max|Δpc| = 1.5e-17`).
   It does **not** guard the sign of `−sumh2`, which encodes a y-up convention. Writing
   `atan2(+sumh2, sumh1)` reflects every orientation about the x-axis, and `0` and `π/2` are their
   own mirror images mod `π` — so an axis-aligned test pair is blind to it in both `pc` *and*
   `orientation`. **Test both bugs on `starsine`**, whose rays span every orientation at once:
   recovered orientation must match the generator's own `theta` field to `< 2°` median. Each bug
   shifts it by ~45°. See `references.md` §10.2 and `verify_claims.py::check_17`.
8. **Feature-type invariance (the positive control).** On `step2line`, `pc` at the congruency column
   must stay within `~1.5×` while `feature_type` sweeps `0 → π/2`. Gradient magnitude collapses
   `18×` over the same sweep — that gap is the operator's reason to exist.
   `verify_claims.py::check_15`.
9. **Noise rejection.** On `noiseonf(sze, 1.5)`, `pc₀.₉₉₉ < 0.5` with the threshold on and `> 0.5`
   with it off. Congruency alone does not reject `1/f` noise; `T` does.
   `verify_claims.py::check_16`.
10. `FocusEdgeMonogenicPhase` and `FocusEdgePhase` localize a synthetic step edge to within 1 px of
    each other.
11. Doctests on `load_synth_yeast_plate()`, per the repo rule.
12. `ValidationError` on out-of-bounds fields; `n_scale=1` rejected.

Tests 7–9 use Peter Kovesi's synthetic image generators (`step2line`, `circsine`, `starsine`,
`noiseonf`), MIT-licensed and already ported — with the notice — in `verify_claims.py`. Lift them
into the test helper from there rather than re-porting.

---

## 7. The golden fixture — **done**

`phasepack` 1.5 (MIT, unmaintained since 2016) was installed once with the user's approval, used to
generate `tests/fixtures/phasecongmono_golden.npz`, and removed. No runtime dep, no CI dep, no
reliance on a 2016 package continuing to install.

The fixture holds `pc`, `ft` and `T` for five 64×64 images — a step edge plus all four of Kovesi's
generators — at `nscale=4, minWaveLength=3, mult=2.1, sigmaOnf=0.55, k=3.0, cutOff=0.5, g=10.0,
deviationGain=1.5`, and it pins the **`periodic=True` branch** (§2.0), since that is what `phasepack`
computes. Sizes are **even** on purpose: at odd sizes `phasepack`'s frequency grid disagrees with
*both* of Kovesi's — it divides by `N−1` where they divide by `N`, in `filtergrid` and again in
`lowpassfilter`. That is a `phasepack` bug, and an odd-sized fixture would bake it in.
`verify_claims.py::check_18`.

Current agreement, `verify_claims.py::check_19`: `max|Δpc| = 3.5e-14`, `max|Δft| = 6.7e-13`,
`max|ΔT| = 4.4e-16`. Eight orders inside the `rtol = 1e-6` target.

**It has already earned its keep.** The fixture is what surfaced the `perfft2` fork (§2.0) and the
`T`-floor misattribution (§2.2). The behavioural controls — `step2line`, `noiseonf`, `starsine` — are
blind to both. Reuse the same fixture for `FocusEdgeMonogenicPhase`'s test suite rather than
regenerating it; call the kernel with `periodic=True` in that one test only.

Two deliberate departures from `phasepack`, both to be preserved in the operation:

- **Clip `acos`'s argument to `[−1, 1]`.** `phasepack` does not, and would emit `NaN`. Instrument the
  clamp and assert it never fires (test 4); it never has.
- **`ε = 1e-4`**, matching `phasepack` and Kovesi — not `FocusEdgePhase`'s `1e-5` (test 3).
- **`T` floored at `ε`.** `phasepack`'s, not Kovesi's; inactive on any non-constant image (§2.2).

Regenerate only with `uv add --group dev phasepack` and a deliberate decision to move the goalposts.
Per `references.md` §10, `phasepack` ships **no tests of its own**, so this fixture is stronger
validation than the reference implementation carries — and "it matches the reference" is a claim about
*transcription*, not about *correctness*. Tests 7–9 and `verify_claims.py::check_09b` are what speak
to correctness.

---

## 8. Acceptance criteria

1. `test_phase_congruency.py` passes unchanged.
2. Golden-fixture agreement at `rtol=1e-6`.
3. The `acos` clamp is provably inert on all three shipped plates.
4. The axis-convention test passes.
5. `uv run mypy src/phenotypic` and `uv run ruff check` clean.
6. The operation appears in the GUI builder's enhancer dropdown.
7. Class docstring attributes the algorithm to Kovesi's `phasecongmono`, not to CCDC 2014.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Transcription error in the port | Golden fixture at `rtol=1e-6` — **generated, committed, and already proven** (§7). It is the only thing that caught the missing `perfft2`. |
| Silently switching `fft2`/`perfft2` | §2.0 + `verify_claims.py::check_18`. **No behavioural test catches this** — `pc` shifts by 0.67 while `step2line`/`noiseonf`/`starsine` all still pass. Only the fixture sees it, and only because it pins the `perfft2` branch. |
| `fx`/`fy` axis swap silently rotates orientation by 90° | Test 7. `pc` is invariant to the swap (`1.5e-17`), so nothing else catches it. |
| `atan2(+sumh2, …)` sign flip mirrors orientation about the x-axis | Test 7's `starsine` arm. **Axis-aligned edges are blind to this one**, in `pc` and `orientation` alike — the bug the obvious test cannot see. |
| The §3.2 refactor regresses shipped `FocusEdgePhase` | Scope is four helper functions; the existing 292-line test file is the gate. |
| Someone "unifies" `ε` with `FocusEdgePhase`'s | Test 3, plus a comment naming the three references. |
| ~~`phasepack` cannot be installed even once~~ | **Resolved.** Installed once with approval, fixture generated, dependency removed. |
