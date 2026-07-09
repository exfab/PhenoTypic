# The conformal lift — gated research path

> **STATUS: GATED. Not implemented. May never be.**
>
> The mathematics is now understood and corrected (`references.md` §4). What is *not* established is
> that the lift buys anything. It contributes **nothing** to the congruency output, and its one
> remaining deliverable — `φ`, an i1D/i2D discriminant — has three problems that may be fatal.
>
> `lift="conformal"` in `FocusEdgeColorPhase` raises `NotImplementedError` until §4's gate passes.

---

## 1. What the conformal lift is, and what it is for

The plain **monogenic signal** produces three numbers per pixel: an even channel `f` and two odd
(Riesz) channels `h₁, h₂`, giving amplitude, phase, and orientation. Its model is a **plane wave** — a
straight oriented pattern, "intrinsically 1-D" (i1D). Three numbers leave no room for curvature; a
straight edge and a tightly curved one are indistinguishable to it.

The **conformal lift** (Wietzke & Sommer) buys a degree of freedom by changing where the geometry
lives. Stereographic projection maps the plane onto a sphere, and it has one magic property:
**circles and lines in the plane both become circles on the sphere** (lines are the circles through
the north pole). Every circle on a sphere is the intersection of the sphere with a **plane** in `R³`.

So a *curved* isophote in 2-D becomes a *flat* plane in 3-D. Run the same plane-wave machinery in
`R³` — which only ever sees planes — and the curvature reappears as the plane's **tilt**. With unit
normal `ω = (sin φ cos θ, sin φ sin θ, cos φ)`:

- `θ` — the structure's orientation, as before;
- `φ` — the tilt. A straight line's plane passes through the north pole, so `ω₃ = 0` and `φ = π/2`. A
  circle of radius `r` gives `tan φ = 2r`.

That third component, the `Q³` channel, is the entire point: **curvature and i1D/i2D discrimination
out of a quadrature filter, without derivatives.** The pitch for this project was that at a hyphal
crossing Hessian vesselness collapses (both eigenvalues grow), while the conformal signal has a
dedicated coordinate saying "this is i2D."

## 2. Why it disappoints

On this sphere `v₃ = |v|²`, so the third coordinate is *second-order small* near the tangent point,
and the third Riesz channel — which ought to be odd like the other two — comes out **even**.

Measured (`references.md` §9):

```
|corr(c_bp, f_x_bp)| = 0.0008     proper quadrature: even vs odd, uncorrelated
|corr(c_bp, f_z_bp)| = 0.8928     f_z is EVEN  (1175x larger than f_x's)
|corr(f_z, ∇²f)|     = 0.8994     f_z is, to leading order, a LAPLACIAN
```

> The magnitude is configuration-dependent (`0.65` at `s₀ = 0.5`, `0.92` at `s₀*`). The *structure* is
> the claim. Reproduce: `verify_claims.py::check_10`.

`Q̃³` is a positive radially symmetric kernel, so `f ⋆ Q̃³` is a smoothing; the P1 value-removal makes it
`(blur − identity)·f`, i.e. a Laplacian. Within a band the Laplacian is multiplication by `−|ω|²`,
roughly constant, so `f_z_bp` is nearly a scalar multiple of `c_bp`.

Two consequences:

**For `pc`, the lift contributes nothing.** `f_z` adds a rescaled copy of `c` to both `E` and `A_Σ`.
The redundancy holds *at crossings too* (`corr = −0.995` there), so it is not an artifact of averaging
over uninteresting pixels.

**`φ` is not a phase.** `φ = atan2(‖(f_x,f_y,f_z)‖, c)` folds an even channel into the odd magnitude.
The i2D content lives only in the ratio `f_z / √(f_x²+f_y²)` — smoothed Laplacian over gradient
magnitude, the classical isophote-curvature estimator. "Curvature without derivatives" is a derivative
ratio with the derivatives hidden inside the kernel. That is why it needs a length scale `σ`, and why
it measures the radial profile rather than the isophote (§5).

## 3. Three problems with `φ`, in order of severity

**(a) `φ` is undefined on the hyphal centreline.** A hypha is a bright *ridge*, not a step. At a ridge
centre the odd response is identically zero — measured `0.0000` — so `φ` is `0/0` exactly where a
junction label is wanted. It can only be read on the flanking edges, where junction geometry is a mess.

**(b) `φ` cannot separate a corner from a curve.** Not a tuning problem: the conformal signal models
the local structure as **one circle**. A crossing is **two superimposed lines** — out of model. `φ`
measures "not a straight line", and hyphae are curved everywhere.

Measured on step edges (`σ = 8`), `φ/(π/2)` at the boundary:

| structure | `φ/(π/2)` |
|---|---|
| straight step edge | 0.8585 |
| disk boundary `r = 24` | 0.8716 |
| disk boundary `r = 12` | 0.9135 |
| disk boundary `r = 6` | **0.9804** |
| disk boundary `r = 3` | 0.6328 |
| **90° corner** | **0.7997** |

The corner sits inside the spread that curvature alone produces (`|Δφ| = 0.07 … 0.18`).

**(c) `φ` is non-monotone outside `r/σ ≲ 0.35`.** A disk of radius 6 reads *higher* than a straight
edge. That band is Theorem 2's own validity bound; beyond it the plane-wave approximation fails and the
numbers are meaningless.

> These are single-point, unmasked measurements on synthetic step edges. Suggestive, not conclusive —
> but they point the same way as the mechanism in §2.

## 4. The gate

Before `lift="conformal"` is implemented, run a **three-arm junction experiment** on real hyphal
crossings. Ship `FocusEdgeColorPhase` first; nothing here blocks it.

| arm | what it is |
|---|---|
| **A. `φ`** | The conformal i1D/i2D angle, masked to `√(f_x²+f_y²) ≥ ε_g·‖(f_x,f_y,f_z)‖`, `σ` swept |
| **B. Hessian** | The baseline. Vesselness eigenvalues; the failure mode is that both grow at a crossing |
| **C. Signal multi-vector** | Wietzke & Sommer, *The Signal Multi-Vector*, **JMIV 37:132–150 (2010)**. Models **superimposed i1D signals** — which is what a crossing actually is. This is the paper for the junction problem; the conformal monogenic signal is the paper for the *curvature* problem. |

**Metric:** junction precision/recall against hand-marked crossings on `load_fungi_plate()` and the
synthetic filamentous plate's `objmap` skeleton.

**Decision rule:**

- If **A** beats **B**, implement `lift="conformal"` and expose `φ` (masked) as a feature.
- If **C** beats both, the conformal lift is the wrong tool and a separate spec should pursue the
  signal multi-vector.
- If neither beats **B**, **delete the conformal path.** Keep the recovered mathematics in
  `references.md` as the record of why.

Honest prior: the mechanism in §2 and the measurements in §3 argue against **A**. Arm **C** is the one
this project probably wants. But two of this spec's conclusions have already been overturned by
measurement after argument had settled them, so the experiment runs.

## 5. If the gate passes

The construction is fully specified in `references.md` §4.3–4.4. Summary:

```
Step A (components)   c, Q¹, Q² = f ⋆ (P̃, Q̃¹, Q̃²)
                      Q³        = [f − f(x)] ⋆ Q̃³                    ← P1, forced
Step B (scale space)  bⱼ = p_{t·sⱼ} − p_{sⱼ}, planar n=2, unit-mass  ← Reading A, from Eq. (12)
                      sⱼ = 1,2,3,4,  t = 1.5
Feature               tan ϕ = (B/A)·√(f_x²+f_y²)/|f_z|               ← P2, forced
                      A = Σ u₁²wJ, B = Σ u₃²wJ, mask constants
Length scale          σ = pixels per sphere diameter                 ← P3, forced, no source states it
```

**Do not ship `κ = (2/σ)/tan ϕ` as a curvature — it is not curvature.**

An earlier revision descoped `κ` for the wrong reason ("not scale-free"). `κ` **is** scale-covariant:
`κ·r` depends only on `(r/σ, R/σ)`, verified to 4 digits. The `2.04×` spread that revision cited was
measured at `R/σ ≈ 0.5`, a mask too small to contain the isophote — the regime this spec itself calls
invalid.

The real problem is worse. Three radial profiles have **identical** isophote curvature `1/r`
(concentric circles), yet:

| profile | `κ·r` | ratio |
|---|---|---|
| `f = r` (cone) | 1.256 | 1.00 |
| `f = r²` | 2.077 | 1.65 |
| `f = r³` | 3.136 | 2.50 |

Isophote curvature predicts `1, 1, 1`. `Laplacian/|∇f| = f''/f' + 1/r` predicts `1, 2, 3`. The
estimator tracks the latter, and coincides with curvature only when `f'' = 0` — i.e. for a **cone**,
which is exactly JMIV's Fig. 13 test signal.

Even inside JMIV's own signal class (an oscillatory circular signal, where Eq. (89) implies the
frequency cancels), `κ·r` varies **3×** with wavelength at fixed `r`, converging to the cone constant
only as `λ → ∞`. That is a third, independent failure of Eq. (89).

Reproduce: `verify_claims.py::check_09`.

**JMIV Eq. (89) is wrong** and must not be used as printed: it omits a DC term `g(0)·μ₃` and assumes an
isotropy `A = B` that holds only at `s₀* = 0.19269068`. `references.md` §4.3.1.

Also: JMIV's `ω₃` sign is wrong — the lifted isophote satisfies `u₃ = 2⟨u₁₂, m⟩`, so the plane normal
is `(2m₁, 2m₂, −1)`. `|κ|` is unaffected; `κ` flips sign and `θ` rotates by `π`.
