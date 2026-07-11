# `FocusEdgeColorPhase` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `FocusEdgeColorPhase` — per-channel monogenic phase congruency over three colour
channels, combined by a selectable cross-channel fusion — reusing `_monogenic_kernels.py` verbatim
and adding no new signal theory.

**Architecture:** Three scalar channels are extracted from `image.rgb` (via `color.Lab` or
`color.hsv`) in **luminance-first order**. Each is run through the *existing*, unmodified monogenic
chain to produce a set of accumulators. The accumulators are then fused — `joint` (shared
denominator, L1 energy), `coherent` (shared denominator, vector-summed energy), or `l2` (three
independent congruency maps, root-sum-of-squares) — and the result is written to `detect_mat`.
Task 1 refactors `_monogenic_kernels.py` to *expose* those accumulators without changing a single
floating-point value; every later task is additive.

**Tech Stack:** Python 3.12, numpy, pydantic v2, pytest, `uv` as the sole runner.

---

## Global Constraints

Copied verbatim from the specs and from prior decisions on this branch. Every task's requirements
implicitly include this section.

- **Faithfulness to validated reference logic beats a convenient shortcut.** A deviation is
  admissible only if it is (a) forced, (b) recorded in `drift-register.md`, and (c) tested.
- **Before touching any pydantic field, `TuneSpec`, `Field` bound, or `Literal`:** read the
  `adding-an-operation` skill. A new numeric field on an `enhance/` operation is pulled into
  `tests/unit/tune/test_annotation_coverage.py` and **must** carry a `TuneSpec` or a `Field` bound.
- **Before making any claim about a reference:** read the `porting-a-reference-algorithm` skill.
  Cite `file:line`. Never say "the references agree" without opening all three.
- **`uv` is the sole package manager and runner.** Never bare `python` / `pip`.
  `pyproject.toml` and `uv.lock` must be **unchanged** at the end of this plan.
  **`phasepack` must not be installed** when the branch is pushed.
- **Copyrighted PDFs must never be committed.** They live only under
  `~/.claude/refs/phenotypic-alt-phase-detection/papers/`. Verify with
  `git ls-files | grep -c '\.pdf$'` → `0`.
- **`Vivianyuwei/Image-Edge-Detection-Based-on-Conformal-Phase` has no licence** (all rights
  reserved). Read-only for cross-checking geometry; **never copy its code**.
- **Operations are keyword-only pydantic models.** `FocusEdgeColorPhase(fusion="l2")`, never
  positional. No hand-written `__init__`.
- `detect_mat` is `float32` on assignment and must lie in `[0, 1]`.
- **A test that cannot run must fail, not skip.** A test must be *proven able to fail* — reintroduce
  the bug it guards before trusting that it passes.
- **Numeric tolerances come from a mechanism**, not a guess. A tolerance loose enough that the
  anchor could move without the test noticing means the anchor does no work.
- **Sub-agents never commit** and never modify files outside their stated `Files:` block.
- Worktree rules: never `cd` to the original repo root; **never** use bare `git stash` /
  `git stash pop` (the stash stack is shared across worktrees).
- **Reviewer memory ceiling** (an earlier reviewer OOM-killed the machine): no numpy array above
  `5e6` elements, no meshgrid above `2000 × 2000`, total live memory under 500 MB.

  > **`ulimit -v 4000000` does not work on this machine, and never did.** Measured on
  > `Darwin 25.5.0`: `ulimit -v` and `ulimit -d` both fail with
  > `setrlimit: invalid argument`, print to stderr, and **do not abort the command chain** — so
  > `ulimit -v 4000000 && python foo.py` runs `foo.py` with no limit whatsoever. Python's
  > `resource.setrlimit(RLIMIT_AS, …)` fails too (`ValueError: current limit exceeds maximum
  > limit`), even though `getrlimit` reports `INF/INF`. macOS does not enforce an address-space
  > rlimit. Every brief on this branch that said "wrap scratch runs in `ulimit -v 4000000`" was
  > mandating a **no-op that looks like a guard** — including the one added *because* a reviewer
  > OOM-killed the machine.
  >
  > The ceiling is therefore a **discipline, enforced in the script and at review**, not by the
  > shell: assert `arr.size <= 5_000_000` before allocating, never build a meshgrid above
  > `2000 × 2000`, and bound every scratch run with `timeout`. On Linux, `ulimit -v 4000000` does
  > work and remains worth using.

### Constants that must not drift

| Constant | Value | Where |
|---|---|---|
| `EPSILON_MONOGENIC` | `1e-4` | `_monogenic_kernels.py`; `phasecongmono`'s, in all three references |
| `_phasecong3`'s epsilon | `1e-5` | passed **explicitly** into `spread_weight` at the call site |
| `k` | `3.0` | `phasecongmono`'s default, **not** `phasecong3`'s `2.0` |
| `deviation_gain` | `1.5` | `phasecongmono`'s default |
| luminance weight | `1.0`, pinned | spec §4.2 — two chroma degrees of freedom, not three |

---

## What this plan corrects in the spec, before any code is written

Three claims in `color-phase-congruency.md` were re-derived from scratch and **do not reproduce**.
They are fixed in Task 0. They are listed here because an implementer who reads the spec first will
otherwise build against them.

**1. §3.1's `response` column is wrong.** The `ratio` column is exact — `1.0000 / 0.5774 / 0.8247`
reproduce to four digits. The `response` column does not, at any single `deviation_gain`: row 2
would need `dg = 1.0373`, row 3 would need `dg = 1.4265`, and the shipped `dg = 1.5` gives
`0.0000` and `0.0982`. The argument survives and gets **stronger**: at the shipped gain, an
L2-numerator-over-L1-denominator annihilates a coherent three-channel edge *exactly*, not to
`0.0091`.

| firing channels | `√(Σ(wE)²)/Σ(wA)` | response @ `dg=1.5` | `ΣwE/ΣwA` | response @ `dg=1.5` |
|---|---|---|---|---|
| one only | 1.0000 | **1.0000** | 1.0000 | 1.0000 |
| all three, equally | 0.5774 | **0.0000** | 1.0000 | 1.0000 |
| `(0.804, 0.013, 0.183)` | 0.8247 | **0.0982** | 1.0000 | 1.0000 |

> An earlier revision of this very plan printed `0.0983` in that last cell, from round-tripping the
> displayed four-decimal ratio. Exact: `ratio = 0.824665992993527`,
> `response = 0.0982226557669601`. `d(response)/d(ratio) = 2.6520`, and `0.8247` is rounded by
> `3.4e-05`, so the round-trip shifts the fourth decimal. Caught by `fusion_algebra.py` on its first
> run. **Publish the full-precision literal, or publish only the digits the mechanism supports.**

**2. §4.2's "invariant up to `O(ε/A_total)` — ~1%" is false at the shipped `ε`.** The `ε` that
breaks 1-homogeneity is **not** the one in `E_total + ε`. It is the one in `A_max + ε`, which sits
inside `width = (A_total/(A_max + ε) − 1)/(n_scale − 1)` and is then fed to a sigmoid with `g = 10`.
Near the sigmoid's knee a small `width` shift produces a large *relative* change in `W`, and `W`
multiplies the whole output. Measured, over `c ∈ {0.01, 100}` and 20 000 random draws per regime:

| amplitude regime | max relative change |
|---|---|
| `A ∈ [1e-3, 0.5]` | 778.9% |
| `A ∈ [0.05, 0.5]` (edge-pixel scale) | 670.0% |
| `A ∈ [0.5, 5.0]` (Lab `L*` scale) | 100.0% |

So §7's test 4 (*"weight-vector scale invariance to `rtol=2e-2`"*) is, as written, either flaky or
vacuous depending on whether it masks low-response pixels. Task 6 restates it over a masked pixel
set with a tolerance derived from `ε/(c·A_total)` and the sigmoid's local Lipschitz constant.

**3. `E_total/(A_total + ε) ≤ 1` holds for both `joint` and `coherent`,** analytically:
`‖Σₛvᵢₛ‖ ≤ Σₛ‖vᵢₛ‖` per channel gives `E_joint ≤ A_total`, and `‖Σᵢwᵢvᵢ‖ ≤ Σᵢwᵢ‖vᵢ‖` gives
`E_coherent ≤ E_joint`. So the `acos` is safe and `n_clamped == 0` is assertable, exactly as for
`FocusEdgeMonogenicPhase`. Never previously stated; now a test. `fusion_algebra.py` confirms it over
200 000 draws, with sampled maxima `0.996808` and `0.966120` **at seed `20260709`** — sampled
figures, so they travel with their seed or not at all.

**4. §3 and §4.2 contradict each other on HSV channel order.** §3 says `w = (1, chroma_weight_1,
chroma_weight_2)` *"in `color_space` channel order"*. The `hsv` accessor's native order is
`(H, S, V)`, so read literally §3 pins **`H`** at `1.0`. §4.2's table pins **`V`**, and carries the
argument. §4.2 wins. Task 0 replaces "channel order" with an explicit **luminance-first** order:
`lab → (L*, a*, b*)`, `hsv → (V, H, S)`.

**5. Cross-reference rot.** §8's acceptance criteria 1 and 2 cite "§7.1" for per-channel fidelity and
"§7.2" for fusion sanity; those are §7's *numbered list items* 1 and 2, while §7.1 is the PFOM
ranking regression and §7.2 is the chromatic-aberration experiment. Drift row `C7` cites a
non-existent "§8.4.13".

---

## The decision taken before this plan was written

Spec §5 declared `ColorPhaseOutput = Literal["pc", "orientation", "feature_type"]` but never defined
what the two angle maps *mean* after fusion. **Only `coherent` builds a fused monogenic vector.**
`joint` sums scalar energies; `l2` combines three finished congruency maps. So the angle maps would
be unreferenced inventions under the shipped default, not just under `l2`.

**Resolved (user, 2026-07-09):** ship `ColorPhaseOutput = Literal["pc"]`. The protected helper
`FocusEdgeColorPhase._color_phase_congruency()` nevertheless computes and returns `orientation` and
`feature_type` on its `ColorPhaseResult`, taken from the fused vector `Σᵢ wᵢ·vᵢ` — exactly as
`FocusEdgePhase._phasecong3()` returns `orientation` and `feature_type` on `_PhaseCong3Result` while
`output` exposes only `M`/`m`/`pc_sum` (`_focus_edge_phase.py:392-425`). A future consumer can reach
them; nothing is exposed that has no defined meaning; adding an `output` value later is
non-breaking.

Because the fused vector collapses to `v_L` when both chroma weights are `0`, Task 6's per-channel
fidelity test pins the angles against `FocusEdgeMonogenicPhase` **for free**.

---

## How the three operators obtain their angles (read this before Task 2)

All three collapse everything they compute into one 3-vector per pixel and read both angles off it.
Only the middle slot differs.

| | vector | orientation | feature_type |
|---|---|---|---|
| `monogenic_phase_congruency` | `(Σₛeven, Σₛh₁, Σₛh₂)` — Riesz | `arctan2(-v₂, v₁)`, folded | `arctan2(v₀, √(v₁²+v₂²))` |
| `_phasecong3` | `(Σ_θ even, Σ_θ cosθ·odd, Σ_θ sinθ·odd)` — bank projection | `arctan(-v₂/v₁)`, then **reflected** `π/2 − or` (drift M13) | `arctan2(v₀, √(v₁²+v₂²))` |
| `color_phase_congruency` (new) | `(Σᵢwᵢ·evenᵢ, Σᵢwᵢ·h₁ᵢ, Σᵢwᵢ·h₂ᵢ)` — channel sum | `arctan2(-v₂, v₁)`, folded | `arctan2(v₀, √(v₁²+v₂²))` |

Two traps, both already paid for elsewhere on this branch:

- **`√(v₁² + v₂²)`, never `np.hypot`.** `hypot` appears in no reference and disagrees on 4.5% of
  elements. The golden fixture's `rtol=1e-6` cannot see it.
- The fused vector carries `coherent`'s hazard: the odd pair's sign encodes edge polarity, so a
  boundary where `L*` falls as `b*` rises has `v_L` and `v_b` pointing opposite and the sum cancels.
  The angle is least reliable exactly where colour is doing the most work. Record as drift `C15`.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `src/phenotypic/enhance/_monogenic_kernels.py` | **modify** | Expose `MonogenicChannel` + `monogenic_channel_response` + `congruency_from_accumulators`. Zero numeric change. |
| `src/phenotypic/enhance/_color_phase_kernels.py` | **create** | `ColorPhaseResult`, `fuse_joint`, `fuse_coherent`, `fuse_l2`, `color_phase_congruency`. Pure functions, no `Image` dependency. |
| `src/phenotypic/enhance/_focus_edge_color_phase.py` | **create** | The operation. Channel extraction, achromatic guard, `lift` gate, `_color_phase_congruency` helper. |
| `src/phenotypic/sdk_/typing_.py` | **modify** | `ColorSpaceName`, `PhaseFusion`, `PhaseLift`, `ColorPhaseOutput`. |
| `src/phenotypic/enhance/__init__.py` | **modify** | Export. `ImageEnhancer` subclasses in `__all__`: 30 → 31 (`__all__` itself 31 → 32; `SetDetectMode` is an `ImageOperation`, not an enhancer). |
| `tests/unit/enhance/test_monogenic_kernels.py` | **modify** | Bit-identity of the refactor. |
| `tests/unit/enhance/test_color_phase_kernels.py` | **create** | Fusion algebra, `[0,1]` bound, `n_clamped == 0`. |
| `tests/unit/enhance/test_focus_edge_color_phase.py` | **create** | Operation contract, fidelity, invariance, guards, round-trip. |
| `tests/unit/enhance/test_color_phase_pfom.py` | **create** | §7.1 ranking regression. Slow; own file so it can run parallel to the CA experiment. |
| `docs/superpowers/logic_validation_scripts/2026-07-09-focus-edge-color-phase/fusion_algebra.py` | **create** | Re-derives §3.1, the `E ≤ A` bound, and the ε-homogeneity bound. numpy only; never imports `phenotypic`. |
| `docs/superpowers/plans/2026-07-09-focus-edge-color-phase/experiments/chromatic_aberration.py` | **create** | §7.2. Imports `phenotypic`, so it is **not** a logic-validation script. |
| `docs/superpowers/specs/2026-07-08-alt-phase-detection/color-phase-congruency.md` | **modify** | The five corrections above. |
| `docs/superpowers/specs/2026-07-08-alt-phase-detection/drift-register.md` | **modify** | Rows `C15`–`C17`; fix `C7`'s dangling cite. |
| `src/phenotypic/enhance/CLAUDE.md` | **modify** | Colour section; the `rgb`-source warning. |
| `src/phenotypic/abc_/_enhance_markers/_focus_edge.py` | **modify** | Note that one subclass sources from `rgb`. |
| `docs/source/explanation/what_enhancement_does.md` | **modify** | Add the operation. |
| `docs/superpowers/logic_validation_scripts/2026-07-09-focus-edge-monogenic-phase/math_review_corpus.py` | **modify** | Extend the strip pipeline to the colour files; add gate 8. |

**Never touched by any task in this plan:** `_focus_edge_phase.py`, `tests/fixtures/*.npz`,
`pyproject.toml`, `uv.lock`.

---

## Execution: cluster-and-isolate DAG

Derived from the `Files:`/`Interfaces:` blocks below, not from task numbering.

```
T0  Spec corrections                     Seam    Opus/high
     │  (an implementer reading §3 pins the wrong HSV channel)
     ├──────────────┐
     ▼              ▼
T1  Kernel        T3  fusion_algebra.py            ← PARALLEL (zero file overlap)
    refactor          Leaf    Sonnet/medium
    Seam+Keystone
    Opus/high
     │              │
     ▼              │
  ╔═ GATE A ═╗      │   deep review, Opus/high — BIT-IDENTITY PROOF, blocking
     │              │
     ▼              │
T2  Fusion kernels ◄┘   Keystone  Opus/high
     │
     ▼
T4  The operation       Keystone  Opus/high
     │
     ▼
  ╔═ GATE B ═╗   deep review over T2+T3+T4 combined diff, Opus/high
     │
     ├──────────────┐
     ▼              ▼
T5  CA experiment  T6  PFOM + contract tests       ← PARALLEL (zero file overlap)
    Seam  Opus/high    Keystone  Opus/high
     │              │
     ▼              │
  ╔═ GATE C ═╗      │   DECISION. May flip `fusion` default joint → l2.
     │              │   Surface to the user if it does.
     └──────┬───────┘
            ▼
T7  Docs sweep          Sweep  Sonnet/medium   (depends on T5: the default may have moved)
            │
            ▼
  ╔═ GATE D ═╗   light: diff + full affected suite + mypy + ruff
            │
            ▼
    Simplify pass       Opus/high
            │
            ▼
  ╔══════════════ FINAL GATE ══════════════╗
E   Extend the stripped sandbox   Seam+Sweep  Opus/high
            │
            ▼
F   Scoped Fable review           Fable 5 / high effort
            │
            ▼
    Triage → apply to the real tree → regression suite
```

**Parallel pairs** (verified write-disjoint): `T1 ∥ T3`, `T5 ∥ T6`. Both run in isolated worktrees
with `worktree.baseRef: "head"` — agent worktrees must branch from **local** HEAD, because
`origin/main` has none of this.

**Everything else is sequential**, because each step consumes the previous step's *interface*, not
merely its files.

**Model rule:** never review or verify with a weaker model than implemented. Every gate is Opus/high.
`T3` and `T7` are the only Sonnet-tier tasks and both are reviewed at an Opus-tier gate.

---

## Task 0: Correct the spec before anyone builds against it

**Shape:** Seam. Tiny, no code, and it gates everything.

**Files:**
- Modify: `docs/superpowers/specs/2026-07-08-alt-phase-detection/color-phase-congruency.md`
- Modify: `docs/superpowers/specs/2026-07-08-alt-phase-detection/drift-register.md`

**Interfaces:**
- Consumes: nothing.
- Produces: the **luminance-first channel order** that Task 4 implements, and the
  `ColorPhaseOutput = Literal["pc"]` surface that Tasks 2 and 4 build.

- [ ] **Step 1: Replace §3.1's table with the re-derived one**

Use the four-column table from *"What this plan corrects in the spec"* above. Add one sentence:

> Re-derived 2026-07-09. An earlier revision of this table printed responses of `0.0091` and
> `0.1425`, which no single `deviation_gain` produces (they would need `1.0373` and `1.4265`). At the
> shipped `deviation_gain = 1.5` the coherent three-channel edge is annihilated **exactly**. The
> conclusion is unchanged and strengthened. Re-derived on demand by
> `logic_validation_scripts/2026-07-09-focus-edge-color-phase/fusion_algebra.py`.

- [ ] **Step 2: Replace §4.2's homogeneity claim**

Delete *"up to `O(ε/A_total)`… ~1% over `c ∈ [0.01, 100]`"*. Replace with:

> `E_total`, `A_total`, `T_total`, `A_max` are 1-homogeneous in `w`, so the ratio is invariant to a
> global rescale **only in the limit `ε → 0`**. The `ε` that breaks it is the one in `A_max + ε`, not
> the one in `E_total + ε`: it sits inside `width = (A_total/(A_max + ε) − 1)/(n_scale − 1)`, which
> is fed to a sigmoid of sharpness `g = 10`. Near the knee, a small shift in `width` moves `W` by a
> large *relative* amount, and `W` multiplies the entire output. Measured over `c ∈ {0.01, 100}`:
> **100% relative change** at `A ∈ [0.5, 5.0]`, rising to **779%** at `A ∈ [1e-3, 0.5]`. Any
> invariance test must therefore mask low-response pixels and derive its tolerance from
> `ε/(c·A_total)` and the local slope of `W`. See `fusion_algebra.py`, and drift `C17`.

- [ ] **Step 3: Fix the HSV channel-order contradiction**

In §3, replace *"in `color_space` channel order"* with *"in **luminance-first** order (§4.2)"*. In
§4.2, after the table, add:

> The accessors' native orders are `Lab → (L*, a*, b*)` and `hsv → (H, S, V)`. This operation
> reorders `hsv` to `(V, H, S)` so that index `0` is always the pinned luminance axis. Read literally,
> an earlier §3 (*"in `color_space` channel order"*) pinned **`H`** at `1.0` under `hsv`. Pinned by
> `test_focus_edge_color_phase.py::TestChannelOrderIsLuminanceFirst`.

- [ ] **Step 4: Narrow `ColorPhaseOutput` in §5, and document the protected helper**

```markdown
| `output` | `"pc"` | `ColorPhaseOutput` | — |

ColorPhaseOutput = Literal["pc"]
```

> **Only `pc` is exposed.** Of the three fusion modes only `coherent` builds a fused monogenic
> vector; `joint` sums scalar energies and `l2` combines three finished congruency maps. So
> `orientation` and `feature_type` would be unreferenced inventions under the shipped default, not
> merely under `l2`.
>
> They are nevertheless **computed and returned** by the protected helper
> `FocusEdgeColorPhase._color_phase_congruency()`, on its `ColorPhaseResult`, taken from the fused
> vector `Σᵢ wᵢ·vᵢ` — mirroring `FocusEdgePhase._phasecong3()`, which returns `orientation` and
> `feature_type` while `output` exposes only `M`/`m`/`pc_sum`. A future consumer can reach them
> without a breaking change. Drift `C15`.

- [ ] **Step 5: Restate §7 test 4**

> 4. **Weight-vector scale invariance, over a masked pixel set.** Compare `out(w)` against
>    `out(c·w)` for `c ∈ {0.01, 100}` on `load_synth_yeast_plate()`, restricted to pixels where
>    `out(w) > 0.05`. The tolerance is **derived, not chosen**: `ε/(c·A_total)` bounds the
>    `E_total + ε` term, and `|dW/d(width)| ≤ g/4` bounds the sigmoid term. Assert the measured
>    deviation against that bound, and assert the bound itself is tighter than `0.05` on this image —
>    otherwise the test cannot fail. **Unmasked, this comparison changes by up to 779%** (§4.2).

- [ ] **Step 6: Fix §8's cross-references and `C7`'s dangling cite**

§8.1 → "§7 test 1 (per-channel fidelity)". §8.2 → "§7 test 2 (fusion sanity)". §8.4 → "§7.2's CA
experiment". In `drift-register.md`, `C7`: replace "§8.4.13" with
"`color-phase-congruency.md` §7.2".

- [ ] **Step 7: Add drift rows C15, C16, C17**

Append to the `FocusEdgeColorPhase` table. **Check the table stays contiguous — a blank line before a
row orphans it** (this happened to `M12`).

```markdown
| C15 | `_color_phase_congruency` returns `orientation` / `feature_type` from the fused vector `Σᵢwᵢ·vᵢ`; neither is exposed via `output` | no reference; CMPCM emits one edge map | CAPABILITY, internal | Only `coherent` builds a fused vector natively — `joint` sums scalar energies, `l2` sums finished maps — so under the shipped default these angles are computed from a quantity the response never touches. Precedent: `_phasecong3` already reports angles off an `energy_v` that did not produce its `pc_sum` (`_focus_edge_phase.py:344-347, 392`). **Hazard:** the odd pair's sign encodes edge polarity, so an anti-correlated chromatic edge (`L*` falling as `b*` rises) cancels in the sum — the angle is least reliable exactly where colour matters most. Not exposed, so nothing consumes it. Pinned at `chroma_weight_* = 0`, where the fused vector collapses to `v_L` and the angles must equal `FocusEdgeMonogenicPhase`'s bit-for-bit. |
| C16 | `color_space="hsv"` bandpasses **raw** hue, across its wrap discontinuity | CMPCM uses HSV and does not unwrap | CAPABILITY, recorded hazard | `H` is circular on `[0, 1)`. A log-Gabor bandpass across the `0.99 → 0.01` seam sees a unit step where the colour is continuous, so near-red boundaries manufacture a phantom edge. We do **not** unwrap: the paper does not, and `("hsv", "l2")` is §7.1's regression configuration, which must exercise the paper's actual quantity. `color_space="lab"` is the default and has no seam. Demonstrated, not merely asserted, by `test_focus_edge_color_phase.py::TestHueWrapArtifactIsReal`. |
| C17 | The `[0,1]` output is **not** invariant to a global rescale of `w` | §4.2 of an earlier revision claimed "~1%" | **CORRECTED — the claim was false** | The offending `ε` is in `A_max + ε`, inside a sigmoid of sharpness `g = 10`, not in `E_total + ε`. Measured over `c ∈ {0.01, 100}`, 20 000 draws per regime: **100%** relative change at `A ∈ [0.5, 5.0]`, **670%** at `A ∈ [0.05, 0.5]`, **779%** at `A ∈ [1e-3, 0.5]`. The two chromatic degrees of freedom of §4.2 survive — the *ratio* is still 1-homogeneous — but a test asserting invariance must mask low-response pixels and derive its tolerance. Re-derived by `fusion_algebra.py`. |
```

- [ ] **Step 8: Commit**

```bash
git add docs/superpowers/specs/2026-07-08-alt-phase-detection/
git commit -m "docs(spec)!: correct color-PC §3.1, §4.2, HSV channel order; narrow ColorPhaseOutput

Three claims did not reproduce. §3.1's response column matches no single
deviation_gain. §4.2's '~1%' understates the true figure by 100x -- the
offending epsilon is in A_max + eps, inside a g=10 sigmoid. §3 and §4.2
disagreed on which HSV channel is pinned.

ColorPhaseOutput narrows to Literal['pc']: only 'coherent' builds a fused
monogenic vector, so the angle maps would be unreferenced inventions under
the shipped default. They move to the protected helper's result object.

Drift C15, C16, C17."
```

---

## Task 1: Expose the per-channel accumulators, without moving one bit

**Shape:** Seam **and** Keystone. The riskiest task in this plan. `_monogenic_kernels.py` is a port
pinned by two fixtures and consumed by two shipped operations.

**Files:**
- Modify: `src/phenotypic/enhance/_monogenic_kernels.py`
- Modify: `tests/unit/enhance/test_monogenic_kernels.py`

**Interfaces:**
- Consumes: nothing.
- Produces, for Task 2:
  ```python
  @dataclass(frozen=True)
  class MonogenicChannel:
      sum_even: np.ndarray       # (rows, cols) float64
      sum_h1: np.ndarray
      sum_h2: np.ndarray
      sum_amplitude: np.ndarray  # A_Sigma
      max_amplitude: np.ndarray  # max over scales
      threshold: float           # T
      @property
      def energy(self) -> np.ndarray: ...

  def monogenic_channel_response(
      img: np.ndarray, *, n_scale: int = 4, min_wavelength: float = 3.0,
      mult: float = 2.1, sigma_onf: float = 0.55, k: float = 3.0,
      noise_method: float = -1.0, periodic: bool = False,
  ) -> MonogenicChannel: ...

  def congruency_from_accumulators(
      energy: np.ndarray, sum_amplitude: np.ndarray, max_amplitude: np.ndarray,
      threshold: float, *, n_scale: int, cutoff: float, g: float,
      deviation_gain: float, epsilon: float = EPSILON_MONOGENIC,
  ) -> tuple[np.ndarray, int]: ...   # (pc, n_clamped)
  ```
  `monogenic_phase_congruency` keeps its exact signature and return type. It becomes a composition of
  the two new functions plus the two angle lines.

**Why the split lands where it does.** `cutoff`, `g` and `deviation_gain` are *congruency*
parameters, not *accumulator* parameters — they never touch the scale loop. `k`, `mult` and
`noise_method` are needed for `threshold`, so they stay with the accumulators. Fusion needs the
accumulators of three channels and one congruency evaluation, so the seam must fall exactly here.

- [ ] **Step 1: Write the bit-identity test first — against an INDEPENDENT oracle**

> **The version of this step that shipped in the first draft of this plan was a tautology.** It
> compared `monogenic_phase_congruency` against `monogenic_channel_response` +
> `congruency_from_accumulators` — which is exactly what the refactored function *calls*. Both sides
> move together. **Measured: with `np.hypot` substituted into `MonogenicChannel.energy`, that test
> passes 9/9.** It could not fail.
>
> The oracle must be an **independent restatement of the pre-refactor body** —
> `_monogenic_pc_from_primitives`, built only from `construct_filter_grids`, `log_gabor_scale`,
> `riesz_multiplier`, `spread_weight` and `rayleigh_mode`, calling none of the refactored API
> (assert this with `ast`, not by reading). Under the same mutation it fails **8/9**. Its scope is
> the accumulator seam and the congruency block; a mutation *inside* a shared primitive is invisible
> to it, and each primitive has its own test class.

This is the oracle for the whole task. Append to `tests/unit/enhance/test_monogenic_kernels.py` a
`_monogenic_pc_from_primitives(img, n_scale, noise_method)` helper returning
`(pc, threshold, n_clamped, sum_even, sum_h1, sum_h2, sum_amplitude, max_amplitude)`, reproducing the
pre-refactor statement order exactly — `weight` before `threshold`, and the `if s == 0:` branch
reading `sum_amplitude` rather than `amplitude`. Then compare against it:

```python
class TestTheRefactorMovesNoBits:
    """`monogenic_phase_congruency` must be bit-identical across the accumulator split.

    Not `rtol`. Bit-identical. The golden fixture's `rtol=1e-6` has 6.7 orders of slack
    and has already hidden two substitutions on this branch (`np.hypot` for
    `sqrt(a**2+b**2)`, and numpy's reciprocal-multiply for a componentwise divide).
    A refactor that reorders a float accumulation would sail straight through it.
    """

    @pytest.mark.parametrize("noise_method", [-1.0, -2.0, 0.0])
    @pytest.mark.parametrize("n_scale", [2, 4, 5])
    def test_pc_orientation_feature_type_and_T_are_bit_identical(self, n_scale, noise_method):
        rng = np.random.default_rng(20260709)
        img = rng.normal(size=(64, 64))
        img += np.add.outer(np.arange(64) > 31, np.zeros(64))  # a step, so T is non-degenerate

        result = monogenic_phase_congruency(
            img, n_scale=n_scale, noise_method=noise_method
        )

        # INDEPENDENT oracle. Never `monogenic_channel_response` + `congruency_from_
        # accumulators` -- that composition IS the function under test.
        (expected_pc, expected_threshold, expected_n_clamped, sum_even, sum_h1, sum_h2,
         sum_amplitude, max_amplitude) = _monogenic_pc_from_primitives(
            img, n_scale, noise_method
        )

        assert np.array_equal(result.pc, expected_pc), "pc moved"
        assert result.threshold == expected_threshold, "T moved"
        assert result.n_clamped == expected_n_clamped

        orientation = np.arctan2(-sum_h2, sum_h1)
        orientation = np.where(orientation > np.pi / 2, orientation - np.pi, orientation)
        orientation = np.where(orientation <= -np.pi / 2, orientation + np.pi, orientation)
        assert np.array_equal(result.orientation, orientation), "orientation moved"

        feature_type = np.arctan2(sum_even, np.sqrt(sum_h1 ** 2 + sum_h2 ** 2))
        assert np.array_equal(result.feature_type, feature_type), "feature_type moved"

        # The seam itself: the exported functions expose exactly the accumulators the
        # monolith held.
        channel = monogenic_channel_response(img, n_scale=n_scale, noise_method=noise_method)
        assert np.array_equal(channel.sum_even, sum_even)
        assert np.array_equal(channel.sum_amplitude, sum_amplitude)
```

**Keep all nine parametrisations.** Under the `np.hypot` mutation, `[n_scale=2,
noise_method=-2.0]` stays **green** even against the independent oracle. A single-configuration
version of this test would have missed the mutation entirely.

- [ ] **Step 2: Run it against the un-refactored tree — it must fail with `NameError`**

```bash
uv run pytest tests/unit/enhance/test_monogenic_kernels.py::TestTheRefactorMovesNoBits -x -q
```
Expected: `NameError: name 'monogenic_channel_response' is not defined` (collection error). If it
reports anything else, stop — the test file is not importing what you think it is.

- [ ] **Step 3: Extract `MonogenicChannel` and `monogenic_channel_response`**

Lift lines `490–541` of `_monogenic_kernels.py` verbatim. **Preserve statement order exactly**, including
the `if s == 0:` branch that reads `sum_amplitude` (not `amplitude`) for `tau` — Kovesi reads the
accumulator, and at `s == 0` they are equal, which is why it survives.

```python
@dataclass(frozen=True)
class MonogenicChannel:
    """Per-channel monogenic accumulators, before the congruency formula is applied.

    Split out of :func:`monogenic_phase_congruency` so that
    :class:`FocusEdgeColorPhase` can fuse three channels *before* evaluating the
    congruency once, rather than evaluating it three times and combining after
    (which is what ``fusion="l2"`` does, and is a different operator).

    Attributes:
        sum_even: Sum over scales of the even (log-Gabor) response.
        sum_h1: Sum over scales of the first Riesz (odd) response.
        sum_h2: Sum over scales of the second Riesz (odd) response.
        sum_amplitude: ``A_Sigma``, sum over scales of the monogenic amplitude.
        max_amplitude: Elementwise maximum over scales of the monogenic amplitude.
        threshold: The Rayleigh noise threshold ``T`` for this channel.
    """

    sum_even: np.ndarray
    sum_h1: np.ndarray
    sum_h2: np.ndarray
    sum_amplitude: np.ndarray
    max_amplitude: np.ndarray
    threshold: float

    @property
    def energy(self) -> np.ndarray:
        """``||(sum_even, sum_h1, sum_h2)||``.

        ``sqrt(a**2 + b**2 + c**2)``, never ``np.hypot`` -- ``hypot`` appears in no
        reference and rounds differently on **~21%** of elements (21.4% on
        ``load_synth_yeast_plate``'s ``L*``, 19.3% on 64x64 gaussian noise). Not 4.5%:
        that is the *two*-component ``sqrt(h1**2+h2**2)`` figure, which does not transfer.
        """
        return np.sqrt(self.sum_even ** 2 + self.sum_h1 ** 2 + self.sum_h2 ** 2)


def monogenic_channel_response(
        img: np.ndarray,
        *,
        n_scale: int = 4,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_onf: float = 0.55,
        k: float = 3.0,
        noise_method: float = -1.0,
        periodic: bool = False,
) -> MonogenicChannel:
    """Run the monogenic scale loop and return its accumulators.

    Everything :func:`monogenic_phase_congruency` computes *before* the congruency
    formula. Guards (``n_scale``, ``mult``, ``noise_method``) live here, because this
    is where their divisions are; ``sigma_onf`` is guarded inside
    :func:`log_gabor_scale`, at its own division.

    Args:
        img: Real 2-D array.
        n_scale: Number of log-Gabor scales. Must be at least 2.
        min_wavelength: Wavelength of the finest scale, in pixels.
        mult: Wavelength multiplier between successive scales. Must exceed 1.
        sigma_onf: Ratio of each filter's Gaussian sigma to its centre frequency.
        k: Number of noise standard deviations above the mean at which ``T`` sits.
        noise_method: ``-1`` median, ``-2`` Rayleigh mode, ``>= 0`` a literal ``T``.
        periodic: Bandpass the periodic component (Kovesi's MATLAB) rather than the raw
            FFT (his Julia, which we ship). Drift ``M4``.

    Returns:
        A :class:`MonogenicChannel`.

    Raises:
        ValueError: If ``n_scale < 2`` (M9), ``mult <= 1`` (M10), or ``noise_method`` is
            negative and neither ``-1`` nor ``-2`` (M7).
    """
    # ... lines 446-541 of the pre-refactor file, verbatim, ending:
    return MonogenicChannel(
            sum_even, sum_h1, sum_h2, sum_amplitude, max_amplitude, threshold
    )
```

- [ ] **Step 4: Extract `congruency_from_accumulators`**

Lift lines `543–550` verbatim. `spread_weight` moves inside it. Note the argument order: `weight`
is computed *first* in the original, and float operations on independent arrays do not depend on
statement order, so this is safe — but do not fold `weight` into the `pc` expression, which *would*
change association.

```python
def congruency_from_accumulators(
        energy: np.ndarray,
        sum_amplitude: np.ndarray,
        max_amplitude: np.ndarray,
        threshold: float,
        *,
        n_scale: int,
        cutoff: float,
        g: float,
        deviation_gain: float,
        epsilon: float = EPSILON_MONOGENIC,
) -> tuple[np.ndarray, int]:
    """Kovesi's ``phasecongmono`` congruency, given accumulators from any source.

    ``PC = W * max(1 - deviation_gain*acos(E/(A + eps)), 0) * max(E - T, 0)/(E + eps)``

    The accumulators may come from one channel (:func:`monogenic_channel_response`) or
    from a weighted fusion of several (:mod:`_color_phase_kernels`). The formula does
    not care, which is the entire reason for the split.

    Args:
        energy: ``E``, the monogenic energy.
        sum_amplitude: ``A_Sigma``.
        max_amplitude: Elementwise max over scales.
        threshold: ``T``.
        n_scale: Number of scales; ``spread_weight`` divides by ``n_scale - 1``.
        cutoff: Frequency-spread sigmoid centre.
        g: Frequency-spread sigmoid sharpness.
        deviation_gain: Scales the phase-deviation term.
        epsilon: Division guard. ``1e-4`` for ``phasecongmono``.

    Returns:
        ``(pc, n_clamped)``. ``n_clamped`` counts pixels whose ``acos`` argument had to
        be clipped into ``[-1, 1]``; it must be ``0`` (drift ``M1``).
    """
    weight = spread_weight(sum_amplitude, max_amplitude, n_scale, cutoff, g, epsilon)

    ratio = energy / (sum_amplitude + epsilon)
    n_clamped = int(np.count_nonzero((ratio > 1.0) | (ratio < -1.0)))
    phase_deviation = np.maximum(
            1.0 - deviation_gain * np.arccos(np.clip(ratio, -1.0, 1.0)), 0.0
    )
    pc = weight * phase_deviation * np.maximum(energy - threshold, 0.0) / (energy + epsilon)
    return pc, n_clamped
```

- [ ] **Step 5: Rebuild `monogenic_phase_congruency` on the two new functions**

Its signature, docstring, `MonogenicResult` and the two angle lines are unchanged.

```python
def monogenic_phase_congruency(img, *, n_scale=4, min_wavelength=3.0, mult=2.1,
                               sigma_onf=0.55, k=3.0, cutoff=0.5, g=10.0,
                               deviation_gain=1.5, noise_method=-1.0,
                               periodic=False) -> MonogenicResult:
    channel = monogenic_channel_response(
            img, n_scale=n_scale, min_wavelength=min_wavelength, mult=mult,
            sigma_onf=sigma_onf, k=k, noise_method=noise_method, periodic=periodic,
    )
    energy = channel.energy
    pc, n_clamped = congruency_from_accumulators(
            energy, channel.sum_amplitude, channel.max_amplitude, channel.threshold,
            n_scale=n_scale, cutoff=cutoff, g=g, deviation_gain=deviation_gain,
            epsilon=EPSILON_MONOGENIC,
    )

    orientation = np.arctan2(-channel.sum_h2, channel.sum_h1)
    orientation = np.where(orientation > np.pi / 2, orientation - np.pi, orientation)
    orientation = np.where(orientation <= -np.pi / 2, orientation + np.pi, orientation)

    feature_type = np.arctan2(
            channel.sum_even, np.sqrt(channel.sum_h1 ** 2 + channel.sum_h2 ** 2)
    )

    return MonogenicResult(pc, orientation, feature_type, channel.threshold, n_clamped)
```

- [ ] **Step 6: Run the bit-identity test, the golden fixture, and both operations**

```bash
uv run pytest tests/unit/enhance/ -q
```
Expected: all green, including `TestTheRefactorMovesNoBits` (9 parametrisations),
`phasecongmono_golden.npz` at `rtol=1e-6`, and `phasecong3_characterization.npz` at
`rtol=1e-9, atol=1e-12`.

- [ ] **Step 7: Prove the bit-identity test can fail**

Temporarily change `channel.energy` to `np.hypot(np.hypot(self.sum_even, self.sum_h1), self.sum_h2)`.

```bash
uv run pytest tests/unit/enhance/test_monogenic_kernels.py::TestTheRefactorMovesNoBits -q
```
Expected: **FAIL** on `pc moved`. Then revert. If it passes, the test is not comparing what you
think — `np.array_equal` on `float64`, not on a rounded view.

Also confirm the *golden fixture* does **not** catch that mutation. It won't; `rtol=1e-6` has 6.7
orders of slack. That asymmetry is the reason this test exists, and belongs in the commit message.

- [ ] **Step 8: Confirm the two shipped operations are untouched**

```bash
uv run pytest tests/unit/enhance/test_phase_congruency.py \
              tests/unit/enhance/test_focus_edge_monogenic_phase.py \
              tests/unit/detect/test_filamentous_fungi_detector.py -q
uv run ruff check src/phenotypic/enhance/_monogenic_kernels.py
uv run mypy src/phenotypic/enhance/_monogenic_kernels.py
```

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/enhance/_monogenic_kernels.py tests/unit/enhance/test_monogenic_kernels.py
git commit -m "refactor(enhance): split monogenic_phase_congruency at the accumulator seam

MonogenicChannel + monogenic_channel_response + congruency_from_accumulators.
Composition is bit-identical: TestTheRefactorMovesNoBits asserts np.array_equal
on pc, T and n_clamped across 9 parametrisations. Proven able to fail by
substituting np.hypot for sqrt(a**2+b**2+c**2) -- which the golden fixture's
rtol=1e-6 does NOT catch.

FocusEdgeColorPhase needs three channels' accumulators fused before one
congruency evaluation. That is not the same operator as three congruencies
combined afterwards (fusion='l2'), which is why the seam falls here."
```

---

## Task 2: The fusion kernels

**Shape:** Keystone. New file, so its diff is reviewable on its own.

**Files:**
- Create: `src/phenotypic/enhance/_color_phase_kernels.py`
- Create: `tests/unit/enhance/test_color_phase_kernels.py`

**Interfaces:**
- Consumes: `MonogenicChannel`, `congruency_from_accumulators`, `EPSILON_MONOGENIC` (Task 1).
- Produces, for Task 4:
  ```python
  @dataclass(frozen=True)
  class ColorPhaseResult:
      pc: np.ndarray
      orientation: np.ndarray     # radians, (-pi/2, pi/2]
      feature_type: np.ndarray    # radians, [-pi/2, pi/2]
      threshold: float
      n_clamped: int

  def color_phase_congruency(
      channels: Sequence[MonogenicChannel], weights: np.ndarray, *,
      fusion: str, n_scale: int, cutoff: float = 0.5, g: float = 10.0,
      deviation_gain: float = 1.5, epsilon: float = EPSILON_MONOGENIC,
  ) -> ColorPhaseResult: ...
  ```
  **`pc` is returned un-clipped.** §7.1's PFOM regression needs the paper's actual quantity, whose
  range under `l2` is `[0, ||w||]`. Task 4 clips at the write site. Drift `C3`.

- [ ] **Step 1: Write the failing tests**

```python
"""Cross-channel fusion of monogenic phase congruency."""
import numpy as np
import pytest

from phenotypic.enhance._color_phase_kernels import (
    ColorPhaseResult, color_phase_congruency, fuse_coherent, fuse_joint, fuse_l2,
)
from phenotypic.enhance._monogenic_kernels import (
    MonogenicChannel, monogenic_channel_response, monogenic_phase_congruency,
)


def _channels(rng, rows=48, cols=48, n_scale=4):
    """Three channels from three genuinely different images."""
    base = np.add.outer(np.zeros(rows), np.arange(cols) > cols // 2).astype(float)
    return [
        monogenic_channel_response(base + 0.1 * rng.normal(size=(rows, cols)), n_scale=n_scale)
        for _ in range(3)
    ]


class TestZeroChromaReducesToTheMonogenicPort:
    """With both chroma weights at 0, every fusion mode must return the luminance port."""

    @pytest.mark.parametrize("fusion", ["joint", "coherent", "l2"])
    def test_pc_is_bit_identical_to_monogenic_phase_congruency(self, fusion):
        rng = np.random.default_rng(7)
        img = np.add.outer(np.zeros(48), np.arange(48) > 24).astype(float)
        img += 0.05 * rng.normal(size=(48, 48))

        chans = [monogenic_channel_response(img)] + [
            monogenic_channel_response(rng.normal(size=(48, 48))) for _ in range(2)
        ]
        fused = color_phase_congruency(
            chans, np.array([1.0, 0.0, 0.0]), fusion=fusion, n_scale=4
        )
        reference = monogenic_phase_congruency(img)

        assert np.array_equal(fused.pc, reference.pc)
        assert np.array_equal(fused.orientation, reference.orientation)
        assert np.array_equal(fused.feature_type, reference.feature_type)


class TestTheAcosArgumentNeverLeavesTheUnitInterval:
    """E_total <= A_total for joint and coherent, so `n_clamped` must be 0.

    Analytic, not empirical: ||sum_s v_is|| <= sum_s ||v_is|| per channel gives
    E_joint <= A_total, and ||sum_i w_i v_i|| <= sum_i w_i ||v_i|| gives
    E_coherent <= E_joint. `fusion_algebra.py` check 03 confirms it over 200_000 draws.
    """

    @pytest.mark.parametrize("fusion", ["joint", "coherent", "l2"])
    def test_n_clamped_is_zero(self, fusion):
        rng = np.random.default_rng(11)
        chans = _channels(rng)
        w = np.array([1.0, rng.uniform(0, 8), rng.uniform(0, 8)])
        assert color_phase_congruency(chans, w, fusion=fusion, n_scale=4).n_clamped == 0


class TestFusionSanity:
    """§7 test 2. An edge in all three channels must outscore the same edge in one.

    The L2-numerator-over-L1-denominator form fails this by ~inf: at
    `deviation_gain=1.5` it annihilates the three-channel edge exactly (spec §3.1).
    """

    def test_three_coherent_channels_outscore_one(self):
        edge = np.add.outer(np.zeros(48), np.arange(48) > 24).astype(float)
        flat = np.zeros((48, 48))

        all_three = [monogenic_channel_response(edge) for _ in range(3)]
        one_only = [monogenic_channel_response(edge)] + [
            monogenic_channel_response(flat) for _ in range(2)
        ]
        w = np.ones(3)

        a = color_phase_congruency(all_three, w, fusion="joint", n_scale=4).pc.max()
        b = color_phase_congruency(one_only, w, fusion="joint", n_scale=4).pc.max()
        assert a > b


class TestL2IsNotDividedByTheWeightNorm:
    """Drift C3. The paper does not normalise; §7.1 must see the paper's quantity."""

    def test_l2_can_exceed_one_before_clipping(self):
        edge = np.add.outer(np.zeros(48), np.arange(48) > 24).astype(float)
        chans = [monogenic_channel_response(edge) for _ in range(3)]
        out = color_phase_congruency(chans, np.ones(3), fusion="l2", n_scale=4).pc
        assert out.max() > 1.0, "l2 must return the un-clipped root-sum-of-squares"
```

- [ ] **Step 2: Run them — collection must fail**

```bash
uv run pytest tests/unit/enhance/test_color_phase_kernels.py -q
```
Expected: `ModuleNotFoundError: No module named 'phenotypic.enhance._color_phase_kernels'`.

- [ ] **Step 3: Implement `_color_phase_kernels.py`**

```python
"""Cross-channel fusion of monogenic phase congruency.

Three colour channels, each run through the *unmodified* monogenic chain of
:mod:`_monogenic_kernels`, then combined. Adds no signal theory: every fusion rule
here is arithmetic over accumulators that module already produces.

Only ``fuse_l2`` has a reference -- Shi et al., "Colour edge detection based on the
CMPCM", *Multimed. Tools Appl.* 78, 10701--10716 (2019). ``joint`` and ``coherent`` are
ours, recorded as drift ``C7`` and ``C8``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ._monogenic_kernels import (
    EPSILON_MONOGENIC, MonogenicChannel, congruency_from_accumulators,
)

_FUSIONS = ("joint", "coherent", "l2")


@dataclass(frozen=True)
class ColorPhaseResult:
    """Output of :func:`color_phase_congruency`.

    Attributes:
        pc: Fused congruency, **un-clipped**. ``joint`` and ``coherent`` land in
            ``[0, 1]``; ``l2`` lands in ``[0, ||w||]``. The caller clips at the write
            site, because §7.1's PFOM regression needs the paper's actual quantity
            (drift ``C3``).
        orientation: Feature orientation in radians, ``(-pi/2, pi/2]``, from the fused
            vector ``sum_i w_i * v_i``. ``0`` is a vertical edge. **Not exposed by
            `FocusEdgeColorPhase.output`** -- drift ``C15``. Under ``fusion="l2"`` and
            ``fusion="joint"`` this vector is not what produced ``pc``.
        feature_type: Local weighted mean phase angle, ``[-pi/2, pi/2]``, from the same
            fused vector. ``0`` is a step edge.
        threshold: ``T_total = sum_i w_i * T_i`` (drift ``C10``). Under ``l2`` each
            channel applied its own ``T_i``; this value is then **informational only**.
        n_clamped: Pixels whose ``acos`` argument needed clipping. Must be ``0``.
    """

    pc: np.ndarray
    orientation: np.ndarray
    feature_type: np.ndarray
    threshold: float
    n_clamped: int


def _weighted_scalars(channels, weights):
    """``(A_total, T_total, A_max_total)`` -- the three 1-homogeneous denominators."""
    a_total = sum(w * c.sum_amplitude for w, c in zip(weights, channels))
    t_total = float(sum(w * c.threshold for w, c in zip(weights, channels)))
    a_max = sum(w * c.max_amplitude for w, c in zip(weights, channels))
    return a_total, t_total, a_max


def _fused_vector(channels, weights):
    """``sum_i w_i * (even_i, h1_i, h2_i)``. Coherent's numerator; everyone's angles."""
    v0 = sum(w * c.sum_even for w, c in zip(weights, channels))
    v1 = sum(w * c.sum_h1 for w, c in zip(weights, channels))
    v2 = sum(w * c.sum_h2 for w, c in zip(weights, channels))
    return v0, v1, v2


def fuse_joint(channels, weights, *, n_scale, cutoff, g, deviation_gain, epsilon):
    """Shared denominator, L1 energy: ``E_total = sum_i w_i * ||v_i||``.

    A channel's amplitude enters the denominator whether or not its structure is
    coherent, so a loud but incoherent channel **vetoes** the others' edges. That veto is
    the only mechanism by which colour is expected to help, and ``l2`` has no analogue of
    it. Drift ``C7``.
    """
    energy = sum(w * c.energy for w, c in zip(weights, channels))
    a_total, t_total, a_max = _weighted_scalars(channels, weights)
    pc, n_clamped = congruency_from_accumulators(
            energy, a_total, a_max, t_total, n_scale=n_scale, cutoff=cutoff, g=g,
            deviation_gain=deviation_gain, epsilon=epsilon,
    )
    return pc, t_total, n_clamped


def fuse_coherent(channels, weights, *, n_scale, cutoff, g, deviation_gain, epsilon):
    """As ``joint``, but ``E_total = ||sum_i w_i * v_i||``.

    Cancels opposite-phase responses across channels -- **including a genuine
    anti-correlated chromatic edge**, where lightness falls as yellowness rises. Opt-in,
    never default. Drift ``C8``.
    """
    v0, v1, v2 = _fused_vector(channels, weights)
    energy = np.sqrt(v0 ** 2 + v1 ** 2 + v2 ** 2)   # not np.hypot
    a_total, t_total, a_max = _weighted_scalars(channels, weights)
    pc, n_clamped = congruency_from_accumulators(
            energy, a_total, a_max, t_total, n_scale=n_scale, cutoff=cutoff, g=g,
            deviation_gain=deviation_gain, epsilon=epsilon,
    )
    return pc, t_total, n_clamped


def fuse_l2(channels, weights, *, n_scale, cutoff, g, deviation_gain, epsilon):
    """CMPCM's rule: ``out = sqrt(sum_i (w_i * F_i)**2)`` over per-channel congruencies.

    **Not divided by ``||w||``** -- the paper does not, and §7.1 must check the paper's
    actual quantity. Range ``[0, ||w||]``; the caller clips. Drift ``C3``.

    Three independent detectors combined after the fact: no cross-channel term reaches
    any denominator, so incoherent chroma amplitude can never veto a spurious luminance
    edge. §3.2.
    """
    total = np.zeros_like(channels[0].sum_amplitude)
    n_clamped = 0
    for w, c in zip(weights, channels):
        pc_i, clamped_i = congruency_from_accumulators(
                c.energy, c.sum_amplitude, c.max_amplitude, c.threshold,
                n_scale=n_scale, cutoff=cutoff, g=g, deviation_gain=deviation_gain,
                epsilon=epsilon,
        )
        total += (w * pc_i) ** 2
        n_clamped += clamped_i
    _, t_total, _ = _weighted_scalars(channels, weights)
    return np.sqrt(total), t_total, n_clamped


def color_phase_congruency(
        channels: Sequence[MonogenicChannel],
        weights: np.ndarray,
        *,
        fusion: str,
        n_scale: int,
        cutoff: float = 0.5,
        g: float = 10.0,
        deviation_gain: float = 1.5,
        epsilon: float = EPSILON_MONOGENIC,
) -> ColorPhaseResult:
    """Fuse three channels' monogenic accumulators into one congruency map.

    Args:
        channels: Exactly three :class:`MonogenicChannel`, in **luminance-first** order.
        weights: ``(3,)`` non-negative; ``weights[0]`` is pinned to ``1.0`` by the
            operation (§4.2).
        fusion: ``"joint"``, ``"coherent"`` or ``"l2"``.
        n_scale: Must match the ``n_scale`` used to build ``channels``.
        cutoff: Frequency-spread sigmoid centre.
        g: Frequency-spread sigmoid sharpness.
        deviation_gain: Scales the phase-deviation term.
        epsilon: Division guard.

    Returns:
        A :class:`ColorPhaseResult` whose ``pc`` is **un-clipped**.

    Raises:
        ValueError: If ``fusion`` is unknown, ``channels`` is not length 3, or
            ``weights`` is not length 3.
    """
    if fusion not in _FUSIONS:
        raise ValueError(f"fusion must be one of {_FUSIONS}; got {fusion!r}.")
    if len(channels) != 3 or len(weights) != 3:
        raise ValueError(
                f"expected 3 channels and 3 weights; got {len(channels)} and {len(weights)}."
        )

    dispatch = {"joint": fuse_joint, "coherent": fuse_coherent, "l2": fuse_l2}
    pc, threshold, n_clamped = dispatch[fusion](
            channels, weights, n_scale=n_scale, cutoff=cutoff, g=g,
            deviation_gain=deviation_gain, epsilon=epsilon,
    )

    # Angles always come from the fused vector, in every mode. Under `joint` and `l2`
    # that vector did not produce `pc`. Not exposed; drift C15.
    v0, v1, v2 = _fused_vector(channels, weights)
    orientation = np.arctan2(-v2, v1)
    orientation = np.where(orientation > np.pi / 2, orientation - np.pi, orientation)
    orientation = np.where(orientation <= -np.pi / 2, orientation + np.pi, orientation)
    feature_type = np.arctan2(v0, np.sqrt(v1 ** 2 + v2 ** 2))

    return ColorPhaseResult(pc, orientation, feature_type, threshold, n_clamped)
```

- [ ] **Step 4: Run the tests — all green**

```bash
uv run pytest tests/unit/enhance/test_color_phase_kernels.py -q
```

- [ ] **Step 5: Prove `TestFusionSanity` can fail**

Temporarily change `fuse_joint`'s energy line to the L2-over-L1 form,
`energy = np.sqrt(sum((w * c.energy) ** 2 for w, c in zip(weights, channels)))`.

```bash
uv run pytest tests/unit/enhance/test_color_phase_kernels.py::TestFusionSanity -q
```
Expected: **FAIL**. Revert. This is §3.1 as an executable regression.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/enhance/_color_phase_kernels.py tests/unit/enhance/test_color_phase_kernels.py
git commit -m "feat(enhance): add cross-channel fusion kernels for colour phase congruency

joint (shared denominator, L1 energy), coherent (shared denominator, vector
energy), l2 (CMPCM's root-sum-of-squares over three independent congruencies).

Angles come from the fused vector in every mode and are returned but not
exposed -- only 'coherent' builds that vector natively. Drift C15.

l2 is NOT divided by ||w||: the paper does not, and the PFOM regression must
see the paper's quantity. Drift C3."
```

---

## Task 3: `fusion_algebra.py` — the numeric oracle

**Shape:** Leaf. **Parallel with Task 1** (zero file overlap).

**Files:**
- Create: `docs/superpowers/logic_validation_scripts/2026-07-09-focus-edge-color-phase/fusion_algebra.py`

**Interfaces:**
- Consumes: nothing. **Must never `import phenotypic`** — it re-derives the load-bearing numbers from
  scratch, which is the entire point. stdlib + numpy only. Exits non-zero on failure.
- Produces: the numbers Task 0 wrote into the spec. If the spec and this script disagree, the script
  is right and the spec is edited.

- [ ] **Step 1: Write it**

Four checks, each printing its measured value and asserting against the spec's claim.

```python
#!/usr/bin/env python3
"""Re-derive `color-phase-congruency.md`'s load-bearing numeric claims from scratch.

Depends only on the stdlib and numpy. Never imports `phenotypic`: the point is to check
the *spec*, not the implementation of it. Exits non-zero if any claim has stopped being
true.

Run:  uv run --no-project --with numpy python fusion_algebra.py
"""
import sys
import numpy as np

EPS = 1e-4
DEVIATION_GAIN = 1.5
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def _response(ratio: float, dg: float = DEVIATION_GAIN) -> float:
    return max(1.0 - dg * float(np.arccos(np.clip(ratio, -1.0, 1.0))), 0.0)


def check_01_l2_over_l1_annihilates_a_coherent_edge() -> None:
    """§3.1. An L2 numerator over an L1 denominator inverts the CA acceptance criterion.

    Expected responses are FULL-PRECISION literals at 1e-12, not the four decimals the
    spec table prints. d(response)/d(ratio) = dg/sqrt(1 - ratio**2) = 2.6520 at row 3,
    and the printed ratio 0.8247 is itself rounded by 3.4e-05, so round-tripping it
    through the formula yields 0.0983 where the true value is 0.0982. That is how a
    wrong number entered this spec. A 4-dp intermediate cannot determine a 4-dp result.
    """
    rows = [
        # label,      weights,                        firing,               ratio, response
        ("one only",  np.ones(3), np.array([1.0, 0.0, 0.0]), 1.0, 1.0),
        ("all three", np.ones(3), np.ones(3), 0.5773502691896258, 0.0),
        ("real prior", np.array([0.804, 0.013, 0.183]), np.ones(3),
         0.824665992993527, 0.0982226557669601),
    ]
    ok = True
    for label, w, fires, want_ratio, want_resp in rows:
        e = a = fires
        ratio = float(np.sqrt(np.sum((w * e) ** 2)) / np.sum(w * a))
        resp = _response(ratio)
        ok &= abs(ratio - want_ratio) < 1e-12 and abs(resp - want_resp) < 1e-12
        print(f"      {label:12s} ratio={ratio:.6f} response={resp:.6f}")
        # The L1/L1 form must pass every row at full strength.
        ok &= abs(_response(float(np.sum(w * e) / np.sum(w * a))) - 1.0) < 1e-12
    # The load-bearing claim: the annihilation is EXACT, not merely small.
    ok &= _response(float(np.sqrt(3.0)) / 3.0) == 0.0
    check("01 l2-over-l1 annihilates a coherent edge", ok,
          "single-channel 1.000000, three-channel exactly 0.0 at deviation_gain=1.5")


def check_02_no_single_deviation_gain_reproduces_the_old_table() -> None:
    """The retracted §3.1 printed 0.0091 and 0.1425. Show they are mutually inconsistent."""
    dg_a = (1 - 0.0091) / float(np.arccos(0.5774))
    dg_b = (1 - 0.1425) / float(np.arccos(0.8247))
    check("02 the retracted response column is self-inconsistent",
          abs(dg_a - dg_b) > 0.3,
          f"row 2 needs dg={dg_a:.4f}, row 3 needs dg={dg_b:.4f}; shipped dg={DEVIATION_GAIN}")


def check_03_energy_never_exceeds_amplitude() -> None:
    """The `acos` argument stays in [-1, 1] for joint AND coherent. So n_clamped == 0."""
    rng = np.random.default_rng(20260709)
    worst_j = worst_c = -np.inf
    for _ in range(200_000):
        n_scale = int(rng.integers(2, 7))
        v = rng.normal(size=(3, n_scale, 3))
        w = np.array([1.0, rng.uniform(0, 8), rng.uniform(0, 8)])
        a = np.linalg.norm(v, axis=2).sum(axis=1)
        s = v.sum(axis=1)
        e = np.linalg.norm(s, axis=1)
        a_total = float((w * a).sum())
        worst_j = max(worst_j, float((w * e).sum()) / (a_total + EPS))
        worst_c = max(worst_c, float(np.linalg.norm((w[:, None] * s).sum(axis=0))) / (a_total + EPS))
    check("03 E_total <= A_total for joint and coherent", worst_j <= 1.0 and worst_c <= 1.0,
          f"max joint {worst_j:.6f}, max coherent {worst_c:.6f} over 200000 draws")


def check_04_epsilon_breaks_scale_invariance_through_the_sigmoid() -> None:
    """§4.2 / drift C17. The old '~1%' was wrong by two orders. Name the real culprit."""
    def joint(w, e, a, t, a_max, n_scale=4, cutoff=0.5, g=10.0):
        e_t, a_t = float((w * e).sum()), float((w * a).sum())
        t_t, m_t = float((w * t).sum()), float((w * a_max).sum())
        width = (a_t / (m_t + EPS) - 1.0) / (n_scale - 1)
        weight = 1.0 / (1.0 + np.exp(g * (cutoff - width)))
        return (weight * _response(e_t / (a_t + EPS))
                * max(e_t - t_t, 0.0) / (e_t + EPS))

    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20_000):
        a = rng.uniform(0.5, 5.0, 3)
        e, t, a_max = a * rng.uniform(0.5, 1, 3), a * rng.uniform(0, 0.3, 3), a * rng.uniform(0.3, 0.9, 3)
        w = np.array([1.0, rng.uniform(0, 8), rng.uniform(0, 8)])
        base = joint(w, e, a, t, a_max)
        if base <= 1e-6:
            continue
        for c in (0.01, 100.0):
            worst = max(worst, abs(joint(c * w, e, a, t, a_max) - base) / base)
    check("04 eps breaks 1-homogeneity through A_max + eps, not E + eps", worst > 0.5,
          f"max relative change {worst * 100:.1f}% over c in [0.01, 100] at Lab L* amplitudes "
          f"-- the retracted claim was '~1%'")


if __name__ == "__main__":
    check_01_l2_over_l1_annihilates_a_coherent_edge()
    check_02_no_single_deviation_gain_reproduces_the_old_table()
    check_03_energy_never_exceeds_amplitude()
    check_04_epsilon_breaks_scale_invariance_through_the_sigmoid()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("4/4 checks passed.")
```

- [ ] **Step 2: Run it**

```bash
cd docs/superpowers/logic_validation_scripts/2026-07-09-focus-edge-color-phase
# `ulimit -v` is a no-op on Darwin (see Global Constraints). `timeout` is the real bound.
timeout 900 uv run --no-project --with numpy python fusion_algebra.py; echo "exit=$?"
```
Expected: `4/4 checks passed.`, exit `0`.

- [ ] **Step 3: Prove it can fail — twice**

```bash
# (a) the physics: at dg=1.0369 the three-channel response is 0.0094, not exactly 0.
sed -i '' 's/^DEVIATION_GAIN = 1.5$/DEVIATION_GAIN = 1.0369/' fusion_algebra.py
uv run --no-project --with numpy python fusion_algebra.py; echo "exit=$?"   # -> check_01 FAIL, exit 1

# (b) the rounding trap: reintroduce the wrong literal the tolerance exists to catch.
sed -i '' 's/0.824665992993527, 0.0982226557669601/0.8247, 0.0983/' fusion_algebra.py
uv run --no-project --with numpy python fusion_algebra.py; echo "exit=$?"   # -> check_01 FAIL, exit 1
```
Revert both. Both are verified to redden it.

- [ ] **Step 4: Prove it never imports `phenotypic`**

```bash
grep -Ec '^[[:space:]]*(import|from)[[:space:]]+phenotypic' fusion_algebra.py   # -> 0
```

**Not** `grep -c 'phenotypic'`. The module docstring says *"Never imports `phenotypic`"*, so a bare
substring grep matches its own disclaimer and can never return `0`. Match an actual import statement.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/logic_validation_scripts/2026-07-09-focus-edge-color-phase/
git commit -m "test(spec): re-derive color-PC's numeric claims from scratch

fusion_algebra.py, numpy only, never imports phenotypic. Four checks:
§3.1's table (the retracted response column is self-inconsistent -- row 2
needs dg=1.0373, row 3 needs dg=1.4265); E_total <= A_total for joint and
coherent over 200k draws, so n_clamped == 0 is assertable; and the real
source of §4.2's broken 1-homogeneity, which is A_max + eps inside a g=10
sigmoid, not E + eps. Measured 100%, not '~1%'."
```

---

## GATE A — after Task 1, before Task 2

Blocking. Dispatch a fresh review agent, **Opus/high** (never weaker than the implementer).

Charge:
1. Reproduce the bit-identity claim independently. Do not trust `TestTheRefactorMovesNoBits`; write
   your own comparison against `git show HEAD~1:src/phenotypic/enhance/_monogenic_kernels.py`
   loaded into a temp module, on all three shipped plates and both noise methods. **Done — 27
   configurations, zero mismatches.**
1b. Confirm the test's oracle is not the function under test. Parse `_monogenic_pc_from_primitives`
   with `ast` and assert it calls **none** of `monogenic_channel_response`,
   `congruency_from_accumulators`, `MonogenicChannel`, `monogenic_phase_congruency`. The first draft
   of this plan specified an oracle that failed exactly this check.
2. Confirm `weight` is still computed as its own statement, not folded into the `pc` expression
   (folding changes float association).
3. Confirm the `if s == 0:` branch still reads `sum_amplitude`, not `amplitude`.
4. Confirm `channel.energy` is `sqrt(a**2 + b**2 + c**2)` and not `np.hypot`.
5. Confirm no guard moved: `n_scale` and `mult` in `monogenic_channel_response`, `sigma_onf` in
   `log_gabor_scale`, `noise_method`'s epsilon-compare intact.
6. `git diff HEAD~1 --stat` must show exactly two files.

**Do not proceed to Task 2 until every item is confirmed with evidence.** Surface any design-level
conflict to the user.

---

## Task 4: The operation

**Shape:** Keystone.

**Files:**
- Create: `src/phenotypic/enhance/_focus_edge_color_phase.py`
- Modify: `src/phenotypic/sdk_/typing_.py`
- Modify: `src/phenotypic/enhance/__init__.py`

**Interfaces:**
- Consumes: `MonogenicChannel`, `monogenic_channel_response` (Task 1); `ColorPhaseResult`,
  `color_phase_congruency` (Task 2).
- Produces, for Tasks 5–7:
  ```python
  class FocusEdgeColorPhase(FocusEdge):
      color_space: ColorSpaceName = "lab"
      fusion: PhaseFusion = "joint"
      chroma_weight_1: Annotated[float, TuneSpec(0.0, 8.0)] = Field(1.0, ge=0.0)
      chroma_weight_2: Annotated[float, TuneSpec(0.0, 8.0)] = Field(1.0, ge=0.0)
      lift: PhaseLift = "monogenic"
      # + the nine FocusEdgeMonogenicPhase fields, verbatim
      output: ColorPhaseOutput = "pc"

      def _color_phase_congruency(self, image: Image) -> ColorPhaseResult: ...
  ```

**Read the `adding-an-operation` skill before this task.** Four new `Literal` aliases and two new
numeric fields, both of which enter the tune annotation-coverage gate.

**Why the nine monogenic fields are duplicated rather than inherited or mixed in.** A shared
`BaseModel` mixin would put two `pydantic.BaseModel` bases in the MRO of a class that the operation
registry and `from_json` both walk; and subclassing `FocusEdgeMonogenicPhase` would make
`isinstance(color_op, FocusEdgeMonogenicPhase)` true, which it is not. Duplicate the fields, and pin
them with `TestFieldParityWithTheMonogenicPort` (Step 7) so they cannot drift. The parity test is a
real oracle; the mixin would only have been an assertion.

- [ ] **Step 1: Add the four `Literal` aliases**

In `src/phenotypic/sdk_/typing_.py`, beside `MonogenicOutput` (line 34):

```python
#: Colour space whose channels feed :class:`FocusEdgeColorPhase`. Luminance-first order
#: is imposed by the operation: ``lab -> (L*, a*, b*)``, ``hsv -> (V, H, S)``.
ColorSpaceName: TypeAlias = Literal["lab", "hsv"]

#: Cross-channel fusion rule. ``l2`` is CMPCM's; ``joint`` and ``coherent`` are ours.
PhaseFusion: TypeAlias = Literal["joint", "coherent", "l2"]

#: Signal lift. ``conformal`` is gated on `conformal-lift.md` §4 and currently raises.
PhaseLift: TypeAlias = Literal["monogenic", "conformal"]

#: Only ``pc`` is exposed. The angle maps live on the protected helper's result object;
#: see drift ``C15``.
ColorPhaseOutput: TypeAlias = Literal["pc"]
```

- [ ] **Step 2: Write the failing contract test**

Create `tests/unit/enhance/test_focus_edge_color_phase.py` with just this, for now:

```python
import numpy as np
import pytest

from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import FocusEdgeColorPhase


class TestConstruction:
    def test_constructible_with_no_arguments(self):
        assert FocusEdgeColorPhase().fusion == "joint"

    def test_is_keyword_only(self):
        with pytest.raises(TypeError):
            FocusEdgeColorPhase("lab")  # type: ignore[misc]

    def test_conformal_lift_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="conformal"):
            FocusEdgeColorPhase(lift="conformal")
```

- [ ] **Step 3: Run it — expect `ImportError`**

```bash
uv run pytest tests/unit/enhance/test_focus_edge_color_phase.py -q
```
Expected: `ImportError: cannot import name 'FocusEdgeColorPhase'`.

- [ ] **Step 4: Implement the operation**

```python
"""Colour phase congruency: per-channel monogenic PC, then a cross-channel fusion.

Reuses :mod:`_monogenic_kernels` verbatim and adds no new signal theory. The ``l2``
fusion is Shi et al.'s CMPCM rule; ``joint`` and ``coherent`` are ours.

References:
    Shi, Y. et al. "Colour edge detection based on the fusion of monogenic phase
    congruency and colour morphology." *Multimed. Tools Appl.* 78, 10701--10716 (2019).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import numpy as np
from pydantic import Field, model_validator

from ._color_phase_kernels import ColorPhaseResult, color_phase_congruency
from ._monogenic_kernels import monogenic_channel_response
from ..abc_ import FocusEdge
from ..sdk_.typing_ import (
    ColorPhaseOutput, ColorSpaceName, PhaseFusion, PhaseLift, TuneSpec,
)

if TYPE_CHECKING:
    from phenotypic._core._image import Image

#: Accessor channel indices in **luminance-first** order (spec §4.2). ``Lab`` is native
#: ``(L*, a*, b*)``; ``hsv`` is native ``(H, S, V)`` and must be reordered so that index
#: ``0`` is always the axis whose weight is pinned at ``1.0``.
_LUMINANCE_FIRST: dict[str, tuple[int, int, int]] = {
    "lab": (0, 1, 2),
    "hsv": (2, 0, 1),
}


class FocusEdgeColorPhase(FocusEdge):
    """Enhance colony edges using colour phase congruency across three channels.

    Runs :class:`FocusEdgeMonogenicPhase`'s monogenic chain independently on each of
    three colour channels, then fuses the results. Phase congruency is already invariant
    to illumination level; fusing across colour additionally lets a channel with
    amplitude but no phase agreement -- pigment noise, agar speckle, Bayer artefacts --
    **veto** an edge the luminance channel would otherwise assert.

    Args:
        color_space: ``"lab"`` (default) or ``"hsv"``. Channels are taken in
            luminance-first order: ``lab`` gives ``(L*, a*, b*)``, ``hsv`` gives
            ``(V, H, S)``. Raw CIELAB is already the perceptual common scale -- CIE76's
            ``dE*ab`` is the Euclidean norm over raw ``L*a*b*`` -- so no per-axis
            rescaling is applied. **Hue is circular and bandpassed across its wrap
            discontinuity**, so ``"hsv"`` manufactures a phantom edge at near-red
            boundaries; it is retained because CMPCM uses HSV. Drift ``C16``.
        fusion: ``"joint"`` (default) shares one denominator across channels, so
            incoherent chroma amplitude vetoes a spurious luminance edge. ``"l2"`` is
            CMPCM's rule -- three independent congruencies combined by root-sum-of-squares
            -- and has **no** cross-channel interaction at all. ``"coherent"`` sums the
            monogenic vectors before taking their norm; it cancels opposite-phase
            responses, **including a genuine anti-correlated chromatic edge** where
            lightness falls as yellowness rises. Opt-in, never default.
        chroma_weight_1: Weight on the first chromatic axis (``a*`` under ``lab``,
            ``H`` under ``hsv``). Luminance is pinned at ``1.0``, so there are two
            degrees of freedom, not three. ``0.0`` disables the axis entirely and the
            operation reduces exactly to :class:`FocusEdgeMonogenicPhase` on luminance.
        chroma_weight_2: Weight on the second chromatic axis (``b*`` / ``S``). The upper
            search bound of ``8.0`` brackets "chroma off" through "chroma dominates" for
            the axis that carries signal on real plates, and deliberately refuses to let
            ``a*`` reach parity with ``L*`` (which needs ``19``--``61``).
        lift: ``"monogenic"`` (default). ``"conformal"`` raises
            :exc:`NotImplementedError` -- the field exists so the surface is stable, but
            the path is gated on an experiment it may fail.
        n_scale: Number of log-Gabor scales. Must be at least 2.
        min_wavelength: Wavelength of the finest scale, in pixels.
        mult: Wavelength multiplier between successive scales.
        sigma_onf: Ratio of each filter's Gaussian sigma to its centre frequency.
        k: Noise standard deviations above the mean at which the threshold sits.
        deviation_gain: Scales the phase-deviation term.
        cutoff: Fractional frequency-spread below which the response is penalized.
        g: Sharpness of the frequency-spread sigmoid.
        noise_method: ``-1`` median, ``-2`` Rayleigh mode, ``>= 0`` a literal threshold.
        output: Only ``"pc"``. The fused ``orientation`` and ``feature_type`` are
            computed and returned by :meth:`_color_phase_congruency`, but not exposed:
            of the three fusion modes only ``"coherent"`` builds a fused monogenic vector,
            so under the default they would describe a quantity the response never
            touched. Drift ``C15``.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the fused congruency map,
        clipped to ``[0, 1]``. ``rgb`` and ``gray`` are unchanged.

    Raises:
        NotImplementedError: If ``lift="conformal"``.
        ValueError: If the image is achromatic (all three RGB channels identical), since
            ``a*``/``b*`` are then identically zero and ``joint`` degenerates to a
            luminance congruency divided by itself. Drift ``C11``.
        ValidationError: On any out-of-range field.

    Note:
        **This operation reads ``image.rgb``, not ``detect_mat``.** It is a *source*, like
        :class:`SetDetectMode`: **any enhancer placed before it in an**
        :class:`ImagePipeline` **has no effect on its output.** Colour phase congruency
        is defined on colour, and ``rgb`` is not a supported ``detect_mat`` layer. Legal
        under ``@validate_operation_integrity``, which forbids *mutating* ``rgb``/``gray``
        and says nothing about reading them. Drift ``C2``.

    Examples:
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import FocusEdgeColorPhase
        >>> enhanced = FocusEdgeColorPhase().apply(load_synth_yeast_plate())
        >>> bool(0.0 <= enhanced.detect_mat[:].min() <= enhanced.detect_mat[:].max() <= 1.0)
        True

    See Also:
        :class:`FocusEdgeMonogenicPhase`, which this reduces to when both chroma weights
        are ``0.0``.
    """

    color_space: ColorSpaceName = "lab"
    fusion: PhaseFusion = "joint"
    chroma_weight_1: Annotated[float, TuneSpec(0.0, 8.0)] = Field(1.0, ge=0.0)
    chroma_weight_2: Annotated[float, TuneSpec(0.0, 8.0)] = Field(1.0, ge=0.0)
    lift: PhaseLift = "monogenic"

    # Ported verbatim from FocusEdgeMonogenicPhase. Pinned by
    # TestFieldParityWithTheMonogenicPort -- do not let these drift.
    n_scale: Annotated[int, TuneSpec(3, 6)] = Field(4, ge=2)
    min_wavelength: Annotated[float, TuneSpec(2.0, 10.0)] = Field(3.0, ge=2.0)
    mult: Annotated[float, TuneSpec(1.5, 3.0)] = Field(2.1, gt=1.0)
    sigma_onf: Annotated[float, TuneSpec(0.1, 0.99)] = Field(0.55, ge=0.1, lt=1.0)
    k: Annotated[float, TuneSpec(0.5, 20.0)] = Field(3.0, ge=0.0)
    deviation_gain: Annotated[float, TuneSpec(1.0, 2.0)] = Field(1.5, gt=0.0)
    cutoff: Annotated[float, TuneSpec(0.3, 0.7)] = Field(0.5, gt=0.0, lt=1.0)
    g: Annotated[float, TuneSpec(2.0, 20.0)] = Field(10.0, gt=0.0)
    noise_method: Annotated[float, TuneSpec(tunable=False)] = -1.0

    output: ColorPhaseOutput = "pc"

    @model_validator(mode="after")
    def _reject_the_gated_conformal_lift(self) -> "FocusEdgeColorPhase":
        """`lift="conformal"` raises at construction, not at apply.

        pydantic v2 does not trap `NotImplementedError`, so it propagates with its type
        intact -- unlike a `ValueError` raised inside `_operate`, which `ImageOperation`
        wraps **twice**: into a `RuntimeError` at `_image_operation.py:423` and then a
        bare `Exception` at `:470`. Measured chain on `FocusEdgePhase(sigma_onf=1.0)`:
        ``RuntimeError -> Exception -> ValueError``.
        Verified in `TestConstruction::test_conformal_lift_raises_not_implemented`.
        """
        if self.lift == "conformal":
            raise NotImplementedError(
                    "lift='conformal' is gated on conformal-lift.md §4's three-arm "
                    "junction experiment and is not implemented. The field exists so the "
                    "surface is stable. Use lift='monogenic'."
            )
        return self

    def _extract_channels(self, image: Image) -> list[np.ndarray]:
        """Three scalar channels from ``rgb``, in luminance-first order.

        Raises:
            ValueError: If the image is achromatic. Drift ``C11``.
        """
        rgb = image.rgb[:]
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                    f"FocusEdgeColorPhase needs a 3-channel RGB image; got shape {rgb.shape}."
            )
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
            raise ValueError(
                    "FocusEdgeColorPhase requires a chromatic image: all three RGB channels "
                    "are identical, so a* and b* are identically zero and fusion='joint' "
                    "degenerates to a luminance congruency divided by itself. Use "
                    "FocusEdgeMonogenicPhase on a greyscale image."
            )

        stack = image.color.Lab[:] if self.color_space == "lab" else image.color.hsv[:]
        return [np.asarray(stack[..., i], dtype=np.float64)
                for i in _LUMINANCE_FIRST[self.color_space]]

    def _color_phase_congruency(self, image: Image) -> ColorPhaseResult:
        """Fuse three channels' monogenic accumulators. **`pc` is un-clipped.**

        Protected, and returns more than :meth:`_operate` exposes: ``orientation`` and
        ``feature_type`` ride along on the result so a future consumer can reach them
        without a breaking change -- mirroring
        :meth:`FocusEdgePhase._phasecong3`, whose result carries ``orientation`` and
        ``feature_type`` while ``output`` exposes only ``M``/``m``/``pc_sum``.
        Drift ``C15``.
        """
        channels = [
            monogenic_channel_response(
                    channel, n_scale=self.n_scale, min_wavelength=self.min_wavelength,
                    mult=self.mult, sigma_onf=self.sigma_onf, k=self.k,
                    noise_method=self.noise_method,
            )
            for channel in self._extract_channels(image)
        ]
        weights = np.array([1.0, self.chroma_weight_1, self.chroma_weight_2])
        return color_phase_congruency(
                channels, weights, fusion=self.fusion, n_scale=self.n_scale,
                cutoff=self.cutoff, g=self.g, deviation_gain=self.deviation_gain,
        )

    def _operate(self, image: Image) -> Image:
        """Replace the detection matrix with the fused congruency map."""
        result = self._color_phase_congruency(image)
        # The clip is load-bearing for `l2` (range [0, ||w||]) and redundant for the
        # other two. Keep it in all cases. detect_mat enforces float32 on assignment.
        image.detect_mat[:] = np.clip(result.pc, 0.0, 1.0)
        return image
```

- [ ] **Step 5: Export it**

In `src/phenotypic/enhance/__init__.py`, add `from ._focus_edge_color_phase import FocusEdgeColorPhase`
next to the monogenic import, and `"FocusEdgeColorPhase",` to `__all__`.

- [ ] **Step 6: Verify `NotImplementedError` survives pydantic**

```bash
uv run python -c "
from phenotypic.enhance import FocusEdgeColorPhase
try:
    FocusEdgeColorPhase(lift='conformal')
except NotImplementedError as e:
    print('OK, NotImplementedError:', e)
except Exception as e:
    print('WRAPPED into', type(e).__name__, '-- fall back to raising in _operate')
"
```
If pydantic wraps it into `ValidationError`, move the raise into `_operate` and have the test walk
the `__cause__` chain (`ImageOperation.apply` wraps twice — see drift `M10`). **Do not** leave a test
asserting a type the code cannot raise; that mistake was made once on this branch already.

- [ ] **Step 7: Add the field-parity test**

```python
class TestFieldParityWithTheMonogenicPort:
    """The nine shared fields are duplicated, not inherited. They must not drift.

    A mixin would have put two BaseModel bases in the MRO of a class the operation
    registry walks. Duplication + this test is the cheaper, checkable option.
    """

    SHARED = ["n_scale", "min_wavelength", "mult", "sigma_onf", "k",
              "deviation_gain", "cutoff", "g", "noise_method"]

    @pytest.mark.parametrize("name", SHARED)
    def test_default_and_bounds_match(self, name):
        colour = FocusEdgeColorPhase.model_fields[name]
        mono = FocusEdgeMonogenicPhase.model_fields[name]
        assert colour.default == mono.default, f"{name} default drifted"
        assert repr(colour.metadata) == repr(mono.metadata), f"{name} bounds drifted"

    def test_no_shared_field_was_forgotten(self):
        mono = set(FocusEdgeMonogenicPhase.model_fields) - {"output"}
        assert mono == set(self.SHARED), (
            "FocusEdgeMonogenicPhase gained or lost a field; mirror it here."
        )
```

- [ ] **Step 8: Run the tune annotation-coverage gate**

```bash
uv run pytest tests/unit/tune/test_annotation_coverage.py -q
uv run pytest tests/unit/enhance/ tests/unit/abc_/ -q
uv run mypy src/phenotypic/enhance/ && uv run ruff check src/phenotypic/enhance/
```
`chroma_weight_1` and `chroma_weight_2` are new numeric fields on an `enhance/` operation, so the
coverage gate will fail unless both carry a `TuneSpec`. They do.

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/enhance/_focus_edge_color_phase.py \
        src/phenotypic/enhance/__init__.py src/phenotypic/sdk_/typing_.py \
        tests/unit/enhance/test_focus_edge_color_phase.py
git commit -m "feat(enhance): add FocusEdgeColorPhase

Per-channel monogenic PC over three luminance-first colour channels, fused by
joint (default) / coherent / l2. Sources from image.rgb, so it is a pipeline
source like SetDetectMode -- drift C2.

output is Literal['pc']. The fused orientation and feature_type ride on
_color_phase_congruency's result, unexposed, as _phasecong3 already does.

lift='conformal' raises NotImplementedError at construction."
```

---

## Task 5: The chromatic-aberration experiment — a decision, not a test

**Shape:** Seam. **Parallel with Task 6** (zero file overlap). **This is `GATE C`.**

**Files:**
- Create: `docs/superpowers/plans/2026-07-09-focus-edge-color-phase/experiments/chromatic_aberration.py`
- Modify: `docs/superpowers/specs/2026-07-08-alt-phase-detection/color-phase-congruency.md` (§7.2's
  results table)
- Possibly modify: `src/phenotypic/enhance/_focus_edge_color_phase.py` (the `fusion` default)

**Interfaces:**
- Consumes: `FocusEdgeColorPhase`, `FocusEdgeMonogenicPhase`, `FocusEdgePhase`.
- Produces: the shipped `fusion` default, and §8's acceptance criterion 4.

It imports `phenotypic`, so it is **not** a logic-validation script and does not live under
`logic_validation_scripts/`.

- [ ] **Step 1: Write the experiment**

`load_synth_filamentous_plate()` returns a `GridImage`, `600 × 800`, with an `objmap` of **60**
objects — verified, not assumed. **Do not use `make_synthetic_filamentous_plate()`**: it returns a
bare `np.ndarray` with no label map.

Inject a radial chromatic aberration of `δ ∈ {0, 1, 2, 3}` px into R and B (scale R by `1 + δ/r_max`
and B by `1 − δ/r_max` about the image centre, bilinear). For each of `FocusEdgePhase` (baseline),
`FocusEdgeMonogenicPhase` on luminance, and `FocusEdgeColorPhase` with each of the three fusions,
measure mean boundary-localization error against the `objmap`'s object boundaries (distance transform
of the true boundary, sampled at the response's non-maximum-suppressed ridge, thresholded at Otsu).

Print a `4 × 5` table of mean error in pixels, and write it to
`experiments/chromatic_aberration_results.md`.

- [ ] **Step 2: Run it**

```bash
# `ulimit -v` is a no-op on Darwin. Bound with `timeout`; keep arrays under 5e6 elements.
timeout 1800 uv run python docs/superpowers/plans/2026-07-09-focus-edge-color-phase/experiments/chromatic_aberration.py
```

- [ ] **Step 3: Read the result against the stated prediction**

Spec §7.2's prediction: *"`joint` merges the displaced edges, so its error stays roughly flat in `δ`,
while `l2` degrades."*

- **If `joint` is flat and `l2` degrades** → the default stays `joint`. Record the table.
- **If it fails** → **ship `l2`.** Change the field default, and Task 7's docs must follow.
- **Either way, record the result.** *A null result must not be buried.* This is spec §7.2, verbatim,
  and §8's acceptance criterion 4.

- [ ] **Step 4: If the default flips, STOP and surface to the user**

A flipped default changes the operation's headline behaviour, `enhance/CLAUDE.md`, the class
docstring, and drift `C7`'s status (which currently reads *"a null result flips the default back to
`l2`"*). Do not make that call unilaterally.

- [ ] **Step 5: Write §7.2's results table into the spec, and commit**

```bash
git add docs/superpowers/plans/2026-07-09-focus-edge-color-phase/experiments/ \
        docs/superpowers/specs/2026-07-08-alt-phase-detection/color-phase-congruency.md
git commit -m "exp(enhance): run and record the chromatic-aberration experiment (spec §7.2)"
```

---

## Task 6: The remaining tests

**Shape:** Keystone. **Parallel with Task 5.**

**Files:**
- Modify: `tests/unit/enhance/test_focus_edge_color_phase.py`
- Create: `tests/unit/enhance/test_color_phase_pfom.py`

**Interfaces:**
- Consumes: `FocusEdgeColorPhase` (Task 4).
- Produces: spec §7's tests 1–7 and §8's acceptance criteria 1–3, 5.

- [ ] **Step 1: §7 test 1 — per-channel fidelity, including the angles**

```python
class TestZeroChromaReducesToTheMonogenicPort:
    """§7 test 1. The fusion must not perturb the port -- to `rtol=1e-10`, per spec.

    Because the fused vector collapses to v_L at zero chroma weight, this also pins the
    unexposed `orientation` and `feature_type` (drift C15) for free.
    """

    @pytest.mark.parametrize("fusion", ["joint", "coherent", "l2"])
    def test_pc_matches_focus_edge_monogenic_phase_on_luminance(self, fusion):
        image = load_synth_yeast_plate()
        colour = FocusEdgeColorPhase(
            fusion=fusion, chroma_weight_1=0.0, chroma_weight_2=0.0
        )
        result = colour._color_phase_congruency(image)

        lum = np.asarray(image.color.Lab[..., 0], dtype=np.float64)
        reference = monogenic_phase_congruency(lum)

        np.testing.assert_allclose(result.pc, reference.pc, rtol=1e-10)
        np.testing.assert_allclose(result.orientation, reference.orientation, rtol=1e-10)
        np.testing.assert_allclose(result.feature_type, reference.feature_type, rtol=1e-10)
```

- [ ] **Step 2: §7 test 3 — the `[0,1]` bound under random weights**

```python
class TestTheUnitIntervalBoundHolds:
    @pytest.mark.parametrize("fusion", ["joint", "coherent", "l2"])
    @pytest.mark.parametrize("seed", range(5))
    def test_detect_mat_stays_in_the_unit_interval(self, fusion, seed):
        rng = np.random.default_rng(seed)
        op = FocusEdgeColorPhase(
            fusion=fusion,
            chroma_weight_1=float(rng.uniform(0, 8)),
            chroma_weight_2=float(rng.uniform(0, 8)),
        )
        out = op.apply(load_synth_yeast_plate()).detect_mat[:]
        assert np.isfinite(out).all(), "NaN passes a naive 0 <= x <= 1 check"
        assert 0.0 <= out.min() and out.max() <= 1.0
```

The `isfinite` assertion is not decoration. Drift `M10` records that an all-NaN `detect_mat` passes a
naive range check, because NaN compares false to everything.

- [ ] **Step 3: §7 test 4 — masked scale invariance, with a derived tolerance**

Per Task 0 Step 5. Compare `out(w)` and `out(c·w)` over pixels where `out(w) > 0.05`; bound the
deviation by `ε/(c·A_total) + (g/4)·|Δwidth|`; assert the measured deviation is inside the bound
**and** that the bound is tighter than `0.05` on this image, so the test can fail.

- [ ] **Step 4: §7 test 5 — the operation contract**

`rgb`/`gray` unmutated; `to_json`/`from_json` round-trip; constructible with no arguments; 90°
rotation equivariance (`np.rot90` the `rgb`, then `np.rot90` the output back — equal to `rtol=1e-8`,
which is the FFT's own reproducibility, not a guess).

- [ ] **Step 5: §7 test 6 — the two guards**

```python
class TestGuards:
    def test_achromatic_input_raises(self):
        image = load_synth_yeast_plate()
        gray = image.rgb[:].mean(axis=2)
        image.rgb[:] = np.stack([gray] * 3, axis=-1).astype(image.rgb[:].dtype)
        with pytest.raises(Exception) as excinfo:
            FocusEdgeColorPhase().apply(image)
        chain, err = [], excinfo.value
        while err is not None:
            chain.append(err)
            err = err.__cause__
        assert any(isinstance(e, ValueError) and "chromatic" in str(e) for e in chain), (
            "ImageOperation.apply wraps twice -- RuntimeError at "
            "_image_operation.py:423, then a bare Exception at :470 -- so the "
            "ValueError's type survives only on the __cause__ chain. Measured: "
            "RuntimeError -> Exception -> ValueError."
        )
```

- [ ] **Step 6: Drift `C16` — demonstrate the hue-wrap artifact, don't just assert it**

```python
class TestHueWrapArtifactIsReal:
    """Drift C16. `hsv` bandpasses raw hue across its 0.99 -> 0.01 seam.

    Build a flat, constant-saturation, constant-value image whose hue varies smoothly
    THROUGH red. There is no edge in it. `color_space="hsv"` must respond anyway;
    `color_space="lab"` must not. If this test ever goes green on `hsv`, the operation
    silently started unwrapping and diverged from CMPCM.
    """

    def test_hsv_manufactures_an_edge_where_lab_sees_none(self):
        ...  # hue ramp through 0.0, S = 0.6, V = 0.6, 128x128
        assert hsv_response.max() > 10 * lab_response.max()
```

- [ ] **Step 7: §7 test 7 — doctests. They must not assert chroma behaviour**

`load_synth_yeast_plate`'s `b*` has `std = 0.158` and would need `chroma_weight_2 = 27.8` to reach
parity with `L*` — far outside `TuneSpec(0.0, 8.0)`. It is **not** representative, and it is the
doctest plate. Assert only the `[0,1]` bound.

```bash
uv run pytest --doctest-modules src/phenotypic/enhance/_focus_edge_color_phase.py -q
```

- [ ] **Step 8: §7.1 — the PFOM ranking regression**

In `tests/unit/enhance/test_color_phase_pfom.py`. Build a synthetic geometric colour image with
known ideal edges. Compute Pratt's Figure of Merit on the **un-normalized** `l2` output (via
`_color_phase_congruency`, not `apply`, so the clip does not truncate). Assert the *ranking*
`colour PC > PC > Canny`, at `color_space="hsv", fusion="l2"`.

**Do not attempt to reproduce CMPCM's Table 1.** Its PFOM values
(`Canny 0.8888 · Log 0.9008 · VPMM 0.9934 · PC 0.9099 · MPC 0.9321 · CMPCM 0.9989`) are measured on
Fig. 4a's "geometry image" (`176 × 298`), **whose pixels are published nowhere**. The paper's §2
pixel spec (`173 × 299`) describes Fig. 1a1, a different image. The numbers are transcribed in
`references.md` for one purpose only: the CMPCM/VPMM gap is `0.0055`, so a ranking regression that
separated them would need better than ~0.5% PFOM resolution. Ours separates `colour PC` from `Canny`,
a much larger gap.

- [ ] **Step 9: Prove three of these tests can fail**

| Test | Mutation | Must go red |
|---|---|---|
| `TestZeroChromaReducesToTheMonogenicPort` | pin `weights[0] = 0.9` | yes |
| `TestHueWrapArtifactIsReal` | `_LUMINANCE_FIRST["hsv"] = (2, 1, 0)` (swap H and S) | yes |
| `test_achromatic_input_raises` | delete the `np.array_equal` guard | yes |

Revert each. A test that cannot fail is not a test.

- [ ] **Step 10: Full suite + commit**

```bash
uv run pytest tests/unit/enhance/ tests/unit/abc_/ tests/unit/tune/ -q
git add tests/unit/enhance/
git commit -m "test(enhance): FocusEdgeColorPhase contract, fidelity, guards, PFOM ranking

Masked scale-invariance with a derived tolerance (unmasked it moves 779%, C17).
Hue-wrap artifact demonstrated, not asserted (C16). Achromatic guard walks the
__cause__ chain, because ImageOperation wraps twice. Three mutations proven to
redden the suite."
```

---

## GATE B — after Tasks 2, 3, 4

Deep review over the combined diff. **Opus/high.** Charge:

1. Every `file:line` citation in the new docstrings — open the file, check the line.
2. `fuse_l2` must not divide by `‖w‖`; `_color_phase_congruency` must not clip.
3. The `hsv` reorder `(2, 0, 1)` must actually put `V` first. Print the channel means and compare
   against `image.color.hsv[..., 2].mean()`.
4. `np.hypot` must appear nowhere.
5. `chroma_weight_*` must be in the tune coverage gate's output, not merely annotated.
6. Ask the question the whole suite exists to answer: **which of these tests would still pass if the
   code were wrong?** Report every one that would.

---

## Task 7: Documentation sweep

**Shape:** Sweep. **Depends on Task 5** — the `fusion` default may have moved.

**Files:**
- Modify: `src/phenotypic/enhance/CLAUDE.md`
- Modify: `src/phenotypic/abc_/_enhance_markers/_focus_edge.py`
- Modify: `docs/source/explanation/what_enhancement_does.md`

- [ ] **Step 1: `enhance/CLAUDE.md` — extend the phase-congruency section**

Add, after the existing three bullets:

> - **`FocusEdgeColorPhase` reads `image.rgb`, not `detect_mat`.** It is the second operation in this
>   package that is a pipeline *source* rather than a transform — `SetDetectMode` is the first — so
>   **any enhancer placed before it has no effect.** Drift `C2`.
> - **Only `coherent` builds a fused monogenic vector.** `joint` sums scalar energies and `l2` sums
>   three finished congruency maps. The `orientation` / `feature_type` on `ColorPhaseResult` are
>   therefore computed from a vector that, under two of three modes, did not produce `pc`. They are
>   deliberately **not** exposed via `output`. Drift `C15`.
> - **`color_space="hsv"` bandpasses raw hue across its wrap discontinuity** and manufactures a
>   phantom edge at near-red boundaries. Retained because CMPCM uses HSV; `lab` is the default and
>   has no seam. Drift `C16`.
> - **The output is not invariant to a global rescale of the weight vector.** The `ε` in
>   `A_max + ε` sits inside a `g = 10` sigmoid; measured, `c ∈ {0.01, 100}` moves the output by 100%
>   at `L*` amplitudes and 779% at low ones. Drift `C17`. An earlier spec revision claimed "~1%".

- [ ] **Step 2: `_focus_edge.py` — amend the marker ABC docstring**

The class docstring currently promises *"Edge isolation is therefore confined to
`image.detect_mat`."* That is still true of **writes**, but one subclass now *reads* `rgb`. Add:

> **One subclass sources from `rgb`.** :class:`FocusEdgeColorPhase` reads ``image.rgb`` (through
> ``image.color``) because colour phase congruency is defined on colour, and ``rgb`` is not a
> supported ``detect_mat`` layer. It still writes only ``detect_mat``, so the integrity check holds.
> The user-visible consequence is that it ignores any enhancer placed before it in a pipeline.

- [ ] **Step 3: `what_enhancement_does.md` — add the operation**

Mirror the `FocusEdgeMonogenicPhase` entry. State the `rgb`-source caveat.

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/enhance/CLAUDE.md src/phenotypic/abc_/_enhance_markers/_focus_edge.py \
        docs/source/explanation/what_enhancement_does.md
git commit -m "docs(enhance): document FocusEdgeColorPhase's rgb sourcing and three hazards"
```

---

## GATE D, then the simplify pass

- [ ] `uv run pytest tests/unit/enhance/ tests/unit/abc_/ tests/unit/tune/ tests/unit/detect/ -q`
- [ ] `uv run mypy --no-incremental src/phenotypic` — no **new** errors (**422** in 128 files at
      HEAD on a clean cache). **Use `--no-incremental`**: the incremental cache reports `418` and
      the count drifts as files are touched, so a stale cache can invent or hide a regression. To
      attribute an error, `grep` for the changed filename rather than trusting the total.
- [ ] `uv run ruff check src/phenotypic` — no **new** errors (31 pre-existing, none in files this
      branch owns)
- [ ] `git diff main --stat` — no change to `pyproject.toml`, `uv.lock`, `tests/fixtures/*.npz`
- [ ] `uv run python -c "import phasepack"` → `ModuleNotFoundError`
- [ ] `git ls-files | grep -c '\.pdf$'` → `0`
- [ ] Simplify pass (Opus/high): dedupe, reduce, clarify. **No behaviour change.** Then re-run the
      full affected suite. The last simplify pass on this branch found drift `M10`'s misplaced guard
      and drift `M12` — treat its findings as first-class.

---

## FINAL GATE — Cluster E: extend the stripped sandbox

**Shape:** Seam **and** Sweep. The strip is a correctness boundary: if it leaks, the review is
worthless. **Opus/high.**

The sandbox already exists and is **additive** — do not rebuild it.

```
~/.claude/refs/phenotypic-alt-phase-detection/math-review/
```

48 files, 11 MB, 7/7 gates, 156 tests passing standalone. It lives **outside every git work tree**,
which is the hard guarantee for the copyrighted PDFs. `math_review_corpus.py refresh` **cannot damage
it**: if any banned word survives a strip, it writes `<file>.candidate`, leaves the live file
byte-for-byte untouched, prints the offending lines, and exits `1`.

**This is not just a `refresh` run.** The colour operation reads `image.rgb`, and the sandbox's
hand-authored `Image` shim has no colour accessor. Four things must be built:

- [ ] **Step 1: Extend `math_review_corpus.py`'s tables**

- `RENAMES`: add `(r"\bFocusEdgeColorPhase\b", "ColorPhaseCongruencyEnhancer")`. Place it **above**
  `(r"\bFocusEdge\b", ...)`. (It would in fact survive below it — `\b` does not match inside
  `FocusEdgeColorPhase` — but the table's own comment establishes the longest-first convention, and
  the next editor will not re-derive that.) Add `load_synth_filamentous_plate → load_sample_image_d`.
- `DERIVED`: `_color_phase_kernels.py`, `_focus_edge_color_phase.py`, the three new test files,
  `fusion_algebra.py`.
- `IMPORT_REWRITES`: rewire each to the flat `kernels.` / `tests.` package.
- `PROSE_PATCHES`: `FocusEdgeColorPhase`'s class docstring names "colony", "agar", "plate" in
  framing, not just in nouns. Record the rewrite verbatim, as was done for the other two.
- `MANIFEST`: add all of the above. **A silently-missing file is the likeliest failure**, and gate 4
  exists for exactly that.
- `MUST_BE_AST_IDENTICAL`: add `_color_phase_kernels.py` — it has no host imports, so nothing
  legitimate can change in it. `fusion_algebra.py` likewise.

- [ ] **Step 2: Grow `kernels/_standalone.py` a colour accessor** (hand-authored, `PRESERVED`)

`Image` needs `.color.Lab[...]` and `.color.hsv[...]` over its `rgb`. Use the standard sRGB → XYZ →
CIELAB chain (D65) and `colorsys`-equivalent HSV, both vectorised. **Do not import `skimage`** — the
sandbox is stdlib + numpy + pytest.

- [ ] **Step 3: Grow `kernels/_data.py` three generators** (hand-authored, `PRESERVED`)

`load_sample_image_a/b/c` exist. Add:
- a **chromatic** sample image (the current ones may be near-achromatic; check, or the achromatic
  guard fires on every colour test),
- an **achromatic** one, so the guard test has an input,
- a **labelled** one with an object map, for the CA experiment's stripped analogue.

- [ ] **Step 4: Add gate 8, and extend gate 6**

- **Gate 8:** the stripped `fusion_algebra.py` runs and prints `4/4 checks passed.`, exit `0`.
- **Gate 6** (fixture reproduction) already covers `phasecongmono_golden.npz`. Extend it to assert the
  stripped colour kernels reduce to the stripped monogenic port at zero chroma weight — the sandbox's
  own version of §7 test 1.

- [ ] **Step 5: Run all eight gates**

```bash
# `ulimit -v` is a no-op on Darwin. Bound with `timeout`.
timeout 1800 uv run python docs/superpowers/logic_validation_scripts/2026-07-09-focus-edge-monogenic-phase/math_review_corpus.py \
    refresh --sandbox ~/.claude/refs/phenotypic-alt-phase-detection/math-review --repo .
uv run python docs/superpowers/logic_validation_scripts/2026-07-09-focus-edge-monogenic-phase/math_review_corpus.py \
    verify  --sandbox ~/.claude/refs/phenotypic-alt-phase-detection/math-review --repo .
```

All eight must pass, and the standalone suite must run green **inside the sandbox**:

```bash
cd ~/.claude/refs/phenotypic-alt-phase-detection/math-review && uv run --no-project --with numpy --with pytest pytest -q
```

**Ban-list** (unchanged): `colony colonies agar plate yeast fungi fungal hypha hyphal mycel septa
microbe microbio phenotyp petri culture biolog organism`. Gate 1 sweeps the whole sandbox **excluding
only `papers/`** — `refs/` is *not* excluded, and provably has zero hits. The two hits in `papers/`
are the word "Biolog" inside *Biological Cybernetics* in two reference lists.

- [ ] **Step 6: Mirror to the scratchpad working copy, and commit only the tool**

The sandbox itself **never enters the repo**. Only `math_review_corpus.py`'s diff is committed.

---

## FINAL GATE — Cluster F: the scoped Fable review

**Model: Fable 5 (`claude-fable-5`), high effort.** Frontier tier, so it never reviews work produced
by a stronger model.

`cwd = <sandbox>/math-review/`. **It may read only files beneath that directory.** No repo paths
anywhere in the brief. No biological framing: the corpus is described purely as a signal-processing
port and its colour extension.

**What F receives:** `kernels/` (now including `_color_phase_kernels.py` and
`_focus_edge_color_phase.py`) · `tests/` and both `.npz` fixtures · `verify_claims.py` (21 executable
checks) and `fusion_algebra.py` (4 more) — it can run both and try to break them · all six `spec/`
documents, as **claims to check against sources, not to trust** · `plan/plan.md` and `plan/reviews/`,
because shortcuts hide in the justification rather than in the diff · `refs/` (11 reference files) so
that every *"the reference says X"* is checkable at `file:line` · `refimpl/` · `papers/` (5 PDFs) ·
`WebSearch` / `WebFetch`.

**What F must be told is ABSENT, so it cannot "verify" against a source it cannot open:**

- **Wang Lijuan, Zhang Changsheng, Liu Ziyu, Sun Bin, Tian Haiyong (2014)**, 26th CCDC, 2033–2038,
  DOI `10.1109/CCDC.2014.6852502` — the nominal monogenic-PC source. **Never read.** IEEE returns
  `502` to every non-browser client. Any finding turning on it must be reported `unverifiable` —
  never confirmed, never refuted.
- `refs/cmpcm_matlab/` (Vivianyuwei) — never fetched. **No licence, all rights reserved.** If
  obtained, read-only; **never copy its code.** `references.md:709` records that it implements a
  *precursor grayscale* paper, not CMPCM.

**Constraints:** memory ceiling — no array above `5e6` elements, no meshgrid above `2000 × 2000`,
under 500 MB live. **Do not rely on `ulimit -v`**: it is a silent no-op on macOS and does not abort
the command chain (Global Constraints). Assert `arr.size <= 5_000_000` before allocating, and bound
every run with `timeout`. `papers/` is copyrighted — never commit it.

**Charge.** *Faithfulness to validated reference logic beats a convenient shortcut.* Find shortcuts,
unstated assumptions, silent side-picking where the references disagree, tolerances too loose to
catch anything, and tests that could pass while the code is wrong. Specifically:

> This specification has been **wrong about its references five times** — `perfft2`, the `max(T, ε)`
> floor, the odd-grid divisor, the Riesz division, and `np.hypot` — each time by generalising from
> the one implementation that happened to be runnable. It has since been wrong about **its own
> arithmetic twice more**: §3.1's response column matches no single `deviation_gain`, and §4.2's
> "~1%" understated a 100% effect by two orders of magnitude. Both were caught before implementation.
> **Assume an eighth error exists and find it.**

Give F the mutation-audit table from the monogenic review, and the question the entire suite exists
to answer: **"Which of these tests would still pass if the code were wrong?"**

**Leak detector.** If F names the application domain unprompted — colonies, agar, plates,
microbiology — then **E leaked.** Treat that as an E failure, re-strip, and do not believe the review
until it is clean.

**F fixes nothing.** Report only. The orchestrator triages, surfaces design-level conflicts to the
user, then applies changes to the *real* spec and code. The report lands in
`docs/superpowers/plans/2026-07-09-focus-edge-color-phase/reviews/`.

**Gate.** Every finding is closed exactly one of three ways: **(a)** reproduced and fixed;
**(b)** reproduced and consciously accepted, with a new drift row; **(c)** refuted with evidence.
**No finding is closed by assertion.**

---

## Known gaps in this plan

Found by running `writing-plans`' self-review over the finished draft. Recorded rather than papered
over, because a step that *looks* complete and is not is worse than one that admits it.

Four steps carry a **specification** — inputs, method, and the exact assertion — rather than literal
code. Each is a body an Opus/high implementer writes from the spec, and each has a stated pass
condition that cannot be satisfied by accident:

| Step | What is specified but not written out | Why |
|---|---|---|
| Task 5 Step 1 | The CA injection and the boundary-error metric | It is an *experiment*. Writing its body here would fix its outcome in the plan, which is exactly what §7.2 forbids. |
| Task 6 Step 3 | The masked scale-invariance test | Its tolerance must be *derived at implementation time* from the measured `A_total` on the real image, not transcribed. |
| Task 6 Step 4 | Round-trip, rotation equivariance, immutability | Mechanical; the four assertions are named. |
| Task 6 Steps 6, 8 | The hue ramp and the PFOM harness | The assertion is exact (`hsv > 10 × lab`; ranking `colour PC > PC > Canny`); the image construction is not. |

Two further items are deliberately deferred, not forgotten:

- **The GUI builder dropdown** (§8 criterion 7) has no task of its own. The registry walks
  `phenotypic.enhance.__all__`, so Task 4 Step 5 satisfies it, and GATE D's suite covers it. If the
  builder turns out to enumerate operations some other way, that becomes a Task 7 step.
- **`conformal-lift.md`'s three-arm junction gate** (`U4`, the only open question in the drift
  register) is out of scope. `lift="conformal"` raises. Nothing in this plan advances or retires it.

---

## Acceptance criteria

Spec §8, with the cross-references repaired.

1. §7 test 1 (per-channel fidelity) passes — the fusion does not perturb the port, `rtol=1e-10`.
2. §7 test 2 (fusion sanity) passes for `joint` and `coherent`.
3. The `[0,1]` bound holds for all three fusion modes under random weights, **and the output is
   finite** (NaN passes a naive range check).
4. §7.2's CA experiment has been **run and recorded**, and the shipped `fusion` default matches it.
   A null result is acceptable and must not be buried.
5. `lift="conformal"` raises `NotImplementedError`.
6. `uv run mypy src/phenotypic` and `uv run ruff check` introduce no new errors.
7. The operation appears in the GUI builder's enhancer dropdown (the registry walks
   `phenotypic.enhance.__all__`; `ImageEnhancer` subclasses go 30 → 31).
8. `monogenic_phase_congruency` is **bit-identical** across Task 1's refactor, proven by a test that
   the golden fixture's `rtol=1e-6` demonstrably cannot replace.
9. Every finding from cluster F is closed by (a), (b) or (c) above.

## Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | Task 1's refactor silently moves a bit and every fixture still passes. | `TestTheRefactorMovesNoBits` uses `np.array_equal`, not `rtol`. Proven able to fail via `np.hypot`, a mutation the golden fixture does **not** catch. GATE A reproduces it independently. |
| 2 | `joint` is unvalidated against any external reference. | Task 5 is its acceptance test, with a stated falsification and a fallback (`l2`). |
| 3 | Chroma is 69–80% outvoted by luminance on real plates, so colour may buy little. | Task 5 measures it against a luminance-only baseline. A null result is acceptable and must be recorded. |
| 4 | `T_total = Σ wᵢTᵢ` and `A_max = Σ wᵢ maxₛA` are inventions with no reference. | 1-homogeneous, so §4.2's two-degrees-of-freedom argument survives. Drift `C10`. Untested against anything external — **say so**. |
| 5 | `rgb` sourcing surprises pipeline authors. | Class docstring `Note:`, `enhance/CLAUDE.md`, the marker ABC. Drift `C2`. |
| 6 | Someone enables `lift="conformal"` before the gate passes. | It raises at construction. Task 4 Step 6 verifies pydantic does not swallow the type. |
| 7 | Cluster E leaks the application domain and F's review is worthless. | Eight gates, ban-list over everything but `papers/`, and the leak detector: if F names the domain unprompted, re-strip. |
| 8 | `hsv` hue-wrap artifact is discovered by a user rather than by us. | Drift `C16`, demonstrated by a test that would go green if someone "fixed" it by unwrapping. |
