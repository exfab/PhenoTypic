# Alternative phase-congruency detectors — spec index

**Date:** 2026-07-08 (restructured 2026-07-09)
**Branch:** `alt-phase-detection` (stacked on `brainstorm-branch-detection`)

Two `FocusEdge` enhancers plus one gated research path. They share a private kernels module and a
source document.

| Document | Operation | Status |
|---|---|---|
| [`monogenic-phase-congruency.md`](./monogenic-phase-congruency.md) | `FocusEdgeMonogenicPhase` | **Built and shipped.** `src/phenotypic/enhance/_focus_edge_monogenic_phase.py`, on the shared `_monogenic_kernels.py`. Eleven recorded drifts, `M1`–`M11`. |
| [`color-phase-congruency.md`](./color-phase-congruency.md) | `FocusEdgeColorPhase` | **Unblocked.** Per-channel monogenic PC + cross-channel fusion. **This is the colour variant.** Builds on the monogenic port; adds no new signal theory. |
| [`conformal-lift.md`](./conformal-lift.md) | `lift="conformal"` | **Gated. May never ship.** The maths is corrected, but `f_z` is an even channel (a Laplacian) that contributes nothing to `pc`. Gated on a three-arm junction experiment. |
| [`references.md`](./references.md) | — | Recovered source math, citations, corrected equations, reference implementations. **Read first.** |
| [`drift-register.md`](./drift-register.md) | — | Every deviation from a validated reference, with its justification and status. |
| [`verify_claims.py`](./verify_claims.py) | — | **Executable.** 21 checks re-deriving every mathematical claim in this folder, four against Kovesi's own synthetic test images (MIT) and one against a golden fixture. `uv run python <path>`; exits non-zero on failure. **It does not import `phenotypic`** (project file rule), so it validates its own parallel implementation, *not* `src/`. Its prototype is kept bit-identical to `_monogenic_kernels.py` on `pc`, `orientation` and `feature_type` — and it once was not, which is how the `[0,π)` / `(-π/2,π/2]` orientation split went unnoticed. Shipped code is gated by `tests/unit/enhance/test_monogenic_kernels.py`, which runs against the same fixture. |
| [`tests/fixtures/phasecongmono_golden.npz`](../../../../tests/fixtures/phasecongmono_golden.npz) | — | Golden fixture: `phasepack` 1.5's `pc`/`ft`/`T`/`or` on five 64×64 images. Generated once, dependency dropped. Backs `check_19`. Lives under `tests/fixtures/` so this script and `test_monogenic_kernels.py` share one canonical copy. |

## Why they are split

`FocusEdgeMonogenicPhase` is a **port**. Its correctness question is "does our transcription match?",
which a golden fixture answers.

> **The three references do NOT "agree verbatim."** An earlier revision of this line said they did,
> and that sentence is the single most expensive falsehood in this folder — it licensed four separate
> misattributions, because a reader who believes the references agree will happily quote whichever one
> is open. They disagree in at least four places, and in three of them the *source text looks
> identical*:
>
> | Fork | Julia | MATLAB | `phasepack` | We ship |
> |---|---|---|---|---|
> | Spectrum (`M4`) | `fft2` | `perfft2` | `perfft2` | Julia |
> | `rayleigh_mode` bins (`M6`) | min-anchored | zero-anchored | min-anchored | Julia |
> | Riesz division (`M8`) | componentwise | componentwise | reciprocal-multiply | Kovesi (both) |
> | `noise_method` dispatch (`M7`) | `abs(nm+1) < eps` | exact `== -1` | — | Julia |
> | `T` floor (`M5`) | none | none | `max(…, eps)` | `phasepack`'s guard, attributed |
>
> `M8` is the one to remember: all three print `H = (i*u1 - u2)./radius`, and numpy's `/` is not
> MATLAB's `./`. **Identical glyphs are not identical arithmetic.** `grep` finds agreement; only
> running the reference finds the truth. Both `M6` and `M8` were settled by *executing* Kovesi's Julia
> and diffing the result, not by reading it.
>
> `phasepack` is the least authoritative of the three — it ships no tests, and it has a real bug
> (`references.md` §10.3, the odd-size grid). It is nonetheless the one that generated our golden
> fixture, and the one most likely to be misread as Kovesi, because it is the only one that runs
> under `import`. **Runnability and authority are unrelated.**

`FocusEdgeConformalPhase` is a **derivation**. No trustworthy implementation exists. The single
public one is unlicensed, implements a different (precursor) paper, and contradicts our reading of
the source on the one structural question that matters. Its correctness question is "is the maths
right?", and right now the answer is no: it fails JMIV's analytic curvature ground truth, and the
cause is unknown.

Bundling them would let an unresolved research question block a piece of work that is finished.

## Working principle

Recorded because this spec has already violated it twice, at real cost:

> **Faithfulness to validated reference logic beats a convenient shortcut.**
> When a reference exists, match it. When one does not, apply the reference's *stated principle*,
> instantiated correctly for our case — never its *bank-specific constants*. Every deviation goes in
> `drift-register.md` with the reason and the evidence.

The two shortcuts that cost the most:

1. Folding the paper's planar band-pass into the conformal kernel, justified by an identity that
   holds in `R³` but was applied to an `R²` convolution. It deleted the band-pass's one necessary
   function (guaranteeing a DC-free input to an even, single-signed kernel) and sent two commits
   chasing a DC leak that could not be patched where it was being patched.
2. Copying Kovesi's `(1/mult)^s` noise extrapolation into CMPCM. It looks faithful — it *is* his
   formula, verbatim — but it is the log-Gabor instantiation of a principle, transplanted onto a
   filter bank it was never derived for, and it is 13–28% wrong per scale.

## Next steps

1. ~~Plan and implement `FocusEdgeMonogenicPhase`.~~ **Done.** `_monogenic_kernels.py` +
   `_focus_edge_monogenic_phase.py`, with `FocusEdgePhase` refactored onto the same kernels and
   proven bit-identical to its pre-refactor self at both noise methods.
2. Build `FocusEdgeColorPhase` — per-channel monogenic PC + fusion. **This is the goal.** It reuses
   `_monogenic_kernels` and depends on nothing conformal.

   Two things now exist that did not before. The **CMPCM paper's Table 1 is transcribed**
   (`references.md` §2): `Canny 0.8888, Log 0.9008, VPMM 0.9934, PC 0.9099, MPC 0.9321, CMPCM 0.9989`
   — *not* reproduction targets (the Fig. 4a source image is unpublished), but a real tolerance: the
   `CMPCM`/`VPMM` gap is `0.0055`, so §7.1's ranking regression must resolve better than ~0.5% PFOM.
   And `monogenic_phase_congruency` now **raises** on `n_scale < 2`, `sigma_onf >= 1.0`, `mult <= 1.0`
   (`M9`, `M10`) — it is the direct caller those guards were written for.
3. Only then, run the three-arm junction gate (`conformal-lift.md` §4) to decide whether the conformal
   lift ships at all. Prior evidence argues against it.
4. Separately: correct the CMPCM card in `index-fieldnotebook.html` and `breadth-survey.md` line 129,
   which describe a mechanism the paper does not have (`references.md` §3).
