# Alternative phase-congruency detectors — spec index

**Date:** 2026-07-08 (restructured 2026-07-09)
**Branch:** `alt-phase-detection` (stacked on `brainstorm-branch-detection`)

Two `FocusEdge` enhancers plus one gated research path. They share a private kernels module and a
source document.

| Document | Operation | Status |
|---|---|---|
| [`monogenic-phase-congruency.md`](./monogenic-phase-congruency.md) | `FocusEdgeMonogenicPhase` | **Closed. Buildable now.** A port against three agreeing reference implementations. |
| [`color-phase-congruency.md`](./color-phase-congruency.md) | `FocusEdgeColorPhase` | **Unblocked.** Per-channel monogenic PC + cross-channel fusion. **This is the colour variant.** Builds on the monogenic port; adds no new signal theory. |
| [`conformal-lift.md`](./conformal-lift.md) | `lift="conformal"` | **Gated. May never ship.** The maths is corrected, but `f_z` is an even channel (a Laplacian) that contributes nothing to `pc`. Gated on a three-arm junction experiment. |
| [`references.md`](./references.md) | — | Recovered source math, citations, corrected equations, reference implementations. **Read first.** |
| [`drift-register.md`](./drift-register.md) | — | Every deviation from a validated reference, with its justification and status. |
| [`verify_claims.py`](./verify_claims.py) | — | **Executable.** 15 checks re-deriving every mathematical claim in this folder. `uv run python <path>`; exits non-zero on failure. |

## Why they are split

`FocusEdgeMonogenicPhase` is a **port**. Kovesi's Julia, Kovesi's MATLAB, and the MIT-licensed
`phasepack` agree on it verbatim. Its correctness question is "does our transcription match?", which
a golden fixture answers.

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

1. Plan and implement `FocusEdgeMonogenicPhase` (`superpowers:writing-plans`).
2. Build `FocusEdgeColorPhase` — per-channel monogenic PC + fusion. **This is the goal.** It reuses
   `_monogenic_kernels` and depends on nothing conformal.
3. Only then, run the three-arm junction gate (`conformal-lift.md` §4) to decide whether the conformal
   lift ships at all. Prior evidence argues against it.
4. Separately: correct the CMPCM card in `index-fieldnotebook.html` and `breadth-survey.md` line 129,
   which describe a mechanism the paper does not have (`references.md` §3).
