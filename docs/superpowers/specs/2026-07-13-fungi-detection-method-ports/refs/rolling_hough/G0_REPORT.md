# A09 Rolling Hough G0 report

## Verdict

**The current planned A09 contract remains blocked. A narrower Clark source core has complete G0
evidence and is a candidate for independent approval. Production remains frozen.**

The source-author executable does not define the plan's pixelwise `coherence` field, does not call
raw counts its persisted angular accumulator, uses a smoothing radius rather than the planned
diameter, and exposes a Hough-normal angle with source-specific zero behavior. Implementing the
current `RollingHoughResult` would therefore require invented or unresolved semantics.

`SOURCE_CONTRACT.md` removes those unsupported fields and translations. It freezes an exact Clark
core returning theta, supports, dense raw counts, source threshold residuals, unnormalized
response, Hough-normal orientation, eligibility, and validity. Its eight Python-boundary and dense
representation deviations are explicit in `DRIFT.md`.

## Gate evidence

| G0 requirement | Status |
|---|---|
| Primary paper, PDF source, DOI, age caveat | Pass |
| Complete immutable source-author archive | Pass, MIT commit `4d06f9f...` |
| Complete stable FilFinder test-oracle archive | Pass, MIT v1.8 commit `22539cf...` |
| Exact line-addressable executable sources | Pass |
| Runnable source probes | Pass |
| All-output source-generated fixture | Pass |
| Fixture integrity manifest and source-free verifier | Pass |
| Axes, theta range, line rasterization, threshold, borders, ties, sentinels, dtypes | Pass for candidate core |
| Every deviation registered | Pass for candidate core |
| Planned `coherence` and wrapper semantics | **Fail, explicitly excluded** |
| Independent G0 reviewer | Pending |

The fixture also establishes an executable defect absent from the preliminary plan: a no-output
image reaches source persistence with one-dimensional empty `Hthets`, then raises `IndexError`
while requesting `shape[1]`. D06 defines a safe empty candidate result and preserves the exact
source failure as fixture evidence.

## Required decision

An independent reviewer should either:

1. approve `SOURCE_CONTRACT.md` as the only authorized A09 implementation scope; or
2. keep A09 blocked and request a separately specified novel coherence/wrapper design.

Approval of the narrow core would not approve `FocusEdgeRollingHough`, public exports, registries,
or a biological-performance claim. Any later coherence field or wrapper must receive a new design
gate and returning source-fidelity review.
