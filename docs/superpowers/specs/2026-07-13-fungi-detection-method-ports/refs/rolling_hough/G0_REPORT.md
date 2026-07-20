# A09 Rolling Hough G0 report

## Verdict

**The narrow Clark core is ready for independent G0 re-review. Production remains frozen until
PASS.**

The source-author executable does not define pixelwise coherence, does not call raw counts its
persisted angular accumulator, uses a smoothing radius, and exposes a Hough-normal angle with
source-specific invalid behavior. The corrected plan and `SOURCE_CONTRACT.md` therefore authorize
only a source-faithful core returning theta, supports, dense raw counts, threshold residuals,
unnormalized response, Hough-normal orientation, eligibility, and dense Boolean validity.

The core input is exactly a nonempty 2-D float64 NumPy array. Integer and float32 conversion,
coherence, wrappers, and global normalization remain deferred.

## Gate evidence

| G0 requirement | Status |
|---|---|
| Primary paper authority, DOI, revision, and age caveat | Pass |
| Paper license and redistribution disposition | Pass; review evidence only, excluded from package and release artifacts |
| Complete immutable source-author archive | Pass, MIT commit `4d06f9f...` and full 64-character archive hash |
| Complete stable FilFinder test-oracle archive | Pass, MIT v1.8 commit `22539cf...` |
| Exact line-addressable executable sources | Pass |
| Runnable source probes with exact zero-ULP controls | Pass |
| All-output source-generated fixture | Pass |
| Pinned generator, verifier, source revisions, and canonical fixture hash | Pass |
| Standalone source-free numerical logic validator | Pass |
| Wheel package exclusion | Pass; no A09 evidence, docs, fixtures, PDF, or source archive member |
| Axes, rasterization, thresholds, borders, ties, sentinels, and dtypes | Pass for candidate core |
| Every deviation, including Boolean validity conversion | Pass, nine rows |
| Planned coherence and wrapper semantics | Deferred and absent |
| Independent G0 reviewer | Pending |

The fixture preserves the executable's empty-output `IndexError` as evidence. D06 defines the
candidate's safe empty numerical result without relabeling the source failure as success.

## Required decision

The same independent reviewer must either approve the exact narrow contract or return concrete
findings. Approval does not authorize coherence, `FocusEdgeRollingHough`, public exports,
registries, paper redistribution, or biological-performance claims.
