# A09 source and fixture reconciliation

## Paper to executable

The 2014 paper is more than five years old in a fast-moving software field. It establishes the
top-hat unsharp-mask procedure, strict zero threshold, circular rolling domain, rho-zero Hough
mapping, percentage gate, canonical theta count, and angular backprojection
(`clark_2014_fibers_cpp.tex:88-108`). The selected source-author executable is authoritative for
the exact discrete behavior.

| Contract stage | Executable lines | Fixture fields | Result |
|---|---|---|---|
| Bad-pixel and edge masks | `source_clark/rht/rht.py:403-488,518-528` | `cNN_smoothing_mask`, `cNN_window_mask`; nonfinite/border case | Exact source calls |
| Inclusive smoothing disk | `source_clark/rht/rht.py:530-548` | `cNN_smoothing_kernel`, `cNN_correlated`, `cNN_smoothed` | Exact source calls and source SciPy correlate |
| Strict-positive unsharp bitmask | `source_clark/rht/rht.py:549-556` | `cNN_unsharp`, `cNN_bitmask`; constant case | Exact source call |
| Theta count and grid | `source_clark/rht/rht.py:259-264,780-789` | `cNN_theta`, `local_theta` | Exact source calls plus source construction |
| Circular window and rho-zero center lines | `source_clark/rht/rht.py:530-537,592-664` | `cNN_circular_window`, `cNN_center_lines`, local templates | Exact source calls |
| Angle-dependent support | `source_clark/rht/rht.py:799-801` | `cNN_support_counts`, `local_support_counts` | Exact source `fast_hough` |
| Window raw counts | `source_clark/rht/rht.py:827-834` | `cNN_raw_counts`, local template counts | Instrumented source `fast_hough` at every source-eligible center |
| Threshold residual and sparse validity | `source_clark/rht/rht.py:835-846` | `cNN_threshold_residual`, `accepted_bins`, `valid`, sparse source fields | Dense instrumented calculation is compared exactly with `window_step`'s persisted sparse arrays |
| Raw response and persisted backprojection | `source_clark/rht/rht.py:813-846,864-882` | `raw_response`, `source_backprojection`, `source_attempted_backprojection` | Raw sum exact; source normalized output exact when nonempty |
| Axial Hough-normal angle | `source_clark/rht/rht.py:667-692` | `derived_orientation`, local source angles | Exact source helper on positive residuals |
| Empty-output failure | `source_clark/rht/rht.py:842-866` | constant case `source_error` and all-NaN attempted backprojection | Exact `IndexError` captured, not suppressed or relabeled as a successful source output |

## Fixture corpus

`generate_fixture.py` imports both pinned source files directly and never imports `phenotypic`.
Five full Clark cases cover an asymmetric horizontal structure, nondefault crossing, gapped
diagonal, border plus NaN/Inf, and constant input. It records every preprocessing stage, mask,
theta bin, center-line voxel, support, raw count, threshold residual, accepted bin, sparse source
coordinate/residual, response, attempted/persisted backprojection, Boolean validity, and
orientation. The source threshold parameter is recorded as `threshold_fraction`; coherence is
deferred and absent.

Six local diameter-11 templates capture horizontal, vertical, diagonal, crossing, gap, and full
circle counts, fraction-one equality behavior, and source angles. The same fixture records exact
FilFinder v1.8 outputs for three straight binary skeletons. The manifest hashes all numeric and
string variables canonically, independent of NPZ ZIP metadata. `verify_fixture.py` checks the hash
and all cross-field invariants without importing either source.

Integer counts and masks compare exactly. Clark sparse residuals and backprojections compare at
zero tolerance because both fixture paths execute the same NumPy operations in the same runtime.
The pinned source probes use exact zero-ULP angle controls rather than a decimal tolerance. Future
production comparisons must derive any wider bound from its operation count and runtime contract.
The standalone logic validator independently re-counts every eligible local window from captured
geometry and re-derives residual, response, Boolean validity, and axial collapse without importing
either source.

## FilFinder reconciliation

FilFinder v1.8 is a separate MIT implementation and a deliberately limited cross-check. It uses a
strict-radius circle, duplicated angular endpoint followed by endpoint removal, global aggregation
over nonzero skeleton pixels, percentile background subtraction, and a doubled-angle mean
(`source_filfinder/fil_finder/rollinghough.py:8-100,103-128,147-167`). It does not match Clark's
preprocessing, center-line discretization, local threshold residual, borders, or pixelwise result.
Agreement on simple horizontal, vertical, and diagonal Hough-normal angles is corroboration only.
