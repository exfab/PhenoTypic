# A09 Rolling Hough G0 Research Checkpoint

## Status

G0 is not yet approved. Production is frozen. This checkpoint pins the primary paper,
source-author executable, and stable FilFinder test oracle, then records the contract mismatches
found before A09 was preempted by an approved A06 implementation.

## Pinned authorities

- Clark, Peek, and Putman, *Magnetically Aligned H I Fibers and the Rolling Hough Transform*,
  arXiv:1312.1338v2 / ApJ 789, 82 (2014). The PDF, complete arXiv source archive, and line-addressable
  `clark_2014_fibers_cpp.tex` are local. The work is more than five years old.
- Source-author repository `seclark/RHT`, immutable commit
  `4d06f9fa4cafe9022011a0bec0315390d7e23c39`, MIT license
  (`source_clark/LICENSE:1-21`). The numerical `rht.py` file last changed at commit
  `40cdc11ebcad9625c1dac46e5f8eef61db8857c5` on 2021-05-18; later selected-commit changes do not
  alter it.
- Test-only FilFinder stable v1.8, signed tag commit
  `22539cf2176ad9b717658652e8da749158597f4d`, MIT license
  (`source_filfinder/LICENSE.rst:1-20`). It is not a runtime dependency or the full-pipeline
  authority.

The exact source probe in `source_contract_probe.py` imports both pinned files directly. Clark's
deprecated `np.int` and `np.float` names are restored as import compatibility aliases, and Astropy
is stubbed because the probed numerical helpers do not perform FITS I/O. No numerical source line
is patched or reimplemented.

## Established Clark behavior

- The paper defines top-hat unsharp masking by smoothing diameter $D_K$, subtraction, and a strict
  positive bitmask (`clark_2014_fibers_cpp.tex:88-90`). The executable instead names its parameter
  `radius` and constructs a disk of diameter `2 * radius + 1`
  (`source_clark/rht/rht.py:539-556`).
- The paper specifies a circular window of diameter $D_W$, rho zero, percentage threshold $Z$,
  and canonical theta-bin count (`clark_2014_fibers_cpp.tex:98-104`).
- The executable requires a positive odd window diameter and positive integer smoothing radius
  (`source_clark/rht/rht.py:764-792,905-942`). It uses
  `ceil(pi * (wlen - 1) / sqrt(2))` bins on `[0, pi)`
  (`source_clark/rht/rht.py:259-264,780-789`).
- The circular mask includes pixels whose center distance is exactly the integer radius
  (`source_clark/rht/rht.py:530-537`).
- Center-line rasterization uses the Hough normal equation, NumPy round-to-nearest-even, and the
  rho-zero accumulator bin (`source_clark/rht/rht.py:592-630`). The resulting line support is
  angle dependent. For `wlen=11`, the probe observes support from 7 through 13 pixels, even though
  $D_W=11$.
- The executable computes raw integer counts `h`, divides each bin by its angle-dependent support
  `h1`, subtracts `frac`, and clips negative values
  (`source_clark/rht/rht.py:799-837`). Its persisted `hthets` are therefore floating threshold
  residuals, not raw integer center-line counts (`source_clark/rht/rht.py:842-856`).
- Exact threshold equality yields residual zero. Although the source multiplies by a `>= 0` mask,
  the later `np.any(hout)` means equality alone is not emitted
  (`source_clark/rht/rht.py:835-846`). The source probe proves this case.
- Sparse output exists only for centers admitted by `wlen_mask`; the two-stage source mask excludes
  edges and neighborhoods around nonfinite input rather than rejecting the full image
  (`source_clark/rht/rht.py:403-488,518-528,827-845`).
- The source backprojection is the sum of positive residuals, then is globally divided by its image
  maximum before persistence (`source_clark/rht/rht.py:813-846,864-882`). The all-zero case has no
  separately specified finite normalization.
- The axial angle uses doubled-angle weighted sums and returns values equivalent modulo pi
  (`source_clark/rht/rht.py:667-692`). A zero-weight vector returns pi rather than an invalid
  sentinel, and a perfectly orthogonal equal-weight crossing has no source tie policy beyond
  floating-point `atan2`.
- The source does not define the plan's `coherence` field.

## FilFinder scope

FilFinder v1.8 is a materially modified, global skeleton orientation analysis. It pads a binary
mask, builds a strict-radius circle, samples a caller-selected theta count with a duplicated
endpoint, aggregates votes across every nonzero skeleton pixel, drops the duplicated endpoint,
subtracts a background percentile, and reports a global circular mean and confidence interval
(`source_filfinder/fil_finder/rollinghough.py:8-100,103-128,147-167,186-232`). It does not perform
Clark preprocessing or return a pixelwise Clark accumulator. The source probe confirms that an
`ntheta=18` call returns 17 bins and produces axial orientation agreement on simple straight
skeletons only.

FilFinder can therefore be a simple-template orientation cross-check. It cannot be a golden oracle
for Clark window rasterization, preprocessing, threshold residuals, borders, response, coherence,
or per-pixel validity.

## Blocking mismatches with the planned contract

1. `smoothing_diameter` is not mapped to the executable's smoothing radius. An odd-diameter API
   could deliberately set `radius = (diameter - 1) // 2`, but that translation needs approval and
   a drift row.
2. `angular_accumulator` is described as raw integer counts "from the pinned source," while the
   source's public angular product is a float residual after support normalization and threshold
   subtraction. The result schema must either expose both products under accurate names or select
   one explicitly.
3. The plan does not define `response`. A source-near local response is
   `sum(max(count / support - Z, 0))`; the persisted source additionally normalizes this map by its
   global maximum. These contracts are different.
4. The plan does not define `coherence`. Neither Clark nor FilFinder v1.8 supplies that pixelwise
   field. An axial mean-resultant length would be a new derived capability and requires an explicit
   equation, zero-weight behavior, and source or reasoning label.
5. The plan says threshold rounding and inclusive comparison must be frozen, but the executable
   uses floating residuals with effective strict-positive emission. No integer `ceil(Z * D_W)` rule
   matches it because support varies by theta.
6. Border support, nonfinite pixels, all-zero input, zero-weight orientation, and orthogonal ties do
   not match the proposed finite/NaN result semantics without intentional deviations.
7. Dtype is unresolved: source counts are platform integers, residual calculations are float64 in
   memory, and saved angular arrays are converted to float32
   (`source_clark/rht/rht.py:83-85,605-630,835-856`).

## Required continuation

Before implementation, an amended A09 contract must freeze:

- smoothing radius versus odd smoothing diameter;
- whether raw counts, source residuals, or both are public;
- local raw response versus globally normalized backprojection;
- the exact coherence equation and zero/tie semantics;
- source-compatible masked borders/nonfinite handling versus whole-input rejection;
- axial range and the representation of source angle pi as zero;
- in-memory and returned dtypes.

After amendment, extend the source harness into an all-output fixture, add an independent direct
oracle, derive tolerances, create the mutation matrix, and request independent G0 review. No A09
production file, wrapper, test, logic-validation script, or dependency change exists in this
checkpoint.
