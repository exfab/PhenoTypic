# A08 source-free behavioral contract

This document and `source_fixture.json` are the behavioral inputs for a clean-room production
implementer. They contain equations, public decisions, and source-generated observations, but no
restricted source text or transcription.

## Frozen release slice

- The core accepts a nonempty two-dimensional `numpy.uint8` image containing bright structures on
  a dark background. No float input, wrapper conversion, normalization, or dark-ridge inversion is
  included in this release.
- The response is `numpy.uint8`. Public direction components are floating point in `(row, column)`
  order. `direction_valid` is boolean. Multiscale `winning_scale` is `numpy.int64`, with `-1` where
  the merged response is zero.
- A path length `L` counts vertices, so a complete path has `L - 1` directed edges.
- The exact four forward step sets use `(row, column)` order. Reversing every step in one set
  describes the same complete-path family.

| Orientation | Forward steps | Axial basis `(row, column)` |
|---|---|---|
| o1 | `(1,-1), (1,0), (1,1)` | `(1,0)` |
| o2 | `(-1,1), (0,1), (1,1)` | `(0,1)` |
| o3 | `(-1,0), (-1,1), (0,1)` | `(1,1)` |
| o4 | `(-1,-1), (-1,0), (0,-1)` | `(1,-1)` |

- At each gray level, a pixel survives one path opening exactly when it belongs to at least one
  admissible complete path of `L` vertices in that upper-level set. The grayscale opening is the
  greatest surviving level. Equal-valued plateaus are processed atomically. This deterministic
  paper definition is authoritative.
- Boundary traversal is clipped and never wraps. A path may touch the boundary.
- Robustness `R` applies a clipped square maximum filter with radius `floor(R / 2)`, computes the
  four path openings, then takes `min(original, opened_dilation)` per orientation. The response is
  anti-extensive. In particular, `R=1` equals `R=0`.
- Sort the four robust responses pointwise in ascending order. Intensity is exactly `largest -
  smallest`, with no normalization.
- `path_lengths` is a nonempty tuple of positive, nonboolean integers. Caller order is preserved;
  unsorted values and duplicates are allowed. Multiscale response is the pointwise maximum. A
  strict `>` update means the first caller-supplied length owns an equal positive response.

## Frozen direction validity

Direction is considered only where response is strictly greater than one `uint8` intensity unit.
For the four ascending orientation responses, evaluate splits containing the 1, 2, or 3 largest
responses in the high class. Each cost is the population standard deviation of the low class plus
that of the high class, computed in float32 and then truncated to `uint8`. A strict comparison
retains the first split on a cost tie.

The selected orientation vectors are made axially coherent by fixing the first vector and testing
later signs in `+1`, then `-1`, order. Candidate pairwise-angle sums are computed in degrees and
truncated to an integer before a strict-minimum comparison. The vector sum is normalized.

`direction_valid` is true if and only if all of the following hold:

1. Response is greater than 1.
2. Exactly one of the three truncated split costs is minimal.
3. The value at the low/high split boundary is strictly lower on the low side. This makes the
   selected orientation set independent of rank ordering within equal values.
4. Exactly one tested correction-sign assignment has the minimum truncated angle objective. A
   singleton selected set has one assignment and satisfies this condition.
5. The corrected vector sum is nonzero.

When valid, swap the private `(column, row)` components exactly once, normalize, then canonicalize
the axial sign so `row > 0`, or `row == 0` and `column >= 0`. When invalid, return `(0, 0)` and
`direction_valid=False`. The fixture records the executable's raw direction even where the public
predicate invalidates it; it is oracle evidence, not the public expected direction.

## Explicit source drift and invalid inputs

The executable is an oracle only away from equal-level path-opening plateaus. Its unstable
equal-value ordering diverges from the paper result in exactly two named fixture arrays:
`gap_robustness_2/po_raw_o1` and `border_horizontal/po_raw_o1`. Production must match the paper for
those arrays. No source-output compatibility is claimed on other plateau-order-sensitive inputs.

Invalid core inputs raise `ValueError`: non-2-D or empty arrays, any dtype other than `uint8`,
boolean or nonintegral lengths, `L < 1`, an empty `path_lengths` tuple, and boolean or negative
robustness. Dark-ridge input support is deferred rather than approximated. A caller requiring dark
ridges must perform an explicitly documented inversion outside this release.
