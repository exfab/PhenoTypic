# Linton `calc_vote_stick` reconciliation

## Authority boundary

The production helper ports only the oriented stick-voting pass. Linton's full framework
normalizes an image, encodes binary ball tensors, performs a ball refinement, removes the
minor eigenvalue, then performs stick voting (`source/find_features.m:17-37`). Those stages are
not represented by a `(response, theta)` input and are therefore out of scope.

Risser's paper is also not the executable authority. It constructs segment-end, island, and
segment tokens from a skeletonized 3-D binary network, whereas this helper consumes a dense 2-D
response and tangent field. It remains contextual evidence only.

## Line-by-line mapping

| Production block | Source evidence | Decision |
|---|---|---|
| Validate finite 2-D arrays, matching shapes, nonnegative response, and finite `sigma > 0` | `source/calc_vote_stick.m:1-15` performs no validation | Contract-required guard before the compiled kernel |
| Encode `T0=response*n*n^T`, `n=(-sin(theta), cos(theta))` | `source/read_dot_edge_file.m:27-33` converts input tangent degrees by +90 degrees to a rank-1 normal tensor | Contract adaptation from arrays to Linton's required input tensor field |
| Compute exact odd window and half-width | `source/calc_vote_stick.m:17-18` | Exact transcription |
| Zero-pad and retain the input tensor | `source/calc_vote_stick.m:20-30` | Exact logic, streamed without materializing padded four-component storage |
| Decompose `T0` and activate only positive `l1-l2` | `source/calc_vote_stick.m:32-36`; `source/convert_tensor_ev.m:16-44` | Rank-1 encoding makes this equivalent to `response > 0`; source column-major order retained |
| Build a continuous field for every voter | `source/calc_vote_stick.m:58-70` | The `nargin < 6` condition is always true for this three-argument function, so the cached one-degree branch is unreachable and omitted |
| Convert the principal normal to a tangent before field creation | `source/calc_vote_stick.m:58-65` | Exact logic; axial sign is immaterial |
| Rotate coordinates and form the vote normal outer product | `source/create_stick_tensorfield.m:37-60` | Algebraically simplified to three symmetric components |
| Apply the author's fourfold-angle fork | `source/create_stick_tensorfield.m:63-83`, especially line 70; `source/README:9-12` | Exact archive behavior retained, despite its documented departure from Medioni's text |
| Scale each field by `l1-l2` | `source/calc_vote_stick.m:73-75` | Exact; equals response for rank-1 `T0` |
| Add fields to retained `T0` and crop zero-padded margins | `source/calc_vote_stick.m:77-87` | Exact; out-of-image votes are discarded, and the center field adds one self-vote on top of `T0` |
| Decompose accumulated symmetric tensor | `source/convert_tensor_ev.m:29-44` | Use the closed-form `hypot(a-d, 2b)` equivalent; return `stick=lambda1-lambda2`, `ball=lambda2` |
| Return only saliencies | `source/demo.m:16-21` uses `l1-l2` for curve saliency; tensor-voting decomposition defines the minor component separately | Public-surface adaptation; fixture retains all three tensor components and both eigenvalues |

## Frozen semantics

- Arrays use `(row, column)` with rows increasing downward. Tensor components use Cartesian
  `(x, y)` with x right and y up.
- `theta` is an axial Cartesian tangent angle in radians and is periodic modulo pi.
- Positive response pixels vote. Zero response pixels do not vote, regardless of `theta`.
- The accumulator and outputs are float64.
- Voters are traversed in MATLAB `find` order: columns first, then rows.
- The input tensor is retained. The voting field includes its center. An isolated token therefore
  contributes two coincident rank-1 tensors at its own pixel.
- Boundaries use zero extension followed by cropping, not wrapping or reflection.
- Stick and ball outputs are raw and unnormalized.
- The standalone operation wrapper and detector mapping are deferred.

## Drift register

| ID | Category | Deviation | Evidence and consequence |
|---|---|---|---|
| TV-D01 | contract-required | Accept response and tangent arrays instead of a prebuilt tensor field | Mapping is fixed by `source/read_dot_edge_file.m:27-33`; it restricts inputs to rank-1 oriented tokens |
| TV-D02 | forced | Stream three symmetric components instead of padding a `(H,W,2,2)` tensor | Algebraically identical to `source/calc_vote_stick.m:20-30,77-87`; reduces peak memory |
| TV-D03 | forced | Omit GUI waitbar calls | `source/calc_vote_stick.m:38-56,85`; no numerical effect |
| TV-D04 | forced | Omit unreachable cached-field branch | `source/calc_vote_stick.m:61-70`; `nargin` can never reach six in a three-argument function |
| TV-D05 | forced | Use closed-form symmetric 2x2 eigenvalues instead of materializing eigenvectors at output | Equivalent to `source/convert_tensor_ev.m:29-44`; Weyl-bounded tests cover saliencies |
| TV-D06 | capability-added | Validate shape, finiteness, response sign, and sigma | The source has no public guards; invalid values now fail before compilation |
| TV-D07 | contract-required | Return raw stick and ball saliency rather than the full tensor field | All tensor components remain available through a private test seam and in the fixture |
| TV-D08 | defect-retained | Retain the archive's fourfold-angle line | `source/README:9-12` says it differs from Medioni, but source fidelity requires preserving the selected executable behavior |
| TV-D09 | contract-required | Make `sigma` an explicit required argument instead of using source default `18.25` | `source/calc_vote_stick.m:13-15`; explicit scale prevents hidden policy in a pure helper |
