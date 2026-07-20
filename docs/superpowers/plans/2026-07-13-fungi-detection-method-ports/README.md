# Fungi detection method ports: implementation plan

**Date:** 2026-07-13
**Input design:** `docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/design.md`
**Status:** partial release candidate: A01, A02, and A06-A11 implemented;
A03-A05 explicitly deferred at their reference-contract gates
**Scope:** 11 algorithm clusters, one implementation agent and one independent reviewer per cluster;
eight approved implementation paths in this release candidate

## Outcome of the design review

The proposed `sdk_.reconnect` purity boundary is sound and matches the existing
`sdk_.branch_pathfinding` rule that callers own domain-specific cost construction
(`src/phenotypic/sdk_/branch_pathfinding/CLAUDE.md:45-51`). The design is not yet
implementation-ready. Several algorithm contracts either contradict their cited source or
omit information needed to produce deterministic numerical behavior.

Implementation may begin only after the applicable correction row below is resolved in the
design and recorded in the drift register.

| ID | Severity | Required correction |
|---|---:|---|
| C1 | blocking | Split faithful APP2 GWDT from PhenoTypic's inverse traversal-cost adaptation. APP2 defines a background-seeded gray-weighted distance map and later uses a separate cost transform. The current `seeds, weight_exp, eps` contract conflates those stages. The APP2 paper describes GWDT as an intensity sum along a shortest path to background, not the inverse-intensity transform in the design ([Xiao and Peng 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3661058/)). |
| C2 | blocking | Replace the claim that `weight_exp=0` generally equals Euclidean EDT. A 4- or 8-neighbor graph yields Manhattan or octile path distance, not exact Euclidean distance. [Based on general reasoning, no specific citation available] |
| C3 | blocking | Define where `FocusEdgeTensorVoting` obtains `theta`. A `FocusEdge` wrapper receives an `Image`, while the detector's phase orientation exists only inside `FilamentousFungiDetector._build_cost_surface` (`src/phenotypic/detect/_filamentous_fungi_detector.py:641-647`). Do not serialize an image-sized orientation array as an operation parameter. |
| C4 | blocking | Pin tensor-voting token semantics, axes, tangent-versus-normal convention, self-votes, finite support, boundary rule, accumulator dtype, and normalization from an executable reference. Risser et al. is an older, domain-specific port source and must be treated as such ([Risser et al. 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC3298375/)). |
| C5 | blocking | Decide whether Jerman is a faithful MATLAB Hessian-chain port or the Jerman response law applied to scikit-image Hessians. Those are different numerical contracts. Replace “Frangi goes to zero at a bifurcation” with a parameterized analytic comparison. The author implementation is explicitly tied to the 2016 vesselness work ([author implementation](https://jp.mathworks.com/matlabcentral/fileexchange/63171-jerman-enhancement-filter), [paper](https://openreview.net/pdf?id=SJgwy9CNxV)). |
| C6 | blocking | Define the Kalman measurement sequence, state vector, `F/H/Q/R/P0`, sampling step, association tie-break, maximum coast length, termination, and returned colony-labelled candidate-path contract. `endpoints, theta, gate_chi2` alone cannot determine a Kalman track. [Based on established linear-filter algebra and current detector interface; no single source contract has yet been selected.] |
| C7 | blocking | Define CA segment construction, directed-neighbor graph, acyclicity, collinearity score, synchronous update, convergence bound, fork ordering, and returned colony-labelled candidate-path contract. The standard CA state is the longest compatible predecessor chain only after the directed graph and friendship relation are fixed ([Track Finding, 2021](https://link.springer.com/chapter/10.1007/978-3-030-65771-0_5)). |
| C8 | blocking | Add local orientation evidence and a concrete `N_tests` formula to the NFA helper. LSD's NFA is defined for `k` aligned pixels among `n` with an explicit precision `p` and image-size-dependent test count, not a segment label map alone ([LSD derivation](https://dev.ipol.im/~morel/Dossier_MVA_2010_Cours_Transparents_Documents/Cours_1_texte_Line_Segment_Detector.pdf)). |
| C9 | blocking | Replace the arbitrary `n_orientations`/rank contract with the published 2-D rule: compute the four canonical path-opening responses and subtract their pointwise minimum from their pointwise maximum ([Merveille et al. 2018](https://www.ipol.im/pub/art/2017/207/)). |
| C10 | blocking | Make RHT return all data the wrapper promises, at minimum response, axial orientation, and validity. The current helper promises only orientation while the wrapper exposes response and orientation. Pin thresholding, angular bins, circular mask discretization, and undefined-orientation behavior from the primary RHT source ([Clark et al. 2014](https://arxiv.org/abs/1312.1338)). |
| C11 | blocking | Define persistence filtration polarity and a sourced reconstruction/cancellation algorithm. GUDHI's cubical API computes intervals and persistence-pair cells, but it does not itself reconstruct a denoised scalar image ([GUDHI cubical API](https://gudhi.inria.fr/python/3.8.0/cubical_complex_ref.html)). “Cancellation/Morse step is small” is therefore unsupported. |
| C12 | high | Make FilFinder an explicit exception to “every algorithm has a pure helper.” It is an external wrapper. Define its threshold-to-mask rule, unit handling, output labels, and which FilFinder products become `objmap`. The maintained API exposes skeletonization and pruning as a multi-stage workflow ([FilFinder2D API](https://fil-finder.readthedocs.io/en/latest/api/fil_finder.FilFinder2D.html)). |
| C13 | high | Remove explicit `FootprintMixin` from `FocusEdgeBowlerHat`. `ImageEnhancer` already supplies footprint behavior (`src/phenotypic/abc_/CLAUDE.md:90-94`), and the pure helper, not the wrapper, owns line/disk structuring elements. |
| C14 | high | State whether GWDT replaces or complements the existing EDT gap penalty. The detector already applies that penalty (`src/phenotypic/detect/_filamentous_fungi_detector.py:598-614`), and applying a cumulative distance map again as local Dijkstra cost can integrate distance twice. |

## Plan artifact map

- `00-orchestration.md`: dependency DAG, agent ownership, shared seams, and gates.
- `01-reference-and-validation.md`: common reference, golden-fixture, numerical-oracle,
  tolerance, and mutation requirements.
- `algorithms/01-gwdt.md` through `algorithms/11-persistence.md`: one bounded algorithm
  cluster per requested implementation agent.
- `99-integration-and-release.md`: serialized shared-file integration and final regression.

## Non-negotiable completion definition

An algorithm is complete only when all of the following are present and independently reviewed:

1. Every external source is pinned locally with provenance, license, immutable revision, and SHA-256.
2. Every source claim in the reconciliation cites `refs/<method>/<file>:line`.
3. A source-generated golden fixture stores every public output.
4. Behavioral controls test the kind of algorithm independently of the fixture.
5. A standalone logic script under
   `docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/`
   re-derives the load-bearing numerical claim using only stdlib, NumPy, and SciPy.
6. The logic script never imports `phenotypic`, is deterministic, prints its assumptions and
   derived tolerance, and exits nonzero on failure.
7. Every plausible single-change mutant is killed or documented as mathematically equivalent.
8. Every deviation has one drift-register row, even when numerically small.
9. Helper, wrapper, serialization, taxonomy, tune-annotation, doctest, mypy, and ruff gates
   applicable to that cluster are green.
10. A reviewer who did not author the implementation reruns the oracle, golden failure proof,
    mutation matrix, and affected regression tests.

## Accuracy and evidence calibration

- **Established from the current repository:** the detector owns cost construction and its
  phase orientation internally; exported enhancers must satisfy the `detect_mat` contract; tune
  coverage scans public detect/enhance operations. These claims cite local code above.
- **Established from primary sources:** APP2's GWDT/background semantics, LSD's binomial-tail
  NFA structure, RORPO's ranked residual, and GUDHI's persistence-pair API. The linked sources
  are the basis for the blocking corrections.
- **Contested until the corpus is pinned:** exact tensor-voting, bowler-hat, Jerman Hessian,
  RHT, CA, and persistence-cancellation line-level behavior. The plan deliberately does not
  select among forks from memory.
- **Speculative domain benefit:** whether any method improves fungal reconstruction on real
  plates. Numerical fidelity does not establish biological utility. Each cluster therefore
  ends with an opt-in synthetic/filamentous benchmark, not a claim of detection improvement.
